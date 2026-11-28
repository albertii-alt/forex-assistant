from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.app import journal_view
from src.journal.store import list_trades, open_journal
import src.news.service as news_service
from src.news.store import save_calendar


RAW_JAN = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-01-14T08:30:00-05:00", "impact": "High", "forecast": "0.3%", "previous": "0.2%"},
    {"title": "Retail Sales m/m", "country": "USD", "date": "2026-01-15T08:30:00-05:00", "impact": "High", "forecast": "0.4%", "previous": "0.6%"},
]


def _utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _load_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    now: datetime,
    *,
    news_name: str = "news",
    journal_name: str = "journal.sqlite3",
    raw: list[dict] | None = None,
    fetched_at: datetime | None = None,
) -> tuple[AppTest, Path, Path]:
    news_directory = tmp_path / news_name
    journal_path = tmp_path / journal_name
    monkeypatch.setenv(journal_view.JOURNAL_PATH_ENV, str(journal_path))
    monkeypatch.setenv("FOREX_NEWS_DIR", str(news_directory))
    monkeypatch.setattr(journal_view, "get_now", lambda: now)
    if raw is not None:
        assert fetched_at is not None
        save_calendar(
            raw,
            fetched_at,
            source="feed",
            directory=news_directory,
        )
    app = AppTest.from_file(
        "../src/app/journal_page.py",
        default_timeout=30,
    ).run()
    return app, news_directory, journal_path


def _news_warnings(app: AppTest) -> list[str]:
    return [
        item.value
        for item in app.warning
        if item.value.startswith(
            ("News calendar unavailable", "News blackout window active")
        )
    ]


def _set_reason(app: AppTest) -> None:
    app.text_input(key="reason").set_value("news test").run()


def _saved_trades(path: Path) -> list[dict]:
    conn = open_journal(path)
    try:
        return list_trades(conn)
    finally:
        conn.close()


def test_g1_missing_calendar_warns_without_disabling_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _load_app(tmp_path, monkeypatch, _utc(2026, 1, 14, 12))

    assert not app.exception
    warnings = _news_warnings(app)
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "News calendar unavailable (calendar_not_found)."
    )
    assert "Do not assume there is no news." in warnings[0]
    assert "python -m scripts.fetch_news_calendar" in warnings[0]
    assert any(
        item.value.startswith("CALENDAR UNAVAILABLE")
        for item in app.text
    )

    _set_reason(app)
    assert not app.button(key="save_trade").disabled
    assert len(_news_warnings(app)) == 1


def test_g2_usable_calendar_shows_expected_briefing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _load_app(
        tmp_path,
        monkeypatch,
        _utc(2026, 1, 14, 12),
        raw=RAW_JAN,
        fetched_at=_utc(2026, 1, 14, 2),
    )

    assert not app.exception
    assert _news_warnings(app) == []
    expected = "\n".join([
        "News briefing: EUR and USD, next 24 hours, as of 2026-01-14 20:00 Manila time",
        "Avoid new trades (windows of 30 min before and 30 min after high-impact events):",
        "2026-01-14 21:00 to 2026-01-14 22:00 Manila: USD CPI m/m",
        "Events:",
        "2026-01-14 21:30 | USD | HIGH | CPI m/m | forecast 0.3% | previous 0.2%",
        "Source: Forex Factory weekly feed (unofficial). Times are Manila time. Planning aid only, not financial advice.",
    ])
    assert expected in [item.value for item in app.text]
    assert (
        "Calendar fetched: 2026-01-14 10:00 Manila (02:00 UTC), source: feed"
        in [item.value for item in app.text]
    )


def test_g3_blackout_warning_does_not_change_saved_trade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _utc(2026, 1, 14, 13, 15)
    app, _, news_journal_path = _load_app(
        tmp_path,
        monkeypatch,
        now,
        news_name="news-calendar",
        journal_name="with-news.sqlite3",
        raw=RAW_JAN,
        fetched_at=_utc(2026, 1, 14, 2),
    )
    assert not app.exception
    assert _news_warnings(app) == [
        "News blackout window active until 2026-01-14 22:00 Manila time "
        "(USD CPI m/m). Practice rule: avoid opening new trades now."
    ]

    _set_reason(app)
    assert not app.button(key="save_trade").disabled
    assert len(_news_warnings(app)) == 1
    app.button(key="save_trade").click().run()
    news_trades = _saved_trades(news_journal_path)
    assert len(news_trades) == 1

    baseline_app, _, baseline_journal_path = _load_app(
        tmp_path,
        monkeypatch,
        now,
        news_name="no-calendar",
        journal_name="without-news.sqlite3",
    )
    _set_reason(baseline_app)
    assert not baseline_app.button(key="save_trade").disabled
    baseline_app.button(key="save_trade").click().run()
    baseline_trades = _saved_trades(baseline_journal_path)
    assert len(baseline_trades) == 1

    saved = news_trades[0]
    baseline = baseline_trades[0]
    assert saved["size_units"] == baseline["size_units"] == 8000
    assert saved["planned_risk_amount"] == pytest.approx(9.12, abs=1e-9)
    assert saved["planned_risk_amount"] == pytest.approx(
        baseline["planned_risk_amount"],
        abs=1e-9,
    )
    assert saved["reason"] == baseline["reason"] == "news test"


def test_g4_stale_calendar_warns_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _load_app(
        tmp_path,
        monkeypatch,
        _utc(2026, 1, 14, 13, 15),
        raw=RAW_JAN,
        fetched_at=_utc(2025, 12, 20),
    )

    warnings = _news_warnings(app)
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "News calendar unavailable (calendar_too_old)."
    )
    assert not any("News blackout" in warning for warning in warnings)
    assert not app.exception


def test_unavailable_context_takes_precedence_over_blackout_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _utc(2026, 1, 14, 13, 15)
    monkeypatch.setattr(
        news_service,
        "load_news_context",
        lambda current_now, directory: {
            "usable": False,
            "problems": ["calendar_too_old"],
            "briefing": (
                "CALENDAR UNAVAILABLE: check news manually at forexfactory.com "
                "before trading. Do not assume there is no news.\n"
                "Problems: calendar_too_old"
            ),
            "in_blackout": True,
            "blackout_window": {
                "end": _utc(2026, 1, 14, 14),
                "titles": ["USD CPI m/m"],
            },
            "fetched_at": _utc(2025, 12, 20),
            "source": "feed",
            "error": None,
        },
    )
    app, _, _ = _load_app(tmp_path, monkeypatch, now)

    warnings = _news_warnings(app)
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "News calendar unavailable (calendar_too_old)."
    )
    assert not app.exception


def test_g6_existing_widget_keys_and_metric_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _load_app(tmp_path, monkeypatch, _utc(2026, 1, 14, 12))
    assert not app.exception
    for key in ("starting_balance", "entry", "stop", "risk_pct"):
        assert len([item for item in app.number_input if item.key == key]) == 1
    assert len([item for item in app.radio if item.key == "side"]) == 1
    assert len([item for item in app.text_input if item.key == "reason"]) == 1
    assert len([item for item in app.text_input if item.key == "strategy"]) == 1
    assert len([item for item in app.selectbox if item.key == "mode"]) == 1
    assert len([item for item in app.button if item.key == "save_trade"]) == 1
    assert len([
        item for item in app.metric if item.label == "Balance now (USD)"
    ]) == 1
