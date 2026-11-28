from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.app import journal_view
from src.journal.store import add_trade, close_trade, open_journal


NOW = datetime(2026, 1, 14, 12, tzinfo=timezone.utc)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def load_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv(journal_view.JOURNAL_PATH_ENV, str(tmp_path / "j.sqlite3"))
    monkeypatch.setenv("FOREX_NEWS_DIR", str(tmp_path / "no_news_for_tests"))
    monkeypatch.setattr(journal_view, "get_now", lambda: NOW)
    return AppTest.from_file(
        "../src/app/journal_page.py",
        default_timeout=30,
    ).run()


def add_closed_trade(
    tmp_path: Path,
    *,
    opened: datetime,
    closed: datetime,
    pnl: float,
) -> None:
    conn = open_journal(tmp_path / "j.sqlite3")
    try:
        trade_id = add_trade(
            conn,
            opened_at=opened,
            side="long",
            entry_price=1.1,
            stop_price=1.099,
            size_units=1000,
            planned_risk_amount=10,
            mode="paper",
            reason="r",
        )
        close_trade(
            conn,
            trade_id,
            closed_at=closed,
            exit_price=1.1,
            realized_pnl=pnl,
        )
    finally:
        conn.close()


def test_h1_empty_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app(tmp_path, monkeypatch)
    assert not app.exception
    expected = "\n".join([
        "Journal review (closed trades only)",
        "Closed trades: 0 (open: 0)",
        "Wins / losses / breakeven: 0 / 0 / 0",
        "Win rate: -",
        "Average R per trade: -",
        "Total R: 0.00",
        "Total result (USD): 0.00",
        "Average win R: - | average loss R: -",
        "Profit factor: -",
        "Max drawdown (R): 0.00",
        "Longest losing streak: 0",
        "No closed trades yet.",
        "Planning aid only, not financial advice.",
    ])
    assert expected in [item.value for item in app.text]
    assert any(item.value == "Review" for item in app.header)


def test_h2_two_closed_trades_exact_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_closed_trade(
        tmp_path,
        opened=utc(2026, 1, 14, 8),
        closed=utc(2026, 1, 14, 9),
        pnl=20,
    )
    add_closed_trade(
        tmp_path,
        opened=utc(2026, 1, 14, 10),
        closed=utc(2026, 1, 14, 11),
        pnl=-10,
    )
    app = load_app(tmp_path, monkeypatch)

    expected = "\n".join([
        "Journal review (closed trades only)",
        "Closed trades: 2 (open: 0)",
        "Wins / losses / breakeven: 1 / 1 / 0",
        "Win rate: 50.0%",
        "Average R per trade: 0.50 (95% range shown from 30 closed trades)",
        "Total R: 1.00",
        "Total result (USD): 10.00",
        "Average win R: 2.00 | average loss R: -1.00",
        "Profit factor: 2.00",
        "Max drawdown (R): 1.00",
        "Longest losing streak: 1",
        "Too few closed trades (2) to conclude anything. A few wins or losses in a row are normal luck.",
        "Planning aid only, not financial advice.",
    ])
    assert expected in [item.value for item in app.text]
    assert not app.exception


def test_h3_review_includes_other_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_closed_trade(
        tmp_path,
        opened=utc(2026, 1, 10, 8),
        closed=utc(2026, 1, 10, 9),
        pnl=10,
    )
    app = load_app(tmp_path, monkeypatch)
    review_texts = [item.value for item in app.text if item.value.startswith("Journal review")]
    assert len(review_texts) == 1
    assert "Closed trades: 1 (open: 0)" in review_texts[0]


def test_h4_existing_elements_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = load_app(tmp_path, monkeypatch)
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


def test_h5_page_source_checks() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "app" / "journal_page.py"
    ).read_text(encoding="utf-8")
    assert "review_trades" in source
    assert "format_review" in source
    for forbidden in ("urllib", "socket", "requests"):
        assert forbidden not in source


@pytest.mark.parametrize(
    ("count", "expected_line"),
    [
        (30, "Average R per trade: 0.00 (rough 95% range -0.37 to 0.37)"),
        (
            29,
            "Average R per trade: 0.03 "
            "(95% range shown from 30 closed trades)",
        ),
    ],
)
def test_p30_page_range_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    expected_line: str,
) -> None:
    start = utc(2026, 2, 6, 10)
    for k in range(count):
        closed_at = start + timedelta(hours=k)
        add_closed_trade(
            tmp_path,
            opened=closed_at - timedelta(minutes=30),
            closed=closed_at,
            pnl=10 if k % 2 == 0 else -10,
        )
    app = load_app(tmp_path, monkeypatch)

    review_texts = [
        item.value for item in app.text
        if item.value.startswith("Journal review")
    ]
    assert len(review_texts) == 1
    assert review_texts[0].splitlines()[4] == expected_line
    assert not app.exception
