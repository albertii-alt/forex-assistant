from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.news.service import (
    NEWS_DIR_ENV,
    load_news_context,
    manila_text,
    news_directory,
)
from src.news.store import CALENDAR_FILENAME, save_calendar


RAW_F = [
    {"title": "Unemployment Claims", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "Low", "forecast": "201K", "previous": "203K"},
    {"title": "Prelim GDP q/q", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "High", "forecast": "3.5%", "previous": "3.4%"},
    {"title": "Core PCE Price Index m/m", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "High", "forecast": "0.3%", "previous": "0.3%"},
    {"title": "Prelim GDP Price Index q/q", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "Medium", "forecast": "2.3%", "previous": "2.3%"},
    {"title": "Bank Holiday", "country": "USD", "date": "2026-11-26T08:00:00-05:00", "impact": "Holiday", "forecast": "", "previous": ""},
    {"title": "CPI m/m", "country": "AUD", "date": "2026-11-24T19:30:00-05:00", "impact": "High", "forecast": "0.4%", "previous": "0.8%"},
    {"title": "German Flash Manufacturing PMI", "country": "EUR", "date": "2026-11-23T03:30:00-05:00", "impact": "Medium", "forecast": "54.1", "previous": "54.2"},
    {"title": "FOMC Member Williams Speaks", "country": "USD", "date": "2026-11-23T16:00:00-05:00", "impact": "Low", "forecast": "", "previous": ""},
    {"title": "Beige Book", "country": "USD", "date": "2026-11-25T14:00:00-05:00", "impact": "Low", "forecast": "", "previous": ""},
    {"title": "German 10-y Bond Auction", "country": "EUR", "date": "2026-11-25T05:32:00-05:00", "impact": "Low", "forecast": "", "previous": "3.37|1.1"},
]
FETCHED_AT = datetime(2026, 11, 25, 2, tzinfo=timezone.utc)
EXPECTED_F_BRIEFING = "\n".join([
    "News briefing: EUR and USD, next 24 hours, as of 2026-11-25 20:00 Manila time",
    "Avoid new trades (windows of 30 min before and 30 min after high-impact events):",
    "2026-11-25 21:00 to 2026-11-25 22:00 Manila: USD Core PCE Price Index m/m, USD Prelim GDP q/q",
    "Events:",
    "2026-11-25 21:30 | USD | HIGH | Core PCE Price Index m/m | forecast 0.3% | previous 0.3%",
    "2026-11-25 21:30 | USD | MEDIUM | Prelim GDP Price Index q/q | forecast 2.3% | previous 2.3%",
    "2026-11-25 21:30 | USD | HIGH | Prelim GDP q/q | forecast 3.5% | previous 3.4%",
    "Source: Forex Factory weekly feed (unofficial). Times are Manila time. Planning aid only, not financial advice.",
])


def utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def save_raw(
    directory: Path,
    raw: list[dict] = RAW_F,
    fetched_at: datetime = FETCHED_AT,
) -> Path:
    return save_calendar(raw, fetched_at, source="feed", directory=directory)


def test_v1_missing_and_empty_directories(tmp_path: Path) -> None:
    for directory in (tmp_path / "missing", tmp_path / "empty"):
        directory.mkdir(exist_ok=True)
        context = load_news_context(utc(2026, 11, 25, 12), directory)
        assert context == {
            "usable": False,
            "problems": ["calendar_not_found"],
            "briefing": "\n".join([
                "CALENDAR UNAVAILABLE: check news manually at forexfactory.com before trading. Do not assume there is no news.",
                "Problems: calendar_not_found",
            ]),
            "in_blackout": False,
            "blackout_window": None,
            "fetched_at": None,
            "source": None,
            "error": "calendar file not found",
        }


def test_v2_invalid_file_is_fail_safe(tmp_path: Path) -> None:
    (tmp_path / CALENDAR_FILENAME).write_text("not json", encoding="utf-8")
    context = load_news_context(utc(2026, 11, 25, 12), tmp_path)
    assert context["usable"] is False
    assert context["problems"] == ["calendar_file_invalid"]
    assert context["error"].startswith("calendar file invalid: ")
    assert context["briefing"].splitlines()[1] == (
        "Problems: calendar_file_invalid"
    )


def test_v3_usable_calendar_and_exact_briefing(tmp_path: Path) -> None:
    save_raw(tmp_path)
    context = load_news_context(utc(2026, 11, 25, 12), tmp_path)
    assert context["usable"] is True
    assert context["problems"] == []
    assert context["in_blackout"] is False
    assert context["blackout_window"] is None
    assert context["source"] == "feed"
    assert context["fetched_at"] == FETCHED_AT
    assert context["briefing"] == EXPECTED_F_BRIEFING


def test_v4_active_blackout_context(tmp_path: Path) -> None:
    save_raw(tmp_path)
    context = load_news_context(utc(2026, 11, 25, 13, 15), tmp_path)
    assert context["usable"] is True
    assert context["in_blackout"] is True
    window = context["blackout_window"]
    assert window["start"] == utc(2026, 11, 25, 13)
    assert window["end"] == utc(2026, 11, 25, 14)
    assert window["titles"] == [
        "USD Core PCE Price Index m/m",
        "USD Prelim GDP q/q",
    ]
    assert context["briefing"].splitlines()[1] == (
        "BLACKOUT NOW: until 2026-11-25 22:00 Manila time "
        "(USD Core PCE Price Index m/m, USD Prelim GDP q/q)"
    )


def test_v5_stale_and_outside_calendar_context(tmp_path: Path) -> None:
    save_raw(tmp_path)
    context = load_news_context(utc(2026, 11, 30), tmp_path)
    assert context["usable"] is False
    assert context["problems"] == [
        "calendar_too_old",
        "now_outside_calendar_week",
    ]
    assert context["in_blackout"] is False
    assert context["briefing"].splitlines()[1] == (
        "Problems: calendar_too_old, now_outside_calendar_week"
    )
    assert all(record["title"] not in context["briefing"] for record in RAW_F)


def test_v6_stale_calendar_is_not_trusted_for_blackout(tmp_path: Path) -> None:
    save_raw(tmp_path, fetched_at=utc(2026, 11, 20))
    context = load_news_context(utc(2026, 11, 25, 13, 15), tmp_path)
    assert context["usable"] is False
    assert context["problems"] == ["calendar_too_old"]
    assert context["in_blackout"] is False
    assert context["blackout_window"] is None


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (utc(2026, 11, 25, 13), True),
        (utc(2026, 11, 25, 12, 59, 59), False),
        (utc(2026, 11, 25, 14), True),
        (utc(2026, 11, 25, 14, 0, 1), False),
    ],
)
def test_v7_blackout_boundaries(
    tmp_path: Path,
    now: datetime,
    expected: bool,
) -> None:
    save_raw(tmp_path)
    assert load_news_context(now, tmp_path)["in_blackout"] is expected


def test_v8_now_validation_and_microsecond_timestamp(tmp_path: Path) -> None:
    save_raw(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        load_news_context(datetime(2026, 11, 25, 12), tmp_path)

    now = pd.Timestamp("2026-11-25 12:00:00", tz="UTC").as_unit("us")
    context = load_news_context(now, tmp_path)
    assert context["usable"] is True
    assert context["problems"] == []
    assert context["briefing"] == EXPECTED_F_BRIEFING


def test_v9_hours_ahead_includes_window_touching_cutoff(tmp_path: Path) -> None:
    save_raw(tmp_path)
    context = load_news_context(
        utc(2026, 11, 25, 12),
        tmp_path,
        hours_ahead=1,
    )
    assert "next 1 hours" in context["briefing"]
    assert "No high-impact windows in the next 1 hours." not in context["briefing"]
    assert (
        "Avoid new trades (windows of 30 min before and 30 min after high-impact events):"
        in context["briefing"]
    )


def test_v10_news_directory_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEWS_DIR_ENV, "custom/news")
    assert news_directory() == "custom/news"
    monkeypatch.delenv(NEWS_DIR_ENV)
    assert news_directory() == "data/news"
    monkeypatch.setenv(NEWS_DIR_ENV, "")
    assert news_directory() == "data/news"


def test_v11_manila_text_offsets_and_naive_validation() -> None:
    assert manila_text(utc(2026, 11, 25, 13, 30)) == "2026-11-25 21:30"
    new_york = datetime(
        2026, 11, 25, 8, 30, tzinfo=ZoneInfo("America/New_York")
    )
    assert manila_text(new_york) == "2026-11-25 21:30"
    with pytest.raises(ValueError, match="timezone-aware"):
        manila_text(datetime(2026, 11, 25, 13, 30))


def _reference_status_and_blackout(now: datetime) -> tuple[bool, bool]:
    fetched_at = utc(2026, 11, 25, 2)
    first_event = utc(2026, 11, 23, 8, 30)
    last_event = utc(2026, 11, 26, 13)
    usable = (
        fetched_at <= now
        and now - fetched_at <= timedelta(hours=36)
        and first_event - timedelta(hours=24)
        <= now
        <= last_event + timedelta(hours=24)
    )
    in_blackout = (
        usable
        and utc(2026, 11, 25, 13)
        <= now
        <= utc(2026, 11, 25, 14)
    )
    return usable, in_blackout


def test_v12_seeded_independent_reference(tmp_path: Path) -> None:
    save_raw(tmp_path)
    start = utc(2026, 11, 21)
    end = utc(2026, 11, 29)
    rng = random.Random(55)
    now_values = [
        start + timedelta(seconds=rng.randrange(int((end - start).total_seconds())))
        for _ in range(250)
    ]
    now_values.extend([
        utc(2026, 11, 22, 8, 30),
        utc(2026, 11, 22, 8, 29, 59),
        utc(2026, 11, 25, 13),
        utc(2026, 11, 25, 12, 59, 59),
        utc(2026, 11, 25, 14),
        utc(2026, 11, 25, 14, 0, 1),
        utc(2026, 11, 27, 13),
        utc(2026, 11, 27, 13, 0, 1),
    ])

    for now in now_values:
        expected_usable, expected_blackout = _reference_status_and_blackout(now)
        context = load_news_context(now, tmp_path)
        assert context["usable"] is expected_usable, now
        assert context["in_blackout"] is expected_blackout, now


def test_v13_future_huge_forecast_does_not_change_briefing(
    tmp_path: Path,
) -> None:
    now = utc(2026, 11, 25, 12)
    save_raw(tmp_path)
    original = load_news_context(now, tmp_path)["briefing"]

    later_events = RAW_F + [{
        "title": "Very Far Future Event",
        "country": "USD",
        "date": "2026-11-27T08:30:00-05:00",
        "impact": "High",
        "forecast": "X" * 100_000,
        "previous": "",
    }]
    save_raw(tmp_path, later_events)
    updated = load_news_context(now, tmp_path)["briefing"]
    assert updated == original


def test_v14_source_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    service_source = (root / "src" / "news" / "service.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("urllib", "socket", "requests", "fetch_news_calendar"):
        assert forbidden not in service_source

    page_source = (root / "src" / "app" / "journal_page.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("urllib", "socket", "requests", "import scripts"):
        assert forbidden not in page_source
