from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.news.blackout import (
    blackout_state,
    build_blackout_windows,
    format_briefing,
    upcoming_events,
)
from src.news.events import calendar_status, parse_events


F = [
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


def dt(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def test_p1_fixture_order_and_fields() -> None:
    events = parse_events(F)
    expected = [
        (dt(2026, 11, 23, 8, 30), "EUR", "German Flash Manufacturing PMI", "medium"),
        (dt(2026, 11, 23, 21), "USD", "FOMC Member Williams Speaks", "low"),
        (dt(2026, 11, 25, 0, 30), "AUD", "CPI m/m", "high"),
        (dt(2026, 11, 25, 10, 32), "EUR", "German 10-y Bond Auction", "low"),
        (dt(2026, 11, 25, 13, 30), "USD", "Core PCE Price Index m/m", "high"),
        (dt(2026, 11, 25, 13, 30), "USD", "Prelim GDP Price Index q/q", "medium"),
        (dt(2026, 11, 25, 13, 30), "USD", "Prelim GDP q/q", "high"),
        (dt(2026, 11, 25, 13, 30), "USD", "Unemployment Claims", "low"),
        (dt(2026, 11, 25, 19), "USD", "Beige Book", "low"),
        (dt(2026, 11, 26, 13), "USD", "Bank Holiday", "holiday"),
    ]
    assert len(events) == 10
    assert [(e["time_utc"], e["currency"], e["title"], e["impact"]) for e in events] == expected
    assert all(e["time_utc"].tzinfo is not None and e["time_utc"].utcoffset() == timedelta(0) for e in events)
    assert events[4]["forecast"] == "0.3%"
    assert events[4]["previous"] == "0.3%"
    assert events[6]["forecast"] == "3.5%"
    assert events[6]["previous"] == "3.4%"
    assert events[9]["forecast"] == ""
    assert events[9]["previous"] == ""
    assert events[3]["previous"] == "3.37|1.1"


def test_p2_whitespace_and_case() -> None:
    raw = {"title": "  Core PCE Price Index m/m ", "country": " usd", "date": "2026-11-25T08:30:00-05:00", "impact": "HIGH ", "forecast": " 0.3% ", "previous": "0.3%"}
    assert parse_events([raw]) == [{
        "title": "Core PCE Price Index m/m",
        "currency": "USD",
        "impact": "high",
        "time_utc": dt(2026, 11, 25, 13, 30),
        "forecast": "0.3%",
        "previous": "0.3%",
    }]


def test_p3_offsets_and_z() -> None:
    raw = [
        {"title": "A", "country": "EUR", "date": "2026-07-15T08:30:00-04:00", "impact": "low"},
        {"title": "B", "country": "EUR", "date": "2026-07-15T20:30:00+08:00", "impact": "low"},
        {"title": "C", "country": "EUR", "date": "2026-07-15T12:30:00Z", "impact": "low"},
        {"title": "D", "country": "EUR", "date": "2026-07-15T12:30:00+00:00", "impact": "low"},
    ]
    assert [event["time_utc"] for event in parse_events(raw)] == [dt(2026, 7, 15, 12, 30)] * 4


def test_p4_optional_text_defaults() -> None:
    base = {"title": "A", "country": "USD", "date": "2026-01-01T00:00:00+00:00", "impact": "Low"}
    event = parse_events([base])[0]
    assert event["forecast"] == event["previous"] == ""
    event = parse_events([{**base, "forecast": None, "previous": None}])[0]
    assert event["forecast"] == event["previous"] == ""


@pytest.mark.parametrize("raw, message", [
    ([{"title": "A", "country": "USD", "date": "2026-11-25T08:30:00", "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": "USD", "date": "not a date", "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": "USD", "date": 123, "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": "USD", "date": "2026-11-25T08:30:00+00:00", "impact": "Extreme"}], "record 0"),
    ([{"title": "A", "country": "USD", "date": "2026-11-25T08:30:00+00:00"}], "record 0"),
    ([{"country": "USD", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": "", "country": "USD", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": "   ", "country": "USD", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": 5, "country": "USD", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": "US", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": "USDX", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": "U5D", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": 5, "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}], "record 0"),
    ([{"title": "A", "country": "USD", "date": "2026-11-25T08:30:00+00:00", "impact": "Low", "forecast": 5}], "record 0"),
])
def test_p5_invalid_record(raw: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_events(raw)


@pytest.mark.parametrize("raw", [{"title": "A"}, "not a list"])
def test_p5_raw_shape(raw) -> None:
    with pytest.raises(ValueError):
        parse_events(raw)


def test_p5_second_record_index() -> None:
    valid = {"title": "A", "country": "USD", "date": "2026-11-25T08:30:00+00:00", "impact": "Low"}
    with pytest.raises(ValueError, match="record 1"):
        parse_events([valid, {"title": "A"}])
    with pytest.raises(ValueError):
        parse_events([valid, "bad"])


def test_p6_duplicate_rules() -> None:
    base = {"title": "A", "country": "USD", "date": "2026-01-01T00:00:00+00:00", "impact": "Low"}
    assert len(parse_events([base, dict(base)])) == 1
    assert len(parse_events([base, {**base, "country": "EUR"}])) == 2
    assert len(parse_events([base, {**base, "date": "2026-01-01T01:00:00+00:00"}])) == 2


def test_p7_empty() -> None:
    assert parse_events([]) == []


@pytest.mark.parametrize(("word", "lower"), [("Low", "low"), ("Medium", "medium"), ("High", "high"), ("Holiday", "holiday")])
def test_p8_impacts(word: str, lower: str) -> None:
    raw = {"title": "A", "country": "USD", "date": "2026-01-01T00:00:00+00:00", "impact": word}
    assert parse_events([raw])[0]["impact"] == lower


def test_s1_status_usable() -> None:
    events = parse_events(F)
    assert calendar_status(events, dt(2026, 11, 25, 2), dt(2026, 11, 25, 12)) == {"usable": True, "problems": []}


def test_s2_age_edges() -> None:
    events = parse_events(F)
    now = dt(2026, 11, 25, 12)
    assert calendar_status(events, dt(2026, 11, 24), now)["usable"]
    assert calendar_status(events, dt(2026, 11, 23, 23, 59), now)["problems"] == ["calendar_too_old"]


def test_s3_lower_coverage_edge() -> None:
    events = parse_events(F)
    assert calendar_status(events, dt(2026, 11, 22, 8), dt(2026, 11, 22, 8, 30))["usable"]
    assert calendar_status(events, dt(2026, 11, 22, 8), dt(2026, 11, 22, 8, 29))["problems"] == ["now_outside_calendar_week"]


def test_s4_upper_coverage_edge() -> None:
    events = parse_events(F)
    assert calendar_status(events, dt(2026, 11, 27), dt(2026, 11, 27, 13))["usable"]
    assert calendar_status(events, dt(2026, 11, 27), dt(2026, 11, 27, 13, 1))["problems"] == ["now_outside_calendar_week"]


def test_s5_empty() -> None:
    assert calendar_status([], dt(2026, 1, 1), dt(2026, 1, 1)) == {"usable": False, "problems": ["empty_calendar"]}


def test_s6_future_fetch() -> None:
    now = dt(2026, 11, 25, 12)
    assert calendar_status(parse_events(F), now + timedelta(seconds=1), now)["problems"] == ["fetched_in_future"]


def test_s7_combined_status() -> None:
    now = dt(2026, 11, 30)
    assert calendar_status(parse_events(F), dt(2026, 11, 20), now)["problems"] == ["calendar_too_old", "now_outside_calendar_week"]
    assert calendar_status([], dt(2026, 12, 1), now)["problems"] == ["empty_calendar", "fetched_in_future"]


@pytest.mark.parametrize(("name", "kwargs"), [
    ("now", {"now": datetime(2026, 1, 1), "fetched_at": dt(2026, 1, 1)}),
    ("fetched", {"now": dt(2026, 1, 1), "fetched_at": datetime(2026, 1, 1)}),
])
def test_s8_naive_status(name: str, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        calendar_status([], **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"max_age_hours": -1},
    {"max_age_hours": math.nan},
    {"max_age_hours": True},
    {"margin_hours": -1},
])
def test_s8_status_invalid_numbers(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        calendar_status([], dt(2026, 1, 1), dt(2026, 1, 1), **kwargs)


def test_s8_pandas_microseconds() -> None:
    now = pd.Timestamp("2026-11-25 12:00:00", tz="UTC").as_unit("us")
    fetched = pd.Timestamp("2026-11-25 02:00:00", tz="UTC").as_unit("us")
    assert calendar_status(parse_events(F), fetched, now) == {"usable": True, "problems": []}


@pytest.mark.parametrize(
    ("events", "fetched_at", "now", "expected"),
    [
        (
            parse_events(F),
            dt(2026, 11, 22, 8),
            dt(2026, 11, 22, 8, 29),
            {"usable": False, "problems": ["now_outside_calendar_week"]},
        ),
        (
            parse_events(F),
            dt(2026, 11, 27),
            dt(2026, 11, 27, 13, 1),
            {"usable": False, "problems": ["now_outside_calendar_week"]},
        ),
        (
            parse_events(F),
            dt(2026, 11, 23, 23, 59),
            dt(2026, 11, 25, 12),
            {"usable": False, "problems": ["calendar_too_old"]},
        ),
        (
            parse_events(F),
            dt(2026, 11, 25, 12, 0, 1),
            dt(2026, 11, 25, 12),
            {"usable": False, "problems": ["fetched_in_future"]},
        ),
        (
            [],
            dt(2026, 11, 25, 2),
            dt(2026, 11, 25, 12),
            {"usable": False, "problems": ["empty_calendar"]},
        ),
        (
            parse_events(F),
            dt(2026, 11, 22, 8),
            dt(2026, 11, 22, 8, 30),
            {"usable": True, "problems": []},
        ),
        (
            parse_events(F),
            dt(2026, 11, 27),
            dt(2026, 11, 27, 13),
            {"usable": True, "problems": []},
        ),
    ],
)
def test_s9_status_usable_and_single_problem(
    events: list[dict],
    fetched_at: datetime,
    now: datetime,
    expected: dict,
) -> None:
    assert calendar_status(events, fetched_at, now) == expected


def test_c1_default_window() -> None:
    windows = build_blackout_windows(parse_events(F))
    assert len(windows) == 1
    assert windows[0]["start"] == dt(2026, 11, 25, 13)
    assert windows[0]["end"] == dt(2026, 11, 25, 14)
    assert windows[0]["titles"] == [
        "USD Core PCE Price Index m/m",
        "USD Prelim GDP q/q",
    ]


def synthetic(title: str, time_utc: datetime) -> dict:
    return parse_events([{"title": title, "country": "USD", "date": time_utc.isoformat(), "impact": "High"}])[0]


@pytest.mark.parametrize(
    ("times", "minutes_before", "minutes_after", "expected"),
    [
        (["13:30", "14:10"], 30, 30, [((13, 0), (14, 40))]),
        (["13:30", "14:30"], 30, 30, [((13, 0), (15, 0))]),
        (["13:30", "14:31"], 30, 30, [((13, 0), (14, 0)), ((14, 1), (15, 1))]),
        (["13:30", "14:10", "14:50"], 30, 30, [((13, 0), (15, 20))]),
    ],
)
def test_w2_merging(times: list[str], minutes_before: int, minutes_after: int, expected: list[tuple]) -> None:
    events = []
    for index, clock in enumerate(times):
        hour, minute = (int(part) for part in clock.split(":"))
        events.append(synthetic(chr(65 + index), dt(2026, 1, 1, hour, minute)))
    windows = build_blackout_windows(events, minutes_before=minutes_before, minutes_after=minutes_after)
    actual = [
        ((window["start"].hour, window["start"].minute),
         (window["end"].hour, window["end"].minute))
        for window in windows
    ]
    assert actual == expected


def test_w3_parameters() -> None:
    events = parse_events(F)
    small = build_blackout_windows(events, minutes_before=10, minutes_after=5)
    assert small[0]["start"] == dt(2026, 11, 25, 13, 20)
    assert small[0]["end"] == dt(2026, 11, 25, 13, 35)
    medium = build_blackout_windows(events, impacts=("high", "medium"))
    usd_medium = next(window for window in medium if window["start"] == dt(2026, 11, 25, 13))
    assert (usd_medium["start"], usd_medium["end"]) == (dt(2026, 11, 25, 13), dt(2026, 11, 25, 14))
    assert usd_medium["titles"] == [
        "USD Core PCE Price Index m/m",
        "USD Prelim GDP Price Index q/q",
        "USD Prelim GDP q/q",
    ]
    eur = build_blackout_windows(events, currencies=("EUR",), impacts=("medium",))
    assert eur[0]["start"] == dt(2026, 11, 23, 8)
    assert eur[0]["titles"] == ["EUR German Flash Manufacturing PMI"]
    aud = build_blackout_windows(events, currencies=("AUD",))
    assert aud[0]["start"] == dt(2026, 11, 25, 0)
    assert aud[0]["titles"] == ["AUD CPI m/m"]


def test_window_titles_follow_event_time_not_title_order() -> None:
    events = [
        synthetic("A at later time", dt(2026, 1, 1, 14, 10)),
        synthetic("Z at earlier time", dt(2026, 1, 1, 13, 30)),
    ]
    window = build_blackout_windows(events)[0]
    assert window["titles"] == [
        "USD Z at earlier time",
        "USD A at later time",
    ]


@pytest.mark.parametrize("kwargs", [
    {"minutes_before": -1},
    {"minutes_before": 5.0},
    {"minutes_before": True},
    {"currencies": ()},
    {"currencies": ("EU",)},
    {"impacts": ("extreme",)},
    {"impacts": ()},
])
def test_w4_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_blackout_windows(parse_events(F), **kwargs)


def test_w4_no_matches() -> None:
    assert build_blackout_windows(parse_events(F), currencies=("GBP",)) == []


def test_d_blackout_state() -> None:
    window = build_blackout_windows(parse_events(F))[0]
    assert blackout_state([window], dt(2026, 11, 25, 12, 59, 59)) == {"in_blackout": False, "window": None}
    assert blackout_state([window], dt(2026, 11, 25, 13))["in_blackout"]
    assert blackout_state([window], dt(2026, 11, 25, 14))["in_blackout"]
    assert blackout_state([window], dt(2026, 11, 25, 14, 0, 1)) == {"in_blackout": False, "window": None}
    micro = pd.Timestamp("2026-11-25 13:30:00", tz="UTC").as_unit("us")
    assert blackout_state([window], micro)["in_blackout"]
    with pytest.raises(ValueError):
        blackout_state([window], datetime(2026, 11, 25, 13))


def test_u1_next_24() -> None:
    events = parse_events(F)
    upcoming = upcoming_events(events, dt(2026, 11, 25, 12), hours_ahead=24)
    assert [event["title"] for event in upcoming] == [
        "Core PCE Price Index m/m",
        "Prelim GDP Price Index q/q",
        "Prelim GDP q/q",
    ]


def test_u2_cutoffs() -> None:
    events = parse_events(F)
    assert [e["title"] for e in upcoming_events(events, dt(2026, 11, 25, 12), hours_ahead=25)][-1] == "Bank Holiday"
    assert [e["title"] for e in upcoming_events(events, dt(2026, 11, 25, 11, 59, 59), hours_ahead=25)] == [
        "Core PCE Price Index m/m", "Prelim GDP Price Index q/q", "Prelim GDP q/q"
    ]


def test_u3_now_inclusive_and_second_past() -> None:
    events = parse_events(F)
    assert len(upcoming_events(events, dt(2026, 11, 25, 13, 30), hours_ahead=1)) == 3
    assert upcoming_events(events, dt(2026, 11, 25, 13, 30, 1), hours_ahead=1) == []


def test_u4_filters() -> None:
    events = parse_events(F)
    assert [e["title"] for e in upcoming_events(events, dt(2026, 11, 23), hours_ahead=24, currencies=("EUR",))] == ["German Flash Manufacturing PMI"]
    assert [e["title"] for e in upcoming_events(events, dt(2026, 11, 25), hours_ahead=24, currencies=("EUR", "USD"), impacts=("low",))] == [
        "German 10-y Bond Auction", "Unemployment Claims", "Beige Book"
    ]


@pytest.mark.parametrize("hours", [0, -1, math.nan, True])
def test_u5_invalid_hours(hours: float) -> None:
    with pytest.raises(ValueError):
        upcoming_events(parse_events(F), dt(2026, 11, 25, 12), hours_ahead=hours)


def test_u5_naive() -> None:
    with pytest.raises(ValueError):
        upcoming_events(parse_events(F), datetime(2026, 11, 25, 12))


def test_f1_exact_briefing() -> None:
    events = parse_events(F)
    status = calendar_status(events, dt(2026, 11, 25, 2), dt(2026, 11, 25, 12))
    assert format_briefing(events, status, dt(2026, 11, 25, 12), hours_ahead=24) == "\n".join([
        "News briefing: EUR and USD, next 24 hours, as of 2026-11-25 20:00 Manila time",
        "Avoid new trades (windows of 30 min before and 30 min after high-impact events):",
        "2026-11-25 21:00 to 2026-11-25 22:00 Manila: USD Core PCE Price Index m/m, USD Prelim GDP q/q",
        "Events:",
        "2026-11-25 21:30 | USD | HIGH | Core PCE Price Index m/m | forecast 0.3% | previous 0.3%",
        "2026-11-25 21:30 | USD | MEDIUM | Prelim GDP Price Index q/q | forecast 2.3% | previous 2.3%",
        "2026-11-25 21:30 | USD | HIGH | Prelim GDP q/q | forecast 3.5% | previous 3.4%",
        "Source: Forex Factory weekly feed (unofficial). Times are Manila time. Planning aid only, not financial advice.",
    ])


def test_f2_inside_window() -> None:
    events = parse_events(F)
    status = calendar_status(events, dt(2026, 11, 25, 2), dt(2026, 11, 25, 13, 15))
    text = format_briefing(events, status, dt(2026, 11, 25, 13, 15), hours_ahead=24)
    assert "BLACKOUT NOW: until 2026-11-25 22:00 Manila time (USD Core PCE Price Index m/m, USD Prelim GDP q/q)" in text
    assert "Avoid new trades" in text
    assert "2026-11-25 21:30 | USD" in text


def test_f3_no_windows_or_events() -> None:
    events = parse_events(F)
    status = calendar_status(events, dt(2026, 11, 23, 9), dt(2026, 11, 23, 10))
    text = format_briefing(events, status, dt(2026, 11, 23, 10), hours_ahead=1)
    assert text.splitlines() == [
        "News briefing: EUR and USD, next 1 hours, as of 2026-11-23 18:00 Manila time",
        "No high-impact windows in the next 1 hours.",
        "No events (high, medium, holiday) in the next 1 hours.",
        "Source: Forex Factory weekly feed (unofficial). Times are Manila time. Planning aid only, not financial advice.",
    ]


def test_f4_unusable() -> None:
    events = parse_events(F)
    status = calendar_status(events, dt(2026, 11, 20), dt(2026, 11, 30))
    text = format_briefing(events, status, dt(2026, 11, 30))
    assert text == "\n".join([
        "CALENDAR UNAVAILABLE: check news manually at forexfactory.com before trading. Do not assume there is no news.",
        "Problems: calendar_too_old, now_outside_calendar_week",
    ])
    assert "Beige Book" not in text


@pytest.mark.parametrize(
    ("fetched_at", "now", "problem"),
    [
        (
            dt(2026, 11, 22, 8),
            dt(2026, 11, 22, 8, 29),
            "now_outside_calendar_week",
        ),
        (
            dt(2026, 11, 23, 23, 59),
            dt(2026, 11, 25, 12),
            "calendar_too_old",
        ),
    ],
)
def test_f7_status_drives_unavailable_briefing(
    fetched_at: datetime,
    now: datetime,
    problem: str,
) -> None:
    status = calendar_status(parse_events(F), fetched_at, now)
    text = format_briefing(parse_events(F), status, now)

    assert text == "\n".join([
        "CALENDAR UNAVAILABLE: check news manually at forexfactory.com before trading. Do not assume there is no news.",
        f"Problems: {problem}",
    ])
    assert all(record["title"] not in text for record in F)


def test_f5_empty_status() -> None:
    text = format_briefing([], {"usable": False, "problems": ["empty_calendar"]}, dt(2026, 1, 1))
    assert text.endswith("Problems: empty_calendar")


def test_f6_empty_forecast_previous() -> None:
    events = parse_events(F)
    status = {"usable": True, "problems": []}
    text = format_briefing(events, status, dt(2026, 11, 25, 18), hours_ahead=2, impacts=("low",))
    assert "2026-11-26 03:00 | USD | LOW | Beige Book | forecast - | previous -" in text


def reference_utc(text: str) -> datetime:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:\d{2})$", text)
    assert match
    year, month, day, hour, minute, second = (int(match.group(i)) for i in range(1, 7))
    suffix = match.group(7)
    offset_minutes = 0 if suffix == "Z" else (1 if suffix[0] == "+" else -1) * (int(suffix[1:3]) * 60 + int(suffix[4:6]))
    local = datetime(year, month, day, hour, minute, second)
    return (local - timedelta(minutes=offset_minutes)).replace(tzinfo=timezone.utc)


def test_g1_seeded_parse_reference() -> None:
    rng = random.Random(31)
    offsets = ["-05:00", "-04:00", "+00:00", "+01:00", "+08:00", "Z"]
    impacts = ["Low", "Medium", "HIGH", "hOlIdAy"]
    raw = []
    for index in range(200):
        day = rng.randint(1, 28)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        offset = rng.choice(offsets)
        suffix = offset
        if offset == "Z":
            suffix = "Z"
        text = f"2026-01-{day:02d}T{hour:02d}:{minute:02d}:00{suffix}"
        raw.append({"title": f"E{index}", "country": "USD", "date": text, "impact": rng.choice(impacts)})
    events = parse_events(raw)
    references = [reference_utc(record["date"]) for record in raw]
    duplicates = set()
    unique_references = []
    for record, time_utc in zip(raw, references):
        key = (record["title"].strip(), record["country"].strip().upper(), time_utc)
        if key not in duplicates:
            duplicates.add(key)
            unique_references.append(time_utc)
    assert sorted(event["time_utc"] for event in events) == sorted(unique_references)


def reference_windows(events: list[dict], before: int, after: int) -> list[dict]:
    intervals = []
    for event in events:
        start = event["time_utc"] - timedelta(minutes=before)
        end = event["time_utc"] + timedelta(minutes=after)
        intervals.append({"start": start, "end": end, "title": f"{event['currency']} {event['title']}"})
    while True:
        merged = False
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                a, b = intervals[i], intervals[j]
                if a["start"] <= b["end"] and b["start"] <= a["end"]:
                    start = min(a["start"], b["start"])
                    end = max(a["end"], b["end"])
                    combined_titles = sorted(
                        [x for x in (a, b)],
                        key=lambda x: (x["start"], x["title"]),
                    )
                    intervals[i] = {"start": start, "end": end, "titles": sorted({a["title"], b.get("title", ""), *(b.get("titles", []))})}
                    intervals[j] = None
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
        intervals = [item for item in intervals if item is not None]
    return sorted(intervals, key=lambda item: item["start"])


def test_g2_seeded_windows_reference_and_g3_blackout_membership() -> None:
    rng = random.Random(77)
    for case in range(300):
        events = []
        for index in range(rng.randint(1, 12)):
            minute = rng.randint(0, 600) // 5 * 5
            event_time = dt(2026, 1, 1) + timedelta(minutes=minute)
            currency = rng.choice(["EUR", "USD", "GBP"])
            impact = rng.choice(["low", "medium", "high"])
            event = parse_events([{"title": f"T{index}", "country": currency, "date": event_time.isoformat(), "impact": impact}])[0]
            events.append(event)
        before = rng.randint(0, 45)
        after = rng.randint(0, 45)
        windows = build_blackout_windows(events, currencies=("EUR", "USD", "GBP"), impacts=("low", "medium", "high"), minutes_before=before, minutes_after=after)
        for window in windows:
            assert window["start"] <= window["end"]
        for index, window in enumerate(windows):
            if index:
                assert windows[index - 1]["end"] < window["start"]
        if case < 50:
            for _ in range(40):
                random_minute = rng.randint(-60, 660)
                now = dt(2026, 1, 1) + timedelta(minutes=random_minute)
                state = blackout_state(windows, now)
                membership = any(window["start"] <= now <= window["end"] for window in windows)
                assert state["in_blackout"] == membership


def test_h_no_future_huge_forecast_in_output() -> None:
    events = parse_events(F)
    status = {"usable": True, "problems": []}
    text = format_briefing(events, status, dt(2026, 11, 23, 10), hours_ahead=1)
    assert "Core PCE" not in text
    assert "German 10-y" not in text


def test_i_source_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/news/events.py", "src/news/blackout.py"):
        text = (root / relative).read_text(encoding="utf-8")
        for forbidden in ("urllib", "requests", "socket", "open(", "load_candles"):
            assert forbidden not in text
