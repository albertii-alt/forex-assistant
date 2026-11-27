from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.strategies.candidate_d import (
    PLAN_COLUMNS,
    _ema_50_trend,
    candidate_d_funnel,
    count_candidate_d,
    find_candidate_d_plans,
)


UTC = timezone.utc
LONDON = ZoneInfo("Europe/London")


def flat(start_utc: str, end_utc_exclusive: str) -> pd.DataFrame:
    index = pd.date_range(
        start_utc,
        end_utc_exclusive,
        freq="15min",
        inclusive="left",
        tz="UTC",
    ).as_unit("us")
    return pd.DataFrame(
        {
            "open": np.full(len(index), 1.10000),
            "high": np.full(len(index), 1.10000),
            "low": np.full(len(index), 1.10000),
            "close": np.full(len(index), 1.10000),
            "volume": np.ones(len(index)),
        },
        index=index,
    )


def set_candle(
    df: pd.DataFrame,
    utc_time: str,
    o: float,
    h: float,
    l: float,
    c: float,
) -> None:
    timestamp = pd.Timestamp(utc_time, tz="UTC")
    df.loc[timestamp, ["open", "high", "low", "close"]] = [
        round(o, 5),
        round(h, 5),
        round(l, 5),
        round(c, 5),
    ]


def summer_base() -> pd.DataFrame:
    df = flat("2023-06-12 00:00", "2023-06-14 18:00")
    set_candle(df, "2023-06-13 23:45", 1.10000, 1.10000, 1.09500, 1.10000)
    set_candle(df, "2023-06-14 00:00", 1.10000, 1.10000, 1.09900, 1.10000)
    set_candle(df, "2023-06-14 06:45", 1.10000, 1.10300, 1.10000, 1.10050)
    return df


def winter_base() -> pd.DataFrame:
    df = flat("2023-12-11 00:00", "2023-12-13 19:00")
    set_candle(df, "2023-12-13 00:45", 1.10000, 1.10500, 1.10000, 1.10000)
    set_candle(df, "2023-12-13 01:00", 1.10000, 1.10100, 1.10000, 1.10000)
    set_candle(df, "2023-12-13 07:45", 1.10000, 1.10000, 1.09700, 1.09950)
    return df


def assert_plan(
    plan: pd.Series,
    *,
    side: str,
    start: str,
    entry: float,
    stop: float,
    exit_time: str,
    plan_id: str,
) -> None:
    assert plan["side"] == side
    assert plan["order_type"] == "market"
    assert plan["start_time"] == pd.Timestamp(start, tz="UTC")
    assert plan["entry_price"] == pytest.approx(entry, abs=1e-12)
    assert plan["stop_price"] == pytest.approx(stop, abs=1e-12)
    assert plan["target_r"] == 1.5
    assert pd.isna(plan["target_price"])
    assert bool(plan["no_target"]) is False
    assert plan["valid_candles"] == 1
    assert plan["exit_mode"] == "time_close"
    assert plan["time_exit_at"] == pd.Timestamp(exit_time, tz="UTC")
    assert plan["plan_id"] == plan_id


@pytest.mark.parametrize(
    ("case", "breakout_time", "expected_start", "expected_entry"),
    [
        ("S1", "2023-06-14 07:00", "2023-06-14 07:15", 1.10350),
        ("S2", "2023-06-14 08:15", "2023-06-14 08:30", 1.10350),
    ],
)
def test_summer_primary(
    case: str,
    breakout_time: str,
    expected_start: str,
    expected_entry: float,
) -> None:
    df = summer_base()
    set_candle(df, breakout_time, 1.10000, 1.10400, 1.10000, 1.10350)
    if breakout_time == "2023-06-14 08:15":
        set_candle(df, "2023-06-14 07:45", 1.10000, 1.10050, 1.10000, 1.10050)
    set_candle(df, expected_start, expected_entry, expected_entry, expected_entry, expected_entry)
    plans = find_candidate_d_plans(df)
    assert len(plans) == 1
    assert_plan(
        plans.iloc[0],
        side="long",
        start=expected_start,
        entry=expected_entry,
        stop=1.09900,
        exit_time="2023-06-14 15:45",
        plan_id="D-2023-06-14",
    )


def test_s1_funnel_and_s3_s4_windows() -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    assert candidate_d_funnel(df) == {
        "dates_with_data": 3,
        "complete_range": 3,
        "range_ok": 1,
        "breakout": 1,
        "trend_ok": 1,
        "plans": 1,
        "dropped_missing_entry_candle": 0,
        "plans_missing_exit_candle": 0,
    }

    for breakout_time in ("2023-06-14 08:30", "2023-06-14 07:30"):
        changed = summer_base()
        if breakout_time == "2023-06-14 08:30":
            set_candle(changed, "2023-06-14 07:45", 1.10000, 1.10050, 1.10000, 1.10050)
        set_candle(changed, breakout_time, 1.10000, 1.10400, 1.10000, 1.10350)
        funnel = candidate_d_funnel(changed)
        assert funnel["breakout"] == (0 if breakout_time.endswith("08:30") else 1)
        if breakout_time == "2023-06-14 08:30":
            assert funnel == {
                "dates_with_data": 3,
                "complete_range": 3,
                "range_ok": 1,
                "breakout": 0,
                "trend_ok": 0,
                "plans": 0,
                "dropped_missing_entry_candle": 0,
                "plans_missing_exit_candle": 0,
            }

    no_bo = summer_base()
    set_candle(no_bo, "2023-06-14 06:30", 1.10000, 1.10500, 1.10000, 1.10450)
    assert find_candidate_d_plans(no_bo).empty
    assert candidate_d_funnel(no_bo) == {
        "dates_with_data": 3,
        "complete_range": 3,
        "range_ok": 1,
        "breakout": 0,
        "trend_ok": 0,
        "plans": 0,
        "dropped_missing_entry_candle": 0,
        "plans_missing_exit_candle": 0,
    }


@pytest.mark.parametrize(
    ("high", "plan_count", "entry"),
    [(1.10099, 0, None), (1.10100, 1, 1.10150), (1.10500, 1, 1.10550), (1.10501, 0, None)],
)
def test_width_boundaries(high: float, plan_count: int, entry: float | None) -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 06:45", 1.10000, high, 1.10000, 1.10050)
    set_candle(df, "2023-06-14 07:00", 1.10000, high + 0.00050, 1.10000, high + 0.00050)
    if entry is not None:
        set_candle(df, "2023-06-14 07:15", entry, entry, entry, entry)
    plans = find_candidate_d_plans(df)
    assert len(plans) == plan_count


@pytest.mark.parametrize(
    ("close", "plan_count"),
    [(1.10300, 0), (1.10301, 1)],
)
def test_breakout_close_strict(close: float, plan_count: int) -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, close)
    if plan_count:
        set_candle(df, "2023-06-14 07:15", close, close, close, close)
    assert len(find_candidate_d_plans(df)) == plan_count


def test_first_breakout_and_trend_fail() -> None:
    s7 = summer_base()
    set_candle(s7, "2023-06-14 06:45", 1.10000, 1.10300, 1.09950, 1.09950)
    set_candle(s7, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(s7, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    assert candidate_d_funnel(s7)["trend_ok"] == 0

    s8 = s7.copy()
    set_candle(s8, "2023-06-14 07:30", 1.10000, 1.10000, 1.09850, 1.09850)
    assert find_candidate_d_plans(s8).empty

    s9 = summer_base()
    set_candle(s9, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(s9, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    set_candle(s9, "2023-06-14 07:30", 1.10000, 1.10000, 1.09850, 1.09850)
    assert len(find_candidate_d_plans(s9)) == 1


def test_one_plan_per_date_even_with_second_valid_breakout() -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    set_candle(df, "2023-06-14 07:30", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:45", 1.10350, 1.10350, 1.10350, 1.10350)

    plans = find_candidate_d_plans(df)

    assert len(plans) == 1
    assert plans.iloc[0]["start_time"] == pd.Timestamp("2023-06-14 07:15", tz="UTC")


def test_ema_50_alpha_hand_derived() -> None:
    close_times = pd.date_range(
        "2023-01-01 01:00",
        periods=60,
        freq="h",
        tz="UTC",
    )
    closes = [1.10000] * 58 + [1.20000, 1.10396]
    h1 = [
        {"close_time": close_time, "close": close}
        for close_time, close in zip(close_times, closes)
    ]

    trend = _ema_50_trend(
        h1,
        ema_period=50,
        breakout_close=close_times[-1] + pd.Timedelta(minutes=1),
    )

    assert trend == "long"


@pytest.mark.parametrize(
    ("breakout_close", "plan_count"),
    [(1.09700, 0), (1.09699, 1)],
)
def test_short_mirror(breakout_close: float, plan_count: int) -> None:
    df = summer_base()
    set_candle(df, "2023-06-13 23:45", 1.10000, 1.10500, 1.10000, 1.10000)
    set_candle(df, "2023-06-14 00:00", 1.10000, 1.10100, 1.10000, 1.10000)
    set_candle(df, "2023-06-14 06:45", 1.10000, 1.10000, 1.09700, 1.09950)
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10000, 1.09650, breakout_close)
    if plan_count:
        set_candle(df, "2023-06-14 07:15", breakout_close, breakout_close, breakout_close, breakout_close)
    plans = find_candidate_d_plans(df)
    assert len(plans) == plan_count


@pytest.mark.parametrize(
    ("start", "plan_count", "complete_range"),
    [("2023-06-12 05:00", 1, 2), ("2023-06-12 06:00", 0, 2)],
)
def test_warmup(start: str, plan_count: int, complete_range: int) -> None:
    df = flat(start, "2023-06-14 18:00")
    set_candle(df, "2023-06-13 23:45", 1.10000, 1.10000, 1.09500, 1.10000)
    set_candle(df, "2023-06-14 00:00", 1.10000, 1.10000, 1.09900, 1.10000)
    set_candle(df, "2023-06-14 06:45", 1.10000, 1.10300, 1.10000, 1.10050)
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    plans = find_candidate_d_plans(df)
    assert len(plans) == plan_count
    assert candidate_d_funnel(df)["complete_range"] == complete_range


def test_missing_range_entry_and_exit_candle() -> None:
    s12 = summer_base()
    set_candle(s12, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    s12 = s12.drop(pd.Timestamp("2023-06-14 03:15", tz="UTC"))
    assert candidate_d_funnel(s12)["range_ok"] == 0

    s13 = summer_base()
    set_candle(s13, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    s13 = s13.drop(pd.Timestamp("2023-06-14 07:15", tz="UTC"))
    funnel = candidate_d_funnel(s13)
    assert funnel["plans"] == 0
    assert funnel["dropped_missing_entry_candle"] == 1

    s14 = summer_base()
    set_candle(s14, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(s14, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    s14 = s14.drop(pd.Timestamp("2023-06-14 15:45", tz="UTC"))
    plans = find_candidate_d_plans(s14)
    assert len(plans) == 1
    assert candidate_d_funnel(s14)["plans_missing_exit_candle"] == 1


def test_winter_fixture() -> None:
    df = winter_base()
    set_candle(df, "2023-12-13 08:45", 1.10000, 1.10000, 1.09650, 1.09680)
    set_candle(df, "2023-12-13 09:00", 1.09680, 1.09680, 1.09680, 1.09680)
    plans = find_candidate_d_plans(df)
    assert len(plans) == 1
    assert_plan(
        plans.iloc[0],
        side="short",
        start="2023-12-13 09:00",
        entry=1.09680,
        stop=1.10100,
        exit_time="2023-12-13 16:45",
        plan_id="D-2023-12-13",
    )
    assert candidate_d_funnel(df)["plans"] == 1


@pytest.mark.parametrize(
    ("breakout_time", "entry_time", "plan_count"),
    [
        ("2023-12-13 08:00", "2023-12-13 08:15", 1),
        ("2023-12-13 09:30", None, 0),
    ],
)
def test_winter_window(breakout_time: str, entry_time: str | None, plan_count: int) -> None:
    df = winter_base()
    set_candle(df, breakout_time, 1.10000, 1.10000, 1.09650, 1.09680)
    if entry_time:
        set_candle(df, entry_time, 1.09680, 1.09680, 1.09680, 1.09680)
    assert len(find_candidate_d_plans(df)) == plan_count


def test_winter_h1_incomplete_and_range() -> None:
    df = winter_base()
    set_candle(df, "2023-12-13 08:45", 1.10000, 1.10000, 1.09650, 1.09680)
    set_candle(df, "2023-12-13 09:00", 1.09680, 1.09680, 1.09680, 1.09680)
    df = df.drop(pd.Timestamp("2023-12-13 08:15", tz="UTC"))
    assert candidate_d_funnel(df)["trend_ok"] == 0

    w4 = winter_base()
    set_candle(w4, "2023-12-13 07:15", 1.10000, 1.10000, 1.09650, 1.09650)
    assert find_candidate_d_plans(w4).empty


def test_output_columns_timezone_sort_and_empty() -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    plans = find_candidate_d_plans(df)
    assert list(plans.columns) == list(PLAN_COLUMNS)
    assert str(plans["start_time"].dt.tz) == "UTC"
    assert str(plans["time_exit_at"].dt.tz) == "UTC"
    assert plans["start_time"].is_monotonic_increasing

    empty = pd.DataFrame(
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    assert find_candidate_d_plans(empty).empty
    assert list(find_candidate_d_plans(empty).columns) == list(PLAN_COLUMNS)


def test_naive_timezone_microsecond_ns_and_copy() -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    with pytest.raises(ValueError):
        find_candidate_d_plans(df.set_axis(df.index.tz_localize(None)))

    ny = df.copy()
    ny.index = ny.index.tz_convert("America/New_York")
    ns = df.copy()
    ns.index = ns.index.as_unit("ns")

    assert str(df.index.dtype).endswith("[us, UTC]")
    pd.testing.assert_frame_equal(df, df.copy())
    original = df.copy(deep=True)
    find_candidate_d_plans(df)
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(
        find_candidate_d_plans(df).reset_index(drop=True),
        find_candidate_d_plans(ny).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        find_candidate_d_plans(df).reset_index(drop=True),
        find_candidate_d_plans(ns).reset_index(drop=True),
    )


def reference_find(df: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, int]]:
    records: list[dict[str, object]] = []
    utc_records: dict[datetime, dict[str, float]] = {}
    for timestamp, row in df.iterrows():
        dt = timestamp.to_pydatetime().astimezone(UTC)
        utc_records[dt] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }

    london_dates: set[object] = set()
    for dt in utc_records:
        london_dates.add(dt.astimezone(LONDON).date())
    funnel = {
        "dates_with_data": len(london_dates),
        "complete_range": 0,
        "range_ok": 0,
        "breakout": 0,
        "trend_ok": 0,
        "plans": 0,
        "dropped_missing_entry_candle": 0,
        "plans_missing_exit_candle": 0,
    }

    h1: dict[datetime, float] = {}
    hour_groups: dict[datetime, list[datetime]] = {}
    for dt in sorted(utc_records):
        hour = dt.replace(minute=0, second=0, microsecond=0)
        hour_groups.setdefault(hour, []).append(dt)
    complete_h1: list[tuple[datetime, float]] = []
    for hour in sorted(hour_groups):
        times = sorted(hour_groups[hour])
        if len(times) == 4 and [t.minute for t in times] == [0, 15, 30, 45]:
            close = utc_records[times[-1]]["close"]
            complete_h1.append((hour + timedelta(hours=1), close))

    for london_date in sorted(london_dates):
        range_times: list[datetime] = []
        complete = True
        for index in range(28):
            hour = 1 + index // 4
            minute = (index % 4) * 15
            local_naive = datetime(london_date.year, london_date.month, london_date.day, hour, minute)
            utc_dt = local_naive.replace(tzinfo=LONDON).astimezone(UTC)
            if utc_dt not in utc_records:
                complete = False
                break
            range_times.append(utc_dt)
        if not complete:
            continue
        funnel["complete_range"] += 1
        range_high = max(utc_records[t]["high"] for t in range_times)
        range_low = min(utc_records[t]["low"] for t in range_times)
        width = round((range_high - range_low) / 0.0001, 6)
        if not 20.0 <= width <= 60.0:
            continue
        funnel["range_ok"] += 1

        breakout = None
        side = None
        for index in range(20):
            hour = 8 + index // 4
            minute = (index % 4) * 15
            if hour == 9 and minute > 15:
                break
            local_naive = datetime(london_date.year, london_date.month, london_date.day, hour, minute)
            utc_dt = local_naive.replace(tzinfo=LONDON).astimezone(UTC)
            if utc_dt not in utc_records:
                continue
            close = utc_records[utc_dt]["close"]
            if close > range_high:
                breakout, side = utc_dt, "long"
                break
            if close < range_low:
                breakout, side = utc_dt, "short"
                break
        if breakout is None:
            continue
        funnel["breakout"] += 1

        ema = None
        ema_values: list[float] = []
        if len(complete_h1) >= 50:
            ema = complete_h1[0][1]
            ema_values = [ema]
            for _, close in complete_h1[1:]:
                ema = (2.0 / 51.0) * close + (49.0 / 51.0) * ema
                ema_values.append(ema)

        target_close = (breakout + timedelta(minutes=15)).replace(
            minute=0, second=0, microsecond=0
        )
        trend_side = None
        if ema is not None:
            for index, ((close_time, close), ema_value) in enumerate(
                zip(complete_h1, ema_values)
            ):
                if index < 49:
                    continue
                if close_time == target_close:
                    trend_side = "long" if close > ema_value else "short" if close < ema_value else None
                    break
        if trend_side != side:
            continue
        funnel["trend_ok"] += 1

        entry = breakout + timedelta(minutes=15)
        if entry not in utc_records:
            funnel["dropped_missing_entry_candle"] += 1
            continue
        stop = range_low if side == "long" else range_high
        exit_local = datetime(
            london_date.year,
            london_date.month,
            london_date.day,
            16,
            45,
            tzinfo=LONDON,
        )
        exit_utc = exit_local.astimezone(UTC)
        records.append(
            {
                "plan_id": f"D-{london_date.isoformat()}",
                "side": side,
                "start_time": entry,
                "entry_price": utc_records[entry]["open"],
                "stop_price": stop,
                "target_r": 1.5,
                "time_exit_at": exit_utc,
            }
        )
        funnel["plans"] += 1
        if exit_utc not in utc_records:
            funnel["plans_missing_exit_candle"] += 1
    return records, funnel


def seeded_frames(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 2, tzinfo=UTC)
    rows: list[tuple[datetime, float, float, float, float]] = []
    close = 1.10
    day = start
    while day.date() <= datetime(2023, 7, 31).date():
        if day.weekday() < 5:
            for position in range(96):
                timestamp = day + timedelta(minutes=15 * position)
                log_return = rng.normal(0.0, 0.0004)
                next_close = close * math.exp(log_return)
                open_ = close
                high = round(max(open_, next_close) + abs(rng.normal(0.0, 0.00005)), 5)
                low = round(min(open_, next_close) - abs(rng.normal(0.0, 0.00005)), 5)
                close = round(next_close, 5)
                open_ = round(open_, 5)
                rows.append((timestamp, open_, high, low, close))
        day += timedelta(days=1)

    rng_remove = np.random.default_rng(seed + 100)
    keep = rng_remove.random(len(rows)) >= 0.005
    rows = [row for row, keep_row in zip(rows, keep) if keep_row]
    index = pd.DatetimeIndex([row[0] for row in rows], tz="UTC").as_unit("us")
    return pd.DataFrame(
        {
            "open": [row[1] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[3] for row in rows],
            "close": [row[4] for row in rows],
            "volume": np.ones(len(rows)),
        },
        index=index,
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_seeded_independent_reference(seed: int) -> None:
    df = seeded_frames(seed)
    plans = find_candidate_d_plans(df)
    reference, ref_funnel = reference_find(df)
    assert len(plans) == len(reference)
    for actual, expected in zip(plans.to_dict("records"), reference):
        assert actual["side"] == expected["side"]
        assert actual["start_time"].to_pydatetime().astimezone(UTC) == expected["start_time"]
        assert actual["entry_price"] == pytest.approx(expected["entry_price"], abs=1e-12)
        assert actual["stop_price"] == pytest.approx(expected["stop_price"], abs=1e-12)
        assert actual["target_r"] == pytest.approx(expected["target_r"], abs=1e-12)
        assert actual["time_exit_at"].to_pydatetime().astimezone(UTC) == expected["time_exit_at"]
        assert actual["plan_id"] == expected["plan_id"]
    assert candidate_d_funnel(df) == ref_funnel


def test_seeded_total_and_invariants() -> None:
    frames = [seeded_frames(seed) for seed in range(1, 6)]
    plans_list = [find_candidate_d_plans(df) for df in frames]
    total = sum(len(plans) for plans in plans_list)
    longs = sum(int((plans["side"] == "long").sum()) for plans in plans_list)
    shorts = sum(int((plans["side"] == "short").sum()) for plans in plans_list)
    assert total >= 20
    assert longs >= 3
    assert shorts >= 3

    for df, plans in zip(frames, plans_list):
        funnel = candidate_d_funnel(df)
        assert funnel["plans"] == len(plans)
        keys = ["dates_with_data", "complete_range", "range_ok", "breakout", "trend_ok", "plans"]
        assert all(funnel[key] <= funnel[previous] for previous, key in zip(keys, keys[1:]))
        counts = count_candidate_d(df)
        assert sum(counts["per_year"].values()) == counts["plans"]
        assert counts["long"] + counts["short"] == counts["plans"]


def test_engine_consistency_for_seeded_frames() -> None:
    for seed in range(1, 6):
        df = seeded_frames(seed)
        plans = find_candidate_d_plans(df)
        trades = run_backtest(df, plans)
        assert len(trades.loc[trades["status"] == "filled"]) == len(plans)
        assert not (trades["status"] == "skipped_busy").any()
        assert not (trades["status"] == "unfilled_expired").any()
        for _, plan in plans.iterrows():
            trade = trades.loc[trades["plan_id"] == plan["plan_id"]].iloc[0]
            assert trade["fill_time"] == plan["start_time"]
            assert trade["exit_time"] <= plan["time_exit_at"] + pd.Timedelta(minutes=15)
            if trade["exit_reason"] == "time":
                assert trade["exit_time"] == plan["time_exit_at"]


def test_summer_long_time_close_at_candle_close() -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    set_candle(df, "2023-06-14 15:45", 1.10000, 1.10100, 1.09950, 1.10080)
    set_candle(df, "2023-06-14 16:00", 1.05000, 1.05000, 1.05000, 1.05000)

    plans = find_candidate_d_plans(df)
    trades = run_backtest(df, plans)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["status"] == "filled"
    assert trade["side"] == "long"
    assert trade["fill_time"] == pd.Timestamp("2023-06-14 07:15", tz="UTC")
    assert trade["entry_price"] == pytest.approx(1.10362, abs=1e-9)
    assert trade["risk_pips"] == pytest.approx(46.2, abs=1e-9)
    assert trade["exit_time"] == pd.Timestamp("2023-06-14 15:45", tz="UTC")
    assert trade["exit_price"] == pytest.approx(1.10078, abs=1e-9)
    assert trade["pnl_pips"] == pytest.approx(-28.4, abs=1e-9)
    assert trade["r_multiple"] == pytest.approx(-0.614719, abs=1e-4)


def test_winter_short_time_close_at_candle_close() -> None:
    df = winter_base()
    set_candle(df, "2023-12-13 08:45", 1.10000, 1.10000, 1.09650, 1.09680)
    set_candle(df, "2023-12-13 09:00", 1.09680, 1.09680, 1.09680, 1.09680)
    set_candle(df, "2023-12-13 16:45", 1.09700, 1.09700, 1.09600, 1.09650)
    set_candle(df, "2023-12-13 17:00", 1.20000, 1.20000, 1.20000, 1.20000)

    plans = find_candidate_d_plans(df)
    trades = run_backtest(df, plans)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["status"] == "filled"
    assert trade["side"] == "short"
    assert trade["fill_time"] == pd.Timestamp("2023-12-13 09:00", tz="UTC")
    assert trade["entry_price"] == pytest.approx(1.09678, abs=1e-9)
    assert trade["risk_pips"] == pytest.approx(42.2, abs=1e-9)
    assert trade["exit_time"] == pd.Timestamp("2023-12-13 16:45", tz="UTC")
    assert trade["exit_price"] == pytest.approx(1.09662, abs=1e-9)
    assert trade["pnl_pips"] == pytest.approx(1.6, abs=1e-9)
    assert trade["r_multiple"] == pytest.approx(0.037915, abs=1e-4)


def test_no_lookahead_and_entry_truncation() -> None:
    compare_columns = (
        "plan_id",
        "side",
        "start_time",
        "entry_price",
        "stop_price",
        "target_r",
        "time_exit_at",
    )
    ohlcv_columns = ["open", "high", "low", "close", "volume"]

    for seed in (1, 2):
        df = seeded_frames(seed)
        plans = find_candidate_d_plans(df)
        assert len(plans) >= 25
        for _, plan in plans.iloc[:25].iterrows():
            entry = plan["start_time"]
            entry_pos = int(df.index.get_loc(entry))
            garbage = df.copy()
            price_columns = garbage.columns.get_indexer(["high", "low", "close"])
            garbage.iloc[entry_pos, price_columns] = -999.0
            later_columns = garbage.columns.get_indexer(ohlcv_columns)
            garbage.iloc[entry_pos + 1 :, later_columns] = -999.0

            copy_plans = find_candidate_d_plans(garbage)
            matching = copy_plans.loc[copy_plans["plan_id"] == plan["plan_id"]]
            assert len(matching) == 1
            for column in compare_columns:
                assert matching.iloc[0][column] == plan[column]

        if seed == 1:
            plan = plans.iloc[0]
            truncated = df.loc[: plan["start_time"]]
            truncated_plans = find_candidate_d_plans(truncated)
            assert len(
                truncated_plans.loc[
                    truncated_plans["plan_id"] == plan["plan_id"]
                ]
            ) == 1


def test_count_and_format_report() -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    counts = count_candidate_d(df)
    assert counts["plans"] == 1
    assert counts["long"] == 1
    assert counts["short"] == 0
    assert counts["per_year"][2023] == 1

    from scripts.count_candidate_d_dev import format_count_report

    funnel = candidate_d_funnel(df)
    report129 = format_count_report(funnel, {"plans": 129, "long": 70, "short": 59, "per_year": {2023: 129}}, np.array([20.0]))
    report130 = format_count_report(funnel, {"plans": 130, "long": 71, "short": 59, "per_year": {2023: 130}}, np.array([20.0]))
    assert "COUNT GATE (>= 130 development plans): NOT MET" in report129
    assert "COUNT GATE (>= 130 development plans): MET" in report130
    for key in funnel:
        assert f"{key}: " in report129

    source = open("scripts/count_candidate_d_dev.py", encoding="utf-8").read()
    assert "run_backtest" not in source
    assert "summarize_trades" not in source


def test_london_date_grouping_across_utc_midnight() -> None:
    df = flat("2023-06-13 22:45", "2023-06-14 00:00")

    funnel = candidate_d_funnel(df)

    assert funnel["dates_with_data"] == 2
