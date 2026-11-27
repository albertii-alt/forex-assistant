"""Candidate D London-open Asian-range plan discovery and counts."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


PLAN_COLUMNS = (
    "plan_id",
    "side",
    "order_type",
    "start_time",
    "entry_price",
    "stop_price",
    "target_r",
    "target_price",
    "valid_candles",
    "time_exit_at",
    "no_target",
    "exit_mode",
    "signal_time",
    "atr_n",
    "channel_level",
    "exit_signal_time",
)


def _empty_plans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plan_id": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"),
            "order_type": pd.Series(dtype="object"),
            "start_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "entry_price": pd.Series(dtype="float64"),
            "stop_price": pd.Series(dtype="float64"),
            "target_r": pd.Series(dtype="float64"),
            "target_price": pd.Series(dtype="float64"),
            "valid_candles": pd.Series(dtype="int64"),
            "time_exit_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "no_target": pd.Series(dtype="bool"),
            "exit_mode": pd.Series(dtype="object"),
            "signal_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "atr_n": pd.Series(dtype="float64"),
            "channel_level": pd.Series(dtype="float64"),
            "exit_signal_time": pd.Series(dtype="datetime64[ns, UTC]"),
        },
        columns=PLAN_COLUMNS,
    )


def _validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    missing = [column for column in ("open", "high", "low", "close") if column not in df]
    if missing:
        raise ValueError(f"df is missing required columns: {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df index must be timezone-aware")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")

    result = df.copy()
    result.index = result.index.tz_convert("UTC").as_unit("ns")
    return result, result.index


def _validate_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(value)


def _validate_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _build_complete_h1(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    utc = df.index
    hours: dict[pd.Timestamp, list[int]] = {}
    for position, timestamp in enumerate(utc):
        hour = timestamp.floor("h")
        hours.setdefault(hour, []).append(position)

    complete: list[dict[str, Any]] = []
    required_minutes = (0, 15, 30, 45)
    for open_hour in sorted(hours):
        positions = hours[open_hour]
        timestamps = [utc[position] for position in positions]
        if len(timestamps) != 4:
            continue
        minute_set = tuple(int(timestamp.minute) for timestamp in sorted(timestamps))
        if minute_set != required_minutes:
            continue
        source = df.iloc[positions]
        complete.append(
            {
                "open": open_hour,
                "close_time": open_hour + pd.Timedelta(hours=1),
                "close": float(source["close"].iloc[-1]),
            }
        )
    complete.sort(key=lambda row: row["close_time"])
    return complete


def _ema_50_trend(
    h1: list[dict[str, Any]],
    ema_period: int,
    breakout_close: pd.Timestamp,
) -> str | None:
    if len(h1) < ema_period:
        return None

    closes = [float(row["close"]) for row in h1]
    alpha = 2.0 / (ema_period + 1)
    ema = closes[0]
    target_close_time = breakout_close.floor("h")
    for index, row in enumerate(h1):
        if index == 0:
            ema_value = closes[0]
        else:
            ema_value = alpha * closes[index] + (1.0 - alpha) * ema_value
        if index >= ema_period - 1 and row["close_time"] == target_close_time:
            if float(row["close"]) > ema_value:
                return "long"
            if float(row["close"]) < ema_value:
                return "short"
            return None
    return None


def _candidate_d_internal(
    df: pd.DataFrame,
    *,
    range_min_pips: float,
    range_max_pips: float,
    ema_period: int,
    target_r: float,
    pip_size: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    data, utc = _validate(df)
    if data.empty:
        funnel = {
            "dates_with_data": 0,
            "complete_range": 0,
            "range_ok": 0,
            "breakout": 0,
            "trend_ok": 0,
            "plans": 0,
            "dropped_missing_entry_candle": 0,
            "plans_missing_exit_candle": 0,
        }
        return _empty_plans(), funnel

    london = data.index.tz_convert("Europe/London")
    london_by_date: dict[pd.Timestamp, list[int]] = {}
    for position, timestamp in enumerate(london):
        date = pd.Timestamp(timestamp.date())
        london_by_date.setdefault(date, []).append(position)

    utc_positions: dict[pd.Timestamp, int] = {
        timestamp: position for position, timestamp in enumerate(utc)
    }
    h1 = _build_complete_h1(data)
    funnel = {
        "dates_with_data": len(london_by_date),
        "complete_range": 0,
        "range_ok": 0,
        "breakout": 0,
        "trend_ok": 0,
        "plans": 0,
        "dropped_missing_entry_candle": 0,
        "plans_missing_exit_candle": 0,
    }
    rows: list[dict[str, Any]] = []

    for london_date in sorted(london_by_date):
        positions = london_by_date[london_date]
        if not positions:
            continue

        range_positions: list[int] = []
        complete_range = True
        for minute_index in range(28):
            local_hour = 1 + minute_index // 4
            local_minute = (minute_index % 4) * 15
            # Construct the local timestamp through pandas so DST is resolved.
            naive = pd.Timestamp(
                year=london_date.year,
                month=london_date.month,
                day=london_date.day,
                hour=local_hour,
                minute=local_minute,
            )
            local_timestamp = naive.tz_localize(
                "Europe/London",
                ambiguous=False,
                nonexistent="NaT",
            )
            if pd.isna(local_timestamp):
                complete_range = False
                break
            utc_timestamp = local_timestamp.tz_convert("UTC").as_unit("ns")
            position = utc_positions.get(utc_timestamp)
            if position is None:
                complete_range = False
                break
            range_positions.append(position)
        if not complete_range:
            continue
        funnel["complete_range"] += 1

        range_high = float(data.iloc[range_positions]["high"].max())
        range_low = float(data.iloc[range_positions]["low"].min())
        width_pips = round((range_high - range_low) / pip_size, 6)
        if not (range_min_pips <= width_pips <= range_max_pips):
            continue
        funnel["range_ok"] += 1

        breakout_position: int | None = None
        breakout_side: str | None = None
        for minute_index in range(5 * 4):
            local_hour = 8 + minute_index // 4
            local_minute = (minute_index % 4) * 15
            if local_hour == 9 and local_minute > 15:
                break
            naive = pd.Timestamp(
                year=london_date.year,
                month=london_date.month,
                day=london_date.day,
                hour=local_hour,
                minute=local_minute,
            )
            local_timestamp = naive.tz_localize(
                "Europe/London",
                ambiguous=False,
                nonexistent="NaT",
            )
            if pd.isna(local_timestamp):
                continue
            utc_timestamp = local_timestamp.tz_convert("UTC").as_unit("ns")
            position = utc_positions.get(utc_timestamp)
            if position is None:
                continue
            close = float(data.iloc[position]["close"])
            if close > range_high:
                breakout_position = position
                breakout_side = "long"
                break
            if close < range_low:
                breakout_position = position
                breakout_side = "short"
                break
        if breakout_position is None:
            continue
        funnel["breakout"] += 1

        breakout_close = utc[breakout_position] + pd.Timedelta(minutes=15)
        trend_side = _ema_50_trend(h1, ema_period, breakout_close)
        if trend_side != breakout_side:
            continue
        funnel["trend_ok"] += 1

        entry_time = utc[breakout_position] + pd.Timedelta(minutes=15)
        entry_position = utc_positions.get(entry_time)
        if entry_position is None:
            funnel["dropped_missing_entry_candle"] += 1
            continue
        entry_price = float(data.iloc[entry_position]["open"])
        stop_price = range_low if breakout_side == "long" else range_high

        exit_naive = pd.Timestamp(
            year=london_date.year,
            month=london_date.month,
            day=london_date.day,
            hour=16,
            minute=45,
        )
        exit_local = exit_naive.tz_localize("Europe/London", ambiguous=False)
        exit_time = exit_local.tz_convert("UTC").as_unit("ns")

        rows.append(
            {
                "plan_id": f"D-{london_date.strftime('%Y-%m-%d')}",
                "side": breakout_side,
                "order_type": "market",
                "start_time": entry_time,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_r": target_r,
                "target_price": np.nan,
                "valid_candles": 1,
                "time_exit_at": exit_time,
                "no_target": False,
                "exit_mode": "time_close",
                "signal_time": utc[breakout_position],
                "atr_n": np.nan,
                "channel_level": np.nan,
                "exit_signal_time": pd.NaT,
            }
        )
        funnel["plans"] += 1
        if utc_positions.get(exit_time) is None:
            funnel["plans_missing_exit_candle"] += 1

    if not rows:
        return _empty_plans(), funnel
    result = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    result["start_time"] = pd.to_datetime(result["start_time"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    result["time_exit_at"] = pd.to_datetime(result["time_exit_at"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    result["signal_time"] = pd.to_datetime(result["signal_time"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    result["exit_signal_time"] = pd.to_datetime(
        result["exit_signal_time"], utc=True
    ).astype("datetime64[ns, UTC]")
    for column in (
        "entry_price",
        "stop_price",
        "target_r",
        "target_price",
        "atr_n",
        "channel_level",
    ):
        result[column] = result[column].astype("float64")
    result["valid_candles"] = result["valid_candles"].astype("int64")
    result["no_target"] = result["no_target"].astype("bool")
    for column in ("plan_id", "side", "order_type", "exit_mode"):
        result[column] = result[column].astype("object")
    return result.sort_values("start_time", kind="stable").reset_index(drop=True), funnel


def find_candidate_d_plans(
    df: pd.DataFrame,
    *,
    range_min_pips: float = 20.0,
    range_max_pips: float = 60.0,
    ema_period: int = 50,
    target_r: float = 1.5,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Find one fixed Candidate D plan per London date."""
    range_min_pips = _validate_number("range_min_pips", range_min_pips)
    range_max_pips = _validate_number("range_max_pips", range_max_pips)
    ema_period = _validate_integer("ema_period", ema_period)
    target_r = _validate_number("target_r", target_r)
    pip_size = _validate_number("pip_size", pip_size)
    if range_min_pips < 0 or range_max_pips < range_min_pips:
        raise ValueError("range limits are invalid")
    if target_r <= 0:
        raise ValueError("target_r must be greater than zero")
    if pip_size <= 0:
        raise ValueError("pip_size must be greater than zero")
    plans, _ = _candidate_d_internal(
        df,
        range_min_pips=range_min_pips,
        range_max_pips=range_max_pips,
        ema_period=ema_period,
        target_r=target_r,
        pip_size=pip_size,
    )
    return plans


def candidate_d_funnel(df: pd.DataFrame) -> dict[str, int]:
    """Return the shared-rule cumulative Candidate D funnel."""
    _, funnel = _candidate_d_internal(
        df,
        range_min_pips=20.0,
        range_max_pips=60.0,
        ema_period=50,
        target_r=1.5,
        pip_size=0.0001,
    )
    return funnel


def count_candidate_d(df: pd.DataFrame) -> dict[str, Any]:
    """Count plans and plans by UTC year of start time."""
    plans = find_candidate_d_plans(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df index must be timezone-aware")
    utc = df.index.tz_convert("UTC")
    first_year = int(utc.min().year) if len(utc) else 0
    last_year = int(utc.max().year) if len(utc) else 0
    per_year = (
        {year: 0 for year in range(first_year, last_year + 1)}
        if len(utc)
        else {}
    )
    if not plans.empty:
        start_years = plans["start_time"].dt.tz_convert("UTC").dt.year
        for year in start_years:
            per_year[int(year)] += 1
    return {
        "plans": len(plans),
        "long": int((plans["side"] == "long").sum()) if not plans.empty else 0,
        "short": int((plans["side"] == "short").sum()) if not plans.empty else 0,
        "per_year": per_year,
    }
