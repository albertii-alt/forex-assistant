"""Candidate A setup discovery; emits plans without simulating trades."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from numbers import Real
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.detectors.fvg import OUTPUT_COLUMNS as FVG_COLUMNS
from src.detectors.fvg import detect_fvgs
from src.detectors.session_levels import detect_session_levels
from src.detectors.structure import OUTPUT_COLUMNS as BREAK_COLUMNS
from src.detectors.structure import detect_structure_breaks


NEW_YORK = ZoneInfo("America/New_York")
WINDOW_START_ET = time(9, 30)
SETUP_DEADLINE_ET = time(11, 30)
TIME_EXIT_ET = time(11, 45)
CANDLE_DURATION = pd.Timedelta(minutes=15)
PIP_SIZE = 0.0001
LEVEL_COLUMNS = (
    "trading_day",
    "prev_day_high",
    "prev_day_high_set_at",
    "prev_day_low",
    "prev_day_low_set_at",
    "prev_day_known_at",
    "overnight_high",
    "overnight_high_set_at",
    "overnight_low",
    "overnight_low_set_at",
    "overnight_known_at",
)
PLAN_COLUMNS = (
    "trading_day",
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
    "min_risk_pips",
    "max_risk_pips",
    "risk_pips",
    "sweep_time",
    "break_time",
    "fvg_formed_at",
    "fvg_zone_low",
    "fvg_zone_high",
)
UTC_COLUMNS = (
    "start_time",
    "time_exit_at",
    "sweep_time",
    "break_time",
    "fvg_formed_at",
)
PRICE_COLUMNS = (
    "entry_price",
    "stop_price",
    "target_r",
    "target_price",
    "min_risk_pips",
    "max_risk_pips",
    "risk_pips",
    "fvg_zone_low",
    "fvg_zone_high",
)


def _validate_numeric(name: str, value: object, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if strictly_positive and number <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if not strictly_positive and number < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return number


def _validate_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(value)


def _validate_df(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")
    missing = [column for column in ("open", "high", "low", "close") if column not in df]
    if missing:
        raise ValueError(f"df is missing required columns: {', '.join(missing)}")


def _validate_frame(
    frame: pd.DataFrame,
    required: tuple[str, ...] | list[str],
    name: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _as_utc(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if timestamp.tz is None or str(timestamp.tz).upper() != "UTC":
        return None
    return timestamp.tz_convert("UTC").as_unit("ns")


def _utc_wall_time(day: date, wall_time: time) -> pd.Timestamp:
    local = datetime.combine(day, wall_time, tzinfo=NEW_YORK)
    return pd.Timestamp(local).tz_convert("UTC").as_unit("ns")


def _empty_plan_frame() -> pd.DataFrame:
    result = pd.DataFrame(columns=PLAN_COLUMNS)
    result["trading_day"] = pd.Series(dtype="datetime64[ns]")
    for column in UTC_COLUMNS:
        result[column] = pd.Series(dtype="datetime64[ns, UTC]")
    result["plan_id"] = pd.Series(dtype="object")
    result["side"] = pd.Series(dtype="object")
    result["order_type"] = pd.Series(dtype="object")
    result["valid_candles"] = pd.Series(dtype="int64")
    for column in PRICE_COLUMNS:
        result[column] = pd.Series(dtype="float64")
    return result.loc[:, list(PLAN_COLUMNS)]


def _plan_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty_plan_frame()
    result = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    result["trading_day"] = pd.to_datetime(result["trading_day"]).astype(
        "datetime64[ns]"
    )
    for column in UTC_COLUMNS:
        result[column] = pd.Series(
            pd.DatetimeIndex(pd.to_datetime(result[column], utc=True)).as_unit("ns"),
            index=result.index,
        )
    result["valid_candles"] = result["valid_candles"].astype("int64")
    for column in PRICE_COLUMNS:
        result[column] = result[column].astype("float64")
    result["plan_id"] = result["plan_id"].astype("object")
    result["side"] = result["side"].astype("object")
    result["order_type"] = result["order_type"].astype("object")
    return result.loc[:, list(PLAN_COLUMNS)]


def _index_positions(index: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    positions: dict[pd.Timestamp, int] = {}
    for position, timestamp in enumerate(index):
        positions.setdefault(timestamp, position)
    return positions


def _precompute_break_positions(
    breaks: pd.DataFrame,
    timestamp_positions: dict[pd.Timestamp, int],
) -> dict[str, np.ndarray]:
    result: dict[str, list[tuple[int, int]]] = {
        "bullish": [],
        "bearish": [],
    }
    for row_order, break_row in breaks.iterrows():
        direction = break_row["direction"]
        if direction not in result:
            continue
        broke_at = _as_utc(break_row["broke_at"])
        if broke_at is None:
            continue
        break_pos = timestamp_positions.get(broke_at)
        if break_pos is not None:
            result[direction].append((break_pos, row_order))

    arrays: dict[str, np.ndarray] = {}
    for direction, entries in result.items():
        if not entries:
            arrays[direction] = np.empty(0, dtype=np.int64)
            continue
        positions = np.asarray([entry[0] for entry in entries], dtype=np.int64)
        original_positions = positions.copy()
        if len(positions) > 1:
            non_decreasing = bool(np.all(positions[1:] >= positions[:-1]))
            if not non_decreasing:
                stable_order = np.argsort(positions, kind="stable")
                positions = positions[stable_order]
        arrays[direction] = positions
    return arrays


def _precompute_fvg_rows(
    fvgs: pd.DataFrame,
    timestamp_positions: dict[pd.Timestamp, int],
    min_fvg_size: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result: dict[str, list[tuple[int, int, float, float, pd.Timestamp]]] = {
        "bullish": [],
        "bearish": [],
    }
    for row_order, fvg_row in fvgs.iterrows():
        direction = fvg_row["direction"]
        if direction not in result:
            continue
        try:
            size = float(fvg_row["size"])
            zone_low = float(fvg_row["zone_low"])
            zone_high = float(fvg_row["zone_high"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite([size, zone_low, zone_high]).all():
            continue
        if size < min_fvg_size - 1e-9:
            continue
        formed_at = _as_utc(fvg_row["formed_at"])
        if formed_at is None:
            continue
        formed_pos = timestamp_positions.get(formed_at)
        if formed_pos is not None:
            result[direction].append(
                (formed_pos, row_order, zone_low, zone_high, formed_at)
            )

    arrays: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for direction, entries in result.items():
        if not entries:
            empty = np.empty(0, dtype=np.int64)
            arrays[direction] = (
                empty,
                empty.copy(),
                np.empty(0),
                np.empty(0),
                np.empty(0, dtype=object),
            )
            continue
        entries.sort(key=lambda item: (item[0], -item[1]))
        formed_positions = np.asarray([item[0] for item in entries], dtype=np.int64)
        row_orders = np.asarray([item[1] for item in entries], dtype=np.int64)
        zone_low = np.asarray([item[2] for item in entries], dtype=np.float64)
        zone_high = np.asarray([item[3] for item in entries], dtype=np.float64)
        arrays[direction] = (
            formed_positions,
            row_orders,
            zone_low,
            zone_high,
            np.asarray([item[4] for item in entries], dtype=object),
        )
    return arrays


def _setup_for_direction(
    df: pd.DataFrame,
    day: date,
    day_levels: dict[str, object] | None,
    sweep_pos: int,
    side: str,
    breaks_by_direction: dict[str, np.ndarray],
    fvgs_by_direction: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    index: pd.DatetimeIndex,
    index_asi8: np.ndarray,
    deadline: pd.Timestamp,
    time_exit: pd.Timestamp,
    positions: dict[str, object],
) -> dict[str, object] | None:
    if day_levels is None:
        return None

    sweep_time = index[sweep_pos]
    if side == "long":
        sweep_valid = (
            (
                not np.isnan(float(day_levels["overnight_low"]))
                and day_levels["overnight_known_at"] is not None
                and day_levels["overnight_known_at"] <= sweep_time
                and low[sweep_pos] < float(day_levels["overnight_low"])
                and close[sweep_pos] >= float(day_levels["overnight_low"])
            )
            or (
                not np.isnan(float(day_levels["prev_day_low"]))
                and day_levels["prev_day_known_at"] is not None
                and day_levels["prev_day_known_at"] <= sweep_time
                and low[sweep_pos] < float(day_levels["prev_day_low"])
                and close[sweep_pos] >= float(day_levels["prev_day_low"])
            )
        )
    else:
        sweep_valid = (
            (
                not np.isnan(float(day_levels["overnight_high"]))
                and day_levels["overnight_known_at"] is not None
                and day_levels["overnight_known_at"] <= sweep_time
                and high[sweep_pos] > float(day_levels["overnight_high"])
                and close[sweep_pos] <= float(day_levels["overnight_high"])
            )
            or (
                not np.isnan(float(day_levels["prev_day_high"]))
                and day_levels["prev_day_known_at"] is not None
                and day_levels["prev_day_known_at"] <= sweep_time
                and high[sweep_pos] > float(day_levels["prev_day_high"])
                and close[sweep_pos] <= float(day_levels["prev_day_high"])
            )
        )
    if not sweep_valid:
        return None

    expected_direction = "bullish" if side == "long" else "bearish"
    latest_break_pos = min(sweep_pos + int(positions["confirm_candles"]), len(index) - 1)
    break_positions = breaks_by_direction[expected_direction]
    candidate = int(np.searchsorted(break_positions, sweep_pos + 1, side="left"))
    if candidate >= len(break_positions):
        return None
    break_pos = int(break_positions[candidate])
    if break_pos > latest_break_pos:
        return None
    if index[break_pos] + CANDLE_DURATION > deadline:
        return None

    formed_positions, row_orders, zone_lows, zone_highs, formed_times = (
        fvgs_by_direction[expected_direction]
    )
    lower = int(np.searchsorted(formed_positions, sweep_pos + 1, side="left"))
    upper = int(np.searchsorted(formed_positions, break_pos, side="right"))
    if lower >= upper:
        return None
    selected = lower
    latest_formed_pos = int(formed_positions[upper - 1])
    group_start = int(np.searchsorted(formed_positions, latest_formed_pos, side="left"))
    group_end = int(np.searchsorted(formed_positions, latest_formed_pos, side="right"))
    if group_start < lower:
        group_start = lower
    if group_start >= group_end:
        return None
    selected = group_start
    zone_low = float(zone_lows[selected])
    zone_high = float(zone_highs[selected])

    entry_price = zone_high if side == "long" else zone_low
    stop_price = (
        float(low[sweep_pos]) - float(positions["stop_buffer"])
        if side == "long"
        else float(high[sweep_pos]) + float(positions["stop_buffer"])
    )
    if (side == "long" and entry_price <= stop_price) or (
        side == "short" and entry_price >= stop_price
    ):
        return None
    risk_pips = abs(entry_price - stop_price) / float(positions["pip_size"])
    if (
        risk_pips < float(positions["min_risk_pips"]) - 1e-9
        or risk_pips > float(positions["max_risk_pips"]) + 1e-9
    ):
        return None

    start_time = index[break_pos] + CANDLE_DURATION
    start_pos = int(np.searchsorted(index_asi8, start_time.value, side="left"))
    if start_pos >= len(index) or index[start_pos] > time_exit:
        return None

    return {
        "trading_day": pd.Timestamp(day),
        "plan_id": day.isoformat(),
        "side": side,
        "order_type": "limit",
        "start_time": start_time,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_r": float(positions["target_r"]),
        "target_price": np.nan,
        "valid_candles": int(positions["valid_candles"]),
        "time_exit_at": time_exit,
        "min_risk_pips": float(positions["min_risk_pips"]),
        "max_risk_pips": float(positions["max_risk_pips"]),
        "risk_pips": risk_pips,
        "sweep_time": sweep_time,
        "break_time": index[break_pos],
        "fvg_formed_at": formed_times[selected],
        "fvg_zone_low": zone_low,
        "fvg_zone_high": zone_high,
        "_sweep_pos": sweep_pos,
        "_side_order": 0 if side == "long" else 1,
    }


def find_candidate_a_plans(
    df: pd.DataFrame,
    levels: pd.DataFrame | None = None,
    breaks: pd.DataFrame | None = None,
    fvgs: pd.DataFrame | None = None,
    swing_left: int = 3,
    swing_right: int = 3,
    min_fvg_pips: float = 2.0,
    stop_buffer_pips: float = 1.0,
    min_risk_pips: float = 5.0,
    max_risk_pips: float = 25.0,
    target_r: float = 2.0,
    confirm_candles: int = 8,
    valid_candles: int = 8,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Find at most one first-valid Candidate A plan per ET calendar date.

    The setup window is 09:30-11:30 ET. A break must close by 11:30 ET.
    Plans are placed at the following candle open and use the 11:45 ET candle
    open as the exit boundary (that candle closes at 12:00 ET). Prices are
    BID-side levels; spread and slippage are applied later by the backtester.
    Detector break frames are expected in position order. If an injected frame
    is unsorted, equal positions retain row order and the earliest position
    is used instead of the old first-row-position behavior.
    """
    _validate_df(df)
    df = df.copy()
    df.index = df.index.as_unit("ns")
    swing_left = _validate_count("swing_left", swing_left)
    swing_right = _validate_count("swing_right", swing_right)
    confirm_candles = _validate_count("confirm_candles", confirm_candles)
    valid_candles = _validate_count("valid_candles", valid_candles)
    min_fvg_pips = _validate_numeric("min_fvg_pips", min_fvg_pips)
    stop_buffer_pips = _validate_numeric("stop_buffer_pips", stop_buffer_pips)
    min_risk_pips = _validate_numeric("min_risk_pips", min_risk_pips)
    max_risk_pips = _validate_numeric("max_risk_pips", max_risk_pips)
    target_r = _validate_numeric("target_r", target_r, strictly_positive=True)
    pip_size = _validate_numeric("pip_size", pip_size, strictly_positive=True)

    if levels is None:
        levels = detect_session_levels(df)
    if breaks is None:
        breaks = detect_structure_breaks(df, left=swing_left, right=swing_right)
    if fvgs is None:
        fvgs = detect_fvgs(df, min_gap=min_fvg_pips * pip_size)

    _validate_frame(levels, LEVEL_COLUMNS, "levels")
    _validate_frame(breaks, BREAK_COLUMNS, "breaks")
    _validate_frame(fvgs, FVG_COLUMNS, "fvgs")

    if df.empty:
        return _empty_plan_frame()
    levels_by_day: dict[date, dict[str, object]] = {}
    for row in levels.to_dict(orient="records"):
        day_value = row["trading_day"]
        if pd.isna(day_value):
            continue
        day = pd.Timestamp(day_value).date()
        if day in levels_by_day:
            raise ValueError(f"levels contains duplicate trading_day {day.isoformat()}")
        levels_by_day[day] = {
            "overnight_low": (
                np.nan
                if pd.isna(row["overnight_low"])
                else float(row["overnight_low"])
            ),
            "overnight_high": (
                np.nan
                if pd.isna(row["overnight_high"])
                else float(row["overnight_high"])
            ),
            "prev_day_low": (
                np.nan
                if pd.isna(row["prev_day_low"])
                else float(row["prev_day_low"])
            ),
            "prev_day_high": (
                np.nan
                if pd.isna(row["prev_day_high"])
                else float(row["prev_day_high"])
            ),
            "overnight_known_at": _as_utc(row["overnight_known_at"]),
            "prev_day_known_at": _as_utc(row["prev_day_known_at"]),
        }

    local_dates = [timestamp.to_pydatetime().astimezone(NEW_YORK).date() for timestamp in df.index]
    days = sorted(set(local_dates))
    positions_by_day: dict[date, list[int]] = {}
    for position, day in enumerate(local_dates):
        positions_by_day.setdefault(day, []).append(position)
    timestamp_positions = _index_positions(df.index)
    low = df["low"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    index = df.index
    index_asi8 = index.asi8
    breaks_by_direction = _precompute_break_positions(
        breaks,
        timestamp_positions,
    )
    fvgs_by_direction = _precompute_fvg_rows(
        fvgs,
        timestamp_positions,
        min_fvg_pips * pip_size,
    )
    config: dict[str, object] = {
        "confirm_candles": confirm_candles,
        "stop_buffer": stop_buffer_pips * pip_size,
        "min_risk_pips": min_risk_pips,
        "max_risk_pips": max_risk_pips,
        "target_r": target_r,
        "valid_candles": valid_candles,
        "pip_size": pip_size,
    }
    candidates: list[dict[str, object]] = []
    for day in days:
        day_positions = positions_by_day[day]
        day_levels = levels_by_day.get(day)
        window_start = _utc_wall_time(day, WINDOW_START_ET)
        deadline = _utc_wall_time(day, SETUP_DEADLINE_ET)
        time_exit = _utc_wall_time(day, TIME_EXIT_ET)
        for sweep_pos in day_positions:
            candle_open = index[sweep_pos]
            if candle_open < window_start or candle_open + CANDLE_DURATION > deadline:
                continue
            for side in ("long", "short"):
                plan = _setup_for_direction(
                    df,
                    day,
                    day_levels,
                    sweep_pos,
                    side,
                    breaks_by_direction,
                    fvgs_by_direction,
                    low,
                    high,
                    close,
                    index,
                    index_asi8,
                    deadline,
                    time_exit,
                    config,
                )
                if plan is not None:
                    candidates.append(plan)
    selected: dict[date, dict[str, object]] = {}
    for plan in candidates:
        day = pd.Timestamp(plan["trading_day"]).date()
        current = selected.get(day)
        order_key = (
            plan["start_time"],
            int(plan["_sweep_pos"]),
            int(plan["_side_order"]),
        )
        if current is None or order_key < (
            current["start_time"],
            int(current["_sweep_pos"]),
            int(current["_side_order"]),
        ):
            selected[day] = plan

    rows = []
    for plan in sorted(selected.values(), key=lambda item: item["start_time"]):
        rows.append({column: plan[column] for column in PLAN_COLUMNS})
    return _plan_frame(rows)
