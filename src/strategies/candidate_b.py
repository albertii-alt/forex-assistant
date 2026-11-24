"""Candidate B supply/demand setup discovery."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from numbers import Integral, Real
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.resample_et import align_htf_to_ltf, resample_ohlcv_et
from src.detectors.trend import htf_trend_state
from src.detectors.zones import ZONE_COLUMNS, detect_zones
from src.indicators.basic import stochastic


ET = ZoneInfo("America/New_York")
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
    "zone_base_time",
    "zone_known_at",
    "zone_low",
    "zone_high",
    "touch_time",
    "confirm_time",
    "htf_state",
)
UTC_COLUMNS = (
    "start_time",
    "time_exit_at",
    "zone_base_time",
    "zone_known_at",
    "touch_time",
    "confirm_time",
)
FLOAT_COLUMNS = (
    "entry_price",
    "stop_price",
    "target_r",
    "target_price",
    "min_risk_pips",
    "max_risk_pips",
    "risk_pips",
    "zone_low",
    "zone_high",
)


def _finite_number(name: str, value: float, *, minimum: float, strict: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value <= minimum if strict else value < minimum)
    ):
        relation = "greater than" if strict else "greater than or equal to"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return float(value)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(value)


def _validate_et_time(name: str, value: time) -> time:
    if not isinstance(value, time) or value.utcoffset() is not None:
        raise ValueError(f"{name} must be a timezone-naive datetime.time in ET")
    return value


def _empty_plans() -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "trading_day": pd.Series(dtype="datetime64[ns]"),
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
            "min_risk_pips": pd.Series(dtype="float64"),
            "max_risk_pips": pd.Series(dtype="float64"),
            "risk_pips": pd.Series(dtype="float64"),
            "zone_base_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "zone_known_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "zone_low": pd.Series(dtype="float64"),
            "zone_high": pd.Series(dtype="float64"),
            "touch_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "confirm_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "htf_state": pd.Series(dtype="object"),
        },
        columns=PLAN_COLUMNS,
    )
    return result


def _aligned_series(
    name: str,
    series: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a UTC DatetimeIndex")
    if series.index.tz is None or str(series.index.tz).upper() != "UTC":
        raise ValueError(f"{name} index must be timezone-aware UTC")
    if not series.index.as_unit("ns").equals(index):
        raise ValueError(f"{name} index must equal df.index")
    return series


def _zone_position(
    value: object,
    name: str,
    positions: dict[int, int],
    *,
    optional: bool = False,
) -> int | None:
    if pd.isna(value):
        if optional:
            return None
        raise ValueError(f"zone {name} must be a timezone-aware UTC timestamp")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"zone {name} must be a timezone-aware UTC timestamp") from error
    if timestamp.tz is None or str(timestamp.tz).upper() != "UTC":
        raise ValueError(f"zone {name} must be a timezone-aware UTC timestamp")
    return positions.get(timestamp.tz_convert("UTC").as_unit("ns").value)


def _build_plans(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty_plans()
    result = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    result["trading_day"] = pd.to_datetime(result["trading_day"]).astype(
        "datetime64[ns]"
    )
    for column in UTC_COLUMNS:
        result[column] = pd.to_datetime(result[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    for column in FLOAT_COLUMNS:
        result[column] = result[column].astype("float64")
    result["valid_candles"] = result["valid_candles"].astype("int64")
    for column in ("plan_id", "side", "order_type", "htf_state"):
        result[column] = result[column].astype("object")
    return result.loc[:, list(PLAN_COLUMNS)].reset_index(drop=True)


def find_candidate_b_plans(
    df: pd.DataFrame,
    zones: pd.DataFrame | None = None,
    k: pd.Series | None = None,
    state: pd.Series | None = None,
    swing_left: int = 3,
    swing_right: int = 3,
    stop_buffer_pips: float = 2.0,
    min_risk_pips: float = 5.0,
    max_risk_pips: float = 30.0,
    target_r: float = 2.0,
    confirm_candles: int = 6,
    overbought: float = 80.0,
    oversold: float = 20.0,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.2,
    pip_size: float = 0.0001,
    *,
    window_start_et: time = time(9, 30),
    entry_deadline_et: time = time(11, 45),
    time_exit_et: time = time(15, 45),
) -> pd.DataFrame:
    """Find at most one Candidate B market-entry plan per ET calendar date.

    Only the first eligible touch and first stochastic crossing are considered.
    Internally computed 4-hour states are aligned after each candle's close;
    when needed, a constant volume column is added only to a private copy for
    the shared OHLCV resampler.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    missing = [column for column in ("open", "high", "low", "close") if column not in df]
    if missing:
        raise ValueError(f"df is missing required columns: {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")

    swing_left = _positive_integer("swing_left", swing_left)
    swing_right = _positive_integer("swing_right", swing_right)
    confirm_candles = _positive_integer("confirm_candles", confirm_candles)
    stop_buffer_pips = _finite_number(
        "stop_buffer_pips", stop_buffer_pips, minimum=0.0
    )
    min_risk_pips = _finite_number("min_risk_pips", min_risk_pips, minimum=0.0)
    max_risk_pips = _finite_number("max_risk_pips", max_risk_pips, minimum=0.0)
    target_r = _finite_number("target_r", target_r, minimum=0.0, strict=True)
    overbought = _finite_number("overbought", overbought, minimum=0.0)
    oversold = _finite_number("oversold", oversold, minimum=0.0)
    spread_pips = _finite_number("spread_pips", spread_pips, minimum=0.0)
    slippage_pips = _finite_number("slippage_pips", slippage_pips, minimum=0.0)
    pip_size = _finite_number("pip_size", pip_size, minimum=0.0, strict=True)
    if max_risk_pips < min_risk_pips:
        raise ValueError("max_risk_pips must be greater than or equal to min_risk_pips")
    if not 0.0 < oversold < overbought < 100.0:
        raise ValueError("levels must satisfy 0 < oversold < overbought < 100")

    window_start_et = _validate_et_time("window_start_et", window_start_et)
    entry_deadline_et = _validate_et_time("entry_deadline_et", entry_deadline_et)
    time_exit_et = _validate_et_time("time_exit_et", time_exit_et)
    if not window_start_et <= entry_deadline_et <= time_exit_et:
        raise ValueError(
            "times must satisfy window_start_et <= entry_deadline_et <= time_exit_et"
        )

    index = df.index.as_unit("ns")
    try:
        opens, highs, lows, closes = (
            df[column].to_numpy(dtype=np.float64, copy=True)
            for column in ("open", "high", "low", "close")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("df OHLC columns must contain numeric prices") from error
    if not all(
        np.isfinite(values).all() for values in (opens, highs, lows, closes)
    ):
        raise ValueError("df OHLC prices must be finite")

    if zones is None:
        zones = detect_zones(df)
    if not isinstance(zones, pd.DataFrame):
        raise ValueError("zones must be a pandas DataFrame")
    missing_zones = [column for column in ZONE_COLUMNS if column not in zones.columns]
    if missing_zones:
        raise ValueError(
            f"zones is missing required columns: {', '.join(missing_zones)}"
        )

    if k is None:
        k = stochastic(df)["k"]
    else:
        k = _aligned_series("k", k, index)
    if not isinstance(k, pd.Series):
        raise ValueError("k must be a pandas Series")
    if not isinstance(k.index, pd.DatetimeIndex) or k.index.tz is None:
        raise ValueError("k must have a timezone-aware UTC DatetimeIndex")
    if str(k.index.tz).upper() != "UTC" or not k.index.as_unit("ns").equals(index):
        raise ValueError("k index must equal df.index")
    try:
        k_values = k.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("k values must be numeric or NaN") from error
    if np.isinf(k_values).any():
        raise ValueError("k values must be finite or NaN")

    if state is None:
        htf_source = df.copy()
        htf_source.index = index
        if "volume" not in htf_source.columns:
            htf_source["volume"] = 1.0
        four_hour = resample_ohlcv_et(htf_source, "4h")
        trend = htf_trend_state(
            four_hour,
            left=swing_left,
            right=swing_right,
        )
        htf_frame = four_hour.copy()
        htf_frame["state"] = trend["state"]
        state = align_htf_to_ltf(
            index,
            htf_frame,
            columns=["state"],
        )["state"]
    else:
        state = _aligned_series("state", state, index)
    if not isinstance(state, pd.Series):
        raise ValueError("state must be a pandas Series")
    if not isinstance(state.index, pd.DatetimeIndex) or state.index.tz is None:
        raise ValueError("state must have a timezone-aware UTC DatetimeIndex")
    if (
        str(state.index.tz).upper() != "UTC"
        or not state.index.as_unit("ns").equals(index)
    ):
        raise ValueError("state index must equal df.index")
    state_values = state.to_numpy(dtype=object, copy=True)
    if any(
        not pd.isna(value) and value not in ("up", "down", "none")
        for value in state_values
    ):
        raise ValueError("state values must be 'up', 'down', 'none', or NaN")

    count = len(index)
    if count == 0 or zones.empty:
        return _empty_plans()

    positions = {timestamp: position for position, timestamp in enumerate(index.asi8)}
    local_index = index.tz_convert(ET)
    try:
        zone_lows = zones["zone_low"].to_numpy(dtype=np.float64, copy=False)
        zone_highs = zones["zone_high"].to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("zone_low and zone_high must be numeric") from error
    if not np.isfinite(zone_lows).all() or not np.isfinite(zone_highs).all():
        raise ValueError("zone_low and zone_high must be finite")

    spread = spread_pips * pip_size
    slippage = slippage_pips * pip_size
    chosen: dict[date, tuple[tuple[int, int, int], dict[str, object]]] = {}

    for row_number, zone in enumerate(zones.to_dict(orient="records")):
        direction = zone["direction"]
        if direction not in ("supply", "demand"):
            raise ValueError("zone direction must be 'supply' or 'demand'")
        base_pos = _zone_position(zone["base_time"], "base_time", positions)
        known_pos = _zone_position(zone["known_at"], "known_at", positions)
        closed_pos = _zone_position(
            zone["closed_through_at"],
            "closed_through_at",
            positions,
            optional=True,
        )
        expiry_pos = _zone_position(
            zone["expires_at"],
            "expires_at",
            positions,
            optional=True,
        )
        if base_pos is None or known_pos is None:
            continue
        if pd.isna(zone["expires_at"]):
            expiry_pos = count
        if (
            (not pd.isna(zone["closed_through_at"]) and closed_pos is None)
            or (not pd.isna(zone["expires_at"]) and expiry_pos is None)
        ):
            continue
        assert expiry_pos is not None

        zone_low = float(zone_lows[row_number])
        zone_high = float(zone_highs[row_number])
        touch_start = known_pos + 1
        touch_stop = min(expiry_pos, count)
        if touch_start >= touch_stop:
            continue
        if direction == "supply":
            touches = np.flatnonzero(
                highs[touch_start:touch_stop] >= zone_low - 1e-9
            )
        else:
            touches = np.flatnonzero(
                lows[touch_start:touch_stop] <= zone_high + 1e-9
            )
        if not touches.size:
            continue
        touch_pos = touch_start + int(touches[0])

        last_confirm = min(touch_pos + confirm_candles, expiry_pos - 1, count - 1)
        confirm_pos: int | None = None
        for candidate_pos in range(touch_pos, last_confirm + 1):
            if candidate_pos < 1:
                continue
            previous_k = k_values[candidate_pos - 1]
            current_k = k_values[candidate_pos]
            if np.isnan(previous_k) or np.isnan(current_k):
                continue
            crossed = (
                previous_k >= overbought and current_k < overbought
                if direction == "supply"
                else previous_k <= oversold and current_k > oversold
            )
            if crossed:
                confirm_pos = candidate_pos
                break
        if confirm_pos is None:
            continue
        if closed_pos is not None and closed_pos <= confirm_pos:
            continue

        expected_state = "down" if direction == "supply" else "up"
        if state_values[confirm_pos] != expected_state:
            continue

        entry_pos = confirm_pos + 1
        if entry_pos >= count:
            continue
        entry_time = local_index[entry_pos].time()
        if not window_start_et <= entry_time <= entry_deadline_et:
            continue

        side = "short" if direction == "supply" else "long"
        fill_price = (
            opens[entry_pos] - slippage
            if side == "short"
            else opens[entry_pos] + spread + slippage
        )
        stop_price = (
            zone_high + stop_buffer_pips * pip_size
            if side == "short"
            else zone_low - stop_buffer_pips * pip_size
        )
        risk = stop_price - fill_price if side == "short" else fill_price - stop_price
        risk_pips = risk / pip_size
        if (
            risk <= 0
            or risk_pips < min_risk_pips - 1e-9
            or risk_pips > max_risk_pips + 1e-9
        ):
            continue

        day = local_index[entry_pos].date()
        key = (entry_pos, -known_pos, -base_pos)
        exit_local = datetime.combine(day, time_exit_et, ET)
        exit_utc = pd.Timestamp(exit_local.astimezone(timezone.utc)).as_unit("ns")
        candidate = {
            "trading_day": pd.Timestamp(day),
            "plan_id": day.isoformat(),
            "side": side,
            "order_type": "market",
            "start_time": index[entry_pos],
            "entry_price": np.nan,
            "stop_price": stop_price,
            "target_r": target_r,
            "target_price": np.nan,
            "valid_candles": 1,
            "time_exit_at": exit_utc,
            "min_risk_pips": min_risk_pips,
            "max_risk_pips": max_risk_pips,
            "risk_pips": risk_pips,
            "zone_base_time": index[base_pos],
            "zone_known_at": index[known_pos],
            "zone_low": zone_low,
            "zone_high": zone_high,
            "touch_time": index[touch_pos],
            "confirm_time": index[confirm_pos],
            "htf_state": expected_state,
        }
        if day not in chosen or key < chosen[day][0]:
            chosen[day] = (key, candidate)

    rows = [
        selection[1]
        for selection in sorted(chosen.values(), key=lambda item: item[0][0])
    ]
    return _build_plans(rows)
