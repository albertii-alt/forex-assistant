"""Candidate C 4-hour Donchian breakout plan discovery and price-only counts."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.data.resample_et import resample_ohlcv_et
from src.indicators.basic import atr


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


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
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

    result = df.copy()
    result.index = result.index.as_unit("ns")
    if "volume" not in result:
        result["volume"] = np.ones(len(result), dtype=np.float64)
    return result


def _validate_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(value)


def find_candidate_c_plans(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Find every fixed Candidate C 4-hour Donchian breakout plan."""
    data = _validate_df(df)
    entry_period = _validate_integer("entry_period", entry_period)
    exit_period = _validate_integer("exit_period", exit_period)
    atr_period = _validate_integer("atr_period", atr_period)
    if (
        isinstance(atr_mult, bool)
        or not isinstance(atr_mult, Real)
        or not np.isfinite(float(atr_mult))
        or float(atr_mult) <= 0
    ):
        raise ValueError("atr_mult must be finite and greater than 0")
    atr_mult = float(atr_mult)

    if data.empty:
        return _empty_plans()

    h4 = resample_ohlcv_et(data, "4h")
    h4["close_time"] = pd.to_datetime(h4["close_time"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    closed = h4["close_time"] <= data.index[-1].as_unit("ns") + pd.Timedelta(minutes=15)
    if len(h4) <= entry_period:
        return _empty_plans()

    atr_values = atr(h4.loc[:, ["high", "low", "close"]], atr_period)
    atr_values = atr_values.to_numpy(dtype=np.float64)
    highs = h4["high"].to_numpy(dtype=np.float64)
    lows = h4["low"].to_numpy(dtype=np.float64)
    closes = h4["close"].to_numpy(dtype=np.float64)
    opens = h4["open"].to_numpy(dtype=np.float64)
    h4_times = h4.index.as_unit("ns")
    ltf_times = data.index.as_unit("ns")
    ltf_values = ltf_times.asi8
    rows: list[dict[str, object]] = []

    for c in range(entry_period, len(h4)):
        if not bool(closed.iloc[c]) or not np.isfinite(atr_values[c]):
            continue

        long_level = float(np.max(highs[c - entry_period:c]))
        short_level = float(np.min(lows[c - entry_period:c]))
        side: str | None = None
        channel_level = np.nan
        if closes[c] > long_level:
            side = "long"
            channel_level = long_level
        elif closes[c] < short_level:
            side = "short"
            channel_level = short_level
        if side is None or c + 1 >= len(h4):
            continue

        start_time = h4_times[c + 1]
        start_pos = int(np.searchsorted(ltf_values, start_time.value, side="left"))
        if start_pos >= len(data) or start_pos == len(data) - 1:
            continue

        open4h = float(opens[c + 1])
        n_c = float(atr_values[c])
        stop = open4h - atr_mult * n_c if side == "long" else open4h + atr_mult * n_c

        exit_signal_time = pd.NaT
        time_exit_at = data.index[-1].as_unit("ns") + pd.Timedelta(days=1)
        for e in range(c + 1, len(h4)):
            if not bool(closed.iloc[e]) or e < exit_period:
                continue
            if side == "long":
                exit_level = float(np.min(lows[e - exit_period:e]))
                exit_hit = closes[e] < exit_level
            else:
                exit_level = float(np.max(highs[e - exit_period:e]))
                exit_hit = closes[e] > exit_level
            if exit_hit:
                exit_signal_time = h4_times[e]
                if e + 1 < len(h4):
                    exit_start = h4_times[e + 1]
                    exit_pos = int(
                        np.searchsorted(ltf_values, exit_start.value, side="left")
                    )
                    if exit_pos < len(data):
                        time_exit_at = ltf_times[exit_pos]
                break

        start_timestamp = ltf_times[start_pos]
        plan_id = start_timestamp.isoformat()
        rows.append(
            {
                "plan_id": plan_id,
                "side": side,
                "order_type": "market",
                "start_time": start_timestamp,
                "entry_price": np.nan,
                "stop_price": float(stop),
                "target_r": np.nan,
                "target_price": np.nan,
                "valid_candles": 1,
                "time_exit_at": time_exit_at,
                "no_target": True,
                "exit_mode": "time_open",
                "signal_time": h4_times[c],
                "atr_n": n_c,
                "channel_level": float(channel_level),
                "exit_signal_time": exit_signal_time,
            }
        )

    if not rows:
        return _empty_plans()
    result = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    for column in ("start_time", "time_exit_at", "signal_time", "exit_signal_time"):
        result[column] = pd.to_datetime(result[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    for column in ("entry_price", "stop_price", "target_r", "target_price", "atr_n", "channel_level"):
        result[column] = result[column].astype("float64")
    result["valid_candles"] = result["valid_candles"].astype("int64")
    result["no_target"] = result["no_target"].astype("bool")
    for column in ("plan_id", "side", "order_type", "exit_mode"):
        result[column] = result[column].astype("object")
    return result.sort_values("start_time", kind="stable").reset_index(drop=True)


def count_candidate_c_entries(
    df: pd.DataFrame,
    plans: pd.DataFrame,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.2,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Count Candidate C entries and exits using only candle prices."""
    data = _validate_df(df)
    spread = float(spread_pips) * float(pip_size)
    slippage = float(slippage_pips) * float(pip_size)
    if data.empty:
        return pd.DataFrame(
            {
                "plan_id": pd.Series(dtype="object"),
                "side": pd.Series(dtype="object"),
                "fill_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "exit_reason": pd.Series(dtype="object"),
            }
        )

    times = data.index.as_unit("ns")
    values = times.asi8
    opens = data["open"].to_numpy(dtype=np.float64)
    highs = data["high"].to_numpy(dtype=np.float64)
    lows = data["low"].to_numpy(dtype=np.float64)
    prepared: list[tuple[object, str, int, float, pd.Timestamp]] = []
    for row in plans.to_dict(orient="records"):
        start = pd.Timestamp(row["start_time"]).tz_convert("UTC").as_unit("ns")
        exit_at = pd.Timestamp(row["time_exit_at"]).tz_convert("UTC").as_unit("ns")
        start_pos = int(np.searchsorted(values, start.value, side="left"))
        prepared.append((row["plan_id"], str(row["side"]), start_pos, float(row["stop_price"]), exit_at))
    prepared.sort(key=lambda item: item[2])

    busy_until = -1
    entries: list[dict[str, object]] = []
    for plan_id, side, start_pos, stop, exit_at in prepared:
        if start_pos >= len(data) or start_pos <= busy_until:
            continue
        if side == "long":
            fill_price = float(opens[start_pos]) + spread + slippage
        else:
            fill_price = float(opens[start_pos]) - slippage
        risk = fill_price - stop if side == "long" else stop - fill_price
        if risk <= 0:
            continue

        exit_pos = int(np.searchsorted(values, exit_at.value, side="right") - 1)
        exit_is_after_data = exit_at > times[-1]
        exit_reason = ""
        actual_exit = -1
        for position in range(start_pos, max(start_pos, exit_pos) + 1):
            if not exit_is_after_data and position == exit_pos:
                actual_exit = position
                exit_reason = "time"
                break
            if side == "long":
                stopped = lows[position] <= stop + 1e-9
            else:
                stopped = highs[position] + spread >= stop - 1e-9
            if stopped:
                actual_exit = position
                exit_reason = "stop"
                break
            if position == exit_pos:
                actual_exit = position
                exit_reason = "data_end"
                break
        if actual_exit < 0:
            continue
        busy_until = actual_exit
        entries.append(
            {
                "plan_id": plan_id,
                "side": side,
                "fill_time": times[start_pos],
                "exit_time": times[actual_exit],
                "exit_reason": exit_reason,
            }
        )

    if not entries:
        return pd.DataFrame(
            {
                "plan_id": pd.Series(dtype="object"),
                "side": pd.Series(dtype="object"),
                "fill_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "exit_reason": pd.Series(dtype="object"),
            }
        )
    result = pd.DataFrame(entries)
    for column in ("fill_time", "exit_time"):
        result[column] = pd.to_datetime(result[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    result["plan_id"] = result["plan_id"].astype("object")
    result["side"] = result["side"].astype("object")
    result["exit_reason"] = result["exit_reason"].astype("object")
    return result.reset_index(drop=True)
