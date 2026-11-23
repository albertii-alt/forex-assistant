"""Cost-aware simulation of precomputed 15-minute trade plans."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data.splits import HOLDOUT_START


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
)
OUTPUT_COLUMNS = (
    "plan_id",
    "side",
    "order_type",
    "status",
    "start_time",
    "fill_time",
    "entry_price",
    "stop_price",
    "target_price",
    "risk_pips",
    "exit_time",
    "exit_price",
    "exit_reason",
    "r_multiple",
    "pnl_pips",
    "bars_held",
)
PRICE_TOLERANCE = 1e-9


def _is_missing(value: object) -> bool:
    return bool(pd.isna(value))


def _utc_timestamp(value: object, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a timezone-aware UTC timestamp") from exc
    if pd.isna(timestamp) or timestamp.tz is None or str(timestamp.tz).upper() != "UTC":
        raise ValueError(f"{field} must be a timezone-aware UTC timestamp")
    return timestamp.tz_convert("UTC").as_unit("ns")


def _number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _empty_results() -> pd.DataFrame:
    result = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for column in ("start_time", "fill_time", "exit_time"):
        result[column] = pd.Series(dtype="datetime64[ns, UTC]")
    for column in (
        "entry_price",
        "stop_price",
        "target_price",
        "risk_pips",
        "exit_price",
        "r_multiple",
        "pnl_pips",
        "bars_held",
    ):
        result[column] = pd.Series(dtype="float64")
    return result.loc[:, list(OUTPUT_COLUMNS)]


def _plan_time(value: object, field: str, plan_id: object) -> pd.Timestamp:
    return _utc_timestamp(value, f"plan {plan_id!r} {field}")


def _prepare_plans(
    df: pd.DataFrame,
    plans: pd.DataFrame,
) -> list[dict[str, object]]:
    missing = [column for column in PLAN_COLUMNS if column not in plans.columns]
    if missing:
        raise ValueError(f"plans is missing required columns: {', '.join(missing)}")
    if plans["plan_id"].isna().any() or plans["plan_id"].duplicated().any():
        raise ValueError("plan_id values must be non-null and unique")

    timestamps = df.index.as_unit("ns")
    timestamp_values = timestamps.asi8
    prepared: list[dict[str, object]] = []
    for raw in plans.to_dict(orient="records"):
        plan_id = raw["plan_id"]
        side = raw["side"]
        order_type = raw["order_type"]
        if side not in ("long", "short"):
            raise ValueError(f"plan {plan_id!r} has unknown side {side!r}")
        if order_type not in ("market", "limit"):
            raise ValueError(f"plan {plan_id!r} has unknown order_type {order_type!r}")

        start_time = _plan_time(raw["start_time"], "start_time", plan_id)
        time_exit_at = _plan_time(raw["time_exit_at"], "time_exit_at", plan_id)
        start_pos = int(
            np.searchsorted(timestamp_values, start_time.value, side="left")
        )

        try:
            valid_raw = raw["valid_candles"]
            valid_number = float(valid_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"plan {plan_id!r} valid_candles must be an integer >= 1") from exc
        if (
            not math.isfinite(valid_number)
            or valid_number < 1
            or not valid_number.is_integer()
        ):
            raise ValueError(f"plan {plan_id!r} valid_candles must be an integer >= 1")
        valid_candles = int(valid_number)

        stop_price = _number(raw["stop_price"], f"plan {plan_id!r} stop_price")
        target_r_missing = _is_missing(raw["target_r"])
        target_price_missing = _is_missing(raw["target_price"])
        if target_r_missing == target_price_missing:
            raise ValueError(
                f"plan {plan_id!r} must specify exactly one of target_r or target_price"
            )
        target_r = (
            None
            if target_r_missing
            else _number(raw["target_r"], f"plan {plan_id!r} target_r")
        )
        if target_r is not None and target_r <= 0:
            raise ValueError(f"plan {plan_id!r} target_r must be greater than zero")
        explicit_target = (
            None
            if target_price_missing
            else _number(raw["target_price"], f"plan {plan_id!r} target_price")
        )

        entry_value = raw["entry_price"]
        entry_price = (
            None
            if _is_missing(entry_value)
            else _number(entry_value, f"plan {plan_id!r} entry_price")
        )
        if order_type == "limit" and entry_price is None:
            raise ValueError(f"plan {plan_id!r} limit entry_price must not be NaN")
        if order_type == "limit":
            if side == "long" and stop_price >= entry_price:
                raise ValueError(
                    f"plan {plan_id!r} long limit stop must be below entry_price"
                )
            if side == "short" and stop_price <= entry_price:
                raise ValueError(
                    f"plan {plan_id!r} short limit stop must be above entry_price"
                )
            if explicit_target is not None:
                if side == "long" and explicit_target <= entry_price:
                    raise ValueError(
                        f"plan {plan_id!r} long limit target_price must be above entry_price"
                    )
                if side == "short" and explicit_target >= entry_price:
                    raise ValueError(
                        f"plan {plan_id!r} short limit target_price must be below entry_price"
                    )

        min_risk_raw = raw.get("min_risk_pips", np.nan)
        max_risk_raw = raw.get("max_risk_pips", np.nan)
        min_risk = (
            None
            if _is_missing(min_risk_raw)
            else _number(min_risk_raw, f"plan {plan_id!r} min_risk_pips")
        )
        max_risk = (
            None
            if _is_missing(max_risk_raw)
            else _number(max_risk_raw, f"plan {plan_id!r} max_risk_pips")
        )

        exit_pos = -1
        if start_pos < len(timestamps):
            exit_pos = int(
                np.searchsorted(timestamp_values, time_exit_at.value, side="right") - 1
            )
            if exit_pos < start_pos:
                raise ValueError(
                    f"plan {plan_id!r} time_exit_at is earlier than its start candle"
                )

        prepared.append(
            {
                "plan_id": plan_id,
                "side": side,
                "order_type": order_type,
                "start_time": start_time,
                "start_pos": start_pos,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_r": target_r,
                "explicit_target": explicit_target,
                "valid_candles": valid_candles,
                "time_exit_at": time_exit_at,
                "exit_pos": exit_pos,
                "min_risk_pips": min_risk,
                "max_risk_pips": max_risk,
            }
        )

    ordered = sorted(prepared, key=lambda plan: int(plan["start_pos"]))
    previous_end = -1
    previous_id: object = None
    for plan in ordered:
        start_pos = int(plan["start_pos"])
        if start_pos >= len(timestamps):
            continue
        if start_pos <= previous_end:
            raise ValueError(
                f"plan {plan['plan_id']!r} has a live window overlapping "
                f"plan {previous_id!r}"
            )
        previous_end = start_pos + int(plan["valid_candles"]) - 1
        previous_id = plan["plan_id"]
    return ordered


def _empty_output_row(plan: Mapping[str, object], status: str) -> dict[str, object]:
    return {
        "plan_id": plan["plan_id"],
        "side": plan["side"],
        "order_type": plan["order_type"],
        "status": status,
        "start_time": plan["start_time"],
        "fill_time": pd.NaT,
        "entry_price": np.nan,
        "stop_price": np.nan,
        "target_price": np.nan,
        "risk_pips": np.nan,
        "exit_time": pd.NaT,
        "exit_price": np.nan,
        "exit_reason": np.nan,
        "r_multiple": np.nan,
        "pnl_pips": np.nan,
        "bars_held": np.nan,
    }


def _results_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty_results()
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in ("start_time", "fill_time", "exit_time"):
        result[column] = pd.to_datetime(result[column], utc=True)
    for column in (
        "entry_price",
        "stop_price",
        "target_price",
        "risk_pips",
        "exit_price",
        "r_multiple",
        "pnl_pips",
        "bars_held",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    return result.loc[:, list(OUTPUT_COLUMNS)]


def run_backtest(
    df: pd.DataFrame,
    plans: pd.DataFrame,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.2,
    pip_size: float = 0.0001,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Simulate independent, precomputed trade plans against BID OHLC candles."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")
    missing_prices = [
        column for column in ("open", "high", "low", "close") if column not in df.columns
    ]
    if missing_prices:
        raise ValueError(f"df is missing required columns: {', '.join(missing_prices)}")
    if len(df) and df.index.max() >= HOLDOUT_START and not allow_holdout:
        raise ValueError(
            "The holdout is sealed. Use development_set() for research; "
            "allow_holdout=True is reserved for a final one-time evaluation "
            "after a candidate passes on development data."
        )

    spread_pips = _number(spread_pips, "spread_pips")
    slippage_pips = _number(slippage_pips, "slippage_pips")
    pip_size = _number(pip_size, "pip_size")
    if spread_pips < 0:
        raise ValueError("spread_pips must be >= 0")
    if slippage_pips < 0:
        raise ValueError("slippage_pips must be >= 0")
    if pip_size <= 0:
        raise ValueError("pip_size must be > 0")

    spread = spread_pips * pip_size
    slippage = slippage_pips * pip_size
    prepared = _prepare_plans(df, plans)
    if not prepared:
        return _empty_results()

    source = df.loc[:, ["open", "high", "low", "close"]]
    try:
        opens, highs, lows, closes = (
            source[column].to_numpy(dtype="float64")
            for column in ("open", "high", "low", "close")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("df OHLC columns must contain numeric prices") from exc
    if not all(np.isfinite(values).all() for values in (opens, highs, lows, closes)):
        raise ValueError("df OHLC prices must be finite")

    index = df.index.as_unit("ns")
    rows: list[dict[str, object]] = []
    busy_until = -1

    for plan in prepared:
        plan_id = plan["plan_id"]
        side = str(plan["side"])
        order_type = str(plan["order_type"])
        start_pos = int(plan["start_pos"])
        if start_pos >= len(index):
            rows.append(_empty_output_row(plan, "no_data"))
            continue
        if start_pos <= busy_until:
            rows.append(_empty_output_row(plan, "skipped_busy"))
            continue

        exit_pos = int(plan["exit_pos"])
        exit_reason_at_end = (
            "data_end" if plan["time_exit_at"] > index[-1] else "time"
        )
        fill_pos = start_pos
        if order_type == "market":
            fill_price = (
                opens[start_pos] + spread + slippage
                if side == "long"
                else opens[start_pos] - slippage
            )
        else:
            fill_price = float(plan["entry_price"])
            fill_pos = -1
            final_live_pos = min(
                start_pos + int(plan["valid_candles"]) - 1,
                exit_pos,
            )
            for position in range(start_pos, final_live_pos + 1):
                if side == "long":
                    reached = lows[position] + spread <= fill_price + PRICE_TOLERANCE
                else:
                    reached = highs[position] >= fill_price - PRICE_TOLERANCE
                if reached:
                    fill_pos = position
                    break
            if fill_pos < 0:
                rows.append(_empty_output_row(plan, "unfilled_expired"))
                continue

        stop_price = float(plan["stop_price"])
        risk = fill_price - stop_price if side == "long" else stop_price - fill_price
        explicit_target = plan["explicit_target"]
        if side == "long":
            bad_target = explicit_target is not None and float(explicit_target) <= fill_price
        else:
            bad_target = explicit_target is not None and float(explicit_target) >= fill_price
        if risk <= 0 or bad_target:
            rows.append(_empty_output_row(plan, "skipped_bad_levels"))
            continue

        risk_pips = risk / pip_size
        minimum = plan["min_risk_pips"]
        maximum = plan["max_risk_pips"]
        if (
            (minimum is not None and risk_pips + PRICE_TOLERANCE < float(minimum))
            or (maximum is not None and risk_pips - PRICE_TOLERANCE > float(maximum))
        ):
            rows.append(_empty_output_row(plan, "skipped_risk_filter"))
            continue

        if plan["target_r"] is not None:
            distance = float(plan["target_r"]) * risk
            target_price = fill_price + distance if side == "long" else fill_price - distance
        else:
            target_price = float(explicit_target)

        exit_pos_actual = -1
        exit_price = np.nan
        exit_reason = ""
        for position in range(fill_pos, exit_pos + 1):
            if side == "long":
                stop_hit = lows[position] <= stop_price + PRICE_TOLERANCE
                target_hit = (
                    order_type != "limit" or position != fill_pos
                ) and highs[position] >= target_price - PRICE_TOLERANCE
            else:
                stop_hit = highs[position] + spread >= stop_price - PRICE_TOLERANCE
                target_hit = (
                    order_type != "limit" or position != fill_pos
                ) and lows[position] + spread <= target_price + PRICE_TOLERANCE

            if stop_hit:
                if side == "long":
                    exit_price = (
                        stop_price - slippage
                        if order_type == "limit" and position == fill_pos
                        else min(stop_price, opens[position]) - slippage
                    )
                else:
                    exit_price = (
                        stop_price + slippage
                        if order_type == "limit" and position == fill_pos
                        else max(stop_price, opens[position] + spread) + slippage
                    )
                exit_reason = "stop"
                exit_pos_actual = position
                break
            if target_hit:
                exit_price = target_price
                exit_reason = "target"
                exit_pos_actual = position
                break
            if position == exit_pos:
                if side == "long":
                    exit_price = closes[position] - slippage
                else:
                    exit_price = closes[position] + spread + slippage
                exit_reason = exit_reason_at_end
                exit_pos_actual = position
                break

        pnl_price = exit_price - fill_price if side == "long" else fill_price - exit_price
        rows.append(
            {
                "plan_id": plan_id,
                "side": side,
                "order_type": order_type,
                "status": "filled",
                "start_time": plan["start_time"],
                "fill_time": index[fill_pos],
                "entry_price": fill_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_pips": risk_pips,
                "exit_time": index[exit_pos_actual],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "r_multiple": pnl_price / risk,
                "pnl_pips": pnl_price / pip_size,
                "bars_held": exit_pos_actual - fill_pos,
            }
        )
        busy_until = exit_pos_actual

    return _results_frame(rows)
