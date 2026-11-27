"""Position sizing helpers for EUR/USD-denominated accounts."""

from __future__ import annotations

import math
from decimal import Decimal
from numbers import Real
from typing import Any


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def position_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    *,
    cost_pips: float = 1.4,
    max_risk_pct: float = 2.0,
    pip_size: float = 0.0001,
    units_per_lot: float = 100000,
    lot_step_units: int = 1000,
) -> dict[str, str | int | float]:
    """Return a cost-aware EUR/USD position size, rounded down to its step."""
    account_balance = _finite_real("account_balance", account_balance)
    risk_pct = _finite_real("risk_pct", risk_pct)
    entry_price = _finite_real("entry_price", entry_price)
    stop_price = _finite_real("stop_price", stop_price)
    cost_pips = _finite_real("cost_pips", cost_pips)
    max_risk_pct = _finite_real("max_risk_pct", max_risk_pct)
    pip_size = _finite_real("pip_size", pip_size)
    units_per_lot = _finite_real("units_per_lot", units_per_lot)

    if isinstance(lot_step_units, bool) or not isinstance(lot_step_units, int):
        raise ValueError("lot_step_units must be a positive int")
    if lot_step_units <= 0:
        raise ValueError("lot_step_units must be a positive int")
    if account_balance <= 0:
        raise ValueError("account_balance must be greater than zero")
    if entry_price <= 0:
        raise ValueError("entry_price must be greater than zero")
    if stop_price <= 0:
        raise ValueError("stop_price must be greater than zero")
    if entry_price == stop_price:
        raise ValueError("entry_price and stop_price must differ")
    if risk_pct <= 0:
        raise ValueError("risk_pct must be greater than zero")
    if max_risk_pct <= 0:
        raise ValueError("max_risk_pct must be greater than zero")
    if risk_pct > max_risk_pct:
        raise ValueError("risk_pct must not exceed max_risk_pct")
    if cost_pips < 0:
        raise ValueError("cost_pips must not be negative")
    if pip_size <= 0:
        raise ValueError("pip_size must be greater than zero")
    if units_per_lot <= 0:
        raise ValueError("units_per_lot must be greater than zero")

    risk_amount = account_balance * risk_pct / 100
    stop_distance_pips = round(abs(entry_price - stop_price) / pip_size, 6)
    effective_risk_pips = round(stop_distance_pips + cost_pips, 6)
    risk_per_unit = effective_risk_pips * pip_size
    if not math.isfinite(risk_amount) or not math.isfinite(risk_per_unit):
        raise ValueError("calculated risk values must be finite")
    if risk_per_unit <= 0:
        raise ValueError("effective risk per unit must be greater than zero")

    raw_units = risk_amount / risk_per_unit
    if not math.isfinite(raw_units):
        raise ValueError("calculated raw units must be finite")
    units = math.floor(raw_units / lot_step_units + 1e-9) * lot_step_units

    if units < lot_step_units:
        return {
            "status": "too_small",
            "risk_amount": risk_amount,
            "stop_distance_pips": stop_distance_pips,
            "effective_risk_pips": effective_risk_pips,
            "units": 0,
            "lots": 0.0,
            "actual_risk_amount": 0.0,
            "actual_risk_pct": 0.0,
        }

    lots = float(units / units_per_lot)
    actual_risk_amount = units * effective_risk_pips * pip_size
    actual_risk_pct = actual_risk_amount / account_balance * 100
    return {
        "status": "ok",
        "risk_amount": risk_amount,
        "stop_distance_pips": stop_distance_pips,
        "effective_risk_pips": effective_risk_pips,
        "units": int(units),
        "lots": lots,
        "actual_risk_amount": actual_risk_amount,
        "actual_risk_pct": actual_risk_pct,
    }
