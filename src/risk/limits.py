"""Daily trade and realized-loss limits."""

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


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def evaluate_daily_limits(
    start_of_day_balance: float,
    realized_pnl_today: float,
    trades_taken_today: int,
    open_positions: int,
    *,
    max_trades_per_day: int = 2,
    daily_loss_limit_pct: float = 2.0,
    max_open_positions: int = 1,
) -> dict[str, bool | list[str]]:
    """Return whether another trade is allowed and the applicable reasons."""
    start_of_day_balance = _finite_real(
        "start_of_day_balance", start_of_day_balance
    )
    realized_pnl_today = _finite_real("realized_pnl_today", realized_pnl_today)
    daily_loss_limit_pct = _finite_real(
        "daily_loss_limit_pct", daily_loss_limit_pct
    )
    trades_taken_today = _nonnegative_int(
        "trades_taken_today", trades_taken_today
    )
    open_positions = _nonnegative_int("open_positions", open_positions)
    max_trades_per_day = _positive_int(
        "max_trades_per_day", max_trades_per_day
    )
    max_open_positions = _positive_int(
        "max_open_positions", max_open_positions
    )

    if start_of_day_balance <= 0:
        raise ValueError("start_of_day_balance must be greater than zero")
    if daily_loss_limit_pct <= 0:
        raise ValueError("daily_loss_limit_pct must be greater than zero")

    reasons: list[str] = []
    if trades_taken_today >= max_trades_per_day:
        reasons.append("max_trades_reached")
    loss_limit = start_of_day_balance * daily_loss_limit_pct / 100
    if realized_pnl_today <= -loss_limit + 1e-9:
        reasons.append("daily_loss_limit_reached")
    if open_positions >= max_open_positions:
        reasons.append("position_already_open")
    return {"allowed": not reasons, "reasons": reasons}
