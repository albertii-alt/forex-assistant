"""Risk-aware planning and checked trade saving."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from numbers import Real
from typing import Any

import pandas as pd

from src.data.resample_et import trading_day
from src.journal.store import (
    add_trade,
    list_trades,
    trading_day_summary,
)
from src.risk.limits import evaluate_daily_limits
from src.risk.sizing import position_size


def _finite_starting_balance(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError("starting_balance must be a finite real number")
    try:
        balance = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("starting_balance must be a finite real number") from exc
    if not math.isfinite(balance) or balance <= 0:
        raise ValueError("starting_balance must be finite and greater than zero")
    return balance


def _aware_utc(value: Any, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def account_balances(
    conn: sqlite3.Connection,
    starting_balance: float,
    now: datetime,
) -> dict[str, float]:
    """Return current and start-of-trading-day account balances."""
    opening_balance = _finite_starting_balance(starting_balance)
    now_utc = _aware_utc(now, "now")
    realized_trades = [
        trade
        for trade in list_trades(conn)
        if trade["closed_at"] is not None and trade["closed_at"] <= now_utc
    ]
    labels = trading_day(
        pd.DatetimeIndex(
            [now_utc] + [trade["closed_at"] for trade in realized_trades]
        )
    )
    current_day = labels[0]
    current_balance = opening_balance
    start_of_day_balance = opening_balance

    for position, trade in enumerate(realized_trades, start=1):
        pnl = float(trade["realized_pnl"])
        current_balance += pnl
        if labels[position] < current_day:
            start_of_day_balance += pnl

    return {
        "current_balance": float(current_balance),
        "start_of_day_balance": float(start_of_day_balance),
    }


def plan_trade(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    starting_balance: float,
    risk_pct: float,
    side: str,
    entry_price: float,
    stop_price: float,
) -> dict[str, Any]:
    """Evaluate balances, daily limits, and cost-aware position sizing."""
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    balances = account_balances(conn, starting_balance, now)
    summary = trading_day_summary(conn, now)
    limits = evaluate_daily_limits(
        balances["start_of_day_balance"],
        summary["realized_pnl_today"],
        summary["trades_taken_today"],
        summary["open_positions"],
    )
    sizing = position_size(
        balances["current_balance"],
        risk_pct,
        entry_price,
        stop_price,
    )

    if side == "long" and stop_price >= entry_price:
        raise ValueError("long stop_price must be below entry_price")
    if side == "short" and stop_price <= entry_price:
        raise ValueError("short stop_price must be above entry_price")

    blocking_reasons = list(limits["reasons"])
    if sizing["status"] == "too_small":
        blocking_reasons.append("size_too_small")
    can_save = bool(limits["allowed"] and sizing["status"] == "ok")
    return {
        "balances": balances,
        "summary": summary,
        "limits": limits,
        "sizing": sizing,
        "blocking_reasons": blocking_reasons,
        "can_save": can_save,
    }


def open_planned_trade(
    conn: sqlite3.Connection,
    *,
    opened_at: datetime,
    starting_balance: float,
    risk_pct: float,
    side: str,
    entry_price: float,
    stop_price: float,
    mode: str,
    reason: str,
    strategy: str = "manual",
    target_price: float | None = None,
) -> dict[str, Any]:
    """Recompute a plan and save it only when every check passes."""
    plan = plan_trade(
        conn,
        now=opened_at,
        starting_balance=starting_balance,
        risk_pct=risk_pct,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
    )
    if not plan["can_save"]:
        reasons = ", ".join(plan["blocking_reasons"])
        raise ValueError(f"trade cannot be saved: {reasons}")

    trade_id = add_trade(
        conn,
        opened_at=opened_at,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        size_units=plan["sizing"]["units"],
        planned_risk_amount=plan["sizing"]["actual_risk_amount"],
        mode=mode,
        reason=reason,
        strategy=strategy,
        target_price=target_price,
    )
    return {"trade_id": trade_id, "plan": plan}
