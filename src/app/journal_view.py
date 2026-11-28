"""Pure helpers for the paper and demo journal screen."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from decimal import Decimal
from numbers import Real
from zoneinfo import ZoneInfo

from src.journal.store import DEFAULT_JOURNAL_PATH


JOURNAL_PATH_ENV = "FOREX_JOURNAL_PATH"
_MANILA = ZoneInfo("Asia/Manila")

_REASONS = {
    "max_trades_reached": "Maximum trades for today reached (2).",
    "daily_loss_limit_reached": (
        "Daily loss limit reached (2% of the start-of-day balance)."
    ),
    "position_already_open": "A position is already open (maximum 1).",
    "size_too_small": (
        "The account is too small to take this trade within the risk limit."
    ),
}


def journal_path() -> str:
    """Return the configured journal path or the project default."""
    return os.environ.get(JOURNAL_PATH_ENV) or DEFAULT_JOURNAL_PATH


def get_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def describe_reason(reason: str) -> str:
    """Return the user-facing description for a known blocking reason."""
    try:
        return _REASONS[reason]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown blocking reason: {reason!r}") from exc


def format_local_time(dt: datetime) -> str:
    """Format an aware time in Manila and UTC."""
    if (
        not isinstance(dt, datetime)
        or dt.tzinfo is None
        or dt.utcoffset() is None
    ):
        raise ValueError("dt must be a timezone-aware datetime")
    utc = dt.astimezone(timezone.utc)
    local = utc.astimezone(_MANILA)
    return f"{local:%Y-%m-%d %H:%M} Manila ({utc:%H:%M} UTC)"


def _finite_price(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        price = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return price


def gross_pnl_estimate(
    side: str,
    entry_price: float,
    exit_price: float,
    size_units: int,
) -> float:
    """Estimate gross EUR/USD P&L without spread or slippage costs."""
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    entry = _finite_price("entry_price", entry_price)
    exit_value = _finite_price("exit_price", exit_price)
    if isinstance(size_units, bool) or not isinstance(size_units, int):
        raise ValueError("size_units must be a positive int")
    if size_units <= 0:
        raise ValueError("size_units must be a positive int")

    price_change = exit_value - entry if side == "long" else entry - exit_value
    return float(price_change * size_units)


def trade_table_rows(trades: list[dict]) -> list[dict]:
    """Convert journal trades to ordered, presentation-ready rows."""
    rows = []
    for trade in trades:
        is_open = trade["closed_at"] is None
        rows.append(
            {
                "id": trade["id"],
                "opened": format_local_time(trade["opened_at"]),
                "side": trade["side"],
                "mode": trade["mode"],
                "units": trade["size_units"],
                "planned_risk_usd": round(
                    float(trade["planned_risk_amount"]),
                    2,
                ),
                "status": "open" if is_open else "closed",
                "result_usd": (
                    None if is_open else round(float(trade["realized_pnl"]), 2)
                ),
                "r_multiple": (
                    None if is_open else round(float(trade["r_multiple"]), 2)
                ),
            }
        )
    return rows
