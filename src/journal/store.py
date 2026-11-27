"""SQLite storage for personal paper and demo trade records."""

from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.resample_et import trading_day


DEFAULT_JOURNAL_PATH = "data/journal.sqlite3"
ALLOWED_MODES = ("paper", "demo")

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    pair TEXT NOT NULL DEFAULT 'EURUSD',
    side TEXT NOT NULL,
    mode TEXT NOT NULL,
    strategy TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL,
    size_units INTEGER NOT NULL,
    planned_risk_amount REAL NOT NULL,
    reason TEXT NOT NULL,
    exit_price REAL,
    realized_pnl REAL,
    r_multiple REAL,
    notes TEXT NOT NULL DEFAULT ''
)
"""

_SELECT_TRADES = "SELECT * FROM trades ORDER BY opened_at, id"
_TIMESTAMP_COLUMNS = ("opened_at", "closed_at")


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _aware_utc_datetime(name: str, value: Any) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _trade_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for column in _TIMESTAMP_COLUMNS:
        value = result[column]
        if value is not None:
            result[column] = datetime.fromisoformat(value).astimezone(timezone.utc)
    return result


def _trading_day_labels(timestamps: list[datetime]) -> pd.DatetimeIndex:
    return trading_day(pd.DatetimeIndex(timestamps))


def open_journal(
    path: str | os.PathLike[str] = DEFAULT_JOURNAL_PATH,
) -> sqlite3.Connection:
    """Open or initialize a journal database."""
    database_path = os.fspath(path)
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TRADES_TABLE)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    return conn


def add_trade(
    conn: sqlite3.Connection,
    *,
    opened_at: datetime,
    side: str,
    entry_price: float,
    stop_price: float,
    size_units: int,
    planned_risk_amount: float,
    mode: str,
    reason: str,
    strategy: str = "manual",
    target_price: float | None = None,
) -> int:
    """Insert and commit a new open trade."""
    opened_utc = _aware_utc_datetime("opened_at", opened_at)
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {ALLOWED_MODES}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must not be empty")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must not be empty")

    entry = _finite_number("entry_price", entry_price)
    stop = _finite_number("stop_price", stop_price)
    target = (
        None
        if target_price is None
        else _finite_number("target_price", target_price)
    )
    planned_risk = _finite_number(
        "planned_risk_amount",
        planned_risk_amount,
    )
    if entry <= 0:
        raise ValueError("entry_price must be greater than zero")
    if stop <= 0:
        raise ValueError("stop_price must be greater than zero")
    if side == "long" and stop >= entry:
        raise ValueError("long stop_price must be below entry_price")
    if side == "short" and stop <= entry:
        raise ValueError("short stop_price must be above entry_price")
    if target is not None:
        if side == "long" and target <= entry:
            raise ValueError("long target_price must be above entry_price")
        if side == "short" and target >= entry:
            raise ValueError("short target_price must be below entry_price")
    if isinstance(size_units, bool) or not isinstance(size_units, int):
        raise ValueError("size_units must be a positive int")
    if size_units <= 0:
        raise ValueError("size_units must be a positive int")
    if planned_risk <= 0:
        raise ValueError("planned_risk_amount must be greater than zero")

    cursor = conn.execute(
        """
        INSERT INTO trades (
            opened_at,
            side,
            mode,
            strategy,
            entry_price,
            stop_price,
            target_price,
            size_units,
            planned_risk_amount,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _timestamp_text(opened_utc),
            side,
            mode,
            strategy.strip(),
            entry,
            stop,
            target,
            size_units,
            planned_risk,
            reason.strip(),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def close_trade(
    conn: sqlite3.Connection,
    trade_id: int,
    *,
    closed_at: datetime,
    exit_price: float,
    realized_pnl: float,
    notes: str = "",
) -> None:
    """Close an open trade and commit its realized result."""
    row = conn.execute(
        "SELECT opened_at, closed_at, planned_risk_amount "
        "FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown trade id: {trade_id}")
    if row["closed_at"] is not None:
        raise ValueError(f"trade {trade_id} is already closed")

    closed_utc = _aware_utc_datetime("closed_at", closed_at)
    opened_utc = datetime.fromisoformat(row["opened_at"]).astimezone(
        timezone.utc
    )
    if closed_utc < opened_utc:
        raise ValueError("closed_at must not be earlier than opened_at")
    exit_value = _finite_number("exit_price", exit_price)
    pnl_value = _finite_number("realized_pnl", realized_pnl)
    if exit_value <= 0:
        raise ValueError("exit_price must be greater than zero")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")

    conn.execute(
        """
        UPDATE trades
        SET closed_at = ?,
            exit_price = ?,
            realized_pnl = ?,
            r_multiple = ?,
            notes = ?
        WHERE id = ?
        """,
        (
            _timestamp_text(closed_utc),
            exit_value,
            pnl_value,
            pnl_value / float(row["planned_risk_amount"]),
            notes.strip(),
            trade_id,
        ),
    )
    conn.commit()


def get_trade(
    conn: sqlite3.Connection,
    trade_id: int,
) -> dict[str, Any] | None:
    """Return one trade, with timestamps as timezone-aware UTC datetimes."""
    row = conn.execute(
        "SELECT * FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    return None if row is None else _trade_dict(row)


def list_trades(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return trades ordered by open time and id."""
    if status is None:
        rows = conn.execute(_SELECT_TRADES).fetchall()
    elif status == "open":
        rows = conn.execute(
            "SELECT * FROM trades WHERE closed_at IS NULL "
            "ORDER BY opened_at, id"
        ).fetchall()
    elif status == "closed":
        rows = conn.execute(
            "SELECT * FROM trades WHERE closed_at IS NOT NULL "
            "ORDER BY opened_at, id"
        ).fetchall()
    else:
        raise ValueError("status must be None, 'open', or 'closed'")
    return [_trade_dict(row) for row in rows]


def trading_day_summary(
    conn: sqlite3.Connection,
    now: datetime,
) -> dict[str, Any]:
    """Summarize trade counts and realized P&L as of a timezone-aware time."""
    now_utc = _aware_utc_datetime("now", now)
    trades = list_trades(conn)
    event_times = [now_utc]
    for trade in trades:
        event_times.append(trade["opened_at"])
        if trade["closed_at"] is not None:
            event_times.append(trade["closed_at"])
    labels = _trading_day_labels(event_times)
    current_day = labels[0]

    trades_taken_today = 0
    realized_values: list[float] = []
    open_positions = 0
    label_position = 1
    for trade in trades:
        opened_at = trade["opened_at"]
        opened_day = labels[label_position]
        label_position += 1
        closed_at = trade["closed_at"]
        closed_day = None
        if closed_at is not None:
            closed_day = labels[label_position]
            label_position += 1

        if opened_at <= now_utc and opened_day == current_day:
            trades_taken_today += 1
        if (
            closed_at is not None
            and closed_at <= now_utc
            and closed_day == current_day
        ):
            realized_values.append(float(trade["realized_pnl"]))
        if opened_at <= now_utc and (
            closed_at is None or closed_at > now_utc
        ):
            open_positions += 1

    return {
        "trading_day": current_day,
        "trades_taken_today": trades_taken_today,
        "realized_pnl_today": float(sum(realized_values, start=0.0)),
        "open_positions": open_positions,
    }
