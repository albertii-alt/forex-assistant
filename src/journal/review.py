"""Pure statistics and text formatting for closed journal trades."""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from decimal import Decimal
from numbers import Real
from typing import Any


MIN_TRADES_FOR_RANGE = 30


def sample_size_note(n_closed: int) -> str:
    """Return the plain-language sample-size warning."""
    if isinstance(n_closed, bool) or not isinstance(n_closed, int):
        raise ValueError("n_closed must be a non-negative int")
    if n_closed < 0:
        raise ValueError("n_closed must be a non-negative int")
    if n_closed == 0:
        return "No closed trades yet."
    if n_closed < 30:
        return (
            f"Too few closed trades ({n_closed}) to conclude anything. "
            "A few wins or losses in a row are normal luck."
        )
    if n_closed < 100:
        return (
            f"Still a small sample ({n_closed} closed trades). "
            "Results can easily be luck; around 100 or more trades are needed "
            "before averages mean much."
        )
    return (
        f"Larger sample ({n_closed} closed trades), but past results do not "
        "guarantee future results, and paper or demo fills are easier than real ones."
    )


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


def review_trades(trades: list[dict]) -> dict:
    """Return aggregate review statistics for all journal trades."""
    if not isinstance(trades, list):
        raise ValueError("trades must be a list")

    required = ("id", "mode", "closed_at", "r_multiple", "realized_pnl")
    closed: list[dict] = []
    n_open = 0

    for index, trade in enumerate(trades):
        if not isinstance(trade, dict):
            raise ValueError(f"record {index}: trade must be a dict")
        missing = [key for key in required if key not in trade]
        if missing:
            raise ValueError(f"record {index}: missing key {missing[0]}")

        closed_at = trade["closed_at"]
        if closed_at is None:
            n_open += 1
            continue
        if (
            not isinstance(closed_at, datetime)
            or closed_at.tzinfo is None
            or closed_at.utcoffset() is None
        ):
            raise ValueError(
                f"record {index}: closed_at must be a timezone-aware datetime"
            )
        try:
            r_multiple = _finite_real(
                "r_multiple",
                trade["r_multiple"],
            )
            realized_pnl = _finite_real(
                "realized_pnl",
                trade["realized_pnl"],
            )
        except ValueError as exc:
            raise ValueError(f"record {index}: {exc}") from exc

        closed.append({
            "id": trade["id"],
            "mode": trade["mode"],
            "closed_at": closed_at,
            "r_multiple": r_multiple,
            "realized_pnl": realized_pnl,
        })

    closed.sort(key=lambda trade: (trade["closed_at"], trade["id"]))
    r_values = [trade["r_multiple"] for trade in closed]
    n_closed = len(closed)
    n_wins = sum(value > 0 for value in r_values)
    n_losses = sum(value < 0 for value in r_values)
    n_breakeven = sum(value == 0 for value in r_values)

    win_rate = n_wins / n_closed if n_closed else None
    total_r = sum(r_values, 0.0)
    avg_r = total_r / n_closed if n_closed else None

    if n_closed >= 2:
        sd = statistics.stdev(r_values)
        avg_r_stderr = sd / math.sqrt(n_closed)
        avg_r_low = avg_r - 2 * avg_r_stderr
        avg_r_high = avg_r + 2 * avg_r_stderr
    else:
        avg_r_stderr = None
        avg_r_low = None
        avg_r_high = None

    wins = [value for value in r_values if value > 0]
    losses = [value for value in r_values if value < 0]
    avg_win_r = sum(wins, 0.0) / len(wins) if wins else None
    avg_loss_r = sum(losses, 0.0) / len(losses) if losses else None

    positive_r = sum(wins, 0.0)
    negative_r = sum(losses, 0.0)
    if losses:
        profit_factor = positive_r / abs(negative_r)
    elif wins:
        profit_factor = math.inf
    else:
        profit_factor = None

    cumulative = 0.0
    peak = 0.0
    max_drawdown_r = 0.0
    current_losing_streak = 0
    longest_losing_streak = 0
    for value in r_values:
        cumulative += value
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_drawdown_r:
            max_drawdown_r = drawdown
        if value < 0:
            current_losing_streak += 1
            longest_losing_streak = max(
                longest_losing_streak,
                current_losing_streak,
            )
        else:
            current_losing_streak = 0

    total_pnl_usd = sum(
        (trade["realized_pnl"] for trade in closed),
        0.0,
    )
    by_mode: dict[str, int] = {}
    for trade in closed:
        mode = trade["mode"]
        by_mode[mode] = by_mode.get(mode, 0) + 1

    return {
        "n_closed": n_closed,
        "n_open": n_open,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_breakeven": n_breakeven,
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "avg_r_stderr": avg_r_stderr,
        "avg_r_low": avg_r_low,
        "avg_r_high": avg_r_high,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_r,
        "longest_losing_streak": longest_losing_streak,
        "total_pnl_usd": total_pnl_usd,
        "by_mode": by_mode,
        "sample_note": sample_size_note(n_closed),
    }


def format_review(review: dict) -> str:
    """Format review statistics as plain text."""
    win_rate = (
        "-" if review["win_rate"] is None
        else f"{review['win_rate'] * 100:.1f}%"
    )
    average_r = (
        "-" if review["avg_r"] is None
        else f"{review['avg_r']:.2f}"
    )
    if (
        review["avg_r_low"] is not None
        and review["n_closed"] >= MIN_TRADES_FOR_RANGE
    ):
        average_r += (
            f" (rough 95% range {review['avg_r_low']:.2f} "
            f"to {review['avg_r_high']:.2f})"
        )
    elif review["avg_r_low"] is not None:
        average_r += (
            f" (95% range shown from {MIN_TRADES_FOR_RANGE} closed trades)"
        )

    avg_win_r = (
        "-" if review["avg_win_r"] is None
        else f"{review['avg_win_r']:.2f}"
    )
    avg_loss_r = (
        "-" if review["avg_loss_r"] is None
        else f"{review['avg_loss_r']:.2f}"
    )
    if review["profit_factor"] is None:
        profit_factor = "-"
    elif math.isinf(review["profit_factor"]):
        profit_factor = "inf"
    else:
        profit_factor = f"{review['profit_factor']:.2f}"

    lines = [
        "Journal review (closed trades only)",
        f"Closed trades: {review['n_closed']} (open: {review['n_open']})",
        (
            "Wins / losses / breakeven: "
            f"{review['n_wins']} / {review['n_losses']} / "
            f"{review['n_breakeven']}"
        ),
        f"Win rate: {win_rate}",
        f"Average R per trade: {average_r}",
        f"Total R: {review['total_r']:.2f}",
        f"Total result (USD): {review['total_pnl_usd']:.2f}",
        f"Average win R: {avg_win_r} | average loss R: {avg_loss_r}",
        f"Profit factor: {profit_factor}",
        f"Max drawdown (R): {review['max_drawdown_r']:.2f}",
        f"Longest losing streak: {review['longest_losing_streak']}",
        review["sample_note"],
        "Planning aid only, not financial advice.",
    ]
    return "\n".join(lines)
