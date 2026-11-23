"""Summary metrics for simulated trade-plan results."""

import math
from typing import Any

import numpy as np
import pandas as pd


def _empty_summary(n_plans: int, status_counts: dict[object, int]) -> dict[str, Any]:
    return {
        "n_plans": n_plans,
        "status_counts": status_counts,
        "n_trades": 0,
        "win_rate": 0.0,
        "expectancy_r": None,
        "total_r": 0.0,
        "profit_factor": None,
        "avg_win_r": None,
        "avg_loss_r": None,
        "expectancy_r_ex_top5": None,
        "max_drawdown_r": 0.0,
        "longest_losing_streak": 0,
    }


def summarize_trades(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize filled rows in stable exit-time order.

    With no trades, expectancy, profit factor, and average win/loss are None;
    counts, win rate, total R, drawdown, and losing streak are zero. Missing
    win or loss groups also have a None average. Break-even trades count as
    non-winners for losing streaks, but are not included in average loss.
    """
    n_plans = len(results)
    if "status" not in results.columns:
        if results.empty:
            return _empty_summary(0, {})
        raise ValueError("results must contain a status column")

    status_counts = {
        status: int(count)
        for status, count in results["status"].value_counts(
            sort=False,
            dropna=False,
        ).items()
    }
    if results.empty:
        return _empty_summary(n_plans, status_counts)

    missing = [
        column
        for column in ("r_multiple", "exit_time")
        if column not in results.columns
    ]
    if missing:
        raise ValueError(f"results is missing required columns: {', '.join(missing)}")
    filled = results.loc[results["status"] == "filled"].sort_values(
        "exit_time",
        kind="stable",
    )
    if filled.empty:
        return _empty_summary(n_plans, status_counts)

    multiples = filled["r_multiple"].to_numpy(dtype="float64")
    if not np.isfinite(multiples).all():
        raise ValueError("filled r_multiple values must be finite")
    n_trades = len(multiples)
    winners = multiples[multiples > 0]
    losers = multiples[multiples < 0]
    gross_profit = float(winners.sum())
    gross_loss = float(-losers.sum())
    if gross_loss > 0:
        profit_factor: float | None = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = None

    cumulative = np.cumsum(multiples)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[1:]
    drawdown = peaks - cumulative
    longest_losing_streak = 0
    current_losing_streak = 0
    for multiple in multiples:
        if multiple <= 0:
            current_losing_streak += 1
            longest_losing_streak = max(
                longest_losing_streak,
                current_losing_streak,
            )
        else:
            current_losing_streak = 0

    trimmed = sorted((float(value) for value in multiples), reverse=True)[5:]
    return {
        "n_plans": n_plans,
        "status_counts": status_counts,
        "n_trades": n_trades,
        "win_rate": float(len(winners) / n_trades),
        "expectancy_r": float(multiples.mean()),
        "total_r": float(multiples.sum()),
        "profit_factor": profit_factor,
        "avg_win_r": float(winners.mean()) if len(winners) else None,
        "avg_loss_r": float(losers.mean()) if len(losers) else None,
        "expectancy_r_ex_top5": float(np.mean(trimmed)) if n_trades > 5 else None,
        "max_drawdown_r": float(drawdown.max()) if len(drawdown) else 0.0,
        "longest_losing_streak": longest_losing_streak,
    }
