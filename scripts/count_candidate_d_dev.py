"""Print the development-only Candidate D plan count report."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_candles_10y
from src.data.splits import HOLDOUT_START, development_set
from src.strategies.candidate_d import (
    candidate_d_funnel,
    count_candidate_d,
    find_candidate_d_plans,
)


def format_count_report(
    funnel: dict[str, int],
    counts: dict[str, Any],
    stop_pips: np.ndarray,
) -> str:
    """Format structure-only Candidate D counts and funnel values."""
    lines = ["Candidate D development count (development set only)"]
    for key in (
        "dates_with_data",
        "complete_range",
        "range_ok",
        "breakout",
        "trend_ok",
        "plans",
        "dropped_missing_entry_candle",
        "plans_missing_exit_candle",
    ):
        lines.append(f"{key}: {funnel[key]}")
    lines.append(f"plans: {counts['plans']}")
    lines.append(f"long: {counts['long']}")
    lines.append(f"short: {counts['short']}")
    lines.append("per_year:")
    for year, count in counts["per_year"].items():
        lines.append(f"{year}: {count}")
    if len(stop_pips):
        median = float(np.percentile(stop_pips, 50))
        p10 = float(np.percentile(stop_pips, 10))
        p90 = float(np.percentile(stop_pips, 90))
    else:
        median = p10 = p90 = 0.0
    lines.append(f"stop distance pips median: {median:.6f}")
    lines.append(f"stop distance pips p10: {p10:.6f}")
    lines.append(f"stop distance pips p90: {p90:.6f}")
    gate = "MET" if counts["plans"] >= 130 else "NOT MET"
    lines.append(f"COUNT GATE (>= 130 development plans): {gate}")
    return "\n".join(lines)


def main() -> None:
    candles = load_candles_10y()
    dev = development_set(candles)
    if len(dev) and dev.index.min() >= HOLDOUT_START or (
        len(dev) and dev.index.max() >= HOLDOUT_START
    ):
        raise ValueError("development set reaches the holdout boundary")
    funnel = candidate_d_funnel(dev)
    counts = count_candidate_d(dev)
    plans = find_candidate_d_plans(dev)
    if plans.empty:
        stop_pips = np.array([], dtype="float64")
    else:
        stop_pips = (
            (plans["entry_price"] - plans["stop_price"]).abs().to_numpy(dtype="float64")
            / 0.0001
        )
    print(f"first timestamp: {dev.index.min()}")
    print(f"last timestamp: {dev.index.max()}")
    print(f"candle count: {len(dev)}")
    print(format_count_report(funnel, counts, stop_pips))


if __name__ == "__main__":
    main()
