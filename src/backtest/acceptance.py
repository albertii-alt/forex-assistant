"""Development acceptance checks for pre-registered backtest outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd


MIN_TRADES = 100
MIN_PROFIT_FACTOR = 1.2


def evaluate_development(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the fixed Candidate C development acceptance criteria."""
    n_trades = summary["n_trades"]
    expectancy_r = summary["expectancy_r"]
    expectancy_r_ex_top5 = summary["expectancy_r_ex_top5"]
    profit_factor = summary["profit_factor"]

    def finite_comparison(value: Any, predicate: Any) -> bool:
        try:
            if pd.isna(value):
                return False
            return bool(predicate(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    min_trades_passed = finite_comparison(n_trades, lambda value: value >= MIN_TRADES)
    expectancy_passed = finite_comparison(
        expectancy_r, lambda value: value > 0
    )
    ex_top5_passed = finite_comparison(
        expectancy_r_ex_top5, lambda value: value > 0
    )

    try:
        pf_value = float(profit_factor)
        pf_nan = math.isnan(pf_value)
    except (TypeError, ValueError, OverflowError):
        pf_passed = False
    else:
        pf_passed = not pf_nan and pf_value > MIN_PROFIT_FACTOR

    criteria = [
        {
            "name": "min_trades",
            "required": "n_trades >= 100",
            "actual": n_trades,
            "passed": min_trades_passed,
        },
        {
            "name": "expectancy",
            "required": "expectancy_r > 0",
            "actual": expectancy_r,
            "passed": expectancy_passed,
        },
        {
            "name": "expectancy_ex_top5",
            "required": "expectancy_r_ex_top5 > 0",
            "actual": expectancy_r_ex_top5,
            "passed": ex_top5_passed,
        },
        {
            "name": "profit_factor",
            "required": "profit_factor > 1.2",
            "actual": profit_factor,
            "passed": pf_passed,
        },
    ]

    if not min_trades_passed:
        outcome = "INCONCLUSIVE"
    elif expectancy_passed and ex_top5_passed and pf_passed:
        outcome = "PASSED_DEVELOPMENT"
    else:
        outcome = "REJECTED"
    return {"outcome": outcome, "criteria": criteria}


def trades_per_year(
    entry_times: pd.DatetimeIndex | pd.Series,
    first_year: int,
    last_year: int,
) -> dict[int, int]:
    """Count trade entry times by UTC calendar year, including zero years."""
    if isinstance(entry_times, pd.DatetimeIndex):
        values = entry_times
    elif isinstance(entry_times, pd.Series):
        try:
            values = pd.DatetimeIndex(pd.to_datetime(entry_times))
        except (TypeError, ValueError) as exc:
            raise ValueError("entry_times must contain timezone-aware timestamps") from exc
    else:
        raise ValueError("entry_times must be a pandas DatetimeIndex or Series")

    if values.tz is None:
        raise ValueError("entry_times must be timezone-aware")
    utc_values = values.tz_convert("UTC")
    counts = {year: 0 for year in range(first_year, last_year + 1)}
    for timestamp in utc_values:
        year = int(timestamp.year)
        if year < first_year or year > last_year:
            raise ValueError(
                f"entry UTC year {year} is outside [{first_year}, {last_year}]"
            )
        counts[year] += 1
    return counts
