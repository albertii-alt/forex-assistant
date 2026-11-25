"""Run the one-time Candidate C development backtest report."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.backtest.acceptance import evaluate_development, trades_per_year
from src.backtest.engine import run_backtest
from src.backtest.metrics import summarize_trades
from src.data.loader import load_candles_10y
from src.data.splits import HOLDOUT_START, development_set
from src.strategies.candidate_c import find_candidate_c_plans


class ConsistencyError(Exception):
    """Raised when a pre-registration or cross-check count differs."""


def _compare_values(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isinf(expected) or math.isinf(actual):
            return math.isinf(expected) and math.isinf(actual)
        if math.isnan(expected) or math.isnan(actual):
            return math.isnan(expected) and math.isnan(actual)
    try:
        return abs(float(expected) - float(actual)) <= 1e-9
    except (TypeError, ValueError, OverflowError):
        return expected == actual


def run_development_backtest(
    dev_candles: pd.DataFrame,
    expected_plans: int | None = None,
    expected_trades: int | None = None,
) -> dict[str, Any]:
    """Run the fixed development check and report-only spread sensitivity."""
    if not isinstance(dev_candles.index, pd.DatetimeIndex):
        raise ValueError("dev_candles must have a DatetimeIndex")
    if dev_candles.index.tz is None:
        raise ValueError("dev_candles must be timezone-aware")
    utc_index = dev_candles.index.tz_convert("UTC")
    if len(utc_index) and utc_index.min() >= HOLDOUT_START:
        raise ValueError("development candles reach the holdout boundary")
    if len(utc_index) and utc_index.max() >= HOLDOUT_START:
        raise ValueError("development candles reach the holdout boundary")

    plans = find_candidate_c_plans(dev_candles)
    if expected_plans is not None and len(plans) != expected_plans:
        raise ConsistencyError(
            f"plan count mismatch: expected {expected_plans}, got {len(plans)}"
        )

    trades = run_backtest(dev_candles, plans)
    executed = trades.loc[trades["status"] == "filled"].copy()
    executed_count = len(executed)
    if expected_trades is not None and executed_count != expected_trades:
        raise ConsistencyError(
            f"executed trade count mismatch: expected {expected_trades}, "
            f"got {executed_count}"
        )

    summary = summarize_trades(trades)
    r_values = [float(value) for value in executed["r_multiple"].tolist()]
    if r_values:
        remainder = sorted(r_values, reverse=True)[5:]
        reference_ex_top5 = (
            sum(remainder) / len(remainder) if len(remainder) else None
        )
        positives = sum(value for value in r_values if value > 0)
        negatives = sum(value for value in r_values if value < 0)
        if negatives < 0:
            reference_profit_factor = positives / abs(negatives)
        elif positives > 0:
            reference_profit_factor = math.inf
        else:
            reference_profit_factor = None
    else:
        reference_ex_top5 = None
        reference_profit_factor = None

    reference_expectancy = sum(r_values) / len(r_values) if r_values else None
    if not _compare_values(reference_ex_top5, summary["expectancy_r_ex_top5"]):
        raise ConsistencyError(
            "ex-top5 cross-check mismatch: "
            f"plain-Python {reference_ex_top5}, summarize_trades "
            f"{summary['expectancy_r_ex_top5']}"
        )
    if not _compare_values(reference_profit_factor, summary["profit_factor"]):
        raise ConsistencyError(
            "profit-factor cross-check mismatch: "
            f"plain-Python {reference_profit_factor}, summarize_trades "
            f"{summary['profit_factor']}"
        )
    if not _compare_values(reference_expectancy, summary["expectancy_r"]):
        raise ConsistencyError(
            "expectancy cross-check mismatch: "
            f"plain-Python {reference_expectancy}, summarize_trades "
            f"{summary['expectancy_r']}"
        )

    evaluation = evaluate_development(summary)

    sensitivity_trades = run_backtest(
        dev_candles, plans, spread_pips=2.0
    )
    sensitivity_summary = summarize_trades(sensitivity_trades)

    first_year = int(utc_index.min().year)
    last_year = int(utc_index.max().year)
    entry_times = executed["fill_time"]
    result = {
        "plans": len(plans),
        "executed_trades": executed_count,
        "long_trades": int((executed["side"] == "long").sum()),
        "short_trades": int((executed["side"] == "short").sum()),
        "trades_per_year": trades_per_year(
            entry_times, first_year, last_year
        ),
        "summary": summary,
        "evaluation": evaluation,
        "sensitivity_2pip": {
            "n_trades": sensitivity_summary["n_trades"],
            "expectancy_r": sensitivity_summary["expectancy_r"],
            "expectancy_r_ex_top5": sensitivity_summary[
                "expectancy_r_ex_top5"
            ],
            "profit_factor": sensitivity_summary["profit_factor"],
            "max_drawdown_r": sensitivity_summary["max_drawdown_r"],
        },
    }
    return result


def _metric(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, float):
        return str(value)
    return str(value)


def format_report(result: dict[str, Any]) -> str:
    """Format a development result as plain text."""
    summary = result["summary"]
    evaluation = result["evaluation"]
    lines = [
        "PRE-REGISTERED ONE-TIME RUN: Candidate C, development set only",
        f"plans: {result['plans']}",
        f"executed trades: {result['executed_trades']}",
        f"long trades: {result['long_trades']}",
        f"short trades: {result['short_trades']}",
        "trades per year:",
    ]
    for year, count in result["trades_per_year"].items():
        lines.append(f"{year}: {count}")

    lines.extend(
        [
            "metrics:",
            f"n_trades: {_metric(summary['n_trades'])}",
            f"win_rate: {_metric(summary['win_rate'])}",
            f"expectancy_r: {_metric(summary['expectancy_r'])}",
            "expectancy_r_ex_top5: "
            f"{_metric(summary['expectancy_r_ex_top5'])}",
            f"profit_factor: {_metric(summary['profit_factor'])}",
            f"max_drawdown_r: {_metric(summary['max_drawdown_r'])}",
        ]
    )
    for criterion in evaluation["criteria"]:
        status = "PASS" if criterion["passed"] else "FAIL"
        lines.append(
            f"{status} {criterion['name']}: "
            f"required {criterion['required']}; "
            f"actual {_metric(criterion['actual'])}"
        )
    lines.append(f"OUTCOME: {evaluation['outcome']}")
    if evaluation["outcome"] == "PASSED_DEVELOPMENT":
        lines.append(
            "This is not ACCEPTED: the sealed holdout evaluation is still required."
        )

    sensitivity = result["sensitivity_2pip"]
    lines.extend(
        [
            "REPORT-ONLY: 2.0 pip spread (not used for acceptance)",
            f"n_trades: {_metric(sensitivity['n_trades'])}",
            f"expectancy_r: {_metric(sensitivity['expectancy_r'])}",
            "expectancy_r_ex_top5: "
            f"{_metric(sensitivity['expectancy_r_ex_top5'])}",
            f"profit_factor: {_metric(sensitivity['profit_factor'])}",
            f"max_drawdown_r: {_metric(sensitivity['max_drawdown_r'])}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    candles = load_candles_10y()
    dev = development_set(candles)
    print(
        f"first timestamp: {dev.index.min()}",
    )
    print(f"last timestamp: {dev.index.max()}")
    print(f"candle count: {len(dev)}")
    result = run_development_backtest(
        dev,
        expected_plans=1445,
        expected_trades=416,
    )
    print(format_report(result))


if __name__ == "__main__":
    main()
