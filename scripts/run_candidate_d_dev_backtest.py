"""Run the one-time Candidate D development backtest report."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.backtest.acceptance import evaluate_development, trades_per_year
from src.backtest.engine import run_backtest
from src.backtest.metrics import summarize_trades
from src.data.loader import load_candles_10y
from src.data.splits import HOLDOUT_START, development_set
from src.strategies.candidate_d import find_candidate_d_plans


class ConsistencyError(Exception):
    """Raised when pre-registered counts or independent metrics disagree."""


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
) -> dict[str, Any]:
    """Run Candidate D's fixed development backtest and sensitivity report."""
    if not isinstance(dev_candles.index, pd.DatetimeIndex):
        raise ValueError("dev_candles must have a DatetimeIndex")
    if dev_candles.index.tz is None:
        raise ValueError("dev_candles index must be timezone-aware")
    utc_index = dev_candles.index.tz_convert("UTC")
    if len(utc_index) and utc_index[-1] >= HOLDOUT_START:
        raise ValueError("development candles reach the holdout boundary")

    plans = find_candidate_d_plans(dev_candles)
    if expected_plans is not None and len(plans) != expected_plans:
        raise ConsistencyError(
            f"plan count mismatch: expected {expected_plans}, got {len(plans)}"
        )

    trades = run_backtest(dev_candles, plans)
    executed = trades.loc[trades["status"] == "filled"].copy()
    executed_count = len(executed)
    if executed_count != len(plans):
        status_counts = {
            status: int(count)
            for status, count in trades["status"]
            .value_counts(sort=False, dropna=False)
            .items()
        }
        raise ConsistencyError(
            f"executed trade count mismatch: plans {len(plans)}, "
            f"filled {executed_count}, status counts {status_counts}"
        )

    summary = summarize_trades(trades)
    r_values = [float(value) for value in executed["r_multiple"].tolist()]
    if r_values:
        remainder = sorted(r_values, reverse=True)[5:]
        reference_ex_top5 = (
            sum(remainder) / len(remainder) if remainder else None
        )
        gross_profit = sum(value for value in r_values if value > 0)
        gross_loss = sum(value for value in r_values if value < 0)
        if gross_loss < 0:
            reference_profit_factor = gross_profit / abs(gross_loss)
        elif gross_profit > 0:
            reference_profit_factor = math.inf
        else:
            reference_profit_factor = None
        reference_expectancy = sum(r_values) / len(r_values)
    else:
        reference_ex_top5 = None
        reference_profit_factor = None
        reference_expectancy = None

    if not _compare_values(
        reference_ex_top5,
        summary["expectancy_r_ex_top5"],
    ):
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
        dev_candles,
        plans,
        spread_pips=2.0,
    )
    sensitivity_summary = summarize_trades(sensitivity_trades)

    first_year = int(utc_index[0].year) if len(utc_index) else 0
    last_year = int(utc_index[-1].year) if len(utc_index) else -1
    exit_reasons = {
        reason: int(count)
        for reason, count in executed["exit_reason"]
        .value_counts(sort=False, dropna=False)
        .items()
    }
    return {
        "plans": len(plans),
        "executed_trades": executed_count,
        "long_trades": int((executed["side"] == "long").sum()),
        "short_trades": int((executed["side"] == "short").sum()),
        "trades_per_year": trades_per_year(
            executed["fill_time"],
            first_year,
            last_year,
        ),
        "exit_reasons": exit_reasons,
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


def _metric(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return str(value)


def format_report(result: dict[str, Any]) -> str:
    """Format the development result as plain text."""
    summary = result["summary"]
    evaluation = result["evaluation"]
    lines = [
        "PRE-REGISTERED ONE-TIME RUN: Candidate D, development set only",
        f"plans: {result['plans']}",
        f"executed trades: {result['executed_trades']}",
        f"long trades: {result['long_trades']}",
        f"short trades: {result['short_trades']}",
        "trades per year:",
    ]
    for year, count in result["trades_per_year"].items():
        lines.append(f"{year}: {count}")

    lines.append("exit reasons (report only)")
    for reason, count in result["exit_reasons"].items():
        lines.append(f"{reason}: {count}")

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
    print(f"first timestamp: {dev.index.min()}")
    print(f"last timestamp: {dev.index.max()}")
    print(f"candle count: {len(dev)}")
    result = run_development_backtest(dev, expected_plans=513)
    print(format_report(result))


if __name__ == "__main__":
    main()
