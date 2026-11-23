import math

import pandas as pd
import pytest

from src.backtest.metrics import summarize_trades


def results_frame(
    multiples: list[float],
    *,
    statuses: list[str] | None = None,
    exit_times: list[str] | None = None,
) -> pd.DataFrame:
    if statuses is None:
        statuses = ["filled"] * len(multiples)
    if exit_times is None:
        exit_times = [
            f"2024-01-02 {position:02d}:00:00+00:00"
            for position in range(len(multiples))
        ]
    return pd.DataFrame(
        {
            "status": statuses,
            "r_multiple": multiples,
            "exit_time": pd.to_datetime(exit_times, utc=True),
        }
    )


def test_hand_derived_summary_metrics() -> None:
    results = results_frame([2.0, -1.0, -1.0, 2.0, -1.0, 0.5, -1.0])

    summary = summarize_trades(results)

    assert summary == {
        "n_plans": 7,
        "status_counts": {"filled": 7},
        "n_trades": 7,
        "win_rate": pytest.approx(3 / 7),
        "expectancy_r": pytest.approx(0.5 / 7),
        "total_r": pytest.approx(0.5),
        "profit_factor": pytest.approx(1.125),
        "avg_win_r": pytest.approx(1.5),
        "avg_loss_r": pytest.approx(-1.0),
        "expectancy_r_ex_top5": pytest.approx(-1.0),
        "max_drawdown_r": pytest.approx(2.0),
        "longest_losing_streak": 2,
    }


def test_exit_time_sort_is_stable_for_drawdown_and_streaks() -> None:
    results = results_frame(
        [2.0, -1.0, -1.0],
        exit_times=[
            "2024-01-02 01:00:00+00:00",
            "2024-01-02 02:00:00+00:00",
            "2024-01-02 02:00:00+00:00",
        ],
    ).iloc[[0, 2, 1]]

    summary = summarize_trades(results)

    assert summary["max_drawdown_r"] == pytest.approx(2.0)
    assert summary["longest_losing_streak"] == 2


def test_winners_only_has_infinite_profit_factor() -> None:
    summary = summarize_trades(results_frame([0.5, 2.0]))

    assert summary["profit_factor"] == math.inf
    assert summary["avg_loss_r"] is None


def test_nonfilled_statuses_count_but_do_not_affect_trade_metrics() -> None:
    results = results_frame(
        [2.0, -99.0, 100.0],
        statuses=["filled", "unfilled_expired", "skipped_busy"],
    )

    summary = summarize_trades(results)

    assert summary["n_plans"] == 3
    assert summary["status_counts"] == {
        "filled": 1,
        "unfilled_expired": 1,
        "skipped_busy": 1,
    }
    assert summary["n_trades"] == 1
    assert summary["total_r"] == pytest.approx(2.0)
    assert summary["expectancy_r"] == pytest.approx(2.0)
    assert summary["profit_factor"] == math.inf


def test_no_trades_returns_documented_empty_values() -> None:
    summary = summarize_trades(
        results_frame(
            [float("nan"), float("nan")],
            statuses=["unfilled_expired", "skipped_busy"],
        )
    )

    assert summary == {
        "n_plans": 2,
        "status_counts": {"unfilled_expired": 1, "skipped_busy": 1},
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
    assert summarize_trades(pd.DataFrame())["n_plans"] == 0


def test_ex_top5_is_none_with_five_or_fewer_trades() -> None:
    for count in range(6):
        summary = summarize_trades(results_frame([1.0] * count))
        if count <= 5:
            assert summary["expectancy_r_ex_top5"] is None


def test_break_even_counts_as_nonpositive_streak_but_not_average_loss() -> None:
    summary = summarize_trades(results_frame([1.0, 0.0, -1.0]))

    assert summary["longest_losing_streak"] == 2
    assert summary["avg_loss_r"] == pytest.approx(-1.0)
