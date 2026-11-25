from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

import scripts.run_candidate_c_dev_backtest as report
from src.backtest.acceptance import (
    evaluate_development,
    trades_per_year,
)


def craft_trades(multiples: list[float]) -> pd.DataFrame:
    times = pd.date_range(
        "2023-01-01", periods=len(multiples), freq="15min", tz="UTC"
    ).as_unit("us")
    return pd.DataFrame(
        {
            "plan_id": [f"p{i}" for i in range(len(multiples))],
            "side": ["long" if i % 2 == 0 else "short" for i in range(len(multiples))],
            "order_type": ["market"] * len(multiples),
            "status": ["filled"] * len(multiples),
            "start_time": times,
            "fill_time": times,
            "entry_price": [1.1] * len(multiples),
            "stop_price": [1.09] * len(multiples),
            "target_price": [pd.NA] * len(multiples),
            "risk_pips": [10.0] * len(multiples),
            "exit_time": times + pd.Timedelta(hours=1),
            "exit_price": [1.1] * len(multiples),
            "exit_reason": ["time"] * len(multiples),
            "r_multiple": multiples,
            "pnl_pips": multiples,
            "bars_held": [1] * len(multiples),
        }
    )


def synthetic_candles(seed: int = 12345, periods: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        "2023-01-01", periods=periods, freq="15min", tz="UTC"
    ).as_unit("us")
    rets = rng.normal(0.0, 0.00008, periods)
    close = 1.1 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0.0, 0.00002, periods))
    high = np.maximum(open_, close) * (
        1.0 + np.abs(rng.normal(0.0, 0.00005, periods))
    )
    low = np.minimum(open_, close) * (
        1.0 - np.abs(rng.normal(0.0, 0.00005, periods))
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(periods),
        },
        index=index,
    )


@pytest.mark.parametrize(
    ("n_trades", "expectancy", "ex_top5", "pf", "outcome"),
    [
        (99, 0.5, 0.4, 2.0, "INCONCLUSIVE"),
        (100, 0.5, 0.4, 2.0, "PASSED_DEVELOPMENT"),
        (100, 0.0, 0.4, 2.0, "REJECTED"),
        (100, 0.5, 0.0, 2.0, "REJECTED"),
        (100, 0.5, -0.01, 2.0, "REJECTED"),
        (100, 0.5, 0.4, 1.2, "REJECTED"),
        (100, 0.5, 0.4, 1.21, "PASSED_DEVELOPMENT"),
        (100, 0.5, 0.4, math.inf, "PASSED_DEVELOPMENT"),
        (100, 0.5, 0.4, math.nan, "REJECTED"),
        (50, -0.3, -0.5, 0.5, "INCONCLUSIVE"),
        (100, 0.01, 0.001, 1.2001, "PASSED_DEVELOPMENT"),
        (100, math.nan, 0.4, 2.0, "REJECTED"),
    ],
)
def test_evaluate_development_fixtures(
    n_trades: int,
    expectancy: float,
    ex_top5: float,
    pf: float,
    outcome: str,
) -> None:
    result = evaluate_development(
        {
            "n_trades": n_trades,
            "expectancy_r": expectancy,
            "expectancy_r_ex_top5": ex_top5,
            "profit_factor": pf,
        }
    )
    assert result["outcome"] == outcome


def test_evaluate_development_criteria_order() -> None:
    result = evaluate_development(
        {
            "n_trades": 100,
            "expectancy_r": 0.5,
            "expectancy_r_ex_top5": 0.4,
            "profit_factor": 2.0,
        }
    )
    assert [item["name"] for item in result["criteria"]] == [
        "min_trades",
        "expectancy",
        "expectancy_ex_top5",
        "profit_factor",
    ]


def test_evaluate_development_seeded_reference() -> None:
    rng = random.Random(12345)
    n_choices = [0, 1, 99, 100, 101]
    expectancy_choices = [-0.1, 0.0, 0.2, math.nan]
    pf_choices = [0.5, 1.0, 1.2, 1.2000001, math.inf, math.nan]

    for _ in range(300):
        n = rng.choice(n_choices) if rng.random() < 0.5 else rng.randrange(0, 401)
        expectancy = (
            rng.choice(expectancy_choices)
            if rng.random() < 0.5
            else rng.uniform(-0.5, 1.0)
        )
        ex_top5 = (
            rng.choice([-0.5, 0.0, 0.2, math.nan])
            if rng.random() < 0.5
            else rng.uniform(-0.5, 1.0)
        )
        pf = (
            rng.choice(pf_choices)
            if rng.random() < 0.5
            else rng.uniform(0.0, 3.0)
        )
        summary = {
            "n_trades": n,
            "expectancy_r": expectancy,
            "expectancy_r_ex_top5": ex_top5,
            "profit_factor": pf,
        }

        if n < 100:
            reference = "INCONCLUSIVE"
        else:
            other_passes = []
            for value, strict_positive in (
                (expectancy, True),
                (ex_top5, True),
            ):
                try:
                    other_passes.append(not math.isnan(value) and value > 0)
                except (TypeError, ValueError):
                    other_passes.append(False)
            try:
                pf_pass = not math.isnan(pf) and pf > 1.2
            except (TypeError, ValueError):
                pf_pass = False
            other_passes.append(pf_pass)
            if all(other_passes):
                reference = "PASSED_DEVELOPMENT"
            else:
                reference = "REJECTED"

        assert evaluate_development(summary)["outcome"] == reference


def test_trades_per_year_us_ns_and_naive() -> None:
    strings = [
        "2017-12-31 23:45",
        "2018-01-01 00:00",
        "2018-06-15 12:00",
        "2018-12-31 23:45",
        "2019-01-01 00:00",
    ]
    expected = {2016: 0, 2017: 1, 2018: 3, 2019: 1, 2020: 0}
    for unit in ("us", "ns"):
        index = pd.to_datetime(strings).tz_localize("UTC").as_unit(unit)
        assert trades_per_year(index, 2016, 2020) == expected

    naive = pd.DatetimeIndex(strings)
    with pytest.raises(ValueError):
        trades_per_year(naive, 2016, 2020)

    outside = pd.DatetimeIndex(["2021-01-01"], tz="UTC")
    with pytest.raises(ValueError):
        trades_per_year(outside, 2016, 2020)


def test_trades_per_year_converts_non_utc_timezone() -> None:
    ny = ZoneInfo("America/New_York")
    index = pd.DatetimeIndex(
        [pd.Timestamp("2017-12-31 23:30", tz=ny)]
    )
    assert trades_per_year(index, 2017, 2018) == {2017: 0, 2018: 1}


def test_synthetic_run_and_format_report(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=12345, periods=5000)
    result = report.run_development_backtest(frame)
    assert set(result) == {
        "plans",
        "executed_trades",
        "long_trades",
        "short_trades",
        "trades_per_year",
        "summary",
        "evaluation",
        "sensitivity_2pip",
    }
    assert result["executed_trades"] >= 2
    assert result["evaluation"]["outcome"] in (
        "PASSED_DEVELOPMENT",
        "REJECTED",
        "INCONCLUSIVE",
    )
    text = report.format_report(result)
    assert "PRE-REGISTERED ONE-TIME RUN: Candidate C, development set only" in text
    assert "OUTCOME:" in text
    assert "REPORT-ONLY: 2.0 pip spread" in text


def test_holdout_reaches_before_find(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=1, periods=100)
    frame.index = pd.date_range(
        "2024-10-31", periods=len(frame), freq="15min", tz="UTC"
    ).as_unit("us")
    called = False

    def spy(_df: pd.DataFrame) -> pd.DataFrame:
        nonlocal called
        called = True
        return pd.DataFrame()

    monkeypatch.setattr(report, "find_candidate_c_plans", spy)
    with pytest.raises(ValueError):
        report.run_development_backtest(frame)
    assert not called


def test_wrong_plan_count_before_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=1, periods=100)
    monkeypatch.setattr(
        report,
        "find_candidate_c_plans",
        lambda _df: pd.DataFrame({"plan_id": [1, 2]}),
    )

    def spy(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("run_backtest should not be called")

    monkeypatch.setattr(report, "run_backtest", spy)
    with pytest.raises(report.ConsistencyError, match="expected 1445, got 2"):
        report.run_development_backtest(frame, expected_plans=1445)


def test_wrong_trade_count_before_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=1, periods=100)
    monkeypatch.setattr(
        report, "find_candidate_c_plans", lambda _df: pd.DataFrame()
    )
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **kwargs: craft_trades([1.0, -1.0]),
    )

    def spy(_trades: pd.DataFrame) -> dict:
        raise AssertionError("summarize_trades should not be called")

    monkeypatch.setattr(report, "summarize_trades", spy)
    with pytest.raises(report.ConsistencyError, match="expected 416, got 2"):
        report.run_development_backtest(frame, expected_trades=416)


def test_find_is_called_with_dataframe_only(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=1, periods=500)
    seen: dict[str, object] = {}

    def spy(df: pd.DataFrame) -> pd.DataFrame:
        seen["args"] = (df,)
        seen["kwargs"] = {}
        return pd.DataFrame()

    monkeypatch.setattr(report, "find_candidate_c_plans", spy)
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda df, plans, **kwargs: pd.DataFrame(
            {
                "status": pd.Series(dtype="object"),
                "r_multiple": pd.Series(dtype="float64"),
                "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "side": pd.Series(dtype="object"),
                "fill_time": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        ),
    )
    report.run_development_backtest(frame)
    assert seen["args"] == (frame,)
    assert seen["kwargs"] == {}


def test_backtest_spread_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=1, periods=500)
    monkeypatch.setattr(
        report, "find_candidate_c_plans", lambda _df: pd.DataFrame()
    )
    calls: list[dict[str, object]] = []

    def spy(_df: pd.DataFrame, _plans: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "status": pd.Series(dtype="object"),
                "r_multiple": pd.Series(dtype="float64"),
                "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "side": pd.Series(dtype="object"),
                "fill_time": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )

    monkeypatch.setattr(report, "run_backtest", spy)
    report.run_development_backtest(frame)
    assert len(calls) == 2
    assert "allow_holdout" not in calls[0]
    assert calls[0].get("spread_pips", 1.0) == 1.0
    assert calls[1]["spread_pips"] == 2.0


def test_sensitivity_does_not_change_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=1, periods=500)
    monkeypatch.setattr(
        report, "find_candidate_c_plans", lambda _df: pd.DataFrame()
    )

    def spy(_df: pd.DataFrame, _plans: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        if kwargs.get("spread_pips", 1.0) == 1.0:
            return craft_trades([1.0] * 100)
        return craft_trades([-1.0] * 100)

    monkeypatch.setattr(report, "run_backtest", spy)
    result = report.run_development_backtest(frame)
    assert result["evaluation"]["outcome"] == "PASSED_DEVELOPMENT"
    assert result["sensitivity_2pip"]["expectancy_r"] == -1.0


def test_hand_derived_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles(seed=1, periods=500)
    monkeypatch.setattr(
        report, "find_candidate_c_plans", lambda _df: pd.DataFrame()
    )
    trades = craft_trades([4, 3, 2, 1, 0.5, -1, -1, -1, -1, -1])
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **kwargs: trades,
    )
    result = report.run_development_backtest(frame)
    summary = result["summary"]
    assert summary["n_trades"] == 10
    assert summary["expectancy_r"] == pytest.approx(0.55)
    assert summary["expectancy_r_ex_top5"] == pytest.approx(-1.0)
    assert summary["profit_factor"] == pytest.approx(2.1)
    assert result["evaluation"]["outcome"] == "INCONCLUSIVE"


def test_crosscheck_rejects_wrong_expectancy_ex_top5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = synthetic_candles(seed=1, periods=500)
    monkeypatch.setattr(
        report, "find_candidate_c_plans", lambda _df: pd.DataFrame()
    )
    trades = craft_trades([1.0] * 10)
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **kwargs: trades,
    )
    real_summary = report.summarize_trades

    def wrong(trades_frame: pd.DataFrame) -> dict:
        summary = real_summary(trades_frame)
        summary["expectancy_r_ex_top5"] = -123.0
        return summary

    monkeypatch.setattr(report, "summarize_trades", wrong)
    with pytest.raises(report.ConsistencyError, match="ex-top5 cross-check"):
        report.run_development_backtest(frame)
