from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_candidate_d_dev_backtest as report
from src.backtest.metrics import summarize_trades


UTC = timezone.utc


def flat(start_utc: str, end_utc_exclusive: str) -> pd.DataFrame:
    index = pd.date_range(
        start_utc,
        end_utc_exclusive,
        freq="15min",
        inclusive="left",
        tz="UTC",
    ).as_unit("us")
    return pd.DataFrame(
        {
            "open": np.full(len(index), 1.10000),
            "high": np.full(len(index), 1.10000),
            "low": np.full(len(index), 1.10000),
            "close": np.full(len(index), 1.10000),
            "volume": np.ones(len(index)),
        },
        index=index,
    )


def set_candle(
    df: pd.DataFrame,
    utc_time: str,
    o: float,
    h: float,
    l: float,
    c: float,
) -> None:
    timestamp = pd.Timestamp(utc_time, tz="UTC")
    df.loc[timestamp, ["open", "high", "low", "close"]] = [
        round(o, 5),
        round(h, 5),
        round(l, 5),
        round(c, 5),
    ]


def summer_base() -> pd.DataFrame:
    df = flat("2023-06-12 00:00", "2023-06-14 18:00")
    set_candle(df, "2023-06-13 23:45", 1.10000, 1.10000, 1.09500, 1.10000)
    set_candle(df, "2023-06-14 00:00", 1.10000, 1.10000, 1.09900, 1.10000)
    set_candle(df, "2023-06-14 06:45", 1.10000, 1.10300, 1.10000, 1.10050)
    return df


def seeded_frames(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 2, tzinfo=UTC)
    rows: list[tuple[datetime, float, float, float, float]] = []
    close = 1.10
    day = start
    while day.date() <= datetime(2023, 7, 31).date():
        if day.weekday() < 5:
            for position in range(96):
                timestamp = day + timedelta(minutes=15 * position)
                log_return = rng.normal(0.0, 0.0004)
                next_close = close * math.exp(log_return)
                open_ = close
                high = round(
                    max(open_, next_close) + abs(rng.normal(0.0, 0.00005)),
                    5,
                )
                low = round(
                    min(open_, next_close) - abs(rng.normal(0.0, 0.00005)),
                    5,
                )
                close = round(next_close, 5)
                open_ = round(open_, 5)
                rows.append((timestamp, open_, high, low, close))
        day += timedelta(days=1)

    rng_remove = np.random.default_rng(seed + 100)
    keep = rng_remove.random(len(rows)) >= 0.005
    rows = [row for row, keep_row in zip(rows, keep) if keep_row]
    index = pd.DatetimeIndex([row[0] for row in rows], tz="UTC").as_unit("us")
    return pd.DataFrame(
        {
            "open": [row[1] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[3] for row in rows],
            "close": [row[4] for row in rows],
            "volume": np.ones(len(rows)),
        },
        index=index,
    )


def craft_trades(multiples: list[float]) -> pd.DataFrame:
    times = pd.date_range(
        "2023-01-01", periods=len(multiples), freq="15min", tz="UTC"
    ).as_unit("us")
    return pd.DataFrame(
        {
            "plan_id": [f"p{i}" for i in range(len(multiples))],
            "side": [
                "long" if i % 2 == 0 else "short"
                for i in range(len(multiples))
            ],
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


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"),
            "exit_reason": pd.Series(dtype="object"),
            "r_multiple": pd.Series(dtype="float64"),
            "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "fill_time": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def plans_frame(count: int) -> pd.DataFrame:
    return pd.DataFrame({"plan_id": [f"D-{i}" for i in range(count)]})


def simple_candles(periods: int = 200) -> pd.DataFrame:
    return flat("2023-01-02", f"2023-01-04")


def test_end_to_end_hand_report() -> None:
    df = summer_base()
    set_candle(df, "2023-06-14 07:00", 1.10000, 1.10400, 1.10000, 1.10350)
    set_candle(df, "2023-06-14 07:15", 1.10350, 1.10350, 1.10350, 1.10350)
    set_candle(df, "2023-06-14 15:45", 1.10000, 1.10100, 1.09950, 1.10080)
    set_candle(df, "2023-06-14 16:00", 1.05000, 1.05000, 1.05000, 1.05000)
    assert str(df.index.dtype).endswith("[us, UTC]")

    result = report.run_development_backtest(df, expected_plans=1)

    assert result["plans"] == 1
    assert result["executed_trades"] == 1
    assert result["long_trades"] == 1
    assert result["short_trades"] == 0
    assert result["exit_reasons"] == {"time": 1}
    assert result["summary"]["n_trades"] == 1
    assert result["summary"]["expectancy_r"] == pytest.approx(
        -0.614719, abs=1e-4
    )
    assert result["summary"]["expectancy_r_ex_top5"] is None
    assert result["summary"]["profit_factor"] == 0.0
    assert result["evaluation"]["outcome"] == "INCONCLUSIVE"
    assert result["trades_per_year"] == {2023: 1}

    text = report.format_report(result)
    assert "PRE-REGISTERED ONE-TIME RUN: Candidate D, development set only" in text
    assert "exit reasons (report only)" in text
    assert "OUTCOME: INCONCLUSIVE" in text
    assert "REPORT-ONLY: 2.0 pip spread" in text


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_seeded_end_to_end(seed: int) -> None:
    df = seeded_frames(seed)
    assert str(df.index.dtype).endswith("[us, UTC]")

    result = report.run_development_backtest(df)

    assert set(result) == {
        "plans",
        "executed_trades",
        "long_trades",
        "short_trades",
        "trades_per_year",
        "exit_reasons",
        "summary",
        "evaluation",
        "sensitivity_2pip",
    }
    assert result["executed_trades"] == result["plans"]
    assert result["long_trades"] + result["short_trades"] == result["executed_trades"]
    assert sum(result["trades_per_year"].values()) == result["executed_trades"]
    assert sum(result["exit_reasons"].values()) == result["executed_trades"]
    assert result["evaluation"]["outcome"] in {
        "PASSED_DEVELOPMENT",
        "REJECTED",
        "INCONCLUSIVE",
    }
    for year in range(df.index[0].year, df.index[-1].year + 1):
        assert year in result["trades_per_year"]


def test_holdout_guard_precedes_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    df = simple_candles()
    df.index = pd.date_range(
        "2024-10-31", periods=len(df), freq="15min", tz="UTC"
    ).as_unit("us")
    called = False

    def spy(_df: pd.DataFrame) -> pd.DataFrame:
        nonlocal called
        called = True
        return plans_frame(0)

    monkeypatch.setattr(report, "find_candidate_d_plans", spy)
    with pytest.raises(ValueError):
        report.run_development_backtest(df)
    assert not called


def test_naive_index_rejected() -> None:
    df = simple_candles()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        report.run_development_backtest(df)


def test_wrong_expected_plans_before_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    df = simple_candles()
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(2),
    )

    def engine_spy(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("run_backtest must not be called")

    monkeypatch.setattr(report, "run_backtest", engine_spy)
    with pytest.raises(
        report.ConsistencyError,
        match="expected 99, got 2",
    ):
        report.run_development_backtest(df, expected_plans=99)


def test_finder_receives_only_candles(monkeypatch: pytest.MonkeyPatch) -> None:
    df = simple_candles()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def finder_spy(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append((args, kwargs))
        return plans_frame(0)

    monkeypatch.setattr(report, "find_candidate_d_plans", finder_spy)
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **_kwargs: empty_trades(),
    )

    report.run_development_backtest(df)

    assert len(calls) == 1
    assert calls[0][0] == (df,)
    assert calls[0][1] == {}


def test_backtest_default_and_sensitivity_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = simple_candles()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(0),
    )

    def engine_spy(
        _df: pd.DataFrame,
        _plans: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        calls.append(kwargs)
        return empty_trades()

    monkeypatch.setattr(report, "run_backtest", engine_spy)
    report.run_development_backtest(df)

    assert calls == [{}, {"spread_pips": 2.0}]
    assert all("allow_holdout" not in kwargs for kwargs in calls)


def test_filled_count_mismatch_precedes_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = simple_candles()
    trades = craft_trades([1.0])
    skipped = trades.iloc[0].copy()
    skipped["plan_id"] = "skipped"
    skipped["status"] = "skipped_busy"
    trades = pd.concat([trades, skipped.to_frame().T], ignore_index=True)
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(2),
    )
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans: trades,
    )

    def summary_spy(_trades: pd.DataFrame) -> dict[str, object]:
        raise AssertionError("summarize_trades must not be called")

    monkeypatch.setattr(report, "summarize_trades", summary_spy)
    with pytest.raises(report.ConsistencyError) as exc_info:
        report.run_development_backtest(df)

    message = str(exc_info.value)
    assert "plans 2" in message
    assert "filled 1" in message
    assert "skipped_busy" in message


def test_all_winner_sensitivity_does_not_change_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = simple_candles()
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(100),
    )

    def engine_spy(
        _df: pd.DataFrame,
        _plans: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        return craft_trades(
            [1.0] * 100
            if kwargs.get("spread_pips", 1.0) == 1.0
            else [-1.0] * 100
        )

    monkeypatch.setattr(report, "run_backtest", engine_spy)
    result = report.run_development_backtest(df)

    assert result["summary"]["profit_factor"] == math.inf
    assert result["evaluation"]["outcome"] == "PASSED_DEVELOPMENT"
    assert result["sensitivity_2pip"]["expectancy_r"] == -1.0


def test_mixed_sensitivity_does_not_change_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = simple_candles()
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(100),
    )
    mixed = [2.0] * 40 + [-1.0] * 60

    def engine_spy(
        _df: pd.DataFrame,
        _plans: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        return craft_trades(
            mixed if kwargs.get("spread_pips", 1.0) == 1.0 else [-1.0] * 100
        )

    monkeypatch.setattr(report, "run_backtest", engine_spy)
    result = report.run_development_backtest(df)

    assert result["summary"]["expectancy_r"] == pytest.approx(0.2, abs=1e-6)
    assert result["summary"]["profit_factor"] == pytest.approx(
        1.3333333333, abs=1e-6
    )
    assert result["summary"]["expectancy_r_ex_top5"] == pytest.approx(
        0.1052631579, abs=1e-6
    )
    assert result["evaluation"]["outcome"] == "PASSED_DEVELOPMENT"
    assert result["sensitivity_2pip"]["expectancy_r"] == -1.0
    assert (
        "This is not ACCEPTED: the sealed holdout evaluation is still required."
        in report.format_report(result)
    )


def test_hand_derived_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    df = simple_candles()
    multiples = [1.5, 1.5, 1.5, -1, -1, -1, -1, -1, -1, -0.5]
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(len(multiples)),
    )
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **_kwargs: craft_trades(multiples),
    )

    result = report.run_development_backtest(df)
    summary = result["summary"]

    assert summary["n_trades"] == 10
    assert summary["expectancy_r"] == pytest.approx(-0.2)
    assert summary["expectancy_r_ex_top5"] == pytest.approx(-1.0)
    assert summary["profit_factor"] == pytest.approx(4.5 / 6.5, abs=1e-6)
    assert result["evaluation"]["outcome"] == "INCONCLUSIVE"


@pytest.mark.parametrize(
    ("metric", "wrong_value", "message"),
    [
        ("expectancy_r_ex_top5", -123.0, "ex-top5 cross-check"),
        ("profit_factor", 123.0, "profit-factor cross-check"),
    ],
)
def test_metric_crosschecks_reject_wrong_summary(
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    wrong_value: float,
    message: str,
) -> None:
    df = simple_candles()
    trades = craft_trades([1.0] * 10)
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(10),
    )
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **_kwargs: trades,
    )
    real_summary = summarize_trades

    def wrong_summary(results: pd.DataFrame) -> dict[str, object]:
        summary = real_summary(results)
        summary[metric] = wrong_value
        return summary

    monkeypatch.setattr(report, "summarize_trades", wrong_summary)
    with pytest.raises(report.ConsistencyError, match=message):
        report.run_development_backtest(df)


def test_trades_per_year_uses_fill_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.DatetimeIndex(
        ["2022-12-31 23:45", "2024-01-01 00:00"],
        tz="UTC",
    )
    df = pd.DataFrame(
        {
            "open": [1.1, 1.1],
            "high": [1.1, 1.1],
            "low": [1.1, 1.1],
            "close": [1.1, 1.1],
            "volume": [1.0, 1.0],
        },
        index=index,
    )
    trades = craft_trades([1.0])
    trades["start_time"] = pd.to_datetime(["2023-01-01"], utc=True)
    trades["fill_time"] = pd.to_datetime(["2024-01-01"], utc=True)
    trades["exit_time"] = pd.to_datetime(["2024-01-01 01:00"], utc=True)
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(1),
    )
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **_kwargs: trades,
    )

    result = report.run_development_backtest(df)

    assert result["trades_per_year"] == {2022: 0, 2023: 0, 2024: 1}


def test_long_count_and_exit_reasons_use_executed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = simple_candles()
    trades = craft_trades([1.0])
    skipped = trades.iloc[0].copy()
    skipped["plan_id"] = "extra"
    skipped["status"] = "unfilled_expired"
    skipped["side"] = "short"
    skipped["exit_reason"] = "not_executed"
    trades = pd.concat([trades, skipped.to_frame().T], ignore_index=True)
    monkeypatch.setattr(
        report,
        "find_candidate_d_plans",
        lambda _df: plans_frame(1),
    )
    monkeypatch.setattr(
        report,
        "run_backtest",
        lambda _df, _plans, **_kwargs: trades,
    )

    result = report.run_development_backtest(df)

    assert result["long_trades"] == 1
    assert result["short_trades"] == 0
    assert result["exit_reasons"] == {"time": 1}


def test_script_has_no_holdout_override() -> None:
    source = Path("scripts/run_candidate_d_dev_backtest.py").read_text(
        encoding="utf-8"
    )
    assert "allow_holdout" not in source
