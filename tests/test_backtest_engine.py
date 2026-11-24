from __future__ import annotations

import bisect
import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import OUTPUT_COLUMNS, run_backtest
from src.backtest.metrics import summarize_trades
from src.data.splits import HOLDOUT_START


SPREAD_PIPS = 1.0
SLIPPAGE_PIPS = 0.2
PIP = 0.0001
SPREAD = SPREAD_PIPS * PIP
SLIPPAGE = SLIPPAGE_PIPS * PIP
FILLER = (1.1000, 1.1010, 1.0995, 1.1005)


def make_index(count: int, start: str = "2024-01-02 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=count, freq="15min", tz="UTC").as_unit("ns")


def make_df(
    candles: list[tuple[float, float, float, float]],
    start: str = "2024-01-02 00:00",
) -> pd.DataFrame:
    return pd.DataFrame(
        candles,
        columns=["open", "high", "low", "close"],
        index=make_index(len(candles), start),
    )


def make_plan(
    index: pd.DatetimeIndex,
    plan_id: object = "p1",
    *,
    side: str = "long",
    order_type: str = "market",
    start_pos: int = 1,
    stop_price: float = 1.09812,
    target_r: float = 2.0,
    target_price: float = np.nan,
    valid_candles: int = 1,
    time_exit_pos: int = 4,
    **extra: object,
) -> dict[str, object]:
    return {
        "plan_id": plan_id,
        "side": side,
        "order_type": order_type,
        "start_time": index[start_pos],
        "entry_price": np.nan,
        "stop_price": stop_price,
        "target_r": target_r,
        "target_price": target_price,
        "valid_candles": valid_candles,
        "time_exit_at": index[time_exit_pos],
        **extra,
    }


def one_result(
    candles: list[tuple[float, float, float, float]],
    plan: dict[str, object],
) -> pd.Series:
    df = make_df(candles)
    return run_backtest(df, pd.DataFrame([plan])).iloc[0]


def assert_filled(
    row: pd.Series,
    *,
    entry: float,
    risk_pips: float,
    target: float,
    exit_pos: int,
    exit_price: float,
    reason: str,
    r_multiple: float,
    pnl_pips: float,
    bars_held: int,
    fill_pos: int = 1,
    index: pd.DatetimeIndex,
) -> None:
    assert row["status"] == "filled"
    assert row["entry_price"] == pytest.approx(entry, abs=1e-9)
    assert row["risk_pips"] == pytest.approx(risk_pips, abs=1e-9)
    assert row["target_price"] == pytest.approx(target, abs=1e-9)
    assert row["fill_time"] == index[fill_pos]
    assert row["exit_time"] == index[exit_pos]
    assert row["exit_price"] == pytest.approx(exit_price, abs=1e-9)
    assert row["exit_reason"] == reason
    assert row["r_multiple"] == pytest.approx(r_multiple, abs=1e-9)
    assert row["pnl_pips"] == pytest.approx(pnl_pips, abs=1e-9)
    assert row["bars_held"] == bars_held


def t1_candles() -> list[tuple[float, float, float, float]]:
    return [
        FILLER,
        (1.1000, 1.1010, 1.0995, 1.1005),
        (1.1005, 1.1030, 1.1000, 1.1020),
        (1.1020, 1.1045, 1.1010, 1.1040),
        (1.1040, 1.1050, 1.1030, 1.1045),
    ]


def t6_candles() -> list[tuple[float, float, float, float]]:
    return [
        FILLER,
        (1.1010, 1.1015, 1.09995, 1.1008),
        (1.1008, 1.1050, 1.0999, 1.1040),
        (1.1040, 1.1045, 1.1030, 1.1042),
        FILLER,
        FILLER,
        FILLER,
    ]


def test_t1_long_market_target() -> None:
    candles = t1_candles()
    index = make_index(len(candles))
    result = one_result(candles, make_plan(index))

    assert_filled(
        result,
        entry=1.10012,
        risk_pips=20.0,
        target=1.10412,
        exit_pos=3,
        exit_price=1.10412,
        reason="target",
        r_multiple=2.0,
        pnl_pips=40.0,
        bars_held=2,
        index=index,
    )


def test_t2_stop_wins_when_stop_and_target_both_trigger() -> None:
    candles = t1_candles()
    candles[3] = (1.1020, 1.1045, 1.0975, 1.1040)
    index = make_index(len(candles))
    result = one_result(candles, make_plan(index))

    assert result["exit_time"] == index[3]
    assert result["exit_price"] == pytest.approx(1.09810, abs=1e-9)
    assert result["exit_reason"] == "stop"
    assert result["r_multiple"] == pytest.approx(-1.01, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(-20.2, abs=1e-9)


def test_t3_long_stop_gap_uses_open_and_slippage() -> None:
    candles = t1_candles()
    candles[2] = (1.0950, 1.0960, 1.0940, 1.0955)
    index = make_index(len(candles))
    result = one_result(candles, make_plan(index))

    assert result["exit_time"] == index[2]
    assert result["exit_price"] == pytest.approx(1.09498, abs=1e-9)
    assert result["exit_reason"] == "stop"
    assert result["r_multiple"] == pytest.approx(-2.57, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(-51.4, abs=1e-9)


def t4_candles() -> list[tuple[float, float, float, float]]:
    return [
        FILLER,
        (1.1000, 1.1010, 1.0990, 1.0995),
        (1.0995, 1.1005, 1.0970, 1.0980),
        (1.0980, 1.0990, 1.09595, 1.0975),
        (1.0975, 1.0985, 1.0958, 1.0965),
        (1.0965, 1.0975, 1.0955, 1.0960),
    ]


def test_t4_short_target_requires_ask_to_reach() -> None:
    candles = t4_candles()
    index = make_index(len(candles))
    plan = make_plan(
        index,
        side="short",
        stop_price=1.10198,
        time_exit_pos=5,
    )
    result = one_result(candles, plan)

    assert_filled(
        result,
        entry=1.09998,
        risk_pips=20.0,
        target=1.09598,
        exit_pos=4,
        exit_price=1.09598,
        reason="target",
        r_multiple=2.0,
        pnl_pips=40.0,
        bars_held=3,
        index=index,
    )


@pytest.mark.parametrize(
    ("high", "exit_pos", "exit_price", "reason"),
    [
        (1.10188, 2, 1.10200, "stop"),
        (1.10187, 4, 1.09598, "target"),
    ],
)
def test_t5_short_stop_boundary(
    high: float,
    exit_pos: int,
    exit_price: float,
    reason: str,
) -> None:
    candles = t4_candles()
    candles[2] = (1.0995, high, 1.0970, 1.0980)
    index = make_index(len(candles))
    result = one_result(
        candles,
        make_plan(index, side="short", stop_price=1.10198, time_exit_pos=5),
    )

    assert result["exit_time"] == index[exit_pos]
    assert result["exit_price"] == pytest.approx(exit_price, abs=1e-9)
    assert result["exit_reason"] == reason
    if reason == "stop":
        assert result["r_multiple"] == pytest.approx(-1.01, abs=1e-9)


def test_t6_long_limit_ignores_target_on_fill_candle() -> None:
    candles = t6_candles()
    index = make_index(len(candles))
    plan = make_plan(
        index,
        order_type="limit",
        start_pos=1,
        stop_price=1.0980,
        target_r=2.0,
        valid_candles=3,
        time_exit_pos=6,
        entry_price=1.1000,
    )
    result = one_result(candles, plan)

    assert_filled(
        result,
        entry=1.1000,
        risk_pips=20.0,
        target=1.1040,
        exit_pos=3,
        exit_price=1.1040,
        reason="target",
        r_multiple=2.0,
        pnl_pips=40.0,
        bars_held=1,
        fill_pos=2,
        index=index,
    )


def test_t7_limit_fill_and_stop_same_candle_uses_no_gap_adjustment() -> None:
    candles = t6_candles()
    candles[2] = (1.1008, 1.1010, 1.0975, 1.0990)
    index = make_index(len(candles))
    plan = make_plan(
        index,
        order_type="limit",
        stop_price=1.0980,
        valid_candles=3,
        time_exit_pos=6,
        entry_price=1.1000,
    )
    result = one_result(candles, plan)

    assert result["fill_time"] == index[2]
    assert result["exit_time"] == index[2]
    assert result["exit_price"] == pytest.approx(1.09798, abs=1e-9)
    assert result["exit_reason"] == "stop"
    assert result["r_multiple"] == pytest.approx(-1.01, abs=1e-9)
    assert result["bars_held"] == 0


def test_t8_short_limit_and_target() -> None:
    candles = [
        FILLER,
        (1.0990, 1.0999, 1.0985, 1.0995),
        (1.0995, 1.1000, 1.0950, 1.0960),
        (1.0960, 1.0970, 1.0958, 1.0962),
        FILLER,
        FILLER,
        FILLER,
    ]
    index = make_index(len(candles))
    plan = make_plan(
        index,
        side="short",
        order_type="limit",
        stop_price=1.1020,
        valid_candles=3,
        time_exit_pos=6,
        entry_price=1.1000,
    )
    result = one_result(candles, plan)

    assert result["fill_time"] == index[2]
    assert result["target_price"] == pytest.approx(1.0960, abs=1e-9)
    assert result["exit_time"] == index[3]
    assert result["exit_price"] == pytest.approx(1.0960, abs=1e-9)
    assert result["r_multiple"] == pytest.approx(2.0, abs=1e-9)


def test_t9_long_time_exit() -> None:
    candles = [
        FILLER,
        (1.1000, 1.1010, 1.0995, 1.1005),
        (1.1005, 1.1020, 1.1000, 1.1010),
        (1.1010, 1.1030, 1.1005, 1.1015),
        FILLER,
    ]
    index = make_index(len(candles))
    result = one_result(candles, make_plan(index, time_exit_pos=3))

    assert result["exit_time"] == index[3]
    assert result["exit_price"] == pytest.approx(1.10148, abs=1e-9)
    assert result["exit_reason"] == "time"
    assert result["r_multiple"] == pytest.approx(0.68, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(13.6, abs=1e-9)


def test_t10_short_time_exit() -> None:
    candles = [
        FILLER,
        (1.1000, 1.1010, 1.0995, 1.0998),
        (1.0998, 1.1005, 1.0990, 1.1000),
        (1.1000, 1.1008, 1.0995, 1.0996),
        FILLER,
    ]
    index = make_index(len(candles))
    result = one_result(
        candles,
        make_plan(index, side="short", stop_price=1.10198, time_exit_pos=3),
    )

    assert result["entry_price"] == pytest.approx(1.09998, abs=1e-9)
    assert result["exit_time"] == index[3]
    assert result["exit_price"] == pytest.approx(1.09972, abs=1e-9)
    assert result["exit_reason"] == "time"
    assert result["r_multiple"] == pytest.approx(0.13, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(2.6, abs=1e-9)


def test_t11_limit_expiration_and_valid_candle_count() -> None:
    candles = t6_candles()
    index = make_index(len(candles))
    base = make_plan(
        index,
        order_type="limit",
        stop_price=1.0980,
        valid_candles=3,
        time_exit_pos=6,
        entry_price=1.1000,
    )
    no_fill = list(candles)
    no_fill[1] = (1.1010, 1.1020, 1.1001, 1.1010)
    no_fill[2] = (1.1010, 1.1020, 1.1001, 1.1010)
    no_fill[3] = (1.1010, 1.1020, 1.1001, 1.1010)
    unfilled = one_result(no_fill, base)
    assert unfilled["status"] == "unfilled_expired"
    for column in (
        "fill_time",
        "entry_price",
        "stop_price",
        "target_price",
        "risk_pips",
        "exit_time",
        "exit_price",
        "exit_reason",
        "r_multiple",
        "pnl_pips",
        "bars_held",
    ):
        assert pd.isna(unfilled[column])

    expires_early = dict(base, valid_candles=1)
    assert one_result(candles, expires_early)["status"] == "unfilled_expired"

    exits_before_fill = dict(base, time_exit_at=index[1])
    assert one_result(candles, exits_before_fill)["status"] == "unfilled_expired"


def test_t12_busy_position_and_later_plan() -> None:
    candles = t1_candles() + [(1.1045, 1.1055, 1.1035, 1.1050)]
    index = make_index(len(candles))
    plans = pd.DataFrame(
        [
            make_plan(index, plan_id="first", time_exit_pos=4),
            make_plan(
                index,
                plan_id="busy",
                start_pos=3,
                stop_price=1.09812,
                time_exit_pos=5,
            ),
            make_plan(
                index,
                plan_id="later",
                start_pos=4,
                stop_price=1.10312,
                time_exit_pos=5,
            ),
        ]
    )

    result = run_backtest(make_df(candles), plans)

    assert result["plan_id"].tolist() == ["first", "busy", "later"]
    assert result["status"].tolist() == ["filled", "skipped_busy", "filled"]
    for column in (
        "fill_time",
        "entry_price",
        "stop_price",
        "target_price",
        "risk_pips",
        "exit_time",
        "exit_price",
        "exit_reason",
        "r_multiple",
        "pnl_pips",
        "bars_held",
    ):
        assert pd.isna(result.loc[1, column])
    assert result.loc[2, "entry_price"] == pytest.approx(1.10412, abs=1e-9)
    assert result.loc[2, "risk_pips"] == pytest.approx(10.0, abs=1e-9)
    assert result.loc[2, "exit_time"] == index[4]
    assert result.loc[2, "exit_price"] == pytest.approx(1.10310, abs=1e-9)
    assert result.loc[2, "r_multiple"] == pytest.approx(-1.02, abs=1e-9)
    assert result.loc[2, "bars_held"] == 0


def test_t13_risk_filters_and_bad_market_levels() -> None:
    candles = t1_candles()
    index = make_index(len(candles))
    for filters in (
        {"min_risk_pips": 25},
        {"max_risk_pips": 15},
    ):
        plan = make_plan(index, **filters)
        assert one_result(candles, plan)["status"] == "skipped_risk_filter"

    allowed = make_plan(index, min_risk_pips=10, max_risk_pips=25)
    assert one_result(candles, allowed)["status"] == "filled"
    wrong_stop = make_plan(index, stop_price=1.1100)
    assert one_result(candles, wrong_stop)["status"] == "skipped_bad_levels"


def test_t14_explicit_limit_target() -> None:
    candles = t6_candles()
    index = make_index(len(candles))
    plan = make_plan(
        index,
        order_type="limit",
        stop_price=1.0980,
        target_r=np.nan,
        target_price=1.1030,
        valid_candles=3,
        time_exit_pos=6,
        entry_price=1.1000,
    )
    result = one_result(candles, plan)

    assert result["fill_time"] == index[2]
    assert result["risk_pips"] == pytest.approx(20.0, abs=1e-9)
    assert result["target_price"] == pytest.approx(1.1030, abs=1e-9)
    assert result["exit_time"] == index[3]
    assert result["exit_price"] == pytest.approx(1.1030, abs=1e-9)
    assert result["r_multiple"] == pytest.approx(1.5, abs=1e-9)


def test_t15_data_end_and_no_data() -> None:
    candles = [
        FILLER,
        (1.1000, 1.1010, 1.0995, 1.1005),
        (1.1005, 1.1010, 1.1000, 1.1005),
        (1.1005, 1.1010, 1.1000, 1.1005),
    ]
    index = make_index(len(candles))
    data_end = make_plan(
        index,
        plan_id="end",
        time_exit_pos=3,
        stop_price=1.0900,
        time_exit_at=index[-1] + pd.Timedelta(minutes=15),
    )
    no_data = dict(
        data_end,
        plan_id="missing",
        start_time=index[-1] + pd.Timedelta(minutes=15),
        time_exit_at=index[-1] + pd.Timedelta(minutes=30),
    )
    result = run_backtest(
        make_df(candles),
        pd.DataFrame([data_end, no_data]),
    )

    assert result["status"].tolist() == ["filled", "no_data"]
    assert result.loc[0, "exit_time"] == index[-1]
    assert result.loc[0, "exit_reason"] == "data_end"
    assert result.loc[0, "exit_price"] == pytest.approx(candles[-1][3] - SLIPPAGE)
    assert pd.isna(result.loc[1, "entry_price"])
    assert pd.isna(result.loc[1, "exit_time"])


def test_market_explicit_target_on_wrong_side_is_skipped() -> None:
    candles = t1_candles()
    index = make_index(len(candles))
    plan = make_plan(index, target_r=np.nan, target_price=1.0990)

    assert one_result(candles, plan)["status"] == "skipped_bad_levels"


def test_t16_validation_errors() -> None:
    candles = t1_candles()
    df = make_df(candles)
    index = df.index
    valid = make_plan(index)

    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest(df, pd.DataFrame([{"plan_id": "p1"}]))
    with pytest.raises(ValueError, match="side"):
        run_backtest(df, pd.DataFrame([dict(valid, side="buy")]))
    with pytest.raises(ValueError, match="order_type"):
        run_backtest(df, pd.DataFrame([dict(valid, order_type="stop")]))
    with pytest.raises(ValueError, match="unique"):
        run_backtest(df, pd.DataFrame([valid, dict(valid)]))
    with pytest.raises(ValueError, match="exactly one"):
        run_backtest(df, pd.DataFrame([dict(valid, target_price=1.105)]))
    with pytest.raises(ValueError, match="exactly one"):
        run_backtest(df, pd.DataFrame([dict(valid, target_r=np.nan)]))
    with pytest.raises(ValueError, match="greater than zero"):
        run_backtest(df, pd.DataFrame([dict(valid, target_r=0)]))
    with pytest.raises(ValueError, match="valid_candles"):
        run_backtest(df, pd.DataFrame([dict(valid, valid_candles=0)]))
    with pytest.raises(ValueError, match="valid_candles"):
        run_backtest(df, pd.DataFrame([dict(valid, valid_candles=1.5)]))

    limit = dict(valid, order_type="limit", entry_price=1.1000)
    with pytest.raises(ValueError, match="entry_price"):
        run_backtest(df, pd.DataFrame([dict(limit, entry_price=np.nan)]))
    with pytest.raises(ValueError, match="stop"):
        run_backtest(df, pd.DataFrame([dict(limit, stop_price=1.1000)]))
    with pytest.raises(ValueError, match="target_price"):
        run_backtest(
            df,
            pd.DataFrame(
                [dict(limit, target_r=np.nan, target_price=1.0990)]
            ),
        )
    with pytest.raises(ValueError, match="time_exit_at"):
        run_backtest(
            df,
            pd.DataFrame([dict(valid, start_time=index[2], time_exit_at=index[1])]),
        )
    with pytest.raises(ValueError, match="overlapping"):
        run_backtest(
            df,
            pd.DataFrame(
                [
                    dict(valid, valid_candles=2),
                    dict(valid, plan_id="p2", start_time=index[2]),
                ]
            ),
        )

    naive_index = pd.date_range("2024-01-02", periods=len(candles), freq="15min")
    with pytest.raises(ValueError, match="UTC"):
        run_backtest(make_df(candles).set_axis(naive_index), pd.DataFrame([valid]))
    with pytest.raises(ValueError, match="UTC"):
        run_backtest(
            df,
            pd.DataFrame(
                [dict(valid, start_time=pd.Timestamp("2024-01-02 00:15"))]
            ),
        )
    with pytest.raises(ValueError, match="UTC"):
        run_backtest(
            df,
            pd.DataFrame(
                [dict(valid, time_exit_at=pd.Timestamp("2024-01-02 01:00"))]
            ),
        )
    with pytest.raises(ValueError, match="stop"):
        run_backtest(
            df,
            pd.DataFrame(
                [
                    dict(
                        valid,
                        side="short",
                        order_type="limit",
                        entry_price=1.1000,
                        stop_price=1.1000,
                    )
                ]
            ),
        )
    with pytest.raises(ValueError, match="target_price"):
        run_backtest(
            df,
            pd.DataFrame(
                [
                    dict(
                        valid,
                        side="short",
                        order_type="limit",
                        entry_price=1.1000,
                        stop_price=1.1020,
                        target_r=np.nan,
                        target_price=1.1010,
                    )
                ]
            ),
        )
    with pytest.raises(ValueError, match="sorted"):
        run_backtest(df.iloc[::-1], pd.DataFrame([valid]))
    with pytest.raises(ValueError, match="spread_pips"):
        run_backtest(df, pd.DataFrame([valid]), spread_pips=-0.1)
    with pytest.raises(ValueError, match="slippage_pips"):
        run_backtest(df, pd.DataFrame([valid]), slippage_pips=-0.1)
    with pytest.raises(ValueError, match="pip_size"):
        run_backtest(df, pd.DataFrame([valid]), pip_size=0)


def test_t16_holdout_guard_and_explicit_override() -> None:
    index = pd.date_range(
        "2024-10-31 00:00",
        "2024-11-02 00:00",
        freq="15min",
        tz="UTC",
    ).as_unit("ns")
    df = pd.DataFrame([FILLER] * len(index), columns=["open", "high", "low", "close"], index=index)
    index = df.index
    plan = make_plan(
        index,
        stop_price=1.0900,
        time_exit_pos=2,
    )
    assert index[-1] >= HOLDOUT_START

    with pytest.raises(ValueError, match="holdout is sealed"):
        run_backtest(df, pd.DataFrame([plan]))
    result = run_backtest(df, pd.DataFrame([plan]), allow_holdout=True)
    assert result.loc[0, "status"] == "filled"


def test_non_data_plans_do_not_create_overlap_errors_and_output_is_sorted() -> None:
    candles = [FILLER] * 5
    df = make_df(candles)
    index = df.index
    later = make_plan(
        index,
        plan_id="later",
        start_pos=3,
        time_exit_pos=4,
        stop_price=1.0900,
    )
    earlier = make_plan(
        index,
        plan_id="earlier",
        start_pos=1,
        time_exit_pos=2,
        valid_candles=2,
        stop_price=1.0900,
    )
    no_data = dict(
        later,
        plan_id="no-data",
        start_time=index[-1] + pd.Timedelta(minutes=15),
        time_exit_at=index[-1] + pd.Timedelta(minutes=30),
    )

    result = run_backtest(df, pd.DataFrame([later, no_data, earlier]))

    assert result["plan_id"].tolist() == ["earlier", "later", "no-data"]
    assert result["status"].tolist() == ["filled", "filled", "no_data"]


def test_t17_inputs_unchanged_and_output_dtypes() -> None:
    df = make_df(t1_candles())
    plans = pd.DataFrame([make_plan(df.index)])
    df_before = df.copy(deep=True)
    plans_before = plans.copy(deep=True)

    result = run_backtest(df, plans)

    pd.testing.assert_frame_equal(df, df_before)
    pd.testing.assert_frame_equal(plans, plans_before)
    assert list(result.columns) == list(OUTPUT_COLUMNS)
    for column in ("start_time", "fill_time", "exit_time"):
        assert str(result[column].dtype) == "datetime64[ns, UTC]"
    assert result["entry_price"].dtype == "float64"
    assert result["bars_held"].dtype == "float64"
    assert isinstance(result.index, pd.RangeIndex)


def no_target_plan(
    index: pd.DatetimeIndex,
    **overrides: object,
) -> dict[str, object]:
    start_pos = int(overrides.pop("start_pos", 1))
    time_exit_pos = int(overrides.pop("time_exit_pos", 4))
    plan = make_plan(
        index,
        start_pos=start_pos,
        time_exit_pos=time_exit_pos,
        target_r=np.nan,
        target_price=np.nan,
        no_target=True,
        exit_mode="time_open",
    )
    plan.update(overrides)
    return plan


def test_e1_time_open_long_without_target() -> None:
    candles = t1_candles()
    index = make_index(len(candles))
    result = one_result(candles, no_target_plan(index))

    assert result["status"] == "filled"
    assert result["fill_time"] == index[1]
    assert result["entry_price"] == pytest.approx(1.10012, abs=1e-9)
    assert result["risk_pips"] == pytest.approx(20.0, abs=1e-9)
    assert pd.isna(result["target_price"])
    assert result["exit_time"] == index[4]
    assert result["exit_price"] == pytest.approx(1.10398, abs=1e-9)
    assert result["exit_reason"] == "time"
    assert result["r_multiple"] == pytest.approx(1.93, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(38.6, abs=1e-9)
    assert result["bars_held"] == 3
    assert summarize_trades(pd.DataFrame([result]))["n_trades"] == 1


def test_e2_time_open_ignores_exit_candle_stop_but_time_close_does_not() -> None:
    candles = t1_candles()
    candles[4] = (1.1040, 1.1050, 1.0970, 1.1045)
    index = make_index(len(candles))

    opened = one_result(candles, no_target_plan(index))
    assert opened["exit_time"] == index[4]
    assert opened["exit_price"] == pytest.approx(1.10398, abs=1e-9)
    assert opened["exit_reason"] == "time"
    assert opened["r_multiple"] == pytest.approx(1.93, abs=1e-9)
    assert opened["pnl_pips"] == pytest.approx(38.6, abs=1e-9)
    assert opened["bars_held"] == 3

    closed = one_result(
        candles,
        no_target_plan(index, exit_mode="time_close"),
    )
    assert closed["exit_time"] == index[4]
    assert closed["exit_price"] == pytest.approx(1.09810, abs=1e-9)
    assert closed["exit_reason"] == "stop"
    assert closed["r_multiple"] == pytest.approx(-1.01, abs=1e-9)
    assert closed["pnl_pips"] == pytest.approx(-20.2, abs=1e-9)
    assert closed["bars_held"] == 3


def test_e3_time_open_checks_stops_before_exit_candle() -> None:
    candles = t1_candles()
    candles[3] = (1.1020, 1.1045, 1.0975, 1.1040)
    index = make_index(len(candles))
    result = one_result(candles, no_target_plan(index))

    assert result["exit_time"] == index[3]
    assert result["exit_price"] == pytest.approx(1.09810, abs=1e-9)
    assert result["exit_reason"] == "stop"
    assert result["r_multiple"] == pytest.approx(-1.01, abs=1e-9)
    assert result["bars_held"] == 2


def test_e4_time_open_stop_gap_uses_open_and_slippage() -> None:
    candles = t1_candles()
    candles[2] = (1.0950, 1.0960, 1.0940, 1.0955)
    index = make_index(len(candles))
    result = one_result(candles, no_target_plan(index))

    assert result["exit_time"] == index[2]
    assert result["exit_price"] == pytest.approx(1.09498, abs=1e-9)
    assert result["exit_reason"] == "stop"
    assert result["r_multiple"] == pytest.approx(-2.57, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(-51.4, abs=1e-9)


def test_e5_time_open_short_includes_spread_and_slippage() -> None:
    candles = [
        FILLER,
        (1.1000, 1.1010, 1.0995, 1.0998),
        (1.0998, 1.1005, 1.0990, 1.1000),
        (1.1000, 1.1008, 1.0995, 1.0996),
        (1.0990, 1.1000, 1.0985, 1.0992),
    ]
    index = make_index(len(candles))
    plan = no_target_plan(
        index,
        side="short",
        stop_price=1.10198,
    )
    result = one_result(candles, plan)

    assert result["entry_price"] == pytest.approx(1.09998, abs=1e-9)
    assert result["risk_pips"] == pytest.approx(20.0, abs=1e-9)
    assert result["exit_time"] == index[4]
    assert result["exit_price"] == pytest.approx(1.09912, abs=1e-9)
    assert result["exit_reason"] == "time"
    assert result["r_multiple"] == pytest.approx(0.43, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(8.6, abs=1e-9)
    assert result["bars_held"] == 3


def test_e6_time_open_after_data_uses_data_end_close() -> None:
    candles = t1_candles()
    index = make_index(len(candles))
    plan = no_target_plan(
        index,
        time_exit_at=index[4] + pd.Timedelta(days=1),
    )
    result = one_result(candles, plan)

    assert result["exit_time"] == index[4]
    assert result["exit_price"] == pytest.approx(1.10448, abs=1e-9)
    assert result["exit_reason"] == "data_end"
    assert result["r_multiple"] == pytest.approx(2.18, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(43.6, abs=1e-9)
    assert result["bars_held"] == 3


def test_e7_time_open_between_candle_opens_uses_last_open() -> None:
    candles = t1_candles()
    index = make_index(len(candles))
    plan = no_target_plan(
        index,
        time_exit_at=index[3] + pd.Timedelta(minutes=5),
    )
    result = one_result(candles, plan)

    assert result["exit_time"] == index[3]
    assert result["exit_price"] == pytest.approx(1.10198, abs=1e-9)
    assert result["exit_reason"] == "time"
    assert result["r_multiple"] == pytest.approx(0.93, abs=1e-9)
    assert result["pnl_pips"] == pytest.approx(18.6, abs=1e-9)
    assert result["bars_held"] == 2


def test_e8_optional_plan_column_validation() -> None:
    candles = t1_candles()
    df = make_df(candles)
    index = df.index
    valid = no_target_plan(index)
    invalid = [
        dict(valid, target_r=2.0),
        dict(valid, target_price=1.1040),
        {
            key: value
            for key, value in dict(
                valid,
                target_r=np.nan,
                target_price=np.nan,
                no_target=False,
            ).items()
            if key != "no_target"
        },
        dict(valid, no_target="yes"),
        dict(valid, exit_mode="foo"),
        dict(valid, order_type="limit", entry_price=1.1000),
        dict(valid, time_exit_at=index[1]),
    ]
    for plan in invalid:
        with pytest.raises(ValueError):
            run_backtest(df, pd.DataFrame([plan]))


def test_e9_default_plan_columns_are_backward_compatible() -> None:
    candles = t1_candles()
    df = make_df(candles)
    plan = make_plan(df.index)

    without_columns = run_backtest(df, pd.DataFrame([plan]))
    with_defaults = run_backtest(
        df,
        pd.DataFrame([dict(plan, no_target=False, exit_mode="time_close")]),
    )
    pd.testing.assert_frame_equal(without_columns, with_defaults)


@pytest.mark.parametrize(
    ("side", "candles", "stop_price"),
    [
        (
            "long",
            t1_candles(),
            1.09812,
        ),
        (
            "short",
            [
                FILLER,
                (1.1000, 1.1010, 1.0995, 1.0998),
                (1.0998, 1.1005, 1.0990, 1.1000),
                (1.1000, 1.1008, 1.0995, 1.0996),
                (1.0990, 1.1000, 1.0985, 1.0992),
            ],
            1.10198,
        ),
    ],
)
def test_e10_time_open_unit_precision(
    side: str,
    candles: list[tuple[float, float, float, float]],
    stop_price: float,
) -> None:
    ns_index = make_index(len(candles))
    ns_plan = no_target_plan(ns_index, side=side, stop_price=stop_price)
    ns_result = run_backtest(make_df(candles), pd.DataFrame([ns_plan]))

    us_index = ns_index.as_unit("us")
    us_plan = no_target_plan(us_index, side=side, stop_price=stop_price)
    us_result = run_backtest(
        pd.DataFrame(
            candles,
            columns=["open", "high", "low", "close"],
            index=us_index,
        ),
        pd.DataFrame([us_plan]),
    )
    pd.testing.assert_frame_equal(ns_result, us_result)


def test_e11_time_open_exit_respects_busy_until_exit_candle() -> None:
    candles = t1_candles() + [FILLER, FILLER]
    df = make_df(candles)
    index = df.index
    plans = pd.DataFrame(
        [
            no_target_plan(index, plan_id="first"),
            no_target_plan(
                index,
                plan_id="busy",
                start_pos=4,
                time_exit_pos=6,
            ),
            no_target_plan(
                index,
                plan_id="later",
                start_pos=5,
                time_exit_pos=6,
            ),
        ]
    )

    result = run_backtest(df, plans)

    assert result["status"].tolist() == ["filled", "skipped_busy", "filled"]
    assert result.loc[0, "exit_time"] == index[4]
    assert result.loc[2, "fill_time"] == index[5]


def normalize_reference_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in ("start_time", "fill_time", "exit_time"):
        result[column] = pd.Series(
            pd.DatetimeIndex(pd.to_datetime(result[column], utc=True)).as_unit("ns"),
            index=result.index,
        )
    for column in (
        "entry_price",
        "stop_price",
        "target_price",
        "risk_pips",
        "exit_price",
        "r_multiple",
        "pnl_pips",
        "bars_held",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    return result


def independent_reference(
    candle_rows: list[tuple[float, float, float, float]],
    candle_times: list[pd.Timestamp],
    plans: list[dict[str, object]],
    spread_pips: float,
    slippage_pips: float,
    pip_size: float,
) -> pd.DataFrame:
    """Simple Python reference; the per-plan/candle simulation uses no pandas."""
    times_ns = [int(timestamp.value) for timestamp in candle_times]
    python_times = [timestamp.to_pydatetime() for timestamp in candle_times]
    candles = [
        tuple(float(value) for value in candle)
        for candle in candle_rows
    ]
    spread = spread_pips * pip_size
    slip = slippage_pips * pip_size
    prepared = []
    for original_order, plan in enumerate(plans):
        start_time = pd.Timestamp(plan["start_time"])
        time_exit_at = pd.Timestamp(plan["time_exit_at"])
        start_ns = int(start_time.value)
        exit_ns = int(time_exit_at.value)
        start_pos = bisect.bisect_left(times_ns, start_ns)
        exit_pos = bisect.bisect_right(times_ns, exit_ns) - 1
        exit_is_after_data = time_exit_at > candle_times[-1]
        prepared.append(
            (
                start_pos,
                original_order,
                plan,
                exit_pos,
                exit_is_after_data,
                start_time.to_pydatetime(),
            )
        )
    prepared.sort(key=lambda item: (item[0], item[1]))

    rows = []
    busy_until = -1
    missing_number = float("nan")
    for (
        start_pos,
        _,
        plan,
        exit_pos,
        exit_is_after_data,
        start_time,
    ) in prepared:
        side = str(plan["side"])
        order_type = str(plan["order_type"])
        no_target_value = plan.get("no_target", False)
        no_target = (
            bool(no_target_value)
            if isinstance(no_target_value, (bool, np.bool_))
            else False
        )
        exit_mode_value = plan.get("exit_mode", "time_close")
        exit_mode = (
            exit_mode_value
            if isinstance(exit_mode_value, str)
            else "time_close"
        )
        plan_id = plan["plan_id"]
        row = {
            "plan_id": plan_id,
            "side": side,
            "order_type": order_type,
            "status": "",
            "start_time": start_time,
            "fill_time": None,
            "entry_price": missing_number,
            "stop_price": missing_number,
            "target_price": missing_number,
            "risk_pips": missing_number,
            "exit_time": None,
            "exit_price": missing_number,
            "exit_reason": None,
            "r_multiple": missing_number,
            "pnl_pips": missing_number,
            "bars_held": missing_number,
        }
        if start_pos >= len(candles):
            row["status"] = "no_data"
            rows.append(row)
            continue
        if start_pos <= busy_until:
            row["status"] = "skipped_busy"
            rows.append(row)
            continue

        valid = int(plan["valid_candles"])
        stop = float(plan["stop_price"])
        explicit_target = None
        target_r = None
        if not no_target:
            explicit_target = (
                None
                if math.isnan(float(plan["target_price"]))
                else float(plan["target_price"])
            )
            target_r = (
                None
                if math.isnan(float(plan["target_r"]))
                else float(plan["target_r"])
            )
        fill_pos = start_pos
        if order_type == "market":
            opening = candles[start_pos][0]
            fill = opening + spread + slip if side == "long" else opening - slip
        else:
            fill = float(plan["entry_price"])
            fill_pos = -1
            last_live = min(start_pos + valid - 1, exit_pos)
            for position in range(start_pos, last_live + 1):
                opening, high, low, _ = candles[position]
                _ = opening
                reached = (
                    low + spread <= fill + 1e-9
                    if side == "long"
                    else high >= fill - 1e-9
                )
                if reached:
                    fill_pos = position
                    break
            if fill_pos < 0:
                row["status"] = "unfilled_expired"
                rows.append(row)
                continue

        risk = fill - stop if side == "long" else stop - fill
        bad_target = (
            explicit_target is not None
            and (
                explicit_target <= fill
                if side == "long"
                else explicit_target >= fill
            )
        )
        if risk <= 0 or bad_target:
            row["status"] = "skipped_bad_levels"
            rows.append(row)
            continue
        risk_pips = risk / pip_size
        minimum = plan.get("min_risk_pips", np.nan)
        maximum = plan.get("max_risk_pips", np.nan)
        if (
            (not math.isnan(float(minimum)) and risk_pips + 1e-9 < float(minimum))
            or (not math.isnan(float(maximum)) and risk_pips - 1e-9 > float(maximum))
        ):
            row["status"] = "skipped_risk_filter"
            rows.append(row)
            continue

        if no_target:
            target = missing_number
        elif target_r is not None:
            distance = target_r * risk
            target = fill + distance if side == "long" else fill - distance
        else:
            target = float(explicit_target)
        reason_at_end = "data_end" if exit_is_after_data else "time"
        actual_exit_pos = -1
        exit_price = 0.0
        exit_reason = ""
        for position in range(fill_pos, exit_pos + 1):
            opening, high, low, close = candles[position]
            if (
                exit_mode == "time_open"
                and not exit_is_after_data
                and position == exit_pos
            ):
                exit_price = (
                    opening - slip
                    if side == "long"
                    else opening + spread + slip
                )
                exit_reason = "time"
                actual_exit_pos = position
                break

            if side == "long":
                stop_hit = low <= stop + 1e-9
                target_hit = (
                    not no_target
                    and not (order_type == "limit" and position == fill_pos)
                    and high >= target - 1e-9
                )
            else:
                stop_hit = high + spread >= stop - 1e-9
                target_hit = (
                    not no_target
                    and not (order_type == "limit" and position == fill_pos)
                    and low + spread <= target + 1e-9
                )

            if stop_hit:
                if side == "long":
                    exit_price = (
                        stop - slip
                        if order_type == "limit" and position == fill_pos
                        else min(stop, opening) - slip
                    )
                else:
                    exit_price = (
                        stop + slip
                        if order_type == "limit" and position == fill_pos
                        else max(stop, opening + spread) + slip
                    )
                exit_reason = "stop"
                actual_exit_pos = position
                break
            if target_hit:
                exit_price = target
                exit_reason = "target"
                actual_exit_pos = position
                break
            if position == exit_pos:
                exit_price = close - slip if side == "long" else close + spread + slip
                exit_reason = reason_at_end
                actual_exit_pos = position
                break

        pnl = exit_price - fill if side == "long" else fill - exit_price
        row.update(
            {
                "status": "filled",
                "fill_time": python_times[fill_pos],
                "entry_price": fill,
                "stop_price": stop,
                "target_price": target,
                "risk_pips": risk_pips,
                "exit_time": python_times[actual_exit_pos],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "r_multiple": pnl / risk,
                "pnl_pips": pnl / pip_size,
                "bars_held": actual_exit_pos - fill_pos,
            }
        )
        busy_until = actual_exit_pos
        rows.append(row)
    return normalize_reference_rows(rows)


def make_random_case(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = make_index(3000, "2024-01-01 00:00")
    opens = 1.1000 + np.cumsum(rng.normal(0.0, 0.00004, len(index)))
    closes = opens + rng.normal(0.0, 0.00004, len(index))
    upper_wicks = rng.uniform(0.00002, 0.00015, len(index))
    lower_wicks = rng.uniform(0.00002, 0.00015, len(index))
    df = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + upper_wicks,
            "low": np.minimum(opens, closes) - lower_wicks,
            "close": closes,
        },
        index=index,
    )

    def set_candle(
        position: int,
        opening: float,
        high: float,
        low: float,
        close: float,
    ) -> None:
        candle = (
            opening,
            max(high, opening, close),
            min(low, opening, close),
            close,
        )
        df.iloc[position] = candle

    plans = []
    start_pos = 2
    for number in range(120):
        valid = int(rng.integers(1, 9))
        scenario = number % 7
        new_style = number % 5 in (0, 1) or number == 119
        new_scenario = (number // 5) % 4
        opening = float(opens[start_pos])
        target_r: float = 2.0
        target_price = np.nan
        entry_price = np.nan
        risk_pips = int(rng.integers(3, 31))
        new_time_exit_at: pd.Timestamp | None = None

        if new_style:
            side = "long" if (number // 5) % 2 == 0 else "short"
            order_type = "market"
            risk_pips = 5000 if number == 119 else 20
            fill = (
                opening + SPREAD + SLIPPAGE
                if side == "long"
                else opening - SLIPPAGE
            )
            stop = (
                fill - risk_pips * PIP
                if side == "long"
                else fill + risk_pips * PIP
            )
            target_r = np.nan
            target_price = np.nan
            time_offset = (
                4
                if new_scenario == 3
                else int(rng.integers(4, 41))
            )
            if number == 119:
                new_time_exit_at = index[-1] + pd.Timedelta(minutes=15)
            else:
                new_time_exit_at = index[start_pos + time_offset]

            set_candle(
                start_pos,
                opening,
                opening + 2 * PIP,
                opening - 2 * PIP,
                opening,
            )
            if number != 119 and new_scenario == 1:
                if side == "long":
                    set_candle(
                        start_pos + 1,
                        opening,
                        opening + 2 * PIP,
                        stop - PIP,
                        opening,
                    )
                else:
                    set_candle(
                        start_pos + 1,
                        opening,
                        stop - SPREAD + PIP,
                        opening - 2 * PIP,
                        opening,
                    )
            elif number != 119 and new_scenario == 2:
                if side == "long":
                    gap_open = stop - 3 * PIP
                else:
                    gap_open = stop - SPREAD + 3 * PIP
                set_candle(
                    start_pos + 1,
                    gap_open,
                    gap_open + 2 * PIP,
                    gap_open - 2 * PIP,
                    gap_open + PIP,
                )
            elif number != 119 and new_scenario == 3:
                exit_pos = start_pos + time_offset
                if side == "long":
                    set_candle(
                        exit_pos,
                        float(opens[exit_pos]),
                        float(opens[exit_pos]) + 2 * PIP,
                        stop - PIP,
                        float(opens[exit_pos]),
                    )
                else:
                    set_candle(
                        exit_pos,
                        float(opens[exit_pos]),
                        stop - SPREAD + PIP,
                        float(opens[exit_pos]) - 2 * PIP,
                        float(opens[exit_pos]),
                    )
        elif scenario == 0:
            side = "long" if (number // 7) % 2 == 0 else "short"
            order_type = "limit"
            risk_pips = 20
            entry_price = opening - 2 * PIP if side == "long" else opening + 2 * PIP
            stop = (
                entry_price - risk_pips * PIP
                if side == "long"
                else entry_price + risk_pips * PIP
            )
            target_r = np.nan
            target_price = (
                entry_price + 4 * PIP
                if side == "long"
                else entry_price - 4 * PIP
            )
            if side == "long":
                set_candle(
                    start_pos,
                    opening,
                    entry_price + 10 * PIP,
                    entry_price - 2 * PIP,
                    opening,
                )
                set_candle(
                    start_pos + 1,
                    opening,
                    entry_price + 8 * PIP,
                    entry_price - 3 * PIP,
                    entry_price + 5 * PIP,
                )
            else:
                set_candle(
                    start_pos,
                    opening,
                    entry_price + 2 * PIP,
                    entry_price - 10 * PIP,
                    opening,
                )
                set_candle(
                    start_pos + 1,
                    opening,
                    entry_price + 5 * PIP,
                    entry_price - 8 * PIP,
                    entry_price - 5 * PIP,
                )
        elif scenario == 1:
            side = "long" if (number // 7) % 2 == 0 else "short"
            order_type = "market"
            risk_pips = 10
            fill = opening + SPREAD + SLIPPAGE if side == "long" else opening - SLIPPAGE
            stop = fill - risk_pips * PIP if side == "long" else fill + risk_pips * PIP
            target_distance = 2 * risk_pips * PIP
            target = fill + target_distance if side == "long" else fill - target_distance
            set_candle(start_pos, opening, opening + 2 * PIP, opening - 2 * PIP, opening)
            set_candle(
                start_pos + 1,
                opening,
                target + 2 * PIP if side == "long" else stop + 2 * PIP,
                stop - 2 * PIP if side == "long" else target - 2 * PIP,
                opening,
            )
        elif scenario == 2:
            side = "short"
            order_type = "market"
            risk_pips = 10
            fill = opening - SLIPPAGE
            stop = fill + risk_pips * PIP
            set_candle(start_pos, opening, opening + 2 * PIP, opening - 2 * PIP, opening)
            set_candle(
                start_pos + 1,
                opening,
                stop - 0.5 * PIP,
                opening - 2 * PIP,
                opening,
            )
            for offset in range(2, 5):
                set_candle(
                    start_pos + offset,
                    opening,
                    opening + 2 * PIP,
                    opening - 2 * PIP,
                    opening,
                )
        elif scenario == 3:
            side = "long"
            order_type = "market"
            risk_pips = 10
            fill = opening + SPREAD + SLIPPAGE
            stop = fill - risk_pips * PIP
            set_candle(start_pos, opening, opening + 2 * PIP, opening - 2 * PIP, opening)
            gap_open = stop - 3 * PIP
            set_candle(
                start_pos + 1,
                gap_open,
                gap_open + 2 * PIP,
                gap_open - 2 * PIP,
                gap_open + PIP,
            )
        elif scenario == 4:
            side = "long"
            order_type = "limit"
            valid = 1
            risk_pips = 10
            entry_price = opening
            stop = entry_price - risk_pips * PIP
            set_candle(
                start_pos,
                opening,
                opening + 3 * PIP,
                entry_price - 0.5 * PIP,
                opening,
            )
            for offset in range(1, 5):
                set_candle(
                    start_pos + offset,
                    opening,
                    opening + 2 * PIP,
                    opening - 2 * PIP,
                    opening,
                )
        elif scenario == 5:
            side = "short"
            order_type = "market"
            risk_pips = 20
            fill = opening - SLIPPAGE
            stop = fill + risk_pips * PIP
            target_r = np.nan
            target_price = fill - 5 * PIP
            set_candle(start_pos, opening, opening + 2 * PIP, opening - 2 * PIP, opening)
            set_candle(
                start_pos + 1,
                opening,
                opening + 2 * PIP,
                target_price - 0.5 * PIP,
                opening,
            )
            for offset in range(2, 5):
                set_candle(
                    start_pos + offset,
                    opening,
                    opening + 2 * PIP,
                    opening - 2 * PIP,
                    opening,
                )
        else:
            side = "long" if number % 2 == 0 else "short"
            order_type = "market" if rng.integers(0, 2) else "limit"
            offset_pips = float(rng.uniform(-2.0, 2.0))
            entry_price = (
                opening + offset_pips * PIP
                if side == "long"
                else opening - offset_pips * PIP
            )
            stop = (
                entry_price - risk_pips * PIP
                if side == "long"
                else entry_price + risk_pips * PIP
            )
            target_r = float(rng.choice([1.5, 2.0]))
            if rng.integers(0, 4) == 0:
                target_r = np.nan
                distance = risk_pips * float(rng.choice([1.5, 2.0])) * PIP
                target_price = (
                    entry_price + distance
                    if side == "long"
                    else entry_price - distance
                )

        if order_type == "market":
            entry_price = np.nan
        plan = {
            "plan_id": number,
            "side": side,
            "order_type": order_type,
            "start_time": index[start_pos],
            "entry_price": entry_price,
            "stop_price": float(stop),
            "target_r": float(target_r),
            "target_price": float(target_price),
            "valid_candles": valid,
            "time_exit_at": (
                new_time_exit_at
                if new_style
                else index[start_pos + int(rng.integers(4, 41))]
            ),
        }
        if new_style:
            plan["no_target"] = True
            plan["exit_mode"] = "time_open"
        if not new_style and scenario == 6 and number % 5 == 0:
            plan["min_risk_pips"] = float(rng.integers(2, 8))
            plan["max_risk_pips"] = float(rng.integers(18, 35))
        if number in (117, 118):
            plan["time_exit_at"] = index[start_pos + 4]
        plans.append(plan)
        start_pos += valid + 10
    return df, pd.DataFrame(plans)


def _seed_diagnostics(
    df: pd.DataFrame,
    plans: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, int]:
    counts = {
        "time_open_time": 0,
        "no_target_stop": 0,
        "exit_candle_stop": 0,
        "data_end": 0,
        "skipped_busy": 0,
    }
    results_by_id = {
        result["plan_id"]: result
        for result in results.to_dict(orient="records")
    }
    for plan in plans.to_dict(orient="records"):
        no_target_value = plan.get("no_target", False)
        if not isinstance(no_target_value, (bool, np.bool_)) or not bool(
            no_target_value
        ):
            continue
        result = results_by_id[plan["plan_id"]]
        if result["status"] == "skipped_busy":
            counts["skipped_busy"] += 1
            continue
        if result["status"] != "filled":
            continue
        if result["exit_reason"] == "stop":
            counts["no_target_stop"] += 1
        elif result["exit_reason"] == "data_end":
            counts["data_end"] += 1
        elif result["exit_reason"] == "time":
            counts["time_open_time"] += 1
            exit_pos = int(df.index.get_indexer([result["exit_time"]])[0])
            if plan["side"] == "long":
                would_hit_stop = (
                    float(df["low"].iloc[exit_pos])
                    <= float(plan["stop_price"]) + 1e-9
                )
            else:
                would_hit_stop = (
                    float(df["high"].iloc[exit_pos]) + SPREAD
                    >= float(plan["stop_price"]) - 1e-9
                )
            counts["exit_candle_stop"] += int(would_hit_stop)
    return counts


def test_independent_reference_on_seeded_random_walk() -> None:
    totals = {
        "time_open_time": 0,
        "no_target_stop": 0,
        "exit_candle_stop": 0,
        "data_end": 0,
        "skipped_busy": 0,
    }
    print(
        "seed | time_open exits | no_target stops | exit-candle stop cases "
        "| data_end exits | skipped_busy new-style"
    )
    for seed in (11, 23, 37, 59):
        df, plans = make_random_case(seed)
        actual = run_backtest(df, plans)
        expected = independent_reference(
            list(
                df[["open", "high", "low", "close"]].itertuples(
                    index=False,
                    name=None,
                )
            ),
            list(df.index),
            plans.to_dict(orient="records"),
            SPREAD_PIPS,
            SLIPPAGE_PIPS,
            PIP,
        )

        pd.testing.assert_frame_equal(actual, expected, rtol=1e-9)
        no_target = plans.loc[plans["no_target"].fillna(False)]
        assert len(no_target) == 49
        assert set(no_target["side"]) == {"long", "short"}
        assert set(no_target["order_type"]) == {"market"}
        assert set(no_target["exit_mode"]) == {"time_open"}
        diagnostics = _seed_diagnostics(df, plans, actual)
        print(
            f"{seed} | {diagnostics['time_open_time']} | "
            f"{diagnostics['no_target_stop']} | "
            f"{diagnostics['exit_candle_stop']} | "
            f"{diagnostics['data_end']} | {diagnostics['skipped_busy']}"
        )
        for name, count in diagnostics.items():
            totals[name] += count

    assert totals["time_open_time"] > 0
    assert totals["no_target_stop"] > 0
    assert totals["exit_candle_stop"] > 0
    assert totals["data_end"] > 0
    assert totals["skipped_busy"] > 0
