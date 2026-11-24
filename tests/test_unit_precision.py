from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.data.resample_et import resample_ohlcv_et
from src.detectors.fvg import detect_fvgs
from src.detectors.session_levels import detect_session_levels
from src.detectors.structure import detect_structure_breaks
from src.strategies.candidate_a import find_candidate_a_plans
from tests import test_backtest_engine
from tests import test_candidate_a


UTC_COLUMNS = (
    "start_time",
    "time_exit_at",
    "sweep_time",
    "break_time",
    "fvg_formed_at",
)


def normalize_plan_units(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in UTC_COLUMNS:
        if column in result:
            result[column] = pd.Series(
                pd.DatetimeIndex(pd.to_datetime(result[column], utc=True)).as_unit("ns"),
                index=result.index,
            )
    return result


@pytest.mark.parametrize("case", ["h1", "h2", "h4a", "h13"])
def test_candidate_hand_fixture_microsecond_index(case: str) -> None:
    if case == "h1":
        df, levels, breaks, fvgs = test_candidate_a.h1_inputs()
    elif case == "h2":
        candles = (
            [test_candidate_a.Q_SHORT] * 4
            + [
                test_candidate_a.S_SHORT,
                test_candidate_a.A1_SHORT,
                test_candidate_a.A2_SHORT,
                test_candidate_a.T_SHORT,
            ]
            + [test_candidate_a.T_SHORT] * 8
        )
        df = test_candidate_a.make_df(candles)
        levels = test_candidate_a.standard_levels(df.index, short=True)
        breaks = test_candidate_a.make_breaks(df.index, [("bearish", 6)])
        fvgs = test_candidate_a.make_fvgs(
            df.index,
            [("bearish", 6, 1.0990, 1.0992)],
        )
    elif case == "h4a":
        df = test_candidate_a.make_df(
            [test_candidate_a.Q] * 7
            + [test_candidate_a.S, test_candidate_a.A1, test_candidate_a.A2]
            + [test_candidate_a.T] * 6
        )
        levels = test_candidate_a.standard_levels(df.index)
        breaks = test_candidate_a.make_breaks(df.index, [("bullish", 9)])
        fvgs = test_candidate_a.make_fvgs(
            df.index,
            [("bullish", 9, 1.1008, 1.1010)],
        )
    else:
        df = test_candidate_a.make_df(
            [test_candidate_a.Q, test_candidate_a.Q, test_candidate_a.S,
             test_candidate_a.A1, test_candidate_a.A2]
            + [test_candidate_a.T] * 11,
            start="2024-07-09 13:00",
        )
        levels = test_candidate_a.standard_levels(
            df.index,
            trading_day="2024-07-09",
            overnight_known_at=df.index[2],
            prev_day_known_at=pd.Timestamp("2024-07-08 21:00", tz="UTC"),
        )
        breaks = test_candidate_a.make_breaks(df.index, [("bullish", 4)])
        fvgs = test_candidate_a.make_fvgs(
            df.index,
            [("bullish", 4, 1.1008, 1.1010)],
        )

    expected = find_candidate_a_plans(df, levels, breaks, fvgs)
    us_df = df.copy()
    us_df.index = us_df.index.as_unit("us")
    actual = find_candidate_a_plans(us_df, levels, breaks, fvgs)

    pd.testing.assert_frame_equal(
        normalize_plan_units(actual),
        normalize_plan_units(expected),
        check_dtype=True,
    )


def normalize_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        dtype = str(result[column].dtype)
        if "datetime64" in dtype:
            result[column] = pd.to_datetime(result[column], utc=True).dt.as_unit("ns")
    if isinstance(result.index, pd.DatetimeIndex):
        result.index = result.index.as_unit("ns")
    return result


def test_backtest_t1_microsecond_index() -> None:
    candles = test_backtest_engine.t1_candles()
    df = test_backtest_engine.make_df(candles)
    ns_plan = test_backtest_engine.make_plan(df.index)
    ns_result = run_backtest(df, pd.DataFrame([ns_plan]))

    us_df = df.copy()
    us_df.index = us_df.index.as_unit("us")
    us_plan = test_backtest_engine.make_plan(us_df.index)
    us_result = run_backtest(us_df, pd.DataFrame([us_plan]))

    pd.testing.assert_frame_equal(
        normalize_datetime_columns(us_result),
        normalize_datetime_columns(ns_result),
        check_dtype=True,
    )


def detector_fixture() -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-09 14:00",
        periods=20,
        freq="15min",
        tz="UTC",
    )
    candles = [
        (1.1000, 1.1010, 1.0990, 1.1005),
        (1.1005, 1.1020, 1.1000, 1.1015),
        (1.1015, 1.1030, 1.1010, 1.1025),
        (1.1025, 1.1040, 1.1020, 1.1035),
        (1.1035, 1.1045, 1.1010, 1.1015),
        (1.1015, 1.1025, 1.1005, 1.1008),
        (1.1008, 1.1028, 1.0998, 1.1020),
        (1.1020, 1.1035, 1.1015, 1.1030),
        (1.1030, 1.1045, 1.1025, 1.1040),
        (1.1040, 1.1050, 1.1035, 1.1045),
    ] * 2
    return pd.DataFrame(
        candles,
        columns=["open", "high", "low", "close", "volume"]
        if False
        else ["open", "high", "low", "close"],
        index=index,
    ).assign(volume=100)


def test_detectors_microsecond_index() -> None:
    ns_df = detector_fixture()
    us_df = ns_df.copy()
    us_df.index = us_df.index.as_unit("us")

    ns_structure = detect_structure_breaks(ns_df, left=1, right=1)
    us_structure = detect_structure_breaks(us_df, left=1, right=1)
    pd.testing.assert_frame_equal(
        normalize_datetime_columns(us_structure),
        normalize_datetime_columns(ns_structure),
        check_dtype=True,
    )

    ns_fvgs = detect_fvgs(ns_df)
    us_fvgs = detect_fvgs(us_df)
    pd.testing.assert_frame_equal(
        normalize_datetime_columns(us_fvgs),
        normalize_datetime_columns(ns_fvgs),
        check_dtype=True,
    )

    ns_levels = detect_session_levels(ns_df)
    us_levels = detect_session_levels(us_df)
    pd.testing.assert_frame_equal(
        normalize_datetime_columns(us_levels),
        normalize_datetime_columns(ns_levels),
        check_dtype=True,
    )

    ns_resampled = resample_ohlcv_et(ns_df, "4h")
    us_resampled = resample_ohlcv_et(us_df, "4h")
    pd.testing.assert_frame_equal(
        normalize_datetime_columns(us_resampled),
        normalize_datetime_columns(ns_resampled),
        check_dtype=True,
    )


@pytest.mark.parametrize("seed", [17, 31, 43])
def test_seeded_candidate_microsecond_index(seed: int) -> None:
    for _, start, end in (
        ("spring", "2023-03-01", "2023-04-15"),
        ("fall", "2023-10-15", "2023-11-15"),
    ):
        ns_df = test_candidate_a.seeded_market_data(seed, start, end)
        us_df = ns_df.copy()
        us_df.index = us_df.index.as_unit("us")
        expected = find_candidate_a_plans(ns_df)
        actual = find_candidate_a_plans(us_df)
        pd.testing.assert_frame_equal(
            normalize_plan_units(actual),
            normalize_plan_units(expected),
            check_dtype=True,
        )
