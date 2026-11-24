from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from src.strategies.candidate_a import find_candidate_a_plans
from tests import test_candidate_a
from tests.helpers_candidate_a_slow import (
    find_candidate_a_plans as slow_find_candidate_a_plans,
)


UTC_COLUMNS = (
    "start_time",
    "time_exit_at",
    "sweep_time",
    "break_time",
    "fvg_formed_at",
)


def normalize_units(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in UTC_COLUMNS:
        if column in result:
            result[column] = pd.Series(
                pd.DatetimeIndex(pd.to_datetime(result[column], utc=True)).as_unit("ns"),
                index=result.index,
            )
    return result


def test_p1_window_start_9am() -> None:
    df = test_candidate_a.make_df(
        [test_candidate_a.Q, test_candidate_a.S,
         test_candidate_a.A1, test_candidate_a.A2]
        + [test_candidate_a.T] * 12
    )
    levels = test_candidate_a.standard_levels(
        df.index,
        overnight_known_at=df.index[0],
    )
    breaks = test_candidate_a.make_breaks(df.index, [("bullish", 3)])
    fvgs = test_candidate_a.make_fvgs(
        df.index,
        [("bullish", 3, 1.1008, 1.1010)],
    )

    default = find_candidate_a_plans(df, levels, breaks, fvgs)
    pd.testing.assert_frame_equal(
        default,
        test_candidate_a.expected_frame([]),
        check_dtype=True,
    )

    actual = find_candidate_a_plans(
        df,
        levels,
        breaks,
        fvgs,
        window_start_et=time(9, 0),
    )
    expected = test_candidate_a.expected_long_plan(
        df.index,
        sweep_pos=1,
        break_pos=3,
        entry=1.1010,
        stop=1.0995,
        risk=15.0,
        formed_pos=3,
        time_exit=df.index[11],
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


def test_p2_setup_deadline_12pm() -> None:
    df = test_candidate_a.make_df(
        [test_candidate_a.Q] * 8
        + [test_candidate_a.S, test_candidate_a.A1, test_candidate_a.A2]
        + [test_candidate_a.T] * 5
    )
    levels = test_candidate_a.standard_levels(df.index)
    breaks = test_candidate_a.make_breaks(df.index, [("bullish", 10)])
    fvgs = test_candidate_a.make_fvgs(
        df.index,
        [("bullish", 10, 1.1008, 1.1010)],
    )

    default = find_candidate_a_plans(df, levels, breaks, fvgs)
    pd.testing.assert_frame_equal(
        default,
        test_candidate_a.expected_frame([]),
        check_dtype=True,
    )

    actual = find_candidate_a_plans(
        df,
        levels,
        breaks,
        fvgs,
        setup_deadline_et=time(12, 0),
        time_exit_et=time(12, 45),
    )
    expected = test_candidate_a.expected_long_plan(
        df.index,
        sweep_pos=8,
        break_pos=10,
        entry=1.1010,
        stop=1.0995,
        risk=15.0,
        formed_pos=10,
        time_exit=df.index[15],
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


def test_p3_time_exit_1245() -> None:
    df, levels, breaks, fvgs = test_candidate_a.h1_inputs()
    actual = find_candidate_a_plans(
        df,
        levels,
        breaks,
        fvgs,
        time_exit_et=time(12, 45),
    )
    expected = test_candidate_a.expected_long_plan(
        df.index,
        time_exit=df.index[15],
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_start_et": "09:30"}, "window_start_et"),
        (
            {
                "window_start_et": time(11, 30),
                "setup_deadline_et": time(11, 30),
            },
            "window_start_et",
        ),
        (
            {
                "setup_deadline_et": time(13, 0),
                "time_exit_et": time(12, 45),
            },
            "window_start_et",
        ),
        (
            {"setup_deadline_et": time(12, 0)},
            "window_start_et",
        ),
    ],
)
def test_p4_validation(
    kwargs: dict[str, object],
    match: str,
) -> None:
    df, levels, breaks, fvgs = test_candidate_a.h1_inputs()
    with pytest.raises(ValueError, match=match):
        find_candidate_a_plans(df, levels, breaks, fvgs, **kwargs)


@pytest.mark.parametrize("seed", [17, 31, 43])
def test_p5_seeded_slow_equivalence(seed: int) -> None:
    kwargs = {
        "window_start_et": time(8, 0),
        "setup_deadline_et": time(12, 30),
        "time_exit_et": time(12, 45),
        "confirm_candles": 12,
        "min_fvg_pips": 1.0,
    }
    for period_name, start, end in (
        ("spring", "2023-03-01", "2023-04-15"),
        ("fall", "2023-10-15", "2023-11-15"),
    ):
        df = test_candidate_a.seeded_market_data(seed, start, end)
        actual = find_candidate_a_plans(df, **kwargs)
        expected = slow_find_candidate_a_plans(df, **kwargs)
        pd.testing.assert_frame_equal(
            normalize_units(actual),
            normalize_units(expected),
            check_dtype=True,
        )
        print(
            f"seed={seed} period={period_name} plans={len(actual)}"
        )
