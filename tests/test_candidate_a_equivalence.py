from __future__ import annotations

import pandas as pd
import pytest

from src.strategies import candidate_a
from tests import test_candidate_a
from tests import helpers_candidate_a_slow as slow


def _compare(
    df: pd.DataFrame,
    name: str,
    *,
    levels: pd.DataFrame | None = None,
    breaks: pd.DataFrame | None = None,
    fvgs: pd.DataFrame | None = None,
    **kwargs: object,
) -> None:
    actual = candidate_a.find_candidate_a_plans(
        df,
        levels=levels,
        breaks=breaks,
        fvgs=fvgs,
        **kwargs,
    )
    expected = slow.find_candidate_a_plans(
        df,
        levels=levels,
        breaks=breaks,
        fvgs=fvgs,
        **kwargs,
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
    print(f"{name}: plans={len(actual)}")


@pytest.mark.parametrize("seed", [17, 31, 43])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "swing_left": 2,
            "swing_right": 2,
            "confirm_candles": 4,
            "min_risk_pips": 3.0,
            "max_risk_pips": 40.0,
        },
        {
            "min_fvg_pips": 0.0,
            "valid_candles": 4,
        },
    ],
)
def test_seeded_equivalence(seed: int, kwargs: dict[str, object]) -> None:
    for period_name, start, end in (
        ("spring", "2023-03-01", "2023-04-15"),
        ("fall", "2023-10-15", "2023-11-15"),
    ):
        df = test_candidate_a.seeded_market_data(seed, start, end)
        _compare(
            df,
            f"seed={seed} period={period_name}",
            **kwargs,
        )
        actual = candidate_a.find_candidate_a_plans(df, **kwargs)
        assert len(actual) >= 8, (
            f"seed {seed} {period_name} produced only {len(actual)} plans"
        )


def test_h1_h2_and_h6_equivalence() -> None:
    df, levels, breaks, fvgs = test_candidate_a.h1_inputs()
    _compare(
        df,
        "H1",
        levels=levels,
        breaks=breaks,
        fvgs=fvgs,
    )

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
    _compare(
        df,
        "H2",
        levels=levels,
        breaks=breaks,
        fvgs=fvgs,
    )

    df, levels, breaks, _ = test_candidate_a.h1_inputs()
    fvgs = test_candidate_a.make_fvgs(
        df.index,
        [
            ("bullish", 5, 1.1000, 1.1004),
            ("bullish", 6, 1.1008, 1.1010),
            ("bullish", 7, 1.1012, 1.1020),
            ("bearish", 6, 1.1050, 1.1060),
            ("bullish", 4, 1.1011, 1.1015),
        ],
    )
    _compare(
        df,
        "H6",
        levels=levels,
        breaks=breaks,
        fvgs=fvgs,
    )
