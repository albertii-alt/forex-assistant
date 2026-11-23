import time

import numpy as np
import pandas as pd
import pytest

from src.detectors.swings import detect_swings


def make_df(high, low, start="2024-01-01 00:00", freq="15min") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=len(high), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": np.asarray(low, dtype=float),
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": np.asarray(high, dtype=float),
            "volume": np.ones(len(high)),
        },
        index=index,
    )


def test_simple_swing_high_and_low_with_left_right_two() -> None:
    high = [1, 2, 1.5, 3, 2.5, 1.2, 2.0, 1.5, 1.8, 1.1]
    low = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.2, 0.8, 0.7]
    df = make_df(high, low)

    result = detect_swings(df, left=2, right=2)

    assert result["is_swing_high"].iloc[[3]].all()
    assert result["is_swing_high"].sum() >= 1
    assert result["is_swing_low"].iloc[[7]].all()
    assert result["is_swing_low"].sum() >= 1


def test_equal_highs_and_lows_only_first_marked() -> None:
    df = make_df(
        [1.0, 3.0, 3.0, 1.0, 1.0, 1.0],
        [1.0, 0.5, 0.5, 0.8, 0.8, 0.9],
    )

    result = detect_swings(df, left=1, right=1)

    assert result["is_swing_high"].iloc[1] and not result["is_swing_high"].iloc[2]
    assert result["is_swing_low"].iloc[1] and not result["is_swing_low"].iloc[2]


def test_edges_are_never_swings() -> None:
    df = make_df(
        [5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 5.0],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.5],
    )

    result = detect_swings(df, left=2, right=2)

    assert not result["is_swing_high"].iloc[:2].any()
    assert not result["is_swing_high"].iloc[-2:].any()
    assert not result["is_swing_low"].iloc[:2].any()
    assert not result["is_swing_low"].iloc[-2:].any()


def test_outside_bar_can_be_both() -> None:
    df = make_df(
        [1.0, 2.0, 1.5, 3.0, 2.0, 1.0, 1.2],
        [1.0, 0.5, 0.8, 0.2, 0.7, 0.9, 1.1],
    )

    result = detect_swings(df, left=2, right=2)

    assert result.loc[result.index[3], "is_swing_high"]
    assert result.loc[result.index[3], "is_swing_low"]


def test_confirmed_at_is_right_candles_later() -> None:
    df = make_df([1.0, 1.0, 1.0, 2.0, 1.0, 1.0], [1.0] * 6)

    result = detect_swings(df, left=2, right=2)

    swing_positions = np.flatnonzero(result["is_swing_high"].to_numpy())
    for position in swing_positions:
        assert result["confirmed_at"].iloc[position] == df.index[position + 2]


def test_input_not_mutated_and_output_index_equal() -> None:
    df = make_df([1.0, 2.0, 1.5, 1.8, 1.2], [0.9, 0.8, 0.7, 0.6, 0.5])
    original = df.copy(deep=True)

    result = detect_swings(df, left=2, right=2)

    pd.testing.assert_frame_equal(df, original)
    assert result.index.equals(df.index)
    assert str(result.index.tz).upper() == "UTC"


def test_invalid_parameters_and_columns() -> None:
    df = make_df([1.0, 2.0, 1.5], [0.9, 0.8, 0.7])

    with pytest.raises(ValueError):
        detect_swings(df, left=0)
    with pytest.raises(ValueError):
        detect_swings(df, right=0)
    with pytest.raises(ValueError):
        detect_swings(df.drop(columns="high"))


def test_weekend_style_gap_is_counted_in_candles() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-05 21:00", "2024-01-05 21:15", "2024-01-05 21:30",
         "2024-01-07 21:00", "2024-01-07 21:15", "2024-01-07 21:30"],
        tz="UTC",
    )
    df = pd.DataFrame(
        {
            "high": [1.0, 2.0, 1.5, 1.0, 2.5, 1.0],
            "low": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        },
        index=index,
    )

    result = detect_swings(df, left=1, right=1)

    assert result["is_swing_high"].iloc[1]
    assert str(result["confirmed_at"].dtype) == "datetime64[ns, UTC]"


def assert_no_lookahead(full: pd.DataFrame, truncated: pd.DataFrame, left: int, right: int) -> None:
    full_result = detect_swings(full, left=left, right=right)
    truncated_result = detect_swings(truncated, left=left, right=right)
    cutoff = truncated.index[-1]

    for timestamp in full_result.index:
        full_row = full_result.loc[timestamp]
        confirmed = full_row["confirmed_at"]
        if pd.notna(confirmed) and confirmed <= cutoff:
            assert timestamp in truncated_result.index
            truncated_row = truncated_result.loc[timestamp]
            assert full_row["is_swing_high"] == truncated_row["is_swing_high"]
            assert full_row["is_swing_low"] == truncated_row["is_swing_low"]
            assert confirmed == truncated_row["confirmed_at"]
        elif timestamp in full_result.index and (
            full_row["is_swing_high"] or full_row["is_swing_low"]
        ):
            assert pd.isna(confirmed) or confirmed > cutoff


def test_hand_made_no_lookahead_consistency() -> None:
    df = make_df(
        [1.0, 2.0, 1.5, 3.0, 2.0, 1.0, 1.5, 1.2, 2.5, 1.0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8],
    )
    cutoff = 6

    assert_no_lookahead(df, df.iloc[: cutoff + 1], left=2, right=2)


def test_real_size_seeded_random_walk_no_lookahead() -> None:
    rng = np.random.default_rng(20241123)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, 500))
    high = close + rng.uniform(0.0001, 0.0010, 500)
    low = close - rng.uniform(0.0001, 0.0010, 500)
    df = make_df(high, low)

    for cutoff in (50, 127, 249, 499):
        truncated = df.iloc[: cutoff + 1]
        assert_no_lookahead(df, truncated, left=5, right=5)
