import time

import numpy as np
import pandas as pd
import pytest

from src.detectors.fvg import detect_fvgs


def make_df(high, low, close=None, start="2024-01-01 00:00", freq="15min"):
    close = high if close is None else close
    index = pd.date_range(start=start, periods=len(high), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": np.asarray(close, dtype=float),
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": np.asarray(close, dtype=float),
            "volume": np.ones(len(high)),
        },
        index=index,
    )


def test_one_bullish_and_one_bearish_gap_known_answers() -> None:
    df = make_df(
        high=[1.00, 1.10, 1.20, 1.30, 1.20, 1.10],
        low=[0.90, 1.00, 1.12, 1.30, 1.20, 1.00],
        close=[1.00, 1.10, 1.15, 1.25, 1.15, 1.05],
    )

    result = detect_fvgs(df)

    assert set(result["direction"]) == {"bullish", "bearish"}
    bullish = result.loc[result["direction"] == "bullish"].iloc[0]
    bearish = result.loc[result["direction"] == "bearish"].iloc[0]
    assert bullish["zone_low"] == 1.00
    assert bullish["zone_high"] == 1.12
    assert bullish["size"] == pytest.approx(0.12)
    assert bearish["zone_low"] == 1.10
    assert bearish["zone_high"] == 1.30
    assert bearish["size"] == pytest.approx(0.20)


def test_overlapping_candles_and_equality_produce_no_gap() -> None:
    overlapping = make_df(
        high=[1.10, 1.10, 1.10],
        low=[1.00, 1.00, 1.00],
    )
    equality = make_df(
        high=[1.10, 1.00, 1.20],
        low=[1.00, 0.90, 1.10],
    )

    assert detect_fvgs(overlapping).empty
    assert detect_fvgs(equality).empty


def test_min_gap_keeps_exact_size_and_drops_smaller() -> None:
    df = make_df(
        high=[1.0000, 1.0010, 1.0022],
        low=[0.9990, 1.0005, 1.0002],
    )

    exact = detect_fvgs(df, min_gap=0.0002)
    smaller = detect_fvgs(df, min_gap=0.0002001)

    assert len(exact) == 1
    assert exact.iloc[0]["size"] == pytest.approx(0.0002)
    assert smaller.empty


def test_formed_at_and_c1_time_are_c3_and_c1_timestamps() -> None:
    df = make_df(
        high=[1.00, 1.10, 1.20],
        low=[0.90, 1.00, 1.12],
    )

    result = detect_fvgs(df)

    assert len(result) == 1
    assert result.iloc[0]["c1_time"] == df.index[0]
    assert result.iloc[0]["formed_at"] == df.index[2]


def test_touch_starts_only_after_formation_and_can_be_missing() -> None:
    df = make_df(
        high=[1.10, 1.10, 1.10, 1.05, 1.30],
        low=[1.00, 1.05, 1.12, 1.02, 1.20],
        close=[1.05, 1.08, 1.15, 1.04, 1.25],
    )

    result = detect_fvgs(df)

    bullish = result.loc[result["direction"] == "bullish"]
    if not bullish.empty:
        row = bullish.iloc[0]
        assert pd.notna(row["touched_at"])
        assert row["touched_at"] != df.index[2]

    no_touch = make_df(
        high=[1.10, 1.10, 1.10, 1.50, 1.60],
        low=[1.00, 1.05, 1.12, 1.40, 1.50],
        close=[1.05, 1.08, 1.15, 1.45, 1.55],
    )
    assert detect_fvgs(no_touch).iloc[0]["touched_at"] is pd.NaT or pd.isna(
        detect_fvgs(no_touch).iloc[0]["touched_at"]
    )


def test_closed_through_requires_close_beyond_far_side() -> None:
    bullish = make_df(
        high=[1.10, 1.10, 1.10, 1.09, 1.08, 1.09],
        low=[1.00, 1.05, 1.12, 1.07, 0.95, 1.00],
        close=[1.05, 1.08, 1.15, 1.105, 0.99, 1.01],
    )
    result = detect_fvgs(bullish)
    row = result.loc[result["direction"] == "bullish"].iloc[0]
    assert pd.isna(row["closed_through_at"]) is False
    assert row["closed_through_at"] == bullish.index[4]

    bearish = make_df(
        high=[1.30, 1.20, 1.10, 1.20, 1.31, 1.25],
        low=[1.20, 1.10, 1.00, 1.15, 1.21, 1.10],
        close=[1.25, 1.15, 1.05, 1.18, 1.32, 1.20],
    )
    result = detect_fvgs(bearish)
    row = result.loc[result["direction"] == "bearish"].iloc[0]
    assert pd.notna(row["closed_through_at"])
    assert row["closed_through_at"] == bearish.index[4]


def test_spans_gap_across_weekend_only_when_spacing_is_exceeded() -> None:
    across = make_df(
        high=[1.10, 1.10, 1.10],
        low=[1.00, 1.05, 1.12],
        start="2024-01-05 21:00",
        freq="15min",
    )
    across = pd.concat(
        [
            across.iloc[:2],
            make_df(
                high=[1.10],
                low=[1.12],
                start="2024-01-07 21:00",
                freq="15min",
            ),
        ]
    )
    across.index = pd.DatetimeIndex(across.index)

    result = detect_fvgs(across)
    assert not result.empty
    assert bool(result.iloc[0]["spans_gap"]) is True

    regular = make_df(
        high=[1.10, 1.10, 1.10],
        low=[1.00, 1.05, 1.12],
    )
    assert not bool(detect_fvgs(regular).iloc[0]["spans_gap"])


def test_fewer_than_three_is_empty_and_input_is_not_mutated() -> None:
    df = make_df(high=[1.10, 1.10], low=[1.00, 1.05])
    original = df.copy(deep=True)

    result = detect_fvgs(df)

    assert list(result.columns) == [
        "direction", "c1_time", "formed_at", "zone_low", "zone_high",
        "size", "spans_gap", "touched_at", "closed_through_at",
    ]
    assert result.empty
    pd.testing.assert_frame_equal(df, original)


def test_invalid_params_and_columns() -> None:
    df = make_df(high=[1.10, 1.10, 1.10], low=[1.00, 1.05, 1.12])
    with pytest.raises(ValueError):
        detect_fvgs(df, min_gap=-0.0001)
    with pytest.raises(ValueError):
        detect_fvgs(df.drop(columns="close"))


def assert_no_lookahead(full, truncated) -> None:
    full_result = detect_fvgs(full)
    truncated_result = detect_fvgs(truncated)
    cutoff = truncated.index[-1]

    full_by_formation = {
        row["formed_at"]: row for _, row in full_result.iterrows()
    }
    truncated_by_formation = {
        row["formed_at"]: row for _, row in truncated_result.iterrows()
    }

    for formed, row in full_by_formation.items():
        if formed <= cutoff:
            assert formed in truncated_by_formation
            other = truncated_by_formation[formed]
            for column in ("direction", "c1_time", "zone_low", "zone_high", "size", "spans_gap"):
                assert row[column] == other[column]
            for column in ("touched_at", "closed_through_at"):
                if pd.notna(row[column]) and row[column] > cutoff:
                    assert pd.isna(other[column])
                elif pd.notna(row[column]):
                    assert row[column] == other[column]
        else:
            assert formed > cutoff

    for formed in truncated_by_formation:
        assert formed in full_by_formation


def test_hand_made_no_lookahead_consistency() -> None:
    df = make_df(
        high=[1.10, 1.10, 1.10, 1.30, 1.20, 1.10, 1.20, 1.30, 1.40],
        low=[1.00, 1.05, 1.12, 1.20, 1.10, 1.00, 1.05, 1.15, 1.20],
        close=[1.05, 1.08, 1.15, 1.25, 1.15, 1.05, 1.10, 1.25, 1.35],
    )
    full = detect_fvgs(df)
    cutoff = 5
    truncated = df.iloc[: cutoff + 1]
    truncated_result = detect_fvgs(truncated)

    assert_no_lookahead(df, truncated)
    for _, row in full.iterrows():
        if row["formed_at"] > truncated.index[-1]:
            assert row["formed_at"] > truncated.index[-1]
        else:
            assert row["formed_at"] in set(truncated_result["formed_at"])


def test_seeded_random_walk_no_lookahead_at_several_cutoffs() -> None:
    rng = np.random.default_rng(20241123)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, 500))
    high = close + rng.uniform(0.0001, 0.0010, 500)
    low = close - rng.uniform(0.0001, 0.0010, 500)
    df = make_df(high, low, close=close)

    for cutoff in (50, 127, 249, 499):
        full_result = detect_fvgs(df)
        truncated = df.iloc[: cutoff + 1]
        truncated_result = detect_fvgs(truncated)
        cutoff_time = truncated.index[-1]

        for _, row in full_result.iterrows():
            if row["formed_at"] <= cutoff_time:
                matches = truncated_result[
                    truncated_result["formed_at"] == row["formed_at"]
                ]
                assert len(matches) == 1
                other = matches.iloc[0]
                for column in ("direction", "c1_time", "zone_low", "zone_high", "size", "spans_gap"):
                    assert row[column] == other[column]
                for column in ("touched_at", "closed_through_at"):
                    if pd.notna(row[column]) and row[column] > cutoff_time:
                        assert pd.isna(other[column])
                    elif pd.notna(row[column]):
                        assert row[column] == other[column]
            else:
                assert row["formed_at"] > cutoff_time
