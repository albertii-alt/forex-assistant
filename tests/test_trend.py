from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.resample_et import align_htf_to_ltf, resample_ohlcv_et
from src.detectors.swings import detect_swings
from src.detectors.trend import htf_trend_state
from tests.test_indicators import make_market


def make_hand_frame(high: list[float], low: list[float]) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-08 22:00",
        periods=len(high),
        freq="4h",
        tz="UTC",
    )
    return pd.DataFrame({"high": high, "low": low}, index=index)


def reference_trend(
    df: pd.DataFrame,
    swings: pd.DataFrame,
) -> pd.DataFrame:
    index = df.index.as_unit("ns")
    swing_rows = swings.to_dict(orient="records")
    high_values = df["high"].astype(float).tolist()
    low_values = df["low"].astype(float).tolist()
    confirmed_positions = []
    for swing_pos, row in enumerate(swing_rows):
        value = row["confirmed_at"]
        if pd.isna(value):
            continue
        confirmed = pd.Timestamp(value).as_unit("ns")
        if confirmed in index:
            confirmed_positions.append((swing_pos, index.get_loc(confirmed)))

    states = []
    for candle_pos in range(len(df)):
        confirmed_highs = []
        confirmed_lows = []
        for swing_pos, confirmed_pos in confirmed_positions:
            if confirmed_pos <= candle_pos:
                row = swing_rows[swing_pos]
                if row["is_swing_high"]:
                    confirmed_highs.append((swing_pos, high_values[swing_pos]))
                if row["is_swing_low"]:
                    confirmed_lows.append((swing_pos, low_values[swing_pos]))
        confirmed_highs.sort()
        confirmed_lows.sort()
        if len(confirmed_highs) < 2 or len(confirmed_lows) < 2:
            states.append("none")
            continue
        h1, h2 = confirmed_highs[-2][1], confirmed_highs[-1][1]
        l1, l2 = confirmed_lows[-2][1], confirmed_lows[-1][1]
        if h2 > h1 and l2 > l1:
            states.append("up")
        elif h2 < h1 and l2 < l1:
            states.append("down")
        else:
            states.append("none")
    return pd.DataFrame({"state": states}, index=df.index)


@pytest.mark.parametrize(
    ("high", "low", "expected"),
    [
        (
            [1, 3, 2, 5, 4, 7, 6, 8, 7],
            [0, 2, 1, 4, 3, 6, 5, 7, 6],
            ["none"] * 5 + ["up"] * 4,
        ),
        (
            [10, 8, 9, 6, 7, 4, 5, 3, 4],
            [9, 7, 8, 5, 6, 3, 4, 2, 3],
            ["none"] * 5 + ["down"] * 4,
        ),
        (
            [1, 3, 2, 5, 4, 7, 6, 8, 7],
            [0, -2, -1, -4, -3, -6, -5, -8, -7],
            ["none"] * 9,
        ),
    ],
)
def test_hand_computed_trend_states(
    high: list[float],
    low: list[float],
    expected: list[str],
) -> None:
    df = make_hand_frame(high, low)
    result = htf_trend_state(df, left=1, right=1)
    assert result["state"].tolist() == expected


@pytest.mark.parametrize("seed", [11, 23, 37, 59])
@pytest.mark.parametrize(("left", "right"), [(1, 1), (2, 2), (3, 3)])
def test_trend_loop_reference(seed: int, left: int, right: int) -> None:
    df = make_market(seed)
    swings = detect_swings(df, left=left, right=right)
    expected = reference_trend(df, swings)
    actual = htf_trend_state(df, swings=swings, left=left, right=right)
    pd.testing.assert_frame_equal(actual, expected)


def test_precomputed_swings_and_input_immutability() -> None:
    df = make_hand_frame(
        [1, 3, 2, 5, 4, 7, 6, 8, 7],
        [0, 2, 1, 4, 3, 6, 5, 7, 6],
    )
    before = df.copy(deep=True)
    swings = detect_swings(df, left=1, right=1)
    computed = htf_trend_state(df, left=1, right=1)
    supplied = htf_trend_state(df, swings=swings, left=1, right=1)
    pd.testing.assert_frame_equal(computed, supplied)
    pd.testing.assert_frame_equal(df, before)


def test_trend_validation() -> None:
    df = make_hand_frame(
        [1, 3, 2, 5, 4, 7, 6, 8, 7],
        [0, 2, 1, 4, 3, 6, 5, 7, 6],
    )
    with pytest.raises(ValueError, match="left"):
        htf_trend_state(df, left=0)
    with pytest.raises(ValueError, match="right"):
        htf_trend_state(df, right=1.5)
    with pytest.raises(ValueError, match="UTC"):
        htf_trend_state(
            df.set_axis(pd.date_range("2024-01-01", periods=len(df), freq="4h"))
        )
    with pytest.raises(ValueError, match="sorted"):
        htf_trend_state(df.iloc[::-1])
    with pytest.raises(ValueError, match="high"):
        htf_trend_state(df.drop(columns="high"))
    swings = detect_swings(df, left=1, right=1)
    with pytest.raises(ValueError, match="index"):
        htf_trend_state(df, swings=swings.set_axis(df.index + pd.Timedelta(hours=1)))
    with pytest.raises(ValueError, match="missing"):
        htf_trend_state(df, swings=pd.DataFrame(index=df.index))


def test_trend_no_lookahead() -> None:
    df = make_market(23)
    full = htf_trend_state(df, left=2, right=2)
    for cutoff in (150, 400, 700):
        prefix = df.iloc[: cutoff + 1]
        truncated = htf_trend_state(prefix, left=2, right=2)
        pd.testing.assert_frame_equal(
            truncated,
            full.iloc[: cutoff + 1],
            check_dtype=True,
        )


def test_hand_trend_preserves_microsecond_index() -> None:
    df = make_hand_frame(
        [1, 3, 2, 5, 4, 7, 6, 8, 7],
        [0, 2, 1, 4, 3, 6, 5, 7, 6],
    )
    us_df = df.copy()
    us_df.index = us_df.index.as_unit("us")
    expected = htf_trend_state(df, left=1, right=1)
    actual = htf_trend_state(us_df, left=1, right=1)
    expected.index = expected.index.as_unit("ns")
    actual.index = actual.index.as_unit("ns")
    pd.testing.assert_frame_equal(actual, expected)


def test_align_uses_only_closed_four_hour_trend() -> None:
    high = [1, 3, 2, 5, 4, 7, 6, 8, 7]
    low = [value - 1 for value in high]
    start = pd.Timestamp("2024-01-08 22:00", tz="UTC")
    source_index = pd.date_range(start, periods=9 * 16, freq="15min")
    opens = []
    highs = []
    lows = []
    closes = []
    for value_high, value_low in zip(high, low):
        midpoint = (value_high + value_low) / 2
        opens.extend([midpoint] * 16)
        closes.extend([midpoint] * 16)
        highs.extend([value_high] * 16)
        lows.extend([value_low] * 16)
    ltf = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(len(source_index)),
        },
        index=source_index,
    )
    htf = resample_ohlcv_et(ltf, "4h")
    htf["state"] = htf_trend_state(htf, left=1, right=1)["state"]
    htf["close_time"] = htf.index + pd.Timedelta(hours=4)
    aligned = align_htf_to_ltf(ltf.index, htf, columns=["state"])

    first_close = htf.index[0] + pd.Timedelta(hours=4)
    after_close = ltf.index[ltf.index >= first_close]
    assert aligned.loc[ltf.index < first_close, "state"].isna().all()
    assert aligned.loc[after_close[0], "state"] == "none"

    fifth_close = htf.index[5] + pd.Timedelta(hours=4)
    first_after_fifth_close = ltf.index[ltf.index >= fifth_close][0]
    assert aligned.loc[first_after_fifth_close, "state"] == "up"
