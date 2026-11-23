import numpy as np
import pandas as pd
import pytest

from src.detectors.structure import detect_structure_breaks
from src.detectors.swings import detect_swings


def make_df(close, high=None, low=None, start="2024-01-01 00:00", freq="15min"):
    close = np.asarray(close, dtype=float)
    high = close if high is None else np.asarray(high, dtype=float)
    low = close if low is None else np.asarray(low, dtype=float)
    index = pd.date_range(
        start=start,
        periods=len(close),
        freq=freq,
        tz="UTC",
    ).as_unit("ns")
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(len(close)),
        },
        index=index,
    )


def empty_structure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "level": pd.Series(dtype="float64"),
            "swing_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "swing_confirmed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "broke_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "kind": pd.Series(dtype="object"),
        }
    )


def exact_frame(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_structure_frame()
    frame = pd.DataFrame(
        rows,
        columns=[
            "direction",
            "level",
            "swing_time",
            "swing_confirmed_at",
            "broke_at",
            "kind",
        ],
    )
    for column in ("swing_time", "swing_confirmed_at", "broke_at"):
        frame[column] = pd.Series(frame[column], dtype="datetime64[ns, UTC]")
    return frame


def test_case_a_basic_bullish_break() -> None:
    # Pos1 is the high=1.5 swing, confirmed at pos2. The close first exceeds
    # 1.5 at pos4, which is index[4].
    df = make_df(
        high=[1.0, 1.5, 1.2, 1.3, 1.6],
        low=[0.9, 1.4, 1.1, 1.2, 1.3],
        close=[1.0, 1.4, 1.15, 1.25, 1.55],
    )
    result = detect_structure_breaks(df, left=1, right=1)
    expected = exact_frame(df, [
        {
            "direction": "bullish",
            "level": 1.5,
            "swing_time": df.index[1],
            "swing_confirmed_at": df.index[2],
            "broke_at": df.index[4],
            "kind": "first",
        }
    ])
    pd.testing.assert_frame_equal(result, expected)


def test_case_b_wick_only_does_not_break() -> None:
    # Pos1 is confirmed at pos2 and remains active. Candle pos3 wicks to 1.6
    # but closes 1.25, so no break. Pos4 is the first close above 1.5.
    df = make_df(
        high=[1.0, 1.5, 1.2, 1.6, 1.7],
        low=[0.9, 1.4, 1.1, 1.2, 1.3],
        close=[1.0, 1.4, 1.15, 1.25, 1.55],
    )
    result = detect_structure_breaks(df, left=1, right=1)
    expected = exact_frame(df, [
        {
            "direction": "bullish",
            "level": 1.5,
            "swing_time": df.index[1],
            "swing_confirmed_at": df.index[2],
            "broke_at": df.index[4],
            "kind": "first",
        }
    ])
    pd.testing.assert_frame_equal(result, expected)


def test_case_c_close_equal_to_level_is_not_a_break() -> None:
    # Pos1 is 1.5 and is confirmed at pos2. Pos4 closes exactly 1.5, not
    # strictly above, so the only candidate produces no break.
    df = make_df(
        high=[1.0, 1.5, 1.2, 1.3, 1.6],
        low=[0.9, 1.4, 1.1, 1.2, 1.3],
        close=[1.0, 1.4, 1.15, 1.25, 1.5],
    )
    result = detect_structure_breaks(df, left=1, right=1)
    pd.testing.assert_frame_equal(result, empty_structure_frame())


def test_case_d_state_as_of_previous_candle() -> None:
    # Pos1's 1.5 swing is confirmed at pos2 and active at pos4. The newer
    # pos3 high=1.8 is confirmed at pos4 and is not eligible during pos4;
    # therefore pos4 evaluates and breaks only the older level 1.5.
    df = make_df(
        high=[1.0, 1.5, 1.2, 1.8, 1.7, 1.4],
        low=[0.9, 1.3, 1.1, 1.2, 1.3, 1.2],
        close=[0.95, 1.4, 1.15, 1.4, 1.6, 1.3],
    )
    result = detect_structure_breaks(df, left=1, right=1)
    expected = exact_frame(df, [
        {
            "direction": "bullish",
            "level": 1.5,
            "swing_time": df.index[1],
            "swing_confirmed_at": df.index[2],
            "broke_at": df.index[4],
            "kind": "first",
        }
    ])
    pd.testing.assert_frame_equal(result, expected)


def test_case_e_newer_swing_supersedes_old_unbroken_swing() -> None:
    # Pos1 high=1.5 is confirmed at pos2. Pos3 high=1.3 is confirmed at pos4
    # and supersedes it. Pos5 closes 1.4, breaking level 1.3; pos6 closes
    # 1.6 but must not produce the abandoned level 1.5.
    df = make_df(
        high=[1.0, 1.5, 1.1, 1.3, 1.25, 1.45, 1.7],
        low=[0.9, 1.2, 1.0, 1.05, 1.1, 1.15, 1.3],
        close=[0.95, 1.3, 1.05, 1.2, 1.15, 1.4, 1.6],
    )
    result = detect_structure_breaks(df, left=1, right=1)
    expected = exact_frame(df, [
        {
            "direction": "bullish",
            "level": 1.3,
            "swing_time": df.index[3],
            "swing_confirmed_at": df.index[4],
            "broke_at": df.index[5],
            "kind": "first",
        }
    ])
    pd.testing.assert_frame_equal(result, expected)


def test_case_f_broken_swing_never_breaks_twice() -> None:
    # Pos1 high=1.5 breaks at pos4. The subsequent close at pos5 is not a
    # second break because the swing was removed after pos4.
    df = make_df(
        high=[1.0, 1.5, 1.2, 1.3, 1.6, 1.9],
        low=[0.9, 1.4, 1.1, 1.2, 1.3, 1.6],
        close=[1.0, 1.4, 1.15, 1.25, 1.55, 1.8],
    )
    result = detect_structure_breaks(df, left=1, right=1)
    expected = exact_frame(df, [
        {
            "direction": "bullish",
            "level": 1.5,
            "swing_time": df.index[1],
            "swing_confirmed_at": df.index[2],
            "broke_at": df.index[4],
            "kind": "first",
        }
    ])
    pd.testing.assert_frame_equal(result, expected)


def test_case_g_exact_kind_sequence() -> None:
    # Precomputed swings in this 12-candle series (some candles are not valid
    # OHLC, for example idx7 low 1.5 is above high 1.1). The series is not
    # realistic data, but swings are injected directly so this is harmless.
    # At idx2 close 2.1 cannot break the swing confirmed at idx2: newly
    # confirmed swings are not eligible until idx3. At idx5 the close is 3.0,
    # equality with the separate 3.0 level, so that level is not broken there.
    # The 3.0 level is first strictly exceeded at idx10. The idx9 swing
    # becomes active only after idx9, so it cannot break at idx9.
    # Bullish level 2.0: confirmed idx2, first strict close above is idx5:
    #   first.
    # Bearish level 1.5: swing idx7 is confirmed idx8, and close idx9 is 1.0:
    #   reversal.
    # Bullish level 3.0: swing idx5 is confirmed idx6 and close idx10 is 4.1:
    #   reversal.
    # Bullish level 4.0: swing idx9 is confirmed idx10 and close idx11 is 4.1:
    #   continuation.
    df = make_df(
        close=[1.0, 1.0, 2.1, 1.0, 1.0, 3.0, 3.1, 1.0, 1.4, 1.0, 4.1, 4.1],
        high=[1.1, 2.0, 2.2, 1.1, 1.1, 3.0, 3.2, 1.1, 1.1, 4.0, 4.2, 1.1],
        low=[0.9, 1.0, 1.0, 0.9, 0.9, 2.9, 3.0, 1.5, 1.3, 3.9, 4.0, 0.9],
    )
    swings = pd.DataFrame(
        {
            "is_swing_high": [False, True, False, False, False, True, False,
                              False, False, True, False, False],
            "is_swing_low": [False, False, False, False, False, False, False,
                             True, False, False, False, False],
            "confirmed_at": [
                pd.NaT, df.index[2], pd.NaT, pd.NaT, pd.NaT, df.index[6],
                pd.NaT, df.index[8], pd.NaT, df.index[10], pd.NaT, pd.NaT,
            ],
        },
        index=df.index,
    )
    swings["confirmed_at"] = pd.Series(
        swings["confirmed_at"],
        index=df.index,
        dtype="datetime64[ns, UTC]",
    )
    result = detect_structure_breaks(df, swings=swings, left=1, right=1)
    expected = exact_frame(df, [
        {
            "direction": "bullish",
            "level": 2.0,
            "swing_time": df.index[1],
            "swing_confirmed_at": df.index[2],
            "broke_at": df.index[5],
            "kind": "first",
        },
        {
            "direction": "bearish",
            "level": 1.5,
            "swing_time": df.index[7],
            "swing_confirmed_at": df.index[8],
            "broke_at": df.index[9],
            "kind": "reversal",
        },
        {
            "direction": "bullish",
            "level": 3.0,
            "swing_time": df.index[5],
            "swing_confirmed_at": df.index[6],
            "broke_at": df.index[10],
            "kind": "reversal",
        },
        {
            "direction": "bullish",
            "level": 4.0,
            "swing_time": df.index[9],
            "swing_confirmed_at": df.index[10],
            "broke_at": df.index[11],
            "kind": "continuation",
        },
    ])
    pd.testing.assert_frame_equal(result, expected)


def test_precomputed_swings_identical_and_validation() -> None:
    df = make_df(
        close=[1.0, 2.0, 1.5, 2.5, 1.0, 3.0],
        high=[1.0, 2.0, 1.5, 2.5, 1.0, 3.0],
        low=[0.9, 1.9, 1.4, 2.4, 0.9, 2.9],
    )
    swings = detect_swings(df, left=2, right=2)
    pd.testing.assert_frame_equal(
        detect_structure_breaks(df, left=2, right=2),
        detect_structure_breaks(df, swings=swings, left=2, right=2),
    )
    with pytest.raises(ValueError):
        detect_structure_breaks(df, swings=swings.iloc[:-1], left=2, right=2)
    with pytest.raises(ValueError):
        detect_structure_breaks(df, left=0)
    with pytest.raises(ValueError):
        detect_structure_breaks(df, right=0)


def test_input_not_mutated_and_empty_output() -> None:
    df = make_df(close=[1.0, 1.1], high=[1.0, 1.1], low=[0.9, 1.0])
    original = df.copy(deep=True)
    result = detect_structure_breaks(df, left=1, right=1)
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(result, empty_structure_frame())


def independent_swing_reference(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    left: int,
    right: int,
) -> pd.DataFrame:
    index = df.index
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    collected = []

    high_positions = [
        pos for pos in range(len(df))
        if bool(swings["is_swing_high"].iloc[pos])
    ]
    low_positions = [
        pos for pos in range(len(df))
        if bool(swings["is_swing_low"].iloc[pos])
    ]

    for pos_number, swing_pos in enumerate(high_positions):
        confirmation = swing_pos + right
        if confirmation >= len(df):
            continue
        next_pos = (
            high_positions[pos_number + 1]
            if pos_number + 1 < len(high_positions)
            else None
        )
        window_end = (
            next_pos + right if next_pos is not None else len(df) - 1
        )
        window_end = min(window_end, len(df) - 1)
        for j in range(confirmation + 1, window_end + 1):
            if close[j] > high[swing_pos]:
                collected.append((
                    "bullish",
                    float(high[swing_pos]),
                    index[swing_pos],
                    swings["confirmed_at"].iloc[swing_pos],
                    index[j],
                    j,
                ))
                break

    for pos_number, swing_pos in enumerate(low_positions):
        confirmation = swing_pos + right
        if confirmation >= len(df):
            continue
        next_pos = (
            low_positions[pos_number + 1]
            if pos_number + 1 < len(low_positions)
            else None
        )
        window_end = (
            next_pos + right if next_pos is not None else len(df) - 1
        )
        window_end = min(window_end, len(df) - 1)
        for j in range(confirmation + 1, window_end + 1):
            if close[j] < low[swing_pos]:
                collected.append((
                    "bearish",
                    float(low[swing_pos]),
                    index[swing_pos],
                    swings["confirmed_at"].iloc[swing_pos],
                    index[j],
                    j,
                ))
                break

    collected.sort(key=lambda row: (row[5], 0 if row[0] == "bullish" else 1))
    if not collected:
        return empty_structure_frame()

    kinds = []
    previous = None
    for direction, *_ in collected:
        if previous is None:
            kinds.append("first")
        elif direction == previous:
            kinds.append("continuation")
        else:
            kinds.append("reversal")
        previous = direction

    return pd.DataFrame(
        {
            "direction": [row[0] for row in collected],
            "level": [row[1] for row in collected],
            "swing_time": [row[2] for row in collected],
            "swing_confirmed_at": [row[3] for row in collected],
            "broke_at": [row[4] for row in collected],
            "kind": kinds,
        },
        columns=[
            "direction",
            "level",
            "swing_time",
            "swing_confirmed_at",
            "broke_at",
            "kind",
        ],
    )


@pytest.mark.parametrize(
    "seed,left,right,size",
    [
        (101, 1, 1, 500),
        (202, 2, 2, 900),
        (303, 3, 5, 1200),
        (404, 5, 5, 2000),
        (505, 1, 1, 1400),
        (606, 2, 2, 1800),
        (707, 5, 5, 700),
    ],
)
def test_independent_per_swing_reference(seed, left, right, size) -> None:
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, size))
    high = close + rng.uniform(0.0001, 0.0010, size)
    low = close - rng.uniform(0.0001, 0.0010, size)
    df = make_df(close, high=high, low=low)
    swings = detect_swings(df, left=left, right=right)
    actual = detect_structure_breaks(df, swings=swings, left=left, right=right)
    expected = independent_swing_reference(df, swings, left, right)
    pd.testing.assert_frame_equal(actual, expected)


def test_no_lookahead_with_truncated_swings() -> None:
    rng = np.random.default_rng(404)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, 700))
    high = close + rng.uniform(0.0001, 0.0010, 700)
    low = close - rng.uniform(0.0001, 0.0010, 700)
    df = make_df(close, high=high, low=low)

    for cutoff in (120, 300, 500):
        truncated_df = df.iloc[: cutoff + 1]
        full = detect_structure_breaks(df, left=5, right=5)
        truncated = detect_structure_breaks(truncated_df, left=5, right=5)
        cutoff_time = truncated_df.index[-1]
        expected = full.loc[full["broke_at"] <= cutoff_time].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            truncated.reset_index(drop=True),
            expected,
        )
