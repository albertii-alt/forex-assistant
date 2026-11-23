import numpy as np
import pandas as pd
import pytest

from src.detectors.sweeps import detect_liquidity_sweeps
from src.detectors.swings import detect_swings


COLUMNS = [
    "source", "side", "level", "level_set_at", "level_known_at",
    "swept_at", "extreme", "close", "depth",
]


def make_df(high, low, close, index=None, freq="15min"):
    if index is None:
        index = pd.date_range(
            "2024-01-01 00:00",
            periods=len(close),
            freq=freq,
            tz="UTC",
        ).as_unit("ns")
    return pd.DataFrame(
        {
            "open": np.asarray(close, dtype=float),
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": np.asarray(close, dtype=float),
            "volume": np.ones(len(close)),
        },
        index=index,
    )


def inject_swings(df, high_positions=(), low_positions=(), right=1):
    confirmed = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns, UTC]",
    )
    for pos in high_positions:
        confirmed.iloc[pos] = df.index[pos + right]
    for pos in low_positions:
        if pd.isna(confirmed.iloc[pos]):
            confirmed.iloc[pos] = df.index[pos + right]
    return pd.DataFrame(
        {
            "is_swing_high": [pos in set(high_positions) for pos in range(len(df))],
            "is_swing_low": [pos in set(low_positions) for pos in range(len(df))],
            "confirmed_at": confirmed,
        },
        index=df.index,
    )


def empty_sweep_frame():
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"),
            "level": pd.Series(dtype="float64"),
            "level_set_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "level_known_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "swept_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "extreme": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "depth": pd.Series(dtype="float64"),
        },
        columns=COLUMNS,
    )


def exact_frame(df, rows):
    if not rows:
        return empty_sweep_frame()
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("level_set_at", "level_known_at", "swept_at"):
        frame[column] = pd.Series(frame[column], dtype="datetime64[ns, UTC]")
    return frame


def assert_sweep_frame(result, df, rows):
    pd.testing.assert_frame_equal(
        result,
        exact_frame(df, rows),
        check_dtype=True,
    )


def test_s1_swing_high_swept() -> None:
    df = make_df(
        [1.0, 1.5, 1.2, 1.3, 1.6, 1.4],
        [0.9, 1.4, 1.1, 1.2, 1.3, 1.3],
        [1.0, 1.4, 1.15, 1.25, 1.45, 1.35],
    )
    swings = inject_swings(df, high_positions=[1], right=1)
    result = detect_liquidity_sweeps(
        df, swings, right=1, use_swings=True, use_prev_day=False
    )
    assert_sweep_frame(result, df, [{
        "source": "swing", "side": "high", "level": 1.5,
        "level_set_at": df.index[1], "level_known_at": df.index[2],
        "swept_at": df.index[4], "extreme": 1.6,
        "close": 1.45, "depth": 0.1,
    }])


def test_s1b_max_age() -> None:
    df = make_df(
        [1.0, 1.5, 1.2, 1.3, 1.6, 1.4],
        [0.9, 1.4, 1.1, 1.2, 1.3, 1.3],
        [1.0, 1.4, 1.15, 1.25, 1.45, 1.35],
    )
    swings = inject_swings(df, high_positions=[1])
    age1 = detect_liquidity_sweeps(
        df, swings, right=1, max_age=1, use_swings=True, use_prev_day=False
    )
    pd.testing.assert_frame_equal(age1, empty_sweep_frame())
    age2 = detect_liquidity_sweeps(
        df, swings, right=1, max_age=2, use_swings=True, use_prev_day=False
    )
    assert len(age2) == 1
    assert age2.iloc[0]["level"] == 1.5
    assert age2.iloc[0]["swept_at"] == df.index[4]


def test_s2_break_consumes_level() -> None:
    df = make_df(
        [1.0, 1.5, 1.2, 1.3, 1.6, 1.7],
        [0.9, 1.4, 1.1, 1.2, 1.3, 1.3],
        [1.0, 1.4, 1.15, 1.25, 1.55, 1.4],
    )
    swings = inject_swings(df, high_positions=[1])
    result = detect_liquidity_sweeps(
        df, swings, right=1, use_swings=True, use_prev_day=False
    )
    pd.testing.assert_frame_equal(result, empty_sweep_frame())


def test_s3_close_equal_to_level_is_sweep() -> None:
    df = make_df(
        [1.0, 1.5, 1.2, 1.3, 1.6, 1.4],
        [0.9, 1.4, 1.1, 1.2, 1.3, 1.3],
        [1.0, 1.4, 1.15, 1.25, 1.5, 1.35],
    )
    swings = inject_swings(df, high_positions=[1])
    result = detect_liquidity_sweeps(
        df, swings, right=1, use_swings=True, use_prev_day=False
    )
    assert_sweep_frame(result, df, [{
        "source": "swing", "side": "high", "level": 1.5,
        "level_set_at": df.index[1], "level_known_at": df.index[2],
        "swept_at": df.index[4], "extreme": 1.6,
        "close": 1.5, "depth": 0.1,
    }])


def test_s4_high_equal_does_not_consume() -> None:
    df = make_df(
        [1.0, 1.5, 1.2, 1.3, 1.5, 1.6],
        [0.9, 1.4, 1.1, 1.2, 1.3, 1.3],
        [1.0, 1.4, 1.15, 1.25, 1.45, 1.4],
    )
    swings = inject_swings(df, high_positions=[1])
    result = detect_liquidity_sweeps(
        df, swings, right=1, use_swings=True, use_prev_day=False
    )
    assert_sweep_frame(result, df, [{
        "source": "swing", "side": "high", "level": 1.5,
        "level_set_at": df.index[1], "level_known_at": df.index[2],
        "swept_at": df.index[5], "extreme": 1.6,
        "close": 1.4, "depth": 0.1,
    }])


def test_s5_confirmation_plus_one() -> None:
    # Artificial data: idx2 violates the swing-high definition, but pos1 is
    # injected on purpose and idx2 therefore must not be usable for a sweep.
    df = make_df(
        [1.0, 1.5, 1.6, 1.7],
        [0.9, 1.4, 1.3, 1.3],
        [1.0, 1.4, 1.4, 1.4],
    )
    swings = inject_swings(df, high_positions=[1])
    result = detect_liquidity_sweeps(
        df, swings, right=1, use_swings=True, use_prev_day=False
    )
    assert_sweep_frame(result, df, [{
        "source": "swing", "side": "high", "level": 1.5,
        "level_set_at": df.index[1], "level_known_at": df.index[2],
        "swept_at": df.index[3], "extreme": 1.7,
        "close": 1.4, "depth": 0.2,
    }])


def test_s6_swing_low_swept() -> None:
    df = make_df(
        [1.3, 1.1, 1.2, 1.15, 1.1, 1.2],
        [1.2, 1.0, 1.1, 1.05, 0.9, 1.1],
        [1.25, 1.05, 1.15, 1.1, 1.05, 1.15],
    )
    swings = inject_swings(df, low_positions=[1])
    result = detect_liquidity_sweeps(
        df, swings, right=1, use_swings=True, use_prev_day=False
    )
    assert_sweep_frame(result, df, [{
        "source": "swing", "side": "low", "level": 1.0,
        "level_set_at": df.index[1], "level_known_at": df.index[2],
        "swept_at": df.index[4], "extreme": 0.9,
        "close": 1.05, "depth": 0.1,
    }])


def test_s7_one_candle_sweeps_two_levels() -> None:
    df = make_df(
        [1.0, 1.5, 1.1, 1.4, 1.1, 1.0, 1.6],
        [0.9, 1.4, 1.0, 1.3, 1.0, 0.9, 1.2],
        [1.0, 1.4, 1.05, 1.35, 1.05, 0.95, 1.3],
    )
    swings = inject_swings(df, high_positions=[1, 3])
    result = detect_liquidity_sweeps(
        df, swings, right=1, use_swings=True, use_prev_day=False
    )
    assert_sweep_frame(result, df, [
        {
            "source": "swing", "side": "high", "level": 1.4,
            "level_set_at": df.index[3], "level_known_at": df.index[4],
            "swept_at": df.index[6], "extreme": 1.6, "close": 1.3, "depth": 0.2,
        },
        {
            "source": "swing", "side": "high", "level": 1.5,
            "level_set_at": df.index[1], "level_known_at": df.index[2],
            "swept_at": df.index[6], "extreme": 1.6, "close": 1.3, "depth": 0.1,
        },
    ])


def test_p1_previous_day_levels() -> None:
    df = make_df(
        [1.2, 1.5, 1.3, 1.1, 1.3, 1.4, 1.6, 1.3],
        [1.0, 1.2, 1.0, 0.9, 1.0, 1.1, 1.2, 0.8],
        [1.1, 1.4, 1.1, 1.0, 1.2, 1.3, 1.4, 1.0],
        freq="6h",
    )
    result = detect_liquidity_sweeps(
        df, left=5, right=5, use_swings=False, use_prev_day=True
    )
    assert_sweep_frame(result, df, [
        {
            "source": "prev_day", "side": "high", "level": 1.5,
            "level_set_at": df.index[1], "level_known_at": df.index[3],
            "swept_at": df.index[6], "extreme": 1.6, "close": 1.4, "depth": 0.1,
        },
        {
            "source": "prev_day", "side": "low", "level": 0.9,
            "level_set_at": df.index[3], "level_known_at": df.index[3],
            "swept_at": df.index[7], "extreme": 0.8, "close": 1.0, "depth": 0.1,
        },
    ])


def test_p2_previous_day_break_consumes() -> None:
    df = make_df(
        [1.2, 1.5, 1.3, 1.1, 1.3, 1.4, 1.6, 1.7],
        [1.0, 1.2, 1.0, 0.9, 1.0, 1.1, 1.2, 0.8],
        [1.1, 1.4, 1.1, 1.0, 1.2, 1.3, 1.55, 1.0],
        freq="6h",
    )
    result = detect_liquidity_sweeps(
        df, use_swings=False, use_prev_day=True
    )
    assert_sweep_frame(result, df, [{
        "source": "prev_day", "side": "low", "level": 0.9,
        "level_set_at": df.index[3], "level_known_at": df.index[3],
        "swept_at": df.index[7], "extreme": 0.8, "close": 1.0, "depth": 0.1,
    }])


def test_p3_sunday_merged_into_monday() -> None:
    index = pd.DatetimeIndex([
        "2024-01-05 00:00", "2024-01-05 06:00",
        "2024-01-05 12:00", "2024-01-05 18:00",
        "2024-01-07 18:00",
        "2024-01-08 00:00", "2024-01-08 06:00",
        "2024-01-08 12:00", "2024-01-08 18:00",
    ], tz="UTC").as_unit("ns")
    df = make_df(
        [1.2, 1.5, 1.3, 1.1, 1.6, 1.7, 1.2, 1.2, 1.2],
        [1.0, 1.2, 1.0, 0.9, 1.0, 1.0, 0.8, 1.0, 1.0],
        [1.1, 1.4, 1.1, 1.0, 1.2, 1.2, 1.0, 1.1, 1.1],
        index=index,
    )
    # Treating Sunday as its own day would make Sunday day1 and Monday key
    # follow it, rather than merging Sunday and Monday under Monday.
    result = detect_liquidity_sweeps(
        df, use_swings=False, use_prev_day=True
    )
    assert_sweep_frame(result, df, [
        {
            "source": "prev_day", "side": "high", "level": 1.5,
            "level_set_at": df.index[1], "level_known_at": df.index[3],
            "swept_at": df.index[4], "extreme": 1.6, "close": 1.2, "depth": 0.1,
        },
        {
            "source": "prev_day", "side": "low", "level": 0.9,
            "level_set_at": df.index[3], "level_known_at": df.index[3],
            "swept_at": df.index[6], "extreme": 0.8, "close": 1.0, "depth": 0.1,
        },
    ])


def test_validation_immutability_and_empty() -> None:
    df = make_df([1.0, 1.5, 1.2], [0.9, 1.4, 1.1], [1.0, 1.4, 1.15])
    original = df.copy(deep=True)
    for kwargs in (
        {"left": 0},
        {"right": 0},
        {"max_age": 0},
        {"max_age": -1},
    ):
        with pytest.raises(ValueError):
            detect_liquidity_sweeps(df, **kwargs)

    swings = inject_swings(df, high_positions=[1])
    mismatched = swings.iloc[:-1]
    with pytest.raises(ValueError):
        detect_liquidity_sweeps(df, swings=mismatched, use_prev_day=False)

    computed = detect_liquidity_sweeps(
        df, left=1, right=1, use_swings=True, use_prev_day=False
    )
    precomputed = detect_liquidity_sweeps(
        df,
        swings=detect_swings(df, left=1, right=1),
        left=1,
        right=1,
        use_swings=True,
        use_prev_day=False,
    )
    pd.testing.assert_frame_equal(computed, precomputed)
    pd.testing.assert_frame_equal(df, original)

    short = df.iloc[:2]
    pd.testing.assert_frame_equal(
        detect_liquidity_sweeps(short, use_prev_day=False),
        empty_sweep_frame(),
    )


def independent_reference(df, left, right, max_age, use_swings=True, use_prev_day=True):
    rows = []
    index = df.index
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    if use_swings:
        swings = detect_swings(df, left=left, right=right)
        for p in range(len(df)):
            if pd.isna(swings["confirmed_at"].iloc[p]):
                continue
            c = list(index).index(swings["confirmed_at"].iloc[p])
            if swings["is_swing_high"].iloc[p]:
                level = float(high[p])
                j = c + 1
                while j < len(df):
                    if max_age is not None and j > c + max_age:
                        break
                    if high[j] == level:
                        j += 1
                        continue
                    if high[j] > level:
                        if close[j] <= level:
                            rows.append({
                                "source": "swing", "side": "high", "level": level,
                                "level_set_at": index[p],
                                "level_known_at": index[c],
                                "swept_at": index[j], "extreme": float(high[j]),
                                "close": float(close[j]), "depth": float(high[j] - level),
                            })
                        break
                    j += 1
            if swings["is_swing_low"].iloc[p]:
                level = float(low[p])
                j = c + 1
                while j < len(df):
                    if max_age is not None and j > c + max_age:
                        break
                    if low[j] == level:
                        j += 1
                        continue
                    if low[j] < level:
                        if close[j] >= level:
                            rows.append({
                                "source": "swing", "side": "low", "level": level,
                                "level_set_at": index[p],
                                "level_known_at": index[c],
                                "swept_at": index[j], "extreme": float(low[j]),
                                "close": float(close[j]), "depth": float(level - low[j]),
                            })
                        break
                    j += 1

    if use_prev_day:
        keys = {}
        for pos, timestamp in enumerate(index):
            key = timestamp.date()
            if timestamp.weekday() == 6:
                monday = timestamp.date()
                from datetime import timedelta
                monday = monday + timedelta(days=1)
                key = monday
            keys.setdefault(key, []).append(pos)
        ordered = list(dict.fromkeys(keys))
        for key_number in range(1, len(ordered)):
            previous = keys[ordered[key_number - 1]]
            current = keys[ordered[key_number]]
            pdh = max(float(high[p]) for p in previous)
            pdl = min(float(low[p]) for p in previous)
            pdh_set = next(index[p] for p in previous if high[p] == pdh)
            pdl_set = next(index[p] for p in previous if low[p] == pdl)
            known = index[previous[-1]]
            for j in current:
                if high[j] > pdh:
                    if close[j] <= pdh:
                        rows.append({
                            "source": "prev_day", "side": "high", "level": pdh,
                            "level_set_at": pdh_set, "level_known_at": known,
                            "swept_at": index[j], "extreme": float(high[j]),
                            "close": float(close[j]), "depth": float(high[j] - pdh),
                        })
                    break
                if j == current[-1]:
                    pass
            for j in current:
                if low[j] < pdl:
                    if close[j] >= pdl:
                        rows.append({
                            "source": "prev_day", "side": "low", "level": pdl,
                            "level_set_at": pdl_set, "level_known_at": known,
                            "swept_at": index[j], "extreme": float(low[j]),
                            "close": float(close[j]), "depth": float(pdl - low[j]),
                        })
                    break

    if not rows:
        return empty_sweep_frame()
    frame = pd.DataFrame(rows, columns=COLUMNS)
    source_order = frame["source"].map({"swing": 0, "prev_day": 1})
    side_order = frame["side"].map({"high": 0, "low": 1})
    frame["_s"] = source_order
    frame["_p"] = side_order
    frame = frame.sort_values(
        ["swept_at", "_s", "_p", "level"], kind="stable"
    ).reset_index(drop=True)
    return frame[COLUMNS]


def weekend_hourly_index(size):
    timestamps = []
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    timestamp = start
    while len(timestamps) < size:
        if timestamp.weekday() == 5:
            timestamp += pd.Timedelta(hours=1)
            continue
        if timestamp.weekday() == 4 and timestamp.hour >= 21:
            timestamp += pd.Timedelta(hours=1)
            continue
        if timestamp.weekday() == 6 and timestamp.hour < 21:
            timestamp += pd.Timedelta(hours=1)
            continue
        timestamps.append(timestamp)
        timestamp += pd.Timedelta(hours=1)
    return pd.DatetimeIndex(timestamps).as_unit("ns")


@pytest.mark.parametrize(
    "seed,left,right,max_age,size",
    [
        (101, 1, 1, None, 1500),
        (202, 2, 2, 50, 1700),
        (303, 3, 5, None, 1800),
        (404, 5, 5, 50, 2200),
        (505, 2, 2, None, 2500),
    ],
)
def test_independent_reference(seed, left, right, max_age, size):
    rng = np.random.default_rng(seed)
    index = weekend_hourly_index(size)
    open_values = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, size))
    close = open_values + rng.normal(0.0, 0.0003, size)
    high = np.maximum(open_values, close) + rng.uniform(0.0001, 0.0010, size)
    low = np.minimum(open_values, close) - rng.uniform(0.0001, 0.0010, size)
    df = pd.DataFrame(
        {
            "open": open_values,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(size),
        },
        index=index,
    )
    actual = detect_liquidity_sweeps(
        df,
        left=left,
        right=right,
        max_age=max_age,
        use_swings=True,
        use_prev_day=True,
    )
    expected = independent_reference(
        df,
        left,
        right,
        max_age,
        use_swings=True,
        use_prev_day=True,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_no_lookahead_with_weekend_gaps() -> None:
    rng = np.random.default_rng(808)
    size = 900
    index = weekend_hourly_index(size)
    open_values = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, size))
    close = open_values + rng.normal(0.0, 0.0003, size)
    high = np.maximum(open_values, close) + rng.uniform(0.0001, 0.0010, size)
    low = np.minimum(open_values, close) - rng.uniform(0.0001, 0.0010, size)
    df = pd.DataFrame(
        {
            "open": open_values,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(size),
        },
        index=index,
    )
    for cutoff in (150, 400, 800):
        truncated_df = df.iloc[: cutoff + 1]
        full = detect_liquidity_sweeps(
            df,
            left=5,
            right=5,
            use_swings=True,
            use_prev_day=True,
        )
        truncated = detect_liquidity_sweeps(
            truncated_df,
            left=5,
            right=5,
            use_swings=True,
            use_prev_day=True,
        )
        cutoff_time = truncated_df.index[-1]
        expected = full.loc[full["swept_at"] <= cutoff_time].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            truncated.reset_index(drop=True),
            expected,
        )

