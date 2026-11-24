from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.detectors.zones import ZONE_COLUMNS, detect_zones
from src.indicators.basic import atr as calculate_atr


def make_index(size: int, unit: str = "ns") -> pd.DatetimeIndex:
    return pd.date_range(
        "2024-01-09 14:00",
        periods=size,
        freq="15min",
        tz="UTC",
    ).as_unit(unit)


def make_frame(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    close_values = np.asarray(close, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close_values.copy(),
            "high": (
                np.asarray(high, dtype=np.float64)
                if high is not None
                else close_values + 0.0003
            ),
            "low": (
                np.asarray(low, dtype=np.float64)
                if low is not None
                else close_values - 0.0003
            ),
            "close": close_values.copy(),
        },
        index=index if index is not None else make_index(len(close_values)),
    )


def make_expected(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                "direction": pd.Series(dtype="object"),
                "base_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "known_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "zone_low": pd.Series(dtype="float64"),
                "zone_high": pd.Series(dtype="float64"),
                "atr_at_base": pd.Series(dtype="float64"),
                "move_size": pd.Series(dtype="float64"),
                "closed_through_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "expires_at": pd.Series(dtype="datetime64[ns, UTC]"),
            },
            columns=ZONE_COLUMNS,
        )
    expected = pd.DataFrame(rows, columns=ZONE_COLUMNS)
    for column in ("base_time", "known_at", "closed_through_at", "expires_at"):
        expected[column] = pd.to_datetime(expected[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    for column in ("zone_low", "zone_high", "atr_at_base", "move_size"):
        expected[column] = expected[column].astype("float64")
    expected["direction"] = expected["direction"].astype(object)
    return expected


def zone_row(
    direction: str,
    index: pd.DatetimeIndex,
    base: int,
    known: int,
    low: float,
    high: float,
    atr_value: float,
    move: float,
    closed: int | None = None,
    expires: int | None = None,
) -> dict[str, object]:
    return {
        "direction": direction,
        "base_time": index[base],
        "known_at": index[known],
        "zone_low": low,
        "zone_high": high,
        "atr_at_base": atr_value,
        "move_size": move,
        "closed_through_at": pd.NaT if closed is None else index[closed],
        "expires_at": pd.NaT if expires is None else index[expires],
    }


def z1_data() -> tuple[pd.DataFrame, pd.Series]:
    index = make_index(6)
    df = make_frame(
        [1.1000, 1.1000, 1.0980, 1.0960, 1.0940, 1.0930],
        high=[1.1010, 1.1010, 1.1000, 1.0975, 1.0955, 1.0945],
        low=[1.0990, 1.0990, 1.0970, 1.0950, 1.0930, 1.0920],
        index=index,
    )
    values = pd.Series(
        [np.nan, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004],
        index=index,
        name="atr",
        dtype="float64",
    )
    return df, values


def test_z1_supply_hand_computed() -> None:
    df, atr_values = z1_data()
    expected = make_expected(
        [
            zone_row("supply", df.index, 1, 2, 1.0990, 1.1010, 0.0004, 0.0010),
            zone_row("supply", df.index, 2, 3, 1.0970, 1.1000, 0.0004, 0.0010),
            zone_row("supply", df.index, 3, 4, 1.0950, 1.0975, 0.0004, 0.0010),
        ]
    )
    actual = detect_zones(df, atr=atr_values)
    pd.testing.assert_frame_equal(actual, expected, rtol=0, atol=1e-9)


def test_z2_demand_hand_computed() -> None:
    index = make_index(6)
    df = make_frame(
        [1.1000, 1.1000, 1.1020, 1.1040, 1.1060, 1.1070],
        high=[1.1010, 1.1010, 1.1030, 1.1050, 1.1070, 1.1080],
        low=[1.0990, 1.0990, 1.1000, 1.1025, 1.1045, 1.1055],
        index=index,
    )
    atr_values = pd.Series(
        [np.nan, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004],
        index=index,
        name="atr",
    )
    expected = make_expected(
        [
            zone_row("demand", index, 1, 2, 1.0990, 1.1010, 0.0004, 0.0010),
            zone_row("demand", index, 2, 3, 1.1000, 1.1030, 0.0004, 0.0010),
            zone_row("demand", index, 3, 4, 1.1025, 1.1050, 0.0004, 0.0010),
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(df, atr=atr_values),
        expected,
        rtol=0,
        atol=1e-9,
    )


def test_one_base_candle_can_create_both_zone_directions() -> None:
    index = make_index(5)
    df = make_frame(
        [1.1003, 1.1005, 1.0980, 1.1018, 1.1003],
        index=index,
    )
    df.iloc[1, df.columns.get_loc("low")] = 1.0990
    df.iloc[1, df.columns.get_loc("high")] = 1.1010
    values = pd.Series(np.nan, index=index, name="atr")
    values.iloc[1] = 0.0004
    expected = make_expected(
        [
            zone_row(
                "supply",
                index,
                1,
                2,
                1.0990,
                1.1010,
                0.0004,
                0.0010,
                closed=3,
            ),
            zone_row(
                "demand",
                index,
                1,
                3,
                1.0990,
                1.1010,
                0.0004,
                0.0008,
            ),
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(df, atr=values),
        expected,
        rtol=0,
        atol=1e-9,
    )


def z3_data(case: int) -> tuple[pd.DataFrame, pd.Series]:
    index = make_index(8)
    close = [1.1003] * 8
    close[1] = 1.1005
    if case == 1:
        close[4] = 1.0990
    else:
        close[5] = 1.0990
    df = make_frame(close, index=index)
    high = df["high"].to_numpy(copy=True)
    low = df["low"].to_numpy(copy=True)
    high[1], low[1] = 1.1010, 1.1000
    df["high"], df["low"] = high, low
    atr_values = pd.Series(np.nan, index=index, name="atr", dtype="float64")
    atr_values.iloc[1] = 0.0004
    return df, atr_values


def test_z3_lookahead_limit_and_z3b_extended_limit() -> None:
    early_df, early_atr = z3_data(1)
    # Only idx[1] has ATR; idx[4]'s close qualifies at the third candle ahead.
    expected_early = make_expected(
        [
            zone_row(
                "supply",
                early_df.index,
                1,
                4,
                1.1000,
                1.1010,
                0.0004,
                0.0010,
            )
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(early_df, atr=early_atr),
        expected_early,
        rtol=0,
        atol=1e-9,
    )

    late_df, late_atr = z3_data(2)
    # idx[5] is four candles after idx[1], so it is excluded at the default limit.
    assert detect_zones(late_df, atr=late_atr).empty
    expected_late = make_expected(
        [
            zone_row(
                "supply",
                late_df.index,
                1,
                5,
                1.1000,
                1.1010,
                0.0004,
                0.0010,
            )
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(late_df, atr=late_atr, max_lookahead=4),
        expected_late,
        rtol=0,
        atol=1e-9,
    )


def test_z4_threshold_boundary_is_inclusive_with_tolerance() -> None:
    index = make_index(4)

    def boundary_frame(close_at_q: float) -> tuple[pd.DataFrame, pd.Series]:
        df = make_frame(
            [1.1003, 1.1005, close_at_q, 1.1003],
            index=index,
        )
        df.iloc[1, df.columns.get_loc("low")] = 1.1000
        df.iloc[1, df.columns.get_loc("high")] = 1.1010
        atr_values = pd.Series(np.nan, index=index, name="atr", dtype="float64")
        atr_values.iloc[1] = 0.0004
        return df, atr_values

    df, atr_values = boundary_frame(1.0992)
    expected = make_expected(
        [
            zone_row(
                "supply", index, 1, 2, 1.1000, 1.1010, 0.0004, 0.0008
            )
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(df, atr=atr_values),
        expected,
        rtol=0,
        atol=1e-9,
    )
    above, above_atr = boundary_frame(1.09921)
    assert detect_zones(above, atr=above_atr).empty


def z5_data(close_through_at: int | None) -> tuple[pd.DataFrame, pd.Series]:
    index = make_index(6 if close_through_at == 5 else 5)
    closes = [1.1003, 1.1000, 1.0980, 1.1005, 1.1011]
    if close_through_at == 5:
        closes[4] = 1.1005
        closes.append(1.1011)
    df = make_frame(closes, index=index)
    df.iloc[1, df.columns.get_loc("low")] = 1.0990
    df.iloc[1, df.columns.get_loc("high")] = 1.1010
    values = pd.Series(np.nan, index=index, name="atr", dtype="float64")
    values.iloc[1] = 0.0004
    return df, values


def test_z5_close_through_and_expiry_hand_computed() -> None:
    df, values = z5_data(4)
    # q=idx[2], so expiry is idx[4]; the close-through at idx[4] is included.
    expected_closed = make_expected(
        [
            zone_row(
                "supply",
                df.index,
                1,
                2,
                1.0990,
                1.1010,
                0.0004,
                0.0010,
                closed=4,
                expires=4,
            )
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(df, expiry_candles=2, atr=values),
        expected_closed,
        rtol=0,
        atol=1e-9,
    )

    late_df, late_values = z5_data(5)
    # The idx[5] close is past q+2; expiry remains idx[4], with no close-through.
    expected_open = make_expected(
        [
            zone_row(
                "supply",
                late_df.index,
                1,
                2,
                1.0990,
                1.1010,
                0.0004,
                0.0010,
                expires=4,
            )
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(late_df, expiry_candles=2, atr=late_values),
        expected_open,
        rtol=0,
        atol=1e-9,
    )


def test_atr_is_measured_at_base_candle_close() -> None:
    index = make_index(4)
    df = make_frame([1.1003, 1.1005, 1.0992, 1.1003], index=index)
    df.iloc[1, df.columns.get_loc("low")] = 1.1000
    df.iloc[1, df.columns.get_loc("high")] = 1.1010
    values = pd.Series([np.nan, 0.0004, 0.0010, np.nan], index=index, name="atr")
    expected = make_expected(
        [
            zone_row(
                "supply", index, 1, 2, 1.1000, 1.1010, 0.0004, 0.0008
            )
        ]
    )
    pd.testing.assert_frame_equal(
        detect_zones(df, atr=values),
        expected,
        rtol=0,
        atol=1e-9,
    )


def test_creation_threshold_uses_base_low_and_high() -> None:
    index = make_index(4)
    df = make_frame([1.1003, 1.1005, 1.0994, 1.1003], index=index)
    df.iloc[1, df.columns.get_loc("low")] = 1.1000
    df.iloc[1, df.columns.get_loc("high")] = 1.1010
    values = pd.Series(np.nan, index=index, name="atr")
    values.iloc[1] = 0.0004
    # 1.0994 is above the low-based threshold (1.0992), despite being below
    # the incorrect close-based threshold (1.0997).
    assert detect_zones(df, atr=values).empty


def reference_zones(
    df: pd.DataFrame,
    atr_values: pd.Series,
    atr_mult: float,
    max_lookahead: int,
    expiry_candles: int,
) -> pd.DataFrame:
    index = df.index.as_unit("ns")
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    atr_list = atr_values.astype(float).tolist()
    rows: list[tuple[int, int, int, dict[str, object]]] = []

    for base in range(len(df)):
        base_atr = atr_list[base]
        if np.isnan(base_atr):
            continue
        for direction in ("demand", "supply"):
            known = None
            for candidate in range(base + 1, min(len(df), base + max_lookahead + 1)):
                if direction == "supply":
                    qualifies = closes[candidate] <= lows[base] - atr_mult * base_atr + 1e-9
                else:
                    qualifies = closes[candidate] >= highs[base] + atr_mult * base_atr - 1e-9
                if qualifies:
                    known = candidate
                    break
            if known is None:
                continue

            expiry = known + expiry_candles
            expiry_value = index[expiry] if expiry < len(df) else None
            closed = None
            for later in range(known + 1, min(len(df), expiry + 1)):
                crossed = (
                    closes[later] < lows[base]
                    if direction == "demand"
                    else closes[later] > highs[base]
                )
                if crossed:
                    closed = later
                    break
            move = (
                closes[known] - highs[base]
                if direction == "demand"
                else lows[base] - closes[known]
            )
            row = {
                "direction": direction,
                "base_time": index[base],
                "known_at": index[known],
                "zone_low": lows[base],
                "zone_high": highs[base],
                "atr_at_base": base_atr,
                "move_size": move,
                "closed_through_at": pd.NaT if closed is None else index[closed],
                "expires_at": pd.NaT if expiry_value is None else expiry_value,
            }
            rows.append((known, base, 0 if direction == "demand" else 1, row))

    rows.sort(key=lambda item: item[:3])
    return make_expected([item[3] for item in rows])


def make_random_market(seed: int, size: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0.0, 0.0004, size))
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) + rng.uniform(0.00005, 0.0004, size)
    low = np.minimum(open_price, close) - rng.uniform(0.00005, 0.0004, size)
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close},
        index=make_index(size),
    )


@pytest.mark.parametrize("seed", [17, 29, 41, 73])
@pytest.mark.parametrize(
    ("atr_mult", "max_lookahead", "expiry_candles"),
    [(2.0, 3, 192), (1.0, 3, 50), (2.0, 5, 192), (0.5, 2, 30)],
)
def test_seeded_loop_reference(
    seed: int,
    atr_mult: float,
    max_lookahead: int,
    expiry_candles: int,
) -> None:
    df = make_random_market(seed)
    atr_values = calculate_atr(df, period=14)
    expected = reference_zones(
        df,
        atr_values,
        atr_mult,
        max_lookahead,
        expiry_candles,
    )
    actual = detect_zones(
        df,
        atr=atr_values,
        atr_mult=atr_mult,
        max_lookahead=max_lookahead,
        expiry_candles=expiry_candles,
    )
    print(
        f"seed={seed}, atr_mult={atr_mult}, max_lookahead={max_lookahead}, "
        f"expiry_candles={expiry_candles}, zones={len(actual)}"
    )
    assert len(actual) >= 10
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-9, atol=1e-9)


def test_zones_no_lookahead() -> None:
    df = make_random_market(29)
    full = detect_zones(df)
    for cutoff in (300, 700, 1100):
        prefix = df.iloc[: cutoff + 1]
        last_time = prefix.index[-1]
        truncated = detect_zones(prefix)
        expected = full.loc[full["known_at"] <= last_time].copy()
        for column in ("closed_through_at", "expires_at"):
            too_late = expected[column].notna() & (expected[column] > last_time)
            expected.loc[too_late, column] = pd.NaT
        pd.testing.assert_frame_equal(truncated, expected)


def test_z6_nan_atr_short_and_empty_inputs() -> None:
    index = make_index(4)
    df = make_frame([1.1, 1.1, 1.1, 1.1], index=index)
    values = pd.Series(np.nan, index=index, name="atr")
    assert detect_zones(df, atr=values).empty
    assert detect_zones(df, atr_period=14).empty

    empty = pd.DataFrame(
        {
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    result = detect_zones(empty)
    pd.testing.assert_frame_equal(result, make_expected([]))


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("atr_period", 0),
        ("atr_period", True),
        ("atr_period", 1.5),
        ("atr_mult", 0),
        ("atr_mult", np.nan),
        ("max_lookahead", 0),
        ("max_lookahead", True),
        ("expiry_candles", 0),
    ],
)
def test_z7_invalid_parameters(parameter: str, value: object) -> None:
    df, values = z1_data()
    with pytest.raises(ValueError, match=parameter):
        detect_zones(df, atr=values, **{parameter: value})


def test_z7_validation_immutability_and_output_dtypes() -> None:
    df, values = z1_data()
    before_df, before_atr = df.copy(deep=True), values.copy(deep=True)
    result = detect_zones(df, atr=values)
    pd.testing.assert_frame_equal(df, before_df)
    pd.testing.assert_series_equal(values, before_atr)

    with pytest.raises(ValueError, match="UTC"):
        detect_zones(
            df.set_axis(pd.date_range("2024-01-01", periods=len(df), freq="15min")),
            atr=values,
        )
    with pytest.raises(ValueError, match="sorted"):
        detect_zones(df.iloc[::-1], atr=values.iloc[::-1])
    with pytest.raises(ValueError, match="open"):
        detect_zones(df.drop(columns="open"), atr=values)
    with pytest.raises(ValueError, match="same index"):
        detect_zones(df, atr=values.set_axis(values.index + pd.Timedelta(minutes=1)))

    assert result.index.equals(pd.RangeIndex(len(result)))
    assert result["direction"].dtype == object
    for column in ("base_time", "known_at", "closed_through_at", "expires_at"):
        assert result[column].dtype == pd.DatetimeTZDtype(unit="ns", tz="UTC")
    for column in ("zone_low", "zone_high", "atr_at_base", "move_size"):
        assert result[column].dtype == np.dtype("float64")
    assert result.columns.tolist() == list(ZONE_COLUMNS)


def test_z1_preserves_microsecond_input_precision() -> None:
    df, values = z1_data()
    expected = detect_zones(df, atr=values)
    us_df, us_values = df.copy(), values.copy()
    us_df.index = us_df.index.as_unit("us")
    us_values.index = us_values.index.as_unit("us")
    actual = detect_zones(us_df, atr=us_values)
    for column in ("base_time", "known_at", "closed_through_at", "expires_at"):
        expected[column] = expected[column].dt.as_unit("ns")
        actual[column] = actual[column].dt.as_unit("ns")
    pd.testing.assert_frame_equal(actual, expected)
