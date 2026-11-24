from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.basic import atr, stochastic


def make_market(seed: int, size: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        "2020-01-01",
        periods=size,
        freq="15min",
        tz="UTC",
    )
    close = 1.1 + np.cumsum(rng.normal(0.0, 0.0005, size))
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) + rng.uniform(0.0001, 0.001, size)
    low = np.minimum(open_price, close) - rng.uniform(0.0001, 0.001, size)
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close},
        index=index,
    )


def reference_atr(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int,
) -> list[float]:
    true_range = []
    for position in range(len(high)):
        if position == 0:
            true_range.append(high[position] - low[position])
        else:
            true_range.append(
                max(
                    high[position] - low[position],
                    abs(high[position] - close[position - 1]),
                    abs(low[position] - close[position - 1]),
                )
            )
    result = [float("nan")] * len(high)
    if len(high) >= period:
        result[period - 1] = sum(true_range[:period]) / period
        for position in range(period, len(high)):
            result[position] = (
                result[position - 1] * (period - 1) + true_range[position]
            ) / period
    return result


def reference_stochastic(
    high: list[float],
    low: list[float],
    close: list[float],
    k_period: int,
    smooth_k: int,
    smooth_d: int,
) -> tuple[list[float], list[float], list[float]]:
    raw = [float("nan")] * len(high)
    for position in range(k_period - 1, len(high)):
        highest = max(high[position - k_period + 1 : position + 1])
        lowest = min(low[position - k_period + 1 : position + 1])
        if highest != lowest:
            raw[position] = 100.0 * (close[position] - lowest) / (highest - lowest)

    def full_window_mean(values: list[float], window: int) -> list[float]:
        means = [float("nan")] * len(values)
        for position in range(window - 1, len(values)):
            sample = values[position - window + 1 : position + 1]
            if all(np.isfinite(value) for value in sample):
                means[position] = sum(sample) / window
        return means

    k = full_window_mean(raw, smooth_k)
    d = full_window_mean(k, smooth_d)
    return raw, k, d


def assert_series_equal(actual: pd.Series, expected: list[float]) -> None:
    expected_series = pd.Series(
        expected,
        index=actual.index,
        name=actual.name,
        dtype="float64",
    )
    pd.testing.assert_series_equal(actual, expected_series, rtol=1e-9, atol=1e-9)


def test_atr_hand_computed_wilder_and_period_one() -> None:
    index = pd.date_range("2024-01-01", periods=16, freq="15min", tz="UTC")
    high = [2.0] * 14 + [17.0, 2.0]
    low = [0.0] * 14 + [1.0, 0.0]
    df = pd.DataFrame(
        {"high": high, "low": low, "close": [1.0] * 16},
        index=index,
    )
    result = atr(df)
    assert result.iloc[:13].isna().all()
    assert result.iloc[13] == pytest.approx(2.0, abs=1e-9)
    assert result.iloc[14] == pytest.approx(3.0, abs=1e-9)
    assert result.iloc[15] == pytest.approx(41 / 14, abs=1e-9)
    assert_series_equal(
        atr(df, period=1),
        reference_atr(high, low, [1.0] * 16, 1),
    )


def test_atr_short_input_is_all_nan() -> None:
    df = pd.DataFrame(
        {"high": [2.0, 2.1], "low": [1.0, 1.1], "close": [1.5, 1.8]},
        index=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
    )
    assert atr(df, period=3).isna().all()


def test_stochastic_hand_computed_and_zero_range() -> None:
    index = pd.date_range("2024-01-01", periods=9, freq="15min", tz="UTC")
    closes = [5.0, 5.0, 5.0, 5.0, 2.0, 4.0, 10.0, 6.0, 3.0]
    df = pd.DataFrame(
        {"high": [10.0] * 9, "low": [0.0] * 9, "close": closes},
        index=index,
    )
    result = stochastic(df)
    expected_raw, expected_k, expected_d = reference_stochastic(
        [10.0] * 9,
        [0.0] * 9,
        closes,
        5,
        3,
        3,
    )
    assert expected_raw[4:] == [20.0, 40.0, 100.0, 60.0, 30.0]
    assert expected_k[6:] == pytest.approx(
        [53.333333333333336, 66.66666666666667, 63.333333333333336],
        abs=1e-9,
    )
    assert expected_d[8] == pytest.approx(61.111111111111114, abs=1e-9)
    for column, expected in zip(
        ("k_raw", "k", "d"),
        (expected_raw, expected_k, expected_d),
    ):
        assert_series_equal(result[column].rename(column), expected)

    zero_range = pd.DataFrame(
        {"high": [1.0] * 9, "low": [1.0] * 9, "close": [1.0] * 9},
        index=index,
    )
    assert stochastic(zero_range).isna().all().all()


@pytest.mark.parametrize("seed", [11, 23, 37, 59])
def test_indicator_loop_references(seed: int) -> None:
    df = make_market(seed)
    high = df["high"].tolist()
    low = df["low"].tolist()
    close = df["close"].tolist()

    pd.testing.assert_series_equal(
        atr(df),
        pd.Series(
            reference_atr(high, low, close, 14),
            index=df.index,
            name="atr",
        ),
        rtol=1e-9,
        atol=1e-9,
    )
    expected = reference_stochastic(high, low, close, 5, 3, 3)
    actual = stochastic(df)
    expected_frame = pd.DataFrame(
        dict(zip(("k_raw", "k", "d"), expected)),
        index=df.index,
    ).astype("float64")
    pd.testing.assert_frame_equal(actual, expected_frame, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("seed", [11])
def test_indicator_no_lookahead(seed: int) -> None:
    df = make_market(seed)
    full_atr = atr(df)
    full_stochastic = stochastic(df)
    for cutoff in (150, 400, 700):
        prefix = df.iloc[: cutoff + 1]
        pd.testing.assert_series_equal(
            atr(prefix),
            full_atr.iloc[: cutoff + 1],
            rtol=1e-9,
            atol=1e-9,
        )
        pd.testing.assert_frame_equal(
            stochastic(prefix),
            full_stochastic.iloc[: cutoff + 1],
            rtol=1e-9,
            atol=1e-9,
        )


def test_hand_computed_indicators_preserve_microsecond_index() -> None:
    index = pd.date_range("2024-01-01", periods=16, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "high": [2.0] * 14 + [17.0, 2.0],
            "low": [0.0] * 14 + [1.0, 0.0],
            "close": [1.0] * 16,
        },
        index=index,
    )
    us_df = df.copy()
    us_df.index = us_df.index.as_unit("us")
    expected_atr = atr(df)
    actual_atr = atr(us_df)
    expected_atr.index = expected_atr.index.as_unit("ns")
    actual_atr.index = actual_atr.index.as_unit("ns")
    pd.testing.assert_series_equal(actual_atr, expected_atr)

    nine = df.iloc[:9].copy()
    nine["high"] = 10.0
    nine["low"] = 0.0
    nine["close"] = [5.0, 5.0, 5.0, 5.0, 2.0, 4.0, 10.0, 6.0, 3.0]
    us_nine = nine.copy()
    us_nine.index = us_nine.index.as_unit("us")
    expected_stoch = stochastic(nine)
    actual_stoch = stochastic(us_nine)
    expected_stoch.index = expected_stoch.index.as_unit("ns")
    actual_stoch.index = actual_stoch.index.as_unit("ns")
    pd.testing.assert_frame_equal(actual_stoch, expected_stoch)


@pytest.mark.parametrize("kwargs", [{"period": 0}, {"period": True}, {"period": 1.5}])
def test_atr_rejects_invalid_period(kwargs: dict[str, object]) -> None:
    df = make_market(1, 20)
    with pytest.raises(ValueError):
        atr(df, **kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("k_period", 0),
        ("smooth_k", True),
        ("smooth_d", 1.5),
    ],
)
def test_stochastic_rejects_invalid_period(name: str, value: object) -> None:
    df = make_market(1, 20)
    with pytest.raises(ValueError):
        stochastic(df, **{name: value})


def test_indicator_validation_and_immutability() -> None:
    df = make_market(1, 20)
    before = df.copy(deep=True)
    atr(df)
    stochastic(df)
    pd.testing.assert_frame_equal(df, before)

    with pytest.raises(ValueError, match="UTC"):
        atr(df.set_axis(pd.date_range("2024-01-01", periods=len(df), freq="15min")))
    with pytest.raises(ValueError, match="sorted"):
        stochastic(df.iloc[::-1])
    with pytest.raises(ValueError, match="Missing"):
        atr(df.drop(columns="high"))
    with pytest.raises(ValueError):
        stochastic(df, k_period=0)
