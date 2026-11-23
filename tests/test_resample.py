from pathlib import Path

import pandas as pd
import pytest

from src.data.resample import resample_ohlcv


UTC = "UTC"


def make_candles(values: list[tuple], start: str = "2024-01-01 00:00") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=len(values), freq="15min", tz=UTC)
    return pd.DataFrame(values, index=index, columns=["open", "high", "low", "close", "volume"])


def test_1h_ohlcv_exact_aggregation() -> None:
    df = make_candles(
        [
            (1.1000, 1.1020, 1.0990, 1.1010, 10),
            (1.1010, 1.1030, 1.1005, 1.1025, 20),
            (1.1025, 1.1040, 1.1015, 1.1035, 30),
            (1.1035, 1.1050, 1.1020, 1.1040, 40),
            (1.1040, 1.1060, 1.1030, 1.1050, 50),
        ]
    )

    result = resample_ohlcv(df, "1h")

    expected = pd.DataFrame(
        {
            "open": [1.1000, 1.1040],
            "high": [1.1050, 1.1060],
            "low": [1.0990, 1.1030],
            "close": [1.1040, 1.1050],
            "volume": [100, 50],
        },
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 01:00"],
            tz=UTC,
        ),
    )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_4h_bins_align_to_utc_boundaries() -> None:
    df = make_candles(
        [
            (1, 4, 1, 2, 10),
            (2, 5, 2, 3, 20),
            (3, 6, 3, 4, 30),
            (4, 7, 4, 5, 40),
        ],
        start="2024-01-01 00:00",
    )
    df = df.iloc[:1].reindex(
        pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 03:00", "2024-01-01 04:00"],
            tz=UTC,
        )
    )
    df = df.fillna(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 5}
    )

    result = resample_ohlcv(df, "4h")

    assert list(result.index.strftime("%H:%M")) == ["00:00", "04:00"]
    assert result.index.tz is not None
    assert str(result.index.tz).upper() == "UTC"


def test_missing_candles_do_not_create_fake_rows() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [10, 20],
        },
        index=pd.DatetimeIndex(
            ["2024-01-05 15:00", "2024-01-08 15:00"],
            tz=UTC,
        ),
    )

    result = resample_ohlcv(df, "1h")

    assert len(result) == 2
    assert list(result.index) == list(df.index)


def test_input_is_not_modified_and_output_index_is_utc() -> None:
    df = make_candles(
        [(1.0, 2.0, 0.5, 1.5, 10), (1.5, 2.5, 1.0, 2.0, 20)]
    )
    original = df.copy(deep=True)

    result = resample_ohlcv(df, "1h")

    pd.testing.assert_frame_equal(df, original)
    assert str(result.index.tz).upper() == "UTC"


def test_invalid_timeframe_raises_value_error() -> None:
    df = make_candles([(1.0, 2.0, 0.5, 1.5, 10)])

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        resample_ohlcv(df, "5m")


def test_15m_returns_identical_values_as_copy() -> None:
    df = make_candles(
        [(1.0, 1.1, 0.9, 1.05, 10), (1.05, 1.2, 1.0, 1.15, 20)]
    )

    result = resample_ohlcv(df, "15m")

    pd.testing.assert_frame_equal(result, df)
    assert result is not df
