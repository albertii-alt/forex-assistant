import numpy as np
import pandas as pd
import pytest

from src.app.rangebreaks import TIMEFRAME_DURATIONS, weekend_rangebreaks
from src.data.loader import load_candles
from src.data.resample import resample_ohlcv


def utc_index(*timestamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(timestamps, tz="UTC")


def test_daily_break_starts_after_friday_candle() -> None:
    index = utc_index(
        "2026-10-08 00:00",
        "2026-10-09 00:00",
        "2026-10-11 00:00",
        "2026-10-12 00:00",
    )

    assert weekend_rangebreaks(index, "1D") == [
        {
            "bounds": [
                "2026-10-10T00:00:00+00:00",
                "2026-10-11T00:00:00+00:00",
            ]
        }
    ]


def test_hourly_break_starts_after_last_friday_candle() -> None:
    index = utc_index("2026-10-09 20:00", "2026-10-11 21:00")

    assert weekend_rangebreaks(index, "1h") == [
        {
            "bounds": [
                "2026-10-09T21:00:00+00:00",
                "2026-10-11T21:00:00+00:00",
            ]
        }
    ]


def test_15m_break_starts_after_last_friday_candle() -> None:
    index = utc_index("2026-10-09 20:45", "2026-10-11 21:00")

    assert weekend_rangebreaks(index, "15m") == [
        {
            "bounds": [
                "2026-10-09T21:00:00+00:00",
                "2026-10-11T21:00:00+00:00",
            ]
        }
    ]


def test_4h_break_starts_after_last_friday_candle() -> None:
    index = utc_index("2026-10-09 20:00", "2026-10-11 20:00")

    assert weekend_rangebreaks(index, "4h") == [
        {
            "bounds": [
                "2026-10-10T00:00:00+00:00",
                "2026-10-11T20:00:00+00:00",
            ]
        }
    ]


def test_continuous_data_and_small_gap_are_not_hidden() -> None:
    continuous = utc_index(
        "2026-10-09 10:00",
        "2026-10-09 11:00",
        "2026-10-09 12:00",
    )
    three_hour_gap = utc_index("2026-10-09 10:00", "2026-10-09 14:00")

    assert weekend_rangebreaks(continuous, "1h") == []
    assert weekend_rangebreaks(three_hour_gap, "1h") == []


def test_fewer_than_two_candles_returns_empty() -> None:
    assert weekend_rangebreaks(utc_index(), "15m") == []
    assert weekend_rangebreaks(utc_index("2026-10-09 20:00"), "4h") == []


def test_invalid_timeframe_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        weekend_rangebreaks(utc_index("2026-10-09 20:00"), "30m")


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1D"])
def test_real_data_breaks_do_not_overlap_candle_spans(timeframe: str) -> None:
    candles = load_candles()
    resampled = resample_ohlcv(candles, timeframe)
    breaks = weekend_rangebreaks(resampled.index, timeframe)
    duration = TIMEFRAME_DURATIONS[timeframe]
    candle_opens = resampled.index.asi8
    duration_ns = pd.Timedelta(duration).value

    for interval in breaks:
        start, end = pd.to_datetime(interval["bounds"], utc=True)
        first_possible_overlap = np.searchsorted(
            candle_opens,
            start.value - duration_ns,
            side="right",
        )
        first_at_or_after_end = np.searchsorted(
            candle_opens,
            end.value,
            side="left",
        )
        assert first_possible_overlap >= first_at_or_after_end
