"""Timeframe-aware weekend rangebreak calculations."""

from datetime import timedelta

import pandas as pd


TIMEFRAME_DURATIONS = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1D": timedelta(days=1),
}


def weekend_rangebreaks(
    index: pd.DatetimeIndex,
    timeframe: str,
) -> list[dict]:
    """Return closures that do not overlap any candle's time span."""
    if timeframe not in TIMEFRAME_DURATIONS:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Choose one of: {', '.join(TIMEFRAME_DURATIONS)}."
        )
    if len(index) < 2:
        return []

    duration = TIMEFRAME_DURATIONS[timeframe]
    breaks = []
    for previous, following in zip(index[:-1], index[1:]):
        candle_end = previous + duration
        if following - candle_end >= timedelta(hours=12):
            breaks.append(
                {
                    "bounds": [candle_end.isoformat(), following.isoformat()],
                }
            )
    return breaks
