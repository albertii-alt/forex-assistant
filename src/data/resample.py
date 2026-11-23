"""OHLCV resampling for the chart viewer."""

import pandas as pd


SUPPORTED_TIMEFRAMES = ("15m", "1h", "4h", "1D")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV candles without lookahead or fake weekend candles.

    Bins are left-closed and left-labeled in UTC. Daily candles use UTC
    00:00-23:59 boundaries, not the forex 5pm New York convention.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Choose one of: {', '.join(SUPPORTED_TIMEFRAMES)}."
        )

    source = df.copy()
    if timeframe == "15m":
        return source

    rules = source.resample(
        timeframe,
        closed="left",
        label="left",
        origin="start_day",
    )
    counts = rules["close"].count()
    result = rules.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )[OHLCV_COLUMNS]
    return result.loc[counts > 0].copy()
