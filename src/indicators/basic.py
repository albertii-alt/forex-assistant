"""Basic volatility and momentum indicators."""

from numbers import Integral

import numpy as np
import pandas as pd


def _validate_input(
    df: pd.DataFrame,
    required: tuple[str, ...],
) -> pd.DatetimeIndex:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    return df.index.as_unit("ns")


def _validate_period(name: str, period: int) -> int:
    if isinstance(period, bool) or not isinstance(period, Integral) or period < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(period)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return Wilder-smoothed average true range."""
    index = _validate_input(df, ("high", "low", "close"))
    period = _validate_period("period", period)

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    true_range = high - low
    if len(df) > 1:
        true_range[1:] = np.maximum.reduce(
            (
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            )
        )

    values = np.full(len(df), np.nan, dtype=np.float64)
    if len(df) >= period:
        first = period - 1
        values[first] = np.mean(true_range[:period])
        for position in range(period, len(df)):
            values[position] = (
                values[position - 1] * (period - 1) + true_range[position]
            ) / period
    return pd.Series(values, index=df.index, name="atr", dtype="float64")


def stochastic(
    df: pd.DataFrame,
    k_period: int = 5,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Return raw, smoothed %K and smoothed %D stochastic values."""
    _validate_input(df, ("high", "low", "close"))
    k_period = _validate_period("k_period", k_period)
    smooth_k = _validate_period("smooth_k", smooth_k)
    smooth_d = _validate_period("smooth_d", smooth_d)

    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    highest = high.rolling(k_period, min_periods=k_period).max()
    lowest = low.rolling(k_period, min_periods=k_period).min()
    spread = highest - lowest
    k_raw = 100.0 * (close - lowest) / spread
    k_raw = k_raw.where(spread != 0.0)
    k = k_raw.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    result = pd.DataFrame(
        {
            "k_raw": k_raw.to_numpy(dtype=np.float64),
            "k": k.to_numpy(dtype=np.float64),
            "d": d.to_numpy(dtype=np.float64),
        },
        index=df.index,
    )
    return result.astype("float64")
