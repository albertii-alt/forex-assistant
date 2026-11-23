"""Swing high and swing low detection."""

import numpy as np
import pandas as pd


def detect_swings(
    df: pd.DataFrame,
    left: int = 5,
    right: int = 5,
) -> pd.DataFrame:
    """Detect left/right-window swings without lookahead."""
    if not isinstance(left, (int, np.integer)) or left < 1:
        raise ValueError("left must be an integer greater than or equal to 1")
    if not isinstance(right, (int, np.integer)) or right < 1:
        raise ValueError("right must be an integer greater than or equal to 1")

    missing = [column for column in ("high", "low") if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    size = len(df)

    previous_high_max = high.shift(1).rolling(
        window=int(left),
        min_periods=int(left),
    ).max()
    previous_low_min = low.shift(1).rolling(
        window=int(left),
        min_periods=int(left),
    ).min()

    reversed_high = high.iloc[::-1]
    reversed_low = low.iloc[::-1]
    future_or_self_high = reversed_high.rolling(
        window=int(right) + 1,
        min_periods=int(right) + 1,
    ).max().iloc[::-1]
    future_or_self_low = reversed_low.rolling(
        window=int(right) + 1,
        min_periods=int(right) + 1,
    ).min().iloc[::-1]

    is_swing_high = (
        (high > previous_high_max)
        & (high >= future_or_self_high)
    ).fillna(False).to_numpy(dtype=bool)
    is_swing_low = (
        (low < previous_low_min)
        & (low <= future_or_self_low)
    ).fillna(False).to_numpy(dtype=bool)

    confirmed_at = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns, UTC]",
    )
    positions = np.flatnonzero(is_swing_high | is_swing_low)
    valid_positions = positions[positions + int(right) < size]
    if valid_positions.size:
        confirmed_at.iloc[valid_positions] = df.index[
            valid_positions + int(right)
        ]

    return pd.DataFrame(
        {
            "is_swing_high": is_swing_high,
            "is_swing_low": is_swing_low,
            "confirmed_at": confirmed_at,
        },
        index=df.index,
    )
