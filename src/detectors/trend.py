"""Confirmed higher-timeframe swing trend state."""

from bisect import insort

import numpy as np
import pandas as pd

from src.detectors.swings import detect_swings


SWING_COLUMNS = ("is_swing_high", "is_swing_low", "confirmed_at")


def htf_trend_state(
    df: pd.DataFrame,
    swings: pd.DataFrame | None = None,
    left: int = 3,
    right: int = 3,
) -> pd.DataFrame:
    """Classify higher-timeframe trend using only swings confirmed by each close.

    On lower timeframes, consume this state only through
    ``align_htf_to_ltf``, which requires the higher-timeframe ``close_time``.
    """
    if not isinstance(left, (int, np.integer)) or isinstance(left, bool) or left < 1:
        raise ValueError("left must be an integer greater than or equal to 1")
    if not isinstance(right, (int, np.integer)) or isinstance(right, bool) or right < 1:
        raise ValueError("right must be an integer greater than or equal to 1")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    missing = [column for column in ("high", "low") if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")

    index = df.index.as_unit("ns")
    if swings is None:
        swings = detect_swings(df, left=left, right=right)
    else:
        if not isinstance(swings, pd.DataFrame):
            raise ValueError("swings must be a pandas DataFrame")
        if not swings.index.equals(df.index):
            raise ValueError("swings index must equal df.index")
        missing_swings = [column for column in SWING_COLUMNS if column not in swings]
        if missing_swings:
            raise ValueError(
                f"swings is missing required column(s): {', '.join(missing_swings)}"
            )

    size = len(df)
    events: list[list[tuple[int, bool, bool]]] = [[] for _ in range(size)]
    positions = {timestamp: position for position, timestamp in enumerate(index)}
    confirmed_values = swings["confirmed_at"].to_numpy()
    high_flags = swings["is_swing_high"].to_numpy(dtype=bool)
    low_flags = swings["is_swing_low"].to_numpy(dtype=bool)
    for swing_pos in range(size):
        confirmed = confirmed_values[swing_pos]
        if pd.isna(confirmed):
            continue
        confirmed_at = pd.Timestamp(confirmed)
        if confirmed_at.tz is None or str(confirmed_at.tz).upper() != "UTC":
            continue
        confirmed_pos = positions.get(
            confirmed_at.tz_convert("UTC").as_unit("ns")
        )
        if confirmed_pos is None:
            continue
        if high_flags[swing_pos] or low_flags[swing_pos]:
            events[confirmed_pos].append(
                (swing_pos, bool(high_flags[swing_pos]), bool(low_flags[swing_pos]))
            )

    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    high_values = df["high"].to_numpy(dtype=np.float64)
    low_values = df["low"].to_numpy(dtype=np.float64)
    states = np.full(size, "none", dtype=object)
    for candle_pos in range(size):
        for swing_pos, is_high, is_low in events[candle_pos]:
            if is_high:
                insort(highs, (swing_pos, high_values[swing_pos]))
            if is_low:
                insort(lows, (swing_pos, low_values[swing_pos]))

        if len(highs) < 2 or len(lows) < 2:
            continue
        high_1, high_2 = highs[-2][1], highs[-1][1]
        low_1, low_2 = lows[-2][1], lows[-1][1]
        if high_2 > high_1 and low_2 > low_1:
            states[candle_pos] = "up"
        elif high_2 < high_1 and low_2 < low_1:
            states[candle_pos] = "down"

    return pd.DataFrame(
        {"state": states},
        index=df.index,
    )
