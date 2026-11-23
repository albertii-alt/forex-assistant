"""Break-of-structure detector."""

from typing import Iterable

import numpy as np
import pandas as pd

from src.detectors.swings import detect_swings


OUTPUT_COLUMNS = [
    "direction",
    "level",
    "swing_time",
    "swing_confirmed_at",
    "broke_at",
    "kind",
]


def detect_structure_breaks(
    df: pd.DataFrame,
    swings: pd.DataFrame | None = None,
    left: int = 5,
    right: int = 5,
) -> pd.DataFrame:
    """Detect confirmed swing breaks.

    Breaks are sorted by broke_at, with bearish ordered after bullish for
    the same candle.
    """
    if not isinstance(left, (int, np.integer)) or left < 1:
        raise ValueError("left must be an integer greater than or equal to 1")
    if not isinstance(right, (int, np.integer)) or right < 1:
        raise ValueError("right must be an integer greater than or equal to 1")

    missing = [
        column for column in ("high", "low", "close") if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")

    if swings is None:
        detected_swings = detect_swings(df, left=left, right=right)
    else:
        if not swings.index.equals(df.index):
            raise ValueError("swings index must equal df.index")
        required = {"is_swing_high", "is_swing_low", "confirmed_at"}
        missing_swings = required.difference(swings.columns)
        if missing_swings:
            raise ValueError(
                f"swings is missing required column(s): "
                f"{', '.join(sorted(missing_swings))}"
            )
        detected_swings = swings

    if len(df) == 0:
        return _empty_structure_frame()

    index = df.index
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()

    confirmations: dict[int, list[tuple[str, int]]] = {}
    for swing_pos in range(len(df)):
        confirmed = detected_swings["confirmed_at"].iloc[swing_pos]
        if pd.isna(confirmed):
            continue
        if confirmed not in index:
            continue
        confirmed_pos = index.get_loc(confirmed)
        if detected_swings["is_swing_high"].iloc[swing_pos]:
            confirmations.setdefault(int(confirmed_pos), []).append(
                ("bullish", swing_pos)
            )
        if detected_swings["is_swing_low"].iloc[swing_pos]:
            confirmations.setdefault(int(confirmed_pos), []).append(
                ("bearish", swing_pos)
            )

    active_high: tuple[int, float, pd.Timestamp] | None = None
    active_low: tuple[int, float, pd.Timestamp] | None = None
    rows: list[dict] = []
    confirmation_positions = sorted(confirmations)
    confirmation_cursor = 0

    for j in range(len(df)):
        while (
            confirmation_cursor < len(confirmation_positions)
            and confirmation_positions[confirmation_cursor] <= j - 1
        ):
            confirmed_pos = confirmation_positions[confirmation_cursor]
            confirmation_cursor += 1
            for direction, swing_pos in confirmations.pop(confirmed_pos, []):
                confirmed_at = detected_swings["confirmed_at"].iloc[swing_pos]
                if direction == "bullish":
                    active_high = (swing_pos, float(high[swing_pos]), confirmed_at)
                else:
                    active_low = (swing_pos, float(low[swing_pos]), confirmed_at)

        if active_high is not None and close[j] > active_high[1]:
            swing_pos, level, confirmed_at = active_high
            rows.append(
                {
                    "direction": "bullish",
                    "level": level,
                    "swing_time": index[swing_pos],
                    "swing_confirmed_at": confirmed_at,
                    "broke_at": index[j],
                    "kind": "",
                    "_position": j,
                }
            )
            active_high = None

        if active_low is not None and close[j] < active_low[1]:
            swing_pos, level, confirmed_at = active_low
            rows.append(
                {
                    "direction": "bearish",
                    "level": level,
                    "swing_time": index[swing_pos],
                    "swing_confirmed_at": confirmed_at,
                    "broke_at": index[j],
                    "kind": "",
                    "_position": j,
                }
            )
            active_low = None

        for direction, swing_pos in confirmations.pop(j, []):
            confirmed_at = detected_swings["confirmed_at"].iloc[swing_pos]
            if direction == "bullish":
                active_high = (swing_pos, float(high[swing_pos]), confirmed_at)
            else:
                active_low = (swing_pos, float(low[swing_pos]), confirmed_at)

    if not rows:
        return _empty_structure_frame()

    rows.sort(
        key=lambda row: (
            row["_position"],
            0 if row["direction"] == "bullish" else 1,
        )
    )
    previous_direction: str | None = None
    for row in rows:
        if previous_direction is None:
            row["kind"] = "first"
        elif row["direction"] == previous_direction:
            row["kind"] = "continuation"
        else:
            row["kind"] = "reversal"
        previous_direction = row["direction"]
        del row["_position"]

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _empty_structure_frame() -> pd.DataFrame:
    """Build an empty structure-break result with stable dtypes."""
    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "level": pd.Series(dtype="float64"),
            "swing_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "swing_confirmed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "broke_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "kind": pd.Series(dtype="object"),
        },
        columns=OUTPUT_COLUMNS,
    )
