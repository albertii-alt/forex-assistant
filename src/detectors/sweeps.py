"""Liquidity sweep detection for swing and previous-day levels."""

from datetime import timedelta

import numpy as np
import pandas as pd

from src.detectors.swings import detect_swings


OUTPUT_COLUMNS = [
    "source",
    "side",
    "level",
    "level_set_at",
    "level_known_at",
    "swept_at",
    "extreme",
    "close",
    "depth",
]


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swings: pd.DataFrame | None = None,
    left: int = 5,
    right: int = 5,
    max_age: int | None = None,
    use_swings: bool = True,
    use_prev_day: bool = True,
) -> pd.DataFrame:
    """Detect swing and UTC previous-day liquidity sweeps.

    Daily keys are UTC calendar dates. Sunday UTC candles are merged into the
    following Monday key.
    """
    if not isinstance(left, (int, np.integer)) or left < 1:
        raise ValueError("left must be an integer greater than or equal to 1")
    if not isinstance(right, (int, np.integer)) or right < 1:
        raise ValueError("right must be an integer greater than or equal to 1")
    if max_age is not None:
        if not isinstance(max_age, (int, np.integer)) or max_age < 1:
            raise ValueError("max_age must be None or an integer greater than or equal to 1")

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

    if use_swings:
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
    else:
        detected_swings = None

    if len(df) < 1:
        return _empty_sweep_frame()

    index = df.index
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    rows: list[dict] = []

    if use_swings and detected_swings is not None:
        for p in range(len(df)):
            confirmed_at = detected_swings["confirmed_at"].iloc[p]
            if pd.isna(confirmed_at):
                continue
            if confirmed_at not in index:
                continue
            c = int(index.get_loc(confirmed_at))

            if bool(detected_swings["is_swing_high"].iloc[p]):
                level = float(high[p])
                start = c + 1
                end = len(df)
                if max_age is not None:
                    end = min(end, c + max_age + 1)
                candidates = np.flatnonzero(high[start:end] > level)
                if candidates.size:
                    j = start + int(candidates[0])
                    if close[j] <= level:
                        rows.append(
                            _sweep_row(
                                "swing", "high", level, index[p],
                                confirmed_at, index[j], high[j], close[j],
                            )
                        )

            if bool(detected_swings["is_swing_low"].iloc[p]):
                level = float(low[p])
                start = c + 1
                end = len(df)
                if max_age is not None:
                    end = min(end, c + max_age + 1)
                candidates = np.flatnonzero(low[start:end] < level)
                if candidates.size:
                    j = start + int(candidates[0])
                    if close[j] >= level:
                        rows.append(
                            _sweep_row(
                                "swing", "low", level, index[p],
                                confirmed_at, index[j], low[j], close[j],
                            )
                        )

    if use_prev_day:
        day_positions: dict[pd.Timestamp, list[int]] = {}
        for pos, timestamp in enumerate(index):
            key = timestamp.normalize()
            if timestamp.weekday() == 6:
                key = key + pd.Timedelta(days=1)
            day_positions.setdefault(key, []).append(pos)

        ordered_keys = sorted(day_positions)
        for key_number in range(1, len(ordered_keys)):
            previous_key = ordered_keys[key_number - 1]
            current_key = ordered_keys[key_number]
            previous_positions = day_positions[previous_key]
            current_positions = day_positions[current_key]

            previous_high = high[previous_positions].max()
            previous_low = low[previous_positions].min()
            previous_high_pos = previous_positions[
                int(np.argmax(high[previous_positions]))
            ]
            previous_low_pos = previous_positions[
                int(np.argmin(low[previous_positions]))
            ]
            set_high_at = index[previous_high_pos]
            set_low_at = index[previous_low_pos]
            known_at = index[previous_positions[-1]]

            start_pos = current_positions[0]
            end_pos = current_positions[-1] + 1
            candidates = np.flatnonzero(high[start_pos:end_pos] > previous_high)
            if candidates.size:
                j = start_pos + int(candidates[0])
                if close[j] <= previous_high:
                    rows.append(
                        _sweep_row(
                            "prev_day", "high", float(previous_high),
                            set_high_at, known_at, index[j], high[j], close[j],
                        )
                    )

            candidates = np.flatnonzero(low[start_pos:end_pos] < previous_low)
            if candidates.size:
                j = start_pos + int(candidates[0])
                if close[j] >= previous_low:
                    rows.append(
                        _sweep_row(
                            "prev_day", "low", float(previous_low),
                            set_low_at, known_at, index[j], low[j], close[j],
                        )
                    )

    if not rows:
        return _empty_sweep_frame()

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result["_source_order"] = result["source"].map({"swing": 0, "prev_day": 1})
    result["_side_order"] = result["side"].map({"high": 0, "low": 1})
    result = result.sort_values(
        ["swept_at", "_source_order", "_side_order", "level"],
        kind="stable",
    ).reset_index(drop=True)
    return result[OUTPUT_COLUMNS]


def _sweep_row(
    source: str,
    side: str,
    level: float,
    level_set_at: pd.Timestamp,
    level_known_at: pd.Timestamp,
    swept_at: pd.Timestamp,
    extreme: float,
    close: float,
) -> dict:
    return {
        "source": source,
        "side": side,
        "level": float(level),
        "level_set_at": level_set_at,
        "level_known_at": level_known_at,
        "swept_at": swept_at,
        "extreme": float(extreme),
        "close": float(close),
        "depth": abs(float(extreme) - float(level)),
    }


def _empty_sweep_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"),
            "level": pd.Series(dtype="float64"),
            "level_set_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "level_known_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "swept_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "extreme": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "depth": pd.Series(dtype="float64"),
        },
        columns=OUTPUT_COLUMNS,
    )
