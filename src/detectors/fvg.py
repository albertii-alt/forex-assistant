"""Fair value gap detection."""

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "direction",
    "c1_time",
    "formed_at",
    "zone_low",
    "zone_high",
    "size",
    "spans_gap",
    "touched_at",
    "closed_through_at",
]


def detect_fvgs(
    df: pd.DataFrame,
    min_gap: float = 0.0,
) -> pd.DataFrame:
    """Detect three-candle fair value gaps and later touch/close events."""
    if not isinstance(min_gap, (int, float, np.integer, np.floating)):
        raise ValueError("min_gap must be a non-negative number")
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError("min_gap must be a non-negative number")

    missing = [column for column in ("high", "low", "close") if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")

    if len(df) < 3:
        return _empty_fvg_frame()

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    index = df.index

    c1_high = high[:-2]
    c1_low = low[:-2]
    c3_high = high[2:]
    c3_low = low[2:]
    base_positions = np.arange(2, len(df))

    bullish = c3_low > c1_high
    bearish = c3_high < c1_low
    gap_positions = np.flatnonzero(bullish | bearish)

    if gap_positions.size:
        actual_positions = base_positions[gap_positions]
        directions = np.where(bullish[gap_positions], "bullish", "bearish")
        zone_low = np.where(
            directions == "bullish",
            c1_high[gap_positions],
            c3_high[gap_positions],
        )
        zone_high = np.where(
            directions == "bullish",
            c3_low[gap_positions],
            c1_low[gap_positions],
        )
        size = zone_high - zone_low
        keep = (size >= float(min_gap)) | np.isclose(
            size,
            float(min_gap),
            rtol=1e-9,
            atol=1e-12,
        )
        actual_positions = actual_positions[keep]
        c1_positions = actual_positions - 2
        directions = directions[keep]
        zone_low = zone_low[keep]
        zone_high = zone_high[keep]
        size = size[keep]
    else:
        actual_positions = np.array([], dtype=int)
        c1_positions = np.array([], dtype=int)
        directions = np.array([], dtype=object)
        zone_low = np.array([], dtype=float)
        zone_high = np.array([], dtype=float)
        size = np.array([], dtype=float)

    if len(actual_positions) == 0:
        return _empty_fvg_frame()

    spacings = index.to_series().diff().dropna()
    mode_spacing = spacings.mode().iloc[0] if not spacings.empty else pd.Timedelta(0)
    spans_gap = np.array(
        [
            index[formed] - index[c1] > 2 * mode_spacing
            for c1, formed in zip(c1_positions, actual_positions)
        ],
        dtype=bool,
    )

    touched_at = pd.Series(
        pd.NaT,
        index=range(len(actual_positions)),
        dtype="datetime64[ns, UTC]",
    )
    closed_at = pd.Series(
        pd.NaT,
        index=range(len(actual_positions)),
        dtype="datetime64[ns, UTC]",
    )

    for row_number, (direction, formed, low_zone, high_zone) in enumerate(
        zip(directions, actual_positions, zone_low, zone_high)
    ):
        later = np.arange(formed + 1, len(df))
        if later.size == 0:
            continue

        if direction == "bullish":
            touch_matches = np.flatnonzero(low[later] <= high_zone)
            close_matches = np.flatnonzero(close[later] < low_zone)
        else:
            touch_matches = np.flatnonzero(high[later] >= low_zone)
            close_matches = np.flatnonzero(close[later] > high_zone)

        if touch_matches.size:
            touched_at.iloc[row_number] = index[later[touch_matches[0]]]
        if close_matches.size:
            closed_at.iloc[row_number] = index[later[close_matches[0]]]

    return pd.DataFrame(
        {
            "direction": directions,
            "c1_time": index[c1_positions].copy(),
            "formed_at": index[actual_positions].copy(),
            "zone_low": zone_low,
            "zone_high": zone_high,
            "size": size,
            "spans_gap": spans_gap,
            "touched_at": touched_at,
            "closed_through_at": closed_at,
        },
        columns=OUTPUT_COLUMNS,
    )


def _empty_fvg_frame() -> pd.DataFrame:
    """Build an empty FVG result with stable columns and dtypes."""
    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "c1_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "formed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "zone_low": pd.Series(dtype="float64"),
            "zone_high": pd.Series(dtype="float64"),
            "size": pd.Series(dtype="float64"),
            "spans_gap": pd.Series(dtype="bool"),
            "touched_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "closed_through_at": pd.Series(dtype="datetime64[ns, UTC]"),
        },
        columns=OUTPUT_COLUMNS,
    )
