"""Previous-trading-day and overnight session levels in New York time."""

import numpy as np
import pandas as pd

from src.data.resample_et import ET, trading_day


def _validate_input(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("Index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Input index must be sorted")
    missing = [column for column in ("high", "low") if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def detect_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate known previous-day and overnight levels for each ET trading day.

    An ET trading day starts at 17:00. Its overnight window includes candles
    from 17:00 onward and candles before 09:30 ET on the following local date.
    """
    _validate_input(df)
    days = trading_day(df.index)
    local = df.index.tz_convert(ET)
    hours = local.hour.to_numpy()
    minutes = local.minute.to_numpy()
    positions_by_day = pd.Series(
        np.arange(len(df), dtype=np.int64),
        index=days,
    ).groupby(level=0, sort=True).indices

    rows: list[dict[str, object]] = []
    ordered_days = list(positions_by_day)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    for day_number, day in enumerate(ordered_days):
        positions = np.asarray(positions_by_day[day], dtype=np.int64)
        row: dict[str, object] = {
            "trading_day": day,
            "prev_day_high": np.nan,
            "prev_day_high_set_at": pd.NaT,
            "prev_day_low": np.nan,
            "prev_day_low_set_at": pd.NaT,
            "prev_day_known_at": pd.NaT,
            "overnight_high": np.nan,
            "overnight_high_set_at": pd.NaT,
            "overnight_low": np.nan,
            "overnight_low_set_at": pd.NaT,
            "overnight_known_at": pd.NaT,
        }

        if day_number > 0:
            previous = np.asarray(
                positions_by_day[ordered_days[day_number - 1]],
                dtype=np.int64,
            )
            previous_high = np.max(high[previous])
            previous_low = np.min(low[previous])
            row.update(
                {
                    "prev_day_high": previous_high,
                    "prev_day_high_set_at": df.index[
                        previous[np.flatnonzero(high[previous] == previous_high)[0]]
                    ],
                    "prev_day_low": previous_low,
                    "prev_day_low_set_at": df.index[
                        previous[np.flatnonzero(low[previous] == previous_low)[0]]
                    ],
                    "prev_day_known_at": df.index[positions[0]],
                }
            )

        overnight_mask = (hours[positions] >= 17) | (
            (hours[positions] < 9)
            | ((hours[positions] == 9) & (minutes[positions] < 30))
        )
        overnight = positions[overnight_mask]
        morning_mask = (
            ((hours[positions] > 9) | ((hours[positions] == 9) & (minutes[positions] >= 30)))
            & (hours[positions] < 17)
        )
        known_positions = positions[morning_mask]
        if len(overnight) and len(known_positions):
            overnight_high = np.max(high[overnight])
            overnight_low = np.min(low[overnight])
            row.update(
                {
                    "overnight_high": overnight_high,
                    "overnight_high_set_at": df.index[
                        overnight[np.flatnonzero(high[overnight] == overnight_high)[0]]
                    ],
                    "overnight_low": overnight_low,
                    "overnight_low_set_at": df.index[
                        overnight[np.flatnonzero(low[overnight] == overnight_low)[0]]
                    ],
                    "overnight_known_at": df.index[known_positions[0]],
                }
            )
        rows.append(row)

    columns = [
        "trading_day",
        "prev_day_high",
        "prev_day_high_set_at",
        "prev_day_low",
        "prev_day_low_set_at",
        "prev_day_known_at",
        "overnight_high",
        "overnight_high_set_at",
        "overnight_low",
        "overnight_low_set_at",
        "overnight_known_at",
    ]
    result = pd.DataFrame(rows, columns=columns)
    result["trading_day"] = pd.to_datetime(result["trading_day"]).astype(
        "datetime64[ns]"
    )
    for column in (
        "prev_day_high_set_at",
        "prev_day_low_set_at",
        "prev_day_known_at",
        "overnight_high_set_at",
        "overnight_low_set_at",
        "overnight_known_at",
    ):
        result[column] = pd.to_datetime(result[column], utc=True)
    return result
