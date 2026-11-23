"""New York trading-day labeling and ET-aligned OHLCV aggregation."""

import numpy as np
import pandas as pd


ET = "America/New_York"
TRADING_DAY_START_HOUR = 17
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
TIMEFRAME_DURATIONS = {
    "15m": pd.Timedelta(minutes=15),
    "4h": pd.Timedelta(hours=4),
    "1D": pd.Timedelta(days=1),
}


def _validate_utc_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    if index.tz is None or str(index.tz).upper() != "UTC":
        raise ValueError("Index must be timezone-aware UTC")


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map UTC candle opens to naive midnight trading-day labels in ET."""
    _validate_utc_index(index)
    local = index.tz_convert(ET)
    dates = local.tz_localize(None).normalize()
    increments = (local.hour >= TRADING_DAY_START_HOUR).astype("int64")
    result = dates + pd.to_timedelta(increments, unit="D")
    result.name = index.name
    return result.as_unit("ns")


def _et_bin_starts(index: pd.DatetimeIndex, timeframe: str) -> pd.DatetimeIndex:
    local_wall = index.tz_convert(ET).tz_localize(None)
    shifted = local_wall - pd.Timedelta(hours=TRADING_DAY_START_HOUR)
    frequency = "D" if timeframe == "1D" else "4h"
    local_starts = shifted.floor(frequency) + pd.Timedelta(
        hours=TRADING_DAY_START_HOUR
    )

    try:
        localized = local_starts.tz_localize(
            ET,
            ambiguous="raise",
            nonexistent="raise",
        )
    except Exception as exc:
        for start in local_starts.unique():
            try:
                pd.Timestamp(start).tz_localize(
                    ET,
                    ambiguous="raise",
                    nonexistent="raise",
                )
            except Exception as localize_error:
                position = int(np.flatnonzero(local_starts == start)[0])
                raise ValueError(
                    f"Cannot localize ET bin start {start} for candle "
                    f"{index[position]}: ambiguous or nonexistent local time"
                ) from localize_error
        raise ValueError(f"Cannot localize ET bin starts: {exc}") from exc

    result = localized.tz_convert("UTC").as_unit("ns")
    result.name = index.name
    return result


def resample_ohlcv_et(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate OHLCV into ET-aligned bins, retaining only populated bins.

    Labels are bin OPEN times; a bin is only known at ``close_time`` (see the
    timestamp convention in ``docs/DATA.md``). ``close_time`` is the bin start
    plus its elapsed UTC duration; DST transitions occur while the market is
    closed, so this matches the intended wall-clock close.
    """
    if timeframe not in TIMEFRAME_DURATIONS:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Choose one of: {', '.join(TIMEFRAME_DURATIONS)}."
        )
    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    _validate_utc_index(df.index)
    if not df.index.is_monotonic_increasing:
        raise ValueError("Input index must be sorted")

    duration = TIMEFRAME_DURATIONS[timeframe]
    if timeframe == "15m":
        result = df.copy()
        result["n_candles"] = np.ones(len(result), dtype=np.int64)
        result["close_time"] = result.index + duration
        return result

    bin_starts = _et_bin_starts(df.index, timeframe)
    source = df.loc[:, list(OHLCV_COLUMNS)].copy()
    source["__bin_start"] = bin_starts
    grouped = source.groupby("__bin_start", sort=True)
    result = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    result["n_candles"] = grouped.size().astype("int64")
    result["close_time"] = result.index + duration
    result.index = result.index.as_unit("ns")
    result.index.name = df.index.name
    return result[
        ["open", "high", "low", "close", "volume", "n_candles", "close_time"]
    ]


def align_htf_to_ltf(
    ltf_index: pd.DatetimeIndex,
    htf: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Align the latest closed HTF candle to each LTF open time.

    This is conservative: an HTF candle is used from the OPEN of the first
    LTF candle at or after its ``close_time``.
    """
    _validate_utc_index(ltf_index)
    if not ltf_index.is_monotonic_increasing:
        raise ValueError("ltf_index must be sorted")
    if not isinstance(htf.index, pd.DatetimeIndex):
        raise ValueError("htf must have a DatetimeIndex")
    _validate_utc_index(htf.index)
    if "close_time" not in htf.columns:
        raise ValueError("htf must contain close_time")

    requested = list(
        ["open", "high", "low", "close", "volume"] if columns is None else columns
    )
    missing = [column for column in requested if column not in htf.columns]
    if missing:
        raise ValueError(f"Missing requested HTF columns: {', '.join(missing)}")

    close_times = pd.DatetimeIndex(htf["close_time"]).as_unit("ns")
    _validate_utc_index(close_times)
    if not close_times.is_monotonic_increasing:
        raise ValueError("htf close_time must be sorted")

    output_columns = requested + ([] if "htf_start" in requested else ["htf_start"])
    if ltf_index.empty:
        return pd.DataFrame(index=ltf_index, columns=output_columns)

    left = pd.DataFrame({"__ltf_time": ltf_index.as_unit("ns")})
    right = htf.loc[:, requested].copy()
    right["htf_start"] = htf.index.as_unit("ns")
    right["__htf_close_time"] = close_times
    right = right.sort_values("__htf_close_time")
    aligned = pd.merge_asof(
        left,
        right,
        left_on="__ltf_time",
        right_on="__htf_close_time",
        direction="backward",
        allow_exact_matches=True,
    )
    result = aligned.loc[:, output_columns]
    result.index = ltf_index
    result.index.name = ltf_index.name
    return result
