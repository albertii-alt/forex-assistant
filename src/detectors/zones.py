"""Supply and demand zone detection from displacement candles."""

from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.indicators.basic import atr as calculate_atr


ZONE_COLUMNS = (
    "direction",
    "base_time",
    "known_at",
    "zone_low",
    "zone_high",
    "atr_at_base",
    "move_size",
    "closed_through_at",
    "expires_at",
)


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(value)


def _empty_zones() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "base_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "known_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "zone_low": pd.Series(dtype="float64"),
            "zone_high": pd.Series(dtype="float64"),
            "atr_at_base": pd.Series(dtype="float64"),
            "move_size": pd.Series(dtype="float64"),
            "closed_through_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "expires_at": pd.Series(dtype="datetime64[ns, UTC]"),
        },
        columns=ZONE_COLUMNS,
    )


def detect_zones(
    df: pd.DataFrame,
    atr_period: int = 14,
    atr_mult: float = 2.0,
    max_lookahead: int = 3,
    expiry_candles: int = 192,
    atr: pd.Series | None = None,
) -> pd.DataFrame:
    """Detect displacement-based supply and demand zones.

    A qualifying move is measured from the base candle using its ATR at the
    close. ``known_at`` is the open time of the confirming candle; the zone is
    knowable at that candle's close and usable from the next candle's open.
    Creation thresholds are inclusive with a 1e-9 tolerance. A zone is closed
    through only by a later close strictly beyond its far side.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    missing = [column for column in ("open", "high", "low", "close") if column not in df]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")

    atr_period = _validate_positive_integer("atr_period", atr_period)
    max_lookahead = _validate_positive_integer("max_lookahead", max_lookahead)
    expiry_candles = _validate_positive_integer("expiry_candles", expiry_candles)
    if (
        isinstance(atr_mult, bool)
        or not isinstance(atr_mult, Real)
        or not np.isfinite(atr_mult)
        or atr_mult <= 0
    ):
        raise ValueError("atr_mult must be a finite number greater than 0")

    if atr is None:
        atr = calculate_atr(df, period=atr_period)
    elif not isinstance(atr, pd.Series) or not atr.index.equals(df.index):
        raise ValueError("atr must be a Series with the same index as df")

    index = df.index.as_unit("ns")
    size = len(df)
    if size == 0:
        return _empty_zones()

    try:
        atr_values = atr.to_numpy(dtype=np.float64, copy=False)
        high = df["high"].to_numpy(dtype=np.float64, copy=False)
        low = df["low"].to_numpy(dtype=np.float64, copy=False)
        close = df["close"].to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("OHLC and ATR values must be numeric") from error

    valid_atr = ~np.isnan(atr_values)
    supply_q = np.full(size, -1, dtype=np.int64)
    demand_q = np.full(size, -1, dtype=np.int64)
    limit = min(max_lookahead, size - 1)

    for offset in range(1, limit + 1):
        base_positions = np.arange(size - offset)
        pending_supply = valid_atr[: size - offset] & (supply_q[: size - offset] < 0)
        supply_match = pending_supply & (
            close[offset:]
            <= low[: size - offset] - atr_mult * atr_values[: size - offset] + 1e-9
        )
        matched_positions = base_positions[supply_match]
        supply_q[matched_positions] = matched_positions + offset

        pending_demand = valid_atr[: size - offset] & (demand_q[: size - offset] < 0)
        demand_match = pending_demand & (
            close[offset:]
            >= high[: size - offset] + atr_mult * atr_values[: size - offset] - 1e-9
        )
        matched_positions = base_positions[demand_match]
        demand_q[matched_positions] = matched_positions + offset

    supply_bases = np.flatnonzero(supply_q >= 0)
    demand_bases = np.flatnonzero(demand_q >= 0)
    base_positions = np.concatenate((supply_bases, demand_bases))
    if base_positions.size == 0:
        return _empty_zones()

    known_positions = np.concatenate(
        (supply_q[supply_bases], demand_q[demand_bases])
    )
    is_demand = np.concatenate(
        (np.zeros(len(supply_bases), dtype=bool), np.ones(len(demand_bases), dtype=bool))
    )
    direction_order = np.where(is_demand, 0, 1)
    order = np.lexsort(
        (
            direction_order,
            index.asi8[base_positions],
            index.asi8[known_positions],
        )
    )
    base_positions = base_positions[order]
    known_positions = known_positions[order]
    is_demand = is_demand[order]

    zone_low = low[base_positions]
    zone_high = high[base_positions]
    atr_at_base = atr_values[base_positions]
    move_size = np.where(
        is_demand,
        close[known_positions] - zone_high,
        zone_low - close[known_positions],
    )

    closed_positions = np.full(len(base_positions), -1, dtype=np.int64)
    expiry_positions = np.full(len(base_positions), -1, dtype=np.int64)
    for row, (known_pos, demand) in enumerate(zip(known_positions, is_demand)):
        expiry_pos = known_pos + expiry_candles
        if expiry_pos < size:
            expiry_positions[row] = expiry_pos

        first = known_pos + 1
        last = min(expiry_pos, size - 1)
        if first > last:
            continue
        later_closes = close[first : last + 1]
        through = (
            np.flatnonzero(later_closes < zone_low[row])
            if demand
            else np.flatnonzero(later_closes > zone_high[row])
        )
        if through.size:
            closed_positions[row] = first + through[0]

    def timestamps_at(positions: np.ndarray) -> pd.DatetimeIndex:
        nanoseconds = np.full(
            len(positions),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )
        valid = positions >= 0
        nanoseconds[valid] = index.asi8[positions[valid]]
        return pd.DatetimeIndex(pd.to_datetime(nanoseconds, unit="ns", utc=True))

    return pd.DataFrame(
        {
            "direction": pd.Series(
                np.where(is_demand, "demand", "supply"),
                dtype="object",
            ),
            "base_time": index.take(base_positions),
            "known_at": index.take(known_positions),
            "zone_low": zone_low.astype(np.float64, copy=False),
            "zone_high": zone_high.astype(np.float64, copy=False),
            "atr_at_base": atr_at_base.astype(np.float64, copy=False),
            "move_size": move_size.astype(np.float64, copy=False),
            "closed_through_at": timestamps_at(closed_positions),
            "expires_at": timestamps_at(expiry_positions),
        },
        columns=ZONE_COLUMNS,
    )
