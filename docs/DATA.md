# Data Notes

## Timestamp convention

- Candle timestamps are candle OPEN times. A candle stamped `T` closes at
  `T` plus the timeframe length, and its OHLC values are only knowable after
  that close.
- Anything known at candle `T`'s close cannot be acted on before the NEXT
  candle's open.
- Detector columns ending in `_at` or named `formed_at` use the OPEN time of
  the candle whose close made the fact knowable.

## Current dataset
- File: `data/EURUSD_M15_bid.parquet` (not tracked by Git)
- Rows: 74,689 candles, from 2023-11-01 00:00 UTC to 2026-10-30 20:45 UTC
- Columns: `open`, `high`, `low`, `close`, `volume`
- Index: `timestamp`, timezone-aware UTC, 15-minute spacing
- Price side: BID only. There is no ask or spread data, so spread cost must
  be modeled separately in backtests.
- Volume is not true market-wide volume in spot forex. Do not build logic
  that depends on it.

## 10-year research dataset
- File: `data/EURUSD_M15_bid_10y.parquet` (not tracked by Git)
- Rows: 249,368 candles, from 2016-11-01 00:00 UTC to 2026-10-30 20:45 UTC.
- Download range starts 2016-11-01 UTC and ends exclusively at 2026-11-01 UTC.
- The development set contains timestamps before 2024-11-01 00:00 UTC. The
  sealed holdout contains timestamps at or after that boundary. Only use the
  holdout for one final evaluation of a candidate that has already passed on
  the development set, as specified in `docs/strategy_rules.md`.
- The older 3-year file remains in use by the viewer until it is explicitly
  switched to the 10-year dataset.

## Known characteristics
- No candles on weekends (the market closes Friday evening and reopens
  Sunday evening UTC), so there are time gaps. Do not treat them as errors
  and do not forward-fill across them.
- Session times (London, New York) must be derived by converting UTC with
  `zoneinfo`, which handles daylight saving changes.

## Source and regeneration
- Source: Dukascopy, via `dukascopy-python` 7.0.0 (new API: `fetch_candles`
  and `to_dataframe`).
- Regenerate with: `python -m src.data.download`