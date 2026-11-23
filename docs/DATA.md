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

## ET-aligned strategy data
- `src.data.resample_et.trading_day` assigns candles to an ET trading date;
  the trading day rolls at 17:00 America/New_York.
- `resample_ohlcv_et` supports 15-minute passthrough, 4-hour, and daily data.
  The 4-hour boundaries are 17:00, 21:00, 01:00, 05:00, 09:00, and 13:00 ET.
  Daily bins begin at 17:00 ET. Empty bins are omitted, including weekend
  closures.
- Aggregated rows include `n_candles` and `close_time`. Labels are bin open
  times; higher-timeframe values must not be used until `close_time`. Close
  times add the bin duration in elapsed UTC time.
- `align_htf_to_ltf` joins only higher-timeframe candles whose `close_time`
  is at or before the lower-timeframe candle open, and returns their
  `htf_start` timestamp for auditability.
- Strategy session levels are per ET trading day. Previous-day levels use
  the preceding trading day with data and become known at the first open of
  the current trading day. Overnight levels cover 17:00 ET through (but not
  including) 09:30 ET and become known at the first 09:30-or-later candle
  before 17:00 ET.

## Source and regeneration
- Source: Dukascopy, via `dukascopy-python` 7.0.0 (new API: `fetch_candles`
  and `to_dataframe`).
- Regenerate with: `python -m src.data.download`