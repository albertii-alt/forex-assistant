# Data Notes

## Current dataset
- File: `data/EURUSD_M15_bid.parquet` (not tracked by Git)
- Rows: 74,689 candles, from 2023-11-01 00:00 UTC to 2026-10-30 20:45 UTC
- Columns: `open`, `high`, `low`, `close`, `volume`
- Index: `timestamp`, timezone-aware UTC, 15-minute spacing
- Price side: BID only. There is no ask or spread data, so spread cost must
  be modeled separately in backtests.
- Volume is not true market-wide volume in spot forex. Do not build logic
  that depends on it.

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