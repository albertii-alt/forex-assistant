# Roadmap

Work in order. Do not start a phase until the owner assigns it.

- Phase 0: Project setup. DONE
- Phase 1: Data (EUR/USD 15-minute candles, 2023-11 to 2026-10). DONE
- Phase 2: Chart viewer (Streamlit). DONE
  Acceptance: loads the parquet file, interactive candlestick chart, date
  range picker, timeframe switcher (15m, 1h, 4h, 1D) built by resampling,
  runs with `streamlit run src/app/main.py`, and has tests for the
  resampling function.
- Phase 3: Pattern detectors (IN PROGRESS)
  - swings (DONE)
  - fair value gaps (DONE)
  - structure breaks (DONE)
  - liquidity sweeps (DONE)
  - session levels (basic previous-day and overnight levels): DONE
  - remaining, undecided: equal highs/lows, inverse FVG, equilibrium,
    daily bias
- Phase 4: IN PROGRESS
  - Strategy rules v0.3 approved.
  - Groundwork step 1 (10-year data and development/holdout split): DONE.
  - Groundwork step 2 (ET-aligned resampling, higher-timeframe alignment,
    session levels): DONE.
  - Groundwork step 3 (cost-aware backtest engine and trade metrics): DONE.
  - Candidate A plan generation: DONE.
  - Candidate B building blocks (ATR, stochastic, 4h trend state, supply and
    demand zones): DONE.
  - Candidate B setup finder (plans only; no real-data backtest): DONE.
- Phase 5: Backtester with spread and slippage costs, out-of-sample and
  walk-forward testing.
- Phase 6: Risk module (position sizing, daily loss limit, max trades per
  day, news blackout windows).
- Phase 7: Trade journal and alerts.
- Phase 8: LLM features (news and event summaries, journal review).
- Phase 9: Paper trading on a demo account.
- Phase 10: ML setup filtering, only if earlier results justify it.