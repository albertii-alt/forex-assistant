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
  - fair value gaps (this task)
  - structure breaks
  - liquidity sweeps
- Phase 4: Finalize exact rules in `docs/strategy_rules.md`. The owner
  approves definitions BEFORE any strategy code is written.
- Phase 5: Backtester with spread and slippage costs, out-of-sample and
  walk-forward testing.
- Phase 6: Risk module (position sizing, daily loss limit, max trades per
  day, news blackout windows).
- Phase 7: Trade journal and alerts.
- Phase 8: LLM features (news and event summaries, journal review).
- Phase 9: Paper trading on a demo account.
- Phase 10: ML setup filtering, only if earlier results justify it.