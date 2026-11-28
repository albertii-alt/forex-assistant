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
    - Optional no-target plans and market time-open exits: DONE.
  - Candidate A plan generation: DONE.
  - Candidate B building blocks (ATR, stochastic, 4h trend state, supply and
    demand zones): DONE.
  - Candidate B setup finder (plans only; no real-data backtest): DONE.
  - Candidate C 4h Donchian plan finder and price-only entry counter:
    DONE.
      - Candidate C development backtest: REJECTED (one run, logged).
  - Candidate D (Asian-range London breakout) finder, count gate (513 plans)
    and development backtest: DONE, REJECTED (one run, logged).
  - Candidates A and B: INCONCLUSIVE at the count gate (logged, no backtest).
  - Round 2 complete. No candidate has passed and the holdout is untouched.
    Any further strategy work needs a new pre-registered round.
- Phase 5: Backtester with spread and slippage costs, out-of-sample and
  walk-forward testing.
- Phase 6: Risk module (IN PROGRESS)
  - Position sizing (cost-aware, rounds down, risk cap): DONE
  - Daily limits (max trades per day, daily loss limit, one open position): DONE
  - News calendar logic (parse, freshness check, blackout windows, briefing): DONE
  - Calendar fetch script (manual run, fake-network tests): DONE
  - News briefing and advisory warnings on the journal screen: DONE
    (advice only, never blocks saving)
  - Scheduled or automatic calendar fetching: later (manual command for now)
- Phase 7: Trade journal and alerts (IN PROGRESS)
  - Trade journal storage (SQLite) and trading-day summary: DONE
  - Planning workflow (balances, plan check, checked save): DONE
  - Entry and close screen (Streamlit, paper/demo only): DONE
  - Alerts: later
- Phase 8: LLM features (news and event summaries, journal review)
  - Journal review report (plain statistics, no LLM, shown on the journal
    screen with a sample-size warning): DONE
  - News briefing (plain text from the calendar, no LLM): DONE
  - LLM summaries and explanations: later (needs a free option first)
- Phase 9: Paper trading on a demo account.
- Phase 10: ML setup filtering, only if earlier results justify it.