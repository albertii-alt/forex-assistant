# Backtest engine

The backtest engine evaluates caller-supplied trade plans against sorted,
UTC-indexed 15-minute BID OHLC candles. It contains no setup finder, strategy
rules, broker integration, or live order execution.

## Running a simulation

Call `src.backtest.engine.run_backtest(df, plans, ...)`. Plans must provide a
unique `plan_id`, `side`, `order_type`, UTC `start_time` and `time_exit_at`,
`entry_price`, `stop_price`, and `valid_candles`. By default, exactly one of
`target_r` or `target_price` is required. Optional `no_target=True` disables
the target and requires both target columns to be NaN; an absent or NaN
`no_target` means `False`. Optional `exit_mode` accepts `"time_close"` (the
default) or `"time_open"`; an absent or NaN value means `"time_close"`.
Market entries fill at the first eligible candle open; limit entries are
tested for a fill on their live candles. Results are one row per plan, sorted
by start-candle position.

Output rows with status `filled` contain realized entry, stop, target, risk,
exit, R-multiple, pip P&L, and bars held. Other statuses are
`unfilled_expired`, `skipped_busy`, `skipped_risk_filter`,
`skipped_bad_levels`, and `no_data`; their trade-specific fields are null.
Plans whose requested live windows overlap are invalid. A filled trade keeps
the single-position lock through its exit candle.

The price model uses the configured constant spread and slippage. Long
entries pay spread and market-entry slippage; short stop/target triggers use
the ask. Stop exits include slippage and gap-through handling, while targets
do not receive price improvement. Stop wins when both stop and target are
reachable in one candle. A limit-fill candle checks only the stop because
intracandle ordering is unknown.

`"time_close"` evaluates the exit candle's stop and target before exiting at
its close, as in the original behavior. `"time_open"` is only valid for market
plans: candles before the exit position are checked for stops and any target;
the exit candle is not checked and the position exits at its open, with the
usual side-specific spread and slippage. If `time_exit_at` is later than the
last candle, `"time_open"` uses the existing `data_end` behavior instead:
the last candle is checked and the trade exits at its close.

## Holdout protection

By default, `run_backtest` rejects any input whose last timestamp is at or
after `HOLDOUT_START`. Research and development tests must use
`development_set()` first. `allow_holdout=True` exists only for a final,
one-time evaluation after a candidate has passed development; this task did
not run any real-data backtest or calculate strategy statistics.

## Metrics

`src.backtest.metrics.summarize_trades(results)` reports counts for statuses
present in the results and calculates performance metrics only from `filled` rows,
ordered stably by exit time. Win rate counts strictly positive R; losing
streaks count R values less than or equal to zero. Average loss uses strictly
negative R values. Expectancy excluding the top five removes the five largest
individual R multiples and is `None` for five or fewer trades.

With no filled trades, expectancy, profit factor, and average win/loss are
`None`; total R, win rate, drawdown, and losing streak are zero. With trades
but no losses, profit factor is infinity only when gross profit is positive;
it is `None` for a zero-only result. A missing win or loss group has a `None`
average. These metrics describe a mechanical, unproven hypothesis and do not
imply profitability.
