# Risk arithmetic helpers

`src.risk.sizing.position_size(...)` estimates a position size from the
account balance, risk percentage, entry, stop, and estimated pip costs. It
rounds size down to the configured unit step and reports planned and actual
risk amounts. It does not choose trade direction.

`src.risk.limits.evaluate_daily_limits(...)` checks the supplied counts and
realized P&L against the configured daily limits, returning whether another
trade is allowed and the reasons that apply. It does not track trades or
positions.

The default cost of 1.4 pips = 1.0 spread + 0.2 slippage on entry + 0.2
slippage on the stop exit in the backtest cost model. Real spreads, slippage
and gaps can be larger, so the function is a planning aid and not a guarantee
of maximum loss.

Assumptions:

- Account currency is USD.
- The pair is EUR/USD (USD is the quote currency, so one unit of the pair
  moves the account by exactly the price change in USD).
- One standard lot is 100,000 units.
- The smallest tradable size and size step are 0.01 lot, or 1,000 units.
- Realized P&L is in account currency and includes closed trades belonging to
  the current risk day only.

Deciding which trades belong to the 17:00 New York risk day is the caller's
responsibility; tracking and assigning trades to that day is a future journal
task.

Arithmetic helper only, not financial advice; verify contract specifications with your broker.
