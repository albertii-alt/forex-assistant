# Trade journal

The SQLite journal records manual paper and demo trades for personal review
and supplies an as-of trading-day summary to the risk module. It does not
place orders or enforce position limits.

The `trades` table stores: `id`, `opened_at`, `closed_at`, `pair`, `side`,
`mode`, `strategy`, `entry_price`, `stop_price`, `target_price`, `size_units`,
`planned_risk_amount`, `reason`, `exit_price`, `realized_pnl`, `r_multiple`,
and `notes`. Times are stored as UTC ISO text at whole-second precision and
returned as timezone-aware UTC Python `datetime` objects.

Only `paper` and `demo` modes are accepted; `live` is deliberately unavailable.
Trades are inserted open and closed once; reason, strategy, and closing notes
are stripped. The summary counts trades opened in the current trading day,
realized P&L from closes in that day, and positions open as of the requested
time. Events after that time are excluded. The journal does not enforce the
one-position rule; `evaluate_daily_limits` does.

Trading days use `trading_day` from `src.data.resample_et`: they begin at
17:00 America/New_York, inclusive, and use that function's naive midnight
trading-day labels.

The default journal file is `data/journal.sqlite3`, in the git-ignored
`data/` directory.

Personal record only, not financial advice.

Limits note: the daily loss limit built on this journal uses realized (closed)
results only. An open trade that is currently losing is not counted. The
one-open-position rule is what limits exposure while a trade is open.

## Planning workflow

`src.journal.workflow` combines realized journal balances, the as-of
`trading_day_summary`, daily limits, and cost-aware position sizing. The
start-of-day balance includes realized P&L only from trading days strictly
before the current trading day; the current balance includes realized P&L
closed by the requested time. Future opens and closes do not affect a plan.

`plan_trade` is a read-only decision aid. Its `blocking_reasons` preserve the
daily-limit reason order and append `size_too_small` when the rounded position
does not meet the minimum lot step. `open_planned_trade` recomputes the plan
at the proposed open time and saves only when no blocking reason remains. It
stores the rounded position size and its actual planned risk, not the nominal
risk budget. This workflow records plans in the journal; it does not place
orders or guarantee a maximum loss.

## Journal screen

Run the screen from the project folder with
`streamlit run src/app/journal_page.py`. The Today section shows the current balance,
trading-day limits, and recent journal entries; Plan a trade checks a proposed
paper or demo trade before saving; Close a trade records the result from the
account statement. The starting balance is typed on the screen and is NOT
stored, so enter the same number every time. The saved result must come from
the user's paper or demo account statement.

Paper and demo only, experimental. Not financial advice.

## Review

The Review section summarizes all closed journal trades, not just the current
trading day. R is the realized result divided by planned risk. Win rate is the
share of closed trades with positive R. Average R is the mean result per closed
trade. The range is only shown from 30 closed trades, because with fewer trades
plus or minus 2 standard errors is far too narrow and would look more precise
than it is; from 30 trades it is a rough guide that assumes outcomes are roughly
bell-shaped.

Profit factor is gross positive R divided by the absolute gross negative R.
Max drawdown in R is the largest fall in cumulative R, while the longest losing
streak counts consecutive losses. With few trades, luck can dominate the
outcome, so the sample-size note explains why averages are not yet conclusive.

Planning aid only, not financial advice.