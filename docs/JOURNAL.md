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