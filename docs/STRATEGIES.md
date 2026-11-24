# Strategy Plan Generation

Candidate A plan discovery is implemented in
`src/strategies/candidate_a.py` according to Candidate A in
`docs/strategy_rules.md` v0.3. The rules are a fixed, unproven hypothesis;
plan generation does not establish profitability.

`find_candidate_a_plans(df)` accepts sorted, timezone-aware UTC 15-minute
OHLC data and returns at most one plan per America/New_York calendar date.
It uses the configured session levels, structure breaks, and fair value gaps
to identify a valid setup. The resulting limit plans are inputs to the
cost-aware backtester; this function does not simulate fills or trades.

For research, load the 10-year candles and apply `development_set()` before
calling the plan finder. The holdout remains sealed unless a candidate first
passes the development acceptance criteria, and may then be evaluated only
once.

Candidate A detector frames are expected in detector order, with break and
FVG rows ordered by detector time. For an injected unsorted break frame,
equal positions keep their relative row order and the earliest valid position
is used. This fallback does not change detector-produced frames.
