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

The setup window, setup deadline, and candle-open time exit may be supplied
through keyword-only parameters. Their defaults remain 09:30, 11:30, and
11:45 America/New_York respectively. The parameters are research settings
for the pre-registered settings ladder, not additional strategy concepts.

For research, load the 10-year candles and apply `development_set()` before
calling the plan finder. The holdout remains sealed unless a candidate first
passes the development acceptance criteria, and may then be evaluated only
once.

Candidate A detector frames are expected in detector order, with break and
FVG rows ordered by detector time. For an injected unsorted break frame,
equal positions keep their relative row order and the earliest valid position
is used. This fallback does not change detector-produced frames.

## Candidate B plan generation

`find_candidate_b_plans(df)` implements the fixed B-2R setup rules in
`docs/strategy_rules.md` v0.3. It returns plans only; it does not simulate
fills or establish profitability. Candidate B uses 15-minute BID candles,
15-minute supply/demand zones, the slow stochastic %K, and the 4-hour trend
state aligned only after each 4-hour candle closes.

For each zone, the finder considers only the first post-known, pre-expiry
touch. It then considers only the first %K crossing back inside the relevant
80/20 extreme in the confirmation window. A crossing rejected by the closed
through, trend, entry-time, or cost-adjusted risk filters does not cause a
later touch or crossing to be tried. The trend must be down for a supply-zone
short or up for a demand-zone long at the confirmation candle.

The market order starts at the next candle's open. Its ET entry window is
09:30 through 11:45 inclusive. Risk includes the engine's entry-cost model:
shorts fill at BID open less 0.2 pip slippage; longs fill at BID open plus
1.0 pip spread and 0.2 pip slippage. Stops are 2 pips beyond the zone's far
side, and only 5–30 pip cost-adjusted risk is accepted. The fixed target is
2R, and the planned time exit is the open of the 15:45 ET candle. The finder
selects at most one plan per ET calendar date: earliest entry first, then
latest zone `known_at`, then latest `base_time`.

All timestamps are normalized to nanoseconds internally. For the default
trend calculation, a constant `volume` column is added only to an internal
copy when the input has no volume; volume is needed by the shared ET resampler
but does not affect trend state. For real-data plan counts, call
`development_set(load_candles_10y())` before the finder. The holdout remains
sealed until the pre-registered development gate is passed.

## Candidate C: 4-hour Donchian breakout

`find_candidate_c_plans(df)` discovers fixed 20-period Donchian breakout
plans on 4-hour candles aligned to 17:00 America/New_York. A long signal
requires a closed candle close above the previous 20 4-hour highs; a short
requires a close below the previous 20 lows. The current signal candle is
excluded. It uses Wilder ATR(20) at the signal candle and a fixed initial
stop of 2 ATR from the following 4-hour candle open.

The market entry is the open of the first 15-minute candle in the following
4-hour bin. The plan has no target and uses `exit_mode="time_open"`. A long
exit signal is the first closed candle after entry whose close is below the
previous 10 lows; the short mirror uses the previous 10 highs. Exit is at the
open of the following 4-hour bin. Partial final bins may provide entry timing
but never provide signal or exit information until closed.

`count_candidate_c_entries(df, plans)` independently applies the busy rule,
fixed stop, time-open exit, and data-end behavior for price-only entry counts.
