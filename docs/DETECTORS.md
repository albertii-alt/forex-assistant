# Detectors

## Swing highs and swing lows

`detect_swings(left, right)` marks a candle as a swing high when its high is
strictly above all `left` earlier highs and greater than or equal to all
`right` later highs. It marks a low when its low is strictly below all
`left` earlier lows and lower than or equal to all `right` later lows.
`confirmed_at` is the OPEN time of the candle `right` positions later.
Windows count candles, not elapsed time.

## Fair value gaps

`detect_fvgs(min_gap)` marks a bullish gap where candle 3's low is strictly
above candle 1's high, and a bearish gap where candle 3's high is strictly
below candle 1's low. Candle 2 is not used for the zone. `formed_at` is the
third candle's OPEN time. Touches and closes-through are searched only in
candles strictly after formation.

## Structure breaks

`detect_structure_breaks()` tracks the latest confirmed swing high or low.
A bullish break requires a close strictly above an active swing-high level;
a bearish break requires a close strictly below an active swing-low level.
Wicks and equality do not break the level. A broken or superseded swing is
abandoned and cannot break again. `broke_at` is the breaking candle's OPEN
time.

## Liquidity sweeps

`detect_liquidity_sweeps()` tracks swing highs/lows and UTC previous-day
extremes. A swing level is usable after its confirmation candle; a
previous-day level is known at the end of the prior trading key. The first
candle that trades beyond the level consumes it. If it closes back inside,
it emits a sweep; if it closes beyond the level, it emits no sweep.
Daily keys are UTC-based, and Sunday UTC candles are merged into Monday.

## Timing rule

All timestamps are candle OPEN times. A fact confirmed at a candle close is
stamped with that candle's open and can only be acted on from the next
candle's open.
