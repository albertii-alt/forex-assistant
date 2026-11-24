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

## Strategy session levels

`detect_session_levels()` returns one row per ET trading date. Previous-day
highs and lows use the preceding trading date that has data, with the first
candle of the current date recorded as `prev_day_known_at`. The overnight
range covers 17:00 ET through, but not including, 09:30 ET; its levels are
known at the first 09:30-or-later candle before the 17:00 ET rollover.
Set-at fields identify the first candle attaining each extreme. If either
the overnight range or its confirming morning candle is absent, its level,
set-at time, and known-at time are null.

## Higher-timeframe trend state

`htf_trend_state()` classifies each higher-timeframe candle using the last two
confirmed swing highs and lows available at that candle's close. Both pairs
must rise for `up` or fall for `down`; all other cases are `none`. When using
this state on a lower timeframe, consume it only through
`src.data.resample_et.align_htf_to_ltf`, which uses the higher-timeframe
`close_time` so the state is unavailable before its candle closes.

## Supply and demand zones

`detect_zones()` marks the full wick range of a base candle when a later close
within the configured lookahead moves at least the configured ATR multiple
beyond its low (supply) or high (demand). ATR is measured at the base candle's
close. `known_at` is the confirming candle's open time: the zone is known at
that candle's close and usable from the next candle's open. Creation thresholds
are inclusive with a 1e-9 tolerance. A zone is closed through only by a later
close strictly beyond its far side; the close-through search includes the
expiry candle. `expires_at` is the open time `expiry_candles` after confirmation
when that candle exists in the data, otherwise it is `NaT`.

## Timing rule

All timestamps are candle OPEN times. A fact confirmed at a candle close is
stamped with that candle's open and can only be acted on from the next
candle's open.
