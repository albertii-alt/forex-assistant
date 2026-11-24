# Strategy Rules v0.3 (APPROVED, UNPROVEN HYPOTHESIS)

Status: approved by the owner on 2026-11-23. Strategy code may be written only
after the groundwork (data, ET-aligned candles, session levels, new detectors)
is done and tested.

IMPORTANT: These are OUR mechanical interpretations of ideas from YouTube
videos and a free checklist. They are not the creators' strategies. The
creators describe discretionary methods and show no verified results. A pass
or fail here says nothing about their discretionary methods, only about OUR
rules. Rules marked [VIDEO] come from a video or the SLC blueprint; rules
marked [OURS] are our own choices.

## Pre-declared tests (fixed before any results)
Exactly two candidates, built in this order: A, then B.
- Candidate A, primary config: A-2R.   Exploratory variant: A-LIQ.
- Candidate B, primary config: B-2R.   Exploratory variant: B-RETEST.
Only the PRIMARY configs can be accepted. A variant that passes while its
primary fails is NOT accepted (it needs new data to confirm, which we will
not have). No new candidates or variants are added without recording them in
the changelog at the bottom.

## Data and splits [OURS]
- EUR/USD, 15-minute bid candles, about 10 years (2016-11 to 2026-10;
  availability to be verified when downloading).
- Development set: everything before 2024-11-01.
- Holdout set: 2024-11-01 to the end of data (sealed). Nobody looks at
  holdout results until a candidate has passed on the development set. Each
  candidate's primary config touches the holdout exactly once.
- Any rule change after seeing development results is logged in the
  changelog as a new variant. The holdout stays sealed.

## Time conventions [OURS]
- All session logic uses America/New_York (DST-aware).
- A trading day runs 17:00 ET to 17:00 ET. Daily candles and 4h candles are
  aligned to 17:00 ET (4h boundaries: 17, 21, 01, 05, 09, 13 ET). Weekend
  gaps are not filled.
- A higher-timeframe candle may be used only after it has fully closed.
- Previous trading day = the preceding trading day that has data.

## Shared mechanics [OURS]
- Results in R multiples (1R = initial risk). No compounding.
- Spread: constant 1.0 pip, modeled as ask = bid + 1.0 pip. Longs enter at
  ask and exit at bid; shorts enter at bid and exit at ask (their stops and
  targets trigger on the ask side).
- Slippage: 0.2 pip against us on market entries, stop exits and time exits.
  None on limit fills or target fills.
- If stop and target are both reachable within the same candle, the STOP is
  assumed hit first.
- Report-only sensitivity check: rerun with 2.0 pip spread. Not used for
  acceptance.
- One open position at a time. Every setup uses information available at the
  close of each candle only (no lookahead).

## Candidate A: New York morning day trade (15m)
Inspired by video 1 (liquidity sweep, then confirmation, then continuation).
- [VIDEO] Concepts: liquidity sweep, break of structure, fair value gap,
  New York time. Examples there used S&P 500 / NASDAQ, not EUR/USD.
- [OURS] Everything below, including all numbers.

Window: setups start at 09:30 ET and must be complete by 11:30 ET. Unfilled
orders are cancelled and open trades closed at the 12:00 ET candle close.
(Window start 08:00 ET may be reported as a sensitivity check only.)

Reference levels: overnight high/low (17:00 ET to 09:30 ET) and previous
trading day high/low (17:00 ET basis).

Long setup (short is the exact mirror):
1. SWEEP inside the window: a candle's low is below a reference low and its
   close is back above it.
2. CONFIRMATION within 8 candles after the sweep candle: a bullish structure
   break (close above a confirmed swing high; swings left=right=3).
3. ENTRY ZONE: a bullish fair value gap, size >= 2 pips, formed after the
   sweep candle and no later than the break candle.
4. ENTRY: limit order at the gap's upper edge, placed once the gap is formed,
   valid for 8 candles. Fills if a later candle's low <= that price.
5. STOP: 1 pip below the sweep candle's low. Skip if the stop distance is
   under 5 or over 25 pips.
6. CANCEL before fill if a candle closes below the sweep low.
7. Maximum 1 trade per trading day; only the first valid setup counts.

A-2R (primary): target = 2 x the entry-to-stop distance.
A-LIQ (variant): target = the nearest opposite reference level; skip the
trade if that is less than 1.5R away.

## Candidate B: SLC day trade (4h structure, 15m entries)
Inspired by video 3 and the free "SLC Execution Blueprint" checklist.
- [VIDEO] Trade only in the direction of a non-ranging 4h structure; supply
  and demand levels from aggressive moves; a level broken more than once is
  invalid; the stochastic (5,3,3) moves into an extreme zone then crosses
  back inside while price is at the level; stop beyond the level; target at
  least 2R; market order at the candle close.
- [OURS] All numbers and definitions below. The video's break-and-retest
  scenario is a logged variant, not part of the primary. Examples used
  NASDAQ futures on 5m; we use EUR/USD on 15m.

Structure: 4h candles aligned to 17:00 ET, confirmed swings left=right=3. The
last two swing highs AND the last two swing lows both rising = uptrend (longs
only); both falling = downtrend (shorts only); anything else = no trade. Only
swings confirmed at or before the current candle count.

Zones (15m): a supply zone is the full range (wicks included) of a base
candle, where within the next 3 candles a candle closes at least 2 x ATR(14)
below the base candle's low (ATR measured at the base candle's close). The
zone is known at the close of that candle. Demand is the mirror. A zone
expires after 192 candles (2 days) or when a candle closes through its far
side. Primary: only zones that have never been closed through.

Confirmation (short; long is the mirror): first touch s = the first candle
after the zone is known whose high reaches the zone's lower edge. Stochastic
(5,3,3, slow) levels 80/20. Confirmation candle t = the first candle in
[s, s+6] where %K[t-1] >= 80 and %K[t] < 80, provided the zone has not been
closed through before t. The trade direction must match the 4h structure at
t.

Entry: market order at the open of the candle after t (the conservative
version of "at the candle close", which cannot be acted on at its own close
price). Stop: 2 pips beyond the zone's far side. Skip if the stop distance is
under 5 or over 30 pips. Target: 2R, fixed.

Session: entries only 09:30-12:00 ET; any open trade is closed at the 16:00
ET candle close; maximum 1 trade per trading day.

B-2R (primary): as above.
B-RETEST (exploratory variant): also allows a zone that was closed through
exactly once, entered only after price retests it from the other side.

## Acceptance test (primary configs only; defined before any results)
Development set:
- At least 100 trades, otherwise the result is INCONCLUSIVE (not accepted).
- Expectancy after costs > 0R.
- Expectancy stays > 0R after removing the 5 best trades.
- Profit factor > 1.2.
Holdout (only if the development test passed):
- At least 30 trades, otherwise INCONCLUSIVE (not accepted, and the
  strategy stays on paper only).
- Expectancy > 0.1R and profit factor > 1.2 after costs.
Outcomes:
- ACCEPTED: all tests passed. Next step is paper trading, not real money.
- REJECTED: development or holdout test failed.
- INCONCLUSIVE: too few trades. More data or a longer wait is needed.
Never report a variant, a different window, or a different parameter as an
accepted result.

## Process rules
- Build and test A first. Build B afterwards, reusing the backtester.
- Parameters above are fixed starting points. No optimization.
- Daily bias, inverse FVGs, equilibrium filters, news filters, and weekly
  momentum are NOT part of this version. Each may be added later as a single
  logged variant.

## Changelog
- v0.2: Declared candidates A and B (swing version), acceptance rules, data
  splits.
- v0.3: Replaced the swing Candidate B with the SLC day-trading Candidate B,
  before any backtest or result existed. Added the SLC blueprint as a source
  (stochastic 5,3,3 and market-order entry taken from it). Slippage now also
  applies to market entries.
  - Protocol log (2026-11-23): during groundwork step 2, structural checks
  (candle counts, bin alignment, volume totals) briefly ran over the full file
  including the holdout period. No returns or performance statistics were
  computed. Corrected so real-data tests use development_set() only. The data
  quality report (counts, gaps, largest candle ranges) also describes the whole
  file by design.

  - Clarification (2026-11-23, before any results): in the backtest engine a
  long limit order fills when the ASK reaches the limit (bid low + 1 pip <=
  limit), consistent with "longs enter at ask"; a short limit fills when the
  bid high >= limit. Candidate A rule 6 (cancel if a candle closes below the
  sweep low) can never trigger before a fill, because price must trade
  through the limit first; a failed setup simply fills and stops out.

  - Clarifications for Candidate A implementation (2026-11-23, before any
  results): (1) sweep = a candle in the window whose low is below a known
  reference low and whose close is >= that level (mirror for highs); levels
  are not consumed by earlier breaches. (2) Only the first matching break
  within 8 candles after a sweep candle is considered. (3) Entry zone = the
  most recently formed qualifying FVG (size >= 2 pips, formed after the sweep
  candle and no later than the break candle). (4) The limit order is placed at
  the close of the break candle, never before. (5) "Complete by 11:30 ET": the
  break candle must close by 11:30 ET. (6) "Open trades closed at 12:00 ET":
  exit at the close of the candle that closes at 12:00 ET (the 11:45 ET
  candle). (7) "Only the first valid setup counts": at most one plan per
  trading day, chosen by earliest order placement; setups failing the
  stop-size filter are not valid; a plan that never fills produces no trade
  that day. (8) The 8-candle limit never binds inside the window but is
  implemented. (9) A-LIQ and the 08:00 window sensitivity check are deferred.

  - Pre-registered plan-count test (2026-11-23, BEFORE any Candidate A backtest
  and before counts at the settings below were seen): v0.3 Candidate A yields
  68 plans on development data (about 8 per year), too few for the 100-trade
  minimum. Settings ladder, each rung cumulative:
  R0 = v0.3 as written.
  R1 = R0 with min_fvg_pips 1.0.
  R2 = R1 with window start 08:00 ET.
  R3 = R2 with setup deadline 12:30 ET, time exit at the 12:45 ET candle open
       (13:00 ET close), confirm_candles 12.
  Selection rule: using development data and counts only (never returns), the
  lowest rung whose estimated FILLED development trades is at least 150
  becomes v0.4's Candidate A primary (A-2R at those settings). If no rung
  reaches 150, Candidate A is dropped as INCONCLUSIVE and nothing further is
  loosened. No other settings are tried. Candidate A at v0.4 settings is a
  different strategy from v0.3 and is described as such; the acceptance test,
  costs and sealed holdout are unchanged.

  - Candidate A result (2026-11-23): the count-only ladder on development data
  gave 19, 22, 51 and 80 estimated filled plans for R0-R3 (68, 78, 139 and
  206 plans). No rung reached the pre-registered 150, so Candidate A is
  INCONCLUSIVE and is dropped. No Candidate A backtest was run and no returns
  were seen. No further rungs will be tried. The setup-finder code stays in
  the repository as a tested, unused module.
- Candidate B gate and clarifications (2026-11-23, before any Candidate B
  results): (1) Count gate: after the B setup finder is built, count plans on
  development data (no backtest). B uses market entries, so plans = trades.
  Require at least 130 development plans (the holdout is about a quarter of
  the development length, leaving about 30 for it). If fewer, B is
  INCONCLUSIVE and nothing is loosened. (2) ATR(14) uses Wilder smoothing:
  TR = max(H-L, |H-prevC|, |L-prevC|), first candle TR = H-L, first ATR = mean
  of the first 14 TRs, then ATR = (prev*13 + TR)/14. (3) Stochastic (5,3,3):
  raw K = 100*(C - lowest low of the last 5 candles including the current one)
  / (highest high - lowest low), NaN when the range is zero; %K = SMA(3) of
  raw K; %D = SMA(3) of %K. Rules use %K with levels 80/20. (4) The 4h trend
  state at 4h candle c uses only swings (left=right=3) confirmed at or before
  c, and is usable on 15m candles only from the first open at or after c's
  close_time.