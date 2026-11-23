# Strategy Rules v0.2 (DRAFT, UNPROVEN HYPOTHESIS)

Status: awaiting owner approval. No strategy code until approved.

IMPORTANT: These are OUR mechanical interpretations of ideas from two YouTube
videos. They are not the creators' strategies. Both creators describe
discretionary methods and show no verified results. A pass or fail here says
nothing about their discretionary methods, only about OUR rules. Rules marked
[VIDEO] come from a video; rules marked [OURS] are our own choices.

## Pre-declared tests (fixed before any results)
Exactly two candidates, built in this order: A, then B.
- Candidate A, primary config: A-2R.   Exploratory variant: A-LIQ.
- Candidate B, primary config: B-1.5R. Exploratory variant: B-2R.
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
- A trading day runs 17:00 ET to 17:00 ET. Daily candles and 4h candles for
  both candidates are aligned to 17:00 ET (4h boundaries: 17, 21, 01, 05, 09,
  13 ET). Weekend gaps are not filled.
- Previous trading day = the preceding trading day that has data.

## Shared mechanics [OURS]
- Results in R multiples (1R = initial risk). No compounding.
- Spread: constant 1.0 pip, modeled as ask = bid + 1.0 pip. Longs enter at
  ask and exit at bid; shorts enter at bid and exit at ask (their stops and
  targets trigger on the ask side).
- Slippage: 0.2 pip against us on stop exits and time exits. None on limit
  fills.
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
7. Maximum 1 trade per day; only the first valid setup counts.

A-2R (primary): target = 2 x the entry-to-stop distance.
A-LIQ (variant): target = the nearest opposite reference level; skip the
trade if that is less than 1.5R away.

## Candidate B: daily structure swing trade (4h entries)
Inspired by video 2 (four confluences on weekly/daily/4h charts).
- [VIDEO] Four confluences (market structure, liquidity, premium/discount
  at the 0.5 level, change of character); stop beyond the swing; target of
  at least 1:1.5 and never above 1:2; 4h entries.
- [OURS] All definitions and numbers below. Weekly momentum is not coded.
  The video's subjective filters ("choppy", "strong low", "clear liquidity",
  "overextended") are replaced by the measurable rules here.
- The video's 2% risk and two positions per leg are NOT used.

Long setup (short is the exact mirror):
1. STRUCTURE (daily): using confirmed daily swings (left=right=3), the last
   two swing highs are rising and the last two swing lows are rising. If the
   daily structure is not clearly up or down, no trade.
2. LIQUIDITY: a 4h candle sweeps (wick through, close back above) a
   confirmed daily swing low formed within the last 20 daily candles.
3. CHANGE OF CHARACTER: within 12 4h candles after the sweep, a bullish 4h
   structure break (close above a confirmed 4h swing high; swings 3/3).
4. PREMIUM/DISCOUNT: the entry price must be at or below 50% of the range
   from the latest confirmed daily swing low to the latest confirmed daily
   swing high.
5. ENTRY: at the open of the 4h candle after the break candle closes,
   provided rule 4 holds; otherwise skip.
6. STOP: 1 pip beyond the swept extreme (the low of the sweep candle).
   Skip if the stop distance is under 20 or over 200 pips.
7. TIME EXIT: close at the candle close after 180 4h candles (about 30
   days) if neither stop nor target was hit.

B-1.5R (primary): target = 1.5 x the entry-to-stop distance.
B-2R (variant): target = 2 x the entry-to-stop distance.

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
- v0.2: Declared candidates A and B, acceptance rules, data splits.