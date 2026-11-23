# Strategy Rules (DRAFT, UNPROVEN HYPOTHESIS)

Status: not yet defined. No strategy code may be written until the owner
approves the exact definitions below.

## Origin
Based on a summary of a YouTube tutorial by TJR, which describes a
discretionary approach (liquidity, market structure, fair value gaps). It has
not been independently tested. Our version is our own interpretation.

## Candidate concepts (exact definitions TBD in Phase 4)
- Swing high / swing low
- Liquidity levels (equal highs/lows, previous day and session highs/lows)
- Liquidity sweep
- Change in market structure (break of structure)
- Fair value gap and inverse fair value gap
- Equilibrium (50% of a swing range)
- Daily bias

## Setup under test (rough outline)
Liquidity sweep, then structure break, then entry at a fair value gap, with
a defined stop and target and a minimum reward-to-risk.

## Rules
- Definitions are chosen after viewing real charts, and each one gets tests.
- The strategy is only accepted if it shows an edge after spread costs,
  out-of-sample, across different periods.