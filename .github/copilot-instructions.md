# Personal Forex Assistant: Standing Instructions

## Purpose
A personal research and decision-support tool for forex day trading. The owner
is a beginner. A separate reviewer (Claude) reviews every report before the
next task is assigned.

## Hard rules
- This is a research and decision-support tool. NEVER add live order
  execution, and never put broker credentials or API keys in code.
- Free tools only. Do not install or add any package, paid service, or API
  without asking first. If you think one is needed, explain why and wait.
- Environment: Windows, PowerShell, Python virtual environment in `.venv`.
  Give PowerShell-compatible commands. Run modules as `python -m src.<path>`.
- NEVER run `git commit`, `git push`, or any git command that changes the
  repository. The owner commits after review.
- Do ONLY the task requested. Do not start the next roadmap item, and do not
  refactor unrelated files.
- Do not modify or delete anything in `data/`. Do not delete existing files
  without asking.

## Code standards
- Python 3.11+ style, type hints, small functions, short docstrings.
- Use `pathlib` for paths. Never hardcode absolute paths.
- All timestamps are timezone-aware UTC. Convert to other zones (for example
  New York or London sessions) with `zoneinfo`, only at the point of use.
- NO LOOKAHEAD BIAS: any signal or pattern at candle N may only use
  information available at the close of candle N. If a pattern needs future
  candles to confirm, it must be labeled as confirmed at the later candle,
  not at the earlier one.
- Every detector or calculation gets pytest tests using small hand-made
  examples with known correct answers. Run the tests and show the output.
- The `dukascopy-python` library changes its API often. Only
  `src/data/download.py` may import it.

## Honesty about the strategy
- The trading strategy is an UNPROVEN hypothesis. Never claim or imply
  profitability. Backtests must include spread costs and must be free of
  lookahead bias.
- If something is untested, broken, or you are unsure, say so plainly.

## Required report after EVERY task
1. Summary (what you did, in plain language)
2. Files created or changed
3. How to run it (PowerShell commands)
4. Test results (paste the actual output)
5. Assumptions you made
6. Deviations from the request, and why
7. Problems or open questions
8. Suggested next step (do NOT implement it)