# News calendar

This module provides a local calendar parser and news briefing logic. It has no
network fetching or file-reading behavior.

The feed format is a JSON list of records with `title`, `country`, `date`,
`impact`, `forecast`, and `previous`. Dates use ISO 8601 offsets, impact values
are Low, Medium, High, or Holiday, and forecast/previous remain free text.
Forecast and previous values are never parsed as numbers.

Defaults are 30 minutes before and after high-impact events, with EUR and USD
selected for blackout checks. Briefings use a 24-hour look-ahead.

The feed is an unofficial weekly JSON feed of Forex Factory data and may change
or disappear.

Times are shown in Manila time.

If the calendar is missing or stale the tools say so and never assume there is
no news.

Planning aid only, not financial advice.
