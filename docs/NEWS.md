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

## Fetching the calendar

From the project folder, run `python -m scripts.fetch_news_calendar`. Fetch at
most about once a day; the script enforces a six-hour minimum interval. The
weekly feed is unofficial and may block requests or change.

If the script says the feed refused the request (403 or 429) or otherwise
fails, open <https://nfs.faireconomy.media/ff_calendar_thisweek.json> in a
browser, save it as `data/news/manual.json`, then run
`python -m scripts.fetch_news_calendar --from-file data/news/manual.json`.
The `data/news/` directory is git-ignored.

If the calendar is missing or stale the tools say so and never assume there is
no news.

## News on the screen

The journal screen uses the saved calendar to show a Manila-time briefing and
warn about unavailable calendars or active blackout windows. Fetching remains
a manual command; the screen does not fetch calendar data. These warnings are
advice only and do not block saving a journal plan.

If the calendar is missing or stale the tools say so and never assume there is
no news.

Planning aid only, not financial advice.
