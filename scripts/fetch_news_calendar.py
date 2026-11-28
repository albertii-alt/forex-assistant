"""Fetch and store the unofficial weekly Forex Factory calendar."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from src.news.events import parse_events
from src.news.store import (
    DEFAULT_NEWS_DIR,
    load_calendar,
    save_calendar,
)


FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MIN_INTERVAL_HOURS = 6
TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
USER_AGENT = (
    "forex-assistant-personal/0.1 "
    "(personal non-commercial research tool; at most one fetch per day)"
)


class FeedError(Exception):
    """Raised for rejected or invalid feed responses."""


def fetch_feed(
    opener: Callable[..., Any] | None = None,
    url: str = FEED_URL,
    timeout: int = TIMEOUT_SECONDS,
) -> bytes:
    """Fetch one bounded response from the calendar endpoint."""
    if opener is None:
        opener = urlopen
    request = Request(
        url,
        method="GET",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code in (403, 429):
            raise FeedError(
                f"Feed refused with HTTP {exc.code}; use --from-file with a manual file."
            ) from exc
        raise FeedError(f"HTTP {exc.code} during calendar fetch") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"could not reach the calendar feed: {exc}") from exc

    if len(data) > MAX_RESPONSE_BYTES:
        raise FeedError("calendar response too large")
    return data


def parse_feed_bytes(data: bytes) -> list:
    """Decode a JSON feed response and require a list at the top level."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeedError("feed is not valid JSON") from exc
    if not isinstance(payload, list):
        raise FeedError("feed JSON top level must be a list")
    return payload


def _manila_time(value: datetime) -> str:
    return value.astimezone(ZoneInfo("Asia/Manila")).strftime("%Y-%m-%d %H:%M")


def _usage() -> str:
    return "Usage: python -m scripts.fetch_news_calendar [--from-file PATH]"


def run(
    argv: list[str] | None = None,
    *,
    opener: Callable[..., Any] | None = None,
    now: datetime | None = None,
    directory: str | Path = DEFAULT_NEWS_DIR,
) -> int:
    """Fetch or import a calendar, validating it before saving."""
    args = sys.argv[1:] if argv is None else argv
    manual_path: Path | None = None
    if args == []:
        pass
    elif len(args) == 2 and args[0] == "--from-file":
        manual_path = Path(args[1])
    else:
        print(_usage())
        return 2

    if now is None:
        now = datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        print("Calendar NOT updated: now must be timezone-aware")
        return 1
    now = now.astimezone(timezone.utc)

    if manual_path is None:
        existing = load_calendar(directory)
        if existing["found"] and now - existing["fetched_at"] < timedelta(
            hours=MIN_INTERVAL_HOURS
        ):
            print("Calendar was fetched less than 6 hours ago; not fetching again.")
            print(f"Last fetched: {_manila_time(existing['fetched_at'])} Manila time")
            return 1
        try:
            data = fetch_feed(opener)
        except FeedError as exc:
            print(f"Calendar NOT updated: {exc}")
            return 1
    else:
        try:
            data = manual_path.read_bytes()
        except OSError as exc:
            print(f"Calendar NOT updated: could not read manual file: {exc}")
            return 1

    try:
        raw_events = parse_feed_bytes(data)
        parsed_events = parse_events(raw_events)
        saved_path = save_calendar(
            raw_events,
            now,
            source="manual_file" if manual_path is not None else "feed",
            directory=directory,
        )
    except (FeedError, ValueError, OSError) as exc:
        print(f"Calendar NOT updated: {exc}")
        return 1

    print(f"Saved {len(parsed_events)} events to {saved_path}")
    print(
        f"First event: {_manila_time(parsed_events[0]['time_utc'])} Manila time; "
        f"last event: {_manila_time(parsed_events[-1]['time_utc'])} Manila time"
    )
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
