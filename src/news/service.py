"""News-calendar context for application screens."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.news.blackout import (
    blackout_state,
    build_blackout_windows,
    format_briefing,
)
from src.news.events import calendar_status
from src.news.store import DEFAULT_NEWS_DIR, load_calendar


NEWS_DIR_ENV = "FOREX_NEWS_DIR"


def news_directory() -> str:
    """Return the configured calendar directory or its default."""
    return os.environ.get(NEWS_DIR_ENV) or DEFAULT_NEWS_DIR


def manila_text(dt: datetime) -> str:
    """Format an aware datetime in Manila local time."""
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("dt must be a timezone-aware datetime")
    return dt.astimezone(ZoneInfo("Asia/Manila")).strftime("%Y-%m-%d %H:%M")


def load_news_context(
    now: datetime,
    directory: str | Path = DEFAULT_NEWS_DIR,
    *,
    hours_ahead: float = 24,
) -> dict:
    """Load a calendar and return its briefing and trusted blackout state."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")

    loaded = load_calendar(directory)
    if not loaded["found"]:
        problems = (
            ["calendar_not_found"]
            if loaded["error"] == "calendar file not found"
            else ["calendar_file_invalid"]
        )
        status = {"usable": False, "problems": problems}
        briefing = format_briefing([], status, now, hours_ahead=hours_ahead)
        return {
            "usable": False,
            "problems": problems,
            "briefing": briefing,
            "in_blackout": False,
            "blackout_window": None,
            "fetched_at": None,
            "source": None,
            "error": loaded["error"],
        }

    events = loaded["events"]
    status = calendar_status(events, loaded["fetched_at"], now)
    briefing = format_briefing(
        events,
        status,
        now,
        hours_ahead=hours_ahead,
    )
    if status["usable"]:
        state = blackout_state(build_blackout_windows(events), now)
        in_blackout = bool(state["in_blackout"])
        blackout_window = state["window"]
    else:
        in_blackout = False
        blackout_window = None

    return {
        "usable": status["usable"],
        "problems": status["problems"],
        "briefing": briefing,
        "in_blackout": in_blackout,
        "blackout_window": blackout_window,
        "fetched_at": loaded["fetched_at"],
        "source": loaded["source"],
        "error": None,
    }
