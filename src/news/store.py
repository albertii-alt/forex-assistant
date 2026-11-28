"""Atomic local storage for validated news calendars."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.news.events import parse_events


DEFAULT_NEWS_DIR = "data/news"
CALENDAR_FILENAME = "calendar_thisweek.json"
FORMAT_VERSION = 1
SOURCES = ("feed", "manual_file")


def _aware_datetime(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("fetched_at must be timezone-aware")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("fetched_at must be timezone-aware")
    return value


def save_calendar(
    raw_events: Any,
    fetched_at: datetime,
    *,
    source: str,
    directory: str | Path = DEFAULT_NEWS_DIR,
) -> Path:
    """Validate and atomically save raw feed records."""
    events = parse_events(raw_events)
    if not events:
        raise ValueError("no events to save")

    fetched_at = _aware_datetime(fetched_at).astimezone(timezone.utc)
    if source not in SOURCES:
        raise ValueError(f"source must be one of: {', '.join(SOURCES)}")

    path = Path(directory) / CALENDAR_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "source": source,
        "events": raw_events,
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{CALENDAR_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise

    return path


def load_calendar(directory: str | Path = DEFAULT_NEWS_DIR) -> dict:
    """Load a validated calendar without raising for missing or bad files."""
    path = Path(directory) / CALENDAR_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {
            "found": False,
            "events": [],
            "fetched_at": None,
            "source": None,
            "error": "calendar file not found",
        }
    except (OSError, UnicodeError) as exc:
        return _invalid_calendar(f"could not read file: {exc}")

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        return _invalid_calendar(f"invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return _invalid_calendar("top level must be an object")

    required = ("format_version", "fetched_at", "source", "events")
    missing = [key for key in required if key not in payload]
    if missing:
        return _invalid_calendar(f"missing key {missing[0]}")
    if type(payload["format_version"]) is not int or payload["format_version"] != FORMAT_VERSION:
        return _invalid_calendar("unsupported format version")

    fetched_text = payload["fetched_at"]
    if not isinstance(fetched_text, str):
        return _invalid_calendar("fetched_at must be a string")
    try:
        fetched_at = datetime.fromisoformat(fetched_text)
    except ValueError as exc:
        return _invalid_calendar(f"invalid fetched_at: {exc}")
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        return _invalid_calendar("fetched_at must include a UTC offset")
    fetched_at = fetched_at.astimezone(timezone.utc)

    source = payload["source"]
    if source not in SOURCES:
        return _invalid_calendar("invalid source")

    try:
        events = parse_events(payload["events"])
    except ValueError as exc:
        return _invalid_calendar(str(exc))
    if not events:
        return _invalid_calendar("no events")

    return {
        "found": True,
        "events": events,
        "fetched_at": fetched_at,
        "source": source,
        "error": None,
    }


def _invalid_calendar(message: str) -> dict:
    return {
        "found": False,
        "events": [],
        "fetched_at": None,
        "source": None,
        "error": f"calendar file invalid: {message}",
    }
