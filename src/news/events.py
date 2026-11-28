"""Parsing and freshness checks for weekly news-calendar events."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from numbers import Real
from typing import Any


IMPACTS = ("low", "medium", "high", "holiday")


def _aware_datetime(name: str, value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be timezone-aware")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def parse_events(raw: Any) -> list[dict]:
    """Validate and normalize feed records."""
    if not isinstance(raw, list):
        raise ValueError("raw must be a list of dicts")

    events: list[dict] = []
    seen: set[tuple[str, str, datetime]] = set()
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"record {index}: record must be a dict")

        missing = [key for key in ("title", "country", "date", "impact") if key not in record]
        if missing:
            raise ValueError(f"record {index}: missing required key {missing[0]}")

        title = record["title"]
        if not isinstance(title, str):
            raise ValueError(f"record {index}: title must be a string")
        title = title.strip()
        if not title:
            raise ValueError(f"record {index}: title must not be empty")

        country = record["country"]
        if not isinstance(country, str):
            raise ValueError(f"record {index}: country must be a string")
        country = country.strip().upper()
        if len(country) != 3 or not country.isascii() or not country.isalpha():
            raise ValueError(f"record {index}: country must be a 3-letter currency code")

        date_text = record["date"]
        if not isinstance(date_text, str):
            raise ValueError(f"record {index}: date must be a string")
        try:
            parsed = datetime.fromisoformat(date_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {index}: invalid date") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"record {index}: date must include a UTC offset")
        time_utc = _utc(parsed)

        impact = record["impact"]
        if not isinstance(impact, str):
            raise ValueError(f"record {index}: impact must be a string")
        impact = impact.strip().lower()
        if impact not in IMPACTS:
            raise ValueError(f"record {index}: unknown impact")

        forecast = record.get("forecast", "")
        if forecast is None:
            forecast = ""
        if not isinstance(forecast, str):
            raise ValueError(f"record {index}: forecast must be a string")
        forecast = forecast.strip()

        previous = record.get("previous", "")
        if previous is None:
            previous = ""
        if not isinstance(previous, str):
            raise ValueError(f"record {index}: previous must be a string")
        previous = previous.strip()

        key = (title, country, time_utc)
        if key in seen:
            continue
        seen.add(key)
        events.append({
            "title": title,
            "currency": country,
            "impact": impact,
            "time_utc": time_utc,
            "forecast": forecast,
            "previous": previous,
        })

    events.sort(key=lambda event: (
        event["time_utc"],
        event["currency"],
        event["title"],
    ))
    return events


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def calendar_status(
    events: list[dict],
    fetched_at: datetime,
    now: datetime,
    *,
    max_age_hours: float = 36,
    margin_hours: float = 24,
) -> dict:
    """Assess whether the supplied calendar is fresh and usable."""
    fetched_at = _aware_datetime("fetched_at", fetched_at)
    now = _aware_datetime("now", now)
    max_age = _finite_real("max_age_hours", max_age_hours)
    margin = _finite_real("margin_hours", margin_hours)
    if max_age < 0:
        raise ValueError("max_age_hours must be at least 0")
    if margin < 0:
        raise ValueError("margin_hours must be at least 0")

    fetched_at = _utc(fetched_at)
    now = _utc(now)
    problems: list[str] = []
    if not events:
        problems.append("empty_calendar")
    if fetched_at > now:
        problems.append("fetched_in_future")
    if now - fetched_at > timedelta(hours=max_age):
        problems.append("calendar_too_old")
    if events:
        first = events[0]["time_utc"]
        last = events[-1]["time_utc"]
        if now < first - timedelta(hours=margin) or now > last + timedelta(hours=margin):
            problems.append("now_outside_calendar_week")

    return {"usable": len(problems) == 0, "problems": problems}
