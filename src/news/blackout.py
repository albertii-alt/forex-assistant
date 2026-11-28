"""News blackout windows and briefing text."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from numbers import Real
from typing import Any
from zoneinfo import ZoneInfo

from src.news.events import IMPACTS


DEFAULT_CURRENCIES = ("EUR", "USD")
DEFAULT_BLACKOUT_IMPACTS = ("high",)
DEFAULT_MINUTES_BEFORE = 30
DEFAULT_MINUTES_AFTER = 30


def _aware_datetime(name: str, value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be timezone-aware")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validated_sequence(name: str, value: Any, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a non-empty tuple or list")
    result = tuple(value)
    if allowed is not None:
        if any(item not in allowed for item in result):
            raise ValueError(f"{name} contains an invalid value")
    return result


def build_blackout_windows(
    events: list[dict],
    *,
    currencies=DEFAULT_CURRENCIES,
    impacts=DEFAULT_BLACKOUT_IMPACTS,
    minutes_before: int = 30,
    minutes_after: int = 30,
) -> list[dict]:
    """Build merged closed intervals for matching events."""
    currencies = _validated_sequence("currencies", currencies)
    for currency in currencies:
        if (
            not isinstance(currency, str)
            or len(currency) != 3
            or not currency.isascii()
            or not currency.isalpha()
            or currency != currency.upper()
        ):
            raise ValueError("currencies must contain 3-letter uppercase codes")
    impacts = _validated_sequence("impacts", impacts, IMPACTS)
    minutes_before = _nonnegative_int("minutes_before", minutes_before)
    minutes_after = _nonnegative_int("minutes_after", minutes_after)

    intervals: list[dict] = []
    before = timedelta(minutes=minutes_before)
    after = timedelta(minutes=minutes_after)
    for event in events:
        if event["currency"] in currencies and event["impact"] in impacts:
            time_utc = event["time_utc"].astimezone(timezone.utc)
            intervals.append({
                "start": time_utc - before,
                "end": time_utc + after,
                "title": f"{event['currency']} {event['title']}",
                "time": time_utc,
                "currency": event["currency"],
                "event_title": event["title"],
            })

    intervals.sort(key=lambda item: (
        item["start"],
        item["end"],
        item["time"],
        item["currency"],
        item["event_title"],
    ))
    merged: list[dict] = []
    for interval in intervals:
        if merged and interval["start"] <= merged[-1]["end"]:
            if interval["end"] > merged[-1]["end"]:
                merged[-1]["end"] = interval["end"]
            merged[-1]["intervals"].append(interval)
        else:
            merged.append({
                "start": interval["start"],
                "end": interval["end"],
                "intervals": [interval],
            })

    windows: list[dict] = []
    for window in merged:
        ordered = sorted(
            window["intervals"],
            key=lambda item: (item["time"], item["currency"], item["event_title"]),
        )
        windows.append({
            "start": window["start"],
            "end": window["end"],
            "titles": [item["title"] for item in ordered],
        })
    return windows


def blackout_state(windows: list[dict], now: datetime) -> dict:
    """Return the first blackout window containing now, if any."""
    now = _aware_datetime("now", now)
    for window in windows:
        if window["start"] <= now <= window["end"]:
            return {"in_blackout": True, "window": window}
    return {"in_blackout": False, "window": None}


def _positive_finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite real number greater than zero")
    return number


def upcoming_events(
    events: list[dict],
    now: datetime,
    *,
    hours_ahead: float = 24,
    currencies=DEFAULT_CURRENCIES,
    impacts=("high", "medium", "holiday"),
) -> list[dict]:
    """Return matching events in the inclusive upcoming window."""
    now = _aware_datetime("now", now)
    hours_ahead = _positive_finite_real("hours_ahead", hours_ahead)
    currencies = _validated_sequence("currencies", currencies)
    impacts = _validated_sequence("impacts", impacts, IMPACTS)

    cutoff = now + timedelta(hours=hours_ahead)
    return [
        event for event in events
        if event["currency"] in currencies
        and event["impact"] in impacts
        and now <= event["time_utc"].astimezone(timezone.utc) <= cutoff
    ]


def format_briefing(
    events: list[dict],
    status: dict,
    now: datetime,
    *,
    hours_ahead: float = 24,
    currencies=DEFAULT_CURRENCIES,
    impacts=("high", "medium", "holiday"),
    blackout_impacts=DEFAULT_BLACKOUT_IMPACTS,
    minutes_before: int = 30,
    minutes_after: int = 30,
) -> str:
    """Format a concise Manila-time calendar briefing."""
    now = _aware_datetime("now", now)
    hours_ahead = _positive_finite_real("hours_ahead", hours_ahead)

    if not status["usable"]:
        return "\n".join((
            "CALENDAR UNAVAILABLE: check news manually at forexfactory.com before trading. "
            "Do not assume there is no news.",
            "Problems: " + ", ".join(status["problems"]),
        ))

    manila = ZoneInfo("Asia/Manila")

    def M(value: datetime) -> str:
        return value.astimezone(manila).strftime("%Y-%m-%d %H:%M")

    currency_text = " and ".join(currencies)
    lines = [
        f"News briefing: {currency_text}, next {hours_ahead:g} hours, "
        f"as of {M(now)} Manila time"
    ]

    windows = build_blackout_windows(
        events,
        currencies=currencies,
        impacts=blackout_impacts,
        minutes_before=minutes_before,
        minutes_after=minutes_after,
    )
    current = blackout_state(windows, now)
    if current["in_blackout"]:
        window = current["window"]
        lines.append(
            f"BLACKOUT NOW: until {M(window['end'])} Manila time "
            f"({', '.join(window['titles'])})"
        )

    cutoff = now + timedelta(hours=hours_ahead)
    touching = [
        window for window in windows
        if window["end"] >= now and window["start"] <= cutoff
    ]
    if not touching:
        lines.append(
            f"No high-impact windows in the next {hours_ahead:g} hours."
        )
    else:
        lines.append(
            f"Avoid new trades (windows of {minutes_before} min before "
            f"and {minutes_after} min after high-impact events):"
        )
        for window in touching:
            lines.append(
                f"{M(window['start'])} to {M(window['end'])} Manila: "
                f"{', '.join(window['titles'])}"
            )

    upcoming = upcoming_events(
        events,
        now,
        hours_ahead=hours_ahead,
        currencies=currencies,
        impacts=impacts,
    )
    if not upcoming:
        lines.append(
            f"No events ({', '.join(impacts)}) in the next {hours_ahead:g} hours."
        )
    else:
        lines.append("Events:")
        for event in upcoming:
            lines.append(
                f"{M(event['time_utc'])} | {event['currency']} | "
                f"{event['impact'].upper()} | {event['title']} | "
                f"forecast {event['forecast'] or '-'} | "
                f"previous {event['previous'] or '-'}"
            )

    lines.append(
        "Source: Forex Factory weekly feed (unofficial). Times are Manila time. "
        "Planning aid only, not financial advice."
    )
    return "\n".join(lines)
