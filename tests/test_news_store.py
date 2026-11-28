from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.news.store as store
from src.news.events import parse_events
from src.news.store import (
    CALENDAR_FILENAME,
    FORMAT_VERSION,
    SOURCES,
    load_calendar,
    save_calendar,
)


RAW3 = [
    {"title": "Core PCE Price Index m/m", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "High", "forecast": "0.3%", "previous": "0.3%"},
    {"title": "Bank Holiday", "country": "USD", "date": "2026-11-26T08:00:00-05:00", "impact": "Holiday", "forecast": "", "previous": ""},
    {"title": "German Flash Manufacturing PMI", "country": "EUR", "date": "2026-11-23T03:30:00-05:00", "impact": "Medium", "forecast": "54.1", "previous": "54.2"},
]
FIXED_NOW = datetime(2026, 11, 25, 2, tzinfo=timezone.utc)


def test_s1_save_load_round_trip(tmp_path: Path) -> None:
    path = save_calendar(RAW3, FIXED_NOW, source="feed", directory=tmp_path)
    assert path == tmp_path / CALENDAR_FILENAME

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "format_version": 1,
        "fetched_at": "2026-11-25T02:00:00+00:00",
        "source": "feed",
        "events": RAW3,
    }

    loaded = load_calendar(tmp_path)
    assert loaded["found"] is True
    assert loaded["error"] is None
    assert loaded["source"] == "feed"
    assert loaded["fetched_at"] == FIXED_NOW
    assert loaded["fetched_at"].utcoffset() == timedelta(0)
    assert loaded["events"] == parse_events(RAW3)
    assert [event["title"] for event in loaded["events"]] == [
        "German Flash Manufacturing PMI",
        "Core PCE Price Index m/m",
        "Bank Holiday",
    ]


def test_s2_timestamp_normalization(tmp_path: Path) -> None:
    new_york = datetime(
        2026, 11, 24, 21, tzinfo=ZoneInfo("America/New_York")
    )
    path = save_calendar(RAW3, new_york, source="feed", directory=tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["fetched_at"] == (
        "2026-11-25T02:00:00+00:00"
    )

    microsecond_timestamp = pd.Timestamp(
        "2026-11-25 02:00:00", tz="UTC"
    ).as_unit("us")
    path = save_calendar(
        RAW3, microsecond_timestamp, source="feed", directory=tmp_path
    )
    assert json.loads(path.read_text(encoding="utf-8"))["fetched_at"] == (
        "2026-11-25T02:00:00+00:00"
    )


@pytest.mark.parametrize(
    ("raw_events", "fetched_at", "source", "message"),
    [
        (RAW3, datetime(2026, 11, 25, 2), "feed", "timezone-aware"),
        (RAW3, FIXED_NOW, "other", "source"),
        ([], FIXED_NOW, "feed", "no events"),
        ([{**RAW3[0], "impact": "Extreme"}], FIXED_NOW, "feed", "record 0"),
        ({"events": RAW3}, FIXED_NOW, "feed", "list"),
        ([{**RAW3[0], "date": "2026-11-25T08:30:00"}], FIXED_NOW, "feed", "record 0"),
    ],
)
def test_s3_invalid_saves_write_nothing(
    tmp_path: Path,
    raw_events: object,
    fetched_at: datetime,
    source: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        save_calendar(
            raw_events, fetched_at, source=source, directory=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_s3_invalid_saves_preserve_existing_calendar(tmp_path: Path) -> None:
    path = save_calendar(RAW3, FIXED_NOW, source="feed", directory=tmp_path)
    original_bytes = path.read_bytes()
    invalid_calls = [
        ([], FIXED_NOW, "feed"),
        ([{**RAW3[0], "impact": "Extreme"}], FIXED_NOW, "feed"),
        ({"events": RAW3}, FIXED_NOW, "feed"),
        ([{**RAW3[0], "date": "2026-11-25T08:30:00"}], FIXED_NOW, "feed"),
        (RAW3, datetime(2026, 11, 25, 2), "feed"),
        (RAW3, FIXED_NOW, "other"),
    ]
    for raw_events, fetched_at, source in invalid_calls:
        with pytest.raises(ValueError):
            save_calendar(
                raw_events,
                fetched_at,
                source=source,
                directory=tmp_path,
            )
        assert path.read_bytes() == original_bytes
        assert [item.name for item in tmp_path.iterdir()] == [CALENDAR_FILENAME]


def test_s4_creates_nested_directory(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "news"
    saved = save_calendar(RAW3, FIXED_NOW, source="feed", directory=directory)
    assert saved == directory / CALENDAR_FILENAME
    assert saved.is_file()


@pytest.mark.parametrize("directory_exists", [True, False])
def test_s5_missing_calendar(tmp_path: Path, directory_exists: bool) -> None:
    directory = tmp_path / "missing"
    if directory_exists:
        directory.mkdir()
    assert load_calendar(directory) == {
        "found": False,
        "events": [],
        "fetched_at": None,
        "source": None,
        "error": "calendar file not found",
    }


def _valid_payload() -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "fetched_at": "2026-11-25T02:00:00+00:00",
        "source": "feed",
        "events": RAW3,
    }


def test_s6_invalid_calendar_files(tmp_path: Path) -> None:
    path = tmp_path / CALENDAR_FILENAME
    invalid_files: list[bytes] = [
        b"not json",
        json.dumps([]).encode("utf-8"),
        json.dumps({**_valid_payload(), "format_version": 2}).encode("utf-8"),
        json.dumps({key: value for key, value in _valid_payload().items() if key != "events"}).encode("utf-8"),
        json.dumps({**_valid_payload(), "fetched_at": "2026-11-25T02:00:00"}).encode("utf-8"),
        json.dumps({**_valid_payload(), "fetched_at": 5}).encode("utf-8"),
        json.dumps({**_valid_payload(), "source": "other"}).encode("utf-8"),
        json.dumps({**_valid_payload(), "events": []}).encode("utf-8"),
        json.dumps({**_valid_payload(), "events": [{**RAW3[0], "impact": "Extreme"}]}).encode("utf-8"),
        b"\xff\xfe",
    ]
    for contents in invalid_files:
        path.write_bytes(contents)
        loaded = load_calendar(tmp_path)
        assert loaded["found"] is False
        assert loaded["events"] == []
        assert loaded["fetched_at"] is None
        assert loaded["error"].startswith("calendar file invalid: ")


def test_s7_seeded_round_trips(tmp_path: Path) -> None:
    rng = random.Random(2027)
    offsets = ["-05:00", "-04:00", "+00:00", "+01:00", "+08:00", "Z"]
    impacts = ["Low", "Medium", "High", "Holiday", "hIgH", "mEdIuM"]
    titles = [f"Event {index}" for index in range(8)]
    currencies = ["EUR", "USD", "GBP", "JPY"]

    for case in range(100):
        raw_events = []
        for _ in range(rng.randint(1, 12)):
            day = rng.randint(1, 28)
            month = rng.randint(1, 12)
            hour = rng.randint(0, 23)
            minute = rng.randint(0, 59)
            offset = rng.choice(offsets)
            date_text = (
                f"2026-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"
                f"{offset}"
            )
            raw_events.append({
                "title": rng.choice(titles),
                "country": rng.choice(currencies),
                "date": date_text,
                "impact": rng.choice(impacts),
                "forecast": "text",
                "previous": "",
            })

        directory = tmp_path / f"case-{case}"
        save_calendar(
            raw_events, FIXED_NOW, source="feed", directory=directory
        )
        loaded = load_calendar(directory)
        assert loaded["found"] is True
        assert loaded["events"] == parse_events(raw_events)
        assert loaded["fetched_at"] == FIXED_NOW


def test_s8_stored_json_has_exact_keys(tmp_path: Path) -> None:
    path = save_calendar(RAW3, FIXED_NOW, source="feed", directory=tmp_path)
    with path.open(encoding="utf-8") as calendar_file:
        payload = json.load(calendar_file)
    assert set(payload) == {"format_version", "fetched_at", "source", "events"}


def test_s9_store_has_no_network_imports() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "news" / "store.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("urllib", "socket", "http", "requests"):
        assert forbidden not in source


def test_atomic_replace_failure_preserves_file_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = save_calendar(RAW3, FIXED_NOW, source="feed", directory=tmp_path)
    original_bytes = path.read_bytes()

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_calendar(
            RAW3,
            FIXED_NOW + timedelta(hours=1),
            source="manual_file",
            directory=tmp_path,
        )

    assert path.read_bytes() == original_bytes
    assert [item.name for item in tmp_path.iterdir()] == [CALENDAR_FILENAME]
