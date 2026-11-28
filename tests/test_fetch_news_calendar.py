from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from scripts.fetch_news_calendar import (
    FEED_URL,
    MAX_RESPONSE_BYTES,
    TIMEOUT_SECONDS,
    USER_AGENT,
    FeedError,
    fetch_feed,
    parse_feed_bytes,
    run,
)
from src.news.store import (
    CALENDAR_FILENAME,
    load_calendar,
    save_calendar,
)


RAW3 = [
    {"title": "Core PCE Price Index m/m", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "High", "forecast": "0.3%", "previous": "0.3%"},
    {"title": "Bank Holiday", "country": "USD", "date": "2026-11-26T08:00:00-05:00", "impact": "Holiday", "forecast": "", "previous": ""},
    {"title": "German Flash Manufacturing PMI", "country": "EUR", "date": "2026-11-23T03:30:00-05:00", "impact": "Medium", "forecast": "54.1", "previous": "54.2"},
]
FIXED_NOW = datetime(2026, 11, 25, 2, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def disable_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_socket(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(socket, "socket", blocked_socket)


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.read_requests: list[int | None] = []

    def read(self, n: int | None = None) -> bytes:
        self.read_requests.append(n)
        if n is None:
            return self.data
        return self.data[:n]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def fake_opener_for(
    response_or_error: bytes | BaseException,
) -> tuple[Any, list[tuple[Any, int]]]:
    calls: list[tuple[Any, int]] = []

    def opener(request: Any, timeout: int) -> FakeResponse:
        calls.append((request, timeout))
        if isinstance(response_or_error, BaseException):
            raise response_or_error
        return FakeResponse(response_or_error)

    return opener, calls


def encoded_raw3() -> bytes:
    return json.dumps(RAW3).encode("utf-8")


def test_t1_feed_success_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    opener, calls = fake_opener_for(encoded_raw3())
    result = run([], opener=opener, now=FIXED_NOW, directory=tmp_path)

    assert result == 0
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == FEED_URL
    assert request.get_method() == "GET"
    assert timeout == TIMEOUT_SECONDS == 20
    user_agent = request.get_header("User-agent")
    assert user_agent == USER_AGENT
    assert "forex-assistant" in user_agent
    assert "Mozilla" not in user_agent
    assert request.get_header("Accept") == "application/json"
    assert capsys.readouterr().out == (
        f"Saved 3 events to {tmp_path / CALENDAR_FILENAME}\n"
        "First event: 2026-11-23 16:30 Manila time; last event: "
        "2026-11-26 21:00 Manila time\n"
    )
    saved = load_calendar(tmp_path)
    assert saved["found"] is True
    assert saved["source"] == "feed"
    assert len(saved["events"]) == 3


def test_t2_fetch_interval_and_invalid_existing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_calendar(RAW3, FIXED_NOW, source="feed", directory=tmp_path)
    path = tmp_path / CALENDAR_FILENAME
    original_bytes = path.read_bytes()

    early_opener, early_calls = fake_opener_for(encoded_raw3())
    early_now = FIXED_NOW + timedelta(hours=5, minutes=59, seconds=59)
    assert run([], opener=early_opener, now=early_now, directory=tmp_path) == 1
    assert early_calls == []
    assert path.read_bytes() == original_bytes
    assert "less than 6 hours ago" in capsys.readouterr().out

    exact_opener, exact_calls = fake_opener_for(encoded_raw3())
    exact_now = FIXED_NOW + timedelta(hours=6)
    assert run([], opener=exact_opener, now=exact_now, directory=tmp_path) == 0
    assert len(exact_calls) == 1

    invalid_directory = tmp_path / "invalid-existing"
    invalid_directory.mkdir()
    (invalid_directory / CALENDAR_FILENAME).write_text(
        "not json", encoding="utf-8"
    )
    invalid_opener, invalid_calls = fake_opener_for(encoded_raw3())
    assert run(
        [],
        opener=invalid_opener,
        now=early_now,
        directory=invalid_directory,
    ) == 0
    assert len(invalid_calls) == 1

    new_directory = tmp_path / "no-existing-file"
    new_opener, new_calls = fake_opener_for(encoded_raw3())
    assert run([], opener=new_opener, now=FIXED_NOW, directory=new_directory) == 0
    assert len(new_calls) == 1


@pytest.mark.parametrize(("code", "refused"), [(403, True), (429, True), (500, False)])
def test_t3_http_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    code: int,
    refused: bool,
) -> None:
    error = HTTPError(FEED_URL, code, "test failure", {}, None)
    opener, calls = fake_opener_for(error)
    assert run([], opener=opener, now=FIXED_NOW, directory=tmp_path) == 1
    assert len(calls) == 1
    output = capsys.readouterr().out
    assert output.startswith("Calendar NOT updated: ")
    assert str(code) in output
    if refused:
        assert "refused" in output
        assert "--from-file" in output
        assert "manual" in output
    else:
        assert "HTTP" in output
    assert list(tmp_path.iterdir()) == []


def _failure_cases() -> list[tuple[str, bytes | BaseException, str]]:
    return [
        (
            "http 403",
            HTTPError(FEED_URL, 403, "Forbidden", {}, None),
            "refused",
        ),
        (
            "http 429",
            HTTPError(FEED_URL, 429, "Too Many Requests", {}, None),
            "refused",
        ),
        ("http 500", HTTPError(FEED_URL, 500, "Server Error", {}, None), "HTTP"),
        ("url error", URLError("dns"), "could not reach"),
        ("timeout", TimeoutError("timed out"), "could not reach"),
        ("connection reset", ConnectionResetError("reset"), "could not reach"),
        ("html", b"<html>blocked</html>", "not valid JSON"),
        ("json object", b'{"a":1}', "list"),
        ("empty list", b"[]", "no events"),
        (
            "unknown impact",
            json.dumps([{**RAW3[0], "impact": "Extreme"}]).encode("utf-8"),
            "record 0",
        ),
        ("invalid utf-8", b"\xff\xfe", "not valid JSON"),
    ]


def test_t4_error_messages_are_reported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name, response_or_error, message in _failure_cases():
        directory = tmp_path / name.replace(" ", "-")
        opener, calls = fake_opener_for(response_or_error)
        assert run([], opener=opener, now=FIXED_NOW, directory=directory) == 1
        assert len(calls) == 1
        output = capsys.readouterr().out
        assert output.startswith("Calendar NOT updated: ")
        assert message in output
        assert not directory.exists()


@pytest.mark.parametrize(
    ("name", "response_or_error", "message"),
    _failure_cases(),
    ids=[case[0] for case in _failure_cases()],
)
def test_t5_failures_preserve_last_good_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    response_or_error: bytes | BaseException,
    message: str,
) -> None:
    directory = tmp_path / name.replace(" ", "-")
    path = save_calendar(
        RAW3,
        FIXED_NOW - timedelta(hours=30),
        source="feed",
        directory=directory,
    )
    original_bytes = path.read_bytes()
    opener, calls = fake_opener_for(response_or_error)

    assert run([], opener=opener, now=FIXED_NOW, directory=directory) == 1
    assert len(calls) == 1
    output = capsys.readouterr().out
    assert output.startswith("Calendar NOT updated: ")
    assert message in output
    assert path.read_bytes() == original_bytes
    assert [item.name for item in directory.iterdir()] == [CALENDAR_FILENAME]


def test_t6_response_size_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    too_large_opener, too_large_calls = fake_opener_for(
        b"x" * (MAX_RESPONSE_BYTES + 1)
    )
    assert run(
        [], opener=too_large_opener, now=FIXED_NOW, directory=tmp_path
    ) == 1
    assert len(too_large_calls) == 1
    assert "too large" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []

    valid = encoded_raw3()
    padded = valid + b" " * (MAX_RESPONSE_BYTES - len(valid))
    exact_opener, exact_calls = fake_opener_for(padded)
    assert run([], opener=exact_opener, now=FIXED_NOW, directory=tmp_path) == 0
    assert len(exact_calls) == 1
    assert load_calendar(tmp_path)["found"] is True


def test_t7_manual_file_bypasses_interval_and_opener(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "news"
    save_calendar(RAW3, FIXED_NOW, source="feed", directory=directory)
    manual = tmp_path / "manual.json"
    manual.write_text(json.dumps(RAW3), encoding="utf-8")

    def should_not_open(request: Any, timeout: int) -> FakeResponse:
        raise AssertionError("manual mode must not call the opener")

    assert run(
        ["--from-file", str(manual)],
        opener=should_not_open,
        now=FIXED_NOW,
        directory=directory,
    ) == 0
    assert load_calendar(directory)["source"] == "manual_file"
    assert load_calendar(directory)["fetched_at"] == FIXED_NOW
    assert "Saved 3 events" in capsys.readouterr().out


def test_t7_manual_file_errors_preserve_existing_calendar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "news"
    path = save_calendar(RAW3, FIXED_NOW, source="feed", directory=directory)
    original_bytes = path.read_bytes()

    missing = tmp_path / "missing.json"
    assert run(
        ["--from-file", str(missing)],
        now=FIXED_NOW,
        directory=directory,
    ) == 1
    assert "Calendar NOT updated: " in capsys.readouterr().out
    assert path.read_bytes() == original_bytes

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    assert run(
        ["--from-file", str(invalid)],
        now=FIXED_NOW,
        directory=directory,
    ) == 1
    assert capsys.readouterr().out.startswith("Calendar NOT updated: ")
    assert path.read_bytes() == original_bytes


def test_t8_usage_errors_do_not_call_opener(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opener, calls = fake_opener_for(encoded_raw3())
    assert run(["--bogus"], opener=opener, now=FIXED_NOW, directory=tmp_path) == 2
    assert "Usage:" in capsys.readouterr().out
    assert run(
        ["--from-file"],
        opener=opener,
        now=FIXED_NOW,
        directory=tmp_path,
    ) == 2
    assert "Usage:" in capsys.readouterr().out
    assert calls == []


def test_t9_socket_use_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="network disabled in tests"):
        socket.socket()


def test_t10_fetch_source_uses_safe_user_agent_and_no_retry_loop() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "fetch_news_calendar.py"
    ).read_text(encoding="utf-8")
    assert "USER_AGENT" in source
    assert "Mozilla" not in source
    assert "while " not in source
    assert "retry" not in source.lower()
    assert "sleep" not in source.lower()
    assert "time.sleep" not in source


def test_parse_feed_bytes_top_level_must_be_list() -> None:
    with pytest.raises(FeedError, match="list"):
        parse_feed_bytes(b'{"a":1}')


def test_fetch_feed_reads_only_the_requested_limit() -> None:
    response = FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1))

    def opener(request: Any, timeout: int) -> FakeResponse:
        return response

    with pytest.raises(FeedError, match="too large"):
        fetch_feed(opener)
    assert response.read_requests == [MAX_RESPONSE_BYTES + 1]
