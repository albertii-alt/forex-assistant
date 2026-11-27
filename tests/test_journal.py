from __future__ import annotations

import math
import random
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.risk.limits import evaluate_daily_limits
from src.journal.store import (
    ALLOWED_MODES,
    DEFAULT_JOURNAL_PATH,
    add_trade,
    close_trade,
    get_trade,
    list_trades,
    open_journal,
    trading_day_summary,
)


UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _add_trade(conn: sqlite3.Connection, **overrides: object) -> int:
    values: dict[str, object] = {
        "opened_at": _utc("2026-01-14T15:00:00"),
        "side": "long",
        "entry_price": 1.10000,
        "stop_price": 1.09900,
        "size_units": 1000,
        "planned_risk_amount": 20.0,
        "mode": "paper",
        "reason": "test",
    }
    values.update(overrides)
    return add_trade(conn, **values)


def _close_trade(
    conn: sqlite3.Connection,
    trade_id: int,
    **overrides: object,
) -> None:
    values: dict[str, object] = {
        "closed_at": _utc("2026-01-14T16:00:00"),
        "exit_price": 1.10000,
        "realized_pnl": 0.0,
    }
    values.update(overrides)
    close_trade(conn, trade_id, **values)


def _row_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])


def test_add_close_get_and_list_basics() -> None:
    conn = open_journal(":memory:")
    opened = _utc("2026-01-14T15:00:00")
    trade_id = add_trade(
        conn,
        opened_at=opened,
        side="long",
        entry_price=1.10000,
        stop_price=1.09900,
        size_units=8000,
        planned_risk_amount=8.96,
        mode="paper",
        reason="  test setup  ",
    )

    trade = get_trade(conn, trade_id)
    assert trade is not None
    assert trade["id"] == 1
    assert trade["opened_at"] == opened
    assert trade["side"] == "long"
    assert trade["mode"] == "paper"
    assert trade["strategy"] == "manual"
    assert trade["pair"] == "EURUSD"
    assert trade["reason"] == "test setup"
    assert trade["closed_at"] is None
    assert trade["exit_price"] is None
    assert trade["realized_pnl"] is None
    assert trade["r_multiple"] is None
    assert type(trade["opened_at"]) is datetime

    _close_trade(
        conn,
        trade_id,
        closed_at=_utc("2026-01-14T16:30:00"),
        exit_price=1.10200,
        realized_pnl=17.92,
        notes="ok",
    )
    closed = get_trade(conn, trade_id)
    assert closed is not None
    assert closed["closed_at"] == _utc("2026-01-14T16:30:00")
    assert closed["notes"] == "ok"
    assert closed["r_multiple"] == pytest.approx(2.0, abs=1e-9)
    assert list_trades(conn, status="open") == []
    assert len(list_trades(conn, status="closed")) == 1
    assert get_trade(conn, 999) is None
    conn.close()


def _invalid_add(overrides: dict[str, object]) -> None:
    conn = open_journal(":memory:")
    before = _row_count(conn)
    with pytest.raises(ValueError):
        _add_trade(conn, **overrides)
    assert _row_count(conn) == before
    assert list_trades(conn) == []
    conn.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"opened_at": datetime(2026, 1, 14, 15, 0)},
        {"side": "buy"},
        {"mode": "live"},
        {"mode": "real"},
        {"reason": ""},
        {"reason": "   \t"},
        {"strategy": ""},
        {"entry_price": 0},
        {"stop_price": -1},
        {"entry_price": math.nan},
        {"entry_price": math.inf},
        {"stop_price": True},
        {"entry_price": "1.1"},
        {"side": "long", "stop_price": 1.101},
        {"side": "short", "stop_price": 1.099},
        {"side": "long", "target_price": 1.099},
        {"side": "short", "target_price": 1.101},
        {"size_units": 0},
        {"size_units": -1000},
        {"size_units": 1000.5},
        {"size_units": 1000.0},
        {"size_units": True},
        {"planned_risk_amount": 0},
        {"planned_risk_amount": -1},
        {"planned_risk_amount": math.nan},
        {"planned_risk_amount": True},
    ],
)
def test_add_trade_validation_leaves_journal_unchanged(
    overrides: dict[str, object],
) -> None:
    _invalid_add(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"closed_at": datetime(2026, 1, 14, 16, 0)},
        {"closed_at": _utc("2026-01-14T14:59:59")},
        {"exit_price": 0},
        {"exit_price": math.inf},
        {"realized_pnl": math.nan},
        {"realized_pnl": True},
    ],
)
def test_close_trade_validation_leaves_journal_unchanged(
    overrides: dict[str, object],
) -> None:
    conn = open_journal(":memory:")
    trade_id = _add_trade(conn)
    before = list_trades(conn)
    with pytest.raises(ValueError):
        _close_trade(conn, trade_id, **overrides)
    assert list_trades(conn) == before
    conn.close()


def test_unknown_or_already_closed_trade_is_unchanged() -> None:
    conn = open_journal(":memory:")
    before = list_trades(conn)
    with pytest.raises(ValueError, match="unknown trade"):
        _close_trade(conn, 999)
    assert list_trades(conn) == before

    trade_id = _add_trade(conn)
    _close_trade(conn, trade_id)
    closed_before = list_trades(conn)
    with pytest.raises(ValueError, match="already closed"):
        _close_trade(conn, trade_id, realized_pnl=100.0)
    assert list_trades(conn) == closed_before
    conn.close()


def test_equal_open_and_close_times_are_allowed() -> None:
    conn = open_journal(":memory:")
    opened = _utc("2026-01-14T15:00:00")
    trade_id = _add_trade(conn, opened_at=opened)

    _close_trade(conn, trade_id, closed_at=opened)

    trade = get_trade(conn, trade_id)
    assert trade is not None
    assert trade["closed_at"] == opened
    conn.close()


def test_persistence_idempotent_schema_and_version(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "journal.sqlite3"
    conn = open_journal(path)
    trade_id = _add_trade(conn)
    conn.close()

    reopened = open_journal(path)
    assert get_trade(reopened, trade_id) is not None
    assert reopened.execute("PRAGMA user_version").fetchone()[0] == 1
    opened_again = open_journal(path)
    assert len(list_trades(opened_again)) == 1
    assert opened_again.execute("PRAGMA user_version").fetchone()[0] == 1
    reopened.close()
    opened_again.close()
    assert DEFAULT_JOURNAL_PATH == "data/journal.sqlite3"
    assert ALLOWED_MODES == ("paper", "demo")


def test_quotes_and_sql_looking_text_are_stored_literally() -> None:
    conn = open_journal(":memory:")
    payload = "x'); DROP TABLE trades;--"
    trade_id = _add_trade(conn, reason=f"  {payload}  ")
    _close_trade(conn, trade_id, notes=f"  {payload}  ")

    trade = get_trade(conn, trade_id)
    assert trade is not None
    assert trade["reason"] == payload
    assert trade["notes"] == payload
    assert _add_trade(conn, reason=payload) == 2
    assert _row_count(conn) == 2
    conn.close()


def test_timezone_conversion_and_second_precision_round_trip() -> None:
    conn = open_journal(":memory:")
    opened = pd.Timestamp(
        "2026-01-15 09:00:37.123456",
        tz="America/New_York",
    ).as_unit("us")
    trade_id = _add_trade(conn, opened_at=opened)

    raw = conn.execute(
        "SELECT opened_at FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()[0]
    trade = get_trade(conn, trade_id)
    assert raw == "2026-01-15T14:00:37+00:00"
    assert trade is not None
    assert trade["opened_at"] == _utc("2026-01-15T14:00:37")
    conn.close()


def _winter_journal() -> tuple[sqlite3.Connection, dict[str, int]]:
    conn = open_journal(":memory:")
    ids: dict[str, int] = {}
    ids["T1"] = _add_trade(conn, opened_at=_utc("2026-01-15T14:00:00"))
    _close_trade(
        conn,
        ids["T1"],
        closed_at=_utc("2026-01-15T21:59:00"),
        realized_pnl=-20.0,
    )
    ids["T2"] = _add_trade(conn, opened_at=_utc("2026-01-15T22:00:00"))
    _close_trade(
        conn,
        ids["T2"],
        closed_at=_utc("2026-01-16T10:00:00"),
        realized_pnl=30.0,
    )
    ids["T3"] = _add_trade(conn, opened_at=_utc("2026-01-16T12:00:00"))
    ids["T4"] = _add_trade(conn, opened_at=_utc("2026-01-14T15:00:00"))
    _close_trade(
        conn,
        ids["T4"],
        closed_at=_utc("2026-01-15T22:00:00"),
        realized_pnl=-5.0,
    )
    return conn, ids


@pytest.mark.parametrize(
    ("now", "expected_day", "expected_counts"),
    [
        (
            "2026-01-15T21:00:00",
            "2026-01-15",
            (1, 0.0, 2),
        ),
        (
            "2026-01-15T21:59:59",
            "2026-01-15",
            (1, -20.0, 1),
        ),
        (
            "2026-01-15T22:30:00",
            "2026-01-16",
            (1, -5.0, 1),
        ),
        (
            "2026-01-16T13:00:00",
            "2026-01-16",
            (2, 25.0, 1),
        ),
        (
            "2026-01-14T12:00:00",
            "2026-01-14",
            (0, 0.0, 0),
        ),
        (
            "2026-01-16T09:00:00",
            "2026-01-16",
            (1, -5.0, 1),
        ),
    ],
    ids=["F1", "F2", "F3", "F4", "F5", "J13"],
)
def test_winter_trading_day_hand_fixtures(
    now: str,
    expected_day: str,
    expected_counts: tuple[int, float, int],
) -> None:
    conn, _ = _winter_journal()

    summary = trading_day_summary(conn, _utc(now))

    assert summary["trading_day"] == pd.Timestamp(expected_day)
    actual = (
        summary["trades_taken_today"],
        summary["realized_pnl_today"],
        summary["open_positions"],
    )
    assert actual == expected_counts
    conn.close()


def test_hand_derived_r_multiples() -> None:
    conn, ids = _winter_journal()

    assert get_trade(conn, ids["T1"])["r_multiple"] == pytest.approx(-1.0, abs=1e-9)
    assert get_trade(conn, ids["T2"])["r_multiple"] == pytest.approx(1.5, abs=1e-9)
    assert get_trade(conn, ids["T4"])["r_multiple"] == pytest.approx(-0.25, abs=1e-9)
    conn.close()


@pytest.mark.parametrize(
    ("now", "expected_day", "expected"),
    [
        ("2026-07-15T21:30:00", "2026-07-16", (1, 0.0, 2)),
        ("2026-07-15T21:00:00", "2026-07-16", (1, 0.0, 2)),
        ("2026-07-15T20:59:30", "2026-07-15", (1, 0.0, 1)),
    ],
    ids=["after-boundary", "at-boundary", "before-boundary"],
)
def test_summer_trading_day_boundary(
    now: str,
    expected_day: str,
    expected: tuple[int, float, int],
) -> None:
    conn = open_journal(":memory:")
    u1 = _add_trade(conn, opened_at=_utc("2026-07-15T20:59:00"))
    u2 = _add_trade(conn, opened_at=_utc("2026-07-15T21:00:00"))
    assert u1 == 1
    assert u2 == 2

    summary = trading_day_summary(conn, _utc(now))

    assert summary["trading_day"] == pd.Timestamp(expected_day)
    assert (
        summary["trades_taken_today"],
        summary["realized_pnl_today"],
        summary["open_positions"],
    ) == expected
    conn.close()


def test_summary_rejects_naive_now() -> None:
    conn = open_journal(":memory:")
    with pytest.raises(ValueError, match="timezone-aware"):
        trading_day_summary(conn, datetime(2026, 1, 15, 21, 0))
    conn.close()


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (
            "2026-01-16T13:00:00",
            {
                "allowed": False,
                "reasons": ["max_trades_reached", "position_already_open"],
            },
        ),
        (
            "2026-01-15T21:00:00",
            {"allowed": False, "reasons": ["position_already_open"]},
        ),
        (
            "2026-01-14T12:00:00",
            {"allowed": True, "reasons": []},
        ),
    ],
    ids=["end-of-day", "open-positions", "before-any-trades"],
)
def test_summary_integrates_with_risk_limits(
    now: str,
    expected: dict[str, object],
) -> None:
    conn, _ = _winter_journal()

    summary = trading_day_summary(conn, _utc(now))
    result = evaluate_daily_limits(
        10000,
        summary["realized_pnl_today"],
        summary["trades_taken_today"],
        summary["open_positions"],
    )

    assert result == expected
    conn.close()


def _seeded_journal_data() -> tuple[list[dict[str, object]], list[datetime]]:
    rng = random.Random(2026)
    start = _utc("2026-03-04T00:00:00")
    end = _utc("2026-03-12T00:00:00")
    span_seconds = int((end - start).total_seconds())
    records: list[dict[str, object]] = []
    for _ in range(40):
        opened_at = start + timedelta(seconds=rng.randrange(span_seconds))
        closed_at = None
        realized_pnl = None
        if rng.random() < 0.7:
            closed_at = opened_at + timedelta(
                seconds=rng.randint(1, 72 * 60 * 60)
            )
            realized_pnl = rng.uniform(-10, 15)
        records.append(
            {
                "opened_at": opened_at,
                "closed_at": closed_at,
                "realized_pnl": realized_pnl,
            }
        )

    now_values = [
        start + timedelta(seconds=rng.randrange(span_seconds))
        for _ in range(250)
    ]
    boundary_day = date(2026, 3, 4)
    final_day = date(2026, 3, 11)
    while boundary_day <= final_day:
        boundary = datetime.combine(
            boundary_day,
            time(17, 0),
            tzinfo=NEW_YORK,
        ).astimezone(UTC)
        if start <= boundary < end:
            now_values.extend((boundary, boundary - timedelta(seconds=1)))
        boundary_day += timedelta(days=1)
    return records, now_values


def _insert_seeded_records(
    conn: sqlite3.Connection,
    records: list[dict[str, object]],
) -> None:
    for record in records:
        trade_id = _add_trade(conn, opened_at=record["opened_at"])
        if record["closed_at"] is not None:
            _close_trade(
                conn,
                trade_id,
                closed_at=record["closed_at"],
                realized_pnl=record["realized_pnl"],
            )


def _reference_trading_day(value: datetime) -> date:
    return (value.astimezone(NEW_YORK) + timedelta(hours=7)).date()


def _reference_summary(
    records: list[dict[str, object]],
    now: datetime,
) -> tuple[date, int, float, int]:
    current_day = _reference_trading_day(now)
    taken = 0
    realized = 0.0
    open_count = 0
    for record in records:
        opened_at = record["opened_at"]
        closed_at = record["closed_at"]
        if opened_at <= now and _reference_trading_day(opened_at) == current_day:
            taken += 1
        if (
            closed_at is not None
            and closed_at <= now
            and _reference_trading_day(closed_at) == current_day
        ):
            realized += float(record["realized_pnl"])
        if opened_at <= now and (closed_at is None or closed_at > now):
            open_count += 1
    return current_day, taken, realized, open_count


def test_independent_trading_day_reference_across_dst() -> None:
    records, now_values = _seeded_journal_data()
    conn = open_journal(":memory:")
    _insert_seeded_records(conn, records)

    offsets = {
        record["opened_at"].astimezone(NEW_YORK).utcoffset()
        for record in records
    }
    assert timedelta(hours=-5) in offsets
    assert timedelta(hours=-4) in offsets

    for now in now_values:
        expected_day, expected_taken, expected_pnl, expected_open = (
            _reference_summary(records, now)
        )
        actual = trading_day_summary(conn, now)
        assert actual["trading_day"].date() == expected_day
        assert actual["trades_taken_today"] == expected_taken
        assert actual["realized_pnl_today"] == pytest.approx(
            expected_pnl,
            abs=1e-9,
        )
        assert actual["open_positions"] == expected_open
    conn.close()


def test_future_journal_events_do_not_change_as_of_summary() -> None:
    records, now_values = _seeded_journal_data()
    for now in now_values[:20]:
        conn = open_journal(":memory:")
        _insert_seeded_records(conn, records)
        before = trading_day_summary(conn, now)
        future_id = _add_trade(
            conn,
            opened_at=now + timedelta(seconds=1),
            reason="future event",
        )
        _close_trade(
            conn,
            future_id,
            closed_at=now + timedelta(seconds=2),
            realized_pnl=1_000_000_000.0,
        )
        after = trading_day_summary(conn, now)
        assert after == before
        conn.close()

