from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta, timezone
from sqlite3 import Connection
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.journal.store import (
    add_trade,
    close_trade,
    get_trade,
    list_trades,
    open_journal,
    trading_day_summary,
)
from src.journal.workflow import (
    account_balances,
    open_planned_trade,
    plan_trade,
)
from src.risk.limits import evaluate_daily_limits
from src.risk.sizing import position_size


UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _add_trade(
    conn: Connection,
    *,
    opened_at: datetime,
    reason: str = "fixture",
    planned_risk_amount: float = 20.0,
) -> int:
    return add_trade(
        conn,
        opened_at=opened_at,
        side="long",
        entry_price=1.10000,
        stop_price=1.09900,
        size_units=1000,
        planned_risk_amount=planned_risk_amount,
        mode="paper",
        reason=reason,
    )


def _close_trade(
    conn: Connection,
    trade_id: int,
    *,
    closed_at: datetime,
    realized_pnl: float,
) -> None:
    close_trade(
        conn,
        trade_id,
        closed_at=closed_at,
        exit_price=1.10000,
        realized_pnl=realized_pnl,
    )


def _count(conn: Connection) -> int:
    return len(list_trades(conn))


def _balance_journal() -> Connection:
    conn = open_journal(":memory:")
    t1 = _add_trade(conn, opened_at=_utc("2026-01-15T14:00:00"))
    _close_trade(
        conn,
        t1,
        closed_at=_utc("2026-01-15T21:59:00"),
        realized_pnl=-20.0,
    )
    t2 = _add_trade(conn, opened_at=_utc("2026-01-15T22:00:00"))
    _close_trade(
        conn,
        t2,
        closed_at=_utc("2026-01-16T10:00:00"),
        realized_pnl=30.0,
    )
    t4 = _add_trade(conn, opened_at=_utc("2026-01-14T15:00:00"))
    _close_trade(
        conn,
        t4,
        closed_at=_utc("2026-01-15T22:00:00"),
        realized_pnl=-5.0,
    )
    return conn


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-01-15T21:00:00", (1000.0, 1000.0)),
        ("2026-01-15T21:59:59", (980.0, 1000.0)),
        ("2026-01-15T22:30:00", (975.0, 980.0)),
        ("2026-01-16T13:00:00", (1005.0, 980.0)),
        ("2026-01-14T12:00:00", (1000.0, 1000.0)),
    ],
    ids=["A1", "A2", "A3", "A4", "A5"],
)
def test_account_balances_hand_fixtures(
    now: str,
    expected: tuple[float, float],
) -> None:
    conn = _balance_journal()

    balances = account_balances(conn, 1000, _utc(now))

    assert balances["current_balance"] == pytest.approx(expected[0], abs=1e-9)
    assert balances["start_of_day_balance"] == pytest.approx(
        expected[1],
        abs=1e-9,
    )
    conn.close()


def test_account_balances_empty_journal() -> None:
    conn = open_journal(":memory:")

    assert account_balances(
        conn,
        1000,
        _utc("2026-01-14T12:00:00"),
    ) == {
        "current_balance": 1000.0,
        "start_of_day_balance": 1000.0,
    }
    conn.close()


@pytest.mark.parametrize(
    "starting_balance",
    [0, -1, float("nan"), True, "1000"],
    ids=["zero", "negative", "nan", "bool", "string"],
)
def test_account_balances_rejects_invalid_starting_balance(
    starting_balance: object,
) -> None:
    conn = open_journal(":memory:")
    with pytest.raises(ValueError):
        account_balances(conn, starting_balance, _utc("2026-01-14T12:00:00"))
    conn.close()


def test_account_balances_rejects_naive_now() -> None:
    conn = open_journal(":memory:")
    with pytest.raises(ValueError, match="timezone-aware"):
        account_balances(conn, 1000, datetime(2026, 1, 14, 12, 0))
    conn.close()


def _plan(
    conn: Connection,
    *,
    now: datetime,
    starting_balance: float = 1000,
    risk_pct: float = 1.0,
    side: str = "long",
    entry_price: float = 1.10000,
    stop_price: float = 1.09900,
) -> dict[str, object]:
    return plan_trade(
        conn,
        now=now,
        starting_balance=starting_balance,
        risk_pct=risk_pct,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
    )


def _open_planned(
    conn: Connection,
    *,
    opened_at: datetime,
    starting_balance: float = 1000,
    risk_pct: float = 1.0,
    side: str = "long",
    entry_price: float = 1.10000,
    stop_price: float = 1.09900,
    mode: str = "paper",
    reason: str = "first test",
    strategy: str = "manual",
) -> dict[str, object]:
    return open_planned_trade(
        conn,
        opened_at=opened_at,
        starting_balance=starting_balance,
        risk_pct=risk_pct,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        mode=mode,
        reason=reason,
        strategy=strategy,
    )


def test_b1_plan_hand_example_and_microsecond_timestamp() -> None:
    conn = open_journal(":memory:")
    now = pd.Timestamp(
        "2026-01-14 12:00:00.123456",
        tz="UTC",
    ).as_unit("us")
    assert now.unit == "us"

    plan = _plan(conn, now=now)

    assert plan["balances"] == {
        "current_balance": 1000.0,
        "start_of_day_balance": 1000.0,
    }
    assert (
        plan["summary"]["trades_taken_today"],
        plan["summary"]["realized_pnl_today"],
        plan["summary"]["open_positions"],
    ) == (0, 0.0, 0)
    assert plan["limits"] == {"allowed": True, "reasons": []}
    assert plan["sizing"]["status"] == "ok"
    assert plan["sizing"]["units"] == 8000
    assert plan["sizing"]["lots"] == pytest.approx(0.08, abs=1e-9)
    assert plan["sizing"]["effective_risk_pips"] == pytest.approx(
        11.4,
        abs=1e-9,
    )
    assert plan["sizing"]["actual_risk_amount"] == pytest.approx(
        9.12,
        abs=1e-9,
    )
    assert plan["sizing"]["actual_risk_pct"] == pytest.approx(
        0.912,
        abs=1e-9,
    )
    assert plan["blocking_reasons"] == []
    assert plan["can_save"] is True
    conn.close()


def test_b2_checked_open_saves_cost_aware_risk_and_microsecond_timestamp() -> None:
    conn = open_journal(":memory:")
    opened_at = pd.Timestamp(
        "2026-01-14 12:00:00.654321",
        tz="UTC",
    ).as_unit("us")
    assert opened_at.unit == "us"

    result = _open_planned(conn, opened_at=opened_at)

    assert result["trade_id"] == 1
    trade = get_trade(conn, result["trade_id"])
    assert trade is not None
    assert trade["size_units"] == 8000
    assert trade["planned_risk_amount"] == pytest.approx(9.12, abs=1e-9)
    assert trade["planned_risk_amount"] != 10.0
    assert trade["entry_price"] == 1.1
    assert trade["stop_price"] == 1.099
    assert trade["mode"] == "paper"
    assert trade["side"] == "long"
    assert trade["strategy"] == "manual"
    assert trade["opened_at"] == _utc("2026-01-14T12:00:00")
    conn.close()


def test_b3_open_position_blocks_plan_and_checked_save() -> None:
    conn = open_journal(":memory:")
    _open_planned(conn, opened_at=_utc("2026-01-14T12:00:00"))
    now = _utc("2026-01-14T12:30:00")

    plan = _plan(conn, now=now)

    assert (
        plan["summary"]["trades_taken_today"],
        plan["summary"]["realized_pnl_today"],
        plan["summary"]["open_positions"],
    ) == (1, 0.0, 1)
    assert plan["limits"]["reasons"] == ["position_already_open"]
    assert plan["blocking_reasons"] == ["position_already_open"]
    assert plan["can_save"] is False
    assert plan["sizing"]["status"] == "ok"
    assert plan["sizing"]["units"] == 8000
    with pytest.raises(ValueError, match="position_already_open"):
        _open_planned(conn, opened_at=now)
    assert _count(conn) == 1
    conn.close()


def test_b4_daily_loss_blocks_checked_save() -> None:
    conn = open_journal(":memory:")
    trade_id = _open_planned(
        conn,
        opened_at=_utc("2026-01-14T12:00:00"),
    )["trade_id"]
    close_trade(
        conn,
        trade_id,
        closed_at=_utc("2026-01-14T13:00:00"),
        exit_price=1.09900,
        realized_pnl=-20.0,
    )
    now = _utc("2026-01-14T13:30:00")

    plan = _plan(conn, now=now)

    assert plan["balances"] == {
        "current_balance": 980.0,
        "start_of_day_balance": 1000.0,
    }
    assert (
        plan["summary"]["trades_taken_today"],
        plan["summary"]["realized_pnl_today"],
        plan["summary"]["open_positions"],
    ) == (1, -20.0, 0)
    assert plan["limits"]["reasons"] == ["daily_loss_limit_reached"]
    assert plan["blocking_reasons"] == ["daily_loss_limit_reached"]
    assert plan["can_save"] is False
    with pytest.raises(ValueError, match="daily_loss_limit_reached"):
        _open_planned(conn, opened_at=now)
    assert _count(conn) == 1
    conn.close()


def test_b5_uses_start_balance_for_limit_and_current_balance_for_size() -> None:
    conn = open_journal(":memory:")
    trade_id = _open_planned(
        conn,
        opened_at=_utc("2026-01-14T12:00:00"),
    )["trade_id"]
    close_trade(
        conn,
        trade_id,
        closed_at=_utc("2026-01-14T13:00:00"),
        exit_price=1.09900,
        realized_pnl=-19.9,
    )

    plan = _plan(conn, now=_utc("2026-01-14T13:30:00"))

    assert plan["balances"] == {
        "current_balance": pytest.approx(980.1, abs=1e-9),
        "start_of_day_balance": 1000.0,
    }
    assert plan["limits"]["allowed"] is True
    assert plan["blocking_reasons"] == []
    assert plan["can_save"] is True
    assert plan["sizing"]["risk_amount"] == pytest.approx(9.801, abs=1e-9)
    assert plan["sizing"]["units"] == 8000
    conn.close()


def test_b6_two_trades_reach_daily_trade_limit() -> None:
    conn = open_journal(":memory:")
    trade_a = _add_trade(conn, opened_at=_utc("2026-01-14T12:00:00"))
    _close_trade(
        conn,
        trade_a,
        closed_at=_utc("2026-01-14T12:30:00"),
        realized_pnl=-5.0,
    )
    trade_b = _add_trade(conn, opened_at=_utc("2026-01-14T13:00:00"))
    _close_trade(
        conn,
        trade_b,
        closed_at=_utc("2026-01-14T13:30:00"),
        realized_pnl=5.0,
    )

    plan = _plan(conn, now=_utc("2026-01-14T14:00:00"))

    assert (
        plan["summary"]["trades_taken_today"],
        plan["summary"]["realized_pnl_today"],
        plan["summary"]["open_positions"],
    ) == (2, 0.0, 0)
    assert plan["limits"]["reasons"] == ["max_trades_reached"]
    assert plan["can_save"] is False
    conn.close()


def test_b7_too_small_size_and_combined_blocking_order() -> None:
    now = _utc("2026-01-14T12:00:00")
    conn = open_journal(":memory:")
    plan = _plan(
        conn,
        now=now,
        starting_balance=100,
        stop_price=1.09500,
    )
    assert plan["sizing"]["status"] == "too_small"
    assert plan["sizing"]["effective_risk_pips"] == pytest.approx(51.4)
    assert plan["limits"]["allowed"] is True
    assert plan["blocking_reasons"] == ["size_too_small"]
    assert plan["can_save"] is False
    with pytest.raises(ValueError, match="size_too_small"):
        _open_planned(
            conn,
            opened_at=now,
            starting_balance=100,
            stop_price=1.09500,
        )
    assert _count(conn) == 0
    conn.close()

    conn = open_journal(":memory:")
    _open_planned(conn, opened_at=_utc("2026-01-14T11:00:00"))
    combined = _plan(
        conn,
        now=now,
        starting_balance=100,
        stop_price=1.09500,
    )
    assert combined["blocking_reasons"] == [
        "position_already_open",
        "size_too_small",
    ]
    assert combined["can_save"] is False
    conn.close()


def test_b8_short_plan_and_save() -> None:
    conn = open_journal(":memory:")
    now = _utc("2026-01-14T12:00:00")

    plan = _plan(
        conn,
        now=now,
        side="short",
        stop_price=1.10100,
    )
    saved = _open_planned(
        conn,
        opened_at=now,
        side="short",
        stop_price=1.10100,
    )

    assert plan["sizing"]["units"] == 8000
    assert plan["sizing"]["actual_risk_amount"] == pytest.approx(9.12)
    trade = get_trade(conn, saved["trade_id"])
    assert trade is not None
    assert trade["side"] == "short"
    assert trade["size_units"] == 8000
    conn.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"side": "buy"}, "side"),
        ({"side": "long", "stop_price": 1.101}, "long stop"),
        ({"side": "short", "stop_price": 1.099}, "short stop"),
        ({"risk_pct": 2.01}, "max_risk_pct"),
        ({"risk_pct": 0}, "risk_pct"),
    ],
    ids=["side", "long-stop", "short-stop", "risk-cap", "zero-risk"],
)
def test_b9_plan_validation(
    kwargs: dict[str, object],
    message: str,
) -> None:
    conn = open_journal(":memory:")
    args: dict[str, object] = {
        "now": _utc("2026-01-14T12:00:00"),
        "starting_balance": 1000,
        "risk_pct": 1.0,
        "side": "long",
        "entry_price": 1.10000,
        "stop_price": 1.09900,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        plan_trade(conn, **args)
    assert _count(conn) == 0
    conn.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"opened_at": datetime(2026, 1, 14, 12, 0)}, "timezone-aware"),
        ({"mode": "live"}, "mode"),
        ({"reason": ""}, "reason"),
        ({"strategy": ""}, "strategy"),
        ({"starting_balance": 0}, "starting_balance"),
    ],
    ids=["naive-open", "live", "empty-reason", "empty-strategy", "balance"],
)
def test_b9_open_validation_leaves_journal_unchanged(
    kwargs: dict[str, object],
    message: str,
) -> None:
    conn = open_journal(":memory:")
    args: dict[str, object] = {
        "opened_at": _utc("2026-01-14T12:00:00"),
        "starting_balance": 1000,
        "risk_pct": 1.0,
        "side": "long",
        "entry_price": 1.10000,
        "stop_price": 1.09900,
        "mode": "paper",
        "reason": "first test",
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        open_planned_trade(conn, **args)
    assert _count(conn) == 0
    conn.close()


def _seeded_records() -> tuple[list[dict[str, object]], list[datetime]]:
    rng = random.Random(909)
    start = _utc("2026-03-04T00:00:00")
    end = _utc("2026-03-12T00:00:00")
    span_seconds = int((end - start).total_seconds())
    records: list[dict[str, object]] = []
    for _ in range(40):
        opened_at = start + timedelta(seconds=rng.randrange(span_seconds))
        closed_at = None
        pnl = None
        if rng.random() < 0.7:
            closed_at = opened_at + timedelta(
                seconds=rng.randint(1, 72 * 60 * 60)
            )
            pnl = rng.uniform(-10, 15)
        records.append(
            {
                "opened_at": opened_at,
                "closed_at": closed_at,
                "realized_pnl": pnl,
            }
        )

    random_nows = [
        start + timedelta(seconds=rng.randrange(span_seconds))
        for _ in range(250)
    ]
    boundary_nows: list[datetime] = []
    current_date = date(2026, 3, 4)
    while current_date <= date(2026, 3, 11):
        boundary = datetime.combine(
            current_date,
            time(17, 0),
            tzinfo=NEW_YORK,
        ).astimezone(UTC)
        if start <= boundary < end:
            boundary_nows.extend(
                (boundary, boundary - timedelta(seconds=1))
            )
        current_date += timedelta(days=1)
    return records, random_nows + boundary_nows


def _seeded_journal(records: list[dict[str, object]]) -> Connection:
    conn = open_journal(":memory:")
    for record in records:
        trade_id = _add_trade(
            conn,
            opened_at=record["opened_at"],
            planned_risk_amount=10.0,
        )
        if record["closed_at"] is not None:
            close_trade(
                conn,
                trade_id,
                closed_at=record["closed_at"],
                exit_price=1.10000,
                realized_pnl=record["realized_pnl"],
            )
    return conn


def _reference_label(value: datetime) -> date:
    return (value.astimezone(NEW_YORK) + timedelta(hours=7)).date()


def _reference_balances(
    records: list[dict[str, object]],
    starting_balance: float,
    now: datetime,
) -> tuple[float, float]:
    now_day = _reference_label(now)
    current_balance = starting_balance
    start_of_day_balance = starting_balance
    for record in records:
        closed_at = record["closed_at"]
        if closed_at is not None and closed_at <= now:
            pnl = float(record["realized_pnl"])
            current_balance += pnl
            if _reference_label(closed_at) < now_day:
                start_of_day_balance += pnl
    return current_balance, start_of_day_balance


def test_independent_account_balance_reference_across_dst() -> None:
    records, now_values = _seeded_records()
    conn = _seeded_journal(records)
    offsets = {
        record["opened_at"].astimezone(NEW_YORK).utcoffset()
        for record in records
    }
    assert timedelta(hours=-5) in offsets
    assert timedelta(hours=-4) in offsets

    for now in now_values:
        expected_current, expected_start = _reference_balances(
            records,
            1000.0,
            now,
        )
        actual = account_balances(conn, 1000, now)
        assert actual["current_balance"] == pytest.approx(
            expected_current,
            abs=1e-9,
        )
        assert actual["start_of_day_balance"] == pytest.approx(
            expected_start,
            abs=1e-9,
        )
    conn.close()


def test_plan_trade_ignores_events_added_after_now() -> None:
    records, now_values = _seeded_records()
    for now in now_values[:20]:
        conn = _seeded_journal(records)
        before = _plan(conn, now=now)
        for position, pnl in enumerate((1_000_000_000.0, -1_000_000_000.0)):
            opened_at = now + timedelta(seconds=1 + position * 2)
            trade_id = _add_trade(
                conn,
                opened_at=opened_at,
                reason="future event",
            )
            _close_trade(
                conn,
                trade_id,
                closed_at=opened_at + timedelta(seconds=1),
                realized_pnl=pnl,
            )
        after = _plan(conn, now=now)
        assert after == before
        conn.close()


def test_plan_trade_components_match_the_public_helpers() -> None:
    records, now_values = _seeded_records()
    conn = _seeded_journal(records)
    for now in now_values[:30]:
        plan = _plan(conn, now=now)
        balances = account_balances(conn, 1000, now)
        summary = trading_day_summary(conn, now)
        limits = evaluate_daily_limits(
            balances["start_of_day_balance"],
            summary["realized_pnl_today"],
            summary["trades_taken_today"],
            summary["open_positions"],
        )
        sizing = position_size(
            balances["current_balance"],
            1.0,
            1.10000,
            1.09900,
        )
        assert plan["balances"] == balances
        assert plan["summary"] == summary
        assert plan["limits"] == limits
        assert plan["sizing"] == sizing
    conn.close()
