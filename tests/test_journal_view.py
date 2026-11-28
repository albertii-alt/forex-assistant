from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.app import journal_view
from src.journal.store import DEFAULT_JOURNAL_PATH


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("max_trades_reached", "Maximum trades for today reached (2)."),
        (
            "daily_loss_limit_reached",
            "Daily loss limit reached (2% of the start-of-day balance).",
        ),
        (
            "position_already_open",
            "A position is already open (maximum 1).",
        ),
        (
            "size_too_small",
            "The account is too small to take this trade within the risk limit.",
        ),
    ],
)
def test_describe_reason_exact_text(reason: str, expected: str) -> None:
    assert journal_view.describe_reason(reason) == expected


def test_describe_reason_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        journal_view.describe_reason("x")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            datetime(2026, 1, 14, 12, 0, tzinfo=timezone.utc),
            "2026-01-14 20:00 Manila (12:00 UTC)",
        ),
        (
            datetime(2026, 1, 14, 20, 0, tzinfo=timezone.utc),
            "2026-01-15 04:00 Manila (20:00 UTC)",
        ),
        (
            datetime(
                2026,
                1,
                14,
                7,
                0,
                tzinfo=ZoneInfo("America/New_York"),
            ),
            "2026-01-14 20:00 Manila (12:00 UTC)",
        ),
        (
            pd.Timestamp("2026-01-14 12:00:00", tz="UTC").as_unit("us"),
            "2026-01-14 20:00 Manila (12:00 UTC)",
        ),
    ],
)
def test_format_local_time(value: datetime, expected: str) -> None:
    assert journal_view.format_local_time(value) == expected


def test_format_local_time_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        journal_view.format_local_time(datetime(2026, 1, 14, 12, 0))


@pytest.mark.parametrize(
    ("side", "exit_price", "expected"),
    [
        ("long", 1.10200, 16.0),
        ("short", 1.09800, 16.0),
        ("long", 1.09900, -8.0),
        ("short", 1.10100, -8.0),
        ("long", 1.10000, 0.0),
    ],
)
def test_gross_pnl_estimate_hand_values(
    side: str,
    exit_price: float,
    expected: float,
) -> None:
    assert journal_view.gross_pnl_estimate(
        side,
        1.10000,
        exit_price,
        8000,
    ) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("side", "entry", "exit_price", "units"),
    [
        ("buy", 1.10000, 1.10100, 1000),
        ("long", 0, 1.10100, 1000),
        ("long", 1.10000, float("nan"), 1000),
        ("long", 1.10000, 1.10100, 0),
        ("long", 1.10000, 1.10100, 1000.0),
        ("long", 1.10000, 1.10100, True),
    ],
)
def test_gross_pnl_estimate_rejects_invalid_values(
    side: str,
    entry: float,
    exit_price: float,
    units: object,
) -> None:
    with pytest.raises(ValueError):
        journal_view.gross_pnl_estimate(side, entry, exit_price, units)


def test_trade_table_rows_exact_shape_and_preserved_order() -> None:
    trades = [
        {
            "id": 1,
            "opened_at": datetime(2026, 1, 14, 12, 0, tzinfo=timezone.utc),
            "side": "long",
            "mode": "paper",
            "size_units": 8000,
            "planned_risk_amount": 9.12,
            "closed_at": datetime(2026, 1, 14, 13, 0, tzinfo=timezone.utc),
            "realized_pnl": -4.56,
            "r_multiple": -0.5,
        },
        {
            "id": 2,
            "opened_at": datetime(2026, 1, 14, 14, 0, tzinfo=timezone.utc),
            "side": "short",
            "mode": "demo",
            "size_units": 5000,
            "planned_risk_amount": 12.3456,
            "closed_at": None,
            "realized_pnl": None,
            "r_multiple": None,
        },
    ]

    rows = journal_view.trade_table_rows(trades)

    assert rows == [
        {
            "id": 1,
            "opened": "2026-01-14 20:00 Manila (12:00 UTC)",
            "side": "long",
            "mode": "paper",
            "units": 8000,
            "planned_risk_usd": 9.12,
            "status": "closed",
            "result_usd": -4.56,
            "r_multiple": -0.5,
        },
        {
            "id": 2,
            "opened": "2026-01-14 22:00 Manila (14:00 UTC)",
            "side": "short",
            "mode": "demo",
            "units": 5000,
            "planned_risk_usd": 12.35,
            "status": "open",
            "result_usd": None,
            "r_multiple": None,
        },
    ]
    assert all(
        set(row)
        == {
            "id",
            "opened",
            "side",
            "mode",
            "units",
            "planned_risk_usd",
            "status",
            "result_usd",
            "r_multiple",
        }
        for row in rows
    )
    assert journal_view.trade_table_rows([]) == []


def test_journal_path_uses_environment_override(monkeypatch) -> None:
    monkeypatch.setenv(journal_view.JOURNAL_PATH_ENV, "custom/journal.sqlite3")
    assert journal_view.journal_path() == "custom/journal.sqlite3"


@pytest.mark.parametrize("configured", [None, ""])
def test_journal_path_uses_default_when_unset_or_empty(
    monkeypatch,
    configured: str | None,
) -> None:
    if configured is None:
        monkeypatch.delenv(journal_view.JOURNAL_PATH_ENV, raising=False)
    else:
        monkeypatch.setenv(journal_view.JOURNAL_PATH_ENV, configured)
    assert journal_view.journal_path() == DEFAULT_JOURNAL_PATH


def test_get_now_returns_timezone_aware_utc() -> None:
    now = journal_view.get_now()
    assert now.tzinfo is timezone.utc
    assert now.utcoffset().total_seconds() == 0
