from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.app import journal_view
from src.journal.store import add_trade, get_trade, list_trades, open_journal
from src.journal.workflow import open_planned_trade


NOW = datetime(2026, 1, 14, 12, 0, tzinfo=timezone.utc)


def _load_app(tmp_path: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv(
        journal_view.JOURNAL_PATH_ENV,
        str(tmp_path / "j.sqlite3"),
    )
    monkeypatch.setattr(journal_view, "get_now", lambda: NOW)
    return AppTest.from_file(
        "../src/app/journal_page.py",
        default_timeout=30,
    ).run()


def _metric(app: AppTest, label: str) -> str:
    matches = [item for item in app.metric if item.label == label]
    assert len(matches) == 1
    return matches[0].value


def _set_plan(
    app: AppTest,
    *,
    reason: str = "ui test",
    side: str = "long",
    entry: float = 1.10000,
    stop: float = 1.09900,
    risk_pct: float = 1.0,
    starting_balance: float = 1000.0,
) -> None:
    app.number_input(key="starting_balance").set_value(starting_balance)
    app.radio(key="side").set_value(side)
    app.number_input(key="entry").set_value(entry)
    app.number_input(key="stop").set_value(stop)
    app.number_input(key="risk_pct").set_value(risk_pct)
    app.text_input(key="reason").set_value(reason)
    app.run()


def _trade_file(path: Path) -> list[dict]:
    conn = open_journal(path)
    try:
        return list_trades(conn)
    finally:
        conn.close()


def test_first_load_shows_empty_journal_and_safe_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _load_app(tmp_path, monkeypatch)

    assert not app.exception
    assert any("Paper and demo only" in item.value for item in app.warning)
    assert app.selectbox(key="mode").options == ["paper", "demo"]
    assert app.number_input(key="risk_pct").max == 2.0
    assert "No trades yet." in [item.value for item in app.text]
    assert "No open trades." in [item.value for item in app.text]
    assert _metric(app, "Trades today") == "0"
    assert _metric(app, "Balance now (USD)") == "1000.00"


def test_plan_shows_cost_aware_hand_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _load_app(tmp_path, monkeypatch)
    _set_plan(app)

    assert not app.exception
    assert _metric(app, "Stop distance (pips)") == "10.0"
    assert _metric(app, "Cost-aware risk (pips)") == "11.4"
    assert _metric(app, "Units") == "8,000"
    assert _metric(app, "Lots") == "0.08"
    assert _metric(app, "Planned risk (USD)") == "9.12"
    assert _metric(app, "Risk % of balance") == "0.91%"
    assert _metric(app, "Can save today?") == "Yes"
    assert not app.button(key="save_trade").disabled


def test_save_updates_journal_metrics_and_displays_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _load_app(tmp_path, monkeypatch)
    _set_plan(app)

    app.button(key="save_trade").click().run()

    path = tmp_path / "j.sqlite3"
    trades = _trade_file(path)
    assert len(trades) == 1
    assert trades[0]["size_units"] == 8000
    assert trades[0]["planned_risk_amount"] == pytest.approx(9.12, abs=1e-9)
    assert trades[0]["reason"] == "ui test"
    assert trades[0]["mode"] == "paper"
    assert trades[0]["side"] == "long"
    assert trades[0]["opened_at"] == NOW
    assert any("Saved trade #1" in item.value for item in app.success)
    assert _metric(app, "Trades today") == "1"
    assert _metric(app, "Open positions") == "1"
    assert _metric(app, "Can save today?") == "No"
    assert any(
        item.value == "A position is already open (maximum 1)."
        for item in app.warning
    )
    assert app.button(key="save_trade").disabled
    assert not app.exception


@pytest.mark.parametrize("reason", ["", "   "])
def test_empty_reason_disables_save_without_writing(
    tmp_path: Path,
    monkeypatch,
    reason: str,
) -> None:
    app = _load_app(tmp_path, monkeypatch)
    _set_plan(app, reason=reason)

    assert app.button(key="save_trade").disabled
    assert _trade_file(tmp_path / "j.sqlite3") == []


@pytest.mark.parametrize(
    ("side", "stop"),
    [("long", 1.10100), ("short", 1.09900)],
)
def test_invalid_plan_shows_error_without_exception(
    tmp_path: Path,
    monkeypatch,
    side: str,
    stop: float,
) -> None:
    app = _load_app(tmp_path, monkeypatch)
    _set_plan(app, side=side, stop=stop)

    assert app.error
    assert not app.exception
    assert app.button(key="save_trade").disabled
    assert _trade_file(tmp_path / "j.sqlite3") == []


def test_close_selected_trade_uses_account_result_and_refreshes_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "j.sqlite3"
    conn = open_journal(path)
    try:
        open_planned_trade(
            conn,
            opened_at=datetime(2026, 1, 14, 11, 0, tzinfo=timezone.utc),
            starting_balance=1000,
            risk_pct=1.0,
            side="long",
            entry_price=1.10000,
            stop_price=1.09900,
            mode="paper",
            reason="pre",
        )
        add_trade(
            conn,
            opened_at=datetime(2026, 1, 14, 11, 30, tzinfo=timezone.utc),
            side="short",
            entry_price=1.10000,
            stop_price=1.10100,
            size_units=1000,
            planned_risk_amount=11.4,
            mode="demo",
            reason="second",
        )
    finally:
        conn.close()

    app = _load_app(tmp_path, monkeypatch)
    app.selectbox(key="close_trade_id").select(2).run()
    app.number_input(key="close_exit_price").set_value(1.10050).run()
    assert (
        "Gross estimate from prices (no costs): -0.50 USD"
        in [item.value for item in app.text]
    )
    app.number_input(key="close_pnl").set_value(-5.7)
    app.text_input(key="close_notes").set_value("stopped")
    app.button(key="close_trade").click().run()

    trades = _trade_file(path)
    assert trades[0]["closed_at"] is None
    assert trades[1]["closed_at"] == NOW
    assert trades[1]["exit_price"] == 1.1005
    assert trades[1]["realized_pnl"] == -5.7
    assert trades[1]["r_multiple"] == pytest.approx(-0.5, abs=1e-9)
    assert trades[1]["notes"] == "stopped"
    assert any(
        "Closed trade #2: R = -0.50" in item.value for item in app.success
    )
    assert _metric(app, "Realized P&L today (USD)") == "-5.70"
    assert _metric(app, "Open positions") == "1"
    assert not app.exception


def test_blocked_plan_with_multiple_open_trades_cannot_save(
    tmp_path: Path,
    monkeypatch,
) -> None:
    conn = open_journal(tmp_path / "j.sqlite3")
    try:
        for position, opened_at in enumerate(
            (
                datetime(2026, 1, 14, 11, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 14, 11, 30, tzinfo=timezone.utc),
            )
        ):
            add_trade(
                conn,
                opened_at=opened_at,
                side="long",
                entry_price=1.10000,
                stop_price=1.09900,
                size_units=1000,
                planned_risk_amount=11.4,
                mode="paper",
                reason=f"open {position}",
            )
    finally:
        conn.close()

    app = _load_app(tmp_path, monkeypatch)
    _set_plan(app, reason="valid reason")

    assert any(
        item.value == "A position is already open (maximum 1)."
        for item in app.warning
    )
    assert app.button(key="save_trade").disabled
    assert len(_trade_file(tmp_path / "j.sqlite3")) == 2
    assert not app.exception


def test_nondefault_starting_balance_drives_balances_and_sizing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _load_app(tmp_path, monkeypatch)
    _set_plan(app, starting_balance=2000.0)

    assert _metric(app, "Balance now (USD)") == "2000.00"
    assert _metric(app, "Start-of-day balance (USD)") == "2000.00"
    assert _metric(app, "Units") == "17,000"
    assert _metric(app, "Planned risk (USD)") == "19.38"
    assert not app.exception


def test_page_uses_checked_workflow_without_data_loaders() -> None:
    source = Path("src/app/journal_page.py").read_text(encoding="utf-8")

    assert "open_planned_trade" in source
    assert "close_trade" in source
    assert "get_now" in source
    assert "add_trade(" not in source
    assert "load_candles" not in source
    assert "live" not in source
