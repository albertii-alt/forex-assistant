"""Streamlit interface for the personal paper and demo trade journal."""

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.app.journal_view as journal_view
import src.news.service as news_service
from src.journal.store import (
    ALLOWED_MODES,
    close_trade,
    get_trade,
    list_trades,
    open_journal,
    trading_day_summary,
)
from src.journal.workflow import account_balances, open_planned_trade, plan_trade


def _queue_save() -> None:
    st.session_state["_save_pending"] = True


def _queue_close() -> None:
    st.session_state["_close_pending"] = True


st.set_page_config(page_title="Journal (paper/demo)")
st.title("Trade journal")
st.warning("Paper and demo only. Experimental. Not financial advice.")

now = journal_view.get_now()
context = news_service.load_news_context(now, news_service.news_directory())
conn = open_journal(journal_view.journal_path())
save_success = None
save_error = None
close_success = None
close_error = None

try:
    if st.session_state.get("_save_pending", False):
        st.session_state["_save_pending"] = False
        try:
            result = open_planned_trade(
                conn,
                opened_at=now,
                starting_balance=float(
                    st.session_state.get("starting_balance", 1000.0)
                ),
                risk_pct=float(st.session_state.get("risk_pct", 1.0)),
                side=st.session_state.get("side", "long"),
                entry_price=float(st.session_state.get("entry", 1.10000)),
                stop_price=float(st.session_state.get("stop", 1.09900)),
                mode=st.session_state.get("mode", ALLOWED_MODES[0]),
                reason=st.session_state.get("reason", ""),
                strategy=st.session_state.get("strategy", "").strip()
                or "manual",
            )
            trade_id = result["trade_id"]
            mode = st.session_state.get("mode", ALLOWED_MODES[0])
            save_success = f"Saved trade #{trade_id} ({mode})."
        except ValueError as exc:
            save_error = str(exc)

    if st.session_state.get("_close_pending", False):
        st.session_state["_close_pending"] = False
        try:
            selected_id = int(st.session_state["close_trade_id"])
            close_trade(
                conn,
                selected_id,
                closed_at=now,
                exit_price=float(
                    st.session_state.get("close_exit_price", 1.10000)
                ),
                realized_pnl=float(st.session_state.get("close_pnl", 0.0)),
                notes=st.session_state.get("close_notes", ""),
            )
            closed_trade = get_trade(conn, selected_id)
            if closed_trade is None or closed_trade["r_multiple"] is None:
                raise ValueError(f"Closed trade #{selected_id} could not be read.")
            close_success = (
                f"Closed trade #{selected_id}: "
                f"R = {closed_trade['r_multiple']:.2f}"
            )
        except ValueError as exc:
            close_error = str(exc)

    st.write("Now: " + journal_view.format_local_time(now))
    st.header("News")
    if context["fetched_at"] is not None:
        st.text(
            f"Calendar fetched: "
            f"{journal_view.format_local_time(context['fetched_at'])}, "
            f"source: {context['source']}"
        )
    st.text(context["briefing"])
    st.header("Today")
    starting_balance = st.number_input(
        "Paper account starting balance (USD) - always enter the same number",
        min_value=1.0,
        value=1000.0,
        step=100.0,
        key="starting_balance",
    )
    st.text(
        "Limits: max 2 trades per day, daily loss limit 2% of the "
        "start-of-day balance, 1 open position, risk per trade at most 2%."
    )

    balances = account_balances(conn, starting_balance, now)
    summary = trading_day_summary(conn, now)
    metrics = st.columns(5)
    metrics[0].metric(
        "Balance now (USD)",
        f"{balances['current_balance']:.2f}",
    )
    metrics[1].metric(
        "Start-of-day balance (USD)",
        f"{balances['start_of_day_balance']:.2f}",
    )
    metrics[2].metric("Trades today", str(summary["trades_taken_today"]))
    metrics[3].metric(
        "Realized P&L today (USD)",
        f"{summary['realized_pnl_today']:.2f}",
    )
    metrics[4].metric("Open positions", str(summary["open_positions"]))
    st.write(f"Trading day: {summary['trading_day'].date().isoformat()}")

    recent_trades = list_trades(conn)[-20:][::-1]
    if not recent_trades:
        st.text("No trades yet.")
    else:
        st.dataframe(
            journal_view.trade_table_rows(recent_trades),
            key="recent_trades",
            hide_index=True,
        )

    st.header("Plan a trade")
    side = st.radio(
        "Side",
        options=("long", "short"),
        index=0,
        key="side",
    )
    entry_price = st.number_input(
        "Entry price",
        min_value=0.00001,
        value=1.10000,
        step=0.0001,
        format="%.5f",
        key="entry",
    )
    stop_price = st.number_input(
        "Stop price",
        min_value=0.00001,
        value=1.09900,
        step=0.0001,
        format="%.5f",
        key="stop",
    )
    risk_pct = st.number_input(
        "Risk per trade (%)",
        min_value=0.01,
        max_value=2.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        key="risk_pct",
    )
    reason = st.text_input(
        "Why are you taking this trade? (required)",
        value="",
        key="reason",
    )
    strategy = st.text_input("Strategy", value="manual", key="strategy")
    mode = st.selectbox(
        "Mode",
        options=list(ALLOWED_MODES),
        key="mode",
    )

    plan = None
    plan_error = None
    try:
        plan = plan_trade(
            conn,
            now=now,
            starting_balance=starting_balance,
            risk_pct=risk_pct,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
        )
    except ValueError as exc:
        plan_error = str(exc)
        st.error(plan_error)

    plan_can_save = False
    if plan is not None:
        sizing = plan["sizing"]
        plan_metrics = st.columns(4)
        plan_metrics[0].metric(
            "Stop distance (pips)",
            f"{sizing['stop_distance_pips']:.1f}",
        )
        plan_metrics[1].metric(
            "Cost-aware risk (pips)",
            f"{sizing['effective_risk_pips']:.1f}",
        )
        plan_metrics[2].metric("Units", f"{sizing['units']:,}")
        plan_metrics[3].metric("Lots", f"{sizing['lots']:.2f}")
        risk_metrics = st.columns(3)
        risk_metrics[0].metric(
            "Planned risk (USD)",
            f"{sizing['actual_risk_amount']:.2f}",
        )
        risk_metrics[1].metric(
            "Risk % of balance",
            f"{sizing['actual_risk_pct']:.2f}%",
        )
        plan_can_save = bool(plan["can_save"])
        risk_metrics[2].metric(
            "Can save today?",
            "Yes" if plan_can_save else "No",
        )
        for blocking_reason in plan["blocking_reasons"]:
            st.warning(journal_view.describe_reason(blocking_reason))

    if not context["usable"]:
        st.warning(
            "News calendar unavailable ("
            + ", ".join(context["problems"])
            + "). Check forexfactory.com manually before trading. "
            "Do not assume there is no news. To update it, run: "
            "python -m scripts.fetch_news_calendar"
        )
    elif context["in_blackout"]:
        st.warning(
            f"News blackout window active until "
            f"{news_service.manila_text(context['blackout_window']['end'])} "
            f"Manila time ({', '.join(context['blackout_window']['titles'])}). "
            "Practice rule: avoid opening new trades now."
        )

    st.button(
        "Save to journal",
        key="save_trade",
        disabled=not (plan_can_save and bool(reason.strip())),
        on_click=_queue_save,
    )
    if save_success is not None:
        st.success(save_success)
    if save_error is not None:
        st.error(save_error)

    st.header("Close a trade")
    open_trades = list_trades(conn, status="open")
    if not open_trades:
        st.text("No open trades.")
    else:
        open_by_id = {trade["id"]: trade for trade in open_trades}
        open_ids = list(open_by_id)
        if st.session_state.get("close_trade_id") not in open_ids:
            st.session_state["close_trade_id"] = open_ids[0]
        selected_id = st.selectbox(
            "Open trade",
            options=open_ids,
            format_func=lambda trade_id: (
                f"#{trade_id} {open_by_id[trade_id]['side']} "
                f"{open_by_id[trade_id]['mode']} opened "
                f"{journal_view.format_local_time(open_by_id[trade_id]['opened_at'])}"
            ),
            key="close_trade_id",
        )
        selected_trade = open_by_id[selected_id]
        exit_price = st.number_input(
            "Exit price",
            min_value=0.00001,
            value=1.10000,
            step=0.0001,
            format="%.5f",
            key="close_exit_price",
        )
        realized_result = st.number_input(
            "Realized result (USD) from your paper/demo account statement",
            value=0.0,
            step=1.0,
            format="%.2f",
            key="close_pnl",
        )
        estimate = journal_view.gross_pnl_estimate(
            selected_trade["side"],
            selected_trade["entry_price"],
            exit_price,
            selected_trade["size_units"],
        )
        st.text(f"Gross estimate from prices (no costs): {estimate:.2f} USD")
        notes = st.text_input("Closing notes", value="", key="close_notes")
        st.button(
            "Close trade",
            key="close_trade",
            on_click=_queue_close,
        )
    if close_success is not None:
        st.success(close_success)
    if close_error is not None:
        st.error(close_error)
finally:
    conn.close()
