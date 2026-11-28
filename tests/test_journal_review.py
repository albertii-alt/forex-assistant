from __future__ import annotations

import itertools
import math
import random
import statistics
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.journal.review import (
    format_review,
    review_trades,
    sample_size_note,
)
from src.journal.store import add_trade, close_trade, list_trades, open_journal


UTC = timezone.utc


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def build_closed_fixture(pnls: list[float], modes: list[str] | None = None) -> tuple[object, list[dict]]:
    conn = open_journal(":memory:")
    trades = []
    for index, pnl in enumerate(pnls):
        opened = utc(2026, 2, 3, 9 + index)
        closed = utc(2026, 2, 3, 10 + index)
        trade_id = add_trade(
            conn,
            opened_at=opened,
            side="long",
            entry_price=1.1,
            stop_price=1.099,
            size_units=1000,
            planned_risk_amount=10,
            mode=modes[index] if modes else "paper",
            reason="r",
        )
        close_trade(
            conn,
            trade_id,
            closed_at=closed,
            exit_price=1.1,
            realized_pnl=pnl,
        )
    trades = list_trades(conn)
    return conn, trades


def test_a_main_fixture_reverse_insert_and_format() -> None:
    conn = open_journal(":memory:")
    pnls = [20, -10, -10, -10, 15, 0, -10, -10, 30, -5]
    ids = []
    for k in range(9, -1, -1):
        opened = utc(2026, 2, 2, 10 + k) - timedelta(minutes=30)
        closed = utc(2026, 2, 2, 10 + k)
        trade_id = add_trade(
            conn,
            opened_at=opened,
            side="long",
            entry_price=1.1,
            stop_price=1.099,
            size_units=1000,
            planned_risk_amount=10,
            mode="demo" if k in (0, 5) else "paper",
            reason="r",
        )
        ids.append(trade_id)
        close_trade(
            conn,
            trade_id,
            closed_at=closed,
            exit_price=1.1,
            realized_pnl=pnls[k],
        )
    add_trade(
        conn,
        opened_at=utc(2026, 2, 2, 21),
        side="long",
        entry_price=1.1,
        stop_price=1.099,
        size_units=1000,
        planned_risk_amount=10,
        mode="demo",
        reason="r",
    )
    trades = list_trades(conn)
    review = review_trades(trades)

    assert review["n_closed"] == 10
    assert review["n_open"] == 1
    assert (review["n_wins"], review["n_losses"], review["n_breakeven"]) == (3, 6, 1)
    assert review["win_rate"] == pytest.approx(0.3)
    assert review["total_r"] == pytest.approx(1.0)
    assert review["avg_r"] == pytest.approx(0.1)
    assert review["avg_win_r"] == pytest.approx(6.5 / 3)
    assert review["avg_loss_r"] == pytest.approx(-5.5 / 6)
    assert review["profit_factor"] == pytest.approx(6.5 / 5.5)
    assert review["longest_losing_streak"] == 3
    assert review["total_pnl_usd"] == pytest.approx(10.0)
    assert review["by_mode"] == {"paper": 8, "demo": 2}

    values = [2.0, -1.0, -1.0, -1.0, 1.5, 0.0, -1.0, -1.0, 3.0, -0.5]
    mean = sum(values) / len(values)
    deviations_squared = sum((value - mean) ** 2 for value in values)
    variance = deviations_squared / (len(values) - 1)
    sd = math.sqrt(variance)
    stderr = sd / math.sqrt(len(values))
    low = mean - 2 * stderr
    high = mean + 2 * stderr
    assert deviations_squared == pytest.approx(20.4)
    assert variance == pytest.approx(20.4 / 9)
    assert sd == pytest.approx(1.505545305, abs=1e-6)
    assert stderr == pytest.approx(0.476095228, abs=1e-6)
    assert low == pytest.approx(-0.852190455, abs=1e-6)
    assert high == pytest.approx(1.052190455, abs=1e-6)
    assert review["avg_r_stderr"] == pytest.approx(stderr, abs=1e-6)
    assert review["avg_r_low"] == pytest.approx(low, abs=1e-6)
    assert review["avg_r_high"] == pytest.approx(high, abs=1e-6)
    assert review["max_drawdown_r"] == pytest.approx(3.5)

    assert format_review(review) == "\n".join([
        "Journal review (closed trades only)",
        "Closed trades: 10 (open: 1)",
        "Wins / losses / breakeven: 3 / 6 / 1",
        "Win rate: 30.0%",
        "Average R per trade: 0.10 (95% range shown from 30 closed trades)",
        "Total R: 1.00",
        "Total result (USD): 10.00",
        "Average win R: 2.17 | average loss R: -0.92",
        "Profit factor: 1.18",
        "Max drawdown (R): 3.50",
        "Longest losing streak: 3",
        "Too few closed trades (10) to conclude anything. A few wins or losses in a row are normal luck.",
        "Planning aid only, not financial advice.",
    ])
    assert len(format_review(review).splitlines()) == 13


@pytest.mark.parametrize(
    ("pnls", "expected"),
    [
        ([10, 20], {"win_rate": 1.0, "avg_r": 1.5, "total_r": 3.0, "profit_factor": math.inf, "avg_loss_r": None, "longest_losing_streak": 0, "max_drawdown_r": 0.0, "avg_r_stderr": 0.5, "avg_r_low": 0.5, "avg_r_high": 2.5}),
        ([-10], {"win_rate": 0.0, "avg_r": -1.0, "avg_r_stderr": None, "avg_r_low": None, "avg_r_high": None, "profit_factor": 0.0, "avg_win_r": None, "avg_loss_r": -1.0, "longest_losing_streak": 1, "max_drawdown_r": 1.0, "total_r": -1.0}),
        ([0, 0], {"n_breakeven": 2, "n_wins": 0, "n_losses": 0, "win_rate": 0.0, "profit_factor": None, "avg_r": 0.0, "avg_r_stderr": 0.0, "avg_r_low": 0.0, "avg_r_high": 0.0, "max_drawdown_r": 0.0, "longest_losing_streak": 0}),
        ([-10, -10, 30], {"max_drawdown_r": 2.0, "longest_losing_streak": 2, "total_r": 1.0, "win_rate": 1 / 3, "profit_factor": 1.5}),
        ([-10, 0, -10], {"longest_losing_streak": 1, "max_drawdown_r": 2.0, "profit_factor": 0.0}),
    ],
)
def test_b_small_fixtures(pnls: list[float], expected: dict) -> None:
    conn, trades = build_closed_fixture(pnls)
    review = review_trades(trades)
    for key, value in expected.items():
        if isinstance(value, float) and math.isfinite(value):
            assert review[key] == pytest.approx(value, abs=1e-9)
        else:
            assert review[key] == value
    conn.close()


def test_b1_exact_format_and_b2_b6_formats() -> None:
    _, winners = build_closed_fixture([10, 20])
    winners_text = format_review(review_trades(winners))
    assert "Profit factor: inf" in winners_text
    assert (
        "Average R per trade: 1.50 "
        "(95% range shown from 30 closed trades)"
        in winners_text
    )
    _, loser = build_closed_fixture([-10])
    loser_text = format_review(review_trades(loser))
    assert "Average R per trade: -1.00" in loser_text
    assert "rough 95%" not in loser_text
    _, breakeven = build_closed_fixture([0, 0])
    breakeven_text = format_review(review_trades(breakeven))
    assert "Profit factor: -" in breakeven_text
    assert (
        "Average R per trade: 0.00 "
        "(95% range shown from 30 closed trades)"
        in breakeven_text
    )

    empty = review_trades([])
    assert format_review(empty) == "\n".join([
        "Journal review (closed trades only)",
        "Closed trades: 0 (open: 0)",
        "Wins / losses / breakeven: 0 / 0 / 0",
        "Win rate: -",
        "Average R per trade: -",
        "Total R: 0.00",
        "Total result (USD): 0.00",
        "Average win R: - | average loss R: -",
        "Profit factor: -",
        "Max drawdown (R): 0.00",
        "Longest losing streak: 0",
        "No closed trades yet.",
        "Planning aid only, not financial advice.",
    ])


def test_b7_only_open_trades() -> None:
    conn = open_journal(":memory:")
    for index in range(2):
        add_trade(
            conn,
            opened_at=utc(2026, 2, 3, 9 + index),
            side="long",
            entry_price=1.1,
            stop_price=1.099,
            size_units=1000,
            planned_risk_amount=10,
            mode="demo",
            reason="r",
        )
    review = review_trades(list_trades(conn))
    assert review["n_closed"] == 0
    assert review["n_open"] == 2
    assert review["win_rate"] is None
    assert review["profit_factor"] is None
    conn.close()


def test_c1_closed_order_differs_from_opened_order() -> None:
    trades = [
        {"id": 1, "mode": "paper", "closed_at": utc(2026, 2, 4, 10, 30), "r_multiple": 3.0, "realized_pnl": 30.0, "opened_at": utc(2026, 2, 4, 8)},
        {"id": 2, "mode": "paper", "closed_at": utc(2026, 2, 4, 10), "r_multiple": -1.0, "realized_pnl": -10.0, "opened_at": utc(2026, 2, 4, 9)},
        {"id": 3, "mode": "paper", "closed_at": utc(2026, 2, 4, 11), "r_multiple": -2.0, "realized_pnl": -20.0, "opened_at": utc(2026, 2, 4, 9, 30)},
    ]
    review = review_trades(trades)
    assert review["max_drawdown_r"] == pytest.approx(2.0)
    assert review["longest_losing_streak"] == 1
    assert review["total_r"] == pytest.approx(0.0)


def test_c2_closed_at_tie_broken_by_id_and_input_reversal() -> None:
    trades = [
        {"id": 1, "mode": "paper", "closed_at": utc(2026, 2, 5, 10), "r_multiple": -1.0, "realized_pnl": -10.0},
        {"id": 2, "mode": "paper", "closed_at": utc(2026, 2, 5, 10), "r_multiple": 3.0, "realized_pnl": 30.0},
        {"id": 3, "mode": "paper", "closed_at": utc(2026, 2, 5, 9), "r_multiple": -1.0, "realized_pnl": -10.0},
        {"id": 4, "mode": "paper", "closed_at": utc(2026, 2, 5, 11), "r_multiple": -2.0, "realized_pnl": -20.0},
    ]
    review = review_trades(trades)
    assert review["max_drawdown_r"] == pytest.approx(2.0)
    assert review["longest_losing_streak"] == 2
    assert review["total_r"] == pytest.approx(-1.0)
    assert review_trades(trades) == review_trades(list(reversed(trades)))


def test_c3_seeded_shuffle_order_independence() -> None:
    rng = random.Random(88)
    base_trades = [
        {
            "id": index + 1,
            "mode": "demo" if index in (0, 5) else "paper",
            "closed_at": utc(2026, 2, 2, 10 + index),
            "r_multiple": value,
            "realized_pnl": value * 10,
        }
        for index, value in enumerate(
            [2.0, -1.0, -1.0, -1.0, 1.5, 0.0, -1.0, -1.0, 3.0, -0.5]
        )
    ]
    expected = review_trades(base_trades)
    for seed in range(20):
        shuffled = list(base_trades)
        rng.shuffle(shuffled)
        assert review_trades(shuffled) == expected


@pytest.mark.parametrize("n", [0, 1, 29, 30, 99, 100, 250])
def test_d_sample_size_boundaries(n: int) -> None:
    note = sample_size_note(n)
    if n == 0:
        assert note == "No closed trades yet."
    elif n < 30:
        assert note.startswith(f"Too few closed trades ({n})")
    elif n < 100:
        assert note.startswith(f"Still a small sample ({n} closed trades)")
    else:
        assert note.startswith(f"Larger sample ({n} closed trades)")


@pytest.mark.parametrize("n", [-1, 2.0, True, "5"])
def test_d_invalid_sample_size(n: object) -> None:
    with pytest.raises(ValueError):
        sample_size_note(n)


@pytest.mark.parametrize(
    ("trades", "match"),
    [
        (None, "list"),
        ([None], "record 0"),
        ([{"id": 1, "mode": "paper", "closed_at": utc(2026, 1, 1), "r_multiple": None, "realized_pnl": 1}], "record 0"),
        ([{"id": 1, "mode": "paper", "closed_at": utc(2026, 1, 1), "r_multiple": math.nan, "realized_pnl": 1}], "record 0"),
        ([{"id": 1, "mode": "paper", "closed_at": utc(2026, 1, 1), "r_multiple": True, "realized_pnl": 1}], "record 0"),
        ([{"id": 1, "mode": "paper", "closed_at": utc(2026, 1, 1), "r_multiple": 1, "realized_pnl": "5"}], "record 0"),
        ([{"id": 1, "mode": "paper", "closed_at": datetime(2026, 1, 1), "r_multiple": 1, "realized_pnl": 1}], "record 0"),
        ([{"id": 1, "closed_at": utc(2026, 1, 1), "r_multiple": 1, "realized_pnl": 1}], "record 0"),
    ],
)
def test_e_validation(trades: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        review_trades(trades)


def test_e_second_record_index() -> None:
    valid = {"id": 1, "mode": "paper", "closed_at": utc(2026, 1, 1), "r_multiple": 1, "realized_pnl": 10}
    bad = {**valid, "r_multiple": None}
    with pytest.raises(ValueError, match="record 1"):
        review_trades([valid, bad])


def test_e_open_trade_garbage_r_is_accepted() -> None:
    trade = {"id": 1, "mode": "demo", "closed_at": None, "r_multiple": None, "realized_pnl": "garbage"}
    review = review_trades([trade])
    assert review["n_open"] == 1
    assert review["n_closed"] == 0


def test_f_pd_timestamp_ties_and_out_of_order() -> None:
    timestamps = [
        pd.Timestamp("2026-02-05 10:00:00", tz="UTC").as_unit("us"),
        pd.Timestamp("2026-02-05 09:00:00", tz="UTC").as_unit("us"),
    ]
    trades = [
        {"id": 1, "mode": "paper", "closed_at": timestamps[0], "r_multiple": -1.0, "realized_pnl": -10},
        {"id": 2, "mode": "paper", "closed_at": timestamps[1], "r_multiple": 3.0, "realized_pnl": 30},
    ]
    review = review_trades(trades)
    assert review["total_r"] == pytest.approx(2.0)
    assert review["max_drawdown_r"] == pytest.approx(1.0)


def reference_review(trades: list[dict]) -> dict:
    closed = [trade for trade in trades if trade["closed_at"] is not None]
    closed.sort(key=lambda trade: (trade["closed_at"], trade["id"]))
    values = [float(trade["r_multiple"]) for trade in closed]
    n_closed = len(values)
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    positive = sum(wins)
    negative = sum(losses)
    if losses:
        profit_factor = positive / abs(negative)
    elif wins:
        profit_factor = math.inf
    else:
        profit_factor = None
    mean = sum(values) / n_closed if n_closed else None
    stderr = statistics.stdev(values) / math.sqrt(n_closed) if n_closed >= 2 else None
    low = mean - 2 * stderr if stderr is not None else None
    high = mean + 2 * stderr if stderr is not None else None
    curve = [0.0]
    for value in values:
        curve.append(curve[-1] + value)
    max_drawdown = 0.0
    for i in range(len(curve)):
        for j in range(i, len(curve)):
            max_drawdown = max(max_drawdown, curve[i] - curve[j])
    streak = longest = 0
    for _, group in itertools.groupby(values, key=lambda value: value < 0):
        run = list(group)
        if run and run[0] < 0:
            longest = max(longest, len(run))
    by_mode = {}
    for trade in closed:
        by_mode[trade["mode"]] = by_mode.get(trade["mode"], 0) + 1
    return {
        "n_closed": n_closed,
        "n_open": sum(trade["closed_at"] is None for trade in trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_breakeven": sum(value == 0 for value in values),
        "win_rate": len(wins) / n_closed if n_closed else None,
        "total_r": sum(values),
        "avg_r": mean,
        "avg_r_stderr": stderr,
        "avg_r_low": low,
        "avg_r_high": high,
        "avg_win_r": sum(wins) / len(wins) if wins else None,
        "avg_loss_r": sum(losses) / len(losses) if losses else None,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown,
        "longest_losing_streak": longest,
        "total_pnl_usd": sum(float(trade["realized_pnl"]) for trade in closed),
        "by_mode": by_mode,
    }


def test_g_seeded_reference() -> None:
    rng = random.Random(88)
    r_choices = [-2, -1, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
    for case in range(200):
        n_closed = rng.randint(0, 60)
        trades = []
        base = utc(2026, 2, 1)
        for index in range(n_closed):
            r_value = rng.choice(r_choices) if rng.choice([True, False]) else rng.uniform(-2, 3)
            minute = rng.randint(0, 24 * 60)
            trades.append({
                "id": rng.randint(1, 100000),
                "mode": rng.choice(["paper", "demo"]),
                "closed_at": base + timedelta(minutes=minute),
                "r_multiple": r_value,
                "realized_pnl": r_value * 10,
            })
        for index in range(rng.randint(0, 5)):
            trades.append({
                "id": 100000 + index,
                "mode": rng.choice(["paper", "demo"]),
                "closed_at": None,
                "r_multiple": None,
                "realized_pnl": None,
            })
        actual = review_trades(trades)
        reference = reference_review(trades)
        for key in (
            "n_closed", "n_open", "n_wins", "n_losses", "n_breakeven",
            "total_r", "avg_r", "avg_r_stderr", "avg_r_low", "avg_r_high",
            "avg_win_r", "avg_loss_r", "profit_factor", "max_drawdown_r",
            "longest_losing_streak", "total_pnl_usd", "win_rate",
        ):
            if reference[key] is None:
                assert actual[key] is None
            elif isinstance(reference[key], float) and math.isinf(reference[key]):
                assert math.isinf(actual[key])
            else:
                assert actual[key] == pytest.approx(reference[key], abs=1e-9)
        assert actual["by_mode"] == reference["by_mode"]
        formatted_lines = format_review(actual).splitlines()
        assert len(formatted_lines) == 13
        average_text = (
            "-" if reference["avg_r"] is None
            else f"{reference['avg_r']:.2f}"
        )
        if reference["avg_r_stderr"] is not None:
            if n_closed >= 30:
                average_text += (
                    f" (rough 95% range {reference['avg_r_low']:.2f} "
                    f"to {reference['avg_r_high']:.2f})"
                )
            else:
                average_text += " (95% range shown from 30 closed trades)"
        assert formatted_lines[4] == f"Average R per trade: {average_text}"


def build_alternating_trades(n: int) -> tuple[object, list[dict]]:
    conn = open_journal(":memory:")
    start = utc(2026, 2, 6, 10)
    for k in range(n):
        closed_at = start + timedelta(hours=k)
        trade_id = add_trade(
            conn,
            opened_at=closed_at - timedelta(minutes=30),
            side="long",
            entry_price=1.1,
            stop_price=1.099,
            size_units=1000,
            planned_risk_amount=10,
            mode="paper",
            reason="r",
        )
        close_trade(
            conn,
            trade_id,
            closed_at=closed_at,
            exit_price=1.1,
            realized_pnl=10 if k % 2 == 0 else -10,
        )
    return conn, list_trades(conn)


def test_n30_range_shown_at_boundary() -> None:
    conn, trades = build_alternating_trades(30)
    review = review_trades(trades)
    assert review["n_wins"] == 15
    assert review["n_losses"] == 15
    assert review["avg_r"] == pytest.approx(0.0, abs=1e-9)
    assert review["total_r"] == pytest.approx(0.0, abs=1e-9)
    assert review["avg_r_stderr"] == pytest.approx(0.1856953, abs=1e-6)
    assert review["avg_r_low"] == pytest.approx(-0.3713907, abs=1e-6)
    assert review["avg_r_high"] == pytest.approx(0.3713907, abs=1e-6)
    assert review["win_rate"] == pytest.approx(0.5, abs=1e-9)
    assert review["profit_factor"] == pytest.approx(1.0, abs=1e-9)
    assert review["max_drawdown_r"] == pytest.approx(1.0, abs=1e-9)
    assert review["longest_losing_streak"] == 1
    assert review["sample_note"].startswith("Still a small sample (30 closed trades)")
    average_line = format_review(review).splitlines()[4]
    assert average_line == (
        "Average R per trade: 0.00 (rough 95% range -0.37 to 0.37)"
    )
    conn.close()


def test_n29_range_withheld_below_boundary() -> None:
    conn, trades = build_alternating_trades(29)
    review = review_trades(trades)
    assert review["avg_r"] == pytest.approx(1 / 29, abs=1e-9)
    assert review["win_rate"] == pytest.approx(15 / 29, abs=1e-9)
    assert review["total_r"] == pytest.approx(1.0, abs=1e-9)
    assert review["profit_factor"] == pytest.approx(15 / 14, abs=1e-9)
    assert review["max_drawdown_r"] == pytest.approx(1.0, abs=1e-9)
    assert review["longest_losing_streak"] == 1
    assert review["sample_note"].startswith("Too few closed trades (29)")
    average_line = format_review(review).splitlines()[4]
    assert average_line == (
        "Average R per trade: 0.03 "
        "(95% range shown from 30 closed trades)"
    )
    conn.close()


def test_i_source_check() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "journal" / "review.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("urllib", "socket", "requests", "open(", "load_candles", "sqlite3"):
        assert forbidden not in source
