from __future__ import annotations

import inspect
import math
import random
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN
from typing import Any

import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.risk.limits import evaluate_daily_limits
from src.risk.sizing import position_size


def _assert_close(actual: float, expected: float) -> None:
    assert actual == pytest.approx(expected, rel=0, abs=1e-9)


def _size(**overrides: Any) -> dict[str, str | int | float]:
    args: dict[str, Any] = {
        "account_balance": 1000,
        "risk_pct": 1.0,
        "entry_price": 1.1,
        "stop_price": 1.099,
    }
    args.update(overrides)
    return position_size(**args)


def test_p1_default_cost_and_rounding() -> None:
    result = _size()

    assert result["status"] == "ok"
    assert result["units"] == 8000
    for key, expected in {
        "risk_amount": 10.0,
        "stop_distance_pips": 10.0,
        "effective_risk_pips": 11.4,
        "lots": 0.08,
        "actual_risk_amount": 9.12,
        "actual_risk_pct": 0.912,
    }.items():
        _assert_close(result[key], expected)


def test_p2_short_stop_above_entry() -> None:
    result = _size(
        account_balance=10000,
        entry_price=1.1,
        stop_price=1.105,
        cost_pips=0.0,
    )

    assert result["risk_amount"] == 100.0
    assert result["stop_distance_pips"] == 50.0
    assert result["effective_risk_pips"] == 50.0
    assert result["units"] == 20000
    assert result["lots"] == 0.2
    assert result["actual_risk_amount"] == 100.0
    assert result["actual_risk_pct"] == 1.0


def test_p3_too_small_returns_zero_size() -> None:
    result = _size(
        account_balance=100,
        entry_price=1.1,
        stop_price=1.095,
        cost_pips=0.0,
    )

    assert result == {
        "status": "too_small",
        "risk_amount": 1.0,
        "stop_distance_pips": 50.0,
        "effective_risk_pips": 50.0,
        "units": 0,
        "lots": 0.0,
        "actual_risk_amount": 0.0,
        "actual_risk_pct": 0.0,
    }


def test_p4_cost_aware_rounding() -> None:
    result = _size(
        account_balance=5000,
        risk_pct=0.5,
        entry_price=1.1,
        stop_price=1.098,
    )

    _assert_close(result["risk_amount"], 25.0)
    _assert_close(result["stop_distance_pips"], 20.0)
    _assert_close(result["effective_risk_pips"], 21.4)
    assert result["units"] == 11000
    _assert_close(result["lots"], 0.11)
    _assert_close(result["actual_risk_amount"], 23.54)
    _assert_close(result["actual_risk_pct"], 0.4708)


def test_p5_exact_multiple_is_not_dropped() -> None:
    result = _size(
        account_balance=2000,
        entry_price=1.1,
        stop_price=1.098,
        cost_pips=0.0,
    )

    assert result["units"] == 10000
    assert result["lots"] == 0.1
    assert result["actual_risk_amount"] == 20.0


def test_exact_float_multiple_uses_floor_tolerance() -> None:
    result = _size(
        account_balance=300,
        risk_pct=0.1,
        entry_price=1.1,
        stop_price=1.0999,
        cost_pips=0.0,
    )

    assert result["risk_amount"] == 0.3
    assert result["stop_distance_pips"] == 1.0
    assert result["units"] == 3000
    assert result["lots"] == 0.03


def test_p6_smallest_size_boundary() -> None:
    smallest = _size(
        account_balance=100,
        entry_price=1.1,
        stop_price=1.099,
        cost_pips=0.0,
    )
    below = _size(
        account_balance=100,
        entry_price=1.1,
        stop_price=1.09899,
        cost_pips=0.0,
    )

    assert smallest["status"] == "ok"
    assert smallest["units"] == 1000
    assert smallest["lots"] == 0.01
    assert below["status"] == "too_small"


def test_p7_risk_cap_and_equal_cap() -> None:
    assert _size(risk_pct=2.0)["risk_amount"] == 20.0
    assert _size(risk_pct=3.0, max_risk_pct=5.0)["risk_amount"] == 30.0
    with pytest.raises(ValueError, match="max_risk_pct"):
        _size(risk_pct=2.01)


@pytest.mark.parametrize(
    "overrides",
    [
        {"account_balance": 0},
        {"account_balance": -5},
        {"account_balance": math.nan},
        {"account_balance": True},
        {"risk_pct": 0},
        {"risk_pct": -1},
        {"entry_price": 1.099},
        {"entry_price": 0},
        {"stop_price": -1},
        {"cost_pips": -0.1},
        {"pip_size": 0},
        {"account_balance": "1000"},
        {"entry_price": math.inf},
        {"max_risk_pct": 0},
        {"units_per_lot": 0},
        {"lot_step_units": 0},
        {"risk_pct": True},
        {"entry_price": True},
        {"stop_price": True},
        {"cost_pips": True},
        {"max_risk_pct": True},
        {"pip_size": True},
        {"units_per_lot": True},
        {"lot_step_units": True},
        {"lot_step_units": 1000.0},
        {"lot_step_units": 1000.5},
        {"lot_step_units": math.nan},
        {"lot_step_units": math.inf},
        {"lot_step_units": -1000},
    ],
)
def test_p8_invalid_position_size_arguments(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _size(**overrides)


def test_p9_side_independent_absolute_distance() -> None:
    below = _size(entry_price=1.1, stop_price=1.099)
    above = _size(entry_price=1.1, stop_price=1.101)

    assert below == above


def test_p10_exact_keys_and_python_return_types() -> None:
    result = _size()

    assert set(result) == {
        "status",
        "risk_amount",
        "stop_distance_pips",
        "effective_risk_pips",
        "units",
        "lots",
        "actual_risk_amount",
        "actual_risk_pct",
    }
    assert type(result["units"]) is int
    assert type(result["lots"]) is float


def test_units_per_lot_accepts_positive_real_values() -> None:
    result = _size(units_per_lot=100000.5)

    assert type(result["lots"]) is float
    _assert_close(result["lots"], 8000 / 100000.5)


def test_default_cost_pips_is_1_4() -> None:
    assert inspect.signature(position_size).parameters["cost_pips"].default == 1.4


def _engine_stop_out(side: str, stop_price: float) -> pd.Series:
    index = pd.date_range(
        "2023-06-14 08:00",
        periods=6,
        freq="15min",
        tz="UTC",
    ).as_unit("us")
    candles = [(1.10000, 1.10000, 1.10000, 1.10000)] * len(index)
    if side == "long":
        candles[1] = (1.10000, 1.10000, 1.09850, 1.09900)
    else:
        candles[1] = (1.10000, 1.10150, 1.10000, 1.10100)
    frame = pd.DataFrame(
        candles,
        columns=["open", "high", "low", "close"],
        index=index,
    )
    plans = pd.DataFrame(
        [
            {
                "plan_id": f"risk-consistency-{side}",
                "side": side,
                "order_type": "market",
                "start_time": index[0],
                "entry_price": math.nan,
                "stop_price": stop_price,
                "target_r": 5.0,
                "target_price": math.nan,
                "valid_candles": 1,
                "time_exit_at": index[-1] + pd.Timedelta(hours=6),
                "exit_mode": "time_close",
            }
        ]
    )
    return run_backtest(frame, plans).iloc[0]


def test_long_engine_stop_out_within_sizing_risk() -> None:
    sizing = position_size(1000, 1.0, 1.10000, 1.09900)
    trade = _engine_stop_out("long", 1.09900)

    assert trade["status"] == "filled"
    assert trade["exit_reason"] == "stop"
    assert abs(trade["pnl_pips"]) <= sizing["effective_risk_pips"] + 1e-6


def test_short_engine_stop_out_within_sizing_risk() -> None:
    sizing = position_size(1000, 1.0, 1.10000, 1.10100)
    trade = _engine_stop_out("short", 1.10100)

    assert trade["status"] == "filled"
    assert trade["exit_reason"] == "stop"
    assert abs(trade["pnl_pips"]) <= sizing["effective_risk_pips"] + 1e-6


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _decimal_position_size(args: dict[str, Any]) -> dict[str, Any]:
    balance = _decimal(args["account_balance"])
    risk_pct = _decimal(args["risk_pct"])
    entry = _decimal(args["entry_price"])
    stop = _decimal(args["stop_price"])
    cost = _decimal(args["cost_pips"])
    pip = _decimal(args["pip_size"])
    units_per_lot = _decimal(args["units_per_lot"])
    step = Decimal(args["lot_step_units"])

    risk_amount = balance * risk_pct / Decimal(100)
    stop_pips = (abs(entry - stop) / pip).quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_EVEN,
    )
    effective = (stop_pips + cost).quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_EVEN,
    )
    raw_units = risk_amount / (effective * pip)
    units = int(
        (raw_units / step).to_integral_value(rounding=ROUND_FLOOR)
    ) * args["lot_step_units"]
    if units < args["lot_step_units"]:
        return {
            "status": "too_small",
            "risk_amount": risk_amount,
            "stop_distance_pips": stop_pips,
            "effective_risk_pips": effective,
            "units": 0,
            "lots": Decimal(0),
            "actual_risk_amount": Decimal(0),
            "actual_risk_pct": Decimal(0),
            "raw_units": raw_units,
        }
    actual_risk = Decimal(units) * effective * pip
    return {
        "status": "ok",
        "risk_amount": risk_amount,
        "stop_distance_pips": stop_pips,
        "effective_risk_pips": effective,
        "units": units,
        "lots": Decimal(units) / units_per_lot,
        "actual_risk_amount": actual_risk,
        "actual_risk_pct": actual_risk / balance * Decimal(100),
        "raw_units": raw_units,
    }


def _seeded_sizing_cases() -> list[dict[str, Any]]:
    rng = random.Random(777)
    cases = []
    for _ in range(400):
        entry = round(rng.uniform(1.0, 1.3), 5)
        distance = round(rng.uniform(3.0, 120.0), 1)
        direction = rng.choice((-1, 1))
        cases.append(
            {
                "account_balance": rng.uniform(50, 100000),
                "risk_pct": rng.uniform(0.1, 2.0),
                "entry_price": entry,
                "stop_price": round(entry + direction * distance * 0.0001, 6),
                "cost_pips": rng.choice((0.0, 1.4, 2.5)),
                "max_risk_pct": 2.0,
                "pip_size": 0.0001,
                "units_per_lot": 100000,
                "lot_step_units": 1000,
            }
        )
    return cases


def _assert_relative(actual: float, expected: Decimal) -> None:
    expected_float = float(expected)
    tolerance = 1e-9 * abs(expected_float)
    assert abs(actual - expected_float) <= tolerance


def test_decimal_reference_on_400_seeded_cases() -> None:
    tie_cases = 0
    for args in _seeded_sizing_cases():
        expected = _decimal_position_size(args)
        actual = position_size(**args)
        assert actual["status"] == expected["status"]

        expected_units = expected["units"]
        actual_units = actual["units"]
        tie = False
        if actual_units != expected_units:
            step = args["lot_step_units"]
            difference = abs(actual_units - expected_units)
            ratio = expected["raw_units"] / Decimal(step)
            nearest_integer = ratio.to_integral_value(rounding=ROUND_HALF_EVEN)
            assert difference == step
            assert abs(ratio - nearest_integer) <= Decimal("1e-7")
            tie_cases += 1
            tie = True
        else:
            assert actual_units == expected_units

        for key in (
            "risk_amount",
            "stop_distance_pips",
            "effective_risk_pips",
        ):
            _assert_relative(actual[key], expected[key])
        if tie:
            expected_lots = Decimal(actual_units) / _decimal(
                args["units_per_lot"]
            )
            effective = expected["effective_risk_pips"]
            pip = _decimal(args["pip_size"])
            actual_risk = Decimal(actual_units) * effective * pip
            expected_actual_pct = (
                actual_risk
                / _decimal(args["account_balance"])
                * Decimal(100)
            )
            _assert_relative(actual["lots"], expected_lots)
            _assert_relative(actual["actual_risk_amount"], actual_risk)
            _assert_relative(actual["actual_risk_pct"], expected_actual_pct)
        else:
            for key in ("lots", "actual_risk_amount", "actual_risk_pct"):
                _assert_relative(actual[key], expected[key])

    print(f"TEST_B_FLOAT_TIE_CASES={tie_cases}")


def test_sizing_safety_properties_on_400_cases() -> None:
    for args in _seeded_sizing_cases():
        result = position_size(**args)
        risk_amount = result["risk_amount"]
        unit_risk = result["effective_risk_pips"] * args["pip_size"]
        step = args["lot_step_units"]
        if result["status"] == "ok":
            assert result["actual_risk_amount"] <= risk_amount + 1e-9
            assert (result["units"] + step) * unit_risk > risk_amount - 1e-9
        else:
            assert step * unit_risk > risk_amount - 1e-9


@pytest.mark.parametrize(
    ("pnl", "trades", "open_positions", "expected_allowed", "expected_reasons"),
    [
        (0, 0, 0, True, []),
        (-100, 1, 0, True, []),
        (
            -200,
            2,
            0,
            False,
            ["max_trades_reached", "daily_loss_limit_reached"],
        ),
        (-200, 1, 0, False, ["daily_loss_limit_reached"]),
        (-199.99, 1, 0, True, []),
        (150, 2, 0, False, ["max_trades_reached"]),
        (0, 0, 1, False, ["position_already_open"]),
        (
            -250,
            3,
            1,
            False,
            [
                "max_trades_reached",
                "daily_loss_limit_reached",
                "position_already_open",
            ],
        ),
    ],
    ids=["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
)
def test_daily_limit_hand_fixtures(
    pnl: float,
    trades: int,
    open_positions: int,
    expected_allowed: bool,
    expected_reasons: list[str],
) -> None:
    assert evaluate_daily_limits(10000, pnl, trades, open_positions) == {
        "allowed": expected_allowed,
        "reasons": expected_reasons,
    }


def test_l9_max_trades_configuration() -> None:
    assert evaluate_daily_limits(
        10000, 0, 1, 0, max_trades_per_day=1
    ) == {"allowed": False, "reasons": ["max_trades_reached"]}
    assert evaluate_daily_limits(
        10000, 0, 1, 0, max_trades_per_day=3
    ) == {"allowed": True, "reasons": []}


def test_l10_configured_loss_limit_boundary() -> None:
    assert evaluate_daily_limits(
        10000, -299.99, 0, 0, daily_loss_limit_pct=3.0
    ) == {"allowed": True, "reasons": []}
    assert evaluate_daily_limits(
        10000, -300, 0, 0, daily_loss_limit_pct=3.0
    ) == {"allowed": False, "reasons": ["daily_loss_limit_reached"]}


def test_l11_configured_open_position_limit() -> None:
    assert evaluate_daily_limits(
        10000, 0, 0, 1, max_open_positions=2
    ) == {"allowed": True, "reasons": []}
    assert evaluate_daily_limits(
        10000, 0, 0, 2, max_open_positions=2
    ) == {"allowed": False, "reasons": ["position_already_open"]}


def test_l12_small_start_of_day_balance() -> None:
    assert evaluate_daily_limits(500, -10, 0, 0) == {
        "allowed": False,
        "reasons": ["daily_loss_limit_reached"],
    }


def test_daily_loss_tolerance_near_boundary() -> None:
    assert evaluate_daily_limits(10000, -199.9999999995, 0, 0) == {
        "allowed": False,
        "reasons": ["daily_loss_limit_reached"],
    }


def test_daily_loss_tolerance_boundary_is_inclusive() -> None:
    limit = 10000 * 2.0 / 100
    pnl = -limit + 1e-9

    assert evaluate_daily_limits(10000, pnl, 0, 0) == {
        "allowed": False,
        "reasons": ["daily_loss_limit_reached"],
    }


@pytest.mark.parametrize(
    "args",
    [
        (0, 0, 0, 0, {}),
        (-1, 0, 0, 0, {}),
        (10000, 0, -1, 0, {}),
        (10000, 0, 1.5, 0, {}),
        (10000, 0, 2.0, 0, {}),
        (10000, 0, True, 0, {}),
        (10000, 0, 0, -1, {}),
        (10000, 0, 0, 2.0, {}),
        (10000, math.nan, 0, 0, {}),
        (10000, True, 0, 0, {}),
        (math.inf, 0, 0, 0, {}),
        (10000, 0, 0, 0, {"max_trades_per_day": 0}),
        (10000, 0, 0, 0, {"max_trades_per_day": 2.0}),
        (10000, 0, 0, 0, {"daily_loss_limit_pct": 0}),
        (10000, 0, 0, 0, {"max_open_positions": 0}),
        (10000, 0, 0, 0, {"max_open_positions": 1.0}),
    ],
)
def test_l13_invalid_daily_limit_arguments(
    args: tuple[float, float, Any, Any, dict[str, Any]],
) -> None:
    balance, pnl, trades, open_positions, options = args
    with pytest.raises(ValueError):
        evaluate_daily_limits(
            balance,
            pnl,
            trades,
            open_positions,
            **options,
        )


def _plain_limits_reference(
    balance: float,
    pnl: float,
    trades: int,
    positions: int,
    max_trades: int,
    loss_pct: float,
    max_positions: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if trades >= max_trades:
        if pnl <= -(balance * loss_pct / 100) + 1e-9:
            reasons.extend(
                ["max_trades_reached", "daily_loss_limit_reached"]
            )
        else:
            reasons.append("max_trades_reached")
    elif pnl <= -(balance * loss_pct / 100) + 1e-9:
        reasons.append("daily_loss_limit_reached")
    if positions >= max_positions:
        reasons.append("position_already_open")
    if len(reasons) == 0:
        return {"allowed": True, "reasons": reasons}
    return {"allowed": False, "reasons": reasons}


def test_plain_reference_on_300_seeded_limit_cases() -> None:
    rng = random.Random(4242)
    balances = (500.0, 1000.0, 10000.0, 25000.0)
    loss_pcts = (1.0, 2.0, 3.0)
    for _ in range(300):
        balance = rng.choice(balances)
        loss_pct = rng.choice(loss_pcts)
        limit = balance * loss_pct / 100
        pnl = rng.choice(
            (
                0.0,
                -limit,
                -limit + 0.01,
                -limit - 0.01,
                150.0,
                rng.uniform(-balance * 0.05, balance * 0.05),
            )
        )
        trades = rng.choice((0, 1, 2, 3))
        positions = rng.choice((0, 1, 2))
        max_trades = rng.choice((1, 2, 3))
        max_positions = rng.choice((1, 2))
        expected = _plain_limits_reference(
            balance,
            pnl,
            trades,
            positions,
            max_trades,
            loss_pct,
            max_positions,
        )
        actual = evaluate_daily_limits(
            balance,
            pnl,
            trades,
            positions,
            max_trades_per_day=max_trades,
            daily_loss_limit_pct=loss_pct,
            max_open_positions=max_positions,
        )
        assert actual == expected

