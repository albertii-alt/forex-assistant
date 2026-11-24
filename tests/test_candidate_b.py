from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.data.resample_et import align_htf_to_ltf, resample_ohlcv_et
from src.detectors.trend import htf_trend_state
from src.detectors.zones import ZONE_COLUMNS, detect_zones
from src.indicators.basic import stochastic
from src.strategies.candidate_b import (
    PLAN_COLUMNS,
    find_candidate_b_plans,
)


ET = ZoneInfo("America/New_York")
PIP = 0.0001
PLAN_UTC_COLUMNS = (
    "start_time",
    "time_exit_at",
    "zone_base_time",
    "zone_known_at",
    "touch_time",
    "confirm_time",
)


def make_index(
    size: int = 16,
    start: str = "2024-01-09 14:00",
    unit: str = "ns",
) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=size, freq="15min", tz="UTC").as_unit(unit)


def make_zones(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=ZONE_COLUMNS)
    if frame.empty:
        return pd.DataFrame(
            {
                "direction": pd.Series(dtype="object"),
                "base_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "known_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "zone_low": pd.Series(dtype="float64"),
                "zone_high": pd.Series(dtype="float64"),
                "atr_at_base": pd.Series(dtype="float64"),
                "move_size": pd.Series(dtype="float64"),
                "closed_through_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "expires_at": pd.Series(dtype="datetime64[ns, UTC]"),
            },
            columns=ZONE_COLUMNS,
        )
    for column in ("base_time", "known_at", "closed_through_at", "expires_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    for column in ("zone_low", "zone_high", "atr_at_base", "move_size"):
        frame[column] = frame[column].astype("float64")
    frame["direction"] = frame["direction"].astype("object")
    return frame


def zone(
    index: pd.DatetimeIndex,
    direction: str = "supply",
    base: int = 0,
    known: int = 1,
    low: float = 1.1020,
    high: float = 1.1030,
    closed: int | None = None,
    expires: int | None = None,
) -> dict[str, object]:
    return {
        "direction": direction,
        "base_time": index[base],
        "known_at": index[known],
        "zone_low": low,
        "zone_high": high,
        "atr_at_base": 0.0004,
        "move_size": 0.0012,
        "closed_through_at": pd.NaT if closed is None else index[closed],
        "expires_at": pd.NaT if expires is None else index[expires],
    }


def fs_inputs(
    start: str = "2024-01-09 14:00",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    index = make_index(16, start)
    opens = np.full(16, 1.1010)
    highs = np.full(16, 1.1012)
    highs[4] = 1.1022
    lows = np.full(16, 1.1008)
    closes = opens.copy()
    k_values = np.full(16, 50.0)
    k_values[4] = 85.0
    k_values[5] = 70.0
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )
    k = pd.Series(k_values, index=index, name="k", dtype="float64")
    state = pd.Series("down", index=index, name="state", dtype="object")
    return df, k, state, make_zones([zone(index)])


def fl_inputs(
    start: str = "2024-01-09 14:00",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    index = make_index(16, start)
    opens = np.full(16, 1.0990)
    highs = np.full(16, 1.0992)
    lows = np.full(16, 1.0988)
    lows[4] = 1.0978
    closes = opens.copy()
    k_values = np.full(16, 50.0)
    k_values[4] = 15.0
    k_values[5] = 30.0
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )
    k = pd.Series(k_values, index=index, name="k", dtype="float64")
    state = pd.Series("up", index=index, name="state", dtype="object")
    return (
        df,
        k,
        state,
        make_zones(
            [
                zone(
                    index,
                    direction="demand",
                    low=1.0970,
                    high=1.0980,
                )
            ]
        ),
    )


def expected_plan(
    index: pd.DatetimeIndex,
    *,
    side: str = "short",
    start_pos: int = 6,
    touch_pos: int = 4,
    confirm_pos: int = 5,
    base_pos: int = 0,
    known_pos: int = 1,
    zone_low: float = 1.1020,
    zone_high: float = 1.1030,
    stop: float = 1.1032,
    risk: float = 22.2,
    target_r: float = 2.0,
    min_risk_pips: float = 5.0,
    max_risk_pips: float = 30.0,
    exit_time_et: time = time(15, 45),
) -> pd.DataFrame:
    local = index[start_pos].tz_convert(ET)
    day = local.date()
    exit_time = pd.Timestamp(
        datetime.combine(day, exit_time_et, ET).astimezone(timezone.utc)
    ).as_unit("ns")
    row = {
        "trading_day": pd.Timestamp(day),
        "plan_id": day.isoformat(),
        "side": side,
        "order_type": "market",
        "start_time": index[start_pos],
        "entry_price": np.nan,
        "stop_price": stop,
        "target_r": target_r,
        "target_price": np.nan,
        "valid_candles": 1,
        "time_exit_at": exit_time,
        "min_risk_pips": min_risk_pips,
        "max_risk_pips": max_risk_pips,
        "risk_pips": risk,
        "zone_base_time": index[base_pos],
        "zone_known_at": index[known_pos],
        "zone_low": zone_low,
        "zone_high": zone_high,
        "touch_time": index[touch_pos],
        "confirm_time": index[confirm_pos],
        "htf_state": "down" if side == "short" else "up",
    }
    return expected_frame([row])


def expected_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        result = pd.DataFrame(columns=PLAN_COLUMNS)
        result["trading_day"] = pd.Series(dtype="datetime64[ns]")
        for column in ("plan_id", "side", "order_type", "htf_state"):
            result[column] = pd.Series(dtype="object")
        for column in PLAN_UTC_COLUMNS:
            result[column] = pd.Series(dtype="datetime64[ns, UTC]")
        for column in (
            "entry_price",
            "stop_price",
            "target_r",
            "target_price",
            "min_risk_pips",
            "max_risk_pips",
            "risk_pips",
            "zone_low",
            "zone_high",
        ):
            result[column] = pd.Series(dtype="float64")
        result["valid_candles"] = pd.Series(dtype="int64")
        return result.loc[:, list(PLAN_COLUMNS)]
    result = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    result["trading_day"] = pd.to_datetime(result["trading_day"]).astype(
        "datetime64[ns]"
    )
    for column in PLAN_UTC_COLUMNS:
        result[column] = pd.to_datetime(result[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    for column in (
        "entry_price",
        "stop_price",
        "target_r",
        "target_price",
        "min_risk_pips",
        "max_risk_pips",
        "risk_pips",
        "zone_low",
        "zone_high",
    ):
        result[column] = result[column].astype("float64")
    result["valid_candles"] = result["valid_candles"].astype("int64")
    for column in ("plan_id", "side", "order_type", "htf_state"):
        result[column] = result[column].astype("object")
    return result.loc[:, list(PLAN_COLUMNS)].reset_index(drop=True)


def find_with_inputs(
    df: pd.DataFrame,
    k: pd.Series,
    state: pd.Series,
    zones: pd.DataFrame,
    **kwargs: object,
) -> pd.DataFrame:
    return find_candidate_b_plans(df, zones, k, state, **kwargs)


def assert_plans_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True, rtol=0, atol=1e-9)


def test_v1_short_and_v2_long_hand_computed() -> None:
    df, k, state, zones = fs_inputs()
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index),
    )
    long_df, long_k, long_state, long_zones = fl_inputs()
    expected_long = expected_plan(
        long_df.index,
        side="long",
        zone_low=1.0970,
        zone_high=1.0980,
        stop=1.0968,
        risk=23.2,
    )
    assert_plans_equal(
        find_with_inputs(long_df, long_k, long_state, long_zones),
        expected_long,
    )


@pytest.mark.parametrize(
    ("side", "state_value"),
    [
        ("short", "up"),
        ("short", "none"),
        ("short", np.nan),
        ("long", "down"),
    ],
)
def test_v3_direction_requires_matching_state(side: str, state_value: object) -> None:
    if side == "short":
        df, k, state, zones = fs_inputs()
    else:
        df, k, state, zones = fl_inputs()
    state[:] = state_value
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))


def test_v4_state_is_checked_at_confirmation() -> None:
    df, k, state, zones = fs_inputs()
    state[:] = "up"
    state.iloc[5] = "down"
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index),
    )
    state[:] = "down"
    state.iloc[5] = "up"
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))


def test_v5_touch_boundary_is_inclusive() -> None:
    df, k, state, zones = fs_inputs()
    df.loc[df.index[4], "high"] = 1.1020
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index),
    )
    df.loc[df.index[4], "high"] = 1.10199
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))


def test_v6_zone_is_not_touched_until_after_known_at() -> None:
    df, k, state, zones = fs_inputs()
    df.loc[df.index[1], "high"] = 1.1022
    df.loc[df.index[4], "high"] = 1.1012
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))

    df, k, state, zones = fs_inputs()
    df.loc[df.index[2], "high"] = 1.1022
    df.loc[df.index[4], "high"] = 1.1012
    k.iloc[2] = 85.0
    k.iloc[3] = 70.0
    k.iloc[4:6] = 50.0
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index, start_pos=4, touch_pos=2, confirm_pos=3),
    )


def test_v7_stochastic_crossings_and_window_limit() -> None:
    df, k, state, zones = fs_inputs()
    k.iloc[4] = 80.0
    k.iloc[5] = 79.99
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index),
    )

    df, k, state, zones = fs_inputs()
    k.iloc[4] = 79.99
    k.iloc[5] = 70.0
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))

    df, k, state, zones = fs_inputs()
    k.iloc[4] = 85.0
    k.iloc[5] = 80.0
    k.iloc[6:11] = 85.0
    k.iloc[11:] = 50.0
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))

    for missing_position in (4, 5):
        df, k, state, zones = fs_inputs()
        k.iloc[missing_position] = np.nan
        assert_plans_equal(
            find_with_inputs(df, k, state, zones),
            expected_frame([]),
        )

    df, k, state, zones = fs_inputs()
    k.iloc[4:10] = 85.0
    k.iloc[10] = 70.0
    k.iloc[11:] = 50.0
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index, start_pos=11, touch_pos=4, confirm_pos=10),
    )

    df, k, state, zones = fs_inputs()
    k.iloc[4:11] = 85.0
    k.iloc[11] = 70.0
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))

    long_df, long_k, long_state, long_zones = fl_inputs()
    long_k.iloc[4] = 20.0
    long_k.iloc[5] = 20.01
    assert_plans_equal(
        find_with_inputs(long_df, long_k, long_state, long_zones),
        expected_plan(
            long_df.index,
            side="long",
            zone_low=1.0970,
            zone_high=1.0980,
            stop=1.0968,
            risk=23.2,
        ),
    )
    long_df, long_k, long_state, long_zones = fl_inputs()
    long_k.iloc[4] = 20.01
    long_k.iloc[5] = 30.0
    assert_plans_equal(
        find_with_inputs(long_df, long_k, long_state, long_zones),
        expected_frame([]),
    )


def test_v8_entry_window_uses_et_and_inclusive_deadline() -> None:
    df, k, state, zones = fs_inputs(start="2024-01-09 13:00")
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index),
    )

    df, k, state, zones = fs_inputs(start="2024-01-09 13:00")
    df.loc[df.index[3], "high"] = 1.1022
    df.loc[df.index[4], "high"] = 1.1012
    k.iloc[3] = 85.0
    k.iloc[4] = 70.0
    k.iloc[5] = 50.0
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))

    df, k, state, zones = fs_inputs()
    df.loc[df.index[4], "high"] = 1.1012
    df.loc[df.index[7], "high"] = 1.1022
    k.iloc[4:6] = 50.0
    k.iloc[7:11] = 85.0
    k.iloc[11] = 70.0
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))


def test_custom_et_times_control_entry_window_and_exit_timestamp() -> None:
    df, k, state, zones = fs_inputs(start="2024-01-09 13:00")
    expected = expected_plan(df.index, exit_time_et=time(16, 0))
    assert expected.loc[0, "time_exit_at"] == pd.Timestamp(
        "2024-01-09 21:00", tz="UTC"
    )
    assert_plans_equal(
        find_with_inputs(
            df,
            k,
            state,
            zones,
            window_start_et=time(9, 15),
            entry_deadline_et=time(9, 30),
            time_exit_et=time(16, 0),
        ),
        expected,
    )
    assert_plans_equal(
        find_with_inputs(
            df,
            k,
            state,
            zones,
            window_start_et=time(9, 31),
            entry_deadline_et=time(10, 0),
            time_exit_et=time(16, 0),
        ),
        expected_frame([]),
    )


def test_v9_zone_expiry_and_close_through_boundaries() -> None:
    for position in (5, 4):
        df, k, state, zones = fs_inputs()
        zones.loc[0, "closed_through_at"] = df.index[position]
        assert_plans_equal(
            find_with_inputs(df, k, state, zones),
            expected_frame([]),
        )
    for position in (6,):
        df, k, state, zones = fs_inputs()
        zones.loc[0, "closed_through_at"] = df.index[position]
        assert_plans_equal(
            find_with_inputs(df, k, state, zones),
            expected_plan(df.index),
        )

    for position in (6,):
        df, k, state, zones = fs_inputs()
        zones.loc[0, "expires_at"] = df.index[position]
        assert_plans_equal(
            find_with_inputs(df, k, state, zones),
            expected_plan(df.index),
        )
    for position in (5, 4):
        df, k, state, zones = fs_inputs()
        zones.loc[0, "expires_at"] = df.index[position]
        assert_plans_equal(
            find_with_inputs(df, k, state, zones),
            expected_frame([]),
        )


def test_v10_only_the_first_touch_is_considered() -> None:
    df, k, state, zones = fs_inputs()
    df.loc[df.index[2], "high"] = 1.1022
    df.loc[df.index[4], "high"] = 1.1012
    df.loc[df.index[9], "high"] = 1.1022
    k.iloc[4:6] = 50.0
    k.iloc[9] = 85.0
    k.iloc[10] = 70.0
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))


def test_v11_only_the_first_stochastic_crossing_is_considered() -> None:
    df, k, state, zones = fs_inputs()
    k.iloc[4] = 85.0
    k.iloc[5] = 70.0
    k.iloc[6] = 85.0
    k.iloc[7] = 70.0
    state.iloc[5] = "none"
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))

    state[:] = "down"
    zones.loc[0, "closed_through_at"] = df.index[5]
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))


def test_v12_cost_adjusted_risk_filter() -> None:
    df, k, state, zones = fs_inputs()
    assert_plans_equal(
        find_with_inputs(df, k, state, zones, max_risk_pips=22.2),
        expected_plan(df.index, max_risk_pips=22.2),
    )
    assert_plans_equal(
        find_with_inputs(df, k, state, zones, max_risk_pips=22.1),
        expected_frame([]),
    )
    assert_plans_equal(
        find_with_inputs(df, k, state, zones, min_risk_pips=22.2),
        expected_plan(df.index, min_risk_pips=22.2),
    )
    assert_plans_equal(
        find_with_inputs(df, k, state, zones, min_risk_pips=22.3),
        expected_frame([]),
    )

    long_df, long_k, long_state, long_zones = fl_inputs()
    long_expected = expected_plan(
        long_df.index,
        side="long",
        zone_low=1.0970,
        zone_high=1.0980,
        stop=1.0968,
        risk=23.2,
        max_risk_pips=23.2,
    )
    assert_plans_equal(
        find_with_inputs(
            long_df, long_k, long_state, long_zones, max_risk_pips=23.2
        ),
        long_expected,
    )
    assert_plans_equal(
        find_with_inputs(
            long_df, long_k, long_state, long_zones, max_risk_pips=23.1
        ),
        expected_frame([]),
    )

    df, k, state, zones = fs_inputs()
    df.loc[df.index[6], "open"] = 1.1040
    df.loc[df.index[6], "high"] = 1.1045
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))
    assert_plans_equal(
        find_with_inputs(df, k, state, zones, stop_buffer_pips=9.8),
        expected_frame([]),
    )

    df, k, state, zones = fs_inputs()
    assert_plans_equal(
        find_with_inputs(df, k, state, zones, stop_buffer_pips=9.8),
        expected_plan(df.index, stop=1.10398, risk=30.0),
    )
    assert_plans_equal(
        find_with_inputs(df, k, state, zones, stop_buffer_pips=9.9),
        expected_frame([]),
    )


def test_v13_one_plan_per_day_selection_ties_and_dates() -> None:
    df, k, state, _ = fs_inputs()
    df.loc[df.index[3], "high"] = 1.1022
    df.loc[df.index[5], "high"] = 1.1035
    k[:] = 50.0
    k.iloc[3:5] = [85.0, 70.0]
    k.iloc[6:8] = [85.0, 70.0]
    early = zone(df.index, base=0, known=1, low=1.1020, high=1.1030)
    late = zone(df.index, base=1, known=2, low=1.1033, high=1.1037)
    zones = make_zones([early, late])
    assert_plans_equal(
        find_with_inputs(df, k, state, zones),
        expected_plan(df.index, start_pos=5, touch_pos=3, confirm_pos=4),
    )

    df, k, state, _ = fs_inputs()
    z_a = zone(df.index, base=0, known=1, low=1.1020, high=1.1030)
    z_c = zone(df.index, base=0, known=2, low=1.1019, high=1.1027)
    z_b = zone(df.index, base=1, known=2, low=1.1018, high=1.1026)
    expected_tie = expected_plan(
        df.index,
        base_pos=1,
        known_pos=2,
        zone_low=1.1018,
        zone_high=1.1026,
        stop=1.1028,
        risk=18.2,
    )
    for ordered in ([z_b, z_a, z_c], [z_c, z_a, z_b]):
        assert_plans_equal(
            find_with_inputs(df, k, state, make_zones(ordered)),
            expected_tie,
        )

    first = fs_inputs()
    second = fs_inputs(start="2024-01-10 14:00")
    two_df = pd.concat([first[0], second[0]])
    two_k = pd.concat([first[1], second[1]])
    two_state = pd.concat([first[2], second[2]])
    two_zones = make_zones(
        [
            zone(first[0].index, base=0, known=1),
            zone(second[0].index, base=0, known=1),
        ]
    )
    expected_two = pd.concat(
        [
            expected_plan(first[0].index),
            expected_plan(second[0].index),
        ],
        ignore_index=True,
    )
    assert_plans_equal(
        find_with_inputs(two_df, two_k, two_state, two_zones),
        expected_two,
    )


def test_v14_daylight_saving_time_exit_conversion() -> None:
    df, k, state, zones = fs_inputs(start="2024-07-09 13:00")
    expected = expected_plan(df.index)
    assert expected.loc[0, "time_exit_at"] == pd.Timestamp("2024-07-09 19:45", tz="UTC")
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected)


def test_v15_plan_is_directly_fillable_by_backtest_engine() -> None:
    df, k, state, zones = fs_inputs()
    plans = find_with_inputs(df, k, state, zones)
    result = run_backtest(df, plans)
    assert len(result) == 1
    assert result.loc[0, "status"] == "filled"
    assert result.loc[0, "fill_time"] == df.index[6]
    assert result.loc[0, "entry_price"] == pytest.approx(1.10098, abs=1e-9)
    assert result.loc[0, "risk_pips"] == pytest.approx(plans.loc[0, "risk_pips"])
    assert result.loc[0, "exit_reason"] == "data_end"
    assert result.loc[0, "bars_held"] == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"swing_left": 0},
        {"swing_right": 0},
        {"confirm_candles": 0},
        {"stop_buffer_pips": -1},
        {"min_risk_pips": -1},
        {"min_risk_pips": 31, "max_risk_pips": 30},
        {"max_risk_pips": np.inf},
        {"target_r": 0},
        {"overbought": 100},
        {"oversold": 80},
        {"spread_pips": -1},
        {"slippage_pips": -1},
        {"pip_size": 0},
        {"pip_size": np.nan},
        {"window_start_et": "09:30"},
        {"window_start_et": time(12, 0), "entry_deadline_et": time(11, 45)},
        {"entry_deadline_et": time(15, 46), "time_exit_et": time(15, 45)},
    ],
)
def test_v16_invalid_parameters_raise_value_error(
    kwargs: dict[str, object],
) -> None:
    df, k, state, zones = fs_inputs()
    with pytest.raises(ValueError):
        find_with_inputs(df, k, state, zones, **kwargs)


def test_v16_validation_empty_frames_dtypes_and_no_mutation() -> None:
    df, k, state, zones = fs_inputs()
    originals = (df.copy(deep=True), k.copy(deep=True), state.copy(deep=True), zones.copy(deep=True))
    result = find_with_inputs(df, k, state, zones)
    pd.testing.assert_frame_equal(df, originals[0])
    pd.testing.assert_series_equal(k, originals[1])
    pd.testing.assert_series_equal(state, originals[2])
    pd.testing.assert_frame_equal(zones, originals[3])
    assert result.columns.tolist() == list(PLAN_COLUMNS)
    assert result["trading_day"].dtype == np.dtype("datetime64[ns]")
    assert result["valid_candles"].dtype == np.dtype("int64")
    assert result["start_time"].dtype == "datetime64[ns, UTC]"
    assert result["entry_price"].dtype == np.dtype("float64")

    empty_index = make_index(0)
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close"],
        index=empty_index,
        dtype="float64",
    )
    empty_series = pd.Series(dtype="float64", index=empty_index)
    empty_state = pd.Series(dtype="object", index=empty_index)
    assert_plans_equal(
        find_candidate_b_plans(
            empty,
            make_zones([]),
            empty_series,
            empty_state,
        ),
        expected_frame([]),
    )
    assert_plans_equal(find_candidate_b_plans(empty), expected_frame([]))
    assert_plans_equal(
        find_candidate_b_plans(
            df.iloc[:0],
            make_zones([]),
            k.iloc[:0],
            state.iloc[:0],
        ),
        expected_frame([]),
    )
    no_volume = df.copy(deep=True)
    no_volume_snapshot = no_volume.copy(deep=True)
    find_candidate_b_plans(no_volume)
    pd.testing.assert_frame_equal(no_volume, no_volume_snapshot)

    with pytest.raises(ValueError):
        find_candidate_b_plans(df.drop(columns=["close"]), zones, k, state)
    with pytest.raises(ValueError):
        find_candidate_b_plans(df.set_axis(df.index.tz_localize(None)), zones, k, state)
    with pytest.raises(ValueError):
        find_candidate_b_plans(df.iloc[::-1], zones, k, state)
    with pytest.raises(ValueError):
        find_candidate_b_plans(df, zones.drop(columns=["direction"]), k, state)
    with pytest.raises(ValueError):
        find_candidate_b_plans(df, zones, k.iloc[:-1], state)
    with pytest.raises(ValueError):
        find_candidate_b_plans(df, zones, k, state.iloc[:-1])
    invalid_state = state.copy()
    invalid_state.iloc[5] = "sideways"
    with pytest.raises(ValueError):
        find_candidate_b_plans(df, zones, k, invalid_state)


def test_v16_zone_timestamps_not_in_df_are_skipped() -> None:
    df, k, state, zones = fs_inputs()
    zones.loc[0, "known_at"] = pd.Timestamp("2024-01-08", tz="UTC")
    assert_plans_equal(find_with_inputs(df, k, state, zones), expected_frame([]))


def normalize_plan_units(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in PLAN_UTC_COLUMNS:
        result[column] = pd.Series(
            pd.DatetimeIndex(result[column]).as_unit("ns"),
            index=result.index,
        )
    return result


def test_v17_microsecond_timestamp_precision() -> None:
    for maker in (fs_inputs, fl_inputs):
        df, k, state, zones = maker()
        expected = find_with_inputs(df, k, state, zones)
        us_df = df.copy()
        us_df.index = us_df.index.as_unit("us")
        us_k = k.copy()
        us_k.index = us_k.index.as_unit("us")
        us_state = state.copy()
        us_state.index = us_state.index.as_unit("us")
        us_zones = zones.copy()
        for column in ("base_time", "known_at", "closed_through_at", "expires_at"):
            us_zones[column] = pd.Series(
                pd.DatetimeIndex(us_zones[column]).as_unit("us"),
                index=us_zones.index,
            )
        actual = find_with_inputs(us_df, us_k, us_state, us_zones)
        assert_plans_equal(normalize_plan_units(actual), expected)


def _reference_timestamp_position(
    value: object,
    positions: dict[int, int],
) -> int | None:
    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value).tz_convert("UTC").as_unit("ns")
    return positions.get(timestamp.value)


def reference_candidate_b(
    df: pd.DataFrame,
    zones: pd.DataFrame,
    k: pd.Series,
    state: pd.Series,
    funnel: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Literal test oracle for the fixed Candidate B plan rules."""
    index = df.index.as_unit("ns")
    local = index.tz_convert(ET)
    positions = {int(value): position for position, value in enumerate(index.asi8)}
    op = [float(value) for value in df["open"]]
    hi = [float(value) for value in df["high"]]
    lo = [float(value) for value in df["low"]]
    kvals = [float(value) for value in k]
    states = [value for value in state]
    counts = {
        stage: {"all": 0, "long": 0, "short": 0}
        for stage in (
            "zones",
            "with_touch",
            "with_crossing",
            "not_closed_through",
            "state_ok",
            "entry_in_window",
            "risk_ok",
            "plans",
        )
    }

    def count(stage: str, side: str) -> None:
        counts[stage]["all"] += 1
        counts[stage][side] += 1

    candidates: dict[object, tuple[tuple[int, int, int], dict[str, object]]] = {}
    for row in zones.itertuples(index=False):
        direction = row.direction
        side = "short" if direction == "supply" else "long"
        count("zones", side)
        base = _reference_timestamp_position(row.base_time, positions)
        known = _reference_timestamp_position(row.known_at, positions)
        closed = _reference_timestamp_position(row.closed_through_at, positions)
        expiry = _reference_timestamp_position(row.expires_at, positions)
        if base is None or known is None:
            continue
        if pd.isna(row.expires_at):
            expiry = len(index)
        if not pd.isna(row.expires_at) and expiry is None:
            continue
        if not pd.isna(row.closed_through_at) and closed is None:
            continue
        assert expiry is not None

        touch = None
        for candle in range(known + 1, min(expiry, len(index))):
            reached = (
                hi[candle] >= float(row.zone_low) - 1e-9
                if direction == "supply"
                else lo[candle] <= float(row.zone_high) + 1e-9
            )
            if reached:
                touch = candle
                break
        if touch is None:
            continue
        count("with_touch", side)

        crossing = None
        last_candle = min(touch + 6, expiry - 1, len(index) - 1)
        for candle in range(touch, last_candle + 1):
            if candle < 1:
                continue
            before, after = kvals[candle - 1], kvals[candle]
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            reversed_inside = (
                before >= 80.0 and after < 80.0
                if direction == "supply"
                else before <= 20.0 and after > 20.0
            )
            if reversed_inside:
                crossing = candle
                break
        if crossing is None:
            continue
        count("with_crossing", side)
        if closed is not None and closed <= crossing:
            continue
        count("not_closed_through", side)

        expected_state = "down" if direction == "supply" else "up"
        if states[crossing] != expected_state:
            continue
        count("state_ok", side)

        entry = crossing + 1
        if entry >= len(index):
            continue
        local_time = local[entry].time()
        if not time(9, 30) <= local_time <= time(11, 45):
            continue
        count("entry_in_window", side)

        fill = op[entry] - 0.2 * PIP if side == "short" else op[entry] + 1.2 * PIP
        stop = (
            float(row.zone_high) + 2.0 * PIP
            if side == "short"
            else float(row.zone_low) - 2.0 * PIP
        )
        risk = stop - fill if side == "short" else fill - stop
        risk_pips = risk / PIP
        if (
            risk <= 0
            or risk_pips < 5.0 - 1e-9
            or risk_pips > 30.0 + 1e-9
        ):
            continue
        count("risk_ok", side)

        day = local[entry].date()
        entry_time = index[entry]
        exit_time = pd.Timestamp(
            datetime.combine(day, time(15, 45), ET).astimezone(timezone.utc)
        ).as_unit("ns")
        key = (entry, -known, -base)
        plan = {
            "trading_day": pd.Timestamp(day),
            "plan_id": day.isoformat(),
            "side": side,
            "order_type": "market",
            "start_time": entry_time,
            "entry_price": np.nan,
            "stop_price": stop,
            "target_r": 2.0,
            "target_price": np.nan,
            "valid_candles": 1,
            "time_exit_at": exit_time,
            "min_risk_pips": 5.0,
            "max_risk_pips": 30.0,
            "risk_pips": risk_pips,
            "zone_base_time": index[base],
            "zone_known_at": index[known],
            "zone_low": float(row.zone_low),
            "zone_high": float(row.zone_high),
            "touch_time": index[touch],
            "confirm_time": index[crossing],
            "htf_state": expected_state,
        }
        if day not in candidates or key < candidates[day][0]:
            candidates[day] = (key, plan)

    plans = [
        item[1]
        for item in sorted(candidates.values(), key=lambda candidate: candidate[0][0])
    ]
    for plan in plans:
        count("plans", str(plan["side"]))
    if funnel is not None:
        funnel.clear()
        funnel.update(counts)
    return expected_frame(plans)


def synthetic_series(
    seed: int,
    start: str,
    end: str,
    sigma: float = 0.0008,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    full_index = pd.date_range(start, end, freq="15min", tz="UTC")
    local = full_index.tz_convert(ET)
    minutes = local.hour * 60 + local.minute
    closed = (
        ((local.day_of_week == 4) & (minutes >= 17 * 60))
        | (local.day_of_week == 5)
        | ((local.day_of_week == 6) & (minutes < 17 * 60))
    )
    index = full_index[~np.asarray(closed)]
    changes = rng.normal(0.0, sigma, len(index))
    close = 1.1 + np.cumsum(changes)
    open_price = np.r_[1.1, close[:-1]]
    wick_high = rng.uniform(0.00005, 0.0004, len(index))
    wick_low = rng.uniform(0.00005, 0.0004, len(index))
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + wick_high,
            "low": np.minimum(open_price, close) - wick_low,
            "close": close,
            "volume": np.ones(len(index)),
        },
        index=index,
    )


def computed_inputs(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    zones = detect_zones(df)
    k = stochastic(df)["k"]
    htf = resample_ohlcv_et(df, "4h")
    htf["state"] = htf_trend_state(htf)["state"]
    state = align_htf_to_ltf(df.index, htf, columns=["state"])["state"]
    return zones, k, state


@pytest.mark.parametrize(
    ("seed", "start", "end"),
    [
        (11, "2023-01-02", "2023-06-30 23:45"),
        (22, "2023-01-02", "2023-06-30 23:45"),
        (5, "2023-09-01", "2023-12-29 23:45"),
        (17, "2023-09-01", "2023-12-29 23:45"),
    ],
)
def test_seeded_reference_engine_and_no_lookahead(
    seed: int,
    start: str,
    end: str,
) -> None:
    df = synthetic_series(seed, start, end)
    zones, k, state = computed_inputs(df)
    funnel: dict[str, dict[str, int]] = {}
    expected = reference_candidate_b(df, zones, k, state, funnel)
    injected = find_candidate_b_plans(df, zones, k, state)
    internal = find_candidate_b_plans(df)
    assert_plans_equal(injected, expected)
    assert_plans_equal(internal, expected)
    assert len(expected) >= 8
    print(f"seed={seed} plans={len(expected)}")
    assert funnel["plans"]["all"] == len(expected)

    trades = run_backtest(df, expected)
    assert (trades["status"] == "filled").all()
    np.testing.assert_allclose(
        trades["risk_pips"].to_numpy(),
        expected["risk_pips"].to_numpy(),
        rtol=1e-9,
    )

    local = df.index.tz_convert(ET)
    eligible = np.flatnonzero(
        (local.hour >= 10)
        & ((local.hour < 11) | ((local.hour == 11) & (local.minute <= 30)))
    )
    if len(eligible) < 6:
        raise AssertionError("synthetic date span did not provide six no-lookahead cutoffs")
    cutoff_positions = np.unique(
        eligible[np.linspace(0, len(eligible) - 1, 6, dtype=int)]
    )
    assert len(cutoff_positions) == 6
    for cutoff in cutoff_positions:
        truncated = df.iloc[: cutoff + 1]
        truncated_plans = find_candidate_b_plans(truncated)
        restricted_full = expected.loc[
            expected["start_time"] <= truncated.index[-1]
        ].reset_index(drop=True)
        assert_plans_equal(truncated_plans, restricted_full)


def test_seeded_inputs_keep_requested_microsecond_index() -> None:
    df = synthetic_series(55, "2023-09-01", "2023-09-14 23:45")
    df.index = df.index.as_unit("us")
    zones, k, state = computed_inputs(df)
    actual = find_candidate_b_plans(df, zones, k, state)
    ns_df = df.copy()
    ns_df.index = ns_df.index.as_unit("ns")
    ns_zones, ns_k, ns_state = computed_inputs(ns_df)
    expected = find_candidate_b_plans(ns_df, ns_zones, ns_k, ns_state)
    assert_plans_equal(normalize_plan_units(actual), expected)
