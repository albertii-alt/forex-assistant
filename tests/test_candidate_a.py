from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.detectors.fvg import OUTPUT_COLUMNS as FVG_COLUMNS
from src.detectors.fvg import detect_fvgs
from src.detectors.session_levels import detect_session_levels
from src.detectors.structure import OUTPUT_COLUMNS as BREAK_COLUMNS
from src.detectors.structure import detect_structure_breaks
from src.strategies.candidate_a import (
    PLAN_COLUMNS,
    find_candidate_a_plans,
)


ET = ZoneInfo("America/New_York")
PIP = 0.0001
Q = (1.1010, 1.1020, 1.1005, 1.1010)
S = (1.1006, 1.1008, 1.0996, 1.1004)
A1 = (1.1004, 1.1020, 1.1002, 1.1018)
A2 = (1.1018, 1.1030, 1.1010, 1.1028)
T = (1.1028, 1.1035, 1.1020, 1.1030)
Q_SHORT = (1.0990, 1.0995, 1.0980, 1.0990)
S_SHORT = (1.0994, 1.1004, 1.0992, 1.0996)
A1_SHORT = (1.0996, 1.0998, 1.0980, 1.0982)
A2_SHORT = (1.0982, 1.0990, 1.0970, 1.0972)
T_SHORT = (1.0972, 1.0980, 1.0965, 1.0970)


def utc_index(
    count: int = 16,
    start: str = "2024-01-09 14:00",
) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=count, freq="15min", tz="UTC").as_unit(
        "ns"
    )


def make_df(
    candles: list[tuple[float, float, float, float]],
    start: str = "2024-01-09 14:00",
) -> pd.DataFrame:
    return pd.DataFrame(
        candles,
        columns=["open", "high", "low", "close"],
        index=utc_index(len(candles), start),
    )


def standard_levels(
    index: pd.DatetimeIndex,
    *,
    short: bool = False,
    trading_day: str = "2024-01-09",
    overnight_known_at: pd.Timestamp | None = None,
    prev_day_known_at: pd.Timestamp | None = None,
    overnight_low: float | None = None,
    overnight_high: float | None = None,
    prev_day_low: float | None = None,
    prev_day_high: float | None = None,
) -> pd.DataFrame:
    if short:
        values = {
            "overnight_low": 1.0900,
            "overnight_high": 1.1000,
            "prev_day_low": 1.0800,
            "prev_day_high": 1.1100,
        }
    else:
        values = {
            "overnight_low": 1.1000,
            "overnight_high": 1.1100,
            "prev_day_low": 1.0900,
            "prev_day_high": 1.1200,
        }
    overrides = {
        "overnight_low": overnight_low,
        "overnight_high": overnight_high,
        "prev_day_low": prev_day_low,
        "prev_day_high": prev_day_high,
    }
    for column, value in overrides.items():
        if value is not None:
            values[column] = value
    overnight_known = overnight_known_at or index[2]
    previous_known = prev_day_known_at or pd.Timestamp(
        "2024-01-08 22:00",
        tz="UTC",
    )
    row = {
        "trading_day": pd.Timestamp(trading_day),
        "prev_day_high": values["prev_day_high"],
        "prev_day_high_set_at": previous_known,
        "prev_day_low": values["prev_day_low"],
        "prev_day_low_set_at": previous_known,
        "prev_day_known_at": previous_known,
        "overnight_high": values["overnight_high"],
        "overnight_high_set_at": overnight_known,
        "overnight_low": values["overnight_low"],
        "overnight_low_set_at": overnight_known,
        "overnight_known_at": overnight_known,
    }
    frame = pd.DataFrame([row])
    frame["trading_day"] = pd.to_datetime(frame["trading_day"]).astype(
        "datetime64[ns]"
    )
    for column in (
        "prev_day_high_set_at",
        "prev_day_low_set_at",
        "prev_day_known_at",
        "overnight_high_set_at",
        "overnight_low_set_at",
        "overnight_known_at",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def make_breaks(
    index: pd.DatetimeIndex,
    entries: list[tuple[str, int]],
) -> pd.DataFrame:
    rows = []
    for direction, position in entries:
        rows.append(
            {
                "direction": direction,
                "level": 1.1010 if direction == "bullish" else 1.0990,
                "swing_time": index[max(0, position - 2)],
                "swing_confirmed_at": index[max(0, position - 1)],
                "broke_at": index[position],
                "kind": "first",
            }
        )
    frame = pd.DataFrame(rows, columns=BREAK_COLUMNS)
    for column in ("swing_time", "swing_confirmed_at", "broke_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def make_fvgs(
    index: pd.DatetimeIndex,
    entries: list[tuple[str, int, float, float]],
) -> pd.DataFrame:
    rows = []
    for direction, position, zone_low, zone_high in entries:
        rows.append(
            {
                "direction": direction,
                "c1_time": index[position - 2],
                "formed_at": index[position],
                "zone_low": zone_low,
                "zone_high": zone_high,
                "size": zone_high - zone_low,
                "spans_gap": False,
                "touched_at": pd.NaT,
                "closed_through_at": pd.NaT,
            }
        )
    frame = pd.DataFrame(rows, columns=FVG_COLUMNS)
    for column in ("c1_time", "formed_at", "touched_at", "closed_through_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def expected_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        result = pd.DataFrame(columns=PLAN_COLUMNS)
        result["trading_day"] = pd.Series(dtype="datetime64[ns]")
        for column in (
            "start_time",
            "time_exit_at",
            "sweep_time",
            "break_time",
            "fvg_formed_at",
        ):
            result[column] = pd.Series(dtype="datetime64[ns, UTC]")
        for column in ("plan_id", "side", "order_type"):
            result[column] = pd.Series(dtype="object")
        result["valid_candles"] = pd.Series(dtype="int64")
        for column in (
            "entry_price",
            "stop_price",
            "target_r",
            "target_price",
            "min_risk_pips",
            "max_risk_pips",
            "risk_pips",
            "fvg_zone_low",
            "fvg_zone_high",
        ):
            result[column] = pd.Series(dtype="float64")
        return result.loc[:, list(PLAN_COLUMNS)]

    result = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    result["trading_day"] = pd.to_datetime(result["trading_day"]).astype(
        "datetime64[ns]"
    )
    for column in (
        "start_time",
        "time_exit_at",
        "sweep_time",
        "break_time",
        "fvg_formed_at",
    ):
        result[column] = pd.Series(
            pd.DatetimeIndex(pd.to_datetime(result[column], utc=True)).as_unit("ns"),
            index=result.index,
        )
    result["valid_candles"] = result["valid_candles"].astype("int64")
    for column in (
        "entry_price",
        "stop_price",
        "target_r",
        "target_price",
        "min_risk_pips",
        "max_risk_pips",
        "risk_pips",
        "fvg_zone_low",
        "fvg_zone_high",
    ):
        result[column] = result[column].astype("float64")
    for column in ("plan_id", "side", "order_type"):
        result[column] = result[column].astype("object")
    return result.loc[:, list(PLAN_COLUMNS)]


def expected_long_plan(
    index: pd.DatetimeIndex,
    *,
    sweep_pos: int = 4,
    break_pos: int = 6,
    entry: float = 1.1010,
    stop: float = 1.0995,
    risk: float = 15.0,
    zone_low: float = 1.1008,
    zone_high: float = 1.1010,
    formed_pos: int = 6,
    time_exit: pd.Timestamp | None = None,
    day: str = "2024-01-09",
) -> pd.DataFrame:
    return expected_frame(
        [
            {
                "trading_day": pd.Timestamp(day),
                "plan_id": day,
                "side": "long",
                "order_type": "limit",
                "start_time": index[break_pos] + pd.Timedelta(minutes=15),
                "entry_price": entry,
                "stop_price": stop,
                "target_r": 2.0,
                "target_price": np.nan,
                "valid_candles": 8,
                "time_exit_at": time_exit or index[11],
                "min_risk_pips": 5.0,
                "max_risk_pips": 25.0,
                "risk_pips": risk,
                "sweep_time": index[sweep_pos],
                "break_time": index[break_pos],
                "fvg_formed_at": index[formed_pos],
                "fvg_zone_low": zone_low,
                "fvg_zone_high": zone_high,
            }
        ]
    )


def h1_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = make_df([Q, Q, Q, Q, S, A1, A2, T] + [T] * 8)
    return (
        df,
        standard_levels(df.index),
        make_breaks(df.index, [("bullish", 6)]),
        make_fvgs(df.index, [("bullish", 6, 1.1008, 1.1010)]),
    )


def assert_plan(
    result: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    pd.testing.assert_frame_equal(result, expected, check_dtype=True)


def test_h1_hand_computed_long_plan() -> None:
    df, levels, breaks, fvgs = h1_inputs()

    result = find_candidate_a_plans(df, levels, breaks, fvgs)

    assert_plan(result, expected_long_plan(df.index))


def test_h2_hand_computed_short_plan() -> None:
    candles = [Q_SHORT] * 4 + [S_SHORT, A1_SHORT, A2_SHORT, T_SHORT] + [T_SHORT] * 8
    df = make_df(candles)
    levels = standard_levels(df.index, short=True)
    breaks = make_breaks(df.index, [("bearish", 6)])
    fvgs = make_fvgs(df.index, [("bearish", 6, 1.0990, 1.0992)])
    expected = expected_frame(
        [
            {
                "trading_day": pd.Timestamp("2024-01-09"),
                "plan_id": "2024-01-09",
                "side": "short",
                "order_type": "limit",
                "start_time": df.index[7],
                "entry_price": 1.0990,
                "stop_price": 1.1005,
                "target_r": 2.0,
                "target_price": np.nan,
                "valid_candles": 8,
                "time_exit_at": df.index[11],
                "min_risk_pips": 5.0,
                "max_risk_pips": 25.0,
                "risk_pips": 15.0,
                "sweep_time": df.index[4],
                "break_time": df.index[6],
                "fvg_formed_at": df.index[6],
                "fvg_zone_low": 1.0990,
                "fvg_zone_high": 1.0992,
            }
        ]
    )

    assert_plan(find_candidate_a_plans(df, levels, breaks, fvgs), expected)


def test_h3a_sweep_before_window_is_ignored() -> None:
    df = make_df([Q, S, A1, A2] + [T] * 12)
    levels = standard_levels(
        df.index,
        overnight_known_at=df.index[0],
    )
    breaks = make_breaks(df.index, [("bullish", 3)])
    fvgs = make_fvgs(df.index, [("bullish", 3, 1.1008, 1.1010)])

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_frame([]),
    )


def test_h3b_close_equal_to_level_is_a_sweep() -> None:
    df, levels, breaks, fvgs = h1_inputs()
    df.iloc[4] = (1.1006, 1.1008, 1.0996, 1.1000)

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_long_plan(df.index),
    )


def test_h3c_low_equal_to_level_is_not_a_sweep() -> None:
    df, levels, breaks, fvgs = h1_inputs()
    df.iloc[4] = (1.1006, 1.1008, 1.1000, 1.1004)

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_frame([]),
    )


def test_h4a_deadline_boundary_break_is_accepted() -> None:
    df = make_df([Q] * 7 + [S, A1, A2] + [T] * 6)
    levels = standard_levels(df.index)
    breaks = make_breaks(df.index, [("bullish", 9)])
    fvgs = make_fvgs(df.index, [("bullish", 9, 1.1008, 1.1010)])

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_long_plan(
            df.index,
            sweep_pos=7,
            break_pos=9,
            formed_pos=9,
            time_exit=df.index[11],
        ),
    )


def test_h4b_break_closing_after_deadline_is_rejected() -> None:
    df = make_df([Q] * 8 + [S, A1, A2] + [T] * 5)
    levels = standard_levels(df.index)
    breaks = make_breaks(df.index, [("bullish", 10)])
    fvgs = make_fvgs(df.index, [("bullish", 10, 1.1008, 1.1010)])

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_frame([]),
    )


def test_h6_latest_qualifying_fvg_after_sweep_and_by_break_is_selected() -> None:
    df, levels, breaks, _ = h1_inputs()
    fvgs = make_fvgs(
        df.index,
        [
            ("bullish", 5, 1.1000, 1.1004),
            ("bullish", 6, 1.1008, 1.1010),
            ("bullish", 7, 1.1012, 1.1020),
            ("bearish", 6, 1.1050, 1.1060),
            ("bullish", 4, 1.1011, 1.1015),
        ],
    )

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_long_plan(df.index),
    )


def test_fvg_formed_on_sweep_candle_is_not_eligible() -> None:
    df, levels, breaks, _ = h1_inputs()
    fvgs = make_fvgs(df.index, [("bullish", 4, 1.1008, 1.1010)])

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_frame([]),
    )


def test_h7_fvg_below_minimum_size_is_rejected() -> None:
    df, levels, breaks, _ = h1_inputs()
    fvgs = make_fvgs(df.index, [("bullish", 6, 1.10080, 1.10099)])

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_frame([]),
    )


@pytest.mark.parametrize(
    ("zone_high", "expected_risk", "accepted"),
    [
        (1.1000, 5.0, True),
        (1.1020, 25.0, True),
        (1.09998, 4.8, False),
        (1.10201, 25.1, False),
        (1.0990, None, False),
    ],
)
def test_h8_risk_boundaries_and_wrong_side(
    zone_high: float,
    expected_risk: float | None,
    accepted: bool,
) -> None:
    df, levels, breaks, _ = h1_inputs()
    fvgs = make_fvgs(
        df.index,
        [("bullish", 6, zone_high - 0.0002, zone_high)],
    )

    result = find_candidate_a_plans(df, levels, breaks, fvgs)

    if accepted:
        expected = expected_long_plan(
            df.index,
            entry=zone_high,
            risk=float(expected_risk),
            zone_low=zone_high - 0.0002,
            zone_high=zone_high,
        )
        assert_plan(result, expected)
    else:
        assert_plan(result, expected_frame([]))


def h9_inputs(first_zone: tuple[float, float]) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    df = make_df([Q, Q, Q, S, A1, A2, S, A1, A2] + [T] * 7)
    levels = standard_levels(df.index)
    breaks = make_breaks(df.index, [("bullish", 5), ("bullish", 8)])
    fvgs = make_fvgs(
        df.index,
        [
            ("bullish", 5, first_zone[0], first_zone[1]),
            ("bullish", 8, 1.1008, 1.1010),
        ],
    )
    return df, levels, breaks, fvgs


def test_h9a_first_invalid_setup_does_not_hide_second_valid_setup() -> None:
    df, levels, breaks, fvgs = h9_inputs((1.09978, 1.09998))

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_long_plan(
            df.index,
            sweep_pos=6,
            break_pos=8,
            formed_pos=8,
        ),
    )


def test_h9b_earliest_valid_plan_wins() -> None:
    df, levels, breaks, fvgs = h9_inputs((1.1008, 1.1010))

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_long_plan(
            df.index,
            sweep_pos=3,
            break_pos=5,
            formed_pos=5,
            entry=1.1010,
            stop=1.0995,
            risk=15.0,
        ),
    )


def test_h11a_level_known_after_sweep_cannot_be_used() -> None:
    df, _, breaks, fvgs = h1_inputs()
    levels = standard_levels(
        df.index,
        overnight_known_at=df.index[5],
    )

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_frame([]),
    )


def test_h11b_previous_day_level_can_supply_sweep() -> None:
    df, _, breaks, fvgs = h1_inputs()
    levels = standard_levels(
        df.index,
        overnight_low=np.nan,
        prev_day_low=1.1000,
    )

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_long_plan(df.index),
    )


def test_h11c_previous_day_level_known_after_sweep_cannot_be_used() -> None:
    df, _, breaks, fvgs = h1_inputs()
    levels = standard_levels(
        df.index,
        overnight_low=np.nan,
        prev_day_low=1.1000,
        prev_day_known_at=df.index[5],
    )

    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected_frame([]),
    )


@pytest.mark.parametrize(
    ("break_entries", "confirm_candles", "has_plan", "break_pos"),
    [
        ([], 8, False, None),
        ([("bullish", 3)], 8, False, None),
        ([("bearish", 6)], 8, False, None),
        ([("bullish", 7)], 8, True, 7),
        ([("bullish", 7)], 2, False, None),
        ([("bullish", 7)], 3, True, 7),
    ],
)
def test_h12_break_direction_position_and_confirmation_window(
    break_entries: list[tuple[str, int]],
    confirm_candles: int,
    has_plan: bool,
    break_pos: int | None,
) -> None:
    df, levels, _, _ = h1_inputs()
    breaks = make_breaks(df.index, break_entries)
    fvgs = make_fvgs(df.index, [("bullish", 6, 1.1008, 1.1010)])

    result = find_candidate_a_plans(
        df,
        levels,
        breaks,
        fvgs,
        confirm_candles=confirm_candles,
    )

    if has_plan:
        assert break_pos is not None
        assert_plan(
            result,
            expected_long_plan(
                df.index,
                break_pos=break_pos,
                formed_pos=6,
            ),
        )
    else:
        assert_plan(result, expected_frame([]))


def test_h13_summer_dst_uses_new_york_wall_times() -> None:
    df = make_df([Q, Q, S, A1, A2] + [T] * 11, start="2024-07-09 13:00")
    levels = standard_levels(
        df.index,
        trading_day="2024-07-09",
        overnight_known_at=df.index[2],
        prev_day_known_at=pd.Timestamp("2024-07-08 21:00", tz="UTC"),
    )
    breaks = make_breaks(df.index, [("bullish", 4)])
    fvgs = make_fvgs(df.index, [("bullish", 4, 1.1008, 1.1010)])

    expected = expected_long_plan(
        df.index,
        sweep_pos=2,
        break_pos=4,
        entry=1.1010,
        stop=1.0995,
        risk=15.0,
        formed_pos=4,
        time_exit=pd.Timestamp("2024-07-09 15:45", tz="UTC"),
        day="2024-07-09",
    )
    assert_plan(
        find_candidate_a_plans(df, levels, breaks, fvgs),
        expected,
    )


def test_h14_validation_empty_output_immutability_and_dtypes() -> None:
    df, levels, breaks, fvgs = h1_inputs()
    original_df = df.copy(deep=True)
    original_levels = levels.copy(deep=True)
    original_breaks = breaks.copy(deep=True)
    original_fvgs = fvgs.copy(deep=True)

    result = find_candidate_a_plans(df, levels, breaks, fvgs)
    pd.testing.assert_frame_equal(df, original_df)
    pd.testing.assert_frame_equal(levels, original_levels)
    pd.testing.assert_frame_equal(breaks, original_breaks)
    pd.testing.assert_frame_equal(fvgs, original_fvgs)
    assert list(result.columns) == list(PLAN_COLUMNS)
    assert str(result["trading_day"].dtype) == "datetime64[ns]"
    for column in (
        "start_time",
        "time_exit_at",
        "sweep_time",
        "break_time",
        "fvg_formed_at",
    ):
        assert str(result[column].dtype) == "datetime64[ns, UTC]"
    assert result["valid_candles"].dtype == "int64"
    for column in (
        "entry_price",
        "stop_price",
        "target_r",
        "target_price",
        "min_risk_pips",
        "max_risk_pips",
        "risk_pips",
        "fvg_zone_low",
        "fvg_zone_high",
    ):
        assert result[column].dtype == "float64"
    assert result["plan_id"].dtype == "object"

    empty = pd.DataFrame(
        {
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    assert_plan(find_candidate_a_plans(empty), expected_frame([]))
    with pytest.raises(ValueError, match="pip_size"):
        find_candidate_a_plans(df, levels, breaks, fvgs, pip_size=0)
    with pytest.raises(ValueError, match="target_r"):
        find_candidate_a_plans(df, levels, breaks, fvgs, target_r=0)
    with pytest.raises(ValueError, match="confirm_candles"):
        find_candidate_a_plans(df, levels, breaks, fvgs, confirm_candles=0)
    with pytest.raises(ValueError, match="valid_candles"):
        find_candidate_a_plans(df, levels, breaks, fvgs, valid_candles=0)
    with pytest.raises(ValueError, match="min_fvg_pips"):
        find_candidate_a_plans(df, levels, breaks, fvgs, min_fvg_pips=-1)
    with pytest.raises(ValueError, match="stop_buffer_pips"):
        find_candidate_a_plans(df, levels, breaks, fvgs, stop_buffer_pips=-1)
    with pytest.raises(ValueError, match="min_risk_pips"):
        find_candidate_a_plans(df, levels, breaks, fvgs, min_risk_pips=-1)
    with pytest.raises(ValueError, match="max_risk_pips"):
        find_candidate_a_plans(df, levels, breaks, fvgs, max_risk_pips=-1)

    naive = df.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        find_candidate_a_plans(naive, levels, breaks, fvgs)
    with pytest.raises(ValueError, match="sorted"):
        find_candidate_a_plans(df.iloc[::-1], levels, breaks, fvgs)
    with pytest.raises(ValueError, match="levels is missing"):
        find_candidate_a_plans(df, levels.drop(columns=["overnight_low"]), breaks, fvgs)
    with pytest.raises(ValueError, match="breaks is missing"):
        find_candidate_a_plans(df, levels, breaks.drop(columns=["direction"]), fvgs)
    with pytest.raises(ValueError, match="fvgs is missing"):
        find_candidate_a_plans(df, levels, breaks, fvgs.drop(columns=["formed_at"]))


def test_h15_candidate_plan_is_accepted_by_engine() -> None:
    df, levels, breaks, fvgs = h1_inputs()
    plans = find_candidate_a_plans(df, levels, breaks, fvgs)

    results = run_backtest(df, plans)

    assert len(results) == 1
    assert results.loc[0, "status"] == "unfilled_expired"


def is_market_open(timestamp: datetime) -> bool:
    return not (
        (timestamp.weekday() == 4 and timestamp.hour >= 17)
        or timestamp.weekday() == 5
        or (timestamp.weekday() == 6 and timestamp.hour < 17)
    )


def seeded_market_data(
    seed: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    candidates = pd.date_range(
        start,
        end,
        freq="15min",
        inclusive="left",
        tz="UTC",
    )
    index = pd.DatetimeIndex(
        [
            timestamp
            for timestamp in candidates
            if is_market_open(timestamp.to_pydatetime().astimezone(ET))
        ]
    ).as_unit("ns")
    rng = np.random.default_rng(seed)
    opens = 1.1000 + np.cumsum(rng.normal(0.0, 0.0004, len(index)))
    closes = opens + rng.normal(0.0, 0.0004, len(index))
    upper_wicks = rng.uniform(0.00005, 0.0004, len(index))
    lower_wicks = rng.uniform(0.00005, 0.0004, len(index))
    df = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + upper_wicks,
            "low": np.minimum(opens, closes) - lower_wicks,
            "close": closes,
        },
        index=index,
    )
    return _inject_seeded_setups(df, seed)


def _inject_seeded_setups(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    local_times = [
        timestamp.to_pydatetime().astimezone(ET)
        for timestamp in df.index
    ]
    positions_by_day: dict[date, list[int]] = {}
    for position, timestamp in enumerate(local_times):
        positions_by_day.setdefault(timestamp.date(), []).append(position)

    eligible_days = [
        day
        for day in sorted(positions_by_day)
        if day.weekday() in (1, 2, 3)
        and any(
            local_times[position].time() == time(9, 30)
            for position in positions_by_day[day]
        )
    ]
    setup_days = eligible_days[2:-1]

    def set_candle(
        position: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
    ) -> None:
        df.iloc[position] = (
            open_price,
            max(open_price, high, close),
            min(open_price, low, close),
            close,
        )

    for offset, day in enumerate(setup_days):
        session_start = next(
            position
            for position in positions_by_day[day]
            if local_times[position].time() == time(9, 30)
        )
        swing_position = session_start - 4
        anchor = float(df["open"].iloc[session_start])
        is_long = (offset + seed) % 2 == 0

        overnight_positions = [
            position
            for position, timestamp in enumerate(local_times)
            if (
                timestamp.date() == day - timedelta(days=1)
                and timestamp.time() >= time(17)
            )
            or (
                timestamp.date() == day
                and timestamp.time() < time(9, 30)
            )
        ]
        if is_long:
            for position in overnight_positions:
                set_candle(
                    position,
                    anchor + 0.0003,
                    anchor + 0.0006,
                    anchor + 0.0001,
                    anchor + 0.0003,
                )
            level_position = next(
                position
                for position in positions_by_day[day]
                if local_times[position].time() == time(8, 0)
            )
            set_candle(
                level_position,
                anchor + 0.0003,
                anchor + 0.0006,
                anchor,
                anchor + 0.0003,
            )
            for position in range(swing_position - 3, swing_position + 4):
                set_candle(
                    position,
                    anchor + 0.0004,
                    anchor + 0.0006,
                    anchor + 0.0002,
                    anchor + 0.0004,
                )
            set_candle(
                swing_position,
                anchor + 0.0005,
                anchor + 0.0010,
                anchor + 0.0002,
                anchor + 0.0006,
            )
            set_candle(
                session_start,
                anchor + 0.0003,
                anchor + 0.0005,
                anchor - 0.0005,
                anchor + 0.0001,
            )
            set_candle(
                session_start + 1,
                anchor + 0.0001,
                anchor + 0.0003,
                anchor + 0.00005,
                anchor + 0.0002,
            )
            set_candle(
                session_start + 2,
                anchor + 0.0002,
                anchor + 0.0006,
                anchor + 0.0001,
                anchor + 0.0004,
            )
            set_candle(
                session_start + 3,
                anchor + 0.0007,
                anchor + 0.0009,
                anchor + 0.0006,
                anchor + 0.0008,
            )
            set_candle(
                session_start + 4,
                anchor + 0.0008,
                anchor + 0.0014,
                anchor + 0.0007,
                anchor + 0.0012,
            )
        else:
            for position in overnight_positions:
                set_candle(
                    position,
                    anchor - 0.0003,
                    anchor - 0.0001,
                    anchor - 0.0006,
                    anchor - 0.0003,
                )
            level_position = next(
                position
                for position in positions_by_day[day]
                if local_times[position].time() == time(8, 0)
            )
            set_candle(
                level_position,
                anchor - 0.0003,
                anchor,
                anchor - 0.0006,
                anchor - 0.0003,
            )
            for position in range(swing_position - 3, swing_position + 4):
                set_candle(
                    position,
                    anchor - 0.0004,
                    anchor - 0.0002,
                    anchor - 0.0006,
                    anchor - 0.0004,
                )
            set_candle(
                swing_position,
                anchor - 0.0005,
                anchor - 0.0002,
                anchor - 0.0010,
                anchor - 0.0006,
            )
            set_candle(
                session_start,
                anchor - 0.0003,
                anchor + 0.0005,
                anchor - 0.0005,
                anchor - 0.0001,
            )
            set_candle(
                session_start + 1,
                anchor - 0.0001,
                anchor + 0.0002,
                anchor - 0.0003,
                anchor - 0.0002,
            )
            set_candle(
                session_start + 2,
                anchor - 0.0002,
                anchor + 0.0001,
                anchor - 0.0006,
                anchor - 0.0004,
            )
            set_candle(
                session_start + 3,
                anchor - 0.0007,
                anchor - 0.0006,
                anchor - 0.0009,
                anchor - 0.0008,
            )
            set_candle(
                session_start + 4,
                anchor - 0.0008,
                anchor - 0.0007,
                anchor - 0.0014,
                anchor - 0.0012,
            )
    return df


def independent_reference(
    df: pd.DataFrame,
    swing_left: int = 3,
    swing_right: int = 3,
    min_fvg_pips: float = 2.0,
    stop_buffer_pips: float = 1.0,
    min_risk_pips: float = 5.0,
    max_risk_pips: float = 25.0,
    target_r: float = 2.0,
    confirm_candles: int = 8,
    valid_candles: int = 8,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Literal reference loop using detector outputs but no candidate code."""
    levels = detect_session_levels(df)
    breaks = detect_structure_breaks(df, left=swing_left, right=swing_right)
    fvgs = detect_fvgs(df, min_gap=min_fvg_pips * pip_size)
    level_by_day = {
        pd.Timestamp(row["trading_day"]).date(): row
        for row in levels.to_dict(orient="records")
    }
    positions = {timestamp: pos for pos, timestamp in enumerate(df.index)}
    local_dates = [
        timestamp.to_pydatetime().astimezone(ET).date()
        for timestamp in df.index
    ]
    results: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for day in sorted(set(local_dates)):
        local_start = datetime.combine(day, time(9, 30), tzinfo=ET)
        local_deadline = datetime.combine(day, time(11, 30), tzinfo=ET)
        local_exit = datetime.combine(day, time(11, 45), tzinfo=ET)
        window_start = pd.Timestamp(local_start.astimezone(timezone.utc))
        deadline = pd.Timestamp(local_deadline.astimezone(timezone.utc))
        time_exit = pd.Timestamp(local_exit.astimezone(timezone.utc))
        day_levels = level_by_day.get(day)
        day_candidates = []
        if day_levels is None:
            continue
        for sweep_pos, timestamp in enumerate(df.index):
            if local_dates[sweep_pos] != day:
                continue
            if timestamp < window_start or timestamp + timedelta(minutes=15) > deadline:
                continue
            for side, direction in (("long", "bullish"), ("short", "bearish")):
                level_columns = (
                    ("overnight_low", "overnight_known_at"),
                    ("prev_day_low", "prev_day_known_at"),
                ) if side == "long" else (
                    ("overnight_high", "overnight_known_at"),
                    ("prev_day_high", "prev_day_known_at"),
                )
                swept = False
                for level_column, known_column in level_columns:
                    value = day_levels[level_column]
                    known = day_levels[known_column]
                    if pd.isna(value) or pd.isna(known) or known > timestamp:
                        continue
                    if side == "long":
                        swept = (
                            df["low"].iloc[sweep_pos] < value
                            and df["close"].iloc[sweep_pos] >= value
                        )
                    else:
                        swept = (
                            df["high"].iloc[sweep_pos] > value
                            and df["close"].iloc[sweep_pos] <= value
                        )
                    if swept:
                        break
                if not swept:
                    continue

                first_break_pos = None
                for break_row in breaks.to_dict(orient="records"):
                    if break_row["direction"] != direction:
                        continue
                    broke_at = break_row["broke_at"]
                    break_pos = positions.get(broke_at)
                    if break_pos is None:
                        continue
                    if not sweep_pos < break_pos <= sweep_pos + confirm_candles:
                        continue
                    if df.index[break_pos] + timedelta(minutes=15) > deadline:
                        continue
                    first_break_pos = break_pos
                    break
                if first_break_pos is None:
                    continue

                eligible_fvgs = []
                for order, fvg_row in enumerate(fvgs.to_dict(orient="records")):
                    if fvg_row["direction"] != direction:
                        continue
                    if fvg_row["size"] < min_fvg_pips * pip_size - 1e-9:
                        continue
                    formed_pos = positions.get(fvg_row["formed_at"])
                    if formed_pos is None:
                        continue
                    if not sweep_pos < formed_pos <= first_break_pos:
                        continue
                    eligible_fvgs.append((formed_pos, -order, fvg_row))
                if not eligible_fvgs:
                    continue
                _, _, selected_fvg = max(eligible_fvgs, key=lambda row: (row[0], row[1]))
                entry = (
                    float(selected_fvg["zone_high"])
                    if side == "long"
                    else float(selected_fvg["zone_low"])
                )
                stop = (
                    float(df["low"].iloc[sweep_pos]) - stop_buffer_pips * pip_size
                    if side == "long"
                    else float(df["high"].iloc[sweep_pos]) + stop_buffer_pips * pip_size
                )
                if (side == "long" and entry <= stop) or (
                    side == "short" and entry >= stop
                ):
                    continue
                risk_pips = abs(entry - stop) / pip_size
                if (
                    risk_pips < min_risk_pips - 1e-9
                    or risk_pips > max_risk_pips + 1e-9
                ):
                    continue
                plan_start = df.index[first_break_pos] + timedelta(minutes=15)
                start_pos = next(
                    (
                        position
                        for position, open_time in enumerate(df.index)
                        if open_time >= plan_start
                    ),
                    None,
                )
                if start_pos is None or df.index[start_pos] > time_exit:
                    continue
                plan = {
                    "trading_day": pd.Timestamp(day),
                    "plan_id": day.isoformat(),
                    "side": side,
                    "order_type": "limit",
                    "start_time": plan_start,
                    "entry_price": entry,
                    "stop_price": stop,
                    "target_r": target_r,
                    "target_price": np.nan,
                    "valid_candles": valid_candles,
                    "time_exit_at": time_exit,
                    "min_risk_pips": min_risk_pips,
                    "max_risk_pips": max_risk_pips,
                    "risk_pips": risk_pips,
                    "sweep_time": timestamp,
                    "break_time": df.index[first_break_pos],
                    "fvg_formed_at": selected_fvg["formed_at"],
                    "fvg_zone_low": float(selected_fvg["zone_low"]),
                    "fvg_zone_high": float(selected_fvg["zone_high"]),
                }
                day_candidates.append(
                    (
                        (
                            plan_start,
                            sweep_pos,
                            0 if side == "long" else 1,
                        ),
                        plan,
                    )
                )
        if day_candidates:
            results.append(min(day_candidates, key=lambda item: item[0]))
    return expected_frame([plan for _, plan in sorted(results, key=lambda item: item[0])])


@pytest.mark.parametrize("seed", [17, 31, 43])
def test_seeded_reference_and_six_no_lookahead_cutoffs(seed: int) -> None:
    periods = (
        ("spring", "2023-03-01", "2023-04-15"),
        ("fall", "2023-10-15", "2023-11-15"),
    )
    for period_name, start, end in periods:
        df = seeded_market_data(seed, start, end)
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        actual = find_candidate_a_plans(df)
        expected = independent_reference(df)
        pd.testing.assert_frame_equal(actual, expected)
        assert len(actual) >= 8, (
            f"seed {seed} {period_name} produced only {len(actual)} plans"
        )
        print(
            f"seed={seed} period={period_name} plans={len(actual)}"
        )

        eligible_cutoffs = [
            position
            for position, timestamp in enumerate(df.index)
            if 10 <= timestamp.to_pydatetime().astimezone(ET).hour <= 11
        ]
        picks = np.linspace(
            0,
            len(eligible_cutoffs) - 1,
            num=3,
            dtype=int,
        )
        for cutoff_position in (eligible_cutoffs[pick] for pick in picks):
            truncated = df.iloc[: cutoff_position + 1]
            truncated_plans = find_candidate_a_plans(truncated)
            full_prefix = actual.loc[
                actual["start_time"] <= truncated.index[-1]
            ]
            pd.testing.assert_frame_equal(
                truncated_plans.reset_index(drop=True),
                full_prefix.reset_index(drop=True),
            )
