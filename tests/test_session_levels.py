from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_candles_10y
from src.data.resample_et import trading_day
from src.data.splits import development_set
from src.detectors.session_levels import detect_session_levels


ET = ZoneInfo("America/New_York")
LEVEL_COLUMNS = [
    "trading_day",
    "prev_day_high",
    "prev_day_high_set_at",
    "prev_day_low",
    "prev_day_low_set_at",
    "prev_day_known_at",
    "overnight_high",
    "overnight_high_set_at",
    "overnight_low",
    "overnight_low_set_at",
    "overnight_known_at",
]
TIME_COLUMNS = [
    "prev_day_high_set_at",
    "prev_day_low_set_at",
    "prev_day_known_at",
    "overnight_high_set_at",
    "overnight_low_set_at",
    "overnight_known_at",
]


def utc_index(*values: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(values, tz="UTC").as_unit("ns")


def make_df(
    index: pd.DatetimeIndex,
    highs: list[float],
    lows: list[float],
) -> pd.DataFrame:
    return pd.DataFrame({"high": highs, "low": lows}, index=index)


def expected_levels(rows: list[dict[str, object]]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=LEVEL_COLUMNS)
    result["trading_day"] = pd.to_datetime(result["trading_day"]).astype(
        "datetime64[ns]"
    )
    for column in TIME_COLUMNS:
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def independent_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Derive session levels directly from zoneinfo-localized candle times."""
    groups: dict[object, list[int]] = {}
    local_times = []
    for position, timestamp in enumerate(df.index):
        local = timestamp.to_pydatetime().astimezone(ET)
        local_times.append(local)
        day = local.date() + timedelta(days=local.hour >= 17)
        groups.setdefault(day, []).append(position)

    rows = []
    days = sorted(groups)
    for number, day in enumerate(days):
        positions = groups[day]
        row: dict[str, object] = {
            "trading_day": pd.Timestamp(day),
            "prev_day_high": np.nan,
            "prev_day_high_set_at": pd.NaT,
            "prev_day_low": np.nan,
            "prev_day_low_set_at": pd.NaT,
            "prev_day_known_at": pd.NaT,
            "overnight_high": np.nan,
            "overnight_high_set_at": pd.NaT,
            "overnight_low": np.nan,
            "overnight_low_set_at": pd.NaT,
            "overnight_known_at": pd.NaT,
        }
        if number:
            previous = groups[days[number - 1]]
            previous_high = max(df["high"].iloc[position] for position in previous)
            previous_low = min(df["low"].iloc[position] for position in previous)
            row.update(
                {
                    "prev_day_high": previous_high,
                    "prev_day_high_set_at": df.index[
                        next(
                            position
                            for position in previous
                            if df["high"].iloc[position] == previous_high
                        )
                    ],
                    "prev_day_low": previous_low,
                    "prev_day_low_set_at": df.index[
                        next(
                            position
                            for position in previous
                            if df["low"].iloc[position] == previous_low
                        )
                    ],
                    "prev_day_known_at": df.index[positions[0]],
                }
            )

        overnight = []
        morning = []
        for position in positions:
            local = local_times[position]
            local_time = local.timetz().replace(tzinfo=None)
            if local_time >= time(17) or local_time < time(9, 30):
                overnight.append(position)
            if time(9, 30) <= local_time < time(17):
                morning.append(position)
        if overnight and morning:
            overnight_high = max(df["high"].iloc[position] for position in overnight)
            overnight_low = min(df["low"].iloc[position] for position in overnight)
            row.update(
                {
                    "overnight_high": overnight_high,
                    "overnight_high_set_at": df.index[
                        next(
                            position
                            for position in overnight
                            if df["high"].iloc[position] == overnight_high
                        )
                    ],
                    "overnight_low": overnight_low,
                    "overnight_low_set_at": df.index[
                        next(
                            position
                            for position in overnight
                            if df["low"].iloc[position] == overnight_low
                        )
                    ],
                    "overnight_known_at": df.index[morning[0]],
                }
            )
        rows.append(row)
    return expected_levels(rows)


def is_market_open(local: datetime) -> bool:
    return not (
        local.weekday() == 5
        or (local.weekday() == 6 and local.hour < 17)
        or (local.weekday() == 4 and local.hour >= 17)
    )


def seeded_dst_data(seed: int, start: str, end: str) -> pd.DataFrame:
    candidates = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    index = pd.DatetimeIndex(
        [
            timestamp
            for timestamp in candidates
            if is_market_open(timestamp.to_pydatetime().astimezone(ET))
        ]
    ).as_unit("ns")
    rng = np.random.default_rng(seed)
    base = 1.1 + np.cumsum(rng.normal(0, 0.0001, len(index)))
    spread = rng.uniform(0.00001, 0.0002, len(index))
    return pd.DataFrame(
        {
            "high": base + spread,
            "low": base - spread,
        },
        index=index,
    )


def test_previous_day_and_overnight_levels_exactly() -> None:
    index = utc_index(
        "2024-01-07 22:00",
        "2024-01-08 15:00",
        "2024-01-08 21:45",
        "2024-01-08 22:00",
        "2024-01-09 03:00",
        "2024-01-09 14:15",
        "2024-01-09 14:30",
        "2024-01-09 15:00",
    )
    source = make_df(
        index,
        [1.10, 1.20, 1.08, 1.15, 1.18, 1.16, 1.30, 1.14],
        [1.05, 1.15, 1.00, 1.11, 1.09, 1.10, 1.02, 1.10],
    )
    expected = expected_levels(
        [
            {
                "trading_day": "2024-01-08",
                "prev_day_high": np.nan,
                "prev_day_high_set_at": pd.NaT,
                "prev_day_low": np.nan,
                "prev_day_low_set_at": pd.NaT,
                "prev_day_known_at": pd.NaT,
                "overnight_high": 1.10,
                "overnight_high_set_at": index[0],
                "overnight_low": 1.05,
                "overnight_low_set_at": index[0],
                "overnight_known_at": index[1],
            },
            {
                "trading_day": "2024-01-09",
                "prev_day_high": 1.20,
                "prev_day_high_set_at": index[1],
                "prev_day_low": 1.00,
                "prev_day_low_set_at": index[2],
                "prev_day_known_at": index[3],
                "overnight_high": 1.18,
                "overnight_high_set_at": index[4],
                "overnight_low": 1.09,
                "overnight_low_set_at": index[4],
                "overnight_known_at": index[6],
            },
        ]
    )

    pd.testing.assert_frame_equal(detect_session_levels(source), expected)


def test_0915_candle_is_in_overnight_range_and_0930_is_excluded() -> None:
    index = utc_index("2024-07-09 13:15", "2024-07-09 13:30")
    source = make_df(index, [1.50, 1.40], [1.40, 1.30])
    expected = expected_levels(
        [
            {
                "trading_day": "2024-07-09",
                "prev_day_high": np.nan,
                "prev_day_high_set_at": pd.NaT,
                "prev_day_low": np.nan,
                "prev_day_low_set_at": pd.NaT,
                "prev_day_known_at": pd.NaT,
                "overnight_high": 1.50,
                "overnight_high_set_at": index[0],
                "overnight_low": 1.40,
                "overnight_low_set_at": index[0],
                "overnight_known_at": index[1],
            }
        ]
    )

    pd.testing.assert_frame_equal(detect_session_levels(source), expected)


def test_empty_overnight_window_leaves_overnight_fields_null() -> None:
    index = utc_index("2024-01-09 15:00")
    source = make_df(index, [1.2], [1.0])
    expected = expected_levels(
        [
            {
                "trading_day": "2024-01-09",
                "prev_day_high": np.nan,
                "prev_day_high_set_at": pd.NaT,
                "prev_day_low": np.nan,
                "prev_day_low_set_at": pd.NaT,
                "prev_day_known_at": pd.NaT,
                "overnight_high": np.nan,
                "overnight_high_set_at": pd.NaT,
                "overnight_low": np.nan,
                "overnight_low_set_at": pd.NaT,
                "overnight_known_at": pd.NaT,
            }
        ]
    )

    pd.testing.assert_frame_equal(detect_session_levels(source), expected)


def test_first_occurrence_wins_equal_level_ties() -> None:
    index = utc_index(
        "2024-01-07 22:00",
        "2024-01-08 14:15",
        "2024-01-08 14:30",
    )
    source = make_df(index, [1.20, 1.20, 1.15], [1.00, 1.00, 1.05])
    result = detect_session_levels(source)

    assert result.loc[0, "overnight_high"] == 1.20
    assert result.loc[0, "overnight_high_set_at"] == index[0]
    assert result.loc[0, "overnight_low"] == 1.00
    assert result.loc[0, "overnight_low_set_at"] == index[0]


def test_trading_day_boundaries_follow_spring_and_fall_dst() -> None:
    index = utc_index("2024-03-10 21:00", "2024-11-03 22:00")

    assert trading_day(index).equals(
        pd.DatetimeIndex(["2024-03-11", "2024-11-04"])
    )


def test_seeded_session_levels_match_independent_zoneinfo_reference() -> None:
    source = pd.concat(
        [
            seeded_dst_data(23, "2023-03-01", "2023-04-15"),
            seeded_dst_data(37, "2023-10-15", "2023-11-15"),
        ]
    ).sort_index()

    pd.testing.assert_frame_equal(
        detect_session_levels(source),
        independent_reference(source),
    )


def test_input_is_not_modified() -> None:
    source = make_df(
        utc_index("2024-01-07 22:00", "2024-01-08 15:00"),
        [1.2, 1.3],
        [1.0, 1.1],
    )
    before = source.copy(deep=True)

    detect_session_levels(source)

    pd.testing.assert_frame_equal(source, before)


def test_naive_and_unsorted_indexes_are_rejected() -> None:
    naive = pd.DatetimeIndex(["2024-01-08 10:00"])
    with pytest.raises(ValueError, match="UTC"):
        detect_session_levels(make_df(naive, [1.2], [1.0]))

    unsorted = utc_index("2024-01-08 15:00", "2024-01-08 14:00")
    with pytest.raises(ValueError, match="sorted"):
        detect_session_levels(make_df(unsorted, [1.2, 1.3], [1.0, 1.1]))


def test_real_10y_session_levels_have_no_lookahead() -> None:
    try:
        source = development_set(load_candles_10y())
    except FileNotFoundError as exc:
        pytest.skip(f"10-year candle file is unavailable: {exc}")

    result = detect_session_levels(source)
    assert result["trading_day"].is_monotonic_increasing
    assert result["trading_day"].is_unique
    assert result.iloc[0]["prev_day_high"] != result.iloc[0]["prev_day_high"]

    for value, set_at, known_at in (
        ("prev_day_high", "prev_day_high_set_at", "prev_day_known_at"),
        ("prev_day_low", "prev_day_low_set_at", "prev_day_known_at"),
        ("overnight_high", "overnight_high_set_at", "overnight_known_at"),
        ("overnight_low", "overnight_low_set_at", "overnight_known_at"),
    ):
        valid = result[value].notna()
        assert result.loc[valid, known_at].gt(result.loc[valid, set_at]).all()

    overnight_valid = result["overnight_high"].notna()
    assert (
        result.loc[overnight_valid, "overnight_high"]
        >= result.loc[overnight_valid, "overnight_low"]
    ).all()
