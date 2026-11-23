from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_candles_10y
from src.data.splits import development_set
from src.data.resample_et import (
    TIMEFRAME_DURATIONS,
    align_htf_to_ltf,
    resample_ohlcv_et,
    trading_day,
)


ET = ZoneInfo("America/New_York")
OHLCV = ["open", "high", "low", "close", "volume"]


def utc_index(*values: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(values, tz="UTC").as_unit("ns")


def make_df(
    index: pd.DatetimeIndex,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )


def candle_window_is_open(local: datetime) -> bool:
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
            if candle_window_is_open(timestamp.to_pydatetime().astimezone(ET))
        ]
    ).as_unit("ns")
    rng = np.random.default_rng(seed)
    opens = 1.1 + np.cumsum(rng.normal(0, 0.0001, len(index)))
    closes = opens + rng.normal(0, 0.0001, len(index))
    spread = rng.uniform(0.00001, 0.0002, len(index))
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + spread,
            "low": np.minimum(opens, closes) - spread,
            "close": closes,
            "volume": rng.integers(1, 100, len(index)),
        },
        index=index,
    )


def reference_resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Independent, per-candle wall-clock reference using zoneinfo."""
    duration = TIMEFRAME_DURATIONS[timeframe]
    grouped: dict[pd.Timestamp, list[tuple[object, ...]]] = {}
    for position, timestamp in enumerate(df.index):
        local = timestamp.to_pydatetime().astimezone(ET)
        wall = local.replace(tzinfo=None)
        if timeframe == "1D":
            start_date = local.date()
            if local.hour < 17:
                start_date -= timedelta(days=1)
            wall_start = datetime.combine(start_date, time(17))
        else:
            shifted = wall - timedelta(hours=17)
            bucket_hour = (shifted.hour // 4) * 4
            wall_start = (
                shifted.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
                + timedelta(hours=17)
            )
        localized_start = wall_start.replace(tzinfo=ET)
        start = pd.Timestamp(localized_start.astimezone(ZoneInfo("UTC")))
        grouped.setdefault(start, []).append(
            tuple(df[column].iloc[position] for column in OHLCV)
        )

    rows = []
    labels = []
    for start, candles in sorted(grouped.items()):
        rows.append(
            {
                "open": candles[0][0],
                "high": max(candle[1] for candle in candles),
                "low": min(candle[2] for candle in candles),
                "close": candles[-1][3],
                "volume": sum(candle[4] for candle in candles),
                "n_candles": len(candles),
                "close_time": start + duration,
            }
        )
        labels.append(start)
    result = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(labels).as_unit("ns"),
    )
    result.index.name = df.index.name
    result["n_candles"] = result["n_candles"].astype("int64")
    result["close_time"] = pd.Series(
        pd.DatetimeIndex(pd.to_datetime(result["close_time"], utc=True)).as_unit("ns"),
        index=result.index,
    )
    return result[
        ["open", "high", "low", "close", "volume", "n_candles", "close_time"]
    ]


def reference_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    days = []
    for timestamp in index:
        local = timestamp.to_pydatetime().astimezone(ET)
        day = local.date() + timedelta(days=local.hour >= 17)
        days.append(pd.Timestamp(day))
    return pd.DatetimeIndex(days, name=index.name)


def test_trading_day_rolls_at_1700_et() -> None:
    index = utc_index(
        "2024-01-07 22:00",
        "2024-01-08 21:45",
        "2024-01-08 22:00",
    )

    assert trading_day(index).equals(
        pd.DatetimeIndex(
            ["2024-01-08", "2024-01-08", "2024-01-09"],
            name=index.name,
        )
    )


def test_daily_bins_follow_et_trading_days_and_report_elapsed_close() -> None:
    index = utc_index(
        "2024-01-07 22:00",
        "2024-01-08 21:45",
        "2024-01-08 22:00",
        "2024-01-09 05:00",
    )
    source = make_df(
        index,
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5, 4.5],
        [0.5, 1.5, 2.5, 3.5],
        [1.2, 2.2, 3.2, 4.2],
        [10.0, 20.0, 30.0, 40.0],
    )
    expected = pd.DataFrame(
        {
            "open": [1.0, 3.0],
            "high": [2.5, 4.5],
            "low": [0.5, 2.5],
            "close": [2.2, 4.2],
            "volume": [30.0, 70.0],
            "n_candles": [2, 2],
            "close_time": utc_index("2024-01-08 22:00", "2024-01-09 22:00"),
        },
        index=utc_index("2024-01-07 22:00", "2024-01-08 22:00"),
    )
    expected["n_candles"] = expected["n_candles"].astype("int64")

    pd.testing.assert_frame_equal(resample_ohlcv_et(source, "1D"), expected)


def test_four_hour_bins_align_to_et_boundaries_and_aggregate_exactly() -> None:
    index = utc_index(
        "2024-01-09 01:45",
        "2024-01-09 02:00",
        "2024-01-09 05:45",
        "2024-01-09 06:00",
    )
    source = make_df(
        index,
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5, 4.5],
        [0.5, 1.5, 2.5, 3.5],
        [1.1, 2.1, 3.1, 4.1],
        [1.0, 2.0, 3.0, 4.0],
    )
    expected = pd.DataFrame(
        {
            "open": [1.0, 2.0, 4.0],
            "high": [1.5, 3.5, 4.5],
            "low": [0.5, 1.5, 3.5],
            "close": [1.1, 3.1, 4.1],
            "volume": [1.0, 5.0, 4.0],
            "n_candles": [1, 2, 1],
            "close_time": utc_index(
                "2024-01-09 02:00",
                "2024-01-09 06:00",
                "2024-01-09 10:00",
            ),
        },
        index=utc_index(
            "2024-01-08 22:00",
            "2024-01-09 02:00",
            "2024-01-09 06:00",
        ),
    )
    expected["n_candles"] = expected["n_candles"].astype("int64")

    pd.testing.assert_frame_equal(resample_ohlcv_et(source, "4h"), expected)


def test_summer_candles_use_the_expected_et_bins() -> None:
    index = utc_index(
        "2024-07-09 12:45",
        "2024-07-09 13:00",
        "2024-07-09 16:45",
        "2024-07-09 17:00",
        "2024-07-09 20:45",
        "2024-07-09 21:00",
    )
    source = make_df(
        index,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        [0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
        [1.05, 2.05, 3.05, 4.05, 5.05, 6.05],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    daily = resample_ohlcv_et(source, "1D")
    four_hour = resample_ohlcv_et(source, "4h")
    assert daily.index.equals(
        utc_index("2024-07-08 21:00", "2024-07-09 21:00")
    )
    assert four_hour.index.equals(
        utc_index(
            "2024-07-09 09:00",
            "2024-07-09 13:00",
            "2024-07-09 17:00",
            "2024-07-09 21:00",
        )
    )


def test_daily_bins_cross_both_dst_transitions_without_shift() -> None:
    spring = make_df(
        utc_index(
            "2024-03-08 21:45",
            "2024-03-10 21:00",
            "2024-03-11 20:45",
        ),
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.9, 1.9, 2.9],
        [1.05, 2.05, 3.05],
        [1.0, 2.0, 3.0],
    )
    fall = make_df(
        utc_index(
            "2024-11-01 20:45",
            "2024-11-03 22:00",
            "2024-11-04 21:45",
        ),
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.9, 1.9, 2.9],
        [1.05, 2.05, 3.05],
        [1.0, 2.0, 3.0],
    )
    spring_result = resample_ohlcv_et(spring, "1D")
    fall_result = resample_ohlcv_et(fall, "1D")
    spring_four_hour = resample_ohlcv_et(spring, "4h")
    fall_four_hour = resample_ohlcv_et(fall, "4h")

    assert spring_result.index.equals(
        utc_index("2024-03-07 22:00", "2024-03-10 21:00")
    )
    assert fall_result.index.equals(
        utc_index("2024-10-31 21:00", "2024-11-03 22:00")
    )
    assert spring_result.loc[pd.Timestamp("2024-03-10 21:00", tz="UTC"), "n_candles"] == 2
    assert fall_result.loc[pd.Timestamp("2024-11-03 22:00", tz="UTC"), "n_candles"] == 2
    assert spring_four_hour.index.equals(
        utc_index("2024-03-08 18:00", "2024-03-10 21:00", "2024-03-11 17:00")
    )
    assert fall_four_hour.index.equals(
        utc_index("2024-11-01 17:00", "2024-11-03 22:00", "2024-11-04 18:00")
    )


def test_seeded_resampling_matches_independent_zoneinfo_reference() -> None:
    frames = [
        seeded_dst_data(17, "2023-03-01", "2023-04-15"),
        seeded_dst_data(29, "2023-10-15", "2023-11-15"),
    ]
    source = pd.concat(frames).sort_index()

    assert trading_day(source.index).equals(reference_trading_days(source.index))
    for timeframe in ("4h", "1D"):
        pd.testing.assert_frame_equal(
            resample_ohlcv_et(source, timeframe),
            reference_resample(source, timeframe),
        )


def test_15m_is_a_copy_with_expected_metadata() -> None:
    source = make_df(
        utc_index("2024-01-08 15:00", "2024-01-08 15:15"),
        [1.0, 2.0],
        [1.2, 2.2],
        [0.8, 1.8],
        [1.1, 2.1],
        [10.0, 20.0],
    )
    before = source.copy(deep=True)
    result = resample_ohlcv_et(source, "15m")

    assert result.index.equals(source.index)
    pd.testing.assert_frame_equal(result[OHLCV], source)
    pd.testing.assert_frame_equal(source, before)
    assert result["n_candles"].tolist() == [1, 1]
    assert pd.DatetimeIndex(result["close_time"]).equals(
        utc_index("2024-01-08 15:15", "2024-01-08 15:30")
    )


def test_resampling_does_not_create_empty_bins_across_weekends() -> None:
    source = make_df(
        utc_index("2024-01-05 21:45", "2024-01-07 22:00"),
        [1.0, 2.0],
        [1.2, 2.2],
        [0.8, 1.8],
        [1.1, 2.1],
        [10.0, 20.0],
    )

    result = resample_ohlcv_et(source, "4h")

    assert result.index.equals(utc_index("2024-01-05 18:00", "2024-01-07 22:00"))
    assert result["n_candles"].tolist() == [1, 1]


def test_htf_alignment_uses_close_time_not_bin_start() -> None:
    htf = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [10.0, 20.0],
            "close_time": utc_index("2024-01-08 22:00", "2024-01-09 02:00"),
        },
        index=utc_index("2024-01-08 18:00", "2024-01-08 22:00"),
    )
    ltf_index = utc_index(
        "2024-01-09 01:45",
        "2024-01-09 02:00",
        "2024-01-09 03:45",
        "2024-01-09 04:00",
    )
    result = align_htf_to_ltf(ltf_index, htf, ["close"])
    expected = pd.DataFrame(
        {
            "close": [1.1, 1.2, 1.2, 1.2],
            "htf_start": utc_index(
                "2024-01-08 18:00",
                "2024-01-08 22:00",
                "2024-01-08 22:00",
                "2024-01-08 22:00",
            ),
        },
        index=ltf_index,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_alignment_is_no_lookahead_on_seeded_data() -> None:
    source = seeded_dst_data(41, "2023-03-01", "2023-03-08")
    htf = resample_ohlcv_et(source, "4h")
    full = align_htf_to_ltf(source.index, htf)

    for cutoff in (120, 300, 500):
        if cutoff >= len(source):
            continue
        truncated_source = source.iloc[: cutoff + 1]
        truncated_htf = resample_ohlcv_et(truncated_source, "4h")
        truncated = align_htf_to_ltf(truncated_source.index, truncated_htf)
        expected = full.loc[full.index <= truncated.index[-1]]
        pd.testing.assert_frame_equal(
            truncated.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


def test_invalid_timeframe_raises_value_error() -> None:
    source = make_df(
        utc_index("2024-01-08 15:00"),
        [1.0],
        [1.2],
        [0.8],
        [1.1],
        [10.0],
    )

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        resample_ohlcv_et(source, "1h")


@pytest.mark.parametrize("timeframe", ["4h", "1D"])
def test_real_10y_resampling_invariants(timeframe: str) -> None:
    try:
        source = development_set(load_candles_10y())
    except FileNotFoundError as exc:
        pytest.skip(f"10-year candle file is unavailable: {exc}")

    result = resample_ohlcv_et(source, timeframe)
    duration = TIMEFRAME_DURATIONS[timeframe]
    assert result["n_candles"].sum() == len(source)
    assert result["volume"].sum() == pytest.approx(source["volume"].sum())
    assert pd.DatetimeIndex(result["close_time"]).equals(result.index + duration)
    assert result["n_candles"].gt(0).all()

    local_starts = result.index.tz_convert(ET)
    if timeframe == "1D":
        assert (local_starts.hour == 17).all()
    else:
        assert set(local_starts.hour).issubset({1, 5, 9, 13, 17, 21})
        assert (local_starts.minute == 0).all()
    assert result["high"].max() == source["high"].max()
    assert result["low"].min() == source["low"].min()


def test_naive_or_non_utc_index_is_rejected() -> None:
    naive = pd.DatetimeIndex(["2024-01-08 15:00"])
    source = make_df(naive, [1.0], [1.2], [0.8], [1.1], [10.0])

    with pytest.raises(ValueError, match="UTC"):
        trading_day(naive)
    with pytest.raises(ValueError, match="UTC"):
        resample_ohlcv_et(source, "1D")


def test_ambiguous_et_bin_start_raises_with_source_timestamp() -> None:
    index = utc_index("2024-11-03 05:15")
    source = make_df(index, [1.0], [1.2], [0.8], [1.1], [10.0])

    with pytest.raises(ValueError, match="2024-11-03 05:15:00\\+00:00"):
        resample_ohlcv_et(source, "4h")
