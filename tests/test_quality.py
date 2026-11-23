import numpy as np
import pandas as pd

from src.data.quality import quality_report


def make_df(
    index: pd.DatetimeIndex,
    open_values=None,
    high=None,
    low=None,
    close=None,
    volume=None,
) -> pd.DataFrame:
    size = len(index)
    if open_values is None:
        open_values = np.full(size, 1.0)
    if high is None:
        high = np.full(size, 1.2)
    if low is None:
        low = np.full(size, 0.8)
    if close is None:
        close = np.full(size, 1.1)
    if volume is None:
        volume = np.ones(size)
    return pd.DataFrame(
        {
            "open": open_values,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_clean_frame_report() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=3,
        freq="15min",
        tz="UTC",
    )
    report = quality_report(make_df(index))

    assert report["row_count"] == 3
    assert report["first_timestamp"] == index[0]
    assert report["last_timestamp"] == index[-1]
    assert report["rows_per_year"] == {2024: 3}
    assert report["duplicate_timestamps"] == 0
    assert report["index_sorted"] is True
    assert report["invalid_ohlc_rows"] == 0
    assert report["zero_volume_rows"] == 0
    assert report["normal_weekend_gaps"] == 0
    assert report["abnormal_gap_count"] == 0


def test_duplicate_timestamp_count() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:15"],
        tz="UTC",
    )

    report = quality_report(make_df(index))

    assert report["duplicate_timestamps"] == 1


def test_bad_ohlc_row_count() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    df = make_df(index, high=[1.2, 0.9], low=[0.8, 0.8])

    report = quality_report(df)

    assert report["invalid_ohlc_rows"] == 1


def test_abnormal_gap_is_listed() -> None:
    index = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-02 12:00"], tz="UTC")

    report = quality_report(make_df(index))

    assert report["normal_weekend_gaps"] == 0
    assert report["abnormal_gap_count"] == 1
    assert report["abnormal_gaps"] == [
        {"start": index[0], "end": index[1], "hours": 36.0}
    ]


def test_normal_friday_to_sunday_gap_is_counted_separately() -> None:
    index = pd.DatetimeIndex(["2024-01-05 21:00", "2024-01-07 21:00"], tz="UTC")

    report = quality_report(make_df(index))

    assert report["normal_weekend_gaps"] == 1
    assert report["abnormal_gap_count"] == 0
    assert report["abnormal_gaps"] == []
