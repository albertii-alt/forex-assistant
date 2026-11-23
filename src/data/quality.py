"""Data quality checks for the 10-year EUR/USD candle file."""

import numpy as np
import pandas as pd

from src.data.loader import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
)


TEN_YEAR_DATA_PATH = PROJECT_ROOT / "data" / "EURUSD_M15_bid_10y.parquet"
OVERLAP_START = pd.Timestamp("2023-11-01", tz="UTC")
OVERLAP_END = pd.Timestamp("2026-10-31", tz="UTC")
PRICE_COLUMNS = ("open", "high", "low", "close")
GAP_THRESHOLD = pd.Timedelta(hours=12)


def quality_report(df: pd.DataFrame) -> dict:
    """Summarize candle quality, gaps, and largest intrabar ranges."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    prices = df.loc[:, list(PRICE_COLUMNS)].to_numpy(dtype=float)
    invalid_price = ~np.isfinite(prices) | (prices <= 0)
    open_values, high_values, low_values, close_values = prices.T
    bad_high = high_values < np.maximum.reduce(
        [open_values, close_values, low_values]
    )
    bad_low = low_values > np.minimum.reduce(
        [open_values, close_values, high_values]
    )
    invalid_ohlc_rows = int(
        np.any(invalid_price, axis=1).sum()
        + np.sum((bad_high | bad_low) & ~np.any(invalid_price, axis=1))
    )

    rows_per_year = {
        int(year): int(count)
        for year, count in df.index.year.value_counts().sort_index().items()
    }

    normal_weekend_gaps = 0
    abnormal_gaps = []
    for previous, following in zip(df.index[:-1], df.index[1:]):
        gap = following - previous
        if gap <= GAP_THRESHOLD:
            continue
        if previous.weekday() == 4 and following.weekday() == 6:
            normal_weekend_gaps += 1
        else:
            abnormal_gaps.append(
                {
                    "start": previous,
                    "end": following,
                    "hours": gap.total_seconds() / 3600,
                }
            )
    abnormal_gaps.sort(key=lambda item: item["hours"], reverse=True)

    ranges_pips = (high_values - low_values) * 10_000
    top_positions = np.argsort(ranges_pips)[::-1][:10]
    largest_ranges = [
        {
            "timestamp": df.index[int(pos)],
            "range_pips": float(ranges_pips[pos]),
        }
        for pos in top_positions
    ]

    return {
        "row_count": len(df),
        "first_timestamp": df.index.min() if len(df) else None,
        "last_timestamp": df.index.max() if len(df) else None,
        "rows_per_year": rows_per_year,
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "index_sorted": bool(df.index.is_monotonic_increasing),
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "zero_volume_rows": int((df["volume"] == 0).sum()),
        "normal_weekend_gaps": normal_weekend_gaps,
        "abnormal_gap_count": len(abnormal_gaps),
        "abnormal_gaps": abnormal_gaps[:30],
        "largest_ranges": largest_ranges,
    }


def _overlap_report(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """Compare timestamp coverage and OHLC values in the shared date span."""
    new_overlap = new.loc[(new.index >= OVERLAP_START) & (new.index < OVERLAP_END)]
    old_overlap = old.loc[(old.index >= OVERLAP_START) & (old.index < OVERLAP_END)]
    new_index = new_overlap.index
    old_index = old_overlap.index
    shared = new_index.intersection(old_index)
    new_only = new_index.difference(old_index)
    old_only = old_index.difference(new_index)

    max_differences = {}
    for column in PRICE_COLUMNS:
        if len(shared):
            difference = (
                new_overlap.loc[shared, column].astype(float)
                - old_overlap.loc[shared, column].astype(float)
            ).abs()
            max_differences[column] = float(difference.max())
        else:
            max_differences[column] = None

    return {
        "period_start": OVERLAP_START,
        "period_end_exclusive": OVERLAP_END,
        "new_only_timestamps": len(new_only),
        "old_only_timestamps": len(old_only),
        "shared_timestamps": len(shared),
        "max_abs_ohlc_difference": max_differences,
    }


def _format_timestamp(timestamp: pd.Timestamp | None) -> str:
    return "None" if timestamp is None else timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    new_candles = pd.read_parquet(TEN_YEAR_DATA_PATH)
    old_candles = pd.read_parquet(DEFAULT_DATA_PATH)
    report = quality_report(new_candles)
    overlap = _overlap_report(new_candles, old_candles)

    print("EUR/USD M15 10-year data quality report")
    print(f"File: {TEN_YEAR_DATA_PATH}")
    print(f"Rows: {report['row_count']}")
    print(f"First timestamp: {_format_timestamp(report['first_timestamp'])}")
    print(f"Last timestamp: {_format_timestamp(report['last_timestamp'])}")
    print("Rows per calendar year:")
    for year, count in report["rows_per_year"].items():
        print(f"  {year}: {count}")
    print(f"Duplicate timestamps: {report['duplicate_timestamps']}")
    print(f"Index sorted: {report['index_sorted']}")
    print(f"Invalid OHLC/price rows: {report['invalid_ohlc_rows']}")
    print(f"Rows with zero volume: {report['zero_volume_rows']}")
    print(f"Normal weekend gaps: {report['normal_weekend_gaps']}")
    print(f"Abnormal gaps over 12h: {report['abnormal_gap_count']}")
    print("Longest abnormal gaps (maximum 30):")
    for gap in report["abnormal_gaps"]:
        print(
            f"  {_format_timestamp(gap['start'])} -> "
            f"{_format_timestamp(gap['end'])}: {gap['hours']:.2f} hours"
        )
    print("10 largest single-candle high-low ranges:")
    for item in report["largest_ranges"]:
        print(
            f"  {_format_timestamp(item['timestamp'])}: "
            f"{item['range_pips']:.2f} pips"
        )
    print(
        "Overlap period: "
        f"{overlap['period_start'].date()} through "
        f"{(overlap['period_end_exclusive'] - pd.Timedelta(days=1)).date()}"
    )
    print(f"New-only timestamps: {overlap['new_only_timestamps']}")
    print(f"Old-only timestamps: {overlap['old_only_timestamps']}")
    print(f"Shared timestamps: {overlap['shared_timestamps']}")
    print("Maximum absolute shared OHLC differences:")
    for column, difference in overlap["max_abs_ohlc_difference"].items():
        print(f"  {column}: {difference}")


if __name__ == "__main__":
    main()
