from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import dukascopy_python as duka
from dukascopy_python.instruments import I_EUR_USD

START = datetime(2023, 11, 1, tzinfo=timezone.utc)
END = datetime(2026, 11, 1, tzinfo=timezone.utc)
OUT_FILE = Path("data/EURUSD_M15_bid.parquet")


def month_ranges(start, end):
    """Split the full range into month-sized chunks."""
    current = start
    while current < end:
        nxt = datetime(
            current.year + (current.month == 12),
            current.month % 12 + 1,
            1,
            tzinfo=timezone.utc,
        )
        yield current, min(nxt, end)
        current = nxt


def main():
    frames = []
    for chunk_start, chunk_end in month_ranges(START, END):
        print(f"Downloading {chunk_start:%Y-%m} ...")
        candles = list(
            duka.fetch_candles(
                instrument=I_EUR_USD,
                interval=15,
                time_unit=duka.TimeUnit.MINUTE,
                offer_side=duka.OfferSide.BID,
                start=chunk_start,
                end=chunk_end,
            )
        )
        if candles:
            frames.append(duka.to_dataframe(candles))

    df = pd.concat(frames)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()

    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT_FILE)
    print(f"Saved {len(df)} candles to {OUT_FILE}")
    print(f"Range: {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    main()