from datetime import datetime, timezone
import os
from pathlib import Path
import time

import pandas as pd
import dukascopy_python as duka
from dukascopy_python.instruments import I_EUR_USD

PROJECT_ROOT = Path(__file__).resolve().parents[2]
START = datetime(2016, 11, 1, tzinfo=timezone.utc)
END = datetime(2026, 11, 1, tzinfo=timezone.utc)
OUT_FILE = PROJECT_ROOT / "data" / "EURUSD_M15_bid_10y.parquet"
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2


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
    ranges = list(month_ranges(START, END))
    for month_number, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        month_name = chunk_start.strftime("%Y-%m")
        print(
            f"[{month_number}/{len(ranges)}] Downloading {month_name} ...",
            flush=True,
        )
        for attempt in range(MAX_RETRIES + 1):
            try:
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
                if not candles:
                    raise RuntimeError(f"No candles returned for {month_name}")
                frame = duka.to_dataframe(candles)
                frame = frame[["open", "high", "low", "close", "volume"]]
                if not isinstance(frame.index, pd.DatetimeIndex):
                    raise ValueError(
                        f"Downloaded {month_name} data has no DatetimeIndex"
                    )
                if frame.index.tz is None:
                    frame.index = frame.index.tz_localize("UTC")
                else:
                    frame.index = frame.index.tz_convert("UTC")
                frames.append(frame)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to download {month_name} after "
                        f"{MAX_RETRIES + 1} attempts: {exc}"
                    ) from exc
                print(
                    f"{month_name} attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {RETRY_WAIT_SECONDS}s.",
                    flush=True,
                )
                time.sleep(RETRY_WAIT_SECONDS)

    df = pd.concat(frames)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.loc[(df.index >= START) & (df.index < END)]
    df = df[~df.index.duplicated(keep="first")].sort_index()

    temporary_file = OUT_FILE.with_name(f"{OUT_FILE.name}.tmp")
    try:
        OUT_FILE.parent.mkdir(exist_ok=True)
        df.to_parquet(temporary_file)
        os.replace(temporary_file, OUT_FILE)
    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise

    print(f"Saved {len(df)} candles to {OUT_FILE}", flush=True)
    print(f"Range: {df.index.min()} to {df.index.max()}", flush=True)


if __name__ == "__main__":
    main()