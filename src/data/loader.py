"""Loading and validation for the downloaded OHLCV data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "EURUSD_M15_bid.parquet"
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def load_candles(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the EUR/USD 15-minute BID candle file."""
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing candle file: {data_path}. Fix by running "
            "`python -m src.data.download`."
        )

    df = pd.read_parquet(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Candle file is missing required columns: {', '.join(missing)}. "
            "Fix by running `python -m src.data.download`."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Candle index must be a DatetimeIndex in UTC. "
            "Fix by running `python -m src.data.download`."
        )
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError(
            "Candle timestamps must be timezone-aware UTC. "
            "Fix by running `python -m src.data.download`."
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Candle timestamps must be sorted. "
            "Fix by running `python -m src.data.download`."
        )
    if df.index.has_duplicates:
        raise ValueError(
            "Candle timestamps contain duplicates. "
            "Fix by running `python -m src.data.download`."
        )

    return df.loc[:, list(REQUIRED_COLUMNS)].copy()


def load_candles_10y() -> pd.DataFrame:
    """Load and validate the 10-year EUR/USD 15-minute BID candle file."""
    return load_candles(PROJECT_ROOT / "data" / "EURUSD_M15_bid_10y.parquet")
