import pandas as pd
import pytest

from src.data.loader import load_candles


def make_valid_path(tmp_path) -> object:
    path = tmp_path / "candles.parquet"
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100],
        },
        index=pd.DatetimeIndex(["2024-01-01 00:00"], tz="UTC"),
    )
    df.to_parquet(path)
    return path


def test_loader_loads_valid_file(tmp_path) -> None:
    path = make_valid_path(tmp_path)

    result = load_candles(path)

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


def test_loader_missing_file_names_download_fix(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m src.data.download"):
        load_candles(tmp_path / "missing.parquet")
