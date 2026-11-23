"""Development and sealed holdout split helpers."""

import pandas as pd


# Boundary declared in docs/strategy_rules.md.
HOLDOUT_START = pd.Timestamp("2024-11-01", tz="UTC")


def _validate_utc_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz).upper() != "UTC":
        raise ValueError("df index must be timezone-aware UTC")


def development_set(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of rows strictly before the holdout boundary."""
    _validate_utc_index(df)
    return df.loc[df.index < HOLDOUT_START].copy()


def holdout_set(df: pd.DataFrame) -> pd.DataFrame:
    """Return sealed rows for one final evaluation of a dev-qualified candidate.

    Use only for a final one-time evaluation of a candidate that already
    passed on the development set.
    """
    _validate_utc_index(df)
    return df.loc[df.index >= HOLDOUT_START].copy()
