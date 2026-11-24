"""Candle loading for the chart application."""

import pandas as pd
import streamlit as st

from src.data.loader import load_candles
from src.data.splits import development_set


@st.cache_data
def cached_load_candles() -> pd.DataFrame:
    """Load and cache development candles, excluding the sealed holdout."""
    return development_set(load_candles())
