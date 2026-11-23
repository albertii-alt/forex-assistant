"""Streamlit chart viewer for the EUR/USD BID candle data."""

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_candles
from src.data.resample import resample_ohlcv
from src.detectors.swings import detect_swings


MAX_CANDLES = 5_000


@st.cache_data
def cached_load_candles() -> pd.DataFrame:
    """Load and validate candles once per Streamlit process."""
    return load_candles()


@st.cache_data
def cached_resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample the full dataset once for the selected timeframe."""
    return resample_ohlcv(df, timeframe)


@st.cache_data
def cached_detect_swings(
    df: pd.DataFrame,
    left: int,
    right: int,
) -> pd.DataFrame:
    """Cache swing detection for the full resampled dataset."""
    return detect_swings(df, left=left, right=right)


def weekend_rangebreaks(df: pd.DataFrame) -> list[dict]:
    """Build exact rangebreaks around gaps of at least two days."""
    if len(df) < 2:
        return []

    breaks = []
    timestamps = df.index
    for previous, following in zip(timestamps[:-1], timestamps[1:]):
        if following - previous >= timedelta(days=2):
            breaks.append(
                {
                    "bounds": [previous.isoformat(), following.isoformat()],
                }
            )
    return breaks


st.set_page_config(page_title="Forex Assistant", layout="wide")
st.sidebar.title("Chart controls")

timeframe = st.sidebar.selectbox("Timeframe", ("15m", "1h", "4h", "1D"))
show_swings = st.sidebar.checkbox("Show swing highs/lows", value=False)
left = st.sidebar.number_input(
    "Left candles",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
)
right = st.sidebar.number_input(
    "Right candles",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
)

try:
    candles = cached_load_candles()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except (ValueError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

available_dates = sorted(set(candles.index.normalize()))
if not available_dates:
    st.error("The candle file contains no timestamps.")
    st.stop()

first_available = available_dates[0].date()
last_available = available_dates[-1].date()
last_five_trading_dates = available_dates[-5:]
default_start = last_five_trading_dates[0].date()
default_end = last_available

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(default_start, default_end),
    min_value=first_available,
    max_value=last_available,
)

if isinstance(selected_dates, (list, tuple)):
    if len(selected_dates) == 0:
        selected_dates = (default_start, default_end)
    elif len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

if start_date > end_date:
    start_date, end_date = end_date, start_date

start_utc = pd.Timestamp(datetime.combine(start_date, time.min), tz="UTC")
end_utc = pd.Timestamp(
    datetime.combine(end_date, time.max),
    tz="UTC",
)
selected = candles.loc[(candles.index >= start_utc) & (candles.index <= end_utc)]

if selected.empty:
    st.warning("No candles are available for the selected date range.")
    st.stop()

resampled = resample_ohlcv(selected, timeframe)
resampled = resampled.loc[~resampled.index.duplicated(keep="first")].sort_index()

was_capped = len(resampled) > MAX_CANDLES
if was_capped:
    resampled = resampled.iloc[-MAX_CANDLES:]
    st.warning(
        "The selected range exceeds 5,000 candles. Showing only the most recent "
        "5,000; narrow the range or choose a higher timeframe."
    )

if resampled.empty:
    st.warning("No candles are available for the selected date range.")
    st.stop()

figure = go.Figure(
    data=[
        go.Candlestick(
            x=resampled.index,
            open=resampled["open"],
            high=resampled["high"],
            low=resampled["low"],
            close=resampled["close"],
            name="EUR/USD",
        )
    ]
)

if show_swings:
    full_resampled = cached_resample_ohlcv(candles, timeframe)
    swings = cached_detect_swings(
        full_resampled,
        left=int(left),
        right=int(right),
    )
    visible_swings = swings.reindex(resampled.index)

    high_mask = visible_swings["is_swing_high"].fillna(False)
    low_mask = visible_swings["is_swing_low"].fillna(False)
    high_indexes = resampled.index[high_mask.to_numpy()]
    low_indexes = resampled.index[low_mask.to_numpy()]

    if len(high_indexes):
        high_confirmed = visible_swings.loc[high_indexes, "confirmed_at"]
        high_hover = [
            (
                f"Swing high: {resampled.loc[index, 'high']:.5f}<br>"
                f"Confirmed: {confirmed.strftime('%Y-%m-%d %H:%M UTC')}"
                if pd.notna(confirmed)
                else f"Swing high: {resampled.loc[index, 'high']:.5f}"
            )
            for index, confirmed in high_confirmed.items()
        ]
        figure.add_trace(
            go.Scatter(
                x=high_indexes,
                y=resampled.loc[high_indexes, "high"],
                mode="markers",
                marker={"symbol": "triangle-down", "size": 10, "color": "red"},
                name="Swing highs",
                hovertext=high_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    if len(low_indexes):
        low_confirmed = visible_swings.loc[low_indexes, "confirmed_at"]
        low_hover = [
            (
                f"Swing low: {resampled.loc[index, 'low']:.5f}<br>"
                f"Confirmed: {confirmed.strftime('%Y-%m-%d %H:%M UTC')}"
                if pd.notna(confirmed)
                else f"Swing low: {resampled.loc[index, 'low']:.5f}"
            )
            for index, confirmed in low_confirmed.items()
        ]
        figure.add_trace(
            go.Scatter(
                x=low_indexes,
                y=resampled.loc[low_indexes, "low"],
                mode="markers",
                marker={"symbol": "triangle-up", "size": 10, "color": "green"},
                name="Swing lows",
                hovertext=low_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

figure.update_layout(
    xaxis_rangeslider_visible=False,
    xaxis={"rangebreaks": weekend_rangebreaks(resampled)},
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
)

st.plotly_chart(figure, use_container_width=True)

first_timestamp = resampled.index.min().strftime("%Y-%m-%d %H:%M UTC")
last_timestamp = resampled.index.max().strftime("%Y-%m-%d %H:%M UTC")
st.write(
    f"Displayed range: {first_timestamp} to {last_timestamp} "
    f"| Candle count: {len(resampled)}"
)
st.caption("Bid prices, UTC time.")
if show_swings:
    st.caption("Swings are only confirmed `right` candles after they form.")
