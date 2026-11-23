"""Streamlit chart viewer for the EUR/USD BID candle data."""

import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_candles
from src.data.resample import resample_ohlcv
from src.app.rangebreaks import weekend_rangebreaks
from src.detectors.fvg import detect_fvgs
from src.detectors.sweeps import detect_liquidity_sweeps
from src.detectors.structure import detect_structure_breaks
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


@st.cache_data
def cached_detect_fvgs(df: pd.DataFrame, min_gap: float) -> pd.DataFrame:
    """Cache FVG detection for the full resampled dataset."""
    return detect_fvgs(df, min_gap=min_gap)


@st.cache_data
def cached_detect_structure_breaks(
    df: pd.DataFrame,
    left: int,
    right: int,
) -> pd.DataFrame:
    """Cache structure-break detection for the full resampled dataset."""
    return detect_structure_breaks(df, left=left, right=right)


@st.cache_data
def cached_detect_liquidity_sweeps(
    df: pd.DataFrame,
    left: int,
    right: int,
    max_age: int | None,
    use_swings: bool,
    use_prev_day: bool,
) -> pd.DataFrame:
    """Cache liquidity-sweep detection for the full resampled dataset."""
    return detect_liquidity_sweeps(
        df,
        left=left,
        right=right,
        max_age=max_age,
        use_swings=use_swings,
        use_prev_day=use_prev_day,
    )


st.set_page_config(page_title="Forex Assistant", layout="wide")
st.sidebar.title("Chart controls")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ("15m", "1h", "4h", "1D"),
    key="timeframe",
)
show_swings = st.sidebar.checkbox(
    "Show swing highs/lows",
    value=False,
    key="show_swings",
)
show_structure = st.sidebar.checkbox(
    "Show structure breaks",
    value=False,
    key="show_structure",
)
show_sweeps = st.sidebar.checkbox(
    "Show liquidity sweeps",
    value=False,
    key="show_sweeps",
)
if show_sweeps:
    use_swing_levels = st.sidebar.checkbox(
        "Swing levels",
        value=True,
        key="use_swing_levels",
    )
    use_prev_day_levels = st.sidebar.checkbox(
        "Previous-day levels",
        value=True,
        key="use_prev_day_levels",
    )
    max_age_input = st.sidebar.number_input(
        "Max level age (candles, 0 = unlimited)",
        min_value=0,
        max_value=2000,
        value=0,
        step=1,
        key="max_level_age",
    )
else:
    use_swing_levels = True
    use_prev_day_levels = True
    max_age_input = 0

if show_swings or show_structure or show_sweeps:
    left = st.sidebar.number_input(
        "Left candles",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="left_candles",
    )
    right = st.sidebar.number_input(
        "Right candles",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="right_candles",
    )
else:
    left = 5
    right = 5
show_fvgs = st.sidebar.checkbox(
    "Show fair value gaps",
    value=False,
    key="show_fvgs",
)
min_gap_pips = st.sidebar.number_input(
    "Min gap (pips)",
    min_value=0.0,
    value=0.0,
    step=0.5,
    key="min_gap_pips",
)
hide_closure_gaps = st.sidebar.checkbox(
    "Hide gaps that span market closures",
    value=True,
    key="hide_closure_gaps",
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
    key="date_range",
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

if show_fvgs:
    full_resampled = cached_resample_ohlcv(candles, timeframe)
    fvgs = cached_detect_fvgs(full_resampled, float(min_gap_pips) * 0.0001)
    first_displayed = resampled.index.min()
    last_displayed = resampled.index.max()
    visible_fvgs = fvgs.loc[
        fvgs["formed_at"].between(first_displayed, last_displayed)
    ].copy()
    if hide_closure_gaps:
        visible_fvgs = visible_fvgs.loc[~visible_fvgs["spans_gap"]]

    if len(visible_fvgs) > 150:
        visible_fvgs = visible_fvgs.iloc[-150:]
        st.warning(
            "More than 150 fair value gaps match this view. "
            "Showing only the 150 most recent."
        )

    marker_x = []
    marker_y = []
    marker_colors = []
    marker_hover = []
    marker_customdata = []

    for _, fvg in visible_fvgs.iterrows():
        closed_through = fvg["closed_through_at"]
        if pd.notna(closed_through):
            end_time = min(closed_through, last_displayed)
        else:
            end_time = last_displayed

        direction = fvg["direction"]
        fill_color = "rgba(0, 180, 0, 0.20)" if direction == "bullish" else "rgba(220, 0, 0, 0.20)"
        touched = fvg["touched_at"]
        touched_text = (
            touched.strftime("%Y-%m-%d %H:%M UTC")
            if pd.notna(touched)
            else "never"
        )
        closed_text = (
            closed_through.strftime("%Y-%m-%d %H:%M UTC")
            if pd.notna(closed_through)
            else "never"
        )
        hover_text = (
            f"Direction: {direction}<br>"
            f"Zone: {fvg['zone_low']:.5f}-{fvg['zone_high']:.5f}<br>"
            f"Size: {fvg['size'] / 0.0001:.1f} pips<br>"
            f"Formed: {fvg['formed_at'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Touched: {touched_text}<br>"
            f"Closed through: {closed_text}"
        )
        figure.add_shape(
            type="rect",
            x0=fvg["c1_time"],
            x1=end_time,
            y0=float(fvg["zone_low"]),
            y1=float(fvg["zone_high"]),
            fillcolor=fill_color,
            line={"width": 0},
            layer="below",
        )
        marker_x.append(fvg["formed_at"])
        marker_y.append((float(fvg["zone_low"]) + float(fvg["zone_high"])) / 2)
        marker_colors.append("green" if direction == "bullish" else "red")
        marker_hover.append(hover_text)
        marker_customdata.append([
            direction,
            float(fvg["zone_low"]),
            float(fvg["zone_high"]),
            float(fvg["size"]),
        ])

    if len(visible_fvgs):
        figure.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                marker={
                    "symbol": "diamond",
                    "size": 7,
                    "color": marker_colors,
                },
                customdata=marker_customdata,
                hovertext=marker_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name="Fair value gaps",
            )
        )

if show_structure:
    full_resampled = cached_resample_ohlcv(candles, timeframe)
    breaks = cached_detect_structure_breaks(
        full_resampled,
        left=int(left),
        right=int(right),
    )
    first_displayed = resampled.index.min()
    last_displayed = resampled.index.max()
    visible_breaks = breaks.loc[
        breaks["broke_at"].between(first_displayed, last_displayed)
    ].copy()

    if len(visible_breaks) > 150:
        visible_breaks = visible_breaks.iloc[-150:]
        st.warning(
            "More than 150 structure breaks match this view. "
            "Showing only the 150 most recent."
        )

    marker_x = []
    marker_y = []
    marker_colors = []
    marker_hover = []

    for _, break_row in visible_breaks.iterrows():
        start_time = max(break_row["swing_time"], first_displayed)
        direction = break_row["direction"]
        line_color = "green" if direction == "bullish" else "red"
        dash = "dash" if break_row["kind"] == "reversal" else "solid"
        figure.add_shape(
            type="line",
            x0=start_time,
            x1=break_row["broke_at"],
            y0=float(break_row["level"]),
            y1=float(break_row["level"]),
            line={"color": line_color, "width": 1.5, "dash": dash},
            layer="above",
        )
        marker_x.append(break_row["broke_at"])
        marker_y.append(float(break_row["level"]))
        marker_colors.append(line_color)
        marker_hover.append(
            f"Direction: {direction}<br>"
            f"Kind: {break_row['kind']}<br>"
            f"Level: {break_row['level']:.5f}<br>"
            f"Swing time: {break_row['swing_time'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Swing confirmed: {break_row['swing_confirmed_at'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Broke at: {break_row['broke_at'].strftime('%Y-%m-%d %H:%M UTC')}"
        )

    if len(visible_breaks):
        figure.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                marker={"symbol": "circle", "size": 7, "color": marker_colors},
                hovertext=marker_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name="Structure breaks",
            )
        )

if show_sweeps:
    full_resampled = cached_resample_ohlcv(candles, timeframe)
    max_age = None if int(max_age_input) == 0 else int(max_age_input)
    sweeps = cached_detect_liquidity_sweeps(
        full_resampled,
        left=int(left),
        right=int(right),
        max_age=max_age,
        use_swings=bool(use_swing_levels),
        use_prev_day=bool(use_prev_day_levels),
    )
    first_displayed = resampled.index.min()
    last_displayed = resampled.index.max()
    visible_sweeps = sweeps.loc[
        sweeps["swept_at"].between(first_displayed, last_displayed)
    ].copy()

    if len(visible_sweeps) > 150:
        visible_sweeps = visible_sweeps.iloc[-150:]
        st.warning(
            "More than 150 liquidity sweeps match this view. "
            "Showing only the 150 most recent."
        )

    marker_x = []
    marker_y = []
    marker_colors = []
    marker_hover = []

    for _, sweep in visible_sweeps.iterrows():
        start_time = max(sweep["level_known_at"], first_displayed)
        line_color = "orange" if sweep["side"] == "high" else "blue"
        line_dash = "dot" if sweep["source"] == "swing" else "dash"
        line_width = 1 if sweep["source"] == "swing" else 1.5
        figure.add_shape(
            type="line",
            x0=start_time,
            x1=sweep["swept_at"],
            y0=float(sweep["level"]),
            y1=float(sweep["level"]),
            line={
                "color": line_color,
                "width": line_width,
                "dash": line_dash,
            },
            layer="above",
        )
        marker_x.append(sweep["swept_at"])
        marker_y.append(float(sweep["extreme"]))
        marker_colors.append(line_color)
        marker_hover.append(
            f"Source: {sweep['source']}<br>"
            f"Side: {sweep['side']}<br>"
            f"Level: {sweep['level']:.5f}<br>"
            f"Level known: {sweep['level_known_at'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Swept at: {sweep['swept_at'].strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Extreme: {sweep['extreme']:.5f}<br>"
            f"Close: {sweep['close']:.5f}<br>"
            f"Depth: {sweep['depth'] / 0.0001:.1f} pips"
        )

    if len(visible_sweeps):
        figure.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                marker={"symbol": "circle", "size": 7, "color": marker_colors},
                hovertext=marker_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                name="Liquidity sweeps",
            )
        )

figure.update_layout(
    xaxis_rangeslider_visible=False,
    xaxis={"rangebreaks": weekend_rangebreaks(resampled.index, timeframe)},
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
if show_fvgs:
    st.caption("A gap is only known after its third candle closes.")
    st.caption(
        "Min gap uses 0.0001 price units per pip, assuming a 4-decimal pair like EUR/USD."
    )
if show_structure:
    st.caption(
        "A break needs a candle CLOSE beyond a swing that has already been confirmed."
    )
if show_sweeps:
    st.caption(
        "A sweep is a wick through a level that closes back inside it. "
        "It is only known at the sweeping candle's close."
    )
    st.caption(
        "Swing levels use dotted thin lines; previous-day levels use dashed wider lines."
    )
