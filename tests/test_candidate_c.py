from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.backtest.engine import run_backtest
from src.data.resample_et import resample_ohlcv_et
from src.indicators.basic import atr
from src.strategies.candidate_c import (
    PLAN_COLUMNS,
    count_candidate_c_entries,
    find_candidate_c_plans,
)


START = pd.Timestamp("2024-01-08 22:00", tz="UTC")
PIP = 0.0001


def expand(rows: list[tuple[float, float, float, float]], start=START) -> pd.DataFrame:
    times = pd.date_range(start, periods=len(rows) * 16, freq="15min", tz="UTC").as_unit("ns")
    records = []
    for k, (open_, high, low, close) in enumerate(rows):
        for j in range(16):
            records.append((open_, high, low, close, 1.0))
    return pd.DataFrame(records, index=times, columns=["open", "high", "low", "close", "volume"])


BASE = [(1.1000, 1.1005, 1.0995, 1.1000)] * 20
T = (1.1030, 1.1040, 1.1020, 1.1030)
F = (1.1018, 1.1022, 1.1014, 1.1018)


def hc1() -> pd.DataFrame:
    rows = list(BASE)
    rows += [
        (1.1000, 1.1030, 1.0998, 1.1025),
        (1.1026, 1.1040, 1.1020, 1.1030),
    ]
    rows += [T] * 9
    rows += [
        (1.1030, 1.1032, 1.1015, 1.1018),
        F,
        F,
    ]
    return expand(rows)


def test_expand_resamples_to_source_rows() -> None:
    source = list(BASE) + [(1.1000, 1.1030, 1.0998, 1.1025)]
    frame = expand(source)
    h4 = resample_ohlcv_et(frame, "4h")
    assert len(h4) == len(source)
    for position, row in enumerate(source):
        assert h4.iloc[position]["open"] == pytest.approx(row[0], abs=1e-9)
        assert h4.iloc[position]["high"] == pytest.approx(row[1], abs=1e-9)
        assert h4.iloc[position]["low"] == pytest.approx(row[2], abs=1e-9)
        assert h4.iloc[position]["close"] == pytest.approx(row[3], abs=1e-9)


def test_hc1_find_engine_and_count() -> None:
    df = hc1()
    plans = find_candidate_c_plans(df)
    assert len(plans) == 1
    plan = plans.iloc[0]
    assert list(plans.columns) == list(PLAN_COLUMNS)
    assert plan["side"] == "long"
    assert plan["start_time"] == START + pd.Timedelta(hours=84)
    assert plan["stop_price"] == pytest.approx(1.10038, abs=1e-9)
    assert plan["atr_n"] == pytest.approx(0.00111, abs=1e-9)
    assert plan["channel_level"] == pytest.approx(1.1005, abs=1e-9)
    assert plan["exit_signal_time"] == START + pd.Timedelta(hours=124)

    results = run_backtest(df, plans, spread_pips=1.0, slippage_pips=0.2)
    trade = results.iloc[0]
    assert trade["status"] == "filled"
    assert trade["fill_time"] == START + pd.Timedelta(hours=84)
    assert trade["entry_price"] == pytest.approx(1.10272, abs=1e-9)
    assert trade["risk_pips"] == pytest.approx(23.4, abs=1e-9)
    assert trade["exit_time"] == START + pd.Timedelta(hours=128)
    assert trade["exit_price"] == pytest.approx(1.10178, abs=1e-9)
    assert trade["exit_reason"] == "time"
    assert trade["r_multiple"] == pytest.approx(-47 / 117, abs=1e-9)
    assert trade["pnl_pips"] == pytest.approx(-9.4, abs=1e-9)
    assert trade["bars_held"] == 176

    counted = count_candidate_c_entries(df, plans)
    assert len(counted) <= 2
    assert counted.iloc[0]["exit_reason"] == "time"


def test_hc2_mirror_short() -> None:
    rows = list(BASE)
    rows += [
        (1.1000, 1.1002, 1.0970, 1.0975),
        (1.0974, 1.0980, 1.0960, 1.0970),
    ]
    rows += [(1.0970, 1.0980, 1.0960, 1.0970)] * 9
    rows += [
        (1.0970, 1.0985, 1.0968, 1.0982),
        (1.0982, 1.0986, 1.0978, 1.0982),
        (1.0982, 1.0986, 1.0978, 1.0982),
    ]
    df = expand(rows)
    plans = find_candidate_c_plans(df)
    assert len(plans) == 1
    plan = plans.iloc[0]
    assert plan["side"] == "short"
    assert plan["stop_price"] == pytest.approx(1.09962, abs=1e-9)
    assert plan["channel_level"] == pytest.approx(1.0995, abs=1e-9)

    trade = run_backtest(df, plans, spread_pips=1.0, slippage_pips=0.2).iloc[0]
    assert trade["entry_price"] == pytest.approx(1.09738, abs=1e-9)
    assert trade["risk_pips"] == pytest.approx(22.4, abs=1e-9)
    assert trade["exit_price"] == pytest.approx(1.09832, abs=1e-9)
    assert trade["r_multiple"] == pytest.approx(-47 / 112, abs=1e-9)
    assert trade["pnl_pips"] == pytest.approx(-9.4, abs=1e-9)


@pytest.mark.parametrize(
    ("close", "has_plan"),
    [(1.1005, False), (1.10051, True), (1.1004, False)],
)
def test_strict_entry(close: float, has_plan: bool) -> None:
    rows = list(BASE)
    rows += [
        (1.1000, 1.1030, 1.0998, close),
        (1.1026, 1.1040, 1.1020, 1.1030),
    ]
    rows += [T] * 9
    rows += [(1.1030, 1.1032, 1.1015, 1.1018), F, F]
    plans = find_candidate_c_plans(expand(rows))
    assert len(plans) == (1 if has_plan else 0)


@pytest.mark.parametrize(
    ("close", "has_plan"),
    [(1.0995, False), (1.09949, True)],
)
def test_strict_short_entry(close: float, has_plan: bool) -> None:
    rows = list(BASE) + [
        (1.1000, 1.1002, 1.0970, close),
        (1.0974, 1.0980, 1.0960, 1.0970),
    ]
    rows += [(1.0970, 1.0980, 1.0960, 1.0970)] * 9
    mirror_f = (1.0982, 1.0986, 1.0978, 1.0982)
    rows += [(1.0970, 1.0985, 1.0968, 1.0982), mirror_f, mirror_f]
    assert len(find_candidate_c_plans(expand(rows))) == (1 if has_plan else 0)


def test_partial_bin_and_last_candle_rules() -> None:
    df = hc1()
    cut = 16 * 31 + 5
    plans = find_candidate_c_plans(df.iloc[:cut])
    assert len(plans) == 1
    assert pd.isna(plans.iloc[0]["exit_signal_time"])
    assert plans.iloc[0]["time_exit_at"] > df.index[cut - 1]

    cut21 = 16 * 21 + 1
    assert find_candidate_c_plans(df.iloc[:cut21]).empty
    cut21 += 1
    plans = find_candidate_c_plans(df.iloc[:cut21])
    assert len(plans) == 1
    assert pd.isna(plans.iloc[0]["exit_signal_time"])
    assert plans.iloc[0]["time_exit_at"] == df.index[cut21 - 1] + pd.Timedelta(days=1)


def test_validation_and_unit_precision() -> None:
    df = hc1()
    for kwargs in (
        {"entry_period": 0},
        {"exit_period": 0},
        {"atr_period": 0},
        {"atr_mult": 0},
    ):
        with pytest.raises(ValueError):
            find_candidate_c_plans(df, **kwargs)
    with pytest.raises(ValueError):
        find_candidate_c_plans(df.set_axis(df.index.tz_localize(None)))
    with pytest.raises(ValueError):
        find_candidate_c_plans(df.iloc[::-1])

    no_volume = df.drop(columns=["volume"])
    plans = find_candidate_c_plans(no_volume)
    assert len(plans) == 1

    us_df = df.copy()
    us_df.index = us_df.index.as_unit("us")
    us_plans = find_candidate_c_plans(us_df)
    pd.testing.assert_frame_equal(
        plans.reset_index(drop=True),
        us_plans.reset_index(drop=True),
        rtol=0,
        atol=1e-9,
    )

    empty = pd.DataFrame(
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    assert find_candidate_c_plans(empty).empty


def test_strict_exit_and_reversal_busy() -> None:
    rows = list(BASE)
    rows += [
        (1.1000, 1.1030, 1.0998, 1.1025),
        (1.1026, 1.1027, 1.0990, 1.0992),
        (1.0992, 1.0996, 1.0988, 1.0992),
    ]
    df = expand(rows)
    plans = find_candidate_c_plans(df)
    assert len(plans) == 2
    assert plans.iloc[0]["side"] == "long"
    assert plans.iloc[0]["exit_signal_time"] == START + pd.Timedelta(hours=84)
    assert plans.iloc[1]["side"] == "short"
    assert pd.isna(plans.iloc[1]["exit_signal_time"])
    counted = count_candidate_c_entries(df, plans)
    assert len(counted) >= 1


def test_t7_busy_rule() -> None:
    rows = list(BASE)
    rows += [
        (1.1000, 1.1030, 1.0998, 1.1025),
        (1.1026, 1.1040, 1.1020, 1.1030),
    ]
    rows += [T] * 8
    rows += [
        (1.1030, 1.1060, 1.1025, 1.1050),
        (1.1030, 1.1032, 1.1015, 1.1018),
        F,
        F,
    ]
    df = expand(rows)
    plans = find_candidate_c_plans(df)
    assert len(plans) == 2
    assert plans.iloc[0]["side"] == "long"
    assert plans.iloc[1]["side"] == "long"
    counted = count_candidate_c_entries(df, plans)
    assert len(counted) == 1


def test_strict_exit_equal_is_not_an_exit() -> None:
    rows = list(BASE)
    rows += [
        (1.1000, 1.1030, 1.0998, 1.1025),
        (1.1026, 1.1040, 1.1020, 1.1030),
    ]
    rows += [T] * 9
    rows += [(1.1030, 1.1032, 1.1020, 1.1020)]
    plans = find_candidate_c_plans(expand(rows))
    assert len(plans) == 1
    assert pd.isna(plans.iloc[0]["exit_signal_time"])


ET = ZoneInfo("America/New_York")
ENTRY_COLUMNS = (
    "plan_id",
    "side",
    "start_time",
    "stop_price",
    "atr_n",
    "channel_level",
    "signal_time",
)


def synthetic_candidate_c_series(
    seed: int,
    start: str,
    end: str,
    sigma: float = 0.0008,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    full_index = pd.date_range(start, end, freq="15min", tz="UTC")
    local = full_index.tz_convert(ET)
    minutes = local.hour * 60 + local.minute
    weekend = (
        ((local.day_of_week == 4) & (minutes >= 17 * 60))
        | (local.day_of_week == 5)
        | ((local.day_of_week == 6) & (minutes < 17 * 60))
    )
    index = full_index[~np.asarray(weekend)]
    changes = rng.normal(0.0, sigma, len(index))
    close = 1.1 + np.cumsum(changes)
    open_price = np.r_[1.1, close[:-1]]
    wick_high = rng.uniform(0.00005, 0.0004, len(index))
    wick_low = rng.uniform(0.00005, 0.0004, len(index))
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + wick_high,
            "low": np.minimum(open_price, close) - wick_low,
            "close": close,
            "volume": np.ones(len(index)),
        },
        index=index.as_unit("ns"),
    )


def empty_candidate_c_reference() -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "plan_id": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"),
            "order_type": pd.Series(dtype="object"),
            "start_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "entry_price": pd.Series(dtype="float64"),
            "stop_price": pd.Series(dtype="float64"),
            "target_r": pd.Series(dtype="float64"),
            "target_price": pd.Series(dtype="float64"),
            "valid_candles": pd.Series(dtype="int64"),
            "time_exit_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "no_target": pd.Series(dtype="bool"),
            "exit_mode": pd.Series(dtype="object"),
            "signal_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "atr_n": pd.Series(dtype="float64"),
            "channel_level": pd.Series(dtype="float64"),
            "exit_signal_time": pd.Series(dtype="datetime64[ns, UTC]"),
        },
        columns=PLAN_COLUMNS,
    )
    return result


def reference_candidate_c(
    df: pd.DataFrame,
    entry_period: int,
    exit_period: int,
    atr_period: int,
    atr_mult: float,
) -> pd.DataFrame:
    data = df.copy()
    data.index = data.index.as_unit("ns")
    if "volume" not in data:
        data["volume"] = np.ones(len(data), dtype=np.float64)
    h4 = resample_ohlcv_et(data, "4h")
    atr_values = atr(h4, atr_period)
    last_ltf_open = data.index[-1].as_unit("ns")
    closed = h4["close_time"] <= last_ltf_open + pd.Timedelta(minutes=15)
    rows = []
    nan_atr = 0
    long_signals = 0
    short_signals = 0
    for c in range(entry_period, len(h4)):
        if not bool(closed.iloc[c]):
            continue
        n_value = atr_values.iloc[c]
        if pd.isna(n_value):
            nan_atr += 1
            continue
        if c + 1 >= len(h4):
            continue
        long_level = float(h4["high"].iloc[c-entry_period:c].max())
        short_level = float(h4["low"].iloc[c-entry_period:c].min())
        close = float(h4["close"].iloc[c])
        side = None
        channel_level = np.nan
        if close > long_level:
            side = "long"
            channel_level = long_level
            long_signals += 1
        elif close < short_level:
            side = "short"
            channel_level = short_level
            short_signals += 1
        if side is None:
            continue

        start_pos = int(
            np.searchsorted(data.index.asi8, h4.index[c+1].value, side="left")
        )
        if start_pos >= len(data) or start_pos == len(data) - 1:
            continue
        start_time = data.index[start_pos]
        open4 = float(h4["open"].iloc[c+1])
        n = float(n_value)
        stop = open4 - atr_mult*n if side == "long" else open4 + atr_mult*n

        exit_signal_time = pd.NaT
        time_exit_at = data.index[-1] + pd.Timedelta(days=1)
        for e in range(c+1, len(h4)):
            if not bool(closed.iloc[e]) or e < exit_period:
                continue
            if side == "long":
                exit_level = float(h4["low"].iloc[e-exit_period:e].min())
                hit = float(h4["close"].iloc[e]) < exit_level
            else:
                exit_level = float(h4["high"].iloc[e-exit_period:e].max())
                hit = float(h4["close"].iloc[e]) > exit_level
            if hit:
                exit_signal_time = h4.index[e]
                if e+1 < len(h4):
                    exit_pos = int(
                        np.searchsorted(data.index.asi8, h4.index[e+1].value, side="left")
                    )
                    if exit_pos < len(data):
                        time_exit_at = data.index[exit_pos]
                break

        rows.append(
            {
                "plan_id": start_time.isoformat(),
                "side": side,
                "order_type": "market",
                "start_time": start_time,
                "entry_price": np.nan,
                "stop_price": float(stop),
                "target_r": np.nan,
                "target_price": np.nan,
                "valid_candles": 1,
                "time_exit_at": time_exit_at,
                "no_target": True,
                "exit_mode": "time_open",
                "signal_time": h4.index[c],
                "atr_n": n,
                "channel_level": float(channel_level),
                "exit_signal_time": exit_signal_time,
            }
        )
    if not rows:
        return empty_candidate_c_reference()
    result = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    for column in ("start_time", "time_exit_at", "signal_time", "exit_signal_time"):
        result[column] = pd.to_datetime(result[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    for column in (
        "entry_price", "stop_price", "target_r", "target_price",
        "atr_n", "channel_level",
    ):
        result[column] = result[column].astype("float64")
    result["valid_candles"] = result["valid_candles"].astype("int64")
    result["no_target"] = result["no_target"].astype("bool")
    for column in ("plan_id", "side", "order_type", "exit_mode"):
        result[column] = result[column].astype("object")
    return result.sort_values("start_time", kind="stable").reset_index(drop=True)


@pytest.mark.parametrize(
    ("seed", "start", "end"),
    [
        (11, "2023-01-02", "2023-06-30 23:45"),
        (22, "2023-01-02", "2023-06-30 23:45"),
        (5, "2023-09-01", "2023-12-29 23:45"),
        (17, "2023-09-01", "2023-12-29 23:45"),
    ],
)
@pytest.mark.parametrize(
    "params",
    [(20, 10, 20, 2.0), (10, 5, 10, 1.5)],
)
def test_seeded_candidate_c_reference_engine_and_prefix(
    seed: int,
    start: str,
    end: str,
    params: tuple[int, int, int, float],
) -> None:
    df = synthetic_candidate_c_series(seed, start, end)
    h4 = resample_ohlcv_et(df, "4h")
    closed_count = int(
        (h4["close_time"] <= df.index[-1] + pd.Timedelta(minutes=15)).sum()
    )
    atr_values = atr(h4, params[2])
    nan_atr = int(atr_values.isna().sum())

    expected = reference_candidate_c(df, *params)
    actual = find_candidate_c_plans(df, *params)
    pd.testing.assert_frame_equal(actual, expected, rtol=0, atol=1e-9)
    assert len(expected) >= 8
    long_signals = 0
    short_signals = 0
    for c in range(params[0], len(h4)):
        if h4["close_time"].iloc[c] > df.index[-1] + pd.Timedelta(minutes=15):
            continue
        n = atr_values.iloc[c]
        if pd.isna(n):
            continue
        if float(h4["close"].iloc[c]) > float(h4["high"].iloc[c-params[0]:c].max()):
            long_signals += 1
        if float(h4["close"].iloc[c]) < float(h4["low"].iloc[c-params[0]:c].min()):
            short_signals += 1
    print(
        f"seed={seed} candles={len(df)} closed_4h={closed_count} "
        f"nan_atr={nan_atr} signals={long_signals+short_signals} "
        f"long_signals={long_signals} short_signals={short_signals} "
        f"plans={len(expected)} "
        f"entries={len(count_candidate_c_entries(df, actual))}"
    )

    engine = run_backtest(
        df,
        actual,
        spread_pips=1.0,
        slippage_pips=0.2,
    )
    filled = engine.loc[
        engine["status"] == "filled",
        ["plan_id", "side", "fill_time", "exit_time", "exit_reason"],
    ].reset_index(drop=True)
    counted = count_candidate_c_entries(df, actual).loc[
        :, ["plan_id", "side", "fill_time", "exit_time", "exit_reason"]
    ].reset_index(drop=True)
    filled["plan_id"] = filled["plan_id"].astype("object")
    counted["plan_id"] = counted["plan_id"].astype("object")
    filled["side"] = filled["side"].astype("object")
    counted["side"] = counted["side"].astype("object")
    filled["exit_reason"] = filled["exit_reason"].astype("object")
    counted["exit_reason"] = counted["exit_reason"].astype("object")
    pd.testing.assert_frame_equal(filled, counted, rtol=0, atol=1e-9)

    entry_col = ENTRY_COLUMNS
    full_entry = actual.loc[:, entry_col].reset_index(drop=True)
    h4_closed = h4.loc[
        h4["close_time"] <= df.index[-1] + pd.Timedelta(minutes=15)
    ]

    all_boundary_cutoffs = []
    for bin_index, bin_start in enumerate(h4_closed.index):
        start_pos = int(
            np.searchsorted(df.index.asi8, bin_start.value, side="left")
        )
        if bin_index + 1 < len(h4_closed):
            end_pos = int(
                np.searchsorted(
                    df.index.asi8,
                    h4_closed.index[bin_index+1].value,
                    side="left",
                )
            )
        else:
            end_pos = len(df)
        boundary = end_pos - 1
        if start_pos <= boundary < len(df):
            all_boundary_cutoffs.append(boundary)
    all_boundary_cutoffs = sorted(set(all_boundary_cutoffs))
    assert len(all_boundary_cutoffs) >= 6
    boundary_cutoffs = list(np.asarray(all_boundary_cutoffs)[
        np.linspace(0, len(all_boundary_cutoffs)-1, 6, dtype=int)
    ])
    mid_cutoffs = []
    for bin_start in h4_closed.index:
        start_pos = int(
            np.searchsorted(df.index.asi8, bin_start.value, side="left")
        )
        if start_pos + 2 < len(df) and start_pos + 2 < start_pos + 16:
            mid_cutoffs.append(start_pos + 2)
    mid_cutoffs = sorted(set(mid_cutoffs))
    if len(mid_cutoffs) >= 6:
        mid_cutoffs = list(np.asarray(mid_cutoffs)[
            np.linspace(0, len(mid_cutoffs)-1, 6, dtype=int)
        ])
    assert len(boundary_cutoffs) == 6
    assert len(mid_cutoffs) == 6

    cutoffs = boundary_cutoffs + mid_cutoffs
    for cutoff in cutoffs:
        truncated = df.iloc[:cutoff+1]
        truncated_plans = find_candidate_c_plans(truncated, *params)
        restricted_full = actual.loc[
            actual["start_time"] < truncated.index[-1]
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            truncated_plans.loc[:, entry_col].reset_index(drop=True),
            restricted_full.loc[:, entry_col].reset_index(drop=True),
            rtol=0,
            atol=1e-9,
        )
        truncated_h4 = resample_ohlcv_et(truncated, "4h")
        truncated_closed = truncated_h4.loc[
            truncated_h4["close_time"]
            <= truncated.index[-1] + pd.Timedelta(minutes=15)
        ]
        last_closed_open = (
            truncated_closed.index[-1]
            if len(truncated_closed)
            else pd.NaT
        )
        assert len(truncated_plans) == len(restricted_full)
        for row in range(len(truncated_plans)):
            full_exit_signal = restricted_full.iloc[row]["exit_signal_time"]
            if pd.isna(full_exit_signal) or (
                pd.isna(last_closed_open)
                or full_exit_signal > last_closed_open
            ):
                assert pd.isna(truncated_plans.iloc[row]["exit_signal_time"])
            else:
                assert truncated_plans.iloc[row]["exit_signal_time"] == full_exit_signal
            full_time_exit = restricted_full.iloc[row]["time_exit_at"]
            truncated_time_exit = truncated_plans.iloc[row]["time_exit_at"]
            if full_time_exit <= truncated.index[-1]:
                assert truncated_time_exit == full_time_exit
            else:
                assert truncated_time_exit > truncated.index[-1]

    for cutoff in mid_cutoffs:
        start_pos = int(
            np.searchsorted(df.index.asi8, df.index[cutoff].value, side="left")
        )
        containing_bin = h4_closed.index[
            h4_closed.index.get_indexer([df.index[cutoff]], method="pad")
        ][0] if False else None
        bin_starts = h4_closed.index
        hits = np.flatnonzero(bin_starts.asi8 <= df.index[cutoff].value)
        if not len(hits):
            continue
        bin_index = int(hits[-1])
        bin_start = bin_starts[bin_index]
        bin_pos = int(np.searchsorted(df.index.asi8, bin_start.value, side="left"))
        if cutoff == bin_pos + 1:
            previous_boundary = bin_pos - 1
            if previous_boundary < 0:
                continue
            boundary = previous_boundary
        else:
            if bin_index == 0:
                continue
            previous_bin = bin_starts[bin_index-1]
            previous_bin_pos = int(
                np.searchsorted(df.index.asi8, previous_bin.value, side="left")
            )
            boundary = previous_bin_pos + 15
        boundary_plans = find_candidate_c_plans(
            df.iloc[:boundary+1], *params
        ).loc[:, entry_col].reset_index(drop=True)
        mid_plans = find_candidate_c_plans(
            df.iloc[:cutoff+1], *params
        ).loc[:, entry_col].reset_index(drop=True)
        if cutoff == bin_pos + 1:
            pd.testing.assert_frame_equal(
                mid_plans, boundary_plans, rtol=0, atol=1e-9
            )
        else:
            base_ids = set(boundary_plans["plan_id"])
            extra = mid_plans.loc[
                ~mid_plans["plan_id"].isin(base_ids)
            ].reset_index(drop=True)
            assert len(extra) <= 1
            if len(extra):
                assert extra.iloc[0]["start_time"] == bin_start
                assert extra.iloc[0]["signal_time"] == h4_closed.index[max(bin_index-1, 0)]


def test_seeded_unit_precision() -> None:
    df = synthetic_candidate_c_series(
        11,
        "2023-01-02",
        "2023-06-30 23:45",
    )
    params = (20, 10, 20, 2.0)
    ns_plans = find_candidate_c_plans(df, *params)
    ns_entries = count_candidate_c_entries(df, ns_plans)
    us_df = df.copy()
    us_df.index = us_df.index.as_unit("us")
    us_plans = find_candidate_c_plans(us_df, *params)
    us_entries = count_candidate_c_entries(us_df, us_plans)
    pd.testing.assert_frame_equal(
        ns_plans, us_plans, rtol=0, atol=1e-9
    )
    pd.testing.assert_frame_equal(
        ns_entries, us_entries, rtol=0, atol=1e-9
    )


