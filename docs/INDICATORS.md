# Indicators

## ATR

`src.indicators.basic.atr(df, period=14)` returns Wilder-smoothed average true
range. The first true range is high minus low; later true ranges include the
previous close. The first ATR is the mean of the initial `period` true ranges.

## Stochastic

`src.indicators.basic.stochastic(df, k_period=5, smooth_k=3, smooth_d=3)`
returns `k_raw`, `k`, and `d`. The raw value uses the rolling high/low range
including the current candle. `%K` and `%D` are full-window simple moving
averages; unavailable windows and zero-range raw values remain NaN.

Both functions require sorted, timezone-aware UTC candle indexes and do not
mutate their inputs. Timestamp arithmetic uses a nanosecond-normalized index.
