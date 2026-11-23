from src.data.loader import load_candles_10y
from src.data.splits import development_set
from src.data.resample_et import resample_ohlcv_et
from src.detectors.session_levels import detect_session_levels

df = development_set(load_candles_10y())
h4 = resample_ohlcv_et(df, "4h")
d1 = resample_ohlcv_et(df, "1D")
lv = detect_session_levels(df)
print("range", df.index.min(), "to", df.index.max())
print("4h bins", len(h4), "partial(<16)", int((h4.n_candles < 16).sum()))
print("1D bins", len(d1), "partial(<96)", int((d1.n_candles < 96).sum()))
print("trading days", len(lv), "NaN overnight", int(lv.overnight_high.isna().sum()))