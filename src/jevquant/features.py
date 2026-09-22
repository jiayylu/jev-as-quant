"""Technical features. All are computed from data up to and including the row's own close."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "close", "dist_sma20", "dist_sma50", "sma20_slope", "sma50_slope", "rsi14", "ret_5d",
    "ret_20d", "ret_60d", "vol_20d", "vol_ratio", "dd_60d", "volume_ratio", "zscore20",
]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(up.notna())


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """df has open, high, low, close, volume. Returns FEATURE_COLUMNS, NaN during warm-up."""
    c = df["close"]
    logret = np.log(c).diff()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    std20 = c.rolling(20).std()
    vol_20d = logret.rolling(20).std() * np.sqrt(252)
    vol_250d = logret.rolling(250, min_periods=120).std() * np.sqrt(252)
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["dist_sma20"] = c / sma20 - 1
    out["dist_sma50"] = c / sma50 - 1
    out["sma20_slope"] = sma20 / sma20.shift(5) - 1
    out["sma50_slope"] = sma50 / sma50.shift(10) - 1
    out["rsi14"] = rsi(c)
    out["ret_5d"] = c / c.shift(5) - 1
    out["ret_20d"] = c / c.shift(20) - 1
    out["ret_60d"] = c / c.shift(60) - 1
    out["vol_20d"] = vol_20d
    out["vol_ratio"] = vol_20d / vol_250d
    out["dd_60d"] = c / c.rolling(60).max() - 1
    out["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    out["zscore20"] = (c - sma20) / std20
    return out[FEATURE_COLUMNS]
