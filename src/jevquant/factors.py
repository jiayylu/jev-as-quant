"""Cross-sectional stock features computed with code (never with a language model).

Every feature for date t uses prices up to the close of t and fundamentals that were public
by then (see `fundamentals_panel`, which lags annual filings by four months).
Features are converted to cross-sectional ranks in [-0.5, 0.5] among the stocks that are
index members on that date, which makes them comparable across time and robust to outliers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# textbook signs: +1 means "higher value, higher expected return" (used by the no-fit composite)
FACTOR_SIGNS = {
    "mom_12_1": +1, "mom_6_1": +1, "rev_1w": -1, "rev_1m": -1, "vol_60d": -1, "beta_1y": -1,
    "to_52w_high": +1, "max_1m": -1, "dollar_vol": -1, "volume_trend": +1,
    "earnings_yield": +1, "book_to_market": +1, "op_profitability": +1, "asset_growth": -1,
    "accruals": -1, "log_mcap": -1,
}


def price_features(close: pd.DataFrame, volume: pd.DataFrame, market: pd.Series) -> dict[str, pd.DataFrame]:
    """Panel (date x ticker) price/volume features."""
    ret = close.pct_change(fill_method=None)
    f = {}
    f["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    f["mom_6_1"] = close.shift(21) / close.shift(126) - 1
    f["rev_1w"] = close / close.shift(5) - 1
    f["rev_1m"] = close / close.shift(21) - 1
    f["vol_60d"] = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    mret = market.pct_change()
    cov = ret.rolling(252, min_periods=126).cov(mret)
    f["beta_1y"] = cov.div(mret.rolling(252, min_periods=126).var(), axis=0)
    f["to_52w_high"] = close / close.rolling(252, min_periods=126).max() - 1
    f["max_1m"] = ret.rolling(21, min_periods=15).max()
    dv = (close * volume).rolling(21, min_periods=15).mean()
    f["dollar_vol"] = np.log(dv.replace(0, np.nan))
    f["volume_trend"] = np.log(volume.rolling(5).mean() / volume.rolling(60, min_periods=40).mean())
    return f


def cross_sectional_rank(panel: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Rank each row among member stocks only, scaled to [-0.5, 0.5]; non-members -> NaN."""
    x = panel.where(members)
    r = x.rank(axis=1, pct=True)
    return r - 0.5


def forward_excess_return(open_: pd.DataFrame, members: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Open(t+1) -> open(t+1+horizon) return minus the member cross-sectional mean (the label)."""
    fwd = open_.shift(-(1 + horizon)) / open_.shift(-1) - 1
    fwd = fwd.where(members)
    return fwd.sub(fwd.mean(axis=1), axis=0)
