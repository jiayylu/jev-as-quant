"""Event panels from SEC 8-K press releases, aligned to the first tradable open.

Each release lands on the trading day whose open is the first one after it became public
(US/Eastern acceptance time; before 09:30 -> that day, else the next day). Weekly decisions
are taken at Friday's close, so a release is only ever used after it could have been traded.

Panels (date x ticker, 0 on days without a release):
  pr_count     number of releases
  earn_flag    1 if the 8-K reports results of operations (item 2.02)
  laya_tone    Laya P(bullish) - P(bearish), calibrated as in E3 (summed if several)
  laya_bear    1 if Laya's calibrated call is bearish
  gap          market-adjusted move from the last close before the release to the first tradable
               open: the announcement reaction (known at that open)
plus one panel per Laya question promoted by the factory (q_<name>).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.sec import first_tradable_open


def align_releases(pr: pd.DataFrame, days: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    pr = pr.copy()
    pr["symbol"] = pr["symbol"].str.replace(".", "-", regex=False)
    pr = pr[pr["symbol"].isin(tickers)]
    pr["accepted_et"] = pd.to_datetime(pr["accepted_utc"], utc=True).dt.tz_convert("America/New_York")
    pr["exec_date"] = first_tradable_open(pr["accepted_et"].reset_index(drop=True), days).values
    pr = pr.dropna(subset=["exec_date"])
    # last close strictly before the release became public
    day = pr["accepted_et"].dt.tz_localize(None).dt.normalize().values
    k = np.searchsorted(days.values, day, side="left")
    k_ok = np.minimum(k, len(days) - 1)
    after_close = (pr["accepted_et"].dt.hour * 60 + pr["accepted_et"].dt.minute >= 960).values
    is_td = (k < len(days)) & (days.values[k_ok] == day)
    pr["pre_close_idx"] = np.where(is_td & after_close, k, k - 1)
    pr["exec_idx"] = np.searchsorted(days.values, pr["exec_date"].values)
    return pr.reset_index(drop=True)


def event_panels(pr: pd.DataFrame, days: pd.DatetimeIndex, tickers: list[str], open_: pd.DataFrame,
                 close: pd.DataFrame, member: pd.DataFrame, probs: np.ndarray | None = None,
                 labels=("bullish", "bearish", "neutral"), extra: dict[str, np.ndarray] | None = None) -> dict[str, pd.DataFrame]:
    """Build the event panels; `probs` are calibrated Laya probabilities aligned with `pr` rows."""
    col = {t: j for j, t in enumerate(tickers)}
    shape = (len(days), len(tickers))
    out = {k: np.zeros(shape) for k in ("pr_count", "earn_flag", "laya_tone", "laya_bear", "gap")}
    extra = extra or {}
    for k in extra:
        out[k] = np.zeros(shape)
    O, C = open_.values, close.values
    M = member.values
    for n, r in enumerate(pr.itertuples()):
        i, j, p = r.exec_idx, col[r.symbol], r.pre_close_idx
        out["pr_count"][i, j] += 1
        out["earn_flag"][i, j] = max(out["earn_flag"][i, j], float("2.02" in str(r.items)))
        if probs is not None:
            pb, pbear = probs[n, labels.index("bullish")], probs[n, labels.index("bearish")]
            out["laya_tone"][i, j] += pb - pbear
            out["laya_bear"][i, j] = max(out["laya_bear"][i, j], float(probs[n].argmax() == labels.index("bearish")))
        if 0 <= p < i and np.isfinite(C[p, j]) and np.isfinite(O[i, j]):
            rets = O[i] / C[p] - 1
            m = M[i] & np.isfinite(rets)
            out["gap"][i, j] += rets[j] - (np.nanmean(rets[m]) if m.any() else 0.0)
        for k, v in extra.items():
            out[k][i, j] += v[n]
    return {k: pd.DataFrame(v, index=days, columns=tickers) for k, v in out.items()}
