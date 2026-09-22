"""The research panel: aligned daily prices for every stock that was ever an index member.

Rows are SPY trading days, columns tickers. `member` marks the days a ticker was actually in
the S&P 500 (point-in-time), with renamed companies merged under their current symbol.
Every alpha, label and portfolio is computed on this panel, and only members are ever ranked
or traded on a given day.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.sec import sp500_from_wikitext
from ..data.universe import Membership, detect_renames

# verified by hand against SEC former names; the date heuristic cannot see multi-step renames
EXTRA_RENAMES = {"SYMC": "GEN", "NLOK": "GEN", "HCP": "DOC", "PEAK": "DOC", "HRS": "LHX", "BHGE": "BKR",
                 "CBS": "PSKY", "VIAC": "PSKY", "PARA": "PSKY", "MYL": "VTRS"}
# the date heuristic's false positives, rejected after the same check
REJECTED_RENAMES = {"CA", "UCC"}


@dataclass
class Panel:
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    member: pd.DataFrame  # bool
    spy: pd.DataFrame  # open/close of SPY
    info: dict = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def ew_market(self) -> pd.Series:
        """Equal-weight member index (close to close)."""
        r = self.close.pct_change(fill_method=None).where(self.member.shift(1, fill_value=False))
        return (1 + r.mean(axis=1).fillna(0)).cumprod()


def build_panel(data_cache: str | Path, start: str = "2013-01-01", end: str = "2026-09-22") -> Panel:
    dc = Path(data_cache)
    mem = Membership(dc / "sp500" / "components.csv")
    current = sp500_from_wikitext(sorted((dc / "sec").glob("sp500_*.wiki"))[-1].read_text())
    renames = {k: v for k, v in detect_renames(mem, current).items() if k not in REJECTED_RENAMES}
    renames.update(EXTRA_RENAMES)
    renames = {k.replace(".", "-"): v.replace(".", "-") for k, v in renames.items()}

    spy = pd.read_csv(dc / "prices_pit" / "SPY.csv", index_col=0, parse_dates=True).loc[start:end]
    days = spy.index
    frames = {}
    for p in sorted((dc / "prices_pit").glob("*.csv")):
        t = p.stem
        if t.startswith("_") or t == "SPY":
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated()].reindex(days)
        frames[t] = df
    O = pd.DataFrame({t: f["open"] for t, f in frames.items()})
    C = pd.DataFrame({t: f["close"] for t, f in frames.items()})
    V = pd.DataFrame({t: f["volume"] for t, f in frames.items()})

    # membership: sample the dated lists on every trading day, merging renamed symbols
    idx = np.searchsorted(mem.dates, days.values, side="right") - 1
    col = {t: j for j, t in enumerate(C.columns)}
    M = np.zeros((len(days), len(C.columns)), dtype=bool)
    for i, k in enumerate(idx):
        if k < 0:
            continue
        for t in mem.lists[k]:
            t = t.replace(".", "-")
            t = renames.get(t, t)
            j = col.get(t)
            if j is not None:
                M[i, j] = True
    member = pd.DataFrame(M, index=days, columns=C.columns) & C.notna() & O.notna()

    # coverage: share of point-in-time member-days we can actually price
    total = sum(len(mem.lists[k]) for k in idx if k >= 0)
    info = {"renames": renames, "member_days": int(total), "priced_member_days": int(member.values.sum()),
            "coverage": float(member.values.sum() / total) if total else float("nan")}
    return Panel(open=O, close=C, volume=V, member=member, spy=spy[["open", "close"]], info=info)
