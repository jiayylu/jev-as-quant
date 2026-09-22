"""Alpha lab: time splits, IC evaluation, acceptance gates and the research ledger.

Splits (fixed before any alpha was looked at):
    discovery   2014-01-01 .. 2020-12-31   alphas are generated and screened here
    validation  2021-01-01 .. 2023-12-31   an alpha must hold up here, untouched by screening
    holdout     2024-01-01 .. end          locked: only the final combined model is run on it

Label: open(t+1) -> open(t+6) return minus the member average, i.e. what a weekly strategy
deciding at Friday's close and trading at Monday's open would earn over the next week.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .panel import Panel

SPLITS = {"discovery": ("2014-01-01", "2020-12-31"), "validation": ("2021-01-01", "2023-12-31"),
          "holdout": ("2024-01-01", "2099-12-31")}
COST_BPS = 10.0  # per side, commission + slippage, large caps


def weekly_dates(p: Panel, start: str = "2014-01-01") -> pd.DatetimeIndex:
    d = p.dates[p.dates >= start]
    s = pd.Series(d, index=d)
    return pd.DatetimeIndex(s.groupby(d.to_period("W-FRI")).last().values)


def rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Spearman correlation per row over the entries finite in both."""
    m = a.notna() & b.notna()
    ra = a.where(m).rank(axis=1)
    rb = b.where(m).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    out = num / den
    out[m.sum(axis=1) < 30] = np.nan
    return out


@dataclass
class Lab:
    panel: Panel
    horizon: int = 5

    def __post_init__(self):
        p = self.panel
        self.dates = weekly_dates(p)
        O = p.open
        fwd = O.shift(-(1 + self.horizon)) / O.shift(-1) - 1
        mem = p.member
        fwd = fwd.where(mem)
        self.label_full = fwd.sub(fwd.mean(axis=1), axis=0)
        self.label = self.label_full.reindex(self.dates)
        self.member_w = mem.reindex(self.dates)
        # when the label of decision date t is fully known (open of t+1+horizon)
        pos = np.searchsorted(p.dates.values, self.dates.values)
        known_idx = np.minimum(pos + 1 + self.horizon, len(p.dates) - 1)
        self.label_known_on = pd.Series(p.dates.values[known_idx], index=self.dates)

    def split_mask(self, name: str) -> np.ndarray:
        a, b = SPLITS[name]
        return (self.dates >= pd.Timestamp(a)) & (self.dates <= pd.Timestamp(b))

    def weekly(self, alpha: pd.DataFrame) -> pd.DataFrame:
        return alpha.reindex(self.dates).where(self.member_w)

    def ic_series(self, alpha: pd.DataFrame) -> pd.Series:
        return rowwise_spearman(self.weekly(alpha), self.label)

    def evaluate(self, alpha: pd.DataFrame, splits=("discovery", "validation")) -> dict:
        A = self.weekly(alpha)
        ic = rowwise_spearman(A, self.label)
        r = A.rank(axis=1, pct=True)
        # turnover of the top/bottom deciles from week to week
        top, bot = r >= 0.9, r <= 0.1
        churn = ((top != top.shift()).sum(axis=1) / top.sum(axis=1).replace(0, np.nan)).fillna(0) / 2
        ls = (self.label.where(top).mean(axis=1) - self.label.where(bot).mean(axis=1))
        ls_net = ls - churn * 2 * 2 * COST_BPS / 1e4  # both legs, buy and sell
        out = {}
        for s in splits:
            m = self.split_mask(s)
            x = ic[m].dropna()
            y = ls_net[m].dropna()
            out[s] = {
                "ic": float(x.mean()) if len(x) else float("nan"),
                "ic_t": float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else float("nan"),
                "ic_hit": float((x > 0).mean()) if len(x) else float("nan"),
                "ls_net_ann": float(y.mean() * 52) if len(y) else float("nan"),
                "ls_net_sharpe": float(y.mean() / y.std() * math.sqrt(52)) if len(y) > 5 and y.std() > 0 else float("nan"),
                "turnover": float(churn[m].mean()),
                "coverage": float(A[m].notna().sum(axis=1).mean() / max(1, self.member_w[m].sum(axis=1).mean())),
                "n_weeks": int(len(x)),
            }
        return out

    def corr(self, a: pd.DataFrame, b: pd.DataFrame, split: str = "discovery", every: int = 4) -> float:
        m = self.split_mask(split)
        d = self.dates[m][::every]
        return float(rowwise_spearman(a.reindex(d).where(self.member_w.reindex(d)), b.reindex(d)).mean())


def bonferroni_t(n_tests: int, alpha: float = 0.05) -> float:
    """Two-sided t threshold after Bonferroni over n tests (normal approximation)."""
    return float(norm.ppf(1 - alpha / (2 * max(1, n_tests))))


# ------------------------------------------------------------------------------ ledger


class Ledger:
    """Every candidate ever tried, with where it came from and how it did (SQLite)."""

    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS alphas (
            id INTEGER PRIMARY KEY, expr TEXT UNIQUE, family TEXT, source TEXT, rationale TEXT,
            iteration INTEGER, created REAL, status TEXT, sign INTEGER, metrics TEXT, reason TEXT)""")
        self.db.commit()

    def seen(self, expr: str) -> bool:
        return self.db.execute("SELECT 1 FROM alphas WHERE expr=?", (expr,)).fetchone() is not None

    def add(self, expr, family, source, rationale, iteration, status, sign, metrics, reason):
        self.db.execute("INSERT OR REPLACE INTO alphas (expr, family, source, rationale, iteration, created, status, "
                        "sign, metrics, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (expr, family, source, rationale, iteration, time.time(), status, sign,
                         json.dumps(metrics), reason))
        self.db.commit()

    def count(self, family: str | None = None) -> int:
        q = "SELECT COUNT(*) FROM alphas" + (" WHERE family=?" if family else "")
        return self.db.execute(q, (family,) if family else ()).fetchone()[0]

    def rows(self, status: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM alphas" + (" WHERE status=?" if status else "")
        df = pd.read_sql_query(q, self.db, params=(status,) if status else ())
        df["metrics"] = df["metrics"].map(json.loads)
        return df

    def summary_for_prompt(self, k: int = 12) -> str:
        """What has worked and what has not, in a compact form Claude can learn from."""
        df = self.rows()
        if df.empty:
            return "No alphas tried yet."
        df["disc_t"] = df["metrics"].map(lambda m: m.get("discovery", {}).get("ic_t"))
        df["val_t"] = df["metrics"].map(lambda m: m.get("validation", {}).get("ic_t"))
        acc = df[df.status == "accepted"].sort_values("val_t", ascending=False).head(k)
        rej = df[df.status != "accepted"].sort_values("created", ascending=False).head(k)
        fmt = lambda r: f"- `{r.expr}` [{r.source}] discovery t={r.disc_t:.1f}, validation t={r.val_t if r.val_t == r.val_t else float('nan'):.1f}: {r.reason}"
        lines = [f"{len(df)} alphas tried so far, {int((df.status == 'accepted').sum())} accepted.", "Accepted:"]
        lines += [fmt(r) for r in acc.itertuples()] or ["- none yet"]
        lines += ["Recently rejected:"] + [fmt(r) for r in rej.itertuples()]
        return "\n".join(lines)
