"""The alpha factory loop: generate -> screen -> validate -> combine -> trade -> feed back.

Acceptance gates (fixed before the first run):
  discovery   |IC t-stat| above the family threshold (the sign is set here)
                seeds 2.0 · claude 2.5 · laya-question 2.5 · random: Bonferroni over every random
                expression ever tried
  validation  same sign, t >= 1.65, and the long-short decile still earns after costs
  redundancy  |rank correlation| < 0.7 with every alpha already in the library

Combination is adaptive and uses only the past: each week every accepted alpha is weighted by
its trailing information ratio over the last three years of *already realized* labels, and an
alpha whose trailing IR turns negative is switched off. This is the loop that keeps feeding
realized outcomes back into the model while it trades.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dsl import AlphaError, Evaluator
from .lab import COST_BPS, Lab, Ledger, bonferroni_t

FAMILY_T = {"seed": 2.0, "claude": 2.5, "laya-question": 2.5}
VAL_T = 1.65
MAX_CORR = 0.7


@dataclass
class Factory:
    lab: Lab
    evaluator: Evaluator
    ledger: Ledger
    library: dict[str, pd.DataFrame] = field(default_factory=dict)  # expr -> oriented alpha panel
    log: list[dict] = field(default_factory=list)

    def threshold(self, family: str) -> float:
        if family == "random":
            return max(3.0, bonferroni_t(self.ledger.count("random")))
        return FAMILY_T[family]

    def consider(self, expr: str, family: str, source: str, rationale: str, iteration: int) -> str:
        if self.ledger.seen(expr):
            return "duplicate"
        try:
            a = self.evaluator(expr)
        except (AlphaError, ValueError, TypeError, ZeroDivisionError) as e:
            self.ledger.add(expr, family, source, rationale, iteration, "invalid", 0, {}, str(e)[:200])
            return "invalid"
        m = self.lab.evaluate(a)
        sign = 1 if (m["discovery"]["ic"] or 0) >= 0 else -1
        if sign < 0:  # judge the oriented signal, costs included
            a = -a
            m = self.lab.evaluate(a)
        d, v = m["discovery"], m["validation"]
        status, reason = "rejected", ""
        if not np.isfinite(d["ic_t"]) or d["ic_t"] < self.threshold(family):
            reason = f"discovery t={d['ic_t']:.2f} < {self.threshold(family):.2f}"
        elif not np.isfinite(v["ic_t"]) or v["ic_t"] < VAL_T:
            reason = f"validation t={v['ic_t']:.2f} < {VAL_T}"
        elif not v["ls_net_sharpe"] > 0:
            reason = f"validation long-short net Sharpe {v['ls_net_sharpe']:.2f} <= 0 after costs"
        else:
            corrs = {e: self.lab.corr(a, lib) for e, lib in self.library.items()}
            worst = max(corrs.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
            if abs(worst[1]) >= MAX_CORR:
                reason = f"corr {worst[1]:+.2f} with `{worst[0]}`"
            else:
                status, reason = "accepted", f"discovery t={d['ic_t']:.2f}, validation t={v['ic_t']:.2f}"
                self.library[expr] = a
        self.ledger.add(expr, family, source, rationale, iteration, status, sign, m, reason)
        self.log.append({"expr": expr, "family": family, "status": status, "reason": reason})
        return status

    # ------------------------------------------------------------------ combination
    def composite(self, adaptive: bool = True, trailing_weeks: int = 156, min_weeks: int = 52) -> pd.DataFrame:
        """Weekly composite score from the library (dates x tickers)."""
        lab = self.lab
        ranks = {e: lab.weekly(a).rank(axis=1, pct=True) - 0.5 for e, a in self.library.items()}
        if not adaptive:
            return sum(ranks.values()) / max(1, len(ranks))
        ics = {e: lab.ic_series(a) for e, a in self.library.items()}
        known = lab.label_known_on
        score = pd.DataFrame(0.0, index=lab.dates, columns=lab.label.columns)
        wsum = pd.Series(0.0, index=lab.dates)
        self.weights = pd.DataFrame(0.0, index=lab.dates, columns=list(self.library))
        for e, ic in ics.items():
            w = []
            vals = ic.values
            kn = known.values
            for i, t in enumerate(lab.dates.values):
                ok = kn[:i] <= t  # labels already realized at decision time t
                x = vals[:i][ok][-trailing_weeks:]
                x = x[np.isfinite(x)]
                w.append(max(0.0, x.mean() / x.std() * math.sqrt(52)) if len(x) >= min_weeks and x.std() > 0 else 0.0)
            w = pd.Series(w, index=lab.dates)
            self.weights[e] = w
            score = score.add(ranks[e].mul(w, axis=0), fill_value=0.0)
            wsum += w
        return score.div(wsum.replace(0, np.nan), axis=0)


def weekly_portfolio(score: pd.DataFrame, lab: Lab, n: int = 50, buffer: int = 100,
                     cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Long-only equal-weight top-n with a keep-until-rank-`buffer` rule; weekly, next-open trades.

    Returns weekly net returns of the portfolio, the equal-weight member benchmark and SPY over
    the same holding periods (open after the decision to open after the next decision).
    """
    p = lab.panel
    O = p.open.ffill()  # a stock delisted mid-week exits at its last traded price
    pos = np.searchsorted(p.dates.values, lab.dates.values)
    rows, held = [], []
    for k in range(len(lab.dates) - 1):
        t = lab.dates[k]
        i0, i1 = pos[k] + 1, pos[k + 1] + 1
        if i1 >= len(p.dates):
            break
        s = score.loc[t].where(lab.member_w.loc[t]).dropna()
        if len(s) < n * 2:
            rows.append((t, np.nan, np.nan, np.nan, 0.0))
            continue
        order = s.rank(ascending=False)
        keep = [h for h in held if h in order.index and order[h] <= buffer]
        new = [x for x in order.sort_values().index if x not in keep][: max(0, n - len(keep))]
        port = keep + new
        changed = len(set(port) ^ set(held)) / 2 / n if held else 1.0
        r = O.iloc[i1] / O.iloc[i0] - 1
        members = lab.member_w.loc[t]
        bench = r[members[members].index].mean()
        spy = p.spy["open"].iloc[i1] / p.spy["open"].iloc[i0] - 1
        gross = r[port].mean()
        net = gross - changed * 2 * cost_bps / 1e4
        rows.append((t, net, bench, spy, changed))
        held = port
    return pd.DataFrame(rows, columns=["date", "portfolio", "ew_members", "spy", "turnover"]).set_index("date")


def summarize(weekly: pd.DataFrame, a: str, b: str) -> dict:
    """Performance of column a vs benchmark column b over weekly returns."""
    w = weekly[[a, b]].dropna()
    if len(w) < 10:
        return {}
    ex = w[a] - w[b]
    years = len(w) / 52
    cagr = lambda r: float((1 + r).prod() ** (1 / years) - 1)
    return {"cagr": cagr(w[a]), "bench_cagr": cagr(w[b]), "excess_cagr": cagr(w[a]) - cagr(w[b]),
            "sharpe": float(w[a].mean() / w[a].std() * math.sqrt(52)),
            "info_ratio": float(ex.mean() / ex.std() * math.sqrt(52)) if ex.std() > 0 else float("nan"),
            "excess_t": float(ex.mean() / ex.std() * math.sqrt(len(ex))) if ex.std() > 0 else float("nan"),
            "max_dd": float(((1 + w[a]).cumprod() / (1 + w[a]).cumprod().cummax() - 1).min()),
            "hit_rate_vs_bench": float((ex > 0).mean()), "weeks": int(len(w))}


def enhanced_index(score: pd.DataFrame | None, lab: Lab, mcap: pd.DataFrame, tilt: float = 0.5,
                   cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Cap-weighted member portfolio tilted by the alpha score: w = w_cap * (1 + tilt * z), z in [-1, 1].

    With score=None this is the reconstructed cap-weighted benchmark itself. Weekly rebalance at
    the next open; holdings drift in between; costs on the traded weight.
    """
    p = lab.panel
    O = p.open.ffill()
    pos = np.searchsorted(p.dates.values, lab.dates.values)
    mc = mcap.reindex(lab.dates).where(lab.member_w)
    rows, w_prev = [], None
    for k in range(len(lab.dates) - 1):
        t = lab.dates[k]
        i0, i1 = pos[k] + 1, pos[k + 1] + 1
        if i1 >= len(p.dates):
            break
        m = mc.loc[t].dropna()
        m = m[m > 0]
        if len(m) < 100:
            rows.append((t, np.nan, np.nan, 0.0))
            continue
        w = m / m.sum()
        if score is not None:
            s = score.loc[t].reindex(w.index)
            z = (s.rank(pct=True) - 0.5) * 2
            w = w * (1 + tilt * z.fillna(0.0))
            w = w / w.sum()
        r = (O.iloc[i1] / O.iloc[i0] - 1).reindex(w.index).fillna(0.0)
        traded = (w.sub(w_prev, fill_value=0.0).abs().sum() / 2) if w_prev is not None else 1.0
        net = float((w * r).sum()) - traded * 2 * cost_bps / 1e4
        spy = p.spy["open"].iloc[i1] / p.spy["open"].iloc[i0] - 1
        rows.append((t, net, spy, traded))
        w_prev = w * (1 + r) / (1 + float((w * r).sum()))
    return pd.DataFrame(rows, columns=["date", "portfolio", "spy", "turnover"]).set_index("date")
