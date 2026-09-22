"""E7 - The alpha factory on point-in-time S&P 500 data.

Stage A (this script): price/volume and point-in-time fundamental inputs.
  iteration 0      textbook seeds
  iterations 1..R  random search (Bonferroni-gated) + Claude proposals that read the ledger
Then the accepted library is combined (adaptive, past-only weights) into a weekly long-only
top-50 portfolio, compared with the equal-weight member index and SPY.

The holdout (2024-01 .. 2026-09) stays locked unless --final is passed; the development
report covers discovery (2014-2020) and validation (2021-2023) only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import DATA_CACHE, REPORTS, save
from jevquant.alpha.dsl import Evaluator
from jevquant.alpha.factory import Factory, summarize, weekly_portfolio
from jevquant.alpha.fundamentals import build_fundamentals
from jevquant.alpha.generators import SEEDS, claude_alphas, random_alphas
from jevquant.alpha.lab import SPLITS, Lab, Ledger
from jevquant.alpha.panel import build_panel


def build_inputs():
    P = build_panel(DATA_CACHE)
    u = pd.read_csv(DATA_CACHE / "sec" / "universe_hist.csv", dtype=str)
    F = build_fundamentals(DATA_CACHE / "sec" / "fundamentals_pit.csv.gz", P.dates,
                           u.groupby("cik")["ticker"].apply(list).to_dict(), P.tickers)
    inputs = {"open": P.open, "close": P.close, "volume": P.volume,
              "ret": P.close.pct_change(fill_method=None), "dollar_vol": P.close * P.volume, **F}
    return P, F, inputs


def split_perf(weekly: pd.DataFrame, which: tuple[str, ...]) -> dict:
    out = {}
    for s in which:
        a, b = SPLITS[s]
        w = weekly.loc[a:b]
        out[s] = {"vs_ew_members": summarize(w, "portfolio", "ew_members"), "vs_spy": summarize(w, "portfolio", "spy"),
                  "turnover_per_week": float(w["turnover"].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-rounds", type=int, default=2)
    ap.add_argument("--random-per-round", type=int, default=300)
    ap.add_argument("--claude-rounds", type=int, default=3)
    ap.add_argument("--claude-per-round", type=int, default=15)
    ap.add_argument("--fresh", action="store_true", help="start a new ledger")
    ap.add_argument("--final", action="store_true", help="unlock the holdout and report it")
    a = ap.parse_args()

    t0 = time.time()
    P, F, inputs = build_inputs()
    lab = Lab(P)
    path = REPORTS / "alpha_ledger.sqlite"
    if a.fresh and path.exists():
        path.unlink()
    ledger = Ledger(path)
    fac = Factory(lab, Evaluator(inputs, P.member), ledger)
    fundamentals = set(F)
    print(f"panel {P.close.shape}, coverage {P.info['coverage']:.1%}, setup {time.time() - t0:.0f}s", flush=True)

    # rebuild the library from a previous run of the same ledger
    for r in ledger.rows("accepted").itertuples():
        fac.library[r.expr] = fac.evaluator(r.expr) * r.sign

    it = 0
    for name, expr in SEEDS.items():
        fac.consider(expr, "seed", "textbook", name, it)
    cost = 0.0
    for it in range(1, max(a.random_rounds, a.claude_rounds) + 1):
        if it <= a.random_rounds:
            cands = random_alphas(a.random_per_round, list(inputs), seed=it, fundamentals=fundamentals)
            res = [fac.consider(e, "random", "grammar", "", it) for e in cands]
            print(f"iter {it} random: {res.count('accepted')} accepted of {len(res)} "
                  f"(threshold t={fac.threshold('random'):.2f})", flush=True)
        if it <= a.claude_rounds:
            try:
                props, c = claude_alphas(a.claude_per_round, list(inputs), ledger.summary_for_prompt())
                cost += c
            except Exception as e:  # keep the loop alive; the ledger records what was tried
                print("claude proposal failed:", e, flush=True)
                props = []
            res = [fac.consider(p["expr"], "claude", "claude-opus-5", p["rationale"], it) for p in props]
            print(f"iter {it} claude: {res.count('accepted')} accepted of {len(res)} "
                  f"({res.count('invalid')} invalid), cost so far ${cost:.2f}", flush=True)

    lib = ledger.rows("accepted")
    print(f"library: {len(lib)} alphas; ledger: {ledger.count()} tried", flush=True)
    for r in lib.itertuples():
        print(f"  [{r.family}] {r.expr}   {r.reason}", flush=True)

    splits = ("discovery", "validation") + (("holdout",) if a.final else ())
    out = {"coverage": P.info["coverage"], "n_tried": ledger.count(),
           "tried_by_family": {f: ledger.count(f) for f in ("seed", "random", "claude")},
           "library": [{"expr": r.expr, "family": r.family, "rationale": r.rationale, "reason": r.reason,
                        "metrics": r.metrics} for r in lib.itertuples()],
           "claude_cost_usd": cost, "holdout_unlocked": a.final, "portfolios": {}}
    if len(lib):
        for mode in ("adaptive", "static"):
            score = fac.composite(adaptive=(mode == "adaptive"))
            weekly = weekly_portfolio(score, lab)
            out["portfolios"][mode] = split_perf(weekly, splits)
            if a.final:
                weekly.to_csv(REPORTS / "data" / f"e7_weekly_{mode}.csv")
            for s in splits:
                x = out["portfolios"][mode][s]
                print(f"{mode:8s} {s:10s} vs EW: excess {x['vs_ew_members'].get('excess_cagr', float('nan')):+.2%}/yr "
                      f"IR {x['vs_ew_members'].get('info_ratio', float('nan')):+.2f} (t {x['vs_ew_members'].get('excess_t', float('nan')):+.2f}) | "
                      f"vs SPY: excess {x['vs_spy'].get('excess_cagr', float('nan')):+.2%}/yr | turnover {x['turnover_per_week']:.1%}/wk",
                      flush=True)
    save("e7_alpha_factory" + ("_final" if a.final else "_dev"), out)


if __name__ == "__main__":
    main()
