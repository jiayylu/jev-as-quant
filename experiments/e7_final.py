"""E7 final exam: freeze the alpha library, then run the locked holdout (2024-01 .. 2026-09) once.

Two portfolios from the adaptive composite (weights from past realized ICs only):
  aggressive  long-only equal-weight top 50 (keep until rank 100), vs equal-weight members and SPY
  enhanced    cap-weighted members tilted by the score (w = w_cap * (1 + 0.5 z)), vs SPY and the
              reconstructed cap-weighted member index
Reported for discovery, validation and holdout. Nothing is tuned after this runs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_CACHE, REPORTS, laya, save
from e7_alpha_factory import build_inputs, split_perf
from e7_news_stage import answer_value, laya_probs, load_releases, to_question
from jevquant.alpha.dsl import Evaluator
from jevquant.alpha.events import align_releases, event_panels
from jevquant.alpha.factory import Factory, enhanced_index, summarize, weekly_portfolio
from jevquant.alpha.fundamentals import build_fundamentals
from jevquant.alpha.lab import SPLITS, Lab, Ledger


def main():
    P, F, inputs = build_inputs()
    lab = Lab(P)
    pr = align_releases(load_releases(), P.dates, P.tickers)
    inputs.update(event_panels(pr, P.dates, P.tickers, P.open, P.close, P.member, laya_probs(list(pr["text"]))))
    stage = json.loads((REPORTS / "e7_news_stage.json").read_text()) if (REPORTS / "e7_news_stage.json").exists() else {}
    for s in stage.get("screened_questions", []):
        if s["name"] in stage.get("promoted", {}):
            q = {s["name"]: to_question(s["q"])}
            vals = np.array([answer_value(d, s["name"]) for d in laya("typed-decisions").system_one_many(list(pr["text"]), q)])
            inputs[s["name"]] = event_panels(pr, P.dates, P.tickers, P.open, P.close, P.member, None,
                                             extra={s["name"]: vals})[s["name"]]
    ledger = Ledger(REPORTS / "alpha_ledger.sqlite")
    fac = Factory(lab, Evaluator(inputs, P.member), ledger)
    lib = ledger.rows("accepted")
    for r in lib.itertuples():
        fac.library[r.expr] = fac.evaluator(r.expr) * r.sign
    u = pd.read_csv(DATA_CACHE / "sec" / "universe_hist.csv", dtype=str)
    mcap = build_fundamentals(DATA_CACHE / "sec" / "fundamentals_pit.csv.gz", P.dates,
                              u.groupby("cik")["ticker"].apply(list).to_dict(), P.tickers, P.close, P.volume)["mcap"]
    splits = ("discovery", "validation", "holdout")
    out = {"library": [{"expr": r.expr, "family": r.family, "reason": r.reason} for r in lib.itertuples()],
           "n_tried": ledger.count(), "tried_by_family": {f: ledger.count(f) for f in ("seed", "random", "claude", "laya-question")},
           "coverage": P.info["coverage"], "portfolios": {}}
    bench = enhanced_index(None, lab, mcap)
    if len(lib) == 0:
        print("empty library: nothing to trade; reporting benchmarks only", flush=True)
    else:
        score = fac.composite(adaptive=True)
        agg = weekly_portfolio(score, lab)
        enh = enhanced_index(score, lab, mcap).join(bench["portfolio"].rename("cap_bench"))
        agg.to_csv(REPORTS / "data" / "e7_final_aggressive.csv")
        enh.to_csv(REPORTS / "data" / "e7_final_enhanced.csv")
        out["portfolios"]["aggressive_top50"] = split_perf(agg, splits)
        out["portfolios"]["enhanced_index"] = {
            s: {"vs_spy": summarize(enh.loc[SPLITS[s][0]:SPLITS[s][1]], "portfolio", "spy"),
                "vs_cap_bench": summarize(enh.loc[SPLITS[s][0]:SPLITS[s][1]], "portfolio", "cap_bench"),
                "turnover_per_week": float(enh.loc[SPLITS[s][0]:SPLITS[s][1], "turnover"].mean())} for s in splits}
        out["weights_last"] = fac.weights.iloc[-1].round(3).to_dict()
        for name, block in out["portfolios"].items():
            for s in splits:
                b = block[s]
                key = "vs_ew_members" if "vs_ew_members" in b else "vs_cap_bench"
                print(f"{name:17s} {s:10s} {key}: excess {b[key].get('excess_cagr', np.nan):+.2%}/yr "
                      f"IR {b[key].get('info_ratio', np.nan):+.2f} t {b[key].get('excess_t', np.nan):+.2f} | "
                      f"vs SPY excess {b['vs_spy'].get('excess_cagr', np.nan):+.2%}/yr", flush=True)
    save("e7_final", out)


if __name__ == "__main__":
    main()
