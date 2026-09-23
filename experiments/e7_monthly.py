"""E7 second track: judge the SAME candidate pool at a monthly horizon.

Many anomalies that drown in weekly noise and turnover survive monthly. This re-runs every
expression already in the weekly ledger - no new candidates, so no new search - with decisions
on the last trading day of each month, a 21-day holding period and the same gates, into its own
ledger. Running two horizons is itself a second look at the data, so the two tracks are
reported together and the holdout is only unlocked once, by e7_final.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import REPORTS, save
from e7_alpha_factory import build_inputs
from e7_news_stage import laya_probs, load_releases
from jevquant.alpha.dsl import Evaluator
from jevquant.alpha.events import align_releases, event_panels
from jevquant.alpha.factory import Factory
from jevquant.alpha.lab import Lab, Ledger


def main():
    P, F, inputs = build_inputs()
    pr = align_releases(load_releases(), P.dates, P.tickers)
    inputs.update(event_panels(pr, P.dates, P.tickers, P.open, P.close, P.member, laya_probs(list(pr["text"]))))
    lab = Lab(P, horizon=21, freq="M")
    weekly = Ledger(REPORTS / "alpha_ledger.sqlite").rows()
    monthly = Ledger(REPORTS / "alpha_ledger_monthly.sqlite")
    fac = Factory(lab, Evaluator(inputs, P.member), monthly)
    print(f"monthly decision dates: {len(lab.dates)}; re-judging {len(weekly)} expressions", flush=True)
    counts = {}
    for r in weekly.itertuples():
        if r.status == "invalid":
            continue
        s = fac.consider(r.expr, r.family, r.source, r.rationale, 0)
        counts[s] = counts.get(s, 0) + 1
    print(counts, flush=True)
    lib = monthly.rows("accepted")
    for r in lib.itertuples():
        print(f"  [{r.family}] {r.expr}   {r.reason}   {r.rationale[:90]}", flush=True)
    save("e7_monthly", {"counts": counts, "n_judged": int(monthly.count()),
                        "library": [{"expr": r.expr, "family": r.family, "reason": r.reason,
                                     "rationale": r.rationale, "metrics": r.metrics} for r in lib.itertuples()]})


if __name__ == "__main__":
    main()
