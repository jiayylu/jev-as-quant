"""E6 exploratory (post-hoc, NOT pre-registered): drift after releases a reader calls bearish.

Found after looking at E6 results; treat as a hypothesis to test on new data, not as a result.
"""
import json
import math

import numpy as np
import pandas as pd

from common import REPORTS, claude
import e6_real_news as e6
from jevquant.judges import NewsJudge

m, syms = e6.load_market()
ev = e6.load_events(m)
texts = list(ev["text"])
P_laya = e6.calibrated_laya_probs(texts)                      # cached
thr = json.loads((REPORTS / "e3_news.json").read_text())["cascade_threshold_from_train_q20"]
hard = np.where(e6.certainty(P_laya) < thr)[0]
cl = claude().system_one_many([texts[i] for i in hard], NewsJudge.questions)  # cached
Pc = P_laya.copy()
for i, d in zip(hard, cl):
    Pc[i] = [d["sentiment"].p(k) for k in e6.LABELS]
sigs = {"laya": e6.to_signal(P_laya), "cascade": e6.to_signal(Pc)}
esc = np.zeros(len(ev), bool); esc[hard] = True
pos = {d: i for i, d in enumerate(m.dates)}
out = {}
for name, s in sigs.items():
    rows = []
    for j, (e, v) in enumerate(zip(ev.itertuples(), s)):
        i = pos.get(pd.Timestamp(e.exec_date))
        if v != -1 or i is None or i + 10 >= len(m.dates):
            continue
        rows.append({"d": m.dates[i], "r5": e6.abnormal(m, e.symbol, i, i + 5), "esc": esc[j], "items": e.items})
    df = pd.DataFrame(rows)
    x = df.groupby("d")["r5"].mean()
    res = {"n": len(df), "mean_bp": df.r5.mean() * 1e4, "t_by_date": x.mean() / x.std() * math.sqrt(len(x)),
           "hit_down": (df.r5 < 0).mean(), "earnings_share": df["items"].str.contains("2.02").mean()}
    if name == "cascade":
        for flag in [True, False]:
            sub = df[df.esc == flag]
            xs = sub.groupby("d")["r5"].mean()
            res[f"{'claude' if flag else 'laya'}_called_bear"] = {"n": len(sub), "mean_bp": sub.r5.mean() * 1e4,
                                                                 "t": xs.mean() / xs.std() * math.sqrt(len(xs))}
    out[name] = res
    # short-only strategy on bearish calls (market neutral), same costs and holding period
    only_bear = np.where(s == -1, -1.0, 0.0)
    perf, _ = e6.backtest(m, ev, only_bear, name=f"{name} short bearish only")
    res["short_only_sharpe"] = perf["sharpe"]; res["short_only_ci"] = perf["sharpe_ci95"]
    res["short_only_by_year"] = perf["sharpe_by_year"]
print(json.dumps(out, indent=1, default=lambda v: round(float(v), 3)))
json.dump(out, open(REPORTS / "e6_exploratory_bearish.json", "w"), indent=1, default=lambda v: round(float(v), 4))
