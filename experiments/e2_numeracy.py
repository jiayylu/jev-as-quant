"""E2 - Can Laya read numbers? (Why the verbalizer exists.)

Five yes/no questions whose true answer is a simple threshold on one number ("RSI(14) is above
70", "price is above its 50-day average", ...). The same market states are shown three ways:

  raw_json      {"rsi14": 72.1, "dist_sma50": 0.042, ...}      what a naive API caller sends
  numeric_text  "RSI(14) 72.1; distance to 50-day average +4.2%; ..."   numbers in prose
  verbal        our verbalizer: numbers plus a fixed qualitative word per indicator

A one-line `if` answers all five perfectly, so this measures reading, not judgment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import laya, save
from jevquant.backtest import build_market
from jevquant.data.synthetic import SyntheticConfig, generate
from jevquant.typed import Noul
from jevquant.verbalize import describe, raw_state

QUESTIONS = {
    "rsi_above_70": (Noul("The RSI(14) indicator is above 70."), lambda r: r["rsi14"] > 70),
    "above_sma50": (Noul("The price is above its 50-day moving average."), lambda r: r["dist_sma50"] > 0),
    "up_5d": (Noul("The price rose over the last 5 days."), lambda r: r["ret_5d"] > 0),
    "vol_above_norm": (Noul("Volatility is above its one-year norm."), lambda r: r["vol_ratio"] > 1),
    "near_60d_high": (Noul("The price is within 2% of its 60-day high."), lambda r: r["dd_60d"] > -0.02),
}


def numeric_text(row: pd.Series) -> str:
    return (f"Close {row['close']:.2f}. Distance to 20-day average {row['dist_sma20'] * 100:+.1f}%. "
            f"Distance to 50-day average {row['dist_sma50'] * 100:+.1f}%. "
            f"50-day average change over two weeks {row['sma50_slope'] * 100:+.1f}%. RSI(14) {row['rsi14']:.1f}. "
            f"Returns {row['ret_5d'] * 100:+.1f}% (5 days), {row['ret_20d'] * 100:+.1f}% (20 days), "
            f"{row['ret_60d'] * 100:+.1f}% (60 days). Volatility {row['vol_20d'] * 100:.0f}% annualized, "
            f"{row['vol_ratio']:.2f}x its one-year norm. Drawdown from 60-day high {row['dd_60d'] * 100:.1f}%. "
            f"Volume {row['volume_ratio']:.2f}x its 20-day average.")


def main(n_states: int = 400):
    mkt = build_market(generate(SyntheticConfig(n_assets=4, n_days=1512, seed=11)).prices)
    rows = pd.concat([mkt.features[s].dropna() for s in mkt.symbols])
    rows = rows.sample(n_states, random_state=0)
    reps = {"raw_json": [raw_state(r) for _, r in rows.iterrows()],
            "numeric_text": [numeric_text(r) for _, r in rows.iterrows()],
            "verbal": [describe(r) for _, r in rows.iterrows()]}
    questions = {k: q for k, (q, _) in QUESTIONS.items()}
    truth = {k: np.array([bool(f(r)) for _, r in rows.iterrows()]) for k, (_, f) in QUESTIONS.items()}

    results = []
    for ckpt in ["english", "typed-decisions"]:
        eng = laya(ckpt)
        for rep, states in reps.items():
            ds = eng.system_one_many(states, questions)
            for qid in questions:
                p = np.array([d[qid].noul for d in ds])
                y = truth[qid]
                pred = p > 0.5
                bal = 0.5 * ((pred[y]).mean() + (~pred[~y]).mean()) if y.any() and (~y).any() else float("nan")
                results.append({"checkpoint": ckpt, "representation": rep, "question": qid,
                                "base_rate": float(y.mean()), "accuracy": float((pred == y).mean()),
                                "balanced_accuracy": float(bal), "auc": float(roc_auc_score(y, p)),
                                "brier": float(((p - y) ** 2).mean())})
                print(results[-1])
    df = pd.DataFrame(results)
    summary = df.groupby(["checkpoint", "representation"])[["balanced_accuracy", "auc", "brier"]].mean()
    print(summary)
    save("e2_numeracy", {"n_states": n_states, "results": results,
                         "summary": summary.reset_index().to_dict(orient="records"),
                         "example": {k: v[0] for k, v in reps.items()}})


if __name__ == "__main__":
    main()
