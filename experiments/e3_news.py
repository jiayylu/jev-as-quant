"""E3 - Reading financial news: where a System-1 model should shine (text is its native input).

Data: zeroshot/twitter-financial-news-sentiment (human labels bearish / bullish / neutral).
  train (9,543)      fits TF-IDF, the Laya calibrators, and the cascade threshold
  validation (2,388) is only ever used for evaluation

Readers
  lexicon            a priori finance word lists, no training                    (rules)
  tfidf-logreg       supervised on all 9,543 training labels                     (classic ML)
  laya/<ckpt>        zero-shot, three checkpoints                                (System 1)
  laya/<ckpt>+cal    same + temperature/bias fitted on 1,500 training headlines
  claude             Claude Opus 5 via headless Claude Code, same typed question (System 2)
  cascade            best Laya variant, lowest-certainty x% escalated to Claude  (System 1 -> 2)

Claude labels are collected for a random 300 validation headlines (to score Claude alone) and
for the 25% least certain ones under the chosen Laya variant (to trace the cascade curve).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_CACHE, DATA_OUT, claude, laya, save, Timer
from jevquant.calibration import ChoiceCalibrator, probs_matrix
from jevquant.data.news import load_headlines
from jevquant.judges import NewsJudge, RuleReader, TfidfNewsReader
from jevquant.metrics import classification_report, reliability_bins

LABELS = NewsJudge.labels
Q = NewsJudge.questions


def main(n_cal: int = 1500, n_claude_random: int = 300, max_escalation: float = 0.25):
    train = load_headlines("train", DATA_CACHE)
    val = load_headlines("validation", DATA_CACHE)
    cal_set = train.groupby("label").sample(frac=n_cal / len(train), random_state=0).reset_index(drop=True)
    y_val = val["label"].tolist()
    table, probs, extras = {}, {}, {}

    def record(name, P, latency_ms=None, cost_per_1k=0.0):
        rep = classification_report(y_val, P, LABELS)
        rep.update({"latency_ms_per_item": latency_ms, "cost_usd_per_1k": cost_per_1k})
        table[name] = rep
        probs[name] = P
        print(f"{name:32s} acc={rep['accuracy']:.3f} macroF1={rep['macro_f1']:.3f} "
              f"brier={rep['brier']:.3f} ece={rep['ece']:.3f}")

    # --- rules and supervised baseline
    lex = RuleReader(NewsJudge()).read(val["text"])
    record("lexicon (rules)", probs_matrix(lex, "sentiment", LABELS), float(np.mean([d.latency_ms for d in lex])))
    with Timer() as t:
        tf = TfidfNewsReader(train)
    extras["tfidf_train_seconds"] = t.s
    tfd = tf.read(val["text"])
    record("tf-idf + logreg (supervised)", probs_matrix(tfd, "sentiment", LABELS), tfd[0].latency_ms)

    # --- Laya, zero-shot and calibrated on the training subset
    fast_candidates = {}
    for ckpt in ["english", "typed-decisions", "multilingual"]:
        eng = laya(ckpt)
        dv = eng.system_one_many(list(val["text"]), Q)
        dc = eng.system_one_many(list(cal_set["text"]), Q)
        lat = float(np.nanmedian([d.latency_ms for d in dv]))
        Pv, Pc = probs_matrix(dv, "sentiment", LABELS), probs_matrix(dc, "sentiment", LABELS)
        record(f"laya/{ckpt} (zero-shot)", Pv, lat)
        cal = ChoiceCalibrator(LABELS).fit(Pc, cal_set["label"])
        record(f"laya/{ckpt} + calibration", cal.transform(Pv), lat)
        extras[f"calibrator_{ckpt}"] = {"T": cal.T, "bias": dict(zip(LABELS, cal.b))}
        # choose the fast engine on TRAINING data only
        train_f1 = classification_report(cal_set["label"], cal.transform(Pc), LABELS)["macro_f1"]
        fast_candidates[ckpt] = (train_f1, cal, Pc)

    best = max(fast_candidates, key=lambda k: fast_candidates[k][0])
    extras["fast_engine"] = f"laya/{best} + calibration (chosen by training macro-F1)"
    extras["fast_engine_train_f1"] = {k: v[0] for k, v in fast_candidates.items()}
    P_fast = probs[f"laya/{best} + calibration"]
    cert = 1 + (P_fast * np.log(np.clip(P_fast, 1e-12, 1))).sum(1) / np.log(len(LABELS))
    order = np.argsort(cert)  # least certain first

    # cascade threshold fixed on the training subset: the certainty at the 20% quantile
    Pc_best = fast_candidates[best][1].transform(fast_candidates[best][2])
    cert_train = 1 + (Pc_best * np.log(np.clip(Pc_best, 1e-12, 1))).sum(1) / np.log(len(LABELS))
    extras["cascade_threshold_from_train_q20"] = float(np.quantile(cert_train, 0.20))

    # --- Claude on the random sample + the least-certain slice
    rng = np.random.default_rng(0)
    rand_idx = np.sort(rng.choice(len(val), n_claude_random, replace=False))
    hard_idx = order[: int(max_escalation * len(val))]
    need = np.unique(np.concatenate([rand_idx, hard_idx]))
    cl = claude()
    with Timer() as t:
        dcl = cl.system_one_many([val["text"][i] for i in need], Q)
    by_idx = dict(zip(need.tolist(), dcl))
    fresh = [d for d in dcl if not d.cached]
    extras["claude"] = {"model": "claude-opus-5", "effort": "low", "transport": "claude -p (Claude Code)",
                        "items": int(len(need)), "wall_seconds_this_run": t.s,
                        "fresh_items": len(fresh),
                        "notional_cost_usd": float(sum(d.cost_usd for d in dcl)),
                        "median_batch_latency_ms": float(np.median([d.latency_ms for d in dcl])),
                        "items_per_request": 40}
    P_claude_all = np.array([[by_idx[i]["sentiment"].p(k) for k in LABELS] for i in need])
    cost_per_item = extras["claude"]["notional_cost_usd"] / len(need)

    sub = [y_val[i] for i in rand_idx]
    P_claude_rand = np.array([[by_idx[i]["sentiment"].p(k) for k in LABELS] for i in rand_idx])
    extras["random_300"] = {
        "claude": classification_report(sub, P_claude_rand, LABELS),
        **{name: classification_report(sub, probs[name][rand_idx], LABELS)
           for name in ["lexicon (rules)", "tf-idf + logreg (supervised)", f"laya/{best} + calibration"]},
    }
    print("random-300:", {k: round(v["macro_f1"], 3) for k, v in extras["random_300"].items()})

    # --- cascade curve: escalate the least-certain x% to Claude
    curve = []
    for rate in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        P = P_fast.copy()
        k = int(rate * len(val))
        for i in order[:k]:
            P[i] = [by_idx[i]["sentiment"].p(lbl) for lbl in LABELS]
        rep = classification_report(y_val, P, LABELS)
        curve.append({"escalation_rate": rate, "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
                      "brier": rep["brier"], "claude_cost_usd_per_1k": rate * 1000 * cost_per_item})
        print(curve[-1])
    # the operating point with the train-fixed threshold
    thr = extras["cascade_threshold_from_train_q20"]
    esc = cert < thr
    P = P_fast.copy()
    for i in np.where(esc)[0]:
        if i in by_idx:
            P[i] = [by_idx[i]["sentiment"].p(lbl) for lbl in LABELS]
    covered = all(i in by_idx for i in np.where(esc)[0])
    record(f"cascade laya/{best}+cal -> claude", P, None, float(esc.mean() * 1000 * cost_per_item))
    extras["cascade_operating_point"] = {"threshold": thr, "escalation_rate": float(esc.mean()),
                                         "all_escalations_labeled": covered}

    # --- reliability diagram data for the fast engine, before and after calibration
    rel = {}
    for name in [f"laya/{best} (zero-shot)", f"laya/{best} + calibration", "tf-idf + logreg (supervised)"]:
        P = probs[name]
        correct = P.argmax(1) == np.array([LABELS.index(y) for y in y_val])
        rel[name] = reliability_bins(P.max(1), correct).to_dict(orient="records")

    out = pd.DataFrame({"idx": np.arange(len(val)), "label": y_val})
    for name, P in probs.items():
        for j, lbl in enumerate(LABELS):
            out[f"{name}|{lbl}"] = np.round(P[:, j], 4)
    out.to_csv(DATA_OUT / "e3_validation_probabilities.csv", index=False)  # no tweet text (not ours to redistribute)
    save("e3_news", {"labels": LABELS, "n_val": len(val), "n_train": len(train), "n_calibration": len(cal_set),
                     "class_balance_val": val["label"].value_counts(normalize=True).to_dict(),
                     "table": table, "cascade_curve": curve, "reliability": rel, **extras})


if __name__ == "__main__":
    main()
