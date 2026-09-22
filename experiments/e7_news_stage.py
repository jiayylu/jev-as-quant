"""E7 stage B - news inputs for the alpha factory: earnings press releases read by Laya.

News universe: every 8-K earnings release (item 2.02) of the point-in-time members, 2016-2026
(backfilled from SEC for 2016-2023, the E6 collection for 2024-2026). Each release is read by
Laya (typed-decisions + E3 calibration) and aligned to its first tradable open.

Loops that feed results back:
  1. event seeds (fixed a priori) and Claude proposals that can now use the news inputs;
  2. error mining: discovery-period releases where Laya's reading and the market's reaction
     disagree most are shown to Claude, which proposes new typed questions;
  3. Laya answers each question on a 3,000-release discovery sample (fast screen); questions
     whose answers predict the post-open drift are run on every release and become new
     factory inputs, judged by the same gates as everything else.
The holdout stays locked.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from common import DATA_CACHE, REPORTS, laya, save
from e7_alpha_factory import build_inputs
from jevquant.alpha.dsl import Evaluator
from jevquant.alpha.events import align_releases, event_panels
from jevquant.alpha.factory import Factory
from jevquant.alpha.generators import claude_alphas
from jevquant.alpha.lab import Lab, Ledger
from jevquant.calibration import ChoiceCalibrator, probs_matrix
from jevquant.judges import NewsJudge
from jevquant.typed import Noul, Score

LABELS = NewsJudge.labels
EVENT_SEEDS = {
    "post-earnings drift (announcement return)": "ts_sum(gap, 63)",
    "earnings tone (Laya)": "ts_sum(laya_tone, 63)",
    "bearish earnings call-out (Laya)": "-ts_sum(laya_bear, 63)",
    "no-news reversal": "-pct(close, 5) * (1 - sign(ts_sum(pr_count, 5)))",
    "news continuation": "pct(close, 5) * sign(ts_sum(pr_count, 5))",
}
NEWS_NOTE = """
News inputs (earnings press releases from SEC 8-K item 2.02, placed on the first trading day
whose open came after the release; 0 on other days): pr_count = number of earnings releases,
earn_flag = 1 on an earnings release day, laya_tone = Laya P(bullish) - P(bearish) of the
release, laya_bear = 1 if Laya reads it as bearish, gap = market-adjusted move from the last
close before the release to the first tradable open (the announcement reaction). Use ts_sum /
ts_mean over a window to carry an event forward (e.g. ts_sum(gap, 63) = last quarter's reaction)."""


def load_releases() -> pd.DataFrame:
    old = pd.read_csv(DATA_CACHE / "sec" / "earnings_2016_2023.csv.gz", dtype={"cik": str})
    new = pd.read_csv(DATA_CACHE / "sec" / "press_releases.csv.gz", dtype={"cik": str})
    new = new[new["items"].str.contains("2.02", regex=False)]
    pr = pd.concat([old, new], ignore_index=True)
    pr = pr[~pr["headline"].str.match(r"^Part \d+ [–-] ")]
    pr["accepted_utc"] = pd.to_datetime(pr["accepted_utc"], utc=True)
    pr = pr.sort_values("accepted_utc").drop_duplicates(["cik", "accession"])
    pr["text"] = (pr["headline"].fillna("").str.strip() + ". " + pr["lead"].fillna("").str.strip()).str.slice(0, 1200)
    return pr.reset_index(drop=True)


def laya_probs(texts: list[str]) -> np.ndarray:
    e3 = json.loads((REPORTS / "e3_news.json").read_text())["calibrator_typed-decisions"]
    cal = ChoiceCalibrator(LABELS)
    cal.T, cal.b = float(e3["T"]), np.array([float(e3["bias"][k]) for k in LABELS])
    eng = laya("typed-decisions")
    out = []
    for i in range(0, len(texts), 2000):
        out.append(cal.transform(probs_matrix(eng.system_one_many(texts[i:i + 2000], NewsJudge.questions),
                                              "sentiment", LABELS)))
        print(f"  laya sentiment {min(i + 2000, len(texts)):,}/{len(texts):,}", flush=True)
    return np.vstack(out)


def post_open_drift(pr: pd.DataFrame, open_: pd.DataFrame, member: pd.DataFrame, days: int = 5) -> np.ndarray:
    """Market-adjusted open(exec) -> open(exec+days) return of each release (the tradable drift)."""
    O = open_.ffill().values
    M = member.values
    col = {t: j for j, t in enumerate(open_.columns)}
    out = np.full(len(pr), np.nan)
    for n, r in enumerate(pr.itertuples()):
        i, j = r.exec_idx, col[r.symbol]
        if i + days < len(O):
            rets = O[i + days] / O[i] - 1
            m = M[i] & np.isfinite(rets)
            out[n] = rets[j] - np.nanmean(rets[m])
    return out


QUESTION_PROMPT = """You help a quant team decide what to ask a fast typed-decision model (Laya) about
US company earnings press releases. Laya reads the headline and lead paragraph (about 110 words)
and answers yes/no (noul) or ordinal (score) questions with probabilities. It is weak at
arithmetic, so questions must be answerable from wording, not from computing numbers.

Goal: questions whose answers predict the stock's return over the WEEK AFTER the market can first
trade on the release (the immediate reaction is already gone by then), i.e. information the
market tends to under-react to.

Below are releases where Laya's overall bullish/bearish reading disagreed with the market's
immediate reaction. Study what the reading missed:
{examples}

Research ledger (what has and has not predicted returns so far):
{ledger}

Propose {k} diverse questions. For each give: a short snake_case name, the type (noul or score),
the instruction text for Laya, for a score the ordered levels (3-5), and a one-sentence reason
why the answer could predict post-announcement drift."""

QUESTION_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {
    "type": "object", "properties": {
        "name": {"type": "string"}, "type": {"type": "string", "enum": ["noul", "score"]},
        "instructions": {"type": "string"}, "levels": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"}},
    "required": ["name", "type", "instructions", "levels", "reason"], "additionalProperties": False}}},
    "required": ["questions"], "additionalProperties": False}


def claude_questions(examples: str, ledger: str, k: int = 6) -> tuple[list[dict], float]:
    cmd = ["claude", "-p", "--model", "claude-opus-5", "--effort", "medium", "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--max-budget-usd", "1.5", "--json-schema", json.dumps(QUESTION_SCHEMA)]
    p = subprocess.run(cmd, input=QUESTION_PROMPT.format(examples=examples, ledger=ledger, k=k), capture_output=True,
                       text=True, timeout=900, cwd=tempfile.mkdtemp(), env={**os.environ})
    d = json.loads(p.stdout)
    if d.get("is_error") or "structured_output" not in d:
        raise RuntimeError(str(d)[:300])
    return d["structured_output"]["questions"], float(d.get("total_cost_usd", 0.0))


def to_question(q: dict):
    if q["type"] == "score" and len(q["levels"]) >= 3:
        return Score(q["instructions"], q["levels"][:5])
    return Noul(q["instructions"])


def answer_value(d, qid: str) -> float:
    a = d[qid]
    return a.noul if hasattr(a, "noul") else a.unit


def release_ic(values: np.ndarray, drift: np.ndarray, dates: pd.Series) -> tuple[float, float]:
    """Mean over weeks of the cross-release rank correlation, with its t-stat."""
    df = pd.DataFrame({"v": values, "y": drift, "w": pd.to_datetime(dates).dt.to_period("W")}).dropna()
    ics = df.groupby("w").apply(lambda g: g.v.rank().corr(g.y.rank()) if len(g) >= 8 and g.v.nunique() > 1 else np.nan)
    ics = ics.dropna()
    if len(ics) < 10:
        return float("nan"), float("nan")
    return float(ics.mean()), float(ics.mean() / ics.std() * math.sqrt(len(ics)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-rounds", type=int, default=2)
    ap.add_argument("--question-rounds", type=int, default=2)
    ap.add_argument("--screen-n", type=int, default=3000)
    ap.add_argument("--promote-max", type=int, default=3)
    a = ap.parse_args()

    P, F, inputs = build_inputs()
    lab = Lab(P)
    pr = align_releases(load_releases(), P.dates, P.tickers)
    print(f"earnings releases: {len(pr):,} ({pr.accepted_et.min():%Y-%m} .. {pr.accepted_et.max():%Y-%m}), "
          f"{pr.symbol.nunique()} companies", flush=True)
    probs = laya_probs(list(pr["text"]))
    E = event_panels(pr, P.dates, P.tickers, P.open, P.close, P.member, probs)
    inputs.update(E)
    ledger = Ledger(REPORTS / "alpha_ledger.sqlite")
    fac = Factory(lab, Evaluator(inputs, P.member), ledger)
    for r in ledger.rows("accepted").itertuples():
        fac.library[r.expr] = fac.evaluator(r.expr) * r.sign
    cost = 0.0

    # 1. event seeds, then Claude proposals that may use news inputs
    for name, expr in EVENT_SEEDS.items():
        print(f"seed {name}: {fac.consider(expr, 'seed', 'textbook-events', name, 10)}", flush=True)
    for it in range(a.claude_rounds):
        try:
            props, c = claude_alphas(15, list(inputs), ledger.summary_for_prompt(), news_note=NEWS_NOTE)
            cost += c
        except Exception as e:
            print("claude proposals failed:", e, flush=True)
            continue
        res = [fac.consider(p["expr"], "claude", "claude-opus-5+news", p["rationale"], 11 + it) for p in props]
        print(f"claude news round {it + 1}: {res.count('accepted')} accepted of {len(res)}, cost ${cost:.2f}", flush=True)

    # 2-3. error mining -> Claude questions -> Laya screen -> promote
    drift = post_open_drift(pr, P.open, P.member)
    disc = (pr["exec_date"] >= "2016-01-01") & (pr["exec_date"] <= "2020-12-31")
    tone = probs[:, LABELS.index("bullish")] - probs[:, LABELS.index("bearish")]
    gap = np.array([E["gap"].values[r.exec_idx, P.tickers.index(r.symbol)] for r in pr.itertuples()])
    promoted, screened = {}, []
    rng = np.random.default_rng(0)
    for it in range(a.question_rounds):
        cand = pr[disc & (np.sign(tone) != np.sign(gap)) & (np.abs(gap) > 0.03)]
        ex = cand.sample(min(30, len(cand)), random_state=it)
        examples = "\n".join(f"- [{r.symbol}] Laya tone {tone[i]:+.2f}, market reaction {gap[i]:+.1%}: {r.text[:350]}"
                             for i, r in zip(ex.index, ex.itertuples()))
        try:
            qs, c = claude_questions(examples, ledger.summary_for_prompt())
            cost += c
        except Exception as e:
            print("claude questions failed:", e, flush=True)
            continue
        sample_idx = rng.choice(np.where(disc)[0], min(a.screen_n, int(disc.sum())), replace=False)
        texts = [pr.at[i, "text"] for i in sample_idx]
        for q in qs:
            name = "q_" + "".join(ch for ch in q["name"].lower() if ch.isalnum() or ch == "_")[:30]
            question = {name: to_question(q)}
            ds = laya("typed-decisions").system_one_many(texts, question)
            vals = np.array([answer_value(d, name) for d in ds])
            ic, t = release_ic(vals, drift[sample_idx], pr.loc[sample_idx, "exec_date"])
            screened.append({"name": name, "q": q, "screen_ic": ic, "screen_t": t})
            print(f"  question {name}: screen IC {ic:+.3f} (t {t:+.2f}) — {q['instructions'][:80]}", flush=True)
        best = sorted([s for s in screened if s["name"] not in promoted and np.isfinite(s["screen_t"])
                       and abs(s["screen_t"]) >= 2.5], key=lambda s: -abs(s["screen_t"]))
        for s in best[: max(0, a.promote_max - len(promoted))]:
            question = {s["name"]: to_question(s["q"])}
            ds = laya("typed-decisions").system_one_many(list(pr["text"]), question)
            vals = np.array([answer_value(d, s["name"]) for d in ds])
            panel = event_panels(pr, P.dates, P.tickers, P.open, P.close, P.member, None,
                                 extra={s["name"]: vals})[s["name"]]
            inputs[s["name"]] = panel
            fac.evaluator.inputs = inputs
            sign = "" if s["screen_t"] > 0 else "-"
            status = fac.consider(f"{sign}ts_sum({s['name']}, 63)", "laya-question", "claude-question+laya",
                                  f"{s['q']['instructions']} | {s['q']['reason']}", 20 + it)
            promoted[s["name"]] = status
            print(f"  promoted {s['name']}: {status}", flush=True)

    lib = ledger.rows("accepted")
    print(f"library: {len(lib)} alphas; ledger {ledger.count()} tried; Claude cost ${cost:.2f}", flush=True)
    for r in lib.itertuples():
        print(f"  [{r.family}] {r.expr}   {r.reason}", flush=True)
    save("e7_news_stage", {"n_releases": len(pr), "screened_questions": screened, "promoted": promoted,
                           "library": [{"expr": r.expr, "family": r.family, "reason": r.reason, "rationale": r.rationale}
                                       for r in lib.itertuples()], "claude_cost_usd": cost})


if __name__ == "__main__":
    main()
