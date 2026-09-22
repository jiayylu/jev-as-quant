"""Three high-level operations shared by the CLI and the MCP server.

  judge            any typed question set about any state
  screen_headlines classify many headlines; flag the ones worth a closer (System-2) read
  market_state     fetch recent bars, verbalize, and run the regime / signal / risk judges
  recent_filings   a company's latest SEC 8-K press releases, each read by Laya

Engines load lazily (Laya takes ~30 s the first time) and stay resident.
Research tooling only: nothing here places orders or constitutes investment advice.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .calibration import ChoiceCalibrator
from .judges import MarketObs, NewsJudge, RegimeJudge, RiskJudge, SignalJudge
from .typed import Choice, Noul, Questions, Score

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def _engine(checkpoint: str):
    from .engines.laya import LayaEngine
    return LayaEngine(checkpoint)


def _news_setup() -> tuple[str, ChoiceCalibrator | None]:
    """Use the news checkpoint and calibrator fitted in experiment E3, if its report exists."""
    ckpt = os.environ.get("JEVQUANT_NEWS_CHECKPOINT")
    cal = None
    report = ROOT / "reports" / "e3_news.json"
    if report.exists():
        e3 = json.loads(report.read_text())
        best = e3["fast_engine"].split("/")[1].split(" ")[0]
        ckpt = ckpt or best
        c = e3.get(f"calibrator_{ckpt}")
        if c:
            cal = ChoiceCalibrator(NewsJudge.labels)
            cal.T = float(c["T"])
            cal.b = np.array([float(c["bias"][k]) for k in NewsJudge.labels])
    return ckpt or "english", cal


def parse_questions(spec: dict) -> Questions:
    """Wire-format dict -> typed questions (the same JSON Jev and Laya accept)."""
    out = {}
    for qid, q in spec.items():
        t = q.get("type")
        if t == "choice":
            crit = q["criteria"]
            out[qid] = Choice(q["instructions"], crit if isinstance(crit, dict) else {c: "" for c in crit})
        elif t == "score":
            out[qid] = Score(q["instructions"], list(q["criteria"]))
        elif t == "noul":
            crit = q.get("criteria") or {}
            out[qid] = Noul(q["instructions"], true=crit.get("true"), false=crit.get("false"))
        else:
            raise ValueError(f"question {qid!r}: type must be choice, score or noul")
    return out


def judge(state, questions: dict, checkpoint: str = "typed-decisions") -> dict:
    qs = parse_questions(questions)
    d = _engine(checkpoint).system_one(state, qs)
    return {"model": f"laya/{checkpoint}", "latency_ms": round(d.latency_ms, 1),
            "answers": {k: {**a.to_wire(), "certainty": round(a.certainty, 4)} for k, a in d.answers.items()}}


def screen_headlines(headlines: list[str], review_below: float = 0.3) -> dict:
    ckpt, cal = _news_setup()
    decisions = _engine(ckpt).system_one_many(list(headlines), NewsJudge.questions)
    labels = NewsJudge.labels
    P = np.array([[d["sentiment"].p(k) for k in labels] for d in decisions])
    if cal is not None:
        P = cal.transform(P)
    rows = []
    for text, p in zip(headlines, P):
        cert = 1 + float((p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(labels)))
        probs = {k: round(float(v), 4) for k, v in zip(labels, p)}
        rows.append({"headline": text, "sentiment": max(probs, key=probs.get), "probabilities": probs,
                     "certainty": round(cert, 4), "needs_review": cert < review_below})
    return {"model": f"laya/{ckpt}" + ("+calibration" if cal is not None else ""),
            "n": len(rows), "n_needs_review": sum(r["needs_review"] for r in rows), "items": rows}


def recent_filings(symbol: str, since_days: int = 120, review_below: float = 0.3) -> dict:
    """Latest SEC 8-K press releases of a US company, each read by Laya (needs SEC_USER_AGENT)."""
    import datetime as dt

    from .data.sec import SecClient, exhibit99, list_8k

    client = SecClient(cache_dir=ROOT / "data_cache" / "sec" / "raw")
    tickers = client.json("https://www.sec.gov/files/company_tickers.json") or {}
    cik = next((str(v["cik_str"]).zfill(10) for v in tickers.values()
                if v["ticker"].upper() == symbol.upper().replace(".", "-")), None)
    if cik is None:
        raise ValueError(f"unknown ticker {symbol!r}")
    since = str(dt.date.today() - dt.timedelta(days=since_days))
    filings = list_8k(client, cik, since)
    items = []
    for r in filings.itertuples():
        ex = exhibit99(client, cik, r.accessionNumber)
        if ex:
            items.append({"accepted_et": r.accepted_et.strftime("%Y-%m-%d %H:%M ET"), "items": r.items,
                          "headline": ex["headline"], "text": f"{ex['headline']}. {ex['lead']}"})
    if not items:
        return {"symbol": symbol, "cik": cik, "since": since, "n": 0, "items": []}
    screened = screen_headlines([it["text"] for it in items], review_below)
    for it, sc in zip(items, screened["items"]):
        it.update({k: sc[k] for k in ("sentiment", "probabilities", "certainty", "needs_review")})
        it.pop("text")
    return {"symbol": symbol, "cik": cik, "since": since, "model": screened["model"], "n": len(items),
            "items": sorted(items, key=lambda x: x["accepted_et"], reverse=True),
            "disclaimer": "Research output; not investment advice."}


def market_state(symbol: str, checkpoint: str = "typed-decisions") -> dict:
    import datetime as dt

    import yfinance as yf

    from .features import compute_features
    from .verbalize import describe

    end = dt.date.today() + dt.timedelta(days=1)
    df = yf.download(symbol, start=str(end - dt.timedelta(days=600)), end=str(end), auto_adjust=True,
                     progress=False)
    if df.empty:
        raise ValueError(f"no data for {symbol!r}")
    if getattr(df.columns, "nlevels", 1) > 1:
        df = df.droplevel(1, axis=1)
    df = df.rename(columns=str.lower)
    row = compute_features(df).dropna().iloc[-1]
    obs = MarketObs(symbol, row.name, row)
    eng = _engine(checkpoint)
    out = {"symbol": symbol, "as_of": str(row.name.date()), "state_text": describe(row), "model": f"laya/{checkpoint}",
           "judges": {}}
    for j in (RegimeJudge(), SignalJudge(), RiskJudge()):
        d = eng.system_one(j.state(obs), j.questions)
        out["judges"][j.name] = {
            "laya": {k: {**a.to_wire(), "certainty": round(a.certainty, 4)} for k, a in d.answers.items()},
            "rules_baseline": {k: a.to_wire() for k, a in j.baseline(obs).items()},
        }
    out["disclaimer"] = "Research output from a technical-indicator model; not investment advice."
    return out
