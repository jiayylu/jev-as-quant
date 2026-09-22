"""Judges = a typed question set + how to show the state to a model + a code-only baseline.

The four use cases from the brief map to four judges:

  SignalJudge  (1. signal)      Choice buy/hold/sell  + Score trend strength
  RiskJudge    (2. risk)        Noul risk-off         + Score market stress
  RegimeJudge  (3. routing)     Choice uptrend/downtrend/range/crisis
  NewsJudge    (4. news)        Choice bullish/bearish/neutral

Every judge can be answered by any Engine (Laya, Jev, Claude, a cascade) through
`ModelReader`, or by hand-written rules through `RuleReader`. Readers return `Decision`s
with identical types, so strategies never know which one they are talking to.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import pandas as pd

from .engines.base import Engine
from .typed import Choice, ChoiceAnswer, Decision, Noul, NoulAnswer, Questions, Score, ScoreAnswer
from .verbalize import describe, raw_state


@dataclass(frozen=True)
class MarketObs:
    symbol: str
    date: Any
    row: pd.Series


def _soft_choice(labels: list[str], pick: str, p: float = 0.7) -> ChoiceAnswer:
    rest = (1 - p) / (len(labels) - 1)
    return ChoiceAnswer(choice=pick, probabilities={k: (p if k == pick else rest) for k in labels})


def _soft_score(levels: int, level: int, legend: list[str], p: float = 0.7) -> ScoreAnswer:
    rest = (1 - p) / (levels - 1)
    probs = [p if i == level else rest for i in range(levels)]
    return ScoreAnswer(score=sum(i * x for i, x in enumerate(probs)), probabilities=probs, legend=legend)


def _nz(x: float, default: float = 0.0) -> float:
    return default if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


class Judge(Protocol):
    name: str
    questions: Questions

    def state(self, obs: Any) -> Any: ...

    def baseline(self, obs: Any) -> dict: ...


# ----------------------------------------------------------------------------- market judges


class _MarketJudge:
    representation = "verbal"  # or "raw" for the numeracy ablation

    def state(self, obs: MarketObs):
        if self.representation == "raw":
            return raw_state(obs.row, obs.symbol)
        return describe(obs.row)


class RegimeJudge(_MarketJudge):
    name = "regime"
    labels = ["uptrend", "downtrend", "range", "crisis"]
    questions = {
        "regime": Choice(
            instructions="Which market regime is this stock in right now?",
            criteria={
                "uptrend": "prices rising steadily with moderate volatility",
                "downtrend": "prices falling steadily",
                "range": "prices moving sideways around a stable level with low volatility",
                "crisis": "a sharp crash with extremely high volatility and panic selling",
            },
        ),
    }

    def baseline(self, obs: MarketObs) -> dict:
        r = obs.row
        vr, ret20 = _nz(r["vol_ratio"], 1.0), _nz(r["ret_20d"])
        d50, s50 = _nz(r["dist_sma50"]), _nz(r["sma50_slope"])
        if vr > 1.8 and ret20 < -0.06:
            pick = "crisis"
        elif d50 > 0.02 and s50 > 0.002:
            pick = "uptrend"
        elif d50 < -0.02 and s50 < -0.002:
            pick = "downtrend"
        else:
            pick = "range"
        return {"regime": _soft_choice(self.labels, pick)}


class SignalJudge(_MarketJudge):
    name = "signal"
    trend_levels = ["strong downtrend", "mild downtrend", "no clear trend", "mild uptrend", "strong uptrend"]
    questions = {
        "action": Choice(
            instructions="A disciplined trend-following trader reads this chart. What should they do?",
            criteria={
                "buy": "trend and momentum point up: hold a full long position",
                "hold": "mixed or unclear evidence: keep a small position",
                "sell": "trend and momentum point down: stay out of the market",
            },
        ),
        "trend": Score(instructions="Direction and strength of the price trend", criteria=trend_levels),
    }

    def baseline(self, obs: MarketObs) -> dict:
        r = obs.row
        votes = sum(math.copysign(1, v) if abs(v) > eps else 0 for v, eps in
                    [(_nz(r["dist_sma50"]), 0.01), (_nz(r["sma50_slope"]), 0.002), (_nz(r["ret_60d"]), 0.02)])
        level = int(2 + max(-2, min(2, round(votes * 2 / 3))))
        action = "buy" if level >= 3 else "sell" if level <= 1 else "hold"
        return {"action": _soft_choice(["buy", "hold", "sell"], action),
                "trend": _soft_score(5, level, self.trend_levels)}


class RiskJudge(_MarketJudge):
    name = "risk"
    stress_levels = ["calm", "normal", "stressed", "panic"]
    questions = {
        "risk_off": Noul(
            instructions="Market conditions are dangerous and a disciplined trader should cut exposure now.",
            true="crash-like selling, a volatility spike, panic",
            false="normal market conditions, even if prices drift lower",
        ),
        "stress": Score(instructions="How stressed is the market for this stock?", criteria=stress_levels),
    }

    def baseline(self, obs: MarketObs) -> dict:
        r = obs.row
        vr, dd = _nz(r["vol_ratio"], 1.0), _nz(r["dd_60d"])
        danger = vr > 1.8 and dd < -0.10
        level = 3 if danger else 2 if vr > 1.3 else 1 if vr > 0.7 else 0
        return {"risk_off": NoulAnswer(noul=0.9 if danger else 0.1),
                "stress": _soft_score(4, level, self.stress_levels)}


# ----------------------------------------------------------------------------- news judge

# Written once, before looking at any results, and never tuned on the data.
POSITIVE_WORDS = set("""beat beats beating upgrade upgrades upgraded raise raises raised record surge surges
surged soar soars soared jump jumps jumped rally rallies rallied gain gains gained outperform buy bullish
strong growth profit profits higher rise rises rose top tops topped boost boosts boosted approval approved
win wins won expand expands expanded exceed exceeds exceeded upbeat rebound rebounds""".split())
NEGATIVE_WORDS = set("""miss misses missed downgrade downgrades downgraded cut cuts lower lowers lowered
fall falls fell drop drops dropped plunge plunges plunged slump slumps sink sinks sank tumble tumbles
tumbled loss losses weak lawsuit probe investigation recall bearish sell underperform decline declines
declined warn warns warning layoffs bankruptcy halt halts fraud negative slash slashes slashed""".split())


class NewsJudge:
    name = "news"
    labels = ["bullish", "bearish", "neutral"]
    questions = {
        "sentiment": Choice(
            instructions="How will this financial headline move the price of the stock it is about?",
            criteria={
                "bullish": "good news for the stock: likely to push its price up",
                "bearish": "bad news for the stock: likely to push its price down",
                "neutral": "no clear positive or negative effect on the price",
            },
        ),
    }

    def state(self, text: str) -> str:
        return text

    def baseline(self, text: str) -> dict:
        words = re.findall(r"[a-z]+", text.lower())
        s = sum(w in POSITIVE_WORDS for w in words) - sum(w in NEGATIVE_WORDS for w in words)
        pick = "bullish" if s > 0 else "bearish" if s < 0 else "neutral"
        return {"sentiment": _soft_choice(self.labels, pick, p=0.6)}


# ----------------------------------------------------------------------------- readers


class Reader(Protocol):
    name: str

    def read(self, observations: Sequence[Any]) -> list[Decision]: ...


class ModelReader:
    """Asks an Engine the judge's questions about each observation."""

    def __init__(self, judge: Judge, engine: Engine):
        self.judge, self.engine = judge, engine
        self.name = f"{judge.name}:{engine.name}"

    def read(self, observations):
        states = [self.judge.state(o) for o in observations]
        return self.engine.system_one_many(states, self.judge.questions)


class RuleReader:
    """Answers with the judge's hand-written baseline."""

    def __init__(self, judge: Judge):
        self.judge = judge
        self.name = f"{judge.name}:rules"

    def read(self, observations):
        out = []
        for o in observations:
            t0 = time.perf_counter()
            ans = self.judge.baseline(o)
            out.append(Decision(answers=ans, engine="rules", latency_ms=(time.perf_counter() - t0) * 1000))
        return out


class FixedReader:
    """Wraps precomputed answers (oracles, supervised models) as a Reader."""

    def __init__(self, name: str, fn):
        self.name, self.fn = name, fn

    def read(self, observations):
        return [Decision(answers=self.fn(o), engine=self.name, latency_ms=0.0) for o in observations]


class TfidfNewsReader:
    """Supervised baseline: TF-IDF + logistic regression trained on labeled headlines."""

    name = "news:tfidf-logreg"

    def __init__(self, train: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                   LogisticRegression(max_iter=2000, C=4.0))
        self.model.fit(train["text"], train["label"])
        self.classes = list(self.model.classes_)

    def read(self, texts):
        t0 = time.perf_counter()
        proba = self.model.predict_proba(list(texts))
        dt = (time.perf_counter() - t0) * 1000 / max(1, len(texts))
        out = []
        for row in proba:
            pd_ = {k: float(row[self.classes.index(k)]) for k in NewsJudge.labels}
            out.append(Decision(answers={"sentiment": ChoiceAnswer(choice=max(pd_, key=pd_.get), probabilities=pd_)},
                                engine=self.name, latency_ms=dt))
        return out
