"""Typed questions and answers shared by every decision engine.

The wire format is the one Jev (TypeSafe `/v1/systemone`) and Laya (`agent.predict`) both
speak, so a question set written once runs unchanged on Laya, Jev, Claude, or the rule
baselines:

    {"type": "choice", "instructions": "...", "criteria": {"buy": "...", "sell": "..."}}
    {"type": "score",  "instructions": "...", "criteria": ["low", "mid", "high"]}
    {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}}
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Union

State = Union[str, dict, list]


# --------------------------------------------------------------------------- questions


@dataclass
class Choice:
    """Pick one option. `criteria` maps option key -> description (order is preserved)."""

    instructions: str
    criteria: Mapping[str, str]

    @property
    def labels(self) -> list[str]:
        return list(self.criteria)

    def to_wire(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass
class Score:
    """Place the state on an ordered rubric; `criteria[0]` is level 0 (lowest)."""

    instructions: str
    criteria: Sequence[str]

    @property
    def levels(self) -> int:
        return len(self.criteria)

    def to_wire(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass
class Noul:
    """Yes/no statement; the answer is P(true)."""

    instructions: str
    true: str | None = None
    false: str | None = None

    def to_wire(self) -> dict:
        d: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.true or self.false:
            d["criteria"] = {"true": self.true or "", "false": self.false or ""}
        return d


Question = Union[Choice, Score, Noul]
Questions = Mapping[str, Question]


def questions_to_wire(questions: Questions) -> dict:
    return {qid: q.to_wire() for qid, q in questions.items()}


# --------------------------------------------------------------------------- answers


def normalized_certainty(probs: Sequence[float]) -> float:
    """1 - H(p)/log(k): 0 for a uniform answer, 1 for a one-hot answer.

    Computed the same way for every engine so that escalation thresholds mean the same
    thing whether the answer came from Laya, Jev or Claude.
    """
    p = [max(float(x), 0.0) for x in probs]
    s = sum(p)
    k = len(p)
    if k < 2 or s <= 0:
        return 1.0
    ent = -sum((x / s) * math.log(x / s) for x in p if x > 0)
    return max(0.0, min(1.0, 1.0 - ent / math.log(k)))


def _normalize(values: Sequence[float]) -> list[float]:
    v = [max(float(x), 0.0) for x in values]
    s = sum(v)
    if s <= 0:
        return [1.0 / len(v)] * len(v)
    return [x / s for x in v]


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None  # engine-reported, semantics differ between vendors

    type = "choice"

    @property
    def certainty(self) -> float:
        return normalized_certainty(list(self.probabilities.values()))

    def p(self, label: str) -> float:
        return float(self.probabilities.get(label, 0.0))

    def to_wire(self) -> dict:
        return {"type": "choice", "choice": self.choice, "probabilities": self.probabilities,
                "confidence": self.confidence}


@dataclass
class ScoreAnswer:
    score: float  # expected level, can fall between levels
    probabilities: list[float]  # index = level
    legend: list[str] = field(default_factory=list)
    confidence: float | None = None

    type = "score"

    @property
    def levels(self) -> int:
        return len(self.probabilities)

    @property
    def unit(self) -> float:
        """Expected level rescaled to [0, 1]."""
        return self.score / max(1, self.levels - 1)

    @property
    def signed(self) -> float:
        """Expected level rescaled to [-1, 1] (useful for bearish..bullish rubrics)."""
        return 2.0 * self.unit - 1.0

    @property
    def certainty(self) -> float:
        return normalized_certainty(self.probabilities)

    def to_wire(self) -> dict:
        return {"type": "score", "score": self.score,
                "probabilities": {str(i): p for i, p in enumerate(self.probabilities)},
                "legend": {str(i): c for i, c in enumerate(self.legend)},
                "confidence": self.confidence}


@dataclass
class NoulAnswer:
    noul: float  # P(true)

    type = "noul"

    @property
    def certainty(self) -> float:
        return normalized_certainty([self.noul, 1.0 - self.noul])

    def to_wire(self) -> dict:
        return {"type": "noul", "noul": self.noul}


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


def parse_answer(question: Question, raw: Mapping[str, Any]) -> Answer:
    """Parse one answer from Laya / Jev / our own JSON into a typed Answer.

    Tolerates the small differences between vendors (Jev omits `confidence` on noul, Laya
    adds an `action` block, probabilities may be keyed by level string or given as a list).
    """
    if isinstance(question, Choice):
        labels = question.labels
        probs_raw = raw.get("probabilities") or {}
        probs = _normalize([float(probs_raw.get(k, 0.0)) for k in labels])
        pd = dict(zip(labels, probs))
        choice = raw.get("choice")
        if choice not in pd:
            choice = max(pd, key=pd.get)
        return ChoiceAnswer(choice=choice, probabilities=pd, confidence=raw.get("confidence"))
    if isinstance(question, Score):
        k = question.levels
        pr = raw.get("probabilities")
        if isinstance(pr, Mapping):
            vec = [float(pr.get(str(i), pr.get(i, 0.0))) for i in range(k)]
        elif isinstance(pr, Sequence):
            vec = [float(x) for x in pr][:k]
        else:
            vec = []
        if len(vec) != k or sum(vec) <= 0:
            # only an expected score was returned: put the mass on the two nearest levels
            s = min(max(float(raw["score"]), 0.0), k - 1)
            lo = int(math.floor(s))
            vec = [0.0] * k
            vec[lo] = 1.0 - (s - lo)
            if lo + 1 < k:
                vec[lo + 1] = s - lo
        vec = _normalize(vec)
        score = sum(i * p for i, p in enumerate(vec))
        return ScoreAnswer(score=score, probabilities=vec, legend=list(question.criteria),
                           confidence=raw.get("confidence"))
    if isinstance(question, Noul):
        return NoulAnswer(noul=min(1.0, max(0.0, float(raw["noul"]))))
    raise TypeError(f"unknown question type: {question!r}")


def parse_answers(questions: Questions, raw_answers: Mapping[str, Any]) -> dict[str, Answer]:
    missing = [q for q in questions if q not in raw_answers]
    if missing:
        raise ValueError(f"engine returned no answer for {missing}")
    return {qid: parse_answer(q, raw_answers[qid]) for qid, q in questions.items()}


@dataclass
class Decision:
    """All answers for one state, plus provenance (which engine, how long, what it cost)."""

    answers: dict[str, Answer]
    engine: str
    model: str = ""
    latency_ms: float = float("nan")
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    cached: bool = False
    escalated: bool = False

    def __getitem__(self, qid: str) -> Answer:
        return self.answers[qid]

    @property
    def min_certainty(self) -> float:
        return min((a.certainty for a in self.answers.values()), default=1.0)

    def to_json(self) -> dict:
        return {"answers": {k: a.to_wire() for k, a in self.answers.items()}, "engine": self.engine,
                "model": self.model, "latency_ms": self.latency_ms, "usage": self.usage,
                "cost_usd": self.cost_usd, "escalated": self.escalated}

    @classmethod
    def from_json(cls, questions: Questions, d: Mapping[str, Any], cached: bool = False) -> "Decision":
        return cls(answers=parse_answers(questions, d["answers"]), engine=d.get("engine", ""),
                   model=d.get("model", ""), latency_ms=d.get("latency_ms", float("nan")),
                   usage=d.get("usage", {}), cost_usd=d.get("cost_usd", 0.0), cached=cached,
                   escalated=d.get("escalated", False))
