"""Post-hoc calibration of Choice probabilities: one temperature + one bias per class.

Laya ships raw-temperature logits and says calibration must be refit per domain; this is
that refit. Fit on a *training* split only, then freeze.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .typed import ChoiceAnswer, Decision


class ChoiceCalibrator:
    def __init__(self, labels: Sequence[str], use_bias: bool = True):
        self.labels = list(labels)
        self.use_bias = use_bias
        self.T = 1.0
        self.b = np.zeros(len(self.labels))

    def _apply(self, P: np.ndarray, T: float, b: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(P, 1e-9, 1.0)) / T + b
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def fit(self, P: np.ndarray, y: Sequence[str]) -> "ChoiceCalibrator":
        P = np.asarray(P, float)
        yi = np.array([self.labels.index(t) for t in y])
        k = len(self.labels)

        def unpack(x):
            T = float(np.exp(x[0]))
            b = np.concatenate([[0.0], x[1:]]) if self.use_bias else np.zeros(k)
            return T, b

        def nll(x):
            T, b = unpack(x)
            Q = self._apply(P, T, b)
            return -np.log(np.clip(Q[np.arange(len(yi)), yi], 1e-12, 1)).mean()

        x0 = np.zeros(1 + (k - 1 if self.use_bias else 0))
        res = minimize(nll, x0, method="L-BFGS-B")
        self.T, self.b = unpack(res.x)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        return self._apply(np.asarray(P, float), self.T, self.b)


def probs_matrix(decisions: Sequence[Decision], qid: str, labels: Sequence[str]) -> np.ndarray:
    return np.array([[d[qid].p(k) for k in labels] for d in decisions])


class CalibratedReader:
    """Wraps a Reader; rewrites one Choice question's probabilities through a fitted calibrator."""

    def __init__(self, inner, qid: str, calibrator: ChoiceCalibrator, name: str | None = None):
        self.inner, self.qid, self.cal = inner, qid, calibrator
        self.name = name or f"{inner.name}+calibrated"

    def read(self, observations):
        decisions = self.inner.read(observations)
        P = self.cal.transform(probs_matrix(decisions, self.qid, self.cal.labels))
        out = []
        for d, row in zip(decisions, P):
            d2 = copy.copy(d)
            d2.answers = dict(d.answers)
            pd_ = dict(zip(self.cal.labels, map(float, row)))
            d2.answers[self.qid] = ChoiceAnswer(choice=max(pd_, key=pd_.get), probabilities=pd_)
            out.append(d2)
        return out
