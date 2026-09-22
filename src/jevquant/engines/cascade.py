"""System 1 -> System 2 cascade: Laya answers everything, Claude only the hard cases.

The fast engine sees every state. A state is escalated when `should_escalate(decision)` is
true (by default: the least certain answer is below `threshold`). Escalated states are sent
to the slow engine in one batch, so a Claude engine can answer them in a single request.
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..typed import Decision, Questions, State
from .base import Engine


class CascadeEngine(Engine):
    def __init__(self, fast: Engine, slow: Engine, threshold: float = 0.3,
                 should_escalate: Callable[[Decision], bool] | None = None):
        self.fast = fast
        self.slow = slow
        self.threshold = threshold
        self.should_escalate = should_escalate or (lambda d: d.min_certainty < self.threshold)
        self.name = f"cascade({fast.name}->{slow.name}@{threshold:g})"
        self.n_seen = 0
        self.n_escalated = 0

    @property
    def fingerprint(self) -> str:
        return f"cascade({self.fast.fingerprint}->{self.slow.fingerprint}@{self.threshold:g})"

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / self.n_seen if self.n_seen else 0.0

    def system_one(self, state: State, questions: Questions) -> Decision:
        return self.system_one_many([state], questions)[0]

    def system_one_many(self, states: Sequence[State], questions: Questions) -> list[Decision]:
        fast = self.fast.system_one_many(states, questions)
        idx = [i for i, d in enumerate(fast) if self.should_escalate(d)]
        self.n_seen += len(states)
        self.n_escalated += len(idx)
        if not idx:
            return fast
        slow = self.slow.system_one_many([states[i] for i in idx], questions)
        out = list(fast)
        for i, d in zip(idx, slow):
            d.escalated = True
            d.latency_ms = fast[i].latency_ms + d.latency_ms  # the fast pass still happened
            d.cost_usd += fast[i].cost_usd
            out[i] = d
        return out
