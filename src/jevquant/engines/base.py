from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..typed import Decision, Questions, State


class Engine(ABC):
    """Answers typed questions about a state.

    `system_one` is the latency path (one state). `system_one_many` is the throughput path
    used by backtests; engines that can batch (Laya on a GPU, Claude with many items per
    request) override it.
    """

    name: str = "engine"

    @property
    def fingerprint(self) -> str:
        """Identifies the model + settings, used as part of cache keys."""
        return self.name

    @abstractmethod
    def system_one(self, state: State, questions: Questions) -> Decision: ...

    def system_one_many(self, states: Sequence[State], questions: Questions) -> list[Decision]:
        return [self.system_one(s, questions) for s in states]
