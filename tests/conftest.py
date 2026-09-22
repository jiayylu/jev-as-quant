import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevquant.engines.base import Engine  # noqa: E402
from jevquant.typed import Choice, Decision, Noul, Score, parse_answers  # noqa: E402


class FakeEngine(Engine):
    """Answers from a user function state -> {qid: raw answer dict}; counts calls."""

    def __init__(self, fn, name="fake"):
        self.fn, self.name = fn, name
        self.calls = 0
        self.batches = []

    def system_one(self, state, questions):
        self.calls += 1
        return Decision(answers=parse_answers(questions, self.fn(state, questions)), engine=self.name, latency_ms=1.0)

    def system_one_many(self, states, questions):
        self.batches.append(len(states))
        return [self.system_one(s, questions) for s in states]


@pytest.fixture
def qs():
    return {
        "side": Choice(instructions="side?", criteria={"buy": "b", "hold": "h", "sell": "s"}),
        "strength": Score(instructions="how strong?", criteria=["low", "mid", "high"]),
        "danger": Noul(instructions="dangerous?"),
    }
