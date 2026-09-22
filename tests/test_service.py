import json
from pathlib import Path

import pytest

from jevquant.service import parse_questions
from jevquant.typed import Choice, Noul, Score, questions_to_wire


def test_parse_questions_roundtrips_the_example_file():
    spec = json.loads((Path(__file__).parents[1] / "examples" / "questions.json").read_text())
    qs = parse_questions(spec)
    assert isinstance(qs["sentiment"], Choice) and isinstance(qs["surprise"], Score) and isinstance(qs["guidance"], Noul)
    assert questions_to_wire(qs)["sentiment"] == spec["sentiment"]


def test_parse_questions_accepts_list_criteria_and_rejects_unknown_types():
    qs = parse_questions({"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}})
    assert qs["q"].labels == ["a", "b"]
    with pytest.raises(ValueError):
        parse_questions({"q": {"type": "freeform", "instructions": "?"}})
