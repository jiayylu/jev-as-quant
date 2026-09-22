import math

from jevquant.typed import (Choice, ChoiceAnswer, Decision, Noul, Score, normalized_certainty,
                            parse_answers, questions_to_wire)


def test_wire_format_matches_jev_and_laya(qs):
    w = questions_to_wire(qs)
    assert w["side"] == {"type": "choice", "instructions": "side?", "criteria": {"buy": "b", "hold": "h", "sell": "s"}}
    assert w["strength"]["criteria"] == ["low", "mid", "high"]
    assert w["danger"] == {"type": "noul", "instructions": "dangerous?"}
    assert Noul("x", true="yes", false="no").to_wire()["criteria"] == {"true": "yes", "false": "no"}


def test_parse_documented_jev_response_shapes():
    # response shapes copied from the public Jev API reference
    questions = {
        "department": Choice("team?", {"billing": "", "technical": "", "sales": ""}),
        "frustration": Score("how frustrated?", ["Calm", "Frustrated", "Very angry"]),
        "is_urgent": Noul("urgent?"),
    }
    raw = {
        "is_urgent": {"type": "noul", "noul": 0.95},
        "department": {"type": "choice", "choice": "billing", "confidence": 0.8,
                       "probabilities": {"billing": 0.87, "sales": 0, "technical": 0.13}},
        "frustration": {"type": "score", "score": 1.04, "confidence": 0.94,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0, "1": 0.96, "2": 0.04}},
    }
    a = parse_answers(questions, raw)
    assert a["department"].choice == "billing" and math.isclose(a["department"].p("technical"), 0.13)
    assert math.isclose(a["frustration"].score, 1.04, abs_tol=1e-9)
    assert a["is_urgent"].noul == 0.95


def test_score_from_expected_value_only():
    a = parse_answers({"s": Score("?", ["a", "b", "c", "d"])}, {"s": {"score": 2.25}})["s"]
    assert math.isclose(a.score, 2.25) and math.isclose(sum(a.probabilities), 1.0)
    assert math.isclose(a.unit, 0.75) and math.isclose(a.signed, 0.5)


def test_certainty_is_engine_agnostic():
    assert normalized_certainty([1 / 3] * 3) < 1e-9
    assert math.isclose(normalized_certainty([1, 0, 0]), 1.0)
    assert 0 < ChoiceAnswer("a", {"a": 0.6, "b": 0.4}).certainty < 1


def test_decision_json_roundtrip(qs):
    raw = {"side": {"probabilities": {"buy": 0.2, "hold": 0.5, "sell": 0.3}},
           "strength": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}, "danger": {"noul": 0.4}}
    d = Decision(answers=parse_answers(qs, raw), engine="x", latency_ms=3.0)
    d2 = Decision.from_json(qs, d.to_json())
    assert d2["side"].choice == "hold" and math.isclose(d2["strength"].score, 1.6)
    assert d2["danger"].noul == 0.4 and d2.latency_ms == 3.0
