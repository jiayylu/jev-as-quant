"""Checks against the real Laya checkpoint. Opt in: JEVQUANT_TEST_LAYA=1 (downloads ~1.7 GB once)."""
import os

import pytest

pytestmark = [pytest.mark.laya,
              pytest.mark.skipif(os.environ.get("JEVQUANT_TEST_LAYA") != "1", reason="set JEVQUANT_TEST_LAYA=1")]


@pytest.fixture(scope="module")
def engine():
    from jevquant.engines.laya import LayaEngine
    return LayaEngine("typed-decisions")


def test_batched_forward_matches_official_single_call(engine, qs):
    states = ["Price is 6% above its 50-day average. RSI(14) is 74, overbought zone.",
              "Shares plunge 12% after the company slashes guidance and the CEO resigns.",
              {"note": "Board schedules annual shareholder meeting"}]
    batched = engine.system_one_many(states, qs)
    for s, b in zip(states, batched):
        single = engine.system_one(s, qs)
        for k in qs:
            assert b[k].certainty == pytest.approx(single[k].certainty, abs=2e-3)
        assert b["side"].choice == single["side"].choice
        assert b["strength"].score == pytest.approx(single["strength"].score, abs=5e-3)
        assert b["danger"].noul == pytest.approx(single["danger"].noul, abs=5e-3)
