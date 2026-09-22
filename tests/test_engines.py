import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from conftest import FakeEngine
from jevquant.engines.cache import CachedEngine
from jevquant.engines.cascade import CascadeEngine
from jevquant.engines.claude import ClaudeAPIEngine, ClaudeCodeEngine, answer_schema, render_questions
from jevquant.engines.jev import JevEngine


def _answers(state, questions, p_buy=0.8):
    out = {}
    for qid, q in questions.items():
        t = q.to_wire()["type"]
        if t == "choice":
            rest = (1 - p_buy) / (len(q.labels) - 1)
            out[qid] = {"probabilities": {k: (p_buy if i == 0 else rest) for i, k in enumerate(q.labels)}}
        elif t == "score":
            out[qid] = {"score": 1.0}
        else:
            out[qid] = {"noul": 0.3}
    return out


# ------------------------------------------------------------------ Jev over HTTP

class _JevHandler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _JevHandler.seen.append((self.path, self.headers.get("Authorization"), body))
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                answers[qid] = {"type": "choice", "choice": keys[0], "confidence": 0.9,
                                "probabilities": {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)}}
            elif q["type"] == "score":
                answers[qid] = {"type": "score", "score": 2.0,
                                "probabilities": {str(i): (1.0 if i == 2 else 0.0) for i in range(len(q["criteria"]))}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.96}
        data = json.dumps({"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 1000}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_jev_engine_speaks_documented_wire_format(qs):
    srv = HTTPServer(("127.0.0.1", 0), _JevHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        eng = JevEngine(api_key="k", base_url=f"http://127.0.0.1:{srv.server_port}")
        d = eng.system_one({"ticker": "X"}, qs)
    finally:
        srv.shutdown()
    path, auth, body = _JevHandler.seen[-1]
    assert path == "/v1/systemone" and auth == "Bearer k"
    assert body["model"] == "jev-1.13.0" and body["state"] == {"ticker": "X"}
    assert body["questions"]["side"]["type"] == "choice"
    assert d["side"].choice == "buy" and d["strength"].score == 2.0 and d["danger"].noul == 0.96
    assert abs(d.cost_usd - 1000 * 0.042 / 1e6) < 1e-12


# ------------------------------------------------------------------ cache

def test_cache_serves_repeats_without_calling_engine(tmp_path, qs):
    fake = FakeEngine(_answers)
    eng = CachedEngine(fake, tmp_path / "c.sqlite")
    a = eng.system_one_many(["s1", "s2", "s1"], qs)
    assert fake.calls == 2  # duplicate inside a batch computed once
    b = eng.system_one_many(["s2", "s1"], qs)
    assert fake.calls == 2 and all(d.cached for d in b)
    assert b[0]["side"].probabilities == a[1]["side"].probabilities
    # a different question set is a different key
    eng.system_one("s1", {"side": qs["side"]})
    assert fake.calls == 3


# ------------------------------------------------------------------ cascade

def test_cascade_escalates_only_uncertain_items_in_one_batch(qs):
    fast = FakeEngine(lambda s, q: _answers(s, q, p_buy=0.9 if s.startswith("easy") else 0.36), "fast")
    slow = FakeEngine(lambda s, q: _answers(s, q, p_buy=0.99), "slow")
    eng = CascadeEngine(fast, slow, threshold=0.3)
    out = eng.system_one_many(["easy1", "hard1", "easy2", "hard2"], {"side": qs["side"]})
    assert [d.escalated for d in out] == [False, True, False, True]
    assert slow.batches == [2]  # hard cases sent together
    assert out[1].engine == "slow" and out[0].engine == "fast"
    assert eng.escalation_rate == 0.5


# ------------------------------------------------------------------ Claude (no network)

def test_claude_schema_and_prompt_cover_every_question(qs):
    sch = answer_schema(qs)
    ans = sch["properties"]["items"]["items"]["properties"]["answers"]
    assert set(ans["required"]) == set(qs)
    assert ans["properties"]["side"]["required"] == ["buy", "hold", "sell"]
    assert ans["properties"]["strength"]["required"] == ["0", "1", "2"]
    assert ans["properties"]["danger"]["required"] == ["p_true"]
    text = render_questions(qs)
    assert "side" in text and '"2" = high' in text


def _claude_items(n):
    return [{"id": i, "answers": {"side": {"buy": 0.1, "hold": 0.2, "sell": 0.7},
                                  "strength": {"0": 0.0, "1": 0.5, "2": 0.5}, "danger": {"p_true": 0.8}}}
            for i in range(n)]


def test_claude_api_engine_parses_structured_output(qs):
    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text=json.dumps({"items": _claude_items(3)}))],
                                   usage=SimpleNamespace(input_tokens=2000, output_tokens=300))

    eng = ClaudeAPIEngine(client=SimpleNamespace(messages=FakeMessages()), batch_size=10)
    out = eng.system_one_many(["a", "b", "c"], qs)
    kw = calls[0]
    assert kw["model"] == "claude-opus-5" and kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["extra_body"] == {"fallbacks": "default"}
    assert '<item id="2">' in kw["messages"][0]["content"]
    assert out[2]["side"].choice == "sell" and abs(out[0]["strength"].score - 1.5) < 1e-9
    assert out[1]["danger"].noul == 0.8
    assert abs(eng.total_cost_usd - (2000 * 5 + 300 * 25) / 1e6) < 1e-12


def test_claude_api_engine_raises_on_refusal(qs):
    class Refuse:
        def create(self, **kw):
            return SimpleNamespace(stop_reason="refusal", stop_details=None, content=[],
                                   usage=SimpleNamespace(input_tokens=0, output_tokens=0))

    with pytest.raises(RuntimeError, match="declined"):
        ClaudeAPIEngine(client=SimpleNamespace(messages=Refuse())).system_one("a", qs)


def test_claude_code_engine_via_fake_cli(tmp_path, qs):
    fake = tmp_path / "claude"
    payload = json.dumps({"is_error": False, "total_cost_usd": 0.02, "usage": {"input_tokens": 10, "output_tokens": 5},
                          "structured_output": {"items": _claude_items(2)}})
    fake.write_text(f"#!/bin/sh\ncat > {tmp_path}/stdin.txt\necho '{payload}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    eng = ClaudeCodeEngine(binary=str(fake))
    out = eng.system_one_many(["headline one", "headline two"], qs)
    assert out[0]["side"].choice == "sell" and out[1].cost_usd == 0.01
    assert "headline two" in (tmp_path / "stdin.txt").read_text()
