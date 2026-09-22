"""Jev (TypeSafe AI): hosted System-1 model behind `POST /v1/systemone`.

Written against the public API reference (request = {model, state, questions}, Bearer auth,
answers keyed by question id). Uses only the standard library so the core package has no
HTTP dependency. Not exercised against the live service in this repo (Jev is waitlisted);
`tests/test_engines.py` drives it against a local server that returns the documented shapes.
Point `base_url` at a Jev-compatible server (e.g. a local shim) to use it without a key.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..typed import Decision, Questions, State, parse_answers, questions_to_wire
from .base import Engine

PRICE_PER_MTOK_INPUT = 0.042  # USD, published list price; output tokens are free


class JevEngine(Engine):
    def __init__(self, model: str = "jev-1.13.0", api_key: str | None = None,
                 base_url: str | None = None, timeout: float = 10.0, retries: int = 2):
        self.model = model
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.name = f"jev/{model}"

    def system_one(self, state: State, questions: Questions) -> Decision:
        body = json.dumps({"model": self.model, "state": state,
                           "questions": questions_to_wire(questions)}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, headers=headers,
                                     method="POST")
        t0 = time.perf_counter()
        payload = self._send(req)
        dt = (time.perf_counter() - t0) * 1000
        usage = payload.get("usage", {})
        cost = usage.get("input_tokens", 0) * PRICE_PER_MTOK_INPUT / 1e6
        return Decision(answers=parse_answers(questions, payload["answers"]), engine=self.name,
                        model=payload.get("model", self.model), latency_ms=dt, usage=usage,
                        cost_usd=cost)

    def _send(self, req: urllib.request.Request) -> dict:
        delay = 0.5
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.retries:
                    raise RuntimeError(f"Jev HTTP {e.code}: {e.read()[:300]!r}") from e
                delay = float(e.headers.get("retry-after", delay))
            except urllib.error.URLError:
                if attempt == self.retries:
                    raise
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")
