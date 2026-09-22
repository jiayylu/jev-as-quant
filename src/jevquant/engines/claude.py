"""Claude as System 2: the same typed questions, answered by a frontier reasoning model.

Two transports, one prompt and one JSON schema:

* `ClaudeAPIEngine`  - official `anthropic` SDK, structured outputs (`output_config.format`).
* `ClaudeCodeEngine` - the `claude -p` CLI (headless Claude Code) with `--json-schema`; lets
  anyone logged in to Claude Code run the cascade without an API key.

Both answer many states per request (Claude reads the batch once), and both ask for a
probability per option so the answers are comparable with Laya's.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from ..typed import Choice, Decision, Noul, Questions, Score, State, parse_answers
from .base import Engine

# USD per million tokens (input, output); cache reads/writes ignored for estimates
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}

SYSTEM_PROMPT = """You are System 2 in a two-speed decision pipeline for systematic-trading research.
A fast System-1 model screens every item and routes some of them to you for a closer look. For
every item, answer every question below and give a probability for every option.

Report calibrated probabilities: when you say 0.7, you should be right about 70% of the time.
Put real mass on the alternatives when the text is ambiguous. Judge only from the text of the
item; do not assume facts it does not state.

Questions:
{questions}"""


def render_questions(questions: Questions) -> str:
    lines = []
    for qid, q in questions.items():
        if isinstance(q, Choice):
            opts = "; ".join(f'"{k}" = {v}' for k, v in q.criteria.items())
            lines.append(f"- {qid} (choose one): {q.instructions}\n  options: {opts}\n"
                         f"  answer: a probability for each option, summing to 1")
        elif isinstance(q, Score):
            lv = "; ".join(f'"{i}" = {c}' for i, c in enumerate(q.criteria))
            lines.append(f"- {qid} (ordered levels): {q.instructions}\n  levels: {lv}\n"
                         f"  answer: a probability for each level, summing to 1")
        elif isinstance(q, Noul):
            extra = ""
            if q.true or q.false:
                extra = f"\n  true means: {q.true or 'the statement holds'}; false means: {q.false or 'it does not'}"
            lines.append(f"- {qid} (yes/no): {q.instructions}{extra}\n  answer: p_true, the probability the statement is true")
    return "\n".join(lines)


def answer_schema(questions: Questions) -> dict:
    def num():
        return {"type": "number"}

    props: dict[str, Any] = {}
    for qid, q in questions.items():
        if isinstance(q, Choice):
            keys = q.labels
        elif isinstance(q, Score):
            keys = [str(i) for i in range(q.levels)]
        else:
            keys = ["p_true"]
        props[qid] = {"type": "object", "properties": {k: num() for k in keys}, "required": keys,
                      "additionalProperties": False}
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"},
                           "answers": {"type": "object", "properties": props,
                                       "required": list(questions), "additionalProperties": False}},
            "required": ["id", "answers"], "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False,
    }


def render_items(states: Sequence[State]) -> str:
    parts = []
    for i, s in enumerate(states):
        text = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
        parts.append(f"<item id=\"{i}\">\n{text}\n</item>")
    return "Answer every question for each item below, and return one entry per item id.\n\n" + "\n\n".join(parts)


def to_raw_answers(questions: Questions, answers: dict) -> dict:
    raw = {}
    for qid, q in questions.items():
        a = answers[qid]
        if isinstance(q, Noul):
            raw[qid] = {"noul": min(1.0, max(0.0, float(a["p_true"])))}
        else:
            raw[qid] = {"probabilities": {k: max(0.0, float(v)) for k, v in a.items()}}
    return raw


class _ClaudeBase(Engine):
    def __init__(self, model: str, effort: str, batch_size: int, max_workers: int = 1):
        self.model = model
        self.effort = effort
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.total_cost_usd = 0.0

    @property
    def fingerprint(self) -> str:
        return f"{self.name}@effort={self.effort}"

    def system_one(self, state: State, questions: Questions) -> Decision:
        return self.system_one_many([state], questions)[0]

    def system_one_many(self, states: Sequence[State], questions: Questions) -> list[Decision]:
        chunks = [list(states[i:i + self.batch_size]) for i in range(0, len(states), self.batch_size)]
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            results = list(pool.map(lambda c: self._run_chunk(c, questions), chunks))
        return [d for chunk in results for d in chunk]

    def _run_chunk(self, chunk: list[State], questions: Questions, attempts: int = 3) -> list[Decision]:
        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                items, cost, usage = self._request(chunk, questions)
                self.total_cost_usd += cost
                by_id = {int(it["id"]): it["answers"] for it in items}
                missing = [j for j in range(len(chunk)) if j not in by_id]
                if missing:
                    raise RuntimeError(f"{self.name} returned no answer for items {missing}")
                break
            except (RuntimeError, subprocess.TimeoutExpired, KeyError, ValueError):
                if attempt == attempts - 1:
                    raise
                time.sleep(2 ** attempt)
        dt = (time.perf_counter() - t0) * 1000
        return [Decision(answers=parse_answers(questions, to_raw_answers(questions, by_id[j])),
                         engine=self.name, model=self.model, latency_ms=dt,
                         usage={**usage, "batch": len(chunk)}, cost_usd=cost / len(chunk))
                for j in range(len(chunk))]

    def _request(self, states: list[State], questions: Questions) -> tuple[list[dict], float, dict]:
        raise NotImplementedError


class ClaudeAPIEngine(_ClaudeBase):
    """Claude through the official `anthropic` SDK (reads ANTHROPIC_API_KEY or an `ant` profile)."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "low", batch_size: int = 40,
                 fallbacks: bool = True, client: Any = None, max_workers: int = 4):
        super().__init__(model, effort, batch_size, max_workers)
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.fallbacks = fallbacks
        self.name = f"claude-api/{model}"

    def _request(self, states, questions):
        kwargs: dict[str, Any] = {}
        if self.fallbacks:
            # on a safety decline, re-run server-side on Anthropic's recommended fallback model
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT.format(questions=render_questions(questions)),
            messages=[{"role": "user", "content": render_items(states)}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": answer_schema(questions)}},
            **kwargs,
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Claude hit max_tokens; lower batch_size")
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        pin, pout = PRICES.get(self.model, (5.0, 25.0))
        u = resp.usage
        cost = (u.input_tokens * pin + u.output_tokens * pout) / 1e6
        return data["items"], cost, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


class ClaudeCodeEngine(_ClaudeBase):
    """Claude through headless Claude Code (`claude -p`), using the caller's Claude Code login.

    Runs with every tool disabled, a replacement system prompt, no session persistence and a
    per-call budget cap, from an empty working directory (so no project CLAUDE.md is loaded).
    """

    def __init__(self, model: str = "claude-opus-5", effort: str = "low", batch_size: int = 40,
                 max_budget_usd: float = 1.0, binary: str = "claude", timeout: float = 600,
                 max_workers: int = 4):
        super().__init__(model, effort, batch_size, max_workers)
        self.binary = binary
        self.max_budget_usd = max_budget_usd
        self.timeout = timeout
        self.name = f"claude-code/{model}"
        self._cwd = tempfile.mkdtemp(prefix="jevquant-claude-")

    def _request(self, states, questions):
        cmd = [self.binary, "-p", "--model", self.model, "--effort", self.effort,
               "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--max-budget-usd", str(self.max_budget_usd),
               "--system-prompt", SYSTEM_PROMPT.format(questions=render_questions(questions)),
               "--json-schema", json.dumps(answer_schema(questions))]
        proc = subprocess.run(cmd, input=render_items(states), capture_output=True, text=True,
                              timeout=self.timeout, cwd=self._cwd, env={**os.environ})
        try:
            res = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {proc.stderr[-500:]}") from e
        if res.get("is_error") or "structured_output" not in res:
            raise RuntimeError(f"claude -p error: {str(res)[:500]}")
        usage = res.get("usage", {})
        return (res["structured_output"]["items"], float(res.get("total_cost_usd", 0.0)),
                {"input_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                 + usage.get("cache_creation_input_tokens", 0),
                 "output_tokens": usage.get("output_tokens", 0)})
