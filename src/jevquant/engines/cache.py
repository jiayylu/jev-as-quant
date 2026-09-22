"""Content-addressed cache so reruns of an experiment never pay for the same judgment twice.

Key = sha256(engine fingerprint, state, questions). The cached Decision keeps the latency and
cost measured on the original call, flagged `cached=True`, so reports stay honest.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from ..typed import Decision, Questions, State, questions_to_wire
from .base import Engine


def cache_key(fingerprint: str, state: State, questions: Questions) -> str:
    blob = json.dumps([fingerprint, state, questions_to_wire(questions)], ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedEngine(Engine):
    def __init__(self, inner: Engine, path: str | Path = ".cache/decisions.sqlite"):
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS decisions (k TEXT PRIMARY KEY, v TEXT)")
        self.hits = 0
        self.misses = 0

    @property
    def fingerprint(self) -> str:
        return self.inner.fingerprint

    def _get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM decisions WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put_many(self, rows: list[tuple[str, Decision]]) -> None:
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO decisions VALUES (?, ?)",
                                 [(k, json.dumps(d.to_json())) for k, d in rows])
            self._db.commit()

    def system_one(self, state: State, questions: Questions) -> Decision:
        return self.system_one_many([state], questions)[0]

    def system_one_many(self, states: Sequence[State], questions: Questions) -> list[Decision]:
        fp = self.inner.fingerprint
        keys = [cache_key(fp, s, questions) for s in states]
        out: list[Decision | None] = [None] * len(states)
        todo: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            hit = self._get(k)
            if hit is not None:
                out[i] = Decision.from_json(questions, hit, cached=True)
                self.hits += 1
            else:
                todo.setdefault(k, []).append(i)  # duplicates inside one batch are computed once
        if todo:
            first = [idxs[0] for idxs in todo.values()]
            fresh = self.inner.system_one_many([states[i] for i in first], questions)
            self.misses += len(first)
            self._put_many([(keys[i], d) for i, d in zip(first, fresh)])
            for (k, idxs), d in zip(todo.items(), fresh):
                for j in idxs:
                    out[j] = d
        return out  # type: ignore[return-value]
