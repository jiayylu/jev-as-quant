"""Shared plumbing for the experiment scripts: paths, cached engines, JSON output."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORTS = ROOT / "reports"
DATA_OUT = REPORTS / "data"
CACHE = ROOT / ".cache" / "decisions.sqlite"
DATA_CACHE = ROOT / "data_cache"
REPORTS.mkdir(exist_ok=True)
DATA_OUT.mkdir(exist_ok=True)

_ENGINES: dict = {}


class _LazyLaya:
    """Loads the checkpoint only on the first cache miss, so fully cached reruns need no GPU."""

    def __init__(self, checkpoint, device):
        import laya as _laya

        self.checkpoint, self.device = checkpoint, device
        self.name = f"laya/{checkpoint}"
        self.fingerprint = f"{self.name}@laya-{getattr(_laya, '__version__', '?')}"  # == LayaEngine.fingerprint

    def _engine(self):
        from jevquant.engines.laya import LayaEngine

        key = (self.checkpoint, self.device)
        if key not in _ENGINES:
            _ENGINES[key] = LayaEngine(self.checkpoint, device=self.device)
        return _ENGINES[key]

    def system_one(self, state, questions):
        return self._engine().system_one(state, questions)

    def system_one_many(self, states, questions):
        return self._engine().system_one_many(states, questions)


def laya(checkpoint: str = "typed-decisions", cached: bool = True, device: str | None = None):
    """Laya engine wrapped in the on-disk decision cache (or the bare engine, for latency tests)."""
    from jevquant.engines.cache import CachedEngine

    lazy = _LazyLaya(checkpoint, device)
    return CachedEngine(lazy, CACHE) if cached else lazy._engine()


def claude(model: str = "claude-opus-5", effort: str = "low"):
    from jevquant.engines.cache import CachedEngine
    from jevquant.engines.claude import ClaudeCodeEngine

    return CachedEngine(ClaudeCodeEngine(model=model, effort=effort, batch_size=40, max_workers=4,
                                         max_budget_usd=1.0), CACHE)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def save(name: str, obj) -> Path:
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
