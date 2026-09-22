"""Laya (Convai Innovations, Apache-2.0): local, open-weight System-1 decision model.

`system_one` calls the official `laya` runtime unchanged. `system_one_many` packs the
questions of many states into one padded forward pass — the same computation as
`laya.Agent.system_one`, just batched across states — which is what makes multi-year
backtests affordable on a laptop. `tests/test_laya_live.py` checks the two paths agree.
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Sequence

import numpy as np

from ..typed import Decision, Questions, State, parse_answers, questions_to_wire
from .base import Engine

CHECKPOINTS = {
    # name used here -> subfolder inside the convaiinnovations/laya hub repo
    "english": None,              # ModernBERT-large, 421M, 512 tokens
    "typed-decisions": "typed-decisions",  # ModernBERT-large fine-tuned on typed workflows, 1024 tokens
    "multilingual": "multilingual",        # mmBERT-base, 322M, 1024 tokens
}


def _local_snapshot(repo: str, subfolder: str | None) -> str:
    """Resolve an already-downloaded checkpoint without touching the network (laya.load always
    re-checks the Hub otherwise, which fails offline or behind a flaky proxy)."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    patterns = [prefix + n for n in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")]
    try:
        return snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
    except Exception:
        return repo  # not cached yet: let laya download it


class LayaEngine(Engine):
    def __init__(self, checkpoint: str = "typed-decisions", repo: str = "convaiinnovations/laya",
                 device: str | None = None, batch_size: int = 32):
        os.environ.setdefault("USE_TF", "0")  # model card: loading can hang if TF is importable
        import laya  # heavy import (torch, transformers) only when this engine is used

        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown Laya checkpoint {checkpoint!r}; choose from {list(CHECKPOINTS)}")
        sub = CHECKPOINTS[checkpoint]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.agent = laya.load(_local_snapshot(repo, sub), subfolder=sub, device=device)
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.name = f"laya/{checkpoint}"
        self.device = str(self.agent.device)

    @property
    def fingerprint(self) -> str:
        # device changes numerics only at float noise level; not part of the key
        return f"{self.name}@laya-{self._laya_version()}"

    @staticmethod
    def _laya_version() -> str:
        import laya
        return getattr(laya, "__version__", "?")

    # ------------------------------------------------------------------ latency path
    def system_one(self, state: State, questions: Questions) -> Decision:
        t0 = time.perf_counter()
        raw = self.agent.predict(state, questions_to_wire(questions))
        dt = (time.perf_counter() - t0) * 1000
        return Decision(answers=parse_answers(questions, raw["answers"]), engine=self.name,
                        model=raw.get("model", ""), latency_ms=dt, usage=raw.get("usage", {}))

    # ------------------------------------------------------------------ throughput path
    def system_one_many(self, states: Sequence[State], questions: Questions) -> list[Decision]:
        if not states:
            return []
        out: list[Decision] = []
        per_state = max(1, len(questions))
        chunk = max(1, self.batch_size // per_state)
        for i in range(0, len(states), chunk):
            out.extend(self._forward(states[i:i + chunk], questions))
        return out

    def _forward(self, states: Sequence[State], questions: Questions) -> list[Decision]:
        import torch
        from laya.common import (QTYPES, build_sequence, collate_items, confidence_from_probs,
                                 render_options, temp_bucket)

        ag = self.agent
        wire = questions_to_wire(questions)
        qids = list(wire)
        internal = {qid: ag._to_internal(wire[qid]) for qid in qids}
        max_len = ag.cfg.get("max_len", 512)
        head_max_len = ag.cfg.get("head_max_len", 192)

        groups = []
        for st in states:
            items = []
            for qid in qids:
                q = internal[qid]
                seq, markers = build_sequence(ag.tok, st, q, max_len, head_max_len)
                if len(markers) != len(render_options(q)):
                    raise ValueError(f"question {qid!r} options exceed head_max_len={head_max_len}")
                items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
            groups.append(items)

        t0 = time.perf_counter()
        b = collate_items(groups, ag.tok.pad_token_id)
        use_amp = ag.device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type=ag.device.type, dtype=ag.dtype, enabled=use_amp):
            logits, _act = ag.model(b["input_ids"].to(ag.device), b["attention_mask"].to(ag.device),
                                    b["marker_pos"].to(ag.device), b["marker_mask"].to(ag.device),
                                    b["qtype"].to(ag.device))
        logits = logits.float().cpu().numpy()
        dt = (time.perf_counter() - t0) * 1000 / len(states)  # amortized per state

        decisions = []
        row = 0
        for s_idx, items in enumerate(groups):
            raw = {}
            for qid, it in zip(qids, items):
                q = internal[qid]
                k = len(it["markers"])
                qt = QTYPES[q["t"]]
                t_scale = ag.temperature_by_options.get(temp_bucket(qt, k), ag.temperature[qt])
                z = logits[row, :k] / t_scale
                p = np.exp(z - z.max())
                p = p / p.sum()
                row += 1
                if q["t"] == "choice":
                    keys = list(q["crit"].keys())
                    raw[qid] = {"choice": keys[int(p.argmax())],
                                "probabilities": {kk: float(v) for kk, v in zip(keys, p)},
                                "confidence": confidence_from_probs(p, k)}
                elif q["t"] == "score":
                    raw[qid] = {"score": float((np.arange(k) * p).sum()),
                                "probabilities": {str(j): float(v) for j, v in enumerate(p)},
                                "confidence": confidence_from_probs(p, k)}
                else:
                    raw[qid] = {"noul": float(p[1])}
            n_tok = int(b["attention_mask"][s_idx * len(qids):(s_idx + 1) * len(qids)].sum())
            decisions.append(Decision(answers=parse_answers(questions, raw), engine=self.name,
                                      model="laya-rl-agent", latency_ms=dt,
                                      usage={"input_tokens": n_tok, "output_tokens": 0}))
        return decisions
