"""E1 - How fast is Laya on this machine, really?

Latency path: the official `laya` single-state call, 1 / 3 / 6 questions, MPS and CPU.
Throughput path: our batched forward over 64 states (amortized per state).
Compared against the published numbers (Laya: 32.8 ms on a T4; Jev: 70-500 ms end to end).
"""
from __future__ import annotations

import platform
import statistics

import numpy as np

from common import laya, save
from jevquant.data.synthetic import SyntheticConfig, generate
from jevquant.backtest import build_market
from jevquant.judges import MarketObs, NewsJudge, RegimeJudge, RiskJudge, SignalJudge
from jevquant.verbalize import describe

REPS = 20


def main():
    mkt = build_market(generate(SyntheticConfig(n_assets=1, n_days=600, seed=0)).prices)
    rows = mkt.features["SYN0"].dropna()
    states = [describe(r) for _, r in rows.iloc[::5].iterrows()][:64]
    qsets = {
        1: RegimeJudge.questions,
        3: {**SignalJudge.questions, **{"risk_off": RiskJudge.questions["risk_off"]}},
        6: {**SignalJudge.questions, **RiskJudge.questions, **RegimeJudge.questions,
            "sentiment": NewsJudge.questions["sentiment"]},
    }
    headline = "$AAPL - Apple beats estimates and raises its full-year revenue outlook"

    results = []
    import torch
    devices = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    for ckpt in ["english", "typed-decisions"]:
        for dev in devices:
            eng = laya(ckpt, cached=False, device=dev)
            for n_q, qs in qsets.items():
                for _ in range(3):
                    eng.system_one(states[0], qs)  # warm-up
                lat = [eng.system_one(states[i % len(states)], qs).latency_ms for i in range(REPS)]
                results.append({"checkpoint": ckpt, "device": dev, "path": "single", "questions": n_q,
                                "state": "market (~110 tokens)", "p50_ms": statistics.median(lat),
                                "p95_ms": float(np.percentile(lat, 95)), "reps": REPS})
                print(results[-1])
            lat = [eng.system_one(headline, NewsJudge.questions).latency_ms for _ in range(REPS)]
            results.append({"checkpoint": ckpt, "device": dev, "path": "single", "questions": 1,
                            "state": "headline (~20 tokens)", "p50_ms": statistics.median(lat),
                            "p95_ms": float(np.percentile(lat, 95)), "reps": REPS})
            print(results[-1])
            for n_q in (1, 3):
                qs = qsets[n_q]
                eng.system_one_many(states[:8], qs)  # warm-up
                per_state = []
                for _ in range(3):
                    ds = eng.system_one_many(states, qs)
                    per_state.append(ds[0].latency_ms)
                results.append({"checkpoint": ckpt, "device": dev, "path": "batched(64 states)",
                                "questions": n_q, "state": "market (~110 tokens)",
                                "p50_ms": statistics.median(per_state), "p95_ms": max(per_state), "reps": 3})
                print(results[-1])
            del eng
            from common import _ENGINES
            _ENGINES.clear()
            if dev == "mps":
                torch.mps.empty_cache()

    save("e1_latency", {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "chip": "Apple M3 Pro (18 GB)", "torch": torch.__version__},
        "published": {"laya_t4_single_ms": 32.8, "laya_t4_batched10_per_q_ms": 7.2,
                      "laya_cpu_preloaded_ms": [193, 464], "jev_end_to_end_ms": [70, 500]},
        "results": results,
    })


if __name__ == "__main__":
    main()
