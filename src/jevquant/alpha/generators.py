"""Where candidate alphas come from.

  seeds     textbook anomalies written once, before any result was seen
  random    random expressions from a small grammar (the classic brute-force alpha search);
            they pay for their volume with a Bonferroni threshold on the whole family
  claude    Claude proposes hypotheses in the DSL, each with an economic rationale, after
            reading the research ledger (what has worked, what has failed)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile

from .dsl import WINDOWS, AlphaError, complexity, validate

SEEDS = {
    # price / volume
    "momentum 12-1": "delay(close, 21) / delay(close, 252) - 1",
    "momentum 6-1": "delay(close, 21) / delay(close, 126) - 1",
    "reversal 1w": "-pct(close, 5)",
    "reversal 1m": "-pct(close, 21)",
    "low volatility": "-ts_std(ret, 63)",
    "52-week high": "close / ts_max(close, 252)",
    "lottery (MAX)": "-ts_max(ret, 21)",
    "volume surge": "log(ts_mean(volume, 5) / ts_mean(volume, 63))",
    "illiquidity": "ts_mean(abs(ret) / dollar_vol, 21)",
    "skewness": "-ts_skew(ret, 63)",
    # fundamentals (point-in-time)
    "earnings yield": "ep", "book to market": "bm", "operating profitability": "op_prof",
    "gross profitability": "gp_prof", "low accruals": "-accruals", "low asset growth": "-asset_growth",
    "revenue growth": "rev_growth", "earnings surprise (SUE)": "sue", "small size": "-size",
}

TS_OPS = ["ts_mean", "ts_std", "ts_z", "ts_rank", "pct", "delta", "decay", "ts_max", "ts_min"]


def random_alphas(n: int, inputs: list[str], seed: int, fundamentals: set[str] = frozenset()) -> list[str]:
    """Random expressions from five template families, all within the DSL limits."""
    rng = random.Random(seed)
    price = [i for i in inputs if i not in fundamentals]
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        w = lambda: rng.choice(WINDOWS[3:])
        x, y = rng.choice(price), rng.choice(inputs)
        kind = rng.randrange(5)
        if kind == 0:
            e = f"{rng.choice(TS_OPS)}({x}, {w()})"
        elif kind == 1:
            e = f"{rng.choice(TS_OPS)}({x}, {w()}) {rng.choice('-/')} {rng.choice(TS_OPS)}({x}, {w()})"
        elif kind == 2:
            e = f"rank({rng.choice(TS_OPS)}({x}, {w()})) * rank({y if y in fundamentals else rng.choice(TS_OPS) + f'({y}, {w()})'})"
        elif kind == 3:
            e = f"ts_corr({x}, {rng.choice(price)}, {rng.choice([21, 42, 63, 126])})"
        else:
            e = f"ts_z({x}, {w()}) * {rng.choice(['rank', 'zscore'])}({y})"
        if rng.random() < 0.5:
            e = f"-({e})"
        try:
            validate(e, set(inputs))
        except AlphaError:
            continue
        if e not in out:
            out.append(e)
    return out


PROMPT = """You are an equity quant researcher proposing new cross-sectional alphas for S&P 500 stocks.

Horizon: the signal is computed at Friday's close and held for one week from Monday's open;
the portfolio is long-only top 50 names, rebalanced weekly, 10 bp per side. Large caps are
efficient: single classic factors mostly fail here; think about interactions, conditioning,
events, and behavioural or institutional frictions that plausibly survive.

Write each alpha in this DSL (higher value = expected to outperform):
  inputs: {inputs}
  time-series (per stock): ts_mean ts_std ts_sum ts_min ts_max ts_skew ts_rank ts_z delay delta pct decay (x, n)
                           ts_corr(x, y, n); n must be one of {windows}
  cross-section (per date): rank(x) zscore(x) demean(x)
  element-wise: abs sign log neg max(x, y) min(x, y) clip(x, lo, hi), + - * /, numeric constants
  at most {max_nodes} nodes and depth {max_depth}.

Input meanings: open/close are split- and dividend-adjusted prices, ret is the daily close-to-close
return, volume is shares, dollar_vol = close * volume. Fundamentals are point-in-time (visible
after filing): ep = net income / public float, bm = equity / public float, op_prof and gp_prof
= operating / gross profit over assets, accruals = (net income - operating cash flow) / assets,
asset_growth and rev_growth are year over year, sue = (quarterly net income - same quarter last
year) / assets, size = log public float.{news}

Research ledger so far (learn from it; do not resubmit or trivially rescale what is there):
{ledger}

Propose {k} new, diverse alphas. Give each a one-sentence economic rationale."""

SCHEMA = {"type": "object", "properties": {"alphas": {"type": "array", "items": {
    "type": "object", "properties": {"expr": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["expr", "rationale"], "additionalProperties": False}}},
    "required": ["alphas"], "additionalProperties": False}


def claude_alphas(k: int, inputs: list[str], ledger_summary: str, news_note: str = "",
                  model: str = "claude-opus-5", effort: str = "medium", budget_usd: float = 1.5) -> tuple[list[dict], float]:
    """Ask Claude (headless Claude Code) for k DSL alphas with rationales. Returns (alphas, cost)."""
    from .dsl import MAX_DEPTH, MAX_NODES

    prompt = PROMPT.format(inputs=", ".join(inputs), windows=WINDOWS, max_nodes=MAX_NODES, max_depth=MAX_DEPTH,
                           news=news_note, ledger=ledger_summary, k=k)
    cmd = ["claude", "-p", "--model", model, "--effort", effort, "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--max-budget-usd", str(budget_usd), "--json-schema", json.dumps(SCHEMA)]
    import time

    last = None
    for attempt in range(3):
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900,
                              cwd=tempfile.mkdtemp(prefix="jevquant-alpha-"), env={**os.environ})
        try:
            res = json.loads(proc.stdout)
        except json.JSONDecodeError:
            res = {"is_error": True, "result": proc.stdout[-300:] + proc.stderr[-300:]}
        if not res.get("is_error") and "structured_output" in res:
            return res["structured_output"]["alphas"], float(res.get("total_cost_usd", 0.0))
        last = {k: res.get(k) for k in ("subtype", "stop_reason", "api_error_status", "result")}
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"claude -p failed 3 times: {str(last)[:400]}")
