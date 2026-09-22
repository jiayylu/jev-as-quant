"""Turn a feature row into the text a System-1 model reads.

Design rule (see docs/03-architecture.md): code does the arithmetic, the model does the
judgment. So each indicator is described on its own, as a number plus a fixed qualitative
word ("4.2% above", "RSI 72, overbought zone") — but the verbalizer never states an
aggregate conclusion such as "uptrend" or "buy". Combining the cues is the model's job,
and the rule baselines do the same combining in hand-written code. That keeps the
Laya-vs-rules comparison fair.

`raw_state` is the ablation: the same numbers as JSON, with no words (experiment E2).
"""
from __future__ import annotations

import math

import pandas as pd


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _band(x: float, cuts: list[tuple[float, str]], top: str) -> str:
    for edge, word in cuts:
        if x < edge:
            return word
    return top


def _vs_average(dist: float, name: str) -> str:
    word = _band(dist, [(-0.05, "far below"), (-0.01, "below"), (0.01, "close to"), (0.05, "above")],
                 "far above")
    if word == "close to":
        return f"Price is close to its {name} ({_pct(dist)})."
    return f"Price is {abs(dist) * 100:.1f}% {word} its {name}."


def _slope(s: float, name: str) -> str:
    word = _band(s, [(-0.01, "falling fast"), (-0.002, "falling"), (0.002, "flat"), (0.01, "rising")],
                 "rising fast")
    return f"The {name} is {word} ({_pct(s)})."


def describe(row: pd.Series, symbol: str | None = None) -> str:
    """One compact paragraph (≈90 tokens) describing a feature row."""
    parts = []
    if symbol:
        parts.append(f"{symbol} daily chart.")
    parts.append(_vs_average(row["dist_sma50"], "50-day average"))
    parts.append(_vs_average(row["dist_sma20"], "20-day average"))
    parts.append(_slope(row["sma50_slope"], "50-day average over two weeks"))
    r = row["rsi14"]
    rsi_word = _band(r, [(30, "oversold zone"), (45, "weak"), (55, "neutral"), (70, "strong")], "overbought zone")
    parts.append(f"RSI(14) is {r:.0f}, {rsi_word}.")
    parts.append(f"Returns: {_pct(row['ret_5d'])} over 5 days, {_pct(row['ret_20d'])} over 20 days, "
                 f"{_pct(row['ret_60d'])} over 60 days.")
    vr = row["vol_ratio"]
    if not math.isnan(vr):
        vol_word = _band(vr, [(0.7, "unusually low"), (1.3, "normal"), (2.0, "elevated")], "extremely high")
        parts.append(f"Volatility is {vol_word}: {row['vol_20d'] * 100:.0f}% annualized, "
                     f"{vr:.1f}x its one-year norm.")
    dd = row["dd_60d"]
    if dd > -0.02:
        parts.append("Price is at or near its 60-day high.")
    else:
        parts.append(f"Price is {abs(dd) * 100:.0f}% below its 60-day high.")
    vol_r = row["volume_ratio"]
    vw = _band(vol_r, [(0.7, "light"), (1.4, "normal"), (2.2, "heavy")], "very heavy")
    parts.append(f"Trading volume is {vw} ({vol_r:.1f}x its 20-day average).")
    return " ".join(parts)


def raw_state(row: pd.Series, symbol: str | None = None) -> dict:
    """Ablation: the same information as bare numbers (what an API user would naively send)."""
    d = {k: round(float(row[k]), 4) for k in ["close", "dist_sma20", "dist_sma50", "sma50_slope", "rsi14",
                                             "ret_5d", "ret_20d", "ret_60d", "vol_20d", "vol_ratio",
                                             "dd_60d", "volume_ratio"]}
    if symbol:
        d = {"symbol": symbol, **d}
    return d
