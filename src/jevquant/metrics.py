"""Performance, predictive-power and calibration metrics."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance(equity: pd.Series, turnover: pd.Series | None = None, weights: pd.DataFrame | None = None) -> dict:
    r = equity.pct_change().dropna()
    years = len(r) / TRADING_DAYS
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    vol = float(r.std() * math.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * math.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * math.sqrt(TRADING_DAYS)) if downside and downside > 0 else float("nan")
    dd = equity / equity.cummax() - 1
    mdd = float(dd.min())
    out = {"total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
           "max_drawdown": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else float("nan")}
    if turnover is not None:
        out["turnover_per_year"] = float(turnover.sum() / years) if years else float("nan")
    if weights is not None:
        out["avg_gross"] = float(weights.abs().sum(axis=1).mean())
    return out


def block_bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000, block: int = 20, seed: int = 0) -> tuple[float, float]:
    """95% CI for the annualized Sharpe ratio via a moving-block bootstrap."""
    r = returns.dropna().values
    n = len(r)
    if n < block * 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    stats = []
    for _ in range(n_boot):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)).ravel()[:n]
        x = r[idx]
        sd = x.std()
        stats.append(x.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 3) -> dict:
    """Cross-sectional Spearman IC per date (signal & fwd: date x symbol), with a t-stat."""
    ics = []
    for d in signal.index.intersection(fwd.index):
        a, b = signal.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= min_names and a[m].nunique() > 1:
            ics.append(a[m].rank().corr(b[m].rank()))
    ics = pd.Series(ics, dtype=float).dropna()
    if len(ics) < 3:
        return {"ic_mean": float("nan"), "ic_t": float("nan"), "n": len(ics)}
    return {"ic_mean": float(ics.mean()), "ic_t": float(ics.mean() / ics.std() * math.sqrt(len(ics))),
            "hit_rate": float((ics > 0).mean()), "n": int(len(ics))}


def pooled_rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    return float(s["x"].rank().corr(s["y"].rank())) if len(s) > 2 else float("nan")


# ----------------------------------------------------------------------------- classification


def classification_report(y_true: Sequence[str], probs: np.ndarray, labels: Sequence[str]) -> dict:
    """Accuracy, macro-F1, multiclass Brier, NLL, top-label ECE."""
    from sklearn.metrics import confusion_matrix, f1_score

    labels = list(labels)
    y = np.array([labels.index(t) for t in y_true])
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1)
    p = p / p.sum(1, keepdims=True)
    pred = p.argmax(1)
    onehot = np.eye(len(labels))[y]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(len(labels))))),
        "brier": float(((p - onehot) ** 2).sum(1).mean()),
        "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
        "ece": ece(p.max(1), pred == y),
        "confusion": confusion_matrix(y, pred, labels=list(range(len(labels)))).tolist(),
        "labels": labels,
    }


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            e += m.mean() * abs(confidence[m].mean() - correct[m].mean())
    return float(e)


def reliability_bins(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            rows.append({"lo": lo, "hi": hi, "confidence": float(confidence[m].mean()),
                         "accuracy": float(np.asarray(correct)[m].mean()), "n": int(m.sum())})
    return pd.DataFrame(rows)
