"""Risk layer. Hard limits live in code; the model may only ever *reduce* exposure.

Order of operations for every target the strategy proposes:

  1. clip per-asset weights, enforce long-only if configured      (code)
  2. scale down to the gross-exposure cap                          (code)
  3. scale down to the volatility target, never up                 (code)
  4. model veto: multiply each weight by m in [0, 1]               (model, reduce-only)
  5. drawdown kill-switch: flat for `cooldown_days`                (code, overrides all)

Step 4 is where a Laya/Claude risk judgment plugs in. Because m is clipped to [0, 1] after
the hard limits, a wrong or adversarial model answer can cost upside but cannot push the
book past a limit. `tests/test_risk.py` checks this property on random inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

VetoFn = Callable[[pd.Timestamp], pd.Series]  # date -> multiplier per symbol in [0, 1]


@dataclass
class RiskConfig:
    max_weight: float = 1.0
    max_gross: float = 1.0
    long_only: bool = True
    vol_target: float | None = None  # annualized, e.g. 0.12
    vol_lookback: int = 60
    kill_drawdown: float | None = None  # e.g. 0.25 = go flat after a 25% drawdown
    cooldown_days: int = 20


class RiskManager:
    def __init__(self, config: RiskConfig | None = None, veto: VetoFn | None = None):
        self.cfg = config or RiskConfig()
        self.veto = veto
        self.peak = 1.0
        self.kill_until: int | None = None
        self.n_vetoes = 0
        self.n_kills = 0

    def limits(self, target: pd.Series, returns_hist: pd.DataFrame | None = None) -> pd.Series:
        """Steps 1-3: the hard limits."""
        c = self.cfg
        w = target.fillna(0.0).clip(-c.max_weight, c.max_weight)
        if c.long_only:
            w = w.clip(lower=0.0)
        gross = w.abs().sum()
        if gross > c.max_gross:
            w = w * (c.max_gross / gross)
        if c.vol_target and returns_hist is not None and len(returns_hist) >= 20:
            cov = returns_hist[w.index].tail(c.vol_lookback).cov().fillna(0.0).values
            vol = float(np.sqrt(max(w.values @ cov @ w.values, 0.0) * 252))
            if vol > c.vol_target:
                w = w * (c.vol_target / vol)
        return w

    def apply(self, target: pd.Series, date, day_index: int,
              returns_hist: pd.DataFrame | None = None) -> pd.Series:
        w = self.limits(target, returns_hist)
        if self.veto is not None:
            m = self.veto(date).reindex(w.index).fillna(1.0).clip(0.0, 1.0)
            self.n_vetoes += int((m < 1.0).sum())
            w = w * m
        if self.kill_until is not None and day_index < self.kill_until:
            w = w * 0.0
        return w

    def on_close(self, equity: float, day_index: int) -> bool:
        """Update the drawdown tracker; returns True when the kill-switch fires today."""
        self.peak = max(self.peak, equity)
        c = self.cfg
        if c.kill_drawdown is None:
            return False
        if self.kill_until is not None and day_index >= self.kill_until:
            self.kill_until = None
            self.peak = equity  # restart the drawdown clock after the cooldown
        if self.kill_until is None and equity < self.peak * (1 - c.kill_drawdown):
            self.kill_until = day_index + 1 + c.cooldown_days
            self.n_kills += 1
            return True
        return False
