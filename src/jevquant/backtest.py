"""Daily event loop with no look-ahead.

Timeline for each trading day t:

  open(t)   execute the target decided at close(t-1); pay commission + slippage on turnover
  close(t)  mark to market; risk tracker sees equity; if t is a decision date the strategy
            reads features computed up to close(t) and proposes a target for open(t+1)

Holdings drift with prices between rebalances. Long/short positions and cash are supported;
cash earns nothing. `tests/test_backtest.py` checks that a strategy peeking at tomorrow's
return is *not* rewarded (its trades land one bar late) and that costs are charged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .features import compute_features
from .risk import RiskManager


@dataclass
class Market:
    open: pd.DataFrame  # date x symbol
    close: pd.DataFrame
    features: dict[str, pd.DataFrame]
    extra: dict = field(default_factory=dict)  # e.g. ground-truth regimes for synthetic data

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def close_returns(self) -> pd.DataFrame:
        return self.close.pct_change()


def build_market(prices: dict[str, pd.DataFrame]) -> Market:
    idx = None
    for df in prices.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    opens = pd.DataFrame({s: df.loc[idx, "open"] for s, df in prices.items()})
    closes = pd.DataFrame({s: df.loc[idx, "close"] for s, df in prices.items()})
    feats = {s: compute_features(df.loc[idx]) for s, df in prices.items()}
    extra = {}
    if all("regime" in df.columns for df in prices.values()):
        extra["regime"] = pd.DataFrame({s: df.loc[idx, "regime"] for s, df in prices.items()})
    return Market(open=opens, close=closes, features=feats, extra=extra)


class Strategy(Protocol):
    name: str

    def prepare(self, market: Market, decision_dates: pd.DatetimeIndex) -> None: ...

    def target(self, date: pd.Timestamp) -> pd.Series | None: ...


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series  # at each close
    weights: pd.DataFrame  # holdings at each close (after drift)
    turnover: pd.Series  # traded fraction of equity at each open
    costs: pd.Series
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


def decision_calendar(dates: pd.DatetimeIndex, every: str | int = "W-FRI", start: int = 0) -> pd.DatetimeIndex:
    """Decision dates: every n-th bar, or the last trading day of each pandas period (e.g. "W-FRI")."""
    dates = dates[start:]
    if isinstance(every, int):
        return dates[::every]
    last = pd.Series(dates, index=dates).groupby(dates.to_period(every)).last()
    return pd.DatetimeIndex(last.values)


def run_backtest(market: Market, strategy: Strategy, decision_dates: pd.DatetimeIndex,
                 risk: RiskManager | None = None, cost_bps: float = 5.0, slippage_bps: float = 2.0) -> BacktestResult:
    risk = risk or RiskManager()
    strategy.prepare(market, decision_dates)
    dates = market.dates
    syms = market.symbols
    O = market.open.values
    C = market.close.values
    rets_hist = market.close_returns
    decide = set(decision_dates)
    fee = (cost_bps + slippage_bps) / 1e4

    w = np.zeros(len(syms))
    equity = 1.0
    pending: np.ndarray | None = None
    eq, W, TO, CO = [], [], [], []
    for i, d in enumerate(dates):
        # --- open: execute yesterday's decision
        to = cost = 0.0
        if pending is not None:
            to = float(np.abs(pending - w).sum())
            cost = to * fee
            equity *= 1 - cost
            w = pending.copy()
            pending = None
        # --- open -> close
        r = np.nan_to_num(C[i] / O[i] - 1)
        pr = float(w @ r)
        equity *= 1 + pr
        if 1 + pr > 0:
            w = w * (1 + r) / (1 + pr)
        killed = risk.on_close(equity, i)
        # --- close: decide
        if d in decide or killed:
            tgt = strategy.target(d) if d in decide else None
            if tgt is None:  # no view: keep the drifted book
                tgt = pd.Series(w, index=syms)
            tgt = risk.apply(tgt.reindex(syms).fillna(0.0), d, i, rets_hist.loc[:d].iloc[-120:])
            pending = tgt.values.astype(float)
        eq.append(equity)
        W.append(w.copy())
        TO.append(to)
        CO.append(cost)
        # --- close -> next open
        if i + 1 < len(dates):
            r = np.nan_to_num(O[i + 1] / C[i] - 1)
            pr = float(w @ r)
            equity *= 1 + pr
            if 1 + pr > 0:
                w = w * (1 + r) / (1 + pr)
    return BacktestResult(
        name=strategy.name,
        equity=pd.Series(eq, index=dates),
        weights=pd.DataFrame(W, index=dates, columns=syms),
        turnover=pd.Series(TO, index=dates),
        costs=pd.Series(CO, index=dates),
        meta={"n_vetoes": risk.n_vetoes, "n_kills": risk.n_kills},
    )
