"""Strategies turn typed judgments into target weights. Thresholds live here, in code.

Every judgment-driven strategy is written against a `Reader`, so the same strategy runs on
Laya, Jev, Claude, a cascade or the rule baseline. `prepare()` reads all decision dates in
one batch (judgments only depend on data up to that close, so this is look-ahead free and
lets Laya batch on the GPU).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Market
from .judges import MarketObs, Reader
from .typed import Decision


def _read_market(reader: Reader, market: Market, dates) -> dict[tuple, Decision]:
    obs, keys = [], []
    for d in dates:
        for s in market.symbols:
            row = market.features[s].loc[d]
            if row.isna().any():
                continue
            obs.append(MarketObs(symbol=s, date=d, row=row))
            keys.append((d, s))
    decisions = reader.read(obs)
    return dict(zip(keys, decisions))


class BuyAndHold:
    def __init__(self, name: str = "buy&hold (equal weight)"):
        self.name = name

    def prepare(self, market, decision_dates):
        self.syms = market.symbols

    def target(self, date):
        return pd.Series(1.0 / len(self.syms), index=self.syms)


class SignalStrategy:
    """Use case 1. Argmax of the typed action: buy -> 1/N, hold -> hold_weight/N, sell -> short_weight/N."""

    def __init__(self, reader: Reader, hold_weight: float = 0.5, short_weight: float = 0.0, name: str | None = None):
        self.reader = reader
        self.hold_weight = hold_weight
        self.short_weight = short_weight
        self.name = name or f"signal[{reader.name}]"

    def prepare(self, market, decision_dates):
        self.syms = market.symbols
        self.decisions = _read_market(self.reader, market, decision_dates)

    def target(self, date):
        n = len(self.syms)
        w = {}
        for s in self.syms:
            d = self.decisions.get((date, s))
            if d is None:
                continue
            a = d["action"].choice
            w[s] = (1.0 if a == "buy" else self.hold_weight if a == "hold" else self.short_weight) / n
        return pd.Series(w, dtype=float).reindex(self.syms).fillna(0.0)

    def conviction(self) -> pd.DataFrame:
        """Continuous signal P(buy) - P(sell), for threshold-free IC analysis."""
        rows = [(d, s, dec["action"].p("buy") - dec["action"].p("sell"), dec["trend"].signed)
                for (d, s), dec in self.decisions.items()]
        return pd.DataFrame(rows, columns=["date", "symbol", "action_edge", "trend"])


class RouterStrategy:
    """Use case 3. The regime judgment picks a sub-strategy per asset.

    uptrend -> long, downtrend -> short (or flat if long-only), range -> fade the 20-day
    z-score, crisis -> flat.
    """

    def __init__(self, reader: Reader, long_only: bool = False, name: str | None = None):
        self.reader = reader
        self.long_only = long_only
        self.name = name or f"router[{reader.name}]"

    def prepare(self, market, decision_dates):
        self.market = market
        self.syms = market.symbols
        self.decisions = _read_market(self.reader, market, decision_dates)

    def regime(self, date, sym) -> str | None:
        d = self.decisions.get((date, sym))
        return None if d is None else d["regime"].choice

    def target(self, date):
        n = len(self.syms)
        w = {}
        for s in self.syms:
            reg = self.regime(date, s)
            if reg is None:
                continue
            if reg == "uptrend":
                x = 1.0
            elif reg == "downtrend":
                x = -1.0
            elif reg == "range":
                z = self.market.features[s].loc[date, "zscore20"]
                x = float(-np.clip(z / 2.0, -1, 1))
            else:
                x = 0.0
            if self.long_only:
                x = max(x, 0.0)
            w[s] = x / n
        return pd.Series(w, dtype=float).reindex(self.syms).fillna(0.0)


class NewsStrategy:
    """Use case 4. Trade each headline for `hold_days` sessions in the direction of its label.

    `events` needs columns date, symbol, text. `date` is the *decision* date: the trading day
    whose close the headline is read at, so it is traded from the next open. Pass `signals`
    (+1 / -1 / 0 per event) to reuse labels already computed, e.g. for placebo permutations.
    With `market_neutral`, the equal-weight universe is shorted against the net position every
    day, so the book earns only the stock-specific (abnormal) part of each move.
    """

    def __init__(self, reader, events: pd.DataFrame, hold_days: int = 5, name: str | None = None,
                 signals=None, market_neutral: bool = False):
        self.reader = reader
        self.events = events.reset_index(drop=True)
        self.hold_days = hold_days
        self.signals = None if signals is None else np.asarray(signals, dtype=float)
        self.market_neutral = market_neutral
        self.name = name or f"news[{getattr(reader, 'name', 'given signals')}]"

    def prepare(self, market, decision_dates):
        self.syms = market.symbols
        self.dates = market.dates
        if self.signals is None:
            decisions = self.reader.read(list(self.events["text"])) if len(self.events) else []
            sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
            self.events["signal"] = [sign[d["sentiment"].choice] for d in decisions]
            self.decisions = decisions
        else:
            self.events["signal"] = self.signals
        pos = {d: i for i, d in enumerate(self.dates)}
        book = np.zeros((len(self.dates), len(self.syms)))
        col = {s: j for j, s in enumerate(self.syms)}
        for ev in self.events.itertuples():
            i = pos.get(ev.date)
            j = col.get(ev.symbol)
            if i is None or j is None or ev.signal == 0:
                continue
            book[i:i + self.hold_days, j] += ev.signal
        book = np.clip(book, -1, 1) / len(self.syms)
        if self.market_neutral:
            book = book - book.mean(axis=1, keepdims=True)
        self.book = pd.DataFrame(book, index=self.dates, columns=self.syms)

    def target(self, date):
        return self.book.loc[date]


def risk_veto_from_reader(reader: Reader, market: Market, dates, threshold: float = 0.6, floor: float = 0.0):
    """Use case 2. Build a reduce-only veto: multiplier `floor` where P(risk_off) > threshold, else 1."""
    decisions = _read_market(reader, market, dates)
    table: dict = {}
    for (d, s), dec in decisions.items():
        table.setdefault(d, {})[s] = floor if dec["risk_off"].noul > threshold else 1.0
    frames = {d: pd.Series(v, dtype=float) for d, v in table.items()}

    def veto(date):
        return frames.get(date, pd.Series(dtype=float))

    veto.decisions = decisions  # type: ignore[attr-defined]
    return veto
