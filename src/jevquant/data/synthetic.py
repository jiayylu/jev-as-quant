"""A synthetic market where the truth is known.

Each asset follows its own hidden Markov chain over four regimes (uptrend, downtrend, range,
crisis). Headlines arrive at random; their *text* is drawn from a pool of real, human-labeled
financial tweets, and their *price impact* follows the human label (bullish up, bearish down,
neutral none). So we can measure exactly how much of the available news edge a reader
(Laya, a lexicon, TF-IDF, Claude...) captures, and how well each one recovers the regime.

Timing: a headline is published after the close of day t. Part of the move happens in the
overnight gap into day t+1 (not capturable by a trader who acts at the next open); the rest
drifts in over the following `drift_days` sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIMES = ("uptrend", "downtrend", "range", "crisis")

# daily log-return drift, volatility, OU mean-reversion speed (range only), mean duration (days)
REGIME_SPECS = {
    "uptrend": dict(mu=0.0010, sigma=0.010, kappa=0.0, duration=120),
    "downtrend": dict(mu=-0.0010, sigma=0.014, kappa=0.0, duration=80),
    "range": dict(mu=0.0, sigma=0.008, kappa=0.10, duration=90),
    "crisis": dict(mu=-0.0060, sigma=0.035, kappa=0.0, duration=15),
}
# where a regime goes when it ends
NEXT_REGIME = {
    "uptrend": {"range": 0.5, "downtrend": 0.3, "crisis": 0.2},
    "downtrend": {"uptrend": 0.3, "range": 0.4, "crisis": 0.3},
    "range": {"uptrend": 0.45, "downtrend": 0.45, "crisis": 0.10},
    "crisis": {"downtrend": 0.4, "range": 0.3, "uptrend": 0.3},
}


@dataclass
class SyntheticConfig:
    n_assets: int = 4
    n_days: int = 1260  # five trading years
    seed: int = 0
    rho: float = 0.4  # loading on a common market shock
    overnight_share: float = 0.3
    news_rate: float = 1 / 12  # headlines per asset per day
    impact_range: tuple[float, float] = (0.01, 0.04)  # absolute log impact of a non-neutral headline
    news_gap_share: float = 0.3  # share of the impact already in the next open
    drift_days: int = 5
    start: str = "2020-01-01"


@dataclass
class SyntheticMarket:
    prices: dict[str, pd.DataFrame]  # open, high, low, close, volume, regime
    events: pd.DataFrame  # date, symbol, text, label, impact
    config: SyntheticConfig = field(default_factory=SyntheticConfig)

    @property
    def symbols(self) -> list[str]:
        return list(self.prices)


def _draw_regime_path(rng: np.random.Generator, n_days: int) -> list[str]:
    reg = rng.choice(["uptrend", "downtrend", "range"])
    path = []
    for _ in range(n_days):
        path.append(reg)
        if rng.random() < 1.0 / REGIME_SPECS[reg]["duration"]:
            nxt = NEXT_REGIME[reg]
            reg = rng.choice(list(nxt), p=list(nxt.values()))
    return path


def generate(config: SyntheticConfig | None = None, headline_pool: pd.DataFrame | None = None) -> SyntheticMarket:
    """Build a market. `headline_pool` needs columns text, label in {bullish, bearish, neutral}."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, periods=cfg.n_days, name="date")
    symbols = [f"SYN{i}" for i in range(cfg.n_assets)]
    market_z = rng.standard_normal(cfg.n_days)

    if headline_pool is None:
        headline_pool = _toy_pool()
    pool = headline_pool.reset_index(drop=True)

    prices, events = {}, []
    for sym in symbols:
        regimes = _draw_regime_path(rng, cfg.n_days)
        z = cfg.rho * market_z + np.sqrt(1 - cfg.rho ** 2) * rng.standard_normal(cfg.n_days)
        gap_news = np.zeros(cfg.n_days)
        drift_news = np.zeros(cfg.n_days)
        news_today = np.zeros(cfg.n_days, dtype=bool)
        for t in range(cfg.n_days - 1):
            if rng.random() < cfg.news_rate:
                row = pool.iloc[int(rng.integers(len(pool)))]
                sign = {"bullish": 1.0, "bearish": -1.0}.get(row["label"], 0.0)
                impact = sign * rng.uniform(*cfg.impact_range)
                events.append({"date": dates[t], "symbol": sym, "text": row["text"],
                               "label": row["label"], "impact": impact})
                news_today[t] = True
                gap_news[t + 1] += cfg.news_gap_share * impact
                days = range(t + 1, min(cfg.n_days, t + 1 + cfg.drift_days))
                for d in days:
                    drift_news[d] += (1 - cfg.news_gap_share) * impact / cfg.drift_days

        log_c = np.zeros(cfg.n_days)
        log_o = np.zeros(cfg.n_days)
        anchor = 0.0
        prev_reg = None
        lc_prev = np.log(100.0)
        for t in range(cfg.n_days):
            reg = regimes[t]
            spec = REGIME_SPECS[reg]
            if reg != prev_reg:
                anchor = lc_prev
                prev_reg = reg
            mu = spec["mu"] - spec["kappa"] * (lc_prev - anchor)
            day = mu + spec["sigma"] * z[t]
            log_o[t] = lc_prev + cfg.overnight_share * day + gap_news[t]
            log_c[t] = log_o[t] + (1 - cfg.overnight_share) * day + drift_news[t]
            lc_prev = log_c[t]

        sig = np.array([REGIME_SPECS[r]["sigma"] for r in regimes])
        wick = np.abs(rng.standard_normal((2, cfg.n_days))) * 0.4 * sig
        o, c = np.exp(log_o), np.exp(log_c)
        high = np.maximum(o, c) * np.exp(wick[0])
        low = np.minimum(o, c) * np.exp(-wick[1])
        crisis = np.array([r == "crisis" for r in regimes])
        volume = 1e6 * np.exp(0.5 * np.abs(z) + 0.2 * rng.standard_normal(cfg.n_days))
        volume *= np.where(crisis, 2.5, 1.0) * np.where(np.roll(news_today, 1), 1.8, 1.0)
        prices[sym] = pd.DataFrame({"open": o, "high": high, "low": low, "close": c,
                                    "volume": volume.round(), "regime": regimes}, index=dates)

    ev = pd.DataFrame(events, columns=["date", "symbol", "text", "label", "impact"])
    return SyntheticMarket(prices=prices, events=ev, config=cfg)


def _toy_pool() -> pd.DataFrame:
    """Tiny built-in pool so the generator works offline (tests); experiments use real headlines."""
    rows = [("Company beats estimates and raises full-year guidance", "bullish"),
            ("Analyst upgrades the stock to Buy", "bullish"),
            ("Shares fall after the company cuts its outlook", "bearish"),
            ("Regulator opens probe into accounting practices", "bearish"),
            ("Company to present at industry conference next week", "neutral"),
            ("Board schedules annual shareholder meeting", "neutral")]
    return pd.DataFrame(rows, columns=["text", "label"])
