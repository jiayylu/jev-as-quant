import numpy as np
import pandas as pd

from jevquant.backtest import Market, decision_calendar, run_backtest
from jevquant.risk import RiskConfig, RiskManager


def _market(n=300, seed=0, k=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    closes, opens = {}, {}
    for j in range(k):
        r = rng.normal(0, 0.01, n)
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[100, c[:-1]] * np.exp(rng.normal(0, 0.003, n))
        closes[f"A{j}"], opens[f"A{j}"] = c, o
    return Market(open=pd.DataFrame(opens, index=idx), close=pd.DataFrame(closes, index=idx), features={})


class Peeker:
    """Cheats: at close t it looks at the return of day t+1 (open->close) and bets on it."""
    name = "peeker"

    def prepare(self, market, dates):
        self.m = market
        fwd = (market.close / market.open - 1).shift(-1)
        self.sig = np.sign(fwd).fillna(0.0) / market.close.shape[1]

    def target(self, date):
        return self.sig.loc[date]


class Oracle:
    """Legit only if the engine executes at t's close; used to prove it does not."""
    name = "same-bar"

    def prepare(self, market, dates):
        self.sig = np.sign(market.close / market.open - 1).fillna(0.0) / market.close.shape[1]

    def target(self, date):
        return self.sig.loc[date]


def test_decision_at_close_executes_next_open():
    m = _market()
    dates = m.dates
    same_bar = run_backtest(m, Oracle(), dates, RiskManager(RiskConfig(long_only=False)), 0, 0)
    # knowing today's candle at today's close is worthless: the trade lands tomorrow
    assert abs(same_bar.equity.iloc[-1] - 1) < 0.5
    peek = run_backtest(m, Peeker(), dates, RiskManager(RiskConfig(long_only=False)), 0, 0)
    # peeking one day ahead *is* rewarded, which proves the one-bar execution lag
    assert peek.equity.iloc[-1] > 3


def test_costs_are_charged_on_turnover():
    m = _market()

    class Flip:
        name = "flip"

        def prepare(self, market, dates):
            self.i = 0

        def target(self, date):
            self.i += 1
            return pd.Series([0.5 * (self.i % 2), 0.0], index=["A0", "A1"])

    free = run_backtest(m, Flip(), m.dates, cost_bps=0, slippage_bps=0)
    paid = run_backtest(m, Flip(), m.dates, cost_bps=10, slippage_bps=0)
    assert paid.costs.sum() > 0 and paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert abs(paid.turnover.sum() - free.turnover.sum()) < 1.0


def test_buy_and_hold_matches_asset_return():
    m = _market(k=1)

    class Hold:
        name = "hold"

        def prepare(self, market, dates):
            pass

        def target(self, date):
            return pd.Series([1.0], index=["A0"])

    res = run_backtest(m, Hold(), m.dates[:1], cost_bps=0, slippage_bps=0)
    expected = m.close["A0"].iloc[-1] / m.open["A0"].iloc[1]
    assert abs(res.equity.iloc[-1] - expected) < 1e-9


def test_weekly_calendar_takes_last_bar_of_week():
    idx = pd.bdate_range("2024-01-01", periods=15)
    cal = decision_calendar(idx, "W-FRI")
    assert all(d.weekday() == 4 for d in cal[:2])
    assert list(decision_calendar(idx, 5)) == list(idx[::5])
