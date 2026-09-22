import numpy as np
import pandas as pd
import pytest

from jevquant.alpha.dsl import AlphaError, Evaluator, complexity, validate
from jevquant.alpha.factory import Factory, weekly_portfolio
from jevquant.alpha.lab import Lab, Ledger, rowwise_spearman
from jevquant.alpha.panel import Panel


def synthetic_panel(n_days=1400, n=120, seed=0, planted=0.08):
    """Prices where next-week returns load on a hidden, observable characteristic `quality`."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2013-01-01", periods=n_days)
    tick = [f"S{i:03d}" for i in range(n)]
    quality = pd.DataFrame(np.cumsum(rng.normal(0, 0.05, (n_days, n)), axis=0) + rng.normal(0, 1, n), index=days, columns=tick)
    q_rank = quality.rank(axis=1, pct=True) - 0.5
    ret = rng.normal(0.0003, 0.015, (n_days, n)) + planted * 0.01 * q_rank.shift(1).fillna(0).values
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=days, columns=tick)
    open_ = close.shift(1).fillna(100.0) * np.exp(rng.normal(0, 0.002, (n_days, n)))
    member = pd.DataFrame(True, index=days, columns=tick)
    spy = pd.DataFrame({"open": open_.mean(axis=1), "close": close.mean(axis=1)})
    p = Panel(open=open_, close=close, volume=close * 0 + 1e6, member=member, spy=spy)
    return p, {"close": close, "open": open_, "ret": close.pct_change(), "quality": quality,
               "noise": pd.DataFrame(rng.normal(size=(n_days, n)), index=days, columns=tick)}


def test_dsl_rejects_anything_outside_the_whitelist():
    inputs = {"close", "volume"}
    for bad in ["__import__('os')", "close.values", "open", "ts_mean(close, 7)", "lambda: 1", "close[0]",
                "ts_mean(close, n=5)", "exec('1')"]:
        with pytest.raises(AlphaError):
            validate(bad, inputs)
    validate("-pct(close, 5) + rank(ts_mean(volume, 21))", inputs)
    assert complexity("-pct(close, 5)") == (3, 3)


def test_evaluator_matches_pandas_and_ranks_members_only():
    p, inputs = synthetic_panel(n_days=300, n=20)
    mem = p.member.copy()
    mem.iloc[:, :5] = False
    ev = Evaluator(inputs, mem)
    assert np.allclose(ev("pct(close, 5)").values[10:], (p.close / p.close.shift(5) - 1).values[10:], equal_nan=True)
    r = ev("rank(close)")
    assert r.iloc[:, :5].isna().all().all() and r.iloc[-1].dropna().between(-0.5, 0.5).all()


def test_gates_accept_a_planted_alpha_and_reject_noise(tmp_path):
    p, inputs = synthetic_panel()
    lab = Lab(p)
    # rescale the splits to the synthetic calendar
    import jevquant.alpha.lab as L
    old = dict(L.SPLITS)
    L.SPLITS.update({"discovery": ("2013-01-01", "2016-12-31"), "validation": ("2017-01-01", "2018-06-30")})
    try:
        fac = Factory(lab, Evaluator(inputs, p.member), Ledger(tmp_path / "l.sqlite"))
        assert fac.consider("rank(quality)", "seed", "test", "planted", 0) == "accepted"
        assert fac.consider("rank(noise)", "seed", "test", "noise", 0) == "rejected"
        assert fac.consider("rank(quality) * 2", "seed", "test", "clone", 0) == "rejected"  # redundant
    finally:
        L.SPLITS.clear(); L.SPLITS.update(old)


def test_adaptive_weights_only_use_realized_labels(tmp_path):
    p, inputs = synthetic_panel(n_days=700, n=60)
    lab = Lab(p)
    fac = Factory(lab, Evaluator(inputs, p.member), Ledger(tmp_path / "l.sqlite"))
    fac.library["q"] = Evaluator(inputs, p.member)("rank(quality)")
    fac.composite(adaptive=True, min_weeks=10)
    w = fac.weights["q"]
    ic = lab.ic_series(fac.library["q"])
    # recompute the weight at one date by hand from labels realized strictly before it
    t = lab.dates[80]
    known = lab.label_known_on
    x = ic[(ic.index < t) & (known <= t)].dropna().values[-156:]
    assert np.isclose(w.loc[t], max(0, x.mean() / x.std() * np.sqrt(52)))
    # and changing a *future* label must not change today's weight
    lab.label.iloc[81:] = lab.label.iloc[81:] * -1
    fac2 = Factory(lab, fac.evaluator, Ledger(tmp_path / "l2.sqlite"), library={"q": fac.library["q"]})
    fac2.composite(adaptive=True, min_weeks=10)
    assert np.isclose(fac2.weights["q"].loc[t], w.loc[t])


def test_weekly_portfolio_beats_benchmark_on_planted_alpha():
    p, inputs = synthetic_panel(planted=0.3)
    lab = Lab(p)
    score = lab.weekly(Evaluator(inputs, p.member)("rank(quality)"))
    wk = weekly_portfolio(score, lab, n=20, buffer=30, cost_bps=0)
    ex = (wk["portfolio"] - wk["ew_members"]).dropna()
    assert ex.mean() > 0 and wk["turnover"].iloc[1:].mean() < 0.5


def test_rowwise_spearman():
    a = pd.DataFrame([[1, 2, 3, 4] * 10], dtype=float)
    assert np.isclose(rowwise_spearman(a, a).iloc[0], 1.0)
    assert np.isclose(rowwise_spearman(a, -a).iloc[0], -1.0)
