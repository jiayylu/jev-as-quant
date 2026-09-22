import numpy as np

from jevquant.backtest import build_market, decision_calendar, run_backtest
from jevquant.data.synthetic import REGIMES, SyntheticConfig, generate
from jevquant.judges import (FixedReader, MarketObs, NewsJudge, RegimeJudge, RiskJudge, RuleReader,
                             SignalJudge)
from jevquant.risk import RiskConfig, RiskManager
from jevquant.typed import ChoiceAnswer
from jevquant.strategies import BuyAndHold, NewsStrategy, RouterStrategy, SignalStrategy
from jevquant.verbalize import describe, raw_state


def test_synthetic_market_is_reproducible_and_has_all_regimes():
    a = generate(SyntheticConfig(n_assets=3, n_days=1500, seed=7))
    b = generate(SyntheticConfig(n_assets=3, n_days=1500, seed=7))
    assert a.prices["SYN0"].equals(b.prices["SYN0"]) and a.events.equals(b.events)
    seen = set().union(*[set(df["regime"]) for df in a.prices.values()])
    assert seen == set(REGIMES)
    assert (a.prices["SYN1"][["open", "high", "low", "close"]] > 0).all().all()
    ev = a.events
    assert (np.sign(ev.loc[ev.label == "bullish", "impact"]) > 0).all()
    assert (ev.loc[ev.label == "neutral", "impact"] == 0).all()


def test_news_oracle_captures_the_planted_edge():
    mkt = generate(SyntheticConfig(n_assets=4, n_days=800, seed=3, news_rate=0.2))
    m = build_market(mkt.prices)
    truth = dict(zip(mkt.events["text"], mkt.events["label"]))
    oracle = FixedReader("oracle", lambda t: {"sentiment": ChoiceAnswer(
        choice=truth[t], probabilities={k: float(k == truth[t]) for k in NewsJudge.labels})})
    res = run_backtest(m, NewsStrategy(oracle, mkt.events), m.dates,
                       RiskManager(RiskConfig(long_only=False)), cost_bps=0, slippage_bps=0)
    assert res.equity.iloc[-1] > 1.2


def test_verbalizer_describes_but_never_concludes():
    mkt = generate(SyntheticConfig(n_assets=1, n_days=400, seed=1))
    m = build_market(mkt.prices)
    rows = m.features["SYN0"].dropna()
    for _, row in rows.iloc[::25].iterrows():
        text = describe(row).lower()
        for forbidden in ["uptrend", "downtrend", "crisis", "buy", "sell", "bullish", "bearish"]:
            assert forbidden not in text
        assert "rsi(14)" in text and len(text.split()) < 120
        assert set(raw_state(row)) >= {"rsi14", "dist_sma50"}


def test_rule_judges_answer_every_question_and_strategies_run():
    mkt = generate(SyntheticConfig(n_assets=3, n_days=600, seed=2))
    m = build_market(mkt.prices)
    row = m.features["SYN0"].dropna().iloc[-1]
    for judge in [RegimeJudge(), SignalJudge(), RiskJudge()]:
        ans = judge.baseline(MarketObs("SYN0", None, row))
        assert set(ans) == set(judge.questions)
    dates = decision_calendar(m.dates, 5, start=260)
    for strat in [BuyAndHold(), SignalStrategy(RuleReader(SignalJudge())),
                  RouterStrategy(RuleReader(RegimeJudge()))]:
        res = run_backtest(m, strat, dates)
        assert np.isfinite(res.equity).all() and len(res.equity) == len(m.dates)


def test_rule_regime_beats_chance_on_synthetic_truth():
    mkt = generate(SyntheticConfig(n_assets=4, n_days=1500, seed=5))
    m = build_market(mkt.prices)
    judge, hits, n = RegimeJudge(), 0, 0
    for s in m.symbols:
        f = m.features[s].dropna().iloc[::5]
        truth = m.extra["regime"][s]
        for d, row in f.iterrows():
            hits += judge.baseline(MarketObs(s, d, row))["regime"].choice == truth.loc[d]
            n += 1
    assert hits / n > 0.4  # four classes, chance ~0.25-0.35 given class imbalance


def test_lexicon_baseline():
    j = NewsJudge()
    assert j.baseline("$SAN: Deutsche Bank cuts to Hold")["sentiment"].choice == "bearish"
    assert j.baseline("Apple beats estimates, raises guidance")["sentiment"].choice == "bullish"
    assert j.baseline("Company to present at a conference")["sentiment"].choice == "neutral"
