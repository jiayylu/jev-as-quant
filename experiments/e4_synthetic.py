"""E4 - A market where the truth is known.

Seed 0 is the development seed (thresholds are chosen there); seeds 1-3 are the test seeds
that are reported. Each seed: 4 assets, 6 trading years, regime-switching prices, and
headlines whose text is a real labeled tweet (validation split) and whose price impact follows
the human label.

  A  regime recognition   Laya (verbalized features) vs rules vs truth           use case 3
  B  regime router        long/short book routed by each reader's regime call    use case 3
  C  news trading         lexicon / TF-IDF / Laya / cascade / oracle             use case 4
  D  risk veto            router[rules] with no veto, rules veto, Laya veto      use case 2
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_CACHE, DATA_OUT, claude, laya, save
from jevquant.backtest import build_market, decision_calendar, run_backtest
from jevquant.calibration import CalibratedReader, ChoiceCalibrator, probs_matrix
from jevquant.data.news import load_headlines
from jevquant.data.synthetic import REGIMES, SyntheticConfig, generate
from jevquant.engines.cascade import CascadeEngine
from jevquant.judges import (FixedReader, MarketObs, ModelReader, NewsJudge, RegimeJudge, RiskJudge,
                             RuleReader, TfidfNewsReader)
from jevquant.metrics import block_bootstrap_sharpe, performance
from jevquant.risk import RiskConfig, RiskManager
from jevquant.strategies import BuyAndHold, NewsStrategy, RouterStrategy, risk_veto_from_reader
from jevquant.typed import ChoiceAnswer

DEV_SEED, TEST_SEEDS = 0, [1, 2, 3]
WARMUP = 260
LS = RiskConfig(long_only=False, max_weight=0.5, max_gross=1.0)


def market(seed, pool):
    syn = generate(SyntheticConfig(n_assets=4, n_days=1512, seed=seed), headline_pool=pool)
    return syn, build_market(syn.prices)


def regime_accuracy(reader, m, dates):
    obs = [MarketObs(s, d, m.features[s].loc[d]) for d in dates for s in m.symbols]
    ds = reader.read(obs)
    pred = [d["regime"].choice for d in ds]
    truth = [m.extra["regime"].loc[o.date, o.symbol] for o in obs]
    cm = pd.crosstab(pd.Series(truth, name="truth"), pd.Series(pred, name="pred")).reindex(
        index=list(REGIMES), columns=list(REGIMES), fill_value=0)
    recall = {r: float(cm.loc[r, r] / cm.loc[r].sum()) if cm.loc[r].sum() else float("nan") for r in REGIMES}
    return {"accuracy": float(np.mean([p == t for p, t in zip(pred, truth)])),
            "balanced_accuracy": float(np.nanmean(list(recall.values()))), "recall": recall,
            "confusion": cm.values.tolist()}


def summarize(res, with_ci=False):
    # score only the live period (after the indicator warm-up), not the idle first year
    p = performance(res.equity.iloc[WARMUP:], res.turnover.iloc[WARMUP:], res.weights.iloc[WARMUP:])
    if with_ci:
        p["sharpe_ci95"] = block_bootstrap_sharpe(res.returns.iloc[WARMUP:])
    return p


def main():
    val = load_headlines("validation", DATA_CACHE)
    train = load_headlines("train", DATA_CACHE)
    cal_set = train.groupby("label").sample(frac=1500 / len(train), random_state=0).reset_index(drop=True)

    laya_td, laya_en = laya("typed-decisions"), laya("english")
    readers_regime = {"rules": RuleReader(RegimeJudge()),
                      "laya/english": ModelReader(RegimeJudge(), laya_en),
                      "laya/typed-decisions": ModelReader(RegimeJudge(), laya_td)}

    # news readers: fast engine + calibrator chosen in E3 on training data (re-fit here, same data)
    import json
    from common import REPORTS
    e3 = json.loads((REPORTS / "e3_news.json").read_text())
    best = e3["fast_engine"].split("/")[1].split(" ")[0]
    fast = laya(best)
    cal = ChoiceCalibrator(NewsJudge.labels).fit(
        probs_matrix(fast.system_one_many(list(cal_set["text"]), NewsJudge.questions), "sentiment", NewsJudge.labels),
        cal_set["label"])
    laya_news = CalibratedReader(ModelReader(NewsJudge(), fast), "sentiment", cal, name=f"laya/{best}+cal")

    class CalibratedEngine:  # the cascade must escalate on *calibrated* certainty, as in E3
        name = f"laya/{best}+cal"
        fingerprint = name

        def system_one_many(self, states, questions):
            return laya_news.read(states)

    cascade = CascadeEngine(CalibratedEngine(), claude(), threshold=e3["cascade_threshold_from_train_q20"])
    news_readers = {"lexicon": RuleReader(NewsJudge()), "tf-idf": TfidfNewsReader(train),
                    f"laya/{best}+cal": laya_news, "cascade->claude": ModelReader(NewsJudge(), cascade)}

    out = {"dev_seed": DEV_SEED, "test_seeds": TEST_SEEDS, "news_fast_engine": f"laya/{best}+cal",
           "regime": {}, "router": {}, "news": {}, "risk": {}, "equity_seed1": {}}

    # ---------------------------------------------------------------- D (dev): choose the veto threshold
    syn, m = market(DEV_SEED, val)
    dates = decision_calendar(m.dates, 5, start=WARMUP)
    risk_reader = ModelReader(RiskJudge(), laya_td)
    grid = {}
    for tau in [0.4, 0.5, 0.6, 0.7, 0.8]:
        veto = risk_veto_from_reader(risk_reader, m, dates, threshold=tau)
        res = run_backtest(m, RouterStrategy(RuleReader(RegimeJudge())), dates, RiskManager(LS, veto))
        grid[tau] = performance(res.equity)["sharpe"]
    tau_star = max(grid, key=grid.get)
    out["risk"]["dev_grid_sharpe"] = grid
    out["risk"]["tau"] = tau_star
    print("veto threshold chosen on dev seed:", tau_star, grid)

    for seed in TEST_SEEDS:
        syn, m = market(seed, val)
        dates = decision_calendar(m.dates, 5, start=WARMUP)
        daily = m.dates[WARMUP:]
        truth = m.extra["regime"]

        # A: regime recognition
        out["regime"][seed] = {k: regime_accuracy(r, m, dates) for k, r in readers_regime.items()}
        print(seed, {k: round(v["balanced_accuracy"], 3) for k, v in out["regime"][seed].items()})

        # B: router backtests
        oracle = FixedReader("oracle", lambda o: {"regime": ChoiceAnswer(
            choice=truth.loc[o.date, o.symbol],
            probabilities={r: float(r == truth.loc[o.date, o.symbol]) for r in REGIMES})})
        strategies = {"buy&hold": BuyAndHold(), **{f"router[{k}]": RouterStrategy(r) for k, r in readers_regime.items()},
                      "router[oracle]": RouterStrategy(oracle)}
        out["router"][seed] = {}
        for name, strat in strategies.items():
            res = run_backtest(m, strat, dates, RiskManager(LS))
            out["router"][seed][name] = summarize(res)
            if seed == 1:
                out["equity_seed1"][name] = res.equity.iloc[WARMUP:].round(5).tolist()

        # C: news backtests (daily decisions)
        ev = syn.events[syn.events["date"] >= daily[0]].reset_index(drop=True)
        truth_news = FixedReader("oracle", lambda t, lab=dict(zip(ev["text"], ev["label"])): {
            "sentiment": ChoiceAnswer(choice=lab[t], probabilities={k: float(k == lab[t]) for k in NewsJudge.labels})})
        out["news"][seed] = {"n_events": int(len(ev)),
                             "event_mix": ev["label"].value_counts().to_dict()}
        cascade.n_seen = cascade.n_escalated = 0
        for name, reader in {**news_readers, "oracle": truth_news}.items():
            strat = NewsStrategy(reader, ev)
            res = run_backtest(m, strat, daily, RiskManager(LS), cost_bps=5, slippage_bps=2)
            perf = summarize(res)
            correct = np.mean([s == {"bullish": 1, "bearish": -1, "neutral": 0}[l]
                               for s, l in zip(strat.events["signal"], ev["label"])])
            perf["headline_accuracy"] = float(correct)
            # noise-free view: share of the planted post-news drift the reader's calls would earn
            # (sum of signal x impact over sum of |impact|; wrong-way calls count negative)
            perf["planted_edge_capture"] = float((strat.events["signal"] * ev["impact"]).sum() / ev["impact"].abs().sum())
            if name.startswith("cascade"):
                perf["escalation_rate"] = cascade.escalation_rate
            out["news"][seed][name] = perf
        o_ret = out["news"][seed]["oracle"]["total_return"]
        for name in out["news"][seed]:
            if isinstance(out["news"][seed][name], dict) and "total_return" in out["news"][seed][name]:
                r = out["news"][seed][name]["total_return"]
                out["news"][seed][name]["capture_vs_oracle"] = float(np.log1p(r) / np.log1p(o_ret))
        print(seed, {k: round(v.get("capture_vs_oracle", 0), 3) for k, v in out["news"][seed].items() if isinstance(v, dict) and "sharpe" in v})

        # D: risk veto on top of the rules router
        veto_laya = risk_veto_from_reader(risk_reader, m, dates, threshold=tau_star)
        veto_rules = risk_veto_from_reader(RuleReader(RiskJudge()), m, dates, threshold=0.5)
        out["risk"][seed] = {}
        for name, veto in {"no veto": None, "rules veto": veto_rules, "laya veto": veto_laya}.items():
            rm = RiskManager(LS, veto)
            res = run_backtest(m, RouterStrategy(RuleReader(RegimeJudge())), dates, rm)
            perf = summarize(res)
            perf["vetoes"] = rm.n_vetoes
            if veto is not None:
                dec = veto.decisions
                flags = np.array([dec[k]["risk_off"].noul > (tau_star if name == "laya veto" else 0.5) for k in dec])
                in_crisis = np.array([truth.loc[d, s] == "crisis" for (d, s) in dec])
                perf["veto_precision_crisis"] = float(in_crisis[flags].mean()) if flags.any() else float("nan")
                perf["veto_recall_crisis"] = float(flags[in_crisis].mean()) if in_crisis.any() else float("nan")
            out["risk"][seed][name] = perf
        print(seed, {k: (round(v["sharpe"], 2), round(v["max_drawdown"], 3)) for k, v in out["risk"][seed].items()})

    save("e4_synthetic", out)


if __name__ == "__main__":
    main()
