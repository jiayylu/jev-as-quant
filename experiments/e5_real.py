"""E5 - Real markets: eight liquid US-listed ETFs, weekly decisions, long-only, with costs.

Universe: SPY QQQ IWM EFA EEM TLT GLD VNQ (equities, bonds, gold, real estate).
Data: Yahoo Finance daily bars (split/dividend adjusted), 2013-01 to 2026-08; decisions every
Friday close from 2014, executed at Monday's open; 5 bps commission + 2 bps slippage.

Nothing is fitted on this data: judges, thresholds and the veto threshold come from the
synthetic dev seed (E4). Laya reads only the verbalized technical state - there is no
historical news feed here, so this is the "numbers-only" setting where we expect the model to
add the least.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_CACHE, REPORTS, laya, save
from jevquant.backtest import build_market, decision_calendar, run_backtest
from jevquant.data.yahoo import load_ohlcv
from jevquant.judges import ModelReader, RegimeJudge, RiskJudge, RuleReader, SignalJudge
from jevquant.metrics import block_bootstrap_sharpe, information_coefficient, performance, pooled_rank_corr
from jevquant.risk import RiskConfig, RiskManager
from jevquant.strategies import BuyAndHold, RouterStrategy, SignalStrategy, risk_veto_from_reader

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"]
START, END, FIRST_DECISION = "2013-01-01", "2026-09-01", "2014-01-01"
LONG_ONLY = RiskConfig(long_only=True, max_weight=0.25, max_gross=1.0)


def main():
    prices = load_ohlcv(UNIVERSE, START, END, DATA_CACHE)
    m = build_market(prices)
    first = int(np.searchsorted(m.dates, pd.Timestamp(FIRST_DECISION)))
    dates = decision_calendar(m.dates, "W-FRI", start=first)
    live = slice(first, None)
    tau = json.loads((REPORTS / "e4_synthetic.json").read_text())["risk"]["tau"]

    td, en = laya("typed-decisions"), laya("english")
    readers = {"rules": RuleReader(SignalJudge()), "laya/english": ModelReader(SignalJudge(), en),
               "laya/typed-decisions": ModelReader(SignalJudge(), td)}
    strategies = {"buy&hold (equal weight)": BuyAndHold()}
    for k, r in readers.items():
        strategies[f"signal[{k}]"] = SignalStrategy(r)
    strategies["router[rules]"] = RouterStrategy(RuleReader(RegimeJudge()), long_only=True)
    strategies["router[laya/typed-decisions]"] = RouterStrategy(ModelReader(RegimeJudge(), td), long_only=True)

    out = {"universe": UNIVERSE, "period": [str(m.dates[first].date()), str(m.dates[-1].date())],
           "n_decisions": len(dates), "veto_tau_from_synthetic_dev": tau, "strategies": {}, "equity": {},
           "drawdown": {}, "ic": {}, "agreement": {}}
    results = {}
    for name, strat in strategies.items():
        res = run_backtest(m, strat, dates, RiskManager(LONG_ONLY))
        results[name] = (strat, res)
    # risk veto on top of the rules signal: rules veto vs Laya veto
    for vname, vreader, thr in [("rules veto", RuleReader(RiskJudge()), 0.5),
                                ("laya veto", ModelReader(RiskJudge(), td), tau)]:
        veto = risk_veto_from_reader(vreader, m, dates, threshold=thr)
        rm = RiskManager(LONG_ONLY, veto)
        strat = SignalStrategy(RuleReader(SignalJudge()), name=f"signal[rules] + {vname}")
        res = run_backtest(m, strat, dates, rm)
        res.meta["vetoes"] = rm.n_vetoes
        results[strat.name] = (strat, res)

    for name, (strat, res) in results.items():
        eq = res.equity.iloc[live]
        perf = performance(eq, res.turnover.iloc[live], res.weights.iloc[live])
        perf["sharpe_ci95"] = block_bootstrap_sharpe(eq.pct_change().dropna())
        perf.update({k: v for k, v in res.meta.items() if k == "vetoes"})
        out["strategies"][name] = perf
        weekly = eq.resample("W-FRI").last()
        out["equity"][name] = {"dates": [str(d.date()) for d in weekly.index], "values": weekly.round(5).tolist()}
        dd = (eq / eq.cummax() - 1).resample("W-FRI").min()
        out["drawdown"][name] = dd.round(5).tolist()
        print(f"{name:40s} sharpe={perf['sharpe']:.2f} cagr={perf['cagr']:.3f} mdd={perf['max_drawdown']:.3f} "
              f"to={perf['turnover_per_year']:.1f}")

    # predictive power of the continuous signals: next-week open-to-open return
    fwd = {}
    opens = m.open
    for i, d in enumerate(dates[:-1]):
        a = m.dates.get_loc(d) + 1
        b = m.dates.get_loc(dates[i + 1]) + 1
        if b < len(m.dates):
            fwd[d] = opens.iloc[b] / opens.iloc[a] - 1
    fwd = pd.DataFrame(fwd).T
    for k in readers:
        conv = results[f"signal[{k}]"][0].conviction()
        for col in ["action_edge", "trend"]:
            sig = conv.pivot(index="date", columns="symbol", values=col)
            ic = information_coefficient(sig, fwd)
            s_, r_ = sig.stack(), fwd.stack()
            s_.index.names = r_.index.names = ["date", "symbol"]
            long = pd.concat([s_.rename("s"), r_.rename("r")], axis=1, join="inner")
            ic["pooled_rank_corr"] = pooled_rank_corr(long["s"], long["r"])
            out["ic"][f"{k}:{col}"] = ic
            print(k, col, {kk: round(v, 4) if isinstance(v, float) else v for kk, v in ic.items()})

    # how often do Laya and the rules agree on the action?
    acts = {k: {key: d["action"].choice for key, d in results[f"signal[{k}]"][0].decisions.items()} for k in readers}
    keys = sorted(acts["rules"])
    for k in ["laya/english", "laya/typed-decisions"]:
        a = np.array([acts["rules"][x] for x in keys])
        b = np.array([acts[k][x] for x in keys])
        from sklearn.metrics import cohen_kappa_score
        out["agreement"][k] = {"agree": float((a == b).mean()), "cohen_kappa": float(cohen_kappa_score(a, b)),
                               "action_mix": pd.Series(b).value_counts(normalize=True).round(4).to_dict()}
    out["agreement"]["rules_action_mix"] = pd.Series(a).value_counts(normalize=True).round(4).to_dict()
    save("e5_real", out)


if __name__ == "__main__":
    main()
