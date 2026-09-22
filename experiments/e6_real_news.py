"""E6 - Real company news, real prices, 2024-2026: does reading press releases with Laya pay?

News: every 8-K press release (EX-99.x) filed with the SEC since 2024-01-01 by the 453 current
S&P 500 members that joined the index before 2024 (collect_sec.py). Timestamps are the SEC
acceptance times (to the second). Prices: Yahoo daily bars, split/dividend adjusted.

Fixed before looking at any result:
  reading    Laya typed-decisions + the E3 calibrator reads "headline. lead paragraph"
  timing     accepted before 09:30 ET on a trading day -> that day's open, else the next open
             (robustness: always the next day's open)
  book       +1/N for bullish, -1/N for bearish, held 5 sessions, summed per stock and clipped;
             equal-weight hedge on the 453 names (market neutral); 5 bp + 2 bp per trade
  baselines  lexicon (a priori) · TF-IDF (trained on labeled tweets) · "every release bullish"
             (is there a press-release premium regardless of content?) · placebo = Laya's labels
             shuffled across releases (same trades and timing, content removed; 20 draws)
  cascade    the least certain releases (certainty < the E3 threshold) re-read by Claude

Also reported, as a non-tradable validity check: the market-adjusted move between the last
close before the release and the first tradable open (did the market agree with the reading?).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from common import DATA_CACHE, REPORTS, claude, laya, save
from jevquant.backtest import Market, run_backtest
from jevquant.calibration import ChoiceCalibrator, probs_matrix
from jevquant.data.news import load_headlines as load_tweets
from jevquant.data.sec import first_tradable_open
from jevquant.judges import NewsJudge, RuleReader, TfidfNewsReader
from jevquant.metrics import block_bootstrap_sharpe, performance
from jevquant.risk import RiskConfig, RiskManager
from jevquant.strategies import NewsStrategy

START, END = "2024-01-01", "2026-09-22"
HOLD = 5
LABELS = NewsJudge.labels
BOOK = RiskConfig(long_only=False, max_weight=1.0, max_gross=2.0)


# ------------------------------------------------------------------------------ data
def load_market() -> tuple[Market, list[str]]:
    uni = pd.read_csv(DATA_CACHE / "sec" / "universe.csv", dtype=str)
    opens, closes = {}, {}
    for sym in uni["symbol"]:
        y = sym.replace(".", "-")
        f = DATA_CACHE / "prices_sec" / f"{y}_2023-10-01_{END}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        opens[sym], closes[sym] = df["open"], df["close"]
    O, C = pd.DataFrame(opens), pd.DataFrame(closes)
    O = O.loc[O.index >= "2023-10-02"]
    C = C.loc[O.index]
    return Market(open=O, close=C, features={}), list(O.columns)


def load_events(m: Market, conservative: bool = False) -> pd.DataFrame:
    pr = pd.read_csv(DATA_CACHE / "sec" / "press_releases.csv.gz", dtype={"cik": str})
    pr = pr[pr["symbol"].isin(m.symbols)].copy()
    pr["accepted_et"] = pd.to_datetime(pr["accepted_utc"], utc=True).dt.tz_convert("America/New_York")
    pr["text"] = (pr["headline"].fillna("").str.strip() + ". " + pr["lead"].fillna("").str.strip()).str.slice(0, 1200)
    # cleaning rules (fixed before the run): News Corp's ASX buy-back forms are routine filings, not
    # news; a release filed for two share classes counts once; a headline refiled within 7 days once
    pr = pr[~pr["headline"].str.match(r"^Part \d+ [–-] ")]
    pr = pr.sort_values("accepted_et").drop_duplicates(["cik", "accession"])
    gap = pr.groupby(["symbol", "headline"])["accepted_et"].diff()
    pr = pr[~(gap < pd.Timedelta(days=7))]
    pr["exec_date"] = first_tradable_open(pr["accepted_et"].reset_index(drop=True), m.dates, conservative).values
    pr = pr.dropna(subset=["exec_date"])
    di = np.searchsorted(m.dates.values, pr["exec_date"].values, side="left") - 1
    pr = pr[di >= 0].copy()
    pr["date"] = m.dates.values[di[di >= 0]]  # decision close; the engine trades the next open
    return pr.sort_values("accepted_et").reset_index(drop=True)


# ------------------------------------------------------------------------------ readers
def calibrated_laya_probs(texts: list[str]) -> np.ndarray:
    e3 = json.loads((REPORTS / "e3_news.json").read_text())["calibrator_typed-decisions"]
    cal = ChoiceCalibrator(LABELS)
    cal.T, cal.b = float(e3["T"]), np.array([float(e3["bias"][k]) for k in LABELS])
    eng = laya("typed-decisions")
    out = []
    for i in range(0, len(texts), 2000):
        ds = eng.system_one_many(texts[i:i + 2000], NewsJudge.questions)
        out.append(cal.transform(probs_matrix(ds, "sentiment", LABELS)))
        print(f"  laya {min(i + 2000, len(texts)):,}/{len(texts):,}", flush=True)
    return np.vstack(out)


def certainty(P: np.ndarray) -> np.ndarray:
    return 1 + (P * np.log(np.clip(P, 1e-12, 1))).sum(1) / np.log(P.shape[1])


def to_signal(P: np.ndarray) -> np.ndarray:
    lab = P.argmax(1)
    return np.select([lab == LABELS.index("bullish"), lab == LABELS.index("bearish")], [1.0, -1.0], 0.0)


# ------------------------------------------------------------------------------ evaluation
def abnormal(m: Market, sym: str, i0: int, i1: int, a: str = "open", b: str = "open") -> float:
    """Stock return minus equal-weight universe return between two price points."""
    A = m.open if a == "open" else m.close
    B = m.open if b == "open" else m.close
    r = B.iloc[i1] / A.iloc[i0] - 1
    return float(r[sym] - r.mean())


def event_study(m: Market, ev: pd.DataFrame, sig: np.ndarray) -> dict:
    pos = {d: i for i, d in enumerate(m.dates)}
    rows = []
    for e, s in zip(ev.itertuples(), sig):
        i = pos.get(pd.Timestamp(e.exec_date))
        if i is None or i < 1 or s == 0 or i + 10 >= len(m.dates):
            continue
        # non-tradable: last close before the release became public -> first tradable open
        day = np.datetime64(e.accepted_et.tz_localize(None).normalize())
        k = int(np.searchsorted(m.dates.values, day, "left"))
        after_close = e.accepted_et.hour * 60 + e.accepted_et.minute >= 16 * 60
        pre = k if (k < len(m.dates) and m.dates.values[k] == day and after_close) else k - 1
        gap = abnormal(m, e.symbol, pre, i, "close", "open") if 0 <= pre < i else np.nan
        rows.append({"d": m.dates[i], "s": s, "gap": gap,
                     "r1": abnormal(m, e.symbol, i, i + 1), "r5": abnormal(m, e.symbol, i, i + 5),
                     "r10": abnormal(m, e.symbol, i, i + 10)})
    df = pd.DataFrame(rows)
    out = {"n_bullish": int((df.s == 1).sum()), "n_bearish": int((df.s == -1).sum())}
    for col in ["gap", "r1", "r5", "r10"]:
        x = df.assign(x=df.s * df[col]).dropna(subset=["x"]).groupby("d")["x"].mean()  # long bull / short bear
        out[col] = {"ls_mean_bp": float(x.mean() * 1e4), "t": float(x.mean() / x.std() * math.sqrt(len(x))),
                    "bull_mean_bp": float(df.loc[df.s == 1, col].mean() * 1e4),
                    "bear_mean_bp": float(df.loc[df.s == -1, col].mean() * 1e4) if (df.s == -1).any() else float("nan"),
                    "bull_hit": float((df.loc[df.s == 1, col] > 0).mean()),
                    "bear_hit": float((df.loc[df.s == -1, col] < 0).mean()) if (df.s == -1).any() else float("nan")}
    return out


def backtest(m: Market, ev: pd.DataFrame, sig: np.ndarray, hold: int = HOLD, name: str = "") -> tuple[dict, pd.Series]:
    strat = NewsStrategy(None, ev[["date", "symbol", "text"]], hold_days=hold, name=name, signals=sig,
                         market_neutral=True)
    res = run_backtest(m, strat, m.dates, RiskManager(BOOK), cost_bps=5, slippage_bps=2)
    t0 = int(np.searchsorted(m.dates, pd.Timestamp(START)))
    eq = res.equity.iloc[t0:] / res.equity.iloc[t0]
    perf = performance(eq, res.turnover.iloc[t0:], res.weights.iloc[t0:])
    perf["sharpe_ci95"] = block_bootstrap_sharpe(eq.pct_change().dropna())
    r = eq.pct_change().dropna()
    perf["sharpe_by_year"] = {str(y): float(g.mean() / g.std() * math.sqrt(252)) for y, g in r.groupby(r.index.year)}
    gross = float(res.weights.iloc[t0:].abs().sum(axis=1).mean())
    perf["avg_gross"] = gross
    perf["gross_cost_drag_per_year"] = float(res.costs.iloc[t0:].sum() / (len(eq) / 252))
    return perf, eq


def main():
    m, syms = load_market()
    ev = load_events(m)
    ev_cons = load_events(m, conservative=True)
    texts = list(ev["text"])
    print(f"universe {len(syms)} · press releases {len(ev):,} · {ev.accepted_et.min():%Y-%m-%d} → "
          f"{ev.accepted_et.max():%Y-%m-%d}", flush=True)

    P = {"lexicon": np.array([[d["sentiment"].p(k) for k in LABELS] for d in RuleReader(NewsJudge()).read(texts)]),
         "tf-idf (tweets)": np.array([[d["sentiment"].p(k) for k in LABELS]
                                     for d in TfidfNewsReader(load_tweets("train", DATA_CACHE)).read(texts)])}
    P["laya/typed-decisions+cal"] = calibrated_laya_probs(texts)

    # cascade: re-read the least certain releases with Claude (same question, same schema)
    thr = json.loads((REPORTS / "e3_news.json").read_text())["cascade_threshold_from_train_q20"]
    hard = np.where(certainty(P["laya/typed-decisions+cal"]) < thr)[0]
    cl = claude().system_one_many([texts[i] for i in hard], NewsJudge.questions)
    Pc = P["laya/typed-decisions+cal"].copy()
    for i, d in zip(hard, cl):
        Pc[i] = [d["sentiment"].p(k) for k in LABELS]
    P["cascade laya->claude"] = Pc
    claude_cost = float(sum(d.cost_usd for d in cl))
    print(f"cascade: {len(hard):,} of {len(texts):,} escalated ({len(hard) / len(texts):.0%}), "
          f"notional Claude cost ${claude_cost:.2f}", flush=True)

    sigs = {k: to_signal(v) for k, v in P.items()}
    sigs["every release bullish"] = np.ones(len(ev))

    out = {"period": [START, str(m.dates[-1].date())], "universe_size": len(syms), "n_releases": len(ev),
           "n_companies_with_releases": int(ev["symbol"].nunique()),
           "before_open_share": float((ev.accepted_et.dt.hour * 60 + ev.accepted_et.dt.minute < 570).mean()),
           "items_top": ev["items"].str.split(",").explode().value_counts().head(8).to_dict(),
           "cascade": {"threshold": thr, "escalated": int(len(hard)), "rate": float(len(hard) / len(ev)),
                       "notional_cost_usd": claude_cost},
           "label_mix": {}, "event_study": {}, "backtest": {}, "equity": {}, "hold_sensitivity": {},
           "conservative_timing": {}, "placebo": {}}
    for name, s in sigs.items():
        out["label_mix"][name] = {"bullish": float((s == 1).mean()), "bearish": float((s == -1).mean())}
        out["event_study"][name] = event_study(m, ev, s)
        perf, eq = backtest(m, ev, s, name=name)
        out["backtest"][name] = perf
        wk = eq.resample("W-FRI").last()
        out["equity"][name] = {"dates": [str(d.date()) for d in wk.index], "values": wk.round(5).tolist()}
        out["hold_sensitivity"][name] = {str(h): backtest(m, ev, s, hold=h)[0]["sharpe"] for h in (1, 3, 10)}
        out["hold_sensitivity"][name][str(HOLD)] = perf["sharpe"]
        es = out["event_study"][name]
        print(f"{name:26s} bull {out['label_mix'][name]['bullish']:.0%} bear {out['label_mix'][name]['bearish']:.0%} | "
              f"gap LS {es['gap']['ls_mean_bp']:+.0f}bp (t {es['gap']['t']:.1f}) | 5d LS {es['r5']['ls_mean_bp']:+.1f}bp "
              f"(t {es['r5']['t']:.2f}) | Sharpe {perf['sharpe']:.2f} {perf['sharpe_ci95']}", flush=True)

    # robustness: always trade the next day's open
    for name in ["laya/typed-decisions+cal", "cascade laya->claude"]:
        by_acc = dict(zip(ev["accession"], sigs[name]))
        sc = ev_cons["accession"].map(by_acc).fillna(0.0).values
        out["conservative_timing"][name] = backtest(m, ev_cons, sc, name=name)[0]["sharpe"]

    # placebo: Laya's labels shuffled across releases
    rng = np.random.default_rng(0)
    base = sigs["laya/typed-decisions+cal"]
    sh, ls = [], []
    for _ in range(20):
        perm = rng.permutation(base)
        sh.append(backtest(m, ev, perm)[0]["sharpe"])
        ls.append(event_study(m, ev, perm)["r5"]["ls_mean_bp"])
    out["placebo"] = {"sharpe_mean": float(np.mean(sh)), "sharpe_sd": float(np.std(sh)), "sharpe_all": sh,
                      "ls5d_bp_mean": float(np.mean(ls)), "ls5d_bp_sd": float(np.std(ls))}
    print(f"placebo Sharpe {np.mean(sh):.2f} ± {np.std(sh):.2f}", flush=True)
    save("e6_real_news", out)


if __name__ == "__main__":
    main()
