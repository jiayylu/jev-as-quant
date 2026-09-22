"""Render reports/figures/*.png (light + dark) and reports/RESULTS.md from the experiment JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORTS = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e8e7e3", "ref": "#8a8983",
              "laya": "#2a78d6", "rules": "#eb6834", "tfidf": "#1baf7a", "laya_en": "#4a3aa7", "claude": "#008300",
              "laya_ml": "#e87ba4", "band": "#f0efec", "ramp": ["#86b6ef", "#3987e5", "#184f95"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#2f2f2d", "ref": "#8f8e87",
             "laya": "#3987e5", "rules": "#d95926", "tfidf": "#199e70", "laya_en": "#9085e9", "claude": "#008300",
             "laya_ml": "#d55181", "band": "#383835", "ramp": ["#184f95", "#3987e5", "#86b6ef"]},
}


def load(name):
    p = REPORTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def entity_color(name: str, t: dict) -> str:
    n = name.lower()
    if "oracle" in n or "buy&hold" in n:
        return t["ref"]
    if "cascade" in n or "claude" in n:
        return t["claude"]
    if "tf-idf" in n or "tfidf" in n:
        return t["tfidf"]
    if "laya/english" in n or "laya-en" in n:
        return t["laya_en"]
    if "multilingual" in n:
        return t["laya_ml"]
    if "laya" in n:
        return t["laya"]
    return t["rules"]


def style(ax, t, xgrid=False):
    ax.set_facecolor(t["surface"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], labelsize=9, length=0)
    ax.yaxis.label.set_color(t["text2"])
    ax.xaxis.label.set_color(t["text2"])
    ax.grid(axis="x" if xgrid else "y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t, w=9.0, h=4.2, ncols=1, **kw):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=t["surface"], **kw)
    return fig, np.atleast_1d(axes)


def titles(fig, t, title, subtitle):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=t["text"])
    fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=9.5, color=t["text2"])


def save(fig, name, mode):
    fig.savefig(FIG / (f"{name}.png" if mode == "light" else f"{name}-dark.png"), dpi=180,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def pretty(name: str) -> str:
    n = name.replace("router[", "").replace("signal[", "").rstrip("]")
    return (n.replace("laya/typed-decisions", "Laya typed-decisions").replace("laya/english", "Laya english")
             .replace("cascade->claude", "cascade → Claude").replace("tf-idf", "TF-IDF")
             .replace("buy&hold (equal weight)", "buy & hold").replace("buy&hold", "buy & hold"))


def label_ends(ax, t, xs, ys, names, log=False, min_gap=0.06):
    """Direct labels at line ends, nudged apart so they never collide (gap in axis fraction)."""
    lo, hi = ax.get_ylim()
    tf = (lambda v: np.log(v)) if log else (lambda v: v)
    inv = (lambda v: np.exp(v)) if log else (lambda v: v)
    span = tf(hi) - tf(lo)
    order = np.argsort([tf(y) for y in ys])
    placed = []
    for i in order:
        y = tf(ys[i])
        if placed and y - placed[-1] < min_gap * span:
            y = placed[-1] + min_gap * span
        placed.append(y)
        ax.text(xs[i], inv(y), "  " + names[i], fontsize=8, color=t["text2"], va="center")


def plain_log_axis(ax):
    lo, hi = ax.get_ylim()
    nice = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
    ticks = [v for v in nice if lo <= v <= hi]
    if len(ticks) > 7:
        ticks = ticks[::2] if 1 in ticks[::2] else ticks[1::2]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def legend(ax, t, **kw):
    kw.setdefault("fontsize", 8.5)
    return ax.legend(frameon=False, labelcolor=t["text2"], **kw)


# --------------------------------------------------------------------------------------- E1
def fig_latency(mode):
    d = load("e1_latency")
    if not d:
        return
    t = THEMES[mode]
    rows = [r for r in d["results"] if r["checkpoint"] == "typed-decisions"]
    cats = [("single", "headline (~20 tokens)", 1, "headline, 1 question"),
            ("single", "market (~110 tokens)", 1, "market state, 1 question"),
            ("single", "market (~110 tokens)", 3, "market state, 3 questions"),
            ("single", "market (~110 tokens)", 6, "market state, 6 questions"),
            ("batched(64 states)", "market (~110 tokens)", 3, "batched ×64, 3 questions (per state)")]
    fig, (ax,) = figure(t, 9, 4.4)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.80, bottom=0.14)
    ax.axvspan(70, 500, color=t["band"], zorder=0)
    ax.text(187, len(cats) - 0.55, "Jev published: 70–500 ms (network API)", ha="center", va="bottom",
            fontsize=8, color=t["text2"])
    ax.axvline(32.8, color=t["ref"], linewidth=1, linestyle=(0, (2, 2)))
    ax.text(32.8, -0.75, " Laya on T4 (published): 32.8 ms", fontsize=8, color=t["text2"], va="center")
    for yi, (path, st, nq, label) in enumerate(cats):
        for dev, filled in [("mps", True), ("cpu", False)]:
            r = next((x for x in rows if x["path"] == path and x["state"] == st and x["questions"] == nq
                      and x["device"] == dev), None)
            if not r:
                continue
            y = len(cats) - 1 - yi
            ax.plot([r["p50_ms"], r["p95_ms"]], [y, y], color=t["laya"], linewidth=2, solid_capstyle="round")
            ax.plot(r["p50_ms"], y, "o", markersize=8, color=t["laya"],
                    markerfacecolor=t["laya"] if filled else t["surface"], markeredgewidth=2)
            ax.text(r["p50_ms"], y + 0.22, f"{r['p50_ms']:.0f}", ha="center", fontsize=8, color=t["text2"])
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c[3] for c in cats][::-1], color=t["text"])
    ax.set_xscale("log")
    ax.set_xlim(15, 2000)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([20, 50, 100, 200, 500, 1000, 2000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-1.1, len(cats) - 0.1)
    ax.set_xlabel("milliseconds (log scale) · dot = median, line to p95")
    style(ax, t, xgrid=True)
    ax.plot([], [], "o", color=t["laya"], label="Apple M3 Pro GPU (MPS)")
    ax.plot([], [], "o", color=t["laya"], markerfacecolor=t["surface"], markeredgewidth=2, label="same machine, CPU")
    legend(ax, t, loc="lower right")
    titles(fig, t, "E1 · Laya latency on a laptop",
           "Tens of milliseconds per question, but cost grows linearly with the number of questions per state")
    save(fig, "e1_latency", mode)


# --------------------------------------------------------------------------------------- E2
def fig_numeracy(mode):
    d = load("e2_numeracy")
    if not d:
        return
    t = THEMES[mode]
    rows = [r for r in d["results"] if r["checkpoint"] == "typed-decisions"]
    qs = ["rsi_above_70", "above_sma50", "up_5d", "vol_above_norm", "near_60d_high"]
    qlabel = {"rsi_above_70": "RSI(14) > 70", "above_sma50": "price > 50-day avg", "up_5d": "5-day return > 0",
              "vol_above_norm": "vol > 1-year norm", "near_60d_high": "within 2% of 60-day high"}
    reps = [("raw_json", "raw JSON numbers"), ("numeric_text", "numbers in sentences"),
            ("verbal", "numbers + qualitative words")]
    fig, (ax,) = figure(t, 9, 4.8)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.76, bottom=0.12)
    h = 0.25
    for k, (rep, lab) in enumerate(reps):
        ys = [len(qs) - 1 - i + (1 - k) * h for i in range(len(qs))]
        vals = [next(r["balanced_accuracy"] for r in rows if r["question"] == q and r["representation"] == rep)
                for q in qs]
        ax.barh(ys, vals, height=h - 0.03, color=t["ramp"][k], label=lab)
        for y, v in zip(ys, vals):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7.5, color=t["text2"])
    ax.axvline(0.5, color=t["ref"], linewidth=1, linestyle=(0, (2, 2)))
    ax.text(0.505, -0.62, "coin flip", ha="left", fontsize=8, color=t["text2"])
    ax.set_yticks(range(len(qs)))
    ax.set_yticklabels([qlabel[q] for q in qs][::-1], color=t["text"])
    ax.set_xlim(0.3, 1.08)
    ax.set_xlabel("balanced accuracy of the yes/no answer (P(true) > 0.5); a one-line `if` scores 1.00")
    style(ax, t, xgrid=True)
    legend(ax, t, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3)
    titles(fig, t, "E2 · Given raw numbers, Laya's yes/no answers are coin flips",
           "Five threshold questions, the same 400 market states written three ways (checkpoint: typed-decisions)")
    save(fig, "e2_numeracy", mode)


# --------------------------------------------------------------------------------------- E3
def fig_news(mode):
    d = load("e3_news")
    if not d:
        return
    t = THEMES[mode]
    fig, axes = figure(t, 12.5, 4.6, ncols=3, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fig.subplots_adjust(left=0.235, right=0.985, top=0.78, bottom=0.14, wspace=0.40)
    ax = axes[0]
    items = sorted(d["table"].items(), key=lambda kv: kv[1]["macro_f1"])
    for i, (name, r) in enumerate(items):
        ax.barh(i, r["macro_f1"], height=0.62, color=entity_color(name, t))
        ax.text(r["macro_f1"] + 0.008, i, f"{r['macro_f1']:.2f}", va="center", fontsize=8, color=t["text2"])
    def short(n):
        if n.startswith("cascade"):
            return f"cascade: Laya → Claude ({d['cascade_operating_point']['escalation_rate'] * 100:.0f}% escalated)"
        return (n.replace("laya/", "Laya ").replace(" + calibration", " + calibration").replace("tf-idf + logreg", "TF-IDF + logreg"))
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([short(n) for n, _ in items], fontsize=8.2, color=t["text"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1 on 2,388 held-out headlines")
    style(ax, t, xgrid=True)

    ax = axes[1]
    cur = d["cascade_curve"]
    x = [c["escalation_rate"] * 100 for c in cur]
    y = [c["macro_f1"] for c in cur]
    ax.plot(x, y, "-o", color=t["claude"], linewidth=2, markersize=6, label="Laya, hardest x% → Claude")
    tf = d["table"]["tf-idf + logreg (supervised)"]["macro_f1"]
    ax.axhline(tf, color=t["tfidf"], linewidth=1.5, linestyle=(0, (4, 2)), label="TF-IDF (9.5k labels)")
    for xi, yi, c in zip(x, y, cur):
        if xi in (10, 20):
            ax.text(xi + 0.6, yi - 0.004, f"${c['claude_cost_usd_per_1k']:.2f} per 1k", ha="left", va="top",
                    fontsize=7.5, color=t["text2"])
    ax.set_xlabel("% of headlines escalated to Claude")
    ax.set_ylabel("macro-F1")
    lo = min(min(y), tf) - 0.05
    ax.set_ylim(lo, max(max(y), tf) + 0.06)
    style(ax, t)
    legend(ax, t, loc="best")
    ax.set_title("System 1 → System 2 cascade", fontsize=10, color=t["text"], loc="left")

    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=t["ref"], linewidth=1, linestyle=(0, (2, 2)))
    for name, rel in d["reliability"].items():
        conf = [b["confidence"] for b in rel]
        acc = [b["accuracy"] for b in rel]
        ls = "--" if "zero-shot" in name else "-"
        short = name.replace(" (supervised)", "").replace("tf-idf + logreg", "TF-IDF").replace("laya/", "Laya ")
        ax.plot(conf, acc, ls, marker="o", markersize=5, linewidth=2, color=entity_color(name, t), label=short)
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed accuracy")
    style(ax, t)
    legend(ax, t, loc="lower right", fontsize=7.5)
    ax.set_title("Reliability", fontsize=10, color=t["text"], loc="left")
    titles(fig, t, "E3 · Reading financial headlines",
           "Zero-shot Laya vs a word list, a supervised TF-IDF model, and Claude; calibration fitted on training data only")
    save(fig, "e3_news", mode)


# --------------------------------------------------------------------------------------- E4
def fig_synthetic(mode):
    d = load("e4_synthetic")
    if not d:
        return
    t = THEMES[mode]
    seeds = [str(s) for s in d["test_seeds"]]
    fig, axes = figure(t, 12, 4.4, ncols=3, gridspec_kw={"width_ratios": [1, 1, 1.35]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.2, wspace=0.35)

    def dotbar(ax, names, getter, xlabel_fmt):
        for i, n in enumerate(names):
            vals = [getter(s, n) for s in seeds]
            c = entity_color(n, t)
            ax.bar(i, np.mean(vals), width=0.6, color=c, alpha=0.9)
            ax.scatter([i] * len(vals), vals, s=18, color=t["text2"], zorder=3)
            top = max(max(vals), np.mean(vals), 0)
            ax.text(i, top + 0.025, f"{np.mean(vals):.2f}", ha="center", va="bottom", fontsize=8.5, color=t["text"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([pretty(xlabel_fmt(n)) for n in names], rotation=20, ha="right", fontsize=8, color=t["text"])
        style(ax, t)

    ax = axes[0]
    names = list(d["regime"][seeds[0]].keys())
    dotbar(ax, names, lambda s, n: d["regime"][s][n]["balanced_accuracy"], lambda n: n)
    ax.axhline(0.25, color=t["ref"], linewidth=1, linestyle=(0, (2, 2)))
    ax.text(len(names) - 0.5, 0.26, "chance", fontsize=7.5, color=t["text2"], ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Regime recognition (balanced acc.)", fontsize=10, color=t["text"], loc="left")

    ax = axes[1]
    names = [n for n in d["news"][seeds[0]] if isinstance(d["news"][seeds[0]][n], dict)
             and "planted_edge_capture" in d["news"][seeds[0]][n] and n != "oracle"]
    dotbar(ax, names, lambda s, n: d["news"][s][n]["planted_edge_capture"], lambda n: n)
    ax.axhline(1.0, color=t["ref"], linewidth=1, linestyle=(0, (2, 2)))
    ax.text(len(names) - 0.5, 1.02, "oracle", fontsize=7.5, color=t["text2"], ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Planted news edge captured", fontsize=10, color=t["text"], loc="left")

    ax = axes[2]
    ends_x, ends_y, names = [], [], []
    for n, vals in d["equity_seed1"].items():
        ls = (0, (4, 2)) if ("oracle" in n or "buy&hold" in n) else "-"
        ax.plot(vals, color=entity_color(n, t), linewidth=1.6 if ls == "-" else 1.3, linestyle=ls)
        ends_x.append(len(vals))
        ends_y.append(vals[-1])
        names.append(pretty(n))
    ax.set_yscale("log")
    plain_log_axis(ax)
    ax.set_xlim(0, len(vals) * 1.45)
    label_ends(ax, t, ends_x, ends_y, names, log=True)
    ax.set_xlabel("trading days (test seed 1)")
    ax.set_title("Regime router equity, seed 1", fontsize=10, color=t["text"], loc="left")
    style(ax, t)
    titles(fig, t, "E4 · Synthetic market with known truth (3 test seeds)",
           "Regimes read from verbalized indicators · headlines are real labeled tweets whose price impact follows "
           "the label · bars = mean, dots = seeds")
    save(fig, "e4_synthetic", mode)


# --------------------------------------------------------------------------------------- E5
def fig_real(mode):
    d = load("e5_real")
    if not d:
        return
    t = THEMES[mode]
    show = ["buy&hold (equal weight)", "signal[rules]", "signal[laya/typed-decisions]", "signal[laya/english]"]
    fig = plt.figure(figsize=(10, 5.6), facecolor=t["surface"])
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.08, left=0.07, right=0.83, top=0.84, bottom=0.08)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    import pandas as pd
    ends_x, ends_y, names = [], [], []
    bh = d["equity"]["buy&hold (equal weight)"]["values"]
    same_as_bh = [n for n in show if n != "buy&hold (equal weight)" and d["equity"][n]["values"] == bh]
    show = [n for n in show if n not in same_as_bh]
    for n in show:
        e = d["equity"][n]
        x = pd.to_datetime(e["dates"])
        ls = (0, (4, 2)) if "buy&hold" in n else "-"
        ax.plot(x, e["values"], color=entity_color(n, t), linewidth=1.8, linestyle=ls)
        ax2.plot(x, d["drawdown"][n], color=entity_color(n, t), linewidth=1.2, linestyle=ls)
        ends_x.append(x[-1])
        ends_y.append(e["values"][-1])
        label = pretty(n)
        if "buy&hold" in n and same_as_bh:
            label += "\n  = " + ", ".join(pretty(m) for m in same_as_bh) + "\n  (always says buy)"
        names.append(label)
    ax.set_yscale("log")
    plain_log_axis(ax)
    label_ends(ax, t, ends_x, ends_y, names, log=True)
    ax.set_ylabel("growth of $1 (log)")
    ax.tick_params(labelbottom=False)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylabel("drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style(ax, t)
    style(ax2, t)
    titles(fig, t, "E5 · Eight ETFs, weekly, long-only, after costs",
           f"{d['period'][0]} → {d['period'][1]} · SPY QQQ IWM EFA EEM TLT GLD VNQ · Laya reads verbalized indicators only")
    save(fig, "e5_real", mode)


# --------------------------------------------------------------------------------------- tables
def pct(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def results_md() -> str:
    out = ["# Results (auto-generated by experiments/make_report.py)\n"]
    e1 = load("e1_latency")
    if e1:
        out += ["## E1 latency (ms)\n", "| checkpoint | device | path | questions | state | p50 | p95 |",
                "|---|---|---|---|---|---|---|"]
        for r in e1["results"]:
            out.append(f"| {r['checkpoint']} | {r['device']} | {r['path']} | {r['questions']} | {r['state']} | "
                       f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} |")
        out.append("")
    e2 = load("e2_numeracy")
    if e2:
        out += ["## E2 reading numbers (mean over 5 threshold questions)\n",
                "| checkpoint | representation | balanced accuracy | AUC | Brier |", "|---|---|---|---|---|"]
        for r in e2["summary"]:
            out.append(f"| {r['checkpoint']} | {r['representation']} | {num(r['balanced_accuracy'])} | "
                       f"{num(r['auc'])} | {num(r['brier'], 3)} |")
        out.append("")
    e3 = load("e3_news")
    if e3:
        out += [f"## E3 headlines (n={e3['n_val']} held-out; fast engine: {e3['fast_engine']})\n",
                "| reader | accuracy | macro-F1 | Brier | ECE | latency ms/item | Claude $ per 1k |",
                "|---|---|---|---|---|---|---|"]
        for name, r in sorted(e3["table"].items(), key=lambda kv: -kv[1]["macro_f1"]):
            lat = r.get("latency_ms_per_item")
            out.append(f"| {name} | {pct(r['accuracy'])} | {num(r['macro_f1'])} | {num(r['brier'], 3)} | "
                       f"{num(r['ece'], 3)} | {'–' if lat is None else f'{lat:.2f}'} | {num(r['cost_usd_per_1k'])} |")
        out += ["", "Random 300 held-out headlines (Claude scored alone):", "",
                "| reader | accuracy | macro-F1 |", "|---|---|---|"]
        for name, r in e3["random_300"].items():
            out.append(f"| {name} | {pct(r['accuracy'])} | {num(r['macro_f1'])} |")
        out += ["", "Cascade curve:", "", "| escalated | accuracy | macro-F1 | Claude $ per 1k headlines |",
                "|---|---|---|---|"]
        for c in e3["cascade_curve"]:
            out.append(f"| {pct(c['escalation_rate'], 0)} | {pct(c['accuracy'])} | {num(c['macro_f1'])} | "
                       f"{num(c['claude_cost_usd_per_1k'])} |")
        out.append("")
    e4 = load("e4_synthetic")
    if e4:
        seeds = [str(s) for s in e4["test_seeds"]]

        def mean_sd(get):
            v = [get(s) for s in seeds]
            return f"{np.mean(v):.2f} ± {np.std(v):.2f}"
        out += [f"## E4 synthetic market (test seeds {', '.join(seeds)}; mean ± sd)\n",
                "Regime recognition:", "", "| reader | accuracy | balanced accuracy | crisis recall |", "|---|---|---|---|"]
        for n in e4["regime"][seeds[0]]:
            out.append(f"| {n} | {mean_sd(lambda s: e4['regime'][s][n]['accuracy'])} | "
                       f"{mean_sd(lambda s: e4['regime'][s][n]['balanced_accuracy'])} | "
                       f"{mean_sd(lambda s: e4['regime'][s][n]['recall']['crisis'] or 0)} |")
        out += ["", "Regime router (long/short):", "", "| strategy | Sharpe | CAGR | max drawdown |", "|---|---|---|---|"]
        for n in e4["router"][seeds[0]]:
            out.append(f"| {n} | {mean_sd(lambda s: e4['router'][s][n]['sharpe'])} | "
                       f"{mean_sd(lambda s: e4['router'][s][n]['cagr'])} | "
                       f"{mean_sd(lambda s: e4['router'][s][n]['max_drawdown'])} |")
        out += ["", "News trading (daily, long/short, after costs):", "",
                "| reader | headline accuracy | planted edge captured | Sharpe | P&L vs oracle (noisy) |",
                "|---|---|---|---|---|"]
        for n, v in e4["news"][seeds[0]].items():
            if isinstance(v, dict) and "sharpe" in v:
                out.append(f"| {n} | {mean_sd(lambda s: e4['news'][s][n]['headline_accuracy'])} | "
                           f"{mean_sd(lambda s: e4['news'][s][n].get('planted_edge_capture', float('nan')))} | "
                           f"{mean_sd(lambda s: e4['news'][s][n]['sharpe'])} | "
                           f"{mean_sd(lambda s: e4['news'][s][n]['capture_vs_oracle'])} |")
        out += ["", f"Risk veto on the rules router (Laya threshold τ={e4['risk']['tau']} chosen on dev seed 0):", "",
                "| variant | Sharpe | max drawdown | crisis precision | crisis recall |", "|---|---|---|---|---|"]
        for n in e4["risk"][seeds[0]]:
            g = lambda s, k: e4["risk"][s][n].get(k)  # noqa: E731
            prec = "–" if g(seeds[0], "veto_precision_crisis") is None else mean_sd(lambda s: g(s, "veto_precision_crisis") or 0)
            rec = "–" if g(seeds[0], "veto_recall_crisis") is None else mean_sd(lambda s: g(s, "veto_recall_crisis") or 0)
            out.append(f"| {n} | {mean_sd(lambda s: g(s, 'sharpe'))} | {mean_sd(lambda s: g(s, 'max_drawdown'))} | "
                       f"{prec} | {rec} |")
        out.append("")
    e5 = load("e5_real")
    if e5:
        out += [f"## E5 real ETFs ({e5['period'][0]} → {e5['period'][1]}, {e5['n_decisions']} weekly decisions)\n",
                "| strategy | CAGR | vol | Sharpe [95% CI] | max drawdown | turnover / yr |", "|---|---|---|---|---|---|"]
        for n, r in e5["strategies"].items():
            ci = r.get("sharpe_ci95") or [None, None]
            out.append(f"| {n} | {pct(r['cagr'])} | {pct(r['vol'])} | {num(r['sharpe'])} [{num(ci[0])}, {num(ci[1])}] | "
                       f"{pct(r['max_drawdown'])} | {num(r['turnover_per_year'], 1)}x |")
        out += ["", "Predictive power (next-week return):", "", "| signal | mean cross-sectional IC | t-stat | pooled rank corr |",
                "|---|---|---|---|"]
        for n, r in e5["ic"].items():
            out.append(f"| {n} | {num(r['ic_mean'], 3)} | {num(r['ic_t'])} | {num(r['pooled_rank_corr'], 3)} |")
        out += ["", "Agreement with the rules on buy/hold/sell:", "", "| reader | agree | Cohen's κ | action mix |",
                "|---|---|---|---|"]
        for n, r in e5["agreement"].items():
            if n != "rules_action_mix":
                out.append(f"| {n} | {pct(r['agree'])} | {num(r['cohen_kappa'])} | {r['action_mix']} |")
        out.append(f"| rules | – | – | {e5['agreement']['rules_action_mix']} |")
        out.append("")
    return "\n".join(out)


def main():
    for mode in THEMES:
        for f in (fig_latency, fig_numeracy, fig_news, fig_synthetic, fig_real):
            f(mode)
    (REPORTS / "RESULTS.md").write_text(results_md())
    print("figures ->", FIG, "| tables ->", REPORTS / "RESULTS.md")


if __name__ == "__main__":
    main()
