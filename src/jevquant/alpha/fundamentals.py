"""Point-in-time fundamental panels from first-reported SEC XBRL facts.

A value becomes visible on the first trading day after the filing that first reported it,
and at any date the most recent reporting period visible wins. Market value comes from the
issuer-reported public float (USD), which, unlike price x shares, needs no split adjustment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANNUAL = (300, 400)  # duration in days of a fiscal-year fact
QUARTER = (70, 100)


def _latest_by_filed(df: pd.DataFrame) -> pd.DataFrame:
    """Walk facts in filing order; emit a row whenever a newer period end becomes visible."""
    out, best = [], None
    for r in df.sort_values(["filed", "end"]).itertuples():
        if best is None or r.end > best:
            best = r.end
            out.append((r.filed, r.end, r.val))
    return pd.DataFrame(out, columns=["filed", "end", "val"])


def _with_prior_year(events: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """Attach the value of the same period one year earlier, as known at each event's filing."""
    prior = []
    for r in events.itertuples():
        lo, hi = r.end - pd.Timedelta(days=400), r.end - pd.Timedelta(days=330)
        c = facts[(facts.end >= lo) & (facts.end <= hi) & (facts.filed <= r.filed)]
        prior.append(c.sort_values("end").val.iloc[-1] if len(c) else np.nan)
    return events.assign(prior=prior)


def build_fundamentals(path: str | Path, days: pd.DatetimeIndex, cik_to_tickers: dict[str, list[str]],
                       tickers: list[str], close: pd.DataFrame | None = None,
                       volume: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    f = pd.read_csv(path, dtype={"cik": str})
    f = f[f["unit"].isin(["USD", "shares"])]
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c])
    f["days"] = (f["end"] - f["start"]).dt.days
    # revenue: prefer the legacy tag, fall back to ASC 606 revenue
    f.loc[f.concept == "RevenueFromContractWithCustomerExcludingAssessedTax", "concept"] = "RevenueASC606"

    series: dict[str, dict[str, pd.Series]] = {}

    def put(name, cik, idx, vals):
        series.setdefault(name, {})[cik] = pd.Series(vals, index=pd.DatetimeIndex(idx))

    for cik, g in f.groupby("cik"):
        def dur(concept, lo_hi):
            x = g[(g.concept == concept) & g.days.between(*lo_hi)]
            return x[["filed", "end", "val"]]

        def inst(concept):
            return g[(g.concept == concept) & g.start.isna()][["filed", "end", "val"]]

        for name, facts, yoy in [
            ("ni_fy", dur("NetIncomeLoss", ANNUAL), False), ("oi_fy", dur("OperatingIncomeLoss", ANNUAL), False),
            ("gp_fy", dur("GrossProfit", ANNUAL), False), ("cfo_fy", dur("NetCashProvidedByUsedInOperatingActivities", ANNUAL), False),
            ("rev_fy", pd.concat([dur("Revenues", ANNUAL), dur("RevenueASC606", ANNUAL)]), True),
            ("assets", inst("Assets"), True), ("equity", inst("StockholdersEquity"), False),
            ("float", inst("EntityPublicFloat"), False), ("ni_q", dur("NetIncomeLoss", QUARTER), True),
        ]:
            if facts.empty:
                continue
            ev = _latest_by_filed(facts)
            if yoy:
                ev = _with_prior_year(ev, facts)
                put(name + "_prior", cik, ev.filed, ev.prior.values)
            put(name, cik, ev.filed, ev.val.values)
            if name == "float":  # the date the float was measured, to roll it forward with prices
                put("float_end", cik, ev.filed, ev.end.map(pd.Timestamp.toordinal).values.astype(float))

    def to_panel(name: str, max_age: int) -> pd.DataFrame:
        cols = {}
        for cik, s in series.get(name, {}).items():
            s = s[~s.index.duplicated(keep="last")].sort_index()
            eff = s.copy()
            eff.index = eff.index + pd.Timedelta(days=1)  # visible from the next day
            daily = eff.reindex(days.union(eff.index)).ffill(limit=None).reindex(days)
            last = pd.Series(eff.index, index=eff.index).reindex(days.union(eff.index)).ffill().reindex(days)
            daily[(days.to_series() - last).dt.days > max_age] = np.nan  # too stale
            for t in cik_to_tickers.get(cik, []):
                cols[t] = daily
        return pd.DataFrame(cols, index=days).reindex(columns=tickers)

    P = {k: to_panel(k, 500) for k in ["ni_fy", "oi_fy", "gp_fy", "cfo_fy", "rev_fy", "rev_fy_prior", "assets",
                                        "assets_prior", "equity", "float"]}
    P["ni_q"], P["ni_q_prior"] = to_panel("ni_q", 200), to_panel("ni_q_prior", 200)
    P["float_end"] = to_panel("float_end", 500)
    A = P["assets"].where(P["assets"] > 0)
    FL = P["float"].where(P["float"] > 0)
    out_mcap = {}
    if close is not None:
        # market value today = public float at its measurement date x price change since then
        end_ord = P["float_end"]
        idx = np.searchsorted(days.map(pd.Timestamp.toordinal).values, np.nan_to_num(end_ord.values, nan=0).astype(int))
        idx = np.clip(idx, 0, len(days) - 1)
        c = close.reindex(columns=tickers).values
        base = np.take_along_axis(c, idx, axis=0)
        mcap = pd.DataFrame(np.where(end_ord.notna().values, FL.values * c / base, np.nan), index=days, columns=tickers)
        if volume is not None:
            # a few issuers file the float with a wrong scale (Jefferies at $5.9T); keep market values
            # whose implied annual turnover is between 5% and 1,000% of the company's value
            adv = (close * volume).reindex(columns=tickers).rolling(63, min_periods=20).mean() * 252
            ratio = mcap / adv
            mcap = mcap.where((ratio > 0.1) & (ratio < 20))
        out_mcap["mcap"] = mcap
    return {
        **out_mcap,
        "ep": P["ni_fy"] / FL,
        "bm": P["equity"] / FL,
        "op_prof": P["oi_fy"] / A,
        "gp_prof": P["gp_fy"] / A,
        "accruals": (P["ni_fy"] - P["cfo_fy"]) / A,
        "asset_growth": P["assets"] / P["assets_prior"].where(P["assets_prior"] > 0) - 1,
        "rev_growth": P["rev_fy"] / P["rev_fy_prior"].where(P["rev_fy_prior"] > 0) - 1,
        "sue": (P["ni_q"] - P["ni_q_prior"]) / A,
        "size": np.log(FL),
    }
