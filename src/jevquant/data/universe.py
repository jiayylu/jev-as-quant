"""Point-in-time S&P 500 membership (fja05680/sp500, MIT): who was in the index on each date.

Using today's constituents for a historical backtest silently drops every company that was
removed (often after doing badly) and inflates results. This module answers "was ticker X in
the index on date d?" from the dated history of the full member list.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
       "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


class Membership:
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        self.dates = df["date"].values
        self.lists = [frozenset(t.split(",")) for t in df["tickers"]]

    def members(self, date) -> frozenset:
        """Members on `date` (the latest published list on or before it)."""
        import numpy as np

        i = int(np.searchsorted(self.dates, pd.Timestamp(date).to_datetime64(), side="right")) - 1
        if i < 0:
            raise ValueError(f"no membership data before {date}")
        return self.lists[i]

    def ever(self, start, end) -> set[str]:
        """Every ticker that was a member at some point in [start, end]."""
        out = set(self.members(start))
        s, e = pd.Timestamp(start).to_datetime64(), pd.Timestamp(end).to_datetime64()
        for d, lst in zip(self.dates, self.lists):
            if s <= d <= e:
                out |= lst
        return out

    def matrix(self, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
        """Boolean date x ticker table of membership."""
        return pd.DataFrame({t: [t in self.members(d) for d in dates] for t in tickers}, index=dates)


def first_seen(m: "Membership") -> dict[str, pd.Timestamp]:
    out: dict[str, pd.Timestamp] = {}
    for d, lst in zip(m.dates, m.lists):
        for t in lst:
            out.setdefault(t, pd.Timestamp(d))
    return out


def detect_renames(m: "Membership", current: pd.DataFrame, tolerance_days: int = 120) -> dict[str, str]:
    """Old ticker -> current ticker for companies that changed symbol while in the index.

    Wikipedia's constituents table dates a member by when the *company* joined (META: 2013),
    while the dated lists only show the *symbol* META from 2022. When the two disagree, the
    old symbol is the one that left the list on the day the new one appeared.
    """
    import numpy as np

    seen = first_seen(m)
    renames = {}
    for r in current.itertuples():
        y = r.symbol
        if not r.added or y not in seen:
            continue
        added, fs = pd.Timestamp(r.added), seen[y]
        if added >= fs - pd.Timedelta(days=30):
            continue  # joined under its current symbol
        i = int(np.searchsorted(m.dates, fs.to_datetime64(), side="left"))
        if i == 0:
            continue
        gone = m.lists[i - 1] - m.lists[i]
        start = pd.Timestamp(m.dates[0])
        cands = [x for x in gone if x != y and x not in renames and (
            abs((seen[x] - added).days) <= tolerance_days or (seen[x] == start and added <= start))]
        if len(cands) == 1:
            renames[cands[0]] = y
    return renames
