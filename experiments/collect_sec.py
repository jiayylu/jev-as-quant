"""Collect 8-K press releases (EX-99.x) from SEC EDGAR.

Default (E6): current S&P 500 members that joined the index before 2024-01-01, releases since
2024-01-01 -> data_cache/sec/press_releases.csv.gz. With --universe/--since/--until/--out it
backfills other periods (e.g. every point-in-time member since 2016 for the alpha factory).
Needs SEC_USER_AGENT="<name> <email>". Every response is cached; the run can be resumed.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import DATA_CACHE
from jevquant.data.sec import SecClient, exhibit99, list_8k, sp500_from_wikitext

SINCE = "2024-01-01"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", help="csv with ticker/symbol and cik columns (default: E6 universe)")
    ap.add_argument("--since", default=SINCE)
    ap.add_argument("--until", default="2099-12-31")
    ap.add_argument("--out", default=str(DATA_CACHE / "sec" / "press_releases.csv.gz"))
    ap.add_argument("--require-item", default=None, help="only open 8-Ks listing this item, e.g. 9.01 (exhibits)")
    a = ap.parse_args()
    client = SecClient(cache_dir=DATA_CACHE / "sec" / "raw")
    if a.universe:
        uni = pd.read_csv(a.universe, dtype=str).rename(columns={"ticker": "symbol"})
        print(f"universe: {len(uni)} companies from {a.universe}", flush=True)
    else:
        wiki = sorted((DATA_CACHE / "sec").glob("sp500_*.wiki"))[-1]
        sp = sp500_from_wikitext(wiki.read_text())
        uni = sp[sp["added"] < SINCE].reset_index(drop=True)
        uni.to_csv(DATA_CACHE / "sec" / "universe.csv", index=False)
        print(f"universe: {len(uni)} companies (S&P 500 members since before {SINCE}) from {wiki.name}", flush=True)

    filings = []
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(list_8k, client, r.cik, a.since): r for r in uni.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            df = f.result()
            if len(df):
                df["symbol"] = r.symbol
                filings.append(df)
            if i % 50 == 0:
                print(f"  submissions {i}/{len(uni)}", flush=True)
    filings = pd.concat(filings, ignore_index=True)
    filings = filings[filings["acceptanceDateTime"].str[:10] <= a.until]
    if a.require_item:
        filings = filings[filings["items"].str.contains(a.require_item, regex=False)]
    print(f"8-K filings {a.since} .. {a.until}: {len(filings):,}", flush=True)

    rows, skipped = [], set()
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(exhibit99, client, r.cik, r.accessionNumber): r for r in filings.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                ex = f.result()
            except Exception as e:  # keep going; retried once more at the end
                print("  skip", r.accessionNumber, type(e).__name__, file=sys.stderr)
                skipped.add(r.accessionNumber)
                continue
            if ex:
                rows.append({"symbol": r.symbol, "cik": r.cik, "accession": r.accessionNumber,
                             "accepted_utc": r.accepted_utc, "items": r.items, **ex})
            if i % 500 == 0:
                print(f"  exhibits {i:,}/{len(filings):,} · press releases so far {len(rows):,}", flush=True)
    # second pass over filings that failed on network errors (successes are cached, so this is quick)
    failed = [r for r in filings.itertuples() if r.accessionNumber in skipped]
    for r in failed:
        try:
            ex = exhibit99(client, r.cik, r.accessionNumber)
        except Exception:
            continue
        if ex:
            rows.append({"symbol": r.symbol, "cik": r.cik, "accession": r.accessionNumber,
                         "accepted_utc": r.accepted_utc, "items": r.items, **ex})
    print(f"retried {len(failed)} failed filings", flush=True)
    out = pd.DataFrame(rows).sort_values("accepted_utc").reset_index(drop=True)
    out.to_csv(a.out, index=False)
    print(f"DONE press releases: {len(out):,} from {out['symbol'].nunique()} companies", flush=True)


if __name__ == "__main__":
    main()
