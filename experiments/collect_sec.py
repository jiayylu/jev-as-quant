"""Collect 8-K press releases (EX-99.x) for the E6 universe from SEC EDGAR.

Universe: current S&P 500 members that joined the index before 2024-01-01 (Wikipedia list,
saved in data_cache/sec/). Needs SEC_USER_AGENT="<name> <email>". Every response is cached,
so the script can be stopped and resumed. Output: data_cache/sec/press_releases.csv.gz
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import DATA_CACHE
from jevquant.data.sec import SecClient, exhibit99, list_8k, sp500_from_wikitext

SINCE = "2024-01-01"


def main():
    wiki = sorted((DATA_CACHE / "sec").glob("sp500_*.wiki"))[-1]
    sp = sp500_from_wikitext(wiki.read_text())
    uni = sp[sp["added"] < SINCE].reset_index(drop=True)
    uni.to_csv(DATA_CACHE / "sec" / "universe.csv", index=False)
    client = SecClient(cache_dir=DATA_CACHE / "sec" / "raw")
    print(f"universe: {len(uni)} companies (S&P 500 members since before {SINCE}) from {wiki.name}", flush=True)

    filings = []
    with ThreadPoolExecutor(10) as pool:
        futs = {pool.submit(list_8k, client, r.cik, SINCE): r for r in uni.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            df = f.result()
            if len(df):
                df["symbol"] = r.symbol
                filings.append(df)
            if i % 50 == 0:
                print(f"  submissions {i}/{len(uni)}", flush=True)
    filings = pd.concat(filings, ignore_index=True)
    print(f"8-K filings since {SINCE}: {len(filings):,}", flush=True)

    rows = []
    with ThreadPoolExecutor(10) as pool:
        futs = {pool.submit(exhibit99, client, r.cik, r.accessionNumber): r for r in filings.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                ex = f.result()
            except Exception as e:  # keep going; a handful of malformed filings should not stop the run
                print("  skip", r.accessionNumber, type(e).__name__, file=sys.stderr)
                continue
            if ex:
                rows.append({"symbol": r.symbol, "cik": r.cik, "accession": r.accessionNumber,
                             "accepted_utc": r.accepted_utc, "items": r.items, **ex})
            if i % 500 == 0:
                print(f"  exhibits {i:,}/{len(filings):,} · press releases so far {len(rows):,}", flush=True)
    out = pd.DataFrame(rows).sort_values("accepted_utc").reset_index(drop=True)
    out.to_csv(DATA_CACHE / "sec" / "press_releases.csv.gz", index=False)
    print(f"DONE press releases: {len(out):,} from {out['symbol'].nunique()} companies", flush=True)


if __name__ == "__main__":
    main()
