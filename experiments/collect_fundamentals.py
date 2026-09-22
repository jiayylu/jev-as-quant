"""Point-in-time fundamentals from SEC XBRL company facts.

For every (company, concept, reporting period) keep the value from the FIRST filing that
reported it, with that filing's date, so a backtest only sees numbers that were public at the
time (the frames API returns later restatements instead). Output:
data_cache/sec/fundamentals_pit.csv.gz with cik, concept, start, end, fp, form, filed, val.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import DATA_CACHE
from jevquant.data.sec import SecClient

CONCEPTS = {
    "us-gaap": ["NetIncomeLoss", "OperatingIncomeLoss", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "GrossProfit", "Assets", "StockholdersEquity", "NetCashProvidedByUsedInOperatingActivities",
                "WeightedAverageNumberOfDilutedSharesOutstanding"],
    "dei": ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"],
}


def facts(client: SecClient, cik: str) -> list[dict]:
    d = client.json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    if not d:
        return []
    rows = []
    for tax, names in CONCEPTS.items():
        for name in names:
            node = d.get("facts", {}).get(tax, {}).get(name)
            if not node:
                continue
            for unit, items in node["units"].items():
                for it in items:
                    if it.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                        continue
                    rows.append({"cik": cik, "concept": name, "unit": unit, "start": it.get("start"), "end": it["end"],
                                 "fp": it.get("fp"), "form": it["form"], "filed": it["filed"], "val": it["val"]})
    return rows


def main():
    uni = pd.read_csv(DATA_CACHE / "sec" / "universe_hist.csv", dtype=str)
    client = SecClient(cache_dir=DATA_CACHE / "sec" / "raw")
    rows = []
    with ThreadPoolExecutor(6) as pool:
        futs = [pool.submit(facts, client, c) for c in uni["cik"].unique()]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 100 == 0:
                print(f"  companyfacts {i}/{len(futs)}", flush=True)
    df = pd.DataFrame(rows)
    # first report of each period wins (no restatements), then keep the earliest filing date
    df = df.sort_values("filed").drop_duplicates(["cik", "concept", "unit", "start", "end"], keep="first")
    df.to_csv(DATA_CACHE / "sec" / "fundamentals_pit.csv.gz", index=False)
    print(f"DONE {len(df):,} point-in-time facts for {df.cik.nunique()} companies", flush=True)


if __name__ == "__main__":
    main()
