"""Daily OHLCV from Yahoo Finance via yfinance, cached as CSV under data_cache/ (never committed)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ohlcv(symbols: list[str], start: str, end: str, cache_dir: str | Path = "data_cache") -> dict[str, pd.DataFrame]:
    """Split/dividend-adjusted daily bars: {symbol: DataFrame[open, high, low, close, volume]}."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    missing = [s for s in symbols if not (cache / f"{s}_{start}_{end}.csv").exists()]
    if missing:
        import yfinance as yf

        raw = yf.download(missing, start=start, end=end, auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
        for s in missing:
            df = raw[s] if len(missing) > 1 else raw
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1) if df.columns.nlevels > 1 else df
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            if df.empty:
                raise RuntimeError(f"no data for {s}")
            df.to_csv(cache / f"{s}_{start}_{end}.csv")
    for s in symbols:
        df = pd.read_csv(cache / f"{s}_{start}_{end}.csv", index_col=0, parse_dates=True)
        df.index.name = "date"
        out[s] = df
    return out
