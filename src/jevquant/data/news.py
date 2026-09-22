"""Labeled financial headlines: zeroshot/twitter-financial-news-sentiment (MIT, 11,931 tweets).

Downloaded from the Hugging Face Hub on first use and cached locally; never committed.
Labels: 0 = bearish, 1 = bullish, 2 = neutral.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = "zeroshot/twitter-financial-news-sentiment"
FILES = {"train": "sent_train.csv", "validation": "sent_valid.csv"}
LABELS = {0: "bearish", 1: "bullish", 2: "neutral"}


def load_headlines(split: str = "validation", cache_dir: str | Path = "data_cache") -> pd.DataFrame:
    """Return a DataFrame with columns text, label (bearish/bullish/neutral), label_id."""
    path = Path(cache_dir) / f"tfns_{split}.csv"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(REPO, FILES[split], repo_type="dataset")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(src).to_csv(path, index=False)
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    df["label_id"] = df["label"].astype(int)
    df["label"] = df["label_id"].map(LABELS)
    return df[["text", "label", "label_id"]].reset_index(drop=True)
