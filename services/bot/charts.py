"""Chart utilities for Discord bot."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


CACHE_DIR = Path("/tmp/midnight_charts")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_chart(symbol: str, df: pd.DataFrame) -> Path:
    """Create a PNG chart with price and moving averages."""
    path = CACHE_DIR / f"{symbol}.png"
    plt.figure(figsize=(6, 3))
    plt.plot(df["date"], df["close"], label="Close", color="black")
    if "sma50" in df:
        plt.plot(df["date"], df["sma50"], label="SMA50")
    if "ema20" in df:
        plt.plot(df["date"], df["ema20"], label="EMA20")
    if "ema9" in df:
        plt.plot(df["date"], df["ema9"], label="EMA9")
    if "hh60" in df:
        plt.plot(df["date"], df["hh60"], label="60d High", linestyle="--")
    plt.legend(loc="upper left")
    plt.title(symbol)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
