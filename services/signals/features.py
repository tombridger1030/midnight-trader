"""Indicator and feature computation."""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


DEFS = {
    "sma50": 50,
    "sma200": 200,
    "ema20": 20,
    "ema9": 9,
    "atr20": 20,
    "hh60": 60,
}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with indicators and trigger columns."""
    df = df.copy()
    df["sma50"] = ta.sma(df["close"], length=50)
    df["sma200"] = ta.sma(df["close"], length=200)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema9"] = ta.ema(df["close"], length=9)
    df["atr20"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=20)
    df["hh60"] = df["high"].rolling(60).max().shift(1)

    df["trend_ok"] = (
        (df["sma50"] > df["sma200"]) &
        (df["sma50"].diff() > 0) &
        (df["sma200"].diff() > 0)
    )

    vol = df["close"].pct_change().rolling(60).std()
    rng = (df["high"].rolling(60).max() - df["low"].rolling(60).min()) / df["close"]
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min())
    rng_norm = (rng - rng.min()) / (rng.max() - rng.min())
    df["base_quality"] = 1 - (vol_norm + rng_norm) / 2

    df["trigger"] = (df["close"] > df["hh60"]) & df["trend_ok"]
    return df


def compute_telemetry_row(df: pd.DataFrame) -> dict:
    """Compute telemetry fields from latest row."""
    last = df.iloc[-1]
    telem = {
        "sma50": float(last.sma50),
        "sma200": float(last.sma200),
        "ema20": float(last.ema20),
        "ema9": float(last.ema9),
        "price_gt_sma50": bool(last.close > last.sma50),
        "price_gt_ema20": bool(last.close > last.ema20),
        "ema9_gt_ema20": bool(last.ema9 > last.ema20),
        "ret_1m": float(last.close / df["close"].iloc[-21] - 1) if len(df) > 21 else 0.0,
        "ret_3m": float(last.close / df["close"].iloc[-63] - 1) if len(df) > 63 else 0.0,
        "vol_3m": float(df["close"].pct_change().rolling(63).std().iloc[-1]) if len(df) > 63 else 0.0,
        "atr20_pct": float(last.atr20 / last.close) if last.close else 0.0,
        "rs_sector": 0.0,
        "rs_spy": 0.0,
    }
    return telem
