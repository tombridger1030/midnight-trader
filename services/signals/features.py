"""Indicator and feature computation."""
from __future__ import annotations

import pandas as pd
import numpy as np


DEFS = {
    "sma50": 50,
    "sma200": 200,
    "ema20": 20,
    "ema9": 9,
    "atr20": 20,
    "hh60": 60,
}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with indicators and trigger columns using pandas only (no pandas_ta).

    Adds additional fields for strategy variants: atr14, Donchian(20), Keltner(20,2),
    band position p, and RSI(2).
    """
    df = df.copy()
    # Ensure numeric types and sanitize None -> NaN
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Moving averages
    df["sma50"] = df["close"].rolling(50, min_periods=1).mean()
    df["sma200"] = df["close"].rolling(200, min_periods=1).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False, min_periods=1).mean()
    df["ema9"] = df["close"].ewm(span=9, adjust=False, min_periods=1).mean()
    # ATR(20)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr20"] = tr.rolling(20, min_periods=1).mean().fillna(0.0)
    # ATR(14)
    df["atr14"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)
    # Highest high previous 60 days
    df["hh60"] = df["high"].rolling(60, min_periods=1).max().shift(1)
    # Donchian(20)
    dc_upper = df["high"].rolling(20, min_periods=1).max()
    dc_lower = df["low"].rolling(20, min_periods=1).min()
    df["donchian20_upper"] = dc_upper
    df["donchian20_lower"] = dc_lower
    df["donchian20_mid"] = (dc_upper + dc_lower) / 2.0
    # Keltner(ema20 ± 2*atr20)
    df["keltner_upper"] = df["ema20"] + 2.0 * df["atr20"]
    df["keltner_lower"] = df["ema20"] - 2.0 * df["atr20"]
    width = (df["keltner_upper"] - df["keltner_lower"]).replace(0, np.nan)
    df["p"] = ((df["close"] - df["keltner_lower"]) / width).clip(0.0, 1.0).fillna(0.0)
    # Coerce indicators to numeric and fill NaNs where safe
    for col in ["sma50", "sma200", "ema20", "ema9", "atr20", "atr14", "hh60",
                "donchian20_upper", "donchian20_lower", "donchian20_mid",
                "keltner_upper", "keltner_lower", "p"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["atr20"] = df["atr20"].fillna(0.0)

    df["trend_ok"] = (
        (df["sma50"] > df["sma200"]) &
        (df["sma50"].diff() > 0) &
        (df["sma200"].diff() > 0)
    )

    vol = df["close"].pct_change().rolling(60, min_periods=1).std()
    rng = (df["high"].rolling(60, min_periods=1).max() - df["low"].rolling(60, min_periods=1).min()) / df["close"].replace(0, np.nan)
    vol_range = (vol.max() - vol.min()) or 1
    rng_range = (rng.max() - rng.min()) or 1
    vol_norm = (vol - vol.min()) / vol_range
    rng_norm = (rng - rng.min()) / rng_range
    df["base_quality"] = 1 - (vol_norm + rng_norm) / 2

    df["trigger"] = (df["close"] > df["hh60"]) & df["trend_ok"]
    df["trigger"] = df["trigger"].fillna(False)
    # RSI(2)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(2, min_periods=1).mean()
    avg_loss = loss.rolling(2, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi2"] = (100 - (100 / (1 + rs))).fillna(0.0)
    return df


def compute_telemetry_row(df: pd.DataFrame) -> dict:
    """Compute telemetry fields from latest row."""
    last = df.iloc[-1]
    def _safe_float(x: float) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _safe_bool_gt(a: float, b: float) -> bool:
        a_val = _safe_float(a)
        b_val = _safe_float(b)
        if np.isnan(a_val) or np.isnan(b_val):
            return False
        return bool(a_val > b_val)

    telem = {
        "sma50": _safe_float(last.sma50),
        "sma200": _safe_float(last.sma200),
        "ema20": _safe_float(last.ema20),
        "ema9": _safe_float(last.ema9),
        "price_gt_sma50": _safe_bool_gt(last.close, last.sma50),
        "price_gt_ema20": _safe_bool_gt(last.close, last.ema20),
        "ema9_gt_ema20": _safe_bool_gt(last.ema9, last.ema20),
        "ret_1m": float(last.close / df["close"].iloc[-21] - 1) if len(df) > 21 else 0.0,
        "ret_3m": float(last.close / df["close"].iloc[-63] - 1) if len(df) > 63 else 0.0,
        "vol_3m": float(df["close"].pct_change().rolling(63).std().iloc[-1]) if len(df) > 63 else 0.0,
        "atr20_pct": (float(last.atr20) / float(last.close)) if (pd.notna(last.atr20) and pd.notna(last.close) and float(last.close) != 0.0) else 0.0,
        "rs_sector": 0.0,
        "rs_spy": 0.0,
    }
    return telem
