"""Generate trade signals and persist to DB."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..shared.db import get_session
from ..shared.models import Signal, SignalFeatures, Ticker
from ..shared.utils import load_universe
from .ingest_prices import fetch_symbol_df
from .features import compute_indicators, compute_telemetry_row


LOOKBACK_DAYS = 63
HOLD_DAYS = 63
STOP_MULT = 1.5


def generate_for_symbol(symbol: str) -> None:
    df = fetch_symbol_df(symbol)
    if df.empty:
        return
    df = compute_indicators(df)
    last = df.iloc[-1]
    if not bool(last.trigger):
        return

    telem = compute_telemetry_row(df)
    entry = float(last.close)
    stop = float(last.close - STOP_MULT * last.atr20)
    sig = Signal(
        symbol=symbol,
        as_of=date.fromisoformat(str(last.date)),
        lookback_days=LOOKBACK_DAYS,
        hold_days=HOLD_DAYS,
        trend_ok=bool(last.trend_ok),
        base_quality=float(last.base_quality),
        entry=entry,
        stop=stop,
        target1=entry * 1.15,
        target2=entry * 1.30,
        decision="propose_entry",
        plan_json={
            "rules": "60d breakout",
            "holding_days": HOLD_DAYS,
            "stop_mult_atr20": STOP_MULT,
            "targets": {"t1_pct": 0.15, "t2_pct": 0.30},
            "manage": {"scale_half_at_t1": True},
            "price_gt_sma50": telem["price_gt_sma50"],
            "price_gt_ema20": telem["price_gt_ema20"],
            "ema9_gt_ema20": telem["ema9_gt_ema20"],
        },
    )

    feat = SignalFeatures(signal=sig, **telem)

    with get_session() as session:
        session.add(sig)
        session.add(feat)


def generate_from_universe(universe_path: str | Path) -> None:
    symbols = load_universe(universe_path)
    for sym in symbols:
        generate_for_symbol(sym)


if __name__ == "__main__":
    generate_from_universe(Path("data/universe.csv"))
