"""Backtesting utilities for the breakout strategy."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import yfinance as yf

from ..shared.db import get_session
from ..shared.models import Outcome, Signal
from ..shared.utils import load_universe
from .features import compute_indicators


@dataclass
class TradeResult:
    """Container for a single trade's outcome."""

    symbol: str
    entry_date: date
    exit_date: date
    entry: float
    exit: float
    stop: float
    target1: float
    hold_days: int
    path: str
    r_multiple: float


def simulate_symbol(df: pd.DataFrame, hold_days: int = 63, stop_mult: float = 1.5) -> Tuple[pd.DataFrame, pd.Series]:
    """Simulate trades for a single symbol DataFrame."""
    df = compute_indicators(df)
    trades: List[TradeResult] = []
    equity: List[float] = []
    capital = 0.0
    sym = df.symbol.iloc[0]
    for idx, row in df.iterrows():
        if not row.trigger:
            continue
        entry = float(row.close)
        stop = float(entry - stop_mult * row.atr20)
        t1 = float(entry * 1.15)
        risk = entry - stop
        entry_date = row.date
        exit_price = entry
        exit_idx = idx
        path = "time_only"
        hit_t1 = False
        for i in range(1, hold_days + 1):
            if idx + i >= len(df):
                exit_idx = len(df) - 1
                exit_price = float(df.iloc[exit_idx].close)
                path = "time_only"
                break
            day = df.iloc[idx + i]
            exit_idx = idx + i
            if not hit_t1:
                if day.low <= stop:
                    exit_price = stop
                    path = "stop_first"
                    break
                if day.high >= t1:
                    hit_t1 = True
                    # manage remaining half after T1
                    for j in range(i + 1, hold_days + 1):
                        if idx + j >= len(df):
                            exit_idx = len(df) - 1
                            exit_price = float(df.iloc[exit_idx].close)
                            path = "t1_then_time"
                            break
                        day2 = df.iloc[idx + j]
                        exit_idx = idx + j
                        if day2.low <= stop:
                            exit_price = stop
                            path = "t1_then_stop"
                            break
                        if j == hold_days:
                            exit_price = float(day2.close)
                            path = "t1_then_time"
                            break
                    break
                if i == hold_days:
                    exit_price = float(day.close)
                    path = "time_only"
                    break
        pnl: float
        if path == "stop_first":
            pnl = stop - entry
        elif path == "time_only":
            pnl = exit_price - entry
        elif path == "t1_then_time":
            pnl = (t1 - entry) * 0.5 + (exit_price - entry) * 0.5
        elif path == "t1_then_stop":
            pnl = (t1 - entry) * 0.5 + (stop - entry) * 0.5
        else:
            pnl = 0.0
        r_mult = pnl / risk if risk else 0.0
        hold = exit_idx - idx
        trades.append(
            TradeResult(
                sym,
                entry_date,
                df.iloc[exit_idx].date,
                entry,
                exit_price,
                stop,
                t1,
                hold,
                path,
                r_mult,
            )
        )
        capital += r_mult
        equity.append(capital)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_curve = pd.Series(equity)
    return trades_df, equity_curve


def simulate_universe(start: str, end: str, universe: Iterable[str]) -> Tuple[pd.DataFrame, dict]:
    """Run backtest for a list of symbols."""
    logs: List[pd.DataFrame] = []
    for sym in universe:
        df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            continue
        df = df.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["symbol"] = sym
        trade_df, _ = simulate_symbol(df)
        logs.append(trade_df)
    all_trades = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    with get_session() as session:
        for _, tr in all_trades.iterrows():
            sig = session.query(Signal).filter_by(symbol=tr.symbol, as_of=tr.entry_date).first()
            if sig:
                label = 1 if tr.r_multiple > 0 else 0
                session.merge(Outcome(signal_id=sig.id, label=label, r_multiple=float(tr.r_multiple)))
    wins = all_trades[all_trades.r_multiple > 0]
    losses = all_trades[all_trades.r_multiple <= 0]
    hit_rate = len(wins) / len(all_trades) if len(all_trades) else 0.0
    profit_factor = wins.r_multiple.sum() / abs(losses.r_multiple.sum()) if len(losses) else float("inf")
    expectancy = all_trades.r_multiple.mean() if len(all_trades) else 0.0
    eq = all_trades.r_multiple.cumsum()
    max_dd = (eq.cummax() - eq).max() if not eq.empty else 0.0
    summary = {
        "trades": len(all_trades),
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
    }
    return all_trades, summary


def run_cli() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    universe = load_universe(Path("data/universe.csv"))
    trades, summary = simulate_universe(args.start, args.end, universe)
    out_path = Path("/tmp/backtest.csv")
    trades.to_csv(out_path, index=False)
    print("Summary:", summary)
    print("Saved trades to", out_path)


if __name__ == "__main__":
    run_cli()

