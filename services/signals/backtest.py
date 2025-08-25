"""Backtesting utilities for the breakout strategy."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import pandas as pd
import yfinance as yf

from ..shared.db import get_session
from ..shared.models import Outcome, Signal, Ticker
from ..shared.utils import load_universe
from .features import compute_indicators
from .ingest_prices import fetch_symbol_df


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


def simulate_symbol(df: pd.DataFrame, hold_days: int = 63, stop_mult: float = 1.5, strategy: str = "alt") -> Tuple[pd.DataFrame, pd.Series]:
    """Simulate trades for a single symbol DataFrame.

    Strategies:
    - 'alt' Aggressive Long-Term Trend (weekly trailing)
    - 'breakout' original daily breakout
    """
    df = compute_indicators(df)
    trades: List[TradeResult] = []
    equity: List[float] = []
    capital = 0.0
    sym = df.symbol.iloc[0]

    if strategy == "alt":
        # Prepare weekly series
        dfw = df.copy()
        dfw["date"] = pd.to_datetime(dfw["date"])  # ensure ts
        dfw = dfw.set_index("date")
        wk_close = dfw["close"].resample("W-FRI").last()
        wk_high = dfw["high"].resample("W-FRI").max()
        wk_low = dfw["low"].resample("W-FRI").min()
        wk_prev_close = wk_close.shift(1)
        # Weekly indicators
        trw = pd.concat([(wk_high - wk_low).abs(), (wk_high - wk_prev_close).abs(), (wk_low - wk_prev_close).abs()], axis=1).max(axis=1)
        atrw14 = trw.rolling(14, min_periods=1).mean().fillna(0.0)
        sma40w = wk_close.rolling(40, min_periods=1).mean()

        for week_end in wk_close.index:
            # Align daily info up to week_end
            daily_idx = dfw.index[dfw.index <= week_end]
            if len(daily_idx) == 0:
                continue
            dlast = dfw.loc[daily_idx.max()]
            row_date = daily_idx.max().date()
            trend_ok = bool((dlast["sma50"] > dlast["sma200"]) and (dfw["sma50"].loc[:week_end].diff().iloc[-1] > 0) and (dfw["sma200"].loc[:week_end].diff().iloc[-1] > 0))
            prior_hh60 = dfw["high"].loc[:week_end].rolling(60, min_periods=1).max().shift(1).iloc[-1]
            trigger = bool(dlast["close"] > prior_hh60)
            if not (trend_ok and trigger):
                continue
            entry = float(dlast["close"])
            stop0 = float(entry * 0.70)  # 30% below
            stop_curr = stop0
            risk = max(entry - stop0, 1e-8)

            exit_week = week_end
            exit_price = entry
            path = "hold"

            # Add/stop state
            added_A = False
            added_B = False
            be_stop_set = False
            tenpct_stop_set = False
            moon_started = False
            moon_realized = 0.0  # realized cash from 50% sale at 2x
            moon_high = None
            below_40_count = 0

            # Iterate weekly forward
            for wk in wk_close.index[wk_close.index.get_loc(week_end) + 1:]:
                wc = float(wk_close.loc[wk])
                # Adds and stop moves before moonbag
                move_r = (wc - entry) / risk
                if not added_A and wc >= entry * 1.30:
                    stop_curr = max(stop_curr, entry)  # move to BE
                    be_stop_set = True
                    added_A = True
                if not added_B and wc >= entry * 1.60:
                    stop_curr = max(stop_curr, entry * 1.10)  # lock +10%
                    tenpct_stop_set = True
                    added_B = True

                # Catastrophic stop before 2x
                if not moon_started and wc <= stop_curr:
                    exit_week = wk
                    exit_price = wc
                    path = "stop_pre2x"
                    break

                # Moonbag trigger at 2x
                if not moon_started and wc >= 2.0 * entry:
                    moon_started = True
                    moon_realized = 0.5 * (wc - entry)  # sell half
                    moon_high = wc
                    # After 2x, ignore stop for moonbag half; use fail-safes
                    below_40_count = 0
                    # continue to next week for moonbag management
                    continue

                if moon_started:
                    moon_high = max(moon_high or wc, wc)
                    # Weekly 40w fail-safe counter
                    if wc < float(sma40w.loc[wk]):
                        below_40_count += 1
                    else:
                        below_40_count = 0
                    # 50% drawdown from post-2x high
                    if wc <= 0.5 * float(moon_high):
                        exit_week = wk
                        exit_price = wc
                        path = "moon_dd50"
                        break
                    if below_40_count >= 3:
                        exit_week = wk
                        exit_price = wc
                        path = "moon_40w"
                        break

                exit_week = wk
                exit_price = wc
                path = "trail"

            # Compute PnL with moonbag if any
            if moon_started:
                pnl_total = moon_realized + 0.5 * (exit_price - entry)
            else:
                pnl_total = (exit_price - entry)
            r_mult = pnl_total / risk
            hold = (exit_week - week_end).days // 7 if exit_week != week_end else 0
            trades.append(TradeResult(sym, row_date, exit_week.date() if isinstance(exit_week, pd.Timestamp) else row_date, entry, exit_price, stop0, entry * 1.15, hold, path, r_mult))
            capital += r_mult
            equity.append(capital)
        return pd.DataFrame([t.__dict__ for t in trades]), pd.Series(equity)
    for idx in range(len(df)):
        row = df.iloc[idx]
        # trend_trail removed
        # ema_cross removed
        # default breakout
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
        pnl = stop - entry if path == "stop_first" else (
            (t1 - entry) * 0.5 + (exit_price - entry) * 0.5 if path in ("t1_then_time","t1_then_stop") else exit_price - entry
        )
        r_mult = pnl / risk if risk else 0.0
        hold = exit_idx - idx
        trades.append(TradeResult(sym, entry_date, df.iloc[exit_idx].date, entry, exit_price, stop, t1, hold, path, r_mult))
        capital += r_mult
        equity.append(capital)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_curve = pd.Series(equity)
    return trades_df, equity_curve


def simulate_universe(start: str, end: str, universe: Iterable[str], *,
                      strategy: str = "alt",
                      start_equity: float = 100_000.0,
                      risk_pct: float = 0.01,
                      alloc_pct: float = 0.02,
                      sizing: str = "risk",
                      max_open_risk_R: float = 5.0,
                      max_positions: Optional[int] = None,
                      cost_bps: float = 0.0015,
                      sector_limit: Optional[int] = None,
                      max_symbol_alloc_pct: float = 0.10) -> Tuple[pd.DataFrame, dict]:
    """Run backtest for a list of symbols using DB-ingested OHLCV.

    Adds portfolio simulation with risk-based position sizing and open-risk caps.
    """
    logs: List[pd.DataFrame] = []
    for sym in universe:
        try:
            df = fetch_symbol_df(sym)
            if df.empty:
                continue
            # Date range filter
            mask = (pd.to_datetime(df["date"]) >= pd.to_datetime(start)) & (pd.to_datetime(df["date"]) <= pd.to_datetime(end))
            df = df.loc[mask].reset_index(drop=True)
            if df.empty:
                continue
            # Ensure numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["symbol"] = sym
            trade_df, _ = simulate_symbol(df, strategy=strategy)
            if not trade_df.empty:
                logs.append(trade_df)
        except Exception as exc:
            print(f"Backtest: skip {sym} due to error: {exc}")
            continue
    all_trades = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=[
        "symbol","entry_date","exit_date","entry","stop","r_multiple"
    ])
    with get_session() as session:
        for _, tr in all_trades.iterrows():
            sig = session.query(Signal).filter_by(symbol=tr.symbol, as_of=tr.entry_date).first()
            if sig:
                label = 1 if float(tr.r_multiple) > 0 else 0
                session.merge(Outcome(signal_id=sig.id, label=label, r_multiple=float(tr.r_multiple)))
    # Summary
    if all_trades.empty:
        summary = {"trades": 0, "hit_rate": 0.0, "profit_factor": float("inf"), "expectancy": 0.0, "max_drawdown": 0.0,
                   "portfolio_equity_end": start_equity, "portfolio_trades": 0}
        return all_trades, summary
    wins = all_trades[all_trades["r_multiple"] > 0]
    losses = all_trades[all_trades["r_multiple"] <= 0]
    hit_rate = len(wins) / len(all_trades) if len(all_trades) else 0.0
    profit_factor = wins["r_multiple"].sum() / abs(losses["r_multiple"].sum()) if len(losses) else float("inf")
    expectancy = all_trades["r_multiple"].mean() if len(all_trades) else 0.0
    eq = all_trades["r_multiple"].cumsum()
    max_dd = (eq.cummax() - eq).max() if not eq.empty else 0.0
    # Portfolio-level simulation with risk sizing and open-risk cap
    portfolio_equity = start_equity
    symbol_to_sector: Dict[str, str] = {}
    try:
        with get_session() as s:
            unique_syms = list(set(all_trades["symbol"]))
            if unique_syms:
                rows = s.query(Ticker).filter(Ticker.symbol.in_(unique_syms)).all()
                for t in rows:
                    symbol_to_sector[t.symbol] = (t.sector or "UNKNOWN")
    except Exception:
        pass

    trades = all_trades.copy()
    trades = trades.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    # Build ordered unique dates of all entries and exits
    all_dates = pd.to_datetime(pd.unique(trades[["entry_date", "exit_date"]].values.ravel("K")))
    all_dates = sorted(all_dates)

    open_positions: List[dict] = []
    portfolio_trades = 0
    for current_date in all_dates:
        # Close positions due today
        to_close = [pos for pos in open_positions if pd.to_datetime(pos["exit_date"]) == current_date]
        if to_close:
            remain = []
            for pos in open_positions:
                if pos in to_close:
                    # Cash PnL using allocated shares
                    pnl_cash = pos["shares"] * (float(pos["exit_price"]) - float(pos["entry_price"])) - pos["cost_cash"]
                    portfolio_equity += pnl_cash
                else:
                    remain.append(pos)
            open_positions = remain

        # Consider entries today
        todays = trades[pd.to_datetime(trades["entry_date"]) == current_date]
        # Compute current open position count and open risk units (in R)
        open_count = len(open_positions)
        open_r_units = float(len(open_positions)) if sizing == "risk" else 0.0
        # Sector counts
        sector_counts: Dict[str, int] = {}
        if sector_limit is not None:
            for pos in open_positions:
                sec = symbol_to_sector.get(pos["symbol"], "UNKNOWN")
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

        # Weekly pacing: allow at most 2 new buys per week and within risk budget (ALT only)
        slots = None
        if strategy == "alt" and sizing == "risk":
            remaining_R = max(0.0, max_open_risk_R - open_r_units)
            slots = int(remaining_R / max(risk_pct, 1e-9))
            slots = min(2, max(0, slots))
        accepted_this_date = 0

        for _, tr in todays.sort_values(by=["symbol"]).iterrows():
            if strategy == "alt" and slots is not None and accepted_this_date >= slots:
                continue
            if max_positions is not None and open_count >= max_positions:
                continue
            symbol = str(tr.symbol)
            sec = symbol_to_sector.get(symbol, "UNKNOWN")
            if sector_limit is not None and sector_counts.get(sec, 0) >= sector_limit:
                continue
            # Enforce at most one concurrent position per symbol
            if any(p["symbol"] == symbol for p in open_positions):
                continue
            entry = float(tr.entry)
            stop_price = float(tr.stop) if "stop" in tr else max(entry - 1e-8, 0.0)
            if sizing == "risk":
                # Risk-based shares
                risk_cash = max(portfolio_equity * risk_pct, 0.0)
                risk_per_share = max(entry - stop_price, 1e-8)
                shares = int(risk_cash // risk_per_share)
                # Enforce open risk cap in R units
                if open_r_units + 1.0 > max_open_risk_R:
                    continue
            else:
                # Allocation-based sizing
                alloc_cash = max(portfolio_equity * alloc_pct, 0.0)
                shares = int(alloc_cash // max(entry, 1e-8))
            if shares <= 0:
                continue
            notional = shares * entry
            # Enforce per-symbol max allocation cap
            current_symbol_notional = sum(p["shares"] * p["entry_price"] for p in open_positions if p["symbol"] == symbol)
            if (current_symbol_notional + notional) > (portfolio_equity * max_symbol_alloc_pct):
                continue
            cost_cash = 2.0 * cost_bps * notional
            open_positions.append({
                "symbol": symbol,
                "exit_date": tr.exit_date,
                "entry_price": entry,
                "exit_price": float(tr.exit),
                "shares": shares,
                "cost_cash": cost_cash,
            })
            open_count += 1
            if sizing == "risk":
                open_r_units += 1.0
            if sector_limit is not None:
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            portfolio_trades += 1
            if strategy == "alt" and slots is not None:
                accepted_this_date += 1

    summary = {
        "trades": int(len(all_trades)),
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": float(max_dd),
        "portfolio_equity_end": float(portfolio_equity),
        "portfolio_trades": int(portfolio_trades),
        "params": {
            "start_equity": start_equity,
            "risk_pct": risk_pct,
            "sizing": sizing,
            "max_open_risk_R": max_open_risk_R,
            "max_positions": max_positions,
            "cost_bps": cost_bps,
            "sector_limit": sector_limit,
        },
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

