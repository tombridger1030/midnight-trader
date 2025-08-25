"""Plot strategy performance charts from a backtest CSV.

Usage:
  # Raw R-based charts (from r_multiple/exit_date)
  python services/analysis/plot_backtest.py --csv .tmp/backtest_10y.csv --outdir .tmp/charts --mode r

  # Portfolio-based charts (simulate allocation/slots over the trade CSV)
  python services/analysis/plot_backtest.py --csv .tmp/backtest_10y_slots_alloc.csv --outdir .tmp/charts --mode portfolio --equity 10000 --alloc 0.02 --max_positions 10 --max_symbol_alloc 0.10 --cost_bps 0.0005
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _portfolio_series_from_trades(df: pd.DataFrame, *, start_equity: float, alloc_pct: float, max_positions: int, max_symbol_alloc: float, cost_bps: float) -> pd.Series:
    """Simulate portfolio equity series given trade CSV and allocation rules."""
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])  # type: ignore
    df["exit_date"] = pd.to_datetime(df["exit_date"])  # type: ignore
    df = df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    if df.empty:
        return pd.Series([], dtype=float)

    all_dates = pd.to_datetime(pd.unique(df[["entry_date", "exit_date"]].values.ravel("K")))
    all_dates = sorted(all_dates)

    equity = start_equity
    equity_by_date = {}
    open_positions: list[dict] = []

    for current_date in all_dates:
        # Close positions exiting today
        to_close = [pos for pos in open_positions if pd.to_datetime(pos["exit_date"]) == current_date]
        if to_close:
            remain = []
            for pos in open_positions:
                if pos in to_close:
                    pnl_cash = pos["shares"] * (float(pos["exit_price"]) - float(pos["entry_price"])) - pos["cost_cash"]
                    equity += pnl_cash
                else:
                    remain.append(pos)
            open_positions = remain
        equity_by_date[current_date] = equity

        # Consider entries today
        todays = df[pd.to_datetime(df["entry_date"]) == current_date]
        open_count = len(open_positions)
        for _, tr in todays.iterrows():
            if open_count >= max_positions:
                continue
            symbol = str(tr.symbol)
            if any(p["symbol"] == symbol for p in open_positions):
                continue
            entry = float(tr.entry)
            alloc_cash = max(equity * alloc_pct, 0.0)
            shares = int(alloc_cash // max(entry, 1e-8))
            if shares <= 0:
                continue
            notional = shares * entry
            current_symbol_notional = sum(p["shares"] * p["entry_price"] for p in open_positions if p["symbol"] == symbol)
            if (current_symbol_notional + notional) > (equity * max_symbol_alloc):
                continue
            cost_cash = 2.0 * cost_bps * notional
            open_positions.append({
                "symbol": symbol,
                "entry_price": entry,
                "exit_price": float(tr.exit),
                "exit_date": tr.exit_date,
                "shares": shares,
                "cost_cash": cost_cash,
            })
            open_count += 1

    # Build daily equity series by forward fill
    s = pd.Series(equity_by_date)
    s.index = pd.to_datetime(s.index)
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_index).ffill()
    return s


def analyze_and_plot(csv_path: Path, outdir: Path, *, mode: str = "r", start_equity: float = 100_000.0,
                     alloc_pct: float = 0.02, max_positions: int = 10, max_symbol_alloc: float = 0.10, cost_bps: float = 0.0005) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["entry_date", "exit_date"]) if csv_path.exists() else pd.DataFrame()
    if df.empty:
        return {"trades": 0}
    if mode == "portfolio":
        equity = _portfolio_series_from_trades(df,
                                               start_equity=start_equity,
                                               alloc_pct=alloc_pct,
                                               max_positions=max_positions,
                                               max_symbol_alloc=max_symbol_alloc,
                                               cost_bps=cost_bps)
        if equity.empty:
            return {"trades": 0}
        monthly_nav = equity.resample("ME").last()
        monthly_ret = monthly_nav.pct_change().fillna(0.0)
        rolling_max = equity.cummax()
        drawdown = (rolling_max - equity) / rolling_max.replace(0, 1)
        max_dd = float(drawdown.max()) if not drawdown.empty else 0.0
        yearly_nav = equity.resample("YE").last()
        yearly_ret = yearly_nav.pct_change().fillna(0.0)
        std_m = monthly_ret.std(ddof=1)
        sharpe = float(monthly_ret.mean() / std_m * (12 ** 0.5)) if pd.notna(std_m) and std_m != 0 else float("inf")
        heat = (monthly_ret.to_frame("RET")
                .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
                .pivot(index="Year", columns="Month", values="RET").fillna(0.0))
    else:
        daily_r = df.set_index("exit_date")["r_multiple"].sort_index().groupby(level=0).sum()
        equity = daily_r.cumsum()
        rolling_max = equity.cummax()
        drawdown = rolling_max - equity
        max_dd = float(drawdown.max()) if not drawdown.empty else 0.0
        monthly = daily_r.resample("ME").sum()
        yearly = daily_r.resample("YE").sum()
        std_m = monthly.std(ddof=1)
        sharpe = float(monthly.mean() / std_m * (12 ** 0.5)) if pd.notna(std_m) and std_m != 0 else float("inf")
        heat = (monthly.to_frame("R")
                .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
                .pivot(index="Year", columns="Month", values="R").fillna(0.0))

    # Equity & Drawdown
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(equity.index, equity.values, label="Equity")
    ax[0].set_title("Equity Curve")
    ax[0].grid(True, alpha=0.3)
    ax[1].fill_between(drawdown.index, drawdown.values, step="pre", color="red", alpha=0.3)
    ax[1].set_title(f"Drawdown (Max: {max_dd:.2%})" if mode == "portfolio" else f"Drawdown (Max: {max_dd:.2f} R)")
    ax[1].grid(True, alpha=0.3)
    eq_path = outdir / "backtest_equity.png"
    fig.tight_layout()
    fig.savefig(eq_path)
    plt.close(fig)

    # Monthly heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=heat.values.min(), vmax=heat.values.max())
    ax2.set_title("Monthly Returns (% nav)" if mode == "portfolio" else "Monthly Returns (R)")
    ax2.set_yticks(range(len(heat.index)))
    ax2.set_yticklabels(heat.index.astype(int))
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    fig2.colorbar(im, ax=ax2)
    mh_path = outdir / "backtest_monthly_heatmap.png"
    fig2.tight_layout()
    fig2.savefig(mh_path)
    plt.close(fig2)

    # Rolling 12-month return
    if mode == "portfolio":
        roll12 = monthly_nav.pct_change(12).fillna(0.0)
    else:
        roll12 = monthly.rolling(12, min_periods=1).sum()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(roll12.index, roll12.values)
    ax3.set_title("Rolling 12-Month Return (% nav)" if mode == "portfolio" else "Rolling 12-Month Return (R)")
    ax3.grid(True, alpha=0.3)
    r12_path = outdir / "backtest_rolling12.png"
    fig3.tight_layout()
    fig3.savefig(r12_path)
    plt.close(fig3)

    # Yearly returns
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    if mode == "portfolio":
        ax4.bar(yearly_ret.index.year.astype(int), 100 * yearly_ret.values)
        ax4.set_title("Yearly Returns (% nav)")
    else:
        ax4.bar(yearly.index.year.astype(int), yearly.values)
        ax4.set_title("Yearly Returns (R)")
    ax4.grid(True, axis="y", alpha=0.3)
    yr_path = outdir / "backtest_yearly.png"
    fig4.tight_layout()
    fig4.savefig(yr_path)
    plt.close(fig4)

    return {
        "trades": int(len(df)),
        "sharpe_monthly": float(sharpe),
        "max_drawdown": float(max_dd),
        "equity_path": str(eq_path),
        "monthly_heatmap_path": str(mh_path),
        "rolling12_path": str(r12_path),
        "yearly_path": str(yr_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default=str(Path(".tmp/charts")))
    parser.add_argument("--mode", choices=["r", "portfolio"], default="r")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--alloc", type=float, default=0.02)
    parser.add_argument("--max_positions", type=int, default=10)
    parser.add_argument("--max_symbol_alloc", type=float, default=0.10)
    parser.add_argument("--cost_bps", type=float, default=0.0005)
    args = parser.parse_args()
    res = analyze_and_plot(Path(args.csv), Path(args.outdir), mode=args.mode,
                           start_equity=args.equity, alloc_pct=args.alloc, max_positions=args.max_positions,
                           max_symbol_alloc=args.max_symbol_alloc, cost_bps=args.cost_bps)
    if res.get("trades", 0) == 0:
        print("No trades to analyze.")
        return
    print("Charts saved:")
    print(" -", res["equity_path"])
    print(" -", res["monthly_heatmap_path"])
    print(" -", res["rolling12_path"])
    print(" -", res["yearly_path"])
    print(f"Sharpe (monthly): {res['sharpe_monthly']:.2f}  Max DD: {res['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()


