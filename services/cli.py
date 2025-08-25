"""Midnight Trader CLI.

Commands:
- db-init: create tables
- db-test: test DB connectivity
- ingest: fetch OHLCV and store
- generate: compute signals from universe
- backtest: run quick backtest
- discord-test: verify Discord token login
- smoke-test: end-to-end tiny pipeline
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .shared.db import init_db, test_connection, get_session
from .signals.ingest_prices import ingest_universe
from .signals.generate import generate_from_universe, generate_for_symbol
from .signals.backtest import simulate_universe
from .shared.utils import load_universe
from .shared.models import Outcome
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .shared import models


def cmd_db_init(_: argparse.Namespace) -> int:
    try:
        init_db()
        print("DB INIT: PASS")
        return 0
    except Exception as exc:
        print(f"DB INIT: FAIL -> {exc}")
        return 1


def cmd_db_test(_: argparse.Namespace) -> int:
    ok = test_connection()
    print("DB TEST: PASS" if ok else "DB TEST: FAIL")
    return 0 if ok else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        ingest_universe(args.universe, start_date=args.start, symbols_override=None)
        print("INGEST: PASS")
        return 0
    except Exception as exc:
        print(f"INGEST: FAIL -> {exc}")
        return 1


def cmd_generate(args: argparse.Namespace) -> int:
    try:
        if args.strategy:
            # ad-hoc: iterate and call per symbol to pass strategy
            from .shared.utils import load_universe as _lu
            for sym in _lu(Path(args.universe)):
                generate_for_symbol(sym, strategy=args.strategy)
        else:
            generate_from_universe(Path(args.universe))
        print("GENERATE: PASS")
        return 0
    except Exception as exc:
        print(f"GENERATE: FAIL -> {exc}")
        return 1


def cmd_backtest(args: argparse.Namespace) -> int:
    try:
        symbols = load_universe(Path(args.universe))
        # Count outcomes before
        before = 0
        try:
            with get_session() as s:
                before = s.query(Outcome).count()
        except Exception:
            pass
        trades, summary = simulate_universe(
            args.start,
            args.end,
            symbols,
            strategy=args.strategy,
            start_equity=args.equity,
            risk_pct=args.risk,
            alloc_pct=args.alloc,
            sizing=args.sizing,
            max_open_risk_R=args.max_open_r,
            max_positions=args.max_positions,
            cost_bps=args.cost_bps,
            sector_limit=args.sector_limit,
            max_symbol_alloc_pct=args.max_symbol_alloc,
        )
        out = Path(args.output)
        trades.to_csv(out, index=False)
        print("BACKTEST SUMMARY:", summary)
        try:
            preview = trades.head(5)
            print("BACKTEST TRADES (head):\n", preview.to_string(index=False))
        except Exception:
            pass
        # Count outcomes after
        after = before
        try:
            with get_session() as s:
                after = s.query(Outcome).count()
        except Exception:
            pass
        delta = after - before
        print(f"BACKTEST: PASS -> trades csv: {out}; Outcomes saved: {delta}")
        return 0
    except Exception as exc:
        print(f"BACKTEST: FAIL -> {exc}")
        return 1


def cmd_discord_test(_: argparse.Namespace) -> int:
    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("DISCORD TEST: FAIL -> DISCORD_BOT_TOKEN missing")
        return 1
    try:
        import asyncio
        import discord  # type: ignore

        async def _run() -> None:
            intents = discord.Intents.none()
            client = discord.Client(intents=intents)

            @client.event
            async def on_ready() -> None:  # type: ignore
                await client.close()

            await client.start(token)

        asyncio.run(_run())
        print("DISCORD TEST: PASS")
        return 0
    except Exception as exc:
        print(f"DISCORD TEST: FAIL -> {exc}")
        return 1


def cmd_smoke(_: argparse.Namespace) -> int:
    """Run a minimal e2e pipeline on a tiny universe."""
    # Prepare tiny universe file
    tmp_dir = Path(".tmp")
    tmp_dir.mkdir(exist_ok=True)
    uni_path = tmp_dir / "universe_smoke.csv"
    uni_path.write_text("AAPL\nMSFT\n")

    # Fast window: ~90 days
    start = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    end = datetime.utcnow().strftime("%Y-%m-%d")

    failures: List[str] = []

    try:
        init_db()
        print("SMOKE: DB INIT PASS")
    except Exception as exc:
        print(f"SMOKE: DB INIT FAIL -> {exc}")
        failures.append("db-init")

    if not test_connection():
        print("SMOKE: DB TEST FAIL")
        failures.append("db-test")
    else:
        print("SMOKE: DB TEST PASS")

    try:
        ingest_universe(str(uni_path), start_date=start)
        print("SMOKE: INGEST PASS")
    except Exception as exc:
        print(f"SMOKE: INGEST FAIL -> {exc}")
        failures.append("ingest")

    try:
        generate_from_universe(str(uni_path))
        print("SMOKE: GENERATE PASS")
    except Exception as exc:
        print(f"SMOKE: GENERATE FAIL -> {exc}")
        failures.append("generate")

    try:
        symbols = load_universe(uni_path)
        trades, summary = simulate_universe(start, end, symbols)
        print("SMOKE BACKTEST SUMMARY:", summary)
        try:
            print("SMOKE BACKTEST TRADES (head):\n", trades.head(5).to_string(index=False))
        except Exception:
            pass
        print("SMOKE: BACKTEST PASS")
    except Exception as exc:
        print(f"SMOKE: BACKTEST FAIL -> {exc}")
        failures.append("backtest")

    if failures:
        print(f"SMOKE: FAIL -> steps failed: {', '.join(failures)}")
        return 1
    print("SMOKE: PASS")
    return 0


def _copy_table(session_src, session_dst, model) -> int:
    rows = session_src.query(model).all()
    count = 0
    for row in rows:
        # Build a new instance for destination to avoid identity conflicts
        data = {c.name: getattr(row, c.name) for c in model.__table__.columns}
        session_dst.merge(model(**data))
        count += 1
    session_dst.commit()
    return count


def cmd_migrate_to_supabase(args: argparse.Namespace) -> int:
    try:
        src_url = args.source or os.getenv("DB_URL") or "sqlite:///midnight.db"
        dst_url = args.target or os.getenv("SUPABASE_DB_URL")
        if not dst_url:
            print("MIGRATE: FAIL -> SUPABASE_DB_URL not set and --target not provided")
            return 1
        if "sslmode=" not in dst_url and (dst_url.startswith("postgresql://") or dst_url.startswith("postgres://")):
            dst_url = dst_url + ("&" if "?" in dst_url else "?") + "sslmode=require"

        eng_src = create_engine(src_url, future=True, pool_pre_ping=True)
        eng_dst = create_engine(dst_url, future=True, pool_pre_ping=True)

        SessionSrc = sessionmaker(bind=eng_src, autoflush=False)
        SessionDst = sessionmaker(bind=eng_dst, autoflush=False)

        # Ensure tables exist on destination
        models.Base.metadata.create_all(bind=eng_dst)

        with SessionSrc() as s_src, SessionDst() as s_dst:
            total = 0
            for model in [models.Ticker, models.OHLCVDaily, models.Signal, models.SignalFeatures, models.Position, models.Journal, models.Outcome]:
                copied = _copy_table(s_src, s_dst, model)
                print(f"MIGRATE: Copied {copied} rows for {model.__tablename__}")
                total += copied
        print(f"MIGRATE: PASS -> total rows copied: {total}")
        return 0
    except Exception as exc:
        print(f"MIGRATE: FAIL -> {exc}")
        return 1


def _analyze_backtest(csv_path: Path, outdir: Path) -> dict:
    import pandas as pd
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["entry_date", "exit_date"])
    if df.empty:
        return {"trades": 0}

    # Build daily equity (in R) from exit dates
    daily_r = (
        df.set_index("exit_date")["r_multiple"].sort_index().groupby(level=0).sum()
    )
    equity = daily_r.cumsum()
    rolling_max = equity.cummax()
    drawdown = rolling_max - equity
    max_dd = float(drawdown.max()) if not drawdown.empty else 0.0

    # Monthly and yearly returns (sum of R per month/year)
    monthly = daily_r.resample("M").sum()
    yearly = daily_r.resample("Y").sum()

    # Sharpe based on monthly R (approximation)
    sharpe = float("inf")
    if monthly.std(ddof=1) not in (0, None) and not monthly.std(ddof=1) != monthly.std(ddof=1):
        try:
            sharpe = float(monthly.mean() / monthly.std(ddof=1) * (12 ** 0.5))
        except Exception:
            sharpe = float("nan")

    # Monthly heatmap data
    mh = monthly.to_frame("R").copy()
    mh["Year"] = mh.index.year
    mh["Month"] = mh.index.month
    heat = mh.pivot(index="Year", columns="Month", values="R").fillna(0.0)

    # Plot equity and drawdown
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(equity.index, equity.values, label="Equity (R)")
    ax[0].set_title("Equity Curve (R)")
    ax[0].grid(True, alpha=0.3)
    ax[1].fill_between(drawdown.index, drawdown.values, step="pre", color="red", alpha=0.3)
    ax[1].set_title(f"Drawdown (Max: {max_dd:.2f} R)")
    ax[1].grid(True, alpha=0.3)
    eq_path = outdir / "backtest_equity.png"
    fig.tight_layout()
    fig.savefig(eq_path)
    plt.close(fig)

    # Plot monthly heatmap (as annotated table-like image)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=heat.values.min(), vmax=heat.values.max())
    ax2.set_title("Monthly Returns (R)")
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
    roll12 = monthly.rolling(12, min_periods=1).sum()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(roll12.index, roll12.values)
    ax3.set_title("Rolling 12-Month Return (R)")
    ax3.grid(True, alpha=0.3)
    r12_path = outdir / "backtest_rolling12.png"
    fig3.tight_layout()
    fig3.savefig(r12_path)
    plt.close(fig3)

    # Yearly returns bar
    fig4, ax4 = plt.subplots(figsize=(12, 4))
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
        "max_drawdown_R": max_dd,
        "equity_path": str(eq_path),
        "monthly_heatmap_path": str(mh_path),
        "rolling12_path": str(r12_path),
        "yearly_path": str(yr_path),
    }


def cmd_analyze_backtest(args: argparse.Namespace) -> int:
    try:
        csv_path = Path(args.csv)
        outdir = Path(args.outdir)
        res = _analyze_backtest(csv_path, outdir)
        if res.get("trades", 0) == 0:
            print("ANALYZE: PASS -> No trades found in CSV, charts skipped")
            return 0
        print("ANALYZE: PASS -> charts saved:")
        for k in ["equity_path", "monthly_heatmap_path", "rolling12_path", "yearly_path"]:
            print(" -", res[k])
        print(f"Sharpe (monthly): {res['sharpe_monthly']:.2f}  Max DD (R): {res['max_drawdown_R']:.2f}")
        return 0
    except Exception as exc:
        print(f"ANALYZE: FAIL -> {exc}")
        return 1

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="midnight")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("db-init")
    s.set_defaults(func=cmd_db_init)

    s = sub.add_parser("db-test")
    s.set_defaults(func=cmd_db_test)

    s = sub.add_parser("ingest")
    s.add_argument("--universe", default="data/universe.csv")
    s.add_argument("--start", default=os.getenv("INGEST_START_DATE"))
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("generate")
    s.add_argument("--universe", default="data/universe.csv")
    s.add_argument("--strategy", choices=["breakout"], default=None)
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("backtest")
    s.add_argument("--universe", default="data/universe.csv")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--output", default=str(Path(".tmp/backtest.csv")))
    s.add_argument("--strategy", choices=["breakout","alt"], default="alt")
    s.add_argument("--equity", type=float, default=10_000.0)
    s.add_argument("--sizing", choices=["risk","alloc"], default="risk")
    s.add_argument("--risk", type=float, default=0.005, help="risk per trade (fraction of equity), if sizing=risk")
    s.add_argument("--alloc", type=float, default=0.02, help="allocation per position as fraction of equity, if sizing=alloc")
    s.add_argument("--max_open_r", type=float, default=3.0, help="max open risk cap in R units (sizing=risk)")
    s.add_argument("--max_positions", type=int, default=None, help="max simultaneous open positions (optional)")
    s.add_argument("--cost_bps", type=float, default=0.0005, help="per-trade cost in decimal (5 bps = 0.0005)")
    s.add_argument("--sector_limit", type=int, default=None, help="max positions per sector")
    s.add_argument("--max_symbol_alloc", type=float, default=0.10, help="max allocation per symbol (fraction of equity)")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("discord-test")
    s.set_defaults(func=cmd_discord_test)

    s = sub.add_parser("smoke-test")
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("migrate-to-supabase", help="Copy data from source DB to Supabase")
    s.add_argument("--source", help="Source SQLAlchemy URL (default env DB_URL or sqlite)")
    s.add_argument("--target", help="Target Supabase SQLAlchemy URL (default env SUPABASE_DB_URL)")
    s.set_defaults(func=cmd_migrate_to_supabase)

    s = sub.add_parser("analyze-backtest", help="Generate charts from backtest CSV")
    s.add_argument("--csv", required=True, help="Path to backtest CSV (e.g., .tmp/backtest_10y.csv)")
    s.add_argument("--outdir", default=str(Path(".tmp/charts")), help="Output directory for charts")
    s.set_defaults(func=cmd_analyze_backtest)

    return p


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()


