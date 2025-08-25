"""Discord bot for Midnight Trader."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from ..shared.db import get_session
from ..shared.models import Journal, Position, Signal
from ..shared.utils import load_universe
from ..signals.backtest import simulate_universe
from ..signals.features import compute_indicators
from ..signals.ingest_prices import fetch_symbol_df
from .charts import make_chart

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready() -> None:
    await bot.tree.sync()
    print(f"Logged in as {bot.user}")


@bot.tree.command(name="screen", description="Show latest proposed entries")
async def screen(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    with get_session() as session:
        signals = (
            session.query(Signal)
            .order_by(Signal.created_at.desc())
            .limit(10)
            .all()
        )
    for sig in signals:
        df = fetch_symbol_df(sig.symbol)
        df = compute_indicators(df)
        chart_path = make_chart(sig.symbol, df.tail(120))
        flags = sig.plan_json
        badges = "".join([
            f"SMA50{'✓' if flags.get('price_gt_sma50') else '✗'} ",
            f"EMA20{'✓' if flags.get('price_gt_ema20') else '✗'} ",
            f"9>20{'✓' if flags.get('ema9_gt_ema20') else '✗'}",
        ])
        desc = (
            f"entry {sig.entry:.2f}  stop {sig.stop:.2f}  t1 {sig.target1:.2f}  "
            f"time {sig.hold_days}d  manage: half at t1\n{badges}"
        )
        embed = discord.Embed(title=sig.symbol, description=desc)
        file = discord.File(chart_path)
        embed.set_image(url=f"attachment://{chart_path.name}")
        await interaction.followup.send(embed=embed, file=file)


universe_group = app_commands.Group(name="universe", description="Universe commands")


@universe_group.command(name="show", description="Show universe")
async def universe_show(interaction: discord.Interaction) -> None:
    try:
        symbols = load_universe(Path("data/universe.csv"))
    except FileNotFoundError:
        await interaction.response.send_message(
            "Universe file missing. Create data/universe.csv", ephemeral=True
        )
        return
    preview = " ".join(symbols[:30])
    await interaction.response.send_message(
        f"```\n{preview}\n```\nTotal: {len(symbols)}"
    )


bot.tree.add_command(universe_group)


backtest_group = app_commands.Group(name="backtest", description="Backtest commands")


@backtest_group.command(name="run", description="Run backtest")
@app_commands.describe(start="Start date YYYY-MM-DD", end="End date YYYY-MM-DD")
async def backtest_run(interaction: discord.Interaction, start: str, end: str) -> None:
    await interaction.response.defer(thinking=True)
    try:
        symbols = load_universe(Path("data/universe.csv"))
    except FileNotFoundError:
        await interaction.followup.send("Universe file missing. Create data/universe.csv", ephemeral=True)
        return
    trades, summary = simulate_universe(start, end, symbols)
    out_path = Path("/tmp/backtest.csv")
    trades.to_csv(out_path, index=False)
    embed = discord.Embed(title="Backtest Summary")
    embed.add_field(name="Trades", value=str(summary["trades"]))
    embed.add_field(name="Hit rate", value=f"{summary['hit_rate']:.2%}")
    embed.add_field(name="Profit factor", value=f"{summary['profit_factor']:.2f}")
    embed.add_field(name="Expectancy", value=f"{summary['expectancy']:.2f}R")
    embed.add_field(name="Max DD", value=f"{summary['max_drawdown']:.2f}R")
    file = discord.File(out_path)
    await interaction.followup.send(embed=embed, file=file)


bot.tree.add_command(backtest_group)


@bot.tree.command(name="plan", description="Approve a trade plan")
@app_commands.describe(ticker="Ticker to approve")
async def plan(interaction: discord.Interaction, ticker: str) -> None:
    with get_session() as session:
        sig = (
            session.query(Signal)
            .filter_by(symbol=ticker.upper())
            .order_by(Signal.created_at.desc())
            .first()
        )
        if not sig:
            await interaction.response.send_message("No signal found", ephemeral=True)
            return
        pos = Position(symbol=sig.symbol, entry=sig.entry, stop=sig.stop,
                       target1=sig.target1, target2=sig.target2, state="planned")
        session.add(pos)
    await interaction.response.send_message(f"Plan approved for {ticker.upper()}")


@bot.tree.command(name="fill", description="Log a fill")
@app_commands.describe(ticker="Ticker", price="Fill price", qty="Quantity")
async def fill(interaction: discord.Interaction, ticker: str, price: float, qty: float) -> None:
    with get_session() as session:
        pos = (
            session.query(Position)
            .filter_by(symbol=ticker.upper())
            .order_by(Position.id.desc())
            .first()
        )
        if not pos:
            await interaction.response.send_message("No position found", ephemeral=True)
            return
        pos.entry = price
        pos.size_qty = qty
        pos.opened_at = datetime.utcnow()
        pos.state = "open"
    await interaction.response.send_message(f"Filled {ticker.upper()} @ {price}")


@bot.tree.command(name="exit", description="Log an exit")
@app_commands.describe(ticker="Ticker", price="Exit price")
async def exit_trade(interaction: discord.Interaction, ticker: str, price: float) -> None:
    with get_session() as session:
        pos = (
            session.query(Position)
            .filter_by(symbol=ticker.upper(), state="open")
            .order_by(Position.id.desc())
            .first()
        )
        if not pos:
            await interaction.response.send_message("No open position", ephemeral=True)
            return
        pos.exit_price = price
        pos.closed_at = datetime.utcnow()
        pos.state = "closed"
        pnl = (price - pos.entry) * (pos.size_qty or 0)
        session.add(Journal(position_id=pos.id, note=f"Exit PnL: {pnl:.2f}"))
    await interaction.response.send_message(f"Exited {ticker.upper()} @ {price}")


@bot.tree.command(name="journal", description="Add a journal note")
@app_commands.describe(ticker="Ticker", note="Note text")
async def journal(interaction: discord.Interaction, ticker: str, note: str) -> None:
    with get_session() as session:
        pos = (
            session.query(Position)
            .filter_by(symbol=ticker.upper())
            .order_by(Position.id.desc())
            .first()
        )
        if not pos:
            await interaction.response.send_message("No position found", ephemeral=True)
            return
        session.add(Journal(position_id=pos.id, note=note))
    await interaction.response.send_message("Journal added")


@bot.tree.command(name="stats", description="Show performance stats")
async def stats(interaction: discord.Interaction) -> None:
    with get_session() as session:
        positions = session.query(Position).filter_by(state="closed").all()
    if not positions:
        await interaction.response.send_message("No closed positions", ephemeral=True)
        return
    pnl = [((p.exit_price - p.entry) / (p.entry - p.stop)) for p in positions if p.entry and p.stop]
    win_rate = sum(1 for x in pnl if x > 0) / len(pnl)
    profit_factor = sum(x for x in pnl if x > 0) / abs(sum(x for x in pnl if x <= 0)) if any(x <= 0 for x in pnl) else float("inf")
    expectancy = sum(pnl) / len(pnl)
    embed = discord.Embed(title="Stats")
    embed.add_field(name="Win rate", value=f"{win_rate:.2%}")
    embed.add_field(name="Profit factor", value=f"{profit_factor:.2f}")
    embed.add_field(name="Expectancy", value=f"{expectancy:.2f}R")
    await interaction.response.send_message(embed=embed)


def main() -> None:
    bot.run(TOKEN)


if __name__ == "__main__":
    main()
