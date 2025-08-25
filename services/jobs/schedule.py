"""APScheduler jobs for nightly pipeline."""
from __future__ import annotations

import asyncio
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from datetime import datetime, timedelta, timezone
from discord import Intents, Client, AllowedMentions

from ..signals.ingest_prices import ingest_universe
from ..signals.generate import generate_from_universe
from ..signals.uplift_report import generate_weekly_report
from ..shared.db import get_session
from ..shared.models import Signal

UNIVERSE_PATH = "data/universe.csv"
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
TOKEN = os.getenv("DISCORD_BOT_TOKEN")


async def post_message(content: str) -> None:
    if not TOKEN or not CHANNEL_ID:
        print("Discord channel not configured")
        return
    intents = Intents.none()
    client = Client(intents=intents)

    async def send_and_close() -> None:
        await client.wait_until_ready()
        channel = client.get_channel(CHANNEL_ID)
        if channel:
            await channel.send(content, allowed_mentions=AllowedMentions(everyone=True))
        await client.close()

    client.loop.create_task(send_and_close())
    await client.start(TOKEN)


def job_ingest() -> None:
    ingest_universe(UNIVERSE_PATH)


def job_generate() -> None:
    generate_from_universe(UNIVERSE_PATH)
    notify_recent_signals()


def job_summary() -> None:
    with get_session() as session:
        count = session.query(Signal).order_by(Signal.created_at.desc()).count()
    asyncio.run(post_message(f"Generated signals: {count}"))


def weekly_uplift() -> None:
    report = generate_weekly_report()
    asyncio.run(post_message(f"Weekly uplift report\n{report.to_string(index=False)}"))


def weekly_run() -> None:
    """Run weekly pipeline and post summary."""
    ingest_universe(UNIVERSE_PATH)
    generate_from_universe(UNIVERSE_PATH)
    with get_session() as session:
        count = session.query(Signal).order_by(Signal.created_at.desc()).count()
    asyncio.run(post_message(f"Weekly run complete. New signals: {count}"))
    notify_recent_signals()


def notify_recent_signals(window_minutes: int = 720) -> None:
    """Ping @everyone with any signals created in the recent window (default 12h)."""
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    with get_session() as session:
        sigs = (
            session.query(Signal)
            .filter(Signal.created_at >= since)
            .order_by(Signal.created_at.desc())
            .all()
        )
    if not sigs:
        return
    syms = [s.symbol for s in sigs]
    preview = ", ".join(syms[:12])
    more = f" (+{len(syms)-12} more)" if len(syms) > 12 else ""
    asyncio.run(post_message(f"@everyone New signals: {preview}{more}"))


def main() -> None:
    pacific = pytz.timezone("US/Pacific")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    scheduler = AsyncIOScheduler(timezone=pacific, event_loop=loop)
    # Nightly pipeline (Pacific times)
    scheduler.add_job(job_ingest, CronTrigger(day_of_week="mon-fri", hour=1, minute=30, timezone=pacific))
    scheduler.add_job(job_generate, CronTrigger(day_of_week="mon-fri", hour=2, minute=0, timezone=pacific))
    scheduler.add_job(job_summary, CronTrigger(day_of_week="mon-fri", hour=2, minute=5, timezone=pacific))
    # Weekly run every Monday at 8:15am Pacific
    scheduler.add_job(weekly_run, CronTrigger(day_of_week="mon", hour=8, minute=15, timezone=pacific))
    # Weekly uplift (Sunday 11:00pm Pacific)
    scheduler.add_job(weekly_uplift, CronTrigger(day_of_week="sun", hour=23, minute=0, timezone=pacific))
    scheduler.start()
    print("Scheduler started")
    loop.run_forever()


if __name__ == "__main__":
    main()
