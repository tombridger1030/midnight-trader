"""APScheduler jobs for nightly pipeline."""
from __future__ import annotations

import asyncio
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from discord import Intents, Client

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
            await channel.send(content)
        await client.close()

    client.loop.create_task(send_and_close())
    await client.start(TOKEN)


def job_ingest() -> None:
    ingest_universe(UNIVERSE_PATH)


def job_generate() -> None:
    generate_from_universe(UNIVERSE_PATH)


def job_summary() -> None:
    with get_session() as session:
        count = session.query(Signal).order_by(Signal.created_at.desc()).count()
    asyncio.run(post_message(f"Generated signals: {count}"))


def weekly_uplift() -> None:
    report = generate_weekly_report()
    asyncio.run(post_message(f"Weekly uplift report\n{report.to_string(index=False)}"))


def main() -> None:
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(job_ingest, "cron", hour=1, minute=30)
    scheduler.add_job(job_generate, "cron", hour=2, minute=0)
    scheduler.add_job(job_summary, "cron", hour=2, minute=5)
    scheduler.add_job(weekly_uplift, "cron", day_of_week="sun", hour=23, minute=0)
    scheduler.start()
    print("Scheduler started")
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
