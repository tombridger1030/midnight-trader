# Midnight Trader

Discord-only trading assistant for 12–16 week position swings in stocks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill DISCORD_BOT_TOKEN
```

`data/universe.csv` ships with a default list of liquid US large caps.
Edit this file to customise the universe.

## Run

Start APScheduler jobs:

```bash
python -m services.jobs.schedule
```

Run the Discord bot:

```bash
python -m services.bot.main
```

## Backtest

```bash
python -m services.signals.backtest --start 2019-01-01 --end 2024-12-31
```

## Environment

- Python 3.11
- SQLite by default, override with `DB_URL`.

## Disclaimer

No broker integration; all fills and exits are manual.
