"""Load OHLCV data via yfinance and store in database."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf

from ..shared.db import get_session
from ..shared.models import OHLCVDaily, Ticker
from ..shared.utils import load_universe

START_DATE = os.getenv("INGEST_START_DATE", "2015-01-01")
UNIVERSE_CSV = Path(os.getenv("UNIVERSE_CSV", "data/universe.csv"))


def download_symbol(symbol: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """Download adjusted daily bars for a symbol."""
    df = yf.download(symbol, start=start_date or START_DATE, progress=False, auto_adjust=True)
    df = df.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high",
                                          "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_symbols(path: Path | None = None) -> List[str]:
    """Load symbols from CSV path or environment fallback."""
    csv_path = path or UNIVERSE_CSV
    if csv_path.exists():
        symbols = load_universe(csv_path)
    else:
        env_syms = os.getenv("UNIVERSE_SYMBOLS", "")
        symbols = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
    print(f"Loaded {len(symbols)} symbols")
    return symbols


def ingest_universe(universe_path: str | Path | None = None, symbols_override: Optional[List[str]] = None, start_date: Optional[str] = None) -> None:
    symbols = symbols_override or load_symbols(Path(universe_path) if universe_path else None)
    with get_session() as session:
        for sym in symbols:
            session.merge(Ticker(symbol=sym, is_active=True))
            df = download_symbol(sym, start_date=start_date)
            df = df[["date", "open", "high", "low", "close", "volume"]]
            for date, open_, high, low, close, volume in df.itertuples(index=False, name=None):
                session.merge(OHLCVDaily(
                    symbol=sym,
                    date=date,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                ))


def fetch_symbol_df(symbol: str) -> pd.DataFrame:
    """Fetch OHLCV data for a symbol from the database."""
    with get_session() as session:
        rows = session.query(OHLCVDaily).filter_by(symbol=symbol).order_by(OHLCVDaily.date).all()
    return pd.DataFrame([
        {"date": r.date, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
        for r in rows
    ])


if __name__ == "__main__":
    ingest_universe(Path("data/universe.csv"))
