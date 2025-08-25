"""ORM models for Midnight Trader."""
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Ticker(Base):
    __tablename__ = "tickers"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]]
    sector: Mapped[Optional[str]]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    ohlcv: Mapped[list["OHLCVDaily"]] = relationship(back_populates="ticker")


class OHLCVDaily(Base):
    __tablename__ = "ohlcv_daily"

    symbol: Mapped[str] = mapped_column(ForeignKey("tickers.symbol"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    ticker: Mapped[Ticker] = relationship(back_populates="ohlcv")


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    lookback_days: Mapped[int]
    hold_days: Mapped[int]
    trend_ok: Mapped[bool]
    base_quality: Mapped[float]
    entry: Mapped[float]
    stop: Mapped[float]
    target1: Mapped[float]
    target2: Mapped[float]
    decision: Mapped[str] = mapped_column(String)
    plan_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    features: Mapped["SignalFeatures"] = relationship(back_populates="signal", uselist=False)
    outcome: Mapped[Optional["Outcome"]] = relationship(back_populates="signal", uselist=False)


class SignalFeatures(Base):
    __tablename__ = "signal_features"

    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), primary_key=True)
    sma50: Mapped[float]
    sma200: Mapped[float]
    ema20: Mapped[float]
    ema9: Mapped[float]
    price_gt_sma50: Mapped[bool]
    price_gt_ema20: Mapped[bool]
    ema9_gt_ema20: Mapped[bool]
    ret_1m: Mapped[float]
    ret_3m: Mapped[float]
    vol_3m: Mapped[float]
    atr20_pct: Mapped[float]
    rs_sector: Mapped[float] = mapped_column(default=0.0)
    rs_spy: Mapped[float] = mapped_column(default=0.0)

    signal: Mapped[Signal] = relationship(back_populates="features")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    opened_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    entry: Mapped[Optional[float]]
    stop: Mapped[Optional[float]]
    target1: Mapped[Optional[float]]
    target2: Mapped[Optional[float]]
    size_qty: Mapped[Optional[float]]
    state: Mapped[str] = mapped_column(String, default="planned")
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    exit_price: Mapped[Optional[float]]

    journals: Mapped[list["Journal"]] = relationship(back_populates="position")


class Journal(Base):
    __tablename__ = "journals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[int] = mapped_column(ForeignKey("positions.id"))
    note: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    position: Mapped[Position] = relationship(back_populates="journals")


class Outcome(Base):
    __tablename__ = "outcomes"

    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), primary_key=True)
    label: Mapped[int]
    r_multiple: Mapped[float]

    signal: Mapped[Signal] = relationship(back_populates="outcome")
