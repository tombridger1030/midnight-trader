"""Database setup utilities.

Environment variables:
- DB_URL or SUPABASE_DB_URL: SQLAlchemy URL. For Supabase Postgres, include sslmode=require
  or it will be enforced automatically.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session


RAW_DB_URL = os.getenv("DB_URL") or os.getenv("SUPABASE_DB_URL") or "sqlite:///midnight.db"


def _enforce_ssl_in_url(db_url: str) -> str:
    """Ensure sslmode=require for Postgres URLs.

    For non-Postgres URLs, return unchanged.
    """
    if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
        if "sslmode=" not in db_url:
            sep = "&" if "?" in db_url else "?"
            return f"{db_url}{sep}sslmode=require"
    return db_url


DB_URL = _enforce_ssl_in_url(RAW_DB_URL)

engine = create_engine(
    DB_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)


class Base(DeclarativeBase):
    """Base class for ORM models."""


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables based on ORM models."""
    # Import locally to avoid circular imports at module import time
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def test_connection() -> bool:
    """Attempt a simple connection and SELECT 1. Returns True on success."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
