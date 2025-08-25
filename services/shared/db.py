"""Database setup utilities."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session


DB_URL = os.getenv("DB_URL", "sqlite:///midnight.db")
engine = create_engine(DB_URL, echo=False, future=True)


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
