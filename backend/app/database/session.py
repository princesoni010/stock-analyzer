"""
Async SQLAlchemy session setup for Bharat Market AI.

Provides:
- Async engine creation from settings.DATABASE_URL
- AsyncSession factory
- get_db() FastAPI dependency for yielding a session per request
- init_db() to create all tables on startup
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


# ---------------------------------------------------------------------------
# Base ORM class
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

def _build_engine() -> AsyncEngine:
    """Create and return the async SQLAlchemy engine.

    Reads DATABASE_URL from settings and configures pool parameters suitable
    for an async PostgreSQL workload. Falls back gracefully if the URL is
    already prefixed with the asyncpg scheme.

    Returns:
        AsyncEngine: Configured async engine instance.
    """
    db_url: str = settings.DATABASE_URL

    # Ensure we are using the asyncpg driver for PostgreSQL.
    # Handles both "postgresql://" and "postgresql+asyncpg://" inputs.
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(
        db_url,
        echo=settings.DEBUG,           # log SQL in DEBUG mode
        pool_pre_ping=True,            # verify connections before use
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,             # recycle stale connections every hour
        future=True,
    )
    logger.info("Async database engine created for host indicated by DATABASE_URL.")
    return engine


engine: AsyncEngine = _build_engine()

# Session factory – do NOT use directly; use get_db() instead.
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    Usage::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...

    The session is automatically closed (returned to pool) after the request,
    even if an exception occurs.

    Yields:
        AsyncSession: An open, ready-to-use database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Table initialisation helper
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create all database tables defined by ORM models.

    Imports all model modules so that their table metadata is registered on
    ``Base.metadata`` before ``create_all`` is called.  Safe to call multiple
    times; SQLAlchemy uses ``IF NOT EXISTS`` semantics.

    This function is typically called from the FastAPI ``lifespan`` context
    manager on application startup.

    Raises:
        Exception: Re-raises any SQLAlchemy or connectivity exception so the
            caller can decide whether startup should abort.
    """
    # Import models here to ensure they are registered on Base.metadata before
    # create_all is invoked.  Add new model modules here as they are created.
    try:
        import app.models  # noqa: F401 – registers all ORM models
    except ImportError:
        logger.warning(
            "app.models could not be imported; tables may not be created. "
            "Ensure all model files exist and are importable."
        )

    logger.info("Running init_db(): creating all tables if they do not exist…")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("init_db() completed successfully.")
    except Exception as exc:
        logger.error("init_db() failed: %s", exc, exc_info=True)
        raise
