"""
Bharat Market AI – Async Database Connection
=============================================
Provides:
  - Async SQLAlchemy engine configured for asyncpg
  - Session factory (AsyncSessionLocal)
  - FastAPI dependency: get_db()
  - Startup helper:    init_db()

Usage in FastAPI:
    from app.database.connection import get_db, init_db

    @app.on_event("startup")
    async def on_startup():
        await init_db()

    @app.get("/example")
    async def example(db: AsyncSession = Depends(get_db)):
        ...
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database URL Normalisation
# Render / Heroku pass 'postgres://' or 'postgresql://' which fails with asyncpg.
# ---------------------------------------------------------------------------
raw_db_url = settings.DATABASE_URL or ""
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif raw_db_url.startswith("postgresql://") and not raw_db_url.startswith("postgresql+asyncpg://"):
    raw_db_url = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Fallback for empty DATABASE_URL in dev/test
if not raw_db_url:
    raw_db_url = "sqlite+aiosqlite:///:memory:"

engine: AsyncEngine = create_async_engine(
    raw_db_url,
    echo=(settings.APP_ENV != "production"),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session and ensure it is closed after the request.

    Rolls back the transaction automatically if an unhandled exception
    propagates out of the request handler.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Context-manager variant (for use outside FastAPI, e.g. background tasks)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that provides a committed-or-rolled-back session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """Create all tables that do not yet exist in the database."""
    from app.database.models import Base  # noqa: F401

    logger.info("Initialising database schema …")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema ready.")
    except Exception as exc:
        logger.warning("Could not initialise database on startup: %s. App will continue.", exc)


# ---------------------------------------------------------------------------
# Teardown helper (call on application shutdown)
# ---------------------------------------------------------------------------
async def close_db() -> None:
    """Dispose the engine connection pool gracefully."""
    logger.info("Closing database connection pool …")
    await engine.dispose()
    logger.info("Database connection pool closed.")
