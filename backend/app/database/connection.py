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
# Engine
# ---------------------------------------------------------------------------
# pool_pre_ping ensures stale connections are detected and refreshed before use.
# echo is enabled in non-production environments for SQL tracing.
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=not settings.is_production,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,           # recycle connections every hour
    connect_args={
        "server_settings": {
            "application_name": "bharat-market-ai",
            "timezone": "UTC",
        }
    },
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
    """Create all tables that do not yet exist in the database.

    This is intentionally NOT a migration tool – it is a safe no-op if the
    schema is already up to date.  Use Alembic for schema migrations.
    """
    # Import here to avoid circular imports; models must be registered before
    # `metadata.create_all` is called.
    from app.database.models import Base  # noqa: F401  (side-effect import)

    logger.info("Initialising database schema …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema ready.")


# ---------------------------------------------------------------------------
# Teardown helper (call on application shutdown)
# ---------------------------------------------------------------------------
async def close_db() -> None:
    """Dispose the engine connection pool gracefully."""
    logger.info("Closing database connection pool …")
    await engine.dispose()
    logger.info("Database connection pool closed.")
