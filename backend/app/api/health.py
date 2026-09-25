"""
Health check API router for Bharat Market AI.

Exposes a single GET /health endpoint that probes every external dependency
and returns an aggregated status payload suitable for load-balancer checks and
monitoring dashboards.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.session import get_db

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/health", tags=["health"])

# ---------------------------------------------------------------------------
# Response schema (local – no separate schema file needed for health)
# ---------------------------------------------------------------------------

DependencyStatus = Literal["ok", "error"]
OverallStatus = Literal["healthy", "degraded", "unhealthy"]


class DependenciesStatus(BaseModel):
    """Status of individual external dependencies."""

    db: DependencyStatus
    redis: DependencyStatus
    nvidia_nim: DependencyStatus
    telegram: DependencyStatus


class HealthResponse(BaseModel):
    """Full health-check response payload."""

    status: OverallStatus
    version: str
    dependencies: DependenciesStatus
    timestamp: datetime


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

async def _check_db(db: AsyncSession) -> DependencyStatus:
    """Run a trivial SQL query to verify database connectivity.

    Args:
        db: An open AsyncSession injected by FastAPI dependency.

    Returns:
        ``"ok"`` if the query succeeds, ``"error"`` otherwise.
    """
    try:
        await db.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        logger.warning("Health DB check failed: %s", exc)
        return "error"


async def _check_redis() -> DependencyStatus:
    """Ping the Redis server configured in settings.

    Attempts to import ``redis.asyncio`` (redis-py ≥4.2) and execute PING.

    Returns:
        ``"ok"`` if PING succeeds, ``"error"`` otherwise.
    """
    try:
        import redis.asyncio as aioredis  # type: ignore[import]

        client: aioredis.Redis = aioredis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        pong = await client.ping()
        await client.aclose()
        return "ok" if pong else "error"
    except Exception as exc:
        logger.warning("Health Redis check failed: %s", exc)
        return "error"


async def _check_nvidia_nim() -> DependencyStatus:
    """Send a lightweight GET request to the NVIDIA NIM base URL.

    Uses a 5-second timeout to avoid blocking the health endpoint.

    Returns:
        ``"ok"`` if a non-5xx response is received, ``"error"`` otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(settings.NVIDIA_BASE_URL)
            # Any non-5xx response is treated as "reachable"
            if response.status_code < 500:
                return "ok"
            logger.warning(
                "NVIDIA NIM health check returned HTTP %d", response.status_code
            )
            return "error"
    except Exception as exc:
        logger.warning("Health NVIDIA NIM check failed: %s", exc)
        return "error"


def _check_telegram() -> DependencyStatus:
    """Verify that the Telegram bot token is configured.

    This is a synchronous, configuration-only check – it does not make a
    network call.

    Returns:
        ``"ok"`` if the token is non-empty, ``"error"`` otherwise.
    """
    token: Optional[str] = getattr(settings, "TELEGRAM_BOT_TOKEN", None)
    if token and token.strip():
        return "ok"
    logger.warning("TELEGRAM_BOT_TOKEN is not configured.")
    return "error"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=HealthResponse,
    summary="System Health Check",
    description=(
        "Probes database, Redis, NVIDIA NIM, and Telegram connectivity. "
        "Returns ``healthy`` when all dependencies are up, ``degraded`` when "
        "some are down, and ``unhealthy`` when all are down."
    ),
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Aggregate health check for all external dependencies.

    Args:
        db: Injected async database session.

    Returns:
        HealthResponse: Status payload with per-dependency breakdown.
    """
    db_status, redis_status, nim_status = (
        await _check_db(db),
        await _check_redis(),
        await _check_nvidia_nim(),
    )
    telegram_status = _check_telegram()

    statuses: Dict[str, DependencyStatus] = {
        "db": db_status,
        "redis": redis_status,
        "nvidia_nim": nim_status,
        "telegram": telegram_status,
    }

    ok_count = sum(1 for s in statuses.values() if s == "ok")
    total = len(statuses)

    if ok_count == total:
        overall: OverallStatus = "healthy"
    elif ok_count == 0:
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        dependencies=DependenciesStatus(**statuses),
        timestamp=datetime.now(tz=timezone.utc),
    )
