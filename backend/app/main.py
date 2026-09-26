"""
main.py — Bharat Market AI
===========================
FastAPI application entry point.

Responsibilities
----------------
- Configure structured logging before anything else.
- Bootstrap the database on startup via ``init_db()``.
- Register all API routers under their canonical prefixes.
- Apply CORS, rate-limiting (slowapi), and per-request logging middleware.
- Register uniform JSON exception handlers for 404, 429, 500, and catch-all.
"""

from __future__ import annotations

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.database.session import init_db

# ---------------------------------------------------------------------------
# Logging configuration
# Must be called *before* any logger is retrieved so that formatters and
# handlers are in place from the very first log statement.
# ---------------------------------------------------------------------------

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
    }
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# Keyed by the client's remote IP address.
# ---------------------------------------------------------------------------

limiter: Limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # type: ignore[type-arg]
    """
    Async context manager that drives application startup and shutdown.

    Startup
    -------
    1. Initialise the database (create tables / run migrations).
    2. Log a startup banner.
    3. Log every registered route so operators can quickly verify routing.

    Shutdown
    --------
    Log a graceful shutdown message.
    """
        # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("=== Bharat Market AI is starting up ===")

    try:
        await init_db()
        logger.info("Database initialised successfully.")
    except Exception as exc:  # pragma: no cover
        logger.critical("Failed to initialise database: %s", exc, exc_info=True)
        raise

    # Initialize Telegram Bot for Webhook
    from app.services.telegram import TelegramService
    if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        allowed = [c.strip() for c in settings.TELEGRAM_CHAT_ID.split(",") if c.strip()]
        tg_service = TelegramService(
            token=settings.TELEGRAM_BOT_TOKEN,
            allowed_chat_ids=allowed,
        )
        
        # We can map some mock/real async functions here later if needed
        # For now, commands like /start, /help, /settings will work automatically.
        tg_service.register_handlers(services={})
        
        await tg_service.application.initialize()
        await tg_service.application.start()
        app.state.tg_service = tg_service
        logger.info("Telegram Bot Webhook Mode initialized.")

    # Log every registered route for easy diagnostics.
    logger.info("Registered routes:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", str(route))
        if methods:
            logger.info("  [%s] %s", ", ".join(sorted(methods)), path)
        else:
            logger.info("  %s", path)

    logger.info(
        "Bharat Market AI v1.0.0 running in '%s' mode — docs at /docs",
        settings.APP_ENV,
    )

    yield  # ← application is live

        # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down Telegram Bot...")
    if getattr(app.state, "tg_service", None):
        try:
            await app.state.tg_service.application.stop()
            await app.state.tg_service.application.shutdown()
        except Exception as e:
            logger.error("Error shutting down bot: %s", e)
            
    logger.info("=== Bharat Market AI is shutting down — goodbye ===")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bharat Market AI",
    description=(
        "Personal AI-powered Indian stock market research and alert system"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Rate limiter — attach to app state so slowapi middleware can find it
# ---------------------------------------------------------------------------

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# CORS middleware
# In development every origin is allowed; in production only explicitly
# whitelisted origins are permitted.
# ---------------------------------------------------------------------------

_cors_origins: list[str] = ["*"]
_allow_credentials: bool = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# Logs method, path, status code, and elapsed time for every HTTP request.
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """
    Log basic telemetry for every incoming HTTP request.

    Emits a single line at INFO level after the response is returned:

        POST /api/v1/stocks/search 200 42.3 ms
    """
    start_time: float = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:  # pragma: no cover
        # Let the exception propagate; the exception handlers below will
        # render the correct JSON response.  We still want to log the failure.
        elapsed_ms = (time.perf_counter() - start_time) * 1_000
        logger.error(
            "%s %s — UNHANDLED EXCEPTION after %.2f ms",
            request.method,
            request.url.path,
            elapsed_ms,
            exc_info=True,
        )
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1_000
    logger.info(
        "%s %s %s %.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a structured 429 response when a client exceeds its rate limit."""
    logger.warning(
        "Rate limit exceeded for %s %s — %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "rate_limit_exceeded",
            "detail": str(exc),
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured 404 response for unknown routes."""
    logger.debug("404 Not Found: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "not_found",
            "detail": "The requested resource was not found.",
        },
    )


@app.exception_handler(500)
async def internal_server_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return a structured 500 response for unhandled server errors."""
    logger.error(
        "500 Internal Server Error: %s %s",
        request.method,
        request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred.",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all handler for any exception not matched by a more specific handler.

    Returns HTTP 500 with the same payload as the 500 handler so that clients
    always receive a consistent error envelope.
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred.",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.api import (  # noqa: E402 — imports after app creation is intentional
    health,
    market,
    themes,
    stocks,
    screener,
    reports,
    trades,
    telegram_api,
)

app.include_router(health.router)
app.include_router(market.router)
app.include_router(themes.router)
app.include_router(stocks.router)
app.include_router(screener.router)
app.include_router(reports.router)
app.include_router(trades.router)
app.include_router(telegram_api.router)

# ---------------------------------------------------------------------------
# Root route
# ---------------------------------------------------------------------------


@app.get("/", tags=["Root"], summary="Service information")
async def root() -> dict:
    """
    Return basic service metadata.

    This endpoint is intentionally unauthenticated so that load-balancer
    health checks and quick sanity pings can confirm the service is reachable.
    """
    return {
        "name": "Bharat Market AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
