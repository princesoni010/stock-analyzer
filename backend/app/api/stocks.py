"""
Stocks API router for Bharat Market AI.

Provides a detailed per-symbol view aggregating technical indicators,
fundamental data, recent news, and the latest screening score into a single
StockRead response object.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.stocks import StockRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stocks", tags=["stocks"])


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

async def _fetch_indicators(db: AsyncSession, symbol: str) -> Dict[str, Any]:
    """Fetch the latest technical indicator row for *symbol*.

    Args:
        db: Async database session.
        symbol: NSE/BSE ticker symbol (e.g. ``"RELIANCE"``).

    Returns:
        Dict containing indicator columns, or an empty dict if not found.
    """
    query = text(
        """
        SELECT *
        FROM indicators
        WHERE symbol = :symbol
        ORDER BY date DESC, created_at DESC
        LIMIT 1
        """
    )
    try:
        result = await db.execute(query, {"symbol": symbol.upper()})
        row = result.mappings().first()
        return dict(row) if row else {}
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("indicators table query failed for %s: %s", symbol, exc)
        return {}


async def _fetch_fundamentals(db: AsyncSession, symbol: str) -> Dict[str, Any]:
    """Fetch the fundamental data row for *symbol*.

    Args:
        db: Async database session.
        symbol: NSE/BSE ticker symbol.

    Returns:
        Dict containing fundamental columns, or an empty dict if not found.
    """
    query = text(
        """
        SELECT *
        FROM fundamentals
        WHERE symbol = :symbol
        ORDER BY updated_at DESC
        LIMIT 1
        """
    )
    try:
        result = await db.execute(query, {"symbol": symbol.upper()})
        row = result.mappings().first()
        return dict(row) if row else {}
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("fundamentals table query failed for %s: %s", symbol, exc)
        return {}


async def _fetch_news(
    db: AsyncSession, symbol: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """Fetch recent news articles mentioning *symbol*.

    Args:
        db: Async database session.
        symbol: NSE/BSE ticker symbol.
        limit: Maximum number of articles to return (default 5).

    Returns:
        List of dicts, each representing a news article row.
    """
    query = text(
        """
        SELECT
            id,
            headline,
            source,
            url,
            published_at,
            sentiment_score,
            sentiment_label
        FROM news
        WHERE symbol = :symbol
        ORDER BY published_at DESC
        LIMIT :limit
        """
    )
    try:
        result = await db.execute(query, {"symbol": symbol.upper(), "limit": limit})
        rows = result.mappings().all()
        return [dict(r) for r in rows]
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("news table query failed for %s: %s", symbol, exc)
        return []


async def _fetch_screening_score(db: AsyncSession, symbol: str) -> Dict[str, Any]:
    """Fetch the latest screening result score for *symbol*.

    Args:
        db: Async database session.
        symbol: NSE/BSE ticker symbol.

    Returns:
        Dict with screening score columns, or an empty dict if not found.
    """
    query = text(
        """
        SELECT
            total_score,
            technical_score,
            fundamental_score,
            sentiment_score,
            rank,
            screened_at
        FROM screening_results
        WHERE symbol = :symbol
        ORDER BY screened_at DESC
        LIMIT 1
        """
    )
    try:
        result = await db.execute(query, {"symbol": symbol.upper()})
        row = result.mappings().first()
        return dict(row) if row else {}
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("screening_results query failed for %s: %s", symbol, exc)
        return {}


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get(
    "/{symbol}",
    response_model=StockRead,
    summary="Get Stock Detail",
    description=(
        "Returns a comprehensive view of a single stock including technical "
        "indicators, fundamentals, the five most recent news articles, and the "
        "latest composite screening score."
    ),
    responses={
        404: {"description": "Symbol not found in any data table."},
    },
)
async def get_stock(
    symbol: str,
    db: AsyncSession = Depends(get_db),
) -> StockRead:
    """Return full stock detail for the given *symbol*.

    Aggregates data from four tables: ``indicators``, ``fundamentals``,
    ``news``, and ``screening_results``.  Returns **404** when the symbol is
    absent from both the indicators *and* fundamentals tables (the two primary
    sources of truth for whether a stock is tracked).

    Args:
        symbol: The stock ticker (case-insensitive; normalised to upper case).
        db: Injected async database session.

    Returns:
        StockRead: Fully populated stock detail object.

    Raises:
        HTTPException 404: When the symbol is not found in any tracked table.
    """
    symbol = symbol.upper().strip()

    # Run all queries concurrently via coroutines
    indicators, fundamentals, news, screening = (
        await _fetch_indicators(db, symbol),
        await _fetch_fundamentals(db, symbol),
        await _fetch_news(db, symbol),
        await _fetch_screening_score(db, symbol),
    )

    # Symbol is considered "not found" if neither primary data source has it.
    if not indicators and not fundamentals:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found. It may not be in the tracked universe.",
        )

    # Merge all data sources; indicators take precedence for shared keys.
    merged: Dict[str, Any] = {
        **fundamentals,
        **indicators,
        "symbol": symbol,
        "news": news,
        "screening": screening if screening else None,
    }

    try:
        return StockRead(**merged)
    except Exception as exc:
        logger.error(
            "Failed to construct StockRead for symbol %s: %s", symbol, exc, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error constructing stock response.",
        ) from exc
