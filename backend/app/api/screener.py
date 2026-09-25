"""
Screener API router for Bharat Market AI.

Provides two endpoints:
1. GET /watchlist  – returns the top-50 pre-screened stocks from the database.
2. POST /screen/run – runs the screener on-demand against a requested universe.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.stocks import RunScreenerRequest, ScreeningResultRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["screener"])


# ---------------------------------------------------------------------------
# GET /watchlist
# ---------------------------------------------------------------------------

@router.get(
    "/watchlist",
    response_model=List[ScreeningResultRead],
    summary="Get Watchlist",
    description=(
        "Returns the top 50 stocks from the most recent screening run, "
        "sorted by composite score (highest first)."
    ),
)
async def get_watchlist(
    db: AsyncSession = Depends(get_db),
) -> List[ScreeningResultRead]:
    """Return the top-50 screening results from the latest pipeline run.

    Queries the ``screening_results`` table, ordering by ``total_score DESC``
    and limiting to 50 records.  Returns an empty list when no data is
    available (e.g. before the first pipeline run).

    Args:
        db: Injected async database session.

    Returns:
        List[ScreeningResultRead]: Up to 50 screened stock records.
    """
    query = text(
        """
        SELECT
            id,
            symbol,
            company_name,
            sector,
            total_score,
            technical_score,
            fundamental_score,
            sentiment_score,
            rank,
            signal,
            price,
            change_pct,
            screened_at
        FROM screening_results
        ORDER BY total_score DESC
        LIMIT 50
        """
    )

    try:
        result = await db.execute(query)
        rows = result.mappings().all()
    except (OperationalError, ProgrammingError) as exc:
        logger.warning(
            "screening_results table unavailable (pipeline may not have run): %s", exc
        )
        return []
    except Exception as exc:
        logger.error("Unexpected error fetching watchlist: %s", exc, exc_info=True)
        return []

    results: List[ScreeningResultRead] = []
    for row in rows:
        try:
            results.append(ScreeningResultRead(**dict(row)))
        except Exception as exc:
            logger.warning("Skipping malformed screening row: %s – %s", dict(row), exc)

    return results


# ---------------------------------------------------------------------------
# POST /screen/run
# ---------------------------------------------------------------------------

@router.post(
    "/screen/run",
    response_model=List[ScreeningResultRead],
    summary="Run Screener On-Demand",
    description=(
        "Triggers an on-demand screening run over the requested stock universe. "
        "Returns results sorted by composite score (highest first). "
        "Pass ``universe='nifty50'``, ``'nifty200'``, ``'all'``, or a custom "
        "list of symbols via ``symbols``."
    ),
    responses={
        422: {"description": "Invalid screener request parameters."},
    },
)
async def run_screener(
    request: RunScreenerRequest,
    db: AsyncSession = Depends(get_db),
) -> List[ScreeningResultRead]:
    """Execute the stock screener on-demand.

    Imports and instantiates ``app.services.screener.Screener``, runs it
    against the universe specified in *request*, and returns results ranked
    by composite score.

    Args:
        request: Screener configuration including universe and optional
            filter thresholds.
        db: Injected async database session.

    Returns:
        List[ScreeningResultRead]: Screened and ranked stock list.

    Raises:
        HTTPException 422: For invalid or unsupported request parameters.
        HTTPException 500: For unexpected screener failures.
    """
    # Validate universe early before spinning up the screener.
    valid_universes = {"nifty50", "nifty100", "nifty200", "nifty500", "all", "custom"}
    universe: str = (request.universe or "nifty50").lower().strip()

    if universe not in valid_universes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid universe '{universe}'. "
                f"Choose from: {', '.join(sorted(valid_universes))}."
            ),
        )

    if universe == "custom" and not request.symbols:
        raise HTTPException(
            status_code=422,
            detail="'symbols' must be provided when universe='custom'.",
        )

    try:
        from app.services.screener import Screener  # noqa: PLC0415
    except ImportError as exc:
        logger.error("Could not import Screener service: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Screener service is not available.",
        ) from exc

    try:
        screener = Screener(db=db)
        raw_results: List[Dict[str, Any]] = await screener.run(
            universe=universe,
            symbols=request.symbols or [],
            min_score=request.min_score,
            max_results=request.max_results or 50,
        )
    except ValueError as exc:
        logger.warning("Screener validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Screener run failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="The screener encountered an unexpected error.",
        ) from exc

    results: List[ScreeningResultRead] = []
    for item in raw_results:
        try:
            results.append(ScreeningResultRead(**item))
        except Exception as exc:
            logger.warning("Skipping malformed screener result: %s – %s", item, exc)

    # Sort by total_score descending (screener may already do this, but ensure it)
    results.sort(key=lambda r: r.total_score or 0.0, reverse=True)

    return results
