"""
Market summary API router for Bharat Market AI.

Exposes market-level aggregate data (NIFTY, SENSEX, advance/decline, FII/DII
flows, VIX) with a computed market-regime label derived from the latest
intraday snapshot stored in the database.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.market import MarketSummaryRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])


# ---------------------------------------------------------------------------
# Regime helper
# ---------------------------------------------------------------------------

def _compute_regime(nifty_change: float, advance_decline: float) -> str:
    """Compute the market regime label from key indicators.

    Rules:
    - **bullish**  – NIFTY day-change > +0.5 % **and** A/D ratio > 1.2
    - **bearish**  – NIFTY day-change < -0.5 % **and** A/D ratio < 0.8
    - **neutral**  – everything else

    Args:
        nifty_change: NIFTY 50 percentage change for the day.
        advance_decline: Advance/Decline ratio (advancing stocks / declining).

    Returns:
        str: One of ``"bullish"``, ``"bearish"``, or ``"neutral"``.
    """
    if nifty_change > 0.5 and advance_decline > 1.2:
        return "bullish"
    if nifty_change < -0.5 and advance_decline < 0.8:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get(
    "/summary",
    response_model=MarketSummaryRead,
    summary="Get Latest Market Summary",
    description=(
        "Returns the latest market snapshot including NIFTY/SENSEX levels, "
        "advance/decline ratio, FII/DII flows, India VIX, and a computed "
        "market regime label."
    ),
    responses={
        503: {"description": "Database unavailable or no market data found."},
    },
)
async def get_market_summary(
    db: AsyncSession = Depends(get_db),
) -> MarketSummaryRead:
    """Retrieve the latest market summary record from the database.

    Queries the ``market_snapshots`` table for the most recent row and derives
    the market regime from NIFTY change and advance/decline ratio.

    Args:
        db: Injected async database session.

    Returns:
        MarketSummaryRead: Populated market summary schema.

    Raises:
        HTTPException 503: If the database is unreachable or no data exists.
    """
    query = text(
        """
        SELECT
            id,
            snapshot_date,
            nifty_open,
            nifty_high,
            nifty_low,
            nifty_close,
            nifty_change,
            nifty_change_pct,
            sensex_open,
            sensex_high,
            sensex_low,
            sensex_close,
            sensex_change,
            sensex_change_pct,
            advances,
            declines,
            unchanged,
            total_traded_volume,
            total_traded_value,
            fii_net_flow,
            dii_net_flow,
            india_vix,
            created_at
        FROM market_snapshots
        ORDER BY snapshot_date DESC, created_at DESC
        LIMIT 1
        """
    )

    try:
        result = await db.execute(query)
        row = result.mappings().first()
    except (OperationalError, ProgrammingError) as exc:
        logger.error("DB error fetching market summary: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Market data is temporarily unavailable. Please try again later.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error fetching market summary: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="An unexpected error occurred while fetching market data.",
        ) from exc

    if row is None:
        logger.warning("No market snapshot data found in the database.")
        raise HTTPException(
            status_code=503,
            detail="No market data available yet. The data pipeline may not have run.",
        )

    row_dict: Dict[str, Any] = dict(row)

    # Compute advance/decline ratio safely
    advances: int = row_dict.get("advances") or 0
    declines: int = row_dict.get("declines") or 1  # avoid division by zero
    ad_ratio: float = advances / max(declines, 1)

    nifty_change_pct: float = float(row_dict.get("nifty_change_pct") or 0.0)
    regime: str = _compute_regime(nifty_change_pct, ad_ratio)

    return MarketSummaryRead(
        **row_dict,
        advance_decline_ratio=round(ad_ratio, 4),
        market_regime=regime,
    )
