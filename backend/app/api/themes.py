"""
Themes API router for Bharat Market AI.

Returns the current list of thematic sector scores ranked by their composite
score, allowing the frontend to render a heat-map or ranked list of market
themes (e.g. EV, Defence, Renewable Energy).
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.reports import ThemeRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/themes", tags=["themes"])


@router.get(
    "",
    response_model=List[ThemeRead],
    summary="List Market Themes",
    description=(
        "Returns all market themes ordered by their composite score "
        "(highest first). Returns an empty list when no data is available."
    ),
)
async def list_themes(
    db: AsyncSession = Depends(get_db),
) -> List[ThemeRead]:
    """Retrieve all market theme scores from the database.

    Queries the ``theme_scores`` table and returns rows sorted by
    ``score DESC``.  An empty list is returned rather than an error when the
    table is empty or doesn't exist yet, to support a graceful startup
    experience before the first pipeline run.

    Args:
        db: Injected async database session.

    Returns:
        List[ThemeRead]: Ranked list of theme score records.
    """
    query = text(
        """
        SELECT
            id,
            theme_name,
            score,
            momentum_score,
            fundamental_score,
            news_sentiment_score,
            stock_count,
            top_stocks,
            description,
            created_at,
            updated_at
        FROM theme_scores
        ORDER BY score DESC
        """
    )

    try:
        result = await db.execute(query)
        rows = result.mappings().all()
    except (OperationalError, ProgrammingError) as exc:
        # Table may not exist on first boot – return empty list gracefully.
        logger.warning(
            "theme_scores table unavailable (pipeline may not have run yet): %s", exc
        )
        return []
    except Exception as exc:
        logger.error("Unexpected error fetching themes: %s", exc, exc_info=True)
        return []

    if not rows:
        logger.info("No theme score records found.")
        return []

    themes: List[ThemeRead] = []
    for row in rows:
        try:
            themes.append(ThemeRead(**dict(row)))
        except Exception as exc:
            logger.warning("Could not deserialize theme row %s: %s", dict(row), exc)

    return themes
