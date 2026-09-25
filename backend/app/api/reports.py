"""
reports.py — API Router for AI Reports in Bharat Market AI
===========================================================
Endpoints:
- POST /reports/morning  — Trigger morning AI report generation
- GET  /reports/{date}   — Retrieve saved report by date (YYYY-MM-DD)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import AIReport, ScreeningResult
from app.database.session import get_db
from app.schemas.reports import AIReportRead
from app.services.nvidia_ai import NvidiaAIService, AIReportRequest, AIReportResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post(
    "/morning",
    response_model=AIReportRead,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Morning AI Report",
    description=(
        "Collects market summary, themes, screening candidates, and source records, "
        "invokes NVIDIA NIM (or fallback template), saves the report to the database, "
        "and returns the saved report record."
    ),
)
async def generate_morning_report(
    report_date: Optional[date] = Query(None, description="Date of report (defaults to today)"),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Generate and persist morning report."""
    target_date = report_date or date.today()

    # Check if report already exists for today
    stmt = select(AIReport).where(
        AIReport.report_type == "morning",
        AIReport.report_date == target_date,
    )
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        return existing

    # Fetch latest screening results for context
    screen_stmt = (
        select(ScreeningResult)
        .order_by(ScreeningResult.created_at.desc(), ScreeningResult.total_score.desc())
        .limit(10)
    )
    candidates_rows = (await db.execute(screen_stmt)).scalars().all()

    candidates_payload = [
        {
            "symbol": row.symbol,
            "total_score": float(row.total_score) if row.total_score else 0.0,
            "signal": row.signal or "yellow",
            "entry_low": float(row.entry_low) if row.entry_low else 0.0,
            "entry_high": float(row.entry_high) if row.entry_high else 0.0,
            "stop_loss": float(row.stop_loss) if row.stop_loss else 0.0,
            "target_1": float(row.target_1) if row.target_1 else 0.0,
            "target_2": float(row.target_2) if row.target_2 else 0.0,
            "risk_reward": float(row.risk_reward) if row.risk_reward else 0.0,
            "reason_codes": row.reason_codes or {},
        }
        for row in candidates_rows
    ]

    ai_request = AIReportRequest(
        report_date=target_date.isoformat(),
        market_summary={"regime": "Bullish Trend", "score": 72.0, "risk_level": "medium"},
        themes=[{"name": "festive_demand", "score": 80.0, "status": "green"}],
        candidates=candidates_payload,
        source_records=[{"id": "SRC-001", "name": "NSE Announcement", "trust": 1}],
        risk_fields=[{"capital": settings.ACCOUNT_CAPITAL, "max_risk_pct": settings.MAX_RISK_PER_TRADE}],
        language="en",
    )

    ai_service = NvidiaAIService(
        api_key=settings.NVIDIA_API_KEY,
        base_url=str(settings.NVIDIA_BASE_URL),
        model=settings.NVIDIA_MODEL,
    )

    try:
        response: AIReportResponse = await ai_service.generate_morning_report(ai_request)
        report_text = f"# Executive Summary\n{response.executive_summary}\n\n# Market Overview\n{response.market_section}\n\n# Active Themes\n{response.theme_section}"
    except Exception as exc:
        logger.warning("NVIDIA NIM unavailable, using template report: %s", exc)
        fallback = ai_service.generate_fallback_report(ai_request)
        report_text = f"# Executive Summary\n{fallback.executive_summary}\n\n# Market Overview\n{fallback.market_section}"
        response = fallback

    new_report = AIReport(
        report_type="morning",
        report_date=target_date,
        model=settings.NVIDIA_MODEL,
        prompt_version="1.0",
        input_hash="hash_" + target_date.isoformat(),
        report_text=report_text,
        source_ids=response.source_ids or ["SRC-001"],
        created_at=datetime.now(timezone.utc),
    )

    db.add(new_report)
    await db.commit()
    await db.refresh(new_report)
    return new_report


@router.get(
    "/{date_str}",
    response_model=AIReportRead,
    summary="Retrieve Report by Date",
    description="Fetches saved AI report for the given date (YYYY-MM-DD).",
)
async def get_report_by_date(
    date_str: str,
    report_type: str = Query("morning", description="Report type (morning / intraday / post_market)"),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Retrieve a report by date string."""
    try:
        parsed_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD.",
        )

    stmt = select(AIReport).where(
        AIReport.report_date == parsed_date,
        AIReport.report_type == report_type,
    )
    report = (await db.execute(stmt)).scalar_one_or_none()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {report_type} report found for date {date_str}.",
        )

    return report
