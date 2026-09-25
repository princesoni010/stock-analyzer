"""
trades.py — API Router for Paper Trading in Bharat Market AI
============================================================
Endpoints:
- POST /paper-trades — Create a paper trade signal/entry
- GET  /paper-trades — List paper trades and calculate performance metrics
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import PaperTrade
from app.database.session import get_db
from app.schemas.trades import PaperTradeCreate, PaperTradeRead, PaperTradeUpdate
from app.services.backtesting import BacktestingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper-trades", tags=["paper-trades"])


@router.post(
    "",
    response_model=PaperTradeRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Paper Trade",
    description="Logs a new paper trade signal with entry, stop loss, targets, and quantity.",
)
async def create_paper_trade(
    payload: PaperTradeCreate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a new paper trade entry."""
    trade = PaperTrade(
        symbol=payload.symbol.upper(),
        signal_time=payload.signal_time,
        entry=payload.entry,
        stop_loss=payload.stop_loss,
        target_1=payload.target_1,
        target_2=payload.target_2,
        quantity=payload.quantity,
        status="open",
        outcome="open",
        screening_result_id=payload.screening_result_id,
    )

    db.add(trade)
    await db.commit()
    await db.refresh(trade)

    logger.info("Paper trade created for %s (qty=%d)", trade.symbol, trade.quantity)
    return trade


@router.get(
    "",
    summary="List Paper Trades and Metrics",
    description="Retrieves paper trades list and overall backtest performance metrics.",
)
async def list_paper_trades(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status: open / closed"),
    symbol: Optional[str] = Query(None, description="Filter by stock symbol"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieve list of paper trades and overall performance metrics."""
    stmt = select(PaperTrade).order_by(PaperTrade.signal_time.desc())

    if status_filter:
        stmt = stmt.where(PaperTrade.status == status_filter)
    if symbol:
        stmt = stmt.where(PaperTrade.symbol == symbol.upper())

    stmt = stmt.limit(limit)
    trades = (await db.execute(stmt)).scalars().all()

    trade_dicts = []
    for t in trades:
        trade_dicts.append({
            "trade_id": str(t.id),
            "symbol": t.symbol,
            "signal_time": t.signal_time,
            "entry": float(t.entry),
            "stop_loss": float(t.stop_loss),
            "target_1": float(t.target_1),
            "target_2": float(t.target_2) if t.target_2 else None,
            "quantity": t.quantity,
            "exit_price": float(t.exit_price) if t.exit_price else None,
            "exit_time": t.exit_time,
            "pnl": float(t.pnl) if t.pnl is not None else 0.0,
            "outcome": t.outcome,
            "status": t.status,
        })

    bt_service = BacktestingService()
    metrics = bt_service.get_metrics(trade_dicts)

    return {
        "trades": trades,
        "total_count": len(trades),
        "metrics": metrics,
    }
