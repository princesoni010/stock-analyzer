"""
telegram_api.py — API Router for Telegram Test Delivery
======================================================
Endpoints:
- POST /telegram/test — Send test message in development environment
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.services.telegram import TelegramService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])


class TelegramTestRequest(BaseModel):
    """Payload for test message sending."""

    chat_id: Optional[str] = Field(None, description="Telegram chat ID (defaults to TELEGRAM_CHAT_ID)")
    message: str = Field(..., description="Message text to send")


class TelegramTestResponse(BaseModel):
    """Response for test message sending."""

    success: bool
    chat_id: str
    message_ids: list[int]


@router.post(
    "/test",
    response_model=TelegramTestResponse,
    summary="Send Test Telegram Message",
    description="Sends a test message via Telegram Bot API (Development mode only).",
)
async def send_test_telegram(payload: TelegramTestRequest) -> Dict[str, Any]:
    """Send a test notification over Telegram."""
    if settings.APP_ENV == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Test telegram endpoint disabled in production.",
        )

    target_chat_id = payload.chat_id or settings.TELEGRAM_CHAT_ID
    if not target_chat_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TELEGRAM_CHAT_ID is not configured in settings or request.",
        )

    if not settings.TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TELEGRAM_BOT_TOKEN is not configured.",
        )

    tg_service = TelegramService(
        token=settings.TELEGRAM_BOT_TOKEN,
        allowed_chat_ids=[target_chat_id],
    )

    try:
        msg_ids = await tg_service.send_message(
            chat_id=target_chat_id,
            text=payload.message,
            parse_mode="MarkdownV2",
        )
        return {
            "success": True,
            "chat_id": target_chat_id,
            "message_ids": msg_ids,
        }
    except Exception as exc:
        logger.error("Failed to send test Telegram message: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Telegram send failure: {str(exc)}",
        )
