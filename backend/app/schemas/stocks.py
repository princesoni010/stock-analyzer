"""
Pydantic schemas for stock screening and stock detail API responses.

Covers:
- Full per-stock analysis reads (StockRead).
- Lightweight screening result rows (ScreeningResultRead).
- Request body for triggering an on-demand screener run (RunScreenerRequest).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class StockRead(BaseModel):
    """Schema for a comprehensive single-stock analysis record.

    Attributes:
        symbol: NSE/BSE ticker symbol (e.g. 'RELIANCE', 'TCS').
        company_name: Full registered company name.
        sector: Sector the company belongs to (e.g. 'IT', 'Banking').
        total_score: Composite AI score combining all sub-scores (0–100).
        technical_score: Score derived from technical indicator analysis (0–100).
        fundamental_score: Score derived from fundamental / financial data (0–100).
        news_score: Score derived from recent news sentiment analysis (0–100).
        indicators: Dict of technical indicator name → value / signal pairs.
        fundamentals: Dict of fundamental metric name → value pairs.
        recent_news: List of recent news article dicts (title, url, sentiment, date).
        entry_price: Suggested entry price, if available.
        stop_loss: Suggested stop-loss price, if available.
        target_1: First price target, if available.
        target_2: Second (stretch) price target, if available.
        timestamp: UTC datetime when this analysis was computed.
    """

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="NSE/BSE ticker symbol.",
        examples=["RELIANCE"],
    )
    company_name: str = Field(
        ...,
        description="Full registered company name.",
        examples=["Reliance Industries Limited"],
    )
    sector: str = Field(
        ...,
        description="Sector the company belongs to.",
        examples=["Energy"],
    )
    total_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite AI score (0–100).",
        examples=[78.4],
    )
    technical_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Technical analysis sub-score (0–100).",
        examples=[80.0],
    )
    fundamental_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Fundamental analysis sub-score (0–100).",
        examples=[75.5],
    )
    news_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="News sentiment sub-score (0–100).",
        examples=[70.0],
    )
    indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of technical indicator name to value/signal.",
        examples=[{"RSI": 58.3, "MACD_signal": "bullish"}],
    )
    fundamentals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of fundamental metric name to value.",
        examples=[{"PE_ratio": 22.1, "ROE": 18.5, "debt_equity": 0.4}],
    )
    recent_news: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of recent news items with title, url, sentiment, and date.",
    )
    entry_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Suggested entry price level.",
        examples=[2450.0],
    )
    stop_loss: Optional[float] = Field(
        None,
        gt=0.0,
        description="Suggested stop-loss price level.",
        examples=[2380.0],
    )
    target_1: Optional[float] = Field(
        None,
        gt=0.0,
        description="First price target.",
        examples=[2560.0],
    )
    target_2: Optional[float] = Field(
        None,
        gt=0.0,
        description="Second (stretch) price target.",
        examples=[2700.0],
    )
    timestamp: datetime = Field(
        ...,
        description="UTC datetime when this analysis was computed.",
    )

    model_config = {"from_attributes": True}


class ScreeningResultRead(BaseModel):
    """Schema for a lightweight row returned in screener result lists.

    Attributes:
        symbol: NSE/BSE ticker symbol.
        company_name: Full registered company name.
        total_score: Composite AI score (0–100).
        technical_score: Technical analysis sub-score (0–100).
        fundamental_score: Fundamental analysis sub-score (0–100).
        news_score: News sentiment sub-score (0–100).
        entry_price: Suggested entry price, if available.
        stop_loss: Suggested stop-loss price, if available.
        target_1: First price target, if available.
        target_2: Second price target, if available.
        rationale: One-line AI-generated rationale for selecting this stock.
        risk_reward: Computed risk/reward ratio for the trade setup, if available.
        screened_at: UTC datetime when the screener evaluated this stock.
    """

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="NSE/BSE ticker symbol.",
        examples=["TCS"],
    )
    company_name: str = Field(
        ...,
        description="Full registered company name.",
        examples=["Tata Consultancy Services Limited"],
    )
    total_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite AI score (0–100).",
        examples=[82.1],
    )
    technical_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Technical analysis sub-score (0–100).",
        examples=[85.0],
    )
    fundamental_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Fundamental analysis sub-score (0–100).",
        examples=[80.0],
    )
    news_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="News sentiment sub-score (0–100).",
        examples=[75.0],
    )
    entry_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Suggested entry price level.",
        examples=[3600.0],
    )
    stop_loss: Optional[float] = Field(
        None,
        gt=0.0,
        description="Suggested stop-loss price level.",
        examples=[3500.0],
    )
    target_1: Optional[float] = Field(
        None,
        gt=0.0,
        description="First price target.",
        examples=[3800.0],
    )
    target_2: Optional[float] = Field(
        None,
        gt=0.0,
        description="Second (stretch) price target.",
        examples=[4000.0],
    )
    rationale: str = Field(
        ...,
        description="AI-generated one-line rationale for selecting this stock.",
        examples=["Strong momentum with bullish MACD crossover and healthy ROE of 42%."],
    )
    risk_reward: Optional[float] = Field(
        None,
        ge=0.0,
        description="Risk/reward ratio for the trade setup (reward / risk).",
        examples=[2.0],
    )
    screened_at: datetime = Field(
        ...,
        description="UTC datetime when the screener evaluated this stock.",
    )

    model_config = {"from_attributes": True}


class RunScreenerRequest(BaseModel):
    """Request body for triggering an on-demand screener run.

    Attributes:
        universe: Optional list of ticker symbols to restrict the screener to.
            When ``None`` the screener runs over the full configured universe.
        min_score: Minimum composite score threshold; stocks below this are
            excluded from results. Defaults to 60.0.
        max_results: Maximum number of results to return. Defaults to 20.
    """

    universe: Optional[List[str]] = Field(
        None,
        description=(
            "Optional list of NSE/BSE ticker symbols to screen. "
            "Pass null / omit to use the full configured universe."
        ),
        examples=[["RELIANCE", "TCS", "INFY"]],
    )
    min_score: float = Field(
        60.0,
        ge=0.0,
        le=100.0,
        description="Minimum composite score for a stock to appear in results.",
        examples=[60.0],
    )
    max_results: int = Field(
        20,
        ge=1,
        le=200,
        description="Maximum number of screened stocks to return.",
        examples=[20],
    )

    @field_validator("universe", mode="before")
    @classmethod
    def _normalise_universe(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Uppercase and strip whitespace from every ticker in the list."""
        if v is None:
            return None
        cleaned = [ticker.strip().upper() for ticker in v if ticker.strip()]
        return cleaned if cleaned else None
