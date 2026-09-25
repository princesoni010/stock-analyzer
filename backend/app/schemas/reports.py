"""
Pydantic schemas for AI report and market theme API responses.

Covers:
- Full AI morning report reads (AIReportRead).
- Request body for generating / fetching a morning report (MorningReportRequest).
- Market theme reads (ThemeRead).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Regex pattern for ISO date strings (YYYY-MM-DD).
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class AIReportRead(BaseModel):
    """Schema for a complete AI-generated morning market report.

    Attributes:
        report_id: Unique identifier for this report.
        report_date: The trading date this report covers, in 'YYYY-MM-DD' format.
        executive_summary: High-level AI-generated summary of the session outlook.
        market_section: Detailed AI narrative on market conditions and regime.
        theme_section: AI narrative on active sectoral / thematic opportunities.
        stock_sections: List of per-stock dict blocks (symbol, narrative, levels).
        warnings: List of risk warnings or caution flags raised by the AI.
        source_ids: List of data source identifiers used to generate this report.
        uncertainty_notes: List of notes describing areas of model uncertainty.
        generated_at: UTC datetime when the report was generated.
        is_fallback: True if this is a cached / fallback report due to a
            generation error, False if freshly generated.
    """

    report_id: str = Field(
        ...,
        description="Unique identifier for this report.",
        examples=["rpt_20260926"],
    )
    report_date: str = Field(
        ...,
        description="Trading date this report covers (ISO 8601: 'YYYY-MM-DD').",
        examples=["2026-09-26"],
    )
    executive_summary: str = Field(
        ...,
        description="High-level AI-generated summary of the session outlook.",
        examples=["Markets are in a bullish regime with strong breadth; bias remains long."],
    )
    market_section: str = Field(
        ...,
        description="Detailed AI narrative on market conditions and current regime.",
    )
    theme_section: str = Field(
        ...,
        description="AI narrative on active sectoral and thematic opportunities.",
    )
    stock_sections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of per-stock analysis blocks, each containing at minimum: "
            "symbol, narrative, entry_price, stop_loss, target_1, target_2."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Risk warnings or caution flags raised by the AI model.",
        examples=[["High VIX reading; reduce position sizes.", "FII selling pressure detected."]],
    )
    source_ids: List[str] = Field(
        default_factory=list,
        description="Identifiers of data sources used to generate this report.",
        examples=[["nse_eod_20260926", "gdelt_news_20260926"]],
    )
    uncertainty_notes: List[str] = Field(
        default_factory=list,
        description="Notes describing areas where the AI model has low confidence.",
        examples=[["Earnings data for Q2 not yet released for 3 stocks."]],
    )
    generated_at: datetime = Field(
        ...,
        description="UTC datetime when this report was generated.",
    )
    is_fallback: bool = Field(
        ...,
        description=(
            "True when this is a cached fallback report served due to a "
            "generation error; False when freshly generated."
        ),
        examples=[False],
    )

    @field_validator("report_date", mode="before")
    @classmethod
    def _validate_report_date(cls, v: str) -> str:
        """Ensure report_date follows 'YYYY-MM-DD' format."""
        if not isinstance(v, str) or not _ISO_DATE_RE.match(v):
            raise ValueError(
                f"report_date must be a string in 'YYYY-MM-DD' format, got: {v!r}"
            )
        return v

    model_config = {"from_attributes": True}


class MorningReportRequest(BaseModel):
    """Request body for fetching or regenerating the AI morning report.

    Attributes:
        report_date: Target trading date in 'YYYY-MM-DD' format.
            Defaults to today's date when ``None``.
        force_regenerate: When ``True``, bypass any cached report and force
            a fresh AI generation. Defaults to ``False``.
    """

    report_date: Optional[str] = Field(
        None,
        description=(
            "Target trading date in 'YYYY-MM-DD' format. "
            "Omit or pass null to default to today's date."
        ),
        examples=["2026-09-26"],
    )
    force_regenerate: bool = Field(
        False,
        description=(
            "When true, bypass any cached report and regenerate fresh AI content. "
            "Defaults to false."
        ),
        examples=[False],
    )

    @field_validator("report_date", mode="before")
    @classmethod
    def _validate_report_date(cls, v: Optional[str]) -> Optional[str]:
        """Validate 'YYYY-MM-DD' format when report_date is provided."""
        if v is None:
            return None
        if not isinstance(v, str) or not _ISO_DATE_RE.match(v):
            raise ValueError(
                f"report_date must be a string in 'YYYY-MM-DD' format, got: {v!r}"
            )
        return v


class ThemeRead(BaseModel):
    """Schema for a single market theme or sector rotation opportunity.

    Attributes:
        theme_id: Unique identifier for this theme.
        name: Short descriptive name of the theme (e.g. 'EV & Battery Storage').
        description: Longer AI-generated description of the theme, its drivers,
            and investment rationale.
        score: AI confidence / strength score for this theme (0–100).
        status: Lifecycle status of the theme
            ('EMERGING', 'ACTIVE', 'FADING', or 'INACTIVE').
        top_stocks: Ordered list of ticker symbols most aligned with the theme.
        updated_at: UTC datetime when this theme record was last updated.
    """

    theme_id: str = Field(
        ...,
        description="Unique identifier for this theme.",
        examples=["thm_ev_battery"],
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Short descriptive name of the theme.",
        examples=["EV & Battery Storage"],
    )
    description: str = Field(
        ...,
        description="AI-generated description of the theme, its drivers, and rationale.",
        examples=[
            "Government PLI incentives and rising EV penetration are driving strong "
            "revenue growth across battery manufacturers and component suppliers."
        ],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="AI confidence / strength score for this theme (0–100).",
        examples=[84.5],
    )
    status: str = Field(
        ...,
        description="Lifecycle status: 'EMERGING', 'ACTIVE', 'FADING', or 'INACTIVE'.",
        examples=["ACTIVE"],
    )
    top_stocks: List[str] = Field(
        default_factory=list,
        description="Ordered list of ticker symbols most aligned with this theme.",
        examples=[["TATAMOTORS", "EXIDEIND", "AMARA_RAJA"]],
    )
    updated_at: datetime = Field(
        ...,
        description="UTC datetime when this theme record was last updated.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        """Ensure status is one of the allowed lifecycle values."""
        allowed = {"EMERGING", "ACTIVE", "FADING", "INACTIVE"}
        normalised = v.strip().upper()
        if normalised not in allowed:
            raise ValueError(
                f"status must be one of {sorted(allowed)}, got: {v!r}"
            )
        return normalised

    @field_validator("top_stocks", mode="before")
    @classmethod
    def _normalise_top_stocks(cls, v: List[str]) -> List[str]:
        """Uppercase and strip whitespace from every ticker in top_stocks."""
        return [ticker.strip().upper() for ticker in v if ticker.strip()]

    model_config = {"from_attributes": True}
