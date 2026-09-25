"""
Pydantic schemas for market-related API responses.

Covers market summary data (regime, breadth, index changes) and
individual regime classification reads returned by the market endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class MarketSummaryRead(BaseModel):
    """Schema for a full market summary snapshot returned by the API.

    Attributes:
        regime: Current market regime label (e.g. 'Bull', 'Bear', 'Sideways').
        regime_score: Numeric score that drives regime classification (0–100).
        nifty_change_pct: Percentage change of NIFTY 50 for the session.
        banknifty_change_pct: Percentage change of Bank NIFTY for the session.
        advance_decline_ratio: Ratio of advancing stocks to declining stocks on NSE.
        breadth_pct: Market breadth expressed as a percentage (advancers / total).
        risk_level: Qualitative risk level derived from regime and breadth
            (e.g. 'Low', 'Medium', 'High', 'Very High').
        timestamp: UTC timestamp when the snapshot was captured.
        indices: Mapping of index name → latest price / change data dict.
    """

    regime: str = Field(
        ...,
        description="Current market regime label (e.g. 'Bull', 'Bear', 'Sideways').",
        examples=["Bull"],
    )
    regime_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Numeric regime score in the range [0, 100].",
        examples=[72.5],
    )
    nifty_change_pct: float = Field(
        ...,
        description="Percentage change of NIFTY 50 for the current session.",
        examples=[0.85],
    )
    banknifty_change_pct: float = Field(
        ...,
        description="Percentage change of Bank NIFTY for the current session.",
        examples=[1.23],
    )
    advance_decline_ratio: float = Field(
        ...,
        ge=0.0,
        description="Ratio of advancing stocks to declining stocks on NSE.",
        examples=[2.4],
    )
    breadth_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Market breadth as a percentage (advancers / total traded).",
        examples=[68.3],
    )
    risk_level: str = Field(
        ...,
        description="Qualitative risk level derived from regime and breadth.",
        examples=["Medium"],
    )
    timestamp: datetime = Field(
        ...,
        description="UTC datetime when this market snapshot was captured.",
    )
    indices: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Mapping of index name to a dict containing price, change, "
            "and other metadata for that index."
        ),
        examples=[{"NIFTY 50": {"price": 22150.5, "change_pct": 0.85}}],
    )

    model_config = {"from_attributes": True}


class MarketRegimeRead(BaseModel):
    """Schema for a single market regime classification record.

    Attributes:
        name: Human-readable regime name (e.g. 'Bull', 'Bear', 'Sideways').
        score: Numeric score that places the market in this regime (0–100).
        description: Detailed description of what this regime implies for traders.
    """

    name: str = Field(
        ...,
        description="Human-readable regime name.",
        examples=["Bull"],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Numeric regime score in the range [0, 100].",
        examples=[72.5],
    )
    description: str = Field(
        ...,
        description="Detailed description of what this regime implies for traders.",
        examples=[
            "Strong uptrend with high breadth; favour long positions and momentum plays."
        ],
    )

    model_config = {"from_attributes": True}
