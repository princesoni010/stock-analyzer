"""
Pydantic schemas for paper-trading API requests and responses.

Covers:
- Creating a new paper trade (PaperTradeCreate).
- Reading a single paper trade (PaperTradeRead).
- Aggregate performance metrics for a set of trades (PerformanceMetrics).
- Combined list response with trades + metrics (PaperTradesListRead).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class PaperTradeCreate(BaseModel):
    """Request body for opening a new paper trade.

    Attributes:
        symbol: NSE/BSE ticker symbol to trade (e.g. 'RELIANCE').
        entry: Entry price for the trade.
        stop_loss: Stop-loss price; must be strictly below entry for long trades.
        target_1: First take-profit target; must be strictly above entry.
        target_2: Second take-profit target; must be strictly above target_1.
        quantity: Number of shares / lots for the paper trade.
        screening_result_id: Optional reference to the screening result that
            generated this trade idea.
    """

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="NSE/BSE ticker symbol.",
        examples=["RELIANCE"],
    )
    entry: float = Field(
        ...,
        gt=0.0,
        description="Entry price for the trade.",
        examples=[2450.0],
    )
    stop_loss: float = Field(
        ...,
        gt=0.0,
        description="Stop-loss price (must be strictly less than entry).",
        examples=[2380.0],
    )
    target_1: float = Field(
        ...,
        gt=0.0,
        description="First take-profit target (must be strictly greater than entry).",
        examples=[2560.0],
    )
    target_2: float = Field(
        ...,
        gt=0.0,
        description="Second take-profit target (must be strictly greater than target_1).",
        examples=[2700.0],
    )
    quantity: int = Field(
        ...,
        ge=1,
        description="Number of shares or lots for this paper trade.",
        examples=[10],
    )
    screening_result_id: Optional[str] = Field(
        None,
        description="Optional ID of the screening result that generated this trade idea.",
        examples=["scr_abc123"],
    )

    @model_validator(mode="after")
    def _validate_price_levels(self) -> "PaperTradeCreate":
        """Ensure price levels are logically consistent for a long trade."""
        errors: List[str] = []
        if self.stop_loss >= self.entry:
            errors.append(
                f"stop_loss ({self.stop_loss}) must be strictly less than "
                f"entry ({self.entry})."
            )
        if self.target_1 <= self.entry:
            errors.append(
                f"target_1 ({self.target_1}) must be strictly greater than "
                f"entry ({self.entry})."
            )
        if self.target_2 <= self.target_1:
            errors.append(
                f"target_2 ({self.target_2}) must be strictly greater than "
                f"target_1 ({self.target_1})."
            )
        if errors:
            raise ValueError(" | ".join(errors))
        return self

    @model_validator(mode="after")
    def _normalise_symbol(self) -> "PaperTradeCreate":
        """Uppercase and strip whitespace from ticker symbol."""
        self.symbol = self.symbol.strip().upper()
        return self


class PaperTradeRead(BaseModel):
    """Schema for reading a single paper trade record from the API.

    Attributes:
        trade_id: Unique identifier for the paper trade.
        symbol: NSE/BSE ticker symbol.
        entry: Entry price recorded when the trade was opened.
        stop_loss: Stop-loss price set when the trade was opened.
        target_1: First take-profit target set when the trade was opened.
        target_2: Second take-profit target set when the trade was opened.
        quantity: Number of shares / lots.
        status: Current trade status ('OPEN', 'CLOSED', 'STOPPED').
        outcome: Trade outcome label once closed
            (e.g. 'TARGET_1_HIT', 'TARGET_2_HIT', 'STOPPED_OUT').
        exit_price: Price at which the trade was closed, if closed.
        exit_time: UTC datetime when the trade was closed, if closed.
        pnl: Realised profit-and-loss in INR, if closed.
        risk_reward: Risk/reward ratio for the original trade setup.
        open_time: UTC datetime when the trade was opened.
    """

    trade_id: str = Field(
        ...,
        description="Unique identifier for the paper trade.",
        examples=["trd_xyz789"],
    )
    symbol: str = Field(
        ...,
        description="NSE/BSE ticker symbol.",
        examples=["RELIANCE"],
    )
    entry: float = Field(
        ...,
        gt=0.0,
        description="Entry price recorded when the trade was opened.",
        examples=[2450.0],
    )
    stop_loss: float = Field(
        ...,
        gt=0.0,
        description="Stop-loss price set when the trade was opened.",
        examples=[2380.0],
    )
    target_1: float = Field(
        ...,
        gt=0.0,
        description="First take-profit target.",
        examples=[2560.0],
    )
    target_2: float = Field(
        ...,
        gt=0.0,
        description="Second take-profit target.",
        examples=[2700.0],
    )
    quantity: int = Field(
        ...,
        ge=1,
        description="Number of shares or lots.",
        examples=[10],
    )
    status: str = Field(
        ...,
        description="Current trade status: 'OPEN', 'CLOSED', or 'STOPPED'.",
        examples=["OPEN"],
    )
    outcome: Optional[str] = Field(
        None,
        description=(
            "Trade outcome once closed: 'TARGET_1_HIT', 'TARGET_2_HIT', "
            "or 'STOPPED_OUT'."
        ),
        examples=["TARGET_1_HIT"],
    )
    exit_price: Optional[float] = Field(
        None,
        gt=0.0,
        description="Exit price when the trade was closed.",
        examples=[2560.0],
    )
    exit_time: Optional[datetime] = Field(
        None,
        description="UTC datetime when the trade was closed.",
    )
    pnl: Optional[float] = Field(
        None,
        description="Realised profit-and-loss in INR once the trade is closed.",
        examples=[1100.0],
    )
    risk_reward: float = Field(
        ...,
        ge=0.0,
        description="Risk/reward ratio for the original trade setup.",
        examples=[1.57],
    )
    open_time: datetime = Field(
        ...,
        description="UTC datetime when the trade was opened.",
    )

    model_config = {"from_attributes": True}


class PerformanceMetrics(BaseModel):
    """Aggregate performance metrics computed over a collection of paper trades.

    Attributes:
        num_trades: Total number of closed trades in the sample.
        win_rate: Fraction of trades that hit at least target_1 (0–1).
        hit_t1_rate: Fraction of trades that hit target_1 exactly (0–1).
        hit_t2_rate: Fraction of trades that hit target_2 (0–1).
        stopped_out_rate: Fraction of trades that were stopped out (0–1).
        total_pnl: Sum of realised P&L across all closed trades (INR).
        avg_pnl_per_trade: Mean realised P&L per closed trade (INR).
        avg_rr_realized: Mean realised risk/reward ratio across closed trades.
        max_drawdown: Maximum peak-to-trough cumulative P&L drawdown (INR).
    """

    num_trades: int = Field(
        ...,
        ge=0,
        description="Total number of closed trades in the sample.",
        examples=[50],
    )
    win_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of trades that hit at least target_1 (0–1).",
        examples=[0.62],
    )
    hit_t1_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of trades that hit target_1 (0–1).",
        examples=[0.40],
    )
    hit_t2_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of trades that hit target_2 (0–1).",
        examples=[0.22],
    )
    stopped_out_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of trades that were stopped out (0–1).",
        examples=[0.38],
    )
    total_pnl: float = Field(
        ...,
        description="Sum of realised P&L across all closed trades (INR).",
        examples=[48750.0],
    )
    avg_pnl_per_trade: float = Field(
        ...,
        description="Mean realised P&L per closed trade (INR).",
        examples=[975.0],
    )
    avg_rr_realized: float = Field(
        ...,
        ge=0.0,
        description="Mean realised risk/reward ratio across closed trades.",
        examples=[1.45],
    )
    max_drawdown: float = Field(
        ...,
        description=(
            "Maximum peak-to-trough decline in cumulative P&L (INR). "
            "Negative values indicate a loss period."
        ),
        examples=[-12300.0],
    )

    model_config = {"from_attributes": True}


class PaperTradesListRead(BaseModel):
    """Combined response schema containing a list of paper trades and aggregate metrics.

    Attributes:
        trades: Ordered list of paper trade records.
        metrics: Aggregate performance metrics computed over the ``trades`` list.
    """

    trades: List[PaperTradeRead] = Field(
        default_factory=list,
        description="Ordered list of paper trade records.",
    )
    metrics: PerformanceMetrics = Field(
        ...,
        description="Aggregate performance metrics computed over the trades list.",
    )

    model_config = {"from_attributes": True}
