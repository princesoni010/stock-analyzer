"""
test_risk_engine.py
===================
Pytest unit tests for deterministic Risk Engine calculations.

Tests cover:
- Position sizing calculation based on account capital & risk %
- Stop loss validation (rejecting invalid stop loss >= entry)
- Risk-reward threshold checks (rejecting R:R < minimum)
- Position capping using max position value
- Sizing with costs and slippage estimates included
- Exclusion of invalid setups
"""

from __future__ import annotations

import pytest
from app.services.risk_engine import RiskEngine, RiskOutput


@pytest.fixture
def default_risk_engine() -> RiskEngine:
    return RiskEngine(
        capital=100000.0,
        max_risk_pct=0.01,         # 1% risk = 1000 INR
        max_position_value=20000.0, # max 20,000 INR per trade
        min_risk_reward=1.5,
        slippage_pct=0.001,
        transaction_cost_pct=0.0005,
    )


def test_valid_trade_calculation(default_risk_engine: RiskEngine) -> None:
    # Entry zone 100-102, stop loss 95, target 115
    # Risk per share = 102 - 95 = 7 INR
    # Risk amount = 1000 INR
    # Qty by risk = floor(1000 / 7) = 142
    # Qty by max position = floor(20000 / 102) = 196
    # Final qty = min(142, 196) = 142
    result: RiskOutput = default_risk_engine.calculate(
        symbol="RELIANCE",
        entry_low=100.0,
        entry_high=102.0,
        stop_loss=95.0,
        target_1=115.0,
        target_2=125.0,
        atr=3.5,
    )

    assert result.status == "trade"
    assert result.quantity == 142
    assert result.risk_reward >= 1.5
    assert result.invalidation == "Close below 95.00"


def test_reject_invalid_stop_loss(default_risk_engine: RiskEngine) -> None:
    # Stop loss (105) >= entry_low (100) -> invalid
    result: RiskOutput = default_risk_engine.calculate(
        symbol="TCS",
        entry_low=100.0,
        entry_high=102.0,
        stop_loss=105.0,
        target_1=120.0,
        target_2=None,
        atr=2.0,
    )

    assert result.status == "no_trade"
    assert any("stop loss" in w.lower() or "invalid" in w.lower() for w in result.warnings)


def test_reject_poor_risk_reward(default_risk_engine: RiskEngine) -> None:
    # Entry 100, Stop 90 (risk 10), Target 105 (reward 5) -> R:R = 0.5 (< min 1.5)
    result: RiskOutput = default_risk_engine.calculate(
        symbol="INFY",
        entry_low=100.0,
        entry_high=100.0,
        stop_loss=90.0,
        target_1=105.0,
        target_2=None,
        atr=2.0,
    )

    assert result.status == "no_trade"
    assert any("risk_reward" in w.lower() or "risk" in w.lower() for w in result.warnings)


def test_cap_quantity_by_max_position_value(default_risk_engine: RiskEngine) -> None:
    # High price stock: 5000 INR
    # Risk per share: 5000 - 4900 = 100 INR
    # Risk qty = 1000 / 100 = 10
    # Max position value = 20000 / 5000 = 4 shares
    result: RiskOutput = default_risk_engine.calculate(
        symbol="TCS",
        entry_low=5000.0,
        entry_high=5000.0,
        stop_loss=4900.0,
        target_1=5300.0,
        target_2=5500.0,
        atr=50.0,
    )

    assert result.status == "trade"
    assert result.quantity == 4  # Capped at 4 instead of 10


def test_zero_quantity_rejection(default_risk_engine: RiskEngine) -> None:
    # Price 25000 > max position value 20000 -> quantity 0 -> no_trade
    result: RiskOutput = default_risk_engine.calculate(
        symbol="MRF",
        entry_low=25000.0,
        entry_high=25000.0,
        stop_loss=24000.0,
        target_1=28000.0,
        target_2=None,
        atr=200.0,
    )

    assert result.status == "no_trade"
    assert any("quantity" in w.lower() or "position" in w.lower() for w in result.warnings)
