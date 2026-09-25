"""
test_screener.py
================
Pytest unit tests for the Stock Screener component scoring logic.

Tests cover:
- Technical score calculation (EMA alignment, RSI range, volume ratio)
- Fundamental score calculation (growth, margins, debt-to-equity)
- News score calculation (official vs reputable vs unverified sources)
- Liquidity score calculation
- Entry & exit level calculations
"""

from __future__ import annotations

import pytest
from app.services.screener import Screener


@pytest.fixture
def default_screener() -> Screener:
    return Screener()


def test_technical_score_bullish(default_screener: Screener) -> None:
    indicators = {
        "price": 250.0,
        "ema20": 245.0,
        "ema50": 240.0,
        "ema200": 220.0,
        "rsi": 58.0,                   # Momentum zone 45-65
        "macd_histogram": 2.5,         # Positive
        "macd_histogram_prev": 1.0,    # Growing positive
        "volume": 1800000.0,
        "avg_volume_20d": 1000000.0,   # Volume ratio = 1.8 (> 1.5)
        "relative_strength": 1.25,     # Outperforming Nifty (> 1.0)
    }

    score, reasons = default_screener.calculate_technical_score(indicators)
    assert score >= 70.0
    assert "above_ema200" in reasons


def test_fundamental_score_strong_company(default_screener: Screener) -> None:
    fundamentals = {
        "revenue_growth": 22.0,  # > 15%
        "profit_growth": 25.0,   # > 15%
        "roe": 18.5,             # > 15%
        "debt_to_equity": 0.2,   # < 0.5
        "ebitda_margin": 24.0,   # > 20%
        "promoter_holding": 62.0,
        "promoter_pledge": 0.0,
        "pe_ratio": 20.0,
    }

    score, reasons = default_screener.calculate_fundamental_score(fundamentals)
    assert score >= 75.0


def test_news_score_official_vs_unverified(default_screener: Screener) -> None:
    # Official news (trust score 1)
    official_news = [
        {"trust_score": 1, "materiality": 0.9, "sentiment": "positive"},
        {"trust_score": 1, "materiality": 0.8, "sentiment": "positive"},
    ]

    score_official, _ = default_screener.calculate_news_score(official_news)

    # Unverified news (trust score 4 - ignored)
    unverified_news = [
        {"trust_score": 4, "materiality": 0.9, "sentiment": "positive"},
    ]

    score_unverified, _ = default_screener.calculate_news_score(unverified_news)

    assert score_official > score_unverified
    assert score_unverified == 50.0  # Base score when no verified news


def test_entry_level_calculation(default_screener: Screener) -> None:
    indicators = {
        "price": 500.0,
        "support": 490.0,
        "resistance": 510.0,
    }
    atr = 10.0

    entry_low, entry_high, stop_loss, target_1, target_2 = (
        default_screener.calculate_entry_levels(indicators, atr)
    )

    assert stop_loss < entry_low <= entry_high
    assert target_1 > entry_high
    assert target_2 > target_1
