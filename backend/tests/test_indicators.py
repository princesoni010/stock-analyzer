"""
test_indicators.py
==================
Pytest unit tests for technical indicator calculations in IndicatorService.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from app.services.indicators import IndicatorService


def test_ema_calculation() -> None:
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    ema = IndicatorService.calculate_ema(prices, period=5)
    assert pd.isna(ema.iloc[3])
    assert not pd.isna(ema.iloc[4])
    assert round(float(ema.iloc[-1]), 2) > 17.0


def test_rsi_calculation() -> None:
    # 20 bars of price data
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(30)))
    rsi = IndicatorService.calculate_rsi(prices, period=14)
    assert 0 <= float(rsi.iloc[-1]) <= 100


def test_macd_calculation() -> None:
    prices = pd.Series(np.linspace(100.0, 150.0, 40))
    macd, signal, hist = IndicatorService.calculate_macd(prices, fast=12, slow=26, signal=9)

    assert len(macd) == 40
    assert len(signal) == 40
    assert len(hist) == 40
    assert not pd.isna(macd.iloc[-1])
    assert macd.iloc[-1] > 0  # Uptrend produces positive MACD


def test_atr_calculation() -> None:
    high = pd.Series([105.0] * 20)
    low = pd.Series([95.0] * 20)
    close = pd.Series([100.0] * 20)

    atr = IndicatorService.calculate_atr(high, low, close, period=14)
    assert not pd.isna(atr.iloc[-1])
    assert abs(float(atr.iloc[-1]) - 10.0) < 1.0


def test_gap_pct() -> None:
    gap = IndicatorService.calculate_gap_pct(prev_close=100.0, open_price=105.0)
    assert gap == 5.0

    gap_down = IndicatorService.calculate_gap_pct(prev_close=100.0, open_price=95.0)
    assert gap_down == -5.0


def test_relative_strength() -> None:
    stock = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
    index = pd.Series([1000.0, 1010.0, 1020.0, 1030.0, 1040.0])

    rs = IndicatorService.calculate_relative_strength(stock, index, period=4)
    assert rs > 1.0  # Stock outperforms index


def test_find_support_resistance() -> None:
    highs = pd.Series([110.0, 112.0, 108.0, 115.0, 114.0])
    lows = pd.Series([98.0, 95.0, 99.0, 97.0, 96.0])

    support, resistance = IndicatorService.find_support_resistance(highs, lows, lookback=5)
    assert support == 95.0
    assert resistance == 115.0


def test_calculate_all() -> None:
    dates = pd.date_range("2026-01-01", periods=250, freq="B")
    df = pd.DataFrame({
        "open": np.linspace(100, 200, 250),
        "high": np.linspace(102, 202, 250),
        "low": np.linspace(98, 198, 250),
        "close": np.linspace(101, 201, 250),
        "volume": [1000000] * 250,
    }, index=dates)

    index_df = pd.DataFrame({
        "close": np.linspace(10000, 20000, 250),
    }, index=dates)

    res = IndicatorService.calculate_all(df, index_df)
    assert "ema_20" in res
    assert "rsi_14" in res
    assert "macd" in res
    assert "atr_14" in res
    assert "support" in res
    assert "resistance" in res
