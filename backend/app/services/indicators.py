"""
services/indicators.py
Pure deterministic technical indicator calculations for Bharat Market AI.
No external API calls. Uses pandas and numpy only.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndicatorService:
    """
    Stateless service class that exposes deterministic technical indicator
    calculations.  All methods are pure functions and do not mutate their
    inputs.
    """

    # ------------------------------------------------------------------
    # Trend / momentum
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            prices:  Closing prices (or any numeric series).
            period:  Lookback window.

        Returns:
            pd.Series aligned to *prices* index, NaN where insufficient data.
        """
        if len(prices) < period:
            logger.debug("EMA(%d): insufficient data (have %d bars).", period, len(prices))
        return prices.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder's Relative Strength Index (RSI).

        Uses Wilder's smoothing (equivalent to EMA with alpha = 1/period).

        Args:
            prices: Closing price series.
            period: RSI period (default 14).

        Returns:
            RSI values (0–100) aligned to *prices* index.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing = EMA with com = period - 1
        avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(100.0)  # When avg_loss = 0, RSI = 100
        return rsi

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD).

        Args:
            prices: Closing price series.
            fast:   Fast EMA period (default 12).
            slow:   Slow EMA period (default 26).
            signal: Signal EMA period (default 9).

        Returns:
            Tuple of (macd_line, signal_line, histogram) as pd.Series.
        """
        ema_fast = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average True Range (ATR).

        True Range = max(H-L, |H-Cprev|, |L-Cprev|).
        ATR = Wilder-smoothed average of True Range.

        Args:
            high, low, close: OHLC series aligned by index.
            period:           Smoothing period (default 14).

        Returns:
            ATR series aligned to input index.
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        return atr

    # ------------------------------------------------------------------
    # Volume-weighted
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Volume-Weighted Average Price (VWAP).

        Calculated cumulatively from the first row of the series (reset
        per session when the DataFrame slice covers a single day).

        Args:
            high, low, close, volume: Series aligned by index.

        Returns:
            VWAP series.
        """
        typical_price = (high + low + close) / 3.0
        cum_vol = volume.cumsum()
        cum_tp_vol = (typical_price * volume).cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        return vwap

    @staticmethod
    def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> float:
        """
        Current volume relative to *period*-day average volume.

        Args:
            volume: Volume series (latest value is index -1).
            period: Lookback window for average (default 20).

        Returns:
            Ratio as float.  Returns NaN if insufficient data.
        """
        if len(volume) < period + 1:
            logger.debug(
                "volume_ratio: insufficient data (have %d, need %d).",
                len(volume), period + 1,
            )
            return float("nan")
        avg = volume.iloc[-(period + 1):-1].mean()
        if avg == 0:
            return float("nan")
        return float(volume.iloc[-1] / avg)

    # ------------------------------------------------------------------
    # Relative / gap metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_relative_strength(
        stock_close: pd.Series,
        index_close: pd.Series,
        period: int = 14,
    ) -> float:
        """
        Relative strength of the stock vs its benchmark index.

        RS = (stock_close[-1] / stock_close[-period]) /
             (index_close[-1] / index_close[-period])

        A value > 1.0 means the stock outperformed the index.

        Args:
            stock_close:  Stock closing price series.
            index_close:  Index closing price series.
            period:       Lookback (default 14 bars).

        Returns:
            Relative strength as float.
        """
        if len(stock_close) < period or len(index_close) < period:
            logger.debug("relative_strength: insufficient data.")
            return float("nan")

        stock_ret = stock_close.iloc[-1] / stock_close.iloc[-period]
        index_ret = index_close.iloc[-1] / index_close.iloc[-period]

        if index_ret == 0:
            return float("nan")
        return float(stock_ret / index_ret)

    @staticmethod
    def calculate_gap_pct(prev_close: float, open_price: float) -> float:
        """
        Overnight gap percentage.

        Positive → gap up; Negative → gap down.

        Args:
            prev_close:  Previous session's close.
            open_price:  Current session's open.

        Returns:
            Gap as a percentage (e.g. 1.5 means +1.5 %).
        """
        if prev_close == 0:
            return 0.0
        return float((open_price - prev_close) / prev_close * 100.0)

    # ------------------------------------------------------------------
    # Support / Resistance / ORB
    # ------------------------------------------------------------------

    @staticmethod
    def find_support_resistance(
        highs: pd.Series,
        lows: pd.Series,
        lookback: int = 20,
    ) -> tuple[float, float]:
        """
        Identify the most recent support and resistance levels using a
        rolling pivot-point approach over *lookback* bars.

        The resistance is the highest *high* in the lookback window.
        The support is the lowest *low* in the lookback window.

        Args:
            highs:    High price series.
            lows:     Low price series.
            lookback: Number of bars to consider (default 20).

        Returns:
            Tuple of (support, resistance).
        """
        n = min(lookback, len(highs), len(lows))
        if n < 1:
            return float("nan"), float("nan")

        resistance = float(highs.iloc[-n:].max())
        support = float(lows.iloc[-n:].min())
        return support, resistance

    @staticmethod
    def calculate_orb(
        first_candle_high: float,
        first_candle_low: float,
    ) -> tuple[float, float]:
        """
        Opening Range Breakout (ORB) levels.

        The ORB is simply the high and low of the first intraday candle
        (typically the 5-min or 15-min opening candle).

        Args:
            first_candle_high: High of the first candle.
            first_candle_low:  Low of the first candle.

        Returns:
            Tuple of (orb_low, orb_high).
        """
        return float(first_candle_low), float(first_candle_high)

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> dict:
        """
        Calculate all indicators for the latest bar in *df*.

        Args:
            df:       OHLCV DataFrame with columns:
                      ``open``, ``high``, ``low``, ``close``, ``volume``.
                      Index should be datetime.  Latest bar = ``df.iloc[-1]``.
            index_df: Nifty 50 OHLCV DataFrame with at minimum a ``close`` column,
                      same index length as *df* (for relative-strength calc).

        Returns:
            Dict with the following keys (all values for the *latest* bar):

            ema_20, ema_50, ema_200,
            rsi_14,
            macd, macd_signal, macd_hist,
            atr_14,
            vwap,
            avg_volume_20d,
            volume_ratio,
            relative_strength,
            gap_pct,
            support, resistance,
            orb_high, orb_low.
        """
        # Validate required columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Normalise column names to lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_ = df["open"]

        # --- EMA ---
        ema_20_s = IndicatorService.calculate_ema(close, 20)
        ema_50_s = IndicatorService.calculate_ema(close, 50)
        ema_200_s = IndicatorService.calculate_ema(close, 200)

        ema_20 = float(ema_20_s.iloc[-1]) if not ema_20_s.empty else float("nan")
        ema_50 = float(ema_50_s.iloc[-1]) if not ema_50_s.empty else float("nan")
        ema_200 = float(ema_200_s.iloc[-1]) if not ema_200_s.empty else float("nan")

        # --- RSI ---
        rsi_s = IndicatorService.calculate_rsi(close, 14)
        rsi_14 = float(rsi_s.iloc[-1]) if not rsi_s.empty else float("nan")

        # --- MACD ---
        macd_s, signal_s, hist_s = IndicatorService.calculate_macd(close)
        macd_val = float(macd_s.iloc[-1]) if not macd_s.empty else float("nan")
        macd_signal_val = float(signal_s.iloc[-1]) if not signal_s.empty else float("nan")
        macd_hist_val = float(hist_s.iloc[-1]) if not hist_s.empty else float("nan")

        # --- ATR ---
        atr_s = IndicatorService.calculate_atr(high, low, close, 14)
        atr_14 = float(atr_s.iloc[-1]) if not atr_s.empty else float("nan")

        # --- VWAP ---
        vwap_s = IndicatorService.calculate_vwap(high, low, close, volume)
        vwap_val = float(vwap_s.iloc[-1]) if not vwap_s.empty else float("nan")

        # --- Volume ---
        avg_volume_20d = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float("nan")
        volume_ratio = IndicatorService.calculate_volume_ratio(volume, 20)

        # --- Relative Strength ---
        if index_df is not None and not index_df.empty:
            idx_df = index_df.copy()
            idx_df.columns = idx_df.columns.str.lower()
            index_close = idx_df["close"] if "close" in idx_df.columns else pd.Series(dtype=float)
            relative_strength = IndicatorService.calculate_relative_strength(close, index_close, 14)
        else:
            relative_strength = float("nan")

        # --- Gap ---
        if len(close) >= 2 and len(open_) >= 1:
            gap_pct = IndicatorService.calculate_gap_pct(
                prev_close=float(close.iloc[-2]),
                open_price=float(open_.iloc[-1]),
            )
        else:
            gap_pct = 0.0

        # --- Support / Resistance ---
        support, resistance = IndicatorService.find_support_resistance(high, low, lookback=20)

        # --- ORB (first candle of *df*; caller should pass intraday slice) ---
        orb_low, orb_high = IndicatorService.calculate_orb(
            first_candle_high=float(high.iloc[0]),
            first_candle_low=float(low.iloc[0]),
        )

        return {
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "rsi_14": rsi_14,
            "macd": macd_val,
            "macd_signal": macd_signal_val,
            "macd_hist": macd_hist_val,
            "atr_14": atr_14,
            "vwap": vwap_val,
            "avg_volume_20d": avg_volume_20d,
            "volume_ratio": volume_ratio,
            "relative_strength": relative_strength,
            "gap_pct": gap_pct,
            "support": support,
            "resistance": resistance,
            "orb_high": orb_high,
            "orb_low": orb_low,
        }
