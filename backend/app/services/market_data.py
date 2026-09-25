"""
services/market_data.py
Market data collector for Bharat Market AI.
Primary source: yfinance (wrapped in asyncio thread-pool for non-blocking I/O).
"""

import asyncio
import logging
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .normalizer import is_stale

logger = logging.getLogger(__name__)

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Thread-pool for yfinance (synchronous library)
# ---------------------------------------------------------------------------
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yfinance-worker")

# Retry / rate-limit config
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0   # seconds
_REQUEST_SLEEP = 0.1  # seconds between successive downloads

# Staleness threshold defaults
_INTRADAY_STALE_MINUTES = 15
_DAILY_STALE_MINUTES = 1440  # 24 h


# ---------------------------------------------------------------------------
# Helper: run synchronous yfinance call in executor
# ---------------------------------------------------------------------------

async def _run_in_executor(func, *args, **kwargs):
    """Execute a synchronous callable in the shared thread-pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _EXECUTOR, lambda: func(*args, **kwargs)
    )


# ---------------------------------------------------------------------------
# Helper: retry with exponential back-off
# ---------------------------------------------------------------------------

async def _fetch_with_retry(func, *args, retries: int = _MAX_RETRIES, **kwargs):
    """
    Call *func* up to *retries* times with exponential back-off.
    Returns the result of the first successful call.
    Raises the last exception if all attempts fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return await _run_in_executor(func, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            sleep_time = _BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "yfinance attempt %d/%d failed: %s. Retrying in %.1fs …",
                attempt + 1, retries, exc, sleep_time,
            )
            await asyncio.sleep(sleep_time)
    raise RuntimeError(
        f"All {retries} yfinance attempts failed."
    ) from last_exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nse_ticker(symbol: str) -> str:
    """Return the yfinance ticker string for an NSE symbol or index."""
    s = symbol.upper()
    if s.startswith("^") or "." in s:
        return s
    return f"{s}.NS"


def _df_to_candles(ticker_symbol: str, df, timeframe: str, source: str = "yfinance") -> list[dict]:
    """
    Convert a yfinance OHLCV DataFrame to a list of candle dicts.
    The DataFrame index is the timestamp (timezone-aware or naive).
    """
    candles: list[dict] = []
    if df is None or df.empty:
        return candles

    # Flatten multi-level columns if present (yfinance v0.2+)
    if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "levels"):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            pass

    # Normalise column names to lowercase
    df.columns = [str(c).lower().split()[0] for c in df.columns]

    # Drop rows with NaN price data (e.g. non-trading session placeholders)
    if "close" in df.columns:
        df = df.dropna(subset=["open", "high", "low", "close"])

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        logger.error(
            "yfinance DataFrame for %s missing columns. Got: %s",
            ticker_symbol, list(df.columns),
        )
        return candles

    for ts_idx, row in df.iterrows():
        # Convert index to UTC datetime
        if hasattr(ts_idx, "tzinfo") and ts_idx.tzinfo is not None:
            utc_ts = ts_idx.to_pydatetime().astimezone(_UTC)
        else:
            utc_ts = ts_idx.to_pydatetime().replace(tzinfo=_UTC)

        c_open = float(row["open"]) if pd.notna(row["open"]) else 0.0
        c_high = float(row["high"]) if pd.notna(row["high"]) else 0.0
        c_low = float(row["low"]) if pd.notna(row["low"]) else 0.0
        c_close = float(row["close"]) if pd.notna(row["close"]) else 0.0
        c_vol = int(row["volume"]) if pd.notna(row["volume"]) else 0

        candle: dict = {
            "symbol": ticker_symbol,
            "timeframe": timeframe,
            "open": c_open,
            "high": c_high,
            "low": c_low,
            "close": c_close,
            "volume": c_vol,
            "timestamp": utc_ts,
            "source": source,
            "is_stale": False,  # will be checked by caller if needed
        }
        candles.append(candle)

    return candles


# ---------------------------------------------------------------------------
# MarketDataService
# ---------------------------------------------------------------------------

class MarketDataService:
    """
    Async market data service backed by yfinance.

    All public methods are coroutines and safe to call concurrently.
    """

    # ------------------------------------------------------------------
    # Daily candles
    # ------------------------------------------------------------------

    async def get_daily_candles(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> list[dict]:
        """
        Fetch daily OHLCV candles for *symbols* between *start* and *end*.

        Args:
            symbols:    List of NSE symbol strings (e.g. ``["RELIANCE", "TCS"]``).
            start:      Inclusive start date.
            end:        Inclusive end date.

        Returns:
            List of candle dicts with keys:
            symbol, timeframe, open, high, low, close, volume,
            timestamp (UTC datetime), source, is_stale.
        """
        all_candles: list[dict] = []
        for symbol in symbols:
            ticker_str = _nse_ticker(symbol)
            try:
                start_str = start.isoformat() if start else None
                end_str = end.isoformat() if end else None
                df = await _fetch_with_retry(
                    self._download_daily,
                    ticker_str,
                    start=start_str,
                    end=end_str,
                )
                candles = _df_to_candles(symbol, df, timeframe="1d")
                # Mark stale if the latest candle is older than threshold
                if candles:
                    latest_ts: datetime = candles[-1]["timestamp"]
                    stale = is_stale(latest_ts, threshold_minutes=_DAILY_STALE_MINUTES)
                    for c in candles:
                        c["is_stale"] = stale
                all_candles.extend(candles)
            except Exception as exc:
                logger.error("Failed to fetch daily candles for %s: %s", symbol, exc)
            finally:
                await asyncio.sleep(_REQUEST_SLEEP)

        return all_candles

    @staticmethod
    def _download_daily(ticker_str: str, start: Optional[str] = None, end: Optional[str] = None):
        """Synchronous yfinance download (runs in thread pool)."""
        ticker = yf.Ticker(ticker_str)
        if start and end:
            df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        else:
            df = ticker.history(period="1y", interval="1d", auto_adjust=True)
        return df

    # ------------------------------------------------------------------
    # Intraday candles
    # ------------------------------------------------------------------

    async def get_intraday_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
    ) -> list[dict]:
        """
        Fetch intraday OHLCV candles for *symbol*.

        yfinance supports intraday intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h.
        Data is limited to the last 60 days for intervals >= 1m.

        Args:
            symbol:    NSE symbol string.
            timeframe: yfinance interval string (default ``'5m'``).

        Returns:
            List of candle dicts (same schema as :meth:`get_daily_candles`).
        """
        ticker_str = _nse_ticker(symbol)
        try:
            df = await _fetch_with_retry(
                self._download_intraday,
                ticker_str,
                timeframe,
            )
        except Exception as exc:
            logger.error("Failed to fetch intraday candles for %s: %s", symbol, exc)
            return []

        candles = _df_to_candles(symbol, df, timeframe=timeframe)
        # Mark stale if the latest candle is older than threshold
        if candles:
            latest_ts: datetime = candles[-1]["timestamp"]
            stale = is_stale(latest_ts, threshold_minutes=_INTRADAY_STALE_MINUTES)
            for c in candles:
                c["is_stale"] = stale

        return candles

    @staticmethod
    def _download_intraday(ticker_str: str, interval: str):
        """Synchronous yfinance intraday download (runs in thread pool)."""
        ticker = yf.Ticker(ticker_str)
        df = ticker.history(period="5d", interval=interval, auto_adjust=True)
        return df

    # ------------------------------------------------------------------
    # Index data
    # ------------------------------------------------------------------

    async def get_index_data(self) -> dict:
        """
        Fetch current values for major Indian indices.

        Returns:
            Dict with keys: nifty50, banknifty, india_vix.
            Each value is a dict: { price, change, change_pct, timestamp, source, is_stale }.
        """
        index_tickers = {
            "nifty50": "^NSEI",
            "banknifty": "^NSEBANK",
            "india_vix": "^INDIAVIX",
        }
        result: dict = {}
        for name, ticker_str in index_tickers.items():
            try:
                data = await _fetch_with_retry(
                    self._fetch_latest_quote, ticker_str
                )
                result[name] = data
            except Exception as exc:
                logger.error("Failed to fetch index data for %s (%s): %s", name, ticker_str, exc)
                result[name] = {
                    "price": None,
                    "change": None,
                    "change_pct": None,
                    "timestamp": None,
                    "source": "yfinance",
                    "is_stale": True,
                    "error": str(exc),
                }
            await asyncio.sleep(_REQUEST_SLEEP)

        return result

    # ------------------------------------------------------------------
    # Global cues
    # ------------------------------------------------------------------

    async def get_global_cues(self) -> dict:
        """
        Fetch global market cues relevant to Indian equity markets.

        Returns:
            Dict with keys: sp500, crude_oil, usd_inr, us_10y_yield.
            Each value is a dict: { price, change, change_pct, timestamp, source, is_stale }.
        """
        global_tickers = {
            "sp500": "^GSPC",
            "crude_oil": "CL=F",
            "usd_inr": "USDINR=X",
            "us_10y_yield": "^TNX",
        }
        result: dict = {}
        for name, ticker_str in global_tickers.items():
            try:
                data = await _fetch_with_retry(
                    self._fetch_latest_quote, ticker_str
                )
                result[name] = data
            except Exception as exc:
                logger.error("Failed to fetch global cue %s (%s): %s", name, ticker_str, exc)
                result[name] = {
                    "price": None,
                    "change": None,
                    "change_pct": None,
                    "timestamp": None,
                    "source": "yfinance",
                    "is_stale": True,
                    "error": str(exc),
                }
            await asyncio.sleep(_REQUEST_SLEEP)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_latest_quote(ticker_str: str) -> dict:
        """
        Fetch the most recent close/current price for *ticker_str*.
        Returns a standardised dict for index / global cue data.
        Runs synchronously inside the thread pool.
        """
        ticker = yf.Ticker(ticker_str)
        # Use 5d/1d to ensure we always get at least one bar
        df = ticker.history(period="5d", interval="1d", auto_adjust=True)

        if df is None or df.empty:
            raise ValueError(f"No data returned for ticker '{ticker_str}'")

        # Normalise column names
        df.columns = [str(c).lower().split()[0] for c in df.columns]

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        price = float(latest["close"])
        prev_close = float(prev["close"])
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0.0

        # Timestamp from index
        ts_idx = df.index[-1]
        if hasattr(ts_idx, "tzinfo") and ts_idx.tzinfo is not None:
            utc_ts = ts_idx.to_pydatetime().astimezone(_UTC)
        else:
            utc_ts = ts_idx.to_pydatetime().replace(tzinfo=_UTC)

        stale = is_stale(utc_ts, threshold_minutes=_DAILY_STALE_MINUTES)

        return {
            "price": price,
            "change": round(change, 4),
            "change_pct": round(change_pct, 4),
            "timestamp": utc_ts,
            "source": "yfinance",
            "is_stale": stale,
        }
