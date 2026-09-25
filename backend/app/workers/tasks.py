"""
Bharat Market AI — Celery Tasks
================================
All 9 background Celery tasks for data ingestion, analysis, report generation,
and delivery. Tasks use asyncio.run() to bridge sync Celery with async service
methods and SQLAlchemy async DB sessions.

Task list
---------
1.  overnight_news_fetch     — Fetch news articles + NSE announcements
2.  official_filings_fetch   — Fetch NSE/BSE corporate filings
3.  weather_theme_update     — Update monsoon/weather-driven theme scores
4.  market_data_update       — Fetch OHLCV candles + compute indicators
5.  daily_screener           — Run full stock screener, persist results
6.  morning_report           — Generate AI morning report via NVIDIA API
7.  telegram_delivery        — Deliver morning report via Telegram
8.  intraday_scanner         — Real-time alert scanning during market hours
9.  post_market_review       — Update paper trades + compute metrics
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, time as dtime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.workers.celery_app import celery_app

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IST timezone constant
# ---------------------------------------------------------------------------
IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Market hours constants (IST)
# ---------------------------------------------------------------------------
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# ---------------------------------------------------------------------------
# Shared DB session factory helpers
# ---------------------------------------------------------------------------


def _make_session_factory(database_url: str) -> sessionmaker:
    """Create a SQLAlchemy async session factory for the given *database_url*."""
    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        echo=False,
    )
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _get_session() -> AsyncSession:
    """Return a ready-to-use :class:`AsyncSession` using the configured DB URL."""
    factory = _make_session_factory(settings.DATABASE_URL)
    return factory()


# ---------------------------------------------------------------------------
# Helper: build standard return dict
# ---------------------------------------------------------------------------


def _result(
    task_name: str,
    success: bool,
    records: int,
    start: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical task result dictionary."""
    return {
        "task": task_name,
        "success": success,
        "records": records,
        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        "error": error,
    }


# ===========================================================================
# Task 1 — overnight_news_fetch
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.overnight_news_fetch",
    bind=True,
    max_retries=3,
)
def overnight_news_fetch(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Fetch latest news articles from NewsIngestionService and NSE
    announcements from NSEService. Deduplicate by URL/announcement ID and
    persist new records to the database.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of new rows saved.
    """
    task_name = "overnight_news_fetch"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.news_ingestion import NewsIngestionService  # noqa: PLC0415
        from app.services.nse_bse import NSEService  # noqa: PLC0415

        news_service = NewsIngestionService()
        nse_service = NSEService()

        # ---- fetch --------------------------------------------------------
        try:
            articles: List[Dict[str, Any]] = await news_service.fetch_latest_news()
            logger.info("[%s] Fetched %d news articles", task_name, len(articles))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] NewsIngestionService error: %s", task_name, exc)
            articles = []

        try:
            announcements: List[Dict[str, Any]] = await nse_service.fetch_announcements()
            logger.info(
                "[%s] Fetched %d NSE announcements", task_name, len(announcements)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] NSEService error: %s", task_name, exc)
            announcements = []

        # ---- deduplicate --------------------------------------------------
        seen_urls: set[str] = set()
        unique_articles: List[Dict[str, Any]] = []
        for item in articles:
            url = item.get("url") or item.get("link") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                item.setdefault("source_type", "news")
                unique_articles.append(item)

        seen_ids: set[str] = set()
        unique_announcements: List[Dict[str, Any]] = []
        for item in announcements:
            ann_id = str(item.get("id") or item.get("ann_id") or item.get("url") or "")
            if ann_id and ann_id not in seen_ids:
                seen_ids.add(ann_id)
                item.setdefault("source_type", "nse_announcement")
                unique_announcements.append(item)

        all_records = unique_articles + unique_announcements
        if not all_records:
            logger.info("[%s] No new records to save", task_name)
            return 0

        # ---- persist ------------------------------------------------------
        saved = 0
        async with await _get_session() as session:
            async with session.begin():
                for record in all_records:
                    try:
                        await session.execute(
                            text(
                                """
                                INSERT INTO news_items
                                    (url, title, body, source, source_type, published_at, fetched_at)
                                VALUES
                                    (:url, :title, :body, :source, :source_type, :published_at, NOW())
                                ON CONFLICT (url) DO NOTHING
                                """
                            ),
                            {
                                "url": record.get("url") or record.get("link") or "",
                                "title": record.get("title") or "",
                                "body": record.get("body") or record.get("content") or "",
                                "source": record.get("source") or "",
                                "source_type": record.get("source_type") or "unknown",
                                "published_at": record.get("published_at"),
                            },
                        )
                        saved += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[%s] Failed to insert record '%s': %s",
                            task_name,
                            record.get("url"),
                            exc,
                        )
        return saved

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — saved %d records", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=60)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 2 — official_filings_fetch
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.official_filings_fetch",
    bind=True,
    max_retries=3,
)
def official_filings_fetch(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Fetch corporate filings from NSE and BSE. Deduplicate by filing ID and
    persist new records to the ``filings`` table.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of new filings saved.
    """
    task_name = "official_filings_fetch"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.nse_bse import BSEService, NSEService  # noqa: PLC0415

        nse_service = NSEService()
        bse_service = BSEService()

        # ---- fetch --------------------------------------------------------
        try:
            nse_filings: List[Dict[str, Any]] = (
                await nse_service.fetch_corporate_announcements()
            )
            logger.info("[%s] Fetched %d NSE filings", task_name, len(nse_filings))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] NSEService.fetch_corporate_announcements error: %s", task_name, exc)
            nse_filings = []

        try:
            bse_filings: List[Dict[str, Any]] = await bse_service.fetch_filings()
            logger.info("[%s] Fetched %d BSE filings", task_name, len(bse_filings))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] BSEService.fetch_filings error: %s", task_name, exc)
            bse_filings = []

        # ---- deduplicate --------------------------------------------------
        seen_ids: set[str] = set()
        unique_filings: List[Dict[str, Any]] = []

        for exchange, filings in (("NSE", nse_filings), ("BSE", bse_filings)):
            for item in filings:
                filing_id = str(
                    item.get("filing_id")
                    or item.get("id")
                    or item.get("ann_id")
                    or ""
                )
                dedup_key = f"{exchange}:{filing_id}"
                if dedup_key and dedup_key not in seen_ids:
                    seen_ids.add(dedup_key)
                    item["exchange"] = exchange
                    unique_filings.append(item)

        if not unique_filings:
            logger.info("[%s] No new filings to save", task_name)
            return 0

        # ---- persist ------------------------------------------------------
        saved = 0
        async with await _get_session() as session:
            async with session.begin():
                for filing in unique_filings:
                    try:
                        await session.execute(
                            text(
                                """
                                INSERT INTO filings
                                    (filing_id, exchange, symbol, category, headline,
                                     attachment_url, filed_at, fetched_at)
                                VALUES
                                    (:filing_id, :exchange, :symbol, :category, :headline,
                                     :attachment_url, :filed_at, NOW())
                                ON CONFLICT (exchange, filing_id) DO NOTHING
                                """
                            ),
                            {
                                "filing_id": str(
                                    filing.get("filing_id")
                                    or filing.get("id")
                                    or filing.get("ann_id")
                                    or ""
                                ),
                                "exchange": filing.get("exchange") or "",
                                "symbol": filing.get("symbol") or filing.get("scrip") or "",
                                "category": filing.get("category") or "",
                                "headline": filing.get("headline") or filing.get("subject") or "",
                                "attachment_url": filing.get("attachment_url") or filing.get("url") or "",
                                "filed_at": filing.get("filed_at") or filing.get("date"),
                            },
                        )
                        saved += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[%s] Failed to insert filing '%s': %s",
                            task_name,
                            filing.get("filing_id"),
                            exc,
                        )
        return saved

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — saved %d filings", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=60)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 3 — weather_theme_update
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.weather_theme_update",
    bind=True,
    max_retries=3,
)
def weather_theme_update(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Fetch weather and monsoon data from IMDService, run it through
    ThemeEngine to score all sector themes, and persist the scores to DB.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of themes updated.
    """
    task_name = "weather_theme_update"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.imd import IMDService  # noqa: PLC0415
        from app.services.theme_engine import ThemeEngine  # noqa: PLC0415

        imd_service = IMDService()
        theme_engine = ThemeEngine()

        # ---- fetch weather data -------------------------------------------
        try:
            weather_data: Dict[str, Any] = await imd_service.fetch_weather_data()
            logger.info("[%s] Fetched weather data for %d regions", task_name, len(weather_data.get("regions", [])))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] IMDService error: %s", task_name, exc)
            weather_data = {}

        # ---- score themes -------------------------------------------------
        try:
            theme_scores: List[Dict[str, Any]] = await theme_engine.score_all_themes(
                weather_data
            )
            logger.info("[%s] Scored %d themes", task_name, len(theme_scores))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] ThemeEngine error: %s", task_name, exc)
            theme_scores = []

        if not theme_scores:
            return 0

        # ---- persist ------------------------------------------------------
        saved = 0
        scored_on = date.today().isoformat()
        async with await _get_session() as session:
            async with session.begin():
                for theme in theme_scores:
                    try:
                        await session.execute(
                            text(
                                """
                                INSERT INTO theme_scores
                                    (theme_name, score, rationale, scored_on, updated_at)
                                VALUES
                                    (:theme_name, :score, :rationale, :scored_on, NOW())
                                ON CONFLICT (theme_name, scored_on)
                                DO UPDATE SET
                                    score = EXCLUDED.score,
                                    rationale = EXCLUDED.rationale,
                                    updated_at = NOW()
                                """
                            ),
                            {
                                "theme_name": theme.get("theme_name") or theme.get("name") or "",
                                "score": float(theme.get("score") or 0.0),
                                "rationale": theme.get("rationale") or "",
                                "scored_on": scored_on,
                            },
                        )
                        saved += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[%s] Failed to upsert theme '%s': %s",
                            task_name,
                            theme.get("theme_name"),
                            exc,
                        )
        return saved

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — updated %d theme scores", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=120)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 4 — market_data_update
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.market_data_update",
    bind=True,
    max_retries=3,
)
def market_data_update(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Fetch OHLCV candles for all universe stocks, compute technical indicators
    for each stock, and persist candles + indicator rows to the database.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of stocks updated.
    """
    task_name = "market_data_update"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.indicators import IndicatorService  # noqa: PLC0415
        from app.services.market_data import MarketDataService  # noqa: PLC0415

        market_service = MarketDataService()
        indicator_service = IndicatorService()

        # ---- fetch candles ------------------------------------------------
        try:
            universe_candles: Dict[str, List[Dict[str, Any]]] = (
                await market_service.fetch_universe_candles()
            )
            logger.info(
                "[%s] Fetched candles for %d stocks", task_name, len(universe_candles)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] MarketDataService error: %s", task_name, exc)
            return 0

        if not universe_candles:
            logger.warning("[%s] No candles fetched", task_name)
            return 0

        stocks_updated = 0
        async with await _get_session() as session:
            async with session.begin():
                for symbol, candles in universe_candles.items():
                    if not candles:
                        continue
                    try:
                        # ---- compute indicators ---------------------------
                        try:
                            indicators: Dict[str, Any] = (
                                indicator_service.calculate_all(candles)
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "[%s] IndicatorService error for %s: %s",
                                task_name, symbol, exc,
                            )
                            indicators = {}

                        # ---- persist candles ------------------------------
                        for candle in candles:
                            await session.execute(
                                text(
                                    """
                                    INSERT INTO ohlcv_candles
                                        (symbol, ts, open, high, low, close, volume)
                                    VALUES
                                        (:symbol, :ts, :open, :high, :low, :close, :volume)
                                    ON CONFLICT (symbol, ts) DO NOTHING
                                    """
                                ),
                                {
                                    "symbol": symbol,
                                    "ts": candle.get("timestamp") or candle.get("ts"),
                                    "open": float(candle.get("open") or 0),
                                    "high": float(candle.get("high") or 0),
                                    "low": float(candle.get("low") or 0),
                                    "close": float(candle.get("close") or 0),
                                    "volume": int(candle.get("volume") or 0),
                                },
                            )

                        # ---- persist indicators ---------------------------
                        if indicators:
                            await session.execute(
                                text(
                                    """
                                    INSERT INTO stock_indicators
                                        (symbol, calculated_on, rsi, macd, macd_signal,
                                         bb_upper, bb_lower, sma_20, sma_50, ema_9,
                                         atr, adx, indicators_json, updated_at)
                                    VALUES
                                        (:symbol, :calculated_on, :rsi, :macd, :macd_signal,
                                         :bb_upper, :bb_lower, :sma_20, :sma_50, :ema_9,
                                         :atr, :adx, :indicators_json, NOW())
                                    ON CONFLICT (symbol, calculated_on)
                                    DO UPDATE SET
                                        rsi = EXCLUDED.rsi,
                                        macd = EXCLUDED.macd,
                                        macd_signal = EXCLUDED.macd_signal,
                                        bb_upper = EXCLUDED.bb_upper,
                                        bb_lower = EXCLUDED.bb_lower,
                                        sma_20 = EXCLUDED.sma_20,
                                        sma_50 = EXCLUDED.sma_50,
                                        ema_9 = EXCLUDED.ema_9,
                                        atr = EXCLUDED.atr,
                                        adx = EXCLUDED.adx,
                                        indicators_json = EXCLUDED.indicators_json,
                                        updated_at = NOW()
                                    """
                                ),
                                {
                                    "symbol": symbol,
                                    "calculated_on": date.today().isoformat(),
                                    "rsi": indicators.get("rsi"),
                                    "macd": indicators.get("macd"),
                                    "macd_signal": indicators.get("macd_signal"),
                                    "bb_upper": indicators.get("bb_upper"),
                                    "bb_lower": indicators.get("bb_lower"),
                                    "sma_20": indicators.get("sma_20"),
                                    "sma_50": indicators.get("sma_50"),
                                    "ema_9": indicators.get("ema_9"),
                                    "atr": indicators.get("atr"),
                                    "adx": indicators.get("adx"),
                                    "indicators_json": str(indicators),
                                },
                            )

                        stocks_updated += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[%s] Failed to persist data for '%s': %s",
                            task_name, symbol, exc,
                        )

        return stocks_updated

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — updated %d stocks", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=120)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 5 — daily_screener
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.daily_screener",
    bind=True,
    max_retries=3,
)
def daily_screener(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Load latest indicators, fundamentals, and news sentiment from the DB,
    pass to Screener.run_full_screen(), and persist screening results for
    today's date.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of candidates found.
    """
    task_name = "daily_screener"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.screener import Screener  # noqa: PLC0415

        screener = Screener()
        screened_on = date.today().isoformat()

        # ---- load data from DB -------------------------------------------
        async with await _get_session() as session:
            # Latest indicators per symbol
            indicators_rows = (
                await session.execute(
                    text(
                        """
                        SELECT DISTINCT ON (symbol)
                            symbol, rsi, macd, macd_signal,
                            bb_upper, bb_lower, sma_20, sma_50, ema_9,
                            atr, adx, indicators_json, calculated_on
                        FROM stock_indicators
                        ORDER BY symbol, calculated_on DESC
                        """
                    )
                )
            ).mappings().all()

            # Latest fundamentals per symbol
            fundamentals_rows = (
                await session.execute(
                    text(
                        """
                        SELECT DISTINCT ON (symbol)
                            symbol, pe_ratio, pb_ratio, roe, debt_to_equity,
                            promoter_holding, institutional_holding, updated_at
                        FROM stock_fundamentals
                        ORDER BY symbol, updated_at DESC
                        """
                    )
                )
            ).mappings().all()

            # Recent news sentiment per symbol (last 24 h)
            news_rows = (
                await session.execute(
                    text(
                        """
                        SELECT symbol, AVG(sentiment_score) AS avg_sentiment,
                               COUNT(*) AS news_count
                        FROM news_items
                        WHERE symbol IS NOT NULL
                          AND published_at >= NOW() - INTERVAL '24 hours'
                        GROUP BY symbol
                        """
                    )
                )
            ).mappings().all()

        # ---- assemble stocks_data ----------------------------------------
        indicators_map = {r["symbol"]: dict(r) for r in indicators_rows}
        fundamentals_map = {r["symbol"]: dict(r) for r in fundamentals_rows}
        news_map = {r["symbol"]: dict(r) for r in news_rows}

        all_symbols = set(indicators_map) | set(fundamentals_map)
        stocks_data: List[Dict[str, Any]] = []
        for symbol in all_symbols:
            record: Dict[str, Any] = {"symbol": symbol}
            record.update(indicators_map.get(symbol, {}))
            record.update(fundamentals_map.get(symbol, {}))
            record.update(news_map.get(symbol, {}))
            stocks_data.append(record)

        logger.info("[%s] Running screener on %d stocks", task_name, len(stocks_data))

        # ---- run screener ------------------------------------------------
        try:
            results: List[Dict[str, Any]] = await screener.run_full_screen(stocks_data)
            logger.info("[%s] Screener returned %d candidates", task_name, len(results))
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Screener error: %s", task_name, exc)
            return 0

        if not results:
            return 0

        # ---- persist results ---------------------------------------------
        saved = 0
        async with await _get_session() as session:
            async with session.begin():
                # Clear today's existing results first
                await session.execute(
                    text("DELETE FROM screening_results WHERE screened_on = :d"),
                    {"d": screened_on},
                )
                for idx, result in enumerate(results):
                    try:
                        await session.execute(
                            text(
                                """
                                INSERT INTO screening_results
                                    (symbol, screened_on, rank, composite_score,
                                     entry_price, stop_loss, target_price,
                                     rationale, tags, created_at)
                                VALUES
                                    (:symbol, :screened_on, :rank, :composite_score,
                                     :entry_price, :stop_loss, :target_price,
                                     :rationale, :tags, NOW())
                                """
                            ),
                            {
                                "symbol": result.get("symbol") or "",
                                "screened_on": screened_on,
                                "rank": idx + 1,
                                "composite_score": float(result.get("composite_score") or 0),
                                "entry_price": result.get("entry_price"),
                                "stop_loss": result.get("stop_loss"),
                                "target_price": result.get("target_price"),
                                "rationale": result.get("rationale") or "",
                                "tags": str(result.get("tags") or ""),
                            },
                        )
                        saved += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[%s] Failed to insert screening result for '%s': %s",
                            task_name, result.get("symbol"), exc,
                        )
        return saved

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — %d candidates saved", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=120)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 6 — morning_report
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.morning_report",
    bind=True,
    max_retries=3,
)
def morning_report(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Load today's top screening results, current market regime, and active
    themes from the DB. Build an AIReportRequest and call
    NvidiaAIService.generate_morning_report(). Persist the resulting AI
    report to the database.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = 1 if a report was generated.
    """
    task_name = "morning_report"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.nvidia_ai import AIReportRequest, NvidiaAIService  # noqa: PLC0415

        nvidia_service = NvidiaAIService()
        report_date = date.today().isoformat()

        async with await _get_session() as session:
            # ---- top 10 screening results ---------------------------------
            sr_rows = (
                await session.execute(
                    text(
                        """
                        SELECT symbol, rank, composite_score, entry_price,
                               stop_loss, target_price, rationale, tags
                        FROM screening_results
                        WHERE screened_on = :d
                        ORDER BY rank ASC
                        LIMIT 10
                        """
                    ),
                    {"d": report_date},
                )
            ).mappings().all()

            # ---- market regime -------------------------------------------
            regime_row = (
                await session.execute(
                    text(
                        """
                        SELECT regime, description, updated_at
                        FROM market_regime
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """
                    )
                )
            ).mappings().first()

            # ---- active themes -------------------------------------------
            theme_rows = (
                await session.execute(
                    text(
                        """
                        SELECT theme_name, score, rationale
                        FROM theme_scores
                        WHERE scored_on = :d
                        ORDER BY score DESC
                        LIMIT 5
                        """
                    ),
                    {"d": report_date},
                )
            ).mappings().all()

        top_stocks = [dict(r) for r in sr_rows]
        market_regime = dict(regime_row) if regime_row else {}
        themes = [dict(r) for r in theme_rows]

        logger.info(
            "[%s] Loaded %d stocks, regime=%s, %d themes",
            task_name,
            len(top_stocks),
            market_regime.get("regime"),
            len(themes),
        )

        # ---- build AI request -------------------------------------------
        request = AIReportRequest(
            report_date=report_date,
            top_stocks=top_stocks,
            market_regime=market_regime,
            themes=themes,
        )

        # ---- generate report --------------------------------------------
        try:
            report_dict: Dict[str, Any] = await nvidia_service.generate_morning_report(
                request
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] NvidiaAIService error: %s", task_name, exc)
            return 0

        # ---- persist report ---------------------------------------------
        async with await _get_session() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        INSERT INTO ai_reports
                            (report_date, report_type, title, summary, full_text,
                             top_picks_json, themes_json, metadata_json, created_at)
                        VALUES
                            (:report_date, 'morning', :title, :summary, :full_text,
                             :top_picks_json, :themes_json, :metadata_json, NOW())
                        ON CONFLICT (report_date, report_type)
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            full_text = EXCLUDED.full_text,
                            top_picks_json = EXCLUDED.top_picks_json,
                            themes_json = EXCLUDED.themes_json,
                            metadata_json = EXCLUDED.metadata_json,
                            created_at = NOW()
                        """
                    ),
                    {
                        "report_date": report_date,
                        "title": report_dict.get("title") or f"Morning Report {report_date}",
                        "summary": report_dict.get("summary") or "",
                        "full_text": report_dict.get("full_text") or str(report_dict),
                        "top_picks_json": str(report_dict.get("top_picks") or top_stocks),
                        "themes_json": str(themes),
                        "metadata_json": str(market_regime),
                    },
                )

        return 1

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — report generated: %s", task_name, bool(records))
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=180)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 7 — telegram_delivery
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.telegram_delivery",
    bind=True,
    max_retries=3,
)
def telegram_delivery(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Load today's AI morning report from the DB and deliver it to the
    configured Telegram channel via TelegramService.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of messages sent.
    """
    task_name = "telegram_delivery"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.telegram import TelegramService  # noqa: PLC0415

        telegram_service = TelegramService(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            chat_ids=[settings.TELEGRAM_CHAT_ID],
        )
        report_date = date.today().isoformat()

        # ---- load today's report -----------------------------------------
        async with await _get_session() as session:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT report_date, report_type, title, summary,
                               full_text, top_picks_json, themes_json, metadata_json
                        FROM ai_reports
                        WHERE report_date = :d AND report_type = 'morning'
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"d": report_date},
                )
            ).mappings().first()

        if not row:
            logger.warning("[%s] No morning report found for %s", task_name, report_date)
            return 0

        report_dict = dict(row)
        logger.info("[%s] Loaded report: '%s'", task_name, report_dict.get("title"))

        # ---- send to Telegram --------------------------------------------
        try:
            sent = await telegram_service.send_morning_report(
                chat_id=settings.TELEGRAM_CHAT_ID,
                report=report_dict,
            )
            messages_sent: int = sent if isinstance(sent, int) else 1
            logger.info("[%s] Sent %d message(s) to Telegram", task_name, messages_sent)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] TelegramService error: %s", task_name, exc)
            return 0

        return messages_sent

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — %d message(s) delivered", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=60)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 8 — intraday_scanner
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.intraday_scanner",
    bind=True,
    max_retries=3,
)
def intraday_scanner(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    Real-time alert scanner that runs during NSE market hours (09:15–15:30 IST,
    weekdays only). Loads the top 20 screening results, fetches 5-minute
    intraday candles, checks alert conditions (price cross, stop-loss breach,
    volume spike), and fires Telegram alerts for triggered conditions.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of alerts sent.
        Returns immediately with records=0 outside market hours.
    """
    task_name = "intraday_scanner"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    # ---- market-hours guard (sync, before spinning up asyncio) -----------
    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5:  # Saturday=5, Sunday=6
        logger.info(
            "[%s] Weekend (%s) — skipping scan", task_name, now_ist.strftime("%A")
        )
        return _result(task_name, True, 0, start)

    current_time = now_ist.time().replace(tzinfo=None)
    if not (MARKET_OPEN <= current_time <= MARKET_CLOSE):
        logger.info(
            "[%s] Outside market hours (%s IST) — skipping scan",
            task_name,
            current_time.strftime("%H:%M"),
        )
        return _result(task_name, True, 0, start)

    logger.info("[%s] Market is open — starting intraday scan", task_name)

    async def _run() -> int:
        from app.services.market_data import MarketDataService  # noqa: PLC0415
        from app.services.telegram import TelegramService  # noqa: PLC0415

        market_service = MarketDataService()
        telegram_service = TelegramService(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            chat_ids=[settings.TELEGRAM_CHAT_ID],
        )
        screened_on = date.today().isoformat()

        # ---- load top 20 candidates -------------------------------------
        async with await _get_session() as session:
            sr_rows = (
                await session.execute(
                    text(
                        """
                        SELECT symbol, entry_price, stop_loss, target_price, rationale
                        FROM screening_results
                        WHERE screened_on = :d
                        ORDER BY rank ASC
                        LIMIT 20
                        """
                    ),
                    {"d": screened_on},
                )
            ).mappings().all()

        candidates = [dict(r) for r in sr_rows]
        if not candidates:
            logger.info("[%s] No candidates found for today — nothing to scan", task_name)
            return 0

        logger.info("[%s] Scanning %d candidates", task_name, len(candidates))
        alerts_sent = 0

        for candidate in candidates:
            symbol: str = candidate["symbol"]
            entry_price: Optional[float] = candidate.get("entry_price")
            stop_loss: Optional[float] = candidate.get("stop_loss")

            # ---- fetch intraday 5-min candles ----------------------------
            try:
                candles: List[Dict[str, Any]] = (
                    await market_service.fetch_intraday_candles(
                        symbol=symbol, interval="5min"
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Failed to fetch candles for %s: %s", task_name, symbol, exc
                )
                continue

            if not candles:
                continue

            latest = candles[-1]
            latest_close: float = float(latest.get("close") or 0)
            latest_volume: int = int(latest.get("volume") or 0)

            # ---- compute average volume (last 20 bars) -------------------
            recent_volumes = [int(c.get("volume") or 0) for c in candles[-20:]]
            avg_volume: float = (
                sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
            )

            triggered_alerts: List[Dict[str, Any]] = []

            # Condition 1: Price crosses above entry_price (buy signal)
            if entry_price and latest_close >= entry_price:
                triggered_alerts.append(
                    {
                        "type": "BUY",
                        "symbol": symbol,
                        "message": (
                            f"🚀 *BUY ALERT* — {symbol}\n"
                            f"Price ₹{latest_close:.2f} crossed above entry ₹{entry_price:.2f}"
                        ),
                    }
                )
                logger.info(
                    "[%s] BUY alert for %s: close=%.2f entry=%.2f",
                    task_name, symbol, latest_close, entry_price,
                )

            # Condition 2: Price drops below stop_loss (stop alert)
            if stop_loss and latest_close <= stop_loss:
                triggered_alerts.append(
                    {
                        "type": "STOP",
                        "symbol": symbol,
                        "message": (
                            f"🔴 *STOP ALERT* — {symbol}\n"
                            f"Price ₹{latest_close:.2f} dropped below SL ₹{stop_loss:.2f}"
                        ),
                    }
                )
                logger.info(
                    "[%s] STOP alert for %s: close=%.2f sl=%.2f",
                    task_name, symbol, latest_close, stop_loss,
                )

            # Condition 3: Volume spike > 2x average
            if avg_volume > 0 and latest_volume > 2 * avg_volume:
                triggered_alerts.append(
                    {
                        "type": "VOLUME",
                        "symbol": symbol,
                        "message": (
                            f"📊 *VOLUME SPIKE* — {symbol}\n"
                            f"Volume {latest_volume:,} is {latest_volume / avg_volume:.1f}x "
                            f"the 20-bar average ({avg_volume:,.0f})"
                        ),
                    }
                )
                logger.info(
                    "[%s] VOLUME alert for %s: vol=%d avg=%.0f",
                    task_name, symbol, latest_volume, avg_volume,
                )

            # ---- fire alerts --------------------------------------------
            for alert in triggered_alerts:
                try:
                    await telegram_service.send_alert(
                        chat_id=settings.TELEGRAM_CHAT_ID,
                        alert_type=alert["type"],
                        message=alert["message"],
                    )
                    alerts_sent += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[%s] Failed to send %s alert for %s: %s",
                        task_name, alert["type"], symbol, exc,
                    )

        return alerts_sent

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — %d alert(s) sent", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=30)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))


# ===========================================================================
# Task 9 — post_market_review
# ===========================================================================


@celery_app.task(
    name="app.workers.tasks.post_market_review",
    bind=True,
    max_retries=3,
)
def post_market_review(self) -> Dict[str, Any]:  # type: ignore[override]
    """
    End-of-day paper trade review. Loads all open paper trades from the DB,
    fetches the day's closing price for each symbol, updates trade status via
    BacktestingService, calculates portfolio metrics, generates a performance
    summary, and optionally delivers a Telegram summary when trades were closed.

    Returns
    -------
    dict
        Standard task result dict with ``records`` = number of trades updated.
    """
    task_name = "post_market_review"
    start = time.perf_counter()
    logger.info("[%s] Starting task", task_name)

    async def _run() -> int:
        from app.services.backtesting import BacktestingService  # noqa: PLC0415
        from app.services.market_data import MarketDataService  # noqa: PLC0415
        from app.services.telegram import TelegramService  # noqa: PLC0415

        backtesting_service = BacktestingService()
        market_service = MarketDataService()
        telegram_service = TelegramService(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            chat_ids=[settings.TELEGRAM_CHAT_ID],
        )

        market_close_time = datetime.now(IST).replace(
            hour=15, minute=30, second=0, microsecond=0
        )
        today = date.today().isoformat()

        # ---- load open trades -------------------------------------------
        async with await _get_session() as session:
            trade_rows = (
                await session.execute(
                    text(
                        """
                        SELECT id, symbol, entry_price, stop_loss, target_price,
                               quantity, entry_date, status, trade_type, notes
                        FROM paper_trades
                        WHERE status = 'OPEN'
                        ORDER BY entry_date ASC
                        """
                    )
                )
            ).mappings().all()

        open_trades = [dict(r) for r in trade_rows]
        if not open_trades:
            logger.info("[%s] No open trades to review", task_name)
            return 0

        logger.info("[%s] Reviewing %d open trades", task_name, len(open_trades))

        trades_updated = 0
        trades_closed_today: List[Dict[str, Any]] = []

        for trade in open_trades:
            symbol: str = trade["symbol"]

            # ---- fetch closing price -------------------------------------
            try:
                closing_price: Optional[float] = (
                    await market_service.fetch_closing_price(symbol)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Failed to fetch closing price for %s: %s",
                    task_name, symbol, exc,
                )
                closing_price = None

            if closing_price is None:
                logger.warning(
                    "[%s] No closing price available for %s — skipping", task_name, symbol
                )
                continue

            # ---- update trade via BacktestingService --------------------
            try:
                updated_trade: Dict[str, Any] = await backtesting_service.update_trade(
                    trade=trade,
                    closing_price=closing_price,
                    market_close_time=market_close_time,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] BacktestingService.update_trade error for %s: %s",
                    task_name, symbol, exc,
                )
                continue

            # ---- persist updated trade ----------------------------------
            async with await _get_session() as session:
                async with session.begin():
                    try:
                        await session.execute(
                            text(
                                """
                                UPDATE paper_trades SET
                                    status        = :status,
                                    exit_price    = :exit_price,
                                    exit_date     = :exit_date,
                                    pnl           = :pnl,
                                    pnl_pct       = :pnl_pct,
                                    close_reason  = :close_reason,
                                    updated_at    = NOW()
                                WHERE id = :trade_id
                                """
                            ),
                            {
                                "trade_id": trade["id"],
                                "status": updated_trade.get("status") or "OPEN",
                                "exit_price": updated_trade.get("exit_price"),
                                "exit_date": updated_trade.get("exit_date"),
                                "pnl": updated_trade.get("pnl"),
                                "pnl_pct": updated_trade.get("pnl_pct"),
                                "close_reason": updated_trade.get("close_reason") or "",
                            },
                        )
                        trades_updated += 1

                        if updated_trade.get("status") in ("CLOSED", "SL_HIT", "TARGET_HIT"):
                            trades_closed_today.append(updated_trade)

                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[%s] Failed to persist updated trade for %s: %s",
                            task_name, symbol, exc,
                        )

        logger.info(
            "[%s] Updated %d trades; %d closed today",
            task_name, trades_updated, len(trades_closed_today),
        )

        # ---- compute metrics on all closed trades -----------------------
        async with await _get_session() as session:
            closed_rows = (
                await session.execute(
                    text(
                        """
                        SELECT symbol, entry_price, exit_price, pnl, pnl_pct,
                               entry_date, exit_date, status, trade_type
                        FROM paper_trades
                        WHERE status IN ('CLOSED', 'SL_HIT', 'TARGET_HIT')
                        ORDER BY exit_date DESC
                        LIMIT 200
                        """
                    )
                )
            ).mappings().all()

        all_closed_trades = [dict(r) for r in closed_rows]

        try:
            metrics: Dict[str, Any] = await backtesting_service.get_metrics(
                all_closed_trades
            )
            logger.info(
                "[%s] Metrics: win_rate=%.1f%% total_pnl=%.2f",
                task_name,
                metrics.get("win_rate", 0) * 100,
                metrics.get("total_pnl", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] BacktestingService.get_metrics error: %s", task_name, exc)
            metrics = {}

        # ---- persist performance report ---------------------------------
        async with await _get_session() as session:
            async with session.begin():
                try:
                    await session.execute(
                        text(
                            """
                            INSERT INTO performance_reports
                                (report_date, total_trades, open_trades, closed_trades,
                                 win_rate, total_pnl, avg_pnl, max_drawdown,
                                 sharpe_ratio, metrics_json, created_at)
                            VALUES
                                (:report_date, :total_trades, :open_trades, :closed_trades,
                                 :win_rate, :total_pnl, :avg_pnl, :max_drawdown,
                                 :sharpe_ratio, :metrics_json, NOW())
                            ON CONFLICT (report_date)
                            DO UPDATE SET
                                win_rate     = EXCLUDED.win_rate,
                                total_pnl    = EXCLUDED.total_pnl,
                                avg_pnl      = EXCLUDED.avg_pnl,
                                max_drawdown = EXCLUDED.max_drawdown,
                                sharpe_ratio = EXCLUDED.sharpe_ratio,
                                metrics_json = EXCLUDED.metrics_json,
                                created_at   = NOW()
                            """
                        ),
                        {
                            "report_date": today,
                            "total_trades": metrics.get("total_trades", len(all_closed_trades)),
                            "open_trades": len(open_trades) - trades_updated,
                            "closed_trades": len(trades_closed_today),
                            "win_rate": metrics.get("win_rate"),
                            "total_pnl": metrics.get("total_pnl"),
                            "avg_pnl": metrics.get("avg_pnl"),
                            "max_drawdown": metrics.get("max_drawdown"),
                            "sharpe_ratio": metrics.get("sharpe_ratio"),
                            "metrics_json": str(metrics),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] Failed to persist performance report: %s", task_name, exc)

        # ---- optional Telegram summary if trades closed today -----------
        if trades_closed_today:
            total_pnl = sum(
                float(t.get("pnl") or 0) for t in trades_closed_today
            )
            win_count = sum(
                1 for t in trades_closed_today if float(t.get("pnl") or 0) > 0
            )
            summary_lines = [
                f"📈 *Post-Market Summary — {today}*",
                f"Trades closed today: {len(trades_closed_today)}",
                f"Winners / Losers: {win_count} / {len(trades_closed_today) - win_count}",
                f"Total P&L: ₹{total_pnl:,.2f}",
                "",
            ]
            for t in trades_closed_today[:5]:  # show max 5 trades
                emoji = "✅" if float(t.get("pnl") or 0) > 0 else "❌"
                summary_lines.append(
                    f"{emoji} {t.get('symbol')} | "
                    f"Entry ₹{t.get('entry_price')} → Exit ₹{t.get('exit_price')} | "
                    f"P&L ₹{float(t.get('pnl') or 0):,.2f}"
                )

            summary_text = "\n".join(summary_lines)
            try:
                await telegram_service.send_alert(
                    chat_id=settings.TELEGRAM_CHAT_ID,
                    alert_type="POST_MARKET",
                    message=summary_text,
                )
                logger.info("[%s] Post-market Telegram summary sent", task_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Failed to send Telegram summary: %s", task_name, exc)

        return trades_updated

    try:
        records = asyncio.run(_run())
        logger.info("[%s] Completed — %d trade(s) updated", task_name, records)
        return _result(task_name, True, records, start)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] Unhandled error: %s", task_name, exc)
        try:
            raise self.retry(exc=exc, countdown=60)
        except self.MaxRetriesExceededError:
            return _result(task_name, False, 0, start, error=str(exc))
