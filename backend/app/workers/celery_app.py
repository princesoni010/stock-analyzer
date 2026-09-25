"""
Celery Application Configuration — Bharat Market AI
====================================================

This module initialises the Celery application instance and defines the
complete **Beat Schedule** that drives all automated market-research jobs
running on Indian Standard Time (IST, UTC+05:30).

Beat Schedule Overview
----------------------
The following 9 periodic tasks are registered:

1.  overnight_news_fetch   — 06:00 IST daily
    Fetches overnight global and domestic news headlines, earnings
    announcements, and macro data released after the previous trading
    session's close.  Results are stored and used by the morning report.

2.  official_filings_fetch — 06:30 IST daily
    Downloads BSE/NSE corporate filings (results, board meetings,
    shareholding patterns, insider trades) published after market close.
    Parses and indexes them for downstream analysis.

3.  weather_theme_update   — 06:45 IST daily
    Pulls IMD / external weather API data and maps regional weather
    patterns to sector themes (agriculture, power, FMCG, etc.) so that
    the screener can surface weather-driven trading ideas.

4.  market_data_update     — 07:00 IST daily
    Refreshes end-of-day OHLCV data, F&O OI/volume, delivery
    percentage, and index constituents from NSE/BSE data feeds.
    Also recalculates all technical indicators stored in the database.

5.  daily_screener         — 07:15 IST daily
    Runs the multi-factor stock screener across the full NSE universe
    (breakouts, momentum, value, quality) and generates a ranked
    watchlist that is persisted for the morning report.

6.  morning_report         — 07:45 IST daily
    Aggregates outputs from the preceding tasks (news, filings, weather
    themes, market data, screener) and uses the LLM pipeline to
    compose the structured morning briefing document.

7.  telegram_delivery      — 08:00 IST daily
    Delivers the finalised morning report to configured Telegram
    channels/users.  Retries automatically on Telegram API failures.

8.  intraday_scanner       — Every 5 minutes, 09:00–15:59 IST, Mon–Fri
    Scans live price feeds for intraday breakout, reversal, and
    volume-spike signals.  The crontab fires every 5 minutes during the
    broad hour window (9–15); the task implementation enforces the
    precise market-hours boundary (09:15–15:30 IST).

9.  post_market_review     — 15:45 IST daily, Mon–Fri
    Runs after market close to evaluate the day's trades vs.
    morning calls, compute P&L attribution, and persist analytics
    data used for model retraining and performance dashboards.

Redis
-----
Both the **broker** and the **result backend** are backed by Redis whose
URL is read from ``settings.REDIS_URL`` (e.g. ``redis://localhost:6379/0``).

Timezone
--------
All crontab expressions are interpreted in ``Asia/Kolkata`` (IST, UTC+05:30).
Celery uses pytz internally; ensure ``pytz`` is installed.
"""

import logging
from celery import Celery
from celery.schedules import crontab

from app.config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery application instance
# ---------------------------------------------------------------------------
celery_app = Celery(
    "bharat_market_ai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# ---------------------------------------------------------------------------
# Core configuration
# ---------------------------------------------------------------------------
celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Result TTL — 1 hour
    result_expires=3600,

    # Reliability: tasks are acknowledged only after successful execution,
    # preventing silent loss if a worker dies mid-task.
    task_acks_late=True,

    # Disable prefetching so each worker slot processes one task at a time,
    # keeping memory usage predictable for long-running research tasks.
    worker_prefetch_multiplier=1,

    # Timezone — all crontab schedules are in IST (UTC+05:30).
    timezone="Asia/Kolkata",
    enable_utc=True,

    # ---------------------------------------------------------------------------
    # Beat Schedule
    # ---------------------------------------------------------------------------
    beat_schedule={
        # ------------------------------------------------------------------
        # 1. Overnight News Fetch — 06:00 IST daily
        # ------------------------------------------------------------------
        "overnight_news_fetch": {
            "task": "app.workers.tasks.overnight_news_fetch",
            "schedule": crontab(hour=6, minute=0),
            "options": {"queue": "default"},
        },

        # ------------------------------------------------------------------
        # 2. Official Filings Fetch — 06:30 IST daily
        # ------------------------------------------------------------------
        "official_filings_fetch": {
            "task": "app.workers.tasks.official_filings_fetch",
            "schedule": crontab(hour=6, minute=30),
            "options": {"queue": "default"},
        },

        # ------------------------------------------------------------------
        # 3. Weather Theme Update — 06:45 IST daily
        # ------------------------------------------------------------------
        "weather_theme_update": {
            "task": "app.workers.tasks.weather_theme_update",
            "schedule": crontab(hour=6, minute=45),
            "options": {"queue": "default"},
        },

        # ------------------------------------------------------------------
        # 4. Market Data Update — 07:00 IST daily
        # ------------------------------------------------------------------
        "market_data_update": {
            "task": "app.workers.tasks.market_data_update",
            "schedule": crontab(hour=7, minute=0),
            "options": {"queue": "default"},
        },

        # ------------------------------------------------------------------
        # 5. Daily Screener — 07:15 IST daily
        # ------------------------------------------------------------------
        "daily_screener": {
            "task": "app.workers.tasks.daily_screener",
            "schedule": crontab(hour=7, minute=15),
            "options": {"queue": "default"},
        },

        # ------------------------------------------------------------------
        # 6. Morning Report — 07:45 IST daily
        # ------------------------------------------------------------------
        "morning_report": {
            "task": "app.workers.tasks.morning_report",
            "schedule": crontab(hour=7, minute=45),
            "options": {"queue": "default"},
        },

        # ------------------------------------------------------------------
        # 7. Telegram Delivery — 08:00 IST daily
        # ------------------------------------------------------------------
        "telegram_delivery": {
            "task": "app.workers.tasks.telegram_delivery",
            "schedule": crontab(hour=8, minute=0),
            "options": {"queue": "default"},
        },

        # ------------------------------------------------------------------
        # 8. Intraday Scanner — every 5 minutes, 09:00–15:59 IST, Mon–Fri
        #
        # The crontab fires every 5 minutes across hours 9–15 on weekdays.
        # The task implementation itself enforces the precise NSE session
        # window (09:15–15:30 IST) and exits immediately outside that range.
        # Using hour='9-15' rather than '9,10,11,12,13,14,15' is equivalent
        # but the comma-separated form is used here for explicitness.
        # ------------------------------------------------------------------
        "intraday_scanner": {
            "task": "app.workers.tasks.intraday_scanner",
            "schedule": crontab(
                minute="*/5",
                hour="9,10,11,12,13,14,15",
                day_of_week="1-5",  # Monday=1 … Friday=5 (Celery convention)
            ),
            "options": {"queue": "intraday"},
        },

        # ------------------------------------------------------------------
        # 9. Post-Market Review — 15:45 IST daily, Mon–Fri
        # ------------------------------------------------------------------
        "post_market_review": {
            "task": "app.workers.tasks.post_market_review",
            "schedule": crontab(hour=15, minute=45, day_of_week="1-5"),
            "options": {"queue": "default"},
        },
    },
)

# ---------------------------------------------------------------------------
# Auto-discover tasks from the app.workers.tasks module
# ---------------------------------------------------------------------------
celery_app.autodiscover_tasks(["app.workers"])

logger.info(
    "Celery application '%s' initialised with broker=%s and %d beat entries.",
    celery_app.main,
    settings.REDIS_URL,
    len(celery_app.conf.beat_schedule),
)
