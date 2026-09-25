"""
Bharat Market AI – SQLAlchemy ORM Models
=========================================
All database tables are defined here using the modern SQLAlchemy 2.x
Declarative API with Python-typed `Mapped` annotations.

Table inventory
---------------
  stocks                 – Master list of traded instruments
  market_candles         – OHLCV price history
  news_articles          – Ingested news items with trust scoring
  news_entities          – Stock/sector entities extracted from articles
  corporate_announcements– Exchange filings (NSE/BSE)
  fundamentals           – Quarterly/annual fundamental data
  weather_data           – IMD rainfall and reservoir data
  themes                 – Investable macro/sector themes
  theme_stock_mapping    – Which stocks belong to which themes
  indicator_values       – Pre-computed technical indicators
  screening_results      – Daily screening output with scores & signals
  ai_reports             – LLM-generated market reports
  paper_trades           – Simulated trades for back-testing / tracking

Notes
-----
* All primary keys use UUIDs (PostgreSQL `uuid` type via asyncpg).
* All timestamps are stored in UTC.
* JSONB is used for semi-structured fields (PostgreSQL-native binary JSON).
* Numeric(12, 4) is used for price fields to avoid floating-point drift.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Common declarative base for all ORM models."""

    type_annotation_map = {
        uuid.UUID: UUID(as_uuid=True),
        dict: JSONB,
        list: JSONB,
    }


# ---------------------------------------------------------------------------
# stocks
# ---------------------------------------------------------------------------
class Stock(Base):
    __tablename__ = "stocks"

    __table_args__ = (
        UniqueConstraint("exchange", "symbol", name="uq_stocks_exchange_symbol"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="NSE or BSE"
    )
    isin: Mapped[str] = mapped_column(String(12), nullable=False, unique=True, index=True)
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_fno: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    candles: Mapped[List["MarketCandle"]] = relationship(
        "MarketCandle", back_populates="stock_ref", primaryjoin="Stock.symbol == foreign(MarketCandle.symbol)", viewonly=True
    )
    indicators: Mapped[List["IndicatorValue"]] = relationship(
        "IndicatorValue", back_populates="stock_ref", primaryjoin="Stock.symbol == foreign(IndicatorValue.symbol)", viewonly=True
    )
    screening_results: Mapped[List["ScreeningResult"]] = relationship(
        "ScreeningResult", back_populates="stock_ref", primaryjoin="Stock.symbol == foreign(ScreeningResult.symbol)", viewonly=True
    )
    paper_trades: Mapped[List["PaperTrade"]] = relationship(
        "PaperTrade", back_populates="stock_ref", primaryjoin="Stock.symbol == foreign(PaperTrade.symbol)", viewonly=True
    )

    def __repr__(self) -> str:
        return f"<Stock {self.exchange}:{self.symbol}>"


# ---------------------------------------------------------------------------
# market_candles
# ---------------------------------------------------------------------------
class MarketCandle(Base):
    __tablename__ = "market_candles"

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candles_symbol_tf_ts"),
        Index("ix_candles_symbol_tf_ts_desc", "symbol", "timeframe", "timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(
        String(5), nullable=False, comment="e.g. 1m 5m 15m 1h 1d"
    )
    open: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="Candle open time in UTC"
    )
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationship (symbol-based, viewonly)
    stock_ref: Mapped[Optional["Stock"]] = relationship(
        "Stock",
        primaryjoin="foreign(MarketCandle.symbol) == Stock.symbol",
        viewonly=True,
    )

    def __repr__(self) -> str:
        return f"<MarketCandle {self.symbol}[{self.timeframe}] @ {self.timestamp}>"


# ---------------------------------------------------------------------------
# news_articles
# ---------------------------------------------------------------------------
class NewsArticle(Base):
    __tablename__ = "news_articles"

    __table_args__ = (
        Index("ix_news_published_desc", "published_at"),
        Index("ix_news_content_hash", "content_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    source_name: Mapped[str] = mapped_column(String(120), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="official | reputable | opinion | unverified",
    )
    trust_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1 (unverified) to 4 (official exchange/regulator)",
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    event_type: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    materiality_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="0.0 (immaterial) to 1.0 (highly material)"
    )
    is_primary: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, comment="False if this is a duplicate/follow-up"
    )
    primary_article_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("news_articles.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Self-referential relationship for duplicate grouping
    primary_article: Mapped[Optional["NewsArticle"]] = relationship(
        "NewsArticle",
        remote_side="NewsArticle.id",
        backref="duplicates",
        foreign_keys=[primary_article_id],
    )
    entities: Mapped[List["NewsEntity"]] = relationship(
        "NewsEntity", back_populates="article", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<NewsArticle [{self.source_name}] '{self.title[:60]}…'>"


# ---------------------------------------------------------------------------
# news_entities
# ---------------------------------------------------------------------------
class NewsEntity(Base):
    __tablename__ = "news_entities"

    __table_args__ = (
        Index("ix_news_entities_symbol_article", "symbol", "article_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    symbol: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="e.g. STOCK, SECTOR, INDEX, COMMODITY"
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0, comment="NLP confidence score 0.0–1.0"
    )

    article: Mapped["NewsArticle"] = relationship("NewsArticle", back_populates="entities")

    def __repr__(self) -> str:
        return f"<NewsEntity {self.entity_type}:{self.symbol or self.sector}>"


# ---------------------------------------------------------------------------
# corporate_announcements
# ---------------------------------------------------------------------------
class CorporateAnnouncement(Base):
    __tablename__ = "corporate_announcements"

    __table_args__ = (
        Index("ix_corp_ann_symbol_published", "symbol", "published_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    exchange: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    announcement_type: Mapped[str] = mapped_column(
        String(80), nullable=False, comment="e.g. BOARDMEETING, RESULTS, AGM, DIVIDEND"
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<CorporateAnnouncement {self.exchange}:{self.symbol} [{self.announcement_type}]>"


# ---------------------------------------------------------------------------
# fundamentals
# ---------------------------------------------------------------------------
class Fundamental(Base):
    __tablename__ = "fundamentals"

    __table_args__ = (
        UniqueConstraint("symbol", "period", name="uq_fundamentals_symbol_period"),
        Index("ix_fundamentals_symbol", "symbol"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    period: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="e.g. Q2FY25 or FY25"
    )
    revenue: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    ebitda_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roe: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Return on equity %")
    roce: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Return on capital employed %")
    debt_to_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_cash_flow: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    promoter_holding: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Promoter holding %")
    promoter_pledge: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Pledged promoter holding %")
    pe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revenue_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="YoY revenue growth %")
    profit_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="YoY profit growth %")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<Fundamental {self.symbol} {self.period}>"


# ---------------------------------------------------------------------------
# weather_data
# ---------------------------------------------------------------------------
class WeatherData(Base):
    __tablename__ = "weather_data"

    __table_args__ = (
        UniqueConstraint("region", "data_date", name="uq_weather_region_date"),
        Index("ix_weather_region_date", "region", "data_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    region: Mapped[str] = mapped_column(String(100), nullable=False)
    data_date: Mapped[date] = mapped_column(Date, nullable=False)
    rainfall_actual: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rainfall_normal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    deviation_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="% deviation from long-period average"
    )
    reservoir_level: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="% of full reservoir capacity"
    )
    forecast: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    warnings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="IMD")
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    is_stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<WeatherData {self.region} {self.data_date}>"


# ---------------------------------------------------------------------------
# themes
# ---------------------------------------------------------------------------
class Theme(Base):
    __tablename__ = "themes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active_months: Mapped[Optional[List[Any]]] = mapped_column(
        JSONB, nullable=True, comment="List of month numbers 1-12 when this theme is active"
    )
    positive_factors: Mapped[Optional[List[Any]]] = mapped_column(
        JSONB, nullable=True, comment="List of positive-signal factor descriptors"
    )
    negative_factors: Mapped[Optional[List[Any]]] = mapped_column(
        JSONB, nullable=True, comment="List of negative-signal factor descriptors"
    )
    score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="Composite score 0-100"
    )
    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default="no_trade",
        comment="green | yellow | red | no_trade"
    )
    calculation_version: Mapped[str] = mapped_column(String(10), nullable=False, default="1.0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    stock_mappings: Mapped[List["ThemeStockMapping"]] = relationship(
        "ThemeStockMapping", back_populates="theme", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Theme '{self.name}' status={self.status} score={self.score:.1f}>"


# ---------------------------------------------------------------------------
# theme_stock_mapping
# ---------------------------------------------------------------------------
class ThemeStockMapping(Base):
    __tablename__ = "theme_stock_mapping"

    __table_args__ = (
        UniqueConstraint("theme_id", "symbol", name="uq_theme_stock"),
        Index("ix_theme_stock_symbol", "symbol"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    theme_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("themes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    exposure_level: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="high | medium | low"
    )
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    theme: Mapped["Theme"] = relationship("Theme", back_populates="stock_mappings")

    def __repr__(self) -> str:
        return f"<ThemeStockMapping theme_id={self.theme_id} symbol={self.symbol} [{self.exposure_level}]>"


# ---------------------------------------------------------------------------
# indicator_values
# ---------------------------------------------------------------------------
class IndicatorValue(Base):
    __tablename__ = "indicator_values"

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_indicator_symbol_tf_ts"),
        Index("ix_indicator_symbol_tf_ts_desc", "symbol", "timeframe", "timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="Bar timestamp (UTC)"
    )

    # Moving averages
    ema_20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema_50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema_200: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Momentum
    rsi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_hist: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Volatility / Volume
    atr_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vwap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_volume_20d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_ratio: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Today's volume / 20-day avg"
    )

    # Relative strength vs Nifty 50
    relative_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Support / Resistance
    support: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resistance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Gap & ORB
    gap_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Opening gap %")
    orb_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Opening Range Breakout high")
    orb_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Opening Range Breakout low")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationship
    stock_ref: Mapped[Optional["Stock"]] = relationship(
        "Stock",
        primaryjoin="foreign(IndicatorValue.symbol) == Stock.symbol",
        viewonly=True,
    )

    def __repr__(self) -> str:
        return f"<IndicatorValue {self.symbol}[{self.timeframe}] @ {self.timestamp}>"


# ---------------------------------------------------------------------------
# screening_results
# ---------------------------------------------------------------------------
class ScreeningResult(Base):
    __tablename__ = "screening_results"

    __table_args__ = (
        Index("ix_screening_created_score_desc", "created_at", "total_score"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Component scores (0-100)
    technical_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fundamental_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    news_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    theme_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    liquidity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, index=True)

    signal: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="green | yellow | red | no_trade"
    )

    # Trade levels
    entry_low: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    entry_high: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    target_1: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    target_2: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    risk_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    reason_codes: Mapped[Optional[List[Any]]] = mapped_column(
        JSONB, nullable=True, comment="Machine-readable list of reason code strings"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True
    )

    # Relationship
    stock_ref: Mapped[Optional["Stock"]] = relationship(
        "Stock",
        primaryjoin="foreign(ScreeningResult.symbol) == Stock.symbol",
        viewonly=True,
    )
    paper_trades: Mapped[List["PaperTrade"]] = relationship(
        "PaperTrade", back_populates="screening_result"
    )

    def __repr__(self) -> str:
        return f"<ScreeningResult {self.symbol} score={self.total_score:.1f} [{self.signal}]>"


# ---------------------------------------------------------------------------
# ai_reports
# ---------------------------------------------------------------------------
class AIReport(Base):
    __tablename__ = "ai_reports"

    __table_args__ = (
        UniqueConstraint("report_type", "report_date", name="uq_ai_report_type_date"),
        Index("ix_ai_report_type_date", "report_type", "report_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    report_type: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="morning | intraday | post_market"
    )
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    input_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="SHA-256 of the rendered prompt input"
    )
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_ids: Mapped[Optional[List[Any]]] = mapped_column(
        JSONB, nullable=True, comment="List of UUIDs of screening_results / news_articles used"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<AIReport {self.report_type} {self.report_date} model={self.model}>"


# ---------------------------------------------------------------------------
# paper_trades
# ---------------------------------------------------------------------------
class PaperTrade(Base):
    __tablename__ = "paper_trades"

    __table_args__ = (
        Index("ix_paper_trades_status_signal_time", "status", "signal_time"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    signal_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="When the screening signal was generated (UTC)"
    )
    entry: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    stop_loss: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    target_1: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    target_2: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, comment="Number of shares")

    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)
    exit_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pnl: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2), nullable=True, comment="Realised PnL in INR"
    )
    outcome: Mapped[str] = mapped_column(
        String(15),
        nullable=False,
        default="open",
        comment="hit_t1 | hit_t2 | stopped_out | expired | open",
    )
    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default="open", index=True, comment="open | closed"
    )

    screening_result_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("screening_results.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    screening_result: Mapped[Optional["ScreeningResult"]] = relationship(
        "ScreeningResult", back_populates="paper_trades"
    )
    stock_ref: Mapped[Optional["Stock"]] = relationship(
        "Stock",
        primaryjoin="foreign(PaperTrade.symbol) == Stock.symbol",
        viewonly=True,
    )

    def __repr__(self) -> str:
        return f"<PaperTrade {self.symbol} entry={self.entry} [{self.status}/{self.outcome}]>"
