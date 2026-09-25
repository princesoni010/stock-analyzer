"""
Bharat Market AI – Application Configuration
=============================================
All runtime configuration is drawn from environment variables (or a .env file).
Use:  from app.config import settings
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object.  Values come from environment variables or a
    .env file located at the project root.  Pydantic-settings handles all
    coercion and validation automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_ENV: str = Field(default="development", description="development | staging | production")
    LOG_LEVEL: str = Field(default="INFO", description="Python logging level")
    CALCULATION_VERSION: str = Field(default="1.0", description="Version tag stamped on computed rows")
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], description="CORS allowed origins")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DATABASE_URL: str = Field(
        ...,
        description="Async PostgreSQL URL, e.g. postgresql+asyncpg://user:pass@host/db",
    )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # ------------------------------------------------------------------
    # NVIDIA NIM / LLM
    # ------------------------------------------------------------------
    NVIDIA_API_KEY: str = Field(default="", description="NVIDIA NIM API key")
    NVIDIA_BASE_URL: AnyHttpUrl = Field(
        default="https://integrate.api.nvidia.com/v1",  # type: ignore[assignment]
        description="NVIDIA NIM base URL",
    )
    NVIDIA_MODEL: str = Field(
        default="meta/llama3-70b-instruct",
        description="Model identifier served by NVIDIA NIM",
    )

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram bot token from @BotFather")
    TELEGRAM_CHAT_ID: str = Field(
        default="",
        description="Comma-separated list of allowed Telegram chat IDs",
    )

    @property
    def telegram_chat_ids(self) -> List[int]:
        """Return TELEGRAM_CHAT_ID as a list of integers."""
        return [int(cid.strip()) for cid in self.TELEGRAM_CHAT_ID.split(",") if cid.strip()]

    # ------------------------------------------------------------------
    # External Data APIs
    # ------------------------------------------------------------------
    MARKET_DATA_API_KEY: str = Field(default="", description="Market data provider API key")
    NEWS_API_KEY: str = Field(default="", description="News aggregator API key")
    IMD_API_KEY: str = Field(default="", description="India Meteorological Department API key")

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    REPORT_TIME: str = Field(
        default="08:00",
        description="Daily report generation time in HH:MM (local timezone)",
    )
    TIMEZONE: str = Field(
        default="Asia/Kolkata",
        description="Timezone for scheduling and display",
    )

    # ------------------------------------------------------------------
    # Risk Management
    # ------------------------------------------------------------------
    ACCOUNT_CAPITAL: float = Field(
        default=100_000.0,
        description="Total trading capital in INR",
    )
    MAX_POSITION_VALUE: float = Field(
        default=20_000.0,
        description="Maximum value of a single position in INR",
    )
    MAX_RISK_PER_TRADE: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of capital risked per trade (e.g. 0.01 = 1%)",
    )
    MIN_RISK_REWARD: float = Field(
        default=1.5,
        ge=1.0,
        description="Minimum acceptable risk-to-reward ratio",
    )
    MAX_SIMULTANEOUS_TRADES: int = Field(
        default=5,
        ge=1,
        description="Maximum number of open paper trades at one time",
    )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    ALERT_COOLDOWN_MINUTES: int = Field(
        default=60,
        ge=0,
        description="Minimum minutes between repeated alerts for the same symbol",
    )
    QUIET_HOURS_START: str = Field(
        default="22:00",
        description="Start of quiet hours (alerts suppressed) in HH:MM local time",
    )
    QUIET_HOURS_END: str = Field(
        default="07:00",
        description="End of quiet hours in HH:MM local time",
    )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    MAX_STOCKS_IN_REPORT: int = Field(
        default=10,
        ge=1,
        description="Maximum number of stocks to include in a daily report",
    )

    # ------------------------------------------------------------------
    # Scoring Weights  (must sum to 1.0)
    # ------------------------------------------------------------------
    SCORING_WEIGHTS_TECHNICAL: float = Field(default=0.30, ge=0.0, le=1.0)
    SCORING_WEIGHTS_FUNDAMENTAL: float = Field(default=0.25, ge=0.0, le=1.0)
    SCORING_WEIGHTS_NEWS: float = Field(default=0.20, ge=0.0, le=1.0)
    SCORING_WEIGHTS_THEME: float = Field(default=0.15, ge=0.0, le=1.0)
    SCORING_WEIGHTS_LIQUIDITY: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_scoring_weights(self) -> "Settings":
        total = (
            self.SCORING_WEIGHTS_TECHNICAL
            + self.SCORING_WEIGHTS_FUNDAMENTAL
            + self.SCORING_WEIGHTS_NEWS
            + self.SCORING_WEIGHTS_THEME
            + self.SCORING_WEIGHTS_LIQUIDITY
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.6f}. "
                "Adjust SCORING_WEIGHTS_* environment variables."
            )
        return self

    # ------------------------------------------------------------------
    # Theme Thresholds
    # ------------------------------------------------------------------
    THEME_GREEN_THRESHOLD: int = Field(
        default=75,
        ge=0,
        le=100,
        description="Score >= this value → theme status 'green'",
    )
    THEME_YELLOW_THRESHOLD: int = Field(
        default=55,
        ge=0,
        le=100,
        description="Score >= this value (but < GREEN) → theme status 'yellow'",
    )

    @model_validator(mode="after")
    def _validate_theme_thresholds(self) -> "Settings":
        if self.THEME_YELLOW_THRESHOLD >= self.THEME_GREEN_THRESHOLD:
            raise ValueError(
                "THEME_YELLOW_THRESHOLD must be strictly less than THEME_GREEN_THRESHOLD. "
                f"Got yellow={self.THEME_YELLOW_THRESHOLD}, green={self.THEME_GREEN_THRESHOLD}."
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def scoring_weights(self) -> dict[str, float]:
        return {
            "technical": self.SCORING_WEIGHTS_TECHNICAL,
            "fundamental": self.SCORING_WEIGHTS_FUNDAMENTAL,
            "news": self.SCORING_WEIGHTS_NEWS,
            "theme": self.SCORING_WEIGHTS_THEME,
            "liquidity": self.SCORING_WEIGHTS_LIQUIDITY,
        }

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("APP_ENV")
    @classmethod
    def _validate_app_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}, got '{v}'")
        return lower


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()  # type: ignore[call-arg]


# Module-level convenience alias
settings: Settings = get_settings()
