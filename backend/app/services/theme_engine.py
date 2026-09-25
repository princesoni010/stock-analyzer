"""
Theme Engine — seasonal and weather-driven theme scoring for Indian equities.

Scores themes on a 0-100 scale using a weighted formula:
    season*0.25 + weather*0.20 + demand*0.15 + policy*0.15
    + exposure*0.15 + technical*0.10

No AI, no external calls. All data is passed in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThemeScore:
    """Scored output for a single investment theme."""

    name: str
    score: float                            # 0-100 composite
    status: str                             # 'green' | 'yellow' | 'red' | 'no_trade'

    # Component scores (each 0-100)
    season_strength: float
    weather_score: float
    demand_score: float
    policy_score: float
    exposure_score: float
    technical_score: float

    positive_factors: list[str] = field(default_factory=list)
    negative_factors: list[str] = field(default_factory=list)
    affected_stocks: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status in ("green", "yellow")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 2),
            "status": self.status,
            "season_strength": round(self.season_strength, 2),
            "weather_score": round(self.weather_score, 2),
            "demand_score": round(self.demand_score, 2),
            "policy_score": round(self.policy_score, 2),
            "exposure_score": round(self.exposure_score, 2),
            "technical_score": round(self.technical_score, 2),
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "affected_stocks": self.affected_stocks,
        }


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

_THEME_DEFINITIONS: dict[str, dict] = {
    "kharif_agriculture": {
        "display_name": "Kharif Agriculture",
        "active_months": [6, 7, 8, 9, 10],
        "stocks": ["COROMANDEL", "UPL", "RALLIS", "PI", "BAYER", "CHAMBAL", "GNFC"],
        "demand_keywords": ["fertiliser", "pesticide", "kharif", "sowing", "monsoon"],
        "policy_keywords": ["msp", "minimum support price", "crop insurance", "pm-kisan",
                            "fertiliser subsidy", "urea"],
        "weather_sensitive": True,
        "description": "Agrochemical and fertiliser companies benefit from kharif sowing season.",
    },
    "rabi_agriculture": {
        "display_name": "Rabi Agriculture",
        "active_months": [10, 11, 12, 1, 2, 3],
        "stocks": ["COROMANDEL", "UPL", "RALLIS"],
        "demand_keywords": ["rabi", "wheat", "sowing", "mustard", "pulses"],
        "policy_keywords": ["msp", "minimum support price", "rabi crop", "fertiliser"],
        "weather_sensitive": True,
        "description": "Winter crop cycle demand for agri inputs.",
    },
    "monsoon": {
        "display_name": "Monsoon Play",
        "active_months": [6, 7, 8, 9],
        "stocks": ["KANSAINER", "SHREECEM", "PIIND", "UPL"],
        "demand_keywords": ["monsoon", "rainfall", "imd forecast", "deficit", "surplus"],
        "policy_keywords": ["flood relief", "drought", "reservoir"],
        "weather_sensitive": True,
        "description": "Stocks with direct revenue linkage to monsoon performance.",
    },
    "festive_demand": {
        "display_name": "Festive Demand",
        "active_months": [9, 10, 11],
        "stocks": ["MARUTI", "TITAN", "ASIANPAINT", "VOLTAS", "HAVELLS", "WHIRLPOOL"],
        "demand_keywords": ["festive", "diwali", "navratri", "durga puja", "dussehra",
                            "auto sales", "retail"],
        "policy_keywords": ["gst reduction", "consumer scheme", "incentive"],
        "weather_sensitive": False,
        "description": "Consumer discretionary surge during Indian festive season.",
    },
    "infrastructure": {
        "display_name": "Infrastructure Cycle",
        "active_months": list(range(1, 13)),
        "stocks": ["LT", "KNR", "PNC", "ASHOKA", "IRB", "ULTRATECH"],
        "demand_keywords": ["capex", "infrastructure", "road", "highway", "order book",
                            "order inflow", "construction"],
        "policy_keywords": ["national infrastructure pipeline", "nip", "pli", "budget",
                            "capex allocation", "pm gati shakti"],
        "weather_sensitive": False,
        "description": "Government capital expenditure driving construction and EPC companies.",
    },
    "crude_cycle": {
        "display_name": "Crude Oil Cycle",
        "active_months": list(range(1, 13)),
        "stocks": ["ONGC", "BPCL", "IOC", "RELIANCE", "GAIL"],
        "demand_keywords": ["crude", "brent", "wti", "opec", "oil price", "refinery",
                            "petrochemical"],
        "policy_keywords": ["fuel pricing", "petrol price", "windfall tax", "upstream",
                            "downstream"],
        "weather_sensitive": False,
        "description": "Oil & gas upstream and downstream sensitivity to crude price cycle.",
    },
    "metal_cycle": {
        "display_name": "Metal & Mining Cycle",
        "active_months": list(range(1, 13)),
        "stocks": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA"],
        "demand_keywords": ["steel", "aluminium", "copper", "zinc", "coal", "lme",
                            "china demand", "metal prices"],
        "policy_keywords": ["import duty", "anti-dumping", "export duty", "mining",
                            "royalty"],
        "weather_sensitive": False,
        "description": "Global and domestic metal price cycle affecting Indian producers.",
    },
    "interest_rate_cycle": {
        "display_name": "Interest Rate Cycle",
        "active_months": list(range(1, 13)),
        "stocks": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "BAJFINANCE"],
        "demand_keywords": ["credit growth", "nim", "npa", "loan", "deposit rate",
                            "repo rate"],
        "policy_keywords": ["rbi rate", "monetary policy", "mpc", "repo cut", "repo hike",
                            "cpi", "inflation"],
        "weather_sensitive": False,
        "description": "Banking and NBFC performance tied to RBI rate cycle.",
    },
    "festive_export": {
        "display_name": "IT/Export Festive Season",
        "active_months": [10, 11, 12, 1],
        "stocks": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
        "demand_keywords": ["it spend", "deal win", "tcy", "discretionary spend",
                            "digital transformation"],
        "policy_keywords": ["visa", "h1b", "trade deal", "software exports"],
        "weather_sensitive": False,
        "description": "Global enterprise IT spending uptick in Q4 (Oct-Dec) benefits Indian IT.",
    },
    "earnings_season": {
        "display_name": "Earnings Season",
        "active_months": [1, 4, 7, 10],
        "stocks": [],
        "demand_keywords": ["results", "earnings", "quarterly results", "q1", "q2",
                            "q3", "q4", "revenue beat", "profit beat"],
        "policy_keywords": ["guidance", "management commentary"],
        "weather_sensitive": False,
        "description": "Broad market volatility during quarterly earnings announcement months.",
    },
}

# ---------------------------------------------------------------------------
# Theme Engine
# ---------------------------------------------------------------------------

class ThemeEngine:
    """
    Score investment themes based on seasonal, weather, demand, policy,
    exposure, and technical factors.

    Usage::

        engine = ThemeEngine()
        scores = engine.score_all_themes(
            weather_score=70.0,
            demand_indicators={"fertiliser_demand_index": 80},
            policy_announcements=[{"title": "MSP raised for kharif crops", "date": "2026-06-01"}],
            stock_scores={"COROMANDEL": 72.5, "UPL": 60.0},
            stock_indicators={"COROMANDEL": {"above_ema50": True}, "UPL": {"above_ema50": False}},
        )
    """

    THEMES: dict[str, dict] = _THEME_DEFINITIONS

    # Composite score weights (must sum to 1.0)
    WEIGHTS: dict[str, float] = {
        "season":    0.25,
        "weather":   0.20,
        "demand":    0.15,
        "policy":    0.15,
        "exposure":  0.15,
        "technical": 0.10,
    }

    # ---------------------------------------------------------------------------
    # Season strength
    # ---------------------------------------------------------------------------

    def calculate_season_strength(self, theme_name: str, current_month: int) -> float:
        """
        Return seasonal strength score (0-100).

        - 100: current month is in the theme's active window.
        - 50:  adjacent month (one month before or after active window).
        - 0:   off-season.

        Args:
            theme_name:    One of THEMES keys.
            current_month: Integer 1-12.
        """
        theme = self.THEMES.get(theme_name)
        if theme is None:
            logger.warning("calculate_season_strength: unknown theme '%s'", theme_name)
            return 0.0

        active = set(theme["active_months"])
        if current_month in active:
            return 100.0

        # Adjacent month check (handle wrap-around: month 1 neighbors 12 and 2)
        prev_month = 12 if current_month == 1 else current_month - 1
        next_month = 1 if current_month == 12 else current_month + 1

        if prev_month in active or next_month in active:
            return 50.0

        return 0.0

    # ---------------------------------------------------------------------------
    # Demand score
    # ---------------------------------------------------------------------------

    def calculate_demand_score(
        self, theme_name: str, indicators: dict
    ) -> float:
        """
        Derive a 0-100 demand score from available macro / sector indicators.

        Heuristics applied:
        - If the theme has a named index in indicators (e.g. 'fertiliser_demand_index'),
          use it directly as a score (0-100).
        - Otherwise, check for generic 'demand_growth' (%), 'capacity_utilisation' (%),
          or 'order_book_growth' (%) keys and score them.
        - Falls back to 50 (neutral) if no relevant indicator found.
        """
        theme = self.THEMES.get(theme_name)
        if theme is None:
            return 50.0

        # Attempt theme-specific named index
        for keyword in theme.get("demand_keywords", []):
            index_key = f"{keyword.replace(' ', '_')}_index"
            if index_key in indicators:
                raw = float(indicators[index_key])
                return max(0.0, min(100.0, raw))

        score = 50.0  # neutral baseline
        adjustments: list[float] = []

        # Generic demand signals
        if "demand_growth" in indicators:
            g = float(indicators["demand_growth"])
            # >15% very strong, >5% good, negative poor
            if g >= 20:
                adjustments.append(30.0)
            elif g >= 15:
                adjustments.append(20.0)
            elif g >= 5:
                adjustments.append(10.0)
            elif g < 0:
                adjustments.append(-20.0)
            else:
                adjustments.append(-5.0)

        if "capacity_utilisation" in indicators:
            cu = float(indicators["capacity_utilisation"])
            if cu >= 85:
                adjustments.append(20.0)
            elif cu >= 75:
                adjustments.append(10.0)
            elif cu < 60:
                adjustments.append(-15.0)

        if "order_book_growth" in indicators:
            ob = float(indicators["order_book_growth"])
            if ob >= 20:
                adjustments.append(20.0)
            elif ob >= 10:
                adjustments.append(10.0)
            elif ob < 0:
                adjustments.append(-10.0)

        if "volume_growth" in indicators:
            vg = float(indicators["volume_growth"])
            if vg >= 10:
                adjustments.append(10.0)
            elif vg < -5:
                adjustments.append(-10.0)

        # PMI / IIP signals
        if "pmi" in indicators:
            pmi = float(indicators["pmi"])
            if pmi >= 55:
                adjustments.append(15.0)
            elif pmi >= 50:
                adjustments.append(5.0)
            else:
                adjustments.append(-15.0)

        if adjustments:
            score = score + sum(adjustments) / max(len(adjustments), 1)

        return max(0.0, min(100.0, score))

    # ---------------------------------------------------------------------------
    # Policy score
    # ---------------------------------------------------------------------------

    def calculate_policy_score(
        self, theme_name: str, recent_announcements: list[dict]
    ) -> float:
        """
        Score 0-100 based on recent policy announcements relevant to the theme.

        Each announcement dict should have at least a 'title' and optionally 'description'.
        Keywords from theme['policy_keywords'] are matched (case-insensitive).
        Positive / negative sentiment is detected via simple word lists.
        """
        theme = self.THEMES.get(theme_name)
        if theme is None:
            return 50.0

        if not recent_announcements:
            return 50.0

        policy_keywords = [kw.lower() for kw in theme.get("policy_keywords", [])]
        if not policy_keywords:
            return 50.0

        POSITIVE_WORDS = {
            "hike", "increase", "boost", "benefit", "support", "raise", "subsidy",
            "incentive", "positive", "approved", "fund", "allocation", "promote",
            "relaxed", "reform", "growth",
        }
        NEGATIVE_WORDS = {
            "cut", "reduce", "withdraw", "ban", "penalt", "tighten", "restrict",
            "tax", "duty hike", "probe", "investigation", "negative", "freeze",
            "lower", "decrease",
        }

        positive_hits = 0
        negative_hits = 0

        for ann in recent_announcements:
            text = " ".join([
                str(ann.get("title", "")),
                str(ann.get("description", "")),
                str(ann.get("content", "")),
            ]).lower()

            if not any(kw in text for kw in policy_keywords):
                continue

            pos = sum(1 for w in POSITIVE_WORDS if w in text)
            neg = sum(1 for w in NEGATIVE_WORDS if w in text)

            if pos > neg:
                positive_hits += 1
            elif neg > pos:
                negative_hits += 1
            # Neutral if equal

        # Compute score
        total_hits = positive_hits + negative_hits
        if total_hits == 0:
            return 50.0

        # Ratio-based: pure positive → 90, pure negative → 10
        ratio = positive_hits / total_hits  # 0.0 to 1.0
        # Map [0, 1] → [10, 90]
        score = 10.0 + ratio * 80.0

        # Volume bonus: many relevant announcements is itself positive
        volume_bonus = min(10.0, total_hits * 2.0)
        score = min(100.0, score + volume_bonus)

        return round(score, 2)

    # ---------------------------------------------------------------------------
    # Exposure score
    # ---------------------------------------------------------------------------

    def calculate_exposure_score(
        self, theme_name: str, stock_scores: dict[str, float]
    ) -> float:
        """
        Return the average technical/fundamental score of theme stocks (0-100).

        Args:
            theme_name:   Theme key.
            stock_scores: {symbol: score} mapping (score 0-100).
        """
        theme = self.THEMES.get(theme_name)
        if theme is None or not theme.get("stocks"):
            return 50.0

        theme_stocks = theme["stocks"]
        available_scores: list[float] = []

        for sym in theme_stocks:
            if sym in stock_scores:
                available_scores.append(float(stock_scores[sym]))

        if not available_scores:
            logger.debug(
                "calculate_exposure_score: no stock scores found for theme '%s'",
                theme_name,
            )
            return 50.0  # neutral when no data

        return round(sum(available_scores) / len(available_scores), 2)

    # ---------------------------------------------------------------------------
    # Technical score
    # ---------------------------------------------------------------------------

    def calculate_technical_score(
        self, theme_name: str, stock_indicators: dict
    ) -> float:
        """
        Return percentage of theme stocks currently trading above their EMA50 (0-100).

        Args:
            theme_name:       Theme key.
            stock_indicators: {symbol: {"above_ema50": bool, "rsi": float, ...}}
        """
        theme = self.THEMES.get(theme_name)
        if theme is None or not theme.get("stocks"):
            return 50.0

        theme_stocks = theme["stocks"]
        total = 0
        bullish = 0

        for sym in theme_stocks:
            ind = stock_indicators.get(sym)
            if ind is None:
                continue
            total += 1
            if ind.get("above_ema50", False):
                bullish += 1

        if total == 0:
            return 50.0  # no data → neutral

        return round((bullish / total) * 100.0, 2)

    # ---------------------------------------------------------------------------
    # Score a single theme
    # ---------------------------------------------------------------------------

    def score_theme(
        self,
        theme_name: str,
        weather_score: float,
        demand_indicators: dict,
        policy_announcements: list[dict],
        stock_scores: dict[str, float],
        stock_indicators: dict,
        current_month: int,
    ) -> ThemeScore:
        """
        Compute a composite ThemeScore for a single named theme.

        Args:
            theme_name:          Key from THEMES.
            weather_score:       External weather/IMD score 0-100.
            demand_indicators:   Macro / sector demand data dict.
            policy_announcements: List of recent govt / regulatory announcement dicts.
            stock_scores:        {symbol: composite_score} for exposure calculation.
            stock_indicators:    {symbol: {above_ema50, rsi, ...}} for technical calculation.
            current_month:       Current calendar month (1-12).

        Returns:
            ThemeScore with all component scores and composite weighted score.
        """
        theme = self.THEMES.get(theme_name)
        if theme is None:
            logger.error("score_theme: unknown theme '%s'", theme_name)
            return ThemeScore(
                name=theme_name, score=0.0, status="no_trade",
                season_strength=0.0, weather_score=0.0, demand_score=0.0,
                policy_score=0.0, exposure_score=0.0, technical_score=0.0,
            )

        # ── Component calculation ───────────────────────────────────────
        season_s = self.calculate_season_strength(theme_name, current_month)

        # Weather score only counts if theme is weather sensitive
        eff_weather = weather_score if theme.get("weather_sensitive", False) else 50.0

        demand_s = self.calculate_demand_score(theme_name, demand_indicators)
        policy_s = self.calculate_policy_score(theme_name, policy_announcements)
        exposure_s = self.calculate_exposure_score(theme_name, stock_scores)
        technical_s = self.calculate_technical_score(theme_name, stock_indicators)

        # ── Composite score ─────────────────────────────────────────────
        composite = (
            season_s   * self.WEIGHTS["season"]   +
            eff_weather * self.WEIGHTS["weather"]  +
            demand_s   * self.WEIGHTS["demand"]   +
            policy_s   * self.WEIGHTS["policy"]   +
            exposure_s * self.WEIGHTS["exposure"] +
            technical_s * self.WEIGHTS["technical"]
        )
        composite = max(0.0, min(100.0, composite))

        # ── Factor narrative ────────────────────────────────────────────
        positive_factors: list[str] = []
        negative_factors: list[str] = []

        if season_s == 100.0:
            positive_factors.append("Peak season active")
        elif season_s == 50.0:
            positive_factors.append("Adjacent to peak season window")
        else:
            negative_factors.append("Off-season")

        if eff_weather >= 70:
            positive_factors.append(f"Favourable weather (score={eff_weather:.0f})")
        elif eff_weather <= 40 and theme.get("weather_sensitive", False):
            negative_factors.append(f"Adverse weather conditions (score={eff_weather:.0f})")

        if demand_s >= 65:
            positive_factors.append(f"Strong demand indicators (score={demand_s:.0f})")
        elif demand_s <= 40:
            negative_factors.append(f"Weak demand indicators (score={demand_s:.0f})")

        if policy_s >= 65:
            positive_factors.append("Supportive policy environment")
        elif policy_s <= 40:
            negative_factors.append("Adverse policy signals")

        if exposure_s >= 65:
            positive_factors.append(f"Theme stocks performing well (avg={exposure_s:.0f})")
        elif exposure_s <= 40:
            negative_factors.append(f"Weak theme stock performance (avg={exposure_s:.0f})")

        if technical_s >= 70:
            positive_factors.append(f"{technical_s:.0f}% of theme stocks above EMA50")
        elif technical_s <= 40:
            negative_factors.append(f"Only {technical_s:.0f}% of stocks above EMA50")

        return ThemeScore(
            name=theme_name,
            score=round(composite, 2),
            status=self.get_status(composite),
            season_strength=season_s,
            weather_score=eff_weather,
            demand_score=demand_s,
            policy_score=policy_s,
            exposure_score=exposure_s,
            technical_score=technical_s,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            affected_stocks=list(theme.get("stocks", [])),
        )

    # ---------------------------------------------------------------------------
    # Score all themes
    # ---------------------------------------------------------------------------

    def score_all_themes(
        self,
        weather_score: float,
        demand_indicators: dict,
        policy_announcements: list[dict],
        stock_scores: dict[str, float],
        stock_indicators: dict,
        current_month: Optional[int] = None,
    ) -> list[ThemeScore]:
        """
        Score every defined theme and return results sorted by composite score DESC.

        Args:
            weather_score:        IMD / weather quality score 0-100.
            demand_indicators:    Dict of demand/macro data.
            policy_announcements: Recent govt announcement dicts.
            stock_scores:         {symbol: score} — used for exposure.
            stock_indicators:     {symbol: {...}} — used for technical.
            current_month:        Calendar month (1-12). Defaults to today's month.

        Returns:
            List of ThemeScore objects, sorted by composite score descending.
        """
        if current_month is None:
            current_month = date.today().month

        results: list[ThemeScore] = []
        for theme_name in self.THEMES:
            try:
                ts = self.score_theme(
                    theme_name=theme_name,
                    weather_score=weather_score,
                    demand_indicators=demand_indicators,
                    policy_announcements=policy_announcements,
                    stock_scores=stock_scores,
                    stock_indicators=stock_indicators,
                    current_month=current_month,
                )
                results.append(ts)
            except Exception as exc:
                logger.exception(
                    "score_all_themes: error scoring theme '%s': %s", theme_name, exc
                )

        results.sort(key=lambda t: t.score, reverse=True)
        active = [t for t in results if t.is_active]
        logger.info(
            "score_all_themes: scored %d themes — %d active (green/yellow)",
            len(results), len(active),
        )
        return results

    # ---------------------------------------------------------------------------
    # Status helper
    # ---------------------------------------------------------------------------

    @staticmethod
    def get_status(score: float) -> str:
        """
        Map a composite 0-100 score to a traffic-light status.

        - green    : score >= 75
        - yellow   : score >= 55
        - red      : score >= 30
        - no_trade : score < 30
        """
        if score >= 75:
            return "green"
        if score >= 55:
            return "yellow"
        if score >= 30:
            return "red"
        return "no_trade"
