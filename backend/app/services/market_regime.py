"""
services/market_regime.py
Deterministic market regime classifier for Bharat Market AI.

The classifier is purely score-based – no ML model required.  Every
intermediate calculation is exposed via MarketRegime.factors so the
result is fully explainable.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Version string – bump when scoring logic changes
# ---------------------------------------------------------------------------
CALCULATION_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MarketRegime:
    """
    Immutable result object produced by :class:`MarketRegimeService`.

    Attributes:
        regime:               Human-readable regime label.
        score:                Composite score, 0–100.
        risk_level:           ``'low'``, ``'medium'``, ``'high'``, or ``'extreme'``.
        factors:              List of factor dicts; each has
                              ``name``, ``value``, ``contribution`` (points added).
        calculation_version:  Semver string of the scoring algorithm.
        generated_at:         UTC datetime when the regime was calculated.
    """

    regime: str
    score: float
    risk_level: str
    factors: list[dict] = field(default_factory=list)
    calculation_version: str = CALCULATION_VERSION
    generated_at: datetime = field(default_factory=lambda: datetime.now(tz=_UTC))

    # Convenience
    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "score": round(self.score, 2),
            "risk_level": self.risk_level,
            "factors": self.factors,
            "calculation_version": self.calculation_version,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MarketRegimeService:
    """
    Deterministic, score-based market-regime classifier.

    The total score is built from five orthogonal factors:

    +--------------------------+--------+
    | Factor                   | Weight |
    +==========================+========+
    | Nifty trend              |   30   |
    | India VIX                |   20   |
    | Market breadth           |   20   |
    | Sector breadth           |   15   |
    | Global cues              |   15   |
    +--------------------------+--------+
    | **Maximum total**        | **100**|
    +--------------------------+--------+

    Score bands → regime labels:

    * ≥ 75  : Bullish Trend
    * 60–74 : Mildly Bullish
    * 40–59 : Range-bound
    * 25–39 : Mildly Bearish
    * 10–24 : Bearish Trend
    * < 10  : High-volatility / No-trade
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_regime(self, market_data: dict) -> MarketRegime:
        """
        Calculate the current market regime from *market_data*.

        Args:
            market_data: Dict with the following keys (all required unless
                         marked optional):

                * ``nifty_price``          – Current Nifty 50 price.
                * ``nifty_ema20``          – 20-day EMA of Nifty.
                * ``nifty_ema50``          – 50-day EMA of Nifty.
                * ``nifty_ema200``         – 200-day EMA of Nifty.
                * ``banknifty_price``      – Current Bank Nifty price (informational).
                * ``india_vix``            – India VIX value.
                * ``advance_count``        – Number of advancing stocks (NSE universe).
                * ``decline_count``        – Number of declining stocks.
                * ``sector_bullish_count`` – Count of bullish sectors (0–10).
                * ``sector_bearish_count`` – Count of bearish sectors (0–10).
                * ``global_score``         – Pre-calculated global cue score (0–100).
                * ``crude_change_pct``     – % change in crude oil (informational).
                * ``usdinr_change_pct``    – % change in USD/INR (informational).

        Returns:
            :class:`MarketRegime` instance.
        """
        factors: list[dict] = []

        # ---- 1. Nifty trend (max 30 pts) ----
        nifty_score, nifty_factor = self._score_nifty_trend(market_data)
        factors.append(nifty_factor)

        # ---- 2. VIX (max 20 pts; can be negative) ----
        vix_score, vix_factor = self._score_vix(market_data)
        factors.append(vix_factor)

        # ---- 3. Market breadth (max 20 pts) ----
        breadth_score, breadth_factor = self._score_breadth(market_data)
        factors.append(breadth_factor)

        # ---- 4. Sector breadth (max 15 pts) ----
        sector_score, sector_factor = self._score_sector_breadth(market_data)
        factors.append(sector_factor)

        # ---- 5. Global cues (max 15 pts) ----
        global_score_pts, global_factor = self._score_global(market_data)
        factors.append(global_factor)

        # ---- Total ----
        raw_score = nifty_score + vix_score + breadth_score + sector_score + global_score_pts
        # Clamp to [0, 100]
        total_score = float(max(0.0, min(100.0, raw_score)))

        regime = self._classify_regime(total_score)
        risk_level = self._classify_risk(total_score)

        logger.info(
            "MarketRegime calculated: regime=%s score=%.1f risk=%s",
            regime, total_score, risk_level,
        )

        return MarketRegime(
            regime=regime,
            score=total_score,
            risk_level=risk_level,
            factors=factors,
            calculation_version=CALCULATION_VERSION,
            generated_at=datetime.now(tz=_UTC),
        )

    # ------------------------------------------------------------------
    # Factor scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_nifty_trend(data: dict) -> tuple[float, dict]:
        """
        Nifty trend factor (max 30 pts).

        Rules (mutually exclusive, highest match wins):
        * +30 → price > EMA20 > EMA50 > EMA200 (full bull alignment)
        * +20 → price > EMA50 (price above medium-term trend)
        * +10 → price > EMA200 (price above long-term trend)
        *  +0 → otherwise
        """
        price = float(data.get("nifty_price", 0) or 0)
        ema20 = float(data.get("nifty_ema20", 0) or 0)
        ema50 = float(data.get("nifty_ema50", 0) or 0)
        ema200 = float(data.get("nifty_ema200", 0) or 0)

        if price > ema20 > ema50 > ema200:
            pts = 30.0
            label = "Full bull alignment (price > EMA20 > EMA50 > EMA200)"
        elif price > ema50:
            pts = 20.0
            label = "Price above EMA50"
        elif price > ema200:
            pts = 10.0
            label = "Price above EMA200"
        else:
            pts = 0.0
            label = "Price below all key EMAs"

        return pts, {
            "name": "Nifty Trend",
            "value": {
                "price": price,
                "ema20": ema20,
                "ema50": ema50,
                "ema200": ema200,
                "label": label,
            },
            "contribution": pts,
            "max_contribution": 30,
        }

    @staticmethod
    def _score_vix(data: dict) -> tuple[float, dict]:
        """
        India VIX factor (max 20 pts; can go negative).

        * +20  → VIX < 15  (low fear, bullish environment)
        * +15  → VIX 15–18
        * +8   → VIX 18–22
        *  0   → VIX 22–28 (elevated anxiety)
        * -10  → VIX > 28  (high fear, risk-off)
        """
        vix = float(data.get("india_vix", 20) or 20)

        if vix < 15:
            pts = 20.0
            label = "VIX < 15: low fear"
        elif vix < 18:
            pts = 15.0
            label = "VIX 15–18: moderate"
        elif vix < 22:
            pts = 8.0
            label = "VIX 18–22: elevated"
        elif vix <= 28:
            pts = 0.0
            label = "VIX 22–28: high anxiety"
        else:
            pts = -10.0
            label = "VIX > 28: extreme fear"

        return pts, {
            "name": "India VIX",
            "value": {"vix": vix, "label": label},
            "contribution": pts,
            "max_contribution": 20,
        }

    @staticmethod
    def _score_breadth(data: dict) -> tuple[float, dict]:
        """
        Market breadth factor (max 20 pts).

        advance_ratio = advance / (advance + decline)
        pts = advance_ratio * 20
        """
        advances = float(data.get("advance_count", 0) or 0)
        declines = float(data.get("decline_count", 0) or 0)
        total = advances + declines

        if total <= 0:
            ratio = 0.5  # neutral assumption when data unavailable
        else:
            ratio = advances / total

        pts = ratio * 20.0

        return pts, {
            "name": "Market Breadth",
            "value": {
                "advances": int(advances),
                "declines": int(declines),
                "advance_ratio": round(ratio, 4),
            },
            "contribution": round(pts, 4),
            "max_contribution": 20,
        }

    @staticmethod
    def _score_sector_breadth(data: dict) -> tuple[float, dict]:
        """
        Sector breadth factor (max 15 pts).

        sector_ratio = bullish_sectors / (bullish + bearish)
        pts = sector_ratio * 15
        """
        bullish = float(data.get("sector_bullish_count", 0) or 0)
        bearish = float(data.get("sector_bearish_count", 0) or 0)
        total = bullish + bearish

        if total <= 0:
            ratio = 0.5
        else:
            ratio = bullish / total

        pts = ratio * 15.0

        return pts, {
            "name": "Sector Breadth",
            "value": {
                "sector_bullish": int(bullish),
                "sector_bearish": int(bearish),
                "sector_ratio": round(ratio, 4),
            },
            "contribution": round(pts, 4),
            "max_contribution": 15,
        }

    @staticmethod
    def _score_global(data: dict) -> tuple[float, dict]:
        """
        Global cues factor (max 15 pts).

        global_score (0–100) is pre-calculated by the caller (e.g. from
        S&P 500, crude oil, USD/INR, US 10Y movements).

        pts = (global_score / 100) * 15
        """
        g_score = float(data.get("global_score", 50) or 50)
        g_score = max(0.0, min(100.0, g_score))  # clamp to [0, 100]
        pts = (g_score / 100.0) * 15.0

        return pts, {
            "name": "Global Cues",
            "value": {
                "global_score": g_score,
                "crude_change_pct": data.get("crude_change_pct"),
                "usdinr_change_pct": data.get("usdinr_change_pct"),
            },
            "contribution": round(pts, 4),
            "max_contribution": 15,
        }

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_regime(score: float) -> str:
        """Map a total score to a regime label."""
        if score >= 75:
            return "Bullish Trend"
        if score >= 60:
            return "Mildly Bullish"
        if score >= 40:
            return "Range-bound"
        if score >= 25:
            return "Mildly Bearish"
        if score >= 10:
            return "Bearish Trend"
        return "High-volatility / No-trade"

    @staticmethod
    def _classify_risk(score: float) -> str:
        """Map a total score to a risk level."""
        if score > 70:
            return "low"
        if score >= 50:
            return "medium"
        if score >= 30:
            return "high"
        return "extreme"
