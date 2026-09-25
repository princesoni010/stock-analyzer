"""
Stock Screener — combines technical, fundamental, news, theme, and liquidity
scores to rank NSE/BSE candidates.

Score weights are passed via a config object so they can be tuned without
changing this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STOCKS_IN_REPORT = 20

# 150-symbol universe: Nifty 50 + Nifty Next 50 + key sector additions
UNIVERSE: list[str] = [
    # ── Nifty 50 ──────────────────────────────────────────────────────────
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "BAJFINANCE", "NESTLEIND", "WIPRO", "ULTRACEMCO",
    "HCLTECH", "POWERGRID", "NTPC", "TATAMOTORS", "ONGC", "JSWSTEEL",
    "TATASTEEL", "BAJAJFINSV", "TECHM", "GRASIM", "ADANIENT", "ADANIPORTS",
    "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "SHREECEM", "COALINDIA", "HINDALCO", "VEDL", "BPCL", "IOC", "GAIL",
    "INDUSINDBK", "UPL", "M&M", "APOLLOHOSP", "SBILIFE",
    # ── Nifty Next 50 ──────────────────────────────────────────────────────
    "ADANIGREEN", "ADANITRANS", "AMBUJACEM", "AUROPHARMA", "BANKBARODA",
    "BERGEPAINT", "BIOCON", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL",
    "CONCOR", "DABUR", "DLF", "GLAND", "GLAXO", "GODREJCP", "GODREJPROP",
    "HAVELLS", "ICICIGI", "ICICIPRULI", "INDUSTOWER", "IRCTC", "LICI",
    "LUPIN", "MARICO", "MUTHOOTFIN", "NAUKRI", "PAGEIND", "PETRONET",
    "PIDILITIND", "PNBHOUSING", "RECLTD", "SAIL", "SIEMENS", "STAR",
    "TATACONSUM", "TORNTPHARM", "TRENT", "TVSMOTORS", "VOLTAS", "WHIRLPOOL",
    "ZOMATO", "ZYDUSLIFE", "PIIND", "KANSAINER", "ALKEM", "MAXHEALTH",
    "MPHASIS", "PERSISTENT",
    # ── Key sector additions ───────────────────────────────────────────────
    "COROMANDEL", "RALLIS", "BAYER", "CHAMBAL", "GNFC",               # Agri
    "KNR", "PNC", "ASHOKA", "IRB",                                    # Infra
    "BAJAJ-AUTO", "MOTHERSON", "ASHOKLEY",                            # Auto
    "SUNTV", "ZEEL", "PVR",                                           # Media
    "JUBLFOOD", "DEVYANI", "WESTLIFE",                                 # QSR
    "LINDEINDIA", "SRF", "AAVAS", "HOMEFIRST",                        # Specialty
    "POLICYBZR", "PAYTM", "NYKAA",                                    # Fintech
    "DIXON", "AMBER", "POLYCAB", "PGEL",                              # Electronics
    "TATAPOWER", "TORNTPOWER", "CESC",                                # Power
    "METROPOLIS", "THYROCARE", "KRBL",                                # Others
    "HDFCAMC", "NIPPONLIFE", "UTIAMC",                                # AMC
    "GSPL", "MGL", "IGL",                                             # Gas
    "APLAPOLLO", "RATNAMANI",                                          # Steel Products
    "ATUL", "DEEPAKNTR", "NAVINFLUOR", "FLUOROCHEM",                  # Specialty Chem
]

# Deduplicate preserving order
_seen: set[str] = set()
_UNIVERSE_DEDUPED: list[str] = []
for _s in UNIVERSE:
    if _s not in _seen:
        _seen.add(_s)
        _UNIVERSE_DEDUPED.append(_s)
UNIVERSE = _UNIVERSE_DEDUPED


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScreeningCandidate:
    """Fully scored and ranked screening result for a single symbol."""

    symbol: str
    technical_score: float
    fundamental_score: float
    news_score: float
    theme_score: float
    liquidity_score: float
    total_score: float
    signal: str                     # 'buy' | 'watch' | 'no_trade'
    entry_low: float
    entry_high: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    reason_codes: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.signal == "buy"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "technical_score": round(self.technical_score, 2),
            "fundamental_score": round(self.fundamental_score, 2),
            "news_score": round(self.news_score, 2),
            "theme_score": round(self.theme_score, 2),
            "liquidity_score": round(self.liquidity_score, 2),
            "total_score": round(self.total_score, 2),
            "signal": self.signal,
            "entry_low": round(self.entry_low, 2),
            "entry_high": round(self.entry_high, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_1": round(self.target_1, 2),
            "target_2": round(self.target_2, 2),
            "risk_reward": round(self.risk_reward, 4),
            "reason_codes": self.reason_codes,
        }


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

class Screener:
    """
    Multi-factor stock screener.

    Score components and default weights::

        technical   : 0.30
        fundamental : 0.20
        news        : 0.20
        theme       : 0.20
        liquidity   : 0.10

    Pass a config object (or dict) to override weights and the score threshold.

    Example config keys::

        config.weights = {
            "technical": 0.30, "fundamental": 0.20, "news": 0.20,
            "theme": 0.20, "liquidity": 0.10
        }
        config.min_total_score = 50.0
        config.max_stocks = 20
    """

    UNIVERSE: list[str] = UNIVERSE

    # Default weights (sum = 1.0)
    DEFAULT_WEIGHTS: dict[str, float] = {
        "technical":   0.30,
        "fundamental": 0.20,
        "news":        0.20,
        "theme":       0.20,
        "liquidity":   0.10,
    }

    DEFAULT_MIN_SCORE: float = 50.0

    # ---------------------------------------------------------------------------
    # Technical score
    # ---------------------------------------------------------------------------

    def calculate_technical_score(self, indicators: dict) -> tuple[float, dict]:
        """
        Score technical indicators 0-100.

        Args:
            indicators: dict with keys: price, ema200, ema50, ema20, rsi,
                        macd_histogram (current), macd_histogram_prev,
                        volume, avg_volume_20d, relative_strength.

        Returns:
            (score: float, reason_codes: dict)
        """
        score = 0.0
        codes: dict[str, str] = {}

        price = float(indicators.get("price", 0))
        ema200 = float(indicators.get("ema200", 0))
        ema50 = float(indicators.get("ema50", 0))
        ema20 = float(indicators.get("ema20", 0))
        rsi = float(indicators.get("rsi", 50))
        macd_hist = float(indicators.get("macd_histogram", 0))
        macd_hist_prev = float(indicators.get("macd_histogram_prev", 0))
        volume = float(indicators.get("volume", 0))
        avg_volume_20d = float(indicators.get("avg_volume_20d", 1))
        relative_strength = float(indicators.get("relative_strength", 1.0))

        volume_ratio = (volume / avg_volume_20d) if avg_volume_20d > 0 else 0.0

        # Price vs EMAs
        if price > 0 and ema200 > 0 and price > ema200:
            score += 20
            codes["above_ema200"] = "+20"
        elif ema200 > 0:
            codes["below_ema200"] = "0"

        if price > 0 and ema50 > 0 and price > ema50:
            score += 15
            codes["above_ema50"] = "+15"
        elif ema50 > 0:
            codes["below_ema50"] = "0"

        if price > 0 and ema20 > 0 and price > ema20:
            score += 10
            codes["above_ema20"] = "+10"
        elif ema20 > 0:
            codes["below_ema20"] = "0"

        # RSI
        if 45 <= rsi <= 65:
            score += 15
            codes["rsi_momentum_zone"] = f"+15 (RSI={rsi:.1f})"
        elif rsi > 75:
            score -= 10
            codes["rsi_overbought"] = f"-10 (RSI={rsi:.1f})"
        elif rsi < 30:
            score -= 10
            codes["rsi_oversold_trap"] = f"-10 (RSI={rsi:.1f})"

        # MACD histogram
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            score += 10
            codes["macd_positive_growing"] = "+10"
        elif macd_hist > 0:
            codes["macd_positive_flat"] = "0"

        # Volume confirmation
        if volume_ratio > 1.5:
            score += 10
            codes["volume_confirmation"] = f"+10 (ratio={volume_ratio:.2f})"
        else:
            codes["volume_weak"] = f"0 (ratio={volume_ratio:.2f})"

        # Relative strength vs Nifty
        if relative_strength > 1.0:
            score += 10
            codes["outperforming_nifty"] = f"+10 (RS={relative_strength:.2f})"
        else:
            codes["underperforming_nifty"] = f"0 (RS={relative_strength:.2f})"

        score = max(0.0, min(100.0, score))
        return score, codes

    # ---------------------------------------------------------------------------
    # Fundamental score
    # ---------------------------------------------------------------------------

    def calculate_fundamental_score(self, fundamentals: dict) -> tuple[float, dict]:
        """
        Score fundamental data 0-100.

        Args:
            fundamentals: dict with keys: revenue_growth (%), profit_growth (%),
                          roe (%), debt_to_equity, ebitda_margin (%),
                          promoter_holding (%), promoter_pledge (%),
                          pe_ratio, sector_avg_pe.

        Returns:
            (score: float, reason_codes: dict)
        """
        score = 0.0
        codes: dict[str, str] = {}

        rev_growth = float(fundamentals.get("revenue_growth", 0))
        profit_growth = float(fundamentals.get("profit_growth", 0))
        roe = float(fundamentals.get("roe", 0))
        d2e = float(fundamentals.get("debt_to_equity", 999))
        ebitda_margin = float(fundamentals.get("ebitda_margin", 0))
        promoter_holding = float(fundamentals.get("promoter_holding", 0))
        promoter_pledge = float(fundamentals.get("promoter_pledge", 100))
        pe_ratio = float(fundamentals.get("pe_ratio", 999))

        # Revenue growth
        if rev_growth > 15:
            score += 20
            codes["strong_revenue_growth"] = f"+20 ({rev_growth:.1f}%)"
        else:
            codes["weak_revenue_growth"] = f"0 ({rev_growth:.1f}%)"

        # Profit growth
        if profit_growth > 15:
            score += 15
            codes["strong_profit_growth"] = f"+15 ({profit_growth:.1f}%)"
        else:
            codes["weak_profit_growth"] = f"0 ({profit_growth:.1f}%)"

        # ROE
        if roe > 15:
            score += 15
            codes["good_roe"] = f"+15 ({roe:.1f}%)"
        else:
            codes["poor_roe"] = f"0 ({roe:.1f}%)"

        # Debt-to-equity
        if d2e < 0.5:
            score += 15
            codes["low_debt"] = f"+15 (D/E={d2e:.2f})"
        elif d2e > 2.0:
            codes["high_debt"] = f"0 (D/E={d2e:.2f})"
        else:
            codes["moderate_debt"] = f"0 (D/E={d2e:.2f})"

        # EBITDA margin
        if ebitda_margin > 20:
            score += 15
            codes["strong_margin"] = f"+15 ({ebitda_margin:.1f}%)"
        else:
            codes["weak_margin"] = f"0 ({ebitda_margin:.1f}%)"

        # Promoter quality
        if promoter_holding > 50 and promoter_pledge < 5:
            score += 10
            codes["promoter_quality"] = (
                f"+10 (holding={promoter_holding:.1f}%, pledge={promoter_pledge:.1f}%)"
            )
        elif promoter_pledge >= 20:
            codes["high_pledge"] = f"0 (pledge={promoter_pledge:.1f}%)"

        # PE valuation
        if pe_ratio < 25:
            score += 10
            codes["reasonable_valuation"] = f"+10 (PE={pe_ratio:.1f})"
        elif pe_ratio > 50:
            codes["expensive_valuation"] = f"0 (PE={pe_ratio:.1f})"
        else:
            codes["moderate_valuation"] = f"0 (PE={pe_ratio:.1f})"

        score = max(0.0, min(100.0, score))
        return score, codes

    # ---------------------------------------------------------------------------
    # News score
    # ---------------------------------------------------------------------------

    def calculate_news_score(self, news_items: list[dict]) -> tuple[float, dict]:
        """
        Score news sentiment 0-100.

        Each news item dict expected keys:
            trust_score (int 1-4), sentiment ('positive'|'negative'|'neutral'),
            materiality (float 0-1).

        Trust levels:
            1 = Official (NSE/BSE/SEBI/RBI/PIB)
            2 = Reputable (ET, Mint, Moneycontrol, etc.)
            3 = Opinion/analyst
            4 = Unverified

        Rules:
            Base 50
            +20 per trust_score=1 positive (materiality > 0.7)
            -20 per trust_score=1 negative
            +10 per trust_score=2 positive
            -10 per trust_score=2 negative
            trust_score 3,4 ignored.

        Returns:
            (score: float, reason_codes: dict)
        """
        score = 50.0
        codes: dict[str, str] = {}

        official_pos = 0
        official_neg = 0
        reputable_pos = 0
        reputable_neg = 0

        for item in news_items:
            trust = int(item.get("trust_score", 4))
            if trust not in (1, 2):
                continue

            sentiment = str(item.get("sentiment", "neutral")).lower()
            materiality = float(item.get("materiality", 0.0))

            if trust == 1:
                if sentiment == "positive" and materiality > 0.7:
                    official_pos += 1
                    score += 20
                elif sentiment == "negative":
                    official_neg += 1
                    score -= 20

            elif trust == 2:
                if sentiment == "positive":
                    reputable_pos += 1
                    score += 10
                elif sentiment == "negative":
                    reputable_neg += 1
                    score -= 10

        codes["official_positive"] = str(official_pos)
        codes["official_negative"] = str(official_neg)
        codes["reputable_positive"] = str(reputable_pos)
        codes["reputable_negative"] = str(reputable_neg)

        score = max(0.0, min(100.0, score))
        return score, codes

    # ---------------------------------------------------------------------------
    # Liquidity score
    # ---------------------------------------------------------------------------

    def calculate_liquidity_score(self, candle: dict) -> tuple[float, dict]:
        """
        Score trading liquidity 0-100 based on volume and gap analysis.

        Args:
            candle: dict with keys: volume, avg_volume_20d, open, prev_close, price.

        Returns:
            (score: float, reason_codes: dict)
        """
        score = 0.0
        codes: dict[str, str] = {}

        volume = float(candle.get("volume", 0))
        avg_volume_20d = float(candle.get("avg_volume_20d", 0))
        open_price = float(candle.get("open", 0))
        prev_close = float(candle.get("prev_close", 0))

        # Volume thresholds
        if volume > 1_000_000:
            score += 75  # 50 base + 25 bonus
            codes["high_volume"] = f"+75 ({volume:,.0f} shares)"
        elif volume > 500_000:
            score += 50
            codes["adequate_volume"] = f"+50 ({volume:,.0f} shares)"
        else:
            codes["low_volume"] = f"+0 ({volume:,.0f} shares)"

        if avg_volume_20d > 500_000:
            score += 25
            codes["good_avg_volume"] = f"+25 (avg={avg_volume_20d:,.0f})"
        else:
            codes["low_avg_volume"] = f"+0 (avg={avg_volume_20d:,.0f})"

        # Gap penalty — large gaps suggest erratic liquidity / event risk
        if open_price > 0 and prev_close > 0:
            gap_pct = abs(open_price - prev_close) / prev_close * 100
            if gap_pct > 5:
                score -= 20
                codes["large_gap_penalty"] = f"-20 (gap={gap_pct:.1f}%)"
            elif gap_pct > 3:
                score -= 10
                codes["moderate_gap_penalty"] = f"-10 (gap={gap_pct:.1f}%)"

        score = max(0.0, min(100.0, score))
        return score, codes

    # ---------------------------------------------------------------------------
    # Entry level calculation
    # ---------------------------------------------------------------------------

    def calculate_entry_levels(
        self, indicators: dict, atr: float
    ) -> tuple[float, float, float, float, float]:
        """
        Derive entry zone, stop-loss, and targets from indicators and ATR.

        Args:
            indicators: dict with keys: price, support (optional), resistance (optional).
            atr:        Average True Range for the relevant timeframe.

        Returns:
            (entry_low, entry_high, stop_loss, target_1, target_2)
        """
        price = float(indicators.get("price", 0))
        support = float(indicators.get("support", 0))
        resistance = float(indicators.get("resistance", 0))

        if atr <= 0:
            # Fallback: use 1% of price as ATR proxy
            atr = price * 0.01 if price > 0 else 1.0

        # Entry zone
        if support > 0 and support < price:
            entry_low = support
        else:
            entry_low = price - 0.5 * atr

        if resistance > 0 and resistance > price:
            # Entry high should be below target; cap at resistance
            entry_high = min(resistance, price + 0.3 * atr)
        else:
            entry_high = price + 0.3 * atr

        # entry_high must be > entry_low
        if entry_high <= entry_low:
            entry_high = entry_low + 0.1 * atr

        # Stop-loss
        if support > 0:
            stop_loss = support - 0.5 * atr
        else:
            stop_loss = entry_low - 1.5 * atr

        # Targets (ensure Target 1 gives >= 2:1 R:R above entry_high)
        risk_per_share = entry_high - stop_loss
        target_1 = entry_high + 2.0 * risk_per_share
        target_2 = entry_high + 3.5 * risk_per_share

        return (
            round(entry_low, 2),
            round(entry_high, 2),
            round(stop_loss, 2),
            round(target_1, 2),
            round(target_2, 2),
        )

    # ---------------------------------------------------------------------------
    # Determine signal
    # ---------------------------------------------------------------------------

    @staticmethod
    def _determine_signal(total_score: float) -> str:
        """Map composite score to an actionable signal."""
        if total_score >= 65:
            return "buy"
        if total_score >= 50:
            return "watch"
        return "no_trade"

    # ---------------------------------------------------------------------------
    # Main screen method
    # ---------------------------------------------------------------------------

    def screen(
        self,
        indicators_map: dict,
        fundamentals_map: dict,
        news_map: dict,
        theme_scores: dict,
        candles_map: dict,
        config=None,
    ) -> list[ScreeningCandidate]:
        """
        Screen the full universe and return ranked candidates.

        Args:
            indicators_map:  {symbol: indicators_dict}
            fundamentals_map: {symbol: fundamentals_dict}
            news_map:        {symbol: [news_item_dict, ...]}
            theme_scores:    {symbol: float} — theme score for each symbol
                             (can be pre-computed by ThemeEngine and mapped per stock)
            candles_map:     {symbol: candle_dict} — latest OHLCV data
            config:          Object or dict with optional keys:
                                weights (dict), min_total_score (float),
                                max_stocks (int).

        Returns:
            List of ScreeningCandidate sorted by total_score DESC, capped at
            max_stocks (default 20).
        """
        # ── Config resolution ───────────────────────────────────────────
        if config is None:
            weights = self.DEFAULT_WEIGHTS.copy()
            min_score = self.DEFAULT_MIN_SCORE
            max_stocks = MAX_STOCKS_IN_REPORT
        elif isinstance(config, dict):
            weights = config.get("weights", self.DEFAULT_WEIGHTS)
            min_score = float(config.get("min_total_score", self.DEFAULT_MIN_SCORE))
            max_stocks = int(config.get("max_stocks", MAX_STOCKS_IN_REPORT))
        else:
            weights = getattr(config, "weights", self.DEFAULT_WEIGHTS)
            min_score = float(getattr(config, "min_total_score", self.DEFAULT_MIN_SCORE))
            max_stocks = int(getattr(config, "max_stocks", MAX_STOCKS_IN_REPORT))

        candidates: list[ScreeningCandidate] = []
        skipped = 0

        for symbol in self.UNIVERSE:
            ind = indicators_map.get(symbol)
            if ind is None:
                skipped += 1
                continue

            price = float(ind.get("price", 0))
            if price <= 0:
                skipped += 1
                continue

            # ── Component scores ────────────────────────────────────────
            tech_score, tech_codes = self.calculate_technical_score(ind)

            fund_data = fundamentals_map.get(symbol, {})
            fund_score, fund_codes = self.calculate_fundamental_score(fund_data)

            news_items = news_map.get(symbol, [])
            news_score, news_codes = self.calculate_news_score(news_items)

            t_score = float(theme_scores.get(symbol, 50.0))

            candle = candles_map.get(symbol, {})
            # Merge candle volume into candle data for liquidity
            liq_candle = {**candle, "avg_volume_20d": ind.get("avg_volume_20d", 0)}
            liq_score, liq_codes = self.calculate_liquidity_score(liq_candle)

            # ── Composite score ─────────────────────────────────────────
            total_score = (
                tech_score  * weights.get("technical",   0.30) +
                fund_score  * weights.get("fundamental", 0.20) +
                news_score  * weights.get("news",        0.20) +
                t_score     * weights.get("theme",       0.20) +
                liq_score   * weights.get("liquidity",   0.10)
            )
            total_score = max(0.0, min(100.0, total_score))

            signal = self._determine_signal(total_score)

            # ── Filter ──────────────────────────────────────────────────
            if total_score < min_score or signal == "no_trade":
                continue

            # ── Entry levels ────────────────────────────────────────────
            atr = float(ind.get("atr", price * 0.015))
            entry_low, entry_high, stop_loss, target_1, target_2 = (
                self.calculate_entry_levels(ind, atr)
            )

            risk_per_share = entry_high - stop_loss
            risk_reward = (
                (target_1 - entry_high) / risk_per_share
                if risk_per_share > 0 else 0.0
            )

            reason_codes = {
                "technical": tech_codes,
                "fundamental": fund_codes,
                "news": news_codes,
                "liquidity": liq_codes,
                "theme_score": round(t_score, 2),
            }

            candidates.append(
                ScreeningCandidate(
                    symbol=symbol,
                    technical_score=round(tech_score, 2),
                    fundamental_score=round(fund_score, 2),
                    news_score=round(news_score, 2),
                    theme_score=round(t_score, 2),
                    liquidity_score=round(liq_score, 2),
                    total_score=round(total_score, 2),
                    signal=signal,
                    entry_low=entry_low,
                    entry_high=entry_high,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    risk_reward=round(risk_reward, 4),
                    reason_codes=reason_codes,
                )
            )

        # ── Sort and cap ────────────────────────────────────────────────
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        result = candidates[:max_stocks]

        logger.info(
            "Screener.screen: universe=%d, skipped=%d, passed_filter=%d, "
            "returned=%d (min_score=%.1f)",
            len(self.UNIVERSE), skipped, len(candidates), len(result), min_score,
        )
        return result
