"""
Risk Engine — pure deterministic position sizing and trade validation.

No AI, no external calls. All math is reproducible.

Output contract:
    { symbol, status, entry_low, entry_high, stop_loss, target_1, target_2,
      quantity, risk_amount, risk_reward, invalidation, warnings[] }
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskOutput:
    """Complete output for a single risk-engine calculation."""

    symbol: str
    status: str                     # 'trade' | 'no_trade'
    entry_low: float
    entry_high: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    quantity: int
    risk_amount: float              # capital * max_risk_pct
    risk_reward: float              # (target_1 - entry_high) / (entry_high - stop_loss)
    invalidation: str               # plain-english exit condition
    warnings: list[str] = field(default_factory=list)
    cost_estimate: float = 0.0      # slippage + transaction cost in INR
    slippage_estimate: float = 0.0  # slippage component alone in INR

    # Convenience helpers -------------------------------------------------

    @property
    def is_tradeable(self) -> bool:
        return self.status == "trade"

    @property
    def gross_position_value(self) -> float:
        return self.quantity * self.entry_high

    @property
    def max_loss_inr(self) -> float:
        """Worst-case loss if stop is hit (includes cost estimate)."""
        return self.quantity * (self.entry_high - self.stop_loss) + self.cost_estimate

    @property
    def potential_profit_t1(self) -> float:
        if self.target_1 <= 0:
            return 0.0
        return self.quantity * (self.target_1 - self.entry_high) - self.cost_estimate

    @property
    def potential_profit_t2(self) -> float:
        if self.target_2 is None or self.target_2 <= 0:
            return 0.0
        return self.quantity * (self.target_2 - self.entry_high) - self.cost_estimate


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Deterministic risk engine for Indian equities.

    Rules enforced:
    - Capital at risk never exceeds max_risk_pct of total capital.
    - Position value never exceeds max_position_value.
    - Minimum risk:reward ratio enforced.
    - Stop-loss must be below entry zone.
    - No duplicate trades per symbol (validated externally via validate_no_duplicate).
    - Never average down — single-entry design by contract.
    """

    def __init__(
        self,
        capital: float,
        max_risk_pct: float,
        max_position_value: float,
        min_risk_reward: float,
        slippage_pct: float = 0.001,
        transaction_cost_pct: float = 0.0005,
    ) -> None:
        """
        Args:
            capital:              Total trading capital in INR.
            max_risk_pct:         Max fraction of capital risked per trade (e.g. 0.01 = 1%).
            max_position_value:   Hard cap on gross position value in INR.
            min_risk_reward:      Minimum acceptable risk:reward ratio (e.g. 2.0).
            slippage_pct:         Expected slippage as fraction of price (default 0.1%).
            transaction_cost_pct: Brokerage + STT + other costs as fraction of turnover.
        """
        if capital <= 0:
            raise ValueError(f"capital must be > 0, got {capital}")
        if not (0 < max_risk_pct < 1):
            raise ValueError(f"max_risk_pct must be between 0 and 1, got {max_risk_pct}")
        if max_position_value <= 0:
            raise ValueError(f"max_position_value must be > 0, got {max_position_value}")
        if min_risk_reward <= 0:
            raise ValueError(f"min_risk_reward must be > 0, got {min_risk_reward}")

        self.capital = capital
        self.max_risk_pct = max_risk_pct
        self.max_position_value = max_position_value
        self.min_risk_reward = min_risk_reward
        self.slippage_pct = slippage_pct
        self.transaction_cost_pct = transaction_cost_pct

        logger.info(
            "RiskEngine initialised — capital=%.2f, risk_pct=%.2f%%, "
            "max_pos=%.2f, min_rr=%.2f",
            capital, max_risk_pct * 100, max_position_value, min_risk_reward,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        symbol: str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        target_1: float,
        target_2: Optional[float],
        atr: float,
    ) -> RiskOutput:
        """
        Calculate position size and risk metrics for a single trade candidate.

        Args:
            symbol:     NSE/BSE ticker symbol.
            entry_low:  Lower bound of buy zone.
            entry_high: Upper bound of buy zone (used for conservative sizing).
            stop_loss:  Invalidation / exit price.
            target_1:   Primary profit target.
            target_2:   Secondary (extended) profit target, or None.
            atr:        Average True Range for the relevant timeframe.

        Returns:
            RiskOutput with status='trade' if viable, 'no_trade' otherwise.
        """
        warnings: list[str] = []
        rejections: list[str] = []

        # ── Sanity checks on inputs ─────────────────────────────────────
        if entry_high <= 0:
            rejections.append(f"Invalid entry_high={entry_high:.2f} (must be > 0)")

        if entry_low <= 0:
            rejections.append(f"Invalid entry_low={entry_low:.2f} (must be > 0)")

        if entry_low > entry_high:
            rejections.append(
                f"entry_low ({entry_low:.2f}) > entry_high ({entry_high:.2f})"
            )

        # Stop must be strictly below entry_low
        if stop_loss >= entry_low:
            rejections.append(
                f"stop_loss ({stop_loss:.2f}) >= entry_low ({entry_low:.2f}) — "
                "invalid stop placement"
            )

        if target_1 <= entry_high:
            rejections.append(
                f"target_1 ({target_1:.2f}) must be above entry_high ({entry_high:.2f})"
            )

        if target_2 is not None and target_2 <= target_1:
            warnings.append(
                f"target_2 ({target_2:.2f}) is not above target_1 ({target_1:.2f}); "
                "target_2 ignored"
            )
            target_2 = None

        # Early bail-out if structural inputs are invalid
        if rejections:
            return self._no_trade(
                symbol, entry_low, entry_high, stop_loss,
                target_1, target_2, warnings + rejections,
            )

        # ── Risk per share ──────────────────────────────────────────────
        risk_per_share = entry_high - stop_loss
        if risk_per_share <= 0:
            rejections.append(
                f"risk_per_share={risk_per_share:.2f} (entry_high - stop_loss must be > 0)"
            )
            return self._no_trade(
                symbol, entry_low, entry_high, stop_loss,
                target_1, target_2, warnings + rejections,
            )

        # ── Position sizing ─────────────────────────────────────────────
        risk_amount = self.capital * self.max_risk_pct

        raw_qty = math.floor(risk_amount / risk_per_share)
        max_qty_by_position = math.floor(self.max_position_value / entry_high)

        quantity = min(raw_qty, max_qty_by_position)

        if quantity < 1:
            rejections.append(
                f"Calculated quantity={quantity} < 1 "
                f"(risk_amount={risk_amount:.2f}, risk_per_share={risk_per_share:.2f}, "
                f"max_pos={self.max_position_value:.2f})"
            )
            return self._no_trade(
                symbol, entry_low, entry_high, stop_loss,
                target_1, target_2, warnings + rejections,
            )

        if raw_qty > max_qty_by_position:
            warnings.append(
                f"Quantity capped by max_position_value: raw_qty={raw_qty} → {quantity}"
            )

        # ── Risk:Reward check ───────────────────────────────────────────
        risk_reward = (target_1 - entry_high) / risk_per_share

        if risk_reward < self.min_risk_reward:
            rejections.append(
                f"risk_reward={risk_reward:.2f} < min_required={self.min_risk_reward:.2f}"
            )
            return self._no_trade(
                symbol, entry_low, entry_high, stop_loss,
                target_1, target_2, warnings + rejections,
            )

        # ── Cost estimation ─────────────────────────────────────────────
        gross_value = quantity * entry_high
        slippage_estimate = gross_value * self.slippage_pct
        transaction_cost = gross_value * self.transaction_cost_pct
        cost_estimate = slippage_estimate + transaction_cost

        # ── ATR-based warnings ──────────────────────────────────────────
        if entry_high > 0 and atr > 0.05 * entry_high:
            warnings.append(
                f"ATR ({atr:.2f}) > 5% of price ({entry_high:.2f}) — high volatility"
            )

        if atr <= 0:
            warnings.append("ATR is zero or negative — volatility data may be missing")

        # Liquidity proxy: check if position size vs ATR is reasonable
        # (Very large ATR relative to stop implies wide stops → small qty warning)
        if quantity < 5:
            warnings.append(f"Low quantity ({quantity} shares) — consider smaller position size")

        # ── Build output ────────────────────────────────────────────────
        invalidation = f"Close below {stop_loss:.2f}"

        logger.debug(
            "[%s] trade: qty=%d entry=%.2f-%.2f sl=%.2f t1=%.2f rr=%.2f cost=%.2f",
            symbol, quantity, entry_low, entry_high, stop_loss,
            target_1, risk_reward, cost_estimate,
        )

        return RiskOutput(
            symbol=symbol,
            status="trade",
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2) if target_2 is not None else None,
            quantity=quantity,
            risk_amount=round(risk_amount, 2),
            risk_reward=round(risk_reward, 4),
            invalidation=invalidation,
            warnings=warnings,
            cost_estimate=round(cost_estimate, 2),
            slippage_estimate=round(slippage_estimate, 2),
        )

    def batch_calculate(self, candidates: list[dict]) -> list[RiskOutput]:
        """
        Process a list of candidate dicts through the risk engine.

        Each dict must have keys matching calculate() parameters.
        Dicts missing required keys are skipped with a warning.

        Args:
            candidates: List of dicts, each with:
                symbol, entry_low, entry_high, stop_loss,
                target_1, target_2 (optional), atr.

        Returns:
            List of RiskOutput objects (one per candidate, including no_trade).
        """
        required = {"symbol", "entry_low", "entry_high", "stop_loss", "target_1", "atr"}
        results: list[RiskOutput] = []

        for i, c in enumerate(candidates):
            missing = required - set(c.keys())
            if missing:
                logger.warning(
                    "batch_calculate: candidate[%d] missing keys %s — skipped", i, missing
                )
                continue

            try:
                result = self.calculate(
                    symbol=str(c["symbol"]),
                    entry_low=float(c["entry_low"]),
                    entry_high=float(c["entry_high"]),
                    stop_loss=float(c["stop_loss"]),
                    target_1=float(c["target_1"]),
                    target_2=float(c["target_2"]) if c.get("target_2") is not None else None,
                    atr=float(c["atr"]),
                )
                results.append(result)
            except Exception as exc:
                logger.exception(
                    "batch_calculate: error processing candidate[%d] symbol=%s: %s",
                    i, c.get("symbol", "?"), exc,
                )

        trade_count = sum(1 for r in results if r.is_tradeable)
        logger.info(
            "batch_calculate: processed %d candidates → %d tradeable, %d rejected",
            len(results), trade_count, len(results) - trade_count,
        )
        return results

    def validate_no_duplicate(self, symbol: str, open_trades: list[str]) -> bool:
        """
        Return True if the symbol is NOT already in open_trades (no duplicate).

        Enforces the 'never double-up on same symbol' rule.
        Comparison is case-insensitive.

        Args:
            symbol:      The symbol to check.
            open_trades: List of symbols currently in the portfolio.

        Returns:
            True  → safe to trade (no duplicate).
            False → symbol already open, reject.
        """
        normalised_symbol = symbol.strip().upper()
        normalised_open = {t.strip().upper() for t in open_trades}

        if normalised_symbol in normalised_open:
            logger.warning(
                "validate_no_duplicate: %s already in open_trades — trade blocked",
                normalised_symbol,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _no_trade(
        self,
        symbol: str,
        entry_low: float,
        entry_high: float,
        stop_loss: float,
        target_1: float,
        target_2: Optional[float],
        warnings: list[str],
    ) -> RiskOutput:
        """Build a no_trade RiskOutput with all rejection reasons in warnings."""
        logger.debug("[%s] no_trade — reasons: %s", symbol, " | ".join(warnings))
        return RiskOutput(
            symbol=symbol,
            status="no_trade",
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            quantity=0,
            risk_amount=round(self.capital * self.max_risk_pct, 2),
            risk_reward=0.0,
            invalidation=f"Close below {stop_loss:.2f}" if stop_loss > 0 else "N/A",
            warnings=warnings,
            cost_estimate=0.0,
            slippage_estimate=0.0,
        )
