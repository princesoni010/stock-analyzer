"""
backtesting.py
==============
Paper-trading / backtesting service for the Bharat Market AI system.

Handles the full lifecycle of a paper trade:
  open → update (stop-out / T1 / T2 / expiry) → P&L → metrics → report
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List

import pytz

logger = logging.getLogger(__name__)

# Indian Standard Time zone
_IST = pytz.timezone("Asia/Kolkata")


class BacktestingService:
    """
    Stateless service for paper-trading / backtesting operations.

    All methods operate purely on plain Python dicts and lists; no
    database or constructor dependencies are required.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        entry: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        quantity: int,
        screening_result_id: str,
    ) -> dict:
        """
        Create and return a new paper-trade record in the 'open' state.

        Parameters
        ----------
        symbol : str
            NSE/BSE ticker symbol (e.g. ``"RELIANCE"``).
        entry : float
            Entry price for the trade.  Must be > 0.
        stop_loss : float
            Initial stop-loss price.  Must be < ``entry``.
        target_1 : float
            First profit target.  Must be > ``entry``.
        target_2 : float
            Second (final) profit target.  Must be > ``target_1``.
        quantity : int
            Number of shares/units.  Must be > 0.
        screening_result_id : str
            Foreign-key reference to the screening result that triggered
            this trade.

        Returns
        -------
        dict
            A trade record dict with the following keys:
            ``trade_id``, ``symbol``, ``entry``, ``stop_loss``,
            ``target_1``, ``target_2``, ``quantity``,
            ``screening_result_id``, ``open_time``, ``status``,
            ``outcome``, ``current_stop``, ``exit_price``,
            ``exit_time``, ``hit_t1``.

        Raises
        ------
        ValueError
            If any of the validation constraints are violated.
        """
        self._validate_trade_inputs(
            symbol=symbol,
            entry=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            quantity=quantity,
        )

        trade_id = str(uuid.uuid4())
        open_time = datetime.now(tz=timezone.utc).isoformat()

        trade: dict = {
            "trade_id": trade_id,
            "symbol": symbol,
            "entry": entry,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "quantity": quantity,
            "screening_result_id": screening_result_id,
            "open_time": open_time,
            "status": "open",
            "outcome": None,
            "current_stop": stop_loss,
            "exit_price": None,
            "exit_time": None,
            "hit_t1": False,
        }

        logger.info(
            "Opened paper trade %s for %s | entry=%.2f SL=%.2f T1=%.2f T2=%.2f qty=%d",
            trade_id,
            symbol,
            entry,
            stop_loss,
            target_1,
            target_2,
            quantity,
        )
        return trade

    def update_trade(
        self,
        trade: dict,
        current_price: float,
        current_time: datetime,
    ) -> dict:
        """
        Evaluate a new price tick against an open trade and update its state.

        Conditions are checked in strict priority order:

        1. **Stop-out** — ``current_price <= trade['current_stop']``
           Sets status → ``'closed'``, outcome → ``'stopped_out'``.

        2. **Target 2 hit** — ``current_price >= trade['target_2']``
           Sets status → ``'closed'``, outcome → ``'hit_t2'``,
           exit_price → ``trade['target_2']``.

        3. **Target 1 hit** (trade stays open) — ``current_price >= trade['target_1']``
           and ``hit_t1 == False``.
           Sets ``hit_t1 = True``, moves ``current_stop`` to breakeven
           (``trade['entry']``), outcome → ``'hit_t1'``.

        4. **Market-close expiry** — IST hour ≥ 15 and IST minute ≥ 30
           and trade still open.
           Sets status → ``'closed'``, outcome → ``'expired'``,
           exit_price → ``current_price``.

        Parameters
        ----------
        trade : dict
            The trade record (as returned by :meth:`open_trade`).
            Modified **in-place** and also returned for convenience.
        current_price : float
            Latest market price for the symbol.
        current_time : datetime
            Timestamp of the price tick.  May be tz-aware or tz-naive
            (assumed UTC when naive).

        Returns
        -------
        dict
            The (possibly updated) trade record.
        """
        if trade.get("status") != "open":
            logger.debug(
                "Trade %s is not open (status=%s); skipping update.",
                trade.get("trade_id"),
                trade.get("status"),
            )
            return trade

        # Normalise current_time to UTC-aware then convert to IST
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        current_time_ist: datetime = current_time.astimezone(_IST)
        exit_time_iso: str = current_time.isoformat()

        trade_id: str = trade.get("trade_id", "<unknown>")
        symbol: str = trade.get("symbol", "<unknown>")

        # ------------------------------------------------------------------
        # 1. Stop-out check
        # ------------------------------------------------------------------
        if current_price <= trade["current_stop"]:
            trade["status"] = "closed"
            trade["outcome"] = "stopped_out"
            trade["exit_price"] = current_price
            trade["exit_time"] = exit_time_iso
            logger.info(
                "Trade %s (%s) STOPPED OUT at %.2f (stop=%.2f)",
                trade_id,
                symbol,
                current_price,
                trade["current_stop"],
            )
            return trade

        # ------------------------------------------------------------------
        # 2. Target 2 hit
        # ------------------------------------------------------------------
        if current_price >= trade["target_2"]:
            trade["status"] = "closed"
            trade["outcome"] = "hit_t2"
            trade["exit_price"] = trade["target_2"]  # book at exact T2 price
            trade["exit_time"] = exit_time_iso
            logger.info(
                "Trade %s (%s) HIT TARGET 2 (%.2f) at tick price %.2f",
                trade_id,
                symbol,
                trade["target_2"],
                current_price,
            )
            return trade

        # ------------------------------------------------------------------
        # 3. Target 1 hit — trail stop to breakeven, stay open
        # ------------------------------------------------------------------
        if current_price >= trade["target_1"] and not trade["hit_t1"]:
            trade["hit_t1"] = True
            trade["current_stop"] = trade["entry"]  # move stop to breakeven
            trade["outcome"] = "hit_t1"
            logger.info(
                "Trade %s (%s) HIT TARGET 1 (%.2f) — stop moved to breakeven %.2f",
                trade_id,
                symbol,
                trade["target_1"],
                trade["entry"],
            )
            return trade

        # ------------------------------------------------------------------
        # 4. Market-close expiry (IST 15:30 or later)
        # ------------------------------------------------------------------
        ist_hour: int = current_time_ist.hour
        ist_minute: int = current_time_ist.minute
        if ist_hour >= 15 and ist_minute >= 30:
            trade["status"] = "closed"
            trade["outcome"] = "expired"
            trade["exit_price"] = current_price
            trade["exit_time"] = exit_time_iso
            logger.info(
                "Trade %s (%s) EXPIRED at market close %.2f (IST %02d:%02d)",
                trade_id,
                symbol,
                current_price,
                ist_hour,
                ist_minute,
            )
            return trade

        # No condition triggered — trade remains open, unchanged
        logger.debug(
            "Trade %s (%s) unchanged at price %.2f (IST %02d:%02d)",
            trade_id,
            symbol,
            current_price,
            ist_hour,
            ist_minute,
        )
        return trade

    def calculate_pnl(self, trade: dict) -> float:
        """
        Compute the realised P&L for a closed trade.

        For open trades the unrealised P&L is not considered; ``0.0`` is
        returned to avoid confusion with paper-trade accounting.

        Parameters
        ----------
        trade : dict
            A trade record dict.

        Returns
        -------
        float
            Realised P&L in Indian Rupees, rounded to 2 decimal places.
            Returns ``0.0`` if the trade is still open.
        """
        if trade.get("status") != "closed":
            logger.debug(
                "Trade %s is open; returning P&L = 0.0",
                trade.get("trade_id"),
            )
            return 0.0

        exit_price: float = trade["exit_price"]
        entry: float = trade["entry"]
        quantity: int = trade["quantity"]

        pnl: float = round((exit_price - entry) * quantity, 2)
        logger.debug(
            "Trade %s P&L: (%.2f - %.2f) × %d = ₹%.2f",
            trade.get("trade_id"),
            exit_price,
            entry,
            quantity,
            pnl,
        )
        return pnl

    def get_metrics(self, trades: List[dict]) -> dict:
        """
        Compute aggregate performance metrics over a list of trades.

        Only **closed** trades are included in the calculations.

        Parameters
        ----------
        trades : list[dict]
            Collection of trade records (open or closed).

        Returns
        -------
        dict
            A dict containing:

            * ``num_trades`` (*int*) — number of closed trades.
            * ``win_rate`` (*float*) — fraction of profitable trades.
            * ``hit_t1_rate`` (*float*) — fraction that reached T1 or T2.
            * ``hit_t2_rate`` (*float*) — fraction that reached T2.
            * ``stopped_out_rate`` (*float*) — fraction stopped out.
            * ``total_pnl`` (*float*) — sum of all P&L.
            * ``avg_pnl_per_trade`` (*float*) — mean P&L per closed trade.
            * ``avg_rr_realized`` (*float*) — mean realised risk-reward ratio.
            * ``max_drawdown`` (*float*) — largest peak-to-trough equity drop.

            Float values are rounded to 4 decimal places.
        """
        closed_trades: List[dict] = [
            t for t in trades if t.get("status") == "closed"
        ]
        num_trades: int = len(closed_trades)

        if num_trades == 0:
            logger.info("No closed trades found; returning zeroed metrics.")
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "hit_t1_rate": 0.0,
                "hit_t2_rate": 0.0,
                "stopped_out_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "avg_rr_realized": 0.0,
                "max_drawdown": 0.0,
            }

        pnl_list: List[float] = [self.calculate_pnl(t) for t in closed_trades]
        outcomes: List[str] = [t.get("outcome", "") for t in closed_trades]

        wins: int = sum(1 for p in pnl_list if p > 0)
        hit_t1_count: int = sum(
            1 for o in outcomes if o in ("hit_t1", "hit_t2")
        )
        hit_t2_count: int = sum(1 for o in outcomes if o == "hit_t2")
        stopped_count: int = sum(1 for o in outcomes if o == "stopped_out")

        total_pnl: float = sum(pnl_list)
        avg_pnl: float = total_pnl / num_trades

        # Realised R:R  = (exit_price - entry) / (entry - stop_loss)
        # Avoid division by zero when entry == stop_loss (shouldn't happen
        # due to validation, but guard defensively).
        rr_list: List[float] = []
        for t in closed_trades:
            risk: float = t["entry"] - t["stop_loss"]
            if risk != 0.0:
                rr_list.append((t["exit_price"] - t["entry"]) / risk)
            else:
                logger.warning(
                    "Trade %s has zero risk (entry == stop_loss); "
                    "skipping R:R contribution.",
                    t.get("trade_id"),
                )
        avg_rr: float = (sum(rr_list) / len(rr_list)) if rr_list else 0.0

        # Maximum drawdown — peak-to-trough decline in running equity curve
        max_drawdown: float = self._compute_max_drawdown(pnl_list)

        metrics: dict = {
            "num_trades": num_trades,
            "win_rate": round(wins / num_trades, 4),
            "hit_t1_rate": round(hit_t1_count / num_trades, 4),
            "hit_t2_rate": round(hit_t2_count / num_trades, 4),
            "stopped_out_rate": round(stopped_count / num_trades, 4),
            "total_pnl": round(total_pnl, 4),
            "avg_pnl_per_trade": round(avg_pnl, 4),
            "avg_rr_realized": round(avg_rr, 4),
            "max_drawdown": round(max_drawdown, 4),
        }

        logger.info("Metrics computed over %d closed trades.", num_trades)
        return metrics

    def generate_performance_report(self, trades: List[dict]) -> str:
        """
        Generate a human-readable performance report for a set of trades.

        Parameters
        ----------
        trades : list[dict]
            Collection of trade records (open or closed).

        Returns
        -------
        str
            A multi-line text report suitable for logging or display.
        """
        metrics: dict = self.get_metrics(trades)
        closed_trades: List[dict] = [
            t for t in trades if t.get("status") == "closed"
        ]

        # ------------------------------------------------------------------
        # Summary header
        # ------------------------------------------------------------------
        lines: List[str] = [
            "=== Paper Trade Performance Report ===",
            f"Total Trades:     {metrics['num_trades']}",
            f"Win Rate:         {metrics['win_rate'] * 100:.1f}%",
            f"Hit T1 Rate:      {metrics['hit_t1_rate'] * 100:.1f}%",
            f"Hit T2 Rate:      {metrics['hit_t2_rate'] * 100:.1f}%",
            f"Stopped Out:      {metrics['stopped_out_rate'] * 100:.1f}%",
            f"Total P&L:        ₹{metrics['total_pnl']:,.2f}",
            f"Avg P&L/Trade:    ₹{metrics['avg_pnl_per_trade']:,.2f}",
            f"Avg R:R Realized: {metrics['avg_rr_realized']:.2f}",
            f"Max Drawdown:     ₹{metrics['max_drawdown']:,.2f}",
            "===================================",
        ]

        # ------------------------------------------------------------------
        # Per-trade breakdown
        # ------------------------------------------------------------------
        if closed_trades:
            lines.append("")
            lines.append("--- Individual Trade Breakdown ---")
            lines.append(
                f"{'#':<4} {'Symbol':<15} {'Outcome':<14} {'P&L':>12}"
            )
            lines.append("-" * 47)
            for idx, trade in enumerate(closed_trades, start=1):
                pnl: float = self.calculate_pnl(trade)
                symbol: str = trade.get("symbol", "N/A")
                outcome: str = trade.get("outcome") or "N/A"
                pnl_str: str = f"₹{pnl:,.2f}"
                lines.append(
                    f"{idx:<4} {symbol:<15} {outcome:<14} {pnl_str:>12}"
                )
            lines.append("-" * 47)
            lines.append(
                f"{'':>34} {'₹' + f\"{metrics['total_pnl']:,.2f}\":>12}"
            )

        report: str = "\n".join(lines)
        logger.debug("Performance report generated:\n%s", report)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_trade_inputs(
        symbol: str,
        entry: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        quantity: int,
    ) -> None:
        """
        Validate trade parameters and raise ``ValueError`` with a
        descriptive message on the first violated constraint.

        Parameters
        ----------
        symbol : str
            Ticker symbol — must be a non-empty string.
        entry : float
            Entry price — must be > 0.
        stop_loss : float
            Stop-loss price — must be < ``entry``.
        target_1 : float
            First target — must be > ``entry``.
        target_2 : float
            Second target — must be > ``target_1``.
        quantity : int
            Trade size — must be > 0.

        Raises
        ------
        ValueError
            On the first violated constraint.
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError(
                f"'symbol' must be a non-empty string; got: {symbol!r}"
            )
        if entry <= 0:
            raise ValueError(
                f"'entry' must be greater than 0; got: {entry}"
            )
        if stop_loss >= entry:
            raise ValueError(
                f"'stop_loss' ({stop_loss}) must be less than 'entry' ({entry})."
            )
        if target_1 <= entry:
            raise ValueError(
                f"'target_1' ({target_1}) must be greater than 'entry' ({entry})."
            )
        if target_2 <= target_1:
            raise ValueError(
                f"'target_2' ({target_2}) must be greater than 'target_1' ({target_1})."
            )
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError(
                f"'quantity' must be a positive integer; got: {quantity!r}"
            )

    @staticmethod
    def _compute_max_drawdown(pnl_list: List[float]) -> float:
        """
        Compute the maximum peak-to-trough decline in a running equity curve.

        The equity curve is built by cumulatively summing ``pnl_list``.
        The drawdown at each point is defined as the difference between the
        running peak and the current equity value.

        Parameters
        ----------
        pnl_list : list[float]
            Ordered list of per-trade P&L values.

        Returns
        -------
        float
            Maximum drawdown (non-negative).  Returns ``0.0`` for an
            empty list.
        """
        if not pnl_list:
            return 0.0

        max_drawdown: float = 0.0
        peak: float = 0.0
        equity: float = 0.0

        for pnl in pnl_list:
            equity += pnl
            if equity > peak:
                peak = equity
            drawdown: float = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown
