"""
Bharat Market AI — Telegram Service
====================================
Provides the TelegramService class that manages all Telegram bot interactions,
including morning reports, intraday alerts, command handling, quiet-hours
enforcement, per-symbol alert cooldowns, and MarkdownV2 message formatting.

Dependencies:
    pip install python-telegram-bot[job-queue] pytz
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, time
from typing import Any, Optional

import pytz
from telegram import Bot, Message, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone constant
# ---------------------------------------------------------------------------
IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# MarkdownV2 special characters that must be escaped
# ---------------------------------------------------------------------------
_MD2_SPECIAL = r"_*[]()~`>#+-=|{}.!"


class TelegramService:
    """
    High-level Telegram bot service for Bharat Market AI.

    Handles:
    - Morning report broadcasts
    - Intraday symbol alerts with cooldown / quiet-hours gating
    - Interactive command responses (/start, /morning, /market, …)
    - MarkdownV2 safe formatting helpers
    - Long-message splitting

    Parameters
    ----------
    token : str
        Telegram bot token from BotFather.
    allowed_chat_ids : list[str]
        Whitelist of chat / user IDs permitted to interact with the bot.
    max_message_length : int
        Maximum characters per outgoing Telegram message (default 4096).
    alert_cooldown_minutes : int
        Minimum minutes between successive alerts for the same symbol.
    quiet_hours_start : str
        Start of quiet hours in ``HH:MM`` (IST). Default ``'22:00'``.
    quiet_hours_end : str
        End of quiet hours in ``HH:MM`` (IST). Default ``'07:00'``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        token: str,
        allowed_chat_ids: list[str],
        max_message_length: int = 4096,
        alert_cooldown_minutes: int = 60,
        quiet_hours_start: str = "22:00",
        quiet_hours_end: str = "07:00",
    ) -> None:
        self.token: str = token
        self.allowed_chat_ids: list[str] = [str(c) for c in allowed_chat_ids]
        self.max_message_length: int = max_message_length
        self.alert_cooldown_minutes: int = alert_cooldown_minutes
        self.quiet_hours_start: str = quiet_hours_start
        self.quiet_hours_end: str = quiet_hours_end

        # Per-symbol last-alert timestamp for cooldown tracking
        self.last_alert: dict[str, datetime] = {}

        # Build the python-telegram-bot Application
        self.application: Application = (
            ApplicationBuilder().token(token).build()
        )
        self.bot: Bot = self.application.bot

        logger.info(
            "TelegramService initialised | allowed_chats=%s | cooldown=%dmin | "
            "quiet=%s–%s",
            self.allowed_chat_ids,
            self.alert_cooldown_minutes,
            self.quiet_hours_start,
            self.quiet_hours_end,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_quiet_hours(self) -> bool:
        """
        Return ``True`` if the current IST wall-clock time falls inside the
        configured quiet-hours window (which may wrap midnight).

        Examples
        --------
        Quiet 22:00–07:00 at 23:30 IST → True
        Quiet 22:00–07:00 at 12:00 IST → False
        """
        now_ist: datetime = datetime.now(tz=IST)
        current: time = now_ist.time().replace(second=0, microsecond=0)

        start_h, start_m = (int(x) for x in self.quiet_hours_start.split(":"))
        end_h, end_m = (int(x) for x in self.quiet_hours_end.split(":"))
        t_start = time(start_h, start_m)
        t_end = time(end_h, end_m)

        if t_start <= t_end:
            # Simple range — does NOT wrap midnight
            return t_start <= current <= t_end
        else:
            # Wraps midnight: quiet if current >= start OR current <= end
            return current >= t_start or current <= t_end

    def _is_allowed(self, chat_id: str) -> bool:
        """Return ``True`` when *chat_id* is in the allowed-chat whitelist."""
        return str(chat_id) in self.allowed_chat_ids

    def _escape_markdown(self, text: str) -> str:
        """
        Escape all MarkdownV2 special characters in *text* so it can be
        embedded inside a MarkdownV2-parsed Telegram message without breaking
        the formatting.

        Special characters escaped: ``_ * [ ] ( ) ~ ` > # + - = | { } . !``
        """
        return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))

    def _split_message(self, text: str) -> list[str]:
        """
        Split *text* into chunks whose length does not exceed
        ``self.max_message_length``.

        The algorithm tries to split at newline boundaries first; if a single
        line exceeds the limit it falls back to splitting at word boundaries;
        as a last resort it hard-splits the line.

        Parameters
        ----------
        text : str
            The full message text to split.

        Returns
        -------
        list[str]
            Ordered list of chunks, each ≤ ``max_message_length`` characters.
        """
        limit: int = self.max_message_length
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0

        lines = text.split("\n")
        for line in lines:
            # +1 for the newline we'll re-join with
            line_with_nl = line + "\n"

            if current_len + len(line_with_nl) > limit:
                # Flush current buffer
                if current:
                    chunks.append("".join(current).rstrip("\n"))
                    current = []
                    current_len = 0

                # The line itself might exceed the limit
                while len(line_with_nl) > limit:
                    # Try to split at last space within limit
                    segment = line_with_nl[:limit]
                    split_at = segment.rfind(" ")
                    if split_at == -1:
                        # No space found — hard split
                        split_at = limit
                    chunks.append(line_with_nl[:split_at].rstrip())
                    line_with_nl = line_with_nl[split_at:].lstrip(" ")

            current.append(line_with_nl)
            current_len += len(line_with_nl)

        if current:
            chunks.append("".join(current).rstrip("\n"))

        return [c for c in chunks if c]  # remove empty strings

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = ParseMode.MARKDOWN_V2,
    ) -> list[int]:
        """
        Send a (possibly long) text message to *chat_id*.

        The message is split into ≤ ``max_message_length`` chunks if needed.
        Each chunk is sent sequentially; on a ``TelegramError`` it retries
        once before logging and continuing.

        Parameters
        ----------
        chat_id : str
            Destination chat / user ID.
        text : str
            Full message text (MarkdownV2 by default).
        parse_mode : str
            Telegram parse mode; defaults to ``MarkdownV2``.

        Returns
        -------
        list[int]
            Telegram message IDs of every successfully sent chunk.

        Raises
        ------
        PermissionError
            If *chat_id* is not in the allowed whitelist.
        """
        if not self._is_allowed(chat_id):
            logger.warning("Blocked message to unauthorised chat_id=%s", chat_id)
            raise PermissionError(
                f"chat_id '{chat_id}' is not in the allowed list."
            )

        chunks = self._split_message(text)
        message_ids: list[int] = []

        for idx, chunk in enumerate(chunks, start=1):
            logger.debug(
                "Sending chunk %d/%d to chat_id=%s (%d chars)",
                idx,
                len(chunks),
                chat_id,
                len(chunk),
            )
            sent: Optional[Message] = await self._send_with_retry(
                chat_id=chat_id,
                text=chunk,
                parse_mode=parse_mode,
            )
            if sent is not None:
                message_ids.append(sent.message_id)

        return message_ids

    async def _send_with_retry(
        self,
        chat_id: str,
        text: str,
        parse_mode: str,
        _attempt: int = 1,
    ) -> Optional[Message]:
        """
        Internal helper: send a single chunk, retrying once on failure.

        Parameters
        ----------
        chat_id : str
            Destination chat ID.
        text : str
            Message chunk.
        parse_mode : str
            Telegram parse mode.
        _attempt : int
            Current attempt number (max 2).

        Returns
        -------
        Optional[Message]
            The sent ``telegram.Message`` or ``None`` on persistent failure.
        """
        try:
            return await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except TelegramError as exc:
            if _attempt < 2:
                logger.warning(
                    "TelegramError on attempt %d for chat_id=%s: %s — retrying…",
                    _attempt,
                    chat_id,
                    exc,
                )
                return await self._send_with_retry(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    _attempt=_attempt + 1,
                )
            logger.error(
                "TelegramError persisted after retry for chat_id=%s: %s",
                chat_id,
                exc,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Morning report
    # ------------------------------------------------------------------

    async def send_morning_report(
        self, chat_id: str, report: dict[str, Any]
    ) -> list[int]:
        """
        Format and send a full morning market report.

        Parameters
        ----------
        chat_id : str
            Destination chat ID.
        report : dict
            Report payload expected to contain:
            - ``date`` (str)
            - ``data_cutoff`` (str | datetime)
            - ``market_summary`` (dict)
            - ``themes`` (list[dict])
            - ``candidates`` (list[dict])
            - ``risk_warnings`` (list[str])
            - ``sources`` (list[dict] with keys ``url``, ``timestamp``)

        Returns
        -------
        list[int]
            Message IDs of sent chunks.
        """
        date_str: str = str(report.get("date", datetime.now(tz=IST).strftime("%d %b %Y")))
        cutoff_raw = report.get("data_cutoff", "N/A")
        cutoff_str: str = (
            cutoff_raw.strftime("%d %b %Y %H:%M IST")
            if isinstance(cutoff_raw, datetime)
            else str(cutoff_raw)
        )

        lines: list[str] = [
            "🌅 *Bharat Market AI — Morning Report*",
            "",
            f"📅 *Date:* {self._escape_markdown(date_str)}",
            f"🕐 *Data Cutoff:* {self._escape_markdown(cutoff_str)}",
            "",
            self._format_market_section(report.get("market_summary", {})),
            "",
            self._format_theme_section(report.get("themes", [])),
            "",
            "📊 *Top Stocks for Today*",
        ]

        candidates: list[dict] = report.get("candidates", [])[:5]
        if candidates:
            for rank, candidate in enumerate(candidates, start=1):
                lines.append(f"\n*#{rank}*")
                lines.append(self._format_stock_section(candidate))
        else:
            lines.append("_No qualified candidates today\\._")

        # Risk warnings
        risk_warnings: list[str] = report.get("risk_warnings", [])
        if risk_warnings:
            lines.append("")
            lines.append("⚠️ *Risk Warnings*")
            for warn in risk_warnings:
                lines.append(f"• {self._escape_markdown(warn)}")

        # Source links
        sources: list[dict] = report.get("sources", [])
        if sources:
            lines.append("")
            lines.append("🔗 *Sources*")
            for src in sources:
                url = src.get("url", "")
                ts = src.get("timestamp", "")
                ts_str = (
                    ts.strftime("%H:%M IST") if isinstance(ts, datetime) else str(ts)
                )
                escaped_url = self._escape_markdown(url)
                escaped_ts = self._escape_markdown(ts_str)
                lines.append(f"• {escaped_url} \\({escaped_ts}\\)")

        lines.append("")
        lines.append(
            "_Not investment advice\\. Research only\\._"
        )

        text = "\n".join(lines)
        logger.info("Sending morning report to chat_id=%s", chat_id)
        return await self.send_message(chat_id=chat_id, text=text)

    # ------------------------------------------------------------------
    # Intraday alert
    # ------------------------------------------------------------------

    async def send_alert(
        self, chat_id: str, symbol: str, alert_data: dict[str, Any]
    ) -> Optional[int]:
        """
        Send an intraday alert for *symbol*, subject to cooldown and quiet hours.

        Parameters
        ----------
        chat_id : str
            Destination chat ID.
        symbol : str
            NSE/BSE ticker symbol (e.g. ``'RELIANCE'``).
        alert_data : dict
            Alert payload with keys:
            - ``trigger_reason`` (str)
            - ``entry_price`` (float)
            - ``stop_loss`` (float)
            - ``target_1`` (float)
            - ``target_2`` (float)
            - ``timestamp`` (datetime | str, optional)

        Returns
        -------
        Optional[int]
            Telegram message ID if sent, or ``None`` if skipped.
        """
        now_ist: datetime = datetime.now(tz=IST)

        # Quiet-hours gate
        if self._is_quiet_hours():
            logger.info(
                "Alert for %s suppressed — quiet hours (%s–%s IST)",
                symbol,
                self.quiet_hours_start,
                self.quiet_hours_end,
            )
            return None

        # Cooldown gate
        last: Optional[datetime] = self.last_alert.get(symbol)
        if last is not None:
            elapsed_minutes = (now_ist - last).total_seconds() / 60
            if elapsed_minutes < self.alert_cooldown_minutes:
                logger.info(
                    "Alert for %s suppressed — cooldown (%.1f/%d min elapsed)",
                    symbol,
                    elapsed_minutes,
                    self.alert_cooldown_minutes,
                )
                return None

        # Format alert
        ts_raw = alert_data.get("timestamp", now_ist)
        ts_str: str = (
            ts_raw.strftime("%d %b %Y %H:%M:%S IST")
            if isinstance(ts_raw, datetime)
            else str(ts_raw)
        )

        entry = alert_data.get("entry_price", "N/A")
        sl = alert_data.get("stop_loss", "N/A")
        t1 = alert_data.get("target_1", "N/A")
        t2 = alert_data.get("target_2", "N/A")
        reason = alert_data.get("trigger_reason", "Signal triggered")

        def fmt(val: Any) -> str:
            """Format price value safely."""
            try:
                return f"₹{float(val):,.2f}"
            except (TypeError, ValueError):
                return str(val)

        text = "\n".join(
            [
                f"🚨 *Intraday Alert — {self._escape_markdown(symbol)}*",
                "",
                f"📌 *Trigger:* {self._escape_markdown(reason)}",
                "",
                f"🟢 *Entry:* {self._escape_markdown(fmt(entry))}",
                f"🔴 *Stop Loss:* {self._escape_markdown(fmt(sl))}",
                f"🎯 *Target 1:* {self._escape_markdown(fmt(t1))}",
                f"🎯 *Target 2:* {self._escape_markdown(fmt(t2))}",
                "",
                f"🕐 *Time:* {self._escape_markdown(ts_str)}",
                "",
                "_Not investment advice\\. Research only\\._",
            ]
        )

        message_ids = await self.send_message(chat_id=chat_id, text=text)
        if message_ids:
            self.last_alert[symbol] = now_ist
            logger.info(
                "Alert sent for %s to chat_id=%s (msg_id=%d)",
                symbol,
                chat_id,
                message_ids[0],
            )
            return message_ids[0]

        return None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_stock_section(self, candidate: dict[str, Any]) -> str:
        """
        Format a single stock candidate for MarkdownV2 display.

        Parameters
        ----------
        candidate : dict
            Stock candidate with keys: ``symbol``, ``company_name``,
            ``total_score``, ``entry_price``, ``stop_loss``, ``target_1``,
            ``target_2``, ``technical_rationale``, ``risk_reward``.

        Returns
        -------
        str
            MarkdownV2-formatted string block.
        """

        def fp(val: Any) -> str:
            """Format a price, escaping for MarkdownV2."""
            try:
                return self._escape_markdown(f"₹{float(val):,.2f}")
            except (TypeError, ValueError):
                return self._escape_markdown(str(val))

        def fv(val: Any, decimals: int = 2) -> str:
            try:
                return self._escape_markdown(f"{float(val):.{decimals}f}")
            except (TypeError, ValueError):
                return self._escape_markdown(str(val))

        symbol = self._escape_markdown(str(candidate.get("symbol", "N/A")))
        company = self._escape_markdown(str(candidate.get("company_name", "N/A")))
        score = fv(candidate.get("total_score", 0), decimals=1)
        entry = fp(candidate.get("entry_price"))
        sl = fp(candidate.get("stop_loss"))
        t1 = fp(candidate.get("target_1"))
        t2 = fp(candidate.get("target_2"))
        rationale = self._escape_markdown(
            str(candidate.get("technical_rationale", "N/A"))
        )
        rr = fv(candidate.get("risk_reward", 0), decimals=2)

        return "\n".join(
            [
                f"📈 *{symbol}* — {company}",
                f"⭐ Score: *{score}*/10",
                f"🟢 Entry: {entry}  |  🔴 SL: {sl}",
                f"🎯 T1: {t1}  |  T2: {t2}",
                f"📐 R:R \\= {rr}",
                f"🔍 {rationale}",
            ]
        )

    def _format_market_section(self, regime: dict[str, Any]) -> str:
        """
        Format a market regime summary block for MarkdownV2.

        Parameters
        ----------
        regime : dict
            Keys: ``regime_name``, ``regime_score``, ``nifty_pct_change``,
            ``breadth``, ``risk_level``.

        Returns
        -------
        str
            MarkdownV2-formatted string.
        """
        regime_name = str(regime.get("regime_name", "Neutral")).lower()
        if "bull" in regime_name:
            emoji = "🟢"
        elif "bear" in regime_name:
            emoji = "🔴"
        else:
            emoji = "🟡"

        def fv(val: Any, decimals: int = 2, suffix: str = "") -> str:
            try:
                return self._escape_markdown(f"{float(val):.{decimals}f}{suffix}")
            except (TypeError, ValueError):
                return self._escape_markdown(str(val))

        name_escaped = self._escape_markdown(str(regime.get("regime_name", "Neutral")))
        score = fv(regime.get("regime_score", 0), decimals=1)
        nifty_chg = fv(regime.get("nifty_pct_change", 0), decimals=2, suffix="%")
        breadth = self._escape_markdown(str(regime.get("breadth", "N/A")))
        risk = self._escape_markdown(str(regime.get("risk_level", "Medium")))

        return "\n".join(
            [
                f"{emoji} *Market Regime*",
                f"Regime: *{name_escaped}*  \\(Score: {score}\\)",
                f"NIFTY Change: {nifty_chg}",
                f"Breadth: {breadth}",
                f"Risk Level: {risk}",
            ]
        )

    def _format_theme_section(self, themes: list[dict[str, Any]]) -> str:
        """
        Format active investment themes for MarkdownV2.

        Parameters
        ----------
        themes : list[dict]
            Each theme dict: ``name``, ``score``, ``top_stocks`` (list[str]).

        Returns
        -------
        str
            MarkdownV2-formatted string. Limited to 5 themes.
        """
        if not themes:
            return "🔖 *Active Themes*\n_No active themes identified\\._"

        lines = ["🔖 *Active Themes*"]
        for theme in themes[:5]:
            name = self._escape_markdown(str(theme.get("name", "Unknown")))
            score_raw = theme.get("score", 0)
            try:
                score_str = self._escape_markdown(f"{float(score_raw):.1f}")
            except (TypeError, ValueError):
                score_str = self._escape_markdown(str(score_raw))
            top_stocks: list[str] = theme.get("top_stocks", [])
            stocks_str = self._escape_markdown(", ".join(top_stocks) if top_stocks else "—")
            lines.append(f"• *{name}* \\(Score: {score_str}\\) — {stocks_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    async def handle_command(
        self,
        chat_id: str,
        command: str,
        args: list[str],
        services: dict[str, Any],
    ) -> str:
        """
        Dispatch a bot command and return the MarkdownV2 response text.

        Parameters
        ----------
        chat_id : str
            The originating chat ID (for permission checks).
        command : str
            Command string without the leading ``/`` (e.g. ``'start'``).
        args : list[str]
            Positional arguments following the command.
        services : dict
            Service callables keyed by name:
            - ``'morning_report'`` → async callable() → dict
            - ``'market_data'``   → async callable() → dict
            - ``'theme_engine'``  → async callable() → list[dict]
            - ``'screener'``      → async callable() → list[dict]
            - ``'stock_data'``    → async callable(symbol) → dict
            - ``'paper_trades'``  → async callable() → list[dict]
            - ``'backtest'``      → async callable() → dict
            - ``'settings'``      → callable() → dict

        Returns
        -------
        str
            MarkdownV2-formatted response string.
        """
        cmd = command.lower().strip().lstrip("/")

        if cmd == "start":
            return (
                "👋 *Welcome to Bharat Market AI\\!*\n"
                "\n"
                "I provide AI\\-powered Indian stock market research\\.\n"
                "\n"
                "*Features:*\n"
                "• 🌅 Morning market reports\n"
                "• 🚨 Intraday momentum alerts\n"
                "• 🔍 Theme\\-based stock screening\n"
                "• 📊 Market regime analysis\n"
                "• 📝 Paper trading simulation\n"
                "• 📈 Backtesting performance metrics\n"
                "\n"
                "Type /help to see all available commands\\."
            )

        elif cmd == "morning":
            try:
                report_fn = services.get("morning_report")
                if report_fn is None:
                    return "⚠️ Morning report service is not available\\."
                report: dict = await report_fn()
                market_sec = self._format_market_section(
                    report.get("market_summary", {})
                )
                theme_sec = self._format_theme_section(report.get("themes", []))
                candidates = report.get("candidates", [])[:3]
                stock_lines: list[str] = []
                for rank, c in enumerate(candidates, start=1):
                    stock_lines.append(f"\n*#{rank}*\n{self._format_stock_section(c)}")
                stocks_sec = "".join(stock_lines) if stock_lines else "_No candidates\\._"
                return (
                    "🌅 *Latest Morning Report*\n\n"
                    + market_sec
                    + "\n\n"
                    + theme_sec
                    + "\n\n📊 *Top Picks*\n"
                    + stocks_sec
                )
            except Exception as exc:
                logger.error("Error fetching morning report: %s", exc, exc_info=True)
                return f"❌ Failed to fetch morning report: {self._escape_markdown(str(exc))}"

        elif cmd == "market":
            try:
                market_fn = services.get("market_data")
                if market_fn is None:
                    return "⚠️ Market data service is not available\\."
                market: dict = await market_fn()
                return self._format_market_section(market)
            except Exception as exc:
                logger.error("Error fetching market data: %s", exc, exc_info=True)
                return f"❌ Failed to fetch market data: {self._escape_markdown(str(exc))}"

        elif cmd == "themes":
            try:
                theme_fn = services.get("theme_engine")
                if theme_fn is None:
                    return "⚠️ Theme engine service is not available\\."
                themes: list[dict] = await theme_fn()
                return self._format_theme_section(themes)
            except Exception as exc:
                logger.error("Error fetching themes: %s", exc, exc_info=True)
                return f"❌ Failed to fetch themes: {self._escape_markdown(str(exc))}"

        elif cmd == "watchlist":
            try:
                screener_fn = services.get("screener")
                if screener_fn is None:
                    return "⚠️ Screener service is not available\\."
                results: list[dict] = await screener_fn()
                if not results:
                    return "📋 *Watchlist*\n_No stocks on the watchlist currently\\._"
                lines = ["📋 *Watchlist — Top Screener Results*\n"]
                for rank, stock in enumerate(results[:10], start=1):
                    sym = self._escape_markdown(str(stock.get("symbol", "N/A")))
                    co = self._escape_markdown(str(stock.get("company_name", "")))
                    sc = stock.get("total_score", 0)
                    try:
                        sc_str = self._escape_markdown(f"{float(sc):.1f}")
                    except (TypeError, ValueError):
                        sc_str = self._escape_markdown(str(sc))
                    lines.append(f"{rank}\\. *{sym}* {co} — Score: {sc_str}")
                return "\n".join(lines)
            except Exception as exc:
                logger.error("Error fetching watchlist: %s", exc, exc_info=True)
                return f"❌ Failed to fetch watchlist: {self._escape_markdown(str(exc))}"

        elif cmd == "stock":
            if not args:
                return "⚠️ Usage: /stock SYMBOL \\(e\\.g\\. /stock RELIANCE\\)"
            symbol = args[0].upper()
            try:
                stock_fn = services.get("stock_data")
                if stock_fn is None:
                    return "⚠️ Stock data service is not available\\."
                data: dict = await stock_fn(symbol)
                return self._format_stock_section(data)
            except Exception as exc:
                logger.error(
                    "Error fetching stock data for %s: %s", symbol, exc, exc_info=True
                )
                return (
                    f"❌ Failed to fetch data for "
                    f"{self._escape_markdown(symbol)}: "
                    f"{self._escape_markdown(str(exc))}"
                )

        elif cmd == "papertrade":
            try:
                pt_fn = services.get("paper_trades")
                if pt_fn is None:
                    return "⚠️ Paper trade service is not available\\."
                trades: list[dict] = await pt_fn()
                if not trades:
                    return "📝 *Paper Trades*\n_No open paper trades\\._"
                lines = ["📝 *Open Paper Trades*\n"]
                for trade in trades:
                    sym = self._escape_markdown(str(trade.get("symbol", "N/A")))
                    entry = trade.get("entry_price", 0)
                    cmp = trade.get("current_price", 0)
                    try:
                        pnl = ((float(cmp) - float(entry)) / float(entry)) * 100
                        pnl_str = self._escape_markdown(f"{pnl:+.2f}%")
                        entry_str = self._escape_markdown(f"₹{float(entry):,.2f}")
                        cmp_str = self._escape_markdown(f"₹{float(cmp):,.2f}")
                    except (TypeError, ValueError, ZeroDivisionError):
                        pnl_str = "N/A"
                        entry_str = self._escape_markdown(str(entry))
                        cmp_str = self._escape_markdown(str(cmp))
                    lines.append(
                        f"• *{sym}* Entry: {entry_str} → CMP: {cmp_str} "
                        f"\\({pnl_str}\\)"
                    )
                return "\n".join(lines)
            except Exception as exc:
                logger.error("Error fetching paper trades: %s", exc, exc_info=True)
                return f"❌ Failed to fetch paper trades: {self._escape_markdown(str(exc))}"

        elif cmd == "results":
            try:
                bt_fn = services.get("backtest")
                if bt_fn is None:
                    return "⚠️ Backtest service is not available\\."
                metrics: dict = await bt_fn()
                total = self._escape_markdown(
                    str(metrics.get("total_trades", "N/A"))
                )
                win_rate_raw = metrics.get("win_rate", 0)
                try:
                    win_rate = self._escape_markdown(
                        f"{float(win_rate_raw):.1f}%"
                    )
                except (TypeError, ValueError):
                    win_rate = self._escape_markdown(str(win_rate_raw))
                pf_raw = metrics.get("profit_factor", 0)
                try:
                    pf = self._escape_markdown(f"{float(pf_raw):.2f}")
                except (TypeError, ValueError):
                    pf = self._escape_markdown(str(pf_raw))
                dd_raw = metrics.get("max_drawdown", 0)
                try:
                    dd = self._escape_markdown(f"{float(dd_raw):.2f}%")
                except (TypeError, ValueError):
                    dd = self._escape_markdown(str(dd_raw))
                sharpe_raw = metrics.get("sharpe_ratio", 0)
                try:
                    sharpe = self._escape_markdown(f"{float(sharpe_raw):.2f}")
                except (TypeError, ValueError):
                    sharpe = self._escape_markdown(str(sharpe_raw))
                period = self._escape_markdown(
                    str(metrics.get("period", "N/A"))
                )
                return "\n".join(
                    [
                        "📈 *Backtesting Performance Metrics*",
                        "",
                        f"🗓 Period: {period}",
                        f"📊 Total Trades: {total}",
                        f"✅ Win Rate: {win_rate}",
                        f"💹 Profit Factor: {pf}",
                        f"📉 Max Drawdown: {dd}",
                        f"📐 Sharpe Ratio: {sharpe}",
                    ]
                )
            except Exception as exc:
                logger.error("Error fetching backtest results: %s", exc, exc_info=True)
                return f"❌ Failed to fetch results: {self._escape_markdown(str(exc))}"

        elif cmd == "settings":
            esc = self._escape_markdown
            return "\n".join(
                [
                    "⚙️ *Bot Settings*",
                    "",
                    f"🔕 Quiet Hours: {esc(self.quiet_hours_start)} – {esc(self.quiet_hours_end)} IST",
                    f"⏱ Alert Cooldown: {esc(str(self.alert_cooldown_minutes))} minutes",
                    f"📏 Max Message Length: {esc(str(self.max_message_length))} chars",
                    f"👥 Allowed Chats: {esc(str(len(self.allowed_chat_ids)))}",
                    f"🤫 Quiet Hours Active: {esc(str(self._is_quiet_hours()))}",
                ]
            )

        elif cmd == "help":
            return "\n".join(
                [
                    "📖 *Bharat Market AI — Commands*",
                    "",
                    "/start — Welcome message and feature overview",
                    "/morning — Latest morning market report",
                    "/market — Current market regime and NIFTY summary",
                    "/themes — Active investment themes with top stocks",
                    "/watchlist — Top 10 screener results",
                    "/stock SYMBOL — Detailed analysis for a specific stock",
                    "/papertrade — Open paper trade positions and P&L",
                    "/results — Backtesting performance metrics",
                    "/settings — View current bot configuration",
                    "/help — Show this help message",
                    "",
                    "_All data is for research purposes only\\._",
                ]
            )

        else:
            return (
                f"❓ Unknown command: `{self._escape_markdown(cmd)}`\\.\n"
                "Type /help to see available commands\\."
            )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def register_handlers(self, services: dict[str, Any]) -> None:
        """
        Register all command handlers.
        """
        def _make_handler(cmd_name: str):

            """Create an async Telegram CommandHandler callback for *cmd_name*."""

            async def _handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                if update.effective_chat is None or update.effective_message is None:
                    return
                chat_id = str(update.effective_chat.id)
                if not self._is_allowed(chat_id):
                    logger.warning(
                        "Unauthorised access attempt from chat_id=%s for /%s",
                        chat_id,
                        cmd_name,
                    )
                    try:
                        await update.effective_message.reply_text(
                            "⛔ You are not authorised to use this bot.",
                        )
                    except TelegramError as exc:
                        logger.error("Could not send auth error: %s", exc)
                    return

                args: list[str] = context.args or []  # type: ignore[assignment]
                logger.info(
                    "Command /%s from chat_id=%s args=%s",
                    cmd_name,
                    chat_id,
                    args,
                )

                try:
                    response_text = await self.handle_command(
                        chat_id=chat_id,
                        command=cmd_name,
                        args=args,
                        services=services,
                    )
                    await self.send_message(
                        chat_id=chat_id,
                        text=response_text,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                except PermissionError as exc:
                    logger.error("PermissionError for chat_id=%s: %s", chat_id, exc)
                except Exception as exc:
                    logger.error(
                        "Unhandled error in /%s handler: %s",
                        cmd_name,
                        exc,
                        exc_info=True,
                    )
                    try:
                        await self.send_message(
                            chat_id=chat_id,
                            text=(
                                f"❌ An error occurred while processing "
                                f"`/{self._escape_markdown(cmd_name)}`\\."
                            ),
                        )
                    except Exception:
                        pass  # Best-effort error notification

            return _handler

        # ----------------------------------------------------------------
        # Register all commands
        # ----------------------------------------------------------------
        _commands = [
            "start",
            "morning",
            "market",
            "themes",
            "watchlist",
            "stock",
            "papertrade",
            "results",
            "settings",
            "help",
        ]

        for cmd in _commands:
            self.application.add_handler(
                CommandHandler(cmd, _make_handler(cmd))
            )
            logger.debug("Registered handler for /%s", cmd)

        logger.info("Telegram command handlers registered.")
