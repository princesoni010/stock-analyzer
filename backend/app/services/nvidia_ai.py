"""
nvidia_ai.py
============
Production-quality NVIDIA NIM AI service for the Bharat Market AI system.

Responsibilities
----------------
- Generate AI-powered morning reports using the NVIDIA NIM API (meta/llama3-70b-instruct).
- Generate per-stock explanations and news summaries.
- Validate all AI responses for hallucinated symbols, guaranteed-return language,
  and missing source attribution.
- Apply safety filters to all textual output before returning to callers.
- Fall back gracefully to a deterministic template report when the NIM API is unavailable.

Prompt version: 1.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import openai

from app.config import settings

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PromptVersion: str = "1.0"

_GUARANTEED_RETURN_PATTERNS: list[str] = [
    "guaranteed",
    "will definitely",
    "certain profit",
    "100%",
]

_DEFINITIVE_FUTURE_PATTERNS: list[tuple[str, str]] = [
    (r"\bwill rise\b", "may rise"),
    (r"\bwill fall\b", "may fall"),
    (r"\bwill reach\b", "may reach"),
    (r"\bwill increase\b", "may increase"),
    (r"\bwill decrease\b", "may decrease"),
    (r"\bwill gain\b", "may gain"),
    (r"\bwill drop\b", "may drop"),
    (r"\bwill surge\b", "may surge"),
    (r"\bwill decline\b", "may decline"),
]

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class AIReportRequest:
    """Input payload for the AI morning-report generation pipeline.

    Attributes
    ----------
    report_date:
        ISO-8601 date string for the report (e.g., ``"2026-09-26"``).
    market_summary:
        High-level market regime, indices, breadth, and volatility data.
    themes:
        Active market themes (sector rotation, event-driven, macro, etc.).
    candidates:
        Ranked list of stock candidates produced by the scoring pipeline.
        Each dict **must** contain at minimum a ``"symbol"`` key.
    source_records:
        Raw source records (news articles, filings) that back the candidates.
    risk_fields:
        Quantitative risk metrics (VaR, drawdown, beta, etc.).
    language:
        BCP-47 language code for the report output. Defaults to English.
    """

    report_date: str
    market_summary: dict[str, Any]
    themes: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    source_records: list[dict[str, Any]]
    risk_fields: list[dict[str, Any]]
    language: str = "en"


@dataclass
class AIReportResponse:
    """Structured AI-generated morning report.

    Attributes
    ----------
    executive_summary:
        One-paragraph executive summary of the day's outlook.
    market_section:
        Narrative description of current market conditions.
    theme_section:
        Narrative description of active investment themes.
    stock_sections:
        Per-stock explanations keyed by ``symbol``.
    warnings:
        Compliance and data-quality warnings to surface to the reader.
    source_ids:
        Identifiers of source records cited in the report.
    uncertainty_notes:
        Explicit statements of uncertainty or data limitations.
    """

    executive_summary: str
    market_section: str
    theme_section: str
    stock_sections: list[dict[str, Any]]
    warnings: list[str]
    source_ids: list[str]
    uncertainty_notes: list[str]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class NvidiaAIService:
    """Async client for the NVIDIA NIM API used to generate Bharat Market AI reports.

    Parameters
    ----------
    api_key:
        NVIDIA NIM API key.
    base_url:
        Base URL for the NIM API endpoint.
    model:
        Model identifier (e.g. ``"meta/llama3-70b-instruct"``).
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of retry attempts before falling back to the deterministic
        template report.
    """

    def __init__(
        self,
        api_key: str = settings.NVIDIA_API_KEY,
        base_url: str = settings.NVIDIA_BASE_URL,
        model: str = settings.NVIDIA_MODEL,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self._api_key: str = api_key
        self._base_url: str = base_url
        self._model: str = model
        self._timeout: int = timeout
        self._max_retries: int = max_retries

        self._client: openai.AsyncOpenAI = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout),
        )

        logger.info(
            "NvidiaAIService initialised | model=%s | base_url=%s | "
            "timeout=%ds | max_retries=%d",
            self._model,
            self._base_url,
            self._timeout,
            self._max_retries,
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Return the system-level instruction prompt for the NIM model.

        The prompt enforces strict grounding, JSON output format,
        source attribution, and compliance constraints.

        Returns
        -------
        str
            Fully formatted system prompt string.
        """
        return (
            f"You are a professional Indian equity market analyst AI assistant "
            f"(Prompt Version: {PromptVersion}).\n\n"
            "## STRICT RULES — YOU MUST FOLLOW ALL OF THEM\n\n"
            "1. **Use only the data provided in the user payload.** "
            "Do NOT invent, extrapolate, or hallucinate any price targets, "
            "stock symbols, analyst ratings, source identifiers, or financial figures "
            "that are not explicitly present in the input JSON.\n\n"
            "2. **Never guarantee returns.** Do not use language such as "
            "'guaranteed', 'will definitely', 'certain profit', or '100%' in any form. "
            "All forward-looking statements must be explicitly conditional "
            "(e.g., 'may rise', 'could outperform', 'subject to market risk').\n\n"
            "3. **Separate facts from opinions.** Clearly label analytical opinions "
            "as opinions and grounded data points as facts.\n\n"
            "4. **Cite sources for all material claims.** Every claim of substance "
            "must reference at least one source_id from the provided source_records list. "
            "Do not cite sources that are not present in the payload.\n\n"
            "5. **Output strictly valid JSON** matching the schema below. "
            "Do not include prose, markdown fences, or commentary outside the JSON object.\n\n"
            "## OUTPUT SCHEMA\n\n"
            "{\n"
            '  "executive_summary": "<string>",\n'
            '  "market_section": "<string>",\n'
            '  "theme_section": "<string>",\n'
            '  "stock_sections": [\n'
            '    {"symbol": "<string>", "explanation": "<string>", "score": <number>, '
            '"source_ids": ["<string>"]}\n'
            "  ],\n"
            '  "warnings": ["<string>"],\n'
            '  "source_ids": ["<string>"],\n'
            '  "uncertainty_notes": ["<string>"]\n'
            "}\n\n"
            "Be factual, concise, and avoid unnecessary hedging beyond what is required "
            "for compliance. Language of the report must match the `language` field in the payload."
        )

    def _build_user_prompt(self, request: AIReportRequest) -> str:
        """Serialise the full report request as a structured JSON user message.

        Parameters
        ----------
        request:
            The report request dataclass instance.

        Returns
        -------
        str
            JSON-encoded user prompt string.
        """
        payload: dict[str, Any] = asdict(request)
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError) as exc:
            logger.error("Failed to serialise AIReportRequest to JSON: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------

    def _validate_response(
        self,
        raw: dict[str, Any],
        request: AIReportRequest,
    ) -> AIReportResponse:
        """Validate and coerce the raw API response dict into ``AIReportResponse``.

        Validation rules
        ----------------
        - All required fields from ``AIReportResponse`` must be present.
        - ``stock_sections`` must not contain symbols absent from ``request.candidates``.
        - No field may contain guaranteed-return language.
        - ``source_ids`` must be a non-empty list.

        Parameters
        ----------
        raw:
            Parsed JSON dict from the NIM API response.
        request:
            Original report request (used to cross-check candidate symbols).

        Returns
        -------
        AIReportResponse
            Validated response object.

        Raises
        ------
        ValueError
            If any validation rule is violated, with a human-readable reason.
        """
        required_fields: list[str] = [
            "executive_summary",
            "market_section",
            "theme_section",
            "stock_sections",
            "warnings",
            "source_ids",
            "uncertainty_notes",
        ]

        # --- 1. Structural completeness check ---
        missing: list[str] = [f for f in required_fields if f not in raw]
        if missing:
            raise ValueError(
                f"AI response missing required fields: {missing}"
            )

        # --- 2. Type coercion checks ---
        if not isinstance(raw["stock_sections"], list):
            raise ValueError("'stock_sections' must be a JSON array.")
        if not isinstance(raw["source_ids"], list):
            raise ValueError("'source_ids' must be a JSON array.")
        if not isinstance(raw["warnings"], list):
            raise ValueError("'warnings' must be a JSON array.")
        if not isinstance(raw["uncertainty_notes"], list):
            raise ValueError("'uncertainty_notes' must be a JSON array.")

        # --- 3. Source attribution check ---
        if not raw["source_ids"]:
            raise ValueError(
                "AI response contains no source_ids. "
                "All material claims must cite at least one source."
            )

        # --- 4. Symbol hallucination check ---
        allowed_symbols: set[str] = {
            str(c.get("symbol", "")).upper()
            for c in request.candidates
            if c.get("symbol")
        }
        for section in raw["stock_sections"]:
            section_symbol: str = str(section.get("symbol", "")).upper()
            if section_symbol and section_symbol not in allowed_symbols:
                raise ValueError(
                    f"AI response contains hallucinated symbol '{section_symbol}' "
                    f"not present in request candidates {allowed_symbols}."
                )

        # --- 5. Guaranteed-return language check (all text fields) ---
        text_fields_to_check: list[str] = [
            str(raw.get("executive_summary", "")),
            str(raw.get("market_section", "")),
            str(raw.get("theme_section", "")),
        ]
        for section in raw["stock_sections"]:
            text_fields_to_check.append(str(section.get("explanation", "")))
        for warning in raw["warnings"]:
            text_fields_to_check.append(str(warning))

        combined_text: str = " ".join(text_fields_to_check).lower()
        for pattern in _GUARANTEED_RETURN_PATTERNS:
            if pattern.lower() in combined_text:
                raise ValueError(
                    f"AI response contains prohibited guaranteed-return language: '{pattern}'."
                )

        return AIReportResponse(
            executive_summary=str(raw["executive_summary"]),
            market_section=str(raw["market_section"]),
            theme_section=str(raw["theme_section"]),
            stock_sections=list(raw["stock_sections"]),
            warnings=list(raw["warnings"]),
            source_ids=list(raw["source_ids"]),
            uncertainty_notes=list(raw["uncertainty_notes"]),
        )

    # ------------------------------------------------------------------
    # Safety filter
    # ------------------------------------------------------------------

    def _safety_filter(self, text: str) -> str:
        """Remove or soften prohibited language from a text string.

        Processing steps
        ----------------
        1. Remove any line that contains guaranteed-return language.
        2. Replace definitive future-tense predictions with conditional equivalents.

        Parameters
        ----------
        text:
            Raw text string to sanitise.

        Returns
        -------
        str
            Sanitised text string.
        """
        # Step 1 — Remove lines with guaranteed-return language
        filtered_lines: list[str] = []
        for line in text.splitlines():
            line_lower = line.lower()
            if any(pat in line_lower for pat in _GUARANTEED_RETURN_PATTERNS):
                logger.warning(
                    "Safety filter removed guaranteed-return line: %s",
                    line[:120],
                )
                continue
            filtered_lines.append(line)
        filtered_text: str = "\n".join(filtered_lines)

        # Step 2 — Soften definitive future-tense predictions
        for pattern, replacement in _DEFINITIVE_FUTURE_PATTERNS:
            filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)

        return filtered_text

    def _apply_safety_filter_to_response(
        self, response: AIReportResponse
    ) -> AIReportResponse:
        """Apply ``_safety_filter`` to all text fields of an ``AIReportResponse``.

        Parameters
        ----------
        response:
            The response object whose text fields should be sanitised.

        Returns
        -------
        AIReportResponse
            A new response object with all text fields filtered.
        """
        filtered_sections: list[dict[str, Any]] = []
        for section in response.stock_sections:
            sec_copy = dict(section)
            if "explanation" in sec_copy:
                sec_copy["explanation"] = self._safety_filter(str(sec_copy["explanation"]))
            filtered_sections.append(sec_copy)

        return AIReportResponse(
            executive_summary=self._safety_filter(response.executive_summary),
            market_section=self._safety_filter(response.market_section),
            theme_section=self._safety_filter(response.theme_section),
            stock_sections=filtered_sections,
            warnings=[self._safety_filter(w) for w in response.warnings],
            source_ids=response.source_ids,
            uncertainty_notes=[self._safety_filter(n) for n in response.uncertainty_notes],
        )

    # ------------------------------------------------------------------
    # Core report generation
    # ------------------------------------------------------------------

    async def generate_morning_report(
        self, request: AIReportRequest
    ) -> AIReportResponse:
        """Generate a structured AI morning report via the NVIDIA NIM API.

        The method retries up to ``max_retries`` times on transient API errors
        or validation failures, using exponential backoff. If all retries are
        exhausted, it falls back to the deterministic template report.

        Parameters
        ----------
        request:
            Fully populated report request payload.

        Returns
        -------
        AIReportResponse
            AI-generated (or fallback) morning report.
        """
        # Compute a content-stable hash for traceability
        request_json: str = json.dumps(asdict(request), sort_keys=True, default=str)
        input_hash: str = hashlib.sha256(request_json.encode()).hexdigest()[:16]

        logger.info(
            "generate_morning_report START | model=%s | prompt_version=%s | "
            "report_date=%s | input_hash=%s",
            self._model,
            PromptVersion,
            request.report_date,
            input_hash,
        )

        system_prompt: str = self._build_system_prompt()
        user_prompt: str = self._build_user_prompt(request)

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            start_ts: float = time.monotonic()
            try:
                logger.debug(
                    "NIM API call attempt %d/%d | input_hash=%s",
                    attempt,
                    self._max_retries,
                    input_hash,
                )
                completion = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )

                latency_ms: float = (time.monotonic() - start_ts) * 1000
                usage = completion.usage

                logger.info(
                    "NIM API call SUCCESS | attempt=%d | latency_ms=%.1f | "
                    "prompt_tokens=%s | completion_tokens=%s | total_tokens=%s | "
                    "input_hash=%s",
                    attempt,
                    latency_ms,
                    usage.prompt_tokens if usage else "n/a",
                    usage.completion_tokens if usage else "n/a",
                    usage.total_tokens if usage else "n/a",
                    input_hash,
                )

                raw_content: str = completion.choices[0].message.content or ""
                raw_dict: dict[str, Any] = json.loads(raw_content)

                validated: AIReportResponse = self._validate_response(raw_dict, request)
                filtered: AIReportResponse = self._apply_safety_filter_to_response(validated)

                logger.info(
                    "generate_morning_report COMPLETE | input_hash=%s | "
                    "source_ids=%d | warnings=%d",
                    input_hash,
                    len(filtered.source_ids),
                    len(filtered.warnings),
                )
                return filtered

            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "Validation/parse error on attempt %d/%d | input_hash=%s | error=%s",
                    attempt,
                    self._max_retries,
                    input_hash,
                    exc,
                )

            except openai.APIError as exc:
                last_error = exc
                logger.warning(
                    "NIM API error on attempt %d/%d | input_hash=%s | error=%s",
                    attempt,
                    self._max_retries,
                    input_hash,
                    exc,
                )

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.error(
                    "Unexpected error on attempt %d/%d | input_hash=%s | error=%s",
                    attempt,
                    self._max_retries,
                    input_hash,
                    exc,
                    exc_info=True,
                )

            if attempt < self._max_retries:
                backoff_seconds: float = 2.0 ** (attempt - 1)
                logger.info(
                    "Retrying in %.1fs | attempt=%d | input_hash=%s",
                    backoff_seconds,
                    attempt,
                    input_hash,
                )
                await asyncio.sleep(backoff_seconds)

        logger.error(
            "All %d retries exhausted | input_hash=%s | last_error=%s — "
            "falling back to deterministic template report.",
            self._max_retries,
            input_hash,
            last_error,
        )
        return self.generate_fallback_report(request)

    # ------------------------------------------------------------------
    # Per-stock explanation
    # ------------------------------------------------------------------

    async def generate_stock_explanation(
        self,
        symbol: str,
        score_breakdown: dict[str, Any],
        indicators: dict[str, Any],
    ) -> str:
        """Generate a 2-3 sentence explanation for why a stock scored well.

        Parameters
        ----------
        symbol:
            NSE/BSE ticker symbol (e.g., ``"RELIANCE"``).
        score_breakdown:
            Scoring sub-components produced by the quant pipeline.
        indicators:
            Technical and fundamental indicator values for the stock.

        Returns
        -------
        str
            Plain-text explanation (safety-filtered).
        """
        user_prompt: str = json.dumps(
            {
                "task": "stock_explanation",
                "symbol": symbol,
                "score_breakdown": score_breakdown,
                "indicators": indicators,
                "instructions": (
                    "Write exactly 2-3 sentences explaining why this stock scored well "
                    "based solely on the data provided. "
                    "Do not invent data. Do not guarantee returns. "
                    "Use conditional language for all forward-looking statements. "
                    "Return plain text only — no JSON, no markdown."
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        system_prompt: str = (
            "You are a concise Indian equity market analyst. "
            "Explain why a stock scored well based only on the provided data. "
            "Never guarantee returns. Separate facts from opinions. Plain text only."
        )

        logger.debug("generate_stock_explanation | symbol=%s", symbol)

        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=256,
            )
            raw_text: str = completion.choices[0].message.content or ""
            filtered_text: str = self._safety_filter(raw_text.strip())
            logger.info(
                "generate_stock_explanation SUCCESS | symbol=%s | chars=%d",
                symbol,
                len(filtered_text),
            )
            return filtered_text

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "generate_stock_explanation FAILED | symbol=%s | error=%s",
                symbol,
                exc,
                exc_info=True,
            )
            return (
                f"{symbol} scored well based on the provided quantitative indicators. "
                "Detailed AI explanation is temporarily unavailable."
            )

    # ------------------------------------------------------------------
    # Fallback report
    # ------------------------------------------------------------------

    def generate_fallback_report(self, request: AIReportRequest) -> AIReportResponse:
        """Generate a deterministic template-based report when the NIM API is unavailable.

        The fallback uses only data already present in ``request`` — no AI inference.
        Top-3 candidates are selected by the ``"score"`` key (descending), with a
        safe default of 0 for missing scores.

        Parameters
        ----------
        request:
            The original report request payload.

        Returns
        -------
        AIReportResponse
            Template-based report with a compliance warning about AI unavailability.
        """
        logger.info(
            "generate_fallback_report | report_date=%s | candidates=%d",
            request.report_date,
            len(request.candidates),
        )

        # --- Market summary extraction ---
        regime: str = str(
            request.market_summary.get("regime")
            or request.market_summary.get("market_regime")
            or request.market_summary.get("trend")
            or "unknown"
        )
        nifty_level: str = str(request.market_summary.get("nifty50", "N/A"))
        sensex_level: str = str(request.market_summary.get("sensex", "N/A"))
        breadth: str = str(
            request.market_summary.get("breadth")
            or request.market_summary.get("advance_decline", "N/A")
        )
        volatility: str = str(
            request.market_summary.get("volatility")
            or request.market_summary.get("india_vix", "N/A")
        )

        # --- Top-3 candidates ---
        sorted_candidates: list[dict[str, Any]] = sorted(
            request.candidates,
            key=lambda c: float(c.get("score", 0) or 0),
            reverse=True,
        )
        top_3: list[dict[str, Any]] = sorted_candidates[:3]

        # --- Active themes ---
        theme_names: list[str] = [
            str(t.get("name") or t.get("theme") or t.get("title", "Unnamed Theme"))
            for t in request.themes
        ]

        # --- Source IDs ---
        source_ids: list[str] = [
            str(s.get("id") or s.get("source_id") or s.get("record_id", ""))
            for s in request.source_records
            if s.get("id") or s.get("source_id") or s.get("record_id")
        ]

        # --- Candidate names for summary ---
        top_symbols: list[str] = [str(c.get("symbol", "N/A")) for c in top_3]
        top_scores: list[str] = [str(c.get("score", "N/A")) for c in top_3]

        executive_summary: str = (
            f"Bharat Market AI Morning Report — {request.report_date}. "
            f"Market regime: {regime}. "
            f"Nifty 50: {nifty_level} | Sensex: {sensex_level}. "
            f"Top-ranked candidates for today: {', '.join(top_symbols) or 'None'}. "
            "This report was generated using a deterministic template because the AI "
            "explanation service is temporarily unavailable. All data is sourced from "
            "the quantitative scoring pipeline without AI inference."
        )

        market_section: str = (
            f"**Market Regime:** {regime}\n"
            f"**Nifty 50:** {nifty_level}\n"
            f"**Sensex:** {sensex_level}\n"
            f"**Market Breadth (Advance/Decline):** {breadth}\n"
            f"**Volatility (India VIX):** {volatility}\n"
        )

        theme_section: str = (
            "**Active Themes:**\n"
            + (
                "\n".join(f"- {name}" for name in theme_names)
                if theme_names
                else "- No active themes identified."
            )
        )

        stock_sections: list[dict[str, Any]] = []
        for candidate in top_3:
            sym: str = str(candidate.get("symbol", "N/A"))
            score: Any = candidate.get("score", "N/A")
            sector: str = str(candidate.get("sector", "N/A"))
            explanation: str = (
                f"{sym} has a quantitative score of {score} (sector: {sector}). "
                "Detailed AI explanation is unavailable; refer to the score breakdown "
                "for factor-level attribution. All investments carry risk."
            )
            stock_sections.append(
                {
                    "symbol": sym,
                    "score": score,
                    "explanation": explanation,
                    "source_ids": source_ids[:3],  # attach top-3 source refs
                }
            )

        warnings: list[str] = [
            "AI explanation temporarily unavailable. Showing deterministic analysis.",
            "This report does not constitute financial advice. Past performance is not "
            "indicative of future results. All investments carry risk.",
        ]

        uncertainty_notes: list[str] = [
            "Generated without AI — deterministic template only.",
            "Factor weights and scores are produced by the quantitative pipeline and "
            "have not been validated by an AI model in this report.",
        ]

        return AIReportResponse(
            executive_summary=executive_summary,
            market_section=market_section,
            theme_section=theme_section,
            stock_sections=stock_sections,
            warnings=warnings,
            source_ids=source_ids,
            uncertainty_notes=uncertainty_notes,
        )

    # ------------------------------------------------------------------
    # News summary
    # ------------------------------------------------------------------

    async def generate_news_summary(
        self, articles: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Summarise a list of news articles with event classification and materiality rating.

        The NIM model is instructed to:
        - Summarise the articles concisely.
        - Classify the event type: earnings | regulatory | macro |
          sector-specific | geopolitical.
        - Separate factual claims from opinion claims.
        - Rate materiality: high | medium | low.
        - Cite source_ids for all claims.

        Parameters
        ----------
        articles:
            List of article dicts, each expected to contain at minimum
            ``"id"``, ``"title"``, and ``"content"`` (or ``"text"``) keys.

        Returns
        -------
        dict
            Keys: ``summary``, ``event_type``, ``materiality``,
            ``fact_claims``, ``opinion_claims``, ``source_ids``.
            Falls back to a safe default dict on failure.
        """
        _SAFE_FALLBACK: dict[str, Any] = {
            "summary": "News summary temporarily unavailable.",
            "event_type": "unknown",
            "materiality": "low",
            "fact_claims": [],
            "opinion_claims": [],
            "source_ids": [],
        }

        if not articles:
            logger.warning("generate_news_summary called with empty articles list.")
            return _SAFE_FALLBACK

        system_prompt: str = (
            "You are a financial news analyst for the Indian equity market. "
            "Analyse the provided news articles and return strictly valid JSON "
            "matching the schema below. Do not include text outside the JSON object.\n\n"
            "## RULES\n"
            "1. Separate facts (verifiable data) from opinions (analyst views, predictions).\n"
            "2. Classify the event type as exactly one of: "
            "earnings | regulatory | macro | sector-specific | geopolitical.\n"
            "3. Rate materiality as exactly one of: high | medium | low.\n"
            "4. Cite source_ids from the provided article list for all claims.\n"
            "5. Never invent data.\n\n"
            "## OUTPUT SCHEMA\n"
            "{\n"
            '  "summary": "<string>",\n'
            '  "event_type": "<earnings|regulatory|macro|sector-specific|geopolitical>",\n'
            '  "materiality": "<high|medium|low>",\n'
            '  "fact_claims": ["<string>"],\n'
            '  "opinion_claims": ["<string>"],\n'
            '  "source_ids": ["<string>"]\n'
            "}"
        )

        user_prompt: str = json.dumps(
            {"articles": articles},
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        logger.debug(
            "generate_news_summary | articles=%d",
            len(articles),
        )

        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            raw_content: str = completion.choices[0].message.content or ""
            parsed: dict[str, Any] = json.loads(raw_content)

            # Validate required keys exist with correct types
            required: dict[str, type] = {
                "summary": str,
                "event_type": str,
                "materiality": str,
                "fact_claims": list,
                "opinion_claims": list,
                "source_ids": list,
            }
            for key, expected_type in required.items():
                if key not in parsed:
                    raise ValueError(f"Missing key in news summary response: '{key}'")
                if not isinstance(parsed[key], expected_type):
                    raise ValueError(
                        f"Key '{key}' expected {expected_type.__name__}, "
                        f"got {type(parsed[key]).__name__}"
                    )

            # Validate controlled-vocabulary fields
            valid_event_types: set[str] = {
                "earnings",
                "regulatory",
                "macro",
                "sector-specific",
                "geopolitical",
            }
            if parsed["event_type"] not in valid_event_types:
                logger.warning(
                    "Unexpected event_type '%s' — keeping but flagging.",
                    parsed["event_type"],
                )

            valid_materiality: set[str] = {"high", "medium", "low"}
            if parsed["materiality"] not in valid_materiality:
                logger.warning(
                    "Unexpected materiality '%s' — defaulting to 'low'.",
                    parsed["materiality"],
                )
                parsed["materiality"] = "low"

            # Apply safety filter to summary
            parsed["summary"] = self._safety_filter(str(parsed["summary"]))

            logger.info(
                "generate_news_summary SUCCESS | articles=%d | event_type=%s | "
                "materiality=%s | source_ids=%d",
                len(articles),
                parsed["event_type"],
                parsed["materiality"],
                len(parsed["source_ids"]),
            )
            return parsed

        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "generate_news_summary parse/validation error | articles=%d | error=%s",
                len(articles),
                exc,
            )
            return _SAFE_FALLBACK

        except openai.APIError as exc:
            logger.error(
                "generate_news_summary NIM API error | articles=%d | error=%s",
                len(articles),
                exc,
                exc_info=True,
            )
            return _SAFE_FALLBACK

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "generate_news_summary unexpected error | articles=%d | error=%s",
                len(articles),
                exc,
                exc_info=True,
            )
            return _SAFE_FALLBACK
