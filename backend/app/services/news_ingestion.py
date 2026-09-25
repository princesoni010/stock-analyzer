"""
News Ingestion Service — multi-source news collection, trust scoring,
symbol extraction, event classification, materiality scoring,
and deduplication.

Sources ranked by trust:
    1 = Official  (NSE, BSE, SEBI, RBI, PIB)
    2 = Reputable (ET, Mint, Moneycontrol, BS, FE, Reuters, Bloomberg)
    3 = Opinion   (Analyst reports, blogs)
    4 = Unverified (anything else)
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 20.0

# ---------------------------------------------------------------------------
# Trust level mapping
# ---------------------------------------------------------------------------

# Maps domain substrings to trust score (lower = more trusted)
_TRUST_DOMAIN_MAP: dict[str, int] = {
    # Trust 1 — Official
    "nse.com":         1,
    "nseindia.com":    1,
    "bse.com":         1,
    "bseindia.com":    1,
    "sebi.gov.in":     1,
    "rbi.org.in":      1,
    "pib.gov.in":      1,
    "mca.gov.in":      1,
    "finmin.nic.in":   1,
    # Trust 2 — Reputable
    "economictimes.com":       2,
    "indiatimes.com":          2,   # ET is under indiatimes
    "livemint.com":            2,
    "moneycontrol.com":        2,
    "businessstandard.com":    2,
    "financialexpress.com":    2,
    "reuters.com":             2,
    "bloomberg.com":           2,
    "thehindu.com":            2,
    "hindustantimes.com":      2,
    "cnbctv18.com":            2,
    "ndtv.com":                2,
    "zeebiz.com":              2,
    # Trust 3 — Opinion / Analysts
    "seekingalpha.com":        3,
    "valuepickr.com":          3,
    "screener.in":             3,
    "smallcases.com":          3,
}


def _get_trust_score(source_url: str, source_name: str) -> int:
    """
    Return trust level 1-4 for a news source.

    Checks URL domain first; falls back to source_name substring matching.
    Returns 4 (Unverified) if no match is found.
    """
    def _check_string(text: str) -> Optional[int]:
        text_lower = text.lower()
        for domain, score in _TRUST_DOMAIN_MAP.items():
            if domain in text_lower:
                return score
        return None

    # Try domain from URL
    if source_url:
        try:
            domain = urlparse(source_url).netloc.lower()
            result = _check_string(domain)
            if result is not None:
                return result
        except Exception:
            pass

    # Try source name
    if source_name:
        result = _check_string(source_name)
        if result is not None:
            return result

    return 4  # Unverified


# ---------------------------------------------------------------------------
# Event type keywords
# ---------------------------------------------------------------------------

_EVENT_KEYWORDS: dict[str, list[str]] = {
    "earnings": [
        "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
        "annual results", "net profit", "revenue", "ebitda", "pat", "earnings",
        "financial results", "turnover",
    ],
    "merger_acquisition": [
        "merger", "acquisition", "takeover", "amalgamation", "buyout", "stake sale",
        "demerger", "spin-off", "joint venture", "strategic alliance",
    ],
    "management_change": [
        "ceo appointed", "ceo resigned", "md appointed", "director resigned",
        "chairman", "board change", "key managerial", "cfo", "coo",
    ],
    "regulatory": [
        "sebi", "rbi", "cci", "nclt", "nclat", "show cause", "penalty",
        "fine", "investigation", "compliance", "adjudication", "order",
        "probe",
    ],
    "dividend": [
        "dividend declared", "interim dividend", "final dividend", "special dividend",
        "dividend per share",
    ],
    "buyback": [
        "buyback", "share repurchase", "open offer",
    ],
    "expansion": [
        "capex", "expansion", "new plant", "greenfield", "brownfield", "capacity",
        "order win", "contract win", "new order",
    ],
    "macro": [
        "rbi policy", "repo rate", "gdp", "inflation", "iip", "cpi", "pmi",
        "budget", "fiscal policy", "government policy", "federal reserve",
    ],
}

# ---------------------------------------------------------------------------
# Materiality base scores
# ---------------------------------------------------------------------------

_MATERIALITY_BASE: dict[str, float] = {
    "earnings":           0.90,
    "merger_acquisition": 0.85,
    "regulatory":         0.80,
    "management_change":  0.65,
    "expansion":          0.60,
    "buyback":            0.55,
    "dividend":           0.50,
    "macro":              0.40,
    "other":              0.30,
}

# ---------------------------------------------------------------------------
# News Ingestion Service
# ---------------------------------------------------------------------------

class NewsIngestionService:
    """
    Collect, classify, score, and deduplicate financial news from multiple sources.

    Sources:
    - NewsAPI v2 (requires API key)
    - RSS feeds (ET Markets, MoneyControl)
    - NSE/BSE announcements (handled by nse_bse.py, passed in externally)

    Usage::

        svc = NewsIngestionService()
        articles = await svc.fetch_news_api(since=datetime(...), api_key="...")
        rss = await svc.fetch_rss_feeds()
        all_articles = svc.deduplicate(articles + rss)
    """

    TRUST_LEVELS: dict[str, int] = _TRUST_DOMAIN_MAP  # exposed for inspection

    RSS_FEEDS: list[dict[str, str]] = [
        {
            "name": "Economic Times Markets",
            "url": "https://economictimes.indiatimes.com/markets/rss.cms",
            "trust_score": 2,
        },
        {
            "name": "MoneyControl Latest",
            "url": "https://www.moneycontrol.com/rss/latestnews.xml",
            "trust_score": 2,
        },
    ]

    def __init__(self, timeout: float = _REQUEST_TIMEOUT) -> None:
        self.timeout = timeout
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; BharatMarketAI/1.0; "
                "+https://bharatmarketai.in/bot)"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }

    # ---------------------------------------------------------------------------
    # NewsAPI
    # ---------------------------------------------------------------------------

    async def fetch_news_api(self, since: datetime, api_key: str) -> list[dict]:
        """
        Fetch financial news from NewsAPI v2.

        Args:
            since:   Fetch articles published after this datetime.
            api_key: NewsAPI key. Pass empty string to skip (returns []).

        Returns:
            List of article dicts with: title, url, source_name, source_type,
            trust_score, published_at, raw_text.
        """
        if not api_key:
            logger.info("NewsIngestionService.fetch_news_api: no api_key, skipping")
            return []

        since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        from_date = since_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "q": "india stock market NSE BSE",
            "language": "en",
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key,
        }
        url = "https://newsapi.org/v2/everything"

        logger.info("NewsIngestionService.fetch_news_api: from=%s", from_date)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "NewsIngestionService.fetch_news_api: HTTP %s — %s",
                    exc.response.status_code, exc.response.text[:200],
                )
                return []
            except Exception as exc:
                logger.warning(
                    "NewsIngestionService.fetch_news_api: error — %s", exc
                )
                return []

        articles_raw = data.get("articles", [])
        results: list[dict] = []

        for art in articles_raw:
            try:
                source = art.get("source", {})
                source_name = str(source.get("name", ""))
                source_url = str(art.get("url", ""))
                title = str(art.get("title", ""))
                description = str(art.get("description", "") or "")
                content = str(art.get("content", "") or "")
                raw_text = description or content
                pub_str = str(art.get("publishedAt", ""))

                trust = _get_trust_score(source_url, source_name)
                source_type = {1: "official", 2: "reputable", 3: "opinion"}.get(
                    trust, "unverified"
                )

                # Parse published_at
                try:
                    pub_dt = datetime.strptime(
                        pub_str, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    pub_dt = datetime.now(tz=timezone.utc)

                results.append({
                    "title": title,
                    "url": source_url,
                    "source_name": source_name,
                    "source_type": source_type,
                    "trust_score": trust,
                    "published_at": pub_dt.isoformat(),
                    "raw_text": raw_text,
                    "content_hash": _article_content_hash(title, source_name, pub_str),
                })
            except Exception as exc:
                logger.debug("NewsIngestionService: parse error on article: %s", exc)

        logger.info(
            "NewsIngestionService.fetch_news_api: fetched %d articles",
            len(results),
        )
        return results

    # ---------------------------------------------------------------------------
    # RSS feeds
    # ---------------------------------------------------------------------------

    async def fetch_rss_feeds(self) -> list[dict]:
        """
        Fetch and parse configured RSS feeds.

        Returns:
            List of article dicts: title, url, source_name, source_type,
            trust_score, published_at, raw_text.
        """
        all_articles: list[dict] = []

        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            for feed in self.RSS_FEEDS:
                feed_name = feed["name"]
                feed_url = feed["url"]
                trust = int(feed["trust_score"])
                source_type = {1: "official", 2: "reputable"}.get(trust, "unverified")

                try:
                    resp = await client.get(feed_url)
                    resp.raise_for_status()
                    articles = self._parse_rss(
                        resp.text, feed_name, feed_url, trust, source_type
                    )
                    all_articles.extend(articles)
                    logger.info(
                        "NewsIngestionService.fetch_rss_feeds: %s → %d articles",
                        feed_name, len(articles),
                    )
                except Exception as exc:
                    logger.warning(
                        "NewsIngestionService.fetch_rss_feeds: error fetching %s: %s",
                        feed_name, exc,
                    )

        return all_articles

    def _parse_rss(
        self,
        xml_text: str,
        source_name: str,
        source_url: str,
        trust: int,
        source_type: str,
    ) -> list[dict]:
        """Parse an RSS 2.0 or Atom feed string and return normalised articles."""
        articles: list[dict] = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("_parse_rss: XML parse error for %s: %s", source_name, exc)
            return []

        # Handle both RSS 2.0 (<item>) and Atom (<entry>)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        items = (
            root.findall(".//item")
            or root.findall(".//atom:entry", ns)
            or root.findall(".//entry")
        )

        for item in items:
            try:
                def _text(tag: str, default: str = "") -> str:
                    el = item.find(tag)
                    return (el.text or default).strip() if el is not None else default

                title = _text("title")
                link = _text("link") or _text("guid")
                pub_str = _text("pubDate") or _text("published") or _text("updated")
                description = _text("description") or _text("summary")

                # Parse date leniently
                pub_dt = _parse_rss_date(pub_str)

                articles.append({
                    "title": title,
                    "url": link,
                    "source_name": source_name,
                    "source_type": source_type,
                    "trust_score": trust,
                    "published_at": pub_dt.isoformat() if pub_dt else "",
                    "raw_text": description,
                    "content_hash": _article_content_hash(title, source_name, pub_str),
                })
            except Exception as exc:
                logger.debug("_parse_rss: item parse error: %s", exc)

        return articles

    # ---------------------------------------------------------------------------
    # Symbol extraction
    # ---------------------------------------------------------------------------

    def extract_symbols(self, text: str, symbol_master: dict) -> list[str]:
        """
        Find NSE/BSE symbols mentioned in `text`.

        Strategy:
        1. Direct regex match of uppercase 2-12 letter tokens against symbol_master keys.
        2. Company name substring search using symbol_master values.

        Args:
            text:          Article title + content string.
            symbol_master: {symbol: company_name} mapping.

        Returns:
            Deduplicated list of matched NSE symbols.
        """
        found: set[str] = set()
        text_clean = text.strip()
        text_upper = text_clean.upper()

        # 1. Direct symbol token matching
        #    NSE symbols are 2-12 uppercase letters, optionally ending in -EQ etc.
        token_pattern = re.compile(r"\b([A-Z][A-Z0-9&\-]{1,11})\b")
        for m in token_pattern.finditer(text_upper):
            token = m.group(1).split("-")[0]  # strip series suffix
            if token in symbol_master:
                found.add(token)

        # 2. Company name fuzzy matching (case-insensitive)
        text_lower = text_clean.lower()
        for symbol, company_name in symbol_master.items():
            if not company_name:
                continue
            # Match the first word of company name (≥4 chars) in the text
            name_lower = company_name.lower()
            first_word = name_lower.split()[0] if name_lower.split() else ""
            if len(first_word) >= 4 and first_word in text_lower:
                # Verify full company name or first 2 words match
                words = name_lower.split()[:2]
                if all(w in text_lower for w in words):
                    found.add(symbol)

        return sorted(found)

    # ---------------------------------------------------------------------------
    # Event classification
    # ---------------------------------------------------------------------------

    def classify_event_type(self, title: str, content: str) -> str:
        """
        Classify news into one of:
            earnings, merger_acquisition, management_change, regulatory,
            dividend, buyback, expansion, macro, other.

        Uses keyword matching on combined title + content (case-insensitive).
        First match wins (ordered by specificity).
        """
        combined = (title + " " + content).lower()

        # Check in priority order
        priority_order = [
            "merger_acquisition",
            "regulatory",
            "buyback",
            "dividend",
            "earnings",
            "management_change",
            "expansion",
            "macro",
        ]

        for event_type in priority_order:
            keywords = _EVENT_KEYWORDS.get(event_type, [])
            if any(kw in combined for kw in keywords):
                return event_type

        return "other"

    # ---------------------------------------------------------------------------
    # Materiality
    # ---------------------------------------------------------------------------

    def calculate_materiality(
        self,
        event_type: str,
        trust_score: int,
        mentions_count: int,
    ) -> float:
        """
        Compute materiality score (0.0 - 1.0).

        Formula:
            base = _MATERIALITY_BASE[event_type]
            boost +0.05 if trust_score=1 (official source)
            penalty -0.10 if trust_score=4 (unverified)
            boost +0.02 per additional mention (max +0.10)

        Args:
            event_type:     Classified event type string.
            trust_score:    1-4 trust level of the source.
            mentions_count: Number of different sources covering this event.

        Returns:
            Materiality float clamped to [0.0, 1.0].
        """
        base = _MATERIALITY_BASE.get(event_type, 0.30)

        if trust_score == 1:
            base += 0.05
        elif trust_score == 4:
            base -= 0.10

        mention_bonus = min(0.10, (max(0, mentions_count - 1)) * 0.02)
        base += mention_bonus

        return round(max(0.0, min(1.0, base)), 4)

    # ---------------------------------------------------------------------------
    # Deduplication
    # ---------------------------------------------------------------------------

    def deduplicate(self, articles: list[dict]) -> list[dict]:
        """
        Deduplicate articles by content_hash.

        The first occurrence of each hash is kept as the primary.
        Subsequent duplicates are marked with ``is_duplicate=True``
        and ``primary_article_id`` pointing to the hash of the primary.

        If an article lacks a ``content_hash``, one is computed on the fly.

        Args:
            articles: List of article dicts (any source mix).

        Returns:
            Full list with duplicate flags set (duplicates are NOT dropped —
            callers can filter with ``[a for a in articles if not a.get('is_duplicate')]``).
        """
        seen: dict[str, str] = {}  # hash → primary_article_id (also hash for simplicity)
        result: list[dict] = []

        for art in articles:
            # Ensure hash exists
            if not art.get("content_hash"):
                art["content_hash"] = _article_content_hash(
                    str(art.get("title", "")),
                    str(art.get("source_name", "")),
                    str(art.get("published_at", "")),
                )

            h = art["content_hash"]
            if h not in seen:
                seen[h] = h
                art["is_duplicate"] = False
                art["primary_article_id"] = h
            else:
                art["is_duplicate"] = True
                art["primary_article_id"] = seen[h]

            result.append(art)

        total = len(result)
        dups = sum(1 for a in result if a["is_duplicate"])
        logger.info(
            "NewsIngestionService.deduplicate: %d total, %d unique, %d duplicates",
            total, total - dups, dups,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _article_content_hash(title: str, source: str, pub_date: str) -> str:
    """SHA-256 of normalised (title, source, date) triplet."""
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    canonical = "|".join([_norm(title), _norm(source), _norm(pub_date)])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_rss_date(date_str: str) -> Optional[datetime]:
    """
    Parse an RSS pubDate string into a UTC-aware datetime.
    Handles RFC 2822 (Mon, 01 Jan 2024 12:00:00 +0000) and ISO variants.
    """
    if not date_str:
        return None

    # Strip day-of-week prefix if present (e.g. "Sat, ")
    date_str = re.sub(r"^\w+,\s*", "", date_str.strip())

    formats = [
        "%d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %Z",
        "%d %b %Y %H:%M %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d %b %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # Last resort: try email.utils
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        pass

    return None
