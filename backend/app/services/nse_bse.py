"""
NSE & BSE corporate announcement collectors.

Both services use httpx for async HTTP, mimic browser headers to avoid
bot-detection, and return normalised dicts with a deterministic content hash.
On any error they log a warning and return an empty list — callers should
handle missing data gracefully.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "DNT": "1",
}

_REQUEST_TIMEOUT = 20.0   # seconds


def compute_content_hash(title: str, symbol: str, date_str: str) -> str:
    """
    Return a SHA-256 hex digest of the normalised announcement identity.

    Used to detect duplicate announcements across NSE and BSE feeds.

    Args:
        title:    Announcement title / subject.
        symbol:   Exchange ticker symbol.
        date_str: Publication date string (any format, normalised internally).

    Returns:
        64-char lowercase hex digest.
    """
    # Normalise: lowercase, strip punctuation, collapse whitespace
    def _normalise(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    canonical = "|".join([
        _normalise(title),
        _normalise(symbol),
        _normalise(date_str),
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_iso_or_none(date_str: str) -> Optional[datetime]:
    """
    Try to parse a date string in common NSE/BSE formats.
    Returns a timezone-aware UTC datetime, or None on failure.
    """
    formats = [
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# NSE Service
# ---------------------------------------------------------------------------

class NSEService:
    """
    Fetch corporate announcements, board meeting notices, and results calendar
    from the NSE India public JSON API.
    """

    BASE_URL = "https://www.nseindia.com"
    ANNOUNCEMENTS_URL = (
        f"{BASE_URL}/api/corporate-announcements?index=equities"
    )
    BOARD_MEETINGS_URL = (
        f"{BASE_URL}/api/corporate-announcements?index=equities"
        "&type=board-meeting"
    )
    RESULTS_CALENDAR_URL = (
        f"{BASE_URL}/api/corporate-announcements?index=equities"
        "&type=quarterly-results"
    )

    # NSE requires a session cookie; a warm-up GET on the homepage is needed.
    HOME_URL = f"{BASE_URL}/"

    def __init__(self, timeout: float = _REQUEST_TIMEOUT) -> None:
        self.timeout = timeout
        self._headers = {
            **_BROWSER_HEADERS,
            "Referer": self.BASE_URL + "/",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_client_with_cookie(self) -> httpx.AsyncClient:
        """
        Return an AsyncClient that has visited the NSE homepage to obtain
        session cookies (required by NSE's API layer).
        """
        client = httpx.AsyncClient(
            headers=self._headers,
            timeout=self.timeout,
            follow_redirects=True,
        )
        try:
            await client.get(self.HOME_URL)
        except Exception as exc:
            logger.warning("NSEService: failed to warm up session cookie: %s", exc)
        return client

    async def _fetch_json(self, url: str) -> Optional[list | dict]:
        """Fetch JSON from NSE API with browser headers and session cookie."""
        client = await self._get_client_with_cookie()
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "NSEService: HTTP %s from %s", exc.response.status_code, url
            )
        except httpx.RequestError as exc:
            logger.warning("NSEService: request error for %s: %s", url, exc)
        except Exception as exc:
            logger.warning("NSEService: unexpected error for %s: %s", url, exc)
        finally:
            await client.aclose()
        return None

    @staticmethod
    def _normalise_symbol(raw: str) -> str:
        """Strip series suffix like '-EQ', uppercase."""
        return raw.strip().upper().split("-")[0]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_announcements(self, since: datetime) -> list[dict]:
        """
        Fetch corporate announcements from NSE, filtered to those published
        after `since`.

        Args:
            since: Timezone-aware datetime. Announcements before this are dropped.

        Returns:
            List of normalised announcement dicts:
            {
                exchange, symbol, announcement_type, subject, url,
                published_at (ISO str), content_hash
            }
        """
        logger.info("NSEService.fetch_announcements: since=%s", since.isoformat())
        raw = await self._fetch_json(self.ANNOUNCEMENTS_URL)
        if not raw:
            return []

        # NSE returns either a list directly or {'data': [...]}
        if isinstance(raw, dict):
            items = raw.get("data", [])
        else:
            items = raw

        results: list[dict] = []
        for item in items:
            try:
                date_str = str(
                    item.get("an_dt")
                    or item.get("date")
                    or item.get("submitted_date", "")
                )
                pub_dt = _parse_iso_or_none(date_str)
                if pub_dt is None:
                    continue

                # Make `since` comparable (both UTC)
                since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                if pub_dt < since_utc:
                    continue

                symbol_raw = str(
                    item.get("symbol") or item.get("sm_symbol", "")
                )
                symbol = self._normalise_symbol(symbol_raw)
                if not symbol:
                    continue

                subject = str(
                    item.get("subject")
                    or item.get("desc")
                    or item.get("an_subject", "")
                )
                ann_type = str(
                    item.get("an_type")
                    or item.get("category", "General")
                )
                url = str(item.get("an_pdf_file") or item.get("url", ""))
                if url and not url.startswith("http"):
                    url = self.BASE_URL + url

                results.append({
                    "exchange": "NSE",
                    "symbol": symbol,
                    "announcement_type": ann_type,
                    "subject": subject,
                    "url": url,
                    "published_at": pub_dt.isoformat(),
                    "content_hash": compute_content_hash(subject, symbol, date_str),
                })
            except Exception as exc:
                logger.debug("NSEService: parse error on item %s: %s", item, exc)

        logger.info(
            "NSEService.fetch_announcements: fetched %d items, returned %d after filter",
            len(items), len(results),
        )
        return results

    async def fetch_board_meetings(self, since: datetime) -> list[dict]:
        """
        Fetch upcoming board meeting notices from NSE.

        Returns list of dicts: {exchange, symbol, meeting_date, purpose, url}
        """
        logger.info("NSEService.fetch_board_meetings: since=%s", since.isoformat())
        raw = await self._fetch_json(self.BOARD_MEETINGS_URL)
        if not raw:
            return []

        items = raw.get("data", raw) if isinstance(raw, dict) else raw
        results: list[dict] = []

        for item in items:
            try:
                date_str = str(
                    item.get("meeting_date")
                    or item.get("an_dt")
                    or item.get("date", "")
                )
                pub_dt = _parse_iso_or_none(date_str)
                if pub_dt is None:
                    continue

                since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                if pub_dt < since_utc:
                    continue

                symbol = self._normalise_symbol(
                    str(item.get("symbol") or item.get("sm_symbol", ""))
                )
                if not symbol:
                    continue

                purpose = str(item.get("purpose") or item.get("subject", "Board Meeting"))
                url = str(item.get("an_pdf_file") or item.get("url", ""))
                if url and not url.startswith("http"):
                    url = self.BASE_URL + url

                results.append({
                    "exchange": "NSE",
                    "symbol": symbol,
                    "meeting_date": pub_dt.date().isoformat(),
                    "purpose": purpose,
                    "url": url,
                    "content_hash": compute_content_hash(purpose, symbol, date_str),
                })
            except Exception as exc:
                logger.debug("NSEService: board meeting parse error: %s", exc)

        logger.info("NSEService.fetch_board_meetings: returned %d items", len(results))
        return results

    async def fetch_results_calendar(self) -> list[dict]:
        """
        Fetch the upcoming results calendar from NSE (no date filter — returns all).

        Returns list of dicts: {exchange, symbol, result_date, period, url}
        """
        logger.info("NSEService.fetch_results_calendar")
        raw = await self._fetch_json(self.RESULTS_CALENDAR_URL)
        if not raw:
            return []

        items = raw.get("data", raw) if isinstance(raw, dict) else raw
        results: list[dict] = []

        for item in items:
            try:
                date_str = str(
                    item.get("result_date")
                    or item.get("meeting_date")
                    or item.get("an_dt", "")
                )
                symbol = self._normalise_symbol(
                    str(item.get("symbol") or item.get("sm_symbol", ""))
                )
                if not symbol:
                    continue

                period = str(item.get("period") or item.get("quarter", ""))
                url = str(item.get("an_pdf_file") or item.get("url", ""))
                if url and not url.startswith("http"):
                    url = self.BASE_URL + url

                results.append({
                    "exchange": "NSE",
                    "symbol": symbol,
                    "result_date": date_str,
                    "period": period,
                    "url": url,
                    "content_hash": compute_content_hash(period, symbol, date_str),
                })
            except Exception as exc:
                logger.debug("NSEService: results calendar parse error: %s", exc)

        logger.info("NSEService.fetch_results_calendar: returned %d items", len(results))
        return results


# ---------------------------------------------------------------------------
# BSE Service
# ---------------------------------------------------------------------------

class BSEService:
    """
    Fetch corporate announcements from the BSE India public API.

    BSE API endpoint documentation:
        https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w
        Parameters:
            pageno       : page number (1-indexed)
            strCat       : category code (-1 = all)
            strPrevDate  : from date (dd/mm/yyyy)
            strScrip     : scrip code (empty = all)
            strSearch    : P (published)
            strToDate    : to date (dd/mm/yyyy)
            strType      : C
            subcategory  : -1 (all)
    """

    API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    def __init__(self, timeout: float = _REQUEST_TIMEOUT) -> None:
        self.timeout = timeout
        self._headers = {
            **_BROWSER_HEADERS,
            "Referer": "https://www.bseindia.com/",
            "Origin": "https://www.bseindia.com",
        }

    def _date_to_bse_fmt(self, dt: datetime) -> str:
        """Format datetime as dd/mm/yyyy (BSE API format)."""
        return dt.strftime("%d/%m/%Y")

    async def _fetch_json(self, url: str) -> Optional[dict]:
        """Fetch JSON from BSE API."""
        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "BSEService: HTTP %s from %s", exc.response.status_code, url
                )
            except httpx.RequestError as exc:
                logger.warning("BSEService: request error for %s: %s", url, exc)
            except Exception as exc:
                logger.warning("BSEService: unexpected error for %s: %s", url, exc)
        return None

    async def fetch_announcements(self, since: datetime) -> list[dict]:
        """
        Fetch corporate announcements from BSE published after `since`.

        Returns list of normalised dicts:
        {
            exchange, symbol, scrip_code, announcement_type, subject, url,
            published_at (ISO str), content_hash
        }
        """
        since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        today = datetime.now(tz=timezone.utc)

        from_date_str = self._date_to_bse_fmt(since_utc)
        to_date_str = self._date_to_bse_fmt(today)

        params = {
            "pageno": "1",
            "strCat": "-1",
            "strPrevDate": from_date_str,
            "strScrip": "",
            "strSearch": "P",
            "strToDate": to_date_str,
            "strType": "C",
            "subcategory": "-1",
        }
        url = f"{self.API_URL}?{urlencode(params)}"

        logger.info("BSEService.fetch_announcements: url=%s", url)
        raw = await self._fetch_json(url)
        if not raw:
            return []

        # BSE returns {"Table": [...], "Table1": [...]}
        items = raw.get("Table", [])
        results: list[dict] = []

        for item in items:
            try:
                date_str = str(
                    item.get("News_submission_dt")
                    or item.get("SLONGDATE")
                    or item.get("DissemDT", "")
                )
                pub_dt = _parse_iso_or_none(date_str)
                if pub_dt is None:
                    # Try BSE-specific format dd/mm/yyyy HH:MM:SS
                    for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"]:
                        try:
                            pub_dt = datetime.strptime(
                                date_str.strip(), fmt
                            ).replace(tzinfo=timezone.utc)
                            break
                        except ValueError:
                            pass

                if pub_dt is None or pub_dt < since_utc:
                    continue

                symbol = str(
                    item.get("SLONGNAME")
                    or item.get("short_name", "")
                ).strip().upper()
                scrip_code = str(item.get("SCRIP_CD") or item.get("ScripCode", ""))

                subject = str(
                    item.get("NEWSSUB")
                    or item.get("Headline")
                    or item.get("subject", "")
                )
                ann_type = str(
                    item.get("CATEGORYNAME")
                    or item.get("Categoryname", "General")
                )

                # Build BSE announcement URL
                news_id = str(item.get("NEWSID") or item.get("Newsid", ""))
                if news_id:
                    ann_url = (
                        f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/"
                        f"{news_id}.pdf"
                    )
                else:
                    ann_url = ""

                results.append({
                    "exchange": "BSE",
                    "symbol": symbol,
                    "scrip_code": scrip_code,
                    "announcement_type": ann_type,
                    "subject": subject,
                    "url": ann_url,
                    "published_at": pub_dt.isoformat(),
                    "content_hash": compute_content_hash(subject, symbol, date_str),
                })
            except Exception as exc:
                logger.debug("BSEService: parse error on item %s: %s", item, exc)

        logger.info(
            "BSEService.fetch_announcements: fetched %d raw items, returned %d",
            len(items), len(results),
        )
        return results
