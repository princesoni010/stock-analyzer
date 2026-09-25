"""
services/normalizer.py
Symbol normalizer and data quality service for Bharat Market AI.
"""

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol Master – top 50 Nifty constituents
# ---------------------------------------------------------------------------

_SYMBOL_MASTER: dict[str, dict] = {
    "RELIANCE": {
        "isin": "INE002A01018",
        "company_name": "Reliance Industries Limited",
        "sector": "Energy",
        "industry": "Oil & Gas - Diversified",
        "exchange": "NSE",
        "aliases": ["RELI", "RIL"],
    },
    "TCS": {
        "isin": "INE467B01029",
        "company_name": "Tata Consultancy Services Limited",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "exchange": "NSE",
        "aliases": ["TATACONSULTANCY"],
    },
    "HDFCBANK": {
        "isin": "INE040A01034",
        "company_name": "HDFC Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Bank",
        "exchange": "NSE",
        "aliases": ["HDFC BANK"],
    },
    "INFY": {
        "isin": "INE009A01021",
        "company_name": "Infosys Limited",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "exchange": "NSE",
        "aliases": ["INFOSYS"],
    },
    "ICICIBANK": {
        "isin": "INE090A01021",
        "company_name": "ICICI Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Bank",
        "exchange": "NSE",
        "aliases": ["ICICI BANK"],
    },
    "HINDUNILVR": {
        "isin": "INE030A01027",
        "company_name": "Hindustan Unilever Limited",
        "sector": "Fast Moving Consumer Goods",
        "industry": "Diversified FMCG",
        "exchange": "NSE",
        "aliases": ["HUL"],
    },
    "BHARTIARTL": {
        "isin": "INE397D01024",
        "company_name": "Bharti Airtel Limited",
        "sector": "Telecom",
        "industry": "Telecom - Cellular & Fixed line services",
        "exchange": "NSE",
        "aliases": ["AIRTEL"],
    },
    "ITC": {
        "isin": "INE154A01025",
        "company_name": "ITC Limited",
        "sector": "Fast Moving Consumer Goods",
        "industry": "Cigarettes",
        "exchange": "NSE",
        "aliases": [],
    },
    "KOTAKBANK": {
        "isin": "INE237A01028",
        "company_name": "Kotak Mahindra Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Bank",
        "exchange": "NSE",
        "aliases": ["KOTAK"],
    },
    "AXISBANK": {
        "isin": "INE238A01034",
        "company_name": "Axis Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Bank",
        "exchange": "NSE",
        "aliases": ["AXIS BANK"],
    },
    "LT": {
        "isin": "INE018A01030",
        "company_name": "Larsen & Toubro Limited",
        "sector": "Construction",
        "industry": "Construction & Engineering",
        "exchange": "NSE",
        "aliases": ["L&T", "LARSENTOUBRO"],
    },
    "SBIN": {
        "isin": "INE062A01020",
        "company_name": "State Bank of India",
        "sector": "Financial Services",
        "industry": "Public Sector Bank",
        "exchange": "NSE",
        "aliases": ["SBI"],
    },
    "WIPRO": {
        "isin": "INE075A01022",
        "company_name": "Wipro Limited",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "exchange": "NSE",
        "aliases": [],
    },
    "HCLTECH": {
        "isin": "INE860A01027",
        "company_name": "HCL Technologies Limited",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "exchange": "NSE",
        "aliases": ["HCL"],
    },
    "ASIANPAINT": {
        "isin": "INE021A01026",
        "company_name": "Asian Paints Limited",
        "sector": "Consumer Discretionary",
        "industry": "Paints",
        "exchange": "NSE",
        "aliases": ["ASIAN PAINTS"],
    },
    "MARUTI": {
        "isin": "INE585B01010",
        "company_name": "Maruti Suzuki India Limited",
        "sector": "Automobile and Auto Components",
        "industry": "Passenger Cars & Utility Vehicles",
        "exchange": "NSE",
        "aliases": ["MARUTISUZUKI"],
    },
    "BAJFINANCE": {
        "isin": "INE296A01024",
        "company_name": "Bajaj Finance Limited",
        "sector": "Financial Services",
        "industry": "Consumer Finance",
        "exchange": "NSE",
        "aliases": ["BAJAJ FINANCE"],
    },
    "NESTLEIND": {
        "isin": "INE239A01024",
        "company_name": "Nestle India Limited",
        "sector": "Fast Moving Consumer Goods",
        "industry": "Food Products",
        "exchange": "NSE",
        "aliases": ["NESTLE"],
    },
    "TITAN": {
        "isin": "INE280A01028",
        "company_name": "Titan Company Limited",
        "sector": "Consumer Discretionary",
        "industry": "Gems Jewellery and Watches",
        "exchange": "NSE",
        "aliases": [],
    },
    "ULTRACEMCO": {
        "isin": "INE481G01011",
        "company_name": "UltraTech Cement Limited",
        "sector": "Construction Materials",
        "industry": "Cement & Cement Products",
        "exchange": "NSE",
        "aliases": ["ULTRATECH"],
    },
    "POWERGRID": {
        "isin": "INE752E01010",
        "company_name": "Power Grid Corporation of India Limited",
        "sector": "Utilities",
        "industry": "Electric Utilities",
        "exchange": "NSE",
        "aliases": ["PGCIL"],
    },
    "NTPC": {
        "isin": "INE733E01010",
        "company_name": "NTPC Limited",
        "sector": "Utilities",
        "industry": "Electric Utilities",
        "exchange": "NSE",
        "aliases": [],
    },
    "SUNPHARMA": {
        "isin": "INE044A01036",
        "company_name": "Sun Pharmaceutical Industries Limited",
        "sector": "Healthcare",
        "industry": "Pharmaceuticals",
        "exchange": "NSE",
        "aliases": ["SUN PHARMA"],
    },
    "ONGC": {
        "isin": "INE213A01029",
        "company_name": "Oil and Natural Gas Corporation Limited",
        "sector": "Energy",
        "industry": "Oil Exploration & Production",
        "exchange": "NSE",
        "aliases": [],
    },
    "TATAMOTORS": {
        "isin": "INE155A01022",
        "company_name": "Tata Motors Limited",
        "sector": "Automobile and Auto Components",
        "industry": "Commercial Vehicles",
        "exchange": "NSE",
        "aliases": ["TATA MOTORS"],
    },
    "M&M": {
        "isin": "INE101A01026",
        "company_name": "Mahindra & Mahindra Limited",
        "sector": "Automobile and Auto Components",
        "industry": "Passenger Cars & Utility Vehicles",
        "exchange": "NSE",
        "aliases": ["MM", "MAHINDRA"],
    },
    "BAJAJFINSV": {
        "isin": "INE918I01026",
        "company_name": "Bajaj Finserv Limited",
        "sector": "Financial Services",
        "industry": "Diversified Financial Services",
        "exchange": "NSE",
        "aliases": ["BAJAJ FINSERV"],
    },
    "BRITANNIA": {
        "isin": "INE216A01030",
        "company_name": "Britannia Industries Limited",
        "sector": "Fast Moving Consumer Goods",
        "industry": "Food Products",
        "exchange": "NSE",
        "aliases": [],
    },
    "BPCL": {
        "isin": "INE029A01011",
        "company_name": "Bharat Petroleum Corporation Limited",
        "sector": "Energy",
        "industry": "Oil & Gas - Refining & Marketing",
        "exchange": "NSE",
        "aliases": ["BHARAT PETROLEUM"],
    },
    "HEROMOTOCO": {
        "isin": "INE158A01026",
        "company_name": "Hero MotoCorp Limited",
        "sector": "Automobile and Auto Components",
        "industry": "Two Wheelers",
        "exchange": "NSE",
        "aliases": ["HERO", "HEROMOTOCORP"],
    },
    "GRASIM": {
        "isin": "INE047A01021",
        "company_name": "Grasim Industries Limited",
        "sector": "Construction Materials",
        "industry": "Diversified",
        "exchange": "NSE",
        "aliases": [],
    },
    "INDUSINDBK": {
        "isin": "INE095A01012",
        "company_name": "IndusInd Bank Limited",
        "sector": "Financial Services",
        "industry": "Private Sector Bank",
        "exchange": "NSE",
        "aliases": ["INDUSIND"],
    },
    "HINDALCO": {
        "isin": "INE038A01020",
        "company_name": "Hindalco Industries Limited",
        "sector": "Metals & Mining",
        "industry": "Aluminium",
        "exchange": "NSE",
        "aliases": [],
    },
    "TATASTEEL": {
        "isin": "INE081A01012",
        "company_name": "Tata Steel Limited",
        "sector": "Metals & Mining",
        "industry": "Iron & Steel",
        "exchange": "NSE",
        "aliases": ["TATA STEEL"],
    },
    "JSWSTEEL": {
        "isin": "INE019A01038",
        "company_name": "JSW Steel Limited",
        "sector": "Metals & Mining",
        "industry": "Iron & Steel",
        "exchange": "NSE",
        "aliases": ["JSW"],
    },
    "COALINDIA": {
        "isin": "INE522F01014",
        "company_name": "Coal India Limited",
        "sector": "Metals & Mining",
        "industry": "Coal",
        "exchange": "NSE",
        "aliases": ["CIL"],
    },
    "EICHERMOT": {
        "isin": "INE066A01021",
        "company_name": "Eicher Motors Limited",
        "sector": "Automobile and Auto Components",
        "industry": "Two Wheelers",
        "exchange": "NSE",
        "aliases": ["EICHER", "ROYALENFIELD"],
    },
    "TECHM": {
        "isin": "INE669C01036",
        "company_name": "Tech Mahindra Limited",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "exchange": "NSE",
        "aliases": ["TECH MAHINDRA"],
    },
    "DRREDDY": {
        "isin": "INE089A01023",
        "company_name": "Dr. Reddy's Laboratories Limited",
        "sector": "Healthcare",
        "industry": "Pharmaceuticals",
        "exchange": "NSE",
        "aliases": ["DR REDDY", "DRREDDYS"],
    },
    "DIVISLAB": {
        "isin": "INE361B01024",
        "company_name": "Divi's Laboratories Limited",
        "sector": "Healthcare",
        "industry": "Pharmaceuticals",
        "exchange": "NSE",
        "aliases": ["DIVIS", "DIVISLAB"],
    },
    "CIPLA": {
        "isin": "INE059A01026",
        "company_name": "Cipla Limited",
        "sector": "Healthcare",
        "industry": "Pharmaceuticals",
        "exchange": "NSE",
        "aliases": [],
    },
    "APOLLOHOSP": {
        "isin": "INE437A01024",
        "company_name": "Apollo Hospitals Enterprise Limited",
        "sector": "Healthcare",
        "industry": "Healthcare Facilities",
        "exchange": "NSE",
        "aliases": ["APOLLO HOSPITALS", "APOLLO"],
    },
    "ADANIPORTS": {
        "isin": "INE742F01042",
        "company_name": "Adani Ports and Special Economic Zone Limited",
        "sector": "Services",
        "industry": "Port & Port services",
        "exchange": "NSE",
        "aliases": ["APSEZ"],
    },
    "TATACONSUM": {
        "isin": "INE192A01025",
        "company_name": "Tata Consumer Products Limited",
        "sector": "Fast Moving Consumer Goods",
        "industry": "Food Products",
        "exchange": "NSE",
        "aliases": ["TATA CONSUMER"],
    },
    "VEDL": {
        "isin": "INE205A01025",
        "company_name": "Vedanta Limited",
        "sector": "Metals & Mining",
        "industry": "Diversified Metals",
        "exchange": "NSE",
        "aliases": ["VEDANTA"],
    },
    "PIDILITIND": {
        "isin": "INE318A01026",
        "company_name": "Pidilite Industries Limited",
        "sector": "Chemicals",
        "industry": "Specialty Chemicals",
        "exchange": "NSE",
        "aliases": ["PIDILITE", "FEVICOL"],
    },
    "SBICARD": {
        "isin": "INE018E01016",
        "company_name": "SBI Cards and Payment Services Limited",
        "sector": "Financial Services",
        "industry": "Credit Card",
        "exchange": "NSE",
        "aliases": ["SBI CARD"],
    },
    "HAVELLS": {
        "isin": "INE176B01034",
        "company_name": "Havells India Limited",
        "sector": "Consumer Discretionary",
        "industry": "Consumer Durables",
        "exchange": "NSE",
        "aliases": [],
    },
    "MUTHOOTFIN": {
        "isin": "INE414G01012",
        "company_name": "Muthoot Finance Limited",
        "sector": "Financial Services",
        "industry": "Gold Financing",
        "exchange": "NSE",
        "aliases": ["MUTHOOT"],
    },
    "COLPAL": {
        "isin": "INE259A01022",
        "company_name": "Colgate-Palmolive (India) Limited",
        "sector": "Fast Moving Consumer Goods",
        "industry": "Personal Care",
        "exchange": "NSE",
        "aliases": ["COLGATE"],
    },
}

# Build reverse alias lookup: alias_upper -> canonical_symbol
_ALIAS_TO_SYMBOL: dict[str, str] = {}
for _sym, _meta in _SYMBOL_MASTER.items():
    for _alias in _meta.get("aliases", []):
        _ALIAS_TO_SYMBOL[_alias.upper()] = _sym

_IST = ZoneInfo("Asia/Kolkata")
_UTC = timezone.utc


class SymbolMaster:
    """
    Central registry for NSE equity symbols.
    Provides metadata lookup, alias resolution and validation.
    """

    def __init__(self) -> None:
        self._master: dict[str, dict] = _SYMBOL_MASTER
        self._alias_map: dict[str, str] = _ALIAS_TO_SYMBOL

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_metadata(self, symbol: str) -> Optional[dict]:
        """Return the master metadata dict for *symbol*, or None."""
        return self._master.get(symbol.strip().upper())

    def resolve_alias(self, raw: str) -> Optional[str]:
        """
        Try to resolve *raw* string to canonical NSE symbol.
        Returns canonical symbol string or None if unresolvable.
        """
        upper = raw.strip().upper()
        if upper in self._master:
            return upper
        return self._alias_map.get(upper)

    def all_symbols(self) -> list[str]:
        """Return all canonical NSE symbols in the master."""
        return list(self._master.keys())

    def symbols_by_sector(self, sector: str) -> list[str]:
        """Return all symbols that belong to *sector*."""
        return [
            sym
            for sym, meta in self._master.items()
            if meta["sector"].lower() == sector.lower()
        ]


# Module-level singleton (import-safe)
_symbol_master_instance: Optional[SymbolMaster] = None


@lru_cache(maxsize=1)
def get_symbol_master() -> SymbolMaster:
    """Return the module-level SymbolMaster singleton (cached)."""
    return SymbolMaster()


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def normalize_symbol(raw: str) -> str:
    """
    Strip whitespace, uppercase and validate *raw* against the symbol master.

    Raises:
        ValueError: if the cleaned symbol (or its alias) is not in the master.
    Returns:
        The canonical NSE symbol string.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError(f"Symbol must be a non-empty string, got: {raw!r}")

    master = get_symbol_master()
    resolved = master.resolve_alias(raw)
    if resolved is None:
        raise ValueError(
            f"Symbol '{raw.strip().upper()}' not found in master. "
            "Use validate_symbol() to check before normalising."
        )
    return resolved


def normalize_timestamp(ts, tz_str: str = "UTC") -> datetime:
    """
    Convert *ts* (int unix epoch, float, ISO string, or datetime) to an
    aware UTC datetime.

    Args:
        ts:      The timestamp to normalise.
        tz_str:  IANA timezone name for naive datetimes / epoch interpretation
                 (default ``'UTC'``).

    Returns:
        datetime in UTC (tzinfo=timezone.utc).
    """
    try:
        source_tz = ZoneInfo(tz_str)
    except Exception:
        logger.warning("Unknown tz_str '%s', falling back to UTC.", tz_str)
        source_tz = ZoneInfo("UTC")

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            # Treat as source_tz-aware
            ts = ts.replace(tzinfo=source_tz)
        return ts.astimezone(_UTC)

    if isinstance(ts, (int, float)):
        # Unix epoch – always UTC
        return datetime.fromtimestamp(ts, tz=_UTC)

    if isinstance(ts, str):
        ts = ts.strip()
        # Try ISO 8601 variants
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=source_tz)
                return dt.astimezone(_UTC)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp string: {ts!r}")

    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


def ist_to_display(utc_dt: datetime) -> str:
    """
    Convert a UTC-aware datetime to an IST string suitable for reports.

    Returns:
        e.g. ``"26-Sep-2026 07:10:50 IST"``
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=_UTC)
    ist_dt = utc_dt.astimezone(_IST)
    return ist_dt.strftime("%d-%b-%Y %H:%M:%S IST")


def compute_content_hash(title: str, symbol: str, published_date: str) -> str:
    """
    Return a SHA-256 hex digest that uniquely identifies a news/event record.
    Used for deduplication in the ingestion pipeline.

    Args:
        title:          Article or event title.
        symbol:         NSE symbol string.
        published_date: Date string (any format; whitespace-stripped).

    Returns:
        64-character lowercase hex string.
    """
    canonical = f"{title.strip()}|{symbol.strip().upper()}|{published_date.strip()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_stale(fetched_at: datetime, threshold_minutes: int) -> bool:
    """
    Return True if *fetched_at* is older than *threshold_minutes* from now (UTC).

    Args:
        fetched_at:         The timestamp when the data was originally fetched.
                            May be naive (assumed UTC) or tz-aware.
        threshold_minutes:  Maximum acceptable age in minutes.

    Returns:
        bool
    """
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=_UTC)
    now = datetime.now(tz=_UTC)
    age = now - fetched_at
    return age > timedelta(minutes=threshold_minutes)


def validate_symbol(symbol: str) -> bool:
    """
    Return True if *symbol* (or a known alias) exists in the symbol master.

    Args:
        symbol: Raw symbol string (will be stripped & uppercased).

    Returns:
        bool
    """
    master = get_symbol_master()
    return master.resolve_alias(symbol) is not None
