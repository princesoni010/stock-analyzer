"""
services/imd.py
IMD (India Meteorological Department) weather data collector for Bharat Market AI.

Attempts to fetch live data from IMD's public endpoints. Falls back to
structured mock data (marked is_stale=True) when the live endpoint is
unavailable or returns unexpected responses.

The weather score produced by this service feeds the theme engine to
identify agri-commodity and irrigation-related trading opportunities.
"""

import asyncio
import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# IMD endpoint configuration
# ---------------------------------------------------------------------------

_IMD_BASE_URL = "https://imdpune.gov.in"
_REQUEST_TIMEOUT = 10.0  # seconds

# Regions we care about for agri-commodity themes
_DEFAULT_REGIONS = [
    "Northwest India",
    "Central India",
    "Northeast India",
    "Peninsular India",
    "East & Northeast India",
]

# ---------------------------------------------------------------------------
# Mock / fallback data
# ---------------------------------------------------------------------------

def _build_mock_record(region: str, date_str: str) -> dict:
    """
    Return a plausible but clearly mock weather record for *region* on *date_str*.
    Used when the IMD live endpoint is unavailable.
    """
    # Deterministic but varied mock values seeded by region name length
    seed = len(region)
    rainfall_actual = round(5.0 + (seed % 7) * 3.5, 1)    # mm
    rainfall_normal = round(8.0 + (seed % 5) * 2.0, 1)    # mm
    deviation_pct = round((rainfall_actual - rainfall_normal) / rainfall_normal * 100, 1)
    reservoir_level = round(45.0 + (seed % 30), 1)         # % capacity
    forecast_options = ["Partly cloudy", "Moderate rain expected", "Clear skies", "Heavy rain warning"]
    forecast = forecast_options[seed % len(forecast_options)]

    return {
        "region": region,
        "data_date": date_str,
        "rainfall_actual": rainfall_actual,
        "rainfall_normal": rainfall_normal,
        "deviation_pct": deviation_pct,
        "reservoir_level": reservoir_level,
        "forecast": forecast,
        "warnings": [],
        "source": "imd_mock",
        "is_stale": True,
    }


# ---------------------------------------------------------------------------
# IMD Service
# ---------------------------------------------------------------------------

class IMDService:
    """
    Async weather data service that wraps the IMD public API.

    All public methods are coroutines.  When the live API is unreachable,
    the service degrades gracefully by returning mock data with
    ``is_stale=True``.
    """

    def __init__(self, base_url: str = _IMD_BASE_URL, timeout: float = _REQUEST_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_theme_inputs(
        self,
        regions: list[str],
        date_str: str,
    ) -> list[dict]:
        """
        Fetch weather theme inputs for the given *regions* and *date_str*.

        Attempts to call the IMD public rainfall/reservoir API.  On any
        network or parse failure, returns mock records with ``is_stale=True``.

        Args:
            regions:  List of geographic region names.
            date_str: Date string in ``'YYYY-MM-DD'`` format.

        Returns:
            List of dicts, one per region, with keys:
            ``region``, ``data_date``, ``rainfall_actual``,
            ``rainfall_normal``, ``deviation_pct``, ``reservoir_level``,
            ``forecast``, ``warnings``, ``source``, ``is_stale``.
        """
        if not regions:
            regions = _DEFAULT_REGIONS

        results: list[dict] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [
                self._fetch_region(client, region, date_str)
                for region in regions
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for region, result in zip(regions, raw_results):
            if isinstance(result, Exception):
                logger.warning(
                    "IMD fetch failed for region '%s': %s. Using mock data.", region, result
                )
                results.append(_build_mock_record(region, date_str))
            else:
                results.append(result)

        return results

    async def get_monsoon_status(self) -> dict:
        """
        Fetch the overall India monsoon status for the current season.

        Returns a summary dict with:
        ``season``, ``cumulative_rainfall_actual``, ``cumulative_rainfall_normal``,
        ``deviation_pct``, ``active_zones``, ``deficient_zones``,
        ``status_label``, ``source``, ``is_stale``, ``fetched_at``.
        """
        today_str = datetime.now(tz=_UTC).strftime("%Y-%m-%d")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self._base_url}/Climatology/rainfall/sw_monsoon_rainfall.php",
                    headers={"Accept": "application/json, text/html"},
                )
                response.raise_for_status()
                # IMD often returns HTML; try JSON first, fall back to mock
                try:
                    data = response.json()
                    return self._parse_monsoon_response(data, today_str)
                except Exception:
                    logger.debug("IMD monsoon endpoint returned non-JSON; using mock.")
                    return self._mock_monsoon_status(today_str)
        except Exception as exc:
            logger.warning("IMD monsoon status fetch failed: %s. Using mock.", exc)
            return self._mock_monsoon_status(today_str)

    def calculate_weather_score(self, weather_records: list[dict]) -> float:
        """
        Compute a 0–100 weather/monsoon score from *weather_records*.

        Scoring logic:
        * Base score starts at 50 (neutral monsoon assumed).
        * For each region record:
          - Rainfall deviation > +10 %  → +6 pts (excess monsoon)
          - Rainfall deviation > 0 %    → +3 pts (normal-to-above)
          - Rainfall deviation < -20 %  → -10 pts (severe deficit)
          - Rainfall deviation < -10 %  → -6 pts (moderate deficit)
          - Rainfall deviation < 0 %    → -2 pts (mild deficit)
          - Reservoir level > 70 %      → +4 pts bonus
          - Reservoir level < 30 %      → -4 pts penalty
          - Active warnings              → -3 pts per warning (capped at -9)
          - is_stale = True              → ×0.8 confidence multiplier on that region's score
        * Final score is clamped to [0, 100].

        Args:
            weather_records: List of dicts as returned by :meth:`fetch_theme_inputs`.

        Returns:
            Composite weather score as float in [0, 100].
        """
        if not weather_records:
            return 50.0  # neutral when no data

        base = 50.0
        per_record_max = 10.0  # maximum per-region absolute contribution
        total_contribution = 0.0

        for record in weather_records:
            deviation = float(record.get("deviation_pct", 0) or 0)
            reservoir = float(record.get("reservoir_level", 50) or 50)
            warnings = record.get("warnings", []) or []
            stale = bool(record.get("is_stale", False))

            region_score = 0.0

            # Rainfall deviation contribution
            if deviation > 10:
                region_score += 6.0
            elif deviation > 0:
                region_score += 3.0
            elif deviation < -20:
                region_score -= 10.0
            elif deviation < -10:
                region_score -= 6.0
            else:
                region_score -= 2.0

            # Reservoir bonus/penalty
            if reservoir > 70:
                region_score += 4.0
            elif reservoir < 30:
                region_score -= 4.0

            # Warnings penalty (capped)
            warning_penalty = min(len(warnings) * 3.0, 9.0)
            region_score -= warning_penalty

            # Confidence multiplier for stale data
            if stale:
                region_score *= 0.8

            total_contribution += region_score

        # Average contribution normalised to [−50, +50] range
        n = len(weather_records)
        avg_contribution = total_contribution / n

        # Scale: 1 avg point → 1 point deviation from 50, clamped
        final_score = base + avg_contribution
        return float(max(0.0, min(100.0, final_score)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_region(
        self,
        client: httpx.AsyncClient,
        region: str,
        date_str: str,
    ) -> dict:
        """
        Attempt to fetch weather data for a single *region* from IMD API.

        IMD does not expose a standard REST JSON API; this method makes a
        best-effort request to known endpoints and parses whatever is returned.
        On any failure it raises an exception so the caller can fall back.
        """
        # IMD's rainfall monitoring page (district/region-wise)
        url = f"{self._base_url}/Radar/index.jsp"
        params = {"region": region, "date": date_str}

        response = await client.get(url, params=params)
        response.raise_for_status()

        # IMD currently returns HTML pages; try to extract JSON-LD or embedded data
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            return self._parse_region_response(data, region, date_str)

        # HTML fallback: return mock with live attempt noted
        logger.debug("IMD region '%s' returned HTML (no structured data). Using mock.", region)
        record = _build_mock_record(region, date_str)
        record["source"] = "imd_html_fallback"
        return record

    @staticmethod
    def _parse_region_response(data: dict, region: str, date_str: str) -> dict:
        """Parse a structured JSON response from IMD for a single region."""
        return {
            "region": region,
            "data_date": date_str,
            "rainfall_actual": float(data.get("rainfall_actual", 0) or 0),
            "rainfall_normal": float(data.get("rainfall_normal", 0) or 0),
            "deviation_pct": float(data.get("deviation_pct", 0) or 0),
            "reservoir_level": float(data.get("reservoir_level", 50) or 50),
            "forecast": str(data.get("forecast", "N/A")),
            "warnings": list(data.get("warnings", [])),
            "source": "imd_api",
            "is_stale": False,
        }

    @staticmethod
    def _parse_monsoon_response(data: dict, date_str: str) -> dict:
        """Parse structured IMD monsoon JSON into a standard summary dict."""
        return {
            "season": data.get("season", "Southwest Monsoon"),
            "cumulative_rainfall_actual": float(data.get("cumulative_actual", 0) or 0),
            "cumulative_rainfall_normal": float(data.get("cumulative_normal", 0) or 0),
            "deviation_pct": float(data.get("deviation_pct", 0) or 0),
            "active_zones": list(data.get("active_zones", [])),
            "deficient_zones": list(data.get("deficient_zones", [])),
            "status_label": str(data.get("status_label", "Normal")),
            "source": "imd_api",
            "is_stale": False,
            "fetched_at": datetime.now(tz=_UTC).isoformat(),
        }

    @staticmethod
    def _mock_monsoon_status(date_str: str) -> dict:
        """Return a plausible mock monsoon status for use when the API is down."""
        return {
            "season": "Southwest Monsoon",
            "cumulative_rainfall_actual": 812.4,
            "cumulative_rainfall_normal": 858.0,
            "deviation_pct": -5.3,
            "active_zones": ["Central India", "Peninsular India"],
            "deficient_zones": ["Northwest India"],
            "status_label": "Below Normal",
            "source": "imd_mock",
            "is_stale": True,
            "fetched_at": datetime.now(tz=_UTC).isoformat(),
        }
