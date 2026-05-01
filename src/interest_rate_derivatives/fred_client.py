"""FRED (Federal Reserve Economic Data) API client for interest rate data."""

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class FREDAPIClient:
    """
    Client for fetching interest rate data from FRED API.
    
    Supports both authenticated and unauthenticated requests.
    API key should be stored in the FRED_API_KEY environment variable.
    
    Get a free API key at: https://fred.stlouisfed.org/user/register
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize FRED API client.
        
        Args:
            api_key: Optional API key. If not provided, reads from FRED_API_KEY env var.
                    Without API key, requests are limited.
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        
        if not self.api_key:
            logger.warning(
                "FRED_API_KEY not set. Requests will be limited. "
                "Get a free key at https://fred.stlouisfed.org/user/register"
            )
    
    def get_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object] | None:
        """
        Fetch a single series from FRED API.
        
        Args:
            series_id: FRED series ID (e.g., "DGS10" for 10-year Treasury)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with observations data or None if request fails
        """
        from urllib.parse import urlencode
        
        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "file_type": "json",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        
        query_string = urlencode(params)
        full_url = f"{url}?{query_string}"
        
        try:
            request = Request(full_url, headers={"User-Agent": "ir-derivatives/0.1"})
            with urlopen(request, timeout=10) as response:
                content = response.read().decode("utf-8")
                return json.loads(content)
        except HTTPError as e:
            logger.exception("HTTP error fetching %s: %s %s", series_id, e.code, e.reason)
            return None
        except URLError as e:
            logger.exception("Network error fetching %s: %s", series_id, e.reason)
            return None
        except TimeoutError:
            logger.exception("Timeout fetching %s", series_id)
            return None
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse JSON response for %s: %s", series_id, e)
            return None
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error fetching %s: %s", series_id, e)
            return None
    
    def get_multiple_series(
        self,
        series_ids: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, object]:
        """
        Fetch multiple series from FRED API.
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping series_id to observations data
        """
        results = {}
        for series_id in series_ids:
            data = self.get_series(series_id, start_date, end_date)
            if data:
                results[series_id] = data
        return results
