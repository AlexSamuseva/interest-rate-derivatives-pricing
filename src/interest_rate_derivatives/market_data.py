import datetime
import logging
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from interest_rate_derivatives.fred_client import FREDAPIClient

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class MarketDataClient:
    """Client for fetching real-time interest rate market data."""
    
    # Tenors in years mapped to FRED series IDs (Treasury Constant Maturity)
    TENOR_MAPPING = {
        1/12: "DGS1MO", 0.25: "DGS3MO", 0.5: "DGS6MO", 
        1.0: "DGS1", 2.0: "DGS2", 5.0: "DGS5", 
        10.0: "DGS10", 30.0: "DGS30"
    }
    
    def __init__(self, provider: str = "fred", api_key: Optional[str] = None):
        """
        Initialize market data client.
        
        Args:
            provider: Data provider (currently only "fred" supported)
            api_key: Optional FRED API key. If not provided, reads from FRED_API_KEY env var.
        """
        self.provider = provider
        if provider == "fred":
            self.fred_client = FREDAPIClient(api_key=api_key)
        else:
            self.fred_client = None

    def get_term_structure(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches the current interest rate term structure.
        
        Args:
            date: Optional date in YYYY-MM-DD format. If not provided, uses today's date.
            
        Returns:
            DataFrame with columns ["Maturity", "Rate"] containing the yield curve
        """
        logger.info(f"Fetching term structure from provider: {self.provider}")
        
        if self.provider != "fred" or not self.fred_client:
            return self._get_placeholder_curve()
        
        # Calculate date range for FRED API
        end_date = pd.to_datetime(date).strftime("%Y-%m-%d") if date else datetime.date.today().isoformat()
        start_date = (pd.to_datetime(end_date) - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        
        # Fetch all series
        series_ids = list(self.TENOR_MAPPING.values())
        data = self.fred_client.get_multiple_series(series_ids, start_date, end_date)
        
        if not data:
            logger.warning("No data from FRED API; falling back to placeholder curve")
            return self._get_placeholder_curve()
        
        # Extract latest rates
        latest_rates = []
        for tenor, series_id in self.TENOR_MAPPING.items():
            if series_id not in data:
                logger.debug(f"Series {series_id} not found in API response")
                continue
            
            observations = data[series_id].get("observations", [])
            if not observations:
                logger.debug(f"No observations for series {series_id}")
                continue
            
            # Get the latest non-null observation
            latest_obs = None
            for obs in reversed(observations):
                if obs.get("value") and obs["value"] != ".":
                    latest_obs = obs
                    break
            
            if latest_obs:
                try:
                    rate = float(latest_obs["value"]) / 100.0
                    latest_rates.append((tenor, rate))
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse rate for {series_id}: {latest_obs.get('value')}")
                    continue
        
        if latest_rates:
            df = pd.DataFrame(latest_rates, columns=["Maturity", "Rate"])
            df = df.sort_values("Maturity").reset_index(drop=True)
            return df
        
        logger.warning("No valid rates extracted from FRED data; falling back to placeholder curve")
        return self._get_placeholder_curve()
    
    @staticmethod
    def _get_placeholder_curve() -> pd.DataFrame:
        """Return a placeholder yield curve for testing."""
        maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.051, 0.052, 0.050, 0.048, 0.045, 0.046, 0.048]
        return pd.DataFrame({"Maturity": maturities, "Rate": rates})
