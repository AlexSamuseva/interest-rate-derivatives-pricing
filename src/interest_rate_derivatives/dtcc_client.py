"""DTCC data client for downloading cumulative CFTC/DTCC reports (swaption prices).

This is a lightweight proof-of-concept client that mirrors the approach
used in the repository's `test_data.py` PoC. It downloads the ZIP report,
extracts the first CSV and normalises column names.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DTCCClient:
    BASE_URL = "https://pddata.dtcc.com/ppd/api/report/cumulative/cftc/"

    def __init__(self) -> None:
        pass

    def fetch_swaptions(self, target_date: str | None = None) -> pd.DataFrame:
        """
        Download and return DTCC swaption rows as a cleaned DataFrame.

        Args:
            target_date: date string in format used by DTCC filenames, e.g.
                         "2026_05_06". If not provided, defaults to today's
                         date in YYYY_MM_DD format (best-effort).

        Returns:
            pd.DataFrame: cleaned DataFrame (columns lowercased). Returns
            an empty DataFrame if the file cannot be downloaded or parsed.
        """
        if not target_date:
            # best-effort default formatting — caller should provide explicit date

            target_date = datetime.now(timezone.utc).strftime("%Y_%m_%d")

        url = f"{self.BASE_URL}CFTC_CUMULATIVE_RATES_{target_date}.zip"
        logger.info("Fetching DTCC report: %s", url)

        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                logger.warning(
                    "DTCC response status %s for %s", response.status_code, url
                )
                return pd.DataFrame()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    df = pd.read_csv(f, low_memory=False)

            # Standardise column names
            df.columns = [c.lower().strip() for c in df.columns]

            # Basic filtering: interest-rate instruments with strike price and USD notional
            mask = pd.Series([True] * len(df))
            if "asset class" in df.columns:
                mask &= df["asset class"].astype(str).str.upper() == "IR"
            if "strike price" in df.columns:
                mask &= df["strike price"].notna()
            if "notional currency-leg 1" in df.columns:
                mask &= df["notional currency-leg 1"].astype(str).str.upper() == "USD"

            swaptions = df.loc[mask].copy()

            # Attempt numeric cleaning for common columns
            for col in [
                "option premium amount",
                "notional amount-leg 1",
                "strike price",
            ]:
                if col in swaptions.columns:
                    swaptions[col] = pd.to_numeric(
                        swaptions[col]
                        .astype(str)
                        .str.replace(r"[^0-9.eE+-]", "", regex=True),
                        errors="coerce",
                    )

        except Exception:  # pragma: no cover - network/IO
            logger.exception("Failed to download or parse DTCC report: %s", url)
            return pd.DataFrame()
        else:
            return swaptions
