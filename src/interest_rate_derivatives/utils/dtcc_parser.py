"""Utilities to transform DTCC swaption DataFrame rows into
CalibrationInstrument instances used by the Hull-White calibrator.

The implementation contains reasonable defaults and heuristics; callers
should validate units (strike / premium) for production use.
"""

from __future__ import annotations

import logging

import pandas as pd

from interest_rate_derivatives.pricing.calibration import CalibrationInstrument
from interest_rate_derivatives.utils.periods import (
    PERIOD_UNIT_ALIASES,
    normalize_period_unit,
)

logger = logging.getLogger(__name__)


_PERIOD_UNIT_ALIASES = PERIOD_UNIT_ALIASES


_CODE_TO_INTERVAL = {
    # Common CFTC/DTCC frequency codes observed in rates reporting.
    "A001": ("YEAR", 1.0),
    "A004": ("MNTH", 3.0),
    "A005": ("MNTH", 1.0),
    "A020": ("MNTH", 6.0),
}


def _to_float(x: object) -> float | None:
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            s = str(x).strip()
            s = s.replace(",", "")
            return float(s)
        except (TypeError, ValueError):
            return None


def _convert_strike(raw: object, strike_format: str | None) -> float:
    """Convert raw strike to decimal (e.g. 0.025 for 2.5%).

    Heuristics:
    - If strike_format provided, use it ('decimal','percent','bps').
    - If raw > 100: assume bps -> /10000
    - If 1 < raw <= 100: assume percent -> /100
    - If raw <= 1: assume already decimal
    """
    val = _to_float(raw)
    if val is None:
        msg = "Missing strike value"
        raise ValueError(msg)

    if strike_format == "decimal":
        return val
    if strike_format == "percent":
        return val / 100.0
    if strike_format == "bps":
        return val / 10000.0

    # Infer
    if val > 100:
        return val / 10000.0
    if val > 1:
        return val / 100.0
    return val


def _normalize_period_unit(raw: object) -> str | None:
    if raw is None:
        return None

    raw_text = str(raw).strip().upper()
    if raw_text in _CODE_TO_INTERVAL:
        return _CODE_TO_INTERVAL[raw_text][0]
    return normalize_period_unit(raw_text)


def _extract_payment_interval(
    row: pd.Series,
) -> tuple[str | None, float | None]:
    """Extract payment interval as (period_unit, multiplier) from a DTCC row.

    DTCC stores payment cadence as "every <multiplier> <period_unit>", where
    period_unit is typically YEAR/MNTH/DAIL and multiplier is numeric.
    """

    interval_fields = [
        (
            "fixed rate payment frequency period-leg 1",
            "fixed rate payment frequency period multiplier-leg 1",
        ),
        (
            "floating rate payment frequency period-leg 1",
            "floating rate payment frequency period multiplier-leg 1",
        ),
        (
            "fixed rate payment frequency period-leg 2",
            "fixed rate payment frequency period multiplier-leg 2",
        ),
        (
            "floating rate payment frequency period-leg 2",
            "floating rate payment frequency period multiplier-leg 2",
        ),
    ]

    for period_col, multiplier_col in interval_fields:
        if period_col not in row.index:
            continue

        period_raw = row.get(period_col)
        multiplier_raw = row.get(multiplier_col)
        period_text = str(period_raw).strip().upper() if period_raw is not None else ""

        if period_text in _CODE_TO_INTERVAL:
            return _CODE_TO_INTERVAL[period_text]

        period_unit = _normalize_period_unit(period_raw)
        multiplier = _to_float(multiplier_raw)
        if period_unit is None or multiplier is None or multiplier <= 0:
            continue

        return period_unit, float(multiplier)

    return None, None


def _interval_to_standard_frequency(period_unit: str, multiplier: float) -> int | None:
    if multiplier <= 0:
        return None

    if period_unit == "YEAR":
        periods_per_year = 1.0 / multiplier
    elif period_unit == "MNTH":
        periods_per_year = 12.0 / multiplier
    elif period_unit == "WEEK":
        periods_per_year = 52.0 / multiplier
    elif period_unit == "DAIL":
        periods_per_year = 365.25 / multiplier
    else:
        return None

    rounded = round(periods_per_year)
    return (
        rounded
        if rounded in {1, 2, 4, 12} and abs(periods_per_year - rounded) < 1e-6
        else None
    )


def _extract_frequency(
    period_unit: str | None,
    multiplier: float | None,
    fallback: int = 2,
) -> int:
    """Extract a standard annual frequency when the interval is standard.

    For bespoke schedules (e.g. every 5 months), this returns the provided
    fallback and schedule generation should use explicit period+multiplier.
    """

    if period_unit is not None and multiplier is not None:
        standard_frequency = _interval_to_standard_frequency(period_unit, multiplier)
        if standard_frequency is not None:
            return standard_frequency

    logger.debug("Row frequency missing or unsupported; falling back to %s", fallback)
    return fallback


def dtcc_df_to_calibration_instruments(
    df: pd.DataFrame,
    reference_date: str | None = None,
    frequency: int | None = None,
    strike_format: str | None = None,
    price_format: str | None = None,
    min_strike: float = 0.0001,
    max_strike: float = 0.20,
    min_price: float = 1e-6,
    max_price: float = 1.0,
) -> list[CalibrationInstrument]:
    """Transform DTCC DataFrame into a list of `CalibrationInstrument`.

    Filters out invalid or implausible rows based on sanity checks.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DTCC swaption data.
    reference_date : str, optional
        Reference date for year calculations. If None, uses execution timestamp.
    frequency : int, optional
        Fallback payment frequency used only when DTCC row data is missing or
        cannot be interpreted. Defaults to 2 (semi-annual).
    strike_format : str, optional
        Strike unit format: 'decimal', 'percent', 'bps', or None (auto-infer).
    price_format : str, optional
        Premium unit format: 'dollar', 'per_notional', or None (auto-detect).
    min_strike : float
        Minimum plausible strike rate (decimal). Default 0.01%. Skip if below.
    max_strike : float
        Maximum plausible strike rate (decimal). Default 20%. Skip if above.
    min_price : float
        Minimum plausible price per unit notional. Default 1e-6. Skip if below.
    max_price : float
        Maximum plausible price per unit notional. Default 100%. Skip if above.

    Returns
    -------
    list[CalibrationInstrument]
        Valid calibration instruments.
    """
    instruments: list[CalibrationInstrument] = []
    if df is None or df.empty:
        return instruments

    if reference_date:
        ref = pd.to_datetime(reference_date)
    elif "execution timestamp" in df.columns:
        ref = pd.to_datetime(df["execution timestamp"].max(), errors="coerce")
        if pd.isna(ref):
            ref = pd.Timestamp.today()
    else:
        ref = pd.Timestamp.today()

    skipped_count = 0
    for idx, row in df.iterrows():
        try:
            # Step 1: Validate premium (most critical filter)
            premium = _to_float(row.get("option premium amount"))
            if premium is None or premium <= 0:
                skipped_count += 1
                logger.debug(
                    "Row %s: Skipped - premium is None or <= 0 (premium=%s)",
                    idx,
                    premium,
                )
                continue

            # Step 2: Validate notional
            notional = _to_float(row.get("notional amount-leg 1")) or _to_float(
                row.get("notional amount")
            )
            if notional is None or notional <= 0:
                skipped_count += 1
                logger.debug(
                    "Row %s: Skipped - notional is None or <= 0 (notional=%s)",
                    idx,
                    notional,
                )
                continue

            # Step 3: Parse expiration date
            exp = pd.to_datetime(row.get("expiration date"), errors="coerce")
            if pd.isna(exp):
                skipped_count += 1
                logger.debug("Row %s: Skipped - missing expiration date", idx)
                continue
            option_expiry = max((exp - ref).days, 0) / 365.25

            # Step 4: Validate expiry is reasonable (0.01Y to 30Y)
            if option_expiry < 0.01 or option_expiry > 30:
                skipped_count += 1
                logger.debug(
                    "Row %s: Skipped - expiry out of range (%.3fY)", idx, option_expiry
                )
                continue

            # Step 5: Parse strike and convert to decimal
            strike_raw = row.get("strike price")
            if strike_raw is None or (
                isinstance(strike_raw, float) and pd.isna(strike_raw)
            ):
                skipped_count += 1
                logger.debug("Row %s: Skipped - missing strike price", idx)
                continue

            strike_rate = _convert_strike(strike_raw, strike_format)

            # Step 6: Validate strike is in reasonable range
            if strike_rate < min_strike or strike_rate > max_strike:
                skipped_count += 1
                logger.debug(
                    "Row %s: Skipped - strike out of range (%.3f%%)",
                    idx,
                    strike_rate * 100,
                )
                continue

            # Step 7: Compute swap tenor
            swap_start = option_expiry
            swap_tenor = None
            if "maturity date" in df.columns:
                mat = pd.to_datetime(row.get("maturity date"), errors="coerce")
                if not pd.isna(mat):
                    swap_tenor = max((mat - ref).days, 0) / 365.25 - swap_start

            if swap_tenor is None or swap_tenor <= 0:
                # Fallback: use 'swap tenor' column or default 5y
                cand = row.get("swap tenor") or row.get("swap_tenor")
                swap_tenor = _to_float(cand) if cand is not None else 5.0
                if swap_tenor is None:
                    swap_tenor = 5.0

            # Step 8: Validate swap tenor is reasonable (0.01Y to 30Y)
            if not 0.01 <= swap_tenor <= 30:
                skipped_count += 1
                logger.debug(
                    "Row %s: Skipped - swap tenor out of range (%.3fY)", idx, swap_tenor
                )
                continue

            # Step 9: Normalize price to per-unit-notional
            if price_format == "per_notional":
                price = float(premium)
            elif 0 < premium < 1:
                logger.warning(
                    "Row %s: Auto-detected premium %.6f as per-notional; set "
                    "price_format='per_notional' to make this explicit",
                    idx,
                    premium,
                )
                price = float(premium)
            else:
                price = float(premium) / float(notional)

            # Step 10: Validate price is in reasonable range
            if price < min_price or price > max_price:
                skipped_count += 1
                logger.debug(
                    "Row %s: Skipped - price out of range (%.6f per notional)",
                    idx,
                    price,
                )
                continue

            # Step 11: Create instrument
            period_unit, period_multiplier = _extract_payment_interval(row)
            row_frequency = _extract_frequency(
                period_unit,
                period_multiplier,
                fallback=frequency or 2,
            )
            label = f"{option_expiry:.2f}Yx{swap_tenor:.2f}Y"
            instr = CalibrationInstrument(
                option_expiry=float(option_expiry),
                swap_tenor=float(swap_tenor),
                swap_start=float(swap_start),
                strike_rate=float(strike_rate),
                market_price=float(price),
                frequency=int(row_frequency),
                payment_period_unit=period_unit,
                payment_period_multiplier=period_multiplier,
                weight=1.0,
                label=label,
            )
            instruments.append(instr)
            logger.debug(
                "Row %s: Accepted - %s strike=%.3f%% price=%.6f",
                idx,
                label,
                strike_rate * 100,
                price,
            )

        except (ValueError, TypeError, KeyError, OverflowError) as e:
            skipped_count += 1
            logger.debug("Row %s: Skipped - exception: %s", idx, e)

    logger.info(
        "Converted %s / %s rows to CalibrationInstrument (%s skipped due to validation)",
        len(instruments),
        len(df),
        skipped_count,
    )
    return instruments
