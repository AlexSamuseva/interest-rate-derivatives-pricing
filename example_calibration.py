"""
End-to-end example: Hull-White calibration using DTCC swaption data and FRED yields.

This script demonstrates the complete workflow:
1. Fetch interest rate term structure from FRED
2. Build a discount curve
3. Fetch swaption prices from DTCC
4. Transform DTCC data into calibration instruments
5. Calibrate the Hull-White model
"""

import logging
import sys

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

logger.info("%s", "=" * 70)
logger.info("Hull-White Calibration: DTCC Swaptions + FRED Yields")
logger.info("%s", "=" * 70)
logger.info("")

# ============================================================================
# Step 1: Fetch interest rate term structure from FRED
# ============================================================================
logger.info("Step 1: Fetching interest rate data from FRED...")
logger.info("%s", "-" * 70)

try:
    from interest_rate_derivatives.market_data import MarketDataClient
    from interest_rate_derivatives.utils.curves import DiscountCurve

    # Instantiate FRED client
    mdc = MarketDataClient(provider="fred")

    # Fetch today's term structure (uses placeholder if FRED unavailable)
    yield_curve_df = mdc.get_term_structure()

    logger.info("✓ Retrieved %s yield curve points", len(yield_curve_df))
    logger.info("")
    logger.info("Yield Curve (first 5 points):")
    logger.info("%s", yield_curve_df.head())
    logger.info("")

    # Build discount curve
    curve = DiscountCurve(
        maturities=yield_curve_df["Maturity"].values,
        zero_rates=yield_curve_df["Rate"].values,
    )
    logger.info("✓ Discount curve built")
    logger.info("")

except (ImportError, AttributeError, KeyError, OSError, RuntimeError, ValueError):
    logger.exception("✗ Failed to fetch FRED data")
    sys.exit(1)


# ============================================================================
# Step 2: Fetch DTCC swaption data (with date specification)
# ============================================================================
logger.info("Step 2: Fetching DTCC swaption pricing data...")
logger.info("%s", "-" * 70)

try:
    from interest_rate_derivatives.dtcc_client import DTCCClient

    dtcc_client = DTCCClient()

    # Use a specific date or today's date
    target_date = "2026_05_06"  # Example: May 6, 2026 (format: YYYY_MM_DD)
    logger.info("Attempting to fetch DTCC data for: %s", target_date)

    raw_swaptions = dtcc_client.fetch_swaptions(target_date=target_date)

    if raw_swaptions.empty:
        logger.warning(
            "⚠ Warning: DTCC download returned empty (this is expected if date not available)"
        )
        logger.info("  Creating synthetic calibration instruments for demonstration...")
        logger.info("")

        # Create synthetic data for testing
        synthetic_data = pd.DataFrame(
            {
                "asset class": ["IR", "IR", "IR", "IR"],
                "strike price": [2.5, 2.5, 2.5, 2.5],
                "option premium amount": [0.025, 0.035, 0.015, 0.028],
                "notional amount-leg 1": [
                    10_000_000,
                    10_000_000,
                    10_000_000,
                    10_000_000,
                ],
                "notional currency-leg 1": ["USD", "USD", "USD", "USD"],
                "execution timestamp": [
                    "2026-05-06",
                    "2026-05-06",
                    "2026-05-06",
                    "2026-05-06",
                ],
                "expiration date": [
                    "2027-05-06",
                    "2027-05-06",
                    "2028-05-06",
                    "2029-05-06",
                ],
                "maturity date": [
                    "2032-05-06",
                    "2032-05-06",
                    "2033-05-06",
                    "2034-05-06",
                ],
                "fixed rate payment frequency period-leg 1": [
                    "A020",
                    "A020",
                    "A004",
                    "A001",
                ],
                "fixed rate payment frequency period multiplier-leg 1": [1, 1, 1, 1],
            }
        )
        raw_swaptions = synthetic_data
        logger.info("✓ Using %s synthetic swaption records", len(raw_swaptions))
    else:
        logger.info("✓ Retrieved %s swaption records from DTCC", len(raw_swaptions))

    logger.info("Columns in data: %s", list(raw_swaptions.columns[:8]))
    logger.info("")

except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
    logger.exception("✗ Failed to fetch DTCC data")
    sys.exit(1)


# ============================================================================
# Step 3: Transform DTCC data into CalibrationInstrument
# ============================================================================
logger.info("Step 3: Transforming DTCC data into calibration instruments...")
logger.info("%s", "-" * 70)

try:
    from interest_rate_derivatives.utils.dtcc_parser import (
        dtcc_df_to_calibration_instruments,
    )

    instruments = dtcc_df_to_calibration_instruments(
        raw_swaptions,
        discount_curve=curve,
        reference_date="2026-05-06",
        is_payer=True,
        strike_format="decimal",  # strikes in %, e.g., 2.5 = 0.025
        price_format="dollar",  # premiums in dollars; will divide by notional
    )

    if not instruments:
        logger.error("✗ No calibration instruments created")
        sys.exit(1)

    logger.info("✓ Created %s calibration instruments", len(instruments))
    logger.info("")
    logger.info("First 3 instruments:")
    for i, instr in enumerate(instruments[:3], 1):
        logger.info("  %s. %s", i, instr.label)
        logger.info(
            "     Expiry: %.3fY | Tenor: %.3fY | Freq: %s | Strike: %.2f%% | Price: %.6f",
            instr.option_expiry,
            instr.swap_tenor,
            instr.frequency,
            instr.strike_rate * 100,
            instr.market_price,
        )
    logger.info("")

except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
    logger.exception("✗ Failed to transform DTCC data")
    sys.exit(1)


# ============================================================================
# Step 4: Calibrate Hull-White model
# ============================================================================
logger.info("Step 4: Calibrating Hull-White model...")
logger.info("%s", "-" * 70)

try:
    from interest_rate_derivatives.pricing.calibration import HullWhiteCalibrator

    calibrator = HullWhiteCalibrator(
        discount_factor=curve, instruments=instruments, is_payer=True
    )

    logger.info("Running L-BFGS-B optimization...")
    result = calibrator.calibrate(
        a_init=0.1,  # Initial mean reversion speed
        sigma_init=0.01,  # Initial volatility
        bounds=((0.001, 2.0), (0.0001, 0.10)),  # Search bounds
    )

    logger.info("")
    logger.info("✓ Calibration complete!")
    logger.info("  Converged: %s", result.converged)
    logger.info("  Message: %s", result.message)
    logger.info("")
    logger.info("Calibrated Parameters:")
    logger.info("  a (mean reversion speed): %.4f", result.a)
    logger.info("  sigma (volatility): %.6f", result.sigma)
    logger.info("  Objective value: %.2e", result.objective_value)
    logger.info("")

    # # Print calibration diagnostics
    # print("Instrument Pricing Errors (basis points):")
    # print(f"{'Label':<15} {'Market':<12} {'Model':<12} {'Error (bps)':<12}")
    # print("-" * 51)
    # for label, mkt_price, mdl_price, error_bps in zip(
    #     result.labels,
    #     result.market_prices,
    #     result.model_prices,
    #     result.errors_bps
    # ):
    #     print(f"{label:<15} {mkt_price:<12.6f} {mdl_price:<12.6f} {error_bps:<12.2f}")

    logger.info("")
    logger.info("%s", "=" * 70)
    logger.info("Calibration Summary:")
    logger.info("%s", result.summary())
    logger.info("%s", "=" * 70)

except (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    np.linalg.LinAlgError,
):
    logger.exception("✗ Failed to calibrate")
    sys.exit(1)

logger.info("")
logger.info("✓ End-to-end workflow complete!")
