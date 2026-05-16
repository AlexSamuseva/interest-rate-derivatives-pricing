"""
Hull-White Model Calibration to Market Swaption Prices.

This module calibrates the Hull-White one-factor model parameters
(a, sigma) to a basket of market swaption prices by minimising the
weighted sum of squared relative pricing errors.

Calibration Overview
--------------------
The Hull-White model has two free parameters:
    a     : mean reversion speed
    sigma : short rate volatility

The initial discount curve P(0, T) is taken directly from the market
and does not need to be calibrated — it is an input, not a free parameter.

The calibration targets are market swaption prices. We minimise:

    min_{a, sigma > 0} sum_k w_k * [(V_k^HW - V_k^mkt) / V_k^mkt]^2

where w_k are instrument weights and V_k^HW is the model price computed
via Jamshidian's decomposition for given (a, sigma).

The optimisation uses L-BFGS-B with box constraints:
    a     in (0.001, 2.0)
    sigma in (0.0001, 0.10)

Calibration Basket
------------------
A typical calibration basket covers a grid of expiries and tenors:
    1Y x 2Y, 1Y x 5Y, 2Y x 3Y, 2Y x 5Y, 5Y x 5Y

These are chosen to span the range of maturities relevant to the
portfolio being hedged or risk-managed.

References
----------
Hull, J., & White, A. (1990). Pricing interest-rate derivative securities.
    The Review of Financial Studies, 3(4), 573-592.
Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and
    Practice (2nd ed.). Springer Finance. Chapter 3.3.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from interest_rate_derivatives.models.hull_white import HullWhiteModel
from interest_rate_derivatives.pricing.swaption import SwaptionPricer
from interest_rate_derivatives.utils.curves import generate_payment_schedule

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class CalibrationInstrument:
    """
    A single market swaption used as a calibration target.

    Each instrument represents one market-quoted swaption with its
    observed price. The calibrator fits (a, sigma) so that the
    model reproduces these prices as closely as possible.

    Parameters
    ----------
    option_expiry : float
        Swaption expiry in years from today.
    swap_tenor : float
        Underlying swap tenor in years.
    swap_start : float
        Swap effective date in years. Typically equals option_expiry
        for a standard at-expiry swaption.
    strike_rate : float
        Fixed rate K of the underlying swap.
        Use the par swap rate for ATM instruments.
    market_price : float
        Observed market price of the swaption per unit notional.
    frequency : int
        Payment frequency of the underlying swap. Default 2 (semi-annual).
    payment_period_unit : str, optional
        Explicit payment period unit for bespoke schedules (e.g. 'MNTH',
        'YEAR', 'DAIL'). When set with `payment_period_multiplier`, this
        takes precedence over `frequency` in schedule generation.
    payment_period_multiplier : float, optional
        Interval multiplier for bespoke schedules. Example: unit='MNTH',
        multiplier=5 means a payment every 5 months.
    weight : float
        Calibration weight for this instrument. Higher weight means
        the calibrator prioritises fitting this instrument more
        closely. Default 1.0 (equal weights).
    label : str
        Human-readable label. Auto-generated if not provided.
        Example: '1Yx5Y' for a 1-year into 5-year swaption.
    """

    option_expiry: float
    swap_tenor: float
    swap_start: float
    strike_rate: float
    market_price: float
    frequency: int = 2
    payment_period_unit: str | None = None
    payment_period_multiplier: float | None = None
    weight: float = 1.0
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.option_expiry:.0f}Yx{self.swap_tenor:.0f}Y"


@dataclass
class CalibrationResult:
    """
    Results of a Hull-White calibration.

    Contains the calibrated parameters, diagnostics, and per-instrument
    pricing errors for assessing calibration quality.

    Parameters
    ----------
    a : float
        Calibrated mean reversion speed.
    sigma : float
        Calibrated short rate volatility.
    objective_value : float
        Final objective function value (sum of squared relative errors).
        Lower is better. Values below 1e-6 indicate excellent fit.
    model_prices : list of float
        Model swaption prices at the calibrated parameters.
    market_prices : list of float
        Target market swaption prices.
    errors_bps : list of float
        Pricing errors in basis points of notional for each instrument.
        Computed as (model_price - market_price) * 10_000.
    converged : bool
        True if the optimiser converged to a solution.
    message : str
        Optimiser convergence message.
    labels : list of str
        Instrument labels for display.
    """

    a: float
    sigma: float
    objective_value: float
    model_prices: list[float]
    market_prices: list[float]
    errors_bps: list[float]
    converged: bool
    message: str
    labels: list[str]

    def summary(self) -> str:
        """
        Return a formatted summary of the calibration results.

        Returns
        -------
        str
            Multi-line summary string showing calibrated parameters
            and per-instrument pricing errors.
        """
        lines = [
            "Hull-White Calibration Results",
            "=" * 50,
            f"  a (mean reversion)  : {self.a:.6f}",
            f"  sigma (volatility)  : {self.sigma:.6f}",
            f"  Objective value     : {self.objective_value:.2e}",
            f"  Converged           : {self.converged}",
            "",
            f"  {'Instrument':>12} {'Market':>12} {'Model':>12} {'Error (bps)':>14}",
            "  " + "-" * 54,
        ]
        for label, mkt, mdl, err in zip(
            self.labels,
            self.market_prices,
            self.model_prices,
            self.errors_bps,
            strict=True,
        ):
            lines.append(f"  {label:>12} {mkt:>12.6f} {mdl:>12.6f} {err:>+14.2f}")
        return "\n".join(lines)


class HullWhiteCalibrator:
    """
    Calibrates the Hull-White model to a basket of market swaptions.

    Minimises the weighted sum of squared relative pricing errors
    between model prices (computed via Jamshidian's decomposition)
    and observed market prices, over the parameter space (a, sigma).

    The discount curve is fixed as an input — it is taken directly
    from the market and is not part of the optimisation.

    Parameters
    ----------
    discount_factor : Callable
        Market discount curve P(0, T). Either a DiscountCurve built
        from FRED data or a FlatCurve for testing.
    instruments : list of CalibrationInstrument
        Calibration basket of market swaptions.
    is_payer : bool
        If True, all instruments are payer swaptions. Default True.

    Examples
    --------
    >>> from interest_rate_derivatives.utils.curves import FlatCurve
    >>> from interest_rate_derivatives.utils.curves import generate_payment_schedule
    >>> from interest_rate_derivatives.pricing.swaption import SwaptionPricer
    >>> from interest_rate_derivatives.models.hull_white import HullWhiteModel
    >>> curve = FlatCurve(0.05)
    >>> model = HullWhiteModel(a=0.1, sigma=0.01, discount_factor=curve)
    >>> pricer = SwaptionPricer(model)
    >>> dates, dcfs = generate_payment_schedule(1.0, 6.0, frequency=2)
    >>> atm = pricer.par_swap_rate(1.0, dates, dcfs)
    >>> mkt_price = pricer.price(1.0, dates, dcfs, atm)['price']
    >>> instr = CalibrationInstrument(1.0, 5.0, 1.0, atm, mkt_price)
    >>> calibrator = HullWhiteCalibrator(curve, [instr])
    >>> result = calibrator.calibrate()
    >>> result.converged
    True
    """

    def __init__(
        self,
        discount_factor: Callable,
        instruments: list[CalibrationInstrument],
        is_payer: bool = True,
    ) -> None:
        self.discount_factor = discount_factor
        self.instruments = instruments
        self.is_payer = is_payer

    def _model_price(
        self,
        a: float,
        sigma: float,
        instrument: CalibrationInstrument,
    ) -> float:
        """
        Compute the model swaption price for given parameters (a, sigma).

        Builds a HullWhiteModel and SwaptionPricer with the given
        parameters and prices the instrument using Jamshidian's
        decomposition.

        This method is called repeatedly by the optimiser at each
        trial value of (a, sigma).

        Parameters
        ----------
        a : float
            Trial mean reversion speed.
        sigma : float
            Trial short rate volatility.
        instrument : CalibrationInstrument
            The swaption to price.

        Returns
        -------
        float
            Model swaption price per unit notional.
        """
        model = HullWhiteModel(
            a=a,
            sigma=sigma,
            discount_factor=self.discount_factor,
        )
        pricer = SwaptionPricer(model)

        payment_dates, dcfs = generate_payment_schedule(
            swap_start=instrument.swap_start,
            swap_end=instrument.swap_start + instrument.swap_tenor,
            frequency=instrument.frequency,
            period_unit=instrument.payment_period_unit,
            period_multiplier=instrument.payment_period_multiplier,
        )

        result = pricer.price(
            option_expiry=instrument.option_expiry,
            payment_dates=payment_dates,
            day_count_fractions=dcfs,
            strike_rate=instrument.strike_rate,
            notional=1.0,
            is_payer=self.is_payer,
        )
        price = result["price"]
        assert isinstance(price, float)
        return price

    def _safe_model_price(
        self,
        a: float,
        sigma: float,
        instrument: CalibrationInstrument,
    ) -> float:
        """
        Compute model price with error handling for invalid parameters.

        Returns a large penalty value if pricing fails — this guides
        the optimiser away from degenerate parameter combinations.

        Parameters
        ----------
        a : float
            Trial mean reversion speed.
        sigma : float
            Trial short rate volatility.
        instrument : CalibrationInstrument
            The swaption to price.

        Returns
        -------
        float
            Model price or large penalty on failure.
        """
        try:
            return self._model_price(a, sigma, instrument)
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            return instrument.market_price * 1e6

    def _objective(self, params: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        """
        Compute the weighted sum of squared relative pricing errors.

        Objective function minimised by the optimiser:

            f(a, sigma) = sum_k w_k * [(V_k^HW - V_k^mkt) / V_k^mkt]^2

        Relative errors are used so that instruments with different
        price magnitudes contribute equally to the objective.

        Parameters
        ----------
        params : ndarray of shape (2,)
            Trial parameters [a, sigma].

        Returns
        -------
        float
            Objective function value. Lower is better.
        """
        a = float(params[0])
        sigma = float(params[1])
        total = 0.0
        for instr in self.instruments:
            model_p = self._safe_model_price(a, sigma, instr)
            relative_error = (model_p - instr.market_price) / max(
                instr.market_price, 1e-10
            )
            total += instr.weight * relative_error**2
        return total

    def calibrate(
        self,
        a_init: float = 0.1,
        sigma_init: float = 0.01,
        bounds: tuple[tuple[float, float], tuple[float, float]] = (
            (0.001, 2.0),
            (0.0001, 0.10),
        ),
    ) -> CalibrationResult:
        """
        Run the calibration using L-BFGS-B optimisation.

        Minimises the objective function over (a, sigma) subject to
        positivity constraints. Uses the L-BFGS-B algorithm which
        handles box constraints and is efficient for smooth objectives.

        Parameters
        ----------
        a_init : float
            Initial guess for mean reversion speed. Default 0.1.
        sigma_init : float
            Initial guess for volatility. Default 0.01.
        bounds : tuple of pairs
            Box constraints for (a, sigma).
            Default: a in (0.001, 2.0), sigma in (0.0001, 0.10).

        Returns
        -------
        CalibrationResult
            Calibrated parameters and diagnostics.
        """
        x0 = np.array([a_init, sigma_init])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt_result = minimize(
                self._objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
            )

        a_cal = float(opt_result.x[0])
        sigma_cal = float(opt_result.x[1])

        # Compute diagnostics using _safe_model_price to avoid crashes on outliers
        model_prices = []
        errors_bps = []
        failed_instruments = []
        for instr in self.instruments:
            mp = self._safe_model_price(a_cal, sigma_cal, instr)
            model_prices.append(mp)
            # Mark as NaN if pricing failed (penalty was returned)
            if mp > instr.market_price * 100:
                errors_bps.append(float("nan"))
                failed_instruments.append(instr.label)
            else:
                errors_bps.append((mp - instr.market_price) * 10_000)

        if failed_instruments:
            logger.warning(
                "Failed to price %s instrument(s) at calibrated parameters: %s%s",
                len(failed_instruments),
                ", ".join(failed_instruments[:5]),
                "..." if len(failed_instruments) > 5 else "",
            )

        return CalibrationResult(
            a=a_cal,
            sigma=sigma_cal,
            objective_value=float(opt_result.fun),
            model_prices=model_prices,
            market_prices=[i.market_price for i in self.instruments],
            errors_bps=errors_bps,
            converged=bool(opt_result.success),
            message=str(opt_result.message),
            labels=[i.label for i in self.instruments],
        )
