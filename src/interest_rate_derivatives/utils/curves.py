"""
Discount curve construction and yield curve utilities.

This module provides tools for building discount factor curves from market
data and computing standard fixed-income quantities used in swaption pricing.

The primary class is DiscountCurve, which takes market zero rates at pillar
maturities and produces a smooth, continuous discount curve P(0, T) for any
maturity T using cubic spline interpolation on zero rates.

Connection to market data
--------------------------
The DiscountCurve class is designed to consume the output of the
MarketDataClient from market_data.py directly:

    from interest_rate_derivatives.market_data import MarketDataClient
    from interest_rate_derivatives.utils.curves import DiscountCurve

    client = MarketDataClient(provider="fred", api_key="...")
    df = client.get_term_structure()

    curve = DiscountCurve(
        maturities=df["Maturity"].tolist(),
        zero_rates=df["Rate"].tolist(),
    )

References
----------
Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and
    Practice (2nd ed.). Springer Finance. Chapter 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from interest_rate_derivatives.utils.periods import normalize_period_unit

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DiscountCurve:
    """
    A smooth discount curve P(0, T) built from market zero rates.

    Interpolates continuously-compounded zero rates using a cubic spline
    and converts to discount factors via P(0, T) = exp(-z(T) * T).

    Cubic spline interpolation is used because it produces a smooth curve
    with continuous first and second derivatives — this matters because the
    Hull-White model requires the instantaneous forward rate f(0, T) which
    involves the derivative of the zero rate curve.

    Parameters
    ----------
    maturities : list of float or ndarray
        Pillar maturities in years. Must be strictly increasing.
        Example: [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    zero_rates : list of float or ndarray
        Continuously-compounded zero rates at each pillar maturity.
        Must be expressed as decimals, not percentages.
        Example: [0.043, 0.044, 0.045, 0.047, 0.049, 0.050, 0.051]

    Examples
    --------
    >>> maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    >>> rates = [0.043, 0.044, 0.045, 0.047, 0.049, 0.050, 0.051]
    >>> curve = DiscountCurve(maturities, rates)
    >>> curve(5.0)
    0.7827...
    """

    def __init__(
        self,
        maturities: list[float] | NDArray,
        zero_rates: list[float] | NDArray,
    ) -> None:
        self._mats = np.asarray(maturities, dtype=float)
        self._rates = np.asarray(zero_rates, dtype=float)

        if self._mats.shape != self._rates.shape:
            msg = (
                "maturities and zero_rates must have the same length. "
                f"Got {len(self._mats)} maturities and {len(self._rates)} rates."
            )
            raise ValueError(msg)

        if not np.all(np.diff(self._mats) > 0):
            msg = f"maturities must be strictly increasing. Got: {self._mats.tolist()}"
            raise ValueError(msg)

        if np.any(self._rates < 0):
            msg = (
                "zero_rates must be non-negative. "
                f"Got negative rates at maturities: "
                f"{self._mats[self._rates < 0].tolist()}"
            )
            raise ValueError(msg)

        # Build cubic spline interpolator on zero rates
        # extrapolate=True means flat extrapolation beyond the pillar points
        self._spline = CubicSpline(
            self._mats,
            self._rates,
            extrapolate=True,
        )

    def zero_rate(self, t: float | NDArray) -> float | NDArray:
        """
        Return the interpolated continuously-compounded zero rate at maturity t.

        Zero rates are interpolated using a cubic spline fitted to the
        market pillar rates. Values outside the pillar range are
        extrapolated using the natural extension of the spline.

        Parameters
        ----------
        t : float or array-like
            Maturity in years. Must be positive.

        Returns
        -------
        float or ndarray
            Continuously-compounded zero rate z(T).
        """
        t_arr = np.asarray(t, dtype=float)
        return np.clip(self._spline(t_arr), 0.0, None)

    def __call__(self, t: float | NDArray) -> float | NDArray:
        """
        Return the discount factor P(0, T) = exp(-z(T) * T).

        This is the main method used by the Hull-White model. It converts
        the interpolated zero rate into a discount factor.

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Discount factor P(0, T) in the range (0, 1].
        """
        t_arr = np.asarray(t, dtype=float)
        z = self.zero_rate(t_arr)
        return np.exp(-z * t_arr)

    def instantaneous_forward_rate(self, t: float | NDArray) -> float | NDArray:
        """
        Return the instantaneous forward rate f(0, T) at maturity T.

        The instantaneous forward rate is defined as:

            f(0, T) = -d ln P(0, T) / dT = z(T) + T * dz(T)/dT

        It is computed analytically from the cubic spline derivative,
        which is more accurate than finite differences.

        This quantity is used directly in the Hull-White model to compute
        the time-dependent drift theta(t).

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Instantaneous forward rate f(0, T).
        """
        t_arr = np.asarray(t, dtype=float)
        z = self.zero_rate(t_arr)
        dz_dt = self._spline(t_arr, 1)
        return z + t_arr * dz_dt

    def forward_rate(
        self,
        t1: float | NDArray,
        t2: float | NDArray,
    ) -> float | NDArray:
        """
        Return the simply-compounded forward rate for the period [t1, t2].

        The forward rate F(t1, t2) is the rate agreed today for borrowing
        over the future period [t1, t2]. It is implied by the discount
        curve via no-arbitrage:

            F(t1, t2) = (P(0, t1) / P(0, t2) - 1) / (t2 - t1)

        Parameters
        ----------
        t1 : float or array-like
            Start of the forward period in years.
        t2 : float or array-like
            End of the forward period in years. Must be greater than t1.

        Returns
        -------
        float or ndarray
            Simply-compounded forward rate for [t1, t2].
        """
        t1_arr = np.asarray(t1, dtype=float)
        t2_arr = np.asarray(t2, dtype=float)

        if np.any(t2_arr <= t1_arr):
            msg = f"t2 must be greater than t1. Got t1={t1}, t2={t2}."
            raise ValueError(msg)

        P1 = self(t1_arr)
        P2 = self(t2_arr)
        tau = t2_arr - t1_arr
        return (P1 / P2 - 1.0) / tau

    def par_swap_rate(
        self,
        swap_start: float,
        payment_dates: list[float],
        day_count_fractions: list[float],
    ) -> float:
        """
        Compute the par swap rate for a vanilla interest rate swap.

        The par swap rate S is the fixed rate that makes the swap have
        zero NPV at inception:

            S = (P(0, T_0) - P(0, T_n)) / A

        where A = sum_i delta_i * P(0, T_i) is the annuity factor.

        Parameters
        ----------
        swap_start : float
            Swap effective date T_0 in years from today.
        payment_dates : list of float
            Fixed leg payment dates [T_1, ..., T_n] in years.
        day_count_fractions : list of float
            Day count fractions [delta_1, ..., delta_n] for each period.

        Returns
        -------
        float
            Par swap rate S.
        """
        P_start = self(swap_start)
        P_end = self(payment_dates[-1])
        annuity = sum(
            delta * self(T)
            for T, delta in zip(payment_dates, day_count_fractions, strict=True)
        )
        return float((P_start - P_end) / annuity)

    def annuity_factor(
        self,
        payment_dates: list[float],
        day_count_fractions: list[float],
    ) -> float:
        """
        Compute the present value annuity factor for a payment schedule.

        The annuity factor is:

            A = sum_{i=1}^{n} delta_i * P(0, T_i)

        It represents the present value of receiving one unit of currency
        at each payment date, scaled by the day count fraction.

        Parameters
        ----------
        payment_dates : list of float
            Payment dates in years.
        day_count_fractions : list of float
            Day count fractions for each period.

        Returns
        -------
        float
            Annuity factor A.
        """
        return float(
            sum(
                delta * self(T)
                for T, delta in zip(payment_dates, day_count_fractions, strict=True)
            )
        )

    def summary(self) -> dict[str, NDArray]:
        """
        Return a summary of the curve at its pillar points.

        Returns
        -------
        dict with keys:
            - maturities : ndarray of pillar maturities
            - zero_rates : ndarray of zero rates at pillars
            - discount_factors : ndarray of discount factors at pillars
            - forward_rates : ndarray of instantaneous forward rates at pillars
        """
        return {
            "maturities": self._mats.copy(),
            "zero_rates": self._rates.copy(),
            "discount_factors": self(self._mats),
            "forward_rates": self.instantaneous_forward_rate(self._mats),
        }


class FlatCurve:
    """
    A flat (constant) discount curve for testing and simple examples.

    Under a flat curve, the zero rate is constant at all maturities:

        z(T) = r  for all T
        P(0, T) = exp(-r * T)

    This is not realistic but is useful for unit testing the Hull-White
    model and Jamshidian pricing in isolation from curve construction.

    Parameters
    ----------
    flat_rate : float
        Constant continuously-compounded zero rate.

    Examples
    --------
    >>> curve = FlatCurve(0.05)
    >>> curve(1.0)
    0.9512...
    """

    def __init__(self, flat_rate: float) -> None:
        if flat_rate < 0:
            msg = f"flat_rate must be non-negative. Got {flat_rate}."
            raise ValueError(msg)
        self.flat_rate = float(flat_rate)

    def __call__(self, t: float | NDArray) -> float | NDArray:
        """
        Return discount factor P(0, T) = exp(-r * T).

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Discount factor.
        """
        return np.exp(-self.flat_rate * np.asarray(t, dtype=float))

    def instantaneous_forward_rate(self, t: float | NDArray) -> float | NDArray:
        """
        Return the instantaneous forward rate, which equals flat_rate everywhere.

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Instantaneous forward rate (constant = flat_rate).
        """
        return self.flat_rate * np.ones_like(np.asarray(t, dtype=float))

    def zero_rate(self, t: float | NDArray) -> float | NDArray:
        """
        Return the zero rate, which equals flat_rate everywhere.

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Zero rate (constant = flat_rate).
        """
        return self.flat_rate * np.ones_like(np.asarray(t, dtype=float))


def generate_payment_schedule(
    swap_start: float,
    swap_end: float,
    frequency: int = 2,
    period_unit: str | None = None,
    period_multiplier: float | None = None,
) -> tuple[list[float], list[float]]:
    """
    Generate a fixed-leg payment schedule for a vanilla interest rate swap.

    Produces evenly-spaced payment dates and their corresponding day count
    fractions using the Act/Act approximation.

    Parameters
    ----------
    swap_start : float
        Swap effective date in years from today.
    swap_end : float
        Swap maturity date in years from today.
    frequency : int
        Number of payments per year.
        1  = annual
        2  = semi-annual (most common for USD swaps)
        4  = quarterly
        12 = monthly
    period_unit : str, optional
        Custom DTCC-style payment period unit ('YEAR', 'MNTH', 'WEEK', 'DAIL').
        If provided with `period_multiplier`, schedule generation uses explicit
        interval steps instead of `frequency`.
    period_multiplier : float, optional
        Custom interval multiplier. Example: unit='MNTH', multiplier=5 means
        payments every 5 months.

    Returns
    -------
    payment_dates : list of float
        Payment dates [T_1, ..., T_n] in years from today.
    day_count_fractions : list of float
        Day count fractions [delta_1, ..., delta_n].

    Examples
    --------
    >>> dates, dcfs = generate_payment_schedule(1.0, 6.0, frequency=2)
    >>> len(dates)
    10
    >>> dates[0]
    1.5
    """
    if swap_end <= swap_start:
        msg = (
            f"swap_end must be greater than swap_start. "
            f"Got swap_start={swap_start}, swap_end={swap_end}."
        )
        raise ValueError(msg)

    if period_unit is not None or period_multiplier is not None:
        if period_unit is None or period_multiplier is None:
            msg = (
                "period_unit and period_multiplier must be provided together "
                "for custom schedules."
            )
            raise ValueError(msg)

        unit = period_unit.strip().upper()
        normalized = normalize_period_unit(unit)
        if normalized is None:
            msg = f"Unsupported period_unit '{period_unit}'."
            raise ValueError(msg)

        multiplier = float(period_multiplier)
        if multiplier <= 0:
            msg = f"period_multiplier must be positive. Got {period_multiplier}."
            raise ValueError(msg)

        year_factor = {
            "YEAR": 1.0,
            "MNTH": 1.0 / 12.0,
            "WEEK": 7.0 / 365.25,
            "DAIL": 1.0 / 365.25,
        }[normalized]
        period = year_factor * multiplier

        payment_dates: list[float] = []
        day_count_fractions: list[float] = []
        current = swap_start
        tol = 1e-12

        while current + period < swap_end - tol:
            nxt = current + period
            payment_dates.append(float(nxt))
            day_count_fractions.append(float(nxt - current))
            current = nxt

        payment_dates.append(float(swap_end))
        day_count_fractions.append(float(swap_end - current))
        return payment_dates, day_count_fractions

    if frequency not in (1, 2, 4, 12):
        msg = f"frequency must be 1, 2, 4, or 12. Got {frequency}."
        raise ValueError(msg)

    period = 1.0 / frequency
    n_periods = round((swap_end - swap_start) * frequency)

    payment_dates = [swap_start + (i + 1) * period for i in range(n_periods)]
    day_count_fractions = [period] * n_periods

    return payment_dates, day_count_fractions
