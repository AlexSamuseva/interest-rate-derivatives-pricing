"""
Hull-White One-Factor Short Rate Model.

This module implements the Hull-White (1990) one-factor short rate model,
which is the foundation of the swaption pricing engine in this project.

The Model
---------
Under the risk-neutral measure Q, the short rate r(t) follows:

    dr(t) = [theta(t) - a * r(t)] dt + sigma * dW(t)

where:
    a       : mean reversion speed (positive constant)
    sigma   : short rate volatility (positive constant)
    theta(t): time-dependent drift, calibrated to match the initial
              market discount curve P(0, T) exactly
    W(t)    : standard Brownian motion under Q

The time-dependent drift theta(t) is given by:

    theta(t) = df(0,t)/dt + a * f(0,t) + sigma^2/(2a) * (1 - exp(-2at))

where f(0,t) is the market instantaneous forward rate.

Affine Term Structure
---------------------
The model belongs to the affine term structure class. Zero-coupon bond
prices have the closed-form expression:

    P(t, T) = A(t, T) * exp(-B(t, T) * r(t))

where:
    B(t, T) = (1 - exp(-a*(T-t))) / a

    ln A(t, T) = ln(P(0,T)/P(0,t)) + B(t,T)*f(0,t)
                 - sigma^2/(4a) * B(t,T)^2 * (1 - exp(-2at))

This file contains only the core model mathematics. Swaption pricing,
Monte Carlo simulation, and calibration are implemented in the pricing/
folder and use this class as their foundation.

References
----------
Hull, J., & White, A. (1990). Pricing interest-rate derivative securities.
    The Review of Financial Studies, 3(4), 573-592.
Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and
    Practice (2nd ed.). Springer Finance. Chapter 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class HullWhiteModel:
    """
    Hull-White one-factor short rate model.

    Provides analytical pricing of zero-coupon bonds and European options
    on zero-coupon bonds. These are the building blocks used by the
    swaption pricer via Jamshidian's decomposition.

    The model takes a discount curve object as input — either a
    DiscountCurve built from real FRED market data or a FlatCurve
    for testing purposes. This design means the model is automatically
    consistent with today's market curve regardless of which curve
    is provided.

    Parameters
    ----------
    a : float
        Mean reversion speed. Must be strictly positive.
        Controls how quickly the short rate reverts to its long-run level.
        Typical calibrated values: 0.01 to 0.30.
    sigma : float
        Short rate volatility. Must be strictly positive.
        Controls the overall level of interest rate uncertainty.
        Typical calibrated values: 0.005 to 0.020.
    discount_factor : Callable
        A callable that takes a maturity T (float or ndarray) and returns
        the discount factor P(0, T). This should be either a DiscountCurve
        or FlatCurve instance from utils/curves.py.

    Examples
    --------
    >>> from interest_rate_derivatives.utils.curves import FlatCurve
    >>> curve = FlatCurve(0.05)
    >>> model = HullWhiteModel(a=0.1, sigma=0.01, discount_factor=curve)
    >>> model.B(0.0, 1.0)
    0.9516...
    >>> model.zcb_price(0.0, 1.0, 0.05)
    0.9512...
    """

    def __init__(
        self,
        a: float,
        sigma: float,
        discount_factor: Callable,
    ) -> None:
        if a <= 0:
            msg = f"Mean reversion speed 'a' must be positive. Got {a}."
            raise ValueError(msg)
        if sigma <= 0:
            msg = f"Volatility 'sigma' must be positive. Got {sigma}."
            raise ValueError(msg)

        self.a = float(a)
        self.sigma = float(sigma)
        self.discount_factor = discount_factor

    # ------------------------------------------------------------------
    # Core building blocks — B(t,T) and ln A(t,T)
    # ------------------------------------------------------------------

    def B(self, t: float | NDArray, T: float | NDArray) -> float | NDArray:
        """
        Compute B(t, T) — the short rate sensitivity of the ZCB price.

        B(t, T) = (1 - exp(-a * (T - t))) / a

        This function appears in the affine ZCB formula:

            P(t, T) = A(t, T) * exp(-B(t, T) * r(t))

        It represents the sensitivity of the log ZCB price to the current
        short rate r(t) — analogous to modified duration in classical
        fixed income.

        Key properties:
        - B(t, T) > 0 always — ZCB prices fall when rates rise
        - B(t, T) -> T - t as a -> 0 (recovers standard duration)
        - B(t, T) <= 1/a (mean reversion caps the effective duration)

        Parameters
        ----------
        t : float or array-like
            Current time in years.
        T : float or array-like
            Bond maturity in years. Must be greater than t.

        Returns
        -------
        float or ndarray
            B(t, T) values.

        Examples
        --------
        >>> model.B(0.0, 5.0)
        3.9347...
        """
        tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def ln_A(
        self,
        t: float | NDArray,
        T: float | NDArray,
    ) -> float | NDArray:
        """
        Compute ln A(t, T) — the log of the fitting and convexity correction.

        ln A(t, T) = ln(P(0,T) / P(0,t))
                     + B(t,T) * f(0,t)
                     - sigma^2 / (4a) * B(t,T)^2 * (1 - exp(-2at))

        This term ensures the model prices today's discount curve exactly.
        Without it, the model would produce its own arbitrary yield curve
        inconsistent with market prices.

        The three components are:
        1. ln(P(0,T)/P(0,t))      : ratio of market discount factors
        2. B(t,T) * f(0,t)        : forward rate adjustment
        3. sigma^2/(4a)*B^2*(...) : convexity correction arising from
                                    Jensen's inequality

        Parameters
        ----------
        t : float or array-like
            Current time in years.
        T : float or array-like
            Bond maturity in years.

        Returns
        -------
        float or ndarray
            ln A(t, T) values.
        """
        t = np.asarray(t, dtype=float)
        T = np.asarray(T, dtype=float)

        P0t = self.discount_factor(t)
        P0T = self.discount_factor(T)
        f0t = self._forward_rate_from_curve(t)
        Bval = self.B(t, T)

        # Convexity correction term
        convexity = (
            (self.sigma**2 / (4.0 * self.a))
            * Bval**2
            * (1.0 - np.exp(-2.0 * self.a * t))
        )

        return np.log(P0T / P0t) + Bval * f0t - convexity

    # ------------------------------------------------------------------
    # Zero-coupon bond pricing
    # ------------------------------------------------------------------

    def zcb_price(
        self,
        t: float | NDArray,
        T: float | NDArray,
        r_t: float | NDArray,
    ) -> float | NDArray:
        """
        Price a zero-coupon bond P(t, T) given the short rate r(t).

        Uses the affine term structure formula:

            P(t, T) = A(t, T) * exp(-B(t, T) * r(t))
                    = exp(ln A(t, T) - B(t, T) * r(t))

        This is an exact analytical result under the Hull-White model.
        The bond price depends on the current short rate r(t) — this is
        the single state variable that drives the entire yield curve
        in the one-factor model.

        Parameters
        ----------
        t : float or array-like
            Current time in years.
        T : float or array-like
            Bond maturity in years. Must be greater than t.
        r_t : float or array-like
            Short rate at time t.

        Returns
        -------
        float or ndarray
            Zero-coupon bond price P(t, T).

        Examples
        --------
        >>> model.zcb_price(0.0, 5.0, 0.05)
        0.7788...
        """
        return np.exp(self.ln_A(t, T) - self.B(t, T) * np.asarray(r_t, dtype=float))

    # ------------------------------------------------------------------
    # European option on a zero-coupon bond
    # ------------------------------------------------------------------

    def zcb_option_price(
        self,
        option_expiry: float,
        bond_maturity: float,
        strike: float,
        is_call: bool = True,
    ) -> float:
        """
        Price a European option on a zero-coupon bond.

        This is the exact analytical formula derived by Jamshidian (1989)
        for the Hull-White model. It is structurally identical to Black's
        formula for bond options, with sigma_P playing the role of the
        implied bond volatility.

        For a call option:
            ZBCall = P(0, T_mat) * N(h) - K * P(0, T_exp) * N(h - sigma_P)

        For a put option:
            ZBPut = K * P(0, T_exp) * N(-h + sigma_P) - P(0, T_mat) * N(-h)

        where:
            sigma_P = sigma * B(T_exp, T_mat)
                      * sqrt((1 - exp(-2a * T_exp)) / (2a))

            h = (1/sigma_P) * ln(P(0,T_mat) / (P(0,T_exp) * K))
                + sigma_P / 2

        sigma_P is the standard deviation of the log ZCB price at expiry.
        It combines the short rate volatility sigma, the bond duration
        B(T_exp, T_mat), and the accumulated variance up to expiry.

        This formula is the building block of Jamshidian's decomposition
        in pricing/swaption.py — a swaption is priced as a sum of these
        ZCB options.

        Parameters
        ----------
        option_expiry : float
            Option expiry date T_exp in years from today.
        bond_maturity : float
            Maturity of the underlying ZCB T_mat in years from today.
            Must be greater than option_expiry.
        strike : float
            Option strike price K.
        is_call : bool
            True for a call option, False for a put option.
            In Jamshidian's decomposition for payer swaptions, put
            options are used.

        Returns
        -------
        float
            Option price.

        Examples
        --------
        >>> model.zcb_option_price(1.0, 5.0, 0.80, is_call=False)
        0.0023...
        """
        T_exp = option_expiry
        T_mat = bond_maturity

        P0_Texp = self.discount_factor(T_exp)
        P0_Tmat = self.discount_factor(T_mat)

        # Bond volatility — standard deviation of log ZCB price at expiry
        sigma_P = (
            self.sigma
            * self.B(T_exp, T_mat)
            * np.sqrt((1.0 - np.exp(-2.0 * self.a * T_exp)) / (2.0 * self.a))
        )

        # Degenerate case — no uncertainty, return intrinsic value
        if sigma_P < 1e-12:
            if is_call:
                return float(max(P0_Tmat - strike * P0_Texp, 0.0))
            return float(max(strike * P0_Texp - P0_Tmat, 0.0))

        h = np.log(P0_Tmat / (P0_Texp * strike)) / sigma_P + sigma_P / 2.0

        if is_call:
            return float(
                P0_Tmat * norm.cdf(h) - strike * P0_Texp * norm.cdf(h - sigma_P)
            )
        return float(strike * P0_Texp * norm.cdf(-h + sigma_P) - P0_Tmat * norm.cdf(-h))

    # ------------------------------------------------------------------
    # Short rate distribution and drift
    # ------------------------------------------------------------------

    def short_rate_std(self, s: float, t: float) -> float:
        """
        Compute the conditional standard deviation of r(t) given r(s).

        Under the Hull-White model, r(t) | r(s) is normally distributed
        with standard deviation:

            nu(s, t) = sigma * sqrt((1 - exp(-2a*(t-s))) / (2a))

        This quantity is used in Monte Carlo simulation to generate
        short rate realisations at future times.

        Parameters
        ----------
        s : float
            Start time in years.
        t : float
            End time in years. Must be greater than s.

        Returns
        -------
        float
            Conditional standard deviation of r(t) given r(s).
        """
        tau = t - s
        variance = (self.sigma**2 / (2.0 * self.a)) * (
            1.0 - np.exp(-2.0 * self.a * tau)
        )
        return float(np.sqrt(variance))

    def theta(self, t: float) -> float:
        """
        Compute the time-dependent drift theta(t).

        theta(t) is a fundamental component of the Hull-White model — it
        is the time-dependent drift that ensures the model reproduces the
        initial market discount curve P(0, T) exactly for all maturities.

        Without theta(t), the model reduces to Vasicek (1977) which does
        not fit the initial term structure. theta(t) is what makes
        Hull-White an arbitrage-free model consistent with market prices.

        The formula is:

            theta(t) = df(0,t)/dt + a*f(0,t) + sigma^2/(2a)*(1-exp(-2at))

        where:
            df(0,t)/dt                : slope of instantaneous forward curve
            a*f(0,t)                  : mean reversion adjustment
            sigma^2/(2a)*(1-exp(...)) : convexity correction

        Computed numerically using finite differences on f(0,t).

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            theta(t).
        """
        h = 1e-5
        t_val = max(float(t), h)

        f0t = self._forward_rate_from_curve(t_val)
        f0t_h = self._forward_rate_from_curve(t_val + h)

        df_dt = (f0t_h - f0t) / h
        vol_term = (self.sigma**2 / (2.0 * self.a)) * (
            1.0 - np.exp(-2.0 * self.a * t_val)
        )

        return float(df_dt + self.a * f0t + vol_term)

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _forward_rate_from_curve(
        self,
        t: float | NDArray,
    ) -> float | NDArray:
        """
        Compute the instantaneous forward rate f(0, t) from the discount curve.

        Uses the analytical instantaneous_forward_rate method if available
        on the curve object — which both DiscountCurve and FlatCurve
        provide. Falls back to finite differences for any other callable
        discount curve.

        This is a private method used internally by ln_A and theta.

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Instantaneous forward rate f(0, t).
        """
        if hasattr(self.discount_factor, "instantaneous_forward_rate"):
            return self.discount_factor.instantaneous_forward_rate(t)

        # Fallback: finite differences
        h = 1e-5
        t_arr = np.asarray(t, dtype=float)
        t_up = np.maximum(t_arr + h, h)
        t_dn = np.maximum(t_arr - h, h)
        P_up = self.discount_factor(t_up)
        P_dn = self.discount_factor(t_dn)
        return -(np.log(P_up) - np.log(P_dn)) / (t_up - t_dn)
