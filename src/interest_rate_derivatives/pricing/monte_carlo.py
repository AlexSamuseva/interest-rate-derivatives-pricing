"""
Monte Carlo Validation of Hull-White Swaption Prices.

This module provides Monte Carlo simulation of swaption prices under
the Hull-White one-factor model. Its primary purpose is to validate
the analytical Jamshidian prices computed by SwaptionPricer.

Simulation Approach
-------------------
The short rate is simulated from t=0 to the swaption expiry T_0 using
the Euler-Maruyama discretisation of the Hull-White SDE:

    r(t + dt) = r(t) + [theta(t) - a * r(t)] * dt + sigma * sqrt(dt) * Z

where Z ~ N(0, 1) and theta(t) is taken from HullWhiteModel.theta(t).

At each simulated path, the swaption payoff is evaluated analytically
using the Hull-White ZCB formula P(T_0, T_i; r(T_0)). This avoids
simulating beyond T_0 and is exact conditional on r(T_0).

The Monte Carlo estimator for a payer swaption is:

    V_MC = P(0, T_0) * (1/N) * sum_j max(1 - CB(T_0, r_j), 0)

where CB(T_0, r_j) = sum_i c_i * P(T_0, T_i; r_j).

Variance Reduction
------------------
Antithetic variates are used by default. For each set of random draws Z,
a mirrored set -Z is also simulated. This halves the variance of the
estimator at no additional model evaluation cost.

Validation
----------
The Monte Carlo price should agree with the Jamshidian analytical price
to within 2 standard errors. Disagreement indicates either a bug in the
implementation or insufficient paths.

References
----------
Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    Springer. Chapter 3.
Hull, J., & White, A. (1990). Pricing interest-rate derivative securities.
    The Review of Financial Studies, 3(4), 573-592.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from interest_rate_derivatives.models.hull_white import HullWhiteModel


class MonteCarloPricer:
    """
    Monte Carlo swaption pricer for validating Hull-White analytical prices.

    Simulates short rate paths to the swaption expiry T_0 using the
    Euler-Maruyama scheme and evaluates the swaption payoff using the
    analytical ZCB formula conditional on r(T_0).

    Parameters
    ----------
    model : HullWhiteModel
        Calibrated Hull-White model instance. Must be the same model
        used by SwaptionPricer to ensure a fair comparison.
    n_paths : int
        Number of Monte Carlo paths. Higher values reduce standard error
        but increase computation time. Default 50_000.
    n_steps : int
        Number of Euler time steps from 0 to T_0. More steps improve
        discretisation accuracy. Default 100.
    seed : int or None
        Random seed for reproducibility. Set to None for random results.
        Default 42.
    antithetic : bool
        If True, use antithetic variates to reduce variance. This
        effectively doubles the number of paths. Default True.

    Examples
    --------
    >>> from interest_rate_derivatives.utils.curves import FlatCurve
    >>> from interest_rate_derivatives.models.hull_white import HullWhiteModel
    >>> from interest_rate_derivatives.utils.curves import generate_payment_schedule
    >>> curve = FlatCurve(0.05)
    >>> model = HullWhiteModel(a=0.1, sigma=0.01, discount_factor=curve)
    >>> mc = MonteCarloPricer(model, n_paths=10_000, seed=42)
    >>> dates, dcfs = generate_payment_schedule(1.0, 6.0, frequency=2)
    >>> result = mc.price(1.0, dates, dcfs, 0.05, r0=0.05)
    >>> result["price"] > 0
    True
    """

    def __init__(
        self,
        model: HullWhiteModel,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: int | None = 42,
        antithetic: bool = True,
    ) -> None:
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.antithetic = antithetic

    # ------------------------------------------------------------------
    # Short rate simulation
    # ------------------------------------------------------------------

    def _simulate_paths(
        self,
        r0: float,
        option_expiry: float,
    ) -> NDArray:
        """
        Simulate short rate realisations r(T_0) using the Euler scheme.

        Discretises the Hull-White SDE:

            r(t+dt) = r(t) + [theta(t) - a*r(t)]*dt + sigma*sqrt(dt)*Z

        where theta(t) is obtained from model.theta(t), and Z ~ N(0,1).

        If antithetic=True, each set of random draws Z is paired with
        -Z to reduce variance. The effective number of paths is n_paths
        regardless of antithetic setting.

        Parameters
        ----------
        r0 : float
            Initial short rate r(0).
        option_expiry : float
            Swaption expiry T_0 in years. Simulation runs from 0 to T_0.

        Returns
        -------
        ndarray of shape (n_paths,)
            Simulated short rate values r(T_0), one per path.
        """
        rng = np.random.default_rng(self.seed)
        dt = option_expiry / self.n_steps
        time_grid = np.linspace(0.0, option_expiry, self.n_steps + 1)

        # Number of base paths — antithetic doubles them
        n_base = self.n_paths // 2 if self.antithetic else self.n_paths

        # Simulate base paths
        r = np.full(n_base, float(r0))
        for t in time_grid[:-1]:
            z = rng.standard_normal(n_base)
            drift = (self.model.theta(t) - self.model.a * r) * dt
            diffusion = self.model.sigma * np.sqrt(dt) * z
            r = r + drift + diffusion

        if not self.antithetic:
            return r

        # Simulate antithetic paths using negated random draws
        rng_anti = np.random.default_rng(self.seed)
        r_anti = np.full(n_base, float(r0))
        for t in time_grid[:-1]:
            z = -rng_anti.standard_normal(n_base)
            drift = (self.model.theta(t) - self.model.a * r_anti) * dt
            diffusion = self.model.sigma * np.sqrt(dt) * z
            r_anti = r_anti + drift + diffusion

        return np.concatenate([r, r_anti])

    # ------------------------------------------------------------------
    # Swaption pricing
    # ------------------------------------------------------------------

    def price(
        self,
        option_expiry: float,
        payment_dates: list[float],
        day_count_fractions: list[float],
        strike_rate: float,
        r0: float,
        notional: float = 1.0,
        is_payer: bool = True,
    ) -> dict[str, object]:
        """
        Estimate the swaption price by Monte Carlo simulation.

        For each simulated short rate r_j at T_0, the swaption payoff
        is computed using the analytical Hull-White ZCB formula:

            payoff_j = max(1 - CB(T_0, r_j), 0)   [payer]
            payoff_j = max(CB(T_0, r_j) - 1, 0)   [receiver]

        where CB(T_0, r_j) = sum_i c_i * P(T_0, T_i; r_j).

        The price is:

            V_MC = P(0, T_0) * notional * mean(payoff_j)

        Parameters
        ----------
        option_expiry : float
            Swaption expiry T_0 in years.
        payment_dates : list of float
            Fixed leg payment dates in years.
        day_count_fractions : list of float
            Day count fractions for each period.
        strike_rate : float
            Fixed rate K of the underlying swap.
        r0 : float
            Initial short rate r(0). Typically set to the current
            instantaneous forward rate f(0,0).
        notional : float
            Notional principal. Default 1.0.
        is_payer : bool
            True for payer swaption, False for receiver.

        Returns
        -------
        dict containing:
            price : float
                Monte Carlo estimate of the swaption price.
            std_error : float
                Standard error of the estimate.
            confidence_interval : tuple of float
                95% confidence interval (lower, upper).
            n_paths : int
                Actual number of paths simulated.
            analytical_check : str
                Reminder to compare with SwaptionPricer result.
        """
        # Step 1 — simulate r(T_0) realisations
        r_paths = self._simulate_paths(r0, option_expiry)
        n_actual = len(r_paths)

        # Step 2 — coupon weights
        coupons = [strike_rate * delta for delta in day_count_fractions]
        coupons[-1] += 1.0

        # Step 3 — compute payoff for each path
        payoffs = np.zeros(n_actual)
        for j, r_j in enumerate(r_paths):
            # ZCB prices at T_0 conditional on r_j
            cb = sum(
                c_i * float(self.model.zcb_price(option_expiry, T_i, r_j))
                for c_i, T_i in zip(coupons, payment_dates, strict=True)
            )
            if is_payer:
                payoffs[j] = max(1.0 - cb, 0.0)
            else:
                payoffs[j] = max(cb - 1.0, 0.0)

        # Step 4 — discount back to today
        p0_t0 = float(self.model.discount_factor(option_expiry))
        discounted = p0_t0 * payoffs * notional

        # Step 5 — compute statistics
        mc_price = float(np.mean(discounted))
        std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_actual))
        z95 = 1.96
        ci = (mc_price - z95 * std_error, mc_price + z95 * std_error)

        return {
            "price": mc_price,
            "std_error": std_error,
            "confidence_interval": ci,
            "n_paths": n_actual,
            "analytical_check": "Compare with SwaptionPricer.price() result",
        }
