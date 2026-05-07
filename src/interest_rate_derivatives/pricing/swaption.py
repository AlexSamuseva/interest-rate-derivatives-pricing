"""
Swaption Pricing via Jamshidian's Decomposition.

This module implements European swaption pricing under the Hull-White
one-factor model using Jamshidian's (1989) decomposition theorem.

Swaption Payoff
---------------
A payer swaption gives the right to enter a swap paying fixed rate K
at expiry T_0. The payoff at T_0 is:

    V_payer(T_0) = N * A(T_0) * max(S(T_0) - K, 0)

where S(T_0) is the par swap rate at T_0 and A(T_0) is the annuity.

Equivalently, in terms of a coupon bond:

    V_payer(T_0) = N * max(1 - CB(T_0), 0)

where the coupon bond is:

    CB(t) = sum_i c_i * P(t, T_i)
    c_i = K * delta_i  for i < n
    c_n = 1 + K * delta_n

So a payer swaption is a put option on a coupon bond with strike 1.

Jamshidian's Decomposition
--------------------------
In a one-factor model, all ZCB prices at T_0 are deterministic functions
of the single state variable r(T_0). Since P(T_0, T_i) is monotonically
decreasing in r(T_0) for all i, the coupon bond CB(T_0) is also
monotonically decreasing in r(T_0).

Therefore there exists a unique critical rate r* such that:

    CB(T_0, r*) = 1

The swaption decomposes exactly into a portfolio of ZCB options:

    V_payer = sum_i c_i * ZBPut(0; T_0, T_i, K_i)
    V_receiver = sum_i c_i * ZBCall(0; T_0, T_i, K_i)

where K_i = P(T_0, T_i; r*) are the individual ZCB strikes.

This decomposition is exact — not an approximation — because all ZCB
puts are exercised simultaneously when r(T_0) > r*.

References
----------
Jamshidian, F. (1989). An exact bond option formula.
    The Journal of Finance, 44(1), 205-209.
Hull, J., & White, A. (1990). Pricing interest-rate derivative securities.
    The Review of Financial Studies, 3(4), 573-592.
Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and
    Practice (2nd ed.). Springer Finance. Chapter 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.optimize import brentq

if TYPE_CHECKING:
    from interest_rate_derivatives.models.hull_white import HullWhiteModel


class SwaptionPricer:
    """
    Prices European swaptions under the Hull-White model using
    Jamshidian's decomposition.

    Takes a calibrated HullWhiteModel instance and prices payer or
    receiver swaptions for any strike, expiry, and swap schedule.

    Parameters
    ----------
    model : HullWhiteModel
        A calibrated Hull-White model instance containing the discount
        curve and model parameters a and sigma.

    Examples
    --------
    >>> from interest_rate_derivatives.utils.curves import FlatCurve
    >>> from interest_rate_derivatives.models.hull_white import HullWhiteModel
    >>> from interest_rate_derivatives.utils.curves import generate_payment_schedule
    >>> curve = FlatCurve(0.05)
    >>> model = HullWhiteModel(a=0.1, sigma=0.01, discount_factor=curve)
    >>> pricer = SwaptionPricer(model)
    >>> dates, dcfs = generate_payment_schedule(1.0, 6.0, frequency=2)
    >>> result = pricer.price(1.0, dates, dcfs, strike_rate=0.05)
    >>> result["price"] > 0
    True
    """

    def __init__(self, model: HullWhiteModel) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Swap rate and annuity factor
    # ------------------------------------------------------------------

    def par_swap_rate(
        self,
        swap_start: float,
        payment_dates: list[float],
        day_count_fractions: list[float],
    ) -> float:
        """
        Compute the par swap rate S(0) using the market discount curve.

        The par swap rate is the fixed rate K that makes the swap have
        zero NPV at inception:

            S(0) = (P(0, T_0) - P(0, T_n)) / A(0)

        where A(0) = sum_i delta_i * P(0, T_i) is the annuity factor.

        This is computed directly from the market discount curve stored
        in the Hull-White model — it does not depend on a or sigma.

        Parameters
        ----------
        swap_start : float
            Swap effective date T_0 in years from today.
        payment_dates : list of float
            Fixed leg payment dates [T_1, ..., T_n] in years.
        day_count_fractions : list of float
            Day count fractions [delta_1, ..., delta_n].

        Returns
        -------
        float
            Par swap rate S(0).
        """
        P_start = self.model.discount_factor(swap_start)
        P_end = self.model.discount_factor(payment_dates[-1])
        annuity = self.annuity_factor(payment_dates, day_count_fractions)
        return float((P_start - P_end) / annuity)

    def annuity_factor(
        self,
        payment_dates: list[float],
        day_count_fractions: list[float],
    ) -> float:
        """
        Compute the present value annuity factor A(0).

        The annuity factor is:

            A(0) = sum_{i=1}^{n} delta_i * P(0, T_i)

        It represents the present value of the fixed payment stream
        per unit of notional per unit of rate. It converts a rate
        difference (S(T_0) - K) into a dollar amount in the swaption
        payoff.

        Parameters
        ----------
        payment_dates : list of float
            Payment dates [T_1, ..., T_n] in years.
        day_count_fractions : list of float
            Day count fractions [delta_1, ..., delta_n].

        Returns
        -------
        float
            Annuity factor A(0).
        """
        return float(
            sum(
                delta * self.model.discount_factor(T)
                for T, delta in zip(payment_dates, day_count_fractions, strict=True)
            )
        )

    # ------------------------------------------------------------------
    # Jamshidian's decomposition — critical rate
    # ------------------------------------------------------------------

    def _coupon_bond_price(
        self,
        r_val: float,
        option_expiry: float,
        payment_dates: list[float],
        day_count_fractions: list[float],
        strike_rate: float,
    ) -> float:
        """
        Price the fixed-rate coupon bond at expiry T_0 given short rate r.

        The coupon bond is:

            CB(T_0, r) = sum_i c_i * P(T_0, T_i; r)

        where:
            c_i = strike_rate * delta_i   for i = 1, ..., n-1
            c_n = 1 + strike_rate * delta_n  (includes principal)

        This is a private method used only by find_critical_rate.
        It uses model.zcb_price to evaluate each ZCB price at the
        given short rate r.

        Parameters
        ----------
        r_val : float
            Short rate value at option expiry T_0.
        option_expiry : float
            Swaption expiry T_0 in years.
        payment_dates : list of float
            Coupon payment dates.
        day_count_fractions : list of float
            Day count fractions.
        strike_rate : float
            Fixed coupon rate K.

        Returns
        -------
        float
            Coupon bond price CB(T_0, r_val).
        """
        n = len(payment_dates)
        total = 0.0
        for i, (T_i, delta_i) in enumerate(
            zip(payment_dates, day_count_fractions, strict=True)
        ):
            coupon = strike_rate * delta_i
            if i == n - 1:
                coupon += 1.0
            zcb = self.model.zcb_price(option_expiry, T_i, r_val)
            total += coupon * zcb
        return total

    def find_critical_rate(
        self,
        option_expiry: float,
        payment_dates: list[float],
        day_count_fractions: list[float],
        strike_rate: float,
        bracket: tuple[float, float] = (-0.30, 0.80),
    ) -> float:
        """
        Find the critical rate r* such that CB(T_0, r*) = 1.

        The critical rate is the unique short rate at the option expiry
        date for which the coupon bond price equals the strike price of
        1. It exists and is unique because CB(T_0, r) is continuous and
        strictly monotonically decreasing in r.

        The event that the swaption is in the money is identical to the
        event that r(T_0) > r*:

            {CB(T_0) < 1} = {r(T_0) > r*}

        This is the key insight of Jamshidian's decomposition — it
        reduces a coupon bond option to a portfolio of ZCB options
        all exercised at the same critical rate.

        Uses Brent's root-finding method, which is guaranteed to
        converge given the monotonicity of CB(T_0, r).

        Parameters
        ----------
        option_expiry : float
            Swaption expiry T_0 in years.
        payment_dates : list of float
            Coupon payment dates.
        day_count_fractions : list of float
            Day count fractions.
        strike_rate : float
            Fixed coupon rate K.
        bracket : tuple of float
            Search bracket [r_low, r_high] for Brent's method.
            Default (-0.30, 0.80) covers all realistic rate environments.

        Returns
        -------
        float
            Critical rate r*.

        Raises
        ------
        ValueError
            If no root is found within the bracket.
        """

        def objective(r: float) -> float:
            return (
                self._coupon_bond_price(
                    r,
                    option_expiry,
                    payment_dates,
                    day_count_fractions,
                    strike_rate,
                )
                - 1.0
            )

        try:
            return float(brentq(objective, bracket[0], bracket[1], xtol=1e-10))
        except ValueError:
            lo = objective(bracket[0])
            hi = objective(bracket[1])
            msg = (
                f"Could not find critical rate in bracket {bracket}. "
                f"CB({bracket[0]:.4f}) - 1 = {lo:.6f}, "
                f"CB({bracket[1]:.4f}) - 1 = {hi:.6f}. "
                "Check model parameters or widen the bracket."
            )
            raise ValueError(msg) from None

    def jamshidian_strikes(
        self,
        r_star: float,
        option_expiry: float,
        payment_dates: list[float],
    ) -> list[float]:
        """
        Compute the ZCB option strikes K_i from the critical rate r*.

        Each strike is the ZCB price that would prevail at expiry T_0
        if the short rate equals the critical rate r*:

            K_i = P(T_0, T_i; r*)

        These strikes ensure that each component ZCB put option is
        exercised exactly when the coupon bond put is exercised —
        i.e. when r(T_0) > r*.

        Parameters
        ----------
        r_star : float
            Critical rate found by find_critical_rate.
        option_expiry : float
            Swaption expiry T_0 in years.
        payment_dates : list of float
            Coupon payment dates.

        Returns
        -------
        list of float
            ZCB strikes [K_1, ..., K_n].
        """
        return [
            float(self.model.zcb_price(option_expiry, T_i, r_star))
            for T_i in payment_dates
        ]

    # ------------------------------------------------------------------
    # Main pricing method
    # ------------------------------------------------------------------

    def price(
        self,
        option_expiry: float,
        payment_dates: list[float],
        day_count_fractions: list[float],
        strike_rate: float,
        notional: float = 1.0,
        is_payer: bool = True,
    ) -> dict[str, object]:
        """
        Price a European swaption using Jamshidian's decomposition.

        This is the main public method of SwaptionPricer. It implements
        the full Jamshidian algorithm:

        1. Compute coupon weights c_i
        2. Find r* by solving CB(T_0, r*) = 1
        3. Compute ZCB strikes K_i = P(T_0, T_i; r*)
        4. Price each ZCB option using the HW analytical formula
        5. Sum: V = notional * sum_i c_i * ZBOpt(0; T_0, T_i, K_i)

        For a payer swaption, ZCB puts are used.
        For a receiver swaption, ZCB calls are used.

        Parameters
        ----------
        option_expiry : float
            Swaption expiry T_0 in years from today.
        payment_dates : list of float
            Fixed leg payment dates [T_1, ..., T_n] in years.
        day_count_fractions : list of float
            Day count fractions [delta_1, ..., delta_n].
        strike_rate : float
            Fixed rate of the underlying swap K.
            Use par_swap_rate() to get the ATM strike.
        notional : float
            Notional principal. Default 1.0.
        is_payer : bool
            True for a payer swaption (right to pay fixed).
            False for a receiver swaption (right to receive fixed).

        Returns
        -------
        dict containing:
            price : float
                Swaption price in currency units.
            par_swap_rate : float
                Current ATM swap rate.
            critical_rate : float
                r* from Jamshidian's decomposition.
            zcb_strikes : list of float
                K_i values for each payment date.
            coupons : list of float
                Coupon weights c_i for each payment date.
            component_prices : list of float
                Individual ZCB option prices c_i * ZBOpt_i.
            option_type : str
                'Payer' or 'Receiver'.
            notional : float
                Notional principal.
            strike_rate : float
                Strike rate K.
            option_expiry : float
                Swaption expiry T_0.
            annuity : float
                Present value annuity factor A(0).
        """
        swap_start = option_expiry

        # Step 1 — par swap rate and annuity
        par_rate = self.par_swap_rate(swap_start, payment_dates, day_count_fractions)
        annuity = self.annuity_factor(payment_dates, day_count_fractions)

        # Step 2 — coupon weights
        coupons = [strike_rate * delta for delta in day_count_fractions]
        coupons[-1] += 1.0

        # Step 3 — find critical rate r*
        r_star = self.find_critical_rate(
            option_expiry, payment_dates, day_count_fractions, strike_rate
        )

        # Step 4 — ZCB strikes from r*
        strikes = self.jamshidian_strikes(r_star, option_expiry, payment_dates)

        # Step 5 — price each ZCB option and sum
        # payer swaption = put on coupon bond = puts on ZCBs
        # receiver swaption = call on coupon bond = calls on ZCBs
        use_call = not is_payer

        component_prices = []
        total = 0.0
        for c_i, T_i, K_i in zip(coupons, payment_dates, strikes, strict=True):
            zcb_opt = self.model.zcb_option_price(
                option_expiry=option_expiry,
                bond_maturity=T_i,
                strike=K_i,
                is_call=use_call,
            )
            component = c_i * zcb_opt
            component_prices.append(component)
            total += component

        return {
            "price": total * notional,
            "par_swap_rate": par_rate,
            "critical_rate": r_star,
            "zcb_strikes": strikes,
            "coupons": coupons,
            "component_prices": component_prices,
            "option_type": "Payer" if is_payer else "Receiver",
            "notional": notional,
            "strike_rate": strike_rate,
            "option_expiry": option_expiry,
            "annuity": annuity,
        }
