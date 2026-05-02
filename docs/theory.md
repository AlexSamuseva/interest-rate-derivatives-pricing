# Theory and Methodology

## 1. Introduction

This document provides the theoretical foundation for the pricing of interest
rate derivatives — specifically European swaptions — using the Hull-White
one-factor short rate model and Jamshidian's decomposition. The framework
presented here underpins all numerical implementations in this project.

The pricing pipeline consists of four conceptual layers:

1. **Market data** — the initial discount curve extracted from US Treasury
   yields via the FRED API
2. **The model** — the Hull-White one-factor short rate model, calibrated to
   market swaption prices
3. **The pricing engine** — Jamshidian's decomposition, which reduces a
   swaption to a portfolio of zero-coupon bond options
4. **Validation** — Monte Carlo simulation of the short rate dynamics to
   verify analytical prices

Each layer is described in full below.

---

## 2. Interest Rate Markets

### 2.1 The Discount Factor

The most fundamental quantity in fixed income is the **discount factor**
$P(0, T)$ — the time-0 price of one unit of currency delivered at time $T$.
It answers the question: what is one dollar received in $T$ years worth today?

Discount factors are directly observable from the prices of government bonds
and money market instruments. They satisfy:

$$P(0, 0) = 1, \quad P(0, T) \in (0, 1) \text{ for } T > 0$$

and are strictly decreasing in $T$ under positive interest rates.

### 2.2 Zero Rates

The **continuously-compounded zero rate** $z(T)$ is the constant rate that
equates a single compounding to the observed discount factor:

$$P(0, T) = e^{-z(T) \cdot T}$$

equivalently:

$$z(T) = -\frac{\ln P(0, T)}{T}$$

A plot of $z(T)$ against $T$ is the **zero curve** or **zero-coupon yield
curve**. Under normal market conditions it is upward sloping — longer
maturities carry higher yields to compensate investors for the additional
risk of locking up capital for longer.

### 2.3 The Instantaneous Forward Rate

The **instantaneous forward rate** $f(0, T)$ is the rate agreed today for
borrowing over an infinitesimally short interval $[T, T + dT]$:

$$f(0, T) = -\frac{\partial \ln P(0, T)}{\partial T}$$

Forward rates are not directly observable but are implied by the shape of the
discount curve. They play a central role in the Hull-White model because the
time-dependent drift $\theta(t)$ is expressed in terms of $f(0, t)$.

The relationship between discount factors and forward rates is:

$$P(0, T) = \exp\left(-\int_0^T f(0, u)\, du\right)$$

### 2.4 The Market Discount Curve

In practice the discount curve is bootstrapped from market instruments —
overnight index swap (OIS) rates, Treasury yields, or SOFR swap rates. In
this project we use **US Treasury constant maturity yields** sourced from
the FRED API (Federal Reserve Economic Data), which provides daily yields
at standardised maturities: 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, and 30Y.

These yields are interpolated using a cubic spline on the zero rates to
produce a smooth, continuous discount curve $P(0, T)$ for any maturity $T$.

---

## 3. Zero-Coupon Bonds

A **zero-coupon bond** (ZCB) with maturity $T$ pays exactly one unit of
currency at time $T$ and nothing before. Its time-0 price is by definition
the discount factor $P(0, T)$.

ZCBs are the fundamental building blocks of all fixed income pricing. Any
deterministic cash flow stream can be priced as a linear combination of ZCBs.
Crucially, under the Hull-White model, ZCB prices at a future time $t > 0$
retain an analytical closed form — this is what makes derivative pricing
tractable.

The **time-$t$ price** of a ZCB maturing at $T > t$ is a random variable
because it depends on the future state of the yield curve. Under the
Hull-White model it is given by the affine formula derived in Section 7.

---

## 4. Interest Rate Swaps

### 4.1 Definition

A **vanilla interest rate swap (IRS)** is a bilateral agreement to exchange
cash flows between two counterparties over a fixed schedule of dates
$T_0 < T_1 < \cdots < T_n$:

- The **fixed leg** pays the rate $K$ on a notional $N$ at each date
  $T_i$, with cash flow $N \cdot K \cdot \delta_i$
- The **floating leg** pays the prevailing reference rate (e.g. SOFR)
  reset at the start of each period $[T_{i-1}, T_i]$

where $\delta_i = T_i - T_{i-1}$ is the **day count fraction** for period
$i$, representing the length of the period in years under the relevant
day count convention.

### 4.2 Valuation

By no-arbitrage, the value of the floating leg at time $t \leq T_0$ equals:

$$V_{\text{float}}(t) = N \cdot [P(t, T_0) - P(t, T_n)]$$

This result follows from the fact that a floating rate bond resets to par
at each payment date, so its value equals par at $T_0$, discounted back
to $t$.

The value of the fixed leg is simply the present value of the fixed cash flows:

$$V_{\text{fixed}}(t) = N \cdot K \cdot \sum_{i=1}^n \delta_i \cdot P(t, T_i)$$

The **net present value of the swap** from the perspective of the fixed-rate
payer is:

$$\text{NPV}_{\text{payer}}(t) = V_{\text{float}}(t) - V_{\text{fixed}}(t)
= N \left[ P(t, T_0) - P(t, T_n) - K \sum_{i=1}^n \delta_i P(t, T_i) \right]$$

### 4.3 The Par Swap Rate and Annuity Factor

The **annuity factor** $A(t)$ is the present value of one basis point paid
on each payment date — the swap's sensitivity to a change in the fixed rate:

$$A(t) = \sum_{i=1}^n \delta_i \cdot P(t, T_i)$$

The **par swap rate** $S(t)$ is defined as the fixed rate $K$ that makes the
swap have zero NPV at time $t$:

$$S(t) = \frac{P(t, T_0) - P(t, T_n)}{A(t)}$$

The par swap rate is the market's best estimate of the average level of the
floating rate over the life of the swap. It is directly observable from
interbank swap markets.

---

## 5. Swaptions

### 5.1 Definition

A **European swaption** is an option to enter into a pre-specified interest
rate swap at a future expiry date $T_0$, at a fixed rate $K$ agreed today.

There are two types:

- A **payer swaption** gives the holder the right — but not the obligation —
  to *pay fixed* and *receive floating* at rate $K$ from $T_0$ to $T_n$.
  It is profitable when market rates at $T_0$ exceed $K$.
- A **receiver swaption** gives the right to *receive fixed* and *pay
  floating*. It is profitable when market rates fall below $K$.

Swaptions are among the most liquid interest rate derivatives and are the
primary instruments used to hedge interest rate volatility risk in fixed
income portfolios.

### 5.2 Payoff

The **payoff of a payer swaption** at expiry $T_0$ is:

$$V_{\text{payer}}(T_0) = N \cdot A(T_0) \cdot \max(S(T_0) - K,\ 0)$$

where $S(T_0)$ is the par swap rate at $T_0$ and $A(T_0)$ is the realised
annuity factor at $T_0$. Both $S(T_0)$ and $A(T_0)$ are random at time 0
because they depend on the future yield curve.

Similarly, the **payoff of a receiver swaption** is:

$$V_{\text{receiver}}(T_0) = N \cdot A(T_0) \cdot \max(K - S(T_0),\ 0)$$

### 5.3 Equivalence to a Coupon Bond Option

A key insight that enables analytical pricing is the equivalence between a
swaption and an option on a coupon bond. Entering the swap at $T_0$ as a
fixed-rate payer is equivalent to selling a fixed-rate coupon bond and
receiving a floating-rate bond (which is worth par at $T_0$). Therefore:

$$V_{\text{payer}}(T_0) = N \cdot \max\!\left(1 - CB(T_0),\ 0\right)$$

where the **coupon bond** $CB(t)$ is:

$$CB(t) = \sum_{i=1}^n c_i \cdot P(t, T_i)$$

with coupon weights:

$$c_i = K \cdot \delta_i \quad \text{for } i = 1, \ldots, n-1, \qquad
c_n = 1 + K \cdot \delta_n$$

The final weight $c_n$ includes the principal repayment of 1.

This means:

- A **payer swaption** = **put option on a coupon bond** with strike 1
- A **receiver swaption** = **call option on a coupon bond** with strike 1

This equivalence is exact and model-independent. It is the starting point
for Jamshidian's decomposition in Section 8.

---

## 6. Short Rate Models — Motivation

The swaption payoff involves $S(T_0)$ and $A(T_0)$ — quantities that depend
on the **entire yield curve at time $T_0$**, which is unknown today. To price
the swaption we need a model for the **stochastic evolution of interest rates**.

**Short rate models** specify dynamics for the instantaneous short rate
$r(t)$ — the rate at which money can be borrowed or lent over an
infinitesimally short interval $[t, t+dt]$. The short rate is not directly
observable but is the building block from which the entire yield curve is
derived.

Under the **risk-neutral measure** $\mathbb{Q}$, the time-0 price of any
interest rate derivative with payoff $X$ at time $T$ is:

$$V(0) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{-\int_0^T r(u)\, du} \cdot X\right]$$

The key requirement of any short rate model is that it **reproduces today's
market discount curve exactly** — otherwise the model misprices even the
simplest instruments such as government bonds. This is the primary motivation
for the Hull-White model.

---

## 7. The Hull-White One-Factor Model

### 7.1 The Stochastic Differential Equation

Hull and White (1990) proposed extending the Vasicek (1977) model by
introducing a **time-dependent drift** $\theta(t)$ that allows the model to
fit the initial term structure exactly. Under the risk-neutral measure
$\mathbb{Q}$, the short rate $r(t)$ follows:

$$dr(t) = [\theta(t) - a \cdot r(t)]\, dt + \sigma\, dW^{\mathbb{Q}}(t)$$

where:

- $a > 0$ is the **mean reversion speed**
- $\sigma > 0$ is the **short rate volatility**
- $\theta(t)$ is the **time-dependent drift**, calibrated to the market curve
- $W^{\mathbb{Q}}(t)$ is a standard Brownian motion under $\mathbb{Q}$

### 7.2 The Mean Reversion Parameter $a$

The term $-a \cdot r(t)$ in the drift creates a **restoring force** that
pulls the short rate back toward its long-run level. If $r(t)$ is high,
the drift is negative, pushing rates down. If $r(t)$ is low, the drift is
positive, pushing rates up.

- **Large $a$**: fast mean reversion — rate shocks are short-lived, the
  yield curve is relatively flat, and long rates are less volatile than
  short rates
- **Small $a$**: slow mean reversion — rate shocks persist for many years,
  and the yield curve exhibits large parallel shifts
- **Typical calibrated values**: $a \in [0.01, 0.30]$ in practice

The mean reversion parameter is not observable directly — it is calibrated
to market swaption prices.

### 7.3 The Volatility Parameter $\sigma$

$\sigma$ is the **instantaneous volatility of the short rate**. It controls
the overall level of interest rate uncertainty. A higher $\sigma$ produces
more volatile yield curves and therefore higher swaption prices.

In practice $\sigma$ is calibrated jointly with $a$ to match the market
prices of a basket of benchmark swaptions.

### 7.4 The Time-Dependent Drift $\theta(t)$

$\theta(t)$ is determined analytically from the initial market discount
curve by requiring that the model reproduces $P(0, T)$ for all $T$. The
result is:

$$\theta(t) = \frac{\partial f(0, t)}{\partial t} + a \cdot f(0, t)
+ \frac{\sigma^2}{2a}\left(1 - e^{-2at}\right)$$

where $f(0, t)$ is the market instantaneous forward rate. This formula
shows that $\theta(t)$ is fully determined by the market curve and the
model parameters — it is not a free parameter.

The first two terms ensure the model matches today's forward rates. The
third term is a **convexity adjustment** arising from the non-linearity
of the relationship between short rates and bond prices.

### 7.5 Distribution of the Short Rate

Because the SDE is linear in $r(t)$, the short rate has a **Gaussian
(normal) distribution** at any future time $t$, conditional on $r(s)$:

$$r(t)\,|\,r(s) \sim \mathcal{N}\!\left(\mu(s,t),\ \nu^2(s,t)\right)$$

$$\mu(s,t) = r(s)\,e^{-a(t-s)}
+ \int_s^t \theta(u)\,e^{-a(t-u)}\,du$$

$$\nu^2(s,t) = \frac{\sigma^2}{2a}\left(1 - e^{-2a(t-s)}\right)$$

The Gaussian structure is what makes the model analytically tractable.
It also implies that the short rate can become negative — a known
limitation that is discussed in Section 11.

### 7.6 Affine Term Structure — ZCB Pricing Formula

The Hull-White model belongs to the **affine term structure** class
(Duffie and Kan, 1996). This means that ZCB prices have the closed-form
**affine** expression:

$$P(t, T) = A(t, T) \cdot \exp\!\bigl(-B(t, T) \cdot r(t)\bigr)$$

where the two functions $B(t,T)$ and $A(t,T)$ are deterministic:

$$B(t, T) = \frac{1 - e^{-a(T-t)}}{a}$$

$$\ln A(t, T) = \ln\frac{P(0,T)}{P(0,t)} + B(t,T) \cdot f(0,t)
- \frac{\sigma^2}{4a}\,B(t,T)^2\,\left(1 - e^{-2at}\right)$$

**Interpretation of $B(t,T)$:**

$B(t,T)$ is the **sensitivity of the log ZCB price to the short rate**
— analogous to modified duration in classical fixed income. It satisfies:

$$\frac{\partial \ln P(t,T)}{\partial r(t)} = -B(t,T)$$

Key properties:

- $B(t,T) > 0$ always — ZCB prices fall when rates rise
- $B(t,T) \to T-t$ as $a \to 0$ — recovers standard duration in the
  limit of no mean reversion
- $B(t,T) \leq 1/a$ — fast mean reversion caps the duration at $1/a$,
  reflecting the fact that rate shocks do not persist indefinitely

**Interpretation of $A(t,T)$:**

$\ln A(t,T)$ is a **convexity and fitting correction**. It ensures the
model prices today's discount curve exactly by adjusting for the difference
between the model's implied forward rates and the market forward rates.
Without this term the model would misprice even simple ZCBs.

---

## 8. European Options on Zero-Coupon Bonds

Before pricing swaptions we need the price of a European option on a ZCB,
as this is the building block of Jamshidian's decomposition.

The time-0 price of a **European call option** on a ZCB maturing at
$T_{\text{mat}}$, with option expiry $T_{\text{exp}} < T_{\text{mat}}$
and strike $K$, is given by the exact Hull-White formula
(Jamshidian, 1989):

$$\text{ZBCall}(0;\ T_{\exp},\ T_{\text{mat}},\ K)
= P(0, T_{\text{mat}})\,N(h) - K\,P(0, T_{\exp})\,N(h - \sigma_P)$$

The **put price** follows from put-call parity:

$$\text{ZBPut}(0;\ T_{\exp},\ T_{\text{mat}},\ K)
= K\,P(0, T_{\exp})\,N(-h + \sigma_P) - P(0, T_{\text{mat}})\,N(-h)$$

where $N(\cdot)$ is the standard normal CDF and:

$$\sigma_P = \sigma \cdot B(T_{\exp}, T_{\text{mat}})
\cdot \sqrt{\frac{1 - e^{-2a\,T_{\exp}}}{2a}}$$

$$h = \frac{1}{\sigma_P}\ln\frac{P(0, T_{\text{mat}})}
{P(0, T_{\exp}) \cdot K} + \frac{\sigma_P}{2}$$

**Interpretation of $\sigma_P$:**

$\sigma_P$ is the **standard deviation of the log ZCB price** at the
option expiry. It combines three components:

- $\sigma$: the short rate volatility
- $B(T_{\exp}, T_{\text{mat}})$: the duration of the ZCB — how much the
  bond price moves per unit change in the short rate
- The square root factor: how much variance accumulates from $0$ to
  $T_{\exp}$ under mean reversion

This formula is structurally identical to Black's formula for bond options,
with $\sigma_P$ playing the role of implied bond volatility.

---

## 9. Jamshidian's Decomposition

### 9.1 The Challenge

A coupon bond option does not have a simple closed form in general because
the cash flows at different maturities are driven by different parts of the
yield curve. In a multi-factor model, rates at different tenors can move
independently, making the joint distribution of all cash flows intractable.

### 9.2 The One-Factor Insight

Define the **ZCB strikes**:

$$K_i = P(T_0, T_i;\ r^*)
= A(T_0, T_i)\,\exp\!\bigl(-B(T_0, T_i)\cdot r^*\bigr)$$

These are the ZCB prices that would prevail if the short rate equals $r^*$
at expiry.

Now observe that for any realisation of $r(T_0)$:

$$\max\!\bigl(1 - CB(T_0),\ 0\bigr)
= \sum_{i=1}^n c_i\,\max\!\bigl(K_i - P(T_0, T_i),\ 0\bigr)$$

**Why does this hold?** Because all ZCB prices move together through the
single factor $r(T_0)$. The event $\{CB < 1\} = \{r > r^*\}$ implies
$\{P(T_0, T_i) < K_i\}$ for **all** $i$ simultaneously. So either all
component puts are in the money together, or none of them are — the
decomposition is exact.

### 9.5 The Pricing Formula

Taking expectations under $\mathbb{Q}$ and discounting:

$$\boxed{V_{\text{payer}} = \sum_{i=1}^n c_i
\cdot \text{ZBPut}(0;\ T_0,\ T_i,\ K_i)}$$

$$\boxed{V_{\text{receiver}} = \sum_{i=1}^n c_i
\cdot \text{ZBCall}(0;\ T_0,\ T_i,\ K_i)}$$

Each $\text{ZBPut}$ and $\text{ZBCall}$ is priced using the analytical
formula from Section 8.

### 9.6 Algorithm Summary

The complete pricing algorithm is:

1. Given the market discount curve, model parameters $(a, \sigma)$, swap
   schedule, and strike $K$
2. Compute coupon weights $c_i = K\delta_i$ for $i < n$ and
   $c_n = 1 + K\delta_n$
3. Find $r^*$ by solving $\sum_i c_i\,P(T_0, T_i; r) = 1$ using Brent's
   method
4. Compute ZCB strikes $K_i = P(T_0, T_i; r^*)$ for each $i$
5. Price each ZCB option using the Hull-White analytical formula
6. Sum: $V = \sum_i c_i \cdot \text{ZBOpt}(0; T_0, T_i, K_i)$

This algorithm is **exact** under the Hull-White model, requires only one
numerical root-find plus $n$ closed-form evaluations, and is consistent
with the initial discount curve by construction.

---

## 10. Model Calibration

### 10.1 Calibration Targets

With the discount curve $P(0, T)$ fixed by the market, the two free
parameters $(a, \sigma)$ are calibrated to a **basket of liquid market
swaptions**. A typical calibration basket covers a grid of expiries and
tenors — for example 1Y×2Y, 1Y×5Y, 2Y×3Y, 2Y×5Y, 5Y×5Y — chosen to span
the range of maturities relevant to the portfolio being hedged.

### 10.2 Objective Function

Calibration minimises the **weighted sum of squared relative pricing
errors**:

$$\min_{a,\,\sigma\,>\,0}\ \sum_{k=1}^{K} w_k
\left[\frac{V_k^{\text{HW}}(a,\sigma) - V_k^{\text{mkt}}}{V_k^{\text{mkt}}}
\right]^2$$

where $w_k$ are instrument weights (typically equal weights, or
vega-weighted to emphasise liquid instruments). The optimisation is carried
out using the **L-BFGS-B** algorithm with box constraints $a \in (0, 2)$
and $\sigma \in (0, 0.10)$.

### 10.3 Interpretation of Calibrated Parameters

After calibration, the parameters have clear economic meaning:

- **$a \in [0.01, 0.10]$**: slow mean reversion, consistent with
  environments where rate persistence is high and the yield curve exhibits
  large parallel shifts. Typical in normal market conditions.
- **$a \in [0.10, 0.30]$**: moderate mean reversion, reflecting central
  bank active rate management.
- **$\sigma \in [0.005, 0.020]$**: short rate volatility translating to
  roughly 50–200 basis points of Normal swaption volatility depending on
  the swap tenor and option expiry.

---

## 11. Market Quoting Convention — Normal Volatility

Market participants quote swaption prices not in currency units but as
**implied Normal (Bachelier) volatilities** $\sigma_N$. Under the Bachelier
(1900) model the swap rate $S$ follows:

$$dS = \sigma_N\,dW$$

giving the swaption price:

$$V = N \cdot A \cdot \left[\omega\,(S_0 - K)\,N(\omega\,d)
+ \sigma_N\sqrt{T}\,\phi(d)\right]$$

$$d = \frac{\omega\,(S_0 - K)}{\sigma_N\sqrt{T}},
\quad \omega = +1\ \text{(payer)},\ -1\ \text{(receiver)}$$

where $\phi(\cdot)$ is the standard normal PDF.

Normal volatility is preferred over lognormal (Black-76) volatility because
it handles near-zero and negative interest rates without modification.
The Hull-White model price can be inverted numerically to give an implied
Normal volatility for comparison with market quotes.

---

## 12. Model Limitations

The Hull-White one-factor model is the **industry standard** for vanilla
swaption pricing but has well-known limitations:

| Limitation | Description |
|---|---|
| **Negative rates** | The Gaussian short rate can become negative. This was considered a theoretical deficiency before 2008 but became empirically relevant during the negative rate environment in Europe (2014–2022). |
| **One factor** | All rates are driven by a single Brownian motion. The model cannot reproduce imperfect correlations between rates at different tenors, which are observed in practice. |
| **No volatility smile** | The model assigns the same $\sigma_P$ to all options on ZCBs of the same expiry, regardless of strike. It cannot fit the volatility smile or skew observed in swaption markets. |
| **Constant parameters** | $a$ and $\sigma$ are assumed constant over time. Time-dependent extensions $a(t)$, $\sigma(t)$ improve fit but lose some analytical tractability. |
| **Single curve** | The model uses a single discount curve. Post-2008 multi-curve frameworks (OIS discounting, tenor basis) require extensions. |

For a linear pricing exercise — pricing and hedging vanilla swaptions at a
single strike — the one-factor Hull-White model is entirely appropriate and
widely used in practice for book management and risk reporting.

Extensions that address these limitations include:

- **G2++ (Hull-White two-factor)**: adds a second factor to capture
  imperfect correlation between short and long rates
- **SABR-HW**: combines stochastic volatility (SABR) with Hull-White
  dynamics to capture the volatility smile
- **LMM (LIBOR Market Model)**: models forward rates directly, naturally
  accommodating multi-curve frameworks and smile

---

## 13. References

Bachelier, L. (1900). *Théorie de la spéculation*. Annales Scientifiques
de l'École Normale Supérieure, 17, 21–86.

Brigo, D., & Mercurio, F. (2006). *Interest Rate Models — Theory and
Practice* (2nd ed.). Springer Finance.

Duffie, D., & Kan, R. (1996). A yield-factor model of interest rates.
*Mathematical Finance*, 6(4), 379–406.

Hull, J., & White, A. (1990). Pricing interest-rate derivative securities.
*The Review of Financial Studies*, 3(4), 573–592.

Hull, J., & White, A. (1994). Numerical procedures for implementing term
structure models I: Single-factor models. *The Journal of Derivatives*,
2(1), 7–16.

Jamshidian, F. (1989). An exact bond option formula. *The Journal of
Finance*, 44(1), 205–209.

Vasicek, O. (1977). An equilibrium characterization of the term structure.
*Journal of Financial Economics*, 5(2), 177–188.
