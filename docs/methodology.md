# Extended Methodology

This document expands on the summary in the main `README.md` for readers who want the full quantitative reasoning.

## 1. Universe construction

The candidate universe is restricted to liquid US equities and ETFs with continuous price history over the test window. Filters applied in order:

1. Average daily dollar volume (ADV) ≥ $20M over the trailing 60 trading days.
2. Sector tagging via GICS; only intra-sector pairs are considered to keep economic linkage plausible.
3. Rolling 60-day return correlation ≥ 0.75 as a coarse pre-filter before formal cointegration testing.

## 2. Cointegration tests

**Engle–Granger (pairs).** For candidate $(y, x)$ we estimate $y_t = \beta x_t + \alpha + u_t$ by OLS and apply ADF to $\hat{u}_t$. Pairs with ADF p-value < 0.05 advance.

**Johansen (baskets).** For multi-leg baskets we apply the trace and max-eigenvalue tests at the 95% critical level. Cointegrating vectors are normalised so the first element equals 1.

## 3. Kalman state-space model

We model the hedge ratio as a random walk with observation noise:

- State: $\beta_t = \beta_{t-1} + \eta_t$, $\eta_t \sim \mathcal{N}(0, Q)$
- Observation: $y_t = \beta_t x_t + \alpha_t + \varepsilon_t$, $\varepsilon_t \sim \mathcal{N}(0, R)$

Hyperparameters $Q$ and $R$ are tuned by maximum likelihood on a hold-out training window. The filtered state $\hat{\beta}_{t|t}$ is used to construct the spread $s_t = y_t - \hat{\beta}_{t|t} x_t$, which is then standardised by its rolling estimated standard deviation to form the trading z-score.

## 4. Trading rules

- **Entry:** $|z_t| \geq 2.0$
- **Exit:** $|z_t| \leq 0.5$ or stop-out at $|z_t| \geq 4.0$
- **Sizing:** equal-dollar legs scaled by inverse trailing realised volatility to a portfolio target of 10% annualised.

These thresholds are illustrative defaults; the research notebook performs a walk-forward grid search.

## 5. Transaction cost model

Per-trade cost in basis points is modelled as:

$$\text{cost}_{\text{bps}} = \kappa \cdot \sigma_{\text{daily}} \cdot \sqrt{\frac{Q}{\text{ADV}}}$$

with $\kappa$ calibrated from public TCA studies (Almgren et al. 2005). Borrow on the short leg is deducted at the broker rebate rate.

## 6. Performance attribution

Daily strategy returns $r_t$ are regressed on the Fama–French 3-factor returns:

$$r_t - r_{f,t} = \alpha + \beta_{\text{MKT}}(R_{m,t}-r_{f,t}) + \beta_{\text{SMB}}\text{SMB}_t + \beta_{\text{HML}}\text{HML}_t + \epsilon_t$$

We report annualised $\alpha$, its Newey–West t-statistic, and the rolling 60-day betas.

## 7. References

- Engle, R., & Granger, C. (1987). *Co-integration and Error Correction*. Econometrica.
- Johansen, S. (1991). *Estimation and Hypothesis Testing of Cointegration Vectors*. Econometrica.
- Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). *Direct Estimation of Equity Market Impact*.
- Fama, E., & French, K. (1993). *Common risk factors in the returns on stocks and bonds*. JFE.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
