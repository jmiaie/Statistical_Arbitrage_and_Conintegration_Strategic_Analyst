# Statistical_Arbitrage_and_Conintegration_Strategic_Analyst
PUBLIC: Statistical Arbitrage and Conintegration Strategic Analyst (Python)
## Project Title: Dynamic Statistical Arbitrage & Risk-Aware Backtesting

## 👤 Author: Jeff Milam, EMBA | jmilam.emba@gmail.com | https://www.github.com/jmiaie



### 1. Executive Summary

This repository implements an institutional-grade statistical arbitrage strategy. Unlike traditional static pairs trading, this project utilizes a Kalman Filter for dynamic state estimation of hedge ratios and a Fama-French Factor Model to isolate idiosyncratic alpha. The strategy is rigorously tested against a non-linear Transaction Cost Model to determine capacity and execution sensitivity.

### 2. Quantitative Methodology

A. Pair Selection & Cointegration

    Screening: Universe selection based on sector-neutrality and high correlation.

    Statistical Validation: Implementation of the Engle-Granger two-step method and Johansen Test for multivariate cointegration (baskets).

    Stationarity: Checking for I(0) residuals in the spread to ensure mean-reversion.

B. Signal Processing (Kalman Filter)

The strategy moves beyond static OLS by using a state-space representation to model the relationship between assets:

    Observation: Yt​=βt​Xt​+αt​+ϵt​

    Transition: βt​=βt−1​+ηt​ This allows the algorithm to adapt to "Regime Shifts" and structural breaks in the market without look-ahead bias.

C. Risk & Factor Attribution

To ensure the strategy isn't just a proxy for a known risk factor, we regress strategy returns against the Fama-French 3-Factor Model:

    Market (Mkt-RF): Proving market-neutrality (Beta ≈ 0).

    Size (SMB) & Value (HML): Ensuring returns are not driven by simple style tilts.

    Alpha Tracking: Measuring the intercept to confirm statistically significant skill.

### 3. Execution & TCA (Transaction Cost Analysis)

Recognizing that "paper trades" are deceptive, this project incorporates:

    Slippage Model: A Square-Root Law impact function based on daily volatility and % of ADV.

    Alpha Decay: Analysis of P&L sensitivity to execution delay (latency).

    Capacity Estimation: Determining the maximum trade size before market impact erodes the Sharpe ratio.

### 4. Key Performance Indicators (KPIs)

    Annualized Sharpe Ratio: (Net of costs).

    Maximum Drawdown: Recovery time analysis.

    Information Coefficient (IC): Measuring signal predictive power.

    Factor Exposures: Heatmap of rolling betas to major indices.

### 5. Tech Stack

    Languages: Python 3.x

    Quantitative Libraries: NumPy, Pandas, Statsmodels, SciPy

    Signal Processing: PyKalman (or custom implementation)

    Data/Backtesting: yfinance, Pandas-Datareader, Matplotlib

### 6. How to Run

    Bash

    **Clone the repository**
    git clone https://github.com/jmiaie/stat-arb-kalman.git

    **Install dependencies**
    pip install -r requirements.txt

    **Run the research notebook**
    jupyter notebook research/main_backtest.ipynb


__________________________________________________________________________________________________________

## Copyright (c) 2026 Jeff Milam, EMBA. All Rights Reserved.
## GitHub: https://github.com/jmiaie
##
## This code is proprietary and private. Unauthorized copying of this file,
## via any medium, is strictly prohibited.
