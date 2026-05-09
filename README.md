<div align="center">

# quant-pairs-lab

### Dynamic Statistical Arbitrage &middot; Cointegration &middot; Risk-Aware Backtesting

*A Kalman-filtered pairs trading research stack with Fama–French attribution and a square-root market-impact model.*

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-3B5C8C)](https://www.statsmodels.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-research-blue)](#)

**Author:** Jeff Milam, MBA &nbsp;·&nbsp; [GitHub](https://github.com/jmiaie) &nbsp;·&nbsp; [Email](mailto:jmilam.emba@gmail.com)

[English](./README.md) &nbsp;·&nbsp; [Español](./docs/i18n/README.es.md) &nbsp;·&nbsp; [中文](./docs/i18n/README.zh.md) &nbsp;·&nbsp; [日本語](./docs/i18n/README.ja.md) &nbsp;·&nbsp; [Français](./docs/i18n/README.fr.md)

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture](#2-architecture)
3. [Quantitative Methodology](#3-quantitative-methodology)
4. [Execution & Transaction Cost Analysis](#4-execution--transaction-cost-analysis)
5. [Key Performance Indicators](#5-key-performance-indicators)
6. [Tech Stack](#6-tech-stack)
7. [Quickstart](#7-quickstart)
8. [Repository Layout](#8-repository-layout)
9. [Roadmap](#9-roadmap)
10. [Citation](#10-citation)
11. [License](#11-license)

---

## 1. Executive Summary

`quant-pairs-lab` is a research-grade statistical arbitrage stack built around three principles that distinguish institutional pairs trading from textbook examples:

- **Adaptive hedge ratios.** A Kalman filter replaces static OLS so that the relationship between paired assets evolves with the market.
- **Pure idiosyncratic alpha.** Returns are decomposed against the Fama–French 3-Factor Model to confirm market-neutrality and isolate skill from style tilts.
- **Honest economics.** A non-linear (square-root) transaction cost model and latency-sensitivity analysis stress-test capacity before any P&L is celebrated.

The result is a strategy artifact that can be discussed credibly with quant researchers, risk officers, and execution traders alike.

---

## 2. Architecture

```mermaid
flowchart LR
    A[Universe<br/>Selection] --> B[Cointegration<br/>Engle-Granger / Johansen]
    B --> C[Kalman Filter<br/>Dynamic β_t]
    C --> D[Spread / Z-Score<br/>Signal Generation]
    D --> E[Position Sizing<br/>Vol-Targeted]
    E --> F[Execution Layer<br/>Slippage + Latency]
    F --> G[P&L Engine]
    G --> H[Fama-French<br/>Attribution]
    G --> I[Risk Metrics<br/>Sharpe · MDD · IC]
    H --> J[Research Report]
    I --> J
```

---

## 3. Quantitative Methodology

### A. Pair Selection & Cointegration

- **Screening** — sector-neutral, liquidity-filtered universe with rolling correlation thresholds.
- **Statistical validation** — Engle–Granger two-step procedure for pairs and the Johansen test for multivariate baskets.
- **Stationarity** — ADF / KPSS confirmation that the residual spread is I(0) and mean-reverting.

### B. Signal Processing — Kalman Filter

A state-space formulation lets the hedge ratio drift smoothly through regime shifts without look-ahead bias:

$$
\begin{aligned}
y_t &= \beta_t \, x_t + \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R) \\
\beta_t &= \beta_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)
\end{aligned}
$$

Trading signals are generated from the standardised innovation (z-score) of the spread, with entry/exit thresholds calibrated on out-of-sample residual distributions.

### C. Risk & Factor Attribution

Strategy returns $r_t$ are regressed on the Fama–French factors:

$$
r_t - r_{f,t} = \alpha + \beta_{\text{MKT}}(R_{m,t}-r_{f,t}) + \beta_{\text{SMB}} \cdot \text{SMB}_t + \beta_{\text{HML}} \cdot \text{HML}_t + \epsilon_t
$$

The intercept $\alpha$ is the headline number; the betas are diagnostic guardrails confirming the strategy is not a closet beta, size, or value bet.

---

## 4. Execution & Transaction Cost Analysis

| Component | Model | Purpose |
|---|---|---|
| **Slippage** | Square-root law: $\text{cost} \propto \sigma \sqrt{Q / \text{ADV}}$ | Realistic impact at scale |
| **Latency** | P&L decay vs. execution delay (ms) | Quantify alpha half-life |
| **Capacity** | Sharpe degradation curve vs. notional | Maximum deployable AUM |
| **Borrow** | Short-rebate haircut on leg-2 | Fair net-of-financing returns |

---

## 5. Key Performance Indicators

| Metric | Definition | Why it matters |
|---|---|---|
| **Net Sharpe** | Annualised, post-cost, post-borrow | The only Sharpe that counts |
| **Max Drawdown** | Peak-to-trough with recovery time | Tail risk and investor patience |
| **Information Coefficient** | Rank-correlation of signal to forward return | Predictive power, not just P&L |
| **Rolling Factor Betas** | 60-day window vs. Mkt / SMB / HML | Confirms persistent neutrality |
| **Turnover & Hit Rate** | % capital recycled, % winning trades | Sanity checks on style |

---

## 6. Tech Stack

- **Language:** Python 3.10+
- **Quantitative:** NumPy, pandas, SciPy, statsmodels
- **State estimation:** PyKalman (or in-house Kalman implementation)
- **Data:** yfinance, pandas-datareader, Ken French data library
- **Visualisation:** Matplotlib, seaborn
- **Reporting:** Jupyter, ReportLab (`generate_pdf.py`)

---

## 7. Quickstart

```bash
# 1. Clone
git clone https://github.com/jmiaie/quant-pairs-lab.git
cd quant-pairs-lab

# 2. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the research notebook
jupyter notebook research/main_backtest.ipynb

# 4. Generate the institutional PDF tear-sheet
python generate_pdf.py
```

---

## 8. Repository Layout

```
quant-pairs-lab/
├── README.md                  # You are here
├── LICENSE                    # Proprietary, all rights reserved
├── CITATION.cff               # How to cite this work
├── CONTRIBUTING.md            # Collaboration guidelines
├── requirements.txt
├── generate_pdf.py            # Tear-sheet generator
├── docs/
│   ├── i18n/                  # Translated READMEs (es, zh, ja, fr)
│   └── methodology.md         # Extended quantitative notes
├── examples/                  # Minimal runnable examples
└── .github/                   # Issue & PR templates
```

---

## 9. Roadmap

- [ ] Vectorised multi-pair backtester with walk-forward optimisation
- [ ] Bayesian online learning for noise covariances $Q, R$
- [ ] Regime-switching (HMM) overlay for entry suppression
- [ ] Intraday extension with limit-order book microstructure costs
- [ ] LLM-assisted research agent (cointegration candidate screener)

---

## 10. Citation

If you reference this work, please cite via the `CITATION.cff` metadata or:

> Milam, J. (2026). *quant-pairs-lab: Dynamic Statistical Arbitrage & Risk-Aware Backtesting*. GitHub. https://github.com/jmiaie/quant-pairs-lab

---

## 11. License

Copyright © 2026 Jeff Milam, MBA. All rights reserved. This code is proprietary; unauthorised copying, distribution, or derivative use via any medium is strictly prohibited. See [`LICENSE`](./LICENSE) for full terms.

<div align="center">
<sub>Built for rigour. Designed for the desk.</sub>
</div>
