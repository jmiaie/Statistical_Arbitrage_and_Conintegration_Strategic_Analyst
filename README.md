# Statistical_Arbitrage_and_Conintegration_Strategic_Analyst

> **Status (2026-09-30):** **Stalled / WIP name-hold** — not an institutional product and **not** a complete statistical-arbitrage codebase. See [`STATUS.md`](STATUS.md).

**Public** Python remote historically pitched as “Dynamic Statistical Arbitrage & Risk-Aware Backtesting.”  
**Author context:** Jeff Milam ([`jmiaie`](https://github.com/jmiaie)).

## What you will find here

- A **Kalshi market-data helper** under `copytrade_bot/copytradebot/kalshi.py` (exploratory venue adapter).
- A leftover `generate_pdf.py` aimed at a **different** project title (sentiment PDF).
- **No** cointegration screen, Kalman hedge-ratio engine, Fama–French attribution, TCA stack, or verified backtest results in-tree.

Older README language that described Sharpe / IC / capacity KPIs and “institutional-grade” delivery described **intent**, not measured outcomes in this repository. **No performance figures are claimed here.**

## Canonical / related homes

| Concern | Prefer |
|---------|--------|
| Quant portfolio narrative | [`quant-research-portfolio`](https://github.com/jmiaie/quant-research-portfolio) |
| Sentiment methodology chapter | [`ML_Sentiment_Augmented_Price_Predictor`](https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor) |
| Private twin of this name | [`…_priv`](https://github.com/jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst_priv) |

## Historical methodology outline (aspirational — not implemented here)

The following bullets are retained only as a **research wishlist**, not as a claim that this repo executes them:

- Pair / basket cointegration checks (Engle–Granger / Johansen-style)
- Dynamic hedge ratios (state-space / Kalman-style)
- Factor-neutrality checks (e.g. Fama–French-style regressions)
- Transaction-cost and capacity sensitivity

If those land, they should ship with reproducible code, offline fixtures, and honest null/failure reporting — not KPI theater.

## How to inspect (offline)

```bash
git clone https://github.com/jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst.git
cd Statistical_Arbitrage_and_Conintegration_Strategic_Analyst
# Read STATUS.md first. There is no requirements.txt or strategy CLI on main today.
```

Private twin clone URL (if needed): `https://github.com/jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst_priv.git`
