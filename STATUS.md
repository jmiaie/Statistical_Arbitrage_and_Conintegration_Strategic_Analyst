# Status — Statistical_Arbitrage_and_Conintegration_Strategic_Analyst

**Updated:** 2026-09-30 (PT)  
**Visibility:** public  
**Maturity:** stalled / WIP scaffold (not a shipped strategy)  
**Private twin:** [`Statistical_Arbitrage_and_Conintegration_Strategic_Analyst_priv`](https://github.com/jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst_priv)

## Honest positioning

This repository’s **name and older README pitch** describe a full Kalman / cointegration / Fama–French / TCA research stack. **That product is not present in this tree.**

What is actually on `main` today:

| Path | What it is |
|------|------------|
| `copytrade_bot/copytradebot/kalshi.py` | Kalshi **public market-data** helper (fetch/parse helpers) — unfinished copy-trade / venue adapter, not a pairs engine |
| `generate_pdf.py` | PDF report generator whose header still says **“Quant ML: Sentiment-Augmented Price Prediction”** — leftover from another chapter |
| `README.md` (pre-honesty) | Marketing outline of methods that **were never landed as code here** |

There is **no** `requirements.txt`, research notebook, backtester, cointegration screen, Kalman filter module, or factor-attribution pipeline in this remote.

## What is **not** claimed

- Institutional-grade live or paper **alpha**, Sharpe, IC, capacity, or drawdown figures
- That KPI language in older drafts equals measured results
- That this public remote is the canonical home for quant chapters (see hub below)
- Parity with open GitHub issues titled as full “implement statistical arbitrage pipeline” work — those issues describe aspirational scope, not delivered code

## Related repos (pointers only)

| Repo | Role |
|------|------|
| [`quant-research-portfolio`](https://github.com/jmiaie/quant-research-portfolio) | Private portfolio hub / methodology chapters |
| [`ML_Sentiment_Augmented_Price_Predictor`](https://github.com/jmiaie/ML_Sentiment_Augmented_Price_Predictor) | Canonical sentiment methodology (failure-to-demonstrate historical study) |
| [`kalshi-scalping-bot`](https://github.com/jmiaie/kalshi-scalping-bot) / [`kalshi-arbitrage-bot`](https://github.com/jmiaie/kalshi-arbitrage-bot) | Separate Kalshi experiments — **not** the same as this repo |
| Private twin (above) | Same pitch README historically; also **not** a complete strategy package |

## Offline / local notes

- No install or test entrypoint is defined for a strategy run.
- `kalshi.py` talks to Kalshi’s **public** HTTP API shapes; treat any live calls as exploratory and rate-limit politely. Do not invent filled-trade or PnL claims from this file alone.
- Prefer reading/writing methodology under the quant hub or dedicated bot extracts rather than growing this name-hold into a second parallel stack.

## Next (owner)

1. Either **archive** this name as historical, or land a **minimal** reproducible pairs notebook under a clearer repo
2. Close or retarget the open “full pipeline” issue so it matches reality
3. Keep career/recruiting outbound gated (CAREER gate off for bots)
