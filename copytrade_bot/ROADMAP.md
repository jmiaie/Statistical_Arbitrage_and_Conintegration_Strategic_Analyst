# Project Direction & Roadmap

*Full-codebase review, 2026-06-10. This document steers what gets built next and why.*

## What this project actually is

Three things share this repo today:

1. **A copy-trade pipeline** (`copytradebot/`): Telegram alert → parse → enrich → filter →
   size → risk gate → executor. Complete and tested through paper execution.
2. **A scanner research library** (`scanners/`): arbitrage, longshot-bias, mean-reversion,
   cointegration. Validated only against synthetic data (`scanner_lab.py`); **not wired to
   live data or execution** — multi-leg opportunities cannot flow through the pipeline at all.
3. **A simulation/optimization toolkit** (`simulation/`): Monte Carlo sweeps with an honest
   channel model (report bias, no-edge stress), robustness ranking by worst case. This is the
   strongest part of the codebase.

The root README describes a *fourth* thing (Kalman-filter equity stat-arb) that does not exist
here. Decision: **the project is the Polymarket system.** The README gets rewritten to match
reality, not the other way around.

## Honest edge assessment (what is worth real money, in order)

| Edge | Verdict |
|---|---|
| **Copy-trading** | The product. Edge is borrowed from channels; our edge is *measurement* — log everything, settle against real resolutions, shrink quoted win rates toward realized. Most infrastructure already exists. |
| **Arbitrage** (outcome asks sum < 1) | Real and mechanical but heavily competed; expect near-zero volume. Its true value: **the safest live shakedown** of the execution stack, since every trade is hedged by construction. |
| **Longshot bias** | Documented, thin, slow. Worth one calibration script against *real resolved markets* (free via Gamma API) before any trading. The default bias curve is a placeholder — do not trade it. |
| **Mean reversion** | Weakest. News-vs-noise guard is crude and NO-leg pricing is optimistic by the full spread. Parked: research only. |
| **Cointegration** | The repo's title, the weakest implementation. EG/ADF on 60-point bounded series across all pairs is a false-positive machine, and there is no exit path to realize convergence. Either rebuild properly (related-market universes, real critical values, share-count hedging, exit engine) or keep as research. Parked. |

## Blocking defects (fix before any live order)

These were found in review and are tracked as P0:

1. **Live executor does not match its own safety claim.** `polymarket.py:_resolve_token` takes
   `markets[0]` from Gamma search (not the word-overlap ranker that already exists in
   `marketdata.py`), silently falls back to outcome index 0, and defaults price to 0.5. The
   docstring says it "refuses rather than guesses" — make that true: reuse `find_market`,
   require a minimum match score, refuse on ambiguity, never default the price.
2. **No idempotency.** `edited_channel_post` re-fires the pipeline; reposts double-trade.
   Store Telegram message ids; skip already-seen ids.
3. **No cash/exposure accounting.** Defaults allow 25 open × $200 = $5,000 exposure on a
   $1,000 bankroll; paper stakes are never debited from anything. Add a tracked cash balance
   and a total-open-exposure cap (e.g. ≤ 50% of bankroll). Also fix `Pipeline._day_start`
   (computed once at construction — "daily" loss never rolls over).
4. **No exit support anywhere.** Executors can only open and settle-at-resolution. The parser
   has no exit intent — a channel posting "closing NVDA position" parses as a fresh entry.
   Needed: parser exit/update detection + a sell/close path in paper and live executors.
5. **Multi-leg execution missing.** `Opportunity.to_signal()` raises for multi-leg, which
   excludes exactly the two scanners worth running (arbitrage, cointegration).

## Phased plan

**P0 — Safety floor (before any live dollar):** items 1–5 above, each with tests.

**P1 — Real-data validation (no money at risk):**
- Scanner daemon: poll Gamma for active markets on a schedule, build snapshots/series, run
  scanners, log opportunities to SQLite, route through the realistic paper engine. Converts
  scanners from synthetic toys into evidence generators (expected finding: arb is rare/instantly
  taken — that's a result, not a failure).
- Longshot calibration script: resolved markets → realized frequency vs. price → fitted bias curve.
- **Channel calibration layer (the durable edge):** per source, compare quoted vs. realized win
  rate over settled positions; shrink quoted WR before filtering/Kelly. Kelly on an inflated WR
  is the most dangerous default in the system; calibration is what makes copy-trading safe.
- Parser hardening on real alerts from the user's channels (blocked on samples).

**P2 — Live, tiny:**
- Arbitrage first, $5–10 legs, as an execution shakedown (order placement, fills, settlement).
- Copy-trading live only after ≥ 30–50 *settled* paper trades from the actual channel show
  positive realized EV at realistic fills.

**P3 — Positioning/cleanup:**
- Rewrite root README around what exists (the honest-simulation machinery is the showcase
  piece). Remove `generate_pdf.py` and dead references.

## Principles (carry into every change)

- Paper results must track live mechanics: book-walk fills, fees, real resolution. Anything
  that can't be verified live gets a `degraded` flag, never a silent fallback.
- Quoted numbers from channels are claims, not facts. Every claim gets calibrated against
  settled outcomes before it sizes a position.
- Prefer strategies that fail cheap and measurably. Ranking is by worst case across scenarios,
  not best case in one.
