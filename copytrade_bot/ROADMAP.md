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

These were found in review and tracked as P0. **All five are now resolved** (each with tests;
suite at 66 passing).

1. ✅ **Live executor now refuses rather than guesses.** `select_market()` ranks Gamma
   candidates by word overlap and raises on a weak (<2 shared words) or ambiguous match;
   side mapping refuses instead of falling back to outcome 0; a missing entry price is
   refused instead of defaulting to 0.5. *(commit: live-executor + idempotency)*
2. ✅ **Idempotency.** `seen_updates` table + `Storage.mark_seen()`; the bot dedupes source
   posts by chat+message id, so edits/redelivery can't re-fire a trade. Survives restarts.
3. ✅ **Exposure accounting + daily rollover.** `RiskConfig.max_exposure_fraction` (default
   0.5×bankroll) enforced via `Storage.open_exposure()`; `_today_start()` recomputed per
   check so the daily-loss limit rolls over. Surfaced in `/status`.
4. ✅ **Exit support.** `Signal.intent` (ENTRY/EXIT), conservative exit + exit-price parsing,
   `Storage.find_open_by_market()`, and `Pipeline._process_exit()` that settles matching
   open positions and never opens one. Live-mode auto-close explicitly declined for now.
5. ✅ **Multi-leg execution.** `Opportunity.to_signals()` (plural) + `Pipeline.place_opportunity()`
   gate on basket-level edge/confidence, size once and split by leg weight, and link legs via a
   shared `signal_id`. Unblocks arbitrage + cointegration execution.

**Remaining P0 follow-ups (smaller, not yet done):** NO-leg pricing optimism (scanners price
NO at `1−YES_ask`, ignoring its own spread); Kelly sizing still keys off the channel's *quoted*
win rate (real fix is the P1 calibration layer); cointegration multiple-testing control.

## Phased plan

**P0 — Safety floor (before any live dollar):** ✅ items 1–5 complete. Follow-ups above remain.

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
