# Telegram Copy-Trading / Alert-Filtering Bot

An agentic copy-trading bot that listens to trade alerts coming through a
Telegram channel/group (via a bot you control), **parses the freeform text**,
runs each alert through **fully adjustable filters** (win rate, EV, ROI,
expected return, entry price, trade size, source, …), **sizes** a position,
and routes the survivors to a pluggable **executor** — paper trading now, live
**Polymarket** when you're ready to put money at risk.

```
Telegram alert ─▶ parse ─▶ enrich ─▶ calibrate ─▶ filter ─▶ size ─▶ risk gate ─▶ executor
                  │          │          │            │         │         │           │
              regex/heur  derive EV  per-source   thresholds Kelly/frac exposure  paper |
              + intent              WR correction                       + caps    polymarket
                                      (adjustable, persisted to YAML, live via /set)
```

Entry alerts open positions; **exit alerts** ("closing NVDA", "sold out", "TP
hit", "out at 0.82") settle the matching open position instead of opening a new
one. Each source's **quoted win rate is calibrated** against its realized
settled outcomes before it reaches the filter and Kelly sizing.

## Why each piece exists

| Concern | Where | Notes |
|---|---|---|
| Read alerts | `telegram_client.py` | Bot API long-polling, `requests` only; updates deduped by message id |
| Understand alerts | `parser.py` | Tolerant regexes; entry/exit **intent**; `enrich()` derives EV from win-rate + price |
| Calibrate | `calibration.py` | Shrink each source's quoted win rate toward its realized rate |
| Decide | `filters.py` + `config.py` | Every threshold is adjustable and persisted |
| Size | `sizing.py` | `fixed` / `fraction` / fractional-`kelly` (off the *calibrated* win rate) |
| Execute | `executors/` | `paper.py` (simulated) and `polymarket.py` (live, guarded) |
| Remember | `storage.py` | SQLite: signals, positions, P&L, win rate, dedup, exposure |
| Control | `bot.py` | Telegram command surface |

See [`ROADMAP.md`](ROADMAP.md) for the project review, what's safe to trade,
and the phased plan.

## Quick start

```bash
cd copytrade_bot
pip install -r requirements.txt
cp .env.example .env          # fill in TELEGRAM_BOT_TOKEN etc.

# Try it offline first — no Telegram needed:
python run.py --test "Market: Will BTC close >70k? YES entry 0.42 win rate 68% EV +14% size $75"

# Then run the live listener:
python run.py
```

Add your bot to the channel/group as an admin so it receives posts, and put
that chat's numeric id in `TELEGRAM_SOURCE_CHAT_IDS`. Put your own user id in
`TELEGRAM_ADMIN_CHAT_IDS` so only you can change settings.

## Adjusting the filters

Settings live in `config/filters.yaml` and can be changed **live from Telegram**
(changes are written back to the YAML, so they survive restarts):

```
/status                       show mode + filter summary
/filters                      dump full config as JSON
/set min_win_rate 0.62        require ≥62% win rate
/set min_ev 8                 require ≥+8% EV
/set min_roi 20               require ≥+20% ROI potential
/set max_entry_price 0.6      skip plays priced above 0.60
/set min_size 25              ignore tiny suggested sizes
/set mode kelly               size by half-Kelly
/set bankroll 5000            update bankroll
/set max_position 250         hard cap per trade
/require win_rate,ev          require these fields be parsed
/dryrun on                    evaluate + size but don't place
/calibrate on|off             correct quoted win rates vs realized history
/calibration                  per-source quoted vs realized win rate + bias
/enable | /disable            master switch
/test <alert>                 dry-run any alert text through the filters
```

Once positions settle, `/calibration` shows each source's quoted vs. realized
win rate and the bias being applied — the channels that overstate get
discounted automatically.

Performance & positions:

```
/stats                        signals seen/passed, win rate, realized P&L
/positions                    open positions
/resolve 7 win                settle paper position #7 as a win
/resolve 7 0.80               settle at an exit price of 0.80
/recent 20                    last 20 signals with pass/fail reasons
```

### Adjustable criteria (all optional bounds; `null`/`none` disables)

- `min_win_rate` — fraction 0–1
- `min_ev`, `min_roi`, `min_expected_return` — percent
- `min_entry_price` / `max_entry_price`
- `min_size` / `max_size`
- `require_fields` — alert is skipped unless these parsed
- `blocked_keywords`, `allowed_sources`, `blocked_sources`
- Sizing: `mode`, `fixed_amount`, `bankroll`, `fraction`, `kelly_fraction`,
  `max_position`, `min_position`
- Risk: `max_open_positions`, `max_daily_loss`, `max_exposure_fraction`
  (caps total open stake as a fraction of bankroll; the daily-loss window rolls
  over each day)

## Realistic paper trading (verify likely outcomes)

Paper trading defaults to a naive fill (alert's quoted price, no costs). To
make paper results track what live execution would actually produce, turn on
the realistic engine — it fills against **real Polymarket data** with
**slippage and fees**, then **auto-settles against the market's real
resolution**:

```
/set data_source polymarket    # pull live market data (public, no API key)
/set fill_model book           # size-aware fills walking the live order book
/set slippage_bps 50           # extra pad (used by alert/mid models)
/set fee_bps 0                 # taker fee (Polymarket is 0 today)
```

Fill models:

| `fill_model` | Fill price | Use when |
|---|---|---|
| `alert` | the alert's quoted entry (+ slippage pad) | no market data |
| `mid`   | live midpoint/ask (+ slippage pad) | quick, book not needed |
| `book`  | walks the **live order book** so bigger stakes pay up | most realistic |

What you get:

- **Effective fill price + share count** recorded per position, with slippage
  attribution in `meta` (`reference_price`, `alert_price`, `slippage_bps`,
  `fee_paid`, `unfilled_stake`).
- **`/mtm`** — mark every open position to the live market for unrealized P&L.
- **`/settle`** — auto-settle positions whose Polymarket market has resolved
  on-chain (real win/loss, real ROI), feeding the rolling win rate in `/stats`.
- **Honest degradation** — if a live market can't be matched it falls back to
  the alert price and flags `degraded` in `meta` (or set
  `require_live_market: true` to skip instead, keeping results pure).

This requires outbound network access to Polymarket's public Gamma + CLOB
endpoints. No API keys are needed for *data* (only for live order placement).
The endpoint field-mapping lives in one place (`marketdata.py`) — verify it
against live responses before sizing up.

## Finding the best strategy (Monte Carlo)

Before risking anything, sweep filter + sizing strategies over thousands of
simulated alert streams and let the data pick the most profitable *and*
consistent one:

```bash
python simulate.py                         # synthetic, default channel model
python simulate.py --quick                 # fast, smaller sweep
python simulate.py --channel-skill 0.0     # stress: a channel with NO edge
python simulate.py --report-bias 0.10      # channel inflates win rate by +10pp
python simulate.py --history alerts.jsonl  # backtest on YOUR real alerts
python simulate.py --save-best             # write the winner into the config
```

It reuses the **same** filter/sizing/slippage code the live bot runs, so the
strategy you optimise is the one you deploy. Each candidate is scored across
paths on median return, downside (p5), probability of loss, probability of
ruin, average max drawdown, realized win rate, and Sharpe — then ranked by a
consistency-adjusted `composite` (or `--objective median_return|sharpe|calmar`).

**Two data modes:**
- **Synthetic** — an explicit channel model with knobs for the channel's real
  edge (`--channel-skill`), how much it inflates its quoted win rate
  (`--report-bias`), and price/noise. Outcomes are decided by the *true*
  probability, so a strategy that blindly trusts inflated win rates gets
  punished in the sim — which is the point.
- **Historical** (`--history`) — bootstrap-resample your own logged alerts
  with known outcomes for a model-free forward test. **This is the most
  trustworthy mode** — switch to it as soon as you've collected real alerts.

> ⚠️ Synthetic results are only as good as the channel model. Sample run with a
> skilled channel (`skill=0.03, bias=0.05, slippage=50bps`) favoured a 5%-of-
> bankroll, `win_rate≥0.55, EV≥5` strategy (~+23% median, 1% ruin); the same
> sweep against a *no-edge* channel correctly collapsed to a defensive
> `win_rate≥0.70, fixed 2%` strategy (~+2% median, 0% ruin). **Calibrate to
> your channel or use `--history` before sizing up.**

A logged-alerts file for `--history` is JSON/JSONL of:
```json
{"win_rate": 0.66, "entry_price": 0.45, "outcome": 1, "size": 80}
```
The bot already records every alert to SQLite; once positions are settled (via
`/settle` against real Polymarket resolution) you can export that table to feed
the backtest.

## Going live on Polymarket (real money)

Live execution is **off by default** and guarded by four independent gates so
it can't fire by accident:

1. `mode: live` in config (or `/mode live`)
2. `LIVE_TRADING=true` in the environment
3. Polymarket credentials present (`POLYMARKET_PRIVATE_KEY`, …)
4. `pip install py-clob-client`

The live executor matches the parsed market text to an active Polymarket
market via the public Gamma API, selects the YES/NO token from the parsed
side, and posts a limit order through the CLOB client. **Caveat:** mapping
freeform English to the exact on-chain market is inherently fuzzy — the
executor refuses rather than guesses when it can't confidently identify a
single market. Harden/verify that matching layer (and test with tiny size)
before trusting it. Start in `paper` mode, confirm `/stats` win rate matches
expectations, then graduate.

## Tests

```bash
python -m pytest tests/ -q
```

Covers parsing, filtering, sizing/Kelly, and the paper-trade + resolution flow.

## Extending

- **New venue?** Implement `Executor.place()` in `executors/`, register it in
  `executors/__init__.build_executor`.
- **New alert phrasing?** Add a label to the `*_PATTERNS` lists in `parser.py`
  and pin it with a test.
- **New filter?** Add the field to `FilterConfig`, a check in
  `FilterEngine.evaluate`, and an entry in `SETTABLE_FIELDS` to make it
  `/set`-able.
