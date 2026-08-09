#!/usr/bin/env python3
"""Synthetic backtest lab for the alert-free scanners.

Each strategy is run over many simulated markets/paths so we can see the edge
it carries *before* trusting it live. Like the alert-side Monte Carlo, the
numbers are only as good as the synthetic model — they show the mechanism and
its sensitivities, not a promise. Calibrate against real Polymarket history.

    python scanner_lab.py                 # run all four
    python scanner_lab.py --slippage-bps 100
"""

from __future__ import annotations

import argparse
import random
import statistics as st

from copytradebot.scanners import (
    ArbitrageScanner, LongshotScanner, MeanReversionScanner,
    CointegrationScanner)
from copytradebot.scanners.arbitrage import MarketSnapshot


def _summary(returns):
    if not returns:
        return {"trades": 0, "mean": 0.0, "win_rate": 0.0, "median": 0.0,
                "p5": 0.0}
    s = sorted(returns)
    wins = sum(1 for r in returns if r > 0)
    return {
        "trades": len(returns),
        "mean": st.mean(returns),
        "median": s[len(s) // 2],
        "p5": s[max(0, int(0.05 * len(s)) - 1)],
        "win_rate": wins / len(returns),
    }


def lab_arbitrage(seed, n=5000, buffer=0.01):
    rng = random.Random(seed)
    sc = ArbitrageScanner(cost_buffer=buffer, min_edge=0.002)
    edges, hits = [], 0
    for i in range(n):
        # Market with two outcomes; ask sum centered slightly above 1.
        s = rng.gauss(1.015, 0.02)
        ya = rng.uniform(0.2, 0.8) * s
        na = s - ya
        opps = sc.scan([MarketSnapshot(f"m{i}", "q", ["Yes", "No"], [ya, na])])
        if opps:
            hits += 1
            edges.append(opps[0].edge)
    return {"scanned": n, "arbs_found": hits,
            "hit_rate": hits / n, "avg_locked_edge": st.mean(edges) if edges else 0.0}


def lab_longshot(seed, n=20000, slip=0.005):
    rng = random.Random(seed)
    # spread=0 here: the lab models the fill cost itself via `slip` below, so
    # the scanner shouldn't also pad the price (avoid double-counting).
    sc = LongshotScanner(min_edge=0.005, spread=0.0)
    rets = []
    for i in range(n):
        p = rng.uniform(0.05, 0.95)
        true = min(0.99, max(0.01, p + 0.08 * (p - 0.5) + rng.gauss(0, 0.03)))
        outcome = 1 if rng.random() < true else 0
        opps = sc.scan([MarketSnapshot(f"m{i}", "q", ["Yes", "No"], [p, 1 - p])])
        for o in opps:
            leg = o.legs[0]
            entry = min(0.99, leg.price + slip)
            backing_yes = (leg.side == "Yes")
            won = (outcome == 1) if backing_yes else (outcome == 0)
            rets.append((1 / entry - 1) if won else -1.0)
    return _summary(rets)


def _ar1_series(rng, length, mean, phi, sigma, news_prob=0.0, news_size=0.0):
    s, x = [], mean
    news_at = rng.randint(length // 3, length - 5) if rng.random() < news_prob else -1
    shift = 0.0
    for t in range(length):
        if t == news_at:
            shift = rng.choice([-1, 1]) * news_size
        x = mean + shift + phi * (x - mean - shift) + rng.gauss(0, sigma)
        s.append(min(0.95, max(0.05, x)))
    return s


def lab_mean_reversion(seed, paths=3000, length=60, hold=5, slip=0.005,
                       news_prob=0.15):
    rng = random.Random(seed)
    sc = MeanReversionScanner(lookback=20, entry_z=2.0, max_move=0.25,
                              min_edge=0.005, spread=0.0)  # lab adds slip itself
    rets = []
    for _ in range(paths):
        prices = _ar1_series(rng, length, mean=rng.uniform(0.35, 0.65),
                             phi=0.6, sigma=0.03, news_prob=news_prob,
                             news_size=0.20)
        # Walk forward; enter at the first signal with room to exit.
        for t in range(21, length - hold):
            hist = {"m": {"question": "q", "prices": prices[:t + 1],
                          "token_yes": "ty", "token_no": "tn"}}
            opps = sc.scan(hist)
            if not opps:
                continue
            leg = opps[0].legs[0]
            entry = min(0.99, leg.price + slip)
            future_yes = prices[t + hold]
            exit_price = future_yes if leg.side == "Yes" else 1 - future_yes
            rets.append(exit_price / entry - 1)
            break
    return _summary(rets)


def lab_cointegration(seed, pairs=3000, length=80, hold=8, slip=0.005):
    rng = random.Random(seed)
    sc = CointegrationScanner(lookback=60, entry_z=2.0, adf_threshold=-2.5,
                              min_half_life=0.0, max_half_life=40, spread=0.0)
    rets = []
    for _ in range(pairs):
        # B random walk; spread mean-reverts; A = B + spread.
        b, x = [], 0.5
        for _ in range(length):
            x = min(0.9, max(0.1, x + rng.gauss(0, 0.012)))
            b.append(x)
        spread, sp = [], 0.0
        for _ in range(length):
            sp = 0.6 * sp + rng.gauss(0, 0.02)
            spread.append(sp)
        a = [min(0.95, max(0.05, b[i] + spread[i])) for i in range(length)]
        for t in range(60, length - hold):
            series = {
                "A": {"question": "A", "prices": a[:t + 1],
                      "token_yes": "ay", "token_no": "an"},
                "B": {"question": "B", "prices": b[:t + 1],
                      "token_yes": "by", "token_no": "bn"},
            }
            opps = sc.scan(series)
            if not opps:
                continue
            o = opps[0]
            tot_w = sum(l.weight for l in o.legs)
            pnl = 0.0
            for leg in o.legs:
                entry = min(0.99, leg.price + slip)
                price_now = a[t + hold] if leg.market_id == "A" else b[t + hold]
                exit_price = price_now if leg.side == "Yes" else 1 - price_now
                pnl += (leg.weight / tot_w) * (exit_price / entry - 1)
            rets.append(pnl)
            break
    return _summary(rets)


def main():
    ap = argparse.ArgumentParser(description="Scanner backtest lab")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--slippage", type=float, default=0.005,
                    help="per-leg entry slippage as a price add (0.005 = half a cent)")
    args = ap.parse_args()

    print("Alert-free strategy lab (synthetic). Edge = expected return per "
          "trade.\n")

    arb = lab_arbitrage(args.seed, buffer=max(0.01, args.slippage * 2))
    print(f"[arbitrage]     scanned {arb['scanned']}, found {arb['arbs_found']} "
          f"({arb['hit_rate']*100:.1f}%), avg locked edge "
          f"{arb['avg_locked_edge']*100:.2f}% (near-riskless when found)")

    for name, fn in [("longshot", lab_longshot),
                     ("meanrevert", lab_mean_reversion),
                     ("cointegr.", lab_cointegration)]:
        m = fn(args.seed, slip=args.slippage) if name != "longshot" \
            else fn(args.seed, slip=args.slippage)
        print(f"[{name:11}] trades {m['trades']:5d} | mean/trade "
              f"{m['mean']*100:+.2f}% | win {m['win_rate']*100:.0f}% | "
              f"median {m['median']*100:+.2f}% | p5 {m['p5']*100:+.1f}%")

    print("\nTakeaways:")
    print(" - Arbitrage: edge is mechanical and near-riskless, but rare and "
          "capacity-limited; needs fast execution before it's gone.")
    print(" - Longshot/MR/cointegration: edges are real-but-thin and depend on "
          "the model assumptions above; positive expectancy is a *volume* game "
          "and dies under high slippage.")
    print(" - Calibrate on real Polymarket history (and respect fees) before "
          "sizing up. These are diversifiers to the copy-trading bot, not a "
          "free lunch.")


if __name__ == "__main__":
    raise SystemExit(main())
