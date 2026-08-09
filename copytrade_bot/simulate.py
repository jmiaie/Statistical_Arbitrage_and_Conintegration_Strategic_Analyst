#!/usr/bin/env python3
"""Monte Carlo strategy finder.

Sweeps a grid of filter + sizing strategies over many simulated alert streams
and reports which is the most profitable *and* consistent, then optionally
writes the winner into the bot config so you can deploy it immediately.

Examples:
    python simulate.py                       # synthetic, default channel model
    python simulate.py --quick               # fast, smaller sweep
    python simulate.py --channel-skill 0.0   # stress: a channel with NO edge
    python simulate.py --report-bias 0.10    # channel inflates win rate +10pp
    python simulate.py --history alerts.jsonl # backtest on YOUR real alerts
    python simulate.py --save-best           # write winner to config/filters.yaml
"""

from __future__ import annotations

import argparse

from copytradebot.config import StrategyConfig
from copytradebot.simulation.generator import (
    Scenario, generate_paths, load_history, bootstrap_paths)
from copytradebot.simulation.optimize import (
    build_grid, run_optimization, rank, format_report,
    run_robustness, rank_robust, format_robust_report)


# Scenario panel for --robust: pessimistic -> optimistic channel quality.
ROBUST_PANEL = {
    "no-edge":  dict(mean_edge=0.00, report_bias=0.08, report_noise=0.10),
    "weak":     dict(mean_edge=0.02, report_bias=0.06, report_noise=0.08),
    "base":     dict(mean_edge=0.03, report_bias=0.05, report_noise=0.08),
    "strong":   dict(mean_edge=0.06, report_bias=0.04, report_noise=0.06),
}


def run_robust_mode(args) -> int:
    panel = {}
    for name, kw in ROBUST_PANEL.items():
        panel[name] = generate_paths(Scenario(**kw), args.paths, args.alerts,
                                     args.seed)
    grid = build_grid(args.bankroll, args.slippage_bps, args.fee_bps)
    print(f"Robustness sweep: {len(grid)} strategies x {len(panel)} scenarios "
          f"x {args.paths} paths x {args.alerts} alerts")
    print(f"Costs: slippage={args.slippage_bps}bps fee={args.fee_bps}bps | "
          f"bankroll={args.bankroll:g}\n")

    results = run_robustness(grid, panel, args.bankroll, args.ruin_fraction)
    ranked = rank_robust(results)
    print(format_robust_report(ranked, list(panel), top=args.top))

    best = ranked[0]
    print("\n" + "=" * 64)
    print(f"MOST ROBUST (best worst-case): {best.name}")
    print("=" * 64)
    for sc in panel:
        m = best.per_scenario[sc]
        print(f"  [{sc:8}] median {m['median_return']*100:+6.1f}%  "
              f"p5 {m['p5_return']*100:+6.1f}%  pLoss {m['prob_loss']*100:3.0f}%  "
              f"pRuin {m['prob_ruin']*100:3.0f}%  maxDD {m['avg_max_drawdown']*100:3.0f}%")
    print(f"\n  Worst-case median across scenarios: {best.worst_median*100:+.1f}%")
    print(f"  Worst-case prob of ruin:            {best.worst_prob_ruin*100:.0f}%")

    if args.save_best:
        best.config.save(args.config) if args.config else best.config.save()
        print(f"\n✅ Wrote most-robust strategy to config "
              f"({args.config or 'config/filters.yaml'}). Review before live.")
    else:
        print("\n(Use --save-best to write this strategy into the bot config.)")
    print("\nNOTE: this is the strategy that holds up across the scenario "
          "panel.\nFeed real logged alerts with --history once available to "
          "confirm.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Monte Carlo strategy optimiser")
    ap.add_argument("--paths", type=int, default=2000, help="Monte Carlo paths")
    ap.add_argument("--alerts", type=int, default=150, help="alerts per path")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--objective", default="composite",
                    choices=["composite", "median_return", "sharpe", "calmar",
                             "mean_return"])
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--quick", action="store_true",
                    help="smaller, faster sweep")
    ap.add_argument("--robust", action="store_true",
                    help="rank by worst-case across a panel of channel scenarios")
    # Channel model (synthetic mode)
    ap.add_argument("--channel-skill", type=float, default=0.03,
                    help="mean true edge over market price (0 = no skill)")
    ap.add_argument("--edge-std", type=float, default=0.05)
    ap.add_argument("--report-bias", type=float, default=0.05,
                    help="how much the channel overstates win rate (fraction)")
    ap.add_argument("--report-noise", type=float, default=0.08)
    # Execution realism
    ap.add_argument("--slippage-bps", type=float, default=50.0)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--ruin-fraction", type=float, default=0.5,
                    help="bankroll fraction counted as 'ruin'")
    # Data / output
    ap.add_argument("--history", help="JSON/JSONL of real past alerts to bootstrap")
    ap.add_argument("--save-best", action="store_true",
                    help="write the winning strategy to the bot config")
    ap.add_argument("--config", default=None, help="config path for --save-best")
    args = ap.parse_args()

    if args.quick:
        args.paths = min(args.paths, 400)
        args.alerts = min(args.alerts, 80)

    if args.robust and not args.history:
        return run_robust_mode(args)

    # Build alert paths.
    if args.history:
        history = load_history(args.history)
        paths = bootstrap_paths(history, args.paths, args.alerts, args.seed)
        source_desc = f"bootstrap of {len(history)} real alerts ({args.history})"
    else:
        scenario = Scenario(
            mean_edge=args.channel_skill, edge_std=args.edge_std,
            report_bias=args.report_bias, report_noise=args.report_noise,
        )
        paths = generate_paths(scenario, args.paths, args.alerts, args.seed)
        source_desc = (f"synthetic: skill={args.channel_skill} "
                       f"bias={args.report_bias} noise={args.report_noise}")

    grid = build_grid(args.bankroll, args.slippage_bps, args.fee_bps)
    if args.quick:
        grid = build_grid(args.bankroll, args.slippage_bps, args.fee_bps,
                          win_rates=(0.55, 0.65), min_evs=(0.0,))

    print(f"Running {len(grid)} strategies x {args.paths} paths x "
          f"{args.alerts} alerts")
    print(f"Data: {source_desc}")
    print(f"Costs: slippage={args.slippage_bps}bps fee={args.fee_bps}bps | "
          f"bankroll={args.bankroll:g}\n")

    results = run_optimization(grid, paths, args.bankroll, args.ruin_fraction)
    ranked = rank(results, objective=args.objective)

    print(format_report(ranked, top=args.top, objective=args.objective))

    best = ranked[0]
    m = best.metrics
    print("\n" + "=" * 60)
    print(f"RECOMMENDED: {best.name}")
    print("=" * 60)
    print(f"  Median return:   {m['median_return']*100:+.1f}%  "
          f"(mean {m['mean_return']*100:+.1f}%)")
    print(f"  Downside (p5):   {m['p5_return']*100:+.1f}%   "
          f"prob of loss {m['prob_loss']*100:.0f}%   "
          f"prob of ruin {m['prob_ruin']*100:.0f}%")
    print(f"  Avg max drawdown:{m['avg_max_drawdown']*100:.0f}%   "
          f"avg bets {m['avg_bets']:.0f}   "
          f"realized win rate {m['avg_win_rate']*100:.0f}%")
    print(f"  Sharpe {m['sharpe']:.2f} | Calmar {m['calmar']:.2f}")

    if args.save_best:
        path = args.config
        if path:
            best.config.save(path)
        else:
            best.config.save()
        print(f"\n✅ Wrote winning strategy to config "
              f"({path or 'config/filters.yaml'}). Review before going live.")
    else:
        print("\n(Use --save-best to write this strategy into the bot config.)")

    print("\nNOTE: synthetic results are only as good as the channel model "
          "above.\nCalibrate --channel-skill/--report-bias to your channel, or "
          "feed real\nlogged alerts with --history for a model-free backtest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
