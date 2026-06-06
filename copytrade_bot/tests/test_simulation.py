import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copytradebot.config import (StrategyConfig, FilterConfig, SizingConfig,
                                 ExecutionConfig)
from copytradebot.simulation.generator import Scenario, generate_paths, SimAlert
from copytradebot.simulation.engine import simulate_path, evaluate_strategy
from copytradebot.simulation.optimize import build_grid, run_optimization, rank


def _strategy(min_wr=0.55, mode="fraction"):
    return StrategyConfig(
        mode="paper",
        filters=FilterConfig(min_win_rate=min_wr, min_ev=0.0),
        sizing=SizingConfig(mode=mode, bankroll=1000, fraction=0.05,
                            max_position=1000, min_position=1),
        execution=ExecutionConfig(slippage_bps=0, fee_bps=0),
    )


def test_generate_paths_reproducible():
    sc = Scenario()
    a = generate_paths(sc, 5, 10, seed=1)
    b = generate_paths(sc, 5, 10, seed=1)
    assert a[0][0].price == b[0][0].price
    assert a[0][0].win_rate == b[0][0].win_rate


def test_all_winners_make_money():
    # Every alert wins -> bankroll must grow.
    alerts = [SimAlert(price=0.5, win_rate=0.9, ev=80, roi=80,
                       true_prob=1.0, outcome_uniform=0.0) for _ in range(20)]
    res = simulate_path(_strategy(), alerts, start_bankroll=1000)
    assert res["return"] > 0
    assert res["win_rate"] == 1.0
    assert res["bets"] == 20


def test_all_losers_lose_money_and_can_ruin():
    alerts = [SimAlert(price=0.5, win_rate=0.9, ev=80, roi=80,
                       true_prob=0.0, outcome_uniform=0.99) for _ in range(50)]
    res = simulate_path(_strategy(mode="fraction"), alerts, start_bankroll=1000)
    assert res["return"] < 0
    assert res["win_rate"] == 0.0


def test_skilled_channel_beats_no_skill():
    paths_skill = generate_paths(Scenario(mean_edge=0.08, report_bias=0.0),
                                 300, 60, seed=7)
    paths_none = generate_paths(Scenario(mean_edge=-0.02, report_bias=0.0),
                                300, 60, seed=7)
    strat = _strategy(min_wr=0.55)
    skilled = evaluate_strategy(strat, paths_skill, 1000)
    noskill = evaluate_strategy(strat, paths_none, 1000)
    assert skilled["median_return"] > noskill["median_return"]


def test_optimizer_runs_and_ranks():
    paths = generate_paths(Scenario(mean_edge=0.06), 200, 50, seed=3)
    grid = build_grid(1000, slippage_bps=0, win_rates=(0.55, 0.65),
                      min_evs=(0.0,))
    results = run_optimization(grid, paths, 1000)
    # With caps relaxed, every strategy is eligible.
    ranked = rank(results, objective="composite", min_avg_bets=0.0,
                  max_prob_ruin=1.0)
    assert len(ranked) == len(results)
    # Ranking is descending by composite.
    comps = [r.metrics["composite"] for r in ranked]
    assert comps == sorted(comps, reverse=True)


def test_no_edge_channel_is_not_wildly_profitable():
    # Efficient market (price = true prob on average) minus slippage should
    # NOT produce a strongly positive median once costs bite.
    paths = generate_paths(Scenario(mean_edge=0.0, report_bias=0.05), 400, 80,
                           seed=11)
    grid = build_grid(1000, slippage_bps=100, win_rates=(0.6,), min_evs=(0.0,))
    results = run_optimization(grid, paths, 1000)
    best = rank(results, min_avg_bets=0.0)[0]
    # With no real edge and costs, the best median return should be modest.
    assert best.metrics["median_return"] < 0.5
