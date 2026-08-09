"""Strategy grid construction, ranking, and reporting."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (StrategyConfig, FilterConfig, SizingConfig, RiskConfig,
                      ExecutionConfig)
from .engine import evaluate_strategy
from .generator import SimAlert


@dataclass
class StrategyResult:
    name: str
    config: StrategyConfig
    metrics: dict


def build_grid(bankroll: float, slippage_bps: float = 50.0,
               fee_bps: float = 0.0,
               win_rates=(0.55, 0.60, 0.65, 0.70),
               min_evs=(0.0, 5.0)) -> list[tuple[str, StrategyConfig]]:
    """Cartesian grid of filter thresholds x sizing methods to evaluate."""
    # Sizing methods. max_position is set high so the method's character (not an
    # arbitrary cap) drives results; min_position small. Tune caps for live use.
    sizers = [
        ("fixed2%", SizingConfig(mode="fixed", fixed_amount=0.02 * bankroll)),
        ("frac2%", SizingConfig(mode="fraction", fraction=0.02)),
        ("frac5%", SizingConfig(mode="fraction", fraction=0.05)),
        ("kelly0.25", SizingConfig(mode="kelly", kelly_fraction=0.25, fraction=0.02)),
        ("kelly0.5", SizingConfig(mode="kelly", kelly_fraction=0.50, fraction=0.02)),
    ]
    execu = ExecutionConfig(slippage_bps=slippage_bps, fee_bps=fee_bps)

    grid = []
    for wr in win_rates:
        for ev in min_evs:
            for sname, sizer in sizers:
                sizer = SizingConfig(**{**sizer.__dict__})
                sizer.bankroll = bankroll
                sizer.max_position = bankroll
                sizer.min_position = 1.0
                cfg = StrategyConfig(
                    mode="paper",
                    filters=FilterConfig(min_win_rate=wr, min_ev=ev,
                                         require_fields=["win_rate"]),
                    sizing=sizer,
                    risk=RiskConfig(),
                    execution=ExecutionConfig(**execu.__dict__),
                )
                name = f"wr>={wr:.2f} ev>={ev:g} {sname}"
                grid.append((name, cfg))
    return grid


def run_optimization(grid, paths: list[list[SimAlert]], bankroll: float,
                     ruin_fraction: float = 0.5) -> list[StrategyResult]:
    out = []
    for name, cfg in grid:
        metrics = evaluate_strategy(cfg, paths, bankroll, ruin_fraction)
        out.append(StrategyResult(name, cfg, metrics))
    return out


def rank(results: list[StrategyResult], objective: str = "composite",
         max_prob_ruin: float = 0.05, min_avg_bets: float = 3.0
         ) -> list[StrategyResult]:
    """Rank by objective after filtering out reckless / inactive strategies."""
    eligible = [r for r in results
                if r.metrics["prob_ruin"] <= max_prob_ruin
                and r.metrics["avg_bets"] >= min_avg_bets]
    pool = eligible or results  # fall back so we always return something
    return sorted(pool, key=lambda r: r.metrics.get(objective, 0.0), reverse=True)


def format_report(ranked: list[StrategyResult], top: int = 10,
                  objective: str = "composite") -> str:
    header = (f"{'strategy':32} {'med%':>7} {'mean%':>7} {'p5%':>7} "
              f"{'pLoss':>6} {'pRuin':>6} {'maxDD':>6} {'bets':>5} "
              f"{'wr%':>5} {'sharpe':>7}")
    lines = [f"Ranked by {objective} (most profitable + consistent first)\n",
             header, "-" * len(header)]
    for r in ranked[:top]:
        m = r.metrics
        lines.append(
            f"{r.name:32} {m['median_return']*100:7.1f} "
            f"{m['mean_return']*100:7.1f} {m['p5_return']*100:7.1f} "
            f"{m['prob_loss']*100:6.0f} {m['prob_ruin']*100:6.0f} "
            f"{m['avg_max_drawdown']*100:6.0f} {m['avg_bets']:5.0f} "
            f"{m['avg_win_rate']*100:5.0f} {m['sharpe']:7.2f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Robustness: rank strategies by worst-case behaviour across a scenario panel
# --------------------------------------------------------------------------- #
@dataclass
class RobustResult:
    name: str
    config: StrategyConfig
    per_scenario: dict        # scenario_name -> metrics
    worst_composite: float
    mean_composite: float
    worst_median: float
    worst_prob_ruin: float


def run_robustness(grid, scenario_paths: dict, bankroll: float,
                   ruin_fraction: float = 0.5) -> list[RobustResult]:
    """Evaluate every strategy across a panel of scenarios.

    ``scenario_paths``: {scenario_name: paths}. Returns one RobustResult per
    strategy summarising its *worst-case* and average behaviour across worlds.
    """
    out = []
    for name, cfg in grid:
        per = {sc: evaluate_strategy(cfg, paths, bankroll, ruin_fraction)
               for sc, paths in scenario_paths.items()}
        comps = [m["composite"] for m in per.values()]
        out.append(RobustResult(
            name=name, config=cfg, per_scenario=per,
            worst_composite=min(comps),
            mean_composite=sum(comps) / len(comps),
            worst_median=min(m["median_return"] for m in per.values()),
            worst_prob_ruin=max(m["prob_ruin"] for m in per.values()),
        ))
    return out


def rank_robust(results: list[RobustResult], max_prob_ruin: float = 0.05,
                min_avg_bets: float = 3.0) -> list[RobustResult]:
    def active(r: RobustResult) -> bool:
        return all(m["avg_bets"] >= min_avg_bets for m in r.per_scenario.values())
    eligible = [r for r in results
                if r.worst_prob_ruin <= max_prob_ruin and active(r)]
    pool = eligible or results
    # Worst-case composite first (max-min): be good even in the bad world.
    return sorted(pool, key=lambda r: (r.worst_composite, r.mean_composite),
                  reverse=True)


def format_robust_report(ranked: list[RobustResult], scenarios: list[str],
                         top: int = 10) -> str:
    cols = "".join(f"{sc[:10]:>11}" for sc in scenarios)
    header = f"{'strategy':30}{cols} {'worstMed%':>10} {'worstRuin%':>11}"
    lines = ["Ranked by worst-case composite across scenarios "
             "(median return per scenario shown)\n", header, "-" * len(header)]
    for r in ranked[:top]:
        cells = "".join(f"{r.per_scenario[sc]['median_return']*100:>10.1f} "
                        for sc in scenarios)
        lines.append(f"{r.name:30}{cells}{r.worst_median*100:>10.1f} "
                     f"{r.worst_prob_ruin*100:>10.0f}")
    return "\n".join(lines)

