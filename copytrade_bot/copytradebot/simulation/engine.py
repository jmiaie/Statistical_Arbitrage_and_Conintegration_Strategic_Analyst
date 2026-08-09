"""Monte Carlo engine — run a strategy over alert paths and score it."""

from __future__ import annotations

import math
from dataclasses import replace

from ..config import StrategyConfig
from ..filters import FilterEngine
from ..sizing import compute_stake
from ..slippage import SlippageModel
from .generator import SimAlert


def simulate_path(strategy: StrategyConfig, alerts: list[SimAlert],
                  start_bankroll: float, ruin_fraction: float = 0.5) -> dict:
    """Walk one alert stream with compounding bankroll. Returns path metrics."""
    engine = FilterEngine(strategy.filters)
    slip = SlippageModel(strategy.execution.slippage_bps,
                         strategy.execution.fee_bps)
    sizing = replace(strategy.sizing)  # mutable copy; bankroll updated per bet

    equity = start_bankroll
    peak = start_bankroll
    ruin_level = start_bankroll * ruin_fraction
    max_dd = 0.0
    bets = wins = 0
    ruined = False

    for a in alerts:
        if equity <= ruin_level:
            ruined = True
            break
        sig = a.to_signal()
        if not engine.evaluate(sig).passed:
            continue
        sizing.bankroll = equity
        stake = min(compute_stake(sig, sizing), equity)
        if stake <= 0:
            continue
        fill = slip.buy_at_price(stake, a.price)
        if a.won():
            pnl = fill.shares - fill.filled_stake
            wins += 1
        else:
            pnl = -fill.filled_stake
        equity += pnl
        bets += 1
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)

    return {
        "return": equity / start_bankroll - 1.0,
        "final": equity,
        "bets": bets,
        "win_rate": wins / bets if bets else 0.0,
        "max_drawdown": max_dd,
        "ruined": ruined or equity <= ruin_level,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def aggregate(paths: list[dict]) -> dict:
    """Aggregate per-path results into a strategy scorecard."""
    n = len(paths)
    rets = [p["return"] for p in paths]
    dds = [p["max_drawdown"] for p in paths]
    bets = [p["bets"] for p in paths]
    wrs = [p["win_rate"] for p in paths if p["bets"]]
    mean = sum(rets) / n if n else 0.0
    var = sum((r - mean) ** 2 for r in rets) / n if n else 0.0
    std = math.sqrt(var)
    median = _percentile(rets, 0.5)
    avg_dd = sum(dds) / n if n else 0.0
    prob_loss = sum(1 for r in rets if r < 0) / n if n else 0.0
    prob_ruin = sum(1 for p in paths if p["ruined"]) / n if n else 0.0
    avg_bets = sum(bets) / n if n else 0.0

    # Risk-adjusted scores. Calmar-like = consistency-aware profitability.
    sharpe = mean / std if std > 1e-9 else 0.0
    calmar = median / (avg_dd + 0.05)
    # Composite favours strategies that are profitable AND rarely lose money.
    composite = median * (1 - prob_loss) - 0.5 * avg_dd

    return {
        "mean_return": mean,
        "median_return": median,
        "std_return": std,
        "p5_return": _percentile(rets, 0.05),
        "p95_return": _percentile(rets, 0.95),
        "prob_loss": prob_loss,
        "prob_ruin": prob_ruin,
        "avg_max_drawdown": avg_dd,
        "avg_bets": avg_bets,
        "avg_win_rate": (sum(wrs) / len(wrs)) if wrs else 0.0,
        "sharpe": sharpe,
        "calmar": calmar,
        "composite": composite,
    }


def evaluate_strategy(strategy: StrategyConfig, paths: list[list[SimAlert]],
                      start_bankroll: float, ruin_fraction: float = 0.5) -> dict:
    results = [simulate_path(strategy, p, start_bankroll, ruin_fraction)
               for p in paths]
    return aggregate(results)
