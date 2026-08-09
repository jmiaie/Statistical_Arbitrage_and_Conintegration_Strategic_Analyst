"""Alert-stream generation for Monte Carlo simulation.

Synthetic model (per alert):

  price        ~ Uniform(price_min, price_max)        # market implied prob
  edge         ~ Normal(mean_edge, edge_std)          # channel's real skill
  true_prob    = clip(price + edge)                   # actual win probability
  quoted_wr    = clip(true_prob + report_bias + Normal(0, report_noise))
  quoted_ev    = (quoted_wr*(1/price - 1) - (1-quoted_wr)) * 100

The key levers for realism:
  * ``mean_edge``  — does the channel actually beat the market? (0 = no skill)
  * ``report_bias``— does it overstate its win rate? (e.g. +0.05 = +5pp)
  * ``report_noise``— how noisy is the quoted win rate vs. the truth?

Outcomes are decided by ``true_prob`` (not the quoted number), so a strategy
that naively trusts inflated win rates will look good on paper and bleed in
the sim — exactly what we want to catch.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional

from ..models import Signal, Side


def _clip(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


@dataclass
class SimAlert:
    price: float                 # entry / implied probability
    win_rate: float              # quoted, fraction 0-1
    ev: float                    # quoted, percent
    roi: float                   # quoted, percent
    true_prob: float             # actual win probability (hidden from strategy)
    outcome_uniform: float       # pre-drawn U(0,1) to decide the outcome
    size: Optional[float] = None
    side: str = "YES"
    outcome: Optional[int] = None  # set for historical data (1=win, 0=loss)

    def won(self) -> bool:
        if self.outcome is not None:
            return self.outcome == 1
        return self.outcome_uniform < self.true_prob

    def to_signal(self) -> Signal:
        return Signal(
            raw_text="", source="sim", market="sim",
            side=Side.from_text(self.side),
            win_rate=self.win_rate, ev=self.ev, roi=self.roi,
            entry_price=self.price, size=self.size,
        )


@dataclass
class Scenario:
    price_min: float = 0.20
    price_max: float = 0.80
    mean_edge: float = 0.03       # average true edge over the market price
    edge_std: float = 0.05
    report_bias: float = 0.05     # channel overstates win rate by this much
    report_noise: float = 0.08
    size_suggestion: Optional[float] = None

    def draw(self, rng: random.Random) -> SimAlert:
        price = rng.uniform(self.price_min, self.price_max)
        edge = rng.gauss(self.mean_edge, self.edge_std)
        true_prob = _clip(price + edge)
        wr = _clip(true_prob + rng.gauss(self.report_bias, self.report_noise))
        ev = (wr * (1.0 / price - 1.0) - (1 - wr)) * 100
        return SimAlert(
            price=round(price, 4), win_rate=round(wr, 4),
            ev=round(ev, 2), roi=round(ev, 2),
            true_prob=true_prob, outcome_uniform=rng.random(),
            size=self.size_suggestion,
        )


def generate_paths(scenario: Scenario, n_paths: int, n_alerts: int,
                   seed: int = 12345) -> list[list[SimAlert]]:
    """Generate ``n_paths`` independent alert streams of ``n_alerts`` each.

    Each path is seeded deterministically (``seed + path_index``) so that every
    candidate strategy is evaluated on the *same* alert streams — common random
    numbers, which sharply reduces comparison variance.
    """
    paths = []
    for i in range(n_paths):
        rng = random.Random(seed + i)
        paths.append([scenario.draw(rng) for _ in range(n_alerts)])
    return paths


def load_history(path: str) -> list[SimAlert]:
    """Load real past alerts for bootstrap backtesting.

    Accepts JSON Lines or a JSON array of objects with at least::

        {"win_rate": 0.66, "entry_price": 0.45, "outcome": 1, "size": 80}

    ``win_rate`` may be a percent or fraction; ``outcome`` is 1 (win)/0 (loss).
    """
    with open(path) as fh:
        text = fh.read().strip()
    records = (json.loads(text) if text.startswith("[")
               else [json.loads(ln) for ln in text.splitlines() if ln.strip()])

    alerts = []
    for r in records:
        wr = r.get("win_rate")
        if wr is not None and wr > 1:
            wr = wr / 100.0
        price = r.get("entry_price") or r.get("price")
        ev = r.get("ev")
        if ev is None and wr is not None and price:
            ev = (wr * (1.0 / price - 1.0) - (1 - wr)) * 100
        alerts.append(SimAlert(
            price=price or 0.5, win_rate=wr if wr is not None else 0.5,
            ev=ev if ev is not None else 0.0,
            roi=r.get("roi", ev if ev is not None else 0.0),
            true_prob=float(r.get("outcome", 0)),
            outcome_uniform=0.0, size=r.get("size"),
            side=r.get("side", "YES"),
            outcome=int(r["outcome"]) if "outcome" in r else None,
        ))
    return alerts


def bootstrap_paths(history: list[SimAlert], n_paths: int, n_alerts: int,
                    seed: int = 12345) -> list[list[SimAlert]]:
    """Resample (with replacement) real alerts into Monte Carlo paths."""
    paths = []
    for i in range(n_paths):
        rng = random.Random(seed + i)
        paths.append([rng.choice(history) for _ in range(n_alerts)])
    return paths
