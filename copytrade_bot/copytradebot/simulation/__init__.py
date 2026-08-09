"""Monte Carlo strategy simulation & optimisation.

Find the most profitable *and* consistent filter/sizing strategy before
risking anything, then export it straight into the bot's config.

Two data modes:

* **Synthetic** (default) — generate alert streams from an explicit channel
  model (its real skill/edge, how much it inflates its quoted win rate, the
  market price distribution). Every assumption is a knob, so you can stress
  the strategy under pessimistic and optimistic worlds.
* **Historical** — bootstrap-resample your own past alerts (with known
  outcomes) for a model-free forward test. This is the gold standard; use it
  as soon as you have logged alerts.

The same FilterEngine, sizing and slippage code the live bot uses is reused
here, so what you optimise is what you deploy.
"""

from .generator import Scenario, SimAlert, generate_paths, load_history
from .engine import simulate_path, aggregate, evaluate_strategy
from .optimize import build_grid, run_optimization, rank, format_report

__all__ = [
    "Scenario", "SimAlert", "generate_paths", "load_history",
    "simulate_path", "aggregate", "evaluate_strategy",
    "build_grid", "run_optimization", "rank", "format_report",
]
