"""quantpairs — dynamic statistical arbitrage research stack."""

from quantpairs.adaptive import FitResult, fit_kalman_mle
from quantpairs.attribution import FamaFrenchAttribution
from quantpairs.backtest import BacktestResult, run_backtest
from quantpairs.benchmarks import compare_to_kalman, static_ols_backtest
from quantpairs.cointegration import CointegrationResult, engle_granger_test, johansen_test
from quantpairs.kalman import KalmanHedge, KalmanState
from quantpairs.portfolio import PortfolioResult, run_portfolio
from quantpairs.regime import apply_regime_filter, regime_mask
from quantpairs.signals import ZScoreSignal, generate_signals
from quantpairs.sizing import kelly_fraction, vol_target
from quantpairs.stats import BootstrapCI, sharpe_ci
from quantpairs.tca import SquareRootImpact, apply_costs
from quantpairs.tearsheet import write_tearsheet
from quantpairs.tracking import RunManifest, list_runs, log_run
from quantpairs.wfo import WalkForwardResult, walk_forward

__all__ = [
    "BacktestResult",
    "BootstrapCI",
    "CointegrationResult",
    "FamaFrenchAttribution",
    "FitResult",
    "KalmanHedge",
    "KalmanState",
    "PortfolioResult",
    "RunManifest",
    "SquareRootImpact",
    "WalkForwardResult",
    "ZScoreSignal",
    "apply_costs",
    "apply_regime_filter",
    "compare_to_kalman",
    "engle_granger_test",
    "fit_kalman_mle",
    "generate_signals",
    "johansen_test",
    "kelly_fraction",
    "list_runs",
    "log_run",
    "regime_mask",
    "run_backtest",
    "run_portfolio",
    "sharpe_ci",
    "static_ols_backtest",
    "vol_target",
    "walk_forward",
    "write_tearsheet",
]

__version__ = "0.3.0"
