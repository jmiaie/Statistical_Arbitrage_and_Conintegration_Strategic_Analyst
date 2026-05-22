"""quantpairs — dynamic statistical arbitrage research stack."""

from quantpairs.attribution import FamaFrenchAttribution
from quantpairs.backtest import BacktestResult, run_backtest
from quantpairs.cointegration import CointegrationResult, engle_granger_test, johansen_test
from quantpairs.kalman import KalmanHedge, KalmanState
from quantpairs.signals import ZScoreSignal, generate_signals
from quantpairs.tca import SquareRootImpact, apply_costs
from quantpairs.wfo import WalkForwardResult, walk_forward

__all__ = [
    "BacktestResult",
    "CointegrationResult",
    "FamaFrenchAttribution",
    "KalmanHedge",
    "KalmanState",
    "SquareRootImpact",
    "WalkForwardResult",
    "ZScoreSignal",
    "apply_costs",
    "engle_granger_test",
    "generate_signals",
    "johansen_test",
    "run_backtest",
    "walk_forward",
]

__version__ = "0.1.0"
