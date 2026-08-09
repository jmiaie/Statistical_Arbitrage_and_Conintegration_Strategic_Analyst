"""Alert-free strategies: scanners that generate trade opportunities from
market data instead of from a Telegram channel.

Each scanner consumes market data (snapshots and/or price history) and emits
:class:`Opportunity` objects. Single-leg opportunities (longshot, mean
reversion) can convert to a :class:`~copytradebot.models.Signal` and reuse the
existing filter/sizing/execution pipeline; multi-leg ones (arbitrage,
cointegration) carry their legs explicitly.

These complement the copy-trading bot: they don't need anyone's alerts, and
the cointegration scanner is the repo's stat-arb theme applied to prediction
markets.
"""

from .base import Leg, Opportunity, Scanner
from .arbitrage import ArbitrageScanner
from .longshot import LongshotScanner
from .meanreversion import MeanReversionScanner
from .cointegration import CointegrationScanner

__all__ = [
    "Leg", "Opportunity", "Scanner",
    "ArbitrageScanner", "LongshotScanner", "MeanReversionScanner",
    "CointegrationScanner",
]
