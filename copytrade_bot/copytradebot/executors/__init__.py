"""Pluggable trade executors."""

from .base import Executor, ExecutionError
from .paper import PaperExecutor
from .realistic_paper import RealisticPaperExecutor

__all__ = [
    "Executor", "ExecutionError", "PaperExecutor", "RealisticPaperExecutor",
    "build_executor", "build_paper_executor",
]


def build_paper_executor(config, settings, storage):
    """Construct the paper executor for the active execution config.

    With the default ``execution.fill_model == 'alert'`` and no live data this
    returns the lightweight :class:`PaperExecutor`. Any realistic setting
    (mid/book fills, slippage against live data) returns
    :class:`RealisticPaperExecutor`.
    """
    from ..slippage import SlippageModel
    from ..marketdata import build_provider

    ex = config.execution
    if ex.fill_model == "alert" and ex.data_source == "none":
        return PaperExecutor(storage)

    provider = build_provider(ex, settings)
    slippage = SlippageModel(slippage_bps=ex.slippage_bps, fee_bps=ex.fee_bps)
    return RealisticPaperExecutor(storage, provider, slippage, ex)


def build_executor(config, settings, storage):
    """Factory: return the executor for the active strategy mode."""
    if (config.mode or "paper").lower() == "live":
        from .polymarket import PolymarketExecutor
        return PolymarketExecutor(settings, storage)
    return build_paper_executor(config, settings, storage)
