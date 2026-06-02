"""Pluggable trade executors."""

from .base import Executor, ExecutionError
from .paper import PaperExecutor

__all__ = ["Executor", "ExecutionError", "PaperExecutor", "build_executor"]


def build_executor(mode: str, settings, storage):
    """Factory: return the executor for the active strategy mode.

    ``paper`` -> simulated fills. ``live`` -> Polymarket (imported lazily so
    the optional ``py-clob-client`` dependency is only needed for live mode).
    """
    mode = (mode or "paper").lower()
    if mode == "live":
        from .polymarket import PolymarketExecutor
        return PolymarketExecutor(settings, storage)
    return PaperExecutor(storage)
