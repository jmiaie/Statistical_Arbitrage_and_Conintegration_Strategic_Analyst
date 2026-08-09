"""Executor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Signal, Position


class ExecutionError(RuntimeError):
    """Raised when an order cannot be placed."""


class Executor(ABC):
    """Routes a sized signal to a venue and returns the resulting Position."""

    name: str = "base"

    @abstractmethod
    def place(self, signal: Signal, stake: float, signal_id: int | None = None) -> Position:
        ...
