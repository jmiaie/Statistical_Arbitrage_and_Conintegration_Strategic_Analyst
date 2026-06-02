"""Position sizing strategies."""

from __future__ import annotations

from .config import SizingConfig
from .models import Signal


def kelly_fraction(win_rate: float, price: float) -> float:
    """Kelly-optimal fraction of bankroll for a binary bet bought at ``price``
    (implied probability in (0, 1)). Net odds ``b = (1/price) - 1``.

        f* = (b*p - (1-p)) / b = p - (1-p)/b
    """
    if not (0 < price < 1) or not (0 <= win_rate <= 1):
        return 0.0
    b = (1.0 / price) - 1.0
    if b <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / b
    return max(0.0, f)


def compute_stake(signal: Signal, cfg: SizingConfig) -> float:
    """Return the stake (currency units), clamped to [min_position, max_position]."""
    mode = (cfg.mode or "fraction").lower()

    if mode == "fixed":
        stake = cfg.fixed_amount
    elif mode == "kelly":
        wr = signal.win_rate if signal.win_rate is not None else 0.0
        price = signal.entry_price if signal.entry_price is not None else 0.0
        f = kelly_fraction(wr, price) * cfg.kelly_fraction
        stake = cfg.bankroll * f
        # If the signal isn't a 0-1 priced market, Kelly is undefined -> fall
        # back to the configured fraction so we still size something sane.
        if stake <= 0:
            stake = cfg.bankroll * cfg.fraction
    else:  # fraction
        stake = cfg.bankroll * cfg.fraction

    # Respect an explicit, smaller suggested size from the alert when present.
    if signal.size is not None and signal.size > 0:
        stake = min(stake, signal.size)

    stake = max(cfg.min_position, min(stake, cfg.max_position))
    return round(stake, 2)
