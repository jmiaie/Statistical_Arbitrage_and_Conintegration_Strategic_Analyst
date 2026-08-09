"""Paper (simulated) executor.

Records positions to storage with an assumed immediate fill at the alert's
entry price. Positions are settled later via ``resolve`` (manually through the
``/resolve`` Telegram command, or programmatically), at which point realized
P&L and the rolling win rate are updated.
"""

from __future__ import annotations

from .base import Executor
from ..models import Signal, Position
from ..storage import Storage


class PaperExecutor(Executor):
    name = "paper"

    def __init__(self, storage: Storage):
        self.storage = storage

    def place(self, signal: Signal, stake: float, signal_id: int | None = None) -> Position:
        pos = Position(
            signal_id=signal_id,
            market=signal.market,
            side=signal.side.value,
            entry_price=signal.entry_price,
            stake=stake,
            venue=self.name,
            status="open",
            external_id=f"paper-{signal_id or 'na'}",
        )
        pos_id = self.storage.record_position(pos)
        pos.external_id = f"paper-{pos_id}"
        return pos

    # ---- settlement helpers -------------------------------------------- #
    def resolve(self, pos_id: int, outcome: str) -> tuple[bool, float, str]:
        """Settle an open paper position.

        ``outcome`` may be:
          * ``win``  -> binary-market style payoff using entry price as the
                        implied probability: profit = stake*(1/price - 1).
          * ``loss`` -> lose the full stake.
          * a number -> an exit price; P&L = stake*(exit/entry - 1).

        Returns ``(ok, pnl, status)``.
        """
        row = self.storage.get_position(pos_id)
        if row is None or row["status"] != "open":
            return False, 0.0, "not-open"

        stake = float(row["stake"])
        entry = row["entry_price"]
        o = outcome.strip().lower()

        if o in {"win", "won", "w", "yes", "1"}:
            if entry and 0 < entry < 1:
                pnl = stake * ((1.0 / entry) - 1.0)
            else:
                pnl = stake  # default: assume even-money win if no price
            status = "won"
        elif o in {"loss", "lost", "l", "no", "0"}:
            pnl = -stake
            status = "lost"
        else:
            try:
                exit_price = float(o)
            except ValueError:
                return False, 0.0, "bad-outcome"
            if not entry:
                return False, 0.0, "no-entry-price"
            pnl = stake * ((exit_price / entry) - 1.0)
            status = "won" if pnl >= 0 else "lost"

        pnl = round(pnl, 2)
        self.storage.resolve_position(pos_id, status, pnl)
        return True, pnl, status
