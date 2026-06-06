"""Realistic paper executor.

Unlike the naive paper executor (immediate fill at the alert's quoted price),
this one:

  * resolves the alert to a real Polymarket market/token via a data provider,
  * fills against the **live order book** (size-aware) or a live midpoint,
  * applies **slippage and fees**, recording the effective fill price,
  * later **auto-settles** against the market's real on-chain resolution and
    can **mark open positions to market** for live unrealized P&L.

The goal: paper results that track what live execution would actually have
produced, so the win-rate / EV / ROI you verify are trustworthy. It needs a
provider with network access; offline it degrades gracefully to the alert
price (flagged in ``meta``) unless ``require_live_market`` is set.
"""

from __future__ import annotations

from .base import Executor, ExecutionError
from ..config import ExecutionConfig
from ..marketdata import MarketDataProvider
from ..models import Signal, Position
from ..slippage import SlippageModel
from ..storage import Storage


class RealisticPaperExecutor(Executor):
    name = "paper"

    def __init__(self, storage: Storage, provider: MarketDataProvider | None,
                 slippage: SlippageModel, cfg: ExecutionConfig):
        self.storage = storage
        self.provider = provider
        self.slippage = slippage
        self.cfg = cfg

    # ---- entry ---------------------------------------------------------- #
    def place(self, signal: Signal, stake: float, signal_id: int | None = None) -> Position:
        ref = None
        token_id = None
        outcome_index = None
        asks = None
        ref_price = signal.entry_price
        degraded = None

        want_live = self.cfg.fill_model in {"mid", "book"} and self.provider is not None
        if want_live:
            ref = self.provider.find_market(signal.market or "")
            if ref is None:
                if self.cfg.require_live_market:
                    raise ExecutionError(
                        f"No live market matched '{signal.market}' "
                        "(require_live_market is on).")
                degraded = "no-live-market"
            else:
                token_id, outcome_index = ref.token_for_side(signal.side.value)
                if self.cfg.fill_model == "book":
                    asks = self.provider.get_book(token_id) or None
                if not asks:  # book empty or mid model -> use a live price
                    live = self.provider.get_price(token_id, "buy")
                    if live is not None:
                        ref_price = live
                    elif self.cfg.fill_model == "book":
                        degraded = "no-live-book"

        if ref_price is None and asks is None:
            # Nothing to fill against at all.
            raise ExecutionError("No price available (alert had no entry "
                                 "price and no live data).")

        fill = self.slippage.simulate_buy(stake, ref_price, asks)

        meta = {
            "market_id": ref.market_id if ref else None,
            "token_id": token_id,
            "outcome_index": outcome_index,
            "reference_price": round(fill.reference_price, 4),
            "alert_price": signal.entry_price,
            "slippage_bps": round(fill.slippage_bps, 1),
            "fee_paid": round(fill.fee_paid, 4),
            "unfilled_stake": round(fill.unfilled_stake, 2),
            "fill_model": self.cfg.fill_model,
        }
        if degraded:
            meta["degraded"] = degraded

        pos = Position(
            signal_id=signal_id,
            market=(ref.question if ref else signal.market),
            side=signal.side.value,
            entry_price=round(fill.effective_price, 4),
            stake=round(fill.filled_stake, 2),
            shares=round(fill.shares, 4),
            venue=self.name,
            status="open",
            external_id=f"paper-{signal_id or 'na'}",
            meta=meta,
        )
        pos_id = self.storage.record_position(pos)
        pos.external_id = f"paper-{pos_id}"
        return pos

    # ---- mark to market ------------------------------------------------- #
    def mark_to_market(self) -> list[dict]:
        """Live unrealized P&L for each open position (needs a provider)."""
        out = []
        for row in self.storage.open_positions():
            meta = _meta(row)
            token_id = meta.get("token_id")
            price = None
            if self.provider and token_id:
                price = self.provider.get_price(token_id, "sell")
            unreal = None
            if price is not None:
                unreal = row["shares"] * price - row["stake"]
            out.append({
                "id": row["id"], "market": row["market"], "side": row["side"],
                "stake": row["stake"], "shares": row["shares"],
                "entry_price": row["entry_price"], "mark_price": price,
                "unrealized_pnl": None if unreal is None else round(unreal, 2),
            })
        return out

    # ---- settlement ----------------------------------------------------- #
    def settle_resolved(self) -> list[tuple[int, str, float]]:
        """Auto-settle open positions whose markets have resolved on-chain."""
        results = []
        if self.provider is None:
            return results
        for row in self.storage.open_positions():
            meta = _meta(row)
            market_id = meta.get("market_id")
            held = meta.get("outcome_index")
            if market_id is None or held is None:
                continue
            ref = self.provider.get_market(str(market_id))
            if ref is None:
                continue
            winner = ref.winning_index()
            if winner is None:
                continue  # not resolved yet
            if winner == held:
                pnl = row["shares"] - row["stake"]
                status = "won"
            else:
                pnl = -row["stake"]
                status = "lost"
            pnl = round(pnl, 2)
            if self.storage.resolve_position(row["id"], status, pnl):
                results.append((row["id"], status, pnl))
        return results

    # ---- manual settlement (parity with the simple paper executor) ------ #
    def resolve(self, pos_id: int, outcome: str) -> tuple[bool, float, str]:
        row = self.storage.get_position(pos_id)
        if row is None or row["status"] != "open":
            return False, 0.0, "not-open"
        o = outcome.strip().lower()
        shares = row["shares"] or 0.0
        stake = float(row["stake"])
        if o in {"win", "won", "w", "yes", "1"}:
            pnl = (shares - stake) if shares else stake
            status = "won"
        elif o in {"loss", "lost", "l", "no", "0"}:
            pnl = -stake
            status = "lost"
        else:
            try:
                exit_price = float(o)
            except ValueError:
                return False, 0.0, "bad-outcome"
            pnl = shares * exit_price - stake if shares else 0.0
            status = "won" if pnl >= 0 else "lost"
        pnl = round(pnl, 2)
        self.storage.resolve_position(pos_id, status, pnl)
        return True, pnl, status


def _meta(row) -> dict:
    import json
    raw = row["meta"] if "meta" in row.keys() else None
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
