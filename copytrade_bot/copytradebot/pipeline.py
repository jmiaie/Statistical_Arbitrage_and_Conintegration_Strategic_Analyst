"""The decision pipeline: alert text in, trade decision out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .config import StrategyConfig, Settings
from .filters import FilterEngine
from .models import Signal, FilterResult, Position, Intent
from .parser import parse_alert, enrich
from .sizing import compute_stake
from .storage import Storage
from .executors import build_executor


@dataclass
class Decision:
    signal: Signal
    result: FilterResult
    signal_id: int
    stake: Optional[float] = None
    position: Optional[Position] = None
    placed: bool = False
    note: str = ""
    # For EXIT alerts: (position_id, status, pnl) for each position closed.
    closed: list = field(default_factory=list)

    def summary(self) -> str:
        s = self.signal
        if s.intent is Intent.EXIT:
            if self.closed:
                rows = ", ".join(f"#{pid} {st} {pnl:+g}"
                                 for pid, st, pnl in self.closed)
                return f"💰 CLOSE · {s.market or 'unknown market'}\n{rows}"
            return (f"🚫 CLOSE · {s.market or 'unknown market'}\n"
                    f"{self.note or 'no matching open position'}")
        head = f"{'✅ TRADE' if self.placed else '🚫 SKIP'} · {s.market or 'unknown market'}"
        bits = []
        if s.side.value != "UNKNOWN":
            bits.append(s.side.value)
        if s.win_rate is not None:
            bits.append(f"WR {s.win_rate*100:.0f}%")
        if s.ev is not None:
            bits.append(f"EV {s.ev:+g}%")
        if s.roi is not None:
            bits.append(f"ROI {s.roi:+g}%")
        if s.entry_price is not None:
            bits.append(f"@ {s.entry_price:g}")
        line2 = " · ".join(bits) if bits else "(no metrics parsed)"
        if self.placed:
            tail = f"staked {self.stake:g} via {self.position.venue} (id {self.position.external_id})"
        elif self.result.passed and self.note:
            tail = self.note
        else:
            tail = "; ".join(self.result.reasons) or self.note or "filtered out"
        return f"{head}\n{line2}\n{tail}"


class Pipeline:
    """Owns config + storage + executor and processes incoming alerts."""

    def __init__(self, config: StrategyConfig, settings: Settings, storage: Storage):
        self.config = config
        self.settings = settings
        self.storage = storage

    @staticmethod
    def _today_start() -> float:
        now = time.localtime()
        return time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                            0, 0, 0, 0, 0, -1))

    def _executor(self):
        return build_executor(self.config, self.settings, self.storage)

    def _risk_ok(self) -> tuple[bool, str]:
        risk = self.config.risk
        if self.storage.count_open_positions() >= risk.max_open_positions:
            return False, f"max open positions ({risk.max_open_positions}) reached"
        # Recompute the day boundary each call so the daily loss limit actually
        # rolls over (it must not be frozen at process-start).
        loss = -self.storage.realized_pnl_since(self._today_start())
        if loss >= risk.max_daily_loss:
            return False, f"daily loss limit hit ({loss:.2f} >= {risk.max_daily_loss})"
        return True, ""

    def _exposure_ok(self, stake: float) -> tuple[bool, str]:
        """Reject a prospective stake that would breach the total-exposure cap."""
        frac = self.config.risk.max_exposure_fraction
        if frac is None:
            return True, ""
        cap = self.config.sizing.bankroll * frac
        projected = self.storage.open_exposure() + stake
        if projected > cap:
            return (False, f"exposure cap hit (open+new {projected:.2f} > "
                           f"{cap:.2f} = {frac:g}x bankroll)")
        return True, ""

    def _process_exit(self, signal: Signal, decision: Decision) -> Decision:
        """Settle the open position(s) an exit/close alert refers to.

        Matches by market word-overlap so an exit only ever closes related
        trades. Prices the close from the alert's stated exit price, falling
        back to a live mark (realistic executor) — and refuses to guess a price
        rather than fabricate P&L. Live-mode exits aren't auto-handled yet.
        """
        if self.config.dry_run:
            decision.note = "dry-run: exit not executed"
            return decision
        if (self.config.mode or "paper").lower() == "live":
            decision.note = ("exit detected but live-mode auto-close is not "
                             "supported yet; close manually on Polymarket")
            return decision
        if not signal.market:
            decision.note = "exit had no market text to match an open position"
            return decision

        matches = self.storage.find_open_by_market(signal.market)
        if not matches:
            decision.note = f"no open position matched '{signal.market}'"
            return decision

        executor = self._executor()
        if not hasattr(executor, "resolve"):
            decision.note = "active executor cannot close positions"
            return decision

        # Live marks (only available on the realistic executor) let us price an
        # exit that didn't state a price.
        marks = {}
        if signal.entry_price is None and hasattr(executor, "mark_to_market"):
            try:
                marks = {r["id"]: r.get("mark_price")
                         for r in executor.mark_to_market()}
            except Exception:  # network/provider issues shouldn't crash a close
                marks = {}

        for row in matches:
            price = signal.entry_price
            if price is None:
                price = marks.get(row["id"])
            if price is None:
                if not decision.note:
                    decision.note = ("matched open position(s) but no exit price "
                                     "available; use /resolve <id> <win|loss|price>")
                continue
            ok, pnl, status = executor.resolve(row["id"], str(price))
            if ok:
                decision.closed.append((row["id"], status, pnl))

        if not decision.closed and not decision.note:
            decision.note = "exit matched positions but none could be closed"
        return decision

    def process(self, text: str, source: str = "unknown") -> Decision:
        signal = enrich(parse_alert(text, source))
        engine = FilterEngine(self.config.filters)
        result = engine.evaluate(signal)
        signal_id = self.storage.record_signal(signal, result)
        decision = Decision(signal=signal, result=result, signal_id=signal_id)

        if not self.config.enabled:
            decision.note = "bot disabled"
            return decision

        # Exit/close alerts never open a position; they settle matching ones.
        if signal.intent is Intent.EXIT:
            return self._process_exit(signal, decision)

        if not result.passed:
            return decision

        ok, why = self._risk_ok()
        if not ok:
            decision.note = f"blocked by risk: {why}"
            return decision

        stake = compute_stake(signal, self.config.sizing)
        decision.stake = stake

        ok, why = self._exposure_ok(stake)
        if not ok:
            decision.note = f"blocked by risk: {why}"
            return decision

        if self.config.dry_run:
            decision.note = f"dry-run: would stake {stake:g}"
            return decision

        try:
            executor = self._executor()
            decision.position = executor.place(signal, stake, signal_id)
            decision.placed = True
        except Exception as exc:  # surface execution failures, don't crash loop
            decision.note = f"execution error: {exc}"
        return decision
