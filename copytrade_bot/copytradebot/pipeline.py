"""The decision pipeline: alert text in, trade decision out."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .config import StrategyConfig, Settings
from .filters import FilterEngine
from .models import Signal, FilterResult, Position
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

    def summary(self) -> str:
        s = self.signal
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
        self._day_start = self._today_start()

    @staticmethod
    def _today_start() -> float:
        now = time.localtime()
        return time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                            0, 0, 0, 0, 0, -1))

    def _executor(self):
        return build_executor(self.config.mode, self.settings, self.storage)

    def _risk_ok(self) -> tuple[bool, str]:
        risk = self.config.risk
        if self.storage.count_open_positions() >= risk.max_open_positions:
            return False, f"max open positions ({risk.max_open_positions}) reached"
        loss = -self.storage.realized_pnl_since(self._day_start)
        if loss >= risk.max_daily_loss:
            return False, f"daily loss limit hit ({loss:.2f} >= {risk.max_daily_loss})"
        return True, ""

    def process(self, text: str, source: str = "unknown") -> Decision:
        signal = enrich(parse_alert(text, source))
        engine = FilterEngine(self.config.filters)
        result = engine.evaluate(signal)
        signal_id = self.storage.record_signal(signal, result)
        decision = Decision(signal=signal, result=result, signal_id=signal_id)

        if not self.config.enabled:
            decision.note = "bot disabled"
            return decision
        if not result.passed:
            return decision

        ok, why = self._risk_ok()
        if not ok:
            decision.note = f"blocked by risk: {why}"
            return decision

        stake = compute_stake(signal, self.config.sizing)
        decision.stake = stake

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
