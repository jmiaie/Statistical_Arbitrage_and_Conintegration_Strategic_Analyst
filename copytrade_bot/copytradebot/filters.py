"""Filter engine — decides whether a parsed Signal should be traded."""

from __future__ import annotations

from .config import FilterConfig
from .models import Signal, FilterResult


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:g}"


class FilterEngine:
    """Evaluate a :class:`Signal` against an (adjustable) :class:`FilterConfig`."""

    def __init__(self, config: FilterConfig):
        self.config = config

    def evaluate(self, signal: Signal) -> FilterResult:
        c = self.config
        result = FilterResult(passed=True)

        # Required fields must have parsed.
        present = signal.present_fields()
        for fld in c.require_fields:
            result.add(fld in present, f"missing required field '{fld}'")

        # Blocked keywords in the raw text.
        low = (signal.raw_text or "").lower()
        for kw in c.blocked_keywords:
            result.add(kw.lower() not in low, f"contains blocked keyword '{kw}'")

        # Source allow/deny.
        if c.allowed_sources:
            result.add(
                signal.source in c.allowed_sources,
                f"source '{signal.source}' not in allowed list",
            )
        if c.blocked_sources:
            result.add(
                signal.source not in c.blocked_sources,
                f"source '{signal.source}' is blocked",
            )

        # Numeric thresholds — only enforced when both the bound and the value
        # are present. A missing value is handled via require_fields above.
        # Gate on the calibrated win rate when available (falls back to quoted).
        self._min(result, "win rate", signal.effective_win_rate(), c.min_win_rate)
        self._min(result, "EV", signal.ev, c.min_ev)
        self._min(result, "ROI", signal.roi, c.min_roi)
        self._min(result, "expected return", signal.expected_return,
                  c.min_expected_return)

        self._range(result, "entry price", signal.entry_price,
                    c.min_entry_price, c.max_entry_price)
        self._range(result, "size", signal.size, c.min_size, c.max_size)

        return result

    @staticmethod
    def _min(result: FilterResult, name: str, value, lo) -> None:
        if lo is None or value is None:
            return
        result.add(value >= lo, f"{name} {_pct(value)} < min {_pct(lo)}")

    @staticmethod
    def _range(result: FilterResult, name: str, value, lo, hi) -> None:
        if value is None:
            return
        if lo is not None:
            result.add(value >= lo, f"{name} {_pct(value)} < min {_pct(lo)}")
        if hi is not None:
            result.add(value <= hi, f"{name} {_pct(value)} > max {_pct(hi)}")
