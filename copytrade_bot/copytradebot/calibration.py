"""Per-source win-rate calibration.

Channels quote win rates; those quotes are *claims*, and the Monte Carlo work
in ``simulation/`` shows a strategy that trusts inflated quotes bleeds. This
module turns settled history into a correction: for each source we compare the
win rate it *quoted* against the rate it actually *realized*, and shrink new
quotes toward the truth before they reach the filter and (critically) Kelly
sizing.

Model (deliberately simple and robust):

    raw_bias  = mean_quoted - realized          # >0 means the source overstates
    bias_hat  = (n / (n + K)) * raw_bias        # shrink toward 0 by sample size
    calibrated(q) = clip(q - bias_hat)

``K`` (prior strength, in pseudo-samples) controls how much settled history we
need before trusting the correction: with few settled trades the adjustment is
small; it grows as evidence accumulates. With no history the quote is returned
unchanged, so enabling calibration is always safe.
"""

from __future__ import annotations

from dataclasses import dataclass

PRIOR_STRENGTH = 20.0     # pseudo-samples pulling the bias estimate toward 0
_LO, _HI = 0.01, 0.99


@dataclass
class SourceStats:
    source: str
    n: int = 0
    wins: int = 0
    sum_quoted: float = 0.0

    @property
    def realized(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def mean_quoted(self) -> float:
        return self.sum_quoted / self.n if self.n else 0.0

    def bias(self, prior: float = PRIOR_STRENGTH) -> float:
        """Sample-size-shrunk estimate of how much this source overstates."""
        if not self.n:
            return 0.0
        raw = self.mean_quoted - self.realized
        weight = self.n / (self.n + prior)
        return weight * raw


class Calibrator:
    """Maps a source's quoted win rate to a calibrated one."""

    def __init__(self, stats: dict[str, SourceStats] | None = None,
                 prior: float = PRIOR_STRENGTH):
        self.stats = stats or {}
        self.prior = prior

    @classmethod
    def from_rows(cls, rows, prior: float = PRIOR_STRENGTH) -> "Calibrator":
        """Build from ``(source, quoted_win_rate, won_bool)`` rows.

        Rows with no quoted win rate are skipped so the mean stays consistent.
        """
        agg: dict[str, SourceStats] = {}
        for source, quoted, won in rows:
            if quoted is None:
                continue
            st = agg.setdefault(source, SourceStats(source))
            st.n += 1
            st.wins += 1 if won else 0
            st.sum_quoted += float(quoted)
        return cls(agg, prior)

    def calibrate(self, source: str, quoted):
        """Calibrated win rate for a quote from ``source`` (quote unchanged when
        the source has no settled history)."""
        if quoted is None:
            return None
        st = self.stats.get(source)
        if st is None:
            return quoted
        return min(_HI, max(_LO, quoted - st.bias(self.prior)))

    def report(self) -> list[dict]:
        """Per-source diagnostics, most-sampled first."""
        out = []
        for st in sorted(self.stats.values(), key=lambda s: s.n, reverse=True):
            out.append({
                "source": st.source, "n": st.n,
                "realized": round(st.realized, 3),
                "quoted": round(st.mean_quoted, 3),
                "bias": round(st.bias(self.prior), 3),
            })
        return out
