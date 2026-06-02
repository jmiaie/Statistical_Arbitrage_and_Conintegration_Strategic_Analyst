"""Freeform-text alert parser.

Trade alerts arrive as human-written messages, e.g.::

    🔥 NEW PLAY 🔥
    Market: Will BTC close above $70k in June?
    Side: YES
    Entry: 0.42
    Win rate ~ 68%
    EV: +14%   ROI potential: 30%
    Suggested size: $75

This module extracts structured fields with tolerant regexes. It is
intentionally heuristic; unit tests in ``tests/test_parser.py`` pin the
behaviour, and new alert phrasings can be supported by extending the
``*_PATTERNS`` lists below.
"""

from __future__ import annotations

import re
from typing import Optional

from .models import Signal, Side

# Up to 12 non-digit, same-line characters may sit between a label and its
# value ("Win rate of 65%", "EV: +14%", "Entry ~ 0.42").
_GAP = r"[^\d\n]{0,12}?"
_NUM = r"([+-]?\d+(?:\.\d+)?)"
_PCT = r"\s*(%?)"


def _compile(labels: list[str], num: str = _NUM) -> list[re.Pattern]:
    return [re.compile(lbl + _GAP + num + _PCT, re.IGNORECASE) for lbl in labels]


WIN_RATE_PATTERNS = _compile([
    r"win\s*rate", r"winrate", r"\bwr\b", r"hit\s*rate", r"strike\s*rate",
    r"accuracy", r"prob(?:ability)?", r"\bconfidence\b", r"\bedge%",
])
EV_PATTERNS = _compile([r"expected\s*value", r"\bev\b", r"\bedge\b"])
ROI_PATTERNS = _compile([r"roi(?:\s*potential)?", r"return\s*on\s*investment"])
RETURN_PATTERNS = _compile([
    r"expected\s*return", r"target\s*return", r"potential\s*return",
    r"\breturn\b", r"\bupside\b", r"\bpayout\b", r"\bgain\b",
])
ENTRY_PATTERNS = _compile([
    r"entry\s*price", r"\bentry\b", r"buy\s*at", r"fill", r"limit", r"\bprice\b",
    r"@",
])
SIZE_PATTERNS = [
    re.compile(lbl + _GAP + r"\$?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
    for lbl in [r"suggested\s*size", r"\bsize\b", r"\bstake\b", r"\bbet\b",
                r"\brisk\b", r"allocate", r"\bamount\b", r"position\s*size"]
]

SIDE_PATTERN = re.compile(
    r"\b(YES|NO|LONG|SHORT|BUY|SELL|OVER|UNDER|BULL|BEAR|CALL|PUT)\b",
    re.IGNORECASE,
)
SIDE_LABEL_PATTERN = re.compile(
    r"(?:side|direction|action|signal)\s*[:\-]\s*([A-Za-z]+)", re.IGNORECASE
)

# Market / instrument identification.
TICKER_PATTERN = re.compile(r"\$([A-Z]{1,6}(?:[-/][A-Z]{1,6})?)\b")
MARKET_LABEL_PATTERN = re.compile(
    r"(?:market|ticker|symbol|pair|event|trade|play|asset)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)


def _first(patterns: list[re.Pattern], text: str) -> Optional[re.Match]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m
    return None


def _num(patterns: list[re.Pattern], text: str) -> Optional[float]:
    m = _first(patterns, text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except (ValueError, IndexError):
        return None


def _extract_win_rate(text: str) -> Optional[float]:
    """Win rate normalised to a fraction in [0, 1]."""
    m = _first(WIN_RATE_PATTERNS, text)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    # "0.68" stays, "68" or "68%" -> 0.68.
    if val > 1:
        val /= 100.0
    return max(0.0, min(1.0, val))


def _extract_market(text: str) -> Optional[str]:
    m = MARKET_LABEL_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    t = TICKER_PATTERN.search(text)
    if t:
        return t.group(1)
    # Fall back to the first non-empty, non-decorative line.
    for line in text.splitlines():
        cleaned = line.strip().strip("🔥🚀📈📉⭐️*•-_ ").strip()
        if len(cleaned) >= 3 and any(c.isalpha() for c in cleaned):
            return cleaned[:120]
    return None


def _extract_side(text: str) -> Side:
    m = SIDE_LABEL_PATTERN.search(text)
    if m:
        side = Side.from_text(m.group(1))
        if side is not Side.UNKNOWN:
            return side
    m = SIDE_PATTERN.search(text)
    if m:
        return Side.from_text(m.group(1))
    return Side.UNKNOWN


def parse_alert(text: str, source: str = "unknown") -> Signal:
    """Parse a freeform alert into a :class:`Signal`.

    Always returns a Signal (never raises); unfound fields stay ``None`` so
    the filter layer can decide whether missing data disqualifies the alert.
    """
    if text is None:
        text = ""

    return Signal(
        raw_text=text,
        source=source,
        market=_extract_market(text),
        side=_extract_side(text),
        win_rate=_extract_win_rate(text),
        ev=_num(EV_PATTERNS, text),
        roi=_num(ROI_PATTERNS, text),
        expected_return=_num(RETURN_PATTERNS, text),
        entry_price=_num(ENTRY_PATTERNS, text),
        size=_num(SIZE_PATTERNS, text),
    )


def enrich(signal: Signal) -> Signal:
    """Derive missing economics where possible.

    For prediction-market style alerts (entry price is an implied probability
    in (0, 1)) we can compute EV% from win rate and price even if the author
    didn't state it::

        EV_fraction = p*(1/price - 1) - (1 - p)

    where ``p`` is win rate and ``price`` the implied probability paid.
    """
    if (
        signal.ev is None
        and signal.win_rate is not None
        and signal.entry_price is not None
        and 0 < signal.entry_price < 1
    ):
        p = signal.win_rate
        price = signal.entry_price
        payoff = (1.0 / price) - 1.0
        ev_fraction = p * payoff - (1 - p)
        signal.ev = round(ev_fraction * 100, 2)
        if signal.roi is None:
            signal.roi = signal.ev  # for a binary market ROI≈EV per $1 staked
    return signal
