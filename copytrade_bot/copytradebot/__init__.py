"""Telegram alert copy-trading / agentic bot.

A pipeline that ingests freeform trade alerts from a Telegram channel/group,
parses them, filters them against fully-adjustable criteria (win rate, EV,
ROI, return, entry price, trade size, ...), sizes positions, and routes the
survivors to a pluggable executor (paper trading or live Polymarket).
"""

__version__ = "0.1.0"
