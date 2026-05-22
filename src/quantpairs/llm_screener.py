"""LLM-powered cointegration candidate screener.

Given a sector or theme, ask Claude to propose pair candidates with economic
rationale, then hand them to the statistical filter. The LLM contributes
hypotheses; statistics decide which survive.

Uses prompt caching on the universe / system block so iterating on themes
is cheap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantpairs.cointegration import engle_granger_test

DEFAULT_MODEL = "claude-opus-4-7"

_SYSTEM_PROMPT = """You are a senior pairs-trading research analyst. Given a
theme or sector, propose 5-10 candidate equity pairs where economic intuition
suggests cointegration: same supply chain, dual-listed equivalents, peer
duopolies, commodity-linked equity pairs, etc. For each pair, give a
one-sentence economic rationale. Respond as JSON only:

{"candidates": [{"y": "TICKER", "x": "TICKER", "rationale": "..."}]}

Use US-listed liquid tickers. Avoid penny stocks and crypto."""


@dataclass(frozen=True)
class PairCandidate:
    """LLM-proposed pair with economic narrative."""

    y: str
    x: str
    rationale: str


def propose_candidates(
    theme: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> list[PairCandidate]:
    """Ask Claude for plausible cointegrated pairs given a theme."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": f"Theme: {theme}"}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    payload: dict[str, Any] = _extract_json(text)
    return [PairCandidate(**c) for c in payload.get("candidates", [])]


def _extract_json(text: str) -> dict[str, Any]:
    """Robustly extract the first JSON object in the model's response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    result: dict[str, Any] = json.loads(text[start : end + 1])
    return result


def screen_candidates(
    candidates: list[PairCandidate],
    prices: pd.DataFrame,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Run Engle-Granger on each LLM-proposed pair; return ranked survivors."""
    import numpy as np

    rows: list[dict[str, object]] = []
    for c in candidates:
        if c.y not in prices.columns or c.x not in prices.columns:
            continue
        try:
            res = engle_granger_test(
                np.log(prices[c.y]), np.log(prices[c.x]), significance=significance
            )
        except Exception as exc:
            rows.append({"y": c.y, "x": c.x, "rationale": c.rationale, "error": str(exc)})
            continue
        rows.append(
            {
                "y": c.y,
                "x": c.x,
                "rationale": c.rationale,
                "eg_pvalue": res.pvalue,
                "adf_pvalue": res.adf_pvalue,
                "hedge_ratio": res.hedge_ratio,
                "half_life_days": res.half_life,
                "is_cointegrated": res.is_cointegrated,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["is_cointegrated", "eg_pvalue"], ascending=[False, True], na_position="last"
        )
        .reset_index(drop=True)
    )


def cli() -> None:
    """`quantpairs-screen "energy transition"` — quick CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM-powered pair screener.")
    parser.add_argument("theme", help="Theme or sector to brainstorm pairs in.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    cands = propose_candidates(args.theme, model=args.model)
    for c in cands:
        print(f"{c.y:6s} ~ {c.x:6s}  — {c.rationale}")


if __name__ == "__main__":  # pragma: no cover
    cli()
