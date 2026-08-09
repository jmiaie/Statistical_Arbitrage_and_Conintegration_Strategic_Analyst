"""Live Polymarket executor (real money at risk).

This is guarded behind explicit configuration so it can never fire by
accident:

  1. StrategyConfig.mode must be ``live``.
  2. The ``LIVE_TRADING`` env var must be truthy.
  3. Polymarket API credentials must be present in the environment.
  4. The optional ``py-clob-client`` dependency must be installed.

It resolves a freeform alert to a Polymarket market via the public Gamma API
(matching the parsed ``market`` text), picks the YES/NO outcome token from the
parsed ``side``, and submits a limit order through the CLOB client.

Because mapping freeform English to the exact on-chain market is inherently
fuzzy, the executor will *refuse* (raise) rather than guess when it cannot
confidently identify a single market/token. Treat the matching layer as the
piece to harden before trusting it with size.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import json

from .base import Executor, ExecutionError
from ..config import Settings
from ..models import Signal, Position, Side
from ..storage import Storage

GAMMA_API = "https://gamma-api.polymarket.com"

# How confidently we must identify a single market before risking real money.
_MIN_MATCH_SCORE = 2        # query/question must share at least this many words
_AMBIGUITY_MARGIN = 1       # best must beat the runner-up by more than this


def _words(text: str) -> set[str]:
    # Drop 1-2 char tokens so stopwords/punctuation don't inflate the score.
    return {w for w in re.findall(r"\w+", (text or "").lower()) if len(w) > 2}


def select_market(markets: list[dict], query: str) -> dict:
    """Pick the single market that matches ``query``, or refuse.

    Mapping freeform English onto an exact on-chain market is fuzzy, so this is
    deliberately strict: it ranks candidates by word overlap with the alert's
    market text and raises :class:`ExecutionError` rather than guess when the
    best match is weak (< ``_MIN_MATCH_SCORE`` shared words) or ambiguous (a
    runner-up within ``_AMBIGUITY_MARGIN``). Real funds are at stake here.
    """
    qwords = _words(query)
    if not markets:
        raise ExecutionError(f"No active Polymarket market matched '{query}'.")
    if not qwords:
        raise ExecutionError(
            "Alert market text has no usable words to match against.")

    scored = sorted(
        ((len(qwords & _words(str(m.get("question") or m.get("title") or ""))), m)
         for m in markets),
        key=lambda s: s[0], reverse=True,
    )
    best_score, best = scored[0]
    if best_score < _MIN_MATCH_SCORE:
        raise ExecutionError(
            f"Best Polymarket match for '{query}' is too weak "
            f"(only {best_score} shared word(s)); refusing to guess.")
    if len(scored) > 1 and best_score - scored[1][0] <= _AMBIGUITY_MARGIN:
        raise ExecutionError(
            f"Ambiguous market match for '{query}' "
            f"(top candidates score {best_score} vs {scored[1][0]}); "
            "refusing to guess.")
    return best


class PolymarketExecutor(Executor):
    name = "polymarket"

    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage
        self._client = None
        self._verify_guards()

    # ---- safety --------------------------------------------------------- #
    def _verify_guards(self) -> None:
        s = self.settings
        if not s.live_trading:
            raise ExecutionError(
                "Live trading is disabled. Set LIVE_TRADING=true to arm the "
                "Polymarket executor (real funds at risk)."
            )
        if not s.polymarket_private_key:
            raise ExecutionError("POLYMARKET_PRIVATE_KEY is not set.")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ExecutionError(
                "py-clob-client is not installed. Run "
                "`pip install py-clob-client` to enable live Polymarket trading."
            ) from exc

        s = self.settings
        client = ClobClient(
            host=s.polymarket_host,
            key=s.polymarket_private_key,
            chain_id=137,  # Polygon mainnet
            funder=s.polymarket_funder or None,
        )
        if s.polymarket_api_key:
            client.set_api_creds(ApiCreds(
                api_key=s.polymarket_api_key,
                api_secret=s.polymarket_api_secret,
                api_passphrase=s.polymarket_api_passphrase,
            ))
        else:
            # Derive/create API credentials from the private key.
            client.set_api_creds(client.create_or_derive_api_creds())
        self._client = client
        return client

    # ---- market resolution --------------------------------------------- #
    @staticmethod
    def _gamma_get(path: str, params: dict) -> list:
        url = f"{GAMMA_API}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def _resolve_token(self, signal: Signal) -> tuple[str, float]:
        """Return ``(token_id, limit_price)`` for the alert's market & side.

        Refuses (raises) rather than guessing when the market match is weak or
        ambiguous, when the requested side's outcome token can't be identified,
        or when the alert carries no entry price — never invents a price.
        """
        if not signal.market:
            raise ExecutionError("Alert has no parseable market to match.")
        if signal.entry_price is None:
            raise ExecutionError(
                "Alert has no entry price; refusing to invent a limit price "
                "for a live order.")

        markets = self._gamma_get(
            "/markets",
            {"active": "true", "closed": "false", "limit": 20,
             "search": signal.market[:80]},
        )
        market = select_market(markets, signal.market)

        token_ids = market.get("clobTokenIds")
        outcomes = market.get("outcomes")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not token_ids or not outcomes:
            raise ExecutionError("Matched market is missing CLOB token ids.")

        # Map YES/NO (or BUY->YES, SELL->NO) onto the outcome list. Refuse if
        # the requested side has no matching named outcome — don't fall back to
        # outcome 0 and silently trade the wrong direction.
        want = "Yes" if signal.side in (Side.YES, Side.BUY, Side.LONG) else "No"
        idx = next((i for i, o in enumerate(outcomes)
                    if str(o).strip().lower() == want.lower()), None)
        if idx is None:
            raise ExecutionError(
                f"Could not map side '{signal.side.value}' onto market "
                f"outcomes {outcomes}; refusing to guess the token.")
        return str(token_ids[idx]), float(signal.entry_price)

    # ---- order placement ------------------------------------------------ #
    def place(self, signal: Signal, stake: float, signal_id: int | None = None) -> Position:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        client = self._get_client()
        token_id, price = self._resolve_token(signal)

        # Convert a currency stake into a share count at the limit price.
        price = max(0.01, min(0.99, round(price, 2)))
        size = round(stake / price, 2)

        order = client.create_order(OrderArgs(
            token_id=token_id, price=price, size=size, side=BUY,
        ))
        resp = client.post_order(order, OrderType.GTC)
        order_id = resp.get("orderID") or resp.get("orderId") or "unknown"

        pos = Position(
            signal_id=signal_id,
            market=signal.market,
            side=signal.side.value,
            entry_price=price,
            stake=stake,
            venue=self.name,
            status="open",
            external_id=str(order_id),
        )
        self.storage.record_position(pos)
        return pos
