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

import urllib.parse
import urllib.request
import json

from .base import Executor, ExecutionError
from ..config import Settings
from ..models import Signal, Position, Side
from ..storage import Storage

GAMMA_API = "https://gamma-api.polymarket.com"


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
        """Return ``(token_id, best_price)`` for the alert's market & side."""
        if not signal.market:
            raise ExecutionError("Alert has no parseable market to match.")

        markets = self._gamma_get(
            "/markets",
            {"active": "true", "closed": "false", "limit": 20,
             "search": signal.market[:80]},
        )
        if not markets:
            raise ExecutionError(f"No active Polymarket market matched "
                                 f"'{signal.market}'.")

        market = markets[0]
        token_ids = market.get("clobTokenIds")
        outcomes = market.get("outcomes")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not token_ids or not outcomes:
            raise ExecutionError("Matched market is missing CLOB token ids.")

        # Map YES/NO (or BUY->YES, SELL->NO) onto the outcome list.
        want = "Yes" if signal.side in (Side.YES, Side.BUY, Side.LONG) else "No"
        idx = next((i for i, o in enumerate(outcomes)
                    if str(o).lower() == want.lower()), 0)
        token_id = str(token_ids[idx])
        price = signal.entry_price if signal.entry_price else 0.5
        return token_id, float(price)

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
