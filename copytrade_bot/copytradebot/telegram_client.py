"""Minimal Telegram Bot API client (long polling).

Deliberately dependency-light (just ``requests``) so the bot has no heavy
framework to track across Telegram library versions.
"""

from __future__ import annotations

import logging
from typing import Iterator

import requests

log = logging.getLogger("copytrade.telegram")


class TelegramClient:
    def __init__(self, token: str, timeout: int = 30):
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required.")
        self.base = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout
        self._offset = 0
        self._session = requests.Session()

    def _call(self, method: str, **params) -> dict:
        resp = self._session.post(f"{self.base}/{method}", json=params,
                                  timeout=self.timeout + 15)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error on {method}: {data}")
        return data["result"]

    def get_me(self) -> dict:
        return self._call("getMe")

    def send_message(self, chat_id: int, text: str) -> None:
        # Telegram caps messages at 4096 chars.
        for chunk in (text[i:i + 4000] for i in range(0, len(text) or 1, 4000)):
            try:
                self._call("sendMessage", chat_id=chat_id, text=chunk,
                           disable_web_page_preview=True)
            except Exception as exc:  # don't let a send failure kill the loop
                log.warning("sendMessage failed: %s", exc)

    def poll(self) -> Iterator[dict]:
        """Yield new updates via long polling, advancing the offset."""
        updates = self._call(
            "getUpdates", offset=self._offset, timeout=self.timeout,
            allowed_updates=["message", "channel_post", "edited_channel_post"],
        )
        for upd in updates:
            self._offset = max(self._offset, upd["update_id"] + 1)
            yield upd
