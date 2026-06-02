"""The Telegram-facing bot: ingests alerts and serves admin commands."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

from .config import StrategyConfig, Settings, SETTABLE_FIELDS, coerce
from .pipeline import Pipeline
from .storage import Storage
from .telegram_client import TelegramClient
from .executors.paper import PaperExecutor

log = logging.getLogger("copytrade.bot")


class CopyTradeBot:
    def __init__(self, settings: Settings, config: StrategyConfig | None = None):
        self.settings = settings
        self.config = config or StrategyConfig.load(settings.filters_path)
        self.storage = Storage(settings.db_path)
        self.pipeline = Pipeline(self.config, settings, self.storage)
        self.tg = TelegramClient(settings.telegram_token, settings.poll_timeout)

    # ---- helpers -------------------------------------------------------- #
    def _is_admin(self, chat_id: int) -> bool:
        # If no admins configured, allow source chats so it's usable out of box.
        if not self.settings.admin_chat_ids:
            return True
        return chat_id in self.settings.admin_chat_ids

    def _is_source(self, chat_id: int) -> bool:
        # Empty source list = listen everywhere (handy for first-run testing).
        return (not self.settings.source_chat_ids
                or chat_id in self.settings.source_chat_ids)

    def _notify(self, text: str) -> None:
        target = self.settings.notify_chat_id
        if target is None and self.settings.admin_chat_ids:
            target = self.settings.admin_chat_ids[0]
        if target is not None:
            self.tg.send_message(target, text)

    def _save(self) -> None:
        self.config.save(self.settings.filters_path)

    # ---- main loop ------------------------------------------------------ #
    def run(self) -> None:
        me = self.tg.get_me()
        log.info("Connected as @%s. Mode=%s enabled=%s",
                 me.get("username"), self.config.mode, self.config.enabled)
        self._notify(
            f"🤖 Copy-trade bot online as @{me.get('username')} · "
            f"mode={self.config.mode} · enabled={self.config.enabled}"
        )
        while True:
            try:
                for upd in self.tg.poll():
                    self._handle_update(upd)
            except KeyboardInterrupt:
                log.info("Shutting down.")
                break
            except Exception as exc:  # keep the loop alive on transient errors
                log.warning("poll error: %s", exc)
                time.sleep(3)

    def _handle_update(self, upd: dict) -> None:
        msg = (upd.get("message") or upd.get("channel_post")
               or upd.get("edited_channel_post"))
        if not msg:
            return
        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text") or msg.get("caption") or ""
        if not text:
            return

        if text.lstrip().startswith("/"):
            if self._is_admin(chat_id):
                reply = self._handle_command(text.strip())
                self.tg.send_message(chat_id, reply)
            return

        if self._is_source(chat_id):
            self._handle_alert(text, chat_id)

    def _handle_alert(self, text: str, chat_id: int) -> None:
        source = str(chat_id)
        decision = self.pipeline.process(text, source=source)
        log.info("Alert from %s -> placed=%s", source, decision.placed)
        # Notify on every passed trade; skips only when in dry/verbose contexts.
        if decision.placed or decision.note.startswith(("dry-run", "blocked", "execution")):
            self._notify(decision.summary())

    # ---- commands ------------------------------------------------------- #
    def _handle_command(self, text: str) -> str:
        parts = text.split()
        cmd = parts[0].lower().lstrip("/").split("@")[0]
        args = parts[1:]
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return f"Unknown command /{cmd}. Send /help."
        try:
            return handler(args)
        except Exception as exc:
            return f"⚠️ {exc}"

    def _cmd_help(self, args) -> str:
        return (
            "Commands:\n"
            "/status — bot mode & summary\n"
            "/filters — show all filter/sizing/risk settings\n"
            "/set <key> <value> — adjust a setting (e.g. /set min_win_rate 0.6)\n"
            "/get <key> — show one setting\n"
            "/enable | /disable — master on/off switch\n"
            "/mode paper|live — switch execution venue\n"
            "/dryrun on|off — evaluate without placing\n"
            "/require <field,...> — set required parsed fields\n"
            "/stats — performance summary\n"
            "/positions — list open positions\n"
            "/resolve <id> <win|loss|price> — settle a paper position\n"
            "/recent [n] — last n signals with pass/fail reasons\n"
            "/test <alert text> — dry-run an alert through the pipeline\n"
            "Settable keys: " + ", ".join(SETTABLE_FIELDS)
        )

    def _cmd_start(self, args) -> str:
        return "🤖 Copy-trade bot ready. Send /help for commands."

    def _cmd_status(self, args) -> str:
        c = self.config
        return (
            f"Mode: {c.mode} | enabled: {c.enabled} | dry_run: {c.dry_run}\n"
            f"Sizing: {c.sizing.mode} (bankroll {c.sizing.bankroll:g}, "
            f"max_pos {c.sizing.max_position:g})\n"
            f"Filters: WR≥{c.filters.min_win_rate}, EV≥{c.filters.min_ev}, "
            f"ROI≥{c.filters.min_roi}, ret≥{c.filters.min_expected_return}\n"
            f"Entry [{c.filters.min_entry_price}, {c.filters.max_entry_price}], "
            f"Size [{c.filters.min_size}, {c.filters.max_size}]\n"
            f"Required fields: {c.filters.require_fields}"
        )

    def _cmd_filters(self, args) -> str:
        import json
        return json.dumps(self.config.to_dict(), indent=2)

    def _cmd_set(self, args) -> str:
        if len(args) < 2:
            return "Usage: /set <key> <value>"
        key, value = args[0], " ".join(args[1:])
        if key not in SETTABLE_FIELDS:
            return f"Unknown key '{key}'. Settable: {', '.join(SETTABLE_FIELDS)}"
        section, typ = SETTABLE_FIELDS[key]
        coerced = coerce(value, typ)
        setattr(getattr(self.config, section), key, coerced)
        self._save()
        return f"✅ {section}.{key} = {coerced}"

    def _cmd_get(self, args) -> str:
        if not args:
            return "Usage: /get <key>"
        key = args[0]
        if key not in SETTABLE_FIELDS:
            return f"Unknown key '{key}'."
        section, _ = SETTABLE_FIELDS[key]
        return f"{section}.{key} = {getattr(getattr(self.config, section), key)}"

    def _cmd_enable(self, args) -> str:
        self.config.enabled = True
        self._save()
        return "✅ Bot enabled."

    def _cmd_disable(self, args) -> str:
        self.config.enabled = False
        self._save()
        return "⏸️ Bot disabled (alerts parsed but not traded)."

    def _cmd_mode(self, args) -> str:
        if not args or args[0].lower() not in {"paper", "live"}:
            return "Usage: /mode paper|live"
        new_mode = args[0].lower()
        if new_mode == "live" and not self.settings.live_trading:
            return ("⚠️ Refusing: LIVE_TRADING env is not enabled. Set "
                    "LIVE_TRADING=true and provide Polymarket credentials first.")
        self.config.mode = new_mode
        self._save()
        return f"✅ Mode set to {new_mode}." + (
            " 🔴 REAL MONEY AT RISK." if new_mode == "live" else "")

    def _cmd_dryrun(self, args) -> str:
        if not args or args[0].lower() not in {"on", "off"}:
            return "Usage: /dryrun on|off"
        self.config.dry_run = args[0].lower() == "on"
        self._save()
        return f"✅ dry_run = {self.config.dry_run}"

    def _cmd_require(self, args) -> str:
        fields = [f.strip() for f in " ".join(args).replace(",", " ").split() if f.strip()]
        self.config.filters.require_fields = fields
        self._save()
        return f"✅ required fields = {fields}"

    def _cmd_stats(self, args) -> str:
        s = self.storage.stats()
        wr = f"{s['win_rate']*100:.1f}%" if s["win_rate"] is not None else "n/a"
        return (
            f"Signals seen: {s['signals_seen']} | passed: {s['signals_passed']}\n"
            f"Open positions: {s['open_positions']}\n"
            f"Settled: {s['settled']} | wins: {s['wins']} | win rate: {wr}\n"
            f"Realized P&L: {s['realized_pnl']:+.2f}"
        )

    def _cmd_positions(self, args) -> str:
        rows = self.storage.open_positions()
        if not rows:
            return "No open positions."
        lines = [
            f"#{r['id']} {r['market'] or '?'} {r['side']} "
            f"@ {r['entry_price']} stake {r['stake']:g} [{r['venue']}]"
            for r in rows
        ]
        return "Open positions:\n" + "\n".join(lines)

    def _cmd_resolve(self, args) -> str:
        if len(args) < 2:
            return "Usage: /resolve <id> <win|loss|exit_price>"
        try:
            pos_id = int(args[0])
        except ValueError:
            return "Position id must be a number."
        executor = PaperExecutor(self.storage)
        ok, pnl, status = executor.resolve(pos_id, args[1])
        if not ok:
            return f"Could not resolve #{pos_id} ({status})."
        return f"✅ #{pos_id} settled {status}, P&L {pnl:+.2f}"

    def _cmd_recent(self, args) -> str:
        n = int(args[0]) if args and args[0].isdigit() else 10
        rows = self.storage.recent_signals(n)
        if not rows:
            return "No signals yet."
        lines = []
        for r in rows:
            mark = "✅" if r["passed"] else "🚫"
            why = "" if r["passed"] else f" — {r['reasons']}"
            lines.append(f"{mark} #{r['id']} {r['market'] or '?'} "
                         f"WR={r['win_rate']} EV={r['ev']}{why}")
        return "\n".join(lines)

    def _cmd_test(self, args) -> str:
        """Dry-run an alert through the pipeline without placing a trade."""
        text = " ".join(args)
        if not text:
            return "Usage: /test <alert text>"
        from .parser import parse_alert, enrich
        from .filters import FilterEngine
        sig = enrich(parse_alert(text, source="test"))
        res = FilterEngine(self.config.filters).evaluate(sig)
        verdict = "WOULD TRADE ✅" if res.passed else "WOULD SKIP 🚫"
        reasons = "" if res.passed else "\nReasons: " + "; ".join(res.reasons)
        return (f"{verdict}\nParsed: {sig.to_dict()}"
                .replace("'raw_text'", "'raw'") + reasons)
