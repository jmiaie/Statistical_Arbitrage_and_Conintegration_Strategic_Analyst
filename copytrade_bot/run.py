#!/usr/bin/env python3
"""Entry point for the Telegram copy-trading bot.

Usage:
    python run.py                  # connect to Telegram and run the bot
    python run.py --test "<alert>" # run one alert through the pipeline offline
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from copytradebot.config import Settings, StrategyConfig
from copytradebot.storage import Storage
from copytradebot.pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram copy-trading bot")
    parser.add_argument("--test", metavar="ALERT",
                        help="Process a single alert offline and exit")
    parser.add_argument("--log", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()

    if args.test:
        config = StrategyConfig.load(settings.filters_path)
        storage = Storage(settings.db_path)
        pipeline = Pipeline(config, settings, storage)
        decision = pipeline.process(args.test, source="cli-test")
        print(decision.summary())
        return 0

    if not settings.telegram_token:
        print("ERROR: TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env "
              "and fill it in.", file=sys.stderr)
        return 1

    from copytradebot.bot import CopyTradeBot
    CopyTradeBot(settings).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
