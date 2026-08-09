#!/bin/bash
# SessionStart hook: install copytrade_bot dependencies so tests/linters run.
set -euo pipefail

# Only run in Claude Code on the web (remote) sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR/copytrade_bot"

# Runtime + dev dependencies (idempotent; install reuses the cached layer).
python3 -m pip install --quiet -r requirements.txt
python3 -m pip install --quiet pytest

# Make the package importable without installation.
echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR/copytrade_bot:\${PYTHONPATH:-}\"" >> "$CLAUDE_ENV_FILE"

echo "copytrade_bot dependencies ready."
