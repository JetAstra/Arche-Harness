#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_runtime
load_local_config
WORKSPACE="$(prepare_workspace claude)"
mkdir -p "$WORKSPACE/_sessions/claude_home"
if [[ -f "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.credentials.json" ]]; then
  cp "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.credentials.json" "$WORKSPACE/_sessions/claude_home/.credentials.json"
fi
export CLAUDE_CONFIG_DIR="$WORKSPACE/_sessions/claude_home"

set +e
claude -p \
  --strict-mcp-config \
  --mcp-config "$WORKSPACE/.mcp.json" \
  --output-format stream-json \
  --verbose \
  --max-turns 80 \
  --tools "Read,Write,Edit" \
  --allowedTools "Read,Write,Edit,mcp__minichem_toolbox__*" \
  --permission-mode dontAsk \
  --disable-slash-commands \
  --no-session-persistence \
  "$(cat "$WORKSPACE/INSTRUCTIONS.md")" \
  >"$WORKSPACE/_sessions/chat.jsonl" \
  2>"$WORKSPACE/_sessions/claude.stderr.log"
STATUS=$?
set -e
printf '%s\n' "$STATUS" >"$WORKSPACE/_sessions/exit_code.txt"
echo "Claude workspace: $WORKSPACE"
exit "$STATUS"
