#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_runtime
load_local_config
WORKSPACE="$(prepare_workspace codex)"
mkdir -p "$WORKSPACE/_sessions/codex_home"
if [[ -f "${CODEX_HOME:-$HOME/.codex}/auth.json" ]]; then
  cp "${CODEX_HOME:-$HOME/.codex}/auth.json" "$WORKSPACE/_sessions/codex_home/auth.json"
fi
export CODEX_HOME="$WORKSPACE/_sessions/codex_home"

set +e
codex exec \
  --skip-git-repo-check \
  -C "$WORKSPACE" \
  --sandbox workspace-write \
  --json \
  --output-last-message "$WORKSPACE/_sessions/final_message.md" \
  -c "mcp_servers.minichem_toolbox.command=\"$WORKSPACE/start_minichem_mcp.sh\"" \
  -c 'mcp_servers.minichem_toolbox.required=true' \
  -c 'mcp_servers.minichem_toolbox.startup_timeout_sec=60' \
  -c 'mcp_servers.minichem_toolbox.tool_timeout_sec=3600' \
  "$(cat "$WORKSPACE/INSTRUCTIONS.md")" \
  >"$WORKSPACE/_sessions/chat.jsonl" \
  2>"$WORKSPACE/_sessions/codex.stderr.log"
STATUS=$?
set -e
printf '%s\n' "$STATUS" >"$WORKSPACE/_sessions/exit_code.txt"
echo "Codex workspace: $WORKSPACE"
exit "$STATUS"
