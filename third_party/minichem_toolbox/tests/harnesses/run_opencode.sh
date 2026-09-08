#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_runtime
load_local_config
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for the OpenCode harness}"

WORKSPACE="$(prepare_workspace opencode)"
MODEL="${MINICHEM_AGENT_MODEL:-${OPENCODE_MODEL_VALUE:-deepseek/deepseek-v4-flash}}"
[[ "$MODEL" == */* ]] || MODEL="deepseek/$MODEL"

export OPENCODE_DB="$WORKSPACE/_sessions/opencode.db"
export OPENCODE_ROOT="$WORKSPACE/_sessions/opencode_root"
unset OPENCODE_WORKSPACE_ID

set +e
opencode run \
  --pure \
  --auto \
  --dir "$WORKSPACE" \
  --model "$MODEL" \
  --format json \
  --title "MiniChem three-layer test" \
  "$(cat "$WORKSPACE/INSTRUCTIONS.md")" \
  >"$WORKSPACE/_sessions/chat.jsonl" \
  2>"$WORKSPACE/_sessions/opencode.stderr.log"
STATUS=$?
set -e
printf '%s\n' "$STATUS" >"$WORKSPACE/_sessions/exit_code.txt"
echo "OpenCode workspace: $WORKSPACE"
exit "$STATUS"
