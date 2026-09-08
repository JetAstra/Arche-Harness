#!/usr/bin/env bash
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLBOX_ROOT="$(cd "$HARNESS_DIR/../.." && pwd)"
REPOSITORY_ROOT="$(cd "$TOOLBOX_ROOT/../.." && pwd)"
DEFAULT_ARCHE_PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
RUNTIME_PYTHON="${ARCHE_PYTHON:-${MINICHEM_RUNTIME_PYTHON:-$DEFAULT_ARCHE_PYTHON}}"

load_local_config() {
  local candidate
  for candidate in "$TOOLBOX_ROOT/../../config.local.env" "$TOOLBOX_ROOT/config.local.env"; do
    if [[ -f "$candidate" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "$candidate"
      set +a
      break
    fi
  done
}

prepare_workspace() {
  local cli="$1"
  local stamp workspace model base_url
  stamp="$(date -u +'%Y%m%dT%H%M%SZ')"
  workspace="$TOOLBOX_ROOT/tests/results/${cli}/${stamp}_$$"
  model="${MINICHEM_AGENT_MODEL:-${OPENCODE_MODEL_VALUE:-deepseek/deepseek-v4-flash}}"
  [[ "$model" == */* ]] || model="deepseek/$model"
  base_url="${MINICHEM_AGENT_BASE_URL:-${OPENCODE_BASE_URL_VALUE:-https://api.deepseek.com/v1}}"
  "$RUNTIME_PYTHON" "$HARNESS_DIR/prepare_harness.py" \
    --toolbox-root "$TOOLBOX_ROOT" \
    --workspace "$workspace" \
    --model "$model" \
    --base-url "$base_url" >/dev/null
  printf '%s\n' "$workspace"
}

require_runtime() {
  if [[ ! -x "$RUNTIME_PYTHON" ]]; then
    echo "MiniChem Python 不可执行：$RUNTIME_PYTHON" >&2
    exit 1
  fi
}
