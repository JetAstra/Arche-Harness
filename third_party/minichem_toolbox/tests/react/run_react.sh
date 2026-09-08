#!/usr/bin/env bash
set -euo pipefail

REACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLBOX_ROOT="$(cd "$REACT_DIR/../.." && pwd)"
REPOSITORY_ROOT="$(cd "$TOOLBOX_ROOT/../.." && pwd)"
DEFAULT_ARCHE_PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
RUNTIME_PYTHON="${ARCHE_PYTHON:-${MINICHEM_RUNTIME_PYTHON:-$DEFAULT_ARCHE_PYTHON}}"

if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  echo "MiniChem Python 不可执行：$RUNTIME_PYTHON" >&2
  exit 1
fi

for candidate in "$TOOLBOX_ROOT/../../config.local.env" "$TOOLBOX_ROOT/config.local.env"; do
  if [[ -f "$candidate" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$candidate"
    set +a
    break
  fi
done

exec "$RUNTIME_PYTHON" "$REACT_DIR/react_agent.py" "$@"
