#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "$ROOT/../.." && pwd)"
STANDARD_ARCHE_PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
PYTHON="${MINICHEM_RUNTIME_PYTHON:-${ARCHE_PYTHON:-$STANDARD_ARCHE_PYTHON}}"

if [[ "$PYTHON" != "$STANDARD_ARCHE_PYTHON" ]]; then
  echo "MiniChem 必须使用标准 Python：$STANDARD_ARCHE_PYTHON" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "标准 Python 不可执行：$PYTHON；请先准备 arche 环境" >&2
  exit 1
fi

export MINICHEM_HOME="$ROOT"
export ARCHE_PYTHON="$PYTHON"
export MINICHEM_RUNTIME_PYTHON="$PYTHON"
export PYTHONPATH="$REPOSITORY_ROOT/src:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m minichem_mcp_tools.server "$@"
