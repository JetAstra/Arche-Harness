#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOFTWARE_CACHE="$ROOT/.mini_software_cache"
STANDARD_ARCHE_PYTHON=".venv/bin/python"
PIP_SPEC="$ROOT/requirements.txt"
ASH_SOURCE="${MINICHEM_ASH_SOURCE:-$ROOT/vendor/ash-0.95}"
PYPI_INDEX="${MINICHEM_PYPI_INDEX_URL:-https://pypi.org/simple}"
PIP_ARGS=(--index-url "$PYPI_INDEX")

if [[ "${MINICHEM_PIP_OFFLINE:-0}" == "1" ]]; then
  PIP_ARGS=(--no-index)
fi

if [[ ! -x "$STANDARD_ARCHE_PYTHON" ]]; then
  echo "标准 Python 不可执行：$STANDARD_ARCHE_PYTHON；请先准备 arche 环境" >&2
  exit 1
fi
mkdir -p "$ROOT/.mini_model_cache"

# 工具箱与 ARCHE 共用标准环境；脚本只补充依赖，不创建新的 Python 环境。
"$STANDARD_ARCHE_PYTHON" -m pip install \
  "${PIP_ARGS[@]}" \
  -r "$PIP_SPEC"
if [[ ! -d "$ASH_SOURCE/ash" ]]; then
  echo "ASH source is missing: $ASH_SOURCE" >&2
  echo "Restore vendor/ash-0.95 or set MINICHEM_ASH_SOURCE explicitly." >&2
  exit 1
fi
TEMP_ROOT="${TMPDIR:-$ROOT/.tmp}"
mkdir -p "$TEMP_ROOT"
ASH_BUILD_ROOT="$(mktemp -d "$TEMP_ROOT/minichem-ash-build.XXXXXX")"
cleanup_ash_build() {
  rm -r "$ASH_BUILD_ROOT"
}
trap cleanup_ash_build EXIT
mkdir -p "$ASH_BUILD_ROOT/source"
cp -a "$ASH_SOURCE/." "$ASH_BUILD_ROOT/source/"
"$STANDARD_ARCHE_PYTHON" -m pip install \
  "${PIP_ARGS[@]}" \
  --no-build-isolation \
  --no-deps \
  "$ASH_BUILD_ROOT/source"
"$STANDARD_ARCHE_PYTHON" -m pip install \
  "${PIP_ARGS[@]}" \
  --no-build-isolation \
  --no-deps \
  -e "$ROOT"

mkdir -p "$SOFTWARE_CACHE/gaussian/g16/scratch"
chmod 700 "$SOFTWARE_CACHE/gaussian/g16/scratch" 2>/dev/null || true

MINICHEM_HOME="$ROOT" "$STANDARD_ARCHE_PYTHON" "$ROOT/scripts/verify_minichem.py"
MINICHEM_ASH_SOURCE="$ASH_SOURCE" "$ROOT/scripts/export_environment_lock.sh"
echo "Unified MiniChem/FairChem/ASH runtime is ready: $STANDARD_ARCHE_PYTHON"
