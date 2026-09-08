#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARCHE_PYTHON="$ROOT/../../.venv/bin/python"
ARCHE_PYTHON="${MINICHEM_RUNTIME_PYTHON:-${ARCHE_PYTHON:-$DEFAULT_ARCHE_PYTHON}}"
RUNTIME="$(cd "$(dirname "$ARCHE_PYTHON")/.." && pwd)"
LOCK_ROOT="$ROOT/environment-lock"
ASH_SOURCE="${MINICHEM_ASH_SOURCE:-$ROOT/vendor/ash-0.95}"

if [[ "$ARCHE_PYTHON" != "$DEFAULT_ARCHE_PYTHON" ]]; then
  echo "环境锁定必须使用标准 Python：$DEFAULT_ARCHE_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$ARCHE_PYTHON" ]]; then
  echo "ARCHE_PYTHON 不可执行：$ARCHE_PYTHON" >&2
  exit 1
fi

MANAGER="$(command -v conda || command -v mamba || command -v micromamba || true)"
if [[ -z "$MANAGER" ]]; then
  echo "conda, mamba, or micromamba is required to export the conda lock" >&2
  exit 1
fi

mkdir -p "$LOCK_ROOT"
"$MANAGER" list -p "$RUNTIME" --explicit > "$LOCK_ROOT/conda-linux-64.lock.txt"
"$MANAGER" list -p "$RUNTIME" --json \
  | "$ARCHE_PYTHON" -c '
import json
import sys

excluded = {"ash", "minichem-mcp-tools"}
packages = json.load(sys.stdin)
lines = sorted(
    "{}=={}".format(item["name"], item["version"])
    for item in packages
    if item.get("channel") == "pypi" and item["name"].casefold() not in excluded
)
print("\n".join(lines))
' > "$LOCK_ROOT/pip.lock.txt"
MINICHEM_ASH_SOURCE="$ASH_SOURCE" MINICHEM_TOOLBOX_ROOT="$ROOT" \
  "$ARCHE_PYTHON" - <<'PY' > "$LOCK_ROOT/runtime.json"
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

names = ("ash", "ase", "fairchem-core", "numpy", "rdkit", "rxnmapper", "torch")
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None

root = Path(os.environ["MINICHEM_TOOLBOX_ROOT"]).resolve()
ash_source = Path(os.environ["MINICHEM_ASH_SOURCE"]).resolve()
ash_files = (
    "ash/knarr/KNARRatom/utilities.py",
    "ash/interfaces/interface_fairchem.py",
    "pyproject.toml",
)
ash_hashes = {}
for relative in ash_files:
    path = ash_source / relative
    ash_hashes[relative] = (
        hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    )
print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": versions,
    "ash_source_policy": "unchanged repository-local vendor/ash-0.95 source installed without dependencies",
    "ash_source_path": ash_source.relative_to(root).as_posix(),
    "ash_source_sha256": ash_hashes,
}, indent=2, sort_keys=True))
PY

echo "Environment locks written to $LOCK_ROOT"
