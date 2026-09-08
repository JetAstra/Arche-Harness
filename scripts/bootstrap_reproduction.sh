#!/usr/bin/env bash
set -euo pipefail

# 通用环境准备入口：创建或更新仓库标准 `.venv`，并安装 ARCHE、Codex SDK
# 与 MiniChem 依赖。脚本会写入本地环境，但不会启动模型、MCP 服务或研究任务。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDARD_ARCHE_PYTHON="$ROOT/.venv/bin/python"
ARCHE_PYTHON="$STANDARD_ARCHE_PYTHON"
TOOLBOX_ROOT="$ROOT/third_party/minichem_toolbox"
REQUIREMENTS="$ROOT/requirements.txt"
CODEX_SDK_ROOT="$ROOT/third_party/codex_python_sdk"
CODEX_BIN="$ROOT/third_party/codex_cli/0.149.0/linux-x64/codex"
CODEX_MANIFEST="$ROOT/third_party/codex_cli/0.149.0/manifest.json"

if [[ ! -f "$TOOLBOX_ROOT/pyproject.toml" ]]; then
  echo "MiniChem Toolbox 不存在：$TOOLBOX_ROOT" >&2
  exit 1
fi
if [[ ! -f "$CODEX_SDK_ROOT/pyproject.toml" || ! -x "$CODEX_BIN" || ! -f "$CODEX_MANIFEST" ]]; then
  echo "仓库随附的 Codex SDK、0.149.0 CLI 或运行时清单不完整：$CODEX_SDK_ROOT / $CODEX_BIN / $CODEX_MANIFEST" >&2
  exit 1
fi
if [[ ! -x "$ARCHE_PYTHON" ]]; then
  if [[ -n "${ARCHE_BOOTSTRAP_PYTHON:-}" ]]; then
    "$ARCHE_BOOTSTRAP_PYTHON" -m venv "$ROOT/.venv"
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$ROOT/.venv"
  elif command -v conda >/dev/null 2>&1; then
    conda create --yes --prefix "$ROOT/.venv" python=3.11 pip
  else
    echo "无法创建 $ROOT/.venv：需要 Python 3.11（可通过 ARCHE_BOOTSTRAP_PYTHON 指定）或 conda" >&2
    exit 1
  fi
fi
"$ARCHE_PYTHON" -m pip install -r "$REQUIREMENTS"
# SDK 源码来自官方 rust-v0.149.0 快照；--no-deps 防止安装其开发树声明的旧 CLI runtime。
# CodexConfig 会在 Harness 配置中显式绑定仓库内的 0.149.0 入口。
"$ARCHE_PYTHON" -m pip install --no-deps --no-build-isolation "$CODEX_SDK_ROOT"
ARCHE_BOOTSTRAP_EXPECTED_PYTHON="$ARCHE_PYTHON" "$ARCHE_PYTHON" - <<'PY'
import os
import sys
from importlib.metadata import version
from pathlib import Path

from openai_codex import CodexConfig

expected_python = Path(os.environ["ARCHE_BOOTSTRAP_EXPECTED_PYTHON"]).resolve()
if Path(sys.executable).resolve() != expected_python:
    raise SystemExit("Codex SDK 未安装到仓库标准 .venv")
if version("openai-codex") != "0.149.0+arche.1":
    raise SystemExit("Codex Python SDK 版本不是 0.149.0+arche.1")
if CodexConfig is None:
    raise SystemExit("Codex Python SDK 缺少 CodexConfig")
PY

if [[ "${ARCHE_SKIP_MINICHEM_BOOTSTRAP:-0}" != "1" ]]; then
  bash "$TOOLBOX_ROOT/scripts/bootstrap.sh"
fi

echo "统一 Python 环境: $ARCHE_PYTHON"
echo "Run ARCHE tests with: PYTHONPATH=src:third_party/minichem_toolbox/src $ARCHE_PYTHON -m pytest -q tests"
