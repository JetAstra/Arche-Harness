"""Chemistry multi-agent package initialization."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Iterable


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _candidate_env_paths() -> Iterable[Path]:
    """按当前目录和仓库根目录依次查找本地配置文件。"""
    candidates = [Path.cwd() / "config.local.env"]
    try:
        candidates.append(Path(__file__).resolve().parents[2] / "config.local.env")
    except IndexError:
        pass

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        yield candidate


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    if value[0] in {"'", '"'}:
        try:
            parsed = shlex.split(value, comments=False, posix=True)
        except ValueError:
            return value.strip(value[0])
        return parsed[0] if parsed else ""

    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def load_env_files(*, override: bool = False) -> None:
    """Load KEY=VALUE pairs from local config files without requiring dotenv."""
    for env_path in _candidate_env_paths():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].lstrip()
            if "=" not in stripped:
                continue

            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            if not _ENV_KEY_RE.match(key):
                continue
            if override or key not in os.environ:
                os.environ[key] = _parse_env_value(raw_value)


load_env_files()
