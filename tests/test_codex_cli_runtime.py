"""验证仓库随附的 Codex CLI 运行时布局和归档完整性。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = REPOSITORY_ROOT / "third_party" / "codex_cli" / "0.149.0"
CLI_ENTRYPOINT = CLI_ROOT / "linux-x64" / "codex"
MANIFEST_PATH = CLI_ROOT / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_vendored_codex_runtime_materializes_verified_canonical_layout(tmp_path: Path) -> None:
    """入口应校验全部归档，并还原官方平台包的相对路径和 SHA-256。"""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["version"] == "0.149.0"
    assert manifest["target"] == "x86_64-unknown-linux-musl"
    assert manifest["runtime_entrypoint"].endswith("/bin/codex")

    package_manifest = manifest["package_manifest"]
    package_manifest_path = CLI_ROOT / package_manifest["path"]
    assert package_manifest_path.is_file()
    assert _sha256(package_manifest_path) == package_manifest["sha256"]

    for component in manifest["components"]:
        archive_path = CLI_ROOT / component["archive"]
        assert archive_path.is_file(), component["name"]
        assert _sha256(archive_path) == component["archive_sha256"]

    runtime_root = tmp_path / "codex-runtime"
    environment = os.environ.copy()
    environment["ARCHE_CODEX_RUNTIME_ROOT"] = str(runtime_root)
    completed = subprocess.run(
        [str(CLI_ENTRYPOINT), "--version"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "codex-cli 0.149.0"

    runtime_manifest = runtime_root / package_manifest["runtime_path"]
    assert runtime_manifest.is_file()
    assert _sha256(runtime_manifest) == package_manifest["sha256"]
    for component in manifest["components"]:
        runtime_path = runtime_root / component["runtime_path"]
        assert runtime_path.is_file(), component["name"]
        assert _sha256(runtime_path) == component["binary_sha256"]
        assert stat.S_IMODE(runtime_path.stat().st_mode) & stat.S_IXUSR
