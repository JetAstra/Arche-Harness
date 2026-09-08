"""验证当前 MiniChem 启动拓扑的私有路径和固定 profile 边界。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from minichem_mcp_tools.workspace import (
    ensure_private_runtime_directory,
    workspace_root,
)


def test_private_runtime_directory_tightens_runtime_and_target_modes(
    tmp_path: Path,
) -> None:
    """runtime 根和服务目录必须创建为 0700。"""

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o755)
    target = ensure_private_runtime_directory(
        runtime_root / "investigation",
        runtime_root=runtime_root,
    )

    assert target.is_dir()
    assert os.stat(runtime_root).st_mode & 0o777 == 0o700
    assert os.stat(target).st_mode & 0o777 == 0o700


def test_private_runtime_directory_rejects_runtime_or_target_symlink(
    tmp_path: Path,
) -> None:
    """runtime 根和任一目标组件出现 symlink 时必须 fail closed。"""

    real_runtime = tmp_path / "real-runtime"
    real_runtime.mkdir()
    runtime_link = tmp_path / "runtime-link"
    runtime_link.symlink_to(real_runtime, target_is_directory=True)
    with pytest.raises(ValueError, match="private runtime"):
        ensure_private_runtime_directory(
            runtime_link / "service",
            runtime_root=runtime_link,
        )

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target_link = runtime_root / "service"
    target_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="private runtime"):
        ensure_private_runtime_directory(target_link, runtime_root=runtime_root)


def test_private_runtime_directory_without_create_preserves_missing_target_error(
    tmp_path: Path,
) -> None:
    """只检查目录时，缺失末端仍返回标准 FileNotFoundError。"""

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    with pytest.raises(FileNotFoundError):
        ensure_private_runtime_directory(
            runtime_root / "missing",
            runtime_root=runtime_root,
            create=False,
        )


def test_private_runtime_directory_rejects_escape_and_non_private_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标必须位于 runtime 根内；workspace_root 不接受 0755 目录。"""

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    with pytest.raises(ValueError, match="private runtime"):
        ensure_private_runtime_directory(
            tmp_path / "outside",
            runtime_root=runtime_root,
        )

    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o755)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(workspace))
    monkeypatch.delenv("RESEARCHCHEMBENCH_WORKSPACE", raising=False)
    with pytest.raises(RuntimeError, match="private directory"):
        workspace_root()


def test_startup_scripts_keep_profile_transport_and_loopback_after_user_args() -> None:
    """专用入口必须把安全参数放在可覆盖参数之后。"""

    repository_root = Path(__file__).resolve().parents[3]
    investigation = (
        repository_root / "scripts" / "services" / "start_investigation_minichem.sh"
    ).read_text(encoding="utf-8")
    catalog = (
        repository_root / "scripts" / "services" / "start_catalog_bootstrap_mcp.sh"
    ).read_text(encoding="utf-8")

    for source, profile, host, port in (
        (
            investigation,
            "--profile investigation",
            '--host "$HOST"',
            '--port "$PORT"',
        ),
        (
            catalog,
            "--profile catalog_bootstrap",
            '--host "$HOST"',
            '--port "$PORT"',
        ),
    ):
        user_arguments = source.index('  "$@" \\\n')
        assert user_arguments < source.index(f"  {profile} \\\n")
        assert user_arguments < source.index("  --transport streamable_http \\\n")
        assert user_arguments < source.index(f"  {host} \\\n")
        assert user_arguments < source.index(f"  {port}\n")


def test_investigation_launcher_rejects_workspace_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    """专用入口发现 workspace symlink 时不得把帮助运行资料写到外部目录。"""

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    workspace_link = runtime_root / "workspace-link"
    workspace_link.symlink_to(outside, target_is_directory=True)
    config = tmp_path / "empty-config.env"
    config.write_text("", encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[3]
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "RESEARCHCHEMBENCH_CONFIG": str(config),
        "ARCHE_RUN_STORAGE_ROOT": str(runtime_root),
        "MINICHEM_SERVICE_WORKSPACE": str(workspace_link),
    }
    completed = subprocess.run(
        [
            "bash",
            str(
                repository_root
                / "scripts"
                / "services"
                / "start_investigation_minichem.sh"
            ),
            "--help",
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode != 0
    assert not any(outside.iterdir())


def test_end_to_end_entrypoint_defers_authenticated_readiness_to_investigation() -> None:
    """通用入口只冻结端点；带 scope 的 readiness 由 Investigation Session 执行。"""

    repository_root = Path(__file__).resolve().parents[3]
    source = (repository_root / "scripts" / "run_end_to_end.sh").read_text(
        encoding="utf-8"
    )
    for marker in (
        "EXPECTED_CATALOG_ENDPOINT=\"http://127.0.0.1:9011/mcp\"",
        "EXPECTED_INVESTIGATION_ENDPOINT=\"http://127.0.0.1:9012/mcp\"",
    ):
        assert marker in source
    for stale_marker in (
        "ClientSession",
        "session.initialize()",
        "session.list_tools()",
        "session.read_resource(\"minichem://catalog\")",
        "session.call_tool(\n                    \"list_action_domains\"",
        "CAPABILITY_HEADER",
        "catalog identity",
    ):
        assert stale_marker not in source
