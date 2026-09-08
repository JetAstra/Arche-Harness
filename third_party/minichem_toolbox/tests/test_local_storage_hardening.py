"""验证 MiniChem 本地锁、原子写入和私有权限的并发安全边界。"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from minichem_mcp_tools import open_execution
from minichem_toolbox import resource_budget


def _configure_workspace(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """把资源和执行状态指向隔离目录，并设置最小可用预算。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(root))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "2")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "1024")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")


def _mode(path: Path) -> int:
    """读取文件或目录的 Unix 权限位。"""

    return stat.S_IMODE(path.stat().st_mode)


def test_gaussian_queue_lock_rejects_symlink_without_locking_external_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gaussian 队列锁是符号链接时必须失败，不能锁住外部文件。"""

    _configure_workspace(monkeypatch, tmp_path)
    queue_directory = tmp_path / "outputs" / ".resource_budget" / "gaussian_queue"
    queue_directory.mkdir(parents=True)
    external = tmp_path / "external.lock"
    external.write_text("sentinel", encoding="utf-8")
    (queue_directory / "queue.lock").symlink_to(external)

    with pytest.raises((OSError, RuntimeError, resource_budget.ResourceStateUnavailable)):
        with resource_budget._gaussian_queue_lock():
            pass
    assert external.read_text(encoding="utf-8") == "sentinel"


def test_operation_lock_rejects_symlink_without_locking_external_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """执行作业 operation 锁是符号链接时必须拒绝跟随。"""

    _configure_workspace(monkeypatch, tmp_path)
    job_root = tmp_path / "outputs" / "execution_jobs"
    job_root.mkdir(parents=True)
    operation_id = "operation-lock-symlink"
    lock_path = job_root / (
        "."
        + open_execution.hashlib.sha256(operation_id.encode("ascii")).hexdigest()
        + ".lock"
    )
    external = tmp_path / "external-operation.lock"
    external.write_text("sentinel", encoding="utf-8")
    lock_path.symlink_to(external)

    with pytest.raises((OSError, RuntimeError)):
        with open_execution._operation_lock(operation_id):
            pass
    assert external.read_text(encoding="utf-8") == "sentinel"


def test_operation_identity_rejects_symlink_receipt(tmp_path: Path) -> None:
    """恢复身份读取不得通过符号链接读取外部 operation receipt。"""

    external = tmp_path / "external-receipt.json"
    external.write_text('{"operation_id": "outside"}', encoding="utf-8")
    receipt = tmp_path / "operation_receipt.json"
    receipt.symlink_to(external)

    with pytest.raises(open_execution._OperationIdentityConflict):
        open_execution._operation_identity_evidence(
            tmp_path,
            {"job_type": "native_software", "job_id": "job_test"},
        )


def test_cancel_marker_rejects_symlink_target(tmp_path: Path) -> None:
    """取消标记不得通过符号链接修改外部文件。"""

    external = tmp_path / "external-cancel"
    external.write_text("sentinel", encoding="utf-8")
    marker = tmp_path / "cancel_requested"
    marker.symlink_to(external)

    with pytest.raises((OSError, RuntimeError)):
        open_execution._ensure_private_marker(marker)
    assert external.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize(
    "lock_name",
    [
        "quota.lock",
        "gaussian.serial.lock",
    ],
)
def test_resource_locks_reject_symlink_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_name: str,
) -> None:
    """资源预算和 Gaussian 串行锁都不得跟随外部符号链接。"""

    _configure_workspace(monkeypatch, tmp_path)
    budget_directory = tmp_path / "outputs" / ".resource_budget"
    budget_directory.mkdir(parents=True)
    external = tmp_path / f"external-{lock_name}"
    external.write_text("sentinel", encoding="utf-8")
    (budget_directory / lock_name).symlink_to(external)

    if lock_name == "quota.lock":
        lock_context = resource_budget._budget_lock()
    else:
        lock_context = resource_budget.gaussian_serial_gate()
    with pytest.raises((OSError, RuntimeError)):
        with lock_context:
            pass
    assert external.read_text(encoding="utf-8") == "sentinel"


def test_job_environment_uses_unique_private_scratch_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """提交环境与 supervisor 共用固定私有 scratch，避免随机目录泄漏。"""

    monkeypatch.setattr(open_execution, "runtime_environment", lambda _runtime: {})
    job_directory = tmp_path / "job"
    job_directory.mkdir()
    first = open_execution._job_environment(
        "core",
        "job-one",
        job_directory,
        {"cpu_cores": 1, "gpu_count": 0},
        {},
        job_type="native_software",
    )
    second = open_execution._job_environment(
        "core",
        "job-two",
        job_directory,
        {"cpu_cores": 1, "gpu_count": 0},
        {},
        job_type="native_software",
    )
    assert first["TMPDIR"] == second["TMPDIR"] == str(job_directory / ".tmp")
    assert Path(first["TMPDIR"]).name == ".tmp"
    assert Path(second["TMPDIR"]).name == ".tmp"
    assert _mode(Path(first["TMPDIR"])) == 0o700
    assert _mode(Path(second["TMPDIR"])) == 0o700
    assert _mode(Path(first["HOME"])) == 0o700
    assert _mode(Path(second["HOME"])) == 0o700


@pytest.mark.parametrize(
    "writer",
    [
        "resource",
        "execution",
    ],
)
def test_atomic_json_uses_unique_temp_fsyncs_and_private_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
) -> None:
    """原子 JSON 写入必须使用唯一临时文件、同步文件和父目录并收紧权限。"""

    _configure_workspace(monkeypatch, tmp_path)
    parent = tmp_path / writer / "nested"
    parent.mkdir(parents=True)
    os.chmod(parent, 0o755)
    target = parent / "record.json"
    replaced_from: list[Path] = []
    synced_descriptors: list[int] = []
    original_replace = os.replace
    original_fsync = os.fsync

    def record_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        **kwargs: object,
    ) -> None:
        replaced_from.append(Path(source))
        original_replace(source, destination, **kwargs)

    def record_fsync(descriptor: int) -> None:
        synced_descriptors.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "replace", record_replace)
    monkeypatch.setattr(os, "fsync", record_fsync)
    if writer == "resource":
        resource_budget._queue_atomic_write(target, {"value": 1})
    else:
        open_execution._atomic_json(target, {"value": 1})

    assert target.read_text(encoding="utf-8").strip() == '{\n  "value": 1\n}'
    assert replaced_from and replaced_from[0].name != "record.tmp"
    assert not replaced_from[0].exists()
    assert len(synced_descriptors) >= 2
    assert _mode(target) == 0o600
    assert _mode(parent) == 0o700
