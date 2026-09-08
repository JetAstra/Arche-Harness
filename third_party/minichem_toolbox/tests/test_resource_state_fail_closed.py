"""验证 MiniChem 持久资源、幂等收据和进程身份损坏时的安全失败。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from minichem_mcp_tools import job_supervisor, open_execution
from minichem_mcp_tools.execution_models import JobCancelRequest
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
)
from minichem_toolbox.process_identity import (
    ProcessIdentity,
    capture_process_identity,
    process_identity_activity,
)
from minichem_toolbox import resource_budget
from minichem_toolbox.resource_budget import (
    ResourceStateUnavailable,
    active_resource_usage,
    enqueue_gaussian_job,
    read_gaussian_queue_snapshot,
    reserve_resources,
    wait_for_gaussian_turn,
)
from minichem_toolbox.scope import ScopeCapabilityRegistry, activate_scope


def _configure_workspace(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """把资源状态指向隔离测试工作区。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(root))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "4")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")


def test_corrupt_predecessor_ticket_blocks_snapshot_enqueue_and_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """坏前序 ticket 不得被跳过，后续 ticket 不能取得队首。"""

    _configure_workspace(monkeypatch, tmp_path)
    first = enqueue_gaussian_job("job-first", kind="native", job_id="job-first")
    second = enqueue_gaussian_job("job-second", kind="native", job_id="job-second")
    queue_path = (
        tmp_path
        / "outputs"
        / ".resource_budget"
        / "gaussian_queue"
        / f"ticket_{first.ticket_id:020d}.json"
    )
    queue_path.write_text("{broken", encoding="utf-8")

    snapshot = read_gaussian_queue_snapshot()
    assert snapshot == {
        "status": "unavailable",
        "scope": "global",
        "items": [],
        "item_count": 0,
    }
    with pytest.raises(ResourceStateUnavailable):
        enqueue_gaussian_job("job-third", kind="native", job_id="job-third")
    with pytest.raises(ResourceStateUnavailable):
        wait_for_gaussian_turn(second, poll_seconds=0.01)


@pytest.mark.parametrize("entry_name", ["ticket_00000000000000000001.json.tmp", "unknown.dat"])
def test_unknown_or_temporary_queue_entry_blocks_new_gaussian(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_name: str,
) -> None:
    """FIFO 目录中的临时或未知条目不能被扫描器忽略。"""

    _configure_workspace(monkeypatch, tmp_path)
    queue_directory = tmp_path / "outputs" / ".resource_budget" / "gaussian_queue"
    queue_directory.mkdir(parents=True)
    (queue_directory / entry_name).write_text("incomplete", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable):
        enqueue_gaussian_job("job-blocked", kind="native", job_id="job-blocked")


def test_corrupt_reservation_blocks_resource_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """损坏 reservation 不能按零资源继续创建新 reservation。"""

    _configure_workspace(monkeypatch, tmp_path)
    reservations = tmp_path / "outputs" / ".resource_budget" / "reservations"
    reservations.mkdir(parents=True)
    (reservations / "reservation_bad.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable):
        active_resource_usage()
    with pytest.raises(ResourceStateUnavailable):
        reserve_resources(
            {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            kind="native_software",
            label="test",
        )


def test_corrupt_active_status_blocks_resource_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """损坏 active status 不能从预算扫描中静默消失。"""

    _configure_workspace(monkeypatch, tmp_path)
    job_directory = tmp_path / "outputs" / "execution_jobs" / "job_bad_status"
    job_directory.mkdir(parents=True)
    (job_directory / "status.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable):
        reserve_resources(
            {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            kind="native_software",
            label="test",
        )


def _scopes(tmp_path: Path):
    """创建两个可独立激活的 capability scope。"""

    registry = ScopeCapabilityRegistry(tmp_path / "registry")
    values = []
    for index in (1, 2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir()
        capability = registry.provision_scope_capability(
            task_id=f"task-{index}",
            batch_id=f"batch-{index}",
            combination_id=f"combination-{index}",
            scope_id=f"scope-{index}",
            workspace=workspace,
            token=("A" if index == 1 else "B") * 48,
        )
        values.append(capability.scope)
    return values


def test_completed_receipt_copied_from_foreign_scope_cannot_replay(
    tmp_path: Path,
) -> None:
    """复制 completed receipt 到另一个 workspace 后不能重放副作用。"""

    first_scope, second_scope = _scopes(tmp_path)
    operation_id = "copied-completed-operation"
    fingerprint = calculate_request_fingerprint("test", {"value": 1})
    calls: list[int] = []
    with activate_scope(first_scope):
        first_store = OperationIdempotencyStore()
        assert first_store.execute_idempotent_operation(
            operation_id,
            fingerprint,
            lambda: calls.append(1) or {"status": "success", "count": len(calls)},
        ) == {"status": "success", "count": 1}
        source = first_store.root / (
            hashlib.sha256(operation_id.encode("ascii")).hexdigest() + ".json"
        )
    with activate_scope(second_scope):
        second_store = OperationIdempotencyStore()
        target = second_store.root / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        replay = second_store.execute_idempotent_operation(
            operation_id,
            fingerprint,
            lambda: calls.append(2) or {"status": "unexpected"},
        )
    assert replay["status"] == "technical_pause"
    assert replay["error"]["code"] == "operation_scope_mismatch"
    assert calls == [1]


def test_legacy_unscoped_completed_receipt_remains_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有认证 scope 的旧 completed receipt 仍可按原协议回放。"""

    monkeypatch.delenv("MINICHEM_SCOPE_ID", raising=False)
    monkeypatch.delenv("MINICHEM_SCOPE_WORKSPACE", raising=False)
    store = OperationIdempotencyStore(tmp_path / "legacy-operations")
    fingerprint = calculate_request_fingerprint("legacy", {"value": 1})
    first = store.execute_idempotent_operation(
        "legacy-operation",
        fingerprint,
        lambda: {"status": "success", "legacy": True},
    )
    second = store.execute_idempotent_operation(
        "legacy-operation",
        fingerprint,
        lambda: {"status": "unexpected"},
    )
    assert second == first


def test_start_ticks_alias_is_compatible_but_partial_identity_stays_unknown() -> None:
    """旧 start_ticks 可读，但缺少 boot_id 时不能回退成仅 PID 存活判断。"""

    identity = capture_process_identity(open_execution.os.getpid())
    assert identity.start_time_ticks is not None
    assert identity.boot_id
    assert process_identity_activity(
        {
            "pid": identity.pid,
            "start_ticks": identity.start_time_ticks,
            "boot_id": identity.boot_id,
        }
    ) == "alive"
    assert resource_budget._persisted_process_activity(
        {
            "supervisor_pid": identity.pid,
            "supervisor_start_ticks": identity.start_time_ticks,
            "supervisor_boot_id": identity.boot_id,
        },
        "supervisor",
    ) == "alive"
    assert resource_budget._persisted_process_activity(
        {
            "supervisor_pid": identity.pid,
            "supervisor_start_ticks": identity.start_time_ticks,
        },
        "supervisor",
    ) == "unknown"


def test_cancel_rejects_reused_supervisor_pid_without_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """记录的 start ticks 不匹配时取消不得向复用 PID 发 SIGTERM。"""

    _configure_workspace(monkeypatch, tmp_path)
    job_id = "job_0123456789abcdef0123456789abcdef"
    directory = tmp_path / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    current = open_execution.capture_process_identity(open_execution.os.getpid())
    (directory / "status.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "running",
                "supervisor_pid": current.pid,
                "supervisor_start_time_ticks": (current.start_time_ticks or 1) + 1,
                "supervisor_boot_id": current.boot_id,
            }
        ),
        encoding="utf-8",
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(open_execution.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    result = open_execution.cancel_execution_job(JobCancelRequest(job_id=job_id))
    assert result["status"] == "failed"
    assert result["cancellation_sent"] is False
    assert result["error"]["code"] == "supervisor_identity_unavailable"
    assert signals == []


def test_unverified_child_identity_has_bounded_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """child 身份持续不可读时 supervisor 不得无限等待或猜测信号。"""

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    monkeypatch.setattr(
        job_supervisor,
        "capture_process_identity",
        lambda pid: ProcessIdentity(pid=pid, start_time_ticks=None, boot_id=None),
    )
    started = time.monotonic()
    try:
        assert job_supervisor._finish_unverified_child_safely(child) is False
        assert time.monotonic() - started < 4
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_unverified_child_identity_with_fork_descendant_keeps_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """child 身份不可读且 leader fork 后退出时不能仅按 poll 释放状态。"""

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, time; "
                "descendant = os.fork(); "
                "time.sleep(0.15 if descendant else 0.6); "
                "os._exit(0)"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    monkeypatch.setattr(
        job_supervisor,
        "capture_process_identity",
        lambda pid: ProcessIdentity(pid=pid, start_time_ticks=None, boot_id=None),
    )
    started = time.monotonic()
    assert job_supervisor._finish_unverified_child_safely(child) is False
    assert time.monotonic() - started < 4
    child.wait(timeout=5)
    # descendant 仍可能在 leader wait 返回后短暂存在；等待其自然退出，测试不
    # 使用未经核对的 PID/PGID 发信号作为清理手段。
    time.sleep(0.7)


def test_supervisor_child_identity_failure_returns_without_infinite_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 supervisor 分支在 child 身份不可读时有界失败且不猜测 PID。"""

    job_directory = tmp_path / "job-child-identity"
    job_directory.mkdir()
    status_path = job_directory / "status.json"
    stdout_path = job_directory / "stdout.log"
    stderr_path = job_directory / "stderr.log"
    specification_path = job_directory / "supervisor_spec.json"
    specification_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": "job-child-identity",
                "job_type": "native_software",
                "runtime": "core",
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "job_directory": str(job_directory),
                "relative_job_directory": "job-child-identity",
                "status_path": str(status_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "relative_stdout_path": "job-child-identity/stdout.log",
                "relative_stderr_path": "job-child-identity/stderr.log",
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 128,
                    "gpu_count": 0,
                    "walltime_seconds": 30,
                },
                "resource_allocation": {},
                "evaluation_resource_budget": {},
                "metadata": {"execution_timeout_policy": {"execution_class": "fast"}},
                "submitted_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    real_capture = job_supervisor.capture_process_identity

    def capture_with_unreadable_child(pid: int) -> ProcessIdentity:
        if pid == os.getpid():
            return real_capture(pid)
        return ProcessIdentity(pid=pid, start_time_ticks=None, boot_id=None)

    monkeypatch.setattr(
        job_supervisor,
        "capture_process_identity",
        capture_with_unreadable_child,
    )
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    started = time.monotonic()
    child_pid: int | None = None
    try:
        assert job_supervisor._run_supervised_job(specification_path) == 1
        assert time.monotonic() - started < 4
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["status"] == "running"
        assert status["supervisor_exit_pending"] is True
        assert status["error"]["code"] == "child_identity_unavailable"
        assert status.get("child_process_group_id") is None
        assert status.get("child_session_id") is None
        child_pid = status.get("child_pid")
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        if isinstance(child_pid, int) and child_pid > 0:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_supervisor_sigkill_failure_has_bounded_child_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGKILL 核对失败时 child.wait 也必须有界，且不能向未知 PID 发信号。"""

    class NeverExitChild:
        pid = 123
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout):
            timeouts.append(timeout)
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)

    timeouts: list[float] = []
    signals: list[object] = []
    monkeypatch.setattr(
        job_supervisor,
        "_verified_process_group_signal",
        lambda identity, received_signal: signals.append(
            (identity, received_signal)
        )
        or False,
    )
    started = time.monotonic()
    assert (
        job_supervisor._wait_for_child_reap_safely(
            NeverExitChild(),
            ProcessIdentity(pid=123, start_time_ticks=456, boot_id="boot"),
        )
        is False
    )
    assert time.monotonic() - started < 1
    assert timeouts == [5, 1]
    assert signals == [
        (ProcessIdentity(pid=123, start_time_ticks=456, boot_id="boot"), signal.SIGKILL)
    ]


def test_process_group_signal_requires_matching_pgid_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PGID 或 session 不匹配时不得向潜在复用的进程组发信号。"""

    current = capture_process_identity(os.getpid())
    assert current.process_group_id is not None
    assert current.session_id is not None
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        job_supervisor.os,
        "killpg",
        lambda group_id, received_signal: signals.append(
            (group_id, received_signal)
        ),
    )
    mismatched_group = ProcessIdentity(
        pid=current.pid,
        start_time_ticks=current.start_time_ticks,
        boot_id=current.boot_id,
        process_group_id=current.process_group_id + 1,
        session_id=current.session_id,
    )
    missing_group = ProcessIdentity(
        pid=current.pid,
        start_time_ticks=current.start_time_ticks,
        boot_id=current.boot_id,
    )
    assert not job_supervisor._verified_process_group_signal(
        mismatched_group,
        signal.SIGTERM,
    )
    assert not job_supervisor._verified_process_group_signal(
        missing_group,
        signal.SIGTERM,
    )
    assert signals == []
