"""Gaussian 服务级 FIFO 队列的并发、交接和释放测试。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from minichem_mcp_tools import job_supervisor
from minichem_toolbox import process_identity, runtime
from minichem_toolbox.process_identity import (
    capture_process_identity,
    inspect_process_group_membership,
)
from minichem_toolbox.resource_budget import (
    active_resource_jobs,
    active_resource_usage,
    bind_gaussian_queue_owner,
    enqueue_gaussian_job,
    gaussian_execution_slot,
    read_gaussian_queue_snapshot,
    wait_for_gaussian_turn,
)


def _configure_workspace(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """把服务内部队列和 reservation 指向当前测试工作区。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(root))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "4")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")


def _write_typed_recovery_records(
    root: Path,
    *,
    worker: subprocess.Popen[bytes],
    owner_pid: int = 99_999_999,
    worker_start_time_ticks: int | None = None,
) -> tuple[Path, Path]:
    """写入服务崩溃后留下的 typed Gaussian FIFO 和 reservation 记录。"""

    identity = capture_process_identity(worker.pid).as_dict()
    if worker_start_time_ticks is not None:
        identity["start_time_ticks"] = worker_start_time_ticks
    queue_directory = root / "outputs" / ".resource_budget" / "gaussian_queue"
    queue_directory.mkdir(parents=True)
    (queue_directory / "next_ticket").write_text("1\n", encoding="ascii")
    queue_path = queue_directory / "ticket_00000000000000000001.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ticket_id": 1,
                "owner_id": "typed-crashed",
                "owner_pid": owner_pid,
                "kind": "typed",
                "job_id": None,
                "scope_id": None,
                "workspace": str(root),
                "state": "running",
                "created_at": "2026-01-01T00:00:00+00:00",
                "worker_pid": identity["pid"],
                "worker_start_time_ticks": identity["start_time_ticks"],
                "worker_boot_id": identity["boot_id"],
                "worker_status": "running",
            }
        ),
        encoding="utf-8",
    )
    reservation_directory = root / "outputs" / ".resource_budget" / "reservations"
    reservation_directory.mkdir(parents=True)
    reservation_path = reservation_directory / "reservation_typed_crashed.json"
    reservation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": owner_pid,
                "kind": "predefined_action",
                "label": "calculate_hessian/gaussian",
                "job_id": None,
                "workspace": str(root),
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "gpu_count": 0,
                },
                "resource_allocation": {
                    "cpu_ids": sorted(os.sched_getaffinity(0))[:1],
                    "gpu_ids": [],
                },
                "worker_tracking": "gaussian_typed",
                "worker_pid": identity["pid"],
                "worker_start_time_ticks": identity["start_time_ticks"],
                "worker_boot_id": identity["boot_id"],
                "worker_status": "running",
            }
        ),
        encoding="utf-8",
    )
    return queue_path, reservation_path


def test_unreadable_proc_identity_remains_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID 仍存在但 proc 身份不可读时不能把旧 Gaussian 当成已停止。"""

    monkeypatch.setattr(process_identity, "_read_proc_stat", lambda _pid: None)
    monkeypatch.setattr(process_identity.os, "kill", lambda _pid, _signal: None)

    assert process_identity.process_identity_activity(
        {"pid": 123, "start_time_ticks": 456, "boot_id": "boot"}
    ) == "unknown"


def test_missing_pid_identity_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """proc 不可读且内核明确报告 PID 不存在时才允许释放旧记录。"""

    monkeypatch.setattr(process_identity, "_read_proc_stat", lambda _pid: None)

    def report_missing(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(process_identity.os, "kill", report_missing)
    assert process_identity.process_identity_activity(
        {"pid": 123, "start_time_ticks": 456, "boot_id": "boot"}
    ) == "stopped"


def test_typed_gaussian_recovery_keeps_fifo_and_resources_for_live_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务 PID 已死时，仍存活的 detached worker 继续占住 FIFO 和资源。"""

    _configure_workspace(monkeypatch, tmp_path)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    queue_path, reservation_path = _write_typed_recovery_records(
        tmp_path,
        worker=worker,
    )
    waiter: threading.Thread | None = None
    try:
        next_ticket = enqueue_gaussian_job("typed-next", kind="typed")
        bind_gaussian_queue_owner(next_ticket)
        acquired: list[bool] = []
        waiter = threading.Thread(
            target=lambda: acquired.append(wait_for_gaussian_turn(next_ticket, poll_seconds=0.01))
        )
        waiter.start()
        time.sleep(0.15)
        assert waiter.is_alive()
        assert queue_path.is_file()
        assert reservation_path.is_file()
        assert active_resource_usage() == {
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
        }
        worker.terminate()
        worker.wait(timeout=5)
        waiter.join(timeout=5)
        assert acquired == [True]
        next_ticket.release()
        assert active_resource_usage() == {
            "cpu_cores": 0,
            "memory_mb": 0,
            "gpu_count": 0,
        }
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        if waiter is not None:
            waiter.join(timeout=5)


def test_typed_gaussian_recovery_does_not_trust_reused_worker_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker PID 被复用时，启动时钟不匹配即可判定旧记录已经失效。"""

    _configure_workspace(monkeypatch, tmp_path)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        start_time_ticks = capture_process_identity(worker.pid).start_time_ticks
        assert start_time_ticks is not None
        queue_path, reservation_path = _write_typed_recovery_records(
            tmp_path,
            worker=worker,
            worker_start_time_ticks=start_time_ticks + 1,
        )
        enqueue_gaussian_job("typed-after-pid-reuse", kind="typed")
        assert not queue_path.exists()
        assert active_resource_usage() == {
            "cpu_cores": 0,
            "memory_mb": 0,
            "gpu_count": 0,
        }
        assert not reservation_path.exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_legacy_typed_gaussian_records_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有 worker 身份的旧 typed Gaussian 记录不能按已退出的持有进程 PID 清理。"""

    _configure_workspace(monkeypatch, tmp_path)
    queue_directory = tmp_path / "outputs" / ".resource_budget" / "gaussian_queue"
    queue_directory.mkdir(parents=True)
    (queue_directory / "next_ticket").write_text("1\n", encoding="ascii")
    queue_path = queue_directory / "ticket_00000000000000000001.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ticket_id": 1,
                "owner_id": "legacy-typed",
                "owner_pid": 99_999_999,
                "kind": "typed",
                "state": "running",
                "workspace": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    reservation_directory = tmp_path / "outputs" / ".resource_budget" / "reservations"
    reservation_directory.mkdir(parents=True)
    reservation_path = reservation_directory / "reservation_legacy_typed.json"
    reservation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": 99_999_999,
                "kind": "predefined_action",
                "label": "calculate_hessian/gaussian",
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "gpu_count": 0,
                },
                "resource_allocation": {
                    "cpu_ids": sorted(os.sched_getaffinity(0))[:1],
                    "gpu_ids": [],
                },
            }
        ),
        encoding="utf-8",
    )

    enqueue_gaussian_job("typed-after-legacy", kind="typed")
    assert queue_path.is_file()
    assert reservation_path.is_file()
    assert active_resource_usage() == {
        "cpu_cores": 1,
        "memory_mb": 256,
        "gpu_count": 0,
    }


def test_legacy_non_gaussian_typed_reservation_keeps_pid_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 non-Gaussian typed reservation 仍按原 service PID 规则清理。"""

    _configure_workspace(monkeypatch, tmp_path)
    reservation_directory = tmp_path / "outputs" / ".resource_budget" / "reservations"
    reservation_directory.mkdir(parents=True)
    reservation_path = reservation_directory / "reservation_legacy_xtb.json"
    reservation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": 99_999_999,
                "kind": "predefined_action",
                "label": "optimize_geometry/xtb",
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "gpu_count": 0,
                },
                "resource_allocation": {
                    "cpu_ids": sorted(os.sched_getaffinity(0))[:1],
                    "gpu_ids": [],
                },
            }
        ),
        encoding="utf-8",
    )

    assert active_resource_usage() == {
        "cpu_cores": 0,
        "memory_mb": 0,
        "gpu_count": 0,
    }
    assert not reservation_path.exists()


def test_runtime_persists_worker_identity_before_waiting_for_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runtime 在 communicate 前交付 worker 身份，覆盖服务崩溃交接窗口。"""

    python = tmp_path / "python"
    python.touch()
    observed: list[object] = []
    callback_seen_by_communicate: list[bool] = []

    class FakeProcess:
        pid = os.getpid()
        returncode = 0

        def communicate(self, _payload: str, timeout: int):
            callback_seen_by_communicate.append(bool(observed))
            assert timeout == 5
            return '{"status": "success"}', ""

    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: python)
    monkeypatch.setattr(runtime, "runtime_environment", lambda _runtime: {})
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    result = runtime.invoke_worker(
        runtime="gaussian",
        payload={"request": {}},
        timeout_seconds=5,
        on_worker_started=observed.append,
    )

    assert result["status"] == "success"
    assert callback_seen_by_communicate == [True]
    identity = observed[0]
    assert identity.pid == os.getpid()
    assert identity.start_time_ticks is not None


def test_gaussian_queue_is_fifo_across_typed_and_native_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """typed 与 native 条目共享编号，后者不能越过前者取得队首。"""

    _configure_workspace(monkeypatch, tmp_path)
    typed_ticket = enqueue_gaussian_job("typed-first", kind="typed")
    native_ticket = enqueue_gaussian_job(
        "job-native-second",
        kind="native",
        job_id="job-native-second",
    )
    bind_gaussian_queue_owner(native_ticket)

    acquired: list[str] = []
    typed_ready = threading.Event()
    release_typed = threading.Event()

    def hold_typed_slot() -> None:
        assert wait_for_gaussian_turn(typed_ticket)
        acquired.append("typed")
        typed_ready.set()
        assert release_typed.wait(timeout=5)
        typed_ticket.release()

    def wait_native_slot() -> None:
        assert wait_for_gaussian_turn(native_ticket)
        acquired.append("native")
        native_ticket.release()

    typed_thread = threading.Thread(target=hold_typed_slot)
    native_thread = threading.Thread(target=wait_native_slot)
    typed_thread.start()
    assert typed_ready.wait(timeout=5)
    native_thread.start()
    time.sleep(0.15)
    assert acquired == ["typed"]
    release_typed.set()
    typed_thread.join(timeout=5)
    native_thread.join(timeout=5)
    assert not typed_thread.is_alive()
    assert not native_thread.is_alive()
    assert acquired == ["typed", "native"]


def test_gaussian_queue_snapshot_is_ordered_and_scope_filtered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未认证 legacy 请求只返回位置、状态和 owned=false，不暴露作业身份。"""

    _configure_workspace(monkeypatch, tmp_path)
    first = enqueue_gaussian_job("secret-owner-first", kind="typed")
    second = enqueue_gaussian_job(
        "job_second_private_identity",
        kind="native",
        job_id="job_second_private_identity",
    )
    bind_gaussian_queue_owner(first)
    bind_gaussian_queue_owner(second)
    assert wait_for_gaussian_turn(first)

    snapshot = read_gaussian_queue_snapshot()

    assert snapshot["status"] == "available"
    assert snapshot["scope"] == "global"
    assert snapshot["item_count"] == 2
    assert [item["position"] for item in snapshot["items"]] == [1, 2]
    assert [item["status"] for item in snapshot["items"]] == ["running", "queued"]
    assert all(
        item == {"position": item["position"], "status": item["status"], "owned": False}
        for item in snapshot["items"]
    )
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "secret-owner-first" not in serialized
    assert "job_second_private_identity" not in serialized
    assert "/" not in serialized

    first.release()
    second.release()


def test_native_pending_ticket_survives_submitter_exit_before_supervisor_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """提交进程退出时，尚未绑定 supervisor 的 native ticket 仍可被接管。"""

    _configure_workspace(monkeypatch, tmp_path)
    source_root = Path(__file__).resolve().parents[1] / "src"
    child_code = (
        "from minichem_toolbox.resource_budget import enqueue_gaussian_job; "
        "enqueue_gaussian_job('job-handoff', kind='native', job_id='job-handoff')"
    )
    child_environment = {
        **os.environ,
        "MINICHEM_MCP_WORKSPACE": str(tmp_path),
        "PYTHONPATH": str(source_root),
    }
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    job_directory = tmp_path / "outputs" / "execution_jobs" / "job-handoff"
    job_directory.mkdir(parents=True)
    (job_directory / "supervisor_spec.json").write_text("{}", encoding="utf-8")
    ticket = enqueue_gaussian_job(
        "job-handoff",
        kind="native",
        job_id="job-handoff",
    )
    bind_gaussian_queue_owner(ticket)
    assert wait_for_gaussian_turn(ticket)
    ticket.release()
    queue_files = list(
        (tmp_path / "outputs" / ".resource_budget" / "gaussian_queue").glob(
            "ticket_*.json"
        )
    )
    assert queue_files == []


def test_gaussian_queue_releases_after_typed_failure_and_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败或取消都必须删除自己的条目，让后续 Gaussian 继续。"""

    _configure_workspace(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        with gaussian_execution_slot("typed-failing"):
            raise RuntimeError("worker failed")

    cancelled = enqueue_gaussian_job("typed-cancelled", kind="typed")
    bind_gaussian_queue_owner(cancelled)
    assert not wait_for_gaussian_turn(cancelled, should_cancel=lambda: True)

    next_ticket = enqueue_gaussian_job("typed-next", kind="typed")
    bind_gaussian_queue_owner(next_ticket)
    assert wait_for_gaussian_turn(next_ticket)
    next_ticket.release()


def test_native_supervisor_crash_before_queue_bind_does_not_block_next_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """supervisor 已留下 claim 但进程异常退出时，下一项能清理死条目。"""

    _configure_workspace(monkeypatch, tmp_path)
    crashed = enqueue_gaussian_job(
        "job-crashed",
        kind="native",
        job_id="job-crashed",
    )
    job_directory = tmp_path / "outputs" / "execution_jobs" / "job-crashed"
    job_directory.mkdir(parents=True)
    (job_directory / "supervisor_spec.json").write_text("{}", encoding="utf-8")
    (job_directory / "supervisor_claim.json").write_text(
        json.dumps({"supervisor_pid": 99_999_999}),
        encoding="utf-8",
    )

    next_ticket = enqueue_gaussian_job(
        "job-after-crash",
        kind="native",
        job_id="job-after-crash",
    )
    bind_gaussian_queue_owner(next_ticket)
    assert wait_for_gaussian_turn(next_ticket)
    next_ticket.release()
    crashed.release()


def test_dead_supervisor_keeps_ticket_without_legacy_group_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧记录缺少 PGID/SID 时，即使 child leader 消失也不能提前放行。"""

    _configure_workspace(monkeypatch, tmp_path)
    crashed = enqueue_gaussian_job("job_child_alive", job_id="job_child_alive")
    job_directory = tmp_path / "outputs" / "execution_jobs" / "job_child_alive"
    job_directory.mkdir(parents=True)
    (job_directory / "status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "supervisor_pid": 99_999_999,
                "child_pid": os.getpid(),
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "gpu_count": 0,
                },
                "resource_allocation": {
                    "cpu_ids": sorted(os.sched_getaffinity(0))[:1],
                    "gpu_ids": [],
                },
                "metadata": {"software_id": "gaussian"},
            }
        ),
        encoding="utf-8",
    )
    bind_gaussian_queue_owner(crashed, pid=99_999_999)
    next_ticket = enqueue_gaussian_job("job_child_next", job_id="job_child_next")
    queue_directory = tmp_path / "outputs" / ".resource_budget" / "gaussian_queue"
    queue_ids = sorted(path.name for path in queue_directory.glob("ticket_*.json"))
    assert queue_ids == [
        f"ticket_{crashed.ticket_id:020d}.json",
        f"ticket_{next_ticket.ticket_id:020d}.json",
    ]
    assert active_resource_usage() == {
        "cpu_cores": 1,
        "memory_mb": 256,
        "gpu_count": 0,
    }

    (job_directory / "status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "supervisor_pid": 99_999_999,
                "child_pid": 99_999_999,
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "gpu_count": 0,
                },
                "resource_allocation": {
                    "cpu_ids": sorted(os.sched_getaffinity(0))[:1],
                    "gpu_ids": [],
                },
                "metadata": {"software_id": "gaussian"},
            }
        ),
        encoding="utf-8",
    )
    after_exit = enqueue_gaussian_job(
        "job_child_after_exit", job_id="job_child_after_exit"
    )
    failed_status = json.loads(
        (job_directory / "status.json").read_text(encoding="utf-8")
    )
    assert failed_status["status"] == "running"
    assert active_resource_usage() == {
        "cpu_cores": 1,
        "memory_mb": 256,
        "gpu_count": 0,
    }
    assert read_gaussian_queue_snapshot()["item_count"] == 3
    # 旧记录没有安全的 descendant 清空证明，只能由人工明确释放测试夹具。
    next_ticket.release()
    after_exit.release()
    crashed.release()


def test_queued_supervisor_crash_before_child_identity_keeps_ticket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Popen 后落盘 child 身份前崩溃时，queued 记录不能按 handoff 清理。"""

    _configure_workspace(monkeypatch, tmp_path)
    crashed = enqueue_gaussian_job(
        "job-crash-before-child-status",
        kind="native",
        job_id="job-crash-before-child-status",
    )
    job_directory = (
        tmp_path
        / "outputs"
        / "execution_jobs"
        / "job-crash-before-child-status"
    )
    job_directory.mkdir(parents=True)
    (job_directory / "status.json").write_text(
        json.dumps(
            {
                "status": "queued",
                "supervisor_pid": 99_999_999,
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "gpu_count": 0,
                    "walltime_seconds": 30,
                },
                "metadata": {"software_id": "gaussian"},
            }
        ),
        encoding="utf-8",
    )
    (job_directory / "supervisor_claim.json").write_text(
        json.dumps({"supervisor_pid": 99_999_999}),
        encoding="utf-8",
    )

    next_ticket = enqueue_gaussian_job(
        "job-after-crash-before-child-status",
        kind="native",
        job_id="job-after-crash-before-child-status",
    )
    assert read_gaussian_queue_snapshot()["item_count"] == 2
    assert active_resource_usage() == {
        "cpu_cores": 0,
        "memory_mb": 0,
        "gpu_count": 0,
    }
    crashed.release()
    next_ticket.release()


def _write_queued_supervisor_spec(
    root: Path,
    *,
    job_id: str,
    ticket: object,
    script: Path,
    events: Path,
    sleep_seconds: float,
    walltime_seconds: int = 10,
) -> Path:
    """写入带 FIFO 身份的最小 native Gaussian supervisor spec。"""

    job_directory = root / "outputs" / "execution_jobs" / job_id
    job_directory.mkdir(parents=True)
    status_path = job_directory / "status.json"
    stdout_path = job_directory / "stdout.log"
    stderr_path = job_directory / "stderr.log"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "job_type": "native_software",
                "status": "queued",
                "supervisor_pid": None,
                "job_directory": f"outputs/execution_jobs/{job_id}",
                "stdout_path": f"outputs/execution_jobs/{job_id}/stdout.log",
                "stderr_path": f"outputs/execution_jobs/{job_id}/stderr.log",
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "gpu_count": 0,
                    "walltime_seconds": walltime_seconds,
                },
                "resource_allocation": {},
                "metadata": {"software_id": "gaussian", "label": job_id},
                "submitted_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    specification = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "native_software",
        "runtime": "core",
        "command": [
            sys.executable,
            str(script),
            str(events),
            str(sleep_seconds),
            job_id,
        ],
        "job_directory": str(job_directory),
        "relative_job_directory": f"outputs/execution_jobs/{job_id}",
        "status_path": str(status_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "relative_stdout_path": f"outputs/execution_jobs/{job_id}/stdout.log",
        "relative_stderr_path": f"outputs/execution_jobs/{job_id}/stderr.log",
        "stdin_path": None,
        "resource_limits": {
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
            "walltime_seconds": walltime_seconds,
        },
        "resource_allocation": {},
        "evaluation_resource_budget": {},
        "metadata": {"software_id": "gaussian", "label": job_id},
        "submitted_at": "2026-01-01T00:00:00+00:00",
        "gaussian_queue_ticket": ticket.as_dict(),
        "launch_file_snapshots": [],
    }
    spec_path = job_directory / "supervisor_spec.json"
    spec_path.write_text(
        json.dumps(specification),
        encoding="utf-8",
    )
    return spec_path


def _start_supervisors(
    specifications: list[Path],
    *,
    workspace: Path,
) -> list[subprocess.Popen[bytes]]:
    """并发启动多个真实 supervisor，模拟多个 workflow 同时提交。"""

    environment = {
        **os.environ,
        "MINICHEM_MCP_WORKSPACE": str(workspace),
        "RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES": "4",
        "RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB": "4096",
        "RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT": "0",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    return [
        subprocess.Popen(
            [sys.executable, str(job_supervisor.__file__), str(specification)],
            cwd=specification.parent,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for specification in specifications
    ]


def test_native_supervisors_share_fifo_and_release_after_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native supervisor 按 ticket 串行执行，取消队首之外的作业后仍能推进。"""

    _configure_workspace(monkeypatch, tmp_path)
    script = tmp_path / "record_event.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "events = Path(sys.argv[1])\n"
        "delay = float(sys.argv[2])\n"
        "label = sys.argv[3]\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(f'start:{label}\\n')\n"
        "    handle.flush()\n"
        "time.sleep(delay)\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(f'finish:{label}\\n')\n"
        "    handle.flush()\n",
        encoding="utf-8",
    )
    events = tmp_path / "events.log"
    specs = []
    for index, delay in ((1, 0.4), (2, 0.1), (3, 0.1)):
        job_id = f"job-native-{index}"
        ticket = enqueue_gaussian_job(job_id, kind="native", job_id=job_id)
        spec = _write_queued_supervisor_spec(
            tmp_path,
            job_id=job_id,
            ticket=ticket,
            script=script,
            events=events,
            sleep_seconds=delay,
        )
        specs.append(spec)

    processes = _start_supervisors(specs, workspace=tmp_path)
    try:
        second_status = tmp_path / "outputs" / "execution_jobs" / "job-native-2" / "status.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if second_status.is_file():
                value = json.loads(second_status.read_text(encoding="utf-8"))
                if isinstance(value.get("supervisor_pid"), int):
                    break
            time.sleep(0.02)
        (second_status.parent / "cancel_requested").touch()
        exit_codes = [process.wait(timeout=15) for process in processes]
        assert exit_codes == [0, 2, 0]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    lines = events.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "start:job-native-1",
        "finish:job-native-1",
        "start:job-native-3",
        "finish:job-native-3",
    ]
    second_value = json.loads(second_status.read_text(encoding="utf-8"))
    assert second_value["status"] == "cancelled"


def test_native_supervisor_keeps_fifo_when_leader_exits_with_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """leader 提前退出但同组 descendant 存活时必须保持 running 和 FIFO。"""

    _configure_workspace(monkeypatch, tmp_path)
    script = tmp_path / "fork_and_exit.py"
    script.write_text(
        "import os, time\n"
        "descendant = os.fork()\n"
        "if descendant:\n"
        "    os._exit(0)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    job_id = "job-native-fork-descendant"
    ticket = enqueue_gaussian_job(job_id, kind="native", job_id=job_id)
    specification = _write_queued_supervisor_spec(
        tmp_path,
        job_id=job_id,
        ticket=ticket,
        script=script,
        events=tmp_path / "unused-events.log",
        sleep_seconds=0,
    )
    process = _start_supervisors([specification], workspace=tmp_path)[0]
    status_path = tmp_path / "outputs" / "execution_jobs" / job_id / "status.json"
    next_ticket = None
    group_id = None
    session_id = None
    boot_id = None
    try:
        assert process.wait(timeout=15) == 1
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["status"] == "running"
        assert status["supervisor_exit_pending"] is True
        assert status["error"]["code"] == "process_group_descendants_alive"
        assert status["process_group_activity"] == "nonempty"
        group_id = status["child_process_group_id"]
        session_id = status["child_session_id"]
        boot_id = status["child_boot_id"]
        assert inspect_process_group_membership(group_id, session_id, boot_id) == "nonempty"
        next_ticket = enqueue_gaussian_job(
            "job-native-after-descendant",
            kind="native",
            job_id="job-native-after-descendant",
        )
        assert read_gaussian_queue_snapshot()["item_count"] == 2
        assert active_resource_usage() == {
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
        }
    finally:
        if (
            isinstance(group_id, int)
            and isinstance(session_id, int)
            and isinstance(boot_id, str)
            and inspect_process_group_membership(group_id, session_id, boot_id)
            == "nonempty"
        ):
            os.killpg(group_id, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if inspect_process_group_membership(group_id, session_id, boot_id) == "empty":
                    break
                time.sleep(0.05)
        if process.poll() is None:
            process.kill()
            process.wait()
    assert (
        isinstance(group_id, int)
        and isinstance(session_id, int)
        and isinstance(boot_id, str)
    )
    assert inspect_process_group_membership(group_id, session_id, boot_id) == "empty"
    assert next_ticket is not None
    after_recovery = enqueue_gaussian_job(
        "job-native-after-descendant-recovery",
        kind="native",
        job_id="job-native-after-descendant-recovery",
    )
    failed_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert failed_status["status"] == "failed"
    assert read_gaussian_queue_snapshot()["item_count"] == 2
    bind_gaussian_queue_owner(next_ticket)
    assert wait_for_gaussian_turn(next_ticket)
    next_ticket.release()
    after_recovery.release()


def test_gaussian_queue_wait_does_not_consume_execution_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队列等待不计入 walltime，且等待中的 queued 不带 started_at。"""

    _configure_workspace(monkeypatch, tmp_path)
    script = tmp_path / "record_event.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "events = Path(sys.argv[1])\n"
        "delay = float(sys.argv[2])\n"
        "label = sys.argv[3]\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(f'start:{label}\\n')\n"
        "    handle.flush()\n"
        "time.sleep(delay)\n"
        "with events.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(f'finish:{label}\\n')\n"
        "    handle.flush()\n",
        encoding="utf-8",
    )
    events = tmp_path / "events.log"
    first_id = "job-timeout-first"
    second_id = "job-timeout-second"
    first_ticket = enqueue_gaussian_job(first_id, job_id=first_id)
    second_ticket = enqueue_gaussian_job(second_id, job_id=second_id)
    specifications = [
        _write_queued_supervisor_spec(
            tmp_path,
            job_id=first_id,
            ticket=first_ticket,
            script=script,
            events=events,
            sleep_seconds=1.3,
        ),
        _write_queued_supervisor_spec(
            tmp_path,
            job_id=second_id,
            ticket=second_ticket,
            script=script,
            events=events,
            sleep_seconds=0.1,
            walltime_seconds=1,
        ),
    ]
    processes = _start_supervisors(specifications, workspace=tmp_path)
    second_status = tmp_path / "outputs" / "execution_jobs" / second_id / "status.json"
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if second_status.is_file():
                value = json.loads(second_status.read_text(encoding="utf-8"))
                if isinstance(value.get("supervisor_pid"), int):
                    assert value["status"] == "queued"
                    assert value.get("started_at") is None
                    assert "queue_state" not in value
                    break
            time.sleep(0.02)
        else:
            pytest.fail("second supervisor did not publish its queued state")
        assert [process.wait(timeout=15) for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    final_status = json.loads(second_status.read_text(encoding="utf-8"))
    assert final_status["status"] == "success"
    assert final_status["started_at"]
    assert final_status["duration_seconds"] < 1


def test_queue_managed_gaussian_reservation_is_counted_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队首 Gaussian 的 live reservation 与 running status 不重复计资源。"""

    _configure_workspace(monkeypatch, tmp_path)
    script = tmp_path / "sleep_job.py"
    script.write_text(
        "import sys, time\n"
        "time.sleep(float(sys.argv[2]))\n",
        encoding="utf-8",
    )
    job_id = "job-resource-once"
    ticket = enqueue_gaussian_job(job_id, job_id=job_id)
    specification = _write_queued_supervisor_spec(
        tmp_path,
        job_id=job_id,
        ticket=ticket,
        script=script,
        events=tmp_path / "unused-events.log",
        sleep_seconds=1.2,
    )
    processes = _start_supervisors([specification], workspace=tmp_path)
    status_path = tmp_path / "outputs" / "execution_jobs" / job_id / "status.json"
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if status_path.is_file():
                value = json.loads(status_path.read_text(encoding="utf-8"))
                if value.get("status") == "running":
                    break
            time.sleep(0.02)
        else:
            pytest.fail("Gaussian supervisor did not reach running state")
        assert active_resource_usage() == {
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
        }
        assert [item["record_type"] for item in active_resource_jobs()] == [
            "submission_reservation"
        ]
        visible_status = json.loads(status_path.read_text(encoding="utf-8"))
        assert not any(key.startswith("_") for key in visible_status)
        assert processes[0].wait(timeout=15) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    final_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert final_status["status"] == "success"
    assert inspect_process_group_membership(
        final_status["child_process_group_id"],
        final_status["child_session_id"],
        final_status["child_boot_id"],
    ) == "empty"
    assert active_resource_usage() == {
        "cpu_cores": 0,
        "memory_mb": 0,
        "gpu_count": 0,
    }
