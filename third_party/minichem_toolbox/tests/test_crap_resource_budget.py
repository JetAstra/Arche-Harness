"""补齐资源预算内部协议的分支覆盖。

这些测试直接覆盖恢复、scope 隔离、队列交接和资源分配的边界分支。它们不替代
服务级测试，而是把每个持久化记录的 fail-closed 结果固定下来，避免复杂度指标
因为关键异常路径长期没有执行而失真。
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from minichem_toolbox import resource_budget
from minichem_toolbox.process_identity import ProcessIdentity
from minichem_toolbox.resource_budget import (
    ResourceStateUnavailable,
    ResourceBudgetExceeded,
    bind_gaussian_queue_owner,
    bind_gaussian_queue_worker,
    enqueue_gaussian_job,
    reserve_resources,
)
from minichem_toolbox.scope import ScopeContext


def _workspace_scope(tmp_path: Path, name: str = "one") -> ScopeContext:
    """创建一个供资源协议测试使用的真实 workspace/scheduler。"""

    workspace = tmp_path / f"workspace-{name}"
    scheduler = tmp_path / f"scheduler-{name}"
    workspace.mkdir()
    scheduler.mkdir()
    return ScopeContext(
        task_id=f"task-{name}",
        batch_id=f"batch-{name}",
        combination_id=f"combination-{name}",
        scope_id=f"scope-{name}",
        workspace=workspace,
        capability_hash="test",
        scheduler_root=scheduler,
    )


def _configure_workspace(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """把资源目录和可用预算固定到临时工作区。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(root))
    monkeypatch.delenv("MINICHEM_SHARED_SCHEDULER_ROOT", raising=False)
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "2")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "512")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")


def _assert_resource_state_error(callable_object: Any, *args: Any, **kwargs: Any) -> None:
    """断言持久资源记录的非法输入统一 fail-closed。"""

    with pytest.raises(ResourceStateUnavailable):
        callable_object(*args, **kwargs)


def test_private_atomic_write_rejects_non_regular_and_replaced_targets(
    tmp_path: Path,
) -> None:
    """原子写入同时覆盖已有普通文件和 symlink 目标的拒绝路径。"""

    target = tmp_path / "nested" / "record.json"
    resource_budget._write_private_text_atomically(target, "first")
    assert target.read_text(encoding="utf-8") == "first"

    link_target = tmp_path / "outside.txt"
    link_target.write_text("outside", encoding="utf-8")
    target.unlink()
    target.symlink_to(link_target)
    with pytest.raises(RuntimeError, match="regular file"):
        resource_budget._write_private_text_atomically(target, "blocked")


def test_scope_workspace_registration_is_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    """共享 scope 索引首次写入、幂等重放和损坏记录均有确定结果。"""

    scope = _workspace_scope(tmp_path)
    scheduler = scope.scheduler_root
    assert scheduler is not None
    resource_budget._register_scope_workspace(scheduler, scope)
    resource_budget._register_scope_workspace(scheduler, scope)
    index = scheduler / "scope_workspaces"
    records = list(index.glob("workspace_*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["scope_id"] == scope.scope_id

    records[0].write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot be read"):
        resource_budget._register_scope_workspace(scheduler, scope)

    records[0].write_text(
        json.dumps({"scope_id": "other", "workspace": str(scope.workspace)}),
        encoding="utf-8",
    )
    records[0].chmod(0o600)
    with pytest.raises(RuntimeError, match="identity conflict"):
        resource_budget._register_scope_workspace(scheduler, scope)


def test_scope_workspace_registration_rejects_invalid_directory_and_permissions(
    tmp_path: Path,
) -> None:
    """scope 索引目录不能是普通文件、symlink 或过宽权限。"""

    scope = _workspace_scope(tmp_path, "invalid-index")
    scheduler = scope.scheduler_root
    assert scheduler is not None
    index = scheduler / "scope_workspaces"
    index.write_text("not-a-directory", encoding="utf-8")
    with pytest.raises(RuntimeError, match="index is invalid"):
        resource_budget._register_scope_workspace(scheduler, scope)

    index.unlink()
    target = tmp_path / "index-target"
    target.mkdir()
    index.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="index is invalid"):
        resource_budget._register_scope_workspace(scheduler, scope)
    index.unlink()

    resource_budget._register_scope_workspace(scheduler, scope)
    record = next(index.glob("workspace_*.json"))
    record.chmod(0o644)
    with pytest.raises(RuntimeError, match="too open"):
        resource_budget._register_scope_workspace(scheduler, scope)


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"scope_id": 1, "workspace": "/tmp/x"}, "invalid"),
        ({"scope_id": " scope", "workspace": "/tmp/x"}, "incomplete"),
        ({"scope_id": "scope", "workspace": "relative"}, "canonical"),
        ({"scope_id": "scope", "workspace": "/tmp/../tmp/x"}, "canonical"),
    ],
)
def test_scope_workspace_index_identity_rejects_noncanonical_records(
    tmp_path: Path,
    record: dict[str, Any],
    expected: str,
) -> None:
    """索引记录必须含安全身份、正确文件名和绝对规范路径。"""

    if isinstance(record["scope_id"], str) and isinstance(record["workspace"], str):
        identity = f"{record['scope_id']}\0{record['workspace']}".encode("utf-8")
        filename = "workspace_" + resource_budget.hashlib.sha256(identity).hexdigest() + ".json"
    else:
        filename = "workspace_" + "0" * 64 + ".json"
    path = tmp_path / filename
    with pytest.raises(ResourceStateUnavailable, match=expected):
        resource_budget._validate_scope_workspace_index_identity(path, record)


def test_scope_workspace_index_identity_accepts_hashed_absolute_path(
    tmp_path: Path,
) -> None:
    """合法 scope/workspace 组合通过文件名摘要和规范路径核验。"""

    scope_id = "scope-accepted"
    workspace = str((tmp_path / "workspace").resolve())
    identity = f"{scope_id}\0{workspace}".encode("utf-8")
    path = tmp_path / ("workspace_" + resource_budget.hashlib.sha256(identity).hexdigest() + ".json")
    assert resource_budget._validate_scope_workspace_index_identity(
        path,
        {"scope_id": scope_id, "workspace": workspace},
    ) == Path(workspace)


def test_scope_ticket_path_and_registration_validation_cover_identity_edges(
    tmp_path: Path,
) -> None:
    """ticket workspace 必须是存在的绝对目录，并且已经注册到 scheduler。"""

    scope = _workspace_scope(tmp_path)
    scheduler = scope.scheduler_root
    assert scheduler is not None
    resource_budget._register_scope_workspace(scheduler, scope)
    assert resource_budget._validate_scope_workspace_ticket_path(str(scope.workspace)) == scope.workspace
    resource_budget._validate_registered_scope_workspace(
        scheduler, scope.scope_id, str(scope.workspace)
    )

    for value in (None, "", "relative", str(tmp_path / "missing")):
        with pytest.raises(ResourceStateUnavailable):
            resource_budget._validate_scope_workspace_ticket_path(value)
    for scheduler_value, scope_id, workspace in (
        (None, scope.scope_id, str(scope.workspace)),
        (scheduler, "bad scope", str(scope.workspace)),
        (scheduler, scope.scope_id, str(tmp_path / "unregistered")),
    ):
        with pytest.raises(ResourceStateUnavailable):
            resource_budget._validate_registered_scope_workspace(
                scheduler_value, scope_id, workspace
            )


def test_gaussian_ticket_scope_validation_covers_legacy_and_scoped_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """legacy ticket 不需要 scope；认证 ticket 必须验证 job 与 workspace。"""

    scheduler = tmp_path / "scheduler"
    scheduler.mkdir()
    calls: list[tuple[Any, Any, Any]] = []
    monkeypatch.setattr(resource_budget, "current_scope", lambda: None)
    monkeypatch.setattr(resource_budget, "_configured_scheduler_root", lambda: scheduler)
    monkeypatch.setattr(
        resource_budget,
        "_validate_registered_scope_workspace",
        lambda *args: calls.append(args),
    )
    resource_budget._validate_gaussian_ticket_scope({"job_id": None})
    resource_budget._validate_gaussian_ticket_scope({"job_id": "plain-job"})
    with pytest.raises(ResourceStateUnavailable, match="job id"):
        resource_budget._validate_gaussian_ticket_scope({"job_id": 1})
    with pytest.raises(ResourceStateUnavailable, match="scope"):
        resource_budget._validate_gaussian_ticket_scope({"scope_id": 1})
    with pytest.raises(ResourceStateUnavailable, match="workspace"):
        resource_budget._validate_gaussian_ticket_scope({"scope_id": "scope"})
    with pytest.raises(ResourceStateUnavailable, match="path-safe"):
        resource_budget._validate_gaussian_ticket_scope(
            {"scope_id": "scope", "workspace": str(tmp_path), "job_id": "bad"}
        )
    resource_budget._validate_gaussian_ticket_scope(
        {
            "scope_id": "scope",
            "workspace": str(tmp_path),
            "job_id": "job_" + "a" * 32,
        }
    )
    assert calls == [(scheduler, "scope", str(tmp_path))]


def test_wait_for_gaussian_turn_rejects_missing_mismatched_and_corrupt_tickets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """等待队首时发现身份、文件类型或 JSON 损坏都不越过 FIFO。"""

    _configure_workspace(monkeypatch, tmp_path)
    mismatch = enqueue_gaussian_job("wait-mismatch", kind="typed")
    assert resource_budget.wait_for_gaussian_turn(
        replace(mismatch, owner_id="other"), poll_seconds=0.01
    ) is False
    mismatch.release()

    symlink_ticket = enqueue_gaussian_job("wait-symlink", kind="typed")
    queue_path = (
        tmp_path
        / "outputs"
        / ".resource_budget"
        / "gaussian_queue"
        / f"ticket_{symlink_ticket.ticket_id:020d}.json"
    )
    queue_path.unlink()
    queue_path.symlink_to(tmp_path / "outside")
    with pytest.raises(ResourceStateUnavailable):
        resource_budget.wait_for_gaussian_turn(symlink_ticket, poll_seconds=0.01)
    queue_path.unlink()

    corrupt = enqueue_gaussian_job("wait-corrupt", kind="typed")
    corrupt_path = (
        tmp_path
        / "outputs"
        / ".resource_budget"
        / "gaussian_queue"
        / f"ticket_{corrupt.ticket_id:020d}.json"
    )
    corrupt_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable, match="cannot be read"):
        resource_budget.wait_for_gaussian_turn(corrupt, poll_seconds=0.01)
    corrupt_path.unlink()


def test_normalize_worker_identity_accepts_protocol_variants_and_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker 身份既兼容 dataclass、as_dict 对象，也拒绝坏 PID/启动时钟。"""

    identity = ProcessIdentity(123, 456, "boot")
    assert resource_budget._normalize_worker_identity(identity) == {
        "pid": 123,
        "start_time_ticks": 456,
        "boot_id": "boot",
    }

    class IdentityObject:
        def as_dict(self) -> dict[str, Any]:
            return {"pid": "124", "start_time_ticks": "457", "boot_id": " boot-2 "}

    assert resource_budget._normalize_worker_identity(IdentityObject()) == {
        "pid": 124,
        "start_time_ticks": 457,
        "boot_id": "boot-2",
    }
    assert resource_budget._normalize_worker_identity({"pid": 125})["boot_id"] is None
    for value in (
        {"pid": "bad"},
        {"pid": 0},
        {"pid": 1, "start_time_ticks": "bad"},
        {"pid": 1, "start_time_ticks": 0},
    ):
        with pytest.raises(ValueError):
            resource_budget._normalize_worker_identity(value)


def test_native_queue_claim_activity_covers_claim_owner_and_staging_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native 交接在 claim、owner、控制文件和 staging 状态间保持保守路由。"""

    directory = tmp_path / "job"
    directory.mkdir()
    value = {"owner_pid": 0}

    (directory / "supervisor_claim.json").write_text("bad", encoding="utf-8")
    assert resource_budget._native_queue_claim_activity(value, directory) == "unknown"
    (directory / "supervisor_claim.json").write_text(
        json.dumps({"supervisor_pid": 123}), encoding="utf-8"
    )
    monkeypatch.setattr(resource_budget, "_persisted_process_activity", lambda *_: "alive")
    assert resource_budget._native_queue_claim_activity(value, directory) == "supervisor_alive"
    monkeypatch.setattr(resource_budget, "_persisted_process_activity", lambda *_: "unknown")
    assert resource_budget._native_queue_claim_activity(value, directory) == "unknown"
    monkeypatch.setattr(resource_budget, "_persisted_process_activity", lambda *_: "stopped")
    (directory / "supervisor_claim.json").write_text(
        json.dumps({"supervisor_pid": 123}), encoding="utf-8"
    )
    assert resource_budget._native_queue_claim_activity(
        value, directory, status={"status": "running"}
    ) == "unknown"
    assert resource_budget._native_queue_claim_activity(
        value, directory, status={"status": "success"}
    ) == "dead"

    (directory / "supervisor_claim.json").unlink()
    monkeypatch.setattr(resource_budget, "_process_is_alive", lambda _pid: True)
    assert resource_budget._native_queue_claim_activity({"owner_pid": 55}, directory) == "supervisor_alive"
    monkeypatch.setattr(resource_budget, "_process_is_alive", lambda _pid: False)
    assert resource_budget._native_queue_claim_activity({"owner_pid": 55}, directory) == "dead"
    (directory / "request.json").write_text("{}", encoding="utf-8")
    assert resource_budget._native_queue_claim_activity({}, directory) == "handoff_pending"
    (directory / "request.json").unlink()
    assert resource_budget._native_queue_claim_activity({}, directory) == "staging"
    (directory / "request.json").symlink_to(tmp_path / "outside")
    assert resource_budget._native_queue_claim_activity({}, directory) == "unknown"


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"cpu_ids": "bad", "gpu_ids": []},
        {"cpu_ids": [True], "gpu_ids": []},
        {"cpu_ids": [-1], "gpu_ids": []},
        {"cpu_ids": [], "gpu_ids": [1]},
        {"cpu_ids": [], "gpu_ids": [""]},
        {"cpu_ids": [0, 0], "gpu_ids": []},
        {"cpu_ids": [], "gpu_ids": ["0", "0"]},
        {"cpu_ids": [99], "gpu_ids": []},
    ],
)
def test_persisted_allocation_rejects_every_malformed_shape(
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
) -> None:
    """恢复时 CPU/GPU 分配的类型、重复、池边界都必须严格核对。"""

    monkeypatch.setattr(resource_budget, "_cpu_pool", lambda _budget: [0, 1])
    monkeypatch.setattr(resource_budget, "_gpu_pool", lambda _budget: [])
    _assert_resource_state_error(
        resource_budget._validate_persisted_allocation,
        value,
        "test",
    )


def test_persisted_allocation_validates_limits_and_accepts_exact_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合法分配覆盖无 limits 和 limits 精确匹配两条成功路径。"""

    monkeypatch.setattr(resource_budget, "_cpu_pool", lambda _budget: [0, 1])
    monkeypatch.setattr(resource_budget, "_gpu_pool", lambda _budget: [])
    allocation = {"cpu_ids": [0], "gpu_ids": []}
    assert resource_budget._validate_persisted_allocation(allocation, "test") == allocation
    assert resource_budget._validate_persisted_allocation(
        allocation,
        "test",
        resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    ) == allocation
    with pytest.raises(ResourceStateUnavailable, match="count"):
        resource_budget._validate_persisted_allocation(
            allocation,
            "test",
            resource_limits={"cpu_cores": 2, "memory_mb": 128, "gpu_count": 0},
        )


def test_scope_gaussian_records_filter_known_and_foreign_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """资源视图不重复显示 reservation 或已有 job，并保留 foreign owned=false。"""

    scope = SimpleNamespace(scope_id="scope-one", workspace=Path("/workspace/one"))
    entries = [
        (Path("one"), {"job_id": "job-known", "scope_id": scope.scope_id, "workspace": str(scope.workspace), "state": "queued"}),
        (Path("two"), {"job_id": "job-reserved", "scope_id": scope.scope_id, "workspace": str(scope.workspace), "state": "running"}),
        (Path("three"), {"job_id": "job-foreign", "scope_id": "scope-two", "workspace": "/workspace/two", "state": "queued"}),
        (Path("four"), {"job_id": None, "scope_id": scope.scope_id, "workspace": str(scope.workspace), "state": "queued"}),
    ]
    monkeypatch.setattr(resource_budget, "_gaussian_queue_lock", lambda: nullcontext())
    monkeypatch.setattr(resource_budget, "_gaussian_queue_entries", lambda: entries)
    records = [{"job_id": "job-known"}]
    reservations = [(Path("reservation"), {"job_id": "job-reserved"})]
    resource_budget._append_scope_gaussian_queue_records(records, reservations, scope)
    assert [item.get("job_id") for item in records if item.get("job_id")] == [
        "job-known",
    ]
    assert records[-2:] == [
        {
            "record_type": "gaussian_queue",
            "position": 3,
            "status": "queued",
            "owned": False,
        },
        {
            "record_type": "gaussian_queue",
            "position": 4,
            "status": "queued",
            "owned": True,
        },
    ]


def test_queue_worker_binding_and_owner_mismatch_are_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """typed worker 绑定成功，错误归属、native 类型和缺失记录都不改队列。"""

    _configure_workspace(monkeypatch, tmp_path)
    typed = enqueue_gaussian_job("typed-worker", kind="typed")
    identity = ProcessIdentity(123, 456, "boot")
    assert bind_gaussian_queue_worker(typed, identity)
    assert not bind_gaussian_queue_worker(replace(typed, owner_id="other"), identity)
    typed.release()
    assert not bind_gaussian_queue_worker(typed, identity)

    native = enqueue_gaussian_job("native-worker", kind="native", job_id="native-worker")
    assert not bind_gaussian_queue_worker(native, identity)
    native.release()

    missing = enqueue_gaussian_job("owner-missing", kind="typed")
    missing.release()
    bind_gaussian_queue_owner(missing)


def test_queue_owner_binding_rejects_symlink_and_wrong_owner_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native owner 绑定不会跟随 symlink，也不会覆盖其他 ticket 的身份。"""

    _configure_workspace(monkeypatch, tmp_path)
    ticket = enqueue_gaussian_job("owner", kind="native", job_id="owner")
    queue_path = (
        tmp_path
        / "outputs"
        / ".resource_budget"
        / "gaussian_queue"
        / f"ticket_{ticket.ticket_id:020d}.json"
    )
    wrong = replace(ticket, owner_id="other")
    bind_gaussian_queue_owner(wrong, pid=123)
    assert json.loads(queue_path.read_text(encoding="utf-8"))["owner_pid"] is None
    queue_path.unlink()
    queue_path.symlink_to(tmp_path / "outside")
    with pytest.raises(ResourceStateUnavailable):
        bind_gaussian_queue_owner(ticket)
    queue_path.unlink()


def test_reservation_worker_binding_checks_tracking_identity_and_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """typed reservation 绑定 worker 前复核原记录，普通 reservation 不绑定。"""

    _configure_workspace(monkeypatch, tmp_path)
    ordinary = resource_budget.ResourceReservation(
        path=tmp_path / "ordinary.json",
        resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        resource_allocation={"cpu_ids": [0], "gpu_ids": []},
    )
    assert ordinary.bind_worker(ProcessIdentity(1, 2, "boot")) is False

    reservation = reserve_resources(
        {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        kind="predefined_action",
        label="gaussian",
        worker_tracking="gaussian_typed",
    )
    assert reservation.bind_worker(ProcessIdentity(123, 456, "boot")) is True
    assert json.loads(reservation.path.read_text(encoding="utf-8"))["worker_pid"] == 123
    reservation.release()
    assert reservation.bind_worker(ProcessIdentity(123, 456, "boot")) is False

    changed = reserve_resources(
        {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        kind="predefined_action",
        label="gaussian",
        worker_tracking="gaussian_typed",
    )
    record = json.loads(changed.path.read_text(encoding="utf-8"))
    record["label"] = "changed"
    changed.path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable, match="identity changed"):
        changed.bind_worker(ProcessIdentity(123, 456, "boot"))
    changed.path.unlink()


def test_active_allocations_rejects_overlapping_persisted_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨 status/reservation 的 CPU/GPU 分配不能重叠。"""

    limits = {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0}
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "2")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "512")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    monkeypatch.setattr(resource_budget, "_cpu_pool", lambda _budget: [0, 1])
    monkeypatch.setattr(resource_budget, "_gpu_pool", lambda _budget: [])
    monkeypatch.setattr(
        resource_budget,
        "_active_execution_job_statuses",
        lambda _reservations: [
            (Path("root"), {"resource_limits": limits, "resource_allocation": {"cpu_ids": [0], "gpu_ids": []}})
        ],
    )
    monkeypatch.setattr(
        resource_budget,
        "_live_resource_reservations",
        lambda: [(Path("reservation"), {"resource_limits": limits, "resource_allocation": {"cpu_ids": [1], "gpu_ids": []}})],
    )
    assert resource_budget._active_allocations() == ({0, 1}, set())

    monkeypatch.setattr(
        resource_budget,
        "_live_resource_reservations",
        lambda: [(Path("reservation"), {"resource_limits": limits, "resource_allocation": {"cpu_ids": [0], "gpu_ids": []}})],
    )
    with pytest.raises(ResourceStateUnavailable, match="overlapping"):
        resource_budget._active_allocations()


def test_reservation_scheduler_root_resolution_covers_scope_env_and_invalid_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reservation scheduler 优先使用 scope，其次读取安全环境配置。"""

    scheduler = tmp_path / "scheduler"
    scheduler.mkdir()
    scope = _workspace_scope(tmp_path, "scheduler")
    monkeypatch.setattr(resource_budget, "current_scope", lambda: scope)
    assert resource_budget._resolve_reservation_scheduler_root() == scope.scheduler_root

    monkeypatch.setattr(resource_budget, "current_scope", lambda: None)
    monkeypatch.delenv("MINICHEM_SHARED_SCHEDULER_ROOT", raising=False)
    assert resource_budget._resolve_reservation_scheduler_root() is None
    monkeypatch.setenv("MINICHEM_SHARED_SCHEDULER_ROOT", str(scheduler))
    assert resource_budget._resolve_reservation_scheduler_root() == scheduler
    monkeypatch.setenv("MINICHEM_SHARED_SCHEDULER_ROOT", str(tmp_path / "missing"))
    with pytest.raises(ResourceStateUnavailable, match="unavailable"):
        resource_budget._resolve_reservation_scheduler_root()


def test_live_reservations_reject_unknown_entries_and_bad_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reservation 目录中出现未知文件或损坏 JSON 时不继续分配资源。"""

    _configure_workspace(monkeypatch, tmp_path)
    reservations = tmp_path / "outputs" / ".resource_budget" / "reservations"
    reservations.mkdir(parents=True)
    (reservations / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable, match="unknown entry"):
        resource_budget._live_resource_reservations()
    (reservations / "unexpected.txt").unlink()
    (reservations / "reservation_bad.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable, match="cannot be read"):
        resource_budget._live_resource_reservations()


def test_ticket_release_handles_deferred_missing_mismatch_and_corrupt_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队列 ticket release 对延迟、缺失、归属不符和损坏记录保持幂等/拒绝。"""

    _configure_workspace(monkeypatch, tmp_path)
    deferred = enqueue_gaussian_job("deferred", kind="typed")
    replace(deferred, release_deferred=True).release()
    deferred.release()

    missing = enqueue_gaussian_job("missing", kind="typed")
    missing.release()
    missing.release()

    mismatch = enqueue_gaussian_job("mismatch", kind="typed")
    replace(mismatch, owner_id="other").release()
    assert (tmp_path / "outputs" / ".resource_budget" / "gaussian_queue" / f"ticket_{mismatch.ticket_id:020d}.json").exists()
    mismatch.release()

    corrupt = enqueue_gaussian_job("corrupt", kind="typed")
    path = tmp_path / "outputs" / ".resource_budget" / "gaussian_queue" / f"ticket_{corrupt.ticket_id:020d}.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable):
        corrupt.release()
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable):
        corrupt.release()
    path.unlink()


def test_execution_scope_and_resource_record_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """执行作业 scope 校验和脱敏资源记录覆盖 legacy、foreign、owned 分支。"""

    root = tmp_path / "root"
    root.mkdir()
    scope = _workspace_scope(tmp_path, "record")
    monkeypatch.setattr(resource_budget, "current_scope", lambda: None)
    monkeypatch.setattr(resource_budget, "_configured_scheduler_root", lambda: scope.scheduler_root)
    monkeypatch.setattr(resource_budget, "_validate_registered_scope_workspace", lambda *_: None)
    monkeypatch.setattr(resource_budget, "_validate_private_directory_path", lambda path, create: root)
    resource_budget._validate_execution_job_scope(root, {})
    for status in (
        {"scope_id": "scope"},
        {"scope_id": 1, "scope_workspace": str(root)},
        {"scope_id": "scope", "scope_workspace": str(root), "job_id": "bad"},
    ):
        with pytest.raises(ResourceStateUnavailable):
            resource_budget._validate_execution_job_scope(root, status)
    resource_budget._validate_execution_job_scope(
        root,
        {"scope_id": "scope", "scope_workspace": str(root), "job_id": "job_" + "a" * 32},
    )
    monkeypatch.setattr(resource_budget, "_validate_private_directory_path", lambda path, create: tmp_path / "different")
    with pytest.raises(ResourceStateUnavailable, match="does not match"):
        resource_budget._validate_execution_job_scope(
            root, {"scope_id": "scope", "scope_workspace": str(root)}
        )

    foreign = resource_budget._execution_job_resource_record(
        root,
        {"status": "running", "scope_id": "other", "scope_workspace": "/other"},
        scope,
    )
    assert foreign == {"record_type": "execution_job", "status": "running", "owned": False}
    owned = resource_budget._execution_job_resource_record(
        root,
        {
            "status": "running",
            "scope_id": scope.scope_id,
            "scope_workspace": str(root),
            "job_id": "job-record",
            "job_type": "native",
            "metadata": {"label": "label"},
            "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            "resource_allocation": {"cpu_ids": [0], "gpu_ids": []},
        },
        None,
    )
    assert owned["job_id"] == "job-record"
    reservation = resource_budget._reservation_resource_record(
        Path("reservation_id.json"),
        {
            "scope_id": scope.scope_id,
            "workspace": str(scope.workspace),
            "pid": 1,
            "kind": "analysis",
            "label": "label",
            "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            "resource_allocation": {"cpu_ids": [0], "gpu_ids": []},
            "job_id": "job-reservation",
        },
        scope,
    )
    assert reservation["owned"] is True


def test_status_covered_by_gaussian_reservation_matches_scope_and_job(
    tmp_path: Path,
) -> None:
    """同 job 的 Gaussian reservation 只有 scope 身份一致时才去重。"""

    status = {
        "job_id": "job_" + "a" * 32,
        "metadata": {"software_id": "gaussian"},
        "scope_id": "scope-one",
        "scope_workspace": str(tmp_path / "one"),
    }
    reservation = {
        "job_id": status["job_id"],
        "scope_id": "scope-one",
        "workspace": str(tmp_path / "one"),
    }
    assert resource_budget._status_covered_by_gaussian_reservation(
        status, [(Path("r"), reservation)]
    )
    assert not resource_budget._status_covered_by_gaussian_reservation(
        status, [(Path("r"), {**reservation, "scope_id": "scope-two"})]
    )
    assert not resource_budget._status_covered_by_gaussian_reservation(
        {"job_id": status["job_id"], "metadata": {"software_id": "orca"}},
        [(Path("r"), reservation)],
    )
    assert not resource_budget._status_covered_by_gaussian_reservation(
        {"metadata": {"software_id": "gaussian"}},
        [(Path("r"), reservation)],
    )
    legacy_status = {"job_id": status["job_id"], "metadata": {"software_id": "gaussian"}}
    legacy_reservation = {"job_id": status["job_id"]}
    assert resource_budget._status_covered_by_gaussian_reservation(
        legacy_status, [(Path("r"), legacy_reservation)]
    )


def test_execution_job_directory_inventory_handles_valid_locks_and_invalid_entries(
    tmp_path: Path,
) -> None:
    """作业目录只接受约定命名的目录和普通锁文件。"""

    root = tmp_path / "root"
    root.mkdir()
    assert resource_budget._execution_job_directories(root) == []
    jobs = root / "outputs" / "execution_jobs"
    jobs.mkdir(parents=True)
    lock = jobs / ("." + "a" * 64 + ".lock")
    lock.write_text("", encoding="utf-8")
    (jobs / "job-one").mkdir()
    (jobs / "job_two").mkdir()
    assert [item.name for item in resource_budget._execution_job_directories(root)] == [
        "job-one",
        "job_two",
    ]
    (jobs / "bad-entry").write_text("x", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable, match="unknown entry"):
        resource_budget._execution_job_directories(root)
    (jobs / "bad-entry").unlink()
    (jobs / "job-link").symlink_to(root)
    with pytest.raises(ResourceStateUnavailable, match="unknown entry"):
        resource_budget._execution_job_directories(root)
    (jobs / "job-link").unlink()
    jobs.rename(root / "execution-jobs-backup")
    (root / "outputs" / "execution_jobs").symlink_to(root, target_is_directory=True)
    with pytest.raises(ResourceStateUnavailable, match="directory is invalid"):
        resource_budget._execution_job_directories(root)


def test_execution_job_status_payload_rejects_invalid_json_state_identity_and_metadata(
    tmp_path: Path,
) -> None:
    """status 读取器逐项核对文件、状态、job_id 和 metadata。"""

    job = tmp_path / "job-one"
    job.mkdir()
    status_path = job / "status.json"
    status_path.write_text(json.dumps({"status": "success"}), encoding="utf-8")
    assert resource_budget._read_execution_job_status_payload(status_path, job)["status"] == "success"

    status_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable, match="not an object"):
        resource_budget._read_execution_job_status_payload(status_path, job)
    status_path.write_text(json.dumps({"status": "unknown"}), encoding="utf-8")
    with pytest.raises(ResourceStateUnavailable, match="status is invalid"):
        resource_budget._read_execution_job_status_payload(status_path, job)
    status_path.write_text(
        json.dumps({"status": "success", "job_id": "job-other"}),
        encoding="utf-8",
    )
    with pytest.raises(ResourceStateUnavailable, match="identity"):
        resource_budget._read_execution_job_status_payload(status_path, job)
    status_path.write_text(
        json.dumps({"status": "success", "metadata": []}),
        encoding="utf-8",
    )
    with pytest.raises(ResourceStateUnavailable, match="metadata"):
        resource_budget._read_execution_job_status_payload(status_path, job)
    status_path.unlink()
    status_path.symlink_to(tmp_path / "missing-status")
    with pytest.raises(ResourceStateUnavailable, match="missing"):
        resource_budget._read_execution_job_status_payload(status_path, job)


def test_stopped_non_gaussian_execution_status_is_persistently_failed_and_releases_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外部中断后可核对的前一 boot 作业应写 failed 并释放预算。"""

    _configure_workspace(monkeypatch, tmp_path)
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    # worker 重启后 CPU affinity 可能变化；已经死亡的旧作业必须先按进程身份
    # 结束，不能因旧 CPU 编号不属于新资源池而阻断资源回收。
    monkeypatch.setattr(resource_budget, "_cpu_pool", lambda _budget: [64, 65])
    job_id = "job_stale_ts"
    job_directory = tmp_path / "outputs" / "execution_jobs" / job_id
    job_directory.mkdir(parents=True)
    status = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "transition_state_path",
        "status": "running",
        "metadata": {"software_id": "ash_neb"},
        "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 1},
        "resource_allocation": {"cpu_ids": [56], "gpu_ids": ["0"]},
        "supervisor_pid": 101,
        "supervisor_start_time_ticks": 1001,
        "supervisor_boot_id": "boot",
        "supervisor_process_group_id": 101,
        "supervisor_session_id": 101,
        "child_pid": 102,
        "child_start_time_ticks": 1002,
        "child_boot_id": "boot",
        "child_process_group_id": 102,
        "child_session_id": 102,
    }
    status_path = job_directory / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setattr(
        resource_budget,
        "_persisted_process_activity",
        lambda _value, _prefix: "stopped",
    )
    monkeypatch.setattr(
        resource_budget,
        "_persisted_process_group_activity",
        lambda _value, _prefix: "unknown",
    )
    monkeypatch.setattr(
        resource_budget,
        "capture_process_identity",
        lambda _pid: resource_budget.ProcessIdentity(
            pid=999,
            start_time_ticks=999,
            boot_id="new-worker-boot",
        ),
    )

    assert resource_budget.active_resource_usage() == {
        "cpu_cores": 0,
        "memory_mb": 0,
        "gpu_count": 0,
    }
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["error"]["code"] == "execution_process_missing"
    assert "finished_at" in persisted
    assert job_directory.is_dir()


def test_incomplete_non_gaussian_execution_identity_remains_reserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """supervisor/child 身份不完整时必须保留 running 和资源占用。"""

    _configure_workspace(monkeypatch, tmp_path)
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    monkeypatch.setattr(resource_budget, "_cpu_pool", lambda _budget: [0, 1])
    job_id = "job_unknown_ts"
    job_directory = tmp_path / "outputs" / "execution_jobs" / job_id
    job_directory.mkdir(parents=True)
    status = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "transition_state_path",
        "status": "running",
        "metadata": {"software_id": "ash_neb"},
        "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 1},
        "resource_allocation": {"cpu_ids": [0], "gpu_ids": ["0"]},
        "supervisor_pid": 101,
        "child_pid": 102,
    }
    status_path = job_directory / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    assert resource_budget.active_resource_usage() == {
        "cpu_cores": 1,
        "memory_mb": 128,
        "gpu_count": 1,
    }
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "running"
    assert "error" not in persisted


def test_stopped_native_gaussian_preserves_observed_memory_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gaussian 进程组消失后的恢复应保留已经观测到的内存超限原因。"""

    _configure_workspace(monkeypatch, tmp_path)
    job_id = "job_native_memory_exit"
    job_directory = tmp_path / "outputs" / "execution_jobs" / job_id
    job_directory.mkdir(parents=True)
    status_path = job_directory / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "job_type": "native_software",
                "status": "running",
                "supervisor_exit_pending": True,
                "return_code": -15,
                "operation_id": "operation-memory-exit",
                "operation_fingerprint": "f" * 64,
                "resource_usage": {
                    "peak_process_group_rss_kb": 320_000 * 1024 + 1,
                    "memory_limit_mb": 320_000,
                },
            }
        ),
        encoding="utf-8",
    )
    receipt_path = job_directory / "operation_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "job_type": "native_software",
                "operation_id": "operation-memory-exit",
                "operation_fingerprint": "f" * 64,
                "fingerprint": "f" * 64,
                "scope_id": None,
                "scope_workspace": None,
                "phase": "running",
            }
        ),
        encoding="utf-8",
    )
    resource_budget._mark_native_job_supervisor_lost(
        {"workspace": str(tmp_path), "job_id": job_id}
    )

    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["return_code"] == -15
    assert persisted["supervisor_exit_pending"] is False
    assert persisted["process_group_activity"] == "empty"
    assert persisted["error"]["code"] == "memory_limit_exceeded"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["phase"] == "failed"
    assert receipt["status"] == "failed"
    assert receipt["error"]["code"] == "memory_limit_exceeded"


def test_reserve_resources_success_budget_and_pool_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """资源预留覆盖成功、总预算不足和物理资源池不足三个路由。"""

    _configure_workspace(monkeypatch, tmp_path)
    first = reserve_resources(
        {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        kind="analysis",
        label="first",
    )
    assert first.path.is_file()
    with pytest.raises(ResourceBudgetExceeded):
        reserve_resources(
            {"cpu_cores": 2, "memory_mb": 512, "gpu_count": 0},
            kind="analysis",
            label="too-much",
        )
    first.release()

    monkeypatch.setattr(resource_budget, "_cpu_pool", lambda _budget: [])
    with pytest.raises(ResourceBudgetExceeded):
        reserve_resources(
            {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            kind="analysis",
            label="no-pool",
        )
