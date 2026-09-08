"""覆盖 supervisor 新增安全分支，避免复杂安全逻辑只能靠端到端偶然覆盖。"""

from __future__ import annotations

import json
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from minichem_mcp_tools import job_supervisor
from minichem_toolbox.resource_budget import ResourceBudget, ResourceBudgetExceeded
from minichem_toolbox.process_identity import ProcessIdentity


def _queue_context(tmp_path: Path) -> SimpleNamespace:
    """创建只包含队列辅助函数所需字段的轻量 supervisor 上下文。"""

    return SimpleNamespace(
        specification={
            "job_id": "job-crap",
            "job_type": "native_software",
            "metadata": {"label": "gaussian"},
            "gaussian_queue_ticket": {
                "ticket_id": 1,
                "owner_id": "owner",
            },
        },
        job_directory=tmp_path,
        status_path=tmp_path / "status.json",
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        command=["g16"],
        resources={
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
            "walltime_seconds": 10,
        },
        resource_allocation={},
        evaluation_budget={},
        walltime=10,
        execution_class="compute",
        compute_timeout_policy=True,
        hard_timeout=100,
        timeout_policy={},
        memory_limit_mb=256,
        execution_started=False,
        started_at="",
        started_monotonic=0.0,
        cancellation_path=tmp_path / "cancel_requested",
        supervisor_identity=ProcessIdentity(1, 1, "boot", 1, 1),
        child=None,
        child_identity=None,
        cancellation_signal=None,
        preserve_queue_state=False,
        queue_ticket=None,
        queue_reservation=None,
    )


def _complete_identity() -> ProcessIdentity:
    """返回测试中可用于身份核验的完整进程身份。"""

    return ProcessIdentity(123, 456, "boot", 789, 790)


def _status_context(tmp_path: Path) -> SimpleNamespace:
    """补齐状态 JSON 构造所需的公开路径和 operation 字段。"""

    context = _queue_context(tmp_path)
    context.specification.update(
        {
            "relative_job_directory": "job",
            "relative_stdout_path": "job/stdout.log",
            "relative_stderr_path": "job/stderr.log",
            "submitted_at": "2026-01-01T00:00:00+00:00",
            "scope_id": "scope",
            "scope_workspace": str(tmp_path),
            "operation_id": "operation",
            "operation_fingerprint": "fingerprint",
            "submission_key": "submission",
        }
    )
    context.child_identity = _complete_identity()
    return context


def test_explicit_environment_ignores_non_mapping_and_rejects_cache_override(
    tmp_path: Path,
) -> None:
    """显式环境必须是 mapping，模型缓存仍只能由 profile 提供。"""

    assert job_supervisor._explicit_child_environment({"child_environment": []}) == {}
    with pytest.raises(OSError, match="runtime profile"):
        job_supervisor._explicit_child_environment(
            {"child_environment": {"HF_HOME": str(tmp_path)}}
        )
    assert job_supervisor._explicit_child_environment(
        {"child_environment": {"": "ignored", "VISIBLE": 3}}
    ) == {"VISIBLE": "3"}


def test_child_environment_rehomes_explicit_gaussian_scratch_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 Gaussian scratch 和 runtime 都必须被重定向到作业目录。"""

    environment = job_supervisor._build_child_environment(
        {
            "job_id": "job-scratch",
            "runtime": "fairchem",
            "child_environment": {"GAUSS_SCRDIR": "/outside"},
        },
        tmp_path,
    )

    assert environment["GAUSS_SCRDIR"] == str(tmp_path / ".tmp" / "gaussian")
    assert environment["MINICHEM_BACKEND_RUNTIME"] == "fairchem"

    monkeypatch.setenv("MINICHEM_BACKEND_RUNTIME", "trusted-parent")
    inherited = job_supervisor._build_child_environment(
        {"job_id": "job-inherited", "runtime": "fairchem"}, tmp_path / "inherited"
    )
    assert inherited["MINICHEM_BACKEND_RUNTIME"] == "trusted-parent"


def test_child_environment_values_without_runtime_do_not_add_backend_marker(
    tmp_path: Path,
) -> None:
    """没有 runtime 时不应凭空写入后端标记。"""

    temporary, home, matplotlib = job_supervisor._prepare_child_scratch_directories(
        tmp_path
    )
    values = job_supervisor._job_environment_values(
        {"job_id": "job-no-runtime"},
        tmp_path,
        temporary,
        home,
        matplotlib,
        False,
    )

    assert "MINICHEM_BACKEND_RUNTIME" not in values
    assert "GAUSS_SCRDIR" not in values


def test_operation_receipt_updates_only_matching_complete_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """receipt 缺失、损坏、类型错误或身份不匹配时都不能被写入。"""

    specification = {
        "operation_id": "op",
        "operation_fingerprint": "fp",
        "job_id": "job",
        "submission_key": "submission",
    }
    matching_receipt = {
        "operation_id": "op",
        "fingerprint": "fp",
        "job_id": "job",
        "submission_key": "submission",
    }
    receipt_path = tmp_path / "operation_receipt.json"
    job_supervisor._update_operation_receipt(specification, tmp_path, "running")
    receipt_path.write_text("not-json", encoding="utf-8")
    job_supervisor._update_operation_receipt(specification, tmp_path, "running")
    receipt_path.write_text("[]", encoding="utf-8")
    job_supervisor._update_operation_receipt(specification, tmp_path, "running")
    receipt_path.write_text(
        json.dumps({**specification, "operation_id": "other"}), encoding="utf-8"
    )
    job_supervisor._update_operation_receipt(specification, tmp_path, "running")
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["operation_id"] == "other"

    receipt_path.write_text(json.dumps(matching_receipt), encoding="utf-8")
    monkeypatch.setattr(job_supervisor, "_now", lambda: "now")
    job_supervisor._update_operation_receipt(
        specification, tmp_path, "success", status="success"
    )
    updated = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert updated["phase"] == "success"
    assert updated["updated_at"] == "now"
    assert updated["status"] == "success"


def test_operation_receipt_matching_allows_legacy_missing_submission_key(
    tmp_path: Path,
) -> None:
    """旧 receipt 和旧 spec 同时缺少 submission key 时仍可恢复更新。"""

    specification = {
        "operation_id": "op",
        "operation_fingerprint": "fp",
        "job_id": "job",
    }
    receipt = {
        "operation_id": "op",
        "fingerprint": "fp",
        "job_id": "job",
    }
    assert job_supervisor._operation_receipt_matches(receipt, specification)
    assert not job_supervisor._operation_receipt_matches(
        {**receipt, "fingerprint": "changed"}, specification
    )


def test_job_status_value_preserves_identity_scope_operation_and_queue_started_state(
    tmp_path: Path,
) -> None:
    """状态构造同时覆盖排队不计时和运行态身份字段。"""

    context = _status_context(tmp_path)
    queued = job_supervisor._build_job_status_value(context, "queued", {})
    assert queued["started_at"] is None
    assert queued["scope_id"] == "scope"
    assert queued["operation_id"] == "operation"
    assert queued["child_pid"] == 123

    context.execution_started = True
    running = job_supervisor._build_job_status_value(
        context, "running", {"return_code": None}
    )
    assert running["started_at"] == context.started_at
    assert running["return_code"] is None


def test_child_affinity_applies_allocated_ids_and_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """优先使用明确 CPU 列表，否则按 cpu_cores 截取当前允许集合。"""

    calls: list[set[int]] = []
    monkeypatch.setattr(job_supervisor.os, "umask", lambda _mode: None)
    monkeypatch.setattr(job_supervisor.os, "sched_getaffinity", lambda _pid: {2, 3, 4})
    monkeypatch.setattr(
        job_supervisor.os,
        "sched_setaffinity",
        lambda _pid, values: calls.append(set(values)),
    )

    job_supervisor._apply_child_cpu_affinity(
        {"cpu_cores": 1}, {"cpu_ids": [3]}
    )
    job_supervisor._apply_child_cpu_affinity(
        {"cpu_cores": 2}, {"cpu_ids": []}
    )

    assert calls == [{3}, {2, 3}]


def test_child_affinity_rejects_ids_outside_parent_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """资源分配超出 supervisor 亲和性时必须在 child 启动前失败。"""

    monkeypatch.setattr(job_supervisor.os, "umask", lambda _mode: None)
    monkeypatch.setattr(job_supervisor.os, "sched_getaffinity", lambda _pid: {2})
    with pytest.raises(RuntimeError, match="outside the job affinity"):
        job_supervisor._apply_child_cpu_affinity(
            {"cpu_cores": 1}, {"cpu_ids": [3]}
        )


class _FakeChild:
    """只实现安全终止辅助函数使用的 Popen 最小接口。"""

    def __init__(
        self,
        *,
        running: bool = True,
        wait_timeouts: int = 0,
        poll_sequence: list[int | None] | None = None,
    ) -> None:
        self.pid = 123
        self.returncode = None if running else 0
        self._wait_timeouts = wait_timeouts
        self._poll_sequence = list(poll_sequence or [])

    def poll(self) -> int | None:
        if self._poll_sequence:
            value = self._poll_sequence.pop(0)
            self.returncode = value
            return value
        return self.returncode

    def wait(self, timeout: float) -> int:
        if self._wait_timeouts:
            self._wait_timeouts -= 1
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
        self.returncode = 0
        return 0


@pytest.mark.parametrize(
    ("term_result", "kill_result", "wait_timeouts", "expected"),
    [
        (False, False, 0, None),
        (True, True, 0, True),
        (True, False, 1, False),
        (True, True, 2, False),
    ],
)
def test_finish_verified_unresponsive_child_routes_signal_and_reap_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    term_result: bool,
    kill_result: bool,
    wait_timeouts: int,
    expected: bool | None,
) -> None:
    """身份完整时分别覆盖终止失败、成功、SIGKILL 失败和回收超时。"""

    child = _FakeChild(wait_timeouts=wait_timeouts)
    signals: list[signal.Signals] = []
    outcomes = iter([term_result, kill_result])
    monkeypatch.setattr(
        job_supervisor,
        "_verified_process_group_signal",
        lambda _identity, received: signals.append(received) or next(outcomes),
    )
    monkeypatch.setattr(
        job_supervisor,
        "_child_process_group_is_empty",
        lambda _identity: True,
    )

    assert (
        job_supervisor._finish_verified_unresponsive_child(child, _complete_identity())
        == expected
    )
    if term_result:
        assert signals[0] == signal.SIGTERM


def test_finish_unverified_child_accepts_verified_group_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """身份先不可读、随后 leader 退出且进程组为空时可以安全结束。"""

    child = _FakeChild(poll_sequence=[None, 0])
    identity = _complete_identity()
    monkeypatch.setattr(job_supervisor, "capture_process_identity", lambda _pid: identity)
    monkeypatch.setattr(job_supervisor, "_child_process_group_is_empty", lambda _value: True)

    assert job_supervisor._finish_unverified_child_safely(child) is True


def test_gaussian_reservation_attempt_handles_retry_oversize_and_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一次 reservation 的三种结果分别路由为重试、失败或初始化执行状态。"""

    context = _queue_context(tmp_path)
    monkeypatch.setattr(job_supervisor, "_write_job_status", lambda *_a, **_k: None)
    budget = ResourceBudget(cpu_cores=4, memory_mb=4096, gpu_count=0)
    retry_error = ResourceBudgetExceeded(
        requested={"cpu_cores": 1, "memory_mb": 256, "gpu_count": 0},
        budget=budget,
        reserved={"cpu_cores": 3, "memory_mb": 3840, "gpu_count": 0},
        aggregate=True,
    )
    monkeypatch.setattr(job_supervisor, "reserve_resources", lambda *_a, **_k: (_ for _ in ()).throw(retry_error))
    assert job_supervisor._try_reserve_queued_gaussian_resources(context) is False

    context.resources["cpu_cores"] = 8
    oversize_error = ResourceBudgetExceeded(
        requested={"cpu_cores": 8, "memory_mb": 256, "gpu_count": 0},
        budget=budget,
    )
    monkeypatch.setattr(job_supervisor, "reserve_resources", lambda *_a, **_k: (_ for _ in ()).throw(oversize_error))
    assert job_supervisor._try_reserve_queued_gaussian_resources(context) is None

    context.resources["cpu_cores"] = 1
    reservation = SimpleNamespace(
        resource_limits={"cpu_cores": 1, "memory_mb": 256, "gpu_count": 0},
        resource_allocation={"cpu_ids": [2], "gpu_ids": []},
    )
    monkeypatch.setattr(job_supervisor, "reserve_resources", lambda *_a, **_k: reservation)
    monkeypatch.setattr(job_supervisor, "_now", lambda: "now")
    monkeypatch.setattr(job_supervisor.time, "monotonic", lambda: 4.0)
    assert job_supervisor._try_reserve_queued_gaussian_resources(context) is True
    assert context.execution_started is True
    assert context.resource_allocation == {"cpu_ids": [2], "gpu_ids": []}


def test_gaussian_reservation_waits_again_when_capacity_is_temporarily_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """容量暂满时 FIFO 队首保持等待，下一次成功后才继续执行。"""

    context = _queue_context(tmp_path)
    monkeypatch.setattr(job_supervisor, "_cancellation_requested", lambda _ctx: False)
    results = iter([False, True])
    monkeypatch.setattr(
        job_supervisor,
        "_try_reserve_queued_gaussian_resources",
        lambda _ctx: next(results),
    )
    monkeypatch.setattr(job_supervisor.time, "sleep", lambda _seconds: None)

    assert job_supervisor._reserve_queued_gaussian_resources(context) is True


@pytest.mark.parametrize("result", [None, False, True])
def test_gaussian_reservation_wait_routes_cancel_and_terminal_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: bool | None,
) -> None:
    """取消、容量错误和成功三条队列 reservation 路由都保持原语义。"""

    context = _queue_context(tmp_path)
    monkeypatch.setattr(job_supervisor, "_cancellation_requested", lambda _ctx: result is False)
    monkeypatch.setattr(job_supervisor, "_try_reserve_queued_gaussian_resources", lambda _ctx: result)
    if result is False:
        assert job_supervisor._reserve_queued_gaussian_resources(context) is False
    else:
        assert job_supervisor._reserve_queued_gaussian_resources(context) is result


def test_prepare_gaussian_queue_covers_missing_ticket_state_and_wait_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIFO ticket 解析、队首等待失败和 reservation 结果均可独立测试。"""

    context = _queue_context(tmp_path)
    assert job_supervisor._prepare_gaussian_queue(
        SimpleNamespace(specification={}, queue_ticket=None)
    ) is None

    context.specification["gaussian_queue_ticket"] = {"ticket_id": "bad"}
    monkeypatch.setattr(job_supervisor, "_write_job_status", lambda *_a, **_k: None)
    assert job_supervisor._prepare_gaussian_queue(context) == 1

    context = _queue_context(tmp_path)
    monkeypatch.setattr(job_supervisor, "bind_gaussian_queue_owner", lambda _ticket: None)
    monkeypatch.setattr(job_supervisor, "_write_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(job_supervisor, "wait_for_gaussian_turn", lambda *_a, **_k: False)
    monkeypatch.setattr(job_supervisor, "_cancellation_requested", lambda _ctx: False)
    assert job_supervisor._prepare_gaussian_queue(context) == 1

    context = _queue_context(tmp_path)
    monkeypatch.setattr(job_supervisor, "wait_for_gaussian_turn", lambda *_a, **_k: False)
    monkeypatch.setattr(job_supervisor, "_cancellation_requested", lambda _ctx: True)
    monkeypatch.setattr(job_supervisor, "_cancelled_status", lambda _ctx: None)
    assert job_supervisor._prepare_gaussian_queue(context) == 2


def test_monitor_child_routes_cancel_marker_without_polling_untrusted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取消标记出现时立即发起已核验终止，不再读取不必要的监控信息。"""

    context = _queue_context(tmp_path)
    context.child = _FakeChild()
    context.child_identity = _complete_identity()
    context.cancellation_path.touch()
    signals: list[signal.Signals] = []
    monkeypatch.setattr(
        job_supervisor,
        "_verified_process_group_signal",
        lambda _identity, received: signals.append(received) or True,
    )

    observation = job_supervisor._monitor_child_process(context)

    assert observation["timed_out"] is False
    assert context.cancellation_signal == signal.SIGTERM
    assert signals == [signal.SIGTERM]


def test_monitor_child_marks_memory_exceeded_and_requests_verified_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """进程组 RSS 超限时记录 peak 并走 fail-closed 终止分支。"""

    context = _queue_context(tmp_path)
    context.child = _FakeChild()
    context.child_identity = _complete_identity()
    context.compute_timeout_policy = False
    context.started_monotonic = job_supervisor.time.monotonic()
    monkeypatch.setattr(job_supervisor, "_output_file_state", lambda _paths: ())
    monkeypatch.setattr(
        job_supervisor,
        "_process_group_rss_kb",
        lambda _group_id: context.memory_limit_mb * 1024 + 1,
    )
    signals: list[signal.Signals] = []
    monkeypatch.setattr(
        job_supervisor,
        "_verified_process_group_signal",
        lambda _identity, received: signals.append(received) or True,
    )

    observation = job_supervisor._monitor_child_process(context)

    assert observation["memory_exceeded"] is True
    assert observation["peak_process_group_rss_kb"] == context.memory_limit_mb * 1024 + 1
    assert signals == [signal.SIGTERM]


@pytest.mark.parametrize("reservation_result", [None, False, True])
def test_prepare_gaussian_queue_routes_reservation_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reservation_result: bool | None,
) -> None:
    """队首取得后分别覆盖 reservation 失败、取消和继续执行。"""

    context = _queue_context(tmp_path)
    monkeypatch.setattr(job_supervisor, "bind_gaussian_queue_owner", lambda _ticket: None)
    monkeypatch.setattr(job_supervisor, "_write_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(job_supervisor, "wait_for_gaussian_turn", lambda *_a, **_k: True)
    monkeypatch.setattr(job_supervisor, "_reserve_queued_gaussian_resources", lambda _ctx: reservation_result)
    monkeypatch.setattr(job_supervisor, "_cancelled_status", lambda _ctx: None)

    expected = 1 if reservation_result is None else 2 if not reservation_result else None
    assert job_supervisor._prepare_gaussian_queue(context) == expected


def test_gaussian_gate_path_supports_explicit_legacy_and_invalid_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """锁路径优先使用 spec，旧目录结构可推导，缺失信息返回 None。"""

    explicit = job_supervisor._gaussian_gate_path(
        {"gaussian_serial_gate_path": str(tmp_path / "gate")}
    )
    assert explicit == tmp_path / "gate"
    assert job_supervisor._gaussian_gate_path({}) is None
    with monkeypatch.context() as patch:
        patch.setattr(
            job_supervisor,
            "_absolute_path_without_following_symlinks",
            lambda _path: Path("/job"),
        )
        assert job_supervisor._gaussian_gate_path({"job_directory": str(tmp_path)}) is None
    deep = tmp_path / "a" / "b" / "c" / "job"
    assert job_supervisor._gaussian_gate_path({"job_directory": str(deep)}).name == "gaussian.serial.lock"


def test_supervise_routes_gaussian_gate_and_non_gaussian_without_running_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """supervise 的锁路由只负责互斥，实际执行仍委托给统一 supervisor。"""

    specification_path = tmp_path / "spec.json"
    specification_path.write_text(
        json.dumps({"metadata": {"software_id": "orca"}}), encoding="utf-8"
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        job_supervisor, "_run_supervised_job", lambda path: calls.append(path) or 7
    )
    assert job_supervisor.supervise(specification_path) == 7

    specification_path.write_text(
        json.dumps({
            "metadata": {"software_id": "gaussian"},
            "gaussian_queue_ticket": {"ticket_id": 1},
        }),
        encoding="utf-8",
    )
    assert job_supervisor.supervise(specification_path) == 7

    specification_path.write_text(
        json.dumps({
            "metadata": {"software_id": "gaussian"},
            "gaussian_serial_gate_path": str(tmp_path / "gate"),
        }),
        encoding="utf-8",
    )

    @contextmanager
    def fake_lock(_path: Path):
        yield SimpleNamespace(fileno=lambda: 1)

    monkeypatch.setattr(job_supervisor, "_ensure_private_directory", lambda _path: None)
    monkeypatch.setattr(job_supervisor, "_open_private_lock", fake_lock)
    monkeypatch.setattr(job_supervisor.fcntl, "flock", lambda *_args: None)
    assert job_supervisor.supervise(specification_path) == 7
    assert calls == [specification_path, specification_path, specification_path]
