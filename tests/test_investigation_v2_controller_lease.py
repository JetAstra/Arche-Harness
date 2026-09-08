"""V2 Controller lease 与 research_task 失锁边界回归。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.controller_lease import (
    ControllerLease,
    ControllerLeaseBusy,
    ControllerLeaseIntegrityError,
    ControllerLeaseLost,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ControllerLeaseLostResult,
    ModuleCompleted,
    ModuleInvocationRequest,
    ModuleSuspended,
    ResearchWorkflowController,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)


@dataclass
class FakeClock:
    """可推进的 Unix 时钟，用于确定性验证 expiry。"""

    value: float = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _task_kwargs() -> dict[str, Any]:
    """返回最小但完整的 V2 TaskCreated 输入。"""

    return {
        "contract_version": "investigation_v2.task.v2",
        "input": {"scientific_question": "lease boundary"},
        "policy": {"entry_strategy": "standard_hypothesis_generation"},
        "dependencies": {"catalog": {"ref": "catalog-1"}},
    }


def _lease(
    runtime_root: Path,
    clock: FakeClock,
    holder_id: str,
    *,
    ttl_seconds: float = 10,
) -> ControllerLease:
    """构造相同任务运行目录下的独立 Controller lease。"""

    return ControllerLease.for_task_root(
        runtime_root,
        research_task_id="research-task-1",
        holder_id=holder_id,
        ttl_seconds=ttl_seconds,
        clock=clock,
    )


def test_lease_persists_holder_token_renewal_and_expiry(tmp_path: Path) -> None:
    """首次获取和续租保存完整 lease 身份，但不依赖 Working Memory。"""

    clock = FakeClock()
    lease = _lease(tmp_path / "runtime", clock, "holder-a").acquire()
    first = lease.read_record()
    assert first is not None
    assert first.holder_id == "holder-a"
    assert first.fencing_token == 1
    assert first.renewed_at == 100
    assert first.expires_at == 110

    clock.advance(3)
    renewed = lease.renew()
    assert renewed.fencing_token == 1
    assert renewed.renewed_at == 103
    assert renewed.expires_at == 113


def test_expired_lease_is_taken_over_and_stale_holder_cannot_release(
    tmp_path: Path,
) -> None:
    """新 holder 接管后旧 token 不能续租或覆盖新记录。"""

    clock = FakeClock()
    runtime_root = tmp_path / "runtime"
    first = _lease(runtime_root, clock, "holder-a").acquire()
    clock.advance(11)
    second = _lease(runtime_root, clock, "holder-b").acquire()
    assert second.fencing_token == 2

    with pytest.raises(ControllerLeaseLost):
        first.release()
    with pytest.raises(ControllerLeaseLost):
        first.renew()
    current = second.read_record()
    assert current is not None
    assert current.holder_id == "holder-b"
    assert current.fencing_token == 2


def test_lease_context_does_not_mask_primary_exception(tmp_path: Path) -> None:
    """主流程异常优先；过期 lease 的释放不能覆盖接管者记录。"""

    clock = FakeClock()
    lease = _lease(tmp_path / "runtime", clock, "holder-a")
    with pytest.raises(RuntimeError, match="primary"):
        with lease:
            clock.advance(11)
            raise RuntimeError("primary")

    current = lease.read_record()
    assert current is not None
    assert current.holder_id == "holder-a"
    assert current.fencing_token == 1


def test_second_live_holder_is_rejected_without_token_bump(tmp_path: Path) -> None:
    """未过期 lease 不允许并发 Controller。"""

    clock = FakeClock()
    runtime_root = tmp_path / "runtime"
    first = _lease(runtime_root, clock, "holder-a").acquire()
    second = _lease(runtime_root, clock, "holder-b")
    with pytest.raises(ControllerLeaseBusy):
        second.acquire()
    assert first.read_record().fencing_token == 1  # type: ignore[union-attr]


def test_release_keeps_token_monotonic_for_next_acquisition(tmp_path: Path) -> None:
    """显式释放也不能让下一次获取回退 fencing token。"""

    clock = FakeClock()
    runtime_root = tmp_path / "runtime"
    first = _lease(runtime_root, clock, "holder-a").acquire()
    first.release()
    second = _lease(runtime_root, clock, "holder-b").acquire()
    assert second.fencing_token == 2


def test_lease_lifecycle_does_not_append_working_memory_events(tmp_path: Path) -> None:
    """lease 是独立运行时记录，不产生科学账本事件。"""

    clock = FakeClock()
    memory = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    memory.create_research_task(research_task_id="research-task-1", **_task_kwargs())
    before = memory.read_research_task_event_log("research-task-1")
    lease = _lease(tmp_path / "runtime", clock, "holder-a").acquire()
    lease.renew()
    lease.release()
    assert memory.read_research_task_event_log("research-task-1") == before


def test_lease_rejects_symlink_record_and_lock(tmp_path: Path) -> None:
    """lease JSON 与保护锁都必须是普通文件。"""

    clock = FakeClock()
    runtime_root = tmp_path / "runtime"
    lease = _lease(runtime_root, clock, "holder-a")
    lease.lease_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    lease.lease_path.symlink_to(outside)
    with pytest.raises(Exception, match="符号链接|普通文件"):
        lease.acquire()

    lease.lease_path.unlink()
    lease.lock_path.unlink()
    lease.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lease.lock_path.symlink_to(outside)
    with pytest.raises(Exception, match="锁文件|符号链接|普通文件"):
        lease.acquire()


def test_lease_rejects_non_private_record_and_lock(tmp_path: Path) -> None:
    """已存在的 lease JSON 与锁权限变宽时按身份不可信处理。"""

    clock = FakeClock()
    lease = _lease(tmp_path / "runtime", clock, "holder-a").acquire()
    lease.release()

    lease.lease_path.chmod(0o644)
    with pytest.raises(ControllerLeaseIntegrityError, match="0600"):
        lease.read_record()

    lease.lease_path.chmod(0o600)
    lease.lock_path.chmod(0o644)
    with pytest.raises(ControllerLeaseIntegrityError, match="0600"):
        lease.read_record()


def _prepared_output(request: ModuleInvocationRequest) -> PreparedModuleInvocationOutput:
    """构造一个不触发模型或化学工具的最小合法输出。"""

    record = ScientificRecord.from_payload(
        scientific_record_id=f"{request.module_invocation_id}:retrieval",
        research_task_id=request.research_task_id,
        scientific_record_type="RetrievalRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=(),
        payload={
            "summary": "deterministic",
            "citations": [],
            "corpus_snapshot_ref": "corpus-1",
        },
    )
    return PreparedModuleInvocationOutput.from_records(
        module_invocation_id=request.module_invocation_id,
        module_input_digest=request.module_input_digest,
        output_scientific_records=(record,),
    )


class _ReleaseBeforeReturnModule:
    """在返回准备输出前释放 lease，模拟模块调用期间失锁。"""

    def __init__(self, lease: ControllerLease) -> None:
        self.lease = lease
        self.calls: list[tuple[str, ModuleInvocationRequest]] = []

    def start(self, request: ModuleInvocationRequest) -> ModuleCompleted:
        self.calls.append(("start", request))
        self.lease.release()
        return ModuleCompleted(_prepared_output(request))

    def resume(self, request: ModuleInvocationRequest) -> ModuleCompleted:
        self.calls.append(("resume", request))
        return ModuleCompleted(_prepared_output(request))


def test_lost_lease_after_runner_return_leaves_open_invocation_without_failure_event(
    tmp_path: Path,
) -> None:
    """失锁后的 output 不能提交 ModuleInvocationCompleted/TaskFailed。"""

    clock = FakeClock()
    lease = _lease(tmp_path / "runtime", clock, "holder-a").acquire()
    memory = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    module = _ReleaseBeforeReturnModule(lease)
    controller = ResearchWorkflowController(
        memory,
        {"retrieval": module},
        lease_guard=lease,
    )
    controller.create_research_task("research-task-1", **_task_kwargs())

    result = controller.resume_research_task("research-task-1")
    assert isinstance(result, ControllerLeaseLostResult)
    facts = memory.read_research_task_facts("research-task-1")
    assert facts.lifecycle.state == "Open"
    assert facts.open_module_invocation is not None
    assert [event.event_type for event in memory.read_research_task_event_log("research-task-1")] == [
        "ResearchTaskCreated",
        "ModuleInvocationStarted",
    ]
    assert [operation for operation, _request in module.calls] == ["start"]


def test_lost_lease_before_new_invocation_start_does_not_write_memory(
    tmp_path: Path,
) -> None:
    """失锁 fence 在 ModuleInvocationStarted 前生效。"""

    clock = FakeClock()
    lease = _lease(tmp_path / "runtime", clock, "holder-a").acquire()
    memory = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")
    controller = ResearchWorkflowController(memory, {}, lease_guard=lease)
    controller.create_research_task("research-task-1", **_task_kwargs())
    lease.release()

    result = controller.resume_research_task("research-task-1")
    assert isinstance(result, ControllerLeaseLostResult)
    facts = memory.read_research_task_facts("research-task-1")
    assert facts.lifecycle.state == "Open"
    assert facts.open_module_invocation is None
    assert [event.event_type for event in memory.read_research_task_event_log("research-task-1")] == [
        "ResearchTaskCreated"
    ]


def test_new_holder_resumes_original_open_invocation(tmp_path: Path) -> None:
    """过期接管只恢复原 Module Invocation，不创建第二个开始事实。"""

    clock = FakeClock()
    runtime_root = tmp_path / "runtime"
    first_lease = _lease(runtime_root, clock, "holder-a").acquire()
    memory = ResearchTaskWorkingMemory(tmp_path / "memory.sqlite3")

    class SuspendModule:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ModuleInvocationRequest]] = []

        def start(self, request: ModuleInvocationRequest) -> ModuleSuspended:
            self.calls.append(("start", request))
            return ModuleSuspended("provider_wait")

        def resume(self, request: ModuleInvocationRequest) -> ModuleSuspended:
            self.calls.append(("resume", request))
            return ModuleSuspended("provider_wait")

    first_module = SuspendModule()
    first_controller = ResearchWorkflowController(
        memory,
        {"retrieval": first_module},
        lease_guard=first_lease,
    )
    first_controller.create_research_task("research-task-1", **_task_kwargs())
    assert first_controller.resume_research_task("research-task-1").status == "Suspended"
    original_request = first_module.calls[0][1]
    events_after_first_holder = memory.read_research_task_event_log("research-task-1")

    clock.advance(11)
    second_lease = _lease(runtime_root, clock, "holder-b").acquire()
    assert second_lease.fencing_token == first_lease.fencing_token + 1
    second_module = SuspendModule()
    second_controller = ResearchWorkflowController(
        memory,
        {"retrieval": second_module},
        lease_guard=second_lease,
    )
    result = second_controller.resume_research_task("research-task-1")

    assert result.status == "Suspended"
    assert second_module.calls[0][0] == "resume"
    assert second_module.calls[0][1].module_invocation_id == original_request.module_invocation_id
    assert second_module.calls[0][1].context_binding == original_request.context_binding
    assert memory.read_research_task_event_log("research-task-1") == events_after_first_holder
