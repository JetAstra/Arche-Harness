"""research_task Working Memory 的 canonical Module Invocation 边界回归。"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from chemistry_multiagent.memory import (
    CASConflict,
    ContractValidationError,
    IdempotencyConflict,
    ModuleInvocationOpen,
    ModuleInvocationRequest,
    NoOpenModuleInvocation,
    OperationConflict,
    ResearchTaskOpen,
    ResearchTaskWorkingMemory,
    StatelessModule,
    WorkingMemoryIntegrityConflict,
)


def _task_kwargs() -> dict[str, object]:
    """返回一份完整且可精确重放的 research_task 创建请求。"""

    return {
        "contract_version": "investigation_v2.task.v2",
        "input": {
            "scientific_question": "验证 research_task 账本边界",
            "chemistry_context": {"solvent": "water", "charge": 0},
        },
        "policy": {
            "entry_strategy": "standard_hypothesis_generation",
            "budgets": {"max_hypothesis_batches": 2},
        },
        "dependencies": {
            "catalog": {"ref": "catalog-1", "digest": "catalog-digest"},
            "model": {"ref": "model-1", "digest": "model-digest"},
        },
    }


def _module_request(
    *,
    research_task_id: str = "research-task-1",
    module_invocation_id: str = "invocation-1",
    module_kind: str = "retrieval",
    module_input_digest: str = "input-digest-1",
) -> ModuleInvocationRequest:
    """构造不携带 Harness 私有身份的 canonical 请求。"""

    return ModuleInvocationRequest(
        research_task_id=research_task_id,
        module_invocation_id=module_invocation_id,
        module_kind=module_kind,
        input_scientific_record_ids=(),
        module_input_digest=module_input_digest,
        context_binding=StatelessModule(),
    )


def _memory(tmp_path: Path) -> ResearchTaskWorkingMemory:
    """在独立临时目录中创建 research_task Working Memory。"""

    return ResearchTaskWorkingMemory(tmp_path / "research_task.sqlite3")


def _create(memory: ResearchTaskWorkingMemory) -> None:
    """创建一项任务并核对初始生命周期。"""

    facts = memory.create_research_task(
        research_task_id="research-task-1",
        **_task_kwargs(),
    )
    assert facts.research_task_id == "research-task-1"
    assert facts.research_task_sequence == 1
    assert isinstance(facts.lifecycle, ResearchTaskOpen)
    assert isinstance(facts.active_module_invocation, NoOpenModuleInvocation)
    assert facts.open_module_invocation is None


def test_create_research_task_freezes_input_and_exact_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    """ResearchTaskCreated 一次写入完整冻结输入，精确重放不追加事件。"""

    memory = _memory(tmp_path)
    first = memory.create_research_task(
        research_task_id="research-task-1",
        **_task_kwargs(),
    )
    replay = memory.create_research_task(
        research_task_id="research-task-1",
        **_task_kwargs(),
    )

    assert replay == first
    assert replay.research_task.input == _task_kwargs()["input"]
    assert replay.research_task.policy == _task_kwargs()["policy"]
    assert replay.research_task.dependencies == _task_kwargs()["dependencies"]
    assert [
        event.event_type
        for event in memory.read_research_task_event_log("research-task-1")
    ] == ["ResearchTaskCreated"]

    drifted = _task_kwargs()
    drifted["policy"] = {"entry_strategy": "author_hypothesis_import"}
    with pytest.raises(IdempotencyConflict):
        memory.create_research_task(
            research_task_id="research-task-1",
            **drifted,
        )


def test_module_invocation_sequence_and_event_digest_are_canonical(
    tmp_path: Path,
) -> None:
    """ModuleInvocationStarted 接在首个任务事实后并链接前序摘要。"""

    memory = _memory(tmp_path)
    _create(memory)
    started = memory.start_module_invocation(
        "research-task-1",
        expected_research_task_sequence=1,
        request=_module_request(),
    )

    events = memory.read_research_task_event_log("research-task-1")
    assert [event.research_task_sequence for event in events] == [1, 2]
    assert events[0].previous_event_digest is None
    assert events[1].previous_event_digest == events[0].event_digest
    assert started.module_invocation_id == "invocation-1"
    assert started.module_input_digest == "input-digest-1"
    assert isinstance(started.lifecycle, ModuleInvocationOpen)
    assert started.context_binding == StatelessModule()

    with pytest.raises((CASConflict, OperationConflict)):
        memory.start_module_invocation(
            "research-task-1",
            expected_research_task_sequence=1,
            request=_module_request(module_invocation_id="invocation-2"),
        )


def test_start_module_invocation_replays_exact_identity_and_rejects_drift(
    tmp_path: Path,
) -> None:
    """同一个 invocation 的精确开始请求可重放，摘要漂移必须拒绝。"""

    memory = _memory(tmp_path)
    _create(memory)
    first = memory.start_module_invocation(
        "research-task-1", 1, _module_request()
    )
    replay = memory.start_module_invocation(
        "research-task-1", 1, _module_request()
    )

    assert replay == first
    assert len(memory.read_research_task_event_log("research-task-1")) == 2
    with pytest.raises(IdempotencyConflict):
        memory.start_module_invocation(
            "research-task-1",
            1,
            _module_request(module_input_digest="input-digest-drift"),
        )


def test_open_lifecycle_is_not_external_running_state(tmp_path: Path) -> None:
    """Open 只表示账本缺少闭合事实，不暴露 running/unknown。"""

    memory = _memory(tmp_path)
    _create(memory)
    memory.start_module_invocation("research-task-1", 1, _module_request())

    facts = memory.read_research_task_facts("research-task-1")
    assert isinstance(facts.lifecycle, ResearchTaskOpen)
    assert facts.active_module_invocation.state == "Open"
    assert facts.open_module_invocation is not None
    assert facts.open_module_invocation.module_invocation_id == "invocation-1"
    serialized = facts.to_public_dict()
    assert "running" not in str(serialized)
    assert "unknown" not in str(serialized).lower()

    with pytest.raises(OperationConflict):
        memory.start_module_invocation(
            "research-task-1",
            2,
            _module_request(module_invocation_id="invocation-2"),
        )


def test_canonical_objects_reject_legacy_constructor_fields() -> None:
    """活动对象不提供旧字段别名；旧 envelope 只走显式 reader。"""

    with pytest.raises(TypeError):
        ModuleInvocationRequest(  # type: ignore[call-arg]
            task_id="legacy-task",
            module_run_id="legacy-invocation",
            module_kind="retrieval",
            input_record_refs=(),
            input_digest="digest",
            runtime_ref="legacy-runtime",
        )

    canonical = _module_request()
    assert not hasattr(canonical, "task_id")
    assert not hasattr(canonical, "module_run_id")
    assert not hasattr(canonical, "runtime_ref")
    assert not {
        "task_id",
        "module_run_id",
        "runtime_ref",
        "input_digest",
    }.intersection(canonical.to_dict())


def test_schema_v10_module_envelope_is_read_only_converted_at_explicit_boundary() -> None:
    """历史 envelope 仅由显式 parser 转成 canonical 请求，不回写旧键。"""

    request = ModuleInvocationRequest.from_schema_v10_mapping(
        {
            "task_id": "legacy-research-task",
            "module_run_id": "legacy-invocation",
            "module_kind": "retrieval",
            "input_record_refs": [],
            "input_digest": "legacy-digest",
            "runtime": {"kind": "CreateRuntime", "runtime_ref": "legacy-runtime"},
        }
    )
    assert request.research_task_id == "legacy-research-task"
    assert request.module_invocation_id == "legacy-invocation"
    assert request.module_kind == "retrieval"
    assert isinstance(request.context_binding, StatelessModule)
    assert not {"task_id", "module_run_id", "runtime_ref"}.intersection(
        request.to_dict()
    )


def test_corrupt_canonical_event_chain_fails_closed(tmp_path: Path) -> None:
    """读取时重算 canonical event chain，损坏不能由 projection 掩盖。"""

    memory = _memory(tmp_path)
    _create(memory)
    memory.start_module_invocation("research-task-1", 1, _module_request())
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute(
            "UPDATE research_task_events SET event_digest = ? "
            "WHERE research_task_id = ? AND research_task_sequence = ?",
            ("0" * 64, "research-task-1", 2),
        )
        connection.commit()

    with pytest.raises(WorkingMemoryIntegrityConflict):
        memory.read_research_task_facts("research-task-1")


def test_memory_off_rejects_access_without_creating_a_database(tmp_path: Path) -> None:
    """关闭记忆时不能静默切换到旧存储或创建账本。"""

    database = tmp_path / "disabled.sqlite3"
    memory = ResearchTaskWorkingMemory(database, enabled=False)
    with pytest.raises(Exception, match="memory|关闭|禁止"):
        memory.create_research_task(
            research_task_id="research-task-disabled",
            **_task_kwargs(),
        )
    assert not database.exists()


def test_unknown_module_kind_is_rejected_by_canonical_request() -> None:
    """模块注册表只接受正式 research_task module kind。"""

    with pytest.raises(ContractValidationError):
        ModuleInvocationRequest(
            research_task_id="research-task-1",
            module_invocation_id="invocation-1",
            module_kind="selection",
            input_scientific_record_ids=(),
            module_input_digest="digest",
            context_binding=StatelessModule(),
        )
