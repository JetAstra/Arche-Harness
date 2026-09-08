"""Ticket 06：V2 Task Facts 到 Run Bundle 的展示投影。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.investigation.v2_run_bundle_projector import (
    InvestigationV2RunBundleProjector,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    ModuleInvocationRequest,
    PreparedModuleInvocationOutput,
    ScientificRecord,
    StatelessModule,
)
from chemistry_multiagent.utils.run_bundle import RunBundle
from tests.fakes.run_bundle import _SchemaV10BundleFixtureWriter


def _task_kwargs() -> dict[str, Any]:
    """返回最小 V2 TaskCreated 输入。"""

    return {
        "contract_version": "investigation_v2.task.v1",
        "input": {"scientific_question": "bundle projection"},
        "policy": {"mode": "test"},
        "dependencies": {"catalog": {"ref": "catalog-1"}},
    }


def _read_json(path: Path) -> dict[str, Any]:
    """读取测试中的 Bundle JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _assert_projection_documents(
    bundle: RunBundle,
    *,
    sequence: int,
    task_lifecycle: str,
    module_lifecycle: str,
    controller_observation: str,
) -> None:
    """核对根文件、阶段文件和 summary 阶段记录的一致来源序号。"""

    status = _read_json(bundle.status_path)
    summary = _read_json(bundle.summary_path)
    assert status["source_task_sequence"] == sequence
    assert summary["source_task_sequence"] == sequence
    for document in (status, summary):
        assert document["task_lifecycle"] == task_lifecycle
        assert document["module_invocation_lifecycle"] == module_lifecycle
        assert document["controller_observation"] == controller_observation
        assert document["lifecycle"] == {
            "task": task_lifecycle,
            "module_invocation": module_lifecycle,
        }
        assert document["controller"]["presence"] == controller_observation
        assert document["status"] not in {"running", "unknown"}
    for stage in bundle.stage_names:
        stage_document = _read_json(bundle.root / "stages" / f"{stage}.json")
        assert stage_document["source_task_sequence"] == sequence
        assert stage_document["summary"]["source_task_sequence"] == sequence
        assert stage_document["summary"]["task_lifecycle"] == task_lifecycle
        assert stage_document["summary"]["module_invocation_lifecycle"] == module_lifecycle
        assert stage_document["summary"]["controller_observation"] == controller_observation
        assert stage_document["summary"]["lifecycle"] == {
            "task": task_lifecycle,
            "module_invocation": module_lifecycle,
        }
        assert summary["stages"][stage]["source_task_sequence"] == sequence


def test_projector_marks_open_task_and_all_bundle_sources(tmp_path: Path) -> None:
    """Open Task 的展示不借用 running/unknown，所有文件引用 WM head。"""

    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    facts = service.create_research_task(
        research_task_id="task-1", **_task_kwargs()
    )
    bundle = RunBundle(
        tmp_path,
        research_task_id="task-1",
    )

    projection = InvestigationV2RunBundleProjector(bundle).project(
        facts,
        controller_present=False,
    )

    assert projection.source_task_sequence == 1
    assert projection.task_lifecycle == "Open"
    assert projection.module_invocation_lifecycle == "NotStarted"
    assert projection.active_module_invocation == "NoOpenModuleInvocation"
    assert projection.controller_observation == "Absent"
    _assert_projection_documents(
        bundle,
        sequence=1,
        task_lifecycle="Open",
        module_lifecycle="NotStarted",
        controller_observation="Absent",
    )


def test_re_export_open_module_is_display_only_and_uses_fresh_sequence(
    tmp_path: Path,
) -> None:
    """展示滞后重导出只读 WM，不调用任何模块且推进 source sequence。"""

    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    service.create_research_task(
        research_task_id="task-1", **_task_kwargs()
    )
    bundle = RunBundle(
        tmp_path,
        research_task_id="task-1",
    )
    projector = InvestigationV2RunBundleProjector(bundle)
    projector.project(
        service.read_research_task_facts("task-1"), controller_present=True
    )
    events_before_start = service.read_research_task_event_log("task-1")

    service.start_module_invocation(
        "task-1",
        1,
        request=ModuleInvocationRequest(
            research_task_id="task-1",
            module_invocation_id="v2-retrieval-1-bundle",
            module_kind="retrieval",
            input_scientific_record_ids=(),
            module_input_digest="digest-retrieval",
            context_binding=StatelessModule(),
        ),
    )
    facts = service.read_research_task_facts("task-1")
    events_before_re_export = service.read_research_task_event_log("task-1")
    result = projector.re_export(facts, controller_present=False)

    assert result.source_task_sequence == facts.research_task_sequence == 2
    assert [event.event_type for event in events_before_start] == [
        "ResearchTaskCreated"
    ]
    assert service.read_research_task_event_log("task-1") == events_before_re_export
    _assert_projection_documents(
        bundle,
        sequence=2,
        task_lifecycle="Open",
        module_lifecycle="Open",
        controller_observation="Absent",
    )
    assert _read_json(bundle.root / "stages" / "literature_retrieval.json")["status"] == "open"


def test_re_export_does_not_require_module_runner(tmp_path: Path) -> None:
    """仅用已经读取的 TaskFacts 即可修复展示文件。"""

    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    facts = service.create_research_task(
        research_task_id="task-1", **_task_kwargs()
    )
    bundle = RunBundle(
        tmp_path,
        research_task_id="task-1",
    )
    projector = InvestigationV2RunBundleProjector(bundle)
    projector.project(facts)
    events_before = service.read_research_task_event_log("task-1")

    # 重新打开同一 Bundle 并只传入缓存的 TaskFacts；这里不存在 modules 参数，
    # 因而可证明 re-export 不会触发模型、Harness 或 MiniChem。
    reopened = RunBundle.open_existing_research_task(tmp_path, "task-1")
    projector = InvestigationV2RunBundleProjector(reopened, working_memory=service)
    projector.re_export("task-1", controller_present=False)

    assert service.read_research_task_event_log("task-1") == events_before
    _assert_projection_documents(
        reopened,
        sequence=1,
        task_lifecycle="Open",
        module_lifecycle="NotStarted",
        controller_observation="Absent",
    )


def test_run_bundle_stage_summary_api_preserves_source_sequence(tmp_path: Path) -> None:
    """底层阶段摘要入口也能保存序号，且使用 research_task 阶段名称。"""

    bundle = RunBundle(
        tmp_path,
        research_task_id="stage-source",
    )
    bundle.write_stage_summary(
        "literature_retrieval",
        {"status": "completed", "answer": "deterministic"},
        source_task_sequence=7,
    )
    stage = _read_json(bundle.root / "stages" / "literature_retrieval.json")
    summary = _read_json(bundle.summary_path)
    assert stage["source_task_sequence"] == 7
    assert stage["entries"][0]["summary"]["source_task_sequence"] == 7
    assert summary["source_task_sequence"] == 7
    assert summary["stages"]["literature_retrieval"]["source_task_sequence"] == 7


def test_run_bundle_stage_status_api_propagates_source_sequence(tmp_path: Path) -> None:
    """阶段状态入口也把同一来源序号传播到根摘要和嵌套阶段摘要。"""

    bundle = RunBundle(
        tmp_path,
        research_task_id="stage-status-source",
    )
    bundle.update_stage_status(
        "literature_retrieval",
        "open",
        summary={"status": "open", "task_lifecycle": "Open"},
        source_task_sequence=4,
    )
    stage = _read_json(bundle.root / "stages" / "literature_retrieval.json")
    summary = _read_json(bundle.summary_path)
    assert stage["source_task_sequence"] == 4
    assert stage["summary"]["source_task_sequence"] == 4
    assert summary["source_task_sequence"] == 4
    assert summary["stages"]["literature_retrieval"]["source_task_sequence"] == 4
    assert summary["stages"]["literature_retrieval"]["summary"]["source_task_sequence"] == 4


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "7"])
def test_run_bundle_rejects_invalid_source_task_sequence(
    tmp_path: Path,
    invalid: Any,
) -> None:
    """展示序号不能被非整数或负数伪造。"""

    bundle = RunBundle(
        tmp_path,
        research_task_id="invalid-source",
    )
    with pytest.raises(ValueError, match="source_task_sequence"):
        bundle.update_status("open", source_task_sequence=invalid)


def _record(
    *,
    research_task_id: str,
    module_invocation_id: str,
    scientific_record_id: str,
    scientific_record_type: str,
    payload: dict[str, Any],
    source_scientific_record_ids: tuple[str, ...] = (),
    scientific_record_schema_version: str | int = 1,
) -> ScientificRecord:
    """通过公开记录工厂构造投影测试使用的科学事实。"""

    return ScientificRecord.from_payload(
        scientific_record_id=scientific_record_id,
        research_task_id=research_task_id,
        scientific_record_type=scientific_record_type,
        scientific_record_schema_version=scientific_record_schema_version,
        producer_module_invocation_id=module_invocation_id,
        source_scientific_record_ids=source_scientific_record_ids,
        payload=payload,
    )


def _seed_confirmation(
    tmp_path: Path,
    *,
    closed: bool,
) -> tuple[
    ResearchTaskWorkingMemory,
    RunBundle,
    str,
]:
    """准备一个两候选 Confirmation Module Run，返回任务和批次引用。"""

    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    task_id = "confirmation-projection"
    service.create_research_task(
        research_task_id=task_id,
        contract_version="investigation_v2.task.v1",
        input={"scientific_question": "投影确认阶段"},
        policy={"mode": "test"},
        dependencies={"catalog": {"ref": "catalog-1"}},
    )
    service.start_module_invocation(
        task_id,
        1,
        ModuleInvocationRequest(
            research_task_id=task_id,
            module_invocation_id="import-run",
            module_kind="hypothesis-import",
            input_scientific_record_ids=(),
            module_input_digest="import-input",
            context_binding=StatelessModule(),
        ),
    )
    batch = _record(
        research_task_id=task_id,
        module_invocation_id="import-run",
        scientific_record_id="author-batch",
        scientific_record_type="HypothesisBatchRecord",
        scientific_record_schema_version="author-import.v2",
        payload={
            "imported_hypotheses": [
                {
                    "hypothesis_ref": f"h{rank}",
                    "source_ref": f"author-source-{rank}",
                    "source_digest": "a" * 64,
                    "hypothesis": {
                        "hypothesis_id": f"h{rank}",
                        "claim": f"claim-{rank}",
                    },
                    "provenance": {"kind": "author"},
                }
                for rank in (1, 2)
            ]
        },
    )
    selected = tuple(
        _record(
            research_task_id=task_id,
            module_invocation_id="import-run",
            scientific_record_id=f"selected-{rank}",
            scientific_record_type="SelectedHypothesisRecord",
            payload={
                "hypothesis_ref": f"h{rank}",
                "rank": rank,
                "hypothesis": {
                    "hypothesis_id": f"h{rank}",
                    "claim": f"claim-{rank}",
                },
            },
            source_scientific_record_ids=(batch.scientific_record_id,),
        )
        for rank in (1, 2)
    )
    service.complete_module_invocation(
        task_id,
        2,
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id="import-run",
            module_input_digest="import-input",
            output_scientific_records=(batch, *selected),
        ),
    )
    facts = service.read_research_task_facts(task_id)
    service.start_module_invocation(
        task_id,
        facts.research_task_sequence,
        ModuleInvocationRequest(
            research_task_id=task_id,
            module_invocation_id="confirmation-run",
            module_kind="hypothesis-confirmation",
            input_scientific_record_ids=tuple(
                record.scientific_record_id for record in selected
            ),
            module_input_digest="confirmation-input",
            context_binding=StatelessModule(),
        ),
    )
    if closed:
        facts = service.read_research_task_facts(task_id)
        decision = _record(
            research_task_id=task_id,
            module_invocation_id="confirmation-run",
            scientific_record_id="decision-record",
            scientific_record_type="HypothesisExecutionDecisionRecord",
            scientific_record_schema_version="hypothesis-confirmation.v2",
            payload={
                "source_hypothesis_batch_record_id": batch.scientific_record_id,
                "confirmation_id": "confirmation-1",
                "decisions": [
                    {
                        "hypothesis_candidate_record_id": "selected-1",
                        "decision": "execute",
                        "execution_rank": 1,
                    },
                    {
                        "hypothesis_candidate_record_id": "selected-2",
                        "decision": "cancel",
                    },
                ],
                "execution_input_scientific_record_ids": ["execution-input-1"],
            },
            source_scientific_record_ids=(batch.scientific_record_id,),
        )
        material = _record(
            research_task_id=task_id,
            module_invocation_id="confirmation-run",
            scientific_record_id="material-record",
            scientific_record_type="ResearchMaterialRecord",
            scientific_record_schema_version="hypothesis-confirmation.v2",
            payload={
                "material_id": "material-1",
                "content_ref": "cas/material-1",
                "sha256": "b" * 64,
                "size_bytes": 128,
                "filename": "matrix.json",
                "media_type": "application/json",
                "description": "四分子二面角原子映射",
            },
        )
        execution_input = _record(
            research_task_id=task_id,
            module_invocation_id="confirmation-run",
            scientific_record_id="execution-input-1",
            scientific_record_type="HypothesisExecutionInputRecord",
            scientific_record_schema_version="hypothesis-confirmation.v2",
            payload={
                "source_selected_hypothesis_record_id": "selected-1",
                "execution_rank": 1,
                "hypothesis": {"hypothesis_id": "h1", "claim": "edited claim"},
                "guidance": "只验证关键成键步骤",
                "material_upload_refs": ["material-1"],
            },
            source_scientific_record_ids=(
                decision.scientific_record_id,
                "selected-1",
                material.scientific_record_id,
            ),
        )
        service.complete_module_invocation(
            task_id,
            facts.research_task_sequence,
            PreparedModuleInvocationOutput.from_records(
                module_invocation_id="confirmation-run",
                module_input_digest="confirmation-input",
                output_scientific_records=(decision, material, execution_input),
            ),
        )
    bundle = RunBundle(
        tmp_path,
        research_task_id=task_id,
    )
    return service, bundle, batch.scientific_record_id


def test_projector_exposes_awaiting_confirmation_in_hypothesis_preparation_stage(
    tmp_path: Path,
) -> None:
    """Open Confirmation 只投影候选和等待原因，不创建任何下游状态。"""

    service, bundle, batch_ref = _seed_confirmation(tmp_path, closed=False)
    facts = service.read_research_task_facts("confirmation-projection")

    projection = InvestigationV2RunBundleProjector(bundle, service).project(facts)

    assert projection.reason_code == "awaiting_researcher_confirmation"
    confirmation = projection.hypothesis_confirmation
    assert confirmation is not None
    assert confirmation["status"] == "awaiting_confirmation"
    assert confirmation["reason_code"] == "awaiting_researcher_confirmation"
    assert (
        confirmation["expected_research_task_sequence"]
        == facts.research_task_sequence
    )
    assert confirmation["hypothesis_batch_ref"] == batch_ref
    assert confirmation["candidate_count"] == 2
    assert [
        item["hypothesis_candidate_record_id"]
        for item in confirmation["candidates"]
    ] == ["selected-1", "selected-2"]
    assert [item["summary"] for item in confirmation["candidates"]] == [
        "claim-1",
        "claim-2",
    ]

    status = _read_json(bundle.status_path)
    stage = _read_json(bundle.root / "stages" / "hypothesis_preparation.json")
    assert status["status"] == "open"
    assert status["reason_code"] == "awaiting_researcher_confirmation"
    assert status["source_task_sequence"] == facts.research_task_sequence
    assert stage["status"] == "open"
    assert stage["summary"]["hypothesis_confirmation"] == confirmation
    assert (
        service.read_research_task_event_log("confirmation-projection")[-1].event_type
        == "ModuleInvocationStarted"
    )


def test_confirmation_projection_requires_scientific_record_read_seam(
    tmp_path: Path,
) -> None:
    """TaskFacts 只有关系索引时，缺少 WM 读取 seam 必须拒绝猜测正文。"""

    service, bundle, _ = _seed_confirmation(tmp_path, closed=False)
    facts = service.read_research_task_facts("confirmation-projection")

    with pytest.raises(ValueError, match="scientific-record 读取 seam"):
        InvestigationV2RunBundleProjector(bundle).project(facts)


def test_projector_exposes_confirmed_decision_and_safe_material_metadata(
    tmp_path: Path,
) -> None:
    """Closed Confirmation 投影执行/取消数量和材料摘要，不泄露正文或外部路径。"""

    service, bundle, _ = _seed_confirmation(tmp_path, closed=True)
    facts = service.read_research_task_facts("confirmation-projection")

    projection = InvestigationV2RunBundleProjector(bundle, service).project(facts)

    assert projection.reason_code is None
    confirmation = projection.hypothesis_confirmation
    assert confirmation is not None
    assert confirmation["status"] == "confirmed"
    assert confirmation["confirmation_id"] == "confirmation-1"
    assert confirmation["candidate_count"] == 2
    assert confirmation["execution_count"] == 1
    assert confirmation["cancel_count"] == 1
    assert confirmation["decisions"] == [
        {
            "hypothesis_candidate_record_id": "selected-1",
            "decision": "execute",
            "execution_rank": 1,
        },
        {
            "hypothesis_candidate_record_id": "selected-2",
            "decision": "cancel",
        },
    ]
    assert confirmation["execution_inputs"] == [
        {
            "hypothesis_candidate_record_id": "selected-1",
            "execution_rank": 1,
            "hypothesis": {"hypothesis_id": "h1", "claim": "edited claim"},
            "guidance": "只验证关键成键步骤",
            "material_ids": ["material-1"],
        }
    ]
    assert [
        item["hypothesis_candidate_record_id"]
        for item in confirmation["candidates"]
    ] == ["selected-1", "selected-2"]
    assert confirmation["materials"] == [
        {
            "material_id": "material-1",
            "filename": "matrix.json",
            "media_type": "application/json",
            "size_bytes": 128,
            "sha256": "b" * 64,
            "description": "四分子二面角原子映射",
        }
    ]
    serialized = json.dumps(confirmation, ensure_ascii=False)
    assert "cas/material-1" not in serialized
    assert "矩阵正文" not in serialized
    assert "thread" not in serialized
    assert "scope" not in serialized
    assert "job" not in serialized


def test_historical_bundle_without_confirmation_keeps_existing_projection_shape(
    tmp_path: Path,
) -> None:
    """没有新确认记录的旧 V2 Bundle 不被补造确认状态。"""

    service = ResearchTaskWorkingMemory(str(tmp_path / "memory.sqlite3"))
    facts = service.create_research_task(
        research_task_id="legacy-top5", **_task_kwargs()
    )
    bundle = _SchemaV10BundleFixtureWriter.create(
        tmp_path,
        "legacy-top5",
        stage_layout="investigation",
    )

    with pytest.raises(RuntimeError, match="历史 Bundle 只读"):
        InvestigationV2RunBundleProjector(bundle, service).project(facts)

    # 历史 Bundle 只能显式只读打开，不能被当前 projector 覆写；迁移器应先
    # 将其转换到 outputs/research_tasks 后再投影。
    status = _read_json(bundle.status_path)
    assert "hypothesis_confirmation" not in status
