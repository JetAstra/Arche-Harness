"""research_task 公开应用合同的 tracer bullet 测试。"""

from __future__ import annotations

import inspect
import hashlib
import json
import os
import sqlite3
import subprocess
import sys

import pytest
import chemistry_multiagent.memory as memory
import chemistry_multiagent.controllers as controllers

import chemistry_multiagent.controllers.investigation_v2_module_controller as module_controller

from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    _cli_result_view,
    ResearchTaskApplication,
)
from chemistry_multiagent.controllers.investigation_v2_production import (
    PersistentHarnessRuntimeResolver,
)
from chemistry_multiagent.controllers.investigation_v2_confirmation import (
    CONFIRMATION_MODULE_KIND,
    HypothesisConfirmationView,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ENTRY_STRATEGIES,
    AUTHOR_MODULE_SEQUENCE,
    HarnessModuleRunner,
    ResearchWorkflowControllerContractError,
    MODULE_SEQUENCE,
    ModuleInvocationRequest as ControllerModuleInvocationRequest,
    ResearchWorkflowController,
)
from chemistry_multiagent.memory.v2_identities import (
    CreateInvestigationContext,
    StatelessModule,
)
from chemistry_multiagent.memory.v2_ledger import (
    ModuleInvocationRequest as LedgerModuleInvocationRequest,
    ModuleInvocationSnapshot,
    PreparedModuleInvocationOutput,
    ResearchTaskSnapshot,
    ScientificRecord,
)
from chemistry_multiagent.memory.schema_v10_compat import (
    SchemaV10ReadOnlyError,
    SchemaV10ReadOnlyWorkingMemoryReader,
)


def _digest(value: str) -> str:
    """返回测试使用的合法摘要。"""

    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_public_module_request_uses_research_task_identity_names() -> None:
    """公开 Python mapping 只暴露 research_task 和 invocation 合同字段。"""

    request = ControllerModuleInvocationRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="retrieval",
        input_scientific_record_ids=(),
        module_input_digest=_digest("retrieval"),
        context_binding=StatelessModule(),
    )

    public = request.to_dict()
    assert public["research_task_id"] == "research-task-1"
    assert public["module_invocation_id"] == "invocation-1"
    assert public["input_scientific_record_ids"] == []
    assert public["module_input_digest"] == _digest("retrieval")
    assert "task_id" not in public
    assert "inquiry_id" not in public
    assert "module_run_id" not in public
    assert "runtime_ref" not in public
    assert "input_digest" not in public


def test_working_memory_domain_requests_accept_canonical_identity_names() -> None:
    """V2 Working Memory 的领域请求支持 canonical research_task 字段。"""

    request = LedgerModuleInvocationRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="retrieval",
        input_scientific_record_ids=(),
        module_input_digest=_digest("retrieval"),
        context_binding=StatelessModule(),
    )
    assert request.to_dict() == {
        "research_task_id": "research-task-1",
        "module_invocation_id": "invocation-1",
        "module_kind": "retrieval",
        "input_scientific_record_ids": [],
        "module_input_digest": _digest("retrieval"),
        "context_binding": {"kind": "StatelessModule"},
    }


def test_scientific_record_and_prepared_output_use_canonical_envelopes() -> None:
    """科学记录和密封输出的 mapping 不混入旧账本键。"""

    record = ScientificRecord.from_payload(
        scientific_record_id="retrieval-record-1",
        research_task_id="research-task-1",
        scientific_record_type="RetrievalRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id="invocation-1",
        source_scientific_record_ids=(),
        payload={
            "summary": "summary",
            "citations": [],
            "corpus_snapshot_ref": "corpus-1",
        },
    )
    output = PreparedModuleInvocationOutput.from_records(
        module_invocation_id="invocation-1",
        module_input_digest=_digest("retrieval"),
        output_scientific_records=(record,),
    )

    record_public = record.to_dict()
    output_public = output.to_dict()
    for value in (record_public, output_public):
        assert "research_task_id" in value or "module_invocation_id" in value
        assert not {
            "inquiry_id",
            "task_id",
            "module_run_id",
            "runtime_ref",
            "input_digest",
        }.intersection(value)


def test_module_and_entry_strategy_names_are_formal() -> None:
    """模块顺序和入口策略使用 research_task 的正式词汇。"""

    assert MODULE_SEQUENCE == (
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
        "hypothesis-confirmation",
        "hypothesis-investigation",
        "task-review",
    )
    assert AUTHOR_MODULE_SEQUENCE == (
        "hypothesis-import",
        "hypothesis-confirmation",
        "hypothesis-investigation",
        "task-review",
    )
    assert ENTRY_STRATEGIES == {
        "standard_hypothesis_generation",
        "author_hypothesis_import",
    }
    assert CONFIRMATION_MODULE_KIND == "hypothesis-confirmation"


def test_confirmation_view_and_cli_result_use_research_task_id() -> None:
    """确认视图和 CLI JSON 顶层不得同时返回旧任务身份。"""

    view = HypothesisConfirmationView(
        research_task_id="research-task-1",
        expected_research_task_sequence=1,
        hypothesis_batch_record_id="batch-1",
        candidates=(
            {
                "hypothesis_candidate_record_id": "hypothesis-1",
                "rank": 1,
                "hypothesis": {"claim": "claim"},
            },
        ),
    )
    assert view.to_dict()["research_task_id"] == "research-task-1"
    assert "inquiry_id" not in view.to_dict()

    cli = _cli_result_view(
        {
            "status": "open",
            "research_task_id": "research-task-1",
            "task_id": "legacy-task",
        }
    )
    assert cli["research_task_id"] == "research-task-1"
    assert "task_id" not in cli
    assert "inquiry_id" not in cli


def test_controller_exposes_research_task_actions() -> None:
    """应用 Controller 暴露收敛后的 research_task 动作名。"""

    expected = {
        "create_standard_research_task",
        "create_author_hypothesis_task",
        "create_top5_confirmation_task",
        "get_hypothesis_confirmation",
        "submit_hypothesis_confirmation_and_continue",
        "resume_research_task",
    }
    assert expected.issubset(
        {
            name
            for name in dir(ResearchTaskApplication)
            if not name.startswith("_")
        }
    )


def test_cli_exposes_only_canonical_actions_without_runpy_warning() -> None:
    """CLI 帮助列出六个正式动作，且包导入不会触发 runpy 重复模块警告。"""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chemistry_multiagent.controllers.chemistry_multiagent_controller",
            "--help",
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "RuntimeWarning" not in result.stderr
    for action in (
        "create-standard-research-task",
        "create-author-hypothesis-task",
        "create-top5-confirmation-task",
        "get-hypothesis-confirmation",
        "submit-hypothesis-confirmation-and-continue",
        "resume-research-task",
    ):
        assert action in result.stdout
    for legacy_option in (
        "--resume-task",
        "--confirm-task",
        "--run-mode",
        "top5_selection",
        "end_to_end",
    ):
        assert legacy_option not in result.stdout


def test_legacy_actions_and_identity_fields_are_not_public_contract() -> None:
    """旧动作和旧身份不得成为活动类或 canonical dataclass 字段。"""

    legacy_actions = {
        "run_investigation_and_experiment_task",
        "run_end_to_end_investigation_task",
        "run_author_hypothesis_investigation_task",
        "run_hypothesis_top5_selection_task",
        "read_hypothesis_confirmation",
        "confirm_hypothesis_execution",
        "resume_investigation_task",
    }
    assert not legacy_actions.intersection(
        name for name in dir(ResearchTaskApplication) if not name.startswith("_")
    )
    assert not legacy_actions.intersection(
        name for name in dir(ResearchWorkflowController) if not name.startswith("_")
    )

    legacy_identity_names = {
        "task_id",
        "inquiry_id",
        "module_run_id",
        "runtime_ref",
        "input_digest",
    }
    canonical_models = (
        ControllerModuleInvocationRequest,
        LedgerModuleInvocationRequest,
        ModuleInvocationSnapshot,
        PreparedModuleInvocationOutput,
        ScientificRecord,
        ResearchTaskSnapshot,
    )
    for model in canonical_models:
        assert not legacy_identity_names.intersection(model.__dataclass_fields__)
        assert not legacy_identity_names.intersection(inspect.signature(model).parameters)
    for legacy_model_name in (
        "ModuleRunRequest",
        "ModuleRunSnapshot",
        "PreparedModuleOutput",
        "InvestigationV2WorkingMemoryService",
        "WorkingMemoryService",
    ):
        assert not hasattr(memory, legacy_model_name)

    for legacy_module_name in (
        "ModuleRequest",
        "ModuleOutcome",
        "ModuleRunner",
        "InvestigationV2Controller",
        "InvestigationV2ControllerError",
        "InvestigationV2ControllerContractError",
        "InvestigationV2ControllerIntegrityConflict",
        "TaskClosed",
        "TaskFailed",
    ):
        assert not hasattr(module_controller, legacy_module_name)
        assert not hasattr(controllers, legacy_module_name)


def test_schema_v10_reader_is_fail_closed_and_does_not_mutate_database(tmp_path) -> None:
    """历史读取器使用 SQLite mode=ro，构造和查询不会改动原文件。"""

    db_path = tmp_path / "schema-v10.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE legacy_marker(value TEXT)")
        connection.execute("INSERT INTO legacy_marker(value) VALUES ('kept')")
    before_mtime = db_path.stat().st_mtime_ns

    service = SchemaV10ReadOnlyWorkingMemoryReader(
        str(db_path),
        enabled=True,
        scratch_root=str(tmp_path),
    )

    assert service.status()["memory_status"] == "available"
    assert db_path.stat().st_mtime_ns == before_mtime

    assert not hasattr(service, "record_retrieval_completed")
    with pytest.raises(AttributeError):
        service.record_retrieval_completed  # type: ignore[attr-defined]

    connection = service._reader.store.connect()  # type: ignore[attr-defined]
    try:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE should_not_exist(value TEXT)")
        with pytest.raises(SchemaV10ReadOnlyError):
            with service._reader.store.immediate_transaction():  # type: ignore[attr-defined]
                pass
    finally:
        connection.close()
    assert db_path.stat().st_mtime_ns == before_mtime


def test_harness_module_runner_start_and_resume_use_canonical_request() -> None:
    """Investigation adapter 的 start/resume 都只传递 canonical 请求。"""

    request = ControllerModuleInvocationRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("selected-1",),
        module_input_digest=_digest("selected-1"),
        context_binding=StatelessModule(),
    )

    class FakeHarness:
        def __init__(self) -> None:
            self.requests = []

        def start(self, value):
            self.requests.append(value)
            return output

        def resume(self, value):
            self.requests.append(value)
            return output

    record = ScientificRecord.from_payload(
        scientific_record_id="investigation-result-1",
        research_task_id="research-task-1",
        scientific_record_type="InvestigationResultRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id="invocation-1",
        source_scientific_record_ids=("selected-1",),
        payload={
            "conclusion": "verified",
            "self_judgment": "supported",
            "observations": [],
            "limitations": [],
            "evidence_ref": "evidence-1",
        },
    )
    output = PreparedModuleInvocationOutput.from_records(
        module_invocation_id="invocation-1",
        module_input_digest=_digest("selected-1"),
        output_scientific_records=(record,),
    )

    harness = FakeHarness()
    runner = HarnessModuleRunner(harness)
    with pytest.raises(ResearchWorkflowControllerContractError):
        runner.start(request)

    request = ControllerModuleInvocationRequest(
        research_task_id="research-task-1",
        module_invocation_id="invocation-1",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("selected-1",),
        module_input_digest=_digest("selected-1"),
        context_binding=CreateInvestigationContext("context-1", "batch-1"),
    )
    started = runner.start(request)
    resumed = runner.resume(request)
    for value in (started, resumed):
        assert value.prepared_output.module_invocation_id == "invocation-1"
    for value in harness.requests:
        assert value.research_task_id == "research-task-1"
        assert value.module_invocation_id == "invocation-1"
        assert value.module_kind == "hypothesis-investigation"
        assert value.input_scientific_record_ids == ("selected-1",)
        assert value.module_input_digest == _digest("selected-1")
        assert value.context_binding["kind"] == "CreateInvestigationContext"
        assert not hasattr(value, "task_id")
        assert not hasattr(value, "module_run_id")


def test_persistent_runtime_resolver_reads_legacy_manifest_at_explicit_boundary(
    tmp_path,
) -> None:
    """历史 Harness manifest 只能经显式 reader 转成 canonical 匹配条件。"""

    harness_run_ref = "harness-run-1"
    run_directory = tmp_path / "runs" / hashlib.sha256(
        harness_run_ref.encode("utf-8")
    ).hexdigest()
    run_directory.mkdir(parents=True)
    module_input_digest = _digest("legacy-input")
    (run_directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "harness_run_ref": harness_run_ref,
                "request": {
                    "task_id": "research-task-1",
                    "module_run_id": "invocation-1",
                    "module_kind": "investigation-and-experiment",
                    "runtime_ref": "legacy-context-1",
                    "input_record_refs": ["selected-1"],
                    "input_digest": module_input_digest,
                },
                "state": "suspended",
                "failure_code": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resolved = PersistentHarnessRuntimeResolver(str(tmp_path)).resolve(
        "research-task-1",
        source_event_ids=("source-event-1",),
        expected_module_kind="hypothesis-investigation",
        expected_module_input_digest=module_input_digest,
    )
    assert resolved.runtime_ref == "legacy-context-1"
    assert resolved.harness_run_ref == harness_run_ref
    assert resolved.state == "suspended"
