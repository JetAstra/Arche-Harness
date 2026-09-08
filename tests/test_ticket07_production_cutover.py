"""Ticket 07：公开 research_task 入口固定使用当前 investigation_v2 装配。"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    ResearchTaskApplication,
)
from chemistry_multiagent.controllers.investigation_v2_adapters import (
    build_investigation_v2_module_adapters,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    AUTHOR_MODULE_SEQUENCE,
    ENTRY_STRATEGIES,
    MODULE_SEQUENCE,
    ModuleInvocationRequest,
)
from chemistry_multiagent.controllers.investigation_v2_production import (
    PersistentHarnessRuntimeResolver,
    ResearchTaskProductionAssembly,
)
from chemistry_multiagent.memory import ResearchTaskWorkingMemory
from chemistry_multiagent.memory.v2_identities import StatelessModule


def _digest(value: str) -> str:
    """返回测试使用的 SHA-256 输入摘要。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_public_application_has_one_canonical_research_task_surface() -> None:
    """Application 只保留六个 research_task 公开动作。"""

    public_names = {
        name for name in dir(ResearchTaskApplication) if not name.startswith("_")
    }
    assert {
        "create_standard_research_task",
        "create_author_hypothesis_task",
        "create_top5_confirmation_task",
        "get_hypothesis_confirmation",
        "submit_hypothesis_confirmation_and_continue",
        "resume_research_task",
    }.issubset(public_names)
    assert not {
        "run_investigation_and_experiment_task",
        "run_end_to_end_investigation_task",
        "run_author_hypothesis_investigation_task",
        "run_hypothesis_top5_selection_task",
        "read_hypothesis_confirmation",
        "confirm_hypothesis_execution",
        "resume_investigation_task",
    }.intersection(public_names)


def test_active_module_sequences_and_strategies_are_frozen() -> None:
    """标准和作者策略都使用已登记的模块身份。"""

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


def test_canonical_module_invocation_request_has_no_legacy_identity_fields() -> None:
    """活动 Module Invocation 请求不接受旧任务、运行或摘要字段。"""

    request = ModuleInvocationRequest(
        research_task_id="research-task-ticket07",
        module_invocation_id="module-invocation-ticket07",
        module_kind="hypothesis-import",
        input_scientific_record_ids=(),
        module_input_digest=_digest("ticket07"),
        context_binding=StatelessModule(),
    )
    assert request.to_dict() == {
        "research_task_id": "research-task-ticket07",
        "module_invocation_id": "module-invocation-ticket07",
        "module_kind": "hypothesis-import",
        "input_scientific_record_ids": [],
        "module_input_digest": _digest("ticket07"),
        "context_binding": {"kind": "StatelessModule"},
    }
    legacy_fields = {
        "task_id",
        "inquiry_id",
        "module_run_id",
        "runtime_ref",
        "input_record_refs",
        "input_digest",
    }
    assert not legacy_fields.intersection(request.__dataclass_fields__)
    assert not legacy_fields.intersection(inspect.signature(type(request)).parameters)


def test_default_production_assembly_uses_formal_module_keys(
    tmp_path: Path,
) -> None:
    """默认装配的 adapter key 与模块顺序一致，作者导入单独位于显式模块边界。"""

    memory = ResearchTaskWorkingMemory(tmp_path / "arche_memory.sqlite3")
    modules = build_investigation_v2_module_adapters(
        object(),
        task_root=tmp_path,
        working_memory=memory,
        task_input={
            "scientific_question": "固定问题",
            "retrieval": {
                "pdf_dir": "papers",
                "index_dir": "index",
                "search_papers": False,
                "paper_search": {},
            },
            "supplementary_scientific_input": {},
        },
        policy={
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": "standard_hypothesis_generation",
        },
        dependencies={"catalog": {"status": "deferred_to_investigation_adapter"}},
        task_config={
            "num_queries": 1,
            "num_hypotheses_per_query": 1,
            "top_n_hypotheses": 5,
            "max_parallel_investigations": 1,
            "max_investigation_repairs_per_combination": 1,
            "max_hypothesis_batches": 1,
        },
    )
    assert set(modules) == {
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-import",
        "hypothesis-selection",
        "hypothesis-investigation",
        "task-review",
    }


def test_production_assembly_points_at_research_task_bundle(tmp_path: Path) -> None:
    """生产装配的数据库和 Bundle 路径都固定在 research_tasks 命名空间。"""

    class _Application:
        _run_storage_root = tmp_path
        memory_db_path = None
        _memory_db_explicit = False

    assembly = ResearchTaskProductionAssembly(_Application())
    task_id = "research-task-ticket07"
    assert assembly._database_path_for_task(task_id) == (
        tmp_path
        / "outputs"
        / "research_tasks"
        / task_id
        / "internal"
        / "state"
        / "arche_memory.sqlite3"
    )
    bundle = assembly._new_bundle(
        task_id,
        scientific_question="固定问题",
        task_config={"entry_strategy": "standard_hypothesis_generation"},
    )
    assert bundle.root == tmp_path / "outputs" / "research_tasks" / task_id
    assert not (tmp_path / "outputs" / "runs").exists()


def test_legacy_harness_manifest_is_read_only_boundary(tmp_path: Path) -> None:
    """旧 manifest 仅通过 resolver 转成显式迁移结果。"""

    harness_run_ref = "ticket07-harness-run"
    run_root = tmp_path / "runs" / hashlib.sha256(
        harness_run_ref.encode("utf-8")
    ).hexdigest()
    run_root.mkdir(parents=True)
    digest = _digest("legacy-input")
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "harness_run_ref": harness_run_ref,
                "request": {
                    "task_id": "research-task-ticket07",
                    "module_run_id": "module-invocation-ticket07",
                    "module_kind": "investigation-and-experiment",
                    "runtime_ref": "legacy-context-ticket07",
                    "input_record_refs": ["selected-ticket07"],
                    "input_digest": digest,
                },
                "state": "suspended",
                "failure_code": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    resolved = PersistentHarnessRuntimeResolver(tmp_path).resolve(
        "research-task-ticket07",
        source_event_ids=("source-event-ticket07",),
        expected_module_kind="hypothesis-investigation",
        expected_module_input_digest=digest,
    )
    assert resolved.runtime_ref == "legacy-context-ticket07"
    assert resolved.harness_run_ref == harness_run_ref
