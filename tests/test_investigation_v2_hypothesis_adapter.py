"""investigation_v2 Hypothesis adapter 的失败恢复回归。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from minichem_toolbox.catalog import catalog_snapshot

from chemistry_multiagent.agents.hypothesis_agent import HypothesisProtocolError
from chemistry_multiagent.controllers.investigation_v2_adapters import (
    HypothesisModuleAdapter,
)
from chemistry_multiagent.controllers.investigation_v2_module_controller import (
    ModuleCompleted,
    ModuleFailed,
    ModuleInvocationRequest,
    ModuleSuspended,
)
from chemistry_multiagent.memory.v2_identities import StatelessModule


class _WorkingMemory:
    """只提供 Hypothesis adapter 本测试需要的科学记录读取接口。"""

    @staticmethod
    def read_scientific_records(*_args: Any) -> list[Any]:
        return []


class _TransportThenSuccessAgent:
    """首轮留下 transport manifest，恢复轮返回固定排名。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_hypothesis_operation(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            operation_id = kwargs["operation_instance_id"]
            operation_root = (
                Path(kwargs["task_root"])
                / "working_memory"
                / "hypothesis_generation_inputs"
                / operation_id
            )
            operation_root.mkdir(parents=True)
            manifest_path = operation_root / ("manifest_" + "a" * 64 + ".json")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "operation_instance_id": operation_id,
                        "operation": "hypothesis_generation",
                        "prior_manifest_ref": None,
                        "stages": [
                            {
                                "stage_index": 1,
                                "status": "failed",
                                "attempts": [
                                    {"validation_status": "transport_failed"}
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            failure = HypothesisProtocolError("merge", "transport")
            relative_path = manifest_path.relative_to(kwargs["task_root"]).as_posix()
            failure.operation_result = {
                "status": "failed",
                "failure_kind": "transport",
                "manifest_path": str(manifest_path),
                "manifest_ref": {"relative_path": relative_path},
            }
            failure.manifest_path = str(manifest_path)
            failure.manifest_ref = {"relative_path": relative_path}
            raise failure
        groups = [
            {
                "query": f"angle-{query_index}",
                "hypotheses": [
                    {
                        "hypothesis_id": (
                            f"hypothesis-{query_index}-{candidate_index}"
                        ),
                        "strategy_name": (
                            f"strategy-{query_index}-{candidate_index}"
                        ),
                    }
                    for candidate_index in range(1, 4)
                ],
            }
            for query_index in range(1, 5)
        ]
        return {
            "hypothesis_result": {
                "queries": [group["query"] for group in groups],
                "hypotheses_by_query": groups,
                "ranked_strategies": [
                    hypothesis
                    for group in groups
                    for hypothesis in group["hypotheses"]
                ][:5],
            },
            "operation_result": {"status": "validated"},
        }


class _TwelveCandidatesMergedToTenAgent:
    """返回四组各三条原始候选和合并后的十条完整排名。"""

    def __init__(self, group_counts: tuple[int, ...] = (3, 3, 3, 3)) -> None:
        self.group_counts = group_counts
        self.calls = 0

    def run_hypothesis_operation(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        groups = [
            {
                "query": f"angle-{query_index}",
                "hypotheses": [
                    {
                        "hypothesis_id": (
                            f"hypothesis-{query_index}-{candidate_index}"
                        ),
                        "strategy_name": (
                            f"strategy-{query_index}-{candidate_index}"
                        ),
                    }
                    for candidate_index in range(1, candidate_count + 1)
                ],
            }
            for query_index, candidate_count in enumerate(
                self.group_counts,
                start=1,
            )
        ]
        originals = [
            hypothesis
            for group in groups
            for hypothesis in group["hypotheses"]
        ]
        ranked = [
            *originals[:8],
            {"hypothesis_id": "merge-1", "strategy_name": "merged-strategy-1"},
            {"hypothesis_id": "merge-2", "strategy_name": "merged-strategy-2"},
        ]
        return {
            "hypothesis_result": {
                "queries": [group["query"] for group in groups],
                "hypotheses_by_query": groups,
                "ranked_strategies": ranked,
            },
            "operation_result": {"status": "validated"},
        }


def test_hypothesis_transport_failure_suspends_then_reuses_validated_prefix(
    tmp_path: Path,
) -> None:
    """恢复只继承失败 manifest，不重新创建原 operation 身份。"""

    agent = _TransportThenSuccessAgent()
    adapter = HypothesisModuleAdapter(
        controller=SimpleNamespace(hypothesis_agent=agent),
        task_root=tmp_path,
        working_memory=_WorkingMemory(),  # type: ignore[arg-type]
        task_input={"scientific_question": "如何验证乙醇分子式？"},
        policy={
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": "standard_hypothesis_generation",
        },
        dependencies={"catalog_snapshot": catalog_snapshot(include_health=False)},
        task_config={
            "num_queries": 4,
            "num_hypotheses_per_query": 3,
            "top_n_hypotheses": 5,
        },
    )
    request = ModuleInvocationRequest(
        research_task_id="task-hypothesis-recovery",
        module_invocation_id="v2-hypothesis-1-recovery",
        module_kind="hypothesis-generation",
        input_scientific_record_ids=(),
        module_input_digest="input-hypothesis-recovery",
        context_binding=StatelessModule(),
    )

    suspended = adapter.start(request)
    completed = adapter.resume(request)

    assert isinstance(suspended, ModuleSuspended)
    assert suspended.reason_code == "hypothesis_transport_unavailable"
    assert isinstance(completed, ModuleCompleted)
    assert len(
        completed.prepared_output.output_scientific_records[0].payload["hypotheses"]
    ) == 5
    assert [call["operation_instance_id"] for call in agent.calls] == [
        request.module_invocation_id,
        request.module_invocation_id + "-transport-retry-1",
    ]
    assert agent.calls[0].get("prior_manifest_path") is None
    assert agent.calls[1]["prior_manifest_path"].endswith(
        "/manifest_" + "a" * 64 + ".json"
    )


def test_hypothesis_batch_preserves_exact_generated_pool_before_merge(
    tmp_path: Path,
) -> None:
    """Working Memory 同时保存 4×3 原始候选和合并后的完整排名。"""

    adapter = HypothesisModuleAdapter(
        controller=SimpleNamespace(
            hypothesis_agent=_TwelveCandidatesMergedToTenAgent()
        ),
        task_root=tmp_path,
        working_memory=_WorkingMemory(),  # type: ignore[arg-type]
        task_input={"scientific_question": "如何区分两个自拟的小分子机理？"},
        policy={
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": "standard_hypothesis_generation",
        },
        dependencies={"catalog_snapshot": catalog_snapshot(include_health=False)},
        task_config={
            "num_queries": 4,
            "num_hypotheses_per_query": 3,
            "top_n_hypotheses": 5,
        },
    )
    request = ModuleInvocationRequest(
        research_task_id="task-hypothesis-audit",
        module_invocation_id="v2-hypothesis-1-audit",
        module_kind="hypothesis-generation",
        input_scientific_record_ids=(),
        module_input_digest="input-hypothesis-audit",
        context_binding=StatelessModule(),
    )

    completed = adapter.start(request)

    assert isinstance(completed, ModuleCompleted)
    payload = completed.prepared_output.output_scientific_records[0].payload
    assert [len(group["hypotheses"]) for group in payload["hypotheses_by_query"]] == [
        3,
        3,
        3,
        3,
    ]
    assert len(payload["ranked_strategies"]) == 10
    assert payload["hypotheses"] == payload["ranked_strategies"]


def test_hypothesis_batch_rejects_short_group_without_replenishing(
    tmp_path: Path,
) -> None:
    """任一 Query 少于冻结数量时停止，不再次调用 provider 补生成。"""

    agent = _TwelveCandidatesMergedToTenAgent((3, 3, 3, 2))
    adapter = HypothesisModuleAdapter(
        controller=SimpleNamespace(hypothesis_agent=agent),
        task_root=tmp_path,
        working_memory=_WorkingMemory(),  # type: ignore[arg-type]
        task_input={"scientific_question": "如何区分两个自拟的小分子机理？"},
        policy={
            "stage_architecture_version": "investigation_v2",
            "entry_strategy": "standard_hypothesis_generation",
        },
        dependencies={"catalog_snapshot": catalog_snapshot(include_health=False)},
        task_config={
            "num_queries": 4,
            "num_hypotheses_per_query": 3,
            "top_n_hypotheses": 5,
        },
    )
    request = ModuleInvocationRequest(
        research_task_id="task-hypothesis-short",
        module_invocation_id="v2-hypothesis-1-short",
        module_kind="hypothesis-generation",
        input_scientific_record_ids=(),
        module_input_digest="input-hypothesis-short",
        context_binding=StatelessModule(),
    )

    outcome = adapter.start(request)

    assert isinstance(outcome, ModuleFailed)
    assert outcome.failure_code == "verified_integrity_conflict"
    assert outcome.recovery_verification is not None
    assert outcome.recovery_verification.reason_code == (
        "hypothesis_candidate_pool_invalid"
    )
    assert agent.calls == 1
