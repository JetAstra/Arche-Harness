"""native Gaussian foreach 的 Router、科学账本和最终视图边界测试。"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from chemistry_multiagent.controllers.result_view import (
    _project_execution_result_for_result_view,
)
from chemistry_multiagent.memory.projections import project_scientific_attempt_ledger
from chemistry_multiagent.utils.router_evidence import (
    build_router_evidence_index,
    build_router_execution_projection,
)


_CATALOG_HASH = "b" * 64


def _artifact(artifact_id: str, path: str, parent_id: str) -> dict[str, Any]:
    """构造一个 native child 输出 ArtifactRef。"""

    return {
        "artifact_id": artifact_id,
        "semantic_type": "GaussianOutput",
        "media_type": "text/plain",
        "sha256": "a" * 64,
        "path": path,
        "producer_action": "native_software:gaussian",
        "producer_backend": "gaussian",
        "parent_artifact_ids": [parent_id],
    }


def _foreach_facts() -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """构造带 pair/candidate 来源的 canonical foreach 事实。"""

    items: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for index, (pair_id, candidate_id, energy) in enumerate(
        (("pair-a", "candidate-a", -1.2), ("pair-b", "candidate-b", -1.1))
    ):
        job_id = f"job_{index + 1:032x}"
        operation_id = f"child-operation-{index + 1}"
        path = f"outputs/{pair_id}.log"
        artifact = _artifact(f"gaussian-output-{index + 1}", path, f"deck-{index + 1}")
        artifacts.append(artifact)
        file_record = {
            "path": path,
            "sha256": "c" * 64,
            "size_bytes": 128 + index,
            "kind": "log",
        }
        child = {
            "status": "completed",
            "operation_id": operation_id,
            "job_id": job_id,
            "job_status": "success",
            "ready": True,
            "result": {
                "energy": energy,
                "unit": "hartree",
                "operation_id": "nested-operation-must-be-hidden",
                "job_id": "nested-job-must-be-hidden",
                "raw_log": "RAW_LOG_MUST_BE_HIDDEN",
            },
            "output_artifacts": [artifact],
            "files": [file_record],
            "outputs": [],
            "declared_artifacts": [],
            "stdout": "CHILD_LOG_MUST_BE_HIDDEN",
        }
        provenance = {
            "pair_id": pair_id,
            "candidate_id": candidate_id,
            "direction": "forward" if index == 0 else "reverse",
            "source_step_ref": "formal-converter-must-be-hidden",
        }
        items.append(
            {
                "index": index,
                "item_id": pair_id,
                "provenance": provenance,
                "child_operation_id": operation_id,
                "job_id": job_id,
                "status": "completed",
                "request_ref": {"relative_path": f"internal/request-{index}.json"},
                "result": child,
            }
        )
        checks.extend(
            [
                {
                    "check_ref": f"c{len(checks) + 1}",
                    "evaluation": "matched",
                    "actual": {
                        "path": path,
                        "sha256": file_record["sha256"],
                        "size_bytes": file_record["size_bytes"],
                        "kind": "log",
                    },
                    "evidence_ref": (
                        f"/execution_result/steps/0/raw_output/result/items/{index}/result/files/0"
                    ),
                    "diagnostic": None,
                }
            ]
        )

    raw_output = {
        "status": "completed",
        "ready": True,
        "execution_layer": "native_software",
        "operation_id": "parent-operation-must-not-be-published-as-job",
        "request_fingerprint": "d" * 64,
        "operation_fingerprint": "e" * 64,
        "job_status": "success",
        "job_ids": [item["job_id"] for item in items],
        "operation_ids": [item["child_operation_id"] for item in items],
        "foreach": {
            "schema_version": 1,
            "execution_mode": "native_foreach",
            "parent_operation_id": "parent-operation-must-not-be-published-as-job",
            "items": items,
        },
        "result": {
            "items": [
                {
                    "item_id": item["item_id"],
                    "provenance": deepcopy(item["provenance"]),
                    "result": deepcopy(item["result"]),
                }
                for item in items
            ],
            "item_count": len(items),
            "failed_item_count": 0,
        },
        "output_artifacts": artifacts,
        "files": [],
        "outputs": [],
        "declared_artifacts": [],
        "provenance": {
            "pair_id": "parent-pair-must-not-be-used",
            "operation_id": "parent-operation-must-be-hidden",
        },
    }
    execution = {
        "total_steps": 1,
        "overall_status": "completed",
        "workflow_outcome": "completed",
        "steps": [
            {
                "step_id": "formal-native",
                "execution_layer": "native_software",
                "software_id": "gaussian16",
                "status": "completed",
                "raw_output": raw_output,
            }
        ],
    }
    workflow = {
        "workflow_name": "foreach Router test",
        "goal": "验证有序 native child 的科学结果",
        "strategy_name": "固定策略",
        "steps": [
            {
                "step_ref": "s1",
                "step_name": "Gaussian foreach",
                "description": "逐项运行 Gaussian",
                "execution_layer": "native_software",
                "request": {"software_id": "gaussian16"},
                "depends_on": [],
            }
        ],
    }
    catalog = [
        {
            "action_id": "mock_action",
            "backend_ids": ["mock_backend"],
        }
    ]
    validation = {
        "registered_check_results": checks,
        "program_gate_result": {
            "allowed_decisions": [
                "evidence_ready",
                "revise_workflow",
                "revise_hypothesis",
            ],
            "constraint_reasons": [],
        },
    }
    return execution, workflow, catalog, validation


def test_router_accepts_foreach_without_parent_job_and_filters_child_runtime_fields() -> None:
    """Router 接受 Program Gate 已验证的 child 集合，但不接收 fanout 内部身份。"""

    execution, workflow, catalog, validation = _foreach_facts()
    projected = build_router_execution_projection(
        execution,
        {"s1": "formal-native"},
        catalog,
    )
    raw = projected["steps"][0]["raw_output"]
    assert raw["operation_id"] is None
    assert raw["job_id"] is None
    assert raw["job"] is None
    assert "foreach" not in raw
    assert raw["result"]["items"][0]["provenance"] == {
        "candidate_id": "candidate-a",
        "direction": "forward",
        "pair_id": "pair-a",
    }
    serialized = json.dumps(raw, ensure_ascii=False)
    for marker in (
        "child-operation-1",
        "nested-operation-must-be-hidden",
        "nested-job-must-be-hidden",
        "RAW_LOG_MUST_BE_HIDDEN",
        "request_ref",
        "source_step_ref",
        "parent-operation-must-not-be-published-as-job",
    ):
        assert marker not in serialized

    projected_validation = deepcopy(validation)
    evidence = build_router_evidence_index(
        projected,
        projected_validation,
        workflow,
        catalog,
    )
    assert any(item["evidence_kind"] == "artifact_ref" for item in evidence)
    assert any(
        item["evidence_ref"].endswith("/raw_output/result/items")
        for item in evidence
    )


@pytest.mark.parametrize("field", ["job_ids", "operation_ids"])
def test_foreach_identity_lists_must_match_ordered_child_manifest(field: str) -> None:
    """parent 身份列表被篡改时不得让 Router 继续接受结果。"""

    execution, _workflow, catalog, _validation = _foreach_facts()
    execution["steps"][0]["raw_output"][field][0] = "tampered"
    with pytest.raises(ValueError, match="canonical 身份或数量非法|child 身份或状态非法"):
        build_router_execution_projection(
            execution,
            {"s1": "formal-native"},
            catalog,
        )


def test_final_result_view_keeps_science_and_provenance_without_foreach_internals() -> None:
    """最终结果保留 pair/candidate 能量，但不发布 manifest、请求、身份或路径。"""

    execution, _workflow, _catalog, _validation = _foreach_facts()
    projected = _project_execution_result_for_result_view(execution)
    raw = projected["steps"][0]["raw_output"]
    assert raw["result"]["items"][0]["provenance"]["pair_id"] == "pair-a"
    assert raw["result"]["items"][0]["result"]["result"]["energy"] == -1.2
    serialized = json.dumps(raw, ensure_ascii=False)
    for marker in (
        "child-operation-1",
        "nested-operation-must-be-hidden",
        "nested-job-must-be-hidden",
        "RAW_LOG_MUST_BE_HIDDEN",
        "request_ref",
        "parent-operation-must-not-be-published-as-job",
        "outputs/pair-a.log",
    ):
        assert marker not in serialized


def test_scientific_ledger_keeps_pair_candidate_provenance_without_runtime_identity() -> None:
    """科学账本把 child 来源作为条件保留，不把作业身份或路径当观测。"""

    execution, workflow, _catalog, _validation = _foreach_facts()
    ledger = project_scientific_attempt_ledger(execution, workflow=workflow)
    observations = ledger["steps"][0]["observations"]
    assert [item["name"] for item in observations] == ["energy", "energy"]
    assert [item["conditions"] for item in observations] == [
        {
            "candidate_id": "candidate-a",
            "direction": "forward",
            "pair_id": "pair-a",
        },
        {
            "candidate_id": "candidate-b",
            "direction": "reverse",
            "pair_id": "pair-b",
        },
    ]
    serialized = json.dumps(ledger, ensure_ascii=False)
    for marker in (
        "operation_id",
        "job_id",
        "outputs/pair-a.log",
        "RAW_LOG_MUST_BE_HIDDEN",
    ):
        assert marker not in serialized


@pytest.mark.parametrize(
    "mutate",
    [
        lambda execution: execution["steps"][0]["raw_output"]["foreach"].update(
            {"schema_version": 2}
        ),
        lambda execution: execution["steps"][0]["raw_output"]["foreach"].update(
            {"parent_operation_id": "wrong-parent"}
        ),
        lambda execution: execution["steps"][0]["raw_output"]["result"].update(
            {"item_count": 1}
        ),
        lambda execution: execution["steps"][0]["raw_output"]["foreach"]["items"][1].update(
            {"item_id": "pair-a"}
        ),
        lambda execution: execution["steps"][0]["raw_output"]["result"]["items"][0][
            "result"
        ].update({"status": "failed"}),
        lambda execution: execution["steps"][0]["raw_output"]["result"]["items"][0][
            "provenance"
        ].update({"pair_id": "wrong-pair"}),
    ],
)
def test_foreach_canonical_cross_checks_fail_closed(mutate: Any) -> None:
    """manifest、计数、逐项身份或 child 状态被篡改时必须停止投影。"""

    execution, _workflow, catalog, _validation = _foreach_facts()
    mutate(execution)
    with pytest.raises(ValueError):
        build_router_execution_projection(
            execution,
            {"s1": "formal-native"},
            catalog,
        )


def test_foreach_result_projection_redacts_embedded_paths_and_camel_case_runtime_keys() -> None:
    """未知嵌套字段也不能绕过运行身份和绝对路径边界。"""

    execution, _workflow, catalog, _validation = _foreach_facts()
    child_result = execution["steps"][0]["raw_output"]["result"]["items"][0][
        "result"
    ]["result"]
    child_result["nested"] = {
        "operationId": "secret-operation",
        "jobId": "secret-job",
        "notes": "见 /home/private/gaussian.log",
    }
    projected = build_router_execution_projection(
        execution,
        {"s1": "formal-native"},
        catalog,
    )
    serialized = json.dumps(projected, ensure_ascii=False)
    assert "secret-operation" not in serialized
    assert "secret-job" not in serialized
    assert "/home/private/gaussian.log" not in serialized
    assert "[路径已隐藏]" in serialized


@pytest.mark.parametrize("projector", ["result_view", "scientific_ledger"])
def test_downstream_foreach_views_reject_tampered_child_identity(projector: str) -> None:
    """研究员视图和科学账本不得绕过 Router 的 child 身份核对。"""

    execution, workflow, _catalog, _validation = _foreach_facts()
    execution["steps"][0]["raw_output"]["operation_ids"][0] = "tampered-operation"
    with pytest.raises(ValueError):
        if projector == "result_view":
            _project_execution_result_for_result_view(execution)
        else:
            project_scientific_attempt_ledger(execution, workflow=workflow)
