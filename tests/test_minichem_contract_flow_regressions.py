"""MiniChem V2 跨层合同、作业身份和暂停发布回归测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.tools.minichem_adapter import MiniChemMcpAdapter
from chemistry_multiagent.utils.router_evidence import (
    validate_program_gate_for_evidence_ready,
)
from chemistry_multiagent.utils.workflow_contract import (
    WorkflowContractError,
    resolve_minichem_step_request,
    validate_canonical_workflow,
)
from tests.fakes.minichem_mcp import FakeMcpSession, build_artifact_ref, build_catalog_snapshot

_MISSING = object()


def _action_workflow(
    *,
    action_settings: dict[str, Any] | None = None,
    structure: Any = _MISSING,
    depends_on: list[str] | None = None,
    bindings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """构造显式 Action/Backend 请求，便于执行前合同检查。"""

    return {
        "workflow_name": "MiniChem 跨层回归工作流",
        "goal": "验证跨层合同不会在执行边界丢失。",
        "strategy_name": "确定性回归",
        "steps": [
            {
                "step_id": "step-1",
                "step_name": "执行结构 Action",
                "description": "执行一个有明确输入和输出产物的 MiniChem Action。",
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request": {
                    "inputs": {
                        "structure": (
                            {"smiles": "O=Cc1ccccc1"}
                            if structure is _MISSING
                            else structure
                        )
                    },
                    "method_spec": {},
                    "action_settings": deepcopy(action_settings or {}),
                    "resource_limits": {
                        "cpu_cores": 1,
                        "memory_mb": 128,
                        "gpu_count": 0,
                    },
                    "bindings": deepcopy(bindings or []),
                },
                "expected_outputs": {
                    "action_id": "convert_structure",
                    "action_version": "1.0.0",
                    "primary_output": "AtomicStructure",
                    "output_semantic_types": [],
                },
                "depends_on": list(depends_on or []),
            }
        ],
    }


def _contract_catalog(required_action_settings: list[str]) -> dict[str, Any]:
    """为当前 V2 合同投影叠加 pair-specific 请求要求。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "sections": {
                        "inputs": {"required": []},
                        "method_spec": {"required": []},
                        "action_settings": {"required": list(required_action_settings)},
                        "resource_limits": {"required": []},
                    }
                },
                "output_contract": None,
            }
        ],
    }
    return catalog


def _open_execution_catalog() -> dict[str, Any]:
    """构造 native open-execution 所需的最小冻结目录。"""

    catalog = build_catalog_snapshot()
    catalog["open_execution"] = {
        "schema_version": 1,
        "native_software": [
            {
                "software_id": "xtb",
                "display_name": "xTB",
                "commands": [
                    {
                        "executable": "xtb",
                        "synopsis": "运行 xTB 单点计算",
                        "input_mode": "arguments",
                        "required_files": [],
                        "example_request": {},
                    }
                ],
            }
        ],
        "analysis_runtimes": [],
        "authoring_contract": {
            "path_prefix": "code/",
            "max_files": 4,
            "max_chars_per_file": 100000,
        },
    }
    return catalog


def _open_execution_request() -> dict[str, Any]:
    """构造一个不携带 Action 身份的 native job 请求。"""

    return {
        "software_id": "xtb",
        "executable": "xtb",
        "arguments": ["input.xyz"],
        "staged_inputs": [],
        "stdin_target": None,
        "resource_limits": {
            "cpu_cores": 1,
            "memory_mb": 128,
            "gpu_count": 0,
        },
        "calculation_intent": "single_point",
        "label": "v2-regression",
        "parent_job_id": None,
    }


def test_missing_pair_setting_is_blocked_before_any_mcp_submission() -> None:
    """Action 合同要求字段时，缺字段不能越过 V2 workflow 合同。"""

    catalog = _contract_catalog(["random_seed"])
    with pytest.raises(WorkflowContractError, match="random_seed"):
        validate_canonical_workflow(_action_workflow(), catalog)


def test_named_artifact_binding_does_not_depend_on_output_array_position() -> None:
    """下游请求按 ArtifactRef 语义选择输入，而不是按数组下标猜测。"""

    artifact = build_artifact_ref(
        artifact_id="supplementary-sdf",
        media_type="chemical/x-mdl-sdfile",
        path="artifacts/structure.sdf",
    )
    request = resolve_minichem_step_request(
        {
            "request": {
                "inputs": {"structure": None},
                "bindings": [
                    {
                        "source_step_ref": "step-1",
                        "source_field": "artifact.media_type=chemical/x-mdl-sdfile",
                        "target_field": "inputs.structure",
                    }
                ],
            }
        },
        {
            "step-1": {
                "raw_output": {
                    "output_artifacts": [
                        build_artifact_ref(
                            artifact_id="primary-json",
                            media_type="application/json",
                        ),
                        artifact,
                    ],
                    "result": {},
                }
            }
        },
    )
    assert request["inputs"]["structure"]["artifact_id"] == "supplementary-sdf"
    assert request["inputs"]["structure"]["path"] == "artifacts/structure.sdf"
    assert "/output_artifacts/1" not in str(request)


def test_native_submission_and_collection_keep_one_job_identity(tmp_path: Path) -> None:
    """V2 适配器查询和收集提交返回的同一 job，不重新提交替代作业。"""

    session = FakeMcpSession()
    adapter = MiniChemMcpAdapter(
        session_factory=lambda session=session: session,
        operation_root=tmp_path / "operations",
        poll_interval=0.0,
    )
    adapter.open()
    adapter.discover_catalog()
    try:
        submitted = adapter.submit_native_job(_open_execution_request(), "v2-operation")
        job_id = submitted["job_id"]
        assert submitted["status"] == "waiting"
        first = adapter.get_execution_job(job_id)
        second = adapter.get_execution_job(job_id)
        collected = adapter.collect_execution_job(job_id)
        assert first["job_id"] == second["job_id"] == collected["job_id"] == job_id
        assert collected["job_status"] == "success"
        assert session.native_submit_count == 1
        assert session.collect_count[job_id] == 1
    finally:
        adapter.close()


def test_invalid_program_gate_cannot_publish_evidence_ready() -> None:
    """已保存的 Program Gate 约束存在时，最终发布必须 fail closed。"""

    with pytest.raises(ValueError, match="evidence_ready"):
        validate_program_gate_for_evidence_ready(
            {
                "program_gate_result": {
                    "allowed_decisions": ["evidence_ready"],
                    "constraint_reasons": ["scientific_validation_status invalid"],
                },
                "registered_check_results": [],
            }
        )
