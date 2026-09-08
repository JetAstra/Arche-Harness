"""MiniChem open-execution 三层协议的结构、恢复和清单合同测试。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from chemistry_multiagent.tools.minichem_adapter import (
    MiniChemMcpAdapter,
    MiniChemMcpProtocolError,
)
from chemistry_multiagent.utils.workflow_contract import (
    WorkflowContractError,
    project_workflow_refs,
    resolve_minichem_step_request,
    validate_canonical_workflow,
)
from tests.fakes.minichem_mcp import FakeMcpSession, build_catalog_snapshot


def _recompute_catalog_hash(catalog: dict) -> dict:
    """为修改后的 Catalog v7 快照重新计算 canonical hash。"""

    payload = deepcopy(catalog)
    payload.pop("catalog_hash", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["catalog_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _catalog_with_open_execution() -> dict:
    """给 open execution 合同测试增加精确的冻结能力快照。"""

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
                        "input_mode": "stdin",
                        "required_files": [],
                        "example_request": {},
                    }
                ],
            }
        ],
        "analysis_runtimes": [
            {"runtime": "python", "description": "标准 Python", "modules": []}
        ],
        "authoring_contract": {
            "path_prefix": "code/",
            "max_files": 4,
            "max_chars_per_file": 100000,
        },
    }
    # freeze 层把 open_execution 作为内存叠加，不改变 MCP raw catalog_hash。
    return catalog


def _open_execution_workflow(execution_layer: str) -> dict:
    """构造不含 Action/Backend 身份的 native 或 programmable job 工作流。"""

    if execution_layer == "native_software":
        request = {
            "software_id": "xtb",
            "executable": "xtb",
            "arguments": ["input.xyz"],
            "staged_inputs": [],
            "stdin_target": None,
            "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            "calculation_intent": "single_point",
            "label": "runtime-contract-test",
            "parent_job_id": None,
            "bindings": [],
        }
    elif execution_layer == "programmable_analysis":
        request = {
            "runtime": "python",
            "script_path": "scripts/analysis.py",
            "script_target": "code/agent_program.py",
            "arguments": [],
            "staged_inputs": [],
            "required_modules": [],
            "required_module_versions": {},
            "required_symbols": {},
            "inputs": [],
            "outputs": [],
            "execution_policy": "in_process_only",
            "external_execution": "native_job_runner_only",
            "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            "label": "runtime-contract-test",
            "parent_job_id": None,
            "bindings": [],
        }
    else:
        raise ValueError(f"unsupported execution layer: {execution_layer}")
    return {
        "workflow_name": f"MiniChem {execution_layer} 工作流",
        "goal": "验证 open execution 作业身份恢复边界",
        "strategy_name": f"{execution_layer} 作业",
        "steps": [
            {
                "step_id": "job-step-1",
                "step_name": f"提交 {execution_layer}",
                "description": "提交一次可恢复作业并收集其终态输出。",
                "execution_layer": execution_layer,
                "request": request,
                "expected_outputs": {"execution_layer": execution_layer},
                "depends_on": [],
            }
        ],
    }


def test_open_snapshot_is_required_and_authored_file_must_be_referenced() -> None:
    """旧目录不能新建 open step，且 authored file 不能只出现在文件清单中。"""

    workflow = _open_execution_workflow("native_software")
    with pytest.raises(WorkflowContractError, match="open_execution"):
        validate_canonical_workflow(workflow, build_catalog_snapshot())

    catalog = _catalog_with_open_execution()
    request = workflow["steps"][0]["request"]
    request["authored_files"] = [{"path": "code/input.xyz", "content": "0 1\n"}]
    with pytest.raises(WorkflowContractError, match="至少被请求正文引用"):
        validate_canonical_workflow(workflow, catalog)


def test_authored_file_contract_counts_unicode_characters_not_utf8_bytes() -> None:
    """authoring_contract 的 100000 上限按 Python 字符数计算。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow("native_software")
    request = workflow["steps"][0]["request"]
    request["staged_inputs"] = [{"source_path": "code/input.xyz", "target_path": "input.xyz"}]
    request["authored_files"] = [{"path": "code/input.xyz", "content": "中" * 100000}]
    validate_canonical_workflow(workflow, catalog)
    request["authored_files"][0]["content"] += "中"
    with pytest.raises(WorkflowContractError, match="单文件大小"):
        validate_canonical_workflow(workflow, catalog)


@pytest.mark.parametrize("execution_layer", ["native_software", "programmable_analysis"])
def test_authored_file_must_use_a_rewritable_source_field(execution_layer: str) -> None:
    """label、arguments、output 或 target_path 不能伪装成可写入文件引用。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow(execution_layer)
    request = workflow["steps"][0]["request"]
    request["authored_files"] = [{"path": "code/generated.txt", "content": "content"}]
    if execution_layer == "native_software":
        request["label"] = "code/generated.txt"
        request["arguments"] = ["code/generated.txt"]
    else:
        request["outputs"] = [
            {
                "name": "generated",
                "path": "outputs/generated.txt",
                "semantic_type": "Text",
            }
        ]
        request["inputs"] = [
            {
                "name": "generated",
                "source_path": "existing/input.txt",
                "target_path": "code/generated.txt",
            }
        ]
    with pytest.raises(WorkflowContractError, match="至少被请求正文引用"):
        validate_canonical_workflow(workflow, catalog)


def test_open_only_native_first_step_is_recognized_and_projected_without_action_backend() -> None:
    """首步就是 native open job 时仍识别为 MiniChem，并保持身份字段互斥。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow("native_software")

    validate_canonical_workflow(workflow, catalog)
    projected = project_workflow_refs(workflow)
    step = projected["workflow"]["steps"][0]

    assert step["execution_layer"] == "native_software"
    assert "action_id" not in step
    assert "backend_id" not in step
    assert projected["ref_map"]["step_refs"] == {"s1": "job-step-1"}


@pytest.mark.parametrize("execution_layer", ["native_software", "programmable_analysis"])
@pytest.mark.parametrize("bad_field", ["action_id", "backend_id", "operation_id", "unexpected"])
def test_open_job_request_rejects_action_identity_and_unknown_fields(
    execution_layer: str,
    bad_field: str,
) -> None:
    """Native/Analysis 请求只接受各自精确字段，不能带 Action 或执行身份。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow(execution_layer)
    request = workflow["steps"][0]["request"]
    request[bad_field] = (
        "convert_structure"
        if bad_field == "action_id"
        else "openbabel"
        if bad_field == "backend_id"
        else "operation-from-planner"
        if bad_field == "operation_id"
        else True
    )

    with pytest.raises(WorkflowContractError):
        validate_canonical_workflow(workflow, catalog)


@pytest.mark.parametrize("execution_layer", ["native_software", "programmable_analysis"])
def test_open_job_request_requires_layer_specific_required_fields(execution_layer: str) -> None:
    """Native 必须有 software/executable，Analysis 必须有 runtime/script_path。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow(execution_layer)
    required = ("software_id", "executable") if execution_layer == "native_software" else (
        "runtime",
        "script_path",
    )
    workflow["steps"][0]["request"].pop(required[0])

    with pytest.raises(WorkflowContractError):
        validate_canonical_workflow(workflow, catalog)


def test_open_binding_fields_resolve_only_from_completed_step_and_keep_request_schema() -> None:
    """open 请求可声明来源、来源字段和目标字段，解析时只写入尚不存在的字段。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow("native_software")
    first = workflow["steps"][0]
    first["request"]["label"] = "first-job"
    second_request = deepcopy(first["request"])
    second_request["label"] = "second-job"
    second_request.pop("parent_job_id")
    second_request["bindings"] = [
        {
            "source_step_ref": "job-step-1",
            "source_field": "/job_id",
            "target_field": "/parent_job_id",
        }
    ]
    workflow["steps"].append(
        {
            "step_id": "job-step-2",
            "step_name": "依赖前一作业",
            "description": "用前一作业的 job_id 建立父作业关系。",
            "execution_layer": "native_software",
            "request": second_request,
            "expected_outputs": {"execution_layer": "native_software"},
            "depends_on": ["job-step-1"],
        }
    )

    validate_canonical_workflow(workflow, catalog)
    resolved = resolve_minichem_step_request(
        workflow["steps"][1],
        {"job-step-1": {"job_id": "job_" + "a" * 32}},
    )
    assert resolved["parent_job_id"] == "job_" + "a" * 32
    assert "bindings" not in resolved


def test_open_binding_can_fill_temporary_null_staged_source_path() -> None:
    """绑定解析允许把 canonical 阶段的 null source_path 填成已完成步骤路径。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow("native_software")
    request = workflow["steps"][0]["request"]
    request["staged_inputs"] = [{"source_path": None, "target_path": "input.xyz"}]
    request["bindings"] = [
        {
            "source_step_ref": "step-source-id",
            "source_field": "/workspace_path",
            "target_field": "staged_inputs.0.source_path",
        }
    ]
    workflow["steps"][0]["depends_on"] = ["step-source-id"]
    # 首步的依赖顺序由 canonical 合同检查；这里仅验证请求解析边界，使用已完成来源。
    resolved = resolve_minichem_step_request(
        workflow["steps"][0],
        {"step-source-id": {"workspace_path": "code/source/input.xyz"}},
    )
    assert resolved["staged_inputs"][0]["source_path"] == "code/source/input.xyz"


@pytest.mark.parametrize(
    ("mismatch_field", "replacement"),
    [
        ("execution_layer", "programmable_analysis"),
        ("job_id", "job_" + "b" * 32),
        ("operation_id", "different-operation"),
        ("request_ref", {"relative_path": "request_missing.json"}),
        ("catalog_hash", "f" * 64),
    ],
)
def _direct_adapter(tmp_path: Path, session: FakeMcpSession) -> MiniChemMcpAdapter:
    """创建只使用当前 V2 适配层的确定性 MiniChem client。"""

    adapter = MiniChemMcpAdapter(
        session_factory=lambda session=session: session,
        operation_root=tmp_path / "operations",
        poll_interval=0.0,
    )
    adapter.open()
    adapter.discover_catalog()
    return adapter


def test_adapter_validates_and_submits_native_job_with_operation_identity(
    tmp_path: Path,
) -> None:
    """V2 适配层在校验和提交时固定同一个 operation identity。"""

    session = FakeMcpSession()
    adapter = _direct_adapter(tmp_path, session)
    request = deepcopy(_open_execution_workflow("native_software")["steps"][0]["request"])
    request.pop("bindings")
    try:
        checked = adapter.validate_native_job(request, "native-validation")
        assert checked["status"] == "success"
        validation_call = next(
            call for call in session.calls if call.name == "validate_native_job"
        )
        assert validation_call.arguments["request"]["operation_id"] == "native-validation"

        submitted = adapter.submit_native_job(request, "native-validation")
        assert submitted["status"] == "waiting"
        assert submitted["operation_id"] == "native-validation"
        assert session.native_submit_count == 1
    finally:
        adapter.close()

def test_adapter_validates_analysis_and_collects_the_same_job(
    tmp_path: Path,
) -> None:
    """programmable analysis 通过校验后只收集提交返回的同一 job。"""

    session = FakeMcpSession()
    adapter = _direct_adapter(tmp_path, session)
    request = deepcopy(
        _open_execution_workflow("programmable_analysis")["steps"][0]["request"]
    )
    request.pop("bindings")
    # Analysis 的 source_path 必须是工作区内已存在的输入；本测试只验证
    # job/operation 传递，因此使用空输入输出声明。
    request["script_path"] = "code/analysis.py"
    try:
        checked = adapter.validate_analysis_program(request, "analysis-validation")
        assert checked["operation_id"] == "analysis-validation"
        submitted = adapter.submit_analysis_program(request, "analysis-operation")
        job_id = submitted["job_id"]
        assert submitted["status"] == "waiting"
        adapter.get_execution_job(job_id)
        terminal = adapter.get_execution_job(job_id)
        assert terminal["job_id"] == job_id
        collected = adapter.collect_execution_job(job_id)
        assert collected["job_id"] == job_id
        assert collected["job_status"] == "success"
        assert collected["software_status"] == "not_applicable"
        assert collected["artifact_status"] == "valid"
        assert collected["outputs"]
        assert session.analysis_submit_count == 1
    finally:
        adapter.close()


def test_adapter_declares_artifact_with_parent_refs_and_operation(
    tmp_path: Path,
) -> None:
    """Artifact 声明保留调用方提供的父产物身份，不改写来源。"""

    session = FakeMcpSession()
    adapter = _direct_adapter(tmp_path, session)
    try:
        result = adapter.declare_scientific_artifact(
            {
                "path": "outputs/analysis.json",
                "semantic_type": "AnalysisResult",
                "media_type": "application/json",
                "producer_layer": "programmable_analysis",
                "producer_id": "python",
                "parent_artifact_ids": ["art-parent"],
            },
            operation_id="artifact-operation",
        )
        artifact = result["artifact"]
        assert artifact["parent_artifact_ids"] == ["art-parent"]
        declaration = next(
            call for call in session.calls
            if call.name == "declare_scientific_artifact"
        )
        assert declaration.arguments["request"]["operation_id"] == "artifact-operation"
    finally:
        adapter.close()


def test_program_gate_recheck_rejects_invalid_saved_status_axes() -> None:
    """最终发布再次读取已保存 Gate 时，非法状态轴必须 fail closed。"""

    from chemistry_multiagent.utils.router_evidence import (
        validate_program_gate_for_evidence_ready,
    )

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

def test_catalog_v7_rejects_malformed_policy_types() -> None:
    """Catalog v7 的策略字段类型损坏时，adapter 不得冻结该快照。"""

    for field, value in (
        ("exposure_policy", {"mode": "progressive"}),
        ("provider_selection_policies", "agent_backend_required"),
        ("task_specific_tool_filtering", "false"),
        ("automatic_fallback", []),
        ("evaluation_resource_budget", []),
    ):
        catalog = build_catalog_snapshot()
        catalog[field] = value
        catalog = _recompute_catalog_hash(catalog)
        session = FakeMcpSession(catalog=catalog)
        adapter = MiniChemMcpAdapter(session_factory=lambda session=session: session)
        adapter.open()
        try:
            with pytest.raises(MiniChemMcpProtocolError):
                adapter.discover_catalog()
        finally:
            adapter.close()


def test_operation_id_request_fingerprint_conflict_is_technical_pause(tmp_path: Path) -> None:
    """同一 operation_id 但请求内容变化时返回冲突，且不重复提交远端作业。"""

    session = FakeMcpSession()
    adapter = MiniChemMcpAdapter(
        session_factory=lambda session=session: session,
        operation_root=tmp_path / "operations",
        poll_interval=0.0,
    )
    adapter.open()
    request = _open_execution_workflow("native_software")["steps"][0]["request"]
    # bindings 只属于 ARCHE 工作流解析层；真实 NativeJobRequest 不接受该字段。
    request.pop("bindings", None)
    try:
        first = adapter.submit_native_job(request, "fingerprint-conflict")
        changed = deepcopy(request)
        changed["arguments"] = ["different.xyz"]
        conflict = adapter.submit_native_job(changed, "fingerprint-conflict")
        assert first["status"] == "waiting"
        assert conflict["status"] == "technical_pause"
        assert conflict["minichem_status"] == "operation_id_conflict"
        assert conflict["error"]["code"] == "operation_id_conflict"
        assert session.native_submit_count == 1
    finally:
        adapter.close()


def test_action_backend_must_be_declared_provider_before_execution() -> None:
    """Action 的 backend_id 必须属于冻结目录声明的 provider 集合。"""

    session = FakeMcpSession()
    adapter = MiniChemMcpAdapter(session_factory=lambda session=session: session)
    adapter.open()
    try:
        adapter.discover_catalog()
        with pytest.raises(MiniChemMcpProtocolError):
            adapter.execute_action(
                {
                    "action_id": "convert_structure",
                    "backend_id": "undeclared-provider",
                    "inputs": {"structure": {"smiles": "O"}},
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": {},
                }
            )
    finally:
        adapter.close()
