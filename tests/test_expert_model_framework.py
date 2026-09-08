"""专家模型接入框架的边界、可见性和 Executor 路由测试。"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from chemistry_multiagent.agents.expert_model_agent import (
    DEBATE_JUDGE_PROMPT,
    GAUSSIAN_STEP_REVIEW_PROMPT,
    WORKFLOW_EXPERT_OPINION_PROMPT,
    WORKFLOW_GENERAL_OPINION_PROMPT,
    ExpertModelAgent,
    ExpertModelConfig,
    ExpertModelProtocolError,
)
from chemistry_multiagent.utils.expert_request_receipts import (
    build_expert_request_dispatch_observer,
    build_expert_request_receipt_bundle,
)
from tests.fakes.minichem_mcp import build_catalog_snapshot
from chemistry_multiagent.utils.gaussian_routing import (
    GaussianRoutePolicy,
    STRUCTURED_EXTENSIONS_POLICY_ID,
)


_TASK_CONFIG = {
    "num_queries": 1,
    "num_hypotheses_per_query": 1,
    "top_n_workflows": 1,
    "max_parallel_combinations": 1,
    "max_workflow_revisions_per_combination": 1,
    "max_hypothesis_batches": 1,
}


def _hypothesis() -> dict[str, Any]:
    """返回最小科学假设。"""

    return {
        "hypothesis_id": "hypothesis-1",
        "strategy_name": "Gaussian 频率",
        "claims": [{"statement": "频率结果能够检验假设。", "rationale": "结果可观察。"}],
        "verification_requirements": ["检查频率结果"],
        "core_claim": "频率结果能够检验假设。",
        "reasoning": "结果可观察。",
        "detailed_reasoning": "结果可观察。",
    }


def _workflow() -> dict[str, Any]:
    """返回只用于可见性和状态测试的 Gaussian 工作流。"""

    return {
        "workflow_name": "Gaussian 工作流",
        "goal": "频率结果能够检验假设。",
        "strategy_name": "Gaussian 频率",
        "steps": [
            {
                "step_id": "gaussian-step-1",
                "step_name": "Gaussian 频率",
                "description": "运行 Gaussian 频率计算。",
                "action_id": "optimize_geometry",
                "backend_id": "gaussian",
                "request": {"input": "#p test"},
                "expected_outputs": {},
                "depends_on": [],
            }
        ],
    }


def _structured_policy() -> GaussianRoutePolicy:
    """返回不绑定题面的结构化 Gaussian 策略。"""

    return GaussianRoutePolicy(
        policy_id=STRUCTURED_EXTENSIONS_POLICY_ID,
        question_fingerprint="0" * 64,
    )


def _typed_catalog() -> dict[str, Any]:
    """返回带最小 pair contract 的 typed Gaussian 目录。"""

    catalog = build_catalog_snapshot(
        actions=[
            {
                "id": "scan_reaction_coordinates",
                "version": "1.0.0",
                "category": "reaction_and_kinetics",
                "description": "运行显式反应坐标扫描。",
                "backend_ids": ["gaussian"],
                "required_inputs": ["structure"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "ReactionCoordinateScan",
                "output_semantic_types": ["ReactionCoordinateScan"],
                "data_action": False,
                "batch_safe": False,
            }
        ],
        backends=[
            {
                "id": "gaussian",
                "display_name": "Gaussian",
                "runtime": "native",
                "capabilities": ["scan_reaction_coordinates"],
                "health": {"status": "healthy"},
            }
        ],
    )
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "scan_reaction_coordinates",
                "backend_id": "gaussian",
                "request_contract": {
                    "sections": {
                        "inputs": {"allowed": ["structure"], "optional": [], "required": ["structure"]},
                        "method_spec": {
                            "allowed": ["method", "basis", "charge", "multiplicity"],
                            "optional": ["method", "basis", "charge", "multiplicity"],
                            "required": [],
                        },
                        "action_settings": {
                            "allowed": ["coordinate_type", "value_unit", "atom_indices", "steps"],
                            "optional": ["coordinate_type", "value_unit", "atom_indices", "steps"],
                            "required": [],
                        },
                        "resource_limits": {"allowed": [], "optional": [], "required": []},
                    }
                },
                "output_contract": {},
            }
        ],
    }
    return catalog


def _typed_scan_workflow() -> dict[str, Any]:
    """返回包含结构引用但不暴露内部路径的 typed Gaussian 步骤。"""

    return {
        "workflow_name": "结构化扫描",
        "goal": "检验反应坐标变化。",
        "strategy_name": "反应坐标扫描",
        "steps": [
            {
                "step_id": "typed-scan-1",
                "step_name": "Gaussian 反应坐标扫描",
                "description": "在给定结构上扫描显式反应坐标。",
                "action_id": "scan_reaction_coordinates",
                "backend_id": "gaussian",
                "execution_layer": "predefined_actions",
                "request": {
                    "inputs": {
                        "structure": {
                            "artifact_id": "secret-artifact-id",
                            "path": "/private/working/structure.xyz",
                            "semantic_type": "AtomicStructure",
                        }
                    },
                    "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
                    "action_settings": {
                        "coordinate_type": "bond",
                        "value_unit": "angstrom",
                        "atom_indices": [1, 2],
                        "steps": 4,
                    },
                    "resource_limits": {},
                    "bindings": [],
                },
                "expected_outputs": {"output_semantic_types": ["ReactionCoordinateScan"]},
                "depends_on": [],
            }
        ],
    }


def _gaussian_codegen_converter_workflow() -> dict[str, Any]:
    """返回作者固定的 codegen→XYZ converter Gaussian 准备链。"""

    return {
        "workflow_name": "Gaussian 准备链",
        "goal": "频率结果能够检验假设。",
        "strategy_name": "Gaussian 频率",
        "steps": [
            {
                "step_id": "codegen-step-1",
                "step_name": "生成 Gaussian route",
                "description": "根据频率计算目标生成 Gaussian route。",
                "action_id": "generate_gaussian_code",
                "backend_id": "gaussian_codegen",
                "request": {
                    "inputs": {"user_input": "优化结构并计算振动频率"},
                    "bindings": [],
                },
                "expected_outputs": {"output_semantic_types": ["gaussian_code"]},
                "depends_on": [],
            },
            {
                "step_id": "converter-step-1",
                "step_name": "XYZ 转 Gaussian 输入",
                "description": "把已完成结构转换为 Gaussian 输入文件。",
                "action_id": "xyz_to_gjf",
                "backend_id": "format_conversion",
                "request": {
                    "inputs": {
                        "input_xyz_path": "inputs/optimized.xyz",
                        "route_section": "#p B3LYP/6-31G(d) opt freq",
                        "charge": 0,
                        "multiplicity": 1,
                    },
                    "bindings": [
                        {
                            "source_step_ref": "codegen-step-1",
                            "source_field": "/result/gaussian_code",
                            "target_field": "/inputs/route_section",
                        }
                    ],
                },
                "expected_outputs": {"output_semantic_types": ["gaussian_input"]},
                "depends_on": ["codegen-step-1"],
            },
            {
                "step_id": "native-step-1",
                "step_name": "提交 Gaussian 作业",
                "description": "提交已准备的 Gaussian deck。",
                "action_id": None,
                "execution_layer": "native_software",
                "request": {
                    "software_id": "gaussian",
                    "executable": "g16",
                    "staged_inputs": [
                        {"source_path": None, "target_path": "input.com"}
                    ],
                    "bindings": [
                        {
                            "source_step_ref": "converter-step-1",
                            "source_field": "artifact.semantic_type=GaussianInputDeck",
                            "target_field": "staged_inputs.0.source_path",
                        }
                    ],
                },
                "expected_outputs": {"execution_layer": "native_software"},
                "depends_on": ["converter-step-1"],
            },
        ],
    }


def _gaussian_catalog() -> dict[str, Any]:
    """返回足以验证测试 Gaussian request 的最小 MiniChem 目录。"""

    return build_catalog_snapshot(
        actions=[
            {
                "id": "optimize_geometry",
                "version": "1.0.0",
                "category": "electronic_structure",
                "description": "运行 Gaussian 几何优化。",
                "backend_ids": ["gaussian"],
                "required_inputs": ["input"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": None,
                "output_semantic_types": [],
                "data_action": False,
                "batch_safe": False,
            }
        ],
        backends=[
            {
                "id": "gaussian",
                "display_name": "Gaussian",
                "runtime": "native",
                "capabilities": ["optimize_geometry"],
                "health": {"status": "healthy"},
            }
        ],
    )


def test_explicitly_disabled_expert_feature_does_not_call_model() -> None:
    """显式关闭时不调用 provider，并保持明确 disabled 状态。"""

    calls: list[Any] = []

    def provider(messages: Any) -> Any:
        calls.append(messages)
        return {"status": "approved"}

    agent = ExpertModelAgent(
        ExpertModelConfig(enabled=False, provider=provider)
    )
    result = agent.review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=_workflow(),
        catalog_snapshot=_gaussian_catalog(),
    )
    assert result == {
        "status": "disabled",
        "records": [],
        "overrides": {},
        "requested_transition": None,
        "reason": None,
    }
    assert calls == []
    assert ExpertModelAgent(ExpertModelConfig(enabled=True)).configuration_status == "not_configured"


def test_gaussian_review_only_requires_expert_provider() -> None:
    """typed Gaussian 审查只需要 expert provider，不被通用模型入口阻塞。"""

    calls: list[Any] = []

    def expert_provider(messages: Any) -> dict[str, str]:
        calls.append(messages)
        return {"status": "approved", "reason": "参数与问题一致。"}

    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            expert_provider=expert_provider,
            general_provider=None,
        )
    )
    workflow = _typed_scan_workflow()
    result = agent.review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot=_typed_catalog(),
        gaussian_route_policy=_structured_policy(),
    )
    assert result["status"] == "approved"
    assert len(calls) == 1

    completed = agent.evaluate_completed_workflow(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=_workflow(),
        execution_results={"steps": []},
    )
    assert completed == {"status": "not_configured"}
    assert len(calls) == 1


def test_native_gaussian_step_without_deck_evidence_is_skipped() -> None:
    """native Gaussian 不扩展执行前审查职责。"""

    calls: list[Any] = []

    def provider(messages: Any) -> dict[str, str]:
        calls.append(messages)
        return {"status": "approved", "reason": "参数与问题一致。"}

    workflow = deepcopy(_workflow())
    workflow["steps"] = [
        {
            "step_id": "native-step-1",
            "step_name": "运行电子结构计算",
            "description": "执行已准备好的计算输入。",
            "execution_layer": "native_software",
            "request": {"software_id": "gaussian", "executable": "g16"},
            "expected_outputs": {"execution_layer": "native_software"},
            "depends_on": [],
        }
    ]
    result = ExpertModelAgent(
        ExpertModelConfig(enabled=True, expert_provider=provider)
    ).review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot={},
    )
    assert result["status"] == "approved"
    assert result["records"] == []
    assert calls == []


def test_native_gaussian_review_target_is_skipped() -> None:
    """兼容状态中的显式 native target 不触发执行前模型调用。"""

    workflow = _gaussian_codegen_converter_workflow()
    second_codegen = deepcopy(workflow["steps"][0])
    second_codegen["step_id"] = "codegen-step-2"
    second_converter = deepcopy(workflow["steps"][1])
    second_converter["step_id"] = "converter-step-2"
    second_converter["depends_on"] = ["codegen-step-2"]
    second_converter["request"]["bindings"][0]["source_step_ref"] = "codegen-step-2"
    second_native = deepcopy(workflow["steps"][2])
    second_native["step_id"] = "native-step-2"
    second_native["depends_on"] = ["converter-step-2"]
    second_native["request"]["bindings"][0]["source_step_ref"] = "converter-step-2"
    workflow["steps"].extend([second_codegen, second_converter, second_native])
    calls: list[str | None] = []

    def provider(messages: Any) -> dict[str, str]:
        payload = json.loads(messages[1]["content"])
        calls.append(payload["gaussian_unit"].get("generated_route"))
        return {"status": "approved", "reason": "当前 route 一致。"}

    result = ExpertModelAgent(
        ExpertModelConfig(enabled=True, expert_provider=provider)
    ).review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot={},
        gaussian_code_results={
            "codegen-step-2": {
                "status": "completed",
                "result": {"gaussian_code": "#p M062X/def2SVP opt freq"},
            }
        },
        target_step_id="native-step-2",
    )

    assert result["status"] == "approved"
    assert result["records"] == []
    assert calls == []


def test_multiple_native_gaussian_jobs_are_skipped() -> None:
    """多个 native Gaussian job 均不触发执行前专家。"""

    workflow = _gaussian_codegen_converter_workflow()
    second_codegen = deepcopy(workflow["steps"][0])
    second_codegen["step_id"] = "codegen-step-2"
    second_converter = deepcopy(workflow["steps"][1])
    second_converter["step_id"] = "converter-step-2"
    second_converter["depends_on"] = ["codegen-step-2"]
    second_converter["request"]["bindings"][0]["source_step_ref"] = "codegen-step-2"
    second_native = deepcopy(workflow["steps"][2])
    second_native["step_id"] = "native-step-2"
    second_native["depends_on"] = ["converter-step-2"]
    second_native["request"]["bindings"][0]["source_step_ref"] = "converter-step-2"
    workflow["steps"].extend([second_codegen, second_converter, second_native])
    calls: list[str] = []

    def provider(messages: Any) -> dict[str, str]:
        payload = json.loads(messages[1]["content"])
        calls.append(payload["gaussian_unit"]["generated_route"])
        return {"status": "approved", "reason": "route 可用于当前计算。"}

    result = ExpertModelAgent(
        ExpertModelConfig(enabled=True, expert_provider=provider)
    ).review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot={},
        gaussian_code_results={
            "codegen-step-1": {
                "status": "completed",
                "result": {"gaussian_code": "#p B3LYP/6-31G(d) opt freq"},
            },
            "codegen-step-2": {
                "status": "completed",
                "result": {"gaussian_code": "#p M062X/def2SVP opt freq"},
            },
        },
    )

    assert result["status"] == "approved"
    assert result["records"] == []
    assert calls == []


def test_native_target_without_provider_is_empty_approved() -> None:
    """兼容 native target 不依赖执行前 expert provider。"""

    workflow = _gaussian_codegen_converter_workflow()
    result = ExpertModelAgent(
        ExpertModelConfig(enabled=True, expert_provider=None, general_provider=None)
    ).review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot={},
        target_step_id="native-step-1",
    )

    assert result == {
        "status": "approved",
        "records": [],
        "overrides": {},
        "requested_transition": None,
        "reason": None,
    }


def test_gaussian_dispatch_does_not_persist_native_calls(tmp_path) -> None:
    """native Gaussian 不发送执行前请求，也不落 started 收据。"""

    workflow = _gaussian_codegen_converter_workflow()
    second_codegen = deepcopy(workflow["steps"][0])
    second_codegen["step_id"] = "codegen-step-2"
    second_converter = deepcopy(workflow["steps"][1])
    second_converter["step_id"] = "converter-step-2"
    second_converter["depends_on"] = ["codegen-step-2"]
    second_converter["request"]["bindings"][0]["source_step_ref"] = "codegen-step-2"
    second_native = deepcopy(workflow["steps"][2])
    second_native["step_id"] = "native-step-2"
    second_native["depends_on"] = ["converter-step-2"]
    second_native["request"]["bindings"][0]["source_step_ref"] = "converter-step-2"
    workflow["steps"].extend([second_codegen, second_converter, second_native])
    context = SimpleNamespace(task_root=tmp_path)
    combination = {
        "combination_id": "combination-1",
        "workflow_version": 1,
        "execution_id": "execution-1",
    }
    _, receipt = build_expert_request_receipt_bundle(
        context,
        combination=combination,
        operation_instance_id="gaussian-review-dispatch",
        phase="gaussian_step_review",
        request_descriptor={"workflow": workflow},
        backend_fingerprint="1" * 64,
    )
    observer = build_expert_request_dispatch_observer(context, request_receipt=receipt)
    provider_observations: list[tuple[int, int]] = []

    def provider(_messages: Any) -> dict[str, str]:
        started_files = list(
            (tmp_path / "working_memory" / "expert_requests").rglob(
                "started_request_*.json"
            )
        )
        provider_observations.append((len(started_files), len(receipt["calls"])))
        return {"status": "approved", "reason": "route 可用于当前计算。"}

    result = ExpertModelAgent(
        ExpertModelConfig(enabled=True, expert_provider=provider)
    ).review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot={},
        gaussian_code_results={
            "codegen-step-1": {
                "status": "completed",
                "result": {"gaussian_code": "#p B3LYP/6-31G(d) opt freq"},
            },
            "codegen-step-2": {
                "status": "completed",
                "result": {"gaussian_code": "#p M062X/def2SVP opt freq"},
            },
        },
        request_observer=observer,
    )

    assert result["status"] == "approved"
    assert provider_observations == []
    assert receipt["calls"] == []


def test_reject_review_requires_a_nonempty_reason() -> None:
    """reject 事实必须带可持久化的原因，避免状态合同在提交时才失败。"""

    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            provider=lambda _messages: {"status": "reject", "reason": ""},
        )
    )
    workflow = _typed_scan_workflow()

    with pytest.raises(ExpertModelProtocolError, match="reason"):
        agent.review_gaussian_steps(
            scientific_question="问题",
            hypothesis=_hypothesis(),
            workflow=workflow,
            catalog_snapshot=_typed_catalog(),
            gaussian_route_policy=_structured_policy(),
        )


def test_gaussian_review_rejects_full_step_replacement_payload() -> None:
    """Gaussian revise 必须使用白名单参数 patch，不能返回完整步骤替换。"""

    workflow = _typed_scan_workflow()
    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            provider=lambda _messages: {
                "status": "revise",
                "reason": "需要局部修订。",
                "replacement_step": {},
            },
        )
    )

    with pytest.raises(ExpertModelProtocolError, match="恰含"):
        agent.review_gaussian_steps(
            scientific_question="问题",
            hypothesis=_hypothesis(),
            workflow=workflow,
            catalog_snapshot=_typed_catalog(),
            gaussian_route_policy=_structured_policy(),
        )


def test_scientific_opinion_rejects_empty_summary() -> None:
    """空 summary 会被持久化合同拒绝，模型协议应在调用边界报告。"""

    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            provider=lambda _messages: {
                "opinion": "supports_hypothesis",
                "summary": "",
                "evidence_refs": ["e1"],
            },
        )
    )

    with pytest.raises(ExpertModelProtocolError, match="summary 必须是非空字符串"):
        agent.evaluate_completed_workflow(
            scientific_question="问题",
            hypothesis=_hypothesis(),
            workflow=_workflow(),
            execution_results={"steps": []},
        )


def test_expert_input_rejects_nonstandard_json_numbers() -> None:
    """模型输入不能把 NaN 这类非标准 JSON 数字编码成伪 JSON。"""

    calls: list[Any] = []

    def provider(messages: Any) -> dict[str, str]:
        calls.append(messages)
        return {"status": "approved", "reason": "参数与问题一致。"}

    workflow = _typed_scan_workflow()
    workflow["steps"][0]["request"]["method_spec"]["charge"] = math.nan
    agent = ExpertModelAgent(ExpertModelConfig(enabled=True, provider=provider))

    with pytest.raises(ExpertModelProtocolError, match="严格 JSON"):
        agent.review_gaussian_steps(
            scientific_question="问题",
            hypothesis=_hypothesis(),
            workflow=workflow,
            catalog_snapshot=_typed_catalog(),
            gaussian_route_policy=_structured_policy(),
        )
    assert calls == []


def test_expert_inputs_are_separate_and_judge_sees_only_debate_json() -> None:
    """Gaussian 单元、两份完整意见和最终 Debate 的输入边界互不泄露。"""

    calls: list[tuple[str, str, dict[str, Any]]] = []

    def expert_provider(messages: Any) -> Any:
        assert messages[0]["content"] in {
            GAUSSIAN_STEP_REVIEW_PROMPT,
            WORKFLOW_EXPERT_OPINION_PROMPT,
        }
        payload = json.loads(messages[1]["content"])
        calls.append(("expert", messages[0]["content"], payload))
        if "gaussian_unit" in payload:
            return {"status": "approved", "reason": "参数与问题一致。"}
        assert "workflow_evidence" in payload
        refs = [
            fact["evidence_ref"]
            for step in payload["workflow_evidence"]["steps"]
            for field in ("observations", "program_checks")
            for fact in step[field]
        ]
        refs.extend(
            step[field]["evidence_ref"]
            for step in payload["workflow_evidence"]["steps"]
            for field in ("failure", "not_executed")
            if isinstance(step[field], dict)
        )
        return {
            "opinion": "supports_hypothesis",
            "summary": "结果支持。",
            "evidence_refs": refs[:1],
        }

    def general_provider(messages: Any) -> Any:
        assert messages[0]["content"] in {
            WORKFLOW_GENERAL_OPINION_PROMPT,
            DEBATE_JUDGE_PROMPT,
        }
        payload = json.loads(messages[1]["content"])
        calls.append(("general", messages[0]["content"], payload))
        if "workflow_evidence" in payload:
            refs = [
                fact["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("observations", "program_checks")
                for fact in step[field]
            ]
            refs.extend(
                step[field]["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("failure", "not_executed")
                if isinstance(step.get(field), dict)
            )
            refs.extend(
                step[field]["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("failure", "not_executed")
                if isinstance(step.get(field), dict)
            )
            refs.extend(
                step[field]["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("failure", "not_executed")
                if isinstance(step.get(field), dict)
            )
            return {
                "opinion": "supports_hypothesis",
                "summary": "通用结果支持。",
                "evidence_refs": refs[:1],
            }
        assert set(payload) == {
            "scientific_question",
            "hypothesis",
            "expert_opinion",
            "general_opinion",
            "evidence_index",
        }
        return {
            "opinion": "supports_hypothesis",
            "summary": "辩论支持。",
            "evidence_refs": [item["evidence_ref"] for item in payload["evidence_index"][:1]],
        }

    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            expert_provider=expert_provider,
            general_provider=general_provider,
        )
    )
    gaussian_workflow = _gaussian_codegen_converter_workflow()
    review = agent.review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=gaussian_workflow,
        catalog_snapshot={},
        gaussian_code_results={
            "codegen-step-1": {
                "status": "completed",
                "result": {"gaussian_code": "#p B3LYP/6-31G(d) opt freq"},
            }
        },
        target_step_id="native-step-1",
    )
    completed = agent.evaluate_completed_workflow(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=_workflow(),
        execution_results={"steps": [{"step_id": "gaussian-step-1", "result": {"energy": -1.0}}]},
    )
    assert review["status"] == "approved"
    assert completed["route"] == "evidence_ready"
    assert len(calls) == 3
    assert [item[0] for item in calls] == ["expert", "general", "general"]
    assert set(calls[0][2]) == {
        "scientific_question",
        "explicit_conditions",
        "hypothesis",
        "workflow_evidence",
    }
    assert set(calls[1][2]) == set(calls[0][2])
    assert set(calls[2][2]) == {
        "scientific_question",
        "hypothesis",
        "expert_opinion",
        "general_opinion",
        "evidence_index",
    }


def test_reject_review_pauses_before_executor_and_revise_override_is_local() -> None:
    """revise 只返回白名单局部补丁，不替换工作流身份或原始请求。"""

    workflow = _typed_scan_workflow()
    original = deepcopy(workflow)

    def provider(_messages: Any) -> dict[str, Any]:
        return {
            "status": "revise",
            "reason": "调整扫描步数。",
            "parameter_patch": {"method_spec.basis": "def2SVP"},
        }

    result = ExpertModelAgent(
        ExpertModelConfig(enabled=True, provider=provider)
    ).review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot=_typed_catalog(),
        gaussian_route_policy=_structured_policy(),
    )

    assert result["status"] == "revise"
    assert result["overrides"] == {
        "typed-scan-1": {"method_spec.basis": "def2SVP"}
    }
    assert result["requested_transition"] is None
    assert workflow == original


def test_native_review_does_not_request_workflow_revision() -> None:
    """native 作业不在执行前审查阶段生成 workflow revision。"""

    workflow = _gaussian_codegen_converter_workflow()
    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            provider=lambda _messages: {
                "status": "reject",
                "reason": "route 缺少频率计算。",
            },
        )
    )

    result = agent.review_gaussian_steps(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=workflow,
        catalog_snapshot=_gaussian_catalog(),
        gaussian_code_results={
            "codegen-step-1": {
                "status": "completed",
                "result": {"gaussian_code": "#p M062X/def2SVP opt"},
            }
        },
    )

    assert result["status"] == "approved"
    assert result["requested_transition"] is None
    assert result["overrides"] == {}


def test_three_scientific_opinions_map_to_existing_routes() -> None:
    """最终裁判三分类沿用现有 evidence-ready/revise 路由。"""

    for opinion, route in (
        ("supports_hypothesis", "evidence_ready"),
        ("reliable_refutation", "revise_hypothesis"),
        ("insufficient_evidence", "revise_workflow"),
    ):
        def provider(messages: Any, value: str = opinion) -> dict[str, Any]:
            payload = json.loads(messages[1]["content"])
            if "evidence_index" in payload:
                refs = [item["evidence_ref"] for item in payload["evidence_index"]]
            else:
                refs = [
                    fact["evidence_ref"]
                    for step in payload["workflow_evidence"]["steps"]
                    for field in ("observations", "program_checks")
                    for fact in step[field]
                ]
                refs.extend(
                    step[field]["evidence_ref"]
                    for step in payload["workflow_evidence"]["steps"]
                    for field in ("failure", "not_executed")
                    if isinstance(step[field], dict)
                )
            return {
                "opinion": value,
                "summary": "意见",
                "evidence_refs": refs[:1],
            }
        agent = ExpertModelAgent(
            ExpertModelConfig(
                enabled=True,
                provider=provider,
            )
        )
        result = agent.evaluate_completed_workflow(
            scientific_question="问题",
            hypothesis=_hypothesis(),
            workflow=_workflow(),
            execution_results={"steps": []},
        )
        assert result["route"] == route


def test_program_gate_disallows_final_judge_route_without_silent_remap() -> None:
    """Gate 仅允许工作流修订时，裁判越权必须成为协议错误。"""

    def expert_provider(messages: Any) -> Any:
        payload = json.loads(messages[1]["content"])
        refs = [
            fact["evidence_ref"]
            for step in payload["workflow_evidence"]["steps"]
            for field in ("observations", "program_checks")
            for fact in step[field]
        ]
        refs.extend(
            step[field]["evidence_ref"]
            for step in payload["workflow_evidence"]["steps"]
            for field in ("failure", "not_executed")
            if isinstance(step[field], dict)
        )
        return {"opinion": "supports_hypothesis", "summary": "意见", "evidence_refs": refs[:1]}

    def general_provider(messages: Any) -> Any:
        payload = json.loads(messages[1]["content"])
        if "workflow_evidence" in payload:
            refs = [
                fact["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("observations", "program_checks")
                for fact in step[field]
            ]
            refs.extend(
                step[field]["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("failure", "not_executed")
                if isinstance(step[field], dict)
            )
        else:
            refs = [item["evidence_ref"] for item in payload["evidence_index"]]
        return {"opinion": "supports_hypothesis", "summary": "意见", "evidence_refs": refs[:1]}

    result = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            expert_provider=expert_provider,
            general_provider=general_provider,
        )
    ).evaluate_completed_workflow(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=_workflow(),
        execution_results={"steps": [{"step_id": "gaussian-step-1", "result": {"energy": -1.0}}]},
        allowed_decisions=["revise_workflow"],
        constraint_reasons=["required_reliability_not_matched"],
    )
    assert result["status"] == "protocol_error"
    assert result["route"] == "evidence_ready"


def test_saved_debate_routes_without_recalling_expert_or_general_models() -> None:
    """同一完成输入可由模型层生成稳定消息摘要，供 V2 收据复核。"""

    calls: list[Any] = []

    def provider(messages: Any) -> dict[str, Any]:
        calls.append(messages)
        payload = json.loads(messages[1]["content"])
        if "workflow_evidence" in payload:
            refs = [
                fact["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("observations", "program_checks")
                for fact in step[field]
            ]
            refs.extend(
                step[field]["evidence_ref"]
                for step in payload["workflow_evidence"]["steps"]
                for field in ("failure", "not_executed")
                if isinstance(step.get(field), dict)
            )
        else:
            refs = [item["evidence_ref"] for item in payload["evidence_index"]]
        return {
            "opinion": "supports_hypothesis",
            "summary": "证据支持。",
            "evidence_refs": refs[:1],
        }

    agent = ExpertModelAgent(
        ExpertModelConfig(
            enabled=True,
            expert_provider=provider,
            general_provider=provider,
        )
    )
    result = agent.evaluate_completed_workflow(
        scientific_question="问题",
        hypothesis=_hypothesis(),
        workflow=_workflow(),
        execution_results={"steps": []},
    )

    assert result["route"] == "evidence_ready"
    assert len(calls) == 3
    records = agent.last_request_message_records
    assert [record["operation"] for record in records] == [
        "workflow_expert_opinion",
        "workflow_general_opinion",
        "debate_judge",
    ]
    assert all(len(record["message_hash"]) == 64 for record in records)


def test_workflow_opinion_prompts_reference_embedded_evidence_facts() -> None:
    """执行后两份观点应从视图内证据事实取短引用。"""

    for prompt in (WORKFLOW_EXPERT_OPINION_PROMPT, WORKFLOW_GENERAL_OPINION_PROMPT):
        assert "supplied evidence facts" in prompt
        assert "evidence_index" not in prompt
    assert "Debate JSON.evidence_index" in DEBATE_JUDGE_PROMPT


def test_scientific_review_prompts_do_not_fix_hypothesis_cardinality() -> None:
    """科学审查 Prompt 面向所供推理与主张，不冻结假设数量。"""

    prompts = (
        GAUSSIAN_STEP_REVIEW_PROMPT,
        WORKFLOW_EXPERT_OPINION_PROMPT,
        WORKFLOW_GENERAL_OPINION_PROMPT,
        DEBATE_JUDGE_PROMPT,
    )
    joined = "\n".join(prompts).lower()
    for marker in (
        "one hypothesis",
        "single hypothesis",
        "sole hypothesis",
        "only hypothesis",
        "unique hypothesis",
        "selected hypothesis",
        "entire hypothesis",
        "the hypothesis",
        "hypothesis reasoning",
        "main hypothesis",
        "author main hypothesis",
    ):
        assert marker not in joined
    for prompt in prompts:
        assert "supplied scientific reasoning and claims" in prompt
