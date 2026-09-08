"""MiniChem Router 证据索引和最小决定协议的合同测试。"""

import math
from copy import deepcopy

import pytest

from chemistry_multiagent.utils.router_evidence import (
    _redact_failure_value,
    build_router_claim_schema,
    build_router_evidence_index,
    build_router_execution_projection,
    validate_router_claim_decision,
)


_MISSING = object()
_CATALOG_HASH = "b" * 64


def test_failure_redaction_preserves_urls_but_hides_local_absolute_paths():
    """失败事实保留来源 URL，同时隐藏 Unix 与 Windows 本机路径。"""

    projected = _redact_failure_value(
        {
            "repository": "https://github.com/Arvin0313/ARCHE",
            "error": r"failed at /tmp/arche/job.log and C:\\tmp\\job.log",
        }
    )

    assert projected["repository"] == "https://github.com/Arvin0313/ARCHE"
    assert "/tmp/arche/job.log" not in projected["error"]
    assert r"C:\\tmp\\job.log" not in projected["error"]


def _artifact(
    *,
    artifact_id: str = "art_mock_0001",
    sha256: str | None = "a" * 64,
    path: str = "artifacts/mock.xyz",
) -> dict:
    """构造一个完成 Action 必须发布的任务内 ArtifactRef。"""

    value = {
        "artifact_id": artifact_id,
        "semantic_type": "AtomicStructure",
        "media_type": "chemical/x-xyz",
        "sha256": sha256,
        "path": path,
        "producer_action": "mock_action",
        "producer_backend": "mock_backend",
        "parent_artifact_ids": [],
    }
    return value


def _router_input(
    statements,
    *,
    step_count=1,
    actual_values=None,
    allowed_decisions=None,
    constraint_reasons=None,
):
    """构造闭合的 MiniChem ActionResult、Program Gate 和 Router 输入。"""

    count = len(statements)
    actual_values = [True] * count if actual_values is None else list(actual_values)
    if len(actual_values) != count:
        raise AssertionError("测试夹具的实际值数量必须与主张一致")
    allowed_decisions = (
        ["evidence_ready", "revise_workflow", "revise_hypothesis"]
        if allowed_decisions is None
        else list(allowed_decisions)
    )
    constraint_reasons = (
        [] if constraint_reasons is None else list(constraint_reasons)
    )
    result = {
        "observed_values": {
            f"claim_{index}_observed": deepcopy(value)
            for index, value in enumerate(actual_values, start=1)
        }
    }
    workflow_steps = []
    execution_steps = []
    registered_checks = []
    for step_index in range(step_count):
        step_ref = f"s{step_index + 1}"
        artifact = _artifact(
            artifact_id=f"art_mock_{step_index + 1:04d}",
            path=f"artifacts/mock-{step_index + 1}.xyz",
        )
        workflow_steps.append(
            {
                "step_ref": step_ref,
                "step_name": "受控 Action",
                "description": "执行一次确定性的 MiniChem Action。",
                "action_id": "mock_action",
                "backend_id": "mock_backend",
                "request": {},
                "depends_on": [],
            }
        )
        execution_steps.append(
            {
                "step_ref": step_ref,
                "action_id": "mock_action",
                "backend_id": "mock_backend",
                "raw_output": {
                    "status": "completed",
                    "result": deepcopy(result),
                    "output_artifacts": [artifact],
                    "provenance": {"catalog_hash": _CATALOG_HASH},
                    "omitted_result_fields": [],
                },
            }
        )
        registered_checks.append(
            {
                "check_ref": f"c{step_index + 1}",
                "evaluation": "matched",
                "actual": {
                    "artifact_id": artifact["artifact_id"],
                    "semantic_type": artifact["semantic_type"],
                    "media_type": artifact["media_type"],
                    "sha256": artifact["sha256"],
                },
                "evidence_ref": (
                    f"/execution_result/steps/{step_index}/raw_output/output_artifacts/0"
                ),
                "diagnostic": None,
            }
        )
    workflow = {
        "workflow_name": "MiniChem Router 证据测试",
        "goal": "验证 ActionResult 只能通过当前 ArtifactRef 进入 Router。",
        "strategy_name": "受控策略",
        "steps": workflow_steps,
    }
    execution = {
        "total_steps": step_count,
        "steps": execution_steps,
        "overall_status": "completed",
        "workflow_outcome": "completed",
    }
    validation = {
        "registered_check_results": registered_checks,
        "program_gate_result": {
            "allowed_decisions": deepcopy(allowed_decisions),
            "constraint_reasons": deepcopy(constraint_reasons),
        },
    }
    catalog = [
        {
            "action_id": "mock_action",
            "action_version": "1.0.0",
            "category": "structure_and_system",
            "description": "执行受控结构转换。",
            "backend_ids": ["mock_backend"],
            "required_inputs": [],
            "optional_inputs": [],
            "selection_policy": "agent_backend_required",
            "primary_output": "AtomicStructure",
            "data_action": False,
        }
    ]
    router_input = {
        "scientific_context": {
            "scientific_question": "判断当前 MiniChem 证据是否支持主张。",
            "chemistry_context": {},
        },
        "combination": {
            "hypothesis": {
                "strategy_name": "受控策略",
                "reasoning": "\n\n".join(statements)
                + "\n\n只使用当前完成步骤发布的结构化证据。",
            }
        },
        "workflow": workflow,
        "direct_revision": None,
        "execution_result": execution,
        "program_validation": validation,
        "tool_catalog_projection": catalog,
        "allowed_decisions": deepcopy(allowed_decisions),
    }
    router_input["evidence_index"] = build_router_evidence_index(
        execution,
        validation,
        workflow,
        catalog,
    )
    return router_input


def _decision(router_input, *, decision, evidence_refs=None):
    """按证据索引顺序构造一份最小 Router 回答。"""

    selected = evidence_refs or [
        item["evidence_ref"] for item in router_input["evidence_index"]
    ]
    return {
        "decision": decision,
        "diagnosis": {
            "summary": "依据当前可靠 MiniChem 事实给出科学判断。",
            "evidence_refs": list(selected),
        },
    }


def _validate(answer, router_input):
    """使用输入中同一份冻结 MiniChem 目录校验 Router 回答。"""

    return validate_router_claim_decision(
        answer,
        router_input,
        router_input["tool_catalog_projection"],
    )


def test_router_assesses_complete_reasoning_without_legacy_claim_fields():
    """Router 直接审查完整 reasoning，不要求假设携带旧版主张字段。"""

    router_input = _router_input(["当前结构化结果足以支持唯一的科学判断。"])
    schema = build_router_claim_schema(router_input)
    assert set(schema["properties"]["diagnosis"]["properties"]) == {
        "summary",
        "evidence_refs",
    }
    evidence_ref = router_input["evidence_index"][0]["evidence_ref"]
    assert _validate(
        _decision(router_input, decision="evidence_ready", evidence_refs=[evidence_ref]),
        router_input,
    )["decision"] == "evidence_ready"


def test_router_marks_safe_nonempty_numeric_array_as_reliable():
    """完成步骤中的非空数值数组应整体作为可靠证据引用。"""

    router_input = _router_input(
        ["当前键级结果数组足以支持唯一的科学判断。"],
        actual_values=[{"bonds": [1.0, 2.0, 1.5]}],
    )

    bond_evidence = next(
        item
        for item in router_input["evidence_index"]
        if item["evidence_ref"].endswith("/claim_1_observed/bonds")
    )
    assert bond_evidence["reliable"] is True
    assert _validate(
        _decision(
            router_input,
            decision="evidence_ready",
            evidence_refs=[bond_evidence["evidence_ref"]],
        ),
        router_input,
    )["decision"] == "evidence_ready"


@pytest.mark.parametrize(
    "bond_values",
    [
        [],
        [None],
        [math.inf],
        [math.nan],
        [complex(1, 2)],
        [object()],
        [{"order": object()}],
        [[1.0, []]],
    ],
)
def test_router_marks_empty_or_unsafe_result_array_unreliable(bond_values):
    """空数组、空值、非有限浮点和非 JSON 成员不能成为可靠数组证据。"""

    router_input = _router_input(
        ["当前键级结果数组包含不可直接引用的值。"],
        actual_values=[{"bonds": bond_values}],
    )

    bond_evidence = next(
        item
        for item in router_input["evidence_index"]
        if item["evidence_ref"].endswith("/claim_1_observed/bonds")
    )
    assert bond_evidence["reliable"] is False


@pytest.mark.parametrize(
    "bond_values",
    [
        [[1.0, 2.0], [3.0]],
        [{"atom_pair": [1, 2], "order": 1.0}],
    ],
)
def test_router_recursively_accepts_safe_nested_result_array(bond_values):
    """嵌套数组或对象只有成员全部安全时才整体成为可靠证据。"""

    router_input = _router_input(
        ["当前键级结果数组成员均为可公开引用的 JSON 值。"],
        actual_values=[{"bonds": bond_values}],
    )

    bond_evidence = next(
        item
        for item in router_input["evidence_index"]
        if item["evidence_ref"].endswith("/claim_1_observed/bonds")
    )
    assert bond_evidence["reliable"] is True


def test_router_gate_reason_overrides_safe_array_reliability():
    """Program Gate 存在不可靠原因时，安全数组也不能升级为可靠证据。"""

    router_input = _router_input(
        ["当前键级结果数组仍受 Gate 约束。"],
        actual_values=[{"bonds": [1.0, 2.0]}],
        allowed_decisions=["revise_workflow"],
        constraint_reasons=["required_check_not_matched"],
    )

    bond_evidence = next(
        item
        for item in router_input["evidence_index"]
        if item["evidence_ref"].endswith("/claim_1_observed/bonds")
    )
    assert bond_evidence["reliable"] is False


def test_router_evidence_accepts_thirteen_steps_but_checks_total_steps() -> None:
    """Router 证据索引不限制步骤数量，但必须核对总数与实际数组一致。"""

    router_input = _router_input(
        ["十三步工作流仍有完整结构化证据。"],
        step_count=13,
    )

    assert len(router_input["workflow"]["steps"]) == 13
    assert router_input["execution_result"]["total_steps"] == 13
    assert router_input["evidence_index"]
    answer = _decision(router_input, decision="evidence_ready")
    assert _validate(answer, router_input)["decision"] == "evidence_ready"

    projection_input = deepcopy(router_input["execution_result"])
    step_refs = {}
    for index, step in enumerate(projection_input["steps"], start=1):
        step["step_id"] = f"formal-step-{index}"
        step.pop("step_ref")
        step_refs[f"s{index}"] = step["step_id"]
    projection = build_router_execution_projection(
        projection_input,
        step_refs,
        router_input["tool_catalog_projection"],
    )
    assert len(projection["steps"]) == 13

    inconsistent = deepcopy(router_input["execution_result"])
    inconsistent["total_steps"] = 12
    with pytest.raises(ValueError, match="前缀|数量"):
        build_router_evidence_index(
            inconsistent,
            router_input["program_validation"],
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_minichem_artifact_evidence_comes_from_declared_action_result():
    """Router 证据索引必须来自冻结 Action/Backend 的 ArtifactRef。"""

    router_input = _router_input(["得到结构产物。"])
    assert router_input["evidence_index"][0] == {
        "evidence_ref": "/program_validation/registered_check_results/0/actual",
        "evidence_kind": "artifact_ref",
        "reliable": True,
        "step_ref": "s1",
    }


@pytest.mark.parametrize("field_value", [_MISSING, None])
def test_minichem_missing_or_null_artifact_fields_stop_before_router(field_value):
    """ArtifactRef 关键字段缺失或为空时不能进入 Router。"""

    router_input = _router_input(["得到结构产物。"])
    artifact = router_input["execution_result"]["steps"][0]["raw_output"][
        "output_artifacts"
    ][0]
    if field_value is _MISSING:
        artifact.pop("sha256")
    else:
        artifact["sha256"] = field_value
    with pytest.raises(ValueError, match="ArtifactRef|sha256|非法"):
        build_router_evidence_index(
            router_input["execution_result"],
            router_input["program_validation"],
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_minichem_evidence_preserves_false_and_zero_result_values():
    """明确反值和数值零都是有效结构化证据，不能按空值删除。"""

    router_input = _router_input(
        ["标志为假。", "数量为零。"],
        actual_values=[False, 0],
    )
    refs = {
        item["evidence_ref"]: item["reliable"]
        for item in router_input["evidence_index"]
    }
    assert refs["/execution_result/steps/0/raw_output/result/observed_values/claim_1_observed"] is True
    assert refs["/execution_result/steps/0/raw_output/result/observed_values/claim_2_observed"] is True


def test_minichem_evidence_rejects_program_validation_tampering():
    """Program Gate 的实际值被改写时必须回到原始 ArtifactRef 复验。"""

    router_input = _router_input(["得到结构产物。"])
    validation = deepcopy(router_input["program_validation"])
    validation["registered_check_results"][0]["actual"]["artifact_id"] = "forged"
    with pytest.raises(ValueError, match="派生|一致|证据"):
        build_router_evidence_index(
            router_input["execution_result"],
            validation,
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_minichem_evidence_rejects_undeclared_backend():
    """工作流不能绑定 Action projection 未声明的 Backend。"""

    router_input = _router_input(["得到结构产物。"])
    catalog = deepcopy(router_input["tool_catalog_projection"])
    catalog[0]["backend_ids"] = ["other_backend"]
    with pytest.raises(ValueError, match="Action|Backend|声明"):
        build_router_evidence_index(
            router_input["execution_result"],
            router_input["program_validation"],
            router_input["workflow"],
            catalog,
        )


def test_minichem_evidence_index_rejects_broken_check_reference_chain():
    """Program Gate 不能伪造本次请求之外的连续检查引用。"""

    router_input = _router_input(["得到结构产物。"])
    validation = deepcopy(router_input["program_validation"])
    validation["registered_check_results"][0]["check_ref"] = "c999"
    with pytest.raises(ValueError, match="检查|一致"):
        build_router_evidence_index(
            router_input["execution_result"],
            validation,
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_router_rejects_evidence_reference_outside_current_index():
    """模型不能自行编造越界 JSON Pointer。"""

    router_input = _router_input(["生成合法结构产物。"])
    answer = _decision(router_input, decision="evidence_ready")
    answer["diagnosis"]["evidence_refs"] = [
        "/program_validation/registered_check_results/99/actual"
    ]
    with pytest.raises(ValueError, match="index|evidence_ref|登记"):
        _validate(answer, router_input)


@pytest.mark.parametrize("decision", ["evidence_ready", "revise_workflow", "revise_hypothesis"])
def test_open_gate_leaves_scientific_interpretation_to_router(decision):
    """开放 Gate 只限制候选决定，不用关键词替 Router 解释科学含义。"""

    router_input = _router_input(["生成结构产物。"], allowed_decisions=[
        "evidence_ready",
        "revise_workflow",
        "revise_hypothesis",
    ])
    assert _validate(_decision(router_input, decision=decision), router_input)["decision"] == decision


def test_restricted_gate_forces_workflow_revision_even_with_reliable_evidence():
    """受限 Program Gate 下不得由 Router 选择其他决定。"""

    router_input = _router_input(
        ["需要重做工作流。"],
        allowed_decisions=["revise_workflow"],
        constraint_reasons=["required_check_not_matched"],
    )
    assert _validate(_decision(router_input, decision="revise_workflow"), router_input)[
        "decision"
    ] == "revise_workflow"
    with pytest.raises(ValueError, match="program gate"):
        _validate(_decision(router_input, decision="evidence_ready"), router_input)


def test_router_evidence_refs_must_keep_the_index_order():
    """Router 引用必须保持 evidence index 顺序，避免多种事实表示。"""

    router_input = _router_input(["生成结构产物。"])
    answer = _decision(router_input, decision="evidence_ready")
    answer["diagnosis"]["evidence_refs"].reverse()
    # 单条引用反转仍保持顺序；补充第二条结果证据后验证真正的逆序。
    router_input = _router_input(["标志为假。", "数量为零。"], actual_values=[False, 0])
    answer = _decision(router_input, decision="evidence_ready")
    answer["diagnosis"]["evidence_refs"].reverse()
    with pytest.raises(ValueError, match="顺序|evidence_refs"):
        _validate(answer, router_input)


@pytest.mark.parametrize(
    "statement",
    [
        "该结构保持平面。",
        "该构象是 global minimum。",
        "该峰具有 stretching mode 归属。",
        "该路径的 activation energy 低于阈值。",
        "计算相对 experiment 的 error 小于阈值。",
    ],
)
def test_program_does_not_prejudge_scientific_meaning_from_keywords(statement):
    """程序只验证事实来源，不用自然语言关键词替 Router 判断。"""

    assert _validate(_decision(_router_input([statement]), decision="evidence_ready"), _router_input([statement]))[
        "decision"
    ] == "evidence_ready"


def test_router_schema_has_only_minimal_fields_and_nonempty_evidence():
    """动态 Schema 不重复主张、步骤范围、问题类型或置信度。"""

    router_input = _router_input(["生成结构产物。", "生成下游输入。"])
    schema = build_router_claim_schema(router_input)
    diagnosis = schema["properties"]["diagnosis"]
    assert set(schema["properties"]) == {"decision", "diagnosis"}
    assert set(diagnosis["properties"]) == {"summary", "evidence_refs"}
    assert diagnosis["properties"]["evidence_refs"]["minItems"] == 1


def test_non_matched_registered_check_is_not_reliable_and_blocks_evidence_ready():
    """注册检查未匹配时保留负向事实，但不能被 Router 当作成功证据。"""

    router_input = _router_input(["需要可靠结构产物。"])
    validation = router_input["program_validation"]
    validation["registered_check_results"][0]["evaluation"] = "not_matched"
    validation["registered_check_results"][0]["diagnostic"] = {
        "reason": "required artifact missing",
    }
    router_input["evidence_index"] = build_router_evidence_index(
        router_input["execution_result"],
        validation,
        router_input["workflow"],
        router_input["tool_catalog_projection"],
    )
    assert router_input["evidence_index"][0]["reliable"] is False
    with pytest.raises(ValueError, match="Program Gate|不可靠"):
        _validate(_decision(router_input, decision="evidence_ready"), router_input)


def test_invalid_status_axis_blocks_evidence_ready_and_marks_artifact_unreliable():
    """机械或科学状态轴失败时，ArtifactRef 不能继续伪装为可靠证据。"""

    router_input = _router_input(["需要通过状态检查的产物。"])
    validation = router_input["program_validation"]
    validation["registered_check_results"][0]["evaluation"] = "not_matched"
    validation["program_gate_result"]["status_axes"] = {
        "mechanical_status": "invalid",
    }
    validation["program_gate_result"]["allowed_decisions"] = ["revise_workflow"]
    validation["program_gate_result"]["constraint_reasons"] = [
        "mechanical_status_invalid"
    ]
    router_input["allowed_decisions"] = ["revise_workflow"]
    router_input["evidence_index"] = build_router_evidence_index(
        router_input["execution_result"],
        validation,
        router_input["workflow"],
        router_input["tool_catalog_projection"],
    )
    assert router_input["evidence_index"][0]["reliable"] is False
    with pytest.raises(ValueError, match="Program Gate|status_axes|越过"):
        _validate(_decision(router_input, decision="evidence_ready"), router_input)


def test_declaration_validation_failure_blocks_evidence_ready():
    """声明产物的 validation_status/errors 失败时必须保留可审计拒绝原因。"""

    router_input = _router_input(["需要通过声明校验的产物。"])
    validation = router_input["program_validation"]
    validation["registered_check_results"][0]["evaluation"] = "not_matched"
    validation["program_gate_result"]["declaration_checks"] = [
        {
            "validation_status": "invalid",
            "errors": ["media type mismatch"],
        }
    ]
    validation["program_gate_result"]["allowed_decisions"] = ["revise_workflow"]
    validation["program_gate_result"]["constraint_reasons"] = [
        "declaration_validation_failed"
    ]
    router_input["allowed_decisions"] = ["revise_workflow"]
    router_input["evidence_index"] = build_router_evidence_index(
        router_input["execution_result"],
        validation,
        router_input["workflow"],
        router_input["tool_catalog_projection"],
    )
    assert router_input["evidence_index"][0]["reliable"] is False
    with pytest.raises(ValueError, match="Program Gate|declaration_checks|越过"):
        _validate(_decision(router_input, decision="evidence_ready"), router_input)


def _failed_terminal_router_input() -> dict:
    """构造 completed -> failed -> not_executed 的完整 Router 终局。"""

    router_input = _router_input(["首步产物可用。"], step_count=3)
    execution = router_input["execution_result"]
    execution["overall_status"] = "failed"
    execution["workflow_outcome"] = "stopped_after_error"
    failed_step = execution["steps"][1]
    failed_step["raw_output"] = {
        "status": "failed",
        "result": None,
        "output_artifacts": [],
        "provenance": {"catalog_hash": _CATALOG_HASH},
        "omitted_result_fields": [],
        "error": {
            "code": "backend_failure",
            "message": "无法读取输入文件",
        },
    }
    skipped_step = execution["steps"][2]
    skipped_step["raw_output"] = {
        "status": "not_executed",
        "result": None,
        "output_artifacts": [],
        "provenance": {"catalog_hash": _CATALOG_HASH},
        "omitted_result_fields": [],
        "reason": "依赖失败，未执行下游步骤",
    }
    validation = router_input["program_validation"]
    validation["registered_check_results"] = validation["registered_check_results"][:1]
    validation["program_gate_result"] = {
        "allowed_decisions": ["revise_workflow"],
        "constraint_reasons": [
            "formal-step-2:execution_failed",
            "formal-step-3:not_executed",
        ],
    }
    router_input["allowed_decisions"] = ["revise_workflow"]
    router_input["evidence_index"] = build_router_evidence_index(
        execution,
        validation,
        router_input["workflow"],
        router_input["tool_catalog_projection"],
    )
    return router_input


def test_router_projection_preserves_full_failed_terminal_facts() -> None:
    """完整失败终局保留状态、错误和未执行原因，不伪装成 completed。"""

    router_input = _router_input(["首步产物可用。"], step_count=3)
    canonical = deepcopy(router_input["execution_result"])
    step_refs = {}
    for index, step in enumerate(canonical["steps"], start=1):
        step["step_id"] = f"formal-step-{index}"
        step.pop("step_ref")
        step_refs[f"s{index}"] = step["step_id"]
    canonical["overall_status"] = "failed"
    canonical["workflow_outcome"] = "stopped_after_error"
    canonical["steps"][1] = {
        **canonical["steps"][1],
        "status": "failed",
        "raw_output": {
            "status": "technical_pause",
            "result": None,
            "output_artifacts": [],
            "provenance": {"catalog_hash": _CATALOG_HASH},
            "omitted_result_fields": [],
            "error": {
                "code": "backend_failure",
                "message": "无法读取 /tmp/private/catalyst.xyz",
            },
        },
        "error": {
            "code": "backend_failure",
            "message": "无法读取 /tmp/private/catalyst.xyz",
        },
    }
    canonical["steps"][2] = {
        **canonical["steps"][2],
        "status": "not_executed",
        "raw_output": {
            "status": "not_executed",
            "result": None,
            "output_artifacts": [],
            "provenance": {"catalog_hash": _CATALOG_HASH},
            "omitted_result_fields": [],
            "reason": "依赖失败，未执行 /tmp/private/downstream.xyz",
        },
        "reason": "依赖失败，未执行 /tmp/private/downstream.xyz",
    }
    projected = build_router_execution_projection(
        canonical,
        step_refs,
        router_input["tool_catalog_projection"],
    )
    assert [step["raw_output"]["status"] for step in projected["steps"]] == [
        "completed",
        "failed",
        "not_executed",
    ]
    assert "[路径已隐藏]" in projected["steps"][1]["raw_output"]["error"]["message"]
    assert "/tmp/private" not in str(projected)
    assert projected["steps"][2]["raw_output"]["reason"].startswith("依赖失败")


def _canonical_failed_execution_with_error(error: object) -> tuple[dict, dict, list]:
    """构造带指定 error 事实的单步 canonical 失败终局。"""

    router_input = _router_input(["失败事实必须可审计。"])
    canonical = deepcopy(router_input["execution_result"])
    step = canonical["steps"][0]
    step["step_id"] = "formal-step-1"
    step.pop("step_ref")
    step["status"] = "failed"
    step["raw_output"] = {
        "status": "technical_pause",
        "result": None,
        "output_artifacts": [],
        "provenance": {"catalog_hash": _CATALOG_HASH},
        "omitted_result_fields": [],
        "error": deepcopy(error),
    }
    step["error"] = deepcopy(error)
    canonical["overall_status"] = "failed"
    canonical["workflow_outcome"] = "stopped_after_error"
    return canonical, {"s1": "formal-step-1"}, router_input["tool_catalog_projection"]


@pytest.mark.parametrize(
    "error",
    [{"traceback": "stack omitted"}, {"details": {"traceback": "stack omitted"}}, []],
)
def test_router_projection_keeps_marker_when_failure_fact_is_fully_filtered(error) -> None:
    """错误事实被边界过滤为空时仍保留明确的 bounded 省略标记。"""

    canonical, step_refs, catalog = _canonical_failed_execution_with_error(error)
    projected = build_router_execution_projection(canonical, step_refs, catalog)

    assert projected["steps"][0]["raw_output"]["error"] == {
        "omitted": "failure_fact_bounded"
    }


def test_router_projection_rejects_scalar_failure_fact() -> None:
    """canonical error 不得使用无法解释的标量。"""

    canonical, step_refs, catalog = _canonical_failed_execution_with_error(123)
    with pytest.raises(ValueError, match="字符串|object|array"):
        build_router_execution_projection(canonical, step_refs, catalog)


def _open_failed_router_input() -> dict:
    """构造失败的 open execution，并冻结空 collect 投影。"""

    router_input = _router_input(["open job 失败事实必须可见。"])
    router_input["workflow"]["steps"] = [
        {
            "step_ref": "s1",
            "execution_layer": "native_software",
            "request": {"software_id": "gaussian16"},
        }
    ]
    router_input["execution_result"] = {
        "total_steps": 1,
        "overall_status": "failed",
        "workflow_outcome": "stopped_after_error",
        "steps": [
            {
                "step_ref": "s1",
                "execution_layer": "native_software",
                "software_id": "gaussian16",
                "raw_output": {
                    "status": "failed",
                    "result": None,
                    "output_artifacts": [],
                    "provenance": {"catalog_hash": _CATALOG_HASH},
                    "omitted_result_fields": [],
                    "operation_id": None,
                    "job_id": None,
                    "job_status": None,
                    "job": None,
                    "ready": False,
                    "outputs": [],
                    "files": [],
                    "declared_artifacts": [],
                    "artifact_manifest": None,
                    "error": {"code": "backend_failure"},
                },
            }
        ],
    }
    router_input["program_validation"] = {
        "registered_check_results": [],
        "program_gate_result": {
            "allowed_decisions": ["revise_workflow"],
            "constraint_reasons": ["s1:execution_failed"],
        },
    }
    router_input["allowed_decisions"] = ["revise_workflow"]
    router_input["evidence_index"] = build_router_evidence_index(
        router_input["execution_result"],
        router_input["program_validation"],
        router_input["workflow"],
        router_input["tool_catalog_projection"],
    )
    return router_input


def test_router_failed_open_execution_requires_empty_collect_and_ready_false() -> None:
    """失败 open execution 必须固定 ready=false 且不泄漏 job/collect 身份。"""

    router_input = _open_failed_router_input()
    raw_output = router_input["execution_result"]["steps"][0]["raw_output"]
    assert raw_output["ready"] is False
    assert raw_output["operation_id"] is None
    assert raw_output["job_id"] is None
    assert raw_output["outputs"] == []
    assert raw_output["files"] == []

    tampered = deepcopy(router_input["execution_result"])
    tampered["steps"][0]["raw_output"]["ready"] = True
    with pytest.raises(ValueError, match="collect|ready|为空"):
        build_router_evidence_index(
            tampered,
            router_input["program_validation"],
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_router_failed_terminal_exposes_unreliable_refs_and_only_revision() -> None:
    """失败/未执行引用可审计但不可靠，受限 Gate 只能路由 revise_workflow。"""

    router_input = _failed_terminal_router_input()
    refs = router_input["evidence_index"]
    assert any(item["evidence_kind"] == "execution_error" for item in refs)
    assert any(item["evidence_kind"] == "not_executed_reason" for item in refs)
    assert all(
        item["reliable"] is False
        for item in refs
        if item["evidence_kind"] in {"execution_error", "not_executed_reason"}
    )
    assert _validate(
        _decision(router_input, decision="revise_workflow"), router_input
    )["decision"] == "revise_workflow"
    with pytest.raises(ValueError, match="受限|program gate"):
        _validate(_decision(router_input, decision="evidence_ready"), router_input)


def test_router_failed_terminal_requires_restricted_program_gate() -> None:
    """完整失败终局不能被篡改为开放 Gate。"""

    router_input = _failed_terminal_router_input()
    validation = deepcopy(router_input["program_validation"])
    validation["program_gate_result"] = {
        "allowed_decisions": [
            "evidence_ready",
            "revise_workflow",
            "revise_hypothesis",
        ],
        "constraint_reasons": [],
    }
    with pytest.raises(ValueError, match="失败终局|revise_workflow|Program Gate"):
        build_router_evidence_index(
            router_input["execution_result"],
            validation,
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("workflow_outcome", "completed"), ("overall_status", "completed")],
)
def test_router_rejects_failed_terminal_status_or_outcome_tampering(field, value) -> None:
    """失败终局的外层状态或 outcome 被改写时必须拒绝。"""

    router_input = _failed_terminal_router_input()
    tampered = deepcopy(router_input["execution_result"])
    tampered[field] = value
    with pytest.raises(ValueError, match="终局|workflow_outcome|状态"):
        build_router_evidence_index(
            tampered,
            router_input["program_validation"],
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_router_rejects_failed_terminal_order_tampering() -> None:
    """失败点之后重新出现 failed 或 completed 时必须拒绝。"""

    router_input = _failed_terminal_router_input()
    tampered = deepcopy(router_input["execution_result"])
    tampered["steps"][1]["raw_output"]["status"] = "not_executed"
    tampered["steps"][1]["raw_output"].pop("error")
    tampered["steps"][1]["raw_output"]["reason"] = "前一步失败"
    tampered["steps"][2]["raw_output"]["status"] = "failed"
    tampered["steps"][2]["raw_output"].pop("reason")
    tampered["steps"][2]["raw_output"]["error"] = {"code": "forged"}
    with pytest.raises(ValueError, match="终局|顺序|canonical"):
        build_router_evidence_index(
            tampered,
            router_input["program_validation"],
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )


def test_router_rejects_waiting_execution_projection() -> None:
    """尚未完成的异步作业不能伪装成执行后 Router 终局。"""

    router_input = _router_input(["异步作业尚未结束。"])
    waiting = deepcopy(router_input["execution_result"])
    waiting["overall_status"] = "waiting_execution"
    waiting["workflow_outcome"] = "waiting_execution"
    waiting["steps"][0]["raw_output"]["status"] = "waiting"
    with pytest.raises(ValueError, match="waiting_execution|非法|终局"):
        build_router_evidence_index(
            waiting,
            router_input["program_validation"],
            router_input["workflow"],
            router_input["tool_catalog_projection"],
        )
