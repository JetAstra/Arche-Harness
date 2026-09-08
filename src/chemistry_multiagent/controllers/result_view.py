"""把已提交事实整理为统一的 bounded JSON。

本模块只做确定性投影，不访问数据库、不读取目录，也不决定科学状态。调用方必须
先确认当前组合已经由 Router 标记为 evidence_ready，并提供同一组合的完整历史。
"""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from chemistry_multiagent.memory.contracts import (
    BATCH_CLOSABLE_COMBINATION_STATES,
    COMBINATION_TERMINAL_REASONS,
    validate_structured_hypothesis_claims,
    validate_tool_capability_gap,
)
from chemistry_multiagent.utils.router_evidence import (
    _FAILURE_ABSOLUTE_PATH_TEXT,
    _payload_field_name,
    _project_native_foreach_result,
    validate_program_gate_for_evidence_ready,
)


_PACKAGE_KEYS = {
    "scientific_question",
    "workflow_start_time",
    "task_id",
    "task_config",
    "chemistry_context",
    "retrieval_phase",
    "hypothesis_phase",
    "hypothesis",
    "planning_record",
    "attempt_records",
    "revision_records",
    "pause_restart_records",
    "terminal_record_id",
    "batch_combination_outcomes",
}
_DECISION_VIEW = {
    "evidence_ready": "accept",
    "revise_workflow": "revise_workflow",
    "revise_hypothesis": "revise_hypothesis",
}


def _timestamp(value: Any, field: str) -> float:
    """把规范 UTC 时间转换为旧结果使用的 Unix 秒。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是非空时间字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} 不是 ISO 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} 必须包含时区")
    result = parsed.timestamp()
    if not math.isfinite(result):
        raise ValueError(f"{field} 超出可表示范围")
    return result


def _required_mapping(value: Any, field: str) -> dict[str, Any]:
    """复制一份必需的 JSON object，避免渲染过程修改来源事实。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是 object")
    return deepcopy(dict(value))


def _batch_combination_outcomes(value: Any) -> list[dict[str, Any]]:
    """校验并复制整批组合终局，能力缺口只保留其原始规划判断。"""

    if not isinstance(value, list) or not value:
        raise ValueError("batch_combination_outcomes 必须是非空数组")
    outcomes = []
    for offset, raw in enumerate(value, start=1):
        item = _required_mapping(raw, f"batch_combination_outcomes[{offset - 1}]")
        base_keys = {
            "combination_index",
            "terminal_state",
            "terminal_reason",
            "tool_capability_gap",
        }
        if set(item) not in {frozenset(base_keys), frozenset({
            *base_keys,
            "technical_failure",
        })}:
            raise ValueError("batch combination outcome 字段不完整")
        combination_index = item["combination_index"]
        if (
            isinstance(combination_index, bool)
            or not isinstance(combination_index, int)
            or combination_index != offset
        ):
            raise ValueError("batch combination outcome 必须按连续组合序号排列")
        state = item["terminal_state"]
        reason = item["terminal_reason"]
        if state == "technical_pause":
            technical_failure = item.get("technical_failure")
            if (
                reason != "technical_pause"
                or item["tool_capability_gap"] is not None
                or not isinstance(technical_failure, Mapping)
                or technical_failure.get("status") != "technical_pause"
            ):
                raise ValueError("技术暂停终局缺少匹配的真实技术失败正文")
            item["technical_failure"] = deepcopy(dict(technical_failure))
            outcomes.append(item)
            continue
        if "technical_failure" in item:
            raise ValueError("非技术暂停组合不能携带 technical_failure")
        if state not in BATCH_CLOSABLE_COMBINATION_STATES:
            raise ValueError("batch combination outcome 不是可归约终局")
        if reason not in COMBINATION_TERMINAL_REASONS:
            raise ValueError("batch combination outcome 终态原因非法")
        gap = item["tool_capability_gap"]
        if state == "tool_capability_gap":
            if reason != "tool_capability_gap" or not isinstance(gap, Mapping):
                raise ValueError("能力缺口终局缺少匹配的真实 Gap")
            gap = validate_tool_capability_gap(
                deepcopy(dict(gap)),
                path=(
                    f"batch_combination_outcomes[{offset - 1}].tool_capability_gap"
                ),
            )
        elif gap is not None:
            raise ValueError("非 Gap 组合不能携带 Tool Capability Gap")
        if state == "evidence_ready" and reason != "router_evidence_ready":
            raise ValueError("成功组合与终态原因不一致")
        if state == "hypothesis_failed" and reason in {
            "router_evidence_ready",
            "tool_capability_gap",
        }:
            raise ValueError("假设失败组合与终态原因不一致")
        item["tool_capability_gap"] = gap
        outcomes.append(item)
    if not any(item["terminal_state"] == "evidence_ready" for item in outcomes):
        raise ValueError("成功 bounded result 的批次缺少 Evidence Ready")
    return outcomes


_BOUNDED_HYPOTHESIS_RANK_FIELDS = (
    "merged_from",
    "wins",
    "losses",
    "score",
    "rank",
)


def _bounded_hypothesis_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """把新旧假设统一投影为完整 reasoning 和真实排名事实。"""

    copied = dict(value)
    if "claims" in copied:
        validate_structured_hypothesis_claims(
            copied,
            path="bounded_result.hypothesis",
        )
    hypothesis_id = copied.get("hypothesis_id")
    strategy_name = copied.get("strategy_name")
    reasoning = copied.get("reasoning", copied.get("detailed_reasoning"))
    # 极早期快照只有一段 reasoning。读取兼容时沿用当时可见正文，不再把画像、
    # 置信度或元数据带回新结果；数据库中的历史原文仍保持不变。
    required_text = {
        "hypothesis_id": hypothesis_id,
        "strategy_name": strategy_name,
        "reasoning": reasoning,
    }
    if any(
        not isinstance(item, str) or not item.strip()
        for item in required_text.values()
    ):
        raise ValueError("bounded result 的假设科学正文不完整")
    projected = {
        key: item.strip()
        for key, item in required_text.items()
    }
    for field in _BOUNDED_HYPOTHESIS_RANK_FIELDS:
        if field in copied:
            projected[field] = deepcopy(copied[field])
    return projected


def _bounded_hypothesis_phase(value: Mapping[str, Any]) -> dict[str, Any]:
    """只发布当前三段假设结构，兼容旧分组形状但不发布旧画像。"""

    queries = value.get("queries", [])
    if not isinstance(queries, list) or any(
        not isinstance(item, str) or not item.strip() for item in queries
    ):
        raise ValueError("hypothesis_phase.queries 必须是字符串数组")
    normalized_queries = [item.strip() for item in queries]
    raw_groups = value.get("hypotheses_by_query", [])
    if not isinstance(raw_groups, list):
        raise ValueError("hypothesis_phase.hypotheses_by_query 必须是数组")
    groups = []
    for index, raw_group in enumerate(raw_groups):
        if isinstance(raw_group, list):
            if index >= len(normalized_queries):
                raise ValueError("旧假设分组无法对应 query")
            query = normalized_queries[index]
            hypotheses = raw_group
        elif isinstance(raw_group, Mapping):
            query = raw_group.get("query")
            hypotheses = raw_group.get("hypotheses")
        else:
            raise ValueError("hypothesis_phase 假设分组非法")
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(hypotheses, list)
        ):
            raise ValueError("hypothesis_phase 假设分组内容不完整")
        projected_hypotheses = []
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, Mapping):
                raise ValueError("hypothesis_phase 分组候选必须是 object")
            projected = _bounded_hypothesis_projection(hypothesis)
            projected_hypotheses.append(
                {
                    field: projected[field]
                    for field in (
                        "hypothesis_id",
                        "strategy_name",
                        "reasoning",
                    )
                }
            )
        groups.append({"query": query.strip(), "hypotheses": projected_hypotheses})
    ranked = value.get("ranked_strategies")
    if not isinstance(ranked, list):
        raise ValueError("hypothesis_phase 缺少 ranked_strategies")
    if any(not isinstance(item, Mapping) for item in ranked):
        raise ValueError("hypothesis_phase 排名候选必须是 object")
    return {
        "queries": normalized_queries,
        "hypotheses_by_query": groups,
        "ranked_strategies": [
            _bounded_hypothesis_projection(item)
            for item in ranked
        ],
    }


def strip_internal_claims_for_bounded_result(value: Any) -> Any:
    """只在最终结果约定的假设位置执行投影，不按普通字段名猜测语义。"""

    if not isinstance(value, Mapping):
        return deepcopy(value)
    copied = deepcopy(dict(value))
    hypothesis_phase = copied.get("hypothesis_phase")
    if isinstance(hypothesis_phase, Mapping):
        copied["hypothesis_phase"] = _bounded_hypothesis_phase(hypothesis_phase)
    final_conclusion = copied.get("final_conclusion")
    if isinstance(final_conclusion, Mapping):
        final_conclusion = dict(final_conclusion)
        selected = final_conclusion.get("selected_strategy")
        if isinstance(selected, Mapping):
            final_conclusion["selected_strategy"] = _bounded_hypothesis_projection(
                selected
            )
        copied["final_conclusion"] = final_conclusion
    return copied


def _bounded_router_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    """隐藏逐主张内部审计，同时保留旧结果一直展示的 Router 字段。"""

    result = _required_mapping(value, "Router decision")
    diagnosis = _required_mapping(result.get("diagnosis"), "Router diagnosis")
    diagnosis.pop("claim_assessments", None)
    result["diagnosis"] = diagnosis
    return result


def _execution_steps(execution_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """兼容单工作流根结构与旧多工作流外壳，按原顺序取得步骤。"""

    direct = execution_result.get("steps")
    if isinstance(direct, list):
        if any(not isinstance(item, Mapping) for item in direct):
            raise ValueError("execution_result.steps 含有损坏的步骤记录")
        return [deepcopy(dict(item)) for item in direct]
    collected: list[dict[str, Any]] = []
    results = execution_result.get("results")
    if isinstance(results, list):
        for workflow in results:
            if not isinstance(workflow, Mapping):
                raise ValueError("execution_result.results 含有损坏的工作流记录")
            steps = workflow.get("steps")
            if not isinstance(steps, list) or any(
                not isinstance(item, Mapping) for item in steps
            ):
                raise ValueError("execution_result.results 中的步骤记录已损坏")
            collected.extend(deepcopy(dict(item)) for item in steps)
    return collected


_FINAL_FOREACH_PROVENANCE_KEYS = frozenset(
    {
        "pair_id",
        "candidate_id",
        "direction",
        "role",
        "label",
        "pair_index",
        "candidate_index",
    }
)
_FINAL_FOREACH_SCIENTIFIC_KEYS = frozenset(
    {
        "energy",
        "molecules_written",
        "charge",
        "multiplicity",
        "route",
        "gaussian_route",
        "gaussian_input_deck",
        "input_deck",
        "deck_preflight_status",
        "electronic_energy",
        "electronic_energies",
        "zero_point_energy",
        "enthalpy",
        "entropy",
        "gibbs_free_energy",
        "relative_gibbs_free_energies",
        "activation_free_energies",
        "delta_delta_g",
        "enantiomeric_ratio",
        "enantiomeric_excess",
        "rate_determining_step",
        "optimized_geometry",
        "hessian_summary",
        "vibrational_mode_summary",
        "frequencies_cm1",
        "vibrational_frequencies",
        "imaginary_frequencies",
        "imaginary_frequency_count",
        "vibrational_intensities",
        "ir_intensities",
        "raman_intensities",
        "observed_support",
        "normal_termination",
        "optimization_convergence",
        "converged",
        "final_atomic_structure",
        "geometry",
        "coordinates",
        "unit",
        "conditions",
    }
)


def _final_safe_value(value: Any, *, scientific: bool = False) -> Any:
    """递归复制最终结果允许发布的有限值并隐藏内部执行事实。

    输入是任意结果值和是否只保留科学字段的开关；动作递归复制有限标量、允许的
    科学字段及集合，同时删除 operation/job 身份、请求引用、工作区路径和原始日志。
    函数只在内存中构造新值，不修改输入或写文件；返回清洗后的值，非有限数、空值、
    超长文本、受保护字段和不支持的类型返回 ``None``。不因普通非法输入抛出异常，
    上层会据此省略无法安全发布的字段。
    """

    if value is None or isinstance(value, bool):
        return deepcopy(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return deepcopy(value) if math.isfinite(value) else None
    if isinstance(value, str):
        lowered = value.casefold()
        sanitized = _FAILURE_ABSOLUTE_PATH_TEXT.sub("[路径已隐藏]", value)
        if (
            not sanitized.strip()
            or len(sanitized) > 8192
            or any(
                marker in lowered
                for marker in (
                    "operation_id",
                    "job_id",
                    "request_ref",
                    "raw_log",
                    "/mnt/",
                    "\\\\",
                )
            )
        ):
            return None
        return sanitized.strip()
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized_key = _payload_field_name(key)
            if normalized_key in {
                "operation",
                "operation_id",
                "operation_ids",
                "operation_fingerprint",
                "parent_operation_id",
                "child_operation_id",
                "child_result_ref",
                "request",
                "request_ref",
                "request_fingerprint",
                "job",
                "job_id",
                "job_ids",
                "job_status",
                "parent_job_id",
                "foreach",
                "manifest",
                "artifact_manifest",
                "path",
                "workspace_path",
                "job_relative_path",
                "relative_path",
                "target_path",
                "artifact_path",
                "file_path",
                "source_step_ref",
                "stdout",
                "stderr",
                "raw_log",
            } or normalized_key.endswith(("_path", "_paths", "_filepath", "_filepaths")):
                continue
            if scientific and key not in _FINAL_FOREACH_SCIENTIFIC_KEYS:
                continue
            child = _final_safe_value(item, scientific=False)
            if child is not None:
                projected[key] = child
        return projected
    if isinstance(value, (list, tuple)):
        return [
            child
            for item in value
            if (child := _final_safe_value(item, scientific=scientific)) is not None
        ]
    return None


def _project_final_foreach_provenance(value: Any) -> dict[str, Any]:
    """把 child 来源说明投影为可发布的有限标签。

    输入是任意 provenance（来源说明，标识候选与上游步骤的关系）；动作只读取允许
    的 pair、candidate、方向和索引字段，并通过安全值投影器清理文本。函数无副作用，
    返回来源标签字典；输入不是映射或字段不合格时返回空字典或跳过该字段，不抛出
    业务异常。
    """

    source = value if isinstance(value, Mapping) else {}
    result: dict[str, Any] = {}
    for key in sorted(_FINAL_FOREACH_PROVENANCE_KEYS):
        projected = _final_safe_value(source.get(key))
        if projected is not None:
            result[key] = projected
    return result


def _project_final_artifact_summary(value: Any) -> dict[str, Any]:
    """把 ArtifactRef 产物引用投影为不含工作区路径的核对摘要。

    输入是任意产物记录；动作只保留 artifact ID、语义类型、媒体类型、哈希、父产物
    身份和生产者字段，并通过安全值投影器过滤路径及运行身份。函数只在内存中构造
    新字典，不修改输入或写文件；返回可供结果视图核对的摘要，输入不是映射或字段
    不合格时返回空字典，不抛出业务异常。
    """

    source = value if isinstance(value, Mapping) else {}
    result: dict[str, Any] = {}
    for key in (
        "artifact_id",
        "semantic_type",
        "media_type",
        "sha256",
        "parent_artifact_ids",
        "producer_action",
        "producer_backend",
    ):
        projected = _final_safe_value(source.get(key))
        if projected is not None:
            result[key] = projected
    return result


def _project_final_file_summary(value: Any) -> dict[str, Any]:
    """把 child collect 文件记录投影为不发布路径的摘要。

    输入是任意文件记录；动作只保留 SHA-256、字节数和文件类型，丢弃路径与正文。
    函数无外部副作用，返回可发布的文件摘要字典；输入不是映射或字段不能安全复制
    时返回空字典或省略字段，不抛出业务异常。
    """

    source = value if isinstance(value, Mapping) else {}
    result: dict[str, Any] = {}
    for key in ("sha256", "size_bytes", "kind"):
        projected = _final_safe_value(source.get(key))
        if projected is not None:
            result[key] = projected
    return result


def _project_final_foreach_child(value: Any) -> dict[str, Any]:
    """投影一个 child 的科学结果、产物和文件摘要。

    输入是一个 child 原始结果映射；动作保留 ``status``、``ready``、允许的科学值、
    ArtifactRef（产物引用，记录可核对的产物身份）摘要、collect 文件摘要和声明产物校验状态，删除 job、operation、请求、
    路径和日志等内部执行事实。函数只在内存中构造结果，不修改输入或写文件；返回
    可发布的 child 字典，缺少或类型不符的可选字段被省略。输入不是映射时返回空字典，
    不抛出业务异常。
    """

    source = value if isinstance(value, Mapping) else {}
    result: dict[str, Any] = {}
    for key in ("status", "ready"):
        projected = _final_safe_value(source.get(key))
        if projected is not None:
            result[key] = projected
    nested = source.get("result")
    if isinstance(nested, Mapping):
        projected = _final_safe_value(nested, scientific=True)
        if isinstance(projected, Mapping):
            result["result"] = projected
    artifacts = source.get("output_artifacts")
    if isinstance(artifacts, list):
        result["output_artifacts"] = [
            _project_final_artifact_summary(item)
            for item in artifacts
            if isinstance(item, Mapping)
        ]
    for field in ("files", "outputs"):
        values = source.get(field)
        if isinstance(values, list):
            result[field] = [
                _project_final_file_summary(item)
                for item in values
                if isinstance(item, Mapping)
            ]
    declared = source.get("declared_artifacts")
    if isinstance(declared, list):
        result["declared_artifacts"] = [
            {
                key: projected
                for key in (
                    "name",
                    "semantic_type",
                    "media_type",
                    "producer",
                    "validation_status",
                    "validation_errors",
                    "errors",
                )
                if (projected := _final_safe_value(item.get(key))) is not None
            }
            for item in declared
            if isinstance(item, Mapping)
        ]
    return result


def _project_final_native_foreach_output(raw_output: Mapping[str, Any]) -> dict[str, Any]:
    """把 canonical native foreach 结果变为最终结果视图的有界公开外壳。

    输入是 canonical native foreach 原始结果，其中 fanout（分发清单，记录 child 顺序
    和远端身份）包含每项结果；动作先复用 Router 校验和投影，再逐项清理来源说明、
    科学结果与 ArtifactRef（产物引用，记录可核对的产物身份），生成 bounded（有界，
    限制字段和内容规模且不含运行身份）的公开结构。函数只在内存中构造新映射，不修改
    原始结果或写文件；返回可发布的 ``status``、``ready``、结果项和产物摘要。canonical
    清单、child 身份或结果结构不合法时，底层校验抛出 ``ValueError``，不返回部分视图。
    """

    result, _omissions, parent_artifacts = _project_native_foreach_result(
        raw_output,
        "result_view",
    )
    items = result["items"]
    projected_items: list[dict[str, Any]] = []
    for item in items:
        child = item.get("result")
        projected_items.append(
            {
                "item_id": _final_safe_value(item.get("item_id")),
                "provenance": _project_final_foreach_provenance(item.get("provenance")),
                "result": _project_final_foreach_child(child),
            }
    )
    public_result: dict[str, Any] = {"items": projected_items}
    for key in ("item_count", "failed_item_count"):
        public_result[key] = result[key]
    output: dict[str, Any] = {
        "status": raw_output.get("status"),
        "ready": raw_output.get("ready"),
        "execution_layer": "native_software",
        "result": public_result,
    }
    output["output_artifacts"] = [
        _project_final_artifact_summary(item) for item in parent_artifacts
    ]
    return output


def _project_execution_result_for_result_view(
    execution_result: Mapping[str, Any],
) -> dict[str, Any]:
    """为结果视图复制执行结果，并仅投影 native foreach 的内部 fanout 事实。

    输入是完整 execution result；动作深拷贝顶层结构，识别带有 native foreach fanout
    的步骤并调用最终结果投影器，旧的单任务步骤保持原结构。函数不修改调用方数据、
    不写文件；返回新的执行结果映射，步骤列表缺失时返回同样内容的副本。嵌套 native
    foreach 清单或 child 结构不合法时，投影器抛出 ``ValueError``，阻止不安全结果发布。
    """

    projected = deepcopy(dict(execution_result))
    steps = projected.get("steps")
    if not isinstance(steps, list):
        return projected
    for step in steps:
        if not isinstance(step, dict):
            continue
        raw_output = step.get("raw_output")
        foreach = raw_output.get("foreach") if isinstance(raw_output, Mapping) else None
        if (
            isinstance(raw_output, Mapping)
            and isinstance(foreach, Mapping)
            and foreach.get("execution_mode") == "native_foreach"
        ):
            step["raw_output"] = _project_final_native_foreach_output(raw_output)
    return projected


def _looks_like_minichem_attempt(terminal_payload: Mapping[str, Any]) -> bool:
    """根据当前执行记录的 MiniChem 身份字段识别新协议 Attempt。"""

    execution_result = terminal_payload.get("execution_result")
    if not isinstance(execution_result, Mapping):
        return False
    metadata = execution_result.get("metadata")
    if isinstance(metadata, Mapping) and "catalog_hash" in metadata:
        return True
    steps = execution_result.get("steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if any(field in step for field in ("execution_layer", "action_id", "backend_id")):
            return True
        raw_output = step.get("raw_output")
        if isinstance(raw_output, Mapping) and any(
            field in raw_output
            for field in (
                "output_artifacts",
                "declared_artifacts",
                "operation_id",
                "job_id",
            )
        ):
            return True
    validation = terminal_payload.get("program_validation")
    gate = (
        validation.get("program_gate_result")
        if isinstance(validation, Mapping)
        else None
    )
    return isinstance(gate, Mapping) and "catalog_hash" in gate


def _validate_terminal_router_gate(terminal_payload: Mapping[str, Any]) -> None:
    """发布前复核终态 Attempt 的 Gate 事实和旧 Router 兼容约束。

    新的 MiniChem Attempt 一定保存包含 ``program_gate_result`` 的
    ``program_validation``；明确识别为旧非 MiniChem 形状的快照才保留只读兼容。
    所有 MiniChem Attempt 都核对 Router decision 与 Gate 的 allowed_decisions，
    并在 evidence_ready 时复验 Gate 约束。
    状态轴与声明校验只作为执行阶段写入的审计上下文，不在结果视图中重新解释。
    任何冲突都会阻止 bounded result 发布。
    """

    is_minichem = _looks_like_minichem_attempt(terminal_payload)
    if "program_validation" not in terminal_payload:
        if is_minichem:
            raise ValueError("MiniChem 终态 Attempt 缺少 program_validation")
        return
    program_validation = terminal_payload.get("program_validation")
    if not isinstance(program_validation, Mapping):
        if not is_minichem:
            return
        raise ValueError("终态 Attempt 的 program_validation 必须是 object")
    router_decision = _required_mapping(
        terminal_payload.get("router_decision"),
        "terminal router decision",
    )
    decision = router_decision.get("decision")
    gate = program_validation.get("program_gate_result")
    if not isinstance(gate, Mapping):
        if not is_minichem:
            return
        raise ValueError("终态 Attempt 缺少 program_gate_result")
    allowed = gate.get("allowed_decisions")
    if not isinstance(allowed, list) or decision not in allowed:
        raise ValueError(
            "终态 Router decision 不在同一 Program Gate allowed_decisions 中"
        )
    if decision == "evidence_ready":
        validate_program_gate_for_evidence_ready(program_validation)


def _planning_result(
    scientific_question: str,
    workflow: Mapping[str, Any],
) -> dict[str, Any]:
    """把某次 Attempt 实际使用的单条工作流放回旧 Planner 结果外壳。"""

    copied = deepcopy(dict(workflow))
    steps = copied.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Attempt 对应 workflow 缺少 steps")
    return {
        "question": scientific_question,
        "top_n": 1,
        "total_strategies": 1,
        "processed_strategies": 1,
        "protocols": [deepcopy(copied)],
        "optimized_protocols": [deepcopy(copied)],
        "total_original_steps": len(steps),
        "total_optimized_steps": len(steps),
        "optimization_ratio": 1.0,
    }


def _workflow_versions(
    planning_record: Mapping[str, Any],
    revisions: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    """依据 Planning 和顺序修订重建每个 Attempt 使用的 workflow snapshot。"""

    planning_payload = _required_mapping(planning_record.get("payload"), "planning payload")
    initial = _required_mapping(planning_payload.get("workflow"), "planning workflow")
    versions: dict[int, dict[str, Any]] = {1: initial}
    expected_before = 1
    for record in revisions:
        payload = _required_mapping(record.get("payload"), "revision payload")
        before = payload.get("workflow_version_before")
        after = payload.get("workflow_version_after")
        if before != expected_before or after != before + 1:
            raise ValueError("工作流修订版本不连续")
        versions[after] = _required_mapping(
            payload.get("after_workflow"), "revision after_workflow"
        )
        expected_before = after
    return versions


def _validated_bounded_result_inputs(
    package: Mapping[str, Any],
    memory_status: str,
) -> dict[str, Any]:
    """校验最终视图的输入身份、正文和连续历史。"""

    scientific_question = package["scientific_question"]
    task_id = package["task_id"]
    if not isinstance(scientific_question, str) or not scientific_question:
        raise ValueError("scientific_question 必须是非空字符串")
    if memory_status == "available":
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("enabled bounded result 必须包含正式 task_id")
    elif memory_status == "disabled":
        if task_id is not None:
            raise ValueError("disabled bounded result 的 task_id 必须为空")
    else:
        raise ValueError("memory_status 不在最终结果允许集合中")

    task_config = _required_mapping(package["task_config"], "task_config")
    chemistry_context = _required_mapping(
        package["chemistry_context"], "chemistry_context"
    )
    retrieval_phase = _required_mapping(
        package["retrieval_phase"], "retrieval_phase"
    )
    hypothesis_phase = _bounded_hypothesis_phase(
        _required_mapping(package["hypothesis_phase"], "hypothesis_phase")
    )
    hypothesis = _bounded_hypothesis_projection(
        _required_mapping(package["hypothesis"], "hypothesis")
    )
    planning_record = _required_mapping(
        package["planning_record"], "planning_record"
    )
    attempts_raw = package["attempt_records"]
    revisions_raw = package["revision_records"]
    pauses_raw = package["pause_restart_records"]
    batch_outcomes = _batch_combination_outcomes(
        package["batch_combination_outcomes"]
    )
    if (
        not isinstance(attempts_raw, list)
        or not attempts_raw
        or not isinstance(revisions_raw, list)
        or not isinstance(pauses_raw, list)
    ):
        raise ValueError("bounded result 历史数组类型不正确")
    attempts = [_required_mapping(item, "attempt record") for item in attempts_raw]
    revisions = [
        _required_mapping(item, "revision record") for item in revisions_raw
    ]
    attempts.sort(key=lambda item: item.get("attempt_index", 0))
    revisions.sort(
        key=lambda item: (item.get("created_at", ""), item.get("record_id", ""))
    )
    if [item.get("attempt_index") for item in attempts] != list(
        range(1, len(attempts) + 1)
    ):
        raise ValueError("Attempt index 必须从 1 连续排列")
    terminal = attempts[-1]
    terminal_payload = _required_mapping(
        terminal.get("payload"), "terminal attempt payload"
    )
    terminal_router = _required_mapping(
        terminal_payload.get("router_decision"), "terminal router decision"
    )
    _validate_terminal_router_gate(terminal_payload)
    if (
        package["terminal_record_id"] != terminal.get("record_id")
        or terminal_router.get("decision") != "evidence_ready"
    ):
        raise ValueError("终态记录必须是最后一个 evidence_ready Attempt")

    return {
        "scientific_question": scientific_question,
        "task_id": task_id,
        "task_config": task_config,
        "chemistry_context": chemistry_context,
        "retrieval_phase": retrieval_phase,
        "hypothesis_phase": hypothesis_phase,
        "hypothesis": hypothesis,
        "planning_record": planning_record,
        "attempts": attempts,
        "revisions": revisions,
        "batch_outcomes": batch_outcomes,
        "terminal_payload": terminal_payload,
    }


def _build_attempt_history(
    scientific_question: str,
    attempts: Sequence[Mapping[str, Any]],
    workflow_by_version: Mapping[int, Mapping[str, Any]],
    *,
    summarize_execution: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """按 Attempt 顺序构造规划、执行、Router 和修订历史。"""

    planning_rounds: list[dict[str, Any]] = []
    execution_rounds: list[dict[str, Any]] = []
    reflection_rounds: list[dict[str, Any]] = []
    revision_history: list[dict[str, Any]] = []
    attempt_lookup: dict[str, Mapping[str, Any]] = {}

    for attempt in attempts:
        record_id = attempt.get("record_id")
        attempt_index = attempt["attempt_index"]
        if not isinstance(record_id, str) or not record_id or record_id in attempt_lookup:
            raise ValueError("Attempt record_id 非法或重复")
        payload = _required_mapping(attempt.get("payload"), "attempt payload")
        workflow_version = payload.get("workflow_version")
        workflow = workflow_by_version.get(workflow_version)
        if workflow is None:
            raise ValueError("Attempt 找不到对应的 workflow version")
        execution_result = _required_mapping(
            payload.get("execution_result"), "attempt execution_result"
        )
        router_decision = _required_mapping(
            payload.get("router_decision"), "attempt router_decision"
        )
        decision = router_decision.get("decision")
        if decision not in _DECISION_VIEW:
            raise ValueError("Attempt Router decision 非法")

        # 校验每个步骤都能被最终结果完整展示，包括 MiniChem 的 raw_output。
        _execution_steps(execution_result)
        created_at = attempt.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("Attempt created_at 必须是非空字符串")
        attempt_lookup[record_id] = attempt

        execution_summary = summarize_execution(deepcopy(execution_result))
        if not isinstance(execution_summary, Mapping):
            raise ValueError("summarize_execution 必须返回 object")
        execution_summary = deepcopy(dict(execution_summary))
        execution_failed = bool(execution_result.get("error")) or (
            execution_summary.get("overall_status") == "failed"
        )
        planning_rounds.append(
            {
                "round": attempt_index,
                "result": _planning_result(scientific_question, workflow),
                "status": "success",
            }
        )
        execution_rounds.append(
            {
                "round": attempt_index,
                "result": _project_execution_result_for_result_view(execution_result),
                "status": "failed" if execution_failed else "success",
            }
        )
        mapped_decision = _DECISION_VIEW[decision]
        reflection_rounds.append(
            {
                "round": attempt_index,
                "result": _bounded_router_decision(router_decision),
                "decision": mapped_decision,
            }
        )
        diagnosis = router_decision.get("diagnosis")
        if isinstance(diagnosis, Mapping):
            summary = diagnosis.get("summary")
        else:
            summary = None
        if decision != "evidence_ready" and (
            not isinstance(summary, str) or not summary.strip()
        ):
            raise ValueError("非成功 Router 结论必须包含明确问题摘要")
        if decision == "evidence_ready":
            identified_problems = []
        else:
            identified_problems = [summary]
        revision_history.append(
            {
                "round": attempt_index,
                "decision": mapped_decision,
                "identified_problems": identified_problems,
                "workflow_revision_instructions": [],
                "hypothesis_revision_instructions": [],
            }
        )

    return {
        "planning_rounds": planning_rounds,
        "execution_rounds": execution_rounds,
        "reflection_rounds": reflection_rounds,
        "revision_history": revision_history,
        "attempt_lookup": attempt_lookup,
    }


def _build_revision_events(
    revisions: Sequence[Mapping[str, Any]],
    attempt_lookup: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """把模型修订和旧人工作流修订整理为公开事件。

    当前 MiniChem 只会写入模型修订；旧 SQLite 中的人工作流修订仍可读取。
    人工修订只使用已经保存的前后工作流、规范化修改和暂停原因，不会触发任何
    Restart、补丁写入或专家分析逻辑。
    """

    revision_events: list[dict[str, Any]] = []
    for revision in revisions:
        revision_type = revision.get("revision_type")
        source_attempt_id = revision.get("source_attempt_record_id")
        payload = _required_mapping(revision.get("payload"), "revision payload")
        normalized_patch = _required_mapping(
            payload.get("normalized_patch"), "revision normalized_patch"
        )
        after_workflow = _required_mapping(
            payload.get("after_workflow"), "revision after_workflow"
        )
        if revision_type == "model_workflow_revision":
            source_attempt = attempt_lookup.get(source_attempt_id)
            if source_attempt is None:
                raise ValueError("模型修订缺少来源 Attempt")
            round_index = source_attempt["attempt_index"]
            source_payload = _required_mapping(
                source_attempt.get("payload"), "source attempt payload"
            )
            source_router = _required_mapping(
                source_payload.get("router_decision"), "source router decision"
            )
            source_diagnosis = source_router.get("diagnosis")
            if isinstance(source_diagnosis, Mapping):
                problem = source_diagnosis.get("summary")
            else:
                problem = ""
        elif revision_type == "human_workflow_patch":
            if source_attempt_id is not None:
                raise ValueError("历史人工修订不能引用 Attempt")
            before_workflow = _required_mapping(
                payload.get("before_workflow"), "revision before_workflow"
            )
            if before_workflow == after_workflow:
                raise ValueError("工作流修订前后的内容不能完全相同")
            round_index = revision.get("applied_attempt_index")
            if isinstance(round_index, bool) or not isinstance(round_index, int):
                raise ValueError("历史人工修订缺少有效的应用轮次")
            problem = revision.get("source_pause_reason") or "tool_execution_failure"
        else:
            raise ValueError("最终视图遇到未知工作流修订类型")
        revision_events.append(
            {
                "round": round_index,
                "type": "workflow_revision",
                "identified_problems": [str(problem or "")],
                "workflow_revision_instructions": [normalized_patch],
                "recommended_actions": [],
                "status": "success",
                "revised_result": after_workflow,
            }
        )
    return revision_events


def _build_conclusion_sections(
    terminal_payload: Mapping[str, Any],
    retrieval_phase: Mapping[str, Any],
    hypothesis_phase: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
    execution_rounds: Sequence[Mapping[str, Any]],
    revision_events: Sequence[Mapping[str, Any]],
    attempt_count: int,
    *,
    summarize_execution: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """从终态执行和完整轮次构造结论使用的确定性摘要。"""

    terminal_execution = _required_mapping(
        terminal_payload.get("execution_result"), "terminal execution_result"
    )
    terminal_summary = summarize_execution(deepcopy(terminal_execution))
    if not isinstance(terminal_summary, Mapping):
        raise ValueError("terminal execution summary 必须是 object")
    terminal_summary = deepcopy(dict(terminal_summary))
    success_rate = terminal_summary.get("overall_success_rate", 0.0)
    if isinstance(success_rate, bool) or not isinstance(success_rate, (int, float)):
        success_rate = 0.0
    success_rate = float(success_rate)

    ranked = hypothesis_phase.get("ranked_strategies")
    if not isinstance(ranked, list):
        raise ValueError("hypothesis_phase 缺少 ranked_strategies")
    literature_available = bool(retrieval_phase.get("literature_review", ""))
    evidence_summary = {
        "literature_review_available": literature_available,
        "hypotheses_generated": len(ranked),
        "workflows_executed": attempt_count,
        "successful_executions": sum(
            item["status"] == "success" for item in execution_rounds
        ),
    }
    workflow_outcome = {
        "total_rounds": attempt_count,
        "execution_success_rate": success_rate,
        "overall_status": terminal_summary.get("overall_status", "unknown"),
        "workflow_outcome": terminal_summary.get("workflow_outcome", "unknown"),
        "validation_overview": deepcopy(
            terminal_summary.get("validation_overview", {})
        ),
        "failed_steps": deepcopy(terminal_summary.get("failed_steps", [])),
    }
    key_findings = [
        {
            "type": "selected_strategy",
            "strategy_name": hypothesis.get("strategy_name", "Unknown"),
            "reasoning": str(hypothesis.get("reasoning", ""))[:200],
        }
    ]
    if literature_available:
        key_findings.append(
            {
                "type": "literature_context",
                "summary": "Literature review successfully retrieved and analyzed",
            }
        )
    if success_rate > 0.7:
        key_findings.append(
            {
                "type": "execution_success",
                "summary": f"High execution success rate ({success_rate:.1%})",
            }
        )
    unresolved_issues = list(terminal_summary.get("issues", []))[:5]
    if revision_events:
        unresolved_issues.append("Workflow required revisions during execution")
    recommended_next_steps: list[str] = []
    if workflow_outcome["failed_steps"]:
        recommended_next_steps.append("Debug failed computational steps")
    if literature_available:
        recommended_next_steps.append(
            "Compare computational results with literature findings"
        )
    return {
        "success_rate": success_rate,
        "evidence_summary": evidence_summary,
        "workflow_outcome": workflow_outcome,
        "key_findings": key_findings,
        "unresolved_issues": unresolved_issues,
        "recommended_next_steps": recommended_next_steps,
    }


def _build_bounded_result(
    package: Mapping[str, Any],
    *,
    memory_status: str,
    close_timestamp: str,
    summarize_execution: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """为一个成功组合构造最终磁盘 JSON，并按运行模式处理任务身份。"""

    if not isinstance(package, Mapping) or set(package) != _PACKAGE_KEYS:
        raise ValueError("bounded result package 字段不完整")
    if not callable(summarize_execution):
        raise TypeError("summarize_execution 必须可调用")
    inputs = _validated_bounded_result_inputs(package, memory_status)
    scientific_question = inputs["scientific_question"]
    task_id = inputs["task_id"]
    task_config = inputs["task_config"]
    chemistry_context = inputs["chemistry_context"]
    retrieval_phase = inputs["retrieval_phase"]
    hypothesis_phase = inputs["hypothesis_phase"]
    hypothesis = inputs["hypothesis"]
    planning_record = inputs["planning_record"]
    attempts = inputs["attempts"]
    revisions = inputs["revisions"]
    batch_outcomes = inputs["batch_outcomes"]
    terminal_payload = inputs["terminal_payload"]

    workflow_by_version = _workflow_versions(planning_record, revisions)
    attempt_history = _build_attempt_history(
        scientific_question,
        attempts,
        workflow_by_version,
        summarize_execution=summarize_execution,
    )
    planning_rounds = attempt_history["planning_rounds"]
    execution_rounds = attempt_history["execution_rounds"]
    reflection_rounds = attempt_history["reflection_rounds"]
    revision_history = attempt_history["revision_history"]
    attempt_lookup = attempt_history["attempt_lookup"]

    revision_events = _build_revision_events(
        revisions,
        attempt_lookup,
    )
    conclusion = _build_conclusion_sections(
        terminal_payload,
        retrieval_phase,
        hypothesis_phase,
        hypothesis,
        execution_rounds,
        revision_events,
        len(attempts),
        summarize_execution=summarize_execution,
    )
    success_rate = conclusion["success_rate"]
    evidence_summary = conclusion["evidence_summary"]
    workflow_outcome = conclusion["workflow_outcome"]
    key_findings = conclusion["key_findings"]
    unresolved_issues = conclusion["unresolved_issues"]
    recommended_next_steps = conclusion["recommended_next_steps"]

    workflow_start_time = _timestamp(package["workflow_start_time"], "workflow_start_time")
    workflow_end_time = _timestamp(close_timestamp, "close_timestamp")
    max_revisions = task_config.get("max_workflow_revisions_per_combination")
    if isinstance(max_revisions, bool) or not isinstance(max_revisions, int):
        raise ValueError("task_config 缺少工作流修订预算")
    shared_state = {
        "current_question": scientific_question,
        "chemistry_context": chemistry_context,
        "revision_history": revision_history,
    }
    final_conclusion = {
        "scientific_question": scientific_question,
        "conclusion_type": "supported",
        "conclusion_summary": (
            "The selected bounded hypothesis was supported by the recorded "
            "computational evidence; this conclusion does not by itself establish "
            "the full scientific question."
        ),
        "selected_strategy": hypothesis,
        "workflow_outcome": workflow_outcome,
        "evidence_summary": evidence_summary,
        "key_findings": key_findings,
        "unresolved_issues": unresolved_issues,
        "recommended_next_steps": recommended_next_steps,
        "final_status": "accepted",
        "final_decision": "accept",
        "total_reflection_rounds": len(attempts),
        "confidence": min(0.9, success_rate + 0.2),
    }
    result = {
        "scientific_question": scientific_question,
        "workflow_start_time": workflow_start_time,
        "workflow_version": "bounded_closed_loop_v1",
        "task_id": task_id,
        "memory_status": memory_status,
        "memory_db_path": None,
        "shared_state": shared_state,
        "retrieval_phase": retrieval_phase,
        "hypothesis_phase": hypothesis_phase,
        "planning_rounds": planning_rounds,
        "execution_rounds": execution_rounds,
        "reflection_rounds": reflection_rounds,
        "retrieval_followup_rounds": [],
        "revision_events": revision_events,
        "stop_conditions": {
            "max_reflection_rounds": max_revisions + 1,
            "max_unrecoverable_failures": 0,
            "triggered_condition": "reflection_accept",
            "final_round": attempts[-1]["attempt_index"],
        },
        "final_conclusion": final_conclusion,
        "status": "accepted",
        "workflow_end_time": workflow_end_time,
        "total_duration_seconds": max(0.0, workflow_end_time - workflow_start_time),
        "batch_combination_outcomes": batch_outcomes,
    }
    return strip_internal_claims_for_bounded_result(result)


def build_enabled_bounded_result(
    package: Mapping[str, Any],
    *,
    close_timestamp: str,
    summarize_execution: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """把带正式任务身份的成功组合投影成 bounded JSON。"""

    return _build_bounded_result(
        package,
        memory_status="available",
        close_timestamp=close_timestamp,
        summarize_execution=summarize_execution,
    )


def build_disabled_bounded_result(
    package: Mapping[str, Any],
    *,
    close_timestamp: str,
    summarize_execution: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """把进程内成功组合投影成同形结果，并明确省略持久任务身份。"""

    return _build_bounded_result(
        package,
        memory_status="disabled",
        close_timestamp=close_timestamp,
        summarize_execution=summarize_execution,
    )


__all__ = [
    "build_disabled_bounded_result",
    "build_enabled_bounded_result",
    "strip_internal_claims_for_bounded_result",
]
