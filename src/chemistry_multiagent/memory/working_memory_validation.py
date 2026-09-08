"""Hypothesis provider 使用的 Investigation Working Memory 严格校验器。

本模块只依赖 memory 合同，避免导入 Investigation 聚合包而引入循环依赖。它在
provider 消息边界复核固定 JSON schema、身份引用和私有字段，成功后返回独立副本。
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from chemistry_multiagent.memory.contracts import (
    validate_structured_hypothesis_claims,
)


_ROOT_KEYS = {
    "schema",
    "schema_version",
    "task_id",
    "batch_id",
    "scientific_question",
    "chemistry_context",
    "retrieval_context",
    "closed_batches",
    "current_batch",
}
_BATCH_KEYS = {
    "task_id",
    "batch_id",
    "batch_index",
    "route",
    "final_answer",
    "answerability_explanation",
    "sessions",
    "results",
    "validations",
    "combinations",
}
_SESSION_KEYS = {"combination_id", "result_version", "repair_count", "route_marker"}
_COMBINATION_KEYS = {
    "task_id",
    "batch_id",
    "combination_id",
    "combination_index",
    "hypothesis_id",
    "hypothesis",
}
_RESULT_KEYS = {
    "combination_id",
    "result_version",
    "self_judgment",
    "narrative",
    "job_refs",
    "operation_refs",
    "artifact_refs",
    "observations",
    "limitations",
}
_VALIDATION_KEYS = {
    "combination_ids",
    "result_versions",
    "findings",
    "can_answer",
    "final_answer",
    "answerability_explanation",
}
_FINDING_KEYS = {"combination_id", "validity", "hypothesis_judgment", "repair_advice"}
_HYPOTHESIS_KEYS = {
    "hypothesis_id",
    "strategy_name",
    "reasoning",
    "merged_from",
    "synthesis_sources",
    "wins",
    "losses",
    "score",
    "rank",
    "core_claim",
    "claims",
    "detailed_reasoning",
    "verification_requirements",
}
_PRIVATE_KEYS = {
    "workspace",
    "workspace_ref",
    "workspace_path",
    "scope_id",
    "thread_id",
    "thread",
    "codex_thread_id",
    "review_thread",
    "review_thread_id",
    "review_turn",
    "review_turn_id",
    "turn",
    "turn_id",
    "transcript",
    "workflow",
    "current_workflow",
    "runtime",
    "runtime_checkpoint",
    "checkpoint",
    "path",
    "paths",
    "manifest_path",
    "resume_state_path",
    "response_cache_dir",
    "source_ref",
    "source_refs",
    "event_id",
    "record_id",
    "api_key",
    "access_token",
    "secret",
    "password",
    "credential",
    "credentials",
    "raw_output",
    "raw_tool_output",
    "stdout",
    "stderr",
    "tool_output",
    "investigation_task_failed",
}
_PRIVATE_KEY_NAMES = {
    re.sub(r"[^a-z0-9]", "", key.casefold()) for key in _PRIVATE_KEYS
}
_PRIVATE_TEXT_RE = re.compile(
    r"\b(?:workspace_ref|workspace_path|scope_id|thread_id|codex_thread_id|"
    r"review_thread_id|review_turn_id|transcript)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?:^|[\s(])(?:/|[A-Za-z]:[\\/]|file://|\.\.?/|"
    r"(?:local_data|outputs|working_memory|\.runtime|run_bundle|tmp|home|workspace|scope)/)",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"(?:api[_ -]?key|access[_ -]?token|secret|password)\s*[:=]|"
    r"\bsk-[A-Za-z0-9_-]{8,}\b",
    re.IGNORECASE,
)


def _canonicalize(value: Any, label: str) -> dict[str, Any]:
    """把调用方值转换为严格 JSON object。"""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 必须是严格 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 必须是 object")
    return payload


def _walk_safe_mapping(value: Mapping[str, Any], path: str) -> None:
    """递归检查 object 字段名并继续检查字段值。"""

    for key, nested in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path} 的 object key 非法")
        normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
        if normalized in _PRIVATE_KEY_NAMES:
            raise ValueError(f"{path}.{key} 是私有运行字段")
        _walk_safety(nested, f"{path}.{key}")


def _walk_safe_list(value: list[Any], path: str) -> None:
    """递归检查 array 中的每个元素。"""

    for index, nested in enumerate(value):
        _walk_safety(nested, f"{path}[{index}]")


def _walk_safe_scalar(value: Any, path: str) -> None:
    """检查字符串和 JSON 标量不能携带私有身份或不安全值。"""

    if isinstance(value, str):
        if _PRIVATE_TEXT_RE.search(value) or _PATH_RE.search(value) or _SECRET_RE.search(value):
            raise ValueError(f"{path} 不能包含私有身份、路径或凭据")
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{path} 不是严格 JSON 值")


def _walk_safety(value: Any, path: str) -> None:
    """递归拒绝私有字段、路径、URI、控制字符和凭据。"""

    if isinstance(value, Mapping):
        _walk_safe_mapping(value, path)
    elif isinstance(value, list):
        _walk_safe_list(value, path)
    else:
        _walk_safe_scalar(value, path)


def _mapping(value: Any, label: str, expected: set[str]) -> Mapping[str, Any]:
    """确认 object 字段集合精确匹配注册 schema。"""

    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} 字段不符合固定 schema")
    return value


def _text(value: Any, label: str) -> str:
    """确认身份和科学正文是非空字符串。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value


def _strict_index(value: Any, expected: int, label: str) -> None:
    """确认批次或组合序号是严格整数且与冻结顺序一致。"""

    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{label} 不连续或类型非法")


def _validate_sessions(value: Any, label: str) -> tuple[set[str], dict[str, str]]:
    """校验当前批次的 session 并返回结果版本向量。

    v2 的冻结任务配置负责决定应有数量；这里仅拒绝空批次和超过协议上限的输入，
    再由组合集合与 session 集合做精确覆盖校验。
    """

    if not isinstance(value, list) or len(value) not in {1, 5}:
        raise ValueError(f"{label} 必须包含一个或五个 session")
    ids: set[str] = set()
    versions: dict[str, str] = {}
    for index, item in enumerate(value):
        row = _mapping(item, f"{label}[{index}]", _SESSION_KEYS)
        combination_id = _text(row["combination_id"], f"{label}[{index}].combination_id")
        if combination_id in ids:
            raise ValueError(f"{label}[{index}].combination_id 重复")
        result_version = _text(row["result_version"], f"{label}[{index}].result_version")
        repair_count = row["repair_count"]
        if isinstance(repair_count, bool) or not isinstance(repair_count, int) or not 0 <= repair_count <= 3:
            raise ValueError(f"{label}[{index}].repair_count 非法")
        if row["route_marker"] != "revise_hypothesis":
            raise ValueError(f"{label}[{index}].route_marker 非法")
        ids.add(combination_id)
        versions[combination_id] = result_version
    return ids, versions


def _validate_combination_row(
    item: Any,
    *,
    task_id: str,
    batch_id: str,
    index: int,
    session_ids: set[str],
    combination_ids: set[str],
    hypothesis_ids: set[str],
    label: str,
) -> None:
    """校验一项组合并登记其身份。"""

    row = _mapping(item, label, _COMBINATION_KEYS)
    if row["task_id"] != task_id or row["batch_id"] != batch_id:
        raise ValueError(f"{label} 任务或批次身份不一致")
    _strict_index(row["combination_index"], index + 1, f"{label}.combination_index")
    combination_id = _text(row["combination_id"], f"{label}.combination_id")
    hypothesis_id = _text(row["hypothesis_id"], f"{label}.hypothesis_id")
    if combination_id in combination_ids or combination_id not in session_ids or hypothesis_id in hypothesis_ids:
        raise ValueError(f"{label} 身份未与 session 对齐")
    hypothesis = row["hypothesis"]
    if not isinstance(hypothesis, Mapping) or not {"hypothesis_id", "strategy_name", "reasoning"} <= set(hypothesis) or set(hypothesis) - _HYPOTHESIS_KEYS:
        raise ValueError(f"{label}.hypothesis 字段不符合固定 schema")
    if hypothesis.get("hypothesis_id") != hypothesis_id:
        raise ValueError(f"{label}.hypothesis.hypothesis_id 身份漂移")
    _text(hypothesis.get("strategy_name"), f"{label}.hypothesis.strategy_name")
    _text(hypothesis.get("reasoning"), f"{label}.hypothesis.reasoning")
    try:
        validate_structured_hypothesis_claims(
            hypothesis,
            path=f"{label}.hypothesis",
            allow_legacy=True,
        )
    except Exception as exc:
        raise ValueError(f"{label}.hypothesis 结构化正文非法") from exc
    combination_ids.add(combination_id)
    hypothesis_ids.add(hypothesis_id)


def _validate_combinations(
    value: Any,
    *,
    task_id: str,
    batch_id: str,
    session_ids: set[str],
    label: str,
) -> set[str]:
    """校验五个组合以及 selected hypothesis 的身份对齐。"""

    if not isinstance(value, list) or len(value) not in {1, 5}:
        raise ValueError(f"{label} 必须包含一个或五个组合")
    combination_ids: set[str] = set()
    hypothesis_ids: set[str] = set()
    for index, item in enumerate(value):
        _validate_combination_row(
            item,
            task_id=task_id,
            batch_id=batch_id,
            index=index,
            session_ids=session_ids,
            combination_ids=combination_ids,
            hypothesis_ids=hypothesis_ids,
            label=f"{label}[{index}]",
        )
    if combination_ids != session_ids:
        raise ValueError(f"{label} 与 session 没有完整对齐")
    return combination_ids


def _validate_results(value: Any, *, combination_ids: set[str], label: str) -> set[tuple[str, str]]:
    """校验结果正文及其五类证据数组。"""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} 不能为空")
    keys: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        row_label = f"{label}[{index}]"
        row = _mapping(item, row_label, _RESULT_KEYS)
        combination_id = _text(row["combination_id"], f"{row_label}.combination_id")
        result_version = _text(row["result_version"], f"{row_label}.result_version")
        pair = (combination_id, result_version)
        if combination_id not in combination_ids or pair in keys:
            raise ValueError(f"{row_label} 结果身份重复或未知")
        if row["self_judgment"] not in {"supported", "refuted"}:
            raise ValueError(f"{row_label}.self_judgment 非法")
        _text(row["narrative"], f"{row_label}.narrative")
        for field in ("job_refs", "operation_refs", "artifact_refs", "observations", "limitations"):
            items = row[field]
            if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
                raise ValueError(f"{row_label}.{field} 必须是字符串数组")
        keys.add(pair)
    return keys


def _validate_finding_identity(
    row: Mapping[str, Any],
    *,
    combination_ids: set[str],
    label: str,
) -> str:
    """校验 finding 的组合身份。"""

    combination_id = _text(row["combination_id"], f"{label}.combination_id")
    if combination_id not in combination_ids:
        raise ValueError(f"{label}.combination_id 未知")
    return combination_id


def _validate_finding_state(row: Mapping[str, Any], label: str) -> None:
    """校验 finding 的 validity 和 hypothesis judgment 枚举。"""

    if row["validity"] not in {"pass", "fail"} or row["hypothesis_judgment"] not in {
        None,
        "supported",
        "refuted",
    }:
        raise ValueError(f"{label} 判断字段非法")


def _validate_finding_advice_type(row: Mapping[str, Any], label: str) -> None:
    """校验 finding 的 repair_advice 类型。"""

    advice = row["repair_advice"]
    if advice is not None and (not isinstance(advice, str) or not advice.strip()):
        raise ValueError(f"{label}.repair_advice 非法")


def _validate_finding_terminal(row: Mapping[str, Any], label: str) -> None:
    """校验 pass/fail finding 对应的判断或修复建议。"""

    validity = row["validity"]
    judgment = row["hypothesis_judgment"]
    advice = row["repair_advice"]
    if validity == "pass" and (judgment is None or advice is not None):
        raise ValueError(f"{label} pass finding 内容非法")
    if validity == "fail" and (judgment is not None or not isinstance(advice, str) or not advice.strip()):
        raise ValueError(f"{label} fail finding 内容非法")


def _validate_finding_judgment(row: Mapping[str, Any], label: str) -> None:
    """校验 finding 的状态、判断和修复建议字段。"""

    _validate_finding_state(row, label)
    _validate_finding_advice_type(row, label)
    _validate_finding_terminal(row, label)


def _validate_finding(value: Any, *, combination_ids: set[str], label: str) -> str:
    """校验一条 Review finding 的终局形状。"""

    row = _mapping(value, label, _FINDING_KEYS)
    combination_id = _validate_finding_identity(
        row,
        combination_ids=combination_ids,
        label=label,
    )
    _validate_finding_judgment(row, label)
    return combination_id


def _validate_validation_vector_shape(
    ids: Any,
    versions: Any,
    findings: Any,
    *,
    combination_ids: set[str],
    label: str,
) -> tuple[list[Any], list[Any], list[Any]]:
    """确认 Review 的三个五项向量形状正确。"""

    if (
        not isinstance(ids, list) or len(ids) not in {1, 5} or set(ids) != combination_ids
        or len(set(ids)) != len(ids) or not isinstance(versions, list)
        or len(versions) != len(ids) or not isinstance(findings, list)
        or len(findings) != len(ids)
    ):
        raise ValueError(f"{label} 覆盖向量非法")
    return ids, versions, findings


def _validate_validation_result_pairs(
    ids: list[Any],
    versions: list[Any],
    *,
    result_keys: set[tuple[str, str]],
    label: str,
) -> None:
    """确认 Review 的五个结果版本都来自已保存结果。"""

    for combination_id, version in zip(ids, versions):
        if not isinstance(combination_id, str) or not isinstance(version, str):
            raise ValueError(f"{label} 结果版本类型非法")
        if (combination_id, version) not in result_keys:
            raise ValueError(f"{label} 引用了未知结果版本")


def _validate_validation_vectors(
    row: Mapping[str, Any],
    *,
    combination_ids: set[str],
    result_keys: set[tuple[str, str]],
    label: str,
) -> tuple[list[Any], list[Any], list[Any]]:
    """校验一轮 Review 的组合、结果版本和 finding 向量。"""

    ids, versions, findings = _validate_validation_vector_shape(
        row["combination_ids"],
        row["result_versions"],
        row["findings"],
        combination_ids=combination_ids,
        label=label,
    )
    _validate_validation_result_pairs(
        ids,
        versions,
        result_keys=result_keys,
        label=label,
    )
    return ids, versions, findings


def _validate_validation_findings(
    findings: list[Any],
    *,
    combination_ids: set[str],
    label: str,
) -> None:
    """校验 Review 的五条 finding 覆盖全部组合。"""

    finding_ids = {
        _validate_finding(
            finding,
            combination_ids=combination_ids,
            label=f"{label}[{index}]",
        )
        for index, finding in enumerate(findings)
    }
    if finding_ids != combination_ids:
        raise ValueError(f"{label} 覆盖不完整")


def _validate_validation_terminal(row: Mapping[str, Any], label: str) -> None:
    """校验 Review 的可回答标志和可选终局解释。"""

    if not isinstance(row["can_answer"], bool):
        raise ValueError(f"{label}.can_answer 非法")
    for field in ("final_answer", "answerability_explanation"):
        if row[field] is not None and not isinstance(row[field], str):
            raise ValueError(f"{label}.{field} 非法")


def _validate_validation_row(
    item: Any,
    *,
    combination_ids: set[str],
    result_keys: set[tuple[str, str]],
    current_versions: dict[str, str],
    label: str,
    latest: bool,
) -> None:
    """校验一轮 Review，并在最新轮核对当前结果版本。"""

    row = _mapping(item, label, _VALIDATION_KEYS)
    ids, versions, findings = _validate_validation_vectors(
        row,
        combination_ids=combination_ids,
        result_keys=result_keys,
        label=label,
    )
    _validate_validation_findings(
        findings,
        combination_ids=combination_ids,
        label=f"{label}.findings",
    )
    _validate_validation_terminal(row, label)
    if latest and dict(zip(ids, versions)) != current_versions:
        raise ValueError(f"{label} 未绑定 sessions 当前结果版本")


def _validate_validations(
    value: Any,
    *,
    combination_ids: set[str],
    result_keys: set[tuple[str, str]],
    current_versions: dict[str, str],
    label: str,
) -> None:
    """校验每轮五组合覆盖和最新结果版本向量。"""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} 不能为空")
    for index, item in enumerate(value):
        _validate_validation_row(
            item,
            combination_ids=combination_ids,
            result_keys=result_keys,
            current_versions=current_versions,
            label=f"{label}[{index}]",
            latest=index == len(value) - 1,
        )


def _validate_batch(
    value: Any,
    *,
    task_id: str,
    batch_index: int,
    label: str,
    seen_batch_ids: set[str],
) -> str:
    """校验一个完整历史批次并返回其身份。"""

    batch = _mapping(value, label, _BATCH_KEYS)
    current_task = _text(batch["task_id"], f"{label}.task_id")
    current_batch = _text(batch["batch_id"], f"{label}.batch_id")
    if current_task != task_id or current_batch in seen_batch_ids:
        raise ValueError(f"{label} 任务或批次身份不一致")
    _strict_index(batch["batch_index"], batch_index, f"{label}.batch_index")
    if batch["route"] != "regenerate_hypotheses":
        raise ValueError(f"{label}.route 非法")
    sessions, versions = _validate_sessions(batch["sessions"], f"{label}.sessions")
    combinations = _validate_combinations(
        batch["combinations"],
        task_id=task_id,
        batch_id=current_batch,
        session_ids=sessions,
        label=f"{label}.combinations",
    )
    results = _validate_results(
        batch["results"],
        combination_ids=combinations,
        label=f"{label}.results",
    )
    _validate_validations(
        batch["validations"],
        combination_ids=combinations,
        result_keys=results,
        current_versions=versions,
        label=f"{label}.validations",
    )
    seen_batch_ids.add(current_batch)
    return current_batch


def validate_investigation_working_memory_payload(
    value: Any,
    *,
    label: str = "investigation_working_memory",
    expected_task_id: str | None = None,
    expected_batch_id: str | None = None,
) -> dict[str, Any]:
    """严格校验并复制完整 Investigation Working Memory。

    输入可以是普通 JSON mapping；函数再次核对固定 schema、五个组合、全部结果与
    Review 版本以及 task/batch 身份。成功返回独立 dict，失败抛出 ``ValueError``。
    """

    payload = _canonicalize(value, label)
    _walk_safety(payload, label)
    root = _mapping(payload, label, _ROOT_KEYS)
    if (
        root["schema"] != "arche.investigation.regeneration_memory"
        or isinstance(root["schema_version"], bool)
        or root["schema_version"] != 1
    ):
        raise ValueError(f"{label} schema 或版本不受支持")
    task_id = _text(root["task_id"], f"{label}.task_id")
    batch_id = _text(root["batch_id"], f"{label}.batch_id")
    if expected_task_id is not None and task_id != expected_task_id:
        raise ValueError(f"{label}.task_id 与当前 operation 不一致")
    if expected_batch_id is not None and batch_id != expected_batch_id:
        raise ValueError(f"{label}.batch_id 与当前 operation 不一致")
    _text(root["scientific_question"], f"{label}.scientific_question")
    if not isinstance(root["chemistry_context"], Mapping) or not isinstance(root["retrieval_context"], Mapping):
        raise ValueError(f"{label} 科学上下文必须是 object")
    closed_batches = root["closed_batches"]
    current_batch = root["current_batch"]
    if not isinstance(closed_batches, list) or not isinstance(current_batch, Mapping):
        raise ValueError(f"{label} 批次容器非法")
    seen_batch_ids: set[str] = set()
    for index, batch in enumerate(closed_batches, start=1):
        _validate_batch(
            batch,
            task_id=task_id,
            batch_index=index,
            label=f"{label}.closed_batches[{index - 1}]",
            seen_batch_ids=seen_batch_ids,
        )
    if current_batch.get("batch_id") != batch_id:
        raise ValueError(f"{label}.current_batch.batch_id 与根身份不一致")
    _validate_batch(
        current_batch,
        task_id=task_id,
        batch_index=len(closed_batches) + 1,
        label=f"{label}.current_batch",
        seen_batch_ids=seen_batch_ids,
    )
    return copy.deepcopy(payload)


__all__ = ["validate_investigation_working_memory_payload"]
