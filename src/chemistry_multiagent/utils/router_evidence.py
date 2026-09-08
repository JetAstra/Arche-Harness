"""把 MiniChem 当前 Attempt 的完成结果整理为 Router 可审计证据。

本模块只接受冻结 MiniChem Action/Backend 摘要和三层执行结果。它验证执行身份、
ArtifactRef、open-execution collect 摘要以及 Program Gate 的派生值，然后生成有界
结果和固定顺序的 JSON Pointer 索引。它不读取工作区文件、不解释科学主张，也不
替模型选择工作流修改。
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from chemistry_multiagent.utils.expert_views import (
    build_workflow_scientific_evidence_view,
    flatten_workflow_evidence_index,
)

_ROUTER_DECISIONS = (
    "evidence_ready",
    "revise_workflow",
    "revise_hypothesis",
)

# 结果字段只按形状和体积限制，不查看科学数值大小或结论。
_MAX_RESULT_TEXT_BYTES = 4096
_MAX_RESULT_KEY_BYTES = 1024
_MAX_RESULT_SEQUENCE_ITEMS = 4096
_MAX_RESULT_MAPPING_FIELDS = 256
_MAX_RESULT_DEPTH = 12
_MAX_RESULT_TOTAL_BYTES = 262144
MAX_ROUTER_EVIDENCE_PACKAGE_BYTES = 262144
_MAX_EVIDENCE_INDEX_ITEMS = 4096
_EXCLUDED_PAYLOAD_FIELDS = frozenset(
    {
        "raw_log",
        "full_log",
        "log_text",
        "traceback",
        "stack_trace",
        "stacktrace",
        "exception_trace",
        "stdout",
        "stderr",
        "file_bytes",
        "binary",
        "binary_content",
        "binary_payload",
        "image",
        "image_bytes",
        "image_content",
        "image_data",
    }
)
_FILESYSTEM_PATH_MARKERS = frozenset(
    {
        "file",
        "input",
        "output",
        "log",
        "checkpoint",
        "status",
        "artifact",
        "image",
        "scratch",
        "work",
        "working",
        "directory",
        "dir",
        "manifest",
        "database",
        "sqlite",
        "gjf",
        "sdf",
        "xyz",
        "mol",
        "mol2",
        "pdb",
        "csv",
        "json",
        "yaml",
    }
)
_OMITTED = object()
_FAILURE_ABSOLUTE_PATH_TEXT = re.compile(
    r"(?:(?<![A-Za-z0-9_.])[A-Za-z]:[\\/](?:[^\s,;:'\"(){}\[\]]+)"
    r"|(?<![A-Za-z0-9_.:/])/(?!/)(?:[^\s,;:'\"(){}\[\]]+))"
)
_FOREACH_ABSOLUTE_PATH_TEXT = re.compile(
    r"(?i)(?:file://[^\s,;:'\"(){}\[\]]+"
    r"|(?<![A-Za-z0-9_.:/-])(?:\\\\|//)[^\s,;:'\"(){}\[\]]+"
    r"|(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/][^\s,;:'\"(){}\[\]]+)"
)
_FAILURE_FACT_OMITTED_MARKER = {"omitted": "failure_fact_bounded"}


class _EvidenceIndexLimitExceeded(ValueError):
    """表示证据索引条目超过固定上限。"""


def _fail(message: str) -> None:
    """以 ValueError 终止不满足 Router 契约的输入。"""

    raise ValueError(message)


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    """确认 object 没有缺失字段或模型自行添加的字段。"""

    if not isinstance(value, Mapping):
        _fail(f"{label} 必须是 object")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} 字段不符合固定结构："
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _nonempty_text(value: Any, label: str, *, maximum: int = 2000) -> str:
    """读取 Router 说明文字并拒绝空白或超长值。"""

    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} 必须是非空字符串")
    if len(value) > maximum:
        _fail(f"{label} 超过 {maximum} 字符")
    return value.strip()


def _is_minichem_projection(value: Any) -> bool:
    """识别冻结 MiniChem Action 的短摘要。"""

    return isinstance(value, list) and bool(value) and all(
        isinstance(item, Mapping)
        and isinstance(item.get("action_id"), str)
        and bool(item.get("action_id"))
        for item in value
    )


def _minichem_action_map(
    projection: Any,
) -> dict[str, Mapping[str, Any]]:
    """校验 Action/Backend 摘要并按 action_id 建索引。"""

    if not _is_minichem_projection(projection):
        _fail("MiniChem Action projection 必须是非空数组")
    actions: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(projection):
        action_id = item.get("action_id")
        backend_ids = item.get("backend_ids")
        if action_id in actions or not isinstance(action_id, str) or not action_id:
            _fail(f"MiniChem Action projection[{index}] action_id 非法或重复")
        if not isinstance(backend_ids, list) or not backend_ids:
            _fail(f"MiniChem Action {action_id}.backend_ids 非法")
        if any(not isinstance(backend_id, str) or not backend_id for backend_id in backend_ids):
            _fail(f"MiniChem Action {action_id}.backend_ids 非法")
        if len(backend_ids) != len(set(backend_ids)):
            _fail(f"MiniChem Action {action_id}.backend_ids 非法")
        actions[action_id] = item
    return actions


def _minichem_workflow_maps(
    workflow: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """建立 Router 可读 MiniChem 工作流的连续短引用索引。"""

    if not isinstance(workflow, Mapping) or set(workflow) != {
        "workflow_name",
        "goal",
        "strategy_name",
        "steps",
    }:
        _fail("MiniChem Router workflow 字段不符合简化契约")
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("MiniChem Router workflow.steps 必须是非空 array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            _fail(f"MiniChem workflow.steps[{index}] 必须是 object")
        step_ref = step.get("step_ref")
        if step_ref != f"s{index + 1}" or step_ref in result:
            _fail("MiniChem workflow step_ref 必须连续且唯一")
        layer = step.get("execution_layer", "predefined_actions")
        if layer == "predefined_action":
            layer = "predefined_actions"
        if layer not in {
            "predefined_actions",
            "native_software",
            "programmable_analysis",
        }:
            _fail(f"MiniChem workflow {step_ref}.execution_layer 非法")
        if layer == "predefined_actions":
            if not isinstance(step.get("action_id"), str) or not isinstance(
                step.get("backend_id"), str
            ):
                _fail(f"MiniChem workflow {step_ref} 缺少 Action/Backend 身份")
        elif "action_id" in step or "backend_id" in step:
            _fail(f"MiniChem open workflow {step_ref} 不得携带 Action/Backend 身份")
        result[step_ref] = step
    return result


def _pointer_token(value: str) -> str:
    """把对象键编码成不会破坏 JSON Pointer 层级的片段。"""

    return value.replace("~", "~0").replace("/", "~1")


def _decode_pointer_token(value: str) -> str:
    """解码已经通过格式检查的 JSON Pointer token。"""

    return value.replace("~1", "/").replace("~0", "~")


def _payload_field_name(value: str) -> str:
    """规范化字段拼写，用于识别不能发送给模型的文件内容。"""

    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _excluded_result_reason(field_name: str) -> str | None:
    """识别被禁止嵌入的日志、二进制和绝对路径字段。"""

    normalized = _payload_field_name(field_name)
    if normalized in _EXCLUDED_PAYLOAD_FIELDS:
        return "excluded_payload"
    if normalized in {"path", "paths", "filepath", "filepaths"}:
        return "excluded_path"
    parts = normalized.split("_")
    if (
        parts[-1] in {"path", "paths", "filepath", "filepaths"}
        and any(part in _FILESYSTEM_PATH_MARKERS for part in parts[:-1])
    ):
        return "excluded_path"
    return None


def _json_string_size(value: str, limit: int) -> int:
    """逐字符计算 JSON 字符串大小，超过上限后停止。"""

    size = 2
    short_escapes = {"\b", "\t", "\n", "\f", "\r", '"', "\\"}
    for character in value:
        codepoint = ord(character)
        if character in short_escapes:
            size += 2
        elif codepoint < 0x20:
            size += 6
        else:
            size += len(character.encode("utf-8"))
        if size > limit:
            return limit + 1
    return size


def _utf8_size(value: str, limit: int) -> int:
    """逐字符计算 UTF-8 字节数。"""

    size = 0
    for character in value:
        size += len(character.encode("utf-8"))
        if size > limit:
            return limit + 1
    return size


def _bounded_json_size(value: Any, limit: int, *, depth: int = 0) -> int:
    """计算 canonical JSON 大小，不复制超过上限的内容。"""

    if limit < 0 or depth > 64:
        return limit + 1
    if value is None:
        return 4 if limit >= 4 else limit + 1
    if value is True:
        return 4 if limit >= 4 else limit + 1
    if value is False:
        return 5 if limit >= 5 else limit + 1
    if isinstance(value, str):
        return _json_string_size(value, limit)
    if isinstance(value, int) and not isinstance(value, bool):
        estimated_digits = max(1, int(value.bit_length() * 0.30103) + 1)
        if value < 0:
            estimated_digits += 1
        if estimated_digits > limit:
            return limit + 1
        try:
            size = len(str(value))
        except ValueError:
            return limit + 1
        return size if size <= limit else limit + 1
    if isinstance(value, float):
        if not math.isfinite(value):
            return limit + 1
        size = len(json.dumps(value, allow_nan=False).encode("utf-8"))
        return size if size <= limit else limit + 1
    if isinstance(value, Mapping):
        total = 2
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                return limit + 1
            fixed = (1 if index else 0) + 1
            if total + fixed > limit:
                return limit + 1
            total += fixed
            key_size = _bounded_json_size(key, limit - total, depth=depth + 1)
            if key_size > limit - total:
                return limit + 1
            total += key_size
            child_size = _bounded_json_size(item, limit - total, depth=depth + 1)
            if child_size > limit - total:
                return limit + 1
            total += child_size
        return total
    if isinstance(value, list):
        total = 2
        for index, item in enumerate(value):
            if index:
                if total + 1 > limit:
                    return limit + 1
                total += 1
            child_size = _bounded_json_size(item, limit - total, depth=depth + 1)
            if child_size > limit - total:
                return limit + 1
            total += child_size
        return total
    return limit + 1


def _value_size(value: Any) -> int:
    """返回有界大小；超出单步上限时返回上限加一。"""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return min(len(value), _MAX_RESULT_TOTAL_BYTES + 1)
    return _bounded_json_size(value, _MAX_RESULT_TOTAL_BYTES)


def _omission(
    omissions: list[dict[str, Any]],
    *,
    source_ref: str,
    reason: str,
    value: Any,
) -> None:
    """登记被边界排除的字段及其类型和大小。"""

    if isinstance(value, Mapping):
        value_type = "object"
    elif isinstance(value, list):
        value_type = "array"
    elif isinstance(value, str):
        value_type = "string"
    elif isinstance(value, (bytes, bytearray, memoryview)):
        value_type = "binary"
    elif value is None:
        value_type = "null"
    else:
        value_type = type(value).__name__
    omissions.append(
        {
            "source_ref": source_ref,
            "reason": reason,
            "value_type": value_type,
            "size": _value_size(value),
        }
    )


def _bounded_result_value(
    value: Any,
    *,
    source_ref: str,
    depth: int,
    budget: dict[str, int],
    omissions: list[dict[str, Any]],
) -> Any:
    """复制安全 JSON 值并为每次省略保留固定原因。"""

    if depth > _MAX_RESULT_DEPTH:
        _omission(
            omissions,
            source_ref=source_ref,
            reason="maximum_depth_exceeded",
            value=value,
        )
        return _OMITTED
    if isinstance(value, str):
        encoded_size = _utf8_size(value, _MAX_RESULT_TEXT_BYTES)
        if encoded_size > _MAX_RESULT_TEXT_BYTES:
            _omission(omissions, source_ref=source_ref, reason="oversized_text", value=value)
            return _OMITTED
        if encoded_size > budget["remaining"]:
            _omission(
                omissions,
                source_ref=source_ref,
                reason="projection_budget_exhausted",
                value=value,
            )
            return _OMITTED
        budget["remaining"] -= encoded_size
        return value
    if value is None or isinstance(value, (bool, int, float)):
        size = _value_size(value)
        if size <= 0 or size > budget["remaining"]:
            _omission(
                omissions,
                source_ref=source_ref,
                reason="unsupported_json_value" if size <= 0 else "projection_budget_exhausted",
                value=value,
            )
            return _OMITTED
        budget["remaining"] -= size
        return deepcopy(value)
    if isinstance(value, Mapping):
        if len(value) > _MAX_RESULT_MAPPING_FIELDS:
            _omission(omissions, source_ref=source_ref, reason="oversized_object", value=value)
            return _OMITTED
        if any(not isinstance(key, str) for key in value):
            _omission(
                omissions,
                source_ref=source_ref,
                reason="non_string_object_key",
                value=value,
            )
            return _OMITTED
        projected: dict[str, Any] = {}
        for key_index, key in enumerate(sorted(value)):
            key_size = _utf8_size(key, _MAX_RESULT_KEY_BYTES)
            if key_size > _MAX_RESULT_KEY_BYTES:
                _omission(
                    omissions,
                    source_ref=f"{source_ref}/@oversized-field-{key_index}",
                    reason="oversized_field_name",
                    value=value[key],
                )
                continue
            child_ref = f"{source_ref}/{_pointer_token(key)}"
            excluded_reason = _excluded_result_reason(key)
            if excluded_reason is not None:
                _omission(
                    omissions,
                    source_ref=child_ref,
                    reason=excluded_reason,
                    value=value[key],
                )
                continue
            if key_size > budget["remaining"]:
                _omission(
                    omissions,
                    source_ref=child_ref,
                    reason="projection_budget_exhausted",
                    value=value[key],
                )
                continue
            before = budget["remaining"]
            budget["remaining"] -= key_size
            child = _bounded_result_value(
                value[key],
                source_ref=child_ref,
                depth=depth + 1,
                budget=budget,
                omissions=omissions,
            )
            if child is _OMITTED:
                budget["remaining"] = before
                continue
            projected[key] = child
        return projected
    if isinstance(value, list):
        if len(value) > _MAX_RESULT_SEQUENCE_ITEMS:
            _omission(omissions, source_ref=source_ref, reason="oversized_sequence", value=value)
            return _OMITTED
        before_budget = budget["remaining"]
        before_omissions = len(omissions)
        projected_items = []
        for index, item in enumerate(value):
            child = _bounded_result_value(
                item,
                source_ref=f"{source_ref}/{index}",
                depth=depth + 1,
                budget=budget,
                omissions=omissions,
            )
            if child is _OMITTED:
                budget["remaining"] = before_budget
                del omissions[before_omissions:]
                _omission(
                    omissions,
                    source_ref=source_ref,
                    reason="sequence_contains_excluded_value",
                    value=value,
                )
                return _OMITTED
            projected_items.append(child)
        return projected_items
    _omission(
        omissions,
        source_ref=source_ref,
        reason="excluded_payload" if isinstance(value, (bytes, bytearray, memoryview)) else "unsupported_json_value",
        value=value,
    )
    return _OMITTED


def _bounded_result_projection(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    """对单步 result 应用统一边界，并保留所有未超界语义值。"""

    omissions: list[dict[str, Any]] = []
    projected = _bounded_result_value(
        value,
        source_ref="/result",
        depth=0,
        budget={"remaining": _MAX_RESULT_TOTAL_BYTES},
        omissions=omissions,
    )
    if projected is _OMITTED:
        projected = None
    bounded_pair = {"result": projected, "omitted_result_fields": omissions}
    if _value_size(bounded_pair) > _MAX_RESULT_TOTAL_BYTES:
        omissions = []
        _omission(
            omissions,
            source_ref="/result",
            reason="projection_size_limit",
            value=value,
        )
        projected = None
    return projected, omissions


def _redact_failure_value(value: Any) -> Any:
    """递归复制失败事实，并把错误文本中的机器绝对路径替换为占位符。"""

    if isinstance(value, str):
        return _FAILURE_ABSOLUTE_PATH_TEXT.sub("[路径已隐藏]", value)
    if isinstance(value, Mapping):
        return {
            key: _redact_failure_value(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, list):
        return [_redact_failure_value(item) for item in value]
    return deepcopy(value)


def _failure_fact_is_empty(value: Any) -> bool:
    """判断失败事实是否只剩递归空容器或空白文本。"""

    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return not value or all(_failure_fact_is_empty(item) for item in value.values())
    if isinstance(value, list):
        return not value or all(_failure_fact_is_empty(item) for item in value)
    return False


def _project_minichem_failure_fact(value: Any, label: str) -> Any:
    """把 canonical error/reason 变成有界、脱敏且可复验的 Router 事实。"""

    if value is None:
        return None
    if not isinstance(value, (str, Mapping, list)):
        _fail(f"{label} 必须是非空字符串、object 或 array")
    safe_value = _redact_failure_value(value)
    projected, _omissions = _bounded_result_projection(safe_value)
    if projected is None:
        _fail(f"{label} 超过 Router 失败事实边界")
    if _failure_fact_is_empty(projected):
        # 原始错误只有被排除的 traceback/路径等字段时，仍保留一个明确的、
        # 有界的省略事实；不能让失败步骤看起来像没有 error。
        return deepcopy(_FAILURE_FACT_OMITTED_MARKER)
    return projected


def _execution_step_status(step: Mapping[str, Any]) -> str | None:
    """读取 canonical 或 Router 投影步骤的统一外层状态。"""

    value = step.get("status")
    if isinstance(value, str):
        return value.strip().casefold()
    raw_output = step.get("raw_output")
    if isinstance(raw_output, Mapping) and isinstance(raw_output.get("status"), str):
        return raw_output["status"].strip().casefold()
    return None


def validate_router_execution_terminal(
    execution_result: Mapping[str, Any],
    expected_step_count: int,
) -> tuple[str, ...]:
    """校验 Router 只接收 completed、失败终局或受限完成前缀。

    ``waiting_execution`` 以及任意不完整的失败清单都在这里拒绝。失败终局必须
    是完整工作流，并严格遵循 ``completed -> failed -> not_executed``；后续模型只能
    通过 Program Gate 选择 ``revise_workflow``，不能把失败事实当成科学反证。
    """

    if not isinstance(expected_step_count, int) or expected_step_count < 1:
        _fail("Router workflow 步骤数量非法")
    if not isinstance(execution_result, Mapping):
        _fail("MiniChem execution_result 必须是 object")
    steps = execution_result.get("steps")
    if (
        not isinstance(steps, list)
        or not steps
        or len(steps) > expected_step_count
        or execution_result.get("total_steps") != len(steps)
    ):
        _fail("MiniChem execution_result.steps 不是 workflow 的合法前缀")
    statuses: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            _fail(f"MiniChem execution_result.steps[{index}] 必须是 object")
        status = _execution_step_status(step)
        if status not in {"completed", "failed", "not_executed"}:
            _fail("MiniChem execution_result 包含 waiting_execution 或非法步骤状态")
        statuses.append(status)

    failure_positions = [
        index for index, status in enumerate(statuses) if status == "failed"
    ]
    if failure_positions:
        failure_index = failure_positions[0]
        if (
            len(steps) != expected_step_count
            or execution_result.get("overall_status") != "failed"
            or execution_result.get("workflow_outcome") != "stopped_after_error"
            or statuses[:failure_index] != ["completed"] * failure_index
            or statuses[failure_index] != "failed"
            or any(status != "not_executed" for status in statuses[failure_index + 1 :])
        ):
            _fail("MiniChem 失败终局必须是完整的 completed -> failed -> not_executed 顺序")
        return tuple(statuses)

    if any(status != "completed" for status in statuses):
        _fail("MiniChem 未执行步骤必须紧随一个失败步骤，不能单独进入 Router")
    expected_outcome = (
        "stopped_after_required_check_not_matched"
        if len(steps) < expected_step_count
        else "completed"
    )
    if (
        execution_result.get("overall_status") != "completed"
        or execution_result.get("workflow_outcome") != expected_outcome
    ):
        _fail("MiniChem execution_result.workflow_outcome 与步骤前缀不一致")
    return tuple(statuses)


def _safe_relative_path(value: Any, label: str) -> str:
    """确认 ArtifactRef/collect 路径是任务内相对路径，不读取路径内容。"""

    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{label} 路径非法")
    if value.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", value):
        _fail(f"{label} 路径越过任务根目录")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail(f"{label} 路径不是安全的任务内相对路径")
    return value


def _minichem_artifact_refs(
    raw_output: Mapping[str, Any],
    step_ref: str,
    *,
    require_action_provenance: bool = True,
) -> list[dict[str, Any]]:
    """核对 MiniChem 输出 ArtifactRef 的身份、摘要和来源字段。"""

    values = raw_output.get("output_artifacts")
    if values in (None, []) and isinstance(raw_output.get("artifact"), Mapping):
        values = [raw_output["artifact"]]
    if values is None:
        values = []
    if not isinstance(values, list):
        _fail(f"步骤 {step_ref} 的 output_artifacts 必须是 array")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, artifact in enumerate(values):
        if not isinstance(artifact, Mapping):
            _fail(f"步骤 {step_ref} 的 output_artifacts[{index}] 必须是 object")
        artifact_id = artifact.get("artifact_id")
        digest = artifact.get("sha256")
        semantic_type = artifact.get("semantic_type")
        media_type = artifact.get("media_type")
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or artifact_id in seen
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(semantic_type, str)
            or not semantic_type
            or not isinstance(media_type, str)
            or not media_type
        ):
            _fail(f"步骤 {step_ref} 的 ArtifactRef 身份、类型或 sha256 非法")
        _safe_relative_path(artifact.get("path"), f"步骤 {step_ref} 的 ArtifactRef")
        parent_ids = artifact.get("parent_artifact_ids")
        if not isinstance(parent_ids, list) or any(
            not isinstance(parent_id, str) or not parent_id for parent_id in parent_ids
        ):
            _fail(f"步骤 {step_ref} 的 ArtifactRef parent_artifact_ids 非法")
        if require_action_provenance:
            producer_action = artifact.get("producer_action")
            if not isinstance(producer_action, str) or not producer_action:
                _fail(f"步骤 {step_ref} 的 ArtifactRef 缺少 producer_action")
        producer_backend = artifact.get("producer_backend")
        if producer_backend is not None and (
            not isinstance(producer_backend, str) or not producer_backend
        ):
            _fail(f"步骤 {step_ref} 的 ArtifactRef producer_backend 非法")
        seen.add(artifact_id)
        checked.append(deepcopy(dict(artifact)))
    return checked


# native foreach 的 fanout manifest、child request 和远端作业身份只在程序侧复核。
# Router 可以看到每项的科学结果、来源标签和产物摘要，但不能看到这些运行控制字段。
_FOREACH_PROVENANCE_KEYS = frozenset(
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
_FOREACH_INTERNAL_RESULT_KEYS = frozenset(
    {
        "operation",
        "operation_id",
        "operation_ids",
        "operation_fingerprint",
        "parent_operation_id",
        "request",
        "request_ref",
        "request_fingerprint",
        "child_operation_id",
        "child_result_ref",
        "parent_job_id",
        "job",
        "job_id",
        "job_ids",
        "job_status",
        "foreach",
        "manifest",
        "artifact_manifest",
        "path",
        "workspace_path",
        "job_relative_path",
        "relative_path",
        "target_path",
        "source_step_ref",
        "raw_output",
        "stdout",
        "stderr",
        "files",
        "outputs",
        "declared_artifacts",
        "declared_output_artifacts",
        "output_artifacts",
        "artifact",
    }
)
_FOREACH_SUCCESS_JOB_STATUSES = frozenset({"success", "completed", "ready"})
_FOREACH_MAX_ITEMS = 32


def _project_foreach_scalar(value: Any) -> Any:
    """复制 foreach 公开来源标签中的有限标量。

    输入是候选来源标签的任意值；动作只接受有限整数、有限浮点数和短文本，并拒绝
    路径、请求引用及 operation/job 身份，防止标签携带运行内部信息。函数只在内存中
    处理值，不修改输入或写文件；返回清洗后的标量，``None``、布尔值以外的非法类型、
    非有限数、空文本、超长文本和敏感文本返回 ``None``，不会抛出业务异常。
    """

    if value is None or isinstance(value, bool):
        return deepcopy(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return deepcopy(value) if math.isfinite(value) else None
    if isinstance(value, str) and value.strip() and len(value) <= 256:
        # 来源标签只接受短文本；路径和运行身份文本不能通过标签绕过边界。
        if (
            _FAILURE_ABSOLUTE_PATH_TEXT.search(value)
            or _FOREACH_ABSOLUTE_PATH_TEXT.search(value)
        ):
            return None
        normalized = _payload_field_name(value)
        if any(marker in normalized for marker in ("operation_id", "job_id", "path", "request_ref")):
            return None
        return value.strip()
    return None


def _project_foreach_provenance(item: Mapping[str, Any]) -> dict[str, Any]:
    """只保留 pair/candidate 等科学来源标签，不复制分发清单身份。

    输入是一个 foreach 项映射；动作读取 provenance（来源说明，标识候选与上游步骤
    的关系）或项级同名字段，仅投影允许的来源键，并用有限标量投影器清洗值。函数无
    外部副作用，返回可供 Router 和模型读取的来源标签字典；输入不是映射或字段无效
    时返回空字典或跳过字段，不抛出业务异常。
    """

    raw = item.get("provenance")
    raw = raw if isinstance(raw, Mapping) else {}
    projected: dict[str, Any] = {}
    for key in sorted(_FOREACH_PROVENANCE_KEYS):
        value = raw.get(key, item.get(key))
        if value is None:
            continue
        safe = _project_foreach_scalar(value)
        if safe is not None:
            projected[key] = safe
    return projected


def _strip_foreach_internal_values(value: Any) -> Any:
    """递归删除 child 结果中的运行身份、请求引用和原始日志字段。

    输入是 child 结果中的任意嵌套映射、数组、元组或标量；动作复制可公开结构，删除
    operation/job、请求、fanout（分发清单，记录 child 顺序与远端身份）等内部字段，并
    替换文本中出现的工作区绝对路径。函数不修改输入、不写文件；返回新的清洗值，
    不支持的对象按深拷贝结果处理，不因普通输入问题抛出业务异常。
    """

    if isinstance(value, Mapping):
        return {
            key: _strip_foreach_internal_values(item)
            for key, item in value.items()
            if isinstance(key, str)
            and _payload_field_name(key) not in _FOREACH_INTERNAL_RESULT_KEYS
        }
    if isinstance(value, list):
        return [_strip_foreach_internal_values(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_foreach_internal_values(item) for item in value]
    if isinstance(value, str):
        # 普通科学说明也不能携带 child 工作目录；字段级过滤无法覆盖嵌入正文。
        value = _FAILURE_ABSOLUTE_PATH_TEXT.sub("[路径已隐藏]", value)
        return _FOREACH_ABSOLUTE_PATH_TEXT.sub("[路径已隐藏]", value)
    return deepcopy(value)


def _project_foreach_artifact_summary(
    artifact: Mapping[str, Any],
    step_ref: str,
) -> dict[str, Any]:
    """把 child ArtifactRef 产物引用限制为 Router 可审计的摘要。

    输入是 child 的产物记录和步骤引用；动作调用统一校验器核对产物身份、哈希、父
    产物和生产者字段，移除服务附加的未知成员，并拒绝绝对路径。函数只在内存中构造
    摘要，不修改输入或写文件；返回有限的 ArtifactRef 字典。产物结构、身份、哈希或
    路径不合法时通过 ``_fail`` 抛出 ``ValueError``，不会返回不完整或不安全摘要。
    """

    checked = _minichem_artifact_refs(
        {"output_artifacts": [artifact]},
        step_ref,
        require_action_provenance=False,
    )
    if len(checked) != 1:
        _fail(f"步骤 {step_ref} 的 foreach ArtifactRef 数量非法")
    for field in (
        "artifact_id",
        "semantic_type",
        "media_type",
        "sha256",
        "path",
        "producer_action",
        "producer_backend",
    ):
        value = checked[0].get(field)
        if isinstance(value, str) and (
            _FAILURE_ABSOLUTE_PATH_TEXT.search(value)
            or _FOREACH_ABSOLUTE_PATH_TEXT.search(value)
        ):
            _fail(f"步骤 {step_ref} 的 foreach ArtifactRef 含绝对路径")
    parent_ids = checked[0].get("parent_artifact_ids", [])
    if any(
        isinstance(value, str)
        and (
            _FAILURE_ABSOLUTE_PATH_TEXT.search(value)
            or _FOREACH_ABSOLUTE_PATH_TEXT.search(value)
        )
        for value in parent_ids
    ):
        _fail(f"步骤 {step_ref} 的 foreach ArtifactRef parent 身份含绝对路径")
    # 保留既有 Router ArtifactRef 的摘要字段，忽略服务附加的未知成员。
    return {
        key: deepcopy(checked[0][key])
        for key in (
            "artifact_id",
            "semantic_type",
            "media_type",
            "sha256",
            "path",
            "parent_artifact_ids",
            "producer_action",
            "producer_backend",
        )
        if key in checked[0]
    }


def _project_foreach_file_summary(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    """保留 child collect 文件的相对路径、哈希、大小和类型摘要。

    输入是文件记录和用于错误定位的字段标签；动作核对相对路径、64 位十六进制
    SHA-256 和非负字节数，只复制可审计摘要，不读取或复制文件正文。函数无外部副作用，
    返回文件摘要字典；路径、哈希或大小格式错误时通过 ``_fail`` 抛出 ``ValueError``，
    不返回伪造记录。
    """

    path = record.get("path", record.get("job_relative_path"))
    _safe_relative_path(path, label)
    digest = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        _fail(f"{label} 文件摘要非法")
    projected = {"path": path, "sha256": digest, "size_bytes": size_bytes}
    kind = record.get("kind")
    if isinstance(kind, str) and kind.strip():
        projected["kind"] = kind.strip()
    return projected


def _project_foreach_declared_summary(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    """保留 child 声明产物的类型和校验状态，不复制工作区身份。

    输入是声明产物记录和用于错误定位的字段标签；动作核对相对路径，复制名称、语义
    类型、媒体类型、生产者及验证状态，并把有限的失败事实投影到摘要。函数只在内存
    中构造新字典，不读取工作区或写文件；返回声明摘要。路径非法时通过 ``_fail`` 抛出
    ``ValueError``，可选字段缺失或类型不符时省略，不抛出其他业务异常。
    """

    path = record.get("path", record.get("workspace_path"))
    _safe_relative_path(path, label)
    projected: dict[str, Any] = {"path": path}
    for key in (
        "name",
        "semantic_type",
        "media_type",
        "producer",
        "validation_status",
        "validation_errors",
        "errors",
    ):
        value = record.get(key)
        if key in {"validation_errors", "errors"}:
            if isinstance(value, list):
                projected[key] = _project_minichem_failure_fact(value, label)
            continue
        if isinstance(value, str) and value.strip():
            projected[key] = value.strip()
    return projected


def _project_native_foreach_child_result(
    child_raw: Mapping[str, Any],
    *,
    step_ref: str,
    item_index: int,
    budget: dict[str, int],
    omissions: list[dict[str, Any]],
) -> dict[str, Any]:
    """投影一个 child 的科学值和产物摘要，过滤运行控制字段。

    输入是 child 原始结果、步骤引用、child 索引、剩余字节预算和 omission（省略记录，
    说明哪些值因边界被移除）列表；动作删除运行身份，请求和日志字段，按预算复制有界
    科学结果，并投影 ArtifactRef、collect 文件和声明产物摘要。函数无文件副作用，但
    会消耗 ``budget`` 并向 ``omissions`` 追加省略原因；返回 child 公开结果字典。child
    产物或记录结构非法时通过 ``_fail`` 抛出 ``ValueError``，不返回部分可信结果。
    """

    projected: dict[str, Any] = {}
    for key in sorted(child_raw):
        if not isinstance(key, str):
            continue
        if _payload_field_name(key) in _FOREACH_INTERNAL_RESULT_KEYS:
            continue
        value = child_raw[key]
        child_ref = f"/result/items/{item_index}/result/{_pointer_token(key)}"
        before = budget["remaining"]
        key_size = _utf8_size(key, _MAX_RESULT_KEY_BYTES)
        if key_size > _MAX_RESULT_KEY_BYTES or key_size > budget["remaining"]:
            _omission(
                omissions,
                source_ref=child_ref,
                reason="projection_budget_exhausted",
                value=value,
            )
            continue
        budget["remaining"] -= key_size
        safe = _bounded_result_value(
            _strip_foreach_internal_values(value),
            source_ref=child_ref,
            depth=0,
            budget=budget,
            omissions=omissions,
        )
        if safe is not _OMITTED:
            projected[key] = safe
        else:
            budget["remaining"] = before

    artifacts = child_raw.get("output_artifacts")
    if artifacts in (None, []) and isinstance(child_raw.get("artifact"), Mapping):
        artifacts = [child_raw["artifact"]]
    if artifacts is not None:
        if not isinstance(artifacts, list):
            _fail(f"步骤 {step_ref}.foreach.items[{item_index}] 的 output_artifacts 非法")
        projected["output_artifacts"] = [
            _project_foreach_artifact_summary(
                artifact,
                f"步骤 {step_ref}.foreach.items[{item_index}].output_artifacts[{artifact_index}]",
            )
            for artifact_index, artifact in enumerate(artifacts)
            if isinstance(artifact, Mapping)
        ]
        if len(projected["output_artifacts"]) != len(artifacts):
            _fail(f"步骤 {step_ref}.foreach child ArtifactRef 非法")

    for field_name, projector in (
        ("files", _project_foreach_file_summary),
        ("outputs", _project_foreach_file_summary),
        ("declared_artifacts", _project_foreach_declared_summary),
    ):
        values = child_raw.get(field_name)
        if values is None:
            continue
        if not isinstance(values, list):
            _fail(f"步骤 {step_ref}.foreach.items[{item_index}].{field_name} 必须是 array")
        projected[field_name] = [
            projector(
                record,
                f"步骤 {step_ref}.foreach.items[{item_index}].{field_name}[{record_index}]",
            )
            for record_index, record in enumerate(values)
            if isinstance(record, Mapping)
        ]
        if len(projected[field_name]) != len(values):
            _fail(f"步骤 {step_ref}.foreach child {field_name} 记录非法")
    return projected


def _project_native_foreach_result(
    raw_output: Mapping[str, Any],
    step_ref: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """校验 canonical native foreach，并返回不含内部清单的有界公开结果。

    输入是 native foreach 的 canonical 原始结果和步骤引用；动作核对 fanout（分发清单，
    保存 child 顺序、operation/job 身份和结果状态）、每项 provenance（来源说明）及
    child 收据，逐项调用结果投影器，并汇总父级 ArtifactRef（产物引用，记录可核对的产物身份）摘要。函数只在内存中
    读取和构造值，不修改原始结果或写文件；返回 ``(result, omissions, artifacts)``，
    分别是结果、预算省略记录和父级产物摘要。清单、身份、状态、来源、产物或数量不合
    法时通过 ``_fail`` 抛出 ``ValueError``，不会返回部分结果。
    """

    fanout = raw_output.get("foreach")
    result = raw_output.get("result")
    items = fanout.get("items") if isinstance(fanout, Mapping) else None
    result_items = result.get("items") if isinstance(result, Mapping) else None
    job_ids = raw_output.get("job_ids")
    operation_ids = raw_output.get("operation_ids")
    parent_operation = raw_output.get("operation_id")
    parent_job = raw_output.get("job")
    if (
        not isinstance(fanout, Mapping)
        or fanout.get("schema_version") != 1
        or fanout.get("execution_mode") != "native_foreach"
        or not isinstance(items, list)
        or not items
        or len(items) > _FOREACH_MAX_ITEMS
        or not isinstance(result, Mapping)
        or not isinstance(result_items, list)
        or len(items) != len(result_items)
        or not isinstance(job_ids, list)
        or len(job_ids) != len(items)
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"job_[0-9a-f]{32}", item) is None
            for item in job_ids
        )
        or len(set(job_ids)) != len(job_ids)
        or not isinstance(operation_ids, list)
        or len(operation_ids) != len(items)
        or any(not isinstance(item, str) or not item for item in operation_ids)
        or len(set(operation_ids)) != len(operation_ids)
        or raw_output.get("job_id") is not None
        or not isinstance(parent_operation, str)
        or not parent_operation
        or fanout.get("parent_operation_id") != parent_operation
        or (isinstance(parent_job, Mapping) and parent_job.get("job_id") is not None)
        or (parent_job is not None and not isinstance(parent_job, Mapping))
        or result.get("item_count") != len(items)
        or result.get("failed_item_count") != 0
    ):
        _fail(f"步骤 {step_ref} 的 native foreach canonical 身份或数量非法")

    projected_items: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    budget = {"remaining": _MAX_RESULT_TOTAL_BYTES}
    seen_item_ids: set[str] = set()
    seen_child_operations: set[str] = set()
    seen_child_jobs: set[str] = set()
    for item_index, (manifest_item, result_item) in enumerate(zip(items, result_items)):
        if not isinstance(manifest_item, Mapping) or not isinstance(result_item, Mapping):
            _fail(f"步骤 {step_ref}.foreach.items[{item_index}] 必须是 object")
        item_id = manifest_item.get("item_id")
        child_operation = manifest_item.get("child_operation_id")
        child_job = manifest_item.get("job_id")
        child_raw = result_item.get("result")
        manifest_provenance = manifest_item.get("provenance")
        result_provenance = result_item.get("provenance")
        child_job_record = child_raw.get("job") if isinstance(child_raw, Mapping) else None
        child_job_status = child_raw.get("job_status") if isinstance(child_raw, Mapping) else None
        if not isinstance(child_job_status, str) and isinstance(child_job_record, Mapping):
            child_job_status = child_job_record.get("status")
        if (
            manifest_item.get("index") != item_index
            or manifest_item.get("status") != "completed"
            or not isinstance(item_id, str)
            or not item_id
            or item_id in seen_item_ids
            or _FAILURE_ABSOLUTE_PATH_TEXT.search(item_id)
            or _FOREACH_ABSOLUTE_PATH_TEXT.search(item_id)
            or result_item.get("item_id") != item_id
            or not isinstance(child_operation, str)
            or not child_operation
            or child_operation in seen_child_operations
            or not isinstance(child_job, str)
            or re.fullmatch(r"job_[0-9a-f]{32}", child_job) is None
            or child_job in seen_child_jobs
            or job_ids[item_index] != child_job
            or operation_ids[item_index] != child_operation
            or not isinstance(child_raw, Mapping)
            or child_raw.get("status") != "completed"
            or child_raw.get("ready") is not True
            or child_job_status not in _FOREACH_SUCCESS_JOB_STATUSES
            or (child_job_record is not None and not isinstance(child_job_record, Mapping))
            or (
                isinstance(child_job_record, Mapping)
                and child_job_record.get("job_id") not in {None, child_job}
            )
            or child_raw.get("operation_id") != child_operation
            or child_raw.get("job_id") != child_job
            or child_operation == parent_operation
            or not isinstance(manifest_provenance, Mapping)
            or not isinstance(result_provenance, Mapping)
            or dict(manifest_provenance) != dict(result_provenance)
        ):
            _fail(f"步骤 {step_ref}.foreach child 身份或状态非法")
        if "parent_operation_id" in manifest_item and manifest_item.get(
            "parent_operation_id"
        ) != parent_operation:
            _fail(f"步骤 {step_ref}.foreach child parent operation 不一致")
        manifest_child_raw = manifest_item.get("result")
        if manifest_child_raw is not None:
            if not isinstance(manifest_child_raw, Mapping) or any(
                manifest_child_raw.get(field) != child_raw.get(field)
                for field in ("status", "ready", "operation_id", "job_id")
            ):
                _fail(f"步骤 {step_ref}.foreach manifest/result child identity 不一致")
        if "item_id" in child_raw and child_raw.get("item_id") != item_id:
            _fail(f"步骤 {step_ref}.foreach child item identity 不一致")
        seen_item_ids.add(item_id)
        seen_child_operations.add(child_operation)
        seen_child_jobs.add(child_job)
        projected_child = _project_native_foreach_child_result(
            child_raw,
            step_ref=step_ref,
            item_index=item_index,
            budget=budget,
            omissions=omissions,
        )
        projected_items.append(
            {
                "item_id": item_id,
                "provenance": _project_foreach_provenance(result_item),
                "result": projected_child,
            }
        )
    projected = {
        "items": projected_items,
        "item_count": len(projected_items),
        "failed_item_count": 0,
    }
    artifacts = raw_output.get("output_artifacts", [])
    if artifacts is None:
        artifacts = []
    if not isinstance(artifacts, list):
        _fail(f"步骤 {step_ref} 的 foreach output_artifacts 必须是 array")
    artifact_summaries = [
        _project_foreach_artifact_summary(artifact, f"步骤 {step_ref}.output_artifacts[{index}]")
        for index, artifact in enumerate(artifacts)
        if isinstance(artifact, Mapping)
    ]
    if len(artifact_summaries) != len(artifacts):
        _fail(f"步骤 {step_ref} 的 foreach output_artifacts 非法")
    return projected, omissions, artifact_summaries


def _is_public_native_foreach_output(raw_output: Mapping[str, Any]) -> bool:
    """识别是否为已去除内部分发清单的 native foreach Router 投影。

    输入是待检查的公开结果映射；动作只检查状态、结果项外壳和必要类型，不读取文件、
    不修改输入，也不执行完整字段校验。返回布尔值：结构符合公开投影约束时为
    ``True``，否则为 ``False``；普通类型错误不会抛出业务异常。
    """

    result = raw_output.get("result")
    items = result.get("items") if isinstance(result, Mapping) else None
    return (
        raw_output.get("status") == "completed"
        and raw_output.get("operation_id") is None
        and raw_output.get("job_id") is None
        and raw_output.get("job") is None
        and raw_output.get("job_status") is None
        and raw_output.get("ready") is True
        and isinstance(items, list)
        and bool(items)
        and all(
            isinstance(item, Mapping)
            and set(item) == {"item_id", "provenance", "result"}
            and isinstance(item.get("item_id"), str)
            and isinstance(item.get("provenance"), Mapping)
            and isinstance(item.get("result"), Mapping)
            for item in items
        )
    )


def _validate_public_native_foreach_output(
    raw_output: Mapping[str, Any],
    step_ref: str,
) -> None:
    """校验 Router 公开 foreach 结果没有重新带回运行身份。

    输入是公开结果映射和步骤引用；动作核对 bounded（有界，限制字段和内容规模）
    结果外壳、项级文件与 ArtifactRef（产物引用，记录可核对的产物身份）摘要，确认 operation/job、扁平 child 文件和
    内部清单没有重新暴露。函数只读输入，不写文件或修改状态；成功无返回值。公开
    结构、数量、字段或摘要不符合协议时通过 ``_fail`` 抛出 ``ValueError``，阻止结果
    继续进入 Router。
    """

    if not _is_public_native_foreach_output(raw_output):
        _fail(f"步骤 {step_ref} 的 native foreach Router 投影身份非法")
    result = raw_output["result"]
    if set(result) != {"items", "item_count", "failed_item_count"}:
        _fail(f"步骤 {step_ref} 的 native foreach Router result 字段非法")
    if result["item_count"] != len(result["items"]) or result["failed_item_count"] != 0:
        _fail(f"步骤 {step_ref} 的 native foreach Router result 数量非法")
    if raw_output.get("outputs") != [] or raw_output.get("files") != []:
        _fail(f"步骤 {step_ref} 的 native foreach Router collect 不得复制扁平 child 文件")
    if raw_output.get("declared_artifacts") != [] or raw_output.get("artifact_manifest") is not None:
        _fail(f"步骤 {step_ref} 的 native foreach Router collect 身份非法")
    _minichem_artifact_refs(raw_output, step_ref, require_action_provenance=False)
    for item_index, item in enumerate(result["items"]):
        child = item["result"]
        forbidden = set(child) & (
            _FOREACH_INTERNAL_RESULT_KEYS
            - {"files", "outputs", "declared_artifacts", "output_artifacts"}
        )
        if forbidden:
            _fail(f"步骤 {step_ref}.foreach.items[{item_index}] 含内部运行字段")
        for field_name, projector in (
            ("files", _project_foreach_file_summary),
            ("outputs", _project_foreach_file_summary),
            ("declared_artifacts", _project_foreach_declared_summary),
        ):
            values = child.get(field_name, [])
            if not isinstance(values, list):
                _fail(f"步骤 {step_ref}.foreach.items[{item_index}].{field_name} 非法")
            for record_index, record in enumerate(values):
                if not isinstance(record, Mapping):
                    _fail(f"步骤 {step_ref}.foreach.items[{item_index}].{field_name}[{record_index}] 非法")
                projector(record, f"步骤 {step_ref}.foreach.items[{item_index}].{field_name}[{record_index}]")
        artifacts = child.get("output_artifacts", [])
        if not isinstance(artifacts, list):
            _fail(f"步骤 {step_ref}.foreach.items[{item_index}].output_artifacts 非法")
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                _fail(f"步骤 {step_ref}.foreach.items[{item_index}].output_artifacts[{artifact_index}] 非法")
            _project_foreach_artifact_summary(
                artifact,
                f"步骤 {step_ref}.foreach.items[{item_index}].output_artifacts[{artifact_index}]",
            )


def _validate_minichem_open_collect(
    raw_output: Mapping[str, Any],
    step_ref: str,
) -> dict[str, Any]:
    """验证 open job 的 collect 文件、声明记录和终态身份。"""

    if raw_output.get("ready") is not True:
        _fail(f"步骤 {step_ref} 的 open job collect 必须 ready=true")
    job_id = raw_output.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"job_[0-9a-f]{32}", job_id) is None:
        _fail(f"步骤 {step_ref} 的 open job 缺少合法 job_id")
    job_status = raw_output.get("job_status")
    job = raw_output.get("job")
    if not isinstance(job_status, str) and isinstance(job, Mapping):
        job_status = job.get("status")
    if job_status not in {"success", "completed", "ready"}:
        _fail(f"步骤 {step_ref} 的 open job 不是成功终态")
    operation_id = raw_output.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        _fail(f"步骤 {step_ref} 缺少合法 operation_id")
    if job is not None and not isinstance(job, Mapping):
        _fail(f"步骤 {step_ref} 的 open job 字段必须是 object")
    if isinstance(job, Mapping) and job.get("job_id") not in {None, job_id}:
        _fail(f"步骤 {step_ref} 的 open job_id 与 collect 身份不一致")
    files = raw_output.get("files", [])
    outputs = raw_output.get("outputs", [])
    declared = raw_output.get("declared_artifacts", [])
    if not isinstance(files, list) or not isinstance(outputs, list) or not isinstance(declared, list):
        _fail(f"步骤 {step_ref} 的 open collect 文件/声明必须是数组")
    if not files and not outputs and not declared:
        _fail(f"步骤 {step_ref} 的 open collect 没有可审计文件或声明记录")
    for field_name, records in (("files", files), ("outputs", outputs)):
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                _fail(f"步骤 {step_ref}.{field_name}[{index}] 必须是 object")
            _safe_relative_path(
                record.get("path", record.get("job_relative_path")),
                f"步骤 {step_ref}.{field_name}[{index}]",
            )
            digest = record.get("sha256")
            size_bytes = record.get("size_bytes")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                _fail(f"步骤 {step_ref}.{field_name}[{index}] 文件摘要非法")
    for index, record in enumerate(declared):
        if not isinstance(record, Mapping):
            _fail(f"步骤 {step_ref}.declared_artifacts[{index}] 必须是 object")
        _safe_relative_path(
            record.get("path", record.get("workspace_path")),
            f"步骤 {step_ref}.declared_artifacts[{index}]",
        )
    manifest = raw_output.get("artifact_manifest")
    if manifest is not None:
        _safe_relative_path(manifest, f"步骤 {step_ref}.artifact_manifest")
    provenance = raw_output.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        _fail(f"步骤 {step_ref}.provenance 必须是 object")
    if isinstance(provenance, Mapping) and provenance.get("catalog_hash") is not None:
        if re.fullmatch(r"[0-9a-f]{64}", str(provenance["catalog_hash"])) is None:
            _fail(f"步骤 {step_ref} provenance.catalog_hash 非法")
    return {
        "operation_id": operation_id,
        "job_id": job_id,
        "job_status": job_status,
        "job": deepcopy(dict(job)) if isinstance(job, Mapping) else None,
        "ready": True,
        "outputs": deepcopy(outputs),
        "files": deepcopy(files),
        "declared_artifacts": deepcopy(declared),
        "artifact_manifest": manifest,
    }


def _build_minichem_router_execution_projection(
    execution_result: Mapping[str, Any],
    step_refs: Mapping[str, str],
    action_projection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """把三层 MiniChem 完成结果整理为 Router 可读取的步骤证据。"""

    actions = _minichem_action_map(action_projection)
    if not isinstance(execution_result, Mapping):
        _fail("execution_result 必须是 object")
    if not isinstance(step_refs, Mapping) or not step_refs:
        _fail("MiniChem workflow step_refs 必须是非空 object")
    expected_short_refs = [f"s{index}" for index in range(1, len(step_refs) + 1)]
    if list(step_refs) != expected_short_refs or any(
        not isinstance(formal, str) or not formal for formal in step_refs.values()
    ):
        _fail("MiniChem workflow step_refs 必须从 s1 连续排列")
    for field in ("total_steps", "overall_status", "workflow_outcome"):
        if field not in execution_result:
            _fail(f"ExecutionResult 缺少 {field}")
    steps = execution_result.get("steps")
    statuses = validate_router_execution_terminal(execution_result, len(step_refs))
    formal_to_short = {formal: short for short, formal in step_refs.items()}
    expected_formal_steps = list(step_refs.values())
    projected_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("step_id") != expected_formal_steps[index]:
            _fail("MiniChem ExecutionResult 步骤不是当前 workflow 的连续前缀")
        short_ref = formal_to_short.get(step.get("step_id"))
        if short_ref is None:
            _fail("MiniChem ExecutionResult step_id 不属于当前 workflow")
        layer = step.get("execution_layer", "predefined_actions")
        if layer == "predefined_action":
            layer = "predefined_actions"
        if layer not in {"predefined_actions", "native_software", "programmable_analysis"}:
            _fail(f"步骤 {short_ref} 的 execution_layer 非法")
        action_id = step.get("action_id")
        backend_id = step.get("backend_id")
        action = actions.get(action_id) if layer == "predefined_actions" else None
        if layer == "predefined_actions":
            if action is None or backend_id not in action["backend_ids"]:
                _fail(f"步骤 {short_ref} 的 Action/Backend 身份未由目录声明")
        elif action_id is not None or backend_id is not None:
            _fail(f"步骤 {short_ref} open job 不得携带 Action/Backend 身份")
        status = statuses[index]
        raw_output = step.get("raw_output")
        if not isinstance(raw_output, Mapping):
            _fail(f"步骤 {short_ref} 缺少 MiniChem 结果")
        raw_status = raw_output.get("status")
        if status == "completed" and raw_status != "completed":
            _fail(f"步骤 {short_ref} 的 completed 外层状态缺少 completed 结果")
        artifacts = _minichem_artifact_refs(
            raw_output,
            short_ref,
            require_action_provenance=layer == "predefined_actions",
        )
        if layer == "predefined_actions":
            for artifact in artifacts:
                if artifact.get("producer_action") != action_id:
                    _fail(f"步骤 {short_ref} 的 ArtifactRef producer_action 与 Action 不一致")
                producer_backend = artifact.get("producer_backend")
                if producer_backend is not None and producer_backend != backend_id:
                    _fail(f"步骤 {short_ref} 的 ArtifactRef producer_backend 与 Backend 不一致")
        provenance = raw_output.get("provenance")
        if provenance is not None and not isinstance(provenance, Mapping):
            _fail(f"步骤 {short_ref} provenance 必须是 object")
        if isinstance(provenance, Mapping) and provenance.get("catalog_hash") is not None:
            if re.fullmatch(r"[0-9a-f]{64}", str(provenance["catalog_hash"])) is None:
                _fail(f"步骤 {short_ref} provenance.catalog_hash 非法")
        foreach_projection: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None = None
        if (
            status == "completed"
            and layer == "native_software"
            and isinstance(raw_output.get("foreach"), Mapping)
        ):
            foreach_projection = _project_native_foreach_result(raw_output, short_ref)
            result, omissions, foreach_artifacts = foreach_projection
            # ArtifactRef 仍由统一校验器核对；child manifest 本身不进入 Router。
            artifacts = [deepcopy(item) for item in foreach_artifacts]
        else:
            result, omissions = _bounded_result_projection(raw_output.get("result"))
        common_output = {
            "status": status,
            "result": result,
            "output_artifacts": artifacts,
            "provenance": (
                _project_minichem_failure_fact(provenance, f"步骤 {short_ref}.provenance")
                if isinstance(provenance, Mapping)
                else None
            ),
            "omitted_result_fields": omissions,
        }
        if status == "failed":
            failure = step.get("error", raw_output.get("error"))
            if failure is None:
                _fail(f"步骤 {short_ref} 缺少 canonical error 事实")
            common_output["error"] = _project_minichem_failure_fact(
                failure, f"步骤 {short_ref}.error"
            )
        elif status == "not_executed":
            reason = step.get("reason", raw_output.get("reason"))
            if not isinstance(reason, str) or not reason.strip():
                _fail(f"步骤 {short_ref} 缺少 canonical not_executed reason")
            common_output["reason"] = _project_minichem_failure_fact(
                reason.strip(), f"步骤 {short_ref}.reason"
            )
        if layer == "predefined_actions":
            projected_steps.append(
                {
                    "step_ref": short_ref,
                    "action_id": action_id,
                    "backend_id": backend_id,
                    "raw_output": common_output,
                }
            )
            continue
        if status == "completed" and foreach_projection is not None:
            # native foreach 的父结果没有单一远端 job；每个 child 的身份和 collect
            # 已在 Program Gate 复核，Router 只接收上面构造的 bounded item 摘要。
            collect = {
                "operation_id": None,
                "job_id": None,
                "job_status": None,
                "job": None,
                "ready": True,
                "outputs": [],
                "files": [],
                "declared_artifacts": [],
                "artifact_manifest": None,
            }
            safe_provenance = _project_foreach_provenance(raw_output)
            common_output["provenance"] = safe_provenance or None
        elif status == "completed":
            collect = _validate_minichem_open_collect(raw_output, short_ref)
        else:
            # 失败和未执行步骤不能把本机 collect/job 字段带入模型；canonical
            # error/reason 已在上方以 bounded 事实保留。
            collect = {
                "operation_id": None,
                "job_id": None,
                "job_status": None,
                "job": None,
                "ready": False,
                "outputs": [],
                "files": [],
                "declared_artifacts": [],
                "artifact_manifest": None,
            }
        identity_field = "software_id" if layer == "native_software" else "runtime"
        identity_value = step.get(identity_field) or raw_output.get(identity_field)
        if not isinstance(identity_value, str) or not identity_value:
            _fail(f"步骤 {short_ref} 缺少合法 {identity_field}")
        other_identity = "runtime" if identity_field == "software_id" else "software_id"
        if step.get(other_identity) is not None or raw_output.get(other_identity) is not None:
            _fail(f"步骤 {short_ref} open job 不得携带 {other_identity} 身份")
        common_output.update(
            {
                "operation_id": collect["operation_id"],
                "job_id": collect["job_id"],
                "job_status": collect["job_status"],
                "job": collect["job"],
                "ready": collect["ready"],
                "outputs": collect["outputs"],
                "files": collect["files"],
                "declared_artifacts": collect["declared_artifacts"],
                "artifact_manifest": collect["artifact_manifest"],
            }
        )
        projected_steps.append(
            {
                "step_ref": short_ref,
                "execution_layer": layer,
                identity_field: identity_value,
                "raw_output": common_output,
            }
        )
    return {
        "total_steps": len(projected_steps),
        "steps": projected_steps,
        "overall_status": deepcopy(execution_result["overall_status"]),
        "workflow_outcome": deepcopy(execution_result["workflow_outcome"]),
    }


def build_router_execution_projection(
    execution_result: Mapping[str, Any],
    step_refs: Mapping[str, str],
    tool_catalog_projection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """把真实 MiniChem ExecutionResult 转成 Router 有界步骤证据。"""

    if not _is_minichem_projection(tool_catalog_projection):
        _fail("Router execution projection 只接受 MiniChem Action projection")
    return _build_minichem_router_execution_projection(
        execution_result,
        step_refs,
        tool_catalog_projection,
    )


def _validate_router_result_omissions(
    raw_output: Mapping[str, Any],
    step_ref: str,
) -> None:
    """核对结果省略记录的固定字段和来源引用。"""

    omissions = raw_output.get("omitted_result_fields")
    if not isinstance(omissions, list):
        _fail(f"步骤 {step_ref} 的 omitted_result_fields 必须是 array")
    refs: set[str] = set()
    for index, omission in enumerate(omissions):
        if not isinstance(omission, Mapping) or set(omission) != {
            "source_ref",
            "reason",
            "value_type",
            "size",
        }:
            _fail(f"步骤 {step_ref} 的结果省略说明[{index}] 非法")
        source_ref = omission.get("source_ref")
        size = omission.get("size")
        if (
            not isinstance(source_ref, str)
            or not source_ref.startswith("/result")
            or source_ref in refs
            or not isinstance(omission.get("reason"), str)
            or not omission["reason"]
            or not isinstance(omission.get("value_type"), str)
            or not omission["value_type"]
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            _fail(f"步骤 {step_ref} 的结果省略说明非法或重复")
        refs.add(source_ref)


def _validate_minichem_projected_step(
    step: Mapping[str, Any],
    workflow_step: Mapping[str, Any],
    actions: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """核对 Router 输入中单个 MiniChem 步骤的身份和输出字段。"""

    step_ref = workflow_step["step_ref"]
    layer = workflow_step.get("execution_layer", "predefined_actions")
    if layer == "predefined_action":
        layer = "predefined_actions"
    if layer == "predefined_actions":
        expected_keys = {"step_ref", "action_id", "backend_id", "raw_output"}
        if set(step) != expected_keys:
            _fail(f"步骤 {step_ref} 的 MiniChem Action 投影字段非法")
        action_id = workflow_step.get("action_id")
        backend_id = workflow_step.get("backend_id")
        if step.get("action_id") != action_id or step.get("backend_id") != backend_id:
            _fail(f"步骤 {step_ref} 的 Action/Backend 与 workflow 不一致")
        action = actions.get(action_id)
        if action is None or backend_id not in action["backend_ids"]:
            _fail(f"步骤 {step_ref} 的 Action/Backend 未由目录声明")
        expected_raw_keys = {
            "status",
            "result",
            "output_artifacts",
            "provenance",
            "omitted_result_fields",
        }
    else:
        identity_field = "software_id" if layer == "native_software" else "runtime"
        expected_keys = {"step_ref", "execution_layer", identity_field, "raw_output"}
        if set(step) != expected_keys or step.get("execution_layer") != layer:
            _fail(f"步骤 {step_ref} 的 open execution 身份字段非法")
        request = workflow_step.get("request")
        expected_identity = request.get(identity_field) if isinstance(request, Mapping) else None
        if step.get(identity_field) != expected_identity:
            _fail(f"步骤 {step_ref} 的 open execution 身份与 workflow 不一致")
        expected_raw_keys = {
            "status",
            "result",
            "output_artifacts",
            "provenance",
            "omitted_result_fields",
            "operation_id",
            "job_id",
            "job_status",
            "job",
            "ready",
            "outputs",
            "files",
            "declared_artifacts",
            "artifact_manifest",
        }
    raw_output = step.get("raw_output")
    if not isinstance(raw_output, Mapping):
        _fail(f"步骤 {step_ref} 的 MiniChem 结果投影字段非法")
    status = raw_output.get("status")
    if status not in {"completed", "failed", "not_executed"}:
        _fail(f"步骤 {step_ref} 的 MiniChem 结果状态非法")
    allowed_raw_keys = set(expected_raw_keys)
    if status == "failed":
        allowed_raw_keys.add("error")
    elif status == "not_executed":
        allowed_raw_keys.add("reason")
    if set(raw_output) != allowed_raw_keys:
        _fail(f"步骤 {step_ref} 的 MiniChem 结果投影字段非法")
    if status == "failed":
        if raw_output.get("error") is None:
            _fail(f"步骤 {step_ref} 缺少 canonical error 事实")
        if raw_output.get("error") != _project_minichem_failure_fact(
            raw_output.get("error"), f"步骤 {step_ref}.error"
        ):
            _fail(f"步骤 {step_ref}.error 未经过 bounded 脱敏投影")
    elif status == "not_executed":
        if not isinstance(raw_output.get("reason"), str) or not raw_output["reason"].strip():
            _fail(f"步骤 {step_ref} 缺少 canonical not_executed reason")
        if raw_output.get("reason") != _project_minichem_failure_fact(
            raw_output.get("reason"), f"步骤 {step_ref}.reason"
        ):
            _fail(f"步骤 {step_ref}.reason 未经过 bounded 脱敏投影")
    elif raw_output.get("status") != "completed":
        _fail(f"步骤 {step_ref} 缺少 completed MiniChem 结果")
    if layer != "predefined_actions" and status != "completed":
        expected_empty_collect = {
            "operation_id": None,
            "job_id": None,
            "job_status": None,
            "job": None,
            "ready": False,
            "outputs": [],
            "files": [],
            "declared_artifacts": [],
            "artifact_manifest": None,
        }
        if any(raw_output.get(key) != value for key, value in expected_empty_collect.items()):
            _fail(f"步骤 {step_ref} 失败或未执行时 open execution 的 collect 必须为空")
    artifacts = _minichem_artifact_refs(
        raw_output,
        step_ref,
        require_action_provenance=layer == "predefined_actions",
    )
    if layer == "predefined_actions":
        for artifact in artifacts:
            if artifact.get("producer_action") != step.get("action_id"):
                _fail(f"步骤 {step_ref} 的 ArtifactRef producer_action 与 Action 不一致")
            producer_backend = artifact.get("producer_backend")
            if producer_backend is not None and producer_backend != step.get("backend_id"):
                _fail(f"步骤 {step_ref} 的 ArtifactRef producer_backend 与 Backend 不一致")
    if layer != "predefined_actions" and status == "completed":
        if layer == "native_software" and _is_public_native_foreach_output(raw_output):
            _validate_public_native_foreach_output(raw_output, step_ref)
        else:
            _validate_minichem_open_collect(raw_output, step_ref)
    _validate_router_result_omissions(raw_output, step_ref)
    provenance = raw_output.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        _fail(f"步骤 {step_ref} provenance 必须是 object")
    if isinstance(provenance, Mapping) and provenance != _project_minichem_failure_fact(
        provenance, f"步骤 {step_ref}.provenance"
    ):
        _fail(f"步骤 {step_ref}.provenance 未经过 bounded 脱敏投影")
    return raw_output


def _append_evidence_index_item(
    index: list[dict[str, Any]],
    item: dict[str, Any],
) -> None:
    """在索引膨胀前执行固定条目上限。"""

    if len(index) >= _MAX_EVIDENCE_INDEX_ITEMS:
        raise _EvidenceIndexLimitExceeded("Router evidence index 条目过多")
    index.append(item)


def _is_reliable_public_result_value(value: Any) -> bool:
    """递归确认结果容器成员是非空且可公开引用的 JSON 值。

    输入为已进入 Router 结果投影的值；函数只接受字符串、布尔值、整数、有限
    浮点数以及成员同样通过检查的非空数组或对象。空容器、null、非有限浮点、
    非字符串对象键和宿主对象都返回 ``False``，不修改输入值。
    """

    if value is None:
        return False
    if isinstance(value, str) or isinstance(value, bool):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return bool(value) and all(
            _is_reliable_public_result_value(item) for item in value
        )
    if isinstance(value, Mapping):
        return bool(value) and all(
            isinstance(key, str) and _is_reliable_public_result_value(item)
            for key, item in value.items()
        )
    return False


def _append_result_value_evidence(
    index: list[dict[str, Any]],
    indexed_refs: set[str],
    value: Any,
    reference: str,
    field_name: str,
    step_ref: str,
    *,
    reliable: bool = True,
) -> None:
    """递归展开对象；数组作为保留顺序的一项整体证据。"""

    if isinstance(value, Mapping):
        for key in sorted(value):
            _append_result_value_evidence(
                index,
                indexed_refs,
                value[key],
                f"{reference}/{_pointer_token(key)}",
                key,
                step_ref,
                reliable=reliable,
            )
        return
    if reference in indexed_refs:
        return
    value_reliable = _is_reliable_public_result_value(value)
    _append_evidence_index_item(
        index,
        {
            "evidence_ref": reference,
            "evidence_kind": field_name,
            "reliable": reliable and value_reliable,
            "step_ref": step_ref,
        },
    )
    indexed_refs.add(reference)


def _minichem_check_records(
    execution_by_ref: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """从已完成步骤的 ArtifactRef/collect 摘要重新派生 Gate 检查。"""

    expected: list[dict[str, Any]] = []
    for step_position, (step_ref, step) in enumerate(execution_by_ref.items()):
        raw_output = step["raw_output"]
        if _execution_step_status(step) != "completed":
            # failed/not_executed 只作为失败事实进入 Router，不伪造 completed
            # 产物检查；Program Gate 的约束由其自身记录保留。
            continue
        layer = step.get("execution_layer", "predefined_actions")
        if layer == "predefined_action":
            layer = "predefined_actions"
        if layer == "predefined_actions":
            for artifact_position, artifact in enumerate(raw_output["output_artifacts"]):
                expected.append(
                    {
                        "step_ref": step_ref,
                        "step_position": step_position,
                        "evidence_ref": (
                            f"/execution_result/steps/{step_position}/raw_output/"
                            f"output_artifacts/{artifact_position}"
                        ),
                        "actual": {
                            "artifact_id": artifact["artifact_id"],
                            "semantic_type": artifact["semantic_type"],
                            "media_type": artifact["media_type"],
                            "sha256": artifact["sha256"],
                        },
                    }
                )
            continue
        if layer == "native_software" and _is_public_native_foreach_output(raw_output):
            result = raw_output.get("result")
            items = result.get("items", []) if isinstance(result, Mapping) else []
            for item_index, item in enumerate(items):
                child_result = item.get("result") if isinstance(item, Mapping) else None
                if not isinstance(child_result, Mapping):
                    _fail(f"步骤 {step_ref}.foreach.items[{item_index}] result 非法")
                for field_name in ("files", "outputs"):
                    for record_index, record in enumerate(child_result.get(field_name, [])):
                        if not isinstance(record, Mapping):
                            _fail(f"步骤 {step_ref}.foreach child {field_name} 记录非法")
                        expected.append(
                            {
                                "step_ref": step_ref,
                                "step_position": step_position,
                                "evidence_ref": (
                                    f"/execution_result/steps/{step_position}/raw_output/"
                                    f"result/items/{item_index}/result/{field_name}/{record_index}"
                                ),
                                "actual": {
                                    "path": record.get("path"),
                                    "sha256": record.get("sha256"),
                                    "size_bytes": record.get("size_bytes"),
                                    "kind": deepcopy(record.get("kind")),
                                },
                            }
                        )
                for record_index, record in enumerate(child_result.get("declared_artifacts", [])):
                    if not isinstance(record, Mapping):
                        _fail(f"步骤 {step_ref}.foreach child declared_artifact 记录非法")
                    expected.append(
                        {
                            "step_ref": step_ref,
                            "step_position": step_position,
                            "evidence_ref": (
                                f"/execution_result/steps/{step_position}/raw_output/"
                                f"result/items/{item_index}/result/declared_artifacts/{record_index}"
                            ),
                            "actual": {
                                key: deepcopy(record[key])
                                for key in (
                                    "name",
                                    "path",
                                    "semantic_type",
                                    "media_type",
                                    "producer",
                                    "validation_status",
                                    "validation_errors",
                                    "errors",
                                )
                                if key in record
                            },
                        }
                    )
            continue
        for field_name in ("files", "outputs"):
            for record_index, record in enumerate(raw_output[field_name]):
                expected.append(
                    {
                        "step_ref": step_ref,
                        "step_position": step_position,
                        "evidence_ref": (
                            f"/execution_result/steps/{step_position}/raw_output/"
                            f"{field_name}/{record_index}"
                        ),
                        "actual": {
                            "path": record.get("path", record.get("job_relative_path")),
                            "sha256": record["sha256"],
                            "size_bytes": record["size_bytes"],
                            "kind": deepcopy(record.get("kind")),
                        },
                    }
                )
        for record_index, record in enumerate(raw_output["declared_artifacts"]):
            expected.append(
                {
                    "step_ref": step_ref,
                    "step_position": step_position,
                    "evidence_ref": (
                        f"/execution_result/steps/{step_position}/raw_output/"
                        f"declared_artifacts/{record_index}"
                    ),
                    "actual": {
                        key: deepcopy(record[key])
                        for key in (
                            "name",
                            "path",
                            "workspace_path",
                            "semantic_type",
                            "media_type",
                            "producer",
                            "parent_artifacts",
                            "validation_status",
                            "validation_errors",
                            "errors",
                        )
                        if key in record
                    },
                }
            )
    return expected


def _program_gate_reliability_reasons(
    program_validation: Mapping[str, Any],
) -> list[str]:
    """读取 Program Gate 的已注册结果，列出不能支持 ``evidence_ready`` 的原因。

    这里不重新判断科学含义，也不根据结果数值猜测结论。可靠性只来自 Gate 已经
    写入的 ``allowed_decisions``、``constraint_reasons`` 和注册检查的
    ``evaluation``。状态轴和声明校验只作为 Gate 的审计上下文，不在这里维护第二
    套状态值白名单。返回原因而不是直接抛错，供证据索引和最终发布分别保留可审计
    的失败上下文。
    """

    if not isinstance(program_validation, Mapping):
        return ["program_validation 不是 object"]
    gate = program_validation.get("program_gate_result")
    if not isinstance(gate, Mapping):
        return ["program_gate_result 缺失或不是 object"]

    reasons: list[str] = []
    allowed = gate.get("allowed_decisions")
    if allowed != list(_ROUTER_DECISIONS):
        reasons.append("allowed_decisions 未开放 evidence_ready")
    constraint_reasons = gate.get("constraint_reasons")
    if not isinstance(constraint_reasons, list):
        reasons.append("constraint_reasons 缺失或不是 array")
    elif constraint_reasons:
        reasons.append(
            "program gate 仍有约束：" + ", ".join(str(item) for item in constraint_reasons)
        )

    registered = program_validation.get("registered_check_results")
    if not isinstance(registered, list):
        reasons.append("registered_check_results 缺失或不是 array")
    elif not registered:
        reasons.append("没有已注册的 Program Gate 检查")
    else:
        for index, check in enumerate(registered):
            if not isinstance(check, Mapping):
                reasons.append(f"registered_check_results[{index}] 不是 object")
                continue
            if check.get("evaluation") != "matched":
                reasons.append(
                    f"registered_check_results[{index}] evaluation="
                    f"{check.get('evaluation')!r}"
                )

    return reasons


def _registered_check_is_reliable(
    check: Mapping[str, Any],
    *,
    gate_has_unreliable_evidence: bool = False,
) -> bool:
    """将一条注册检查投影为 Router 证据的可靠标志。"""

    if check.get("evaluation") != "matched":
        return False
    return not gate_has_unreliable_evidence


def _program_gate_has_unreliable_evidence(
    program_validation: Mapping[str, Any],
) -> bool:
    """判断 Gate 是否明确记录了未匹配的注册检查。"""

    reasons = _program_gate_reliability_reasons(program_validation)
    return any(
        reason.startswith("registered_check_results[") for reason in reasons
    )


def _build_minichem_router_evidence_index(
    execution_result: Mapping[str, Any],
    program_validation: Mapping[str, Any],
    workflow: Mapping[str, Any],
    action_projection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """从 MiniChem ActionResult/collect 和 Program Gate 派生证据索引。"""

    actions = _minichem_action_map(action_projection)
    steps = _minichem_workflow_maps(workflow)
    if not isinstance(execution_result, Mapping):
        _fail("MiniChem execution_result 必须是 object")
    execution_steps = execution_result.get("steps")
    statuses = validate_router_execution_terminal(execution_result, len(steps))
    registered = program_validation.get("registered_check_results") if isinstance(program_validation, Mapping) else None
    if not isinstance(registered, list):
        _fail("MiniChem program_validation.registered_check_results 必须是 array")
    gate = (
        program_validation.get("program_gate_result")
        if isinstance(program_validation, Mapping)
        else None
    )
    if "failed" in statuses:
        if (
            not isinstance(gate, Mapping)
            or gate.get("allowed_decisions") != ["revise_workflow"]
            or not isinstance(gate.get("constraint_reasons"), list)
            or not gate.get("constraint_reasons")
        ):
            _fail("完整失败终局必须由受限 Program Gate 收紧到 revise_workflow")
    gate_reliability_reasons = _program_gate_reliability_reasons(program_validation)
    gate_has_unreliable_evidence = _program_gate_has_unreliable_evidence(
        program_validation
    )
    if len(execution_steps) < len(steps):
        if not isinstance(gate, Mapping) or gate.get("allowed_decisions") != ["revise_workflow"]:
            _fail("提前结束的 MiniChem 前缀必须由受限 Program Gate 收紧")
    execution_by_ref: dict[str, Mapping[str, Any]] = {}
    for position, step in enumerate(execution_steps):
        if not isinstance(step, Mapping):
            _fail(f"MiniChem execution_result.steps[{position}] 必须是 object")
        step_ref = step.get("step_ref")
        if step_ref != f"s{position + 1}" or step_ref not in steps:
            _fail("MiniChem execution_result 不是 workflow 的连续前缀")
        workflow_step = steps[step_ref]
        layer = workflow_step.get("execution_layer", "predefined_actions")
        if layer == "predefined_action":
            layer = "predefined_actions"
        if layer == "predefined_actions":
            action_id = workflow_step.get("action_id")
            backend_id = workflow_step.get("backend_id")
            action = actions.get(action_id)
            if action is None or backend_id not in action["backend_ids"]:
                _fail(f"步骤 {step_ref} 的 Action/Backend 未由目录声明")
            if step.get("action_id") != action_id or step.get("backend_id") != backend_id:
                _fail(f"步骤 {step_ref} 与 workflow 的 Action/Backend 不一致")
        else:
            if "action_id" in workflow_step or "backend_id" in workflow_step:
                _fail(f"步骤 {step_ref} open job 不得携带 Action/Backend 身份")
        _validate_minichem_projected_step(step, workflow_step, actions)
        execution_by_ref[step_ref] = step

    expected_checks = _minichem_check_records(execution_by_ref)
    if len(registered) < len(expected_checks):
        _fail("MiniChem Program Gate 检查数量与执行证据不一致")
    if len(registered) > len(expected_checks):
        # 缺失必需产物时，Gate 可以登记一条 actual=null 的负向检查；该事实没有
        # 对应的执行产物位置，但仍可通过 registered_check_results 自身审计。
        extra_checks = registered[len(expected_checks) :]
        if any(
            not isinstance(item, Mapping)
            or item.get("evaluation") == "matched"
            for item in extra_checks
        ):
            _fail("MiniChem Program Gate 检查数量与执行证据不一致")
        expected_checks.extend(
            {
                "step_ref": item.get("step_ref") if isinstance(item, Mapping) else None,
                "step_position": None,
                "evidence_ref": None,
                "actual": None,
            }
            for item in extra_checks
        )
    index: list[dict[str, Any]] = []
    for position, (result, expected) in enumerate(zip(registered, expected_checks)):
        if not isinstance(result, Mapping) or set(result) != {
            "check_ref",
            "evaluation",
            "actual",
            "evidence_ref",
            "diagnostic",
        }:
            _fail("MiniChem program validation 检查结果字段不完整")
        if (
            result.get("check_ref") != f"c{position + 1}"
            or (
                expected["evidence_ref"] is not None
                and (
                    result.get("actual") != expected["actual"]
                    or result.get("evidence_ref") != expected["evidence_ref"]
                )
            )
        ):
            _fail("MiniChem Program Gate 检查与执行证据派生值不一致")
        _append_evidence_index_item(
            index,
            {
                "evidence_ref": f"/program_validation/registered_check_results/{position}/actual",
                "evidence_kind": "artifact_ref",
                "reliable": _registered_check_is_reliable(
                    result,
                    gate_has_unreliable_evidence=gate_has_unreliable_evidence,
                ),
                "step_ref": expected["step_ref"],
            },
        )

    indexed_refs = {item["evidence_ref"] for item in index}
    for step_position, (step_ref, step) in enumerate(execution_by_ref.items()):
        raw_output = step["raw_output"]
        status = statuses[step_position]
        reliable_step = status == "completed" and not gate_reliability_reasons
        result = raw_output.get("result")
        if result is not None:
            _append_result_value_evidence(
                index,
                indexed_refs,
                result,
                f"/execution_result/steps/{step_position}/raw_output/result",
                "result",
                step_ref,
                reliable=reliable_step,
            )
        for artifact_position, _artifact in enumerate(raw_output.get("output_artifacts", [])):
            reference = (
                f"/execution_result/steps/{step_position}/raw_output/output_artifacts/{artifact_position}"
            )
            if reference not in indexed_refs:
                _append_evidence_index_item(
                    index,
                    {
                        "evidence_ref": reference,
                        "evidence_kind": "artifact_ref",
                        "reliable": reliable_step,
                        "step_ref": step_ref,
                    },
                )
                indexed_refs.add(reference)
        if step.get("execution_layer") in {"native_software", "programmable_analysis"}:
            for field_name in ("files", "outputs", "declared_artifacts"):
                for record_index, _record in enumerate(raw_output.get(field_name, [])):
                    reference = (
                        f"/execution_result/steps/{step_position}/raw_output/"
                        f"{field_name}/{record_index}"
                    )
                    if reference in indexed_refs:
                        continue
                    _append_evidence_index_item(
                        index,
                        {
                            "evidence_ref": reference,
                            "evidence_kind": field_name,
                            "reliable": reliable_step,
                            "step_ref": step_ref,
                        },
                    )
                    indexed_refs.add(reference)
        provenance = raw_output.get("provenance")
        if isinstance(provenance, Mapping):
            for field in sorted(provenance):
                reference = (
                    f"/execution_result/steps/{step_position}/raw_output/provenance/"
                    f"{_pointer_token(field)}"
                )
                if reference in indexed_refs:
                    continue
                _append_evidence_index_item(
                    index,
                    {
                        "evidence_ref": reference,
                        "evidence_kind": field,
                        "reliable": provenance[field] is not None and reliable_step,
                        "step_ref": step_ref,
                    },
                )
                indexed_refs.add(reference)
        for field_name, evidence_kind in (
            ("error", "execution_error"),
            ("reason", "not_executed_reason"),
        ):
            if field_name not in raw_output:
                continue
            reference = (
                f"/execution_result/steps/{step_position}/raw_output/{field_name}"
            )
            if reference in indexed_refs:
                continue
            _append_evidence_index_item(
                index,
                {
                    "evidence_ref": reference,
                    "evidence_kind": evidence_kind,
                    "reliable": False,
                    "step_ref": step_ref,
                },
            )
            indexed_refs.add(reference)
    return index


def build_router_evidence_index(
    execution_result: Mapping[str, Any],
    program_validation: Mapping[str, Any],
    workflow: Mapping[str, Any],
    tool_catalog_projection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """复验 MiniChem 执行事实并生成 Router 的固定顺序证据索引。"""

    if not _is_minichem_projection(tool_catalog_projection):
        _fail("Router evidence index 只接受 MiniChem Action projection")
    return _build_minichem_router_evidence_index(
        execution_result,
        program_validation,
        workflow,
        tool_catalog_projection,
    )


def build_bounded_router_evidence_package(
    execution_result: Mapping[str, Any],
    program_validation: Mapping[str, Any],
    workflow: Mapping[str, Any],
    tool_catalog_projection: Sequence[Mapping[str, Any]],
    *,
    source_results: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """把 MiniChem 执行结果和证据索引整理成受大小限制的 Router 输入。

    输入包含执行结果、Program Gate、工作流和 Action 摘要；函数按步骤顺序删减可选
    结果正文并重建证据索引，不修改调用方传入的对象。返回值保留身份、检查和产物
    摘要等必要事实；若这些不可删减的事实本身超过总大小限制，则抛出 ``ValueError``。
    """

    projected = deepcopy(dict(execution_result))
    steps = projected.get("steps")
    if not isinstance(steps, list):
        _fail("Router execution_result.steps 必须是 array")
    if source_results is None:
        sources = [
            step.get("raw_output", {}).get("result") if isinstance(step, Mapping) else None
            for step in steps
        ]
    else:
        sources = list(source_results)
        if len(sources) != len(steps):
            _fail("Router source_results 与执行步骤数量不一致")

    def current_package() -> tuple[list[dict[str, Any]], int]:
        try:
            evidence_index = build_router_evidence_index(
                projected,
                program_validation,
                workflow,
                tool_catalog_projection,
            )
        except _EvidenceIndexLimitExceeded:
            return [], MAX_ROUTER_EVIDENCE_PACKAGE_BYTES + 1
        package = {"execution_result": projected, "evidence_index": evidence_index}
        return evidence_index, _bounded_json_size(package, MAX_ROUTER_EVIDENCE_PACKAGE_BYTES)

    evidence, size = current_package()
    if size <= MAX_ROUTER_EVIDENCE_PACKAGE_BYTES:
        return {"execution_result": projected, "evidence_index": evidence}
    for position, step in enumerate(steps):
        raw_output = step.get("raw_output")
        if not isinstance(raw_output, dict):
            _fail(f"步骤 {position + 1} 的 Router raw_output 必须是 object")
        replacement: list[dict[str, Any]] = []
        _omission(
            replacement,
            source_ref="/result",
            reason="evidence_package_size_limit",
            value=sources[position],
        )
        raw_output["result"] = None
        raw_output["omitted_result_fields"] = replacement
        evidence, size = current_package()
        if size <= MAX_ROUTER_EVIDENCE_PACKAGE_BYTES:
            return {"execution_result": projected, "evidence_index": evidence}
    _fail("Router 证据包超过总上限，且固定 MiniChem 身份不能安全缩减")


def build_expert_scientific_evidence_package(
    *,
    scientific_question: str,
    hypothesis: Mapping[str, Any],
    workflow: Mapping[str, Any],
    execution_results: Mapping[str, Any],
    evidence_index: Sequence[Mapping[str, Any]] | None = None,
    program_validation: Mapping[str, Any] | None = None,
    scientific_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造执行后固定科学视图，并为 Debate 生成一次扁平证据索引。

    ``evidence_index`` 参数仅用于兼容旧恢复调用。新视图始终从内嵌观测、检查、
    失败和未执行事实重新收集引用；不再复制 workflow、execution_results 或原始
    结果。固定 schema 的完整字符串和数组由注册 projector 直接保留，不经过旧的
    2048 字符、256 项或深度截断逻辑。
    """

    if not isinstance(scientific_question, str) or not scientific_question.strip():
        _fail("Expert evidence scientific_question 必须是非空字符串")
    if not isinstance(hypothesis, Mapping):
        _fail("Expert evidence hypothesis 必须是 object")
    if not isinstance(workflow, Mapping) or not isinstance(execution_results, Mapping):
        _fail("Expert evidence workflow/execution_results 必须是 object")
    view, generated_index = build_workflow_scientific_evidence_view(
        scientific_question=scientific_question,
        hypothesis=hypothesis,
        workflow=workflow,
        execution_results=execution_results,
        program_validation=program_validation,
        scientific_context=scientific_context,
    )
    # 通过视图重新扁平化，确保正式 index 与模型收到的唯一事实完全一致。
    index = flatten_workflow_evidence_index(view)
    if generated_index != index:
        _fail("Expert evidence 内嵌事实与 Debate evidence_index 不一致")
    if evidence_index is not None:
        supplied = [deepcopy(dict(item)) for item in evidence_index]
        if supplied != index:
            _fail("传入的 Expert evidence_index 与固定视图不一致")
    return {**view, "evidence_index": index}


def _hypothesis_claim_statement(router_input: Mapping[str, Any]) -> str:
    """读取 Router 当前假设的完整 reasoning。"""

    combination = router_input.get("combination")
    if not isinstance(combination, Mapping):
        _fail("Router combination 必须是 object")
    hypothesis = combination.get("hypothesis")
    if not isinstance(hypothesis, Mapping):
        _fail("Router combination.hypothesis 必须是 object")
    return _nonempty_text(
        hypothesis.get("reasoning"),
        "Router hypothesis.reasoning",
        maximum=3200,
    )


def _evidence_index(router_input: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """校验四字段证据索引及其允许的 MiniChem JSON Pointer 根。"""

    value = router_input.get("evidence_index")
    if not isinstance(value, list):
        _fail("Router evidence_index 必须是 array")
    refs: set[str] = set()
    result: list[Mapping[str, Any]] = []
    pointer_token = r"(?:[^~/]|~0|~1)+"
    registered_pattern = r"/program_validation/registered_check_results/(?:0|[1-9][0-9]*)/actual"
    execution_pattern = (
        r"/execution_result/steps/(?:0|[1-9][0-9]*)/raw_output/(?:"
        rf"result(?:/{pointer_token})*|"
        r"output_artifacts/(?:0|[1-9][0-9]*)|"
        rf"provenance(?:/{pointer_token})*|"
        rf"error(?:/{pointer_token})*|"
        r"reason|"
        r"files/(?:0|[1-9][0-9]*)|"
        r"outputs/(?:0|[1-9][0-9]*)|"
        r"declared_artifacts/(?:0|[1-9][0-9]*)"
        r")"
    )
    for index, item in enumerate(value):
        item = _exact_keys(
            item,
            {"evidence_ref", "evidence_kind", "reliable", "step_ref"},
            f"evidence_index[{index}]",
        )
        reference = item["evidence_ref"]
        if (
            not isinstance(reference, str)
            or re.fullmatch(rf"(?:{registered_pattern}|{execution_pattern})", reference) is None
        ):
            _fail(f"evidence_index[{index}].evidence_ref 不是规范 Router 证据路径")
        if reference in refs:
            _fail("evidence_index 不允许重复 evidence_ref")
        refs.add(reference)
        if not isinstance(item["evidence_kind"], str) or not item["evidence_kind"]:
            _fail(f"evidence_index[{index}].evidence_kind 必须是非空字符串")
        if not isinstance(item["reliable"], bool):
            _fail(f"evidence_index[{index}].reliable 必须是布尔值")
        if item["step_ref"] is not None and (
            not isinstance(item["step_ref"], str) or not item["step_ref"]
        ):
            _fail(f"evidence_index[{index}].step_ref 必须是非空短引用或 null")
        result.append(item)
    return result


def _enum_items(values: Sequence[str]) -> Any:
    """空 enum 在 JSON Schema 中非法，空证据时返回 false。"""

    return {"enum": list(values)} if values else False


def build_router_claim_schema(router_input: Mapping[str, Any]) -> dict[str, Any]:
    """根据允许决定和客观证据构造最小 Router 输出 Schema。"""

    if not isinstance(router_input, Mapping):
        _fail("RouterInput 必须是 object")
    _hypothesis_claim_statement(router_input)
    evidence = _evidence_index(router_input)
    allowed = router_input.get("allowed_decisions")
    if allowed not in [list(_ROUTER_DECISIONS), ["revise_workflow"]]:
        _fail("Router allowed_decisions 不是开放或受限形态")
    evidence_refs = [item["evidence_ref"] for item in evidence]
    evidence_schema = {
        "type": "array",
        "minItems": 1,
        "maxItems": len(evidence_refs),
        "uniqueItems": True,
        "items": _enum_items(evidence_refs),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"enum": list(allowed)},
            "diagnosis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "evidence_refs": deepcopy(evidence_schema),
                },
                "required": ["summary", "evidence_refs"],
            },
        },
        "required": ["decision", "diagnosis"],
    }


def _ordered_unique(values: Sequence[str], order: Mapping[str, int]) -> list[str]:
    """按 evidence index 顺序整理并去重引用。"""

    return sorted(set(values), key=order.__getitem__)


def _validate_reference_list(
    value: Any,
    *,
    allowed_refs: Mapping[str, int],
    label: str,
) -> list[str]:
    """确认模型只引用当前索引并保持稳定顺序。"""

    if not isinstance(value, list):
        _fail(f"{label} 必须是 array")
    if len(value) != len(set(value)):
        _fail(f"{label} 不允许重复 evidence_ref")
    if any(not isinstance(item, str) or item not in allowed_refs for item in value):
        _fail(f"{label} 包含 evidence index 之外的登记引用")
    if value != _ordered_unique(value, allowed_refs):
        _fail(f"{label} 必须保持 evidence index 顺序")
    return list(value)


def _resolve_router_evidence_value(
    router_input: Mapping[str, Any],
    reference: str,
) -> Any:
    """只沿冻结 RouterInput 解析证据，禁止模型补全缺失位置。"""

    current: Any = router_input
    for raw_token in reference.lstrip("/").split("/"):
        token = _decode_pointer_token(raw_token)
        if isinstance(current, Mapping):
            if token not in current:
                _fail(f"evidence_ref 指向不存在的对象字段：{reference}")
            current = current[token]
        elif isinstance(current, list):
            if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
                _fail(f"evidence_ref 数组位置非法：{reference}")
            position = int(token)
            if position >= len(current):
                _fail(f"evidence_ref 数组位置越界：{reference}")
            current = current[position]
        else:
            _fail(f"evidence_ref 穿过了不可下钻的值：{reference}")
    return current


def validate_router_claim_decision(
    value: Any,
    router_input: Mapping[str, Any],
    tool_catalog_projection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """校验 Router 决定只引用当前 MiniChem 证据包中的可靠事实。"""

    if not isinstance(router_input, Mapping):
        _fail("RouterInput 必须是 object")
    if router_input.get("tool_catalog_projection") != tool_catalog_projection:
        _fail("RouterInput 的 MiniChem Catalog 投影与校验目录不一致")
    expected_package = build_bounded_router_evidence_package(
        router_input["execution_result"],
        router_input["program_validation"],
        router_input["workflow"],
        tool_catalog_projection,
        source_results=[
            step["raw_output"].get("result")
            for step in router_input["execution_result"]["steps"]
        ],
    )
    if (
        router_input["execution_result"] != expected_package["execution_result"]
        or router_input.get("evidence_index") != expected_package["evidence_index"]
    ):
        _fail("Router 证据包与当前 MiniChem 执行事实不一致或超过上限")
    answer = _exact_keys(value, {"decision", "diagnosis"}, "Router output")
    allowed = router_input.get("allowed_decisions")
    if not isinstance(allowed, list) or answer["decision"] not in allowed:
        _fail("Router decision 越过 program gate")
    diagnosis = _exact_keys(
        answer["diagnosis"],
        {"summary", "evidence_refs"},
        "Router diagnosis",
    )
    summary = _nonempty_text(diagnosis["summary"], "Router diagnosis.summary")
    evidence = _evidence_index(router_input)
    evidence_order = {item["evidence_ref"]: index for index, item in enumerate(evidence)}
    evidence_by_ref = {item["evidence_ref"]: item for item in evidence}
    for item in evidence:
        reference = item["evidence_ref"]
        _resolve_router_evidence_value(router_input, reference)
        if reference.startswith("/execution_result/steps/"):
            step_position = int(reference.split("/")[3])
            execution_steps = router_input["execution_result"]["steps"]
            if (
                step_position >= len(execution_steps)
                or execution_steps[step_position].get("step_ref") != item["step_ref"]
            ):
                _fail("Router evidence 的 step_ref 与执行位置不一致")
    top_refs = _validate_reference_list(
        diagnosis["evidence_refs"],
        allowed_refs=evidence_order,
        label="Router diagnosis.evidence_refs",
    )
    if not top_refs:
        _fail("Router diagnosis.evidence_refs 至少需要一项客观证据")
    gate = router_input.get("program_validation", {}).get("program_gate_result")
    if not isinstance(gate, Mapping):
        _fail("program gate result 必须是 object")
    if gate.get("allowed_decisions") != allowed:
        _fail("Router allowed_decisions 与 program gate 不一致")
    reasons = gate.get("constraint_reasons")
    if not isinstance(reasons, list):
        _fail("program gate constraint_reasons 必须是 array")
    if allowed == ["revise_workflow"] and answer["decision"] != "revise_workflow":
        _fail("受限 program gate 只允许 revise_workflow")
    if answer["decision"] == "evidence_ready":
        gate_reliability_reasons = _program_gate_reliability_reasons(
            router_input.get("program_validation", {})
        )
        if gate_reliability_reasons:
            _fail(
                "Program Gate 不允许 evidence_ready："
                + "; ".join(gate_reliability_reasons)
            )
    if answer["decision"] != "revise_workflow" and any(
        evidence_by_ref[reference]["reliable"] is not True for reference in top_refs
    ):
        _fail("Router 不能引用缺失、为空、类型错误或来源不可信的事实")
    return {
        "decision": answer["decision"],
        "diagnosis": {"summary": summary, "evidence_refs": top_refs},
    }


def validate_program_gate_for_evidence_ready(
    program_validation: Mapping[str, Any],
) -> None:
    """在最终发布前复核同一份 Program Gate 是否仍允许 evidence_ready。

    该函数只读取执行阶段已经保存的 Gate，不重新计算科学结论。若 Gate 的允许
    决定、注册检查、声明校验或状态轴显示出任何不可靠原因，则抛出带原因的
    ``ValueError``，调用方不得把 Router 的旧 ``evidence_ready`` 当作成功结果。
    状态轴和声明校验不会在这里重新解释，执行阶段应把它们的失败注册为
    ``evaluation != matched`` 或写入 Gate 约束原因。
    """

    reasons = _program_gate_reliability_reasons(program_validation)
    if reasons:
        _fail("最终发布拒绝 evidence_ready：" + "; ".join(reasons))


__all__ = [
    "MAX_ROUTER_EVIDENCE_PACKAGE_BYTES",
    "build_expert_scientific_evidence_package",
    "build_bounded_router_evidence_package",
    "build_router_claim_schema",
    "build_router_evidence_index",
    "build_router_execution_projection",
    "validate_router_execution_terminal",
    "validate_program_gate_for_evidence_ready",
    "validate_router_claim_decision",
]
