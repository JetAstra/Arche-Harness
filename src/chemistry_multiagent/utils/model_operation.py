"""统一管理模型调用、响应校验和可追溯的请求记录。

每次业务模型调用都保存实际输入、原始响应和失败原因。模型格式不正确时只能
从头重答，不能靠正则修补后直接影响工作流状态。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Callable, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator

from chemistry_multiagent.utils.model_catalog import (
    compile_model_request,
)
from chemistry_multiagent.utils.api_retry import (
    MODEL_CALL_RETRY_COUNT,
    MODEL_CALL_RETRY_DELAY_SECONDS,
    is_retryable_api_exception,
    is_retryable_http_status,
    response_http_status,
    retry_api_call,
    suppress_nested_api_retries,
)
from chemistry_multiagent.utils.model_defaults import is_codex_model
from chemistry_multiagent.utils.model_prompts import build_model_messages
from chemistry_multiagent.utils.workflow_contract import (
    canonical_json_text,
    safe_join_task_path,
)


_LOGGER = logging.getLogger(__name__)


# ==================== 模型操作协议常量 ====================

PROTOCOL_RESPONSE_LIMIT = 4
MODEL_VALIDATION_KINDS = frozenset(
    {
        "json_schema",
        "query_delimited_text",
        # 兼容历史清单；当前关系和排名阶段使用严格 JSON schema。
        "text_enum_relationship",
        "text_enum_rank",
    }
)
FAILURE_KINDS = frozenset(
    {
        "empty_choices",
        "empty_content",
        "truncated",
        "json_syntax",
        "schema",
        "domain",
        "provider_envelope",
        "transport",
    }
)
_STAGE_KINDS = frozenset({"model", "retrieval", "deterministic"})
_OPERATION_DIRECTORIES = {
    "workflow_generation": "workflow_generation_inputs",
    "workflow_optimization": "workflow_optimization_inputs",
    "router": "router_inputs",
    "workflow_revision": "workflow_revision_inputs",
    "hypothesis_generation": "hypothesis_generation_inputs",
}
_OPERATION_STAGES = {
    "workflow_generation": frozenset({"workflow_generation"}),
    "workflow_optimization": frozenset({"workflow_optimization"}),
    "router": frozenset({"router"}),
    "workflow_revision": frozenset({"workflow_revision"}),
    "hypothesis_generation": frozenset(
        {
            "retrieval",
            "query",
            "hypothesis",
            "relationship",
            "merge",
            "complement",
            "conflict",
            "rank",
            "regeneration",
        }
    ),
}
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
_REF_KEYS = {"relative_path", "sha256", "media_type", "size"}
_MANIFEST_KEYS = {
    "schema_version",
    "operation_instance_id",
    "operation",
    "base_input_ref",
    "prior_manifest_ref",
    "stages",
}
_STAGE_KEYS = {
    "stage_index",
    "stage",
    "stage_kind",
    "query_index",
    "validation_kind",
    "attempts",
    "status",
}
_ATTEMPT_KEYS = {
    "attempt_index",
    "request_ref",
    "response_ref",
    "provider_trace_id",
    "error_ref",
    "validation_status",
}
_RETRIEVAL_RESPONSE_KEYS = {"schema_version", "result", "model_calls"}
_RETRIEVAL_MODEL_CALL_KEYS = {
    "stage",
    "transport_attempt_index",
    "request_ref",
    "response_ref",
    "provider_trace_id",
    "error_ref",
    "validation_status",
}
_RETRIEVAL_MODEL_STAGES = {
    "chemistry_context",
    "keyword",
    "document_qa",
    "literature_review",
}
_ERROR_SUMMARIES = {
    "provider_envelope": "provider response envelope is invalid",
    "empty_choices": "provider returned no choices",
    "truncated": "provider response was truncated",
    "empty_content": "provider returned empty content",
    "json_syntax": "response is not one strict JSON value",
    "schema": "response does not match the required schema",
    "domain": "response violates deterministic domain constraints",
    "transport": "provider transport attempts were exhausted",
}
_MARKDOWN_FENCE_SUMMARY = (
    "previous response used a Markdown code fence; regenerate one raw JSON value "
    "without fences"
)
_JSON_SYNTAX_DETAIL_PREFIX = "json_syntax.v1:"
_JSON_SYNTAX_REASONS = frozenset(
    {
        "syntax error",
        "duplicate object field",
        "non-standard numeric constant",
        "isolated Unicode surrogate",
    }
)
_SCHEMA_DETAIL_PREFIX = "schema_violation.v1:"
_SCHEMA_LENGTH_RULES = frozenset(
    {
        "maxLength",
        "minLength",
        "maxItems",
        "minItems",
        "maxProperties",
        "minProperties",
    }
)
_SCHEMA_PATH_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_SCHEMA_DETAIL_LIMIT = 4
_DOMAIN_DETAIL_PREFIX = "domain_violation.v1:"
_DOMAIN_REASON_LIMIT = 600
_LOCAL_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._-])/(?:tmp|mnt|home|root|var)(?:/[^\s；，。]*)?"
)
_WEB_ADDRESS_PATTERN = re.compile(r"(?:https?|file)://[^\s；，。]+", re.IGNORECASE)
_SECRET_TOKEN_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|[A-Fa-f0-9]{32,})\b"
)
_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_MISSING = object()


def _utc_timestamp() -> str:
    """返回便于排序的 UTC 时间戳；仅供旁路观察事件使用。"""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _protocol_error_with_history(
    history: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    """把本阶段已经出现过的安全错误一并交给下一次完整重答。

    只保留验证器生成的类别和摘要，不携带旧模型正文。相同错误去重后按首次出现
    顺序排列，避免模型修好一种格式问题时又忘记更早已经出现过的问题。
    """

    if not history:
        raise ModelOperationError("协议错误历史不能为空")
    current = history[-1]
    current_pair = (current["failure_kind"], current["summary"])
    earlier: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in history[:-1]:
        pair = (item["failure_kind"], item["summary"])
        if pair == current_pair or pair in seen:
            continue
        seen.add(pair)
        earlier.append(f"{pair[0]}: {pair[1]}")
    summary = current_pair[1]
    if earlier:
        summary = (
            f"{summary}; earlier stage failures must not recur: "
            + " | ".join(earlier)
        )
    return {
        "failure_kind": current_pair[0],
        "summary": summary,
    }


# ==================== 协议错误与阶段类型 ====================

class ModelOperationError(ValueError):
    """模型调用尚未发出前，本地输入或已保存文件就不满足要求。"""


class ModelOperationStageError(RuntimeError):
    """一次动态子调用已留下完整失败收据，外层应停止并冻结逻辑操作。"""

    def __init__(self, stage: str, failure_kind: str):
        self.stage = stage
        self.failure_kind = failure_kind
        super().__init__(f"{stage} 子调用失败：{failure_kind}")


@dataclass(frozen=True)
class ModelStage:
    """描述一次模型子调用要返回什么，以及程序将如何检查它。

    多步骤任务按这些对象的顺序执行；每个对象只负责一个明确输出，避免一个
    提示词同时承担生成、比较和修订等多种职责。
    """

    stage: str
    stage_kind: str
    validation_kind: str
    query_index: int | None = None
    schema: Mapping[str, Any] | None = None
    domain_validator: Callable[[Any], Any] | None = None
    domain_validator_id: str | None = None
    expected_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ModelOperationError("stage 必须是非空字符串")
        if self.stage_kind not in _STAGE_KINDS:
            raise ModelOperationError("stage_kind 未注册")
        if self.stage_kind == "model" and self.validation_kind not in MODEL_VALIDATION_KINDS:
            raise ModelOperationError("model stage 的 validation_kind 未注册")
        if self.validation_kind == "json_schema":
            if not isinstance(self.schema, Mapping):
                raise ModelOperationError("json_schema stage 必须提供 schema")
            Draft202012Validator.check_schema(dict(self.schema))
        elif self.schema is not None:
            raise ModelOperationError("纯文本 stage 不得携带 JSON schema")
        if self.domain_validator is not None and not callable(self.domain_validator):
            raise ModelOperationError("domain_validator 必须可调用")
        if self.domain_validator is not None:
            if (
                not isinstance(self.domain_validator_id, str)
                or not self.domain_validator_id.strip()
                or not _IDENTITY_PATTERN.fullmatch(self.domain_validator_id)
            ):
                raise ModelOperationError(
                    "使用领域校验器时必须提供稳定的 domain_validator_id"
                )
        elif self.domain_validator_id is not None:
            raise ModelOperationError(
                "没有领域校验器时 domain_validator_id 必须为 null"
            )
        if self.validation_kind == "query_delimited_text":
            if not isinstance(self.expected_count, int) or self.expected_count <= 0:
                raise ModelOperationError("query stage 必须提供正整数 expected_count")
        elif self.expected_count is not None:
            raise ModelOperationError("非 query stage 不得携带 expected_count")
        if self.query_index is not None and (
            not isinstance(self.query_index, int) or self.query_index <= 0
        ):
            raise ModelOperationError("query_index 必须是正整数或 null")


class ModelStageObserver(Protocol):
    """接收模型阶段旁路事件的最小协议。

    观察器只用于生成展示或诊断记录，不参与正式 manifest、响应校验和恢复。三个
    方法收到的 ``event`` 都是独立深拷贝；观察器可以保存或修改它，而不会改变模型
    操作内部的输入、响应或返回值。观察器还可以自行实现可选的
    ``mark_model_output_degraded(reason)`` 方法，表示旁路记录已经降级。
    """

    def start_model_stage(self, event: Mapping[str, Any]) -> Any | None:
        """记录阶段开始，并返回供后续事件关联的观察器句柄。"""

    def record_model_attempt(
        self,
        handle: Any,
        event: Mapping[str, Any],
    ) -> None:
        """记录一次逻辑模型回答或最终传输失败。"""

    def finish_model_stage(
        self,
        handle: Any,
        event: Mapping[str, Any],
    ) -> None:
        """记录阶段终态；阶段不会因为旁路记录失败而改变。"""


@dataclass
class _ModelStageObservation:
    """保存一次旁路观察的内部计时和最近收据，不写入正式清单。"""

    handle: Any
    started_at: str
    started_clock: float
    stage: ModelStage
    stage_index: int
    stage_input: Mapping[str, Any]
    display_context: Mapping[str, Any]
    last_attempt: dict[str, Any] | None = None
    attempt_source_refs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _ValidationResult:
    valid: bool
    value: Any = None
    failure_kind: str | None = None
    finish_reason: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class _ModelStageOutcome:
    """记录一次模型 stage 的最终值或失败类别。"""

    value: Any = None
    failure_kind: str | None = None


# ==================== 模型响应规整与校验 ====================

def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_safe(value: Any) -> Any:
    """把不同模型SDK的响应对象复制为可保存的普通JSON数据。

    这里只消除SDK对象差异，不解释正文含义，也不会因此接受原本无效的响应。
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _json_safe(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return {"raw_value_kind": type(value).__name__}


def normalize_provider_envelope(response: Any) -> Any:
    """保留候选、结束原因和追踪ID，同时消除SDK对象类型差异。"""

    return _json_safe(response)


def _failure(
    failure_kind: str,
    *,
    finish_reason: str | None = None,
    summary: str | None = None,
) -> _ValidationResult:
    if failure_kind not in FAILURE_KINDS:
        raise ModelOperationError(f"未知 failure_kind：{failure_kind}")
    if summary is not None:
        schema_summary = failure_kind == "schema" and (
            summary == _ERROR_SUMMARIES["schema"]
            or summary.startswith(_SCHEMA_DETAIL_PREFIX)
        )
        json_summary = failure_kind == "json_syntax" and (
            summary
            in {
                _ERROR_SUMMARIES["json_syntax"],
                _MARKDOWN_FENCE_SUMMARY,
            }
            or summary.startswith(_JSON_SYNTAX_DETAIL_PREFIX)
        )
        domain_summary = failure_kind == "domain" and (
            summary == _ERROR_SUMMARIES["domain"]
            or summary.startswith(_DOMAIN_DETAIL_PREFIX)
        )
        if not schema_summary and not json_summary and not domain_summary:
            raise ModelOperationError("模型错误只能携带已登记的安全格式说明")
    return _ValidationResult(
        valid=False,
        failure_kind=failure_kind,
        finish_reason=finish_reason,
        summary=summary or _ERROR_SUMMARIES[failure_kind],
    )


def _safe_schema_path(path: Sequence[Any]) -> str | None:
    """把校验位置缩成不含原始正文的 JSON 路径。

    数组下标和已登记字段名足以让模型找到错误；遇到异常字段名时宁可退回笼统
    提示，也不把模型生成的任意文本重新带入下一份请求。
    """

    segments = []
    for item in path:
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            segments.append(str(item))
            continue
        if isinstance(item, str) and _SCHEMA_PATH_SEGMENT.fullmatch(item):
            segments.append(item)
            continue
        return None
    return "/" + "/".join(segments) if segments else "/"


def _json_value_kind(value: Any) -> str:
    """返回模型容易理解的 JSON 类型名称，不包含原始字段值。"""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "unknown"


def _safe_schema_coordinate_value(value: Any) -> Any:
    """只保留来自可信 schema 的短标量，绝不回显模型生成的正文。"""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if -1_000_000_000 <= value <= 1_000_000_000:
            return value
        return _MISSING
    if isinstance(value, float):
        if math.isfinite(value) and abs(value) <= 1_000_000_000:
            return value
        return _MISSING
    if isinstance(value, str):
        if (
            1 <= len(value) <= 128
            and value == value.strip()
            and not any(ord(character) < 32 for character in value)
            and "```" not in value
        ):
            return value
        return _MISSING
    if isinstance(value, (list, tuple)) and 1 <= len(value) <= 8:
        normalized = []
        for item in value:
            safe_item = _safe_schema_coordinate_value(item)
            if safe_item is _MISSING or isinstance(safe_item, list):
                return _MISSING
            normalized.append(safe_item)
        return normalized
    return _MISSING


def _schema_error_details(error: Any) -> list[dict[str, Any]]:
    """把一项 jsonschema 错误翻译成不含原响应值的修正坐标。"""

    path = _safe_schema_path(tuple(error.absolute_path)) or "/"
    raw_rule = str(error.validator)
    rule = (
        raw_rule
        if _SCHEMA_PATH_SEGMENT.fullmatch(raw_rule)
        else "schema"
    )
    expected = error.validator_value
    instance = error.instance

    if rule in _SCHEMA_LENGTH_RULES:
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            return []
        if rule in {"maxLength", "minLength"} and isinstance(instance, str):
            observed = len(instance)
        elif rule in {"maxItems", "minItems"} and isinstance(instance, list):
            observed = len(instance)
        elif rule in {"maxProperties", "minProperties"} and isinstance(
            instance, Mapping
        ):
            observed = len(instance)
        else:
            return []
        return [
            {
                "path": path,
                "rule": rule,
                "expected": expected,
                "observed": observed,
            }
        ]

    if rule == "required" and isinstance(expected, list) and isinstance(
        instance, Mapping
    ):
        details = []
        for field in expected:
            if (
                field not in instance
                and isinstance(field, str)
                and _SCHEMA_PATH_SEGMENT.fullmatch(field)
            ):
                details.append(
                    {
                        "path": path,
                        "rule": rule,
                        "expected": field,
                        "observed": "missing",
                    }
                )
        return details

    if rule == "additionalProperties" and isinstance(instance, Mapping):
        properties = error.schema.get("properties", {})
        declared = set(properties) if isinstance(properties, Mapping) else set()
        extras = sorted(
            field
            for field in instance
            if field not in declared
            and isinstance(field, str)
            and _SCHEMA_PATH_SEGMENT.fullmatch(field)
        )
        observed: Any = extras[:8] if extras else "one or more undeclared fields"
        return [
            {
                "path": path,
                "rule": rule,
                "expected": "only fields declared by the schema",
                "observed": observed,
            }
        ]

    if rule == "type":
        safe_expected = _safe_schema_coordinate_value(expected)
        if safe_expected is _MISSING:
            safe_expected = "the type declared by the schema"
        return [
            {
                "path": path,
                "rule": rule,
                "expected": safe_expected,
                "observed": _json_value_kind(instance),
            }
        ]

    if rule == "enum":
        safe_expected = _safe_schema_coordinate_value(expected)
        if safe_expected is _MISSING:
            safe_expected = "one registered enum value"
        return [
            {
                "path": path,
                "rule": rule,
                "expected": safe_expected,
                "observed": f"unregistered {_json_value_kind(instance)} value",
            }
        ]

    if rule == "const":
        safe_expected = _safe_schema_coordinate_value(expected)
        if safe_expected is _MISSING:
            safe_expected = "the constant declared by the schema"
        return [
            {
                "path": path,
                "rule": rule,
                "expected": safe_expected,
                "observed": "different value",
            }
        ]

    if rule in {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }:
        safe_expected = _safe_schema_coordinate_value(expected)
        safe_observed = _safe_schema_coordinate_value(instance)
        if safe_expected is not _MISSING and safe_observed is not _MISSING:
            return [
                {
                    "path": path,
                    "rule": rule,
                    "expected": safe_expected,
                    "observed": safe_observed,
                }
            ]

    fixed_expectations = {
        "anyOf": "match at least one allowed schema branch",
        "oneOf": "match exactly one allowed schema branch",
        "allOf": "match every required schema branch",
        "not": "not match the forbidden schema branch",
        "pattern": "string matching the required pattern",
        "uniqueItems": "array items must be unique",
        "contains": "array containing a required item",
        "dependentRequired": "all dependent fields must be present",
        "propertyNames": "all object field names must be valid",
    }
    return [
        {
            "path": path,
            "rule": rule,
            "expected": fixed_expectations.get(
                rule, f"satisfy the {rule} schema rule"
            ),
            "observed": "rule not satisfied",
        }
    ]


def _actionable_schema_errors(error: Any) -> list[Any]:
    """Select the closest failing branch from nested schema combinators."""

    if error.validator not in {"oneOf", "anyOf"} or not error.context:
        return [error]
    grouped: dict[Any, list[Any]] = {}
    for child in error.context:
        relative_path = tuple(child.relative_schema_path)
        branch = relative_path[0] if relative_path else None
        grouped.setdefault(branch, []).append(child)

    candidates = []
    for branch, children in grouped.items():
        expanded = [
            leaf
            for child in children
            for leaf in _actionable_schema_errors(child)
        ]
        missing_required = 0
        branch_schema = None
        if isinstance(error.schema, Mapping):
            branches = error.schema.get(error.validator)
            if isinstance(branches, list) and isinstance(branch, int) and branch < len(branches):
                branch_schema = branches[branch]
        if isinstance(branch_schema, Mapping) and isinstance(error.instance, Mapping):
            required = branch_schema.get("required", [])
            if isinstance(required, list):
                missing_required = sum(item not in error.instance for item in required)
        const_errors = sum(item.validator == "const" for item in expanded)
        candidates.append((branch, expanded, missing_required, const_errors))
    _, selected, _, _ = min(
        candidates,
        key=lambda item: (
            item[2],
            item[3],
            len(item[1]),
            -sum(len(tuple(error.absolute_path)) for error in item[1]),
            str(item[0]),
        ),
    )
    return selected or [error]


def _schema_failure_summary(errors: Sequence[Any]) -> str:
    """报告具体 JSON 路径和规则，但不回显未通过校验的字段内容。"""

    details = []
    actionable = [
        leaf
        for error in errors
        for leaf in _actionable_schema_errors(error)
    ]
    ordered = sorted(
        actionable,
        key=lambda error: (
            tuple(str(item) for item in error.absolute_path),
            str(error.validator),
        ),
    )
    for error in ordered:
        for detail in _schema_error_details(error):
            details.append(detail)
            if len(details) == _SCHEMA_DETAIL_LIMIT:
                break
        if len(details) == _SCHEMA_DETAIL_LIMIT:
            break
    if not details:
        return _ERROR_SUMMARIES["schema"]
    return _SCHEMA_DETAIL_PREFIX + canonical_json_text(details)


def _safe_domain_reason(error: BaseException | str) -> str:
    """保留程序给出的第一行原因，并清除路径、密钥和可执行包装文字。"""

    if isinstance(error, BaseException) and not isinstance(error, ValueError):
        return "validator encountered an internal error while checking this response"
    raw = str(error)
    first_line = next(
        (line.strip() for line in raw.splitlines() if line.strip()),
        "",
    )
    if not first_line:
        return "response failed the registered domain validator"
    reason = _LOCAL_PATH_PATTERN.sub("<local path>", first_line)
    reason = _WEB_ADDRESS_PATTERN.sub("<external address>", reason)
    reason = _SECRET_TOKEN_PATTERN.sub("<secret>", reason)
    reason = reason.replace("```", "").replace("`", "")
    reason = reason.replace("|", "/")
    reason = " ".join(reason.split())
    if len(reason) > _DOMAIN_REASON_LIMIT:
        reason = reason[: _DOMAIN_REASON_LIMIT - 3].rstrip() + "..."
    return reason or "response failed the registered domain validator"


def _domain_failure_summary(
    validator_id: str,
    error: BaseException | str,
) -> str:
    """把领域校验器身份和可操作原因编码成一份可复验的小型诊断。"""

    if not _IDENTITY_PATTERN.fullmatch(validator_id):
        raise ModelOperationError("领域错误缺少稳定的 validator identity")
    return _DOMAIN_DETAIL_PREFIX + canonical_json_text(
        {
            "validator": validator_id,
            "reason": _safe_domain_reason(error),
        }
    )


def _extract_envelope_content(envelope: Any) -> tuple[str | None, str | None, _ValidationResult | None]:
    choices = _field(envelope, "choices", _MISSING)
    if choices is _MISSING or not isinstance(choices, list):
        return None, None, _failure("provider_envelope")
    if not choices:
        return None, None, _failure("empty_choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None, None, _failure("provider_envelope")
    finish_raw = choice.get("finish_reason")
    finish_reason = str(finish_raw) if finish_raw is not None else None
    normalized_finish = finish_reason.strip().lower() if finish_reason is not None else None
    if normalized_finish in _TRUNCATED_FINISH_REASONS:
        return None, finish_reason, _failure(
            "truncated", finish_reason=finish_reason
        )
    if normalized_finish != "stop":
        return None, finish_reason, _failure(
            "provider_envelope", finish_reason=finish_reason
        )
    message = choice.get("message", _MISSING)
    if not isinstance(message, Mapping):
        return None, finish_reason, _failure("provider_envelope", finish_reason=finish_reason)
    content = message.get("content", _MISSING)
    if content is _MISSING or content is None:
        return None, finish_reason, _failure("empty_content", finish_reason=finish_reason)
    if not isinstance(content, str):
        return None, finish_reason, _failure("provider_envelope", finish_reason=finish_reason)
    if not content.strip():
        return None, finish_reason, _failure("empty_content", finish_reason=finish_reason)
    return content, finish_reason, None


def _parse_strict_json(content: str) -> Any:
    """解析模型 JSON，并只合并内容完全相同的重复字段。

    模型偶尔会在同一个对象的开头和结尾重复输出同一字段。若字段值完全一致，
    合并它不会改变响应含义，同时可以避免为纯格式抖动重新发送整份大请求。
    只要两次取值的类型或内容不同，仍按有歧义的重复字段拒绝。
    """

    def reject_constant(token: str) -> None:
        raise ValueError(f"非法 JSON 常量 {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                previous = value[key]
                if type(previous) is not type(item) or previous != item:
                    raise ValueError("JSON object 含重复键")
                # 原始响应仍会完整留档；这里只为后续 Schema 校验生成无歧义对象。
                continue
            value[key] = item
        return value

    def reject_isolated_surrogates(value: Any) -> None:
        if isinstance(value, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
                raise ValueError("JSON string 含孤立 surrogate")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                reject_isolated_surrogates(key)
                reject_isolated_surrogates(item)
            return
        if isinstance(value, list):
            for item in value:
                reject_isolated_surrogates(item)

    parsed = json.loads(
        content,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    reject_isolated_surrogates(parsed)
    return parsed


def _unwrap_complete_json_fence(content: str) -> str:
    """只移除完整包住响应的单层 JSON 围栏，不改写其中任何 JSON 内容。"""

    candidate = content.strip(" \t\r\n")
    lines = candidate.splitlines()
    if len(lines) < 3:
        return content
    opening = lines[0].strip(" \t").lower()
    closing = lines[-1].strip(" \t")
    if opening not in {"```", "```json"} or closing != "```":
        return content
    return "\n".join(lines[1:-1])


def _json_syntax_failure_summary(
    content: str,
    error: BaseException,
) -> str:
    """说明 JSON 失败种类和位置，但不读取、修补或回显错误正文。"""

    lines = content.strip().splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        return _MARKDOWN_FENCE_SUMMARY

    line = 0
    column = 0
    reason = "syntax error"
    if isinstance(error, json.JSONDecodeError):
        line = max(1, int(error.lineno))
        column = max(1, int(error.colno))
    else:
        message = str(error)
        if message == "JSON object 含重复键":
            reason = "duplicate object field"
        elif message.startswith("非法 JSON 常量 "):
            reason = "non-standard numeric constant"
        elif message == "JSON string 含孤立 surrogate":
            reason = "isolated Unicode surrogate"
    if reason not in _JSON_SYNTAX_REASONS:
        reason = "syntax error"
    return _JSON_SYNTAX_DETAIL_PREFIX + canonical_json_text(
        {
            "reason": reason,
            "line": line,
            "column": column,
        }
    )


def validate_model_envelope(envelope: Any, stage: ModelStage) -> _ValidationResult:
    """从外到内检查模型响应是否完整、格式正确并满足当前任务要求。

    先判断候选是否存在和正文是否截断，再解析JSON或枚举，最后运行具体业务
    检查，避免把截断文本误报成普通JSON错误。
    """

    normalized = normalize_provider_envelope(envelope)
    content, finish_reason, envelope_error = _extract_envelope_content(normalized)
    if envelope_error is not None:
        return envelope_error
    assert content is not None
    if stage.validation_kind == "json_schema":
        parse_content = _unwrap_complete_json_fence(content)
        try:
            value = _parse_strict_json(parse_content)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return _failure(
                "json_syntax",
                finish_reason=finish_reason,
                summary=_json_syntax_failure_summary(parse_content, exc),
            )
        errors = list(Draft202012Validator(dict(stage.schema or {})).iter_errors(value))
        if errors:
            return _failure(
                "schema",
                finish_reason=finish_reason,
                summary=_schema_failure_summary(errors),
            )
    elif stage.validation_kind == "query_delimited_text":
        parts = [item.strip() for item in content.split("<>")]
        if (
            len(parts) != stage.expected_count
            or any(not item for item in parts)
        ):
            format_example = "<>".join(
                f"query{index}"
                for index in range(1, int(stage.expected_count or 0) + 1)
            )
            wrapped_example = "".join(
                f"<query{index}>"
                for index in range(1, int(stage.expected_count or 0) + 1)
            )
            empty_count = sum(not item for item in parts)
            return _failure(
                "domain",
                finish_reason=finish_reason,
                summary=_domain_failure_summary(
                    "query_delimited_text",
                    (
                        f"需要精确返回 {stage.expected_count} 个以 <> 分隔的非空 "
                        f"query；正确格式骨架是 {format_example}，字面量 <> "
                        f"只放在相邻项之间，不要写成 {wrapped_example}；"
                        f"实际得到 {len(parts)} 个，其中空项 {empty_count} 个"
                    ),
                ),
            )
        value = parts
    elif stage.validation_kind == "text_enum_relationship":
        value = content.strip().lower()
        if value not in {"merge", "complement", "conflict", "independent"}:
            return _failure(
                "domain",
                finish_reason=finish_reason,
                summary=_domain_failure_summary(
                    "text_enum_relationship",
                    (
                        "relationship 必须精确返回 merge、complement、conflict 或 "
                        "independent；当前响应不是已登记枚举值"
                    ),
                ),
            )
    elif stage.validation_kind == "text_enum_rank":
        value = content.strip().upper()
        if value not in {"A", "B"}:
            return _failure(
                "domain",
                finish_reason=finish_reason,
                summary=_domain_failure_summary(
                    "text_enum_rank",
                    "rank 必须精确返回 A 或 B；当前响应不是已登记枚举值",
                ),
            )
    else:
        raise ModelOperationError(
            f"stage 使用未注册 validation_kind：{stage.validation_kind}"
        )
    if stage.domain_validator is not None:
        try:
            validated = stage.domain_validator(deepcopy(value))
        except Exception as exc:
            return _failure(
                "domain",
                finish_reason=finish_reason,
                summary=_domain_failure_summary(
                    str(stage.domain_validator_id),
                    exc,
                ),
            )
        if validated is not None:
            value = validated
    return _ValidationResult(valid=True, value=value, finish_reason=finish_reason)


def _extract_observer_parsed_output(
    envelope: Any,
    stage: ModelStage,
    validation: _ValidationResult,
) -> Any:
    """为旁路记录保留已成功解析的 JSON，即使正式 schema/domain 校验失败。

    该值只用于展示，不参与正式重答、清单或工作流状态。语法错误、空响应、截断
    响应以及纯文本枚举/分隔格式没有可安全保存的 JSON 解析值，因此返回 ``None``。
    """

    if validation.valid:
        return deepcopy(validation.value)
    if stage.validation_kind != "json_schema":
        return None
    normalized = normalize_provider_envelope(envelope)
    content, _, envelope_error = _extract_envelope_content(normalized)
    if envelope_error is not None or content is None:
        return None
    try:
        return deepcopy(_parse_strict_json(_unwrap_complete_json_fence(content)))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def extract_gaussian_expert_text(envelope: Any) -> str | None:
    """读取Gaussian专家给出的非空自然语言意见。

    专家意见只是辅助诊断，不驱动确定性状态；即使响应标记为截断也保留已返回
    文字，同时绝不强迫它伪装成JSON。
    """

    normalized = normalize_provider_envelope(envelope)
    choices = _field(normalized, "choices", _MISSING)
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content.strip()


# ==================== 操作产物与路径校验 ====================

def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json_text(value).encode("utf-8")


def _assert_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ModelOperationError(f"{label} 字段不符合固定契约")


def _validate_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModelOperationError(f"{label} 必须是任务相对 POSIX 路径")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelOperationError(f"{label} 不能是绝对路径或包含 ..")
    return path


# ==================== 模型阶段共用执行器 ====================

class _ModelStageExecutor:
    """保存模型阶段共用的请求、校验和清单文件操作。

    该内部类不决定固定阶段或动态阶段的业务顺序，也不提供公开工作流入口。
    两种操作运行器只复用这里的单阶段请求循环和内容寻址文件读写。
    """

    def __init__(
        self,
        *,
        task_root: str | Path,
        operation_instance_id: str,
        operation: str,
        base_input: Mapping[str, Any],
        provider: Callable[[Mapping[str, Any]], Any],
        message_builder: Callable[..., list[dict[str, str]]] = build_model_messages,
        transport_attempts: int | None = None,
        retry_count: int = MODEL_CALL_RETRY_COUNT,
        retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
        memory_enabled: bool | None = None,
        prior_manifest_path: str | Path | None = None,
        retry_failed_operation: bool = False,
        observer: ModelStageObserver | None = None,
        display_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.task_root = Path(task_root).resolve(strict=False)
        if operation not in _OPERATION_DIRECTORIES:
            raise ModelOperationError(f"operation 未注册：{operation}")
        if (
            not isinstance(operation_instance_id, str)
            or not _IDENTITY_PATTERN.fullmatch(operation_instance_id)
        ):
            raise ModelOperationError("operation_instance_id 不是安全身份")
        if not isinstance(base_input, Mapping):
            raise ModelOperationError("base_input 必须是 JSON object")
        if not callable(provider) or not callable(message_builder):
            raise ModelOperationError("provider 与 message_builder 必须可调用")
        if transport_attempts is not None and (
            not isinstance(transport_attempts, int) or transport_attempts <= 0
        ):
            raise ModelOperationError("transport_attempts 必须是正整数或 null")
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
        ):
            raise ModelOperationError("retry_count 必须是非负整数")
        if (
            isinstance(retry_delay, bool)
            or not isinstance(retry_delay, (int, float))
            or not math.isfinite(float(retry_delay))
            or retry_delay < 0
        ):
            raise ModelOperationError("retry_delay 必须是非负数字")
        if memory_enabled is not None and not isinstance(memory_enabled, bool):
            raise ModelOperationError("memory_enabled 必须是布尔值或 null")
        self.operation_instance_id = operation_instance_id
        self.operation = operation
        self.base_input = deepcopy(dict(base_input))
        self.provider = provider
        self.message_builder = message_builder
        # 兼容历史 transport_attempts（总尝试次数）；新公开参数 retry_count
        # 明确表示首次失败后的额外重试次数。
        self.retry_count = (
            max(transport_attempts - 1, 0)
            if transport_attempts is not None
            else retry_count
        )
        self.retry_delay = float(retry_delay)
        self.memory_enabled = memory_enabled
        self.retry_failed_operation_flag = bool(retry_failed_operation)
        if display_context is not None and not isinstance(display_context, Mapping):
            raise ModelOperationError("display_context 必须是 object 或 null")
        self.observer = observer
        self.display_context = deepcopy(
            dict(display_context) if display_context is not None else {}
        )
        self._observer_disabled = False
        self._observer_degraded_reported = False
        self._pending_stage_observations: list[
            tuple[_ModelStageObservation, str, Any, str | None]
        ] = []
        self.operation_dir = safe_join_task_path(
            self.task_root,
            PurePosixPath(
                "working_memory",
                _OPERATION_DIRECTORIES[operation],
                operation_instance_id,
            ).as_posix(),
        )
        self.prior_manifest_path = (
            self._absolute_unresolved_path(prior_manifest_path)
            if prior_manifest_path is not None
            else None
        )
        self._validated_prefix: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def _disable_model_stage_observer(self, method_name: str) -> None:
        """隔离观察器异常，并尽力通知其旁路输出已经降级。

        观察器属于可选显示能力。无论回调本身还是降级通知失败，都只记录 warning，
        不会把异常带回正式模型操作，也不会修改 manifest、重试次数或返回值。
        """

        if self._observer_degraded_reported:
            return
        self._observer_degraded_reported = True
        _LOGGER.warning("模型阶段观察器回调 %s 失败，旁路输出已降级", method_name)
        observer = self.observer
        marker = getattr(observer, "mark_model_output_degraded", None)
        if not callable(marker):
            return
        try:
            marker("model stage observer callback failed")
        except TypeError:
            try:
                marker()
            except BaseException:
                _LOGGER.warning("模型阶段观察器降级通知失败", exc_info=True)
        except BaseException:
            _LOGGER.warning("模型阶段观察器降级通知失败", exc_info=True)

    def _call_model_stage_observer(
        self,
        method_name: str,
        *arguments: Any,
    ) -> Any | None:
        """以深拷贝调用观察器；任何观察器异常都被隔离。"""

        observer = self.observer
        if observer is None or self._observer_disabled:
            return None
        method = getattr(observer, method_name, None)
        if not callable(method):
            self._observer_disabled = True
            self._disable_model_stage_observer(method_name)
            return None
        try:
            # handle 是观察器自己返回的关联对象，不能猜测其是否可深拷贝；事件仍
            # 每次都复制，观察器修改事件不会污染执行器内部状态。
            if method_name == "start_model_stage":
                return method(deepcopy(arguments[0]))
            return method(arguments[0], deepcopy(arguments[1]))
        except BaseException:
            self._observer_disabled = True
            self._disable_model_stage_observer(method_name)
            return None

    def _build_stage_observation_event(
        self,
        *,
        stage: ModelStage,
        stage_index: int,
        stage_input: Mapping[str, Any],
        started_at: str,
        finished_at: str | None,
        duration_ms: int | None,
        event_type: str,
        display_context: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """构造独立旁路事件，保留原始阶段名称和调用上下文。"""

        event = {
            "event_type": event_type,
            "operation": self.operation,
            "operation_id": self.operation_instance_id,
            "stage_index": stage_index,
            "raw_stage": stage.stage,
            "stage": stage.stage,
            "stage_kind": stage.stage_kind,
            "query_index": stage.query_index,
            "validation_kind": stage.validation_kind,
            "domain_validator_id": stage.domain_validator_id,
            "response_contract": {
                "validation_kind": stage.validation_kind,
                "schema": (
                    deepcopy(dict(stage.schema))
                    if stage.schema is not None
                    else None
                ),
                "domain_validator_id": stage.domain_validator_id,
            },
            "stage_input": deepcopy(dict(stage_input)),
            "display_context": deepcopy(
                self.display_context if display_context is None else display_context
            ),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
        }
        event.update(deepcopy(fields))
        return event

    def _start_model_stage_observation(
        self,
        *,
        stage: ModelStage,
        stage_index: int,
        stage_input: Mapping[str, Any],
    ) -> _ModelStageObservation | None:
        """为真实模型阶段分配观察句柄；不为恢复重放制造事件。"""

        if self.observer is None or self._observer_disabled:
            return None
        started_at = _utc_timestamp()
        observation = _ModelStageObservation(
            handle=None,
            started_at=started_at,
            started_clock=monotonic(),
            stage=stage,
            stage_index=stage_index,
            stage_input=deepcopy(dict(stage_input)),
            display_context=deepcopy(self.display_context),
        )
        event = self._build_stage_observation_event(
            stage=stage,
            stage_index=stage_index,
            stage_input=observation.stage_input,
            started_at=started_at,
            finished_at=None,
            duration_ms=None,
            event_type="model_stage_started",
            display_context=observation.display_context,
            source_artifact_refs={},
            source_artifacts={},
        )
        observation.handle = self._call_model_stage_observer(
            "start_model_stage",
            event,
        )
        return observation

    def _record_model_attempt_observation(
        self,
        observation: _ModelStageObservation | None,
        *,
        attempt_index: int,
        attempt_started_at: str,
        attempt_started_clock: float,
        request: Mapping[str, Any],
        raw_response: Any,
        parsed_output: Any,
        validation: Mapping[str, Any],
        source_artifact_refs: Mapping[str, Any],
        provider_trace_id: str | None,
    ) -> None:
        """在正式 attempt 收据写入后，把一次逻辑回答复制给观察器。"""

        if observation is None or self._observer_disabled:
            return
        finished_at = _utc_timestamp()
        event = self._build_stage_observation_event(
            stage=observation.stage,
            stage_index=observation.stage_index,
            stage_input=observation.stage_input,
            started_at=attempt_started_at,
            finished_at=finished_at,
            duration_ms=max(0, int(round((monotonic() - attempt_started_clock) * 1000))),
            event_type="model_attempt_recorded",
            display_context=observation.display_context,
            attempt_index=attempt_index,
            compiled_request=deepcopy(dict(request)),
            raw_response=deepcopy(raw_response),
            parsed_output=deepcopy(parsed_output),
            validation=deepcopy(dict(validation)),
            source_artifact_refs=deepcopy(dict(source_artifact_refs)),
            provider_trace_id=provider_trace_id,
        )
        observation.last_attempt = deepcopy(event)
        observation.attempt_source_refs.append(
            deepcopy(dict(source_artifact_refs))
        )
        self._call_model_stage_observer(
            "record_model_attempt",
            observation.handle,
            event,
        )

    @staticmethod
    def _validated_model_stage_output(
        observation: _ModelStageObservation | None,
    ) -> Any:
        """从已记录的有效回答中取出正式阶段最终值。"""

        if observation is None or not isinstance(observation.last_attempt, Mapping):
            return None
        validation = observation.last_attempt.get("validation")
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            return None
        return deepcopy(observation.last_attempt.get("parsed_output"))

    def _queue_model_stage_observation(
        self,
        observation: _ModelStageObservation | None,
        *,
        status: str,
        final_output: Any = None,
        failure_kind: str | None = None,
    ) -> None:
        """暂存阶段终态，等待 finalized manifest 生成后再通知观察器。"""

        if observation is None:
            return
        self._pending_stage_observations.append(
            (
                observation,
                status,
                deepcopy(final_output),
                failure_kind,
            )
        )

    def _finish_pending_model_stage_observations(
        self,
        *,
        manifest_ref: Mapping[str, Any],
        base_input_ref: Mapping[str, Any],
    ) -> None:
        """在 finalized manifest 已固定后一次性写出所有阶段终态事件。"""

        if self._observer_disabled:
            self._pending_stage_observations.clear()
            return
        pending = self._pending_stage_observations
        self._pending_stage_observations = []
        for observation, status, final_output, failure_kind in pending:
            self._emit_model_stage_finished(
                observation,
                status=status,
                final_output=final_output,
                failure_kind=failure_kind,
                manifest_ref=manifest_ref,
                base_input_ref=base_input_ref,
            )

    def _emit_model_stage_finished(
        self,
        observation: _ModelStageObservation,
        *,
        status: str,
        final_output: Any,
        failure_kind: str | None,
        manifest_ref: Mapping[str, Any],
        base_input_ref: Mapping[str, Any],
    ) -> None:
        """把带 finalized manifest 引用的终态事件发送给观察器。"""

        if self._observer_disabled:
            return
        finished_at = _utc_timestamp()
        last_attempt = observation.last_attempt or {}
        validation = last_attempt.get("validation", {})
        all_attempt_refs = deepcopy(observation.attempt_source_refs)
        request_files = [
            refs["request_ref"]
            for refs in all_attempt_refs
            if refs.get("request_ref") is not None
        ]
        response_files = [
            refs["response_ref"]
            for refs in all_attempt_refs
            if refs.get("response_ref") is not None
        ]
        error_files = [
            refs["error_ref"]
            for refs in all_attempt_refs
            if refs.get("error_ref") is not None
        ]
        attempt_refs = deepcopy(last_attempt.get("source_artifact_refs", {}))
        source_artifacts = {
            "base_input": deepcopy(dict(base_input_ref)),
            "manifest": deepcopy(dict(manifest_ref)),
            "base_input_ref": deepcopy(dict(base_input_ref)),
            "manifest_ref": deepcopy(dict(manifest_ref)),
            "attempt": attempt_refs,
            "attempts": all_attempt_refs,
            "request_files": request_files,
            "response_files": response_files,
            "error_files": error_files,
        }
        event = self._build_stage_observation_event(
            stage=observation.stage,
            stage_index=observation.stage_index,
            stage_input=observation.stage_input,
            started_at=observation.started_at,
            finished_at=finished_at,
            duration_ms=max(0, int(round((monotonic() - observation.started_clock) * 1000))),
            event_type="model_stage_finished",
            display_context=observation.display_context,
            status=status,
            final_output=deepcopy(final_output) if status == "validated" else None,
            failure_kind=failure_kind,
            attempt_count=len(all_attempt_refs),
            validation=deepcopy(validation),
            source_artifact_refs=source_artifacts,
            source_artifacts=deepcopy(source_artifacts),
        )
        self._call_model_stage_observer(
            "finish_model_stage",
            observation.handle,
            event,
        )

    def _absolute_unresolved_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.task_root / path

    def _artifact_ref(self, path: Path, media_type: str) -> dict[str, Any]:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(self.task_root).as_posix()
        except ValueError as exc:
            raise ModelOperationError("artifact 路径越过 task root") from exc
        _validate_relative_path(relative, "artifact relative_path")
        payload = resolved.read_bytes()
        return {
            "relative_path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "media_type": media_type,
            "size": len(payload),
        }

    def _write_json_artifact(
        self, directory: Path, stem: str, value: Any
    ) -> dict[str, Any]:
        payload = _canonical_bytes(value)
        digest = hashlib.sha256(payload).hexdigest()
        path = directory / f"{stem}_{digest}.json"
        if path.exists() and path.read_bytes() != payload:
            raise ModelOperationError("content-addressed artifact 内容冲突")
        if not path.exists():
            _atomic_write(path, payload)
        return self._artifact_ref(path, "application/json")

    def _write_base_input(self) -> dict[str, Any]:
        payload = _canonical_bytes(self.base_input)
        path = self.operation_dir / "base_input.json"
        if path.exists() and path.read_bytes() != payload:
            raise ModelOperationError("同一 operation 的 base_input 发生变化")
        _atomic_write(path, payload)
        return self._artifact_ref(path, "application/json")

    def _write_staging(self, manifest: Mapping[str, Any]) -> None:
        _atomic_write(
            self.operation_dir / "manifest.staging.json",
            _canonical_bytes(manifest),
        )

    def _finalize_manifest(
        self, manifest: Mapping[str, Any]
    ) -> tuple[Path, dict[str, Any]]:
        payload = _canonical_bytes(manifest)
        digest = hashlib.sha256(payload).hexdigest()
        path = self.operation_dir / f"manifest_{digest}.json"
        if path.exists() and path.read_bytes() != payload:
            raise ModelOperationError("finalized manifest hash 冲突")
        if not path.exists():
            _atomic_write(path, payload)
        return path, self._artifact_ref(path, "application/json")

    @staticmethod
    def _read_finalized_manifest(root: Path, path: Path) -> dict[str, Any]:
        candidate = path if path.is_absolute() else root / path
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ModelOperationError("prior manifest 越过 task root") from exc
        _validate_relative_path(relative, "prior manifest")
        path = safe_join_task_path(root, relative)
        if path.is_symlink() or not path.is_file():
            raise ModelOperationError("prior manifest 必须是普通 finalized 文件")
        match = re.fullmatch(r"manifest_([0-9a-f]{64})\.json", path.name)
        if match is None:
            raise ModelOperationError("prior manifest 文件名不含内容 hash")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != match.group(1):
            raise ModelOperationError("prior manifest hash 校验失败")
        try:
            manifest = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ModelOperationError("prior manifest 不是严格 JSON") from exc
        if _canonical_bytes(manifest) != payload:
            raise ModelOperationError("prior manifest 不是 canonical JSON")
        if not isinstance(manifest, dict):
            raise ModelOperationError("prior manifest 根必须是对象")
        _assert_exact_keys(manifest, _MANIFEST_KEYS, "prior manifest")
        return manifest

    @staticmethod
    def _read_ref_bytes_static(root: Path, ref: Mapping[str, Any]) -> bytes:
        if not isinstance(ref, Mapping):
            raise ModelOperationError("artifact ref 必须是对象")
        _assert_exact_keys(ref, _REF_KEYS, "artifact ref")
        relative = _validate_relative_path(ref["relative_path"], "artifact ref")
        if (
            not isinstance(ref["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
            or not isinstance(ref["media_type"], str)
            or not ref["media_type"]
            or not isinstance(ref["size"], int)
            or isinstance(ref["size"], bool)
            or ref["size"] < 0
        ):
            raise ModelOperationError("artifact ref 元数据非法")
        path = safe_join_task_path(root, relative.as_posix())
        if path.is_symlink() or not path.is_file():
            raise ModelOperationError("artifact ref 指向非普通文件")
        payload = path.read_bytes()
        if len(payload) != ref["size"]:
            raise ModelOperationError("artifact ref size 校验失败")
        if hashlib.sha256(payload).hexdigest() != ref["sha256"]:
            raise ModelOperationError("artifact ref hash 校验失败")
        return payload

    @staticmethod
    def _read_ref_json_static(root: Path, ref: Mapping[str, Any]) -> Any:
        if ref.get("media_type") != "application/json":
            raise ModelOperationError("目标 artifact 不是 application/json")
        payload = _ModelStageExecutor._read_ref_bytes_static(root, ref)
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ModelOperationError("JSON artifact 无法解析") from exc
        if _canonical_bytes(value) != payload:
            raise ModelOperationError("JSON artifact 不是 canonical JSON")
        return value

    def _validate_attempt_artifacts(self, attempt: Mapping[str, Any]) -> None:
        _assert_exact_keys(attempt, _ATTEMPT_KEYS, "prior attempt")
        self._read_ref_json_static(self.task_root, attempt["request_ref"])
        status = attempt["validation_status"]
        if status not in {"valid", "invalid", "transport_failed"}:
            raise ModelOperationError("prior attempt validation_status 非法")
        response_ref = attempt["response_ref"]
        error_ref = attempt["error_ref"]
        if status == "valid":
            if response_ref is None or error_ref is not None:
                raise ModelOperationError("prior valid attempt ref 组合非法")
            self._read_ref_json_static(self.task_root, response_ref)
        elif status == "invalid":
            if response_ref is None or error_ref is None:
                raise ModelOperationError("prior invalid attempt ref 组合非法")
            self._read_ref_json_static(self.task_root, response_ref)
            self._validate_error_artifact(error_ref)
        else:
            if response_ref is not None or error_ref is None:
                raise ModelOperationError("prior transport attempt ref 组合非法")
            self._validate_error_artifact(error_ref)
        trace_id = attempt["provider_trace_id"]
        if trace_id is not None and not isinstance(trace_id, str):
            raise ModelOperationError("provider_trace_id 必须是字符串或 null")

    def _validate_error_artifact(self, ref: Mapping[str, Any]) -> dict[str, Any]:
        error = self._read_ref_json_static(self.task_root, ref)
        if not isinstance(error, dict):
            raise ModelOperationError("error artifact 根必须是对象")
        _assert_exact_keys(
            error, {"failure_kind", "finish_reason", "summary"}, "error artifact"
        )
        if error["failure_kind"] not in FAILURE_KINDS:
            raise ModelOperationError("error artifact failure_kind 非法")
        if error["finish_reason"] is not None and not isinstance(
            error["finish_reason"], str
        ):
            raise ModelOperationError("error artifact finish_reason 非法")
        if not isinstance(error["summary"], str) or not error["summary"]:
            raise ModelOperationError("error artifact summary 不能为空")
        return error

    def _build_stage_messages(
        self,
        stage: ModelStage,
        stage_input: Mapping[str, Any],
        protocol_error: Mapping[str, str] | None,
        *,
        explicit_retry_instruction: bool,
    ) -> list[dict[str, str]]:
        """根据当前 stage 输入和安全错误摘要构造并规范化模型消息。

        输入不会被修改；返回值只包含合法的 ``role`` 和 ``content`` 字段。输入
        不是对象、消息构造器返回空数组或消息外壳不完整时，抛出
        ``ModelOperationError``，不会发出模型请求。
        """

        if not isinstance(stage_input, Mapping):
            raise ModelOperationError("model stage input 必须是 object")
        prompt_stage = stage.stage if self.operation == "hypothesis_generation" else None
        messages = self.message_builder(
            self.operation,
            deepcopy(dict(stage_input)),
            stage=prompt_stage,
            protocol_error=deepcopy(protocol_error),
            memory_enabled=self._memory_enabled_for_stage(stage),
            retry_failed_operation=explicit_retry_instruction,
            expert_fallback=False,
        )
        if not isinstance(messages, list) or not messages:
            raise ModelOperationError("message_builder 必须返回非空消息数组")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise ModelOperationError(f"messages[{index}] 字段非法")
            role = message["role"]
            content = message["content"]
            if not isinstance(role, str) or not role:
                raise ModelOperationError(f"messages[{index}].role 非法")
            if not isinstance(content, str) or not content:
                raise ModelOperationError(f"messages[{index}].content 非法")
            normalized.append({"role": role, "content": content})
        return normalized

    def _memory_enabled_for_stage(self, stage: ModelStage) -> bool | None:
        """只把 Memory 开关交给真正需要区分历史范围的模型阶段。

        一批新假设生成完以后还会继续做关系判断、合并和排名。这些比较只应读取
        科学正文；如果它们也收到开关，提示目录会把这种越界输入当成装配错误。
        """

        if self.operation == "workflow_revision":
            return self.memory_enabled
        if (
            self.operation == "hypothesis_generation"
            and stage.stage == "regeneration"
        ):
            return self.memory_enabled
        return None

    def _compile_request(
        self,
        stage: ModelStage,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """把已投影消息与当前响应校验规格固定在同一份请求中。"""

        prompt_stage = (
            stage.stage if self.operation == "hypothesis_generation" else None
        )
        return compile_model_request(
            operation=self.operation,
            stage=prompt_stage,
            messages=messages,
            response_contract={
                "validation_kind": stage.validation_kind,
                "schema": (
                    deepcopy(dict(stage.schema))
                    if stage.schema is not None
                    else None
                ),
                "domain_validator_id": stage.domain_validator_id,
            },
        )

    def _call_provider(self, request: Mapping[str, Any]) -> tuple[bool, Any]:
        def invoke_provider() -> Any:
            # compiled provider 自身也有公开 retry 参数；模型操作作为更深的
            # 业务边界时暂时接管重试，避免一次网络失败被嵌套发送十二次。
            with suppress_nested_api_retries():
                return self.provider(deepcopy(dict(request)))

        def is_retryable_provider_exception(error: BaseException) -> bool:
            # Sol Responses 适配器会标记已经收到事件的异常；这类长请求不能在
            # 上层重新提交。未收到事件的瞬时连接失败仍由 ModelOperation 边界重试。
            if (
                getattr(error, "received_event", False)
                or getattr(error, "stream_event_received", False)
                or getattr(error, "_stream_response_body_received", False)
            ):
                return False
            # 新的脱敏流失败 wrapper 已经计算了可重试事实；不再读取其原始
            # provider 异常正文或依赖原始异常允许 setattr。
            retryable_marker = getattr(error, "retryable", None)
            if isinstance(retryable_marker, bool):
                return retryable_marker
            return is_retryable_api_exception(error)

        try:
            # DeepSeek 的既有流式 workflow_generation 保持单次传输；Sol Responses
            # 由此边界管理未收到事件前的正式传输预算。
            effective_retry_count = (
                self.retry_count
                if is_codex_model(request.get("model"))
                else (0 if request.get("stream") is True else self.retry_count)
            )
            return True, retry_api_call(
                invoke_provider,
                retry_count=effective_retry_count,
                retry_delay=self.retry_delay,
                retryable_result=lambda value: is_retryable_http_status(
                    response_http_status(value)
                ),
                is_retryable_exception=is_retryable_provider_exception,
            )
        except Exception:
            return False, None

    def _run_model_stage(
        self,
        *,
        stage: ModelStage,
        stage_index: int,
        stage_input: Mapping[str, Any],
        stage_dir: Path,
        manifest: dict[str, Any],
        stage_manifest: dict[str, Any],
        observation: _ModelStageObservation | None = None,
    ) -> _ModelStageOutcome:
        """执行一个 model stage 的请求、重试和校验，并写入请求、响应和错误记录。

        输入是当前阶段规格、阶段输入、阶段目录和运行中清单。方法不负责决定
        固定或动态操作如何处理阶段结果，只更新传入的阶段清单并返回阶段值或失败类别。
        每次尝试都会先保存编译请求，再保存响应或传输错误，并立即更新运行中清单。
        内容校验失败时只把安全错误摘要带入下一次完整重答，最多重答四次；传输重试
        耗尽则立即返回失败结果，不把它当成新的模型回答。调用方决定失败后是返回
        固定操作结果，还是写入动态预留位置的结果状态并抛出阶段异常。
        """

        if observation is not None and observation.stage_index != stage_index:
            raise ModelOperationError("观察器 stage_index 与实际阶段不一致")
        existing_attempts = stage_manifest.get("attempts")
        if not isinstance(existing_attempts, list):
            raise ModelOperationError("model stage attempts 必须是数组")
        if len(existing_attempts) > PROTOCOL_RESPONSE_LIMIT:
            raise ModelOperationError("model stage attempts 超出重答上限")

        # 动态 stage 可能在上一次暂存清单写盘失败后再次调用。此时只保留已经
        # 写入清单的无效回答；从这些错误文件恢复提示，继续使用下一个连续编号，
        # 不能把已经成功写盘的错误历史重新当成首答。
        protocol_error_history: list[dict[str, str]] = []
        for expected_index, attempt in enumerate(existing_attempts):
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("attempt_index") != expected_index
            ):
                raise ModelOperationError("model stage attempt_index 不连续")
            self._validate_attempt_artifacts(attempt)
            if attempt["validation_status"] != "invalid":
                raise ModelOperationError(
                    "可重试的 model stage 只能以 invalid attempt 结束"
                )
            error = self._validate_error_artifact(attempt["error_ref"])
            protocol_error_history.append(
                {
                    "failure_kind": str(error["failure_kind"]),
                    "summary": str(error["summary"]),
                }
            )
        protocol_error = (
            _protocol_error_with_history(protocol_error_history)
            if protocol_error_history
            else None
        )
        first_attempt_index = len(existing_attempts)
        if first_attempt_index == PROTOCOL_RESPONSE_LIMIT:
            return _ModelStageOutcome(
                failure_kind=protocol_error_history[-1]["failure_kind"]
            )
        failure_kind: str | None = None
        for attempt_index in range(
            first_attempt_index,
            PROTOCOL_RESPONSE_LIMIT,
        ):
            attempt_started_at = _utc_timestamp()
            attempt_started_clock = monotonic()
            messages = self._build_stage_messages(
                stage,
                stage_input,
                protocol_error,
                explicit_retry_instruction=bool(
                    self.retry_failed_operation_flag
                ),
            )
            request = self._compile_request(stage, messages)
            request_ref = self._write_json_artifact(
                stage_dir,
                f"attempt_{attempt_index}_request",
                request,
            )
            transported, raw_response = self._call_provider(request)
            if not transported:
                error = {
                    "failure_kind": "transport",
                    "finish_reason": None,
                    "summary": _ERROR_SUMMARIES["transport"],
                }
                error_ref = self._write_json_artifact(
                    stage_dir,
                    f"attempt_{attempt_index}_error",
                    error,
                )
                attempt = {
                    "attempt_index": attempt_index,
                    "request_ref": request_ref,
                    "response_ref": None,
                    "provider_trace_id": None,
                    "error_ref": error_ref,
                    "validation_status": "transport_failed",
                }
                with self._lock:
                    stage_manifest["attempts"].append(attempt)
                    self._write_staging(manifest)
                self._record_model_attempt_observation(
                    observation,
                    attempt_index=attempt_index,
                    attempt_started_at=attempt_started_at,
                    attempt_started_clock=attempt_started_clock,
                    request=request,
                    raw_response=None,
                    parsed_output=None,
                    validation={
                        "status": "transport_failed",
                        "valid": False,
                        "failure_kind": "transport",
                        "finish_reason": None,
                        "summary": _ERROR_SUMMARIES["transport"],
                    },
                    source_artifact_refs={
                        "request_ref": request_ref,
                        "response_ref": None,
                        "error_ref": error_ref,
                    },
                    provider_trace_id=None,
                )
                return _ModelStageOutcome(failure_kind="transport")

            envelope = normalize_provider_envelope(raw_response)
            response_ref = self._write_json_artifact(
                stage_dir,
                f"attempt_{attempt_index}_response",
                envelope,
            )
            validation = validate_model_envelope(envelope, stage)
            if validation.valid:
                attempt = {
                    "attempt_index": attempt_index,
                    "request_ref": request_ref,
                    "response_ref": response_ref,
                    "provider_trace_id": self._provider_trace_id(envelope),
                    "error_ref": None,
                    "validation_status": "valid",
                }
                with self._lock:
                    stage_manifest["attempts"].append(attempt)
                    stage_manifest["status"] = "validated"
                    # 有效回答已经进入正式内存清单；即使 staging 暂存失败，后续
                    # finalize 仍会采用它，因此先保留同一回答供展示观察器记录。
                    try:
                        self._write_staging(manifest)
                    except BaseException as error:
                        staging_error = error
                    else:
                        staging_error = None
                valid_observation = {
                    "observation": observation,
                    "attempt_index": attempt_index,
                    "attempt_started_at": attempt_started_at,
                    "attempt_started_clock": attempt_started_clock,
                    "request": request,
                    "raw_response": envelope,
                    "parsed_output": validation.value,
                    "validation": {
                        "status": "valid",
                        "valid": True,
                        "failure_kind": None,
                        "finish_reason": validation.finish_reason,
                        "summary": validation.summary,
                    },
                    "source_artifact_refs": {
                        "request_ref": request_ref,
                        "response_ref": response_ref,
                        "error_ref": None,
                    },
                    "provider_trace_id": self._provider_trace_id(envelope),
                }
                self._record_model_attempt_observation(**valid_observation)
                if staging_error is not None:
                    raise staging_error
                return _ModelStageOutcome(value=validation.value)

            failure_kind = validation.failure_kind
            error = {
                "failure_kind": validation.failure_kind,
                "finish_reason": validation.finish_reason,
                "summary": validation.summary,
            }
            error_ref = self._write_json_artifact(
                stage_dir,
                f"attempt_{attempt_index}_error",
                error,
            )
            attempt = {
                "attempt_index": attempt_index,
                "request_ref": request_ref,
                "response_ref": response_ref,
                "provider_trace_id": self._provider_trace_id(envelope),
                "error_ref": error_ref,
                "validation_status": "invalid",
            }
            with self._lock:
                stage_manifest["attempts"].append(attempt)
                self._write_staging(manifest)
            self._record_model_attempt_observation(
                observation,
                attempt_index=attempt_index,
                attempt_started_at=attempt_started_at,
                attempt_started_clock=attempt_started_clock,
                request=request,
                raw_response=envelope,
                parsed_output=_extract_observer_parsed_output(
                    envelope,
                    stage,
                    validation,
                ),
                validation={
                    "status": "invalid",
                    "valid": False,
                    "failure_kind": validation.failure_kind,
                    "finish_reason": validation.finish_reason,
                    "summary": validation.summary,
                },
                source_artifact_refs={
                    "request_ref": request_ref,
                    "response_ref": response_ref,
                    "error_ref": error_ref,
                },
                provider_trace_id=self._provider_trace_id(envelope),
            )
            protocol_error_history.append(
                {
                    "failure_kind": str(validation.failure_kind),
                    "summary": str(validation.summary),
                }
            )
            protocol_error = _protocol_error_with_history(
                protocol_error_history
            )

        return _ModelStageOutcome(
            failure_kind=str(failure_kind or "domain")
        )

    @staticmethod
    def _provider_trace_id(envelope: Any) -> str | None:
        trace_id = _field(envelope, "id", None)
        return str(trace_id) if trace_id is not None else None

    def _result(
        self,
        *,
        status: str,
        values: list[Any],
        manifest: dict[str, Any],
        failure_kind: str | None,
        finish_observations: bool = True,
    ) -> dict[str, Any]:
        self._write_staging(manifest)
        manifest_path, manifest_ref = self._finalize_manifest(manifest)
        # 只有 finalized manifest 已经有稳定 hash 后才发送阶段终态，展示文件无需
        # 事后补写 source_artifacts.manifest，也不会把未冻结的失败状态当成完成。
        if finish_observations:
            self._finish_pending_model_stage_observations(
                manifest_ref=manifest_ref,
                base_input_ref=manifest["base_input_ref"],
            )
        return {
            "status": status,
            "values": deepcopy(values),
            "failure_kind": failure_kind,
            "manifest": deepcopy(manifest),
            "manifest_path": str(manifest_path),
            "manifest_ref": manifest_ref,
        }


# ==================== 固定阶段操作运行器 ====================

class ModelOperationRunner(_ModelStageExecutor):
    """依次执行固定顺序的模型步骤，并保存可复核的调用轨迹。

    构造时接收任务目录、基础输入、按顺序排列的 ``stages`` 和模型 ``provider``。
    每个阶段都接收同一份基础输入；每次尝试的请求、有效或无效响应、传输失败都会
    写入运行清单和对应文件，只有通过校验的值才会进入最终 ``values``。当前阶段
    失败时不启动后续阶段，并保留已经验证通过的前缀供显式恢复使用。阶段契约错误
    抛出 ``ModelOperationError``，文件系统异常原样向上传递。
    """

    def __init__(
        self,
        *,
        task_root: str | Path,
        operation_instance_id: str,
        operation: str,
        base_input: Mapping[str, Any],
        stages: Sequence[ModelStage],
        provider: Callable[[Mapping[str, Any]], Any],
        message_builder: Callable[..., list[dict[str, str]]] = build_model_messages,
        transport_attempts: int | None = None,
        retry_count: int = MODEL_CALL_RETRY_COUNT,
        retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
        memory_enabled: bool | None = None,
        prior_manifest_path: str | Path | None = None,
        retry_failed_operation: bool = False,
        observer: ModelStageObserver | None = None,
        display_context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            task_root=task_root,
            operation_instance_id=operation_instance_id,
            operation=operation,
            base_input=base_input,
            provider=provider,
            message_builder=message_builder,
            transport_attempts=transport_attempts,
            retry_count=retry_count,
            retry_delay=retry_delay,
            memory_enabled=memory_enabled,
            prior_manifest_path=prior_manifest_path,
            retry_failed_operation=retry_failed_operation,
            observer=observer,
            display_context=display_context,
        )
        if not isinstance(stages, Sequence) or not stages:
            raise ModelOperationError("stages 必须是非空有序序列")
        allowed_stages = _OPERATION_STAGES[operation]
        for stage in stages:
            if not isinstance(stage, ModelStage):
                raise ModelOperationError("stages item 必须是 ModelStage")
            if stage.stage not in allowed_stages:
                raise ModelOperationError(
                    f"stage {stage.stage} 不属于 operation {operation}"
                )
            if operation != "hypothesis_generation" and stage.query_index is not None:
                raise ModelOperationError("非 hypothesis operation 的 query_index 必须为 null")
        self.stages = list(stages)
        if self.prior_manifest_path is not None:
            self._load_prior_manifest()

    @classmethod
    def from_failed_manifest(
        cls,
        *,
        task_root: str | Path,
        operation_instance_id: str,
        prior_manifest_path: str | Path,
        stages: Sequence[ModelStage],
        provider: Callable[[Mapping[str, Any]], Any],
        message_builder: Callable[..., list[dict[str, str]]] = build_model_messages,
        transport_attempts: int | None = None,
        retry_count: int = MODEL_CALL_RETRY_COUNT,
        retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
        memory_enabled: bool | None = None,
        observer: ModelStageObserver | None = None,
        display_context: Mapping[str, Any] | None = None,
    ) -> "ModelOperationRunner":
        """根据指定的失败清单创建固定操作的恢复 runner。

        输入是新的操作身份、最终清单路径、当前阶段规格和 provider；构造过程会读取
        并校验清单、基础输入以及 artifact（保存的请求、响应或错误文件）的内容 hash，
        只保留首个失败阶段之前的已验证前缀，不调用 provider，也不扫描目录猜测进度。
        返回的 runner 会在后续 ``run_model_operation`` 中从首个失败阶段继续；阶段或
        hash 契约不一致时抛出 ``ModelOperationError``，路径和文件读取异常原样传播。
        """

        root = Path(task_root).resolve(strict=False)
        raw_prior = Path(prior_manifest_path)
        prior_path = raw_prior if raw_prior.is_absolute() else root / raw_prior
        manifest = cls._read_finalized_manifest(root, prior_path)
        base_input = cls._read_ref_json_static(root, manifest["base_input_ref"])
        return cls(
            task_root=root,
            operation_instance_id=operation_instance_id,
            operation=manifest["operation"],
            base_input=base_input,
            stages=stages,
            provider=provider,
            message_builder=message_builder,
            transport_attempts=transport_attempts,
            retry_count=retry_count,
            retry_delay=retry_delay,
            memory_enabled=memory_enabled,
            prior_manifest_path=prior_path,
            retry_failed_operation=True,
            observer=observer,
            display_context=display_context,
        )

    @classmethod
    def replay_manifest(
        cls,
        *,
        task_root: str | Path,
        manifest_path: str | Path,
        operation: str,
        operation_instance_id: str,
        expected_base_input: Mapping[str, Any],
        stages: Sequence[ModelStage],
    ) -> dict[str, Any]:
        """读取并复验一份已完成的固定操作清单，不重新调用模型。

        输入是任务根目录、manifest 路径、操作身份、预期基础输入和阶段规格；方法
        会逐项检查清单字段、artifact（保存的请求、响应或错误文件）及其 hash，并
        重新验证每个最终响应。它不调用 provider、不扫描目录，也不写入新的任务文件。
        成功返回包含 ``manifest``、``base_input`` 和 ``values`` 的字典；清单契约、
        身份或内容不一致时抛出 ``ModelOperationError``，路径和文件读取异常原样传播。
        """

        replayed = cls.replay_finalized_manifest(
            task_root=task_root,
            manifest_path=manifest_path,
            operation=operation,
            operation_instance_id=operation_instance_id,
            expected_base_input=expected_base_input,
            stages=stages,
        )
        if replayed["status"] != "validated":
            raise ModelOperationError("业务提交只能引用全部通过的模型清单")
        return {
            "manifest": replayed["manifest"],
            "base_input": replayed["base_input"],
            "values": replayed["values"],
        }

    @classmethod
    def replay_finalized_manifest(
        cls,
        *,
        task_root: str | Path,
        manifest_path: str | Path,
        operation: str,
        operation_instance_id: str,
        expected_base_input: Mapping[str, Any],
        stages: Sequence[ModelStage],
    ) -> dict[str, Any]:
        """严格重放已 finalized 的成功或失败固定操作清单。

        输入是任务根目录、明确的 finalized manifest 路径、冻结的操作身份、基础
        输入和阶段规格。方法逐项核对清单、请求/响应/错误 artifact、阶段顺序以及
        当前 schema/domain 校验器；不会调用 provider、扫描目录或写入任务文件。
        成功清单返回全部 ``values``，失败清单返回已经验证的前缀、最终
        ``failure_kind`` 和原始清单引用。清单不是严格 finalized 文件，或与当前
        操作契约不一致时抛出 ``ModelOperationError``。
        """

        root = Path(task_root).resolve(strict=False)
        validator = cls(
            task_root=root,
            operation_instance_id=operation_instance_id,
            operation=operation,
            base_input=expected_base_input,
            stages=stages,
            provider=lambda _request: None,
            transport_attempts=1,
        )
        raw_path = Path(manifest_path)
        candidate = raw_path if raw_path.is_absolute() else root / raw_path
        manifest = cls._read_finalized_manifest(root, candidate)
        base_input = cls._validate_replay_manifest_header(
            validator=validator,
            manifest=manifest,
            candidate=candidate,
            root=root,
            operation=operation,
            operation_instance_id=operation_instance_id,
            expected_base_input=expected_base_input,
        )
        stage_manifests = cls._validate_replay_stage_collection(
            manifest, len(stages)
        )
        values: list[Any] = []
        for stage_index, (saved, stage) in enumerate(
            zip(stage_manifests, stages), start=1
        ):
            failure = cls._replay_one_finalized_stage(
                validator=validator,
                root=root,
                candidate=candidate,
                manifest=manifest,
                base_input=base_input,
                stage_manifests=stage_manifests,
                stage_index=stage_index,
                saved=saved,
                stage=stage,
                values=values,
            )
            if failure is not None:
                return failure
        if len(stage_manifests) != len(stages):
            raise ModelOperationError("成功 finalized manifest 的 stage 数量不一致")
        return {
            "status": "validated",
            "failure_kind": None,
            "manifest": deepcopy(manifest),
            "base_input": deepcopy(base_input),
            "values": deepcopy(values),
            "manifest_path": str(candidate.resolve(strict=True)),
            "manifest_ref": validator._artifact_ref(
                candidate,
                "application/json",
            ),
        }

    @classmethod
    def _validate_replay_manifest_header(
        cls,
        *,
        validator: "ModelOperationRunner",
        manifest: Mapping[str, Any],
        candidate: Path,
        root: Path,
        operation: str,
        operation_instance_id: str,
        expected_base_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        """校验 finalized 清单身份、基础输入和继承引用。"""

        if candidate.resolve(strict=True).parent != validator.operation_dir.resolve(
            strict=False
        ):
            raise ModelOperationError("finalized manifest 不在对应 operation 目录")
        if (
            manifest["schema_version"] != 1
            or manifest["operation"] != operation
            or manifest["operation_instance_id"] != operation_instance_id
        ):
            raise ModelOperationError("finalized manifest 的操作身份不一致")
        base_input = cls._read_ref_json_static(root, manifest["base_input_ref"])
        if _canonical_bytes(base_input) != _canonical_bytes(expected_base_input):
            raise ModelOperationError("finalized manifest 的 base_input 不一致")
        prior_ref = manifest["prior_manifest_ref"]
        if prior_ref is not None:
            cls._read_ref_json_static(root, prior_ref)
        return base_input

    @staticmethod
    def _validate_replay_stage_collection(
        manifest: Mapping[str, Any], stage_count: int
    ) -> list[Mapping[str, Any]]:
        """校验 finalized 清单阶段数量，返回待重放阶段列表。"""

        stage_manifests = manifest["stages"]
        if not isinstance(stage_manifests, list) or not stage_manifests:
            raise ModelOperationError("finalized manifest 的 stage 数量非法")
        if len(stage_manifests) > stage_count:
            raise ModelOperationError("finalized manifest 的 stage 数量超过当前规格")
        return stage_manifests

    @classmethod
    def _replay_one_finalized_stage(
        cls,
        *,
        validator: "ModelOperationRunner",
        root: Path,
        candidate: Path,
        manifest: Mapping[str, Any],
        base_input: Mapping[str, Any],
        stage_manifests: Sequence[Mapping[str, Any]],
        stage_index: int,
        saved: Any,
        stage: ModelStage,
        values: list[Any],
    ) -> dict[str, Any] | None:
        """复核并重放一个 finalized stage；失败 stage 返回终态结果。"""

        if not isinstance(saved, Mapping):
            raise ModelOperationError("finalized stage 必须是对象")
        _assert_exact_keys(saved, _STAGE_KEYS, "finalized stage")
        cls._validate_replay_stage_metadata(saved, stage, stage_index)
        attempts = cls._validate_replay_attempts(validator, saved, stage)
        valid_attempts = [
            attempt
            for attempt in attempts
            if attempt["validation_status"] == "valid"
        ]
        if saved["status"] == "validated":
            values.append(
                cls._replay_validated_stage_value(
                    root=root,
                    attempts=attempts,
                    valid_attempts=valid_attempts,
                    stage=stage,
                )
            )
            return None
        return cls._replay_failed_stage_result(
            validator=validator,
            candidate=candidate,
            manifest=manifest,
            base_input=base_input,
            stage_manifests=stage_manifests,
            stage_index=stage_index,
            attempts=attempts,
            valid_attempts=valid_attempts,
            values=values,
        )

    @staticmethod
    def _validate_replay_stage_metadata(
        saved: Mapping[str, Any], stage: ModelStage, stage_index: int
    ) -> None:
        """校验 stage 的身份、状态和回答次数边界。"""

        expected_metadata = {
            "stage_index": stage_index,
            "stage": stage.stage,
            "stage_kind": stage.stage_kind,
            "query_index": stage.query_index,
            "validation_kind": stage.validation_kind,
        }
        if any(saved[field] != value for field, value in expected_metadata.items()):
            raise ModelOperationError("finalized stage 与当前校验规格不一致")
        if saved["status"] not in {"validated", "failed"}:
            raise ModelOperationError("finalized stage status 非法")

    @classmethod
    def _validate_replay_attempts(
        cls,
        validator: "ModelOperationRunner",
        saved: Mapping[str, Any],
        stage: ModelStage,
    ) -> list[Mapping[str, Any]]:
        """校验 stage 的每个 attempt、请求编译结果和 artifact hash。"""

        attempts = saved["attempts"]
        if (
            not isinstance(attempts, list)
            or not 1 <= len(attempts) <= PROTOCOL_RESPONSE_LIMIT
        ):
            raise ModelOperationError("finalized stage 的回答次数非法")
        for attempt_index, attempt in enumerate(attempts):
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("attempt_index") != attempt_index
            ):
                raise ModelOperationError("finalized attempt_index 不连续")
            validator._validate_attempt_artifacts(attempt)
            validator._validate_compiled_request_artifact(attempt["request_ref"], stage)
        return attempts

    @classmethod
    def _replay_validated_stage_value(
        cls,
        *,
        root: Path,
        attempts: Sequence[Mapping[str, Any]],
        valid_attempts: Sequence[Mapping[str, Any]],
        stage: ModelStage,
    ) -> Any:
        """重新校验成功 stage，并返回唯一最终模型值。"""

        if len(valid_attempts) != 1 or valid_attempts[0] is not attempts[-1]:
            raise ModelOperationError("finalized stage 必须以唯一有效回答结束")
        if any(attempt["validation_status"] != "invalid" for attempt in attempts[:-1]):
            raise ModelOperationError("finalized stage 的前置回答状态非法")
        envelope = cls._read_ref_json_static(root, attempts[-1]["response_ref"])
        validation = validate_model_envelope(envelope, stage)
        if not validation.valid:
            raise ModelOperationError("最终回答无法按当前规格重新验证")
        return validation.value

    @classmethod
    def _replay_failed_stage_result(
        cls,
        *,
        validator: "ModelOperationRunner",
        candidate: Path,
        manifest: Mapping[str, Any],
        base_input: Mapping[str, Any],
        stage_manifests: Sequence[Mapping[str, Any]],
        stage_index: int,
        attempts: Sequence[Mapping[str, Any]],
        valid_attempts: Sequence[Mapping[str, Any]],
        values: Sequence[Any],
    ) -> dict[str, Any]:
        """复核最终失败 stage，并返回已经验证的成功前缀。"""

        if stage_index != len(stage_manifests):
            raise ModelOperationError("失败 stage 后不应再有 finalized stage")
        final_status = attempts[-1]["validation_status"]
        if valid_attempts or final_status not in {"invalid", "transport_failed"}:
            raise ModelOperationError("failed stage 的最终回答状态非法")
        if any(attempt["validation_status"] != "invalid" for attempt in attempts[:-1]):
            raise ModelOperationError("failed stage 的前置回答状态非法")
        error = validator._validate_error_artifact(attempts[-1]["error_ref"])
        return {
            "status": "failed",
            "values": deepcopy(list(values)),
            "failure_kind": str(error["failure_kind"]),
            "manifest": deepcopy(manifest),
            "base_input": deepcopy(base_input),
            "manifest_path": str(candidate.resolve(strict=True)),
            "manifest_ref": validator._artifact_ref(candidate, "application/json"),
        }

    @classmethod
    def replay_finalized_manifest_in_operation(
        cls,
        *,
        task_root: str | Path,
        operation: str,
        operation_instance_id: str,
        expected_base_input: Mapping[str, Any],
        stages: Sequence[ModelStage],
    ) -> dict[str, Any] | None:
        """按确定的 operation 身份寻找并严格重放 finalized manifest。

        方法只读取该 operation 目录中的固定 ``manifest.staging.json``，由其内容
        hash 推导唯一 finalized 文件名，不会通过 glob、时间或“最新目录”猜测
        历史结果。目录尚未创建时返回 ``None``。若 staging 仍是请求落盘前写出的
        空清单，且目录中只有基础输入与该清单，也返回 ``None`` 允许同一 operation
        重新开始；只要已经出现请求目录或任何阶段记录，就继续严格拒绝猜测状态。
        """

        root = Path(task_root).resolve(strict=False)
        if operation not in _OPERATION_DIRECTORIES:
            raise ModelOperationError(f"operation 未注册：{operation}")
        if (
            not isinstance(operation_instance_id, str)
            or not _IDENTITY_PATTERN.fullmatch(operation_instance_id)
        ):
            raise ModelOperationError("operation_instance_id 不是安全身份")
        operation_dir = safe_join_task_path(
            root,
            PurePosixPath(
                "working_memory",
                _OPERATION_DIRECTORIES[operation],
                operation_instance_id,
            ).as_posix(),
        )
        staging = operation_dir / "manifest.staging.json"
        if not staging.exists():
            return None
        if staging.is_symlink() or not staging.is_file():
            raise ModelOperationError("manifest.staging.json 必须是普通文件")
        payload = staging.read_bytes()
        try:
            manifest = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ModelOperationError("manifest.staging.json 不是严格 JSON") from exc
        if not isinstance(manifest, dict) or _canonical_bytes(manifest) != payload:
            raise ModelOperationError("manifest.staging.json 不是 canonical JSON 对象")
        digest = hashlib.sha256(payload).hexdigest()
        finalized = operation_dir / f"manifest_{digest}.json"
        if not finalized.is_file() or finalized.is_symlink():
            if cls._is_pristine_staging_before_request(
                root=root,
                operation_dir=operation_dir,
                operation=operation,
                operation_instance_id=operation_instance_id,
                expected_base_input=expected_base_input,
                manifest=manifest,
            ):
                return None
            raise ModelOperationError(
                "operation 存在 staging manifest，但对应 finalized manifest 缺失"
            )
        return cls.replay_finalized_manifest(
            task_root=root,
            manifest_path=finalized,
            operation=operation,
            operation_instance_id=operation_instance_id,
            expected_base_input=expected_base_input,
            stages=stages,
        )

    @classmethod
    def _is_pristine_staging_before_request(
        cls,
        *,
        root: Path,
        operation_dir: Path,
        operation: str,
        operation_instance_id: str,
        expected_base_input: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> bool:
        """确认中断发生在第一个模型请求写盘之前，没有可重复的外部调用。"""

        try:
            _assert_exact_keys(manifest, _MANIFEST_KEYS, "staging manifest")
        except ModelOperationError:
            return False
        if (
            manifest.get("schema_version") != 1
            or manifest.get("operation") != operation
            or manifest.get("operation_instance_id") != operation_instance_id
            or manifest.get("prior_manifest_ref") is not None
            or manifest.get("stages") != []
        ):
            return False

        try:
            entries = list(operation_dir.iterdir())
        except OSError:
            return False
        if {entry.name for entry in entries} != {
            "base_input.json",
            "manifest.staging.json",
        }:
            return False
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            return False

        base_input_path = operation_dir / "base_input.json"
        base_payload = base_input_path.read_bytes()
        try:
            relative_path = base_input_path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            return False
        expected_ref = {
            "relative_path": relative_path.as_posix(),
            "sha256": hashlib.sha256(base_payload).hexdigest(),
            "media_type": "application/json",
            "size": len(base_payload),
        }
        return (
            manifest.get("base_input_ref") == expected_ref
            and base_payload == _canonical_bytes(expected_base_input)
        )

    def _load_prior_manifest(self) -> None:
        assert self.prior_manifest_path is not None
        manifest = self._read_finalized_manifest(
            self.task_root, self.prior_manifest_path
        )
        self._validate_fixed_prior_manifest_header(manifest)
        stages = self._read_prior_stage_list(manifest)
        prefix: list[dict[str, Any]] = []
        failed_seen = False
        for index, stage in enumerate(stages, start=1):
            stage, failed_seen = self._validate_fixed_prior_stage(
                stage,
                stage_index=index,
                failed_seen=failed_seen,
            )
            if not failed_seen or stage["status"] == "validated":
                prefix.append(deepcopy(stage))
        if not failed_seen:
            raise ModelOperationError("retry 只能引用含失败 stage 的 manifest")
        if len(prefix) >= len(self.stages):
            raise ModelOperationError("validated prefix 与当前 stage 规格不一致")
        self._validated_prefix = prefix

    def _validate_fixed_prior_manifest_header(
        self,
        manifest: Mapping[str, Any],
    ) -> None:
        """校验固定 retry 清单的身份、继承引用和基础输入。"""

        if manifest["schema_version"] != 1:
            raise ModelOperationError("prior manifest schema_version 非法")
        if manifest["operation"] != self.operation:
            raise ModelOperationError("prior manifest operation 不一致")
        prior_identity = manifest["operation_instance_id"]
        if not isinstance(prior_identity, str) or not _IDENTITY_PATTERN.fullmatch(
            prior_identity
        ):
            raise ModelOperationError("prior operation_instance_id 非法")
        if prior_identity == self.operation_instance_id:
            raise ModelOperationError("retry 必须使用新的 operation_instance_id")
        inherited_prior = manifest["prior_manifest_ref"]
        if inherited_prior is not None:
            self._read_ref_json_static(self.task_root, inherited_prior)
        prior_base = self._read_ref_json_static(
            self.task_root, manifest["base_input_ref"]
        )
        if _canonical_bytes(prior_base) != _canonical_bytes(self.base_input):
            raise ModelOperationError("retry base_input 与 prior manifest 不一致")

    @staticmethod
    def _read_prior_stage_list(
        manifest: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        """读取固定 retry 清单的非空 stage 列表。"""

        stages = manifest.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ModelOperationError("prior manifest 没有失败 stage")
        return stages

    def _validate_fixed_prior_stage(
        self,
        raw_stage: Any,
        *,
        stage_index: int,
        failed_seen: bool,
    ) -> tuple[Mapping[str, Any], bool]:
        """校验一个固定 retry stage，并返回是否已经遇到失败阶段。"""

        stage = self._validate_fixed_prior_stage_metadata(raw_stage, stage_index)
        attempts = self._validate_fixed_prior_attempts(stage)
        status = stage["status"]
        if failed_seen or status not in {"validated", "failed"}:
            raise ModelOperationError("prior manifest stage 状态顺序非法")
        if status == "validated":
            if attempts[-1]["validation_status"] != "valid":
                raise ModelOperationError("validated stage 未以 valid attempt 结束")
            return stage, False
        if attempts[-1]["validation_status"] not in {"invalid", "transport_failed"}:
            raise ModelOperationError("failed stage 的最终 attempt 非法")
        return stage, True

    def _validate_fixed_prior_stage_metadata(
        self,
        raw_stage: Any,
        stage_index: int,
    ) -> dict[str, Any]:
        """校验固定 retry stage 的身份和模型校验规格。"""

        if not isinstance(raw_stage, dict):
            raise ModelOperationError("prior manifest stage 非法")
        stage = raw_stage
        _assert_exact_keys(stage, _STAGE_KEYS, "prior stage")
        if stage["stage_index"] != stage_index:
            raise ModelOperationError("prior stage_index 不连续")
        if stage["stage"] not in _OPERATION_STAGES[self.operation]:
            raise ModelOperationError("prior stage metadata 非法")
        if stage["stage_kind"] not in _STAGE_KINDS:
            raise ModelOperationError("prior stage metadata 非法")
        if stage["validation_kind"] not in MODEL_VALIDATION_KINDS:
            raise ModelOperationError("prior stage metadata 非法")
        if not self._valid_prior_query_index(stage["query_index"]):
            raise ModelOperationError("prior stage metadata 非法")
        return stage

    def _validate_fixed_prior_attempts(
        self,
        stage: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        """校验固定 retry stage 的连续 attempts 和内容摘要。"""

        attempts = stage["attempts"]
        if not isinstance(attempts, list) or not attempts:
            raise ModelOperationError("prior stage attempts 不能为空")
        for attempt_index, attempt in enumerate(attempts):
            if (
                not isinstance(attempt, dict)
                or attempt.get("attempt_index") != attempt_index
            ):
                raise ModelOperationError("prior attempt_index 不连续")
            self._validate_attempt_artifacts(attempt)
        return attempts

    @staticmethod
    def _valid_prior_query_index(value: Any) -> bool:
        """判断旧清单 query_index 是否为空或正整数。"""

        return value is None or (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        )

    def _validate_compiled_request_artifact(
        self,
        request_ref: Mapping[str, Any],
        stage: ModelStage,
    ) -> None:
        """确认旧请求使用的提示版本、模型配置和校验规则仍与当前一致。

        compiled request 中的 ``max_tokens: null`` 是“通用模型未设置预算”的审计
        事实；这里按完整请求逐字复核，但不把它当作发送给 provider 的参数。
        """

        saved = self._read_ref_json_static(self.task_root, request_ref)
        if not isinstance(saved, Mapping) or not isinstance(saved.get("messages"), list):
            raise ModelOperationError("saved compiled request 结构非法")
        try:
            expected = self._compile_request(stage, saved["messages"])
        except (TypeError, ValueError) as exc:
            raise ModelOperationError("saved compiled request 无法按当前规格编译") from exc
        if _canonical_bytes(saved) != _canonical_bytes(expected):
            raise ModelOperationError("saved compiled request 与当前规格不一致")

    def _rehydrate_validated_prefix(self) -> list[Any]:
        values: list[Any] = []
        for index, prior_stage in enumerate(self._validated_prefix, start=1):
            spec = self.stages[index - 1]
            expected_metadata = {
                "stage_index": index,
                "stage": spec.stage,
                "stage_kind": spec.stage_kind,
                "query_index": spec.query_index,
                "validation_kind": spec.validation_kind,
            }
            for field, expected in expected_metadata.items():
                if prior_stage[field] != expected:
                    raise ModelOperationError(
                        f"validated prefix 的 {field} 与当前规格不一致"
                    )
            attempts = prior_stage["attempts"]
            if not isinstance(attempts, list) or not attempts:
                raise ModelOperationError("validated prefix 缺少 attempts")
            valid_attempts = []
            for attempt_index, attempt in enumerate(attempts):
                if not isinstance(attempt, dict):
                    raise ModelOperationError("prior attempt 必须是对象")
                if attempt.get("attempt_index") != attempt_index:
                    raise ModelOperationError("prior attempt_index 不连续")
                self._validate_attempt_artifacts(attempt)
                self._validate_compiled_request_artifact(
                    attempt["request_ref"],
                    spec,
                )
                if attempt["validation_status"] == "valid":
                    valid_attempts.append(attempt)
            if len(valid_attempts) != 1 or valid_attempts[0] is not attempts[-1]:
                raise ModelOperationError("validated prefix 必须以唯一 valid attempt 结束")
            envelope = self._read_ref_json_static(
                self.task_root, valid_attempts[0]["response_ref"]
            )
            validation = validate_model_envelope(envelope, spec)
            if not validation.valid:
                raise ModelOperationError("validated prefix 无法按当前规格重放")
            values.append(validation.value)
        return values

    def _new_manifest(self, base_input_ref: Mapping[str, Any]) -> dict[str, Any]:
        prior_ref = (
            self._artifact_ref(self.prior_manifest_path, "application/json")
            if self.prior_manifest_path is not None
            else None
        )
        return {
            "schema_version": 1,
            "operation_instance_id": self.operation_instance_id,
            "operation": self.operation,
            "base_input_ref": deepcopy(dict(base_input_ref)),
            "prior_manifest_ref": prior_ref,
            "stages": deepcopy(self._validated_prefix),
        }

    def run_model_operation(self) -> dict[str, Any]:
        """执行固定顺序的模型操作，并返回最终结果和清单位置。

        操作使用构造器提供的基础输入、阶段规格和 provider；开始时只信任明确指定的
        最终清单中逐项复验过的成功前缀，随后按声明顺序调用阶段。每个阶段都接收同一
        份基础输入，每次请求、有效或无效响应、传输失败都会立即写入任务目录，协议
        错误最多触发四次完整重答。成功返回 ``status="validated"``、``values``、
        ``failure_kind=None``、最终 ``manifest`` 和文件引用；当前阶段失败则返回
        ``status="failed"``、失败类别和已验证前缀，不启动后续阶段。阶段契约错误
        抛出 ``ModelOperationError``；provider 的普通 ``Exception`` 按传输失败重试，
        provider 的 ``BaseException``、消息构造器未处理的异常以及文件系统异常原样传播。
        """

        self.operation_dir.mkdir(parents=True, exist_ok=True)
        base_input_ref = self._write_base_input()
        # 恢复只信任明确指定的最终清单，并逐项复验其中已经通过的阶段。
        values = self._rehydrate_validated_prefix()
        manifest = self._new_manifest(base_input_ref)
        self._write_staging(manifest)
        start_index = len(self._validated_prefix)
        # 按声明顺序推进；当前阶段通过后才启动下一阶段，各阶段输入始终是 base_input。
        for offset, stage in enumerate(self.stages[start_index:], start=start_index + 1):
            stage_dir = self.operation_dir / "stages" / f"{offset}_{stage.stage}"
            stage_manifest = {
                "stage_index": offset,
                "stage": stage.stage,
                "stage_kind": stage.stage_kind,
                "query_index": stage.query_index,
                "validation_kind": stage.validation_kind,
                "attempts": [],
                "status": "failed",
            }
            manifest["stages"].append(stage_manifest)
            observation = self._start_model_stage_observation(
                stage=stage,
                stage_index=offset,
                stage_input=self.base_input,
            )
            try:
                outcome = self._run_model_stage(
                    stage=stage,
                    stage_index=offset,
                    stage_input=self.base_input,
                    stage_dir=stage_dir,
                    manifest=manifest,
                    stage_manifest=stage_manifest,
                    observation=observation,
                )
            except BaseException:
                # 若写盘异常发生在有效回答已进入内存清单之后，保留正式状态；终态
                # 仍由 _result 在 finalized manifest 生成后统一发送。
                status = (
                    "validated"
                    if stage_manifest.get("status") == "validated"
                    else "failed"
                )
                self._queue_model_stage_observation(
                    observation,
                    status=status,
                    final_output=(
                        self._validated_model_stage_output(observation)
                        if status == "validated"
                        else None
                    ),
                )
                raise
            # 任一阶段失败都会停止后续阶段，但保留此前已经验证通过的前缀。
            if outcome.failure_kind is not None:
                self._queue_model_stage_observation(
                    observation,
                    status="failed",
                    failure_kind=outcome.failure_kind,
                )
                return self._result(
                    status="failed",
                    values=values,
                    manifest=manifest,
                    failure_kind=outcome.failure_kind,
                )
            self._queue_model_stage_observation(
                observation,
                status="validated",
                final_output=outcome.value,
            )
            values.append(outcome.value)
        return self._result(
            status="validated",
            values=values,
            manifest=manifest,
            failure_kind=None,
        )


# ==================== 动态阶段操作运行器 ====================

class DynamicModelOperationRunner(_ModelStageExecutor):
    """按业务算法实际展开的顺序记录一组动态模型子调用。

    假设生成中的比较与合并数量取决于前面已经验证的模型结果，无法在调用开始前
    写成固定 stage 数组。本类允许调用方逐段预留位置，并让并行比较在各自位置写
    收据；最终清单仍使用与其他模型操作完全相同的字段和内容寻址规则。
    """

    def __init__(
        self,
        *,
        task_root: str | Path,
        operation_instance_id: str,
        operation: str,
        base_input: Mapping[str, Any],
        provider: Callable[[Mapping[str, Any]], Any],
        message_builder: Callable[..., list[dict[str, str]]] = build_model_messages,
        transport_attempts: int | None = None,
        retry_count: int = MODEL_CALL_RETRY_COUNT,
        retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
        memory_enabled: bool | None = None,
        prior_manifest_path: str | Path | None = None,
        retry_failed_operation: bool = False,
        observer: ModelStageObserver | None = None,
        display_context: Mapping[str, Any] | None = None,
    ) -> None:
        if operation != "hypothesis_generation":
            raise ModelOperationError("动态模型操作当前只用于 hypothesis_generation")
        super().__init__(
            task_root=task_root,
            operation_instance_id=operation_instance_id,
            operation=operation,
            base_input=base_input,
            provider=provider,
            message_builder=message_builder,
            transport_attempts=transport_attempts,
            retry_count=retry_count,
            retry_delay=retry_delay,
            memory_enabled=memory_enabled,
            prior_manifest_path=prior_manifest_path,
            retry_failed_operation=retry_failed_operation,
            observer=observer,
            display_context=display_context,
        )
        self._manifest: dict[str, Any] | None = None
        self._reserved_specs: dict[int, ModelStage] = {}
        self._completed_slots: set[int] = set()
        self._active_slots: set[int] = set()
        self._failed_slots: set[int] = set()
        self._final_result: dict[str, Any] | None = None
        if self.prior_manifest_path is not None:
            self._load_dynamic_prior_manifest()

    @classmethod
    def from_failed_manifest(
        cls,
        *,
        task_root: str | Path,
        operation_instance_id: str,
        prior_manifest_path: str | Path,
        provider: Callable[[Mapping[str, Any]], Any],
        message_builder: Callable[..., list[dict[str, str]]] = build_model_messages,
        transport_attempts: int | None = None,
        retry_count: int = MODEL_CALL_RETRY_COUNT,
        retry_delay: float = MODEL_CALL_RETRY_DELAY_SECONDS,
        memory_enabled: bool | None = None,
        observer: ModelStageObserver | None = None,
        display_context: Mapping[str, Any] | None = None,
    ) -> "DynamicModelOperationRunner":
        """根据指定的失败清单创建动态操作的恢复 runner。

        输入是新的操作身份、最终清单路径和 provider 等运行参数；构造过程会读取并
        校验清单、基础输入以及 artifact（保存的请求、响应或错误文件）的内容 hash，
        只保留首个失败位置之前的已验证前缀，不调用 provider，也不扫描目录猜测进度。
        返回的 runner 会在后续动态阶段调用中复用该前缀；阶段或 hash 契约不一致时抛出
        ``ModelOperationError``，路径和文件读取异常原样传播。
        """

        root = Path(task_root).resolve(strict=False)
        raw_prior = Path(prior_manifest_path)
        prior_path = raw_prior if raw_prior.is_absolute() else root / raw_prior
        manifest = cls._read_finalized_manifest(root, prior_path)
        base_input = cls._read_ref_json_static(root, manifest["base_input_ref"])
        return cls(
            task_root=root,
            operation_instance_id=operation_instance_id,
            operation=manifest["operation"],
            base_input=base_input,
            provider=provider,
            message_builder=message_builder,
            transport_attempts=transport_attempts,
            retry_count=retry_count,
            retry_delay=retry_delay,
            memory_enabled=memory_enabled,
            prior_manifest_path=prior_path,
            retry_failed_operation=True,
            observer=observer,
            display_context=display_context,
        )

    @staticmethod
    def _valid_stage_metadata(stage: Mapping[str, Any]) -> bool:
        """检查动态清单中的 stage 类型和校验器是否属于固定集合。"""

        kind = stage.get("stage_kind")
        validation_kind = stage.get("validation_kind")
        name = stage.get("stage")
        if name not in _OPERATION_STAGES["hypothesis_generation"]:
            return False
        if kind == "model":
            return validation_kind in MODEL_VALIDATION_KINDS
        if kind == "retrieval":
            return name == "retrieval" and validation_kind == "retrieval_result"
        return kind == "deterministic" and isinstance(validation_kind, str) and bool(
            validation_kind
        )

    @classmethod
    def locate_finalized_manifest_in_operation(
        cls,
        *,
        task_root: str | Path,
        operation: str,
        operation_instance_id: str,
        expected_base_input: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """按固定 operation 目录读取已 finalized 的清单元数据。

        该方法只定位同一 operation 身份的内容寻址清单，并复核 staging、清单 hash、
        base input 和全部 artifact 引用；它不会调用 provider，也不会凭目录时间或
        glob 猜测结果。调用方仍必须把返回的清单交给
        ``replay_finalized_manifest``，用当前阶段 schema/domain validator 重新检查每
        个最终回答；如果只有请求前的 pristine staging，则返回 ``None``。
        """

        root, operation_dir = cls._resolve_operation_locator_paths(
            task_root=task_root,
            operation=operation,
            operation_instance_id=operation_instance_id,
        )
        staging = operation_dir / "manifest.staging.json"
        if not staging.exists():
            return None
        manifest, payload = cls._read_canonical_staging_manifest(staging)
        digest = hashlib.sha256(payload).hexdigest()
        finalized = operation_dir / f"manifest_{digest}.json"
        if not finalized.is_file() or finalized.is_symlink():
            if cls._is_pristine_staging_before_request(
                root=root,
                operation_dir=operation_dir,
                operation=operation,
                operation_instance_id=operation_instance_id,
                expected_base_input=expected_base_input,
                manifest=manifest,
            ):
                return None
            raise ModelOperationError("对应 finalized manifest 缺失，不能安全恢复")
        finalized_manifest = cls._read_finalized_manifest(root, finalized)
        cls._validate_located_manifest_identity(
            finalized_manifest,
            operation=operation,
            operation_instance_id=operation_instance_id,
        )
        base_input = cls._read_ref_json_static(root, finalized_manifest["base_input_ref"])
        if _canonical_bytes(base_input) != _canonical_bytes(expected_base_input):
            raise ModelOperationError("finalized manifest 的 base_input 与 receipt 不一致")
        validator = cls(
            task_root=root,
            operation_instance_id=operation_instance_id,
            operation=operation,
            base_input=expected_base_input,
            provider=lambda _request: None,
            transport_attempts=1,
        )
        cls._validate_located_manifest_artifacts(validator, finalized_manifest)
        return {
            "manifest_path": str(finalized.resolve(strict=True)),
            "manifest": finalized_manifest,
            "base_input": base_input,
        }

    @classmethod
    def _resolve_operation_locator_paths(
        cls,
        *,
        task_root: str | Path,
        operation: str,
        operation_instance_id: str,
    ) -> tuple[Path, Path]:
        """解析固定 operation 目录，拒绝未登记的操作和不安全身份。"""

        root = Path(task_root).resolve(strict=False)
        if operation not in _OPERATION_DIRECTORIES:
            raise ModelOperationError(f"operation 未注册：{operation}")
        if (
            not isinstance(operation_instance_id, str)
            or not _IDENTITY_PATTERN.fullmatch(operation_instance_id)
        ):
            raise ModelOperationError("operation_instance_id 不是安全身份")
        operation_dir = safe_join_task_path(
            root,
            PurePosixPath(
                "working_memory",
                _OPERATION_DIRECTORIES[operation],
                operation_instance_id,
            ).as_posix(),
        )
        return root, operation_dir

    @staticmethod
    def _read_canonical_staging_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
        """读取请求前暂存清单并确认其字节内容是 canonical JSON。"""

        if path.is_symlink() or not path.is_file():
            raise ModelOperationError("manifest.staging.json 必须是普通文件")
        payload = path.read_bytes()
        try:
            manifest = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ModelOperationError("manifest.staging.json 不是严格 JSON") from exc
        if not isinstance(manifest, dict) or _canonical_bytes(manifest) != payload:
            raise ModelOperationError("manifest.staging.json 不是 canonical JSON 对象")
        _assert_exact_keys(manifest, _MANIFEST_KEYS, "staging manifest")
        return manifest, payload

    @staticmethod
    def _validate_located_manifest_identity(
        manifest: Mapping[str, Any],
        *,
        operation: str,
        operation_instance_id: str,
    ) -> None:
        """核对定位结果的 operation 和恢复身份。"""

        if (
            manifest.get("operation") != operation
            or manifest.get("operation_instance_id") != operation_instance_id
        ):
            raise ModelOperationError("finalized manifest 的 operation 身份不一致")

    @classmethod
    def _validate_located_manifest_artifacts(
        cls,
        validator: "DynamicModelOperationRunner",
        manifest: Mapping[str, Any],
    ) -> None:
        """复核动态 finalized 清单中的每个 stage 和 attempt artifact。"""

        for stage in manifest.get("stages", []):
            if not isinstance(stage, Mapping):
                raise ModelOperationError("finalized manifest stage 必须是对象")
            attempts = stage.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise ModelOperationError("finalized manifest stage attempts 非法")
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    raise ModelOperationError("finalized manifest attempt 必须是对象")
                validator._validate_dynamic_manifest_attempt(stage, attempt)

    def _validate_dynamic_manifest_attempt(
        self,
        stage: Mapping[str, Any],
        attempt: Mapping[str, Any],
    ) -> None:
        """按 stage 类型复核模型或动态子调用的 artifact。"""

        if stage.get("stage_kind") == "model":
            self._validate_attempt_artifacts(attempt)
        else:
            self._validate_dynamic_attempt(stage, attempt)

    def _validate_dynamic_attempt(
        self,
        stage: Mapping[str, Any],
        attempt: Mapping[str, Any],
    ) -> None:
        """复核旧动态子调用的引用组合和全部内容摘要。"""

        if stage["stage_kind"] == "model":
            self._validate_attempt_artifacts(attempt)
            return
        _assert_exact_keys(attempt, _ATTEMPT_KEYS, "prior attempt")
        self._read_ref_json_static(self.task_root, attempt["request_ref"])
        status = attempt["validation_status"]
        if status not in {"valid", "invalid", "transport_failed"}:
            raise ModelOperationError("prior attempt validation_status 非法")
        response_ref = attempt["response_ref"]
        error_ref = attempt["error_ref"]
        if status == "valid":
            if response_ref is None or error_ref is not None:
                raise ModelOperationError("prior valid attempt ref 组合非法")
            response_value = self._read_ref_json_static(
                self.task_root,
                response_ref,
            )
            if stage["stage_kind"] == "retrieval":
                if self._validated_retrieval_response(response_value) is None:
                    raise ModelOperationError("prior valid Retrieval 结果非法")
        elif status == "invalid":
            if response_ref is None or error_ref is None:
                raise ModelOperationError("prior invalid attempt ref 组合非法")
            response_value = self._read_ref_json_static(
                self.task_root,
                response_ref,
            )
            if stage["stage_kind"] == "retrieval":
                self._validated_retrieval_response(response_value)
            self._validate_error_artifact(error_ref)
        else:
            if response_ref is not None or error_ref is None:
                raise ModelOperationError("prior transport attempt ref 组合非法")
            self._validate_error_artifact(error_ref)
        if attempt["provider_trace_id"] is not None:
            raise ModelOperationError("非模型 stage 不得保存 provider trace")

    def _load_dynamic_prior_manifest(self) -> None:
        """验证旧清单并仅保留首个失败位置之前的连续成功前缀。"""

        assert self.prior_manifest_path is not None
        manifest = self._read_finalized_manifest(
            self.task_root,
            self.prior_manifest_path,
        )
        self._validate_dynamic_prior_manifest_header(manifest)
        stages = self._read_dynamic_prior_stage_list(manifest)
        prefix: list[dict[str, Any]] = []
        failed_seen = False
        for index, raw_stage in enumerate(stages, start=1):
            stage, failed_seen = self._validate_dynamic_prior_stage(
                raw_stage,
                stage_index=index,
                failed_seen=failed_seen,
            )
            if stage["status"] == "validated" and not failed_seen:
                prefix.append(deepcopy(stage))
        if not failed_seen:
            raise ModelOperationError("retry 只能引用含失败 stage 的 manifest")
        self._validated_prefix = prefix

    def _validate_dynamic_prior_manifest_header(
        self,
        manifest: Mapping[str, Any],
    ) -> None:
        """校验动态 retry 清单的身份、继承引用和基础输入。"""

        if manifest["schema_version"] != 1 or manifest["operation"] != self.operation:
            raise ModelOperationError("prior manifest 版本或 operation 不一致")
        prior_identity = manifest["operation_instance_id"]
        if (
            not isinstance(prior_identity, str)
            or not _IDENTITY_PATTERN.fullmatch(prior_identity)
            or prior_identity == self.operation_instance_id
        ):
            raise ModelOperationError("retry 必须使用新的合法 operation identity")
        inherited = manifest["prior_manifest_ref"]
        if inherited is not None:
            self._read_ref_json_static(self.task_root, inherited)
        prior_base = self._read_ref_json_static(
            self.task_root,
            manifest["base_input_ref"],
        )
        if _canonical_bytes(prior_base) != _canonical_bytes(self.base_input):
            raise ModelOperationError("retry base_input 与 prior manifest 不一致")

    @staticmethod
    def _read_dynamic_prior_stage_list(
        manifest: Mapping[str, Any],
    ) -> list[Any]:
        """读取动态 retry 清单的非空 stage 列表。"""

        stages = manifest.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ModelOperationError("prior manifest 没有可恢复 stage")
        return stages

    def _validate_dynamic_prior_stage(
        self,
        raw_stage: Any,
        *,
        stage_index: int,
        failed_seen: bool,
    ) -> tuple[dict[str, Any], bool]:
        """校验动态 retry stage 及其 attempts，并返回失败状态。"""

        stage = self._validate_dynamic_prior_stage_metadata(raw_stage, stage_index)
        attempts = self._validate_dynamic_prior_attempts(stage)
        if stage["status"] == "validated":
            if attempts[-1]["validation_status"] != "valid":
                raise ModelOperationError("validated stage 未以 valid attempt 结束")
            # 并行动态 stage 可能在首个失败之后仍已完成；这类 stage 可以
            # 继续校验，但不能越过失败位置进入可恢复的连续成功前缀。
            return stage, failed_seen
        if attempts[-1]["validation_status"] not in {"invalid", "transport_failed"}:
            raise ModelOperationError("failed stage 的最终 attempt 非法")
        return stage, True

    def _validate_dynamic_prior_stage_metadata(
        self,
        raw_stage: Any,
        stage_index: int,
    ) -> dict[str, Any]:
        """校验动态 retry stage 的身份、类型和 query_index。"""

        if not isinstance(raw_stage, Mapping):
            raise ModelOperationError("prior manifest stage 非法")
        stage = dict(raw_stage)
        _assert_exact_keys(stage, _STAGE_KEYS, "prior stage")
        if stage["stage_index"] != stage_index:
            raise ModelOperationError("prior stage metadata 非法")
        if not self._valid_stage_metadata(stage):
            raise ModelOperationError("prior stage metadata 非法")
        query_index = stage["query_index"]
        if query_index is not None and (
            isinstance(query_index, bool)
            or not isinstance(query_index, int)
            or query_index <= 0
        ):
            raise ModelOperationError("prior stage query_index 非法")
        return stage

    def _validate_dynamic_prior_attempts(
        self,
        stage: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        """校验动态 retry stage 的连续 attempts 和对应 artifact。"""

        attempts = stage["attempts"]
        if not isinstance(attempts, list) or not attempts:
            raise ModelOperationError("prior stage attempts 不能为空")
        for attempt_index, attempt in enumerate(attempts):
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("attempt_index") != attempt_index
            ):
                raise ModelOperationError("prior attempt_index 不连续")
            self._validate_dynamic_attempt(stage, attempt)
        return attempts

    def _ensure_started(self) -> None:
        """第一次预留 stage 时建立目录、固定输入和运行中清单。"""

        if self._manifest is not None:
            return
        if self._final_result is not None:
            raise ModelOperationError("已经冻结的 operation 不能继续追加 stage")
        self.operation_dir.mkdir(parents=True, exist_ok=True)
        base_input_ref = self._write_base_input()
        prior_ref = (
            self._artifact_ref(self.prior_manifest_path, "application/json")
            if self.prior_manifest_path is not None
            else None
        )
        self._manifest = {
            "schema_version": 1,
            "operation_instance_id": self.operation_instance_id,
            "operation": self.operation,
            "base_input_ref": base_input_ref,
            "prior_manifest_ref": prior_ref,
            "stages": [],
        }
        self._write_staging(self._manifest)

    @staticmethod
    def _stage_metadata(index: int, stage: ModelStage) -> dict[str, Any]:
        return {
            "stage_index": index,
            "stage": stage.stage,
            "stage_kind": stage.stage_kind,
            "query_index": stage.query_index,
            "validation_kind": stage.validation_kind,
        }

    def _reserve_stage(self, stage: ModelStage) -> int:
        """分配下一个稳定位置；旧成功前缀在同一位置原样复用。"""

        if stage.stage not in _OPERATION_STAGES[self.operation]:
            raise ModelOperationError("stage 不属于当前 hypothesis operation")
        with self._lock:
            self._ensure_started()
            assert self._manifest is not None
            index = len(self._manifest["stages"]) + 1
            metadata = self._stage_metadata(index, stage)
            if index <= len(self._validated_prefix):
                prior = deepcopy(self._validated_prefix[index - 1])
                if any(prior.get(field) != value for field, value in metadata.items()):
                    raise ModelOperationError("恢复时动态 stage 顺序或规格发生变化")
                stage_manifest = prior
            else:
                stage_manifest = {
                    **metadata,
                    "attempts": [],
                    "status": "failed",
                }
            self._manifest["stages"].append(stage_manifest)
            self._reserved_specs[index] = stage
            self._write_staging(self._manifest)
            return index

    def reserve_model_stages(self, stages: Sequence[ModelStage]) -> list[int]:
        """在并行提交前一次性锁定各子调用的逻辑位置。"""

        if not isinstance(stages, Sequence) or not stages:
            raise ModelOperationError("并行 stage 预留必须是非空序列")
        indexes = []
        for stage in stages:
            if not isinstance(stage, ModelStage) or stage.stage_kind != "model":
                raise ModelOperationError("只能预留 model stage")
            indexes.append(self._reserve_stage(stage))
        return indexes

    def _replay_model_prefix(
        self,
        index: int,
        stage: ModelStage,
        stage_input: Mapping[str, Any],
    ) -> Any:
        """重放旧 stage 的完整尝试链，并证明每次请求都没有变化。

        一个最终成功的 stage 可能先经历过格式错误。后一次请求会带上前一次错误，
        因此不能只拿“无错误的首答请求”去比较最后一次成功请求。这里按旧清单顺序
        恢复错误提示，并逐次核对精确请求；旧清单已经确认的失败不会因解析器升级而
        被重新解释，最后一次成功响应仍须通过当前校验。
        """

        assert self._manifest is not None
        stage_manifest = self._manifest["stages"][index - 1]
        attempts = stage_manifest["attempts"]
        protocol_error: dict[str, str] | None = None
        protocol_error_history: list[dict[str, str]] = []
        # 前缀可能来自初始 operation，也可能来自更早一次恢复。两种请求的区别由
        # 已保存的精确请求确定，不依赖目录名称或隐含的恢复次数。
        possible_retry_flags = {False, True}
        for attempt_position, attempt in enumerate(attempts):
            request = self._read_ref_json_static(
                self.task_root, attempt["request_ref"]
            )
            matching_flags: set[bool] = set()
            for retry_flag in possible_retry_flags:
                messages = self._build_stage_messages(
                    stage,
                    stage_input,
                    protocol_error,
                    explicit_retry_instruction=retry_flag,
                )
                compiled_request = self._compile_request(stage, messages)
                if _canonical_bytes(request) == _canonical_bytes(compiled_request):
                    matching_flags.add(retry_flag)
            if not matching_flags:
                raise ModelOperationError("恢复时动态 model request 发生变化")
            possible_retry_flags = matching_flags

            validation_status = attempt["validation_status"]
            if validation_status == "invalid":
                error = self._read_ref_json_static(
                    self.task_root, attempt["error_ref"]
                )
                protocol_error_history.append(
                    {
                        "failure_kind": str(error["failure_kind"]),
                        "summary": str(error["summary"]),
                    }
                )
                protocol_error = _protocol_error_with_history(
                    protocol_error_history
                )
                continue
            if validation_status != "valid" or attempt_position != len(attempts) - 1:
                raise ModelOperationError("validated prefix 的尝试顺序非法")
            envelope = self._read_ref_json_static(
                self.task_root,
                attempt["response_ref"],
            )
            validation = validate_model_envelope(envelope, stage)
            if not validation.valid:
                raise ModelOperationError("旧成功响应无法按当前规格重放")
            return validation.value
        raise ModelOperationError("validated prefix 缺少最终成功响应")

    def run_model_stage(
        self,
        stage: ModelStage,
        stage_input: Mapping[str, Any],
    ) -> Any:
        """顺序预留并执行一个模型子调用。"""

        return self.run_reserved_model_stage(
            self._reserve_stage(stage),
            stage,
            stage_input,
        )

    def run_reserved_model_stage(
        self,
        stage_index: int,
        stage: ModelStage,
        stage_input: Mapping[str, Any],
    ) -> Any:
        """在预留位置执行模型调用，并阻止同一预留位置并发或重复启动。

        调用前先标记预留位置正在执行，成功时写入完成标记，协议失败时写入失败
        标记。任何异常都会释放执行标记；如果有效结果已经进入本次执行的内存阶段
        清单，则保留完成标记防止重复调用，否则不伪造完成或失败状态，允许再次显式
        调用。
        """

        if not isinstance(stage, ModelStage) or stage.stage_kind != "model":
            raise ModelOperationError("run_reserved_model_stage 只接受 model stage")
        reused = self._claim_reserved_model_slot(stage_index, stage)
        return self._execute_reserved_model_stage(
            stage_index=stage_index,
            stage=stage,
            stage_input=stage_input,
            reused=reused,
        )

    def _claim_reserved_model_slot(
        self,
        stage_index: int,
        stage: ModelStage,
    ) -> bool:
        """在锁内占用动态模型 stage，并返回是否可重放验证前缀。"""

        with self._lock:
            if self._final_result is not None:
                raise ModelOperationError("已经冻结的 operation 不能继续执行")
            if self._reserved_specs.get(stage_index) != stage:
                raise ModelOperationError("stage_index 没有绑定当前规格")
            if stage_index in self._completed_slots:
                raise ModelOperationError("同一动态 stage 不能执行两次")
            if stage_index in self._active_slots:
                raise ModelOperationError("同一动态 stage 正在执行")
            assert self._manifest is not None
            # 先占用预留位置，再释放锁等待 provider；其他线程会在这里立即被拒绝。
            self._active_slots.add(stage_index)
            return stage_index <= len(self._validated_prefix)

    def _execute_reserved_model_stage(
        self,
        *,
        stage_index: int,
        stage: ModelStage,
        stage_input: Mapping[str, Any],
        reused: bool,
    ) -> Any:
        """执行已占用的动态模型 stage，并统一处理观察和清理。"""

        stage_manifest: dict[str, Any] | None = None
        attempt_count_before: int | None = None
        baseline_status: str | None = None
        stage_call_finished = False
        observation: _ModelStageObservation | None = None
        observation_queued = False
        try:
            if reused:
                value = self._replay_model_prefix(stage_index, stage, stage_input)
                with self._lock:
                    self._active_slots.discard(stage_index)
                    self._completed_slots.add(stage_index)
                return value

            stage_dir = self.operation_dir / "stages" / f"{stage_index}_{stage.stage}"
            assert self._manifest is not None
            stage_manifest = self._manifest["stages"][stage_index - 1]
            attempt_count_before = len(stage_manifest["attempts"])
            baseline_status = stage_manifest["status"]
            observation = self._start_model_stage_observation(
                stage=stage,
                stage_index=stage_index,
                stage_input=stage_input,
            )
            outcome = self._run_model_stage(
                stage=stage,
                stage_index=stage_index,
                stage_input=stage_input,
                stage_dir=stage_dir,
                manifest=self._manifest,
                stage_manifest=stage_manifest,
                observation=observation,
            )
            # 只有完整阶段调用已经返回后，下面抛出的 ModelOperationStageError
            # 才是正常的协议失败；这时 attempts 已经写入清单，不能回滚。
            stage_call_finished = True
            # 清除占用与写入预留位置的最终状态必须是同一个临界区内的原子动作。
            with self._lock:
                self._complete_reserved_model_slot(
                    stage_index=stage_index,
                    outcome=outcome,
                    observation=observation,
                )
                observation_queued = True
            if outcome.failure_kind is None:
                return outcome.value
            raise ModelOperationStageError(stage.stage, outcome.failure_kind)
        except BaseException:
            self._cleanup_reserved_model_slot(
                stage_index=stage_index,
                reused=reused,
                stage_call_finished=stage_call_finished,
                stage_manifest=stage_manifest,
                attempt_count_before=attempt_count_before,
                baseline_status=baseline_status,
                observation=observation,
                observation_queued=observation_queued,
            )
            raise

    def _complete_reserved_model_slot(
        self,
        *,
        stage_index: int,
        outcome: Any,
        observation: _ModelStageObservation | None,
    ) -> None:
        """在同一锁内登记动态 stage 完成、失败和观察结果。"""

        self._active_slots.discard(stage_index)
        self._completed_slots.add(stage_index)
        if outcome.failure_kind is not None:
            self._failed_slots.add(stage_index)
        if outcome.failure_kind is None:
            self._queue_model_stage_observation(
                observation,
                status="validated",
                final_output=outcome.value,
            )
        else:
            self._queue_model_stage_observation(
                observation,
                status="failed",
                failure_kind=outcome.failure_kind,
            )

    def _cleanup_reserved_model_slot(
        self,
        *,
        stage_index: int,
        reused: bool,
        stage_call_finished: bool,
        stage_manifest: Mapping[str, Any] | None,
        attempt_count_before: int | None,
        baseline_status: str | None,
        observation: _ModelStageObservation | None,
        observation_queued: bool,
    ) -> None:
        """异常退出时释放占用，并只撤回未持久化的最后一次尝试。"""

        with self._lock:
            self._active_slots.discard(stage_index)
            attempts = (
                stage_manifest.get("attempts")
                if stage_manifest is not None
                else None
            )
            if self._should_keep_reserved_model_completion(
                reused=reused,
                stage_call_finished=stage_call_finished,
                stage_manifest=stage_manifest,
                attempts=attempts,
                attempt_count_before=attempt_count_before,
            ):
                self._completed_slots.add(stage_index)
                if not observation_queued:
                    self._queue_model_stage_observation(
                        observation,
                        status="validated",
                        final_output=self._validated_model_stage_output(observation),
                    )
                return
            if (
                reused
                or stage_call_finished
                or stage_manifest is None
                or attempt_count_before is None
                or baseline_status is None
            ):
                return
            if isinstance(attempts, list) and len(attempts) > attempt_count_before:
                attempts.pop()
            mutable_manifest = stage_manifest
            if isinstance(mutable_manifest, dict):
                mutable_manifest["status"] = baseline_status

    @staticmethod
    def _should_keep_reserved_model_completion(
        *,
        reused: bool,
        stage_call_finished: bool,
        stage_manifest: Mapping[str, Any] | None,
        attempts: Any,
        attempt_count_before: int | None,
    ) -> bool:
        """判断暂存写盘异常是否已经留下有效结果，需保留完成标记。"""

        return (
            not reused
            and not stage_call_finished
            and stage_manifest is not None
            and stage_manifest.get("status") == "validated"
            and isinstance(attempts, list)
            and attempt_count_before is not None
            and len(attempts) > attempt_count_before
            and bool(attempts)
            and isinstance(attempts[-1], Mapping)
            and attempts[-1].get("validation_status") == "valid"
        )

    @staticmethod
    def _validated_retrieval_result(value: Any) -> dict[str, Any] | None:
        """合法 Retrieval 必须是 object；无 error 的空 object 仍然有效。"""

        if not isinstance(value, Mapping):
            return None
        copied = deepcopy(dict(value))
        error = copied.get("error")
        if error is not None and str(error).strip():
            return None
        return copied

    def _validate_retrieval_model_call(self, value: Any) -> None:
        """复验一条 Retrieval 内部模型调用的完整任务内收据。"""

        if not isinstance(value, Mapping):
            raise ModelOperationError("Retrieval model call receipt 不是 object")
        _assert_exact_keys(
            value,
            _RETRIEVAL_MODEL_CALL_KEYS,
            "Retrieval model call receipt",
        )
        if value["stage"] not in _RETRIEVAL_MODEL_STAGES:
            raise ModelOperationError("Retrieval model call stage 未登记")
        attempt_index = value["transport_attempt_index"]
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise ModelOperationError("Retrieval transport attempt index 非法")
        self._read_ref_json_static(self.task_root, value["request_ref"])
        status = value["validation_status"]
        if status == "valid":
            if value["response_ref"] is None or value["error_ref"] is not None:
                raise ModelOperationError("Retrieval valid receipt 引用组合非法")
            response = self._read_ref_json_static(
                self.task_root,
                value["response_ref"],
            )
            if self._provider_trace_id(response) != value["provider_trace_id"]:
                raise ModelOperationError("Retrieval provider trace id 不一致")
            return
        if status == "invalid":
            if value["response_ref"] is None or value["error_ref"] is None:
                raise ModelOperationError("Retrieval invalid receipt 引用组合非法")
            response = self._read_ref_json_static(
                self.task_root,
                value["response_ref"],
            )
            if self._provider_trace_id(response) != value["provider_trace_id"]:
                raise ModelOperationError("Retrieval provider trace id 不一致")
            self._validate_error_artifact(value["error_ref"])
            return
        if status != "transport_failed":
            raise ModelOperationError("Retrieval receipt validation status 非法")
        if (
            value["response_ref"] is not None
            or value["provider_trace_id"] is not None
            or value["error_ref"] is None
        ):
            raise ModelOperationError("Retrieval transport receipt 引用组合非法")
        self._validate_error_artifact(value["error_ref"])

    def _validated_retrieval_response(
        self,
        value: Any,
    ) -> dict[str, Any] | None:
        """读取带内部模型收据的新响应，同时兼容旧任务的裸 Retrieval 结果。"""

        if isinstance(value, Mapping) and set(value) == _RETRIEVAL_RESPONSE_KEYS:
            if value.get("schema_version") != 1:
                raise ModelOperationError("Retrieval response wrapper 版本非法")
            model_calls = value.get("model_calls")
            if not isinstance(model_calls, list):
                raise ModelOperationError("Retrieval model_calls 必须是数组")
            for receipt in model_calls:
                self._validate_retrieval_model_call(receipt)
            return self._validated_retrieval_result(value.get("result"))
        return self._validated_retrieval_result(value)

    def _freeze_retrieval_model_call(
        self,
        *,
        stage_dir: Path,
        receipt_index: int,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把 Retrieval 内部一次真实传输冻结成独立、可计量的收据。"""

        expected = {
            "stage",
            "transport_attempt_index",
            "request",
            "response",
            "validation_status",
            "failure_kind",
            "summary",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ModelOperationError("Retrieval 模型调用记录字段非法")
        stage = value["stage"]
        attempt_index = value["transport_attempt_index"]
        if stage not in _RETRIEVAL_MODEL_STAGES:
            raise ModelOperationError("Retrieval 模型调用 stage 未登记")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
            or not isinstance(value["request"], Mapping)
        ):
            raise ModelOperationError("Retrieval 模型调用请求或序号非法")
        call_dir = stage_dir / "model_calls" / f"{receipt_index}_{stage}"
        request_ref = self._write_json_artifact(
            call_dir,
            "request",
            value["request"],
        )
        status = value["validation_status"]
        response = value["response"]
        response_ref = None
        provider_trace_id = None
        error_ref = None
        if status in {"valid", "invalid"}:
            if not isinstance(response, Mapping):
                raise ModelOperationError("Retrieval 内容响应不是 provider envelope")
            if status == "valid" and (
                value["failure_kind"] is not None or value["summary"] is not None
            ):
                raise ModelOperationError("Retrieval 有效响应不能携带失败说明")
            if status == "invalid" and (
                value["failure_kind"] not in FAILURE_KINDS
                or not isinstance(value["summary"], str)
                or not value["summary"]
            ):
                raise ModelOperationError("Retrieval 无效响应缺少具体失败说明")
            response_ref = self._write_json_artifact(
                call_dir,
                "response",
                response,
            )
            provider_trace_id = self._provider_trace_id(response)
            if status == "invalid":
                error_ref = self._write_json_artifact(
                    call_dir,
                    "error",
                    {
                        "failure_kind": value["failure_kind"],
                        "finish_reason": None,
                        "summary": value["summary"],
                    },
                )
        elif status == "transport_failed":
            if (
                response is not None
                or value["failure_kind"] != "transport"
                or not isinstance(value["summary"], str)
                or not value["summary"]
            ):
                raise ModelOperationError("Retrieval transport failure 记录非法")
            error_ref = self._write_json_artifact(
                call_dir,
                "error",
                {
                    "failure_kind": "transport",
                    "finish_reason": None,
                    "summary": value["summary"],
                },
            )
        else:
            raise ModelOperationError("Retrieval 模型调用状态非法")
        frozen = {
            "stage": stage,
            "transport_attempt_index": attempt_index,
            "request_ref": request_ref,
            "response_ref": response_ref,
            "provider_trace_id": provider_trace_id,
            "error_ref": error_ref,
            "validation_status": status,
        }
        self._validate_retrieval_model_call(frozen)
        return frozen

    def run_retrieval_stage(
        self,
        *,
        request: Mapping[str, Any],
        retrieve: Callable[[Callable[[Mapping[str, Any]], None]], Any],
    ) -> dict[str, Any]:
        """把首批 Retrieval 作为唯一第一个 stage 保存，失败时不做协议重答。"""

        if not isinstance(request, Mapping) or not callable(retrieve):
            raise ModelOperationError("Retrieval request 和 callable 非法")
        stage = ModelStage(
            stage="retrieval",
            stage_kind="retrieval",
            validation_kind="retrieval_result",
        )
        stage_index = self._reserve_stage(stage)
        if stage_index != 1:
            raise ModelOperationError("Retrieval 必须是首批 hypothesis operation 的 stage 1")
        request_document = {"request": deepcopy(dict(request))}
        with self._lock:
            assert self._manifest is not None
            reused = stage_index <= len(self._validated_prefix)
            stage_manifest = self._manifest["stages"][0]
        if reused:
            attempt = stage_manifest["attempts"][-1]
            old_request = self._read_ref_json_static(
                self.task_root,
                attempt["request_ref"],
            )
            if _canonical_bytes(old_request) != _canonical_bytes(request_document):
                raise ModelOperationError("恢复时 Retrieval request 发生变化")
            value = self._read_ref_json_static(
                self.task_root,
                attempt["response_ref"],
            )
            validated = self._validated_retrieval_response(value)
            if validated is None:
                raise ModelOperationError("旧 Retrieval 成功结果无法重放")
            self._completed_slots.add(stage_index)
            return validated

        stage_dir = self.operation_dir / "stages/1_retrieval"
        request_ref = self._write_json_artifact(
            stage_dir,
            "attempt_0_request",
            request_document,
        )
        model_calls: list[dict[str, Any]] = []

        def record_model_call(value: Mapping[str, Any]) -> None:
            """按实际发生顺序保存 Retrieval 内部调用，不按内容摘要去重。"""

            model_calls.append(
                self._freeze_retrieval_model_call(
                    stage_dir=stage_dir,
                    receipt_index=len(model_calls),
                    value=value,
                )
            )

        try:
            raw_result = retrieve(record_model_call)
        except Exception as exc:
            reported_failure = getattr(exc, "failure_kind", "transport")
            failure_kind = (
                reported_failure
                if reported_failure in FAILURE_KINDS
                else "transport"
            )
            error = {
                "failure_kind": failure_kind,
                "finish_reason": None,
                "summary": "Retrieval 调用未能返回结果",
            }
            error_ref = self._write_json_artifact(
                stage_dir,
                "attempt_0_error",
                error,
            )
            response_ref = None
            validation_status = "transport_failed"
            # 模型已经返回过内容时，即使随后因协议校验失败，仍把收据接入当前
            # stage，避免这份失败回答实际消耗的 token 成为无法统计的孤立文件。
            if model_calls:
                response_ref = self._write_json_artifact(
                    stage_dir,
                    "attempt_0_response",
                    {
                        "schema_version": 1,
                        "result": None,
                        "model_calls": model_calls,
                    },
                )
                validation_status = "invalid"
            attempt = {
                "attempt_index": 0,
                "request_ref": request_ref,
                "response_ref": response_ref,
                "provider_trace_id": None,
                "error_ref": error_ref,
                "validation_status": validation_status,
            }
        else:
            try:
                response_ref = self._write_json_artifact(
                    stage_dir,
                    "attempt_0_response",
                    {
                        "schema_version": 1,
                        "result": raw_result,
                        "model_calls": model_calls,
                    },
                )
            except Exception:
                raw_result = None
                response_ref = self._write_json_artifact(
                    stage_dir,
                    "attempt_0_response",
                    {
                        "schema_version": 1,
                        "result": None,
                        "model_calls": model_calls,
                    },
                )
            validated = self._validated_retrieval_result(raw_result)
            if validated is not None:
                attempt = {
                    "attempt_index": 0,
                    "request_ref": request_ref,
                    "response_ref": response_ref,
                    "provider_trace_id": None,
                    "error_ref": None,
                    "validation_status": "valid",
                }
                with self._lock:
                    stage_manifest["attempts"].append(attempt)
                    stage_manifest["status"] = "validated"
                    self._completed_slots.add(stage_index)
                    self._write_staging(self._manifest)
                return validated
            error = {
                "failure_kind": "domain",
                "finish_reason": None,
                "summary": "Retrieval 返回了不可用的根结果",
            }
            error_ref = self._write_json_artifact(
                stage_dir,
                "attempt_0_error",
                error,
            )
            attempt = {
                "attempt_index": 0,
                "request_ref": request_ref,
                "response_ref": response_ref,
                "provider_trace_id": None,
                "error_ref": error_ref,
                "validation_status": "invalid",
            }
            failure_kind = "domain"
        with self._lock:
            stage_manifest["attempts"].append(attempt)
            self._completed_slots.add(stage_index)
            self._failed_slots.add(stage_index)
            self._write_staging(self._manifest)
        raise ModelOperationStageError("retrieval", failure_kind)

    def finalize_model_operation(self) -> dict[str, Any]:
        """在所有动态阶段完成后保存最终清单并返回操作结果。

        调用前必须已经预留并完成全部阶段；方法本身不会调用 provider，只检查每个
        位置的完成或失败状态，然后把运行中的清单写入 staging 文件并生成最终清单。
        成功返回 ``status``、``values``、``failure_kind``、``manifest`` 及文件引用；
        若有失败阶段则返回 ``status="failed"`` 和首个失败类别。仍有未完成预留位置或
        清单契约错误时抛出 ``ModelOperationError``，不会生成最终清单；文件系统异常
        原样传播。
        """

        with self._lock:
            if self._final_result is not None:
                return deepcopy(self._final_result)
            self._ensure_started()
            assert self._manifest is not None
            reserved = set(self._reserved_specs)
            if not reserved or reserved != self._completed_slots:
                raise ModelOperationError("动态 operation 仍有未完成的预留 stage")
            status = "failed" if self._failed_slots else "validated"
            failure_kind = None
            if self._failed_slots:
                first_failed = min(self._failed_slots)
                attempt = self._manifest["stages"][first_failed - 1]["attempts"][-1]
                error = self._validate_error_artifact(attempt["error_ref"])
                failure_kind = error["failure_kind"]
            result = self._result(
                status=status,
                values=[],
                manifest=self._manifest,
                failure_kind=failure_kind,
                finish_observations=False,
            )
            self._final_result = deepcopy(result)
            manifest_ref = result["manifest_ref"]
            base_input_ref = result["manifest"]["base_input_ref"]
        # 不在动态 runner 的锁内调用外部观察器；观察器可能自行写文件或加锁。
        self._finish_pending_model_stage_observations(
            manifest_ref=manifest_ref,
            base_input_ref=base_input_ref,
        )
        return result
