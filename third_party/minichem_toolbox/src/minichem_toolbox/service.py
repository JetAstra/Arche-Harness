"""Validate one exact agent selection, dispatch it, and build provenance."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from pydantic import ValidationError

from .artifacts import ArtifactStore, canonicalize_artifact_refs, collect_artifact_refs
from .backends.common import (
    output_directory,
    relative_workspace_path,
    resolve_input_file,
    write_xyz,
)
from .catalog import action_specs, active_catalog_hash, backend_specs
from .idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
    scoped_submission_operation_id,
    scoped_submission_request_error,
    validate_submission_key,
)
from .models import ActionRequest, ActionResult, ResourceLimits
from .orca_contract import normalize_orca_method_basis
from .resource_budget import (
    ResourceBudgetExceeded,
    ResourceStateUnavailable,
    apply_gaussian_resource_policy,
    bind_gaussian_queue_worker,
    gaussian_execution_slot,
    fixed_job_resource_limits,
    normalize_resource_limits,
    reserve_resources,
    resource_budget_record,
    validate_resource_limits,
)
from .resources import collect_resource_references
from .runtime import invoke_worker, probe_all_backends
from .scope import current_scope
from .timeout_policy import timeout_policy_record, timeout_seconds_for


_RESULT_PAYLOAD_SEMANTIC_TYPE = "ActionResult"
_PRIMARY_FILE_REGISTRATION_FAILURE_MESSAGE = "primary 文件条目登记失败"
_PRIMARY_RESULT_REGISTRATION_FAILURE_MESSAGE = "primary ActionResult 登记失败"
_PATH_REPLACEMENT = "<path omitted>"
_MAX_PUBLIC_DIAGNOSTIC_TEXT = 4_000
_QUOTED_PATH_PATTERN = re.compile(
    r"['\"](?:(?:file://)|(?:[A-Za-z]:[\\/])|(?:\\\\)|/)[^'\"]+['\"]",
    re.IGNORECASE,
)
_FILE_URI_PATTERN = re.compile(r"file://[^\s\"'<>]+", re.IGNORECASE)
_WINDOWS_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\[^\s\\/]+[\\/][^\s\\/]+[\\/]?)[^\s\"'<>]*"
)
_POSIX_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:[^/\s\"'<>]+/)*[^/\s\"'<>]+"
)
_PRIVATE_ERROR_KEYS = frozenset(
    {
        "traceback",
        "stack",
        "stacktrace",
        "raw",
        "rawexception",
        "raw_exception",
        "exception",
        "exceptiondetails",
        "exception_details",
        "exceptionmessage",
        "exception_message",
    }
)


def _bind_gaussian_action_resources(request: ActionRequest) -> ActionRequest:
    """把 Gaussian Action 的 CPU/内存绑定到服务策略并保留其他资源字段。"""

    if request.backend_id != "gaussian":
        return request
    supplied = request.resource_limits.model_dump(mode="json")
    effective = apply_gaussian_resource_policy(supplied)
    return request.model_copy(
        update={"resource_limits": ResourceLimits.model_validate(effective)}
    )


def _bind_fixed_action_resources(
    request: ActionRequest,
    *,
    backend: Any | None = None,
    selected_backends: tuple[Any, ...] = (),
) -> ActionRequest:
    """把 Action 绑定到服务预算并服从选定 backend 的 CPU 上限。

    ``backend`` 或 ``selected_backends`` 由服务端根据已选 Catalog backend 传入；
    调用方不能提供它来改变资源决策。没有 backend 时保留完整评测预算，兼容
    需要先绑定资源的旧调用点。
    """

    effective = fixed_job_resource_limits()
    candidates = selected_backends or ((backend,) if backend is not None else ())
    maximums = [
        int((getattr(item, "resource_constraints", {}) or {})["maximum_cpu_cores"])
        for item in candidates
        if "maximum_cpu_cores"
        in (getattr(item, "resource_constraints", {}) or {})
    ]
    if maximums:
        effective["cpu_cores"] = min(effective["cpu_cores"], *maximums)

    return request.model_copy(
        update={
            "resource_limits": ResourceLimits.model_validate(effective)
        }
    )


def _record_typed_gaussian_worker(
    worker_identity: Any,
    *,
    queue_ticket: Any,
    reservation: Any,
) -> None:
    """把同一个 worker 身份写入 FIFO 和 reservation，任一写入失败都不放行恢复。"""

    if not bind_gaussian_queue_worker(queue_ticket, worker_identity):
        raise RuntimeError("typed Gaussian FIFO worker binding was lost")
    bound = reservation.bind_worker(worker_identity)
    if bound is False:
        raise RuntimeError("typed Gaussian reservation worker binding was lost")


def _is_private_error_key(normalized_key: str) -> bool:
    """识别 traceback、stack 和原始异常字段的常见序列化命名。"""

    return (
        normalized_key in _PRIVATE_ERROR_KEYS
        or "traceback" in normalized_key
        or "stacktrace" in normalized_key
        or normalized_key.startswith("stack_")
        or normalized_key.startswith("raw_exception")
        or (
            normalized_key.startswith("exception_")
            and normalized_key != "exception_type"
        )
    )


class _PrimaryArtifactRegistrationFailure(RuntimeError):
    """表示文件型 primary 产物无法登记的结构化错误。

    调用方在文件清单缺失、畸形或登记失败时抛出此异常；异常保存稳定错误码和脱敏
    细节，供上层把 Action 记录为失败，不携带原始文件内容或路径。
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class _ResultPointerRegistrationFailure(RuntimeError):
    """表示后端声明的结果指针无法安全写入结果对象。

    结果指针必须指向后端已经返回的 JSON 容器位置。指针格式错误、目标不存在、
    重复声明或对应文件登记失败时抛出此异常，避免生成缺少可交接引用的部分结果。
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class _BatchHandoffArtifactMaterializationFailure(RuntimeError):
    """表示逐项坐标产物无法从 Action 的真实结果中物化。"""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


@dataclass(frozen=True)
class _ActionExecutionPlan:
    """保存已经通过输入、选择和资源校验的 Action 执行上下文。"""

    action_id: str
    request: ActionRequest
    specification: Any
    backend_id: str
    backend: Any
    component_specs: tuple[Any, ...]
    selection_source: str
    execution_timeout: int
    timeout_policy: Any
    resource_references: Any
    input_artifacts: list[Any]


@dataclass(frozen=True)
class _WorkerExecution:
    """保存 worker 原始响应与资源预留快照，供结果封装使用。"""

    worker: Any
    resource_allocation: Any


class _WorkerResourceStateUnknown(RuntimeError):
    """worker 退出状态未知时阻止幂等收据写成 completed。"""

    def __init__(self, response: Mapping[str, Any]) -> None:
        super().__init__("worker termination state is unknown")
        self.response = dict(response)


@dataclass(frozen=True)
class _ActionSubmissionIdentity:
    """保存幂等提交所需的规范请求和服务端身份。"""

    clean_request: ActionRequest | dict[str, Any]
    operation_id: str | None
    submission_key: str | None
    in_scope: bool


def _decode_result_pointer(pointer: Any, *, artifact_index: int) -> tuple[str, ...]:
    """解析后端提供的非根 JSON Pointer，返回解码后的路径令牌。

    输入是文件条目中的原始 `result_pointer` 和该条目在 `artifact_files` 中的索引。
    函数检查指针必须是非空、以 `/` 开头的 JSON Pointer，并将 `~0`、`~1` 转义
    还原为路径令牌；函数不修改 `result_payload`、文件或 ArtifactStore，也不登记文件。
    成功时返回按层级排列的字符串令牌；格式错误时抛出带有
    `result_pointer_malformed` 错误码的 `_ResultPointerRegistrationFailure`。
    """

    if not isinstance(pointer, str) or not pointer or not pointer.startswith("/"):
        raise _ResultPointerRegistrationFailure(
            "result_pointer_malformed",
            "result_pointer 必须是以 '/' 开头的非根 JSON Pointer",
            details={"artifact_index": artifact_index},
        )
    tokens: list[str] = []
    for token in pointer[1:].split("/"):
        if "~" in token:
            if re.search(r"~[^01]", token) or token.endswith("~"):
                raise _ResultPointerRegistrationFailure(
                    "result_pointer_malformed",
                    "result_pointer 包含非法 JSON Pointer 转义",
                    details={"artifact_index": artifact_index},
                )
            token = token.replace("~1", "/").replace("~0", "~")
        tokens.append(token)
    return tuple(tokens)


def _resolve_result_pointer_target(
    result_payload: Any,
    tokens: tuple[str, ...],
    *,
    artifact_index: int,
) -> tuple[MutableMapping[str, Any] | list[Any], str | int]:
    """定位结果对象中已经存在的可写 JSON 字段或数组槽位。

    输入是原始 `result_payload`、已由 `_decode_result_pointer` 解码的路径令牌以及
    文件条目索引。函数沿映射字段和数组索引逐层检查路径，并确认最终父容器可写、
    目标字段或数组槽位已经存在；函数只读取结果对象，不修改 `result_payload`，也不
    注册或修改文件。成功时返回最终父容器及字段名或数组索引，供调用方在文件登记
    完成后写入紧凑 ArtifactRef。路径缺失或容器不可写时抛出带有
    `result_pointer_target_missing` 错误码的结构化失败；数组索引格式错误时抛出
    `result_pointer_malformed` 结构化失败。
    """

    container = result_payload
    for token in tokens[:-1]:
        if isinstance(container, Mapping):
            if token not in container:
                raise _ResultPointerRegistrationFailure(
                    "result_pointer_target_missing",
                    "result_pointer 的中间路径不存在",
                    details={"artifact_index": artifact_index},
                )
            container = container[token]
            continue
        if isinstance(container, list):
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token):
                raise _ResultPointerRegistrationFailure(
                    "result_pointer_malformed",
                    "result_pointer 的数组索引格式非法",
                    details={"artifact_index": artifact_index},
                )
            index = int(token)
            if index >= len(container):
                raise _ResultPointerRegistrationFailure(
                    "result_pointer_target_missing",
                    "result_pointer 的数组索引不存在",
                    details={"artifact_index": artifact_index},
                )
            container = container[index]
            continue
        raise _ResultPointerRegistrationFailure(
            "result_pointer_target_missing",
            "result_pointer 未指向已有 JSON 容器",
            details={"artifact_index": artifact_index},
        )

    token = tokens[-1]
    if isinstance(container, MutableMapping):
        if token not in container:
            raise _ResultPointerRegistrationFailure(
                "result_pointer_target_missing",
                "result_pointer 的目标字段不存在",
                details={"artifact_index": artifact_index},
            )
        return container, token
    if isinstance(container, list):
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token):
            raise _ResultPointerRegistrationFailure(
                "result_pointer_malformed",
                "result_pointer 的数组索引格式非法",
                details={"artifact_index": artifact_index},
            )
        index = int(token)
        if index >= len(container):
            raise _ResultPointerRegistrationFailure(
                "result_pointer_target_missing",
                "result_pointer 的数组索引不存在",
                details={"artifact_index": artifact_index},
            )
        return container, index
    raise _ResultPointerRegistrationFailure(
        "result_pointer_target_missing",
        "result_pointer 未指向可写 JSON 容器",
        details={"artifact_index": artifact_index},
    )


def _collect_result_pointer_targets(
    result_payload: Any,
    entries: list[Any],
) -> dict[int, tuple[MutableMapping[str, Any] | list[Any], str | int]]:
    """预校验文件条目中的结果指针并收集其可写目标。

    输入是后端原始 `result_payload` 和 `artifact_files` 条目列表。函数只处理带有
    `result_pointer` 的条目，调用指针解析与路径定位函数，按解码后的路径检查重复，
    并在任何文件登记前完成全部校验；函数不修改 `result_payload`，不注册或修改文件，
    也不写入 ArtifactStore。成功时返回“文件条目索引 -> 结果父容器及字段或数组索引”
    的映射，供登记成功后注入紧凑 ArtifactRef。指针格式错误、目标不存在或重复声明
    时抛出带有 `result_pointer_malformed`、`result_pointer_target_missing` 或
    `result_pointer_duplicate` 错误码的结构化 `_ResultPointerRegistrationFailure`。
    """

    targets: dict[int, tuple[MutableMapping[str, Any] | list[Any], str | int]] = {}
    seen: dict[tuple[str, ...], int] = {}
    for artifact_index, item in enumerate(entries):
        if not isinstance(item, Mapping) or "result_pointer" not in item:
            continue
        tokens = _decode_result_pointer(
            item.get("result_pointer"), artifact_index=artifact_index
        )
        previous_index = seen.get(tokens)
        if previous_index is not None:
            raise _ResultPointerRegistrationFailure(
                "result_pointer_duplicate",
                "多个文件条目声明了同一个 result_pointer",
                details={
                    "artifact_index": artifact_index,
                    "duplicate_of_artifact_index": previous_index,
                },
            )
        seen[tokens] = artifact_index
        targets[artifact_index] = _resolve_result_pointer_target(
            result_payload,
            tokens,
            artifact_index=artifact_index,
        )
    return targets


def _redact_private_paths(value: object) -> str:
    """将后端诊断文本中的本机路径替换为固定占位符。"""

    text = str(value)
    text = _QUOTED_PATH_PATTERN.sub(_PATH_REPLACEMENT, text)
    text = _FILE_URI_PATTERN.sub(_PATH_REPLACEMENT, text)
    text = _WINDOWS_PATH_PATTERN.sub(_PATH_REPLACEMENT, text)
    return _POSIX_PATH_PATTERN.sub(_PATH_REPLACEMENT, text)


def _sanitize_public_diagnostic_value(value: Any) -> Any:
    """递归保留可行动诊断字段，同时移除异常原文和本机路径。"""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.casefold().replace("-", "_")
            if _is_private_error_key(normalized_key):
                continue
            sanitized[_redact_private_paths(key)] = _sanitize_public_diagnostic_value(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_public_diagnostic_value(item) for item in value]
    if isinstance(value, str):
        redacted = _redact_private_paths(value)
        if len(redacted) <= _MAX_PUBLIC_DIAGNOSTIC_TEXT:
            return redacted
        return redacted[:_MAX_PUBLIC_DIAGNOSTIC_TEXT] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _redact_private_paths(value)


def _sanitize_worker_error(error: Any) -> dict[str, Any] | None:
    """规范化 worker 错误，确保公开 ActionResult 不含异常堆栈或原始路径。"""

    if error is None:
        return None
    if isinstance(error, Mapping):
        sanitized = _sanitize_public_diagnostic_value(error)
        return sanitized if isinstance(sanitized, dict) else None
    return {
        "code": "backend_error",
        "message": _redact_private_paths(error),
    }


def _sanitize_worker_response(worker: Any) -> dict[str, Any]:
    """在生成诊断 Artifact 前清理 worker 的错误、警告和 provenance。"""

    if not isinstance(worker, Mapping):
        return {
            "status": "failed",
            "error": {
                "code": "invalid_worker_response",
                "message": "Backend worker did not return a mapping",
            },
            "retryable": False,
        }
    sanitized = dict(worker)
    if "error" in sanitized:
        sanitized["error"] = _sanitize_worker_error(sanitized.get("error"))
    if "warnings" in sanitized:
        warnings = sanitized.get("warnings") or []
        sanitized["warnings"] = [
            _sanitize_public_diagnostic_value(item) for item in warnings
        ] if isinstance(warnings, (list, tuple)) else [
            _sanitize_public_diagnostic_value(warnings)
        ]
    if "provenance" in sanitized:
        sanitized["provenance"] = _sanitize_public_diagnostic_value(
            sanitized.get("provenance") or {}
        )
    return sanitized


def _read_xyz_frames(path: Path) -> list[str]:
    """读取一个已有多帧 XYZ 文件并按完整坐标帧切分。"""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    frames: list[str] = []
    offset = 0
    while offset < len(lines):
        if not lines[offset].strip():
            offset += 1
            continue
        try:
            atom_count = int(lines[offset].strip())
        except ValueError as exc:
            raise _BatchHandoffArtifactMaterializationFailure(
                "多帧 XYZ 的原子数行无法解析",
                details={"path": relative_workspace_path(path)},
            ) from exc
        if atom_count <= 0:
            raise _BatchHandoffArtifactMaterializationFailure(
                "多帧 XYZ 的原子数必须为正数",
                details={"path": relative_workspace_path(path)},
            )
        end = offset + atom_count + 2
        if end > len(lines):
            raise _BatchHandoffArtifactMaterializationFailure(
                "多帧 XYZ 包含不完整坐标帧",
                details={"path": relative_workspace_path(path)},
            )
        frames.append("".join(lines[offset:end]))
        offset = end
    if not frames:
        raise _BatchHandoffArtifactMaterializationFailure(
            "多帧 XYZ 没有可交接的坐标帧",
            details={"path": relative_workspace_path(path)},
        )
    return frames


def _safe_batch_item_name(item_id: str, index: int) -> str:
    """把稳定 item_id 转为不含路径分隔符的文件名片段。"""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id).strip("._")[:80]
    return f"item_{index:04d}_{safe or 'structure'}.xyz"


def _materialize_structure_batch_handoff(
    *,
    action_id: str,
    backend_id: str,
    result_payload: Any,
    artifact_files: Any,
) -> list[dict[str, Any]]:
    """为真实候选/构象结果登记逐项可读 XYZ 文件并写入结果指针。

    输入是后端已经成功产生的公开结果和文件清单；动作只复制已有坐标，给每个
    accepted candidate/conformer 写一个 XYZ 文件，并在对应结果项放置稳定的
    ``item_id`` 与 ``structure_artifact`` 占位。文件仍由统一 ArtifactStore 登记，
    因此这里不伪造 ArtifactRef，也不改变聚合结果或选择 Action 的科学字段。
    """

    if action_id not in {
        "build_reactive_complex_ensemble",
        "generate_conformer_ensemble",
    }:
        return list(artifact_files or [])
    if not isinstance(result_payload, MutableMapping):
        raise _BatchHandoffArtifactMaterializationFailure(
            "逐项坐标交接要求 Action result 是 object"
        )

    records = result_payload.get("ensemble")
    frame_blocks: list[str] | None = None
    if action_id == "generate_conformer_ensemble" and not isinstance(records, list):
        ensemble_file = result_payload.get("ensemble_file")
        if not isinstance(ensemble_file, str) or not ensemble_file:
            raise _BatchHandoffArtifactMaterializationFailure(
                "ConformerEnsemble 结果缺少逐项 ensemble 或真实 ensemble_file"
            )
        try:
            frame_blocks = _read_xyz_frames(resolve_input_file(ensemble_file))
        except _BatchHandoffArtifactMaterializationFailure:
            raise
        except (OSError, ValueError) as exc:
            raise _BatchHandoffArtifactMaterializationFailure(
                "无法读取真实 ConformerEnsemble 坐标文件",
                details={"exception_type": type(exc).__name__},
            ) from exc
        records = [
            {
                "conformer_id": str(index),
                "item_id": str(index),
                "structure_artifact": None,
            }
            for index in range(len(frame_blocks))
        ]
        result_payload["ensemble"] = records
    if not isinstance(records, list) or not records:
        raise _BatchHandoffArtifactMaterializationFailure(
            "逐项坐标交接要求非空 ensemble 数组"
        )

    entries = list(artifact_files or [])
    try:
        directory = output_directory(action_id, backend_id)
    except OSError as exc:
        raise _BatchHandoffArtifactMaterializationFailure(
            "逐项坐标输出目录创建失败",
            details={"exception_type": type(exc).__name__},
        ) from exc
    seen_item_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, MutableMapping):
            raise _BatchHandoffArtifactMaterializationFailure(
                "ensemble 项必须是 object",
                details={"item_index": index},
            )
        if action_id == "build_reactive_complex_ensemble":
            stable_id = record.get("candidate_id")
            structure = record.get("structure")
        else:
            stable_id = record.get("conformer_id", index)
            structure = record.get("structure")
        if isinstance(stable_id, bool) or not isinstance(stable_id, (str, int)):
            raise _BatchHandoffArtifactMaterializationFailure(
                "逐项结果缺少稳定 candidate_id/conformer_id",
                details={"item_index": index},
            )
        item_id = str(record.get("item_id") or stable_id)
        if not item_id:
            raise _BatchHandoffArtifactMaterializationFailure(
                "逐项结果的 item_id 不能为空",
                details={"item_index": index},
            )
        if item_id in seen_item_ids:
            raise _BatchHandoffArtifactMaterializationFailure(
                "逐项结果的 item_id 必须唯一",
                details={"item_index": index, "item_id": item_id},
            )
        seen_item_ids.add(item_id)
        record["item_id"] = item_id
        if action_id == "build_reactive_complex_ensemble" and not record.get("candidate_id"):
            record["candidate_id"] = item_id
        record.setdefault("structure_artifact", None)
        path = directory / _safe_batch_item_name(item_id, index)
        if isinstance(structure, Mapping):
            try:
                write_xyz(structure, path)
            except (OSError, ValueError, TypeError) as exc:
                raise _BatchHandoffArtifactMaterializationFailure(
                    "候选/构象没有可写出的完整坐标",
                    details={"item_index": index, "exception_type": type(exc).__name__},
                ) from exc
        elif frame_blocks is not None and index < len(frame_blocks):
            try:
                path.write_text(frame_blocks[index], encoding="utf-8")
            except OSError as exc:
                raise _BatchHandoffArtifactMaterializationFailure(
                    "逐项 XYZ 文件写入失败",
                    details={"item_index": index, "exception_type": type(exc).__name__},
                ) from exc
        else:
            raise _BatchHandoffArtifactMaterializationFailure(
                "逐项结果缺少坐标结构，且没有可切分的真实 ensemble_file",
                details={"item_index": index},
            )
        entries.append(
            {
                "path": relative_workspace_path(path),
                "semantic_type": "XYZFile",
                "media_type": "chemical/x-xyz",
                "result_pointer": f"/ensemble/{index}/structure_artifact",
            }
        )
    return entries


def _invalid(
    action_id: str,
    request_backend: str | None,
    message: str,
    *,
    code: str = "invalid_request",
    error_details: Mapping[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return ActionResult(
        status="invalid_request",
        action=action_id,
        action_version=action_specs().get(action_id).version if action_id in action_specs() else "unknown",
        requested_backend=request_backend,
        backend=None,
        selection_source="agent",
        error={"code": code, "message": message, **dict(error_details or {})},
        retryable=retryable,
        provenance={
            "catalog_hash": active_catalog_hash(),
            "automatic_fallback_count": 0,
        },
    ).model_dump(mode="json")


def _attach_safe_resource_diagnostic(
    response: dict[str, Any],
    *,
    action_id: str,
    backend_id: str | None,
) -> dict[str, Any]:
    """为资源容量拒绝登记不含容量数值的 BackendDiagnostic Artifact。"""

    error = response.get("error")
    if not isinstance(error, Mapping):
        return response
    code = str(error.get("code") or "resource_budget_exceeded")
    retryable = bool(response.get("retryable"))
    diagnostic = {
        "action": action_id,
        "backend": backend_id,
        "status": response.get("status", "invalid_request"),
        "error": {
            "code": code,
            "message": (
                "Evaluator capacity is temporarily unavailable."
                if retryable
                else "The requested resource envelope is not available."
            ),
            "retryable": retryable,
        },
    }
    try:
        reference = ArtifactStore().put_json(
            diagnostic,
            semantic_type="BackendDiagnostic",
            producer_action=action_id,
            producer_backend=backend_id,
            parent_artifact_ids=(),
        )
    except (OSError, RuntimeError, ValueError):
        return response
    response["output_artifacts"] = [reference.model_dump(mode="json")]
    return response


def _corrected_request_template(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    *,
    missing_fields: tuple[str, ...] = (),
    replacement: tuple[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a directly reusable request while leaving scientific choices explicit."""

    template = {"action_id": action_id, **request.model_dump(mode="json")}
    template.pop("resource_limits", None)
    selection_policy = action_specs()[action_id].selection_policy
    if selection_policy in {"agent_backend_required", "agent_components_required"}:
        template["backend_id"] = backend_id
    elif selection_policy == "agent_source_required":
        template["source_id"] = backend_id

    def set_path(path: str, value: Any) -> None:
        current = template
        parts = path.split(".")
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current[parts[-1]] = value

    for path in missing_fields:
        set_path(path, f"<required {path}; use inspect_action contract>")
    if replacement is not None:
        set_path(*replacement)
    return template


def _repairable_invalid(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    message: str,
    *,
    missing_fields: tuple[str, ...] = (),
    code: str = "invalid_request",
    replacement: tuple[str, Any] | None = None,
    extra_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _invalid(
        action_id,
        backend_id,
        message,
        code=code,
        retryable=True,
        error_details={
            "missing_fields": list(missing_fields),
            "repair_guidance": (
                "Correct the reported fields in this same Action/Backend request, then retry "
                "once. Do not switch providers or resubmit the unchanged request."
            ),
            "inspect_action_request": {
                "action_id": action_id,
                "backend_id": backend_id,
                "detail_level": "contract",
            },
            "corrected_request_template": _corrected_request_template(
                action_id,
                backend_id,
                request,
                missing_fields=missing_fields,
                replacement=replacement,
            ),
            **dict(extra_details or {}),
        },
    )


def _invalid_explicit_choice(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    *,
    field_group: str,
    supplied: dict[str, Any],
    allowed: Mapping[str, tuple[str, ...]],
) -> dict[str, Any] | None:
    for field_name, choices in allowed.items():
        if field_name not in supplied:
            continue
        received = supplied[field_name]
        normalized = str(received).strip().casefold()
        if normalized not in {str(choice).casefold() for choice in choices}:
            return _repairable_invalid(
                action_id,
                backend_id,
                request,
                f"Invalid {field_group}.{field_name}={received!r} for {backend_id}/{action_id}; "
                f"choose exactly one of {list(choices)}",
                code="invalid_explicit_choice",
                replacement=(
                    f"{field_group}.{field_name}",
                    f"<choose exactly one: {' | '.join(str(item) for item in choices)}>",
                ),
                extra_details={
                    "invalid_field": f"{field_group}.{field_name}",
                    "received_value": received,
                    "allowed_values": list(choices),
                },
            )
    return None


def _validate_deprecated_setting_aliases(
    action_id: str,
    backend_id: str,
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reject ambiguous requests while preserving explicit legacy aliases."""

    if (
        action_id == "calculate_electron_isodensity_surface"
        and backend_id == "multiwfn"
        and "grid_spacing_bohr" in settings
        and "grid_spacing_angstrom" in settings
    ):
        return _invalid(
            action_id,
            backend_id,
            "Supply exactly one of action_settings.grid_spacing_bohr (canonical) or "
            "action_settings.grid_spacing_angstrom (deprecated compatibility alias), not both.",
            code="conflicting_setting_aliases",
        )
    return None


def _required_setting_is_present(
    action_id: str,
    backend_id: str,
    field_name: str,
    settings: Mapping[str, Any],
) -> bool:
    if field_name in settings:
        return True
    return (
        action_id == "calculate_electron_isodensity_surface"
        and backend_id == "multiwfn"
        and field_name == "grid_spacing_bohr"
        and "grid_spacing_angstrom" in settings
    )


def _validate_orca_method_basis_contract(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
) -> dict[str, Any] | None:
    if backend_id != "orca" or "method" not in request.method_spec:
        return None
    method = request.method_spec["method"]
    basis = request.method_spec.get("basis")
    try:
        normalize_orca_method_basis(method, basis)
    except ValueError as exc:
        composite = str(method).strip().casefold().endswith("-3c")
        replacement = (
            "method_spec.basis",
            "method_default" if composite else "<concrete ORCA orbital basis>",
        )
        missing_fields = (
            ("method_spec.basis",)
            if basis is None and not composite
            else ()
        )
        return _repairable_invalid(
            action_id,
            backend_id,
            request,
            str(exc),
            code="invalid_orca_method_basis",
            missing_fields=missing_fields,
            replacement=replacement,
            extra_details={
                "invalid_field": "method_spec.basis",
                "basis_policy": (
                    "Built-in 3c composite methods omit the orbital-basis token. "
                    "All other ORCA methods require a concrete basis keyword."
                ),
            },
        )
    return None


def _validate_composite_calculator_contract(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    backends: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate nested calculator requests before launching a composite worker.

    Sella and geomeTRIC evaluate both energy and forces through the exact
    ``component_backends.calculator`` selected by the Agent.  Their nested
    settings are part of the public request contract and must not fail later as
    an opaque optimizer exception.
    """

    if backend_id not in {"sella", "geometric"}:
        return None
    calculator_id = request.component_backends.get("calculator")
    if not calculator_id:
        return None  # The ordinary required-component check reports this first.
    calculator_method = request.method_spec.get("calculator_method")
    if not isinstance(calculator_method, Mapping):
        return _invalid(
            action_id,
            backend_id,
            "method_spec.calculator_method must be a mapping containing the exact method fields "
            f"for component_backends.calculator={calculator_id!r}",
        )
    nested_settings = request.action_settings.get("calculator_action_settings")
    if not isinstance(nested_settings, Mapping):
        return _invalid(
            action_id,
            backend_id,
            "action_settings.calculator_action_settings must be an action-keyed mapping with "
            "both 'calculate_energy' and 'calculate_forces' entries",
        )
    calculator = backends[calculator_id]
    for nested_action in ("calculate_energy", "calculate_forces"):
        supplied_settings = nested_settings.get(nested_action)
        if not isinstance(supplied_settings, Mapping):
            return _invalid(
                action_id,
                backend_id,
                "action_settings.calculator_action_settings must explicitly contain mapping "
                f"{nested_action!r}; for example {{'calculate_energy': {{}}, "
                "'calculate_forces': {}}}",
            )
        if nested_action not in calculator.capabilities:
            return _invalid(
                action_id,
                backend_id,
                f"Selected calculator {calculator_id!r} does not provide {nested_action!r}",
            )
        missing_methods = [
            field_name
            for field_name in calculator.required_method_fields.get(nested_action, ())
            if field_name not in calculator_method
        ]
        missing_settings = [
            field_name
            for field_name in calculator.required_setting_fields.get(nested_action, ())
            if field_name not in supplied_settings
        ]
        if missing_methods or missing_settings:
            parts = []
            if missing_methods:
                parts.append(
                    f"method_spec.calculator_method fields {missing_methods}"
                )
            if missing_settings:
                parts.append(
                    "action_settings.calculator_action_settings"
                    f".{nested_action} fields {missing_settings}"
                )
            return _invalid(
                action_id,
                backend_id,
                f"Nested calculator contract for {calculator_id}/{nested_action} is incomplete: "
                + "; ".join(parts),
            )
        for field_name, choices in calculator.allowed_method_values.get(
            nested_action, {}
        ).items():
            if field_name in calculator_method and str(
                calculator_method[field_name]
            ).strip().casefold() not in {
                str(choice).casefold() for choice in choices
            }:
                return _invalid(
                    action_id,
                    backend_id,
                    f"Invalid method_spec.calculator_method.{field_name}="
                    f"{calculator_method[field_name]!r} for {calculator_id}/{nested_action}; "
                    f"choose exactly one of {list(choices)}",
                )
        for field_name, choices in calculator.allowed_setting_values.get(
            nested_action, {}
        ).items():
            if field_name in supplied_settings and str(
                supplied_settings[field_name]
            ).strip().casefold() not in {
                str(choice).casefold() for choice in choices
            }:
                return _invalid(
                    action_id,
                    backend_id,
                    "Invalid action_settings.calculator_action_settings"
                    f".{nested_action}.{field_name}={supplied_settings[field_name]!r} "
                    f"for {calculator_id}/{nested_action}; choose exactly one of {list(choices)}",
                )
    return None


def _validate_backend_resource_constraints(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    selected_backends: tuple[Any, ...],
    timeout_seconds: int,
) -> dict[str, Any] | None:
    resources = normalize_resource_limits(request.resource_limits)
    cpu_cores = resources["cpu_cores"]
    gpu_count = resources["gpu_count"]
    for selected in selected_backends:
        minimum_gpu_count = selected.resource_constraints.get("minimum_gpu_count")
        if minimum_gpu_count is not None and gpu_count < int(minimum_gpu_count):
            reason = selected.resource_constraints.get(
                "gpu_reason"
            ) or selected.resource_constraints.get("reason")
            detail = f" Reason: {reason}" if reason else ""
            return _invalid(
                action_id,
                backend_id,
                f"resource_limits.gpu_count={gpu_count} is below the validated minimum "
                f"{minimum_gpu_count} for selected backend {selected.id!r}.{detail} "
                "Select resources or a backend explicitly; no resource substitution or "
                "fallback is performed.",
                code="invalid_resource_limits",
            )
        maximum = selected.resource_constraints.get("maximum_cpu_cores")
        if maximum is not None and cpu_cores > int(maximum):
            reason = selected.resource_constraints.get("reason")
            detail = f" Reason: {reason}" if reason else ""
            return _invalid(
                action_id,
                backend_id,
                f"resource_limits.cpu_cores={cpu_cores} exceeds the validated maximum "
                f"{maximum} for selected backend {selected.id!r}.{detail} Select resources "
                "or a backend explicitly; no resource substitution or fallback is performed.",
                code="invalid_resource_limits",
            )
        maximum_walltime = selected.resource_constraints.get(
            "maximum_walltime_seconds"
        )
        if maximum_walltime is not None and timeout_seconds > int(maximum_walltime):
            reason = selected.resource_constraints.get("walltime_reason")
            detail = f" Reason: {reason}" if reason else ""
            return _invalid(
                action_id,
                backend_id,
                f"The evaluator-controlled timeout_seconds={timeout_seconds} exceeds the "
                f"validated Action maximum {maximum_walltime} for selected backend "
                f"{selected.id!r}.{detail} Adjust the evaluation timeout policy before "
                "running this benchmark; the Agent cannot change this value.",
                code="invalid_timeout_policy",
            )
    try:
        validate_resource_limits(resources)
    except ResourceBudgetExceeded as exc:
        error = exc.as_error()
        response = _invalid(
            action_id,
            backend_id,
            error["message"],
            code=error["code"],
            error_details={
                key: value
                for key, value in error.items()
                if key not in {"code", "message", "retryable"}
            },
            retryable=bool(error.get("retryable")),
        )
        return _attach_safe_resource_diagnostic(
            response,
            action_id=action_id,
            backend_id=backend_id,
        )
    return None


def _validate_artifact_input_semantics(
    action_id: str,
    backend_id: str,
    inputs: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reject a typed ArtifactRef placed in a demonstrably incompatible slot."""

    typed_handoffs: dict[str, dict[str, set[str]]] = {
        "sdf_to_gjf": {
            "sdf_path": {"SDFFile", "SDFRecordFile"},
        },
        "xyz_to_gjf": {
            "input_xyz_path": {"XYZFile", "AtomicStructureFile"},
        },
        "extract_final_structure": {
            "reference_structure": {"AtomicStructure", "AtomicStructureFile"},
        },
        "analyze_reaction_changes": {
            "mapped_reaction": {"AtomMappedReaction"},
        },
        "apply_reaction_changes_to_structure": {
            "structure": {"AtomicStructure"},
            "reaction_changes": {"ReactionChangeSet"},
        },
        "select_reactive_complex_candidate": {
            "ensemble": {"ReactiveComplexEnsemble"},
        },
        "analyze_free_energy_selectivity": {
            "free_energies": {"LabeledFreeEnergyCollection"},
        },
        "analyze_energetic_span": {
            "cycle_states": {"CatalyticCycleFreeEnergyProfile"},
        },
        "validate_reaction_path": {
            "bond_changes": {"ReactionChangeSet"},
        },
        "analyze_reaction_coordinate": {
            "bond_changes": {"ReactionChangeSet"},
        },
        "parse_gaussian_irc_output": {
            "gaussian_irc_log": {"GaussianIRCLog"},
            "transition_state": {"AtomicStructure", "AtomicStructureFile", "XYZFile"},
        },
        "render_spectrum_plot": {
            "spectrum": {"SpectrumResult"},
        },
        "derive_ir_spectrum": {
            "vibrations": {"FrequencyResult", "ParsedQuantumChemistryResult"},
        },
        "derive_uv_vis_spectrum": {
            "excited_states": {"ExcitedStateResult", "ParsedQuantumChemistryResult"},
        },
        "derive_ecd_spectrum": {
            "excited_states": {"ExcitedStateResult", "ParsedQuantumChemistryResult"},
        },
        "derive_raman_spectrum": {
            "spectral_data": {"ParsedQuantumChemistryResult", "SpectrumLineData"},
        },
        "derive_roa_spectrum": {
            "spectral_data": {"ParsedQuantumChemistryResult", "SpectrumLineData"},
        },
        "derive_vcd_spectrum": {
            "spectral_data": {"ParsedQuantumChemistryResult", "SpectrumLineData"},
        },
        "derive_nmr_spectrum": {
            "spectral_data": {"ParsedQuantumChemistryResult", "SpectrumLineData"},
        },
    }
    for field_name, allowed_types in typed_handoffs.get(action_id, {}).items():
        value = inputs.get(field_name)
        if (
            isinstance(value, Mapping)
            and value.get("artifact_id")
            and value.get("semantic_type") not in allowed_types
        ):
            return _invalid(
                action_id,
                backend_id,
                f"inputs.{field_name} received ArtifactRef semantic_type="
                f"{value.get('semantic_type')!r}; accepted types are "
                f"{sorted(allowed_types)}",
                code="artifact_semantic_mismatch",
            )

    if action_id == "build_reactive_complex_ensemble":
        for index, value in enumerate(inputs.get("components") or []):
            if (
                isinstance(value, Mapping)
                and value.get("artifact_id")
                and value.get("semantic_type") != "AtomicStructure"
            ):
                return _invalid(
                    action_id,
                    backend_id,
                    f"inputs.components[{index}] received ArtifactRef semantic_type="
                    f"{value.get('semantic_type')!r}; each component must be AtomicStructure",
                    code="artifact_semantic_mismatch",
                )

    if action_id == "export_electron_density_grid":
        value = inputs.get("electron_density")
        if (
            isinstance(value, Mapping)
            and value.get("artifact_id")
            and value.get("semantic_type") != "ElectronDensityResult"
        ):
            return _invalid(
                action_id,
                backend_id,
                "inputs.electron_density requires the primary ElectronDensityResult artifact "
                "returned by calculate_correlated_electron_density. Pass the output_artifacts "
                "item whose semantic_type is 'ElectronDensityResult', "
                "{'artifact_id': 'art_...'}, or that exact artifact-id string; do not pass "
                "result.files.gbw or another backend file path.",
                code="artifact_semantic_mismatch",
            )
        return None
    if action_id in {
        "calculate_electron_isodensity_surface",
        "analyze_non_covalent_interactions",
    }:
        field_name = (
            "wavefunction_file"
            if action_id == "analyze_non_covalent_interactions"
            else "density_file"
        )
        value = inputs.get(field_name)
        if isinstance(value, Mapping) and value.get("artifact_id"):
            semantic_type = str(value.get("semantic_type") or "")
            suffix = PurePosixPath(str(value.get("path") or "")).suffix.casefold()
            allowed_semantic_types = {
                "ElectronDensityWavefunction",
                "BackendFile",
            }
            allowed_suffixes = {
                ".fch", ".fchk", ".wfn", ".wfx", ".mwfn", ".molden", ".47",
            }
            if action_id == "calculate_electron_isodensity_surface":
                allowed_semantic_types.add("ElectronDensityGrid")
                allowed_suffixes.add(".cube")
            if semantic_type not in allowed_semantic_types and suffix not in allowed_suffixes:
                return _invalid(
                    action_id,
                    backend_id,
                    f"inputs.{field_name} requires a compatible "
                    + (
                        "wavefunction artifact containing basis/orbital information"
                        if action_id == "analyze_non_covalent_interactions"
                        else "wavefunction or electron-density grid artifact returned by export_electron_density_grid"
                    ),
                    code="artifact_semantic_mismatch",
                )
        return None

    if action_id == "derive_thermochemistry" and backend_id == "internal_thermochemistry":
        expected = {
            "energy": "EnergyResult",
            "frequencies": "FrequencyResult",
        }
        for field_name, semantic_type in expected.items():
            value = inputs.get(field_name)
            if (
                isinstance(value, Mapping)
                and value.get("artifact_id")
                and value.get("semantic_type") != semantic_type
            ):
                next_step = (
                    " Call derive_vibrational_modes explicitly before thermochemistry; the "
                    "framework will not insert that Action automatically."
                    if field_name == "frequencies"
                    and value.get("semantic_type") == "Hessian"
                    else ""
                )
                return _invalid(
                    action_id,
                    backend_id,
                    f"inputs.{field_name} received ArtifactRef semantic_type="
                    f"{value.get('semantic_type')!r}; it requires {semantic_type}.{next_step}",
                    code="artifact_semantic_mismatch",
                )
        return None
    if action_id != "derive_vibrational_modes":
        return None
    hessian = inputs.get("hessian")
    structure = inputs.get("structure")
    if (
        isinstance(hessian, Mapping)
        and isinstance(structure, Mapping)
        and hessian.get("artifact_id")
        and hessian.get("artifact_id") == structure.get("artifact_id")
    ):
        return _invalid(
            action_id,
            backend_id,
            "inputs.hessian and inputs.structure require two differently typed values: use the "
            "Hessian primary artifact from calculate_hessian for inputs.hessian and reuse the "
            "matching AtomicStructure input for inputs.structure",
            code="artifact_semantic_mismatch",
        )
    if isinstance(hessian, Mapping) and hessian.get("artifact_id"):
        if hessian.get("semantic_type") != "Hessian":
            return _invalid(
                action_id,
                backend_id,
                f"inputs.hessian received ArtifactRef semantic_type="
                f"{hessian.get('semantic_type')!r}; it requires the primary 'Hessian' JSON "
                "artifact returned by calculate_hessian",
                code="artifact_semantic_mismatch",
            )
    if isinstance(structure, Mapping) and structure.get("artifact_id"):
        semantic_type = str(structure.get("semantic_type") or "")
        media_type = str(structure.get("media_type") or "")
        suffix = PurePosixPath(str(structure.get("path") or "")).suffix.casefold()
        if not (
            semantic_type == "AtomicStructure"
            or media_type.startswith("chemical/")
            or suffix in {".xyz", ".pdb", ".sdf", ".mol"}
        ):
            return _invalid(
                action_id,
                backend_id,
                f"inputs.structure received ArtifactRef semantic_type={semantic_type!r}; use the "
                "matching AtomicStructure artifact or a registered structure file",
                code="artifact_semantic_mismatch",
            )
    return None


def _validate_inline_atomic_structures(
    action_id: str,
    backend_id: str,
    inputs: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reject malformed inline AtomicStructure values before starting a worker."""

    def validate(value: Any, path: str) -> str | None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                message = validate(item, f"{path}[{index}]")
                if message is not None:
                    return message
            return None
        if not isinstance(value, Mapping) or value.get("artifact_id"):
            return None
        if "structure" in value and isinstance(value["structure"], Mapping):
            message = validate(value["structure"], f"{path}.structure")
            if message is not None:
                return message
        if "atoms" not in value:
            return None
        atoms = value["atoms"]
        if not isinstance(atoms, list) or not atoms:
            return f"{path}.atoms must be a non-empty array of atom mappings"
        for index, atom in enumerate(atoms):
            atom_path = f"{path}.atoms[{index}]"
            if not isinstance(atom, Mapping):
                return f"{atom_path} must be a mapping"
            if not (atom.get("element") or atom.get("symbol")):
                return f"{atom_path} requires element (canonical) or symbol"
            position = atom.get("position_angstrom")
            if position is None:
                position = atom.get("position")
            if position is None:
                received = "; received unsupported key 'xyz'" if "xyz" in atom else ""
                return (
                    f"{atom_path} requires position_angstrom=[x, y, z]{received}. "
                    "Canonical example: {'atoms': [{'element': 'H', "
                    "'position_angstrom': [0.0, 0.0, 0.0]}], 'charge': 0, "
                    "'multiplicity': 1}"
                )
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                return f"{atom_path}.position_angstrom must contain exactly three values"
            try:
                numeric = [float(item) for item in position]
            except (TypeError, ValueError):
                return f"{atom_path}.position_angstrom must contain three numeric values"
            if not all(math.isfinite(item) for item in numeric):
                return f"{atom_path}.position_angstrom must contain three finite values"
        return None

    for field_name, value in inputs.items():
        message = validate(value, f"inputs.{field_name}")
        if message is not None:
            return _invalid(
                action_id,
                backend_id,
                message,
                code="invalid_atomic_structure",
            )
    return None


def _register_action_output_artifacts(
    *,
    store: ArtifactStore,
    action_id: str,
    backend_id: str,
    primary_output: str,
    result_payload: Any,
    artifact_files: Any,
    parent_artifact_ids: list[str],
    worker: dict[str, Any],
) -> list[Any]:
    """登记 Action 产物并让可交接的文件拥有唯一语义类型。

    输入是后端返回的结构化结果、文件清单和上游 Artifact 身份；函数登记文件与
    JSON 结果，不改写科学值或文件内容。若后端已经返回与 ``primary_output``
    相同语义类型的文件，JSON 记录只作为 ``ActionResult`` 记录保存，并把真实
    文件排在输出列表前面，避免下游把 JSON 包装误当成可读取的科学文件。文件型
    primary 的条目缺失、畸形或登记失败会抛出结构化异常，由调用方把 Action 转为
    失败；JSON-only Action 仍保留登记 warning 后返回 JSON primary 的旧行为。
    """

    file_primary = primary_output.endswith(("File", "Deck"))
    if file_primary:
        if artifact_files is None:
            entries: list[Any] = []
        elif isinstance(artifact_files, (list, tuple)):
            entries = list(artifact_files)
        else:
            raise _PrimaryArtifactRegistrationFailure(
                "primary_artifact_malformed",
                "后端返回的 primary 文件清单必须是数组",
                details={"received_type": type(artifact_files).__name__},
            )
        if not entries:
            raise _PrimaryArtifactRegistrationFailure(
                "primary_artifact_missing",
                f"后端未返回声明的 primary 文件 {primary_output}",
                details={"primary_output": primary_output},
            )
        for index, item in enumerate(entries):
            if not isinstance(item, Mapping):
                raise _PrimaryArtifactRegistrationFailure(
                    "primary_artifact_malformed",
                    f"primary 文件条目 {index} 必须是对象",
                    details={"artifact_index": index},
                )
            path = item.get("path")
            semantic_type = item.get("semantic_type")
            media_type = item.get("media_type")
            if not isinstance(path, str) or not path.strip():
                raise _PrimaryArtifactRegistrationFailure(
                    "primary_artifact_malformed",
                    f"primary 文件条目 {index} 缺少非空 path",
                    details={"artifact_index": index},
                )
            if not isinstance(semantic_type, str) or not semantic_type.strip():
                raise _PrimaryArtifactRegistrationFailure(
                    "primary_artifact_malformed",
                    f"primary 文件条目 {index} 缺少 semantic_type",
                    details={"artifact_index": index},
                )
            if not isinstance(media_type, str) or not media_type.strip():
                raise _PrimaryArtifactRegistrationFailure(
                    "primary_artifact_malformed",
                    f"primary 文件条目 {index} 缺少 media_type",
                    details={"artifact_index": index},
                )
            if semantic_type == primary_output and media_type == "application/json":
                raise _PrimaryArtifactRegistrationFailure(
                    "primary_artifact_malformed",
                    f"primary 文件条目 {index} 不能使用 application/json",
                    details={"artifact_index": index, "primary_output": primary_output},
                )

    else:
        entries = list(artifact_files or [])

    result_pointer_targets = _collect_result_pointer_targets(result_payload, entries)
    registered_files: list[Any] = []
    registered_pointer_files: dict[int, Any] = {}
    for index, item in enumerate(entries):
        try:
            reference = store.register_file(
                item["path"],
                semantic_type=item.get("semantic_type", "BackendFile"),
                media_type=item.get("media_type", "application/octet-stream"),
                producer_action=action_id,
                producer_backend=backend_id,
                parent_artifact_ids=parent_artifact_ids,
            )
            if file_primary and not isinstance(getattr(reference, "semantic_type", None), str):
                raise ValueError("register_file returned an invalid ArtifactRef")
            registered_files.append(reference)
            if index in result_pointer_targets:
                registered_pointer_files[index] = reference
        except Exception as exc:
            if file_primary:
                raise _PrimaryArtifactRegistrationFailure(
                    "primary_artifact_registration_failed",
                    _PRIMARY_FILE_REGISTRATION_FAILURE_MESSAGE,
                    details={
                        "artifact_index": index,
                        "exception_type": type(exc).__name__,
                    },
                ) from exc
            if index in result_pointer_targets:
                raise _ResultPointerRegistrationFailure(
                    "result_pointer_registration_failed",
                    "result_pointer 对应文件登记失败",
                    details={
                        "artifact_index": index,
                        "exception_type": type(exc).__name__,
                    },
                ) from exc
            worker.setdefault("warnings", []).append(f"Artifact registration failed: {exc}")

    for index, target in result_pointer_targets.items():
        reference = registered_pointer_files.get(index)
        if reference is None:
            raise _ResultPointerRegistrationFailure(
                "result_pointer_registration_failed",
                "result_pointer 对应文件未成功登记",
                details={"artifact_index": index},
            )
        parent, key = target
        parent[key] = {"artifact_id": reference.artifact_id}

    if file_primary:
        primary_files = [
            reference
            for reference in registered_files
            if reference.semantic_type == primary_output
        ]
        if not primary_files:
            raise _PrimaryArtifactRegistrationFailure(
                "primary_artifact_missing",
                f"后端返回的文件清单中没有 {primary_output} primary 文件",
                details={"primary_output": primary_output},
            )
        if len(primary_files) != 1:
            raise _PrimaryArtifactRegistrationFailure(
                "primary_artifact_ambiguous",
                f"后端返回了 {len(primary_files)} 个 {primary_output} primary 文件",
                details={"primary_output": primary_output, "count": len(primary_files)},
            )

    has_typed_primary_file = any(
        reference.semantic_type == primary_output for reference in registered_files
    )
    payload_semantic_type = (
        _RESULT_PAYLOAD_SEMANTIC_TYPE
        if has_typed_primary_file and primary_output != _RESULT_PAYLOAD_SEMANTIC_TYPE
        else primary_output
    )
    try:
        result_reference = store.put_json(
            result_payload,
            semantic_type=payload_semantic_type,
            producer_action=action_id,
            producer_backend=backend_id,
            parent_artifact_ids=parent_artifact_ids,
        )
    except Exception as exc:
        if file_primary:
            raise _PrimaryArtifactRegistrationFailure(
                "primary_artifact_registration_failed",
                _PRIMARY_RESULT_REGISTRATION_FAILURE_MESSAGE,
                details={"exception_type": type(exc).__name__},
            ) from exc
        raise
    if not has_typed_primary_file:
        return [result_reference, *registered_files]

    primary_files = [
        reference
        for reference in registered_files
        if reference.semantic_type == primary_output
    ]
    supplementary_files = [
        reference
        for reference in registered_files
        if reference.semantic_type != primary_output
    ]
    return [*primary_files, result_reference, *supplementary_files]


def _parse_action_execution_request(
    action_id: str,
    request_value: ActionRequest | dict[str, Any],
    actions: Mapping[str, Any],
) -> tuple[Any, ActionRequest] | dict[str, Any]:
    """检查 Action 身份并解析请求；失败时返回统一的 invalid envelope。"""

    if action_id not in actions:
        return _invalid(action_id, None, f"Unknown action: {action_id}", code="unknown_action")
    try:
        request = (
            request_value
            if isinstance(request_value, ActionRequest)
            else ActionRequest.model_validate(request_value)
        )
    except ValidationError as exc:
        return _invalid(action_id, None, str(exc))
    return actions[action_id], request


def _validate_fixed_backend_choice(
    action_id: str,
    request: ActionRequest,
    fixed_backend: str,
) -> dict[str, Any] | None:
    """检查固定数据源 Action 没有收到冲突 backend_id。"""

    if request.backend_id is None or request.backend_id == fixed_backend:
        return None
    return _invalid(
        action_id,
        request.backend_id,
        f"Data Action {action_id} uses fixed data source {fixed_backend}",
    )


def _validate_fixed_source_choice(
    action_id: str,
    request: ActionRequest,
    fixed_backend: str,
) -> dict[str, Any] | None:
    """检查固定数据源 Action 没有收到冲突 source_id。"""

    if request.source_id is None or request.source_id == fixed_backend:
        return None
    return _invalid(
        action_id,
        request.source_id,
        f"Data Action {action_id} uses fixed data source {fixed_backend}",
    )


def _select_fixed_source_backend(
    action_id: str,
    request: ActionRequest,
    specification: Any,
) -> tuple[str, str] | dict[str, Any]:
    """解析固定数据源 Action 的显式选择。"""

    fixed_backend = specification.backend_ids[0]
    invalid_backend = _validate_fixed_backend_choice(action_id, request, fixed_backend)
    if invalid_backend is not None:
        return invalid_backend
    invalid_source = _validate_fixed_source_choice(action_id, request, fixed_backend)
    if invalid_source is not None:
        return invalid_source
    if request.component_backends:
        return _invalid(action_id, fixed_backend, f"Data Action {action_id} does not accept component_backends")
    return fixed_backend, "fixed_data_source"


def _select_internal_deterministic_backend(
    action_id: str,
    request: ActionRequest,
    specification: Any,
) -> tuple[str, str] | dict[str, Any]:
    """解析固定确定性计算器的显式选择。"""

    fixed_backend = specification.backend_ids[0]
    if request.backend_id is not None and request.backend_id != fixed_backend:
        return _invalid(action_id, request.backend_id, f"Action {action_id} uses fixed deterministic provider {fixed_backend}")
    if request.source_id is not None or request.component_backends:
        return _invalid(action_id, request.backend_id, f"Action {action_id} does not accept source_id or component_backends")
    return fixed_backend, "internal_deterministic"


def _select_agent_source_backend(
    action_id: str,
    request: ActionRequest,
    _specification: Any,
) -> tuple[str, str] | dict[str, Any]:
    """解析要求 Agent 明确提供 source_id 的数据 Action。"""

    if request.source_id is None:
        return _invalid(action_id, None, "source_id is required for this Data Action")
    if request.backend_id is not None or request.component_backends:
        return _invalid(action_id, request.backend_id, "This Data Action accepts source_id, not backend_id/component_backends")
    return request.source_id, "agent_data_source"


def _select_agent_backend(
    action_id: str,
    request: ActionRequest,
    specification: Any,
) -> tuple[str, str] | dict[str, Any]:
    """解析要求 Agent 明确提供 backend_id 的科学 Action。"""

    if request.backend_id is None:
        return _invalid(action_id, None, "backend_id is required for every Scientific Action")
    selection_source = (
        "agent_components"
        if specification.selection_policy == "agent_components_required"
        else "agent"
    )
    return request.backend_id, selection_source


def _select_action_backend(
    action_id: str,
    request: ActionRequest,
    specification: Any,
) -> tuple[str, str] | dict[str, Any]:
    """依据 Action 的固定选择策略解析后端，不替请求补默认值。"""

    selectors = {
        "fixed_source": _select_fixed_source_backend,
        "internal_deterministic": _select_internal_deterministic_backend,
        "agent_source_required": _select_agent_source_backend,
        "agent_backend_required": _select_agent_backend,
        "agent_components_required": _select_agent_backend,
    }
    selector = selectors.get(specification.selection_policy)
    if selector is None:
        return _invalid(
            action_id,
            request.backend_id,
            f"Unknown selection policy: {specification.selection_policy}",
        )
    return selector(action_id, request, specification)

def _missing_action_input_fields(
    request: ActionRequest,
    specification: Any,
) -> list[str]:
    """返回 Action 合同要求但请求缺少的公共输入字段。"""

    return [name for name in specification.required_inputs if name not in request.inputs]


def _missing_backend_input_fields(
    action_id: str,
    request: ActionRequest,
    backend: Any,
) -> list[str]:
    """返回后端合同要求但请求缺少的专用输入字段。"""

    return [
        name
        for name in backend.required_input_fields.get(action_id, ())
        if name not in request.inputs
    ]


def _validate_action_input_fields(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    specification: Any,
    backend: Any,
) -> dict[str, Any] | None:
    """检查 Action 和后端声明的必填输入字段。"""

    missing_inputs = _missing_action_input_fields(request, specification)
    if missing_inputs:
        return _repairable_invalid(
            action_id,
            backend_id,
            request,
            f"Missing required inputs: {missing_inputs}",
            missing_fields=tuple(f"inputs.{name}" for name in missing_inputs),
        )
    missing_backend_inputs = _missing_backend_input_fields(action_id, request, backend)
    if missing_backend_inputs:
        return _repairable_invalid(
            action_id,
            backend_id,
            request,
            f"Missing backend-specific required inputs: {missing_backend_inputs}",
            missing_fields=tuple(f"inputs.{name}" for name in missing_backend_inputs),
        )
    return None


def _validate_required_component_roles(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    required_roles: Any,
) -> dict[str, Any] | None:
    """检查 composite backend 要求的角色是否全部显式提供。"""

    missing_roles = [role for role in required_roles if role not in request.component_backends]
    if not missing_roles:
        return None
    return _repairable_invalid(
        action_id,
        backend_id,
        request,
        f"Missing explicitly required component_backends roles: {missing_roles}",
        missing_fields=tuple(f"component_backends.{role}" for role in missing_roles),
    )


def _validate_component_backend_choice(
    action_id: str,
    backend_id: str,
    role: str,
    component_backend_id: str,
    component_options: Mapping[str, Any],
    backends: Mapping[str, Any],
) -> dict[str, Any] | None:
    """检查单个组件角色对应的后端存在且属于允许集合。"""

    if component_backend_id not in backends:
        return _invalid(action_id, backend_id, f"Unknown component backend {component_backend_id!r} for role {role!r}")
    allowed = component_options.get(role)
    if allowed is None:
        return _invalid(action_id, backend_id, f"Unsupported component backend role {role!r} for {backend_id}/{action_id}")
    if component_backend_id not in allowed:
        return _invalid(
            action_id,
            backend_id,
            f"Component backend {component_backend_id!r} is not allowed for role {role!r}; choose one of {list(allowed)}",
        )
    return None


def _validate_component_backend_roles(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    required_roles: Any,
    component_options: Mapping[str, Any],
    backends: Mapping[str, Any],
) -> dict[str, Any] | None:
    """检查组件角色、后端存在性和每个角色的允许集合。"""

    missing_error = _validate_required_component_roles(
        action_id, backend_id, request, required_roles
    )
    if missing_error is not None:
        return missing_error
    for role, component_backend_id in request.component_backends.items():
        invalid_choice = _validate_component_backend_choice(
            action_id,
            backend_id,
            role,
            component_backend_id,
            component_options,
            backends,
        )
        if invalid_choice is not None:
            return invalid_choice
    return None


def _resolve_action_component_backends(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    backend: Any,
    backends: Mapping[str, Any],
) -> tuple[tuple[Any, ...], bool] | dict[str, Any]:
    """检查组件后端并返回其规格，同时标记是否使用了组件选择。"""

    required_roles = backend.required_component_roles.get(action_id, ())
    component_options = backend.component_backend_options.get(action_id, {})
    uses_components = bool(required_roles or component_options)
    if request.component_backends and not uses_components:
        return _invalid(
            action_id,
            backend_id,
            f"Backend {backend_id!r} for action {action_id!r} does not use component_backends",
        )
    invalid_roles = _validate_component_backend_roles(
        action_id,
        backend_id,
        request,
        required_roles,
        component_options,
        backends,
    )
    if invalid_roles is not None:
        return invalid_roles
    component_specs = tuple(
        backends[component_backend_id]
        for component_backend_id in request.component_backends.values()
    )
    return component_specs, uses_components


def _resolve_action_resource_backend_specs(
    action_id: str,
    request: ActionRequest,
) -> tuple[Any, ...]:
    """解析服务端选定的 Action/backend/component 规格，供资源绑定使用。

    输入是已解析的 Action 请求；函数只读取受控 Catalog，解析固定或调用方明确
    选择的主 backend 及组件 backend，不读取调用方的 resource_limits，也不启动
    worker。选择或组件合同本身无效时仍返回已识别的主 backend（若有），让资源
    指纹保持确定；真正的输入和科学合同错误继续由执行准备阶段返回。
    """

    specification = action_specs().get(action_id)
    if specification is None:
        return ()
    selected = _select_action_backend(action_id, request, specification)
    if isinstance(selected, dict):
        return ()
    backend_id, _selection_source = selected
    if backend_id not in specification.backend_ids:
        return ()
    backends = backend_specs()
    backend = backends.get(backend_id)
    if backend is None:
        return ()
    components = _resolve_action_component_backends(
        action_id,
        backend_id,
        request,
        backend,
        backends,
    )
    if isinstance(components, dict):
        return (backend,)
    component_specs, _uses_components = components
    return (backend, *component_specs)

def _missing_method_spec_fields(request: ActionRequest, backend: Any, action_id: str) -> list[str]:
    """返回后端声明但请求未提供的 method_spec 字段。"""

    return [
        name for name in backend.required_method_fields.get(action_id, ())
        if name not in request.method_spec
    ]


def _missing_action_setting_fields(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    backend: Any,
) -> list[str]:
    """返回后端声明但请求未提供的 action_settings 字段。"""

    return [
        name for name in backend.required_setting_fields.get(action_id, ())
        if not _required_setting_is_present(action_id, backend_id, name, request.action_settings)
    ]


def _validate_required_scientific_fields(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    backend: Any,
) -> dict[str, Any] | None:
    """检查后端声明的 method_spec 和 action_settings 字段。"""

    missing_methods = _missing_method_spec_fields(request, backend, action_id)
    missing_settings = _missing_action_setting_fields(
        action_id, backend_id, request, backend
    )
    if not missing_methods and not missing_settings:
        return None
    parts = []
    if missing_methods:
        parts.append(f"method_spec fields {missing_methods}")
    if missing_settings:
        parts.append(f"action_settings fields {missing_settings}")
    return _repairable_invalid(
        action_id,
        backend_id,
        request,
        "Missing explicitly required scientific settings: " + "; ".join(parts),
        missing_fields=tuple(
            [f"method_spec.{name}" for name in missing_methods]
            + [f"action_settings.{name}" for name in missing_settings]
        ),
    )


def _validate_action_choice_values(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    backend: Any,
) -> dict[str, Any] | None:
    """检查 method_spec 和 action_settings 中的显式枚举值。"""

    return _invalid_explicit_choice(
        action_id,
        backend_id,
        request,
        field_group="method_spec",
        supplied=request.method_spec,
        allowed=backend.allowed_method_values.get(action_id, {}),
    ) or _invalid_explicit_choice(
        action_id,
        backend_id,
        request,
        field_group="action_settings",
        supplied=request.action_settings,
        allowed=backend.allowed_setting_values.get(action_id, {}),
    )


def _validate_action_scientific_settings(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
    backend: Any,
    component_specs: tuple[Any, ...],
    backends: Mapping[str, Any],
    execution_timeout: int,
) -> dict[str, Any] | None:
    """检查显式科学参数、嵌套计算器和后端资源合同。"""

    invalid_aliases = _validate_deprecated_setting_aliases(
        action_id, backend_id, request.action_settings
    )
    if invalid_aliases is not None:
        return invalid_aliases
    invalid_orca_basis = _validate_orca_method_basis_contract(action_id, backend_id, request)
    if invalid_orca_basis is not None:
        return invalid_orca_basis
    missing_fields = _validate_required_scientific_fields(
        action_id, backend_id, request, backend
    )
    if missing_fields is not None:
        return missing_fields
    invalid_choice = _validate_action_choice_values(
        action_id, backend_id, request, backend
    )
    if invalid_choice is not None:
        return invalid_choice
    invalid_composite = _validate_composite_calculator_contract(
        action_id, backend_id, request, backends
    )
    if invalid_composite is not None:
        return invalid_composite
    return _validate_backend_resource_constraints(
        action_id,
        backend_id,
        request,
        (backend, *component_specs),
        execution_timeout,
    )

def _canonicalize_action_execution_inputs(
    action_id: str,
    backend_id: str,
    request: ActionRequest,
) -> tuple[ActionRequest, Any] | dict[str, Any]:
    """登记并检查输入 ArtifactRef，返回规范化请求和资源引用。"""

    try:
        canonical_inputs = canonicalize_artifact_refs(request.inputs)
    except (KeyError, OSError, ValueError, ValidationError) as exc:
        return _invalid(action_id, backend_id, f"Invalid explicit ArtifactRef: {exc}", code="invalid_artifact_reference")
    request = request.model_copy(update={"inputs": canonical_inputs})
    invalid_semantics = _validate_artifact_input_semantics(action_id, backend_id, request.inputs)
    if invalid_semantics is not None:
        return invalid_semantics
    invalid_inline = _validate_inline_atomic_structures(action_id, backend_id, request.inputs)
    if invalid_inline is not None:
        return invalid_inline
    resource_references = collect_resource_references(
        {"inputs": request.inputs, "method_spec": request.method_spec}
    )
    return request, resource_references


def _prepare_action_execution(
    action_id: str,
    request_value: ActionRequest | dict[str, Any],
    *,
    resources_bound: bool,
) -> _ActionExecutionPlan | dict[str, Any]:
    """按固定顺序完成 Action 解析、选择、合同和输入校验。"""

    actions = action_specs()
    parsed = _parse_action_execution_request(action_id, request_value, actions)
    if isinstance(parsed, dict):
        return parsed
    specification, request = parsed
    if "timeout_seconds" in request.action_settings:
        return _invalid(
            action_id,
            request.backend_id or request.source_id,
            "action_settings.timeout_seconds is controlled by the evaluation policy and cannot be supplied by the Agent",
            code="evaluator_controlled_timeout",
        )
    execution_timeout = timeout_seconds_for(specification.execution_class)
    timeout_policy = timeout_policy_record(specification.execution_class)
    selected = _select_action_backend(action_id, request, specification)
    if isinstance(selected, dict):
        return selected
    backend_id, selection_source = selected
    if backend_id not in specification.backend_ids:
        return _invalid(
            action_id,
            backend_id,
            f"Backend {backend_id!r} does not declare support for action {action_id!r}",
            code="unsupported_backend",
        )
    backends = backend_specs()
    backend = backends[backend_id]
    invalid_inputs = _validate_action_input_fields(action_id, backend_id, request, specification, backend)
    if invalid_inputs is not None:
        return invalid_inputs
    components = _resolve_action_component_backends(action_id, backend_id, request, backend, backends)
    if isinstance(components, dict):
        return components
    component_specs, uses_components = components
    if uses_components:
        selection_source = "agent_components"
    if not resources_bound:
        request = _bind_fixed_action_resources(
            request,
            selected_backends=(backend, *component_specs),
        )
    invalid_settings = _validate_action_scientific_settings(
        action_id,
        backend_id,
        request,
        backend,
        component_specs,
        backends,
        execution_timeout,
    )
    if invalid_settings is not None:
        return invalid_settings
    canonicalized = _canonicalize_action_execution_inputs(action_id, backend_id, request)
    if isinstance(canonicalized, dict):
        return canonicalized
    request, resource_references = canonicalized
    return _ActionExecutionPlan(
        action_id=action_id,
        request=request,
        specification=specification,
        backend_id=backend_id,
        backend=backend,
        component_specs=component_specs,
        selection_source=selection_source,
        execution_timeout=execution_timeout,
        timeout_policy=timeout_policy,
        resource_references=resource_references,
        input_artifacts=collect_artifact_refs(request.inputs),
    )


def _make_unavailable_action_response(
    plan: _ActionExecutionPlan,
    error: Mapping[str, Any],
    *,
    include_request_details: bool = False,
) -> dict[str, Any]:
    """把后端或组件不可用事实封装为标准 ActionResult。"""

    provenance = {
        "catalog_hash": active_catalog_hash(),
        "agent_selected_action": plan.action_id,
        "agent_selected_backend": plan.backend_id,
        "agent_selected_component_backends": plan.request.component_backends,
        "agent_selected_source_id": plan.request.source_id,
        "automatic_fallback_count": 0,
        "execution_timeout_policy": plan.timeout_policy,
    }
    if include_request_details:
        provenance.update(
            {
                "agent_selected_method_spec": plan.request.method_spec,
                "agent_selected_action_settings": plan.request.action_settings,
                "agent_selected_resource_refs": plan.resource_references,
                "runtime_profile": plan.backend.runtime,
            }
        )
    return ActionResult(
        status="unavailable",
        action=plan.action_id,
        action_version=plan.specification.version,
        requested_backend=plan.backend_id,
        backend=plan.backend_id,
        selection_source=plan.selection_source,
        error=dict(error),
        retryable=False,
        provenance=provenance,
    ).model_dump(mode="json")


def _validate_action_backend_health(plan: _ActionExecutionPlan) -> dict[str, Any] | None:
    """探测主后端和组件后端，并在不可用时停止执行。"""

    health_values = probe_all_backends((plan.backend, *plan.component_specs))
    unavailable_components = {
        role: health_values[component_backend_id]
        for role, component_backend_id in plan.request.component_backends.items()
        if not health_values[component_backend_id]["available"]
    }
    if unavailable_components:
        return _make_unavailable_action_response(
            plan,
            {
                "code": "component_backend_unavailable",
                "message": "One or more Agent-selected component backends are unavailable",
                "components": unavailable_components,
            },
        )
    health = health_values[plan.backend_id]
    if health["available"]:
        return None
    return _make_unavailable_action_response(
        plan,
        {
            "code": "backend_unavailable",
            "message": f"Requested backend {plan.backend_id} is unavailable in runtime {plan.backend.runtime}",
            "health": health,
        },
        include_request_details=True,
    )


def _resource_reservation_failure(
    plan: _ActionExecutionPlan,
    exc: ResourceStateUnavailable | ResourceBudgetExceeded,
) -> dict[str, Any]:
    """将资源状态或容量拒绝转换成不泄露容量详情的响应。"""

    error = exc.as_error()
    details = (
        {
            key: value for key, value in error.items()
            if key not in {"code", "message", "retryable"}
        }
        if isinstance(exc, ResourceBudgetExceeded)
        else {}
    )
    response = _invalid(
        plan.action_id,
        plan.backend_id,
        error["message"],
        code=error["code"],
        error_details=details,
        retryable=bool(error.get("retryable")) if isinstance(exc, ResourceBudgetExceeded) else False,
    )
    return _attach_safe_resource_diagnostic(
        response,
        action_id=plan.action_id,
        backend_id=plan.backend_id,
    )


def _reserve_action_execution_resources(
    plan: _ActionExecutionPlan,
    execution_request: Mapping[str, Any],
    gaussian_queue_ticket: Any,
) -> Any | dict[str, Any]:
    """按后端队列状态预留资源，失败时返回安全的标准错误响应。"""

    try:
        return reserve_resources(
            execution_request["resource_limits"],
            kind="predefined_action",
            label=f"{plan.action_id}/{plan.backend_id}",
            worker_tracking=("gaussian_typed" if gaussian_queue_ticket is not None else None),
        )
    except (ResourceStateUnavailable, ResourceBudgetExceeded) as exc:
        return _resource_reservation_failure(plan, exc)


def _invoke_reserved_action_worker(
    plan: _ActionExecutionPlan,
    execution_request: Mapping[str, Any],
    reservation: Any,
    gaussian_queue_ticket: Any,
) -> tuple[Any, bool]:
    """调用已完成资源预留的 worker，并指出是否必须保留资源收据。"""

    worker_started_callback = None
    if gaussian_queue_ticket is not None:
        worker_started_callback = partial(
            _record_typed_gaussian_worker,
            queue_ticket=gaussian_queue_ticket,
            reservation=reservation,
        )
    worker = invoke_worker(
        runtime=plan.backend.runtime,
        payload={
            "action_id": plan.action_id,
            "backend_id": plan.backend_id,
            "request": execution_request,
        },
        timeout_seconds=plan.execution_timeout,
        resource_allocation=reservation.resource_allocation,
        on_worker_started=worker_started_callback,
    )
    if worker.get("resource_state") != "unknown":
        return worker, True
    if gaussian_queue_ticket is not None:
        defer_release = getattr(gaussian_queue_ticket, "defer_release", None)
        if callable(defer_release):
            defer_release()
    return worker, False


def _derive_gaussian_execution_identity(
    plan: _ActionExecutionPlan,
) -> tuple[str | None, str | None] | dict[str, Any]:
    """为 scope 内 Gaussian 请求派生稳定 owner/job 身份。"""

    if current_scope() is None:
        return None, None
    if plan.request.submission_key is None:
        return _invalid(
            plan.action_id,
            plan.backend_id,
            "当前 scope 的 Gaussian Action 缺少 submission_key",
            code="missing_submission_key",
        )
    owner_id = scoped_submission_operation_id(plan.request.submission_key)
    job_id = "job_" + hashlib.sha256(owner_id.encode("ascii")).hexdigest()[:32]
    return owner_id, job_id


def _invoke_action_worker(plan: _ActionExecutionPlan) -> _WorkerExecution | dict[str, Any]:
    """构造执行请求、进入 Gaussian FIFO 并调用后端 worker。"""

    execution_request = plan.request.model_dump(mode="json")
    execution_request["resource_limits"].update(normalize_resource_limits(plan.request.resource_limits))
    execution_request["resource_limits"]["walltime_seconds"] = plan.execution_timeout
    identity = (None, None)
    if plan.backend_id == "gaussian":
        identity = _derive_gaussian_execution_identity(plan)
        if isinstance(identity, dict):
            return identity
    owner_id, job_id = identity
    if plan.backend_id == "gaussian":
        gate = (
            gaussian_execution_slot(owner_id=owner_id, job_id=job_id)
            if owner_id is not None
            else gaussian_execution_slot()
        )
    else:
        gate = nullcontext()
    with gate as gaussian_queue_ticket:
        reserved = _reserve_action_execution_resources(plan, execution_request, gaussian_queue_ticket)
        if isinstance(reserved, dict):
            return reserved
        release_reservation = True
        try:
            worker, release_reservation = _invoke_reserved_action_worker(
                plan,
                execution_request,
                reserved,
                gaussian_queue_ticket,
            )
        finally:
            if release_reservation:
                reserved.release()
        return _WorkerExecution(worker, reserved.resource_allocation)


def _normalize_action_worker_result(worker_value: Any) -> tuple[dict[str, Any], str, Any]:
    """清理 worker 响应、限制状态枚举并提取结果负载。"""

    worker = _sanitize_worker_response(worker_value)
    status = worker.get("status", "failed")
    valid_statuses = {
        "success", "partial_success", "invalid_request", "unsupported", "unavailable",
        "failed", "timeout", "cancelled",
    }
    if status not in valid_statuses:
        status = "failed"
        worker["error"] = {
            "code": "invalid_backend_status",
            "message": f"Backend returned invalid status {worker.get('status')!r}",
        }
    result_payload = worker.get("result")
    if status not in {"success", "partial_success"}:
        result_payload = _sanitize_public_diagnostic_value(result_payload)
    return worker, status, result_payload


def _register_successful_action_outputs(
    plan: _ActionExecutionPlan,
    store: ArtifactStore,
    worker: dict[str, Any],
    status: str,
    result_payload: Any,
) -> tuple[str, Any, list[Any], dict[str, Any]]:
    """登记成功结果及文件；交接失败时生成确定性失败状态。"""

    try:
        artifact_files = _materialize_structure_batch_handoff(
            action_id=plan.action_id,
            backend_id=plan.backend_id,
            result_payload=result_payload,
            artifact_files=worker.get("artifact_files", []),
        )
        output_artifacts = _register_action_output_artifacts(
            store=store,
            action_id=plan.action_id,
            backend_id=plan.backend_id,
            primary_output=plan.specification.primary_output,
            result_payload=result_payload,
            artifact_files=artifact_files,
            parent_artifact_ids=[item.artifact_id for item in plan.input_artifacts],
            worker=worker,
        )
    except (
        _BatchHandoffArtifactMaterializationFailure,
        _PrimaryArtifactRegistrationFailure,
        _ResultPointerRegistrationFailure,
    ) as exc:
        if isinstance(exc, _BatchHandoffArtifactMaterializationFailure):
            error = {"code": "batch_handoff_artifact_materialization_failed", "message": str(exc), **exc.details}
        else:
            error = {"code": exc.code, "message": str(exc), **exc.details}
            if isinstance(exc, _PrimaryArtifactRegistrationFailure):
                error["primary_output"] = plan.specification.primary_output
        worker["status"] = "failed"
        worker["retryable"] = False
        worker["error"] = error
        return "failed", None, [], worker
    return status, result_payload, output_artifacts, worker


def _register_action_failure_diagnostic(
    plan: _ActionExecutionPlan,
    store: ArtifactStore,
    worker: dict[str, Any],
    status: str,
    output_artifacts: list[Any],
) -> list[Any]:
    """为失败 worker 登记脱敏 BackendDiagnostic，登记失败只留下 warning。"""

    if status in {"success", "partial_success"} or not worker.get("error"):
        return output_artifacts
    try:
        output_artifacts.append(
            store.put_json(
                {
                    "action": plan.action_id,
                    "backend": plan.backend_id,
                    "status": status,
                    "error": worker.get("error"),
                    "warnings": list(worker.get("warnings") or []),
                },
                semantic_type="BackendDiagnostic",
                producer_action=plan.action_id,
                producer_backend=plan.backend_id,
                parent_artifact_ids=[item.artifact_id for item in plan.input_artifacts],
            )
        )
    except (OSError, ValueError) as exc:
        worker.setdefault("warnings", []).append(f"Backend diagnostic artifact registration failed: {exc}")
    return output_artifacts


def _collect_action_result_artifacts(
    plan: _ActionExecutionPlan,
    worker: dict[str, Any],
    status: str,
    result_payload: Any,
) -> tuple[str, Any, list[Any], dict[str, Any]]:
    """统一登记成功文件、失败诊断和结果状态。"""

    store = ArtifactStore()
    output_artifacts: list[Any] = []
    if status in {"success", "partial_success"}:
        status, result_payload, output_artifacts, worker = _register_successful_action_outputs(
            plan, store, worker, status, result_payload
        )
    output_artifacts = _register_action_failure_diagnostic(
        plan, store, worker, status, output_artifacts
    )
    return status, result_payload, output_artifacts, worker


def _build_action_result(
    plan: _ActionExecutionPlan,
    worker: dict[str, Any],
    status: str,
    result_payload: Any,
    output_artifacts: list[Any],
    resource_allocation: Any,
) -> dict[str, Any]:
    """把科学结果、产物、资源和 worker provenance 封装为公开 ActionResult。"""

    provenance = {
        "catalog_hash": active_catalog_hash(),
        "agent_selected_action": plan.action_id,
        "agent_selected_backend": plan.backend_id,
        "agent_selected_method_spec": plan.request.method_spec,
        "agent_selected_action_settings": plan.request.action_settings,
        "agent_selected_resource_refs": plan.resource_references,
        "agent_selected_component_backends": plan.request.component_backends,
        "agent_selected_source_id": plan.request.source_id,
        "runtime_profile": plan.backend.runtime,
        "dispatcher_executed_backend": plan.backend_id,
        "automatic_fallback_count": 0,
        "execution_timeout_policy": plan.timeout_policy,
        "evaluation_resource_budget": resource_budget_record(),
        "requested_resource_limits": normalize_resource_limits(plan.request.resource_limits),
        "resource_allocation": resource_allocation,
        **dict(worker.get("provenance") or {}),
        "execution_memory_enforcement": {
            "mechanism": "resource_budget_validation_and_reservation",
            "rlimit_as_applied": False,
            "rss_monitor_applied": False,
            "reason": (
                "Synchronous scientific workers are not constrained with RLIMIT_AS because "
                "Gaussian, Multiwfn, and numerical runtimes may reserve substantially more "
                "virtual address space than resident memory."
            ),
        },
    }
    if worker.get("resource_state") is not None:
        provenance["resource_state"] = worker["resource_state"]
    return ActionResult(
        status=status,
        action=plan.action_id,
        action_version=plan.specification.version,
        requested_backend=plan.backend_id,
        backend=plan.backend_id,
        backend_version=worker.get("backend_version"),
        selection_source=plan.selection_source,
        result=result_payload,
        input_artifacts=plan.input_artifacts,
        output_artifacts=output_artifacts,
        warnings=list(worker.get("warnings") or []),
        provenance=provenance,
        error=worker.get("error"),
        retryable=bool(worker.get("retryable", False)),
    ).model_dump(mode="json")


def _execute_action_once_for_idempotent_request(
    action_id: str,
    request_value: ActionRequest | dict[str, Any],
    *,
    resources_bound: bool,
) -> dict[str, Any]:
    """执行一次 Action；未知 worker 状态时保留上层 running receipt。"""

    response = _execute_action_once(
        action_id,
        request_value,
        resources_bound=resources_bound,
    )
    provenance = response.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("resource_state") == "unknown":
        raise _WorkerResourceStateUnknown(response)
    return response


def _execute_action_once(
    action_id: str,
    request_value: ActionRequest | dict[str, Any],
    *,
    resources_bound: bool = False,
) -> dict[str, Any]:
    """执行一次 Action；输入合同、资源和产物交接均在内部阶段完成。"""

    prepared = _prepare_action_execution(
        action_id,
        request_value,
        resources_bound=resources_bound,
    )
    if isinstance(prepared, dict):
        return prepared
    health_error = _validate_action_backend_health(prepared)
    if health_error is not None:
        return health_error
    worker_execution = _invoke_action_worker(prepared)
    if isinstance(worker_execution, dict):
        return worker_execution
    worker, status, result_payload = _normalize_action_worker_result(
        worker_execution.worker
    )
    status, result_payload, output_artifacts, worker = _collect_action_result_artifacts(
        prepared,
        worker,
        status,
        result_payload,
    )
    return _build_action_result(
        prepared,
        worker,
        status,
        result_payload,
        output_artifacts,
        worker_execution.resource_allocation,
    )


def _request_identity_field(
    request_value: ActionRequest | dict[str, Any],
    field_name: str,
) -> Any:
    """读取请求中的 operation/submission 身份字段，不改变请求对象。"""

    if isinstance(request_value, ActionRequest):
        return getattr(request_value, field_name)
    if isinstance(request_value, Mapping):
        return request_value.get(field_name)
    return None


def _validate_submission_key_pair(
    action_id: str,
    explicit_key: str | None,
    supplied_key: str | None,
) -> dict[str, Any] | None:
    """核对显式提交键与请求提交键，拒绝不一致或非法值。"""

    if explicit_key is None or supplied_key is None:
        return None
    try:
        if validate_submission_key(explicit_key) == validate_submission_key(supplied_key):
            return None
    except ValueError as exc:
        return _invalid(action_id, None, str(exc), code="invalid_submission_key")
    return _invalid(
        action_id,
        None,
        "请求 submission_key 与显式提交键不一致",
        code="submission_key_conflict",
    )


def _validate_scoped_action_identity(
    action_id: str,
    request_value: ActionRequest | dict[str, Any],
    operation_id: str | None,
    selected_submission_key: str | None,
    explicit_submission_key: str | None,
) -> dict[str, Any] | None:
    """检查 capability scope 的服务端 operation 与公开提交键规则。"""

    identity_request: Any = request_value
    if explicit_submission_key is not None and isinstance(request_value, Mapping):
        identity_request = {
            **dict(request_value),
            "submission_key": explicit_submission_key,
        }
    identity_error = scoped_submission_request_error(identity_request)
    if operation_id is not None and identity_error is None:
        identity_error = {
            "code": "operation_id_not_assignable",
            "message": "当前 scope 的 operation_id 由 MiniChem 服务端生成，调用方不得提供",
        }
    if selected_submission_key is None and identity_error is None:
        identity_error = {
            "code": "missing_submission_key",
            "message": "当前 capability scope 的有副作用请求必须携带 submission_key",
        }
    if identity_error is None:
        return None
    return _invalid(
        action_id,
        None,
        str(identity_error["message"]),
        code=str(identity_error["code"]),
    )


def _attach_selected_submission_key(
    request_value: ActionRequest | dict[str, Any],
    selected_submission_key: str | None,
) -> ActionRequest | dict[str, Any]:
    """把 scope 选定的提交键带入后续 ActionRequest，供 Gaussian FIFO 使用。"""

    if selected_submission_key is None:
        return request_value
    if isinstance(request_value, ActionRequest):
        if request_value.submission_key is None:
            return request_value.model_copy(
                update={"submission_key": selected_submission_key}
            )
        return request_value
    if isinstance(request_value, Mapping) and request_value.get("submission_key") is None:
        return {**dict(request_value), "submission_key": selected_submission_key}
    return request_value


def _resolve_action_submission_identity(
    action_id: str,
    request_value: ActionRequest | dict[str, Any],
    operation_id: str | None,
    submission_key: str | None,
) -> _ActionSubmissionIdentity | dict[str, Any]:
    """解析显式/请求身份并清理不可由调用方持久化的 operation_id。"""

    supplied_operation = _request_identity_field(request_value, "operation_id")
    if (
        operation_id is not None
        and supplied_operation is not None
        and supplied_operation != operation_id
    ):
        return _invalid(
            action_id,
            None,
            "request.operation_id 与显式 operation_id 不一致",
            code="operation_id_conflict",
        )
    effective_operation_id = operation_id or supplied_operation
    supplied_submission = _request_identity_field(request_value, "submission_key")
    key_error = _validate_submission_key_pair(
        action_id, submission_key, supplied_submission
    )
    if key_error is not None:
        return key_error
    selected_submission_key = submission_key or supplied_submission
    in_scope = current_scope() is not None
    if in_scope:
        scope_error = _validate_scoped_action_identity(
            action_id,
            request_value,
            effective_operation_id,
            selected_submission_key,
            submission_key,
        )
        if scope_error is not None:
            return scope_error
        effective_operation_id = None
    clean_request: ActionRequest | dict[str, Any] = request_value
    if isinstance(request_value, Mapping) and "operation_id" in request_value:
        clean_request = {
            key: value
            for key, value in request_value.items()
            if key != "operation_id"
        }
    if in_scope:
        clean_request = _attach_selected_submission_key(
            clean_request, selected_submission_key
        )
    return _ActionSubmissionIdentity(
        clean_request=clean_request,
        operation_id=effective_operation_id,
        submission_key=selected_submission_key,
        in_scope=in_scope,
    )


def _execute_idempotent_action_request(
    action_id: str,
    identity: _ActionSubmissionIdentity,
) -> dict[str, Any]:
    """执行无需幂等或按 scope/operation 收据保护的 Action 请求。"""

    if identity.operation_id is None and not identity.in_scope:
        return _execute_action_once(action_id, identity.clean_request)
    store = OperationIdempotencyStore()
    try:
        validated = (
            identity.clean_request
            if isinstance(identity.clean_request, ActionRequest)
            else ActionRequest.model_validate(identity.clean_request)
        )
    except ValidationError as exc:
        # 参数错误没有副作用，不写 running receipt；底层函数返回标准 invalid envelope。
        if identity.in_scope:
            return _invalid(action_id, None, str(exc))
        return _execute_action_once(action_id, identity.clean_request)
    validated = _bind_fixed_action_resources(
        validated,
        selected_backends=_resolve_action_resource_backend_specs(action_id, validated),
    )
    canonical_request = validated.model_dump(mode="json")
    canonical_request.pop("submission_key", None)
    fingerprint = calculate_request_fingerprint(action_id, canonical_request)
    try:
        if identity.in_scope:
            return dict(
                store.execute_scoped_submission(
                    validate_submission_key(str(identity.submission_key)),
                    fingerprint,
                    lambda: _execute_action_once_for_idempotent_request(
                        action_id,
                        validated,
                        resources_bound=True,
                    ),
                )
            )
        result = store.execute_idempotent_operation(
            identity.operation_id,
            fingerprint,
            lambda: _execute_action_once_for_idempotent_request(
                action_id,
                validated,
                resources_bound=True,
            ),
        )
        return dict(result)
    except _WorkerResourceStateUnknown as exc:
        # running receipt 故意保留；恢复前不能把未知 worker 当作一次安全失败重试。
        return dict(exc.response)


def execute_action(
    action_id: str,
    request_value: ActionRequest | dict[str, Any],
    *,
    operation_id: str | None = None,
    submission_key: str | None = None,
) -> dict[str, Any]:
    """执行同步 Action；scope 请求按公开提交键持久化幂等响应。"""

    identity = _resolve_action_submission_identity(
        action_id,
        request_value,
        operation_id,
        submission_key,
    )
    if isinstance(identity, dict):
        return identity
    return _execute_idempotent_action_request(action_id, identity)
