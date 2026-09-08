"""Investigation 异步作业的 Controller 等待与唤醒协议。

本模块保存模型 turn 内异步 MiniChem 作业的可恢复身份。模型在任一时刻只能保留一个
仍活跃且未 collect 的作业，但可在同一 turn 内顺序完成多个 submit/get/collect；Controller
只保存提交身份、轮询最小状态并在异常恢复时沿原 Codex thread 发送 wake 请求。这里不读取
科学 payload，不调用 collect，也不依赖模型侧提交接口；生产装配通过 Controller-side
status/arm 窄接口注入实际通道，测试可以使用内存 fake。
"""

from __future__ import annotations

import copy
import fcntl
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .prompts import build_investigation_job_wake_request
from .resume_contracts import (
    InvestigationResumeCheckpoint,
    ResumeContractError,
    canonical_json_digest,
    compute_resume_record_auth_digest,
    compute_resume_record_digest,
)


__all__ = [
    "INVESTIGATION_JOB_TOOL_NAMES",
    "INVESTIGATION_JOB_TOOL_FAMILIES",
    "TERMINAL_INVESTIGATION_JOB_STATUSES",
    "InvestigationJobIdentity",
    "InvestigationJobReceipt",
    "build_investigation_job_receipt",
    "extract_investigation_job_receipts",
    "InvestigationJobSubmissionError",
    "InvestigationJobTurnClosedError",
    "InvestigationJobStatusError",
    "InvestigationJobState",
    "InvestigationJobPollResult",
    "InvestigationJobGatewayProtocol",
    "InvestigationJobStatusValidatorProtocol",
    "InvestigationJobController",
    "InvestigationWaitingJobStore",
    "InvestigationResumeCheckpoint",
    "ResumeContractError",
]


# Investigation 模型只看到一个提交后的查询/收集接口。TS 仍可在服务端使用专用
# worker，但它的模型回执、继续请求和恢复请求都使用下面两个通用工具。
INVESTIGATION_JOB_TOOL_NAMES = (
    "submit_action_job",
    "submit_native_job",
    "submit_analysis_program",
    "get_execution_job",
    "collect_execution_job",
)
INVESTIGATION_JOB_TOOL_FAMILIES = {
    "transition_state_path": {
        "submit": "submit_action_job",
        "status": "get_execution_job",
        "collect": "collect_execution_job",
    },
    "native_execution": {
        "submit": "submit_native_job",
        "status": "get_execution_job",
        "collect": "collect_execution_job",
    },
    "action_execution": {
        "submit": "submit_action_job",
        "status": "get_execution_job",
        "collect": "collect_execution_job",
    },
    "programmable_analysis": {
        "submit": "submit_analysis_program",
        "status": "get_execution_job",
        "collect": "collect_execution_job",
    },
}
_LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES = {
    "transition_state_path": {
        "submit": "submit_transition_state_path_job",
        "status": "get_transition_state_path_job",
        "collect": "collect_transition_state_path_job",
    },
}
TERMINAL_INVESTIGATION_JOB_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "expired", "unknown"}
)
_NON_TERMINAL_STATUSES = frozenset({"queued", "pending", "running", "active", "submitted"})
_TRANSPORT_SUCCESS_STATUSES = frozenset({"success", "ok", "accepted"})
_JOB_ID_FIELD_ALIASES = ("job_id", "jobId")
_JOB_IDENTITY_FIELD_ALIASES = {
    "operation_id": ("operation_id", "operationId"),
    "submission_key": ("submission_key", "submissionKey"),
    "action_id": ("action_id", "actionId"),
    "backend_id": ("backend_id", "backendId"),
    "job_family": ("job_family", "jobFamily"),
    "job_type": ("job_type", "jobType"),
    "runtime": ("runtime",),
    "software_id": ("software_id", "softwareId"),
    "request_digest": ("request_digest", "requestDigest", "request_fingerprint"),
    "operation_fingerprint": ("operation_fingerprint", "operationFingerprint"),
}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_STATUS_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_WAITING_STORE_LOCKS: dict[str, Any] = {}
_WAITING_STORE_LOCKS_GUARD = threading.Lock()


def _require_text(value: Any, label: str) -> str:
    """校验作业状态机使用的非空标识，拒绝控制字符。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    normalized = value.strip()
    if _CONTROL_RE.search(normalized):
        raise ValueError(f"{label} 不能包含控制字符")
    return normalized


def _copy_json_object(value: Any, label: str) -> dict[str, Any]:
    """复制一个供 fake gateway 使用的 JSON object，不保存调用方可变引用。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        copied = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 不能规范化为 JSON object") from exc
    if not isinstance(copied, dict):  # pragma: no cover - 上面的输入检查保证
        raise TypeError(f"{label} 必须是 JSON object")
    return copied


@dataclass(frozen=True, slots=True)
class InvestigationJobIdentity:
    """一次提交的不可变身份；恢复和唤醒只能复用这一组值。"""

    job_id: str
    operation_id: str
    submission_key: str
    action_id: str | None = None
    backend_id: str | None = None
    job_family: str = "transition_state_path"
    job_type: str | None = None
    runtime: str | None = None
    software_id: str | None = None
    operation_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """校验作业、操作、提交、Action 和 backend 标识。"""

        for field_name in (
            "job_id",
            "operation_id",
            "submission_key",
        ):
            object.__setattr__(self, field_name, _require_text(getattr(self, field_name), field_name))
        family = _require_text(self.job_family, "job_family")
        if family not in INVESTIGATION_JOB_TOOL_FAMILIES:
            raise ValueError(f"不支持的异步作业 family: {family}")
        object.__setattr__(self, "job_family", family)
        if family in {"transition_state_path", "action_execution"}:
            for field_name in ("action_id", "backend_id"):
                object.__setattr__(
                    self,
                    field_name,
                    _require_text(getattr(self, field_name), field_name),
                )
            if family == "action_execution":
                for field_name in ("job_type", "runtime"):
                    object.__setattr__(
                        self,
                        field_name,
                        _require_text(getattr(self, field_name), field_name),
                    )
                if self.software_id is not None:
                    object.__setattr__(
                        self,
                        "software_id",
                        _require_text(self.software_id, "software_id"),
                    )
                if self.operation_fingerprint is not None:
                    fingerprint = _require_text(
                        self.operation_fingerprint,
                        "operation_fingerprint",
                    )
                    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                        raise ValueError("operation_fingerprint 必须是小写 SHA-256")
                    object.__setattr__(self, "operation_fingerprint", fingerprint)
        else:
            for field_name in ("job_type", "runtime"):
                object.__setattr__(
                    self,
                    field_name,
                    _require_text(getattr(self, field_name), field_name),
                )
            if self.software_id is not None:
                object.__setattr__(
                    self,
                    "software_id",
                    _require_text(self.software_id, "software_id"),
                )
            if self.operation_fingerprint is not None:
                fingerprint = _require_text(
                    self.operation_fingerprint,
                    "operation_fingerprint",
                )
                if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                    raise ValueError("operation_fingerprint 必须是小写 SHA-256")
                object.__setattr__(self, "operation_fingerprint", fingerprint)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str = "job_identity") -> "InvestigationJobIdentity":
        """从持久化 object 恢复完整作业身份，缺字段时 fail-closed。"""

        if not isinstance(value, Mapping):
            raise TypeError(f"{label} 必须是 JSON object")
        fields = {
            field_name: _require_text(value.get(field_name), f"{label}.{field_name}")
            for field_name in (
                "job_id",
                "operation_id",
                "submission_key",
            )
        }
        fields["job_family"] = _require_text(
            value.get("job_family", "transition_state_path"),
            f"{label}.job_family",
        )
        for field_name in (
            "action_id",
            "backend_id",
            "job_type",
            "runtime",
            "software_id",
            "operation_fingerprint",
        ):
            if field_name in value and value.get(field_name) is not None:
                fields[field_name] = _require_text(value[field_name], f"{label}.{field_name}")
        return cls(**fields)

    def as_mapping(self) -> dict[str, str]:
        """返回可写入等待快照和 wake prompt 的普通 object。"""

        result: dict[str, str] = {
            "job_id": self.job_id,
            "operation_id": self.operation_id,
            "submission_key": self.submission_key,
            "job_family": self.job_family,
        }
        for field_name in (
            "action_id",
            "backend_id",
            "job_type",
            "runtime",
            "software_id",
            "operation_fingerprint",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


@dataclass(frozen=True, slots=True)
class InvestigationJobReceipt:
    """来自真实 MCP ``item/completed`` 事件的已接受作业回执。

    回执只由 app-server 的工具事件构造，不接受模型最终文本中的身份字段。``arguments``
    保存提交工具实际收到的 JSON，用于生成请求指纹；``result`` 是服务端返回的最小
    作业身份，生产代码随后还会用只读状态工具再次核对。
    """

    tool_name: str
    identity: InvestigationJobIdentity
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]

    def __post_init__(self) -> None:
        """校验工具 family、回执 object 和身份之间的一致性。"""

        tool_name = _require_text(self.tool_name, "tool_name")
        family = self.identity.job_family
        expected = INVESTIGATION_JOB_TOOL_FAMILIES.get(family)
        legacy = _LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES.get(family)
        if expected is None or tool_name not in {
            expected["submit"],
            legacy["submit"] if legacy is not None else "",
        }:
            raise ValueError("submit 工具名与作业 family 不一致")
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "arguments", _copy_json_object(self.arguments, "submit.arguments"))
        object.__setattr__(self, "result", _copy_json_object(self.result, "submit.result"))

    @property
    def request_digest(self) -> str:
        """返回实际 submit 工具参数的稳定 SHA-256 指纹。"""

        server_fingerprint = self.result.get(
            "request_fingerprint",
            self.result.get("request_digest"),
        )
        if isinstance(server_fingerprint, str) and re.fullmatch(
            r"[0-9a-f]{64}", server_fingerprint
        ):
            return server_fingerprint
        return _request_digest(self.arguments)

    def as_mapping(self) -> dict[str, Any]:
        """返回可传给状态机的脱敏回执投影，不包含科学结果正文。"""

        return {
            "tool_name": self.tool_name,
            "identity": self.identity.as_mapping(),
            "arguments": dict(self.arguments),
            "result": dict(self.result),
            "request_digest": self.request_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InvestigationJobReceipt":
        """从 runtime 保存的真实回执投影恢复，不接受模型响应格式。"""

        if not isinstance(value, Mapping):
            raise TypeError("accepted_job 必须是 JSON object")
        identity = InvestigationJobIdentity.from_mapping(value.get("identity", {}))
        arguments = value.get("arguments", {})
        result = value.get("result", {})
        return cls(
            tool_name=value.get("tool_name"),
            identity=identity,
            arguments=arguments,
            result=result,
        )


def _read_tool_field(value: Any, field: str) -> Any:
    """读取 SDK MCP item 的 snake/camel 字段，兼容字典和 Pydantic 对象。"""

    aliases = {
        "job_id": "jobId",
        "operation_id": "operationId",
        "submission_key": "submissionKey",
        "action_id": "actionId",
        "backend_id": "backendId",
        "job_type": "jobType",
        "software_id": "softwareId",
        "structured_content": "structuredContent",
    }
    alias = aliases.get(field, field)
    if isinstance(value, Mapping):
        if field in value:
            return value[field]
        return value.get(alias)
    direct = getattr(value, field, None)
    if direct is not None:
        return direct
    return getattr(value, alias, None)


def _json_tool_result_payload(result: Any) -> dict[str, Any]:
    """从已完成 MCP item 的 result 提取服务端 JSON object。

    只解析工具回执中的 structuredContent 或 content text；这条路径不读取 assistant
    消息，因此不会把模型自报的 job 身份当成权威事实。
    """

    structured = _read_tool_field(result, "structured_content")
    candidates: list[Any] = [structured, result]
    content = _read_tool_field(result, "content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        for item in content:
            text = _read_tool_field(item, "text")
            if isinstance(text, str) and text.strip():
                candidates.append(text)
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            normalized = _copy_json_object(candidate, "submit.result")
            if any(key in normalized for key in ("job_id", "jobId")):
                return normalized
        elif isinstance(candidate, str):
            try:
                decoded = json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(decoded, Mapping):
                normalized = _copy_json_object(decoded, "submit.result")
                if any(key in normalized for key in ("job_id", "jobId")):
                    return normalized
    raise InvestigationJobSubmissionError(
        "submit MCP 回执没有包含可核对的 job_id"
    )


def _submit_was_rejected_before_job_creation(result: Any) -> bool:
    """判断 MiniChem 是否在创建作业前明确拒绝了提交。

    输入是真实 MCP completed item 的 result；只接受
    ``status=invalid_request``/``unavailable``/``failed``、不含任何 job 身份且错误码不
    表示“提交结果不确定”或“身份冲突”的结构化响应。返回 ``True``
    表示没有作业可保存，调用方应让模型读取错误并在同一 turn 修正参数；
    任何模糊状态都返回 ``False``，由严格回执校验失败关闭。
    """

    structured = _read_tool_field(result, "structured_content")
    candidates: list[Any] = [structured, result]
    content = _read_tool_field(result, "content")
    if isinstance(content, Sequence) and not isinstance(
        content,
        (str, bytes, bytearray),
    ):
        for item in content:
            text = _read_tool_field(item, "text")
            if isinstance(text, str) and text.strip():
                candidates.append(text)
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if not isinstance(candidate, Mapping):
            continue
        normalized = _copy_json_object(candidate, "submit.rejection")
        if any(key in normalized for key in ("job_id", "jobId")):
            continue
        error = normalized.get("error")
        code = error.get("code") if isinstance(error, Mapping) else None
        if (
            normalized.get("status") in {"invalid_request", "unavailable", "failed", "error"}
            and isinstance(code, str)
            and code.strip()
            and "uncertain" not in code.casefold()
            and "identity_conflict" not in code.casefold()
        ):
            return True
    return False


def _mcp_item_root(item: Any) -> Any:
    """读取 SDK RootModel 的 root，同时兼容字典夹具。"""

    root = _read_tool_field(item, "root")
    return root if root is not None else item


def build_investigation_job_receipt(item: Any) -> InvestigationJobReceipt | None:
    """从真实 ``mcpToolCall`` 完成 item 构造已验证的提交回执。

    非提交工具返回 ``None``；提交工具若状态、工具名、作业 family 或服务端身份不完整
    则抛出异常。该函数不会调用 MiniChem，也不会使用模型最终输出。
    """

    root = _mcp_item_root(item)
    item_type = _read_tool_field(root, "type")
    if getattr(item_type, "value", item_type) != "mcpToolCall":
        return None
    status = _read_tool_field(root, "status")
    status = getattr(status, "value", status)
    if status not in {None, "completed"}:
        return None
    tool_name = _read_tool_field(root, "tool")
    if not isinstance(tool_name, str):
        raise InvestigationJobSubmissionError("MCP submit item 缺少工具名")
    recognized_submit_tools = {
        tools["submit"] for tools in INVESTIGATION_JOB_TOOL_FAMILIES.values()
    }
    recognized_submit_tools.update(
        tools["submit"] for tools in _LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES.values()
    )
    if tool_name not in recognized_submit_tools:
        return None
    arguments = _read_tool_field(root, "arguments")
    arguments_object = _copy_json_object(arguments or {}, "submit.arguments")
    raw_result = _read_tool_field(root, "result")
    if _submit_was_rejected_before_job_creation(raw_result):
        return None
    result = _json_tool_result_payload(raw_result)
    action_id = result.get("action_id", result.get("actionId"))
    result_job_type = result.get("job_type", result.get("jobType"))
    request_payload = arguments_object.get("request", arguments_object)
    if not isinstance(request_payload, Mapping):
        request_payload = {}
    if tool_name == "submit_action_job":
        # submit_action_job 是 Investigation 唯一的 Action 提交入口；TS 仍由
        # 服务端专用 adapter 执行，但以返回的 action/job_type 身份区分 family。
        family = (
            "transition_state_path"
            if action_id == "search_transition_state_path"
            or result_job_type == "transition_state_path"
            else "action_execution"
        )
    elif tool_name == "submit_analysis_program":
        family = "programmable_analysis"
    elif tool_name == "submit_native_job":
        family = "native_execution"
    else:
        family = "transition_state_path"
    if family == "programmable_analysis":
        # 某些旧 supervisor 回执只返回公共 job_id；AnalysisJobRequest 中的
        # runtime 是稳定身份的一部分，可以作为缺失响应字段的受限回退来源。
        identity_job_type = result_job_type or "programmable_analysis"
        identity_runtime = result.get("runtime") or request_payload.get("runtime")
    elif family == "native_execution":
        identity_job_type = result_job_type or "native_software"
        identity_runtime = result.get("runtime") or request_payload.get("runtime")
    else:
        identity_job_type = result_job_type
        identity_runtime = result.get("runtime")
    try:
        identity = InvestigationJobIdentity(
            job_id=result.get("job_id", result.get("jobId")),
            operation_id=result.get("operation_id", result.get("operationId")),
            submission_key=result.get("submission_key", result.get("submissionKey")),
            action_id=result.get("action_id", result.get("actionId"))
            if family in {"transition_state_path", "action_execution"}
            else None,
            backend_id=result.get("backend_id", result.get("backendId"))
            if family in {"transition_state_path", "action_execution"}
            else None,
            job_family=family,
            job_type=identity_job_type
            if family in {"native_execution", "action_execution", "programmable_analysis"}
            else None,
            runtime=identity_runtime
            if family in {"native_execution", "action_execution", "programmable_analysis"}
            else None,
            software_id=result.get("software_id", result.get("softwareId"))
            if family in {"native_execution", "action_execution", "programmable_analysis"}
            else None,
            operation_fingerprint=(
                result.get("operation_fingerprint")
                if family in {"native_execution", "action_execution", "programmable_analysis"}
                else None
            ),
        )
    except (TypeError, ValueError) as error:
        raise InvestigationJobSubmissionError(
            "submit MCP 回执缺少可核对的完整 job identity"
        ) from error
    return InvestigationJobReceipt(
        tool_name=tool_name,
        identity=identity,
        arguments=arguments_object,
        result=result,
    )


def extract_investigation_job_receipts(items: Sequence[Any]) -> tuple[InvestigationJobReceipt, ...]:
    """提取本 turn 的全部真实 submit 回执，顺序活跃门禁由 runtime 负责。"""

    receipts: list[InvestigationJobReceipt] = []
    for item in items:
        receipt = build_investigation_job_receipt(item)
        if receipt is not None:
            receipts.append(receipt)
    return tuple(receipts)


class InvestigationJobSubmissionError(RuntimeError):
    """提交没有返回合法 job identity，当前 turn 已不可重试。"""


class InvestigationJobTurnClosedError(RuntimeError):
    """同一个模型 turn 仍有未 collect 的活跃作业，禁止并行第二次提交。"""


class InvestigationJobStatusError(RuntimeError):
    """Controller 收到无法解释或身份不一致的最小状态。"""


@dataclass(frozen=True, slots=True)
class InvestigationJobState:
    """可恢复的 Investigation 等待状态，不包含科学结果或 artifact payload。"""

    task_id: str
    batch_id: str
    combination_id: str
    codex_thread_id: str
    turn_id: str
    status: str
    identity: InvestigationJobIdentity
    request_digest: str
    scope_id: str = "unknown"
    job_family: str = "transition_state_path"
    # 直接构造且省略工具名的旧 Session 快照仍代表旧 TS Controller 路径。
    # 新状态不会依赖这些默认值：from_tool_receipt 会按真实回执显式写入
    # generic submit/get/collect；只有缺失工具字段的历史快照才会走这里。
    submit_tool: str = "submit_transition_state_path_job"
    status_tool: str = "get_transition_state_path_job"
    collect_tool: str = "collect_transition_state_path_job"
    # 新恢复合同把这些字段与旧的 request_digest 分开保存。旧 v1 记录没有它们时
    # 仍可读取；生产 submit 回执路径会显式写入全部可核对身份。
    submission_request_digest: str | None = None
    turn_request_digest: str | None = None
    thread_policy_digest: str | None = None
    # canonical v2 使用 turn_count_before；旧 turn_count 仅作为 v1 读取别名。
    turn_count_before: int | None = None
    turn_count: int | None = None
    session_id: str | None = None
    stage: str | None = None
    logical_operation_instance_id: str | None = None
    attempt_operation_instance_id: str | None = None
    attempt_number: int | None = None
    repair_request_number: int | None = None
    workspace_ref: str | None = None
    workspace_ref_digest: str | None = None
    # 提交作业的物理 turn 是不可变起点；continuation 只更新 turn_id 和
    # turn_started_event_ref，不能把起点改成当前 turn。旧 v1 快照没有这两个字段，
    # 只能走兼容读取，不能由读取方猜测。
    submit_turn_id: str | None = None
    submit_turn_started_event_ref: str | None = None
    turn_started_event_ref: str | None = None
    repair_request_event_ref: str | None = None
    expected_result_version: str | None = None
    # canonical v2 只保存一个由当前 scope capability 审计字段计算出的代次摘要；
    # 它绑定 token_sha256、完整组合身份和授权事件，但不保存 token 或路径。
    scope_credential_generation_digest: str | None = None
    # 旧 v1 字段仅供读取适配，不能进入 canonical ResumeCheckpoint。
    credential_record_digest: str | None = None
    credential_generation: int | None = None

    def __post_init__(self) -> None:
        """校验等待状态身份、状态值和请求指纹。"""

        for field_name in (
            "task_id",
            "batch_id",
            "combination_id",
            "codex_thread_id",
            "turn_id",
            "request_digest",
            "scope_id",
            "job_family",
            "submit_tool",
            "status_tool",
            "collect_tool",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )
        normalized_status = _require_text(self.status, "status")
        if normalized_status != "waiting_investigation_job":
            raise ValueError("InvestigationJobState.status 必须是 waiting_investigation_job")
        object.__setattr__(self, "status", normalized_status)
        explicit_submission_digest = self.submission_request_digest is not None
        submission_digest = self.submission_request_digest
        if not explicit_submission_digest:
            submission_digest = self.request_digest
        else:
            submission_digest = _require_text(
                submission_digest,
                "submission_request_digest",
            )
            if submission_digest != self.request_digest:
                raise ValueError(
                    "request_digest 只能作为 submission_request_digest 的旧别名"
                )
        object.__setattr__(self, "submission_request_digest", submission_digest)
        if explicit_submission_digest and re.fullmatch(
            r"[0-9a-f]{64}", submission_digest
        ) is None:
            raise ValueError("submission_request_digest 必须是小写 SHA-256")
        for field_name in (
            "turn_request_digest",
            "thread_policy_digest",
            "session_id",
            "stage",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "workspace_ref",
            "workspace_ref_digest",
            "submit_turn_id",
            "submit_turn_started_event_ref",
            "turn_started_event_ref",
            "repair_request_event_ref",
            "expected_result_version",
            "scope_credential_generation_digest",
            "credential_record_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _require_text(value, field_name))
        for field_name in ("turn_request_digest",):
            value = getattr(self, field_name)
            if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{field_name} 必须是小写 SHA-256")
        if self.thread_policy_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", self.thread_policy_digest
        ) is None:
            raise ValueError("thread_policy_digest 必须是小写 SHA-256")
        if self.turn_count_before is not None and (
            isinstance(self.turn_count_before, bool)
            or not isinstance(self.turn_count_before, int)
            or self.turn_count_before < 0
        ):
            raise ValueError("turn_count_before 必须是非负整数")
        if self.turn_count is not None and (
            isinstance(self.turn_count, bool)
            or not isinstance(self.turn_count, int)
            or self.turn_count < 0
        ):
            raise ValueError("legacy turn_count 必须是非负整数")
        if (
            self.turn_count_before is not None
            and self.turn_count is not None
            and self.turn_count_before != self.turn_count
        ):
            raise ValueError("turn_count_before 与 legacy turn_count 不一致")
        if self.stage is not None and self.stage not in {"initial", "repair"}:
            raise ValueError("stage 只能是 initial 或 repair")
        for field_name in ("attempt_number", "repair_request_number"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field_name} 必须是正整数")
        if self.scope_credential_generation_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", self.scope_credential_generation_digest
        ) is None:
            raise ValueError(
                "scope_credential_generation_digest 必须是小写 SHA-256"
            )
        if self.credential_record_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", self.credential_record_digest
        ) is None:
            raise ValueError("credential_record_digest 必须是小写 SHA-256")
        if self.credential_generation is not None and (
            isinstance(self.credential_generation, bool)
            or not isinstance(self.credential_generation, int)
            or self.credential_generation < 0
        ):
            raise ValueError("credential_generation 必须是非负整数")
        if self.stage == "initial" and self.repair_request_number is not None:
            raise ValueError("initial 状态不能携带 repair_request_number")
        if self.stage == "repair" and self.repair_request_number is None:
            raise ValueError("repair 状态必须携带 repair_request_number")
        if (self.submit_turn_id is None) != (self.submit_turn_started_event_ref is None):
            raise ValueError(
                "submit_turn_id 和 submit_turn_started_event_ref 必须同时存在"
            )
        family = _require_text(self.job_family, "job_family")
        tools = INVESTIGATION_JOB_TOOL_FAMILIES.get(family)
        if tools is None:
            raise ValueError(f"不支持的异步作业 family: {family}")
        expected_tools = {
            "submit_tool": tools["submit"],
            "status_tool": tools["status"],
            "collect_tool": tools["collect"],
        }
        legacy_tools = _LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES.get(family)
        accepted_tools = {
            field_name: {expected_name}
            for field_name, expected_name in expected_tools.items()
        }
        if legacy_tools is not None:
            accepted_tools = {
                "submit_tool": accepted_tools["submit_tool"] | {legacy_tools["submit"]},
                "status_tool": accepted_tools["status_tool"] | {legacy_tools["status"]},
                "collect_tool": accepted_tools["collect_tool"] | {legacy_tools["collect"]},
            }
        for field_name in ("submit_tool", "status_tool", "collect_tool"):
            tool_name = _require_text(getattr(self, field_name), field_name)
            if tool_name not in accepted_tools[field_name]:
                raise ValueError(f"{field_name} 与 job_family 不匹配")
        object.__setattr__(self, "job_family", family)

    @classmethod
    def from_tool_receipt(
        cls,
        receipt: InvestigationJobReceipt,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        codex_thread_id: str,
        turn_id: str,
        request_digest: str,
        scope_id: str = "unknown",
        job_family: str = "transition_state_path",
        submission_request_digest: str | None = None,
        turn_request_digest: str | None = None,
        thread_policy_digest: str | None = None,
        turn_count_before: int | None = None,
        turn_count: int | None = None,
        session_id: str | None = None,
        stage: str | None = None,
        logical_operation_instance_id: str | None = None,
        attempt_operation_instance_id: str | None = None,
        attempt_number: int | None = None,
        repair_request_number: int | None = None,
        workspace_ref: str | None = None,
        workspace_ref_digest: str | None = None,
        submit_turn_id: str | None = None,
        submit_turn_started_event_ref: str | None = None,
        turn_started_event_ref: str | None = None,
        repair_request_event_ref: str | None = None,
        expected_result_version: str | None = None,
        scope_credential_generation_digest: str | None = None,
        credential_record_digest: str | None = None,
        credential_generation: int | None = None,
    ) -> "InvestigationJobState":
        """把真实 submit 工具回执转换为独立等待状态，不登记科学结果。"""

        if not isinstance(receipt, InvestigationJobReceipt):
            raise TypeError("receipt 必须是 InvestigationJobReceipt")
        identity = receipt.identity
        family = identity.job_family
        family_tools = INVESTIGATION_JOB_TOOL_FAMILIES[family]
        legacy_tools = _LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES.get(family)
        if legacy_tools is not None and receipt.tool_name == legacy_tools["submit"]:
            family_tools = legacy_tools
        stable_submission_digest = (
            receipt.request_digest
            if submission_request_digest is None
            else submission_request_digest
        )
        if (submit_turn_id is None) != (submit_turn_started_event_ref is None):
            raise ValueError(
                "submit_turn_id 和 submit_turn_started_event_ref 必须同时提供"
            )
        if submit_turn_id is None and turn_started_event_ref is not None:
            # 真实 Started event 明确标记了本次 submit turn；这不是从旧记录推断，
            # 但只有在调用方没有提供部分 submit identity 时才可建立初始起点。
            submit_turn_id = turn_id
            submit_turn_started_event_ref = turn_started_event_ref
        return cls(
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            codex_thread_id=codex_thread_id,
            turn_id=turn_id,
            status="waiting_investigation_job",
            identity=identity,
            request_digest=stable_submission_digest,
            scope_id=scope_id,
            job_family=family,
            submit_tool=family_tools["submit"],
            status_tool=family_tools["status"],
            collect_tool=family_tools["collect"],
            submission_request_digest=stable_submission_digest,
            turn_request_digest=turn_request_digest,
            thread_policy_digest=thread_policy_digest,
            turn_count_before=turn_count_before,
            turn_count=turn_count,
            session_id=session_id,
            stage=stage,
            logical_operation_instance_id=logical_operation_instance_id,
            attempt_operation_instance_id=attempt_operation_instance_id,
            attempt_number=attempt_number,
            repair_request_number=repair_request_number,
            workspace_ref=workspace_ref,
            workspace_ref_digest=workspace_ref_digest,
            submit_turn_id=submit_turn_id,
            submit_turn_started_event_ref=submit_turn_started_event_ref,
            turn_started_event_ref=turn_started_event_ref,
            repair_request_event_ref=repair_request_event_ref,
            expected_result_version=expected_result_version,
            scope_credential_generation_digest=scope_credential_generation_digest,
            credential_record_digest=credential_record_digest,
            credential_generation=credential_generation,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InvestigationJobState":
        """从持久化等待快照恢复，并验证状态与完整 identity。"""

        if not isinstance(value, Mapping):
            raise TypeError("waiting_investigation_job 必须是 JSON object")
        return cls(
            task_id=value.get("task_id"),
            batch_id=value.get("batch_id"),
            combination_id=value.get("combination_id"),
            codex_thread_id=value.get("codex_thread_id"),
            turn_id=value.get("turn_id"),
            status=value.get("status"),
            identity=InvestigationJobIdentity.from_mapping(value.get("identity", value)),
            request_digest=value.get(
                "request_digest",
                value.get("submission_request_digest"),
            ),
            scope_id=value.get("scope_id", "unknown"),
            job_family=value.get("job_family", "transition_state_path"),
            submit_tool=value.get("submit_tool", "submit_action_job"),
            status_tool=value.get("status_tool", "get_execution_job"),
            collect_tool=value.get("collect_tool", "collect_execution_job"),
            submission_request_digest=value.get(
                "submission_request_digest",
            ),
            turn_request_digest=value.get(
                "turn_request_digest",
                value.get("codex_turn_digest"),
            ),
            thread_policy_digest=value.get("thread_policy_digest"),
            turn_count_before=value.get("turn_count_before"),
            turn_count=value.get("turn_count"),
            session_id=value.get("session_id"),
            stage=value.get("stage"),
            logical_operation_instance_id=value.get("logical_operation_instance_id"),
            attempt_operation_instance_id=value.get("attempt_operation_instance_id"),
            attempt_number=value.get("attempt_number"),
            repair_request_number=value.get("repair_request_number"),
            workspace_ref=value.get("workspace_ref"),
            workspace_ref_digest=value.get("workspace_ref_digest"),
            submit_turn_id=value.get("submit_turn_id"),
            submit_turn_started_event_ref=value.get(
                "submit_turn_started_event_ref"
            ),
            turn_started_event_ref=value.get("turn_started_event_ref"),
            repair_request_event_ref=value.get("repair_request_event_ref"),
            expected_result_version=value.get("expected_result_version"),
            scope_credential_generation_digest=value.get(
                "scope_credential_generation_digest"
            ),
            credential_record_digest=value.get("credential_record_digest"),
            credential_generation=value.get("credential_generation"),
        )

    def as_mapping(self) -> dict[str, Any]:
        """返回不含科学 payload 的可持久化等待快照。"""

        result: dict[str, Any] = {
            "status": self.status,
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "combination_id": self.combination_id,
            "codex_thread_id": self.codex_thread_id,
            "turn_id": self.turn_id,
            "submission_request_digest": self.submission_request_digest,
            "scope_id": self.scope_id,
            "job_family": self.job_family,
            "submit_tool": self.submit_tool,
            "status_tool": self.status_tool,
            "collect_tool": self.collect_tool,
            "identity": self.identity.as_mapping(),
        }
        # 旧 v1 读取和旧 Controller 调用仍保留别名；新 canonical v2 state 不再写入
        # 容易与 Codex turn 摘要混淆的 request_digest。
        if self.turn_request_digest is None:
            result["request_digest"] = self.request_digest
        for field_name in (
            "turn_request_digest",
            "thread_policy_digest",
            "turn_count_before",
            "session_id",
            "stage",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "attempt_number",
            "repair_request_number",
            "workspace_ref",
            "workspace_ref_digest",
            "submit_turn_id",
            "submit_turn_started_event_ref",
            "turn_started_event_ref",
            "repair_request_event_ref",
            "expected_result_version",
            "scope_credential_generation_digest",
        ):
            value = getattr(self, field_name)
            if field_name == "workspace_ref" and self.workspace_ref_digest is not None:
                # 新合同只保存 workspace 引用摘要；绝对路径只能在旧兼容记录中出现。
                continue
            if value is not None:
                result[field_name] = value
        if self.turn_count_before is None and self.turn_count is not None:
            # 旧 v1 state 才写回模糊别名；canonical v2 永不同时写两种字段。
            result["turn_count"] = self.turn_count
        # 仅对没有新代次摘要的旧内存状态保留 legacy 字段，canonical v2 不再写入
        # 含义不完整的 record_digest/generation 拆分。
        if self.scope_credential_generation_digest is None:
            if self.credential_record_digest is not None:
                result["credential_record_digest"] = self.credential_record_digest
            if self.credential_generation is not None:
                result["credential_generation"] = self.credential_generation
        return result

    def to_resume_checkpoint(
        self,
        *,
        provider_id: str = "codex",
        strict: bool = True,
    ) -> InvestigationResumeCheckpoint | None:
        """把等待状态转换为分组恢复合同，不读取科学结果或运行日志。

        新生产路径会写入 session、stage、attempt 和两个独立摘要；旧 v1 状态缺少这些
        字段时，``strict=False`` 返回 ``None``，由 sidecar 保留兼容读取而不是猜测身份。
        """

        try:
            if any(
                value is None
                for value in (
                    self.session_id,
                    self.stage,
                    self.logical_operation_instance_id,
                    self.attempt_operation_instance_id,
                    self.attempt_number,
                    self.turn_request_digest,
                    self.thread_policy_digest,
                    self.turn_count_before,
                    self.scope_credential_generation_digest,
                    self.submit_turn_id,
                    self.submit_turn_started_event_ref,
                )
            ):
                raise ResumeContractError("等待状态缺少新恢复合同字段")
            from .resume_contracts import (
                CapabilityCheckpoint,
                ExecutionCheckpoint,
                ProviderCheckpoint,
                ResumeSubject,
                TurnChainCheckpoint,
            )

            workspace_digest = self.workspace_ref_digest or (
                canonical_json_digest(self.workspace_ref)
                if self.workspace_ref is not None
                else None
            )
            if workspace_digest is None:
                raise ResumeContractError("等待状态缺少 workspace_ref_digest")

            subject = ResumeSubject(
                task_id=self.task_id,
                batch_id=self.batch_id,
                combination_id=self.combination_id,
                session_id=self.session_id,
                stage=self.stage,
                scope_id=self.scope_id,
                codex_thread_id=self.codex_thread_id,
                logical_operation_instance_id=self.logical_operation_instance_id,
                attempt_operation_instance_id=self.attempt_operation_instance_id,
                attempt_number=self.attempt_number,
                repair_request_number=self.repair_request_number,
                workspace_ref_digest=workspace_digest,
                repair_request_event_ref=self.repair_request_event_ref,
                expected_result_version=self.expected_result_version,
            )
            execution = ExecutionCheckpoint(
                job_id=self.identity.job_id,
                submission_operation_id=self.identity.operation_id,
                submission_key=self.identity.submission_key,
                submission_request_digest=self.submission_request_digest,
                job_family=self.identity.job_family,
                action_id=self.identity.action_id,
                backend_id=self.identity.backend_id,
                submission_operation_fingerprint=self.identity.operation_fingerprint,
            )
            provider = ProviderCheckpoint(
                provider_id=provider_id,
                codex_thread_id=self.codex_thread_id,
                codex_turn_id=self.turn_id,
                turn_request_digest=self.turn_request_digest,
                thread_policy_digest=self.thread_policy_digest,
                turn_count_before=self.turn_count_before,
            )
            if self.turn_started_event_ref is None:
                raise ResumeContractError("等待状态缺少真实 Started event ref")
            if self.submit_turn_id is None or self.submit_turn_started_event_ref is None:
                raise ResumeContractError(
                    "等待状态缺少不可变的原始 submit turn/event"
                )
            active_continuation_turn_id = None
            active_continuation_event_ref = None
            if self.turn_id != self.submit_turn_id:
                active_continuation_turn_id = self.turn_id
                active_continuation_event_ref = self.turn_started_event_ref
            turn_chain = TurnChainCheckpoint(
                submit_turn_id=self.submit_turn_id,
                submit_turn_started_event_ref=self.submit_turn_started_event_ref,
                active_continuation_turn_id=active_continuation_turn_id,
                active_continuation_turn_started_event_ref=active_continuation_event_ref,
            )
            capability = CapabilityCheckpoint(
                scope_id=self.scope_id,
                workspace_ref_digest=subject.workspace_ref_digest,
                scope_credential_generation_digest=(
                    self.scope_credential_generation_digest
                ),
            )
            return InvestigationResumeCheckpoint(
                subject=subject,
                turn_chain=turn_chain,
                execution=execution,
                provider=provider,
                capability=capability,
            )
        except (ResumeContractError, TypeError, ValueError):
            if strict:
                raise
            return None


@dataclass(frozen=True, slots=True)
class InvestigationJobPollResult:
    """一次 Controller 最小状态轮询结果，不携带计算结果内容。"""

    waiting: InvestigationJobState
    job_status: str
    terminal: bool
    wake_request: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """校验轮询状态只保存白名单状态和可选 wake request。"""

        normalized = _normalize_job_status(self.job_status)
        object.__setattr__(self, "job_status", normalized)
        expected_terminal = normalized in TERMINAL_INVESTIGATION_JOB_STATUSES
        if self.terminal != expected_terminal:
            raise ValueError("terminal 标志与 job_status 不一致")
        if self.wake_request is not None:
            object.__setattr__(self, "wake_request", _copy_json_object(self.wake_request, "wake_request"))


class InvestigationJobGatewayProtocol(Protocol):
    """MiniChem 侧提供给 Controller 的状态查询和 TS 终态 arm 接口。

    模型提交、查询和收集仍通过其 MCP 工具目录完成；Controller 只接收等待状态，
    查询最小作业状态，并在 TS 作业终态时开放同一作业的查询/收集门禁。
    """

    def get_investigation_job_status(
        self,
        waiting: InvestigationJobState,
    ) -> Mapping[str, Any]:
        """按等待状态的 scope 查询最小作业身份和状态。"""

    def arm_investigation_job(
        self,
        waiting: InvestigationJobState,
        terminal_status: str,
    ) -> Mapping[str, Any]:
        """在 TS 作业确认终态后 arm 同一作业的模型查询/收集门禁。"""


class InvestigationJobStatusValidatorProtocol(Protocol):
    """核对 Controller 状态响应与提交时冻结身份的注入接口。"""

    def __call__(
        self,
        waiting: InvestigationJobState,
        status_response: Mapping[str, Any],
    ) -> None:
        """身份、scope 和 request fingerprint 不一致时抛出异常。"""


def _normalize_job_status(value: Any) -> str:
    """把 MiniChem 常见状态映射到 Controller 的小状态集合。"""

    normalized = _require_text(value, "job_status").casefold().replace("-", "_")
    aliases = {
        "success": "succeeded",
        "completed": "succeeded",
        "complete": "succeeded",
        "done": "succeeded",
        "error": "failed",
        "failure": "failed",
        "cancelled_by_user": "cancelled",
        "canceled": "cancelled",
        "timeout": "expired",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TERMINAL_INVESTIGATION_JOB_STATUSES and normalized not in _NON_TERMINAL_STATUSES:
        if _SAFE_STATUS_RE.fullmatch(normalized) is None:
            raise InvestigationJobStatusError("job 状态格式非法")
        raise InvestigationJobStatusError(f"job 状态不在允许集合: {normalized}")
    return normalized


def _read_mapping_alias(value: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    """从状态 object 读取一组 snake/camel 别名中的第一个字段。"""

    for field_name in aliases:
        if field_name in value:
            return value[field_name]
    return None


def _try_normalize_known_job_status(value: Any) -> str | None:
    """仅把明确属于作业状态集合的值规范化，未知值保留为传输状态。"""

    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold().replace("-", "_")
    normalized = {
        "success": "succeeded",
        "completed": "succeeded",
        "complete": "succeeded",
        "done": "succeeded",
        "error": "failed",
        "failure": "failed",
        "cancelled_by_user": "cancelled",
        "canceled": "cancelled",
        "timeout": "expired",
    }.get(normalized, normalized)
    if normalized in TERMINAL_INVESTIGATION_JOB_STATUSES or normalized in _NON_TERMINAL_STATUSES:
        return normalized
    return None


def _validate_nested_job_identity_conflicts(
    value: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any] | None:
    """核对顶层与 ``job`` 嵌套中的重复身份和作业状态。

    输入是 Controller 或 arm 的窄状态响应；动作只检查重复字段，允许统一接口省略
    operation 等冗余身份。返回嵌套 object，发现明确冲突时抛出状态错误，避免后续
    ``dict.update`` 静默选择一份不可信的值。
    """

    nested = value.get("job")
    if nested is not None and not isinstance(nested, Mapping):
        raise InvestigationJobStatusError(f"{label}.job 必须是 object")
    if not isinstance(nested, Mapping):
        return None

    def reject_conflict(field_name: str, top_value: Any, nested_value: Any) -> None:
        """对两份同时存在的字段执行不相等拒绝。"""

        if top_value is None or nested_value is None:
            return
        if isinstance(top_value, str) and isinstance(nested_value, str):
            if top_value.strip() == nested_value.strip():
                return
        elif top_value == nested_value:
            return
        raise InvestigationJobStatusError(
            f"{label}.{field_name} 在顶层与 job 嵌套中的值冲突"
        )

    reject_conflict(
        "job_id",
        _read_mapping_alias(value, _JOB_ID_FIELD_ALIASES),
        _read_mapping_alias(nested, _JOB_ID_FIELD_ALIASES),
    )
    for field_name, aliases in _JOB_IDENTITY_FIELD_ALIASES.items():
        reject_conflict(
            field_name,
            _read_mapping_alias(value, aliases),
            _read_mapping_alias(nested, aliases),
        )

    top_status_values = [
        value[field_name]
        for field_name in ("job_status", "state")
        if field_name in value
    ]
    nested_status_values = [
        nested[field_name]
        for field_name in ("status", "state")
        if field_name in nested
    ]
    normalized_top = [_normalize_job_status(item) for item in top_status_values]
    normalized_nested = [_normalize_job_status(item) for item in nested_status_values]
    if len(set(normalized_top)) > 1 or len(set(normalized_nested)) > 1:
        raise InvestigationJobStatusError(f"{label} 中重复的 job 状态字段冲突")
    if normalized_top and normalized_nested and normalized_top[0] != normalized_nested[0]:
        raise InvestigationJobStatusError(
            f"{label}.job 状态与顶层 job_status/state 冲突"
        )

    # 顶层 status 在 MCP 中通常是 transport status；裸 success 不能覆盖嵌套
    # 作业状态。若它本身明确使用了作业状态值，则与嵌套 status 也必须一致。
    top_status = value.get("status")
    known_top_status = _try_normalize_known_job_status(top_status)
    if known_top_status is not None and normalized_nested:
        raw_top_status = top_status.strip().casefold().replace("-", "_") if isinstance(top_status, str) else ""
        if raw_top_status not in _TRANSPORT_SUCCESS_STATUSES and known_top_status != normalized_nested[0]:
            raise InvestigationJobStatusError(
                f"{label}.status 与 job.status 的作业状态冲突"
            )
    return nested


def _read_job_status_from_response(value: Mapping[str, Any], *, label: str) -> str:
    """从状态响应读取作业状态，隔离 transport status 与嵌套 job status。"""

    if not isinstance(value, Mapping):
        raise InvestigationJobStatusError(f"{label} 必须返回 object")
    nested = _validate_nested_job_identity_conflicts(value, label=label)
    top_status_values = [
        value[field_name]
        for field_name in ("job_status", "state")
        if field_name in value
    ]
    nested_status_values = (
        [
            nested[field_name]
            for field_name in ("status", "state")
            if field_name in nested
        ]
        if isinstance(nested, Mapping)
        else []
    )
    if top_status_values:
        status = _normalize_job_status(top_status_values[0])
    elif nested_status_values:
        status = _normalize_job_status(nested_status_values[0])
    else:
        raw_status = value.get("status")
        if isinstance(raw_status, str) and raw_status.strip().casefold() in _TRANSPORT_SUCCESS_STATUSES:
            raise InvestigationJobStatusError(
                f"{label} 只有 MCP transport success，缺少作业状态"
            )
        status = _normalize_job_status(raw_status)
    return status


def _read_minimal_job_status(value: Mapping[str, Any], expected_job_id: str) -> str:
    """从状态查询的顶层或 job object 读取状态，并拒绝混入结果 payload。"""

    if not isinstance(value, Mapping):
        raise InvestigationJobStatusError("get_execution_job 必须返回 object")
    _validate_nested_job_identity_conflicts(value, label="job status")
    returned_job_id = _read_mapping_alias(value, _JOB_ID_FIELD_ALIASES)
    nested = value.get("job")
    if returned_job_id is None and isinstance(nested, Mapping):
        returned_job_id = _read_mapping_alias(nested, _JOB_ID_FIELD_ALIASES)
    if returned_job_id is not None and _require_text(returned_job_id, "job_id") != expected_job_id:
        raise InvestigationJobStatusError("状态查询返回了不同的 job_id")
    return _read_job_status_from_response(value, label="job status")


def _request_digest(request: Mapping[str, Any]) -> str:
    """为提交 turn 生成稳定指纹，恢复时用于阻止不同请求重用原 job。"""

    try:
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("submission request 不能生成稳定指纹") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# Controller 的四个状态索引必须按同一组完整上下文加锁和寻址。类型别名保留在本模块
# 内部，避免把恢复合同字段暴露成新的公共协议。
_TurnIndexKey = tuple[
    str,
    str,
    str,
    str,
    str | None,
    str,
    str,
]
_JobIndexKey = tuple[_TurnIndexKey, str, str, str]


def _with_controller_lock(method: Callable[..., Any]) -> Callable[..., Any]:
    """让 Controller 的索引检查、回调和状态转移组成一个不可分割操作。"""

    @wraps(method)
    def locked_method(self: "InvestigationJobController", *args: Any, **kwargs: Any) -> Any:
        with self._state_lock:
            return method(self, *args, **kwargs)

    return locked_method


def _normalize_optional_identity(value: Any, label: str) -> str | None:
    """规范化可选的 session 身份；缺失值只保留为缺失，不猜测默认 session。"""

    if value is None:
        return None
    return _require_text(value, label)


@dataclass(slots=True)
class InvestigationJobController:
    """Controller 侧单作业状态机。

    submit 工具由模型直接调用；本类只接收 SDK 的真实回执并保存等待状态。
    ``poll_waiting_job`` 只调用 get 并从返回值读取状态，绝不 collect 或调用模型；终态
    时可选地生成 wake request 并交给注入的 ``wake_thread``。wake 去重在当前进程内以
    identity key 保存，生产持久化应把 wake receipt 与等待快照一起保存，重启时通过
    ``mark_wake_delivered`` 恢复该门禁。
    """

    gateway: InvestigationJobGatewayProtocol
    wake_thread: Callable[[Mapping[str, Any], str], Any] | None = None
    status_validator: InvestigationJobStatusValidatorProtocol | None = None
    require_status_identity: bool = False
    submission_validator: Callable[[InvestigationJobState, Mapping[str, Any]], None] | None = None
    require_submission_identity: bool = False
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )
    # 下列索引都以 task/batch/combination/scope/session/thread/turn 作为 turn key；
    # 同一 turn 的作业身份再由 job/operation/submission 三元组区分。这样不同任务即使
    # 复用了 provider thread 和 turn，也不会共享“已关闭”或“已 collect”事实。
    _closed_turns: set[_TurnIndexKey] = field(default_factory=set, init=False, repr=False)
    _accepted_states: dict[_TurnIndexKey, InvestigationJobState] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _pending_states: dict[_TurnIndexKey, InvestigationJobState] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _terminal_armed: set[_JobIndexKey] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _collected_jobs: set[_JobIndexKey] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    # 该集合是兼容旧测试/诊断代码的可读三元组视图；实际去重使用下面的完整索引。
    _wake_delivered: set[tuple[str, str, str]] = field(default_factory=set, init=False, repr=False)
    _wake_delivered_keys: set[_JobIndexKey] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    @staticmethod
    def _turn_key(
        *,
        task_id: Any,
        batch_id: Any,
        combination_id: Any,
        scope_id: Any,
        session_id: Any,
        codex_thread_id: Any,
        turn_id: Any,
    ) -> _TurnIndexKey:
        """生成包含全部任务与 provider 上下文的 turn 索引键。"""

        return (
            _require_text(task_id, "task_id"),
            _require_text(batch_id, "batch_id"),
            _require_text(combination_id, "combination_id"),
            _require_text(scope_id, "scope_id"),
            _normalize_optional_identity(session_id, "session_id"),
            _require_text(codex_thread_id, "codex_thread_id"),
            _require_text(turn_id, "turn_id"),
        )

    @classmethod
    def _state_turn_key(cls, state: InvestigationJobState) -> _TurnIndexKey:
        """从已校验等待状态生成完整 turn key。"""

        return cls._turn_key(
            task_id=state.task_id,
            batch_id=state.batch_id,
            combination_id=state.combination_id,
            scope_id=state.scope_id,
            session_id=state.session_id,
            codex_thread_id=state.codex_thread_id,
            turn_id=state.turn_id,
        )

    @classmethod
    def _state_job_key(cls, state: InvestigationJobState) -> _JobIndexKey:
        """生成包含完整 turn 上下文和作业提交身份的索引键。"""

        identity = state.identity
        return (
            cls._state_turn_key(state),
            identity.job_id,
            identity.operation_id,
            identity.submission_key,
        )

    @_with_controller_lock
    def restore_waiting_job(
        self,
        saved: Mapping[str, Any] | InvestigationJobState,
        *,
        request: Mapping[str, Any] | None = None,
    ) -> InvestigationJobState:
        """恢复已有作业身份，不重新提交，并重新打开本地 collect 门禁。

        恢复后的 get/collect 仍发生在模型 wake turn 内；因此 Controller 需要把
        sidecar 中的作业重新登记为该旧 turn 的 accepted state，才能在真实 collect
        后允许 wake turn 按顺序提交下一作业。该登记只更新进程内身份索引，不调用
        status、arm 或任何外部作业接口。
        """

        state = saved if isinstance(saved, InvestigationJobState) else InvestigationJobState.from_mapping(saved)
        if request is not None and _request_digest(request) != state.submission_request_digest:
            raise InvestigationJobStatusError("恢复 request 指纹与原提交不一致")
        state_key = self._state_turn_key(state)
        existing = self._accepted_states.get(state_key)
        if existing is not None:
            existing_job_key = self._state_job_key(existing)
            if existing.identity.as_mapping() != state.identity.as_mapping():
                if existing_job_key not in self._collected_jobs:
                    raise InvestigationJobStatusError(
                        "恢复 waiting 作业身份与当前完整 turn 不一致"
                    )
            elif existing.submission_request_digest != state.submission_request_digest:
                raise InvestigationJobStatusError(
                    "恢复 waiting request 指纹与当前作业不一致"
                )
        pending = self._pending_states.get(state_key)
        if pending is not None and pending.identity.as_mapping() != state.identity.as_mapping():
            if self._state_job_key(pending) not in self._collected_jobs:
                raise InvestigationJobStatusError(
                    "恢复 waiting 作业身份与 pending turn 不一致"
                )
        self._accepted_states[state_key] = state
        self._pending_states.pop(state_key, None)
        self._closed_turns.add(state_key)
        return state

    @_with_controller_lock
    def capture_submit_receipt(
        self,
        receipt: InvestigationJobReceipt,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        codex_thread_id: str,
        turn_id: str,
        scope_id: str = "unknown",
        persist_state: Callable[[InvestigationJobState], Any] | None = None,
        verify_status: bool = True,
        submission_request_digest: str | None = None,
        turn_request_digest: str | None = None,
        thread_policy_digest: str | None = None,
        turn_count_before: int | None = None,
        turn_count: int | None = None,
        session_id: str | None = None,
        stage: str | None = None,
        logical_operation_instance_id: str | None = None,
        attempt_operation_instance_id: str | None = None,
        attempt_number: int | None = None,
        repair_request_number: int | None = None,
        workspace_ref_digest: str | None = None,
        submit_turn_id: str | None = None,
        submit_turn_started_event_ref: str | None = None,
        turn_started_event_ref: str | None = None,
        repair_request_event_ref: str | None = None,
        expected_result_version: str | None = None,
        scope_credential_generation_digest: str | None = None,
    ) -> InvestigationJobState:
        """从真实 submit 回执建立等待状态，并按调用方选择是否查询最小状态。

        该方法不提交作业；它只接受 SDK 已完成的 submit 工具事件。调用方可提供
        ``persist_state``，让真实 job identity 在 status 通道短暂失败时仍先进入
        独立 waiting sidecar。生产模型自轮询路径把 ``verify_status=False``，因此
        submit receipt 的完整身份核对和 sidecar/Working Memory 持久化完成后立即返回，
        不会因为一次瞬时 status 错误中断 Harness；旧 Controller poll/arm 调用方保留
        默认 ``True`` 的同步核对。不同 identity 只有在同一 turn 仍有未 collect 的
        活跃作业时才会被拒绝；前一作业已完成真实 collect 后，当前 turn 可以继续接受
        下一作业。
        """

        if not isinstance(receipt, InvestigationJobReceipt):
            raise TypeError("receipt 必须是 InvestigationJobReceipt")
        normalized_turn = _require_text(turn_id, "turn_id")
        state_key = self._turn_key(
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            session_id=session_id,
            codex_thread_id=codex_thread_id,
            turn_id=normalized_turn,
        )
        existing_state = self._accepted_states.get(state_key)
        if existing_state is not None:
            expected_identity = receipt.identity.as_mapping()
            if existing_state.identity.as_mapping() == expected_identity:
                if existing_state.submission_request_digest != receipt.request_digest:
                    raise InvestigationJobStatusError(
                        "同一完整 turn 的 submit request 指纹发生漂移"
                    )
                if persist_state is not None:
                    persist_state(existing_state)
                return existing_state
            previous_key = self._state_job_key(existing_state)
            if previous_key not in self._collected_jobs:
                raise InvestigationJobTurnClosedError(
                    "当前 turn 仍有未 collect 的活跃计算任务，不能并行接受第二个 submit 回执"
                )
        pending_state = self._pending_states.get(state_key)
        if pending_state is not None:
            if pending_state.identity.as_mapping() != receipt.identity.as_mapping():
                raise InvestigationJobTurnClosedError(
                    "同一 turn 的 submit 回执身份发生漂移"
                )
            if pending_state.submission_request_digest != receipt.request_digest:
                raise InvestigationJobStatusError(
                    "同一完整 turn 的 pending submit request 指纹发生漂移"
                )
            state = pending_state
        else:
            previous_state = self._accepted_states.get(state_key)
            previous_key = (
                self._state_job_key(previous_state)
                if previous_state is not None
                else None
            )
            if state_key in self._closed_turns and (
                previous_state is None or previous_key not in self._collected_jobs
            ):
                raise InvestigationJobTurnClosedError(
                    "当前 turn 仍有未 collect 的活跃计算任务，不能并行接受第二个 submit 回执"
                )
            state = InvestigationJobState.from_tool_receipt(
                receipt,
                task_id=task_id,
                batch_id=batch_id,
                combination_id=combination_id,
                codex_thread_id=codex_thread_id,
                turn_id=normalized_turn,
                request_digest=receipt.request_digest,
                scope_id=scope_id,
                submission_request_digest=submission_request_digest,
                turn_request_digest=turn_request_digest,
                thread_policy_digest=thread_policy_digest,
                turn_count_before=turn_count_before,
                turn_count=turn_count,
                session_id=session_id,
                stage=stage,
                logical_operation_instance_id=logical_operation_instance_id,
                attempt_operation_instance_id=attempt_operation_instance_id,
                attempt_number=attempt_number,
                repair_request_number=repair_request_number,
                workspace_ref_digest=workspace_ref_digest,
                submit_turn_id=submit_turn_id,
                submit_turn_started_event_ref=submit_turn_started_event_ref,
                turn_started_event_ref=turn_started_event_ref,
                repair_request_event_ref=repair_request_event_ref,
                expected_result_version=expected_result_version,
                scope_credential_generation_digest=scope_credential_generation_digest,
            )
            # 先把本次 submit 身份放入可重试的 pending 槽位；status 暂时失败时
            # 保留这份真实身份，不把 turn 变成不可恢复的毒状态。
            self._closed_turns.add(state_key)
            self._pending_states[state_key] = state
        if persist_state is not None:
            persist_state(state)
        if not verify_status:
            self._accepted_states[state_key] = state
            self._pending_states.pop(state_key, None)
            return state
        raw_status = self._query_job_status(state)
        if self.require_status_identity and self.status_validator is None:
            self._validate_status_identity_fields(state, raw_status, label="submit status")
        if self.status_validator is not None:
            self.status_validator(state, raw_status)
        _read_minimal_job_status(raw_status, state.identity.job_id)
        self._accepted_states[state_key] = state
        self._pending_states.pop(state_key, None)
        return state

    @_with_controller_lock
    def mark_job_collected(self, waiting: InvestigationJobState) -> None:
        """登记真实 collect 已完成，释放同一 turn 接受下一顺序作业的门禁。"""

        if not isinstance(waiting, InvestigationJobState):
            raise TypeError("waiting 必须是 InvestigationJobState")
        key = self._state_turn_key(waiting)
        accepted = self._accepted_states.get(key)
        if accepted is None:
            accepted = self._pending_states.get(key)
        if accepted is None or accepted.identity.as_mapping() != waiting.identity.as_mapping():
            raise InvestigationJobTurnClosedError(
                "collect 作业身份与当前 turn 的 submit 回执不一致"
            )
        self._collected_jobs.add(self._state_job_key(waiting))

    def _query_job_status(
        self,
        waiting: InvestigationJobState,
    ) -> Mapping[str, Any]:
        """通过 Controller-side status API 查询最小状态，不调用模型工具。"""

        scoped_method = getattr(self.gateway, "get_investigation_job_status", None)
        if callable(scoped_method):
            response = scoped_method(waiting)
        else:
            # 兼容只提供单一只读 status 方法的窄测试替身；生产 gateway 必须实现
            # get_investigation_job_status，以便按 scope 绑定真实 Controller 通道。
            status_method = getattr(self.gateway, waiting.status_tool, None)
            if not callable(status_method):
                raise InvestigationJobStatusError(
                    "gateway 缺少 Controller-side 作业状态查询接口"
                )
            response = status_method(waiting.identity.job_id)
        if not isinstance(response, Mapping):
            raise InvestigationJobStatusError("Controller 作业状态响应必须是 object")
        return response

    def _arm_terminal_job(
        self,
        waiting: InvestigationJobState,
        terminal_status: str,
    ) -> None:
        """终态后核对一次 Controller 状态；旧 TS 状态保留 arm 兼容。"""

        key = (
            self._state_job_key(waiting)
        )
        if key in self._terminal_armed:
            return
        arm_method = getattr(self.gateway, "arm_investigation_job", None)
        if not callable(arm_method):
            if (
                waiting.job_family == "transition_state_path"
                and waiting.status_tool
                == _LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES["transition_state_path"]["status"]
            ):
                raise InvestigationJobStatusError(
                    "TS 作业缺少 Controller-side terminal arm 接口"
                )
            # 新协议的所有作业只需要 generic get_execution_job 状态。
            self._terminal_armed.add(key)
            return
        response = arm_method(waiting, terminal_status)
        if not isinstance(response, Mapping):
            raise InvestigationJobStatusError("Controller terminal arm 响应必须是 object")
        if self.require_status_identity and self.status_validator is None:
            self._validate_status_identity_fields(waiting, response, label="terminal arm")
        if self.status_validator is not None:
            self.status_validator(waiting, response)
        armed_status = _read_minimal_job_status(response, waiting.identity.job_id)
        if armed_status != terminal_status:
            raise InvestigationJobStatusError("terminal arm 返回的状态与轮询状态不一致")
        if (
            waiting.job_family == "transition_state_path"
            and waiting.status_tool
            == _LEGACY_INVESTIGATION_JOB_TOOL_FAMILIES["transition_state_path"]["status"]
        ):
            self._validate_transition_state_arm_response(
                response,
                waiting=waiting,
                terminal_status=terminal_status,
            )
        self._terminal_armed.add(key)

    @classmethod
    def _validate_transition_state_arm_response(
        cls,
        response: Mapping[str, Any],
        *,
        waiting: InvestigationJobState,
        terminal_status: str,
    ) -> None:
        """确认 TS 终态确实打开了查询门禁，再允许发送 wake。

        正常终态必须由 MiniChem Controller 明确返回 ``wake_armed=true`` 和
        ``phase=armed``。``unknown`` 只允许带有网关声明的 synthetic marker；这个
        例外表示原作业已经无法核对，不能伪装成已打开的作业门禁，也不能触发重提。
        其它 phase（特别是 ``collected``）一律停止，避免模型在未 arm 的作业上读取
        结果。
        """

        item = cls._job_object(response, "terminal arm")
        if (
            terminal_status == "unknown"
            and item.get("synthetic_terminal") == "unknown"
            and item.get("phase") == "unknown"
            and item.get("wake_armed") is not True
        ):
            return
        if item.get("wake_armed") is not True:
            raise InvestigationJobStatusError(
                "TS terminal arm 未明确返回 wake_armed=true"
            )
        if item.get("phase") != "armed":
            raise InvestigationJobStatusError(
                "TS terminal arm 必须返回 phase=armed"
            )

    @_with_controller_lock
    def poll_waiting_job(
        self,
        waiting: InvestigationJobState,
        *,
        terminal_message: str | None = None,
    ) -> InvestigationJobPollResult:
        """由 Controller 轮询一次最小状态，并在终态准备同 thread wake。

        输入是持久化等待状态；动作只调用统一 ``get_execution_job`` 一次，不调用
        collect、不读取科学结果、不调用模型。非终态返回无 wake 的 waiting 结果；终态
        返回包含原 job identity 的 wake request。重复调用不会重复发送已经记录的 wake。
        """

        raw_status = self._query_job_status(waiting)
        if self.require_status_identity and self.status_validator is None:
            self._validate_status_identity_fields(waiting, raw_status, label="status")
        if self.status_validator is not None:
            self.status_validator(waiting, raw_status)
        status = _read_minimal_job_status(raw_status, waiting.identity.job_id)
        if status not in TERMINAL_INVESTIGATION_JOB_STATUSES:
            return InvestigationJobPollResult(
                waiting=waiting,
                job_status=status,
                terminal=False,
            )
        key = self._state_job_key(waiting)
        if key in self._wake_delivered_keys:
            return InvestigationJobPollResult(
                waiting=waiting,
                job_status=status,
                terminal=True,
            )
        self._arm_terminal_job(waiting, status)
        wake_request = build_investigation_job_wake_request(
            waiting.task_id,
            waiting.batch_id,
            waiting.combination_id,
            waiting.codex_thread_id,
            job_id=waiting.identity.job_id,
            operation_id=waiting.identity.operation_id,
            submission_key=waiting.identity.submission_key,
            action_id=waiting.identity.action_id,
            backend_id=waiting.identity.backend_id,
            terminal_status=status,
            job_family=waiting.job_family,
            job_type=waiting.identity.job_type,
            runtime=waiting.identity.runtime,
            software_id=waiting.identity.software_id,
            scope_id=waiting.scope_id,
            terminal_message=terminal_message,
            available_tools=(
                waiting.submit_tool,
                waiting.status_tool,
                waiting.collect_tool,
            ),
        )
        if self.wake_thread is not None:
            self.wake_thread(wake_request, waiting.codex_thread_id)
            self._wake_delivered_keys.add(key)
            # 保留旧的可读三元组视图；它不参与去重，避免不同任务共享 thread/turn
            # 时误合并 wake 事实。
            self._wake_delivered.add(
                (waiting.identity.job_id, waiting.codex_thread_id, waiting.turn_id)
            )
        return InvestigationJobPollResult(
            waiting=waiting,
            job_status=status,
            terminal=True,
            wake_request=wake_request,
        )

    @staticmethod
    def _job_object(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
        """提取服务端状态 object，允许顶层和 job 嵌套两种窄协议。"""

        if not isinstance(value, Mapping):
            raise InvestigationJobStatusError(f"{label} 必须返回 object")
        nested = _validate_nested_job_identity_conflicts(value, label=label)
        if not isinstance(nested, Mapping):
            return value
        merged = dict(value)
        merged.update(nested)
        return merged

    @classmethod
    def _validate_status_identity_fields(
        cls,
        waiting: InvestigationJobState,
        response: Mapping[str, Any],
        *,
        label: str,
    ) -> None:
        """在生产持久化前逐字段核对 job、operation、scope 和请求指纹。"""

        item = cls._job_object(response, label)
        expected = {
            "job_id": waiting.identity.job_id,
            "operation_id": waiting.identity.operation_id,
            "submission_key": waiting.identity.submission_key,
            "job_family": waiting.identity.job_family,
        }
        if waiting.identity.job_family in {"transition_state_path", "action_execution"}:
            expected.update(
                {
                    "action_id": waiting.identity.action_id,
                    "backend_id": waiting.identity.backend_id,
                    # TS Controller 接口按 capability/workspace 核对作业归属；
                    # 若服务端额外返回 request_fingerprint，再进行可选比对。
                    "request_digest": waiting.request_digest,
                }
            )
            if waiting.identity.job_family == "action_execution":
                expected.update(
                    {
                        "job_type": waiting.identity.job_type,
                        "runtime": waiting.identity.runtime,
                    }
                )
        else:
            expected.update(
                {
                    "job_type": waiting.identity.job_type,
                    "runtime": waiting.identity.runtime,
                }
            )
        # Controller 已通过当前 scope 的 capability 通道按冻结 job_id 发起查询；
        # submit 回执和 waiting sidecar 保存完整 operation/submission/family 身份。
        # 状态接口因此只必须回传 job_id。其它冗余身份若出现仍逐字段拒绝冲突值，
        # 但不能因统一查询接口省略它们而中断已经提交的昂贵作业。
        server_required = {"job_id"}
        if (
            waiting.identity.job_family != "transition_state_path"
            and waiting.identity.operation_fingerprint is not None
        ):
            expected["operation_fingerprint"] = waiting.identity.operation_fingerprint
        for field_name, expected_value in expected.items():
            aliases = (
                _JOB_ID_FIELD_ALIASES
                if field_name == "job_id"
                else _JOB_IDENTITY_FIELD_ALIASES.get(field_name, (field_name,))
            )
            if field_name not in server_required and not any(
                alias in item for alias in aliases
            ):
                continue
            actual = _read_mapping_alias(item, aliases)
            if field_name == "request_digest" and actual is None:
                actual = _read_mapping_alias(item, ("request_fingerprint",))
            if field_name not in server_required and actual is None:
                continue
            if not isinstance(actual, str) or not actual.strip():
                raise InvestigationJobStatusError(
                    f"{label} 缺少用于恢复核对的 {field_name}"
                )
            if actual.strip() != expected_value:
                raise InvestigationJobStatusError(
                    f"{label}.{field_name} 与冻结作业身份不一致"
                )

    @_with_controller_lock
    def mark_wake_delivered(self, waiting: InvestigationJobState) -> None:
        """恢复时登记已持久化的 wake receipt，防止重启重复唤醒模型。"""

        key = self._state_job_key(waiting)
        self._wake_delivered_keys.add(key)
        self._wake_delivered.add(
            (waiting.identity.job_id, waiting.codex_thread_id, waiting.turn_id)
        )


class InvestigationWaitingJobStore:
    """把 waiting_investigation_job 独立保存为可恢复的原子 JSON 快照。

    快照只保存作业身份、scope、请求指纹和 wake 送达事实，不保存科学结果或 artifact。
    写入使用临时文件、fsync 和 replace；恢复时每条记录都重新通过
    ``InvestigationJobState`` 校验，损坏或身份重复会 fail-closed。
    """

    schema = "arche.investigation.waiting_job"
    # v1 记录仍可读取；所有新写入都会带明确 record_digest 和可选分组 checkpoint。
    version = 2
    legacy_version = 1

    _DOCUMENT_FIELDS = frozenset({"schema", "version", "records"})
    _RECORD_FIELDS = frozenset(
        {
            "state",
            "resume_checkpoint",
            "legacy_source_version",
            "legacy_source_record_digest",
            "legacy_source_record_auth_digest",
            "incomplete_v2_resume_event_ref",
            "incomplete_v2_source_record_digest",
            "incomplete_v2_source_record_auth_digest",
            "pending_replacement_history",
            "wake_status",
            "terminal_status",
            "terminal_message",
            "wake_request",
            "wake_request_digest",
            "wake_receipt",
            "source_wake_state",
            "source_wake_receipt",
            "collected_source",
            "record_digest",
            "record_auth_digest",
        }
    )
    _LEGACY_WRITEBACK_SOURCE_FIELDS = frozenset(
        {
            "legacy_source_version",
            "legacy_source_record_digest",
            "legacy_source_record_auth_digest",
        }
    )
    _INCOMPLETE_V2_SOURCE_FIELDS = frozenset(
        {
            "incomplete_v2_resume_event_ref",
            "incomplete_v2_source_record_digest",
            "incomplete_v2_source_record_auth_digest",
        }
    )
    _V1_RECORD_FIELDS = (
        _RECORD_FIELDS
        - _LEGACY_WRITEBACK_SOURCE_FIELDS
        - _INCOMPLETE_V2_SOURCE_FIELDS
        - frozenset({"pending_replacement_history"})
    )
    _LEGACY_REPLACEMENT_EDGE_FIELDS = frozenset(
        {
            "predecessor",
            "successor",
            "predecessor_record_auth_digest",
            "authorization_kind",
        }
    )
    _LEGACY_REPLACEMENT_IDENTITY_FIELDS = frozenset(
        {
            "task_id",
            "batch_id",
            "combination_id",
            "scope_id",
            "codex_thread_id",
            "turn_id",
            "submission_request_digest",
            "job_family",
            "identity",
        }
    )
    _DECISION_FIELDS = frozenset(
        {
            "wake_status",
            "terminal_status",
            "terminal_message",
            "wake_request",
            "wake_request_digest",
            "wake_receipt",
        }
    )
    _SOURCE_FIELDS = frozenset({"source_wake_state", "source_wake_receipt"})
    _STATE_FIELDS = frozenset(
        {
            "status",
            "task_id",
            "batch_id",
            "combination_id",
            "codex_thread_id",
            "turn_id",
            "request_digest",
            "submission_request_digest",
            "turn_request_digest",
            "thread_policy_digest",
            "turn_count_before",
            "turn_count",
            # 旧 v1 的别名只允许读取，v2 canonical checkpoint 不使用它。
            "codex_turn_digest",
            "scope_id",
            "job_family",
            "submit_tool",
            "status_tool",
            "collect_tool",
            "identity",
            "session_id",
            "stage",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "attempt_number",
            "repair_request_number",
            "workspace_ref",
            "submit_turn_id",
            "submit_turn_started_event_ref",
            "turn_started_event_ref",
            "repair_request_event_ref",
            "expected_result_version",
            "workspace_ref_digest",
            "scope_credential_generation_digest",
            "credential_record_digest",
            "credential_generation",
        }
    )
    _IDENTITY_FIELDS = frozenset(
        {
            "job_id",
            "operation_id",
            "submission_key",
            "job_family",
            "action_id",
            "backend_id",
            "job_type",
            "runtime",
            "software_id",
            "operation_fingerprint",
        }
    )

    def __init__(self, root: Path | str, *, filename: str = "waiting_investigation_jobs.json") -> None:
        """初始化 Bundle internal/state 下的等待快照位置，不读取科学资料。"""

        path = Path(root)
        if not path.is_absolute() or path == Path("/"):
            raise ValueError("waiting job store root 必须是非根绝对路径")
        self.root = path.resolve(strict=False)
        self.path = self.root / filename
        if self.path.parent != self.root or not filename.endswith(".json"):
            raise ValueError("waiting job store 文件名非法")
        self.root.mkdir(parents=True, exist_ok=True)
        # 原子 replace 只能保证单个写入不出现半截文件，不能保护两个进程各自读取旧
        # 文档后互相覆盖。锁文件由所有进程共享，覆盖整个 read-modify-write 事务。
        self.lock_path = self.root / f".{filename}.lock"
        self._lock_fd = os.open(
            self.lock_path,
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        self._file_lock_depth = 0
        # 一个协调器会并行处理多个 Investigation 组合；所有 read-modify-write
        # 事务必须在进程内串行，否则最后一个 replace 会覆盖其他组合的等待记录。
        # 同一路径即使被恢复代码构造出多个 store 实例，也必须共用同一把锁。
        lock_key = str(self.path)
        with _WAITING_STORE_LOCKS_GUARD:
            self._lock = _WAITING_STORE_LOCKS.setdefault(lock_key, threading.RLock())

    @contextmanager
    def _storage_lock(self):
        """同时取得进程内和跨进程锁，保护一次完整的 sidecar 事务。"""

        with self._lock:
            if self._file_lock_depth == 0:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
                except OSError as exc:
                    raise InvestigationJobStatusError(
                        "waiting job 快照锁无法取得"
                    ) from exc
            self._file_lock_depth += 1
            try:
                yield
            finally:
                self._file_lock_depth -= 1
                if self._file_lock_depth == 0:
                    try:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    except OSError as exc:
                        raise InvestigationJobStatusError(
                            "waiting job 快照锁无法释放"
                        ) from exc

    @staticmethod
    def _key(state: InvestigationJobState) -> str:
        """用结构化摘要建立不可碰撞的 task/batch/combination 键。"""

        return "v2:" + canonical_json_digest(
            {
                "task_id": state.task_id,
                "batch_id": state.batch_id,
                "combination_id": state.combination_id,
            }
        )

    @staticmethod
    def _legacy_key(state: InvestigationJobState) -> str:
        """返回 v1 使用的分隔符键，仅用于严格读取旧快照。"""

        return "|".join((state.task_id, state.batch_id, state.combination_id))

    @classmethod
    def _key_for_values(
        cls,
        task_id: str,
        batch_id: str,
        combination_id: str,
    ) -> str:
        """按已校验值生成与 ``_key`` 相同的结构化键。"""

        return "v2:" + canonical_json_digest(
            {
                "task_id": task_id,
                "batch_id": batch_id,
                "combination_id": combination_id,
            }
        )

    def _read_document(self) -> dict[str, Any]:
        """读取并校验快照文档；不存在时返回空文档。"""

        if not self.path.exists():
            return {"schema": self.schema, "version": self.version, "records": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise InvestigationJobStatusError("waiting job 快照无法读取") from exc
        if not isinstance(raw, Mapping) or raw.get("schema") != self.schema:
            raise InvestigationJobStatusError("waiting job 快照 schema/version 不匹配")
        version = raw.get("version")
        if version not in {self.legacy_version, self.version}:
            raise InvestigationJobStatusError("waiting job 快照 schema/version 不匹配")
        unknown_document_fields = set(raw) - set(self._DOCUMENT_FIELDS)
        if unknown_document_fields:
            raise InvestigationJobStatusError(
                "waiting job 快照含未知顶层字段"
            )
        records = raw.get("records")
        if not isinstance(records, Mapping):
            raise InvestigationJobStatusError("waiting job 快照 records 必须是 object")
        normalized_records: dict[str, Any] = {}
        for key, record in records.items():
            if not isinstance(key, str) or not isinstance(record, Mapping):
                raise InvestigationJobStatusError("waiting job 记录 key/value 非法")
            normalized = dict(record)
            if version == self.version:
                self._validate_v2_record(normalized, label=f"record[{key}]")
                state_mapping = normalized.get("state")
                if not isinstance(state_mapping, Mapping):  # pragma: no cover - validator 已检查
                    raise InvestigationJobStatusError("waiting job state 必须是 object")
                parsed_state = InvestigationJobState.from_mapping(state_mapping)
                canonical_key = self._key(parsed_state)
                # 早期 v2 曾沿用 v1 的 pipe key。只有当 key 能由 record 内完整身份
                # 唯一重建时才接受；不拆分字符串，避免包含 pipe 的 ID 被猜错。
                if key not in {canonical_key, self._legacy_key(parsed_state)}:
                    raise InvestigationJobStatusError(
                        "waiting job 记录 key 与完整身份不一致"
                    )
                normalized["record_digest"] = self._validate_record_digests(
                    normalized,
                    label=f"record[{key}]",
                    require_auth=False,
                )
                if "record_auth_digest" not in normalized:
                    # 旧 v2 没有独立认证摘要。只有尚未进入任何 wake/collect 决策的
                    # 纯 pending 记录可以安全升级；一旦已有终态、request、receipt、
                    # source 或 replacement 许可，缺少认证摘要就无法判断字段是否漂移。
                    unsafe_unauthenticated_fields = (
                        self._DECISION_FIELDS
                        | self._SOURCE_FIELDS
                        | self._LEGACY_WRITEBACK_SOURCE_FIELDS
                        | self._INCOMPLETE_V2_SOURCE_FIELDS
                        | frozenset(
                            {
                                "collected_source",
                                "pending_replacement_history",
                            }
                        )
                    ) - frozenset({"wake_status"})
                    if normalized.get("wake_status", "pending") != "pending" or any(
                        field_name in normalized
                        for field_name in unsafe_unauthenticated_fields
                    ):
                        raise InvestigationJobStatusError(
                            "旧 v2 waiting job 的恢复决策缺少认证摘要"
                        )
                    normalized["record_auth_digest"] = compute_resume_record_auth_digest(
                        normalized
                    )
                canonical_record = normalized_records.get(canonical_key)
                if canonical_record is not None:
                    raise InvestigationJobStatusError(
                        "waiting job 存在重复的 task/batch/combination 身份"
                    )
                normalized_records[canonical_key] = normalized
            else:
                self._validate_v1_record(normalized, label=f"record[{key}]")
                # v1 没有 record digest；读取时升级内存中的 envelope，下一次写入会
                # 原子转换为 v2，但不会补造缺失的 WM/Provider 身份。
                if "state" not in normalized:
                    normalized = {"state": dict(normalized)}
                normalized.pop("record_digest", None)
                state_mapping = normalized.get("state")
                if not isinstance(state_mapping, Mapping):  # pragma: no cover - validator 已检查
                    raise InvestigationJobStatusError("waiting job state 必须是 object")
                parsed_state = InvestigationJobState.from_mapping(state_mapping)
                if key != self._legacy_key(parsed_state):
                    raise InvestigationJobStatusError(
                        "v1 waiting job 记录 key 与完整身份不一致"
                    )
                canonical_key = self._key(parsed_state)
                if canonical_key in normalized_records:
                    raise InvestigationJobStatusError(
                        "waiting job 存在重复的 task/batch/combination 身份"
                    )
                normalized["record_digest"] = compute_resume_record_digest(normalized)
                normalized["record_auth_digest"] = compute_resume_record_auth_digest(
                    normalized
                )
                normalized_records[canonical_key] = normalized
        return {
            "schema": self.schema,
            "version": self.version,
            "records": normalized_records,
        }

    @classmethod
    def _validate_record_digests(
        cls,
        record: Mapping[str, Any],
        *,
        label: str,
        require_auth: bool,
    ) -> str:
        """重算 identity/auth 摘要，并验证 wake 决策字段的结构和相互关系。"""

        expected_digest = compute_resume_record_digest(record)
        actual_digest = record.get("record_digest")
        if actual_digest != expected_digest:
            raise InvestigationJobStatusError(f"{label} record_digest 不匹配")
        actual_auth = record.get("record_auth_digest")
        if require_auth and (
            not isinstance(actual_auth, str)
            or actual_auth != compute_resume_record_auth_digest(record)
        ):
            raise InvestigationJobStatusError(
                f"{label} recovery decision 摘要不匹配"
            )
        if actual_auth is not None and (
            not isinstance(actual_auth, str)
            or actual_auth != compute_resume_record_auth_digest(record)
        ):
            raise InvestigationJobStatusError(
                f"{label} recovery decision 摘要不匹配"
            )
        cls._validate_decision_fields(record, label=label)
        return expected_digest

    @classmethod
    def _validate_decision_fields(
        cls,
        record: Mapping[str, Any],
        *,
        label: str,
    ) -> None:
        """校验 wake request、终态和 receipt 的最小恢复结构。"""

        wake_status = record.get("wake_status", "pending")
        if not isinstance(wake_status, str) or wake_status not in {
            "pending",
            "intent",
            "delivered",
            "in_flight",
        }:
            raise InvestigationJobStatusError(f"{label}.wake_status 非法")
        terminal_status = record.get("terminal_status")
        if terminal_status is not None:
            try:
                normalized_terminal = _normalize_job_status(terminal_status)
            except (TypeError, ValueError, InvestigationJobStatusError) as exc:
                raise InvestigationJobStatusError(
                    f"{label}.terminal_status 非法"
                ) from exc
            if normalized_terminal not in TERMINAL_INVESTIGATION_JOB_STATUSES:
                raise InvestigationJobStatusError(
                    f"{label}.terminal_status 必须是终态"
                )
        request = record.get("wake_request")
        request_digest = record.get("wake_request_digest")
        if request is None and request_digest is not None:
            raise InvestigationJobStatusError(
                f"{label}.wake_request_digest 缺少对应 wake_request"
            )
        if request is not None:
            if not isinstance(request, Mapping):
                raise InvestigationJobStatusError(f"{label}.wake_request 必须是 object")
            try:
                expected_request_digest = _request_digest(request)
            except (TypeError, ValueError) as exc:
                raise InvestigationJobStatusError(
                    f"{label}.wake_request 无法生成摘要"
                ) from exc
            if not isinstance(request_digest, str) or request_digest != expected_request_digest:
                raise InvestigationJobStatusError(
                    f"{label}.wake_request_digest 不匹配"
                )
        if wake_status in {"intent", "delivered", "in_flight"} and (
            terminal_status is None or request is None
        ):
            raise InvestigationJobStatusError(
                f"{label} wake handoff 缺少已核对 terminal_status/request"
            )
        for field_name in ("wake_receipt", "source_wake_state", "source_wake_receipt"):
            value = record.get(field_name)
            if value is not None and not isinstance(value, Mapping):
                raise InvestigationJobStatusError(
                    f"{label}.{field_name} 必须是 object"
                )
        if record.get("source_wake_receipt") is not None and record.get(
            "source_wake_state"
        ) is None:
            raise InvestigationJobStatusError(
                f"{label}.source_wake_receipt 缺少 source_wake_state"
            )
        collected_source = record.get("collected_source")
        if collected_source is not None:
            if not isinstance(collected_source, Mapping):
                raise InvestigationJobStatusError(
                    f"{label}.collected_source 必须是 object"
                )
            state_mapping = record.get("state")
            if not isinstance(state_mapping, Mapping):
                raise InvestigationJobStatusError(
                    f"{label}.collected_source 缺少 state"
                )
            state = InvestigationJobState.from_mapping(state_mapping)
            expected_collection = {
                "job_id": state.identity.job_id,
                "operation_id": state.identity.operation_id,
                "submission_key": state.identity.submission_key,
                "thread_id": state.codex_thread_id,
                "turn_id": state.turn_id,
            }
            if dict(collected_source) != expected_collection:
                raise InvestigationJobStatusError(
                    f"{label}.collected_source 与当前作业身份不一致"
                )

    @classmethod
    def _validate_v1_record(cls, record: Mapping[str, Any], *, label: str) -> None:
        """校验旧 v1 已知记录结构，兼容读取但不接受未知字段。"""

        # 最早的 v1 文件把 state 字段直接写在 record 顶层；它不是 v2 记录缺少
        # state，而是一个合法的旧平面记录。先按旧 state 白名单核对，不能把合法
        # 旧字段误判为 v2 未知字段。
        if "state" not in record:
            cls._validate_state_shape(record, label=label, legacy=True)
            if "submission_request_digest" in record:
                raise InvestigationJobStatusError(
                    f"{label}.submission_request_digest 只允许 v2 记录"
                )
            return
        unknown = set(record) - (
            set(cls._V1_RECORD_FIELDS) - {"record_digest", "resume_checkpoint"}
        )
        if unknown:
            raise InvestigationJobStatusError(f"{label} 含未知字段")
        state = record.get("state", record)
        if not isinstance(state, Mapping):
            raise InvestigationJobStatusError(f"{label}.state 必须是 object")
        cls._validate_state_shape(state, label=f"{label}.state", legacy=True)
        if "submission_request_digest" in state:
            raise InvestigationJobStatusError(
                f"{label}.state.submission_request_digest 只允许 v2 记录"
            )

    @classmethod
    def _validate_v2_record(cls, record: Mapping[str, Any], *, label: str) -> None:
        """严格核对 v2 record、state 和 canonical checkpoint。"""

        unknown = set(record) - set(cls._RECORD_FIELDS)
        if unknown:
            raise InvestigationJobStatusError(f"{label} 含未知字段")
        state = record.get("state")
        if not isinstance(state, Mapping):
            raise InvestigationJobStatusError(f"{label}.state 必须是 object")
        cls._validate_state_shape(state, label=f"{label}.state", legacy=False)
        if "codex_turn_digest" in state:
            raise InvestigationJobStatusError(
                f"{label}.state 的 codex_turn_digest 只允许 v1 兼容读取"
            )
        checkpoint = record.get("resume_checkpoint")
        canonical_identity_fields = {
            "session_id",
            "stage",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "attempt_number",
            "turn_request_digest",
            "thread_policy_digest",
            "turn_count_before",
            "turn_started_event_ref",
            "submit_turn_id",
            "submit_turn_started_event_ref",
            "workspace_ref_digest",
            "scope_credential_generation_digest",
        }
        present_identity_fields = canonical_identity_fields.intersection(state)
        if present_identity_fields and present_identity_fields != canonical_identity_fields:
            missing = sorted(canonical_identity_fields - present_identity_fields)
            raise InvestigationJobStatusError(
                f"{label}.state canonical identity 必须 all-or-none，缺少: {', '.join(missing)}"
            )
        has_new_identity = bool(present_identity_fields)
        if has_new_identity and "turn_count" in state:
            raise InvestigationJobStatusError(
                f"{label}.state canonical v2 不得携带 turn_count 旧别名"
            )
        if checkpoint is None:
            if has_new_identity:
                raise InvestigationJobStatusError(
                    f"{label} 缺少分组 resume_checkpoint"
                )
        else:
            try:
                parsed = InvestigationResumeCheckpoint.from_mapping(
                    checkpoint,
                    label=f"{label}.resume_checkpoint",
                )
                if "request_digest" in state:
                    raise InvestigationJobStatusError(
                        f"{label}.state 不得在 canonical v2 携带 request_digest 别名"
                    )
                cls._validate_checkpoint_matches_state(
                    state,
                    parsed,
                    label=label,
                )
            except (ResumeContractError, TypeError, ValueError) as exc:
                raise InvestigationJobStatusError(
                    f"{label}.resume_checkpoint 身份不完整"
                ) from exc
        legacy_source = cls._validate_legacy_writeback_source(
            record,
            label=label,
        )
        incomplete_v2_source = cls._validate_incomplete_v2_authorization_source(
            record,
            label=label,
        )
        if legacy_source is not None and incomplete_v2_source is not None:
            raise InvestigationJobStatusError(
                f"{label} 不得同时保留 v1 与 incomplete v2 恢复来源"
            )
        cls._validate_pending_replacement_history(
            record,
            InvestigationJobState.from_mapping(state),
            has_recovery_source=(
                legacy_source is not None or incomplete_v2_source is not None
            ),
            label=label,
        )
        if legacy_source is not None or incomplete_v2_source is not None:
            parsed_state = InvestigationJobState.from_mapping(state)
            if (
                checkpoint is not None
                or parsed_state.to_resume_checkpoint(strict=False) is not None
            ):
                raise InvestigationJobStatusError(
                    f"{label} 完整 canonical checkpoint 不得保留迁移恢复来源"
                )
        if legacy_source is not None:
            if "request_digest" not in state:
                raise InvestigationJobStatusError(
                    f"{label}.state legacy 首次写回缺少 request_digest 别名"
                )

    @staticmethod
    def _legacy_replacement_identity(
        state: InvestigationJobState,
    ) -> dict[str, Any]:
        """提取连续作业替换需要逐边保存的完整恢复身份。"""

        return {
            "task_id": state.task_id,
            "batch_id": state.batch_id,
            "combination_id": state.combination_id,
            "scope_id": state.scope_id,
            "codex_thread_id": state.codex_thread_id,
            "turn_id": state.turn_id,
            "submission_request_digest": state.submission_request_digest,
            "job_family": state.job_family,
            "identity": state.identity.as_mapping(),
        }

    @classmethod
    def _validate_legacy_replacement_identity(
        cls,
        value: Any,
        *,
        label: str,
    ) -> dict[str, Any]:
        """校验替换边中的 task、turn、scope 和完整作业身份。"""

        if not isinstance(value, Mapping) or set(value) != set(
            cls._LEGACY_REPLACEMENT_IDENTITY_FIELDS
        ):
            raise InvestigationJobStatusError(f"{label} 字段不完整")
        normalized: dict[str, Any] = {}
        for field_name in (
            "task_id",
            "batch_id",
            "combination_id",
            "scope_id",
            "codex_thread_id",
            "turn_id",
            "job_family",
        ):
            try:
                normalized[field_name] = _require_text(value.get(field_name), field_name)
            except (TypeError, ValueError) as exc:
                raise InvestigationJobStatusError(
                    f"{label}.{field_name} 非法"
                ) from exc
        submission_digest = value.get("submission_request_digest")
        if not isinstance(submission_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", submission_digest
        ) is None:
            raise InvestigationJobStatusError(
                f"{label}.submission_request_digest 非法"
            )
        normalized["submission_request_digest"] = submission_digest
        try:
            identity = InvestigationJobIdentity.from_mapping(value.get("identity"))
        except (TypeError, ValueError) as exc:
            raise InvestigationJobStatusError(
                f"{label}.identity 非法"
            ) from exc
        if identity.job_family != normalized["job_family"]:
            raise InvestigationJobStatusError(
                f"{label}.identity 与 job_family 不一致"
            )
        normalized["identity"] = identity.as_mapping()
        if dict(value) != normalized:
            raise InvestigationJobStatusError(f"{label} 含非规范身份字段")
        return normalized

    @classmethod
    def _validate_pending_replacement_history(
        cls,
        record: Mapping[str, Any],
        state: InvestigationJobState,
        *,
        has_recovery_source: bool,
        label: str,
    ) -> tuple[Mapping[str, Any], ...]:
        """核对显式恢复来源到当前作业的连续 collect/wake 替换记录。"""

        raw_history = record.get("pending_replacement_history")
        if raw_history is None:
            return ()
        if not has_recovery_source:
            raise InvestigationJobStatusError(
                f"{label}.pending_replacement_history 缺少显式恢复来源"
            )
        if not isinstance(raw_history, Sequence) or isinstance(
            raw_history, (str, bytes)
        ) or not raw_history:
            raise InvestigationJobStatusError(
                f"{label}.pending_replacement_history 必须是非空数组"
            )
        normalized_edges: list[Mapping[str, Any]] = []
        previous_successor: Mapping[str, Any] | None = None
        for index, raw_edge in enumerate(raw_history):
            edge_label = f"{label}.pending_replacement_history[{index}]"
            if not isinstance(raw_edge, Mapping) or set(raw_edge) != set(
                cls._LEGACY_REPLACEMENT_EDGE_FIELDS
            ):
                raise InvestigationJobStatusError(f"{edge_label} 字段不完整")
            predecessor = cls._validate_legacy_replacement_identity(
                raw_edge.get("predecessor"),
                label=f"{edge_label}.predecessor",
            )
            successor = cls._validate_legacy_replacement_identity(
                raw_edge.get("successor"),
                label=f"{edge_label}.successor",
            )
            predecessor_auth = raw_edge.get("predecessor_record_auth_digest")
            authorization_kind = raw_edge.get("authorization_kind")
            if not isinstance(predecessor_auth, str) or re.fullmatch(
                r"[0-9a-f]{64}", predecessor_auth
            ) is None:
                raise InvestigationJobStatusError(
                    f"{edge_label}.predecessor_record_auth_digest 非法"
                )
            if authorization_kind not in {"collect", "terminal_wake"}:
                raise InvestigationJobStatusError(
                    f"{edge_label}.authorization_kind 非法"
                )
            stable_fields = (
                "task_id",
                "batch_id",
                "combination_id",
                "scope_id",
                "codex_thread_id",
            )
            if any(
                predecessor[field_name] != successor[field_name]
                for field_name in stable_fields
            ):
                raise InvestigationJobStatusError(
                    f"{edge_label} 跨越了稳定恢复身份"
                )
            if predecessor["identity"] == successor["identity"]:
                raise InvestigationJobStatusError(
                    f"{edge_label} 未替换作业身份"
                )
            if previous_successor is not None and predecessor != previous_successor:
                raise InvestigationJobStatusError(
                    f"{edge_label} 与前一条替换边不连续"
                )
            normalized_edges.append(
                {
                    "predecessor": predecessor,
                    "successor": successor,
                    "predecessor_record_auth_digest": predecessor_auth,
                    "authorization_kind": authorization_kind,
                }
            )
            previous_successor = successor
        if previous_successor != cls._legacy_replacement_identity(state):
            raise InvestigationJobStatusError(
                f"{label}.pending_replacement_history 未到达当前作业"
            )
        return tuple(normalized_edges)

    @classmethod
    def _validate_incomplete_v2_authorization_source(
        cls,
        record: Mapping[str, Any],
        *,
        label: str,
    ) -> tuple[str, str, str] | None:
        """校验显式 incomplete v2 恢复事件及其原 sidecar 摘要。"""

        field_names = cls._INCOMPLETE_V2_SOURCE_FIELDS
        present = {field_name for field_name in field_names if field_name in record}
        if not present:
            return None
        if present != set(field_names):
            raise InvestigationJobStatusError(
                f"{label} incomplete v2 恢复事件与摘要必须同时存在"
            )
        event_ref = record.get("incomplete_v2_resume_event_ref")
        record_digest = record.get("incomplete_v2_source_record_digest")
        record_auth_digest = record.get("incomplete_v2_source_record_auth_digest")
        if not isinstance(event_ref, str) or not event_ref.strip():
            raise InvestigationJobStatusError(
                f"{label}.incomplete_v2_resume_event_ref 非法"
            )
        if (
            not isinstance(record_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", record_digest) is None
            or not isinstance(record_auth_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", record_auth_digest) is None
        ):
            raise InvestigationJobStatusError(
                f"{label} incomplete v2 来源摘要非法"
            )
        return event_ref.strip(), record_digest, record_auth_digest

    @classmethod
    def _validate_legacy_writeback_source(
        cls,
        record: Mapping[str, Any],
        *,
        label: str,
    ) -> tuple[str, str] | None:
        """校验 v2 记录中保留的原 v1 授权来源摘要。

        输入是已经完成字段白名单检查的 record 和错误路径；函数只读取来源版本及
        两个 sealed 摘要，不修改记录。三个字段全部缺失时返回 ``None``；完整且仍
        指向 v1 时返回原 record/auth 摘要，部分缺失、版本错误或摘要非法时抛出
        ``InvestigationJobStatusError``。
        """

        field_names = cls._LEGACY_WRITEBACK_SOURCE_FIELDS
        present = {field_name for field_name in field_names if field_name in record}
        if not present:
            return None
        if present != set(field_names):
            raise InvestigationJobStatusError(
                f"{label} legacy 来源版本与摘要必须同时存在"
            )
        if record.get("legacy_source_version") != cls.legacy_version:
            raise InvestigationJobStatusError(f"{label} legacy 来源版本非法")
        record_digest = record.get("legacy_source_record_digest")
        record_auth_digest = record.get("legacy_source_record_auth_digest")
        if (
            not isinstance(record_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", record_digest) is None
            or not isinstance(record_auth_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", record_auth_digest) is None
        ):
            raise InvestigationJobStatusError(f"{label} legacy 来源摘要非法")
        return record_digest, record_auth_digest

    @classmethod
    def _validate_checkpoint_matches_state(
        cls,
        state_mapping: Mapping[str, Any],
        checkpoint: InvestigationResumeCheckpoint,
        *,
        label: str,
    ) -> None:
        """逐字段核对 state 与分组 checkpoint，阻止一侧串读另一侧身份。

        该校验只比较作业、turn、Working Memory 和 capability 的小型身份字段；不读取
        科学结果、日志或密钥。state 或 checkpoint 任一侧缺字段、值漂移、修复代次不符
        时直接停止恢复。
        """

        state = InvestigationJobState.from_mapping(state_mapping)
        subject = checkpoint.subject
        execution = checkpoint.execution
        provider = checkpoint.provider
        capability = checkpoint.capability
        turn_chain = checkpoint.turn_chain
        comparisons = (
            (state.task_id, subject.task_id, "task_id"),
            (state.batch_id, subject.batch_id, "batch_id"),
            (state.combination_id, subject.combination_id, "combination_id"),
            (state.session_id, subject.session_id, "session_id"),
            (state.stage, subject.stage, "stage"),
            (state.scope_id, subject.scope_id, "scope_id"),
            (state.codex_thread_id, subject.codex_thread_id, "codex_thread_id"),
            (state.logical_operation_instance_id, subject.logical_operation_instance_id, "logical_operation_instance_id"),
            (state.attempt_operation_instance_id, subject.attempt_operation_instance_id, "attempt_operation_instance_id"),
            (state.attempt_number, subject.attempt_number, "attempt_number"),
            (state.repair_request_number, subject.repair_request_number, "repair_request_number"),
            (state.workspace_ref_digest, subject.workspace_ref_digest, "workspace_ref_digest"),
            (state.repair_request_event_ref, subject.repair_request_event_ref, "repair_request_event_ref"),
            (state.expected_result_version, subject.expected_result_version, "expected_result_version"),
            (state.identity.job_id, execution.job_id, "job_id"),
            (state.identity.operation_id, execution.submission_operation_id, "submission_operation_id"),
            (state.identity.submission_key, execution.submission_key, "submission_key"),
            (state.submission_request_digest, execution.submission_request_digest, "submission_request_digest"),
            (state.identity.job_family, execution.job_family, "job_family"),
            (state.identity.action_id, execution.action_id, "action_id"),
            (state.identity.backend_id, execution.backend_id, "backend_id"),
            (state.identity.operation_fingerprint, execution.submission_operation_fingerprint, "submission_operation_fingerprint"),
            (state.codex_thread_id, provider.codex_thread_id, "provider.codex_thread_id"),
            (state.turn_id, provider.codex_turn_id, "provider.codex_turn_id"),
            (state.turn_request_digest, provider.turn_request_digest, "turn_request_digest"),
            (state.thread_policy_digest, provider.thread_policy_digest, "thread_policy_digest"),
            (state.turn_count_before, provider.turn_count_before, "turn_count_before"),
            (state.scope_id, capability.scope_id, "capability.scope_id"),
            (state.workspace_ref_digest, capability.workspace_ref_digest, "capability.workspace_ref_digest"),
            (
                state.scope_credential_generation_digest,
                capability.scope_credential_generation_digest,
                "scope_credential_generation_digest",
            ),
            (state.submit_turn_id, turn_chain.submit_turn_id, "submit_turn_id"),
            (
                state.submit_turn_started_event_ref,
                turn_chain.submit_turn_started_event_ref,
                "submit_turn_started_event_ref",
            ),
        )
        for actual, expected, field_name in comparisons:
            if actual != expected:
                raise InvestigationJobStatusError(
                    f"{label} {field_name} 与 resume_checkpoint 不一致"
                )
        active_turn_id = turn_chain.active_continuation_turn_id
        active_turn_event_ref = turn_chain.active_continuation_turn_started_event_ref
        if active_turn_id is None:
            if state.turn_id != turn_chain.submit_turn_id:
                raise InvestigationJobStatusError(
                    f"{label} 缺少 active continuation turn"
                )
            if state.turn_started_event_ref != turn_chain.submit_turn_started_event_ref:
                raise InvestigationJobStatusError(
                    f"{label} submit Started event 与当前 turn 不一致"
                )
        elif (
            state.turn_id != active_turn_id
            or state.turn_started_event_ref != active_turn_event_ref
        ):
            raise InvestigationJobStatusError(
                f"{label} active continuation turn/event 与 state 不一致"
            )

    @classmethod
    def _validate_state_shape(
        cls,
        state: Mapping[str, Any],
        *,
        label: str,
        legacy: bool,
    ) -> None:
        """校验 state 字段白名单和身份 object，不读取科学 payload。"""

        unknown = set(state) - set(cls._STATE_FIELDS)
        if unknown:
            raise InvestigationJobStatusError(f"{label} 含未知字段")
        identity = state.get("identity")
        if not isinstance(identity, Mapping):
            raise InvestigationJobStatusError(f"{label}.identity 必须是 object")
        if set(identity) - set(cls._IDENTITY_FIELDS):
            raise InvestigationJobStatusError(f"{label}.identity 含未知字段")
        if not legacy and "request_digest" in state and "submission_request_digest" in state:
            if state["request_digest"] != state["submission_request_digest"]:
                raise InvestigationJobStatusError(
                    f"{label} request_digest 与 submission_request_digest 不一致"
                )
        if not legacy:
            submission_digest = state.get("submission_request_digest")
            if not isinstance(submission_digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", submission_digest
            ) is None:
                raise InvestigationJobStatusError(
                    f"{label} submission_request_digest 非法"
                )
            submit_fields = {
                "submit_turn_id",
                "submit_turn_started_event_ref",
            }
            present_submit_fields = submit_fields.intersection(state)
            if present_submit_fields and present_submit_fields != submit_fields:
                raise InvestigationJobStatusError(
                    f"{label} submit turn identity 必须同时存在"
                )

    @classmethod
    def _add_legacy_submission_digest_for_write(
        cls,
        record: Mapping[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        """把已验证旧 state 的请求摘要写入 v2 显式字段。

        输入是 ``_read_document`` 已校验并准备原子写回的一条记录；旧 v1 state 只有
        ``request_digest`` 时，动作要求它是 64 位小写 SHA-256，并将同一个值复制到
        ``submission_request_digest``。返回独立浅副本，不修改调用方对象，也不补造
        provider turn、checkpoint 或作业身份；旧摘要缺失或格式非法时抛出状态错误。
        """

        result = dict(record)
        state = result.get("state")
        if not isinstance(state, Mapping):
            return result
        if "submission_request_digest" in state:
            return result
        legacy_digest = state.get("request_digest")
        if not isinstance(legacy_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", legacy_digest
        ) is None:
            raise InvestigationJobStatusError(
                f"{label}.state 旧 request_digest 无法写成 submission_request_digest"
            )
        upgraded_state = dict(state)
        upgraded_state["submission_request_digest"] = legacy_digest
        result["state"] = upgraded_state
        source_record_digest = result.get("record_digest")
        source_record_auth_digest = result.get("record_auth_digest")
        if (
            not isinstance(source_record_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_record_digest) is None
            or not isinstance(source_record_auth_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_record_auth_digest) is None
        ):
            raise InvestigationJobStatusError(
                f"{label} 旧记录缺少可核对的 sealed 来源摘要"
            )
        result["legacy_source_version"] = cls.legacy_version
        result["legacy_source_record_digest"] = source_record_digest
        result["legacy_source_record_auth_digest"] = source_record_auth_digest
        return result

    def _write_document(self, document: Mapping[str, Any]) -> None:
        """原子写入等待快照并确保 replace 前内容落盘。"""

        self.root.mkdir(parents=True, exist_ok=True)
        records = document.get("records")
        if not isinstance(records, Mapping):
            raise InvestigationJobStatusError("waiting job 写入 records 必须是 object")
        sealed_records: dict[str, Any] = {}
        for key, raw_record in records.items():
            if not isinstance(key, str) or not isinstance(raw_record, Mapping):
                raise InvestigationJobStatusError("waiting job 写入记录 key/value 非法")
            record = self._add_legacy_submission_digest_for_write(
                raw_record,
                label=f"record[{key}]",
            )
            record.pop("record_digest", None)
            record.pop("record_auth_digest", None)
            self._validate_v2_record(record, label=f"record[{key}]")
            state_mapping = record.get("state")
            if not isinstance(state_mapping, Mapping):  # pragma: no cover - validator 已检查
                raise InvestigationJobStatusError("waiting job 写入 state 必须是 object")
            parsed_state = InvestigationJobState.from_mapping(state_mapping)
            if key != self._key(parsed_state):
                raise InvestigationJobStatusError(
                    "waiting job 写入 key 与完整身份不一致"
                )
            record["record_digest"] = compute_resume_record_digest(record)
            self._validate_decision_fields(record, label=f"record[{key}]")
            record["record_auth_digest"] = compute_resume_record_auth_digest(record)
            sealed_records[key] = record
        sealed_document = {
            "schema": self.schema,
            "version": self.version,
            "records": sealed_records,
        }
        payload = json.dumps(sealed_document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @staticmethod
    def _record_mapping(state: InvestigationJobState, **fields: Any) -> dict[str, Any]:
        """构造一条只含可恢复事实的记录。"""

        result: dict[str, Any] = {"state": state.as_mapping()}
        checkpoint = state.to_resume_checkpoint(strict=False)
        if checkpoint is not None:
            result["resume_checkpoint"] = checkpoint.as_mapping()
        result.update(copy.deepcopy(fields))
        return result

    def list_records(self) -> tuple[Mapping[str, Any], ...]:
        """返回当前所有严格校验的等待记录。"""

        with self._storage_lock():
            document = self._read_document()
            result: list[Mapping[str, Any]] = []
            for key, record in document["records"].items():
                if not isinstance(key, str) or not isinstance(record, Mapping):
                    raise InvestigationJobStatusError("waiting job 记录 key/value 非法")
                state = InvestigationJobState.from_mapping(record.get("state", record))
                if self._key(state) != key:
                    raise InvestigationJobStatusError("waiting job 记录 key 与身份不一致")
                result.append(copy.deepcopy(dict(record)))
            return tuple(result)

    def get_record_source_version(
        self,
        task_id: str,
        batch_id: str,
        combination_id: str,
    ) -> int | None:
        """读取指定 waiting 记录可使用的恢复授权来源版本。

        输入是完整 task/batch/combination 身份；动作先完整校验快照，再返回该记录
        的 v1/v2 授权来源。v1 首次受控写回后的 v2 记录在完整 checkpoint 落盘
        前仍返回 v1；这样重启可以复用同一显式 legacy 授权，不能把只补摘要的记录
        误报为 canonical v2。记录不存在时返回 ``None``；文档、身份或摘要损坏时
        抛出状态错误，不修改 sidecar。
        """

        normalized_task_id = _require_text(task_id, "task_id")
        normalized_batch_id = _require_text(batch_id, "batch_id")
        normalized_combination_id = _require_text(combination_id, "combination_id")
        with self._storage_lock():
            if not self.path.exists():
                return None
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise InvestigationJobStatusError("waiting job 快照无法读取") from exc
            if not isinstance(raw, Mapping):
                raise InvestigationJobStatusError("waiting job 快照 schema/version 不匹配")
            source_version = raw.get("version")
            # _read_document 负责所有字段、摘要和重复身份校验；这里不重复实现
            # 那套合同，只保留版本这个不会出现在兼容升级结果中的元数据。
            document = self._read_document()
            for record in document["records"].values():
                if not isinstance(record, Mapping):
                    continue
                state = InvestigationJobState.from_mapping(
                    record.get("state", record)
                )
                if (
                    state.task_id == normalized_task_id
                    and state.batch_id == normalized_batch_id
                    and state.combination_id == normalized_combination_id
                ):
                    if (
                        source_version == self.version
                        and record.get("legacy_source_version")
                        == self.legacy_version
                    ):
                        return self.legacy_version
                    return source_version
            return None

    def get(self, task_id: str, batch_id: str, combination_id: str) -> Mapping[str, Any] | None:
        """读取一个 task/batch/combination 的严格等待记录。"""

        normalized_task_id = _require_text(task_id, "task_id")
        normalized_batch_id = _require_text(batch_id, "batch_id")
        normalized_combination_id = _require_text(combination_id, "combination_id")
        key = self._key_for_values(
            normalized_task_id,
            normalized_batch_id,
            normalized_combination_id,
        )
        with self._storage_lock():
            document = self._read_document()
            record = document["records"].get(key)
            if record is None:
                return None
            state = InvestigationJobState.from_mapping(record.get("state", record))
            if self._key(state) != key:
                raise InvestigationJobStatusError("waiting job 记录 key 与身份不一致")
            return copy.deepcopy(dict(record))

    def save_waiting(self, state: InvestigationJobState) -> Mapping[str, Any]:
        """幂等保存等待状态；不同 job 或 request 指纹绝不覆盖旧记录。"""

        if state.status != "waiting_investigation_job":
            raise ValueError("只能保存 waiting_investigation_job")
        with self._storage_lock():
            document = self._read_document()
            key = self._key(state)
            existing = document["records"].get(key)
            record = self._record_mapping(state, wake_status="pending")
            if existing is not None:
                old = InvestigationJobState.from_mapping(existing.get("state", existing))
                if old.as_mapping() != state.as_mapping():
                    raise InvestigationJobStatusError("等待记录已存在但 job identity/request 指纹不同")
                return copy.deepcopy(dict(existing))
            document["records"][key] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(key)
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("等待记录写入后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    def record_incomplete_v2_resume_authorization(
        self,
        state: InvestigationJobState,
        event: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """把显式恢复事件绑定到尚无完整 checkpoint 的 v2 等待记录。

        输入是当前等待状态和 Working Memory 已持久化并重验的恢复事件；动作要求事件
        精确绑定写入前的 record/auth 摘要及当前作业身份，再把事件引用和原摘要写入
        sidecar。返回重新密封后的记录；重复调用会重验并返回同一记录，身份或摘要变化
        时抛出状态错误。该方法不查询、提交或收集计算作业。
        """

        if not isinstance(state, InvestigationJobState):
            raise TypeError("incomplete v2 恢复授权 state 类型非法")
        if not isinstance(event, Mapping):
            raise TypeError("incomplete v2 恢复授权 event 类型非法")
        if state.to_resume_checkpoint(strict=False) is not None:
            raise InvestigationJobStatusError(
                "完整 canonical waiting 不需要 incomplete v2 恢复授权"
            )
        event_ref = event.get("event_id")
        source_digest = event.get("sidecar_record_digest")
        source_auth_digest = event.get("sidecar_record_auth_digest")
        if not isinstance(event_ref, str) or not event_ref.strip():
            raise InvestigationJobStatusError("incomplete v2 恢复事件缺少 event ID")
        if (
            not isinstance(source_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
            or not isinstance(source_auth_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_auth_digest) is None
        ):
            raise InvestigationJobStatusError("incomplete v2 恢复事件来源摘要非法")
        expected_identity = {
            "task_id": state.task_id,
            "batch_id": state.batch_id,
            "combination_id": state.combination_id,
            "scope_id": state.scope_id,
            "codex_thread_id": state.codex_thread_id,
            "current_turn_id": state.turn_id,
            "job_id": state.identity.job_id,
            "submission_operation_id": state.identity.operation_id,
            "submission_key": state.identity.submission_key,
            "submission_request_digest": state.submission_request_digest,
            "job_family": state.job_family,
        }
        for field_name, expected_value in expected_identity.items():
            if event.get(field_name) != expected_value:
                raise InvestigationJobStatusError(
                    f"incomplete v2 恢复事件 {field_name} 与等待作业不一致"
                )
        for field_name in (
            "action_id",
            "backend_id",
            "job_type",
            "runtime",
            "software_id",
        ):
            if event.get(field_name) != getattr(state.identity, field_name):
                raise InvestigationJobStatusError(
                    f"incomplete v2 恢复事件 {field_name} 与等待作业不一致"
                )
        state_operation_fingerprint = state.identity.operation_fingerprint
        event_operation_fingerprint = event.get("operation_fingerprint")
        if state_operation_fingerprint is not None:
            if event_operation_fingerprint != state_operation_fingerprint:
                raise InvestigationJobStatusError(
                    "incomplete v2 恢复事件 operation_fingerprint 与等待作业不一致"
                )
        elif not isinstance(event_operation_fingerprint, str) or re.fullmatch(
            r"[0-9a-f]{64}", event_operation_fingerprint
        ) is None:
            raise InvestigationJobStatusError(
                "incomplete v2 恢复事件缺少已核验的 operation_fingerprint"
            )
        with self._storage_lock():
            document = self._read_document()
            key = self._key(state)
            existing = document["records"].get(key)
            if existing is None:
                raise InvestigationJobStatusError(
                    "incomplete v2 恢复授权缺少等待记录"
                )
            stored = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored.as_mapping() != state.as_mapping():
                raise InvestigationJobStatusError(
                    "incomplete v2 恢复授权等待身份漂移"
                )
            persisted_source = self._validate_incomplete_v2_authorization_source(
                existing,
                label="incomplete v2 waiting record",
            )
            expected_source = (event_ref.strip(), source_digest, source_auth_digest)
            if persisted_source is not None:
                if persisted_source != expected_source:
                    raise InvestigationJobStatusError(
                        "incomplete v2 恢复授权来源发生变化"
                    )
                return copy.deepcopy(dict(existing))
            if (
                existing.get("record_digest") != source_digest
                or existing.get("record_auth_digest") != source_auth_digest
            ):
                raise InvestigationJobStatusError(
                    "incomplete v2 恢复事件未绑定当前 sealed waiting 记录"
                )
            record = dict(existing)
            record.update(
                {
                    "incomplete_v2_resume_event_ref": event_ref.strip(),
                    "incomplete_v2_source_record_digest": source_digest,
                    "incomplete_v2_source_record_auth_digest": source_auth_digest,
                }
            )
            document["records"][key] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(key)
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError(
                    "incomplete v2 恢复授权写入后无法读取 sealed record"
                )
            return copy.deepcopy(dict(sealed))

    def update_waiting_contract(
        self,
        previous: InvestigationJobState,
        next_state: InvestigationJobState,
    ) -> Mapping[str, Any]:
        """补齐同一已接受作业的 provider receipt 合同并返回落盘记录。

        ``previous`` 是 submit callback 先保存的最小等待身份，``next_state`` 必须复用
        同一 task/batch/combination、job、operation、submission、thread 和 turn。动作
        只允许把真实完成 turn receipt 提供的摘要、事件和分组 checkpoint 补入 sidecar；
        不允许借此替换作业或跨组合写入。返回值包含本次原子写入实际生成的
        ``record_digest``，供 Working Memory waiting 事实逐值引用。
        """

        if not isinstance(previous, InvestigationJobState) or not isinstance(
            next_state, InvestigationJobState
        ):
            raise TypeError("waiting contract update 需要 InvestigationJobState")
        if self._key(previous) != self._key(next_state):
            raise InvestigationJobStatusError("waiting contract update 不能跨组合")
        if previous.identity.as_mapping() != next_state.identity.as_mapping():
            raise InvestigationJobStatusError(
                "waiting contract update 不能替换 job/operation/submission identity"
            )
        if previous.codex_thread_id != next_state.codex_thread_id or previous.turn_id != next_state.turn_id:
            raise InvestigationJobStatusError(
                "waiting contract update 不能替换 submit thread/turn"
            )
        with self._storage_lock():
            document = self._read_document()
            key = self._key(previous)
            existing = document["records"].get(key)
            if existing is None:
                raise InvestigationJobStatusError("waiting contract update 缺少原记录")
            stored = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored.as_mapping() != previous.as_mapping():
                raise InvestigationJobStatusError("waiting contract update 原身份已漂移")
            fields = {
                field_name: copy.deepcopy(existing[field_name])
                for field_name in (
                    "wake_status",
                    "terminal_status",
                    "terminal_message",
                    "wake_request",
                    "wake_request_digest",
                    "wake_receipt",
                    "source_wake_state",
                    "source_wake_receipt",
                )
                if field_name in existing
            }
            document["records"][key] = self._record_mapping(next_state, **fields)
            self._write_document(document)
            record = self._read_document()["records"][key]
            return copy.deepcopy(record)

    def roll_forward_waiting_continuation_contract(
        self,
        previous: InvestigationJobState,
        next_state: InvestigationJobState,
        *,
        allow_provider_migration: bool = False,
        migration_source_thread_id: str | None = None,
        migration_current_source_thread_id: str | None = None,
        migration_verified_thread_ids: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        """把同一物理作业的等待接力原子前移到下一 continuation turn。

        输入是当前 sidecar 状态和新 turn 完成收据补齐后的状态；动作保持作业、提交、
        scope、session、stage 与科学版本身份不变，只允许 turn、attempt 和 provider
        checkpoint 前移。默认禁止 thread 改变；若调用方确实完成了 provider 迁移，必须
        同时提供显式开关、当前 source thread 和不含重复项的已核验 thread 链，存储层不
        会自行推断这些许可。副作用是替换当前组合的单条 sidecar；若旧记录已有 wake
        handoff、作业身份变化、稳定字段漂移或 attempt/count 未按一格前进，则抛出状态错误。
        """

        if not isinstance(previous, InvestigationJobState) or not isinstance(
            next_state, InvestigationJobState
        ):
            raise TypeError("waiting continuation roll-forward 需要 InvestigationJobState")
        if self._key(previous) != self._key(next_state):
            raise InvestigationJobStatusError(
                "waiting continuation roll-forward 不能跨组合"
            )
        if previous.identity.as_mapping() != next_state.identity.as_mapping():
            raise InvestigationJobStatusError(
                "waiting continuation roll-forward 不能替换作业身份"
            )
        if previous.turn_id == next_state.turn_id:
            raise InvestigationJobStatusError(
                "waiting continuation roll-forward 不能复用同一 provider turn"
            )
        if not isinstance(allow_provider_migration, bool):
            raise InvestigationJobStatusError(
                "waiting continuation thread migration 许可必须是 bool"
            )
        if previous.codex_thread_id != next_state.codex_thread_id:
            if not allow_provider_migration:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 默认不能更换 provider thread"
                )
            normalized_migration_source = (
                _require_text(migration_source_thread_id, "migration_source_thread_id")
                if migration_source_thread_id is not None
                else None
            )
            normalized_current_source = (
                _require_text(
                    migration_current_source_thread_id,
                    "migration_current_source_thread_id",
                )
                if migration_current_source_thread_id is not None
                else None
            )
            source_thread_id = normalized_current_source or normalized_migration_source
            if not isinstance(source_thread_id, str) or not source_thread_id.strip():
                raise InvestigationJobStatusError(
                    "waiting continuation thread migration 缺少当前 source thread"
                )
            source_thread_id = source_thread_id.strip()
            if source_thread_id != previous.codex_thread_id:
                raise InvestigationJobStatusError(
                    "waiting continuation thread migration source 与旧 waiting thread 不一致"
                )
            if not isinstance(migration_verified_thread_ids, Sequence) or isinstance(
                migration_verified_thread_ids, (str, bytes)
            ):
                raise InvestigationJobStatusError(
                    "waiting continuation thread migration chain 非法"
                )
            verified_chain = tuple(migration_verified_thread_ids)
            if len(verified_chain) < 2 or any(
                not isinstance(thread_id, str) or not thread_id.strip()
                for thread_id in verified_chain
            ):
                raise InvestigationJobStatusError(
                    "waiting continuation thread migration chain 不完整"
                )
            verified_chain = tuple(thread_id.strip() for thread_id in verified_chain)
            if len(set(verified_chain)) != len(verified_chain):
                raise InvestigationJobStatusError(
                    "waiting continuation thread migration chain 含重复 thread"
                )
            if (
                normalized_migration_source is not None
                and verified_chain[0] != normalized_migration_source
            ):
                raise InvestigationJobStatusError(
                    "waiting continuation thread migration chain 与 source lineage 不一致"
                )
            if verified_chain[-2:] != (
                source_thread_id,
                next_state.codex_thread_id,
            ):
                raise InvestigationJobStatusError(
                    "waiting continuation thread migration chain 未证明当前 source→target"
                )
        elif (
            allow_provider_migration
            or migration_source_thread_id is not None
            or migration_current_source_thread_id is not None
            or migration_verified_thread_ids
        ):
            raise InvestigationJobStatusError(
                "未发生 thread migration 时不能携带 migration 许可或证据"
            )
        canonical_fields = (
            "session_id",
            "stage",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "attempt_number",
            "turn_request_digest",
            "thread_policy_digest",
            "turn_count_before",
            "workspace_ref_digest",
            "scope_credential_generation_digest",
            "submit_turn_id",
            "submit_turn_started_event_ref",
            "turn_started_event_ref",
        )
        previous_canonical = tuple(
            getattr(previous, field_name) for field_name in canonical_fields
        )
        next_canonical = tuple(
            getattr(next_state, field_name) for field_name in canonical_fields
        )
        previous_has_canonical = any(value is not None for value in previous_canonical)
        next_has_canonical = any(value is not None for value in next_canonical)
        upgrades_pending_v2 = not previous_has_canonical and next_has_canonical
        if previous_has_canonical and not next_has_canonical:
            raise InvestigationJobStatusError(
                "waiting continuation roll-forward 不能丢失 canonical identity"
            )
        if next_has_canonical and any(value is None for value in next_canonical):
            raise InvestigationJobStatusError(
                "waiting continuation roll-forward canonical identity 不完整"
            )
        if upgrades_pending_v2:
            # submit callback 会先保存不含 completed receipt 的最小 v2 state。进程
            # 若在 turn 返回前退出，下一次 continuation 只能用真实 Started 事实把
            # 原 submit turn 补回；不得把新的 continuation turn 冒充物理提交起点。
            if (
                next_state.submit_turn_id != previous.turn_id
                or not isinstance(next_state.submit_turn_started_event_ref, str)
                or not next_state.submit_turn_started_event_ref
                or next_state.turn_started_event_ref
                == next_state.submit_turn_started_event_ref
            ):
                raise InvestigationJobStatusError(
                    "pending v2 roll-forward 缺少原 submit Started identity"
                )
        elif previous_has_canonical:
            if any(value is None for value in previous_canonical + next_canonical):
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward canonical identity 不完整"
                )
            if next_state.submit_turn_id != previous.submit_turn_id:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 不能更换原始 submit turn"
                )
            if (
                next_state.submit_turn_started_event_ref
                != previous.submit_turn_started_event_ref
            ):
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 不能更换原始 submit Started event"
                )
            if next_state.attempt_number != previous.attempt_number + 1:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 的 attempt_number 必须精确前进一格"
                )
            if next_state.turn_count_before != previous.turn_count_before + 1:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 的 turn_count_before 必须精确前进一格"
                )
            if next_state.attempt_operation_instance_id == previous.attempt_operation_instance_id:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 不能复用 attempt_operation_instance_id"
                )
            if next_state.turn_request_digest == previous.turn_request_digest:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 不能复用 turn_request_digest"
                )
            if next_state.turn_started_event_ref == previous.turn_started_event_ref:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 不能复用 Started event"
                )
        stable_fields = (
            "submission_request_digest",
            "scope_id",
            "job_family",
            "submit_tool",
            "status_tool",
            "collect_tool",
            "session_id",
            "stage",
            "logical_operation_instance_id",
            "repair_request_number",
            "workspace_ref_digest",
            "repair_request_event_ref",
            "expected_result_version",
            "submit_turn_id",
            "submit_turn_started_event_ref",
            "scope_credential_generation_digest",
            "thread_policy_digest",
        )
        for field_name in stable_fields:
            old_value = getattr(previous, field_name)
            new_value = getattr(next_state, field_name)
            if old_value is not None and old_value != new_value:
                raise InvestigationJobStatusError(
                    f"waiting continuation roll-forward 的 {field_name} 漂移"
                )
        with self._storage_lock():
            document = self._read_document()
            key = self._key(previous)
            existing = document["records"].get(key)
            if existing is None:
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 缺少原记录"
                )
            stored = InvestigationJobState.from_mapping(
                existing.get("state", existing)
            )
            if stored.as_mapping() != previous.as_mapping():
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 原身份已漂移"
                )
            if existing.get("wake_status", "pending") != "pending" or any(
                field_name in existing
                for field_name in (
                    "terminal_status",
                    "terminal_message",
                    "wake_request",
                    "wake_request_digest",
                    "wake_receipt",
                    "source_wake_state",
                    "source_wake_receipt",
                )
            ):
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 不能覆盖 wake handoff"
                )
            document["records"][key] = self._record_mapping(
                next_state,
                wake_status="pending",
            )
            self._write_document(document)
            record = self._read_document()["records"].get(key)
            if not isinstance(record, Mapping):
                raise InvestigationJobStatusError(
                    "waiting continuation roll-forward 后无法读取 sealed record"
                )
            return copy.deepcopy(dict(record))

    def arm_terminal(
        self,
        state: InvestigationJobState,
        *,
        terminal_status: str,
        terminal_message: str | None = None,
    ) -> Mapping[str, Any]:
        """持久化终态和 wake intent 的前置事实，不触发模型。"""

        status = _normalize_job_status(terminal_status)
        if status not in TERMINAL_INVESTIGATION_JOB_STATUSES:
            raise InvestigationJobStatusError("只能为终态作业 arm wake")
        with self._storage_lock():
            existing = self.get(state.task_id, state.batch_id, state.combination_id)
            if existing is None:
                self.save_waiting(state)
                existing = self.get(state.task_id, state.batch_id, state.combination_id)
            if existing is None:  # pragma: no cover - save/get 契约保证
                raise InvestigationJobStatusError("等待记录写入后无法读取")
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != state.as_mapping():
                raise InvestigationJobStatusError("终态 arm 的等待身份不一致")
            record = dict(existing)
            record["terminal_status"] = status
            if terminal_message is not None:
                record["terminal_message"] = _require_text(terminal_message, "terminal_message")
            record["wake_status"] = record.get("wake_status", "pending")
            document = self._read_document()
            document["records"][self._key(state)] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(self._key(state))
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("终态 arm 后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    def record_wake_intent(
        self,
        state: InvestigationJobState,
        wake_request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """保存待发送 wake request；限流失败时保留 intent 以便只重试唤醒。"""

        with self._storage_lock():
            existing = self.get(state.task_id, state.batch_id, state.combination_id)
            if existing is None:
                raise InvestigationJobStatusError("wake intent 缺少已保存等待状态")
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != state.as_mapping():
                raise InvestigationJobStatusError("wake intent identity 不一致")
            request_copy = _copy_json_object(wake_request, "wake_request")
            digest = _request_digest(request_copy)
            old_digest = existing.get("wake_request_digest")
            if old_digest is not None and old_digest != digest:
                raise InvestigationJobStatusError("同一等待作业的 wake request 指纹漂移")
            record = dict(existing)
            # 当前 state 开始的是新一轮 wake；上一轮已完成 handoff 的 source
            # provenance 不能串到本轮 secondary/tertiary source 上。
            for field_name in ("wake_receipt", "source_wake_state", "source_wake_receipt"):
                record.pop(field_name, None)
            record.update({"wake_status": "intent", "wake_request": request_copy, "wake_request_digest": digest})
            document = self._read_document()
            document["records"][self._key(state)] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(self._key(state))
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("wake intent 写入后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    def record_wake_receipt(
        self,
        state: InvestigationJobState,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """保存已完成 wake 的 receipt，但在 Result 成功前保持 intent。

        receipt 只证明同一 Codex thread 的 wake turn 已返回；它不证明科学 Result
        已经写入 Working Memory。因此这里不能把记录标成 ``delivered``。恢复时可用
        receipt 与 Working Memory Result 的 turn identity 对账，成功后再显式调用
        ``mark_wake_delivered``；如果 Result 尚未落盘，Controller 仍复用原 request。
        """

        with self._storage_lock():
            existing = self.get(state.task_id, state.batch_id, state.combination_id)
            if existing is None:
                raise InvestigationJobStatusError("wake receipt 缺少已保存等待状态")
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != state.as_mapping():
                raise InvestigationJobStatusError("wake receipt identity 不一致")
            receipt_copy = _copy_json_object(receipt, "wake_receipt")
            previous_receipt = existing.get("wake_receipt")
            if previous_receipt is not None and previous_receipt != receipt_copy:
                raise InvestigationJobStatusError("同一 wake 的 receipt identity 发生漂移")
            record = dict(existing)
            record.update({"wake_status": "intent", "wake_receipt": receipt_copy})
            document = self._read_document()
            document["records"][self._key(state)] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(self._key(state))
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("wake receipt 写入后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    def mark_wake_delivered(
        self,
        state: InvestigationJobState,
        receipt: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """在科学 Result 成功写入后把 wake intent 转为 delivered。

        输入 state 必须与 sidecar 完整一致；若调用方提供 receipt，它还必须与已保存
        的同一 wake receipt 完全相同。该方法只更新等待事实，不删除记录，调用方随后
        再执行 ``clear_after_result``；中间任一步失败都能由恢复逻辑继续收束。
        """

        with self._storage_lock():
            existing = self.get(state.task_id, state.batch_id, state.combination_id)
            if existing is None:
                raise InvestigationJobStatusError("wake delivered 缺少已保存等待状态")
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != state.as_mapping():
                raise InvestigationJobStatusError("wake delivered identity 不一致")
            stored_receipt = existing.get("wake_receipt")
            if stored_receipt is None:
                # wake turn 内顺序提交下一作业时，当前 state 已经是 replacement；
                # 该 sidecar 保存的是绑定原作业的 source_wake_receipt。它与普通
                # wake_receipt 共享同一份 thread/turn 收束校验，不能被新 job 身份
                # 替换或当作另一轮 wake。
                stored_receipt = existing.get("source_wake_receipt")
            if not isinstance(stored_receipt, Mapping):
                raise InvestigationJobStatusError("wake delivered 缺少已保存 wake receipt")
            if receipt is not None and stored_receipt != _copy_json_object(receipt, "wake_receipt"):
                raise InvestigationJobStatusError("wake delivered receipt identity 发生漂移")
            record = dict(existing)
            record["wake_status"] = "delivered"
            document = self._read_document()
            document["records"][self._key(state)] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(self._key(state))
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("wake delivered 写入后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    def clear_after_result(self, state: InvestigationJobState) -> None:
        """在最终科学结果已写入 WM 后清除独立等待记录。"""

        with self._storage_lock():
            document = self._read_document()
            key = self._key(state)
            existing = document["records"].get(key)
            if existing is None:
                return
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != state.as_mapping():
                raise InvestigationJobStatusError("清理 waiting 时身份不一致")
            document["records"].pop(key, None)
            self._write_document(document)

    def record_collected_for_replacement(
        self,
        state: InvestigationJobState,
        receipt: InvestigationJobReceipt,
    ) -> Mapping[str, Any]:
        """保存旧作业已由真实 collect 回执完成的最小替换许可。

        输入必须是当前 sidecar state 和同一作业的真实 collect 工具回执；动作只保存
        job/operation/submission/thread/turn 身份，不保存计算结果。该许可只允许下一次
        ``replace_waiting`` 消费；缺少或漂移时旧作业记录保持不变并拒绝覆盖。
        """

        if not isinstance(state, InvestigationJobState):
            raise TypeError("collected replacement state 必须是 InvestigationJobState")
        if not isinstance(receipt, InvestigationJobReceipt):
            raise TypeError("collected replacement receipt 必须是 InvestigationJobReceipt")
        if receipt.identity.as_mapping() != state.identity.as_mapping():
            raise InvestigationJobStatusError(
                "collected replacement receipt 与 waiting 作业身份不一致"
            )
        with self._storage_lock():
            document = self._read_document()
            key = self._key(state)
            existing = document["records"].get(key)
            if existing is None:
                raise InvestigationJobStatusError(
                    "collected replacement 缺少 waiting 记录"
                )
            stored_state = InvestigationJobState.from_mapping(
                existing.get("state", existing)
            )
            if stored_state.as_mapping() != state.as_mapping():
                raise InvestigationJobStatusError(
                    "collected replacement waiting 身份漂移"
                )
            record = dict(existing)
            record["collected_source"] = {
                "job_id": state.identity.job_id,
                "operation_id": state.identity.operation_id,
                "submission_key": state.identity.submission_key,
                "thread_id": state.codex_thread_id,
                "turn_id": state.turn_id,
            }
            document["records"][key] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(key)
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError(
                    "collected replacement 许可写入后无法读取"
                )
            return copy.deepcopy(dict(sealed))

    def replace_waiting(
        self,
        previous: InvestigationJobState,
        next_state: InvestigationJobState,
    ) -> Mapping[str, Any]:
        """原子替换同一组合的旧 job 与下一 job，防止顺序链中断后重复提交。"""

        if previous.task_id != next_state.task_id or previous.batch_id != next_state.batch_id or previous.combination_id != next_state.combination_id:
            raise InvestigationJobStatusError("waiting job 替换不能跨 task/batch/combination")
        with self._storage_lock():
            document = self._read_document()
            old_key = self._key(previous)
            new_key = self._key(next_state)
            existing = document["records"].get(old_key)
            if existing is None:
                raise InvestigationJobStatusError("waiting job 替换缺少旧记录")
            old_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if old_state.as_mapping() == next_state.as_mapping():
                return copy.deepcopy(dict(existing))
            if old_state.as_mapping() != previous.as_mapping():
                raise InvestigationJobStatusError("waiting job 替换旧身份不一致")
            old_submission_identity = (
                old_state.identity.job_id,
                old_state.identity.operation_id,
                old_state.identity.submission_key,
            )
            next_submission_identity = (
                next_state.identity.job_id,
                next_state.identity.operation_id,
                next_state.identity.submission_key,
            )
            if old_submission_identity == next_submission_identity:
                raise InvestigationJobStatusError(
                    "wake 后的新 waiting job 不能复用旧 job/operation/submission identity"
                )
            wake_replacement_authorized = (
                isinstance(existing.get("terminal_status"), str)
                and isinstance(existing.get("wake_request"), Mapping)
                and isinstance(existing.get("wake_request_digest"), str)
            )
            collected_replacement_authorized = isinstance(
                existing.get("collected_source"),
                Mapping,
            )
            if not wake_replacement_authorized and not collected_replacement_authorized:
                raise InvestigationJobStatusError(
                    "waiting replacement 缺少已核对的 terminal wake 或 collect 来源"
                )
            replacement_fields: dict[str, Any] = {"wake_status": "pending"}
            # wake turn 可能已经把完整 request 送入 provider，随后才在同一 turn
            # 接受下一作业。replacement 必须继续携带这份 request 和来源审计；否则
            # 进程在 turn 完成收据落盘后崩溃时，恢复只能把 secondary 当成新 wake
            # 源，既可能重复唤醒，也无法核对原 job identity。
            for field_name in (
                "terminal_status",
                "terminal_message",
                "wake_request",
                "wake_request_digest",
                "wake_receipt",
                "source_wake_state",
                "source_wake_receipt",
            ):
                if field_name in existing:
                    replacement_fields[field_name] = copy.deepcopy(existing[field_name])
            legacy_source_present = all(
                field_name in existing
                for field_name in self._LEGACY_WRITEBACK_SOURCE_FIELDS
            )
            incomplete_v2_source_present = all(
                field_name in existing
                for field_name in self._INCOMPLETE_V2_SOURCE_FIELDS
            )
            if (
                (legacy_source_present or incomplete_v2_source_present)
                and next_state.to_resume_checkpoint(strict=False) is None
            ):
                # 显式恢复事件绑定的是迁移前 sidecar。一个 provider turn 可以连续
                # collect→submit 多个作业；完整 turn receipt 落盘前必须保留该来源，
                # 并逐项记录旧作业如何被真实 collect 或 terminal wake 许可替换。
                source_fields = (
                    self._LEGACY_WRITEBACK_SOURCE_FIELDS
                    if legacy_source_present
                    else self._INCOMPLETE_V2_SOURCE_FIELDS
                )
                for field_name in source_fields:
                    replacement_fields[field_name] = copy.deepcopy(
                        existing[field_name]
                    )
                history = list(
                    copy.deepcopy(existing.get("pending_replacement_history", ()))
                )
                predecessor_auth_digest = existing.get("record_auth_digest")
                if not isinstance(predecessor_auth_digest, str) or re.fullmatch(
                    r"[0-9a-f]{64}", predecessor_auth_digest
                ) is None:
                    raise InvestigationJobStatusError(
                        "waiting replacement 缺少旧作业 sealed auth 摘要"
                    )
                history.append(
                    {
                        "predecessor": self._legacy_replacement_identity(old_state),
                        "successor": self._legacy_replacement_identity(next_state),
                        "predecessor_record_auth_digest": predecessor_auth_digest,
                        "authorization_kind": (
                            "collect"
                            if collected_replacement_authorized
                            else "terminal_wake"
                        ),
                    }
                )
                replacement_fields["pending_replacement_history"] = history
            document["records"][new_key] = self._record_mapping(
                next_state,
                **replacement_fields,
            )
            self._write_document(document)
            sealed = self._read_document()["records"].get(new_key)
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("waiting replacement 后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    def record_wake_source_after_replacement(
        self,
        previous: InvestigationJobState,
        next_state: InvestigationJobState,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """在顺序 submit 后保存仍在执行的原 wake 来源。

        输入是 callback 已经核对的旧/新 waiting state 和当前 wake turn 的
        thread/turn/job 收据；动作把旧 state、原 job 收据和 wake request 固定到
        replacement sidecar，并标记为 ``in_flight``。副作用只写等待快照，不发送
        模型或作业请求；身份不一致时抛出状态错误，调用方必须停止恢复。
        """

        if previous.task_id != next_state.task_id or previous.batch_id != next_state.batch_id or previous.combination_id != next_state.combination_id:
            raise InvestigationJobStatusError("wake source replacement 不能跨 task/batch/combination")
        with self._storage_lock():
            existing = self.get(
                next_state.task_id,
                next_state.batch_id,
                next_state.combination_id,
            )
            if existing is None:
                raise InvestigationJobStatusError("wake source replacement 缺少下一等待记录")
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != next_state.as_mapping():
                raise InvestigationJobStatusError("wake source replacement 的下一身份不一致")
            source_mapping = existing.get("source_wake_state")
            if source_mapping is None:
                source_mapping = previous.as_mapping()
            source_state = InvestigationJobState.from_mapping(source_mapping)
            if (
                source_state.task_id != previous.task_id
                or source_state.batch_id != previous.batch_id
                or source_state.combination_id != previous.combination_id
            ):
                raise InvestigationJobStatusError("wake source replacement 的来源组合不一致")
            receipt_copy = _copy_json_object(receipt, "wake_receipt")
            if receipt_copy.get("job_id") != source_state.identity.job_id:
                raise InvestigationJobStatusError("wake source receipt 未绑定原 job")
            if receipt_copy.get("thread_id") != next_state.codex_thread_id:
                raise InvestigationJobStatusError("wake source receipt thread 与原 job 不一致")
            if receipt_copy.get("turn_id") != next_state.turn_id:
                raise InvestigationJobStatusError(
                    "wake source receipt turn 与 replacement turn 不一致"
                )
            previous_receipt = existing.get("source_wake_receipt")
            settled_receipt = self._merge_source_wake_receipts(
                previous_receipt,
                receipt_copy,
            )
            record = dict(existing)
            record.update(
                {
                    "wake_status": "in_flight",
                    "source_wake_state": source_state.as_mapping(),
                    "source_wake_receipt": settled_receipt,
                }
            )
            document = self._read_document()
            document["records"][self._key(next_state)] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(self._key(next_state))
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("wake source 写入后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    @staticmethod
    def _merge_source_wake_receipts(
        previous: Any,
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        """合并同一 wake 的即时身份与 turn 完成后补齐的 checkpoint 摘要。"""

        current_copy = _copy_json_object(current, "wake_receipt")
        if previous is None:
            return current_copy
        previous_copy = _copy_json_object(previous, "source_wake_receipt")
        identity_fields = {"thread_id", "turn_id", "job_id"}
        checkpoint_fields = {
            "turn_request_digest",
            "thread_policy_digest",
            "turn_count_before",
        }
        allowed_fields = identity_fields | checkpoint_fields
        if set(previous_copy) - allowed_fields or set(current_copy) - allowed_fields:
            raise InvestigationJobStatusError(
                "wake source receipt 含未知字段"
            )
        if any(
            previous_copy.get(field_name) != current_copy.get(field_name)
            for field_name in identity_fields
        ):
            raise InvestigationJobStatusError("wake source receipt identity 发生漂移")
        previous_checkpoint = checkpoint_fields.intersection(previous_copy)
        current_checkpoint = checkpoint_fields.intersection(current_copy)
        if previous_checkpoint not in (set(), checkpoint_fields) or current_checkpoint not in (
            set(),
            checkpoint_fields,
        ):
            raise InvestigationJobStatusError(
                "wake source receipt checkpoint 字段不完整"
            )
        if previous_checkpoint and current_checkpoint:
            if previous_copy != current_copy:
                raise InvestigationJobStatusError(
                    "wake source receipt checkpoint 发生漂移"
                )
            return previous_copy
        if current_checkpoint:
            return current_copy
        return previous_copy

    def record_wake_receipt_after_replacement(
        self,
        next_state: InvestigationJobState,
        receipt: Mapping[str, Any],
        *,
        wake_status: str = "pending",
    ) -> Mapping[str, Any]:
        """为 replacement 补写 provider receipt，并保存指定恢复状态。"""

        if wake_status not in {"pending", "intent"}:
            raise InvestigationJobStatusError("replacement wake_status 非法")

        with self._storage_lock():
            existing = self.get(
                next_state.task_id,
                next_state.batch_id,
                next_state.combination_id,
            )
            if existing is None:
                raise InvestigationJobStatusError("下一等待作业缺少 wake receipt 记录")
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != next_state.as_mapping():
                raise InvestigationJobStatusError("下一等待作业的 wake receipt 身份不一致")
            source_mapping = existing.get("source_wake_state")
            if not isinstance(source_mapping, Mapping):
                raise InvestigationJobStatusError(
                    "replacement wake receipt 缺少 source waiting 身份"
                )
            source_state = InvestigationJobState.from_mapping(source_mapping)
            receipt_copy = _copy_json_object(receipt, "wake_receipt")
            if receipt_copy.get("job_id") != source_state.identity.job_id:
                raise InvestigationJobStatusError(
                    "replacement wake receipt 未绑定 source job"
                )
            if receipt_copy.get("thread_id") != next_state.codex_thread_id:
                raise InvestigationJobStatusError(
                    "replacement wake receipt thread 与当前 turn 不一致"
                )
            if receipt_copy.get("turn_id") != next_state.turn_id:
                raise InvestigationJobStatusError(
                    "replacement wake receipt turn 与当前 turn 不一致"
                )
            record = dict(existing)
            record["wake_status"] = wake_status
            previous_receipt = record.get("source_wake_receipt")
            record["source_wake_receipt"] = self._merge_source_wake_receipts(
                previous_receipt,
                receipt_copy,
            )
            document = self._read_document()
            document["records"][self._key(next_state)] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(self._key(next_state))
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("replacement wake receipt 写入后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))

    def record_wake_receipt_after_replacement_before_result(
        self,
        next_state: InvestigationJobState,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """在 Result 尚未落盘时保存 replacement wake receipt，保持可恢复 intent。"""

        return self.record_wake_receipt_after_replacement(
            next_state,
            receipt,
            wake_status="intent",
        )

    def clear_wake_intent_after_replacement(
        self,
        next_state: InvestigationJobState,
    ) -> Mapping[str, Any]:
        """完成只返回 secondary 的 wake handoff，清除旧 source request。"""

        with self._storage_lock():
            existing = self.get(
                next_state.task_id,
                next_state.batch_id,
                next_state.combination_id,
            )
            if existing is None:
                raise InvestigationJobStatusError("下一等待作业缺少 wake handoff 记录")
            stored_state = InvestigationJobState.from_mapping(existing.get("state", existing))
            if stored_state.as_mapping() != next_state.as_mapping():
                raise InvestigationJobStatusError("wake handoff 的下一身份不一致")
            record = dict(existing)
            for field_name in (
                "terminal_status",
                "terminal_message",
                "wake_request",
                "wake_request_digest",
                "wake_receipt",
            ):
                record.pop(field_name, None)
            record["wake_status"] = "pending"
            document = self._read_document()
            document["records"][self._key(next_state)] = record
            self._write_document(document)
            sealed = self._read_document()["records"].get(self._key(next_state))
            if not isinstance(sealed, Mapping):
                raise InvestigationJobStatusError("replacement wake handoff 后无法读取 sealed record")
            return copy.deepcopy(dict(sealed))
