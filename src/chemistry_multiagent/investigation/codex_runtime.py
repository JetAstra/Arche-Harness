"""Investigation and Experiment 专用的 Codex SDK turn runtime。

本模块把请求构造器返回的 mapping 整理成一个可恢复的 SDK turn：每条执行输入
使用独立的 ``CodexSessionStore``，任务级 Review 在同一个 task thread 上跨批次
复用。provider、MiniChem MCP 和 evidence MCP 都由调用方以显式配置注入；本模块
不读取 React provider，不创建化学提交工具，也不把凭据或 MCP capability token
写入 prompt、manifest、trace 或返回值。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from chemistry_multiagent.harness.base import HarnessError, HarnessRequest
from chemistry_multiagent.harness.codex_sdk_adapter import (
    CodexCompletedTurnReceiptHandle,
    CodexGoalReceiptHandle,
    CodexTerminalTurnError,
    CodexSdkRunResult,
    CodexProcessSandboxSpec,
    DEFAULT_CODEX_IDLE_TIMEOUT_SECONDS,
    DEFAULT_INVESTIGATION_IDLE_TIMEOUT_SECONDS,
    DEFAULT_MCP_TOOL_IDLE_TIMEOUT_SECONDS,
    build_codex_capability_config,
    build_codex_config,
    build_codex_completed_turn_receipt_handle,
    build_codex_goal_receipt_handle,
    calculate_codex_request_digest,
    calculate_codex_thread_policy_digest,
    find_codex_completed_turn_receipt,
    read_codex_completed_turn_receipt,
    read_codex_request_intent,
    resolve_codex_completed_turn_receipt_handle,
    resolve_codex_goal_receipt_handle,
    rollback_codex_thread_for_investigation_turn,
    run_codex_sdk_turn,
    validate_codex_completed_turn_receipt,
)
from chemistry_multiagent.harness.codex_auth import (
    CODEX_ALLOW_PROVIDER_MIGRATION_ENV,
    CODEX_AUTH_MODE_API,
    CODEX_AUTH_MODE_ENV,
    CODEX_AUTH_MODE_LOCAL_LOGIN,
    CODEX_AUTH_SOURCE_ENV,
    CODEX_INVESTIGATION_PERMISSION_PROFILE,
    CODEX_LOCAL_LOGIN_ENV_KEY,
    CODEX_LOCAL_LOGIN_MODEL,
    CODEX_LOCAL_LOGIN_MODEL_ENV,
    CODEX_LOCAL_LOGIN_PROVIDER_ID,
    CODEX_LOCAL_LOGIN_REASONING_EFFORT,
    CODEX_LOCAL_LOGIN_REASONING_EFFORT_ENV,
    DEFAULT_CODEX_AUTH_SOURCE,
    build_codex_auth_session,
    normalize_codex_auth_mode,
    parse_explicit_provider_migration_flag,
    prepare_codex_permission_profile,
    resolve_codex_auth_source,
)
from chemistry_multiagent.harness.codex_provider_migration import (
    persist_provider_migration_target_thread,
    persist_provider_migration_turn_id,
    persist_provider_migration_turn_context,
    resolve_explicit_provider_migration,
)
from chemistry_multiagent.harness.config import DEFAULT_CODEX_BIN
from chemistry_multiagent.harness.session_store import CodexSessionOwner, CodexSessionStore
from chemistry_multiagent.utils.api_retry import MODEL_CALL_RETRY_COUNT

from .contracts import (
    InvestigationResponse,
    TaskFinalizationResponse,
    TaskReviewResponse,
    validate_investigation_response,
    validate_task_finalization_response,
    validate_task_review_response,
)
from .async_jobs import (
    INVESTIGATION_JOB_TOOL_FAMILIES,
    TERMINAL_INVESTIGATION_JOB_STATUSES,
    InvestigationJobIdentity,
    InvestigationJobReceipt,
    InvestigationJobState,
    InvestigationJobSubmissionError,
    extract_investigation_job_receipts,
)


__all__ = [
    "CodexProviderSettings",
    "DEFAULT_INVESTIGATION_MODEL",
    "DEFAULT_INVESTIGATION_REASONING_EFFORT",
    "EvidenceMcpSettings",
    "InvestigationCodexRuntime",
    "InvestigationMcpSettings",
    "InvestigationTurnResult",
    "TerminalTurnCallback",
    "TurnStartedCallback",
]


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_ALLOWED_REQUEST_KINDS = frozenset(
    {
        "initial_investigation",
        "deepseek_repair",
        "investigation_job_wake",
        "investigation_job_continuation",
    }
)
_ROLLBACK_ELIGIBLE_INVESTIGATION_STAGES = frozenset(
    {"initial_investigation", "investigation_job_wake"}
)
_ALLOWED_RUNTIME_ROLES = frozenset({"investigation", "task_review"})
DEFAULT_INVESTIGATION_MODEL = "gpt-5.6-sol"
DEFAULT_INVESTIGATION_REASONING_EFFORT = "xhigh"
_SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)
_MCP_TOOL_TIMEOUT_SECONDS = int(DEFAULT_MCP_TOOL_IDLE_TIMEOUT_SECONDS)
# MCP text content 有时会把完整 envelope 再编码到 text 中；只允许有限层数和文本大小，
# 避免把不受信任的响应当作递归 JSON 文档无限展开。
_MAX_MCP_TEXT_ENVELOPE_DEPTH = 4
_MAX_MCP_TEXT_ENVELOPE_BYTES = 2 * 1024 * 1024
_JOB_STATUS_ALIASES = {
    "success": "succeeded",
    "completed": "succeeded",
    "complete": "succeeded",
    "done": "succeeded",
    "error": "failed",
    "failure": "failed",
    "canceled": "cancelled",
    "cancelled_by_user": "cancelled",
    "timeout": "expired",
}
_JOB_STATUS_VALUES = frozenset(
    TERMINAL_INVESTIGATION_JOB_STATUSES
    | {"queued", "pending", "running", "active", "submitted"}
)
_TRANSPORT_SUCCESS_STATUS_VALUES = frozenset({"success", "ok", "accepted"})


def _reject_nonstandard_json_constant(_value: str) -> None:
    """拒绝 Python JSON 解码器额外接受的 NaN、Infinity 等非标准常量。"""

    raise ValueError("non-standard JSON constant")


def _require_text(value: Any, field_name: str) -> str:
    """校验非空短文本；失败信息不复制调用方提供的敏感值。"""

    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{field_name} must be a non-empty string", kind="provider_envelope")
    normalized = value.strip()
    if _CONTROL_CHARACTERS.search(normalized):
        raise HarnessError(f"{field_name} contains control characters", kind="provider_envelope")
    return normalized


def _require_multiline_prompt(value: Any, field_name: str) -> str:
    """校验允许换行和制表符的非空提示词，不复制提示词正文到错误信息。"""

    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{field_name} must be a non-empty string", kind="provider_envelope")
    normalized = value.strip()
    if any(
        (ord(character) < 0x20 and character not in "\t\n\r")
        or ord(character) == 0x7F
        for character in normalized
    ):
        raise HarnessError(f"{field_name} contains control characters", kind="provider_envelope")
    return normalized


def _require_identifier(value: Any, field_name: str) -> str:
    """校验可写入会话归属记录、thread 或 trace 文件名的稳定标识。"""

    normalized = _require_text(value, field_name)
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise HarnessError(f"{field_name} is malformed", kind="session_state")
    return normalized


def _normalize_runtime_role(value: Any) -> str:
    """规范化并限制 runtime 只能绑定 Investigation 或 task Review。"""

    normalized = _require_text(value, "runtime.role").lower().replace("-", "_")
    if normalized == "review":
        normalized = "task_review"
    if normalized not in _ALLOWED_RUNTIME_ROLES:
        raise HarnessError("runtime.role is unsupported", kind="provider_envelope")
    return normalized


def _require_env_name(value: Any, field_name: str) -> str:
    """校验 provider 使用的环境变量名，而不是读取其值。"""

    normalized = _require_text(value, field_name)
    if _ENVIRONMENT_NAME.fullmatch(normalized) is None:
        raise HarnessError(f"{field_name} is malformed", kind="provider_envelope")
    return normalized


def _private_environment_name(prefix: str, seed: str) -> str:
    """为旧字面量凭据派生不含凭据正文的进程私有环境变量名。"""

    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
    return f"ARCHE_CODEX_PRIVATE_{prefix}_{digest}"


def _require_endpoint(value: Any, field_name: str) -> str:
    """校验 Streamable HTTP endpoint，允许 HTTP 或 HTTPS 但拒绝空地址。"""

    normalized = _require_text(value, field_name)
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise HarnessError(f"{field_name} must be an HTTP endpoint", kind="provider_envelope")
    return normalized


def _require_positive_timeout(value: Any, field_name: str) -> float:
    """校验一个有限的 turn 或工具局部等待时限。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessError(f"{field_name} must be a positive finite number", kind="provider_envelope")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise HarnessError(f"{field_name} must be a positive finite number", kind="provider_envelope")
    return normalized


def _require_positive_integer(value: Any, field_name: str) -> int:
    """校验进入 Codex thread config 的有限正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HarnessError(f"{field_name} must be a positive integer", kind="provider_envelope")
    return value


def _copy_json_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """复制请求中的 object，拒绝列表、标量和非字符串 object key。"""

    if not isinstance(value, Mapping):
        raise HarnessError(f"{field_name} must be an object", kind="provider_envelope")
    copied = dict(value)
    if any(not isinstance(key, str) or not key.strip() for key in copied):
        raise HarnessError(f"{field_name} has invalid object keys", kind="provider_envelope")
    return copied


def _copy_message_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    """严格复制非空消息数组，保留 prompt builder 的消息顺序。"""

    if not isinstance(value, list) or not value:
        raise HarnessError("messages must be a non-empty array", kind="provider_envelope")
    messages: list[Mapping[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise HarnessError(
                f"messages[{index}] must be an object",
                kind="provider_envelope",
            )
        messages.append(dict(message))
    return tuple(messages)


def _copy_tool_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    """严格复制可选工具描述；描述不是 MCP server 配置。"""

    if value is None:
        return ()
    if not isinstance(value, list):
        raise HarnessError("tools must be an array", kind="provider_envelope")
    tools: list[Mapping[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, Mapping):
            raise HarnessError(f"tools[{index}] must be an object", kind="provider_envelope")
        tools.append(dict(tool))
    return tuple(tools)


def _normalize_sensitive_values(values: Sequence[str]) -> tuple[str, ...]:
    """去重并冻结 provider/MCP secret 值，空值不会成为误报或替换规则。"""

    normalized: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _calculate_request_digest(request: HarnessRequest) -> str:
    """复用 adapter 的模型请求摘要，供 completed-turn receipt 精确匹配。

    摘要覆盖模型实际可见字段和业务 metadata，但忽略只负责路由到既有 session
    的 ``codex_thread_id``。顶层 thread 与 manifest 的一致性由 runtime 独立栅栏
    核对；请求含不可编码值时，adapter helper 抛出合同错误。
    """

    return calculate_codex_request_digest(request)


def _assert_sensitive_values_absent(
    value: Any,
    sensitive_values: Sequence[str],
    field_name: str,
) -> None:
    """递归拒绝敏感 MCP 值进入请求、响应或其它可持久化结构。"""

    if isinstance(value, str):
        if any(secret in value for secret in sensitive_values):
            raise HarnessError(
                f"{field_name} must not contain an MCP credential",
                kind="provider_envelope",
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_sensitive_values_absent(key, sensitive_values, field_name)
            _assert_sensitive_values_absent(item, sensitive_values, field_name)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            _assert_sensitive_values_absent(item, sensitive_values, field_name)


def _exception_chain_contains_sensitive_value(
    error: BaseException,
    sensitive_values: Sequence[str],
) -> bool:
    """递归检查异常正文、参数、cause/context，防止错误链回显凭据。"""

    seen: set[int] = set()

    def visit(current: BaseException | None) -> bool:
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        if any(
            isinstance(item, str) and any(secret in item for secret in sensitive_values)
            for item in current.args
        ):
            return True
        nested = getattr(current, "exceptions", None)
        if isinstance(nested, (tuple, list)) and any(
            isinstance(item, BaseException) and visit(item) for item in nested
        ):
            return True
        return visit(current.__cause__) or visit(current.__context__)

    return visit(error)


def _sanitize_runtime_error_chain(
    error: BaseException,
    sensitive_values: Sequence[str],
) -> BaseException:
    """原异常链含凭据时返回不带 cause/context 的统一错误。"""

    if _exception_chain_contains_sensitive_value(error, sensitive_values):
        return HarnessError(
            "Codex runtime failed with redacted sensitive error",
            kind="transport",
        )
    return error


def _prepare_secure_directory(path: Path, field_name: str) -> Path:
    """创建并收紧 runtime 根目录，避免 trace 或会话文件被同组用户读取。"""

    try:
        path = path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise HarnessError(
            f"{field_name} cannot be prepared",
            kind="session_state",
        ) from exc
    return path


@dataclass(frozen=True, slots=True)
class CodexProviderSettings:
    """一个 Codex provider 的公开身份、模型和推理强度环境绑定。

    ``reasoning_effort_env`` 只保存环境变量名，不读取或保存变量值；默认值用于
    未设置该环境变量的旧启动入口。Task Review 不设置这两个字段，因此不会把
    Investigation 的 ``max`` 推理强度误传给 DeepSeek Review。
    """

    provider_id: str
    env_key: str
    base_url_env: str
    model_env: str
    reasoning_effort_env: str | None = None
    reasoning_effort_default: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _require_identifier(self.provider_id, "provider_id"))
        object.__setattr__(self, "env_key", _require_env_name(self.env_key, "env_key"))
        object.__setattr__(self, "base_url_env", _require_env_name(self.base_url_env, "base_url_env"))
        object.__setattr__(self, "model_env", _require_env_name(self.model_env, "model_env"))
        if self.reasoning_effort_env is not None:
            object.__setattr__(
                self,
                "reasoning_effort_env",
                _require_env_name(self.reasoning_effort_env, "reasoning_effort_env"),
            )
        if self.reasoning_effort_default is not None:
            normalized = _require_text(
                self.reasoning_effort_default,
                "reasoning_effort_default",
            ).lower()
            if normalized not in _SUPPORTED_REASONING_EFFORTS:
                raise HarnessError(
                    "reasoning_effort_default is unsupported",
                    kind="provider_envelope",
                )
            object.__setattr__(self, "reasoning_effort_default", normalized)
        if self.reasoning_effort_default is not None and self.reasoning_effort_env is None:
            raise HarnessError(
                "reasoning_effort_default requires reasoning_effort_env",
                kind="provider_envelope",
            )


@dataclass(frozen=True, slots=True)
class InvestigationMcpSettings:
    """调查阶段唯一可连接的 MiniChem server 及其 capability 环境变量。"""

    endpoint: str
    capability_token_env: str
    _legacy_capability_token: str | None = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        endpoint: str,
        capability_token_env: str = "ARCHE_MINICHEM_CAPABILITY_TOKEN",
    ) -> None:
        object.__setattr__(self, "endpoint", _require_endpoint(endpoint, "minichem_endpoint"))
        candidate = _require_text(capability_token_env, "minichem_capability_token_env")
        if _ENVIRONMENT_NAME.fullmatch(candidate) and candidate.upper() == candidate:
            object.__setattr__(self, "capability_token_env", candidate)
            object.__setattr__(self, "_legacy_capability_token", None)
            return
        # 兼容旧调用方传入 token 字面量，但不再把它放进 MCP 配置；运行时会
        # 转换为只存在于 app-server 私有环境的临时变量名。
        private_name = _private_environment_name("MINICHEM", "capability")
        object.__setattr__(self, "capability_token_env", private_name)
        object.__setattr__(self, "_legacy_capability_token", candidate)

    def legacy_secret_environment(self) -> dict[str, str]:
        """返回兼容旧构造器所需的私有环境注入，不进入 thread config。"""

        if self._legacy_capability_token is None:
            return {}
        return {self.capability_token_env: self._legacy_capability_token}


@dataclass(frozen=True, slots=True)
class EvidenceMcpSettings:
    """只读 evidence server 及 header 名到环境变量名的映射。"""

    endpoint: str
    env_http_headers: Mapping[str, str] = field(repr=False, compare=False)
    _legacy_secret_environment: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        endpoint: str,
        env_http_headers: Mapping[str, str] | None = None,
        *,
        http_headers: Mapping[str, str] | None = None,
    ) -> None:
        object.__setattr__(self, "endpoint", _require_endpoint(endpoint, "evidence_endpoint"))
        if env_http_headers is not None and http_headers is not None:
            raise HarnessError(
                "evidence HTTP headers were provided twice",
                kind="provider_envelope",
            )
        source = env_http_headers if env_http_headers is not None else http_headers
        if not isinstance(source, Mapping) or not source:
            raise HarnessError(
                "evidence capability header is required via env_http_headers",
                kind="provider_envelope",
            )
        normalized: dict[str, str] = {}
        legacy: dict[str, str] = {}
        for raw_name, raw_value in source.items():
            name = _require_text(raw_name, "evidence_header_name")
            value = _require_text(raw_value, "evidence_header_env")
            if any(char in name for char in "\r\n:"):
                raise HarnessError("evidence HTTP header is malformed", kind="provider_envelope")
            if _ENVIRONMENT_NAME.fullmatch(value):
                normalized[name] = value
                continue
            # 旧版本接受字面量 http_headers。立即转换成私有环境变量，保证
            # Codex thread config 永远只出现 env_http_headers。
            private_name = _private_environment_name("EVIDENCE", name)
            normalized[name] = private_name
            legacy[private_name] = value
        object.__setattr__(self, "env_http_headers", normalized)
        object.__setattr__(self, "_legacy_secret_environment", legacy)

    def legacy_secret_environment(self) -> dict[str, str]:
        """返回兼容旧字面量 header 的 app-server 私有环境注入。"""

        return dict(self._legacy_secret_environment)


@dataclass(frozen=True, slots=True)
class InvestigationTurnResult:
    """返回已通过合同验证的科学结果及可恢复 thread/turn 标识。

    ``accepted_job`` 只由 SDK 收到的真实 MCP submit 回执生成，绝不来自模型最终
    JSON；它只表示当前 turn 结束时仍有未 collect 的活跃作业，供 Controller 持久化
    等待身份。``completed_jobs`` 保存同一 turn 已经真实 collect 的 submit 身份，不能
    替代科学结果。``receipt_handle`` 是延迟确认 completed receipt 的句柄：返回
    accepted job 时必须等等待 sidecar 和 resumable waiting 事实落盘后确认，返回科学
    Result 时必须等 Evidence/Working Memory Result（及 repair 完成）落盘后确认；runtime
    不在返回前删除 receipt。
    """

    result: InvestigationResponse | TaskReviewResponse | TaskFinalizationResponse | None
    thread_id: str
    turn_id: str
    request_digest: str | None = None
    accepted_job: Mapping[str, Any] | None = None
    completed_jobs: tuple[Mapping[str, Any], ...] = ()
    receipt_handle: CodexCompletedTurnReceiptHandle | CodexGoalReceiptHandle | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedProvider:
    """只保留 provider 的非敏感运行值；不保存 API key。"""

    settings: CodexProviderSettings
    base_url: str
    model: str
    reasoning_effort: str | None = None
    secret_environment: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PreparedCodexTurn:
    """保存已经通过会话、策略和秘密检查的一次 SDK turn 输入。"""

    request: HarnessRequest
    request_digest: str
    secret_environment: Mapping[str, str]
    thread_config: Mapping[str, Any]
    thread_policy_digest: str
    manifest: Mapping[str, Any] | None
    recovery: Mapping[str, Any] | None
    session_id: str | None
    turn_count_before: int
    completed_turn_receipt_root: Path
    trace_path: Path
    sdk_config: Any
    stage: str
    permission_profile: str | None = None
    store: CodexSessionStore | None = None
    migrated_from_thread_id: str | None = None
    migration_request_thread_ids: tuple[str, ...] = ()
    replace_unstarted_thread_id: str | None = None
    migration_target_thread_id: str | None = None
    migration_target_thread_callback: Callable[[str], None] | None = None
    migration_turn_started_callback: Callable[[str], None] | None = None
    recovery_only: bool = False
    recovery_turn_id: str | None = None
    replay_turn_id: str | None = None


def _build_wake_job_receipt(request: HarnessRequest) -> InvestigationJobReceipt | None:
    """从已校验 wake request 恢复原 job 的 tracker 身份，不伪造新的 submit。"""

    if request.metadata.get("request_kind") != "investigation_job_wake":
        return None
    family = request.metadata.get("job_family")
    family_tools = INVESTIGATION_JOB_TOOL_FAMILIES.get(family)
    if family_tools is None:
        raise HarnessError("wake request job family is unsupported", kind="provider_envelope")
    identity: dict[str, Any] = {
        "job_family": family,
        "job_id": request.metadata.get("job_id"),
        "operation_id": request.metadata.get("operation_id"),
        "submission_key": request.metadata.get("submission_key"),
    }
    for field_name in (
        "action_id",
        "backend_id",
        "job_type",
        "runtime",
        "software_id",
        "operation_fingerprint",
    ):
        if field_name in request.metadata:
            identity[field_name] = request.metadata[field_name]
    try:
        return InvestigationJobReceipt.from_mapping(
            {
                "tool_name": family_tools["submit"],
                "identity": identity,
                "arguments": {"job_id": identity["job_id"]},
                "result": identity,
            }
        )
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "wake request 缺少可核对的原 job identity",
            kind="provider_envelope",
        ) from exc


def _build_active_job_receipt(
    active_job: InvestigationJobState | Mapping[str, Any],
) -> InvestigationJobReceipt:
    """从 Controller 已保存的等待状态恢复 tracker，不创建或提交新作业。"""

    state = active_job
    if not isinstance(state, InvestigationJobState):
        try:
            state = InvestigationJobState.from_mapping(state)
        except (TypeError, ValueError) as exc:
            raise HarnessError(
                "active Investigation job state is invalid",
                kind="session_state",
            ) from exc
    identity = state.identity
    return InvestigationJobReceipt(
        tool_name=state.submit_tool,
        identity=identity,
        arguments={"job_id": identity.job_id},
        result=identity.as_mapping(),
    )


def _build_request_job_identity(
    metadata: Mapping[str, Any],
) -> InvestigationJobIdentity:
    """从恢复 request metadata 恢复完整作业身份，缺失或非法时 fail-closed。"""

    identity_fields = {
        "job_id": metadata.get("job_id"),
        "operation_id": metadata.get("operation_id"),
        "submission_key": metadata.get("submission_key"),
        "job_family": metadata.get("job_family"),
    }
    for field_name in (
        "action_id",
        "backend_id",
        "job_type",
        "runtime",
        "software_id",
        "operation_fingerprint",
    ):
        if field_name in metadata:
            identity_fields[field_name] = metadata[field_name]
    try:
        return InvestigationJobIdentity.from_mapping(
            identity_fields,
            label="investigation recovery request identity",
        )
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "investigation recovery request job identity is invalid",
            kind="session_state",
        ) from exc


@dataclass(slots=True)
class _CodexTurnCallbacks:
    """保存 adapter 回调的会话状态，避免用闭包隐藏持久化步骤。"""

    store: CodexSessionStore
    manifest: Mapping[str, Any] | None
    session_id: str | None
    stage: str
    request_digest: str
    thread_policy_digest: str
    thread_ready_callback: Callable[[str], None] | None = None
    thread_migration_callback: Callable[[str, tuple[str, ...], str], None] | None = None
    migration_target_thread_callback: Callable[[str], None] | None = None
    migration_turn_started_callback: Callable[[str], None] | None = None
    migration_source_thread_id: str | None = None
    migration_request_thread_ids: tuple[str, ...] = ()
    replace_unstarted_thread_id: str | None = None
    job_accepted_callback: Callable[[Mapping[str, Any]], None] | None = None
    job_collected_callback: Callable[[Mapping[str, Any]], None] | None = None
    registered_thread_id: str | None = None
    accepted_job_receipts: list[InvestigationJobReceipt] = field(default_factory=list)
    completed_job_receipts: list[InvestigationJobReceipt] = field(default_factory=list)
    active_job_receipt: InvestigationJobReceipt | None = None
    active_job_terminal: bool = False
    active_job_collected: bool = False
    submit_attempt_failed: bool = False

    def register_thread_ready(self, thread_id: str) -> None:
        """在真实 turn 前登记 thread，并拒绝恢复时的 thread 漂移。"""

        normalized = _require_identifier(thread_id, "thread_id")
        if self.session_id is not None and normalized != self.session_id:
            raise HarnessError(
                "Codex resume returned a different thread_id",
                kind="session_protocol",
            )
        # target thread 的身份必须先写入 migration marker，再写 target manifest。
        # 如果进程在这两个文件之间退出，下一次恢复会沿 marker resume 同一
        # provider thread，而不会再次 thread_start。
        if (
            self.migration_target_thread_callback is not None
            and self.migration_source_thread_id is not None
            and self.manifest is None
        ):
            self.migration_target_thread_callback(normalized)
        if self.replace_unstarted_thread_id is not None:
            self.store.replace_unstarted_thread(
                self.replace_unstarted_thread_id,
                normalized,
                stage=self.stage,
                status="created",
                thread_policy_digest=self.thread_policy_digest,
            )
        elif self.manifest is None:
            self.store.create(normalized, stage=self.stage, status="created")
        self.registered_thread_id = normalized
        if (
            self.thread_migration_callback is not None
            and self.migration_source_thread_id is not None
        ):
            self.thread_migration_callback(
                self.migration_source_thread_id,
                self.migration_request_thread_ids,
                normalized,
            )
        if self.thread_ready_callback is not None:
            self.thread_ready_callback(normalized)

    def capture_tool_completed(self, item: Any) -> None:
        """按真实 submit/get/collect 事件维护单个活跃作业生命周期。"""

        if self.submit_attempt_failed:
            raise HarnessError(
                "submit 回执已经失败关闭，当前 turn 不能重试计算",
                kind="response_contract",
            )
        try:
            receipts = extract_investigation_job_receipts((item,))
        except InvestigationJobSubmissionError:
            self.submit_attempt_failed = True
            raise
        if receipts:
            receipt = receipts[0]
            if self.active_job_receipt is not None and not self.active_job_collected:
                raise HarnessError(
                    "同一个 Investigation turn 仍有未 collect 的活跃作业，不能并行提交第二个任务",
                    kind="response_contract",
                )
            self.accepted_job_receipts.append(receipt)
            self.active_job_receipt = receipt
            self.active_job_terminal = False
            self.active_job_collected = False
            if self.job_accepted_callback is not None:
                self.job_accepted_callback(receipt.as_mapping())
            return
        self._capture_job_progress(item)

    @staticmethod
    def _item_field(value: Any, field_name: str) -> Any:
        """读取 MCP completed item 的 snake/camel 字段。"""

        aliases = {
            "job_id": "jobId",
            "structured_content": "structuredContent",
            "operation_id": "operationId",
            "submission_key": "submissionKey",
        }
        alias = aliases.get(field_name, field_name)
        if isinstance(value, Mapping):
            if field_name in value:
                return value[field_name]
            return value.get(alias)
        direct = getattr(value, field_name, None)
        return direct if direct is not None else getattr(value, alias, None)

    @classmethod
    def _payload_candidates(cls, value: Any) -> tuple[Mapping[str, Any], ...]:
        """从工具结果取结构化 payload；不读取 assistant 文本。"""

        candidates: list[Mapping[str, Any]] = []
        structured = cls._item_field(value, "structured_content")
        pending: list[tuple[Any, int, bool]] = [
            (value, 0, False),
            (structured, 0, False),
        ]
        seen_container_ids: set[int] = set()
        while pending:
            candidate, text_depth, decode_text = pending.pop()
            # Mapping、序列和 SDK/Pydantic 协议对象都可能通过已知字段互相引用；
            # 标量没有可展开的字段，不加入集合，避免自引用对象耗尽循环。
            if not isinstance(
                candidate,
                (str, bytes, bytearray, int, float, bool, type(None)),
            ):
                candidate_id = id(candidate)
                if candidate_id in seen_container_ids:
                    continue
                seen_container_ids.add(candidate_id)
            if isinstance(candidate, Mapping):
                if decode_text:
                    decoded = cls._decode_json_text_envelope(
                        cls._item_field(candidate, "text")
                    )
                    if decoded is not None and text_depth < _MAX_MCP_TEXT_ENVELOPE_DEPTH:
                        pending.append((decoded, text_depth + 1, False))
                candidates.append(candidate)

                nested_structured = cls._item_field(candidate, "structured_content")
                if isinstance(nested_structured, Mapping) or (
                    isinstance(nested_structured, Sequence)
                    and not isinstance(nested_structured, (str, bytes, bytearray))
                ):
                    pending.append((nested_structured, text_depth, False))
                content = cls._item_field(candidate, "content")
                if isinstance(content, Sequence) and not isinstance(
                    content, (str, bytes, bytearray)
                ):
                    for item in reversed(content):
                        pending.append((item, text_depth, True))
            elif isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray)
            ):
                for item in reversed(candidate):
                    pending.append((item, text_depth, True))
            else:
                # SDK/Pydantic 的 MCP result 和 content block 不是 Mapping；只通过
                # 已知协议字段展开，不能遍历对象属性或把 request.arguments 当作结果。
                nested_structured = cls._item_field(candidate, "structured_content")
                if nested_structured is not None:
                    pending.append((nested_structured, text_depth, False))
                content = cls._item_field(candidate, "content")
                if isinstance(content, Sequence) and not isinstance(
                    content, (str, bytes, bytearray)
                ):
                    for item in reversed(content):
                        pending.append((item, text_depth, True))
                if decode_text and text_depth < _MAX_MCP_TEXT_ENVELOPE_DEPTH:
                    decoded = cls._decode_json_text_envelope(
                        cls._item_field(candidate, "text")
                    )
                    if decoded is not None:
                        pending.append((decoded, text_depth + 1, False))
        return tuple(candidates)

    @staticmethod
    def _decode_json_text_envelope(value: Any) -> Mapping[str, Any] | list[Any] | None:
        """只解码有限大小的严格 JSON object/array text，不从自然语言猜测结果。"""

        if not isinstance(value, str) or not value.strip():
            return None
        try:
            text_size = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            return None
        if text_size > _MAX_MCP_TEXT_ENVELOPE_BYTES:
            return None
        try:
            decoded = json.loads(value, parse_constant=_reject_nonstandard_json_constant)
        except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
            return None
        if isinstance(decoded, Mapping):
            return decoded
        if isinstance(decoded, list):
            return decoded
        return None

    @classmethod
    def _job_ids_from_result(cls, item: Any) -> set[str]:
        """提取工具结果中明确返回的 job_id，并拒绝重复身份冲突。"""

        root = cls._item_field(item, "root") or item
        ids: set[str] = set()
        for payload in cls._payload_candidates(cls._item_field(root, "result")):
            job_id = cls._item_field(payload, "job_id")
            nested = payload.get("job")
            if nested is not None and not isinstance(nested, Mapping):
                raise HarnessError(
                    "异步作业回执的 job 必须是 object",
                    kind="response_contract",
                )
            nested_id = cls._item_field(nested, "job_id") if isinstance(nested, Mapping) else None
            normalized_ids = {
                value.strip()
                for value in (job_id, nested_id)
                if isinstance(value, str) and value.strip()
            }
            if len(normalized_ids) > 1:
                raise HarnessError(
                    "异步作业回执的顶层与嵌套 job_id 冲突",
                    kind="response_contract",
                )
            ids.update(normalized_ids)
        if len(ids) > 1:
            raise HarnessError(
                "异步作业回执包含多个互相冲突的 job_id",
                kind="response_contract",
            )
        return ids

    @classmethod
    def _job_ids_from_arguments(cls, item: Any) -> set[str]:
        """提取工具请求参数中的 job_id，供请求与回执身份逐一核对。"""

        root = cls._item_field(item, "root") or item
        arguments = cls._item_field(root, "arguments")
        ids: set[str] = set()
        stack = [arguments]
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                job_id = cls._item_field(value, "job_id")
                if isinstance(job_id, str) and job_id.strip():
                    ids.add(job_id.strip())
                stack.extend(value.values())
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                stack.extend(value)
        return ids

    @classmethod
    def _job_ids_from_item(cls, item: Any) -> set[str]:
        """提取工具结果和参数中明确出现的 job_id，兼容旧的只读调用方。"""

        return cls._job_ids_from_result(item) | cls._job_ids_from_arguments(item)

    @classmethod
    def _payload_statuses(cls, item: Any) -> set[str]:
        """读取状态工具返回的作业状态，并隔离 transport status 与嵌套状态。"""

        root = cls._item_field(item, "root") or item
        statuses: set[str] = set()
        explicit_statuses: set[str] = set()
        for payload in cls._payload_candidates(cls._item_field(root, "result")):
            nested = payload.get("job")
            if nested is not None and not isinstance(nested, Mapping):
                raise HarnessError(
                    "异步作业状态的 job 必须是 object",
                    kind="response_contract",
                )

            def normalize_explicit_status(value: Any, field_name: str) -> str:
                """规范化明确作业状态字段，未知值停止生命周期推断。"""

                if not isinstance(value, str) or not value.strip():
                    raise HarnessError(
                        f"异步作业状态字段 {field_name} 非法",
                        kind="response_contract",
                    )
                normalized = value.strip().casefold().replace("-", "_")
                normalized = _JOB_STATUS_ALIASES.get(normalized, normalized)
                if normalized not in _JOB_STATUS_VALUES:
                    raise HarnessError(
                        f"异步作业状态字段 {field_name} 不受支持",
                        kind="response_contract",
                    )
                return normalized

            top_values = [
                normalize_explicit_status(payload[field_name], field_name)
                for field_name in ("job_status", "state")
                if field_name in payload
            ]
            nested_values = (
                [
                    normalize_explicit_status(nested[field_name], f"job.{field_name}")
                    for field_name in ("status", "state")
                    if field_name in nested
                ]
                if isinstance(nested, Mapping)
                else []
            )
            if len(set(top_values)) > 1 or len(set(nested_values)) > 1:
                raise HarnessError(
                    "异步作业状态包含互相冲突的重复字段",
                    kind="response_contract",
                )
            if top_values and nested_values and top_values[0] != nested_values[0]:
                raise HarnessError(
                    "异步作业状态的顶层与嵌套字段冲突",
                    kind="response_contract",
                )

            status = top_values[0] if top_values else (nested_values[0] if nested_values else None)
            raw_status = payload.get("status")
            normalized_status = (
                raw_status.strip().casefold().replace("-", "_")
                if isinstance(raw_status, str)
                else ""
            )
            known_status = _JOB_STATUS_ALIASES.get(normalized_status, normalized_status)
            if (
                status is not None
                and known_status in _JOB_STATUS_VALUES
                and normalized_status not in _TRANSPORT_SUCCESS_STATUS_VALUES
                and known_status != status
            ):
                raise HarnessError(
                    "异步作业 status 与明确 job 状态字段冲突",
                    kind="response_contract",
                )
            # MCP 外层通常用 status=success 表示 transport/tool 调用成功；裸
            # success 不能代表作业完成。只有没有专用字段时，才把已知的其它
            # status 值视为作业状态。
            if status is None:
                normalized_transport_status = normalized_status
                if normalized_transport_status not in _TRANSPORT_SUCCESS_STATUS_VALUES:
                    status = _JOB_STATUS_ALIASES.get(
                        normalized_transport_status,
                        normalized_transport_status,
                    )
                    if status not in _JOB_STATUS_VALUES:
                        status = None
            if status is not None:
                statuses.add(status)
                explicit_statuses.add(status)
        if len(explicit_statuses) > 1:
            raise HarnessError(
                "一次异步作业回执包含多个互相冲突的明确作业状态",
                kind="response_contract",
            )
        return statuses

    @classmethod
    def _collect_error_is_terminal_result(
        cls,
        payload: Mapping[str, Any],
        active_job_id: str,
    ) -> bool:
        """输入 collect payload 和当前 job_id；判断错误是否为该作业的已收集终态。

        动作只接受 payload 自身绑定当前 job_id、明确的 transport 成功、``ready=true``
        外层和 terminal job 状态；不修改任何 tracker。返回 ``True`` 表示错误是已
        收集的失败终态诊断，返回 ``False`` 表示 collect 仍不可收集；状态字段若已由
        ``_payload_statuses`` 校验失败，则继续按既有合同抛出异常。
        """

        nested_job = payload.get("job")
        payload_job_ids = {
            value.strip()
            for value in (
                payload.get("job_id"),
                nested_job.get("job_id") if isinstance(nested_job, Mapping) else None,
            )
            if isinstance(value, str) and value.strip()
        }
        if payload_job_ids != {active_job_id}:
            return False
        transport_status = payload.get("status")
        if not isinstance(transport_status, str):
            return False
        normalized_transport_status = transport_status.strip().casefold().replace("-", "_")
        if normalized_transport_status not in _TRANSPORT_SUCCESS_STATUS_VALUES:
            return False
        return bool(
            cls._payload_statuses({"result": payload})
            & TERMINAL_INVESTIGATION_JOB_STATUSES
        )

    @classmethod
    def _list_inventory_is_active_terminal_job(
        cls,
        item: Any,
        receipt: InvestigationJobReceipt,
    ) -> bool:
        """判断当前 turn 的完整 scope 清单是否足以确认活跃作业已终态。

        输入是 ``list_execution_jobs`` 的 MCP completed item 和当前 turn 的提交回执；
        动作只接受 transport 成功、完整且恰有一项匹配当前 active ``job_id`` 的清单；
        其它历史作业可以同时存在。动作逐项核对匹配项的
        ``job_id``、``operation_id``、``submission_key``、``terminal`` 和明确的
        ``job_status``。返回 True 只表示当前状态记录获得了执行层终态证据，绝不
        表示已 collect；清单缺失、没有匹配项、匹配项重复、非终态或不完整时返回
        False，身份明确冲突时抛出异常以保持 fail-closed。
        """

        root = cls._item_field(item, "root") or item
        raw_result = cls._item_field(root, "result")
        structured_content = cls._item_field(raw_result, "structured_content")
        if structured_content is not None:
            # Codex MCP result 可能同时带 structuredContent 和 content.text；前者
            # 是明确的结构化回执，优先使用它，不把同一结果的文本副本算成重复清单。
            if not isinstance(structured_content, Mapping) or (
                "jobs" not in structured_content
            ):
                return False
            payload = structured_content
        else:
            payloads = tuple(
                payload
                for payload in cls._payload_candidates(raw_result)
                if "jobs" in payload
            )
            # 没有 structuredContent 时，只有一个严格解码的 text envelope 才能作为
            # 清单；多个嵌套清单不能猜测哪一个是权威结果。
            if len(payloads) != 1:
                return False
            payload = payloads[0]
        transport_status = payload.get("status")
        if not isinstance(transport_status, str):
            return False
        normalized_transport_status = transport_status.strip().casefold().replace(
            "-", "_"
        )
        if normalized_transport_status not in _TRANSPORT_SUCCESS_STATUS_VALUES:
            return False
        if payload.get("inventory_complete") is not True:
            return False
        jobs = payload.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            return False
        active_job_id = receipt.identity.job_id
        active_operation_id = receipt.identity.operation_id
        active_submission_key = receipt.identity.submission_key
        matching_jobs: list[Mapping[str, Any]] = []
        for candidate in jobs:
            if not isinstance(candidate, Mapping):
                return False
            candidate_job_id = cls._item_field(candidate, "job_id")
            if not isinstance(candidate_job_id, str) or not candidate_job_id.strip():
                return False
            normalized_candidate_job_id = candidate_job_id.strip()
            if normalized_candidate_job_id == active_job_id:
                matching_jobs.append(candidate)
                continue
            candidate_operation_id = cls._item_field(candidate, "operation_id")
            candidate_submission_key = cls._item_field(candidate, "submission_key")
            if candidate_operation_id == active_operation_id or (
                candidate_submission_key == active_submission_key
            ):
                raise HarnessError(
                    "list_execution_jobs 回执把当前 operation/submission 绑定到其它 job_id",
                    kind="response_contract",
                )
        if len(matching_jobs) > 1:
            raise HarnessError(
                "list_execution_jobs 回执重复返回当前 active job_id",
                kind="response_contract",
            )
        if not matching_jobs:
            return False
        job = matching_jobs[0]

        expected_identity = {
            "job_id": active_job_id,
            "operation_id": active_operation_id,
            "submission_key": active_submission_key,
        }
        for field_name, expected_value in expected_identity.items():
            actual_value = cls._item_field(job, field_name)
            if actual_value is None:
                return False
            if not isinstance(actual_value, str) or not actual_value.strip():
                return False
            if actual_value.strip() != expected_value:
                raise HarnessError(
                    "list_execution_jobs 回执与当前 active job identity 冲突",
                    kind="response_contract",
                )

        if job.get("terminal") is not True:
            return False
        raw_job_status = job.get("job_status")
        if not isinstance(raw_job_status, str) or not raw_job_status.strip():
            return False
        normalized_job_status = raw_job_status.strip().casefold().replace("-", "_")
        normalized_job_status = _JOB_STATUS_ALIASES.get(
            normalized_job_status,
            normalized_job_status,
        )
        if normalized_job_status not in _JOB_STATUS_VALUES:
            raise HarnessError(
                "list_execution_jobs 回执的 job_status 不受支持",
                kind="response_contract",
            )
        if normalized_job_status not in TERMINAL_INVESTIGATION_JOB_STATUSES:
            return False
        # 复用既有状态冲突检查，防止 job.status/state 或外层 status 与明确
        # job_status 相互矛盾；list item 的 status 若存在则不能冒充 transport。
        cls._payload_statuses({"result": job})
        return True

    def _capture_job_progress(self, item: Any) -> None:
        """确认当前 job 的 terminal get 和 collect，完成后释放 submit 门禁。"""

        receipt = self.active_job_receipt
        if receipt is None or self.active_job_collected:
            return
        root = self._item_field(item, "root") or item
        tool_name = self._item_field(root, "tool")
        family_tools = INVESTIGATION_JOB_TOOL_FAMILIES[receipt.identity.job_family]
        allowed_status_tools = {family_tools["status"]}
        allowed_collect_tools = {family_tools["collect"]}
        allowed_inventory_tools = {"list_execution_jobs"}
        if receipt.identity.job_family == "transition_state_path":
            allowed_status_tools.add("get_transition_state_path_job")
            allowed_collect_tools.add("collect_transition_state_path_job")
        if tool_name not in (
            allowed_status_tools | allowed_collect_tools | allowed_inventory_tools
        ):
            return
        item_status = self._item_field(root, "status")
        normalized_item_status = getattr(item_status, "value", item_status)
        # app-server 会用 item.status=failed 表示工具参数校验或调用本身失败；
        # 此时 result 只有错误文本，没有可核对的作业回执。保留当前 active job
        # 和 submit 门禁，让模型重试同一查询或由后续恢复继续查询；只有工具调用
        # 成功后返回的结构化身份才参与 job_id 冲突检查。
        if normalized_item_status == "failed":
            return
        if tool_name in allowed_inventory_tools:
            if self._list_inventory_is_active_terminal_job(item, receipt):
                self.active_job_terminal = True
            return
        result_job_ids = self._job_ids_from_result(root)
        argument_job_ids = self._job_ids_from_arguments(root)
        active_job_id = receipt.identity.job_id

        # MCP scope/registry 已经控制模型可查看的作业集合。模型查询当前
        # scope 内任意历史作业时，该回执不能改写当前未收集作业的 active
        # tracker；只有请求参数明确指向 active job，才进入本地身份核对。
        if not argument_job_ids or active_job_id not in argument_job_ids:
            return
        if active_job_id not in result_job_ids:
            raise HarnessError(
                "异步作业 get/collect 回执没有匹配当前 active job_id",
                kind="response_contract",
            )
        if tool_name in allowed_status_tools:
            if self._payload_statuses(root) & TERMINAL_INVESTIGATION_JOB_STATUSES:
                self.active_job_terminal = True
            return
        if not self.active_job_terminal:
            raise HarnessError(
                "collect 前必须在同一 turn 通过 get 核对匹配 job 的终态",
                kind="response_contract",
            )
        payloads = self._payload_candidates(self._item_field(root, "result"))
        terminal_statuses = self._payload_statuses(root) & TERMINAL_INVESTIGATION_JOB_STATUSES
        collect_error_payloads = [
            payload for payload in payloads if isinstance(payload.get("error"), Mapping)
        ]
        # collect 的错误只有在同一个 payload 同时具备 transport 成功、ready 和
        # terminal job 状态时，才是已收集的失败作业诊断；其它错误仍保留 active
        # gate，让模型读取原始错误并修正请求或转入恢复路径。
        if collect_error_payloads and not any(
            payload.get("ready") is True
            and self._collect_error_is_terminal_result(payload, active_job_id)
            for payload in collect_error_payloads
        ):
            return
        if not any(
            payload.get("ready") is True
            or isinstance(payload.get("action_result"), Mapping)
            or terminal_statuses
            for payload in payloads
        ):
            raise HarnessError(
                "collect 回执缺少匹配作业的 ready、终态或完整结果标记",
                kind="response_contract",
            )
        self.active_job_collected = True
        self.completed_job_receipts.append(receipt)
        if self.job_collected_callback is not None:
            self.job_collected_callback(receipt.as_mapping())

    def record_worker_recovery(
        self,
        turn_id: str | None,
        *,
        reason: str = "worker_cleanup",
    ) -> None:
        """在 adapter 抛出恢复错误前保存对应原因的恢复阻断记录。"""

        self.store.record_recovery_pending(
            stage=self.stage,
            request_digest=self.request_digest,
            turn_id=turn_id,
            thread_policy_digest=self.thread_policy_digest,
            reason=reason,
        )


TurnStartedCallback = Callable[[str, str], None]
"""一次性接收真实 thread/turn 身份的生命周期回调。"""


TerminalTurnCallback = Callable[[str, str, str], None]
"""同步接收 failed/interrupted thread、turn 和有限状态枚举的回调。"""


ThreadReadyCallback = Callable[[str], None]
"""在真实 Codex thread 返回、模型 turn 尚未提交时接收 thread 身份。"""


ThreadMigrationCallback = Callable[[str, tuple[str, ...], str], None]
"""接收 runtime 已核验的 source、允许链尾和新 target thread 身份。"""


JobAcceptedCallback = Callable[[Mapping[str, Any]], None]
"""在真实 submit 工具完成时接收脱敏作业回执。"""


JobCollectedCallback = Callable[[Mapping[str, Any]], None]
"""在真实 get 终态和 collect 工具完成时接收脱敏作业回执。"""


LegacyTurnStartedCallback = Callable[[str], None]
"""旧版只接收 turn 身份的兼容回调。"""


def _validate_runtime_mcp_composition(
    role: str | None,
    minichem: InvestigationMcpSettings | None,
    evidence_mcp: EvidenceMcpSettings | None,
) -> str:
    """确定 runtime 角色并验证该角色唯一允许的 MCP 组合。"""

    if minichem is not None and not isinstance(minichem, InvestigationMcpSettings):
        raise TypeError("minichem must be InvestigationMcpSettings or None")
    if evidence_mcp is not None and not isinstance(evidence_mcp, EvidenceMcpSettings):
        raise TypeError("evidence_mcp must be EvidenceMcpSettings or None")
    if role is None:
        if minichem is not None and evidence_mcp is None:
            role = "investigation"
        elif minichem is None and evidence_mcp is not None:
            role = "task_review"
        else:
            raise HarnessError(
                "runtime role and MCP composition are ambiguous",
                kind="provider_envelope",
            )
    normalized_role = _normalize_runtime_role(role)
    if normalized_role == "investigation":
        if minichem is None or evidence_mcp is not None:
            raise HarnessError(
                "investigation runtime requires MiniChem and forbids evidence MCP",
                kind="provider_envelope",
            )
    elif minichem is not None or evidence_mcp is None:
        raise HarnessError(
            "task_review runtime requires evidence MCP and forbids MiniChem",
            kind="provider_envelope",
        )
    return normalized_role


def _validate_provider_settings(
    investigation_provider: CodexProviderSettings,
    review_provider: CodexProviderSettings,
) -> None:
    """确认两个角色都绑定到公开 provider 配置对象。"""

    if not isinstance(investigation_provider, CodexProviderSettings):
        raise TypeError("investigation_provider must be CodexProviderSettings")
    if not isinstance(review_provider, CodexProviderSettings):
        raise TypeError("review_provider must be CodexProviderSettings")


def _build_runtime_environment(
    source_environment: Mapping[str, str],
    active_provider: CodexProviderSettings,
    minichem: InvestigationMcpSettings | None,
    evidence_mcp: EvidenceMcpSettings | None,
    *,
    auth_mode: str = CODEX_AUTH_MODE_API,
) -> dict[str, str]:
    """仅复制当前角色需要的环境名和值，兼容旧的字面量 secret 构造。

    ``local_login`` 明确走 vendored Codex 的内置 ``openai`` provider，因此不把
    API key、base URL 或 API model 环境带进 app-server；这不是隐式 fallback，认证
    模式已在 runtime 构造时冻结。
    """

    normalized_auth_mode = normalize_codex_auth_mode(auth_mode)
    if normalized_auth_mode == CODEX_AUTH_MODE_LOCAL_LOGIN:
        required_names = {
            CODEX_AUTH_MODE_ENV,
            CODEX_AUTH_SOURCE_ENV,
            CODEX_ALLOW_PROVIDER_MIGRATION_ENV,
            CODEX_LOCAL_LOGIN_MODEL_ENV,
            CODEX_LOCAL_LOGIN_REASONING_EFFORT_ENV,
        }
    else:
        required_names = {
            active_provider.env_key,
            active_provider.base_url_env,
            active_provider.model_env,
            CODEX_AUTH_MODE_ENV,
            CODEX_ALLOW_PROVIDER_MIGRATION_ENV,
        }
        if active_provider.reasoning_effort_env is not None:
            required_names.add(active_provider.reasoning_effort_env)
    if minichem is not None:
        required_names.add(minichem.capability_token_env)
    if evidence_mcp is not None:
        required_names.update(evidence_mcp.env_http_headers.values())
    environment = {
        name: source_environment[name]
        for name in required_names
        if name in source_environment
    }
    if minichem is not None:
        environment.update(minichem.legacy_secret_environment())
    if evidence_mcp is not None:
        environment.update(evidence_mcp.legacy_secret_environment())
    return environment


def _collect_runtime_sensitive_values(
    environment: Mapping[str, str],
    active_provider: CodexProviderSettings,
    minichem: InvestigationMcpSettings | None,
    evidence_mcp: EvidenceMcpSettings | None,
) -> tuple[str, ...]:
    """提取当前 provider 和 MCP 的 secret 值，供请求与 trace 脱敏。"""

    names = {active_provider.env_key}
    if minichem is not None:
        names.add(minichem.capability_token_env)
    if evidence_mcp is not None:
        names.update(evidence_mcp.env_http_headers.values())
    return _normalize_sensitive_values(
        value for name, value in environment.items() if name in names
    )


def _resolve_sandbox_alias(
    sandbox_spec: CodexProcessSandboxSpec | None,
    codex_process_sandbox_spec: CodexProcessSandboxSpec | None,
) -> CodexProcessSandboxSpec | None:
    """合并 sandbox_spec 的兼容别名，并拒绝重复传入。"""

    if sandbox_spec is not None and codex_process_sandbox_spec is not None:
        raise TypeError("sandbox_spec was provided twice")
    resolved = sandbox_spec if sandbox_spec is not None else codex_process_sandbox_spec
    if resolved is not None and not isinstance(resolved, CodexProcessSandboxSpec):
        raise TypeError("sandbox_spec must be CodexProcessSandboxSpec or None")
    return resolved


def _callable_accepts_keyword(callback: Callable[..., Any], name: str) -> bool:
    """判断 adapter 是否声明指定扩展参数，无法检查时按不支持处理。"""

    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


class InvestigationCodexRuntime:
    """绑定单一 Investigation 或 task Review 角色的 Codex 运行时。

    调用方只需提交对应 prompt builder 返回的 request mapping 和 scope_id。runtime
    负责严格提取字段、确定会话归属、锁住完整 SDK turn、登记首个 thread、在
    ``turn/completed`` 返回后计数，并在计数后解析和验证 JSON。每个实例只能
    绑定一个角色及其 MCP capability；跨角色方法调用和混合 MCP 构造会失败关闭。
    ``allow_test_doubles`` 只允许 composition 的显式测试配置注入；公共 Controller
    不读取用户输入设置它。生产实例默认使用 gpt-5.6-sol/xhigh，但实际模型和
    推理强度由对应 provider 配置冻结，不把默认值当作模型名称准入。
    """

    _investigation_provider = CodexProviderSettings(
        provider_id="arche_investigation",
        env_key="ARCHE_CODEX_API_KEY",
        base_url_env="ARCHE_CODEX_BASE_URL",
        model_env="ARCHE_CODEX_MODEL_NAME",
        reasoning_effort_env="ARCHE_CODEX_REASONING_EFFORT",
        reasoning_effort_default=DEFAULT_INVESTIGATION_REASONING_EFFORT,
    )
    _review_provider = CodexProviderSettings(
        provider_id="arche_review",
        env_key="ARCHE_REVIEW_CODEX_API_KEY",
        base_url_env="ARCHE_REVIEW_CODEX_BASE_URL",
        model_env="ARCHE_REVIEW_CODEX_MODEL_NAME",
    )
    _local_login_provider = CodexProviderSettings(
        provider_id=CODEX_LOCAL_LOGIN_PROVIDER_ID,
        env_key=CODEX_LOCAL_LOGIN_ENV_KEY,
        base_url_env="ARCHE_CODEX_LOCAL_LOGIN_BASE_URL",
        model_env="ARCHE_CODEX_LOCAL_LOGIN_MODEL",
        reasoning_effort_env="ARCHE_CODEX_LOCAL_LOGIN_REASONING_EFFORT",
        reasoning_effort_default=CODEX_LOCAL_LOGIN_REASONING_EFFORT,
    )

    def __init__(
        self,
        *,
        session_root: Path | str,
        trace_root: Path | str,
        minichem: InvestigationMcpSettings | None = None,
        codex_bin: Path | str = DEFAULT_CODEX_BIN,
        evidence_mcp: EvidenceMcpSettings | None = None,
        investigation_provider: CodexProviderSettings | None = None,
        investigation_compact_prompt: str | None = None,
        investigation_model_context_window: int | None = None,
        review_provider: CodexProviderSettings | None = None,
        review_timeout_seconds: float = 900.0,
        sdk_idle_timeout_seconds: float | None = None,
        sdk_tool_idle_timeout_seconds: float = DEFAULT_MCP_TOOL_IDLE_TIMEOUT_SECONDS,
        environment: Mapping[str, str] | None = None,
        adapter: Callable[..., CodexSdkRunResult] | None = None,
        allow_test_doubles: bool = False,
        sandbox_spec: CodexProcessSandboxSpec | None = None,
        codex_process_sandbox_spec: CodexProcessSandboxSpec | None = None,
        role: str | None = None,
        before_new_investigation_turn: Callable[[Mapping[str, Any]], None] | None = None,
        on_turn_started: LegacyTurnStartedCallback | None = None,
    ) -> None:
        normalized_role = _validate_runtime_mcp_composition(role, minichem, evidence_mcp)
        if normalized_role != "investigation" and (
            investigation_compact_prompt is not None
            or investigation_model_context_window is not None
        ):
            raise HarnessError(
                "Investigation thread policy is only available for Investigation",
                kind="provider_envelope",
            )
        self.session_root = _prepare_secure_directory(Path(session_root), "session_root")
        self.trace_root = _prepare_secure_directory(Path(trace_root), "trace_root")
        self.codex_bin = str(Path(codex_bin).expanduser().resolve())
        self.role = normalized_role
        if not isinstance(allow_test_doubles, bool):
            raise TypeError("allow_test_doubles must be bool")
        self.allow_test_doubles = allow_test_doubles
        self.minichem = minichem
        self.evidence_mcp = evidence_mcp
        self.investigation_provider = investigation_provider or self._investigation_provider
        self.investigation_compact_prompt = (
            None
            if investigation_compact_prompt is None
            else _require_multiline_prompt(
                investigation_compact_prompt,
                "investigation_compact_prompt",
            )
        )
        self.investigation_model_context_window = (
            None
            if investigation_model_context_window is None
            else _require_positive_integer(
                investigation_model_context_window,
                "investigation_model_context_window",
            )
        )
        self.review_provider = review_provider or self._review_provider
        _validate_provider_settings(self.investigation_provider, self.review_provider)
        self.review_timeout_seconds = _require_positive_timeout(
            review_timeout_seconds,
            "review_timeout_seconds",
        )
        default_sdk_idle_timeout = (
            DEFAULT_INVESTIGATION_IDLE_TIMEOUT_SECONDS
            if normalized_role == "investigation"
            else DEFAULT_CODEX_IDLE_TIMEOUT_SECONDS
        )
        self.sdk_idle_timeout_seconds = _require_positive_timeout(
            default_sdk_idle_timeout
            if sdk_idle_timeout_seconds is None
            else sdk_idle_timeout_seconds,
            "sdk_idle_timeout_seconds",
        )
        self.sdk_tool_idle_timeout_seconds = _require_positive_timeout(
            sdk_tool_idle_timeout_seconds,
            "sdk_tool_idle_timeout_seconds",
        )
        source_environment = os.environ if environment is None else environment
        if not isinstance(source_environment, Mapping):
            raise TypeError("environment must be a mapping")
        requested_auth_mode = normalize_codex_auth_mode(
            source_environment.get(CODEX_AUTH_MODE_ENV)
        )
        # 本机登录只替换 Investigation provider。Task Review 继续使用
        # 原有 API/provider 配置，避免一个进程内的全局启动变量误改其它模型角色。
        self.auth_mode = (
            requested_auth_mode
            if normalized_role == "investigation"
            else CODEX_AUTH_MODE_API
        )
        self.auth_source = resolve_codex_auth_source(source_environment.get(CODEX_AUTH_SOURCE_ENV))
        self.allow_provider_migration = (
            parse_explicit_provider_migration_flag(
                source_environment.get(CODEX_ALLOW_PROVIDER_MIGRATION_ENV)
            )
            if normalized_role == "investigation"
            else False
        )
        active_provider = self._active_provider_for_role(normalized_role)
        self._environment = _build_runtime_environment(
            source_environment,
            active_provider,
            minichem,
            evidence_mcp,
            auth_mode=self.auth_mode,
        )
        self._sensitive_values = _collect_runtime_sensitive_values(
            self._environment,
            active_provider,
            minichem,
            evidence_mcp,
        )
        self._adapter = run_codex_sdk_turn if adapter is None else adapter
        if not callable(self._adapter):
            raise TypeError("adapter must be callable")
        if before_new_investigation_turn is not None and not callable(
            before_new_investigation_turn
        ):
            raise TypeError("before_new_investigation_turn must be callable or None")
        if on_turn_started is not None and not callable(on_turn_started):
            raise TypeError("on_turn_started must be callable or None")
        self._before_new_investigation_turn = before_new_investigation_turn
        # 该字段只作为旧调用方的不可变默认值；生产 Working Memory 路径每次
        # 通过 run_* 的同名参数注入独立 callback，不会跨并行 turn 共享它。
        self._legacy_on_turn_started = on_turn_started
        self.sandbox_spec = _resolve_sandbox_alias(
            sandbox_spec,
            codex_process_sandbox_spec,
        )

    @staticmethod
    def resolve_completed_result_receipt_handle(
        session_store: CodexSessionStore,
        *,
        owner: Mapping[str, Any] | CodexSessionOwner,
        thread_id: str,
        turn_id: str,
        request_digest: str | None,
        turn_count_before: int | None,
        thread_policy_digest: str | None,
        objective_digest: str | None = None,
    ) -> CodexCompletedTurnReceiptHandle | None:
        """从既有 SessionStore 恢复 Result handoff 的 receipt 句柄。

        输入是已由外层核对过 Result 和 sidecar digest 的固定 provider 身份；动作只
        委托 adapter 读取 canonical receipt 或返回跨进程已确认的 no-op 句柄，不创建
        receipt 目录、不修改 manifest、不启动 provider。三个 provider checkpoint
        字段可同时缺失，此时 adapter 从精确 thread/turn receipt 读取；若该 receipt 已
        确认删除则返回 None。manifest 没有保存 turn_id，因此 receipt 缺失时，
        Result/sidecar 对 turn_id 的证明仍由调用方负责。
        """

        if not isinstance(session_store, CodexSessionStore):
            raise HarnessError(
                "completed result receipt session store is invalid",
                kind="session_state",
            )
        receipt_root = session_store.manifest_path.parent / "completed_turn_receipts"
        goal_handle = resolve_codex_goal_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=session_store,
            thread_id=thread_id,
            turn_id=turn_id,
            request_digest=request_digest,
            objective_digest=objective_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
            handoff_kind="result",
        )
        if goal_handle is not None:
            return goal_handle
        return resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=session_store,
            thread_id=thread_id,
            turn_id=turn_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
            handoff_kind="result",
        )

    @staticmethod
    def resolve_waiting_receipt_handle(
        session_store: CodexSessionStore,
        *,
        owner: Mapping[str, Any] | CodexSessionOwner,
        thread_id: str,
        turn_id: str,
        request_digest: str,
        turn_count_before: int,
        thread_policy_digest: str,
        objective_digest: str | None = None,
    ) -> CodexCompletedTurnReceiptHandle:
        """只读恢复 sidecar→Working Memory 接力所需的 waiting receipt。

        输入必须是同一 Investigation session 的规范 SessionStore 和完整 provider
        checkpoint；动作只读取仍存在的 completed receipt 并返回延迟确认句柄，不创建
        目录、thread 或 receipt。waiting receipt 缺失时禁止使用 Result 的跨进程 no-op
        兼容，调用方必须保留 sidecar 并停止恢复。
        """

        if not isinstance(session_store, CodexSessionStore):
            raise HarnessError(
                "waiting receipt session store is invalid",
                kind="session_state",
            )
        receipt_root = session_store.manifest_path.parent / "completed_turn_receipts"
        goal_handle = resolve_codex_goal_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=session_store,
            thread_id=thread_id,
            turn_id=turn_id,
            request_digest=request_digest,
            objective_digest=objective_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
            handoff_kind="waiting",
        )
        if goal_handle is not None:
            return goal_handle
        return resolve_codex_completed_turn_receipt_handle(
            receipt_root,
            owner=owner,
            session_store=session_store,
            thread_id=thread_id,
            turn_id=turn_id,
            request_digest=request_digest,
            turn_count_before=turn_count_before,
            thread_policy_digest=thread_policy_digest,
            handoff_kind="waiting",
        )

    def _active_provider_for_role(self, role: str) -> CodexProviderSettings:
        """返回已验证角色对应的 provider 配置。"""

        if role == "investigation":
            return self.investigation_provider
        return self.review_provider

    @staticmethod
    def _validate_investigation_job_wake_envelope(
        envelope: Mapping[str, Any],
    ) -> None:
        """核对 wake 请求的原始作业身份和外部确认终态。

        输入是已经通过外层 request envelope 校验的 wake mapping；动作只读取 user
        JSON 与 metadata，拒绝字段漂移、非终态通知和缺少原始 identity 的唤醒。返回值
        为 None；任一不一致都会在调用 Codex 前抛出 provider_envelope，避免模型在错误
        job 上查询或因重复提交造成副作用。
        """

        metadata = envelope["metadata"]
        required_identity = ("job_id", "operation_id", "submission_key")
        for field_name in required_identity:
            _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
        family = _require_identifier(
            metadata.get("job_family"),
            "metadata.job_family",
        )
        if family not in {
            "transition_state_path",
            "native_execution",
            "action_execution",
            "programmable_analysis",
        }:
            raise HarnessError("wake request job family is unsupported", kind="provider_envelope")
        if family in {"transition_state_path", "action_execution"}:
            for field_name in ("action_id", "backend_id"):
                _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
            if family == "action_execution":
                for field_name in ("job_type", "runtime"):
                    _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
        else:
            for field_name in ("job_type", "runtime"):
                _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
        status = _require_text(
            metadata.get("terminal_status"),
            "metadata.terminal_status",
        ).lower()
        if status not in {"succeeded", "failed", "cancelled", "expired", "unknown"}:
            raise HarnessError(
                "wake request must carry an externally confirmed terminal status",
                kind="provider_envelope",
            )
        messages = envelope["messages"]
        if len(messages) < 2:
            raise HarnessError("wake request is missing its user payload", kind="provider_envelope")
        raw_content = messages[-1].get("content")
        if not isinstance(raw_content, str):
            raise HarnessError("wake request user payload must be JSON text", kind="provider_envelope")
        try:
            payload = json.loads(raw_content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise HarnessError("wake request user payload is not JSON", kind="provider_envelope") from exc
        if not isinstance(payload, Mapping):
            raise HarnessError("wake request user payload is invalid", kind="provider_envelope")
        if payload.get("job_id") != metadata.get("job_id"):
            raise HarnessError(
                "wake request job_id identity drifted",
                kind="provider_envelope",
            )
        if payload.get("terminal_status") != status:
            raise HarnessError(
                "wake request terminal status drifted",
                kind="provider_envelope",
            )

    @staticmethod
    def _validate_investigation_job_continuation_envelope(
        envelope: Mapping[str, Any],
    ) -> None:
        """核对任意时刻 continuation 请求的作业身份和 user payload。

        continuation 不携带 Controller 预先确认的终态；这里仅核对原 job、operation、
        submission、family 和当前 thread，确保模型只能沿 sidecar 指定的作业查询或
        collect。返回值为 None；字段漂移时在调用 provider 前 fail-closed。
        """

        metadata = envelope["metadata"]
        for field_name in ("job_id", "operation_id", "submission_key"):
            _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
        family = _require_identifier(metadata.get("job_family"), "metadata.job_family")
        if family not in {
            "transition_state_path",
            "native_execution",
            "action_execution",
            "programmable_analysis",
        }:
            raise HarnessError(
                "continuation request job family is unsupported",
                kind="provider_envelope",
            )
        if family in {"transition_state_path", "action_execution"}:
            for field_name in ("action_id", "backend_id"):
                _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
            if family == "action_execution":
                for field_name in ("job_type", "runtime"):
                    _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
        else:
            for field_name in ("job_type", "runtime"):
                _require_identifier(metadata.get(field_name), f"metadata.{field_name}")
        if "terminal_status" in metadata:
            raise HarnessError(
                "continuation request cannot carry a Controller terminal status",
                kind="provider_envelope",
            )
        messages = envelope["messages"]
        if len(messages) < 2:
            raise HarnessError(
                "continuation request is missing its user payload",
                kind="provider_envelope",
            )
        raw_content = messages[-1].get("content")
        if not isinstance(raw_content, str):
            raise HarnessError(
                "continuation request user payload must be JSON text",
                kind="provider_envelope",
            )
        try:
            payload = json.loads(raw_content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise HarnessError(
                "continuation request user payload is not JSON",
                kind="provider_envelope",
            ) from exc
        if not isinstance(payload, Mapping):
            raise HarnessError(
                "continuation request user payload is invalid",
                kind="provider_envelope",
            )
        if payload.get("job_id") != metadata.get("job_id"):
            raise HarnessError(
                "continuation request job_id identity drifted",
                kind="provider_envelope",
            )
        if "terminal_status" in payload:
            raise HarnessError(
                "continuation request user payload cannot carry terminal status",
                kind="provider_envelope",
            )

    def run_investigation_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        active_job: InvestigationJobState | Mapping[str, Any] | None = None,
        recover_existing_terminal_turn: bool = False,
        replay_turn_id: str | None = None,
        on_thread_ready: ThreadReadyCallback | None = None,
        on_thread_migration: ThreadMigrationCallback | None = None,
        on_turn_started: TurnStartedCallback | None = None,
        on_terminal_turn: TerminalTurnCallback | None = None,
        on_job_accepted: JobAcceptedCallback | None = None,
        on_job_collected: JobCollectedCallback | None = None,
    ) -> InvestigationTurnResult:
        """运行首次或返修 Investigation turn，并复用执行输入唯一的 Codex thread。

        输入必须是 Investigation prompt builder 返回的 initial、repair、旧 wake 或
        continuation mapping。首个 thread 在真正 turn 前写入 manifest；adapter 返回
        completed turn 后先增加计数，再解析 InvestigationResponse。失败会抛出
        HarnessError，且不会返回部分结果。
        """

        self._require_role("investigation")
        envelope = self._extract_request_envelope(request, expected_profile="investigation")
        metadata = envelope["metadata"]
        request_kind = self._require_request_kind(metadata, _ALLOWED_REQUEST_KINDS)
        research_task_id = _require_identifier(
            metadata.get("research_task_id"),
            "metadata.research_task_id",
        )
        investigation_context_id = _require_identifier(
            metadata.get("investigation_context_id"),
            "metadata.investigation_context_id",
        )
        hypothesis_execution_input_record_id = _require_identifier(
            metadata.get("hypothesis_execution_input_record_id"),
            "metadata.hypothesis_execution_input_record_id",
        )
        metadata_thread_id = metadata.get("codex_thread_id")
        if request_kind in {
            "deepseek_repair",
            "investigation_job_wake",
            "investigation_job_continuation",
        }:
            if envelope["thread_id"] is None:
                raise HarnessError(
                    "repair/wake request must carry thread_id",
                    kind="provider_envelope",
                )
            metadata_thread_id = _require_identifier(
                metadata_thread_id,
                "metadata.codex_thread_id",
            )
            if metadata_thread_id != envelope["thread_id"]:
                raise HarnessError(
                    "repair/wake request thread identities do not match",
                    kind="provider_envelope",
                )
        if request_kind == "investigation_job_wake":
            self._validate_investigation_job_wake_envelope(envelope)
        elif request_kind == "investigation_job_continuation":
            self._validate_investigation_job_continuation_envelope(envelope)
        if not isinstance(recover_existing_terminal_turn, bool):
            raise TypeError("recover_existing_terminal_turn must be bool")
        if replay_turn_id is not None:
            replay_turn_id = _require_identifier(replay_turn_id, "replay_turn_id")
            if active_job is not None:
                raise HarnessError(
                    "explicit completed turn replay cannot carry an active job",
                    kind="session_state",
                )
        if recover_existing_terminal_turn and request_kind != "initial_investigation":
            raise HarnessError(
                "existing terminal turn recovery is only valid for initial Investigation turns",
                kind="session_state",
            )
        if recover_existing_terminal_turn and active_job is None:
            raise HarnessError(
                "existing terminal turn recovery requires an active job",
                kind="session_state",
            )
        if request_kind == "investigation_job_continuation" and active_job is None:
            raise HarnessError(
                "investigation job continuation requires an active job",
                kind="session_state",
            )
        if active_job is not None:
            if request_kind not in {
                "initial_investigation",
                "investigation_job_wake",
                "investigation_job_continuation",
            }:
                raise HarnessError(
                    "active job recovery is only valid for Investigation turns",
                    kind="session_state",
                )
            active_job = (
                active_job
                if isinstance(active_job, InvestigationJobState)
                else InvestigationJobState.from_mapping(active_job)
            )
            if (
                # InvestigationJobState 是 MiniChem 作业状态 adapter；在这一个
                # 边界把它的 schema-v10 身份映射到当前公开身份。
                active_job.task_id != research_task_id
                or active_job.batch_id != investigation_context_id
                or active_job.combination_id
                != hypothesis_execution_input_record_id
                or active_job.scope_id != scope_id
            ):
                raise HarnessError(
                    "active job recovery identity does not match Investigation request",
                    kind="session_state",
                )
            if envelope["thread_id"] != active_job.codex_thread_id and (
                request_kind not in {
                    "investigation_job_wake",
                    "investigation_job_continuation",
                }
                or not self.allow_provider_migration
            ):
                raise HarnessError(
                    "active job recovery thread does not match Investigation request",
                    kind="session_state",
                )
            if request_kind in {
                "investigation_job_wake",
                "investigation_job_continuation",
            }:
                request_identity = _build_request_job_identity(metadata)
                if request_identity.as_mapping() != active_job.identity.as_mapping():
                    raise HarnessError(
                        "continuation request active job identity does not match waiting state",
                        kind="session_state",
                    )
        provider = self._resolve_provider(self.investigation_provider, envelope["model"])
        scope = _require_identifier(scope_id, "scope_id")
        owner = CodexSessionOwner.investigation(
            research_task_id,
            investigation_context_id,
            hypothesis_execution_input_record_id,
        )
        store = CodexSessionStore.from_root(
            self.session_root,
            owner=owner,
            scope_id=scope,
            model=provider.model,
        )
        return self._run_validated_turn(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=self._investigation_mcp_config(),
            validator=validate_investigation_response,
            expected_result_type=InvestigationResponse,
            timeout_seconds=None,
            stage=request_kind,
            require_existing=request_kind
            in {
                "deepseek_repair",
                "investigation_job_wake",
                "investigation_job_continuation",
            },
            before_new_turn=(
                None
                if request_kind == "investigation_job_wake" or active_job is not None
                else self._build_before_new_investigation_turn(envelope)
            ),
            on_thread_ready=self._resolve_thread_ready_callback(on_thread_ready),
            on_thread_migration=on_thread_migration,
            on_turn_started=self._resolve_turn_started_callback(on_turn_started),
            on_terminal_turn=self._resolve_terminal_turn_callback(on_terminal_turn),
            on_job_accepted=on_job_accepted,
            on_job_collected=on_job_collected,
            active_job=active_job,
            recover_existing_terminal_turn=recover_existing_terminal_turn,
            replay_turn_id=replay_turn_id,
        )

    def replay_completed_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        replay_turn_id: str,
        on_thread_ready: ThreadReadyCallback | None = None,
        on_turn_started: TurnStartedCallback | None = None,
        on_job_accepted: JobAcceptedCallback | None = None,
        on_job_collected: JobCollectedCallback | None = None,
    ) -> InvestigationTurnResult:
        """只读重放一个已有 completed receipt，不创建新的 provider turn。"""

        return self.run_investigation_turn(
            request,
            scope_id=scope_id,
            replay_turn_id=replay_turn_id,
            on_thread_ready=on_thread_ready,
            on_turn_started=on_turn_started,
            on_job_accepted=on_job_accepted,
            on_job_collected=on_job_collected,
        )

    def recover_interrupted_investigation_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        active_job: InvestigationJobState | Mapping[str, Any],
        on_thread_ready: ThreadReadyCallback | None = None,
        on_turn_started: TurnStartedCallback | None = None,
        on_terminal_turn: TerminalTurnCallback | None = None,
    ) -> None:
        """在同一 Codex session 中只确认旧 interrupted turn，不启动新模型 turn。

        输入是 created 或 repair_pending session 的稳定身份请求和保存了旧 turn/job
        身份的 waiting sidecar；动作严格复用原 thread、provider、workspace、model
        与 policy，并沿用磁盘 request-intent 的原请求摘要，让 adapter 读取完整
        history、journal-first 地确认唯一失败/中断 turn，再同步调用终态事实回调。该
        阶段不轮询、提交或收集 MiniChem 作业，稳定的旧终态异常会被消费，供同一次
        控制器调用随后启动独立 continuation turn；恢复证据、身份漂移或回调失败时
        抛出错误并保持 fail-closed。
        """

        self._require_role("investigation")
        if not callable(on_turn_started) or not callable(on_terminal_turn):
            raise HarnessError(
                "Codex recovery-only requires turn lifecycle callbacks",
                kind="session_protocol",
            )
        envelope = self._extract_request_envelope(
            request,
            expected_profile="investigation",
        )
        metadata = envelope["metadata"]
        request_kind = self._require_request_kind(metadata, _ALLOWED_REQUEST_KINDS)
        if request_kind not in {
            "initial_investigation",
            "deepseek_repair",
            "investigation_job_continuation",
        }:
            raise HarnessError(
                "Codex recovery-only requires an original Investigation request",
                kind="session_state",
            )
        thread_id = _require_identifier(envelope.get("thread_id"), "thread_id")
        metadata_thread_id = _require_identifier(
            metadata.get("codex_thread_id"),
            "metadata.codex_thread_id",
        )
        if metadata_thread_id != thread_id:
            raise HarnessError(
                "Codex recovery-only request thread identities do not match",
                kind="session_state",
            )
        try:
            waiting = (
                active_job
                if isinstance(active_job, InvestigationJobState)
                else InvestigationJobState.from_mapping(active_job)
            )
        except (TypeError, ValueError) as exc:
            raise HarnessError(
                "Codex recovery-only waiting job identity is invalid",
                kind="session_state",
            ) from exc
        research_task_id = _require_identifier(
            metadata.get("research_task_id"),
            "metadata.research_task_id",
        )
        investigation_context_id = _require_identifier(
            metadata.get("investigation_context_id"),
            "metadata.investigation_context_id",
        )
        hypothesis_execution_input_record_id = _require_identifier(
            metadata.get("hypothesis_execution_input_record_id"),
            "metadata.hypothesis_execution_input_record_id",
        )
        normalized_scope = _require_identifier(scope_id, "scope_id")
        if (
            waiting.task_id != research_task_id
            or waiting.batch_id != investigation_context_id
            or waiting.combination_id != hypothesis_execution_input_record_id
            or waiting.scope_id != normalized_scope
            or waiting.codex_thread_id != thread_id
        ):
            raise HarnessError(
                "Codex recovery-only waiting identity does not match the original request",
                kind="session_state",
            )
        provider = self._resolve_provider(self.investigation_provider, envelope["model"])
        store = CodexSessionStore.from_root(
            self.session_root,
            owner=CodexSessionOwner.investigation(
                research_task_id,
                investigation_context_id,
                hypothesis_execution_input_record_id,
            ),
            scope_id=normalized_scope,
            model=provider.model,
        )
        try:
            self._run_validated_turn(
                envelope=envelope,
                store=store,
                provider=provider,
                mcp_config=self._investigation_mcp_config(),
                validator=validate_investigation_response,
                expected_result_type=InvestigationResponse,
                timeout_seconds=None,
                stage=request_kind,
                require_existing=True,
                before_new_turn=None,
                on_thread_ready=on_thread_ready,
                on_thread_migration=None,
                on_turn_started=on_turn_started,
                on_terminal_turn=on_terminal_turn,
                on_job_accepted=None,
                on_job_collected=None,
                active_job=waiting,
                recovery_only=True,
            )
        except CodexTerminalTurnError as terminal_error:
            if (
                terminal_error.turn_id != waiting.turn_id
                or terminal_error.status not in {"failed", "interrupted"}
            ):
                raise HarnessError(
                    "Codex recovery-only terminal identity does not match waiting turn",
                    kind="session_state",
                ) from None
            return
        raise HarnessError(
            "Codex recovery-only adapter returned without terminal acknowledgement",
            kind="session_protocol",
        )

    def _has_completed_receipt_matching_waiting_turn(
        self,
        manifest: Mapping[str, Any],
        waiting: InvestigationJobState,
        *,
        thread_id: str,
        thread_policy_digest: str,
        receipt_root: Path,
        recovery: Mapping[str, Any] | None = None,
    ) -> bool:
        """确认成功 manifest 与 waiting turn 的 completed receipt 完全一致。

        输入是已经通过会话归属核对的 manifest、等待 sidecar、thread/policy 身份和
        回执目录；动作只读取已完成 turn 回执，不启动 provider、不访问 Controller 或
        MiniChem。只有 manifest 为 completed、thread 相同、计数为正且 waiting turn
        的回执正好对应 ``turn_count - 1`` 时返回 True；任一字段不匹配都返回 False，
        让缺少 request-intent 的其他情况继续 fail-closed。
        """

        return self._has_completed_receipt_matching_turn(
            manifest,
            waiting.turn_id,
            thread_id=thread_id,
            thread_policy_digest=thread_policy_digest,
            receipt_root=receipt_root,
            recovery=recovery,
        )

    @staticmethod
    def _has_completed_receipt_matching_turn(
        manifest: Mapping[str, Any],
        turn_id: str,
        *,
        thread_id: str,
        thread_policy_digest: str,
        receipt_root: Path,
        recovery: Mapping[str, Any] | None = None,
    ) -> bool:
        """核对一个 turn 是当前 manifest 最后一个可重放的 completed receipt。"""

        manifest_status = manifest.get("status")
        if manifest_status == "recovery_pending":
            if not (
                isinstance(recovery, Mapping)
                and recovery.get("state") == "recovery_pending"
                and recovery.get("reason") == "request_intent"
                and recovery.get("action")
                == "request_intent_confirmation_required"
                and recovery.get("thread_policy_digest") == thread_policy_digest
                and recovery.get("turn_count") == manifest.get("turn_count")
            ):
                return False
        elif manifest_status != "completed":
            return False
        if manifest.get("thread_id") != thread_id:
            return False
        turn_count = manifest.get("turn_count")
        if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 1:
            return False
        receipt = read_codex_completed_turn_receipt(
            receipt_root,
            thread_id=thread_id,
            turn_id=turn_id,
            thread_policy_digest=thread_policy_digest,
        )
        if receipt is None:
            return False
        return receipt["turn_count_before"] == turn_count - 1

    def has_recoverable_interrupted_investigation_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        active_job: InvestigationJobState | Mapping[str, Any],
    ) -> bool:
        """只读确认当前 waiting sidecar 可进入旧 interrupted turn 收束。

        输入是 created 或 repair_pending session 的稳定身份请求和当前 waiting 作业快照；动作只核对
        同一 research task、Investigation context、execution input、scope、thread、
        request-intent、manifest 和
        recovery identity，不访问 provider、Controller、MiniChem 或模型。原 prompt
        摘要通常取自持久 request-intent，因为本分支不重发 prompt；若 request-intent
        已按成功路径删除，则只有 completed manifest 与 waiting turn 的 completed
        receipt 在 policy、thread、turn 和计数上完全一致时才返回 False，让调用方进入
        普通 continuation replay。其它 request-intent、policy、作业或 session 事实
        损坏时抛出 HarnessError，避免调用方重复 turn。
        """

        self._require_role("investigation")
        envelope = self._extract_request_envelope(
            request,
            expected_profile="investigation",
        )
        metadata = envelope["metadata"]
        request_kind = self._require_request_kind(metadata, _ALLOWED_REQUEST_KINDS)
        if request_kind not in {
            "initial_investigation",
            "deepseek_repair",
            "investigation_job_continuation",
        }:
            return False
        thread_id = _require_identifier(envelope.get("thread_id"), "thread_id")
        if _require_identifier(metadata.get("codex_thread_id"), "metadata.codex_thread_id") != thread_id:
            raise HarnessError(
                "Codex recovery-only request thread identities do not match",
                kind="session_state",
            )
        try:
            waiting = (
                active_job
                if isinstance(active_job, InvestigationJobState)
                else InvestigationJobState.from_mapping(active_job)
            )
        except (TypeError, ValueError) as exc:
            raise HarnessError(
                "Codex recovery-only waiting job identity is invalid",
                kind="session_state",
            ) from exc
        research_task_id = _require_identifier(
            metadata.get("research_task_id"),
            "metadata.research_task_id",
        )
        investigation_context_id = _require_identifier(
            metadata.get("investigation_context_id"),
            "metadata.investigation_context_id",
        )
        hypothesis_execution_input_record_id = _require_identifier(
            metadata.get("hypothesis_execution_input_record_id"),
            "metadata.hypothesis_execution_input_record_id",
        )
        normalized_scope = _require_identifier(scope_id, "scope_id")
        if (
            waiting.status != "waiting_investigation_job"
            or waiting.task_id != research_task_id
            or waiting.batch_id != investigation_context_id
            or waiting.combination_id != hypothesis_execution_input_record_id
            or waiting.scope_id != normalized_scope
            or waiting.codex_thread_id != thread_id
        ):
            raise HarnessError(
                "Codex recovery-only waiting identity does not match the original request",
                kind="session_state",
            )
        provider = self._resolve_provider(self.investigation_provider, envelope["model"])
        store = CodexSessionStore.from_root(
            self.session_root,
            owner=CodexSessionOwner.investigation(
                research_task_id,
                investigation_context_id,
                hypothesis_execution_input_record_id,
            ),
            scope_id=normalized_scope,
            model=provider.model,
        )
        (
            harness_request,
            _rebuilt_request_digest,
            _secret_environment,
            _thread_config,
            policy_digest,
        ) = self._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=self._investigation_mcp_config(),
            timeout_seconds=None,
        )
        manifest = store.load()
        recovery = store.load_recovery()
        poison = store.load_recovery_poison()
        if manifest is None:
            raise HarnessError(
                "Codex recovery-only session state is incomplete",
                kind="session_state",
            )
        if poison is not None:
            raise HarnessError(
                "Codex recovery poison marker blocks recovery-only",
                kind="recovery_pending",
            )
        if manifest.get("thread_policy_digest") != policy_digest:
            raise HarnessError(
                "Codex recovery-only manifest policy does not match current config",
                kind="session_state",
            )
        receipt_root = store.manifest_path.parent / "completed_turn_receipts"
        if self._has_completed_receipt_matching_waiting_turn(
            manifest,
            waiting,
            thread_id=thread_id,
            thread_policy_digest=policy_digest,
            receipt_root=receipt_root,
            recovery=recovery,
        ):
            # 成功 continuation 已经推进 manifest；即使后来一次失败 Resume 又留下
            # request-intent/recovery，也必须交给普通 completed receipt replay，
            # 不能拿后一次失败 turn 冒充 waiting turn 的中断恢复。
            return False
        intent = read_codex_request_intent(receipt_root)
        if intent is None:
            raise HarnessError(
                "Codex recovery-only request-intent is missing",
                kind="session_state",
            )
        # recovery-only 不会把当前重建的 prompt 再发给 provider。旧 wake/initial
        # prompt 可能在版本升级后无法逐字重建，因此终态确认必须沿用磁盘 intent
        # 记录的原摘要；其它稳定身份和唯一物理 history 仍分别严格核对。
        recovery_request_digest = intent["request_digest"]
        self._assert_recovery_only_available(
            manifest,
            recovery,
            request_digest=recovery_request_digest,
            thread_policy_digest=policy_digest,
            stage=request_kind,
        )
        if (
            manifest.get("thread_id") != thread_id
            or (
                recovery is not None
                and recovery.get("turn_count") != manifest.get("turn_count")
            )
        ):
            raise HarnessError(
                "Codex recovery-only session thread/count identity does not match",
                kind="session_state",
            )
        expected_intent = {
            "thread_id": thread_id,
            "turn_count_before": manifest.get("turn_count"),
            "provider_id": provider.settings.provider_id,
            "env_key": provider.settings.env_key,
            "model": provider.model,
            "role": envelope["role"],
            "stage": harness_request.stage,
            "profile": envelope["profile"],
            "workspace": str(store.workspace),
            "thread_policy_digest": policy_digest,
        }
        if any(intent.get(field) != value for field, value in expected_intent.items()):
            raise HarnessError(
                "Codex recovery-only request-intent identity does not match",
                kind="session_state",
            )
        return True

    def has_durable_investigation_rollback(
        self,
        *,
        research_task_id: str,
        investigation_context_id: str,
        hypothesis_execution_input_record_id: str,
        scope_id: str,
        stage: str = "initial_investigation",
    ) -> bool:
        """确认指定 Investigation session 已完成白名单错误的 thread rollback。

        输入是完整 session identity；动作只读取同一 session 的 manifest 与 rollback
        receipt，不启动模型、不查询或修改 MiniChem 作业。返回 True 仅表示 receipt、
        thread 和成功 turn 计数彼此一致；缺少记录或身份不一致时返回 False，交由上层
        保守地结束本次请求。
        """

        self._require_role("investigation")
        normalized_stage = _require_text(stage, "rollback stage")
        if normalized_stage not in _ROLLBACK_ELIGIBLE_INVESTIGATION_STAGES:
            return False
        owner = CodexSessionOwner.investigation(
            _require_identifier(research_task_id, "research_task_id"),
            _require_identifier(
                investigation_context_id,
                "investigation_context_id",
            ),
            _require_identifier(
                hypothesis_execution_input_record_id,
                "hypothesis_execution_input_record_id",
            ),
        )
        configured_model = _require_text(
            self._environment.get(self.investigation_provider.model_env),
            self.investigation_provider.model_env,
        )
        provider = self._resolve_provider(self.investigation_provider, configured_model)
        store = CodexSessionStore.from_root(
            self.session_root,
            owner=owner,
            scope_id=_require_identifier(scope_id, "scope_id"),
            model=provider.model,
        )
        receipt = store.load_thread_rollback_receipt()
        manifest = store.load()
        if receipt is None or manifest is None:
            return False
        return bool(
            receipt.get("stage") == normalized_stage
            and receipt.get("provider_error_code") == "invalid_encrypted_content"
            and receipt.get("turn_count") == 0
            and manifest.get("turn_count") == 0
            and manifest.get("thread_id") == receipt.get("thread_id")
        )

    def has_durable_investigation_wake_completion(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        thread_id: str,
        turn_id: str,
    ) -> bool:
        """只读确认 wake 的 completed receipt，阻止恢复时盲目新开模型 turn。

        输入是 sidecar 保存的原 wake request 及其 thread/turn 身份；动作按与真实
        ``run_investigation_turn`` 相同的 request/policy 摘要定位本地 session，并
        严格核对 completed receipt、manifest 成功计数和 thread。返回 True 只代表
        provider turn 已完成且可由 history replay；缺失、漂移或本地状态不完整时返回
        False，不访问 Codex provider、MiniChem 或任意异步作业。
        """

        self._require_role("investigation")
        envelope = self._extract_request_envelope(
            request,
            expected_profile="investigation",
        )
        metadata = envelope["metadata"]
        if self._require_request_kind(metadata, _ALLOWED_REQUEST_KINDS) != "investigation_job_wake":
            return False
        normalized_scope = _require_identifier(scope_id, "scope_id")
        normalized_thread = _require_identifier(thread_id, "thread_id")
        normalized_turn = _require_identifier(turn_id, "turn_id")
        if envelope.get("thread_id") != normalized_thread:
            return False
        provider = self._resolve_provider(self.investigation_provider, envelope["model"])
        owner = CodexSessionOwner.investigation(
            _require_identifier(
                metadata.get("research_task_id"),
                "metadata.research_task_id",
            ),
            _require_identifier(
                metadata.get("investigation_context_id"),
                "metadata.investigation_context_id",
            ),
            _require_identifier(
                metadata.get("hypothesis_execution_input_record_id"),
                "metadata.hypothesis_execution_input_record_id",
            ),
        )
        store = CodexSessionStore.from_root(
            self.session_root,
            owner=owner,
            scope_id=normalized_scope,
            model=provider.model,
        )
        _request, request_digest, _secret_environment, _thread_config, policy_digest = (
            self._build_codex_turn_policy_inputs(
                envelope=envelope,
                store=store,
                provider=provider,
                mcp_config=self._investigation_mcp_config(),
                timeout_seconds=None,
            )
        )
        manifest = store.load()
        if not isinstance(manifest, Mapping) or manifest.get("thread_id") != normalized_thread:
            return False
        try:
            turn_count = manifest["turn_count"]
            if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 1:
                return False
            receipt = find_codex_completed_turn_receipt(
                store.manifest_path.parent / "completed_turn_receipts",
                thread_id=normalized_thread,
                request_digest=request_digest,
                thread_policy_digest=policy_digest,
            )
            if receipt is None or receipt.get("turn_id") != normalized_turn:
                return False
            validate_codex_completed_turn_receipt(
                store.manifest_path.parent / "completed_turn_receipts",
                thread_id=normalized_thread,
                turn_id=normalized_turn,
                request_digest=request_digest,
                turn_count_before=receipt["turn_count_before"],
                thread_policy_digest=policy_digest,
            )
            return turn_count == receipt["turn_count_before"] + 1
        except (KeyError, TypeError, ValueError, HarnessError):
            return False

    def has_recoverable_initial_investigation_failure(
        self,
        *,
        research_task_id: str,
        investigation_context_id: str,
        hypothesis_execution_input_record_id: str,
        scope_id: str,
        thread_id: str,
        turn_id: str,
    ) -> bool:
        """确认补丁前留下的初始 Investigation 失败 turn 可以安全恢复。

        输入是等待 sidecar 提供的完整 research task、Investigation context、
        execution input、thread 和失败 turn 身份；动作只
        读取同一会话 manifest、rollback receipt/intent 和已经刷盘的初始 trace。只有
        manifest 仍为零个成功 turn，且 trace 明确记录同一 turn 的白名单
        ``invalid_encrypted_content`` 失败（或已有同身份 rollback receipt）时返回
        True。该检查不启动模型、不调用 MCP，也不修改作业状态。
        """

        self._require_role("investigation")
        owner = CodexSessionOwner.investigation(
            _require_identifier(research_task_id, "research_task_id"),
            _require_identifier(
                investigation_context_id,
                "investigation_context_id",
            ),
            _require_identifier(
                hypothesis_execution_input_record_id,
                "hypothesis_execution_input_record_id",
            ),
        )
        normalized_scope = _require_identifier(scope_id, "scope_id")
        normalized_thread = _require_identifier(thread_id, "thread_id")
        normalized_turn = _require_identifier(turn_id, "turn_id")
        configured_model = _require_text(
            self._environment.get(self.investigation_provider.model_env),
            self.investigation_provider.model_env,
        )
        provider = self._resolve_provider(self.investigation_provider, configured_model)
        store = CodexSessionStore.from_root(
            self.session_root,
            owner=owner,
            scope_id=normalized_scope,
            model=provider.model,
        )
        manifest = store.load()
        if manifest is None or not (
            manifest.get("last_stage") == "initial_investigation"
            and manifest.get("turn_count") == 0
            and manifest.get("thread_id") == normalized_thread
        ):
            return False
        receipt = store.load_thread_rollback_receipt()
        if receipt is not None:
            return bool(
                receipt.get("stage") == "initial_investigation"
                and receipt.get("thread_id") == normalized_thread
                and receipt.get("turn_count") == 0
                and receipt.get("provider_error_code") == "invalid_encrypted_content"
            )
        error_code = self._read_legacy_initial_failure_code(
            owner=owner,
            thread_id=normalized_thread,
            turn_id=normalized_turn,
        )
        return error_code == "invalid_encrypted_content"

    def _read_legacy_initial_failure_code(
        self,
        *,
        owner: CodexSessionOwner,
        thread_id: str,
        turn_id: str,
    ) -> str | None:
        """从同一执行输入的旧初始 trace 读取唯一白名单失败代码。

        输入是已经校验的 Investigation owner 和等待 turn 身份；动作只解析
        ``initial_investigation.0`` 的 JSONL 终态通知，拒绝跨 thread、成功或未知
        错误事件。返回固定 provider code 或 None，不把 trace 原文带入异常或模型。
        """

        terminal_events: list[Mapping[str, Any]] = []
        relative_directory = Path(
            "investigation"
        ) / owner.research_task_id / (owner.investigation_context_id or "") / (
            owner.hypothesis_execution_input_record_id or ""
        )
        # schema-v10 composition 会把执行输入目录作为 trace_root 传入；独立 runtime
        # 和测试则把 investigation 根目录传入。两种旧布局都必须绑定到同一 owner，
        # 不能为了恢复而递归搜索整个 Run Bundle。
        directories = (
            self.trace_root / relative_directory,
            self.trace_root
            / (owner.hypothesis_execution_input_record_id or "")
            / relative_directory,
        )
        for directory in dict.fromkeys(directories):
            if directory.is_symlink():
                raise HarnessError("legacy Investigation trace directory must not be a symlink", kind="session_state")
            if not directory.exists():
                continue
            if not directory.is_dir():
                raise HarnessError("legacy Investigation trace directory is not a directory", kind="session_state")
            try:
                candidates = sorted(directory.glob("initial_investigation.0*.jsonl"))
            except OSError as exc:
                raise HarnessError("legacy Investigation trace directory cannot be listed", kind="session_state") from exc
            for candidate in candidates:
                if candidate.is_symlink():
                    raise HarnessError("legacy Investigation trace must not be a symlink", kind="session_state")
                if not candidate.is_file():
                    raise HarnessError("legacy Investigation trace is not a regular file", kind="session_state")
                try:
                    lines = candidate.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError) as exc:
                    raise HarnessError("legacy Investigation trace cannot be read", kind="session_state") from exc
                for line in lines:
                    try:
                        event = json.loads(line)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise HarnessError("legacy Investigation trace contains invalid JSON", kind="session_state") from exc
                    if not isinstance(event, Mapping) or event.get("method") != "turn/completed":
                        continue
                    params = event.get("params")
                    if not isinstance(params, Mapping):
                        raise HarnessError("legacy Investigation terminal trace has invalid params", kind="session_state")
                    terminal_events.append(params)
        if not terminal_events:
            return None
        if any(
            event.get("thread_id") != thread_id
            or event.get("status") != "failed"
            or event.get("error_code") != "invalid_encrypted_content"
            for event in terminal_events
        ):
            return None
        if not any(event.get("turn_id") == turn_id for event in terminal_events):
            return None
        return "invalid_encrypted_content"

    def run_task_review_turn(
        self,
        request: Mapping[str, Any],
        *,
        scope_id: str,
        on_thread_ready: ThreadReadyCallback | None = None,
        on_turn_started: TurnStartedCallback | None = None,
        on_terminal_turn: TerminalTurnCallback | None = None,
    ) -> InvestigationTurnResult:
        """运行任务级 Review，并跨批次复用 task 唯一的 deny-all thread。"""

        self._require_role("task_review")
        envelope = self._extract_request_envelope(request, expected_profile="review")
        metadata = envelope["metadata"]
        request_kind = self._require_request_kind(
            metadata,
            {"task_review", "task_finalization"},
        )
        research_task_id = _require_identifier(
            metadata.get("research_task_id"),
            "metadata.research_task_id",
        )
        metadata_thread_id = metadata.get("review_thread_id")
        if metadata_thread_id is not None:
            metadata_thread_id = _require_identifier(
                metadata_thread_id,
                "metadata.review_thread_id",
            )
            if metadata_thread_id != envelope["thread_id"]:
                raise HarnessError(
                    "review request thread identities do not match",
                    kind="provider_envelope",
                )
        provider = self._resolve_provider(self.review_provider, envelope["model"])
        scope = _require_identifier(scope_id, "scope_id")
        owner = CodexSessionOwner.task_review(research_task_id)
        store = CodexSessionStore.from_root(
            self.session_root,
            owner=owner,
            scope_id=scope,
            model=provider.model,
        )
        validator = (
            validate_task_finalization_response
            if request_kind == "task_finalization"
            else validate_task_review_response
        )
        expected_result_type = (
            TaskFinalizationResponse
            if request_kind == "task_finalization"
            else TaskReviewResponse
        )
        return self._run_validated_turn(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=self._review_mcp_config(),
            validator=validator,
            expected_result_type=expected_result_type,
            timeout_seconds=self.review_timeout_seconds,
            stage=request_kind,
            # 首次 Review 没有 thread identity 时允许创建；一旦请求或本地
            # manifest 已带身份，_resolve_turn_session 会拒绝替代会话。
            require_existing=envelope.get("thread_id") is not None,
            before_new_turn=None,
            on_thread_ready=self._resolve_thread_ready_callback(on_thread_ready),
            on_thread_migration=None,
            on_turn_started=self._resolve_turn_started_callback(on_turn_started),
            on_terminal_turn=self._resolve_terminal_turn_callback(on_terminal_turn),
            on_job_accepted=None,
            on_job_collected=None,
            active_job=None,
        )

    def _resolve_thread_ready_callback(
        self,
        callback: ThreadReadyCallback | None,
    ) -> ThreadReadyCallback | None:
        """选择本次真实 thread callback，不跨 turn 保存生命周期状态。"""

        if callback is not None:
            if not callable(callback):
                raise TypeError("on_thread_ready must be callable or None")
            return callback
        return None

    def _resolve_turn_started_callback(
        self,
        callback: TurnStartedCallback | None,
    ) -> TurnStartedCallback | None:
        """选择本次 turn 的 callback，并把旧一参数回调隔离成兼容适配器。

        ``callback`` 属于当前调用栈，只在当前 ``_run_validated_turn`` 内流转。
        生产调用必须传入它；构造器上的旧回调只为历史测试和外部调用方保留，不能
        改变后续 turn 的 callback，也不会被 Working Memory 生命周期路径使用。
        """

        if callback is not None:
            if not callable(callback):
                raise TypeError("on_turn_started must be callable or None")
            return callback
        legacy_callback = self._legacy_on_turn_started
        if legacy_callback is None:
            return None

        def notify_legacy_callback(_thread_id: str, turn_id: str) -> None:
            """把旧接口限制在当前 turn，避免旧调用方看到 thread 合同。"""

            legacy_callback(turn_id)

        return notify_legacy_callback

    @staticmethod
    def _resolve_terminal_turn_callback(
        callback: TerminalTurnCallback | None,
    ) -> TerminalTurnCallback | None:
        """校验当前调用的终态回调；缺失时由 adapter 保持恢复阻断。"""

        if callback is not None and not callable(callback):
            raise TypeError("on_terminal_turn must be callable or None")
        return callback

    def _require_role(self, expected_role: str) -> None:
        """拒绝将一个角色 runtime 当作另一个角色调用。"""

        if self.role != expected_role:
            raise HarnessError(
                f"runtime role {self.role} cannot call {expected_role} turn",
                kind="provider_envelope",
            )

    def _build_before_new_investigation_turn(
        self,
        envelope: Mapping[str, Any],
    ) -> Callable[[], None] | None:
        """把带 envelope 的副作用栅栏适配成 SDK 所需的无参回调。"""

        if self._before_new_investigation_turn is None:
            return None
        return partial(self._before_new_investigation_turn, envelope)

    def _extract_request_envelope(
        self,
        request: Mapping[str, Any],
        *,
        expected_profile: str,
    ) -> dict[str, Any]:
        """严格提取 builder mapping 的 model、messages、schema、metadata 和 thread。"""

        envelope = _copy_json_mapping(request, "request")
        required = {"model", "messages", "response_schema", "metadata"}
        missing = required.difference(envelope)
        if missing:
            raise HarnessError(
                f"request is missing fields: {sorted(missing)}",
                kind="provider_envelope",
            )
        model = _require_text(envelope["model"], "request.model")
        messages = _copy_message_sequence(envelope["messages"])
        schema = _copy_json_mapping(envelope["response_schema"], "request.response_schema")
        metadata = _copy_json_mapping(envelope["metadata"], "request.metadata")
        profile = _require_text(envelope.get("profile"), "request.profile")
        if profile != expected_profile:
            raise HarnessError("request.profile does not match runtime", kind="provider_envelope")
        role = _require_text(envelope.get("role"), "request.role")
        expected_role = "investigation" if expected_profile == "investigation" else "task_review"
        if role != expected_role:
            raise HarnessError("request.role does not match runtime", kind="provider_envelope")
        stage = _require_text(envelope.get("stage"), "request.stage")
        thread_id = envelope.get("thread_id")
        if thread_id is not None:
            thread_id = _require_identifier(thread_id, "request.thread_id")
        tools = _copy_tool_sequence(envelope.get("tools"))
        _assert_sensitive_values_absent(messages, self._sensitive_values, "request.messages")
        _assert_sensitive_values_absent(metadata, self._sensitive_values, "request.metadata")
        _assert_sensitive_values_absent(schema, self._sensitive_values, "request.response_schema")
        _assert_sensitive_values_absent(tools, self._sensitive_values, "request.tools")
        return {
            "model": model,
            "messages": messages,
            "response_schema": schema,
            "metadata": metadata,
            "thread_id": thread_id,
            "tools": tools,
            "role": role,
            "stage": stage,
            "profile": profile,
        }

    def _require_request_kind(self, metadata: Mapping[str, Any], allowed: set[str] | frozenset[str]) -> str:
        """检查 prompt builder 的 request_kind，防止跨角色误用 thread。"""

        request_kind = _require_text(metadata.get("request_kind"), "metadata.request_kind")
        if request_kind not in allowed:
            raise HarnessError("metadata.request_kind is not valid for this runtime", kind="provider_envelope")
        return request_kind

    def _resolve_provider(
        self,
        settings: CodexProviderSettings,
        request_model: str,
    ) -> _ResolvedProvider:
        """解析显式认证模式下的 provider，并核对模型与推理强度。"""

        if self.auth_mode == CODEX_AUTH_MODE_LOCAL_LOGIN:
            if self.role != "investigation" or settings is not self.investigation_provider:
                raise HarnessError(
                    "local_login cannot be used for this Codex role",
                    kind="provider_envelope",
                )
            configured_model = self._environment.get(CODEX_LOCAL_LOGIN_MODEL_ENV)
            model = (
                CODEX_LOCAL_LOGIN_MODEL
                if not isinstance(configured_model, str) or not configured_model.strip()
                else _require_text(configured_model, CODEX_LOCAL_LOGIN_MODEL_ENV)
            )
            if request_model != model:
                raise HarnessError(
                    "local_login request model does not match its configured provider model",
                    kind="provider_envelope",
                )
            configured_effort = self._environment.get(
                CODEX_LOCAL_LOGIN_REASONING_EFFORT_ENV
            )
            reasoning_effort = (
                CODEX_LOCAL_LOGIN_REASONING_EFFORT
                if not isinstance(configured_effort, str) or not configured_effort.strip()
                else _require_text(
                    configured_effort,
                    CODEX_LOCAL_LOGIN_REASONING_EFFORT_ENV,
                ).lower()
            )
            if reasoning_effort not in _SUPPORTED_REASONING_EFFORTS:
                raise HarnessError(
                    "Codex reasoning effort is unsupported",
                    kind="provider_envelope",
                )
            return _ResolvedProvider(
                settings=self._local_login_provider,
                # 仅作为非 secret policy identity；内置 openai provider 不会收到
                # model_providers 覆盖，也不会发送这个占位值给 app-server。
                base_url="builtin://openai",
                model=model,
                reasoning_effort=reasoning_effort,
                secret_environment={},
            )

        base_url = _require_text(self._environment.get(settings.base_url_env), settings.base_url_env)
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise HarnessError("Codex provider base URL is malformed", kind="transport")
        # 只检查存在性；API key 只进入 app-server 私有环境，不进入 thread
        # config、请求、错误或 trace。
        key = self._environment.get(settings.env_key)
        if not isinstance(key, str) or not key.strip():
            raise HarnessError(
                f"Codex provider is missing {settings.env_key}",
                kind="transport",
            )
        model = _require_text(self._environment.get(settings.model_env), settings.model_env)
        if model != request_model:
            raise HarnessError(
                "request.model does not match the explicitly configured provider model",
                kind="provider_envelope",
            )
        reasoning_effort = None
        if settings.reasoning_effort_env is not None:
            configured_effort = self._environment.get(settings.reasoning_effort_env)
            if not isinstance(configured_effort, str) or not configured_effort.strip():
                configured_effort = settings.reasoning_effort_default
            reasoning_effort = _require_text(
                configured_effort,
                settings.reasoning_effort_env,
            ).lower()
            if reasoning_effort not in _SUPPORTED_REASONING_EFFORTS:
                raise HarnessError(
                    "Codex reasoning effort is unsupported",
                    kind="provider_envelope",
                )
        return _ResolvedProvider(
            settings=settings,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            secret_environment={settings.env_key: key},
        )

    def _provider_thread_config(
        self,
        provider: _ResolvedProvider,
        mcp_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """构造 start/resume 完全相同的 provider、responses 和 MCP thread config。"""

        provider_id = provider.settings.provider_id
        config: dict[str, Any] = {
            "model_provider": provider_id,
            "model_reasoning_summary": "none",
            "model_supports_reasoning_summaries": False,
        }
        if self.role == "investigation" and self.investigation_compact_prompt is not None:
            config["compact_prompt"] = self.investigation_compact_prompt
        if (
            self.role == "investigation"
            and self.investigation_model_context_window is not None
        ):
            config["model_context_window"] = self.investigation_model_context_window
        if self.auth_mode == CODEX_AUTH_MODE_API:
            config["model_providers"] = {
                provider_id: {
                    "name": f"ARCHE {provider_id} endpoint",
                    "base_url": provider.base_url,
                    "env_key": provider.settings.env_key,
                    "wire_api": "responses",
                    "supports_websockets": False,
                    "request_max_retries": MODEL_CALL_RETRY_COUNT,
                    "stream_max_retries": MODEL_CALL_RETRY_COUNT,
                }
            }
        if provider.reasoning_effort is not None:
            config["model_reasoning_effort"] = provider.reasoning_effort
        if mcp_config:
            config["mcp_servers"] = json.loads(json.dumps(mcp_config, ensure_ascii=False))
        return config

    def _investigation_mcp_config(self) -> dict[str, Any]:
        """返回只含环境变量名的 MiniChem MCP 配置。"""

        if self.role != "investigation" or self.minichem is None:
            raise HarnessError(
                "MiniChem MCP is unavailable for this runtime role",
                kind="provider_envelope",
            )
        return {
            "minichem_toolbox": {
                "url": self.minichem.endpoint,
                "env_http_headers": {
                    "X-MiniChem-Capability": self.minichem.capability_token_env,
                },
                "required": True,
                "startup_timeout_sec": 60,
                "tool_timeout_sec": _MCP_TOOL_TIMEOUT_SECONDS,
            }
        }

    def _review_mcp_config(self) -> dict[str, Any]:
        """只复制当前 task Review 的 evidence MCP 环境变量映射。"""

        if self.role != "task_review" or self.evidence_mcp is None:
            raise HarnessError(
                "evidence MCP is unavailable for this runtime role",
                kind="provider_envelope",
            )
        return {
            "evidence_reader": {
                "url": self.evidence_mcp.endpoint,
                "env_http_headers": dict(self.evidence_mcp.env_http_headers),
                "required": True,
                "startup_timeout_sec": 60,
                "tool_timeout_sec": _MCP_TOOL_TIMEOUT_SECONDS,
            }
        }

    def _resolve_mcp_secret_environment(self) -> dict[str, str]:
        """解析当前角色 MCP header 环境变量，并只返回 app-server 私有值。"""

        settings: InvestigationMcpSettings | EvidenceMcpSettings
        if self.role == "investigation":
            if self.minichem is None:
                raise HarnessError("MiniChem MCP is unavailable", kind="provider_envelope")
            settings = self.minichem
            names = (self.minichem.capability_token_env,)
        else:
            if self.evidence_mcp is None:
                raise HarnessError("evidence MCP is unavailable", kind="provider_envelope")
            settings = self.evidence_mcp
            names = tuple(self.evidence_mcp.env_http_headers.values())
        resolved: dict[str, str] = {}
        for name in names:
            value = self._environment.get(name)
            if not isinstance(value, str) or not value:
                raise HarnessError(
                    "MCP capability environment is missing",
                    kind="transport",
                )
            resolved[name] = value
        # 旧配置中的字面量已绑定到生成后的变量名，只在构造 app-server 前合并。
        resolved.update(settings.legacy_secret_environment())
        return resolved

    def _sanitize_trace_file(self, trace_path: Path) -> None:
        """替换 trace 中可能回显的 MCP header，并保持文件为 0600。"""

        try:
            if trace_path.is_symlink() or not trace_path.is_file():
                raise HarnessError("trace path is not a regular file", kind="session_state")
            content = trace_path.read_text(encoding="utf-8")
            sanitized = content
            for sensitive in self._sensitive_values:
                sanitized = sanitized.replace(sensitive, "[REDACTED]")
            if sanitized != content:
                trace_path.write_text(sanitized, encoding="utf-8")
            os.chmod(trace_path, 0o600)
        except HarnessError:
            raise
        except OSError as exc:
            raise HarnessError("trace cannot be sanitized", kind="session_state") from exc

    def _trace_path(self, store: CodexSessionStore, stage: str, turn_count: int) -> Path:
        """创建本次 trace；同一未完成 turn 重试时保留旧文件并递增 attempt。"""

        owner = store.owner
        if owner.kind == "investigation":
            relative = (
                Path("investigation")
                / owner.research_task_id
                / (owner.investigation_context_id or "")
                / (owner.hypothesis_execution_input_record_id or "")
            )
        else:
            relative = Path("task_review") / owner.research_task_id
        directory = _prepare_secure_directory(self.trace_root / relative, "trace directory")
        stem = _require_identifier(stage, "stage")
        base_name = f"{stem}.{turn_count}.jsonl"
        attempt = 0
        while attempt <= 10000:
            name = base_name if attempt == 0 else f"{stem}.{turn_count}.attempt-{attempt}.jsonl"
            trace_path = directory / name
            if trace_path.is_symlink():
                raise HarnessError("trace path must not be a symlink", kind="session_state")
            try:
                descriptor = os.open(
                    trace_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                attempt += 1
                continue
            except OSError as exc:
                raise HarnessError("trace path cannot be prepared", kind="session_state") from exc
            else:
                os.close(descriptor)
                os.chmod(trace_path, 0o600)
                return trace_path
        raise HarnessError("trace attempt count is exhausted", kind="session_state")

    @staticmethod
    def _validate_session_recovery_manifest_presence(
        manifest: Mapping[str, Any] | None,
        recovery: Mapping[str, Any] | None,
    ) -> str | None:
        """校验 manifest 与 recovery 的存在关系并返回会话状态。

        输入是本地 session manifest 和恢复记录；函数不启动 worker、不修改文件。
        返回 manifest status；缺失配对、合同失败或 recovery_pending 没有记录时
        抛出 session_state 错误，普通无恢复状态返回后由调用方结束本轮检查。
        """

        if manifest is None:
            if recovery is not None:
                raise HarnessError(
                    "Codex recovery record exists without a session manifest",
                    kind="session_state",
                )
            return None
        status = manifest.get("status")
        if status == "contract_failure":
            raise HarnessError(
                "Codex response contract failure requires explicit recovery",
                kind="session_state",
            )
        if status == "recovery_pending" and recovery is None:
            raise HarnessError(
                "Codex session recovery_pending record is missing",
                kind="session_state",
            )
        if recovery is None:
            if status == "recovery_pending":
                raise HarnessError(
                    "Codex session recovery_pending record is missing",
                    kind="session_state",
                )
            return status
        return status

    @staticmethod
    def _validate_session_recovery_identity(
        status: Any,
        recovery: Mapping[str, Any],
        thread_policy_digest: str,
    ) -> None:
        """核对 recovery 仍属于当前 thread policy 和 recovery_pending 状态。

        输入是 manifest status、恢复记录和当前策略摘要；函数不执行恢复动作。
        策略漂移、过期记录或 state 不一致时抛出 session_state 错误。
        """

        if recovery.get("thread_policy_digest") != thread_policy_digest:
            raise HarnessError(
                "Codex recovery effective thread policy does not match current config",
                kind="session_state",
            )
        if status != "recovery_pending":
            raise HarnessError(
                "Codex recovery record is stale or incomplete",
                kind="session_state",
            )
        if recovery.get("state") != "recovery_pending":
            raise HarnessError(
                "Codex recovery record state does not match the session",
                kind="session_state",
            )

    @staticmethod
    def _validate_session_recovery_action(
        recovery: Mapping[str, Any],
        request_digest: str,
        *,
        allow_goal_recovery: bool = False,
    ) -> None:
        """核对 recovery action，只允许原 receipt 重放或明确人工收束。

        输入是已核对身份的 recovery 记录和当前请求摘要；函数不调用 adapter。
        同请求 completion receipt 可安全返回；request intent、worker 清理和错误
        请求分别抛出 recovery_pending 或 session_state，避免重复提交模型 turn。
        """

        if recovery.get("reason") == "goal":
            if (
                allow_goal_recovery
                and recovery.get("action") == "goal_recovery_required"
                and recovery.get("request_digest") == request_digest
            ):
                return
            raise HarnessError(
                "Codex Goal recovery requires the original Investigation request",
                kind="recovery_pending",
            )
        if recovery.get("reason") == "terminal_turn":
            if (
                recovery.get("action") != "terminal_acknowledgement_required"
                or recovery.get("request_digest") != request_digest
            ):
                raise HarnessError(
                    "Codex terminal recovery requires its original request",
                    kind="session_state",
                )
            return
        if recovery.get("action") != "explicit_new_request_required":
            if not (
                recovery.get("reason") == "completion_receipt"
                and recovery.get("action") == "receipt_replay_required"
                and recovery.get("request_digest") == request_digest
            ):
                if (
                    recovery.get("reason") == "request_intent"
                    and recovery.get("action") == "request_intent_confirmation_required"
                ):
                    raise HarnessError(
                        "Codex request intent requires explicit no-turn confirmation",
                        kind="recovery_pending",
                    )
                raise HarnessError(
                    "Codex worker cleanup requires explicit stop confirmation",
                    kind="recovery_pending",
                )
            return
        if recovery.get("request_digest") == request_digest:
            if recovery.get("reason") == "completion_receipt":
                return
            raise HarnessError(
                "Codex recovery cannot replay the failed request",
                kind="session_state",
            )
        if recovery.get("reason") == "completion_receipt":
            raise HarnessError(
                "Codex completion recovery requires its original request",
                kind="session_state",
            )

    def _assert_session_turn_available(
        self,
        manifest: Mapping[str, Any] | None,
        recovery: Mapping[str, Any] | None,
        request_digest: str,
        thread_policy_digest: str,
        *,
        allow_goal_recovery: bool = False,
    ) -> None:
        """核对会话没有未处理合同失败或无法停止的 worker。"""

        status = self._validate_session_recovery_manifest_presence(
            manifest,
            recovery,
        )
        if manifest is None or recovery is None:
            return
        self._validate_session_recovery_identity(
            status,
            recovery,
            thread_policy_digest,
        )
        self._validate_session_recovery_action(
            recovery,
            request_digest,
            allow_goal_recovery=allow_goal_recovery,
        )

    @staticmethod
    def _assert_recovery_only_available(
        manifest: Mapping[str, Any] | None,
        recovery: Mapping[str, Any] | None,
        *,
        request_digest: str,
        thread_policy_digest: str,
        stage: str,
    ) -> None:
        """核对旧 request-intent 恢复只进入无模型、无新 turn 的 adapter 分支。

        输入是当前 session manifest、可选恢复记录和本次旧请求的策略身份；动作不
        写盘、不访问 provider。明确的 ``request_intent`` 恢复记录按完整身份核对；
        若进程恰好在写入请求意图和服务端 turn 后、写入恢复记录前退出，只允许
        ``created/0`` 或已有成功 turn 对应的 ``completed/N`` 稳定 manifest 继续。
        其它 recovery 原因、摘要/策略、非法观察阶段或不稳定状态一律 fail-closed。
        真正的唯一 history turn 和 provider/workspace/model 身份仍由 adapter 使用
        同一 session 的 request-intent 与完整 history 再次核对。
        """

        if manifest is None:
            raise HarnessError(
                "Codex recovery-only requires a persisted session manifest",
                kind="session_state",
            )
        if recovery is None:
            turn_count = manifest.get("turn_count")
            expected_status = "created" if turn_count == 0 else "completed"
            if (
                isinstance(turn_count, bool)
                or not isinstance(turn_count, int)
                or turn_count < 0
                or manifest.get("status") != expected_status
            ):
                raise HarnessError(
                    "Codex recovery-only session without a recovery record is not stable",
                    kind="session_state",
                )
            return
        if manifest.get("status") != "recovery_pending":
            raise HarnessError(
                "Codex recovery-only session is not recovery_pending",
                kind="session_state",
            )
        expected = {
            "state": "recovery_pending",
            "reason": "request_intent",
            "action": "request_intent_confirmation_required",
            "request_digest": request_digest,
            "thread_policy_digest": thread_policy_digest,
            "turn_id": "",
            "turn_count": manifest.get("turn_count"),
        }
        allowed_observation_stages = {stage}
        if stage == "initial_investigation":
            # 旧 initial turn 已经提交异步作业后，Controller 可能在后续 wake 入口
            # 首次发现其 request-intent，因此 recovery.stage 记录的是发现位置；
            # 原请求身份仍由 digest、policy 和磁盘 request-intent 严格核对。
            allowed_observation_stages.add("investigation_job_wake")
        if (
            recovery.get("stage") not in allowed_observation_stages
            or any(recovery.get(field) != value for field, value in expected.items())
        ):
            raise HarnessError(
                "Codex recovery-only request-intent identity does not match session",
                kind="session_state",
            )


    def _build_harness_request(
        self,
        envelope: Mapping[str, Any],
        store: CodexSessionStore,
        provider: _ResolvedProvider,
        timeout_seconds: float | None,
        stage: str,
    ) -> HarnessRequest:
        """从已验证 envelope 构造一次不可变 Harness 请求。"""

        return HarnessRequest(
            messages=tuple(envelope["messages"]),
            model=provider.model,
            role=str(envelope["role"]),
            stage=str(envelope["stage"]),
            workspace=store.workspace,
            research_task_id=store.owner.research_task_id,
            invocation_id=stage,
            timeout_seconds=timeout_seconds,
            response_schema=dict(envelope["response_schema"]),
            tools=tuple(envelope["tools"]),
            metadata=dict(envelope["metadata"]),
            profile=str(envelope["profile"]),
        )

    def _resolve_turn_session(
        self,
        envelope: Mapping[str, Any],
        store: CodexSessionStore,
        manifest: Mapping[str, Any] | None,
        require_existing: bool,
        *,
        migrated_from_thread_id: str | None = None,
        migration_request_thread_ids: tuple[str, ...] = (),
        replace_unstarted_thread_id: str | None = None,
        migration_target_thread_id: str | None = None,
    ) -> str | None:
        """核对请求 thread 与 manifest，返回 adapter 可用的 session id。"""

        request_thread_id = envelope.get("thread_id")
        if require_existing and manifest is None and migrated_from_thread_id is None:
            raise HarnessError(
                "Investigation repair requires an existing Codex session",
                kind="session_state",
            )
        if manifest is None:
            if migrated_from_thread_id is not None:
                allowed_thread_ids = migration_request_thread_ids or (
                    migrated_from_thread_id,
                )
                if request_thread_id is not None and request_thread_id not in allowed_thread_ids:
                    raise HarnessError(
                        "provider migration request thread is not in the verified migration chain",
                        kind="session_state",
                    )
                if migration_target_thread_id is not None:
                    # marker 已经记录 provider 返回的 target thread，但本地
                    # manifest 还没有刷盘；继续 resume 该 thread，不能重新创建。
                    return _require_identifier(
                        migration_target_thread_id,
                        "provider migration target_thread_id",
                    )
                # 旧 thread 有意不传给 thread_resume；operation/job identity
                # 保留在 wake request 和 tracker 中。
                return None
            if request_thread_id is not None:
                raise HarnessError(
                    "request thread_id cannot start a missing Codex session",
                    kind="session_state",
                )
            return None
        session_id = _require_identifier(manifest.get("thread_id"), "manifest.thread_id")
        if replace_unstarted_thread_id is not None:
            if session_id != replace_unstarted_thread_id:
                raise HarnessError(
                    "provider migration unstarted thread does not match session manifest",
                    kind="session_state",
                )
            if manifest.get("status") != "created" or manifest.get("turn_count") != 0:
                raise HarnessError(
                    "provider migration replacement requires an unstarted thread",
                    kind="session_state",
                )
            # 直接前驱是 Working Memory 当前身份；target manifest 只是上次
            # thread_start 后留下的空登记，不能把它交给 thread_resume。
            return None
        if request_thread_id is not None and request_thread_id != session_id:
            allowed_thread_ids = migration_request_thread_ids or (
                (migrated_from_thread_id,) if migrated_from_thread_id is not None else ()
            )
            if request_thread_id not in allowed_thread_ids:
                raise HarnessError(
                    "request thread_id does not match session manifest",
                    kind="session_state",
                )
        return session_id

    def _build_sdk_turn_config(
        self,
        store: CodexSessionStore,
        secret_environment: Mapping[str, str],
        *,
        migration_source_store: CodexSessionStore | None = None,
    ) -> Any:
        """构造带私有环境和 sandbox 的 SDK 配置；裸官方 adapter 被拒绝。"""

        if self._adapter is run_codex_sdk_turn and self.sandbox_spec is None:
            raise HarnessError(
                "private app-server environment requires a Codex process sandbox",
                kind="transport",
            )
        sandbox_spec = self._select_process_sandbox_for_turn_store(
            store,
            migration_source_store=migration_source_store,
        )
        return build_codex_config(
            codex_bin=self.codex_bin,
            workspace=store.workspace,
            codex_home=store.codex_home,
            sandbox_spec=sandbox_spec,
            environment_keys=tuple(secret_environment),
            secret_environment=secret_environment,
        )

    def _select_process_sandbox_for_turn_store(
        self,
        store: CodexSessionStore,
        *,
        migration_source_store: CodexSessionStore | None,
    ) -> CodexProcessSandboxSpec | None:
        """把进程沙箱精确绑定到当前 turn 的会话 store。

        普通 turn 必须继续使用构造 runtime 时冻结的 workspace 和 CODEX_HOME。
        provider migration 已经核对 source manifest、policy 和 active job 后，会为同一
        owner 创建新的 CODEX_HOME；此处只允许把原 spec 的 CODEX_HOME 改到该固定
        ``state/provider_migrations/<policy>/codex_home`` 子树，其他身份和挂载不变。
        任一 workspace、owner、scope、model 或路径形状不一致都会在启动 app-server
        前失败，不会扩大可写目录。
        """

        sandbox_spec = self.sandbox_spec
        if sandbox_spec is None:
            return None
        target_workspace = store.workspace.resolve()
        target_codex_home = store.codex_home.resolve()
        spec_workspace = sandbox_spec.workspace.expanduser().resolve()
        spec_codex_home = sandbox_spec.codex_home.expanduser().resolve()
        if spec_workspace != target_workspace:
            raise HarnessError(
                "Codex process sandbox workspace does not match the turn store",
                kind="provider_envelope",
            )
        if spec_codex_home == target_codex_home:
            return sandbox_spec
        source = migration_source_store
        if source is None:
            raise HarnessError(
                "Codex process sandbox CODEX_HOME changed without provider migration",
                kind="provider_envelope",
            )
        if (
            source.workspace.resolve() != target_workspace
            or source.codex_home.resolve() != spec_codex_home
            or source.owner != store.owner
            or source.scope_id != store.scope_id
            or source.model != store.model
        ):
            raise HarnessError(
                "Codex provider migration sandbox source identity does not match",
                kind="provider_envelope",
            )
        migration_state_root = source.manifest_path.parent.resolve()
        try:
            relative = target_codex_home.relative_to(migration_state_root)
        except ValueError as exc:
            raise HarnessError(
                "Codex provider migration CODEX_HOME is outside the owner state",
                kind="provider_envelope",
            ) from exc
        parts = relative.parts
        cursor = 0
        while True:
            if (
                cursor + 1 >= len(parts)
                or parts[cursor] != "provider_migrations"
                or re.fullmatch(r"[0-9a-f]{32}", parts[cursor + 1]) is None
            ):
                raise HarnessError(
                    "Codex provider migration CODEX_HOME path is malformed",
                    kind="provider_envelope",
                )
            cursor += 2
            if cursor == len(parts) - 1 and parts[cursor] == "codex_home":
                break
            if cursor >= len(parts) or parts[cursor] != "state":
                raise HarnessError(
                    "Codex provider migration CODEX_HOME path is malformed",
                    kind="provider_envelope",
                )
            cursor += 1
        if not parts or parts[-1] != "codex_home":
            raise HarnessError(
                "Codex provider migration CODEX_HOME path is malformed",
                kind="provider_envelope",
            )
        return replace(sandbox_spec, codex_home=target_codex_home)

    def _build_codex_turn_policy_inputs(
        self,
        *,
        envelope: Mapping[str, Any],
        store: CodexSessionStore,
        provider: _ResolvedProvider,
        mcp_config: Mapping[str, Any],
        timeout_seconds: float | None,
    ) -> tuple[HarnessRequest, str, Mapping[str, str], dict[str, Any], str]:
        """为指定 store 构造 request、秘密环境、能力配置和冻结 policy 摘要。"""

        request = self._build_harness_request(
            envelope,
            store,
            provider,
            timeout_seconds,
            str(envelope["stage"]),
        )
        request_digest = _calculate_request_digest(request)
        secret_environment = {
            **provider.secret_environment,
            **self._resolve_mcp_secret_environment(),
        }
        thread_config = build_codex_capability_config(
            thread_config=self._provider_thread_config(provider, mcp_config),
            workspace=store.workspace,
            profile=str(envelope["profile"]),
            env_key=provider.settings.env_key,
            environment_keys=tuple(secret_environment),
            auth_mode=self.auth_mode,
        )
        _assert_sensitive_values_absent(thread_config, self._sensitive_values, "Codex thread config")
        policy_digest = calculate_codex_thread_policy_digest(
            thread_config=thread_config,
            workspace=store.workspace,
            provider_id=provider.settings.provider_id,
            env_key=provider.settings.env_key,
            model=provider.model,
            role=str(envelope["role"]),
            stage=str(envelope["stage"]),
            profile=str(envelope["profile"]),
            auth_mode=self.auth_mode,
            permission_profile=(
                CODEX_INVESTIGATION_PERMISSION_PROFILE if self.role == "investigation" else None
            ),
        )
        manifest = store.load()
        stored_policy_digest = (
            manifest.get("thread_policy_digest")
            if isinstance(manifest, Mapping)
            else None
        )
        if (
            self.role == "investigation"
            and isinstance(stored_policy_digest, str)
            and stored_policy_digest != policy_digest
        ):
            # Goal 默认关闭前创建并显式冻结为 Goal 的 Investigation thread 已把
            # goals=true 纳入不可变策略摘要。只有其它 provider、认证、MCP、
            # 模型和权限字段仍完全一致、仅 Goal 开关能解释旧摘要时，才继续
            # 使用该实验性运行方式；新 session 默认使用普通 turn。
            goal_thread_config = build_codex_capability_config(
                thread_config=self._provider_thread_config(provider, mcp_config),
                workspace=store.workspace,
                profile=str(envelope["profile"]),
                env_key=provider.settings.env_key,
                environment_keys=tuple(secret_environment),
                auth_mode=self.auth_mode,
                enable_investigation_goals=True,
            )
            goal_policy_digest = calculate_codex_thread_policy_digest(
                thread_config=goal_thread_config,
                workspace=store.workspace,
                provider_id=provider.settings.provider_id,
                env_key=provider.settings.env_key,
                model=provider.model,
                role=str(envelope["role"]),
                stage=str(envelope["stage"]),
                profile=str(envelope["profile"]),
                auth_mode=self.auth_mode,
                permission_profile=CODEX_INVESTIGATION_PERMISSION_PROFILE,
            )
            if stored_policy_digest == goal_policy_digest:
                thread_config = goal_thread_config
                policy_digest = goal_policy_digest
        return request, request_digest, secret_environment, thread_config, policy_digest

    def _prepare_codex_turn(
        self,
        *,
        envelope: Mapping[str, Any],
        store: CodexSessionStore,
        provider: _ResolvedProvider,
        mcp_config: Mapping[str, Any],
        timeout_seconds: float | None,
        stage: str,
        require_existing: bool,
        active_job: InvestigationJobState | Mapping[str, Any] | None = None,
        recover_existing_terminal_turn: bool = False,
        replay_turn_id: str | None = None,
        recovery_only: bool = False,
        recovery_turn_id: str | None = None,
    ) -> _PreparedCodexTurn:
        """在 adapter 启动前完成请求、恢复状态、trace 和 SDK 配置准备。"""

        (
            request,
            request_digest,
            secret_environment,
            thread_config,
            policy_digest,
        ) = self._build_codex_turn_policy_inputs(
            envelope=envelope,
            store=store,
            provider=provider,
            mcp_config=mcp_config,
            timeout_seconds=timeout_seconds,
        )
        source_store = store
        migration_selection = resolve_explicit_provider_migration(
            store,
            session_root=self.session_root,
            target_policy_digest=policy_digest,
            auth_mode=self.auth_mode,
            provider_id=provider.settings.provider_id,
            model=provider.model,
            # 旧 invalid_encrypted_content 恢复必须在原 thread 上做官方 rollback；
            # 迁移后的新 thread 绝不能执行这个旧历史清理 hook。
            allow_migration=(
                self.allow_provider_migration
                and not recover_existing_terminal_turn
                and not recovery_only
            ),
            request_thread_id=envelope.get("thread_id"),
            active_job=active_job,
        )
        migrated_store = migration_selection.store
        migrated_from_thread_id = (
            None
            if migrated_store is store
            else _require_identifier(
                migration_selection.source_thread_id,
                "provider migration source thread_id",
            )
        )
        if migrated_store is not store:
            store = migrated_store
            (
                request,
                request_digest,
                secret_environment,
                thread_config,
                policy_digest,
            ) = self._build_codex_turn_policy_inputs(
                envelope=envelope,
                store=store,
                provider=provider,
                mcp_config=mcp_config,
                timeout_seconds=timeout_seconds,
            )
        permission_profile: str | None = None
        if self.role == "investigation":
            permission_profile = prepare_codex_permission_profile(
                store.codex_home,
                auth_source=(self.auth_source if self.auth_mode == CODEX_AUTH_MODE_LOCAL_LOGIN else None),
            )
        manifest, recovery = store.reconcile_recovery_state(
            thread_policy_digest=policy_digest,
        )
        migration_target_thread_id = migration_selection.target_thread_id
        migration_target_thread_callback: Callable[[str], None] | None = None
        migration_turn_started_callback: Callable[[str], None] | None = None
        if migrated_from_thread_id is not None:
            # marker 在首次选择迁移时已经落盘；这里再把 target turn 的 request
            # 摘要和旧计数补齐。它发生在 SDK 启动前，Result/receipt 恢复只读
            # 这些字段，不会重新调用 provider。
            persist_provider_migration_turn_context(
                migration_selection,
                request_thread_id=envelope.get("thread_id"),
                request_digest=request_digest,
                turn_count_before=(0 if manifest is None else int(manifest["turn_count"])),
            )
            migration_target_thread_callback = partial(
                persist_provider_migration_target_thread,
                migration_selection,
            )
            migration_turn_started_callback = partial(
                persist_provider_migration_turn_id,
                migration_selection,
            )
        session_id = self._resolve_turn_session(
            envelope,
            store,
            manifest,
            require_existing,
            migrated_from_thread_id=migrated_from_thread_id,
            migration_request_thread_ids=migration_selection.request_thread_ids,
            replace_unstarted_thread_id=migration_selection.replace_unstarted_thread_id,
            migration_target_thread_id=migration_target_thread_id,
        )
        turn_count_before = 0 if manifest is None else int(manifest["turn_count"])
        receipt_root = _prepare_secure_directory(
            store.manifest_path.parent / "completed_turn_receipts",
            "completed turn receipt directory",
        )
        waiting_receipt_replay = bool(
            isinstance(active_job, InvestigationJobState)
            and manifest is not None
            and self._has_completed_receipt_matching_waiting_turn(
                manifest,
                active_job,
                thread_id=_require_identifier(
                    manifest.get("thread_id"),
                    "manifest.thread_id",
                ),
                thread_policy_digest=policy_digest,
                receipt_root=receipt_root,
                recovery=recovery,
            )
        )
        completed_turn_replay = bool(
            replay_turn_id is not None
            and manifest is not None
            and self._has_completed_receipt_matching_turn(
                manifest,
                replay_turn_id,
                thread_id=_require_identifier(
                    manifest.get("thread_id"),
                    "manifest.thread_id",
                ),
                thread_policy_digest=policy_digest,
                receipt_root=receipt_root,
                recovery=recovery,
            )
        )
        if replay_turn_id is not None and not completed_turn_replay:
            raise HarnessError(
                "explicit completed turn replay has no matching receipt",
                kind="session_state",
            )
        if not recovery_only and not waiting_receipt_replay and not completed_turn_replay:
            goal_request = (
                self.role == "investigation"
                and envelope.get("profile") == "investigation"
                and envelope.get("metadata", {}).get("request_kind") in _ALLOWED_REQUEST_KINDS
            )
            self._assert_session_turn_available(
                manifest,
                recovery,
                request_digest,
                policy_digest,
                allow_goal_recovery=goal_request,
            )
        if recovery_only:
            recovery_intent = read_codex_request_intent(receipt_root)
            if recovery_intent is None:
                raise HarnessError(
                    "Codex recovery-only request-intent is missing",
                    kind="session_state",
                )
            # 该分支只确认已有终态，不提交模型请求；adapter 使用原 intent 摘要
            # 绑定 terminal journal，不能用升级后重新构造的 prompt 摘要替代它。
            request_digest = recovery_intent["request_digest"]
            self._assert_recovery_only_available(
                manifest,
                recovery,
                request_digest=request_digest,
                thread_policy_digest=policy_digest,
                stage=stage,
            )
        trace_path = self._trace_path(store, stage, turn_count_before)
        sdk_config = self._build_sdk_turn_config(
            store,
            secret_environment,
            migration_source_store=(
                source_store if migrated_from_thread_id is not None else None
            ),
        )
        return _PreparedCodexTurn(
            request=request,
            request_digest=request_digest,
            secret_environment=secret_environment,
            thread_config=thread_config,
            thread_policy_digest=policy_digest,
            manifest=manifest,
            recovery=recovery,
            session_id=session_id,
            turn_count_before=turn_count_before,
            completed_turn_receipt_root=receipt_root,
            trace_path=trace_path,
            sdk_config=sdk_config,
            stage=stage,
            permission_profile=permission_profile,
            store=store,
            migrated_from_thread_id=migrated_from_thread_id,
            migration_request_thread_ids=migration_selection.request_thread_ids,
            replace_unstarted_thread_id=migration_selection.replace_unstarted_thread_id,
            migration_target_thread_id=migration_target_thread_id,
            migration_target_thread_callback=migration_target_thread_callback,
            migration_turn_started_callback=migration_turn_started_callback,
            recovery_only=recovery_only,
            recovery_turn_id=recovery_turn_id,
            replay_turn_id=replay_turn_id,
        )

    def _run_validated_turn(
        self,
        *,
        envelope: Mapping[str, Any],
        store: CodexSessionStore,
        provider: _ResolvedProvider,
        mcp_config: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], Any],
        expected_result_type: type[Any] | tuple[type[Any], ...],
        timeout_seconds: float | None,
        stage: str,
        require_existing: bool,
        before_new_turn: Callable[[], None] | None,
        on_thread_ready: ThreadReadyCallback | None,
        on_thread_migration: ThreadMigrationCallback | None,
        on_turn_started: TurnStartedCallback | None,
        on_terminal_turn: TerminalTurnCallback | None,
        on_job_accepted: JobAcceptedCallback | None,
        on_job_collected: JobCollectedCallback | None,
        active_job: InvestigationJobState | Mapping[str, Any] | None,
        recover_existing_terminal_turn: bool = False,
        replay_turn_id: str | None = None,
        recovery_only: bool = False,
    ) -> InvestigationTurnResult:
        """在 session lock 内执行 SDK、记录 completed turn，再验证结构化 JSON。"""

        with ExitStack() as lock_stack:
            lock_stack.enter_context(store.locked())
            prepared = self._prepare_codex_turn(
                envelope=envelope,
                store=store,
                provider=provider,
                mcp_config=mcp_config,
                timeout_seconds=timeout_seconds,
                stage=stage,
                require_existing=require_existing,
                active_job=active_job,
                recover_existing_terminal_turn=recover_existing_terminal_turn,
                replay_turn_id=replay_turn_id,
                recovery_only=recovery_only,
                recovery_turn_id=(
                    active_job.turn_id
                    if recovery_only and isinstance(active_job, InvestigationJobState)
                    else None
                ),
            )
            turn_store = prepared.store or store
            if turn_store is not store:
                # provider migration 产生的是 owner 私有的新 store；迁移期间仍
                # 保持旧 store 锁，用第二把锁保护目标 manifest/CODEX_HOME，避免
                # 两个恢复进程同时创建 target thread。
                lock_stack.enter_context(turn_store.locked())
            callback_stage = prepared.stage
            if prepared.recovery_only and isinstance(prepared.recovery, Mapping):
                recovery_stage = prepared.recovery.get("stage")
                if not isinstance(recovery_stage, str) or not recovery_stage.strip():
                    raise HarnessError(
                        "Codex recovery-only recovery stage is missing",
                        kind="session_state",
                    )
                # 已有 request-intent 恢复记录时，terminal ack 必须沿用其观察阶段，
                # 才能在锁内把同一记录升级为 terminal_turn。若记录尚未来得及写入，
                # 则沿用 prepared.stage 创建 terminal_turn 恢复记录。
                callback_stage = recovery_stage
            callbacks = _CodexTurnCallbacks(
                store=turn_store,
                manifest=prepared.manifest,
                session_id=prepared.session_id,
                stage=callback_stage,
                request_digest=prepared.request_digest,
                thread_policy_digest=prepared.thread_policy_digest,
                thread_ready_callback=self._build_thread_ready_callback(on_thread_ready),
                thread_migration_callback=on_thread_migration,
                migration_target_thread_callback=prepared.migration_target_thread_callback,
                migration_turn_started_callback=prepared.migration_turn_started_callback,
                migration_source_thread_id=prepared.migrated_from_thread_id,
                migration_request_thread_ids=prepared.migration_request_thread_ids,
                replace_unstarted_thread_id=prepared.replace_unstarted_thread_id,
                job_accepted_callback=on_job_accepted,
                job_collected_callback=on_job_collected,
            )
            if active_job is not None:
                callbacks.active_job_receipt = _build_active_job_receipt(active_job)
            elif prepared.stage == "investigation_job_wake":
                callbacks.active_job_receipt = _build_wake_job_receipt(prepared.request)

            # local_login 的 clone 必须覆盖整个 app-server 生命周期，包括
            # startup、resume、turn 和 close；API 模式上下文无副作用。
            with build_codex_auth_session(
                self.auth_mode,
                codex_home=turn_store.codex_home,
                auth_source=self.auth_source,
            ):
                sdk_result = self._invoke_codex_adapter(
                    prepared=prepared,
                    store=turn_store,
                    provider=provider,
                    before_new_turn=before_new_turn,
                    active_job=active_job,
                    recover_existing_terminal_turn=recover_existing_terminal_turn,
                    replay_turn_id=replay_turn_id,
                    recovery_only=prepared.recovery_only,
                    recovery_turn_id=prepared.recovery_turn_id,
                    callbacks=callbacks,
                    on_turn_started=on_turn_started,
                    on_terminal_turn=on_terminal_turn,
                    on_job_accepted=on_job_accepted,
                    on_job_collected=on_job_collected,
                )
            self._sanitize_trace_file(prepared.trace_path)
            return self._complete_codex_turn(
                prepared=prepared,
                callbacks=callbacks,
                sdk_result=sdk_result,
                store=turn_store,
                validator=validator,
                expected_result_type=expected_result_type,
            )

    def _build_turn_started_callback(
        self,
        callbacks: _CodexTurnCallbacks,
        on_turn_started: TurnStartedCallback | None,
    ) -> Callable[[str], None] | None:
        """准备 SDK accepted hook；缺少底层 hook 时拒绝伪造生命周期。"""

        if on_turn_started is None:
            return None
        if not _callable_accepts_keyword(self._adapter, "on_turn_started"):
            raise HarnessError(
                "on_turn_started requires an adapter with the accepted-turn hook",
                kind="session_protocol",
            )
        return partial(self._notify_turn_started, callbacks, on_turn_started)

    def _build_terminal_turn_callback(
        self,
        callbacks: _CodexTurnCallbacks,
        on_terminal_turn: TerminalTurnCallback | None,
    ) -> Callable[[str, str, str], None] | None:
        """把终态 hook 绑定到当前会话，并拒绝无法同步确认的 adapter。"""

        if on_terminal_turn is None:
            return None
        if not _callable_accepts_keyword(self._adapter, "on_terminal_turn"):
            raise HarnessError(
                "on_terminal_turn requires an adapter with the terminal-turn hook",
                kind="session_protocol",
            )
        return partial(self._notify_terminal_turn, callbacks, on_terminal_turn)

    def _build_goal_terminal_turn_callback(
        self,
        callbacks: _CodexTurnCallbacks,
        on_terminal_turn: TerminalTurnCallback | None,
    ) -> Callable[[str, str, str], None] | None:
        """绑定 Goal 物理终态回调，并保留 Goal 恢复事实。"""

        if on_terminal_turn is None:
            return None
        if not _callable_accepts_keyword(self._adapter, "on_goal_terminal_turn"):
            raise HarnessError(
                "Investigation Goal requires the Goal terminal-turn hook",
                kind="session_protocol",
            )
        return partial(self._notify_goal_terminal_turn, callbacks, on_terminal_turn)

    def _build_thread_ready_callback(
        self,
        on_thread_ready: ThreadReadyCallback | None,
    ) -> ThreadReadyCallback | None:
        """准备真实 thread hook；它发生在 provider turn 之前且只转发稳定身份。"""

        if on_thread_ready is None:
            return None
        if not callable(on_thread_ready):
            raise TypeError("on_thread_ready must be callable or None")

        def notify_thread_ready(thread_id: str) -> None:
            """把 adapter 返回的 thread 身份交给 Working Memory 生命周期。"""

            normalized_thread_id = _require_identifier(thread_id, "adapter.thread_id")
            _assert_sensitive_values_absent(
                normalized_thread_id,
                self._sensitive_values,
                "adapter.thread_id",
            )
            try:
                on_thread_ready(normalized_thread_id)
            except BaseException as exc:
                error = _sanitize_runtime_error_chain(exc, self._sensitive_values)
                raise error from None

        return notify_thread_ready

    def _notify_turn_started(
        self,
        callbacks: _CodexTurnCallbacks,
        on_turn_started: TurnStartedCallback,
        turn_id: str,
    ) -> None:
        """在 adapter 确认 app-server 接受 turn 后通知 Working Memory 回调。"""

        normalized_turn_id = _require_identifier(turn_id, "adapter.turn_id")
        _assert_sensitive_values_absent(
            normalized_turn_id,
            self._sensitive_values,
            "adapter.turn_id",
        )
        if callbacks.registered_thread_id is None:
            raise HarnessError(
                "Codex accepted turn has no registered thread",
                kind="session_state",
            )
        if callbacks.migration_turn_started_callback is not None:
            callbacks.migration_turn_started_callback(normalized_turn_id)
        try:
            on_turn_started(callbacks.registered_thread_id, normalized_turn_id)
        except BaseException as exc:
            error = _sanitize_runtime_error_chain(exc, self._sensitive_values)
            raise error from None

    def _notify_terminal_turn(
        self,
        callbacks: _CodexTurnCallbacks,
        on_terminal_turn: TerminalTurnCallback,
        thread_id: str,
        turn_id: str,
        status: str,
    ) -> None:
        """同步保存上层终态事实，并清理同一 session recovery。

        输入仅接受 adapter 白名单中的 thread、turn 和 failed/interrupted。动作先
        保存本地 ``terminal_turn`` 阻断，再调用上层幂等回调；两步成功后才确认并
        清理本地 recovery。任一失败都以脱敏异常返回 adapter，使其保留 journal
        与 request-intent。
        """

        normalized_thread_id = _require_identifier(thread_id, "adapter.thread_id")
        normalized_turn_id = _require_identifier(turn_id, "adapter.turn_id")
        if status not in {"failed", "interrupted"}:
            raise HarnessError(
                "adapter terminal turn status is invalid",
                kind="session_protocol",
            )
        if callbacks.registered_thread_id != normalized_thread_id:
            raise HarnessError(
                "Codex terminal turn has no matching registered thread",
                kind="session_state",
            )
        try:
            callbacks.store.record_recovery_pending(
                stage=callbacks.stage,
                request_digest=callbacks.request_digest,
                thread_policy_digest=callbacks.thread_policy_digest,
                turn_id=normalized_turn_id,
                reason="terminal_turn",
            )
            on_terminal_turn(normalized_thread_id, normalized_turn_id, status)
            callbacks.store.acknowledge_terminal_turn_recovery(
                request_digest=callbacks.request_digest,
                thread_policy_digest=callbacks.thread_policy_digest,
                turn_id=normalized_turn_id,
            )
        except BaseException as exc:
            error = _sanitize_runtime_error_chain(exc, self._sensitive_values)
            raise error from None

    def _notify_goal_terminal_turn(
        self,
        callbacks: _CodexTurnCallbacks,
        on_terminal_turn: TerminalTurnCallback,
        thread_id: str,
        turn_id: str,
        status: str,
    ) -> None:
        """保存 Goal 终态恢复记录，再通知上层实际物理 turn 状态。"""

        normalized_thread_id = _require_identifier(thread_id, "adapter.thread_id")
        normalized_turn_id = _require_identifier(turn_id, "adapter.turn_id")
        if status not in {"failed", "interrupted"}:
            raise HarnessError(
                "adapter Goal terminal turn status is invalid",
                kind="session_protocol",
            )
        if callbacks.registered_thread_id != normalized_thread_id:
            raise HarnessError(
                "Codex Goal terminal turn has no matching registered thread",
                kind="session_state",
            )
        try:
            callbacks.store.record_recovery_pending(
                stage=callbacks.stage,
                request_digest=callbacks.request_digest,
                thread_policy_digest=callbacks.thread_policy_digest,
                turn_id=normalized_turn_id,
                reason="goal",
            )
            on_terminal_turn(normalized_thread_id, normalized_turn_id, status)
        except BaseException as exc:
            error = _sanitize_runtime_error_chain(exc, self._sensitive_values)
            raise error from None

    def _invoke_codex_adapter(
        self,
        *,
        prepared: _PreparedCodexTurn,
        store: CodexSessionStore,
        provider: _ResolvedProvider,
        before_new_turn: Callable[[], None] | None,
        active_job: InvestigationJobState | Mapping[str, Any] | None,
        recover_existing_terminal_turn: bool,
        replay_turn_id: str | None,
        recovery_only: bool,
        recovery_turn_id: str | None,
        callbacks: _CodexTurnCallbacks,
        on_turn_started: TurnStartedCallback | None,
        on_terminal_turn: TerminalTurnCallback | None,
        on_job_accepted: JobAcceptedCallback | None,
        on_job_collected: JobCollectedCallback | None,
    ) -> Any:
        """调用 SDK adapter，并把失败 trace 与 recovery 状态安全落盘。"""

        adapter_error: BaseException | None = None
        sdk_result: Any = None
        turn_started_callback = self._build_turn_started_callback(
            callbacks,
            on_turn_started,
        )
        terminal_turn_callback = self._build_terminal_turn_callback(
            callbacks,
            on_terminal_turn,
        )
        goal_terminal_turn_callback = None
        if (
            prepared.request.role == "investigation"
            and prepared.request.metadata.get("request_kind") in _ALLOWED_REQUEST_KINDS
        ):
            goal_terminal_turn_callback = self._build_goal_terminal_turn_callback(
                callbacks,
                on_terminal_turn,
            )
        try:
            before_new_turn_with_thread = None
            if recover_existing_terminal_turn:
                if active_job is None:
                    raise HarnessError(
                        "existing terminal turn recovery requires an active job",
                        kind="session_state",
                    )
                before_new_turn_with_thread = self._build_existing_terminal_turn_recovery_callback(
                    prepared,
                    store,
                    active_job,
                )
            adapter_arguments = {
                "config": prepared.sdk_config,
                "request": prepared.request,
                "thread_config": prepared.thread_config,
                "session_id": prepared.session_id,
                "trace_path": prepared.trace_path,
                "provider_id": provider.settings.provider_id,
                "env_key": provider.settings.env_key,
                "on_thread_ready": callbacks.register_thread_ready,
                "trace_sensitive_values": self._sensitive_values,
                "secret_environment": prepared.secret_environment,
                "completed_turn_receipt_root": prepared.completed_turn_receipt_root,
                "request_digest": prepared.request_digest,
                "turn_count_before": prepared.turn_count_before,
                "thread_policy_digest": prepared.thread_policy_digest,
                "before_new_turn": before_new_turn,
                "idle_timeout_seconds": self.sdk_idle_timeout_seconds,
                "tool_idle_timeout_seconds": self.sdk_tool_idle_timeout_seconds,
                "on_recovery_pending": callbacks.record_worker_recovery,
                "completion_recovery": prepared.recovery,
            }
            if self.role == "investigation":
                goal_feature = prepared.thread_config.get("features", {}).get(
                    "goals"
                )
                if not isinstance(goal_feature, bool):
                    raise HarnessError(
                        "Investigation Goal capability selection is missing",
                        kind="session_state",
                    )
                if _callable_accepts_keyword(
                    self._adapter,
                    "enable_investigation_goals",
                ):
                    adapter_arguments["enable_investigation_goals"] = goal_feature
                elif goal_feature is False:
                    raise HarnessError(
                        "ordinary Investigation recovery requires Goal capability control",
                        kind="session_protocol",
                    )
            if replay_turn_id is None and active_job is not None and not recovery_only:
                try:
                    replay_turn_id = (
                        active_job.turn_id
                        if isinstance(active_job, InvestigationJobState)
                        else active_job.get("turn_id")
                    )
                except AttributeError:
                    replay_turn_id = None
            if replay_turn_id is not None and not recovery_only:
                if not _callable_accepts_keyword(self._adapter, "replay_turn_id"):
                    raise HarnessError(
                        "completed turn replay requires an adapter replay hook",
                        kind="session_protocol",
                    )
                adapter_arguments["replay_turn_id"] = _require_identifier(
                    replay_turn_id,
                    "replay_turn_id",
                )
            if recovery_only:
                if recovery_turn_id is None:
                    raise HarnessError(
                        "Codex recovery-only requires the waiting turn identity",
                        kind="session_state",
                    )
                if not _callable_accepts_keyword(self._adapter, "recovery_only"):
                    raise HarnessError(
                        "Codex recovery-only requires an adapter with the recovery-only hook",
                        kind="session_protocol",
                    )
                adapter_arguments["recovery_only"] = True
                if _callable_accepts_keyword(self._adapter, "recovery_turn_id"):
                    adapter_arguments["recovery_turn_id"] = recovery_turn_id
                else:
                    raise HarnessError(
                        "Codex recovery-only requires the waiting-turn identity hook",
                        kind="session_protocol",
                    )
            if _callable_accepts_keyword(self._adapter, "auth_mode"):
                adapter_arguments["auth_mode"] = self.auth_mode
            if (
                prepared.permission_profile is not None
                and _callable_accepts_keyword(self._adapter, "permission_profile")
            ):
                adapter_arguments["permission_profile"] = prepared.permission_profile
            if before_new_turn_with_thread is not None:
                if not _callable_accepts_keyword(self._adapter, "before_new_turn_with_thread"):
                    raise HarnessError(
                        "existing terminal turn recovery requires the thread-aware pre-turn hook",
                        kind="session_protocol",
                    )
                adapter_arguments["before_new_turn_with_thread"] = before_new_turn_with_thread
            if prepared.stage in _ROLLBACK_ELIGIBLE_INVESTIGATION_STAGES:
                rollback_callback = self._build_investigation_rollback_callback(
                    prepared,
                    store,
                )
                if _callable_accepts_keyword(self._adapter, "on_investigation_rollback"):
                    adapter_arguments["on_investigation_rollback"] = rollback_callback
                elif _callable_accepts_keyword(self._adapter, "on_wake_rollback"):
                    # 旧 adapter 只支持 wake 名称时仍接收相同的持久化合同。
                    adapter_arguments["on_wake_rollback"] = rollback_callback
                else:
                    raise HarnessError(
                        "Investigation turn requires the durable rollback hook",
                        kind="session_protocol",
                    )
            if turn_started_callback is not None:
                adapter_arguments["on_turn_started"] = turn_started_callback
            if terminal_turn_callback is not None:
                adapter_arguments["on_terminal_turn"] = terminal_turn_callback
            if goal_terminal_turn_callback is not None:
                adapter_arguments["on_goal_terminal_turn"] = goal_terminal_turn_callback
            if _callable_accepts_keyword(self._adapter, "on_tool_completed"):
                adapter_arguments["on_tool_completed"] = callbacks.capture_tool_completed
            elif on_job_accepted is not None:
                raise HarnessError(
                    "on_job_accepted requires an adapter with the tool-completed hook",
                    kind="session_protocol",
                )
            sdk_result = self._adapter(
                **adapter_arguments,
            )
        except BaseException as exc:
            adapter_error = self._sanitize_failed_adapter_error(prepared.trace_path, exc)
            if getattr(exc, "kind", None) == "recovery_pending":
                adapter_error = self._persist_worker_recovery_after_error(
                    callbacks=callbacks,
                    original_error=exc,
                    adapter_error=adapter_error,
                )
        if adapter_error is not None:
            raise adapter_error
        return sdk_result

    def _build_existing_terminal_turn_recovery_callback(
        self,
        prepared: _PreparedCodexTurn,
        store: CodexSessionStore,
        active_job: InvestigationJobState | Mapping[str, Any],
    ) -> Callable[[Any, str], None]:
        """构造补丁前失败 turn 的 thread-aware rollback 栅栏。

        输入是已经通过初始 Investigation 校验的 prepared turn、会话 store 和原作业
        waiting state；动作在新模型 turn 前重新核对旧 trace，再复用同一持久 rollback
        合同清空失败历史。它不触碰 MiniChem 作业；trace、manifest 或作业身份不一致
        时直接 fail-closed，不允许先提交新的模型 turn。
        """

        if not isinstance(active_job, InvestigationJobState):
            try:
                active_job = InvestigationJobState.from_mapping(active_job)
            except (TypeError, ValueError) as exc:
                raise HarnessError(
                    "legacy Investigation recovery active job is invalid",
                    kind="session_state",
                ) from exc
        if prepared.stage != "initial_investigation" or prepared.turn_count_before != 0:
            raise HarnessError(
                "legacy Investigation recovery requires an untouched initial session",
                kind="session_state",
            )
        owner = store.owner
        rollback_receipt = store.load_thread_rollback_receipt()
        if rollback_receipt is not None:
            receipt_identity = {
                "stage": prepared.stage,
                "thread_id": active_job.codex_thread_id,
                "request_digest": prepared.request_digest,
                "thread_policy_digest": prepared.thread_policy_digest,
                "turn_count": prepared.turn_count_before,
                "provider_error_code": "invalid_encrypted_content",
            }
            if any(
                rollback_receipt.get(field) != value
                for field, value in receipt_identity.items()
            ):
                raise HarnessError(
                    "legacy Investigation rollback receipt identity drifted",
                    kind="session_state",
                )
            # 已有回执表示此前的失败物理历史已清空；回调仍会调用统一 helper，
            # helper 在空 history 上只做幂等核对，不会重复请求 rollback。
            expected_code = rollback_receipt["provider_error_code"]
        else:
            expected_code = self._read_legacy_initial_failure_code(
                owner=owner,
                thread_id=active_job.codex_thread_id,
                turn_id=active_job.turn_id,
            )
        if expected_code != "invalid_encrypted_content":
            raise HarnessError(
                "legacy Investigation recovery has no approved provider failure evidence",
                kind="session_state",
            )
        rollback_callback = self._build_investigation_rollback_callback(prepared, store)

        def recover_existing_terminal_turn(thread: Any, thread_id: str) -> None:
            """在新 request-intent 前复用官方 rollback，随后才允许模型重试。"""

            if thread_id != active_job.codex_thread_id:
                raise HarnessError(
                    "legacy Investigation recovery thread identity does not match waiting job",
                    kind="session_state",
                )
            rollback_callback(thread_id, thread, expected_code)

        return recover_existing_terminal_turn

    def _build_investigation_rollback_callback(
        self,
        prepared: _PreparedCodexTurn,
        store: CodexSessionStore,
    ) -> Callable[[str, Any, str], None]:
        """构造 Investigation turn 的持久 rollback 回调。"""

        def perform_investigation_rollback(
            thread_id: str,
            thread: Any,
            error_code: str,
        ) -> None:
            """先写 intent，再调用官方 rollback，最后写 receipt。"""

            expected_thread_id = prepared.session_id
            if expected_thread_id is None:
                manifest = store.load()
                expected_thread_id = None if manifest is None else manifest.get("thread_id")
            if thread_id != expected_thread_id:
                raise HarnessError(
                    "Investigation turn rollback thread differs from session",
                    kind="session_state",
                )
            if prepared.turn_count_before != 0:
                raise HarnessError(
                    "Investigation turn rollback requires zero successful turns",
                    kind="session_state",
                )
            identity = {
                "stage": prepared.stage,
                "thread_id": thread_id,
                "request_digest": prepared.request_digest,
                "thread_policy_digest": prepared.thread_policy_digest,
                "turn_count": prepared.turn_count_before,
                "provider_error_code": error_code,
            }
            receipt = store.load_thread_rollback_receipt()
            if receipt is not None:
                if any(receipt.get(key) != value for key, value in identity.items()):
                    raise HarnessError(
                        "Investigation turn rollback receipt identity drifted",
                        kind="session_state",
                    )
                # receipt 只证明上一次失败 history 已经清空。若同一有界自动
                # 重试再次产生了一个失败物理 turn，必须用原 intent 的数量再次
                # 执行官方 rollback；仅返回 receipt 会把新的失败 turn 留在 thread
                # 中，后续恢复也无法再安全提交。
                rollback_result = rollback_codex_thread_for_investigation_turn(
                    thread=thread,
                    thread_id=thread_id,
                    successful_turn_count=prepared.turn_count_before,
                    expected_physical_turn_count=int(receipt["physical_turn_count"]),
                )
                store.record_thread_rollback_receipt(
                    **identity,
                    physical_turn_count=int(rollback_result["num_turns"]),
                )
                return
            intent = store.load_thread_rollback_intent()
            if intent is not None:
                if any(intent.get(key) != value for key, value in identity.items()):
                    raise HarnessError(
                        "Investigation turn rollback intent identity drifted",
                        kind="session_state",
                    )
                physical_turn_count = int(intent["physical_turn_count"])
            else:
                physical_turn_count = 0

            def save_intent(turn_count: int) -> None:
                """在任何官方副作用前保存可重启的物理 turn 数。"""

                nonlocal physical_turn_count
                physical_turn_count = turn_count
                if intent is None:
                    store.record_thread_rollback_intent(
                        **identity,
                        physical_turn_count=turn_count,
                    )

            rollback_result = rollback_codex_thread_for_investigation_turn(
                thread=thread,
                thread_id=thread_id,
                successful_turn_count=prepared.turn_count_before,
                expected_physical_turn_count=(
                    physical_turn_count if intent is not None else None
                ),
                before_rollback=save_intent,
            )
            physical_turn_count = int(rollback_result["num_turns"])
            store.record_thread_rollback_receipt(
                **identity,
                physical_turn_count=physical_turn_count,
            )

        return perform_investigation_rollback

    # 兼容已有测试和旧调用方；新路径按 Investigation turn 命名。
    _build_wake_rollback_callback = _build_investigation_rollback_callback

    def _sanitize_failed_adapter_error(
        self,
        trace_path: Path,
        error: BaseException,
    ) -> BaseException:
        """清理失败 turn 的 trace，并移除异常链中的 provider/MCP secret。"""

        try:
            self._sanitize_trace_file(trace_path)
        except BaseException as sanitize_error:
            return sanitize_error
        return _sanitize_runtime_error_chain(error, self._sensitive_values)

    @staticmethod
    def _recovery_reason(error: BaseException) -> str:
        """把 SDK recovery 错误归一化为有限的持久化原因枚举。"""

        reason = getattr(error, "reason", None)
        if reason in {
            "worker_cleanup",
            "completion_receipt",
            "request_intent",
            "terminal_turn",
            "goal",
        }:
            return str(reason)
        return "worker_cleanup"

    @staticmethod
    def _resolve_recovery_request_digest(
        callbacks: _CodexTurnCallbacks,
        error: BaseException,
        reason: str,
    ) -> str:
        """返回应写入 recovery 的原请求摘要。

        worker/receipt 恢复绑定当前请求。request-intent 与 terminal-turn 恢复必须
        使用 adapter 从磁盘事实读取的原摘要；字段缺失或非法时抛出
        recovery_pending，不会把当前请求摘要写成旧请求身份。函数不修改 store。
        """

        if reason not in {"request_intent", "terminal_turn"}:
            return callbacks.request_digest
        intent_digest = getattr(error, "request_digest", None)
        if not isinstance(intent_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", intent_digest
        ) is None:
            raise HarnessError(
                "Codex recovery has no valid original request digest",
                kind="recovery_pending",
            )
        return intent_digest

    def _persist_worker_recovery_after_error(
        self,
        *,
        callbacks: _CodexTurnCallbacks,
        original_error: BaseException,
        adapter_error: BaseException,
    ) -> BaseException:
        """持久化 worker recovery；二次失败时写 poison 并保持阻断分类。"""

        reason = self._recovery_reason(original_error)
        turn_id = getattr(original_error, "turn_id", None)
        try:
            recovery_request_digest = self._resolve_recovery_request_digest(
                callbacks,
                original_error,
                reason,
            )
        except BaseException as recovery_identity_error:
            return recovery_identity_error
        try:
            callbacks.store.record_recovery_pending(
                stage=callbacks.stage,
                request_digest=recovery_request_digest,
                turn_id=turn_id,
                reason=reason,
                thread_policy_digest=callbacks.thread_policy_digest,
            )
        except BaseException:
            try:
                callbacks.store.record_recovery_poison(
                    stage=callbacks.stage,
                    request_digest=recovery_request_digest,
                    turn_id=turn_id,
                    reason=reason,
                    thread_policy_digest=callbacks.thread_policy_digest,
                )
            except BaseException:
                pass
            if getattr(adapter_error, "kind", None) != "recovery_pending":
                adapter_error = _sanitize_runtime_error_chain(
                    original_error,
                    self._sensitive_values,
                )
            if getattr(adapter_error, "kind", None) != "recovery_pending":
                return HarnessError(
                    "Codex recovery remains pending after persistence failure",
                    kind="recovery_pending",
                )
        return adapter_error

    def _validate_adapter_result(
        self,
        prepared: _PreparedCodexTurn,
        callbacks: _CodexTurnCallbacks,
        sdk_result: Any,
    ) -> tuple[str, str]:
        """核对 adapter 返回的类型、thread、turn 和两个摘要。"""

        if not isinstance(sdk_result, CodexSdkRunResult):
            raise HarnessError("Codex adapter returned an invalid result", kind="session_protocol")
        returned_thread_id = _require_identifier(sdk_result.thread_id, "adapter.thread_id")
        _assert_sensitive_values_absent(
            returned_thread_id,
            self._sensitive_values,
            "adapter.thread_id",
        )
        registered_thread_id = callbacks.registered_thread_id
        if registered_thread_id is None:
            raise HarnessError(
                "Codex adapter did not register thread before turn",
                kind="session_state",
            )
        if returned_thread_id != registered_thread_id:
            raise HarnessError(
                "Codex adapter returned a thread different from the registered thread",
                kind="session_protocol",
            )
        if prepared.session_id is not None and returned_thread_id != prepared.session_id:
            raise HarnessError(
                "Codex resume returned a different thread_id",
                kind="session_protocol",
            )
        turn_id = sdk_result.turn_id or sdk_result.response.metadata.get("turn_id")
        turn_id = _require_identifier(turn_id, "adapter.turn_id")
        _assert_sensitive_values_absent(turn_id, self._sensitive_values, "adapter.turn_id")
        if sdk_result.request_digest and sdk_result.request_digest != prepared.request_digest:
            raise HarnessError(
                "adapter request digest does not match the current request",
                kind="session_protocol",
            )
        if (
            sdk_result.thread_policy_digest
            and sdk_result.thread_policy_digest != prepared.thread_policy_digest
        ):
            raise HarnessError(
                "adapter thread policy digest does not match the current config",
                kind="session_protocol",
            )
        goal_fields_present = bool(
            sdk_result.goal_status
            or sdk_result.goal_id
            or sdk_result.logical_goal_id
            or sdk_result.physical_turn_ids
        )
        if goal_fields_present and not sdk_result.goal_status:
            raise HarnessError(
                "Investigation Goal identity is incomplete",
                kind="session_protocol",
            )
        if sdk_result.goal_status:
            if prepared.request.role != "investigation":
                raise HarnessError(
                    "ordinary Codex profile returned a Goal result",
                    kind="session_protocol",
                )
            if sdk_result.goal_status != "complete":
                raise HarnessError(
                    "Investigation Goal result is not complete",
                    kind="recovery_pending",
                )
            logical_goal_id = _require_identifier(
                sdk_result.logical_goal_id,
                "adapter.logical_goal_id",
            )
            goal_id = _require_identifier(sdk_result.goal_id, "adapter.goal_id")
            _assert_sensitive_values_absent(goal_id, self._sensitive_values, "adapter.goal_id")
            physical_turn_ids = tuple(sdk_result.physical_turn_ids)
            if not physical_turn_ids or any(
                not isinstance(item, str) or not item.strip()
                for item in physical_turn_ids
            ):
                raise HarnessError(
                    "Investigation Goal physical turn identities are invalid",
                    kind="session_protocol",
                )
            if len(set(physical_turn_ids)) != len(physical_turn_ids):
                raise HarnessError(
                    "Investigation Goal physical turn identities are not unique",
                    kind="session_protocol",
                )
            if physical_turn_ids[0] != turn_id:
                raise HarnessError(
                    "Investigation Goal physical turn range does not start at turn_id",
                    kind="session_protocol",
                )
            if (
                not isinstance(sdk_result.physical_turn_start, str)
                or not sdk_result.physical_turn_start.strip()
                or not isinstance(sdk_result.physical_turn_end, str)
                or not sdk_result.physical_turn_end.strip()
                or sdk_result.physical_turn_start != physical_turn_ids[0]
                or sdk_result.physical_turn_end != physical_turn_ids[-1]
            ):
                raise HarnessError(
                    "Investigation Goal physical turn endpoints are invalid",
                    kind="session_protocol",
                )
        return returned_thread_id, turn_id

    def _record_adapter_completion(
        self,
        prepared: _PreparedCodexTurn,
        store: CodexSessionStore,
        sdk_result: CodexSdkRunResult,
    ) -> None:
        """按 receipt 的 turn_count_before 补记一次完成，避免重放重复计数。"""

        if not sdk_result.replayed:
            store.record_completed_turn(stage=prepared.stage, status="completed")
            return
        receipt_count_before = sdk_result.receipt_turn_count_before
        if (
            isinstance(receipt_count_before, bool)
            or not isinstance(receipt_count_before, int)
            or receipt_count_before < 0
        ):
            raise HarnessError(
                "adapter replay receipt count is invalid",
                kind="session_protocol",
            )
        if prepared.turn_count_before == receipt_count_before:
            store.record_completed_turn(stage=prepared.stage, status="completed")
            return
        if prepared.turn_count_before != receipt_count_before + 1:
            raise HarnessError(
                "adapter replay receipt does not match session turn_count",
                kind="session_protocol",
            )
        if (
            prepared.recovery is not None
            and prepared.recovery.get("reason") == "request_intent"
        ):
            store.clear_request_intent_recovery_after_replayed_receipt(
                stage=prepared.stage,
                thread_id=sdk_result.thread_id,
                receipt_turn_count_before=receipt_count_before,
                thread_policy_digest=prepared.thread_policy_digest,
            )

    def _decode_completed_response(
        self,
        prepared: _PreparedCodexTurn,
        store: CodexSessionStore,
        turn_id: str,
        sdk_result: CodexSdkRunResult,
        validator: Callable[[Mapping[str, Any]], Any],
        expected_result_type: type[Any] | tuple[type[Any], ...],
    ) -> InvestigationResponse | TaskReviewResponse | TaskFinalizationResponse:
        """在真实完成计数后验证响应；失败时保存持久合同阻断。"""

        try:
            _assert_sensitive_values_absent(
                {
                    "content": sdk_result.response.content,
                    "model": sdk_result.response.model,
                    "usage": sdk_result.response.usage,
                    "raw_trace_path": sdk_result.response.raw_trace_path,
                    "metadata": sdk_result.response.metadata,
                },
                self._sensitive_values,
                "Codex response",
            )
            return self._decode_and_validate_response(
                sdk_result.response.content,
                validator=validator,
                expected_result_type=expected_result_type,
            )
        except BaseException as contract_error:
            try:
                store.record_contract_failure(
                    stage=prepared.stage,
                    request_digest=prepared.request_digest,
                    turn_id=turn_id,
                    thread_policy_digest=prepared.thread_policy_digest,
                )
            except BaseException as recovery_error:
                raise recovery_error from contract_error
            raise

    @staticmethod
    def _rebuild_replayed_job_state(
        callbacks: _CodexTurnCallbacks,
        completed_tool_items: Sequence[Any],
    ) -> None:
        """用 receipt 对应的完整工具历史重建异步作业状态。

        输入是已带 sidecar 作业身份的 tracker 和 adapter 从同一 completed turn
        history 提取的 MCP 工具项；动作从空 tracker 按原顺序重放工具项，但临时关闭
        submit/collect 外部回调，避免重复写 sidecar 或 WM。返回值为 None，重建后的
        最后一个作业必须与 sidecar 的完整身份一致；缺少 submit、历史截断或身份漂移
        时抛出 HarnessError。该方法不调用模型、MiniChem 或 Controller。
        """

        expected_receipt = callbacks.active_job_receipt
        if expected_receipt is None:
            raise HarnessError(
                "waiting receipt replay has no active job identity",
                kind="session_state",
            )
        accepted_callback = callbacks.job_accepted_callback
        collected_callback = callbacks.job_collected_callback
        callbacks.job_accepted_callback = None
        callbacks.job_collected_callback = None
        callbacks.accepted_job_receipts.clear()
        callbacks.completed_job_receipts.clear()
        callbacks.active_job_receipt = None
        callbacks.active_job_terminal = False
        callbacks.active_job_collected = False
        try:
            for item in completed_tool_items:
                callbacks.capture_tool_completed(item)
        finally:
            callbacks.job_accepted_callback = accepted_callback
            callbacks.job_collected_callback = collected_callback
        rebuilt_receipt = callbacks.active_job_receipt
        if rebuilt_receipt is None:
            raise HarnessError(
                "waiting receipt replay lacks its submitted job tool item",
                kind="session_state",
            )
        if (
            rebuilt_receipt.identity.as_mapping()
            != expected_receipt.identity.as_mapping()
        ):
            raise HarnessError(
                "waiting receipt replay active job identity drifted",
                kind="session_state",
            )

    @staticmethod
    def _build_deferred_receipt_handle(
        prepared: _PreparedCodexTurn,
        callbacks: _CodexTurnCallbacks,
        sdk_result: CodexSdkRunResult,
        *,
        thread_id: str,
        turn_id: str,
        handoff_kind: str,
    ) -> CodexCompletedTurnReceiptHandle | CodexGoalReceiptHandle:
        """为成功 turn 构造延迟 receipt 句柄，不在 runtime 内确认 receipt。

        输入是已核对的 SDK 结果、会话和外层 handoff 类型；动作只读取当前 turn 的
        completed receipt。每个成功 turn（包括测试替身）都必须提供匹配 receipt；
        waiting replay 和显式 completed-turn replay 允许 receipt 的原始 request
        digest 与当前重建请求不同，但 thread、turn、策略和计数必须逐值一致；
        其它漂移直接 fail-closed。
        """

        if sdk_result.goal_status:
            goal_handle = build_codex_goal_receipt_handle(
                prepared.completed_turn_receipt_root,
                owner=callbacks.store.owner.identity,
                session_store=callbacks.store,
                thread_id=thread_id,
                turn_id=turn_id,
                request_digest=(sdk_result.request_digest or prepared.request_digest),
                objective_digest=hashlib.sha256(
                    prepared.request.prompt_text().encode("utf-8")
                ).hexdigest(),
                thread_policy_digest=prepared.thread_policy_digest,
                turn_count_before=(
                    sdk_result.receipt_turn_count_before
                    if sdk_result.replayed and sdk_result.receipt_turn_count_before is not None
                    else prepared.turn_count_before
                ),
                handoff_kind=handoff_kind,
            )
            expected_physical_ids = tuple(sdk_result.physical_turn_ids)
            if (
                goal_handle.thread_id != thread_id
                or goal_handle.turn_id != turn_id
                or goal_handle.goal_id != sdk_result.goal_id
                or goal_handle.logical_goal_id != sdk_result.logical_goal_id
                or goal_handle.physical_turn_ids != expected_physical_ids
            ):
                raise HarnessError(
                    "Goal receipt identity does not match the returned operation",
                    kind="session_state",
                )
            return goal_handle
        canonical_receipt_root = (
            callbacks.store.manifest_path.parent / "completed_turn_receipts"
        ).resolve()
        if prepared.completed_turn_receipt_root.resolve() != canonical_receipt_root:
            raise HarnessError(
                "completed receipt root does not match the session owner",
                kind="session_state",
            )
        receipt = read_codex_completed_turn_receipt(
            prepared.completed_turn_receipt_root,
            thread_id=thread_id,
            turn_id=turn_id,
            thread_policy_digest=prepared.thread_policy_digest,
            read_only=True,
        )
        if receipt is None:
            raise HarnessError(
                "completed Codex turn has no durable receipt",
                kind="session_state",
            )
        if receipt["thread_id"] != thread_id or receipt["turn_id"] != turn_id:
            raise HarnessError(
                "completed receipt identity does not match the returned turn",
                kind="session_state",
            )
        expected_count = (
            sdk_result.receipt_turn_count_before
            if sdk_result.replayed
            else prepared.turn_count_before
        )
        if expected_count is None or receipt["turn_count_before"] != expected_count:
            raise HarnessError(
                "completed receipt turn count does not match the returned turn",
                kind="session_state",
            )
        replayed_waiting = bool(sdk_result.replayed and callbacks.active_job_receipt is not None)
        replayed_explicit = bool(
            sdk_result.replayed and prepared.replay_turn_id == turn_id
        )
        if not sdk_result.replayed or not (replayed_waiting or replayed_explicit):
            if receipt["request_digest"] != prepared.request_digest:
                raise HarnessError(
                    "completed receipt request digest does not match the returned turn",
                    kind="session_state",
                )
        owner_identity = callbacks.store.owner.identity
        return build_codex_completed_turn_receipt_handle(
            prepared.completed_turn_receipt_root,
            owner=owner_identity,
            session_store=callbacks.store,
            thread_id=receipt["thread_id"],
            turn_id=receipt["turn_id"],
            request_digest=receipt["request_digest"],
            turn_count_before=receipt["turn_count_before"],
            thread_policy_digest=receipt["thread_policy_digest"],
            handoff_kind=handoff_kind,
        )

    def _complete_codex_turn(
        self,
        *,
        prepared: _PreparedCodexTurn,
        callbacks: _CodexTurnCallbacks,
        sdk_result: Any,
        store: CodexSessionStore,
        validator: Callable[[Mapping[str, Any]], Any],
        expected_result_type: type[Any] | tuple[type[Any], ...],
    ) -> InvestigationTurnResult:
        """完成 adapter 合同、计数、响应验证和旧 recovery 清理。"""

        returned_thread_id, turn_id = self._validate_adapter_result(
            prepared,
            callbacks,
            sdk_result,
        )
        self._record_adapter_completion(prepared, store, sdk_result)
        replayed = bool(getattr(sdk_result, "replayed", False))
        replayed_waiting_receipt = replayed and callbacks.active_job_receipt is not None
        if replayed_waiting_receipt:
            # completed receipt 只证明 provider turn 已完成，不证明该 turn 最后提交的
            # 异步作业已经 collect。adapter 已从同一官方 history 返回完整工具项；
            # 在不触发外部生命周期回调的前提下重建 tracker，保留真实 waiting 状态。
            self._rebuild_replayed_job_state(
                callbacks,
                tuple(getattr(sdk_result, "completed_tool_items", ())),
            )
        elif not callbacks.accepted_job_receipts:
            # 兼容只在 turn 完成时暴露 item 的测试/适配器；仍按真实 item 顺序
            # 重放同一个 tracker，不能只提取 submit 或依赖最终 assistant 文本。
            for item in getattr(sdk_result, "completed_tool_items", ()):
                callbacks.capture_tool_completed(item)
        receipts = tuple(callbacks.accepted_job_receipts)
        if (
            prepared.stage == "investigation_job_wake"
            and prepared.request.metadata.get("terminal_status") != "unknown"
            and not callbacks.accepted_job_receipts
            and callbacks.active_job_receipt is not None
            and not callbacks.active_job_collected
        ):
            raise HarnessError(
                "wake turn 必须先 get 并 collect 原 job，不能直接输出科学 JSON",
                kind="response_contract",
            )
        accepted_job: Mapping[str, Any] | None = None
        active_job_uncollected = (
            callbacks.active_job_receipt is not None
            and not callbacks.active_job_collected
        )
        completed_jobs = tuple(
            receipt.as_mapping() for receipt in callbacks.completed_job_receipts
        )
        if active_job_uncollected or receipts and (
            callbacks.active_job_receipt is None
            or not callbacks.active_job_collected
        ):
            if prepared.request.role != "investigation":
                raise HarnessError(
                    "task review turn cannot submit an asynchronous calculation",
                    kind="response_contract",
                )
            # 当前 turn 仍有未 collect 的活跃 job 时，assistant 正文不是科学结果；
            # waiting 状态只由真实回执和生命周期驱动，不能从正文推断身份。
            validated_result = None
            active_receipt = callbacks.active_job_receipt or receipts[-1]
            accepted_job = active_receipt.as_mapping()
            handoff_kind = "waiting"
        else:
            # 所有 submit 都已经经过匹配的 terminal get 和 collect；此时才允许
            # 解析最终 JSON，即使本 turn 曾经有多个真实 submit 回执。
            validated_result = self._decode_completed_response(
                prepared,
                store,
                turn_id,
                sdk_result,
                validator,
                expected_result_type,
            )
            handoff_kind = "result"
        receipt_handle = self._build_deferred_receipt_handle(
            prepared,
            callbacks,
            sdk_result,
            thread_id=returned_thread_id,
            turn_id=turn_id,
            handoff_kind=handoff_kind,
        )
        if prepared.recovery is not None and store.load_recovery() is not None:
            store.record_validated_turn(
                stage=prepared.stage,
                request_digest=prepared.request_digest,
                turn_id=turn_id,
                thread_policy_digest=prepared.thread_policy_digest,
            )
            store.clear_recovery_after_validated_turn(
                stage=prepared.stage,
                status="completed",
                thread_policy_digest=prepared.thread_policy_digest,
            )
        return InvestigationTurnResult(
            result=validated_result,
            thread_id=returned_thread_id,
            turn_id=turn_id,
            # 普通 turn 的 receipt 摘要属于 Codex 请求层，不能覆盖由
            # Investigation backend 持久化的 attempt 摘要。只有只读重放旧
            # completed turn 时，backend 才需要原 receipt 摘要核对旧 attempt。
            request_digest=(
                receipt_handle.request_digest
                if replayed
                else None
            ),
            accepted_job=accepted_job,
            completed_jobs=completed_jobs,
            receipt_handle=receipt_handle,
        )

    def _decode_and_validate_response(
        self,
        content: Any,
        *,
        validator: Callable[[Mapping[str, Any]], Any],
        expected_result_type: type[Any] | tuple[type[Any], ...],
    ) -> InvestigationResponse | TaskReviewResponse | TaskFinalizationResponse:
        """严格解码 raw JSON 并用指定领域 validator 验证，不猜测或修复字段。"""

        if not isinstance(content, str):
            raise HarnessError("Codex response content must be JSON text", kind="response_contract")
        _assert_sensitive_values_absent(content, self._sensitive_values, "Codex response")
        try:
            decoded = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise HarnessError("Codex response is not valid JSON", kind="response_contract") from exc
        if not isinstance(decoded, Mapping):
            raise HarnessError("Codex response must be a JSON object", kind="response_contract")
        try:
            result = validator(decoded)
        except Exception as exc:
            raise HarnessError("Codex response violates the structured contract", kind="response_contract") from exc
        if not isinstance(result, expected_result_type):
            raise HarnessError("Codex response has the wrong structured contract", kind="response_contract")
        _assert_sensitive_values_absent(
            result.model_dump(mode="json"),
            self._sensitive_values,
            "Codex response",
        )
        return result
