"""Investigation 恢复所需的只读身份合同。

本模块只描述恢复时需要互相核对的身份，不保存科学结果、原始日志、路径正文或认证
内容。提交 turn 的请求摘要和 Codex turn 摘要使用不同字段，调用方不能用其中一个代替
另一个。所有 ``from_mapping`` 都拒绝未知字段，便于 sidecar 和 Working Memory 在
身份串读时停止推进。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


__all__ = [
    "ResumeContractError",
    "ResumeAction",
    "ResumeDecision",
    "ResumeSubject",
    "TurnChainCheckpoint",
    "ExecutionCheckpoint",
    "ProviderCheckpoint",
    "CapabilityCheckpoint",
    "InvestigationResumeCheckpoint",
    "ResumeCheckpoint",
    "canonical_json_digest",
    "compute_scope_credential_generation_digest",
    "compute_resume_record_digest",
    "compute_resume_record_auth_digest",
]


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class ResumeContractError(ValueError):
    """恢复身份缺失、重复、漂移或混用时抛出的合同错误。"""


def _text(value: Any, label: str) -> str:
    """校验恢复合同中的非空文本，并拒绝控制字符。"""

    if not isinstance(value, str) or not value.strip():
        raise ResumeContractError(f"{label} 必须是非空字符串")
    normalized = value.strip()
    if _CONTROL_RE.search(normalized):
        raise ResumeContractError(f"{label} 不能包含控制字符")
    return normalized


def _digest(value: Any, label: str) -> str:
    """校验小写 SHA-256 摘要。"""

    normalized = _text(value, label)
    if _DIGEST_RE.fullmatch(normalized) is None:
        raise ResumeContractError(f"{label} 必须是小写 SHA-256")
    return normalized


def _optional_text(value: Any, label: str) -> str | None:
    """校验可选文本；显式 ``None`` 表示该旧记录没有保存该字段。"""

    if value is None:
        return None
    return _text(value, label)


def _optional_digest(value: Any, label: str) -> str | None:
    """校验可选摘要。"""

    if value is None:
        return None
    return _digest(value, label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """确认输入是 JSON object。"""

    if not isinstance(value, Mapping):
        raise ResumeContractError(f"{label} 必须是 JSON object")
    return value


def _strict_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    """拒绝合同 object 中未声明的字段。"""

    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ResumeContractError(f"{label} 含未知字段: {', '.join(unknown)}")


def _optional_int(value: Any, label: str) -> int | None:
    """校验可选的非负整数。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResumeContractError(f"{label} 必须是非负整数")
    return value


class ResumeAction(str, Enum):
    """恢复入口允许的有限动作集合。"""

    REPLAY_COMPLETED_RECEIPT = "replay_completed_receipt"
    CLOSE_ORPHANED_ATTEMPT = "close_orphaned_attempt"
    QUERY_EXISTING_JOB = "query_existing_job"
    COLLECT_EXISTING_JOB = "collect_existing_job"
    CONTINUE_SESSION = "continue_session"
    CLEANUP_ONLY = "cleanup_only"
    RETURN_WAITING = "return_waiting"
    BLOCK = "block"

    # 兼容早期调用方的枚举别名；序列化时使用上面的明确协议值。
    REPLAY_CHECKPOINT = "replay_completed_receipt"
    CONTINUE_ACTIVE = "continue_session"
    STOP = "block"


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    """一次只读恢复核验给出的有限决定。

    ``allow_resubmit`` 永远默认为 ``False``，本合同没有自动重新提交计算的动作。
    """

    action: ResumeAction
    reason_code: str
    allow_resubmit: bool = False

    def __post_init__(self) -> None:
        """校验决定只能使用白名单动作和非空原因。"""

        if not isinstance(self.action, ResumeAction):
            try:
                object.__setattr__(self, "action", ResumeAction(self.action))
            except (TypeError, ValueError) as exc:
                raise ResumeContractError("resume decision action 非法") from exc
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))
        if self.allow_resubmit is not False:
            raise ResumeContractError("恢复决定禁止 resubmit")

    def as_mapping(self) -> dict[str, Any]:
        """返回不含科学内容的决定 object。"""

        return {
            "action": self.action.value,
            "reason_code": self.reason_code,
            "allow_resubmit": False,
        }


@dataclass(frozen=True, slots=True)
class ResumeSubject:
    """Investigation 组合和 turn 链的稳定主体身份。"""

    task_id: str
    batch_id: str
    combination_id: str
    session_id: str
    stage: str
    scope_id: str
    codex_thread_id: str
    logical_operation_instance_id: str
    attempt_operation_instance_id: str
    attempt_number: int
    workspace_ref_digest: str
    repair_request_number: int | None = None
    repair_request_event_ref: str | None = None
    expected_result_version: str | None = None

    _FIELDS = frozenset(
        {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "stage",
            "scope_id",
            "codex_thread_id",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "attempt_number",
            "repair_request_number",
            "workspace_ref_digest",
            "repair_request_event_ref",
            "expected_result_version",
        }
    )

    def __post_init__(self) -> None:
        """校验 task、batch、组合、session、scope、thread 和 attempt 不可混用。"""

        for field_name in (
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "scope_id",
            "codex_thread_id",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        if self.session_id == "unknown":
            raise ResumeContractError("canonical resume subject 不得使用 unknown session_id")
        stage = _text(self.stage, "stage")
        if stage not in {"initial", "repair"}:
            raise ResumeContractError("stage 只能是 initial 或 repair")
        object.__setattr__(self, "stage", stage)
        if isinstance(self.attempt_number, bool) or not isinstance(self.attempt_number, int):
            raise ResumeContractError("attempt_number 必须是正整数")
        if self.attempt_number < 1:
            raise ResumeContractError("attempt_number 必须是正整数")
        repair_number = _optional_int(self.repair_request_number, "repair_request_number")
        if stage == "repair" and (repair_number is None or repair_number < 1):
            raise ResumeContractError("repair 阶段必须有正的 repair_request_number")
        if stage == "initial" and repair_number is not None:
            raise ResumeContractError("initial 阶段不能携带 repair_request_number")
        object.__setattr__(self, "repair_request_number", repair_number)
        object.__setattr__(
            self,
            "workspace_ref_digest",
            _digest(self.workspace_ref_digest, "workspace_ref_digest"),
        )
        object.__setattr__(
            self,
            "repair_request_event_ref",
            _optional_text(self.repair_request_event_ref, "repair_request_event_ref"),
        )
        object.__setattr__(
            self,
            "expected_result_version",
            _optional_text(self.expected_result_version, "expected_result_version"),
        )
        if stage == "repair" and (
            self.repair_request_event_ref is None
            or self.expected_result_version is None
        ):
            raise ResumeContractError(
                "repair 阶段必须有 repair_request_event_ref 和 expected_result_version"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str = "resume_subject") -> "ResumeSubject":
        """从严格 object 恢复主体身份。"""

        mapping = _mapping(value, label)
        _strict_fields(mapping, set(cls._FIELDS), label)
        required = {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "stage",
            "scope_id",
            "codex_thread_id",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "attempt_number",
            "workspace_ref_digest",
        }
        if not required.issubset(mapping):
            missing = sorted(required - set(mapping))
            raise ResumeContractError(f"{label} 缺少字段: {', '.join(missing)}")
        return cls(**{field: mapping.get(field) for field in cls._FIELDS if field in mapping})

    def as_mapping(self) -> dict[str, Any]:
        """返回可写入 sidecar/WM 的主体身份。"""

        result: dict[str, Any] = {
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "combination_id": self.combination_id,
            "session_id": self.session_id,
            "stage": self.stage,
            "scope_id": self.scope_id,
            "codex_thread_id": self.codex_thread_id,
            "logical_operation_instance_id": self.logical_operation_instance_id,
            "attempt_operation_instance_id": self.attempt_operation_instance_id,
            "attempt_number": self.attempt_number,
            "workspace_ref_digest": self.workspace_ref_digest,
        }
        if self.repair_request_number is not None:
            result["repair_request_number"] = self.repair_request_number
        if self.repair_request_event_ref is not None:
            result["repair_request_event_ref"] = self.repair_request_event_ref
        if self.expected_result_version is not None:
            result["expected_result_version"] = self.expected_result_version
        return result


@dataclass(frozen=True, slots=True)
class TurnChainCheckpoint:
    """区分提交 checkpoint turn 与当前活动 continuation turn。"""

    submit_turn_id: str
    submit_turn_started_event_ref: str
    active_continuation_turn_id: str | None = None
    active_continuation_turn_started_event_ref: str | None = None

    _FIELDS = frozenset(
        {
            "submit_turn_id",
            "submit_turn_started_event_ref",
            "active_continuation_turn_id",
            "active_continuation_turn_started_event_ref",
        }
    )

    def __post_init__(self) -> None:
        """校验提交 turn 与活动 continuation turn 的事件身份不能相互替换。"""

        object.__setattr__(self, "submit_turn_id", _text(self.submit_turn_id, "submit_turn_id"))
        object.__setattr__(
            self,
            "submit_turn_started_event_ref",
            _text(self.submit_turn_started_event_ref, "submit_turn_started_event_ref"),
        )
        continuation_id = _optional_text(self.active_continuation_turn_id, "active_continuation_turn_id")
        continuation_ref = _optional_text(
            self.active_continuation_turn_started_event_ref,
            "active_continuation_turn_started_event_ref",
        )
        if (continuation_id is None) != (continuation_ref is None):
            raise ResumeContractError("active continuation turn 必须同时包含 id 和事件引用")
        if continuation_id == self.submit_turn_id:
            raise ResumeContractError("active continuation turn 不能复用 submit turn")
        if continuation_ref == self.submit_turn_started_event_ref:
            raise ResumeContractError(
                "active continuation turn 不能复用 submit Started event"
            )
        object.__setattr__(self, "active_continuation_turn_id", continuation_id)
        object.__setattr__(self, "active_continuation_turn_started_event_ref", continuation_ref)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str = "turn_chain_checkpoint") -> "TurnChainCheckpoint":
        """从严格 object 恢复 turn 链 checkpoint。"""

        mapping = _mapping(value, label)
        _strict_fields(mapping, set(cls._FIELDS), label)
        required = {"submit_turn_id", "submit_turn_started_event_ref"}
        if not required.issubset(mapping):
            raise ResumeContractError(f"{label} 缺少字段: {', '.join(sorted(required - set(mapping)))}")
        return cls(**{field: mapping.get(field) for field in cls._FIELDS if field in mapping})

    @property
    def checkpoint_turn_id(self) -> str:
        """返回 sidecar 绑定的提交 checkpoint turn。"""

        return self.submit_turn_id

    def as_mapping(self) -> dict[str, Any]:
        """返回显式区分 submit/continuation 的 turn 链 object。"""

        result: dict[str, Any] = {
            "submit_turn_id": self.submit_turn_id,
            "submit_turn_started_event_ref": self.submit_turn_started_event_ref,
        }
        if self.active_continuation_turn_id is not None:
            result.update(
                {
                    "active_continuation_turn_id": self.active_continuation_turn_id,
                    "active_continuation_turn_started_event_ref": self.active_continuation_turn_started_event_ref,
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class ExecutionCheckpoint:
    """一次已提交 MiniChem 作业的只读身份和提交请求摘要。"""

    job_id: str
    submission_operation_id: str
    submission_key: str
    submission_request_digest: str
    job_family: str
    action_id: str | None = None
    backend_id: str | None = None
    submission_operation_fingerprint: str | None = None

    _FIELDS = frozenset(
        {
            "job_id",
            "submission_operation_id",
            "submission_key",
            "submission_request_digest",
            "job_family",
            "action_id",
            "backend_id",
            "submission_operation_fingerprint",
        }
    )

    def __post_init__(self) -> None:
        """校验作业身份字段和 submission request digest。"""

        for field_name in (
            "job_id",
            "submission_operation_id",
            "submission_key",
            "job_family",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "submission_request_digest",
            _digest(self.submission_request_digest, "submission_request_digest"),
        )
        for field_name in ("action_id", "backend_id"):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "submission_operation_fingerprint",
            _optional_digest(
                self.submission_operation_fingerprint,
                "submission_operation_fingerprint",
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str = "execution_checkpoint") -> "ExecutionCheckpoint":
        """从严格 object 恢复作业 checkpoint。"""

        mapping = _mapping(value, label)
        _strict_fields(mapping, set(cls._FIELDS), label)
        required = set(cls._FIELDS) - {
            "action_id",
            "backend_id",
            "submission_operation_fingerprint",
        }
        if not required.issubset(mapping):
            raise ResumeContractError(f"{label} 缺少字段: {', '.join(sorted(required - set(mapping)))}")
        return cls(**{field: mapping.get(field) for field in cls._FIELDS if field in mapping})

    def as_mapping(self) -> dict[str, Any]:
        """返回不含作业结果的 checkpoint object。"""

        result: dict[str, Any] = {
            "job_id": self.job_id,
            "submission_operation_id": self.submission_operation_id,
            "submission_key": self.submission_key,
            "submission_request_digest": self.submission_request_digest,
            "job_family": self.job_family,
        }
        for field_name in (
            "action_id",
            "backend_id",
            "submission_operation_fingerprint",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


@dataclass(frozen=True, slots=True)
class ProviderCheckpoint:
    """Provider 的只读线程/turn 投影，不包含认证或模型正文。"""

    provider_id: str
    codex_thread_id: str
    codex_turn_id: str
    turn_request_digest: str
    thread_policy_digest: str
    turn_count_before: int

    _FIELDS = frozenset(
        {
            "provider_id",
            "codex_thread_id",
            "codex_turn_id",
            "turn_request_digest",
            "thread_policy_digest",
            "turn_count_before",
        }
    )

    def __post_init__(self) -> None:
        """校验 provider、thread、turn 及其专用 digest。"""

        for field_name in ("provider_id", "codex_thread_id", "codex_turn_id"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "turn_request_digest",
            _digest(self.turn_request_digest, "turn_request_digest"),
        )
        object.__setattr__(
            self,
            "thread_policy_digest",
            _digest(self.thread_policy_digest, "thread_policy_digest"),
        )
        if (
            isinstance(self.turn_count_before, bool)
            or not isinstance(self.turn_count_before, int)
            or self.turn_count_before < 0
        ):
            raise ResumeContractError("turn_count_before 必须是非负整数")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str = "provider_checkpoint") -> "ProviderCheckpoint":
        """从严格 object 恢复 provider checkpoint。"""

        mapping = _mapping(value, label)
        _strict_fields(mapping, set(cls._FIELDS), label)
        required = {
            "provider_id",
            "codex_thread_id",
            "codex_turn_id",
            "turn_request_digest",
            "thread_policy_digest",
            "turn_count_before",
        }
        if not required.issubset(mapping):
            raise ResumeContractError(f"{label} 缺少字段: {', '.join(sorted(required - set(mapping)))}")
        return cls(**dict(mapping))

    def as_mapping(self) -> dict[str, Any]:
        """返回 provider 线程/turn 身份。"""

        result: dict[str, Any] = {
            "provider_id": self.provider_id,
            "codex_thread_id": self.codex_thread_id,
            "codex_turn_id": self.codex_turn_id,
            "turn_request_digest": self.turn_request_digest,
            "thread_policy_digest": self.thread_policy_digest,
            "turn_count_before": self.turn_count_before,
        }
        return result


@dataclass(frozen=True, slots=True)
class CapabilityCheckpoint:
    """scope capability 的只读投影，不保存 token 或宿主路径。"""

    scope_id: str
    workspace_ref_digest: str
    scope_credential_generation_digest: str

    _FIELDS = frozenset(
        {
            "scope_id",
            "workspace_ref_digest",
            "scope_credential_generation_digest",
        }
    )

    def __post_init__(self) -> None:
        """校验 scope、workspace 摘要和凭据代次审计摘要。"""

        object.__setattr__(self, "scope_id", _text(self.scope_id, "scope_id"))
        object.__setattr__(self, "workspace_ref_digest", _digest(self.workspace_ref_digest, "workspace_ref_digest"))
        object.__setattr__(
            self,
            "scope_credential_generation_digest",
            _digest(
                self.scope_credential_generation_digest,
                "scope_credential_generation_digest",
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str = "capability_checkpoint") -> "CapabilityCheckpoint":
        """从严格 object 恢复 capability checkpoint。"""

        mapping = _mapping(value, label)
        _strict_fields(mapping, set(cls._FIELDS), label)
        required = {
            "scope_id",
            "workspace_ref_digest",
            "scope_credential_generation_digest",
        }
        if not required.issubset(mapping):
            raise ResumeContractError(f"{label} 缺少字段: {', '.join(sorted(required - set(mapping)))}")
        return cls(**dict(mapping))

    def as_mapping(self) -> dict[str, Any]:
        """返回不含 capability token 的 scope 投影。"""

        result: dict[str, Any] = {
            "scope_id": self.scope_id,
            "workspace_ref_digest": self.workspace_ref_digest,
            "scope_credential_generation_digest": self.scope_credential_generation_digest,
        }
        return result


@dataclass(frozen=True, slots=True)
class InvestigationResumeCheckpoint:
    """把主体、turn 链、作业、provider 和 capability checkpoint 分组。"""

    subject: ResumeSubject
    turn_chain: TurnChainCheckpoint
    execution: ExecutionCheckpoint
    provider: ProviderCheckpoint
    capability: CapabilityCheckpoint

    _FIELDS = frozenset({"subject", "turn_chain", "execution", "provider", "capability"})

    def __post_init__(self) -> None:
        """逐组核对共享 task/session/thread/scope 与 digest 边界。"""

        if not isinstance(self.subject, ResumeSubject):
            raise ResumeContractError("resume checkpoint subject 类型非法")
        if not isinstance(self.turn_chain, TurnChainCheckpoint):
            raise ResumeContractError("resume checkpoint turn_chain 类型非法")
        if not isinstance(self.execution, ExecutionCheckpoint):
            raise ResumeContractError("resume checkpoint execution 类型非法")
        if not isinstance(self.provider, ProviderCheckpoint):
            raise ResumeContractError("resume checkpoint provider 类型非法")
        if not isinstance(self.capability, CapabilityCheckpoint):
            raise ResumeContractError("resume checkpoint capability 类型非法")
        if self.subject.codex_thread_id != self.provider.codex_thread_id:
            raise ResumeContractError("resume checkpoint thread identity 不一致")
        if self.subject.scope_id != self.capability.scope_id:
            raise ResumeContractError("resume checkpoint scope identity 不一致")
        if self.subject.workspace_ref_digest != self.capability.workspace_ref_digest:
            raise ResumeContractError("resume checkpoint workspace identity 不一致")
        active_turn_id = self.turn_chain.active_continuation_turn_id
        if active_turn_id is None:
            if self.turn_chain.submit_turn_id != self.provider.codex_turn_id:
                raise ResumeContractError(
                    "没有 continuation 时 submit turn 必须是 provider 当前 turn"
                )
        elif active_turn_id != self.provider.codex_turn_id:
            raise ResumeContractError(
                "active continuation turn 与 provider 当前 turn identity 不一致"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str = "resume_checkpoint") -> "InvestigationResumeCheckpoint":
        """从严格分组 object 恢复完整 checkpoint。"""

        mapping = _mapping(value, label)
        _strict_fields(mapping, set(cls._FIELDS), label)
        missing = set(cls._FIELDS) - set(mapping)
        if missing:
            raise ResumeContractError(f"{label} 缺少字段: {', '.join(sorted(missing))}")
        return cls(
            subject=ResumeSubject.from_mapping(mapping["subject"], label=f"{label}.subject"),
            turn_chain=TurnChainCheckpoint.from_mapping(mapping["turn_chain"], label=f"{label}.turn_chain"),
            execution=ExecutionCheckpoint.from_mapping(mapping["execution"], label=f"{label}.execution"),
            provider=ProviderCheckpoint.from_mapping(mapping["provider"], label=f"{label}.provider"),
            capability=CapabilityCheckpoint.from_mapping(mapping["capability"], label=f"{label}.capability"),
        )

    def as_mapping(self) -> dict[str, Any]:
        """返回显式分组的只读恢复合同。"""

        return {
            "subject": self.subject.as_mapping(),
            "turn_chain": self.turn_chain.as_mapping(),
            "execution": self.execution.as_mapping(),
            "provider": self.provider.as_mapping(),
            "capability": self.capability.as_mapping(),
        }


ResumeCheckpoint = InvestigationResumeCheckpoint


def canonical_json_digest(value: Any) -> str:
    """为 JSON-compatible 恢复记录生成稳定 SHA-256。"""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ResumeContractError("恢复记录不能生成稳定摘要") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_scope_credential_generation_digest(
    capability: Any,
    *,
    workspace_ref_digest: str | None = None,
) -> str:
    """委托 scope credential 模块计算唯一的无密凭据代次摘要。

    ``workspace_ref_digest`` 仅保留为兼容调用签名；摘要的 canonical 输入和初始
    provision 的授权语义由 ``scope_credentials.scope_credential_generation_digest``
    统一维护。本合同模块不再复制另一套算法，也不把 capability 的路径或 token 写入
    sidecar；调用方应传入 capability/审计 mapping 本身并只持久化返回摘要。
    """

    del workspace_ref_digest
    try:
        from .scope_credentials import scope_credential_generation_digest

        digest = scope_credential_generation_digest(capability)
    except Exception as exc:
        raise ResumeContractError(
            "scope credential generation digest 计算失败"
        ) from exc
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ResumeContractError("scope credential generation digest 非法")
    return digest


def compute_resume_record_digest(record: Mapping[str, Any]) -> str:
    """计算不包含自身及 wake 可变字段的 sidecar 身份摘要。

    wake intent、普通 wake receipt 和 delivered 状态会在同一个作业生命周期中变化；
    replacement 的 source provenance 则是下一作业的不可变来源身份，必须纳入摘要。
    因此 WM waiting 事件引用的摘要不会因合法 wake handoff 状态变化而漂移，但 source
    provenance 被篡改时会 fail-closed。
    """

    mapping = _mapping(record, "record")
    mutable_fields = {
        "wake_status",
        "terminal_status",
        "terminal_message",
        "wake_request",
        "wake_request_digest",
        "wake_receipt",
        "collected_source",
    }
    payload = {
        key: value
        for key, value in mapping.items()
        if key not in {"record_digest", "record_auth_digest"}
        and key not in mutable_fields
    }
    return canonical_json_digest(payload)


def compute_resume_record_auth_digest(record: Mapping[str, Any]) -> str:
    """计算覆盖完整 sidecar envelope 的恢复字段认证摘要。

    ``record_digest`` 只绑定不会随 wake 生命周期变化的作业身份，便于 Working Memory
    引用同一条等待事实；本摘要另外覆盖 wake request、其摘要、终态、receipt 及 source
    provenance。读取 sidecar 时必须重新计算该值，避免攻击者只改决策字段而保留旧身份
    摘要。摘要自身不参与输入，方便每次写入后确定性重算。
    """

    mapping = _mapping(record, "record")
    payload = {
        key: value
        for key, value in mapping.items()
        if key not in {"record_auth_digest"}
    }
    return canonical_json_digest(payload)
