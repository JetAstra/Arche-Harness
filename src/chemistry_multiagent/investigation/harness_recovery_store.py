"""Harness 私有恢复事实的持久化 seam。

``HarnessRecoveryStore`` 只保存 Investigation Harness 恢复所需的身份引用和摘要：
session、provider attempt、MiniChem job、receipt、证据核验和已提交模块输出的
不透明引用。科学结果正文、模型 transcript、scope 凭据和计算产物仍由各自 owner
保存。本模块不读取或写入 Working Memory，也不会根据进程是否消失推断任务失败。

公开接口使用 typed transition；SQLite 中的 JSON 只是实现细节。每次 transition
都会在同一事务中追加不可变事件并更新当前快照，因而进程在事务提交前退出时不会
留下半个状态。重复使用相同 event id 和内容摘要是幂等的，内容漂移或身份漂移会
fail-closed。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sqlite3
from typing import Any, ClassVar, Final, TypeAlias


class HarnessRecoveryStoreError(RuntimeError):
    """Harness 恢复存储无法安全完成操作。"""


class HarnessRecoveryContractError(HarnessRecoveryStoreError, ValueError):
    """输入或持久化合同不合法。"""


class HarnessRecoveryNotFound(HarnessRecoveryStoreError):
    """要求读取的 Harness invocation 尚未持久化。"""


class HarnessRecoveryIdentityConflict(HarnessRecoveryStoreError):
    """Task、Thread、Turn、Scope、Job、Submission 或 Digest 发生漂移。"""


class HarnessRecoveryTransitionConflict(HarnessRecoveryStoreError):
    """同一 event id 的内容漂移，或同一事实被另一个 event id 重复写入。"""


class HarnessRecoveryStateConflict(HarnessRecoveryStoreError):
    """transition 与当前 Harness 生命周期或活动约束不相容。"""


class HarnessLifecycle(Enum):
    """Harness invocation 的四个持久生命周期状态。"""

    OPEN = "open"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


class ProviderAttemptOutcome(Enum):
    """provider attempt 的明确终态；不把进程消失当作其中任何一种。"""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class ReceiptKind(Enum):
    """外部 owner 回执的来源类别。"""

    PROVIDER_TURN = "provider_turn"
    MINICHEM_OPERATION = "minichem_operation"


_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REF_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SAFE_CODE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SCHEMA_VERSION: Final[int] = 2
_LEGACY_SCHEMA_VERSION: Final[int] = 1


def _require_ref(value: Any, field_name: str) -> str:
    """校验不透明引用，拒绝路径、控制字符和空值。"""

    if not isinstance(value, str) or not value or not _SAFE_REF_RE.fullmatch(value):
        raise HarnessRecoveryContractError(
            f"{field_name} 必须是非空且不含路径分隔符的不透明引用"
        )
    return value


def _require_workspace_identity(value: Any) -> str:
    """校验 Run Bundle 内的 workspace 相对身份。

    其他恢复引用必须保持单段不透明标识；workspace_identity 则是由
    Context owner 分配的相对目录身份，可以包含多个 POSIX 路径段。仍拒绝
    绝对路径、父目录和当前目录段，避免把恢复存储指向任务根之外。
    """

    if not isinstance(value, str) or not value:
        raise HarnessRecoveryContractError(
            "workspace_identity 必须是非空相对目录"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise HarnessRecoveryContractError(
            "workspace_identity 不能包含绝对路径或父目录"
        )
    return path.as_posix()


def _require_digest(value: Any, field_name: str) -> str:
    """校验固定格式的 SHA-256 摘要。"""

    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise HarnessRecoveryContractError(f"{field_name} 必须是 64 位小写 SHA-256")
    return value


def _require_code(value: Any, field_name: str) -> str:
    """校验脱敏的结构化原因码。"""

    if not isinstance(value, str) or not _SAFE_CODE_RE.fullmatch(value):
        raise HarnessRecoveryContractError(f"{field_name} 必须是安全的结构化代码")
    return value


def _canonical_json(value: Any) -> str:
    """生成事件摘要使用的确定性 JSON。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    """计算不包含 event id 的 typed transition 内容摘要。"""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_mapping(value: Any, expected: set[str], field_name: str) -> Mapping[str, Any]:
    """读取内部 JSON 时拒绝缺失或额外字段。"""

    if not isinstance(value, Mapping):
        raise HarnessRecoveryContractError(f"{field_name} 必须是 JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"缺少 {', '.join(missing)}")
        if extra:
            details.append(f"含额外字段 {', '.join(extra)}")
        raise HarnessRecoveryContractError(
            f"{field_name} 字段合同不匹配：{'；'.join(details)}"
        )
    return value


@dataclass(frozen=True)
class HarnessRunIdentity:
    """一个 Harness invocation 的不可变公开身份。"""

    research_task_id: str
    module_invocation_id: str
    harness_invocation_id: str
    module_input_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "research_task_id",
            _require_ref(self.research_task_id, "research_task_id"),
        )
        object.__setattr__(
            self,
            "module_invocation_id",
            _require_ref(self.module_invocation_id, "module_invocation_id"),
        )
        object.__setattr__(
            self,
            "harness_invocation_id",
            _require_ref(self.harness_invocation_id, "harness_invocation_id"),
        )
        object.__setattr__(
            self,
            "module_input_digest",
            _require_digest(self.module_input_digest, "module_input_digest"),
        )

@dataclass(frozen=True)
class SessionReference:
    """Investigation session 的稳定引用，不保存 workspace 内容或凭据。"""

    research_task_id: str
    investigation_context_id: str
    hypothesis_execution_input_record_id: str
    investigation_session_id: str
    workspace_identity: str
    minichem_scope_id: str
    scope_contract_digest: str
    capability_generation_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "research_task_id",
            "investigation_context_id",
            "hypothesis_execution_input_record_id",
            "investigation_session_id",
            "minichem_scope_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_ref(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "workspace_identity",
            _require_workspace_identity(self.workspace_identity),
        )
        object.__setattr__(
            self,
            "scope_contract_digest",
            _require_digest(self.scope_contract_digest, "scope_contract_digest"),
        )
        object.__setattr__(
            self,
            "capability_generation_digest",
            _require_digest(
                self.capability_generation_digest,
                "capability_generation_digest",
            ),
        )

@dataclass(frozen=True)
class ProviderAttemptReference:
    """一次 provider turn attempt 的身份和请求摘要。"""

    investigation_session_id: str
    model_thread_id: str
    model_turn_id: str
    model_attempt_id: str
    model_request_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "investigation_session_id",
            "model_thread_id",
            "model_turn_id",
            "model_attempt_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_ref(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "model_request_digest",
            _require_digest(self.model_request_digest, "model_request_digest"),
        )

@dataclass(frozen=True)
class JobReference:
    """MiniChem job 的身份、提交身份和提交请求摘要。"""

    investigation_session_id: str
    minichem_job_id: str
    minichem_operation_id: str
    minichem_submission_key: str
    minichem_request_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "investigation_session_id",
            "minichem_job_id",
            "minichem_operation_id",
            "minichem_submission_key",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_ref(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "minichem_request_digest",
            _require_digest(self.minichem_request_digest, "minichem_request_digest"),
        )

@dataclass(frozen=True)
class ReceiptReference:
    """外部 owner 回执的引用和摘要。"""

    kind: ReceiptKind
    receipt_ref: str
    receipt_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReceiptKind):
            raise HarnessRecoveryContractError("kind 必须是 ReceiptKind")
        object.__setattr__(self, "receipt_ref", _require_ref(self.receipt_ref, "receipt_ref"))
        object.__setattr__(
            self,
            "receipt_digest",
            _require_digest(self.receipt_digest, "receipt_digest"),
        )


@dataclass(frozen=True)
class EvidenceVerificationReference:
    """已核验证据的引用；不保存证据正文或 artifact 内容。"""

    verification_ref: str
    verification_digest: str
    job_id: str
    operation_id: str
    submission_key: str

    def __post_init__(self) -> None:
        for field_name in (
            "verification_ref",
            "job_id",
            "operation_id",
            "submission_key",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_ref(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "verification_digest",
            _require_digest(self.verification_digest, "verification_digest"),
        )


@dataclass(frozen=True)
class ModuleOutputReference:
    """已提交模块输出的可重放引用；不保存科学输出正文。"""

    output_ref: str
    output_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_ref", _require_ref(self.output_ref, "output_ref"))
        object.__setattr__(
            self,
            "output_digest",
            _require_digest(self.output_digest, "output_digest"),
        )


@dataclass(frozen=True)
class RecoveryTransition:
    """所有公开恢复事件的 typed 基类。"""

    event_id: str
    transition_type: ClassVar[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_ref(self.event_id, "event_id"))

    def _content(self) -> Mapping[str, Any]:
        """返回用于摘要和持久化的 typed 内容；仅供 store 内部使用。"""

        raise NotImplementedError

    @property
    def event_digest(self) -> str:
        """返回不含 event id 的内容摘要。"""

        return _digest({"transition_type": self.transition_type, "content": self._content()})


@dataclass(frozen=True)
class RecordManifestRead(RecoveryTransition):
    """记录一次旧展示索引协调，避免后续恢复再次读取它。"""

    manifest_digest: str
    transition_type: ClassVar[str] = "manifest_read_recorded"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "manifest_digest",
            _require_digest(self.manifest_digest, "manifest_digest"),
        )

    def _content(self) -> Mapping[str, Any]:
        return {"manifest_digest": self.manifest_digest}


@dataclass(frozen=True)
class BindSession(RecoveryTransition):
    """绑定一个 invocation 的唯一 session。"""

    session: SessionReference
    transition_type: ClassVar[str] = "session_bound"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.session, SessionReference):
            raise HarnessRecoveryContractError("session 必须是 SessionReference")

    def _content(self) -> Mapping[str, Any]:
        return {"session": _serialize_session(self.session)}


@dataclass(frozen=True)
class StartProviderAttempt(RecoveryTransition):
    """开始一个 provider attempt。"""

    attempt: ProviderAttemptReference
    transition_type: ClassVar[str] = "provider_attempt_started"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.attempt, ProviderAttemptReference):
            raise HarnessRecoveryContractError("attempt 必须是 ProviderAttemptReference")

    def _content(self) -> Mapping[str, Any]:
        return {"attempt": _serialize_attempt(self.attempt)}


@dataclass(frozen=True)
class CloseProviderAttempt(RecoveryTransition):
    """记录 provider 明确返回的 attempt 终态。"""

    attempt: ProviderAttemptReference
    outcome: ProviderAttemptOutcome
    receipt: ReceiptReference | None = None
    transition_type: ClassVar[str] = "provider_attempt_closed"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.attempt, ProviderAttemptReference):
            raise HarnessRecoveryContractError("attempt 必须是 ProviderAttemptReference")
        if not isinstance(self.outcome, ProviderAttemptOutcome):
            raise HarnessRecoveryContractError("outcome 必须是 ProviderAttemptOutcome")
        if self.outcome is ProviderAttemptOutcome.COMPLETED and self.receipt is None:
            raise HarnessRecoveryContractError("completed attempt 必须有 provider receipt")
        if self.receipt is not None and not isinstance(self.receipt, ReceiptReference):
            raise HarnessRecoveryContractError("receipt 必须是 ReceiptReference")
        if self.receipt is not None and self.receipt.kind is not ReceiptKind.PROVIDER_TURN:
            raise HarnessRecoveryContractError("provider attempt 只能绑定 provider_turn receipt")

    def _content(self) -> Mapping[str, Any]:
        return {
            "attempt": _serialize_attempt(self.attempt),
            "outcome": self.outcome.value,
            "receipt": None if self.receipt is None else _serialize_receipt(self.receipt),
        }


@dataclass(frozen=True)
class SubmitJob(RecoveryTransition):
    """记录一次物理 MiniChem job 的提交身份。"""

    attempt: ProviderAttemptReference
    job: JobReference
    transition_type: ClassVar[str] = "job_submitted"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.attempt, ProviderAttemptReference):
            raise HarnessRecoveryContractError("attempt 必须是 ProviderAttemptReference")
        if not isinstance(self.job, JobReference):
            raise HarnessRecoveryContractError("job 必须是 JobReference")

    def _content(self) -> Mapping[str, Any]:
        return {
            "attempt": _serialize_attempt(self.attempt),
            "job": _serialize_job(self.job),
        }


@dataclass(frozen=True)
class CollectJob(RecoveryTransition):
    """记录同一 job 已从 MiniChem owner 收集。"""

    job: JobReference
    receipt: ReceiptReference | None = None
    transition_type: ClassVar[str] = "job_collected"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.job, JobReference):
            raise HarnessRecoveryContractError("job 必须是 JobReference")
        if self.receipt is not None and not isinstance(self.receipt, ReceiptReference):
            raise HarnessRecoveryContractError("receipt 必须是 ReceiptReference")
        if self.receipt is not None and self.receipt.kind is not ReceiptKind.MINICHEM_OPERATION:
            raise HarnessRecoveryContractError(
                "job collect 只能绑定 minichem_operation receipt"
            )

    def _content(self) -> Mapping[str, Any]:
        return {
            "job": _serialize_job(self.job),
            "receipt": None if self.receipt is None else _serialize_receipt(self.receipt),
        }


@dataclass(frozen=True)
class ReconcileCompletedProviderTurn(RecoveryTransition):
    """用同一 completed provider receipt 原子补齐未持久化的作业序列。"""

    attempt: ProviderAttemptReference
    provider_receipt: ReceiptReference
    collected_jobs: tuple[tuple[JobReference, ReceiptReference], ...]
    transition_type: ClassVar[str] = "completed_provider_turn_reconciled"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.attempt, ProviderAttemptReference):
            raise HarnessRecoveryContractError("attempt 必须是 ProviderAttemptReference")
        if (
            not isinstance(self.provider_receipt, ReceiptReference)
            or self.provider_receipt.kind is not ReceiptKind.PROVIDER_TURN
        ):
            raise HarnessRecoveryContractError(
                "completed turn reconciliation 必须绑定 provider_turn receipt"
            )
        normalized: list[tuple[JobReference, ReceiptReference]] = []
        for item in self.collected_jobs:
            if not isinstance(item, tuple) or len(item) != 2:
                raise HarnessRecoveryContractError(
                    "completed turn reconciliation 作业项必须是 job/receipt 对"
                )
            job, receipt = item
            if not isinstance(job, JobReference):
                raise HarnessRecoveryContractError("reconciled job 必须是 JobReference")
            if (
                not isinstance(receipt, ReceiptReference)
                or receipt.kind is not ReceiptKind.MINICHEM_OPERATION
            ):
                raise HarnessRecoveryContractError(
                    "reconciled job 必须绑定 minichem_operation receipt"
                )
            normalized.append((job, receipt))
        if not normalized:
            raise HarnessRecoveryContractError(
                "completed turn reconciliation 至少需要一个已收集 job"
            )
        object.__setattr__(self, "collected_jobs", tuple(normalized))

    def _content(self) -> Mapping[str, Any]:
        return {
            "attempt": _serialize_attempt(self.attempt),
            "provider_receipt": _serialize_receipt(self.provider_receipt),
            "collected_jobs": [
                {
                    "job": _serialize_job(job),
                    "receipt": _serialize_receipt(receipt),
                }
                for job, receipt in self.collected_jobs
            ],
        }


@dataclass(frozen=True)
class VerifyEvidence(RecoveryTransition):
    """记录外部 verifier 已核验某个已收集 job 的证据。"""

    verification: EvidenceVerificationReference
    transition_type: ClassVar[str] = "evidence_verified"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.verification, EvidenceVerificationReference):
            raise HarnessRecoveryContractError(
                "verification 必须是 EvidenceVerificationReference"
            )

    def _content(self) -> Mapping[str, Any]:
        return {"verification": _serialize_verification(self.verification)}


@dataclass(frozen=True)
class CommitModuleOutput(RecoveryTransition):
    """保存已提交模块输出的可重放引用。"""

    output: ModuleOutputReference
    transition_type: ClassVar[str] = "module_output_committed"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.output, ModuleOutputReference):
            raise HarnessRecoveryContractError("output 必须是 ModuleOutputReference")

    def _content(self) -> Mapping[str, Any]:
        return {"output": _serialize_output(self.output)}


@dataclass(frozen=True)
class SuspendInvocation(RecoveryTransition):
    """显式记录暂时无法继续；不代表 provider 或进程失败。"""

    reason_code: str
    transition_type: ClassVar[str] = "invocation_suspended"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "reason_code", _require_code(self.reason_code, "reason_code"))

    def _content(self) -> Mapping[str, Any]:
        return {"reason_code": self.reason_code}


@dataclass(frozen=True)
class CompleteInvocation(RecoveryTransition):
    """显式确认已提交输出的 invocation 完成。"""

    transition_type: ClassVar[str] = "invocation_completed"

    def __post_init__(self) -> None:
        super().__post_init__()

    def _content(self) -> Mapping[str, Any]:
        return {}


@dataclass(frozen=True)
class FailInvocation(RecoveryTransition):
    """显式记录不可恢复失败；不会由缺少进程自动生成。"""

    failure_code: str
    transition_type: ClassVar[str] = "invocation_failed"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "failure_code",
            _require_code(self.failure_code, "failure_code"),
        )

    def _content(self) -> Mapping[str, Any]:
        return {"failure_code": self.failure_code}


RecoveryTransitionType: TypeAlias = (
    BindSession
    | RecordManifestRead
    | StartProviderAttempt
    | CloseProviderAttempt
    | SubmitJob
    | CollectJob
    | ReconcileCompletedProviderTurn
    | VerifyEvidence
    | CommitModuleOutput
    | SuspendInvocation
    | CompleteInvocation
    | FailInvocation
)

# 公开名称使用 Harness 前缀，避免与旧 Working Memory 的通用事件类型混淆。
HarnessRecoveryTransition = RecoveryTransition


@dataclass(frozen=True)
class HarnessRecoverySnapshot:
    """从不可变事件投影出的 Harness 恢复快照。"""

    identity: HarnessRunIdentity
    state: HarnessLifecycle
    failure_code: str | None
    session: SessionReference | None
    attempts: tuple[ProviderAttemptReference, ...]
    active_attempt: ProviderAttemptReference | None
    jobs: tuple[JobReference, ...]
    collected_job_ids: tuple[str, ...]
    active_job: JobReference | None
    last_receipt: ReceiptReference | None
    verified_evidence: tuple[EvidenceVerificationReference, ...]
    committed_output: ModuleOutputReference | None
    manifest_read_digest: str | None
    last_event_id: str
    last_event_digest: str
    event_sequence: int

    @property
    def last_attempt(self) -> ProviderAttemptReference | None:
        """返回最近一次 attempt；仅作为恢复读取便利属性。"""

        return self.attempts[-1] if self.attempts else None


def _serialize_identity(identity: HarnessRunIdentity) -> dict[str, str]:
    return {
        "research_task_id": identity.research_task_id,
        "module_invocation_id": identity.module_invocation_id,
        "harness_invocation_id": identity.harness_invocation_id,
        "module_input_digest": identity.module_input_digest,
    }


def _deserialize_identity(value: Any) -> HarnessRunIdentity:
    mapping = _strict_mapping(
        value,
        {
            "research_task_id",
            "module_invocation_id",
            "harness_invocation_id",
            "module_input_digest",
        },
        "identity",
    )
    return HarnessRunIdentity(
        research_task_id=mapping["research_task_id"],
        module_invocation_id=mapping["module_invocation_id"],
        harness_invocation_id=mapping["harness_invocation_id"],
        module_input_digest=mapping["module_input_digest"],
    )


def _serialize_session(session: SessionReference) -> dict[str, str]:
    return {
        "research_task_id": session.research_task_id,
        "investigation_context_id": session.investigation_context_id,
        "hypothesis_execution_input_record_id": session.hypothesis_execution_input_record_id,
        "investigation_session_id": session.investigation_session_id,
        "workspace_identity": session.workspace_identity,
        "minichem_scope_id": session.minichem_scope_id,
        "scope_contract_digest": session.scope_contract_digest,
        "capability_generation_digest": session.capability_generation_digest,
    }


def _deserialize_session(value: Any) -> SessionReference:
    mapping = _strict_mapping(
        value,
        {
            "research_task_id",
            "investigation_context_id",
            "hypothesis_execution_input_record_id",
            "investigation_session_id",
            "workspace_identity",
            "minichem_scope_id",
            "scope_contract_digest",
            "capability_generation_digest",
        },
        "session",
    )
    return SessionReference(**dict(mapping))


def _serialize_attempt(attempt: ProviderAttemptReference) -> dict[str, str]:
    return {
        "investigation_session_id": attempt.investigation_session_id,
        "model_thread_id": attempt.model_thread_id,
        "model_turn_id": attempt.model_turn_id,
        "model_attempt_id": attempt.model_attempt_id,
        "model_request_digest": attempt.model_request_digest,
    }


def _deserialize_attempt(value: Any) -> ProviderAttemptReference:
    mapping = _strict_mapping(
        value,
        {
            "investigation_session_id",
            "model_thread_id",
            "model_turn_id",
            "model_attempt_id",
            "model_request_digest",
        },
        "attempt",
    )
    return ProviderAttemptReference(**dict(mapping))


def _serialize_job(job: JobReference) -> dict[str, str]:
    return {
        "investigation_session_id": job.investigation_session_id,
        "minichem_job_id": job.minichem_job_id,
        "minichem_operation_id": job.minichem_operation_id,
        "minichem_submission_key": job.minichem_submission_key,
        "minichem_request_digest": job.minichem_request_digest,
    }


def _deserialize_job(value: Any) -> JobReference:
    mapping = _strict_mapping(
        value,
        {
            "investigation_session_id",
            "minichem_job_id",
            "minichem_operation_id",
            "minichem_submission_key",
            "minichem_request_digest",
        },
        "job",
    )
    return JobReference(**dict(mapping))


def _deserialize_legacy_identity(value: Any) -> HarnessRunIdentity:
    """schema-v1 identity reader；旧字段不会流入新序列化。"""

    mapping = _strict_mapping(
        value,
        {"task_id", "module_run_id", "runtime_ref", "input_digest"},
        "legacy identity",
    )
    return HarnessRunIdentity(
        research_task_id=mapping["task_id"],
        module_invocation_id=mapping["module_run_id"],
        harness_invocation_id=mapping["runtime_ref"],
        module_input_digest=mapping["input_digest"],
    )


def _deserialize_legacy_session(value: Any) -> SessionReference:
    """schema-v1 Session reader；v1 generation 暂作合同回填。"""

    mapping = _strict_mapping(
        value,
        {
            "task_id",
            "batch_id",
            "combination_id",
            "session_id",
            "workspace_ref",
            "scope_id",
            "scope_generation_digest",
        },
        "legacy session",
    )
    return SessionReference(
        research_task_id=mapping["task_id"],
        investigation_context_id=mapping["batch_id"],
        hypothesis_execution_input_record_id=mapping["combination_id"],
        investigation_session_id=mapping["session_id"],
        workspace_identity=mapping["workspace_ref"],
        minichem_scope_id=mapping["scope_id"],
        # schema-v1 没有独立合同摘要；该回填只存在于明确的只读 reader。
        scope_contract_digest=mapping["scope_generation_digest"],
        capability_generation_digest=mapping["scope_generation_digest"],
    )


def _deserialize_legacy_attempt(value: Any) -> ProviderAttemptReference:
    mapping = _strict_mapping(
        value,
        {"session_id", "thread_id", "turn_id", "attempt_id", "request_digest"},
        "legacy attempt",
    )
    return ProviderAttemptReference(
        investigation_session_id=mapping["session_id"],
        model_thread_id=mapping["thread_id"],
        model_turn_id=mapping["turn_id"],
        model_attempt_id=mapping["attempt_id"],
        model_request_digest=mapping["request_digest"],
    )


def _deserialize_legacy_job(value: Any) -> JobReference:
    mapping = _strict_mapping(
        value,
        {"session_id", "job_id", "operation_id", "submission_key", "request_digest"},
        "legacy job",
    )
    return JobReference(
        investigation_session_id=mapping["session_id"],
        minichem_job_id=mapping["job_id"],
        minichem_operation_id=mapping["operation_id"],
        minichem_submission_key=mapping["submission_key"],
        minichem_request_digest=mapping["request_digest"],
    )


def _serialize_receipt(receipt: ReceiptReference) -> dict[str, str]:
    return {
        "kind": receipt.kind.value,
        "receipt_ref": receipt.receipt_ref,
        "receipt_digest": receipt.receipt_digest,
    }


def _deserialize_receipt(value: Any) -> ReceiptReference:
    mapping = _strict_mapping(
        value,
        {"kind", "receipt_ref", "receipt_digest"},
        "receipt",
    )
    try:
        kind = ReceiptKind(mapping["kind"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessRecoveryContractError("receipt.kind 不受支持") from exc
    return ReceiptReference(
        kind=kind,
        receipt_ref=mapping["receipt_ref"],
        receipt_digest=mapping["receipt_digest"],
    )


def _serialize_verification(
    verification: EvidenceVerificationReference,
) -> dict[str, str]:
    return {
        "verification_ref": verification.verification_ref,
        "verification_digest": verification.verification_digest,
        "job_id": verification.job_id,
        "operation_id": verification.operation_id,
        "submission_key": verification.submission_key,
    }


def _deserialize_verification(value: Any) -> EvidenceVerificationReference:
    mapping = _strict_mapping(
        value,
        {"verification_ref", "verification_digest", "job_id", "operation_id", "submission_key"},
        "verification",
    )
    return EvidenceVerificationReference(**dict(mapping))


def _serialize_output(output: ModuleOutputReference) -> dict[str, str]:
    return {"output_ref": output.output_ref, "output_digest": output.output_digest}


def _deserialize_output(value: Any) -> ModuleOutputReference:
    mapping = _strict_mapping(value, {"output_ref", "output_digest"}, "output")
    return ModuleOutputReference(**dict(mapping))


def _serialize_transition(transition: RecoveryTransitionType) -> dict[str, Any]:
    return {
        "event_id": transition.event_id,
        "transition_type": transition.transition_type,
        "content": dict(transition._content()),
    }


def _legacy_transition_digest(value: Mapping[str, Any]) -> str:
    """按 schema-v1 原始字段核对历史事件摘要。

    schema-v1 的 event digest 覆盖旧字段名；先转换成 canonical transition
    再调用 ``event_digest`` 会改变摘要，因而只读 reader 必须在转换前按原始
    JSON 内容核对。转换后的内存快照仍保留这个已核对的历史摘要。
    """

    return _digest(
        {
            "transition_type": value["transition_type"],
            "content": value["content"],
        }
    )


def _deserialize_transition(
    value: Any,
    *,
    legacy: bool = False,
) -> RecoveryTransitionType:
    """读取事件；历史字段只能由显式 legacy reader 请求。"""

    mapping = _strict_mapping(
        value,
        {"event_id", "transition_type", "content"},
        "transition",
    )
    event_id = mapping["event_id"]
    transition_type = mapping["transition_type"]
    content = mapping["content"]
    if legacy:
        return _deserialize_legacy_transition(mapping)
    if transition_type == RecordManifestRead.transition_type:
        return RecordManifestRead(event_id, content["manifest_digest"])
    if transition_type == BindSession.transition_type:
        return BindSession(event_id, _deserialize_session(content["session"]))
    if transition_type == StartProviderAttempt.transition_type:
        return StartProviderAttempt(event_id, _deserialize_attempt(content["attempt"]))
    if transition_type == CloseProviderAttempt.transition_type:
        close = _strict_mapping(content, {"attempt", "outcome", "receipt"}, "close content")
        try:
            outcome = ProviderAttemptOutcome(close["outcome"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HarnessRecoveryContractError("attempt outcome 不受支持") from exc
        receipt = None if close["receipt"] is None else _deserialize_receipt(close["receipt"])
        return CloseProviderAttempt(
            event_id,
            _deserialize_attempt(close["attempt"]),
            outcome,
            receipt,
        )
    if transition_type == SubmitJob.transition_type:
        submit = _strict_mapping(content, {"attempt", "job"}, "submit content")
        return SubmitJob(
            event_id,
            _deserialize_attempt(submit["attempt"]),
            _deserialize_job(submit["job"]),
        )
    if transition_type == CollectJob.transition_type:
        collect = _strict_mapping(content, {"job", "receipt"}, "collect content")
        receipt = None if collect["receipt"] is None else _deserialize_receipt(collect["receipt"])
        return CollectJob(event_id, _deserialize_job(collect["job"]), receipt)
    if transition_type == ReconcileCompletedProviderTurn.transition_type:
        reconcile = _strict_mapping(
            content,
            {"attempt", "provider_receipt", "collected_jobs"},
            "completed turn reconciliation content",
        )
        raw_jobs = reconcile["collected_jobs"]
        if isinstance(raw_jobs, (str, bytes)) or not isinstance(raw_jobs, list):
            raise HarnessRecoveryContractError(
                "completed turn reconciliation collected_jobs 非法"
            )
        collected_jobs: list[tuple[JobReference, ReceiptReference]] = []
        for raw_item in raw_jobs:
            item = _strict_mapping(
                raw_item,
                {"job", "receipt"},
                "completed turn reconciliation job",
            )
            collected_jobs.append(
                (_deserialize_job(item["job"]), _deserialize_receipt(item["receipt"]))
            )
        return ReconcileCompletedProviderTurn(
            event_id,
            _deserialize_attempt(reconcile["attempt"]),
            _deserialize_receipt(reconcile["provider_receipt"]),
            tuple(collected_jobs),
        )
    if transition_type == VerifyEvidence.transition_type:
        return VerifyEvidence(event_id, _deserialize_verification(content["verification"]))
    if transition_type == CommitModuleOutput.transition_type:
        return CommitModuleOutput(event_id, _deserialize_output(content["output"]))
    if transition_type == SuspendInvocation.transition_type:
        return SuspendInvocation(event_id, content["reason_code"])
    if transition_type == CompleteInvocation.transition_type:
        return CompleteInvocation(event_id)
    if transition_type == FailInvocation.transition_type:
        return FailInvocation(event_id, content["failure_code"])
    raise HarnessRecoveryContractError(f"不支持的 transition_type: {transition_type!r}")


def _deserialize_legacy_transition(mapping: Mapping[str, Any]) -> RecoveryTransitionType:
    """schema-v1 transition reader；输出对象已是当前 typed identity。"""

    event_id = mapping["event_id"]
    transition_type = mapping["transition_type"]
    content = mapping["content"]
    if transition_type == RecordManifestRead.transition_type:
        return RecordManifestRead(event_id, content["manifest_digest"])
    if transition_type == BindSession.transition_type:
        return BindSession(event_id, _deserialize_legacy_session(content["session"]))
    if transition_type == StartProviderAttempt.transition_type:
        return StartProviderAttempt(event_id, _deserialize_legacy_attempt(content["attempt"]))
    if transition_type == CloseProviderAttempt.transition_type:
        outcome = ProviderAttemptOutcome(content["outcome"])
        receipt = (
            None
            if content["receipt"] is None
            else _deserialize_receipt(content["receipt"])
        )
        return CloseProviderAttempt(
            event_id,
            _deserialize_legacy_attempt(content["attempt"]),
            outcome,
            receipt,
        )
    if transition_type == SubmitJob.transition_type:
        return SubmitJob(
            event_id,
            _deserialize_legacy_attempt(content["attempt"]),
            _deserialize_legacy_job(content["job"]),
        )
    if transition_type == CollectJob.transition_type:
        receipt = (
            None
            if content["receipt"] is None
            else _deserialize_receipt(content["receipt"])
        )
        return CollectJob(event_id, _deserialize_legacy_job(content["job"]), receipt)
    if transition_type == ReconcileCompletedProviderTurn.transition_type:
        jobs = tuple(
            (
                _deserialize_legacy_job(item["job"]),
                _deserialize_receipt(item["receipt"]),
            )
            for item in content["collected_jobs"]
        )
        return ReconcileCompletedProviderTurn(
            event_id,
            _deserialize_legacy_attempt(content["attempt"]),
            _deserialize_receipt(content["provider_receipt"]),
            jobs,
        )
    if transition_type == VerifyEvidence.transition_type:
        return VerifyEvidence(event_id, _deserialize_verification(content["verification"]))
    if transition_type == CommitModuleOutput.transition_type:
        return CommitModuleOutput(event_id, _deserialize_output(content["output"]))
    if transition_type == SuspendInvocation.transition_type:
        return SuspendInvocation(event_id, content["reason_code"])
    if transition_type == CompleteInvocation.transition_type:
        return CompleteInvocation(event_id)
    if transition_type == FailInvocation.transition_type:
        return FailInvocation(event_id, content["failure_code"])
    raise HarnessRecoveryContractError(f"不支持的 schema-v1 transition_type: {transition_type!r}")


def _serialize_snapshot(snapshot: HarnessRecoverySnapshot) -> dict[str, Any]:
    return {
        "identity": _serialize_identity(snapshot.identity),
        "state": snapshot.state.value,
        "failure_code": snapshot.failure_code,
        "session": None if snapshot.session is None else _serialize_session(snapshot.session),
        "attempts": [_serialize_attempt(attempt) for attempt in snapshot.attempts],
        "active_attempt": (
            None
            if snapshot.active_attempt is None
            else _serialize_attempt(snapshot.active_attempt)
        ),
        "jobs": [_serialize_job(job) for job in snapshot.jobs],
        "collected_job_ids": list(snapshot.collected_job_ids),
        "active_job": None if snapshot.active_job is None else _serialize_job(snapshot.active_job),
        "last_receipt": (
            None if snapshot.last_receipt is None else _serialize_receipt(snapshot.last_receipt)
        ),
        "verified_evidence": [
            _serialize_verification(verification)
            for verification in snapshot.verified_evidence
        ],
        "committed_output": (
            None
            if snapshot.committed_output is None
            else _serialize_output(snapshot.committed_output)
        ),
        "manifest_read_digest": snapshot.manifest_read_digest,
        "last_event_id": snapshot.last_event_id,
        "last_event_digest": snapshot.last_event_digest,
        "event_sequence": snapshot.event_sequence,
    }


def _deserialize_snapshot(value: Any) -> HarnessRecoverySnapshot:
    mapping = _strict_mapping(
        value,
        {
            "identity",
            "state",
            "failure_code",
            "session",
            "attempts",
            "active_attempt",
            "jobs",
            "collected_job_ids",
            "active_job",
            "last_receipt",
            "verified_evidence",
            "committed_output",
            "manifest_read_digest",
            "last_event_id",
            "last_event_digest",
            "event_sequence",
        },
        "snapshot",
    )
    try:
        state = HarnessLifecycle(mapping["state"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessRecoveryContractError("snapshot.state 不受支持") from exc
    attempts_value = mapping["attempts"]
    jobs_value = mapping["jobs"]
    collected_value = mapping["collected_job_ids"]
    evidence_value = mapping["verified_evidence"]
    if not isinstance(attempts_value, list) or not isinstance(jobs_value, list):
        raise HarnessRecoveryContractError("snapshot attempts/jobs 必须是数组")
    if not isinstance(collected_value, list) or not isinstance(evidence_value, list):
        raise HarnessRecoveryContractError("snapshot collected/evidence 必须是数组")
    identity = _deserialize_identity(mapping["identity"])
    failure_code = mapping["failure_code"]
    if state is HarnessLifecycle.FAILED:
        failure_code = _require_code(failure_code, "snapshot.failure_code")
    elif failure_code is not None:
        raise HarnessRecoveryContractError(
            "非 failed snapshot 不得携带 failure_code"
        )
    manifest_read_digest = mapping["manifest_read_digest"]
    if manifest_read_digest is not None:
        manifest_read_digest = _require_digest(
            manifest_read_digest,
            "snapshot.manifest_read_digest",
        )
    return HarnessRecoverySnapshot(
        identity=identity,
        state=state,
        failure_code=failure_code,
        session=None if mapping["session"] is None else _deserialize_session(mapping["session"]),
        attempts=tuple(_deserialize_attempt(item) for item in attempts_value),
        active_attempt=(
            None
            if mapping["active_attempt"] is None
            else _deserialize_attempt(mapping["active_attempt"])
        ),
        jobs=tuple(_deserialize_job(item) for item in jobs_value),
        collected_job_ids=tuple(
            _require_ref(item, "collected_job_ids[]") for item in collected_value
        ),
        active_job=(
            None if mapping["active_job"] is None else _deserialize_job(mapping["active_job"])
        ),
        last_receipt=(
            None
            if mapping["last_receipt"] is None
            else _deserialize_receipt(mapping["last_receipt"])
        ),
        verified_evidence=tuple(
            _deserialize_verification(item) for item in evidence_value
        ),
        committed_output=(
            None
            if mapping["committed_output"] is None
            else _deserialize_output(mapping["committed_output"])
        ),
        manifest_read_digest=manifest_read_digest,
        last_event_id=_require_ref(mapping["last_event_id"], "last_event_id"),
        last_event_digest=_require_digest(mapping["last_event_digest"], "last_event_digest"),
        event_sequence=(
            mapping["event_sequence"]
            if isinstance(mapping["event_sequence"], int) and mapping["event_sequence"] > 0
            else (_raise_contract("snapshot.event_sequence 必须是正整数"))
        ),
    )


def _deserialize_legacy_snapshot(
    mapping: Mapping[str, Any],
) -> HarnessRecoverySnapshot:
    """schema-v1 快照读取器；迁移到正式 typed identity 后再复用校验。"""

    value = dict(mapping)
    value["identity"] = _serialize_identity(
        _deserialize_legacy_identity(mapping["identity"])
    )
    if mapping["session"] is not None:
        value["session"] = _serialize_session(
            _deserialize_legacy_session(mapping["session"])
        )
    value["attempts"] = [
        _serialize_attempt(_deserialize_legacy_attempt(item))
        for item in mapping["attempts"]
    ]
    if mapping["active_attempt"] is not None:
        value["active_attempt"] = _serialize_attempt(
            _deserialize_legacy_attempt(mapping["active_attempt"])
        )
    value["jobs"] = [
        _serialize_job(_deserialize_legacy_job(item)) for item in mapping["jobs"]
    ]
    if mapping["active_job"] is not None:
        value["active_job"] = _serialize_job(
            _deserialize_legacy_job(mapping["active_job"])
        )
    return _deserialize_snapshot(value)


class SchemaV1LegacyHarnessRecoveryReader:
    """schema-v1 Harness recovery 的显式只读解析边界。

    该 reader 只把旧 JSON 归一为当前 typed identity，不提供任何写入能力。
    新 schema 的 parser 不会根据字段猜测旧格式，避免旧双协议回到生产路径。
    """

    @staticmethod
    def read_identity(value: Any) -> HarnessRunIdentity:
        return _deserialize_legacy_identity(value)

    @staticmethod
    def read_session(value: Any) -> SessionReference:
        return _deserialize_legacy_session(value)

    @staticmethod
    def read_attempt(value: Any) -> ProviderAttemptReference:
        return _deserialize_legacy_attempt(value)

    @staticmethod
    def read_job(value: Any) -> JobReference:
        return _deserialize_legacy_job(value)

    @staticmethod
    def read_transition(value: Any) -> RecoveryTransitionType:
        return _deserialize_transition(value, legacy=True)

    @staticmethod
    def read_snapshot(value: Any) -> HarnessRecoverySnapshot:
        mapping = _strict_mapping(
            value,
            {
                "identity",
                "state",
                "failure_code",
                "session",
                "attempts",
                "active_attempt",
                "jobs",
                "collected_job_ids",
                "active_job",
                "last_receipt",
                "verified_evidence",
                "committed_output",
                "manifest_read_digest",
                "last_event_id",
                "last_event_digest",
                "event_sequence",
            },
            "legacy snapshot",
        )
        return _deserialize_legacy_snapshot(mapping)


def _raise_contract(message: str) -> Any:
    raise HarnessRecoveryContractError(message)


def _initial_snapshot(identity: HarnessRunIdentity) -> HarnessRecoverySnapshot:
    """创建 invocation 初始快照；初始事件由 store 写入。"""

    event_id = f"invocation-{_digest(_serialize_identity(identity))}"
    event_digest = _digest({"transition_type": "invocation_started", "identity": _serialize_identity(identity)})
    return HarnessRecoverySnapshot(
        identity=identity,
        state=HarnessLifecycle.OPEN,
        failure_code=None,
        session=None,
        attempts=(),
        active_attempt=None,
        jobs=(),
        collected_job_ids=(),
        active_job=None,
        last_receipt=None,
        verified_evidence=(),
        committed_output=None,
        manifest_read_digest=None,
        last_event_id=event_id,
        last_event_digest=event_digest,
        event_sequence=1,
    )


def _identity_differences(
    expected: HarnessRunIdentity,
    actual: HarnessRunIdentity,
) -> tuple[str, ...]:
    return tuple(
        field_name
        for field_name in (
            "research_task_id",
            "module_invocation_id",
            "harness_invocation_id",
            "module_input_digest",
        )
        if getattr(expected, field_name) != getattr(actual, field_name)
    )


def _invocation_key(
    identity: HarnessRunIdentity,
    *,
    legacy: bool = False,
) -> str:
    """按当前或 schema-v1 身份字段定位 invocation。"""

    if legacy:
        return _digest(
            {
                "runtime_ref": identity.harness_invocation_id,
                "module_run_id": identity.module_invocation_id,
            }
        )
    return _digest(
        {
            "harness_invocation_id": identity.harness_invocation_id,
            "module_invocation_id": identity.module_invocation_id,
        }
    )


def _assert_identity(expected: HarnessRunIdentity, actual: HarnessRunIdentity) -> None:
    differences = _identity_differences(expected, actual)
    if differences:
        raise HarnessRecoveryIdentityConflict(
            f"HarnessRunIdentity 不一致：{', '.join(differences)}"
        )


def _assert_session_task(snapshot: HarnessRecoverySnapshot, session: SessionReference) -> None:
    if session.research_task_id != snapshot.identity.research_task_id:
        raise HarnessRecoveryIdentityConflict(
            "session 的 research_task_id 与 invocation 不一致"
        )


def _assert_attempt_session(
    snapshot: HarnessRecoverySnapshot,
    attempt: ProviderAttemptReference,
) -> None:
    if snapshot.session is None:
        raise HarnessRecoveryStateConflict("必须先绑定 session 才能记录 provider attempt")
    if attempt.investigation_session_id != snapshot.session.investigation_session_id:
        raise HarnessRecoveryIdentityConflict(
            "attempt 的 investigation_session_id 发生漂移"
        )
    if snapshot.attempts and attempt.model_thread_id != snapshot.attempts[-1].model_thread_id:
        raise HarnessRecoveryIdentityConflict("provider thread_id 发生漂移")
    if any(previous.model_turn_id == attempt.model_turn_id for previous in snapshot.attempts):
        raise HarnessRecoveryIdentityConflict("turn_id 不能被另一个 attempt 重用")
    if any(previous.model_attempt_id == attempt.model_attempt_id for previous in snapshot.attempts):
        raise HarnessRecoveryIdentityConflict("attempt_id 不能被重用")


def _assert_job_matches_session(
    snapshot: HarnessRecoverySnapshot,
    job: JobReference,
) -> None:
    if snapshot.session is None:
        raise HarnessRecoveryStateConflict("必须先绑定 session 才能记录 job")
    if job.investigation_session_id != snapshot.session.investigation_session_id:
        raise HarnessRecoveryIdentityConflict(
            "job 的 investigation_session_id 发生漂移"
        )


def _replace_snapshot(snapshot: HarnessRecoverySnapshot, **changes: Any) -> HarnessRecoverySnapshot:
    values = {
        "identity": snapshot.identity,
        "state": snapshot.state,
        "failure_code": snapshot.failure_code,
        "session": snapshot.session,
        "attempts": snapshot.attempts,
        "active_attempt": snapshot.active_attempt,
        "jobs": snapshot.jobs,
        "collected_job_ids": snapshot.collected_job_ids,
        "active_job": snapshot.active_job,
        "last_receipt": snapshot.last_receipt,
        "verified_evidence": snapshot.verified_evidence,
        "committed_output": snapshot.committed_output,
        "manifest_read_digest": snapshot.manifest_read_digest,
        "last_event_id": snapshot.last_event_id,
        "last_event_digest": snapshot.last_event_digest,
        "event_sequence": snapshot.event_sequence,
    }
    values.update(changes)
    return HarnessRecoverySnapshot(**values)


def _apply_transition(
    snapshot: HarnessRecoverySnapshot,
    transition: RecoveryTransitionType,
) -> HarnessRecoverySnapshot:
    """在内存中验证并投影一条 typed transition。"""

    if snapshot.state in {HarnessLifecycle.COMPLETED, HarnessLifecycle.FAILED}:
        raise HarnessRecoveryStateConflict("已结束的 invocation 不能追加新的 transition")

    if isinstance(transition, RecordManifestRead):
        if (
            snapshot.manifest_read_digest is not None
            and snapshot.manifest_read_digest != transition.manifest_digest
        ):
            raise HarnessRecoveryIdentityConflict("展示索引摘要发生漂移")
        return _replace_snapshot(
            snapshot,
            manifest_read_digest=transition.manifest_digest,
        )

    if isinstance(transition, BindSession):
        _assert_session_task(snapshot, transition.session)
        if snapshot.session is not None and snapshot.session != transition.session:
            raise HarnessRecoveryIdentityConflict("session 绑定发生漂移")
        return _replace_snapshot(snapshot, session=transition.session)

    if isinstance(transition, StartProviderAttempt):
        if snapshot.committed_output is not None:
            raise HarnessRecoveryStateConflict("已提交模块输出，不能启动新的 provider attempt")
        _assert_attempt_session(snapshot, transition.attempt)
        if snapshot.active_attempt is not None:
            raise HarnessRecoveryStateConflict("同一 invocation 只能有一个活动 provider attempt")
        if snapshot.active_job is not None:
            raise HarnessRecoveryStateConflict("存在活动 job 时不能启动新的 provider attempt")
        return _replace_snapshot(
            snapshot,
            state=HarnessLifecycle.OPEN,
            attempts=snapshot.attempts + (transition.attempt,),
            active_attempt=transition.attempt,
        )

    if isinstance(transition, CloseProviderAttempt):
        if snapshot.active_attempt != transition.attempt:
            raise HarnessRecoveryIdentityConflict("关闭的 provider attempt 不是当前活动 attempt")
        state = (
            HarnessLifecycle.SUSPENDED
            if transition.outcome in {
                ProviderAttemptOutcome.INTERRUPTED,
                ProviderAttemptOutcome.FAILED,
            }
            else snapshot.state
        )
        return _replace_snapshot(
            snapshot,
            state=state,
            active_attempt=None,
            last_receipt=(
                transition.receipt
                if transition.receipt is not None
                else snapshot.last_receipt
            ),
        )

    if isinstance(transition, SubmitJob):
        if snapshot.committed_output is not None:
            raise HarnessRecoveryStateConflict("已提交模块输出，不能提交新的 job")
        if snapshot.active_attempt != transition.attempt:
            raise HarnessRecoveryIdentityConflict("job submission 不属于当前活动 attempt")
        _assert_job_matches_session(snapshot, transition.job)
        if snapshot.active_job is not None:
            raise HarnessRecoveryStateConflict("同一 invocation 只能有一个活动 job")
        if any(
            previous.minichem_job_id == transition.job.minichem_job_id
            or previous.minichem_operation_id == transition.job.minichem_operation_id
            or previous.minichem_submission_key == transition.job.minichem_submission_key
            for previous in snapshot.jobs
        ):
            raise HarnessRecoveryIdentityConflict("job、operation 或 submission 身份不能被重用")
        return _replace_snapshot(
            snapshot,
            state=HarnessLifecycle.OPEN,
            jobs=snapshot.jobs + (transition.job,),
            active_job=transition.job,
        )

    if isinstance(transition, CollectJob):
        if snapshot.active_job != transition.job:
            raise HarnessRecoveryIdentityConflict("collect 的 job 不是当前活动 job")
        collected = snapshot.collected_job_ids + (transition.job.minichem_job_id,)
        return _replace_snapshot(
            snapshot,
            active_job=None,
            collected_job_ids=collected,
            last_receipt=(
                transition.receipt
                if transition.receipt is not None
                else snapshot.last_receipt
            ),
        )

    if isinstance(transition, ReconcileCompletedProviderTurn):
        if snapshot.active_attempt is not None:
            raise HarnessRecoveryStateConflict(
                "活动 provider attempt 不能进入 completed turn reconciliation"
            )
        if snapshot.last_attempt != transition.attempt:
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 不是最近的 provider attempt"
            )
        if snapshot.last_receipt != transition.provider_receipt:
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 的 provider receipt 不匹配"
            )
        reconciled_jobs = tuple(job for job, _receipt in transition.collected_jobs)
        if len({job.minichem_job_id for job in reconciled_jobs}) != len(reconciled_jobs):
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 含重复 job"
            )
        if len({job.minichem_operation_id for job in reconciled_jobs}) != len(reconciled_jobs):
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 含重复 operation"
            )
        if len({job.minichem_submission_key for job in reconciled_jobs}) != len(reconciled_jobs):
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 含重复 submission"
            )
        for job in reconciled_jobs:
            _assert_job_matches_session(snapshot, job)
        if reconciled_jobs[: len(snapshot.jobs)] != snapshot.jobs:
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 与已保存 job 前缀不一致"
            )
        expected_collected_prefix = tuple(
            job.minichem_job_id
            for job in snapshot.jobs
            if snapshot.active_job is None or job != snapshot.active_job
        )
        if snapshot.collected_job_ids != expected_collected_prefix:
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 与已收集 job 前缀不一致"
            )
        if snapshot.active_job is not None and (
            not snapshot.jobs or snapshot.active_job != snapshot.jobs[-1]
        ):
            raise HarnessRecoveryIdentityConflict(
                "completed turn reconciliation 的活动 job 不是已保存前缀末项"
            )
        return _replace_snapshot(
            snapshot,
            state=HarnessLifecycle.OPEN,
            jobs=reconciled_jobs,
            collected_job_ids=tuple(job.minichem_job_id for job in reconciled_jobs),
            active_job=None,
            last_receipt=transition.collected_jobs[-1][1],
        )

    if isinstance(transition, VerifyEvidence):
        verification = transition.verification
        matching_jobs = tuple(
            job for job in snapshot.jobs if job.minichem_job_id == verification.job_id
        )
        if not matching_jobs:
            raise HarnessRecoveryIdentityConflict("证据核验引用了未知 job")
        job = matching_jobs[0]
        if verification.operation_id != job.minichem_operation_id:
            raise HarnessRecoveryIdentityConflict("证据核验的 operation_id 发生漂移")
        if verification.submission_key != job.minichem_submission_key:
            raise HarnessRecoveryIdentityConflict("证据核验的 submission_key 发生漂移")
        if verification.job_id not in snapshot.collected_job_ids:
            raise HarnessRecoveryStateConflict("job 尚未 collect，不能写入 evidence verification")
        if any(
            previous.verification_ref == verification.verification_ref
            or previous.verification_digest == verification.verification_digest
            for previous in snapshot.verified_evidence
        ):
            raise HarnessRecoveryIdentityConflict("evidence verification 引用或摘要不能被重用")
        return _replace_snapshot(
            snapshot,
            verified_evidence=snapshot.verified_evidence + (verification,),
        )

    if isinstance(transition, CommitModuleOutput):
        if snapshot.committed_output is not None:
            raise HarnessRecoveryIdentityConflict("模块输出引用发生漂移")
        if snapshot.active_job is not None or snapshot.active_attempt is not None:
            raise HarnessRecoveryStateConflict("仍有活动 attempt/job，不能提交模块输出")
        return _replace_snapshot(snapshot, committed_output=transition.output)

    if isinstance(transition, SuspendInvocation):
        return _replace_snapshot(snapshot, state=HarnessLifecycle.SUSPENDED)

    if isinstance(transition, CompleteInvocation):
        if snapshot.committed_output is None:
            raise HarnessRecoveryStateConflict("没有已提交模块输出，不能完成 invocation")
        if snapshot.active_attempt is not None or snapshot.active_job is not None:
            raise HarnessRecoveryStateConflict("仍有活动 attempt/job，不能完成 invocation")
        return _replace_snapshot(snapshot, state=HarnessLifecycle.COMPLETED)

    if isinstance(transition, FailInvocation):
        if snapshot.committed_output is not None:
            raise HarnessRecoveryStateConflict("已提交模块输出，不能把 invocation 改写为 failed")
        return _replace_snapshot(
            snapshot,
            state=HarnessLifecycle.FAILED,
            failure_code=transition.failure_code,
        )

    raise HarnessRecoveryContractError("未注册的 Harness recovery transition")


class HarnessRecoveryStore:
    """Harness 私有恢复事件的深模块。

    调用方只需要学习 ``create_or_load``、``load``、``read_snapshot`` 和
    ``append_transition`` 四个 seam。所有外部身份的严格核对、事件幂等、SQLite
    事务和当前快照投影都封装在实现内。
    """

    def __init__(self, database_path: str | os.PathLike[str]) -> None:
        """初始化独立 SQLite 文件；不会自动创建模型、MCP 或化学作业。"""

        if not isinstance(database_path, (str, os.PathLike)):
            raise HarnessRecoveryContractError("database_path 必须是文件路径")
        path = Path(os.path.abspath(os.path.expanduser(os.fspath(database_path))))
        if path.exists() and not path.is_file():
            raise HarnessRecoveryContractError("database_path 必须指向文件")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HarnessRecoveryStoreError("无法创建 Harness recovery store 目录") from exc
        self.database_path = path
        self._legacy_read_only = False
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        try:
            with self._connect() as connection:
                existing_tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if (
                    {"harness_recovery_runs", "harness_recovery_events"}.intersection(
                        existing_tables
                    )
                    and "harness_recovery_metadata" not in existing_tables
                ):
                    self._legacy_read_only = True
                    return
                if "harness_recovery_metadata" in existing_tables:
                    metadata = connection.execute(
                        "SELECT schema_version FROM harness_recovery_metadata "
                        "WHERE singleton_id = 1"
                    ).fetchone()
                    if metadata is None:
                        raise HarnessRecoveryContractError(
                            "Harness recovery schema version 记录缺失"
                        )
                    if metadata["schema_version"] == _LEGACY_SCHEMA_VERSION:
                        self._legacy_read_only = True
                        return
                    if metadata["schema_version"] != _SCHEMA_VERSION:
                        raise HarnessRecoveryContractError(
                            "Harness recovery schema version 不受支持"
                        )
                    metadata_exists = True
                else:
                    metadata_exists = False
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS harness_recovery_metadata (
                        singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                        schema_version INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS harness_recovery_runs (
                        invocation_key TEXT PRIMARY KEY,
                        identity_json TEXT NOT NULL,
                        snapshot_json TEXT NOT NULL,
                        state TEXT NOT NULL,
                        last_sequence INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS harness_recovery_events (
                        invocation_key TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        event_id TEXT NOT NULL,
                        transition_type TEXT NOT NULL,
                        event_digest TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (invocation_key, sequence),
                        UNIQUE (invocation_key, event_id),
                        UNIQUE (invocation_key, event_digest),
                        FOREIGN KEY (invocation_key)
                            REFERENCES harness_recovery_runs(invocation_key)
                    );
                    """
                )
                if not metadata_exists:
                    connection.execute(
                        "INSERT INTO harness_recovery_metadata "
                        "(singleton_id, schema_version) VALUES (1, ?)",
                        (_SCHEMA_VERSION,),
                    )
        except sqlite3.Error as exc:
            raise HarnessRecoveryStoreError("无法初始化 Harness recovery store") from exc

    def _ensure_writable(self) -> None:
        if self._legacy_read_only:
            raise HarnessRecoveryContractError(
                "schema-v1 Harness recovery 只能通过 legacy reader 只读恢复"
            )

    def _invocation_key_for(self, identity: HarnessRunIdentity) -> str:
        """按当前 store schema 选择 canonical 或 schema-v1 定位键。"""

        return _invocation_key(identity, legacy=self._legacy_read_only)

    def _identity_from_row(self, row: sqlite3.Row) -> HarnessRunIdentity:
        try:
            value = json.loads(row["identity_json"])
            if self._legacy_read_only:
                return SchemaV1LegacyHarnessRecoveryReader.read_identity(value)
            return _deserialize_identity(value)
        except (TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
            raise HarnessRecoveryStoreError("Harness recovery identity 已损坏") from exc

    def _snapshot_from_row(self, row: sqlite3.Row) -> HarnessRecoverySnapshot:
        try:
            value = json.loads(row["snapshot_json"])
            if self._legacy_read_only:
                return SchemaV1LegacyHarnessRecoveryReader.read_snapshot(value)
            return _deserialize_snapshot(value)
        except (TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
            raise HarnessRecoveryStoreError("Harness recovery snapshot 已损坏") from exc

    def _validated_snapshot(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> HarnessRecoverySnapshot:
        """从事件日志重建并核对当前快照，避免 projection 成为第二事实源。"""

        identity = self._identity_from_row(row)
        invocation_key = self._invocation_key_for(identity)
        try:
            event_rows = connection.execute(
                """
                SELECT sequence, event_id, transition_type, event_digest, payload_json
                FROM harness_recovery_events
                WHERE invocation_key = ?
                ORDER BY sequence ASC
                """,
                (invocation_key,),
            ).fetchall()
            if not event_rows:
                raise HarnessRecoveryStoreError("Harness recovery 缺少初始事件")
            current = _initial_snapshot(identity)
            for expected_sequence, event_row in enumerate(event_rows, start=1):
                if event_row["sequence"] != expected_sequence:
                    raise HarnessRecoveryStoreError("Harness recovery 事件序号不连续")
                if expected_sequence == 1:
                    if event_row["transition_type"] != "invocation_started":
                        raise HarnessRecoveryStoreError("Harness recovery 初始事件类型损坏")
                    try:
                        payload = json.loads(event_row["payload_json"])
                        payload = _strict_mapping(payload, {"identity"}, "invocation payload")
                        payload_identity = (
                            SchemaV1LegacyHarnessRecoveryReader.read_identity(payload["identity"])
                            if self._legacy_read_only
                            else _deserialize_identity(payload["identity"])
                        )
                    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                        raise HarnessRecoveryStoreError(
                            "Harness recovery 初始事件内容损坏"
                        ) from exc
                    expected_event_id = (
                        "invocation-"
                        + _digest(payload["identity"])
                        if self._legacy_read_only
                        else current.last_event_id
                    )
                    if event_row["event_id"] != expected_event_id:
                        raise HarnessRecoveryStoreError("Harness recovery 初始事件身份损坏")
                    _assert_identity(identity, payload_identity)
                    expected_digest = (
                        _digest(
                            {
                                "transition_type": "invocation_started",
                                "identity": payload["identity"],
                            }
                        )
                        if self._legacy_read_only
                        else current.last_event_digest
                    )
                    if event_row["event_digest"] != expected_digest:
                        raise HarnessRecoveryStoreError("Harness recovery 初始事件摘要损坏")
                    if self._legacy_read_only:
                        current = _replace_snapshot(
                            current,
                            last_event_id=event_row["event_id"],
                            last_event_digest=event_row["event_digest"],
                            event_sequence=expected_sequence,
                        )
                    continue
                try:
                    transition_payload = json.loads(event_row["payload_json"])
                    transition = _deserialize_transition(
                        transition_payload,
                        legacy=self._legacy_read_only,
                    )
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise HarnessRecoveryStoreError(
                        "Harness recovery transition 内容损坏"
                    ) from exc
                if transition.event_id != event_row["event_id"]:
                    raise HarnessRecoveryStoreError("Harness recovery event id 已损坏")
                if transition.transition_type != event_row["transition_type"]:
                    raise HarnessRecoveryStoreError("Harness recovery event type 已损坏")
                expected_digest = (
                    _legacy_transition_digest(transition_payload)
                    if self._legacy_read_only
                    else transition.event_digest
                )
                if expected_digest != event_row["event_digest"]:
                    raise HarnessRecoveryStoreError("Harness recovery event digest 已损坏")
                try:
                    current = _apply_transition(current, transition)
                except HarnessRecoveryStoreError as exc:
                    raise HarnessRecoveryStoreError(
                        "Harness recovery 事件序列无法重建"
                    ) from exc
                current = _replace_snapshot(
                    current,
                    last_event_id=transition.event_id,
                    last_event_digest=event_row["event_digest"],
                    event_sequence=expected_sequence,
                )
            stored = self._snapshot_from_row(row)
            if stored != current:
                raise HarnessRecoveryStoreError(
                    "Harness recovery 当前快照与不可变事件日志不一致"
                )
            if row["state"] != stored.state.value or row["last_sequence"] != stored.event_sequence:
                raise HarnessRecoveryStoreError("Harness recovery projection 索引不一致")
            return stored
        except HarnessRecoveryStoreError:
            raise
        except sqlite3.Error as exc:
            raise HarnessRecoveryStoreError("无法读取 Harness recovery 事件") from exc

    def create_or_load(self, identity: HarnessRunIdentity) -> HarnessRecoverySnapshot:
        """原子创建 invocation，或精确读取已有 invocation。"""

        if not isinstance(identity, HarnessRunIdentity):
            raise HarnessRecoveryContractError("identity 必须是 HarnessRunIdentity")
        self._ensure_writable()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM harness_recovery_runs WHERE invocation_key = ?",
                    (self._invocation_key_for(identity),),
                ).fetchone()
                if row is not None:
                    stored_identity = self._identity_from_row(row)
                    _assert_identity(identity, stored_identity)
                    return self._validated_snapshot(connection, row)
                snapshot = _initial_snapshot(identity)
                connection.execute(
                    """
                    INSERT INTO harness_recovery_runs
                        (invocation_key, identity_json, snapshot_json, state, last_sequence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self._invocation_key_for(identity),
                        _canonical_json(_serialize_identity(identity)),
                        _canonical_json(_serialize_snapshot(snapshot)),
                        snapshot.state.value,
                        snapshot.event_sequence,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO harness_recovery_events
                        (invocation_key, sequence, event_id, transition_type, event_digest, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._invocation_key_for(identity),
                        snapshot.event_sequence,
                        snapshot.last_event_id,
                        "invocation_started",
                        snapshot.last_event_digest,
                        _canonical_json({"identity": _serialize_identity(identity)}),
                    ),
                )
                return snapshot
        except HarnessRecoveryStoreError:
            raise
        except sqlite3.Error as exc:
            raise HarnessRecoveryStoreError("无法创建或读取 Harness invocation") from exc

    def load(self, identity: HarnessRunIdentity) -> HarnessRecoverySnapshot | None:
        """不创建新数据地读取 invocation；身份不一致时 fail-closed。"""

        if not isinstance(identity, HarnessRunIdentity):
            raise HarnessRecoveryContractError("identity 必须是 HarnessRunIdentity")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM harness_recovery_runs WHERE invocation_key = ?",
                    (self._invocation_key_for(identity),),
                ).fetchone()
                if row is None:
                    return None
                stored_identity = self._identity_from_row(row)
                _assert_identity(identity, stored_identity)
                return self._validated_snapshot(connection, row)
        except HarnessRecoveryStoreError:
            raise
        except sqlite3.Error as exc:
            raise HarnessRecoveryStoreError("无法读取 Harness invocation") from exc

    def read_snapshot(self, identity: HarnessRunIdentity) -> HarnessRecoverySnapshot:
        """读取必须存在的 invocation 快照；缺失不会被伪造成失败。"""

        snapshot = self.load(identity)
        if snapshot is None:
            raise HarnessRecoveryNotFound(
                f"Harness invocation 不存在：{identity.harness_invocation_id}"
            )
        return snapshot

    def read_completed_provider_receipt(
        self,
        identity: HarnessRunIdentity,
        attempt: ProviderAttemptReference,
    ) -> ReceiptReference | None:
        """从不可变事件读取指定 completed attempt 的 provider receipt。"""

        if not isinstance(identity, HarnessRunIdentity):
            raise HarnessRecoveryContractError("identity 必须是 HarnessRunIdentity")
        if not isinstance(attempt, ProviderAttemptReference):
            raise HarnessRecoveryContractError("attempt 必须是 ProviderAttemptReference")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM harness_recovery_runs WHERE invocation_key = ?",
                    (self._invocation_key_for(identity),),
                ).fetchone()
                if row is None:
                    raise HarnessRecoveryNotFound(
                        f"Harness invocation 不存在：{identity.harness_invocation_id}"
                    )
                stored_identity = self._identity_from_row(row)
                _assert_identity(identity, stored_identity)
                self._validated_snapshot(connection, row)
                event_rows = connection.execute(
                    """
                    SELECT payload_json
                    FROM harness_recovery_events
                    WHERE invocation_key = ? AND transition_type = ?
                    ORDER BY sequence DESC
                    """,
                    (
                        self._invocation_key_for(identity),
                        CloseProviderAttempt.transition_type,
                    ),
                ).fetchall()
                for event_row in event_rows:
                    transition = _deserialize_transition(
                        json.loads(event_row["payload_json"]),
                        legacy=self._legacy_read_only,
                    )
                    if (
                        isinstance(transition, CloseProviderAttempt)
                        and transition.attempt == attempt
                        and transition.outcome is ProviderAttemptOutcome.COMPLETED
                    ):
                        return transition.receipt
                return None
        except HarnessRecoveryStoreError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
            raise HarnessRecoveryStoreError(
                "无法读取 completed provider receipt"
            ) from exc

    def read_submitted_jobs_for_provider_attempt(
        self,
        identity: HarnessRunIdentity,
        attempt: ProviderAttemptReference,
    ) -> tuple[JobReference, ...]:
        """按不可变事件顺序读取指定 provider attempt 实际提交的 job。"""

        if not isinstance(identity, HarnessRunIdentity):
            raise HarnessRecoveryContractError("identity 必须是 HarnessRunIdentity")
        if not isinstance(attempt, ProviderAttemptReference):
            raise HarnessRecoveryContractError("attempt 必须是 ProviderAttemptReference")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM harness_recovery_runs WHERE invocation_key = ?",
                    (self._invocation_key_for(identity),),
                ).fetchone()
                if row is None:
                    raise HarnessRecoveryNotFound(
                        f"Harness invocation 不存在：{identity.harness_invocation_id}"
                    )
                stored_identity = self._identity_from_row(row)
                _assert_identity(identity, stored_identity)
                self._validated_snapshot(connection, row)
                event_rows = connection.execute(
                    """
                    SELECT payload_json
                    FROM harness_recovery_events
                    WHERE invocation_key = ? AND transition_type IN (?, ?)
                    ORDER BY sequence ASC
                    """,
                    (
                        self._invocation_key_for(identity),
                        SubmitJob.transition_type,
                        ReconcileCompletedProviderTurn.transition_type,
                    ),
                ).fetchall()
                jobs: list[JobReference] = []
                other_attempt_job_ids: set[str] = set()
                reconciled_jobs: tuple[JobReference, ...] | None = None
                for event_row in event_rows:
                    transition = _deserialize_transition(
                        json.loads(event_row["payload_json"]),
                        legacy=self._legacy_read_only,
                    )
                    if isinstance(transition, SubmitJob):
                        if transition.attempt == attempt:
                            jobs.append(transition.job)
                        else:
                            other_attempt_job_ids.add(transition.job.minichem_job_id)
                        continue
                    if (
                        isinstance(transition, ReconcileCompletedProviderTurn)
                        and transition.attempt == attempt
                    ):
                        reconciled_jobs = tuple(
                            job for job, _receipt in transition.collected_jobs
                        )
                if reconciled_jobs is not None:
                    # 旧 reconciliation 内容是 invocation 全量清单；减去其它
                    # attempt 的已知提交后，得到目标 completed turn 的序列。
                    return tuple(
                        job
                        for job in reconciled_jobs
                        if job.minichem_job_id not in other_attempt_job_ids
                    )
                return tuple(jobs)
        except HarnessRecoveryStoreError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
            raise HarnessRecoveryStoreError(
                "无法读取 provider attempt 的 submitted jobs"
            ) from exc

    def read_latest_suspension_reason(
        self,
        identity: HarnessRunIdentity,
    ) -> str | None:
        """从不可变事件读取最近一次显式 suspension 的结构化原因。"""

        if not isinstance(identity, HarnessRunIdentity):
            raise HarnessRecoveryContractError("identity 必须是 HarnessRunIdentity")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM harness_recovery_runs WHERE invocation_key = ?",
                    (self._invocation_key_for(identity),),
                ).fetchone()
                if row is None:
                    raise HarnessRecoveryNotFound(
                        f"Harness invocation 不存在：{identity.harness_invocation_id}"
                    )
                stored_identity = self._identity_from_row(row)
                _assert_identity(identity, stored_identity)
                self._validated_snapshot(connection, row)
                event_row = connection.execute(
                    """
                    SELECT payload_json
                    FROM harness_recovery_events
                    WHERE invocation_key = ? AND transition_type = ?
                    ORDER BY sequence DESC
                    LIMIT 1
                    """,
                    (
                        self._invocation_key_for(identity),
                        SuspendInvocation.transition_type,
                    ),
                ).fetchone()
                if event_row is None:
                    return None
                transition = _deserialize_transition(
                    json.loads(event_row["payload_json"]),
                    legacy=self._legacy_read_only,
                )
                if not isinstance(transition, SuspendInvocation):
                    raise HarnessRecoveryStoreError(
                        "最近 suspension 事件类型无法重建"
                    )
                return transition.reason_code
        except HarnessRecoveryStoreError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
            raise HarnessRecoveryStoreError("无法读取 suspension 原因") from exc

    def append_transition(
        self,
        identity: HarnessRunIdentity,
        transition: RecoveryTransitionType,
    ) -> HarnessRecoverySnapshot:
        """原子追加一条 typed transition 并返回更新后的快照。"""

        if not isinstance(identity, HarnessRunIdentity):
            raise HarnessRecoveryContractError("identity 必须是 HarnessRunIdentity")
        if not isinstance(transition, RecoveryTransition):
            raise HarnessRecoveryContractError("transition 必须是 typed RecoveryTransition")
        self._ensure_writable()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM harness_recovery_runs WHERE invocation_key = ?",
                    (self._invocation_key_for(identity),),
                ).fetchone()
                if row is None:
                    raise HarnessRecoveryNotFound(
                        f"Harness invocation 不存在：{identity.harness_invocation_id}"
                    )
                stored_identity = self._identity_from_row(row)
                _assert_identity(identity, stored_identity)
                current = self._validated_snapshot(connection, row)
                existing = connection.execute(
                    """
                    SELECT event_id, event_digest
                    FROM harness_recovery_events
                    WHERE invocation_key = ? AND event_id = ?
                    """,
                    (self._invocation_key_for(identity), transition.event_id),
                ).fetchone()
                if existing is not None:
                    if existing["event_digest"] != transition.event_digest:
                        raise HarnessRecoveryTransitionConflict(
                            "同一 event_id 的 transition 内容摘要发生漂移"
                        )
                    return current
                duplicate_digest = connection.execute(
                    """
                    SELECT event_id
                    FROM harness_recovery_events
                    WHERE invocation_key = ? AND event_digest = ?
                    """,
                    (self._invocation_key_for(identity), transition.event_digest),
                ).fetchone()
                if duplicate_digest is not None:
                    raise HarnessRecoveryTransitionConflict(
                        "同一 transition 内容必须复用原 event_id"
                    )
                updated = _apply_transition(current, transition)
                updated = _replace_snapshot(
                    updated,
                    last_event_id=transition.event_id,
                    last_event_digest=transition.event_digest,
                    event_sequence=current.event_sequence + 1,
                )
                connection.execute(
                    """
                    INSERT INTO harness_recovery_events
                        (invocation_key, sequence, event_id, transition_type, event_digest, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._invocation_key_for(identity),
                        updated.event_sequence,
                        transition.event_id,
                        transition.transition_type,
                        transition.event_digest,
                        _canonical_json(_serialize_transition(transition)),
                    ),
                )
                connection.execute(
                    """
                    UPDATE harness_recovery_runs
                    SET snapshot_json = ?, state = ?, last_sequence = ?
                    WHERE invocation_key = ?
                    """,
                    (
                        _canonical_json(_serialize_snapshot(updated)),
                        updated.state.value,
                        updated.event_sequence,
                        self._invocation_key_for(identity),
                    ),
                )
                return updated
        except HarnessRecoveryStoreError:
            raise
        except sqlite3.IntegrityError as exc:
            raise HarnessRecoveryTransitionConflict(
                "并发 transition 与当前 event 身份发生冲突"
            ) from exc
        except sqlite3.Error as exc:
            raise HarnessRecoveryStoreError("无法追加 Harness recovery transition") from exc


__all__ = [
    "BindSession",
    "CloseProviderAttempt",
    "CollectJob",
    "CompleteInvocation",
    "EvidenceVerificationReference",
    "FailInvocation",
    "HarnessLifecycle",
    "HarnessRecoveryContractError",
    "HarnessRecoveryIdentityConflict",
    "HarnessRecoveryNotFound",
    "HarnessRecoverySnapshot",
    "HarnessRecoveryStateConflict",
    "HarnessRecoveryStore",
    "HarnessRecoveryStoreError",
    "HarnessRecoveryTransition",
    "HarnessRecoveryTransitionConflict",
    "HarnessRunIdentity",
    "JobReference",
    "ModuleOutputReference",
    "ProviderAttemptOutcome",
    "ProviderAttemptReference",
    "ReconcileCompletedProviderTurn",
    "RecordManifestRead",
    "ReceiptKind",
    "ReceiptReference",
    "RecoveryTransitionType",
    "CommitModuleOutput",
    "SessionReference",
    "SchemaV1LegacyHarnessRecoveryReader",
    "StartProviderAttempt",
    "SubmitJob",
    "SuspendInvocation",
    "VerifyEvidence",
]
