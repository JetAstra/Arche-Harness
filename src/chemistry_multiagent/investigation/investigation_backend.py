"""Investigation V2-native Harness backend。

本模块是 Investigation Harness 的一个窄而完整的内部实现。Controller 只会把
``InvestigationHarnessRequest`` 交给它；session、thread、turn、scope、job、submission
和 receipt 都只在本模块与 ``HarnessRecoveryStore`` 之间流动。这里没有
schema-v10 Working Memory、旧 coordinator 或科学状态副本。

provider 和 execution transport 都是显式依赖。生产组装可以把
``InvestigationCodexRuntime``/MiniChem 适配成这里的 transport；测试则使用确定性的
内存 fake。一次作业的失败被编码为 action result 回传同一个 provider loop，只有
外部 owner 暂不可用或恢复身份经权威来源确认冲突时才离开这个 loop。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import re
from types import SimpleNamespace
from typing import Any, Literal, Protocol

from chemistry_multiagent.memory.v2_ledger import (
    PreparedModuleInvocationOutput,
    ScientificRecord,
)

from .async_jobs import InvestigationJobIdentity
from .contracts import InvestigationResponse, validate_investigation_response
from .harness_recovery_store import (
    BindSession,
    CloseProviderAttempt,
    CollectJob,
    EvidenceVerificationReference,
    HarnessLifecycle,
    HarnessRecoveryContractError,
    HarnessRecoveryIdentityConflict,
    HarnessRecoverySnapshot,
    HarnessRecoveryStateConflict,
    HarnessRecoveryStore,
    HarnessRecoveryTransitionConflict,
    HarnessRunIdentity,
    JobReference,
    ProviderAttemptOutcome,
    ProviderAttemptReference,
    ReconcileCompletedProviderTurn,
    ReceiptKind,
    ReceiptReference,
    SessionReference,
    StartProviderAttempt,
    SubmitJob,
    SuspendInvocation,
    VerifyEvidence,
)
from .harness_runtime import (
    BackendCompleted,
    BackendFatalFailure,
    BackendResult,
    BackendSuspended,
    InvestigationHarnessRequest,
)
from .recovery_contract import (
    Consistent,
    RecoveryEvidenceMissing,
    RecoveryVerification,
    Reconciled,
    TemporarilyUnavailable,
    VerifiedIntegrityConflict,
)


__all__ = [
    "EvidenceResultRejected",
    "NativeAction",
    "NativeExecutionTransport",
    "NativeProviderTransport",
    "NativeSubmissionContractChecker",
    "NativeTurnResult",
    "InvestigationSessionRunner",
    "create_v2_native_investigation_backend",
]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TERMINAL_SUCCESS = frozenset(
    {"completed", "complete", "succeeded", "success", "done", "ready"}
)
_TERMINAL_FAILURE = frozenset(
    {
        "failed",
        "failure",
        "error",
        "cancelled",
        "canceled",
        "expired",
        "aborted",
        "timeout",
        "timed-out",
    }
)
_TERMINAL_STATUSES = _TERMINAL_SUCCESS | _TERMINAL_FAILURE
_MODEL_SUSPENSION_STATUSES = frozenset(
    {
        "idle",
        "paused",
        "blocked",
        "manual_blocked",
        "human_blocked",
        "usage_limited",
        "budget_limited",
    }
)
_NO_PROGRESS_LIMIT = 2
_UNAVAILABLE_EXCEPTION_NAMES = frozenset(
    {
        "connectionerror",
        "timeouterror",
        "oserror",
        "brokenpipeerror",
        "investigationminichemunavailable",
    }
)
_PRIVATE_OUTPUT_KEYS = frozenset(
    {
        "session_id",
        "sessionid",
        "thread_id",
        "threadid",
        "scope_id",
        "scopeid",
        "workspace_ref",
        "workspaceref",
        "credential",
        "credentials",
        "token",
        "transcript",
        "raw_output",
        "rawoutput",
        "stdout",
        "stderr",
        "path",
        "paths",
    }
)


def _canonical_json(value: Any) -> str:
    """把 transport 可见的 JSON 结构编码成稳定文字。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    """计算稳定的 SHA-256 摘要。"""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_text(value: Any, field_name: str) -> str:
    """校验内部不透明引用或 provider 身份文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    normalized = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError(f"{field_name} 不能包含控制字符")
    return normalized


def _safe_ref(value: Any, field_name: str) -> str:
    """校验可以进入 Recovery Store 的不透明引用。"""

    normalized = _safe_text(value, field_name)
    if _SAFE_REF_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} 不是合法不透明引用")
    return normalized


def _safe_code(value: Any, field_name: str, default: str) -> str:
    """把 transport 错误码收敛成安全结构化代码。"""

    if isinstance(value, str):
        normalized = value.strip()
        if _SAFE_CODE_RE.fullmatch(normalized):
            return normalized
    return default


def _copy_json_object(value: Any, field_name: str) -> dict[str, Any]:
    """复制严格 JSON object，避免 fake/transport 反向修改 backend 输入。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} 必须是 JSON object")
    try:
        copied = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} 不是严格 JSON object") from exc
    if not isinstance(copied, dict):  # pragma: no cover - canonical object 已保证
        raise ValueError(f"{field_name} 必须是 JSON object")
    return copied


def _mapping_from_value(value: Any, field_name: str) -> dict[str, Any]:
    """读取 mapping、``as_mapping`` 或 Pydantic model 的公开字段。"""

    if isinstance(value, Mapping):
        return _copy_json_object(value, field_name)
    as_mapping = getattr(value, "as_mapping", None)
    if callable(as_mapping):
        return _copy_json_object(as_mapping(), field_name)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _copy_json_object(model_dump(), field_name)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _copy_json_object(to_dict(), field_name)
    raise ValueError(f"{field_name} 必须是公开 mapping")


def _assert_public_output_value(value: Any, field_name: str) -> None:
    """拒绝 verifier 摘要把 Harness 私有身份或本机资料带进科学 payload。"""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold().replace("-", "_") in _PRIVATE_OUTPUT_KEYS:
                raise ValueError(f"{field_name} 含 Harness 私有字段")
            _assert_public_output_value(nested, f"{field_name}[]")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _assert_public_output_value(nested, f"{field_name}[]")


def _read_field(value: Any, *names: str) -> Any:
    """兼容 snake_case、camelCase 和简单对象属性。"""

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _normalize_status(value: Any) -> str | None:
    """规范化 provider/MiniChem 的有限状态枚举。"""

    if value is None:
        return None
    normalized = getattr(value, "value", value)
    if not isinstance(normalized, str):
        return None
    return normalized.strip().casefold().replace("_", "-")


def _is_terminal(status: str | None) -> bool:
    """判断一个已由 owner 返回的作业状态是否终态。"""

    return status in _TERMINAL_STATUSES


def _is_success(status: str | None) -> bool:
    """判断作业是否以成功终态结束。"""

    return status in _TERMINAL_SUCCESS


def _is_unavailable_error(error: BaseException) -> bool:
    """判断异常是否表示外部 owner 暂时不可用。"""

    return isinstance(error, (ConnectionError, TimeoutError, OSError)) or (
        error.__class__.__name__.casefold() in _UNAVAILABLE_EXCEPTION_NAMES
    )


def _event_id(prefix: str, value: str) -> str:
    """生成与 Harness facade 相同的稳定 transition event id。"""

    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _request_identity(request: InvestigationHarnessRequest) -> HarnessRunIdentity:
    """把公开 Harness request 变成 Recovery Store 的严格 invocation 身份。"""

    input_digest = request.module_input_digest
    if _SHA256_RE.fullmatch(input_digest) is None:
        input_digest = _digest(
            {
                "declared_input_digest": input_digest,
                "module_kind": request.module_kind,
                "input_scientific_record_ids": list(
                    request.input_scientific_record_ids
                ),
            }
        )
    return HarnessRunIdentity(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        harness_invocation_id=request.harness_invocation_id,
        module_input_digest=input_digest,
    )


def _public_request(request: InvestigationHarnessRequest) -> dict[str, Any]:
    """只复制 Controller 允许交给 provider 的公开模块摘要。"""

    return request.to_dict()


@dataclass(frozen=True, slots=True)
class NativeAction:
    """provider 请求执行的一个动作。

    ``payload`` 是交给 execution transport 的请求正文；它不携带 session、scope 或
    凭据。transport 返回的 job identity 才是提交事实的权威来源。
    """

    action_id: str
    action_type: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _safe_ref(self.action_id, "action_id"))
        action_type = _safe_text(self.action_type, "action_type").casefold().replace("-", "_")
        aliases = {
            "submit": "submit_job",
            "submit_native_job": "submit_job",
            "submit_analysis_program": "submit_job",
            "submit_action_job": "submit_job",
        }
        object.__setattr__(self, "action_type", aliases.get(action_type, action_type))
        payload = _copy_json_object(self.payload, "action.payload")
        _assert_public_output_value(payload, "action.payload")
        object.__setattr__(self, "payload", payload)


@dataclass(frozen=True, slots=True)
class NativeTurnResult:
    """provider 一次 turn 的公开返回结构。

    ``accepted_job``/``completed_jobs`` 是 runtime 适配器从权威 MCP 回执提取的事实；
    普通 fake 可以只返回 ``actions``。``result`` 只能在完成 turn 时形成正式科学输出。
    """

    thread_id: str
    turn_id: str
    status: Literal[
        "completed",
        "interrupted",
        "failed",
        "idle",
        "paused",
        "blocked",
        "manual_blocked",
        "human_blocked",
        "usage_limited",
        "budget_limited",
    ] = "completed"
    result: Any | None = None
    actions: tuple[NativeAction, ...] = ()
    provider_receipt: Any | None = None
    attempt_id: str | None = None
    request_digest: str | None = None
    accepted_job: Mapping[str, Any] | None = None
    completed_jobs: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "thread_id", _safe_ref(self.thread_id, "thread_id"))
        object.__setattr__(self, "turn_id", _safe_ref(self.turn_id, "turn_id"))
        status = _safe_text(self.status, "provider turn status").casefold().replace("-", "_")
        if status not in {"completed", "interrupted", "failed"} | _MODEL_SUSPENSION_STATUSES:
            raise ValueError("provider turn status 不受支持")
        object.__setattr__(self, "status", status)
        normalized_actions: list[NativeAction] = []
        for action in self.actions:
            if isinstance(action, NativeAction):
                normalized_actions.append(action)
            elif isinstance(action, Mapping):
                normalized_actions.append(
                    NativeAction(
                        action_id=action.get("action_id", action.get("actionId")),
                        action_type=action.get("action_type", action.get("actionType")),
                        payload=action.get("payload", action.get("request", {})),
                    )
                )
            else:
                raise ValueError("provider turn action 不是 NativeAction")
        object.__setattr__(self, "actions", tuple(normalized_actions))
        if self.attempt_id is not None:
            object.__setattr__(self, "attempt_id", _safe_ref(self.attempt_id, "attempt_id"))
        if self.request_digest is not None and not _SHA256_RE.fullmatch(self.request_digest):
            raise ValueError("request_digest 必须是 64 位小写 SHA-256")
        if self.accepted_job is not None:
            object.__setattr__(
                self,
                "accepted_job",
                _copy_json_object(self.accepted_job, "accepted_job"),
            )
        normalized_completed: list[Mapping[str, Any]] = []
        for completed in self.completed_jobs:
            normalized_completed.append(_copy_json_object(completed, "completed_job"))
        object.__setattr__(self, "completed_jobs", tuple(normalized_completed))


class NativeProviderTransport(Protocol):
    """V2-native provider transport 的最小注入合同。"""

    def run_turn(
        self,
        request: Mapping[str, Any],
        *,
        session: SessionReference,
        action_results: tuple[Mapping[str, Any], ...],
        is_resume: bool,
        thread_id: str | None = None,
    ) -> NativeTurnResult | Mapping[str, Any] | Any:
        """运行一次 provider turn；不得自行持久化 Harness 恢复状态。"""


class NativeExecutionTransport(Protocol):
    """MiniChem/Gaussian/programmable_analysis fake 或适配器的最小合同。"""

    def submit_job(self, action: NativeAction, *, session: SessionReference) -> Mapping[str, Any]:
        """提交一次物理作业并返回权威身份。"""

    def query_job(self, job: JobReference, *, session: SessionReference) -> Mapping[str, Any]:
        """查询同一物理作业的权威状态。"""

    def collect_job(self, job: JobReference, *, session: SessionReference) -> Mapping[str, Any]:
        """收集同一物理作业的终态回执。"""


class NativeSubmissionContractChecker(Protocol):
    """即将新提交 Action 时使用的局部冻结合同检查器。"""

    def check_submission_contract(
        self,
        action: NativeAction,
        *,
        session: SessionReference,
    ) -> Any:
        """核对当前 Action/Backend 合同；不执行全局 catalog/readiness 检查。"""


class EvidenceResultRejected(RuntimeError):
    """Evidence owner 已明确拒绝当前 Result 中的一组执行引用。

    该异常只携带可回传给同一 provider thread 的不透明引用和原因码，不包含
    workspace、日志、凭据或 artifact 正文。它与 Evidence owner 暂时不可用不同：
    前者要求模型修正 Result，后者保持 suspension 并重试同一 completed turn。
    """

    def __init__(
        self,
        *,
        reason_code: str,
        invalid_references: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized_reason = _safe_code(
            reason_code,
            "reason_code",
            "evidence_references_rejected",
        )
        normalized_references: list[Mapping[str, Any]] = []
        for index, reference in enumerate(invalid_references):
            copied = _copy_json_object(reference, f"invalid_references[{index}]")
            _assert_public_output_value(copied, f"invalid_references[{index}]")
            normalized_references.append(copied)
        if not normalized_references:
            raise ValueError("invalid_references 不能为空")
        super().__init__(normalized_reason)
        self.reason_code = normalized_reason
        self.invalid_references = tuple(normalized_references)

    def to_action_result(self) -> Mapping[str, Any]:
        """返回同一 Investigation thread 可消费的结构化修正反馈。"""

        return {
            "kind": "result_validation",
            "status": "failed",
            "reason_code": self.reason_code,
            "invalid_references": [dict(item) for item in self.invalid_references],
        }


@dataclass(frozen=True, slots=True)
class _AttemptCapture:
    """一次 provider 调用期间的短暂回调事实；调用返回后不作为状态保存。"""

    thread_id: str | None = None
    turn_id: str | None = None
    job_events: tuple[tuple[Literal["accepted", "collected"], Mapping[str, Any]], ...] = ()
    accepted_jobs: tuple[Mapping[str, Any], ...] = ()
    completed_jobs: tuple[Mapping[str, Any], ...] = ()


class _BackendSuspension(Exception):
    """内部控制流：保留 Open/Suspended，等待下一次恢复。"""

    def __init__(self, reason_code: str, verification: RecoveryVerification) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.verification = verification


class _ResultRepairRequired(Exception):
    """内部控制流：当前 Result 引用已被权威 verifier 明确拒绝。"""

    def __init__(self, action_result: Mapping[str, Any]) -> None:
        copied = _copy_json_object(action_result, "result repair action")
        super().__init__(copied.get("reason_code", "evidence_references_rejected"))
        self.action_result = copied


@dataclass(frozen=True, slots=True)
class _CompletedAttemptRepair:
    """completed-turn replay 已核对，但其 Result 需要同 thread 继续修正。"""

    snapshot: HarnessRecoverySnapshot
    action_result: Mapping[str, Any]


class _VerifiedConflict(Exception):
    """内部控制流：两个权威身份来源已确认冲突。"""

    def __init__(self, reason_code: str = "verified_integrity_conflict") -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class _ProviderCallFailure(Exception):
    """provider 返回无法安全解析的合同；不是模块失败。"""


@dataclass(frozen=True, slots=True)
class _EvidenceDetails:
    """科学输出使用的证据摘要；Recovery Store 只保存其中的引用和摘要。"""

    verification_ref: str
    verification_digest: str
    source_manifest_ref: str
    summary: str
    evidence_items: tuple[Mapping[str, Any], ...]
    verification_payload: Mapping[str, Any]
    artifact_refs: tuple[str, ...]


class InvestigationSessionRunner:
    """Investigation Harness 的 V2-native backend。

    backend 不创建第二个运行状态库。所有可恢复事实都通过传入的
    ``HarnessRecoveryStore`` 写入；对象自身只保存不可变依赖，turn loop 的局部
    action result 在一次调用结束后即丢弃。
    """

    def __init__(
        self,
        *,
        recovery_store: HarnessRecoveryStore,
        provider_transport: NativeProviderTransport | Any | None = None,
        execution_transport: NativeExecutionTransport | Any | None = None,
        evidence_verifier: Any | None = None,
        session_binder: Any | None = None,
        provider_request_builder: Callable[..., Mapping[str, Any]] | None = None,
        submission_contract_checker: NativeSubmissionContractChecker | Any | None = None,
        max_turns: int = 8,
        provider: NativeProviderTransport | Any | None = None,
        minichem_transport: NativeExecutionTransport | Any | None = None,
        minichem_gateway: NativeExecutionTransport | Any | None = None,
        evidence: Any | None = None,
        execution_contract_checker: NativeSubmissionContractChecker | Any | None = None,
        contract_checker: NativeSubmissionContractChecker | Any | None = None,
        require_evidence_verifier: bool = False,
    ) -> None:
        """绑定 Recovery Store 和显式 transport；不会访问环境变量或真实服务。"""

        if not isinstance(recovery_store, HarnessRecoveryStore):
            raise TypeError("recovery_store 必须是 HarnessRecoveryStore")
        selected_provider = provider_transport if provider_transport is not None else provider
        if selected_provider is None:
            raise ValueError("provider_transport 必须提供")
        selected_execution = (
            execution_transport if execution_transport is not None else minichem_transport
        )
        if selected_execution is None:
            selected_execution = minichem_gateway
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or not 1 <= max_turns <= 100:
            raise ValueError("max_turns 必须位于 1 到 100 之间")
        self._recovery_store = recovery_store
        self._provider = selected_provider
        self._execution = selected_execution
        self._evidence_verifier = evidence_verifier if evidence_verifier is not None else evidence
        if type(require_evidence_verifier) is not bool:
            raise TypeError("require_evidence_verifier 必须是 bool")
        self._strict_evidence = require_evidence_verifier
        self._session_binder = session_binder
        self._provider_request_builder = provider_request_builder
        self._submission_contract_checker = next(
            (
                candidate
                for candidate in (
                    submission_contract_checker,
                    execution_contract_checker,
                    contract_checker,
                )
                if candidate is not None
            ),
            None,
        )
        self._max_turns = max_turns

    @property
    def recovery_store(self) -> HarnessRecoveryStore:
        """返回该 backend 使用的唯一恢复事实 owner。"""

        return self._recovery_store

    def start(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """创建同一 invocation 的内部 session，并驱动 Investigation loop。"""

        return self._begin_invocation(request, harness_invocation_id, is_resume=False)

    def resume(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """从 Recovery Store 读取同一 session/job 后继续 Investigation loop。"""

        return self._begin_invocation(request, harness_invocation_id, is_resume=True)

    def _begin_invocation(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        *,
        is_resume: bool,
    ) -> BackendResult:
        """统一处理 start/resume 的身份核对、恢复和错误边界。"""

        if not isinstance(request, InvestigationHarnessRequest):
            raise TypeError("request 必须是 InvestigationHarnessRequest")
        if harness_invocation_id != request.harness_invocation_id:
            return self._fatal()
        identity = _request_identity(request)
        try:
            snapshot = (
                self._recovery_store.read_snapshot(identity)
                if is_resume
                else self._recovery_store.create_or_load(identity)
            )
        except Exception as error:
            if isinstance(error, HarnessRecoveryIdentityConflict):
                return self._fatal()
            if isinstance(error, HarnessRecoveryContractError):
                raise
            # store 暂时不可读不等于已确认模块失败；保持调用方可重试的边界。
            return BackendSuspended(
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_recovery_unreadable"
                ),
            )

        if snapshot.state is HarnessLifecycle.FAILED:
            return self._fatal()
        if snapshot.state is HarnessLifecycle.COMPLETED or snapshot.committed_output is not None:
            return BackendSuspended(
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="prepared_output_replay_required"
                ),
            )

        try:
            return self._drive_invocation(
                request,
                harness_invocation_id,
                identity,
                snapshot,
                is_resume=is_resume,
            )
        except _VerifiedConflict:
            return self._fatal()
        except _BackendSuspension as suspension:
            return self._record_suspension(identity, suspension)

    def _drive_invocation(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        identity: HarnessRunIdentity,
        snapshot: HarnessRecoverySnapshot,
        *,
        is_resume: bool,
    ) -> BackendResult:
        """以 Recovery Store 快照为唯一 durable 状态运行一条 provider loop。"""

        snapshot = self._ensure_session(
            request,
            harness_invocation_id,
            identity,
            snapshot,
        )
        action_results: tuple[Mapping[str, Any], ...] = ()
        reconciled = False
        consecutive_no_progress = 0

        if self._can_replay_completed_attempt(identity, snapshot):
            replayed = self._reconcile_completed_attempt(
                request,
                identity,
                snapshot,
            )
            if isinstance(replayed, BackendCompleted):
                return replayed
            snapshot = replayed.snapshot
            action_results = (replayed.action_result,)
            reconciled = True

        if snapshot.active_job is not None:
            action_results = self._recover_active_job(
                identity,
                snapshot.active_job,
                snapshot.session,
            )
            reconciled = True
            snapshot = self._read_snapshot(identity)

        for turn_index in range(self._max_turns):
            if snapshot.session is None:  # pragma: no cover - _ensure_session 已保证
                raise _BackendSuspension(
                    "recovery_evidence_missing",
                    RecoveryEvidenceMissing(reason_code="session_binding_missing"),
                )
            turn, capture = self._call_provider(
                identity,
                request,
                snapshot.session,
                action_results,
                is_resume=is_resume or reconciled or turn_index > 0,
                thread_id=(
                    snapshot.attempts[-1].model_thread_id
                    if snapshot.attempts
                    else None
                ),
            )
            attempt = self._ensure_attempt(
                identity,
                snapshot.session,
                request,
                turn,
                capture,
                action_results,
            )
            try:
                new_actions, has_pending_job = self._process_turn_jobs(
                    identity,
                    snapshot.session,
                    attempt,
                    turn,
                    capture,
                )
            except _BackendSuspension:
                self._close_completed_attempt(identity, attempt, turn)
                raise
            action_results = tuple(new_actions)

            if turn.status == "interrupted":
                self._append_transition(
                    identity,
                    CloseProviderAttempt(
                        _event_id("provider-close", attempt.model_attempt_id),
                        attempt,
                        ProviderAttemptOutcome.INTERRUPTED,
                    ),
                )
                raise _BackendSuspension(
                    "provider_interrupted",
                    TemporarilyUnavailable(reason_code="provider_interrupted"),
                )
            if turn.status == "failed":
                self._append_transition(
                    identity,
                    CloseProviderAttempt(
                        _event_id("provider-close", attempt.model_attempt_id),
                        attempt,
                        ProviderAttemptOutcome.FAILED,
                    ),
                )
                raise _BackendSuspension(
                    "provider_failed",
                    TemporarilyUnavailable(reason_code="provider_failed"),
                )

            if turn.status in _MODEL_SUSPENSION_STATUSES:
                self._close_completed_attempt(identity, attempt, turn)
                raise _BackendSuspension(
                    turn.status,
                    TemporarilyUnavailable(reason_code=turn.status),
                )

            self._close_completed_attempt(identity, attempt, turn)
            snapshot = self._read_snapshot(identity)

            if has_pending_job:
                raise _BackendSuspension(
                    "job_not_terminal",
                    TemporarilyUnavailable(reason_code="job_not_terminal"),
                )

            if turn.result is not None:
                try:
                    prepared, verification = self._prepare_module_output(
                        request,
                        snapshot,
                        turn.result,
                    )
                except _ResultRepairRequired as repair:
                    action_results = (repair.action_result,)
                    consecutive_no_progress = 0
                    is_resume = True
                    continue
                return BackendCompleted(prepared, recovery_verification=verification)

            if not action_results:
                consecutive_no_progress += 1
                if consecutive_no_progress >= _NO_PROGRESS_LIMIT:
                    raise _BackendSuspension(
                        "model_no_progress",
                        TemporarilyUnavailable(reason_code="model_no_progress"),
                    )
                # 普通 completed turn 没有产生科学结果或 Harness 协议事实时，
                # 由 Harness 在同一 thread 追加中性的 continuation。Controller
                # 不参与科学动作选择；下一轮收到该结构化提示后仍由 provider 决定。
                action_results = (
                    {
                        "kind": "neutral_continuation",
                        "status": "continue",
                        "reason_code": "model_no_progress",
                    },
                )
            else:
                consecutive_no_progress = 0

            # 新的 action result 将在下一轮作为明确输入回传给同一个 provider loop。
            is_resume = True

        raise _BackendSuspension(
            "provider_loop_limit",
            TemporarilyUnavailable(reason_code="provider_loop_limit"),
        )

    def _can_replay_completed_attempt(
        self,
        identity: HarnessRunIdentity,
        snapshot: HarnessRecoverySnapshot,
    ) -> bool:
        """判断旧的部分持久化状态能否从同一 completed turn 安全重放。"""

        attempt = snapshot.last_attempt
        if attempt is None:
            return False
        try:
            suspension_reason = self._recovery_store.read_latest_suspension_reason(
                identity
            )
        except HarnessRecoveryIdentityConflict as error:
            raise _VerifiedConflict() from error
        except HarnessRecoveryContractError:
            raise
        except Exception as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="completed_turn_receipt_unavailable"),
            ) from error
        # recovery_evidence_missing 和 provider_contract_unavailable 可能是
        # completed-turn replay 在进入模型前留下的旧暂停原因；只有下方 receipt、
        # attempt 和全部已收集作业同时匹配时才允许再重放。
        if suspension_reason not in {
            "recovery_state_unavailable",
            "recovery_evidence_missing",
            "evidence_verification_missing",
            "evidence_unavailable",
            "provider_contract_unavailable",
        }:
            return False
        provider_receipt = self._read_completed_provider_receipt(identity, attempt)
        return bool(
            (
                snapshot.active_job is not None
                or (
                    bool(snapshot.jobs)
                    and snapshot.collected_job_ids
                    == tuple(job.minichem_job_id for job in snapshot.jobs)
                    and snapshot.committed_output is None
                )
            )
            and snapshot.active_attempt is None
            and provider_receipt is not None
            and callable(getattr(self._provider, "replay_completed_turn", None))
        )

    def _read_completed_provider_receipt(
        self,
        identity: HarnessRunIdentity,
        attempt: ProviderAttemptReference,
    ) -> ReceiptReference | None:
        """读取不可变事件中的 completed provider receipt。"""

        try:
            return self._recovery_store.read_completed_provider_receipt(
                identity,
                attempt,
            )
        except HarnessRecoveryIdentityConflict as error:
            raise _VerifiedConflict() from error
        except HarnessRecoveryContractError:
            raise
        except Exception as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="completed_turn_receipt_unavailable"),
            ) from error

    def _read_submitted_jobs_for_attempt(
        self,
        identity: HarnessRunIdentity,
        attempt: ProviderAttemptReference,
    ) -> tuple[JobReference, ...]:
        """从不可变提交事件读取末次 completed turn 自己的 job 序列。"""

        try:
            return self._recovery_store.read_submitted_jobs_for_provider_attempt(
                identity,
                attempt,
            )
        except HarnessRecoveryIdentityConflict as error:
            raise _VerifiedConflict() from error
        except HarnessRecoveryContractError:
            raise
        except Exception as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(
                    reason_code="completed_turn_job_identity_unavailable"
                ),
            ) from error

    def _reconcile_completed_attempt(
        self,
        request: InvestigationHarnessRequest,
        identity: HarnessRunIdentity,
        snapshot: HarnessRecoverySnapshot,
    ) -> BackendCompleted | _CompletedAttemptRepair:
        """只读重放 completed provider turn，并原子补齐其作业事实。"""

        session = snapshot.session
        attempt = snapshot.last_attempt
        provider_receipt = (
            None
            if attempt is None
            else self._read_completed_provider_receipt(identity, attempt)
        )
        if session is None or attempt is None or provider_receipt is None:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="completed_turn_replay_identity_missing"),
            )
        turn, capture = self._call_completed_turn_replay(
            request,
            session,
            replay_turn_id=attempt.model_turn_id,
            thread_id=attempt.model_thread_id,
        )
        # 普通 Codex turn 不向 backend 暴露 provider receipt 的请求摘要，
        # 因此 attempt.model_request_digest 由 Harness 对公开请求计算；只读 replay
        # 则会返回 receipt 自己的原始请求摘要。两者属于不同身份层，不能直接
        # 比较。这里核对 session/thread/turn/attempt，并在下方单独核对不可变
        # provider receipt；这样既不会把 Result 返修误报成冲突，也不放宽
        # completed turn 的物理身份。
        if (
            turn.thread_id != attempt.model_thread_id
            or turn.turn_id != attempt.model_turn_id
            or (
                turn.attempt_id is not None
                and turn.attempt_id != attempt.model_attempt_id
            )
        ):
            raise _VerifiedConflict()
        if turn.status != "completed" or turn.result is None or turn.actions:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="completed_turn_replay_incomplete"),
            )
        if self._provider_receipt(attempt, turn) != provider_receipt:
            raise _VerifiedConflict()
        all_jobs_collected = bool(snapshot.jobs) and (
            snapshot.active_job is None
            and snapshot.collected_job_ids
            == tuple(job.minichem_job_id for job in snapshot.jobs)
        )
        if all_jobs_collected:
            expected_attempt_jobs = self._read_submitted_jobs_for_attempt(
                identity,
                attempt,
            )
            if capture.job_events:
                collected_jobs = self._build_reconciled_job_sequence(
                    session,
                    capture.job_events,
                )
                replayed_jobs = tuple(job for job, _receipt in collected_jobs)
                if replayed_jobs != expected_attempt_jobs:
                    raise _VerifiedConflict()
            elif expected_attempt_jobs:
                raise _BackendSuspension(
                    "recovery_evidence_missing",
                    RecoveryEvidenceMissing(
                        reason_code="completed_turn_replay_incomplete"
                    ),
                )
            # 所有 job 事实已持久化时，replay callbacks 只属于末次 attempt；
            # 不能把它们当成整个 invocation 的 job 清单再次 reconciliation。
        elif capture.job_events:
            collected_jobs = self._build_reconciled_job_sequence(
                session,
                capture.job_events,
            )
            snapshot = self._append_transition(
                identity,
                ReconcileCompletedProviderTurn(
                    _event_id("provider-reconcile", attempt.model_attempt_id),
                    attempt,
                    provider_receipt,
                    collected_jobs,
                ),
            )
        elif (
            snapshot.active_job is not None
            or not snapshot.jobs
            or snapshot.collected_job_ids
            != tuple(job.minichem_job_id for job in snapshot.jobs)
        ):
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(
                    reason_code="completed_turn_job_order_invalid"
                ),
            )
        try:
            prepared, verification = self._prepare_module_output(
                request,
                snapshot,
                turn.result,
            )
        except _ResultRepairRequired as repair:
            return _CompletedAttemptRepair(snapshot, repair.action_result)
        return BackendCompleted(prepared, recovery_verification=verification)

    def _call_completed_turn_replay(
        self,
        request: InvestigationHarnessRequest,
        session: SessionReference,
        *,
        replay_turn_id: str,
        thread_id: str,
    ) -> tuple[NativeTurnResult, _AttemptCapture]:
        """调用 provider 的只读 completed-turn replay 入口并保留事件顺序。"""

        capture_state: dict[str, Any] = {
            "thread_id": None,
            "turn_id": None,
            "job_events": [],
            "accepted_jobs": [],
            "completed_jobs": [],
        }

        def on_thread_ready(value: str) -> None:
            capture_state["thread_id"] = value

        def on_turn_started(value_thread_id: str, value_turn_id: str) -> None:
            capture_state["thread_id"] = value_thread_id
            capture_state["turn_id"] = value_turn_id

        def on_job_accepted(receipt: Mapping[str, Any]) -> None:
            copied = _copy_json_object(receipt, "accepted_job")
            capture_state["job_events"].append(("accepted", copied))
            capture_state["accepted_jobs"].append(copied)

        def on_job_collected(receipt: Mapping[str, Any]) -> None:
            copied = _copy_json_object(receipt, "completed_job")
            capture_state["job_events"].append(("collected", copied))
            capture_state["completed_jobs"].append(copied)

        raw_request = self._build_provider_request(
            request,
            session,
            (),
            is_resume=True,
            thread_id=thread_id,
        )
        method = getattr(self._provider, "replay_completed_turn")
        try:
            raw = method(
                raw_request,
                scope_id=session.minichem_scope_id,
                replay_turn_id=replay_turn_id,
                on_thread_ready=on_thread_ready,
                on_turn_started=on_turn_started,
                on_job_accepted=on_job_accepted,
                on_job_collected=on_job_collected,
            )
        except Exception as error:
            if _is_unavailable_error(error):
                raise _BackendSuspension(
                    "provider_unavailable",
                    TemporarilyUnavailable(reason_code="provider_unavailable"),
                ) from error
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="completed_turn_replay_unavailable"),
            ) from error
        capture = _AttemptCapture(
            thread_id=capture_state["thread_id"],
            turn_id=capture_state["turn_id"],
            job_events=tuple(capture_state["job_events"]),
            accepted_jobs=tuple(capture_state["accepted_jobs"]),
            completed_jobs=tuple(capture_state["completed_jobs"]),
        )
        try:
            return self._coerce_turn(raw, capture), capture
        except Exception as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="completed_turn_replay_invalid"),
            ) from error

    def _build_reconciled_job_sequence(
        self,
        session: SessionReference,
        events: Sequence[tuple[Literal["accepted", "collected"], Mapping[str, Any]]],
    ) -> tuple[tuple[JobReference, ReceiptReference], ...]:
        """把严格交错的 replay callbacks 收敛成原子 reconciliation 内容。"""

        collected_jobs: list[tuple[JobReference, ReceiptReference]] = []
        pending_job: JobReference | None = None
        for event_type, raw_receipt in events:
            job = self._job_from_authoritative(raw_receipt, session)
            if job is None:
                raise _BackendSuspension(
                    "recovery_evidence_missing",
                    RecoveryEvidenceMissing(reason_code="completed_turn_job_identity_missing"),
                )
            if event_type == "accepted":
                if pending_job is not None:
                    raise _BackendSuspension(
                        "recovery_evidence_missing",
                        RecoveryEvidenceMissing(reason_code="completed_turn_job_order_invalid"),
                    )
                pending_job = job
                continue
            if pending_job != job:
                raise _VerifiedConflict()
            collected_jobs.append((job, self._job_receipt(job, raw_receipt)))
            pending_job = None
        if pending_job is not None or not collected_jobs:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="completed_turn_job_order_invalid"),
            )
        return tuple(collected_jobs)

    def _ensure_session(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        identity: HarnessRunIdentity,
        snapshot: HarnessRecoverySnapshot,
    ) -> HarnessRecoverySnapshot:
        """创建或重用同一 session；scope generation 只保存摘要。"""

        if snapshot.session is not None:
            if snapshot.session.research_task_id != request.research_task_id:
                raise _VerifiedConflict()
            return snapshot
        if self._session_binder is not None:
            raw_session = self._invoke_session_binder(
                request,
                harness_invocation_id,
                existing_session=None,
            )
            session = self._coerce_session(raw_session, request)
        else:
            seed = _digest(
                {
                    "research_task_id": identity.research_task_id,
                    "module_invocation_id": identity.module_invocation_id,
                    "harness_invocation_id": identity.harness_invocation_id,
                    "module_input_digest": identity.module_input_digest,
                }
            )
            session = SessionReference(
                research_task_id=request.research_task_id,
                investigation_context_id=f"context-{seed[:24]}",
                hypothesis_execution_input_record_id=(
                    request.input_scientific_record_ids[0]
                ),
                investigation_session_id=f"session-{seed[48:]}",
                workspace_identity=f"workspace-{seed[:24]}",
                minichem_scope_id=f"scope-{seed[24:48]}",
                scope_contract_digest=_digest(
                    {
                        "scope_contract": "native-investigation",
                        "research_task_id": request.research_task_id,
                    }
                ),
                capability_generation_digest=_digest(
                    {
                        "scope_id": f"scope-{seed[24:48]}",
                        "identity": identity.module_input_digest,
                    }
                ),
            )
        return self._append_transition(
            identity,
            BindSession(
                _event_id("session-bind", session.investigation_session_id),
                session,
            ),
        )

    def _invoke_session_binder(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        *,
        existing_session: SessionReference | None,
    ) -> Any:
        """调用可选 session/scope owner；只传递 backend 内部的私有引用。"""

        binder = self._session_binder
        method = getattr(binder, "bind_session", None)
        if not callable(method):
            method = binder if callable(binder) else None
        if method is None:
            raise ValueError("session_binder 必须提供 bind_session")
        try:
            return method(
                request,
                harness_invocation_id=harness_invocation_id,
                existing_session=existing_session,
            )
        except TypeError:
            # 小型 fake 可以只声明 request 和已有 session；这里仅在签名不接收
            # 关键字时重试，绝不吞掉 transport 内部的普通 TypeError。
            signature = inspect.signature(method)
            kwargs = {
                name: value
                for name, value in {
                    "harness_invocation_id": harness_invocation_id,
                    "existing_session": existing_session,
                }.items()
                if name in signature.parameters
            }
            return method(request, **kwargs)

    @staticmethod
    def _coerce_session(raw: Any, request: InvestigationHarnessRequest) -> SessionReference:
        """把 session owner 返回值严格收敛为 Recovery Store 引用。"""

        if isinstance(raw, SessionReference):
            session = raw
        else:
            value = _mapping_from_value(raw, "session")
            session = SessionReference(**value)
        if session.research_task_id != request.research_task_id:
            raise _VerifiedConflict()
        return session

    def _call_provider(
        self,
        identity: HarnessRunIdentity,
        request: InvestigationHarnessRequest,
        session: SessionReference,
        action_results: tuple[Mapping[str, Any], ...],
        *,
        is_resume: bool,
        thread_id: str | None,
    ) -> tuple[NativeTurnResult, _AttemptCapture]:
        """调用 provider transport，并把 runtime callback 收敛成短暂 capture。"""

        capture_state: dict[str, Any] = {
            "thread_id": None,
            "turn_id": None,
            "attempt": None,
            "job_events": [],
            "accepted_jobs": [],
            "completed_jobs": [],
            "terminal_turn": None,
        }

        def capture_attempt() -> _AttemptCapture:
            """冻结当前 callback 已确认的 provider 与 job 生命周期事实。"""

            return _AttemptCapture(
                thread_id=capture_state["thread_id"],
                turn_id=capture_state["turn_id"],
                job_events=tuple(capture_state["job_events"]),
                accepted_jobs=tuple(capture_state["accepted_jobs"]),
                completed_jobs=tuple(capture_state["completed_jobs"]),
            )

        def on_thread_ready(thread_id: str) -> None:
            capture_state["thread_id"] = thread_id

        def on_turn_started(thread_id: str, turn_id: str) -> None:
            capture_state["thread_id"] = thread_id
            capture_state["turn_id"] = turn_id
            turn = NativeTurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
            )
            capture_state["attempt"] = self._ensure_attempt(
                identity,
                session,
                request,
                turn,
                capture_attempt(),
                action_results,
            )

        def on_job_accepted(receipt: Mapping[str, Any]) -> None:
            copied = _copy_json_object(receipt, "accepted_job")
            capture_state["job_events"].append(("accepted", copied))
            capture_state["accepted_jobs"].append(copied)
            attempt = capture_state["attempt"]
            if isinstance(attempt, ProviderAttemptReference):
                self._record_authoritative_submission(
                    identity,
                    session,
                    attempt,
                    copied,
                )

        def on_job_collected(receipt: Mapping[str, Any]) -> None:
            copied = _copy_json_object(receipt, "completed_job")
            capture_state["job_events"].append(("collected", copied))
            capture_state["completed_jobs"].append(copied)
            if isinstance(capture_state["attempt"], ProviderAttemptReference):
                self._record_authoritative_collection(
                    identity,
                    session,
                    copied,
                    explicitly_collected=True,
                )

        def on_terminal_turn(
            terminal_thread_id: str,
            terminal_turn_id: str,
            status: str,
        ) -> None:
            """在 Codex 清理恢复阻断前持久化同一终态 attempt。"""

            if (
                capture_state["thread_id"] not in {None, terminal_thread_id}
                or capture_state["turn_id"] not in {None, terminal_turn_id}
            ):
                raise _VerifiedConflict()
            capture_state["thread_id"] = terminal_thread_id
            capture_state["turn_id"] = terminal_turn_id
            turn = NativeTurnResult(
                thread_id=terminal_thread_id,
                turn_id=terminal_turn_id,
                status=status,
            )
            if turn.status not in {"interrupted", "failed"}:
                raise _VerifiedConflict()
            attempt = self._ensure_attempt(
                identity,
                session,
                request,
                turn,
                capture_attempt(),
                action_results,
            )
            capture_state["attempt"] = attempt
            outcome = (
                ProviderAttemptOutcome.INTERRUPTED
                if turn.status == "interrupted"
                else ProviderAttemptOutcome.FAILED
            )
            self._append_transition(
                identity,
                CloseProviderAttempt(
                    _event_id("provider-close", attempt.model_attempt_id),
                    attempt,
                    outcome,
                ),
            )
            capture_state["terminal_turn"] = turn

        public_request = _public_request(request)
        try:
            # 真实 V2 runtime 使用专用的 ``run_investigation_turn``。部分
            # transport 为了复用类型合同会继承 ``NativeProviderTransport``；
            # 该 Protocol 的 ``run_turn`` 仅是无返回值的声明桩，不能优先
            # 当作真实实现调用，否则会把 None 误判成 provider 身份缺失。
            if callable(getattr(self._provider, "run_investigation_turn", None)):
                raw_request = self._build_provider_request(
                    request,
                    session,
                    action_results,
                    is_resume=is_resume,
                    thread_id=thread_id,
                )
                raw = self._provider.run_investigation_turn(
                    raw_request,
                    scope_id=session.minichem_scope_id,
                    on_thread_ready=on_thread_ready,
                    on_turn_started=on_turn_started,
                    on_terminal_turn=on_terminal_turn,
                    on_job_accepted=on_job_accepted,
                    on_job_collected=on_job_collected,
                )
            elif callable(getattr(self._provider, "run_turn", None)):
                raw = self._invoke_generic_provider_turn(
                    public_request,
                    session,
                    action_results,
                    is_resume=is_resume,
                    thread_id=thread_id,
                )
            else:
                raise ValueError("provider transport 缺少 run_investigation_turn 或 run_turn")
        except _VerifiedConflict:
            raise
        except Exception as error:
            terminal_turn = capture_state["terminal_turn"]
            if isinstance(terminal_turn, NativeTurnResult):
                return terminal_turn, capture_attempt()
            if _is_unavailable_error(error):
                raise _BackendSuspension(
                    "provider_unavailable",
                    TemporarilyUnavailable(reason_code="provider_unavailable"),
                ) from error
            raise _BackendSuspension(
                "provider_contract_unavailable",
                RecoveryEvidenceMissing(reason_code="provider_contract_unavailable"),
            ) from error

        capture = capture_attempt()
        try:
            turn = self._coerce_turn(raw, capture)
        except _BackendSuspension:
            raise
        except Exception as error:
            raise _BackendSuspension(
                "provider_contract_unavailable",
                RecoveryEvidenceMissing(reason_code="provider_contract_unavailable"),
            ) from error
        return turn, capture

    def _invoke_generic_provider_turn(
        self,
        public_request: Mapping[str, Any],
        session: SessionReference,
        action_results: tuple[Mapping[str, Any], ...],
        *,
        is_resume: bool,
        thread_id: str | None,
    ) -> Any:
        """调用普通 provider turn，并在支持时传入既有 thread 身份。"""

        method = self._provider.run_turn
        kwargs: dict[str, Any] = {
            "session": session,
            "action_results": tuple(action_results),
            "is_resume": is_resume,
        }
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if thread_id is not None and (accepts_var_kwargs or "thread_id" in parameters):
            kwargs["thread_id"] = thread_id
        return method(public_request, **kwargs)

    def _build_provider_request(
        self,
        request: InvestigationHarnessRequest,
        session: SessionReference,
        action_results: tuple[Mapping[str, Any], ...],
        *,
        is_resume: bool,
        thread_id: str | None,
    ) -> Mapping[str, Any]:
        """为真实 Codex runtime 提供组装层注入的 prompt/envelope。"""

        if self._provider_request_builder is None:
            # 默认只提供公开摘要；真实 runtime 应由组装层注入 builder，以便
            # 在调用边界外准备经过 Controller 核对的科学 prompt。
            return {
                "harness_request": _public_request(request),
                "action_results": [dict(item) for item in action_results],
                "resume": is_resume,
                "thread_id": thread_id,
            }
        builder = self._provider_request_builder
        try:
            parameters = inspect.signature(builder).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        candidates = {
            "session": session,
            "action_results": tuple(action_results),
            "is_resume": is_resume,
            "thread_id": thread_id,
        }
        kwargs = {
            name: value
            for name, value in candidates.items()
            if accepts_var_kwargs or name in parameters
        }
        return builder(request, **kwargs)

    @staticmethod
    def _coerce_turn(raw: Any, capture: _AttemptCapture) -> NativeTurnResult:
        """把 fake 或 ``InvestigationTurnResult`` 收敛到 backend turn 合同。"""

        if isinstance(raw, NativeTurnResult):
            return raw
        thread_id = _read_field(raw, "thread_id", "threadId") or capture.thread_id
        turn_id = _read_field(raw, "turn_id", "turnId") or capture.turn_id
        if thread_id is None or turn_id is None:
            raise _BackendSuspension(
                "provider_identity_missing",
                RecoveryEvidenceMissing(reason_code="provider_identity_missing"),
            )
        status = _normalize_status(_read_field(raw, "status")) or "completed"
        if status in _TERMINAL_SUCCESS:
            normalized_status = "completed"
        elif status in _TERMINAL_FAILURE or status in {"interrupted", "aborted"}:
            normalized_status = "interrupted" if status == "interrupted" else "failed"
        elif status.replace("-", "_") in _MODEL_SUSPENSION_STATUSES:
            normalized_status = status.replace("-", "_")
        else:
            normalized_status = "completed"
        accepted = _read_field(raw, "accepted_job", "acceptedJob")
        completed = _read_field(raw, "completed_jobs", "completedJobs")
        result = _read_field(raw, "result")
        if completed is None:
            completed = ()
        if isinstance(completed, (str, bytes)) or not isinstance(completed, Sequence):
            completed = ()
        accepted_mapping = None if accepted is None else _mapping_from_value(accepted, "accepted_job")
        completed_mappings = tuple(
            _mapping_from_value(item, "completed_job") for item in completed
        )
        return NativeTurnResult(
            thread_id=thread_id,
            turn_id=turn_id,
            status=normalized_status,
            result=result,
            provider_receipt=_read_field(raw, "provider_receipt", "receipt_handle"),
            attempt_id=_read_field(raw, "attempt_id", "attemptId"),
            request_digest=_read_field(raw, "request_digest", "requestDigest"),
            accepted_job=accepted_mapping,
            completed_jobs=completed_mappings,
        )

    def _ensure_attempt(
        self,
        identity: HarnessRunIdentity,
        session: SessionReference,
        request: InvestigationHarnessRequest,
        turn: NativeTurnResult,
        capture: _AttemptCapture,
        action_results: tuple[Mapping[str, Any], ...],
    ) -> ProviderAttemptReference:
        """保存本次 provider turn 的唯一 attempt 起始事实。"""

        snapshot = self._read_snapshot(identity)
        active_attempt = snapshot.active_attempt
        if active_attempt is not None:
            if (
                active_attempt.investigation_session_id
                != session.investigation_session_id
                or active_attempt.model_thread_id != turn.thread_id
                or active_attempt.model_turn_id != turn.turn_id
                or (
                    turn.attempt_id is not None
                    and turn.attempt_id != active_attempt.model_attempt_id
                )
                or (
                    turn.request_digest is not None
                    and turn.request_digest != active_attempt.model_request_digest
                )
            ):
                raise _VerifiedConflict()
            # transport 中断后的同一物理 turn 恢复必须复用首次已保存的请求摘要；
            # 本次 resume 携带的已收集 job 结果不能反向改写原 attempt 身份。
            return active_attempt
        attempt = self._provider_attempt_reference(
            request,
            session,
            turn,
            action_results=action_results,
        )
        self._append_transition(
            identity,
            StartProviderAttempt(
                _event_id("provider-start", attempt.model_attempt_id),
                attempt,
            ),
        )
        return attempt

    @staticmethod
    def _provider_attempt_reference(
        request: InvestigationHarnessRequest,
        session: SessionReference,
        turn: NativeTurnResult,
        *,
        action_results: tuple[Mapping[str, Any], ...],
    ) -> ProviderAttemptReference:
        """从公开请求和 provider 返回身份构造稳定 attempt 引用。"""

        attempt_id = turn.attempt_id or f"attempt-{_digest({'thread': turn.thread_id, 'turn': turn.turn_id})[:32]}"
        request_digest = turn.request_digest or _digest(
            {
                "request": _public_request(request),
                "action_results": [dict(item) for item in action_results],
                "thread_id": turn.thread_id,
                "turn_id": turn.turn_id,
            }
        )
        try:
            attempt = ProviderAttemptReference(
                investigation_session_id=session.investigation_session_id,
                model_thread_id=turn.thread_id,
                model_turn_id=turn.turn_id,
                model_attempt_id=attempt_id,
                model_request_digest=request_digest,
            )
        except (TypeError, ValueError) as error:
            raise _BackendSuspension(
                "provider_identity_missing",
                RecoveryEvidenceMissing(reason_code="provider_identity_missing"),
            ) from error
        return attempt

    def _process_turn_jobs(
        self,
        identity: HarnessRunIdentity,
        session: SessionReference,
        attempt: ProviderAttemptReference,
        turn: NativeTurnResult,
        capture: _AttemptCapture,
    ) -> tuple[list[Mapping[str, Any]], bool]:
        """保存权威 job receipt，并把失败/完成结果变成下一轮 action result。"""

        action_results: list[Mapping[str, Any]] = []
        pending = False
        job_events: list[tuple[Literal["accepted", "collected"], Mapping[str, Any]]] = []
        seen_events: set[tuple[str, Any, Any, Any]] = set()

        def append_job_event(
            event_type: Literal["accepted", "collected"],
            raw_receipt: Mapping[str, Any],
        ) -> None:
            values = self._job_identity_values(raw_receipt)
            identity_key = (
                event_type,
                values.get("job_id"),
                values.get("operation_id"),
                values.get("submission_key"),
            )
            # 没有任何身份的回执不能被合并；每一份都要经过严格的失败处理。
            if identity_key[1:] == (None, None, None) or identity_key not in seen_events:
                job_events.append((event_type, raw_receipt))
                seen_events.add(identity_key)

        # callback capture 是唯一保留真实事件顺序的来源。turn envelope 只补充
        # 未通过 callback 暴露的回执，并按 accepted→collected 的合同顺序追加。
        for event_type, raw_receipt in capture.job_events:
            append_job_event(event_type, raw_receipt)
        for raw_receipt in capture.accepted_jobs:
            append_job_event("accepted", raw_receipt)
        if turn.accepted_job is not None:
            append_job_event("accepted", turn.accepted_job)
        for raw_receipt in capture.completed_jobs:
            append_job_event("collected", raw_receipt)
        for raw_receipt in turn.completed_jobs:
            append_job_event("collected", raw_receipt)

        collected_event_seen = False
        # runtime callback 已经是权威 submit/collect 回执；普通 fake 则通过
        # turn envelope 或 actions 提供提交事实。这里必须按 callback 到达顺序
        # 逐条推进，Recovery Store 仍然只允许一个 active job。
        for event_type, raw_receipt in job_events:
            if event_type == "accepted":
                result, is_pending = self._record_authoritative_submission(
                    identity,
                    session,
                    attempt,
                    raw_receipt,
                )
                if result is not None:
                    action_results.append(result)
                pending = pending or is_pending
                continue
            collected_event_seen = True
            collected = self._record_authoritative_collection(
                identity,
                session,
                raw_receipt,
                explicitly_collected=True,
            )
            if collected is not None:
                job = self._job_from_authoritative(raw_receipt, session)
                if job is None:  # pragma: no cover - collection already validated above
                    raise _BackendSuspension(
                        "recovery_evidence_missing",
                        RecoveryEvidenceMissing(reason_code="collect_receipt_missing"),
                    )
                status = _normalize_status(_read_field(collected, "status", "state"))
                if status is None:
                    # Codex runtime 的 completed receipt 表示它已经在同一 turn
                    # 通过真实 get/collect 生命周期核验；该脱敏 receipt 不重复
                    # 携带 MiniChem 的 status 字段。
                    status = "completed"
                action_results.append(self._job_action_result(job, collected, status=status))

        for action in turn.actions:
            if action.action_type != "submit_job":
                action_results.append(
                    self._action_failure(action.action_id, "unsupported_action")
                )
                continue
            result, is_pending = self._submit_action_job(identity, session, attempt, action)
            if result is not None:
                action_results.append(result)
            pending = pending or is_pending

        if collected_event_seen:
            # accepted callback 先报告 pending、同一 turn 随后又报告完成时，
            # 不能把已经 collect 的 job 继续当成等待态。
            pending = self._read_snapshot(identity).active_job is not None

        return action_results, pending

    @staticmethod
    def _deduplicate_job_receipts(
        receipts: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """合并 runtime callback 与 turn envelope 的同一份 job receipt。"""

        unique: list[Mapping[str, Any]] = []
        seen: set[tuple[Any, Any, Any]] = set()
        for receipt in receipts:
            values = InvestigationSessionRunner._job_identity_values(receipt)
            key = (values.get("job_id"), values.get("operation_id"), values.get("submission_key"))
            if key == (None, None, None) or key not in seen:
                unique.append(receipt)
                seen.add(key)
        return unique

    def _submit_action_job(
        self,
        identity: HarnessRunIdentity,
        session: SessionReference,
        attempt: ProviderAttemptReference,
        action: NativeAction,
    ) -> tuple[Mapping[str, Any] | None, bool]:
        """调用 execution transport；单作业失败只产生 action result。"""

        if self._execution is None:
            raise _BackendSuspension(
                "minichem_unavailable",
                TemporarilyUnavailable(reason_code="minichem_unavailable"),
            )
        self._check_submission_contract(action, session=session)
        try:
            raw = self._call_execution_method("submit_job", action, session=session)
        except _BackendSuspension:
            raise
        except Exception as error:
            if _is_unavailable_error(error):
                raise _BackendSuspension(
                    "minichem_unavailable",
                    TemporarilyUnavailable(reason_code="minichem_unavailable"),
                ) from error
            return self._action_failure(action.action_id, "minichem_action_failed"), False
        try:
            receipt = _mapping_from_value(raw, "submit receipt")
        except (TypeError, ValueError):
            return self._action_failure(action.action_id, "minichem_receipt_missing"), False
        job = self._job_from_authoritative(receipt, session)
        status = _normalize_status(_read_field(receipt, "status", "state"))
        if job is None:
            if status in _TERMINAL_FAILURE:
                return (
                    self._action_failure(
                        action.action_id,
                        _safe_code(
                            _read_field(receipt, "error_code", "errorCode")
                            or _read_field(_read_field(receipt, "error"), "code"),
                            "minichem_error_code",
                            "minichem_action_failed",
                        ),
                    ),
                    False,
                )
            return self._action_failure(action.action_id, "minichem_receipt_missing"), False
        self._append_transition(
            identity,
            SubmitJob(
                _event_id("job-submit", job.minichem_job_id),
                attempt,
                job,
            ),
        )
        if not _is_terminal(status):
            return None, True
        collected = self._collect_terminal_job(identity, session, job, receipt)
        return self._job_action_result(job, collected, status=status), False

    def _check_submission_contract(
        self,
        action: NativeAction,
        *,
        session: SessionReference,
    ) -> None:
        """只在新 Action submit 前核对局部合同，不阻塞旧 job 恢复。"""

        checker = self._submission_contract_checker
        if checker is None and self._execution is not None:
            for name in (
                "check_submission_contract",
                "validate_submission_contract",
                "assert_submission_contract",
                "check_action_contract",
                "validate_action_contract",
                "check_backend_compatibility",
            ):
                candidate = getattr(self._execution, name, None)
                if callable(candidate):
                    checker = candidate
                    break
        if checker is None:
            return
        method = getattr(checker, "check_submission_contract", None)
        if not callable(method):
            method = getattr(checker, "validate_submission_contract", None)
        if not callable(method):
            method = checker if callable(checker) else None
        if method is None:
            raise _BackendSuspension(
                "minichem_contract_unavailable",
                RecoveryEvidenceMissing(reason_code="minichem_contract_unavailable"),
            )
        try:
            try:
                raw = method(action, session=session)
            except TypeError:
                signature = inspect.signature(method)
                if "session" in signature.parameters:
                    raise
                raw = method(action)
        except _VerifiedConflict:
            raise
        except Exception as error:
            if _is_unavailable_error(error):
                raise _BackendSuspension(
                    "minichem_unavailable",
                    TemporarilyUnavailable(reason_code="minichem_unavailable"),
                ) from error
            raise _BackendSuspension(
                "minichem_contract_unavailable",
                RecoveryEvidenceMissing(reason_code="minichem_contract_unavailable"),
            ) from error

        if raw is None or raw is True:
            return
        if raw is False:
            raise _BackendSuspension(
                "minichem_contract_incompatible",
                RecoveryEvidenceMissing(reason_code="minichem_contract_incompatible"),
            )
        if isinstance(raw, Mapping):
            compatible = _read_field(raw, "compatible", "is_compatible", "ok", "valid")
            if compatible is None:
                status = _normalize_status(_read_field(raw, "status", "state"))
                if status in {None, "compatible", "valid", "ok", "ready"}:
                    return
                compatible = status not in _TERMINAL_FAILURE
            if compatible is True:
                return
            reason_code = _safe_code(
                _read_field(raw, "reason_code", "reasonCode", "error_code", "errorCode"),
                "reason_code",
                "minichem_contract_incompatible",
            )
            raise _BackendSuspension(
                "minichem_contract_incompatible",
                RecoveryEvidenceMissing(reason_code=reason_code),
            )
        raise _BackendSuspension(
            "minichem_contract_unavailable",
            RecoveryEvidenceMissing(reason_code="minichem_contract_unavailable"),
        )

    def _record_authoritative_submission(
        self,
        identity: HarnessRunIdentity,
        session: SessionReference,
        attempt: ProviderAttemptReference,
        raw_receipt: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any] | None, bool]:
        """登记 runtime callback 给出的真实 submit 回执。"""

        job = self._job_from_authoritative(raw_receipt, session)
        if job is None:
            return self._action_failure(
                _safe_code(_read_field(raw_receipt, "action_id", "actionId"), "action_id", "provider_submit"),
                "minichem_receipt_missing",
            ), False
        self._append_transition(
            identity,
            SubmitJob(_event_id("job-submit", job.minichem_job_id), attempt, job),
        )
        status = _normalize_status(_read_field(raw_receipt, "status", "state"))
        if _is_terminal(status):
            collected = self._collect_terminal_job(identity, session, job, raw_receipt)
            return self._job_action_result(job, collected, status=status), False
        return None, True

    def _record_authoritative_collection(
        self,
        identity: HarnessRunIdentity,
        session: SessionReference,
        raw_receipt: Mapping[str, Any],
        *,
        explicitly_collected: bool = False,
    ) -> Mapping[str, Any] | None:
        """登记 runtime callback 给出的真实 collect 回执。

        ``explicitly_collected`` 只供 provider callback 使用：Codex runtime 的
        completed receipt 已经代表同一 turn 内完成的 get/collect，但为了脱敏不会
        复制原始 status。普通 query/collect 响应仍必须显式提供终态 status。
        """

        candidate = self._job_from_authoritative(raw_receipt, session)
        if candidate is None:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="collect_receipt_missing"),
            )
        snapshot = self._read_snapshot(identity)
        if snapshot.active_job is None:
            if candidate.minichem_job_id in snapshot.collected_job_ids:
                # callback 已在 provider 返回前即时持久化时，后续统一处理仍需
                # 看到同一份收集结果，才能把作业终态回传给下一轮模型。
                return raw_receipt
            raise _VerifiedConflict()
        job = snapshot.active_job
        self._assert_authoritative_job_identity(job, raw_receipt, session=session)
        status = _normalize_status(_read_field(raw_receipt, "status", "state"))
        if status is None and explicitly_collected:
            status = "completed"
        if not _is_terminal(status):
            raise _BackendSuspension(
                "job_not_terminal",
                TemporarilyUnavailable(reason_code="job_not_terminal"),
            )
        self._append_transition(
            identity,
            CollectJob(
                _event_id("job-collect", job.minichem_job_id),
                job,
                self._job_receipt(job, raw_receipt),
            ),
        )
        return raw_receipt

    def _recover_active_job(
        self,
        identity: HarnessRunIdentity,
        job: JobReference,
        session: SessionReference | None,
    ) -> tuple[Mapping[str, Any], ...]:
        """恢复时只查询和收集持久化的同一个 job，绝不重复 submit。"""

        if session is None:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="session_binding_missing"),
            )
        if self._execution is None:
            raise _BackendSuspension(
                "minichem_unavailable",
                TemporarilyUnavailable(reason_code="minichem_unavailable"),
            )
        try:
            raw_status = self._call_execution_method("query_job", job, session=session)
        except Exception as error:
            if _is_unavailable_error(error):
                raise _BackendSuspension(
                    "minichem_unavailable",
                    TemporarilyUnavailable(reason_code="minichem_unavailable"),
                ) from error
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="job_status_unavailable"),
            ) from error
        try:
            status_mapping = _mapping_from_value(raw_status, "job status")
        except (TypeError, ValueError) as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="job_status_invalid"),
            ) from error
        self._assert_authoritative_job_identity(job, status_mapping, session=session)
        status = _normalize_status(_read_field(status_mapping, "status", "state"))
        if not _is_terminal(status):
            raise _BackendSuspension(
                "job_not_terminal",
                TemporarilyUnavailable(reason_code="job_not_terminal"),
            )
        collected = self._collect_terminal_job(identity, session, job, status_mapping)
        return (self._job_action_result(job, collected, status=status),)

    def _collect_terminal_job(
        self,
        identity: HarnessRunIdentity,
        session: SessionReference,
        job: JobReference,
        status_mapping: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """收集权威终态并写入 CollectJob；失败回执仍然是可回传事实。"""

        if self._execution is None:
            raise _BackendSuspension(
                "minichem_unavailable",
                TemporarilyUnavailable(reason_code="minichem_unavailable"),
            )
        try:
            raw_collect = self._call_execution_method("collect_job", job, session=session)
        except Exception as error:
            if _is_unavailable_error(error):
                raise _BackendSuspension(
                    "minichem_unavailable",
                    TemporarilyUnavailable(reason_code="minichem_unavailable"),
                ) from error
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="job_collect_unavailable"),
            ) from error
        try:
            collect_mapping = _mapping_from_value(raw_collect, "job collect")
        except (TypeError, ValueError) as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="job_collect_invalid"),
            ) from error
        self._assert_authoritative_job_identity(job, collect_mapping, session=session)
        collect_status = _normalize_status(_read_field(collect_mapping, "status", "state"))
        if collect_status is None:
            collect_status = _normalize_status(_read_field(status_mapping, "status", "state"))
        if not _is_terminal(collect_status):
            raise _BackendSuspension(
                "job_not_terminal",
                TemporarilyUnavailable(reason_code="job_not_terminal"),
            )
        self._append_transition(
            identity,
            CollectJob(
                _event_id("job-collect", job.minichem_job_id),
                job,
                self._job_receipt(job, collect_mapping),
            ),
        )
        return collect_mapping

    def _call_execution_method(self, operation: str, value: Any, *, session: SessionReference) -> Any:
        """调用 execution transport 的规范名称或兼容别名。"""

        aliases = {
            "submit_job": ("submit_job", "submit"),
            "query_job": ("query_job", "get_job_status", "get_investigation_job_status", "query"),
            "collect_job": ("collect_job", "collect"),
        }
        method = None
        selected_name = ""
        for name in aliases[operation]:
            candidate = getattr(self._execution, name, None)
            if callable(candidate):
                method = candidate
                selected_name = name
                break
        if method is None:
            raise _BackendSuspension(
                "minichem_unavailable",
                TemporarilyUnavailable(reason_code="minichem_unavailable"),
            )
        if selected_name == "get_investigation_job_status":
            # InvestigationMiniChemGateway 的公开只读入口接收已有 waiting 领域
            # 对象，而不是 Harness 的私有 JobReference。这里仅构造调用适配器，
            # 不把它写入任何状态；job/session/scope 仍由 Recovery Store 所有。
            if not isinstance(value, JobReference):  # pragma: no cover - 仅供窄适配器
                raise TypeError("job status 查询需要 JobReference")
            value = self._gateway_waiting_view(value, session)
        try:
            return method(value, session=session)
        except TypeError:
            signature = inspect.signature(method)
            if "session" in signature.parameters:
                raise
            return method(value)

    @staticmethod
    def _gateway_waiting_view(job: JobReference, session: SessionReference) -> Any:
        """生成 MiniChem 只读网关所需的临时 waiting view。"""

        identity = InvestigationJobIdentity(
            job_id=job.minichem_job_id,
            operation_id=job.minichem_operation_id,
            submission_key=job.minichem_submission_key,
            job_family="native_execution",
            job_type="native_software",
            runtime="investigation",
        )
        return SimpleNamespace(
            task_id=session.research_task_id,
            batch_id=session.investigation_context_id,
            combination_id=session.hypothesis_execution_input_record_id,
            codex_thread_id=None,
            turn_id=None,
            scope_id=session.minichem_scope_id,
            job_family="native_execution",
            status_tool="get_execution_job",
            identity=identity,
        )

    @staticmethod
    def _job_from_authoritative(raw: Mapping[str, Any], session: SessionReference) -> JobReference | None:
        """只从 owner 回执读取 job identity，不采信模型声称的身份。"""

        values = InvestigationSessionRunner._job_identity_values(raw)
        job_id = values.get("job_id")
        operation_id = values.get("operation_id")
        submission_key = values.get("submission_key")
        request_digest = values.get("request_digest")
        if any(value is None for value in (job_id, operation_id, submission_key)):
            return None
        returned_session_id = values.get("session_id")
        if (
            returned_session_id is not None
            and returned_session_id != session.investigation_session_id
        ):
            raise _VerifiedConflict()
        returned_scope_id = values.get("scope_id")
        if returned_scope_id is not None and returned_scope_id != session.minichem_scope_id:
            raise _VerifiedConflict()
        if request_digest is None:
            request_digest = _digest(
                {
                    "job_id": job_id,
                    "operation_id": operation_id,
                    "submission_key": submission_key,
                }
            )
        try:
            job = JobReference(
                investigation_session_id=session.investigation_session_id,
                minichem_job_id=job_id,
                minichem_operation_id=operation_id,
                minichem_submission_key=submission_key,
                minichem_request_digest=request_digest,
            )
        except (TypeError, ValueError) as error:
            raise _BackendSuspension(
                "minichem_receipt_missing",
                RecoveryEvidenceMissing(reason_code="minichem_receipt_missing"),
            ) from error
        return job

    @staticmethod
    def _assert_authoritative_job_identity(
        job: JobReference,
        raw: Mapping[str, Any],
        *,
        session: SessionReference,
    ) -> None:
        """核对 query/collect 回执的 job、operation、submission、digest。"""

        actual = InvestigationSessionRunner._job_identity_values(raw)
        returned_session_id = actual.get("session_id")
        if (
            returned_session_id is not None
            and returned_session_id != job.investigation_session_id
        ):
            raise _VerifiedConflict()
        returned_scope_id = actual.get("scope_id")
        if returned_scope_id is not None and returned_scope_id != session.minichem_scope_id:
            raise _VerifiedConflict()
        for field_name, expected in (
            ("job_id", job.minichem_job_id),
            ("operation_id", job.minichem_operation_id),
            ("submission_key", job.minichem_submission_key),
        ):
            value = actual.get(field_name)
            if value is not None and value != expected:
                raise _VerifiedConflict()
        request_digest = actual.get("request_digest")
        if (
            request_digest is not None
            and request_digest != job.minichem_request_digest
        ):
            raise _VerifiedConflict()

    @staticmethod
    def _job_identity_values(raw: Mapping[str, Any]) -> dict[str, Any]:
        """提取 owner 回执中的公共 job identity。"""

        if not isinstance(raw, Mapping):
            raise ValueError("authoritative job receipt 必须是 JSON object")
        sources: list[Mapping[str, Any]] = [raw]
        for nested_name in ("identity", "job", "result"):
            nested = _read_field(raw, nested_name)
            if isinstance(nested, Mapping):
                sources.append(nested)
                nested_identity = _read_field(nested, "identity")
                if isinstance(nested_identity, Mapping):
                    sources.append(nested_identity)

        aliases = {
            "session_id": ("session_id", "sessionId"),
            "scope_id": ("scope_id", "scopeId"),
            "job_id": ("job_id", "jobId"),
            "operation_id": ("operation_id", "operationId"),
            "submission_key": ("submission_key", "submissionKey"),
            "request_digest": (
                "request_digest",
                "requestDigest",
                "submission_request_digest",
                "request_fingerprint",
            ),
        }
        values: dict[str, Any] = {}
        for field_name, field_aliases in aliases.items():
            candidates = [
                candidate
                for source in sources
                for candidate in (_read_field(source, *field_aliases),)
                if candidate is not None
            ]
            if candidates and any(candidate != candidates[0] for candidate in candidates[1:]):
                raise _VerifiedConflict()
            values[field_name] = candidates[0] if candidates else None
        return values

    @staticmethod
    def _job_receipt(job: JobReference, raw: Mapping[str, Any]) -> ReceiptReference:
        """从 collect 回执构造脱敏 MiniChem receipt reference。"""

        receipt_ref = _read_field(raw, "receipt_ref", "receiptRef") or f"receipt-{job.minichem_job_id}"
        receipt_digest = _read_field(raw, "receipt_digest", "receiptDigest")
        if not isinstance(receipt_digest, str) or _SHA256_RE.fullmatch(receipt_digest) is None:
            receipt_digest = _digest(
                {
                    "job": job.minichem_job_id,
                    "operation": job.minichem_operation_id,
                    "receipt": dict(raw),
                }
            )
        try:
            return ReceiptReference(
                kind=ReceiptKind.MINICHEM_OPERATION,
                receipt_ref=_safe_ref(receipt_ref, "receipt_ref"),
                receipt_digest=receipt_digest,
            )
        except (TypeError, ValueError) as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="minichem_receipt_missing"),
            ) from error

    @staticmethod
    def _job_action_result(
        job: JobReference,
        collected: Mapping[str, Any],
        *,
        status: str | None,
    ) -> Mapping[str, Any]:
        """把完成或失败的作业封装成下一轮 provider 可读 action result。"""

        normalized = _normalize_status(_read_field(collected, "status", "state")) or status
        if _is_success(normalized):
            result: dict[str, Any] = {
                "status": "completed",
                "job_id": job.minichem_job_id,
                "operation_id": job.minichem_operation_id,
                "submission_key": job.minichem_submission_key,
                "result": dict(collected),
            }
        else:
            error = _read_field(collected, "error")
            result = {
                "status": "failed",
                "job_id": job.minichem_job_id,
                "operation_id": job.minichem_operation_id,
                "submission_key": job.minichem_submission_key,
                "error_code": _safe_code(
                    _read_field(collected, "error_code", "errorCode")
                    or _read_field(error, "code"),
                    "error_code",
                    "minichem_job_failed",
                ),
                "result": dict(collected),
            }
        return result

    @staticmethod
    def _action_failure(action_id: str, error_code: str) -> Mapping[str, Any]:
        """构造不含异常正文的单动作失败回执。"""

        return {
            "action_id": _safe_code(action_id, "action_id", "provider_action"),
            "status": "failed",
            "error_code": _safe_code(error_code, "error_code", "action_failed"),
        }

    def _close_completed_attempt(
        self,
        identity: HarnessRunIdentity,
        attempt: ProviderAttemptReference,
        turn: NativeTurnResult,
    ) -> None:
        """写入 provider completed receipt；不把科学结果正文放入 Recovery Store。"""

        receipt = self._provider_receipt(attempt, turn)
        self._append_transition(
            identity,
            CloseProviderAttempt(
                _event_id("provider-close", attempt.model_attempt_id),
                attempt,
                ProviderAttemptOutcome.COMPLETED,
                receipt,
            ),
        )

    @staticmethod
    def _provider_receipt(attempt: ProviderAttemptReference, turn: NativeTurnResult) -> ReceiptReference:
        """构造 provider turn 的有限 receipt 引用。"""

        raw = turn.provider_receipt
        receipt_ref = _read_field(raw, "receipt_ref", "receiptRef") if raw is not None else None
        receipt_digest = _read_field(raw, "receipt_digest", "receiptDigest") if raw is not None else None
        if not isinstance(receipt_ref, str) or _SAFE_REF_RE.fullmatch(receipt_ref) is None:
            receipt_ref = f"provider-{attempt.model_attempt_id}"
        if not isinstance(receipt_digest, str) or _SHA256_RE.fullmatch(receipt_digest) is None:
            receipt_digest = _digest(
                {
                    "attempt_id": attempt.model_attempt_id,
                    "turn_id": attempt.model_turn_id,
                }
            )
        return ReceiptReference(
            kind=ReceiptKind.PROVIDER_TURN,
            receipt_ref=receipt_ref,
            receipt_digest=receipt_digest,
        )

    def _prepare_module_output(
        self,
        request: InvestigationHarnessRequest,
        snapshot: HarnessRecoverySnapshot,
        raw_result: Any,
    ) -> tuple[PreparedModuleInvocationOutput, RecoveryVerification]:
        """只生成成对 V2 Evidence/Result，不进入 DS Review 或 schema-v10。"""

        if isinstance(raw_result, PreparedModuleInvocationOutput):
            prepared = raw_result
            prepared_mapping = prepared.to_dict()
            if (
                prepared_mapping["module_invocation_id"]
                != request.module_invocation_id
                or prepared_mapping["module_input_digest"]
                != request.module_input_digest
            ):
                raise _VerifiedConflict()
            self._assert_investigation_pairs(prepared.output_scientific_records)
            response = None
            collected_jobs = tuple(
                job
                for job in snapshot.jobs
                if job.minichem_job_id in snapshot.collected_job_ids
            )
            if not collected_jobs:
                raise _BackendSuspension(
                    "recovery_evidence_missing",
                    RecoveryEvidenceMissing(reason_code="investigation_evidence_missing"),
                )
        else:
            try:
                response = validate_investigation_response(raw_result)
            except (TypeError, ValueError) as error:
                raise _BackendSuspension(
                    "provider_result_invalid",
                    RecoveryEvidenceMissing(reason_code="provider_result_invalid"),
                ) from error
            collected_jobs = self._select_collected_jobs(snapshot, response)
            if not collected_jobs:
                raise _BackendSuspension(
                    "recovery_evidence_missing",
                    RecoveryEvidenceMissing(reason_code="investigation_evidence_missing"),
                )
            # 先用稳定的草稿摘要调用 verifier，再把 verifier 的公开摘要写入
            # 最终 pair；verifier 不拥有 Harness 内部身份或事件状态。
            draft = self._build_prepared_records(request, response, collected_jobs, None)
            details = self._get_evidence_details(
                request,
                response,
                draft,
                collected_jobs,
            )
            prepared = self._build_prepared_records(
                request,
                response,
                collected_jobs,
                details,
            )
            verification = self._record_evidence_verifications(
                request,
                details,
                collected_jobs,
            )
            return prepared, verification

        # provider 直接返回已密封 pair 时，仍要求当前作业完成一次证据核验
        # transition；输出正文保持 owner 提供的密封内容不变。
        verification = self._verify_and_record_evidence(
            request,
            response,
            prepared,
            collected_jobs,
        )
        return prepared, verification

    @staticmethod
    def _assert_investigation_pairs(records: Sequence[ScientificRecord]) -> None:
        """在 backend 输出前核对 evidence/result 一一配对。"""

        evidence = {
            record.scientific_record_id
            for record in records
            if record.scientific_record_type == "VerifiedEvidenceRecord"
        }
        results = [
            record
            for record in records
            if record.scientific_record_type == "InvestigationResultRecord"
        ]
        if not evidence or len(evidence) != len(results):
            raise _BackendSuspension(
                "provider_result_invalid",
                RecoveryEvidenceMissing(reason_code="investigation_record_pair_missing"),
            )
        for result in results:
            evidence_ref = result.payload.get("evidence_ref")
            if (
                evidence_ref not in evidence
                or evidence_ref not in result.source_scientific_record_ids
            ):
                raise _BackendSuspension(
                    "provider_result_invalid",
                    RecoveryEvidenceMissing(reason_code="investigation_record_pair_invalid"),
                )
        for record in records:
            _assert_public_output_value(record.payload, "ScientificRecord.payload")

    def _build_prepared_records(
        self,
        request: InvestigationHarnessRequest,
        response: InvestigationResponse,
        collected_jobs: tuple[JobReference, ...],
        details: _EvidenceDetails | None,
    ) -> PreparedModuleInvocationOutput:
        """把 provider response 映射为标准 V2 ScientificRecord pair。"""

        seed = _digest(
            {
                "module_invocation_id": request.module_invocation_id,
                "response": response.model_dump(),
                "jobs": [job.minichem_job_id for job in collected_jobs],
            }
        )
        evidence_id = f"{request.module_invocation_id}:evidence:{seed[:24]}"
        result_id = f"{request.module_invocation_id}:result:{seed[24:48]}"
        if details is None:
            items = tuple(
                {
                    "job_ref": job.minichem_job_id,
                    "operation_ref": job.minichem_operation_id,
                    "status": "collected",
                }
                for job in collected_jobs
            )
            artifact_refs = tuple(response.artifact_refs)
            source_manifest_ref = f"manifest-{seed[:24]}"
            summary = response.narrative
            verification_payload: Mapping[str, Any] = {
                "status": "pending_verification",
                "job_refs": [job.minichem_job_id for job in collected_jobs],
            }
        else:
            items = details.evidence_items
            artifact_refs = details.artifact_refs
            source_manifest_ref = details.source_manifest_ref
            summary = details.summary
            verification_payload = details.verification_payload
        evidence_payload = {
            "evidence_items": [dict(item) for item in items],
            "artifact_refs": list(artifact_refs),
            "source_manifest_ref": source_manifest_ref,
            "summary": summary,
            "verification": dict(verification_payload),
        }
        evidence_record = ScientificRecord.from_payload(
            scientific_record_id=evidence_id,
            research_task_id=request.research_task_id,
            scientific_record_type="VerifiedEvidenceRecord",
            scientific_record_schema_version="2",
            producer_module_invocation_id=request.module_invocation_id,
            source_scientific_record_ids=tuple(request.input_scientific_record_ids),
            payload=evidence_payload,
        )
        result_record = ScientificRecord.from_payload(
            scientific_record_id=result_id,
            research_task_id=request.research_task_id,
            scientific_record_type="InvestigationResultRecord",
            scientific_record_schema_version="2",
            producer_module_invocation_id=request.module_invocation_id,
            # Result 直接列出本 Session 的 Selected Hypothesis/repair 输入，
            # 让后续返修无需从作业历史或列表位置反推原 Session。
            source_scientific_record_ids=(
                evidence_id,
                *request.input_scientific_record_ids,
            ),
            payload={
                "conclusion": response.narrative,
                "self_judgment": response.self_judgment,
                "observations": list(response.observations),
                "limitations": list(response.limitations),
                "evidence_ref": evidence_id,
            },
        )
        return PreparedModuleInvocationOutput.from_records(
            module_invocation_id=request.module_invocation_id,
            module_input_digest=request.module_input_digest,
            output_scientific_records=(evidence_record, result_record),
        )

    def _select_collected_jobs(
        self,
        snapshot: HarnessRecoverySnapshot,
        response: InvestigationResponse | None,
    ) -> tuple[JobReference, ...]:
        """选择账本内已收集 job；其它 scoped 引用交给权威 evidence verifier。"""

        collected = tuple(
            job
            for job in snapshot.jobs
            if job.minichem_job_id in snapshot.collected_job_ids
        )
        if response is None or not response.job_refs:
            return collected
        selected_ids = set(response.job_refs)
        collected_ids = {job.minichem_job_id for job in collected}
        if (
            not selected_ids.issubset(collected_ids)
            and self._evidence_verifier is None
        ):
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="result_job_evidence_missing"),
            )
        # completed provider turn 可能引用此前 interrupted turn 已在同一 9012
        # scope 内成功收集、但未进入 Harness Recovery Store 的作业。生产路径的
        # evidence verifier 会按 scope、thread、终态和来源链逐项核验这些引用；
        # 本地账本不能在权威核验前把它们误判为不存在。这里只把账本已知的交集
        # 用于 Harness VerifyEvidence 事实，最终 EvidenceRecord 由 verifier 摘要
        # 覆盖全部通过核验的引用。没有 verifier 时仍按上方规则 fail-closed。
        return tuple(
            job for job in collected if job.minichem_job_id in selected_ids
        )

    def _verify_and_record_evidence(
        self,
        request: InvestigationHarnessRequest,
        response: InvestigationResponse | None,
        prepared: PreparedModuleInvocationOutput,
    ) -> RecoveryVerification:
        """调用可选 evidence owner，并为每个收集 job 写 VerifyEvidence fact。"""

        identity = _request_identity(request)
        snapshot = self._read_snapshot(identity)
        collected_jobs = tuple(
            job
            for job in snapshot.jobs
            if job.minichem_job_id in snapshot.collected_job_ids
        )
        if not collected_jobs:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="investigation_evidence_missing"),
            )
        details = self._get_evidence_details(request, response, prepared, collected_jobs)
        return self._record_evidence_verifications(request, details, collected_jobs)

    def _record_evidence_verifications(
        self,
        request: InvestigationHarnessRequest,
        details: _EvidenceDetails,
        collected_jobs: tuple[JobReference, ...],
    ) -> RecoveryVerification:
        """把一个证据摘要映射为当前 invocation 的 typed verification facts。"""

        # 一个 pair 可以汇总多个作业；Recovery Store 仍为每个作业登记独立引用。
        for index, job in enumerate(collected_jobs):
            verification_ref = details.verification_ref
            if index:
                verification_ref = f"{verification_ref}-{index}"
            verification_digest = details.verification_digest
            if index:
                verification_digest = _digest(
                    {
                        "digest": verification_digest,
                        "job_id": job.minichem_job_id,
                        "index": index,
                    }
                )
            verification = EvidenceVerificationReference(
                verification_ref=_safe_ref(verification_ref, "verification_ref"),
                verification_digest=verification_digest,
                job_id=job.minichem_job_id,
                operation_id=job.minichem_operation_id,
                submission_key=job.minichem_submission_key,
            )
            self._append_transition(
                _request_identity(request),
                VerifyEvidence(_event_id("evidence-verify", verification.verification_ref), verification),
            )
        return Consistent(reason_code="evidence_verified")

    def _get_evidence_details(
        self,
        request: InvestigationHarnessRequest,
        response: InvestigationResponse | None,
        prepared: PreparedModuleInvocationOutput,
        collected_jobs: tuple[JobReference, ...],
    ) -> _EvidenceDetails:
        """将 evidence owner 返回的摘要规范化；默认实现完全在内存中确定性运行。"""

        raw: Any = None
        if self._evidence_verifier is None and self._strict_evidence:
            raise _BackendSuspension(
                "evidence_verification_missing",
                RecoveryEvidenceMissing(reason_code="evidence_verification_missing"),
            )
        if self._evidence_verifier is not None:
            try:
                verifier = getattr(self._evidence_verifier, "verify_evidence", None)
                if not callable(verifier):
                    verifier = getattr(
                        self._evidence_verifier,
                        "verify_investigation_result_with_evidence",
                        None,
                    )
                if not callable(verifier):
                    verifier = self._evidence_verifier if callable(self._evidence_verifier) else None
                if verifier is None:
                    raise ValueError("evidence verifier 缺少公开 verify 方法")
                raw = self._invoke_evidence_verifier(
                    verifier,
                    request,
                    response,
                    prepared,
                    collected_jobs,
                )
            except EvidenceResultRejected as error:
                raise _ResultRepairRequired(error.to_action_result()) from None
            except _VerifiedConflict:
                raise
            except Exception as error:
                if _is_unavailable_error(error):
                    raise _BackendSuspension(
                        "evidence_unavailable",
                        TemporarilyUnavailable(reason_code="evidence_unavailable"),
                    ) from error
                raise _BackendSuspension(
                    "evidence_verification_missing",
                    RecoveryEvidenceMissing(reason_code="evidence_verification_missing"),
                ) from error
        value: Mapping[str, Any]
        if raw is None:
            value = {}
        else:
            try:
                value = _mapping_from_value(raw, "evidence verification")
            except (TypeError, ValueError) as error:
                raise _BackendSuspension(
                    "evidence_verification_missing",
                    RecoveryEvidenceMissing(reason_code="evidence_verification_missing"),
                ) from error
        seed = _digest(
            {
                "request": _public_request(request),
                "prepared_output": prepared.module_output_digest,
                "jobs": [job.minichem_job_id for job in collected_jobs],
            }
        )
        verification_ref = _read_field(value, "verification_ref", "verificationRef") or f"verification-{seed[:24]}"
        verification_digest = _read_field(value, "verification_digest", "verificationDigest")
        if not isinstance(verification_digest, str) or _SHA256_RE.fullmatch(verification_digest) is None:
            verification_digest = _digest({"verification_ref": verification_ref, "seed": seed})
        summary = _read_field(value, "summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = response.narrative if response is not None else "已完成的 Investigation 证据。"
        source_manifest_ref = _read_field(value, "source_manifest_ref", "sourceManifestRef")
        if not isinstance(source_manifest_ref, str) or not source_manifest_ref.strip():
            source_manifest_ref = f"manifest-{seed[:24]}"
        evidence_items_raw = _read_field(value, "evidence_items", "evidenceItems")
        if not isinstance(evidence_items_raw, Sequence) or isinstance(evidence_items_raw, (str, bytes)):
            evidence_items_raw = [
                {
                    "job_ref": job.minichem_job_id,
                    "operation_ref": job.minichem_operation_id,
                    "status": "collected",
                }
                for job in collected_jobs
            ]
        evidence_items = tuple(
            _copy_json_object(item, "evidence_items[]")
            for item in evidence_items_raw
            if isinstance(item, Mapping)
        )
        if not evidence_items:
            evidence_items = tuple(
                {
                    "job_ref": job.minichem_job_id,
                    "operation_ref": job.minichem_operation_id,
                    "status": "collected",
                }
                for job in collected_jobs
            )
        verification_payload = _copy_json_object(
            _read_field(value, "verification")
            if isinstance(_read_field(value, "verification"), Mapping)
            else {
                "status": "verified",
                "verification_ref": verification_ref,
                "verification_digest": verification_digest,
            },
            "verification",
        )
        _assert_public_output_value(verification_payload, "verification")
        for item in evidence_items:
            _assert_public_output_value(item, "evidence_items[]")
        artifact_refs_raw = _read_field(value, "artifact_refs", "artifactRefs")
        if not isinstance(artifact_refs_raw, Sequence) or isinstance(artifact_refs_raw, (str, bytes)):
            artifact_refs_raw = response.artifact_refs if response is not None else ()
        artifact_refs = tuple(_safe_text(item, "artifact_refs[]") for item in artifact_refs_raw)
        return _EvidenceDetails(
            verification_ref=_safe_ref(verification_ref, "verification_ref"),
            verification_digest=verification_digest,
            source_manifest_ref=_safe_ref(source_manifest_ref, "source_manifest_ref"),
            summary=_safe_text(summary, "evidence summary"),
            evidence_items=evidence_items,
            verification_payload=verification_payload,
            artifact_refs=artifact_refs,
        )

    @staticmethod
    def _invoke_evidence_verifier(
        verifier: Callable[..., Any],
        request: InvestigationHarnessRequest,
        response: InvestigationResponse | None,
        prepared: PreparedModuleInvocationOutput,
        jobs: tuple[JobReference, ...],
    ) -> Any:
        """调用 fake 或低层 evidence owner 的窄适配方法。"""

        try:
            return verifier(
                response,
                request=request,
                jobs=jobs,
                prepared_output=prepared,
            )
        except TypeError:
            signature = inspect.signature(verifier)
            kwargs = {
                name: value
                for name, value in {
                    "request": request,
                    "jobs": jobs,
                    "prepared_output": prepared,
                }.items()
                if name in signature.parameters
            }
            positional: tuple[Any, ...] = ()
            if not signature.parameters or next(iter(signature.parameters.values())).name not in kwargs:
                positional = (response,)
            return verifier(*positional, **kwargs)

    def _append_transition(self, identity: HarnessRunIdentity, transition: Any) -> HarnessRecoverySnapshot:
        """追加 typed recovery fact，并把 event/digest 冲突升级为已核实冲突。"""

        try:
            return self._recovery_store.append_transition(identity, transition)
        except HarnessRecoveryIdentityConflict as error:
            raise _VerifiedConflict() from error
        except HarnessRecoveryTransitionConflict as error:
            raise _VerifiedConflict() from error
        except HarnessRecoveryStateConflict as error:
            raise _BackendSuspension(
                "recovery_state_unavailable",
                RecoveryEvidenceMissing(reason_code="recovery_state_unavailable"),
            ) from error

    def _read_snapshot(self, identity: HarnessRunIdentity) -> HarnessRecoverySnapshot:
        """读取同一 store 的最新 projection。"""

        try:
            return self._recovery_store.read_snapshot(identity)
        except HarnessRecoveryIdentityConflict as error:
            raise _VerifiedConflict() from error
        except HarnessRecoveryContractError:
            raise
        except Exception as error:
            raise _BackendSuspension(
                "recovery_evidence_missing",
                RecoveryEvidenceMissing(reason_code="harness_recovery_unreadable"),
            ) from error

    def _record_suspension(
        self,
        identity: HarnessRunIdentity,
        suspension: _BackendSuspension,
    ) -> BackendSuspended:
        """保存与 facade 相同 event id 的 SuspendInvocation，支持精确重放。"""

        try:
            self._recovery_store.append_transition(
                identity,
                SuspendInvocation(
                    _event_id("suspend", suspension.reason_code),
                    suspension.reason_code,
                ),
            )
        except HarnessRecoveryIdentityConflict as error:
            del error
            return self._fatal()
        except HarnessRecoveryTransitionConflict:
            return self._fatal()
        except HarnessRecoveryStateConflict:
            # Suspend 本身允许在 Open/Suspended 写入；其它状态问题必须等待恢复资料。
            return BackendSuspended(
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_recovery_unreadable"
                ),
            )
        return BackendSuspended(
            suspension.reason_code,
            recovery_verification=suspension.verification,
        )

    @staticmethod
    def _fatal() -> BackendFatalFailure:
        """构造唯一允许的 backend fatal 分支。"""

        return BackendFatalFailure(
            "verified_integrity_conflict",
            recovery_verification=VerifiedIntegrityConflict(
                reason_code="verified_integrity_conflict"
            ),
        )

def create_v2_native_investigation_backend(
    *,
    recovery_store: HarnessRecoveryStore,
    provider_transport: NativeProviderTransport | Any | None = None,
    execution_transport: NativeExecutionTransport | Any | None = None,
    evidence_verifier: Any | None = None,
    **kwargs: Any,
) -> InvestigationSessionRunner:
    """创建 V2-native backend；供组装层一行绑定到 ``InvestigationHarness``。"""

    return InvestigationSessionRunner(
        recovery_store=recovery_store,
        provider_transport=provider_transport,
        execution_transport=execution_transport,
        evidence_verifier=evidence_verifier,
        **kwargs,
    )
