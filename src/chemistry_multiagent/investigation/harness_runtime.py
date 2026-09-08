"""Investigation Harness 的模块边界与持久恢复合同。

本模块把 Investigation and Experiment 作为一个深模块。Controller 只提供任务、
模块、调用身份和固定输入摘要；Codex thread、turn、Goal、scope、MiniChem job、
receipt、sidecar 及路径都由传入的 backend 在模块内部处理。

``InvestigationHarness`` 的 runtime store 独立于 Working Memory。backend 先返回
一个公共结果，facade 将 ``PreparedModuleInvocationOutput`` 原子写入同一 runtime，再向
调用方返回 ``Completed``。因此进程在 facade 返回前退出时，下一进程仍可以从同一
``harness_invocation_id`` 重放固定输出，而无需重新调用模型或 MiniChem。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast

from chemistry_multiagent.memory.contracts import canonical_json
from chemistry_multiagent.memory.v2_ledger import PreparedModuleInvocationOutput
from .recovery_contract import (
    Consistent,
    RecoveryEvidenceMissing,
    RecoveryStatus,
    RecoveryVerification,
    Reconciled,
    TemporarilyUnavailable,
    VerifiedIntegrityConflict,
)
from .harness_recovery_store import (
    CommitModuleOutput,
    CompleteInvocation,
    FailInvocation,
    HarnessLifecycle,
    HarnessRecoveryContractError as RecoveryStoreContractError,
    HarnessRecoveryIdentityConflict as RecoveryStoreIdentityConflict,
    HarnessRecoverySnapshot,
    HarnessRecoveryStateConflict as RecoveryStoreStateConflict,
    HarnessRecoveryStore,
    HarnessRecoveryStoreError,
    HarnessRecoveryTransitionConflict,
    HarnessRunIdentity,
    ModuleOutputReference,
    RecordManifestRead,
    SuspendInvocation,
)


class InvestigationHarnessError(RuntimeError):
    """Harness runtime 拒绝请求或无法安全恢复时的基础异常。"""


class InvestigationHarnessContractError(InvestigationHarnessError, ValueError):
    """Harness 请求、backend 结果或持久合同不合法。"""


class InvestigationHarnessIdentityConflict(InvestigationHarnessContractError):
    """同一 Harness invocation 被请求绑定到另一份任务或输入。"""


class InvestigationHarnessPersistenceError(InvestigationHarnessError):
    """恢复持久化错误基类；调用方应按具体读写子类决定边界行为。"""


class InvestigationHarnessRecoveryEvidenceUnavailable(
    InvestigationHarnessPersistenceError
):
    """恢复所需的已保存资料不存在、不可读或内容损坏。"""


class InvestigationHarnessRecoveryStateWriteError(
    InvestigationHarnessPersistenceError
):
    """恢复状态或固定输出无法可靠写入。"""


class InvestigationHarnessRunNotFound(InvestigationHarnessRecoveryEvidenceUnavailable):
    """resume 引用了尚未持久化的 Harness run。"""


_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: Any, field_name: str) -> str:
    """校验公开身份字段，避免空值进入持久 runtime。"""

    if not isinstance(value, str) or not value.strip():
        raise InvestigationHarnessContractError(f"{field_name} 必须是非空字符串")
    value = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise InvestigationHarnessContractError(f"{field_name} 不能包含控制字符")
    return value


def _require_opaque_ref(value: Any, field_name: str) -> str:
    """校验不透明引用不被当作文件路径使用。"""

    value = _require_text(value, field_name)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise InvestigationHarnessContractError(
            f"{field_name} 必须是不含路径分隔符的不透明引用"
        )
    return value


def _require_code(value: Any, field_name: str) -> str:
    """校验公开原因/失败码不携带路径或自由文本。"""

    value = _require_text(value, field_name)
    if not _SAFE_CODE_RE.fullmatch(value):
        raise InvestigationHarnessContractError(
            f"{field_name} 必须是安全的结构化代码"
        )
    return value


def _recovery_identity(request: "InvestigationHarnessRequest") -> HarnessRunIdentity:
    """把公开请求映射为 HarnessRecoveryStore 的固定 invocation 身份。"""

    input_digest = request.module_input_digest
    if not _SHA256_RE.fullmatch(input_digest):
        input_digest = hashlib.sha256(
            canonical_json(
                {
                    "declared_input_digest": input_digest,
                    "module_kind": request.module_kind,
                    "input_scientific_record_ids": list(
                        request.input_scientific_record_ids
                    ),
                }
            ).encode("utf-8")
        ).hexdigest()
    return HarnessRunIdentity(
        research_task_id=request.research_task_id,
        module_invocation_id=request.module_invocation_id,
        harness_invocation_id=request.harness_invocation_id,
        module_input_digest=input_digest,
    )


def _require_refs(value: Any, field_name: str) -> tuple[str, ...]:
    """校验模块输入记录引用的顺序和唯一性。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvestigationHarnessContractError(f"{field_name} 必须是字符串序列")
    refs = tuple(_require_text(item, f"{field_name}[]") for item in value)
    if len(set(refs)) != len(refs):
        raise InvestigationHarnessContractError(f"{field_name} 不能包含重复引用")
    return refs


@dataclass(frozen=True, slots=True)
class InvestigationHarnessRequest:
    """Controller 传给 Investigation Harness 的最小模块请求。

    这里不接收 module run 的数据库行，也不接收任何 Codex、MiniChem 或文件系统
    身份。``module_input_digest`` 是 Controller 已核对的科学输入摘要，Harness 只把它
    绑定到自己的内部运行上下文。
    """

    research_task_id: str
    module_invocation_id: str
    module_kind: str
    input_scientific_record_ids: tuple[str, ...]
    module_input_digest: str
    context_binding: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """规范化并冻结模块边界字段。"""

        object.__setattr__(
            self,
            "research_task_id",
            _require_text(self.research_task_id, "research_task_id"),
        )
        object.__setattr__(
            self,
            "module_invocation_id",
            _require_opaque_ref(self.module_invocation_id, "module_invocation_id"),
        )
        module_kind = _require_text(self.module_kind, "module_kind")
        if module_kind != "hypothesis-investigation":
            raise InvestigationHarnessContractError(
                "module_kind 必须是 hypothesis-investigation"
            )
        object.__setattr__(self, "module_kind", "hypothesis-investigation")
        object.__setattr__(
            self,
            "input_scientific_record_ids",
            _require_refs(
                self.input_scientific_record_ids,
                "input_scientific_record_ids",
            ),
        )
        object.__setattr__(
            self,
            "module_input_digest",
            _require_text(self.module_input_digest, "module_input_digest"),
        )
        if self.context_binding is not None:
            if not isinstance(self.context_binding, Mapping):
                raise InvestigationHarnessContractError("context_binding 必须是 object")
            try:
                binding = json.loads(canonical_json(dict(self.context_binding)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise InvestigationHarnessContractError(
                    "context_binding 必须是严格 JSON object"
                ) from exc
            if not isinstance(binding, dict):  # pragma: no cover - object 已校验
                raise InvestigationHarnessContractError("context_binding 必须是 object")
            object.__setattr__(self, "context_binding", MappingProxyType(binding))

    @property
    def harness_invocation_id(self) -> str:
        """由正式模块调用与 Context binding 稳定派生的 Harness 身份。"""

        binding = self.context_binding or {}
        seed = {
            "research_task_id": self.research_task_id,
            "module_invocation_id": self.module_invocation_id,
            "context_binding": dict(binding),
        }
        return "harness-" + hashlib.sha256(
            canonical_json(seed).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InvestigationHarnessRequest":
        """从严格 mapping 恢复请求；拒绝混入内部运行字段。"""

        if not isinstance(value, Mapping):
            raise InvestigationHarnessContractError("Harness request 必须是 JSON object")
        expected = {
            "research_task_id",
            "module_invocation_id",
            "module_kind",
            "input_scientific_record_ids",
            "module_input_digest",
            "context_binding",
        }
        actual = set(value)
        if actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            details: list[str] = []
            if missing:
                details.append(f"缺少 {', '.join(missing)}")
            if extra:
                details.append(f"含额外字段 {', '.join(extra)}")
            raise InvestigationHarnessContractError(
                f"Harness request 字段合同不匹配：{'；'.join(details)}"
            )
        return cls(
            research_task_id=value["research_task_id"],
            module_invocation_id=value["module_invocation_id"],
            module_kind=value["module_kind"],
            input_scientific_record_ids=tuple(value["input_scientific_record_ids"]),
            module_input_digest=value["module_input_digest"],
            context_binding=value["context_binding"],
        )

    def to_dict(self) -> dict[str, Any]:
        """返回只包含 Controller/Harness 模块边界的稳定结构。"""

        return {
            "research_task_id": self.research_task_id,
            "module_invocation_id": self.module_invocation_id,
            "module_kind": self.module_kind,
            "input_scientific_record_ids": list(self.input_scientific_record_ids),
            "module_input_digest": self.module_input_digest,
            "context_binding": dict(self.context_binding or {}),
        }


class SchemaV1LegacyInvestigationHarnessRequestReader:
    """schema-v1 Harness request 的显式只读解析器。"""

    @staticmethod
    def read(value: Mapping[str, Any]) -> InvestigationHarnessRequest:
        """将旧 request 一次性转换为正式 request，不把旧字段带入新对象。"""

        if not isinstance(value, Mapping):
            raise InvestigationHarnessContractError("legacy Harness request 必须是 object")
        expected = {
            "task_id",
            "module_run_id",
            "module_kind",
            "runtime_ref",
            "input_record_refs",
            "input_digest",
        }
        if set(value) != expected:
            raise InvestigationHarnessContractError(
                "schema-v1 Harness request 字段合同不匹配"
            )
        legacy_kind = _require_text(value["module_kind"], "module_kind")
        normalized_kind = legacy_kind.casefold().replace("_", "-").replace(" ", "-")
        if normalized_kind not in {
            "hypothesis-investigation",
            "investigation",
            "investigation-and-experiment",
            "investigation-experiment",
        }:
            raise InvestigationHarnessContractError(
                "schema-v1 Harness request 的 module_kind 不受支持"
            )
        return InvestigationHarnessRequest(
            research_task_id=value["task_id"],
            module_invocation_id=value["module_run_id"],
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=tuple(value["input_record_refs"]),
            module_input_digest=value["input_digest"],
            context_binding={"kind": "LegacyRuntime", "identity": value["runtime_ref"]},
        )


class SchemaV1LegacyPreparedModuleOutputReader:
    """schema-v1 旧 PreparedModuleOutput envelope 的显式只读解析器。"""

    _EXPECTED_KEYS = {
        "module_run_id",
        "input_digest",
        "output_records",
        "output_digest",
    }

    @classmethod
    def read(cls, value: Mapping[str, Any]) -> PreparedModuleInvocationOutput:
        """把旧输出包归一为 canonical 对象，不允许旧字段写回。"""

        if not isinstance(value, Mapping) or set(value) != cls._EXPECTED_KEYS:
            raise InvestigationHarnessContractError(
                "schema-v1 PreparedModuleOutput 字段合同不匹配"
            )
        try:
            return PreparedModuleInvocationOutput.from_schema_v10_mapping(value)
        except (TypeError, ValueError) as exc:
            raise InvestigationHarnessContractError(
                "schema-v1 PreparedModuleOutput 内容损坏"
            ) from exc


class SchemaV1LegacyHarnessManifestReader:
    """schema-v1 Harness manifest 的显式只读解析器。"""

    _EXPECTED_KEYS = {
        "schema_version",
        "harness_run_ref",
        "request",
        "state",
        "failure_code",
    }

    @classmethod
    def read(
        cls,
        value: Mapping[str, Any],
        *,
        requested_harness_invocation_id: str,
    ) -> dict[str, Any]:
        """把旧 manifest 的 request 转换为 canonical request。"""

        if not isinstance(value, Mapping) or set(value) != cls._EXPECTED_KEYS:
            raise InvestigationHarnessContractError(
                "schema-v1 Harness manifest 字段合同不匹配"
            )
        if value.get("schema_version") != 1:
            raise InvestigationHarnessContractError(
                "schema-v1 Harness manifest 版本不匹配"
            )
        legacy_ref = _require_opaque_ref(value.get("harness_run_ref"), "harness_run_ref")
        if legacy_ref != requested_harness_invocation_id:
            raise InvestigationHarnessIdentityConflict(
                "schema-v1 Harness manifest 的旧 run ref 发生漂移"
            )
        request = SchemaV1LegacyInvestigationHarnessRequestReader.read(value["request"])
        return {
            "schema_version": 1,
            "harness_invocation_id": requested_harness_invocation_id,
            "request": request,
            "state": value["state"],
            "failure_code": value["failure_code"],
        }


@dataclass(frozen=True)
class BackendCompleted:
    """backend 产生的待提交输出；尚未代表已向 Controller 成功返回。"""

    prepared_output: PreparedModuleInvocationOutput
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: Consistent(reason_code="backend_output_ready")
    )

    def __post_init__(self) -> None:
        """成功结果必须明确证明身份一致或已协调。"""

        if not isinstance(self.prepared_output, PreparedModuleInvocationOutput):
            raise InvestigationHarnessContractError(
                "BackendCompleted.prepared_output 必须是 PreparedModuleInvocationOutput"
            )
        if not isinstance(self.recovery_verification, RecoveryVerification) or not (
            self.recovery_verification.can_continue
        ):
            raise InvestigationHarnessContractError(
                "BackendCompleted 必须携带 Consistent 或 Reconciled"
            )


@dataclass(frozen=True)
class BackendSuspended:
    """backend 报告暂时不能形成正式 Result。"""

    reason_code: str
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: TemporarilyUnavailable(
            reason_code="external_owner_unavailable"
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _require_code(self.reason_code, "reason_code"))
        if not isinstance(self.recovery_verification, RecoveryVerification):
            raise InvestigationHarnessContractError(
                "BackendSuspended.recovery_verification 必须是 RecoveryVerification"
            )
        if self.recovery_verification.status not in {
            RecoveryStatus.TEMPORARILY_UNAVAILABLE,
            RecoveryStatus.RECOVERY_EVIDENCE_MISSING,
        }:
            raise InvestigationHarnessContractError(
                "BackendSuspended 必须使用 TemporarilyUnavailable 或 RecoveryEvidenceMissing"
            )


@dataclass(frozen=True)
class BackendFatalFailure:
    """backend 报告整个模块已确认不可恢复；只保留脱敏错误码。

    单个 MiniChem/Gaussian/programmable_analysis 作业的失败属于 backend
    内部 ActionResult/receipt，必须先回传模型继续 loop，不能用这个类型把
    ModuleFailed 或 TaskFailed 提前暴露给 Harness。整个科学模块的失败或不可
    回答也必须通过正常 Scientific Record 交给 Review；这个类型只表示已经核实
    的身份完整性冲突。
    """

    failure_code: str
    recovery_verification: RecoveryVerification | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failure_code",
            _require_code(self.failure_code, "failure_code"),
        )
        if self.failure_code != "verified_integrity_conflict":
            raise InvestigationHarnessContractError(
                "BackendFatalFailure 只允许表示已核实的 VerifiedIntegrityConflict"
            )
        if self.recovery_verification is not None and not isinstance(
            self.recovery_verification, RecoveryVerification
        ):
            raise InvestigationHarnessContractError(
                "BackendFatalFailure.recovery_verification 必须是 RecoveryVerification"
            )
        if self.recovery_verification is not None and not self.recovery_verification.must_fail_integrity:
            raise InvestigationHarnessContractError(
                "BackendFatalFailure 只能携带 VerifiedIntegrityConflict"
            )
        if (
            self.recovery_verification is None
            and self.failure_code == "verified_integrity_conflict"
        ):
            object.__setattr__(
                self,
                "recovery_verification",
                VerifiedIntegrityConflict(reason_code=self.failure_code),
            )


BackendResult: TypeAlias = (
    PreparedModuleInvocationOutput
    | BackendCompleted
    | BackendSuspended
    | BackendFatalFailure
    | RecoveryVerification
)


class InvestigationHarnessBackend(Protocol):
    """Investigation 内部 adapter；实现者不得把私有身份放入返回结果。

    MiniChem/Gaussian/programmable_analysis 的单作业失败必须在 adapter 内
    作为 ActionResult/receipt 交回模型 loop；只有整个模块确定无法继续时才
    返回 ``BackendSuspended``，而 ``BackendFatalFailure`` 仅用于已核实身份冲突。
    """

    def start(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """创建内部调查上下文并执行一次模块动作。"""

    def resume(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """在同一内部调查上下文中恢复一次模块动作。"""


@dataclass(frozen=True)
class Completed:
    """Harness 已提交 PreparedModuleInvocationOutput 的成功结果。"""

    harness_invocation_id: str
    prepared_output: PreparedModuleInvocationOutput
    status: str = field(init=False, default="Completed")
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: Consistent(reason_code="output_replayed")
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "harness_invocation_id",
            _require_opaque_ref(self.harness_invocation_id, "harness_invocation_id"),
        )
        if not isinstance(self.prepared_output, PreparedModuleInvocationOutput):
            raise InvestigationHarnessContractError(
                "Completed.prepared_output 必须是 PreparedModuleInvocationOutput"
            )
        if self.recovery_verification.status not in {
            RecoveryStatus.CONSISTENT,
            RecoveryStatus.RECONCILED,
        }:
            raise InvestigationHarnessContractError(
                "Completed 必须使用 Consistent 或 Reconciled"
            )

    def to_dict(self) -> dict[str, Any]:
        """返回安全的模块结果，不暴露内部运行身份。"""

        return {
            "status": self.status,
            "harness_invocation_id": self.harness_invocation_id,
            "prepared_output": self.prepared_output.to_dict(),
        }


@dataclass(frozen=True)
class Suspended:
    """Harness 暂停但仍可恢复的模块结果。"""

    harness_invocation_id: str
    reason_code: str
    status: str = field(init=False, default="Suspended")
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: TemporarilyUnavailable(
            reason_code="external_owner_unavailable"
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "harness_invocation_id",
            _require_opaque_ref(self.harness_invocation_id, "harness_invocation_id"),
        )
        object.__setattr__(self, "reason_code", _require_code(self.reason_code, "reason_code"))
        if self.recovery_verification.status not in {
            RecoveryStatus.TEMPORARILY_UNAVAILABLE,
            RecoveryStatus.RECOVERY_EVIDENCE_MISSING,
        }:
            raise InvestigationHarnessContractError(
                "Suspended 必须使用 TemporarilyUnavailable 或 RecoveryEvidenceMissing"
            )

    def to_dict(self) -> dict[str, str]:
        """返回暂停原因和恢复引用，不伪造 Working Memory 状态。"""

        return {
            "status": self.status,
            "harness_invocation_id": self.harness_invocation_id,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class FatalFailure:
    """Harness 已核实身份完整性冲突的模块结果。"""

    harness_invocation_id: str
    failure_code: str
    status: str = field(init=False, default="FatalFailure")
    recovery_verification: RecoveryVerification | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "harness_invocation_id",
            _require_opaque_ref(self.harness_invocation_id, "harness_invocation_id"),
        )
        object.__setattr__(
            self,
            "failure_code",
            _require_code(self.failure_code, "failure_code"),
        )
        if self.failure_code != "verified_integrity_conflict":
            raise InvestigationHarnessContractError(
                "FatalFailure 只允许表示已核实的 VerifiedIntegrityConflict"
            )
        if not isinstance(self.recovery_verification, RecoveryVerification):
            raise InvestigationHarnessContractError(
                "FatalFailure.recovery_verification 必须是 RecoveryVerification"
            )
        if not self.recovery_verification.must_fail_integrity:
            raise InvestigationHarnessContractError(
                "FatalFailure 只能携带 VerifiedIntegrityConflict"
            )

    def to_dict(self) -> dict[str, str]:
        """返回脱敏失败码，不返回 backend 异常正文或路径。"""

        return {
            "status": self.status,
            "harness_invocation_id": self.harness_invocation_id,
            "failure_code": self.failure_code,
        }


HarnessOutcome: TypeAlias = Completed | Suspended | FatalFailure


def derive_harness_invocation_id(seed: str) -> str:
    """从稳定身份种子派生跨进程可定位的 Harness invocation ID。"""

    seed = _require_opaque_ref(seed, "harness_identity_seed")
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"harness-{digest}"


class _PersistentHarnessRuntime:
    """独立保存 Harness 身份、状态和 Prepared Output 的内部 store。"""

    _SCHEMA_VERSION = 2
    _LEGACY_SCHEMA_VERSION = 1
    _MANIFEST_KEYS = {
        "schema_version",
        "harness_invocation_id",
        "request",
        "state",
        "failure_code",
    }
    _LEGACY_MANIFEST_KEYS = {
        "schema_version",
        "harness_run_ref",
        "request",
        "state",
        "failure_code",
    }
    _STATES = {"open", "suspended", "completed", "failed"}

    def __init__(self, runtime_root: str | os.PathLike[str]) -> None:
        """初始化独立 runtime 根，不读写 Working Memory。"""

        if not isinstance(runtime_root, (str, os.PathLike)):
            raise InvestigationHarnessContractError("runtime_root 必须是路径")
        try:
            root = Path(os.path.abspath(os.path.expanduser(os.fspath(runtime_root))))
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise InvestigationHarnessRecoveryStateWriteError(
                    "runtime_root 不是目录"
                )
            self.root = root
            self.runs_root = root / "runs"
            self.runs_root.mkdir(parents=True, exist_ok=True)
        except InvestigationHarnessError:
            raise
        except OSError as exc:
            raise InvestigationHarnessRecoveryStateWriteError(
                "无法初始化 Harness runtime"
            ) from exc

    @staticmethod
    def _directory_key(harness_invocation_id: str) -> str:
        """使用摘要定位目录，避免把 opaque 引用当作路径片段。"""

        return hashlib.sha256(harness_invocation_id.encode("utf-8")).hexdigest()

    def _run_directory(self, harness_invocation_id: str) -> Path:
        """返回一个 Harness invocation 的内部目录。"""

        return self.runs_root / self._directory_key(harness_invocation_id)

    def _manifest_path(self, harness_invocation_id: str) -> Path:
        return self._run_directory(harness_invocation_id) / "manifest.json"

    def _prepared_path(self, harness_invocation_id: str) -> Path:
        return self._run_directory(harness_invocation_id) / "prepared_output.json"

    @staticmethod
    def _invocation_directory_name(module_invocation_id: str) -> str:
        """为同一 Harness owner 下的不同 Module Invocation 生成隔离目录名。"""

        return hashlib.sha256(module_invocation_id.encode("utf-8")).hexdigest()

    def _invocation_directory(
        self,
        harness_invocation_id: str,
        module_invocation_id: str | None,
    ) -> Path:
        """返回某次 Module Invocation 的目录。

        早期 schema-v1 运行继续使用根目录布局；同一 Harness owner 的后续
        invocation 使用 ``invocations/<module_invocation_digest>``，从而允许 repair
        复用 Harness runtime 而不把不同 Module Invocation 混成一次调用。
        """

        root = self._run_directory(harness_invocation_id)
        if module_invocation_id is None:
            return root
        root_manifest = root / "manifest.json"
        if root_manifest.exists():
            try:
                with root_manifest.open("r", encoding="utf-8") as stream:
                    value = json.load(stream)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                # manifest 既是旧任务的一次性协调输入，也是 invocation 路径的
                # 旧索引。它损坏时不能把该 runtime 静默当成新的 invocation。
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "已有 Harness manifest 无法解析"
                ) from exc
            if not isinstance(value, Mapping) or not isinstance(
                value.get("request"), Mapping
            ):
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "已有 Harness manifest 缺少 request"
                )
            request_value = value["request"]
            if value.get("schema_version") == self._LEGACY_SCHEMA_VERSION:
                try:
                    stored_module_invocation_id = (
                        SchemaV1LegacyInvestigationHarnessRequestReader.read(
                            request_value
                        ).module_invocation_id
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise InvestigationHarnessRecoveryEvidenceUnavailable(
                        "schema-v1 Harness manifest 的 request 已损坏"
                    ) from exc
            elif value.get("schema_version") == self._SCHEMA_VERSION:
                try:
                    stored_module_invocation_id = (
                        InvestigationHarnessRequest.from_mapping(
                            request_value
                        ).module_invocation_id
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise InvestigationHarnessRecoveryEvidenceUnavailable(
                        "Harness manifest 的 request 已损坏"
                    ) from exc
            else:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "Harness manifest 版本不受支持"
                )
            if stored_module_invocation_id == module_invocation_id:
                return root
        return root / "invocations" / self._invocation_directory_name(
            module_invocation_id
        )

    def _manifest_path_for_invocation(
        self,
        harness_invocation_id: str,
        module_invocation_id: str | None,
    ) -> Path:
        """返回指定 invocation 的 manifest 路径。"""

        return (
            self._invocation_directory(harness_invocation_id, module_invocation_id)
            / "manifest.json"
        )

    def _prepared_path_for_invocation(
        self,
        harness_invocation_id: str,
        module_invocation_id: str | None,
    ) -> Path:
        """返回指定 invocation 的持久输出路径。"""

        return (
            self._invocation_directory(harness_invocation_id, module_invocation_id)
            / "prepared_output.json"
        )

    def load_manifest(
        self,
        harness_invocation_id: str,
        module_invocation_id: str | None = None,
    ) -> dict[str, Any] | None:
        """读取并严格校验一个 Harness manifest。"""

        harness_invocation_id = _require_opaque_ref(
            harness_invocation_id, "harness_invocation_id"
        )
        path = self._manifest_path_for_invocation(
            harness_invocation_id, module_invocation_id
        )
        if module_invocation_id is None and not path.exists():
            invocation_root = self._run_directory(harness_invocation_id) / "invocations"
            candidates = sorted(invocation_root.glob("*/manifest.json"))
            if len(candidates) == 1:
                path = candidates[0]
            elif len(candidates) > 1:
                raise InvestigationHarnessContractError(
                    "同一 Harness runtime 存在多个 invocation，resume 必须提供 module_invocation_id"
                )
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "Harness manifest 无法读取"
            ) from exc
        if not isinstance(value, Mapping):
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "Harness manifest 字段合同损坏"
            )
        is_legacy = set(value) == self._LEGACY_MANIFEST_KEYS and (
            value.get("schema_version") == self._LEGACY_SCHEMA_VERSION
        )
        if is_legacy:
            try:
                legacy_manifest = SchemaV1LegacyHarnessManifestReader.read(
                    value,
                    requested_harness_invocation_id=harness_invocation_id,
                )
                request = cast(
                    InvestigationHarnessRequest,
                    legacy_manifest["request"],
                )
            except InvestigationHarnessIdentityConflict:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "schema-v1 Harness manifest 的 request 已损坏"
                ) from exc
        else:
            if set(value) != self._MANIFEST_KEYS:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "Harness manifest 字段合同损坏"
                )
            if value["schema_version"] != self._SCHEMA_VERSION:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "Harness manifest 版本不受支持"
                )
            if value["harness_invocation_id"] != harness_invocation_id:
                raise InvestigationHarnessIdentityConflict(
                    "Harness manifest 的 harness invocation identity 发生漂移"
                )
            try:
                request = InvestigationHarnessRequest.from_mapping(value["request"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "Harness manifest 的 request 已损坏"
                ) from exc
        state = value["state"]
        if not isinstance(state, str) or state not in self._STATES:
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "Harness manifest 状态损坏"
            )
        failure_code = value["failure_code"]
        if state == "failed":
            if not isinstance(failure_code, str) or not failure_code.strip():
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "失败 Harness manifest 缺少 failure_code"
                )
            try:
                failure_code = _require_code(failure_code, "failure_code")
            except InvestigationHarnessContractError as exc:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "失败 Harness manifest 的 failure_code 已损坏"
                ) from exc
        elif failure_code is not None:
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "非失败 Harness manifest 不得携带 failure_code"
            )
        return {
            "schema_version": self._SCHEMA_VERSION,
            "harness_invocation_id": harness_invocation_id,
            "request": request,
            "state": state,
            "failure_code": None if failure_code is None else str(failure_code).strip(),
        }

    def create_manifest(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> dict[str, Any]:
        """以排他创建保存身份；并发重复创建只读取已存在 manifest。"""

        harness_invocation_id = _require_opaque_ref(
            harness_invocation_id, "harness_invocation_id"
        )
        manifest_path = self._manifest_path_for_invocation(
            harness_invocation_id,
            request.module_invocation_id,
        )
        directory = manifest_path.parent
        try:
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": self._SCHEMA_VERSION,
                "harness_invocation_id": harness_invocation_id,
                "request": request.to_dict(),
                "state": "open",
                "failure_code": None,
            }
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(
                manifest_path,
                flags,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    descriptor = -1
                    stream.write(canonical_json(payload))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            self._sync_directory(directory)
        except FileExistsError:
            existing = self.load_manifest(
                harness_invocation_id, request.module_invocation_id
            )
            if existing is None:  # pragma: no cover - 文件竞争只会留下完整 manifest
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "Harness manifest 并发创建后无法读取"
                )
            return existing
        except (OSError, TypeError, ValueError) as exc:
            raise InvestigationHarnessRecoveryStateWriteError(
                "无法创建 Harness manifest"
            ) from exc
        return cast(
            dict[str, Any],
            self.load_manifest(harness_invocation_id, request.module_invocation_id),
        )

    def update_state(
        self,
        harness_invocation_id: str,
        *,
        state: str,
        failure_code: str | None = None,
        module_invocation_id: str | None = None,
    ) -> dict[str, Any]:
        """原子更新内部状态；不把状态投影到 Working Memory。"""

        if state not in self._STATES:
            raise InvestigationHarnessContractError("未知 Harness runtime 状态")
        manifest = self.load_manifest(harness_invocation_id, module_invocation_id)
        if manifest is None:
            raise InvestigationHarnessRunNotFound("Harness run 不存在")
        if state == "failed":
            failure_code = _require_code(failure_code, "failure_code")
        elif failure_code is not None:
            raise InvestigationHarnessContractError(
                "非失败 Harness 状态不得携带 failure_code"
            )
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "harness_invocation_id": harness_invocation_id,
            "request": manifest["request"].to_dict(),
            "state": state,
            "failure_code": failure_code,
        }
        self._atomic_write_json(
            self._manifest_path_for_invocation(
                harness_invocation_id, module_invocation_id
            ),
            payload,
        )
        return cast(
            dict[str, Any],
            self.load_manifest(harness_invocation_id, module_invocation_id),
        )

    def load_prepared_output(
        self,
        harness_invocation_id: str,
        module_invocation_id: str | None = None,
    ) -> PreparedModuleInvocationOutput | None:
        """读取已经持久化的固定输出包。"""

        try:
            path = self._prepared_path_for_invocation(
                harness_invocation_id,
                module_invocation_id,
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            # 独立 recovery store 已经证明 invocation 身份时，损坏的旧展示索引
            # 不应阻断对已写输出正文的重放。固定的 invocation 目录只由 opaque
            # module_invocation_id 的摘要决定，不依赖 manifest 内容。
            if module_invocation_id is None:
                raise
            path = (
                self._run_directory(harness_invocation_id)
                / "invocations"
                / self._invocation_directory_name(module_invocation_id)
                / "prepared_output.json"
            )
        candidate_paths = [path]
        if module_invocation_id is not None:
            # 首次旧 runtime 可能把正文放在根目录；只有 canonical invocation
            # 路径不存在时才尝试该兼容位置，正文自身仍会做 module/input 核对。
            legacy_path = self._prepared_path(harness_invocation_id)
            if legacy_path != path:
                candidate_paths.append(legacy_path)
        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as stream:
                    value = json.load(stream)
                if not isinstance(value, Mapping):
                    raise InvestigationHarnessContractError(
                        "PreparedModuleInvocationOutput 必须是 JSON object"
                    )
                canonical_keys = {
                    "module_invocation_id",
                    "module_input_digest",
                    "output_scientific_records",
                    "module_output_digest",
                }
                if set(value) == canonical_keys:
                    return PreparedModuleInvocationOutput.from_mapping(value)
                return SchemaV1LegacyPreparedModuleOutputReader.read(value)
            except InvestigationHarnessContractError as exc:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "PreparedModuleInvocationOutput 内容损坏"
                ) from exc
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "PreparedModuleInvocationOutput 无法读取"
                ) from exc
        return None

    def write_prepared_output(
        self,
        harness_invocation_id: str,
        output: PreparedModuleInvocationOutput,
        module_invocation_id: str | None = None,
    ) -> PreparedModuleInvocationOutput:
        """原子写入输出；已有包只能精确重放，不能覆盖。"""

        if not isinstance(output, PreparedModuleInvocationOutput):
            raise InvestigationHarnessContractError(
                "output 必须是 PreparedModuleInvocationOutput"
            )
        if module_invocation_id is None:
            module_invocation_id = output.to_dict()["module_invocation_id"]
        existing = self.load_prepared_output(
            harness_invocation_id,
            module_invocation_id,
        )
        if existing is not None:
            if existing.to_dict() != output.to_dict():
                raise InvestigationHarnessIdentityConflict(
                    "已保存 PreparedModuleInvocationOutput 发生漂移"
                )
            return existing
        try:
            path = self._prepared_path_for_invocation(
                harness_invocation_id,
                module_invocation_id,
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            if module_invocation_id is None:
                raise
            path = (
                self._run_directory(harness_invocation_id)
                / "invocations"
                / self._invocation_directory_name(module_invocation_id)
                / "prepared_output.json"
            )
        # 当前输出包只写 canonical 字段；旧 output_digest 仅由显式 schema-v1
        # reader 读取，不再混入新持久化结构。
        self._atomic_write_json(path, output.to_dict())
        stored = self.load_prepared_output(
            harness_invocation_id,
            module_invocation_id,
        )
        if stored is None:  # pragma: no cover - 原子写入后立即可读
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "PreparedModuleInvocationOutput 写入后无法读取"
            )
        return stored

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        """同步目录项；目录无法刷盘时报告恢复状态写入失败。"""

        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        """在目标目录内写临时文件、fsync 后原子替换。"""

        directory = path.parent
        descriptor = -1
        temporary_path: str | None = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=directory,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                stream.write(canonical_json(value))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            self._sync_directory(directory)
        except (OSError, TypeError, ValueError) as exc:
            raise InvestigationHarnessRecoveryStateWriteError(
                "无法原子写入 Harness runtime"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


class SchemaV10ReadOnlyInvestigationHarnessAdapter:
    """已停用 composition 的 schema-v10 显式历史迁移 adapter 壳。

    当前源码已删除 composition runner；只有显式历史迁移组装可以注入兼容
    runner。当前 V2 Controller 不构造此类型，也不会接收 schema-v10 的
    composition config、coordinator snapshot 或旧 outcome。
    """

    def __init__(
        self,
        run: Any,
    ) -> None:
        """接收显式历史迁移使用的 ``(request, harness_invocation_id, is_resume)`` runner。"""

        if not callable(run):
            raise InvestigationHarnessContractError("历史迁移 runner 必须可调用")
        self._run = run

    def start(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """通过显式历史迁移 runner 执行首次内部动作。"""

        return self._run(request, harness_invocation_id, False)

    def resume(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
    ) -> BackendResult:
        """通过显式历史迁移 runner 执行同一内部运行的恢复动作。"""

        return self._run(request, harness_invocation_id, True)


class InvestigationHarness:
    """Investigation and Experiment 的持久深模块 facade。"""

    def __init__(
        self,
        runtime_root: str | os.PathLike[str],
        *,
        backend: InvestigationHarnessBackend | None,
        recovery_store: HarnessRecoveryStore | None = None,
    ) -> None:
        """绑定输出目录、恢复事实存储和内部 backend。"""

        self._runtime = _PersistentHarnessRuntime(runtime_root)
        self._backend = backend
        if recovery_store is not None and not isinstance(
            recovery_store,
            HarnessRecoveryStore,
        ):
            raise InvestigationHarnessContractError(
                "recovery_store 必须是 HarnessRecoveryStore"
            )
        self._recovery_store = recovery_store or HarnessRecoveryStore(
            self._runtime.root / "harness_recovery.sqlite3"
        )

    @property
    def recovery_store(self) -> HarnessRecoveryStore:
        """返回当前 invocation 使用的独立恢复事实存储。"""

        return self._recovery_store

    def has_recovery(self, request: InvestigationHarnessRequest | Mapping[str, Any]) -> bool:
        """只读判断一个公开 request 是否已有 Harness 恢复事实。

        Context 在恢复多个 Session 时需要区分“已有子 Harness 应调用 resume”和
        “该 Session 尚未轮到、应首次 start”。这个判断只查询同一
        ``HarnessRecoveryStore``，不会读取展示 manifest、创建 invocation 或连接外部
        provider/MiniChem。
        """

        normalized = self._normalize_request(request)
        return self._read_recovery_snapshot(_recovery_identity(normalized)) is not None

    @staticmethod
    def harness_invocation_id_for(
        request: InvestigationHarnessRequest | Mapping[str, Any],
    ) -> str:
        """从正式 request 返回稳定 Harness invocation 身份。"""

        return InvestigationHarness._normalize_request(request).harness_invocation_id

    def start(
        self,
        request: InvestigationHarnessRequest | Mapping[str, Any],
    ) -> HarnessOutcome:
        """创建或恢复同一 invocation；恢复事实来自 HarnessRecoveryStore。"""

        normalized = self._normalize_request(request)
        harness_invocation_id = normalized.harness_invocation_id
        try:
            snapshot, created, legacy_manifest = self._load_or_create_recovery(
                normalized,
                harness_invocation_id,
                allow_create=True,
            )
            if snapshot is None:  # pragma: no cover - allow_create=True 必然创建
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "Harness recovery invocation 无法创建"
                )
            if created:
                self._runtime.create_manifest(normalized, harness_invocation_id)
            return self._drive(
                normalized,
                harness_invocation_id,
                snapshot,
                operation="start" if created and legacy_manifest is None else "resume",
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            # provider/MCP 暂不可用或恢复资料暂时不可见时保留 store 的 open
            # 事实；调用方收到可重试的 Suspended，而不是一个伪造的失败终态。
            return Suspended(
                harness_invocation_id,
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_recovery_unreadable"
                ),
            )

    def resume(
        self,
        harness_invocation_id: str | InvestigationHarnessRequest | Mapping[str, Any],
        request: InvestigationHarnessRequest | Mapping[str, Any] | None = None,
    ) -> HarnessOutcome:
        """恢复同一 Harness run，可选 request 用于显式核对输入身份。

        正式入口使用 request 的 ``harness_invocation_id``。schema-v1 兼容调用
        仍可传入旧 opaque 引用，由显式 manifest reader 协调，不创建第二个运行。
        """

        candidate_request: InvestigationHarnessRequest | None = None
        supplied_ref: str | None = None
        if isinstance(harness_invocation_id, InvestigationHarnessRequest) or isinstance(
            harness_invocation_id, Mapping
        ):
            if request is not None:
                raise InvestigationHarnessContractError(
                    "resume 不能同时从 request 和第二个 request 参数恢复"
                )
            candidate_request = self._normalize_request(harness_invocation_id)
            requested_ref = candidate_request.harness_invocation_id
        else:
            requested_ref = _require_opaque_ref(
                harness_invocation_id, "harness_invocation_id"
            )
            supplied_ref = requested_ref
            if request is not None:
                candidate_request = self._normalize_request(request)

        if candidate_request is None:
            # 旧调用方只提供 Harness run ref 时，暂时从展示索引得到完整请求；
            # V2 正式恢复入口必须传入 task/module/runtime/input 四项身份。
            try:
                manifest = self._runtime.load_manifest(requested_ref, None)
                if manifest is None:
                    derived_ref = derive_harness_invocation_id(requested_ref)
                    if derived_ref != requested_ref:
                        requested_ref = derived_ref
                        manifest = self._runtime.load_manifest(requested_ref, None)
            except InvestigationHarnessRecoveryEvidenceUnavailable:
                return Suspended(
                    requested_ref,
                    "recovery_evidence_missing",
                    recovery_verification=RecoveryEvidenceMissing(
                        reason_code="harness_manifest_unreadable"
                    ),
                )
            if manifest is None:
                return Suspended(
                    requested_ref,
                    "recovery_evidence_missing",
                    recovery_verification=RecoveryEvidenceMissing(
                        reason_code="harness_manifest_missing"
                    ),
                )
            normalized = cast(InvestigationHarnessRequest, manifest["request"])
        else:
            normalized = candidate_request
        expected_ref = normalized.harness_invocation_id
        if requested_ref != expected_ref:
            # 显式给出的另一份 Harness invocation ID 不能被静默改写。
            if supplied_ref != expected_ref:
                existing = self._read_recovery_snapshot(_recovery_identity(normalized))
                if existing is not None:
                    raise InvestigationHarnessIdentityConflict(
                        "resume 的 harness_invocation_id 与请求身份不一致"
                    )
            requested_ref = expected_ref
        try:
            snapshot, _created, _legacy_manifest = self._load_or_create_recovery(
                normalized,
                requested_ref,
                allow_create=False,
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            return Suspended(
                requested_ref,
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_recovery_unreadable"
                ),
            )
        if snapshot is None:
            return Suspended(
                requested_ref,
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_manifest_missing"
                ),
            )
        try:
            return self._drive(
                normalized,
                requested_ref,
                snapshot,
                operation="resume",
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            return Suspended(
                requested_ref,
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_recovery_unreadable"
                ),
            )

    def start_investigation(
        self,
        request: InvestigationHarnessRequest | Mapping[str, Any],
    ) -> HarnessOutcome:
        """规范名称别名，供 Investigation 模块入口使用。"""

        return self.start(request)

    def resume_investigation(
        self,
        harness_invocation_id: str | InvestigationHarnessRequest | Mapping[str, Any],
        request: InvestigationHarnessRequest | Mapping[str, Any] | None = None,
    ) -> HarnessOutcome:
        """规范名称别名，供 Controller 恢复入口使用。"""

        return self.resume(harness_invocation_id, request)

    @staticmethod
    def _normalize_request(
        request: InvestigationHarnessRequest | Mapping[str, Any],
    ) -> InvestigationHarnessRequest:
        """把调用方输入收敛到不可变公开请求。"""

        if isinstance(request, InvestigationHarnessRequest):
            return request
        try:
            return InvestigationHarnessRequest.from_mapping(request)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, InvestigationHarnessContractError):
                raise
            raise InvestigationHarnessContractError("Harness request 无法解析") from exc

    @staticmethod
    def _assert_same_identity(
        request: InvestigationHarnessRequest,
        persisted: InvestigationHarnessRequest | Mapping[str, Any],
    ) -> None:
        """拒绝同一 Harness invocation 下的任务、模块或输入漂移。"""

        stored = (
            persisted
            if isinstance(persisted, InvestigationHarnessRequest)
            else InvestigationHarnessRequest.from_mapping(persisted)
        )
        if request.to_dict() != stored.to_dict():
            raise InvestigationHarnessIdentityConflict(
                "同一 harness_invocation_id 的任务、模块或输入摘要发生漂移"
            )

    def _read_recovery_snapshot(
        self,
        identity: HarnessRunIdentity,
    ) -> HarnessRecoverySnapshot | None:
        """读取 HarnessRecoveryStore；存储损坏属于恢复资料缺失。"""

        try:
            return self._recovery_store.load(identity)
        except RecoveryStoreIdentityConflict as exc:
            raise InvestigationHarnessIdentityConflict(f"恢复身份发生漂移：{exc}") from exc
        except RecoveryStoreContractError as exc:
            raise InvestigationHarnessContractError(str(exc)) from exc
        except HarnessRecoveryStoreError as exc:
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "Harness recovery store 无法读取"
            ) from exc

    def _append_recovery_transition(
        self,
        identity: HarnessRunIdentity,
        transition: Any,
    ) -> HarnessRecoverySnapshot:
        """追加恢复事实；写入失败必须传播为状态写入错误。"""

        try:
            return self._recovery_store.append_transition(identity, transition)
        except RecoveryStoreIdentityConflict as exc:
            raise InvestigationHarnessIdentityConflict(f"恢复身份发生漂移：{exc}") from exc
        except RecoveryStoreContractError as exc:
            raise InvestigationHarnessContractError(str(exc)) from exc
        except RecoveryStoreStateConflict as exc:
            # 非法生命周期转换是 Harness/backend 合同或程序错误，不把它
            # 误报成 Task/Thread/Job 等身份冲突。
            raise InvestigationHarnessContractError(str(exc)) from exc
        except HarnessRecoveryTransitionConflict as exc:
            # 同一 event id/digest 的事实漂移才属于可核实的完整性冲突。
            raise InvestigationHarnessIdentityConflict(str(exc)) from exc
        except HarnessRecoveryStoreError as exc:
            raise InvestigationHarnessRecoveryStateWriteError(
                "无法写入 Harness recovery store"
            ) from exc

    @staticmethod
    def _manifest_digest(manifest: Mapping[str, Any]) -> str:
        """计算一次旧展示索引协调所需的内容摘要。"""

        request = manifest.get("request")
        if not isinstance(request, InvestigationHarnessRequest):
            raise InvestigationHarnessContractError(
                "Harness manifest request 必须是 InvestigationHarnessRequest"
            )
        body = {
            "schema_version": manifest.get("schema_version"),
            "harness_invocation_id": manifest.get("harness_invocation_id"),
            "request": request.to_dict(),
            "state": manifest.get("state"),
            "failure_code": manifest.get("failure_code"),
        }
        return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()

    @staticmethod
    def _output_reference(
        output: PreparedModuleInvocationOutput,
    ) -> ModuleOutputReference:
        """为持久输出生成不含路径的可重放引用。"""

        return ModuleOutputReference(
            output_ref=f"prepared-output-{output.module_output_digest}",
            output_digest=output.module_output_digest,
        )

    @staticmethod
    def _legacy_manifest_reference(
        request: InvestigationHarnessRequest,
    ) -> str | None:
        """仅为 schema-v1 reader 生成旧 manifest 的只读查找引用。"""

        binding = request.context_binding
        if not isinstance(binding, Mapping) or binding.get("kind") != "LegacyRuntime":
            return None
        legacy_identity = binding.get("identity")
        if not isinstance(legacy_identity, str):
            return None
        try:
            return derive_harness_invocation_id(
                _require_opaque_ref(legacy_identity, "legacy_runtime_identity")
            )
        except InvestigationHarnessContractError:
            return None

    def _write_display_state(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        snapshot: HarnessRecoverySnapshot,
    ) -> None:
        """同步展示索引；它失败时传播写入错误但不改变恢复事实。"""

        try:
            manifest = self._runtime.load_manifest(
                harness_invocation_id,
                request.module_invocation_id,
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            # 展示索引不再是恢复事实来源；已有独立 store 时可以继续恢复。
            return
        if manifest is None:
            return
        self._runtime.update_state(
            harness_invocation_id,
            state=snapshot.state.value,
            failure_code=snapshot.failure_code,
            module_invocation_id=request.module_invocation_id,
        )

    def _migrate_legacy_manifest(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        manifest: Mapping[str, Any],
        snapshot: HarnessRecoverySnapshot,
    ) -> HarnessRecoverySnapshot:
        """只在 store 首次建立时读取旧 manifest，并立即写入独立事实。"""

        identity = _recovery_identity(request)
        if snapshot.manifest_read_digest is None:
            snapshot = self._append_recovery_transition(
                identity,
                RecordManifestRead(
                    self._event_id("manifest-read", self._manifest_digest(manifest)),
                    self._manifest_digest(manifest),
                ),
            )

        state = manifest["state"]
        if state == "failed":
            failure_code = cast(str, manifest["failure_code"])
            if failure_code != "verified_integrity_conflict":
                raise InvestigationHarnessContractError(
                    "旧 Harness manifest 的 failed 状态不是 VerifiedIntegrityConflict"
                )
            if snapshot.state is not HarnessLifecycle.FAILED:
                snapshot = self._append_recovery_transition(
                    identity,
                    FailInvocation(
                        self._event_id("failure", failure_code),
                        failure_code,
                    ),
                )
            return snapshot
        if state == "suspended" and snapshot.state is HarnessLifecycle.OPEN:
            snapshot = self._append_recovery_transition(
                identity,
                SuspendInvocation(
                    self._event_id("suspend", "legacy_manifest_suspended"),
                    "legacy_manifest_suspended",
                ),
            )

        # 即使旧索引声称 open，也优先接管已经完整写入的输出正文。
        try:
            prepared = self._runtime.load_prepared_output(
                harness_invocation_id,
                request.module_invocation_id,
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            prepared = None
        if state == "completed" and prepared is None:
            # 旧索引声称已完成但正文不可读时，当前次恢复只能报告缺失证据；
            # 不能把旧索引的 completed 当成可以重新调用 backend 的授权。
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "已完成旧 Harness invocation 缺少 PreparedModuleInvocationOutput"
            )
        if prepared is not None:
            canonical_ref = request.harness_invocation_id
            if harness_invocation_id != canonical_ref:
                # 旧展示索引仍由显式 reader 读取；迁移后正文复制到当前
                # canonical invocation 目录，后续恢复不再依赖旧引用。
                self._runtime.write_prepared_output(
                    canonical_ref,
                    prepared,
                    request.module_invocation_id,
                )
                harness_invocation_id = canonical_ref
            self._validate_prepared_output(request, prepared)
            self._finish_output(
                request,
                harness_invocation_id,
                snapshot,
                prepared,
                recovery_verification=Reconciled(
                    reason_code="legacy_manifest_imported",
                    reconciled_fields=("display_index",),
                ),
            )
            completed = self._read_recovery_snapshot(identity)
            if completed is None:  # pragma: no cover - finish_output verifies it
                raise InvestigationHarnessRecoveryStateWriteError(
                    "旧展示索引迁移后无法读取 Harness recovery"
                )
            return completed
        return snapshot

    def _load_or_create_recovery(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        *,
        allow_create: bool,
    ) -> tuple[
        HarnessRecoverySnapshot | None,
        bool,
        Mapping[str, Any] | None,
    ]:
        """以完整 invocation 身份读取或创建 store，并协调一次旧 manifest。"""

        identity = _recovery_identity(request)
        existing = self._read_recovery_snapshot(identity)
        if existing is not None:
            return existing, False, None

        manifest_ref = harness_invocation_id
        try:
            legacy_manifest = self._runtime.load_manifest(
                manifest_ref,
                request.module_invocation_id,
            )
            if legacy_manifest is None:
                legacy_ref = self._legacy_manifest_reference(request)
                if legacy_ref is not None and legacy_ref != manifest_ref:
                    legacy_manifest = self._runtime.load_manifest(
                        legacy_ref,
                        request.module_invocation_id,
                    )
                    if legacy_manifest is not None:
                        manifest_ref = legacy_ref
        except InvestigationHarnessIdentityConflict:
            raise
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            raise
        if legacy_manifest is None and not allow_create:
            return None, False, None
        try:
            snapshot = self._recovery_store.create_or_load(identity)
        except RecoveryStoreIdentityConflict as exc:
            raise InvestigationHarnessIdentityConflict(f"恢复身份发生漂移：{exc}") from exc
        except RecoveryStoreContractError as exc:
            raise InvestigationHarnessContractError(str(exc)) from exc
        except HarnessRecoveryStoreError as exc:
            raise InvestigationHarnessRecoveryStateWriteError(
                "无法创建 Harness recovery invocation"
            ) from exc
        if legacy_manifest is not None:
            snapshot = self._migrate_legacy_manifest(
                request,
                manifest_ref,
                legacy_manifest,
                snapshot,
            )
        return snapshot, True, legacy_manifest

    @staticmethod
    def _event_id(prefix: str, value: str) -> str:
        """生成同一事实重试时可精确重放的安全事件身份。"""

        return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"

    def _finish_output(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        snapshot: HarnessRecoverySnapshot,
        prepared: PreparedModuleInvocationOutput,
        *,
        recovery_verification: RecoveryVerification,
    ) -> Completed:
        """确认输出文件可重读后写入输出引用和完成事实。"""

        prepared = self._validate_prepared_output(request, prepared)
        reread = self._runtime.load_prepared_output(
            harness_invocation_id,
            request.module_invocation_id,
        )
        if reread is None:
            raise InvestigationHarnessRecoveryStateWriteError(
                "PreparedModuleInvocationOutput 写入后无法重新读取"
            )
        reread = self._validate_prepared_output(request, reread)
        expected_reference = self._output_reference(reread)
        identity = _recovery_identity(request)
        if snapshot.committed_output is None:
            snapshot = self._append_recovery_transition(
                identity,
                CommitModuleOutput(
                    self._event_id(
                        "output-committed", reread.module_output_digest
                    ),
                    expected_reference,
                ),
            )
        elif snapshot.committed_output != expected_reference:
            raise InvestigationHarnessIdentityConflict(
                "HarnessRecoveryStore 的输出引用与文件摘要不一致"
            )
        if snapshot.state is not HarnessLifecycle.COMPLETED:
            snapshot = self._append_recovery_transition(
                identity,
                CompleteInvocation(self._event_id("invocation-completed", "completed")),
            )
        final_snapshot = self._read_recovery_snapshot(identity)
        if final_snapshot is None or final_snapshot.state is not HarnessLifecycle.COMPLETED:
            raise InvestigationHarnessRecoveryStateWriteError(
                "Harness recovery 完成事实写入后无法确认"
            )
        final_output = self._runtime.load_prepared_output(
            harness_invocation_id,
            request.module_invocation_id,
        )
        if final_output is None:
            raise InvestigationHarnessRecoveryStateWriteError(
                "Harness 完成事实存在但输出正文不可读"
            )
        final_output = self._validate_prepared_output(request, final_output)
        if final_snapshot.committed_output != self._output_reference(final_output):
            raise InvestigationHarnessIdentityConflict(
                "Harness 完成事实与输出正文摘要不一致"
            )
        self._write_display_state(request, harness_invocation_id, final_snapshot)
        return Completed(
            harness_invocation_id,
            final_output,
            recovery_verification=recovery_verification,
        )

    def _record_suspended(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        reason_code: str,
        recovery_verification: RecoveryVerification,
    ) -> Suspended:
        """写入暂停 transition，并返回可继续的公开结果。"""

        updated = self._append_recovery_transition(
            _recovery_identity(request),
            SuspendInvocation(
                self._event_id("suspend", reason_code),
                reason_code,
            ),
        )
        self._write_display_state(request, harness_invocation_id, updated)
        return Suspended(
            harness_invocation_id,
            reason_code,
            recovery_verification=recovery_verification,
        )

    def _record_failed(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        failure_code: str,
        recovery_verification: RecoveryVerification,
    ) -> FatalFailure:
        """写入明确失败 transition，并返回终态结果。"""

        updated = self._append_recovery_transition(
            _recovery_identity(request),
            FailInvocation(
                self._event_id("failure", failure_code),
                failure_code,
            ),
        )
        self._write_display_state(request, harness_invocation_id, updated)
        return FatalFailure(
            harness_invocation_id,
            failure_code,
            recovery_verification=recovery_verification,
        )

    def _drive(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        snapshot: HarnessRecoverySnapshot,
        *,
        operation: str,
    ) -> HarnessOutcome:
        """以 store 状态为准恢复输出，否则只驱动一次内部 backend。"""

        if snapshot.state is HarnessLifecycle.FAILED:
            failure_code = snapshot.failure_code
            if failure_code is None:  # pragma: no cover - store 合同禁止该状态
                raise InvestigationHarnessRecoveryEvidenceUnavailable(
                    "failed Harness invocation 缺少 failure_code"
                )
            if failure_code != "verified_integrity_conflict":
                raise InvestigationHarnessContractError(
                    "Harness recovery 的 failed 状态不是 VerifiedIntegrityConflict"
                )
            verification = (
                VerifiedIntegrityConflict(reason_code=failure_code)
                if failure_code == "verified_integrity_conflict"
                else None
            )
            return FatalFailure(
                harness_invocation_id,
                failure_code,
                recovery_verification=verification,
            )

        try:
            prepared = self._runtime.load_prepared_output(
                harness_invocation_id,
                request.module_invocation_id,
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            if snapshot.state is HarnessLifecycle.COMPLETED or snapshot.committed_output is not None:
                return Suspended(
                    harness_invocation_id,
                    "recovery_evidence_missing",
                    recovery_verification=RecoveryEvidenceMissing(
                        reason_code="prepared_output_unreadable"
                    ),
                )
            prepared = None

        try:
            if prepared is not None:
                self._validate_prepared_output(request, prepared)
                return self._finish_output(
                    request,
                    harness_invocation_id,
                    snapshot,
                    prepared,
                    recovery_verification=(
                        Consistent(reason_code="committed_output_match")
                        if snapshot.state is HarnessLifecycle.COMPLETED
                        else Reconciled(
                            reason_code="committed_output_replayed",
                            reconciled_fields=("invocation_state",),
                        )
                    ),
                )

            if snapshot.state is HarnessLifecycle.COMPLETED:
                return Suspended(
                    harness_invocation_id,
                    "recovery_evidence_missing",
                    recovery_verification=RecoveryEvidenceMissing(
                        reason_code="prepared_output_missing"
                    ),
                )
            if snapshot.committed_output is not None:
                return Suspended(
                    harness_invocation_id,
                    "recovery_evidence_missing",
                    recovery_verification=RecoveryEvidenceMissing(
                        reason_code="committed_output_missing"
                    ),
                )

            raw_result = self._invoke_backend(request, harness_invocation_id, operation)
            result = self._normalize_backend_result(raw_result, harness_invocation_id)
            if isinstance(result, BackendCompleted):
                self._validate_completion_verification(result.recovery_verification)
                prepared = self._validate_prepared_output(request, result.prepared_output)
                stored = self._runtime.write_prepared_output(
                    harness_invocation_id,
                    prepared,
                    request.module_invocation_id,
                )
                return self._finish_output(
                    request,
                    harness_invocation_id,
                    snapshot,
                    stored,
                    recovery_verification=result.recovery_verification,
                )
            if isinstance(result, BackendSuspended):
                return self._record_suspended(
                    request,
                    harness_invocation_id,
                    result.reason_code,
                    result.recovery_verification,
                )
            if isinstance(result, BackendFatalFailure):
                return self._record_failed(
                    request,
                    harness_invocation_id,
                    result.failure_code,
                    result.recovery_verification,
                )
            raise InvestigationHarnessContractError("backend 结果无法归类")
        except InvestigationHarnessIdentityConflict as conflict:
            # 真实身份冲突必须先形成独立 failed 事实，再把异常交给上层停止。
            # 已经写入 output reference 的 invocation 不能再追加 failed；此时
            # 保留原始冲突，让上层按完整性冲突停止，避免用状态错误覆盖根因。
            if (
                snapshot.state is not HarnessLifecycle.FAILED
                and snapshot.committed_output is None
            ):
                try:
                    self._record_failed(
                        request,
                        harness_invocation_id,
                        "verified_integrity_conflict",
                        VerifiedIntegrityConflict(reason_code="harness_identity_conflict"),
                    )
                except InvestigationHarnessIdentityConflict as write_conflict:
                    raise InvestigationHarnessRecoveryStateWriteError(
                        "身份冲突事实无法写入 Harness recovery store"
                    ) from write_conflict
            raise conflict

    def _invoke_backend(
        self,
        request: InvestigationHarnessRequest,
        harness_invocation_id: str,
        operation: str,
    ) -> BackendResult:
        """调用内部 backend；未声明异常保持可重试，不写入失败事实。"""

        if self._backend is None:
            raise InvestigationHarnessRecoveryEvidenceUnavailable(
                "该 Harness run 没有可用 backend"
            )
        method = getattr(self._backend, operation, None)
        if not callable(method):
            raise InvestigationHarnessContractError(
                f"backend 缺少 {operation} 接口"
            )
        # 连接超时、Controller 退出或其他未声明异常不等于已核实的不可恢复
        # 失败。让异常传播，保留 manifest 的 open 状态，使下一次 resume 能够
        # 重新进入同一个 Harness run；只有 backend 明确返回 BackendFatalFailure
        # 或输出身份核对失败时才写入 failed。
        return cast(BackendResult, method(request, harness_invocation_id))

    @staticmethod
    def _normalize_backend_result(
        result: BackendResult | Completed | Suspended | FatalFailure,
        harness_invocation_id: str,
    ) -> BackendCompleted | BackendSuspended | BackendFatalFailure:
        """将 backend 结果收敛成不含内部身份的三分支。"""

        if isinstance(result, PreparedModuleInvocationOutput):
            return BackendCompleted(result)
        if isinstance(result, BackendCompleted):
            return result
        if isinstance(result, BackendSuspended):
            return result
        if isinstance(result, BackendFatalFailure):
            return result
        if isinstance(result, RecoveryVerification):
            if result.must_suspend:
                return BackendSuspended(
                    result.reason_code,
                    recovery_verification=result,
                )
            if result.must_fail_integrity:
                return BackendFatalFailure(
                    "verified_integrity_conflict",
                    recovery_verification=result,
                )
            raise InvestigationHarnessContractError(
                "Consistent 或 Reconciled 必须同时携带 PreparedModuleInvocationOutput"
            )
        if isinstance(result, Completed):
            if result.harness_invocation_id != harness_invocation_id:
                raise InvestigationHarnessIdentityConflict(
                    "backend Completed 的 harness_invocation_id 发生漂移"
                )
            return BackendCompleted(
                result.prepared_output,
                recovery_verification=result.recovery_verification,
            )
        if isinstance(result, Suspended):
            if result.harness_invocation_id != harness_invocation_id:
                raise InvestigationHarnessIdentityConflict(
                    "backend Suspended 的 harness_invocation_id 发生漂移"
                )
            return BackendSuspended(
                result.reason_code,
                recovery_verification=result.recovery_verification,
            )
        if isinstance(result, FatalFailure):
            if result.harness_invocation_id != harness_invocation_id:
                raise InvestigationHarnessIdentityConflict(
                    "backend FatalFailure 的 harness_invocation_id 发生漂移"
                )
            return BackendFatalFailure(
                result.failure_code,
                recovery_verification=result.recovery_verification,
            )
        raise InvestigationHarnessContractError("backend 返回了未注册结果")

    @staticmethod
    def _validate_completion_verification(
        verification: RecoveryVerification,
    ) -> None:
        """成功输出只能来自完全一致或已明确协调的恢复身份。"""

        if not isinstance(verification, RecoveryVerification):
            raise InvestigationHarnessContractError(
                "backend completed 缺少恢复核验结果"
            )
        if verification.status not in {
            RecoveryStatus.CONSISTENT,
            RecoveryStatus.RECONCILED,
        }:
            raise InvestigationHarnessContractError(
                "成功输出不得掩盖暂停或完整性冲突"
            )

    @staticmethod
    def _validate_prepared_output(
        request: InvestigationHarnessRequest,
        output: PreparedModuleInvocationOutput,
    ) -> PreparedModuleInvocationOutput:
        """核对持久输出仍属于当前输入；不解释科学 payload。"""

        if not isinstance(output, PreparedModuleInvocationOutput):
            raise InvestigationHarnessContractError(
                "backend output 必须是 PreparedModuleInvocationOutput"
            )
        output_mapping = output.to_dict()
        if output_mapping["module_input_digest"] != request.module_input_digest:
            raise InvestigationHarnessIdentityConflict(
                "PreparedModuleInvocationOutput 的 module_input_digest 发生漂移"
            )
        if output_mapping["module_invocation_id"] != request.module_invocation_id:
            raise InvestigationHarnessIdentityConflict(
                "PreparedModuleInvocationOutput 的 module_invocation_id 发生漂移"
            )
        if any(
            record.research_task_id != request.research_task_id
            for record in output.output_scientific_records
        ):
            raise InvestigationHarnessIdentityConflict(
                "PreparedModuleInvocationOutput 的 research_task_id 发生漂移"
            )
        if any(
            record.recorded_at is not None
            for record in output.output_scientific_records
        ):
            raise InvestigationHarnessIdentityConflict(
                "PreparedModuleInvocationOutput 不得预先写入 recorded_at"
            )
        return output


__all__ = [
    "BackendCompleted",
    "BackendFatalFailure",
    "BackendResult",
    "BackendSuspended",
    "Completed",
    "SchemaV10ReadOnlyInvestigationHarnessAdapter",
    "SchemaV1LegacyHarnessManifestReader",
    "SchemaV1LegacyPreparedModuleOutputReader",
    "SchemaV1LegacyInvestigationHarnessRequestReader",
    "FatalFailure",
    "HarnessOutcome",
    "InvestigationHarness",
    "InvestigationHarnessBackend",
    "InvestigationHarnessContractError",
    "InvestigationHarnessError",
    "InvestigationHarnessIdentityConflict",
    "InvestigationHarnessPersistenceError",
    "InvestigationHarnessRecoveryEvidenceUnavailable",
    "InvestigationHarnessRecoveryStateWriteError",
    "InvestigationHarnessRequest",
    "InvestigationHarnessRunNotFound",
    "Suspended",
    "RecoveryStatus",
    "RecoveryVerification",
    "Consistent",
    "Reconciled",
    "TemporarilyUnavailable",
    "RecoveryEvidenceMissing",
    "VerifiedIntegrityConflict",
    "derive_harness_invocation_id",
]
