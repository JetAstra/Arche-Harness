"""investigation_v2 的模块级 Controller。

这个模块是当前生产入口使用的 V2 seam。它只在模块边界
读取 Working Memory 的 ``ResearchTaskFacts``，决定下一个公开模块，并把模块内部运行交给
对应 adapter。Investigation and Experiment 仍由 Harness 深模块拥有；Controller
不会读取 thread、turn、Goal、scope、job、submission 或 receipt。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Protocol, TypeAlias

from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    BackendFatalFailure,
    BackendSuspended,
    Completed as HarnessCompleted,
    FatalFailure as HarnessFatalFailure,
    InvestigationHarness,
    InvestigationHarnessIdentityConflict,
    InvestigationHarnessRecoveryEvidenceUnavailable,
    InvestigationHarnessRequest,
    Suspended as HarnessSuspended,
)
from chemistry_multiagent.investigation.recovery_contract import (
    Consistent,
    RecoveryEvidenceMissing,
    RecoveryStatus,
    RecoveryVerification,
    TemporarilyUnavailable,
    VerifiedIntegrityConflict,
)
from chemistry_multiagent.memory import (
    ModuleInvocationClosed,
    IdempotencyConflict,
    ResearchTaskWorkingMemory,
    ModuleInvocationRequest as LedgerModuleInvocationRequest,
    ModuleInvocationSnapshot,
    PreparedModuleInvocationOutput,
    ScientificRecord,
    ResearchTaskFacts,
)
from chemistry_multiagent.memory.contracts import canonical_json
from chemistry_multiagent.memory.v2_identities import (
    ContextBinding,
    CreateResearchTaskReviewSession,
    CreateInvestigationContext,
    ReuseResearchTaskReviewSession,
    ReuseInvestigationContext,
    StatelessModule,
)
from chemistry_multiagent.controllers.controller_lease import (
    ControllerLeaseLost as ControllerLeaseLostError,
)
from chemistry_multiagent.controllers.investigation_v2_confirmation import (
    CONFIRMATION_MODULE_KIND,
    CONFIRMATION_REASON_CODE,
    CONFIRMATION_TASK_CONTRACT_VERSION,
    HypothesisConfirmationService,
    HypothesisConfirmationSubmission,
    HypothesisConfirmationView,
    ResearchMaterialUpload,
)
from chemistry_multiagent.investigation.research_material_store import (
    ResearchMaterialStore,
)


MODULE_SEQUENCE: tuple[str, ...] = (
    "retrieval",
    "research-query-formulation",
    "hypothesis-generation",
    "hypothesis-selection",
    CONFIRMATION_MODULE_KIND,
    "hypothesis-investigation",
    "task-review",
)

AUTHOR_MODULE_SEQUENCE: tuple[str, ...] = (
    "hypothesis-import",
    CONFIRMATION_MODULE_KIND,
    "hypothesis-investigation",
    "task-review",
)

ENTRY_STRATEGIES = frozenset(
    {
        "standard_hypothesis_generation",
        "author_hypothesis_import",
    }
)

_CANONICAL_MODULE_KINDS = frozenset(
    {
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
        "hypothesis-import",
        CONFIRMATION_MODULE_KIND,
        "hypothesis-investigation",
        "task-review",
    }
)
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MISSING = object()


class ResearchWorkflowControllerError(RuntimeError):
    """V2 Controller 无法安全推进模块边界。"""


class ResearchWorkflowControllerContractError(
    ResearchWorkflowControllerError, ValueError
):
    """模块请求或公开结果不满足 V2 合同。"""


class ResearchWorkflowControllerIntegrityConflict(
    ResearchWorkflowControllerError
):
    """Task Facts 与模块输出/恢复身份发生已核实冲突。"""


def _require_text(value: Any, field_name: str) -> str:
    """校验公开字符串字段。"""

    if not isinstance(value, str) or not value.strip():
        raise ResearchWorkflowControllerContractError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _require_safe_token(value: Any, field_name: str) -> str:
    """校验生成的模块身份不携带路径或控制字符。"""

    value = _require_text(value, field_name)
    if not _SAFE_TOKEN_RE.fullmatch(value):
        raise ResearchWorkflowControllerContractError(
            f"{field_name} 必须是不透明安全标识"
        )
    return value


def _coalesce_alias(
    value: Mapping[str, Any],
    canonical_name: str,
    legacy_name: str,
    field_name: str,
) -> Any:
    """读取公开字段别名，并拒绝新旧身份同时漂移。"""

    canonical = value.get(canonical_name, _MISSING)
    legacy = value.get(legacy_name, _MISSING)
    if canonical is not _MISSING and legacy is not _MISSING:
        if canonical_json(canonical) != canonical_json(legacy):
            raise ResearchWorkflowControllerContractError(
                f"{field_name} 的新旧字段别名内容不一致"
            )
    if canonical is not _MISSING:
        return canonical
    if legacy is not _MISSING:
        return legacy
    return None


def normalize_module_kind(value: Any) -> str:
    """校验活动模块名称为正式的 research_task 模块名。

    活动 Controller 不再把 ``query``、``selection`` 或 ``review`` 等历史别名
    当作输入接受。schema-v10 事件需要旧名称时，必须显式调用
    ``ModuleInvocationRequest.from_schema_v10_mapping`` 读取并转换。
    """

    value = _require_text(value, "module_kind")
    normalized = value.casefold().replace("_", "-").replace(" ", "-")
    if normalized not in _CANONICAL_MODULE_KINDS:
        raise ResearchWorkflowControllerContractError(
            f"module_kind 未注册：{value}"
        )
    return normalized


def build_module_input_digest(
    research_task_id: str,
    module_kind: str,
    input_scientific_record_ids: Sequence[str],
) -> str:
    """为公开模块输入引用生成稳定摘要。

    科学内容由不可变 Scientific Record 引用携带；摘要只绑定 task、模块类型和
    有序引用，供 Working Memory 与 Harness 共同核对，不读取任何 Harness 私有
    运行身份。
    """

    research_task_id = _require_text(research_task_id, "research_task_id")
    module_kind = normalize_module_kind(module_kind)
    refs = tuple(
        _require_text(ref, "input_scientific_record_ids[]")
        for ref in input_scientific_record_ids
    )
    if len(set(refs)) != len(refs):
        raise ResearchWorkflowControllerContractError(
            "input_record_refs 不能包含重复引用"
        )
    body = {
        "research_task_id": research_task_id,
        "module_kind": module_kind,
        "input_scientific_record_ids": list(refs),
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModuleInvocationRequest(Mapping[str, Any]):
    """Controller 与公开模块 adapter 之间的 canonical Module Invocation。

    生产构造器只接受 research_task 合同字段。旧 schema-v10 mapping 只能通过
    ``from_schema_v10_mapping`` 只读转换，canonical 对象不提供旧属性。
    """

    research_task_id: str
    module_invocation_id: str
    module_kind: str
    input_scientific_record_ids: tuple[str, ...]
    module_input_digest: str
    context_binding: ContextBinding = field(default_factory=StatelessModule)

    def __post_init__(self) -> None:
        """校验 canonical 身份、输入摘要和 typed context binding。"""

        object.__setattr__(
            self,
            "research_task_id",
            _require_text(self.research_task_id, "research_task_id"),
        )
        object.__setattr__(
            self,
            "module_invocation_id",
            _require_safe_token(self.module_invocation_id, "module_invocation_id"),
        )
        object.__setattr__(self, "module_kind", normalize_module_kind(self.module_kind))
        refs = tuple(
            _require_text(ref, "input_scientific_record_ids[]")
            for ref in self.input_scientific_record_ids
        )
        if len(set(refs)) != len(refs):
            raise ResearchWorkflowControllerContractError(
                "input_scientific_record_ids 不能包含重复引用"
            )
        object.__setattr__(self, "input_scientific_record_ids", refs)
        object.__setattr__(
            self,
            "module_input_digest",
            _require_text(self.module_input_digest, "module_input_digest"),
        )
        if not isinstance(
            self.context_binding,
            (
                StatelessModule,
                CreateInvestigationContext,
                ReuseInvestigationContext,
                CreateResearchTaskReviewSession,
                ReuseResearchTaskReviewSession,
            ),
        ):
            raise ResearchWorkflowControllerContractError(
                "context_binding 必须是已注册的 V2 owner 类型"
            )

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问。"""

        return self.to_dict()[key]

    def __iter__(self):
        """按公开字段顺序迭代。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回公开字段数。"""

        return 6

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical Module Invocation mapping，不泄漏 runtime_ref。"""

        return {
            "research_task_id": self.research_task_id,
            "module_invocation_id": self.module_invocation_id,
            "module_kind": self.module_kind,
            "input_scientific_record_ids": list(self.input_scientific_record_ids),
            "module_input_digest": self.module_input_digest,
            "context_binding": self.context_binding.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModuleInvocationRequest":
        """从 canonical Module Invocation mapping 恢复活动请求。

        此 parser 只接受 canonical 字段；schema-v10 的旧 envelope 必须调用
        ``from_schema_v10_mapping``，以便在代码审查和测试中清楚区分读旧写新。
        """

        if not isinstance(value, Mapping):
            raise ResearchWorkflowControllerContractError("ModuleInvocationRequest 必须是 object")
        allowed = {
            "research_task_id",
            "module_invocation_id",
            "module_kind",
            "input_scientific_record_ids",
            "module_input_digest",
            "context_binding",
        }
        extra = sorted(set(value).difference(allowed))
        if extra:
            raise ResearchWorkflowControllerContractError(
                "ModuleInvocationRequest 含非 canonical 字段：" + ", ".join(extra)
            )
        context_binding = cls._context_binding_from_mapping(value.get("context_binding"))
        try:
            return cls(
                research_task_id=value["research_task_id"],
                module_invocation_id=value["module_invocation_id"],
                module_kind=value["module_kind"],
                input_scientific_record_ids=tuple(value["input_scientific_record_ids"]),
                module_input_digest=value["module_input_digest"],
                context_binding=context_binding,
            )
        except KeyError as exc:
            raise ResearchWorkflowControllerContractError(
                f"ModuleInvocationRequest 缺少 {exc.args[0]}"
            ) from exc

    @classmethod
    def from_schema_v10_mapping(cls, value: Mapping[str, Any]) -> "ModuleInvocationRequest":
        """只读解析 schema-v10 Module Invocation 旧 envelope。

        旧模块别名、旧任务身份和 runtime 字段只在此函数中出现；解析结果立即
        转成 canonical 请求，后续生产流程不再携带旧 mapping。
        """

        if not isinstance(value, Mapping):
            raise ResearchWorkflowControllerContractError("schema-v10 ModuleInvocationRequest 必须是 object")
        legacy_modules = {
            "retrieval": "retrieval",
            "query": "research-query-formulation",
            "hypothesis": "hypothesis-generation",
            "hypotheses": "hypothesis-generation",
            "hypothesis-generation": "hypothesis-generation",
            "hypothesis_import": "hypothesis-import",
            "hypothesis-import": "hypothesis-import",
            "selection": "hypothesis-selection",
            "top5": "hypothesis-selection",
            "top-5": "hypothesis-selection",
            "top5-selection": "hypothesis-selection",
            "top-5-selection": "hypothesis-selection",
            "hypothesis-confirmation": CONFIRMATION_MODULE_KIND,
            "hypothesis_confirmation": CONFIRMATION_MODULE_KIND,
            "confirmation": CONFIRMATION_MODULE_KIND,
            "investigation": "hypothesis-investigation",
            "investigation-and-experiment": "hypothesis-investigation",
            "investigation-experiment": "hypothesis-investigation",
            "review": "task-review",
            "task-review": "task-review",
        }
        raw_kind = _require_text(value.get("module_kind"), "module_kind")
        normalized_kind = raw_kind.casefold().replace("_", "-").replace(" ", "-")
        try:
            module_kind = legacy_modules[normalized_kind]
        except KeyError as exc:
            raise ResearchWorkflowControllerContractError(
                f"schema-v10 module_kind 未注册：{raw_kind}"
            ) from exc
        context_binding = cls._context_binding_from_schema_v10(value.get("context_binding"))
        if context_binding is None:
            context_binding = cls._context_binding_from_schema_v10_runtime(
                value.get("runtime_binding", value.get("runtime")),
                module_kind=module_kind,
                value_mapping=value,
            )
        if context_binding is None and module_kind in {
            "hypothesis-investigation",
            "task-review",
        }:
            raise ResearchWorkflowControllerContractError(
                "schema-v10 有状态 Module Invocation 缺少可转换的 context owner"
            )
        task_id = value.get("research_task_id", value.get("inquiry_id", value.get("task_id")))
        invocation_id = value.get(
            "module_invocation_id",
            value.get("module_run_id"),
        )
        refs = value.get(
            "input_scientific_record_ids",
            value.get("input_record_refs", ()),
        )
        digest = value.get("module_input_digest", value.get("input_digest"))
        if context_binding is None:
            context_binding = StatelessModule()
        return cls(
            research_task_id=task_id,
            module_invocation_id=invocation_id,
            module_kind=module_kind,
            input_scientific_record_ids=tuple(refs),
            module_input_digest=digest,
            context_binding=context_binding,
        )

    @staticmethod
    def _context_binding_from_mapping(value: Any) -> ContextBinding:
        """解析 canonical typed context_binding。"""

        if not isinstance(value, Mapping):
            raise ResearchWorkflowControllerContractError(
                "context_binding 必须是 typed object"
            )
        kind = value.get("kind")
        try:
            if kind == "StatelessModule":
                return StatelessModule()
            if kind == "CreateInvestigationContext":
                return CreateInvestigationContext(
                    value["investigation_context_id"],
                    value["hypothesis_batch_record_id"],
                )
            if kind == "ReuseInvestigationContext":
                return ReuseInvestigationContext(
                    value["investigation_context_id"],
                    value["hypothesis_batch_record_id"],
                    value["source_module_invocation_id"],
                )
            if kind == "CreateResearchTaskReviewSession":
                return CreateResearchTaskReviewSession(
                    value["research_task_review_session_id"],
                )
            if kind == "ReuseResearchTaskReviewSession":
                return ReuseResearchTaskReviewSession(
                    value["research_task_review_session_id"],
                    value["source_module_invocation_id"],
                )
        except KeyError as exc:
            raise ResearchWorkflowControllerContractError(
                f"context_binding 缺少 {exc.args[0]}"
            ) from exc
        raise ResearchWorkflowControllerContractError(
            f"context_binding.kind 未注册：{kind}"
        )

    @classmethod
    def _context_binding_from_schema_v10(cls, value: Any) -> ContextBinding | None:
        """只读解析历史 context owner，并转换为 canonical typed owner。"""

        if not isinstance(value, Mapping):
            return None
        kind = value.get("kind", value.get("type"))
        if kind in {None, "CreateRuntime", "ReuseRuntime", "create", "reuse"}:
            return None
        if kind == "StatelessModule":
            return StatelessModule()
        if kind == "CreateInvestigationContext":
            return CreateInvestigationContext(
                value.get("investigation_context_id"),
                value.get("hypothesis_batch_record_id"),
            )
        if kind == "ReuseInvestigationContext":
            return ReuseInvestigationContext(
                value.get("investigation_context_id"),
                value.get("hypothesis_batch_record_id"),
                value.get("source_module_invocation_id"),
            )
        if kind in {"CreateResearchTaskReviewSession", "CreateInquiryReviewSession"}:
            return CreateResearchTaskReviewSession(
                value.get(
                    "research_task_review_session_id",
                    value.get("inquiry_review_session_id"),
                )
            )
        if kind in {"ReuseResearchTaskReviewSession", "ReuseInquiryReviewSession"}:
            return ReuseResearchTaskReviewSession(
                value.get(
                    "research_task_review_session_id",
                    value.get("inquiry_review_session_id"),
                ),
                value.get("source_module_invocation_id"),
            )
        raise ResearchWorkflowControllerContractError(
            f"schema-v10 context kind 未注册：{kind}"
        )

    @staticmethod
    def _context_binding_from_schema_v10_runtime(
        value: Any,
        *,
        module_kind: str,
        value_mapping: Mapping[str, Any] | None = None,
        value_alias: Mapping[str, Any] | None = None,
    ) -> ContextBinding | None:
        """只在 schema-v10 reader 内把旧 runtime JSON 转成 typed owner。"""

        if not isinstance(value, Mapping):
            return None
        source_value = value_mapping or value_alias or {}
        owner = value.get("runtime_ref", value.get("runtime_identity"))
        if owner is None:
            return None
        kind = value.get("kind", value.get("type"))
        source = value.get(
            "source_module_invocation_id",
            value.get("source_module_run_ref"),
        )
        try:
            if module_kind == "hypothesis-investigation":
                batch_id = source_value.get(
                    "hypothesis_batch_record_id",
                    source_value.get("hypothesis_batch_ref"),
                )
                if batch_id is None:
                    return None
                if kind in {"ReuseRuntime", "reuse"} and source is not None:
                    return ReuseInvestigationContext(owner, batch_id, source)
                return CreateInvestigationContext(owner, batch_id)
            if module_kind == "task-review":
                if kind in {"ReuseRuntime", "reuse"} and source is not None:
                    return ReuseResearchTaskReviewSession(owner, source)
                return CreateResearchTaskReviewSession(owner)
            return StatelessModule()
        except (TypeError, ValueError) as exc:
            raise ResearchWorkflowControllerContractError(
                "schema-v10 runtime 身份不能转换为 typed context_binding"
            ) from exc

    def to_public_dict(self) -> dict[str, Any]:
        """返回字段重构后的 V2 公共 envelope。"""

        value: dict[str, Any] = {
            "research_task_id": self.research_task_id,
            "module_invocation_id": self.module_invocation_id,
            "module_kind": self.module_kind,
            "input_scientific_record_ids": list(self.input_scientific_record_ids),
            "module_input_digest": self.module_input_digest,
        }
        value["context_binding"] = self.context_binding.to_dict()
        return value


@dataclass(frozen=True)
class ModuleCompleted:
    """公开模块已准备好密封输出。"""

    prepared_output: PreparedModuleInvocationOutput
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: Consistent(reason_code="module_output_ready")
    )
    status: str = field(init=False, default="Completed")

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_output, PreparedModuleInvocationOutput):
            raise ResearchWorkflowControllerContractError(
                "ModuleCompleted.prepared_output 必须是 PreparedModuleInvocationOutput"
            )
        if not isinstance(self.recovery_verification, RecoveryVerification) or self.recovery_verification.status not in {
            RecoveryStatus.CONSISTENT,
            RecoveryStatus.RECONCILED,
        }:
            raise ResearchWorkflowControllerContractError(
                "ModuleCompleted 必须使用 Consistent 或 Reconciled"
            )


@dataclass(frozen=True)
class ModuleSuspended:
    """公开模块暂时不能完成，仍保留 Open Module Run。"""

    reason_code: str
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: TemporarilyUnavailable(reason_code="module_suspended")
    )
    status: str = field(init=False, default="Suspended")

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _require_safe_token(self.reason_code, "reason_code"))
        if not isinstance(self.recovery_verification, RecoveryVerification) or self.recovery_verification.status not in {
            RecoveryStatus.TEMPORARILY_UNAVAILABLE,
            RecoveryStatus.RECOVERY_EVIDENCE_MISSING,
        }:
            raise ResearchWorkflowControllerContractError(
                "ModuleSuspended 必须使用 TemporarilyUnavailable 或 RecoveryEvidenceMissing"
            )


@dataclass(frozen=True)
class ModuleFailed:
    """公开模块已确认不可恢复失败；Module Run 仍留在账本中供诊断。"""

    failure_code: str
    recovery_verification: RecoveryVerification | None = None
    status: str = field(init=False, default="Failed")

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_code", _require_safe_token(self.failure_code, "failure_code"))
        if (
            self.recovery_verification is None
            and self.failure_code == "verified_integrity_conflict"
        ):
            object.__setattr__(
                self,
                "recovery_verification",
                VerifiedIntegrityConflict(reason_code=self.failure_code),
            )
        if self.recovery_verification is not None and not self.recovery_verification.must_fail_integrity:
            raise ResearchWorkflowControllerContractError(
                "ModuleFailed 只能携带 VerifiedIntegrityConflict"
            )


ModuleInvocationOutcome: TypeAlias = ModuleCompleted | ModuleSuspended | ModuleFailed


@dataclass(frozen=True)
class ControllerAdvanced:
    """一个模块已闭合，Controller 已写回事实。"""

    research_task_id: str
    completed_module_invocation_id: str
    next_module_kind: str | None
    status: str = field(init=False, default="Advanced")


@dataclass(frozen=True)
class ControllerSuspended:
    """模块暂停，Task 和 Module Run 均保持可恢复。"""

    research_task_id: str
    module_invocation_id: str
    reason_code: str
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: TemporarilyUnavailable(reason_code="module_suspended")
    )
    status: str = field(init=False, default="Suspended")

    def __post_init__(self) -> None:
        """暂停结果必须明确属于暂时不可用或证据缺失。"""

        if not isinstance(self.recovery_verification, RecoveryVerification) or self.recovery_verification.status not in {
            RecoveryStatus.TEMPORARILY_UNAVAILABLE,
            RecoveryStatus.RECOVERY_EVIDENCE_MISSING,
        }:
            raise ResearchWorkflowControllerContractError(
                "ControllerSuspended 必须使用 TemporarilyUnavailable 或 RecoveryEvidenceMissing"
            )


@dataclass(frozen=True)
class ControllerAwaitingResearcherConfirmation:
    """任务停在研究员确认边界，未启动 Investigation 外部副作用。"""

    research_task_id: str
    view: HypothesisConfirmationView
    status: str = field(init=False, default="AwaitingResearcherConfirmation")

    @property
    def reason_code(self) -> str:
        """返回固定的研究员确认原因。"""

        return self.view.reason_code

    @property
    def expected_research_task_sequence(self) -> int:
        """返回提交确认所需的 Working Memory 序号。"""

        return self.view.expected_research_task_sequence

    @property
    def hypothesis_batch_record_id(self) -> str:
        """返回待确认候选所属的 Hypothesis Batch 引用。"""

        return self.view.hypothesis_batch_record_id

    @property
    def candidates(self) -> list[dict[str, Any]]:
        """返回研究员可见候选的独立副本。"""

        return self.view.to_dict()["candidates"]

    def to_public_dict(self) -> dict[str, Any]:
        """返回不包含 Module Run 等内部身份的确认视图。"""

        return self.view.to_dict()


@dataclass(frozen=True)
class ControllerConfirmationReplayed:
    """确认请求已精确应用；不再次推进已有的下游 Open Module。"""

    research_task_id: str
    confirmation_id: str
    status: str = field(init=False, default="ConfirmationReplayed")
    reason_code: str = field(init=False, default="confirmation_already_applied")


@dataclass(frozen=True)
class ControllerLeaseLostResult:
    """Controller 失去运行时 lease；账本事实保持可恢复。"""

    research_task_id: str
    module_invocation_id: str | None
    status: str = field(init=False, default="LeaseLost")
    reason_code: str = field(init=False, default="controller_lease_lost")


@dataclass(frozen=True)
class ControllerFailed:
    """模块或合同失败；不会伪造 ResearchTaskClosedResult。"""

    research_task_id: str
    module_invocation_id: str
    failure_code: str
    status: str = field(init=False, default="Failed")


@dataclass(frozen=True)
class ResearchTaskClosedResult:
    """Task 已由 Working Memory 写入明确终态。"""

    research_task_id: str
    outcome: Mapping[str, Any]
    status: str = field(init=False)

    def __post_init__(self) -> None:
        outcome = dict(self.outcome)
        state = outcome.get("status")
        if state not in {"succeeded", "unanswerable", "stopped"}:
            raise ResearchWorkflowControllerContractError(
                "ResearchTaskClosedResult outcome.status 必须为 succeeded、unanswerable 或 stopped"
            )
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "status", state)


@dataclass(frozen=True)
class ResearchTaskFailedResult:
    """Task 已由 Working Memory 原子写入 ModuleRunFailed/ResearchTaskFailedResult。"""

    research_task_id: str
    outcome: Mapping[str, Any]
    status: str = field(init=False, default="Failed")

    def __post_init__(self) -> None:
        outcome = dict(self.outcome)
        if outcome.get("status") != "failed":
            raise ResearchWorkflowControllerContractError(
                "ResearchTaskFailedResult outcome.status 必须为 failed"
            )
        object.__setattr__(self, "outcome", outcome)


ControllerResult: TypeAlias = (
    ControllerAdvanced
    | ControllerAwaitingResearcherConfirmation
    | ControllerConfirmationReplayed
    | ControllerSuspended
    | ControllerLeaseLostResult
    | ControllerFailed
    | ResearchTaskClosedResult
    | ResearchTaskFailedResult
)


class ModuleInvocationRunner(Protocol):
    """公开模块 adapter；实现者不应暴露内部身份。"""

    def start(self, request: ModuleInvocationRequest) -> ModuleInvocationOutcome:
        """启动一次新的 Module Invocation。"""

    def resume(self, request: ModuleInvocationRequest) -> ModuleInvocationOutcome:
        """恢复同一个 Module Invocation。"""


class HarnessModuleRunner:
    """将 Investigation Harness facade 适配到 Controller 模块接口。"""

    def __init__(self, harness: InvestigationHarness) -> None:
        self.harness = harness

    @staticmethod
    def _request(request: ModuleInvocationRequest) -> InvestigationHarnessRequest:
        """转换最小公开请求，不把额外字段交给 Harness。"""

        binding = request.context_binding
        if not isinstance(binding, (CreateInvestigationContext, ReuseInvestigationContext)):
            raise ResearchWorkflowControllerContractError(
                "hypothesis-investigation 必须绑定 Investigation Context"
            )
        return InvestigationHarnessRequest(
            research_task_id=request.research_task_id,
            module_invocation_id=request.module_invocation_id,
            module_kind="hypothesis-investigation",
            input_scientific_record_ids=request.input_scientific_record_ids,
            module_input_digest=request.module_input_digest,
            context_binding=binding.to_dict(),
        )

    @staticmethod
    def _normalize(outcome: Any) -> ModuleInvocationOutcome:
        """将 Harness 的三种公开结果转换为 Controller 结果。"""

        if isinstance(outcome, HarnessCompleted):
            return ModuleCompleted(
                outcome.prepared_output,
                recovery_verification=outcome.recovery_verification,
            )
        if isinstance(outcome, HarnessSuspended):
            return ModuleSuspended(
                outcome.reason_code,
                recovery_verification=outcome.recovery_verification,
            )
        if isinstance(outcome, HarnessFatalFailure):
            return ModuleFailed(
                outcome.failure_code,
                recovery_verification=outcome.recovery_verification,
            )
        if isinstance(outcome, PreparedModuleInvocationOutput):
            return ModuleCompleted(outcome)
        if isinstance(outcome, BackendCompleted):
            return ModuleCompleted(
                outcome.prepared_output,
                recovery_verification=outcome.recovery_verification,
            )
        if isinstance(outcome, BackendSuspended):
            return ModuleSuspended(
                outcome.reason_code,
                recovery_verification=outcome.recovery_verification,
            )
        if isinstance(outcome, BackendFatalFailure):
            return ModuleFailed(
                outcome.failure_code,
                recovery_verification=outcome.recovery_verification,
            )
        raise ResearchWorkflowControllerContractError("Harness 返回了未注册结果")

    def start(self, request: ModuleInvocationRequest) -> ModuleInvocationOutcome:
        """启动 Investigation Harness。"""

        try:
            return self._normalize(self.harness.start(self._request(request)))
        except InvestigationHarnessIdentityConflict:
            return ModuleFailed(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="harness_identity_conflict"
                ),
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            return ModuleSuspended(
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_recovery_evidence_missing"
                ),
            )

    def resume(self, request: ModuleInvocationRequest) -> ModuleInvocationOutcome:
        """按 typed context_binding 恢复原 Investigation invocation。"""

        try:
            return self._normalize(self.harness.resume(self._request(request)))
        except InvestigationHarnessIdentityConflict:
            return ModuleFailed(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="harness_identity_conflict"
                ),
            )
        except InvestigationHarnessRecoveryEvidenceUnavailable:
            return ModuleSuspended(
                "recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="harness_recovery_evidence_missing"
                ),
            )


class ResearchWorkflowController:
    """只编排 V2 模块边界的确定性 Controller。

    ``run`` 和 ``resume`` 使用同一主循环：它们都先读取 Task Facts，再决定是
    恢复唯一 Open Module Run、短路已闭合模块，还是创建确定的下一个模块。模块
    adapter 负责科学动作；Controller 不推断模块内部下一步。
    """

    def __init__(
        self,
        working_memory: ResearchTaskWorkingMemory,
        modules: Mapping[str, ModuleInvocationRunner],
        *,
        lease: Any | None = None,
        lease_guard: Any | None = None,
        require_entry_strategy: bool = False,
        material_store: ResearchMaterialStore | None = None,
    ) -> None:
        """绑定 V2 Working Memory、公开模块 adapter 和可选 lease guard。

        ``lease_guard`` 只需要提供 ``assert_valid()``；Controller 不负责把
        lease 写入 Working Memory。``lease`` 是兼容性别名，二者不能同时传入。
        未提供 guard 时保持 Ticket04 的无 lease 行为。
        """

        if not isinstance(working_memory, ResearchTaskWorkingMemory):
            raise ResearchWorkflowControllerContractError(
                "working_memory 必须是 ResearchTaskWorkingMemory"
            )
        self.working_memory = working_memory
        self.modules = {
            normalize_module_kind(kind): runner for kind, runner in modules.items()
        }
        if not isinstance(require_entry_strategy, bool):
            raise ResearchWorkflowControllerContractError(
                "require_entry_strategy 必须是 bool"
            )
        # 直接使用 V2 seam 的历史测试允许省略策略并采用标准链；生产装配打开
        # 严格模式，TaskCreated 缺失或漂移策略立即 fail closed。
        self.require_entry_strategy = require_entry_strategy
        if lease is not None and lease_guard is not None:
            raise ResearchWorkflowControllerContractError(
                "lease 和 lease_guard 只能提供一个"
            )
        self.lease_guard = lease if lease is not None else lease_guard
        self.confirmation_service = HypothesisConfirmationService(
            self.working_memory,
            material_store,
            commit_guard=self._assert_lease,
        )
        self.last_result: ControllerResult | None = None

    def _assert_lease(self) -> None:
        """在 Controller 发起新副作用前确认 lease 仍由当前 holder 持有。

        这是本地 check-before-use gate；它不能把已经进入下游 adapter 的调用
        原子撤销。若下游需要强 fencing，必须继续核对同一个 token。
        """

        if self.lease_guard is None:
            return
        assertion = getattr(self.lease_guard, "assert_valid", None)
        if not callable(assertion):
            assertion = getattr(self.lease_guard, "assert_fencing_token", None)
        if not callable(assertion):
            raise ResearchWorkflowControllerContractError(
                "lease_guard 必须提供 assert_valid()"
            )
        if assertion() is False:
            raise ControllerLeaseLostError("lease guard 报告当前 fencing token 已失效")

    def create_research_task(
        self,
        research_task_id: str,
        *,
        contract_version: str | int,
        input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> ResearchTaskFacts:
        """通过 V2 Working Memory 创建或精确重放 TaskCreated。"""

        self._assert_lease()
        return self.working_memory.create_research_task(
            research_task_id=research_task_id,
            contract_version=contract_version,
            input=input,
            policy=policy,
            dependencies=dependencies,
        )

    def run_research_task(
        self,
        research_task_id: str,
        **task_kwargs: Any,
    ) -> ControllerResult:
        """创建 research_task 后进入与恢复相同的模块主循环。"""

        self.create_research_task(research_task_id, **task_kwargs)
        return self._run_research_task(research_task_id)

    def resume_research_task(self, research_task_id: str) -> ControllerResult:
        """从同一个 research_task Facts 入口恢复任务。"""

        return self._run_research_task(research_task_id)

    def get_hypothesis_confirmation(
        self,
        research_task_id: str,
    ) -> HypothesisConfirmationView:
        """读取当前 Open Confirmation 的稳定研究员视图。"""

        return self.confirmation_service.read_view(
            _require_text(research_task_id, "research_task_id")
        )

    def submit_hypothesis_confirmation_and_continue(
        self,
        research_task_id: str,
        *,
        confirmation_id: str,
        expected_research_task_sequence: int,
        decisions: Sequence[Mapping[str, Any]],
        files: Mapping[str, ResearchMaterialUpload] | None = None,
    ) -> ControllerResult:
        """提交完整研究员决定，并从同一任务继续 Investigation。

        Confirmation 服务先通过 Working Memory 原子闭合 Open Module Run；只有
        提交成功后 Controller 才会进入主循环并允许创建 Investigation Context。
        同一 confirmation_id 的精确重放不会再写入确认事实。
        """

        research_task_id = _require_text(research_task_id, "research_task_id")
        try:
            self._assert_lease()
            submission = self.confirmation_service.confirm(
                research_task_id,
                confirmation_id=confirmation_id,
                expected_research_task_sequence=expected_research_task_sequence,
                decisions=decisions,
                files=files,
            )
        except ControllerLeaseLostError:
            # 与 run() 的失锁语义一致：不写科学失败，只返回当前 Open owner。
            facts = self.working_memory.read_research_task_facts(research_task_id)
            result = ControllerLeaseLostResult(
                research_task_id,
                (
                    None
                    if facts.open_module_invocation is None
                    else facts.open_module_invocation.module_invocation_id
                ),
            )
            self.last_result = result
            return result
        if submission.replayed:
            facts = self.working_memory.read_research_task_facts(research_task_id)
            if facts.lifecycle.state == "Closed":
                # 终态读取不会调用模块；返回既有任务事实以便调用方获得稳定结果。
                return self._run_research_task(research_task_id)
            if facts.open_module_invocation is not None:
                return ControllerConfirmationReplayed(
                    research_task_id=research_task_id,
                    confirmation_id=submission.confirmation_id,
                )
        return self._run_research_task(research_task_id)

    def _run_research_task(self, research_task_id: str) -> ControllerResult:
        """读取 Task Facts 并持续推进模块，直到暂停、失败或任务终态。

        每个模块边界先写入 Working Memory；因此进程可以在任意边界退出，下一次
        调用仍从同一 Task Facts 恢复。已闭合的模块只读取输出，不会重新调用
        adapter。
        """

        research_task_id = _require_text(research_task_id, "research_task_id")
        try:
            facts = self.working_memory.read_research_task_facts(research_task_id)
            if facts.lifecycle.state == "Closed":
                if facts.lifecycle.outcome.get("status") == "failed":
                    result = ResearchTaskFailedResult(research_task_id, facts.lifecycle.outcome)
                else:
                    result = ResearchTaskClosedResult(research_task_id, facts.lifecycle.outcome)
                self.last_result = result
                return result
            # 一个公开调用持续推进模块边界，直到模块暂停、已核实失败或任务关闭。
            # 每个边界仍先写入 Working Memory；进程可在任意边界退出并由同一入口恢复。
            while True:
                facts = self.working_memory.read_research_task_facts(research_task_id)
                if facts.lifecycle.state == "Closed":
                    if facts.lifecycle.outcome.get("status") == "failed":
                        result = ResearchTaskFailedResult(research_task_id, facts.lifecycle.outcome)
                    else:
                        result = ResearchTaskClosedResult(research_task_id, facts.lifecycle.outcome)
                    self.last_result = result
                    return result
                if facts.open_module_invocation is not None:
                    result = self._resume_open_module(
                        facts, facts.open_module_invocation
                    )
                else:
                    next_kind = self._select_next_module(facts)
                    if next_kind is None:
                        result = self._close_terminal_task(facts)
                    else:
                        result = self._start_next_module(facts, next_kind)
                if isinstance(result, ControllerAdvanced):
                    continue
                self.last_result = result
                return result
        except ControllerLeaseLostError:
            # 失去 lease 不是科学失败；只返回观察结果，绝不追加失败事实。
            facts = self.working_memory.read_research_task_facts(research_task_id)
            result = ControllerLeaseLostResult(
                research_task_id,
                (
                    None
                    if facts.open_module_invocation is None
                    else facts.open_module_invocation.module_invocation_id
                ),
            )
            self.last_result = result
            return result

    def preview_next_request(self, research_task_id: str) -> ModuleInvocationRequest:
        """返回当前 Task Facts 对应的确定性新模块请求，不写入任何事实。"""

        facts = self.working_memory.read_research_task_facts(
            _require_text(research_task_id, "research_task_id")
        )
        if facts.open_module_invocation is not None:
            return self._request_from_snapshot(
                facts.research_task_id,
                facts.open_module_invocation,
                facts=facts,
            )
        next_kind = self._select_next_module(facts)
        if next_kind is None:
            raise ResearchWorkflowControllerContractError("Task 已无下一个模块")
        return self._build_new_request(facts, next_kind)

    def module_invocation_id_for(
        self,
        research_task_id: str,
        module_kind: str,
        ordinal: int,
    ) -> str:
        """公开返回确定性 invocation 身份，供模块 adapter 准备输出。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        kind = normalize_module_kind(module_kind)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise ResearchWorkflowControllerContractError("ordinal 必须为正整数")
        digest = hashlib.sha256(research_task_id.encode("utf-8")).hexdigest()[:16]
        return f"v2-{kind}-{ordinal}-{digest}"

    def input_scientific_record_ids_for(
        self,
        research_task_id: str,
        module_kind: str,
    ) -> tuple[str, ...]:
        """根据当前科学图返回模块的确定性输入引用。"""

        facts = self.working_memory.read_research_task_facts(
            _require_text(research_task_id, "research_task_id")
        )
        return self._input_refs_for_kind(facts, normalize_module_kind(module_kind))

    def module_input_digest_for(
        self,
        research_task_id: str,
        module_kind: str,
        input_scientific_record_ids: Sequence[str],
    ) -> str:
        """公开计算与 Module Invocation 请求一致的输入摘要。"""

        return build_module_input_digest(
            research_task_id,
            module_kind,
            input_scientific_record_ids,
        )

    def _resume_open_module(
        self,
        facts: ResearchTaskFacts,
        open_module: ModuleInvocationSnapshot,
    ) -> ControllerResult:
        """只用账本中保存的 typed context_binding 恢复唯一 Module Invocation。"""

        if normalize_module_kind(open_module.module_kind) == CONFIRMATION_MODULE_KIND:
            # Confirmation 是研究员输入阶段，不是需要调用 adapter 的暂停状态。
            # 恢复只重建同一个公开候选视图，不创建 Context、Session 或工具 scope。
            return self._awaiting_researcher_confirmation(facts)
        request = self._request_from_snapshot(
            facts.research_task_id,
            open_module,
            facts=facts,
        )
        runner = self._runner_for(request.module_kind)
        self._assert_lease()
        outcome = self._call_runner(runner, "resume", request)
        return self._finish_module(
            facts,
            open_module,
            request,
            outcome,
            expected_research_task_sequence=facts.research_task_sequence,
        )

    def _start_next_module(
        self,
        facts: ResearchTaskFacts,
        module_kind: str,
    ) -> ControllerResult:
        """为下一个模块先写 ModuleInvocationStarted，再调用 adapter。"""

        request = self._build_new_request(facts, module_kind)
        ledger_request = self._ledger_request(request)
        self._assert_lease()
        started = self.working_memory.start_module_invocation(
            facts.research_task_id,
            expected_research_task_sequence=facts.research_task_sequence,
            request=ledger_request,
        )
        # start_module_invocation 的精确幂等重放可能返回已存在 Closed run；此时不调用
        # adapter，直接从新的 Task Facts 读取已保存输出并决定下一个边界。
        if isinstance(started.lifecycle, ModuleInvocationClosed):
            refreshed = self.working_memory.read_research_task_facts(
                facts.research_task_id
            )
            next_kind = self._select_next_module(refreshed)
            if next_kind is None:
                return self._close_terminal_task(refreshed)
            return ControllerAdvanced(
                facts.research_task_id,
                started.module_invocation_id,
                next_kind,
            )
        if request.module_kind == CONFIRMATION_MODULE_KIND:
            refreshed = self.working_memory.read_research_task_facts(
                facts.research_task_id
            )
            return self._awaiting_researcher_confirmation(refreshed)
        runner = self._runner_for(request.module_kind)
        self._assert_lease()
        outcome = self._call_runner(runner, "start", request)
        return self._finish_module(
            facts,
            started,
            request,
            outcome,
            expected_research_task_sequence=facts.research_task_sequence + 1,
        )

    def _awaiting_researcher_confirmation(
        self,
        facts: ResearchTaskFacts,
    ) -> ControllerAwaitingResearcherConfirmation:
        """把当前 Open Confirmation 投影为 Controller 等待结果。"""

        view = self.confirmation_service.read_view(facts.research_task_id)
        return ControllerAwaitingResearcherConfirmation(
            research_task_id=facts.research_task_id,
            view=view,
        )

    def _finish_module(
        self,
        facts: ResearchTaskFacts,
        module: ModuleInvocationSnapshot,
        request: ModuleInvocationRequest,
        outcome: ModuleInvocationOutcome,
        *,
        expected_research_task_sequence: int,
    ) -> ControllerResult:
        """处理公开模块结果并原子提交 Prepared Output。"""

        if isinstance(outcome, ModuleSuspended):
            return ControllerSuspended(
                facts.research_task_id,
                module.module_invocation_id,
                outcome.reason_code,
                recovery_verification=outcome.recovery_verification,
            )
        if isinstance(outcome, ModuleFailed):
            self._assert_lease()
            failed = self.working_memory.fail_research_task(
                facts.research_task_id,
                expected_research_task_sequence=expected_research_task_sequence,
                module_invocation_id=module.module_invocation_id,
                failure_code=outcome.failure_code,
            )
            return ResearchTaskFailedResult(facts.research_task_id, failed.lifecycle.outcome)
        if not isinstance(outcome, ModuleCompleted):
            raise ResearchWorkflowControllerContractError("模块返回了未注册结果")
        prepared = outcome.prepared_output
        if prepared.module_invocation_id != request.module_invocation_id:
            raise ResearchWorkflowControllerIntegrityConflict(
                "模块输出 module_invocation_id 与 Module Invocation 不一致"
            )
        if prepared.module_input_digest != request.module_input_digest:
            raise ResearchWorkflowControllerIntegrityConflict(
                "模块输出 module_input_digest 与 Module Invocation 不一致"
            )
        try:
            self._assert_lease()
            self.working_memory.complete_module_invocation(
                facts.research_task_id,
                expected_research_task_sequence=expected_research_task_sequence,
                output_bundle=prepared,
            )
        except IdempotencyConflict as exc:
            raise ResearchWorkflowControllerIntegrityConflict(
                "模块输出与已保存完成事实发生漂移"
            ) from exc
        refreshed = self.working_memory.read_research_task_facts(
            facts.research_task_id
        )
        next_kind = self._select_next_module(refreshed)
        if next_kind is None:
            return self._close_terminal_task(refreshed)
        return ControllerAdvanced(
            facts.research_task_id,
            module.module_invocation_id,
            next_kind,
        )

    @staticmethod
    def _call_runner(
        runner: ModuleInvocationRunner,
        operation: str,
        request: ModuleInvocationRequest,
    ) -> ModuleInvocationOutcome:
        """调用模块的公开 start/resume seam。"""

        method = getattr(runner, operation, None)
        if not callable(method):
            raise ResearchWorkflowControllerContractError(
                f"模块 {request.module_kind} 缺少 {operation} 接口"
            )
        result = method(request)
        if isinstance(result, (ModuleCompleted, ModuleSuspended, ModuleFailed)):
            return result
        if isinstance(result, PreparedModuleInvocationOutput):
            return ModuleCompleted(result)
        raise ResearchWorkflowControllerContractError(
            f"模块 {request.module_kind} 返回了未注册结果"
        )

    def _runner_for(self, module_kind: str) -> ModuleInvocationRunner:
        """按公开模块类型取得 adapter。"""

        try:
            return self.modules[module_kind]
        except KeyError as exc:
            raise ResearchWorkflowControllerContractError(
                f"未注册模块 adapter：{module_kind}"
            ) from exc

    @staticmethod
    def _ledger_request(request: ModuleInvocationRequest) -> LedgerModuleInvocationRequest:
        """转换到 Working Memory 的 canonical Module Invocation 请求。"""

        return LedgerModuleInvocationRequest(
            research_task_id=request.research_task_id,
            module_invocation_id=request.module_invocation_id,
            module_kind=request.module_kind,
            input_scientific_record_ids=request.input_scientific_record_ids,
            module_input_digest=request.module_input_digest,
            context_binding=request.context_binding,
        )

    def _request_from_snapshot(
        self,
        research_task_id: str,
        snapshot: ModuleInvocationSnapshot,
        *,
        facts: ResearchTaskFacts | None = None,
    ) -> ModuleInvocationRequest:
        """从账本快照还原请求，并保留已冻结的 typed context_binding。"""

        if facts is not None and facts.research_task_id != research_task_id:
            raise ResearchWorkflowControllerIntegrityConflict(
                "Module Invocation 快照所属 research_task 与当前任务不一致"
            )
        return ModuleInvocationRequest(
            research_task_id=research_task_id,
            module_invocation_id=snapshot.module_invocation_id,
            module_kind=snapshot.module_kind,
            input_scientific_record_ids=snapshot.input_scientific_record_ids,
            module_input_digest=snapshot.module_input_digest,
            context_binding=snapshot.context_binding,
        )

    def _build_new_request(
        self, facts: ResearchTaskFacts, module_kind: str
    ) -> ModuleInvocationRequest:
        """依据科学图构造新 invocation，不读取 Harness 私有状态。"""

        kind = normalize_module_kind(module_kind)
        refs = self._input_refs_for_kind(facts, kind)
        ordinal = 1 + sum(
            1
            for module in facts.module_invocations
            if normalize_module_kind(module.module_kind) == kind
        )
        module_invocation_id = self.module_invocation_id_for(
            facts.research_task_id, kind, ordinal
        )
        digest = build_module_input_digest(facts.research_task_id, kind, refs)
        context_binding = self._context_binding_for_new_module(facts, kind, ordinal)
        return ModuleInvocationRequest(
            research_task_id=facts.research_task_id,
            module_invocation_id=module_invocation_id,
            module_kind=kind,
            input_scientific_record_ids=refs,
            module_input_digest=digest,
            context_binding=context_binding,
        )

    def _context_binding_for_new_module(
        self,
        facts: ResearchTaskFacts,
        module_kind: str,
        ordinal: int,
    ) -> ContextBinding:
        """为新 Invocation 生成带 owner 的 V2 context binding。"""

        if module_kind not in {"hypothesis-investigation", "task-review"}:
            # 无状态模块仍需显式声明 owner 类型。
            return StatelessModule()
        if module_kind == "task-review":
            previous_review = self._latest_closed_module(
                facts.module_invocations,
                "task-review",
            )
            if previous_review is not None:
                binding = previous_review.context_binding
                if not isinstance(
                    binding,
                    (
                        CreateResearchTaskReviewSession,
                        ReuseResearchTaskReviewSession,
                    ),
                ):
                    raise ResearchWorkflowControllerIntegrityConflict(
                        "历史 Review 缺少 typed ResearchTaskReviewSession owner"
                    )
                return ReuseResearchTaskReviewSession(
                    research_task_review_session_id=(
                        binding.research_task_review_session_id
                    ),
                    source_module_invocation_id=previous_review.module_invocation_id,
                )
            return CreateResearchTaskReviewSession(
                research_task_review_session_id=self._context_id(
                    facts.research_task_id,
                    module_kind,
                    ordinal,
                ),
            )
        batch_refs = facts.scientific_graph.record_refs_by_type.get(
            "HypothesisBatchRecord",
            (),
        )
        if not batch_refs:
            raise ResearchWorkflowControllerIntegrityConflict(
                "Investigation Context 缺少 HypothesisBatchRecord"
            )
        review = self._latest_review(facts.research_task_id, facts.module_invocations)
        repair_records = (
            []
            if review is None
            else [
                record
                for record in review["records"]
                if record.scientific_record_type == "RepairRequestRecord"
            ]
        )
        if repair_records:
            source: ModuleInvocationSnapshot | None = None
            for repair_record in repair_records:
                result_ref = repair_record.payload["result_ref"]
                candidate = next(
                    (
                        item
                        for item in facts.module_invocations
                        if item.module_invocation_id
                        == self._producer_module_id(facts, result_ref)
                    ),
                    None,
                )
                if candidate is None or not isinstance(
                    candidate.lifecycle, ModuleInvocationClosed
                ):
                    raise ResearchWorkflowControllerIntegrityConflict(
                        "RepairRequestRecord 的原 Investigation Result 没有闭合来源"
                    )
                if normalize_module_kind(candidate.module_kind) != "hypothesis-investigation":
                    raise ResearchWorkflowControllerIntegrityConflict(
                        "RepairRequestRecord 的 Result 来源不是 Investigation Module Invocation"
                    )
                if source is None:
                    source = candidate
                elif source.module_invocation_id != candidate.module_invocation_id:
                    raise ResearchWorkflowControllerIntegrityConflict(
                        "同一 Repair Investigation 的多个 Result 必须来自同一 Context"
                    )
            if source is None:  # pragma: no cover - repair_records 已非空
                raise ResearchWorkflowControllerIntegrityConflict(
                    "RepairRequestRecord 缺少可复用的 Investigation 来源"
                )
            binding = source.context_binding
            if not isinstance(
                binding,
                (CreateInvestigationContext, ReuseInvestigationContext),
            ):
                raise ResearchWorkflowControllerIntegrityConflict(
                    "原 Investigation Module Invocation 缺少 typed Context owner"
                )
            return ReuseInvestigationContext(
                investigation_context_id=binding.investigation_context_id,
                hypothesis_batch_record_id=batch_refs[-1],
                source_module_invocation_id=source.module_invocation_id,
            )
        return CreateInvestigationContext(
            investigation_context_id=self._context_id(
                facts.research_task_id,
                module_kind,
                ordinal,
            ),
            hypothesis_batch_record_id=batch_refs[-1],
        )

    @staticmethod
    def _context_id(
        research_task_id: str,
        module_kind: str,
        ordinal: int,
    ) -> str:
        """为长期 typed context owner 生成稳定不透明引用。"""

        body = f"{research_task_id}:{module_kind}:{ordinal}:context"
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]
        return f"context-{module_kind}-{digest}"

    @staticmethod
    def _latest_closed_module(
        modules: Sequence[ModuleInvocationSnapshot],
        module_kind: str,
    ) -> ModuleInvocationSnapshot | None:
        """返回指定类型最近一次已闭合模块。"""

        kind = normalize_module_kind(module_kind)
        for module in reversed(modules):
            if normalize_module_kind(module.module_kind) == kind and isinstance(
                module.lifecycle, ModuleInvocationClosed
            ):
                return module
        return None

    def _producer_module_id(
        self, facts: ResearchTaskFacts, record_ref: str
    ) -> str:
        """从科学记录读取其公开 producer Module Run 身份。"""

        record = next(
            (
                item
                for item in facts.scientific_graph.record_refs_by_type.values()
                if record_ref in item
            ),
            None,
        )
        if record is None:
            raise ResearchWorkflowControllerIntegrityConflict(
                f"科学图缺少记录 {record_ref}"
            )
        scientific = next(
            (
                item
                for item in self.working_memory.read_research_task_scientific_records(
                    facts.research_task_id,
                    (record_ref,),
                )
                if item.scientific_record_id == record_ref
            ),
            None,
        )
        if scientific is None:
            raise ResearchWorkflowControllerIntegrityConflict(
                f"科学记录 {record_ref} 无法读取"
            )
        return scientific.producer_module_invocation_id

    def _input_refs_for_kind(
        self, facts: ResearchTaskFacts, module_kind: str
    ) -> tuple[str, ...]:
        """从已闭合模块输出和科学图计算下一模块输入。"""

        kind = normalize_module_kind(module_kind)
        modules = facts.module_invocations
        if kind == "retrieval":
            return ()
        if kind == "research-query-formulation":
            return self._latest_output_refs(modules, "retrieval")
        if kind == "hypothesis-generation":
            return self._latest_output_refs(modules, "research-query-formulation")
        if kind == "hypothesis-import":
            return ()
        if kind == CONFIRMATION_MODULE_KIND:
            return self._confirmation_candidate_refs(facts)
        if kind == "hypothesis-selection":
            return self._latest_output_refs(modules, "hypothesis-generation")
        if kind == "hypothesis-investigation":
            review = self._latest_review(facts.research_task_id, modules)
            if review is not None:
                review_records = tuple(review["records"])
                repair_refs = tuple(
                    record.scientific_record_id
                    for record in review_records
                    if record.scientific_record_type == "RepairRequestRecord"
                )
                if repair_refs:
                    result_refs = tuple(
                        record.payload["result_ref"]
                        for record in review_records
                        if record.scientific_record_type == "RepairRequestRecord"
                    )
                    return tuple(
                        dict.fromkeys(
                            (review["review_id"], *repair_refs, *result_refs)
                        )
                    )
            if self._confirmation_enabled(facts):
                input_refs = self._latest_confirmation_execution_input_refs(facts)
                if input_refs:
                    return input_refs
            return self._latest_output_refs(modules, "hypothesis-selection")
        if kind == "task-review":
            return self._result_heads(facts)
        raise ResearchWorkflowControllerContractError(f"模块未注册：{kind}")

    def _confirmation_candidate_refs(
        self, facts: ResearchTaskFacts
    ) -> tuple[str, ...]:
        """按稳定 rank 返回当前 Selection/Import 的 Selected 记录引用。"""

        strategy = self._entry_strategy(facts)
        source_kind = (
            "hypothesis-import"
            if strategy == "author_hypothesis_import"
            else "hypothesis-selection"
        )
        source = self._latest_closed_module(facts.module_invocations, source_kind)
        if source is None:
            raise ResearchWorkflowControllerIntegrityConflict(
                "Confirmation 缺少已闭合 Selection/Import 来源"
            )
        records = self.working_memory.read_research_task_scientific_records(
            facts.research_task_id,
            tuple(source.lifecycle.outcome["output_scientific_record_ids"]),
        )
        selected = sorted(
            (
                record
                for record in records
                if record.scientific_record_type == "SelectedHypothesisRecord"
            ),
            key=lambda record: record.payload["rank"],
        )
        if source_kind == "hypothesis-import":
            valid_count = 1 <= len(selected) <= 5
        else:
            valid_count = len(selected) == 5
        if not valid_count:
            raise ResearchWorkflowControllerIntegrityConflict(
                "Confirmation 候选数量不满足标准五条或作者一至五条合同"
            )
        refs = tuple(record.scientific_record_id for record in selected)
        if len(set(refs)) != len(refs):  # pragma: no cover - ledger 已校验
            raise ResearchWorkflowControllerIntegrityConflict(
                "Confirmation 候选记录引用重复"
            )
        return refs

    @staticmethod
    def _latest_output_refs(
        modules: Sequence[ModuleInvocationSnapshot],
        module_kind: str,
    ) -> tuple[str, ...]:
        """读取指定模块最近一次闭合输出引用。"""

        kind = normalize_module_kind(module_kind)
        for module in reversed(modules):
            if normalize_module_kind(module.module_kind) != kind:
                continue
            if not isinstance(module.lifecycle, ModuleInvocationClosed):
                continue
            return tuple(module.lifecycle.outcome["output_scientific_record_ids"])
        return ()

    def _latest_confirmation_execution_input_refs(
        self,
        facts: ResearchTaskFacts,
    ) -> tuple[str, ...]:
        """按研究员确认的执行顺序读取最新一轮 Investigation 输入。"""

        confirmation = self._latest_closed_module(
            facts.module_invocations,
            CONFIRMATION_MODULE_KIND,
        )
        if confirmation is None:
            return ()
        refs = tuple(
            confirmation.lifecycle.outcome["output_scientific_record_ids"]
        )
        records = self.working_memory.read_research_task_scientific_records(
            facts.research_task_id,
            refs,
        )
        input_records = sorted(
            (
                record
                for record in records
                if record.scientific_record_type == "HypothesisExecutionInputRecord"
            ),
            key=lambda record: record.payload["execution_rank"],
        )
        input_refs = tuple(
            record.scientific_record_id for record in input_records
        )
        if not input_refs:
            raise ResearchWorkflowControllerIntegrityConflict(
                "已确认模块缺少 HypothesisExecutionInputRecord"
            )
        return input_refs

    def _result_heads(self, facts: ResearchTaskFacts) -> tuple[str, ...]:
        """取得当前轮次的 Investigation Result heads。

        Result 被 Review/Repair 引用后仍是科学结果；只有新的 Result 后继才消费
        旧 head。task.v2 还要按最新 Confirmation 的执行输入排除重新生成前的
        旧批次结果。
        """

        result_refs = facts.scientific_graph.record_refs_by_type.get(
            "InvestigationResultRecord", ()
        )
        successors = facts.scientific_graph.successor_refs_by_record
        result_ref_set = set(result_refs)
        heads = tuple(
            ref
            for ref in result_refs
            if not result_ref_set.intersection(successors.get(ref, ()))
        )
        if not self._confirmation_enabled(facts):
            return heads
        execution_input_refs = self._latest_confirmation_execution_input_refs(facts)
        if not execution_input_refs:
            raise ResearchWorkflowControllerIntegrityConflict(
                "Review 缺少最新 Confirmation 执行输入"
            )
        expected = set(execution_input_refs)
        result_records = self.working_memory.read_research_task_scientific_records(
            facts.research_task_id,
            heads,
        )
        scoped: list[str] = []
        covered: list[str] = []
        for result in result_records:
            linked = expected.intersection(result.source_scientific_record_ids)
            if not linked:
                continue
            if len(linked) != 1:
                raise ResearchWorkflowControllerIntegrityConflict(
                    "Investigation Result 同时引用多条本轮执行输入"
                )
            scoped.append(result.scientific_record_id)
            covered.extend(linked)
        if (
            len(scoped) != len(execution_input_refs)
            or set(covered) != expected
            or len(covered) != len(execution_input_refs)
        ):
            raise ResearchWorkflowControllerIntegrityConflict(
                "本轮 Investigation Result 未完整覆盖已确认执行输入"
            )
        return tuple(scoped)

    def _latest_review(
        self,
        research_task_id: str,
        modules: Sequence[ModuleInvocationSnapshot],
    ) -> dict[str, Any] | None:
        """读取最近 Review 的科学记录；不读取内部运行身份。"""

        for module in reversed(modules):
            if normalize_module_kind(module.module_kind) != "task-review":
                continue
            if not isinstance(module.lifecycle, ModuleInvocationClosed):
                continue
            refs = tuple(module.lifecycle.outcome["output_scientific_record_ids"])
            records = self.working_memory.read_research_task_scientific_records(
                research_task_id,
                refs,
            )
            review = next(
                (
                    record
                    for record in records
                    if record.scientific_record_type == "ReviewRecord"
                ),
                None,
            )
            if review is None:
                raise ResearchWorkflowControllerIntegrityConflict(
                    "已闭合 Review 缺少 ReviewRecord"
                )
            return {
                "review_id": review.scientific_record_id,
                "records": records,
                "module": module,
            }
        return None

    def _select_next_module(self, facts: ResearchTaskFacts) -> str | None:
        """根据已闭合模块及 Review 科学输出确定下一边界。"""

        modules = facts.module_invocations
        strategy = self._entry_strategy(facts)
        if not modules:
            return {
                "standard_hypothesis_generation": "retrieval",
                "author_hypothesis_import": "hypothesis-import",
            }[strategy]
        last = modules[-1]
        if not isinstance(last.lifecycle, ModuleInvocationClosed):
            raise ResearchWorkflowControllerIntegrityConflict(
                "没有 Open Module Run 时最后模块必须已闭合"
            )
        if last.lifecycle.outcome.get("status") == "failed":
            raise ResearchWorkflowControllerIntegrityConflict(
                "失败 Module Run 必须同时关闭 Task，不能继续模块循环"
            )
        kind = normalize_module_kind(last.module_kind)
        if kind == "retrieval":
            return "research-query-formulation"
        if kind == "research-query-formulation":
            return "hypothesis-generation"
        if kind == "hypothesis-generation":
            return "hypothesis-selection"
        if kind == "hypothesis-import":
            if self._confirmation_enabled(facts):
                return CONFIRMATION_MODULE_KIND
            return "hypothesis-investigation"
        if kind == CONFIRMATION_MODULE_KIND:
            return "hypothesis-investigation"
        if kind == "hypothesis-selection":
            if self._confirmation_enabled(facts):
                return CONFIRMATION_MODULE_KIND
            return "hypothesis-investigation"
        if kind == "hypothesis-investigation":
            return "task-review"
        if kind == "task-review":
            review = self._review_record_for_module(facts, last)
            outcome = review.payload["outcome"]
            if outcome == "RepairRequired":
                return "hypothesis-investigation"
            if outcome == "RegenerateHypotheses":
                if strategy == "author_hypothesis_import":
                    raise ResearchWorkflowControllerIntegrityConflict(
                        "作者假设直入任务不能从 Review 跳转到 Hypothesis Generation"
                    )
                return "hypothesis-generation"
            if outcome in {"Accepted", "Unanswerable", "Stopped"}:
                return None
            raise ResearchWorkflowControllerIntegrityConflict(
                f"Review outcome 未注册：{outcome}"
            )
        raise ResearchWorkflowControllerIntegrityConflict(
            f"Task Facts 含无法连接的模块：{last.module_kind}"
        )

    def _entry_strategy(self, facts: ResearchTaskFacts) -> str:
        """读取冻结的 canonical 入口策略。

        schema-v10 历史任务不进入活动 Controller；旧策略必须由显式只读迁移器
        先转换，活动 policy 只接受两个正式入口策略。
        """

        strategy = facts.research_task.policy.get("entry_strategy")
        if strategy is None and not self.require_entry_strategy:
            return "standard_hypothesis_generation"
        if not isinstance(strategy, str):
            raise ResearchWorkflowControllerIntegrityConflict(
                "TaskCreated.policy.entry_strategy 缺失或未注册"
            )
        if strategy in ENTRY_STRATEGIES:
            return strategy
        raise ResearchWorkflowControllerIntegrityConflict(
            "TaskCreated.policy.entry_strategy 缺失或未注册"
        )

    @staticmethod
    def _confirmation_enabled(facts: ResearchTaskFacts) -> bool:
        """只对明确升级到 task.v2 的新任务插入 Confirmation。

        生产任务统一在 task.v2 的研究员确认边界暂停；历史任务由显式迁移器处理，
        不在活动 Controller 中重新解释旧路由。
        """

        return facts.research_task.contract_version == CONFIRMATION_TASK_CONTRACT_VERSION

    def _review_record_for_module(
        self,
        facts: ResearchTaskFacts,
        module: ModuleInvocationSnapshot,
    ) -> ScientificRecord:
        """读取已闭合 Review 的公开科学记录。"""

        refs = tuple(module.lifecycle.outcome["output_scientific_record_ids"])
        records = self.working_memory.read_research_task_scientific_records(
            facts.research_task_id, refs
        )
        reviews = [
            record
            for record in records
            if record.scientific_record_type == "ReviewRecord"
        ]
        if len(reviews) != 1:
            raise ResearchWorkflowControllerIntegrityConflict(
                "Review Module Run 必须有且只有一个 ReviewRecord"
            )
        return reviews[0]

    def _close_terminal_task(self, facts: ResearchTaskFacts) -> ResearchTaskClosedResult:
        """将终态 Review 的 FinalAnswerRecord 通过 WM 领域接口闭合 Task。"""

        if not facts.module_invocations or normalize_module_kind(facts.module_invocations[-1].module_kind) != "task-review":
            raise ResearchWorkflowControllerIntegrityConflict(
                "Task 没有终态 Review，不能写入 ResearchTaskClosedResult"
            )
        review_module = facts.module_invocations[-1]
        review = self._review_record_for_module(facts, review_module)
        outcome_name = review.payload["outcome"]
        if outcome_name not in {"Accepted", "Unanswerable", "Stopped"}:
            raise ResearchWorkflowControllerIntegrityConflict(
                "非终态 Review 不能关闭 Task"
            )
        records = self.working_memory.read_research_task_scientific_records(
            facts.research_task_id,
            tuple(review_module.lifecycle.outcome["output_scientific_record_ids"]),
        )
        finals = [
            record
            for record in records
            if record.scientific_record_type == "FinalAnswerRecord"
        ]
        if len(finals) != 1:
            raise ResearchWorkflowControllerIntegrityConflict(
                "终态 Review 必须有且只有一个 FinalAnswerRecord"
            )
        task_status = {
            "Accepted": "succeeded",
            "Unanswerable": "unanswerable",
            "Stopped": "stopped",
        }[outcome_name]
        outcome = {
            "status": task_status,
            "review_scientific_record_id": review.scientific_record_id,
            "final_answer_scientific_record_id": finals[0].scientific_record_id,
        }
        self._assert_lease()
        closed = self.working_memory.close_research_task(
            facts.research_task_id,
            expected_research_task_sequence=facts.research_task_sequence,
            outcome=outcome,
        )
        if closed.lifecycle.state != "Closed":
            raise ResearchWorkflowControllerIntegrityConflict(
                "ResearchTaskClosedResult 写入后仍不是 Closed"
            )
        return ResearchTaskClosedResult(facts.research_task_id, closed.lifecycle.outcome)

__all__ = [
    "AUTHOR_MODULE_SEQUENCE",
    "ControllerAdvanced",
    "ControllerAwaitingResearcherConfirmation",
    "ControllerFailed",
    "ControllerLeaseLostResult",
    "ControllerResult",
    "ControllerSuspended",
    "HarnessModuleRunner",
    "ResearchWorkflowControllerContractError",
    "ResearchWorkflowControllerError",
    "ResearchWorkflowControllerIntegrityConflict",
    "ResearchWorkflowController",
    "ENTRY_STRATEGIES",
    "MODULE_SEQUENCE",
    "ModuleCompleted",
    "ModuleFailed",
    "ModuleInvocationOutcome",
    "ModuleInvocationRequest",
    "ModuleInvocationRunner",
    "ModuleSuspended",
    "ResearchTaskClosedResult",
    "build_module_input_digest",
    "normalize_module_kind",
    "CONFIRMATION_MODULE_KIND",
    "CONFIRMATION_REASON_CODE",
    "CONFIRMATION_TASK_CONTRACT_VERSION",
    "HypothesisConfirmationService",
    "HypothesisConfirmationSubmission",
    "HypothesisConfirmationView",
]
