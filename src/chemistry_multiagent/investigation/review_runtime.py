"""``investigation_v2`` 的 Task Review 纵向切片。

这个模块把公开的 V2 Scientific Record 图接到已有的 task-level Review runtime。
Review 只读取 Working Memory V2 的当前 ``InvestigationResultRecord`` heads、它们
引用的 ``VerifiedEvidenceRecord``，以及 ``TaskCreated`` 冻结的 input/policy；它不
读取 schema-v10 WorkingMemoryService，也不把 thread、turn、job、scope 或 sidecar
解释成科学事实。

模块的外部 seam 是 ``ResearchTaskReviewRunner.review_current_result_heads``。它负责
读取并核对公开事实、创建 path-free Evidence snapshot、构造 Review request，并把
provider 的结构化响应转换成可供 V2 ``ReviewRecord`` 使用的公开 outcome。provider
中断和 Evidence 资料暂不可用返回 typed Suspended；已核实的输入、来源或摘要冲突
返回 typed Failed。真实 Review 失败（例如某个 finding 为 ``validity=fail``）仍是
正常的 ``RepairRequired`` outcome，而不是异常。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import copy
import hashlib
import re
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from chemistry_multiagent.harness.base import HarnessError
from chemistry_multiagent.memory.contracts import (
    ContractValidationError,
    MemoryDisabledError,
    canonical_json,
)
from chemistry_multiagent.memory.v2_ledger import (
    HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION,
    PreparedModuleInvocationOutput,
    ScientificRecord,
    WorkingMemoryIntegrityConflict,
)

from .codex_callbacks import EvidenceSnapshotLeaseProtocol, TaskReviewRuntimeProtocol
from .codex_runtime import InvestigationTurnResult
from .contracts import (
    TaskFinalizationResponse,
    TaskReviewResponse,
    validate_task_finalization_response,
    validate_task_review_response,
)
from .evidence import (
    BatchEvidenceSnapshot,
    EvidenceArtifactIntegrityError,
    EvidenceArtifactManifest,
    EvidenceArtifactNotFound,
    EvidenceRegistryCorruption,
    EvidenceRequestError,
    EvidenceSnapshotNotFound,
    ReviewEvidenceStore,
)
from .evidence_mcp_lease import EvidenceMcpLeaseError, EvidenceMcpLeaseStartupError
from .prompts import build_task_finalization_request, build_task_review_request
from .recovery_contract import (
    Consistent,
    RecoveryEvidenceMissing,
    RecoveryVerification,
    TemporarilyUnavailable,
    VerifiedIntegrityConflict,
)


__all__ = [
    "V2ReviewEvidenceInput",
    "V2ReviewCompleted",
    "V2ReviewFailed",
    "V2ReviewOutcome",
    "V2ReviewSuspended",
    "ResearchTaskReviewRunner",
    "V2TaskReviewModuleAdapter",
    "V2ReviewRecordSourceProtocol",
    "V2ReviewSnapshotBuilderProtocol",
    "build_v2_evidence_snapshot_builder",
    "build_v2_task_review_runtime",
    "build_v2_native_task_review_runtime",
    "build_v2_task_review_adapter",
    "build_v2_native_task_review_adapter",
]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_PATH_RE = re.compile(r"(?:^|[\s])(?:\.\.?[\\/]|/|[A-Za-z]:[\\/])")
_DEFAULT_REVIEW_MODEL = "deepseek-v4-pro"
_DEFAULT_CONTEXT_ID = "v2-current-result-heads"
_MAX_REVIEW_ARTIFACTS = 64


def _error_mentions_missing(error: BaseException) -> bool:
    """识别读取 seam 报告的缺失资料，而不是把它误报为摘要冲突。"""

    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "not found",
            "not exist",
            "does not exist",
            "record missing",
            "task missing",
            "不存在",
            "找不到",
        )
    )


class V2ReviewRecordSourceProtocol(Protocol):
    """V2 Review 需要的最小只读 Working Memory seam。"""

    def read_research_task_facts(self, research_task_id: str) -> Any:
        """从 V2 事件日志读取可重建 Task Facts。"""

    def read_research_task_scientific_records(
        self,
        research_task_id: str,
        record_refs: Sequence[str] | None = None,
    ) -> Sequence[ScientificRecord]:
        """从 V2 事件日志读取指定 Scientific Record。"""


class V2ReviewSnapshotBuilderProtocol(Protocol):
    """从公开 V2 Review 输入构造 Evidence snapshot 的 adapter。"""

    def __call__(self, review_input: "V2ReviewEvidenceInput") -> Mapping[str, Any] | Any:
        """返回 path-free snapshot manifest 或 ``BatchEvidenceSnapshot``。"""


class _ReviewPreparationError(Exception):
    """内部准备阶段错误，外层会转换成 typed Review outcome。"""


class _ReviewEvidenceMissing(_ReviewPreparationError):
    """准备 Review 所需的 V2/Evidence 公开资料尚不可读。"""


class _ReviewIntegrityConflict(_ReviewPreparationError):
    """两个公开权威来源的身份、内容或摘要已经确认冲突。"""


class _ReviewTemporarilyUnavailable(_ReviewPreparationError):
    """provider 或只读 Evidence adapter 暂时不可用。"""


def _require_text(value: Any, label: str) -> str:
    """校验不透明公开身份字段，不生成默认身份。"""

    if not isinstance(value, str) or not value.strip():
        raise _ReviewIntegrityConflict(f"{label}_missing")
    normalized = value.strip()
    if _CONTROL_RE.search(normalized):
        raise _ReviewIntegrityConflict(f"{label}_invalid")
    return normalized


def _require_public_ref(value: Any, label: str) -> str:
    """校验不会把路径带进 Review request 的公开引用。"""

    normalized = _require_text(value, label)
    if _PATH_RE.search(normalized) or "/" in normalized or "\\" in normalized:
        raise _ReviewIntegrityConflict(f"{label}_path")
    return normalized


def _copy_mapping(value: Any, label: str) -> dict[str, Any]:
    """复制 JSON object，避免调用方在读取后改写冻结事实。"""

    if not isinstance(value, Mapping):
        raise _ReviewEvidenceMissing(f"{label}_missing")
    try:
        copied = json_round_trip(value)
    except (TypeError, ValueError) as exc:
        raise _ReviewIntegrityConflict(f"{label}_invalid") from exc
    if not isinstance(copied, dict):  # pragma: no cover - object 校验已保证
        raise _ReviewIntegrityConflict(f"{label}_invalid")
    return copied


def json_round_trip(value: Any) -> Any:
    """复制严格 JSON 值；该局部 helper 不读取文件或外部状态。"""

    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("value is not copyable JSON") from exc


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    """同时支持公开 dataclass 和只读 mapping 的字段读取。"""

    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _freeze_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    """冻结 Review builder 输入中的 JSON object。"""

    return MappingProxyType(_copy_mapping(value, label))


@dataclass(frozen=True, slots=True)
class V2ReviewEvidenceInput:
    """由 V2 当前 Result heads 组成的公开 snapshot builder 输入。

    ``results`` 和 ``evidence_by_result`` 只含 V2 ScientificRecord；没有任何
    Investigation 私有运行身份。``artifact_refs_by_result`` 是 Evidence record
    中已经保存的来源引用，builder 必须用它们逐值绑定 reader manifest。
    """

    research_task_id: str
    investigation_context_id: str
    result_refs: tuple[str, ...]
    result_pairs: tuple[tuple[str, str], ...]
    results: tuple[ScientificRecord, ...]
    evidence_by_result: Mapping[str, ScientificRecord]
    artifact_refs_by_result: Mapping[str, tuple[str, ...]]
    task_input: Mapping[str, Any]
    policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        """规范化 identity 和不可变公开映射。"""

        object.__setattr__(
            self,
            "research_task_id",
            _require_text(self.research_task_id, "research_task_id"),
        )
        object.__setattr__(
            self,
            "investigation_context_id",
            _require_text(
                self.investigation_context_id, "investigation_context_id"
            ),
        )
        refs = tuple(_require_public_ref(item, "result_refs[]") for item in self.result_refs)
        if len(refs) != len(set(refs)):
            raise _ReviewIntegrityConflict("result_refs_duplicate")
        object.__setattr__(self, "result_refs", refs)
        pairs = tuple(
            (
                _require_public_ref(
                    pair[0],
                    "result_pairs[].hypothesis_execution_input_record_id",
                ),
                _require_text(pair[1], "result_pairs[].result_version"),
            )
            for pair in self.result_pairs
        )
        if len(pairs) != len(set(pairs)):
            raise _ReviewIntegrityConflict("result_pairs_duplicate")
        object.__setattr__(self, "result_pairs", pairs)
        object.__setattr__(self, "results", tuple(self.results))
        if not 1 <= len(refs) <= 5 or len(self.results) != len(refs):
            raise _ReviewIntegrityConflict("result_heads_count_invalid")
        if tuple(record.scientific_record_id for record in self.results) != refs:
            raise _ReviewIntegrityConflict("result_records_refs_drift")
        if len(pairs) != len(refs):
            raise _ReviewIntegrityConflict("result_pairs_count_invalid")
        object.__setattr__(
            self,
            "evidence_by_result",
            MappingProxyType(dict(self.evidence_by_result)),
        )
        object.__setattr__(
            self,
            "artifact_refs_by_result",
            MappingProxyType(
                {
                    str(key): tuple(value)
                    for key, value in self.artifact_refs_by_result.items()
                }
            ),
        )
        if set(self.evidence_by_result) != set(refs):
            raise _ReviewIntegrityConflict("evidence_by_result_refs_drift")
        if set(self.artifact_refs_by_result) != set(refs):
            raise _ReviewIntegrityConflict("artifact_refs_by_result_refs_drift")
        object.__setattr__(self, "task_input", _freeze_mapping(self.task_input, "task_input"))
        object.__setattr__(self, "policy", _freeze_mapping(self.policy, "policy"))

    def to_public_dict(self) -> dict[str, Any]:
        """返回 builder 可审计的公开副本，不暴露本机路径或私有身份。"""

        return {
            "research_task_id": self.research_task_id,
            "investigation_context_id": self.investigation_context_id,
            "result_refs": list(self.result_refs),
            "result_pairs": [list(item) for item in self.result_pairs],
            "results": [record.to_dict() for record in self.results],
            "evidence_by_result": {
                key: value.to_dict() for key, value in self.evidence_by_result.items()
            },
            "artifact_refs_by_result": {
                key: list(value) for key, value in self.artifact_refs_by_result.items()
            },
            "task_input": _copy_mapping(self.task_input, "task_input"),
            "policy": _copy_mapping(self.policy, "policy"),
        }


@dataclass(frozen=True, slots=True)
class V2ReviewCompleted:
    """Review provider 成功返回的结构化结果。"""

    response: TaskReviewResponse | TaskFinalizationResponse
    turn_result: InvestigationTurnResult
    request: Mapping[str, Any]
    snapshot: Mapping[str, Any]
    result_refs: tuple[str, ...]
    result_refs_by_execution_input: Mapping[str, str] = field(default_factory=dict)
    recovery_verification: RecoveryVerification = field(
        default_factory=lambda: Consistent(reason_code="review_result_ready")
    )
    status: str = field(init=False, default="Completed")

    def __post_init__(self) -> None:
        """冻结公开结果副本并核对成功状态。"""

        if not isinstance(
            self.response,
            (TaskReviewResponse, TaskFinalizationResponse),
        ):
            raise TypeError(
                "response 必须是 TaskReviewResponse 或 TaskFinalizationResponse"
            )
        if not isinstance(self.turn_result, InvestigationTurnResult):
            raise TypeError("turn_result 必须是 InvestigationTurnResult")
        object.__setattr__(self, "request", _copy_mapping(self.request, "request"))
        object.__setattr__(self, "snapshot", _copy_mapping(self.snapshot, "snapshot"))
        object.__setattr__(
            self,
            "result_refs",
            tuple(_require_public_ref(item, "result_refs[]") for item in self.result_refs),
        )
        pair_refs = {
            _require_public_ref(
                execution_input_id,
                "result_refs_by_execution_input.key",
            ): _require_public_ref(
                result_ref,
                "result_refs_by_execution_input.value",
            )
            for execution_input_id, result_ref in dict(
                self.result_refs_by_execution_input
            ).items()
        }
        if len(pair_refs) != len(self.result_refs_by_execution_input):
            raise _ReviewIntegrityConflict("result_refs_by_execution_input_duplicate")
        if pair_refs and set(pair_refs.values()) != set(self.result_refs):
            raise _ReviewIntegrityConflict("result_refs_by_execution_input_drift")
        object.__setattr__(
            self,
            "result_refs_by_execution_input",
            MappingProxyType(pair_refs),
        )

    def _public_findings(self) -> list[dict[str, Any]]:
        """把 provider finding 转成 V2 ReviewRecord 的公开 finding。"""

        if isinstance(self.response, TaskFinalizationResponse):
            return [
                {
                    "hypothesis_execution_input_record_id": (
                        finding.hypothesis_execution_input_record_id
                    ),
                    "summary": finding.summary,
                }
                for finding in self.response.findings
            ]
        findings: list[dict[str, Any]] = []
        for finding in self.response.findings:
            item: dict[str, Any] = {
                "hypothesis_execution_input_record_id": (
                    finding.hypothesis_execution_input_record_id
                ),
                "validity": finding.validity,
            }
            hypothesis_judgment = getattr(finding, "hypothesis_judgment", None)
            repair_advice = getattr(finding, "repair_advice", None)
            if hypothesis_judgment is not None:
                item["hypothesis_judgment"] = hypothesis_judgment
            if repair_advice is not None:
                item["repair_advice"] = repair_advice
            findings.append(item)
        return findings

    def to_public_outcome(self) -> dict[str, Any]:
        """返回可直接写入 V2 ``ReviewRecord`` 的 outcome mapping。

        ``validity=fail`` 是正常的 Review 结论，转换为 ``RepairRequired``；它
        不会通过异常路径跳过 ReviewRecord/RepairRequestRecord。
        """

        findings = self._public_findings()
        if isinstance(self.response, TaskFinalizationResponse):
            return {
                "outcome": (
                    "Accepted"
                    if self.response.answerability == "answerable"
                    else "Unanswerable"
                ),
                "findings": findings,
                "answer": self.response.final_answer,
            }
        if self.response.can_answer_original_question:
            return {
                "outcome": "Accepted",
                "findings": findings,
                "answer": self.response.final_answer,
            }
        if self.response.answerability_explanation is not None:
            return {
                "outcome": "Unanswerable",
                "findings": findings,
                "answer": self.response.answerability_explanation,
            }
        failed_ids = [
            item["hypothesis_execution_input_record_id"]
            for item in findings
            if item["validity"] == "fail"
        ]
        if failed_ids:
            advice = [
                str(item["repair_advice"])
                for item in findings
                if item.get("repair_advice") is not None
            ]
            return {
                "outcome": "RepairRequired",
                "findings": findings,
                "problem": "Review rejected execution inputs: " + ", ".join(failed_ids),
                "instructions": " ".join(advice)
                or "repeat the rejected Investigation execution inputs",
            }
        # 当前所有 finding 都是 pass/refuted，但 provider 不能回答；V2 的公开
        # 路由由 ReviewRecord 表示，不在这里重新推断历史批次预算。
        return {
            "outcome": "RegenerateHypotheses",
            "findings": findings,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """返回不含 provider thread/turn 的稳定公开成功结果。"""

        return {
            "status": self.status,
            **self.to_public_outcome(),
            "result_refs": list(self.result_refs),
            "result_refs_by_execution_input": dict(
                self.result_refs_by_execution_input
            ),
            "recovery_verification": self.recovery_verification.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class V2ReviewSuspended:
    """Review provider 或 Evidence 暂不可用，保留 Open Module Run。"""

    reason_code: str
    recovery_verification: RecoveryVerification
    status: str = field(init=False, default="Suspended")

    def __post_init__(self) -> None:
        """确保暂停结果只能使用两种可恢复核验状态。"""

        object.__setattr__(self, "reason_code", _require_text(self.reason_code, "reason_code"))
        if not self.recovery_verification.must_suspend:
            raise ValueError("Suspended 必须使用 TemporarilyUnavailable 或 RecoveryEvidenceMissing")

    def to_public_dict(self) -> dict[str, Any]:
        """返回可供 Controller 映射的公开暂停结果。"""

        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "recovery_verification": self.recovery_verification.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class V2ReviewFailed:
    """Review 输入或来源摘要冲突，允许写入 verified integrity failure。"""

    failure_code: str
    recovery_verification: VerifiedIntegrityConflict
    status: str = field(init=False, default="Failed")

    def __post_init__(self) -> None:
        """确保失败结果确实绑定 VerifiedIntegrityConflict。"""

        object.__setattr__(self, "failure_code", _require_text(self.failure_code, "failure_code"))
        if not self.recovery_verification.must_fail_integrity:
            raise ValueError("V2 Review Failed 必须绑定 VerifiedIntegrityConflict")

    def to_public_dict(self) -> dict[str, Any]:
        """返回不含异常正文的公开失败结果。"""

        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "recovery_verification": self.recovery_verification.to_dict(),
        }


V2ReviewOutcome: TypeAlias = V2ReviewCompleted | V2ReviewSuspended | V2ReviewFailed


def _payload_digest(payload: Mapping[str, Any]) -> str:
    """独立重算 Scientific Record payload 摘要，避免只信任物化字段。"""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _read_task_facts(source: Any, research_task_id: str) -> Any:
    """只通过 V2 Working Memory 的公开读取 seam 获取 Task Facts。"""

    reader = getattr(source, "read_research_task_facts", None)
    if not callable(reader):
        # 允许窄的只读 Working Memory port 提供同一 canonical 操作名；不把
        # 已移除的 schema-v10 service 或旧 task alias 带回生产入口。
        reader = getattr(source, "get_research_task_facts", None)
    if not callable(reader):
        raise _ReviewIntegrityConflict("v2_working_memory_reader_missing")
    try:
        return reader(research_task_id)
    except (KeyError, LookupError) as exc:
        raise _ReviewEvidenceMissing("v2_task_facts_missing") from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise _ReviewTemporarilyUnavailable("v2_working_memory_temporarily_unavailable") from exc
    except MemoryDisabledError as exc:
        raise _ReviewTemporarilyUnavailable("v2_working_memory_temporarily_unavailable") from exc
    except WorkingMemoryIntegrityConflict as exc:
        raise _ReviewIntegrityConflict("v2_task_facts_integrity_conflict") from exc
    except ContractValidationError as exc:
        if _error_mentions_missing(exc):
            raise _ReviewEvidenceMissing("v2_task_facts_missing") from exc
        raise _ReviewIntegrityConflict("v2_task_facts_integrity_conflict") from exc
    except RuntimeError as exc:
        message = str(exc).casefold()
        if any(marker in message for marker in ("unavailable", "timeout", "timed out")):
            raise _ReviewTemporarilyUnavailable("v2_working_memory_temporarily_unavailable") from exc
        raise _ReviewIntegrityConflict("v2_task_facts_integrity_conflict") from exc


def _read_records(
    source: Any,
    research_task_id: str,
    refs: Sequence[str],
) -> tuple[ScientificRecord, ...]:
    """按指定引用读取 V2 ScientificRecord，并拒绝隐式全库扫描。"""

    reader = getattr(source, "read_research_task_scientific_records", None)
    if not callable(reader):
        raise _ReviewIntegrityConflict("v2_scientific_record_reader_missing")
    try:
        raw_records = reader(research_task_id, tuple(refs))
    except (KeyError, LookupError) as exc:
        raise _ReviewEvidenceMissing("v2_scientific_record_missing") from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise _ReviewTemporarilyUnavailable("v2_working_memory_temporarily_unavailable") from exc
    except MemoryDisabledError as exc:
        raise _ReviewTemporarilyUnavailable("v2_working_memory_temporarily_unavailable") from exc
    except WorkingMemoryIntegrityConflict as exc:
        raise _ReviewIntegrityConflict("v2_scientific_records_integrity_conflict") from exc
    except ContractValidationError as exc:
        if _error_mentions_missing(exc):
            raise _ReviewEvidenceMissing("v2_scientific_record_missing") from exc
        raise _ReviewIntegrityConflict("v2_scientific_records_integrity_conflict") from exc
    except RuntimeError as exc:
        message = str(exc).casefold()
        if any(marker in message for marker in ("unavailable", "timeout", "timed out")):
            raise _ReviewTemporarilyUnavailable("v2_working_memory_temporarily_unavailable") from exc
        raise _ReviewIntegrityConflict("v2_scientific_records_integrity_conflict") from exc
    if isinstance(raw_records, (str, bytes, bytearray)) or not isinstance(
        raw_records, Sequence
    ):
        raise _ReviewIntegrityConflict("v2_scientific_record_reader_invalid")
    records: list[ScientificRecord] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, ScientificRecord):
            raise _ReviewIntegrityConflict(f"v2_record_{index}_invalid")
        records.append(raw)
    if tuple(record.scientific_record_id for record in records) != tuple(refs):
        raise _ReviewIntegrityConflict("v2_scientific_record_refs_drift")
    return tuple(records)


def _read_graph_facts(facts: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """读取 Scientific Graph 的两个公开索引，不读取 schema-v10 projection。"""

    graph = _mapping_value(facts, "scientific_graph")
    if graph is None:
        raise _ReviewIntegrityConflict("v2_scientific_graph_missing")
    record_refs_by_type = _mapping_value(graph, "record_refs_by_type")
    successors = _mapping_value(graph, "successor_refs_by_record")
    if not isinstance(record_refs_by_type, Mapping) or not isinstance(successors, Mapping):
        # ScientificGraphFacts.to_dict() 是另一种公开 mapping 形状。
        graph_dict = graph.to_dict() if callable(getattr(graph, "to_dict", None)) else None
        if isinstance(graph_dict, Mapping):
            record_refs_by_type = graph_dict.get("record_refs_by_type")
            successors = graph_dict.get("successor_refs_by_record")
    if not isinstance(record_refs_by_type, Mapping) or not isinstance(successors, Mapping):
        raise _ReviewIntegrityConflict("v2_scientific_graph_invalid")
    return record_refs_by_type, successors


def _known_graph_record_refs(record_refs_by_type: Mapping[str, Any]) -> set[str]:
    """严格规范 Scientific Graph 的全部公开记录引用。"""

    known_refs: set[str] = set()
    for record_type, raw_refs in record_refs_by_type.items():
        if not isinstance(record_type, str) or not record_type.strip():
            raise _ReviewIntegrityConflict("v2_scientific_graph_record_type_invalid")
        if isinstance(raw_refs, (str, bytes, bytearray)) or not isinstance(
            raw_refs, Sequence
        ):
            raise _ReviewIntegrityConflict("v2_scientific_graph_record_refs_invalid")
        for ref in raw_refs:
            normalized = _require_public_ref(ref, "scientific_graph.record_ref")
            if normalized in known_refs:
                raise _ReviewIntegrityConflict("v2_scientific_graph_record_ref_duplicate")
            known_refs.add(normalized)
    return known_refs


def _extract_frozen_task_context(
    facts: Any,
    *,
    research_task_id: str,
    task_input: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """读取并核对 TaskCreated 冻结的 input、policy、问题和 Context 标识。"""

    task = _mapping_value(facts, "research_task")
    frozen_research_task_id = _mapping_value(task, "research_task_id")
    if frozen_research_task_id != research_task_id:
        raise _ReviewIntegrityConflict("review_task_identity_drift")
    frozen_input = _copy_mapping(_mapping_value(task, "input"), "v2_task_input")
    frozen_policy = _copy_mapping(_mapping_value(task, "policy"), "v2_task_policy")
    if task_input is not None:
        supplied_input = _copy_mapping(task_input, "review_task_input")
        if canonical_json(supplied_input) != canonical_json(frozen_input):
            raise _ReviewIntegrityConflict("review_task_input_drift")
    if policy is not None:
        supplied_policy = _copy_mapping(policy, "review_task_policy")
        if canonical_json(supplied_policy) != canonical_json(frozen_policy):
            raise _ReviewIntegrityConflict("review_task_policy_drift")

    scientific_question = None
    for candidate in (
        frozen_input.get("scientific_question"),
        frozen_input.get("question"),
        frozen_policy.get("scientific_question"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            scientific_question = candidate.strip()
            break
    if scientific_question is None:
        raise _ReviewEvidenceMissing("review_scientific_question_missing")
    raw_context_id = (
        frozen_input.get("investigation_context_id")
        or frozen_input.get("current_investigation_context_id")
        or frozen_policy.get("investigation_context_id")
        or frozen_policy.get("current_investigation_context_id")
        or _DEFAULT_CONTEXT_ID
    )
    investigation_context_id = _require_public_ref(
        raw_context_id, "investigation_context_id"
    )
    return frozen_input, frozen_policy, scientific_question, investigation_context_id


def _frozen_repair_budget(task_input: Mapping[str, Any]) -> int:
    """读取 TaskCreated 冻结的单组合返修上限；早期 V2 任务回落为 3。"""

    task_config = task_input.get("task_config")
    if task_config is None:
        return 3
    if not isinstance(task_config, Mapping):
        raise _ReviewIntegrityConflict("review_task_config_invalid")
    budget = task_config.get(
        "max_investigation_repairs_per_combination",
        3,
    )
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise _ReviewIntegrityConflict("review_repair_budget_invalid")
    return budget


def _requires_finalization(
    source: V2ReviewRecordSourceProtocol,
    *,
    research_task_id: str,
    record_refs_by_type: Mapping[str, Any],
    current_results: Sequence[ScientificRecord],
    task_input: Mapping[str, Any],
) -> bool:
    """按 Result 前驱链计数返修，任一当前结果达上限后改为终局成稿。"""

    raw_refs = record_refs_by_type.get("InvestigationResultRecord", ())
    if isinstance(raw_refs, (str, bytes, bytearray)) or not isinstance(
        raw_refs,
        Sequence,
    ):
        raise _ReviewIntegrityConflict("review_result_graph_invalid")
    result_refs = tuple(
        _require_public_ref(ref, "InvestigationResultRecord.ref")
        for ref in raw_refs
    )
    all_results = _read_records(source, research_task_id, result_refs)
    result_by_ref = {
        record.scientific_record_id: record
        for record in all_results
        if record.scientific_record_type == "InvestigationResultRecord"
    }
    if set(result_by_ref) != set(result_refs):
        raise _ReviewIntegrityConflict("review_result_graph_drift")
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(result_ref: str) -> int:
        if result_ref in depths:
            return depths[result_ref]
        if result_ref in visiting:
            raise _ReviewIntegrityConflict("review_result_predecessor_cycle")
        visiting.add(result_ref)
        predecessors = tuple(
            source_ref
            for source_ref in result_by_ref[result_ref].source_scientific_record_ids
            if source_ref in result_by_ref
        )
        if len(predecessors) > 1:
            raise _ReviewIntegrityConflict("review_result_predecessor_invalid")
        value = 0 if not predecessors else depth(predecessors[0]) + 1
        visiting.remove(result_ref)
        depths[result_ref] = value
        return value

    budget = _frozen_repair_budget(task_input)
    return any(depth(record.scientific_record_id) >= budget for record in current_results)


def _current_result_heads(facts: Any) -> tuple[str, ...]:
    """从 V2 Scientific Graph 精确取得当前 Investigation Result heads。"""

    record_refs_by_type, successors = _read_graph_facts(facts)
    raw_refs = record_refs_by_type.get("InvestigationResultRecord", ())
    if isinstance(raw_refs, (str, bytes, bytearray)) or not isinstance(raw_refs, Sequence):
        raise _ReviewIntegrityConflict("v2_result_heads_invalid")
    result_refs = tuple(_require_public_ref(ref, "InvestigationResultRecord.ref") for ref in raw_refs)
    if len(set(result_refs)) != len(result_refs):
        raise _ReviewIntegrityConflict("v2_result_heads_duplicate")
    if not result_refs:
        raise _ReviewEvidenceMissing("v2_result_heads_missing")
    # ``successor_refs_by_record`` 包含 Evidence、Review 等所有后继；只有一个
    # InvestigationResultRecord 后继会消费旧 Result head。Review 不按历史批次、
    # 作业数量或终态概念重新推断输入，而是按这张公开图逐引用过滤。
    result_ref_set = set(result_refs)
    consumed: set[str] = set()
    for result_ref in result_refs:
        raw_successors = successors.get(result_ref, ())
        if isinstance(raw_successors, (str, bytes, bytearray)) or not isinstance(
            raw_successors, Sequence
        ):
            raise _ReviewIntegrityConflict("v2_result_successors_invalid")
        for successor in raw_successors:
            normalized_successor = _require_public_ref(successor, "result_successor")
            if normalized_successor in result_ref_set:
                consumed.add(result_ref)
    return tuple(ref for ref in result_refs if ref not in consumed)


def _confirmed_execution_input_refs(
    facts: Any,
    working_memory: Any,
    research_task_id: str,
) -> tuple[str, ...]:
    """从最新闭合 Confirmation 读取本轮应执行的 1–5 条输入。

    ``task.v2`` 的 Review 数量由研究员决定事实唯一确定，不能再从入口策略或
    Top 5 候选数推断。该读取只接受一条 Decision 与连续排序的 Input 记录；
    缺失或图关系漂移均按已核实完整性冲突停止。
    """

    raw_modules = _mapping_value(facts, "module_invocations")
    if isinstance(raw_modules, (str, bytes, bytearray)) or not isinstance(
        raw_modules, Sequence
    ):
        raise _ReviewIntegrityConflict("review_confirmation_modules_invalid")
    confirmations: list[Any] = []
    for module in raw_modules:
        module_kind = _mapping_value(module, "module_kind")
        if not isinstance(module_kind, str):
            raise _ReviewIntegrityConflict("review_module_kind_invalid")
        if module_kind.casefold().replace("_", "-") not in {
            "hypothesis-confirmation",
            "confirmation",
        }:
            continue
        lifecycle = _mapping_value(module, "lifecycle")
        state = _mapping_value(lifecycle, "state")
        if state == "Closed":
            confirmations.append(module)
    if not confirmations:
        raise _ReviewIntegrityConflict("review_confirmation_missing")
    confirmation = confirmations[-1]
    lifecycle = _mapping_value(confirmation, "lifecycle")
    outcome = _mapping_value(lifecycle, "outcome")
    raw_output_refs = _mapping_value(outcome, "output_scientific_record_ids")
    if isinstance(raw_output_refs, (str, bytes, bytearray)) or not isinstance(
        raw_output_refs, Sequence
    ):
        raise _ReviewIntegrityConflict("review_confirmation_outputs_invalid")
    output_refs = tuple(
        _require_public_ref(ref, "confirmation.output_record_refs[]")
        for ref in raw_output_refs
    )
    records = _read_records(working_memory, research_task_id, output_refs)
    decisions = [
        record
        for record in records
        if record.scientific_record_type == "HypothesisExecutionDecisionRecord"
    ]
    inputs = [
        record
        for record in records
        if record.scientific_record_type == "HypothesisExecutionInputRecord"
    ]
    if len(decisions) != 1 or not 1 <= len(inputs) <= 5:
        raise _ReviewIntegrityConflict("review_confirmation_cardinality_invalid")
    decision = decisions[0]
    confirmation_v2 = (
        decision.scientific_record_schema_version
        == HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION
    )
    execution_inputs_field = (
        "execution_input_scientific_record_ids"
        if confirmation_v2
        else "execution_input_refs"
    )
    raw_expected_refs = decision.payload.get(execution_inputs_field)
    if isinstance(raw_expected_refs, (str, bytes, bytearray)) or not isinstance(
        raw_expected_refs, Sequence
    ):
        raise _ReviewIntegrityConflict("review_confirmation_input_refs_invalid")
    expected_refs = tuple(
        _require_public_ref(ref, "decision.execution_input_refs[]")
        for ref in raw_expected_refs
    )
    if len(set(expected_refs)) != len(expected_refs) or len(expected_refs) != len(inputs):
        raise _ReviewIntegrityConflict("review_confirmation_input_refs_invalid")
    input_by_ref = {record.scientific_record_id: record for record in inputs}
    if set(input_by_ref) != set(expected_refs):
        raise _ReviewIntegrityConflict("review_confirmation_input_refs_drift")
    if any(
        (
            record.scientific_record_schema_version
            == HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION
        )
        != confirmation_v2
        for record in inputs
    ):
        raise _ReviewIntegrityConflict("review_confirmation_schema_drift")
    ranks = [input_by_ref[ref].payload.get("execution_rank") for ref in expected_refs]
    if ranks != list(range(1, len(expected_refs) + 1)):
        raise _ReviewIntegrityConflict("review_confirmation_execution_ranks_invalid")
    for execution_input in inputs:
        if decision.scientific_record_id not in execution_input.source_scientific_record_ids:
            raise _ReviewIntegrityConflict("review_confirmation_decision_edge_missing")
    return expected_refs


def _scope_result_heads_to_execution_inputs(
    result_records: Sequence[ScientificRecord],
    execution_input_refs: Sequence[str],
) -> tuple[str, ...]:
    """只保留属于最新 Confirmation 决定的当前 Result heads。

    重新生成假设后，旧批次的 Result 仍是不可变科学事实，也可能继续是全图
    head；它们不能进入新一轮 Review。修复后的 Result 会继续直接引用原本的
    ``HypothesisExecutionInputRecord``，因此可用该稳定来源把新旧轮次分开。
    """

    expected = set(execution_input_refs)
    scoped_refs: list[str] = []
    covered_inputs: list[str] = []
    for result in result_records:
        linked = expected.intersection(result.source_scientific_record_ids)
        if not linked:
            # 旧 Confirmation 批次的 Result 必须保留在图中，但不属于本轮。
            continue
        if len(linked) != 1:
            raise _ReviewIntegrityConflict("review_result_execution_input_edge_invalid")
        scoped_refs.append(result.scientific_record_id)
        covered_inputs.extend(linked)
    if len(scoped_refs) != len(execution_input_refs):
        raise _ReviewIntegrityConflict("review_result_heads_count_invalid")
    if (
        set(covered_inputs) != expected
        or len(covered_inputs) != len(execution_input_refs)
    ):
        raise _ReviewIntegrityConflict(
            "review_result_execution_input_coverage_invalid"
        )
    return tuple(scoped_refs)


def _claim_identity_from_mapping(value: Mapping[str, Any]) -> tuple[str, str] | None:
    """读取 Evidence payload 中显式的公开 result pair claim。"""

    if "combination_id" in value:
        raise _ReviewIntegrityConflict(
            "review_result_legacy_identity_field_forbidden"
        )
    execution_input_id = value.get("hypothesis_execution_input_record_id")
    result_version = value.get("result_version")
    if isinstance(execution_input_id, str) and isinstance(result_version, str):
        return execution_input_id, result_version
    pair = value.get("result_pair")
    if isinstance(pair, Mapping):
        return _claim_identity_from_mapping(pair)
    if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes, bytearray)):
        if len(pair) == 2 and all(isinstance(item, str) and item.strip() for item in pair):
            return pair[0], pair[1]
    # ``evidence_ref`` 只是 Result 到 Evidence record 的图边，常见 record ID
    # 本身含冒号；不能从任意引用的字符串形状推断 execution input/version。
    return None


def _derive_result_identity(
    result: ScientificRecord,
    evidence: ScientificRecord,
    provider: Callable[[ScientificRecord, ScientificRecord], tuple[str, str]] | None,
) -> tuple[str, str]:
    """读取公开 pair identity；没有显式 claim 时使用稳定 Record identity。

    V2 首批 ``InvestigationResultRecord`` payload 没有旧 session 字段。fallback
    只把 immutable ``record_id`` 作为 Review pair identity，不读取或推断旧批次
    创建事件、作业数量或历史终态概念。
    """

    claims: list[tuple[str, str]] = []
    verification = evidence.payload.get("verification")
    if isinstance(verification, Mapping):
        claim = _claim_identity_from_mapping(verification)
        if claim is not None:
            claims.append(claim)
    evidence_items = evidence.payload.get("evidence_items")
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if isinstance(item, Mapping):
                claim = _claim_identity_from_mapping(item)
                if claim is not None:
                    claims.append(claim)
    if isinstance(result.payload, Mapping):
        claim = _claim_identity_from_mapping(result.payload)
        if claim is not None:
            claims.append(claim)
    provider_pair: tuple[str, str] | None = None
    if provider is not None:
        try:
            pair = provider(result, evidence)
        except Exception as exc:  # noqa: BLE001 - adapter 错误转换为完整性失败
            raise _ReviewIntegrityConflict("review_result_identity_provider_failed") from exc
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray)) or len(pair) != 2:
            raise _ReviewIntegrityConflict("review_result_identity_invalid")
        provider_pair = (
            _require_public_ref(
                pair[0],
                "hypothesis_execution_input_record_id",
            ),
            _require_text(pair[1], "result_version"),
        )
    if claims and len(set(claims)) != 1:
        raise _ReviewIntegrityConflict("review_result_identity_conflict")
    if provider_pair is not None and claims:
        normalized_claim = (
            _require_public_ref(
                claims[0][0],
                "hypothesis_execution_input_record_id",
            ),
            _require_text(claims[0][1], "result_version"),
        )
        if provider_pair != normalized_claim:
            raise _ReviewIntegrityConflict("review_result_identity_provider_drift")
        return provider_pair
    if provider_pair is not None:
        return provider_pair
    if claims:
        return _require_public_ref(
            claims[0][0],
            "hypothesis_execution_input_record_id",
        ), _require_text(
            claims[0][1], "result_version"
        )
    return (
        _require_public_ref(
            result.scientific_record_id,
            "hypothesis_execution_input_record_id",
        ),
        "v1",
    )


def _validate_record_common(
    record: ScientificRecord,
    *,
    research_task_id: str,
    expected_type: str,
    known_refs: set[str],
) -> None:
    """重算 V2 record digest 并核对 task/source identity。"""

    if record.scientific_record_type != expected_type:
        raise _ReviewIntegrityConflict("review_record_type_invalid")
    if record.research_task_id != research_task_id:
        raise _ReviewIntegrityConflict("review_record_task_drift")
    if not isinstance(record.payload, Mapping) or record.payload_digest != _payload_digest(record.payload):
        raise _ReviewIntegrityConflict("review_record_payload_digest_conflict")
    source_refs = tuple(
        _require_public_ref(ref, "record.source_record_refs[]")
        for ref in record.source_scientific_record_ids
    )
    if len(set(source_refs)) != len(source_refs):
        raise _ReviewIntegrityConflict("review_record_source_refs_duplicate")
    if any(ref not in known_refs for ref in source_refs):
        raise _ReviewEvidenceMissing("review_record_source_missing")


def _artifact_refs_from_evidence(evidence: ScientificRecord) -> tuple[str, ...]:
    """读取 Evidence record 保存的来源 artifact refs，不读取 job/scope。"""

    raw_refs = evidence.payload.get("artifact_refs")
    if isinstance(raw_refs, (str, bytes, bytearray)) or not isinstance(raw_refs, Sequence):
        raise _ReviewIntegrityConflict("review_evidence_artifact_refs_invalid")
    refs = tuple(_require_public_ref(item, "artifact_refs[]") for item in raw_refs)
    if len(set(refs)) != len(refs):
        raise _ReviewIntegrityConflict("review_evidence_artifact_refs_duplicate")
    return refs


def _snapshot_manifest(value: Any) -> dict[str, Any]:
    """把低层 snapshot 对象转换成独立公开 manifest。"""

    if isinstance(value, BatchEvidenceSnapshot):
        value = value.to_public_manifest()
    if not isinstance(value, Mapping):
        raise _ReviewIntegrityConflict("review_evidence_snapshot_invalid")
    try:
        return _copy_mapping(value, "evidence_snapshot")
    except _ReviewPreparationError:
        raise
    except Exception as exc:  # pragma: no cover - 防御不可信 adapter
        raise _ReviewIntegrityConflict("review_evidence_snapshot_invalid") from exc


def _canonicalize_schema_v10_snapshot(value: Any) -> dict[str, Any]:
    """在显式 schema-v10 只读 adapter 边界转换 Evidence snapshot 字段。"""

    manifest = _snapshot_manifest(value)
    canonical_keys = {
        "snapshot_id",
        "research_task_id",
        "investigation_context_id",
        "result_pairs",
    }
    legacy_keys = {"snapshot_id", "task_id", "batch_id", "result_pairs"}
    has_canonical_identity = {
        "research_task_id",
        "investigation_context_id",
    }.issubset(manifest)
    if has_canonical_identity:
        if set(manifest) != canonical_keys:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_invalid")
        canonical = dict(manifest)
    else:
        if set(manifest) != legacy_keys:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_invalid")
        canonical = dict(manifest)
        canonical["research_task_id"] = canonical.pop("task_id")
        canonical["investigation_context_id"] = canonical.pop("batch_id")
    pairs = canonical.get("result_pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes, bytearray)):
        raise _ReviewIntegrityConflict("review_evidence_snapshot_pairs_invalid")
    normalized_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise _ReviewIntegrityConflict("review_evidence_snapshot_pair_invalid")
        pair_value = dict(pair)
        old_id = pair_value.pop("combination_id", None)
        new_id = pair_value.pop("hypothesis_execution_input_record_id", None)
        if old_id is not None and new_id is not None and old_id != new_id:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_pair_identity_drift")
        execution_input_id = new_id if new_id is not None else old_id
        if execution_input_id is None:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_pair_invalid")
        pair_value["hypothesis_execution_input_record_id"] = execution_input_id
        if set(pair_value) != {
            "hypothesis_execution_input_record_id",
            "result_version",
            "artifacts",
        }:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_pair_invalid")
        normalized_pairs.append(pair_value)
    canonical["result_pairs"] = normalized_pairs
    return canonical


def _validate_snapshot_against_records(
    snapshot: Mapping[str, Any],
    review_input: V2ReviewEvidenceInput,
) -> None:
    """核对 snapshot 的 pair/source digest identity 与 V2 Evidence 完全一致。"""

    if (
        snapshot.get("research_task_id") != review_input.research_task_id
        or snapshot.get("investigation_context_id")
        != review_input.investigation_context_id
    ):
        raise _ReviewIntegrityConflict(
            "review_evidence_snapshot_task_or_context_drift"
        )
    raw_pairs = snapshot.get("result_pairs")
    if isinstance(raw_pairs, (str, bytes, bytearray)) or not isinstance(raw_pairs, Sequence):
        raise _ReviewIntegrityConflict("review_evidence_snapshot_pairs_invalid")
    pair_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    seen_artifact_ids: set[str] = set()
    seen_source_refs: set[str] = set()
    artifact_count = 0
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, Mapping):
            raise _ReviewIntegrityConflict("review_evidence_snapshot_pair_invalid")
        try:
            identity = (
                _require_public_ref(
                    raw_pair.get("hypothesis_execution_input_record_id"),
                    "snapshot.hypothesis_execution_input_record_id",
                ),
                _require_text(raw_pair.get("result_version"), "snapshot.result_version"),
            )
        except _ReviewPreparationError:
            raise
        if identity in pair_by_identity:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_pair_duplicate")
        pair_by_identity[identity] = raw_pair
        raw_artifacts = raw_pair.get("artifacts")
        if isinstance(raw_artifacts, (str, bytes, bytearray)) or not isinstance(
            raw_artifacts, Sequence
        ):
            raise _ReviewIntegrityConflict("review_evidence_snapshot_artifacts_invalid")
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, Mapping):
                raise _ReviewIntegrityConflict("review_evidence_snapshot_artifact_invalid")
            artifact_id = _require_public_ref(raw_artifact.get("artifact_id"), "snapshot.artifact_id")
            source_ref = _require_public_ref(
                raw_artifact.get("source_artifact_ref"),
                "snapshot.source_artifact_ref",
            )
            if artifact_id in seen_artifact_ids or source_ref in seen_source_refs:
                raise _ReviewIntegrityConflict("review_evidence_snapshot_artifact_duplicate")
            seen_artifact_ids.add(artifact_id)
            seen_source_refs.add(source_ref)
            artifact_count += 1
            if artifact_count > _MAX_REVIEW_ARTIFACTS:
                raise _ReviewIntegrityConflict("review_evidence_snapshot_artifact_limit")
    expected_pairs = set(review_input.result_pairs)
    if set(pair_by_identity) != expected_pairs:
        raise _ReviewIntegrityConflict("review_evidence_snapshot_pairs_drift")
    pair_to_result_ref = {
        pair: review_input.result_refs[index]
        for index, pair in enumerate(review_input.result_pairs)
    }
    for identity, raw_pair in pair_by_identity.items():
        expected_source_refs = set(
            review_input.artifact_refs_by_result[pair_to_result_ref[identity]]
        )
        raw_artifacts = raw_pair.get("artifacts", ())
        actual_source_refs = {
            raw_artifact.get("source_artifact_ref")
            for raw_artifact in raw_artifacts
            if isinstance(raw_artifact, Mapping)
        }
        if actual_source_refs != expected_source_refs:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_source_refs_drift")


def _normalise_review_response(
    turn_result: Any,
    expected_execution_inputs: set[str],
    *,
    finalization_required: bool,
) -> TaskReviewResponse | TaskFinalizationResponse:
    """验证 Review/终局成稿响应，并要求 finding 精确覆盖当前 pair。"""

    if not isinstance(turn_result, InvestigationTurnResult):
        raise _ReviewIntegrityConflict("review_turn_result_invalid")
    raw_response = turn_result.result
    try:
        response = (
            validate_task_finalization_response(raw_response)
            if finalization_required
            else validate_task_review_response(raw_response)
        )
    except Exception as exc:  # noqa: BLE001 - provider wire 违反合同
        raise _ReviewIntegrityConflict("review_provider_response_invalid") from exc
    finding_ids = tuple(
        finding.hypothesis_execution_input_record_id
        for finding in response.findings
    )
    if set(finding_ids) != expected_execution_inputs or len(finding_ids) != len(
        expected_execution_inputs
    ):
        raise _ReviewIntegrityConflict("review_findings_coverage_drift")
    return response


def _build_manifest_from_value(
    value: Any,
    expected_source_ref: str,
) -> EvidenceArtifactManifest | str:
    """把无路径 manifest 或已登记 artifact ID 转成低层 store 输入。"""

    if isinstance(value, str):
        # 允许调用方只提供 registry 中已经登记的 reader artifact ID；低层
        # store 会重新读取并核对其 source ref、大小、SHA 和媒体类型。
        return _require_public_ref(value, "artifact_id")

    if isinstance(value, EvidenceArtifactManifest):
        manifest = value
    elif isinstance(value, Mapping):
        try:
            manifest = EvidenceArtifactManifest(
                artifact_id=value["artifact_id"],
                source_artifact_ref=value["source_artifact_ref"],
                size=value["size"],
                sha256=value["sha256"],
                media_type=value["media_type"],
            )
        except (KeyError, TypeError) as exc:
            raise _ReviewIntegrityConflict("review_artifact_manifest_invalid") from exc
    else:
        raise _ReviewEvidenceMissing("review_artifact_manifest_missing")
    if manifest.source_artifact_ref != expected_source_ref:
        raise _ReviewIntegrityConflict("review_artifact_source_ref_drift")
    _require_public_ref(manifest.artifact_id, "artifact_id")
    _require_public_ref(manifest.source_artifact_ref, "source_artifact_ref")
    if isinstance(manifest.size, bool) or not isinstance(manifest.size, int) or manifest.size < 0:
        raise _ReviewIntegrityConflict("review_artifact_size_invalid")
    if not isinstance(manifest.sha256, str) or _SHA256_RE.fullmatch(manifest.sha256) is None:
        raise _ReviewIntegrityConflict("review_artifact_digest_invalid")
    if not isinstance(manifest.media_type, str) or not manifest.media_type.strip():
        raise _ReviewIntegrityConflict("review_artifact_media_type_invalid")
    return manifest


def build_v2_evidence_snapshot_builder(
    evidence_store: ReviewEvidenceStore,
    artifact_manifest_provider: Mapping[str, Any]
    | Callable[[ScientificRecord, ScientificRecord], Sequence[Any] | Mapping[str, Any]]
    | None = None,
) -> V2ReviewSnapshotBuilderProtocol:
    """构造只依赖 V2 Evidence record 的低层 snapshot adapter。

    ``artifact_manifest_provider`` 可由调用方注入只读 manifest registry；它不能调用
    MiniChem 或接收路径。未注入时，本模块按 source ref 从持久
    ``ReviewEvidenceStore`` 恢复 manifest。每个来源 ref 必须映射到一个完整
    ``EvidenceArtifactManifest``，或映射到已登记的 opaque reader artifact ID。
    找不到登记时返回 ``RecoveryEvidenceMissing``，持久登记损坏或摘要漂移时返回
    ``VerifiedIntegrityConflict``。
    """

    if not isinstance(evidence_store, ReviewEvidenceStore):
        raise TypeError("evidence_store 必须是 ReviewEvidenceStore")
    if artifact_manifest_provider is not None and not isinstance(
        artifact_manifest_provider, Mapping
    ) and not callable(artifact_manifest_provider):
        raise TypeError("artifact_manifest_provider 必须是 mapping 或可调用对象")

    def build_snapshot(review_input: V2ReviewEvidenceInput) -> Mapping[str, Any]:
        """为每个 V2 Result pair 绑定 snapshot artifact 并调用低层 store。"""

        groups: dict[tuple[str, str], list[EvidenceArtifactManifest | str]] = {}
        for index, result in enumerate(review_input.results):
            evidence = review_input.evidence_by_result[result.scientific_record_id]
            expected_refs = review_input.artifact_refs_by_result[result.scientific_record_id]
            values: Any
            try:
                if artifact_manifest_provider is None:
                    values = [
                        evidence_store.read_source_artifact_manifest(source_ref)
                        for source_ref in expected_refs
                    ]
                elif isinstance(artifact_manifest_provider, Mapping):
                    values = [
                        artifact_manifest_provider.get(source_ref)
                        for source_ref in expected_refs
                    ]
                else:
                    values = artifact_manifest_provider(result, evidence)
            except (TimeoutError, ConnectionError, OSError) as exc:
                raise _ReviewTemporarilyUnavailable(
                    "review_evidence_temporarily_unavailable"
                ) from exc
            except (KeyError, LookupError, EvidenceArtifactNotFound) as exc:
                raise _ReviewEvidenceMissing(
                    "review_artifact_manifest_missing"
                ) from exc
            except (EvidenceArtifactIntegrityError, EvidenceRegistryCorruption) as exc:
                raise _ReviewIntegrityConflict(
                    "review_evidence_snapshot_integrity_conflict"
                ) from exc
            if isinstance(values, Mapping):
                values = [values.get(source_ref) for source_ref in expected_refs]
            if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
                raise _ReviewIntegrityConflict("review_artifact_manifest_provider_invalid")
            if len(values) != len(expected_refs):
                raise _ReviewEvidenceMissing("review_artifact_manifest_missing")
            groups[review_input.result_pairs[index]] = [
                _build_manifest_from_value(value, expected_refs[item_index])
                for item_index, value in enumerate(values)
            ]
        try:
            snapshot = evidence_store.create_batch_evidence_snapshot(
                review_input.research_task_id,
                review_input.investigation_context_id,
                review_input.result_pairs,
                groups,
            )
        except (EvidenceSnapshotNotFound, EvidenceArtifactNotFound) as exc:
            raise _ReviewEvidenceMissing("review_evidence_snapshot_missing") from exc
        except (EvidenceArtifactIntegrityError, EvidenceRegistryCorruption) as exc:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_integrity_conflict") from exc
        except EvidenceRequestError as exc:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_request_invalid") from exc
        except RuntimeError as exc:
            message = str(exc).casefold()
            if any(
                marker in message
                for marker in ("unavailable", "timeout", "timed out", "interrupted")
            ):
                raise _ReviewTemporarilyUnavailable(
                    "review_evidence_temporarily_unavailable"
                ) from exc
            raise _ReviewIntegrityConflict("review_evidence_snapshot_integrity_conflict") from exc
        if not isinstance(snapshot, BatchEvidenceSnapshot):
            raise _ReviewIntegrityConflict("review_evidence_snapshot_result_invalid")
        # ``ReviewEvidenceStore`` 目前是 schema-v10 历史实现；转换只发生在这个
        # 明确的只读 adapter 边界，新 runtime 内部只继续使用正式字段。
        return _canonicalize_schema_v10_snapshot(snapshot.to_public_manifest())

    return build_snapshot


class ResearchTaskReviewRunner:
    """把 V2 Scientific Graph 接到独立 task Review provider 的深模块。

    这个接口只需要 V2 Working Memory、一个只读 Evidence snapshot builder 和已有
    ``TaskReviewRuntimeProtocol``。Review 的 task-level ``review_scope_id`` 仅用于
    调用 provider 所需的协议参数，不会进入科学 request；它也不作为任何 Result
    或 Evidence 事实的来源。
    """

    def __init__(
        self,
        *,
        working_memory: V2ReviewRecordSourceProtocol,
        review_runtime: TaskReviewRuntimeProtocol,
        review_scope_id: str,
        evidence_snapshot_builder: V2ReviewSnapshotBuilderProtocol | None = None,
        evidence_store: ReviewEvidenceStore | None = None,
        artifact_manifest_provider: Mapping[str, Any]
        | Callable[[ScientificRecord, ScientificRecord], Sequence[Any] | Mapping[str, Any]]
        | None = None,
        review_model: str = _DEFAULT_REVIEW_MODEL,
        evidence_reader_tool: Mapping[str, Any] | None = None,
        review_context: Mapping[str, Any] | None = None,
        review_thread_id_provider: Callable[[], str | None] | None = None,
        evidence_lease: EvidenceSnapshotLeaseProtocol | None = None,
        result_identity_provider: Callable[[ScientificRecord, ScientificRecord], tuple[str, str]]
        | None = None,
    ) -> None:
        """保存 V2 Review 公开依赖并执行构造期合同检查。"""

        if not callable(getattr(working_memory, "read_research_task_facts", None)) and not callable(
            getattr(working_memory, "get_research_task_facts", None)
        ):
            raise TypeError(
                "working_memory 必须提供 read_research_task_facts"
            )
        if not callable(
            getattr(working_memory, "read_research_task_scientific_records", None)
        ):
            raise TypeError(
                "working_memory 必须提供 read_research_task_scientific_records"
            )
        if not callable(getattr(review_runtime, "run_task_review_turn", None)):
            raise TypeError("review_runtime 必须提供 run_task_review_turn")
        self.working_memory = working_memory
        self.review_runtime = review_runtime
        self.review_scope_id = _require_text(review_scope_id, "review_scope_id")
        if evidence_snapshot_builder is None:
            if evidence_store is None:
                raise TypeError("必须提供 evidence_snapshot_builder 或 evidence_store")
            evidence_snapshot_builder = build_v2_evidence_snapshot_builder(
                evidence_store,
                artifact_manifest_provider,
            )
        if not callable(evidence_snapshot_builder):
            raise TypeError("evidence_snapshot_builder 必须可调用")
        self.evidence_snapshot_builder = evidence_snapshot_builder
        self.review_model = _require_text(review_model, "review_model")
        if evidence_reader_tool is not None and not isinstance(evidence_reader_tool, Mapping):
            raise TypeError("evidence_reader_tool 必须是 JSON object")
        if review_context is not None and not isinstance(review_context, Mapping):
            raise TypeError("review_context 必须是 JSON object")
        if review_thread_id_provider is not None and not callable(review_thread_id_provider):
            raise TypeError("review_thread_id_provider 必须可调用")
        if evidence_lease is not None and not callable(
            getattr(evidence_lease, "serve_evidence_snapshot", None)
        ):
            raise TypeError("evidence_lease 必须提供 serve_evidence_snapshot")
        if result_identity_provider is not None and not callable(result_identity_provider):
            raise TypeError("result_identity_provider 必须可调用")
        self.evidence_reader_tool = (
            None if evidence_reader_tool is None else _copy_mapping(evidence_reader_tool, "evidence_reader_tool")
        )
        self.review_context = (
            {} if review_context is None else _copy_mapping(review_context, "review_context")
        )
        self.review_thread_id_provider = review_thread_id_provider
        self.evidence_lease = evidence_lease
        self.result_identity_provider = result_identity_provider
        # 一个 runtime factory 只服务一个 research_task；首次 provider 返回的 thread
        # 作为该 research_task Review Session 的唯一可复用身份。它不是科学事实，也不
        # 通过 Working Memory 的旧 session/thread 字段恢复。
        self._review_research_task_id: str | None = None
        self._review_thread_id: str | None = None

    def _prepare_review_input(
        self,
        research_task_id: str,
        *,
        result_refs: Sequence[str] | None,
        task_input: Mapping[str, Any] | None,
        policy: Mapping[str, Any] | None,
        investigation_context_id: str | None,
    ) -> tuple[V2ReviewEvidenceInput, str, bool]:
        """读取 V2 heads、Result/Evidence pair 和冻结任务上下文。"""

        research_task_id = _require_public_ref(
            research_task_id, "research_task_id"
        )
        if self._review_research_task_id is None:
            self._review_research_task_id = research_task_id
        elif self._review_research_task_id != research_task_id:
            raise _ReviewIntegrityConflict("review_task_runtime_reuse_drift")
        facts = _read_task_facts(self.working_memory, research_task_id)
        frozen_input, frozen_policy, scientific_question, frozen_context_id = _extract_frozen_task_context(
            facts,
            research_task_id=research_task_id,
            task_input=task_input,
            policy=policy,
        )
        selected_context_id = (
            frozen_context_id
            if investigation_context_id is None
            else _require_public_ref(
                investigation_context_id, "investigation_context_id"
            )
        )
        if (
            investigation_context_id is not None
            and selected_context_id != frozen_context_id
        ):
            # 显式 Context ID 只可重复冻结 Task 中的值，不能借调用参数引入第二套
            # 旧批次创建事件或旧批次解释。
            raise _ReviewIntegrityConflict("review_investigation_context_id_drift")
        all_current_heads = _current_result_heads(facts)
        task = _mapping_value(facts, "research_task")
        contract_version = _mapping_value(task, "contract_version")
        expected_execution_inputs: tuple[str, ...] = ()
        all_result_records: tuple[ScientificRecord, ...] | None = None
        if contract_version == "investigation_v2.task.v2":
            expected_execution_inputs = _confirmed_execution_input_refs(
                facts,
                self.working_memory,
                research_task_id,
            )
            all_result_records = _read_records(
                self.working_memory,
                research_task_id,
                all_current_heads,
            )
            current_heads = _scope_result_heads_to_execution_inputs(
                all_result_records,
                expected_execution_inputs,
            )
        else:
            current_heads = all_current_heads
        requested_refs = current_heads if result_refs is None else tuple(result_refs)
        if isinstance(requested_refs, (str, bytes, bytearray)) or not isinstance(
            requested_refs, Sequence
        ):
            raise _ReviewIntegrityConflict("review_result_heads_invalid")
        requested_refs = tuple(_require_public_ref(ref, "result_refs[]") for ref in requested_refs)
        if requested_refs != current_heads:
            raise _ReviewIntegrityConflict("review_result_heads_drift")
        if contract_version != "investigation_v2.task.v2":
            # 历史 task.v1 没有 Confirmation Decision，只能继续按冻结入口策略
            # 解释旧结果数量；恢复时不得用当前 v2 规则改写它。
            entry_strategy = frozen_policy.get("entry_strategy")
            if entry_strategy == "standard_retrieval" and len(requested_refs) != 5:
                raise _ReviewIntegrityConflict("review_result_heads_count_invalid")
            if (
                entry_strategy == "author_main_hypothesis_import"
                and not 1 <= len(requested_refs) <= 5
            ):
                raise _ReviewIntegrityConflict("review_result_heads_count_invalid")
            if entry_strategy not in {
                "standard_retrieval",
                "author_main_hypothesis_import",
            }:
                raise _ReviewIntegrityConflict("review_entry_strategy_invalid")

        record_refs_by_type, _successors = _read_graph_facts(facts)
        known_refs = _known_graph_record_refs(record_refs_by_type)
        if all_result_records is not None:
            records_by_ref = {
                record.scientific_record_id: record for record in all_result_records
            }
            result_records = tuple(records_by_ref[ref] for ref in requested_refs)
        else:
            result_records = _read_records(
                self.working_memory,
                research_task_id,
                requested_refs,
            )
        if expected_execution_inputs:
            expected_input_set = set(expected_execution_inputs)
            covered_inputs: list[str] = []
            for result in result_records:
                linked = expected_input_set.intersection(result.source_scientific_record_ids)
                if len(linked) != 1:
                    raise _ReviewIntegrityConflict(
                        "review_result_execution_input_edge_invalid"
                    )
                covered_inputs.extend(linked)
            if set(covered_inputs) != expected_input_set or len(covered_inputs) != len(
                expected_execution_inputs
            ):
                raise _ReviewIntegrityConflict(
                    "review_result_execution_input_coverage_invalid"
                )
        evidence_refs: list[str] = []
        for result in result_records:
            _validate_record_common(
                result,
                research_task_id=research_task_id,
                expected_type="InvestigationResultRecord",
                known_refs=known_refs,
            )
            raw_evidence_ref = result.payload.get("evidence_ref")
            evidence_ref = _require_public_ref(raw_evidence_ref, "result.evidence_ref")
            if evidence_ref not in result.source_scientific_record_ids:
                raise _ReviewIntegrityConflict("result_evidence_source_ref_missing")
            evidence_refs.append(evidence_ref)
        if len(set(evidence_refs)) != len(evidence_refs):
            raise _ReviewIntegrityConflict("result_evidence_ref_duplicate")
        evidence_records = _read_records(
            self.working_memory, research_task_id, tuple(evidence_refs)
        )
        evidence_by_result: dict[str, ScientificRecord] = {}
        artifact_refs_by_result: dict[str, tuple[str, ...]] = {}
        result_pairs: list[tuple[str, str]] = []
        for result, evidence in zip(result_records, evidence_records):
            _validate_record_common(
                evidence,
                research_task_id=research_task_id,
                expected_type="VerifiedEvidenceRecord",
                known_refs=known_refs,
            )
            if evidence.scientific_record_id != result.payload["evidence_ref"]:
                raise _ReviewIntegrityConflict("result_evidence_identity_drift")
            if evidence.producer_module_invocation_id != result.producer_module_invocation_id:
                raise _ReviewIntegrityConflict("result_evidence_producer_drift")
            evidence_by_result[result.scientific_record_id] = evidence
            artifact_refs_by_result[result.scientific_record_id] = _artifact_refs_from_evidence(evidence)
            result_pairs.append(
                _derive_result_identity(
                    result,
                    evidence,
                    self.result_identity_provider,
                )
            )
        if len({pair[0] for pair in result_pairs}) != len(result_pairs):
            raise _ReviewIntegrityConflict("review_execution_input_identity_duplicate")
        review_input = V2ReviewEvidenceInput(
            research_task_id=research_task_id,
            investigation_context_id=selected_context_id,
            result_refs=requested_refs,
            result_pairs=tuple(result_pairs),
            results=result_records,
            evidence_by_result=evidence_by_result,
            artifact_refs_by_result=artifact_refs_by_result,
            task_input=frozen_input,
            policy=frozen_policy,
        )
        finalization_required = _requires_finalization(
            self.working_memory,
            research_task_id=research_task_id,
            record_refs_by_type=record_refs_by_type,
            current_results=result_records,
            task_input=frozen_input,
        )
        return review_input, scientific_question, finalization_required

    def _build_review_request(
        self,
        review_input: V2ReviewEvidenceInput,
        scientific_question: str,
        *,
        finalization_required: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """创建并核对公开 Evidence snapshot 与 Review/终局成稿请求。"""

        try:
            raw_snapshot = self.evidence_snapshot_builder(review_input)
        except _ReviewPreparationError:
            raise
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise _ReviewTemporarilyUnavailable("review_evidence_temporarily_unavailable") from exc
        except (EvidenceMcpLeaseStartupError, EvidenceMcpLeaseError) as exc:
            # 只读 Evidence 连接只在真正执行 Review 时按 snapshot 装配；若
            # builder 需要该连接，启动失败同样只能保留 Open，不伪造 Review。
            if isinstance(exc, EvidenceMcpLeaseStartupError):
                raise _ReviewTemporarilyUnavailable(
                    "review_evidence_mcp_temporarily_unavailable"
                ) from exc
            raise _ReviewIntegrityConflict("review_evidence_mcp_integrity_conflict") from exc
        except (KeyError, LookupError) as exc:
            raise _ReviewEvidenceMissing("review_evidence_snapshot_missing") from exc
        except (EvidenceSnapshotNotFound, EvidenceArtifactNotFound) as exc:
            raise _ReviewEvidenceMissing("review_evidence_snapshot_missing") from exc
        except (EvidenceArtifactIntegrityError, EvidenceRegistryCorruption) as exc:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_integrity_conflict") from exc
        except EvidenceRequestError as exc:
            raise _ReviewIntegrityConflict("review_evidence_snapshot_request_invalid") from exc
        except RuntimeError as exc:
            message = str(exc).casefold()
            if _error_mentions_missing(exc):
                raise _ReviewEvidenceMissing("review_evidence_snapshot_missing") from exc
            if any(
                marker in message
                for marker in ("unavailable", "timeout", "timed out", "interrupted")
            ):
                raise _ReviewTemporarilyUnavailable(
                    "review_evidence_temporarily_unavailable"
                ) from exc
            raise _ReviewIntegrityConflict("review_evidence_snapshot_integrity_conflict") from exc
        # 自定义 builder 也必须在这个明确的 schema-v10/只读边界完成历史字段转换；
        # 下方核对和返回值永远只使用 canonical snapshot。
        snapshot = _canonicalize_schema_v10_snapshot(raw_snapshot)
        _validate_snapshot_against_records(snapshot, review_input)
        normalized_pairs = [
            {
                "hypothesis_execution_input_record_id": execution_input_id,
                "result_version": result_version,
                "artifacts": list(
                    next(
                        pair["artifacts"]
                        for pair in snapshot["result_pairs"]
                        if pair["hypothesis_execution_input_record_id"]
                        == execution_input_id
                        and pair["result_version"] == result_version
                    )
                ),
            }
            for execution_input_id, result_version in review_input.result_pairs
        ]
        current_results: list[dict[str, Any]] = []
        for result, (execution_input_id, result_version) in zip(
            review_input.results,
            review_input.result_pairs,
        ):
            evidence = review_input.evidence_by_result[result.scientific_record_id]
            current_results.append(
                {
                    "research_task_id": review_input.research_task_id,
                    "investigation_context_id": (
                        review_input.investigation_context_id
                    ),
                    "hypothesis_execution_input_record_id": execution_input_id,
                    "result_version": result_version,
                    "self_judgment": result.payload["self_judgment"],
                    "narrative": result.payload["conclusion"],
                    # V2 public Result 不保存或暴露作业/operation 身份；Review
                    # 通过 Evidence snapshot 读取已绑定 artifact 内容。
                    "job_refs": [],
                    "operation_refs": [],
                    "artifact_refs": list(review_input.artifact_refs_by_result[result.scientific_record_id]),
                    "observations": list(result.payload["observations"]),
                    "limitations": list(result.payload["limitations"]),
                }
            )
            if evidence.research_task_id != review_input.research_task_id:  # 防御性重复核对
                raise _ReviewIntegrityConflict("review_evidence_task_drift")
        review_thread_id: str | None = None
        if self.review_thread_id_provider is not None:
            try:
                review_thread_id = self.review_thread_id_provider()
            except (TimeoutError, ConnectionError, OSError) as exc:
                raise _ReviewTemporarilyUnavailable("review_thread_temporarily_unavailable") from exc
            if review_thread_id is not None:
                review_thread_id = _require_public_ref(review_thread_id, "review_thread_id")
        if self._review_thread_id is not None:
            if review_thread_id is not None and review_thread_id != self._review_thread_id:
                raise _ReviewIntegrityConflict("review_thread_identity_drift")
            review_thread_id = self._review_thread_id
        try:
            request_builder = (
                build_task_finalization_request
                if finalization_required
                else build_task_review_request
            )
            request = request_builder(
                review_input.research_task_id,
                review_input.investigation_context_id,
                scientific_question,
                current_results,
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "research_task_id": snapshot["research_task_id"],
                    "investigation_context_id": snapshot[
                        "investigation_context_id"
                    ],
                    "result_pairs": normalized_pairs,
                },
                evidence_reader_tool=self.evidence_reader_tool,
                review_context={},
                model=self.review_model,
                review_thread_id=review_thread_id,
            )
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise _ReviewTemporarilyUnavailable("review_request_temporarily_unavailable") from exc
        except (TypeError, ValueError) as exc:
            raise _ReviewIntegrityConflict("review_request_contract_invalid") from exc
        return request, snapshot

    def review_current_result_heads(
        self,
        research_task_id: str,
        *,
        result_refs: Sequence[str] | None = None,
        task_input: Mapping[str, Any] | None = None,
        policy: Mapping[str, Any] | None = None,
        investigation_context_id: str | None = None,
    ) -> V2ReviewOutcome:
        """审查当前 V2 Result heads，并返回完成、暂停或完整性失败。

        输入是 task identity、可选的调用方核对副本和明确 result refs；若省略
        refs，runtime 从 V2 Scientific Graph 读取当前 heads。所有摘要和来源核对
        在 provider 调用前完成；因此冲突不会产生新的 Review turn。provider/MCP
        暂不可用只返回 Suspended，调用方应保留 Open Module Run。
        """

        try:
            (
                review_input,
                scientific_question,
                finalization_required,
            ) = self._prepare_review_input(
                research_task_id,
                result_refs=result_refs,
                task_input=task_input,
                policy=policy,
                investigation_context_id=investigation_context_id,
            )
            request, snapshot = self._build_review_request(
                review_input,
                scientific_question,
                finalization_required=finalization_required,
            )
            snapshot_id = snapshot.get("snapshot_id")
            if not isinstance(snapshot_id, str) or not snapshot_id.strip():
                raise _ReviewIntegrityConflict("review_snapshot_id_missing")
            lease_context = (
                self.evidence_lease.serve_evidence_snapshot(snapshot_id)
                if self.evidence_lease is not None
                else nullcontext()
            )
            try:
                with lease_context:
                    turn_result = self.review_runtime.run_task_review_turn(
                        request,
                        scope_id=self.review_scope_id,
                    )
            except (TimeoutError, ConnectionError, OSError, InterruptedError) as exc:
                raise _ReviewTemporarilyUnavailable("review_provider_temporarily_unavailable") from exc
            except HarnessError as exc:
                # Codex runtime 将 transport/terminal/provider interruption 统一为
                # HarnessError；只有 transport 或明确不可用才可保持 Open，响应合同
                # 和 session identity 错误仍进入完整性失败。
                message = str(exc).casefold()
                if exc.kind == "transport" or any(
                    marker in message
                    for marker in ("unavailable", "interrupted", "timeout", "timed out")
                ):
                    raise _ReviewTemporarilyUnavailable(
                        "review_provider_temporarily_unavailable"
                    ) from exc
                raise _ReviewIntegrityConflict("review_provider_protocol_conflict") from exc
            except EvidenceMcpLeaseStartupError as exc:
                raise _ReviewTemporarilyUnavailable(
                    "review_evidence_mcp_temporarily_unavailable"
                ) from exc
            except EvidenceMcpLeaseError as exc:
                raise _ReviewIntegrityConflict("review_evidence_mcp_integrity_conflict") from exc
            except RuntimeError as exc:
                # 测试替代实现和 provider SDK 的窄 wrapper 可能只暴露
                # RuntimeError；仅按明确的暂不可用措辞保持 Open，其余仍 fail-closed。
                message = str(exc).casefold()
                if any(
                    marker in message
                    for marker in ("unavailable", "interrupted", "timeout", "timed out")
                ):
                    raise _ReviewTemporarilyUnavailable(
                        "review_provider_temporarily_unavailable"
                    ) from exc
                raise _ReviewIntegrityConflict("review_provider_protocol_conflict") from exc
            except (TypeError, ValueError) as exc:
                raise _ReviewIntegrityConflict("review_provider_protocol_conflict") from exc
            response = _normalise_review_response(
                turn_result,
                {pair[0] for pair in review_input.result_pairs},
                finalization_required=finalization_required,
            )
            returned_thread_id = _require_public_ref(
                turn_result.thread_id,
                "review_thread_id",
            )
            if (
                self._review_thread_id is not None
                and returned_thread_id != self._review_thread_id
            ):
                raise _ReviewIntegrityConflict("review_thread_identity_drift")
            self._review_thread_id = returned_thread_id
            return V2ReviewCompleted(
                response=response,
                turn_result=turn_result,
                request=request,
                snapshot=snapshot,
                result_refs=review_input.result_refs,
                result_refs_by_execution_input={
                    pair[0]: review_input.result_refs[index]
                    for index, pair in enumerate(review_input.result_pairs)
                },
            )
        except _ReviewTemporarilyUnavailable as exc:
            return V2ReviewSuspended(
                str(exc) or "review_temporarily_unavailable",
                TemporarilyUnavailable(
                    reason_code=str(exc) or "review_temporarily_unavailable"
                ),
            )
        except _ReviewEvidenceMissing as exc:
            return V2ReviewSuspended(
                str(exc) or "review_evidence_missing",
                RecoveryEvidenceMissing(reason_code=str(exc) or "review_evidence_missing"),
            )
        except _ReviewIntegrityConflict as exc:
            return V2ReviewFailed(
                "verified_integrity_conflict",
                VerifiedIntegrityConflict(
                    reason_code=str(exc) or "verified_integrity_conflict"
                ),
            )

    def run_task_review(
        self, research_task_id: str, **kwargs: Any
    ) -> V2ReviewOutcome:
        """公开语义别名，便于 V2 adapter 一行装配。"""

        return self.review_current_result_heads(research_task_id, **kwargs)

    def start(self, request: Any) -> V2ReviewOutcome:
        """把 ModuleInvocationRequest 的公开字段接到独立 Review seam。"""

        research_task_id = _mapping_value(request, "research_task_id")
        result_refs = _mapping_value(request, "input_scientific_record_ids")
        if result_refs is None or isinstance(result_refs, (str, bytes, bytearray)):
            raise TypeError(
                "Review ModuleInvocationRequest 缺少 input_scientific_record_ids"
            )
        context_binding = _mapping_value(request, "context_binding", {})
        investigation_context_id = _mapping_value(
            context_binding, "investigation_context_id"
        )
        return self.review_current_result_heads(
            research_task_id,
            result_refs=tuple(result_refs),
            investigation_context_id=investigation_context_id,
        )

    def resume(self, request: Any) -> V2ReviewOutcome:
        """恢复同一 V2 Result heads；不读取或重建旧 Review state。"""

        return self.start(request)


def _review_prepared_output(
    completed: V2ReviewCompleted,
    *,
    research_task_id: str,
    module_invocation_id: str,
    module_input_digest: str,
) -> PreparedModuleInvocationOutput:
    """把一次已完成 Review 转成 V2 ledger 的原子公开输出包。

    ReviewRecord 总是先写入；Accepted/Unanswerable 再写唯一
    FinalAnswerRecord，RepairRequired 则为每个 fail finding 写 RepairRequestRecord。
    这些记录只引用当前 Result heads 或同一 bundle 中刚形成的 ReviewRecord，因而
    Controller 可以直接把返回值提交给 V2 Working Memory，不需要旧状态投影。
    """

    if not isinstance(completed, V2ReviewCompleted):
        raise TypeError("completed 必须是 V2ReviewCompleted")
    research_task_id = _require_public_ref(research_task_id, "research_task_id")
    module_invocation_id = _require_public_ref(
        module_invocation_id, "module_invocation_id"
    )
    module_input_digest = _require_text(module_input_digest, "module_input_digest")
    if not completed.result_refs_by_execution_input:
        raise _ReviewIntegrityConflict("review_result_identity_map_missing")
    review_record_ref = _require_public_ref(
        f"{module_invocation_id}:review",
        "review_record_ref",
    )
    public_outcome = completed.to_public_outcome()
    findings = completed._public_findings()
    review_record = ScientificRecord.from_payload(
        scientific_record_id=review_record_ref,
        research_task_id=research_task_id,
        scientific_record_type="ReviewRecord",
        scientific_record_schema_version=1,
        producer_module_invocation_id=module_invocation_id,
        source_scientific_record_ids=completed.result_refs,
        payload={
            "reviewed_result_refs": list(completed.result_refs),
            "findings": findings,
            "outcome": public_outcome["outcome"],
        },
    )
    records: list[ScientificRecord] = [review_record]
    outcome_name = public_outcome["outcome"]
    if outcome_name == "RepairRequired":
        for index, finding in enumerate(findings, start=1):
            if finding.get("validity") != "fail":
                continue
            execution_input_id = _require_public_ref(
                finding.get("hypothesis_execution_input_record_id"),
                "finding.hypothesis_execution_input_record_id",
            )
            result_ref = completed.result_refs_by_execution_input.get(execution_input_id)
            if result_ref is None:
                raise _ReviewIntegrityConflict("review_repair_result_ref_missing")
            advice = finding.get("repair_advice")
            if not isinstance(advice, str) or not advice.strip():
                raise _ReviewIntegrityConflict("review_repair_advice_missing")
            repair_record = ScientificRecord.from_payload(
                scientific_record_id=_require_public_ref(
                    f"{module_invocation_id}:repair:{index}",
                    "repair_record_ref",
                ),
                research_task_id=research_task_id,
                scientific_record_type="RepairRequestRecord",
                scientific_record_schema_version=1,
                producer_module_invocation_id=module_invocation_id,
                source_scientific_record_ids=(review_record_ref, result_ref),
                payload={
                    "result_ref": result_ref,
                    "problem": (
                        "Review rejected hypothesis execution input: "
                        f"{execution_input_id}"
                    ),
                    "instructions": advice,
                },
            )
            records.append(repair_record)
        if len(records) == 1:
            raise _ReviewIntegrityConflict("review_repair_record_missing")
    elif outcome_name in {"Accepted", "Unanswerable"}:
        answer = public_outcome.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise _ReviewIntegrityConflict("review_final_answer_missing")
        records.append(
            ScientificRecord.from_payload(
                scientific_record_id=_require_public_ref(
                    f"{module_invocation_id}:answer",
                    "final_answer_record_ref",
                ),
                research_task_id=research_task_id,
                scientific_record_type="FinalAnswerRecord",
                scientific_record_schema_version=1,
                producer_module_invocation_id=module_invocation_id,
                source_scientific_record_ids=(review_record_ref,),
                payload={
                    "answer": answer,
                    "answerability": (
                        "answerable" if outcome_name == "Accepted" else "unanswerable"
                    ),
                },
            )
        )
    elif outcome_name != "RegenerateHypotheses":
        raise _ReviewIntegrityConflict("review_outcome_unregistered")
    try:
        return PreparedModuleInvocationOutput.from_records(
            module_invocation_id=module_invocation_id,
            module_input_digest=module_input_digest,
            output_scientific_records=records,
        )
    except Exception as exc:  # noqa: BLE001 - 转换为公开完整性失败
        raise _ReviewIntegrityConflict("review_prepared_output_invalid") from exc


class V2TaskReviewModuleAdapter:
    """将 V2 Review runtime 接到 ModuleRunner 的 ``start/resume`` seam。

    该 adapter 只做 Controller 边界转换：科学读取、Evidence snapshot、provider
    调用及 ReviewRecord/RepairRequestRecord 组成均由 ``ResearchTaskReviewRunner`` 负责。
    它不创建 schema-v10 状态对象，也不调用 MiniChem。
    """

    def __init__(self, runtime: ResearchTaskReviewRunner) -> None:
        """保存一个可复用的 task-level Review runtime。"""

        if not isinstance(runtime, ResearchTaskReviewRunner):
            raise TypeError("runtime 必须是 ResearchTaskReviewRunner")
        self.runtime = runtime

    def _run(self, request: Any, operation: str) -> Any:
        """运行 Review，并转换为 Controller 公开模块结果。"""

        from chemistry_multiagent.controllers.investigation_v2_module_controller import (
            ModuleCompleted,
            ModuleFailed,
            ModuleSuspended,
        )

        research_task_id = _mapping_value(request, "research_task_id")
        module_invocation_id = _mapping_value(request, "module_invocation_id")
        module_input_digest = _mapping_value(request, "module_input_digest")
        runner = getattr(self.runtime, operation, None)
        if not callable(runner):
            return ModuleFailed("verified_integrity_conflict")
        outcome = runner(request)
        if isinstance(outcome, V2ReviewSuspended):
            return ModuleSuspended(
                outcome.reason_code,
                recovery_verification=outcome.recovery_verification,
            )
        if isinstance(outcome, V2ReviewFailed):
            return ModuleFailed(
                outcome.failure_code,
                recovery_verification=outcome.recovery_verification,
            )
        if not isinstance(outcome, V2ReviewCompleted):
            return ModuleFailed("verified_integrity_conflict")
        try:
            prepared = _review_prepared_output(
                outcome,
                research_task_id=research_task_id,
                module_invocation_id=module_invocation_id,
                module_input_digest=module_input_digest,
            )
        except _ReviewPreparationError as exc:
            return ModuleFailed(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code=str(exc) or "review_prepared_output_invalid"
                ),
            )
        except (TypeError, ValueError) as exc:
            return ModuleFailed(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="review_prepared_output_invalid"
                ),
            )
        return ModuleCompleted(
            prepared,
            recovery_verification=outcome.recovery_verification,
        )

    def start(self, request: Any) -> Any:
        """启动新 Review Module Run。"""

        return self._run(request, "start")

    def resume(self, request: Any) -> Any:
        """恢复同一 Review runtime 和当前 Result heads。"""

        return self._run(request, "resume")

    def review_current_result_heads(self, *args: Any, **kwargs: Any) -> V2ReviewOutcome:
        """转发核心 runtime seam，便于直接测试和只读调用。"""

        return self.runtime.review_current_result_heads(*args, **kwargs)


def build_v2_task_review_runtime(**kwargs: Any) -> ResearchTaskReviewRunner:
    """构造 V2-native Review runtime 的一行 factory。"""

    return ResearchTaskReviewRunner(**kwargs)


def build_v2_native_task_review_runtime(**kwargs: Any) -> ResearchTaskReviewRunner:
    """V2-native factory 的明确命名别名。"""

    return build_v2_task_review_runtime(**kwargs)


def build_v2_task_review_adapter(**kwargs: Any) -> V2TaskReviewModuleAdapter:
    """构造实现 Controller ``start/resume`` 的 V2 Review adapter。"""

    runtime = kwargs.pop("runtime", None)
    if runtime is None:
        runtime = build_v2_task_review_runtime(**kwargs)
    return V2TaskReviewModuleAdapter(runtime)


def build_v2_native_task_review_adapter(**kwargs: Any) -> V2TaskReviewModuleAdapter:
    """V2-native Review module adapter factory 的明确命名别名。"""

    return build_v2_task_review_adapter(**kwargs)
