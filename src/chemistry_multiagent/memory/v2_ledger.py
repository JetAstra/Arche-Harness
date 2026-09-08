"""investigation_v2 的 Task、Module Run 与 Scientific Record 深账本。

本模块在同一物理 SQLite 中与 schema v10 expand-contract 共存。V2 的领域事实只
通过本模块的领域写入口追加；事件按 task 分区分配连续序号并形成不可变摘要链，
Task Facts 和科学记录索引始终可以从事件日志重建。这里不读取或迁移
legacy_workflow_v1 的事实。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any, Self

from .contracts import (
    CASConflict,
    ContractValidationError,
    IdempotencyConflict,
    MemoryDisabledError,
    OperationConflict,
    WorkingMemoryError,
    canonical_json,
)
from .v2_identities import (
    ContextBinding,
    CreateResearchTaskReviewSession,
    CreateInvestigationContext,
    ReuseResearchTaskReviewSession,
    ReuseInvestigationContext,
    StatelessModule,
)


_V2_SCHEMA_VERSION = "2"
_V2_CONTRACT = "investigation_v2"
_SCIENTIFIC_RECORD_TYPES = frozenset(
    {
        "RetrievalRecord",
        "QueryRecord",
        "HypothesisBatchRecord",
        "HypothesisSelectionRecord",
        "SelectedHypothesisRecord",
        "HypothesisExecutionDecisionRecord",
        "HypothesisExecutionInputRecord",
        "ResearchMaterialRecord",
        "VerifiedEvidenceRecord",
        "InvestigationResultRecord",
        "ReviewRecord",
        "RepairRequestRecord",
        "FinalAnswerRecord",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "RetrievalRecord": frozenset({"summary", "citations", "corpus_snapshot_ref"}),
    "QueryRecord": frozenset({"query", "research_angle", "constraints"}),
    "HypothesisBatchRecord": frozenset(
        {"hypotheses", "rationale", "verification_plan"}
    ),
    "HypothesisSelectionRecord": frozenset(
        {"selected_hypotheses", "selection_reason"}
    ),
    "SelectedHypothesisRecord": frozenset(
        {"hypothesis_ref", "rank", "hypothesis"}
    ),
    "HypothesisExecutionDecisionRecord": frozenset(
        {
            "source_hypothesis_batch_ref",
            "confirmation_id",
            "decisions",
            "execution_input_refs",
        }
    ),
    "HypothesisExecutionInputRecord": frozenset(
        {
            "source_selected_hypothesis_ref",
            "execution_rank",
            "hypothesis",
            "guidance",
            "material_refs",
        }
    ),
    "ResearchMaterialRecord": frozenset(
        {
            "material_id",
            "content_ref",
            "sha256",
            "size_bytes",
            "filename",
            "media_type",
            "description",
        }
    ),
    "VerifiedEvidenceRecord": frozenset(
        {"evidence_items", "artifact_refs", "source_manifest_ref", "summary", "verification"}
    ),
    "InvestigationResultRecord": frozenset(
        {"conclusion", "self_judgment", "observations", "limitations", "evidence_ref"}
    ),
    "ReviewRecord": frozenset({"reviewed_result_refs", "findings", "outcome"}),
    "RepairRequestRecord": frozenset({"result_ref", "problem", "instructions"}),
    "FinalAnswerRecord": frozenset({"answer", "answerability"}),
}
# 首批类型暂不声明可选字段；新增字段必须通过 schema_version 注册，避免
# payload 退化为无法核对的任意字段袋。
_RECORD_ALLOWED_FIELDS = _RECORD_REQUIRED_FIELDS.copy()
# v10 迁移保留的旧事件形状不能强行填入当前生产字段。这个版本只允许
# 迁移器注册的投影字段，既能保留缺失 rationale/summary 的事实，也不会把
# 任意旧 payload 当作新合同。
_V10_MIGRATED_SCHEMA_VERSION = "v10-migrated"
_AUTHOR_IMPORT_V1_SCHEMA_VERSION = "author-import.v1"
_AUTHOR_IMPORT_V1_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "HypothesisBatchRecord": frozenset(
        {"fixture_id", "fixture_sha256", "hypothesis", "provenance"}
    ),
}
_AUTHOR_IMPORT_V1_ALLOWED_FIELDS = _AUTHOR_IMPORT_V1_REQUIRED_FIELDS.copy()
_AUTHOR_IMPORT_V2_SCHEMA_VERSION = "author-import.v2"
_AUTHOR_IMPORT_V2_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "HypothesisBatchRecord": frozenset({"imported_hypotheses"}),
}
_AUTHOR_IMPORT_V2_ALLOWED_FIELDS = _AUTHOR_IMPORT_V2_REQUIRED_FIELDS.copy()
_HYPOTHESIS_GENERATION_V2_SCHEMA_VERSION = "hypothesis-generation.v2"
_HYPOTHESIS_GENERATION_V2_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "HypothesisBatchRecord": frozenset(
        {
            "queries",
            "hypotheses_by_query",
            "ranked_strategies",
            "hypotheses",
            "rationale",
            "verification_plan",
        }
    ),
}
_HYPOTHESIS_GENERATION_V2_ALLOWED_FIELDS = (
    _HYPOTHESIS_GENERATION_V2_REQUIRED_FIELDS.copy()
)
HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION = "hypothesis-confirmation.v2"
_HYPOTHESIS_CONFIRMATION_V2_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "HypothesisExecutionDecisionRecord": frozenset(
        {
            "source_hypothesis_batch_record_id",
            "confirmation_id",
            "decisions",
            "execution_input_scientific_record_ids",
        }
    ),
    "HypothesisExecutionInputRecord": frozenset(
        {
            "source_selected_hypothesis_record_id",
            "execution_rank",
            "hypothesis",
            "guidance",
            "material_upload_refs",
        }
    ),
    "ResearchMaterialRecord": frozenset(
        {
            "material_id",
            "content_ref",
            "sha256",
            "size_bytes",
            "filename",
            "media_type",
            "description",
        }
    ),
}
_HYPOTHESIS_CONFIRMATION_V2_ALLOWED_FIELDS = (
    _HYPOTHESIS_CONFIRMATION_V2_REQUIRED_FIELDS.copy()
)
_V10_MIGRATED_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "RetrievalRecord": frozenset(
        {"source_artifact_ref", "result_digest", "retrieval_result"}
    ),
    "QueryRecord": frozenset({"question_snapshot", "query_angles"}),
    "HypothesisBatchRecord": frozenset(
        {
            "batch_id",
            "batch_index",
            "question_snapshot",
            "generation_source",
            "hypothesis_phase",
            "candidates",
            "source_manifest_ref",
        }
    ),
    "VerifiedEvidenceRecord": frozenset(
        {
            "evidence_items",
            "artifact_refs",
            "verification_object_id",
            "verification_digest",
            "verifier_id",
            "verifier_version",
            "operation_instance_id",
        }
    ),
    "RepairRequestRecord": frozenset({"result_ref", "source_payload"}),
}
_V10_MIGRATED_ALLOWED_FIELDS = _V10_MIGRATED_REQUIRED_FIELDS.copy()
_V10_MIGRATED_ALLOWED_FIELDS["VerifiedEvidenceRecord"] = frozenset(
    set(_V10_MIGRATED_REQUIRED_FIELDS["VerifiedEvidenceRecord"])
    | {"reference_binding_version"}
)
_PRIVATE_RECORD_FIELDS = frozenset(
    {
        "codex_thread_id",
        "turn_id",
        "goal_id",
        "scope_id",
        "session_id",
        "job_id",
        "submission_operation_id",
        "sidecar",
        "transcript",
    }
)
_FORMAL_MODULE_KIND_NAMES = frozenset(
    {
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
        "hypothesis-confirmation",
        "hypothesis-investigation",
        "task-review",
        # 作者假设导入是一个只在 author_hypothesis_import 策略中使用的
        # 内部边界；它仍使用稳定的连字符名称，不向外暴露旧 query/task 名。
        "hypothesis-import",
    }
)
_TASK_TERMINAL_STATUSES = frozenset(
    {"succeeded", "unanswerable", "stopped", "failed"}
)
_MISSING = object()


class WorkingMemoryIntegrityConflict(WorkingMemoryError, ValueError):
    """V2 事件日志或其可重建关系不满足账本不变量。"""


class V2LedgerSchemaConflict(ContractValidationError):
    """传入的数据库不具备受支持的 Working Memory 账本合同。"""


def _normalize_task_terminal_outcome(
    outcome: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """校验 TaskClosed 的公开终态，保留可审计的科学引用。"""

    if isinstance(outcome, str):
        value: Any = {"status": outcome}
    elif isinstance(outcome, Mapping):
        value = _copy_json(dict(outcome), "TaskClosed.outcome")
    else:
        raise ContractValidationError("TaskClosed.outcome 必须是字符串或 JSON object")
    if not isinstance(value, dict):  # pragma: no cover - _copy_json 已保证
        raise ContractValidationError("TaskClosed.outcome 必须是 JSON object")
    status = value.get("status")
    if status not in _TASK_TERMINAL_STATUSES:
        raise ContractValidationError(
            "TaskClosed.outcome.status 必须为 succeeded、unanswerable、stopped 或 failed"
        )
    for field_name in (
        "review_scientific_record_id",
        "final_answer_scientific_record_id",
        "selection_scientific_record_id",
    ):
        if field_name in value:
            _require_text(value[field_name], f"TaskClosed.outcome.{field_name}")
    return value


def _require_text(value: Any, field_name: str) -> str:
    """校验一个不透明身份字段并返回去除首尾空白后的值。"""

    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _copy_json(value: Any, field_name: str) -> Any:
    """校验严格 JSON 并返回不与调用方共享的可变副本。"""

    try:
        encoded = canonical_json(value)
        return json.loads(encoded)
    except ContractValidationError as exc:
        raise ContractValidationError(f"{field_name} 不是严格 JSON：{exc}") from exc


def _coalesce_alias(
    value: Mapping[str, Any],
    canonical_name: str,
    legacy_name: str,
    field_name: str,
) -> Any:
    """读取一对字段别名，并拒绝双方同时存在但内容不一致。"""

    canonical = value.get(canonical_name, _MISSING)
    legacy = value.get(legacy_name, _MISSING)
    if canonical is not _MISSING and legacy is not _MISSING:
        try:
            same = canonical_json(canonical) == canonical_json(legacy)
        except ContractValidationError as exc:
            raise ContractValidationError(f"{field_name} 不是严格 JSON") from exc
        if not same:
            raise ContractValidationError(
                f"{field_name} 的新旧字段别名内容不一致"
            )
    if canonical is not _MISSING:
        return canonical
    if legacy is not _MISSING:
        return legacy
    return None


def _require_object(value: Any, field_name: str) -> dict[str, Any]:
    """校验完整子结构必须是 JSON object，而不是字符串或列表。"""

    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} 必须是 JSON object")
    copied = _copy_json(dict(value), field_name)
    if not isinstance(copied, dict):  # pragma: no cover - 由上面的类型校验保证
        raise ContractValidationError(f"{field_name} 必须是 JSON object")
    return copied


def _require_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    """校验记录引用是有序且不重复的字符串序列。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractValidationError(f"{field_name} 必须是字符串序列")
    result = tuple(_require_text(item, f"{field_name}[]") for item in value)
    if len(set(result)) != len(result):
        raise ContractValidationError(f"{field_name} 不能包含重复引用")
    return result


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """复制已校验的 JSON mapping，避免读取结果反向修改账本快照。"""

    copied = _copy_json(dict(value), "payload")
    if not isinstance(copied, dict):  # pragma: no cover
        raise ContractValidationError("payload 必须是 JSON object")
    return copied


def _validate_sha256(value: Any, field_name: str) -> str:
    """校验新合同中的内容摘要为小写 SHA-256。"""

    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractValidationError(f"{field_name} 必须是 64 位小写 SHA-256")
    return value


def _validate_record_schema_version(value: Any) -> str | int:
    """校验 Scientific Record 的持久结构版本。"""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ContractValidationError("schema_version 必须是字符串或整数")
    if isinstance(value, str):
        if not value.strip():
            raise ContractValidationError("schema_version 不能为空")
        return value.strip()
    if value <= 0:
        raise ContractValidationError("schema_version 必须大于 0")
    return value


def _require_nonempty_text(value: Any, field_name: str) -> str:
    """校验科学 payload 中必须提供的非空文本字段。"""

    return _require_text(value, field_name)


def _require_nullable_text(value: Any, field_name: str) -> str | None:
    """校验可选文本字段；显式 null 表示调用方没有提供内容。"""

    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _validate_material_filename(value: Any, field_name: str) -> str:
    """校验材料文件名是安全 basename，不把本地路径带入科学记录。"""

    filename = _require_nonempty_text(value, field_name)
    if (
        filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or any(ord(char) < 32 for char in filename)
    ):
        raise ContractValidationError(
            f"{field_name} 必须是安全 basename，不能包含路径或控制字符"
        )
    return filename


def _validate_material_content_ref(value: Any, field_name: str) -> str:
    """校验内容引用为不透明标识，拒绝绝对路径和路径穿越。"""

    content_ref = _require_nonempty_text(value, field_name)
    if (
        "\x00" in content_ref
        or any(ord(char) < 32 for char in content_ref)
        or content_ref.startswith(("/", "\\"))
        or any(part == ".." for part in re.split(r"[/\\]", content_ref))
    ):
        raise ContractValidationError(
            f"{field_name} 必须是不透明内容引用，不能是本地路径"
        )
    return content_ref


def _require_list(value: Any, field_name: str, *, allow_empty: bool = True) -> list[Any]:
    """校验科学 payload 中的 JSON 数组并复制其内容。"""

    if not isinstance(value, list):
        raise ContractValidationError(f"{field_name} 必须是 JSON array")
    if not allow_empty and not value:
        raise ContractValidationError(f"{field_name} 不能为空")
    copied = _copy_json(value, field_name)
    if not isinstance(copied, list):  # pragma: no cover - 由上面的类型校验保证
        raise ContractValidationError(f"{field_name} 必须是 JSON array")
    return copied


def _require_record_ref_list(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    """校验 Scientific Record 身份列表的顺序、唯一性和文本类型。"""

    values = _require_list(value, field_name, allow_empty=allow_empty)
    refs = [_require_text(item, f"{field_name}[]") for item in values]
    if len(set(refs)) != len(refs):
        raise ContractValidationError(f"{field_name} 不能包含重复引用")
    return refs


def _validate_scientific_payload(
    record_type: str,
    payload: Any,
    schema_version: str | int | None = None,
) -> dict[str, Any]:
    """按注册的记录类型和结构版本校验科学 payload。

    ``v10-migrated`` 是一个封闭的兼容版本，只允许迁移器声明的忠实字段；它
    不会把旧事件缺失的生产摘要解释成新事实。
    """

    if record_type not in _SCIENTIFIC_RECORD_TYPES:
        raise ContractValidationError(f"record_type 未注册：{record_type}")
    value = _require_object(payload, f"{record_type}.payload")
    migrated = schema_version == _V10_MIGRATED_SCHEMA_VERSION
    author_import_v1 = schema_version == _AUTHOR_IMPORT_V1_SCHEMA_VERSION
    author_import_v2 = schema_version == _AUTHOR_IMPORT_V2_SCHEMA_VERSION
    hypothesis_generation_v2 = (
        schema_version == _HYPOTHESIS_GENERATION_V2_SCHEMA_VERSION
    )
    hypothesis_confirmation_v2 = (
        schema_version == HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION
    )
    if migrated and record_type not in _V10_MIGRATED_REQUIRED_FIELDS:
        raise ContractValidationError(
            f"{record_type} 不支持 schema_version={_V10_MIGRATED_SCHEMA_VERSION}"
        )
    if author_import_v1 and record_type not in _AUTHOR_IMPORT_V1_REQUIRED_FIELDS:
        raise ContractValidationError(
            f"{record_type} 不支持 schema_version={_AUTHOR_IMPORT_V1_SCHEMA_VERSION}"
        )
    if author_import_v2 and record_type not in _AUTHOR_IMPORT_V2_REQUIRED_FIELDS:
        raise ContractValidationError(
            f"{record_type} 不支持 schema_version={_AUTHOR_IMPORT_V2_SCHEMA_VERSION}"
        )
    if (
        hypothesis_generation_v2
        and record_type not in _HYPOTHESIS_GENERATION_V2_REQUIRED_FIELDS
    ):
        raise ContractValidationError(
            "record_type 不支持 "
            f"schema_version={_HYPOTHESIS_GENERATION_V2_SCHEMA_VERSION}"
        )
    if (
        hypothesis_confirmation_v2
        and record_type not in _HYPOTHESIS_CONFIRMATION_V2_REQUIRED_FIELDS
    ):
        raise ContractValidationError(
            "record_type 不支持 "
            f"schema_version={HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION}"
        )
    required = (
        _V10_MIGRATED_REQUIRED_FIELDS[record_type]
        if migrated
        else _AUTHOR_IMPORT_V1_REQUIRED_FIELDS[record_type]
        if author_import_v1
        else _AUTHOR_IMPORT_V2_REQUIRED_FIELDS[record_type]
        if author_import_v2
        else _HYPOTHESIS_GENERATION_V2_REQUIRED_FIELDS[record_type]
        if hypothesis_generation_v2
        else _HYPOTHESIS_CONFIRMATION_V2_REQUIRED_FIELDS[record_type]
        if hypothesis_confirmation_v2
        else _RECORD_REQUIRED_FIELDS[record_type]
    )
    missing = sorted(required.difference(value))
    if missing:
        raise ContractValidationError(
            f"{record_type}.payload 缺少必需字段：{', '.join(missing)}"
        )
    allowed = (
        _V10_MIGRATED_ALLOWED_FIELDS[record_type]
        if migrated
        else _AUTHOR_IMPORT_V1_ALLOWED_FIELDS[record_type]
        if author_import_v1
        else _AUTHOR_IMPORT_V2_ALLOWED_FIELDS[record_type]
        if author_import_v2
        else _HYPOTHESIS_GENERATION_V2_ALLOWED_FIELDS[record_type]
        if hypothesis_generation_v2
        else _HYPOTHESIS_CONFIRMATION_V2_ALLOWED_FIELDS[record_type]
        if hypothesis_confirmation_v2
        else _RECORD_ALLOWED_FIELDS[record_type]
    )
    extra = sorted(set(value).difference(allowed))
    if extra:
        raise ContractValidationError(
            f"{record_type}.payload 含未注册字段：{', '.join(extra)}"
        )
    private = sorted(_PRIVATE_RECORD_FIELDS.intersection(value))
    if private:
        raise ContractValidationError(
            f"{record_type}.payload 不得保存 Harness 内部字段：{', '.join(private)}"
        )

    if author_import_v1 and record_type == "HypothesisBatchRecord":
        _require_nonempty_text(value["fixture_id"], "HypothesisBatchRecord.fixture_id")
        _validate_sha256(value["fixture_sha256"], "HypothesisBatchRecord.fixture_sha256")
        _require_object(value["hypothesis"], "HypothesisBatchRecord.hypothesis")
        _require_object(value["provenance"], "HypothesisBatchRecord.provenance")
    elif author_import_v2 and record_type == "HypothesisBatchRecord":
        imported = _require_list(
            value["imported_hypotheses"],
            "HypothesisBatchRecord.imported_hypotheses",
            allow_empty=False,
        )
        if len(imported) > 5:
            raise ContractValidationError(
                "HypothesisBatchRecord.imported_hypotheses 最多包含五条作者假设"
            )
        hypothesis_refs: list[str] = []
        for index, item in enumerate(imported):
            field_name = f"HypothesisBatchRecord.imported_hypotheses[{index}]"
            if not isinstance(item, Mapping):
                raise ContractValidationError(f"{field_name} 必须是 object")
            expected = {
                "hypothesis_ref",
                "source_ref",
                "source_digest",
                "hypothesis",
                "provenance",
            }
            if set(item) != expected:
                raise ContractValidationError(
                    f"{field_name} 字段必须恰好为 {', '.join(sorted(expected))}"
                )
            hypothesis_ref = _require_nonempty_text(
                item["hypothesis_ref"], f"{field_name}.hypothesis_ref"
            )
            _require_nonempty_text(item["source_ref"], f"{field_name}.source_ref")
            _validate_sha256(item["source_digest"], f"{field_name}.source_digest")
            hypothesis = _require_object(
                item["hypothesis"], f"{field_name}.hypothesis"
            )
            embedded_ref = hypothesis.get(
                "hypothesis_id", hypothesis.get("hypothesis_ref")
            )
            if embedded_ref is not None and embedded_ref != hypothesis_ref:
                raise ContractValidationError(
                    f"{field_name}.hypothesis 内部身份与 hypothesis_ref 不一致"
                )
            _require_object(item["provenance"], f"{field_name}.provenance")
            hypothesis_refs.append(hypothesis_ref)
        if len(set(hypothesis_refs)) != len(hypothesis_refs):
            raise ContractValidationError(
                "HypothesisBatchRecord.imported_hypotheses 不能重复 hypothesis_ref"
            )
    elif hypothesis_generation_v2 and record_type == "HypothesisBatchRecord":
        queries = _require_list(
            value["queries"],
            "HypothesisBatchRecord.queries",
            allow_empty=False,
        )
        normalized_queries = [
            _require_nonempty_text(item, f"HypothesisBatchRecord.queries[{index}]")
            for index, item in enumerate(queries)
        ]
        groups = _require_list(
            value["hypotheses_by_query"],
            "HypothesisBatchRecord.hypotheses_by_query",
            allow_empty=False,
        )
        if len(groups) != len(normalized_queries):
            raise ContractValidationError(
                "HypothesisBatchRecord.hypotheses_by_query 与 queries 数量不一致"
            )
        for index, group in enumerate(groups):
            if not isinstance(group, Mapping):
                raise ContractValidationError(
                    f"HypothesisBatchRecord.hypotheses_by_query[{index}] 必须是 object"
                )
            if group.get("query") != normalized_queries[index]:
                raise ContractValidationError(
                    "HypothesisBatchRecord.hypotheses_by_query 必须逐项对应 queries"
                )
            candidates = _require_list(
                group.get("hypotheses"),
                f"HypothesisBatchRecord.hypotheses_by_query[{index}].hypotheses",
                allow_empty=False,
            )
            if any(not isinstance(candidate, Mapping) for candidate in candidates):
                raise ContractValidationError(
                    "HypothesisBatchRecord.hypotheses_by_query.hypotheses "
                    "必须只包含 object"
                )
        ranked = _require_list(
            value["ranked_strategies"],
            "HypothesisBatchRecord.ranked_strategies",
            allow_empty=False,
        )
        hypotheses = _require_list(
            value["hypotheses"],
            "HypothesisBatchRecord.hypotheses",
            allow_empty=False,
        )
        if ranked != hypotheses:
            raise ContractValidationError(
                "HypothesisBatchRecord.hypotheses 必须与 ranked_strategies 一致"
            )
        _require_nonempty_text(value["rationale"], "HypothesisBatchRecord.rationale")
        _require_nonempty_text(
            value["verification_plan"], "HypothesisBatchRecord.verification_plan"
        )
    elif migrated and record_type == "RetrievalRecord":
        _require_nonempty_text(
            value["source_artifact_ref"], "RetrievalRecord.source_artifact_ref"
        )
        _validate_sha256(value["result_digest"], "RetrievalRecord.result_digest")
        _require_object(value["retrieval_result"], "RetrievalRecord.retrieval_result")
    elif migrated and record_type == "QueryRecord":
        _require_object(value["question_snapshot"], "QueryRecord.question_snapshot")
        angles = _require_list(value["query_angles"], "QueryRecord.query_angles")
        for index, angle in enumerate(angles):
            _require_nonempty_text(angle, f"QueryRecord.query_angles[{index}]")
    elif migrated and record_type == "HypothesisBatchRecord":
        _require_nonempty_text(value["batch_id"], "HypothesisBatchRecord.batch_id")
        batch_index = value["batch_index"]
        if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 1:
            raise ContractValidationError(
                "HypothesisBatchRecord.batch_index 必须是正整数"
            )
        _require_object(
            value["question_snapshot"], "HypothesisBatchRecord.question_snapshot"
        )
        _require_object(
            value["generation_source"], "HypothesisBatchRecord.generation_source"
        )
        phase = _require_object(
            value["hypothesis_phase"], "HypothesisBatchRecord.hypothesis_phase"
        )
        _require_list(
            phase.get("ranked_strategies"),
            "HypothesisBatchRecord.hypothesis_phase.ranked_strategies",
            allow_empty=False,
        )
        _require_list(
            value["candidates"], "HypothesisBatchRecord.candidates", allow_empty=False
        )
        _require_nonempty_text(
            value["source_manifest_ref"], "HypothesisBatchRecord.source_manifest_ref"
        )
    elif migrated and record_type == "VerifiedEvidenceRecord":
        _require_list(
            value["evidence_items"],
            "VerifiedEvidenceRecord.evidence_items",
            allow_empty=False,
        )
        _require_list(value["artifact_refs"], "VerifiedEvidenceRecord.artifact_refs")
        _require_nonempty_text(
            value["verification_object_id"],
            "VerifiedEvidenceRecord.verification_object_id",
        )
        _validate_sha256(
            value["verification_digest"],
            "VerifiedEvidenceRecord.verification_digest",
        )
        _require_nonempty_text(value["verifier_id"], "VerifiedEvidenceRecord.verifier_id")
        _require_nonempty_text(
            value["verifier_version"], "VerifiedEvidenceRecord.verifier_version"
        )
        _require_nonempty_text(
            value["operation_instance_id"],
            "VerifiedEvidenceRecord.operation_instance_id",
        )
        if "reference_binding_version" in value:
            _require_nonempty_text(
                value["reference_binding_version"],
                "VerifiedEvidenceRecord.reference_binding_version",
            )
    elif migrated and record_type == "RepairRequestRecord":
        _require_nonempty_text(value["result_ref"], "RepairRequestRecord.result_ref")
        _require_object(value["source_payload"], "RepairRequestRecord.source_payload")
    elif record_type == "RetrievalRecord":
        _require_nonempty_text(value["summary"], "RetrievalRecord.summary")
        _require_list(value["citations"], "RetrievalRecord.citations")
        _require_nonempty_text(
            value["corpus_snapshot_ref"], "RetrievalRecord.corpus_snapshot_ref"
        )
    elif record_type == "QueryRecord":
        _require_nonempty_text(value["query"], "QueryRecord.query")
        _require_nonempty_text(
            value["research_angle"], "QueryRecord.research_angle"
        )
        _require_object(value["constraints"], "QueryRecord.constraints")
    elif record_type == "HypothesisBatchRecord":
        _require_list(value["hypotheses"], "HypothesisBatchRecord.hypotheses", allow_empty=False)
        _require_nonempty_text(value["rationale"], "HypothesisBatchRecord.rationale")
        _require_nonempty_text(
            value["verification_plan"], "HypothesisBatchRecord.verification_plan"
        )
    elif record_type == "HypothesisSelectionRecord":
        selected = _require_list(
            value["selected_hypotheses"],
            "HypothesisSelectionRecord.selected_hypotheses",
            allow_empty=False,
        )
        hypothesis_refs: list[str] = []
        ranks: list[int] = []
        for index, item in enumerate(selected):
            if not isinstance(item, Mapping):
                raise ContractValidationError(
                    f"HypothesisSelectionRecord.selected_hypotheses[{index}] 必须是 object"
                )
            if set(item) != {"hypothesis_ref", "rank"}:
                raise ContractValidationError(
                    "HypothesisSelectionRecord.selected_hypotheses 只能包含 hypothesis_ref 和 rank"
                )
            _require_nonempty_text(
                item["hypothesis_ref"],
                f"HypothesisSelectionRecord.selected_hypotheses[{index}].hypothesis_ref",
            )
            if isinstance(item["rank"], bool) or not isinstance(item["rank"], int) or item["rank"] < 1:
                raise ContractValidationError(
                    "HypothesisSelectionRecord.selected_hypotheses.rank 必须是正整数"
                )
            hypothesis_refs.append(item["hypothesis_ref"])
            ranks.append(item["rank"])
        if len(set(hypothesis_refs)) != len(hypothesis_refs):
            raise ContractValidationError(
                "HypothesisSelectionRecord.selected_hypotheses 不能重复 hypothesis_ref"
            )
        if len(set(ranks)) != len(ranks):
            raise ContractValidationError(
                "HypothesisSelectionRecord.selected_hypotheses 不能重复 rank"
            )
        _require_nonempty_text(
            value["selection_reason"], "HypothesisSelectionRecord.selection_reason"
        )
    elif record_type == "SelectedHypothesisRecord":
        hypothesis_ref = _require_nonempty_text(
            value["hypothesis_ref"], "SelectedHypothesisRecord.hypothesis_ref"
        )
        rank = value["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ContractValidationError(
                "SelectedHypothesisRecord.rank 必须是正整数"
            )
        hypothesis = _require_object(
            value["hypothesis"], "SelectedHypothesisRecord.hypothesis"
        )
        embedded_ref = hypothesis.get(
            "hypothesis_id", hypothesis.get("hypothesis_ref")
        )
        if embedded_ref is not None and embedded_ref != hypothesis_ref:
            raise ContractValidationError(
                "SelectedHypothesisRecord.hypothesis 内部身份与 hypothesis_ref 不一致"
            )
    elif record_type == "HypothesisExecutionDecisionRecord":
        batch_field = (
            "source_hypothesis_batch_record_id"
            if hypothesis_confirmation_v2
            else "source_hypothesis_batch_ref"
        )
        candidate_field = (
            "hypothesis_candidate_record_id"
            if hypothesis_confirmation_v2
            else "candidate_ref"
        )
        execution_inputs_field = (
            "execution_input_scientific_record_ids"
            if hypothesis_confirmation_v2
            else "execution_input_refs"
        )
        _require_nonempty_text(
            value[batch_field],
            f"HypothesisExecutionDecisionRecord.{batch_field}",
        )
        _require_nonempty_text(
            value["confirmation_id"],
            "HypothesisExecutionDecisionRecord.confirmation_id",
        )
        decisions = _require_list(
            value["decisions"],
            "HypothesisExecutionDecisionRecord.decisions",
            allow_empty=False,
        )
        if len(decisions) > 5:
            raise ContractValidationError(
                "HypothesisExecutionDecisionRecord.decisions 最多包含五条候选决定"
            )
        candidate_refs: list[str] = []
        execution_ranks: list[int] = []
        for index, item in enumerate(decisions):
            field_name = f"HypothesisExecutionDecisionRecord.decisions[{index}]"
            if not isinstance(item, Mapping):
                raise ContractValidationError(f"{field_name} 必须是 object")
            decision = item.get("decision")
            if not isinstance(decision, str) or decision not in {"execute", "cancel"}:
                raise ContractValidationError(
                    f"{field_name}.decision 必须为 execute 或 cancel"
                )
            expected_keys = (
                {candidate_field, "decision", "execution_rank"}
                if decision == "execute"
                else {candidate_field, "decision"}
            )
            if set(item) != expected_keys:
                raise ContractValidationError(
                    f"{field_name} 字段必须恰好为 {', '.join(sorted(expected_keys))}"
                )
            candidate_refs.append(
                _require_nonempty_text(
                    item[candidate_field], f"{field_name}.{candidate_field}"
                )
            )
            if decision == "execute":
                rank = item["execution_rank"]
                if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                    raise ContractValidationError(
                        f"{field_name}.execution_rank 必须是正整数"
                    )
                execution_ranks.append(rank)
        if len(set(candidate_refs)) != len(candidate_refs):
            raise ContractValidationError(
                "HypothesisExecutionDecisionRecord.decisions 不能重复 candidate_ref"
            )
        if not execution_ranks:
            raise ContractValidationError(
                "HypothesisExecutionDecisionRecord 至少需要一条 execute 决定"
            )
        if sorted(execution_ranks) != list(range(1, len(execution_ranks) + 1)):
            raise ContractValidationError(
                "HypothesisExecutionDecisionRecord.execute 的 execution_rank 必须从 1 连续排列"
            )
        execution_input_refs = _require_record_ref_list(
            value[execution_inputs_field],
            f"HypothesisExecutionDecisionRecord.{execution_inputs_field}",
            allow_empty=False,
        )
        if len(execution_input_refs) != len(execution_ranks):
            raise ContractValidationError(
                "HypothesisExecutionDecisionRecord.execution_input_refs 必须与 execute 决定一一对应"
            )
    elif record_type == "HypothesisExecutionInputRecord":
        selected_hypothesis_field = (
            "source_selected_hypothesis_record_id"
            if hypothesis_confirmation_v2
            else "source_selected_hypothesis_ref"
        )
        material_refs_field = (
            "material_upload_refs" if hypothesis_confirmation_v2 else "material_refs"
        )
        _require_nonempty_text(
            value[selected_hypothesis_field],
            f"HypothesisExecutionInputRecord.{selected_hypothesis_field}",
        )
        rank = value["execution_rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ContractValidationError(
                "HypothesisExecutionInputRecord.execution_rank 必须是正整数"
            )
        _require_object(value["hypothesis"], "HypothesisExecutionInputRecord.hypothesis")
        _require_nullable_text(value["guidance"], "HypothesisExecutionInputRecord.guidance")
        _require_record_ref_list(
            value[material_refs_field],
            f"HypothesisExecutionInputRecord.{material_refs_field}",
            allow_empty=True,
        )
    elif record_type == "ResearchMaterialRecord":
        _require_nonempty_text(value["material_id"], "ResearchMaterialRecord.material_id")
        _validate_material_content_ref(value["content_ref"], "ResearchMaterialRecord.content_ref")
        _validate_sha256(value["sha256"], "ResearchMaterialRecord.sha256")
        size_bytes = value["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ContractValidationError(
                "ResearchMaterialRecord.size_bytes 必须是非负整数"
            )
        _validate_material_filename(value["filename"], "ResearchMaterialRecord.filename")
        _require_nonempty_text(value["media_type"], "ResearchMaterialRecord.media_type")
        _require_nullable_text(value["description"], "ResearchMaterialRecord.description")
    elif record_type == "VerifiedEvidenceRecord":
        _require_list(
            value["evidence_items"],
            "VerifiedEvidenceRecord.evidence_items",
            allow_empty=False,
        )
        _require_list(value["artifact_refs"], "VerifiedEvidenceRecord.artifact_refs")
        _require_nonempty_text(
            value["source_manifest_ref"], "VerifiedEvidenceRecord.source_manifest_ref"
        )
        _require_nonempty_text(value["summary"], "VerifiedEvidenceRecord.summary")
        _require_object(value["verification"], "VerifiedEvidenceRecord.verification")
    elif record_type == "InvestigationResultRecord":
        _require_nonempty_text(
            value["conclusion"], "InvestigationResultRecord.conclusion"
        )
        if value["self_judgment"] not in {"supported", "refuted"}:
            raise ContractValidationError(
                "InvestigationResultRecord.self_judgment 必须为 supported 或 refuted"
            )
        _require_list(
            value["observations"], "InvestigationResultRecord.observations"
        )
        _require_list(value["limitations"], "InvestigationResultRecord.limitations")
        _require_nonempty_text(
            value["evidence_ref"], "InvestigationResultRecord.evidence_ref"
        )
    elif record_type == "ReviewRecord":
        _require_record_ref_list(
            value["reviewed_result_refs"],
            "ReviewRecord.reviewed_result_refs",
        )
        _require_list(value["findings"], "ReviewRecord.findings", allow_empty=False)
        if value["outcome"] not in {
            "Accepted",
            "RepairRequired",
            "RegenerateHypotheses",
            "Unanswerable",
            "Stopped",
        }:
            raise ContractValidationError("ReviewRecord.outcome 不是注册的互斥结论")
    elif record_type == "RepairRequestRecord":
        _require_nonempty_text(value["result_ref"], "RepairRequestRecord.result_ref")
        _require_nonempty_text(value["problem"], "RepairRequestRecord.problem")
        _require_nonempty_text(
            value["instructions"], "RepairRequestRecord.instructions"
        )
    elif record_type == "FinalAnswerRecord":
        _require_nonempty_text(value["answer"], "FinalAnswerRecord.answer")
        if value["answerability"] not in {"answerable", "unanswerable"}:
            raise ContractValidationError(
                "FinalAnswerRecord.answerability 必须为 answerable 或 unanswerable"
            )
    return value


def _payload_digest(payload: Mapping[str, Any]) -> str:
    """计算 Scientific Record payload 的内容摘要。"""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _prepared_output_digest(records: Sequence["ScientificRecord"]) -> str:
    """按调用方提供的记录顺序计算输出摘要，不把时间戳带入。

    输出顺序属于模块合同：同一 bundle 中后面的记录可以引用前面的记录，重排
    会改变摘要并在重放时报告漂移。
    """

    body = [record.canonical_identity_dict() for record in records]
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    field_name: str,
) -> None:
    """拒绝事件或 envelope 中未注册的顶层字段。"""

    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(actual.difference(expected))
        details: list[str] = []
        if missing:
            details.append(f"缺少 {', '.join(missing)}")
        if extra:
            details.append(f"含额外字段 {', '.join(extra)}")
        raise WorkingMemoryIntegrityConflict(f"{field_name} 字段合同不匹配：{'；'.join(details)}")


def _module_kind_family(module_kind: str) -> str:
    """把模块公开名称归入首批输出成组合同。"""

    normalized = module_kind.strip().casefold().replace("_", "-").replace(" ", "-")
    if normalized in {"query", "research-query-formulation"}:
        return "research-query-formulation"
    if normalized == "retrieval" or normalized.startswith("retrieval-"):
        return "retrieval"
    if normalized in {"hypothesis", "hypotheses", "hypothesis-generation"}:
        return "hypothesis-generation"
    if normalized == "hypothesis-import":
        return "hypothesis-import"
    if normalized in {
        "hypothesis-confirmation",
        "hypothesis-confirm",
        "confirmation",
        "hypothesis-confirmation-module",
    }:
        return "hypothesis-confirmation"
    if "top5" in normalized or "selection" in normalized:
        return "hypothesis-selection"
    if normalized in {
        "investigation",
        "investigation-and-experiment",
        "investigation-experiment",
    } or normalized.startswith("investigation-"):
        return "hypothesis-investigation"
    if normalized == "review" or normalized.endswith("-review"):
        return "task-review"
    return "generic"


def _public_context_binding(
    module_kind: str,
    binding: ContextBinding | None,
) -> dict[str, str]:
    """返回 canonical owner；旧账本缺 owner 时拒绝伪造绑定。"""

    if binding is None:
        if _module_kind_family(module_kind) not in {
            "hypothesis-investigation",
            "task-review",
        }:
            return StatelessModule().to_dict()
        raise ContractValidationError(
            "旧 Module Run 缺少 context_binding，不能生成 canonical 公共 envelope"
        )
    if not isinstance(
        binding,
        (
            StatelessModule,
            CreateInvestigationContext,
            ReuseInvestigationContext,
            CreateResearchTaskReviewSession,
            ReuseResearchTaskReviewSession,
        ),
    ):
        raise ContractValidationError("context_binding 不是已注册的 V2 owner 类型")
    return binding.to_dict()


def _canonical_event_payload(
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """把事件 payload 投影为当前公开合同，不把旧身份键带出账本。"""

    value = _copy_mapping(payload)
    if event_type == "ModuleInvocationStarted":
        aliases = {
            "task_id": "research_task_id",
            "inquiry_id": "research_task_id",
            "module_run_id": "module_invocation_id",
            "input_record_refs": "input_scientific_record_ids",
            "input_digest": "module_input_digest",
            "runtime": "context_binding",
        }
        result = {
            aliases.get(key, key): item
            for key, item in value.items()
            if key not in {"runtime_binding", "runtime_ref"}
        }
        if "context_binding" not in result:
            result["context_binding"] = StatelessModule().to_dict()
        return result
    if event_type == "ScientificRecordCreated":
        aliases = {
            "record_id": "scientific_record_id",
            "task_id": "research_task_id",
            "inquiry_id": "research_task_id",
            "record_type": "scientific_record_type",
            "schema_version": "scientific_record_schema_version",
            "record_schema_version": "scientific_record_schema_version",
            "producer_module_run_id": "producer_module_invocation_id",
            "source_record_refs": "source_scientific_record_ids",
        }
        return {aliases.get(key, key): item for key, item in value.items()}
    if event_type == "ModuleInvocationCompleted":
        return {
            "module_invocation_id": value.get(
                "module_invocation_id", value.get("module_run_id")
            ),
            "output_scientific_record_ids": value.get(
                "output_scientific_record_ids", value.get("output_record_refs", [])
            ),
            "module_output_digest": value.get(
                "module_output_digest", value.get("output_digest")
            ),
        }
    if event_type == "ModuleInvocationFailed":
        return {
            "module_invocation_id": value.get(
                "module_invocation_id", value.get("module_run_id")
            ),
            "failure_code": value.get("failure_code"),
        }
    if event_type in {"ResearchTaskClosed", "ResearchTaskFailed"}:
        outcome = value.get("outcome")
        if isinstance(outcome, Mapping):
            outcome = _canonical_lifecycle_outcome(outcome)
        return {"outcome": outcome}
    return value


def _validate_module_kind(module_kind: Any) -> str:
    """校验首批公开 Module Kind，避免未知模块绕过输出合同。"""

    normalized = _require_text(module_kind, "module_kind").casefold()
    normalized = normalized.replace("_", "-").replace(" ", "-")
    if normalized not in _FORMAL_MODULE_KIND_NAMES:
        raise ContractValidationError(f"module_kind 未注册：{module_kind}")
    return normalized


def _validate_hypothesis_confirmation_output_group(
    records: Sequence["ScientificRecord"],
    available_records: Mapping[str, "ScientificRecord"],
) -> None:
    """核对 Confirmation 的候选、决定、执行输入和材料闭包。

    ``available_records`` 同时包含当前输出和此前已经提交的记录。当前模块只
    允许使用此前的 Hypothesis Batch/SelectedHypothesisRecord 作为候选来源；材料
    和执行输入必须在本次输出中成组保存，避免从展示文件或其他确认批次猜测身份。
    """

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
    materials = [
        record
        for record in records
        if record.scientific_record_type == "ResearchMaterialRecord"
    ]
    if len(decisions) != 1:
        raise ContractValidationError(
            "hypothesis-confirmation Module Run 必须恰好输出一条 HypothesisExecutionDecisionRecord"
        )
    if not 1 <= len(inputs) <= 5:
        raise ContractValidationError(
            "hypothesis-confirmation Module Run 必须输出一至五条 HypothesisExecutionInputRecord"
        )
    if set(record.scientific_record_type for record in records).difference(
        {
            "HypothesisExecutionDecisionRecord",
            "HypothesisExecutionInputRecord",
            "ResearchMaterialRecord",
        }
    ):
        raise ContractValidationError(
            "hypothesis-confirmation Module Run 只能输出 Decision、Input 和 Material 记录"
        )

    decision = decisions[0]
    decision_payload = decision.payload
    canonical_confirmation = (
        decision.scientific_record_schema_version
        == HYPOTHESIS_CONFIRMATION_V2_SCHEMA_VERSION
    )
    batch_field = (
        "source_hypothesis_batch_record_id"
        if canonical_confirmation
        else "source_hypothesis_batch_ref"
    )
    candidate_field = (
        "hypothesis_candidate_record_id"
        if canonical_confirmation
        else "candidate_ref"
    )
    execution_inputs_field = (
        "execution_input_scientific_record_ids"
        if canonical_confirmation
        else "execution_input_refs"
    )
    selected_hypothesis_field = (
        "source_selected_hypothesis_record_id"
        if canonical_confirmation
        else "source_selected_hypothesis_ref"
    )
    material_refs_field = (
        "material_upload_refs" if canonical_confirmation else "material_refs"
    )
    batch_ref = decision_payload[batch_field]
    batch = available_records.get(batch_ref)
    if batch is None or batch.scientific_record_type != "HypothesisBatchRecord":
        raise WorkingMemoryIntegrityConflict(
            "HypothesisExecutionDecisionRecord.source_hypothesis_batch_ref 必须引用当前 task 的 HypothesisBatchRecord"
        )
    if batch_ref not in decision.source_scientific_record_ids:
        raise ContractValidationError(
            "HypothesisExecutionDecisionRecord.source_scientific_record_ids 必须引用 source_hypothesis_batch_ref"
        )

    # Selection/Import 都通过 source_record_refs 把 SelectedHypothesisRecord 绑定
    # 到 Batch；因此确认阶段不从候选列表位置推断身份。
    candidates = [
        record
        for record in available_records.values()
        if record.scientific_record_type == "SelectedHypothesisRecord"
        and batch_ref in record.source_scientific_record_ids
    ]
    if not 1 <= len(candidates) <= 5:
        raise WorkingMemoryIntegrityConflict(
            "Confirmation 的 source_hypothesis_batch_ref 没有一至五条绑定的 SelectedHypothesisRecord"
        )

    candidate_by_ref: dict[str, ScientificRecord] = {}
    for candidate in candidates:
        aliases = {candidate.scientific_record_id}
        if not canonical_confirmation:
            aliases.add(candidate.payload["hypothesis_ref"])
        for alias in aliases:
            previous = candidate_by_ref.get(alias)
            if (
                previous is not None
                and previous.scientific_record_id != candidate.scientific_record_id
            ):
                raise WorkingMemoryIntegrityConflict(
                    "Confirmation 候选引用在 SelectedHypothesisRecord 之间发生歧义"
                )
            candidate_by_ref[alias] = candidate

    selected_by_decision: dict[str, Mapping[str, Any]] = {}
    for item in decision_payload["decisions"]:
        candidate = candidate_by_ref.get(item[candidate_field])
        if candidate is None:
            raise WorkingMemoryIntegrityConflict(
                "HypothesisExecutionDecisionRecord 引用了未知 candidate_ref"
            )
        if candidate.scientific_record_id in selected_by_decision:
            raise ContractValidationError(
                "HypothesisExecutionDecisionRecord 不能重复覆盖同一候选"
            )
        selected_by_decision[candidate.scientific_record_id] = item
    if set(selected_by_decision) != {
        candidate.scientific_record_id for candidate in candidates
    }:
        raise ContractValidationError(
            "HypothesisExecutionDecisionRecord 必须恰好覆盖当前批次全部候选"
        )

    current_record_ids = {record.scientific_record_id for record in records}
    for existing in available_records.values():
        if (
            existing.scientific_record_id not in current_record_ids
            and existing.scientific_record_type
            == "HypothesisExecutionDecisionRecord"
            and (
                existing.payload.get(batch_field) == batch_ref
                or existing.payload["confirmation_id"]
                == decision_payload["confirmation_id"]
            )
        ):
            raise IdempotencyConflict(
                "同一 Hypothesis Batch 或 confirmation_id 已经确认，不能写入第二组决定"
            )

    material_by_ref: dict[str, ScientificRecord] = {}
    for material in materials:
        material_id = material.payload["material_id"]
        previous = material_by_ref.get(material_id)
        if (
            previous is not None
            and previous.scientific_record_id != material.scientific_record_id
        ):
            raise ContractValidationError(
                "ResearchMaterialRecord.material_id 必须唯一"
            )
        material_by_ref[material_id] = material
        # 允许调用方在材料引用中使用记录身份；主合同使用 material_id，二者
        # 相同时自然退化为同一引用，不会形成第二个材料。
        previous = material_by_ref.get(material.scientific_record_id)
        if (
            previous is not None
            and previous.scientific_record_id != material.scientific_record_id
        ):
            raise WorkingMemoryIntegrityConflict(
                "ResearchMaterialRecord 的 record_id 与 material_id 发生歧义"
            )
        material_by_ref[material.scientific_record_id] = material

    inputs_by_id = {record.scientific_record_id: record for record in inputs}
    execution_input_refs = decision_payload[execution_inputs_field]
    if set(execution_input_refs) != set(inputs_by_id):
        raise ContractValidationError(
            "Decision.execution_input_refs 必须精确覆盖本次全部 Input 记录"
        )
    input_selected: dict[str, ScientificRecord] = {}
    referenced_material_records: set[str] = set()
    for input_record in inputs:
        payload = input_record.payload
        candidate = candidate_by_ref.get(payload[selected_hypothesis_field])
        if candidate is None:
            raise WorkingMemoryIntegrityConflict(
                "HypothesisExecutionInputRecord 引用了未知 SelectedHypothesisRecord"
            )
        if candidate.scientific_record_id in input_selected:
            raise ContractValidationError(
                "HypothesisExecutionInputRecord 不能重复绑定同一候选"
            )
        input_selected[candidate.scientific_record_id] = input_record
        decision_item = selected_by_decision.get(candidate.scientific_record_id)
        if decision_item is None or decision_item["decision"] != "execute":
            raise ContractValidationError(
                "每条 HypothesisExecutionInputRecord 必须对应一条 execute 决定"
            )
        if payload["execution_rank"] != decision_item["execution_rank"]:
            raise ContractValidationError(
                "HypothesisExecutionInputRecord.execution_rank 与 Decision 不一致"
            )

        expected_source_refs = {
            decision.scientific_record_id,
            candidate.scientific_record_id,
        }
        for material_ref in payload[material_refs_field]:
            material = material_by_ref.get(material_ref)
            if material is None:
                raise WorkingMemoryIntegrityConflict(
                    "HypothesisExecutionInputRecord 引用了未知 ResearchMaterialRecord"
                )
            expected_source_refs.add(material.scientific_record_id)
            referenced_material_records.add(material.scientific_record_id)
        if set(input_record.source_scientific_record_ids) != expected_source_refs:
            raise ContractValidationError(
                "HypothesisExecutionInputRecord.source_scientific_record_ids 必须精确引用 Decision、原 SelectedHypothesisRecord 和声明的 Material"
            )

    execute_candidates = {
        candidate_id
        for candidate_id, item in selected_by_decision.items()
        if item["decision"] == "execute"
    }
    if execute_candidates != set(input_selected):
        raise ContractValidationError(
            "Decision 的 execute 候选与 HypothesisExecutionInputRecord 必须一一对应"
        )
    if len(execute_candidates) != len(inputs):  # pragma: no cover - 上面已核对集合
        raise ContractValidationError("execute 决定与 Input 数量不一致")

    ordered_inputs = sorted(
        inputs,
        key=lambda record: record.payload["execution_rank"],
    )
    expected_ranks = list(range(1, len(inputs) + 1))
    if [record.payload["execution_rank"] for record in ordered_inputs] != expected_ranks:
        raise ContractValidationError(
            "HypothesisExecutionInputRecord.execution_rank 必须从 1 连续排列"
        )
    if list(execution_input_refs) != [
        record.scientific_record_id for record in ordered_inputs
    ]:
        raise ContractValidationError(
            "Decision.execution_input_refs 必须按 execution_rank 有序排列"
        )
    if set(record.payload["execution_rank"] for record in inputs) != {
        item["execution_rank"]
        for item in decision_payload["decisions"]
        if item["decision"] == "execute"
    }:
        raise ContractValidationError("Decision 的 execution_rank 与 Input 不一致")
    if {
        record.scientific_record_id for record in materials
    } != referenced_material_records:
        raise ContractValidationError(
            "Confirmation 输出中的 Material 必须全部由至少一个 Input 声明引用"
        )


def _validate_module_output_group(
    module_kind: str,
    records: Sequence["ScientificRecord"],
    available_records: Mapping[str, "ScientificRecord"],
) -> None:
    """核对一个 Module Run 的记录类型和互斥成组输出合同。"""

    record_types = [record.scientific_record_type for record in records]
    family = _module_kind_family(module_kind)
    if family == "retrieval":
        if record_types.count("RetrievalRecord") != 1:
            raise ContractValidationError(
                "retrieval Module Run 必须恰好输出一个 RetrievalRecord"
            )
    elif family == "research-query-formulation":
        if record_types.count("QueryRecord") != 1:
            raise ContractValidationError(
                "query Module Run 必须恰好输出一个 QueryRecord"
            )
    elif family == "hypothesis-generation":
        if record_types.count("HypothesisBatchRecord") != 1:
            raise ContractValidationError(
                "hypothesis Module Run 必须恰好输出一个 HypothesisBatchRecord"
            )
    elif family == "hypothesis-import":
        batches = [
            record
            for record in records
            if record.scientific_record_type == "HypothesisBatchRecord"
        ]
        selected_records = [
            record
            for record in records
            if record.scientific_record_type == "SelectedHypothesisRecord"
        ]
        if (
            len(batches) != 1
            or batches[0].scientific_record_schema_version != _AUTHOR_IMPORT_V2_SCHEMA_VERSION
            or set(record_types).difference(
                {"HypothesisBatchRecord", "SelectedHypothesisRecord"}
            )
        ):
            raise ContractValidationError(
                "Hypothesis Import Module Run 必须输出一个 author-import.v2 Batch 和逐假设记录"
            )
        batch = batches[0]
        imported = batch.payload["imported_hypotheses"]
        if len(selected_records) != len(imported):
            raise ContractValidationError(
                "Hypothesis Import Module Run 必须为每条作者假设输出一条逐假设记录"
            )
        expected_by_ref = {
            item["hypothesis_ref"]: (rank, item["hypothesis"])
            for rank, item in enumerate(imported, start=1)
        }
        if {record.payload["hypothesis_ref"] for record in selected_records} != set(
            expected_by_ref
        ):
            raise ContractValidationError(
                "作者导入 Batch 与逐假设记录的身份不一致"
            )
        if sorted(record.payload["rank"] for record in selected_records) != list(
            range(1, len(imported) + 1)
        ):
            raise ContractValidationError(
                "作者导入逐假设记录的 rank 必须从 1 连续排列"
            )
        for selected_record in selected_records:
            if batch.scientific_record_id not in selected_record.source_scientific_record_ids:
                raise ContractValidationError(
                    "作者导入 SelectedHypothesisRecord 必须引用 HypothesisBatchRecord"
                )
            expected_rank, expected_hypothesis = expected_by_ref[
                selected_record.payload["hypothesis_ref"]
            ]
            if selected_record.payload["rank"] != expected_rank:
                raise ContractValidationError(
                    "作者导入 Batch 顺序与逐假设记录的 rank 不一致"
                )
            if selected_record.payload["hypothesis"] != expected_hypothesis:
                raise ContractValidationError(
                    "作者导入 Batch 与逐假设记录的假设内容不一致"
                )
    elif family == "hypothesis-selection":
        selections = [
            record
            for record in records
            if record.scientific_record_type == "HypothesisSelectionRecord"
        ]
        selected_records = [
            record
            for record in records
            if record.scientific_record_type == "SelectedHypothesisRecord"
        ]
        if len(selections) != 1:
            raise ContractValidationError(
                "Top 5 selection Module Run 必须恰好输出一个 HypothesisSelectionRecord"
            )
        selection = selections[0]
        if selection.scientific_record_schema_version in {2, "2"}:
            if len(selection.payload["selected_hypotheses"]) != 5:
                raise ContractValidationError(
                    "新版 Top 5 selection Module Run 必须恰好五条入选假设"
                )
            if set(record_types).difference(
                {"HypothesisSelectionRecord", "SelectedHypothesisRecord"}
            ):
                raise ContractValidationError(
                    "新版 Top 5 selection Module Run 只能输出选择摘要和逐假设记录"
                )
            expected = {
                (item["hypothesis_ref"], item["rank"])
                for item in selection.payload["selected_hypotheses"]
            }
            actual = {
                (record.payload["hypothesis_ref"], record.payload["rank"])
                for record in selected_records
            }
            if len(selected_records) != len(expected):
                raise ContractValidationError(
                    "新版 Top 5 selection Module Run 必须为每个入选假设输出一条逐假设记录"
                )
            if actual != expected:
                raise ContractValidationError(
                    "选择摘要与逐假设记录的身份和排名不一致"
                )
            for selected_record in selected_records:
                if selection.scientific_record_id not in selected_record.source_scientific_record_ids:
                    raise ContractValidationError(
                        "SelectedHypothesisRecord.source_scientific_record_ids 必须引用选择摘要"
                    )
        elif selected_records:
            raise ContractValidationError(
                "旧版 HypothesisSelectionRecord 不能混入新版逐假设记录"
            )
    elif family == "hypothesis-confirmation":
        _validate_hypothesis_confirmation_output_group(records, available_records)
    elif family == "hypothesis-investigation":
        evidence = [
            record for record in records if record.scientific_record_type == "VerifiedEvidenceRecord"
        ]
        results = [
            record for record in records if record.scientific_record_type == "InvestigationResultRecord"
        ]
        if not evidence or not results or len(evidence) != len(results):
            raise ContractValidationError(
                "investigation Module Run 必须输出数量相等且至少一对 VerifiedEvidenceRecord 和 InvestigationResultRecord"
            )
        evidence_by_id = {record.scientific_record_id: record for record in evidence}
        referenced_evidence: list[str] = []
        for result_record in results:
            evidence_ref = result_record.payload["evidence_ref"]
            evidence_record = evidence_by_id.get(evidence_ref)
            if evidence_record is None:
                raise ContractValidationError(
                    "InvestigationResultRecord.evidence_ref 必须引用同一 bundle 的 VerifiedEvidenceRecord"
                )
            if evidence_record.scientific_record_id not in result_record.source_scientific_record_ids:
                raise ContractValidationError(
                    "InvestigationResultRecord.source_scientific_record_ids 必须引用 VerifiedEvidenceRecord"
                )
            referenced_evidence.append(evidence_ref)
        if len(set(referenced_evidence)) != len(referenced_evidence):
            raise ContractValidationError(
                "每个 VerifiedEvidenceRecord 只能与一个 InvestigationResultRecord 配对"
            )
        if set(referenced_evidence) != set(evidence_by_id):
            raise ContractValidationError(
                "investigation Module Run 的证据和结果必须一一配对"
            )
    elif family == "task-review":
        reviews = [record for record in records if record.scientific_record_type == "ReviewRecord"]
        repairs = [
            record for record in records if record.scientific_record_type == "RepairRequestRecord"
        ]
        finals = [record for record in records if record.scientific_record_type == "FinalAnswerRecord"]
        if len(reviews) != 1:
            raise ContractValidationError("Review Module Run 必须恰好输出一个 ReviewRecord")
        review = reviews[0]
        outcome = review.payload["outcome"]
        if outcome == "RepairRequired":
            if not repairs or finals:
                raise ContractValidationError(
                    "RepairRequired Review 必须输出 ReviewRecord 和至少一个 RepairRequestRecord，不能输出 FinalAnswerRecord"
                )
            for repair in repairs:
                if review.scientific_record_id not in repair.source_scientific_record_ids:
                    raise ContractValidationError(
                        "RepairRequestRecord.source_scientific_record_ids 必须引用 ReviewRecord"
                    )
        elif outcome in {"Accepted", "Unanswerable", "Stopped"}:
            if len(finals) != 1 or repairs:
                raise ContractValidationError(
                    "终态 Review 必须输出 ReviewRecord 和一个 FinalAnswerRecord，不能输出 RepairRequestRecord"
                )
            if review.scientific_record_id not in finals[0].source_scientific_record_ids:
                raise ContractValidationError(
                    "FinalAnswerRecord.source_scientific_record_ids 必须引用 ReviewRecord"
                )
        elif outcome == "RegenerateHypotheses":
            if repairs or finals:
                raise ContractValidationError(
                    "RegenerateHypotheses Review 不得同时输出 RepairRequestRecord 或 FinalAnswerRecord"
                )
        else:  # pragma: no cover - payload validator 已限制 outcome
            raise ContractValidationError("ReviewRecord.outcome 不是可处理的分支")

    # 对 payload 中的科学引用也做一次关系核对；私有运行身份不在这里解释。
    for record in records:
        payload_refs: list[str] = []
        if record.scientific_record_type == "InvestigationResultRecord":
            payload_refs.append(record.payload["evidence_ref"])
        elif record.scientific_record_type == "ReviewRecord":
            payload_refs.extend(record.payload["reviewed_result_refs"])
        elif record.scientific_record_type == "RepairRequestRecord":
            payload_refs.append(record.payload["result_ref"])
        for source_ref in payload_refs:
            if source_ref not in record.source_scientific_record_ids:
                raise ContractValidationError(
                    f"{record.scientific_record_type}.source_scientific_record_ids 必须列出 payload 引用 {source_ref}"
                )
            if source_ref not in available_records and source_ref not in {
                item.scientific_record_id for item in records
            }:
                raise WorkingMemoryIntegrityConflict(
                    f"{record.scientific_record_type} 的 payload 引用了未知 Scientific Record {source_ref}"
                )


@dataclass(frozen=True, init=False)
class ModuleInvocationRequest:
    """描述一次 canonical Module Invocation 开始请求。

    新构造器只接受 research_task 合同字段。schema-v10 的旧事件必须先由
    ``from_schema_v10_mapping`` 读取；旧身份和 runtime 只在该只读 parser 中出现。
    """

    research_task_id: str
    module_invocation_id: str
    module_kind: str
    input_scientific_record_ids: tuple[str, ...]
    module_input_digest: str
    context_binding: ContextBinding

    def __init__(
        self,
        *,
        research_task_id: str,
        module_invocation_id: str,
        module_kind: str,
        input_scientific_record_ids: Sequence[str],
        module_input_digest: str,
        context_binding: ContextBinding,
    ) -> None:
        """构造 canonical 请求；旧 task/module/input 参数不会被接受。"""

        object.__setattr__(self, "research_task_id", research_task_id)
        object.__setattr__(self, "module_invocation_id", module_invocation_id)
        object.__setattr__(self, "module_kind", module_kind)
        object.__setattr__(self, "input_scientific_record_ids", input_scientific_record_ids)
        object.__setattr__(self, "module_input_digest", module_input_digest)
        object.__setattr__(self, "context_binding", context_binding)
        self.__post_init__()

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
            _require_text(self.module_invocation_id, "module_invocation_id"),
        )
        object.__setattr__(self, "module_kind", _validate_module_kind(self.module_kind))
        object.__setattr__(
            self,
            "input_scientific_record_ids",
            _require_sequence(
                self.input_scientific_record_ids,
                "input_scientific_record_ids",
            ),
        )
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
            raise ContractValidationError("context_binding 不是已注册的 V2 owner 类型")

    @classmethod
    def from_schema_v10_mapping(cls, value: Mapping[str, Any]) -> Self:
        """只读解析 schema-v10 的旧 ModuleRunStarted payload。

        旧 runtime 只用于判断是否存在可转换的长期 owner；它不会进入当前
        Module Invocation。没有明确 ``context_binding`` 的有状态模块直接拒绝，
        防止从 ``runtime_ref`` 猜测恢复身份。
        """

        if not isinstance(value, Mapping):
            raise ContractValidationError("ModuleRun request 必须是 JSON object")
        context_value = value.get("context_binding")
        raw_kind = _require_text(value.get("module_kind"), "module_kind")
        legacy_modules = {
            "retrieval": "retrieval",
            "query": "research-query-formulation",
            "research-query-formulation": "research-query-formulation",
            "hypothesis": "hypothesis-generation",
            "hypotheses": "hypothesis-generation",
            "hypothesis-generation": "hypothesis-generation",
            "hypothesis-import": "hypothesis-import",
            "hypothesis_confirmation": "hypothesis-confirmation",
            "hypothesis-confirmation": "hypothesis-confirmation",
            "confirmation": "hypothesis-confirmation",
            "selection": "hypothesis-selection",
            "top5": "hypothesis-selection",
            "top-5": "hypothesis-selection",
            "top5-selection": "hypothesis-selection",
            "top-5-selection": "hypothesis-selection",
            "investigation": "hypothesis-investigation",
            "investigation-and-experiment": "hypothesis-investigation",
            "investigation-experiment": "hypothesis-investigation",
            "review": "task-review",
            "task-review": "task-review",
        }
        normalized_kind = raw_kind.casefold().replace("_", "-").replace(" ", "-")
        try:
            module_kind = legacy_modules[normalized_kind]
        except KeyError as exc:
            raise ContractValidationError(f"schema-v10 module_kind 未注册：{raw_kind}") from exc
        research_task_id = _coalesce_alias(
            value,
            "research_task_id",
            "inquiry_id",
            "research_task_id",
        )
        if research_task_id is None:
            research_task_id = value.get("task_id")
        module_invocation_id = _coalesce_alias(
            value,
            "module_invocation_id",
            "module_run_id",
            "module_invocation_id",
        )
        module_input_digest = _coalesce_alias(
            value,
            "module_input_digest",
            "input_digest",
            "module_input_digest",
        )
        input_refs = _coalesce_alias(
            value,
            "input_scientific_record_ids",
            "input_record_refs",
            "input_scientific_record_ids",
        )
        if research_task_id is None:
            raise ContractValidationError("schema-v10 ModuleRun request 缺少 task_id")
        context_binding = cls._context_binding_from_schema_v10(context_value)
        if context_binding is None:
            if module_kind in {"hypothesis-investigation", "task-review"}:
                raise ContractValidationError(
                    "schema-v10 ModuleRun 缺少可转换的 context_binding owner"
                )
            context_binding = StatelessModule()
        return cls(
            research_task_id=research_task_id,
            module_invocation_id=module_invocation_id,
            module_kind=module_kind,
            input_scientific_record_ids=tuple(input_refs or ()),
            module_input_digest=module_input_digest,
            context_binding=context_binding,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        """从 canonical Module Invocation mapping 构造请求。"""

        if not isinstance(value, Mapping):
            raise ContractValidationError("ModuleRun request 必须是 JSON object")
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
            raise ContractValidationError(
                "ModuleRun request 含非 canonical 字段：" + ", ".join(extra)
            )
        missing = sorted(allowed.difference(value))
        if missing:
            raise ContractValidationError(
                "ModuleRun request 缺少 " + ", ".join(missing)
            )
        context_value = value["context_binding"]
        if not isinstance(context_value, Mapping):
            raise ContractValidationError("context_binding 必须是 typed object")
        kind = context_value.get("kind")
        try:
            if kind == "StatelessModule":
                context_binding: ContextBinding = StatelessModule()
            elif kind == "CreateInvestigationContext":
                context_binding = CreateInvestigationContext(
                    context_value["investigation_context_id"],
                    context_value["hypothesis_batch_record_id"],
                )
            elif kind == "ReuseInvestigationContext":
                context_binding = ReuseInvestigationContext(
                    context_value["investigation_context_id"],
                    context_value["hypothesis_batch_record_id"],
                    context_value["source_module_invocation_id"],
                )
            elif kind == "CreateResearchTaskReviewSession":
                context_binding = CreateResearchTaskReviewSession(
                    context_value["research_task_review_session_id"],
                )
            elif kind == "ReuseResearchTaskReviewSession":
                context_binding = ReuseResearchTaskReviewSession(
                    context_value["research_task_review_session_id"],
                    context_value["source_module_invocation_id"],
                )
            else:
                raise ContractValidationError(f"context_binding.kind 未注册：{kind}")
        except KeyError as exc:
            raise ContractValidationError(f"context_binding 缺少 {exc.args[0]}") from exc
        return cls(
            research_task_id=value["research_task_id"],
            module_invocation_id=value["module_invocation_id"],
            module_kind=value["module_kind"],
            input_scientific_record_ids=tuple(value["input_scientific_record_ids"]),
            module_input_digest=value["module_input_digest"],
            context_binding=context_binding,
        )

    @staticmethod
    def _context_binding_from_schema_v10(
        value: Any,
    ) -> ContextBinding | None:
        """把旧事件中的显式 owner 转换为 typed context_binding。"""

        if not isinstance(value, Mapping):
            return None
        kind = value.get("kind", value.get("type"))
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
        if kind in {
            None,
            "CreateRuntime",
            "ReuseRuntime",
            "create",
            "reuse",
        }:
            return None
        raise ContractValidationError(f"schema-v10 context kind 未注册：{kind}")

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical Module Invocation mapping，不携带 runtime_ref。"""

        return {
            "research_task_id": self.research_task_id,
            "module_invocation_id": self.module_invocation_id,
            "module_kind": self.module_kind,
            "input_scientific_record_ids": list(self.input_scientific_record_ids),
            "module_input_digest": self.module_input_digest,
            "context_binding": _public_context_binding(
                self.module_kind,
                self.context_binding,
            ),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """返回字段重构后的 Module Invocation envelope。"""

        return self.to_dict()


@dataclass(frozen=True, init=False)
class ScientificRecord(Mapping[str, Any]):
    """Working Memory V2 的 canonical、不可变科学记录 envelope。"""

    scientific_record_id: str
    research_task_id: str
    scientific_record_type: str
    scientific_record_schema_version: str | int
    producer_module_invocation_id: str
    source_scientific_record_ids: tuple[str, ...]
    payload: Mapping[str, Any]
    payload_digest: str
    recorded_at: str | None = None

    def __init__(
        self,
        *,
        scientific_record_id: str,
        research_task_id: str,
        scientific_record_type: str,
        scientific_record_schema_version: str | int,
        producer_module_invocation_id: str,
        source_scientific_record_ids: Sequence[str],
        payload: Mapping[str, Any],
        payload_digest: str,
        recorded_at: str | None = None,
    ) -> None:
        """构造 canonical 科学记录；旧字段只由 schema-v10 parser 读取。"""

        object.__setattr__(self, "scientific_record_id", scientific_record_id)
        object.__setattr__(self, "research_task_id", research_task_id)
        object.__setattr__(self, "scientific_record_type", scientific_record_type)
        object.__setattr__(
            self,
            "scientific_record_schema_version",
            scientific_record_schema_version,
        )
        object.__setattr__(
            self,
            "producer_module_invocation_id",
            producer_module_invocation_id,
        )
        object.__setattr__(
            self,
            "source_scientific_record_ids",
            source_scientific_record_ids,
        )
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_digest", payload_digest)
        object.__setattr__(self, "recorded_at", recorded_at)
        self.__post_init__()

    def __post_init__(self) -> None:
        """规范化身份、来源和 payload，并核对 payload 摘要。"""

        object.__setattr__(
            self,
            "scientific_record_id",
            _require_text(self.scientific_record_id, "scientific_record_id"),
        )
        object.__setattr__(
            self,
            "research_task_id",
            _require_text(self.research_task_id, "research_task_id"),
        )
        object.__setattr__(
            self,
            "scientific_record_type",
            _require_text(self.scientific_record_type, "scientific_record_type"),
        )
        schema_version = _validate_record_schema_version(
            self.scientific_record_schema_version
        )
        object.__setattr__(self, "scientific_record_schema_version", schema_version)
        object.__setattr__(
            self,
            "producer_module_invocation_id",
            _require_text(
                self.producer_module_invocation_id,
                "producer_module_invocation_id",
            ),
        )
        refs = _require_sequence(
            self.source_scientific_record_ids,
            "source_scientific_record_ids",
        )
        object.__setattr__(self, "source_scientific_record_ids", refs)
        normalized_payload = _validate_scientific_payload(
            self.scientific_record_type,
            self.payload,
            self.scientific_record_schema_version,
        )
        object.__setattr__(self, "payload", normalized_payload)
        digest = _validate_sha256(self.payload_digest, "payload_digest")
        if digest != _payload_digest(normalized_payload):
            raise ContractValidationError("payload_digest 与 payload 内容不一致")
        object.__setattr__(self, "payload_digest", digest)
        if self.recorded_at is not None:
            object.__setattr__(self, "recorded_at", _require_text(self.recorded_at, "recorded_at"))

    @classmethod
    def from_payload(
        cls,
        *,
        scientific_record_id: str,
        research_task_id: str,
        scientific_record_type: str,
        scientific_record_schema_version: str | int,
        producer_module_invocation_id: str,
        source_scientific_record_ids: Sequence[str],
        payload: Mapping[str, Any],
        payload_digest: str | None = None,
        recorded_at: str | None = None,
    ) -> "ScientificRecord":
        """从科学 payload 构造记录并自动计算摘要。

        生产构造器只接受 canonical 字段；schema-v10 的旧 mapping 必须先经
        ``from_schema_v10_mapping`` 只读解析。
        """
        normalized_payload = _validate_scientific_payload(
            scientific_record_type,
            payload,
            scientific_record_schema_version,
        )
        digest = _payload_digest(normalized_payload)
        if payload_digest is not None and payload_digest != digest:
            raise ContractValidationError("payload_digest 与 payload 内容不一致")
        return cls(
            scientific_record_id=scientific_record_id,
            research_task_id=research_task_id,
            scientific_record_type=scientific_record_type,
            scientific_record_schema_version=scientific_record_schema_version,
            producer_module_invocation_id=producer_module_invocation_id,
            source_scientific_record_ids=tuple(source_scientific_record_ids),
            payload=normalized_payload,
            payload_digest=digest,
            recorded_at=recorded_at,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScientificRecord":
        """从 canonical 科学记录 mapping 构造严格记录。"""

        if not isinstance(value, Mapping):
            raise ContractValidationError("ScientificRecord 必须是 JSON object")
        allowed = {
            "scientific_record_id",
            "research_task_id",
            "scientific_record_type",
            "scientific_record_schema_version",
            "producer_module_invocation_id",
            "source_scientific_record_ids",
            "payload",
            "payload_digest",
            "recorded_at",
        }
        extra = sorted(set(value).difference(allowed))
        if extra:
            raise ContractValidationError(
                f"ScientificRecord 含非 canonical 字段：{', '.join(extra)}"
            )
        required = allowed - {"recorded_at"}
        missing = sorted(required.difference(value))
        if missing:
            raise ContractValidationError(
                "ScientificRecord 缺少 " + ", ".join(missing)
            )
        return cls(
            scientific_record_id=value["scientific_record_id"],
            research_task_id=value["research_task_id"],
            scientific_record_type=value["scientific_record_type"],
            scientific_record_schema_version=value["scientific_record_schema_version"],
            producer_module_invocation_id=value["producer_module_invocation_id"],
            source_scientific_record_ids=tuple(value["source_scientific_record_ids"]),
            payload=value["payload"],
            payload_digest=value["payload_digest"],
            recorded_at=value.get("recorded_at"),
        )

    @classmethod
    def from_schema_v10_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        research_task_id: str | None = None,
        recorded_at: str | None = None,
    ) -> "ScientificRecord":
        """只读解析 schema-v10 ScientificRecord mapping。"""

        if not isinstance(value, Mapping):
            raise ContractValidationError("schema-v10 ScientificRecord 必须是 JSON object")
        if research_task_id is not None:
            value = {"research_task_id": research_task_id, **value}
        record_id = _coalesce_alias(
            value,
            "scientific_record_id",
            "record_id",
            "scientific_record_id",
        )
        task_id = _coalesce_alias(
            value,
            "research_task_id",
            "task_id",
            "research_task_id",
        )
        if task_id is None:
            task_id = value.get("inquiry_id")
        record_type = _coalesce_alias(
            value,
            "scientific_record_type",
            "record_type",
            "scientific_record_type",
        )
        schema_version = _coalesce_alias(
            value,
            "scientific_record_schema_version",
            "record_schema_version",
            "scientific_record_schema_version",
        )
        if schema_version is None:
            schema_version = value.get("schema_version")
        producer = _coalesce_alias(
            value,
            "producer_module_invocation_id",
            "producer_module_run_id",
            "producer_module_invocation_id",
        )
        source_refs = _coalesce_alias(
            value,
            "source_scientific_record_ids",
            "source_record_refs",
            "source_scientific_record_ids",
        )
        try:
            return cls(
                scientific_record_id=record_id,
                research_task_id=task_id,
                scientific_record_type=record_type,
                scientific_record_schema_version=schema_version,
                producer_module_invocation_id=producer,
                source_scientific_record_ids=tuple(source_refs or ()),
                payload=value["payload"],
                payload_digest=value["payload_digest"],
                recorded_at=value.get("recorded_at", recorded_at),
            )
        except KeyError as exc:
            raise ContractValidationError(
                f"schema-v10 ScientificRecord 缺少 {exc.args[0]}"
            ) from exc

    def canonical_identity_dict(self) -> dict[str, Any]:
        """返回不包含审计时间的 canonical 科学记录身份。"""

        return {
            "scientific_record_id": self.scientific_record_id,
            "research_task_id": self.research_task_id,
            "scientific_record_type": self.scientific_record_type,
            "scientific_record_schema_version": self.scientific_record_schema_version,
            "producer_module_invocation_id": self.producer_module_invocation_id,
            "source_scientific_record_ids": list(self.source_scientific_record_ids),
            "payload": _copy_mapping(self.payload),
            "payload_digest": self.payload_digest,
        }

    def with_recorded_at(self, recorded_at: str) -> "ScientificRecord":
        """补充由 Working Memory 产生的审计时间。"""

        return ScientificRecord(
            scientific_record_id=self.scientific_record_id,
            research_task_id=self.research_task_id,
            scientific_record_type=self.scientific_record_type,
            scientific_record_schema_version=self.scientific_record_schema_version,
            producer_module_invocation_id=self.producer_module_invocation_id,
            source_scientific_record_ids=self.source_scientific_record_ids,
            payload=self.payload,
            payload_digest=self.payload_digest,
            recorded_at=recorded_at,
        )

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问。"""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按公共 envelope 字段顺序迭代。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回公共 envelope 字段数量。"""

        return 9

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical 科学记录 mapping，不混入旧账本字段。"""

        return {**self.canonical_identity_dict(), "recorded_at": self.recorded_at}

    def to_public_dict(self) -> dict[str, Any]:
        """返回 canonical ScientificRecord envelope。"""

        return self.to_dict()


@dataclass(frozen=True, init=False)
class PreparedModuleInvocationOutput(Mapping[str, Any]):
    """模块在返回成功前持久化的 canonical Module Invocation 输出包。"""

    module_invocation_id: str
    module_input_digest: str
    output_scientific_records: tuple[ScientificRecord, ...]
    module_output_digest: str

    def __init__(
        self,
        *,
        module_invocation_id: str,
        module_input_digest: str,
        output_scientific_records: Sequence[ScientificRecord | Mapping[str, Any]],
        module_output_digest: str,
    ) -> None:
        """构造 canonical 输出包；旧字段只由 schema-v10 parser 读取。"""

        object.__setattr__(self, "module_invocation_id", module_invocation_id)
        object.__setattr__(self, "module_input_digest", module_input_digest)
        object.__setattr__(self, "output_scientific_records", output_scientific_records)
        object.__setattr__(self, "module_output_digest", module_output_digest)
        self.__post_init__()

    def __post_init__(self) -> None:
        """规范化输出记录并核对整个输出包摘要。"""

        object.__setattr__(
            self,
            "module_invocation_id",
            _require_text(self.module_invocation_id, "module_invocation_id"),
        )
        object.__setattr__(
            self,
            "module_input_digest",
            _require_text(self.module_input_digest, "module_input_digest"),
        )
        if isinstance(self.output_scientific_records, (str, bytes)) or not isinstance(
            self.output_scientific_records, Sequence
        ):
            raise ContractValidationError(
                "output_scientific_records 必须是 ScientificRecord 序列"
            )
        records: list[ScientificRecord] = []
        for index, raw_record in enumerate(self.output_scientific_records):
            if isinstance(raw_record, ScientificRecord):
                record = raw_record
            elif isinstance(raw_record, Mapping):
                record = ScientificRecord.from_mapping(raw_record)
            else:
                raise ContractValidationError(
                    f"output_scientific_records[{index}] 不是 ScientificRecord"
                )
            if record.producer_module_invocation_id != self.module_invocation_id:
                raise ContractValidationError(
                    "output_scientific_records 的 producer_module_invocation_id "
                    "必须匹配 module_invocation_id"
                )
            records.append(record)
        if not records:
            raise ContractValidationError("output_scientific_records 不能为空")
        if len({record.scientific_record_id for record in records}) != len(records):
            raise ContractValidationError("output_scientific_records 的记录身份必须唯一")
        normalized = tuple(records)
        object.__setattr__(self, "output_scientific_records", normalized)
        digest = _validate_sha256(self.module_output_digest, "module_output_digest")
        if digest != _prepared_output_digest(normalized):
            raise ContractValidationError(
                "module_output_digest 与 output_scientific_records 内容不一致"
            )
        object.__setattr__(self, "module_output_digest", digest)

    @classmethod
    def from_records(
        cls,
        *,
        module_invocation_id: str,
        module_input_digest: str,
        output_scientific_records: Sequence[ScientificRecord | Mapping[str, Any]],
    ) -> "PreparedModuleInvocationOutput":
        """从记录序列构造密封输出并自动计算摘要。"""
        provisional = tuple(
            record
            if isinstance(record, ScientificRecord)
            else ScientificRecord.from_mapping(record)
            for record in output_scientific_records
        )
        return cls(
            module_invocation_id=module_invocation_id,
            module_input_digest=module_input_digest,
            output_scientific_records=provisional,
            module_output_digest=_prepared_output_digest(provisional),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreparedModuleInvocationOutput":
        """从 canonical 输出包 mapping 恢复密封对象。"""

        if not isinstance(value, Mapping):
            raise ContractValidationError("PreparedModuleInvocationOutput 必须是 JSON object")
        allowed = {
            "module_invocation_id",
            "module_input_digest",
            "output_scientific_records",
            "module_output_digest",
        }
        extra = sorted(set(value).difference(allowed))
        if extra:
            raise ContractValidationError(
                "PreparedModuleInvocationOutput 含非 canonical 字段：" + ", ".join(extra)
            )
        missing = sorted(allowed.difference(value))
        if missing:
            raise ContractValidationError(
                "PreparedModuleInvocationOutput 缺少 " + ", ".join(missing)
            )
        return cls(
            module_invocation_id=value["module_invocation_id"],
            module_input_digest=value["module_input_digest"],
            output_scientific_records=tuple(value["output_scientific_records"]),
            module_output_digest=value["module_output_digest"],
        )

    @classmethod
    def from_schema_v10_mapping(cls, value: Mapping[str, Any]) -> "PreparedModuleInvocationOutput":
        """只读解析 schema-v10 输出包及其中的旧 ScientificRecord。"""

        if not isinstance(value, Mapping):
            raise ContractValidationError("schema-v10 PreparedModuleInvocationOutput 必须是 JSON object")
        invocation_id = value.get("module_invocation_id", value.get("module_run_id"))
        input_digest = value.get("module_input_digest", value.get("input_digest"))
        raw_records = value.get(
            "output_scientific_records",
            value.get("output_records"),
        )
        output_digest = value.get("module_output_digest", value.get("output_digest"))
        if invocation_id is None or input_digest is None or raw_records is None or output_digest is None:
            raise ContractValidationError("schema-v10 PreparedModuleInvocationOutput 字段不完整")
        records = tuple(
            record
            if isinstance(record, ScientificRecord)
            else (
                ScientificRecord.from_mapping(record)
                if "scientific_record_id" in record
                else ScientificRecord.from_schema_v10_mapping(record)
            )
            for record in raw_records
        )
        return cls(
            module_invocation_id=invocation_id,
            module_input_digest=input_digest,
            output_scientific_records=records,
            module_output_digest=output_digest,
        )

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问。"""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按输出包字段顺序迭代。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回输出包字段数量。"""

        return 4

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical Module Invocation 输出 mapping。"""

        return {
            "module_invocation_id": self.module_invocation_id,
            "module_input_digest": self.module_input_digest,
            "output_scientific_records": [
                record.to_dict() for record in self.output_scientific_records
            ],
            "module_output_digest": self.module_output_digest,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """返回 canonical Module Invocation 输出 envelope。"""

        return self.to_dict()


@dataclass(frozen=True)
class ResearchTaskNotStarted:
    """表示没有对应的开始事实。"""

    state: str = field(init=False, default="NotStarted")

    def to_dict(self) -> dict[str, str]:
        """返回公开生命周期结构。"""

        return {"state": self.state}

@dataclass(frozen=True)
class ModuleInvocationOpen:
    """表示开始事实已保存但尚无结束事实。"""

    start_event_ref: str
    state: str = field(init=False, default="Open")

    def to_dict(self) -> dict[str, str]:
        """返回公开生命周期结构。"""

        return {"state": self.state, "start_event_ref": self.start_event_ref}

def _canonical_lifecycle_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """将历史生命周期结果投影为 research_task 公开字段。"""

    aliases = {
        "module_run_id": "module_invocation_id",
        "output_record_refs": "output_scientific_record_ids",
        "output_digest": "module_output_digest",
        "review_record_ref": "review_scientific_record_id",
        "final_answer_ref": "final_answer_scientific_record_id",
        "selection_record_ref": "selection_scientific_record_id",
    }
    return {
        aliases.get(key, key): _copy_json(value, f"lifecycle.outcome.{key}")
        for key, value in outcome.items()
    }


@dataclass(frozen=True)
class ModuleInvocationClosed:
    """表示开始和结束事实都已保存。"""

    start_event_ref: str
    closure_event_ref: str
    outcome: Mapping[str, Any]
    state: str = field(init=False, default="Closed")

    def to_dict(self) -> dict[str, Any]:
        """返回公开生命周期结构。"""

        return {
            "state": self.state,
            "start_event_ref": self.start_event_ref,
            "closure_event_ref": self.closure_event_ref,
            "outcome": _canonical_lifecycle_outcome(self.outcome),
        }


@dataclass(frozen=True)
class ResearchTaskOpen:
    """Task 已有 TaskCreated，但尚无终态事实。"""

    start_event_ref: str
    state: str = field(init=False, default="Open")

    def to_dict(self) -> dict[str, str]:
        """返回 Task 生命周期结构。"""

        return {"state": self.state, "start_event_ref": self.start_event_ref}


@dataclass(frozen=True)
class ResearchTaskClosed:
    """Task 已保存明确终态。"""

    closure_event_ref: str
    outcome: Mapping[str, Any]
    state: str = field(init=False, default="Closed")

    def to_dict(self) -> dict[str, Any]:
        """返回 Task 生命周期结构。"""

        return {
            "state": self.state,
            "closure_event_ref": self.closure_event_ref,
            "outcome": _canonical_lifecycle_outcome(self.outcome),
        }


@dataclass(frozen=True)
class NoOpenModuleInvocation:
    """当前 Task 没有未闭合的 Module Run。"""

    state: str = field(init=False, default="NoOpenModuleInvocation")

    def to_dict(self) -> dict[str, str]:
        """返回公开的活动模块索引。"""

        return {"state": self.state}


@dataclass(frozen=True)
class OpenModuleInvocation:
    """当前 research_task 唯一未闭合的 Module Invocation。"""

    module_invocation_id: str
    state: str = field(init=False, default="Open")

    def to_dict(self) -> dict[str, str]:
        """返回公开的活动模块索引。"""

        return {
            "state": self.state,
            "module_invocation_id": self.module_invocation_id,
        }


@dataclass(frozen=True, init=False)
class ResearchTaskSnapshot(Mapping[str, Any]):
    """ResearchTaskCreated 中冻结的完整 research_task 子结构。"""

    research_task_id: str
    contract_version: str | int
    input: Mapping[str, Any]
    policy: Mapping[str, Any]
    dependencies: Mapping[str, Any]

    def __init__(
        self,
        *,
        research_task_id: str,
        contract_version: str | int,
        input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> None:
        """构造 canonical research_task 快照。"""

        object.__setattr__(self, "research_task_id", research_task_id)
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "dependencies", dependencies)
        self.__post_init__()

    def __post_init__(self) -> None:
        """校验并冻结 TaskCreated 的身份、版本和 JSON 子结构。"""

        object.__setattr__(
            self,
            "research_task_id",
            _require_text(self.research_task_id, "research_task_id"),
        )
        contract_version = self.contract_version
        if isinstance(contract_version, bool) or not isinstance(
            contract_version, (str, int)
        ):
            raise ContractValidationError("contract_version 必须是字符串或整数")
        if isinstance(contract_version, str):
            contract_version = contract_version.strip()
            if not contract_version:
                raise ContractValidationError("contract_version 不能为空")
        elif contract_version <= 0:
            raise ContractValidationError("contract_version 必须大于 0")
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "input", _require_object(self.input, "input"))
        object.__setattr__(self, "policy", _require_object(self.policy, "policy"))
        object.__setattr__(
            self,
            "dependencies",
            _require_object(self.dependencies, "dependencies"),
        )

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问，仍只暴露 TaskCreated 字段。"""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按公开字段顺序迭代快照键。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回公开快照字段数。"""

        return 5

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical research_task 快照。"""

        return {
            "research_task_id": self.research_task_id,
            "contract_version": self.contract_version,
            "input": _copy_mapping(self.input),
            "policy": _copy_mapping(self.policy),
            "dependencies": _copy_mapping(self.dependencies),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """返回字段重构后的 research_task 公开 envelope。"""

        return {
            "research_task_id": self.research_task_id,
            "contract_version": self.contract_version,
            "scientific_input": _copy_mapping(self.input),
            "execution_policy": _copy_mapping(self.policy),
            "frozen_dependencies": _copy_mapping(self.dependencies),
        }


@dataclass(frozen=True, init=False)
class ModuleInvocationSnapshot(Mapping[str, Any]):
    """一次 Module Invocation 的不可变快照。"""

    research_task_id: str
    module_invocation_id: str
    module_kind: str
    input_scientific_record_ids: tuple[str, ...]
    module_input_digest: str
    lifecycle: ModuleInvocationOpen | ModuleInvocationClosed
    start_event_ref: str
    context_binding: ContextBinding

    def __init__(
        self,
        *,
        research_task_id: str,
        module_invocation_id: str,
        module_kind: str,
        input_scientific_record_ids: Sequence[str],
        module_input_digest: str,
        lifecycle: ModuleInvocationOpen | ModuleInvocationClosed,
        start_event_ref: str,
        context_binding: ContextBinding,
    ) -> None:
        """构造 canonical Module Invocation 快照；旧字段不会进入活动构造器。"""

        object.__setattr__(self, "research_task_id", research_task_id)
        object.__setattr__(self, "module_invocation_id", module_invocation_id)
        object.__setattr__(self, "module_kind", module_kind)
        object.__setattr__(self, "input_scientific_record_ids", input_scientific_record_ids)
        object.__setattr__(self, "module_input_digest", module_input_digest)
        object.__setattr__(self, "lifecycle", lifecycle)
        object.__setattr__(self, "start_event_ref", start_event_ref)
        object.__setattr__(self, "context_binding", context_binding)
        self.__post_init__()

    def __post_init__(self) -> None:
        """校验模块快照的稳定身份、输入和生命周期。"""

        object.__setattr__(
            self,
            "research_task_id",
            _require_text(self.research_task_id, "research_task_id"),
        )
        object.__setattr__(
            self,
            "module_invocation_id",
            _require_text(self.module_invocation_id, "module_invocation_id"),
        )
        object.__setattr__(self, "module_kind", _validate_module_kind(self.module_kind))
        object.__setattr__(
            self,
            "input_scientific_record_ids",
            _require_sequence(
                self.input_scientific_record_ids,
                "input_scientific_record_ids",
            ),
        )
        object.__setattr__(
            self,
            "module_input_digest",
            _require_text(self.module_input_digest, "module_input_digest"),
        )
        if not isinstance(self.lifecycle, (ModuleInvocationOpen, ModuleInvocationClosed)):
            raise ContractValidationError("lifecycle 必须是 ModuleInvocationOpen 或 ModuleInvocationClosed")
        object.__setattr__(
            self,
            "start_event_ref",
            _require_text(self.start_event_ref, "start_event_ref"),
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
            raise ContractValidationError("context_binding 不是已注册的 V2 owner 类型")

    @classmethod
    def _lifecycle_from_mapping(cls, value: Any) -> ModuleInvocationOpen | ModuleInvocationClosed:
        """从 canonical 或历史生命周期 mapping 构造不可变生命周期。"""

        if isinstance(value, (ModuleInvocationOpen, ModuleInvocationClosed)):
            return value
        if not isinstance(value, Mapping):
            raise ContractValidationError("lifecycle 必须是 typed object")
        state = value.get("state")
        if state == "Open":
            return ModuleInvocationOpen(start_event_ref=value.get("start_event_ref"))
        if state == "Closed":
            return ModuleInvocationClosed(
                start_event_ref=value.get("start_event_ref"),
                closure_event_ref=value.get("closure_event_ref"),
                outcome=value.get("outcome"),
            )
        raise ContractValidationError(f"lifecycle.state 未注册：{state}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModuleInvocationSnapshot":
        """从 canonical Module Invocation 快照 mapping 恢复对象。"""

        if not isinstance(value, Mapping):
            raise ContractValidationError("ModuleInvocationSnapshot 必须是 JSON object")
        allowed = {
            "research_task_id",
            "module_invocation_id",
            "module_kind",
            "input_scientific_record_ids",
            "module_input_digest",
            "context_binding",
            "lifecycle",
            "start_event_ref",
        }
        extra = sorted(set(value).difference(allowed))
        if extra:
            raise ContractValidationError(
                "ModuleInvocationSnapshot 含非 canonical 字段：" + ", ".join(extra)
            )
        missing = sorted(allowed.difference(value))
        if missing:
            raise ContractValidationError(
                "ModuleInvocationSnapshot 缺少 " + ", ".join(missing)
            )
        request = ModuleInvocationRequest.from_mapping(
            {
                key: value[key]
                for key in (
                    "research_task_id",
                    "module_invocation_id",
                    "module_kind",
                    "input_scientific_record_ids",
                    "module_input_digest",
                    "context_binding",
                )
            }
        )
        return cls(
            research_task_id=request.research_task_id,
            module_invocation_id=request.module_invocation_id,
            module_kind=request.module_kind,
            input_scientific_record_ids=request.input_scientific_record_ids,
            module_input_digest=request.module_input_digest,
            lifecycle=cls._lifecycle_from_mapping(value["lifecycle"]),
            start_event_ref=value["start_event_ref"],
            context_binding=request.context_binding,
        )

    @classmethod
    def from_schema_v10_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        research_task_id: str | None = None,
    ) -> "ModuleInvocationSnapshot":
        """只读解析 schema-v10 ModuleRun 快照；不让旧键进入活动构造器。"""

        if not isinstance(value, Mapping):
            raise ContractValidationError("schema-v10 ModuleInvocationSnapshot 必须是 JSON object")
        request_value: dict[str, Any] = dict(value)
        if research_task_id is not None:
            request_value["research_task_id"] = research_task_id
        request = ModuleInvocationRequest.from_schema_v10_mapping(request_value)
        if "lifecycle" not in value or "start_event_ref" not in value:
            raise ContractValidationError(
                "schema-v10 ModuleInvocationSnapshot 缺少 lifecycle 或 start_event_ref"
            )
        return cls(
            research_task_id=request.research_task_id,
            module_invocation_id=request.module_invocation_id,
            module_kind=request.module_kind,
            input_scientific_record_ids=request.input_scientific_record_ids,
            module_input_digest=request.module_input_digest,
            lifecycle=cls._lifecycle_from_mapping(value["lifecycle"]),
            start_event_ref=value["start_event_ref"],
            context_binding=request.context_binding,
        )

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问，避免调用方接触 SQLite 行。"""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按公开字段顺序迭代模块快照键。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回公开模块快照字段数。"""

        return 8

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical Module Invocation 快照。"""

        return {
            "research_task_id": self.research_task_id,
            "module_invocation_id": self.module_invocation_id,
            "module_kind": self.module_kind,
            "input_scientific_record_ids": list(self.input_scientific_record_ids),
            "module_input_digest": self.module_input_digest,
            "context_binding": _public_context_binding(
                self.module_kind,
                self.context_binding,
            ),
            "lifecycle": self.lifecycle.to_dict(),
            "start_event_ref": self.start_event_ref,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """返回 canonical Module Invocation 公共快照。"""

        return self.to_dict()


@dataclass(frozen=True)
class ScientificGraphFacts:
    """Task Facts 中可重建的科学记录关系索引；Ticket 01 为空图。"""

    record_refs_by_type: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    successor_refs_by_record: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    output_scientific_record_ids_by_module_invocation: Mapping[
        str, tuple[str, ...]
    ] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """返回只读科学关系索引。"""

        return {
            "record_refs_by_type": dict(self.record_refs_by_type),
            "successor_refs_by_record": dict(self.successor_refs_by_record),
            "output_scientific_record_ids_by_module_invocation": dict(
                self.output_scientific_record_ids_by_module_invocation
            ),
        }


@dataclass(frozen=True)
class ResearchTaskFacts(Mapping[str, Any]):
    """从 V2 不可变事件日志归并得到的最小 research_task 事实视图。"""

    research_task: ResearchTaskSnapshot
    research_task_sequence: int
    lifecycle: ResearchTaskOpen | ResearchTaskClosed
    closed_module_invocations: tuple[ModuleInvocationSnapshot, ...]
    active_module_invocation: NoOpenModuleInvocation | OpenModuleInvocation
    open_module_invocation: ModuleInvocationSnapshot | None
    scientific_graph: ScientificGraphFacts

    @property
    def research_task_id(self) -> str:
        """返回 canonical 顶层 research_task 身份。"""

        return self.research_task.research_task_id

    @property
    def module_invocations(self) -> tuple[ModuleInvocationSnapshot, ...]:
        """返回按开始事件顺序排列的全部 Module Invocation。"""

        return self.closed_module_invocations + (
            ()
            if self.open_module_invocation is None
            else (self.open_module_invocation,)
        )

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问，保留 Task Facts 领域结构。"""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按公开字段顺序迭代 Task Facts 键。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回公开 Task Facts 字段数。"""

        return 8

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical Task Facts 公共结构，不暴露 schema 表。"""

        module_invocations = list(self.module_invocations)
        return {
            "research_task": self.research_task.to_dict(),
            "research_task_sequence": self.research_task_sequence,
            "lifecycle": self.lifecycle.to_dict(),
            "module_invocations": [
                invocation.to_dict() for invocation in module_invocations
            ],
            "closed_module_invocations": [
                invocation.to_dict()
                for invocation in self.closed_module_invocations
            ],
            "active_module_invocation": self.active_module_invocation.to_dict(),
            "open_module_invocation": (
                None
                if self.open_module_invocation is None
                else self.open_module_invocation.to_dict()
            ),
            "scientific_graph": self.scientific_graph.to_dict(),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """返回供应用层和 Run Bundle 使用的 research_task 视图。"""

        return self.to_dict()


@dataclass(frozen=True)
class WorkingMemoryEvent(Mapping[str, Any]):
    """V2 账本事件的公共 envelope。"""

    event_id: str
    research_task_id: str
    research_task_sequence: int
    event_type: str
    payload: Mapping[str, Any]
    recorded_at: str
    event_digest: str
    previous_event_digest: str | None

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问，保留完整审计 envelope。"""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按 envelope 字段顺序迭代事件键。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回 envelope 字段数。"""

        return 8

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical 审计 envelope。"""

        return {
            "event_id": self.event_id,
            "research_task_id": self.research_task_id,
            "research_task_sequence": self.research_task_sequence,
            "event_type": self.event_type,
            "payload": _canonical_event_payload(self.event_type, self.payload),
            "recorded_at": self.recorded_at,
            "event_digest": self.event_digest,
            "previous_event_digest": self.previous_event_digest,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """返回 canonical 审计 envelope。"""

        return self.to_dict()


class ResearchTaskLedger:
    """实现 research_task/Module Invocation 领域写入和可重建读取。

    本类只创建并写入 canonical 表。旧的 ``v2_*`` 表和 schema-v10 事件不在
    活动入口中解析；需要重放旧账本时，调用方必须使用显式只读兼容读取器。
    """

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS research_task_schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS research_tasks (
        research_task_id TEXT PRIMARY KEY,
        contract_version TEXT NOT NULL,
        input_json TEXT NOT NULL,
        policy_json TEXT NOT NULL,
        dependencies_json TEXT NOT NULL,
        research_task_created_event_id TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS research_task_events (
        event_id TEXT PRIMARY KEY,
        research_task_id TEXT NOT NULL,
        research_task_sequence INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        event_digest TEXT NOT NULL,
        previous_event_digest TEXT,
        UNIQUE(research_task_id, research_task_sequence),
        FOREIGN KEY(research_task_id) REFERENCES research_tasks(research_task_id)
    );
    CREATE TABLE IF NOT EXISTS module_invocations (
        module_invocation_id TEXT PRIMARY KEY,
        research_task_id TEXT NOT NULL,
        module_kind TEXT NOT NULL,
        input_scientific_record_ids_json TEXT NOT NULL,
        module_input_digest TEXT NOT NULL,
        context_binding_json TEXT NOT NULL,
        started_event_id TEXT NOT NULL UNIQUE,
        completed_event_id TEXT UNIQUE,
        output_scientific_record_ids_json TEXT,
        module_output_digest TEXT,
        FOREIGN KEY(research_task_id) REFERENCES research_tasks(research_task_id),
        FOREIGN KEY(started_event_id) REFERENCES research_task_events(event_id),
        FOREIGN KEY(completed_event_id) REFERENCES research_task_events(event_id)
    );
    CREATE INDEX IF NOT EXISTS module_invocations_task_idx
        ON module_invocations(research_task_id);
    CREATE TABLE IF NOT EXISTS scientific_records (
        scientific_record_id TEXT PRIMARY KEY,
        research_task_id TEXT NOT NULL,
        scientific_record_type TEXT NOT NULL,
        scientific_record_schema_version TEXT NOT NULL,
        producer_module_invocation_id TEXT NOT NULL,
        source_scientific_record_ids_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        payload_digest TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        created_event_id TEXT NOT NULL UNIQUE,
        FOREIGN KEY(research_task_id) REFERENCES research_tasks(research_task_id),
        FOREIGN KEY(producer_module_invocation_id) REFERENCES module_invocations(module_invocation_id),
        FOREIGN KEY(created_event_id) REFERENCES research_task_events(event_id)
    );
    CREATE INDEX IF NOT EXISTS scientific_records_task_idx
        ON scientific_records(research_task_id, scientific_record_type, scientific_record_id);
    CREATE TABLE IF NOT EXISTS research_task_facts_projections (
        research_task_id TEXT PRIMARY KEY,
        projector_version TEXT NOT NULL,
        projected_through_research_task_sequence INTEGER NOT NULL,
        data_json TEXT NOT NULL,
        FOREIGN KEY(research_task_id) REFERENCES research_tasks(research_task_id)
    );
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        """打开 V2 账本；允许在同一 Working Memory v10 数据库旁扩展表。"""

        if not isinstance(db_path, (str, os.PathLike)):
            raise ContractValidationError("V2 db_path 必须是路径")
        self.db_path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize_database()

    def _connection(self) -> sqlite3.Connection:
        """创建一条短生命周期连接；调用方负责关闭连接。"""

        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        """核对数据库合同并创建 V2 表，不触碰旧表或旧数据。"""

        connection = self._connection()
        try:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "research_task_schema_meta" not in names and names:
                # V2 采用 expand-contract：允许在现有 schema v10 的同一物理
                # Working Memory 数据库旁边建立 v2_* 表，但绝不读取旧任务来
                # 补造 V2 Task。没有 schema_meta 的陌生数据库仍然拒绝，避免
                # 把任意外部 SQLite 文件误当作 Working Memory。
                if "schema_meta" not in names:
                    raise V2LedgerSchemaConflict(
                        "目标数据库没有 Working Memory schema_meta，拒绝建立 V2 账本"
                    )
                legacy_version = connection.execute(
                    """
                    SELECT value
                    FROM schema_meta
                    WHERE key IN ('schema_version', 'scientific_record_schema_version')
                    ORDER BY CASE key WHEN 'schema_version' THEN 0 ELSE 1 END
                    LIMIT 1
                    """
                ).fetchone()
                if legacy_version is None or str(legacy_version[0]) != "10":
                    raise V2LedgerSchemaConflict(
                        "目标 Working Memory schema 版本不受 V2 expand-contract 支持"
                    )
            if "research_task_schema_meta" in names:
                metadata = dict(
                    connection.execute("SELECT key, value FROM research_task_schema_meta").fetchall()
                )
                if metadata.get("contract") != _V2_CONTRACT or metadata.get("version") not in {
                    "1",
                    _V2_SCHEMA_VERSION,
                }:
                    raise V2LedgerSchemaConflict("V2 数据库合同版本不受支持")
                # 只补齐缺失的 V2 表定义；不会根据旧任务或外部展示文件补造
                # 任何科学事实。
                connection.executescript(self._SCHEMA_SQL)
                self._ensure_research_task_schema_v2(connection)
                if metadata.get("version") != _V2_SCHEMA_VERSION:
                    connection.execute(
                        "UPDATE research_task_schema_meta SET value=? WHERE key='version'",
                        (_V2_SCHEMA_VERSION,),
                    )
                return
            connection.executescript(self._SCHEMA_SQL)
            connection.executemany(
                "INSERT INTO research_task_schema_meta(key, value) VALUES (?, ?)",
                (("contract", _V2_CONTRACT), ("version", _V2_SCHEMA_VERSION)),
            )
            self._ensure_research_task_schema_v2(connection)
        finally:
            connection.close()

    @staticmethod
    def _ensure_research_task_schema_v2(connection: sqlite3.Connection) -> None:
        """为 canonical 数据库补齐模块完成与科学记录物化列。"""

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(module_invocations)").fetchall()
        }
        additions = {
            # SQLite 不允许 ALTER TABLE ADD COLUMN 携带 UNIQUE；索引在下方
            # 单独创建，空值仍允许已存在的 ModuleInvocationOpen invocation 行继续表示 ModuleInvocationOpen。
            "completed_event_id": "TEXT",
            "output_scientific_record_ids_json": "TEXT",
            "module_output_digest": "TEXT",
        }
        for name, ddl in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE module_invocations ADD COLUMN {name} {ddl}")
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS module_invocations_completion_idx "
            "ON module_invocations(completed_event_id) WHERE completed_event_id IS NOT NULL"
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """以写锁开启事务，保证序号、事件和领域行原子提交。"""

        with self._lock:
            connection = self._connection()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def _event_digest(
        *,
        event_id: str,
        research_task_id: str,
        research_task_sequence: int,
        event_type: str,
        payload: Mapping[str, Any],
        recorded_at: str,
        previous_event_digest: str | None,
    ) -> str:
        """按规范化 envelope 和前序摘要计算当前事件摘要。"""

        body = {
            "event_id": event_id,
            "research_task_id": research_task_id,
            "research_task_sequence": research_task_sequence,
            "event_type": event_type,
            "payload": payload,
            "recorded_at": recorded_at,
            "previous_event_digest": previous_event_digest,
        }
        return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()

    @staticmethod
    def _recorded_at() -> str:
        """产生只用于审计的 UTC 时间，不参与领域排序。"""

        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )

    def _read_events_in_connection(
        self,
        connection: sqlite3.Connection,
        research_task_id: str,
    ) -> tuple[WorkingMemoryEvent, ...]:
        """读取并完整核对一个 research_task 的事件序号和摘要链。"""

        rows = connection.execute(
            """
            SELECT event_id, research_task_id, research_task_sequence, event_type, payload_json,
                   recorded_at, event_digest, previous_event_digest
            FROM research_task_events
            WHERE research_task_id = ?
            ORDER BY research_task_sequence
            """,
            (research_task_id,),
        ).fetchall()
        events: list[WorkingMemoryEvent] = []
        previous_digest: str | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            if row["research_task_sequence"] != expected_sequence:
                raise WorkingMemoryIntegrityConflict(
                    f"research_task {research_task_id} 的事件序号不连续："
                    f"期望 {expected_sequence}，实际 {row['research_task_sequence']}"
                )
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError) as exc:
                raise WorkingMemoryIntegrityConflict(
                    f"research_task {research_task_id} 的事件 {row['event_id']} payload 损坏"
                ) from exc
            if not isinstance(payload, dict):
                raise WorkingMemoryIntegrityConflict(
                    f"research_task {research_task_id} 的事件 {row['event_id']} payload 不是 object"
                )
            if row["previous_event_digest"] != previous_digest:
                raise WorkingMemoryIntegrityConflict(
                    f"research_task {research_task_id} 的事件 {row['event_id']} 前序摘要不匹配"
                )
            calculated = self._event_digest(
                event_id=row["event_id"],
                research_task_id=row["research_task_id"],
                research_task_sequence=row["research_task_sequence"],
                event_type=row["event_type"],
                payload=payload,
                recorded_at=row["recorded_at"],
                previous_event_digest=previous_digest,
            )
            if row["event_digest"] != calculated:
                raise WorkingMemoryIntegrityConflict(
                    f"research_task {research_task_id} 的事件 {row['event_id']} 摘要校验失败"
                )
            event = WorkingMemoryEvent(
                event_id=row["event_id"],
                research_task_id=row["research_task_id"],
                research_task_sequence=row["research_task_sequence"],
                event_type=row["event_type"],
                payload=payload,
                recorded_at=row["recorded_at"],
                event_digest=row["event_digest"],
                previous_event_digest=previous_digest,
            )
            events.append(event)
            previous_digest = event.event_digest
        return tuple(events)

    def _append_event_in_connection(
        self,
        connection: sqlite3.Connection,
        *,
        research_task_id: str,
        event_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        expected_research_task_sequence: int | None,
    ) -> WorkingMemoryEvent:
        """在当前事务内分配下一个序号并追加一个受保护事件。"""

        current_events = self._read_events_in_connection(connection, research_task_id)
        current_sequence = len(current_events)
        if (
            expected_research_task_sequence is not None
            and expected_research_task_sequence != current_sequence
        ):
            raise CASConflict(
                f"research_task {research_task_id} 的 sequence 已变化",
                expected_version=expected_research_task_sequence,
                current_version=current_sequence,
            )
        sequence = current_sequence + 1
        previous_digest = current_events[-1].event_digest if current_events else None
        recorded_at = self._recorded_at()
        copied_payload = _copy_mapping(payload)
        event_digest = self._event_digest(
            event_id=event_id,
            research_task_id=research_task_id,
            research_task_sequence=sequence,
            event_type=event_type,
            payload=copied_payload,
            recorded_at=recorded_at,
            previous_event_digest=previous_digest,
        )
        try:
            connection.execute(
                """
                INSERT INTO research_task_events(
                    event_id, research_task_id, research_task_sequence, event_type, payload_json,
                    recorded_at, event_digest, previous_event_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    research_task_id,
                    sequence,
                    event_type,
                    canonical_json(copied_payload),
                    recorded_at,
                    event_digest,
                    previous_digest,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise WorkingMemoryIntegrityConflict(
                f"无法追加 research_task {research_task_id} 的唯一事件 {event_id}"
            ) from exc
        return WorkingMemoryEvent(
            event_id=event_id,
            research_task_id=research_task_id,
            research_task_sequence=sequence,
            event_type=event_type,
            payload=copied_payload,
            recorded_at=recorded_at,
            event_digest=event_digest,
            previous_event_digest=previous_digest,
        )

    @staticmethod
    def _validate_contract_version(value: Any) -> str | int:
        """验证持久合同版本是稳定的字符串或正整数。"""

        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ContractValidationError("contract_version 必须是字符串或整数")
        if isinstance(value, str) and not value.strip():
            raise ContractValidationError("contract_version 不能为空")
        if isinstance(value, int) and value <= 0:
            raise ContractValidationError("contract_version 必须大于 0")
        return value.strip() if isinstance(value, str) else value

    @staticmethod
    def _task_snapshot_from_row(row: sqlite3.Row) -> ResearchTaskSnapshot:
        """从 Task 行构造不可变事实快照。"""

        try:
            input_value = json.loads(row["input_json"])
            policy_value = json.loads(row["policy_json"])
            dependencies_value = json.loads(row["dependencies_json"])
        except (TypeError, ValueError) as exc:
            raise WorkingMemoryIntegrityConflict(
                f"research_task {row['research_task_id']} 的 TaskCreated 子结构损坏"
            ) from exc
        if not all(
            isinstance(value, dict)
            for value in (input_value, policy_value, dependencies_value)
        ):
            raise WorkingMemoryIntegrityConflict(
                f"research_task {row['research_task_id']} 的 TaskCreated 子结构不是 object"
            )
        contract_version: str | int
        try:
            contract_version_value = json.loads(row["contract_version"])
        except (TypeError, ValueError) as exc:
            raise WorkingMemoryIntegrityConflict(
                f"research_task {row['research_task_id']} 的 contract_version 损坏"
            ) from exc
        contract_version = ResearchTaskLedger._validate_contract_version(contract_version_value)
        return ResearchTaskSnapshot(
            research_task_id=row["research_task_id"],
            contract_version=contract_version,
            input=input_value,
            policy=policy_value,
            dependencies=dependencies_value,
        )

    @staticmethod
    def _task_snapshot_from_created_event(event: WorkingMemoryEvent) -> ResearchTaskSnapshot:
        """从权威 TaskCreated 事件构造 Task 快照。"""

        _require_exact_keys(
            event.payload,
            {"contract_version", "input", "policy", "dependencies"},
            "TaskCreated.payload",
        )
        return ResearchTaskSnapshot(
            research_task_id=event.research_task_id,
            contract_version=ResearchTaskLedger._validate_contract_version(
                event.payload["contract_version"]
            ),
            input=_require_object(event.payload["input"], "TaskCreated.input"),
            policy=_require_object(event.payload["policy"], "TaskCreated.policy"),
            dependencies=_require_object(
                event.payload["dependencies"], "TaskCreated.dependencies"
            ),
        )

    @staticmethod
    def _module_snapshot_from_request(
        request: ModuleInvocationRequest,
        *,
        start_event_ref: str,
        lifecycle: ModuleInvocationOpen | ModuleInvocationClosed,
    ) -> ModuleInvocationSnapshot:
        """从 ModuleInvocationStarted 事件解析出的请求构造模块快照。"""

        return ModuleInvocationSnapshot(
            research_task_id=request.research_task_id,
            module_invocation_id=request.module_invocation_id,
            module_kind=request.module_kind,
            input_scientific_record_ids=request.input_scientific_record_ids,
            module_input_digest=request.module_input_digest,
            lifecycle=lifecycle,
            start_event_ref=start_event_ref,
            context_binding=request.context_binding,
        )

    @staticmethod
    def _module_request_from_started_event(event: WorkingMemoryEvent) -> ModuleInvocationRequest:
        """严格解析 ModuleInvocationStarted 事件，不读取模块物化行。"""

        expected = {
            "research_task_id",
            "module_invocation_id",
            "module_kind",
            "input_scientific_record_ids",
            "module_input_digest",
            "context_binding",
        }
        _require_exact_keys(event.payload, expected, "ModuleInvocationStarted.payload")
        if event.payload["research_task_id"] != event.research_task_id:
            raise WorkingMemoryIntegrityConflict(
                f"ModuleInvocationStarted {event.event_id} 的 research_task_id 与事件分区不一致"
            )
        try:
            request = ModuleInvocationRequest.from_mapping(
                {"research_task_id": event.research_task_id, **event.payload}
            )
            _validate_module_kind(request.module_kind)
            return request
        except ContractValidationError as exc:
            raise WorkingMemoryIntegrityConflict(
                f"Module Run Started 事件 {event.event_id} payload 损坏"
            ) from exc

    @staticmethod
    def _scientific_record_from_created_event(
        research_task_id: str,
        event: WorkingMemoryEvent,
    ) -> ScientificRecord:
        """从 ScientificRecordCreated 事件继承 research_task 和 Working Memory 时间。"""

        _require_exact_keys(
            event.payload,
            {
                "scientific_record_id",
                "research_task_id",
                "scientific_record_type",
                "scientific_record_schema_version",
                "producer_module_invocation_id",
                "source_scientific_record_ids",
                "payload",
                "payload_digest",
            },
            "ScientificRecordCreated.payload",
        )
        try:
            payload = dict(event.payload)
            if payload.get("research_task_id") != research_task_id:
                raise ContractValidationError(
                    "ScientificRecordCreated 的 research_task_id 与事件分区不一致"
                )
            return ScientificRecord.from_mapping(payload).with_recorded_at(
                event.recorded_at
            )
        except (ContractValidationError, TypeError, ValueError) as exc:
            raise WorkingMemoryIntegrityConflict(
                f"Scientific Record 事件 {event.event_id} payload 损坏"
            ) from exc

    @staticmethod
    def _completion_payload_from_event(event: WorkingMemoryEvent) -> tuple[str, tuple[str, ...], str]:
        """严格解析 ModuleInvocationCompleted 的输出引用和摘要。"""

        _require_exact_keys(
            event.payload,
            {"module_invocation_id", "output_scientific_record_ids", "module_output_digest"},
            "ModuleInvocationCompleted.payload",
        )
        try:
            module_invocation_id = _require_text(
                event.payload["module_invocation_id"],
                "ModuleInvocationCompleted.module_invocation_id",
            )
            refs = _require_sequence(
                event.payload["output_scientific_record_ids"],
                "ModuleInvocationCompleted.output_scientific_record_ids",
            )
            module_output_digest = _validate_sha256(
                event.payload["module_output_digest"],
                "ModuleInvocationCompleted.module_output_digest",
            )
        except ContractValidationError as exc:
            raise WorkingMemoryIntegrityConflict(
                f"ModuleInvocationCompleted 事件 {event.event_id} payload 损坏"
            ) from exc
        return module_invocation_id, refs, module_output_digest

    @staticmethod
    def _task_closed_outcome_from_event(event: WorkingMemoryEvent) -> dict[str, Any]:
        """严格解析 TaskClosed 事件的终态 payload。"""

        _require_exact_keys(event.payload, {"outcome"}, "TaskClosed.payload")
        try:
            return _normalize_task_terminal_outcome(event.payload["outcome"])
        except ContractValidationError as exc:
            raise WorkingMemoryIntegrityConflict(
                f"TaskClosed 事件 {event.event_id} payload 损坏"
            ) from exc

    @staticmethod
    def _module_failed_outcome_from_event(
        event: WorkingMemoryEvent,
    ) -> tuple[str, str]:
        """严格解析 ModuleInvocationFailed 事件的模块与失败码。"""

        _require_exact_keys(
            event.payload,
            {"module_invocation_id", "failure_code"},
            "ModuleInvocationFailed.payload",
        )
        try:
            module_invocation_id = _require_text(
                event.payload["module_invocation_id"],
                "ModuleInvocationFailed.module_invocation_id",
            )
            failure_code = _require_text(
                event.payload["failure_code"],
                "ModuleInvocationFailed.failure_code",
            )
        except ContractValidationError as exc:
            raise WorkingMemoryIntegrityConflict(
                f"ModuleInvocationFailed 事件 {event.event_id} payload 损坏"
            ) from exc
        return module_invocation_id, failure_code

    def _load_task_facts_in_connection(
        self,
        connection: sqlite3.Connection,
        research_task_id: str,
    ) -> ResearchTaskFacts:
        """在同一读快照中仅从事件日志核对并归并 Task Facts。

        ``module_invocations``、``scientific_records`` 和
        ``research_task_facts_projections`` 都是可删除的查询投影；读取时即使物化行被
        篡改或删除，也不能覆盖 ModuleInvocationStarted 和 ScientificRecordCreated 中
        的权威值。
        """

        row = connection.execute(
            "SELECT * FROM research_tasks WHERE research_task_id = ?",
            (research_task_id,),
        ).fetchone()
        if row is None:
            raise ContractValidationError(f"V2 research_task {research_task_id} 不存在")
        events = self._read_events_in_connection(connection, research_task_id)
        if not events or events[0].event_type != "ResearchTaskCreated":
            raise WorkingMemoryIntegrityConflict(f"research_task {research_task_id} 缺少首个 TaskCreated 事件")
        if len([event for event in events if event.event_type == "ResearchTaskCreated"]) != 1:
            raise WorkingMemoryIntegrityConflict(f"research_task {research_task_id} 存在多个 TaskCreated 事件")
        created_event = events[0]
        task_snapshot = self._task_snapshot_from_created_event(created_event)
        if row["research_task_created_event_id"] != created_event.event_id:
            raise WorkingMemoryIntegrityConflict(
                f"research_task {research_task_id} 的任务行未引用 TaskCreated 事件"
            )
        try:
            expected_task = {
                "contract_version": json.loads(row["contract_version"]),
                "input": json.loads(row["input_json"]),
                "policy": json.loads(row["policy_json"]),
                "dependencies": json.loads(row["dependencies_json"]),
            }
        except (TypeError, ValueError) as exc:
            raise WorkingMemoryIntegrityConflict(
                f"research_task {research_task_id} 的任务物化行损坏"
            ) from exc
        if canonical_json(created_event.payload) != canonical_json(expected_task):
            raise WorkingMemoryIntegrityConflict(
                f"research_task {research_task_id} 的 TaskCreated 与任务行不一致"
            )

        modules: dict[str, dict[str, Any]] = {}
        module_order: list[str] = []
        records: dict[str, ScientificRecord] = {}
        record_order: list[ScientificRecord] = []
        graph_by_type: dict[str, list[str]] = {}
        successors: dict[str, list[str]] = {}
        result_successors: dict[str, str] = {}
        output_refs_by_module: dict[str, tuple[str, ...]] = {}
        task_closure: tuple[WorkingMemoryEvent, dict[str, Any]] | None = None
        for event in events[1:]:
            if task_closure is not None:
                raise WorkingMemoryIntegrityConflict(
                    f"TaskClosed 之后不能再追加 research_task {research_task_id} 的事件"
                )
            if event.event_type == "ModuleInvocationStarted":
                request = self._module_request_from_started_event(event)
                if request.module_invocation_id in modules:
                    raise WorkingMemoryIntegrityConflict(
                        "research_task "
                        f"{research_task_id} 存在重复 ModuleInvocationStarted "
                        f"{request.module_invocation_id}"
                    )
                modules[request.module_invocation_id] = {
                    "request": request,
                    "start_event": event,
                    "record_ids": [],
                    "completion": None,
                    "failure": None,
                }
                module_order.append(request.module_invocation_id)
                continue
            if event.event_type == "ScientificRecordCreated":
                record = self._scientific_record_from_created_event(research_task_id, event)
                if record.scientific_record_id in records:
                    raise WorkingMemoryIntegrityConflict(
                        f"research_task {research_task_id} 存在重复 Scientific Record {record.scientific_record_id}"
                    )
                module = modules.get(record.producer_module_invocation_id)
                if module is None:
                    raise WorkingMemoryIntegrityConflict(
                        f"Scientific Record {record.scientific_record_id} 引用了未知 Module Run {record.producer_module_invocation_id}"
                    )
                if module["completion"] is not None or module["failure"] is not None:
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {record.producer_module_invocation_id} 结束后仍有 Scientific Record"
                    )
                for source_ref in record.source_scientific_record_ids:
                    if source_ref not in records:
                        raise WorkingMemoryIntegrityConflict(
                            f"Scientific Record {record.scientific_record_id} 引用了缺失或未来来源 {source_ref}"
                        )
                    source_record = records[source_ref]
                    if (
                        record.scientific_record_type == "InvestigationResultRecord"
                        and source_record.scientific_record_type == "InvestigationResultRecord"
                    ):
                        prior_successor = result_successors.get(source_ref)
                        if prior_successor is not None and prior_successor != record.scientific_record_id:
                            raise WorkingMemoryIntegrityConflict(
                                f"Investigation Result {source_ref} 存在多个不允许的后继"
                            )
                        result_successors[source_ref] = record.scientific_record_id
                if record.scientific_record_type == "InvestigationResultRecord":
                    result_predecessors = [
                        source_ref
                        for source_ref in record.source_scientific_record_ids
                        if records[source_ref].scientific_record_type == "InvestigationResultRecord"
                    ]
                    if len(result_predecessors) > 1:
                        raise WorkingMemoryIntegrityConflict(
                            f"Investigation Result {record.scientific_record_id} 只能有一个结果前驱"
                        )
                if record.scientific_record_type == "ReviewRecord":
                    reviewed_refs = tuple(record.payload["reviewed_result_refs"])
                    result_heads = tuple(
                        item.scientific_record_id
                        for item in record_order
                        if item.scientific_record_type == "InvestigationResultRecord"
                        and item.scientific_record_id not in result_successors
                    )
                    if set(reviewed_refs) != set(result_heads):
                        raise WorkingMemoryIntegrityConflict(
                            f"Review {record.scientific_record_id} 未精确审查当前 Investigation Result heads"
                        )
                records[record.scientific_record_id] = record
                record_order.append(record)
                module["record_ids"].append(record.scientific_record_id)
                graph_by_type.setdefault(record.scientific_record_type, []).append(record.scientific_record_id)
                for source_ref in record.source_scientific_record_ids:
                    successors.setdefault(source_ref, []).append(record.scientific_record_id)
                continue
            if event.event_type == "ModuleInvocationCompleted":
                module_invocation_id, output_refs, module_output_digest = self._completion_payload_from_event(event)
                module = modules.get(module_invocation_id)
                if module is None:
                    raise WorkingMemoryIntegrityConflict(
                        "ModuleInvocationCompleted 引用了未知 Module Invocation "
                        f"{module_invocation_id}"
                    )
                if module["completion"] is not None or module["failure"] is not None:
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {module_invocation_id} 存在多个结束事实"
                    )
                currently_open = [
                    item_id
                    for item_id in module_order
                    if modules[item_id]["completion"] is None
                    and modules[item_id]["failure"] is None
                ]
                if currently_open != [module_invocation_id]:
                    raise WorkingMemoryIntegrityConflict(
                        f"ModuleInvocationCompleted {module_invocation_id} "
                        "不是当前唯一打开的 Module Invocation"
                    )
                if tuple(module["record_ids"]) != output_refs:
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {module_invocation_id} 的完成输出引用与 Scientific Record 事件不一致"
                    )
                if not output_refs:
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {module_invocation_id} 的完成输出不能为空"
                    )
                output_records = tuple(records[scientific_record_id] for scientific_record_id in output_refs)
                calculated = _prepared_output_digest(output_records)
                if module_output_digest != calculated:
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {module_invocation_id} 的 module_output_digest 校验失败"
                    )
                try:
                    _validate_module_output_group(
                        module["request"].module_kind,
                        output_records,
                        records,
                    )
                except ContractValidationError as exc:
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {module_invocation_id} 的输出成组合同损坏"
                    ) from exc
                module["completion"] = (event, output_refs, module_output_digest)
                output_refs_by_module[module_invocation_id] = output_refs
                continue
            if event.event_type == "ModuleInvocationFailed":
                module_invocation_id, failure_code = self._module_failed_outcome_from_event(event)
                module = modules.get(module_invocation_id)
                if module is None:
                    raise WorkingMemoryIntegrityConflict(
                        "ModuleInvocationFailed 引用了未知 Module Invocation "
                        f"{module_invocation_id}"
                    )
                if module["completion"] is not None or module["failure"] is not None:
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {module_invocation_id} 存在多个结束事实"
                    )
                currently_open = [
                    item_id
                    for item_id in module_order
                    if modules[item_id]["completion"] is None
                    and modules[item_id]["failure"] is None
                ]
                if currently_open != [module_invocation_id]:
                    raise WorkingMemoryIntegrityConflict(
                        f"ModuleInvocationFailed {module_invocation_id} "
                        "不是当前唯一打开的 Module Invocation"
                    )
                module["failure"] = (event, failure_code)
                continue
            if event.event_type == "ResearchTaskClosed":
                if any(
                    modules[module_invocation_id]["completion"] is None
                    and modules[module_invocation_id]["failure"] is None
                    for module_invocation_id in module_order
                ):
                    raise WorkingMemoryIntegrityConflict(
                        f"research_task {research_task_id} 存在 ModuleInvocationOpen Module Run，不能关闭 Task"
                    )
                outcome = self._task_closed_outcome_from_event(event)
                if outcome["status"] == "failed":
                    raise WorkingMemoryIntegrityConflict(
                        "TaskClosed 的 outcome.status 必须使用 TaskFailed"
                    )
                if any(
                    modules[module_invocation_id]["failure"] is not None
                    for module_invocation_id in module_order
                ):
                    raise WorkingMemoryIntegrityConflict(
                        "存在 ModuleInvocationFailed 时必须写入 ResearchTaskFailed"
                    )
                task_closure = (event, outcome)
                continue
            if event.event_type == "ResearchTaskFailed":
                _require_exact_keys(event.payload, {"outcome"}, "TaskFailed.payload")
                if any(
                    modules[module_invocation_id]["completion"] is None
                    and modules[module_invocation_id]["failure"] is None
                    for module_invocation_id in module_order
                ):
                    raise WorkingMemoryIntegrityConflict(
                        f"research_task {research_task_id} 存在 ModuleInvocationOpen Module Run，不能写入 TaskFailed"
                    )
                outcome = self._task_closed_outcome_from_event(event)
                if outcome["status"] != "failed":
                    raise WorkingMemoryIntegrityConflict(
                        "TaskFailed 的 outcome.status 必须为 failed"
                    )
                failed_modules = [
                    (module_invocation_id, module["failure"])
                    for module_invocation_id, module in modules.items()
                    if module["failure"] is not None
                ]
                if len(failed_modules) != 1:
                    raise WorkingMemoryIntegrityConflict(
                        "ResearchTaskFailed 必须对应唯一 ModuleInvocationFailed"
                    )
                failed_module_id, failure = failed_modules[0]
                _, failure_code = failure
                if (
                    outcome.get("module_invocation_id") != failed_module_id
                    or outcome.get("failure_code") != failure_code
                ):
                    raise WorkingMemoryIntegrityConflict(
                        "ResearchTaskFailed 的失败模块或失败码与 "
                        "ModuleInvocationFailed 不一致"
                    )
                task_closure = (event, outcome)
                continue
            raise WorkingMemoryIntegrityConflict(
                f"research_task {research_task_id} 含 V2 不认识的事件 {event.event_type}"
            )

        open_modules = [
            module_invocation_id
            for module_invocation_id in module_order
            if modules[module_invocation_id]["completion"] is None
            and modules[module_invocation_id]["failure"] is None
        ]
        failed_modules = [
            module_invocation_id
            for module_invocation_id in module_order
            if modules[module_invocation_id]["failure"] is not None
        ]
        if failed_modules and task_closure is None:
            raise WorkingMemoryIntegrityConflict(
                "ModuleInvocationFailed 后必须追加 ResearchTaskFailed"
            )
        if len(open_modules) > 1:
            raise WorkingMemoryIntegrityConflict(
                f"research_task {research_task_id} 存在多个 ModuleInvocationOpen Module Run；账本要求最多一个"
            )
        module_snapshots: dict[str, ModuleInvocationSnapshot] = {}
        for module_invocation_id in module_order:
            module = modules[module_invocation_id]
            completion = module["completion"]
            failure = module["failure"]
            if completion is None and failure is None:
                lifecycle: ModuleInvocationOpen | ModuleInvocationClosed = ModuleInvocationOpen(module["start_event"].event_id)
            elif completion is not None:
                completion_event, output_refs, module_output_digest = completion
                lifecycle = ModuleInvocationClosed(
                    module["start_event"].event_id,
                    completion_event.event_id,
                    {"output_scientific_record_ids": list(output_refs), "module_output_digest": module_output_digest},
                )
            else:
                failure_event, failure_code = failure
                lifecycle = ModuleInvocationClosed(
                    module["start_event"].event_id,
                    failure_event.event_id,
                    {"status": "failed", "failure_code": failure_code},
                )
            module_snapshots[module_invocation_id] = self._module_snapshot_from_request(
                module["request"],
                start_event_ref=module["start_event"].event_id,
                lifecycle=lifecycle,
            )
        closed_module_invocations = tuple(
            module_snapshots[module_invocation_id]
            for module_invocation_id in module_order
            if modules[module_invocation_id]["completion"] is not None
            or modules[module_invocation_id]["failure"] is not None
        )
        active_module_invocation: NoOpenModuleInvocation | OpenModuleInvocation
        if open_modules:
            active_module_invocation = OpenModuleInvocation(open_modules[0])
            open_module_invocation = module_snapshots[open_modules[0]]
        else:
            active_module_invocation = NoOpenModuleInvocation()
            open_module_invocation = None
        lifecycle: ResearchTaskOpen | ResearchTaskClosed
        if task_closure is None:
            lifecycle = ResearchTaskOpen(events[0].event_id)
        else:
            closure_event, outcome = task_closure
            lifecycle = ResearchTaskClosed(closure_event.event_id, outcome)
        return ResearchTaskFacts(
            research_task=task_snapshot,
            research_task_sequence=len(events),
            lifecycle=lifecycle,
            closed_module_invocations=closed_module_invocations,
            active_module_invocation=active_module_invocation,
            open_module_invocation=open_module_invocation,
            scientific_graph=ScientificGraphFacts(
                record_refs_by_type={
                    scientific_record_type: tuple(record_refs)
                    for scientific_record_type, record_refs in graph_by_type.items()
                },
                successor_refs_by_record={
                    scientific_record_id: tuple(record_refs)
                    for scientific_record_id, record_refs in successors.items()
                },
                output_scientific_record_ids_by_module_invocation=output_refs_by_module,
            ),
        )

    def _read_scientific_records_from_events(
        self,
        research_task_id: str,
        events: Sequence[WorkingMemoryEvent],
    ) -> dict[str, ScientificRecord]:
        """从已核对的事件序列提取 Scientific Record，不读取物化表。"""

        records: dict[str, ScientificRecord] = {}
        for event in events:
            if event.event_type != "ScientificRecordCreated":
                continue
            record = self._scientific_record_from_created_event(research_task_id, event)
            if record.scientific_record_id in records:
                raise WorkingMemoryIntegrityConflict(
                    f"research_task {research_task_id} 存在重复 Scientific Record {record.scientific_record_id}"
                )
            records[record.scientific_record_id] = record
        return records

    @staticmethod
    def _upsert_module_invocation_projection(
        connection: sqlite3.Connection,
        *,
        research_task_id: str,
        request: ModuleInvocationRequest,
        started_event_id: str,
        completed_event_id: str | None = None,
        output_scientific_record_ids: Sequence[str] | None = None,
        module_output_digest: str | None = None,
    ) -> None:
        """写入可删除的 Module Invocation 查询投影；事件仍是唯一事实来源。"""

        existing = connection.execute(
            "SELECT research_task_id FROM module_invocations WHERE module_invocation_id = ?",
            (request.module_invocation_id,),
        ).fetchone()
        if existing is not None and existing["research_task_id"] != research_task_id:
            raise WorkingMemoryIntegrityConflict(
                f"Module Run {request.module_invocation_id} 的物化行跨 research_task"
            )
        values = (
            research_task_id,
            request.module_kind,
            canonical_json(list(request.input_scientific_record_ids)),
            request.module_input_digest,
            canonical_json(request.context_binding.to_dict()),
            started_event_id,
            completed_event_id,
            None
            if output_scientific_record_ids is None
            else canonical_json(list(output_scientific_record_ids)),
            module_output_digest,
        )
        if existing is None:
            connection.execute(
                """
                INSERT INTO module_invocations(
                    module_invocation_id, research_task_id, module_kind,
                    input_scientific_record_ids_json, module_input_digest,
                    context_binding_json, started_event_id,
                    completed_event_id, output_scientific_record_ids_json,
                    module_output_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (request.module_invocation_id, *values),
            )
        else:
            connection.execute(
                """
                UPDATE module_invocations
                SET research_task_id=?, module_kind=?,
                    input_scientific_record_ids_json=?, module_input_digest=?,
                    context_binding_json=?,
                    started_event_id=?, completed_event_id=?,
                    output_scientific_record_ids_json=?, module_output_digest=?
                WHERE module_invocation_id=?
                """,
                (*values, request.module_invocation_id),
            )

    @staticmethod
    def _upsert_scientific_record_projection(
        connection: sqlite3.Connection,
        *,
        record: ScientificRecord,
        created_event_id: str,
    ) -> None:
        """写入单条科学记录物化行；读取路径不会信任此行。"""

        existing = connection.execute(
            "SELECT research_task_id FROM scientific_records WHERE scientific_record_id = ?",
            (record.scientific_record_id,),
        ).fetchone()
        if existing is not None and existing["research_task_id"] != record.research_task_id:
            raise WorkingMemoryIntegrityConflict(
                f"Scientific Record {record.scientific_record_id} 的物化行跨 research_task"
            )
        values = (
            record.research_task_id,
            record.scientific_record_type,
            canonical_json(record.scientific_record_schema_version),
            record.producer_module_invocation_id,
            canonical_json(list(record.source_scientific_record_ids)),
            canonical_json(record.payload),
            record.payload_digest,
            record.recorded_at,
            created_event_id,
        )
        if existing is None:
            connection.execute(
                """
                INSERT INTO scientific_records(
                    scientific_record_id, research_task_id, scientific_record_type, scientific_record_schema_version,
                    producer_module_invocation_id, source_scientific_record_ids_json, payload_json,
                    payload_digest, recorded_at, created_event_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record.scientific_record_id, *values),
            )
        else:
            connection.execute(
                """
                UPDATE scientific_records
                SET research_task_id=?, scientific_record_type=?, scientific_record_schema_version=?, producer_module_invocation_id=?,
                    source_scientific_record_ids_json=?, payload_json=?, payload_digest=?,
                    recorded_at=?, created_event_id=?
                WHERE scientific_record_id=?
                """,
                (*values, record.scientific_record_id),
            )

    @staticmethod
    def _upsert_task_facts_projection(
        connection: sqlite3.Connection,
        facts: ResearchTaskFacts,
    ) -> None:
        """保存可删除的 Task Facts 快照，供查询加速而不改变权威来源。"""

        # 物化投影与旧 SQLite 列同属查询兼容层；公开读取走 ResearchTaskFacts.to_dict()。
        data_json = canonical_json(facts.to_dict())
        existing = connection.execute(
            "SELECT 1 FROM research_task_facts_projections WHERE research_task_id = ?",
            (facts.research_task.research_task_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO research_task_facts_projections(
                    research_task_id, projector_version, projected_through_research_task_sequence, data_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    facts.research_task.research_task_id,
                    _V2_SCHEMA_VERSION,
                    facts.research_task_sequence,
                    data_json,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE research_task_facts_projections
                SET projector_version=?, projected_through_research_task_sequence=?, data_json=?
                WHERE research_task_id=?
                """,
                (
                    _V2_SCHEMA_VERSION,
                    facts.research_task_sequence,
                    data_json,
                    facts.research_task.research_task_id,
                ),
            )

    def create_research_task(
        self,
        *,
        research_task_id: str,
        contract_version: str | int,
        input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> ResearchTaskFacts:
        """原子创建 V2 Task，并对同身份重复请求执行幂等重放。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        contract_version = self._validate_contract_version(contract_version)
        input_value = _require_object(input, "input")
        policy_value = _require_object(policy, "policy")
        dependencies_value = _require_object(dependencies, "dependencies")
        payload = {
            "contract_version": contract_version,
            "input": input_value,
            "policy": policy_value,
            "dependencies": dependencies_value,
        }
        event_id = f"research-task-created:{research_task_id}"
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM research_tasks WHERE research_task_id = ?",
                (research_task_id,),
            ).fetchone()
            if existing is not None:
                existing_payload = {
                    "contract_version": json.loads(existing["contract_version"]),
                    "input": json.loads(existing["input_json"]),
                    "policy": json.loads(existing["policy_json"]),
                    "dependencies": json.loads(existing["dependencies_json"]),
                }
                if canonical_json(existing_payload) != canonical_json(payload):
                    raise IdempotencyConflict(
                        f"V2 research_task {research_task_id} 的 TaskCreated 重放内容发生漂移"
                    )
                facts = self._load_task_facts_in_connection(connection, research_task_id)
                self._upsert_task_facts_projection(connection, facts)
                return facts
            connection.execute(
                """
                INSERT INTO research_tasks(
                    research_task_id, contract_version, input_json, policy_json,
                    dependencies_json, research_task_created_event_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    research_task_id,
                    canonical_json(contract_version),
                    canonical_json(input_value),
                    canonical_json(policy_value),
                    canonical_json(dependencies_value),
                    event_id,
                ),
            )
            self._append_event_in_connection(
                connection,
                research_task_id=research_task_id,
                event_id=event_id,
                event_type="ResearchTaskCreated",
                payload=payload,
                expected_research_task_sequence=None,
            )
            facts = self._load_task_facts_in_connection(connection, research_task_id)
            self._upsert_task_facts_projection(connection, facts)
            return facts

    def read_research_task_facts(self, research_task_id: str) -> ResearchTaskFacts:
        """从事件日志重建并返回一个 Task 的公共事实视图。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        with self._lock:
            connection = self._connection()
            try:
                return self._load_task_facts_in_connection(connection, research_task_id)
            finally:
                connection.close()

    def read_research_task_event_log(
        self,
        research_task_id: str,
    ) -> tuple[WorkingMemoryEvent, ...]:
        """只读返回事件审计记录；不提供通用事件写入入口。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        with self._lock:
            connection = self._connection()
            try:
                if connection.execute(
                    "SELECT 1 FROM research_tasks WHERE research_task_id = ?", (research_task_id,)
                ).fetchone() is None:
                    raise ContractValidationError(
                        f"V2 research_task {research_task_id} 不存在"
                    )
                return self._read_events_in_connection(connection, research_task_id)
            finally:
                connection.close()

    def get_research_task_facts(self, research_task_id: str) -> ResearchTaskFacts:
        """提供 Task Facts 的语义化读取别名。"""

        return self.read_research_task_facts(research_task_id)

    def read_research_task_scientific_records(
        self,
        research_task_id: str,
        scientific_record_ids: Sequence[str] | None = None,
    ) -> tuple[ScientificRecord, ...]:
        """按记录引用读取科学内容；只从已核对的事件日志恢复。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        refs = (
            None
            if scientific_record_ids is None
            else _require_sequence(
                scientific_record_ids,
                "scientific_record_ids",
            )
        )
        with self._lock:
            connection = self._connection()
            try:
                if connection.execute(
                    "SELECT 1 FROM research_tasks WHERE research_task_id = ?", (research_task_id,)
                ).fetchone() is None:
                    raise ContractValidationError(
                        f"V2 research_task {research_task_id} 不存在"
                    )
                self._load_task_facts_in_connection(connection, research_task_id)
                events = self._read_events_in_connection(connection, research_task_id)
                records = self._read_scientific_records_from_events(
                    research_task_id,
                    events,
                )
                selected_refs = tuple(records) if refs is None else refs
                result: list[ScientificRecord] = []
                for scientific_record_id in selected_refs:
                    record = records.get(scientific_record_id)
                    if record is None:
                        raise ContractValidationError(
                            f"research_task {research_task_id} 不存在 Scientific Record {scientific_record_id}"
                        )
                    result.append(record)
                return tuple(result)
            finally:
                connection.close()

    def complete_module_invocation(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        output_bundle: PreparedModuleInvocationOutput | Mapping[str, Any],
    ) -> ModuleInvocationSnapshot:
        """在单一事务中接收全部记录并闭合一个 Module Run。

        事务中先验证 bundle、来源、类型成组和输入摘要，再依次追加
        ``ScientificRecordCreated``，最后追加 ``ModuleInvocationCompleted``。任一记录、
        事件或物化投影写入失败都会回滚整个 bundle。
        """

        research_task_id = _require_text(research_task_id, "research_task_id")
        if isinstance(expected_research_task_sequence, bool) or not isinstance(
            expected_research_task_sequence, int
        ):
            raise ContractValidationError("expected_research_task_sequence 必须是整数")
        if expected_research_task_sequence < 1:
            raise ContractValidationError(
                "expected_research_task_sequence 必须大于等于 1"
            )
        normalized = (
            output_bundle
            if isinstance(output_bundle, PreparedModuleInvocationOutput)
            else PreparedModuleInvocationOutput.from_mapping(output_bundle)
        )
        if any(record.recorded_at is not None for record in normalized.output_scientific_records):
            raise ContractValidationError(
                "模块提供的 ScientificRecord.recorded_at 必须为 None，由 Working Memory 写入"
            )

        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM research_tasks WHERE research_task_id = ?", (research_task_id,)
            ).fetchone() is None:
                raise ContractValidationError(
                    f"V2 research_task {research_task_id} 不存在"
                )
            facts = self._load_task_facts_in_connection(connection, research_task_id)
            events = self._read_events_in_connection(connection, research_task_id)
            existing_records = self._read_scientific_records_from_events(
                research_task_id,
                events,
            )
            module = next(
                (
                    item
                    for item in facts.module_invocations
                    if item.module_invocation_id == normalized.module_invocation_id
                ),
                None,
            )
            if module is None:
                global_started = connection.execute(
                    "SELECT research_task_id FROM research_task_events WHERE event_id = ?",
                    (f"module-invocation-started:{normalized.module_invocation_id}",),
                ).fetchone()
                if global_started is not None and global_started["research_task_id"] != research_task_id:
                    raise IdempotencyConflict(
                        f"Module Run {normalized.module_invocation_id} 已绑定到另一个 research_task"
                    )
                raise ContractValidationError(
                    f"research_task {research_task_id} 不存在 Module Run {normalized.module_invocation_id}"
                )

            if isinstance(module.lifecycle, ModuleInvocationClosed):
                if module.lifecycle.outcome.get("status") == "failed":
                    raise IdempotencyConflict(
                        f"Module Run {normalized.module_invocation_id} 已经以 verified fatal 失败"
                    )
                existing_refs = tuple(module.lifecycle.outcome["output_scientific_record_ids"])
                existing_output = tuple(existing_records[ref] for ref in existing_refs)
                same_identity = len(existing_output) == len(normalized.output_scientific_records) and all(
                    left.canonical_identity_dict() == right.canonical_identity_dict()
                    for left, right in zip(existing_output, normalized.output_scientific_records)
                )
                if (
                    normalized.module_input_digest != module.module_input_digest
                    or normalized.module_output_digest != module.lifecycle.outcome["module_output_digest"]
                    or not same_identity
                ):
                    raise IdempotencyConflict(
                        f"Module Run {normalized.module_invocation_id} 的完成输出发生漂移"
                    )
                # 幂等重放修复可能被删除或落后的物化行，但永远不以物化行作事实。
                started_event = next(
                    event
                    for event in events
                    if event.event_id == module.start_event_ref
                )
                started_request = self._module_request_from_started_event(started_event)
                self._upsert_module_invocation_projection(
                    connection,
                    research_task_id=research_task_id,
                    request=started_request,
                    started_event_id=module.start_event_ref,
                    completed_event_id=module.lifecycle.closure_event_ref,
                    output_scientific_record_ids=existing_refs,
                    module_output_digest=module.lifecycle.outcome["module_output_digest"],
                )
                for scientific_record_id in existing_refs:
                    record_event = next(
                        event
                        for event in events
                        if event.event_type == "ScientificRecordCreated"
                        and event.payload.get("scientific_record_id") == scientific_record_id
                    )
                    self._upsert_scientific_record_projection(
                        connection,
                        record=existing_records[scientific_record_id],
                        created_event_id=record_event.event_id,
                    )
                self._upsert_task_facts_projection(connection, facts)
                return module

            if facts.open_module_invocation is None or facts.open_module_invocation.module_invocation_id != module.module_invocation_id:
                raise WorkingMemoryIntegrityConflict(
                    f"Module Run {normalized.module_invocation_id} 的 ModuleInvocationOpen 状态与 Task Facts 不一致"
                )
            if expected_research_task_sequence != facts.research_task_sequence:
                raise CASConflict(
                    f"research_task {research_task_id} 的 sequence 已变化",
                    expected_version=expected_research_task_sequence,
                    current_version=facts.research_task_sequence,
                )
            if normalized.module_input_digest != module.module_input_digest:
                raise IdempotencyConflict(
                    f"Module Run {normalized.module_invocation_id} 的 module_input_digest 发生漂移"
                )

            candidate_records = dict(existing_records)
            for record in normalized.output_scientific_records:
                if record.research_task_id != research_task_id:
                    raise ContractValidationError(
                        f"Scientific Record {record.scientific_record_id} 属于另一个 research_task"
                    )
                if record.producer_module_invocation_id != module.module_invocation_id:
                    raise ContractValidationError(
                        f"Scientific Record {record.scientific_record_id} 的 producer Module Run 不匹配"
                    )
                if record.scientific_record_id in candidate_records:
                    raise IdempotencyConflict(
                        f"Scientific Record {record.scientific_record_id} 已经存在"
                    )
                foreign_event = connection.execute(
                    "SELECT research_task_id FROM research_task_events WHERE event_id = ?",
                    (f"scientific-record-created:{record.scientific_record_id}",),
                ).fetchone()
                if foreign_event is not None:
                    if foreign_event["research_task_id"] != research_task_id:
                        raise WorkingMemoryIntegrityConflict(
                            f"Scientific Record {record.scientific_record_id} 已属于另一个 research_task"
                        )
                    raise WorkingMemoryIntegrityConflict(
                        f"Scientific Record {record.scientific_record_id} 已有未完成事件"
                    )
                for source_ref in record.source_scientific_record_ids:
                    if source_ref not in candidate_records:
                        raise WorkingMemoryIntegrityConflict(
                            f"Scientific Record {record.scientific_record_id} 引用了缺失或未来来源 {source_ref}"
                        )
                candidate_records[record.scientific_record_id] = record

            _validate_module_output_group(
                module.module_kind,
                normalized.output_scientific_records,
                candidate_records,
            )

            self._upsert_module_invocation_projection(
                connection,
                research_task_id=research_task_id,
                request=ModuleInvocationRequest(
                    research_task_id=research_task_id,
                    module_invocation_id=module.module_invocation_id,
                    module_kind=module.module_kind,
                    input_scientific_record_ids=module.input_scientific_record_ids,
                    module_input_digest=module.module_input_digest,
                    context_binding=module.context_binding,
                ),
                started_event_id=module.start_event_ref,
            )
            for record in normalized.output_scientific_records:
                created_event = self._append_event_in_connection(
                    connection,
                    research_task_id=research_task_id,
                    event_id=f"scientific-record-created:{record.scientific_record_id}",
                    event_type="ScientificRecordCreated",
                    payload=record.canonical_identity_dict(),
                    expected_research_task_sequence=None,
                )
                persisted_record = record.with_recorded_at(created_event.recorded_at)
                self._upsert_scientific_record_projection(
                    connection,
                    record=persisted_record,
                    created_event_id=created_event.event_id,
                )

            completion_payload = {
                "module_invocation_id": module.module_invocation_id,
                "output_scientific_record_ids": [
                    record.scientific_record_id for record in normalized.output_scientific_records
                ],
                "module_output_digest": normalized.module_output_digest,
            }
            completion_event = self._append_event_in_connection(
                connection,
                research_task_id=research_task_id,
                event_id=f"module-invocation-completed:{module.module_invocation_id}",
                event_type="ModuleInvocationCompleted",
                payload=completion_payload,
                expected_research_task_sequence=None,
            )
            self._upsert_module_invocation_projection(
                connection,
                research_task_id=research_task_id,
                request=ModuleInvocationRequest(
                    research_task_id=research_task_id,
                    module_invocation_id=module.module_invocation_id,
                    module_kind=module.module_kind,
                    input_scientific_record_ids=module.input_scientific_record_ids,
                    module_input_digest=module.module_input_digest,
                    context_binding=module.context_binding,
                ),
                started_event_id=module.start_event_ref,
                completed_event_id=completion_event.event_id,
                output_scientific_record_ids=completion_payload["output_scientific_record_ids"],
                module_output_digest=normalized.module_output_digest,
            )
            completed_facts = self._load_task_facts_in_connection(
                connection,
                research_task_id,
            )
            self._upsert_task_facts_projection(connection, completed_facts)
            completed_module = next(
                item
                for item in completed_facts.closed_module_invocations
                if item.module_invocation_id == normalized.module_invocation_id
            )
            return completed_module

    def close_research_task(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        outcome: Mapping[str, Any] | str,
    ) -> ResearchTaskFacts:
        """原子写入终态 TaskClosed；相同终态请求可精确重放。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        if isinstance(expected_research_task_sequence, bool) or not isinstance(
            expected_research_task_sequence, int
        ):
            raise ContractValidationError("expected_research_task_sequence 必须是整数")
        if expected_research_task_sequence < 1:
            raise ContractValidationError(
                "expected_research_task_sequence 必须大于等于 1"
            )
        normalized = _normalize_task_terminal_outcome(outcome)
        if normalized["status"] == "failed":
            raise ContractValidationError("failed 必须通过 fail_research_task 写入")
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM research_tasks WHERE research_task_id = ?", (research_task_id,)
            ).fetchone() is None:
                raise ContractValidationError(
                    f"V2 research_task {research_task_id} 不存在"
                )
            facts = self._load_task_facts_in_connection(connection, research_task_id)
            if isinstance(facts.lifecycle, ResearchTaskClosed):
                if facts.lifecycle.outcome != normalized:
                    raise IdempotencyConflict(
                        f"research_task {research_task_id} 的终态发生漂移"
                    )
                self._upsert_task_facts_projection(connection, facts)
                return facts
            if facts.open_module_invocation is not None:
                raise OperationConflict(
                    f"research_task {research_task_id} 存在 ModuleInvocationOpen Module Invocation，不能关闭"
                )
            if facts.research_task_sequence != expected_research_task_sequence:
                raise CASConflict(
                    f"research_task {research_task_id} 的 sequence 已变化",
                    expected_version=expected_research_task_sequence,
                    current_version=facts.research_task_sequence,
                )
            self._append_event_in_connection(
                connection,
                research_task_id=research_task_id,
                event_id=f"research-task-closed:{research_task_id}",
                event_type="ResearchTaskClosed",
                payload={"outcome": normalized},
                expected_research_task_sequence=expected_research_task_sequence,
            )
            closed = self._load_task_facts_in_connection(connection, research_task_id)
            self._upsert_task_facts_projection(connection, closed)
            return closed

    def fail_research_task(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        *,
        module_invocation_id: str,
        failure_code: str,
    ) -> ResearchTaskFacts:
        """原子写入 ModuleInvocationFailed 与 ResearchTaskFailed。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        module_invocation_id = _require_text(
            module_invocation_id,
            "module_invocation_id",
        )
        failure_code = _require_text(failure_code, "failure_code")
        if isinstance(expected_research_task_sequence, bool) or not isinstance(
            expected_research_task_sequence, int
        ):
            raise ContractValidationError("expected_research_task_sequence 必须是整数")
        if expected_research_task_sequence < 1:
            raise ContractValidationError(
                "expected_research_task_sequence 必须大于等于 1"
            )
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM research_tasks WHERE research_task_id = ?", (research_task_id,)
            ).fetchone() is None:
                raise ContractValidationError(
                    f"V2 research_task {research_task_id} 不存在"
                )
            facts = self._load_task_facts_in_connection(connection, research_task_id)
            if isinstance(facts.lifecycle, ResearchTaskClosed):
                existing = facts.lifecycle.outcome
                if (
                    existing.get("status") != "failed"
                    or existing.get("module_invocation_id") != module_invocation_id
                    or existing.get("failure_code") != failure_code
                ):
                    raise IdempotencyConflict(
                        f"research_task {research_task_id} 的失败终态发生漂移"
                    )
                self._upsert_task_facts_projection(connection, facts)
                return facts
            if facts.open_module_invocation is None:
                raise OperationConflict(
                    f"research_task {research_task_id} 没有可失败的 ModuleInvocationOpen Module Invocation"
                )
            if facts.open_module_invocation.module_invocation_id != module_invocation_id:
                raise WorkingMemoryIntegrityConflict(
                    f"失败 Module Invocation {module_invocation_id} 不是当前 ModuleInvocationOpen Module Invocation"
                )
            if facts.research_task_sequence != expected_research_task_sequence:
                raise CASConflict(
                    f"research_task {research_task_id} 的 sequence 已变化",
                    expected_version=expected_research_task_sequence,
                    current_version=facts.research_task_sequence,
                )
            failure_payload = {
                "module_invocation_id": module_invocation_id,
                "failure_code": failure_code,
            }
            self._append_event_in_connection(
                connection,
                research_task_id=research_task_id,
                event_id=f"module-invocation-failed:{module_invocation_id}",
                event_type="ModuleInvocationFailed",
                payload=failure_payload,
                expected_research_task_sequence=expected_research_task_sequence,
            )
            task_outcome = {
                "status": "failed",
                "module_invocation_id": module_invocation_id,
                "failure_code": failure_code,
            }
            self._append_event_in_connection(
                connection,
                research_task_id=research_task_id,
                event_id=f"research-task-failed:{research_task_id}",
                event_type="ResearchTaskFailed",
                payload={"outcome": task_outcome},
                expected_research_task_sequence=None,
            )
            failed = self._load_task_facts_in_connection(connection, research_task_id)
            self._upsert_task_facts_projection(connection, failed)
            return failed

    def start_module_invocation(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        request: ModuleInvocationRequest | Mapping[str, Any],
    ) -> ModuleInvocationSnapshot:
        """原子追加唯一 ModuleInvocationStarted，并支持精确重放与漂移拒绝。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        if isinstance(expected_research_task_sequence, bool) or not isinstance(
            expected_research_task_sequence,
            int,
        ):
            raise ContractValidationError("expected_research_task_sequence 必须是整数")
        if expected_research_task_sequence < 1:
            raise ContractValidationError(
                "expected_research_task_sequence 必须大于等于 1"
            )
        normalized = (
            request
            if isinstance(request, ModuleInvocationRequest)
            else ModuleInvocationRequest.from_mapping(request)
        )
        _validate_module_kind(normalized.module_kind)
        if normalized.research_task_id != research_task_id:
            raise ContractValidationError(
                "Module Invocation 的 research_task_id 与调用目标不一致"
            )
        # 当前账本事件和公开 mapping 都只保存 canonical Module Invocation 字段。
        payload = normalized.to_dict()
        event_id = f"module-invocation-started:{normalized.module_invocation_id}"
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM research_tasks WHERE research_task_id = ?", (research_task_id,)
            ).fetchone() is None:
                raise ContractValidationError(
                    f"V2 research_task {research_task_id} 不存在"
                )
            facts = self._load_task_facts_in_connection(connection, research_task_id)
            events = self._read_events_in_connection(connection, research_task_id)
            existing_module = next(
                (
                    module
                    for module in facts.module_invocations
                    if module.module_invocation_id == normalized.module_invocation_id
                ),
                None,
            )
            if existing_module is not None:
                started_event = next(
                    (
                        event
                        for event in events
                        if event.event_id == existing_module.start_event_ref
                    ),
                    None,
                )
                if started_event is None:  # pragma: no cover - facts 已核对该引用
                    raise WorkingMemoryIntegrityConflict(
                        f"Module Run {normalized.module_invocation_id} 缺少开始事件"
                    )
                existing_request = self._module_request_from_started_event(started_event)
                if canonical_json(existing_request.to_dict()) != canonical_json(payload):
                    raise IdempotencyConflict(
                        f"Module Run {normalized.module_invocation_id} 的身份或摘要发生漂移"
                    )
                self._upsert_module_invocation_projection(
                    connection,
                    research_task_id=research_task_id,
                    request=existing_request,
                    started_event_id=started_event.event_id,
                    completed_event_id=(
                        existing_module.lifecycle.closure_event_ref
                        if isinstance(existing_module.lifecycle, ModuleInvocationClosed)
                        else None
                    ),
                    output_scientific_record_ids=(
                        existing_module.lifecycle.outcome.get("output_scientific_record_ids")
                        if isinstance(existing_module.lifecycle, ModuleInvocationClosed)
                        else None
                    ),
                    module_output_digest=(
                        existing_module.lifecycle.outcome.get("module_output_digest")
                        if isinstance(existing_module.lifecycle, ModuleInvocationClosed)
                        else None
                    ),
                )
                return existing_module
            projection_row = connection.execute(
                "SELECT research_task_id FROM module_invocations WHERE module_invocation_id = ?",
                (normalized.module_invocation_id,),
            ).fetchone()
            if projection_row is not None:
                if projection_row["research_task_id"] != research_task_id:
                    raise IdempotencyConflict(
                        f"Module Run {normalized.module_invocation_id} 已绑定到另一个 research_task"
                    )
                raise WorkingMemoryIntegrityConflict(
                    f"Module Run {normalized.module_invocation_id} 物化行缺少开始事件"
                )
            global_started = connection.execute(
                "SELECT research_task_id FROM research_task_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if global_started is not None and global_started["research_task_id"] != research_task_id:
                raise IdempotencyConflict(
                    f"Module Run {normalized.module_invocation_id} 已绑定到另一个 research_task"
                )
            current_sequence = facts.research_task_sequence
            if expected_research_task_sequence != current_sequence:
                raise CASConflict(
                    f"research_task {research_task_id} 的 sequence 已变化",
                    expected_version=expected_research_task_sequence,
                    current_version=current_sequence,
                )
            if facts.open_module_invocation is not None:
                raise OperationConflict(
                    f"research_task {research_task_id} 已有 ModuleInvocationOpen Module Invocation {facts.open_module_invocation.module_invocation_id}"
                )
            events = self._read_events_in_connection(connection, research_task_id)
            available_records = self._read_scientific_records_from_events(
                research_task_id,
                events,
            )
            missing_input_refs = [
                ref
                for ref in normalized.input_scientific_record_ids
                if ref not in available_records
            ]
            if missing_input_refs:
                raise WorkingMemoryIntegrityConflict(
                    "Module Invocation 的 input_scientific_record_ids 含当前 research_task 不存在的记录："
                    + ", ".join(missing_input_refs)
                )
            binding = normalized.context_binding
            source_invocation_id: str | None = None
            if isinstance(binding, ReuseInvestigationContext):
                source_invocation_id = binding.source_module_invocation_id
            elif isinstance(binding, ReuseResearchTaskReviewSession):
                source_invocation_id = binding.source_module_invocation_id
            if source_invocation_id is not None:
                source = next(
                    (
                        item
                        for item in facts.module_invocations
                        if item.module_invocation_id == source_invocation_id
                    ),
                    None,
                )
                if source is None:
                    foreign = connection.execute(
                        "SELECT research_task_id FROM module_invocations WHERE module_invocation_id = ?",
                        (source_invocation_id,),
                    ).fetchone()
                    if foreign is not None and foreign["research_task_id"] != research_task_id:
                        raise IdempotencyConflict(
                            "context_binding 来源 Module Invocation 属于另一个 research_task"
                        )
                    raise WorkingMemoryIntegrityConflict(
                        "context_binding 来源 Module Invocation 不存在于当前 research_task"
                    )
                if not isinstance(source.lifecycle, ModuleInvocationClosed):
                    raise WorkingMemoryIntegrityConflict(
                        "context_binding 来源 Module Invocation 必须已经闭合"
                    )
                if _module_kind_family(source.module_kind) != _module_kind_family(
                    normalized.module_kind
                ):
                    raise WorkingMemoryIntegrityConflict(
                        "context_binding 来源 Module Invocation 与当前模块类型不一致"
                    )
            started_event = self._append_event_in_connection(
                connection,
                research_task_id=research_task_id,
                event_id=event_id,
                event_type="ModuleInvocationStarted",
                payload=payload,
                expected_research_task_sequence=expected_research_task_sequence,
            )
            self._upsert_module_invocation_projection(
                connection,
                research_task_id=research_task_id,
                request=normalized,
                started_event_id=started_event.event_id,
            )
            facts = self._load_task_facts_in_connection(
                connection,
                research_task_id,
            )
            self._upsert_task_facts_projection(connection, facts)
            if facts.open_module_invocation is None:  # pragma: no cover - start event 已写入
                raise WorkingMemoryIntegrityConflict("Module Run 写入后无法读取")
            return facts.open_module_invocation


class ResearchTaskWorkingMemory:
    """V2 Working Memory 薄门面，只转发领域操作到深账本。"""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        enabled: bool = True,
        scratch_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """创建 V2 门面；关闭记忆时不创建数据库或写入任何事实。"""

        self.enabled = bool(enabled)
        # 保留旧服务构造器的可选参数；V2 不把展示/运行目录当作事实来源。
        self.scratch_root = None if scratch_root is None else os.fspath(scratch_root)
        self.ledger: ResearchTaskLedger | None = None
        self.db_path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))
        # 延迟构造深账本。迁移入口必须先核对 source Task 的架构版本；如果
        # 调用方误把 legacy_workflow_v1 research_task 交给迁移器，不能因为构造 facade
        # 就提前创建 v2_* 表。普通 V2 领域操作仍在第一次访问时初始化同一
        # 物理 SQLite。

    def _require_ledger(self) -> ResearchTaskLedger:
        """拒绝在 Memory Off 模式下写入或读取 V2 事实。"""

        if self.ledger is None:
            if not self.enabled:
                raise MemoryDisabledError("memory 已关闭，禁止访问 investigation_v2 账本")
            self.ledger = ResearchTaskLedger(self.db_path)
        return self.ledger

    def create_research_task(self, **kwargs: Any) -> ResearchTaskFacts:
        """转发 Task 创建领域操作。"""

        return self._require_ledger().create_research_task(**kwargs)

    def read_research_task_facts(self, research_task_id: str) -> ResearchTaskFacts:
        """转发 research_task facts 的语义化读取操作。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        return self._require_ledger().read_research_task_facts(research_task_id)

    def read_research_task_event_log(
        self,
        research_task_id: str,
    ) -> tuple[WorkingMemoryEvent, ...]:
        """转发只读审计查询。"""

        return self._require_ledger().read_research_task_event_log(research_task_id)

    def read_research_task_scientific_records(
        self,
        research_task_id: str,
        scientific_record_ids: Sequence[str] | None = None,
    ) -> tuple[ScientificRecord, ...]:
        """转发科学记录读取领域操作。"""

        return self._require_ledger().read_research_task_scientific_records(
            research_task_id,
            scientific_record_ids,
        )

    def start_module_invocation(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        request: ModuleInvocationRequest | Mapping[str, Any],
    ) -> ModuleInvocationSnapshot:
        """转发唯一 Module Run 开始领域操作。"""

        return self._require_ledger().start_module_invocation(
            research_task_id,
            expected_research_task_sequence,
            request,
        )

    def complete_module_invocation(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        output_bundle: PreparedModuleInvocationOutput | Mapping[str, Any],
    ) -> ModuleInvocationSnapshot:
        """转发原子模块输出接收领域操作。"""

        return self._require_ledger().complete_module_invocation(
            research_task_id,
            expected_research_task_sequence,
            output_bundle,
        )

    def close_research_task(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        outcome: Mapping[str, Any] | str,
    ) -> ResearchTaskFacts:
        """转发 TaskClosed 领域操作。"""

        return self._require_ledger().close_research_task(
            research_task_id,
            expected_research_task_sequence,
            outcome,
        )

    def fail_research_task(
        self,
        research_task_id: str,
        expected_research_task_sequence: int,
        *,
        module_invocation_id: str,
        failure_code: str,
    ) -> ResearchTaskFacts:
        """转发 ModuleInvocationFailed/ResearchTaskFailed 原子领域操作。"""

        return self._require_ledger().fail_research_task(
            research_task_id,
            expected_research_task_sequence,
            module_invocation_id=module_invocation_id,
            failure_code=failure_code,
        )

    def migrate_existing_task(
        self,
        research_task_id: str,
        *,
        source_root: str | os.PathLike[str] | None = None,
        runtime_resolver: Any | None = None,
        migration_version: str | None = None,
    ) -> Any:
        """在不提前构造 V2 ledger 的前提下迁移一个旧 investigation_v2 research_task。"""

        from .v2_migration import MIGRATION_VERSION, InvestigationV2Migration

        return InvestigationV2Migration(
            self.db_path,
            runtime_resolver=runtime_resolver,
            migration_version=(
                MIGRATION_VERSION if migration_version is None else migration_version
            ),
        ).migrate_existing_task(research_task_id, source_root=source_root)


__all__ = [
    "ModuleInvocationClosed",
    "ResearchTaskClosed",
    "ResearchTaskLedger",
    "ResearchTaskWorkingMemory",
    "ModuleInvocationRequest",
    "ModuleInvocationSnapshot",
    "NoOpenModuleInvocation",
    "ResearchTaskNotStarted",
    "ModuleInvocationOpen",
    "OpenModuleInvocation",
    "ResearchTaskOpen",
    "ScientificGraphFacts",
    "ScientificRecord",
    "PreparedModuleInvocationOutput",
    "ResearchTaskFacts",
    "ResearchTaskSnapshot",
    "V2LedgerSchemaConflict",
    "WorkingMemoryEvent",
    "WorkingMemoryIntegrityConflict",
]
