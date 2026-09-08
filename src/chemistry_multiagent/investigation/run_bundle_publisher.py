"""把 Investigation and Experiment 的冻结结果发布到专用 research_task Bundle。

本模块只接收已经完成协调的领域快照，不调用模型、MiniChem 或 Working Memory。
公开入口把可供研究员阅读的科学事实整理到 research_task 布局的阶段摘要中，
并把会话运行时身份留在内部状态之外。所有 Bundle 写入都通过 RunBundle 的公开
接口完成，因此恢复重放会复用同一个调用身份而不会增加阶段 entry。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, TypeAlias

from chemistry_multiagent.utils.run_bundle import (
    RESEARCH_TASK_STAGE_LAYOUT,
    RESEARCH_TASK_STAGE_NAMES,
    RunBundle,
)
from chemistry_multiagent.memory.contracts import (
    AUTHOR_MAIN_HYPOTHESIS_FIXTURE_IDS,
    AUTHOR_MAIN_HYPOTHESIS_MODES,
    canonical_json,
    get_investigation_combination_count,
    validate_investigation_task_config,
)

from .contracts import _require_non_empty_text, _require_safe_reference
from .coordinator import InvestigationBatchOutcome
from .domain import (
    BatchHistory,
    BatchValidation,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    ValidationFinding,
    classify_investigation_failure_kind,
)


_INVESTIGATION_ROUTES = frozenset(
    {
        "investigation",
        "waiting_investigation_job",
        "repair_required",
        "accept",
        "answerability_explanation",
        "regenerate_hypotheses",
        "failed",
    }
)
_TERMINAL_ROUTES = frozenset({"accept", "answerability_explanation", "failed"})
_RUN_STATUS_BY_ROUTE = {
    "investigation": "running",
    "waiting_investigation_job": "running",
    "repair_required": "running",
    "regenerate_hypotheses": "running",
    "accept": "succeeded",
    "answerability_explanation": "succeeded",
    "failed": "failed",
}
_STAGE_STATUS_BY_ROUTE = {
    "investigation": "running",
    "waiting_investigation_job": "running",
    "repair_required": "running",
    "regenerate_hypotheses": "completed",
    "accept": "completed",
    "answerability_explanation": "completed",
    "failed": "failed",
}

InvestigationPublicationInput: TypeAlias = InvestigationBatchOutcome | InvestigationState


@dataclass(frozen=True)
class InvestigationRunBundlePublication:
    """记录一次 Investigation 发布所写入的公开路径和状态。

    ``investigation_stage_path`` 总是指向阶段摘要文件；终局路由才会填写
    ``final_result_path``。路径只作为调用方本地定位信息返回，不写入阶段 JSON；
    ``call_identity`` 只包含任务、批次和排序后的结果版本向量。
    """

    investigation_stage_path: Path
    final_result_path: Path | None
    call_identity: Mapping[str, Any]
    route: str
    stage_status: str
    run_status: str
    outcome_kind: str | None
    revision: int | None


@dataclass(frozen=True)
class HypothesisRunBundlePublication:
    """记录已验证 Hypothesis manifest 的公开阶段投影。"""

    hypothesis_stage_path: Path
    call_identity: Mapping[str, Any]
    stage_status: str
    selected_count: int
    manifest_sha256: str


@dataclass(frozen=True)
class InvestigationRecoveryRunBundlePublication:
    """记录 Investigation 可恢复状态同步到 Bundle 的结果。"""

    investigation_stage_path: Path
    call_identity: Mapping[str, Any]
    stage_status: str
    run_status: str
    reason_code: str

def _validate_public_identifier(value: str, label: str) -> str:
    """校验会写入摘要或调用身份的标识，不允许路径、URI 和凭据。"""

    return _require_safe_reference(value, label)


def _validate_public_scientific_text(
    value: str,
    label: str,
    *,
    allow_public_source_uris: bool = False,
) -> str:
    """校验可公开的科学文本，拒绝本机路径、凭据和不安全 URI。"""

    return _require_non_empty_text(
        value,
        label,
        allow_public_source_uris=allow_public_source_uris,
    )


def _validate_investigation_projection_bundle(bundle: RunBundle) -> None:
    """确认投影目标是固定的 research_task Bundle。"""

    if not isinstance(bundle, RunBundle):
        raise TypeError("bundle 必须是 RunBundle")
    if bundle.stage_layout != RESEARCH_TASK_STAGE_LAYOUT or tuple(bundle.stage_names) != tuple(
        RESEARCH_TASK_STAGE_NAMES
    ):
        raise ValueError("RunBundle 必须使用 research_task 阶段布局")


def _select_hypothesis_manifest_reference(source_refs: Sequence[str]) -> str:
    """从 BatchCreated receipt 中选出唯一 manifest 相对引用。"""

    if isinstance(source_refs, (str, bytes)):
        raise TypeError("source_refs 必须是引用序列")
    references = [
        _validate_public_identifier(reference, f"source_refs[{index}]")
        for index, reference in enumerate(source_refs)
    ]
    manifest_refs = [
        reference
        for reference in references
        if PurePosixPath(reference).name.startswith("manifest_")
        and PurePosixPath(reference).name.endswith(".json")
    ]
    if len(manifest_refs) != 1:
        raise ValueError("BatchCreated receipt 必须唯一指向 Hypothesis manifest")
    return manifest_refs[0]


def _read_hypothesis_manifest_file(
    bundle: RunBundle,
    manifest_ref: str,
) -> tuple[dict[str, Any], str]:
    """读取 Bundle 内 manifest 并校验安全路径、JSON 类型和文件摘要。"""

    relative = PurePosixPath(manifest_ref)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} or "\\" in part for part in relative.parts
    ):
        raise ValueError("Hypothesis manifest 引用必须是 Bundle 内的安全相对路径")
    target = bundle.root.joinpath(*relative.parts)
    if target.is_symlink() or not target.is_file():
        raise ValueError("Hypothesis manifest 文件不存在或不是普通文件")
    try:
        target.resolve(strict=True).relative_to(bundle.root.resolve(strict=True))
        raw = target.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Hypothesis manifest 无法读取或解析") from error
    if not isinstance(manifest, dict):
        raise ValueError("Hypothesis manifest 必须是对象")
    digest = hashlib.sha256(raw).hexdigest()
    if target.name != f"manifest_{digest}.json":
        raise ValueError("Hypothesis manifest 文件名摘要不一致")
    return manifest, digest


def _validate_hypothesis_manifest_metadata(manifest: Mapping[str, Any]) -> str:
    """校验 Hypothesis manifest 的版本、操作和操作实例身份。"""

    if manifest.get("schema_version") != 1:
        raise ValueError("Hypothesis manifest schema_version 非法")
    operation = manifest.get("operation")
    if operation == "author_hypothesis_import":
        if manifest.get("schema") != "arche.investigation.author_hypothesis_import.v1":
            raise ValueError("作者导入 manifest schema 非法")
        if manifest.get("model_calls") != [] or manifest.get("generation_called") is not False:
            raise ValueError("作者导入 manifest 不得伪装模型调用")
    elif operation != "hypothesis_generation":
        raise ValueError("Hypothesis manifest operation 非法")
    return _validate_public_identifier(
        manifest.get("operation_instance_id"),
        "manifest.operation_instance_id",
    )


def _validate_frozen_task_created_hypothesis_summary(
    task_created: Any,
) -> dict[str, Any]:
    """提取并核对 Publisher 使用的 TaskCreated mode/count/fixture 摘要。"""

    if not isinstance(task_created, Mapping):
        raise ValueError("作者 Hypothesis 发布必须提供冻结 TaskCreated")
    try:
        task_config = validate_investigation_task_config(task_created.get("task_config"))
        expected_count = get_investigation_combination_count(task_config)
    except (TypeError, ValueError) as exc:
        raise ValueError("TaskCreated task_config 不是冻结的 Investigation 合同") from exc
    mode = task_config.get("investigation_mode", "standard")
    metadata = task_created.get("hypothesis_import")
    if mode in AUTHOR_MAIN_HYPOTHESIS_MODES:
        if not isinstance(metadata, Mapping):
            raise ValueError("作者 Hypothesis 发布缺少 TaskCreated fixture 摘要")
        expected_metadata_keys = {
            "mode",
            "fixture_id",
            "fixture_sha256",
            "manifest_schema",
            "expected_combination_count",
        }
        if mode == "author_case3_main_hypothesis_exploration":
            expected_metadata_keys.add("structure_input")
        if set(metadata) != expected_metadata_keys:
            raise ValueError("TaskCreated fixture 摘要字段不完整")
        if (
            metadata.get("mode") != mode
            or metadata.get("fixture_id")
            != AUTHOR_MAIN_HYPOTHESIS_FIXTURE_IDS[mode]
            or metadata.get("manifest_schema")
            != "arche.investigation.author_hypothesis_import.v1"
            or metadata.get("expected_combination_count") != 1
            or not isinstance(metadata.get("fixture_sha256"), str)
            or len(metadata["fixture_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in metadata["fixture_sha256"])
        ):
            raise ValueError("TaskCreated fixture 摘要非法")
        if mode == "author_case3_main_hypothesis_exploration":
            try:
                from .case3_prepared_inputs import (
                    validate_case3_prepared_input_metadata,
                )

                validate_case3_prepared_input_metadata(
                    metadata.get("structure_input")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "TaskCreated Case 3 prepared input 摘要非法"
                ) from exc
        return {
            "mode": mode,
            "expected_combination_count": expected_count,
            "fixture_id": metadata["fixture_id"],
            "fixture_sha256": metadata["fixture_sha256"],
            "manifest_schema": metadata["manifest_schema"],
        }
    if mode != "standard" or expected_count != 5 or metadata is not None:
        raise ValueError("标准 Hypothesis 发布的 TaskCreated mode/count 不符合冻结合同")
    return {
        "mode": "standard",
        "expected_combination_count": 5,
        "fixture_id": None,
        "fixture_sha256": None,
        "manifest_schema": None,
    }


def _validate_author_manifest_fixture_reference(
    bundle: RunBundle,
    manifest: Mapping[str, Any],
    task_summary: Mapping[str, Any],
) -> None:
    """核对作者 manifest 的 fixture hash 和内容寻址引用。"""

    if (
        manifest.get("fixture_id") != task_summary["fixture_id"]
        or manifest.get("fixture_sha256") != task_summary["fixture_sha256"]
        or manifest.get("schema") != task_summary["manifest_schema"]
        or manifest.get("expected_combination_count") != 1
    ):
        raise ValueError("作者 manifest 与 TaskCreated fixture 摘要不一致")
    fixture_ref = manifest.get("fixture_ref")
    if not isinstance(fixture_ref, str) or "\\" in fixture_ref:
        raise ValueError("作者 manifest.fixture_ref 非法")
    relative = PurePosixPath(fixture_ref)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("作者 manifest.fixture_ref 必须是 Bundle 内安全相对路径")
    fixture_path = bundle.root.joinpath(*relative.parts)
    if fixture_path.is_symlink() or not fixture_path.is_file():
        raise ValueError("作者 manifest.fixture_ref 文件不存在")
    try:
        fixture_path.resolve(strict=True).relative_to(
            bundle.root.resolve(strict=True)
        )
        raw = fixture_path.read_bytes()
        fixture = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("作者 manifest.fixture_ref 无法解析") from exc
    if not isinstance(fixture, Mapping) or canonical_json(dict(fixture)) != raw.decode("utf-8"):
        raise ValueError("作者 manifest.fixture_ref 不是规范 JSON")
    if hashlib.sha256(canonical_json(dict(fixture)).encode("utf-8")).hexdigest() != task_summary[
        "fixture_sha256"
    ]:
        raise ValueError("作者 manifest.fixture_ref 摘要与 TaskCreated 不一致")
    if fixture_path.name != f"fixture_{hashlib.sha256(raw).hexdigest()}.json":
        raise ValueError("作者 manifest.fixture_ref 文件名摘要不一致")


def _validate_hypothesis_manifest_stage(
    stage: Any,
    index: int,
    stage_indexes: set[int],
) -> int:
    """校验一个 manifest stage 已完成且最后一次尝试有效。"""

    if not isinstance(stage, Mapping):
        raise ValueError(f"manifest.stages[{index}] 必须是对象")
    stage_index = stage.get("stage_index", index)
    if isinstance(stage_index, bool) or not isinstance(stage_index, int):
        raise ValueError("Hypothesis manifest stage_index 非法")
    if stage_index in stage_indexes:
        raise ValueError("Hypothesis manifest stage_index 重复")
    stage_indexes.add(stage_index)
    _validate_public_identifier(
        stage.get("stage", f"stage-{stage_index}"),
        "manifest.stage",
    )
    if stage.get("status") != "validated":
        raise ValueError("Hypothesis manifest 仍有未验证 stage")
    attempts = stage.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("Hypothesis manifest stage 缺少 attempts")
    last_attempt = attempts[-1]
    if not isinstance(last_attempt, Mapping) or last_attempt.get("validation_status") != "valid":
        raise ValueError("Hypothesis manifest stage 的最后一次验证不是 valid")
    return len(attempts)


def _validate_hypothesis_manifest_stages(manifest: Mapping[str, Any]) -> dict[str, int]:
    """校验 manifest 中所有 stage 并统计已验证阶段和尝试次数。"""

    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Hypothesis manifest 缺少 stages")
    stage_indexes: set[int] = set()
    attempt_count = sum(
        _validate_hypothesis_manifest_stage(stage, index, stage_indexes)
        for index, stage in enumerate(stages)
    )
    return {
        "stage_count": len(stages),
        "validated_stage_count": len(stages),
        "attempt_count": attempt_count,
    }


def _read_validated_hypothesis_manifest(
    bundle: RunBundle,
    source_refs: Sequence[str],
    *,
    task_created: Mapping[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """读取并校验 receipt 指向的完整 Hypothesis manifest，返回引用、摘要和对象。"""

    manifest_ref = _select_hypothesis_manifest_reference(source_refs)
    manifest, digest = _read_hypothesis_manifest_file(bundle, manifest_ref)
    operation = manifest.get("operation")
    task_summary = None
    if operation == "author_hypothesis_import":
        task_summary = _validate_frozen_task_created_hypothesis_summary(task_created)
        if task_summary["mode"] not in AUTHOR_MAIN_HYPOTHESIS_MODES:
            raise ValueError("作者导入 manifest 不能绑定标准 TaskCreated")
        _validate_author_manifest_fixture_reference(bundle, manifest, task_summary)
        selected_count = task_summary["expected_combination_count"]
        if selected_count != 1:
            raise ValueError("作者导入 manifest 必须绑定冻结的单组合数量")
    else:
        if task_created is not None:
            task_summary = _validate_frozen_task_created_hypothesis_summary(task_created)
            if task_summary["mode"] != "standard":
                raise ValueError("标准 Hypothesis manifest 不能绑定作者 TaskCreated")
        selected_count = 5
    operation_instance_id = _validate_hypothesis_manifest_metadata(manifest)
    if operation == "author_hypothesis_import":
        summary = {
            "stage_count": 0,
            "validated_stage_count": 0,
            "attempt_count": 0,
            "selected_count": 1,
            "operation": "author_hypothesis_import",
            "model_calls": [],
        }
    else:
        summary = _validate_hypothesis_manifest_stages(manifest)
        summary["selected_count"] = selected_count
    return operation_instance_id, digest, summary


def _validate_hypothesis_batch_metadata(
    batch_payload: Mapping[str, Any],
    manifest_operation_instance_id: str,
) -> tuple[str, int, str, list[Any]]:
    """校验 BatchCreated 的批次、操作身份和候选数组。"""

    if not isinstance(batch_payload, Mapping):
        raise TypeError("batch_payload 必须是对象")
    batch_id = _validate_public_identifier(batch_payload.get("batch_id"), "batch.batch_id")
    batch_index = batch_payload.get("batch_index")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 1:
        raise ValueError("batch.batch_index 必须是正整数")
    operation_instance_id = _validate_public_identifier(
        batch_payload.get("operation_instance_id"),
        "batch.operation_instance_id",
    )
    if operation_instance_id != manifest_operation_instance_id:
        raise ValueError("BatchCreated 与 manifest operation_instance_id 不一致")
    candidates = batch_payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("BatchCreated candidates 必须是数组")
    return batch_id, batch_index, operation_instance_id, candidates


def _project_selected_hypothesis_candidate(
    candidate: Mapping[str, Any],
    *,
    expected_count: int,
) -> dict[str, Any]:
    """校验一个 selected candidate 并保留公开身份字段。"""

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count not in {1, 5}
    ):
        raise ValueError("selected combination 冻结数量只能是 1 或 5")
    combination_id = _validate_public_identifier(
        candidate.get("combination_id"),
        "candidate.combination_id",
    )
    hypothesis_id = _validate_public_identifier(
        candidate.get("hypothesis_id"),
        "candidate.hypothesis_id",
    )
    combination_index = candidate.get("combination_index")
    rank = candidate.get("rank")
    if (
        isinstance(combination_index, bool)
        or not isinstance(combination_index, int)
        or combination_index not in range(1, expected_count + 1)
    ):
        raise ValueError(
            f"selected combination_index 必须覆盖冻结的 1 到 {expected_count}"
        )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("selected rank 必须是正整数")
    return {
        "combination_id": combination_id,
        "combination_index": combination_index,
        "hypothesis_id": hypothesis_id,
        "rank": rank,
    }


def _collect_selected_hypothesis_combinations(
    candidates: list[Any],
    expected_count: int = 5,
) -> list[dict[str, Any]]:
    """收集并校验冻结数量 selected combination 的唯一身份。"""

    selected = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("selection_status") == "selected"
    ]
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count not in {1, 5}
        or len(selected) != expected_count
    ):
        if expected_count == 5:
            raise ValueError(
                "BatchCreated 必须覆盖五个 selected（manifest 冻结的数量）"
            )
        raise ValueError("BatchCreated 必须覆盖 manifest 冻结的 selected 数量")
    projected: list[dict[str, Any]] = []
    indexes: set[int] = set()
    combination_ids: set[str] = set()
    hypothesis_ids: set[str] = set()
    for candidate in selected:
        item = _project_selected_hypothesis_candidate(
            candidate,
            expected_count=expected_count,
        )
        if (
            item["combination_index"] in indexes
            or item["combination_id"] in combination_ids
            or item["hypothesis_id"] in hypothesis_ids
        ):
            raise ValueError("selected combination 身份必须唯一")
        indexes.add(item["combination_index"])
        combination_ids.add(item["combination_id"])
        hypothesis_ids.add(item["hypothesis_id"])
        projected.append(item)
    if {item["combination_index"] for item in projected} != set(
        range(1, expected_count + 1)
    ):
        raise ValueError("selected combination_index 必须完整覆盖冻结数量")
    projected.sort(key=lambda item: item["combination_index"])
    return projected


def _build_hypothesis_stage_projection(
    batch_payload: Mapping[str, Any],
    *,
    research_task_id: str,
    manifest_operation_instance_id: str,
    manifest_sha256: str,
    manifest_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """校验 BatchCreated 的冻结数量 selected combination 并生成脱敏摘要和调用身份。"""

    batch_id, batch_index, operation_instance_id, candidates = (
        _validate_hypothesis_batch_metadata(
            batch_payload,
            manifest_operation_instance_id,
        )
    )
    expected_count = manifest_summary.get("selected_count", 5)
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count not in {1, 5}
    ):
        raise ValueError("Hypothesis manifest selected_count 非法")
    projected = _collect_selected_hypothesis_combinations(candidates, expected_count)
    summary = {
        "status": "completed",
        "batch_id": batch_id,
        "batch_index": batch_index,
        "operation_instance_id": operation_instance_id,
        "manifest": {
            "sha256": _validate_public_identifier(manifest_sha256, "manifest.sha256"),
            "stage_count": manifest_summary["stage_count"],
            "validated_stage_count": manifest_summary["validated_stage_count"],
            "attempt_count": manifest_summary["attempt_count"],
        },
        "selected_count": len(projected),
        "selected_combinations": projected,
    }
    call_identity = {
        "research_task_id": _validate_public_identifier(
            research_task_id,
            "research_task_id",
        ),
        "batch_id": batch_id,
        "operation_instance_id": operation_instance_id,
        "manifest_sha256": manifest_sha256,
    }
    return summary, call_identity


def publish_hypothesis_batch_to_run_bundle(
    bundle: RunBundle,
    *,
    batch_payload: Mapping[str, Any],
    source_refs: Sequence[str],
    task_created: Mapping[str, Any] | None = None,
) -> HypothesisRunBundlePublication:
    """将已验证的 BatchCreated 投影为 Hypothesis completed 阶段摘要。

    输入是 research_task 布局的 RunBundle、Working Memory BatchCreated payload 和
    receipt 的 source_refs；动作读取 Bundle 内带 SHA-256 文件名的 manifest，核对所有
    stage 已 validated 且最后一次校验有效，再按冻结 TaskCreated 核对 selected
    combination 数量。作者单组合 manifest 还必须绑定 TaskCreated 的 mode/count、
    fixture 摘要和 fixture_ref，随后原子更新 ``stages/hypothesis_preparation.json`` 的状态和摘要。
    副作用只写 Hypothesis 阶段文件、``summary.json`` 的阶段索引，不调用模型或写入科学
    正文。返回值包含阶段路径、稳定调用身份和 manifest 摘要；Bundle 布局、文件摘要、
    manifest、TaskCreated 绑定、批次身份或冻结数量校验失败时在任何写入前抛出异常。
    """

    _validate_investigation_projection_bundle(bundle)
    manifest_operation_instance_id, manifest_sha256, manifest_summary = (
        _read_validated_hypothesis_manifest(
            bundle,
            source_refs,
            task_created=task_created,
        )
    )
    summary, call_identity = _build_hypothesis_stage_projection(
        batch_payload,
        research_task_id=bundle.research_task_id,
        manifest_operation_instance_id=manifest_operation_instance_id,
        manifest_sha256=manifest_sha256,
        manifest_summary=manifest_summary,
    )
    hypothesis_stage_path = bundle.write_stage_summary(
        "hypothesis_preparation",
        summary,
        source_file=next(
            reference
            for reference in source_refs
            if PurePosixPath(reference).name.startswith("manifest_")
            and PurePosixPath(reference).name.endswith(".json")
        ),
        call_identity=call_identity,
    )
    bundle.update_stage_status("hypothesis_preparation", "completed", summary=summary)
    return HypothesisRunBundlePublication(
        hypothesis_stage_path=hypothesis_stage_path,
        call_identity=call_identity,
        stage_status="completed",
        selected_count=summary["selected_count"],
        manifest_sha256=manifest_sha256,
    )


def _project_recovery_blocking(blocking: Mapping[str, Any]) -> dict[str, Any]:
    """只保留可公开的恢复阻塞字段，不复制异常正文或清理明细。"""

    if not isinstance(blocking, Mapping):
        raise TypeError("blocking 必须是对象")
    projected = {
        "kind": _validate_public_identifier(blocking.get("kind"), "blocking.kind"),
        "error_type": _validate_public_identifier(
            blocking.get("error_type"),
            "blocking.error_type",
        ),
    }
    cleanup_plan = blocking.get("cleanup_plan")
    if cleanup_plan is not None:
        if not isinstance(cleanup_plan, Mapping):
            raise ValueError("blocking.cleanup_plan 必须是对象")
        cleanup_projection: dict[str, Any] = {}
        for field in (
            "schema",
            "status",
            "configuration_digest",
            "plan_digest",
        ):
            if field in cleanup_plan:
                cleanup_projection[field] = _validate_public_identifier(
                    cleanup_plan[field],
                    f"blocking.cleanup_plan.{field}",
                )
        for field in ("pending_count", "revoked_count"):
            if field in cleanup_plan:
                value = cleanup_plan[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"blocking.cleanup_plan.{field} 必须是非负整数")
                cleanup_projection[field] = value
        projected["cleanup_plan"] = cleanup_projection
    return projected


def _project_waiting_investigation_jobs(value: Any) -> list[dict[str, str]]:
    """校验并复制 V2 waiting Investigation job 的恢复身份。"""

    if not isinstance(value, list):
        raise ValueError("recovery.waiting_investigation_jobs 必须是数组")
    projected: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"recovery.waiting_investigation_jobs[{index}] 必须是对象"
            )
        projected.append(
            {
                field: _validate_public_identifier(
                    item.get(field),
                    f"recovery.waiting_investigation_jobs[{index}].{field}",
                )
                for field in ("job_id", "operation_id", "submission_key")
            }
        )
    return projected


def _project_recovery_pauses(value: Any) -> list[dict[str, Any]]:
    """校验并复制 pause 的稳定恢复身份。"""

    if not isinstance(value, list):
        raise ValueError("recovery.pauses 必须是数组")
    projected: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"recovery.pauses[{index}] 必须是对象")
        restart_allowed = item.get("restart_allowed")
        if not isinstance(restart_allowed, bool):
            raise ValueError("recovery.pauses.restart_allowed 必须是布尔值")
        projected.append(
            {
                "pause_id": _validate_public_identifier(
                    item.get("pause_id"),
                    f"recovery.pauses[{index}].pause_id",
                ),
                "operation": _validate_public_identifier(
                    item.get("operation"),
                    f"recovery.pauses[{index}].operation",
                ),
                "reason_code": _validate_public_identifier(
                    item.get("reason_code"),
                    f"recovery.pauses[{index}].reason_code",
                ),
                "restart_allowed": restart_allowed,
            }
        )
    return projected


def _project_recovery_state(recovery: Mapping[str, Any]) -> dict[str, Any]:
    """复制固定恢复字段并核对等待作业和暂停操作身份。"""

    if not isinstance(recovery, Mapping):
        raise TypeError("recovery 必须是对象")
    waiting_projection = _project_waiting_investigation_jobs(
        recovery.get("waiting_investigation_jobs", [])
    )
    pauses_projection = _project_recovery_pauses(recovery.get("pauses", []))
    if not waiting_projection and not pauses_projection:
        raise ValueError(
            "可恢复 Investigation 状态必须有 waiting Investigation job 或 pause"
        )
    return {
        "available": True,
        "waiting_investigation_jobs": waiting_projection,
        "pauses": pauses_projection,
    }


def publish_investigation_recovery_to_run_bundle(
    bundle: RunBundle,
    *,
    reason_code: str,
    blocking: Mapping[str, Any],
    recovery: Mapping[str, Any],
) -> InvestigationRecoveryRunBundlePublication:
    """同步可恢复 Investigation 失败的阶段和根状态，不生成虚假科学结果。

    输入是 research_task 布局的 RunBundle、稳定 reason_code、阻塞类型摘要和恢复
    字段；动作只复制安全的类型、暂停及等待作业身份，先把
    ``investigation_and_experiment`` 阶段置为 running，再把同一份 blocking/recovery
    写入根 status 和 summary。副作用不会追加 InvestigationResult、阶段 entry 或
    科学结论；返回值包含阶段路径、稳定调用身份和 running 状态。输入包含异常正文、
    路径、密钥、非法恢复字段或 Bundle 布局错误时在写入前抛出异常。
    """

    _validate_investigation_projection_bundle(bundle)
    safe_reason_code = _validate_public_identifier(reason_code, "reason_code")
    safe_blocking = _project_recovery_blocking(blocking)
    safe_recovery = _project_recovery_state(recovery)
    summary = {
        "status": "running",
        "reason_code": safe_reason_code,
        "blocking": safe_blocking,
        "recovery": safe_recovery,
    }
    call_identity = {
        "research_task_id": _validate_public_identifier(
            bundle.research_task_id,
            "research_task_id",
        ),
        "reason_code": safe_reason_code,
        "blocking_kind": safe_blocking["kind"],
    }
    investigation_stage_path = bundle.root / "stages" / "investigation_and_experiment.json"
    bundle.update_stage_status(
        "investigation_and_experiment",
        "running",
        summary=summary,
        blocking=safe_blocking,
        recovery=safe_recovery,
    )
    bundle.update_status(
        "running",
        reason_code=safe_reason_code,
        blocking=safe_blocking,
        recovery=safe_recovery,
    )
    return InvestigationRecoveryRunBundlePublication(
        investigation_stage_path=investigation_stage_path,
        call_identity=call_identity,
        stage_status="running",
        run_status="running",
        reason_code=safe_reason_code,
    )


def _project_opaque_references(
    references: tuple[str, ...],
    label: str,
) -> list[str]:
    """只复制不透明引用标识，不解析、读取或展开其指向内容。"""

    return [
        _validate_public_identifier(reference, f"{label}[{index}]")
        for index, reference in enumerate(references)
    ]


def _sorted_result_version_vector(
    combination_ids: tuple[str, ...],
    result_versions: tuple[str, ...],
    *,
    label: str,
) -> list[dict[str, str]]:
    """校验并返回按 combination_id 排序的当前批次版本向量。"""

    if len(combination_ids) not in {1, 5} or len(result_versions) != len(combination_ids):
        raise ValueError(f"{label} 必须包含完整的一项或五项 result_version 向量")
    if len(set(combination_ids)) != len(combination_ids):
        raise ValueError(f"{label} 的 combination_id 必须唯一")
    pairs = [
        (
            _validate_public_identifier(combination_id, f"{label}.combination_id"),
            _validate_public_identifier(result_version, f"{label}.result_version"),
        )
        for combination_id, result_version in zip(combination_ids, result_versions)
    ]
    return [
        {"combination_id": combination_id, "result_version": result_version}
        for combination_id, result_version in sorted(pairs)
    ]


def _build_call_identity(
    research_task_id: str,
    batch_id: str,
    result_version_vector: list[dict[str, str]],
) -> dict[str, Any]:
    """建立恢复重放使用的稳定调用身份。"""

    return {
        "research_task_id": _validate_public_identifier(
            research_task_id,
            "research_task_id",
        ),
        "batch_id": _validate_public_identifier(batch_id, "batch_id"),
        "result_version_vector": [
            [item["combination_id"], item["result_version"]]
            for item in result_version_vector
        ],
    }


def _project_investigation_result(
    session: InvestigationSession,
    result: InvestigationResult,
) -> dict[str, Any]:
    """复制一个当前结果的科学字段，不携带运行时身份。"""

    if result.combination_id != session.combination_id:
        raise ValueError("当前结果与会话 combination_id 不一致")
    if session.result_version != result.result_version:
        raise ValueError("当前结果与会话 result_version 不一致")
    return {
        "combination_id": _validate_public_identifier(
            result.combination_id,
            "result.combination_id",
        ),
        "result_version": _validate_public_identifier(
            result.result_version,
            "result.result_version",
        ),
        "repair_count": session.repair_count,
        "self_judgment": result.self_judgment,
        "narrative": _validate_public_scientific_text(
            result.narrative,
            "result.narrative",
        ),
        "observations": [
            _validate_public_scientific_text(
                observation,
                f"result.observations[{index}]",
            )
            for index, observation in enumerate(result.observations)
        ],
        "limitations": [
            _validate_public_scientific_text(
                limitation,
                f"result.limitations[{index}]",
            )
            for index, limitation in enumerate(result.limitations)
        ],
        "job_refs": _project_opaque_references(result.job_refs, "result.job_refs"),
        "operation_refs": _project_opaque_references(
            result.operation_refs,
            "result.operation_refs",
        ),
        "artifact_refs": _project_opaque_references(
            result.artifact_refs,
            "result.artifact_refs",
        ),
    }


def _project_validation_finding(finding: ValidationFinding) -> dict[str, Any]:
    """复制一个批次验证判断，不携带审查会话身份。"""

    payload: dict[str, Any] = {
        "combination_id": _validate_public_identifier(
            finding.combination_id,
            "finding.combination_id",
        ),
        "validity": finding.validity,
        "hypothesis_judgment": finding.hypothesis_judgment,
    }
    if finding.repair_advice is not None:
        payload["repair_advice"] = _validate_public_scientific_text(
            finding.repair_advice,
            "finding.repair_advice",
        )
    return payload


def _project_batch_validation(validation: BatchValidation) -> dict[str, Any]:
    """整理一条批次验证记录及其完整结果版本组合。"""

    result_version_vector = _sorted_result_version_vector(
        validation.combination_ids,
        validation.result_versions,
        label="validation",
    )
    payload: dict[str, Any] = {
        "research_task_id": _validate_public_identifier(
            validation.task_id,
            "validation.research_task_id",
        ),
        "batch_id": _validate_public_identifier(validation.batch_id, "validation.batch_id"),
        "result_version_vector": result_version_vector,
        "findings": [
            _project_validation_finding(finding)
            for finding in sorted(validation.findings, key=lambda item: item.combination_id)
        ],
        "can_answer": validation.can_answer,
    }
    if validation.final_answer is not None:
        payload["final_answer"] = _validate_public_scientific_text(
            validation.final_answer,
            "validation.final_answer",
            allow_public_source_uris=True,
        )
    if validation.answerability_explanation is not None:
        payload["answerability_explanation"] = _validate_public_scientific_text(
            validation.answerability_explanation,
            "validation.answerability_explanation",
        )
    return payload


def _project_batch_history(history: BatchHistory) -> dict[str, Any]:
    """整理已关闭批次的公开路由、版本和返修计数。"""

    combination_ids = tuple(session.combination_id for session in history.sessions)
    result_versions = tuple(session.result_version or "" for session in history.sessions)
    result_version_vector = _sorted_result_version_vector(
        combination_ids,
        result_versions,
        label="history",
    )
    return {
        "research_task_id": _validate_public_identifier(
            history.task_id,
            "history.research_task_id",
        ),
        "batch_id": _validate_public_identifier(history.batch_id, "history.batch_id"),
        "route": history.route,
        "result_version_vector": result_version_vector,
        "repair_counts": [
            {
                "combination_id": _validate_public_identifier(
                    session.combination_id,
                    "history.combination_id",
                ),
                "repair_count": session.repair_count,
            }
            for session in sorted(history.sessions, key=lambda item: item.combination_id)
        ],
        "validations": [
            _project_batch_validation(validation)
            for validation in history.validations
        ],
    }


def _validate_and_freeze_input(
    outcome: InvestigationPublicationInput,
) -> tuple[InvestigationState, int | None, dict[str, Any] | None]:
    """校验 outcome/state 一致性并生成只读协调元数据。"""

    if isinstance(outcome, InvestigationBatchOutcome):
        if not isinstance(outcome.state, InvestigationState):
            raise TypeError("outcome.state 必须是 InvestigationState")
        if outcome.route != outcome.state.route:
            raise ValueError("outcome.route 与 outcome.state.route 不一致")
        if isinstance(outcome.revision, bool) or not isinstance(outcome.revision, int):
            raise TypeError("outcome.revision 必须是非负整数")
        if outcome.revision < 0:
            raise ValueError("outcome.revision 必须是非负整数")
        counters = {
            "review_called": outcome.review_called,
            "initial_investigator_calls": outcome.initial_investigator_calls,
            "repair_investigator_calls": outcome.repair_investigator_calls,
            "review_calls": outcome.review_calls,
            "replay_count": outcome.replay_count,
        }
        if not isinstance(outcome.review_called, bool):
            raise TypeError("outcome.review_called 必须是布尔值")
        for name in (
            "initial_investigator_calls",
            "repair_investigator_calls",
            "review_calls",
            "replay_count",
        ):
            value = counters[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"outcome.{name} 必须是非负整数")
            if value < 0:
                raise ValueError(f"outcome.{name} 必须是非负整数")
        metadata: dict[str, Any] | None = counters
        revision: int | None = outcome.revision
        source_state = outcome.state
    elif isinstance(outcome, InvestigationState):
        metadata = None
        revision = None
        source_state = outcome
    else:
        raise TypeError(
            "发布输入必须是 InvestigationBatchOutcome 或 InvestigationState"
        )

    # 通过公开模型重新校验一次，防止 model_copy/update 或恢复反序列化留下不完整
    # 状态；返回的对象只用于本次发布，不会修改调用方持有的冻结快照。
    state = InvestigationState.model_validate(
        source_state.model_dump(exclude_none=True),
    )
    if state.route not in _INVESTIGATION_ROUTES:
        raise ValueError("InvestigationState route 非法")
    return state, revision, metadata


def _validate_current_batch_sessions(
    state: InvestigationState,
    *,
    require_complete_result_vector: bool = True,
) -> tuple[dict[str, InvestigationSession], list[tuple[str, str]]]:
    """校验当前批次会话，并返回按组合索引的当前结果版本。

    等待长作业时允许尚未形成结果的会话不进入版本向量；其它路由仍要求冻结
    数量的所有会话都已有当前结果，缺少任意一项时立即失败。
    """

    if len(state.sessions) != state.expected_combination_count:
        if state.expected_combination_count == 5:
            raise ValueError("InvestigationState 必须包含完整五组合（冻结数量）")
        raise ValueError("InvestigationState 必须包含冻结数量的组合会话")
    session_by_combination = {session.combination_id: session for session in state.sessions}
    if len(session_by_combination) != state.expected_combination_count:
        raise ValueError("InvestigationState 的 combination_id 必须唯一")
    current_pairs: list[tuple[str, str]] = []
    for session in state.sessions:
        if session.result_version is None:
            if require_complete_result_vector:
                raise ValueError("当前组合 result_version 向量不完整")
            continue
        current_pairs.append(
            (
                _validate_public_identifier(session.combination_id, "session.combination_id"),
                _validate_public_identifier(session.result_version, "session.result_version"),
            )
        )
    current_pairs.sort()
    if len(set(current_pairs)) != len(current_pairs):
        raise ValueError("当前组合 result_version 向量包含重复项")
    if (
        require_complete_result_vector
        and len(current_pairs) != state.expected_combination_count
    ):
        raise ValueError("当前组合 result_version 向量不完整")
    return session_by_combination, current_pairs


def _collect_current_batch_results(
    state: InvestigationState,
    current_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], InvestigationResult]:
    """收集并核对当前五组合对应的唯一 InvestigationResult。"""

    result_by_key: dict[tuple[str, str], InvestigationResult] = {}
    for result in state.results:
        key = (result.combination_id, result.result_version)
        if key in result_by_key:
            raise ValueError("当前结果版本重复")
        if key in current_pairs:
            result_by_key[key] = result
    if set(result_by_key) != set(current_pairs):
        raise ValueError("当前五组合 result_version 缺少对应 InvestigationResult")
    return result_by_key


def _build_current_batch_repair_counts(
    state: InvestigationState,
) -> list[dict[str, int]]:
    """构造按组合排序的当前批次返修计数摘要。"""

    return [
        {
            "combination_id": _validate_public_identifier(
                session.combination_id,
                "session.combination_id",
            ),
            "repair_count": session.repair_count,
        }
        for session in sorted(state.sessions, key=lambda item: item.combination_id)
    ]


def _build_current_batch_projection(
    state: InvestigationState,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, int]]]:
    """校验当前结果，并构造版本、结果和返修计数摘要。

    ``waiting_investigation_job`` 表示物理作业已经提交但尚未产生科学结果，允许
    发布空或部分版本向量；其它路由继续要求完整结果屏障。
    """

    session_by_combination, current_pairs = _validate_current_batch_sessions(
        state,
        require_complete_result_vector=(state.route != "waiting_investigation_job"),
    )
    result_by_key = _collect_current_batch_results(state, current_pairs)

    current_results = [
        _project_investigation_result(
            session_by_combination[combination_id],
            result_by_key[(combination_id, result_version)],
        )
        for combination_id, result_version in current_pairs
    ]
    repair_counts = _build_current_batch_repair_counts(state)
    result_version_vector = [
        {"combination_id": combination_id, "result_version": result_version}
        for combination_id, result_version in current_pairs
    ]
    return result_version_vector, current_results, repair_counts


def _build_validation_projection(state: InvestigationState) -> list[dict[str, Any]]:
    """按历史批次顺序合并所有已保存的 validation 摘要。"""

    validations: list[dict[str, Any]] = []
    for history in state.history:
        for validation in history.validations:
            validations.append(_project_batch_validation(validation))
    for validation in state.validations:
        validations.append(_project_batch_validation(validation))
    return validations


def _build_investigation_stage_summary(
    state: InvestigationState,
    *,
    result_version_vector: list[dict[str, str]],
    current_results: list[dict[str, Any]],
    repair_counts: list[dict[str, int]],
    coordination: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """构造不含会话运行时身份的 Investigation 阶段摘要。"""

    summary: dict[str, Any] = {
        "status": _STAGE_STATUS_BY_ROUTE[state.route],
        "research_task_id": _validate_public_identifier(
            state.task_id,
            "state.research_task_id",
        ),
        "batch_id": _validate_public_identifier(state.batch_id, "state.batch_id"),
        "route": state.route,
        "result_version_vector": result_version_vector,
        "current_results": current_results,
        "repair_counts": repair_counts,
        "validations": _build_validation_projection(state),
        "previous_batches": [
            _project_batch_history(history) for history in state.history
        ],
    }
    if coordination is not None:
        summary["coordination"] = dict(coordination)
    if state.route == "regenerate_hypotheses":
        summary["regeneration_request"] = {
            "requested": True,
            "reason_code": "all_combinations_marked_for_hypothesis_revision",
            "next_stage": "hypothesis",
        }
    elif state.route == "failed":
        if (
            state.failure_reason_code is None
            or state.failure_error_type is None
        ):
            raise ValueError("failed 路由缺少失败原因摘要")
        summary["failure"] = {
            "failure_kind": classify_investigation_failure_kind(
                state.failure_reason_code
            ),
            "reason_code": _validate_public_identifier(
                state.failure_reason_code,
                "state.failure_reason_code",
            ),
            "error_type": _validate_public_identifier(
                state.failure_error_type,
                "state.failure_error_type",
            ),
        }
    return summary


def _build_final_result_summary(
    state: InvestigationState,
    result_version_vector: list[dict[str, str]],
) -> dict[str, Any]:
    """构造成功或失败终局的摘要。"""

    base: dict[str, Any] = {
        "status": "failed" if state.route == "failed" else "succeeded",
        "outcome_kind": state.route,
        "route": state.route,
        "research_task_id": _validate_public_identifier(
            state.task_id,
            "state.research_task_id",
        ),
        "batch_id": _validate_public_identifier(state.batch_id, "state.batch_id"),
        "result_version_vector": result_version_vector,
    }
    if state.route == "accept":
        if state.final_answer is None:
            raise ValueError("accept 路由缺失 final_answer")
        base["final_answer"] = _validate_public_scientific_text(
            state.final_answer,
            "state.final_answer",
            allow_public_source_uris=True,
        )
    elif state.route == "answerability_explanation":
        if state.answerability_explanation is None:
            raise ValueError("answerability_explanation 路由缺失说明")
        base["answerability_explanation"] = _validate_public_scientific_text(
            state.answerability_explanation,
            "state.answerability_explanation",
        )
    elif state.route == "failed":
        if (
            state.failure_reason_code is None
            or state.failure_error_type is None
        ):
            raise ValueError("failed 路由缺少失败原因摘要")
        reason_code = _validate_public_identifier(
            state.failure_reason_code,
            "state.failure_reason_code",
        )
        error_type = _validate_public_identifier(
            state.failure_error_type,
            "state.failure_error_type",
        )
        base["reason_code"] = reason_code
        base["failure"] = {
            "failure_kind": classify_investigation_failure_kind(reason_code),
            "error_type": error_type,
            "reason_code": reason_code,
        }
    else:
        raise ValueError("非终局路由不能写入 final_result")
    return base


def publish_investigation_batch_to_run_bundle(
    bundle: RunBundle,
    outcome: InvestigationPublicationInput,
) -> InvestigationRunBundlePublication:
    """发布一个冻结 Investigation 批次并按领域路由更新 Run Bundle。

    输入是已经创建且布局冻结为 ``investigation`` 的 ``RunBundle``，以及同一份
    ``InvestigationBatchOutcome`` 或 ``InvestigationState``。动作会校验完整五项当前
    结果版本；等待长作业时允许尚未形成结果的会话不进入版本向量。动作整理当前
    结果、返修计数和全部历史 validation，随后通过 RunBundle 公开写入接口更新
    Investigation 阶段；accept 和回答能力说明路由还会写入 task_review_and_result 并把运行
    标为 succeeded，等待和重生成路由保持 running。
    返回值包含写入路径、稳定调用身份和最终状态；任何布局、快照一致性、版本向量、
    终局文本或公开字段校验失败都会在写入前抛出异常，避免产生伪造的成功结果。
    """

    if not isinstance(bundle, RunBundle):
        raise TypeError("bundle 必须是 RunBundle")
    if bundle.stage_layout != RESEARCH_TASK_STAGE_LAYOUT or tuple(bundle.stage_names) != tuple(
        RESEARCH_TASK_STAGE_NAMES
    ):
        raise ValueError("RunBundle 必须使用 research_task 阶段布局")

    state, revision, coordination = _validate_and_freeze_input(outcome)
    result_version_vector, current_results, repair_counts = _build_current_batch_projection(
        state
    )
    call_identity = _build_call_identity(
        state.task_id,
        state.batch_id,
        result_version_vector,
    )
    investigation_summary = _build_investigation_stage_summary(
        state,
        result_version_vector=result_version_vector,
        current_results=current_results,
        repair_counts=repair_counts,
        coordination=coordination,
    )

    investigation_stage_path = bundle.write_stage_summary(
        "investigation_and_experiment",
        investigation_summary,
        call_identity=call_identity,
    )
    bundle.update_stage_status(
        "investigation_and_experiment",
        _STAGE_STATUS_BY_ROUTE[state.route],
        summary=investigation_summary,
    )

    final_result_path: Path | None = None
    outcome_kind: str | None = None
    if state.route in _TERMINAL_ROUTES:
        final_summary = _build_final_result_summary(state, result_version_vector)
        final_result_path = bundle.write_stage_summary(
            "task_review_and_result",
            final_summary,
            call_identity=call_identity,
        )
        bundle.update_stage_status(
            "task_review_and_result",
            # task_review_and_result 表示终局结果发布阶段；即使结果是 workflow failed，
            # 只要失败摘要已发布，该阶段本身就是 completed。失败语义由根状态、
            # final_result.summary.status 和 Investigation 阶段的 failed 状态承载。
            "completed",
            summary=final_summary,
        )
        bundle.update_status(_RUN_STATUS_BY_ROUTE[state.route])
        outcome_kind = state.route
    elif state.route == "regenerate_hypotheses":
        bundle.update_status("running")
    else:
        bundle.update_status("running")

    return InvestigationRunBundlePublication(
        investigation_stage_path=investigation_stage_path,
        final_result_path=final_result_path,
        call_identity=call_identity,
        route=state.route,
        stage_status=_STAGE_STATUS_BY_ROUTE[state.route],
        run_status=_RUN_STATUS_BY_ROUTE[state.route],
        outcome_kind=outcome_kind,
        revision=revision,
    )


__all__ = [
    "InvestigationPublicationInput",
    "HypothesisRunBundlePublication",
    "InvestigationRunBundlePublication",
    "InvestigationRecoveryRunBundlePublication",
    "publish_hypothesis_batch_to_run_bundle",
    "publish_investigation_batch_to_run_bundle",
    "publish_investigation_recovery_to_run_bundle",
]
