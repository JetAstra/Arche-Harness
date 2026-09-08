"""Investigation 终局到下一批假设的安全重生 bridge。

本模块把新 Investigation 领域状态、旧 Working Memory 事件写入和现有
Hypothesis 生成依赖接在一个可恢复的窄接口上。两个数据库不能共享一个事务，
因此 bridge 先保存 receipt，再按 ``generated``、``wm_written`` 和
``completed`` 阶段恢复；恢复只重试尚未确认的写入，不会再次调用生成器。生产
Controller 与迁移测试都必须通过本模块进入 Hypothesis 阶段。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from chemistry_multiagent.agents.hypothesis_agent import (
    HypothesisAgent,
    HypothesisProtocolError,
    _read_consumed_hypothesis_ids,
)
from chemistry_multiagent.memory.contracts import (
    INVESTIGATION_PROTOCOL_VERSION,
    validate_structured_hypothesis_claims,
)
from chemistry_multiagent.memory.projections import (
    project_hypothesis_regeneration_context,
)
from chemistry_multiagent.utils.model_operation import DynamicModelOperationRunner

from .domain import InvestigationSession, InvestigationState
from .sqlite_store import (
    InvestigationRegenerationReceiptSnapshot,
    InvestigationRevisionConflict,
)
from .working_memory import (
    HypothesisRegenerationWorkingMemoryInput,
    WorkingMemoryBatchReceiptReader,
    WorkingMemoryReader,
    build_hypothesis_regeneration_working_memory_input,
)


__all__ = [
    "HypothesisInitialGenerationRequest",
    "HypothesisAgentRegenerationGenerator",
    "HypothesisInitialGenerationGenerator",
    "HypothesisRegenerationBridgeError",
    "HypothesisRegenerationGenerationConfig",
    "HypothesisRegenerationGenerationRequest",
    "HypothesisRegenerationGenerator",
    "HypothesisRegenerationSessionIdentity",
    "HypothesisRegenerationBridgeOutcome",
    "InvestigationScopeRevokeProtocol",
    "InvestigationHypothesisRegenerationBridge",
    "WorkingMemoryHypothesisBatchWriter",
    "InvestigationRegenerationPersistence",
]


_SAFE_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_OPAQUE_SOURCE_RECORD_ID_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]{7,127}$"
)
_ABSOLUTE_PATH_RE = re.compile(r"(?:^|\s)(?:/|[A-Za-z]:[\\/]|file://)")
_SECRET_RE = re.compile(r"(?:api[_ -]?key|access[_ -]?token|secret|password)\s*=", re.I)
_BUSY_OPERATION_MARKERS = (
    "operation still in progress",
    "operation busy",
    "operation is busy",
    "不能同时运行",
    "不能同时恢复",
    "正在运行",
)
_CLEANUP_SCHEMA = "arche.investigation.hypothesis_regeneration_cleanup"
_CLEANUP_STATUS_VALUES = frozenset({"not_required", "cleanup_pending", "completed"})
_CLEANUP_ITEM_STATUS_VALUES = frozenset({"pending", "revoked"})
_RECEIPT_SCHEMA = "arche.investigation.hypothesis_regeneration_receipt"
_RECEIPT_STATUS_VALUES = frozenset(
    {"started", "generated", "wm_written", "completed", "failed"}
)
_RECEIPT_BASE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "task_id",
        "previous_batch_id",
        "next_batch_id",
        "next_batch_index",
        "operation_key",
        "operation_instance_id",
        "input_digest",
        "model_request_digest",
        "operation_reference",
        "operation_reference_digest",
        "generation_config_digest",
        "session_identities",
        "source_record_ids",
        "causal_source_ref",
        "cleanup",
    }
)
_RECEIPT_OUTPUT_FIELDS = frozenset(
    {"source_refs", "batch_event_id", "output_digest"}
)
_RECEIPT_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ARTIFACT_REF_KEYS = frozenset({"relative_path", "sha256", "media_type", "size"})
_FINALIZED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "operation_instance_id",
        "operation",
        "base_input_ref",
        "prior_manifest_ref",
        "stages",
    }
)
_MANIFEST_FILE_RE = re.compile(r"^manifest_([0-9a-f]{64})\.json$")
_INITIAL_TRANSPORT_RECOVERY_SCHEMA = (
    "arche.investigation.hypothesis_transport_recovery"
)
_INITIAL_TRANSPORT_RECOVERY_STATUS_VALUES = frozenset({"pending", "completed"})
_INITIAL_TRANSPORT_RECOVERY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "attempt",
        "prior_operation_instance_id",
        "retry_operation_instance_id",
        "prior_manifest_ref",
    }
)


def _require_text(value: Any, label: str) -> str:
    """校验 bridge 的身份字符串，并拒绝空白值。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value


def _require_identity(value: Any, label: str) -> str:
    """校验会进入外部 operation 或事件的稳定身份。"""

    text = _require_text(value, label)
    if not _SAFE_IDENTITY_RE.fullmatch(text):
        raise ValueError(f"{label} 含有未注册字符")
    return text


def _copy_json(value: Any, label: str) -> Any:
    """复制严格 JSON 值，拒绝 bridge receipt 中不可重放的对象。"""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 必须是严格 JSON") from exc


def _canonical_digest(value: Any) -> str:
    """返回 JSON 值的稳定摘要，不记录原始内容。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _generation_config_digest(
    config: HypothesisRegenerationGenerationConfig,
) -> str:
    """为 receipt 计算完整生成配置摘要，防止同一 operation 身份漂移。"""

    return _canonical_digest(
        {
            "task_root": config.task_root,
            "stage_tool_projection": list(config.stage_tool_projection),
            "chemical_capability_projection": list(
                config.chemical_capability_projection
            ),
            "num_queries": config.num_queries,
            "num_hypotheses_per_query": config.num_hypotheses_per_query,
            "hypothesis_ids_by_query": [
                list(group) for group in config.hypothesis_ids_by_query
            ],
            "merge_hypothesis_ids": list(config.merge_hypothesis_ids),
            "top_n": config.top_n,
            "supplementary_scientific_input": (
                dict(config.supplementary_scientific_input)
                if config.supplementary_scientific_input is not None
                else None
            ),
            "case_profile": config.case_profile,
            "task_config_schema": config.task_config_schema,
            "creates_investigation_sessions": config.creates_investigation_sessions,
        }
    )


def _compose_canonical_model_request(
    request: HypothesisRegenerationGenerationRequest,
) -> dict[str, Any]:
    """重建现有 HypothesisAgent 实际使用的完整模型请求。

    receipt 不只保存 Working Memory 摘要；这里同时保存动态 operation 的 base input、
    生成参数和 operation 身份。恢复时必须从当前 request 重新组装并逐字核对，不能
    仅凭一个摘要或把已完成模型调用当作可重新发送的普通任务。
    """

    config = request.generation_config
    scientific_context = request.regeneration_package.get("scientific_context")
    if not isinstance(scientific_context, Mapping):
        raise ValueError("regeneration_package 缺少 scientific_context")
    regeneration_package = copy.deepcopy(dict(request.regeneration_package))
    regeneration_package["investigation_working_memory"] = copy.deepcopy(
        dict(request.working_memory_payload)
    )
    regeneration_package = HypothesisAgent._validate_regeneration_package(
        regeneration_package,
        memory_enabled=True,
        expected_task_id=request.task_id,
        expected_batch_id=request.previous_batch_id,
    )
    normalized_projection = (
        HypothesisAgent._validate_stage_tool_projection(
            copy.deepcopy(list(config.stage_tool_projection))
        )
        if config.stage_tool_projection
        else []
    )
    normalized_capability_projection = _copy_json(
        list(config.chemical_capability_projection),
        "chemical_capability_projection",
    )
    base_input = copy.deepcopy(regeneration_package)
    base_input.pop("source_record_ids", None)
    base_input["stage_tool_projection"] = normalized_projection
    base_input["chemical_capability_projection"] = copy.deepcopy(
        normalized_capability_projection
    )
    model_arguments = {
        "task_root": config.task_root,
        "operation_instance_id": request.operation_instance_id,
        "scientific_question": scientific_context["scientific_question"],
        "stage_tool_projection": normalized_projection,
        "chemical_capability_projection": copy.deepcopy(
            normalized_capability_projection
        ),
        "initial_query_angles": copy.deepcopy(
            list(scientific_context["initial_query_angles"])
        ),
        "initial_retrieval_phase": copy.deepcopy(
            dict(scientific_context["initial_retrieval_phase"])
        ),
        "chemistry_context": copy.deepcopy(dict(scientific_context["chemistry_context"])),
        "num_queries": config.num_queries,
        "num_hypotheses_per_query": config.num_hypotheses_per_query,
        "top_n": config.top_n,
        "hypothesis_ids_by_query": [
            list(group) for group in config.hypothesis_ids_by_query
        ],
        "merge_hypothesis_ids": list(config.merge_hypothesis_ids),
        "regeneration_package": regeneration_package,
        "memory_enabled": True,
        "expected_task_id": request.task_id,
        "expected_batch_id": request.previous_batch_id,
        "case_profile": config.case_profile,
        "supplementary_scientific_input": (
            copy.deepcopy(dict(config.supplementary_scientific_input))
            if config.supplementary_scientific_input is not None
            else None
        ),
    }
    return _copy_json(
        {
            "operation": "hypothesis_generation",
            "task_id": request.task_id,
            "operation_key": request.operation_key,
            "operation_instance_id": request.operation_instance_id,
            "task_root": config.task_root,
            "previous_batch_id": request.previous_batch_id,
            "next_batch_id": request.next_batch_id,
            "next_batch_index": request.next_batch_index,
            "memory_enabled": True,
            "base_input": base_input,
            "model_arguments": model_arguments,
        },
        "canonical_model_request",
    )


def _compose_canonical_initial_model_request(
    request: HypothesisInitialGenerationRequest,
) -> dict[str, Any]:
    """重建首批 HypothesisAgent operation 的完整 canonical request。"""

    config = request.generation_config
    normalized_projection = (
        HypothesisAgent._validate_stage_tool_projection(
            copy.deepcopy(list(config.stage_tool_projection))
        )
        if config.stage_tool_projection
        else []
    )
    normalized_capability_projection = _copy_json(
        list(config.chemical_capability_projection),
        "chemical_capability_projection",
    )
    model_arguments = {
        "task_root": config.task_root,
        "operation_instance_id": request.operation_instance_id,
        "scientific_question": request.scientific_question,
        "stage_tool_projection": normalized_projection,
        "chemical_capability_projection": copy.deepcopy(
            normalized_capability_projection
        ),
        "initial_query_angles": None,
        "initial_retrieval_phase": copy.deepcopy(
            dict(request.initial_retrieval_phase)
        ),
        "chemistry_context": copy.deepcopy(dict(request.chemistry_context)),
        "num_queries": config.num_queries,
        "num_hypotheses_per_query": config.num_hypotheses_per_query,
        "top_n": config.top_n,
        "hypothesis_ids_by_query": [
            list(group) for group in config.hypothesis_ids_by_query
        ],
        "merge_hypothesis_ids": list(config.merge_hypothesis_ids),
        "regeneration_package": None,
        "memory_enabled": None,
        "expected_task_id": None,
        "expected_batch_id": None,
        "case_profile": config.case_profile,
        "supplementary_scientific_input": (
            copy.deepcopy(dict(config.supplementary_scientific_input))
            if config.supplementary_scientific_input is not None
            else None
        ),
    }
    if config.supplementary_scientific_input is not None:
        # 与 HypothesisAgent.run_hypothesis_operation 首批路径保持同一计算单元投影，
        # 不把 supplementary 的未注册包装字段误写进 finalized base_input。
        from chemistry_multiagent.agents.hypothesis_agent import (
            _project_calculation_unit_context,
        )

        projected_context = _project_calculation_unit_context(
            config.supplementary_scientific_input
        )
        base_context = (
            dict(projected_context)
            if isinstance(projected_context, Mapping) and projected_context
            else dict(request.chemistry_context)
        )
    else:
        base_context = dict(request.chemistry_context)
    generation_config = {
        "num_queries": config.num_queries,
        "num_hypotheses_per_query": config.num_hypotheses_per_query,
    }
    if config.task_config_schema == "legacy_workflow_v1":
        # 只有旧 workflow 恢复仍使用旧清单字段；活动 Investigation 清单不能携带它。
        generation_config["top_n_workflows"] = config.top_n
    else:
        generation_config["top_n_hypotheses"] = config.top_n
    base_input = {
        "scientific_question": request.scientific_question,
        "initial_chemistry_context": copy.deepcopy(dict(base_context)),
        "generation_config": generation_config,
        "stage_tool_projection": normalized_projection,
        "chemical_capability_projection": copy.deepcopy(
            normalized_capability_projection
        ),
        "initial_retrieval_phase": copy.deepcopy(
            dict(request.initial_retrieval_phase)
        ),
    }
    return _copy_json(
        {
            "operation": "hypothesis_generation",
            "task_id": request.task_id,
            "operation_key": request.operation_key,
            "operation_instance_id": request.operation_instance_id,
            "task_root": config.task_root,
            "batch_id": request.batch_id,
            "memory_enabled": None,
            "base_input": base_input,
            "model_arguments": model_arguments,
        },
        "canonical_initial_model_request",
    )


def _compose_operation_reference(
    request: HypothesisRegenerationGenerationRequest,
    *,
    identities: Sequence[HypothesisRegenerationSessionIdentity],
    source_record_ids: Sequence[str],
    causal_source_ref: str | None,
    model_request: Mapping[str, Any],
) -> dict[str, Any]:
    """建立 receipt 中完整的模型 operation 引用。

    该引用同时固定业务 operation、模型输入摘要、生成配置、批次身份和下一批
    会话身份。恢复时即使外层参数仍然看似相同，只要其中任一部分发生漂移也必须
    停止；它不会保存 provider secret 或原始凭据。
    """

    reference = {
        "operation": "hypothesis_generation",
        "task_id": request.task_id,
        "operation_key": request.operation_key,
        "operation_instance_id": request.operation_instance_id,
        "task_root": request.generation_config.task_root,
        "previous_batch_id": request.previous_batch_id,
        "next_batch_id": request.next_batch_id,
        "next_batch_index": request.next_batch_index,
        "memory_enabled": True,
        "generation_config_digest": _generation_config_digest(
            request.generation_config
        ),
        "model_request_digest": _canonical_digest(model_request),
        "session_identities": _session_identities_payload(identities),
        "source_record_ids": list(source_record_ids),
        "causal_source_ref": causal_source_ref,
    }
    return _copy_json(reference, "operation_reference")


def _safe_failure_text(error: BaseException) -> str:
    """把外部异常压缩为可写入 receipt 的非敏感首行。"""

    text = next((line.strip() for line in str(error).splitlines() if line.strip()), "外部依赖失败")
    text = _ABSOLUTE_PATH_RE.sub(" <path>", text)
    text = _SECRET_RE.sub("<secret>=", text)
    return text[:400]


def _is_busy_operation_error(error: BaseException) -> bool:
    """识别同一模型 operation 仍由 owner 执行的暂时冲突。"""

    text = str(error).casefold()
    return any(marker in text for marker in _BUSY_OPERATION_MARKERS)


def _initial_transport_retry_operation_id(
    operation_instance_id: str,
    attempt: int,
) -> str:
    """按首批原始 operation 和重试次数生成稳定的恢复 operation 身份。"""

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("transport recovery attempt 必须是正整数")
    _require_identity(operation_instance_id, "operation_instance_id")
    return f"{operation_instance_id}-transport-retry-{attempt}"


def _manifest_ref_from_path(
    task_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """从已定位的 finalized manifest 文件重建不含正文的 artifact 引用。"""

    root = Path(task_root).expanduser().resolve(strict=False)
    candidate = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        relative_path = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Hypothesis manifest 不在 task root 内") from exc
    payload = candidate.read_bytes()
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "media_type": "application/json",
        "size": len(payload),
    }


def _classify_initial_finalized_manifest(
    located: Mapping[str, Any],
    *,
    task_root: str | Path,
) -> Literal["validated", "transport"]:
    """确认 finalized manifest 的阶段顺序，并只把最终传输失败归为可重试。"""

    manifest = located.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Hypothesis finalized manifest 缺少正文")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Hypothesis finalized manifest 没有可恢复 stage")
    failed_seen = False
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise ValueError("Hypothesis finalized manifest stage 非法")
        status = stage.get("status")
        attempts = stage.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("Hypothesis finalized manifest stage attempts 非法")
        if status == "validated":
            if failed_seen or not isinstance(attempts[-1], Mapping):
                raise ValueError("失败 stage 后不能出现 validated stage")
            if attempts[-1].get("validation_status") != "valid":
                raise ValueError("validated stage 未以 valid attempt 结束")
            continue
        if status != "failed" or failed_seen or index != len(stages) - 1:
            raise ValueError("Hypothesis finalized manifest stage 状态顺序非法")
        failed_seen = True
        final_attempt = attempts[-1]
        if not isinstance(final_attempt, Mapping):
            raise ValueError("失败 stage 的最终 attempt 非法")
        validation_status = final_attempt.get("validation_status")
        if validation_status not in {"invalid", "transport_failed"}:
            raise ValueError("失败 stage 的最终 attempt 状态非法")
        if validation_status != "transport_failed":
            raise ValueError("Hypothesis 合同或解析失败不能自动重试")
        error_ref = final_attempt.get("error_ref")
        if not isinstance(error_ref, Mapping):
            raise ValueError("传输失败 stage 缺少 error artifact")
        _path, payload = _validate_artifact_ref_file(
            error_ref,
            task_root=task_root,
            label="Hypothesis transport error artifact",
        )
        try:
            error = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Hypothesis transport error artifact 不是 JSON") from exc
        if not isinstance(error, Mapping) or error.get("failure_kind") != "transport":
            raise ValueError("Hypothesis finalized manifest 的失败类型不是 transport")
    return "transport" if failed_seen else "validated"


def _locate_initial_finalized_manifest(
    manifest_ref: Mapping[str, Any],
    *,
    task_root: str | Path,
    operation_instance_id: str,
    expected_base_input: Mapping[str, Any],
) -> tuple[dict[str, Any], Literal["validated", "transport"]]:
    """按 manifest 引用定位首批 operation，并验证其 artifact 和失败类型。"""

    _validate_hypothesis_manifest_ref(
        manifest_ref,
        task_root=task_root,
        operation_instance_id=operation_instance_id,
    )
    located = DynamicModelOperationRunner.locate_finalized_manifest_in_operation(
        task_root=task_root,
        operation="hypothesis_generation",
        operation_instance_id=operation_instance_id,
        expected_base_input=expected_base_input,
    )
    if located is None:
        raise ValueError("Hypothesis transport failure 缺少 finalized manifest")
    computed_ref = _manifest_ref_from_path(task_root, located["manifest_path"])
    if dict(computed_ref) != dict(manifest_ref):
        raise ValueError("Hypothesis manifest_ref 与 operation finalized manifest 不一致")
    status = _classify_initial_finalized_manifest(located, task_root=task_root)
    return located, status


def _extract_initial_transport_manifest_ref(
    error: BaseException,
    *,
    task_root: str | Path,
    operation_instance_id: str,
    expected_base_input: Mapping[str, Any],
) -> dict[str, Any] | None:
    """从协议异常提取并验证可恢复的传输失败 manifest；其它错误返回空值。"""

    if not isinstance(error, HypothesisProtocolError) or error.failure_kind != "transport":
        return None
    operation_result = getattr(error, "operation_result", None)
    manifest_ref = getattr(error, "manifest_ref", None)
    if manifest_ref is None and isinstance(operation_result, Mapping):
        manifest_ref = operation_result.get("manifest_ref")
    if not isinstance(manifest_ref, Mapping):
        raise ValueError("transport HypothesisProtocolError 缺少 finalized manifest_ref")
    _located, status = _locate_initial_finalized_manifest(
        manifest_ref,
        task_root=task_root,
        operation_instance_id=operation_instance_id,
        expected_base_input=expected_base_input,
    )
    if status != "transport":
        raise ValueError("Hypothesis transport recovery 引用了非失败 manifest")
    return copy.deepcopy(dict(manifest_ref))


def _build_initial_transport_protocol_error(
    located: Mapping[str, Any],
    *,
    task_root: str | Path,
) -> HypothesisProtocolError:
    """把已存在的失败 retry manifest 转回带 receipt 证据的传输协议异常。"""

    manifest = located.get("manifest")
    stages = manifest.get("stages") if isinstance(manifest, Mapping) else None
    stage_name = "hypothesis"
    if isinstance(stages, list) and stages and isinstance(stages[-1], Mapping):
        stage_name = str(stages[-1].get("stage") or stage_name)
    error = HypothesisProtocolError(stage_name, "transport")
    manifest_ref = _manifest_ref_from_path(task_root, located["manifest_path"])
    operation_result = {
        "status": "failed",
        "failure_kind": "transport",
        "manifest": copy.deepcopy(manifest),
        "manifest_path": str(located["manifest_path"]),
        "manifest_ref": copy.deepcopy(manifest_ref),
    }
    error.operation_result = operation_result
    error.manifest_ref = copy.deepcopy(manifest_ref)
    error.manifest_path = str(located["manifest_path"])
    return error


def _read_initial_transport_recovery_plan(
    receipt: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """读取首批 receipt 中可选的传输恢复计划；旧 receipt 返回空值。"""

    reference = receipt.get("operation_reference")
    if not isinstance(reference, Mapping):
        return None
    plan = reference.get("transport_recovery")
    if plan is None:
        return None
    if not isinstance(plan, Mapping):
        raise ValueError("首批 receipt.transport_recovery 必须是 object")
    return plan


def _validate_initial_transport_recovery_plan(
    plan: Mapping[str, Any],
    *,
    request: "HypothesisInitialGenerationRequest",
) -> None:
    """核对首批 transport retry 链，确保每次恢复都指向前一份失败清单。"""

    if set(plan) != _INITIAL_TRANSPORT_RECOVERY_FIELDS:
        raise ValueError("首批 transport recovery 计划字段非法")
    if plan.get("schema") != _INITIAL_TRANSPORT_RECOVERY_SCHEMA:
        raise ValueError("首批 transport recovery schema 未注册")
    if plan.get("schema_version") != 1:
        raise ValueError("首批 transport recovery schema_version 未注册")
    status = plan.get("status")
    if status not in _INITIAL_TRANSPORT_RECOVERY_STATUS_VALUES:
        raise ValueError("首批 transport recovery status 未注册")
    attempt = plan.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("首批 transport recovery attempt 非法")
    prior_operation = _require_identity(
        plan.get("prior_operation_instance_id"),
        "transport recovery.prior_operation_instance_id",
    )
    retry_operation = _require_identity(
        plan.get("retry_operation_instance_id"),
        "transport recovery.retry_operation_instance_id",
    )
    expected_prior = (
        request.operation_instance_id
        if attempt == 1
        else _initial_transport_retry_operation_id(
            request.operation_instance_id,
            attempt - 1,
        )
    )
    if prior_operation != expected_prior:
        raise ValueError("首批 transport recovery prior operation 身份不连续")
    expected_retry = _initial_transport_retry_operation_id(
        request.operation_instance_id,
        attempt,
    )
    if retry_operation != expected_retry:
        raise ValueError("首批 transport recovery retry operation 身份不稳定")
    manifest_ref = plan.get("prior_manifest_ref")
    if not isinstance(manifest_ref, Mapping):
        raise ValueError("首批 transport recovery 缺少 prior_manifest_ref")
    canonical = _compose_canonical_initial_model_request(request)
    _located, status = _locate_initial_finalized_manifest(
        manifest_ref,
        task_root=request.generation_config.task_root,
        operation_instance_id=prior_operation,
        expected_base_input=canonical["base_input"],
    )
    if status != "transport":
        raise ValueError(
            "首批 transport recovery prior manifest 必须是 transport failure"
        )


def _receipt_expected_fields(status: str) -> set[str]:
    """返回某个 receipt 阶段允许的字段集合。"""

    allowed = set(_RECEIPT_BASE_FIELDS)
    if status in {"generated", "wm_written", "completed"}:
        allowed.update(_RECEIPT_OUTPUT_FIELDS)
    if status == "failed":
        allowed.add("failure")
    return allowed


def _validate_receipt_identity_fields(receipt: Mapping[str, Any]) -> None:
    """校验 receipt 的 operation 身份、摘要和五个 session claim。"""

    for field in (
        "task_id",
        "previous_batch_id",
        "next_batch_id",
        "operation_key",
        "operation_instance_id",
    ):
        _require_identity(receipt.get(field), f"receipt.{field}")
    next_batch_index = receipt.get("next_batch_index")
    if isinstance(next_batch_index, bool) or not isinstance(next_batch_index, int):
        raise ValueError("receipt.next_batch_index 必须是整数")
    for field in (
        "input_digest",
        "model_request_digest",
        "operation_reference_digest",
        "generation_config_digest",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or not _RECEIPT_DIGEST_RE.fullmatch(value):
            raise ValueError(f"receipt.{field} 必须是 SHA-256 摘要")
    if not isinstance(receipt.get("operation_reference"), Mapping):
        raise ValueError("receipt.operation_reference 必须是 object")
    identities = receipt.get("session_identities")
    if not isinstance(identities, list) or len(identities) != 5:
        raise ValueError("receipt.session_identities 必须包含五项")
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping) or set(identity) != {
            "combination_index",
            "combination_id",
            "workspace_ref",
            "scope_id",
        }:
            raise ValueError(f"receipt.session_identities[{index}] 字段非法")
        if (
            isinstance(identity["combination_index"], bool)
            or not isinstance(identity["combination_index"], int)
            or identity["combination_index"] not in {1, 2, 3, 4, 5}
        ):
            raise ValueError(f"receipt.session_identities[{index}].combination_index 非法")
        for field in ("combination_id", "workspace_ref", "scope_id"):
            _require_identity(
                identity[field],
                f"receipt.session_identities[{index}].{field}",
            )


def _validate_receipt_cleanup_fields(receipt: Mapping[str, Any]) -> None:
    """校验 receipt 中的旧 scope 清理计划。"""

    source_record_ids = receipt.get("source_record_ids")
    if not isinstance(source_record_ids, list) or any(
        not isinstance(item, str) or not item for item in source_record_ids
    ):
        raise ValueError("receipt.source_record_ids 必须是字符串数组")
    try:
        _normalize_source_records(source_record_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "receipt.source_record_ids 必须是 opaque identity 数组"
        ) from exc
    causal_source_ref = receipt.get("causal_source_ref")
    if causal_source_ref is not None and not isinstance(causal_source_ref, str):
        raise ValueError("receipt.causal_source_ref 必须是字符串或 null")
    if not isinstance(receipt.get("cleanup"), Mapping):
        raise ValueError("receipt.cleanup 必须是 object")


def _validate_receipt_output_fields(receipt: Mapping[str, Any], status: str) -> None:
    """校验 finalized manifest 和 Working Memory 提交游标。"""

    if status in {"generated", "wm_written", "completed"}:
        output_digest = receipt.get("output_digest")
        if not isinstance(output_digest, str) or not _RECEIPT_DIGEST_RE.fullmatch(
            output_digest
        ):
            raise ValueError("receipt.output_digest 必须是 SHA-256 摘要")
        source_refs = receipt.get("source_refs")
        if not isinstance(source_refs, list) or any(
            not isinstance(item, str) or not item for item in source_refs
        ):
            raise ValueError("receipt.source_refs 必须是字符串数组")
        if not isinstance(receipt.get("batch_event_id"), (str, type(None))):
            raise ValueError("receipt.batch_event_id 必须是字符串或 null")
        if status in {"wm_written", "completed"}:
            _require_text(receipt.get("batch_event_id"), "receipt.batch_event_id")
    if status == "failed" and not isinstance(receipt.get("failure"), str):
        raise ValueError("failed receipt.failure 必须是字符串")


def _validate_regeneration_receipt_schema(receipt: Mapping[str, Any]) -> str:
    """严格校验 bridge receipt 的字段、类型和单向阶段形状。"""

    if not isinstance(receipt, Mapping):
        raise ValueError("重生 receipt 必须是 object")
    status = receipt.get("status")
    if receipt.get("schema") != _RECEIPT_SCHEMA:
        raise ValueError("重生 receipt schema 未注册")
    if (
        isinstance(receipt.get("schema_version"), bool)
        or not isinstance(receipt.get("schema_version"), int)
        or receipt.get("schema_version") != 1
    ):
        raise ValueError("重生 receipt schema_version 不受支持")
    if status not in _RECEIPT_STATUS_VALUES:
        raise ValueError("重生 receipt.status 未注册")
    if set(receipt) != _receipt_expected_fields(status):
        raise ValueError("重生 receipt 字段不符合当前阶段 schema")
    _validate_receipt_identity_fields(receipt)
    _validate_receipt_cleanup_fields(receipt)
    _validate_receipt_output_fields(receipt, status)
    return status


def _regeneration_output_digest(
    payload: Mapping[str, Any],
    source_refs: Sequence[str],
) -> str:
    """计算内存中的 BatchCreated payload 与 manifest 来源的联合摘要。

    该函数只用于比较刚从 finalized manifest 重放出的临时 payload；receipt 只保存
    联合摘要和 manifest 来源引用，不保存 BatchCreated 正文，避免形成第二份科学事实。
    """

    return _canonical_digest(
        {
            "batch_payload": _copy_json(dict(payload), "batch_payload"),
            "source_refs": list(source_refs),
        }
    )


def _manifest_source_digest(source_refs: Sequence[str]) -> str:
    """计算 receipt 中仅保存的 finalized manifest 引用摘要。"""

    return _canonical_digest({"source_refs": list(source_refs)})


def _canonical_json_bytes(value: Any, label: str) -> bytes:
    """把清单值编码为唯一 JSON 字节，拒绝 NaN、非 JSON 对象和隐式格式差异。"""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是严格 canonical JSON") from exc


def _read_artifact_ref_path(
    reference: Mapping[str, Any],
    *,
    label: str,
    expected_relative_parts: tuple[str, ...] | None,
) -> tuple[str, list[str]]:
    """读取并核对 artifact 的相对路径，返回路径文本和各段。"""

    relative_path = reference.get("relative_path")
    _validate_artifact_ref_path_text(relative_path, label=label)
    parts = relative_path.split("/")
    _validate_artifact_ref_path_parts(
        relative_path,
        parts,
        label=label,
        expected_relative_parts=expected_relative_parts,
    )
    return relative_path, parts


def _validate_artifact_ref_path_text(value: Any, *, label: str) -> None:
    """核对 artifact 路径是非空、非 URI 的相对路径文本。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.relative_path 非法")
    if "\\" in value or _URI_SCHEME_RE.match(value) is not None:
        raise ValueError(f"{label}.relative_path 不能是 URI、绝对路径或反斜杠路径")
    if value.startswith("/"):
        raise ValueError(f"{label}.relative_path 不能是 URI、绝对路径或反斜杠路径")


def _validate_artifact_ref_path_parts(
    relative_path: str,
    parts: Sequence[str],
    *,
    label: str,
    expected_relative_parts: tuple[str, ...] | None,
) -> None:
    """核对 artifact 路径各段和调用方绑定的 operation 目录。"""

    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label}.relative_path 含空段或点段")
    parsed = PurePosixPath(relative_path)
    if parsed.is_absolute() or parsed.as_posix() != relative_path:
        raise ValueError(f"{label}.relative_path 不是规范相对路径")
    if expected_relative_parts is not None and tuple(parts) != expected_relative_parts:
        raise ValueError(f"{label}.relative_path 未绑定当前 Hypothesis operation")


def _read_artifact_ref_metadata(
    reference: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, int]:
    """读取并核对 artifact 的内容摘要、媒体类型和大小。"""

    digest = reference.get("sha256")
    if not isinstance(digest, str) or not _RECEIPT_DIGEST_RE.fullmatch(digest):
        raise ValueError(f"{label}.sha256 必须是小写 SHA-256")
    if reference.get("media_type") != "application/json":
        raise ValueError(f"{label}.media_type 必须是 application/json")
    size = reference.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label}.size 必须是非负整数")
    return digest, size


def _resolve_artifact_ref_candidate(
    parts: Sequence[str],
    *,
    task_root: str | Path,
    label: str,
) -> Path:
    """把安全相对路径解析为 task root 内的普通文件。"""

    root = Path(task_root).expanduser().resolve(strict=False)
    candidate = root.joinpath(*parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - parts 已阻断常规穿越
        raise ValueError(f"{label}.relative_path 越过 task root") from exc
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label}.relative_path 不能经过 symlink")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label}.relative_path 不是普通文件")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError(f"{label}.relative_path 解析后不是 task root 内的普通文件")
    return candidate


def _read_and_verify_artifact_payload(
    candidate: Path,
    *,
    digest: str,
    size: int,
    label: str,
) -> bytes:
    """读取 artifact 并核对声明的字节数与 SHA-256。"""

    payload = candidate.read_bytes()
    if len(payload) != size:
        raise ValueError(f"{label}.size 与文件内容不一致")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{label}.sha256 与文件内容不一致")
    return payload


def _validate_artifact_ref_file(
    reference: Mapping[str, Any],
    *,
    task_root: str | Path,
    label: str,
    expected_relative_parts: tuple[str, ...] | None = None,
) -> tuple[Path, bytes]:
    """核对 artifact 引用、真实文件和内容摘要，并阻断路径穿越或软链接。"""

    if set(reference) != _ARTIFACT_REF_KEYS:
        raise ValueError(f"{label} 必须恰含 relative_path/sha256/media_type/size")
    _relative_path, parts = _read_artifact_ref_path(
        reference,
        label=label,
        expected_relative_parts=expected_relative_parts,
    )
    digest, size = _read_artifact_ref_metadata(reference, label=label)
    candidate = _resolve_artifact_ref_candidate(parts, task_root=task_root, label=label)
    payload = _read_and_verify_artifact_payload(
        candidate,
        digest=digest,
        size=size,
        label=label,
    )
    return candidate, payload


def _validate_hypothesis_manifest_location(
    manifest_ref: Mapping[str, Any],
    *,
    operation_instance_id: str,
) -> tuple[str, ...]:
    """核对 finalized manifest 的 operation 目录和 content-addressed 文件名。"""

    relative_path = manifest_ref.get("relative_path")
    if not isinstance(relative_path, str):
        raise ValueError("Hypothesis manifest_ref.relative_path 非法")
    parts = relative_path.split("/")
    expected_prefix = (
        "working_memory",
        "hypothesis_generation_inputs",
        operation_instance_id,
    )
    if len(parts) != 4 or tuple(parts[:3]) != expected_prefix:
        raise ValueError("Hypothesis manifest_ref 未绑定当前 operation 目录")
    match = _MANIFEST_FILE_RE.fullmatch(parts[-1])
    if match is None:
        raise ValueError("Hypothesis manifest_ref 文件名不含内容 hash")
    if match.group(1) != manifest_ref.get("sha256"):
        raise ValueError("Hypothesis manifest_ref 文件名与 sha256 不一致")
    return tuple(parts)


def _read_finalized_manifest_payload(
    manifest_ref: Mapping[str, Any],
    *,
    task_root: str | Path,
    relative_parts: tuple[str, ...],
) -> Mapping[str, Any]:
    """读取 manifest 文件并确认其是 canonical JSON object。"""

    _path, payload = _validate_artifact_ref_file(
        manifest_ref,
        task_root=task_root,
        label="Hypothesis manifest_ref",
        expected_relative_parts=relative_parts,
    )
    try:
        manifest = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Hypothesis finalized manifest 不是 JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Hypothesis finalized manifest 必须是 object")
    if _canonical_json_bytes(manifest, "Hypothesis finalized manifest") != payload:
        raise ValueError("Hypothesis finalized manifest 不是 canonical JSON")
    return manifest


def _validate_finalized_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    operation_instance_id: str,
) -> None:
    """核对 finalized manifest 的 schema、operation 和 stages 字段。"""

    if set(manifest) != _FINALIZED_MANIFEST_KEYS:
        raise ValueError("Hypothesis finalized manifest 字段不完整或含未知字段")
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("Hypothesis finalized manifest schema_version 未注册")
    if manifest.get("operation") != "hypothesis_generation":
        raise ValueError("Hypothesis finalized manifest operation 身份不一致")
    if manifest.get("operation_instance_id") != operation_instance_id:
        raise ValueError("Hypothesis finalized manifest operation_instance_id 不一致")
    if not isinstance(manifest.get("stages"), list):
        raise ValueError("Hypothesis finalized manifest.stages 必须是数组")


def _validate_finalized_manifest_input_refs(
    manifest: Mapping[str, Any],
    *,
    task_root: str | Path,
) -> None:
    """核对 finalized manifest 可选输入 artifact 的完整引用。"""

    for field in ("base_input_ref", "prior_manifest_ref"):
        reference = manifest.get(field)
        if reference is None:
            continue
        if not isinstance(reference, Mapping):
            raise ValueError(f"Hypothesis finalized manifest.{field} 非法")
        _validate_artifact_ref_file(
            reference,
            task_root=task_root,
            label=f"Hypothesis finalized manifest.{field}",
        )


def _validate_hypothesis_manifest_ref(
    manifest_ref: Mapping[str, Any],
    *,
    task_root: str | Path,
    operation_instance_id: str,
) -> None:
    """核对 Hypothesis finalized manifest 的位置、身份、canonical 内容和摘要。"""

    relative_parts = _validate_hypothesis_manifest_location(
        manifest_ref,
        operation_instance_id=operation_instance_id,
    )
    manifest = _read_finalized_manifest_payload(
        manifest_ref,
        task_root=task_root,
        relative_parts=relative_parts,
    )
    _validate_finalized_manifest_identity(
        manifest,
        operation_instance_id=operation_instance_id,
    )
    _validate_finalized_manifest_input_refs(manifest, task_root=task_root)


def _validate_hypothesis_manifest_source_ref(
    source_ref: Any,
    *,
    task_root: str | Path,
    operation_instance_id: str,
) -> None:
    """核对 receipt 中的字符串来源仍指向当前 operation 的 finalized manifest。"""

    if not isinstance(source_ref, str) or not source_ref:
        raise ValueError("receipt.source_refs 必须包含 manifest 相对路径")
    parts = source_ref.split("/")
    if len(parts) != 4 or tuple(parts[:3]) != (
        "working_memory",
        "hypothesis_generation_inputs",
        operation_instance_id,
    ):
        raise ValueError("receipt.source_refs 未绑定当前 Hypothesis operation")
    match = _MANIFEST_FILE_RE.fullmatch(parts[-1])
    if match is None:
        raise ValueError("receipt.source_refs 必须引用 content-addressed manifest")
    root = Path(task_root).expanduser().resolve(strict=False)
    candidate = root.joinpath(*parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("receipt.source_refs manifest 不是普通文件")
    payload = candidate.read_bytes()
    reference = {
        "relative_path": source_ref,
        "sha256": match.group(1),
        "media_type": "application/json",
        "size": len(payload),
    }
    _validate_hypothesis_manifest_ref(
        reference,
        task_root=task_root,
        operation_instance_id=operation_instance_id,
    )


@dataclass(frozen=True)
class HypothesisRegenerationSessionIdentity:
    """调用方为下一批五个会话显式提供的组合、工作区和 scope 身份。

    bridge 不从旧会话、组合顺序或本机目录猜测 ``workspace_ref`` 与 ``scope_id``。
    ``combination_index`` 也由调用方明确提供，最终会和 BatchCreated candidate
    的位置逐值核对。
    """

    combination_index: int
    combination_id: str
    workspace_ref: str
    scope_id: str

    def __post_init__(self) -> None:
        """在构造身份记录时拒绝重复使用或空身份。"""

        if isinstance(self.combination_index, bool) or not isinstance(self.combination_index, int):
            raise TypeError("combination_index 必须是整数")
        if self.combination_index < 1:
            raise ValueError("combination_index 必须是正整数")
        _require_identity(self.combination_id, "combination_id")
        _require_identity(self.workspace_ref, "workspace_ref")
        _require_identity(self.scope_id, "scope_id")


@dataclass(frozen=True)
class HypothesisRegenerationGenerationConfig:
    """重生模型调用需要的冻结配置和正式假设身份分配。

    配置来自调用方已经核对的 TaskCreated/运行目录；bridge 只要求本案例的
    ``top_n=5``，不会猜测 Query 数量、模型工具目录或 hypothesis 身份。
    """

    task_root: str
    stage_tool_projection: tuple[Mapping[str, Any], ...]
    num_queries: int
    num_hypotheses_per_query: int
    hypothesis_ids_by_query: tuple[tuple[str, ...], ...]
    merge_hypothesis_ids: tuple[str, ...]
    top_n: int = 5
    supplementary_scientific_input: Mapping[str, Any] | None = None
    case_profile: str | None = None
    task_config_schema: Literal["investigation_v2", "legacy_workflow_v1"] = (
        "investigation_v2"
    )
    creates_investigation_sessions: bool = True
    chemical_capability_projection: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """校验生成预算、身份形状和严格 JSON 输入。"""

        _require_text(self.task_root, "task_root")
        for label, value in (
            ("num_queries", self.num_queries),
            ("num_hypotheses_per_query", self.num_hypotheses_per_query),
            ("top_n", self.top_n),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} 必须是正整数")
        if self.top_n != 5:
            raise ValueError("Investigation 假设重生必须严格选择五个组合")
        if self.task_config_schema not in {"investigation_v2", "legacy_workflow_v1"}:
            raise ValueError(
                "task_config_schema 必须是 investigation_v2 或 legacy_workflow_v1"
            )
        if not isinstance(self.creates_investigation_sessions, bool):
            raise ValueError("creates_investigation_sessions 必须是布尔值")
        if len(self.hypothesis_ids_by_query) != self.num_queries:
            raise ValueError("hypothesis_ids_by_query 数量与 num_queries 不一致")
        all_ids: list[str] = []
        for group in self.hypothesis_ids_by_query:
            if len(group) != self.num_hypotheses_per_query:
                raise ValueError("hypothesis_ids_by_query 的每组数量不一致")
            all_ids.extend(_require_identity(item, "hypothesis_id") for item in group)
        all_ids.extend(_require_identity(item, "merge_hypothesis_id") for item in self.merge_hypothesis_ids)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("同一重生 operation 的 hypothesis 身份不能重复")
        _copy_json(list(self.stage_tool_projection), "stage_tool_projection")
        _copy_json(
            list(self.chemical_capability_projection),
            "chemical_capability_projection",
        )
        if self.supplementary_scientific_input is not None:
            _copy_json(self.supplementary_scientific_input, "supplementary_scientific_input")


@dataclass(frozen=True)
class HypothesisRegenerationGenerationRequest:
    """传给现有 Hypothesis 生成依赖的完整重生请求。

    ``working_memory_payload`` 是新 Investigation 适配器恢复的全部历史；
    ``regeneration_package`` 是旧 HypothesisAgent 仍要求的兼容结构。两者都由
    bridge 生成，调用方不能以 marker 或 narrative 自行拼装历史。
    """

    task_id: str
    previous_batch_id: str
    next_batch_id: str
    next_batch_index: int
    operation_key: str
    operation_instance_id: str
    working_memory_input: HypothesisRegenerationWorkingMemoryInput
    working_memory_payload: Mapping[str, Any]
    regeneration_package: Mapping[str, Any]
    generation_config: HypothesisRegenerationGenerationConfig


@dataclass(frozen=True)
class HypothesisInitialGenerationRequest:
    """首批 Hypothesis operation 的完整、可重建请求。

    首批没有上一批 Investigation 历史，但仍必须把任务、冻结 Retrieval、生成预算
    和稳定 operation 身份放进 receipt 的 canonical request。恢复时由同一请求重建
    finalized manifest，不把首批模型结果另存为第二份科学事实。
    """

    task_id: str
    batch_id: str
    operation_key: str
    operation_instance_id: str
    scientific_question: str
    initial_retrieval_phase: Mapping[str, Any]
    chemistry_context: Mapping[str, Any]
    generation_config: HypothesisRegenerationGenerationConfig


@runtime_checkable
class HypothesisRegenerationGenerator(Protocol):
    """Hypothesis 重生的最小可注入生成接口。"""

    def generate_hypothesis_regeneration(
        self,
        request: HypothesisRegenerationGenerationRequest,
    ) -> Mapping[str, Any]:
        """在给定冻结输入上生成一份带 operation manifest 的候选结果。"""


@runtime_checkable
class HypothesisInitialGenerationGenerator(Protocol):
    """首批 Hypothesis 生成和 finalized manifest replay 的最小接口。"""

    def generate_initial_hypothesis_batch(
        self,
        request: HypothesisInitialGenerationRequest,
    ) -> Mapping[str, Any]:
        """生成首批候选并返回 validated operation manifest 引用。"""

    def recover_initial_hypothesis_batch(
        self,
        request: HypothesisInitialGenerationRequest,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """从同一首批 operation finalized manifest 恢复，不重新调用模型。"""


@runtime_checkable
class InvestigationRegenerationPersistence(Protocol):
    """重生 bridge 所需的 receipt/历史读取窄接口。

    生产实现可以把 receipt 游标放入独立运行数据库；bridge 不要求或读取一份
    自己维护的完整 Investigation 科学状态。``load_investigation_snapshot`` 是由
    Working Memory reader 重建的只读历史视图；独立 SQLite 只保存 receipt 游标。
    """

    def load_investigation_snapshot(self, task_id: str) -> Any:
        """读取上一批只读事实快照。"""

    def load_hypothesis_regeneration_receipt(
        self,
        task_id: str,
        operation_key: str,
    ) -> Any:
        """读取生成 claim 和提交游标。"""

    def claim_hypothesis_regeneration_receipt(
        self,
        task_id: str,
        operation_key: str,
        receipt: Mapping[str, Any],
    ) -> Any:
        """抢占一次生成 claim，已存在时返回空值。"""

    def save_hypothesis_regeneration_receipt(
        self,
        task_id: str,
        operation_key: str,
        receipt: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> int:
        """按版本保存 receipt 游标。"""


@runtime_checkable
class WorkingMemoryHypothesisBatchWriter(Protocol):
    """Working Memory 原子创建下一批事实的最小接口。

    该接口是重生 bridge 与唯一科学事实源之间的边界。实现必须在一个事务中
    校验上一批五个 ``revise_hypothesis`` marker，写入 BatchCreated，并建立五个
    Investigation session；bridge 不再另写一份 ``InvestigationState``。
    """

    def begin_next_investigation_batch(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> str:
        """原子保存下一批 BatchCreated、候选和五个 session，并返回事件 ID。"""


@runtime_checkable
class InvestigationScopeRevokeProtocol(Protocol):
    """撤销调查 scope capability 的最小依赖接口。

    调用方必须按完整旧会话身份定位 capability。实现成功时返回 ``None``；返回其
    他值或抛出异常都按结果不确定处理，bridge 会保留对应的 ``pending`` 状态。
    """

    def revoke_investigation_scope_capability(
        self,
        capability: Any = None,
        **kwargs: Any,
    ) -> None:
        """撤销一个 scope；失败时不应声称本地已完成。"""


@dataclass(frozen=True)
class HypothesisRegenerationBridgeOutcome:
    """bridge 返回的可恢复状态摘要，不把部分写入或清理误报为完成。"""

    status: Literal["completed", "pending", "failed"]
    task_id: str
    previous_batch_id: str
    next_batch_id: str
    receipt_status: str
    generation_called: bool
    working_memory_written: bool
    investigation_committed: bool
    revision: int | None = None
    batch_event_id: str | None = None
    reason: str | None = None
    state: InvestigationState | None = None
    cleanup_status: Literal["not_required", "cleanup_pending", "completed"] = (
        "not_required"
    )
    cleanup_scope_statuses: tuple[tuple[str, str], ...] = ()


class HypothesisRegenerationBridgeError(RuntimeError):
    """重生 bridge 在已保存 receipt 后停止，调用方可按 outcome 恢复。"""

    def __init__(self, message: str, *, outcome: HypothesisRegenerationBridgeOutcome):
        self.outcome = outcome
        super().__init__(message)


class HypothesisAgentRegenerationGenerator:
    """把现有 HypothesisAgent 的 operation 路径接到重生调用接口。

    该 adapter 只调用 ``run_hypothesis_operation``，因此继续沿用现有模型阶段、
    动态 manifest 和严格 regeneration package 校验；它不写 Working Memory。
    """

    def __init__(self, hypothesis_agent: HypothesisAgent) -> None:
        """绑定一个已经装配模型 provider 的 HypothesisAgent。"""

        if not isinstance(hypothesis_agent, HypothesisAgent):
            raise TypeError("hypothesis_agent 必须是 HypothesisAgent")
        self.hypothesis_agent = hypothesis_agent

    def generate_hypothesis_regeneration(
        self,
        request: HypothesisRegenerationGenerationRequest,
    ) -> Mapping[str, Any]:
        """用新 Working Memory 生成的兼容 package 调用现有 operation 路径。"""

        config = request.generation_config
        scientific_context = request.regeneration_package["scientific_context"]
        if not isinstance(scientific_context, Mapping):
            raise ValueError("regeneration_package 缺少 scientific_context")
        regeneration_package = copy.deepcopy(dict(request.regeneration_package))
        regeneration_package["investigation_working_memory"] = copy.deepcopy(
            dict(request.working_memory_payload)
        )
        result = self.hypothesis_agent.run_hypothesis_operation(
            task_root=config.task_root,
            operation_instance_id=request.operation_instance_id,
            scientific_question=str(scientific_context["scientific_question"]),
            stage_tool_projection=copy.deepcopy(list(config.stage_tool_projection)),
            chemical_capability_projection=copy.deepcopy(
                list(config.chemical_capability_projection)
            ),
            initial_query_angles=copy.deepcopy(
                list(scientific_context["initial_query_angles"])
            ),
            initial_retrieval_phase=copy.deepcopy(
                dict(scientific_context["initial_retrieval_phase"])
            ),
            chemistry_context=copy.deepcopy(dict(scientific_context["chemistry_context"])),
            num_queries=config.num_queries,
            num_hypotheses_per_query=config.num_hypotheses_per_query,
            top_n=config.top_n,
            hypothesis_ids_by_query=copy.deepcopy(
                [list(group) for group in config.hypothesis_ids_by_query]
            ),
            merge_hypothesis_ids=copy.deepcopy(list(config.merge_hypothesis_ids)),
            regeneration_package=regeneration_package,
            memory_enabled=True,
            expected_task_id=request.task_id,
            expected_batch_id=request.previous_batch_id,
            case_profile=config.case_profile,
            task_config_schema=config.task_config_schema,
            supplementary_scientific_input=(
                copy.deepcopy(dict(config.supplementary_scientific_input))
                if isinstance(config.supplementary_scientific_input, Mapping)
                else None
            ),
        )
        if not isinstance(result, Mapping):
            raise ValueError("HypothesisAgent operation 必须返回 object")
        return copy.deepcopy(dict(result))

    def recover_hypothesis_regeneration(
        self,
        request: HypothesisRegenerationGenerationRequest,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """通过 Dynamic finalized manifest 严格恢复，不重新调用模型 provider。"""

        canonical_request = _compose_canonical_model_request(request)
        if receipt.get("model_request_digest") != _canonical_digest(canonical_request):
            raise ValueError("started receipt 的 model request 摘要与当前请求不一致")
        config = request.generation_config
        scientific_context = request.regeneration_package.get("scientific_context")
        if not isinstance(scientific_context, Mapping):
            raise ValueError("regeneration_package 缺少 scientific_context")
        regeneration_package = copy.deepcopy(dict(request.regeneration_package))
        regeneration_package["investigation_working_memory"] = copy.deepcopy(
            dict(request.working_memory_payload)
        )
        result = self.hypothesis_agent.replay_hypothesis_operation_from_finalized_manifest(
            task_root=config.task_root,
            operation_instance_id=request.operation_instance_id,
            base_input=canonical_request["base_input"],
            scientific_question=str(scientific_context["scientific_question"]),
            initial_query_angles=copy.deepcopy(
                list(scientific_context["initial_query_angles"])
            ),
            initial_retrieval_phase=copy.deepcopy(
                dict(scientific_context["initial_retrieval_phase"])
            ),
            chemistry_context=copy.deepcopy(dict(scientific_context["chemistry_context"])),
            num_queries=config.num_queries,
            num_hypotheses_per_query=config.num_hypotheses_per_query,
            top_n=config.top_n,
            hypothesis_ids_by_query=copy.deepcopy(
                [list(group) for group in config.hypothesis_ids_by_query]
            ),
            merge_hypothesis_ids=copy.deepcopy(list(config.merge_hypothesis_ids)),
            regeneration_package=regeneration_package,
            memory_enabled=True,
            expected_task_id=request.task_id,
            expected_batch_id=request.previous_batch_id,
            case_profile=config.case_profile,
            task_config_schema=config.task_config_schema,
            supplementary_scientific_input=(
                copy.deepcopy(dict(config.supplementary_scientific_input))
                if isinstance(config.supplementary_scientific_input, Mapping)
                else None
            ),
        )
        if result is not None and not isinstance(result, Mapping):
            raise ValueError("Hypothesis finalized replay 必须返回 object 或 None")
        return copy.deepcopy(dict(result)) if isinstance(result, Mapping) else None

    def generate_initial_hypothesis_batch(
        self,
        request: HypothesisInitialGenerationRequest,
    ) -> Mapping[str, Any]:
        """通过现有 HypothesisAgent operation 生成首批，并保留 finalized manifest。"""

        return self._run_initial_hypothesis_operation(
            request,
            operation_instance_id=request.operation_instance_id,
        )

    def _run_initial_hypothesis_operation(
        self,
        request: HypothesisInitialGenerationRequest,
        *,
        operation_instance_id: str,
        prior_manifest_path: str | None = None,
    ) -> Mapping[str, Any]:
        """执行首批 operation；传入 prior manifest 时只重试其首个失败阶段。"""

        config = request.generation_config
        result = self.hypothesis_agent.run_hypothesis_operation(
            task_root=config.task_root,
            operation_instance_id=operation_instance_id,
            scientific_question=request.scientific_question,
            stage_tool_projection=copy.deepcopy(list(config.stage_tool_projection)),
            chemical_capability_projection=copy.deepcopy(
                list(config.chemical_capability_projection)
            ),
            initial_query_angles=None,
            initial_retrieval_phase=copy.deepcopy(
                dict(request.initial_retrieval_phase)
            ),
            chemistry_context=copy.deepcopy(dict(request.chemistry_context)),
            num_queries=config.num_queries,
            num_hypotheses_per_query=config.num_hypotheses_per_query,
            top_n=config.top_n,
            hypothesis_ids_by_query=copy.deepcopy(
                [list(group) for group in config.hypothesis_ids_by_query]
            ),
            merge_hypothesis_ids=copy.deepcopy(list(config.merge_hypothesis_ids)),
            regeneration_package=None,
            memory_enabled=None,
            case_profile=config.case_profile,
            task_config_schema=config.task_config_schema,
            prior_manifest_path=prior_manifest_path,
            supplementary_scientific_input=(
                copy.deepcopy(dict(config.supplementary_scientific_input))
                if isinstance(config.supplementary_scientific_input, Mapping)
                else None
            ),
        )
        if not isinstance(result, Mapping):
            raise ValueError("首批 HypothesisAgent operation 必须返回 object")
        return copy.deepcopy(dict(result))

    def _replay_initial_hypothesis_operation(
        self,
        request: HypothesisInitialGenerationRequest,
        *,
        canonical: Mapping[str, Any],
        operation_instance_id: str,
    ) -> Mapping[str, Any] | None:
        """只重放指定首批 operation 的完整成功 manifest，不调用 provider。"""

        config = request.generation_config
        replay = self.hypothesis_agent.replay_initial_hypothesis_operation_from_finalized_manifest(
            task_root=config.task_root,
            operation_instance_id=operation_instance_id,
            base_input=canonical["base_input"],
            scientific_question=request.scientific_question,
            initial_retrieval_phase=copy.deepcopy(
                dict(request.initial_retrieval_phase)
            ),
            chemistry_context=copy.deepcopy(dict(request.chemistry_context)),
            num_queries=config.num_queries,
            num_hypotheses_per_query=config.num_hypotheses_per_query,
            top_n=config.top_n,
            hypothesis_ids_by_query=copy.deepcopy(
                [list(group) for group in config.hypothesis_ids_by_query]
            ),
            merge_hypothesis_ids=copy.deepcopy(list(config.merge_hypothesis_ids)),
            case_profile=config.case_profile,
            task_config_schema=config.task_config_schema,
            supplementary_scientific_input=(
                copy.deepcopy(dict(config.supplementary_scientific_input))
                if isinstance(config.supplementary_scientific_input, Mapping)
                else None
            ),
        )
        if replay is not None and not isinstance(replay, Mapping):
            raise ValueError("首批 finalized replay 必须返回 object 或 None")
        return copy.deepcopy(dict(replay)) if isinstance(replay, Mapping) else None

    def recover_initial_hypothesis_batch(
        self,
        request: HypothesisInitialGenerationRequest,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """恢复首批 operation，必要时沿已验证前缀重试传输失败阶段。"""

        canonical = _compose_canonical_initial_model_request(request)
        if receipt.get("model_request_digest") != _canonical_digest(canonical):
            raise ValueError("started receipt 的首批 model request 摘要与当前请求不一致")
        plan = _read_initial_transport_recovery_plan(receipt)
        if plan is None:
            return self._replay_initial_hypothesis_operation(
                request,
                canonical=canonical,
                operation_instance_id=request.operation_instance_id,
            )
        try:
            operation_instance_id = str(plan["retry_operation_instance_id"])
            prior_manifest_ref = plan["prior_manifest_ref"]
            if not isinstance(prior_manifest_ref, Mapping):
                raise ValueError("首批 transport recovery 缺少 prior_manifest_ref")
            located = DynamicModelOperationRunner.locate_finalized_manifest_in_operation(
                task_root=request.generation_config.task_root,
                operation="hypothesis_generation",
                operation_instance_id=operation_instance_id,
                expected_base_input=canonical["base_input"],
            )
            if located is not None:
                status = _classify_initial_finalized_manifest(
                    located,
                    task_root=request.generation_config.task_root,
                )
                if status == "validated":
                    replayed = self._replay_initial_hypothesis_operation(
                        request,
                        canonical=canonical,
                        operation_instance_id=operation_instance_id,
                    )
                    if replayed is None:
                        raise ValueError(
                            "transport recovery finalized manifest 已存在但无法 replay"
                        )
                    return replayed
                if plan.get("status") != "pending":
                    raise ValueError(
                        "已完成的 transport recovery 不能再次引用失败 manifest"
                    )
                raise _build_initial_transport_protocol_error(
                    located,
                    task_root=request.generation_config.task_root,
                )
            if plan.get("status") != "pending":
                raise ValueError(
                    "已完成的 transport recovery 缺少可重放 finalized manifest"
                )
            return self._run_initial_hypothesis_operation(
                request,
                operation_instance_id=operation_instance_id,
                prior_manifest_path=str(prior_manifest_ref["relative_path"]),
            )
        except KeyError as exc:
            raise ValueError("首批 transport recovery 计划字段缺失") from exc


def _require_batch_list(value: Any, label: str) -> list[Any]:
    """读取模型阶段中的非空数组。"""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} 必须是非空数组")
    return value


def _normalize_hypothesis(value: Any, label: str) -> dict[str, Any]:
    """保留新 BatchCreated 允许的结构化 hypothesis 字段。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必须是 object")
    allowed_fields = {
        "hypothesis_id",
        "strategy_name",
        "reasoning",
        "merged_from",
        "synthesis_sources",
        "wins",
        "losses",
        "score",
        "rank",
        "core_claim",
        "claims",
        "detailed_reasoning",
        "verification_requirements",
    }
    if set(value) - allowed_fields:
        raise ValueError(f"{label} 包含未注册字段")
    hypothesis_id = _require_identity(value.get("hypothesis_id"), f"{label}.hypothesis_id")
    try:
        compatibility = validate_structured_hypothesis_claims(
            value,
            path=label,
            allow_legacy=True,
        )
    except Exception as exc:
        raise ValueError(f"{label} 的结构化正文非法") from exc
    normalized: dict[str, Any] = {
        "hypothesis_id": hypothesis_id,
        "strategy_name": compatibility["strategy_name"],
        "reasoning": compatibility["reasoning"],
    }
    for field in ("merged_from", "synthesis_sources", "wins", "losses", "score", "rank"):
        if field in value:
            normalized[field] = _copy_json(value[field], f"{label}.{field}")
    return normalized


def _validate_generation_queries(
    value: Mapping[str, Any],
    *,
    generation_config: HypothesisRegenerationGenerationConfig,
    expected_queries: Sequence[str],
) -> list[str]:
    """核对模型返回的 Query 顺序与冻结生成配置。"""

    queries = _require_batch_list(value.get("queries"), "hypothesis_result.queries")
    if len(queries) != generation_config.num_queries or any(
        not isinstance(item, str) or not item.strip() for item in queries
    ):
        raise ValueError("hypothesis_result.queries 与冻结生成配置不一致")
    if queries != list(expected_queries):
        raise ValueError("hypothesis_result.queries 与冻结 Query 不一致")
    return queries


def _normalize_hypothesis_groups(
    value: Mapping[str, Any],
    *,
    queries: Sequence[str],
    generation_config: HypothesisRegenerationGenerationConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    """核对每个 Query 的候选身份并返回规范化分组。"""

    groups = _require_batch_list(
        value.get("hypotheses_by_query"),
        "hypothesis_result.hypotheses_by_query",
    )
    if len(groups) != len(queries):
        raise ValueError("hypotheses_by_query 数量与 queries 不一致")
    normalized_groups: list[dict[str, Any]] = []
    group_ids: list[str] = []
    for index, raw_group in enumerate(groups):
        if not isinstance(raw_group, Mapping) or raw_group.get("query") != queries[index]:
            raise ValueError("hypotheses_by_query 必须逐项对应 queries")
        raw_hypotheses = _require_batch_list(
            raw_group.get("hypotheses"),
            f"hypotheses_by_query[{index}].hypotheses",
        )
        if len(raw_hypotheses) != generation_config.num_hypotheses_per_query:
            raise ValueError("hypotheses_by_query 的候选数与冻结生成配置不一致")
        normalized = [
            _normalize_hypothesis(item, f"hypotheses_by_query[{index}].hypotheses[{offset}]")
            for offset, item in enumerate(raw_hypotheses)
        ]
        expected_ids = tuple(generation_config.hypothesis_ids_by_query[index])
        actual_ids = tuple(item["hypothesis_id"] for item in normalized)
        if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_ids):
            raise ValueError(f"hypotheses_by_query[{index}] 没有逐组覆盖冻结的 hypothesis_id")
        group_ids.extend(actual_ids)
        normalized_groups.append({"query": queries[index], "hypotheses": normalized})
    return normalized_groups, group_ids


def _normalize_ranked_hypotheses(
    value: Mapping[str, Any],
    *,
    generation_config: HypothesisRegenerationGenerationConfig,
) -> list[dict[str, Any]]:
    """核对完整 ranked_strategies 的身份、数量和连续 rank。"""

    ranked_raw = _require_batch_list(
        value.get("ranked_strategies"),
        "hypothesis_result.ranked_strategies",
    )
    ranked = [
        _normalize_hypothesis(item, f"ranked_strategies[{index}]")
        for index, item in enumerate(ranked_raw)
    ]
    if len(ranked) < generation_config.top_n:
        raise ValueError("ranked_strategies 少于五项，不能形成五个 selected combination")
    ranked_ids = [item["hypothesis_id"] for item in ranked]
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked_strategies 的 hypothesis_id 必须唯一")
    allocated_ids = {
        item for group in generation_config.hypothesis_ids_by_query for item in group
    } | set(generation_config.merge_hypothesis_ids)
    if set(ranked_ids) - allocated_ids:
        raise ValueError("ranked_strategies 使用了调用方未分配的 hypothesis_id")
    for index, item in enumerate(ranked, start=1):
        if "rank" in item:
            if isinstance(item["rank"], bool) or not isinstance(item["rank"], int):
                raise ValueError("ranked_strategies 的 rank 必须是严格整数")
            if item["rank"] != index:
                raise ValueError("ranked_strategies 的 rank 必须从 1 连续排列")
        item.setdefault("rank", index)
    return ranked


def _validate_hypothesis_phase_coverage(
    group_ids: Sequence[str],
    ranked: Sequence[Mapping[str, Any]],
    generation_config: HypothesisRegenerationGenerationConfig,
) -> None:
    """确认分组候选和完整排名覆盖同一组冻结身份。"""

    expected = {
        item for group in generation_config.hypothesis_ids_by_query for item in group
    }
    ranked_ids = {item["hypothesis_id"] for item in ranked}
    if len(group_ids) != len(set(group_ids)) or set(group_ids) != expected:
        raise ValueError("hypotheses_by_query 没有完整且唯一覆盖冻结候选身份")
    consumed_ids = _read_consumed_hypothesis_ids(ranked)
    if set(group_ids) - (ranked_ids | consumed_ids):
        raise ValueError("hypotheses_by_query 使用了不在完整排名中的 hypothesis_id")


def _normalize_hypothesis_phase(
    value: Any,
    *,
    generation_config: HypothesisRegenerationGenerationConfig,
    expected_queries: Sequence[str],
) -> dict[str, Any]:
    """校验生成结果的完整排名，并固定五个 selected candidate 的位置。"""

    if not isinstance(value, Mapping):
        raise ValueError("hypothesis_result 必须是 object")
    queries = _validate_generation_queries(
        value, generation_config=generation_config, expected_queries=expected_queries
    )
    normalized_groups, group_ids = _normalize_hypothesis_groups(
        value, queries=queries, generation_config=generation_config
    )
    ranked = _normalize_ranked_hypotheses(value, generation_config=generation_config)
    _validate_hypothesis_phase_coverage(group_ids, ranked, generation_config)
    return {
        "queries": copy.deepcopy(queries),
        "hypotheses_by_query": normalized_groups,
        "ranked_strategies": ranked,
    }


def _read_initial_query_and_retrieval(
    reader: WorkingMemoryReader,
    state: InvestigationState,
) -> tuple[list[str], dict[str, Any]]:
    """从首批不可变 BatchCreated 恢复冻结 Query 和 Retrieval。"""

    first_batch_id = state.history[0].batch_id if state.history else state.batch_id
    batch = reader.get_batch_created(state.task_id, first_batch_id)
    if not isinstance(batch, Mapping):
        raise ValueError("首批 BatchCreated 必须是 object")
    snapshot = batch.get("question_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("首批 BatchCreated 缺少 question_snapshot")
    queries = snapshot.get("initial_query_angles")
    retrieval = snapshot.get("initial_retrieval_phase")
    if not isinstance(queries, list) or any(
        not isinstance(item, str) or not item.strip() for item in queries
    ):
        raise ValueError("首批冻结 initial_query_angles 非法")
    if len(set(queries)) != len(queries):
        raise ValueError("首批冻结 initial_query_angles 存在重复 Query")
    if not isinstance(retrieval, Mapping):
        raise ValueError("首批冻结 initial_retrieval_phase 非法")
    return copy.deepcopy(queries), copy.deepcopy(dict(retrieval))


def _read_generation_config(
    reader: WorkingMemoryReader,
    task_id: str,
    supplied: HypothesisRegenerationGenerationConfig,
) -> HypothesisRegenerationGenerationConfig:
    """核对调用方配置和 Working Memory TaskCreated 的冻结预算。"""

    task_created = reader.get_task_created(task_id)
    if not isinstance(task_created, Mapping):
        raise ValueError("TaskCreated 必须是 object")
    task_config = task_created.get("task_config")
    if not isinstance(task_config, Mapping):
        raise ValueError("TaskCreated 缺少 task_config")
    expected = _read_hypothesis_budget(task_config)
    if (
        supplied.num_queries != expected["num_queries"]
        or supplied.num_hypotheses_per_query != expected["num_hypotheses_per_query"]
        or supplied.top_n != expected["top_n_hypotheses"]
    ):
        raise ValueError("调用方生成配置与 TaskCreated 冻结配置不一致")
    expected_sessions = task_config.get("run_mode", "end_to_end") != "top5_selection"
    if supplied.creates_investigation_sessions != expected_sessions:
        raise ValueError("调用方 session 创建范围与 TaskCreated.run_mode 不一致")
    return supplied


def _read_hypothesis_budget(task_config: Mapping[str, Any]) -> dict[str, int]:
    """读取 TaskCreated 冻结的 Hypothesis 三项预算。"""

    expected = {
        field: task_config.get(field)
        for field in (
            "num_queries",
            "num_hypotheses_per_query",
            "top_n_hypotheses",
        )
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in expected.values()
    ):
        raise ValueError("TaskCreated.task_config 的 Hypothesis 配置非法")
    if expected["top_n_hypotheses"] != 5:
        raise ValueError("TaskCreated.top_n_hypotheses 必须严格为 5")
    return {key: int(value) for key, value in expected.items()}


def _validate_task_root_against_task_created(
    reader: WorkingMemoryReader,
    task_id: str,
    task_root: str,
) -> None:
    """核对模型 artifact 根目录严格等于 TaskCreated 冻结输出根。"""

    task_created = reader.get_task_created(task_id)
    if not isinstance(task_created, Mapping):
        raise ValueError("TaskCreated 必须是 object")
    relative_dir = task_created.get("task_output_relative_dir")
    if not isinstance(relative_dir, str) or not relative_dir.strip():
        raise ValueError("TaskCreated 缺少冻结 task_output_relative_dir")
    relative_path = PurePosixPath(relative_dir)
    if (
        "\\" in relative_dir
        or _URI_SCHEME_RE.match(relative_dir) is not None
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or str(relative_path) != relative_dir
    ):
        raise ValueError("TaskCreated.task_output_relative_dir 不是安全相对路径")
    scratch_root = getattr(reader, "scratch_root", None)
    if not isinstance(scratch_root, (str, Path)) or not str(scratch_root).strip():
        raise ValueError("Working Memory reader 缺少冻结 scratch_root")
    expected_root = (Path(scratch_root).expanduser().resolve() / relative_path).resolve()
    actual_root = Path(task_root).expanduser().resolve()
    if actual_root != expected_root:
        raise ValueError("generation_config.task_root 与 TaskCreated 冻结输出根不一致")


def _validate_frozen_query_set(expected_queries: Sequence[str] | None) -> tuple[str, ...]:
    """校验首批冻结 Query 集合。"""

    frozen = tuple(expected_queries or ())
    if len(set(frozen)) != len(frozen) or any(
        not isinstance(query, str) or not query.strip() for query in frozen
    ):
        raise ValueError("source_query 的冻结 Query 集合非法")
    return frozen


def _read_query_provenance_phase(
    reader: WorkingMemoryReader,
    state: InvestigationState,
    batch_id: str,
    *,
    expected_queries: Sequence[str] | None,
    required_hypothesis_ids: Sequence[str] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, ...], set[str], list[Any], Any]:
    """读取一个 BatchCreated 的 query provenance 所需原始字段。"""

    payload = reader.get_batch_created(state.task_id, batch_id)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{batch_id} 缺少可审计的 hypothesis_phase")
    phase = payload.get("hypothesis_phase")
    if not isinstance(phase, Mapping):
        raise ValueError(f"{batch_id} 缺少可审计的 hypothesis_phase")
    frozen = _validate_frozen_query_set(expected_queries)
    required = _normalize_required_hypothesis_ids(required_hypothesis_ids)
    phase_queries = _validate_phase_queries(phase.get("queries"), frozen, batch_id)
    ranked = _read_ranked_strategies(phase.get("ranked_strategies"), batch_id)
    return payload, phase, frozen, required, ranked, phase_queries


def _normalize_required_hypothesis_ids(
    required_hypothesis_ids: Sequence[str] | None,
) -> set[str]:
    """规范化需要核对来源 Query 的 selected hypothesis 身份。"""

    return {
        _require_identity(item, "required hypothesis_id")
        for item in (required_hypothesis_ids or ())
    }


def _validate_phase_queries(
    phase_queries: Any,
    frozen: Sequence[str],
    batch_id: str,
) -> list[str] | None:
    """核对批次声明的 Query 列表与首批冻结集合。"""

    if phase_queries is None:
        return None
    if not isinstance(phase_queries, list):
        raise ValueError(f"{batch_id} 的 hypothesis_phase.queries 非法或重复")
    if any(not isinstance(query, str) or not query.strip() for query in phase_queries):
        raise ValueError(f"{batch_id} 的 hypothesis_phase.queries 非法或重复")
    if len(set(phase_queries)) != len(phase_queries):
        raise ValueError(f"{batch_id} 的 hypothesis_phase.queries 非法或重复")
    if frozen and tuple(phase_queries) != tuple(frozen):
        raise ValueError(f"{batch_id} 的 hypothesis_phase.queries 与首批冻结 Query 不一致")
    return phase_queries


def _read_ranked_strategies(value: Any, batch_id: str) -> list[Any]:
    """读取非空的完整 hypothesis 排名。"""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{batch_id} 缺少可核对的 ranked_strategies")
    return value


def _index_ranked_hypothesis_ids(ranked: Sequence[Any], batch_id: str) -> set[str]:
    """索引完整排名中的 hypothesis_id，并拒绝未知或重复身份。"""

    result: set[str] = set()
    for index, raw_hypothesis in enumerate(ranked):
        if not isinstance(raw_hypothesis, Mapping):
            raise ValueError(f"{batch_id}.ranked_strategies[{index}] 必须是 object")
        hypothesis_id = raw_hypothesis.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip() or hypothesis_id in result:
            raise ValueError(f"{batch_id}.ranked_strategies 存在缺失或重复 hypothesis_id")
        result.add(hypothesis_id)
    return result


def _fallback_single_query_provenance(
    payload: Mapping[str, Any],
    ranked: Sequence[Any],
    frozen_queries: Sequence[str],
    required_ids: set[str],
    batch_id: str,
) -> dict[str, str]:
    """兼容只保存完整排名的单 Query 历史。"""

    snapshot = payload.get("question_snapshot")
    snapshot_queries = snapshot.get("initial_query_angles") if isinstance(snapshot, Mapping) else None
    fallback = list(frozen_queries) or snapshot_queries
    if (
        not isinstance(fallback, (list, tuple))
        or len(fallback) != 1
        or not isinstance(fallback[0], str)
        or not fallback[0].strip()
    ):
        raise ValueError(f"{batch_id} 缺少可审计的 hypotheses_by_query source_query")
    result: dict[str, str] = {}
    for raw_hypothesis in ranked:
        if not isinstance(raw_hypothesis, Mapping):
            raise ValueError(f"{batch_id}.ranked_strategies 必须是 object")
        hypothesis_id = raw_hypothesis.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip() or hypothesis_id in result:
            raise ValueError("历史 ranked hypothesis 缺少或重复 hypothesis_id")
        result[hypothesis_id] = fallback[0]
    if required_ids and not required_ids <= set(result):
        raise ValueError(f"{batch_id} 的 selected hypothesis 缺少精确 source_query")
    return result


def _grouped_query_provenance(
    groups: Sequence[Any],
    *,
    batch_id: str,
    frozen_queries: Sequence[str],
    ranked_ids: set[str],
    required_ids: set[str],
) -> dict[str, str]:
    """从完整 hypotheses_by_query 分组恢复每个候选的来源 Query。"""

    result: dict[str, str] = {}
    seen_queries: set[str] = set()
    for group_index, raw_group in enumerate(groups):
        query, hypotheses = _read_query_provenance_group(
            raw_group,
            batch_id=batch_id,
            group_index=group_index,
            frozen_queries=frozen_queries,
        )
        if query in seen_queries:
            raise ValueError(f"{batch_id}.hypotheses_by_query 存在重复 source_query")
        seen_queries.add(query)
        _add_group_hypothesis_sources(
            result,
            hypotheses,
            query=query,
            batch_id=batch_id,
            group_index=group_index,
            ranked_ids=ranked_ids,
        )
    if not result or set(result) != ranked_ids:
        raise ValueError(f"{batch_id} 的 source_query 没有覆盖完整 ranked hypothesis provenance")
    if frozen_queries and set(seen_queries) != set(frozen_queries):
        raise ValueError(f"{batch_id} 的 source_query 分组没有完整覆盖冻结 Query")
    if required_ids and not required_ids <= set(result):
        raise ValueError(f"{batch_id} 的 selected hypothesis 缺少精确 source_query")
    return result


def _read_query_provenance_group(
    raw_group: Any,
    *,
    batch_id: str,
    group_index: int,
    frozen_queries: Sequence[str],
) -> tuple[str, list[Any]]:
    """读取一组 source query 及其候选数组。"""

    if not isinstance(raw_group, Mapping):
        raise ValueError(f"{batch_id}.hypotheses_by_query[{group_index}] 必须是 object")
    query = raw_group.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{batch_id}.hypotheses_by_query[{group_index}] 缺少 source query")
    if frozen_queries and query not in frozen_queries:
        raise ValueError(f"{batch_id}.hypotheses_by_query[{group_index}] 使用未知 source query")
    hypotheses = raw_group.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError(f"{batch_id}.hypotheses_by_query[{group_index}] 缺少 hypotheses")
    return query, hypotheses


def _add_group_hypothesis_sources(
    result: dict[str, str],
    hypotheses: Sequence[Any],
    *,
    query: str,
    batch_id: str,
    group_index: int,
    ranked_ids: set[str],
) -> None:
    """把一组候选的 hypothesis_id 映射到其 source query。"""

    for hypothesis_index, raw_hypothesis in enumerate(hypotheses):
        if not isinstance(raw_hypothesis, Mapping):
            raise ValueError(
                f"{batch_id}.hypotheses_by_query[{group_index}].hypotheses[{hypothesis_index}] 必须是 object"
            )
        hypothesis_id = raw_hypothesis.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise ValueError("历史 hypothesis 缺少明确 hypothesis_id")
        if hypothesis_id in result:
            raise ValueError(f"历史 hypothesis_id 重复，无法确定 source_query: {hypothesis_id}")
        if hypothesis_id not in ranked_ids:
            raise ValueError(f"历史 hypothesis_id 不在完整 ranked_strategies 中: {hypothesis_id}")
        result[hypothesis_id] = query


def _query_by_hypothesis_id(
    reader: WorkingMemoryReader,
    state: InvestigationState,
    batch_id: str,
    *,
    expected_queries: Sequence[str] | None = None,
    required_hypothesis_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """从对应 BatchCreated 的原始分组恢复 hypothesis 的来源 Query。

    ``expected_queries`` 和 ``required_hypothesis_ids`` 由首批冻结 Query 与当前五个
    selected combination 提供。传入后会逐项核对 query 集合、分组中的 hypothesis
    身份和 selected 身份；缺失、重复、未知或只靠文字猜测的 source_query 都直接
    拒绝。没有历史分组的旧快照只允许在首批恰有一个冻结 Query 且 ranked 身份完整
    时走兼容路径。
    """

    payload, phase, frozen, required, ranked, phase_queries = _read_query_provenance_phase(
        reader,
        state,
        batch_id,
        expected_queries=expected_queries,
        required_hypothesis_ids=required_hypothesis_ids,
    )
    ranked_ids = _index_ranked_hypothesis_ids(ranked, batch_id)
    groups = phase.get("hypotheses_by_query")
    if not isinstance(groups, list) or not groups:
        if phase_queries is not None:
            raise ValueError(f"{batch_id} 声明了 hypothesis_phase.queries 却缺少 hypotheses_by_query source_query")
        return _fallback_single_query_provenance(payload, ranked, frozen, required, batch_id)
    return _grouped_query_provenance(
        groups,
        batch_id=batch_id,
        frozen_queries=frozen,
        ranked_ids=ranked_ids,
        required_ids=required,
    )


def _finding_by_combination(batch: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """读取当前批次最新 Review 的原始 finding，不用 marker 推断原因。"""

    validations = batch.get("validations")
    if not isinstance(validations, tuple):
        validations = tuple(validations) if isinstance(validations, list) else ()
    if not validations:
        raise ValueError("重生批次缺少 BatchValidation")
    latest = validations[-1]
    if not isinstance(latest, Mapping) or not isinstance(latest.get("findings"), (list, tuple)):
        raise ValueError("最新 BatchValidation 的 findings 非法")
    result: dict[str, Mapping[str, Any]] = {}
    for finding in latest["findings"]:
        if not isinstance(finding, Mapping) or not isinstance(finding.get("combination_id"), str):
            raise ValueError("BatchValidation finding 非法")
        combination_id = finding["combination_id"]
        if combination_id in result:
            raise ValueError("BatchValidation finding 存在重复 combination_id")
        result[combination_id] = finding
    if len(result) != 5:
        raise ValueError("BatchValidation 必须保留五个 finding")
    return result


def _index_legacy_batch_records(
    batch: Any,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    """索引一个历史批次的 findings、sessions 和结果版本。"""

    payload = batch.model_dump(mode="json")
    findings = _finding_by_combination(payload)
    sessions = _index_legacy_sessions(batch.sessions)
    results = _index_legacy_results(batch.results, sessions)
    return findings, sessions, results


def _index_legacy_sessions(
    raw_sessions: Sequence[Any],
) -> dict[str, Mapping[str, Any]]:
    """索引历史批次的五个 session 当前 result_version。"""

    sessions: dict[str, Mapping[str, Any]] = {}
    for raw_session in raw_sessions:
        if not isinstance(raw_session, Mapping):
            raise ValueError("重生历史 session 必须是 object")
        combination_id = raw_session.get("combination_id")
        result_version = raw_session.get("result_version")
        if not isinstance(combination_id, str) or not combination_id.strip():
            raise ValueError("重生历史 session 缺少 combination_id")
        if combination_id in sessions:
            raise ValueError("重生历史 session 存在重复 combination_id")
        if not isinstance(result_version, str) or not result_version.strip():
            raise ValueError(f"{combination_id} 缺少当前 result_version")
        sessions[combination_id] = raw_session
    return sessions


def _index_legacy_results(
    raw_results: Sequence[Any],
    sessions: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """索引历史 InvestigationResult，并核对 result_version 组合身份。"""

    results: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            raise ValueError("重生历史 result 必须是 object")
        combination_id = raw_result.get("combination_id")
        result_version = raw_result.get("result_version")
        if not isinstance(combination_id, str) or not combination_id.strip():
            raise ValueError("重生历史 result 缺少 combination_id")
        if not isinstance(result_version, str) or not result_version.strip():
            raise ValueError(f"{combination_id} 的 InvestigationResult 缺少 result_version")
        if combination_id not in sessions:
            raise ValueError("重生历史 result 引用了未知 combination_id")
        key = (combination_id, result_version)
        if key in results:
            raise ValueError("重生历史包含重复 combination_id/result_version")
        results[key] = raw_result
    return results


def _legacy_classification(finding: Mapping[str, Any]) -> str:
    """把 Review finding 原样转换为旧兼容分类。"""

    validity = finding.get("validity")
    judgment = finding.get("hypothesis_judgment")
    if validity == "pass" and judgment == "supported":
        return "supported"
    if validity == "pass" and judgment == "refuted":
        return "reliably_refuted"
    if validity == "fail":
        return "insufficient_evidence"
    raise ValueError("Review finding 不能形成合法历史分类")


def _compose_legacy_combination_outcome(
    batch: Any,
    combination: Any,
    *,
    query_by_hypothesis: Mapping[str, str],
    findings: Mapping[str, Mapping[str, Any]],
    sessions: Mapping[str, Mapping[str, Any]],
    results: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """整理一个组合的真实 Result/Review 为旧 HypothesisAgent outcome。"""

    combination_id = combination.combination_id
    hypothesis = _normalize_hypothesis(
        {**combination.hypothesis, "hypothesis_id": combination.hypothesis_id},
        f"{batch.batch_id}.{combination_id}.hypothesis",
    )
    session = sessions.get(combination_id)
    if session is None:
        raise ValueError("重生历史 session 与五个组合没有完整对齐")
    result_version = session["result_version"]
    raw_result = results.get((combination_id, result_version))
    if raw_result is None:
        raise ValueError(f"重生历史缺少 {combination_id} 当前 result_version 对应的 InvestigationResult")
    finding = findings.get(combination_id)
    if finding is None:
        raise ValueError("重生历史缺少组合对应的 Review finding")
    observations = raw_result.get("observations")
    limitations = raw_result.get("limitations")
    if not isinstance(observations, list) or not isinstance(limitations, list):
        raise ValueError("InvestigationResult 的 observations/limitations 非法")
    reason = finding.get("repair_advice") or raw_result.get("narrative")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("重生历史缺少实际 Review 或调查理由")
    source_query = query_by_hypothesis.get(combination.hypothesis_id)
    if source_query is None:
        raise ValueError(f"{batch.batch_id} 的 hypothesis {combination.hypothesis_id} 没有明确 source_query")
    cited = [
        {
            "evidence_ref": f"e{index}",
            "step_ref": None,
            "evidence_kind": "observation",
            "name": "Investigation observation",
            "value": observation,
            "unit": None,
            "conditions": {},
        }
        for index, observation in enumerate(observations, start=1)
    ]
    return {
        "source_query": source_query,
        "hypothesis": {"strategy_name": hypothesis["strategy_name"], "reasoning": hypothesis["reasoning"]},
        "terminal_classification": _legacy_classification(finding),
        "final_judge": {"reason": reason},
        "cited_evidence": cited,
        "attempted_observables": copy.deepcopy(observations),
        "missing_evidence": copy.deepcopy(limitations),
        "tool_capability_gap": None,
    }


def _compose_legacy_batch_outcomes(
    batch: Any,
    *,
    reader: WorkingMemoryReader,
    state: InvestigationState,
    queries: Sequence[str],
) -> dict[str, Any]:
    """整理一个批次的完整历史并保留每个真实分类。"""

    selected_ids = tuple(item.hypothesis_id for item in batch.combinations)
    query_by_hypothesis = _query_by_hypothesis_id(
        reader,
        state,
        batch.batch_id,
        expected_queries=queries,
        required_hypothesis_ids=selected_ids,
    )
    if any(query not in queries for query in query_by_hypothesis.values()):
        raise ValueError(f"{batch.batch_id} 的 source_query 不在首批冻结 Query 中")
    if set(query_by_hypothesis.values()) != set(queries):
        raise ValueError(f"{batch.batch_id} 的 source_query 分组未完整覆盖冻结 Query")
    findings, sessions, results = _index_legacy_batch_records(batch)
    combination_ids = {item.combination_id for item in batch.combinations}
    if set(sessions) != combination_ids or set(findings) != combination_ids:
        raise ValueError("重生历史 session/finding 与五个组合没有完整对齐")
    return {
        "batch_id": batch.batch_id,
        "outcomes": [
            _compose_legacy_combination_outcome(
                batch,
                combination,
                query_by_hypothesis=query_by_hypothesis,
                findings=findings,
                sessions=sessions,
                results=results,
            )
            for combination in batch.combinations
        ],
    }


def _compose_legacy_regeneration_package(
    state: InvestigationState,
    working_memory: HypothesisRegenerationWorkingMemoryInput,
    reader: WorkingMemoryReader,
    generation_config: HypothesisRegenerationGenerationConfig,
) -> dict[str, Any]:
    """把完整新 Investigation 输入转成旧 HypothesisAgent 要求的严格 package。

    旧 agent 只接受固定的终局 outcome 字段；完整 sessions/results/validations 和
    原始 hypothesis 仍原样保存在 ``working_memory_payload``，因此该兼容结构不会
    取代或删减 bridge 的真实历史输入。
    """

    queries, initial_retrieval = _read_initial_query_and_retrieval(reader, state)
    if len(queries) != generation_config.num_queries:
        raise ValueError("首批冻结 Query 数量与生成配置不一致")
    scientific_context = project_hypothesis_regeneration_context(
        scientific_question=working_memory.scientific_question,
        initial_query_angles=queries,
        initial_retrieval_phase=initial_retrieval,
        chemistry_context=working_memory.chemistry_context,
    )
    all_batches = tuple(working_memory.closed_batches) + (working_memory.current_batch,)
    outcomes = [
        item
        for batch in all_batches
        for item in _compose_legacy_batch_outcomes(
            batch,
            reader=reader,
            state=state,
            queries=queries,
        )["outcomes"]
    ]
    return {
        "scientific_context": scientific_context,
        "current_query": None,
        "prior_hypothesis_outcomes": outcomes,
        "next_batch_index": working_memory.current_batch.batch_index + 1,
        "generation_config": {
            "num_queries": generation_config.num_queries,
            "num_hypotheses_per_query": generation_config.num_hypotheses_per_query,
            "top_n": generation_config.top_n,
        },
        "source_record_ids": [],
    }


def _session_identities_payload(
    identities: Sequence[HypothesisRegenerationSessionIdentity],
) -> list[dict[str, str | int]]:
    """把显式下一批身份转成 receipt 可保存的 JSON。"""

    return [
        {
            "combination_index": item.combination_index,
            "combination_id": item.combination_id,
            "workspace_ref": item.workspace_ref,
            "scope_id": item.scope_id,
        }
        for item in sorted(identities, key=lambda value: value.combination_index)
    ]


def _read_committed_session_identities(
    sessions: Sequence[Any],
    *,
    expected_by_combination: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """按组合身份读取已提交 session，避免依赖数据库返回顺序。

    输入是 Working Memory reader 返回的五个 session 行和请求中的组合索引；动作只
    把 reader 的组合、workspace、scope 映射回期望位置，不修改事实。缺少组合或身份
    字段时抛出错误，调用方随后会把恢复保留为 pending。
    """

    actual: list[dict[str, Any]] = []
    for item in sessions:
        if not isinstance(item, Mapping):
            raise ValueError("Working Memory session 行必须是 object")
        combination_id = item.get("combination_id")
        expected = expected_by_combination.get(combination_id)
        if expected is None:
            raise ValueError("Working Memory session 含未知 combination_id")
        actual.append(
            {
                "combination_index": expected["combination_index"],
                "combination_id": combination_id,
                "workspace_ref": item.get("workspace_ref"),
                "scope_id": item.get("scope_id"),
            }
        )
    return actual


def _validate_session_identities(
    identities: Sequence[HypothesisRegenerationSessionIdentity],
    *,
    expected_count: int = 5,
) -> tuple[HypothesisRegenerationSessionIdentity, ...]:
    """确认调用方提供完整的会话身份声明。"""

    values = tuple(identities)
    if expected_count not in {0, 5}:
        raise ValueError("expected_count 必须是 0 或 5")
    if len(values) != expected_count:
        raise ValueError(
            "下一批必须显式提供五个 session identity"
            if expected_count == 5
            else "筛选模式不得提供 Investigation session identity"
        )
    if expected_count == 0:
        return ()
    if sorted(item.combination_index for item in values) != [1, 2, 3, 4, 5]:
        raise ValueError("下一批 combination_index 必须恰好覆盖 1 到 5")
    if len({item.combination_id for item in values}) != 5:
        raise ValueError("下一批 combination_id 必须唯一")
    if len({item.workspace_ref for item in values}) != 5:
        raise ValueError("下一批 workspace_ref 必须逐组合唯一")
    if len({item.scope_id for item in values}) != 5:
        raise ValueError("下一批 scope_id 必须逐组合唯一")
    return tuple(sorted(values, key=lambda item: item.combination_index))


def _validate_fresh_session_identities(
    state: InvestigationState,
    identities: Sequence[HypothesisRegenerationSessionIdentity],
    *,
    next_batch_id: str,
) -> None:
    """拒绝重用历史批次的 combination、workspace 或 scope 身份。

    ``completed`` receipt 恢复时，当前状态本身已经是本次 next batch，因此只把
    关闭历史和真正的前置当前批次纳入比较；首次创建下一批时则会同时核对当前批次。
    这样既保留幂等恢复，也不允许调用方把新调查会话接回旧运行环境。
    """

    previous_sessions = [
        session
        for history in state.history
        for session in history.sessions
    ]
    if state.batch_id != next_batch_id:
        previous_sessions.extend(state.sessions)
    historical_workspaces = {session.workspace_ref for session in previous_sessions}
    historical_scopes = {session.scope_id for session in previous_sessions}
    historical_combinations = {
        session.combination_id for session in previous_sessions
    }
    for identity in identities:
        if identity.combination_id in historical_combinations:
            raise ValueError(
                f"下一批 combination_id 不能复用历史身份: {identity.combination_id}"
            )
        if identity.workspace_ref in historical_workspaces:
            raise ValueError(
                f"下一批 workspace_ref 不能复用历史身份: {identity.workspace_ref}"
            )
        if identity.scope_id in historical_scopes:
            raise ValueError(
                f"下一批 scope_id 不能复用历史身份: {identity.scope_id}"
            )


def _cleanup_workspace_text(value: Any, label: str) -> str:
    """校验 receipt 中保存的工作区引用，保留调用方给出的完整文本。"""

    text = _require_text(value, label)
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in text):
        raise ValueError(f"{label} 不能包含控制字符")
    return text


def _build_scope_cleanup_plan_from_sessions(
    sessions: Sequence[Any],
    *,
    source_batch_id: str,
    next_batch_id: str,
    enabled: bool,
) -> dict[str, Any]:
    """从五个旧会话生成逐 scope 清理计划，不调用外部服务。"""

    if len(sessions) != 5:
        raise ValueError("旧批次必须恰好包含五个 session，才能建立 scope 清理计划")
    if not source_batch_id or source_batch_id == next_batch_id:
        raise ValueError("scope 清理计划必须绑定不同的 source/target batch")
    scopes: list[dict[str, Any]] = []
    seen_combinations: set[str] = set()
    seen_scopes: set[str] = set()
    for session in sessions:
        combination_id = _require_identity(session.combination_id, "旧 scope combination_id")
        scope_id = _require_identity(session.scope_id, "旧 scope scope_id")
        workspace_ref = _cleanup_workspace_text(
            session.workspace_ref,
            "旧 scope workspace_ref",
        )
        if combination_id in seen_combinations:
            raise ValueError("旧批次 scope 清理计划包含重复 combination_id")
        if scope_id in seen_scopes:
            raise ValueError("旧批次 scope 清理计划包含重复 scope_id")
        seen_combinations.add(combination_id)
        seen_scopes.add(scope_id)
        scopes.append(
            {
                "combination_id": combination_id,
                "scope_id": scope_id,
                "workspace_ref": workspace_ref,
                "status": "pending",
            }
        )
    return {
        "schema": _CLEANUP_SCHEMA,
        "schema_version": 1,
        "status": "cleanup_pending" if enabled else "not_required",
        "source_batch_id": _require_identity(source_batch_id, "source_batch_id"),
        "target_batch_id": _require_identity(next_batch_id, "next_batch_id"),
        "scopes": scopes,
    }


def _build_scope_cleanup_plan(
    state: InvestigationState,
    *,
    next_batch_id: str,
    enabled: bool,
) -> dict[str, Any]:
    """从当前 InvestigationState 的五个会话生成 scope 清理计划。"""

    return _build_scope_cleanup_plan_from_sessions(
        state.sessions,
        source_batch_id=state.batch_id,
        next_batch_id=next_batch_id,
        enabled=enabled,
    )


def _validate_scope_cleanup_plan(
    value: Any,
    *,
    task_id: str,
    previous_batch_id: str,
    next_batch_id: str,
) -> dict[str, Any]:
    """严格核对 receipt 内的 scope 清理计划和本次 operation 身份。"""

    _require_identity(task_id, "task_id")
    plan = _read_cleanup_plan_header(
        value,
        previous_batch_id=previous_batch_id,
        next_batch_id=next_batch_id,
    )
    status = plan["status"]
    normalized_scopes = _read_cleanup_plan_scopes(plan["scopes"])
    _validate_cleanup_plan_status(status, normalized_scopes)
    last_error = _read_cleanup_plan_error(plan.get("last_error"))
    result: dict[str, Any] = {
        "schema": _CLEANUP_SCHEMA,
        "schema_version": 1,
        "status": status,
        "source_batch_id": previous_batch_id,
        "target_batch_id": next_batch_id,
        "scopes": normalized_scopes,
    }
    if last_error is not None:
        result["last_error"] = last_error
    return result


def _read_cleanup_plan_header(
    value: Any,
    *,
    previous_batch_id: str,
    next_batch_id: str,
) -> Mapping[str, Any]:
    """核对 scope cleanup 计划的固定字段，并返回原始映射。"""

    if not isinstance(value, Mapping):
        raise ValueError("重生 receipt 缺少 scope cleanup plan")
    allowed_fields = {
        "schema",
        "schema_version",
        "status",
        "source_batch_id",
        "target_batch_id",
        "scopes",
        "last_error",
    }
    if set(value) - allowed_fields:
        raise ValueError("scope cleanup plan 包含未注册字段")
    if value.get("schema") != _CLEANUP_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("scope cleanup plan schema 未注册")
    if value.get("status") not in _CLEANUP_STATUS_VALUES:
        raise ValueError("scope cleanup plan status 非法")
    if value.get("source_batch_id") != previous_batch_id:
        raise ValueError("scope cleanup plan 的 source batch 与 receipt 不一致")
    if value.get("target_batch_id") != next_batch_id:
        raise ValueError("scope cleanup plan 的 target batch 与 receipt 不一致")
    scopes = value.get("scopes")
    if not isinstance(scopes, list) or len(scopes) != 5:
        raise ValueError("scope cleanup plan 必须逐项保存旧批次五个 scope")
    return value


def _read_cleanup_plan_scopes(value: Sequence[Any]) -> list[dict[str, Any]]:
    """规范化 scope cleanup 计划中的五个旧 scope。"""

    normalized: list[dict[str, Any]] = []
    seen_combinations: set[str] = set()
    seen_scopes: set[str] = set()
    for index, raw_scope in enumerate(value):
        scope = _read_cleanup_scope(raw_scope, index=index)
        combination_id = scope["combination_id"]
        scope_id = scope["scope_id"]
        if combination_id in seen_combinations or scope_id in seen_scopes:
            raise ValueError("scope cleanup plan 的 combination_id 或 scope_id 重复")
        seen_combinations.add(combination_id)
        seen_scopes.add(scope_id)
        normalized.append(scope)
    return normalized


def _read_cleanup_scope(raw_scope: Any, *, index: int) -> dict[str, Any]:
    """核对单个旧 scope 的身份、工作区和清理状态。"""

    if not isinstance(raw_scope, Mapping):
        raise ValueError(f"scope cleanup plan scopes[{index}] 必须是 object")
    if set(raw_scope) != {
        "combination_id",
        "scope_id",
        "workspace_ref",
        "status",
    }:
        raise ValueError(f"scope cleanup plan scopes[{index}] 字段不完整")
    combination_id = _require_identity(
        raw_scope.get("combination_id"),
        f"scope cleanup plan scopes[{index}].combination_id",
    )
    scope_id = _require_identity(
        raw_scope.get("scope_id"),
        f"scope cleanup plan scopes[{index}].scope_id",
    )
    workspace_ref = _cleanup_workspace_text(
        raw_scope.get("workspace_ref"),
        f"scope cleanup plan scopes[{index}].workspace_ref",
    )
    item_status = raw_scope.get("status")
    if item_status not in _CLEANUP_ITEM_STATUS_VALUES:
        raise ValueError(f"scope cleanup plan scopes[{index}].status 非法")
    return {
        "combination_id": combination_id,
        "scope_id": scope_id,
        "workspace_ref": workspace_ref,
        "status": item_status,
    }


def _validate_cleanup_plan_status(
    status: str,
    scopes: Sequence[Mapping[str, Any]],
) -> None:
    """核对计划总状态与每个 scope 的逐项状态一致。"""

    if status == "completed" and any(item["status"] != "revoked" for item in scopes):
        raise ValueError("completed scope cleanup plan 不能包含 pending scope")
    if status == "not_required" and any(item["status"] == "revoked" for item in scopes):
        raise ValueError("not_required scope cleanup plan 不能伪造 revoked scope")


def _read_cleanup_plan_error(value: Any) -> str | None:
    """截断并核对清理失败文本；没有失败时返回 None。"""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scope cleanup plan last_error 非法")
    return value[:400]


def _normalize_source_records(source_record_ids: Sequence[str]) -> list[str]:
    """校验旧 Working Memory generation_source 的 opaque 来源身份。

    来源记录只允许由字母、数字、下划线和连字符组成的稳定标识；不接受任何
    路径分隔符、URI scheme、点段或其它可被解释为文件位置的语法。函数只返回
    排序去重后的字符串数组，输入非法、重复或顺序不稳定时抛出 ``ValueError``。
    """

    if isinstance(source_record_ids, (str, bytes, bytearray)):
        raise TypeError("source_record_ids 必须是 opaque identity 数组")
    values = []
    for item in source_record_ids:
        value = _require_text(item, "source_record_id")
        if not _OPAQUE_SOURCE_RECORD_ID_RE.fullmatch(value):
            raise ValueError("source_record_id 必须是 opaque identity，不能包含路径语义")
        if _SECRET_RE.search(value):
            raise ValueError("source_record_id 不能包含凭据")
        values.append(value)
    if values != sorted(set(values)):
        raise ValueError("source_record_ids 必须排序且去重")
    return values


def _validate_saved_operation_reference(
    receipt: Mapping[str, Any],
    *,
    task_id: str,
    next_batch_id: str,
    next_batch_index: int,
    operation_key: str,
    operation_instance_id: str,
    generation_config: HypothesisRegenerationGenerationConfig,
    identities: Sequence[HypothesisRegenerationSessionIdentity],
    source_record_ids: Sequence[str],
    causal_source_ref: str | None,
    expected_model_request: Mapping[str, Any] | None = None,
) -> None:
    """严格核对 receipt 中保存的模型 operation claim。

    输入是已经从 SQLite 读出的 receipt 和本次调用的冻结参数；动作只做字段、摘要
    和身份核对，不调用模型或外部服务。receipt 只保存 canonical request 摘要，
    不保存完整科学请求；调用方可在当前 Working Memory 快照上重建请求并比较摘要。
    """

    previous_batch_id = receipt.get("previous_batch_id")
    _require_identity(previous_batch_id, "receipt.previous_batch_id")
    model_request_digest = receipt.get("model_request_digest")
    if not isinstance(model_request_digest, str) or not _RECEIPT_DIGEST_RE.fullmatch(
        model_request_digest
    ):
        raise ValueError("重生 receipt 缺少 canonical model request 摘要")
    if receipt.get("input_digest") != model_request_digest:
        raise ValueError("重生 receipt 的 canonical input 摘要不一致")
    if (
        expected_model_request is not None
        and _canonical_digest(expected_model_request) != model_request_digest
    ):
        raise ValueError("重生 receipt 的 canonical model request 与当前请求不一致")

    reference = receipt.get("operation_reference")
    reference_digest = receipt.get("operation_reference_digest")
    if not isinstance(reference, Mapping) or not isinstance(reference_digest, str):
        raise ValueError("重生 receipt 缺少完整 operation reference")
    if _canonical_digest(reference) != reference_digest:
        raise ValueError("重生 receipt 的 operation reference 摘要不一致")
    expected_reference = {
        "operation": "hypothesis_generation",
        "task_id": task_id,
        "operation_key": operation_key,
        "operation_instance_id": operation_instance_id,
        "task_root": generation_config.task_root,
        "previous_batch_id": previous_batch_id,
        "next_batch_id": next_batch_id,
        "next_batch_index": next_batch_index,
        "memory_enabled": True,
        "generation_config_digest": _generation_config_digest(generation_config),
        "model_request_digest": model_request_digest,
        "session_identities": _session_identities_payload(identities),
        "source_record_ids": list(source_record_ids),
        "causal_source_ref": causal_source_ref,
    }
    if _copy_json(dict(reference), "operation_reference") != _copy_json(
        expected_reference,
        "expected_operation_reference",
    ):
        raise ValueError("重生 receipt 的完整 operation reference 与当前请求不一致")


def _read_generation_result(
    result: Mapping[str, Any],
    *,
    operation_instance_id: str,
    task_root: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """读取既有 operation 返回的 hypothesis phase 和 finalized manifest。

    若现有 operation 返回清单正文，额外核对其中的 operation 身份；测试或其它
    注入生成器可以只提供受信任的 ``manifest_ref``，但不能提供与本次调用冲突的
    身份。
    """

    hypothesis_result = result.get("hypothesis_result")
    operation_result = result.get("operation_result")
    if not isinstance(hypothesis_result, Mapping):
        raise ValueError("生成结果缺少 hypothesis_result")
    if not isinstance(operation_result, Mapping):
        raise ValueError("生成结果缺少 operation_result")
    if operation_result.get("status") != "validated":
        raise ValueError("Hypothesis operation 尚未达到 validated 终态")
    for container_name, container in (
        ("operation_result", operation_result),
        (
            "operation_result.manifest",
            operation_result.get("manifest"),
        ),
    ):
        if isinstance(container, Mapping):
            returned_operation = container.get("operation_instance_id")
            if returned_operation is not None and returned_operation != operation_instance_id:
                raise ValueError(f"{container_name} 的 operation_instance_id 与请求不一致")
    manifest_ref = operation_result.get("manifest_ref")
    if not isinstance(manifest_ref, Mapping):
        raise ValueError("Hypothesis operation 缺少 manifest_ref")
    _validate_hypothesis_manifest_ref(
        manifest_ref,
        task_root=task_root,
        operation_instance_id=operation_instance_id,
    )
    return hypothesis_result, operation_result


def _expected_queries_from_request(
    request: HypothesisRegenerationGenerationRequest,
) -> tuple[str, ...]:
    """读取请求中已冻结的 Query，供结果逐组核对。"""

    context = request.regeneration_package.get("scientific_context")
    if not isinstance(context, Mapping):
        raise ValueError("regeneration_package 缺少 scientific_context")
    values = context.get("initial_query_angles")
    if not isinstance(values, list) or any(
        not isinstance(item, str) or not item.strip() for item in values
    ):
        raise ValueError("regeneration_package 缺少冻结 Query")
    if len(values) != request.generation_config.num_queries:
        raise ValueError("冻结 Query 数量与生成配置不一致")
    return tuple(values)


def _compose_batch_payload(
    *,
    state: InvestigationState,
    next_batch_id: str,
    next_batch_index: int,
    operation_instance_id: str,
    hypothesis_result: Mapping[str, Any],
    generation_config: HypothesisRegenerationGenerationConfig,
    expected_queries: Sequence[str],
    causal_source_ref: str | None,
    source_record_ids: Sequence[str],
    identities: Sequence[HypothesisRegenerationSessionIdentity],
) -> dict[str, Any]:
    """把已校验模型阶段和显式会话身份组成一个完整 BatchCreated payload。"""

    phase = _normalize_hypothesis_phase(
        hypothesis_result,
        generation_config=generation_config,
        expected_queries=expected_queries,
    )
    ranked = phase["ranked_strategies"]
    ordered_identities = _validate_session_identities(
        identities,
        expected_count=5 if generation_config.creates_investigation_sessions else 0,
    )
    candidates: list[dict[str, Any]] = []
    for candidate_index, hypothesis in enumerate(ranked, start=1):
        selected = candidate_index <= generation_config.top_n
        identity = (
            ordered_identities[candidate_index - 1]
            if selected
            and generation_config.creates_investigation_sessions
            and ordered_identities
            else None
        )
        combination_id = (
            identity.combination_id
            if identity is not None
            else (
                f"top5-combination:{state.task_id}:{next_batch_id}:{candidate_index}"
                if selected
                else None
            )
        )
        combination_index = (
            identity.combination_index
            if identity is not None
            else candidate_index if selected else None
        )
        candidates.append(
            {
                "hypothesis_id": hypothesis["hypothesis_id"],
                "candidate_index": candidate_index,
                "rank": candidate_index,
                "selection_status": "selected" if selected else "not_selected",
                "combination_id": combination_id,
                "combination_index": combination_index,
            }
        )
    payload = {
        "investigation_protocol_version": INVESTIGATION_PROTOCOL_VERSION,
        "batch_id": _require_identity(next_batch_id, "next_batch_id"),
        "batch_index": next_batch_index,
        "operation_instance_id": _require_identity(operation_instance_id, "operation_instance_id"),
        "question_snapshot": None,
        "generation_source": {
            "kind": "working_memory_regeneration",
            "previous_batch_id": state.batch_id,
            "source_record_ids": _normalize_source_records(source_record_ids),
        },
        "hypothesis_phase": phase,
        "candidates": candidates,
        "causal_source_ref": causal_source_ref,
    }
    return _copy_json(payload, "HypothesisBatchCreated.payload")


def _compose_initial_batch_payload(
    *,
    task_id: str,
    batch_id: str,
    operation_instance_id: str,
    scientific_question: str,
    initial_retrieval_phase: Mapping[str, Any],
    hypothesis_result: Mapping[str, Any],
    generation_config: HypothesisRegenerationGenerationConfig,
    identities: Sequence[HypothesisRegenerationSessionIdentity],
) -> dict[str, Any]:
    """把首批 validated Hypothesis 结果组成完整 BatchCreated 正文。"""

    raw_queries = hypothesis_result.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("首批 hypothesis_result.queries 必须是数组")
    phase = _normalize_hypothesis_phase(
        hypothesis_result,
        generation_config=generation_config,
        expected_queries=raw_queries,
    )
    ordered_identities = _validate_session_identities(
        identities,
        expected_count=5 if generation_config.creates_investigation_sessions else 0,
    )
    candidates: list[dict[str, Any]] = []
    for candidate_index, hypothesis in enumerate(
        phase["ranked_strategies"],
        start=1,
    ):
        selected = candidate_index <= generation_config.top_n
        identity = (
            ordered_identities[candidate_index - 1]
            if selected
            and generation_config.creates_investigation_sessions
            and ordered_identities
            else None
        )
        combination_id = (
            identity.combination_id
            if identity is not None
            else (
                f"top5-combination:{task_id}:{batch_id}:{candidate_index}"
                if selected
                else None
            )
        )
        combination_index = (
            identity.combination_index
            if identity is not None
            else candidate_index if selected else None
        )
        candidates.append(
            {
                "hypothesis_id": hypothesis["hypothesis_id"],
                "candidate_index": candidate_index,
                "rank": candidate_index,
                "selection_status": "selected" if selected else "not_selected",
                "combination_id": combination_id,
                "combination_index": combination_index,
            }
        )
    return _copy_json(
        {
            "investigation_protocol_version": INVESTIGATION_PROTOCOL_VERSION,
            "batch_id": _require_identity(batch_id, "batch_id"),
            "batch_index": 1,
            "operation_instance_id": _require_identity(
                operation_instance_id,
                "operation_instance_id",
            ),
            "question_snapshot": {
                "scientific_question": _require_text(
                    scientific_question,
                    "scientific_question",
                ),
                "initial_query_angles": copy.deepcopy(phase["queries"]),
                "initial_retrieval_phase": copy.deepcopy(
                    dict(initial_retrieval_phase)
                ),
            },
            "generation_source": {
                "kind": "initial",
                "previous_batch_id": None,
                "source_record_ids": [],
            },
            "hypothesis_phase": phase,
            "candidates": candidates,
            "causal_source_ref": None,
        },
        "initial HypothesisBatchCreated.payload",
    )


def _read_regeneration_gate_marker_ids(gate: Any) -> list[str]:
    """读取五个 revise_hypothesis marker 的 opaque event 身份。"""

    markers = _read_allowed_regeneration_gate_markers(gate)
    marker_refs = [
        _read_regeneration_marker_id(marker)
        for marker in markers
    ]
    if len(marker_refs) != 5 or len(set(marker_refs)) != 5:
        raise ValueError("Working Memory route gate 必须恰有五个 marker event")
    return marker_refs


def _read_allowed_regeneration_gate_markers(gate: Any) -> Sequence[Any]:
    """读取已通过 route gate 的 marker 数组。"""

    if not isinstance(gate, Mapping) or gate.get("allowed") is not True:
        reason = gate.get("reason") if isinstance(gate, Mapping) else None
        raise ValueError(reason or "Working Memory route marker gate 未满足")
    markers = gate.get("markers")
    if not isinstance(markers, Sequence) or isinstance(markers, (str, bytes)):
        raise ValueError("Working Memory route gate 缺少 marker 来源")
    return markers


def _read_regeneration_marker_id(marker: Any) -> str:
    """读取一个 marker event 的 opaque identity。"""

    if not isinstance(marker, Mapping):
        raise ValueError("Working Memory route marker 结构非法")
    return _require_identity(marker.get("marker_event_id"), "marker_event_id")


def _read_previous_batch_event_id(
    reader: Any,
    *,
    task_id: str,
    previous_batch_id: str,
) -> str:
    """读取上一批 BatchCreated event 身份，作为下一批来源证据。"""

    read_created = getattr(reader, "get_batch_created_receipt", None)
    if not callable(read_created):
        raise ValueError("Working Memory reader 缺少上一批 BatchCreated receipt")
    previous_receipt = read_created(task_id, previous_batch_id)
    if not isinstance(previous_receipt, Mapping):
        raise ValueError("上一批 BatchCreated receipt 结构非法")
    return _require_identity(
        previous_receipt.get("event_id"),
        "previous BatchCreated event_id",
    )


def _read_and_verify_committed_batch_receipt(
    reader: Any,
    *,
    task_id: str,
    next_batch_id: str,
    receipt: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """核对 Working Memory 已提交的 BatchCreated 正文和 receipt 游标。"""

    source_refs = receipt.get("source_refs")
    batch_event_id = receipt.get("batch_event_id")
    if not isinstance(source_refs, list) or not source_refs:
        raise ValueError("wm_written receipt 缺少 BatchCreated source_refs")
    _require_text(batch_event_id, "wm_written receipt.batch_event_id")
    saved_receipt = _read_committed_batch_receipt(
        reader,
        task_id=task_id,
        next_batch_id=next_batch_id,
    )
    _validate_committed_batch_receipt_fields(
        saved_receipt,
        batch_event_id=batch_event_id,
        source_refs=source_refs,
        payload=payload,
        receipt=receipt,
    )


def _read_committed_batch_receipt(
    reader: Any,
    *,
    task_id: str,
    next_batch_id: str,
) -> Mapping[str, Any]:
    """读取 Working Memory 已提交的 BatchCreated receipt。"""

    read_receipt = getattr(reader, "get_batch_created_receipt", None)
    if not callable(read_receipt):
        raise ValueError("Working Memory reader 缺少 BatchCreated receipt 读取接口")
    try:
        saved_receipt = read_receipt(task_id, next_batch_id)
    except BaseException as exc:
        raise ValueError("wm_written receipt 对应的 BatchCreated 事实无法读取") from exc
    return saved_receipt


def _validate_committed_batch_receipt_fields(
    saved_receipt: Any,
    *,
    batch_event_id: str,
    source_refs: Sequence[str],
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """核对已提交 BatchCreated receipt 的身份、正文和摘要。"""

    if not isinstance(saved_receipt, Mapping) or set(saved_receipt) != {
        "event_id",
        "payload",
        "source_refs",
    }:
        raise ValueError("wm_written receipt 对应的 BatchCreated receipt 结构非法")
    if saved_receipt["event_id"] != batch_event_id:
        raise ValueError("wm_written receipt 的 event_id 与 Working Memory 事实不一致")
    if saved_receipt["source_refs"] != source_refs:
        raise ValueError("wm_written receipt 的 source_refs 与 Working Memory 事实不一致")
    saved_payload = saved_receipt["payload"]
    if not isinstance(saved_payload, Mapping):
        raise ValueError("wm_written receipt 对应的 BatchCreated payload 必须是 object")
    if _copy_json(dict(saved_payload), "saved BatchCreated") != _copy_json(
        dict(payload),
        "receipt BatchCreated",
    ):
        raise ValueError("wm_written receipt 与 Working Memory BatchCreated payload 不一致")
    if receipt.get("output_digest") != _regeneration_output_digest(payload, source_refs):
        raise ValueError("receipt 的 canonical BatchCreated output 摘要不一致")


def _verify_committed_session_identities(
    reader: Any,
    *,
    task_id: str,
    next_batch_id: str,
    identities: Sequence[HypothesisRegenerationSessionIdentity],
) -> None:
    """核对已提交 BatchCreated 的五个 session identity。"""

    list_sessions = getattr(reader, "list_investigation_sessions", None)
    if not callable(list_sessions):
        raise ValueError("Working Memory reader 缺少 Investigation session 读取接口")
    sessions = list_sessions(task_id, next_batch_id, include_internal=True)
    if not isinstance(sessions, Sequence) or len(sessions) != 5:
        raise ValueError("BatchCreated 已确认但五个 Investigation session 不完整")
    expected = _session_identities_payload(_validate_session_identities(identities))
    expected_by_combination = {item["combination_id"]: item for item in expected}
    actual = _read_committed_session_identities(
        sessions,
        expected_by_combination=expected_by_combination,
    )
    if len(actual) != 5 or sorted(actual, key=lambda item: item["combination_index"]) != expected:
        raise ValueError("BatchCreated 已确认但五个 session identity 与请求不一致")


def _read_receipt_cleanup_summary(
    receipt: Mapping[str, Any],
) -> tuple[Literal["not_required", "cleanup_pending", "completed"], tuple[tuple[str, str], ...]]:
    """读取 receipt 中清理总状态和逐 scope 状态。"""

    raw_cleanup = receipt.get("cleanup")
    receipt_cleanup_status: Literal["not_required", "cleanup_pending", "completed"] = "not_required"
    cleanup_scope_statuses: tuple[tuple[str, str], ...] = ()
    if not isinstance(raw_cleanup, Mapping):
        return receipt_cleanup_status, cleanup_scope_statuses
    raw_status = raw_cleanup.get("status")
    if raw_status in _CLEANUP_STATUS_VALUES:
        receipt_cleanup_status = raw_status
    raw_scopes = raw_cleanup.get("scopes")
    if isinstance(raw_scopes, list):
        values: list[tuple[str, str]] = []
        for item in raw_scopes:
            if not isinstance(item, Mapping):
                continue
            combination_id = item.get("combination_id")
            item_status = item.get("status")
            if isinstance(combination_id, str) and isinstance(item_status, str):
                values.append((combination_id, item_status))
        cleanup_scope_statuses = tuple(values)
    return receipt_cleanup_status, cleanup_scope_statuses


@dataclass(frozen=True)
class _RegenerationCall:
    """保存已规范化的重生调用参数，供 bridge 的各阶段共享。"""

    task_id: str
    next_batch_id: str
    next_batch_index: int
    operation_instance_id: str
    identities: tuple[HypothesisRegenerationSessionIdentity, ...]
    operation_key: str
    causal_source_ref: str | None
    source_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedRegeneration:
    """保存已核对的状态、receipt 和冻结生成配置。"""

    call: _RegenerationCall
    snapshot: Any
    state: InvestigationState
    receipt_snapshot: InvestigationRegenerationReceiptSnapshot | None
    generation_config: HypothesisRegenerationGenerationConfig


def _normalize_regeneration_call(
    *,
    task_id: str,
    next_batch_id: str,
    next_batch_index: int,
    operation_instance_id: str,
    session_identities: Sequence[HypothesisRegenerationSessionIdentity],
    operation_key: str | None,
    causal_source_ref: str | None,
    source_record_ids: Sequence[str],
) -> _RegenerationCall:
    """校验重生调用参数并生成稳定 operation 身份。"""

    task_id = _require_identity(task_id, "task_id")
    next_batch_id = _require_identity(next_batch_id, "next_batch_id")
    operation_instance_id = _require_identity(operation_instance_id, "operation_instance_id")
    if (
        isinstance(next_batch_index, bool)
        or not isinstance(next_batch_index, int)
        or next_batch_index < 2
    ):
        raise ValueError("next_batch_index 必须至少为 2")
    identities = _validate_session_identities(session_identities)
    normalized_sources = tuple(_normalize_source_records(source_record_ids))
    if causal_source_ref is not None:
        causal_source_ref = _require_text(causal_source_ref, "causal_source_ref")
        if _ABSOLUTE_PATH_RE.search(causal_source_ref) or _SECRET_RE.search(
            causal_source_ref
        ):
            raise ValueError("causal_source_ref 不能包含宿主路径或凭据")
    if operation_key is not None:
        raise ValueError(
            "operation_key 不允许由调用方自定义；bridge 必须使用 task_id 和 next_batch_id 派生值"
        )
    derived_operation_key = _require_identity(
        f"regenerate:{task_id}:{next_batch_id}",
        "operation_key",
    )
    return _RegenerationCall(
        task_id=task_id,
        next_batch_id=next_batch_id,
        next_batch_index=next_batch_index,
        operation_instance_id=operation_instance_id,
        identities=identities,
        operation_key=derived_operation_key,
        causal_source_ref=causal_source_ref,
        source_record_ids=normalized_sources,
    )


class InvestigationHypothesisRegenerationBridge:
    """协调新 Investigation 重生输入和唯一 Working Memory 原子提交。"""

    def __init__(
        self,
        *,
        investigation_store: InvestigationRegenerationPersistence,
        working_memory_reader: WorkingMemoryReader,
        working_memory_writer: WorkingMemoryHypothesisBatchWriter,
        hypothesis_generator: HypothesisRegenerationGenerator,
        scope_credentials: InvestigationScopeRevokeProtocol | None = None,
        allow_test_scope_cleanup_bypass: bool = False,
    ) -> None:
        """注入持久状态、Working Memory、生成器和 scope 撤销依赖。

        输入的 scope 凭据对象只需要提供按完整身份撤销 capability 的方法；动作不
        在构造阶段调用它。生产调用必须注入该依赖，缺失时在任何模型、Working
        Memory 或 Investigation 写入前失败。只有测试明确传入
        ``allow_test_scope_cleanup_bypass=True`` 才能跳过该依赖；跳过路径永远返回
        ``pending``，不能把未执行的清理报告为完成。类型错误立即失败且不产生外部
        副作用。
        """

        if not isinstance(investigation_store, InvestigationRegenerationPersistence):
            raise TypeError("investigation_store 必须实现 InvestigationRegenerationPersistence")
        if not isinstance(working_memory_reader, WorkingMemoryReader):
            raise TypeError("working_memory_reader 必须实现 WorkingMemoryReader")
        if not isinstance(working_memory_reader, WorkingMemoryBatchReceiptReader):
            raise TypeError(
                "working_memory_reader 必须实现 WorkingMemoryBatchReceiptReader；"
                "恢复需要读取 BatchCreated event/payload/source_refs"
            )
        if not isinstance(working_memory_writer, WorkingMemoryHypothesisBatchWriter):
            raise TypeError("working_memory_writer 必须实现 WorkingMemoryHypothesisBatchWriter")
        if not isinstance(hypothesis_generator, HypothesisRegenerationGenerator):
            raise TypeError("hypothesis_generator 必须实现 HypothesisRegenerationGenerator")
        if not isinstance(allow_test_scope_cleanup_bypass, bool):
            raise TypeError("allow_test_scope_cleanup_bypass 必须是布尔值")
        if scope_credentials is not None and not isinstance(
            scope_credentials,
            InvestigationScopeRevokeProtocol,
        ):
            raise TypeError("scope_credentials 必须实现 InvestigationScopeRevokeProtocol")
        if scope_credentials is None and not allow_test_scope_cleanup_bypass:
            raise ValueError(
                "生产 Hypothesis 重生 bridge 必须显式注入 scope_credentials；"
                "测试替身必须明确设置 allow_test_scope_cleanup_bypass=True"
            )
        self.investigation_store = investigation_store
        self.working_memory_reader = working_memory_reader
        self.working_memory_writer = working_memory_writer
        self.hypothesis_generator = hypothesis_generator
        self.scope_credentials = scope_credentials
        self.allow_test_scope_cleanup_bypass = allow_test_scope_cleanup_bypass

    def regenerate_hypothesis_batch_from_investigation(
        self,
        *,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        operation_instance_id: str,
        session_identities: Sequence[HypothesisRegenerationSessionIdentity],
        generation_config: HypothesisRegenerationGenerationConfig,
        operation_key: str | None = None,
        causal_source_ref: str | None = None,
        source_record_ids: Sequence[str] = (),
    ) -> HypothesisRegenerationBridgeOutcome:
        """执行一次可恢复的五组合假设重生。

        输入必须是已落盘的 ``InvestigationState``、显式下一批身份和已冻结生成配置；
        动作先核对五个 ``revise_hypothesis`` marker，再读取全部历史与原始结构化
        hypothesis，调用注入的生成器，原子写入一次 BatchCreated，最后以
        ``begin_next_investigation_batch`` 原子保存下一批 BatchCreated 和五个 session。
        缺少 marker、身份漂移或正文不完整在任何写入前失败；模型或 WM 故障会留下
        可恢复 receipt，后续调用不会重复已确认的生成或写入。只有 WM 批次及五个
        session 都按 event identity 确认成功后，才按同一 receipt 逐项撤销旧 scope。
        撤销未知、失败或进度保存失败返回 ``cleanup_pending``。
        """

        call = _normalize_regeneration_call(
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_instance_id=operation_instance_id,
            session_identities=session_identities,
            operation_key=operation_key,
            causal_source_ref=causal_source_ref,
            source_record_ids=source_record_ids,
        )
        prepared = self._prepare_regeneration(call, generation_config)
        call = prepared.call
        if prepared.receipt_snapshot is not None:
            return self._resume_prepared_receipt(prepared)
        request = self._build_prepared_generation_request(prepared)
        started_receipt = self._create_started_receipt(
            prepared.snapshot,
            request=request,
            identities=call.identities,
            source_record_ids=call.source_record_ids,
            causal_source_ref=call.causal_source_ref,
            cleanup_enabled=self.scope_credentials is not None,
        )
        if started_receipt is None:
            return self._resume_competing_receipt(prepared)
        generated_receipt = self._generate_and_save_regeneration_receipt(
            prepared,
            request=request,
            started_receipt=started_receipt,
        )
        return self._write_working_memory_and_begin_next_batch(
            prepared.snapshot,
            generated_receipt,
            task_id=call.task_id,
            next_batch_id=call.next_batch_id,
            next_batch_index=call.next_batch_index,
            operation_instance_id=call.operation_instance_id,
            identities=call.identities,
            generation_config=prepared.generation_config,
            operation_key=call.operation_key,
            source_record_ids=call.source_record_ids,
            causal_source_ref=call.causal_source_ref,
            expected_queries=_expected_queries_from_request(request),
            generation_request=request,
            verify_saved_operation_output=False,
            generation_called=True,
        )

    def _prepare_regeneration(
        self,
        call: _RegenerationCall,
        supplied_config: HypothesisRegenerationGenerationConfig,
    ) -> _PreparedRegeneration:
        """读取并核对当前 Investigation、receipt 和冻结生成配置。"""

        snapshot = self.investigation_store.load_investigation_snapshot(call.task_id)
        if snapshot is None:
            raise ValueError("InvestigationState 尚未持久化")
        state = snapshot.state
        if state.task_id != call.task_id:
            raise ValueError("InvestigationState.task_id 与请求不匹配")
        _validate_fresh_session_identities(
            state,
            call.identities,
            next_batch_id=call.next_batch_id,
        )
        receipt_snapshot = self.investigation_store.load_hypothesis_regeneration_receipt(
            call.task_id,
            call.operation_key,
        )
        source_record_ids = self._resolve_prepared_source_record_ids(
            call,
            state=state,
            receipt_snapshot=receipt_snapshot,
        )
        generation_config = _read_generation_config(
            self.working_memory_reader,
            call.task_id,
            supplied_config,
        )
        _validate_task_root_against_task_created(
            self.working_memory_reader,
            call.task_id,
            generation_config.task_root,
        )
        prepared_call = replace(call, source_record_ids=source_record_ids)
        return _PreparedRegeneration(
            call=prepared_call,
            snapshot=snapshot,
            state=state,
            receipt_snapshot=receipt_snapshot,
            generation_config=generation_config,
        )

    def _resolve_prepared_source_record_ids(
        self,
        call: _RegenerationCall,
        *,
        state: InvestigationState,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot | None,
    ) -> tuple[str, ...]:
        """读取首次 gate 来源或复用 receipt 已保存的来源身份。"""

        if receipt_snapshot is None:
            self._assert_regeneration_gate(
                state,
                call.next_batch_id,
                call.next_batch_index,
            )
            return self._resolve_next_batch_source_record_ids(
                task_id=call.task_id,
                previous_batch_id=state.batch_id,
                source_record_ids=call.source_record_ids,
            )
        saved_source_record_ids = receipt_snapshot.receipt.get("source_record_ids")
        if not isinstance(saved_source_record_ids, list):
            raise ValueError("已有重生 receipt.source_record_ids 必须是数组")
        return tuple(_normalize_source_records(saved_source_record_ids))

    def _resume_prepared_receipt(
        self,
        prepared: _PreparedRegeneration,
    ) -> HypothesisRegenerationBridgeOutcome:
        """恢复已有 receipt，并在需要时从历史读取前一批只读状态。"""

        receipt_snapshot = prepared.receipt_snapshot
        if receipt_snapshot is None:  # pragma: no cover - 调用方已做屏障
            raise ValueError("缺少待恢复的重生 receipt")
        current_snapshot = self.investigation_store.load_investigation_snapshot(
            prepared.call.task_id
        )
        if current_snapshot is None:
            raise ValueError("重生 receipt 存在但 InvestigationState 已消失")
        return self._resume_saved_receipt(
            current_snapshot,
            receipt_snapshot,
            task_id=prepared.call.task_id,
            next_batch_id=prepared.call.next_batch_id,
            next_batch_index=prepared.call.next_batch_index,
            operation_instance_id=prepared.call.operation_instance_id,
            identities=prepared.call.identities,
            generation_config=prepared.generation_config,
            operation_key=prepared.call.operation_key,
            source_record_ids=prepared.call.source_record_ids,
            causal_source_ref=prepared.call.causal_source_ref,
        )

    def _build_prepared_generation_request(
        self,
        prepared: _PreparedRegeneration,
    ) -> HypothesisRegenerationGenerationRequest:
        """从上一批 Working Memory 事实构造本次 Hypothesis operation 请求。"""

        call = prepared.call
        working_memory = build_hypothesis_regeneration_working_memory_input(
            prepared.state,
            self.working_memory_reader,
        )
        batch_indexes = [item.batch_index for item in working_memory.closed_batches]
        batch_indexes.append(working_memory.current_batch.batch_index)
        if batch_indexes != list(range(1, len(batch_indexes) + 1)):
            raise ValueError("Working Memory 批次序号不连续，不能安全重生")
        if call.next_batch_index != working_memory.current_batch.batch_index + 1:
            raise ValueError("next_batch_index 与 Working Memory 当前批次不连续")
        working_memory_payload = working_memory.to_prompt_payload()
        legacy_package = _compose_legacy_regeneration_package(
            prepared.state,
            working_memory,
            self.working_memory_reader,
            prepared.generation_config,
        )
        return HypothesisRegenerationGenerationRequest(
            task_id=call.task_id,
            previous_batch_id=prepared.state.batch_id,
            next_batch_id=call.next_batch_id,
            next_batch_index=call.next_batch_index,
            operation_key=call.operation_key,
            operation_instance_id=call.operation_instance_id,
            working_memory_input=working_memory,
            working_memory_payload=copy.deepcopy(working_memory_payload),
            regeneration_package=copy.deepcopy(legacy_package),
            generation_config=prepared.generation_config,
        )

    def _resume_competing_receipt(
        self,
        prepared: _PreparedRegeneration,
    ) -> HypothesisRegenerationBridgeOutcome:
        """CAS 竞争后读取对手 receipt；缺失事实时 fail-closed。"""

        call = prepared.call
        competing = self.investigation_store.load_hypothesis_regeneration_receipt(
            call.task_id,
            call.operation_key,
        )
        if competing is None:
            raise RuntimeError("重生 receipt 竞争后无法恢复")
        current_snapshot = self.investigation_store.load_investigation_snapshot(
            call.task_id
        )
        if current_snapshot is None:
            raise ValueError("竞争 receipt 存在但 InvestigationState 已消失")
        return self._resume_saved_receipt(
            current_snapshot,
            competing,
            task_id=call.task_id,
            next_batch_id=call.next_batch_id,
            next_batch_index=call.next_batch_index,
            operation_instance_id=call.operation_instance_id,
            identities=call.identities,
            generation_config=prepared.generation_config,
            operation_key=call.operation_key,
            source_record_ids=call.source_record_ids,
            causal_source_ref=call.causal_source_ref,
        )

    def _generate_and_save_regeneration_receipt(
        self,
        prepared: _PreparedRegeneration,
        *,
        request: HypothesisRegenerationGenerationRequest,
        started_receipt: InvestigationRegenerationReceiptSnapshot,
    ) -> InvestigationRegenerationReceiptSnapshot:
        """调用一次 Hypothesis operation，并保存 finalized manifest 摘要 receipt。"""

        try:
            generated_result = self.hypothesis_generator.generate_hypothesis_regeneration(
                request
            )
            hypothesis_result, operation_result = _read_generation_result(
                generated_result,
                operation_instance_id=prepared.call.operation_instance_id,
                task_root=prepared.generation_config.task_root,
            )
            payload = _compose_batch_payload(
                state=prepared.state,
                next_batch_id=prepared.call.next_batch_id,
                next_batch_index=prepared.call.next_batch_index,
                operation_instance_id=prepared.call.operation_instance_id,
                hypothesis_result=hypothesis_result,
                generation_config=prepared.generation_config,
                expected_queries=_expected_queries_from_request(request),
                causal_source_ref=prepared.call.causal_source_ref,
                source_record_ids=prepared.call.source_record_ids,
                identities=prepared.call.identities,
            )
            manifest_ref = operation_result["manifest_ref"]
            source_refs = [str(manifest_ref["relative_path"])]
        except BaseException as exc:
            if _is_busy_operation_error(exc):
                outcome = self._receipt_outcome(
                    started_receipt,
                    state=prepared.state,
                    generation_called=True,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason="同一 Hypothesis operation 仍在运行；保持 started pending",
                    status="pending",
                )
                raise HypothesisRegenerationBridgeError(
                    "Hypothesis operation 仍在运行；receipt 保持 started pending",
                    outcome=outcome,
                ) from exc
            failed = self._mark_failed_receipt(started_receipt, exc)
            outcome = self._receipt_outcome(
                failed,
                state=prepared.state,
                generation_called=True,
                working_memory_written=False,
                investigation_committed=False,
                reason=_safe_failure_text(exc),
            )
            raise HypothesisRegenerationBridgeError(
                "Hypothesis 重生生成失败；receipt 已冻结，禁止自动重复生成",
                outcome=outcome,
            ) from exc
        try:
            return self._save_generated_receipt(
                started_receipt,
                payload=payload,
                source_refs=source_refs,
            )
        except BaseException as exc:
            outcome = self._receipt_outcome(
                started_receipt,
                state=prepared.state,
                generation_called=True,
                working_memory_written=False,
                investigation_committed=False,
                reason="validated manifest 已存在但 generated receipt 暂未保存；等待严格 replay",
                status="pending",
            )
            raise HypothesisRegenerationBridgeError(
                "generated receipt 写入失败；下次恢复只重放同一 finalized manifest",
                outcome=outcome,
            ) from exc

    def _resolve_next_batch_source_record_ids(
        self,
        *,
        task_id: str,
        previous_batch_id: str,
        source_record_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """在首次提交下一批前重建 WM 会追加的 marker/event 来源身份。

        真实 Working Memory writer 会在事务内把上一批五个 ``revise_hypothesis``
        marker 和上一批 ``BatchCreated`` event 加入 payload。bridge 先读出同一组
        事实并纳入 digest；测试 reader 没有该可选只读接口时保留其显式来源，兼容
        独立 bridge 测试替身。任何生产 gate 结构不完整都会在模型前失败。
        """

        normalized = list(_normalize_source_records(source_record_ids))
        check_gate = getattr(
            self.working_memory_reader,
            "check_revise_hypothesis_batch_gate",
            None,
        )
        if not callable(check_gate):
            return tuple(normalized)
        gate = check_gate(task_id, previous_batch_id)
        marker_refs = _read_regeneration_gate_marker_ids(gate)
        previous_event_id = _read_previous_batch_event_id(
            self.working_memory_reader,
            task_id=task_id,
            previous_batch_id=previous_batch_id,
        )
        return tuple(
            _normalize_source_records(
                sorted(set(normalized + marker_refs + [previous_event_id]))
            )
        )

    def create_initial_hypothesis_batch_from_retrieval(
        self,
        *,
        task_id: str,
        batch_id: str,
        operation_instance_id: str,
        scientific_question: str,
        initial_retrieval_phase: Mapping[str, Any],
        chemistry_context: Mapping[str, Any],
        session_identities: Sequence[HypothesisRegenerationSessionIdentity],
        generation_config: HypothesisRegenerationGenerationConfig,
        operation_key: str | None = None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """以同一 receipt/replay 协议创建首批 BatchCreated。

        首批没有上一批 route marker，但模型请求、manifest、WM BatchCreated 和完成
        游标仍使用稳定 operation 身份。模型返回后到 WM 提交前任一崩溃窗口只重放
        finalized manifest，不会再次发送模型请求。
        """

        task_id = _require_identity(task_id, "task_id")
        batch_id = _require_identity(batch_id, "batch_id")
        operation_instance_id = _require_identity(
            operation_instance_id,
            "operation_instance_id",
        )
        scientific_question = _require_text(scientific_question, "scientific_question")
        if not isinstance(initial_retrieval_phase, Mapping):
            raise ValueError("initial_retrieval_phase 必须是 object")
        if not isinstance(chemistry_context, Mapping):
            raise ValueError("chemistry_context 必须是 object")
        generation_config = _read_generation_config(
            self.working_memory_reader,
            task_id,
            generation_config,
        )
        identities = _validate_session_identities(
            session_identities,
            expected_count=5 if generation_config.creates_investigation_sessions else 0,
        )
        if operation_key is not None:
            raise ValueError(
                "operation_key 不允许由调用方自定义；bridge 必须使用 task_id 和 batch_id 派生值"
            )
        operation_key = _require_identity(
            f"initial:{task_id}:{batch_id}",
            "operation_key",
        )
        request = HypothesisInitialGenerationRequest(
            task_id=task_id,
            batch_id=batch_id,
            operation_key=operation_key,
            operation_instance_id=operation_instance_id,
            scientific_question=scientific_question,
            initial_retrieval_phase=copy.deepcopy(dict(initial_retrieval_phase)),
            chemistry_context=copy.deepcopy(dict(chemistry_context)),
            generation_config=generation_config,
        )
        receipt_snapshot = self.investigation_store.load_hypothesis_regeneration_receipt(
            task_id,
            operation_key,
        )
        if receipt_snapshot is not None:
            return self._resume_initial_receipt(
                receipt_snapshot,
                request=request,
                identities=identities,
            )
        started = self._create_initial_started_receipt(
            request,
            identities=identities,
        )
        if started is None:
            competing = self.investigation_store.load_hypothesis_regeneration_receipt(
                task_id,
                operation_key,
            )
            if competing is None:
                raise RuntimeError("首批 receipt 竞争后无法恢复")
            return self._resume_initial_receipt(
                competing,
                request=request,
                identities=identities,
            )
        try:
            generated = self._generate_initial_operation(request)
            hypothesis_result, operation_result = _read_generation_result(
                generated,
                operation_instance_id=operation_instance_id,
                task_root=generation_config.task_root,
            )
            payload = _compose_initial_batch_payload(
                task_id=task_id,
                batch_id=batch_id,
                operation_instance_id=operation_instance_id,
                scientific_question=scientific_question,
                initial_retrieval_phase=initial_retrieval_phase,
                hypothesis_result=hypothesis_result,
                generation_config=generation_config,
                identities=identities,
            )
            source_refs = [str(operation_result["manifest_ref"]["relative_path"])]
        except BaseException as exc:
            if _is_busy_operation_error(exc):
                return self._initial_receipt_outcome(
                    started,
                    generation_called=True,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason="同一首批 Hypothesis operation 仍在运行；保持 started pending",
                    status="pending",
                )
            try:
                transport_manifest_ref = _extract_initial_transport_manifest_ref(
                    exc,
                    task_root=generation_config.task_root,
                    operation_instance_id=operation_instance_id,
                    expected_base_input=_compose_canonical_initial_model_request(
                        request
                    )["base_input"],
                )
            except BaseException as evidence_error:
                failed = self._mark_failed_receipt(started, evidence_error)
                raise HypothesisRegenerationBridgeError(
                    "首批传输失败证据不完整；receipt 已冻结",
                    outcome=self._initial_receipt_outcome(
                        failed,
                        generation_called=True,
                        working_memory_written=False,
                        investigation_committed=False,
                        reason=_safe_failure_text(evidence_error),
                        status="failed",
                    ),
                ) from evidence_error
            if transport_manifest_ref is not None:
                try:
                    pending = self._save_initial_transport_recovery_plan(
                        started,
                        request=request,
                        attempt=1,
                        prior_operation_instance_id=operation_instance_id,
                        prior_manifest_ref=transport_manifest_ref,
                    )
                except BaseException as persistence_error:
                    failed = self._mark_failed_receipt(started, persistence_error)
                    raise HypothesisRegenerationBridgeError(
                        "首批传输失败恢复计划无法保存；receipt 已冻结",
                        outcome=self._initial_receipt_outcome(
                            failed,
                            generation_called=True,
                            working_memory_written=False,
                            investigation_committed=False,
                            reason=_safe_failure_text(persistence_error),
                            status="failed",
                        ),
                    ) from persistence_error
                raise HypothesisRegenerationBridgeError(
                    "首批 Hypothesis 传输失败；已保存 retry manifest，等待同任务恢复",
                    outcome=self._initial_receipt_outcome(
                        pending,
                        generation_called=True,
                        working_memory_written=False,
                        investigation_committed=False,
                        reason="已验证传输失败；恢复将复用冻结 Retrieval 和成功前缀",
                        status="pending",
                    ),
                ) from exc
            failed = self._mark_failed_receipt(started, exc)
            raise HypothesisRegenerationBridgeError(
                "首批 Hypothesis 生成失败；receipt 已冻结，禁止自动重复生成",
                outcome=self._initial_receipt_outcome(
                    failed,
                    generation_called=True,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason=_safe_failure_text(exc),
                    status="failed",
                ),
            ) from exc
        try:
            generated_receipt = self._save_generated_receipt(
                started,
                payload=payload,
                source_refs=source_refs,
            )
        except BaseException as exc:
            raise HypothesisRegenerationBridgeError(
                "首批 generated receipt 写入失败；恢复只重放同一 finalized manifest",
                outcome=self._initial_receipt_outcome(
                    started,
                    generation_called=True,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason="validated manifest 已存在但 generated receipt 暂未保存",
                    status="pending",
                ),
            ) from exc
        return self._write_initial_batch_to_working_memory(
            generated_receipt,
            request=request,
            identities=identities,
            generation_called=True,
            payload=payload,
            source_refs=source_refs,
        )

    def _generate_initial_operation(
        self,
        request: HypothesisInitialGenerationRequest,
    ) -> Mapping[str, Any]:
        """调用注入的首批生成器并确认其实现了窄接口。"""

        generate = getattr(
            self.hypothesis_generator,
            "generate_initial_hypothesis_batch",
            None,
        )
        if not callable(generate):
            raise ValueError("首批生成必须提供 Hypothesis operation 窄接口")
        result = generate(request)
        if not isinstance(result, Mapping):
            raise ValueError("首批生成器必须返回 object")
        return copy.deepcopy(dict(result))

    def _create_initial_started_receipt(
        self,
        request: HypothesisInitialGenerationRequest,
        *,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
    ) -> InvestigationRegenerationReceiptSnapshot | None:
        """保存首批模型前 receipt，CAS 竞争时不调用模型。"""

        canonical = _compose_canonical_initial_model_request(request)
        operation_reference = {
            "operation": "hypothesis_generation",
            "task_id": request.task_id,
            "operation_key": request.operation_key,
            "operation_instance_id": request.operation_instance_id,
            "task_root": request.generation_config.task_root,
            "previous_batch_id": f"initial:{request.task_id}",
            "next_batch_id": request.batch_id,
            "next_batch_index": 1,
            "memory_enabled": None,
            "generation_config_digest": _generation_config_digest(
                request.generation_config
            ),
            "model_request_digest": _canonical_digest(canonical),
            "session_identities": _session_identities_payload(identities),
            "source_record_ids": [],
            "causal_source_ref": None,
        }
        cleanup = {
            "schema": _CLEANUP_SCHEMA,
            "schema_version": 1,
            "status": "not_required",
            "source_batch_id": f"initial:{request.task_id}",
            "target_batch_id": request.batch_id,
            "scopes": [
                {
                    "combination_id": identity.combination_id,
                    "scope_id": identity.scope_id,
                    "workspace_ref": identity.workspace_ref,
                    "status": "pending",
                }
                for identity in identities
            ],
        }
        receipt = {
            "schema": _RECEIPT_SCHEMA,
            "schema_version": 1,
            "status": "started",
            "task_id": request.task_id,
            "previous_batch_id": f"initial:{request.task_id}",
            "next_batch_id": request.batch_id,
            "next_batch_index": 1,
            "operation_key": request.operation_key,
            "operation_instance_id": request.operation_instance_id,
            "input_digest": _canonical_digest(canonical),
            "model_request_digest": _canonical_digest(canonical),
            "operation_reference": operation_reference,
            "operation_reference_digest": _canonical_digest(operation_reference),
            "generation_config_digest": _generation_config_digest(
                request.generation_config
            ),
            "session_identities": _session_identities_payload(identities),
            "source_record_ids": [],
            "causal_source_ref": None,
            "cleanup": cleanup,
        }
        return self.investigation_store.claim_hypothesis_regeneration_receipt(
            request.task_id,
            request.operation_key,
            receipt,
        )

    def _resume_initial_receipt(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        request: HypothesisInitialGenerationRequest,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
    ) -> HypothesisRegenerationBridgeOutcome:
        """按首批 receipt 阶段恢复，不重新发送已完成模型请求。"""

        receipt = receipt_snapshot.receipt
        status = _validate_regeneration_receipt_schema(receipt)
        self._validate_initial_receipt_identity(
            receipt,
            request=request,
            identities=identities,
        )
        if status == "failed":
            return self._initial_receipt_outcome(
                receipt_snapshot,
                generation_called=False,
                working_memory_written=False,
                investigation_committed=False,
                reason=str(receipt.get("failure") or "首批生成失败，receipt 已冻结"),
                status="failed",
            )
        if status == "started":
            recover = getattr(
                self.hypothesis_generator,
                "recover_initial_hypothesis_batch",
                None,
            )
            if not callable(recover):
                return self._initial_receipt_outcome(
                    receipt_snapshot,
                    generation_called=False,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason="started receipt 尚无 finalized manifest；保持 pending",
                    status="pending",
                )
            try:
                recovered = recover(request, copy.deepcopy(dict(receipt)))
                if recovered is None:
                    return self._initial_receipt_outcome(
                        receipt_snapshot,
                        generation_called=False,
                        working_memory_written=False,
                        investigation_committed=False,
                        reason="同一首批 operation 尚未发现 finalized manifest；保持 pending",
                        status="pending",
                    )
                payload, source_refs = self._compose_initial_replayed_payload(
                    recovered,
                    request=request,
                    identities=identities,
                    receipt=receipt,
                )
                generated = self._save_generated_receipt(
                    receipt_snapshot,
                    payload=payload,
                    source_refs=source_refs,
                )
            except BaseException as exc:
                if _is_busy_operation_error(exc):
                    return self._initial_receipt_outcome(
                        receipt_snapshot,
                        generation_called=False,
                        working_memory_written=False,
                        investigation_committed=False,
                        reason="同一首批 Hypothesis operation 仍在运行；保持 started pending",
                        status="pending",
                    )
                plan = _read_initial_transport_recovery_plan(receipt)
                active_operation_instance_id = request.operation_instance_id
                if plan is not None:
                    active_operation_instance_id = str(
                        plan["retry_operation_instance_id"]
                    )
                try:
                    transport_manifest_ref = _extract_initial_transport_manifest_ref(
                        exc,
                        task_root=request.generation_config.task_root,
                        operation_instance_id=active_operation_instance_id,
                        expected_base_input=_compose_canonical_initial_model_request(
                            request
                        )["base_input"],
                    )
                except BaseException as evidence_error:
                    failed = self._mark_failed_receipt(
                        receipt_snapshot,
                        evidence_error,
                    )
                    raise HypothesisRegenerationBridgeError(
                        "首批恢复传输失败证据不完整；receipt 已冻结",
                        outcome=self._initial_receipt_outcome(
                            failed,
                            generation_called=False,
                            working_memory_written=False,
                            investigation_committed=False,
                            reason=_safe_failure_text(evidence_error),
                            status="failed",
                        ),
                    ) from evidence_error
                if transport_manifest_ref is not None:
                    if plan is None or plan.get("status") != "pending":
                        failed = self._mark_failed_receipt(receipt_snapshot, exc)
                        raise HypothesisRegenerationBridgeError(
                            "首批恢复缺少可继续的 transport recovery 计划；receipt 已冻结",
                            outcome=self._initial_receipt_outcome(
                                failed,
                                generation_called=False,
                                working_memory_written=False,
                                investigation_committed=False,
                                reason=_safe_failure_text(exc),
                                status="failed",
                            ),
                        ) from exc
                    try:
                        pending = self._save_initial_transport_recovery_plan(
                            receipt_snapshot,
                            request=request,
                            attempt=int(plan["attempt"]) + 1,
                            prior_operation_instance_id=active_operation_instance_id,
                            prior_manifest_ref=transport_manifest_ref,
                        )
                    except BaseException as persistence_error:
                        failed = self._mark_failed_receipt(
                            receipt_snapshot,
                            persistence_error,
                        )
                        raise HypothesisRegenerationBridgeError(
                            "首批恢复 transport retry 计划无法保存；receipt 已冻结",
                            outcome=self._initial_receipt_outcome(
                                failed,
                                generation_called=False,
                                working_memory_written=False,
                                investigation_committed=False,
                                reason=_safe_failure_text(persistence_error),
                                status="failed",
                            ),
                        ) from persistence_error
                    raise HypothesisRegenerationBridgeError(
                        "首批恢复再次传输失败；已保存下一次 retry manifest",
                        outcome=self._initial_receipt_outcome(
                            pending,
                            generation_called=False,
                            working_memory_written=False,
                            investigation_committed=False,
                            reason="已验证传输失败；继续复用同一冻结 Retrieval",
                            status="pending",
                        ),
                    ) from exc
                failed = self._mark_failed_receipt(receipt_snapshot, exc)
                raise HypothesisRegenerationBridgeError(
                    "首批 started receipt replay 失败；receipt 已冻结",
                    outcome=self._initial_receipt_outcome(
                        failed,
                        generation_called=False,
                        working_memory_written=False,
                        investigation_committed=False,
                        reason=_safe_failure_text(exc),
                        status="failed",
                    ),
                ) from exc
            return self._write_initial_batch_to_working_memory(
                generated,
                request=request,
                identities=identities,
                generation_called=False,
                payload=payload,
                source_refs=source_refs,
            )
        return self._write_initial_batch_to_working_memory(
            receipt_snapshot,
            request=request,
            identities=identities,
            generation_called=False,
        )

    def _validate_initial_receipt_identity(
        self,
        receipt: Mapping[str, Any],
        *,
        request: HypothesisInitialGenerationRequest,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
    ) -> None:
        """核对首批 receipt 的稳定 operation、配置和会话身份。"""

        canonical = _compose_canonical_initial_model_request(request)
        expected = {
            "task_id": request.task_id,
            "previous_batch_id": f"initial:{request.task_id}",
            "next_batch_id": request.batch_id,
            "next_batch_index": 1,
            "operation_key": request.operation_key,
            "operation_instance_id": request.operation_instance_id,
            "session_identities": _session_identities_payload(identities),
            "source_record_ids": [],
            "causal_source_ref": None,
            "generation_config_digest": _generation_config_digest(
                request.generation_config
            ),
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                raise ValueError(f"已有首批 receipt.{field} 与本次请求不一致")
        digest = _canonical_digest(canonical)
        if receipt.get("input_digest") != digest or receipt.get("model_request_digest") != digest:
            raise ValueError("首批 receipt 的 canonical model request 摘要不一致")
        reference = receipt.get("operation_reference")
        reference_digest = receipt.get("operation_reference_digest")
        if not isinstance(reference, Mapping) or not isinstance(reference_digest, str):
            raise ValueError("首批 receipt 缺少完整 operation reference")
        if _canonical_digest(reference) != reference_digest:
            raise ValueError("首批 receipt 的 operation reference 摘要不一致")
        expected_reference = {
            "operation": "hypothesis_generation",
            "task_id": request.task_id,
            "operation_key": request.operation_key,
            "operation_instance_id": request.operation_instance_id,
            "task_root": request.generation_config.task_root,
            "previous_batch_id": f"initial:{request.task_id}",
            "next_batch_id": request.batch_id,
            "next_batch_index": 1,
            "memory_enabled": None,
            "generation_config_digest": _generation_config_digest(
                request.generation_config
            ),
            "model_request_digest": digest,
            "session_identities": _session_identities_payload(identities),
            "source_record_ids": [],
            "causal_source_ref": None,
        }
        reference_without_recovery = dict(reference)
        plan = reference_without_recovery.pop("transport_recovery", None)
        if _copy_json(reference_without_recovery, "operation_reference") != _copy_json(
            expected_reference,
            "expected_operation_reference",
        ):
            raise ValueError("首批 receipt 的完整 operation reference 与当前请求不一致")
        if plan is not None:
            if not isinstance(plan, Mapping):
                raise ValueError("首批 receipt.transport_recovery 必须是 object")
            _validate_initial_transport_recovery_plan(plan, request=request)
        if receipt.get("cleanup", {}).get("status") != "not_required":
            raise ValueError("首批 receipt 不应携带 scope cleanup 待办")

    def _compose_initial_replayed_payload(
        self,
        replayed: Mapping[str, Any],
        *,
        request: HypothesisInitialGenerationRequest,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        receipt: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """从首批 finalized manifest 重建 payload 并核对 receipt 来源。"""

        operation_instance_id = request.operation_instance_id
        if receipt is not None:
            plan = _read_initial_transport_recovery_plan(receipt)
            if plan is not None:
                operation_instance_id = str(plan["retry_operation_instance_id"])
        hypothesis_result, operation_result = _read_generation_result(
            replayed,
            operation_instance_id=operation_instance_id,
            task_root=request.generation_config.task_root,
        )
        manifest_ref = operation_result["manifest_ref"]
        source_refs = [str(manifest_ref["relative_path"])]
        payload = _compose_initial_batch_payload(
            task_id=request.task_id,
            batch_id=request.batch_id,
            operation_instance_id=request.operation_instance_id,
            scientific_question=request.scientific_question,
            initial_retrieval_phase=request.initial_retrieval_phase,
            hypothesis_result=hypothesis_result,
            generation_config=request.generation_config,
            identities=identities,
        )
        return payload, source_refs

    def _write_initial_batch_to_working_memory(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        request: HypothesisInitialGenerationRequest,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        generation_called: bool,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] | None = None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """按首批 receipt 游标完成唯一 Working Memory BatchCreated 写入。"""

        receipt = receipt_snapshot.receipt
        status = str(receipt.get("status"))
        if payload is None or source_refs is None:
            payload, source_refs = self._compose_initial_replayed_payload_from_receipt(
                receipt,
                request=request,
                identities=identities,
            )
        else:
            payload = copy.deepcopy(dict(payload))
            source_refs = list(source_refs)
            if receipt.get("output_digest") != _regeneration_output_digest(
                payload,
                source_refs,
            ):
                raise ValueError("首批生成结果与 receipt output digest 不一致")
        if status == "generated":
            try:
                create_batch = getattr(
                    self.working_memory_writer,
                    "create_hypothesis_batch",
                    None,
                )
                if not callable(create_batch):
                    raise ValueError("Working Memory writer 缺少 create_hypothesis_batch")
                event_id = _require_text(
                    create_batch(
                        request.task_id,
                        copy.deepcopy(payload),
                        tuple(source_refs),
                    ),
                    "首批 BatchCreated event_id",
                )
            except BaseException as exc:
                raise HypothesisRegenerationBridgeError(
                    "首批 Working Memory BatchCreated 写入失败；下次恢复只重试幂等提交",
                    outcome=self._initial_receipt_outcome(
                        receipt_snapshot,
                        generation_called=generation_called,
                        working_memory_written=False,
                        investigation_committed=False,
                        reason=_safe_failure_text(exc),
                        status="pending",
                    ),
                ) from exc
            try:
                receipt_snapshot = self._save_initial_receipt_stage(
                    receipt_snapshot,
                    status="wm_written",
                    batch_event_id=event_id,
                )
            except BaseException as exc:
                raise HypothesisRegenerationBridgeError(
                    "首批 BatchCreated 已提交但 wm_written 游标暂未保存；下次恢复可幂等确认",
                    outcome=self._initial_receipt_outcome(
                        receipt_snapshot,
                        generation_called=generation_called,
                        working_memory_written=True,
                        investigation_committed=False,
                        reason="首批 wm_written 游标暂未保存",
                        status="pending",
                        batch_event_id=event_id,
                    ),
                ) from exc
            receipt = receipt_snapshot.receipt
            status = "wm_written"
        event_id = _require_text(
            receipt.get("batch_event_id"),
            "首批 BatchCreated event_id",
        )
        self._verify_committed_working_memory_batch(
            task_id=request.task_id,
            next_batch_id=request.batch_id,
            receipt=receipt,
            payload=payload,
            identities=identities,
            creates_investigation_sessions=request.generation_config.creates_investigation_sessions,
        )
        if status == "wm_written":
            try:
                receipt_snapshot = self._save_initial_receipt_stage(
                    receipt_snapshot,
                    status="completed",
                    batch_event_id=event_id,
                )
            except BaseException as exc:
                raise HypothesisRegenerationBridgeError(
                    "首批 Working Memory 已确认但 completed 游标暂未保存；下次恢复可幂等完成",
                    outcome=self._initial_receipt_outcome(
                        receipt_snapshot,
                        generation_called=generation_called,
                        working_memory_written=True,
                        investigation_committed=True,
                        reason="首批 completed 游标暂未保存",
                        status="pending",
                        batch_event_id=event_id,
                    ),
                ) from exc
        return self._initial_receipt_outcome(
            receipt_snapshot,
            generation_called=generation_called,
            working_memory_written=True,
            investigation_committed=True,
            reason=None,
            status="completed",
            batch_event_id=event_id,
        )

    def _compose_initial_replayed_payload_from_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        request: HypothesisInitialGenerationRequest,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
    ) -> tuple[dict[str, Any], list[str]]:
        """读取首批 receipt 的 manifest 并确认 canonical output digest。"""

        source_refs = receipt.get("source_refs")
        if not isinstance(source_refs, list) or len(source_refs) != 1:
            raise ValueError("首批 receipt.source_refs 必须恰含当前 manifest")
        operation_instance_id = request.operation_instance_id
        plan = _read_initial_transport_recovery_plan(receipt)
        if plan is not None:
            operation_instance_id = str(plan["retry_operation_instance_id"])
        _validate_hypothesis_manifest_source_ref(
            source_refs[0],
            task_root=request.generation_config.task_root,
            operation_instance_id=operation_instance_id,
        )
        recover = getattr(
            self.hypothesis_generator,
            "recover_initial_hypothesis_batch",
            None,
        )
        if not callable(recover):
            raise ValueError("首批恢复必须提供 finalized operation replay 接口")
        replayed = recover(request, copy.deepcopy(dict(receipt)))
        if replayed is None:
            raise ValueError("首批 operation 尚未生成 finalized manifest")
        payload, replay_source_refs = self._compose_initial_replayed_payload(
            replayed,
            request=request,
            identities=identities,
            receipt=receipt,
        )
        if replay_source_refs != source_refs:
            raise ValueError("首批 receipt.source_refs 与 finalized operation 不一致")
        if receipt.get("output_digest") != _regeneration_output_digest(
            payload,
            source_refs,
        ):
            raise ValueError("首批 receipt 的 canonical BatchCreated output 摘要不一致")
        return payload, source_refs

    def _save_initial_receipt_stage(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        status: Literal["wm_written", "completed"],
        batch_event_id: str,
    ) -> InvestigationRegenerationReceiptSnapshot:
        """保存首批 WM 写入和 completed 游标，不保存科学正文。"""

        value = dict(receipt_snapshot.receipt)
        value.update({"status": status, "batch_event_id": batch_event_id})
        revision = self.investigation_store.save_hypothesis_regeneration_receipt(
            receipt_snapshot.task_id,
            receipt_snapshot.operation_key,
            value,
            expected_revision=receipt_snapshot.revision,
        )
        return InvestigationRegenerationReceiptSnapshot(
            task_id=receipt_snapshot.task_id,
            operation_key=receipt_snapshot.operation_key,
            receipt=value,
            revision=revision,
        )

    @staticmethod
    def _initial_receipt_outcome(
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        generation_called: bool,
        working_memory_written: bool,
        investigation_committed: bool,
        reason: str | None,
        status: Literal["completed", "pending", "failed"],
        batch_event_id: str | None = None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """返回首批 receipt 的可恢复摘要；首批没有 InvestigationState。"""

        return HypothesisRegenerationBridgeOutcome(
            status=status,
            task_id=receipt_snapshot.task_id,
            previous_batch_id=str(
                receipt_snapshot.receipt.get("previous_batch_id") or ""
            ),
            next_batch_id=str(
                receipt_snapshot.receipt.get("next_batch_id") or ""
            ),
            receipt_status=str(receipt_snapshot.receipt.get("status") or "unknown"),
            generation_called=generation_called,
            working_memory_written=working_memory_written,
            investigation_committed=investigation_committed,
            revision=receipt_snapshot.revision,
            batch_event_id=batch_event_id
            or receipt_snapshot.receipt.get("batch_event_id"),
            reason=reason,
            state=None,
            cleanup_status="not_required",
        )

    @staticmethod
    def _assert_regeneration_gate(
        state: InvestigationState,
        next_batch_id: str,
        next_batch_index: int,
    ) -> None:
        """在任何外部副作用前确认 route、marker 和批次序号屏障。"""

        if not state.can_generate_next_batch:
            raise ValueError("五个组合必须全部保存 revise_hypothesis marker")
        if state.route != "regenerate_hypotheses":
            raise ValueError("只有 regenerate_hypotheses route 可以重生假设")
        if next_batch_id == state.batch_id:
            raise ValueError("下一批 batch_id 必须改变")
        if next_batch_id in {history.batch_id for history in state.history}:
            raise ValueError("下一批 batch_id 不能复用历史批次身份")
        expected_index = len(state.history) + 2
        if next_batch_index != expected_index:
            raise ValueError("next_batch_index 必须紧接当前 Investigation 批次")

    def _build_generation_request_from_state(
        self,
        state: InvestigationState,
        *,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        operation_key: str,
        operation_instance_id: str,
        generation_config: HypothesisRegenerationGenerationConfig,
    ) -> HypothesisRegenerationGenerationRequest:
        """按持久状态重新组装同一份重生请求，供 started receipt 严格恢复。"""

        working_memory = build_hypothesis_regeneration_working_memory_input(
            state,
            self.working_memory_reader,
        )
        batch_indexes = [
            item.batch_index for item in working_memory.closed_batches
        ] + [working_memory.current_batch.batch_index]
        if batch_indexes != list(range(1, len(batch_indexes) + 1)):
            raise ValueError("Working Memory 批次序号不连续，不能恢复 started receipt")
        if next_batch_index != working_memory.current_batch.batch_index + 1:
            raise ValueError("started receipt 的 next_batch_index 与历史不连续")
        working_memory_payload = working_memory.to_prompt_payload()
        legacy_package = _compose_legacy_regeneration_package(
            state,
            working_memory,
            self.working_memory_reader,
            generation_config,
        )
        return HypothesisRegenerationGenerationRequest(
            task_id=task_id,
            previous_batch_id=state.batch_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_key=operation_key,
            operation_instance_id=operation_instance_id,
            working_memory_input=working_memory,
            working_memory_payload=copy.deepcopy(working_memory_payload),
            regeneration_package=copy.deepcopy(legacy_package),
            generation_config=generation_config,
        )

    @staticmethod
    def _restore_previous_state_for_completed_receipt(
        state: InvestigationState,
        *,
        previous_batch_id: str,
    ) -> InvestigationState:
        """从已推进状态恢复 receipt 对应的前一批只读视图。

        ``completed`` receipt 的当前状态已经是下一批，不能直接用当前批次重建模型
        输入。这里只在 history 最后一项正好是 receipt 的 previous batch 时构造一个
        临时、重新校验的 InvestigationState；该对象不写回 store，也不改变正式状态。
        缺失或顺序漂移会 fail-closed，避免只凭 receipt 自己的 digest 接受被篡改的
        模型输入。
        """

        if not state.history or state.history[-1].batch_id != previous_batch_id:
            raise ValueError(
                "completed receipt 的 previous_batch_id 不是当前状态最后一个历史批次"
            )
        previous = state.history[-1]
        try:
            return InvestigationState(
                task_id=state.task_id,
                batch_id=previous.batch_id,
                sessions=previous.sessions,
                results=previous.results,
                validations=previous.validations,
                route=previous.route,
                final_answer=previous.final_answer,
                answerability_explanation=previous.answerability_explanation,
                history=state.history[:-1],
            )
        except Exception as exc:
            raise ValueError(
                "completed receipt 无法从历史批次恢复原始 InvestigationState"
            ) from exc

    def _create_started_receipt(
        self,
        snapshot: Any,
        *,
        request: HypothesisRegenerationGenerationRequest,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
        cleanup_enabled: bool = False,
    ) -> InvestigationRegenerationReceiptSnapshot | None:
        """保存生成前 receipt 和旧 scope 清理计划；CAS 竞争时不调用生成器。

        输入是当前 Investigation 快照、完整生成请求和下一批显式身份；动作把旧批
        次五个 scope 的 ``pending`` 项与主重生 receipt 一起保存。该动作不调用
        Working Memory、模型或 scope 服务；只有显式测试 bypass 才记录
        ``not_required``。receipt CAS 竞争返回 ``None``，其它保存失败向调用方抛出
        并不产生后续副作用。
        """

        model_request = _compose_canonical_model_request(request)
        operation_reference = _compose_operation_reference(
            request,
            identities=identities,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            model_request=model_request,
        )
        cleanup_plan = _build_scope_cleanup_plan(
            snapshot.state,
            next_batch_id=request.next_batch_id,
            enabled=cleanup_enabled,
        )
        receipt = {
            "schema": "arche.investigation.hypothesis_regeneration_receipt",
            "schema_version": 1,
            "status": "started",
            "task_id": request.task_id,
            "previous_batch_id": request.previous_batch_id,
            "next_batch_id": request.next_batch_id,
            "next_batch_index": request.next_batch_index,
            "operation_key": request.operation_key,
            "operation_instance_id": request.operation_instance_id,
            "input_digest": _canonical_digest(model_request),
            "model_request_digest": _canonical_digest(model_request),
            "operation_reference": operation_reference,
            "operation_reference_digest": _canonical_digest(operation_reference),
            "generation_config_digest": _generation_config_digest(
                request.generation_config
            ),
            "session_identities": _session_identities_payload(identities),
            "source_record_ids": list(source_record_ids),
            "causal_source_ref": causal_source_ref,
            "cleanup": cleanup_plan,
        }
        return self.investigation_store.claim_hypothesis_regeneration_receipt(
            request.task_id,
            request.operation_key,
            receipt,
        )

    def _save_initial_transport_recovery_plan(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        request: HypothesisInitialGenerationRequest,
        attempt: int,
        prior_operation_instance_id: str,
        prior_manifest_ref: Mapping[str, Any],
    ) -> InvestigationRegenerationReceiptSnapshot:
        """把传输失败清单和下一次 retry 身份写回同一 started receipt。"""

        plan = {
            "schema": _INITIAL_TRANSPORT_RECOVERY_SCHEMA,
            "schema_version": 1,
            "status": "pending",
            "attempt": attempt,
            "prior_operation_instance_id": prior_operation_instance_id,
            "retry_operation_instance_id": _initial_transport_retry_operation_id(
                request.operation_instance_id,
                attempt,
            ),
            "prior_manifest_ref": copy.deepcopy(dict(prior_manifest_ref)),
        }
        _validate_initial_transport_recovery_plan(plan, request=request)
        value = dict(receipt_snapshot.receipt)
        reference = dict(value["operation_reference"])
        reference["transport_recovery"] = plan
        value["operation_reference"] = reference
        value["operation_reference_digest"] = _canonical_digest(reference)
        revision = self.investigation_store.save_hypothesis_regeneration_receipt(
            receipt_snapshot.task_id,
            receipt_snapshot.operation_key,
            value,
            expected_revision=receipt_snapshot.revision,
        )
        return InvestigationRegenerationReceiptSnapshot(
            task_id=receipt_snapshot.task_id,
            operation_key=receipt_snapshot.operation_key,
            receipt=value,
            revision=revision,
        )

    def _save_generated_receipt(
        self,
        started: InvestigationRegenerationReceiptSnapshot,
        *,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
    ) -> InvestigationRegenerationReceiptSnapshot:
        """把 finalized manifest 来源和提交游标冻结到 receipt。

        生成结果正文只存在 Hypothesis operation 的 content-addressed finalized
        manifest 中。receipt 仅记录其相对来源和摘要，恢复时重新读取同一 manifest，
        因而不会在 Investigation SQLite 中形成第二份科学事实。
        """

        value = dict(started.receipt)
        reference = value.get("operation_reference")
        if isinstance(reference, Mapping):
            recovery = reference.get("transport_recovery")
            if isinstance(recovery, Mapping) and recovery.get("status") == "pending":
                updated_reference = dict(reference)
                updated_recovery = dict(recovery)
                updated_recovery["status"] = "completed"
                updated_reference["transport_recovery"] = updated_recovery
                value["operation_reference"] = updated_reference
                value["operation_reference_digest"] = _canonical_digest(
                    updated_reference
                )
        value.update(
            {
                "status": "generated",
                "source_refs": list(source_refs),
                "batch_event_id": None,
                "output_digest": _regeneration_output_digest(payload, source_refs),
            }
        )
        revision = self.investigation_store.save_hypothesis_regeneration_receipt(
            started.task_id,
            started.operation_key,
            value,
            expected_revision=started.revision,
        )
        return InvestigationRegenerationReceiptSnapshot(
            task_id=started.task_id,
            operation_key=started.operation_key,
            receipt=value,
            revision=revision,
        )

    def _mark_failed_receipt(
        self,
        started: InvestigationRegenerationReceiptSnapshot,
        error: BaseException,
    ) -> InvestigationRegenerationReceiptSnapshot:
        """冻结生成失败状态，确保后续恢复不隐式重调模型。"""

        value = dict(started.receipt)
        value.update({"status": "failed", "failure": _safe_failure_text(error)})
        try:
            revision = self.investigation_store.save_hypothesis_regeneration_receipt(
                started.task_id,
                started.operation_key,
                value,
                expected_revision=started.revision,
            )
        except InvestigationRevisionConflict:
            saved = self.investigation_store.load_hypothesis_regeneration_receipt(
                started.task_id,
                started.operation_key,
            )
            if saved is not None:
                return saved
            raise
        return InvestigationRegenerationReceiptSnapshot(
            task_id=started.task_id,
            operation_key=started.operation_key,
            receipt=value,
            revision=revision,
        )

    def _replay_finalized_generation_payload(
        self,
        *,
        receipt: Mapping[str, Any],
        state: InvestigationState,
        request: HypothesisRegenerationGenerationRequest,
        operation_instance_id: str,
        generation_config: HypothesisRegenerationGenerationConfig,
        next_batch_id: str,
        next_batch_index: int,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
    ) -> dict[str, Any]:
        """从同一 operation 的 finalized manifest 重建临时 BatchCreated payload。

        输入是 receipt、上一批只读状态和原始请求；动作只调用 generator 的
        finalized-manifest replay 接口，不调用模型 provider，也不写任何数据库。
        返回值仅在当前 turn 内存中存在；manifest 缺失、身份漂移或摘要失配都会
        fail-closed，防止把 receipt 当成第二份科学事实。
        """

        source_refs = receipt.get("source_refs")
        if not isinstance(source_refs, list) or len(source_refs) != 1:
            raise ValueError("重生 receipt.source_refs 必须恰含当前 manifest")
        _validate_hypothesis_manifest_source_ref(
            source_refs[0],
            task_root=generation_config.task_root,
            operation_instance_id=operation_instance_id,
        )
        recover = getattr(self.hypothesis_generator, "recover_hypothesis_regeneration", None)
        if not callable(recover):
            raise ValueError("重生恢复必须提供 finalized operation replay 接口")
        replayed = recover(request, copy.deepcopy(dict(receipt)))
        if replayed is None:
            raise ValueError("同一 Hypothesis operation 尚未生成 finalized manifest")
        if not isinstance(replayed, Mapping):
            raise ValueError("finalized operation replay 必须返回 object")
        hypothesis_result, operation_result = _read_generation_result(
            replayed,
            operation_instance_id=operation_instance_id,
            task_root=generation_config.task_root,
        )
        manifest_ref = operation_result["manifest_ref"]
        replay_source_refs = [str(manifest_ref["relative_path"])]
        if replay_source_refs != source_refs:
            raise ValueError("receipt.source_refs 与 finalized operation 不一致")
        payload = _compose_batch_payload(
            state=state,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_instance_id=operation_instance_id,
            hypothesis_result=hypothesis_result,
            generation_config=generation_config,
            expected_queries=_expected_queries_from_request(request),
            causal_source_ref=causal_source_ref,
            source_record_ids=source_record_ids,
            identities=identities,
        )
        if receipt.get("output_digest") != _regeneration_output_digest(
            payload,
            source_refs,
        ):
            raise ValueError("receipt 的 canonical BatchCreated output 摘要不一致")
        return payload

    def _validate_generated_receipt_before_working_memory_write(
        self,
        *,
        receipt: Mapping[str, Any],
        state: InvestigationState,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        operation_instance_id: str,
        generation_config: HypothesisRegenerationGenerationConfig,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
        expected_queries: Sequence[str],
        generation_request: HypothesisRegenerationGenerationRequest,
        verify_saved_operation_output: bool,
    ) -> dict[str, Any]:
        """在任何 WM writer 副作用前复核 generated receipt 的完整输出。

        输入是已保存的 ``generated`` receipt 和仍处于上一批的 InvestigationState；
        动作核对 receipt schema、BatchCreated canonical output digest、manifest 来源、
        批次/operation 身份和完整候选 payload。只要 receipt 被篡改、字段未知、来源
        越界或候选与冻结配置不一致，就在调用 Working Memory writer 前抛出异常。
        """

        del verify_saved_operation_output
        status = _validate_regeneration_receipt_schema(receipt)
        if status != "generated":
            raise ValueError("只有 generated receipt 可以进入 WM writer")
        if receipt.get("task_id") != task_id or receipt.get("next_batch_id") != next_batch_id:
            raise ValueError("generated receipt 的任务或下一批身份不一致")
        if receipt.get("next_batch_index") != next_batch_index:
            raise ValueError("generated receipt 的下一批序号不一致")
        if receipt.get("operation_instance_id") != operation_instance_id:
            raise ValueError("generated receipt 的 operation_instance_id 不一致")
        source_refs = receipt.get("source_refs")
        if not isinstance(source_refs, list) or len(source_refs) != 1:
            raise ValueError("generated receipt.source_refs 必须恰含当前 manifest")
        if receipt.get("batch_event_id") is not None:
            raise ValueError("generated receipt 不能提前携带 batch_event_id")
        expected_request = _compose_canonical_model_request(generation_request)
        _validate_saved_operation_reference(
            receipt,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_key=generation_request.operation_key,
            operation_instance_id=operation_instance_id,
            generation_config=generation_config,
            identities=identities,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            expected_model_request=expected_request,
        )
        payload = self._replay_finalized_generation_payload(
            receipt=receipt,
            state=state,
            request=generation_request,
            operation_instance_id=operation_instance_id,
            generation_config=generation_config,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            identities=identities,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
        )
        if receipt.get("output_digest") != _regeneration_output_digest(
            payload,
            source_refs,
        ):
            raise ValueError("generated receipt 的 canonical BatchCreated output 摘要不一致")
        return payload

    def _verify_committed_working_memory_batch(
        self,
        *,
        task_id: str,
        next_batch_id: str,
        receipt: Mapping[str, Any],
        payload: Mapping[str, Any],
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        creates_investigation_sessions: bool = True,
    ) -> None:
        """核对已提交的 BatchCreated 事实与 receipt 完全一致。

        输入是已经标记为 ``wm_written`` 的 receipt；动作只读取生产 Working Memory
        的完整 BatchCreated 正文，并核对 event 身份、payload 和来源引用。任何事实
        缺失或漂移都会停止恢复，不会把 Investigation 已推进误报为可完成。
        """
        _read_and_verify_committed_batch_receipt(
            self.working_memory_reader,
            task_id=task_id,
            next_batch_id=next_batch_id,
            receipt=receipt,
            payload=payload,
        )
        if creates_investigation_sessions:
            _verify_committed_session_identities(
                self.working_memory_reader,
                task_id=task_id,
                next_batch_id=next_batch_id,
                identities=identities,
            )
        else:
            list_sessions = getattr(
                self.working_memory_reader,
                "list_investigation_sessions",
                None,
            )
            if callable(list_sessions):
                sessions = list_sessions(task_id, next_batch_id, include_internal=True)
                if sessions:
                    raise ValueError(
                        "Top5 BatchCreated 不得产生 Investigation session"
                    )

    def _resume_saved_receipt(
        self,
        snapshot: Any,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        operation_instance_id: str,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        generation_config: HypothesisRegenerationGenerationConfig,
        operation_key: str,
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """按 receipt 游标恢复，不重调模型，也不推进第二份科学状态。"""

        receipt = receipt_snapshot.receipt
        status = _validate_regeneration_receipt_schema(receipt)
        resume_snapshot = self._snapshot_for_saved_receipt(
            snapshot,
            status=status,
            previous_batch_id=str(receipt.get("previous_batch_id") or ""),
            next_batch_id=next_batch_id,
        )
        self._validate_saved_receipt_identity(
            receipt,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_key=operation_key,
            operation_instance_id=operation_instance_id,
            identities=identities,
            generation_config=generation_config,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            state=resume_snapshot.state,
        )
        request = self._build_generation_request_from_state(
            resume_snapshot.state,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_key=operation_key,
            operation_instance_id=operation_instance_id,
            generation_config=generation_config,
        )
        _validate_saved_operation_reference(
            receipt,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_key=operation_key,
            operation_instance_id=operation_instance_id,
            generation_config=generation_config,
            identities=identities,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            expected_model_request=_compose_canonical_model_request(request),
        )
        if status == "failed":
            return self._receipt_outcome(
                receipt_snapshot,
                state=resume_snapshot.state,
                generation_called=False,
                working_memory_written=False,
                investigation_committed=False,
                reason=str(receipt.get("failure") or "生成失败，receipt 已冻结"),
                status="failed",
            )
        if status == "started":
            return self._recover_started_receipt(
                resume_snapshot,
                receipt_snapshot,
                request=request,
                operation_instance_id=operation_instance_id,
                generation_config=generation_config,
                task_id=task_id,
                next_batch_id=next_batch_id,
                next_batch_index=next_batch_index,
                identities=identities,
                operation_key=operation_key,
                source_record_ids=source_record_ids,
                causal_source_ref=causal_source_ref,
            )
        return self._write_working_memory_and_begin_next_batch(
            resume_snapshot,
            receipt_snapshot,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_instance_id=operation_instance_id,
            identities=identities,
            generation_config=generation_config,
            operation_key=operation_key,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            expected_queries=_expected_queries_from_request(request),
            generation_request=request,
            verify_saved_operation_output=False,
            generation_called=False,
        )

    def _snapshot_for_saved_receipt(
        self,
        snapshot: Any,
        *,
        status: str,
        previous_batch_id: str,
        next_batch_id: str,
    ) -> Any:
        """为已提交游标恢复生成时使用的前一批只读快照。

        输入是持久化快照和 receipt 阶段；动作只在本地状态已经被其它迁移路径推进
        到下一批时，从历史构造临时前一批视图，不写回任何存储。生成中或当前状态仍
        是前一批时直接返回原快照；缺少可核对的历史时抛出错误，避免用下一批事实
        重建旧 operation。
        """

        # ``generated`` 也可能已经越过了 WM writer：若写入成功后保存
        # ``wm_written`` 游标崩溃，WM 当前批次已是 next_batch_id，但 receipt 仍是
        # generated。此时必须从 history 恢复上一批只读视图，按同一 payload 做幂等
        # 事件确认，不能拿下一批重建旧 operation。
        if status not in {"generated", "wm_written", "completed"}:
            return snapshot
        if snapshot.state.batch_id != next_batch_id:
            return snapshot
        previous_state = self._restore_previous_state_for_completed_receipt(
            snapshot.state,
            previous_batch_id=previous_batch_id,
        )
        return replace(snapshot, state=previous_state)

    def _validate_saved_receipt_identity(
        self,
        receipt: Mapping[str, Any],
        *,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        operation_key: str,
        operation_instance_id: str,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        generation_config: HypothesisRegenerationGenerationConfig,
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
        state: InvestigationState,
    ) -> None:
        """核对 receipt 身份、配置、历史批次和 finalized manifest 游标。"""

        expected = {
            "task_id": task_id,
            "next_batch_id": next_batch_id,
            "next_batch_index": next_batch_index,
            "operation_key": operation_key,
            "operation_instance_id": operation_instance_id,
            "session_identities": _session_identities_payload(identities),
            "source_record_ids": list(source_record_ids),
            "causal_source_ref": causal_source_ref,
            "generation_config_digest": _generation_config_digest(generation_config),
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                raise ValueError(f"已有重生 receipt.{field} 与本次请求不一致")
        previous_batch_id = receipt.get("previous_batch_id")
        if previous_batch_id not in {state.batch_id, *(item.batch_id for item in state.history)}:
            raise ValueError("重生 receipt.previous_batch_id 与当前 Working Memory 状态不一致")
        if receipt.get("cleanup") is not None:
            _validate_scope_cleanup_plan(
                receipt["cleanup"],
                task_id=task_id,
                previous_batch_id=str(previous_batch_id),
                next_batch_id=next_batch_id,
            )
        if receipt.get("status") in {"generated", "wm_written", "completed"}:
            source_refs = receipt.get("source_refs")
            if not isinstance(source_refs, list) or len(source_refs) != 1:
                raise ValueError("重生 receipt.source_refs 必须恰含当前 manifest")
            _validate_hypothesis_manifest_source_ref(
                source_refs[0],
                task_root=generation_config.task_root,
                operation_instance_id=operation_instance_id,
            )

    def _recover_started_receipt(
        self,
        snapshot: Any,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        request: HypothesisRegenerationGenerationRequest,
        operation_instance_id: str,
        generation_config: HypothesisRegenerationGenerationConfig,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        operation_key: str,
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """仅从同一 operation finalized manifest 恢复 started receipt。"""

        _validate_saved_operation_reference(
            receipt_snapshot.receipt,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_key=operation_key,
            operation_instance_id=operation_instance_id,
            generation_config=generation_config,
            identities=identities,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            expected_model_request=_compose_canonical_model_request(request),
        )
        recover = getattr(self.hypothesis_generator, "recover_hypothesis_regeneration", None)
        if not callable(recover):
            return self._receipt_outcome(
                receipt_snapshot,
                state=snapshot.state,
                generation_called=False,
                working_memory_written=False,
                investigation_committed=False,
                reason="started receipt 尚无 finalized manifest；保持 pending",
                status="pending",
            )
        try:
            recovered = recover(request, copy.deepcopy(dict(receipt_snapshot.receipt)))
            if recovered is None:
                return self._receipt_outcome(
                    receipt_snapshot,
                    state=snapshot.state,
                    generation_called=False,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason="同一 operation 尚未发现 finalized manifest；保持 pending",
                    status="pending",
                )
            hypothesis_result, operation = _read_generation_result(
                recovered,
                operation_instance_id=operation_instance_id,
                task_root=generation_config.task_root,
            )
            manifest_ref = operation["manifest_ref"]
            replayed_payload = _compose_batch_payload(
                state=snapshot.state,
                next_batch_id=next_batch_id,
                next_batch_index=next_batch_index,
                operation_instance_id=operation_instance_id,
                hypothesis_result=hypothesis_result,
                generation_config=generation_config,
                expected_queries=_expected_queries_from_request(request),
                causal_source_ref=causal_source_ref,
                source_record_ids=source_record_ids,
                identities=identities,
            )
            generated_receipt = self._save_generated_receipt(
                receipt_snapshot,
                payload=replayed_payload,
                source_refs=[str(manifest_ref["relative_path"])],
            )
        except BaseException as exc:
            if _is_busy_operation_error(exc):
                return self._receipt_outcome(
                    receipt_snapshot,
                    state=snapshot.state,
                    generation_called=False,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason="同一 Hypothesis operation 仍在运行；保持 started pending",
                    status="pending",
                )
            failed = self._mark_failed_receipt(receipt_snapshot, exc)
            raise HypothesisRegenerationBridgeError(
                "started receipt finalized replay 失败；receipt 已冻结",
                outcome=self._receipt_outcome(
                    failed,
                    state=snapshot.state,
                    generation_called=False,
                    working_memory_written=False,
                    investigation_committed=False,
                    reason=_safe_failure_text(exc),
                    status="failed",
                ),
            ) from exc
        return self._write_working_memory_and_begin_next_batch(
            snapshot,
            generated_receipt,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            operation_instance_id=operation_instance_id,
            identities=identities,
            generation_config=generation_config,
            operation_key=operation_key,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            expected_queries=_expected_queries_from_request(request),
            generation_request=request,
            verify_saved_operation_output=False,
            generation_called=False,
            replayed_payload=replayed_payload,
        )

    def _write_working_memory_and_begin_next_batch(
        self,
        snapshot: Any,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        operation_instance_id: str,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        generation_config: HypothesisRegenerationGenerationConfig,
        operation_key: str,
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
        expected_queries: Sequence[str],
        generation_request: HypothesisRegenerationGenerationRequest,
        verify_saved_operation_output: bool,
        generation_called: bool,
        replayed_payload: Mapping[str, Any] | None = None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """完成 receipt 中尚未确认的唯一 Working Memory 原子提交。

        输入是已声明的生成 receipt 和重放请求；动作从 finalized manifest 得到临时
        payload，调用一次 ``begin_next_investigation_batch``，确认 BatchCreated 与五
        个 session 后仅推进 receipt 游标。这里绝不保存或推进第二份 Investigation
        状态；任一失败都保留可恢复的 ``generated``/``wm_written`` receipt。
        """

        del operation_key, expected_queries
        receipt = receipt_snapshot.receipt
        status = receipt.get("status")
        payload = self._payload_for_receipt_status(
            receipt=receipt,
            status=status,
            state=snapshot.state,
            generation_request=generation_request,
            operation_instance_id=operation_instance_id,
            generation_config=generation_config,
            task_id=task_id,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            identities=identities,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
            verify_saved_operation_output=verify_saved_operation_output,
            replayed_payload=replayed_payload,
        )
        source_refs = receipt.get("source_refs")
        if not isinstance(source_refs, list):
            raise ValueError("重生 receipt 缺少 finalized manifest source_refs")
        batch_event_id = receipt.get("batch_event_id")
        if status == "generated":
            batch_event_id = self._submit_next_batch_to_working_memory(
                receipt_snapshot,
                task_id=task_id,
                payload=payload,
                source_refs=source_refs,
                state=snapshot.state,
                generation_called=generation_called,
            )
            receipt_snapshot = self._save_wm_commit_cursor(
                receipt_snapshot,
                batch_event_id=batch_event_id,
                state=snapshot.state,
                generation_called=generation_called,
            )
            receipt = receipt_snapshot.receipt
            status = "wm_written"
        if status not in {"wm_written", "completed"}:
            raise ValueError("WM 提交后 receipt 状态非法")
        self._verify_committed_working_memory_batch(
            task_id=task_id,
            next_batch_id=next_batch_id,
            receipt=receipt,
            payload=payload,
            identities=identities,
            creates_investigation_sessions=generation_config.creates_investigation_sessions,
        )
        if status == "wm_written":
            receipt_snapshot = self._save_completed_receipt(
                receipt_snapshot,
                batch_event_id=_require_text(batch_event_id, "BatchCreated event_id"),
                state=snapshot.state,
            )
        return self._finish_committed_regeneration(
            receipt_snapshot,
            state=snapshot.state,
            generation_called=generation_called,
            working_memory_written=True,
            investigation_committed=True,
            revision=receipt_snapshot.revision,
            batch_event_id=_require_text(
                receipt_snapshot.receipt.get("batch_event_id"),
                "BatchCreated event_id",
            ),
        )

    def _payload_for_receipt_status(
        self,
        *,
        receipt: Mapping[str, Any],
        status: Any,
        state: InvestigationState,
        generation_request: HypothesisRegenerationGenerationRequest,
        operation_instance_id: str,
        generation_config: HypothesisRegenerationGenerationConfig,
        task_id: str,
        next_batch_id: str,
        next_batch_index: int,
        identities: Sequence[HypothesisRegenerationSessionIdentity],
        source_record_ids: Sequence[str],
        causal_source_ref: str | None,
        verify_saved_operation_output: bool,
        replayed_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """按 receipt 阶段恢复临时 BatchCreated payload。"""

        if status not in {"generated", "wm_written", "completed"}:
            raise ValueError("当前 receipt 没有可提交的 finalized manifest")
        if status == "generated":
            if replayed_payload is not None:
                payload = _copy_json(
                    dict(replayed_payload),
                    "replayed BatchCreated payload",
                )
                source_refs = receipt.get("source_refs")
                if not isinstance(source_refs, list) or receipt.get(
                    "output_digest"
                ) != _regeneration_output_digest(payload, source_refs):
                    raise ValueError(
                        "generated receipt 的 canonical BatchCreated output 摘要不一致"
                    )
                return payload
            return self._validate_generated_receipt_before_working_memory_write(
                receipt=receipt,
                state=state,
                task_id=task_id,
                next_batch_id=next_batch_id,
                next_batch_index=next_batch_index,
                operation_instance_id=operation_instance_id,
                generation_config=generation_config,
                identities=identities,
                source_record_ids=source_record_ids,
                causal_source_ref=causal_source_ref,
                expected_queries=_expected_queries_from_request(generation_request),
                generation_request=generation_request,
                verify_saved_operation_output=verify_saved_operation_output,
            )
        return self._replay_finalized_generation_payload(
            receipt=receipt,
            state=state,
            request=generation_request,
            operation_instance_id=operation_instance_id,
            generation_config=generation_config,
            next_batch_id=next_batch_id,
            next_batch_index=next_batch_index,
            identities=identities,
            source_record_ids=source_record_ids,
            causal_source_ref=causal_source_ref,
        )

    def _submit_next_batch_to_working_memory(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        task_id: str,
        payload: Mapping[str, Any],
        source_refs: Sequence[str],
        state: InvestigationState,
        generation_called: bool,
    ) -> str:
        """调用唯一 Working Memory writer，并把失败转换为可恢复 pending。"""

        try:
            event_id = self.working_memory_writer.begin_next_investigation_batch(
                task_id,
                copy.deepcopy(dict(payload)),
                tuple(source_refs),
            )
            return _require_text(event_id, "BatchCreated event_id")
        except BaseException as exc:
            outcome = self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=False,
                investigation_committed=False,
                reason=_safe_failure_text(exc),
                status="pending",
            )
            raise HypothesisRegenerationBridgeError(
                "Working Memory 下一批原子写入失败；下次恢复只重试幂等提交",
                outcome=outcome,
            ) from exc

    def _save_wm_commit_cursor(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        batch_event_id: str,
        state: InvestigationState,
        generation_called: bool,
    ) -> InvestigationRegenerationReceiptSnapshot:
        """保存已确认的 BatchCreated event 游标，不保存科学正文。"""

        value = dict(receipt_snapshot.receipt)
        value.update({"status": "wm_written", "batch_event_id": batch_event_id})
        try:
            revision = self.investigation_store.save_hypothesis_regeneration_receipt(
                receipt_snapshot.task_id,
                receipt_snapshot.operation_key,
                value,
                expected_revision=receipt_snapshot.revision,
            )
        except InvestigationRevisionConflict:
            competing = self.investigation_store.load_hypothesis_regeneration_receipt(
                receipt_snapshot.task_id,
                receipt_snapshot.operation_key,
            )
            if competing is None:
                raise
            return competing
        except BaseException as exc:
            outcome = self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=False,
                reason="BatchCreated 已提交但 wm_written 游标暂未保存；下次恢复可幂等确认",
                status="pending",
                batch_event_id=batch_event_id,
            )
            raise HypothesisRegenerationBridgeError(
                "wm_written 游标写入失败；下次恢复只确认同一 BatchCreated",
                outcome=outcome,
            ) from exc
        return InvestigationRegenerationReceiptSnapshot(
            task_id=receipt_snapshot.task_id,
            operation_key=receipt_snapshot.operation_key,
            receipt=value,
            revision=revision,
        )

    def _save_completed_receipt(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        batch_event_id: str,
        state: InvestigationState,
    ) -> InvestigationRegenerationReceiptSnapshot:
        """保存完成游标，保留 finalized manifest 引用和清理进度。"""

        value = dict(receipt_snapshot.receipt)
        value.update({"status": "completed", "batch_event_id": batch_event_id})
        try:
            revision = self.investigation_store.save_hypothesis_regeneration_receipt(
                receipt_snapshot.task_id,
                receipt_snapshot.operation_key,
                value,
                expected_revision=receipt_snapshot.revision,
            )
        except InvestigationRevisionConflict:
            competing = self.investigation_store.load_hypothesis_regeneration_receipt(
                receipt_snapshot.task_id,
                receipt_snapshot.operation_key,
            )
            if competing is not None and competing.receipt.get("status") == "completed":
                return competing
            raise
        except BaseException as exc:
            raise HypothesisRegenerationBridgeError(
                "Working Memory 已提交但 completed 游标暂未保存；下次恢复可幂等确认",
                outcome=self._receipt_outcome(
                    receipt_snapshot,
                    state=state,
                    generation_called=False,
                    working_memory_written=True,
                    investigation_committed=True,
                    reason="completed 游标暂未保存",
                    status="pending",
                    batch_event_id=batch_event_id,
                ),
            ) from exc
        return InvestigationRegenerationReceiptSnapshot(
            task_id=receipt_snapshot.task_id,
            operation_key=receipt_snapshot.operation_key,
            receipt=value,
            revision=revision,
        )

    def _find_cleanup_source_sessions(
        self,
        state: InvestigationState,
        *,
        source_batch_id: str,
    ) -> tuple[InvestigationSession, ...]:
        """从当前或已关闭历史中读取 receipt 指定的旧批次五个会话。"""

        if state.batch_id == source_batch_id:
            return tuple(state.sessions)
        matches = tuple(
            history
            for history in state.history
            if history.batch_id == source_batch_id
        )
        if len(matches) != 1:
            raise ValueError("scope cleanup plan 缺少唯一的旧批次会话历史")
        return tuple(matches[0].sessions)

    def _validate_cleanup_plan_against_state(
        self,
        state: InvestigationState,
        plan: Mapping[str, Any],
    ) -> None:
        """核对 receipt 计划仍对应 InvestigationState 中的旧五个 scope。"""

        source_batch_id = plan["source_batch_id"]
        sessions = self._find_cleanup_source_sessions(
            state,
            source_batch_id=source_batch_id,
        )
        expected = tuple(
            (
                _require_identity(session.combination_id, "旧 scope combination_id"),
                _require_identity(session.scope_id, "旧 scope scope_id"),
                _cleanup_workspace_text(session.workspace_ref, "旧 scope workspace_ref"),
            )
            for session in sessions
        )
        actual = tuple(
            (
                item["combination_id"],
                item["scope_id"],
                item["workspace_ref"],
            )
            for item in plan["scopes"]
        )
        if set(actual) != set(expected) or len(actual) != 5:
            raise ValueError("scope cleanup plan 与旧批次五个会话身份不一致")

    def _save_cleanup_plan(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        plan: Mapping[str, Any],
    ) -> InvestigationRegenerationReceiptSnapshot:
        """按同一 operation receipt 保存一份逐 scope 清理进度。"""

        receipt = dict(receipt_snapshot.receipt)
        previous_batch_id = str(receipt.get("previous_batch_id") or "")
        next_batch_id = str(receipt.get("next_batch_id") or "")
        normalized = _validate_scope_cleanup_plan(
            plan,
            task_id=receipt_snapshot.task_id,
            previous_batch_id=previous_batch_id,
            next_batch_id=next_batch_id,
        )
        receipt["cleanup"] = normalized
        revision = self.investigation_store.save_hypothesis_regeneration_receipt(
            receipt_snapshot.task_id,
            receipt_snapshot.operation_key,
            receipt,
            expected_revision=receipt_snapshot.revision,
        )
        return InvestigationRegenerationReceiptSnapshot(
            task_id=receipt_snapshot.task_id,
            operation_key=receipt_snapshot.operation_key,
            receipt=receipt,
            revision=revision,
        )

    def _prepare_scope_cleanup_plan(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        state: InvestigationState,
    ) -> tuple[
        InvestigationRegenerationReceiptSnapshot,
        dict[str, Any] | None,
    ]:
        """读取或补齐同一 receipt 的 scope 清理计划，不调用撤销服务。

        输入必须是主重生已经完成的 receipt 和当前 InvestigationState；动作只在
        receipt 缺少计划或原先未注入凭据、现在明确注入凭据时写入 ``pending`` 计划。
        返回更新后的 receipt 与规范计划；CAS 或历史身份失败会抛出，调用方应保持
        清理未完成并让恢复继续处理。
        """

        receipt = receipt_snapshot.receipt
        previous_batch_id = receipt.get("previous_batch_id")
        next_batch_id = receipt.get("next_batch_id")
        if not isinstance(previous_batch_id, str) or not isinstance(next_batch_id, str):
            raise ValueError("receipt 缺少 scope cleanup 的 source/target batch")
        raw_plan = receipt.get("cleanup")
        if raw_plan is None:
            if self.scope_credentials is None:
                return receipt_snapshot, None
            sessions = self._find_cleanup_source_sessions(
                state,
                source_batch_id=previous_batch_id,
            )
            plan = _build_scope_cleanup_plan_from_sessions(
                sessions,
                source_batch_id=previous_batch_id,
                next_batch_id=next_batch_id,
                enabled=True,
            )
            return self._save_cleanup_plan(receipt_snapshot, plan), plan
        plan = _validate_scope_cleanup_plan(
            raw_plan,
            task_id=receipt_snapshot.task_id,
            previous_batch_id=previous_batch_id,
            next_batch_id=next_batch_id,
        )
        if self.scope_credentials is None or plan["status"] != "not_required":
            return receipt_snapshot, plan
        plan["status"] = "cleanup_pending"
        plan.pop("last_error", None)
        updated = self._save_cleanup_plan(receipt_snapshot, plan)
        return updated, plan

    def _finish_committed_regeneration(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        state: InvestigationState,
        generation_called: bool,
        working_memory_written: bool,
        investigation_committed: bool,
        revision: int | None,
        batch_event_id: str | None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """确认两个主存储都已提交后，执行或恢复旧 scope 清理。"""

        if not working_memory_written or not investigation_committed:
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=working_memory_written,
                investigation_committed=investigation_committed,
                reason="主重生尚未同时提交，禁止撤销旧 scope",
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
            )
        if receipt_snapshot.receipt.get("status") != "completed":
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason="主重生 receipt 尚未保存 completed，保留清理待办",
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
            )
        try:
            event_id = _require_text(
                batch_event_id or receipt_snapshot.receipt.get("batch_event_id"),
                "BatchCreated event_id",
            )
        except BaseException as exc:
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason=_safe_failure_text(exc),
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
            )
        return self._run_scope_cleanup(
            receipt_snapshot,
            state=state,
            generation_called=generation_called,
            revision=revision,
            batch_event_id=event_id,
        )

    def _run_scope_cleanup(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        state: InvestigationState,
        generation_called: bool,
        revision: int | None,
        batch_event_id: str,
    ) -> HypothesisRegenerationBridgeOutcome:
        """在主重生完成后逐项撤销旧 scope，并持久化每项结果。

        输入是 ``completed`` 主 receipt、已经写入 WM 的 event_id 和已提交的下一批
        state；动作只处理 receipt 中仍为 ``pending`` 的旧 scope，并在每次成功撤销
        后先保存该项 ``revoked`` 再处理下一项。撤销异常、返回未知值或进度保存失败
        都返回 ``cleanup_pending``，不把未确认项标成完成；下次恢复只重试仍为
        ``pending`` 的项。该方法不声称两个服务共享原子事务。
        """

        try:
            receipt_snapshot, plan = self._prepare_scope_cleanup_plan(
                receipt_snapshot,
                state,
            )
        except BaseException as exc:
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason=f"scope cleanup plan 暂不可恢复: {_safe_failure_text(exc)}",
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
                cleanup_status="cleanup_pending",
            )
        if plan is None:
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason="显式测试 scope cleanup bypass，未执行旧 scope 清理",
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
                cleanup_status="not_required",
            )
        try:
            self._validate_cleanup_plan_against_state(state, plan)
        except BaseException as exc:
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason=f"scope cleanup plan 与持久状态不一致: {_safe_failure_text(exc)}",
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
                cleanup_status="cleanup_pending",
            )
        if plan["status"] == "completed":
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason=None,
                status="completed",
                revision=revision,
                batch_event_id=batch_event_id,
                cleanup_status="completed",
            )
        if plan["status"] == "not_required":
            if self.scope_credentials is None:
                return self._receipt_outcome(
                    receipt_snapshot,
                    state=state,
                    generation_called=generation_called,
                    working_memory_written=True,
                    investigation_committed=True,
                    reason="显式测试 scope cleanup bypass，未执行旧 scope 清理",
                    status="pending",
                    revision=revision,
                    batch_event_id=batch_event_id,
                    cleanup_status="not_required",
                )
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason=None,
                status="completed",
                revision=revision,
                batch_event_id=batch_event_id,
                cleanup_status="not_required",
            )
        if self.scope_credentials is None:
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason="旧 scope 清理依赖未注入，保留 cleanup_pending",
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
                cleanup_status="cleanup_pending",
            )
        for index, item in enumerate(plan["scopes"]):
            if item["status"] == "revoked":
                continue
            receipt_snapshot, plan, failure_reason = self._process_scope_cleanup_item(
                receipt_snapshot,
                plan,
                index=index,
            )
            if failure_reason is not None:
                return self._receipt_outcome(
                    receipt_snapshot,
                    state=state,
                    generation_called=generation_called,
                    working_memory_written=True,
                    investigation_committed=True,
                    reason=failure_reason,
                    status="pending",
                    revision=revision,
                    batch_event_id=batch_event_id,
                    cleanup_status="cleanup_pending",
                )
        completed_plan = copy.deepcopy(plan)
        completed_plan["status"] = "completed"
        completed_plan.pop("last_error", None)
        try:
            receipt_snapshot = self._save_cleanup_plan(
                receipt_snapshot,
                completed_plan,
            )
        except BaseException as exc:
            return self._receipt_outcome(
                receipt_snapshot,
                state=state,
                generation_called=generation_called,
                working_memory_written=True,
                investigation_committed=True,
                reason=f"scope cleanup completed 标记未保存: {_safe_failure_text(exc)}",
                status="pending",
                revision=revision,
                batch_event_id=batch_event_id,
                cleanup_status="cleanup_pending",
            )
        return self._receipt_outcome(
            receipt_snapshot,
            state=state,
            generation_called=generation_called,
            working_memory_written=True,
            investigation_committed=True,
            reason=None,
            status="completed",
            revision=revision,
            batch_event_id=batch_event_id,
            cleanup_status="completed",
        )

    def _process_scope_cleanup_item(
        self,
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        plan: dict[str, Any],
        *,
        index: int,
    ) -> tuple[
        InvestigationRegenerationReceiptSnapshot,
        dict[str, Any],
        str | None,
    ]:
        """撤销一个旧 scope 并保存该项进度。

        输入是已核对的清理计划和一个待处理索引；动作调用一次撤销 capability，随后
        保存 ``revoked`` 或失败信息。返回值包含最新 receipt、规范计划和失败说明；
        失败说明非空时，调用方必须保留 ``cleanup_pending``，不能把远端不确定结果
        当成已完成。这里不处理最终 completed 标记，避免跳过最后一次进度保存。
        """

        item = plan["scopes"][index]
        try:
            revoke_result = self.scope_credentials.revoke_investigation_scope_capability(
                task_id=receipt_snapshot.task_id,
                batch_id=plan["source_batch_id"],
                combination_id=item["combination_id"],
                scope_id=item["scope_id"],
                workspace=item["workspace_ref"],
            )
            if revoke_result is not None:
                raise RuntimeError("scope revoke 返回未注册结果，无法确认撤销")
        except BaseException as exc:
            failed_plan = copy.deepcopy(plan)
            failed_plan["status"] = "cleanup_pending"
            failed_plan["last_error"] = _safe_failure_text(exc)
            try:
                receipt_snapshot = self._save_cleanup_plan(
                    receipt_snapshot,
                    failed_plan,
                )
                plan = _validate_scope_cleanup_plan(
                    receipt_snapshot.receipt.get("cleanup"),
                    task_id=receipt_snapshot.task_id,
                    previous_batch_id=str(
                        receipt_snapshot.receipt.get("previous_batch_id") or ""
                    ),
                    next_batch_id=str(
                        receipt_snapshot.receipt.get("next_batch_id") or ""
                    ),
                )
            except BaseException:
                # 远端结果不确定或本地保存失败时，保留旧 receipt 的 pending 状态。
                pass
            return (
                receipt_snapshot,
                plan,
                f"旧 scope[{index}] 撤销未确认: {_safe_failure_text(exc)}",
            )
        updated_plan = copy.deepcopy(plan)
        updated_plan["status"] = "cleanup_pending"
        updated_plan["scopes"][index]["status"] = "revoked"
        updated_plan.pop("last_error", None)
        try:
            receipt_snapshot = self._save_cleanup_plan(
                receipt_snapshot,
                updated_plan,
            )
            plan = _validate_scope_cleanup_plan(
                receipt_snapshot.receipt.get("cleanup"),
                task_id=receipt_snapshot.task_id,
                previous_batch_id=str(
                    receipt_snapshot.receipt.get("previous_batch_id") or ""
                ),
                next_batch_id=str(
                    receipt_snapshot.receipt.get("next_batch_id") or ""
                ),
            )
        except BaseException as exc:
            return (
                receipt_snapshot,
                plan,
                f"旧 scope[{index}] 已调用但 revoked 记录未保存: {_safe_failure_text(exc)}",
            )
        return receipt_snapshot, plan, None

    @staticmethod
    def _receipt_outcome(
        receipt_snapshot: InvestigationRegenerationReceiptSnapshot,
        *,
        state: InvestigationState,
        generation_called: bool,
        working_memory_written: bool,
        investigation_committed: bool,
        reason: str | None,
        status: Literal["completed", "pending", "failed"] | None = None,
        revision: int | None = None,
        batch_event_id: str | None = None,
        cleanup_status: Literal["not_required", "cleanup_pending", "completed"] | None = None,
    ) -> HypothesisRegenerationBridgeOutcome:
        """把 receipt 和实际状态汇总为调用方可审计结果。"""

        receipt_status = str(receipt_snapshot.receipt.get("status") or "unknown")
        receipt_cleanup_status, cleanup_scope_statuses = _read_receipt_cleanup_summary(
            receipt_snapshot.receipt,
        )
        return HypothesisRegenerationBridgeOutcome(
            status=status or ("completed" if investigation_committed else "pending"),
            task_id=receipt_snapshot.task_id,
            previous_batch_id=str(receipt_snapshot.receipt.get("previous_batch_id") or ""),
            next_batch_id=str(receipt_snapshot.receipt.get("next_batch_id") or ""),
            receipt_status=receipt_status,
            generation_called=generation_called,
            working_memory_written=working_memory_written,
            investigation_committed=investigation_committed,
            revision=revision,
            batch_event_id=batch_event_id or receipt_snapshot.receipt.get("batch_event_id"),
            reason=reason,
            state=state,
            cleanup_status=cleanup_status or receipt_cleanup_status,
            cleanup_scope_statuses=cleanup_scope_statuses,
        )
