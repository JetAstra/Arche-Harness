"""Investigation Working Memory 的假设重生输入整理。

本模块把已经进入 ``regenerate_hypotheses`` 终局的 :class:`InvestigationState`
整理为下一批假设模型可以读取的严格 JSON object。输出只保留科学结果、完整批次
验证和有限的路由事实；工作区、MiniChem scope、Codex thread/transcript 以及宿主
路径和凭据都不会进入输出。

领域模型主要保证当前快照的内部引用关系，历史 ``BatchHistory`` 还需要由这里
再次核对。任何非终局、跨任务/批次、缺少结果版本或缺少完整验证向量的输入都会
失败，不返回部分记忆。函数不写入状态，也不修改调用方对象。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
)

from .contracts import _assert_no_path_or_secret
from .domain import (
    BatchHistory,
    BatchValidation,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
)

__all__ = [
    "REGENERATION_MEMORY_SCHEMA",
    "REGENERATION_MEMORY_VERSION",
    "build_hypothesis_regeneration_memory",
]


REGENERATION_MEMORY_SCHEMA = "arche.investigation.regeneration_memory"
REGENERATION_MEMORY_VERSION = 1

_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s])/(?!/)[^,;\s)\]}]*")
_PRIVATE_TEXT_RE = re.compile(
    r"\b(?:workspace_ref|scope_id|codex_thread_id|review_thread_id|"
    r"review_turn_id|transcript)\b",
    re.IGNORECASE,
)


class _FrozenProjection(BaseModel):
    """内部模型使用严格字段、冻结和禁止额外字段。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class _ResultProjection(_FrozenProjection):
    """供下一批模型读取的一条安全 InvestigationResult。"""

    combination_id: StrictStr
    result_version: StrictStr
    self_judgment: Literal["supported", "refuted"]
    narrative: StrictStr
    job_refs: tuple[StrictStr, ...]
    operation_refs: tuple[StrictStr, ...]
    artifact_refs: tuple[StrictStr, ...]
    observations: tuple[StrictStr, ...]
    limitations: tuple[StrictStr, ...]


class _FindingProjection(_FrozenProjection):
    """一条批次验证 finding；无关分支字段保留为 JSON null。"""

    combination_id: StrictStr
    validity: Literal["pass", "fail"]
    hypothesis_judgment: Literal["supported", "refuted"] | None
    repair_advice: StrictStr | None


class _ValidationProjection(_FrozenProjection):
    """一轮完整 BatchValidation 的脱敏副本。"""

    combination_ids: tuple[StrictStr, ...]
    result_versions: tuple[StrictStr, ...]
    findings: tuple[_FindingProjection, ...]
    can_answer: StrictBool
    final_answer: StrictStr | None
    answerability_explanation: StrictStr | None


class _SessionProjection(_FrozenProjection):
    """每个组合保留当前结果版本、返修次数和原始路由 marker。"""

    combination_id: StrictStr
    result_version: StrictStr
    repair_count: StrictInt
    route_marker: Literal["revise_hypothesis"] | None


class _BatchProjection(_FrozenProjection):
    """一个已关闭或当前批次的工作记忆事实集合。"""

    task_id: StrictStr
    batch_id: StrictStr
    route: Literal[
        "investigation",
        "repair_required",
        "accept",
        "answerability_explanation",
        "regenerate_hypotheses",
    ]
    final_answer: StrictStr | None
    answerability_explanation: StrictStr | None
    sessions: tuple[_SessionProjection, ...]
    results: tuple[_ResultProjection, ...]
    validations: tuple[_ValidationProjection, ...]


class _RegenerationMemoryProjection(_FrozenProjection):
    """最终严格 JSON working-memory object。"""

    schema_name: StrictStr = Field(alias="schema")
    schema_version: StrictInt
    task_id: StrictStr
    batch_id: StrictStr
    closed_batches: tuple[_BatchProjection, ...]
    current_batch: _BatchProjection


def _raise_corrupt(message: str, cause: BaseException | None = None) -> None:
    """将领域快照问题统一报告为不可恢复的输入整理错误。"""

    if cause is None:
        raise ValueError(message)
    raise ValueError(message) from cause


def _revalidate_state(state: InvestigationState) -> InvestigationState:
    """复制并重新校验状态，拒绝未重新验证的 ``model_copy(update=...)``。"""

    if not isinstance(state, InvestigationState):
        raise TypeError("state 必须是 InvestigationState")
    try:
        return InvestigationState.model_validate(state.model_dump(exclude_none=True))
    except (TypeError, ValueError, ValidationError) as exc:
        _raise_corrupt("InvestigationState 不符合领域合同，不能生成重生记忆", exc)
    raise AssertionError("不可达")  # pragma: no cover


def _result_key(result: InvestigationResult) -> tuple[str, str]:
    """返回组合和版本组成的稳定结果键。"""

    return result.combination_id, result.result_version


def _validate_validation_reference(
    validation: BatchValidation,
    *,
    task_id: str,
    batch_id: str,
    combination_ids: set[str],
    result_keys: set[tuple[str, str]],
    label: str,
) -> None:
    """核对一轮验证的任务、五组合覆盖和每个结果版本引用。"""

    if validation.task_id != task_id or validation.batch_id != batch_id:
        _raise_corrupt(f"{label} 的任务或批次身份不匹配")
    if len(validation.combination_ids) != 5 or set(validation.combination_ids) != combination_ids:
        _raise_corrupt(f"{label} 没有覆盖当前批次的五个组合")
    pairs = set(zip(validation.combination_ids, validation.result_versions))
    if len(pairs) != 5 or any(pair not in result_keys for pair in pairs):
        _raise_corrupt(f"{label} 引用了不存在的 InvestigationResult 版本")


def _validate_batch_identity(
    snapshot: BatchHistory | InvestigationState,
    *,
    task_id: str,
    expected_batch_id: str,
    require_regeneration_route: bool,
    label: str,
) -> dict[str, InvestigationSession]:
    """核对批次身份并索引五个 session。"""

    if snapshot.task_id != task_id or snapshot.batch_id != expected_batch_id:
        _raise_corrupt(f"{label} 的任务或批次身份不匹配")
    if require_regeneration_route and snapshot.route != "regenerate_hypotheses":
        _raise_corrupt(f"{label} 不是已关闭的假设重生终局")
    if len(snapshot.sessions) != 5:
        _raise_corrupt(f"{label} 必须包含五个组合会话")
    session_by_id: dict[str, InvestigationSession] = {}
    for session in snapshot.sessions:
        if session.task_id != task_id or session.batch_id != expected_batch_id:
            _raise_corrupt(f"{label} 的会话身份不匹配")
        if session.combination_id in session_by_id:
            _raise_corrupt(f"{label} 包含重复 combination_id")
        session_by_id[session.combination_id] = session
        if require_regeneration_route and session.route_marker != "revise_hypothesis":
            _raise_corrupt(f"{label} 的五个 revise_hypothesis marker 未全部保存")
        if session.result_version is None:
            _raise_corrupt(f"{label} 的组合缺少当前 result_version")
    return session_by_id


def _index_batch_results(
    snapshot: BatchHistory | InvestigationState,
    *,
    task_id: str,
    expected_batch_id: str,
    sessions: Mapping[str, InvestigationSession],
    label: str,
) -> tuple[dict[tuple[str, str], InvestigationResult], set[tuple[str, str]]]:
    """校验结果的会话身份并按组合/版本索引。"""

    if not snapshot.results:
        _raise_corrupt(f"{label} 缺少 InvestigationResult")
    result_by_key: dict[tuple[str, str], InvestigationResult] = {}
    result_keys: set[tuple[str, str]] = set()
    for result in snapshot.results:
        if result.task_id != task_id or result.batch_id != expected_batch_id:
            _raise_corrupt(f"{label} 的 InvestigationResult 身份不匹配")
        session = sessions.get(result.combination_id)
        if session is None:
            _raise_corrupt(f"{label} 的 InvestigationResult 引用了未知组合")
        if session.codex_thread_id is None or result.codex_thread_id != session.codex_thread_id:
            _raise_corrupt(f"{label} 的 InvestigationResult thread 引用不匹配")
        if result.scope_id != session.scope_id:
            _raise_corrupt(f"{label} 的 InvestigationResult scope 引用不匹配")
        key = _result_key(result)
        if key in result_keys:
            _raise_corrupt(f"{label} 包含重复 InvestigationResult 版本")
        result_keys.add(key)
        result_by_key[key] = result
    return result_by_key, result_keys


def _validate_current_result_versions(
    sessions: Mapping[str, InvestigationSession],
    result_by_key: Mapping[tuple[str, str], InvestigationResult],
    *,
    label: str,
) -> None:
    """确认五个 session 的当前版本都有对应真实结果。"""

    for combination_id, session in sessions.items():
        current_key = (combination_id, session.result_version)  # type: ignore[arg-type]
        if current_key not in result_by_key:
            _raise_corrupt(f"{label} 的会话当前版本没有对应结果")


def _validate_batch_validations(
    snapshot: BatchHistory | InvestigationState,
    *,
    task_id: str,
    expected_batch_id: str,
    combination_ids: set[str],
    result_keys: set[tuple[str, str]],
    sessions: Mapping[str, InvestigationSession],
    label: str,
) -> BatchValidation:
    """校验每轮验证的来源向量，并返回最新验证。"""

    if not snapshot.validations:
        _raise_corrupt(f"{label} 缺少完整 BatchValidation")
    validation_keys: set[str] = set()
    for index, validation in enumerate(snapshot.validations, start=1):
        validation_label = f"{label}.validations[{index}]"
        _validate_validation_reference(
            validation,
            task_id=task_id,
            batch_id=expected_batch_id,
            combination_ids=combination_ids,
            result_keys=result_keys,
            label=validation_label,
        )
        if validation.idempotency_key in validation_keys:
            _raise_corrupt(f"{label} 包含重复批次验证版本向量")
        validation_keys.add(validation.idempotency_key)

    latest = snapshot.validations[-1]
    latest_versions = dict(zip(latest.combination_ids, latest.result_versions))
    expected_versions = {
        combination_id: session.result_version for combination_id, session in sessions.items()
    }
    if latest_versions != expected_versions:
        _raise_corrupt(f"{label} 的最新验证没有绑定当前五项结果版本向量")
    return latest


def _validate_regeneration_findings(
    latest: BatchValidation,
    sessions: Mapping[str, InvestigationSession],
    *,
    label: str,
) -> None:
    """确认最新 Review 与五个 revise_hypothesis marker 事实相容。"""

    finding_by_id = {finding.combination_id: finding for finding in latest.findings}
    # exhausted fail 会让整批进入 hypothesis regeneration。该批次的其余
    # supported 结果仍是有效科学经验，也必须随 marker 一起进入下一批，不能被
    # 错误改写成 refuted 或伪造为 fail。
    has_exhausted_failure = any(
        finding.validity == "fail"
        and sessions[finding.combination_id].repair_count == 3
        for finding in latest.findings
    )
    for combination_id, session in sessions.items():
        finding = finding_by_id[combination_id]
        # refuted 与三次返修耗尽的科学区别保留在 finding 中，不由 marker 推断。
        is_refuted = (
            finding.validity == "pass" and finding.hypothesis_judgment == "refuted"
        )
        is_repair_exhausted = finding.validity == "fail" and session.repair_count == 3
        is_supported_in_exhausted_batch = (
            has_exhausted_failure
            and finding.validity == "pass"
            and finding.hypothesis_judgment == "supported"
        )
        if not (is_refuted or is_repair_exhausted or is_supported_in_exhausted_batch):
            _raise_corrupt(f"{label} 的最新验证与 revise_hypothesis marker 不一致")


def _validate_batch_snapshot(
    snapshot: BatchHistory | InvestigationState,
    *,
    task_id: str,
    expected_batch_id: str,
    require_regeneration_route: bool,
    label: str,
) -> None:
    """二次核对批次身份、结果版本向量和终局 Review。"""

    sessions = _validate_batch_identity(
        snapshot,
        task_id=task_id,
        expected_batch_id=expected_batch_id,
        require_regeneration_route=require_regeneration_route,
        label=label,
    )
    result_by_key, result_keys = _index_batch_results(
        snapshot,
        task_id=task_id,
        expected_batch_id=expected_batch_id,
        sessions=sessions,
        label=label,
    )
    _validate_current_result_versions(sessions, result_by_key, label=label)
    latest = _validate_batch_validations(
        snapshot,
        task_id=task_id,
        expected_batch_id=expected_batch_id,
        combination_ids=set(sessions),
        result_keys=result_keys,
        sessions=sessions,
        label=label,
    )
    if require_regeneration_route:
        _validate_regeneration_findings(latest, sessions, label=label)


def _project_result(result: InvestigationResult) -> _ResultProjection:
    """去掉 thread、scope 和 turn 私有身份，只复制科学结果字段。"""

    return _ResultProjection(
        combination_id=result.combination_id,
        result_version=result.result_version,
        self_judgment=result.self_judgment,
        narrative=result.narrative,
        job_refs=tuple(result.job_refs),
        operation_refs=tuple(result.operation_refs),
        artifact_refs=tuple(result.artifact_refs),
        observations=tuple(result.observations),
        limitations=tuple(result.limitations),
    )


def _project_validation(validation: BatchValidation) -> _ValidationProjection:
    """按组合标识稳定排序一轮完整验证，同时保留原始判断字段。"""

    pairs = sorted(
        zip(validation.combination_ids, validation.result_versions),
        key=lambda item: item[0],
    )
    findings = sorted(validation.findings, key=lambda finding: finding.combination_id)
    return _ValidationProjection(
        combination_ids=tuple(item[0] for item in pairs),
        result_versions=tuple(item[1] for item in pairs),
        findings=tuple(
            _FindingProjection(
                combination_id=finding.combination_id,
                validity=finding.validity,
                hypothesis_judgment=finding.hypothesis_judgment,
                repair_advice=finding.repair_advice,
            )
            for finding in findings
        ),
        can_answer=validation.can_answer,
        final_answer=validation.final_answer,
        answerability_explanation=validation.answerability_explanation,
    )


def _project_batch(snapshot: BatchHistory | InvestigationState) -> _BatchProjection:
    """按稳定字段顺序复制一个已验证的批次快照。"""

    sessions = tuple(
        _SessionProjection(
            combination_id=session.combination_id,
            result_version=session.result_version,  # type: ignore[arg-type]
            repair_count=session.repair_count,
            route_marker=session.route_marker,
        )
        for session in sorted(snapshot.sessions, key=lambda item: item.combination_id)
    )
    results = tuple(
        _project_result(result)
        for result in sorted(
            snapshot.results,
            key=lambda item: (item.combination_id, item.result_version),
        )
    )
    validations = tuple(_project_validation(validation) for validation in snapshot.validations)
    return _BatchProjection(
        task_id=snapshot.task_id,
        batch_id=snapshot.batch_id,
        route=snapshot.route,
        final_answer=snapshot.final_answer,
        answerability_explanation=snapshot.answerability_explanation,
        sessions=sessions,
        results=results,
        validations=validations,
    )


def _reject_unsafe_payload_strings(payload: Mapping[str, object]) -> None:
    """拒绝输出中的路径、URI、控制字符和常见凭据，避免宽松字符串外泄。"""

    try:
        # 保留 contracts 允许的公开科学来源 URI，但拒绝 file://、宿主地址、
        # 私有网段 URI 和所有本机路径/凭据。
        _assert_no_path_or_secret(payload, label="regeneration_memory", allow_public_source_uris=True)
        _reject_private_text_and_absolute_paths(payload, label="regeneration_memory")
    except (TypeError, ValueError) as exc:
        _raise_corrupt("重生记忆包含路径、URI、控制字符或凭据，已拒绝输出", exc)


def _reject_private_text_and_absolute_paths(value: object, *, label: str) -> None:
    """补充拒绝非标准绝对路径和把私有 transcript 写进科学文本的情况。"""

    if isinstance(value, str):
        if _ABSOLUTE_PATH_RE.search(value):
            raise ValueError(f"{label} 包含绝对宿主路径")
        if _PRIVATE_TEXT_RE.search(value):
            raise ValueError(f"{label} 包含私有身份或 transcript 文本")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_private_text_and_absolute_paths(key, label=f"{label}.<key>")
            _reject_private_text_and_absolute_paths(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_text_and_absolute_paths(item, label=f"{label}[{index}]")


def build_hypothesis_regeneration_memory(state: InvestigationState) -> dict[str, object]:
    """构造下一批假设模型的严格 JSON working-memory payload。

    输入必须是当前五组合都带 ``revise_hypothesis`` marker 的终局状态；输出包含
    全部已关闭历史批次和当前批次的结果、每轮验证及组合返修事实。结果按组合和
    版本稳定排序，验证轮次保留领域顺序但其五项向量和 findings 按组合排序。返回
    值是新的 JSON-compatible dict，后续修改不会影响输入。非终局、损坏交叉引用、
    本机路径、私有 URI、控制字符或凭据会抛出 ``ValueError``，不会返回部分 payload。
    """

    validated_state = _revalidate_state(state)
    if validated_state.route != "regenerate_hypotheses":
        _raise_corrupt("只有 regenerate_hypotheses 终局才能生成假设重生记忆")
    if not validated_state.can_generate_next_batch:
        _raise_corrupt("当前批次五个 revise_hypothesis marker 未全部齐全")

    current_batch_label = f"current batch {validated_state.batch_id}"
    _validate_batch_snapshot(
        validated_state,
        task_id=validated_state.task_id,
        expected_batch_id=validated_state.batch_id,
        require_regeneration_route=True,
        label=current_batch_label,
    )

    history_ids: set[str] = set()
    for index, history in enumerate(validated_state.history, start=1):
        if history.batch_id in history_ids or history.batch_id == validated_state.batch_id:
            _raise_corrupt("历史批次身份重复或与当前批次冲突")
        history_ids.add(history.batch_id)
        _validate_batch_snapshot(
            history,
            task_id=validated_state.task_id,
            expected_batch_id=history.batch_id,
            require_regeneration_route=True,
            label=f"closed batch {index} ({history.batch_id})",
        )

    try:
        projection = _RegenerationMemoryProjection(
            schema=REGENERATION_MEMORY_SCHEMA,
            schema_version=REGENERATION_MEMORY_VERSION,
            task_id=validated_state.task_id,
            batch_id=validated_state.batch_id,
            closed_batches=tuple(_project_batch(history) for history in validated_state.history),
            current_batch=_project_batch(validated_state),
        )
        payload = projection.model_dump(mode="json", by_alias=True)
        # 通过标准 JSON 编码再次确认所有值都能作为严格 JSON 传给下一阶段。
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        _reject_unsafe_payload_strings(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("重生记忆包含"):
            raise
        _raise_corrupt("重生记忆无法形成严格 JSON payload", exc)

    # model_dump 已经生成独立值；再通过 JSON round-trip 明确切断任何嵌套引用，
    # 同时保证返回值只含 dict/list/标量，而不是 Pydantic 或 tuple 对象。
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
