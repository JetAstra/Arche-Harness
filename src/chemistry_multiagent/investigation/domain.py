"""Investigation and Experiment 阶段的纯领域状态推进。

本模块只保存调查阶段需要的不可变事实，并通过一个公开的状态推进入口完成
结果登记、整批验证、返修路由和下一批屏障判断。它不调用 Codex、MiniChem、
Working Memory 或数据库，因此可以在控制器接入前独立验证状态合同。
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Mapping, TypeAlias, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .contracts import _require_non_empty_text as _require_contract_text


InvestigationFailureKind: TypeAlias = Literal[
    "investigation_workflow_failure",
    "investigation_controller_failure",
]
_WORKFLOW_FAILURE_REASON_CODES = frozenset({"investigation_repair_exhausted"})


def classify_investigation_failure_kind(
    reason_code: str,
) -> InvestigationFailureKind:
    """按持久化原因区分实验流程失败和 Controller 故障。

    输入是已经过安全标识校验的 ``reason_code``；动作只对已注册的实验流程失败
    原因做确定性分类，不读取运行状态。返回公开 ``failure_kind``；未知原因保守归为
    Controller 故障，避免把基础设施异常误报成科学流程失败。
    """

    if reason_code in _WORKFLOW_FAILURE_REASON_CODES:
        return "investigation_workflow_failure"
    return "investigation_controller_failure"


class _FrozenContract(BaseModel):
    """所有领域记录共用的冻结和严格额外字段策略。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


def _require_text(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    return value


def _require_text_tuple(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} 不能包含重复引用")
    for item in value:
        _require_text(item, field_name)
    return value


class InvestigationSession(_FrozenContract):
    """一个假设组合独占的 Codex 调查会话。

    ``result_version`` 指向该组合当前可供批次验证读取的结果版本；首次调查尚未
    形成结果时为 ``None``。首次 SDK turn_start 前 ``codex_thread_id`` 也可以为
    ``None``，首个结果登记时绑定真实 thread；进入返修或其它非 created 阶段后必须
    保留同一个 thread。``repair_count`` 只统计已经登记的新返修结果版本，DeepSeek
    保存 fail 或进程恢复不会提前消耗预算。``route_marker`` 独立于会话阶段，只能
    保存 ``revise_hypothesis``，不承载导致路由的科学原因。
    """

    task_id: StrictStr
    batch_id: StrictStr
    combination_id: StrictStr
    # 首次 SDK turn_start 之前没有真实 thread；created 会话可暂时为空，首个
    # InvestigationResult 登记时再绑定真实标识。其它阶段必须已完成绑定。
    codex_thread_id: StrictStr | None = None
    workspace_ref: StrictStr
    scope_id: StrictStr
    result_version: StrictStr | None = None
    repair_count: StrictInt = 0
    phase: Literal[
        "created",
        "ready_for_batch_validation",
        "repair_pending",
        "marked_for_hypothesis_revision",
        "validated",
    ] = "created"
    route_marker: Literal["revise_hypothesis"] | None = None

    @field_validator(
        "task_id",
        "batch_id",
        "combination_id",
        "codex_thread_id",
        "workspace_ref",
        "scope_id",
        "result_version",
    )
    @classmethod
    def validate_session_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return value
        return _require_text(value, str(getattr(info, "field_name", "字段")))

    @field_validator("repair_count")
    @classmethod
    def validate_repair_count(cls, value: int) -> int:
        if not 0 <= value <= 3:
            raise ValueError("repair_count 必须位于 0 到 3 之间")
        return value

    @model_validator(mode="after")
    def validate_session_phase(self) -> InvestigationSession:
        if self.phase != "created" and self.codex_thread_id is None:
            raise ValueError("非 created 会话必须绑定真实 codex_thread_id")
        if self.phase == "marked_for_hypothesis_revision" and self.route_marker != (
            "revise_hypothesis"
        ):
            raise ValueError("标记阶段必须保存 revise_hypothesis")
        if self.route_marker is not None and self.phase != (
            "marked_for_hypothesis_revision"
        ):
            raise ValueError("revise_hypothesis 只能作为独立终止标识保存")
        if self.phase in {
            "ready_for_batch_validation",
            "repair_pending",
            "marked_for_hypothesis_revision",
            "validated",
        } and self.result_version is None:
            raise ValueError("已有可验证阶段必须绑定 result_version")
        if self.phase == "repair_pending" and self.repair_count >= 3:
            raise ValueError("三次返修结果已经用尽，不能继续保持 repair_pending")
        if self.phase == "created":
            if self.result_version is not None:
                raise ValueError("created 会话不能预先绑定 result_version")
            if self.repair_count != 0:
                raise ValueError("created 会话的 repair_count 必须为 0")
        return self


class InvestigationResult(_FrozenContract):
    """一次调查产生的只读结果和可追溯计算引用。

    ``narrative`` 是回顾性计算过程，不要求可以冷启动复现。``job_refs``、
    ``operation_refs`` 和 ``artifact_refs`` 分别保存 MiniChem 作业、typed Action
    operation 和产物的稳定标识；异步提交 operation 属于 job/artifact producer
    来源链，不进入 ``operation_refs``。本模块只核对引用形状，归属由外层服务结合
    session 的 scope_id 再核对。
    """

    task_id: StrictStr
    batch_id: StrictStr
    combination_id: StrictStr
    result_version: StrictStr
    codex_thread_id: StrictStr
    scope_id: StrictStr
    turn_id: StrictStr
    self_judgment: Literal["supported", "refuted"]
    narrative: StrictStr
    job_refs: tuple[StrictStr, ...] = ()
    operation_refs: tuple[StrictStr, ...] = ()
    artifact_refs: tuple[StrictStr, ...] = ()
    observations: tuple[StrictStr, ...] = ()
    limitations: tuple[StrictStr, ...] = ()

    @field_validator(
        "task_id",
        "batch_id",
        "combination_id",
        "result_version",
        "codex_thread_id",
        "scope_id",
        "turn_id",
        "narrative",
    )
    @classmethod
    def validate_result_text(cls, value: str, info: object) -> str:
        return _require_text(value, str(getattr(info, "field_name", "字段")))

    @field_validator("job_refs", "operation_refs", "artifact_refs", "observations", "limitations")
    @classmethod
    def validate_result_refs(
        cls,
        value: tuple[str, ...],
        info: object,
    ) -> tuple[str, ...]:
        normalized = _require_text_tuple(
            value,
            str(getattr(info, "field_name", "引用")),
        )
        if getattr(info, "field_name", None) == "observations" and not normalized:
            raise ValueError("InvestigationResult 至少需要一项实际观测")
        return normalized


class ValidationFinding(_FrozenContract):
    """DeepSeek 对一个假设组合的批次验证结果。

    有效性通过时只允许产生科学判断；有效性不通过时只允许产生返修建议。
    这使“计算不足”和“假设被证伪”在数据结构上无法混淆。
    """

    combination_id: StrictStr
    validity: Literal["pass", "fail"]
    hypothesis_judgment: Literal["supported", "refuted"] | None = None
    repair_advice: StrictStr | None = None

    @field_validator("combination_id", "repair_advice")
    @classmethod
    def validate_finding_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return value
        return _require_text(value, str(getattr(info, "field_name", "字段")))

    @model_validator(mode="after")
    def validate_finding_payload(self) -> ValidationFinding:
        if self.validity == "pass":
            if self.hypothesis_judgment is None:
                raise ValueError("validity=pass 必须有 hypothesis_judgment")
            if self.repair_advice is not None:
                raise ValueError("validity=pass 不能有 repair_advice")
        elif self.validity == "fail":
            if self.hypothesis_judgment is not None:
                raise ValueError("validity=fail 不能有 hypothesis_judgment")
            if self.repair_advice is None:
                raise ValueError("validity=fail 必须有 repair_advice")
        return self


class BatchValidation(_FrozenContract):
    """绑定完整五组合结果版本向量的一次 DeepSeek 批次验证。

    ``combination_ids`` 与 ``result_versions`` 按位置组成完整向量。幂等键对
    组合标识排序后计算，因此同一批次同一组结果版本不因输入顺序改变而产生第二次
    模型调用。每个组合的 ``validity``、判断和返修建议保存在 ``findings`` 中；
    ``can_answer=true`` 时五项 validity 必须全部通过且至少一项 supported，并保存
    唯一的 ``final_answer``。只有五项 validity 全部通过、至少一项 supported 但仍
    不能回答原问题时，才保存 ``answerability_explanation``；任一 fail 或全部
    refuted 时两个回答字段都必须为空。
    """

    task_id: StrictStr
    batch_id: StrictStr
    combination_ids: tuple[StrictStr, ...]
    result_versions: tuple[StrictStr, ...]
    findings: tuple[ValidationFinding, ...]
    can_answer: StrictBool
    final_answer: StrictStr | None = None
    answerability_explanation: StrictStr | None = None
    review_thread_id: StrictStr
    review_turn_id: StrictStr

    @field_validator(
        "task_id",
        "batch_id",
        "review_thread_id",
        "review_turn_id",
    )
    @classmethod
    def validate_batch_text(cls, value: str, info: object) -> str:
        return _require_text(value, str(getattr(info, "field_name", "字段")))

    @field_validator("combination_ids", "result_versions")
    @classmethod
    def validate_version_vector_text(
        cls,
        value: tuple[str, ...],
        info: object,
    ) -> tuple[str, ...]:
        if len(value) not in {1, 5}:
            raise ValueError("批次验证必须绑定 1 项或 5 项 result_version 向量")
        field_name = str(getattr(info, "field_name", "向量"))
        if field_name == "combination_ids":
            return _require_text_tuple(value, field_name)
        # result_version 在每个组合内部独立递增；五个首次结果都使用 "v1"
        # 是合法输入。向量身份由 (combination_id, result_version) 配对决定，
        # 不能单独要求版本文本在批次内互异。
        for item in value:
            _require_text(item, field_name)
        return value

    @field_validator("final_answer")
    @classmethod
    def validate_final_answer(cls, value: str | None) -> str | None:
        """校验可直接返回用户的最终回答，并允许公开科学来源 URI。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            "final_answer",
            allow_public_source_uris=True,
        )

    @field_validator("answerability_explanation")
    @classmethod
    def validate_answerability_explanation(cls, value: str | None) -> str | None:
        """校验仅供特殊回答能力终局公开的说明文本。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            "answerability_explanation",
            allow_public_source_uris=True,
        )

    @model_validator(mode="before")
    @classmethod
    def validate_answerability_field_presence(cls, value: object) -> object:
        """按五项 finding 类型在解析前限制回答文本的字段存在性。"""

        if not isinstance(value, Mapping):
            return value
        answerability_fields = _read_answerability_field_values(value)
        can_answer = answerability_fields[0]
        if can_answer is True:
            _validate_answerable_field_presence(answerability_fields)
        elif can_answer is False:
            _validate_non_answerable_field_presence(value, answerability_fields)
        return value

    @model_validator(mode="after")
    def validate_batch_payload(self) -> BatchValidation:
        _validate_batch_finding_coverage(self)
        has_fail, has_supported, all_validity_pass = _summarize_batch_finding_outcomes(
            self.findings
        )
        if self.can_answer:
            _validate_answerable_batch_payload(
                self,
                has_supported=has_supported,
                all_validity_pass=all_validity_pass,
            )
        else:
            _validate_non_answerable_batch_payload(
                self,
                has_fail=has_fail,
                has_supported=has_supported,
            )
        return self

    @property
    def idempotency_key(self) -> str:
        """返回当前五项结果版本向量的稳定幂等键。"""

        vector = sorted(zip(self.combination_ids, self.result_versions))
        payload = {
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "result_versions": vector,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _read_answerability_field_values(
    value: Mapping[str, object],
) -> tuple[object, bool, object, bool, object]:
    """读取回答能力字段及其存在性，供解析前分支校验使用。"""

    return (
        value.get("can_answer"),
        "final_answer" in value,
        value.get("final_answer"),
        "answerability_explanation" in value,
        value.get("answerability_explanation"),
    )


def _validate_answerable_field_presence(
    fields: tuple[object, bool, object, bool, object],
) -> None:
    """确保可回答批次在解析前携带回答且没有说明字段。"""

    _, has_final_answer, final_answer, has_explanation, _ = fields
    if not has_final_answer or final_answer is None:
        raise ValueError("can_answer=true 必须携带 final_answer")
    if has_explanation:
        raise ValueError("can_answer=true 不能携带回答能力说明")


def _validate_non_answerable_field_presence(
    value: Mapping[str, object],
    fields: tuple[object, bool, object, bool, object],
) -> None:
    """拒绝失败或全 refuted 批次携带终局回答字段。"""

    findings = value.get("findings")
    if not isinstance(findings, (list, tuple)) or len(findings) not in {1, 5}:
        return
    validities, judgments = _extract_answerability_finding_values(findings)
    has_fail = "fail" in validities
    all_refuted = all(
        validity == "pass" and judgment == "refuted"
        for validity, judgment in zip(validities, judgments)
    )
    if (has_fail or all_refuted) and (fields[1] or fields[3]):
        raise ValueError(
            "任一 validity=fail 或五项全部 refuted 时，"
            "不能携带 final_answer 或 answerability_explanation"
        )


def _extract_answerability_finding_values(
    findings: list[object] | tuple[object, ...],
) -> tuple[list[object], list[object]]:
    """从解析前的 finding 映射或模型中读取有效性与科学判断。"""

    validities = [
        item.get("validity")
        if isinstance(item, Mapping)
        else getattr(item, "validity", None)
        for item in findings
    ]
    judgments = [
        item.get("hypothesis_judgment")
        if isinstance(item, Mapping)
        else getattr(item, "hypothesis_judgment", None)
        for item in findings
    ]
    return validities, judgments


def _validate_batch_finding_coverage(batch: BatchValidation) -> None:
    """核对批次验证包含非空且相互对齐的组合判断。"""

    if not batch.combination_ids or len(set(batch.combination_ids)) != len(batch.combination_ids):
        raise ValueError("批次验证的 combination_id 必须非空且唯一")
    if (
        len(batch.result_versions) != len(batch.combination_ids)
        or len(batch.findings) != len(batch.combination_ids)
    ):
        raise ValueError("批次验证必须为每个组合保存一个判断")
    finding_ids = tuple(finding.combination_id for finding in batch.findings)
    if (
        len(set(finding_ids)) != len(batch.combination_ids)
        or set(finding_ids) != set(batch.combination_ids)
    ):
        raise ValueError("findings 必须完整覆盖 combination_id")


def _summarize_batch_finding_outcomes(
    findings: tuple[ValidationFinding, ...],
) -> tuple[bool, bool, bool]:
    """汇总批次是否失败、是否支持以及是否全部通过有效性检查。"""

    has_fail = any(finding.validity == "fail" for finding in findings)
    has_supported = any(
        finding.validity == "pass" and finding.hypothesis_judgment == "supported"
        for finding in findings
    )
    all_validity_pass = all(finding.validity == "pass" for finding in findings)
    return has_fail, has_supported, all_validity_pass


def _validate_answerable_batch_payload(
    batch: BatchValidation,
    *,
    has_supported: bool,
    all_validity_pass: bool,
) -> None:
    """核对 can_answer=true 的完整回答字段和五项有效性。"""

    if not all_validity_pass:
        raise ValueError("can_answer=true 时五项调查有效性必须全部为 pass")
    if not has_supported:
        raise ValueError("can_answer=true 时至少一项判断必须为 supported")
    if batch.answerability_explanation is not None:
        raise ValueError("can_answer=true 不能携带回答能力说明")
    if batch.final_answer is None:
        raise ValueError("can_answer=true 必须携带 final_answer")


def _validate_non_answerable_batch_payload(
    batch: BatchValidation,
    *,
    has_fail: bool,
    has_supported: bool,
) -> None:
    """核对 can_answer=false 的说明分支或无回答分支。"""

    if not has_fail and has_supported:
        if batch.answerability_explanation is None:
            raise ValueError(
                "五项 validity 全部为 pass 且存在 supported 时，"
                "can_answer=false 必须携带回答能力说明"
            )
        if batch.final_answer is not None:
            raise ValueError("can_answer=false 不能携带 final_answer")
        return
    if batch.final_answer is not None or batch.answerability_explanation is not None:
        raise ValueError(
            "任一 validity=fail 或五项全部 refuted 时，"
            "不能携带 final_answer 或 answerability_explanation"
        )


class BatchHistory(_FrozenContract):
    """已关闭批次的只读快照，保留真实结果而不是只保留路由标识。"""

    task_id: StrictStr
    batch_id: StrictStr
    sessions: tuple[InvestigationSession, ...]
    results: tuple[InvestigationResult, ...]
    validations: tuple[BatchValidation, ...]
    route: Literal[
        "investigation",
        "waiting_investigation_job",
        "repair_required",
        "accept",
        "answerability_explanation",
        "regenerate_hypotheses",
        "failed",
    ]
    final_answer: StrictStr | None = None
    answerability_explanation: StrictStr | None = None
    failure_reason_code: StrictStr | None = None
    failure_error_type: StrictStr | None = None

    @field_validator("final_answer")
    @classmethod
    def validate_history_final_answer(cls, value: str | None) -> str | None:
        """校验历史快照中的最终回答，不允许路径或凭据泄漏。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            "final_answer",
            allow_public_source_uris=True,
        )

    @field_validator("answerability_explanation")
    @classmethod
    def validate_history_answerability_explanation(
        cls,
        value: str | None,
    ) -> str | None:
        """校验历史快照中的回答能力说明，不允许路径或凭据泄漏。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            "answerability_explanation",
            allow_public_source_uris=True,
        )

    @field_validator("failure_reason_code", "failure_error_type")
    @classmethod
    def validate_history_failure_text(cls, value: str | None, info: object) -> str | None:
        """校验历史失败摘要，不允许路径、凭据或异常正文。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            str(getattr(info, "field_name", "failure")),
        )

    @model_validator(mode="after")
    def validate_history_snapshot(self) -> BatchHistory:
        _validate_history_session_count(self)
        if self.route == "accept":
            _validate_accepted_history_snapshot(self)
        elif self.route == "answerability_explanation":
            _validate_explained_history_snapshot(self)
        else:
            _validate_open_history_snapshot(self)
        _validate_failure_fields(self.route, self.failure_reason_code, self.failure_error_type)
        return self


def _validate_history_session_count(history: BatchHistory) -> None:
    """核对历史快照保留五个组合会话。"""

    if len(history.sessions) not in {1, 5}:
        raise ValueError("批次历史必须保留一个或五个组合会话")


def _validate_accepted_history_snapshot(history: BatchHistory) -> None:
    """核对 accept 历史的回答字段和最新验证来源。"""

    if history.final_answer is None:
        raise ValueError("accept 历史必须保存 final_answer")
    if history.answerability_explanation is not None:
        raise ValueError("accept 历史不能保存回答能力说明")
    latest = _require_latest_history_validation(
        history,
        "accept 历史必须保留批次验证记录",
    )
    if not latest.can_answer or latest.final_answer != history.final_answer:
        raise ValueError("accept 历史的 final_answer 必须来自最新批次验证")


def _validate_explained_history_snapshot(history: BatchHistory) -> None:
    """核对回答能力说明历史的完整 pass 和支持性来源。"""

    if history.answerability_explanation is None:
        raise ValueError("历史回答能力说明路由必须保存说明文本")
    if history.final_answer is not None:
        raise ValueError("回答能力说明历史不能保存 final_answer")
    latest = _require_latest_history_validation(
        history,
        "回答能力说明历史必须保留批次验证记录",
    )
    has_supported = any(
        finding.validity == "pass"
        and finding.hypothesis_judgment == "supported"
        for finding in latest.findings
    )
    if (
        latest.can_answer
        or any(finding.validity != "pass" for finding in latest.findings)
        or not has_supported
        or latest.answerability_explanation != history.answerability_explanation
    ):
        raise ValueError(
            "回答能力说明历史必须来自五项 validity 全部 pass、存在支持性结果"
            "且保存最新批次验证的 explanation"
        )


def _validate_open_history_snapshot(history: BatchHistory) -> None:
    """核对非终局历史不携带回答字段。"""

    if (
        history.final_answer is not None
        or history.answerability_explanation is not None
    ):
        raise ValueError("非终局历史不能保存回答或回答能力说明")


def _validate_failure_fields(
    route: str,
    reason_code: str | None,
    error_type: str | None,
) -> None:
    """核对失败路由只携带安全的失败原因和类型摘要。"""

    if route == "failed":
        if reason_code is None or error_type is None:
            raise ValueError("failed 路由必须保存 reason_code 和 error_type")
    elif reason_code is not None or error_type is not None:
        raise ValueError("失败原因只能保存在 failed 路由")


def _require_latest_history_validation(
    history: BatchHistory,
    missing_message: str,
) -> BatchValidation:
    """读取历史快照最新验证；缺失时用领域错误说明失败原因。"""

    if not history.validations:
        raise ValueError(missing_message)
    return history.validations[-1]


class InvestigationState(_FrozenContract):
    """当前五组合批次及其不可变历史。

    五个组合是批次屏障的固定最小单位；历史批次保留完整结果、验证和会话事实，
    因而重新生成假设时可以读取真实记录，而不必从 ``revise_hypothesis`` 推断原因。
    """

    task_id: StrictStr
    batch_id: StrictStr
    expected_combination_count: StrictInt = 5
    sessions: tuple[InvestigationSession, ...]
    results: tuple[InvestigationResult, ...] = ()
    validations: tuple[BatchValidation, ...] = ()
    route: Literal[
        "investigation",
        "waiting_investigation_job",
        "repair_required",
        "accept",
        "answerability_explanation",
        "regenerate_hypotheses",
        "failed",
    ] = "investigation"
    final_answer: StrictStr | None = None
    answerability_explanation: StrictStr | None = None
    failure_reason_code: StrictStr | None = None
    failure_error_type: StrictStr | None = None
    waiting_investigation_job: Mapping[str, object] | None = None
    waiting_investigation_jobs: tuple[Mapping[str, object], ...] = ()
    history: tuple[BatchHistory, ...] = ()

    @field_validator("final_answer")
    @classmethod
    def validate_state_final_answer(cls, value: str | None) -> str | None:
        """校验当前终局回答，不允许保存本机路径或凭据。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            "final_answer",
            allow_public_source_uris=True,
        )

    @field_validator("answerability_explanation")
    @classmethod
    def validate_state_answerability_explanation(
        cls,
        value: str | None,
    ) -> str | None:
        """校验当前终局的回答能力说明，不允许路径或凭据泄漏。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            "answerability_explanation",
            allow_public_source_uris=True,
        )

    @field_validator("failure_reason_code", "failure_error_type")
    @classmethod
    def validate_state_failure_text(cls, value: str | None, info: object) -> str | None:
        """校验失败路由摘要，不允许路径、凭据或异常正文。"""

        if value is None:
            return None
        return _require_contract_text(
            value,
            str(getattr(info, "field_name", "failure")),
        )

    @field_validator("task_id", "batch_id")
    @classmethod
    def validate_state_text(cls, value: str, info: object) -> str:
        return _require_text(value, str(getattr(info, "field_name", "字段")))

    @model_validator(mode="after")
    def validate_state_contract(self) -> InvestigationState:
        if self.expected_combination_count not in {1, 5}:
            raise ValueError("expected_combination_count 只能是标准 5 或作者实验 1")
        combination_ids = _validate_investigation_state_sessions(self)
        result_by_combination = _validate_investigation_state_results(self)
        _validate_investigation_state_current_versions(self, result_by_combination)
        _validate_investigation_state_validations(self, combination_ids)
        _validate_investigation_state_route(self)
        _validate_failure_fields(
            self.route,
            self.failure_reason_code,
            self.failure_error_type,
        )
        return self

    def _find_session(self, combination_id: str) -> InvestigationSession:
        for session in self.sessions:
            if session.combination_id == combination_id:
                return session
        raise ValueError(f"未知 combination_id: {combination_id}")

    def session_for(self, combination_id: str) -> InvestigationSession:
        """按组合标识读取当前会话；未知组合明确失败。"""

        return self._find_session(combination_id)

    @property
    def can_generate_next_batch(self) -> bool:
        """判断五个独立 revise_hypothesis 标识是否全部存在。"""

        return all(
            session.route_marker == "revise_hypothesis" for session in self.sessions
        )


def _validate_investigation_state_sessions(
    state: InvestigationState,
) -> tuple[str, ...]:
    """核对当前状态的五个会话及其任务、批次归属。"""

    if len(state.sessions) != state.expected_combination_count:
        raise ValueError(
            "InvestigationState 必须包含 TaskCreated 冻结数量的组合会话"
        )
    combination_ids = tuple(session.combination_id for session in state.sessions)
    if len(set(combination_ids)) != state.expected_combination_count:
        raise ValueError("组合会话的 combination_id 必须唯一")
    for session in state.sessions:
        if session.task_id != state.task_id or session.batch_id != state.batch_id:
            raise ValueError("组合会话必须属于当前 task_id 和 batch_id")
    return combination_ids


def _validate_investigation_state_results(
    state: InvestigationState,
) -> dict[str, list[InvestigationResult]]:
    """核对结果身份、会话绑定和 result_version 唯一性。"""

    result_keys: set[tuple[str, str]] = set()
    result_by_combination: dict[str, list[InvestigationResult]] = {}
    for result in state.results:
        if result.task_id != state.task_id or result.batch_id != state.batch_id:
            raise ValueError("InvestigationResult 必须属于当前任务批次")
        session = state._find_session(result.combination_id)
        if (
            session.codex_thread_id is None
            or result.codex_thread_id != session.codex_thread_id
        ):
            raise ValueError("InvestigationResult 的 Codex thread 与组合会话不匹配")
        if result.scope_id != session.scope_id:
            raise ValueError("InvestigationResult 的 scope 与组合会话不匹配")
        key = (result.combination_id, result.result_version)
        if key in result_keys:
            raise ValueError("同一组合不能重复保存同一 result_version")
        result_keys.add(key)
        result_by_combination.setdefault(result.combination_id, []).append(result)
    return result_by_combination


def _validate_investigation_state_current_versions(
    state: InvestigationState,
    result_by_combination: Mapping[str, list[InvestigationResult]],
) -> None:
    """核对每个会话的当前版本确实指向已保存结果。"""

    for session in state.sessions:
        current = session.result_version
        versions = result_by_combination.get(session.combination_id, [])
        if current is None:
            if versions:
                raise ValueError("没有当前 result_version 的会话不能已有结果历史")
        elif not any(item.result_version == current for item in versions):
            raise ValueError("会话当前 result_version 必须指向已保存结果")


def _validate_investigation_state_validations(
    state: InvestigationState,
    combination_ids: tuple[str, ...],
) -> None:
    """核对历史 Review 只引用当前批次已保存的结果版本向量。"""

    validation_keys: set[str] = set()
    saved_result_keys = {
        (item.combination_id, item.result_version) for item in state.results
    }
    for validation in state.validations:
        if validation.task_id != state.task_id or validation.batch_id != state.batch_id:
            raise ValueError("BatchValidation 必须属于当前任务批次")
        if set(validation.combination_ids) != set(combination_ids):
            raise ValueError("BatchValidation 必须覆盖当前批次的五个组合")
        validation_versions = zip(
            validation.combination_ids,
            validation.result_versions,
        )
        if any(pair not in saved_result_keys for pair in validation_versions):
            raise ValueError("BatchValidation 必须引用已保存的结果版本")
        if validation.idempotency_key in validation_keys:
            raise ValueError("同一结果版本向量只能保存一次批次验证")
        validation_keys.add(validation.idempotency_key)


def _validate_investigation_state_route(state: InvestigationState) -> None:
    """核对当前路由、回答文本和终局验证之间的对应关系。"""

    _validate_investigation_state_answer_fields(state)
    if state.route == "waiting_investigation_job":
        waiting = state.waiting_investigation_job
        waiting_jobs = state.waiting_investigation_jobs
        if not isinstance(waiting, Mapping) and not waiting_jobs:
            raise ValueError(
                "waiting_investigation_job 路由必须保存独立等待状态，不能当成科学结果"
            )
        if waiting is not None and (
            not isinstance(waiting, Mapping)
            or waiting.get("status") != "waiting_investigation_job"
        ):
            raise ValueError("waiting_investigation_job 身份状态非法")
        if any(
            not isinstance(item, Mapping)
            or item.get("status") != "waiting_investigation_job"
            for item in waiting_jobs
        ):
            raise ValueError("waiting_investigation_jobs 必须只包含独立等待状态")
    elif state.waiting_investigation_job is not None or state.waiting_investigation_jobs:
        raise ValueError(
            "只有 waiting_investigation_job 路由可以保存等待状态"
        )
    if state.route == "regenerate_hypotheses" and not state.can_generate_next_batch:
        raise ValueError("只有五个组合都带 revise_hypothesis 才能生成下一批")
    if state.route == "repair_required" and not any(
        session.phase == "repair_pending" for session in state.sessions
    ):
        raise ValueError("repair_required 必须存在待返修组合")
    if state.route == "accept":
        _validate_investigation_state_accept_route(state)
    if state.route == "answerability_explanation":
        _validate_investigation_state_explanation_route(state)


def _validate_investigation_state_answer_fields(state: InvestigationState) -> None:
    """核对回答文本只能出现在其对应的终局路由。"""

    if state.route == "answerability_explanation" and state.answerability_explanation is None:
        raise ValueError("回答能力说明路由必须保存说明文本")
    if state.route == "answerability_explanation" and state.final_answer is not None:
        raise ValueError("回答能力说明路由不能保存 final_answer")
    if state.route != "answerability_explanation" and state.answerability_explanation is not None:
        raise ValueError("只有回答能力说明路由可以保存说明文本")
    if state.route == "accept" and state.final_answer is None:
        raise ValueError("accept 路由必须保存 final_answer")
    if state.route != "accept" and state.final_answer is not None:
        raise ValueError("final_answer 只能保存在 accept 路由")
    if state.route == "failed":
        if state.answerability_explanation is not None:
            raise ValueError("failed 路由不能保存回答能力说明")


def _validate_investigation_state_accept_route(state: InvestigationState) -> None:
    """核对 accept 路由确实来自最新完整支持性验证。"""

    if not state.validations:
        raise ValueError("accept 必须有批次验证记录")
    latest = state.validations[-1]
    if (
        not latest.can_answer
        or latest.final_answer is None
        or latest.final_answer != state.final_answer
        or any(finding.validity != "pass" for finding in latest.findings)
        or not any(
            finding.hypothesis_judgment == "supported"
            for finding in latest.findings
        )
    ):
        raise ValueError(
            "accept 必须同时满足五项 validity=pass、can_answer 和 pass+supported"
        )


def _validate_investigation_state_explanation_route(
    state: InvestigationState,
) -> None:
    """核对回答能力说明来自完整 pass 且含支持性结果的最新验证。"""

    if not state.validations:
        raise ValueError("回答能力说明路由必须有批次验证记录")
    latest = state.validations[-1]
    if (
        latest.can_answer
        or any(finding.validity != "pass" for finding in latest.findings)
        or not any(
            finding.validity == "pass"
            and finding.hypothesis_judgment == "supported"
            for finding in latest.findings
        )
    ):
        raise ValueError(
            "回答能力说明必须来自五项 validity 全部 pass、存在支持性结果"
            "但 can_answer=false"
        )
    if latest.answerability_explanation != state.answerability_explanation:
        raise ValueError("回答能力说明路由必须保存最新验证的 explanation")


class RecordInvestigationResult(_FrozenContract):
    """登记一个组合的新 InvestigationResult。"""

    event_type: Literal["record_investigation_result"] = "record_investigation_result"
    result: InvestigationResult


class ApplyBatchValidation(_FrozenContract):
    """应用一次绑定完整五项结果版本的 DeepSeek 验证。"""

    event_type: Literal["apply_batch_validation"] = "apply_batch_validation"
    validation: BatchValidation


class BeginNextBatch(_FrozenContract):
    """在五个 revise_hypothesis 标识齐全后开启下一批假设。"""

    event_type: Literal["begin_next_batch"] = "begin_next_batch"
    batch_id: StrictStr
    sessions: tuple[InvestigationSession, ...]

    @field_validator("batch_id")
    @classmethod
    def validate_next_batch_id(cls, value: str) -> str:
        return _require_text(value, "batch_id")


InvestigationEvent: TypeAlias = Annotated[
    Union[RecordInvestigationResult, ApplyBatchValidation, BeginNextBatch],
    Field(discriminator="event_type"),
]
_EVENT_ADAPTER = TypeAdapter(InvestigationEvent)


def _validated_event(event: InvestigationEvent | Mapping[str, object]) -> InvestigationEvent:
    if isinstance(event, (RecordInvestigationResult, ApplyBatchValidation, BeginNextBatch)):
        # ``model_copy(update=...)`` 在 Pydantic 中不会自动重新校验；即使调用方
        # 传入了模型实例，仍通过公开入口重新构造一次严格事件。
        return _EVENT_ADAPTER.validate_python(event.model_dump(exclude_none=True))
    if not isinstance(event, Mapping):
        raise TypeError("event 必须是 Investigation 领域事件或严格事件映射")
    return _EVENT_ADAPTER.validate_python(event)


def _replace_session(
    state: InvestigationState,
    replacement: InvestigationSession,
) -> tuple[InvestigationSession, ...]:
    return tuple(
        replacement if item.combination_id == replacement.combination_id else item
        for item in state.sessions
    )


def _copy_state(
    state: InvestigationState,
    **updates: object,
) -> InvestigationState:
    payload = state.model_dump(exclude_none=True)
    payload.update(updates)
    return InvestigationState.model_validate(payload)


def _validate_result_task_batch_identity(
    state: InvestigationState,
    result: InvestigationResult,
) -> None:
    """核对调查结果属于当前任务和批次。"""

    if result.task_id != state.task_id or result.batch_id != state.batch_id:
        raise ValueError("InvestigationResult 不属于当前任务批次")


def _validate_result_thread_binding(
    session: InvestigationSession,
    result: InvestigationResult,
) -> None:
    """核对返修结果继续使用该组合原有的 Codex thread。"""

    if (
        session.codex_thread_id is not None
        and result.codex_thread_id != session.codex_thread_id
    ):
        raise ValueError("InvestigationResult 必须恢复该组合原有 Codex thread")


def _validate_result_scope_binding(
    session: InvestigationSession,
    result: InvestigationResult,
) -> None:
    """核对调查结果继续使用该组合原有的 MiniChem scope。"""

    if result.scope_id != session.scope_id:
        raise ValueError("InvestigationResult 必须属于该组合原有 MiniChem scope")


def _validate_result_registration_phase(session: InvestigationSession) -> None:
    """核对当前会话允许登记首次或返修调查结果。"""

    if session.route_marker is not None:
        raise ValueError("已标记 revise_hypothesis 的组合不能继续当前调查")
    if session.phase not in {"created", "repair_pending"}:
        raise ValueError("只有首次调查或待返修会话可以登记新结果")


def _reject_duplicate_investigation_result(
    state: InvestigationState,
    result: InvestigationResult,
) -> None:
    """拒绝同一组合重复登记相同结果版本。"""

    if any(
        item.combination_id == result.combination_id
        and item.result_version == result.result_version
        for item in state.results
    ):
        raise ValueError("同一组合不能重复登记同一 result_version")


def _calculate_next_repair_count(session: InvestigationSession) -> int:
    """计算登记结果后的返修次数，并拒绝超过三次。"""

    next_repair_count = session.repair_count + (
        1 if session.phase == "repair_pending" else 0
    )
    if next_repair_count > 3:
        raise ValueError("返修结果最多登记三次")
    return next_repair_count


def _create_recorded_investigation_session(
    session: InvestigationSession,
    result: InvestigationResult,
    repair_count: int,
) -> InvestigationSession:
    """生成登记结果后的可验证会话快照。"""

    return session.model_copy(
        update={
            "codex_thread_id": result.codex_thread_id,
            "result_version": result.result_version,
            "repair_count": repair_count,
            "phase": "ready_for_batch_validation",
        }
    )


def _select_route_after_result_registration(
    state: InvestigationState,
    sessions: tuple[InvestigationSession, ...],
) -> str:
    """在返修结果全部登记后选择下一次批次验证路由。"""

    if state.route == "repair_required" and not any(
        session.phase == "repair_pending" for session in sessions
    ):
        return "investigation"
    return state.route


def _record_investigation_result(
    state: InvestigationState,
    event: RecordInvestigationResult,
) -> InvestigationState:
    result = event.result
    session = state.session_for(result.combination_id)
    _validate_result_task_batch_identity(state, result)
    _validate_result_thread_binding(session, result)
    _validate_result_scope_binding(session, result)
    _validate_result_registration_phase(session)
    _reject_duplicate_investigation_result(state, result)
    next_repair_count = _calculate_next_repair_count(session)
    updated_session = _create_recorded_investigation_session(
        session,
        result,
        next_repair_count,
    )
    updated_sessions = _replace_session(state, updated_session)
    next_route = _select_route_after_result_registration(state, updated_sessions)
    return _copy_state(
        state,
        sessions=updated_sessions,
        results=state.results + (result,),
        route=next_route,
    )


def _validate_batch_input(
    state: InvestigationState,
    validation: BatchValidation,
) -> None:
    if validation.task_id != state.task_id or validation.batch_id != state.batch_id:
        raise ValueError("BatchValidation 不属于当前任务批次")
    expected_ids = tuple(session.combination_id for session in state.sessions)
    if set(validation.combination_ids) != set(expected_ids):
        raise ValueError("BatchValidation 必须覆盖当前批次的五个组合")
    current_versions = {
        session.combination_id: session.result_version for session in state.sessions
    }
    submitted_versions = dict(zip(validation.combination_ids, validation.result_versions))
    if any(version is None for version in current_versions.values()):
        raise ValueError("五个组合都必须先形成 InvestigationResult")
    if submitted_versions != current_versions:
        raise ValueError("BatchValidation 的 result_version 向量不是当前完整结果快照")
    for session in state.sessions:
        if session.phase not in {
            "ready_for_batch_validation",
            "marked_for_hypothesis_revision",
            "validated",
        }:
            raise ValueError("存在尚未完成调查或正在返修的组合，不能验证批次")


def _replay_existing_batch_validation(
    state: InvestigationState,
    validation: BatchValidation,
) -> InvestigationState | None:
    """按完整结果版本向量重放已有验证，冲突内容明确失败。"""

    existing = next(
        (
            item
            for item in state.validations
            if item.idempotency_key == validation.idempotency_key
        ),
        None,
    )
    if existing is None:
        return None
    if existing != validation:
        raise ValueError("相同 result_version 向量对应了不同的验证内容")
    return state


def _validate_batch_route_is_open(state: InvestigationState) -> None:
    """拒绝向已经关闭的当前批次追加批次验证。"""

    if state.route in {
        "accept",
        "answerability_explanation",
        "regenerate_hypotheses",
        "failed",
    }:
        raise ValueError("当前批次已经进入终局，不能追加新的批次验证")


def _mark_session_for_hypothesis_revision(
    session: InvestigationSession,
) -> InvestigationSession:
    """为 refuted 或耗尽返修预算的组合保存独立重生标识。"""

    return session.model_copy(
        update={
            "phase": "marked_for_hypothesis_revision",
            "route_marker": "revise_hypothesis",
        }
    )


def _mark_session_for_repair(
    session: InvestigationSession,
) -> InvestigationSession:
    """把有效性失败且仍有预算的组合置为待返修。"""

    return session.model_copy(
        update={
            "phase": "repair_pending",
            "route_marker": None,
        }
    )


def _mark_session_as_validated(
    session: InvestigationSession,
) -> InvestigationSession:
    """把通过且未 refuted 的组合标记为本轮已验证。"""

    return session.model_copy(update={"phase": "validated", "route_marker": None})


def _advance_session_after_batch_finding(
    session: InvestigationSession,
    finding: ValidationFinding,
) -> InvestigationSession:
    """按单项 finding 推进一个组合的返修或终止状态。"""

    if finding.validity == "fail":
        if session.repair_count >= 3:
            return _mark_session_for_hypothesis_revision(session)
        return _mark_session_for_repair(session)
    if finding.hypothesis_judgment == "refuted":
        return _mark_session_for_hypothesis_revision(session)
    return _mark_session_as_validated(session)


def _advance_sessions_after_batch_validation(
    state: InvestigationState,
    validation: BatchValidation,
) -> tuple[InvestigationSession, ...]:
    """按五项 finding 推进当前批次的全部组合会话。"""

    finding_by_id = {
        finding.combination_id: finding for finding in validation.findings
    }
    return tuple(
        _advance_session_after_batch_finding(
            session,
            finding_by_id[session.combination_id],
        )
        for session in state.sessions
    )


def _has_supported_batch_finding(
    validation: BatchValidation,
) -> bool:
    """判断批次是否包含至少一项通过且支持性的 finding。"""

    return any(
        finding.validity == "pass" and finding.hypothesis_judgment == "supported"
        for finding in validation.findings
    )


def _all_batch_findings_pass(validation: BatchValidation) -> bool:
    """判断五项 finding 是否全部通过有效性检查。"""

    return all(finding.validity == "pass" for finding in validation.findings)


def _select_batch_validation_route(
    sessions: tuple[InvestigationSession, ...],
    validation: BatchValidation,
) -> tuple[str, str | None, str | None]:
    """根据会话推进和验证内容选择路由及终局文本。"""

    has_pending_repairs = any(
        session.phase == "repair_pending" for session in sessions
    )
    all_marked = all(
        session.route_marker == "revise_hypothesis" for session in sessions
    )
    has_supported_result = _has_supported_batch_finding(validation)
    all_validity_pass = _all_batch_findings_pass(validation)
    if has_pending_repairs:
        return "repair_required", None, None
    if all_marked:
        return "regenerate_hypotheses", None, None
    if has_supported_result and validation.can_answer and all_validity_pass:
        return "accept", None, validation.final_answer
    if all_validity_pass and has_supported_result and not validation.can_answer:
        return "answerability_explanation", validation.answerability_explanation, None
    raise ValueError(
        "任一 fail 或全部 refuted 后，组合必须全部标记 revise_hypothesis；"
        "回答能力说明只允许用于五项 validity 全部 pass 且存在 supported 的特殊终局"
    )


def _apply_batch_validation(
    state: InvestigationState,
    event: ApplyBatchValidation,
) -> InvestigationState:
    validation = event.validation
    replayed_state = _replay_existing_batch_validation(state, validation)
    if replayed_state is not None:
        return replayed_state
    _validate_batch_route_is_open(state)
    _validate_batch_input(state, validation)
    updated_sessions = _advance_sessions_after_batch_validation(state, validation)
    route, explanation, final_answer = _select_batch_validation_route(
        updated_sessions,
        validation,
    )

    return _copy_state(
        state,
        sessions=tuple(updated_sessions),
        validations=state.validations + (validation,),
        route=route,
        final_answer=final_answer,
        answerability_explanation=explanation,
    )


def _begin_next_batch(
    state: InvestigationState,
    event: BeginNextBatch,
) -> InvestigationState:
    if state.expected_combination_count == 1:
        raise ValueError("作者单组合实验模式禁止假设批次重新生成")
    if not state.can_generate_next_batch:
        raise ValueError("只有五个组合都带 revise_hypothesis 才能进入下一批")
    if state.route != "regenerate_hypotheses":
        raise ValueError("当前批次尚未满足假设批次重新生成路由")
    if event.batch_id == state.batch_id:
        raise ValueError("下一批 batch_id 必须改变")
    if len(event.sessions) != state.expected_combination_count:
        raise ValueError("下一批必须保持冻结组合数量")
    next_ids = {session.combination_id for session in event.sessions}
    if len(next_ids) != state.expected_combination_count:
        raise ValueError("下一批的 combination_id 必须唯一")
    for session in event.sessions:
        if session.task_id != state.task_id or session.batch_id != event.batch_id:
            raise ValueError("下一批会话的 task_id/batch_id 不匹配")
        if session.result_version is not None or session.repair_count != 0:
            raise ValueError("下一批会话必须从无结果、零返修的新上下文开始")
        if session.route_marker is not None or session.phase != "created":
            raise ValueError("下一批会话不能继承旧批次路由标识")

    history_item = BatchHistory(
        task_id=state.task_id,
        batch_id=state.batch_id,
        sessions=state.sessions,
        results=state.results,
        validations=state.validations,
        route=state.route,
        final_answer=state.final_answer,
        answerability_explanation=state.answerability_explanation,
    )
    return InvestigationState(
        task_id=state.task_id,
        batch_id=event.batch_id,
        expected_combination_count=state.expected_combination_count,
        sessions=event.sessions,
        history=state.history + (history_item,),
    )


def apply_investigation_event(
    state: InvestigationState,
    event: InvestigationEvent | Mapping[str, object],
) -> InvestigationState:
    """以纯函数方式推进 Investigation and Experiment 领域状态。

    输入是当前不可变状态和一个严格领域事件，返回全新的状态；原对象、旧结果和
    已保存验证记录均不会修改。重复提交同一完整结果版本向量会直接重放已保存的
    ``BatchValidation``，不同内容使用同一向量时明确失败。该入口不执行任何模型、
    作业或持久化副作用，非法阶段、跨组合引用和超过三次返修都会抛出 ``ValueError``
    或 Pydantic 的 ``ValidationError``。
    """

    if not isinstance(state, InvestigationState):
        raise TypeError("state 必须是 InvestigationState")
    validated_state = InvestigationState.model_validate(state.model_dump(exclude_none=True))
    validated_event = _validated_event(event)
    if isinstance(validated_event, RecordInvestigationResult):
        return _record_investigation_result(validated_state, validated_event)
    if isinstance(validated_event, ApplyBatchValidation):
        return _apply_batch_validation(validated_state, validated_event)
    if isinstance(validated_event, BeginNextBatch):
        return _begin_next_batch(validated_state, validated_event)
    raise TypeError(f"不支持的 Investigation 事件: {type(validated_event).__name__}")
