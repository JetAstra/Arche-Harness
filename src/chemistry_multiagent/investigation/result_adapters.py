"""把 Investigation Codex turn 校验并转成不可变领域事实。

本模块只做已经验证的 Codex 响应到领域模型的身份绑定和字段转换，不分配版本、
不修复模型输出、不调用模型或 MiniChem。当前 Review 合同使用执行输入记录身份；
本模块仅在写入 schema-v10 ``ValidationFinding`` 的边界映射成旧
``combination_id``。调查结果的旧 task、batch、combination 和 scope 只能来自
``InvestigationSession``；批次 Review 的旧 task、batch 和五项版本向量只能来自
当前 ``InvestigationState`` 与调用方显式提供的完整 vector。
"""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import (
    FailReviewFinding,
    InvestigationResponse,
    PassReviewFinding,
    TaskReviewResponse,
)
from .codex_runtime import InvestigationTurnResult
from .domain import (
    BatchValidation,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    ValidationFinding,
)


__all__ = [
    "adapt_investigation_turn_to_result",
    "adapt_review_turn_to_batch_validation",
]


def _require_non_empty_text(value: object, field_name: str) -> str:
    """校验适配器身份文本，拒绝空值并返回去除首尾空白的值。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _require_investigation_turn_result(
    turn_result: object,
) -> InvestigationResponse:
    """校验调查 turn 类型并返回其中的 InvestigationResponse。"""

    if not isinstance(turn_result, InvestigationTurnResult):
        raise TypeError("turn_result 必须是 InvestigationTurnResult")
    if not isinstance(turn_result.result, InvestigationResponse):
        if turn_result.accepted_job is not None:
            raise ValueError(
                "真实 submit 回执不是科学结果；Controller 必须先保存等待状态"
            )
        raise TypeError("turn_result.result 必须是 InvestigationResponse")
    return turn_result.result


def _require_review_turn_result(turn_result: object) -> TaskReviewResponse:
    """校验 Review turn 类型并返回其中的 TaskReviewResponse。"""

    if not isinstance(turn_result, InvestigationTurnResult):
        raise TypeError("turn_result 必须是 InvestigationTurnResult")
    if not isinstance(turn_result.result, TaskReviewResponse):
        raise TypeError("turn_result.result 必须是 TaskReviewResponse")
    return turn_result.result


def adapt_investigation_turn_to_result(
    session: InvestigationSession,
    result_version: str,
    turn_result: InvestigationTurnResult,
) -> InvestigationResult:
    """把一个调查 Codex turn 绑定到一个新的 ``InvestigationResult``。

    输入是单个不可变调查会话、调用方明确分配的新版本和 runtime 已验证的
    ``InvestigationTurnResult``。函数从会话读取 task、batch、combination、scope，
    从 runtime turn 读取 thread、turn，并原样保留响应中的科学判断、叙述、引用、
    观测和限制；没有文件、网络或模型副作用。空版本、当前版本重复、返修 thread
    漂移或响应类型错误会立即失败，不会自行生成下一个版本。
    """

    if not isinstance(session, InvestigationSession):
        raise TypeError("session 必须是 InvestigationSession")
    response = _require_investigation_turn_result(turn_result)
    normalized_version = _require_non_empty_text(result_version, "result_version")
    if session.result_version is not None:
        current_version = _require_non_empty_text(
            session.result_version,
            "session.result_version",
        )
        if normalized_version == current_version:
            raise ValueError("result_version 不能重复使用当前版本")

    thread_id = _require_non_empty_text(turn_result.thread_id, "thread_id")
    if session.codex_thread_id is not None:
        session_thread_id = _require_non_empty_text(
            session.codex_thread_id,
            "session.codex_thread_id",
        )
        if thread_id != session_thread_id:
            raise ValueError("Codex thread 不能漂移")
    turn_id = _require_non_empty_text(turn_result.turn_id, "turn_id")
    scope_id = _require_non_empty_text(session.scope_id, "session.scope_id")

    return InvestigationResult(
        task_id=session.task_id,
        batch_id=session.batch_id,
        combination_id=session.combination_id,
        result_version=normalized_version,
        codex_thread_id=thread_id,
        scope_id=scope_id,
        turn_id=turn_id,
        self_judgment=response.self_judgment,
        narrative=response.narrative,
        job_refs=response.job_refs,
        operation_refs=response.operation_refs,
        artifact_refs=response.artifact_refs,
        observations=response.observations,
        limitations=response.limitations,
    )


def _normalize_result_version_vector(
    result_version_vector: object,
) -> tuple[tuple[str, str], ...]:
    """校验显式一项或五项版本 vector，并复制成不可变的二元组序列。"""

    if isinstance(result_version_vector, (str, bytes, bytearray)) or not isinstance(
        result_version_vector,
        Sequence,
    ):
        raise TypeError("result_version_vector 必须是二元组数组")
    if len(result_version_vector) not in {1, 5}:
        raise ValueError("result_version_vector 必须包含一项或五项")

    normalized: list[tuple[str, str]] = []
    for index, item in enumerate(result_version_vector):
        if isinstance(item, (str, bytes, bytearray)) or not isinstance(item, Sequence):
            raise TypeError(
                f"result_version_vector[{index}] 必须是 "
                "(combination_id, result_version)"
            )
        if len(item) != 2:
            raise ValueError(
                f"result_version_vector[{index}] 必须包含 combination_id 和 result_version"
            )
        combination_id = _require_non_empty_text(
            item[0],
            f"result_version_vector[{index}].combination_id",
        )
        result_version = _require_non_empty_text(
            item[1],
            f"result_version_vector[{index}].result_version",
        )
        normalized.append((combination_id, result_version))

    combination_ids = tuple(item[0] for item in normalized)
    if len(set(combination_ids)) != len(combination_ids):
        raise ValueError("result_version_vector 的 combination_id 必须唯一")
    return tuple(normalized)


def _collect_previous_review_threads(state: InvestigationState) -> tuple[str, ...]:
    """读取当前及历史批次已有的 Review thread，供连续性校验。"""

    threads = [validation.review_thread_id for validation in state.validations]
    for history in state.history:
        threads.extend(validation.review_thread_id for validation in history.validations)
    return tuple(threads)


def _convert_review_finding_to_domain_finding(
    finding: PassReviewFinding | FailReviewFinding,
) -> ValidationFinding:
    """把一个合同 Review finding 转成对应的领域 pass/fail union。"""

    if isinstance(finding, PassReviewFinding):
        return ValidationFinding(
            combination_id=finding.hypothesis_execution_input_record_id,
            validity="pass",
            hypothesis_judgment=finding.hypothesis_judgment,
        )
    if isinstance(finding, FailReviewFinding):
        return ValidationFinding(
            combination_id=finding.hypothesis_execution_input_record_id,
            validity="fail",
            repair_advice=finding.repair_advice,
        )
    raise TypeError("Review finding 必须是 PassReviewFinding 或 FailReviewFinding")


def adapt_review_turn_to_batch_validation(
    state: InvestigationState,
    result_version_vector: Sequence[tuple[str, str]],
    turn_result: InvestigationTurnResult,
) -> BatchValidation:
    """把任务级 Review turn 绑定到当前批次的 ``BatchValidation``。

    输入是当前五会话 ``InvestigationState``、调用方明确提供的完整五项
    ``(combination_id, result_version)`` vector 和 runtime 已验证的 Review turn。
    函数只从 state 读取 task、batch 与会话顺序，只从 runtime turn 读取 Review
    thread、turn，严格要求 vector 与每个会话的当前版本完全一致，并按 state 会话
    顺序转换全部 pass/fail finding，同时保存 Review 分支对应的最终回答，或仅在
    五项 validity 全部通过且存在 supported 但不能回答原问题时保存回答能力说明。
    任一 fail 或全部 refuted 时不保存任何回答文本。函数不产生科学判断、返修文字或任何外部副作用；vector 不完整、版本漂移、
    组合覆盖不全、Review thread 漂移或响应类型错误会失败。
    """

    if not isinstance(state, InvestigationState):
        raise TypeError("state 必须是 InvestigationState")
    response = _require_review_turn_result(turn_result)
    normalized_vector = _normalize_result_version_vector(result_version_vector)
    thread_id = _require_non_empty_text(turn_result.thread_id, "review_thread_id")
    turn_id = _require_non_empty_text(turn_result.turn_id, "review_turn_id")

    sessions_by_combination = {session.combination_id: session for session in state.sessions}
    expected_combination_ids = tuple(session.combination_id for session in state.sessions)
    expected_vector: dict[str, str] = {}
    for session in state.sessions:
        if session.result_version is None:
            raise ValueError("当前会话都必须已有 result_version")
        expected_vector[session.combination_id] = _require_non_empty_text(
            session.result_version,
            f"session[{session.combination_id}].result_version",
        )

    received_vector = dict(normalized_vector)
    if set(received_vector) != set(expected_vector) or received_vector != expected_vector:
        raise ValueError("result_version_vector 必须完整覆盖当前五项版本且不能漂移")

    previous_threads = _collect_previous_review_threads(state)
    if previous_threads and any(thread_id != previous for previous in previous_threads):
        raise ValueError("Review thread 不能漂移")

    finding_by_combination: dict[str, ValidationFinding] = {}
    for finding in response.findings:
        converted = _convert_review_finding_to_domain_finding(finding)
        if converted.combination_id not in sessions_by_combination:
            raise ValueError("Review findings 必须完整覆盖当前组合")
        if converted.combination_id in finding_by_combination:
            raise ValueError("Review findings 不能重复组合")
        finding_by_combination[converted.combination_id] = converted
    if set(finding_by_combination) != set(expected_combination_ids):
        raise ValueError("Review findings 必须完整覆盖当前组合")

    validation_payload: dict[str, object] = {
        "task_id": state.task_id,
        "batch_id": state.batch_id,
        "combination_ids": expected_combination_ids,
        "result_versions": tuple(
            expected_vector[combination_id]
            for combination_id in expected_combination_ids
        ),
        "findings": tuple(
            finding_by_combination[combination_id]
            for combination_id in expected_combination_ids
        ),
        "can_answer": response.can_answer_original_question,
        "review_thread_id": thread_id,
        "review_turn_id": turn_id,
    }
    if response.can_answer_original_question:
        validation_payload["final_answer"] = response.final_answer
    elif response.answerability_explanation is not None:
        validation_payload["answerability_explanation"] = (
            response.answerability_explanation
        )
    return BatchValidation(
        **validation_payload,
    )
