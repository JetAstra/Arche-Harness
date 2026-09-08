"""Investigation 当前索引的稳定读取和路由门禁。

这里保存的是可由不可变事件重新计算的当前视图，不是新的科学事实。模块不写事件，
只提供 WorkingMemoryService 和事件重建器共用的查询、稳定快照和 gate 判断，避免
按 SQLite rowid 或多处复制条件误判重建结果。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .contracts import (
    ContractValidationError,
    validate_investigation_batch_review_payload,
)


_SNAPSHOT_TABLES = (
    (
        "investigation_sessions",
        (
            "task_id",
            "session_id",
            "batch_id",
            "combination_id",
            "codex_thread_id",
            "workspace_ref",
            "scope_id",
            "current_result_version",
            "repair_count",
            "phase",
            "route_marker",
            "session_event_id",
            "latest_result_event_id",
            "latest_review_event_id",
            "pending_repair_event_id",
            "current_turn_event_id",
            "current_turn_id",
            "current_turn_attempt_number",
            "current_repair_turn_event_id",
            "current_repair_turn_id",
            "version",
        ),
        "batch_id,combination_id",
    ),
    (
        "investigation_results",
        (
            "result_event_id",
            "task_id",
            "batch_id",
            "combination_id",
            "result_version",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "self_judgment",
            "narrative",
            "job_refs_json",
            "operation_refs_json",
            "artifact_refs_json",
            "observations_json",
            "limitations_json",
            "evidence_verification_event_ref",
            "repair_turn_event_ref",
            "operation_instance_id",
        ),
        "batch_id,combination_id,result_version,result_event_id",
    ),
    (
        "investigation_turns",
        (
            "turn_started_event_id",
            "task_id",
            "session_id",
            "batch_id",
            "combination_id",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "operation_instance_id",
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        ),
        "batch_id,combination_id,turn_id,turn_started_event_id",
    ),
    (
        "investigation_turn_waits",
        (
            "waiting_event_id",
            "task_id",
            "turn_started_event_ref",
            "kind",
            "batch_id",
            "combination_id",
            "session_id",
            "codex_thread_id",
            "workspace_ref_digest",
            "scope_id",
            "turn_id",
            "attempt_number",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "job_id",
            "submission_operation_id",
            "submission_key",
            "submission_request_digest",
            "sidecar_record_digest",
        ),
        "batch_id,combination_id,turn_id,waiting_event_id",
    ),
    (
        "investigation_reviews",
        (
            "review_event_id",
            "task_id",
            "batch_id",
            "combination_ids_json",
            "result_versions_json",
            "result_version_vector_json",
            "findings_json",
            "can_answer",
            "final_answer",
            "answerability_explanation",
            "review_thread_id",
            "review_turn_id",
            "operation_instance_id",
            "idempotency_key",
        ),
        "batch_id,created_at,review_event_id",
    ),
    (
        "investigation_repairs",
        (
            "repair_request_event_id",
            "task_id",
            "batch_id",
            "combination_id",
            "result_version",
            "repair_count",
            "repair_advice",
            "review_event_ref",
            "completion_event_id",
            "result_event_ref",
            "operation_instance_id",
        ),
        "batch_id,combination_id,repair_count,repair_request_event_id",
    ),
    (
        "investigation_route_markers",
        (
            "marker_event_id",
            "task_id",
            "batch_id",
            "combination_id",
            "marker",
            "result_event_ref",
            "review_event_ref",
        ),
        "batch_id,combination_id,marker_event_id",
    ),
    (
        "investigation_evidence_verifications",
        (
            "verification_event_id",
            "task_id",
            "batch_id",
            "combination_id",
            "verification_object_id",
            "verifier_id",
            "verifier_version",
            "reference_binding_version",
            "verification_digest",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "jobs_json",
            "operations_json",
            "artifacts_json",
        ),
        "batch_id,combination_id,verification_object_id,verification_event_id",
    ),
    (
        "task_review_sessions",
        (
            "task_id",
            "session_event_id",
            "provider_id",
            "model",
            "sandbox_policy_digest",
            "workspace_ref",
            "codex_home_ref",
            "scope_id",
            "codex_thread_id",
            "turn_count",
            "last_batch_id",
            "last_turn_id",
            "last_turn_started_event_id",
            "last_review_event_id",
            "phase",
            "version",
        ),
        "task_id",
    ),
    (
        "task_review_turns",
        (
            "turn_started_event_id",
            "task_id",
            "session_event_ref",
            "batch_id",
            "review_thread_id",
            "review_turn_id",
            "turn_index",
            "request_digest",
            "operation_instance_id",
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        ),
        "task_id,turn_index,turn_started_event_id",
    ),
    (
        "investigation_repair_turns",
        (
            "turn_started_event_id",
            "task_id",
            "batch_id",
            "combination_id",
            "request_event_ref",
            "repair_request_number",
            "codex_thread_id",
            "scope_id",
            "turn_id",
            "request_digest",
            "operation_instance_id",
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        ),
        "batch_id,combination_id,repair_request_number,turn_started_event_id",
    ),
    (
        "investigation_turn_attempt_terminations",
        (
            "termination_event_id",
            "task_id",
            "turn_started_event_ref",
            "kind",
            "batch_id",
            "combination_id",
            "turn_id",
            "attempt_number",
            "terminal_status",
            "logical_operation_instance_id",
            "attempt_operation_instance_id",
            "previous_attempt_closure_event_ref",
        ),
        "batch_id,kind,turn_id,attempt_number,termination_event_id",
    ),
    (
        "investigation_task_terminals",
        (
            "terminal_event_id",
            "task_id",
            "batch_id",
            "accepted",
            "can_answer",
            "final_answer",
            "answerability_explanation",
            "review_event_ref",
            "marker_event_refs_json",
        ),
        "batch_id,terminal_event_id",
    ),
)


def read_investigation_index_snapshot(conn: Any, task_id: str) -> tuple[Any, ...]:
    """读取所有 Investigation 当前视图的稳定业务主键快照。

    输入是已打开的 SQLite 连接和 task 身份；动作按表的业务主键/事件身份排序，
    不读取 rowid，不写数据库；返回可直接比较的 tuple。快照保留全部语义字段、
    JSON 正文和不可变事件引用，排除重建时由 projection 重新生成的时间戳，避免
    把合法的时间规范化误判为事实篡改。表结构缺失或排序字段不存在时让 SQLite
    错误暴露，防止把不完整索引当成稳定快照。
    """

    snapshot: list[Any] = []
    for table, columns, order_by in _SNAPSHOT_TABLES:
        selected = ",".join(columns)
        rows = conn.execute(
            f"SELECT {selected} FROM {table} WHERE task_id=? ORDER BY {order_by}",
            (task_id,),
        ).fetchall()
        snapshot.append((table, tuple(tuple(row[column] for column in columns) for row in rows)))
    rows = conn.execute(
        "SELECT combination_id,investigation_phase,investigation_ready_state,"
        "investigation_stage_state,version "
        "FROM combinations WHERE task_id=? ORDER BY batch_id,combination_index,combination_id",
        (task_id,),
    ).fetchall()
    snapshot.append(
        (
            "investigation_combinations",
            tuple(tuple(row) for row in rows),
        )
    )
    return tuple(snapshot)


def evaluate_route_finding_gate(finding: Mapping[str, Any], repair_count: int) -> bool:
    """集中判断一个 finding 是否满足 revise_hypothesis 条件。

    输入是已校验的单组合 finding 和已经成功启动的 repair 次数；动作只返回布尔值，
    不读写数据库。只有 pass+refuted，或 fail 且已消耗三次返修预算，才允许 marker；
    其他 finding 返回 False。
    """

    return (
        finding.get("validity") == "pass"
        and finding.get("hypothesis_judgment") == "refuted"
    ) or (
        finding.get("validity") == "fail"
        and repair_count == 3
    )


def evaluate_combination_route_gate(
    conn: Any,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    result_event_ref: str | None = None,
    review_event_ref: str | None = None,
    require_unmarked: bool = False,
) -> Dict[str, Any]:
    """读取一个组合当前 result/review 并返回统一 route gate 结果。

    输入是任务、批次、组合和可选来源事件；动作只读当前索引与不可变 Review 正文，
    返回 ``allowed``、来源身份、finding 和失败原因。要求来源正好是当前 result/review、
    没有 pending repair；未知 session 直接失败。``require_unmarked`` 用于 marker 写入
    时拒绝重复终态，不会创建或修改任何事实。
    """

    session = _read_route_session(conn, task_id, batch_id, combination_id)
    result_ref = result_event_ref or session["latest_result_event_id"]
    review_ref = review_event_ref or session["latest_review_event_id"]
    response = _new_route_gate_response(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        result_ref=result_ref,
        review_ref=review_ref,
        repair_count=session["repair_count"],
    )
    reason = _check_route_gate_prerequisites(
        session,
        result_ref=result_ref,
        review_ref=review_ref,
        require_unmarked=require_unmarked,
    )
    if reason is not None:
        response["reason"] = reason
        return response
    result, result_reason = _read_route_result(
        conn,
        task_id,
        batch_id,
        combination_id,
        result_ref,
    )
    if result is None:
        response["reason"] = result_reason
        return response
    review, review_reason = _read_route_review(
        conn,
        task_id,
        batch_id,
        review_ref,
    )
    if review is None:
        response["reason"] = review_reason
        return response
    finding = _find_route_finding(review, combination_id)
    expected_version = dict(review["result_version_vector"]).get(combination_id)
    response["result_version"] = result["result_version"]
    if finding is None or expected_version != result["result_version"]:
        response["reason"] = "review 与当前 result 版本不一致"
        return response
    response["validity"] = finding["validity"]
    response["hypothesis_judgment"] = finding["hypothesis_judgment"]
    response["allowed"] = evaluate_route_finding_gate(finding, session["repair_count"])
    if not response["allowed"]:
        response["reason"] = "当前 Review 尚未满足 revise_hypothesis gate"
    return response


def _read_route_session(
    conn: Any,
    task_id: str,
    batch_id: str,
    combination_id: str,
) -> Any:
    """读取指定组合当前 session；缺失或数据库损坏都拒绝路由。"""

    session = conn.execute(
        "SELECT * FROM investigation_sessions WHERE task_id=? AND batch_id=? "
        "AND combination_id=?",
        (task_id, batch_id, combination_id),
    ).fetchone()
    if session is None:
        raise ContractValidationError("未知 Investigation session")
    return session


def _new_route_gate_response(
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    result_ref: Any,
    review_ref: Any,
    repair_count: Any,
) -> Dict[str, Any]:
    """创建稳定 gate 返回对象，不读取或写入索引。"""

    return {
        "allowed": False,
        "marker": "revise_hypothesis",
        "task_id": task_id,
        "batch_id": batch_id,
        "combination_id": combination_id,
        "result_event_ref": result_ref,
        "review_event_ref": review_ref,
        "repair_count": repair_count,
        "result_version": None,
        "validity": None,
        "hypothesis_judgment": None,
        "reason": None,
    }


def _check_route_gate_prerequisites(
    session: Any,
    *,
    result_ref: Any,
    review_ref: Any,
    require_unmarked: bool,
) -> str | None:
    """检查 marker 来源、返修状态和事件身份的前置条件。"""

    if require_unmarked and session["route_marker"] is not None:
        return "组合已经写入 revise_hypothesis marker"
    if (
        result_ref != session["latest_result_event_id"]
        or review_ref != session["latest_review_event_id"]
    ):
        return "marker 来源必须正好对应当前 result 和 review"
    if session["pending_repair_event_id"] is not None:
        return "当前仍有 pending repair"
    if not isinstance(result_ref, str) or not isinstance(review_ref, str):
        return "尚未同时登记 result 和 review"
    return None


def _read_route_result(
    conn: Any,
    task_id: str,
    batch_id: str,
    combination_id: str,
    result_ref: str,
) -> tuple[Any | None, str | None]:
    """读取并核对当前组合的 result 事件引用。"""

    result = conn.execute(
        "SELECT result_version,batch_id,combination_id FROM investigation_results "
        "WHERE task_id=? AND result_event_id=?",
        (task_id, result_ref),
    ).fetchone()
    if result is None:
        return None, "result 来源不存在"
    if result["batch_id"] != batch_id or result["combination_id"] != combination_id:
        return None, "result 来源不属于当前组合"
    return result, None


def _read_route_review(
    conn: Any,
    task_id: str,
    batch_id: str,
    review_ref: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """读取当前 review 并按公开合同校验其正文；损坏时 fail-closed。"""

    review_event = conn.execute(
        "SELECT event_type,payload_json FROM memory_events WHERE task_id=? AND event_id=?",
        (task_id, review_ref),
    ).fetchone()
    if review_event is None or review_event["event_type"] != "investigation_batch_reviewed":
        return None, "review 来源不存在或类型不正确"
    review = validate_investigation_batch_review_payload(
        _decode_json(review_event["payload_json"])
    )
    if review["batch_id"] != batch_id:
        return None, "review 来源不属于当前批次"
    return review, None


def _find_route_finding(
    review: Mapping[str, Any], combination_id: str
) -> Mapping[str, Any] | None:
    """从已校验 Review 中取指定组合 finding。"""

    return next(
        (
            item
            for item in review["findings"]
            if item["combination_id"] == combination_id
        ),
        None,
    )


def _decode_json(value: Any) -> Any:
    """解码索引内部事件正文，损坏时停止而不是猜测。"""

    import json

    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("Investigation Review 正文已损坏") from exc
