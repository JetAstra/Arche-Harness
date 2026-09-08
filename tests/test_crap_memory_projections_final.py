"""Investigation 投影的证据、终态和 revise_hypothesis 绑定边界回归。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from chemistry_multiagent.memory import ContractValidationError
from chemistry_multiagent.memory.projections import (
    _validate_result_evidence_source,
    _validate_terminal_marker_references,
    _validate_terminal_marker_result_bindings,
    _validate_terminal_marker_session_bindings,
    _validate_terminal_session_consistency,
)


COMBINATION_IDS = tuple(f"comb-{index}" for index in range(1, 6))


def _evidence_connection(
    *,
    event_type: str = "investigation_result_recorded",
    source_refs: list[str] | None = None,
    include_evidence: bool = True,
) -> sqlite3.Connection:
    """建立只包含证据投影和 Result 事件的真实 SQLite 边界夹具。"""

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE investigation_evidence_verifications (
            task_id TEXT NOT NULL,
            verification_event_id TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            combination_id TEXT NOT NULL,
            codex_thread_id TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            jobs_json TEXT NOT NULL,
            operations_json TEXT NOT NULL,
            artifacts_json TEXT NOT NULL
        );
        CREATE TABLE memory_events (
            task_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            source_refs_json TEXT NOT NULL
        );
        """
    )
    if include_evidence:
        conn.execute(
            "INSERT INTO investigation_evidence_verifications VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "task-1",
                "evidence-1",
                "batch-1",
                "comb-1",
                "thread-1",
                "scope-1",
                "turn-1",
                json.dumps([{"job_id": "job-2"}, {"job_id": "job-1"}]),
                json.dumps([{"operation_id": "operation-1"}]),
                json.dumps([{"artifact_id": "artifact-1"}]),
            ),
        )
    conn.execute(
        "INSERT INTO memory_events VALUES(?,?,?,?)",
        (
            "task-1",
            "result-event-1",
            event_type,
            json.dumps(source_refs if source_refs is not None else ["evidence-1"]),
        ),
    )
    conn.commit()
    return conn


def _result_value(**updates: Any) -> dict[str, Any]:
    """构造一条通过正文合同后的最小 Result 视图。"""

    value: dict[str, Any] = {
        "investigation_protocol_version": 8,
        "evidence_verification_event_ref": "evidence-1",
        "batch_id": "batch-1",
        "combination_id": "comb-1",
        "codex_thread_id": "thread-1",
        "scope_id": "scope-1",
        "turn_id": "turn-1",
        "job_refs": ["job-1", "job-2"],
        "operation_refs": ["operation-1"],
        "artifact_refs": ["artifact-1"],
    }
    value.update(updates)
    return value


def test_result_evidence_source_accepts_v8_and_legacy_result() -> None:
    """v8 必须沿用证据来源，旧协议仍只确认 Result 不可变事件。"""

    conn = _evidence_connection()
    assert _validate_result_evidence_source(
        conn,
        task_id="task-1",
        event_id="result-event-1",
        value=_result_value(),
    ) == (True, "evidence-1", ["evidence-1"])

    legacy = _result_value(
        investigation_protocol_version=7,
        evidence_verification_event_ref=None,
    )
    assert _validate_result_evidence_source(
        conn,
        task_id="task-1",
        event_id="result-event-1",
        value=legacy,
    ) == (False, None, ["evidence-1"])


@pytest.mark.parametrize(
    "value",
    [
        _result_value(evidence_verification_event_ref=None),
        _result_value(evidence_verification_event_ref=123),
    ],
)
def test_result_evidence_source_rejects_missing_or_non_string_evidence_ref(
    value: dict[str, Any],
) -> None:
    """v8 缺少稳定 evidence ref 时必须在查询前失败。"""

    with pytest.raises(ContractValidationError, match="必须引用 VerifiedInvestigationEvidence"):
        _validate_result_evidence_source(
            _evidence_connection(),
            task_id="task-1",
            event_id="result-event-1",
            value=value,
        )


def test_result_evidence_source_rejects_unknown_evidence_event() -> None:
    """证据事件不存在时不能从 Result 自身的 refs 猜测来源。"""

    with pytest.raises(ContractValidationError, match="验证证据事件"):
        _validate_result_evidence_source(
            _evidence_connection(include_evidence=False),
            task_id="task-1",
            event_id="result-event-1",
            value=_result_value(),
        )


@pytest.mark.parametrize(
    "field",
    ["batch_id", "combination_id", "codex_thread_id", "scope_id", "turn_id"],
)
def test_result_evidence_source_rejects_identity_drift(field: str) -> None:
    """证据摘要的 task 内部身份逐字段漂移时必须 fail closed。"""

    value = _result_value(**{field: f"different-{field}"})
    with pytest.raises(ContractValidationError, match="验证证据事件与当前身份不一致"):
        _validate_result_evidence_source(
            _evidence_connection(),
            task_id="task-1",
            event_id="result-event-1",
            value=value,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("job_refs", ["job-1"]),
        ("operation_refs", ["operation-other"]),
        ("artifact_refs", ["artifact-other"]),
    ],
)
def test_result_evidence_source_rejects_computation_reference_drift(
    field: str,
    replacement: list[str],
) -> None:
    """Result 不能添加、删除或替换证据摘要中的计算来源。"""

    with pytest.raises(ContractValidationError, match="refs 必须逐项沿用"):
        _validate_result_evidence_source(
            _evidence_connection(),
            task_id="task-1",
            event_id="result-event-1",
            value=_result_value(**{field: replacement}),
        )


def test_result_evidence_source_rejects_wrong_event_type_and_missing_evidence_source() -> None:
    """Result 事件类型或 source_refs 缺失时均不能建立结果投影。"""

    with pytest.raises(ContractValidationError, match="不可变事件"):
        _validate_result_evidence_source(
            _evidence_connection(event_type="other_event"),
            task_id="task-1",
            event_id="result-event-1",
            value=_result_value(),
        )
    with pytest.raises(ContractValidationError, match="source_refs 必须引用"):
        _validate_result_evidence_source(
            _evidence_connection(source_refs=["review-1"]),
            task_id="task-1",
            event_id="result-event-1",
            value=_result_value(),
        )


def _session(
    combination_id: str,
    *,
    result: str | None = None,
    review: str | None = "review-1",
    version: str | None = None,
    pending: str | None = None,
    marker: str | None = None,
    phase: str = "ready_for_batch_validation",
) -> dict[str, Any]:
    """构造终态校验使用的 session 行视图。"""

    return {
        "combination_id": combination_id,
        "latest_result_event_id": result or f"result-{combination_id}",
        "latest_review_event_id": review,
        "current_result_version": version or f"v-{combination_id}",
        "pending_repair_event_id": pending,
        "route_marker": marker,
        "phase": phase,
    }


def _terminal_sessions(
    updates: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """构造五项完整 session 集合。"""

    sessions = [_session(combination_id) for combination_id in COMBINATION_IDS]
    for index, update in (updates or {}).items():
        sessions[index].update(update)
    return sessions


def _terminal_review(
    *,
    combination_ids: tuple[str, ...] = COMBINATION_IDS,
    vector: list[list[str]] | None = None,
    findings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """构造与五个 session 对应的 Review 视图。"""

    return {
        "combination_ids": list(combination_ids),
        "result_version_vector": vector
        or [[combination_id, f"v-{combination_id}"] for combination_id in COMBINATION_IDS],
        "findings": findings or [],
    }


def test_terminal_session_consistency_accepts_complete_current_vector() -> None:
    """终态校验返回 Review 的五个组合集合，并接受当前版本向量。"""

    sessions = _terminal_sessions()
    value = {"accepted": True, "review_event_ref": "review-1"}
    assert _validate_terminal_session_consistency(
        value,
        _terminal_review(),
        sessions,
    ) == set(COMBINATION_IDS)


@pytest.mark.parametrize(
    "sessions",
    [
        _terminal_sessions()[:4],
        _terminal_sessions({0: {"latest_result_event_id": None}}),
        _terminal_sessions({1: {"pending_repair_event_id": "repair-1"}}),
    ],
)
def test_terminal_session_consistency_rejects_incomplete_sessions(
    sessions: list[dict[str, Any]],
) -> None:
    """数量、当前结果和 pending repair 任一不完整都不能关闭任务。"""

    with pytest.raises(ContractValidationError, match="TaskClosed 必须绑定"):
        _validate_terminal_session_consistency(
            {"accepted": False, "review_event_ref": "review-1"},
            _terminal_review(),
            sessions,
        )


def test_terminal_session_consistency_allows_accepted_task_with_current_marker_phase() -> None:
    """accepted 的 session 一致性校验不再把合法部分 marker 当作冲突。"""

    sessions = _terminal_sessions({0: {"route_marker": "marker-1"}})
    assert _validate_terminal_session_consistency(
        {"accepted": True, "review_event_ref": "review-1"},
        _terminal_review(),
        sessions,
    ) == set(COMBINATION_IDS)


def test_terminal_session_consistency_rejects_review_identity_or_version_drift() -> None:
    """Review 组合、Review 引用和 result_version vector 必须都指向当前事实。"""

    sessions = _terminal_sessions()
    with pytest.raises(ContractValidationError, match="未覆盖五个"):
        _validate_terminal_session_consistency(
            {"accepted": False, "review_event_ref": "review-1"},
            _terminal_review(combination_ids=("comb-1", "comb-2", "comb-3", "comb-4", "other")),
            sessions,
        )

    stale_review_sessions = _terminal_sessions({2: {"latest_review_event_id": "review-old"}})
    with pytest.raises(ContractValidationError, match="当前 Review"):
        _validate_terminal_session_consistency(
            {"accepted": False, "review_event_ref": "review-1"},
            _terminal_review(),
            stale_review_sessions,
        )

    with pytest.raises(ContractValidationError, match="result 版本"):
        _validate_terminal_session_consistency(
            {"accepted": False, "review_event_ref": "review-1"},
            _terminal_review(vector=[[item, "stale"] for item in COMBINATION_IDS]),
            sessions,
        )


def _markers(
    *,
    result_updates: dict[str, str] | None = None,
    review_updates: dict[str, str] | None = None,
    wrong_combination: bool = False,
) -> list[dict[str, str]]:
    """构造五条绑定当前 Result/Review 的 marker 行。"""

    result_updates = result_updates or {}
    review_updates = review_updates or {}
    return [
        {
            "combination_id": ("other" if wrong_combination and index == 5 else combination_id),
            "marker_event_id": f"marker-{index}",
            "result_event_ref": result_updates.get(
                combination_id, f"result-{combination_id}"
            ),
            "review_event_ref": review_updates.get(combination_id, "review-1"),
        }
        for index, combination_id in enumerate(COMBINATION_IDS, start=1)
    ]


def test_terminal_marker_bindings_accept_empty_or_complete_marker_set() -> None:
    """没有 marker 的 accepted 路径和完整 revise_hypothesis 路径都可正常收束。"""

    sessions = _terminal_sessions()
    value = {"accepted": True, "review_event_ref": "review-1"}
    _validate_terminal_marker_session_bindings(
        value,
        set(COMBINATION_IDS),
        {"findings": []},
        sessions,
        [],
    )

    marked_sessions = _terminal_sessions(
        {
            index: {
                "route_marker": f"marker-{index + 1}",
                "phase": "marked_for_hypothesis_revision",
            }
            for index in range(5)
        }
    )
    _validate_terminal_marker_session_bindings(
        {"accepted": False, "review_event_ref": "review-1"},
        set(COMBINATION_IDS),
        {"findings": []},
        marked_sessions,
        _markers(),
    )


def test_accepted_marker_bindings_require_exact_refuted_finding_set() -> None:
    """accepted 终态必须逐项保存当前 Review 中所有 pass+refuted marker。"""

    findings = [
        {
            "combination_id": combination_id,
            "validity": "pass",
            "hypothesis_judgment": "refuted" if index < 2 else "supported",
        }
        for index, combination_id in enumerate(COMBINATION_IDS)
    ]
    marked_sessions = _terminal_sessions(
        {
            index: {
                "route_marker": f"marker-{index + 1}",
                "phase": "marked_for_hypothesis_revision",
            }
            for index in range(2)
        }
    )
    partial_markers = _markers()[:1]
    with pytest.raises(ContractValidationError, match="精确覆盖"):
        _validate_terminal_marker_session_bindings(
            {"accepted": True, "review_event_ref": "review-1"},
            set(COMBINATION_IDS),
            {"findings": findings},
            marked_sessions,
            partial_markers,
        )

    _validate_terminal_marker_session_bindings(
        {"accepted": True, "review_event_ref": "review-1"},
        set(COMBINATION_IDS),
        {"findings": findings},
        marked_sessions,
        _markers()[:2],
    )

    with pytest.raises(ContractValidationError, match="精确覆盖"):
        _validate_terminal_marker_session_bindings(
            {"accepted": True, "review_event_ref": "review-1"},
            set(COMBINATION_IDS),
            {"findings": findings},
            _terminal_sessions(),
            _markers()[:3],
        )


def test_terminal_marker_bindings_rejects_phase_and_incomplete_marker_set() -> None:
    """marker 阶段漂移、数量不足和组合集合漂移都必须停止终态写入。"""

    bad_phase = _terminal_sessions({0: {"route_marker": "marker-1"}})
    with pytest.raises(ContractValidationError, match="阶段不一致"):
        _validate_terminal_marker_session_bindings(
            {"accepted": False, "review_event_ref": "review-1"},
            set(COMBINATION_IDS),
            {"findings": []},
            bad_phase,
            [],
        )

    with pytest.raises(ContractValidationError, match="必须覆盖当前组合"):
        _validate_terminal_marker_session_bindings(
            {"accepted": False, "review_event_ref": "review-1"},
            set(COMBINATION_IDS),
            {"findings": []},
            _terminal_sessions(),
            _markers()[:4],
        )
    with pytest.raises(ContractValidationError, match="必须覆盖当前组合"):
        _validate_terminal_marker_session_bindings(
            {"accepted": False, "review_event_ref": "review-1"},
            set(COMBINATION_IDS),
            {"findings": []},
            _terminal_sessions(),
            _markers(wrong_combination=True),
        )


@pytest.mark.parametrize(
    "markers",
    [
        _markers(result_updates={"comb-1": "stale-result"}),
        _markers(review_updates={"comb-2": "review-old"}),
    ],
)
def test_terminal_marker_bindings_reject_current_result_or_review_drift(
    markers: list[dict[str, str]],
) -> None:
    """每个 marker 必须精确绑定对应组合的当前 Result 与 Review。"""

    marked_sessions = _terminal_sessions(
        {
            index: {
                "route_marker": f"marker-{index + 1}",
                "phase": "marked_for_hypothesis_revision",
            }
            for index in range(5)
        }
    )
    with pytest.raises(ContractValidationError, match="当前 result 和 Review"):
        _validate_terminal_marker_session_bindings(
            {"accepted": False, "review_event_ref": "review-1"},
            set(COMBINATION_IDS),
            {"findings": []},
            marked_sessions,
            markers,
        )


def test_terminal_marker_result_bindings_rejects_missing_marker_row() -> None:
    """即使组合集合由调用方保证，绑定 helper 仍对缺失 marker fail closed。"""

    sessions = _terminal_sessions()
    marker_by_combination = {
        item["combination_id"]: item for item in _markers()
    }
    marker_by_combination.pop("comb-1")
    with pytest.raises(ContractValidationError, match="当前 result 和 Review"):
        _validate_terminal_marker_result_bindings(
            {"review_event_ref": "review-1"},
            sessions,
            marker_by_combination,
        )


def test_terminal_marker_references_cover_accepted_and_regeneration_contracts() -> None:
    """用内存中的 marker 字典夹具覆盖 accepted 与假设重生的引用合同。

    输入是两条 accepted 部分引用、无 marker 的空集合、完整五条重生引用及其非法变体；
    动作调用只读的引用校验并断言合法输入通过、缺失/数量不足/来源漂移时抛出合同错误。
    测试不写生产数据库或文件；失败表示终态 marker 引用边界或错误提示发生回归。
    """

    markers = _markers()
    marker_refs = [item["marker_event_id"] for item in markers]
    _validate_terminal_marker_references(
        {"accepted": True, "marker_event_refs": []}, []
    )
    _validate_terminal_marker_references(
        {"accepted": True, "marker_event_refs": marker_refs[:2]}, markers[:2]
    )
    _validate_terminal_marker_references(
        {"accepted": False, "marker_event_refs": []}, []
    )
    _validate_terminal_marker_references(
        {"accepted": False, "marker_event_refs": marker_refs}, markers
    )

    with pytest.raises(ContractValidationError, match="明确引用"):
        _validate_terminal_marker_references(
            {"accepted": True, "marker_event_refs": []}, markers[:1]
        )
    with pytest.raises(ContractValidationError, match="来源与当前索引"):
        _validate_terminal_marker_references(
            {"accepted": True, "marker_event_refs": ["other"]}, markers[:1]
        )
    with pytest.raises(ContractValidationError, match="必须完整引用当前 marker"):
        _validate_terminal_marker_references(
            {"accepted": False, "marker_event_refs": marker_refs[:2]}, markers[:2]
        )
    with pytest.raises(ContractValidationError, match="已有 marker"):
        _validate_terminal_marker_references(
            {"accepted": False, "marker_event_refs": []}, markers[:1]
        )
    with pytest.raises(ContractValidationError, match="来源与当前索引"):
        _validate_terminal_marker_references(
            {"accepted": False, "marker_event_refs": [*marker_refs[:-1], "other"]},
            markers,
        )
