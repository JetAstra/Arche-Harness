from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemistry_multiagent.harness.base import HarnessError
from chemistry_multiagent.harness.cli_protocol import extract_cli_session_id
from chemistry_multiagent.harness.session_store import PlannerSessionStore


def test_extract_cli_session_id_supports_thread_and_nested_session_events() -> None:
    assert (
        extract_cli_session_id(
            '{"type":"thread.started","thread":{"id":"thread-123"}}\n'
        )
        == "thread-123"
    )
    assert extract_cli_session_id('{"type":"session.started","session_id":"s-456"}') == "s-456"
    assert extract_cli_session_id('{"type":"result","id":"response-only"}') is None


def test_session_store_atomic_manifest_and_scope_validation(tmp_path: Path) -> None:
    store = PlannerSessionStore(
        tmp_path / "planner",
        task_scope_id="task-1",
        planner_scope_id="planner-task-1",
    )
    with store.locked():
        saved = store.save(
            store.new_manifest(
                session_id="session-1",
                model="gpt-5.6-sol",
                stage="workflow_generation",
            )
        )
    assert saved["session_id"] == "session-1"
    assert store.manifest_path.stat().st_mode & 0o077 == 0
    assert store.load()["turn_count"] == 1
    with store.locked():
        updated = store.record_turn(store.load(), stage="workflow_revision")
    assert updated["turn_count"] == 2
    assert updated["last_stage"] == "workflow_revision"

    wrong_scope = PlannerSessionStore(
        tmp_path / "planner",
        task_scope_id="other-task",
        planner_scope_id="planner-task-1",
    )
    with pytest.raises(HarnessError, match="task_scope_id"):
        wrong_scope.load()


def test_session_store_rejects_corrupt_manifest(tmp_path: Path) -> None:
    store = PlannerSessionStore(
        tmp_path / "planner",
        task_scope_id="task-1",
        planner_scope_id="planner-task-1",
    )
    store.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    store.manifest_path.write_text(json.dumps({"session_id": "broken"}), encoding="utf-8")
    with pytest.raises(HarnessError, match="missing fields"):
        store.load()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 99, "Unsupported Planner session manifest schema"),
        ("turn_count", -1, "turn_count is invalid"),
        ("status", "", "status is invalid"),
        ("thread_policy_digest", "bad", "thread_policy_digest is malformed"),
    ],
)
def test_session_store_load_rejects_invalid_manifest_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Planner manifest 的 schema、计数、状态和策略摘要都必须可验证。"""

    store = PlannerSessionStore(
        tmp_path / field,
        task_scope_id="task-1",
        planner_scope_id="planner-task-1",
    )
    store.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = store.new_manifest(
        session_id="session-1",
        model="gpt-5.6-sol",
        stage="workflow_generation",
    )
    manifest[field] = value
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HarnessError, match=message):
        store.load()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reason", ["worker_cleanup"], "state or action"),
        ("stage", [], "recovery.stage"),
        ("request_digest", 123, "request_digest"),
        ("turn_id", ["turn-1"], "turn_id"),
        ("turn_count", True, "turn_count"),
    ],
)
def test_planner_recovery_rejects_malformed_json_shapes(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """恢复记录中的 JSON 类型异常必须统一停止恢复。"""

    store = PlannerSessionStore(
        tmp_path / field,
        task_scope_id="task-1",
        planner_scope_id="planner-1",
    )
    policy_digest = "a" * 64
    store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
        thread_policy_digest=policy_digest,
    )
    store.record_recovery_pending(
        stage="planning",
        request_digest="b" * 64,
        thread_policy_digest=policy_digest,
        turn_id="turn-1",
    )
    recovery = json.loads(store.recovery_path.read_text(encoding="utf-8"))
    recovery[field] = value
    store.recovery_path.write_text(json.dumps(recovery), encoding="utf-8")

    with pytest.raises(HarnessError, match=message):
        store.load_recovery()


def _planner_store(root: Path) -> PlannerSessionStore:
    """构造直接测试使用的 legacy Planner 会话存储。"""

    return PlannerSessionStore(
        root,
        task_scope_id="task-1",
        planner_scope_id="planner-1",
    )


def test_planner_validated_turn_can_replace_pending_recovery(
    tmp_path: Path,
) -> None:
    """Planner 只有在新 turn 凭证落盘后才清除旧恢复记录。"""

    store = _planner_store(tmp_path / "planner")
    policy_digest = "a" * 64
    store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
        thread_policy_digest=policy_digest,
    )
    store.record_recovery_pending(
        stage="planning",
        request_digest="b" * 64,
        thread_policy_digest=policy_digest,
        turn_id="turn-old",
    )
    store.record_recovery_pending(
        stage="planning",
        request_digest="b" * 64,
        thread_policy_digest=policy_digest,
        turn_id="turn-old",
    )
    updated = store.record_turn(store.load(), stage="planning", status="active")
    validated = store.record_validated_turn(
        stage="planning",
        request_digest="c" * 64,
        thread_policy_digest=policy_digest,
        turn_id="turn-new",
    )

    assert validated["turn_count"] == updated["turn_count"] == 1
    cleared = store.clear_recovery_after_validated_turn(
        stage="planning",
        thread_policy_digest=policy_digest,
    )
    assert cleared["status"] == "active"
    assert not store.recovery_path.exists()
    assert not store.validated_turn_path.exists()


def test_planner_reconcile_keeps_pending_recovery_without_validated_evidence(
    tmp_path: Path,
) -> None:
    """没有 validated 凭证时恢复只返回阻断状态，不猜测 turn 已完成。"""

    store = _planner_store(tmp_path / "planner")
    policy_digest = "d" * 64
    store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
        thread_policy_digest=policy_digest,
    )
    store.record_recovery_pending(
        stage="planning",
        request_digest="e" * 64,
        thread_policy_digest=policy_digest,
        turn_id="turn-pending",
    )
    current, recovery = store.reconcile_recovery_state(
        thread_policy_digest=policy_digest,
    )
    assert current is not None
    assert current["status"] == "recovery_pending"
    assert recovery is not None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_digest", "bad", "request digest"),
        ("thread_policy_digest", "bad", "thread policy digest"),
        ("turn_id", ["turn-1"], "session_id"),
        ("turn_count_before", -1, "turn_count"),
        ("reason", ["worker_cleanup"], "reason"),
    ],
)
def test_planner_recovery_identity_rejects_malformed_arguments(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Planner 恢复身份参数异常时必须在写盘前停止。"""

    store = _planner_store(tmp_path / field)
    policy_digest = "f" * 64
    store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
        thread_policy_digest=policy_digest,
    )
    arguments: dict[str, object] = {
        "stage": "planning",
        "request_digest": "1" * 64,
        "thread_policy_digest": policy_digest,
        "turn_id": "turn-1",
        "reason": "worker_cleanup",
    }
    arguments[field] = value

    with pytest.raises(HarnessError, match=message):
        store.record_recovery_pending(**arguments)
    assert not store.recovery_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_digest", "bad", "request digest"),
        ("thread_policy_digest", "bad", "policy digest"),
        ("turn_id", ["turn-1"], "session_id"),
    ],
)
def test_planner_validated_turn_rejects_malformed_arguments(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Planner validated 凭证参数异常时不应创建凭证文件。"""

    store = _planner_store(tmp_path / field)
    store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
    )
    arguments: dict[str, object] = {
        "stage": "planning",
        "request_digest": "2" * 64,
        "thread_policy_digest": "3" * 64,
        "turn_id": "turn-1",
    }
    arguments[field] = value

    with pytest.raises(HarnessError, match=message):
        store.record_validated_turn(**arguments)
    assert not store.validated_turn_path.exists()


def test_planner_validated_turn_requires_existing_manifest_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    """validated turn 必须绑定已有 manifest，重复身份之外的凭证会被拒绝。"""

    missing = _planner_store(tmp_path / "missing")
    with pytest.raises(HarnessError, match="before creating"):
        missing.record_validated_turn(
            stage="planning",
            request_digest="4" * 64,
            thread_policy_digest="5" * 64,
            turn_id="turn-1",
        )

    store = _planner_store(tmp_path / "conflict")
    store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
    )
    first = store.record_validated_turn(
        stage="planning",
        request_digest="6" * 64,
        thread_policy_digest="7" * 64,
        turn_id="turn-1",
    )
    assert first["turn_count"] == 0
    with pytest.raises(HarnessError, match="conflicts"):
        store.record_validated_turn(
            stage="planning",
            request_digest="8" * 64,
            thread_policy_digest="7" * 64,
            turn_id="turn-1",
        )
