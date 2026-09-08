from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from chemistry_multiagent.harness import CodexSessionOwner, CodexSessionStore, PlannerSessionStore
from chemistry_multiagent.harness.base import HarnessError
from chemistry_multiagent.harness import session_store as session_store_module


POLICY_DIGEST = "e" * 64


def test_session_lock_fails_closed_when_fcntl_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    """固定 Linux 运行环境缺少 fcntl 时不能静默进入无锁 turn。"""

    store = PlannerSessionStore(
        tmp_path / "planner",
        task_scope_id="task-1",
        planner_scope_id="planner-1",
    )
    original_import = __import__

    def reject_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("simulated missing fcntl")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_fcntl)
    with pytest.raises(HarnessError, match="fcntl"):
        with store.locked():
            pass


def test_directory_fsync_failure_is_not_ignored(tmp_path: Path, monkeypatch) -> None:
    """manifest 原子替换后的目录刷盘失败必须让调用方看到状态错误。"""

    store = _investigation_store(tmp_path)
    original_fsync = session_store_module.os.fsync

    def fail_fsync(_descriptor):
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(session_store_module.os, "fsync", fail_fsync)
    with pytest.raises(HarnessError, match="cannot be saved"):
        store.create("thread-1")
    monkeypatch.setattr(session_store_module.os, "fsync", original_fsync)
    assert not store.manifest_path.exists()


def test_codex_manifest_persists_policy_bound_by_resume_reconcile(tmp_path: Path) -> None:
    """首次 reconcile 绑定的策略摘要必须进入 manifest，恢复漂移立即停止。"""

    store = _investigation_store(tmp_path)
    assert store.reconcile_recovery_state(thread_policy_digest=POLICY_DIGEST) == (None, None)
    created = store.create("thread-policy", stage="initial_investigation")
    assert created["thread_policy_digest"] == POLICY_DIGEST

    restarted = _investigation_store(tmp_path)
    assert restarted.reconcile_recovery_state(thread_policy_digest=POLICY_DIGEST)[0] == created
    with pytest.raises(HarnessError, match="policy"):
        restarted.reconcile_recovery_state(thread_policy_digest="f" * 64)


def test_planner_manifest_persists_policy_bound_by_resume_reconcile(tmp_path: Path) -> None:
    """legacy Planner 的首建和后续恢复也必须绑定同一 effective policy。"""

    store = PlannerSessionStore(
        tmp_path / "planner",
        task_scope_id="task-1",
        planner_scope_id="planner-1",
    )
    assert store.reconcile_recovery_state(thread_policy_digest=POLICY_DIGEST) == (None, None)
    created = store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
    )
    assert created["thread_policy_digest"] == POLICY_DIGEST

    restarted = PlannerSessionStore(
        tmp_path / "planner",
        task_scope_id="task-1",
        planner_scope_id="planner-1",
    )
    assert restarted.reconcile_recovery_state(thread_policy_digest=POLICY_DIGEST)[0] == created
    with pytest.raises(HarnessError, match="policy"):
        restarted.reconcile_recovery_state(thread_policy_digest="f" * 64)


def _investigation_store(
    root: Path,
    *,
    research_task_id: str = "task-1",
    investigation_context_id: str = "batch-1",
    hypothesis_execution_input_record_id: str = "combination-1",
    scope_id: str = "scope-1",
    model: str = "model-1",
) -> CodexSessionStore:
    return CodexSessionStore.from_root(
        root,
        owner=CodexSessionOwner.investigation(
            research_task_id,
            investigation_context_id,
            hypothesis_execution_input_record_id,
        ),
        scope_id=scope_id,
        model=model,
    )


def _create_legacy_request_intent_recovery_store(
    root: Path,
    *,
    request_digest: str = "a" * 64,
) -> tuple[CodexSessionStore, str]:
    """创建带 request-intent 阻断的 Codex store，供迁移失败路径复用。"""

    store = _investigation_store(root)
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    store.record_recovery_pending(
        stage="investigation",
        request_digest=request_digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id=None,
        reason="request_intent",
    )
    return store, request_digest


def test_investigation_manifest_binds_owner_workspace_scope_model_and_thread(
    tmp_path: Path,
) -> None:
    """组合调查会话保存唯一归属身份及其所有运行绑定。"""

    store = _investigation_store(tmp_path)
    created = store.create("thread-1", stage="initial_investigation")
    restarted = _investigation_store(tmp_path)
    loaded = restarted.load()

    assert loaded == created
    assert created["status"] == "created"
    assert created["turn_count"] == 0
    assert set(created) == {
        "schema_version",
        "owner",
        "workspace",
        "scope_id",
        "model",
        "codex_home",
        "thread_id",
        "last_stage",
        "turn_count",
        "status",
        "created_at",
        "updated_at",
    }
    assert created["owner"] == {
        "kind": "investigation",
        "research_task_id": "task-1",
        "investigation_context_id": "batch-1",
        "hypothesis_execution_input_record_id": "combination-1",
    }
    assert created["thread_id"] == "thread-1"
    assert created["scope_id"] == "scope-1"
    assert created["model"] == "model-1"
    assert created["workspace"] == str(store.workspace)
    assert created["codex_home"] == str(store.codex_home)
    assert "owner_identity" not in created
    assert "owner_id" not in created
    assert "session_id" not in created
    assert "codex_thread_id" not in created
    assert "review_thread_id" not in created
    owner_root = (
        tmp_path / "investigations" / "task-1" / "batch-1" / "combination-1"
    ).resolve()
    assert store.workspace == owner_root / "workspace"
    assert store.codex_home == owner_root / "codex_home"
    assert store.manifest_path == owner_root / "state" / ".arche_codex_session.json"
    assert store.workspace.parent == store.codex_home.parent == store.manifest_path.parent.parent
    assert store.workspace not in store.codex_home.parents
    assert store.codex_home not in store.workspace.parents
    assert store.workspace not in store.manifest_path.parents
    assert store.manifest_path.parent.stat().st_mode & 0o777 == 0o700
    assert owner_root.stat().st_mode & 0o777 == 0o700
    assert store.workspace.stat().st_mode & 0o777 == 0o700
    assert store.codex_home.stat().st_mode & 0o777 == 0o700
    assert store.manifest_path.stat().st_mode & 0o777 == 0o600
    assert store.lock_path.stat().st_mode & 0o777 == 0o600


def test_schema_v1_owner_is_read_then_rewritten_with_canonical_identity(
    tmp_path: Path,
) -> None:
    """旧 owner 只在显式 reader 中读取，下一次写入改成 schema-v2。"""

    store = _investigation_store(tmp_path)
    created = store.create("thread-legacy-owner")
    legacy = {
        **created,
        "schema_version": 1,
        "owner": {
            "kind": "investigation",
            "task_id": "task-1",
            "batch_id": "batch-1",
            "combination_id": "combination-1",
        },
    }
    store.manifest_path.write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = store.load()
    assert loaded is not None
    assert loaded["schema_version"] == 2
    assert loaded["owner"] == {
        "kind": "investigation",
        "research_task_id": "task-1",
        "investigation_context_id": "batch-1",
        "hypothesis_execution_input_record_id": "combination-1",
    }

    rewritten = store.record_completed_turn(stage="initial_investigation")
    persisted = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert persisted == rewritten
    assert persisted["schema_version"] == 2
    assert "task_id" not in persisted["owner"]
    assert "batch_id" not in persisted["owner"]
    assert "combination_id" not in persisted["owner"]


def test_different_investigation_contexts_and_execution_inputs_get_new_sessions(
    tmp_path: Path,
) -> None:
    """不同 Investigation Context 或执行输入派生独立会话目录。"""

    first = _investigation_store(tmp_path)
    second_context = _investigation_store(
        tmp_path,
        investigation_context_id="batch-2",
    )
    second_execution_input = _investigation_store(
        tmp_path,
        hypothesis_execution_input_record_id="combination-2",
    )

    assert first.workspace != second_context.workspace
    assert first.workspace != second_execution_input.workspace
    assert first.codex_home != second_context.codex_home
    assert first.manifest_path != second_context.manifest_path

    first.create("thread-1")
    second_context.create("thread-2")
    second_execution_input.create("thread-3")
    assert first.load()["thread_id"] == "thread-1"
    assert second_context.load()["thread_id"] == "thread-2"
    assert second_execution_input.load()["thread_id"] == "thread-3"


def test_task_review_is_unique_for_task_across_batches(tmp_path: Path) -> None:
    """任务审查的归属身份只含 task，整个任务生命周期复用同一 thread。"""

    owner = CodexSessionOwner.task_review("task-1")
    first = CodexSessionStore.from_root(
        tmp_path,
        owner=owner,
        scope_id="review-scope",
        model="review-model",
    )
    resumed = CodexSessionStore.from_root(
        tmp_path,
        owner=CodexSessionOwner.task_review("task-1"),
        scope_id="review-scope",
        model="review-model",
    )
    different_task = CodexSessionStore.from_root(
        tmp_path,
        owner=CodexSessionOwner.task_review("task-2"),
        scope_id="review-scope",
        model="review-model",
    )

    assert first.workspace == resumed.workspace
    assert first.codex_home == resumed.codex_home
    assert first.workspace != different_task.workspace
    assert first.workspace.parent == first.codex_home.parent == first.manifest_path.parent.parent
    assert first.workspace.name == "workspace"
    assert first.codex_home.name == "codex_home"
    assert first.manifest_path.parent.name == "state"
    assert first.workspace not in first.codex_home.parents
    assert first.codex_home not in first.workspace.parents
    assert first.workspace not in first.manifest_path.parents
    first.create("review-thread")
    assert resumed.load()["thread_id"] == "review-thread"


def test_owner_and_binding_conflicts_fail_closed(tmp_path: Path) -> None:
    """会话归属、scope、model 或 thread 发生漂移时不得加载或覆盖记录。"""

    shared = tmp_path / "shared"
    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    CodexSessionStore(
        shared,
        owner=owner,
        scope_id="scope-1",
        model="model-1",
    ).create("thread-1")

    with pytest.raises(HarnessError, match="identity conflict"):
        CodexSessionStore(
            shared,
            owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-2"),
            scope_id="scope-1",
            model="model-1",
        ).load()
    with pytest.raises(HarnessError, match="identity conflict"):
        CodexSessionStore(
            shared,
            owner=owner,
            scope_id="scope-2",
            model="model-1",
        ).load()
    with pytest.raises(HarnessError, match="identity conflict"):
        CodexSessionStore(
            shared,
            owner=owner,
            scope_id="scope-1",
            model="model-2",
        ).load()

    same_store = CodexSessionStore(
        shared,
        owner=owner,
        scope_id="scope-1",
        model="model-1",
    )
    with pytest.raises(HarnessError, match="identity conflict"):
        same_store.create("thread-2")
    assert same_store.create("thread-1")["thread_id"] == "thread-1"


def test_record_completed_turn_updates_only_canonical_state(tmp_path: Path) -> None:
    """完成 turn 后保留原 thread，更新阶段、状态和计数。"""

    store = _investigation_store(tmp_path)
    created = store.create("thread-1")
    updated = store.record_completed_turn(stage="job_result", status="completed")

    assert updated["thread_id"] == created["thread_id"]
    assert created["turn_count"] == 0
    assert updated["turn_count"] == 1
    assert updated["last_stage"] == "job_result"
    assert updated["status"] == "completed"
    persisted = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert set(persisted) == set(updated)


def test_competing_threads_create_one_complete_manifest(tmp_path: Path) -> None:
    """并发首次创建只保留一个 thread，失败方不会覆盖成功方。"""

    stores = [_investigation_store(tmp_path) for _ in range(2)]
    results: list[dict[str, object]] = []
    errors: list[HarnessError] = []

    def create(store: CodexSessionStore, thread_id: str) -> None:
        try:
            results.append(store.create(thread_id))
        except HarnessError as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=create, args=(store, f"thread-{index}"))
        for index, store in enumerate(stores)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert "identity conflict" in str(errors[0])
    persisted = json.loads(stores[0].manifest_path.read_text(encoding="utf-8"))
    assert persisted["thread_id"] == results[0]["thread_id"]
    assert (
        persisted["owner"]["hypothesis_execution_input_record_id"]
        == "combination-1"
    )
    assert stores[0].manifest_path.stat().st_mode & 0o077 == 0


def test_locked_covers_nested_store_operations(tmp_path: Path) -> None:
    """runner 可以持锁覆盖整个 turn，再调用 create 或完成记录。"""

    store = _investigation_store(tmp_path)
    with store.locked():
        created = store.create("thread-1")
        updated = store.record_completed_turn(stage="resume")

    assert created["thread_id"] == updated["thread_id"] == "thread-1"
    assert created["turn_count"] == 0
    assert updated["turn_count"] == 1


def test_corrupt_manifest_is_not_replaced(tmp_path: Path) -> None:
    """损坏的 manifest 不能被 create 当作新会话覆盖。"""

    store = _investigation_store(tmp_path)
    store.manifest_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(HarnessError, match="cannot be read"):
        store.load()
    with pytest.raises(HarnessError, match="cannot be read"):
        store.create("thread-1")
    assert store.manifest_path.read_text(encoding="utf-8") == "{not-json"


def test_reconcile_recovery_written_before_manifest(tmp_path: Path, monkeypatch) -> None:
    """恢复记录先落盘而 manifest 尚未更新时，重启可补成 recovery_pending。"""

    store = CodexSessionStore.from_root(
        tmp_path,
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="model-1",
        thread_policy_digest=POLICY_DIGEST,
    )
    store.create("thread-1")
    original_write = store._write_atomic

    def fail_manifest(_manifest):
        raise HarnessError("simulated manifest crash", kind="session_state")

    monkeypatch.setattr(store, "_write_atomic", fail_manifest)
    with pytest.raises(HarnessError, match="manifest crash"):
        store.record_recovery_pending(
            stage="investigation",
            request_digest="a" * 64,
            turn_id="turn-1",
            thread_policy_digest=POLICY_DIGEST,
        )
    monkeypatch.setattr(store, "_write_atomic", original_write)

    manifest, recovery = store.reconcile_recovery_state()
    assert manifest is not None
    assert manifest["status"] == "recovery_pending"
    assert recovery is not None
    assert recovery["reason"] == "worker_cleanup"
    again, same_recovery = store.reconcile_recovery_state()
    assert again == manifest
    assert same_recovery == recovery


def test_reconcile_manifest_written_before_recovery_removal(tmp_path: Path, monkeypatch) -> None:
    """新 turn manifest 已写而旧恢复记录未删时，重启可幂等清理旧记录。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    store.record_recovery_pending(
        stage="investigation",
        request_digest="b" * 64,
        turn_id="turn-1",
        reason="completion_receipt",
        thread_policy_digest=POLICY_DIGEST,
    )
    store.record_completed_turn(stage="investigation", status="completed")
    store.record_validated_turn(
        stage="investigation",
        request_digest="c" * 64,
        turn_id="turn-validated",
        thread_policy_digest=POLICY_DIGEST,
    )
    original_remove = store._remove_recovery_record

    def fail_remove():
        raise HarnessError("simulated recovery cleanup crash", kind="session_state")

    monkeypatch.setattr(store, "_remove_recovery_record", fail_remove)
    with pytest.raises(HarnessError, match="cleanup crash"):
        store.clear_recovery_after_validated_turn(
            stage="investigation",
            thread_policy_digest=POLICY_DIGEST,
        )
    monkeypatch.setattr(store, "_remove_recovery_record", original_remove)

    manifest, recovery = store.reconcile_recovery_state()
    assert manifest is not None
    assert manifest["status"] == "completed"
    assert manifest["turn_count"] == 1
    assert recovery is None
    assert store.load_recovery() is None
    assert store.reconcile_recovery_state() == (manifest, None)


def test_reconcile_keeps_recovery_without_durable_contract_evidence(tmp_path: Path) -> None:
    """仅有 completed manifest 不能证明响应通过合同，恢复记录必须保留。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1")
    store.record_recovery_pending(
        stage="investigation",
        request_digest="c" * 64,
        turn_id="turn-1",
        thread_policy_digest=POLICY_DIGEST,
    )
    store.confirm_worker_stopped_for_recovery(
        request_digest="c" * 64,
        turn_id="turn-1",
        thread_policy_digest=POLICY_DIGEST,
    )
    store.record_completed_turn(stage="investigation", status="completed")

    manifest, recovery = store.reconcile_recovery_state()

    assert manifest is not None
    assert manifest["status"] == "completed"
    assert recovery is not None
    assert recovery["reason"] == "worker_cleanup"


def test_confirmed_worker_contract_failure_converts_recovery_state(tmp_path: Path) -> None:
    """worker 确认后新请求完成但合同失败时，持久状态必须改为 contract_failure。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1")
    old_digest = "d" * 64
    new_digest = "e" * 64
    store.record_recovery_pending(
        stage="investigation",
        request_digest=old_digest,
        turn_id="turn-old",
        thread_policy_digest=POLICY_DIGEST,
    )
    store.confirm_worker_stopped_for_recovery(
        request_digest=old_digest,
        turn_id="turn-old",
        thread_policy_digest=POLICY_DIGEST,
    )
    store.record_completed_turn(stage="investigation", status="completed")

    store.record_contract_failure(
        stage="investigation",
        request_digest=new_digest,
        turn_id="turn-new",
        thread_policy_digest=POLICY_DIGEST,
    )

    manifest = store.load()
    recovery = store.load_recovery()
    assert manifest["status"] == "contract_failure"
    assert manifest["turn_count"] == 1
    assert recovery is not None
    assert recovery["state"] == "contract_failure"
    assert recovery["reason"] == "response_contract"
    assert recovery["request_digest"] == new_digest
    assert recovery["turn_id"] == "turn-new"


def test_clear_recovery_requires_validated_turn_evidence(tmp_path: Path) -> None:
    """没有合同通过凭证时，清理恢复记录必须失败关闭。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1")
    store.record_recovery_pending(
        stage="investigation",
        request_digest="f" * 64,
        turn_id="turn-1",
        reason="completion_receipt",
        thread_policy_digest=POLICY_DIGEST,
    )
    store.record_completed_turn(stage="investigation", status="completed")

    with pytest.raises(HarnessError, match="validated"):
        store.clear_recovery_after_validated_turn(
            stage="investigation",
            thread_policy_digest=POLICY_DIGEST,
        )

    assert store.load_recovery() is not None


def test_codex_validated_turn_is_idempotent_and_requires_completed_manifest(
    tmp_path: Path,
) -> None:
    """Codex validated 凭证同身份幂等，未完成 manifest 则拒绝写入。"""

    store = _investigation_store(tmp_path / "valid")
    store.create("thread-1")
    store.record_completed_turn(stage="investigation", status="completed")
    arguments = {
        "stage": "investigation",
        "request_digest": "a" * 64,
        "thread_policy_digest": POLICY_DIGEST,
        "turn_id": "turn-validated",
    }
    first = store.record_validated_turn(**arguments)
    assert store.record_validated_turn(**arguments) == first

    missing_completion = _investigation_store(tmp_path / "created")
    missing_completion.create("thread-1")
    with pytest.raises(HarnessError, match="completed manifest"):
        missing_completion.record_validated_turn(**arguments)


def test_completed_receipt_replay_clears_later_request_intent_recovery(
    tmp_path: Path,
) -> None:
    """旧成功回合重放后可清除同一计数上的后续失败 Resume 阻断。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    store.record_completed_turn(stage="investigation_and_experiment", status="completed")
    store.record_recovery_pending(
        stage="initial_investigation",
        request_digest="b" * 64,
        turn_id=None,
        reason="request_intent",
        thread_policy_digest=POLICY_DIGEST,
    )

    saved = store.clear_request_intent_recovery_after_replayed_receipt(
        stage="investigation_job_continuation",
        thread_id="thread-1",
        receipt_turn_count_before=0,
        thread_policy_digest=POLICY_DIGEST,
    )

    assert saved["status"] == "completed"
    assert saved["turn_count"] == 1
    assert saved["last_stage"] == "investigation_job_continuation"
    assert store.load_recovery() is None


def test_reconcile_rejects_recovery_when_effective_thread_policy_drifts(
    tmp_path: Path,
) -> None:
    """恢复记录绑定旧 thread policy 时，当前配置漂移必须停止恢复。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1")
    store.record_recovery_pending(
        stage="investigation",
        request_digest="1" * 64,
        turn_id="turn-1",
        thread_policy_digest=POLICY_DIGEST,
    )

    with pytest.raises(HarnessError, match="policy"):
        store.reconcile_recovery_state(thread_policy_digest="f" * 64)

    recovery = store.load_recovery()
    assert recovery is not None
    assert recovery["thread_policy_digest"] == POLICY_DIGEST


def test_codex_recovery_poison_is_durable_idempotent_and_conflict_checked(
    tmp_path: Path,
) -> None:
    """恢复双写失败标记必须可重启读取、同身份幂等且拒绝身份漂移。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    first = store.record_recovery_poison(
        stage="investigation",
        request_digest="a" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-poison",
        reason="worker_cleanup",
    )
    assert first["state"] == "poisoned"
    assert store.load_recovery_poison() == first
    assert store.record_recovery_poison(
        stage="investigation",
        request_digest="a" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-poison",
        reason="worker_cleanup",
    ) == first
    with pytest.raises(HarnessError, match="poison conflicts"):
        store.record_recovery_poison(
            stage="investigation",
            request_digest="b" * 64,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-poison",
            reason="worker_cleanup",
        )


@pytest.mark.parametrize(
    "reason",
    ["worker_cleanup", "completion_receipt", "request_intent", "terminal_turn", "goal"],
)
def test_codex_recovery_pending_reason_actions_are_explicit_and_idempotent(
    tmp_path: Path,
    reason: str,
) -> None:
    """每种恢复原因都保存明确动作，重复写入同一身份不改变事实。"""

    store = _investigation_store(tmp_path / reason)
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    store.record_recovery_pending(
        stage="investigation",
        request_digest="c" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-recovery",
        reason=reason,
    )
    store.record_recovery_pending(
        stage="investigation",
        request_digest="c" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-recovery",
        reason=reason,
    )
    expected_actions = {
        "worker_cleanup": "worker_stop_confirmation_required",
        "completion_receipt": "receipt_replay_required",
        "request_intent": "request_intent_confirmation_required",
        "terminal_turn": "terminal_acknowledgement_required",
        "goal": "goal_recovery_required",
    }
    recovery = store.load_recovery()
    assert recovery is not None
    assert recovery["action"] == expected_actions[reason]
    assert store.load()["status"] == "recovery_pending"


def test_codex_goal_recovery_action_cannot_impersonate_another_recovery_reason(
    tmp_path: Path,
) -> None:
    """Goal 恢复动作只能绑定 goal 原因，普通清理不能复用该动作。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    identity = {
        "stage": "initial_investigation",
        "request_digest": "b" * 64,
        "thread_policy_digest": POLICY_DIGEST,
        "turn_id": "turn-goal",
    }
    store.record_recovery_pending(**identity, reason="goal")
    saved = store.load_recovery()
    assert saved is not None
    assert saved["reason"] == "goal"
    assert saved["action"] == "goal_recovery_required"
    assert store.load_recovery() == saved
    with pytest.raises(HarnessError, match="conflicts"):
        store.record_recovery_pending(**identity, reason="worker_cleanup")


def test_codex_thread_rollback_intent_and_receipt_survive_restart_idempotently(
    tmp_path: Path,
) -> None:
    """Investigation wake 的 thread rollback intent/receipt 必须可重启续跑且同身份幂等。"""

    store = _investigation_store(tmp_path)
    store.create("thread-wake", thread_policy_digest=POLICY_DIGEST)
    identity = {
        "stage": "investigation_job_wake",
        "thread_id": "thread-wake",
        "request_digest": "d" * 64,
        "thread_policy_digest": POLICY_DIGEST,
        "turn_count": 0,
        "physical_turn_count": 3,
        "provider_error_code": "invalid_encrypted_content",
    }
    intent = store.record_thread_rollback_intent(**identity)
    assert store.load_thread_rollback_intent() == intent

    restarted = _investigation_store(tmp_path)
    assert restarted.load_thread_rollback_intent() == intent
    receipt = restarted.record_thread_rollback_receipt(**identity)
    assert restarted.load_thread_rollback_receipt() == receipt
    assert restarted.record_thread_rollback_receipt(**identity) == receipt
    assert restarted.load()["thread_id"] == "thread-wake"
    assert restarted.load()["turn_count"] == 0


def test_codex_thread_rollback_intent_rejects_unapproved_provider_code(
    tmp_path: Path,
) -> None:
    """rollback 持久记录不得接受白名单之外的 provider 文本或代码。"""

    store = _investigation_store(tmp_path)
    store.create("thread-wake", thread_policy_digest=POLICY_DIGEST)
    with pytest.raises(HarnessError, match="error code"):
        store.record_thread_rollback_intent(
            stage="investigation_job_wake",
            thread_id="thread-wake",
            request_digest="d" * 64,
            thread_policy_digest=POLICY_DIGEST,
            turn_count=0,
            physical_turn_count=3,
            provider_error_code="provider free text",
        )


def test_codex_initial_investigation_rollback_intent_survives_restart(
    tmp_path: Path,
) -> None:
    """首次 Investigation turn 也使用同一份 rollback receipt，重启后保持幂等。"""

    store = _investigation_store(tmp_path)
    store.create("thread-initial", thread_policy_digest=POLICY_DIGEST)
    identity = {
        "stage": "initial_investigation",
        "thread_id": "thread-initial",
        "request_digest": "e" * 64,
        "thread_policy_digest": POLICY_DIGEST,
        "turn_count": 0,
        "physical_turn_count": 1,
        "provider_error_code": "invalid_encrypted_content",
    }
    intent = store.record_thread_rollback_intent(**identity)
    restarted = _investigation_store(tmp_path)
    assert restarted.load_thread_rollback_intent() == intent
    receipt = restarted.record_thread_rollback_receipt(**identity)
    assert receipt["stage"] == "initial_investigation"
    assert restarted.record_thread_rollback_receipt(**identity) == receipt


def test_terminal_recovery_dominates_worker_cleanup_and_ack_keeps_success_count(
    tmp_path: Path,
) -> None:
    """同一 turn 的终态事实替换 worker 清理，ack 后不增加成功 turn 计数。"""

    store = _investigation_store(tmp_path)
    created = store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    assert store.acknowledge_terminal_turn_recovery(
        request_digest="3" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-without-recovery",
    ) == created
    digest = "4" * 64
    identity = {
        "stage": "investigation",
        "request_digest": digest,
        "thread_policy_digest": POLICY_DIGEST,
        "turn_id": "turn-terminal",
    }
    store.record_recovery_pending(**identity, reason="worker_cleanup")
    store.record_recovery_pending(**identity, reason="terminal_turn")
    # 晚到的 worker timeout 只能保留较强终态义务，不能制造冲突或 poison。
    store.record_recovery_pending(**identity, reason="worker_cleanup")

    recovery = store.load_recovery()
    assert recovery is not None
    assert recovery["reason"] == "terminal_turn"
    assert recovery["action"] == "terminal_acknowledgement_required"
    with pytest.raises(HarnessError, match="identity"):
        store.acknowledge_terminal_turn_recovery(
            request_digest="5" * 64,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
        )
    acknowledged = store.acknowledge_terminal_turn_recovery(
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-terminal",
    )
    assert acknowledged["turn_count"] == created["turn_count"] == 0
    assert acknowledged["status"] == "created"
    assert store.load_recovery() is None

    store.record_completed_turn(stage="investigation", status="completed")
    completed_identity = {
        **identity,
        "request_digest": "6" * 64,
        "turn_id": "turn-terminal-after-success",
    }
    store.record_recovery_pending(**completed_identity, reason="terminal_turn")
    after_success = store.acknowledge_terminal_turn_recovery(
        request_digest="6" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-terminal-after-success",
    )
    assert after_success["turn_count"] == 1
    assert after_success["status"] == "completed"


def test_terminal_recovery_uses_legacy_request_intent_migration_api(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """终态写入遇到旧 request-intent 时必须调用迁移 API 并保留新 turn 身份。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    old_digest = "a" * 64
    calls: list[dict[str, object]] = []
    original_migrate = store.migrate_request_intent_recovery_to_terminal_turn

    def record_migration(**kwargs):
        calls.append(dict(kwargs))
        return original_migrate(**kwargs)

    monkeypatch.setattr(store, "migrate_request_intent_recovery_to_terminal_turn", record_migration)
    store.record_recovery_pending(
        stage="investigation",
        request_digest=old_digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id=None,
        reason="request_intent",
    )

    migrated = store.record_recovery_pending(
        stage="investigation",
        request_digest=old_digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-terminal",
        reason="terminal_turn",
    )

    assert calls == [
        {
            "stage": "investigation",
            "request_digest": old_digest,
            "thread_policy_digest": POLICY_DIGEST,
            "turn_id": "turn-terminal",
            "legacy_recovery_request_digest": old_digest,
        }
    ]
    assert migrated["status"] == "recovery_pending"
    recovery = store.load_recovery()
    assert recovery is not None
    assert recovery["reason"] == "terminal_turn"
    assert recovery["turn_id"] == "turn-terminal"


def test_legacy_request_intent_migration_api_is_idempotent_and_identity_bound(
    tmp_path: Path,
) -> None:
    """直接调用迁移 API 时同一终态幂等，turn 身份漂移则拒绝。"""

    store, digest = _create_legacy_request_intent_recovery_store(tmp_path)
    migrated = store.migrate_request_intent_recovery_to_terminal_turn(
        stage="investigation",
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-terminal",
    )
    assert store.migrate_request_intent_recovery_to_terminal_turn(
        stage="investigation",
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-terminal",
    ) == migrated
    with pytest.raises(HarnessError, match="identity"):
        store.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest=digest,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-other",
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("stage", "other-stage", "stage"),
        ("request_digest", "b" * 64, "digest"),
        ("thread_policy_digest", "f" * 64, "policy"),
        ("turn_id", "turn-other", "turn id"),
    ],
)
def test_legacy_request_intent_migration_rejects_pending_identity_drift(
    tmp_path: Path,
    argument: str,
    value: str,
    message: str,
) -> None:
    """旧 pending 记录的任一请求身份漂移都不能升级为 terminal。"""

    store, digest = _create_legacy_request_intent_recovery_store(tmp_path / argument)
    if argument == "turn_id":
        recovery = store._read_recovery_record()
        assert recovery is not None
        recovery["turn_id"] = "turn-old"
        store._write_recovery_record(recovery)
    arguments: dict[str, object] = {
        "stage": "investigation",
        "request_digest": digest,
        "thread_policy_digest": POLICY_DIGEST,
        "turn_id": "turn-terminal",
    }
    arguments[argument] = value

    with pytest.raises(HarnessError, match=message):
        store.migrate_request_intent_recovery_to_terminal_turn(**arguments)
    recovery = store.load_recovery()
    assert recovery is not None
    assert recovery["reason"] == "request_intent"


def test_legacy_request_intent_migration_rejects_count_and_old_digest_drift(
    tmp_path: Path,
) -> None:
    """物理 turn 计数或调用方提供的旧摘要漂移时保留旧恢复记录。"""

    store, digest = _create_legacy_request_intent_recovery_store(tmp_path / "count")
    recovery = store._read_recovery_record()
    assert recovery is not None
    recovery["turn_count"] = 1
    store._write_recovery_record(recovery)
    with pytest.raises(HarnessError, match="turn count"):
        store.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest=digest,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
        )

    store, digest = _create_legacy_request_intent_recovery_store(tmp_path / "digest")
    with pytest.raises(HarnessError, match="supplied old identity"):
        store.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest=digest,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
            legacy_recovery_request_digest="b" * 64,
        )


def test_legacy_request_intent_migration_rejects_malformed_old_digest(
    tmp_path: Path,
) -> None:
    """调用方携带的旧摘要格式错误时在锁和写盘前拒绝。"""

    store, digest = _create_legacy_request_intent_recovery_store(tmp_path)
    with pytest.raises(HarnessError, match="malformed"):
        store.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest=digest,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
            legacy_recovery_request_digest="bad",
        )


def test_legacy_request_intent_migration_rejects_missing_manifest_or_recovery(
    tmp_path: Path,
) -> None:
    """缺少 session manifest 或恢复记录时不能凭请求身份创建 terminal 事实。"""

    missing_manifest = _investigation_store(tmp_path / "manifest")
    with pytest.raises(HarnessError, match="missing"):
        missing_manifest.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest="a" * 64,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
        )

    missing_recovery = _investigation_store(tmp_path / "recovery")
    missing_recovery.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    with pytest.raises(HarnessError, match="missing"):
        missing_recovery.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest="a" * 64,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
        )


def test_legacy_request_intent_migration_rejects_non_pending_manifest_and_action(
    tmp_path: Path,
) -> None:
    """manifest 非 pending 或旧动作已确认时都不能再升级旧 request-intent。"""

    store = _investigation_store(tmp_path / "manifest")
    store.create("thread-1", thread_policy_digest=POLICY_DIGEST)
    store.record_contract_failure(
        stage="investigation",
        request_digest="a" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-contract",
    )
    current = store.load()
    assert current is not None
    current["status"] = "recovery_pending"
    store._write_atomic(current)
    with pytest.raises(HarnessError, match="reason"):
        store.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest="a" * 64,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
        )

    store, digest = _create_legacy_request_intent_recovery_store(tmp_path / "action")
    store.confirm_request_intent_without_new_turn(
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id=None,
    )
    with pytest.raises(HarnessError, match="action"):
        store.migrate_request_intent_recovery_to_terminal_turn(
            stage="investigation",
            request_digest=digest,
            thread_policy_digest=POLICY_DIGEST,
            turn_id="turn-terminal",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stage", "", "last_stage"),
        ("request_digest", "bad", "request digest"),
        ("thread_policy_digest", "bad", "thread_policy_digest"),
        ("turn_id", ["turn-1"], "thread_id"),
        ("reason", ["worker_cleanup"], "reason"),
    ],
)
def test_codex_recovery_pending_rejects_malformed_arguments(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """恢复参数的类型和摘要异常必须在写盘前 fail-closed。"""

    store = _investigation_store(tmp_path / field)
    store.create("thread-1")
    arguments: dict[str, object] = {
        "stage": "investigation",
        "request_digest": "c" * 64,
        "thread_policy_digest": POLICY_DIGEST,
        "turn_id": "turn-recovery",
        "reason": "worker_cleanup",
    }
    arguments[field] = value

    with pytest.raises(HarnessError, match=message):
        store.record_recovery_pending(**arguments)
    assert store.load_recovery() is None


def test_codex_contract_failure_recovery_can_be_started_idempotently(
    tmp_path: Path,
) -> None:
    """合同失败必须先显式批准恢复，重复批准只返回同一 pending 状态。"""

    store = _investigation_store(tmp_path)
    store.create("thread-1")
    digest = "d" * 64
    store.record_contract_failure(
        stage="investigation",
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-contract",
    )
    pending = store.begin_contract_failure_recovery(
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-contract",
    )
    assert pending["status"] == "recovery_pending"
    assert store.begin_contract_failure_recovery(
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-contract",
    ) == pending


def test_codex_request_intent_confirmation_changes_only_required_action(
    tmp_path: Path,
) -> None:
    """request-intent 确认只改变动作字段，不删除恢复记录或增加计数。"""

    store = _investigation_store(tmp_path)
    created = store.create("thread-1")
    digest = "e" * 64
    store.record_recovery_pending(
        stage="investigation",
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-intent",
        reason="request_intent",
    )
    confirmed = store.confirm_request_intent_without_new_turn(
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-intent",
    )
    assert confirmed["turn_count"] == created["turn_count"]
    assert store.load_recovery()["action"] == "explicit_new_request_required"
    assert store.confirm_request_intent_without_new_turn(
        request_digest=digest,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-intent",
    ) == confirmed


def test_legacy_planner_recovery_poison_and_policy_manifest_paths(
    tmp_path: Path,
) -> None:
    """legacy Planner 仍保留 poison、policy 绑定和恢复兼容行为。"""

    store = PlannerSessionStore(
        tmp_path / "planner",
        task_scope_id="task-1",
        planner_scope_id="planner-1",
    )
    created = store.create_thread_manifest(
        session_id="planner-thread",
        model="planner-model",
        stage="planning",
        thread_policy_digest=POLICY_DIGEST,
    )
    assert store.new_manifest(
        session_id="planner-thread-2",
        model="planner-model",
        stage="planning",
        thread_policy_digest=POLICY_DIGEST,
    )["thread_policy_digest"] == POLICY_DIGEST
    poison = store.record_recovery_poison(
        stage="planning",
        request_digest="f" * 64,
        thread_policy_digest=POLICY_DIGEST,
        turn_id="turn-poison",
        reason="worker_cleanup",
        turn_count_before=created["turn_count"],
    )
    assert store.load_recovery_poison() == poison
    with pytest.raises(HarnessError, match="poison"):
        store.reconcile_recovery_state(thread_policy_digest=POLICY_DIGEST)
