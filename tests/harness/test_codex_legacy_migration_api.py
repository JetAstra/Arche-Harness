"""Codex v2 request-intent 终态迁移公开接口回归。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemistry_multiagent.harness import CodexSessionOwner, CodexSessionStore
from chemistry_multiagent.harness import codex_sdk_adapter as adapter
from chemistry_multiagent.harness.base import HarnessRequest
from openai_codex import ApprovalMode, CodexConfig


class _HistoryThread:
    """提供 production adapter 所需的最小完整 history 读取接口。"""

    def __init__(self, thread_id: str, turns: list[dict[str, object]]) -> None:
        self.id = thread_id
        self.turns = turns

    def read(self, *, include_turns: bool) -> dict[str, object]:
        assert include_turns is True
        return {"thread": {"id": self.id, "turns": self.turns}}


class _NoNewTurnHistoryThread(_HistoryThread):
    """在 production recovery 测试中记录并拒绝任何 replacement turn。"""

    def __init__(self, thread_id: str, turns: list[dict[str, object]]) -> None:
        super().__init__(thread_id, turns)
        self.turn_calls = 0

    def turn(self, *_args, **_kwargs):
        self.turn_calls += 1
        raise AssertionError("legacy terminal recovery must not start a new turn")


class _RecoveryCodex:
    """只让生产 adapter 进入 recovery 分支，不允许启动真实 turn。"""

    def __init__(self, _config):
        self.thread = None

    def close(self):
        return None


def _write_v2_intent(root: Path, *, digest: str = "a" * 64) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "thread_id": "thread-1",
        "request_digest": digest,
        "turn_count_before": 0,
        "provider_id": "arche_investigation",
        "env_key": "ARCHE_CODEX_API_KEY",
        "model": "gpt-5.6-sol",
        "role": "investigation",
        "stage": "investigation_and_experiment",
        "profile": "investigation",
        "workspace": str((root / "workspace").resolve()),
        "thread_policy_digest": "b" * 64,
    }
    (root.parent / ".arche_codex_request_intent.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _failed_thread(status: str = "failed") -> _HistoryThread:
    return _HistoryThread("thread-1", [{"id": "turn-1", "status": status}])


def test_v2_intent_migration_writes_v3_and_terminal_journal_idempotently(
    tmp_path: Path,
) -> None:
    """成功迁移保留 A 摘要、物理基线为零，重复调用不增加事实。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    result = adapter.migrate_codex_v2_request_intent_to_terminal_turn(
        receipt_root,
        thread=_failed_thread(),
        thread_id="thread-1",
        request_digest="a" * 64,
    )

    intent = json.loads(
        (tmp_path / ".arche_codex_request_intent.json").read_text(encoding="utf-8")
    )
    journal = json.loads(
        (tmp_path / ".arche_codex_terminal_turn.json").read_text(encoding="utf-8")
    )
    assert intent["schema_version"] == 3
    assert intent["request_digest"] == "a" * 64
    assert intent["physical_turn_count_before"] == 0
    assert journal["turn_id"] == "turn-1"
    assert journal["status"] == "failed"
    assert result["physical_turn_count_before"] == 0
    before = (tmp_path / ".arche_codex_terminal_turn.json").read_bytes()

    repeated = adapter.migrate_codex_v2_request_intent_to_terminal_turn(
        receipt_root,
        thread=_failed_thread(),
        thread_id="thread-1",
        request_digest="a" * 64,
    )
    assert repeated == result
    assert (tmp_path / ".arche_codex_terminal_turn.json").read_bytes() == before


def test_v3_migration_repairs_missing_terminal_journal_without_new_turn(
    tmp_path: Path,
) -> None:
    """已替换 v3 intent 但 journal 缺失时只补物理事实，不创建新 turn。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    thread = _failed_thread()
    first = adapter.migrate_codex_v2_request_intent_to_terminal_turn(
        receipt_root,
        thread=thread,
        thread_id="thread-1",
        request_digest="a" * 64,
        terminal_status="failed",
    )
    journal_path = tmp_path / ".arche_codex_terminal_turn.json"
    journal_path.unlink()

    repaired = adapter.migrate_codex_v2_request_intent_to_terminal_turn(
        receipt_root,
        thread=thread,
        thread_id="thread-1",
        request_digest="a" * 64,
        terminal_status="failed",
    )

    assert repaired == first
    assert json.loads(journal_path.read_text(encoding="utf-8"))["turn_id"] == "turn-1"


def test_v2_migration_infers_interrupted_terminal_status(tmp_path: Path) -> None:
    """生产迁移不应把 interrupted 终态错误当成 failed 或重新提交。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    result = adapter.migrate_codex_v2_request_intent_to_terminal_turn(
        receipt_root,
        thread=_failed_thread("interrupted"),
        thread_id="thread-1",
        request_digest="a" * 64,
    )

    assert result["terminal_status"] == "interrupted"
    journal = json.loads(
        (tmp_path / ".arche_codex_terminal_turn.json").read_text(encoding="utf-8")
    )
    assert journal["status"] == "interrupted"


@pytest.mark.parametrize(
    ("prior_status", "new_status"),
    [("failed", "interrupted"), ("interrupted", "failed")],
)
def test_v2_migration_explicit_baseline_skips_prior_terminal_turn(
    tmp_path: Path,
    prior_status: str,
    new_status: str,
) -> None:
    """显式物理基线允许既有终态，但只能选择基线后的新 turn。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    thread = _HistoryThread(
        "thread-1",
        [
            {"id": "turn-prior-terminal", "status": prior_status},
            {"id": "turn-after-baseline", "status": new_status},
        ],
    )

    result = adapter.migrate_codex_v2_request_intent_to_terminal_turn(
        receipt_root,
        thread=thread,
        thread_id="thread-1",
        request_digest="a" * 64,
        physical_turn_count_before=1,
    )

    assert result["turn_id"] == "turn-after-baseline"
    assert result["terminal_status"] == new_status
    assert result["physical_turn_count_before"] == 1
    journal = json.loads(
        (tmp_path / ".arche_codex_terminal_turn.json").read_text(encoding="utf-8")
    )
    assert journal["turn_id"] == "turn-after-baseline"
    assert journal["status"] == new_status


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("terminal_status", "completed", "terminal status"),
        ("physical_turn_count_before", True, "physical baseline"),
    ],
)
def test_v2_migration_rejects_invalid_boundary_arguments_before_reads(
    tmp_path: Path,
    argument: str,
    value: object,
    message: str,
) -> None:
    """边界参数非法时应先 fail-closed，不读取或写入迁移事实。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    before = intent_path.read_bytes()

    kwargs: dict[str, object] = {argument: value}
    with pytest.raises(adapter.HarnessError, match=message):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=_failed_thread(),
            thread_id="thread-1",
            request_digest="a" * 64,
            **kwargs,
        )

    assert intent_path.read_bytes() == before
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_v2_migration_explicit_baseline_rejects_multiple_new_terminal_turns(
    tmp_path: Path,
) -> None:
    """显式基线后有多个终态 turn 时不得擅自选择第一个。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    before = intent_path.read_bytes()
    thread = _HistoryThread(
        "thread-1",
        [
            {"id": "turn-prior-terminal", "status": "failed"},
            {"id": "turn-after-baseline-1", "status": "failed"},
            {"id": "turn-after-baseline-2", "status": "interrupted"},
        ],
    )

    with pytest.raises(adapter.HarnessError, match="one physical turn"):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=thread,
            thread_id="thread-1",
            request_digest="a" * 64,
            physical_turn_count_before=1,
        )

    assert intent_path.read_bytes() == before
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


@pytest.mark.parametrize(
    "turns",
    [
        [{"status": "failed"}],
        [{"id": "turn/illegal", "status": "failed"}],
        [
            {"id": "turn-duplicate", "status": "failed"},
            {"id": "turn-duplicate", "status": "failed"},
        ],
        [
            {"id": "turn-normal", "status": "failed"},
            {"id": "turn-normal ", "status": "failed"},
        ],
    ],
)
def test_v2_migration_rejects_missing_invalid_duplicate_or_ambiguous_turn_ids(
    tmp_path: Path,
    turns: list[dict[str, object]],
) -> None:
    """history turn 身份缺失、非法、重复或规范化歧义时写盘前拒绝。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    before = intent_path.read_bytes()
    baseline = 1 if len(turns) == 2 else 0

    with pytest.raises(adapter.HarnessError, match="history.*identity"):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=_HistoryThread("thread-1", turns),
            thread_id="thread-1",
            request_digest="a" * 64,
            physical_turn_count_before=baseline,
        )

    assert intent_path.read_bytes() == before
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_v2_intent_migration_digest_drift_has_zero_writes(tmp_path: Path) -> None:
    """恢复 metadata 的 B 摘要不能覆盖 intent 原始 A 摘要。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    before = intent_path.read_bytes()
    with pytest.raises(adapter.HarnessError, match="digest"):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=_failed_thread(),
            thread_id="thread-1",
            request_digest="c" * 64,
        )
    assert intent_path.read_bytes() == before
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


@pytest.mark.parametrize(
    "turns",
    [
        [{"id": "turn-1", "status": "completed"}],
        [
            {"id": "turn-1", "status": "failed"},
            {"id": "turn-2", "status": "failed"},
        ],
        [{"id": "turn-1", "status": "in_progress"}],
    ],
)
def test_v2_intent_migration_rejects_non_unique_failed_history_without_writes(
    tmp_path: Path,
    turns: list[dict[str, object]],
) -> None:
    """非 failed、多个新增或未终结 history 都不能被猜测迁移。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    before = intent_path.read_bytes()
    with pytest.raises(adapter.HarnessError):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=_HistoryThread("thread-1", turns),
            thread_id="thread-1",
            request_digest="a" * 64,
        )
    assert intent_path.read_bytes() == before
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_v2_migration_is_entered_by_production_adapter_before_new_turn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """生产入口遇到 v2 且 history 唯一终态时必须走迁移，而不是提交 replacement。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["provider_id"] = "arche_codex"
    intent_path.write_text(json.dumps(intent), encoding="utf-8")
    thread = _failed_thread()
    monkeypatch.setattr(adapter, "Codex", _RecoveryCodex)
    monkeypatch.setattr(
        adapter,
        "_start_codex_thread",
        lambda **_kwargs: type(
            "Startup",
            (),
            {
                "thread": thread,
                "profile": "investigation",
                "thread_policy_digest": "b" * 64,
                "trace_values": (),
                "approval_mode": ApprovalMode.auto_review,
            },
        )(),
    )
    calls: list[dict[str, object]] = []
    original_migrate = adapter.migrate_codex_v2_request_intent_to_terminal_turn

    def record_migration(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_migrate(*args, **kwargs)

    monkeypatch.setattr(
        adapter,
        "migrate_codex_v2_request_intent_to_terminal_turn",
        record_migration,
    )
    acknowledgements: list[tuple[str, str, str]] = []
    request = HarnessRequest(
        messages=({"role": "user", "content": "legacy request intent"},),
        model="gpt-5.6-sol",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=receipt_root / "workspace",
        timeout_seconds=None,
    )

    with pytest.raises(adapter.CodexTerminalTurnError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-1",
            trace_path=tmp_path / "recovered.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="a" * 64,
            turn_count_before=0,
            on_terminal_turn=lambda *identity: acknowledgements.append(identity),
        )

    assert len(calls) == 1
    assert calls[0]["thread_id"] == "thread-1"
    assert calls[0]["request_digest"] == "a" * 64
    assert acknowledgements == [("thread-1", "turn-1", "failed")]
    assert not (tmp_path / ".arche_codex_request_intent.json").exists()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_production_v2_migration_transitions_legacy_request_intent_recovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """旧 request-intent recovery 迁移后应调用 session API，且不启动 replacement turn。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["provider_id"] = "arche_codex"
    intent_path.write_text(json.dumps(intent), encoding="utf-8")

    store = CodexSessionStore.from_root(
        tmp_path / "session",
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    store.create(
        "thread-1",
        stage="investigation_and_experiment",
        thread_policy_digest="b" * 64,
    )
    store.record_recovery_pending(
        stage="investigation_and_experiment",
        request_digest="a" * 64,
        thread_policy_digest="b" * 64,
        turn_id=None,
        reason="request_intent",
    )
    legacy_recovery = store.load_recovery()
    assert legacy_recovery is not None

    thread = _NoNewTurnHistoryThread(
        "thread-1",
        [{"id": "turn-1", "status": "failed"}],
    )
    monkeypatch.setattr(adapter, "Codex", _RecoveryCodex)
    monkeypatch.setattr(
        adapter,
        "_start_codex_thread",
        lambda **_kwargs: type(
            "Startup",
            (),
            {
                "thread": thread,
                "profile": "investigation",
                "thread_policy_digest": "b" * 64,
                "trace_values": (),
                "approval_mode": ApprovalMode.auto_review,
            },
        )(),
    )
    acknowledgements: list[tuple[str, str, str]] = []

    def acknowledge_terminal_turn(thread_id: str, turn_id: str, status: str) -> None:
        acknowledgements.append((thread_id, turn_id, status))
        store.record_recovery_pending(
            stage="investigation_and_experiment",
            request_digest="a" * 64,
            thread_policy_digest="b" * 64,
            turn_id=turn_id,
            reason="terminal_turn",
        )

    request = HarnessRequest(
        messages=({"role": "user", "content": "legacy request intent"},),
        model="gpt-5.6-sol",
        role="investigation",
        stage="investigation_and_experiment",
        workspace=receipt_root / "workspace",
        timeout_seconds=None,
    )

    with pytest.raises(adapter.CodexTerminalTurnError):
        adapter.run_codex_sdk_turn(
            config=CodexConfig(),
            request=request,
            thread_config={},
            session_id="thread-1",
            trace_path=tmp_path / "recovered-with-session.jsonl",
            completed_turn_receipt_root=receipt_root,
            request_digest="a" * 64,
            turn_count_before=0,
            completion_recovery=legacy_recovery,
            on_terminal_turn=acknowledge_terminal_turn,
        )

    assert thread.turn_calls == 0
    assert acknowledgements == [("thread-1", "turn-1", "failed")]
    migrated_recovery = store.load_recovery()
    assert migrated_recovery is not None
    assert migrated_recovery["reason"] == "terminal_turn"
    assert migrated_recovery["turn_id"] == "turn-1"
    assert not intent_path.exists()
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_v2_migration_journal_write_failure_keeps_legacy_intent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """journal 写失败时不能先把 v2 intent 替换成半成品 v3。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    before = intent_path.read_bytes()

    def fail_journal(*_args, **_kwargs):
        raise adapter.HarnessError("terminal journal write failed", kind="session_state")

    monkeypatch.setattr(adapter, "_write_terminal_turn_journal", fail_journal)
    with pytest.raises(adapter.HarnessError, match="journal write failed"):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=_failed_thread(),
            thread_id="thread-1",
            request_digest="a" * 64,
        )

    assert intent_path.read_bytes() == before
    assert json.loads(intent_path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert not (tmp_path / ".arche_codex_terminal_turn.json").exists()


def test_v2_migration_intent_replacement_failure_is_recoverable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """v3 intent 替换失败后保留 v2+匹配 journal，重试必须幂等收束。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    intent_path = tmp_path / ".arche_codex_request_intent.json"
    original_atomic_write = adapter._atomic_write_json_file

    def fail_intent_replacement(path, value, *, description, mode=0o600):
        if path.name == ".arche_codex_request_intent.json" and value["schema_version"] == 3:
            raise adapter.HarnessError("request intent replacement failed", kind="session_state")
        return original_atomic_write(path, value, description=description, mode=mode)

    monkeypatch.setattr(adapter, "_atomic_write_json_file", fail_intent_replacement)
    with pytest.raises(adapter.HarnessError, match="replacement failed"):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=_failed_thread(),
            thread_id="thread-1",
            request_digest="a" * 64,
        )

    assert json.loads(intent_path.read_text(encoding="utf-8"))["schema_version"] == 2
    journal_path = tmp_path / ".arche_codex_terminal_turn.json"
    assert json.loads(journal_path.read_text(encoding="utf-8"))["turn_id"] == "turn-1"

    monkeypatch.setattr(adapter, "_atomic_write_json_file", original_atomic_write)
    result = adapter.migrate_codex_v2_request_intent_to_terminal_turn(
        receipt_root,
        thread=_failed_thread(),
        thread_id="thread-1",
        request_digest="a" * 64,
    )
    assert result["turn_id"] == "turn-1"
    assert json.loads(intent_path.read_text(encoding="utf-8"))["schema_version"] == 3


def test_v2_migration_rejects_conflicting_existing_journal_without_writes(
    tmp_path: Path,
) -> None:
    """已有 journal 身份不匹配时不能覆盖或把旧 v2 猜成另一失败 turn。"""

    receipt_root = tmp_path / "receipts"
    _write_v2_intent(receipt_root)
    journal_path = tmp_path / ".arche_codex_terminal_turn.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "thread_id": "thread-1",
                "turn_id": "turn-other",
                "status": "failed",
                "request_digest": "a" * 64,
                "turn_count_before": 0,
                "thread_policy_digest": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    before_intent = (tmp_path / ".arche_codex_request_intent.json").read_bytes()
    before_journal = journal_path.read_bytes()

    with pytest.raises(adapter.HarnessError, match="conflicts"):
        adapter.migrate_codex_v2_request_intent_to_terminal_turn(
            receipt_root,
            thread=_failed_thread(),
            thread_id="thread-1",
            request_digest="a" * 64,
        )

    assert (tmp_path / ".arche_codex_request_intent.json").read_bytes() == before_intent
    assert journal_path.read_bytes() == before_journal
