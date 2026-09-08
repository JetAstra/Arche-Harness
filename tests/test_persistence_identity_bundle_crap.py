"""持久身份、Run Bundle 原子写入和 Investigation 索引的直接回归测试。"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from chemistry_multiagent.memory.contracts import ContractValidationError
from chemistry_multiagent.memory.investigation_index import (
    evaluate_combination_route_gate,
    read_investigation_index_snapshot,
)
from chemistry_multiagent.memory.sqlite_store import SQLiteMemoryStore
from chemistry_multiagent.utils import model_operation
from chemistry_multiagent.utils import run_bundle
from chemistry_multiagent.utils.run_bundle import RunBundle
from minichem_toolbox import process_identity


def test_atomic_bundle_write_uses_private_file_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    """原子 JSON 使用 0600 文件，并拒绝目标或父目录符号链接。"""

    bundle = RunBundle(tmp_path, "private-write")
    assert bundle.root.stat().st_mode & 0o777 == 0o700
    target = bundle.root / "internal" / "state" / "private.json"
    RunBundle._atomic_write_json(target, {"state": "ready"})
    assert target.stat().st_mode & 0o777 == 0o600
    assert json.loads(target.read_text(encoding="utf-8")) == {"state": "ready"}
    assert not list(target.parent.glob(".private.json.*.tmp"))

    target.unlink()
    target.symlink_to(bundle.run_path)
    with pytest.raises(ValueError, match="目标"):
        RunBundle._atomic_write_json(target, {"state": "tampered"})

    linked_parent = bundle.root / "linked-state"
    linked_parent.symlink_to(target.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="父"):
        RunBundle._atomic_write_json(linked_parent / "state.json", {"x": 1})


def test_atomic_bundle_write_exposes_parent_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目录项 fsync 失败时原子写入必须报告失败而非静默成功。"""

    bundle = RunBundle(tmp_path, "fsync-failure")
    target = bundle.root / "status-copy.json"
    real_open = run_bundle.os.open
    directory_flag = getattr(run_bundle.os, "O_DIRECTORY", 0)

    def fail_directory_open(path: object, flags: int, *args: object) -> int:
        if directory_flag and flags & directory_flag:
            raise OSError("directory fsync unavailable")
        return real_open(path, flags, *args)

    monkeypatch.setattr(run_bundle.os, "open", fail_directory_open)
    with pytest.raises(OSError, match="directory fsync"):
        RunBundle._atomic_write_json(target, {"state": "written"})
    assert target.is_file()


class _SnapshotConnection:
    """为快照测试提供只读空索引连接。"""

    def execute(self, _sql: str, _params: tuple[str, ...]) -> "_SnapshotRows":
        return _SnapshotRows()


class _SnapshotRows:
    """模拟 SQLite cursor 的空结果。"""

    def fetchall(self) -> list[object]:
        return []


def test_investigation_index_snapshot_reads_all_tables_without_rowid() -> None:
    """索引快照按固定表集合读取，空索引仍返回稳定 tuple。"""

    snapshot = read_investigation_index_snapshot(
        _SnapshotConnection(),
        "task-empty",
    )
    assert len(snapshot) == 14
    assert all(not rows for _table, rows in snapshot)


def test_investigation_index_snapshot_fails_closed_on_missing_table() -> None:
    """表结构损坏时快照直接暴露 SQLite 错误，不返回空结果。"""

    class BrokenConnection:
        def execute(self, _sql: str, _params: tuple[str, ...]) -> object:
            raise sqlite3.OperationalError("no such table")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        read_investigation_index_snapshot(BrokenConnection(), "task-broken")


def test_combination_route_gate_fails_closed_on_corrupt_review() -> None:
    """当前 result 存在但 Review 正文损坏时不能允许路由。"""

    class GateConnection:
        def execute(self, sql: str, _params: tuple[str, ...]) -> "_GateRows":
            if "investigation_sessions" in sql:
                return _GateRows(
                    {
                        "latest_result_event_id": "result-1",
                        "latest_review_event_id": "review-1",
                        "repair_count": 0,
                        "route_marker": None,
                        "pending_repair_event_id": None,
                    }
                )
            if "investigation_results" in sql:
                return _GateRows(
                    {
                        "result_version": "1",
                        "batch_id": "batch-1",
                        "combination_id": "comb-1",
                    }
                )
            return _GateRows(
                {
                    "event_type": "investigation_batch_reviewed",
                    "payload_json": "{broken",
                }
            )

    with pytest.raises(ContractValidationError, match="正文已损坏"):
        evaluate_combination_route_gate(
            GateConnection(),
            task_id="task-1",
            batch_id="batch-1",
            combination_id="comb-1",
        )


def test_combination_route_gate_covers_source_and_finding_rejections() -> None:
    """route gate 的来源、返修、版本和 finding 分支都保持拒绝优先。"""

    class Connection:
        def __init__(
            self,
            *,
            marker: str | None = None,
            pending: str | None = None,
            result: dict[str, object] | None = None,
            review_event: dict[str, object] | None = None,
        ) -> None:
            self.session = {
                "latest_result_event_id": "result-1",
                "latest_review_event_id": "review-1",
                "repair_count": 0,
                "route_marker": marker,
                "pending_repair_event_id": pending,
            }
            self.result = result
            self.review_event = review_event

        def execute(self, sql: str, params: tuple[str, ...]) -> "_GateRows":
            if "investigation_sessions" in sql:
                return _GateRows(self.session)
            if "investigation_results" in sql:
                return _GateRows(self.result)
            return _GateRows(self.review_event)

    review_payload = {
        "batch_id": "batch-1",
        "combination_ids": [f"comb-{index}" for index in range(1, 6)],
        "result_versions": [f"v{index}" for index in range(1, 6)],
        "findings": [
            {
                "combination_id": f"comb-{index}",
                "validity": "pass",
                "hypothesis_judgment": "refuted",
                "repair_advice": None,
            }
            for index in range(1, 6)
        ],
        "can_answer": False,
        "review_thread_id": "review-thread-1",
        "review_turn_id": "review-turn-1",
    }
    review_event = {
        "event_type": "investigation_batch_reviewed",
        "payload_json": json.dumps(review_payload),
    }
    valid_result = {
        "result_version": "v1",
        "batch_id": "batch-1",
        "combination_id": "comb-1",
    }
    assert evaluate_combination_route_gate(
        Connection(result=valid_result, review_event=review_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
    )["allowed"] is True
    assert evaluate_combination_route_gate(
        Connection(marker="revise_hypothesis", result=valid_result, review_event=review_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
        require_unmarked=True,
    )["allowed"] is False
    assert evaluate_combination_route_gate(
        Connection(result=valid_result, review_event=review_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
        result_event_ref="old-result",
    )["allowed"] is False
    assert evaluate_combination_route_gate(
        Connection(pending="repair-1", result=valid_result, review_event=review_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
    )["allowed"] is False
    assert evaluate_combination_route_gate(
        Connection(result=None, review_event=review_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
    )["allowed"] is False
    wrong_result = dict(valid_result, combination_id="comb-other")
    assert evaluate_combination_route_gate(
        Connection(result=wrong_result, review_event=review_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
    )["allowed"] is False
    assert evaluate_combination_route_gate(
        Connection(result=valid_result, review_event=None),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
    )["allowed"] is False
    wrong_batch_event = dict(review_event, payload_json=json.dumps({**review_payload, "batch_id": "batch-2"}))
    assert evaluate_combination_route_gate(
        Connection(result=valid_result, review_event=wrong_batch_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
    )["allowed"] is False
    wrong_version = dict(valid_result, result_version="v9")
    assert evaluate_combination_route_gate(
        Connection(result=wrong_version, review_event=review_event),
        task_id="task-1",
        batch_id="batch-1",
        combination_id="comb-1",
    )["allowed"] is False


class _GateRows:
    """模拟按一次 fetchone 返回一个 SQLite Row 的连接结果。"""

    def __init__(self, row: dict[str, object]) -> None:
        self.row = row

    def fetchone(self) -> dict[str, object]:
        return self.row


def test_sqlite_v10_initialization_is_idempotent(tmp_path: Path) -> None:
    """重复启动同一 Working Memory 不重复迁移，也不改变 schema 版本。"""

    database = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(str(database))
    store.initialize()
    store.initialize()
    assert store.fetchone("SELECT value FROM schema_meta WHERE key='schema_version'")[
        "value"
    ] == "10"
    columns = {
        row["name"]
        for row in store.fetchall("PRAGMA table_info(combinations)")
    }
    assert {
        "investigation_ready_state",
        "investigation_stage_state",
    } <= columns


def test_dynamic_finalized_manifest_locator_replays_artifacts(tmp_path: Path) -> None:
    """动态 operation 通过固定 staging 内容寻址，能复核并定位 finalized 清单。"""

    runner = model_operation.DynamicModelOperationRunner(
        task_root=tmp_path,
        operation_instance_id="dynamic-locator",
        operation="hypothesis_generation",
        base_input={"scientific_question": "验证 finalized locator。"},
        provider=lambda _request: {
            "id": "trace-locator",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "A", "reasoning_content": ""},
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    )
    stage = model_operation.ModelStage(
        stage="rank",
        stage_kind="model",
        validation_kind="text_enum_rank",
    )
    slot = runner.reserve_model_stages([stage])[0]
    assert runner.run_reserved_model_stage(slot, stage, {"winner": "A"}) == "A"
    finalized = runner.finalize_model_operation()
    located = model_operation.DynamicModelOperationRunner.locate_finalized_manifest_in_operation(
        task_root=tmp_path,
        operation="hypothesis_generation",
        operation_instance_id="dynamic-locator",
        expected_base_input={"scientific_question": "验证 finalized locator。"},
    )
    assert located is not None
    assert Path(located["manifest_path"]) == Path(finalized["manifest_path"])


def test_process_identity_boot_and_zombie_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动时钟、boot-id 和僵尸状态任一不匹配都不能返回 alive。"""

    monkeypatch.setattr(
        process_identity,
        "_read_proc_stat",
        lambda _pid: ("S", 456, 11, 22),
    )
    monkeypatch.setattr(process_identity, "_read_boot_id", lambda: "boot-new")
    identity = {"pid": 123, "start_time_ticks": 456, "boot_id": "boot-old"}
    assert process_identity.process_identity_activity(identity) == "stopped"

    monkeypatch.setattr(process_identity, "_read_boot_id", lambda: None)
    assert process_identity.process_identity_activity(identity) == "unknown"

    monkeypatch.setattr(process_identity, "_read_boot_id", lambda: "boot-old")
    monkeypatch.setattr(
        process_identity,
        "_read_proc_stat",
        lambda _pid: ("Z", 456, 11, 22),
    )
    assert process_identity.process_identity_activity(identity) == "stopped"
    assert process_identity.process_identity_activity(
        {"pid": 123, "start_ticks": 456, "boot_id": None}
    ) == "unknown"


def test_process_group_membership_parser_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """组身份输入、boot 不匹配和 proc 解析失败均保持 unknown。"""

    assert process_identity.inspect_process_group_membership(True, 1, "boot") == "unknown"
    assert process_identity.inspect_process_group_membership(1, 1, None) == "unknown"
    monkeypatch.setattr(process_identity, "_read_boot_id", lambda: "different")
    assert process_identity.inspect_process_group_membership(1, 1, "boot") == "unknown"
    monkeypatch.setattr(process_identity, "_read_boot_id", lambda: "boot")
    monkeypatch.setattr(
        process_identity,
        "_scan_process_group_membership",
        lambda _group, _session: "empty",
    )
    assert process_identity.inspect_process_group_membership(1, 1, "boot") == "empty"


def test_process_group_scan_handles_missing_malformed_and_matching_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扫描过程中进程退出可跳过，格式损坏保持 unknown，匹配成员返回 nonempty。"""

    class Entry:
        def __init__(self, name: str, value: str | BaseException) -> None:
            self.name = name
            self.value = value

        def __truediv__(self, _part: str) -> "Entry":
            return self

        def read_text(self, **_kwargs: object) -> str:
            if isinstance(self.value, BaseException):
                raise self.value
            return self.value

    class Proc:
        def __init__(self, entries: tuple[Entry, ...]) -> None:
            self.entries = entries

        def iterdir(self) -> tuple[Entry, ...]:
            return self.entries

    monkeypatch.setattr(process_identity, "_read_boot_id", lambda: "boot")
    entries = (
        Entry("not-pid", "ignored"),
        Entry("101", FileNotFoundError()),
        Entry("102", "malformed"),
    )
    monkeypatch.setattr(process_identity, "Path", lambda _value: Proc(entries))
    assert process_identity.inspect_process_group_membership(1, 2, "boot") == "unknown"

    entries = (
        Entry("101", "worker (name) S 0 1 2 3 4"),
    )
    monkeypatch.setattr(process_identity, "Path", lambda _value: Proc(entries))
    assert process_identity.inspect_process_group_membership(1, 2, "boot") == "nonempty"
