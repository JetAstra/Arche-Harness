from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from chemistry_multiagent.controllers.task_lock import TaskRunnerBusy, TaskRunnerFileLock
from chemistry_multiagent.utils.task_artifacts import (
    TaskArtifactError,
    write_content_addressed_json,
    write_named_json,
)


def test_task_lock_serializes_two_live_runners(tmp_path: Path) -> None:
    """同一任务同时只有一个 runner 可以持有内核文件锁。"""

    lock_path = tmp_path / "task" / ".locks" / "task_runner.lock"
    first = TaskRunnerFileLock(lock_path)
    second = TaskRunnerFileLock(lock_path)

    first.acquire()
    try:
        with pytest.raises(TaskRunnerBusy, match="存活 runner"):
            second.acquire()
    finally:
        first.release()

    # 锁文件可以保留；释放内核锁后后继 runner 仍能接管同一路径。
    assert lock_path.is_file()
    second.acquire()
    second.release()


def test_task_lock_rejects_reentrant_acquire_and_is_idempotent_on_release(
    tmp_path: Path,
) -> None:
    """当前对象重复 acquire 失败，未持锁 release 不产生副作用。"""

    lock = TaskRunnerFileLock(tmp_path / "runner.lock")
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="已经持有锁"):
            lock.acquire()
    finally:
        lock.release()
    lock.release()


def test_content_addressed_json_replay_reuses_the_same_file(tmp_path: Path) -> None:
    """相同 checkpoint 内容重放时复用同一摘要文件，不产生第二份事实。"""

    first = write_content_addressed_json(
        tmp_path,
        PurePosixPath("artifacts/checkpoints"),
        "checkpoint",
        {"operation_instance_id": "operation-1", "checkpoint_index": 1},
    )
    second = write_content_addressed_json(
        tmp_path,
        PurePosixPath("artifacts/checkpoints"),
        "checkpoint",
        {"checkpoint_index": 1, "operation_instance_id": "operation-1"},
    )

    assert first == second
    assert list((tmp_path / "artifacts/checkpoints").glob("*.json")) == [
        tmp_path / first
    ]
    assert json.loads((tmp_path / first).read_text(encoding="utf-8")) == {
        "checkpoint_index": 1,
        "operation_instance_id": "operation-1",
    }


def test_content_addressed_json_rejects_unsafe_path_and_non_finite_payload(
    tmp_path: Path,
) -> None:
    """产物不得越过任务根，也不得把不可复验的 NaN 写入清单。"""

    with pytest.raises(TaskArtifactError, match="安全相对路径"):
        write_content_addressed_json(
            tmp_path,
            PurePosixPath("../outside"),
            "checkpoint",
            {"ok": True},
        )
    with pytest.raises(TaskArtifactError, match="规范编码"):
        write_content_addressed_json(
            tmp_path,
            PurePosixPath("artifacts"),
            "checkpoint",
            {"value": float("nan")},
        )


def test_named_json_is_atomic_and_replayable(tmp_path: Path) -> None:
    """固定名称的 Retrieval 清单重放后仍是一个完整 JSON 文件。"""

    reference = write_named_json(tmp_path, "retrieval_context.json", {"query": "Q"})
    replay = write_named_json(tmp_path, "retrieval_context.json", {"query": "Q"})

    assert reference == replay
    assert json.loads((tmp_path / "retrieval_context.json").read_text(encoding="utf-8")) == {
        "query": "Q"
    }

    with pytest.raises(TaskArtifactError, match="直接位于 task_root"):
        write_named_json(tmp_path, "nested/retrieval.json", {"query": "Q"})


def test_artifact_writer_rejects_symlink_artifact_directory(tmp_path: Path) -> None:
    """任务产物目录经过符号链接时拒绝写入，避免越过任务根。"""

    task_root = tmp_path / "task"
    task_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (task_root / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(TaskArtifactError, match="符号链接"):
        write_content_addressed_json(
            task_root,
            PurePosixPath("artifacts/checkpoints"),
            "checkpoint",
            {"status": "ok"},
        )


def test_artifact_writer_rejects_symlink_task_root(tmp_path: Path) -> None:
    """任务根是符号链接时也必须拒绝写入，不能把路径解析当作授权。"""

    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(TaskArtifactError, match="普通目录"):
        write_named_json(linked_root, "result.json", {"status": "ok"})
