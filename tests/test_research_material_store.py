"""通用研究材料存储的公开接口合同测试。"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat

import pytest

from chemistry_multiagent.investigation.research_material_store import (
    ResearchMaterialCountLimitExceeded,
    ResearchMaterialDescriptorError,
    ResearchMaterialIntegrityConflict,
    ResearchMaterialMissing,
    ResearchMaterialPersistenceError,
    ResearchMaterialSizeLimitExceeded,
    ResearchMaterialStore,
    ResearchMaterialTaskSizeLimitExceeded,
    ResearchMaterialTargetConflict,
    ResearchMaterialUnsafePath,
)


def _make_store(tmp_path: Path, *, max_file_size_bytes: int = 1024) -> ResearchMaterialStore:
    """在测试目录创建使用显式单文件上限的材料存储。"""

    return ResearchMaterialStore(
        tmp_path / "task",
        max_file_size_bytes=max_file_size_bytes,
    )


def _find_persisted_content_file(task_root: Path, content: bytes) -> Path:
    """从任务目录找到与公开摄取内容相同的 CAS 文件。"""

    matches = [
        path
        for path in task_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.read_bytes() == content
    ]
    assert len(matches) == 1
    return matches[0]


def test_ingest_material_bytes_returns_json_descriptor_without_source_path(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    source_path = tmp_path / "caller" / "notes.txt"
    source_path.parent.mkdir()

    descriptor = store.ingest_material_bytes(
        filename="notes.txt",
        media_type="text/plain",
        content=b"hello material",
        description="研究员提供的说明",
    )

    assert json.loads(json.dumps(descriptor, ensure_ascii=False)) == descriptor
    assert descriptor == {
        "filename": "notes.txt",
        "media_type": "text/plain",
        "size_bytes": 14,
        "sha256": hashlib.sha256(b"hello material").hexdigest(),
        "content_ref": f"sha256:{hashlib.sha256(b'hello material').hexdigest()}",
        "description": "研究员提供的说明",
    }
    assert str(source_path) not in json.dumps(descriptor, ensure_ascii=False)
    assert all(not Path(value).is_absolute() for value in descriptor.values() if isinstance(value, str))


def test_ingest_material_stream_uses_same_descriptor_contract(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    descriptor = store.ingest_material_stream(
        filename="coords.xyz",
        media_type="chemical/x-xyz",
        stream=io.BytesIO(b"3\nwater\nO 0 0 0\n"),
    )

    assert descriptor["filename"] == "coords.xyz"
    assert descriptor["size_bytes"] == len(b"3\nwater\nO 0 0 0\n")
    assert descriptor["sha256"] == hashlib.sha256(b"3\nwater\nO 0 0 0\n").hexdigest()


def test_same_content_reuses_content_addressed_bytes(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    first = store.ingest_material_bytes("first.txt", "text/plain", b"same")
    second = store.ingest_material_bytes("second.txt", "text/plain", b"same")

    assert first["content_ref"] == second["content_ref"]
    assert first["sha256"] == second["sha256"]
    assert len(
        [
            path
            for path in (tmp_path / "task").rglob("*")
            if path.is_file() and not path.is_symlink() and path.read_bytes() == b"same"
        ]
    ) == 1


def test_exact_descriptor_replay_does_not_consume_material_count(tmp_path: Path) -> None:
    store = ResearchMaterialStore(
        tmp_path / "task",
        max_file_size_bytes=16,
        max_material_count=1,
        max_task_size_bytes=16,
    )

    first = store.ingest_material_bytes("notes.txt", "text/plain", b"same")
    replay = store.ingest_material_bytes("notes.txt", "text/plain", b"same")

    assert replay == first
    with pytest.raises(ResearchMaterialCountLimitExceeded):
        store.ingest_material_bytes("renamed.txt", "text/plain", b"same")


def test_exact_descriptor_replay_survives_store_reopen_without_recount(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "task"
    ResearchMaterialStore(
        task_root,
        max_file_size_bytes=16,
        max_material_count=1,
        max_task_size_bytes=4,
    ).ingest_material_bytes("notes.txt", "text/plain", b"same")
    reopened = ResearchMaterialStore(
        task_root,
        max_file_size_bytes=16,
        max_material_count=1,
        max_task_size_bytes=4,
    )

    assert reopened.ingest_material_bytes("notes.txt", "text/plain", b"same")["size_bytes"] == 4
    with pytest.raises(ResearchMaterialCountLimitExceeded):
        reopened.ingest_material_bytes("other.txt", "text/plain", b"x")


def test_distinct_descriptor_reuses_cas_but_consumes_count_and_logical_size(
    tmp_path: Path,
) -> None:
    store = ResearchMaterialStore(
        tmp_path / "task",
        max_file_size_bytes=16,
        max_material_count=3,
        max_task_size_bytes=8,
    )

    first = store.ingest_material_bytes("first.txt", "text/plain", b"same")
    second = store.ingest_material_bytes("second.txt", "text/plain", b"same")

    assert second["content_ref"] == first["content_ref"]
    with pytest.raises(ResearchMaterialTaskSizeLimitExceeded):
        store.ingest_material_bytes("third.txt", "text/plain", b"xx")


@pytest.mark.parametrize("size", [0, 4])
def test_size_policy_accepts_empty_and_exact_boundary(tmp_path: Path, size: int) -> None:
    store = _make_store(tmp_path, max_file_size_bytes=4)

    descriptor = store.ingest_material_bytes("boundary.bin", "application/octet-stream", b"x" * size)

    assert descriptor["size_bytes"] == size


def test_size_policy_rejects_content_over_explicit_limit(tmp_path: Path) -> None:
    store = _make_store(tmp_path, max_file_size_bytes=4)

    with pytest.raises(ResearchMaterialSizeLimitExceeded):
        store.ingest_material_bytes("too-large.bin", "application/octet-stream", b"12345")


def test_safe_basename_policy_enforces_utf8_byte_boundary(tmp_path: Path) -> None:
    store = ResearchMaterialStore(
        tmp_path / "task",
        max_file_size_bytes=16,
        max_filename_bytes=8,
    )

    assert (
        store.ingest_material_bytes("éé.txt", "text/plain", b"ok")["filename"]
        == "éé.txt"
    )
    with pytest.raises(ResearchMaterialDescriptorError):
        store.ingest_material_bytes("ééé.txt", "text/plain", b"ok")


@pytest.mark.parametrize(
    "filename",
    ["", ".", "..", "../escape.txt", "nested/file.txt", r"nested\file.txt", "/tmp/file.txt"],
)
def test_ingest_rejects_non_basename_filename(tmp_path: Path, filename: str) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(ResearchMaterialDescriptorError):
        store.ingest_material_bytes(filename, "text/plain", b"content")


def test_ingest_rejects_empty_media_type(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(ResearchMaterialDescriptorError):
        store.ingest_material_bytes("notes.txt", "  ", b"content")


@pytest.mark.parametrize("description", ["source=/tmp/caller.txt", r"source=C:\\caller.txt"])
def test_ingest_rejects_absolute_path_in_description(
    tmp_path: Path,
    description: str,
) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(ResearchMaterialDescriptorError):
        store.ingest_material_bytes(
            "notes.txt",
            "text/plain",
            b"content",
            description=description,
        )


def test_ingest_rejects_non_bytes_content(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(ResearchMaterialDescriptorError):
        store.ingest_material_bytes("notes.txt", "text/plain", 3)  # type: ignore[arg-type]


def test_store_rejects_symlinked_task_root(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual-task"
    actual_root.mkdir()
    symlink_root = tmp_path / "task-link"
    symlink_root.symlink_to(actual_root, target_is_directory=True)

    with pytest.raises(ResearchMaterialUnsafePath):
        ResearchMaterialStore(symlink_root, max_file_size_bytes=4)


def test_store_and_materialize_reject_filesystem_root(tmp_path: Path) -> None:
    with pytest.raises(ResearchMaterialUnsafePath):
        ResearchMaterialStore(Path("/"), max_file_size_bytes=4)

    store = _make_store(tmp_path)
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    with pytest.raises(ResearchMaterialUnsafePath):
        store.materialize_material(descriptor, Path("/"))


def test_materialize_generates_deterministic_relative_path_and_strict_mode(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "session-workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")

    first_path = store.materialize_material(descriptor, workspace)
    second_path = store.materialize_material(descriptor, workspace)

    assert first_path == second_path
    relative = Path(first_path)
    assert not relative.is_absolute()
    assert ".." not in relative.parts
    assert relative.parts[:2] == ("inputs", "materials")
    target = workspace / relative
    assert target.read_bytes() == b"material"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.stat().st_mode & 0o111 == 0


def test_materialize_rejects_symlinked_workspace(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    actual_workspace = tmp_path / "actual-workspace"
    actual_workspace.mkdir()
    symlink_workspace = tmp_path / "workspace-link"
    symlink_workspace.symlink_to(actual_workspace, target_is_directory=True)
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")

    with pytest.raises(ResearchMaterialUnsafePath):
        store.materialize_material(descriptor, symlink_workspace)


def test_materialize_rejects_descriptor_path_traversal_without_writing(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    descriptor["filename"] = "../escape.txt"

    with pytest.raises(ResearchMaterialDescriptorError):
        store.materialize_material(descriptor, workspace)
    assert not (tmp_path / "escape.txt").exists()


def test_materialize_reports_missing_cas_bytes_separately(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    cas_file = _find_persisted_content_file(tmp_path / "task", b"material")
    cas_file.unlink()

    with pytest.raises(ResearchMaterialMissing):
        store.materialize_material(descriptor, workspace)
    assert not (workspace / "inputs").exists()


def test_materialize_reports_cas_digest_drift_separately(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    cas_file = _find_persisted_content_file(tmp_path / "task", b"material")
    cas_file.write_bytes(b"tampered")

    with pytest.raises(ResearchMaterialIntegrityConflict):
        store.materialize_material(descriptor, workspace)
    assert not (workspace / "inputs").exists()


def test_materialize_rejects_executable_cas_bytes(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    cas_file = _find_persisted_content_file(tmp_path / "task", b"material")
    os.chmod(cas_file, 0o700)

    with pytest.raises(ResearchMaterialIntegrityConflict):
        store.materialize_material(descriptor, workspace)


def test_materialize_rejects_cas_symlink_and_fifo(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    cas_file = _find_persisted_content_file(tmp_path / "task", b"material")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"material")

    cas_file.unlink()
    cas_file.symlink_to(outside)
    with pytest.raises(ResearchMaterialUnsafePath):
        store.materialize_material(descriptor, workspace)

    cas_file.unlink()
    os.mkfifo(cas_file)
    with pytest.raises(ResearchMaterialUnsafePath):
        store.materialize_material(descriptor, workspace)


def test_materialize_rejects_cas_permission_drift(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    cas_file = _find_persisted_content_file(tmp_path / "task", b"material")
    os.chmod(cas_file, 0o640)

    with pytest.raises(ResearchMaterialIntegrityConflict):
        store.materialize_material(descriptor, workspace)


def test_materialize_does_not_overwrite_target_content_drift(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    relative_path = store.materialize_material(descriptor, workspace)
    target = workspace / relative_path
    target.write_bytes(b"different")

    with pytest.raises(ResearchMaterialTargetConflict):
        store.materialize_material(descriptor, workspace)
    assert target.read_bytes() == b"different"


def test_materialize_rejects_target_symlink_and_special_directory(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = store.ingest_material_bytes("notes.txt", "text/plain", b"material")
    relative_path = store.materialize_material(descriptor, workspace)
    target = workspace / relative_path
    target.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    target.symlink_to(outside)

    with pytest.raises(ResearchMaterialUnsafePath):
        store.materialize_material(descriptor, workspace)
    assert outside.read_bytes() == b"outside"

    target.unlink()
    target.mkdir()
    with pytest.raises(ResearchMaterialUnsafePath):
        store.materialize_material(descriptor, workspace)

    target.rmdir()
    os.mkfifo(target)
    with pytest.raises(ResearchMaterialUnsafePath):
        store.materialize_material(descriptor, workspace)


class _FailingStream:
    """写入一段内容后模拟来源流读取失败。"""

    def __init__(self) -> None:
        self._read = False

    def read(self, _size: int) -> bytes:
        if not self._read:
            self._read = True
            return b"partial"
        raise OSError("source stream failed")


def test_stream_failure_leaves_no_partial_cas_file(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(OSError):
        store.ingest_material_stream("broken.bin", "application/octet-stream", _FailingStream())

    assert not [
        path
        for path in (tmp_path / "task").rglob("*")
        if path.is_file() and path.read_bytes() == b"partial"
    ]
    assert not [path for path in (tmp_path / "task").rglob("*") if path.name.endswith(".tmp")]


def test_cas_publish_failure_leaves_no_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path)

    def fail_link(_source: str | os.PathLike[str], _target: str | os.PathLike[str]) -> None:
        raise OSError("atomic link failed")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(ResearchMaterialPersistenceError):
        store.ingest_material_bytes("broken.bin", "application/octet-stream", b"partial")

    assert not [
        path
        for path in (tmp_path / "task").rglob("*")
        if path.is_file() and not path.is_symlink() and path.read_bytes() == b"partial"
    ]
    assert not [path for path in (tmp_path / "task").rglob("*") if path.name.endswith(".tmp")]
