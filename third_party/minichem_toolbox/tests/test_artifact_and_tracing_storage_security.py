"""验证 ArtifactStore 与工具轨迹文件的安全落盘边界。"""

from __future__ import annotations

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest

from minichem_mcp_tools import tracing
from minichem_toolbox import _secure_storage
from minichem_toolbox.artifacts import ArtifactStore, resolve_workspace_path


def _put(store: ArtifactStore, value: int):
    """在隔离 ArtifactStore 中写入一个 JSON artifact。"""

    return store.put_json(
        {"value": value},
        semantic_type="SecurityProbe",
        producer_action="security_probe",
        producer_backend="test",
    )


def _mode(path: Path) -> int:
    """返回文件系统对象的权限位。"""

    return stat.S_IMODE(path.stat().st_mode)


def _process_put_artifacts(root: str, start: int, count: int) -> None:
    """子进程并发写入 ArtifactStore，异常通过非零退出码暴露。"""

    os.environ["MINICHEM_MCP_WORKSPACE"] = root
    store = ArtifactStore(root=Path(root))
    for value in range(start, start + count):
        _put(store, value)


def _process_write_traces(root: str, count: int) -> None:
    """子进程并发写入工具结果和轨迹。"""

    os.environ["MINICHEM_MCP_WORKSPACE"] = root
    for value in range(count):
        tracing.execute_traced(
            "process_probe",
            {"value": value},
            lambda value=value: {"value": value},
            capture_artifacts=False,
        )


def test_artifact_store_rejects_symlinked_root_and_objects_without_external_write(
    tmp_path: Path,
) -> None:
    """存储根或 objects 目录是符号链接时必须在任何写入前失败。"""

    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "workspace"
    root.mkdir()

    root_link = tmp_path / "workspace-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises((OSError, ValueError, FileNotFoundError)) as root_error:
        ArtifactStore(root=root_link)
    assert str(external) not in str(root_error.value)
    assert not list(external.iterdir())

    (root / "_tool_artifacts").mkdir()
    (root / "_tool_artifacts" / "objects").symlink_to(
        external, target_is_directory=True
    )
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        ArtifactStore(root=root)
    assert not list(external.iterdir())


def test_artifact_store_rejects_symlinked_index_before_put_json_side_effect(
    tmp_path: Path,
) -> None:
    """index.jsonl 是符号链接时 put_json 不得先向对象或外部索引写入。"""

    root = tmp_path / "workspace"
    root.mkdir()
    external_index = tmp_path / "external-index.jsonl"
    external_index.write_text("sentinel\n", encoding="utf-8")
    (root / "_tool_artifacts").mkdir()
    (root / "_tool_artifacts" / "objects").mkdir()
    (root / "_tool_artifacts" / "index.jsonl").symlink_to(external_index)

    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        ArtifactStore(root=root).put_json(
            {"secret": "must-not-escape"},
            semantic_type="SecurityProbe",
            producer_action="security_probe",
            producer_backend="test",
        )
    assert external_index.read_text(encoding="utf-8") == "sentinel\n"
    assert not list((root / "_tool_artifacts" / "objects").iterdir())


def test_artifact_store_storage_permissions_and_compatible_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """artifact、目录和 JSONL 索引使用私有权限，同时保持既有引用协议。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path / "workspace"))
    root = tmp_path / "workspace"
    root.mkdir()
    storage_root = root / "_tool_artifacts"
    objects = storage_root / "objects"
    index_path = storage_root / "index.jsonl"
    storage_root.mkdir(mode=0o755)
    objects.mkdir(mode=0o755)
    index_path.touch(mode=0o400)
    storage_root.chmod(0o755)
    objects.chmod(0o755)
    index_path.chmod(0o400)
    store = ArtifactStore(root=root)
    reference = _put(store, 7)

    object_path = root / reference.path
    assert _mode(storage_root) == 0o700
    assert _mode(objects) == 0o700
    assert _mode(object_path) == 0o600
    assert _mode(index_path) == 0o600
    assert store.find(reference.artifact_id) == reference
    assert store.load(reference) == {"value": 7}
    assert json.loads(index_path.read_text(encoding="utf-8"))["artifact_id"] == (
        reference.artifact_id
    )


def test_resolve_workspace_path_must_exist_false_accepts_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """词法路径检查在非必存模式下保留已有目录的旧协议。"""

    root = tmp_path / "workspace"
    root.mkdir()
    directory = root / "inputs"
    directory.mkdir()
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(root))

    assert resolve_workspace_path("inputs", must_exist=False) == directory


def test_resolve_workspace_path_preserves_missing_and_target_type_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """路径解析区分缺失、目录、符号链接和不支持的文件类型。"""

    root = tmp_path / "workspace"
    root.mkdir()
    directory = root / "inputs"
    directory.mkdir()
    regular = root / "input.txt"
    regular.write_text("input", encoding="utf-8")
    linked_target = root / "linked-target.txt"
    linked_target.write_text("target", encoding="utf-8")
    (root / "linked.txt").symlink_to(linked_target)
    (root / "linked-directory").symlink_to(directory, target_is_directory=True)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(root))

    assert resolve_workspace_path(".", must_exist=False) == root
    assert resolve_workspace_path("missing.txt", must_exist=False) == root / "missing.txt"
    assert resolve_workspace_path("inputs/missing.txt", must_exist=False) == (
        root / "inputs/missing.txt"
    )
    with pytest.raises(FileNotFoundError):
        resolve_workspace_path("missing.txt", must_exist=True)
    with pytest.raises(FileNotFoundError):
        resolve_workspace_path("inputs/missing.txt", must_exist=True)
    with pytest.raises(ValueError):
        resolve_workspace_path("inputs", must_exist=True)
    with pytest.raises(ValueError):
        resolve_workspace_path("linked.txt", must_exist=False)
    with pytest.raises(ValueError):
        resolve_workspace_path("linked-directory/missing.txt", must_exist=False)

    if hasattr(os, "mkfifo"):
        fifo = root / "input.pipe"
        os.mkfifo(fifo)
        with pytest.raises(ValueError):
            resolve_workspace_path("input.pipe", must_exist=False)


def test_artifact_store_load_missing_object_preserves_file_not_found_semantics(
    tmp_path: Path,
) -> None:
    """索引中的对象被删除后，load 仍报告缺失而不是泛化为路径错误。"""

    store = ArtifactStore(root=tmp_path)
    reference = _put(store, 11)
    object_path = tmp_path / reference.path
    object_path.unlink()

    with pytest.raises(FileNotFoundError) as error:
        store.load(reference)
    assert str(tmp_path) not in str(error.value)


def test_tracing_rejects_symlinked_result_directory_before_running_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """结果目录是符号链接时 execute_traced 必须先失败且不调用工具。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "_tool_results").symlink_to(external, target_is_directory=True)
    called: list[bool] = []

    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        tracing.execute_traced("security_probe", {}, lambda: called.append(True))
    assert not called
    assert not list(external.iterdir())


def test_tracing_rejects_symlinked_workspace_root_before_running_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """工作区环境变量指向符号链接时也必须拒绝，而不能解析到外部目录。"""

    real_root = tmp_path / "real-workspace"
    real_root.mkdir()
    root_link = tmp_path / "workspace-link"
    root_link.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(root_link))
    called: list[bool] = []

    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        tracing.execute_traced("security_probe", {}, lambda: called.append(True))
    assert not called
    assert not list(real_root.iterdir())


def test_tracing_rejects_symlinked_trace_and_result_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """轨迹或结果目标是符号链接时不得覆盖目标或运行工具。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    external_trace = tmp_path / "external-trace.jsonl"
    external_trace.write_text("trace-sentinel\n", encoding="utf-8")
    (tmp_path / "_tool_results").mkdir()
    (tmp_path / "_tool_trace.jsonl").symlink_to(external_trace)

    called: list[bool] = []
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        tracing.execute_traced("security_probe", {}, lambda: called.append(True))
    assert not called
    assert external_trace.read_text(encoding="utf-8") == "trace-sentinel\n"

    (tmp_path / "_tool_trace.jsonl").unlink()
    result_path = tmp_path / "_tool_results" / "0001_security_probe.json"
    external_result = tmp_path / "external-result.json"
    external_result.write_text("result-sentinel\n", encoding="utf-8")
    result_path.symlink_to(external_result)
    monkeypatch.setattr(tracing, "_next_sequence", lambda _root: 1)
    called.clear()
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        tracing.execute_traced("security_probe", {}, lambda: called.append(True))
    assert not called
    assert external_result.read_text(encoding="utf-8") == "result-sentinel\n"


def test_tracing_permissions_and_stale_tmp_are_private_and_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """结果、轨迹和目录权限固定，崩溃遗留临时文件不会成为写入目标。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    result_dir = tmp_path / "_tool_results"
    result_dir.mkdir()
    stale = result_dir / ".tmp-crashed-result"
    stale.write_text("stale", encoding="utf-8")
    external = tmp_path / "external-stale"
    external.write_text("outside", encoding="utf-8")
    (result_dir / ".tmp-crashed-link").symlink_to(external)
    artifact_root = tmp_path / "_tool_artifacts"
    artifact_root.mkdir()
    (artifact_root / ".tmp-crashed-link").symlink_to(external)

    returned = tracing.execute_traced(
        "security_probe", {"arg": 1}, lambda: {"status": "success"}, capture_artifacts=False
    )
    assert returned == {"status": "success"}
    assert _mode(result_dir) == 0o700
    result_files = list(result_dir.glob("[0-9][0-9][0-9][0-9]_security_probe.json"))
    assert len(result_files) == 1
    assert _mode(result_files[0]) == 0o600
    trace_path = tmp_path / "_tool_trace.jsonl"
    assert _mode(trace_path) == 0o600
    assert _mode(artifact_root) == 0o700
    assert not (result_dir / ".tmp-crashed-link").exists()
    assert not (artifact_root / ".tmp-crashed-link").exists()
    assert external.read_text(encoding="utf-8") == "outside"


def test_tracing_cleans_nested_sequence_tmp_without_following_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重启时递归清理序列目录临时文件，但不进入外部符号链接目录。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    artifact_root = tmp_path / "_tool_artifacts"
    nested = artifact_root / "0007" / "changed" / "deep"
    nested.mkdir(parents=True)
    stale = nested / ".tmp-crashed-artifact"
    stale.write_text("stale", encoding="utf-8")
    external = tmp_path / "external-nested"
    external.mkdir()
    external_stale = external / ".tmp-must-survive"
    external_stale.write_text("outside", encoding="utf-8")
    (nested / ".tmp-crashed-link").symlink_to(external_stale)
    linked_directory = artifact_root / "0008" / "linked"
    linked_directory.parent.mkdir(parents=True)
    linked_directory.symlink_to(external, target_is_directory=True)

    returned = tracing.execute_traced(
        "nested_cleanup_probe",
        {},
        lambda: {"status": "success"},
        capture_artifacts=False,
    )

    assert returned == {"status": "success"}
    assert not stale.exists()
    assert not (nested / ".tmp-crashed-link").exists()
    assert external_stale.read_text(encoding="utf-8") == "outside"


def test_artifact_and_trace_jsonl_writes_remain_parseable_under_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """并发调用不能产生互相覆盖或半行 JSON。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    store = ArtifactStore(root=tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        references = list(executor.map(lambda value: _put(store, value), range(32)))
    index_lines = (tmp_path / "_tool_artifacts" / "index.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(index_lines) == 32
    assert {json.loads(line)["artifact_id"] for line in index_lines} == {
        reference.artifact_id for reference in references
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda value: tracing.execute_traced(
                    "concurrent_probe", {"value": value}, lambda value=value: {"value": value}, capture_artifacts=False
                ),
                range(32),
            )
        )
    trace_lines = (tmp_path / "_tool_trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 32
    assert all(json.loads(line)["tool"] == "concurrent_probe" for line in trace_lines)


def test_artifact_and_trace_jsonl_writes_remain_parseable_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """跨进程锁必须保证索引与轨迹仍由完整 JSON 行组成。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    ArtifactStore(root=tmp_path)
    context = get_context("spawn")
    artifact_processes = [
        context.Process(target=_process_put_artifacts, args=(str(tmp_path), start, 8))
        for start in range(0, 32, 8)
    ]
    for process in artifact_processes:
        process.start()
    for process in artifact_processes:
        process.join(30)
        assert process.exitcode == 0

    index_lines = (tmp_path / "_tool_artifacts" / "index.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(index_lines) == 32
    assert all(json.loads(line)["artifact_id"].startswith("art_") for line in index_lines)

    trace_processes = [
        context.Process(target=_process_write_traces, args=(str(tmp_path), 8))
        for _ in range(4)
    ]
    for process in trace_processes:
        process.start()
    for process in trace_processes:
        process.join(30)
        assert process.exitcode == 0
    trace_lines = (tmp_path / "_tool_trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 32
    assert all(json.loads(line)["tool"] == "process_probe" for line in trace_lines)


def test_atomic_replace_postcheck_rejects_controlled_target_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """可控 TOCTOU 注入把替换目标换成 symlink 时，替换后复核必须失败。"""

    directory = tmp_path / "private"
    directory.mkdir()
    external = tmp_path / "external-target"
    external.write_text("sentinel", encoding="utf-8")
    original_replace = _secure_storage.os.replace

    def replace_then_inject(source, destination, **kwargs):
        """先完成原子替换，再模拟攻击者把目标换成外部符号链接。"""

        original_replace(source, destination, **kwargs)
        directory_fd = kwargs["dst_dir_fd"]
        _secure_storage.os.unlink(destination, dir_fd=directory_fd)
        _secure_storage.os.symlink(external, destination, dir_fd=directory_fd)

    monkeypatch.setattr(_secure_storage.os, "replace", replace_then_inject)
    with pytest.raises(ValueError, match="internal storage path is invalid"):
        _secure_storage.atomic_write_bytes(directory, "trace.jsonl", b"secret")
    assert external.read_text(encoding="utf-8") == "sentinel"
