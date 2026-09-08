"""在任务目录内原子保存可复核 JSON 文件。"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class TaskArtifactError(ValueError):
    """任务产物路径越界、内容冲突或 JSON 无法规范编码。"""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """把 JSON object 编码为键排序且不含 NaN 的稳定字节。"""

    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TaskArtifactError("任务产物不是可规范编码的 JSON object") from exc


def _safe_task_directory(task_root: Path, relative_directory: PurePosixPath) -> Path:
    """逐层建立任务内目录，并拒绝绝对路径、父目录和符号链接。"""

    if relative_directory.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_directory.parts
    ):
        raise TaskArtifactError("任务产物目录必须是安全相对路径")
    input_root = Path(task_root)
    if input_root.is_symlink():
        raise TaskArtifactError("task_root 必须是已存在的普通目录")
    root = input_root.resolve()
    if not root.is_dir():
        raise TaskArtifactError("task_root 必须是已存在的普通目录")
    directory = root
    for part in relative_directory.parts:
        directory = directory / part
        if directory.exists() and directory.is_symlink():
            raise TaskArtifactError("任务产物目录不得经过符号链接")
        directory.mkdir(exist_ok=True)
    return directory


def _write_json_bytes(target: Path, payload: bytes) -> None:
    """用同目录临时文件和 ``os.replace`` 原子写入稳定字节。"""

    if target.is_symlink():
        raise TaskArtifactError("任务产物文件不得是符号链接")
    if target.exists() and target.read_bytes() == payload:
        return
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_content_addressed_json(
    task_root: str | os.PathLike[str],
    relative_directory: PurePosixPath,
    stem: str,
    value: Mapping[str, Any],
) -> str:
    """按内容摘要命名 JSON，并返回任务根下的相对 POSIX 路径。

    输入是任务根、相对目录、文件名前缀和 JSON object。相同内容重复写入会复用同一
    文件；同名摘要文件内容不同、路径越界或编码失败时抛出 ``TaskArtifactError``。
    """

    if not isinstance(stem, str) or not stem or Path(stem).name != stem:
        raise TaskArtifactError("任务产物文件前缀非法")
    input_root = Path(task_root)
    if input_root.is_symlink():
        raise TaskArtifactError("task_root 必须是已存在的普通目录")
    root = input_root.resolve()
    directory = _safe_task_directory(root, relative_directory)
    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    target = directory / f"{stem}_{digest}.json"
    if target.exists() and target.read_bytes() != payload:
        raise TaskArtifactError("内容摘要文件与实际内容冲突")
    _write_json_bytes(target, payload)
    try:
        return target.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise TaskArtifactError("任务产物越过 task_root") from exc


def write_named_json(
    task_root: str | os.PathLike[str],
    relative_path: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """原子写入任务根目录下的固定文件名并返回摘要、类型和大小。"""

    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise TaskArtifactError("固定任务产物必须使用相对 POSIX 文件名")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise TaskArtifactError("固定任务产物必须直接位于 task_root")
    input_root = Path(task_root)
    if input_root.is_symlink():
        raise TaskArtifactError("task_root 必须是已存在的普通目录")
    root = input_root.resolve()
    if not root.is_dir():
        raise TaskArtifactError("task_root 必须是已存在的普通目录")
    payload = _canonical_json_bytes(value)
    target = root / relative.name
    _write_json_bytes(target, payload)
    return {
        "relative_path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "media_type": "application/json",
        "size": len(payload),
    }


def write_task_relative_json(
    task_root: str | os.PathLike[str],
    relative_path: str,
    value: Mapping[str, Any],
) -> str:
    """在任务目录内按固定相对路径原子写入 JSON。

    输入是任务根目录、安全 POSIX 相对路径和 JSON 映射；动作校验路径，创建临时文件
    并原子替换目标，不覆盖不一致的既有收据。返回值是写入文件的相对路径；根目录、
    路径、JSON 编码或原子写入失败时抛出 ``TaskArtifactError``。
    """

    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise TaskArtifactError("任务产物必须使用非空 POSIX 相对路径")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise TaskArtifactError("任务产物路径必须是安全相对路径")
    input_root = Path(task_root)
    if input_root.is_symlink():
        raise TaskArtifactError("task_root 必须是已存在的普通目录")
    root = input_root.resolve()
    if not root.is_dir():
        raise TaskArtifactError("task_root 必须是已存在的普通目录")
    directory = _safe_task_directory(root, PurePosixPath(*relative.parts[:-1]))
    payload = _canonical_json_bytes(value)
    target = directory / relative.name
    # started 请求收据按顺序编号；同一路径一旦落盘，后续调用只能复用完全相同
    # 的内容，不能用原子替换把较新的快照覆盖掉旧事实。
    if target.is_symlink():
        raise TaskArtifactError("任务产物文件不得是符号链接")
    if target.exists():
        try:
            existing = target.read_bytes()
        except OSError as exc:
            raise TaskArtifactError("任务产物文件无法读取") from exc
        if existing != payload:
            raise TaskArtifactError("固定任务产物路径已有不同内容")
        return relative.as_posix()

    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # link 创建是不可覆盖的；它既保留临时文件写完后的原子性，也能在
            # 并发调用使用同一编号时拒绝覆盖先到达的收据。
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink():
                raise TaskArtifactError("任务产物文件不得是符号链接")
            try:
                existing = target.read_bytes()
            except OSError as exc:
                raise TaskArtifactError("任务产物文件无法读取") from exc
            if existing != payload:
                raise TaskArtifactError("固定任务产物路径已有不同内容")
            return relative.as_posix()
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return relative.as_posix()


__all__ = [
    "TaskArtifactError",
    "write_content_addressed_json",
    "write_named_json",
    "write_task_relative_json",
]
