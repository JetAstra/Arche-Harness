"""Agent-independent result, trace, and artifact capture for MCP tool calls."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from minichem_toolbox._secure_storage import (
    _open_directory_chain,
    atomic_write_bytes,
    cleanup_temporary_files,
    directory_lock,
    ensure_private_file,
    ensure_directory,
    lexical_absolute,
    read_file_in_directory,
    read_regular_file_snapshot,
    validate_directory,
)

from .workspace import WORKSPACE_ENV_NAMES, workspace_root
from minichem_toolbox.scope import current_scope
from .scope_auth import authenticate_current_mcp_request


_LOCK = threading.Lock()
DEFAULT_MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_FILES = 1000
LOGGER = logging.getLogger(__name__)
ACTION_RESULT_STATUSES = {
    "success",
    "partial_success",
    "invalid_request",
    "unsupported",
    "unavailable",
    "failed",
    "timeout",
    "cancelled",
}
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _cleanup_nested_artifact_temporary_files(artifact_root: Path) -> None:
    """在固定 dirfd 链内递归删除崩溃遗留临时文件，不跟随符号链接。"""

    descriptor = _open_directory_chain(artifact_root)
    pending = [descriptor]
    try:
        while pending:
            current = pending.pop()
            try:
                removed = False
                for name in os.listdir(current):
                    if name.startswith(".tmp-"):
                        try:
                            os.unlink(name, dir_fd=current)
                        except FileNotFoundError:
                            continue
                        removed = True
                        continue
                    try:
                        metadata = os.stat(name, dir_fd=current, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    except OSError:
                        raise ValueError("MiniChem trace storage is invalid") from None
                    if not stat.S_ISDIR(metadata.st_mode):
                        continue
                    try:
                        child = os.open(name, _OPEN_DIRECTORY_FLAGS, dir_fd=current)
                    except OSError:
                        raise ValueError("MiniChem trace storage is invalid") from None
                    pending.append(child)
                if removed:
                    os.fsync(current)
            finally:
                os.close(current)
    except (OSError, ValueError):
        while pending:
            leftover = pending.pop()
            with contextlib.suppress(OSError):
                os.close(leftover)
        raise


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, received {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, received {raw!r}")
    return value


def _artifact_limits() -> tuple[int, int]:
    """Validate trace settings before a chemistry operation is started."""

    return (
        _positive_environment_integer(
            "MINICHEM_MCP_MAX_ARTIFACT_FILES", DEFAULT_MAX_ARTIFACT_FILES
        ),
        _positive_environment_integer(
            "MINICHEM_MCP_MAX_ARTIFACT_BYTES", DEFAULT_MAX_ARTIFACT_BYTES
        ),
    )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _redact_persisted_identity(value: Any) -> Any:
    """为落盘副本移除调用方指纹，保持工具返回值不变。

    ``caller_request_fingerprint`` 仍需随 get transport 返回给恢复适配器核对，但不应进入
    工具结果、轨迹或错误预览。函数只复制 JSON 结构；canonical ``operation_fingerprint``
    不属于调用方指纹，继续保留。
    """

    if isinstance(value, dict):
        return {
            key: _redact_persisted_identity(item)
            for key, item in value.items()
            if key != "caller_request_fingerprint"
            and not key.endswith("_caller_request_fingerprint")
        }
    if isinstance(value, list):
        return [_redact_persisted_identity(item) for item in value]
    if isinstance(value, str):
        return value.replace("caller_request_fingerprint", "redacted_caller_identity")
    return value


def _next_sequence(root: Path) -> int:
    """在安全结果目录中跨进程保留不会重用的序号。"""

    with _LOCK:
        result_dir = root / "_tool_results"
        try:
            ensure_directory(result_dir)
            with directory_lock(result_dir, ".trace.lock") as directory_descriptor:
                cleanup_temporary_files(result_dir)
                existing: list[int] = []
                for name in os.listdir(directory_descriptor):
                    if name.startswith(".sequence-"):
                        prefix = name.removeprefix(".sequence-")
                    else:
                        prefix = name.split("_", 1)[0]
                    if prefix.isdigit():
                        existing.append(int(prefix))
                sequence = max(existing, default=0) + 1
                while True:
                    reservation = f".sequence-{sequence:04d}"
                    try:
                        descriptor = os.open(
                            reservation,
                            os.O_CREAT
                            | os.O_EXCL
                            | os.O_WRONLY
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=directory_descriptor,
                        )
                    except FileExistsError:
                        sequence += 1
                        continue
                    except OSError:
                        raise ValueError("MiniChem trace storage is invalid") from None
                    try:
                        os.fchmod(descriptor, 0o600)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    os.fsync(directory_descriptor)
                    return sequence
        except (OSError, ValueError):
            raise ValueError("MiniChem trace storage is invalid") from None


def _excluded(path: Path) -> bool:
    rel = path.relative_to(workspace_root())
    first = rel.parts[0] if rel.parts else ""
    execution_job_file = len(rel.parts) >= 2 and rel.parts[:2] == (
        "outputs",
        "execution_jobs",
    )
    return execution_job_file or first in {
        ".codex",
        ".claude",
        "_opencode",
        "_tool_results",
        "_tool_artifacts",
        "tool_logs",
    } or rel.name.startswith("_agent_output") or rel.name in {
        "_tool_trace.jsonl",
        "_meta.json",
        "_score.json",
        "_toolbox_catalog.json",
    }


def workspace_snapshot() -> dict[str, tuple[int, int]]:
    """Return size/mtime fingerprints for regular workspace files."""

    snapshot: dict[str, tuple[int, int]] = {}
    root = workspace_root()
    try:
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file() or _excluded(path):
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError:
                continue
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("MiniChem workspace snapshot is invalid") from None
    return snapshot


def _sha256(path: Path) -> str:
    """读取安全文件快照并返回摘要。"""

    try:
        return read_regular_file_snapshot(path)[1]
    except (OSError, ValueError):
        raise ValueError("MiniChem artifact path is invalid") from None


def _safe_workspace_root() -> Path:
    """返回并验证当前工作区，错误文本不包含工作区路径。"""

    scoped = current_scope()
    if scoped is not None:
        configured = scoped.workspace
    else:
        configured = next(
            (
                os.environ[name].strip()
                for name in WORKSPACE_ENV_NAMES
                if os.environ.get(name, "").strip()
            ),
            Path.cwd(),
        )
    try:
        root = lexical_absolute(configured)
        validate_directory(root)
    except (OSError, ValueError):
        raise ValueError("MiniChem trace workspace is invalid") from None
    return root


def _relative_path(root: Path, path: Path) -> str:
    """返回工作区内路径的稳定词法相对表示。"""

    try:
        return str(lexical_absolute(path).relative_to(lexical_absolute(root)))
    except (OSError, ValueError):
        raise ValueError("MiniChem trace path is invalid") from None


def _validate_tool_filename(tool_name: str) -> str:
    """把工具名限制为单一文件名组件，避免结果文件路径穿越。"""

    if (
        not tool_name
        or tool_name in {".", ".."}
        or "/" in tool_name
        or "\\" in tool_name
        or "\x00" in tool_name
    ):
        raise ValueError("MiniChem trace tool name is invalid")
    return f"0000_{tool_name}.json"


def _prepare_trace_storage(root: Path) -> tuple[Path, Path, Path]:
    """在执行工具前验证所有内部目录和目标文件，避免失败后才产生越界副作用。"""

    result_dir = root / "_tool_results"
    artifact_root = root / "_tool_artifacts"
    trace_path = root / "_tool_trace.jsonl"
    try:
        ensure_directory(result_dir)
        ensure_directory(artifact_root)
        _cleanup_nested_artifact_temporary_files(artifact_root)
        ensure_private_file(trace_path, allow_missing=True)
    except (OSError, ValueError):
        raise ValueError("MiniChem trace storage is invalid") from None
    return result_dir, artifact_root, trace_path


def _capture_changed_artifacts(
    sequence: int,
    before: dict[str, tuple[int, int]],
    *,
    max_files: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    root = _safe_workspace_root()
    after = workspace_snapshot()
    changed = [rel for rel, fingerprint in after.items() if before.get(rel) != fingerprint]
    artifact_dir = root / "_tool_artifacts" / f"{sequence:04d}"
    try:
        ensure_directory(root / "_tool_artifacts")
        ensure_directory(artifact_dir)
    except (OSError, ValueError):
        raise ValueError("MiniChem artifact storage is invalid") from None
    records: list[dict[str, Any]] = []
    for rel in changed[:max_files]:
        source = lexical_absolute(root / rel)
        try:
            raw, digest = read_regular_file_snapshot(source)
        except (OSError, ValueError):
            # 文件在快照与采集之间消失或变成符号链接时拒绝采集，绝不回退到路径跟随。
            continue
        size = len(raw)
        if size > max_bytes:
            records.append(
                {
                    "path": rel,
                    "snapshot_path": None,
                    "size_bytes": size,
                    "sha256": digest,
                    "snapshot_skipped": f"larger than {max_bytes} bytes",
                }
            )
            continue
        destination = artifact_dir / rel
        try:
            ensure_directory(destination.parent)
            atomic_write_bytes(destination.parent, destination.name, raw)
        except (OSError, ValueError):
            raise ValueError("MiniChem artifact storage is invalid") from None
        records.append(
            {
                "path": rel,
                "snapshot_path": _relative_path(root, destination),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return records


def execute_traced(
    tool_name: str,
    arguments: dict[str, Any],
    function: Callable[[], Any],
    *,
    capture_artifacts: bool = True,
) -> Any:
    """先按 MCP message 认证 scope，再执行并记录当前工作区的工具结果。"""

    with authenticate_current_mcp_request():
        return _execute_traced(
            tool_name,
            arguments,
            function,
            capture_artifacts=capture_artifacts,
        )


def _execute_traced(
    tool_name: str,
    arguments: dict[str, Any],
    function: Callable[[], Any],
    *,
    capture_artifacts: bool = True,
) -> Any:
    """执行一次工具并保存规范结果、轨迹和 artifact 记录。"""

    root = _safe_workspace_root()
    max_artifact_files, max_artifact_bytes = _artifact_limits()
    result_dir, artifact_root, trace_path = _prepare_trace_storage(root)
    result_filename = _validate_tool_filename(tool_name)
    sequence = _next_sequence(root)
    result_path = result_dir / result_filename.replace("0000", f"{sequence:04d}", 1)
    try:
        ensure_private_file(result_path, allow_missing=True)
        ensure_private_file(trace_path, allow_missing=True)
    except (OSError, ValueError):
        raise ValueError("MiniChem trace target is invalid") from None
    before = workspace_snapshot() if capture_artifacts else {}
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    status = "success"
    error = None
    operation_exception: Exception | None = None
    result: Any = None
    try:
        result = function()
        return result
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        operation_exception = exc
        raise
    finally:
        try:
            duration = time.monotonic() - started
            result_payload = _redact_persisted_identity(_jsonable(result))
            transport_status = status
            recorded_status = status
            recorded_error: Any = _redact_persisted_identity(error)
            if (
                transport_status == "success"
                and isinstance(result_payload, dict)
                and result_payload.get("status") in ACTION_RESULT_STATUSES
            ):
                recorded_status = str(result_payload["status"])
                if recorded_status not in {"success", "partial_success"}:
                    recorded_error = _redact_persisted_identity(result_payload.get("error"))
            result_payload_bytes = (
                json.dumps(
                    {
                        "status": recorded_status,
                        "transport_status": transport_status,
                        "result": result_payload,
                        "error": recorded_error,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            with directory_lock(result_dir, ".trace.lock"):
                atomic_write_bytes(result_dir, result_path.name, result_payload_bytes)
            artifacts = (
                _capture_changed_artifacts(
                    sequence,
                    before,
                    max_files=max_artifact_files,
                    max_bytes=max_artifact_bytes,
                )
                if capture_artifacts
                else []
            )
            preview = json.dumps(result_payload, ensure_ascii=False, default=str)
            event = {
                "sequence": sequence,
                "run_id": os.environ.get(
                    "MINICHEM_MCP_RUN_ID",
                    os.environ.get("RESEARCHCHEMBENCH_RUN_ID", ""),
                ),
                "tool": tool_name,
                "arguments": _redact_persisted_identity(_jsonable(arguments)),
                "status": recorded_status,
                "transport_status": transport_status,
                "started_at": started_wall,
                "duration_seconds": round(duration, 6),
                "result_path": _relative_path(root, result_path),
                "result_preview": preview[:2000],
                "error": recorded_error,
                "artifacts": artifacts,
            }
            trace_payload = (
                json.dumps(event, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            with _LOCK:
                with directory_lock(result_dir, ".trace.lock"):
                    current_trace = read_file_in_directory(
                        root,
                        trace_path.name,
                        allow_missing=True,
                    )
                    atomic_write_bytes(root, trace_path.name, current_trace + trace_payload)
        except Exception:
            if operation_exception is None:
                raise
            LOGGER.exception(
                "Trace persistence also failed for tool %s; preserving the "
                "original tool exception",
                tool_name,
            )
