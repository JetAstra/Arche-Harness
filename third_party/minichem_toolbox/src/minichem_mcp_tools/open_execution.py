"""Auditable native-software and Agent-authored program execution tools."""

from __future__ import annotations

import ctypes
import ast
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, get_args

import yaml
from packaging.specifiers import SpecifierSet
import minichem_toolbox.catalog as _catalog_state

from minichem_toolbox.artifacts import ArtifactStore, read_file_snapshot
from minichem_toolbox.gaussian_termination import (
    gaussian_archive_only_completion,
    gaussian_contains_fatal_marker,
    gaussian_output_ends_with_normal_termination,
)
from minichem_toolbox.idempotency import (
    OperationIdempotencyStore,
    calculate_request_fingerprint,
    scoped_submission_operation_id,
    scoped_submission_request_error,
    validate_submission_key,
)
from minichem_toolbox.paths import PROJECT_ROOT
from minichem_toolbox.process_identity import (
    ProcessIdentity,
    capture_process_identity,
    process_identity_activity,
)
from minichem_toolbox.runtime import (
    runtime_environment,
    runtime_names,
    runtime_python,
)
from minichem_toolbox.resource_budget import (
    GaussianQueueTicket,
    ResourceBudgetExceeded,
    ResourceStateUnavailable,
    _create_unique_private_temp,
    _ensure_private_directory,
    _open_private_directory_descriptor,
    _open_private_file_at,
    _open_private_lock,
    _sync_parent_directory,
    _verify_private_directory_identity,
    apply_gaussian_resource_policy,
    active_resource_jobs,
    active_resource_usage,
    enqueue_gaussian_job,
    fixed_job_resource_limits,
    gaussian_serial_gate_path,
    mark_gaussian_queue_handoff,
    normalize_resource_limits,
    read_gaussian_queue_snapshot,
    reconcile_stopped_native_gaussian_jobs,
    reserve_resources,
    resource_budget_record,
    validate_resource_limits,
)
from minichem_toolbox.scope import current_scope
from minichem_toolbox.models import ActionResult, ActionStatus, ArtifactRef
from minichem_toolbox.timeout_policy import (
    timeout_policy_record,
    timeout_seconds_for,
)

from .execution_models import (
    AnalysisInputInspectionRequest,
    AnalysisJobRequest,
    ArtifactDeclarationRequest,
    ExecutionResourceRequest,
    ExecutionJobListRequest,
    JobCancelRequest,
    JobCollectRequest,
    JobStatusRequest,
    NativeJobRequest,
    StagedInput,
    ExecutionArtifactReference,
    ExecutionJobReference,
    ExecutionOperationReference,
    ScientificArtifactTextReadRequest,
    VerifyExecutionReferencesRequest,
    WorkspaceTextReadRequest,
    WorkspaceTextWriteRequest,
)
from .execution_job_registry import (
    ExecutionJobRegistry,
    ExecutionJobRegistryConflict,
    ExecutionJobRegistryError,
    is_execution_job_id,
)
from .software_catalog import native_command_guide, software_documentation_recovery
from .workspace import (
    relative_workspace_path,
    resolve_workspace_output_path,
    resolve_workspace_public_path,
    resolve_workspace_path,
    workspace_root,
)


JOB_ROOT = Path("outputs") / "execution_jobs"
TERMINAL_JOB_STATES = {"success", "failed", "timeout", "cancelled"}
_CANONICAL_ACTION_RESULT_STATUSES = frozenset(get_args(ActionStatus))
# Registry phase 是异步写入的内部摘要，持久 status.json 才是作业当前状态。
# ``unknown`` 和 ``legacy`` 只用于旧记录或提交不确定性恢复，但仍属于终态前的
# 保守阶段；只要持久 status 已核对为 active/terminal，就可以容忍这类摘要滞后。
_REGISTRY_PRETERMINAL_PHASES = frozenset(
    {
        "reserved",
        "preparing",
        "accepted",
        "queued",
        "running",
        "unknown",
        "legacy",
    }
)
_REGISTRY_TERMINAL_PHASES = frozenset((*TERMINAL_JOB_STATES, "collected"))
SUPERVISOR_PATH = Path(__file__).with_name("job_supervisor.py")
JOB_CONTEXT_PATH = Path(__file__).with_name("researchchem_job.py")
SAFE_INHERITED_ENVIRONMENT = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS",
    "LM_LICENSE_FILE",
    "MLM_LICENSE_FILE",
    "RESEARCHCHEMBENCH_COMPUTE_HARD_TIMEOUT_SECONDS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    # 仅由服务启动层注入的固定 Codex 0.149.0 bwrap 配置。supervisor 会再次
    # 核对路径和 SHA-256；这些变量不会被放入 programmable_analysis child。
    "MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP",
    "MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT",
    "MINICHEM_PROGRAMMABLE_ANALYSIS_BWRAP_SHA256",
)
RUNTIME_CACHE_ENVIRONMENT_NAMES = (
    "MINICHEM_MODEL_CACHE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "TORCH_HOME",
    "FAIRCHEM_CACHE_DIR",
)
_PIDFD_OPEN_SYSCALL = 434
_PIDFD_SEND_SIGNAL_SYSCALL = 424
_MISSING_ATTRIBUTE = object()
MAX_INSPECTION_JSON_BYTES = 50 * 1024 * 1024
MAX_SCIENTIFIC_ARTIFACT_TEXT_BYTES = 5 * 1024 * 1024
SUPPORTED_SCIENTIFIC_ARTIFACT_TEXT_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/jsonl",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "chemical/sdf",
        "chemical/x-mdl-sd",
        "chemical/x-mdl-sdfile",
        "chemical/x-sdf",
        "chemical/x-xyz",
    }
)

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


@dataclass
class _ReservedJobContext:
    """保存一次 reserved job 提交期间共享的私有路径和持久记录。"""

    job_id: str
    job_directory: Path
    receipt_path: Path
    receipt: dict[str, Any] | None
    scoped: Any
    metadata: dict[str, Any]
    staged_records: list[dict[str, Any]]
    caller_request_fingerprint: str | None = None
    stdin_path: str | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    submitted_at: str | None = None
    relative_directory: str | None = None
    request_record: dict[str, Any] | None = None
    status: dict[str, Any] | None = None


def _write_private_text_atomically(path: Path, text: str) -> None:
    """用稳定 dirfd、唯一临时文件和双重 fsync 原子写入文本。"""

    _ensure_private_directory(path.parent)
    parent_descriptor, absolute_parent = _open_private_directory_descriptor(
        path.parent,
        create=False,
    )
    descriptor = -1
    temporary_name: str | None = None
    try:
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        try:
            target_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_metadata = None
        if target_metadata is not None and not stat.S_ISREG(target_metadata.st_mode):
            raise RuntimeError("Execution atomic target is not a regular file")
        descriptor, temporary_name = _create_unique_private_temp(
            parent_descriptor,
            f".{path.name}.tmp-",
        )
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        try:
            target_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_metadata = None
        if target_metadata is not None and stat.S_ISLNK(target_metadata.st_mode):
            raise RuntimeError("Execution atomic target became a symlink")
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _copy_private_file_atomically(source: Path, target: Path) -> None:
    """将普通源文件复制到私有目标，并同步唯一临时文件及其父目录。

    源文件必须是非符号链接的普通文件，目标不能预先存在；复制失败时清理
    临时文件并保留目标缺失状态，成功时目标权限为 0600，异常向调用方抛出。
    """

    source_parent_descriptor, source_parent = _open_private_directory_descriptor(
        source.parent,
        create=False,
    )
    target_parent_descriptor = -1
    try:
        _ensure_private_directory(target.parent)
        target_parent_descriptor, target_parent = _open_private_directory_descriptor(
            target.parent,
            create=False,
        )
    except BaseException:
        os.close(source_parent_descriptor)
        if target_parent_descriptor >= 0:
            os.close(target_parent_descriptor)
        raise
    temporary_descriptor = -1
    temporary_name: str | None = None
    source_descriptor = -1
    try:
        _verify_private_directory_identity(source_parent, source_parent_descriptor)
        source_descriptor = os.open(
            source.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_parent_descriptor,
        )
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("Execution copy source must be a regular file")
        _verify_private_directory_identity(target_parent, target_parent_descriptor)
        try:
            target_stat = os.stat(
                target.name,
                dir_fd=target_parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_stat = None
        if target_stat is not None:
            raise ValueError("Execution copy target already exists")
        temporary_descriptor, temporary_name = _create_unique_private_temp(
            target_parent_descriptor,
            f".{target.name}.tmp-",
        )
        os.fchmod(temporary_descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(source_descriptor, "rb") as source_handle:
            source_descriptor = -1
            with os.fdopen(temporary_descriptor, "wb") as target_handle:
                temporary_descriptor = -1
                shutil.copyfileobj(source_handle, target_handle)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        _verify_private_directory_identity(source_parent, source_parent_descriptor)
        _verify_private_directory_identity(target_parent, target_parent_descriptor)
        try:
            target_stat = os.stat(
                target.name,
                dir_fd=target_parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_stat = None
        if target_stat is not None and stat.S_ISLNK(target_stat.st_mode):
            raise RuntimeError("Execution copy target became a symlink")
        if target_stat is not None:
            raise RuntimeError("Execution copy target appeared during copy")
        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=target_parent_descriptor,
            dst_dir_fd=target_parent_descriptor,
        )
        temporary_name = None
        os.fsync(target_parent_descriptor)
        _verify_private_directory_identity(target_parent, target_parent_descriptor)
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=target_parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(source_parent_descriptor)
        os.close(target_parent_descriptor)


def _create_private_file(path: Path) -> None:
    """用稳定父目录创建 0600 空控制文件，并同步目录项。"""

    _ensure_private_directory(path.parent)
    parent_descriptor, absolute_parent = _open_private_directory_descriptor(
        path.parent,
        create=False,
    )
    descriptor = -1
    try:
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        descriptor = _open_private_file_at(
            parent_descriptor,
            path.name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            _PRIVATE_FILE_MODE,
        )
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.fsync(parent_descriptor)
            _verify_private_directory_identity(absolute_parent, parent_descriptor)
        finally:
            os.close(parent_descriptor)


def _ensure_private_marker(path: Path) -> None:
    """创建或核对一个私有标记文件，拒绝把标记写入符号链接目标。"""

    _ensure_private_directory(path.parent)
    parent_descriptor, absolute_parent = _open_private_directory_descriptor(
        path.parent,
        create=False,
    )
    descriptor = -1
    try:
        _verify_private_directory_identity(absolute_parent, parent_descriptor)
        try:
            descriptor = _open_private_file_at(
                parent_descriptor,
                path.name,
                os.O_RDONLY,
            )
        except FileNotFoundError:
            try:
                descriptor = _open_private_file_at(
                    parent_descriptor,
                    path.name,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    _PRIVATE_FILE_MODE,
                )
                os.fchmod(descriptor, _PRIVATE_FILE_MODE)
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
                return
            except FileExistsError:
                descriptor = _open_private_file_at(
                    parent_descriptor,
                    path.name,
                    os.O_RDONLY,
                )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("Execution marker must be a regular file")
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persisted_process_identity(
    record: Mapping[str, Any],
    prefix: str,
) -> ProcessIdentity | None:
    """从私有 status/receipt 读取完整进程身份；缺一字段即不可核对。"""

    raw_pid = record.get(f"{prefix}_pid")
    raw_start = record.get(f"{prefix}_start_time_ticks")
    if raw_start is None:
        raw_start = record.get(f"{prefix}_start_ticks")
    raw_boot = record.get(f"{prefix}_boot_id")
    if (
        isinstance(raw_pid, bool)
        or not isinstance(raw_pid, int)
        or isinstance(raw_start, bool)
        or not isinstance(raw_start, int)
    ):
        return None
    pid = raw_pid
    start = raw_start
    boot_id = str(raw_boot or "").strip()
    if pid <= 0 or start <= 0 or not boot_id:
        return None
    return ProcessIdentity(pid=pid, start_time_ticks=start, boot_id=boot_id)


def _verified_supervisor_identity(status: Mapping[str, Any]) -> ProcessIdentity | None:
    """只有完整且当前仍存活的 supervisor 身份才允许发送取消信号。"""

    identity = _persisted_process_identity(status, "supervisor")
    if identity is None:
        return None
    if process_identity_activity(identity.as_dict()) != "alive":
        return None
    # 再读取一次当前身份，缩短核对和 os.kill 之间的 PID 复用窗口。
    current = capture_process_identity(identity.pid)
    if (
        current.start_time_ticks != identity.start_time_ticks
        or current.boot_id != identity.boot_id
    ):
        return None
    return identity


def _require_scoped_submission(request: Any) -> dict[str, Any] | None:
    """要求 scope 副作用请求使用公开 submission_key，拒绝伪造 operation_id。"""

    identity_error = scoped_submission_request_error(request)
    if identity_error is None:
        return None
    return {
        "status": "invalid_request",
        "valid": False,
        "error": {
            "code": identity_error["code"],
            "message": identity_error["message"],
        },
        "automatic_fallback": False,
    }


def _request_submission_key(request: Any) -> str | None:
    """返回请求提交键；legacy 请求没有 scope 时保留 None。"""

    if current_scope() is None:
        return None
    return validate_submission_key(str(getattr(request, "submission_key", "")))


def _sha256(path: Path) -> str:
    """读取普通执行文件并计算 SHA-256；ArtifactRef 路径使用专用 fd 快照。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """以私有唯一临时文件和双重 fsync 原子保存 JSON 控制记录。"""

    _write_private_text_atomically(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _operation_job_id(operation_id: str) -> str:
    """按旧协议派生 32 hex job_id，仅用于旧作业恢复和兼容校验。"""

    return f"job_{hashlib.sha256(operation_id.encode('ascii')).hexdigest()[:32]}"


def _execution_job_registry() -> ExecutionJobRegistry:
    """返回当前 scope/workspace 的统一 execution job registry。"""

    scoped = current_scope()
    return ExecutionJobRegistry(
        workspace=scoped.workspace if scoped is not None else workspace_root(),
        scope_id=scoped.scope_id if scoped is not None else "",
    )


def _mark_execution_job_registry(
    registry: ExecutionJobRegistry,
    job_id: str,
    status: Mapping[str, Any] | None,
) -> None:
    """把 supervisor 提交结果同步到 registry；无法同步时保留 reservation。"""

    if not isinstance(status, Mapping):
        return
    raw_state = status.get("job_status") or status.get("status")
    if raw_state in {"queued", "running", *TERMINAL_JOB_STATES}:
        try:
            current = registry.get(job_id)
            if current is not None and current.get("phase") == "collected":
                return
            registry.mark(job_id, phase=str(raw_state))
        except (ExecutionJobRegistryError, ValueError):
            # supervisor identity 和原始 reservation 已经先落盘；查询会继续依据目录
            # 与 receipt 恢复，不能为了更新摘要而重新提交作业。
            return


@contextmanager
def _operation_lock(operation_id: str):
    root = _ensure_private_directory(resolve_workspace_output_path(str(JOB_ROOT)))
    lock_path = root / f".{hashlib.sha256(operation_id.encode('ascii')).hexdigest()}.lock"
    descriptor = _open_private_lock(lock_path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _file_snapshot(path: Path, *, role: str, source_path: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    value: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }
    if source_path is not None:
        value["source_path"] = source_path
    return value


def _input_snapshots(items: list[Any]) -> list[dict[str, Any]]:
    return [
        _file_snapshot(
            resolve_workspace_public_path(
                item.source_path,
                must_exist=True,
                purpose="Staged input",
                allow_execution_job_outputs=True,
                allow_action_artifacts=True,
            ),
            role="staged_input",
            source_path=item.source_path,
        )
        | {"target_path": item.target_path}
        for item in items
    ]


def _verify_expected_snapshots(expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for item in expected:
        path = Path(str(item["path"]))
        try:
            actual = _file_snapshot(
                path,
                role=str(item.get("role") or "file"),
                source_path=item.get("source_path"),
            )
        except (FileNotFoundError, OSError) as exc:
            mismatches.append({"path": str(path), "reason": str(exc)})
            continue
        for field in ("size_bytes", "sha256"):
            if actual[field] != item.get(field):
                mismatches.append(
                    {
                        "path": str(path),
                        "field": field,
                        "expected": item.get(field),
                        "actual": actual[field],
                    }
                )
    return mismatches


def _operation_fingerprint(
    *,
    job_type: str,
    request: Any,
    validation: dict[str, Any],
    input_snapshots: list[dict[str, Any]],
) -> str:
    request_value = request.model_dump(mode="json")
    request_value.pop("operation_id", None)
    request_value.pop("submission_key", None)
    material = {
        "schema_version": 1,
        "job_type": job_type,
        "request": request_value,
        "effective_resource_limits": validation.get("resource_limits"),
        "validated_command": validation.get("command"),
        "runtime_python": validation.get("runtime_python"),
        "script_sha256": validation.get("script_sha256"),
        "analysis_contract": validation.get("analysis_contract"),
        "input_snapshots": input_snapshots,
    }
    encoded = json.dumps(
        material, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scoped_submission_fingerprint(job_type: str, request: Any) -> str:
    """为 scope 预留阶段计算不依赖文件读取的调用方请求指纹。

    Investigation 的短 job_id 必须在验证、输入快照和 staging 之前取得，因此预留
    阶段不能使用完整执行指纹。该指纹只绑定调用方 JSON 和作业类型；成功准备后仍
    将文件快照放进作业 request/metadata，不能把模型提交的资源字段当作身份。
    """

    value = request.model_dump(mode="json")
    for identity_field in ("operation_id", "caller_request_fingerprint", "submission_key"):
        value.pop(identity_field, None)
    value["resource_limits"] = fixed_job_resource_limits()
    return calculate_request_fingerprint(
        f"{job_type}_submission",
        {"job_type": job_type, **value},
    )


def _sanitize_scoped_error_value(value: Any) -> Any:
    """保留可操作错误结构，同时替换已知宿主绝对路径。"""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_scoped_error_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_scoped_error_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_scoped_error_value(item) for item in value]
    if not isinstance(value, str):
        return value
    result = value
    roots = {str(workspace_root())}
    scoped = current_scope()
    if scoped is not None:
        roots.add(str(Path(scoped.workspace).expanduser().resolve()))
    for root in sorted(roots, key=len, reverse=True):
        result = result.replace(root, "<workspace>")
    # 验证异常偶尔会包含临时目录或运行时绝对路径；不向模型暴露这些宿主路径，
    # 但保留错误码、字段名、行号、候选修复和其余原始消息。
    result = re.sub(
        r"(?<![A-Za-z0-9_.-])/(?:tmp|mnt|home|root|var|opt)(?:/[^\s,;\]\[{}()]+)*",
        "<private-path>",
        result,
    )
    return result


def _scoped_submission_result(
    *,
    job_type: str,
    job_id: str,
    operation_id: str,
    submission_key: str,
    runtime: str,
    job_status: str,
    error: Mapping[str, Any] | None = None,
    reused: bool = False,
) -> dict[str, Any]:
    """构造 native/analysis 统一可解析的 submit receipt。"""

    response: dict[str, Any] = {
        "status": "success",
        "job_id": job_id,
        "job": {"job_id": job_id, "status": job_status},
        "job_type": job_type,
        "runtime": runtime,
        "job_status": job_status,
        "terminal": job_status in TERMINAL_JOB_STATES,
        "operation_id": operation_id,
        "submission_key": submission_key,
        "idempotent_reused": reused,
        "automatic_fallback": False,
    }
    if isinstance(error, Mapping):
        response["error"] = _sanitize_scoped_error_value(dict(error))
    return response


def _reserve_scoped_execution_identity(
    request: Any,
    *,
    job_type: str,
) -> dict[str, Any]:
    """在 scope 内先预留一个 execution job identity。"""

    scope = current_scope()
    if scope is None:
        raise RuntimeError("scoped execution identity requires an authenticated scope")
    submission_key = _request_submission_key(request)
    if submission_key is None:
        raise ValueError("scoped execution identity requires submission_key")
    operation_id = scoped_submission_operation_id(submission_key)
    fingerprint = _scoped_submission_fingerprint(job_type, request)
    registry = _execution_job_registry()
    try:
        reservation = registry.reserve(
            job_type=job_type,
            operation_id=operation_id,
            operation_fingerprint=fingerprint,
            submission_key=submission_key,
            caller_request_fingerprint=getattr(request, "caller_request_fingerprint", None),
        )
    except ExecutionJobRegistryConflict as exc:
        record = getattr(exc, "record", None)
        job_id = str(record.get("job_id") or "") if isinstance(record, Mapping) else ""
        return _operation_conflict(
            operation_id,
            job_id,
            record.get("operation_fingerprint") if isinstance(record, Mapping) else None,
            existing_caller_fingerprint=(
                record.get("caller_request_fingerprint")
                if isinstance(record, Mapping)
                else None
            ),
        )
    except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
        return _execution_job_unavailable_response()
    return {
        "scope": scope,
        "registry": registry,
        "reservation": reservation,
        "job_type": job_type,
        "operation_id": operation_id,
        "operation_fingerprint": fingerprint,
        "submission_key": submission_key,
    }


def _scoped_existing_execution_result(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """读取预留身份已有状态，重放时仍返回成功外层 receipt。"""

    reservation = identity["reservation"]
    job_id = str(reservation.job_id)
    try:
        _directory, status = _read_status(job_id)
    except _ExecutionJobUnavailable:
        return _execution_job_unavailable_response()
    job_status = status.get("status")
    if not isinstance(job_status, str):
        return _execution_job_unavailable_response()
    return _scoped_submission_result(
        job_type=str(identity["job_type"]),
        job_id=job_id,
        operation_id=str(identity["operation_id"]),
        submission_key=str(identity["submission_key"]),
        runtime=str(status.get("runtime") or "unknown"),
        job_status=job_status,
        error=status.get("error") if isinstance(status.get("error"), Mapping) else None,
        reused=True,
    )


def _persist_scoped_execution_failure(
    *,
    identity: Mapping[str, Any],
    request: Any,
    runtime: str,
    command: list[str],
    error: Mapping[str, Any],
) -> dict[str, Any] | None:
    """把预留后的验证/准备/启动失败写成 terminal failed receipt。"""

    scope = identity["scope"]
    registry = identity["registry"]
    reservation = identity["reservation"]
    job_id = str(reservation.job_id)
    job_type = str(identity["job_type"])
    operation_id = str(identity["operation_id"])
    submission_key = str(identity["submission_key"])
    directory = _job_directory(job_id, must_exist=False)
    sanitized_error = _sanitize_scoped_error_value(dict(error))
    if not isinstance(sanitized_error, Mapping):
        return None
    fixed_resources = dict(fixed_job_resource_limits())
    fixed_resources["walltime_seconds"] = timeout_seconds_for("compute")
    metadata: dict[str, Any] = {
        "label": "investigation_submission_failed",
        "submission_failure": True,
    }
    if hasattr(request, "software_id"):
        metadata["software_id"] = str(request.software_id)
        metadata["executable"] = str(request.executable)
    if hasattr(request, "runtime"):
        metadata["runtime"] = str(request.runtime)
        metadata["script_target"] = str(request.script_target)
    common = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": job_type,
        "runtime": runtime,
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
        "operation_id": operation_id,
        "operation_fingerprint": identity["operation_fingerprint"],
        "submission_key": submission_key,
    }
    caller_request_fingerprint = getattr(request, "caller_request_fingerprint", None)
    if caller_request_fingerprint is not None:
        common["caller_request_fingerprint"] = str(caller_request_fingerprint)
    try:
        if directory.exists() and directory.is_symlink():
            return None
        _ensure_private_directory(directory.parent)
        directory.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
        existing_status_path = directory / "status.json"
        if existing_status_path.is_symlink():
            return None
        if existing_status_path.is_file():
            existing_status = _read_private_json(existing_status_path)
            if not isinstance(existing_status, Mapping):
                return None
            for field_name in (
                "job_id",
                "job_type",
                "scope_id",
                "scope_workspace",
                "operation_id",
                "operation_fingerprint",
                "submission_key",
                "caller_request_fingerprint",
            ):
                expected = common.get(field_name)
                existing_value = existing_status.get(field_name)
                if (
                    expected is not None
                    and existing_value is not None
                    and existing_value != expected
                ):
                    return None
            existing_job_status = existing_status.get("status")
            if not isinstance(existing_job_status, str) or existing_job_status not in {
                "queued",
                "running",
                *TERMINAL_JOB_STATES,
            }:
                return None
            try:
                registry.mark(job_id, phase=str(existing_job_status))
            except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
                return None
            return _scoped_submission_result(
                job_type=job_type,
                job_id=job_id,
                operation_id=operation_id,
                submission_key=submission_key,
                runtime=str(existing_status.get("runtime") or runtime),
                job_status=str(existing_job_status),
                error=(
                    existing_status.get("error")
                    if isinstance(existing_status.get("error"), Mapping)
                    else sanitized_error
                ),
                reused=True,
            )
        for name in ("status.json", "request.json", "operation_receipt.json"):
            path = directory / name
            if path.is_symlink():
                return None
            if path.is_file():
                existing = _read_private_json(path)
                if isinstance(existing, Mapping):
                    for field_name in (
                        "job_id",
                        "scope_id",
                        "scope_workspace",
                        "operation_id",
                        "submission_key",
                    ):
                        existing_value = existing.get(field_name)
                        if (
                            existing_value is not None
                            and existing_value != common[field_name]
                        ):
                            return None
        submitted_at = _now()
        request_record = {
            **common,
            "command": command,
            "stdin_target": None,
            "staged_inputs": [],
            "resource_limits": fixed_resources,
            "resource_allocation": {},
            "metadata": metadata,
            "submitted_at": submitted_at,
            "automatic_fallback": False,
            "shell": False,
        }
        status = {
            **common,
            "status": "failed",
            "command": command,
            "job_directory": relative_workspace_path(directory),
            "resource_limits": fixed_resources,
            "resource_allocation": {},
            "submitted_at": submitted_at,
            "finished_at": _now(),
            "metadata": metadata,
            "execution_mode": "local",
            "error": sanitized_error,
        }
        receipt = {
            **common,
            "fingerprint": identity["operation_fingerprint"],
            "phase": "failed",
            "created_at": submitted_at,
            "updated_at": _now(),
            "error": sanitized_error,
        }
        _atomic_json(directory / "request.json", request_record)
        _atomic_json(directory / "status.json", status)
        _atomic_json(directory / "operation_receipt.json", receipt)
        registry.mark(job_id, phase="failed")
    except (ExecutionJobRegistryError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return _scoped_submission_result(
        job_type=job_type,
        job_id=job_id,
        operation_id=operation_id,
        submission_key=submission_key,
        runtime=runtime,
        job_status="failed",
        error=sanitized_error,
    )


def _operation_conflict(
    operation_id: str,
    job_id: str,
    existing_fingerprint: Any,
    *,
    existing_caller_fingerprint: Any = None,
) -> dict[str, Any]:
    error = {
        "code": "operation_id_conflict",
        "message": "operation_id is already bound to a different canonical request",
        "job_id": job_id,
        "existing_fingerprint": existing_fingerprint,
    }
    if existing_caller_fingerprint is not None:
        error["existing_caller_request_fingerprint"] = existing_caller_fingerprint
    return {
        "status": "invalid_request",
        "valid": False,
        "operation_id": operation_id,
        "job_id": job_id,
        "job_status": None,
        "terminal": False,
        "error": error,
        "automatic_fallback": False,
    }


def _operation_caller_fingerprint_missing(
    operation_id: str, job_id: str, existing_fingerprint: Any
) -> dict[str, Any]:
    """报告旧收据缺少调用方指纹，禁止把定位结果当作身份证明。"""

    return {
        "status": "invalid_request",
        "valid": False,
        "operation_id": operation_id,
        "job_id": job_id,
        "job_status": None,
        "terminal": False,
        "error": {
            "code": "operation_caller_fingerprint_missing",
            "message": (
                "已有 operation 收据缺少 caller_request_fingerprint；"
                "拒绝在未核对调用方请求身份时复用作业"
            ),
            "job_id": job_id,
            "existing_fingerprint": existing_fingerprint,
        },
        "automatic_fallback": False,
    }


def _read_existing_operation_receipt(path: Path) -> dict[str, Any] | None:
    """读取幂等作业收据；路径异常或损坏时返回缺失，交由调用方保守处理。"""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        return None
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _existing_receipt_identity_result(
    operation_id: str,
    job_id: str,
    fingerprint: str,
    receipt: Mapping[str, Any],
    *,
    submission_key: str | None,
    caller_request_fingerprint: str | None,
) -> dict[str, Any] | None:
    """核对旧收据的请求身份；匹配时返回 None 继续读取作业状态。"""

    existing_fingerprint = receipt.get("fingerprint")
    if existing_fingerprint != fingerprint:
        conflict = _operation_conflict(operation_id, job_id, existing_fingerprint)
        if submission_key is not None:
            conflict["error"]["code"] = "submission_key_conflict"
            conflict["error"]["message"] = "同一 scope/submission_key 绑定了不同规范请求"
        return conflict
    if submission_key is not None and receipt.get("submission_key") != submission_key:
        return {
            "status": "submission_in_progress",
            "operation_id": operation_id,
            "job_id": job_id,
            "job_status": None,
            "terminal": False,
            "idempotent_reused": True,
            "error": {
                "code": "operation_record_incomplete",
                "message": "native operation receipt 缺少或篡改 submission_key，已阻止重放",
            },
            "automatic_fallback": False,
        }
    if submission_key is None and receipt.get("submission_key") is not None:
        return {
            "status": "submission_in_progress",
            "operation_id": operation_id,
            "job_id": job_id,
            "job_status": None,
            "terminal": False,
            "idempotent_reused": True,
            "error": {
                "code": "operation_record_incomplete",
                "message": "native operation receipt 属于 submission_key 提交，当前请求无法核对",
            },
            "automatic_fallback": False,
        }
    existing_caller = receipt.get("caller_request_fingerprint")
    if caller_request_fingerprint is not None and existing_caller is None:
        return _operation_caller_fingerprint_missing(
            operation_id, job_id, existing_fingerprint
        )
    if (
        caller_request_fingerprint is not None
        and existing_caller != caller_request_fingerprint
    ):
        return _operation_conflict(
            operation_id,
            job_id,
            existing_fingerprint,
            existing_caller_fingerprint=existing_caller,
        )
    return None


def _existing_operation_status_result(
    operation_id: str,
    job_id: str,
    fingerprint: str,
    job_directory: Path,
    receipt: Mapping[str, Any] | None,
    *,
    submission_key: str | None,
) -> dict[str, Any]:
    """构造已有 operation 的受控恢复响应，不泄漏非普通状态文件。"""

    status_path = job_directory / "status.json"
    status_value: dict[str, Any] | None = None
    status_path_is_regular = status_path.is_file() and not status_path.is_symlink()
    if status_path_is_regular:
        try:
            candidate = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            candidate = None
        if isinstance(candidate, dict):
            status_value = candidate
    result = {
        "status": "submission_in_progress",
        "operation_id": operation_id,
        "operation_fingerprint": fingerprint,
        "job_id": job_id,
        "job_status": status_value.get("status") if status_value else None,
        "terminal": bool(status_value and status_value.get("status") in TERMINAL_JOB_STATES),
        "job_directory": relative_workspace_path(job_directory),
        "status_path": (
            relative_workspace_path(status_path) if status_path_is_regular else None
        ),
        "idempotent_reused": True,
        "error": {
            "code": "operation_record_incomplete",
            "message": (
                "The operation already owns a job identity; inspect that job instead of "
                "submitting another supervisor."
            ),
        },
        "automatic_fallback": False,
    }
    if submission_key is not None:
        result["submission_key"] = submission_key
    if receipt is not None and receipt.get("caller_request_fingerprint") is not None:
        result["caller_request_fingerprint"] = receipt["caller_request_fingerprint"]
    return result


def _existing_operation_result(
    operation_id: str,
    job_id: str,
    fingerprint: str,
    *,
    submission_key: str | None = None,
    caller_request_fingerprint: str | None = None,
) -> dict[str, Any]:
    job_directory = _job_directory(job_id, must_exist=False)
    receipt = _read_existing_operation_receipt(job_directory / "operation_receipt.json")
    if receipt is not None:
        identity_result = _existing_receipt_identity_result(
            operation_id,
            job_id,
            fingerprint,
            receipt,
            submission_key=submission_key,
            caller_request_fingerprint=caller_request_fingerprint,
        )
        if identity_result is not None:
            return identity_result
    return _existing_operation_status_result(
        operation_id,
        job_id,
        fingerprint,
        job_directory,
        receipt,
        submission_key=submission_key,
    )


def _write_submission_checkpoint(
    receipt_path: Path,
    receipt: dict[str, Any],
    supervisor_identity: ProcessIdentity,
) -> None:
    receipt.update(
        {
            "phase": "started",
            "supervisor_pid": supervisor_identity.pid,
            "supervisor_start_time_ticks": supervisor_identity.start_time_ticks,
            "supervisor_start_ticks": supervisor_identity.start_time_ticks,
            "supervisor_boot_id": supervisor_identity.boot_id,
            "updated_at": _now(),
        }
    )
    _atomic_json(receipt_path, receipt)


def _maximum_staged_bytes() -> int:
    raw = os.environ.get("MINICHEM_MAX_STAGED_INPUT_BYTES", "10737418240")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MINICHEM_MAX_STAGED_INPUT_BYTES must be an integer") from exc
    if value <= 0:
        raise ValueError("MINICHEM_MAX_STAGED_INPUT_BYTES must be positive")
    return value


def _job_directory(job_id: str, *, must_exist: bool = True) -> Path:
    path = resolve_workspace_output_path(str(JOB_ROOT / job_id))
    if must_exist and not path.is_dir():
        raise _ExecutionJobUnavailable()
    return path


class _ExecutionJobUnavailable(RuntimeError):
    """作业不存在、损坏或属于另一个 scope 时的统一内部错误。"""


def _validate_job_scope(directory: Path, status: Mapping[str, Any]) -> None:
    """核对作业持久身份与当前 scope，冲突时不泄漏作业是否存在。"""

    scope = current_scope()
    if scope is None:
        return
    persisted: list[Mapping[str, Any]] = [status]
    for name in ("request.json", "operation_receipt.json", "supervisor_spec.json"):
        path = directory / name
        if path.is_symlink():
            raise _ExecutionJobUnavailable()
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise _ExecutionJobUnavailable()
        if isinstance(value, Mapping):
            persisted.append(value)
    for value in persisted:
        if value.get("scope_id") != scope.scope_id:
            raise _ExecutionJobUnavailable()
        if value.get("scope_workspace") != str(scope.workspace):
            raise _ExecutionJobUnavailable()


def _read_existing_scope_records(directory: Path) -> list[Mapping[str, Any]]:
    """读取幂等重放所需的私有身份记录，缺失或损坏时统一拒绝。"""

    records: list[Mapping[str, Any]] = []
    for name in (
        "status.json",
        "request.json",
        "operation_receipt.json",
        "supervisor_spec.json",
    ):
        path = directory / name
        if path.is_symlink():
            raise _ExecutionJobUnavailable()
        if not path.exists():
            continue
        if not path.is_file():
            raise _ExecutionJobUnavailable()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _ExecutionJobUnavailable() from exc
        if not isinstance(value, Mapping):
            raise _ExecutionJobUnavailable()
        records.append(value)
    return records


def _scope_records_match(scope: Any, records: list[Mapping[str, Any]]) -> bool:
    """确认每一份已存在的作业身份记录属于当前 scope。"""

    return all(
        value.get("scope_id") == scope.scope_id
        and value.get("scope_workspace") == str(scope.workspace)
        for value in records
    )


def _validate_existing_job_scope(directory: Path) -> None:
    """在幂等重提返回任何作业身份前核对已有的 scope 收据。"""

    scope = current_scope()
    if scope is None:
        return
    records = _read_existing_scope_records(directory)
    # preparing 阶段可能暂时只有 operation_receipt；没有任何带 scope 的持久证据
    # 时宁可拒绝重放，也不把另一个 scope 的 operation_id 当作存在证明。
    if not records:
        raise _ExecutionJobUnavailable()
    if not _scope_records_match(scope, records):
        raise _ExecutionJobUnavailable()


def _execution_job_unavailable_response() -> dict[str, Any]:
    """生成不含 job id、路径或状态的越权/未知作业响应。"""

    return {
        "status": "invalid_request",
        "valid": False,
        "error": {
            "code": "execution_job_not_accessible",
            "message": "Execution job is unavailable in the current capability scope",
        },
        "automatic_fallback": False,
    }


def _validate_argument_paths(arguments: list[str]) -> None:
    """Reject explicit host paths while allowing ordinary non-shell CLI syntax."""

    for argument in arguments:
        candidate = argument.split("=", 1)[1] if argument.startswith("-") and "=" in argument else argument
        candidate = candidate.strip()
        if not candidate:
            continue
        path = PurePosixPath(candidate.replace("\\", "/"))
        if path.is_absolute() or candidate.startswith("~") or ".." in path.parts:
            raise ValueError(
                f"Native arguments cannot reference absolute paths or '..': {argument!r}; "
                "stage the file and use its target_path instead"
            )


def _compute_resource_limits(request_limits) -> dict[str, Any]:
    resources = normalize_resource_limits(request_limits)
    validate_resource_limits(resources)
    resources["walltime_seconds"] = timeout_seconds_for("compute")
    return resources


def _effective_native_resource_limits(request: NativeJobRequest) -> dict[str, Any]:
    """返回 native 作业实际使用的统一固定资源，忽略模型资源字段。"""

    del request
    return fixed_job_resource_limits()


def _effective_analysis_resource_limits(
    request: AnalysisJobRequest,
) -> dict[str, Any]:
    """返回 Python 分析作业实际使用的统一固定资源。"""

    del request
    return fixed_job_resource_limits()


def _resource_budget_error(exc: ResourceBudgetExceeded) -> dict[str, Any]:
    error = exc.as_error()
    availability = _resource_availability()
    error["blocking_resources"] = [
        name
        for name in ("cpu_cores", "memory_mb", "gpu_count")
        if int(error["requested"].get(name, 0)) > int(error["available"].get(name, 0))
    ]
    error["retry_condition"] = (
        "Retry after one of the listed active jobs reaches a terminal state, or submit a request "
        "that fits the reported available resources."
        if exc.aggregate
        else "Lower the request so every resource fits the fixed per-task budget."
    )
    return {
        "status": "invalid_request",
        "valid": False,
        "error": error,
        "evaluation_resource_budget": resource_budget_record(),
        "resource_availability": availability,
    }


def _analysis_failure(
    *,
    stage: str,
    code: str,
    message: str,
    file: str | None = None,
    line: int | None = None,
    evidence: str | None = None,
    candidate_fixes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "invalid_request",
        "valid": False,
        "error": {
            "stage": stage,
            "code": code,
            "message": message,
            "file": file,
            "line": line,
            "evidence": evidence,
            "likely_cause": message,
            "candidate_fixes": candidate_fixes or [],
            "retryable": True,
        },
        "automatic_fallback": False,
    }


def _stage_inputs(job_directory: Path, items: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    maximum = _maximum_staged_bytes()
    for item in items:
        source = resolve_workspace_public_path(
            item.source_path,
            must_exist=True,
            purpose="Staged input",
            allow_execution_job_outputs=True,
            allow_action_artifacts=True,
        )
        if source.is_symlink() or not source.is_file():
            raise ValueError("Staged input must be a regular non-symlink file")
        size = source.stat().st_size
        total_bytes += size
        if total_bytes > maximum:
            raise ValueError(
                f"Staged inputs exceed MINICHEM_MAX_STAGED_INPUT_BYTES={maximum}"
            )
        target = job_directory.joinpath(*PurePosixPath(item.target_path).parts)
        _ensure_private_directory(target.parent)
        if target.is_symlink() or target.exists():
            raise ValueError(f"Duplicate or pre-existing staged target: {item.target_path}")
        _copy_private_file_atomically(source, target)
        records.append(
            {
                "source_path": relative_workspace_path(source),
                "target_path": item.target_path,
                "size_bytes": size,
                "sha256": _sha256(target),
            }
        )
    return records


def _normalize_gaussian_staged_input(
    job_directory: Path,
    stdin_target: str | None,
    resource_limits: Mapping[str, Any],
) -> dict[str, Any] | None:
    """在隔离作业副本中覆盖 Gaussian CPU/内存指令，不改动 route 或结构。"""

    if not stdin_target:
        return None
    target = job_directory.joinpath(*PurePosixPath(stdin_target).parts)
    if target.is_symlink() or not target.is_file():
        raise ValueError("gaussian_stdin: staged input is not a regular file")
    text = target.read_text(encoding="utf-8")
    cpu_line = f"%NProcShared={int(resource_limits['cpu_cores'])}"
    reserved_memory_mb = int(resource_limits["memory_mb"])
    # Gaussian 的 %Mem 只描述计算数据可用内存，不包含程序本身、共享库和
    # 进程管理开销。资源预约仍占用完整额度，输入文件仅使用其中 99%，避免
    # 正常作业因为几十 MB 的运行开销被误判为越过任务总额度。
    gaussian_memory_mb = max(1, reserved_memory_mb * 99 // 100)
    memory_line = f"%Mem={gaussian_memory_mb}MB"
    segments = re.split(
        r"(^[ \t]*--Link1--[ \t]*$)",
        text,
        flags=re.MULTILINE,
    )
    rewritten: list[str] = []
    segment_count = 0
    for part in segments:
        if (part or "").strip().casefold() == "--link1--":
            rewritten.append(part)
            continue
        segment_count += 1
        lines = part.splitlines(keepends=True)
        found_cpu = False
        found_memory = False
        normalized_lines: list[str] = []
        for line in lines:
            line_ending = "\n" if line.endswith("\n") else ""
            if re.match(r"^[ \t]*%NProcShared\s*=", line, flags=re.IGNORECASE):
                if found_cpu:
                    continue
                normalized_lines.append(cpu_line + line_ending)
                found_cpu = True
            elif re.match(r"^[ \t]*%Mem\s*=", line, flags=re.IGNORECASE):
                if found_memory:
                    continue
                normalized_lines.append(memory_line + line_ending)
                found_memory = True
            else:
                normalized_lines.append(line)
        insertion_index = next(
            (
                index
                for index, line in enumerate(normalized_lines)
                if re.match(r"^[ \t]*#", line)
            ),
            None,
        )
        if insertion_index is None:
            raise ValueError("gaussian_route_missing: normalized deck has no route section")
        insertions: list[str] = []
        if not found_cpu:
            insertions.append(cpu_line + "\n")
        if not found_memory:
            insertions.append(memory_line + "\n")
        if insertions:
            normalized_lines[insertion_index:insertion_index] = insertions
        rewritten.extend(normalized_lines)
    normalized_text = "".join(rewritten)
    if not normalized_text.endswith("\n"):
        normalized_text += "\n"
    _write_private_text_atomically(target, normalized_text)
    return {
        "mode": "service_managed_full_capacity",
        "concurrency": "serial",
        "segment_count": segment_count,
        "cpu_cores": int(resource_limits["cpu_cores"]),
        "memory_mb": gaussian_memory_mb,
        "reserved_memory_mb": reserved_memory_mb,
        "memory_headroom_mb": reserved_memory_mb - gaussian_memory_mb,
        "input_target": stdin_target,
    }


def _write_workspace_text_once(request: WorkspaceTextWriteRequest) -> dict[str, Any]:
    """执行一次工作区文本写入；幂等收据由外层入口负责。"""

    # 先做不创建目录的公开路径检查，再走输出目录约束；框架内部仍可使用
    # resolve_workspace_output_path 写入自己的作业状态。
    resolve_workspace_public_path(request.path, purpose="Workspace text")
    try:
        path = resolve_workspace_output_path(request.path)
    except (OSError, ValueError) as exc:
        raise ValueError("Workspace text target is not writable") from exc
    relative = Path(relative_workspace_path(path))
    if relative.parts[:2] == ("outputs", "execution_jobs"):
        raise ValueError(
            "Execution job directories are immutable through write_workspace_text; "
            "stage all inputs when submitting a new job"
        )
    if path.exists() and not request.overwrite:
        raise FileExistsError(
            f"Workspace file already exists: {relative_workspace_path(path)}; set overwrite=true explicitly"
        )
    if path.exists() and not path.is_file():
        raise ValueError("Workspace text target is not a regular file")
    _write_private_text_atomically(path, request.content)
    return {
        "status": "success",
        "path": relative_workspace_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "encoding": "utf-8",
    }


def write_workspace_text(request: WorkspaceTextWriteRequest) -> dict[str, Any]:
    """写入工作区文本，并按内部身份安全回放已完成的写入。"""

    missing_identity = _require_scoped_submission(request)
    if missing_identity is not None:
        return missing_identity
    operation_id = request.operation_id
    submission_key = _request_submission_key(request)
    if submission_key is not None:
        operation_id = scoped_submission_operation_id(submission_key)
    if operation_id is None:
        return _write_workspace_text_once(request)
    store = OperationIdempotencyStore()
    canonical_request = request.model_dump(mode="json")
    canonical_request.pop("submission_key", None)
    if submission_key is not None:
        return dict(
            store.execute_scoped_submission(
                submission_key,
                calculate_request_fingerprint("write_workspace_text", canonical_request),
                lambda: _write_workspace_text_once(request),
            )
        )
    fingerprint = calculate_request_fingerprint("write_workspace_text", canonical_request)
    result = store.execute_idempotent_operation(
        operation_id,
        fingerprint,
        lambda: _write_workspace_text_once(request),
    )
    return dict(result)


def read_workspace_text(request: WorkspaceTextReadRequest) -> dict[str, Any]:
    """读取工作区文本或 JSON 子树，并返回可验证的字符分页元数据。"""

    path = resolve_workspace_public_path(
        request.path,
        must_exist=True,
        purpose="Workspace text",
        allow_execution_job_outputs=True,
        allow_action_artifacts=True,
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError("Workspace text source is not a regular file")
    text = path.read_text(encoding="utf-8")
    selected_text = text
    if request.json_pointer is not None:
        selected_value = _select_json_pointer_value(text, request.json_pointer)
        selected_text = json.dumps(
            selected_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    total_chars = len(selected_text)
    start = request.offset
    end = min(start + request.max_chars, total_chars)
    content = selected_text[start:end]
    truncated = end < total_chars
    return {
        "status": "success",
        "path": relative_workspace_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "encoding": "utf-8",
        "offset": start,
        "returned_chars": len(content),
        "next_offset": end if truncated else None,
        "total_chars": total_chars,
        "truncated": truncated,
        "json_pointer": request.json_pointer,
        "content": content,
    }


def read_scientific_artifact_text(
    request: ScientificArtifactTextReadRequest,
) -> dict[str, Any]:
    """按当前 scope 的 ArtifactRef 读取有限 UTF-8 文本，不返回宿主路径。

    输入是 artifact_id、字符分页和可选的期望 SHA-256；服务端只从当前 scope 的
    canonical manifest 找文件，并在返回前重新计算整个文件的 SHA。未知、跨 scope、
    非文本、损坏或超出大小限制的引用都不会返回内容或路径。
    """

    if current_scope() is None:
        return {"status": "unavailable", "automatic_fallback": False}
    source = _load_scientific_artifact_source(request)
    if source is None:
        return {"status": "unavailable", "automatic_fallback": False}
    canonical, path = source
    raw_result = _read_scientific_artifact_bytes(canonical, path, request)
    if isinstance(raw_result, dict):
        return raw_result
    text, actual_sha256 = raw_result
    return _scientific_artifact_page(request, canonical, text, actual_sha256)


def _load_scientific_artifact_source(
    request: ScientificArtifactTextReadRequest,
) -> tuple[Any, Path] | None:
    """从当前 scope 的 canonical manifest 解析 artifact 文件，不返回宿主路径给调用方。"""

    try:
        store = ArtifactStore(root=workspace_root(), create=False)
        canonical = store.find(request.artifact_id)
        path = resolve_workspace_path(canonical.path, must_exist=True)
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ):
        return None
    return canonical, path


def _read_scientific_artifact_bytes(
    canonical: Any,
    path: Path,
    request: ScientificArtifactTextReadRequest,
) -> tuple[str, str] | dict[str, Any]:
    """读取并核对 artifact 的文本类型、大小、编码和 SHA。"""

    unavailable = {"status": "unavailable", "automatic_fallback": False}
    if path.is_symlink() or not path.is_file():
        return {"status": "invalid", "reason_code": "artifact_missing", "automatic_fallback": False}
    media_type = canonical.media_type.casefold().split(";", 1)[0].strip()
    if not (
        media_type.startswith("text/")
        or media_type in SUPPORTED_SCIENTIFIC_ARTIFACT_TEXT_MEDIA_TYPES
        or media_type.endswith("+json")
    ):
        return {"status": "invalid", "reason_code": "artifact_not_text", "automatic_fallback": False}
    try:
        raw, actual_sha256 = read_file_snapshot(path)
    except (OSError, ValueError):
        return unavailable
    if len(raw) > MAX_SCIENTIFIC_ARTIFACT_TEXT_BYTES:
        return {"status": "invalid", "reason_code": "artifact_too_large", "automatic_fallback": False}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"status": "invalid", "reason_code": "artifact_not_utf8", "automatic_fallback": False}
    if actual_sha256 != canonical.sha256:
        return {"status": "invalid", "reason_code": "artifact_hash_mismatch", "automatic_fallback": False}
    if request.expected_sha256 is not None and request.expected_sha256 != actual_sha256:
        return {"status": "invalid", "reason_code": "expected_sha256_mismatch", "automatic_fallback": False}
    return text, actual_sha256


def _scientific_artifact_page(
    request: ScientificArtifactTextReadRequest,
    canonical: Any,
    text: str,
    actual_sha256: str,
) -> dict[str, Any]:
    """按字符 offset/limit 生成 artifact 文本分页结果。"""

    total_chars = len(text)
    start = request.offset
    end = min(start + request.limit, total_chars)
    content = text[start:end]
    return {
        "schema_version": 1,
        "status": "success",
        "artifact_id": canonical.artifact_id,
        "semantic_type": canonical.semantic_type,
        "media_type": canonical.media_type,
        "sha256": actual_sha256,
        "encoding": "utf-8",
        "offset": start,
        "returned_chars": len(content),
        "next_offset": end if end < total_chars else None,
        "total_chars": total_chars,
        "truncated": end < total_chars,
        "content": content,
        "automatic_fallback": False,
    }


def _select_json_pointer_value(text: str, pointer: str) -> Any:
    """按 RFC 6901 读取 JSON 子树；任何非法路径都明确报错。"""

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-standard constant: {value}")

    try:
        root = json.loads(text, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace file is not valid UTF-8 JSON") from exc
    if pointer == "":
        return root
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must be empty or start with '/'")

    current: Any = root
    for raw_token in pointer[1:].split("/"):
        token_chars: list[str] = []
        index = 0
        while index < len(raw_token):
            char = raw_token[index]
            if char != "~":
                token_chars.append(char)
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
                raise ValueError(f"Invalid JSON Pointer escape in token: {raw_token!r}")
            token_chars.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        token = "".join(token_chars)

        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"JSON Pointer object key does not exist: {token!r}")
            current = current[token]
            continue
        if isinstance(current, list):
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token):
                raise ValueError(f"JSON Pointer array index is invalid: {token!r}")
            item_index = int(token)
            if item_index >= len(current):
                raise ValueError(f"JSON Pointer array index is out of range: {token!r}")
            current = current[item_index]
            continue
        raise ValueError("JSON Pointer traverses through a scalar value")
    return current


def _validate_pysisyphus_input_deck(
    request: NativeJobRequest,
    guide: dict[str, Any],
) -> dict[str, Any]:
    """Validate pysisyphus' versioned YAML mechanics without choosing chemistry."""

    staged = {item.target_path: item.source_path for item in request.staged_inputs}
    config_targets = [
        argument
        for argument in request.arguments
        if argument in staged and PurePosixPath(argument).suffix.lower() in {".yaml", ".yml"}
    ]
    if len(config_targets) != 1:
        raise ValueError(
            "pysisyphus/pysis requires exactly one staged .yaml or .yml argument; "
            "the argument must equal that file's staged target_path"
        )
    config_target = config_targets[0]
    source = resolve_workspace_public_path(
        staged[config_target],
        must_exist=True,
        purpose="pysisyphus config",
        allow_execution_job_outputs=True,
        allow_action_artifacts=True,
    )
    if source.is_symlink() or not source.is_file():
        raise ValueError("pysisyphus config must be a regular non-symlink file")
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("pysisyphus config exceeds the 10 MiB structural-validation limit")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"pysisyphus config is not valid UTF-8 YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("pysisyphus config must be a YAML mapping at the top level")

    contract = guide.get("configuration_contract") or {}
    allowed = set(contract.get("valid_top_level_sections") or [])
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "pysisyphus 1.0.0 config has invalid top-level section(s) "
            f"{unknown}; endpoint structures belong in geom.fn, not an endpoints section"
        )
    geom = value.get("geom")
    if not isinstance(geom, dict) or not geom.get("fn"):
        raise ValueError(
            "pysisyphus 1.0.0 requires geom.fn to name one structure, a trajectory, "
            "or a list of endpoint structures"
        )

    cos = value.get("cos")
    if cos is not None:
        if not isinstance(cos, dict):
            raise ValueError("pysisyphus cos must be a YAML mapping")
        invalid_cos = sorted(set(cos) & {"images", "endpoints", "fixendpoints"})
        if invalid_cos:
            raise ValueError(
                "pysisyphus 1.0.0 cos contains unsupported field(s) "
                f"{invalid_cos}; put images/endpoints in geom.fn and use fix_first/fix_last"
            )
        opt = value.get("opt")
        if not isinstance(opt, dict) or not opt.get("type"):
            raise ValueError("a pysisyphus chain-of-states config requires opt.type")
        path_optimizers = set(
            (contract.get("common_exact_values") or {}).get("path_optimizer_types") or []
        )
        if path_optimizers and opt["type"] not in path_optimizers:
            raise ValueError(
                f"pysisyphus opt.type={opt['type']!r} is not a supported path optimizer; "
                f"select explicitly from {sorted(path_optimizers)}"
            )

    references = geom["fn"] if isinstance(geom["fn"], list) else [geom["fn"]]
    missing = []
    for reference in references:
        if not isinstance(reference, str):
            continue
        suffix = PurePosixPath(reference).suffix.lower()
        if suffix in {".xyz", ".trj", ".pdb", ".mol", ".sdf", ".cif"} and reference not in staged:
            missing.append(reference)
    if missing:
        raise ValueError(
            "pysisyphus geom.fn references files absent from staged_inputs: "
            f"{sorted(missing)}"
        )
    return {
        "software_version": str(contract.get("tested_version") or "unknown"),
        "config_target": config_target,
        "top_level_sections": sorted(value),
        "referenced_geometry_targets": references,
    }


def _staged_sources(request: NativeJobRequest) -> dict[str, Path]:
    return {
        item.target_path: resolve_workspace_public_path(
            item.source_path,
            must_exist=True,
            purpose="Staged input",
            allow_execution_job_outputs=True,
            allow_action_artifacts=True,
        )
        for item in request.staged_inputs
    }


def _read_staged_text(
    sources: dict[str, Path], target: str, *, software_id: str
) -> str:
    source = sources.get(target)
    if source is None:
        raise ValueError(
            f"{software_id}_missing_staged_input: {target!r} must be present in staged_inputs"
        )
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{software_id}_invalid_input_file: {target!r} is not a regular file")
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError(f"{software_id}_input_too_large: {target!r} exceeds 10 MiB lint limit")
    try:
        return source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{software_id}_input_encoding: {target!r} must be UTF-8 text") from exc


def _validate_orca_input_deck(request: NativeJobRequest) -> dict[str, Any]:
    sources = _staged_sources(request)
    targets = [argument for argument in request.arguments if argument in sources]
    if len(targets) != 1 or PurePosixPath(targets[0]).suffix.casefold() != ".inp":
        raise ValueError(
            "orca_input_argument: ORCA requires exactly one staged .inp argument matching target_path"
        )
    target = targets[0]
    text = _read_staged_text(sources, target, software_id="orca")
    lines = [line.strip() for line in text.splitlines()]
    first = next((line for line in lines if line), "")
    if not first.startswith("!"):
        raise ValueError("orca_keyword_line: first non-empty input line must start with '!'")
    block_names = {
        "basis", "casscf", "cpcm", "elprop", "freq", "geom", "mdci", "method",
        "output", "pal", "plots", "rel", "scf", "tddft",
    }
    stack: list[tuple[str, int]] = []
    for line_number, line in enumerate(lines, start=1):
        match = re.match(r"^%([a-z0-9_]+)\b(.*)$", line, flags=re.IGNORECASE)
        if match and match.group(1).casefold() in block_names:
            if re.search(r"\bend\s*$", match.group(2), flags=re.IGNORECASE):
                continue
            stack.append((match.group(1).casefold(), line_number))
        elif line.casefold() == "end" and stack:
            stack.pop()
    if stack:
        name, line_number = stack[-1]
        raise ValueError(
            f"orca_unclosed_block: %{name} opened on line {line_number} has no matching end"
        )
    coordinate_headers = [index for index, line in enumerate(lines) if re.match(r"^\*\s+xyz(file)?\b", line, re.I)]
    for index in coordinate_headers:
        if not any(line == "*" for line in lines[index + 1 :]):
            raise ValueError(
                f"orca_unclosed_coordinates: coordinate section opened on line {index + 1} has no final '*'"
            )
    parallel = re.search(r"%pal\s+.*?nprocs\s+(\d+)", text, flags=re.I | re.S)
    if parallel and int(parallel.group(1)) > request.resource_limits.cpu_cores:
        raise ValueError(
            "orca_cpu_mismatch: %pal nprocs exceeds resource_limits.cpu_cores"
        )
    keyword_text = first.casefold()
    has_frequency = bool(re.search(r"\bfreq\b", keyword_text))
    has_transition_state = bool(re.search(r"\boptts\b", keyword_text))
    has_optimization = has_transition_state or bool(re.search(r"\bopt\b", keyword_text))
    if has_transition_state:
        calculation_intent = "transition_state"
    elif has_optimization and has_frequency:
        calculation_intent = "optimization_frequency"
    elif has_optimization:
        calculation_intent = "geometry_optimization"
    elif has_frequency:
        calculation_intent = "frequency"
    else:
        calculation_intent = "single_point"
    return {
        "lint_profile": "orca_high_frequency_v1",
        "input_target": target,
        "keyword_line": first,
        "calculation_intent": calculation_intent,
        "coordinate_section_count": len(coordinate_headers),
        "checks": ["keyword_line", "block_closure", "coordinate_closure", "cpu_mapping"],
    }


def _gaussian_segment_sections(segment: str, segment_number: int) -> dict[str, Any]:
    lines = segment.splitlines()
    route_start = next((index for index, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if route_start is None:
        raise ValueError(f"gaussian_route_missing: Link1 segment {segment_number} has no route section")
    route_end = next((index for index in range(route_start + 1, len(lines)) if not lines[index].strip()), None)
    if route_end is None:
        raise ValueError(
            f"gaussian_route_separator: Link1 segment {segment_number} needs a blank line after the route"
        )
    route = " ".join(line.strip() for line in lines[route_start:route_end])
    route_normalized = route.casefold()
    has_frequency = bool(re.search(r"(?:^|[\s,])freq(?:\b|=)", route_normalized))
    has_optimization = bool(re.search(r"(?:^|[\s,])opt(?:\b|=)", route_normalized))
    has_reaction_path = bool(
        re.search(r"(?:^|[\s,])irc(?:\b|=)", route_normalized)
    )
    has_transition_state = has_optimization and bool(
        re.search(r"opt\s*=\s*(?:\([^)]*\bts\b|ts\b)", route_normalized)
    )
    if has_transition_state:
        calculation_intent = "transition_state"
    elif has_reaction_path:
        calculation_intent = "reaction_path"
    elif has_optimization and has_frequency:
        calculation_intent = "optimization_frequency"
    elif has_optimization:
        calculation_intent = "geometry_optimization"
    elif has_frequency:
        calculation_intent = "frequency"
    else:
        calculation_intent = "single_point"
    if "geom=allcheck" in route_normalized:
        return {
            "route": route,
            "geometry_source": "checkpoint",
            "calculation_intent": calculation_intent,
        }
    cursor = route_end + 1
    title_start = next((index for index in range(cursor, len(lines)) if lines[index].strip()), None)
    if title_start is None:
        raise ValueError(f"gaussian_title_missing: Link1 segment {segment_number} has no title")
    title_end = next((index for index in range(title_start + 1, len(lines)) if not lines[index].strip()), None)
    if title_end is None:
        raise ValueError(
            f"gaussian_title_separator: Link1 segment {segment_number} needs a blank line after the title"
        )
    molecule_start = next((index for index in range(title_end + 1, len(lines)) if lines[index].strip()), None)
    if molecule_start is None or not re.fullmatch(r"[+-]?\d+\s+\d+", lines[molecule_start].strip()):
        raise ValueError(
            f"gaussian_charge_multiplicity: Link1 segment {segment_number} needs 'charge multiplicity'"
        )
    coordinate_end = next(
        (index for index in range(molecule_start + 1, len(lines)) if not lines[index].strip()),
        None,
    )
    if coordinate_end is None or coordinate_end == molecule_start + 1:
        raise ValueError(
            f"gaussian_coordinate_separator: Link1 segment {segment_number} needs coordinates and a final blank line"
        )
    return {
        "route": route,
        "geometry_source": "coordinates",
        "calculation_intent": calculation_intent,
    }


def _validate_gaussian_input_deck(
    request: NativeJobRequest,
    resource_limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if request.stdin_target is None:
        raise ValueError("gaussian_stdin: g16 requires stdin_target")
    sources = _staged_sources(request)
    text = _read_staged_text(sources, request.stdin_target, software_id="gaussian")
    if not text.endswith("\n"):
        raise ValueError("gaussian_final_newline: Gaussian input must end with a newline")
    segments = re.split(r"^[ \t]*--Link1--[ \t]*$", text, flags=re.MULTILINE)
    parsed = [
        _gaussian_segment_sections(segment, index)
        for index, segment in enumerate(segments, start=1)
    ]
    step_intents = [item["calculation_intent"] for item in parsed]
    if "transition_state" in step_intents:
        calculation_intent = "transition_state"
    elif "reaction_path" in step_intents:
        calculation_intent = "reaction_path"
    elif "optimization_frequency" in step_intents or {
        "geometry_optimization",
        "frequency",
    }.issubset(step_intents):
        calculation_intent = "optimization_frequency"
    elif "geometry_optimization" in step_intents:
        calculation_intent = "geometry_optimization"
    elif "frequency" in step_intents:
        calculation_intent = "frequency"
    else:
        calculation_intent = "single_point"
    return {
        "lint_profile": "gaussian_high_frequency_v1",
        "input_target": request.stdin_target,
        "link1_segment_count": len(parsed),
        "segments": parsed,
        "calculation_intent": calculation_intent,
        "checks": ["route", "blank_lines", "title", "molecule", "cpu_mapping", "memory_mapping"],
        "resource_policy": "service_managed_full_capacity",
    }


def _validate_crest_invocation(request: NativeJobRequest) -> dict[str, Any]:
    sources = _staged_sources(request)
    xyz_targets = [
        argument
        for argument in request.arguments
        if argument in sources and PurePosixPath(argument).suffix.casefold() == ".xyz"
    ]
    if len(xyz_targets) != 1:
        raise ValueError(
            "crest_xyz_argument: CREST requires exactly one staged XYZ positional argument"
        )
    intent_by_flag = {
        "-protonate": "protonation",
        "--protonate": "protonation",
        "-deprotonate": "deprotonation",
        "--deprotonate": "deprotonation",
        "-tautomerize": "tautomerization",
        "--tautomerize": "tautomerization",
    }
    selected_flags = sorted(
        set(intent_by_flag) & {item.casefold() for item in request.arguments}
    )
    selected_intents = sorted({intent_by_flag[item] for item in selected_flags})
    if len(selected_intents) > 1:
        raise ValueError(
            f"crest_conflicting_modes: choose only one of {selected_flags}"
        )
    for flag in ("--t", "-t"):
        lowered = [item.casefold() for item in request.arguments]
        if flag in lowered:
            index = lowered.index(flag)
            if index + 1 >= len(request.arguments) or not request.arguments[index + 1].isdigit():
                raise ValueError("crest_thread_argument: --T requires a positive integer")
            if int(request.arguments[index + 1]) > request.resource_limits.cpu_cores:
                raise ValueError("crest_cpu_mismatch: --T exceeds resource_limits.cpu_cores")
    text = _read_staged_text(sources, xyz_targets[0], software_id="crest")
    lines = text.splitlines()
    if not lines or not lines[0].strip().isdigit() or len(lines) < int(lines[0]) + 2:
        raise ValueError("crest_xyz_format: XYZ atom count does not match the coordinate records")
    calculation_intent = selected_intents[0] if selected_intents else "conformer_search"
    return {
        "lint_profile": "crest_high_frequency_v1",
        "input_target": xyz_targets[0],
        "selected_mode": selected_flags[0] if selected_flags else "conformer_search",
        "calculation_intent": calculation_intent,
        "checks": ["xyz_format", "mode_exclusivity", "cpu_mapping"],
    }


def _validate_goodvibes_invocation(request: NativeJobRequest) -> dict[str, Any]:
    if any(
        argument.casefold() in {"-h", "--help", "-v", "--version"}
        for argument in request.arguments
    ):
        return {
            "lint_profile": "goodvibes_native_v1",
            "input_targets": [],
            "calculation_intent": "startup_probe",
            "checks": ["help_or_version_probe"],
        }
    sources = _staged_sources(request)
    positional_targets = [argument for argument in request.arguments if argument in sources]
    if not positional_targets:
        raise ValueError(
            "goodvibes_input_argument: GoodVibes requires at least one explicitly staged "
            "quantum-output positional argument"
        )
    unsafe = [
        target
        for target in positional_targets
        if PurePosixPath(target).parent != PurePosixPath(".")
        or not re.fullmatch(r"[A-Za-z0-9_.+,-]+", target)
    ]
    if unsafe:
        raise ValueError(
            "goodvibes_unsafe_staged_target: GoodVibes performs its own filename pattern "
            f"handling and cannot reliably consume staged targets {unsafe}. Stage each input "
            "under a flat basename containing only letters, digits, dot, underscore, plus, "
            "comma, or hyphen, then pass that exact target in arguments."
        )
    lowered = [argument.casefold() for argument in request.arguments]
    spc_suffix = None
    if "--spc" in lowered:
        index = lowered.index("--spc")
        if index + 1 >= len(request.arguments):
            raise ValueError("goodvibes_spc_suffix_missing: --spc requires a suffix")
        spc_suffix = request.arguments[index + 1]
        if spc_suffix.startswith("_"):
            raise ValueError(
                "goodvibes_spc_suffix_leading_underscore: pass the suffix without the "
                "separator underscore; GoodVibes inserts '_' between the frequency basename "
                "and suffix (use --spc DLPNO for name_DLPNO.out)"
            )
        if spc_suffix.casefold() != "link":
            staged_targets = set(sources)
            missing_pairs = []
            for target in positional_targets:
                stem = PurePosixPath(target).stem
                candidates = {
                    f"{stem}_{spc_suffix}.log",
                    f"{stem}_{spc_suffix}.out",
                }
                if not candidates & staged_targets:
                    missing_pairs.append(
                        {"frequency": target, "expected_one_of": sorted(candidates)}
                    )
            if missing_pairs:
                raise ValueError(
                    "goodvibes_spc_pair_missing: stage a matching single-point file for "
                    f"every frequency input; missing={missing_pairs}"
                )
    return {
        "lint_profile": "goodvibes_native_v1",
        "input_targets": positional_targets,
        "calculation_intent": "other",
        "checks": [
            "staged_positional_inputs",
            "safe_flat_basenames",
            "spc_suffix_pairing" if spc_suffix else "no_spc_suffix",
        ],
    }


def _require_fixed_targets(
    request: NativeJobRequest, software_id: str, required: set[str]
) -> dict[str, Path]:
    sources = _staged_sources(request)
    missing = sorted(required - set(sources))
    if missing:
        raise ValueError(
            f"{software_id}_missing_fixed_files: stage exact target names {missing}"
        )
    empty = sorted(target for target in required if sources[target].stat().st_size == 0)
    if empty:
        raise ValueError(f"{software_id}_empty_fixed_files: {empty}")
    return sources


def _validate_vasp_inputs(request: NativeJobRequest) -> dict[str, Any]:
    sources = _require_fixed_targets(request, "vasp", {"INCAR", "POSCAR", "POTCAR", "KPOINTS"})
    incar_text = _read_staged_text(sources, "INCAR", software_id="vasp")
    poscar = _read_staged_text(sources, "POSCAR", software_id="vasp").splitlines()
    if len(poscar) < 8:
        raise ValueError("vasp_poscar_structure: POSCAR is too short")
    species = poscar[5].split()
    counts_line = poscar[6].split()
    if species and all(re.fullmatch(r"[+-]?\d+", item) for item in species):
        counts_line = species
        species = []
    if not counts_line or not all(item.isdigit() and int(item) > 0 for item in counts_line):
        raise ValueError("vasp_poscar_counts: POSCAR species counts are invalid")
    if species and len(species) != len(counts_line):
        raise ValueError("vasp_poscar_species: element and count fields have different lengths")
    potcar = _read_staged_text(sources, "POTCAR", software_id="vasp")
    datasets = max(len(re.findall(r"^\s*TITEL\s*=", potcar, flags=re.M)), len(re.findall(r"^\s*VRHFIN\s*=", potcar, flags=re.M)))
    if species and datasets and datasets != len(species):
        raise ValueError(
            "vasp_potcar_order: POTCAR dataset count does not match POSCAR element count"
        )
    incar_values: dict[str, str] = {}
    for raw_line in incar_text.splitlines():
        active = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
        if "=" not in active:
            continue
        key, value = active.split("=", 1)
        incar_values[key.strip().upper()] = value.strip()
    try:
        ibrion = int(float(incar_values.get("IBRION", "-1")))
        nsw = int(float(incar_values.get("NSW", "0")))
    except ValueError as exc:
        raise ValueError("vasp_incar_integer: IBRION and NSW must be integer values") from exc
    if ibrion in {5, 6, 7, 8}:
        calculation_intent = "frequency"
    elif nsw > 0 and ibrion >= 0:
        calculation_intent = "ionic_relaxation"
    else:
        calculation_intent = "single_point"
    return {
        "lint_profile": "vasp_high_frequency_v1",
        "required_targets": sorted(sources),
        "species": species,
        "atom_count": sum(int(item) for item in counts_line),
        "potcar_dataset_count": datasets,
        "incar_ibrion": ibrion,
        "incar_nsw": nsw,
        "calculation_intent": calculation_intent,
        "checks": ["fixed_files", "poscar_counts", "potcar_dataset_count"],
    }


def _validate_lobster_inputs(request: NativeJobRequest) -> dict[str, Any]:
    required = {
        "lobsterin", "POSCAR", "POTCAR", "WAVECAR", "CONTCAR", "KPOINTS",
        "OUTCAR", "vasprun.xml",
    }
    sources = _require_fixed_targets(request, "lobster", required)
    lobsterin = _read_staged_text(sources, "lobsterin", software_id="lobster")
    if not any(line.strip() and not line.lstrip().startswith(("!", "#")) for line in lobsterin.splitlines()):
        raise ValueError("lobster_input_empty: lobsterin contains no active directives")
    return {
        "lint_profile": "lobster_high_frequency_v1",
        "required_targets": sorted(required),
        "wavecar_size_bytes": sources["WAVECAR"].stat().st_size,
        "calculation_intent": "projection",
        "checks": ["fixed_files", "nonempty_wavecar", "active_lobsterin"],
        "compatibility_boundary": (
            "File presence is validated mechanically. VASP/LOBSTER version, PAW basis, band, "
            "k-point, and projection compatibility still require software output checks."
        ),
    }


def _validate_native_input_deck(
    request: NativeJobRequest,
    guide: dict[str, Any],
    resource_limits: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    key = (guide["software_id"], request.executable)
    if key == ("pysisyphus", "pysis"):
        return _validate_pysisyphus_input_deck(request, guide)
    if key == ("orca", "orca"):
        return _validate_orca_input_deck(request)
    if key == ("gaussian", "g16"):
        return _validate_gaussian_input_deck(request, resource_limits)
    if key == ("crest", "crest"):
        return _validate_crest_invocation(request)
    if key == ("vasp", "vasp_std"):
        return _validate_vasp_inputs(request)
    if key == ("lobster", "lobster-5.1.0"):
        return _validate_lobster_inputs(request)
    if key == ("goodvibes", "goodvibes"):
        return _validate_goodvibes_invocation(request)
    return None


def _intent_is_compatible(
    software_id: str, declared: str, inferred: str
) -> bool:
    if declared in {"other", inferred}:
        return True
    aliases = {
        ("vasp", "geometry_optimization", "ionic_relaxation"),
        ("vasp", "ionic_relaxation", "geometry_optimization"),
    }
    return (software_id, declared, inferred) in aliases


def validate_native_job(request: NativeJobRequest) -> dict[str, Any]:
    guide = native_command_guide(request.software_id, request.executable)
    if not guide.get("resolved_path"):
        return {
            "status": "unavailable",
            "software_id": guide["software_id"],
            "executable": request.executable,
            "runtime": guide["runtime"],
            "error": {
                "code": "executable_missing",
                "message": (
                    f"Configured executable {request.executable!r} is not available in "
                    f"runtime {guide['runtime']!r}"
                ),
            },
            "invocation_guide": guide,
            "documentation_recovery": software_documentation_recovery(
                guide["software_id"], failed=True
            ),
        }
    _validate_argument_paths(request.arguments)
    targets = {item.target_path for item in request.staged_inputs}
    input_mode = str(guide.get("input_mode") or "")
    if input_mode in {"stdin_file", "arguments_and_stdin_file"} and request.stdin_target is None:
        raise ValueError(
            f"{request.software_id}/{request.executable} uses {input_mode}; stdin_target is required"
        )
    if request.stdin_target is not None and request.stdin_target not in targets:
        raise ValueError("stdin_target must be one of the explicitly staged target paths")
    effective_request_limits = _effective_native_resource_limits(request)
    input_deck_validation = _validate_native_input_deck(
        request,
        guide,
        effective_request_limits,
    )
    inferred_intent = (
        str(input_deck_validation.get("calculation_intent"))
        if input_deck_validation and input_deck_validation.get("calculation_intent")
        else None
    )
    if (
        request.calculation_intent
        and inferred_intent
        and not _intent_is_compatible(
            request.software_id, request.calculation_intent, inferred_intent
        )
    ):
        raise ValueError(
            "calculation_intent_mismatch: declared calculation_intent "
            f"{request.calculation_intent!r} conflicts with inferred intent {inferred_intent!r}. "
            f"Use calculation_intent={inferred_intent!r}, omit calculation_intent to accept "
            "the validated inference, or correct the input deck if the inference is wrong."
        )
    calculation_intent = inferred_intent or request.calculation_intent or "unknown"
    try:
        resources = _compute_resource_limits(effective_request_limits)
    except ResourceBudgetExceeded as exc:
        return _resource_budget_error(exc)
    return {
        "status": "success",
        "valid": True,
        "software_id": guide["software_id"],
        "runtime": guide["runtime"],
        "resolved_executable": guide["resolved_path"],
        "command": [guide["resolved_path"], *request.arguments],
        "staged_targets": sorted(targets),
        "stdin_target": request.stdin_target,
        "input_deck_validation": input_deck_validation,
        "calculation_intent": calculation_intent,
        "resource_limits": resources,
        "resource_policy": (
            "service_managed_full_capacity"
            if request.software_id == "gaussian"
            else "caller_selected_within_service_budget"
        ),
        "execution_timeout_policy": timeout_policy_record("compute"),
        "evaluation_resource_budget": resource_budget_record(),
        "resource_availability": _resource_availability(),
        "invocation_guide": guide,
        "documentation_recovery": software_documentation_recovery(
            guide["software_id"]
        ),
        "validation_boundary": (
            "Validation confirms the allowlisted executable, argv/path safety, staging map, "
            "stdin contract, mechanical resources, and any declared version-specific input-deck "
            "syntax. It does not judge scientific correctness or add missing scientific settings."
        ),
    }


def _job_environment(
    runtime: str,
    job_id: str,
    job_directory: Path,
    resources: dict[str, Any],
    resource_allocation: dict[str, Any],
    *,
    job_type: str,
) -> dict[str, str]:
    """组合 supervisor 使用的受控环境、资源可见性和私有临时目录。"""

    environment = {
        **{
            name: os.environ[name]
            for name in SAFE_INHERITED_ENVIRONMENT
            if os.environ.get(name)
        },
        **runtime_environment(runtime),
    }
    _apply_cpu_environment(environment, resources, resource_allocation, job_type)
    _apply_gpu_environment(environment, resources, resource_allocation)
    _apply_private_job_environment(environment, job_id, job_directory)
    if job_type == "programmable_analysis":
        _apply_analysis_environment(environment, job_directory)
    return environment


def _apply_cpu_environment(
    environment: dict[str, str],
    resources: Mapping[str, Any],
    resource_allocation: Mapping[str, Any],
    job_type: str,
) -> None:
    """写入 CPU 亲和性和线程上限，native 作业避免嵌套数值线程池。"""

    allocated = [int(item) for item in resource_allocation.get("cpu_ids") or []]
    if allocated:
        cpu_list = ",".join(str(item) for item in allocated)
        environment["OMPI_MCA_hwloc_base_cpu_list"] = cpu_list
        environment["PRTE_MCA_hwloc_default_cpu_list"] = cpu_list
    cpu_cores = resources.get("cpu_cores")
    if cpu_cores is None:
        return
    threads = str(max(1, int(cpu_cores)))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = threads
    if job_type == "native_software":
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["NUMEXPR_NUM_THREADS"] = "1"


def _apply_gpu_environment(
    environment: dict[str, str],
    resources: Mapping[str, Any],
    resource_allocation: Mapping[str, Any],
) -> None:
    """限制 CUDA/ROCm 可见设备到 reservation 分配的 GPU。"""

    gpu_count = int(resources.get("gpu_count") or 0)
    allocated = [str(item) for item in resource_allocation.get("gpu_ids") or []]
    if gpu_count == 0:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["ROCR_VISIBLE_DEVICES"] = ""
        return
    for variable in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        visible = [item.strip() for item in environment.get(variable, "").split(",") if item.strip()]
        selected = allocated or (visible[:gpu_count] if visible else list(map(str, range(gpu_count))))
        environment[variable] = ",".join(selected)


def _apply_private_job_environment(
    environment: dict[str, str],
    job_id: str,
    job_directory: Path,
) -> None:
    """为 supervisor 和 child 创建同一作业专属的 HOME、TMP 和报告目录。"""

    temporary = job_directory / ".tmp"
    home = job_directory / ".home"
    _ensure_private_directory(temporary)
    _ensure_private_directory(home)
    _ensure_private_directory(temporary / "matplotlib")
    if environment.get("GAUSS_SCRDIR"):
        gaussian_scratch = temporary / "gaussian"
        _ensure_private_directory(gaussian_scratch)
        environment["GAUSS_SCRDIR"] = str(gaussian_scratch)
    environment.setdefault("HOME", str(home))
    environment.update(
        {
            "TMPDIR": str(temporary),
            "MPLCONFIGDIR": str(temporary / "matplotlib"),
            "MINICHEM_EXECUTION_JOB_ID": job_id,
            "MINICHEM_EXECUTION_JOB_DIRECTORY": str(job_directory),
            "MINICHEM_JOB_ROOT": str(job_directory),
            "MINICHEM_JOB_INPUTS": str(job_directory / "inputs"),
            "MINICHEM_JOB_OUTPUTS": str(job_directory / "outputs"),
            "MINICHEM_JOB_REPORT": str(job_directory / "report"),
        }
    )


def _apply_analysis_environment(environment: dict[str, str], job_directory: Path) -> None:
    """为 Agent-authored Python 分析隔离用户 site-packages 和 bytecode。"""

    framework_paths: list[str] = []
    for raw_path in environment.get("PYTHONPATH", "").split(os.pathsep):
        if not raw_path:
            continue
        normalized = raw_path.replace("\\", "/").casefold()
        if "site-packages" in normalized or "/.tool_env" in normalized or "/.venv" in normalized:
            continue
        if raw_path not in framework_paths:
            framework_paths.append(raw_path)
    environment["PYTHONPATH"] = os.pathsep.join([str(job_directory), *framework_paths])
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"


def _start_job(
    *,
    job_id: str | None = None,
    job_type: str,
    runtime: str,
    command: list[str],
    stdin_target: str | None,
    staged_inputs: list[Any],
    resource_limits: dict[str, Any],
    metadata: dict[str, Any],
    operation_id: str | None = None,
    operation_fingerprint: str | None = None,
    submission_key: str | None = None,
    caller_request_fingerprint: str | None = None,
    expected_input_snapshots: list[dict[str, Any]] | None = None,
    launch_file_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if operation_id is not None and operation_fingerprint is None:
        raise ValueError("operation_fingerprint is required with operation_id")
    supplied_caller_fingerprint = caller_request_fingerprint
    effective_caller_fingerprint = caller_request_fingerprint or operation_fingerprint
    lock = _operation_lock(operation_id) if operation_id is not None else nullcontext()
    with lock:
        try:
            registry = _execution_job_registry()
            reservation = registry.reserve(
                job_type=job_type,
                operation_id=operation_id,
                operation_fingerprint=operation_fingerprint,
                submission_key=submission_key,
                caller_request_fingerprint=supplied_caller_fingerprint,
                preferred_job_id=job_id,
            )
        except ExecutionJobRegistryConflict as exc:
            record = getattr(exc, "record", None)
            conflict_job_id = record.get("job_id") if isinstance(record, Mapping) else job_id
            return _operation_conflict(
                str(operation_id or ""),
                str(conflict_job_id or ""),
                record.get("operation_fingerprint") if isinstance(record, Mapping) else None,
                existing_caller_fingerprint=(
                    record.get("caller_request_fingerprint")
                    if isinstance(record, Mapping)
                    else None
                ),
            )
        except ExecutionJobRegistryError:
            return _execution_job_unavailable_response()

        selected_job_id = reservation.job_id
        job_directory = _job_directory(selected_job_id, must_exist=False)
        if reservation.reused:
            if job_directory.exists():
                try:
                    _validate_existing_job_scope(job_directory)
                except _ExecutionJobUnavailable:
                    return _execution_job_unavailable_response()
                if operation_id is None:
                    return _execution_job_unavailable_response()
                return _existing_operation_result(
                    operation_id,
                    selected_job_id,
                    operation_fingerprint or "",
                    submission_key=submission_key,
                    caller_request_fingerprint=supplied_caller_fingerprint,
                )
            # 进程可能在创建私有作业目录前中断；沿用已预留的身份，不能再分配
            # 另一个短 ID。
            if reservation.record.get("phase") not in {"reserved", "preparing", "unknown"}:
                return _execution_job_unavailable_response()

        submitted = _reserve_and_start_job(
            job_id=selected_job_id,
            job_type=job_type,
            runtime=runtime,
            command=command,
            stdin_target=stdin_target,
            staged_inputs=staged_inputs,
            resource_limits=resource_limits,
            metadata=metadata,
            operation_id=operation_id,
            operation_fingerprint=operation_fingerprint,
            submission_key=submission_key,
            caller_request_fingerprint=effective_caller_fingerprint,
            expected_input_snapshots=expected_input_snapshots or [],
            launch_file_snapshots=launch_file_snapshots or [],
        )
        _mark_execution_job_registry(registry, selected_job_id, submitted)
        if isinstance(submitted, dict):
            submitted.setdefault("job_id", selected_job_id)
            job_status = submitted.get("job_status") or submitted.get("status")
            submitted.setdefault("job_status", job_status)
            submitted.setdefault("terminal", job_status in TERMINAL_JOB_STATES)
        return submitted


def _reserve_and_start_job(
    *,
    job_id: str | None = None,
    job_type: str,
    runtime: str,
    command: list[str],
    stdin_target: str | None,
    staged_inputs: list[Any],
    resource_limits: dict[str, Any],
    metadata: dict[str, Any],
    operation_id: str | None = None,
    operation_fingerprint: str | None = None,
    submission_key: str | None = None,
    caller_request_fingerprint: str | None = None,
    expected_input_snapshots: list[dict[str, Any]] | None = None,
    launch_file_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not _parent_job_is_available(metadata):
        return _execution_job_unavailable_response()
    if metadata.get("software_id") == "gaussian":
        return _reserve_and_start_gaussian_job(
            job_id=job_id,
            job_type=job_type,
            runtime=runtime,
            command=command,
            stdin_target=stdin_target,
            staged_inputs=staged_inputs,
            resource_limits=resource_limits,
            metadata=metadata,
            operation_id=operation_id,
            operation_fingerprint=operation_fingerprint,
            submission_key=submission_key,
            caller_request_fingerprint=caller_request_fingerprint,
            expected_input_snapshots=expected_input_snapshots,
            launch_file_snapshots=launch_file_snapshots,
        )
    return _reserve_and_start_regular_job(
        job_id=job_id,
        job_type=job_type,
        runtime=runtime,
        command=command,
        stdin_target=stdin_target,
        staged_inputs=staged_inputs,
        resource_limits=resource_limits,
        metadata=metadata,
        operation_id=operation_id,
        operation_fingerprint=operation_fingerprint,
        submission_key=submission_key,
        caller_request_fingerprint=caller_request_fingerprint,
        expected_input_snapshots=expected_input_snapshots,
        launch_file_snapshots=launch_file_snapshots,
    )


def _parent_job_is_available(metadata: Mapping[str, Any]) -> bool:
    """若声明了父作业，确认父作业仍属于当前 scope。"""

    parent_job_id = str(metadata.get("parent_job_id") or "").strip()
    if not parent_job_id:
        return True
    try:
        _read_status(parent_job_id)
    except _ExecutionJobUnavailable:
        return False
    return True


def _reserve_and_start_gaussian_job(
    *,
    job_id: str | None = None,
    job_type: str,
    runtime: str,
    command: list[str],
    stdin_target: str | None,
    staged_inputs: list[Any],
    resource_limits: dict[str, Any],
    metadata: dict[str, Any],
    operation_id: str | None,
    operation_fingerprint: str | None,
    submission_key: str | None,
    caller_request_fingerprint: str | None,
    expected_input_snapshots: list[dict[str, Any]] | None,
    launch_file_snapshots: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """登记 Gaussian FIFO 并启动排队 supervisor，不提前占满资源。"""

    queued_job_id = job_id or (
        _operation_job_id(operation_id)
        if operation_id is not None
        else f"job_{uuid.uuid4().hex}"
    )
    try:
        queue_ticket = enqueue_gaussian_job(
            queued_job_id,
            kind="native",
            job_id=queued_job_id,
        )
    except ResourceStateUnavailable as exc:
        return {
            "status": "unavailable",
            "valid": False,
            "error": exc.as_error(),
            "automatic_fallback": False,
        }
    try:
        result = _start_reserved_job(
            job_id=queued_job_id,
            job_type=job_type,
            runtime=runtime,
            command=command,
            stdin_target=stdin_target,
            staged_inputs=staged_inputs,
            resource_limits=normalize_resource_limits(resource_limits)
            | {"walltime_seconds": resource_limits["walltime_seconds"]},
            resource_allocation={},
            metadata={**metadata, "evaluation_resource_budget": resource_budget_record()},
            operation_id=operation_id,
            operation_fingerprint=operation_fingerprint,
            submission_key=submission_key,
            caller_request_fingerprint=caller_request_fingerprint,
            expected_input_snapshots=expected_input_snapshots or [],
            launch_file_snapshots=launch_file_snapshots or [],
            gaussian_queue_ticket=queue_ticket,
        )
    except ResourceStateUnavailable as exc:
        queue_ticket.release()
        return {
            "status": "unavailable",
            "valid": False,
            "error": exc.as_error(),
            "automatic_fallback": False,
        }
    except Exception:
        queue_ticket.release()
        raise
    if result.get("status") != "success":
        queue_ticket.release()
    return result


def _reserve_and_start_regular_job(
    *,
    job_id: str | None = None,
    job_type: str,
    runtime: str,
    command: list[str],
    stdin_target: str | None,
    staged_inputs: list[Any],
    resource_limits: dict[str, Any],
    metadata: dict[str, Any],
    operation_id: str | None,
    operation_fingerprint: str | None,
    submission_key: str | None,
    caller_request_fingerprint: str | None,
    expected_input_snapshots: list[dict[str, Any]] | None,
    launch_file_snapshots: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """为非 Gaussian 作业预留资源，并确保 reservation 最终释放。"""

    try:
        reservation = reserve_resources(
            resource_limits,
            kind=job_type,
            label=str(metadata.get("label") or command[0]),
        )
    except ResourceStateUnavailable as exc:
        error = exc.as_error()
        return {
            "status": "unavailable",
            "valid": False,
            "error": error,
            "automatic_fallback": False,
        }
    except ResourceBudgetExceeded as exc:
        return _resource_budget_error(exc)
    try:
        return _start_reserved_job(
            job_id=job_id,
            job_type=job_type,
            runtime=runtime,
            command=command,
            stdin_target=stdin_target,
            staged_inputs=staged_inputs,
            resource_limits=reservation.resource_limits
            | {"walltime_seconds": resource_limits["walltime_seconds"]},
            resource_allocation=reservation.resource_allocation,
            metadata={
                **metadata,
                "evaluation_resource_budget": resource_budget_record(),
            },
            operation_id=operation_id,
            operation_fingerprint=operation_fingerprint,
            submission_key=submission_key,
            caller_request_fingerprint=caller_request_fingerprint,
            expected_input_snapshots=expected_input_snapshots or [],
            launch_file_snapshots=launch_file_snapshots or [],
        )
    finally:
        reservation.release()


def _create_reserved_job_context(
    *,
    job_id: str | None,
    job_type: str,
    metadata: dict[str, Any],
    operation_id: str | None,
    operation_fingerprint: str | None,
    submission_key: str | None,
    caller_request_fingerprint: str | None,
) -> _ReservedJobContext:
    """创建私有作业目录、scope 元数据和初始 operation receipt。"""

    selected_job_id = job_id or (
        _operation_job_id(operation_id)
        if operation_id is not None
        else f"job_{uuid.uuid4().hex}"
    )
    job_directory = _job_directory(selected_job_id, must_exist=False)
    _ensure_private_directory(job_directory.parent)
    job_directory.mkdir(parents=True, exist_ok=False, mode=_PRIVATE_DIRECTORY_MODE)
    _ensure_private_directory(job_directory)
    if operation_id is not None:
        if caller_request_fingerprint is None:
            caller_request_fingerprint = operation_fingerprint
        metadata = {**metadata, "caller_request_fingerprint": caller_request_fingerprint}
    scoped = current_scope()
    if scoped is not None:
        metadata = {
            **metadata,
            "scope_id": scoped.scope_id,
            "scope_workspace": str(scoped.workspace),
        }
    receipt_path = job_directory / "operation_receipt.json"
    receipt = None
    if operation_id is not None:
        receipt = {
            "schema_version": 1,
            "operation_id": operation_id,
            "fingerprint": operation_fingerprint,
            "operation_fingerprint": operation_fingerprint,
            "caller_request_fingerprint": caller_request_fingerprint,
            "job_id": selected_job_id,
            "job_type": job_type,
            "submission_key": submission_key,
            "phase": "preparing",
            "created_at": _now(),
            "updated_at": _now(),
        }
        if scoped is not None:
            receipt.update(
                {
                    "scope_id": scoped.scope_id,
                    "scope_workspace": str(scoped.workspace),
                }
            )
        _atomic_json(receipt_path, receipt)
    return _ReservedJobContext(
        job_id=selected_job_id,
        job_directory=job_directory,
        receipt_path=receipt_path,
        receipt=receipt,
        scoped=scoped,
        metadata=metadata,
        staged_records=[],
        caller_request_fingerprint=caller_request_fingerprint,
    )


def _create_programmable_job_directories(context: _ReservedJobContext) -> None:
    """为 programmable analysis 创建私有目录和受控运行时辅助文件。"""

    for name in ("code", "inputs", "outputs", "report", "logs"):
        _ensure_private_directory(context.job_directory / name)
    context_path = context.job_directory / "researchchem_job.py"
    _copy_private_file_atomically(JOB_CONTEXT_PATH, context_path)
    script_target = Path(str(context.metadata.get("script_target") or "code/agent_program.py"))
    script_context_path = (
        context.job_directory / script_target.parent / "researchchem_job.py"
    )
    if script_context_path != context_path:
        # Python 将脚本所在目录放入 sys.path。辅助模块随脚本放置后，即使作业
        # 目录名称含有 PYTHONPATH 分隔符，分析程序仍能稳定导入 JobContext。
        _copy_private_file_atomically(JOB_CONTEXT_PATH, script_context_path)
    _atomic_json(
        context.job_directory / "analysis_contract.json",
        dict(context.metadata.get("analysis_contract") or {}),
    )


def _stage_reserved_job_inputs(
    context: _ReservedJobContext,
    *,
    staged_inputs: list[Any],
    stdin_target: str | None,
    resource_limits: dict[str, Any],
    expected_input_snapshots: list[dict[str, Any]],
    operation_id: str | None,
) -> dict[str, Any] | None:
    """staging 输入并规范 Gaussian 副本；返回输入变化时的拒绝结果。"""

    try:
        context.staged_records = _stage_reserved_input_files(
            context.job_directory, staged_inputs
        )
    except Exception:
        if context.receipt is None:
            shutil.rmtree(context.job_directory, ignore_errors=True)
        raise
    rejection = _reject_changed_reserved_inputs(
        context, expected_input_snapshots, operation_id
    )
    if rejection is not None:
        return rejection
    if context.metadata.get("software_id") != "gaussian":
        return None
    normalized_deck = _normalize_gaussian_staged_input(
        context.job_directory, stdin_target, resource_limits
    )
    if normalized_deck is None:
        return None
    context.metadata = {
        **context.metadata,
        "gaussian_input_normalization": normalized_deck,
    }
    for record in context.staged_records:
        if record.get("target_path") != stdin_target:
            continue
        staged_path = context.job_directory.joinpath(*PurePosixPath(str(stdin_target)).parts)
        record["size_bytes"] = staged_path.stat().st_size
        record["sha256"] = _sha256(staged_path)
    return None


def _stage_reserved_input_files(
    job_directory: Path,
    staged_inputs: list[Any],
) -> list[dict[str, Any]]:
    """为 reserved job 复制并记录输入文件，异常由上层负责回滚目录。"""

    return _stage_inputs(job_directory, staged_inputs)


def _reject_changed_reserved_inputs(
    context: _ReservedJobContext,
    expected_input_snapshots: list[dict[str, Any]],
    operation_id: str | None,
) -> dict[str, Any] | None:
    """核对提交前快照；变化时冻结失败收据并阻止启动。"""

    mismatches = _verify_expected_snapshots(expected_input_snapshots)
    if not mismatches:
        return None
    if context.receipt is not None:
        context.receipt.update(
            {"phase": "failed", "updated_at": _now(), "mismatches": mismatches}
        )
        _atomic_json(context.receipt_path, context.receipt)
    return {
        "status": "invalid_request",
        "valid": False,
        "operation_id": operation_id,
        "job_id": context.job_id,
        "error": {
            "code": "input_changed_before_launch",
            "message": "One or more staged source files changed during submission",
            "mismatches": mismatches,
        },
        "automatic_fallback": False,
    }


def _write_reserved_job_records(
    context: _ReservedJobContext,
    *,
    job_type: str,
    runtime: str,
    command: list[str],
    stdin_target: str | None,
    resource_limits: dict[str, Any],
    resource_allocation: dict[str, Any],
    operation_id: str | None,
    operation_fingerprint: str | None,
    submission_key: str | None,
    caller_request_fingerprint: str | None,
) -> None:
    """持久化 request/status 初始记录，并创建私有 stdout/stderr 文件。"""

    context.stdin_path = (
        str(context.job_directory / stdin_target) if stdin_target else None
    )
    context.stdout_path = context.job_directory / "stdout.log"
    context.stderr_path = context.job_directory / "stderr.log"
    _create_private_file(context.stdout_path)
    _create_private_file(context.stderr_path)
    context.submitted_at = _now()
    request_record: dict[str, Any] = {
        "schema_version": 1,
        "job_id": context.job_id,
        "job_type": job_type,
        "runtime": runtime,
        "command": command,
        "stdin_target": stdin_target,
        "staged_inputs": context.staged_records,
        "resource_limits": resource_limits,
        "resource_allocation": resource_allocation,
        "metadata": context.metadata,
        "submitted_at": context.submitted_at,
        "automatic_fallback": False,
        "shell": False,
    }
    if context.scoped is not None:
        request_record.update(
            {
                "scope_id": context.scoped.scope_id,
                "scope_workspace": str(context.scoped.workspace),
            }
        )
    if operation_id is not None:
        request_record.update(
            {
                "operation_id": operation_id,
                "operation_fingerprint": operation_fingerprint,
                "caller_request_fingerprint": caller_request_fingerprint,
                "submission_key": submission_key,
            }
        )
    context.request_record = request_record
    _atomic_json(context.job_directory / "request.json", request_record)
    relative_directory = relative_workspace_path(context.job_directory)
    context.relative_directory = relative_directory
    status: dict[str, Any] = {
        "schema_version": 1,
        "job_id": context.job_id,
        "job_type": job_type,
        "status": "queued",
        "supervisor_pid": None,
        "command": command,
        "job_directory": relative_directory,
        "stdout_path": relative_workspace_path(context.stdout_path),
        "stderr_path": relative_workspace_path(context.stderr_path),
        "resource_limits": resource_limits,
        "resource_allocation": resource_allocation,
        "submitted_at": context.submitted_at,
        "metadata": context.metadata,
        "execution_mode": "local",
    }
    if context.scoped is not None:
        status.update(
            {
                "scope_id": context.scoped.scope_id,
                "scope_workspace": str(context.scoped.workspace),
            }
        )
    if operation_id is not None:
        status.update(
            {
                "operation_id": operation_id,
                "operation_fingerprint": operation_fingerprint,
                "caller_request_fingerprint": caller_request_fingerprint,
                "submission_key": submission_key,
            }
        )
    context.status = status
    _atomic_json(context.job_directory / "status.json", status)


def _write_reserved_job_supervisor_spec(
    context: _ReservedJobContext,
    *,
    launch_file_snapshots: list[dict[str, Any]],
    gaussian_queue_ticket: GaussianQueueTicket | None,
) -> Path:
    """持久化 supervisor spec，并在需要时发布 Gaussian 队列 handoff。"""

    assert context.request_record is not None
    assert context.status is not None
    assert context.stdout_path is not None
    assert context.stderr_path is not None
    assert context.relative_directory is not None
    supervisor_spec = {
        **context.request_record,
        "evaluation_resource_budget": resource_budget_record(),
        "job_directory": str(context.job_directory),
        "relative_job_directory": context.relative_directory,
        "status_path": str(context.job_directory / "status.json"),
        "stdout_path": str(context.stdout_path),
        "stderr_path": str(context.stderr_path),
        "relative_stdout_path": relative_workspace_path(context.stdout_path),
        "relative_stderr_path": relative_workspace_path(context.stderr_path),
        "stdin_path": context.stdin_path,
        "launch_file_snapshots": launch_file_snapshots,
    }
    supervisor_spec["runtime_profile_environment"] = _runtime_profile_cache_environment(
        str(context.request_record.get("runtime") or "")
    )
    if context.scoped is not None:
        supervisor_spec.update(
            {
                "scope_id": context.scoped.scope_id,
                "scope_workspace": str(context.scoped.workspace),
            }
        )
    _add_gaussian_queue_spec(supervisor_spec, context, gaussian_queue_ticket)
    spec_path = context.job_directory / "supervisor_spec.json"
    _atomic_json(spec_path, supervisor_spec)
    if gaussian_queue_ticket is not None:
        mark_gaussian_queue_handoff(gaussian_queue_ticket)
    return spec_path


def _runtime_profile_cache_environment(runtime: str) -> dict[str, str]:
    """冻结 runtime profile 解析出的缓存目录，不继承父服务同名变量。"""

    profile_environment = runtime_environment(runtime)
    return {
        name: str(profile_environment[name])
        for name in RUNTIME_CACHE_ENVIRONMENT_NAMES
        if name in profile_environment
    }


def _add_gaussian_queue_spec(
    supervisor_spec: dict[str, Any],
    context: _ReservedJobContext,
    gaussian_queue_ticket: GaussianQueueTicket | None,
) -> None:
    """向 Gaussian supervisor spec 写入 FIFO ticket 或串行门路径。"""

    if context.metadata.get("software_id") != "gaussian":
        return
    if gaussian_queue_ticket is not None:
        supervisor_spec["gaussian_queue_ticket"] = gaussian_queue_ticket.as_dict()
    else:
        supervisor_spec["gaussian_serial_gate_path"] = str(gaussian_serial_gate_path())


def _launch_reserved_job_supervisor(
    context: _ReservedJobContext,
    *,
    runtime: str,
    resource_limits: dict[str, Any],
    resource_allocation: dict[str, Any],
    job_type: str,
    spec_path: Path,
) -> tuple[dict[str, Any] | None, Any | None]:
    """准备环境并启动 supervisor；失败时冻结失败状态并返回受控结果。"""

    if context.receipt is not None:
        context.receipt.update({"phase": "ready", "updated_at": _now()})
        _atomic_json(context.receipt_path, context.receipt)
    environment = _job_environment(
        runtime,
        context.job_id,
        context.job_directory,
        resource_limits,
        resource_allocation,
        job_type=job_type,
    )
    if context.receipt is not None:
        context.receipt.update({"phase": "launching", "updated_at": _now()})
        _atomic_json(context.receipt_path, context.receipt)
    try:
        supervisor = subprocess.Popen(
            [sys.executable, str(SUPERVISOR_PATH), str(spec_path)],
            cwd=context.job_directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        assert context.status is not None
        context.status.update(
            {
                "status": "failed",
                "finished_at": _now(),
                "error": {"code": "supervisor_start_failed", "message": str(exc)},
            }
        )
        _atomic_json(context.job_directory / "status.json", context.status)
        if context.receipt is not None:
            context.receipt.update(
                {
                    "phase": "failed",
                    "updated_at": _now(),
                    "error": context.status["error"],
                }
            )
            _atomic_json(context.receipt_path, context.receipt)
        return {"status": "failed", **context.status}, None
    if context.receipt is not None:
        _write_submission_checkpoint(
            context.receipt_path, context.receipt, capture_process_identity(supervisor.pid)
        )
    return None, supervisor


def _reserved_job_submission_result(
    context: _ReservedJobContext,
    *,
    job_type: str,
    runtime: str,
    command: list[str],
    resource_limits: dict[str, Any],
    resource_allocation: dict[str, Any],
    operation_id: str | None,
    operation_fingerprint: str | None,
    caller_request_fingerprint: str | None,
    submission_key: str | None,
    supervisor: Any,
) -> dict[str, Any]:
    """构造 supervisor 已启动后的公开提交结果。"""

    assert context.relative_directory is not None
    assert context.stdout_path is not None
    assert context.stderr_path is not None
    result: dict[str, Any] = {
        "status": "success",
        "job_id": context.job_id,
        "job_status": "queued",
        "terminal": False,
        "job_type": job_type,
        "runtime": runtime,
        "job_directory": context.relative_directory,
        "status_path": relative_workspace_path(context.job_directory / "status.json"),
        "stdout_path": relative_workspace_path(context.stdout_path),
        "stderr_path": relative_workspace_path(context.stderr_path),
        "command": command,
        "staged_inputs": context.staged_records,
        "resource_limits": resource_limits,
        "resource_allocation": resource_allocation,
        "supervisor_pid": supervisor.pid,
        "automatic_fallback": False,
        "evaluation_resource_budget": resource_budget_record(),
        "next_step": "Call get_execution_job with this exact job_id to inspect state and logs.",
    }
    if operation_id is not None:
        result.update(
            {
                "operation_id": operation_id,
                "operation_fingerprint": operation_fingerprint,
                "caller_request_fingerprint": caller_request_fingerprint,
                "idempotent_reused": False,
            }
        )
        if submission_key is not None:
            result["submission_key"] = submission_key
    return result


def _prepare_scoped_native_job(
    request: NativeJobRequest,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """把已验证 native 请求转换为统一 supervisor 参数。"""

    guide = validation["invocation_guide"]
    snapshots = _input_snapshots(list(request.staged_inputs))
    executable = Path(str(validation["resolved_executable"]))
    resources = dict(fixed_job_resource_limits())
    resources["walltime_seconds"] = timeout_seconds_for("compute")
    return {
        "runtime": str(validation["runtime"]),
        "command": list(validation["command"]),
        "stdin_target": request.stdin_target,
        "staged_inputs": list(request.staged_inputs),
        "resource_limits": resources,
        "metadata": {
            "software_id": validation["software_id"],
            "executable": request.executable,
            "label": request.label,
            "parent_job_id": request.parent_job_id,
            "calculation_intent": validation["calculation_intent"],
            "input_deck_validation": validation["input_deck_validation"],
            "invocation_synopsis": guide.get("synopsis"),
            "execution_timeout_policy": timeout_policy_record("compute"),
        },
        "expected_input_snapshots": snapshots,
        "launch_file_snapshots": [
            _file_snapshot(executable, role="native_executable")
        ],
    }


def _prepare_scoped_analysis_job(
    request: AnalysisJobRequest,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """把已验证 Python 分析请求转换为统一 supervisor 参数。"""

    python = Path(str(validation["runtime_python"]))
    script = resolve_workspace_public_path(
        request.script_path,
        must_exist=True,
        purpose="Analysis program",
        allow_execution_job_outputs=True,
        allow_action_artifacts=True,
    )
    staged = [
        StagedInput(
            source_path=request.script_path,
            target_path=request.script_target,
        ),
        *request.staged_inputs,
        *[
            StagedInput(source_path=item.source_path, target_path=str(item.target_path))
            for item in request.inputs
        ],
    ]
    snapshots = _input_snapshots(staged)
    resources = dict(fixed_job_resource_limits())
    resources["walltime_seconds"] = timeout_seconds_for("compute")
    return {
        "runtime": request.runtime,
        "command": [str(python), request.script_target, *request.arguments],
        "stdin_target": None,
        "staged_inputs": staged,
        "resource_limits": resources,
        "metadata": {
            "runtime": request.runtime,
            "script_source": relative_workspace_path(script),
            "script_target": request.script_target,
            "label": request.label,
            "execution_timeout_policy": timeout_policy_record("compute"),
            "analysis_contract": validation["analysis_contract"],
            "parent_job_id": request.parent_job_id,
            "script_sha256": validation["script_sha256"],
            "preflight": {
                "discovered_imports": validation["discovered_imports"],
                "required_modules": validation["required_modules"],
                "required_module_versions": validation["required_module_versions"],
                "required_symbols": validation["required_symbols"],
                "module_status": validation["module_status"],
                "module_details": validation["module_details"],
                "input_inspection": validation["input_inspection"],
                "external_execution_findings": validation[
                    "external_execution_findings"
                ],
                "job_context_compliance": validation["job_context_compliance"],
            },
            "security_boundary": (
                "Subprocess/resource/workspace convention only; deploy MCP inside an OS container "
                "or scheduler sandbox when executing untrusted programs."
            ),
        },
        "expected_input_snapshots": snapshots,
        "launch_file_snapshots": [
            _file_snapshot(python, role="analysis_runtime_python")
        ],
    }


def _submit_scoped_execution_job(
    request: NativeJobRequest | AnalysisJobRequest,
    *,
    job_type: str,
    validator: Any,
    preparer: Any,
) -> dict[str, Any]:
    """统一处理 scope native/analysis 的预留、验证、启动和失败 receipt。"""

    identity = _reserve_scoped_execution_identity(request, job_type=job_type)
    if "reservation" not in identity:
        return identity
    reservation = identity["reservation"]
    if reservation.reused:
        return _scoped_existing_execution_result(identity)
    runtime = str(getattr(request, "runtime", None) or getattr(request, "software_id", "unknown"))
    command = [
        str(getattr(request, "executable", None) or getattr(request, "script_target", "job")),
        *[str(item) for item in getattr(request, "arguments", [])],
    ]
    try:
        validation = validator(request)
    except Exception as exc:
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error={
                "stage": "validation",
                "code": "execution_validation_failed",
                "message": "execution request validation raised an exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        return failure or _execution_job_unavailable_response()
    if not isinstance(validation, Mapping):
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error={
                "stage": "validation",
                "code": "execution_validation_invalid_response",
                "message": "execution validator did not return a JSON object",
                "response_type": type(validation).__name__,
            },
        )
        return failure or _execution_job_unavailable_response()
    if validation.get("status") != "success":
        raw_error = validation.get("error")
        error = (
            dict(raw_error)
            if isinstance(raw_error, Mapping)
            else {
                "stage": "validation",
                "code": "execution_validation_failed",
                "message": "execution request validation failed",
                **(
                    {"raw_error": str(raw_error)}
                    if raw_error is not None
                    else {}
                ),
            }
        )
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error=error,
        )
        return failure or _execution_job_unavailable_response()
    try:
        prepared = preparer(request, validation)
        if not isinstance(prepared, Mapping):
            raise TypeError("execution preparation did not return a JSON object")
        runtime = str(prepared["runtime"])
        command = [str(item) for item in prepared["command"]]
    except Exception as exc:
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error={
                "stage": "preparation",
                "code": "execution_preparation_failed",
                "message": "execution inputs could not be snapshotted or staged safely",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        return failure or _execution_job_unavailable_response()
    try:
        submitted = _start_job(
            job_type=job_type,
            operation_id=identity["operation_id"],
            operation_fingerprint=identity["operation_fingerprint"],
            submission_key=identity["submission_key"],
            caller_request_fingerprint=getattr(request, "caller_request_fingerprint", None),
            **prepared,
        )
    except Exception as exc:
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error={
                "stage": "supervisor_start",
                "code": "supervisor_start_failed",
                "message": "execution supervisor failed before a job receipt was returned",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        return failure or _execution_job_unavailable_response()
    if not isinstance(submitted, Mapping):
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error={
                "stage": "supervisor_start",
                "code": "invalid_supervisor_response",
                "message": "execution supervisor did not return a JSON object",
                "response_type": type(submitted).__name__,
            },
        )
        return failure or _execution_job_unavailable_response()
    submitted = dict(submitted)
    returned_job_id = submitted.get("job_id")
    expected_job_id = str(identity["reservation"].job_id)
    if returned_job_id is not None and str(returned_job_id) != expected_job_id:
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error={
                "stage": "supervisor_start",
                "code": "execution_job_id_conflict",
                "message": "execution supervisor returned a different job_id",
                "response_job_id": str(returned_job_id),
            },
        )
        return failure or _execution_job_unavailable_response()
    status_value = str(submitted.get("job_status") or submitted.get("status") or "")
    if submitted.get("status") != "success":
        try:
            _directory, persisted = _read_status(str(identity["reservation"].job_id))
        except _ExecutionJobUnavailable:
            persisted = None
        if isinstance(persisted, Mapping) and persisted.get("status") in TERMINAL_JOB_STATES:
            try:
                identity["registry"].mark(
                    str(identity["reservation"].job_id),
                    phase=str(persisted["status"]),
                )
            except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
                return _execution_job_unavailable_response()
            return _scoped_submission_result(
                job_type=job_type,
                job_id=str(identity["reservation"].job_id),
                operation_id=str(identity["operation_id"]),
                submission_key=str(identity["submission_key"]),
                runtime=str(persisted.get("runtime") or runtime),
                job_status=str(persisted["status"]),
                error=(
                    persisted.get("error")
                    if isinstance(persisted.get("error"), Mapping)
                    else submitted.get("error")
                ),
            )
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error=(
                dict(submitted["error"])
                if isinstance(submitted.get("error"), Mapping)
                else {
                    "stage": "supervisor_start",
                    "code": "execution_start_failed",
                    "message": "execution supervisor did not accept the job",
                    **(
                        {"raw_error": str(submitted["error"])}
                        if submitted.get("error") is not None
                        else {}
                    ),
                }
            ),
        )
        return failure or _execution_job_unavailable_response()
    if status_value not in {"queued", "running", *TERMINAL_JOB_STATES}:
        failure = _persist_scoped_execution_failure(
            identity=identity,
            request=request,
            runtime=runtime,
            command=command,
            error={
                "stage": "supervisor_start",
                "code": "invalid_job_status",
                "message": "execution supervisor returned an unknown job status",
                "response_status": status_value,
            },
        )
        return failure or _execution_job_unavailable_response()
    # Investigation 只公开统一的最小 receipt；底层 supervisor 的路径、资源分配、PID
    # 和命令不是模型协议的一部分，后续一律通过 generic get/collect 观察作业。
    submitted_error = submitted.get("error")
    if not isinstance(submitted_error, Mapping) and submitted_error is not None:
        submitted_error = {
            "stage": "supervisor_start",
            "code": "execution_job_failed",
            "message": "execution supervisor returned a failed job",
            "raw_error": str(submitted_error),
        }
    return _scoped_submission_result(
        job_type=job_type,
        job_id=expected_job_id,
        operation_id=str(identity["operation_id"]),
        submission_key=str(identity["submission_key"]),
        runtime=runtime,
        job_status=status_value,
        error=submitted_error if isinstance(submitted_error, Mapping) else None,
    )


def _start_reserved_job(
    *,
    job_id: str | None = None,
    job_type: str,
    runtime: str,
    command: list[str],
    stdin_target: str | None,
    staged_inputs: list[Any],
    resource_limits: dict[str, Any],
    resource_allocation: dict[str, Any],
    metadata: dict[str, Any],
    operation_id: str | None = None,
    operation_fingerprint: str | None = None,
    submission_key: str | None = None,
    caller_request_fingerprint: str | None = None,
    expected_input_snapshots: list[dict[str, Any]] | None = None,
    launch_file_snapshots: list[dict[str, Any]] | None = None,
    gaussian_queue_ticket: GaussianQueueTicket | None = None,
) -> dict[str, Any]:
    """按固定阶段创建作业记录、发布 spec 并启动 supervisor。"""

    context = _create_reserved_job_context(
        job_id=job_id,
        job_type=job_type,
        metadata=metadata,
        operation_id=operation_id,
        operation_fingerprint=operation_fingerprint,
        submission_key=submission_key,
        caller_request_fingerprint=caller_request_fingerprint,
    )
    if job_type == "programmable_analysis":
        _create_programmable_job_directories(context)
    rejection = _stage_reserved_job_inputs(
        context,
        staged_inputs=staged_inputs,
        stdin_target=stdin_target,
        resource_limits=resource_limits,
        expected_input_snapshots=expected_input_snapshots or [],
        operation_id=operation_id,
    )
    if rejection is not None:
        return rejection
    _write_reserved_job_records(
        context,
        job_type=job_type,
        runtime=runtime,
        command=command,
        stdin_target=stdin_target,
        resource_limits=resource_limits,
        resource_allocation=resource_allocation,
        operation_id=operation_id,
        operation_fingerprint=operation_fingerprint,
        submission_key=submission_key,
        caller_request_fingerprint=context.caller_request_fingerprint,
    )
    spec_path = _write_reserved_job_supervisor_spec(
        context,
        launch_file_snapshots=launch_file_snapshots or [],
        gaussian_queue_ticket=gaussian_queue_ticket,
    )
    failure, supervisor = _launch_reserved_job_supervisor(
        context,
        runtime=runtime,
        resource_limits=resource_limits,
        resource_allocation=resource_allocation,
        job_type=job_type,
        spec_path=spec_path,
    )
    if failure is not None:
        return failure
    assert supervisor is not None
    return _reserved_job_submission_result(
        context,
        job_type=job_type,
        runtime=runtime,
        command=command,
        resource_limits=resource_limits,
        resource_allocation=resource_allocation,
        operation_id=operation_id,
        operation_fingerprint=operation_fingerprint,
        caller_request_fingerprint=context.caller_request_fingerprint,
        submission_key=submission_key,
        supervisor=supervisor,
    )


def submit_native_job(request: NativeJobRequest) -> dict[str, Any]:
    missing_identity = _require_scoped_submission(request)
    if missing_identity is not None:
        return missing_identity
    if current_scope() is not None:
        return _submit_scoped_execution_job(
            request,
            job_type="native_software",
            validator=validate_native_job,
            preparer=_prepare_scoped_native_job,
        )
    submission_key = _request_submission_key(request)
    operation_id = request.operation_id
    if submission_key is not None:
        operation_id = scoped_submission_operation_id(submission_key)
    validation = validate_native_job(request)
    if validation["status"] != "success":
        return validation
    guide = validation["invocation_guide"]
    snapshots = _input_snapshots(list(request.staged_inputs))
    fingerprint = _operation_fingerprint(
        job_type="native_software",
        request=request,
        validation=validation,
        input_snapshots=snapshots,
    )
    executable = Path(str(validation["resolved_executable"]))
    return _start_job(
        job_type="native_software",
        runtime=validation["runtime"],
        command=validation["command"],
        stdin_target=request.stdin_target,
        staged_inputs=request.staged_inputs,
        resource_limits=dict(validation["resource_limits"]),
        metadata={
            "software_id": validation["software_id"],
            "executable": request.executable,
            "label": request.label,
            "parent_job_id": request.parent_job_id,
            "calculation_intent": validation["calculation_intent"],
            "input_deck_validation": validation["input_deck_validation"],
            "invocation_synopsis": guide.get("synopsis"),
            "execution_timeout_policy": timeout_policy_record("compute"),
        },
        operation_id=operation_id,
        operation_fingerprint=fingerprint if operation_id else None,
        submission_key=submission_key,
        caller_request_fingerprint=(
            request.caller_request_fingerprint
            if operation_id and submission_key is None
            else None
        ),
        expected_input_snapshots=snapshots,
        launch_file_snapshots=[
            _file_snapshot(executable, role="native_executable")
        ],
    )


def _analysis_imports(tree: ast.AST) -> list[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return sorted(names)


def _attribute_chain(node: ast.AST) -> list[str]:
    values: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        values.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        values.append(current.id)
    return list(reversed(values))


def _job_context_symbols(tree: ast.AST) -> tuple[bool, set[str]]:
    """识别 JobContext 导入别名，并追踪由 ``JobContext.load()`` 得到的变量别名。"""

    imported = False
    class_names = {"JobContext"}
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "researchchem_job":
            for alias in node.names:
                if alias.name == "JobContext":
                    imported = True
                    class_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "researchchem_job":
                    imported = True
                    module_names.add(alias.asname or alias.name.split(".")[0])

    def is_context_load(value: ast.AST) -> bool:
        if not isinstance(value, ast.Call):
            return False
        chain = _attribute_chain(value.func)
        return bool(
            (len(chain) == 2 and chain[0] in class_names and chain[1] == "load")
            or (
                len(chain) == 3
                and chain[0] in module_names
                and chain[1:] == ["JobContext", "load"]
            )
        )

    context_names: set[str] = set()
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            value = node.value
            is_alias = isinstance(value, ast.Name) and value.id in context_names
            if not is_context_load(value) and not is_alias:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in context_names:
                    context_names.add(target.id)
                    changed = True
        if not changed:
            break
    return imported, context_names


def _job_context_compliance(
    tree: ast.AST, request: AnalysisJobRequest
) -> dict[str, Any]:
    imported, context_names = _job_context_symbols(tree)
    helper_calls: dict[str, set[str]] = {
        "input": set(),
        "output": set(),
        "write_json": set(),
        "register_output": set(),
    }
    bypass_findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        chain = _attribute_chain(node)
        if (
            len(chain) >= 3
            and chain[0] in context_names
            and chain[1] == "root"
            and chain[2] in {"parent", "parents"}
        ):
            bypass_findings.append(
                {
                    "line": getattr(node, "lineno", None),
                    "expression": ".".join(chain),
                    "reason": "walks from the isolated job root toward the benchmark workspace",
                }
            )
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        call_chain = _attribute_chain(node.func)
        if len(call_chain) != 2 or call_chain[0] not in context_names:
            continue
        helper = call_chain[1]
        if helper not in helper_calls or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            helper_calls[helper].add(first.value)

    declared_inputs = {item.name for item in request.inputs}
    declared_outputs = {item.name for item in request.outputs}
    used_outputs = (
        helper_calls["output"]
        | helper_calls["write_json"]
        | helper_calls["register_output"]
    )
    missing_input_helpers = sorted(declared_inputs - helper_calls["input"])
    missing_output_helpers = sorted(declared_outputs - used_outputs)
    unknown_input_helpers = sorted(helper_calls["input"] - declared_inputs)
    unknown_output_helpers = sorted(used_outputs - declared_outputs)
    if not imported:
        status = "not_adopted"
    elif bypass_findings:
        status = "bypassed"
    elif unknown_input_helpers or unknown_output_helpers:
        status = "invalid"
    elif not missing_input_helpers and not missing_output_helpers:
        status = "compliant"
    else:
        status = "partial"
    warnings = []
    if status == "not_adopted":
        warnings.append(
            "Use JobContext.input/output helpers for declared paths to reduce isolated-job path errors."
        )
    if missing_input_helpers:
        warnings.append(
            f"Declared inputs not statically observed through JobContext.input: {missing_input_helpers}."
        )
    if missing_output_helpers:
        warnings.append(
            f"Declared outputs not statically observed through JobContext helpers: {missing_output_helpers}."
        )
    if unknown_input_helpers:
        warnings.append(
            f"JobContext.input names absent from the request contract: {unknown_input_helpers}."
        )
    if unknown_output_helpers:
        warnings.append(
            f"JobContext output names absent from the request contract: {unknown_output_helpers}."
        )
    if bypass_findings:
        warnings.append(
            "JobContext was imported but code walks above ctx.root; this bypasses the declared path contract."
        )
    return {
        "status": status,
        "job_context_imported": imported,
        "context_variables": sorted(context_names),
        "helper_calls": {key: sorted(value) for key, value in helper_calls.items()},
        "missing_input_helpers": missing_input_helpers,
        "missing_output_helpers": missing_output_helpers,
        "unknown_input_helpers": unknown_input_helpers,
        "unknown_output_helpers": unknown_output_helpers,
        "bypass_findings": bypass_findings,
        "warnings": warnings,
        "enforcement_boundary": (
            "Static reliability audit only; ordinary Python file and library access is not blocked."
        ),
    }


def _analysis_reliability_findings(
    tree: ast.AST,
    *,
    request: AnalysisJobRequest | None = None,
) -> list[dict[str, Any]]:
    """识别会在分析任务运行时确定性失败的 JobContext/API 用法。"""

    findings: list[dict[str, Any]] = []
    _, context_names = _job_context_symbols(tree)
    # 这些是 researchchem_job.JobContext 对分析程序公开的稳定成员。
    public_context_members = {
        "contract",
        "input",
        "input_dir",
        "inputs_dir",
        "logs_dir",
        "output",
        "output_dir",
        "outputs_dir",
        "register_output",
        "report_dir",
        "root",
        "write_json",
    }
    declared_output_paths = (
        {item.name: PurePosixPath(item.path) for item in request.outputs}
        if request is not None
        else {}
    )
    path_variables: dict[str, PurePosixPath] = {}

    def resolve_context_path(node: ast.AST) -> PurePosixPath | None:
        """解析由公开 JobContext 目录和字面量组成的任务内相对路径。"""

        if isinstance(node, ast.Name):
            return path_variables.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id not in context_names:
                return None
            roots = {
                "root": PurePosixPath("."),
                "inputs_dir": PurePosixPath("inputs"),
                "outputs_dir": PurePosixPath("outputs"),
                "report_dir": PurePosixPath("report"),
                "logs_dir": PurePosixPath("logs"),
            }
            return roots.get(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            chain = _attribute_chain(node.func)
            if len(chain) != 2 or chain[0] not in context_names:
                return None
            if chain[1] in {"input_dir", "output_dir"} and not node.args:
                return PurePosixPath("inputs" if chain[1] == "input_dir" else "outputs")
            if (
                chain[1] == "output"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                return declared_output_paths.get(node.args[0].value)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
        ):
            parent = resolve_context_path(node.left)
            if parent is not None:
                return parent / node.right.value
        return None

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            resolved = resolve_context_path(node.value)
            if resolved is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and path_variables.get(target.id) != resolved:
                    path_variables[target.id] = resolved
                    changed = True
        if not changed:
            break

    written_paths: set[PurePosixPath] = set()
    register_calls: list[tuple[ast.Call, str, PurePosixPath | None, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        chain = _attribute_chain(node.func)
        if node.func.attr in {"write_text", "write_bytes", "touch"}:
            target_path = resolve_context_path(node.func.value)
            if target_path is not None:
                written_paths.add(target_path)
        if len(chain) != 2 or chain[0] not in context_names or not node.args:
            continue
        output_name_node = node.args[0]
        if not (
            isinstance(output_name_node, ast.Constant)
            and isinstance(output_name_node.value, str)
        ):
            continue
        output_name = output_name_node.value
        if chain[1] == "write_json" and output_name in declared_output_paths:
            written_paths.add(declared_output_paths[output_name])
        elif chain[1] == "register_output":
            has_explicit_path = len(node.args) > 1
            explicit_path = None
            if has_explicit_path:
                literal = node.args[1]
                if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                    candidate = PurePosixPath(literal.value)
                    explicit_path = candidate
                else:
                    # 动态路径无法从语法树安全推断，保留给运行时合同核对。
                    explicit_path = resolve_context_path(literal)
            register_calls.append(
                (node, output_name, explicit_path, has_explicit_path)
            )

    for node, output_name, explicit_path, has_explicit_path in register_calls:
        declared_path = declared_output_paths.get(output_name)
        if declared_path is None:
            continue
        if has_explicit_path:
            if explicit_path is None or explicit_path == declared_path:
                continue
            mismatch = explicit_path
            mismatch_message = (
                f"程序显式注册了 {mismatch.as_posix()!r}"
            )
        else:
            if declared_path in written_paths:
                continue
            same_filename = sorted(
                (path for path in written_paths if path.name == declared_path.name),
                key=str,
            )
            mismatch = same_filename[0] if same_filename else None
            mismatch_message = (
                f"程序实际写入 {mismatch.as_posix()!r} 后按默认路径注册"
                if mismatch is not None
                else ""
            )
        if mismatch is None or mismatch == declared_path:
            continue
        findings.append(
            {
                "line": getattr(node, "lineno", None),
                "kind": "declared_output_path_mismatch",
                "evidence": f"{mismatch.as_posix()} != {declared_path.as_posix()}",
                "message": (
                    f"声明输出 {output_name!r} 必须写入 {declared_path.as_posix()!r}；"
                    f"{mismatch_message}"
                ),
            }
        )
    for node in ast.walk(tree):
        chain = _attribute_chain(node)
        if (
            isinstance(node, ast.Attribute)
            and len(chain) == 2
            and chain[0] in context_names
            and chain[1] not in public_context_members
        ):
            findings.append(
                {
                    "line": getattr(node, "lineno", None),
                    "kind": "unknown_job_context_attribute",
                    "evidence": ".".join(chain),
                    "message": (
                        f"JobContext 没有公开成员 {chain[1]!r}；"
                        "请使用已声明输出对应的 ctx.output(name)"
                    ),
                }
            )
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Call)
            and len(_attribute_chain(node.value.func)) == 2
            and _attribute_chain(node.value.func)[0] in context_names
            and _attribute_chain(node.value.func)[1] in {"input", "output"}
        ):
            helper = _attribute_chain(node.value.func)[1]
            findings.append(
                {
                    "line": getattr(node, "lineno", None),
                    "kind": f"job_context_{helper}_path_attribute",
                    "evidence": f"JobContext.{helper}(...).path",
                    "message": (
                        f"JobContext.{helper}(name) already returns pathlib.Path; the Path object "
                        "has no .path attribute"
                    ),
                }
            )
        if not isinstance(node, ast.FormattedValue) or node.format_spec is None:
            continue
        literal_format = "".join(
            str(part.value)
            for part in ast.walk(node.format_spec)
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        if " if " in literal_format and " else " in literal_format:
            findings.append(
                {
                    "line": getattr(node, "lineno", None),
                    "kind": "conditional_inside_format_specifier",
                    "evidence": literal_format,
                    "message": (
                        "A conditional expression written after ':' is parsed as a literal "
                        "format specifier and raises ValueError"
                    ),
                }
            )
    return sorted(findings, key=lambda item: (item["line"] or 0, item["kind"]))


def _runtime_path_injection_findings(tree: ast.AST) -> list[dict[str, Any]]:
    """Find attempts to splice another Python environment into the selected runtime."""

    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        kind = None
        evidence = None
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain in (
                ["sys", "path", "insert"],
                ["sys", "path", "append"],
                ["sys", "path", "extend"],
            ):
                kind = "sys_path_mutation"
                evidence = ".".join(chain)
            elif chain == ["site", "addsitedir"]:
                kind = "site_directory_injection"
                evidence = ".".join(chain)
            elif chain == ["os", "putenv"] and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and str(first.value).upper() == "PYTHONPATH":
                    kind = "pythonpath_environment_mutation"
                    evidence = "os.putenv('PYTHONPATH', ...)"
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                chain = _attribute_chain(target)
                if chain == ["sys", "path"]:
                    kind = "sys_path_assignment"
                    evidence = "sys.path"
                    break
                if (
                    isinstance(target, ast.Subscript)
                    and _attribute_chain(target.value) == ["os", "environ"]
                    and isinstance(target.slice, ast.Constant)
                    and str(target.slice.value).upper() == "PYTHONPATH"
                ):
                    kind = "pythonpath_environment_mutation"
                    evidence = "os.environ['PYTHONPATH']"
                    break
        if kind:
            findings.append(
                {
                    "line": getattr(node, "lineno", None),
                    "kind": kind,
                    "evidence": evidence,
                }
            )
    return sorted(findings, key=lambda item: (item["line"] or 0, item["kind"]))


def _external_execution_findings(tree: ast.AST) -> list[dict[str, Any]]:
    module_aliases: dict[str, str] = {}
    function_aliases: dict[str, str] = {}
    external_ase_modules = {
        "abinit", "aims", "castep", "cp2k", "dftb", "espresso", "exciting",
        "gaussian", "lammpsrun", "nwchem", "octopus", "orca", "siesta", "vasp",
    }
    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
                if alias.name.startswith("ase.calculators."):
                    calculator = alias.name.split(".")[2]
                    if calculator in external_ase_modules:
                        findings.append(
                            {
                                "line": node.lineno,
                                "kind": "ase_external_calculator_import",
                                "evidence": alias.name,
                            }
                        )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "subprocess":
                for alias in node.names:
                    function_aliases[alias.asname or alias.name] = f"subprocess.{alias.name}"
            if node.module.startswith("ase.calculators."):
                calculator = node.module.split(".")[2]
                if calculator in external_ase_modules:
                    findings.append(
                        {
                            "line": node.lineno,
                            "kind": "ase_external_calculator_import",
                            "evidence": node.module,
                        }
                    )
    subprocess_calls = {"run", "Popen", "call", "check_call", "check_output"}
    os_calls = {"system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        evidence = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module = module_aliases.get(node.func.value.id, node.func.value.id)
            if module == "subprocess" and node.func.attr in subprocess_calls:
                evidence = f"subprocess.{node.func.attr}"
            elif module == "os" and node.func.attr in os_calls:
                evidence = f"os.{node.func.attr}"
        elif isinstance(node.func, ast.Name) and node.func.id in function_aliases:
            evidence = function_aliases[node.func.id]
        if evidence:
            findings.append(
                {"line": node.lineno, "kind": "direct_process_launch", "evidence": evidence}
            )
    return sorted(findings, key=lambda item: (item["line"], item["kind"], item["evidence"]))


def _probe_runtime_modules(
    runtime: str,
    modules: list[str],
    *,
    local_modules: set[str],
    required_symbols: dict[str, list[str]],
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    if not modules:
        return {}, {}
    python = runtime_python(runtime)
    probe_modules = [
        name for name in modules if name.split(".", 1)[0] not in local_modules
    ]
    details = {
        name: {
            "available": True,
            "importable": True,
            "source": "staged_or_framework_module",
            "version": None,
            "missing_symbols": [],
        }
        for name in modules
        if name not in probe_modules
    }
    if not probe_modules:
        return {name: True for name in modules}, details
    program = r'''
import contextlib
import importlib
import importlib.metadata
import io
import json
import sys

requirements = json.loads(sys.argv[1])
package_map = importlib.metadata.packages_distributions()
results = {}
for name, symbols in requirements.items():
    try:
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            module = importlib.import_module(name)
        distributions = package_map.get(name.split('.', 1)[0], [])
        versions = {}
        for distribution in distributions:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        module_version = getattr(module, '__version__', None)
        missing_symbols = []
        for symbol in symbols:
            value = module
            try:
                for part in symbol.split('.'):
                    value = getattr(value, part)
            except AttributeError:
                missing_symbols.append(symbol)
        results[name] = {
            'available': True,
            'importable': True,
            'source': 'runtime_import',
            'version': str(module_version) if module_version is not None else None,
            'distribution_versions': versions,
            'missing_symbols': missing_symbols,
            'captured_stdout': captured_stdout.getvalue()[-1000:],
            'captured_stderr': captured_stderr.getvalue()[-1000:],
        }
    except BaseException as exc:
        results[name] = {
            'available': False,
            'importable': False,
            'source': 'runtime_import',
            'version': None,
            'distribution_versions': {},
            'missing_symbols': list(symbols),
            'error': f'{type(exc).__name__}: {exc}',
        }
print('__MINICHEM_MODULE_PROBE__' + json.dumps(results, sort_keys=True))
'''
    environment = runtime_environment(runtime)
    completed = subprocess.run(
        [
            str(python),
            "-c",
            program,
            json.dumps(
                {name: required_symbols.get(name, []) for name in probe_modules},
                sort_keys=True,
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Runtime module probe failed with exit code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    marker = "__MINICHEM_MODULE_PROBE__"
    line = next(
        (item for item in reversed(completed.stdout.splitlines()) if item.startswith(marker)),
        None,
    )
    if line is None:
        raise RuntimeError("Runtime module probe returned no structured result")
    details.update(json.loads(line[len(marker) :]))
    result = {
        name: bool(item.get("importable")) and not item.get("missing_symbols")
        for name, item in details.items()
    }
    return result, details


def _isolated_workspace_path_findings(tree: ast.AST) -> list[dict[str, Any]]:
    """Find literal workspace-relative inputs that are absent from an isolated job."""

    roots = {"data", "_tool_artifacts"}
    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            function_name = ""
        if not (
            function_name in {"Path", "open"}
            or function_name.startswith("read_")
            or function_name.startswith("load")
            or function_name in {"from_file", "read_text", "read_bytes"}
        ):
            continue
        candidates = [*node.args, *(keyword.value for keyword in node.keywords)]
        for candidate in candidates:
            if not isinstance(candidate, ast.Constant) or not isinstance(candidate.value, str):
                continue
            normalized = candidate.value.strip().replace("\\", "/")
            if not normalized:
                continue
            path = PurePosixPath(normalized)
            if path.is_absolute() or not path.parts or path.parts[0] not in roots:
                continue
            findings.append(
                {
                    "line": getattr(candidate, "lineno", getattr(node, "lineno", None)),
                    "path": normalized,
                    "workspace_root": path.parts[0],
                    "function": function_name,
                }
            )
    return sorted(findings, key=lambda item: (item["line"] or 0, item["path"]))


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _json_shape(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_fields: int,
    include_scalar_samples: bool,
) -> dict[str, Any]:
    value_type = _value_type(value)
    result: dict[str, Any] = {"type": value_type}
    if value_type == "object":
        keys = list(value)
        result["field_count"] = len(keys)
        result["fields"] = keys[:max_fields]
        result["truncated_fields"] = len(keys) > max_fields
        if depth < max_depth:
            result["field_shapes"] = {
                str(key): _json_shape(
                    value[key],
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_fields=max_fields,
                    include_scalar_samples=include_scalar_samples,
                )
                for key in keys[:max_fields]
            }
    elif value_type == "array":
        result["length"] = len(value)
        result["item_types"] = sorted({_value_type(item) for item in value})
        result["null_count"] = sum(item is None for item in value)
        sampled_items = value[: min(len(value), 50)]
        if sampled_items and all(isinstance(item, dict) for item in sampled_items):
            sampled_fields = list(
                dict.fromkeys(
                    str(key) for item in sampled_items for key in item.keys()
                )
            )[:max_fields]
            result["object_field_profiles"] = {
                field: {
                    "presence_count": sum(field in item for item in sampled_items),
                    "null_count": sum(item.get(field) is None for item in sampled_items),
                    "types": sorted(
                        {
                            _value_type(item[field])
                            for item in sampled_items
                            if field in item
                        }
                    ),
                }
                for field in sampled_fields
            }
            result["profiled_item_count"] = len(sampled_items)
        if value and depth < max_depth:
            result["first_item_shape"] = _json_shape(
                value[0],
                depth=depth + 1,
                max_depth=max_depth,
                max_fields=max_fields,
                include_scalar_samples=include_scalar_samples,
            )
    elif include_scalar_samples and value is not None:
        result["sample"] = value if not isinstance(value, str) else value[:200]
    return result


def _infer_text_column(values: list[str]) -> dict[str, Any]:
    nonempty = [value.strip() for value in values if value.strip()]
    if not nonempty:
        inferred = "empty"
    else:
        try:
            parsed = [float(value) for value in nonempty]
        except ValueError:
            lowered = {value.casefold() for value in nonempty}
            inferred = "boolean" if lowered <= {"true", "false"} else "string"
        else:
            inferred = (
                "integer"
                if all(number.is_integer() for number in parsed)
                else "number"
            )
    return {
        "inferred_type": inferred,
        "observed_rows": len(values),
        "nonempty_count": len(nonempty),
        "empty_count": len(values) - len(nonempty),
    }


def _inspect_input_path(
    path: Path,
    *,
    max_depth: int,
    max_fields: int,
    max_rows: int,
    include_scalar_samples: bool,
) -> dict[str, Any]:
    base = {
        "path": relative_workspace_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "suffix": path.suffix.casefold(),
    }
    suffix = path.suffix.casefold()
    if suffix == ".json":
        if path.stat().st_size > MAX_INSPECTION_JSON_BYTES:
            return {
                **base,
                "format": "json",
                "valid": None,
                "inspection_status": "skipped_size_limit",
                "maximum_parse_bytes": MAX_INSPECTION_JSON_BYTES,
            }
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {**base, "format": "json", "valid": False, "error": str(exc)}
        return {
            **base,
            "format": "json",
            "valid": True,
            "shape": _json_shape(
                value,
                depth=0,
                max_depth=max_depth,
                max_fields=max_fields,
                include_scalar_samples=include_scalar_samples,
            ),
        }
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                fieldnames = list(reader.fieldnames or [])
                rows = [row for _, row in zip(range(max_rows), reader)]
        except (UnicodeDecodeError, csv.Error) as exc:
            return {**base, "format": "table", "valid": False, "error": str(exc)}
        return {
            **base,
            "format": "table",
            "valid": bool(fieldnames),
            "columns": fieldnames[:max_fields],
            "truncated_columns": len(fieldnames) > max_fields,
            "observed_rows": len(rows),
            "row_limit": max_rows,
            "column_profiles": {
                name: _infer_text_column([str(row.get(name) or "") for row in rows])
                for name in fieldnames[:max_fields]
            },
        }
    return {**base, "format": "opaque", "valid": True}


def inspect_analysis_inputs(request: AnalysisInputInspectionRequest) -> dict[str, Any]:
    summaries = []
    for item in request.inputs:
        path = resolve_workspace_public_path(
            item.source_path,
            must_exist=True,
            purpose="Analysis input",
            allow_execution_job_outputs=True,
            allow_action_artifacts=True,
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError("Analysis input must be a regular file")
        summaries.append(
            {
                "name": item.name,
                "semantic_type": item.semantic_type,
                **_inspect_input_path(
                    path,
                    max_depth=request.max_depth,
                    max_fields=request.max_fields,
                    max_rows=request.max_rows,
                    include_scalar_samples=request.include_scalar_samples,
                ),
            }
        )
    return {
        "status": "success",
        "inputs": summaries,
        "inspection_boundary": (
            "This is bounded structural inspection, not validation of scientific meaning. "
            "Programs must still handle missing, null, and heterogeneous values explicitly."
        ),
    }


def validate_analysis_program(request: AnalysisJobRequest) -> dict[str, Any]:
    if request.runtime not in set(runtime_names()):
        raise KeyError(f"Unknown analysis runtime {request.runtime!r}")
    python = runtime_python(request.runtime)
    if not python.is_file():
        return {
            "status": "unavailable",
            "valid": False,
            "runtime": request.runtime,
            "error": {
                "stage": "preflight",
                "code": "runtime_python_missing",
                "message": f"Runtime Python does not exist: {python}",
                "retryable": False,
            },
        }
    script = resolve_workspace_public_path(
        request.script_path,
        must_exist=True,
        purpose="Analysis program",
        allow_execution_job_outputs=True,
        allow_action_artifacts=True,
    )
    if script.is_symlink() or not script.is_file():
        raise ValueError("Analysis program must be a regular non-symlink file")
    if script.suffix.lower() != ".py":
        raise ValueError("Analysis program source must be a .py file")
    try:
        source = script.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return _analysis_failure(
            stage="preflight",
            code="python_source_encoding",
            message="Analysis program must be UTF-8 text",
            file=request.script_path,
            evidence=str(exc),
            candidate_fixes=["Save the program as UTF-8 without binary content."],
        )
    try:
        tree = ast.parse(source, filename=request.script_path)
        compile(tree, request.script_path, "exec")
    except SyntaxError as exc:
        return _analysis_failure(
            stage="preflight",
            code="python_syntax_error",
            message=str(exc.msg),
            file=request.script_path,
            line=exc.lineno,
            evidence=exc.text.strip() if exc.text else None,
            candidate_fixes=["Correct the reported Python syntax before submission."],
        )
    for item in request.inputs:
        path = resolve_workspace_public_path(
            item.source_path,
            must_exist=True,
            purpose="Analysis input",
            allow_execution_job_outputs=True,
            allow_action_artifacts=True,
        )
        if path.is_symlink() or not path.is_file():
            return _analysis_failure(
                stage="input",
                code="analysis_input_invalid",
                message=f"Declared input {item.name!r} is not a regular file",
                file=item.source_path,
                candidate_fixes=["Create the input in the workspace and declare its exact source_path."],
            )
    for item in request.staged_inputs:
        resolve_workspace_public_path(
            item.source_path,
            must_exist=True,
            purpose="Staged input",
            allow_execution_job_outputs=True,
            allow_action_artifacts=True,
        )
    imports = _analysis_imports(tree)
    external_findings = _external_execution_findings(tree)
    if external_findings:
        return _analysis_failure(
            stage="preflight",
            code="external_execution_not_audited",
            message=(
                "Programmable jobs cannot directly launch external executables; use the native "
                "job runner so each software call has its own resources, timeout, status, and provenance"
            ),
            file=request.script_path,
            line=external_findings[0]["line"],
            evidence=json.dumps(external_findings, sort_keys=True),
            candidate_fixes=[
                "Remove direct subprocess/os process launches from the analysis program.",
                "Submit the external executable with validate_native_job and submit_native_job.",
                "Set parent_job_id on the native request to link the orchestration provenance.",
            ],
        )
    runtime_path_findings = _runtime_path_injection_findings(tree)
    if runtime_path_findings:
        return _analysis_failure(
            stage="preflight",
            code="cross_runtime_path_injection",
            message=(
                "The program mutates Python import paths and may mix binary packages from a "
                "different runtime"
            ),
            file=request.script_path,
            line=runtime_path_findings[0]["line"],
            evidence=json.dumps(runtime_path_findings, sort_keys=True),
            candidate_fixes=[
                "Remove sys.path, site.addsitedir, and PYTHONPATH mutations.",
                "Select one runtime that already provides every required module.",
                "Declare required_modules, versions, and symbols so preflight verifies that runtime.",
            ],
        )
    isolated_path_findings = _isolated_workspace_path_findings(tree)
    if isolated_path_findings:
        return _analysis_failure(
            stage="input",
            code="unstaged_workspace_relative_path",
            message=(
                "The program contains workspace-relative input paths that will not exist in "
                "the isolated analysis job directory"
            ),
            file=request.script_path,
            line=isolated_path_findings[0]["line"],
            evidence=json.dumps(isolated_path_findings, sort_keys=True),
            candidate_fixes=[
                "Declare each required file in inputs and read it with JobContext.input(name).",
                "Alternatively map each file through staged_inputs and use its target_path inside the job.",
                "Do not assume the program starts in the benchmark workspace root.",
            ],
        )
    reliability_findings = _analysis_reliability_findings(tree, request=request)
    if reliability_findings:
        return _analysis_failure(
            stage="preflight",
            code="analysis_program_runtime_error",
            message=reliability_findings[0]["message"],
            file=request.script_path,
            line=reliability_findings[0]["line"],
            evidence=json.dumps(reliability_findings, sort_keys=True),
            candidate_fixes=[
                "Use only the public JobContext members shown by the managed program contract.",
                "Use input_path = ctx.input(name) directly; it is already a pathlib.Path.",
                "Evaluate conditional values before an f-string, then apply a plain format such as :.4f.",
                "Run validate_analysis_program again before submitting the corrected program.",
            ],
        )
    local_modules: set[str] = {"researchchem_job"}
    local_module_details: dict[str, dict[str, Any]] = {
        "researchchem_job": {
            "available": True,
            "importable": True,
            "source": "framework_module",
            "version": None,
            "missing_symbols": [],
        }
    }
    for item in request.staged_inputs:
        target = PurePosixPath(item.target_path)
        if target.suffix.casefold() != ".py":
            continue
        local_name = target.stem
        local_source = resolve_workspace_public_path(
            item.source_path,
            must_exist=True,
            purpose="Staged Python module",
            allow_execution_job_outputs=True,
            allow_action_artifacts=True,
        )
        try:
            local_text = local_source.read_text(encoding="utf-8")
            local_tree = ast.parse(local_text, filename=item.source_path)
            compile(local_tree, item.source_path, "exec")
        except (UnicodeDecodeError, SyntaxError) as exc:
            return _analysis_failure(
                stage="preflight",
                code="staged_python_syntax_error",
                message=f"Staged Python module {local_name!r} is not valid Python",
                file=item.source_path,
                line=getattr(exc, "lineno", None),
                evidence=str(exc),
                candidate_fixes=["Correct or remove the staged Python module before submission."],
            )
        declared_names = {
            node.name
            for node in local_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for node in local_tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                declared_names.update(
                    target_node.id
                    for target_node in targets
                    if isinstance(target_node, ast.Name)
                )
        missing_symbols = [
            symbol
            for symbol in request.required_symbols.get(local_name, [])
            if "." in symbol or symbol not in declared_names
        ]
        local_modules.add(local_name)
        local_module_details[local_name] = {
            "available": True,
            "importable": True,
            "source": "staged_python_syntax_and_symbols",
            "version": None,
            "missing_symbols": missing_symbols,
            "source_sha256": _sha256(local_source),
        }
    modules = sorted(
        set(imports)
        | set(request.required_modules)
        | set(request.required_module_versions)
        | set(request.required_symbols)
    )
    try:
        module_status, module_details = _probe_runtime_modules(
            request.runtime,
            modules,
            local_modules=local_modules,
            required_symbols=request.required_symbols,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return _analysis_failure(
            stage="import",
            code="runtime_module_probe_failed",
            message="Could not inspect modules in the selected runtime",
            file=str(python),
            evidence=str(exc),
            candidate_fixes=["Inspect the runtime and retry with an available configured environment."],
        )
    module_details.update(local_module_details)
    module_status.update(
        {
            name: not details.get("missing_symbols")
            for name, details in local_module_details.items()
            if name in modules or name == "researchchem_job"
        }
    )
    missing = sorted(
        name
        for name, details in module_details.items()
        if name in modules and not details.get("importable")
    )
    if missing:
        return _analysis_failure(
            stage="import",
            code="runtime_modules_missing",
            message=f"Selected runtime lacks imported or required modules: {missing}",
            file=request.script_path,
            evidence=json.dumps(module_status, sort_keys=True),
            candidate_fixes=[
                "Select a runtime returned by list_analysis_runtimes that provides every module.",
                "Remove unused imports or declare and stage a local Python module explicitly.",
            ],
        )
    missing_symbols = {
        name: list(details.get("missing_symbols") or [])
        for name, details in module_details.items()
        if details.get("missing_symbols")
    }
    if missing_symbols:
        return _analysis_failure(
            stage="import",
            code="runtime_symbols_missing",
            message=f"Selected runtime modules lack required symbols: {missing_symbols}",
            file=request.script_path,
            evidence=json.dumps(module_details, sort_keys=True),
            candidate_fixes=[
                "Use symbols provided by the selected runtime version or select another runtime."
            ],
        )
    version_mismatches: dict[str, dict[str, Any]] = {}
    for name, specifier_text in request.required_module_versions.items():
        details = module_details[name]
        versions = list((details.get("distribution_versions") or {}).values())
        if details.get("version"):
            versions.append(str(details["version"]))
        unique_versions = list(dict.fromkeys(versions))
        specifier = SpecifierSet(specifier_text)
        if not unique_versions or not any(version in specifier for version in unique_versions):
            version_mismatches[name] = {
                "required": specifier_text,
                "detected": unique_versions,
            }
    if version_mismatches:
        return _analysis_failure(
            stage="import",
            code="runtime_module_version_mismatch",
            message=f"Selected runtime does not satisfy module versions: {version_mismatches}",
            file=request.script_path,
            evidence=json.dumps(module_details, sort_keys=True),
            candidate_fixes=[
                "Select a runtime with a compatible module version or update the declared constraint."
            ],
        )
    _validate_argument_paths(request.arguments)
    job_context_compliance = _job_context_compliance(tree, request)
    if (
        job_context_compliance["unknown_input_helpers"]
        or job_context_compliance["unknown_output_helpers"]
    ):
        return _analysis_failure(
            stage="input",
            code="job_context_declaration_mismatch",
            message=(
                "The program uses JobContext names that are absent from the declared input/output "
                "contract"
            ),
            file=request.script_path,
            evidence=json.dumps(job_context_compliance, sort_keys=True),
            candidate_fixes=[
                "Make every JobContext.input(name) match one inputs[].name exactly.",
                "Make every JobContext output helper name match one outputs[].name exactly.",
                "Prefer copying names from the request contract instead of inventing aliases.",
            ],
        )
    input_inspection = (
        inspect_analysis_inputs(
            AnalysisInputInspectionRequest(
                inputs=request.inputs,
                max_depth=2,
                max_fields=20,
                max_rows=100,
            )
        )
        if request.inputs
        else {"status": "success", "inputs": []}
    )
    try:
        resources = _compute_resource_limits(_effective_analysis_resource_limits(request))
    except ResourceBudgetExceeded as exc:
        return _resource_budget_error(exc)
    contract = {
        "schema_version": 1,
        "layout": {
            "code": "code/",
            "inputs": "inputs/",
            "outputs": "outputs/",
            "report": "report/",
            "logs": "logs/",
        },
        "inputs": [item.model_dump(mode="json") for item in request.inputs],
        "outputs": [item.model_dump(mode="json") for item in request.outputs],
        "execution_policy": request.execution_policy,
        "external_execution": request.external_execution,
        "job_context_boundary": (
            "JobContext validates declared names and paths but cannot prevent direct open(), "
            "absolute paths, subprocesses, or other library access."
        ),
        "job_context_compliance": job_context_compliance,
    }
    return {
        "status": "success",
        "valid": True,
        "runtime": request.runtime,
        "runtime_python": str(python),
        "script_source": relative_workspace_path(script),
        "script_target": request.script_target,
        "discovered_imports": imports,
        "required_modules": request.required_modules,
        "required_module_versions": request.required_module_versions,
        "required_symbols": request.required_symbols,
        "module_status": module_status,
        "module_details": module_details,
        "input_inspection": input_inspection,
        "external_execution_findings": external_findings,
        "runtime_path_injection_findings": runtime_path_findings,
        "job_context_compliance": job_context_compliance,
        "script_sha256": _sha256(script),
        "analysis_contract": contract,
        "resource_limits": resources,
        "execution_timeout_policy": timeout_policy_record("compute"),
        "evaluation_resource_budget": resource_budget_record(),
        "resource_availability": _resource_availability(),
    }


def submit_analysis_program(request: AnalysisJobRequest) -> dict[str, Any]:
    missing_identity = _require_scoped_submission(request)
    if missing_identity is not None:
        return missing_identity
    if current_scope() is not None:
        return _submit_scoped_execution_job(
            request,
            job_type="programmable_analysis",
            validator=validate_analysis_program,
            preparer=_prepare_scoped_analysis_job,
        )
    submission_key = _request_submission_key(request)
    operation_id = request.operation_id
    if submission_key is not None:
        operation_id = scoped_submission_operation_id(submission_key)
    validation = validate_analysis_program(request)
    if validation["status"] != "success":
        return validation
    python = Path(validation["runtime_python"])
    script = resolve_workspace_public_path(
        request.script_path,
        must_exist=True,
        purpose="Analysis program",
        allow_execution_job_outputs=True,
        allow_action_artifacts=True,
    )
    staged = [
        StagedInput(
            source_path=request.script_path,
            target_path=request.script_target,
        ),
        *request.staged_inputs,
        *[
            StagedInput(source_path=item.source_path, target_path=str(item.target_path))
            for item in request.inputs
        ],
    ]
    snapshots = _input_snapshots(staged)
    fingerprint = _operation_fingerprint(
        job_type="programmable_analysis",
        request=request,
        validation=validation,
        input_snapshots=snapshots,
    )
    submitted = _start_job(
        job_type="programmable_analysis",
        runtime=request.runtime,
        command=[str(python), request.script_target, *request.arguments],
        stdin_target=None,
        staged_inputs=staged,
        resource_limits=dict(validation["resource_limits"]),
        metadata={
            "runtime": request.runtime,
            "script_source": relative_workspace_path(script),
            "script_target": request.script_target,
            "label": request.label,
            "execution_timeout_policy": timeout_policy_record("compute"),
            "analysis_contract": validation["analysis_contract"],
            "parent_job_id": request.parent_job_id,
            "script_sha256": validation["script_sha256"],
            "preflight": {
                "discovered_imports": validation["discovered_imports"],
                "required_modules": validation["required_modules"],
                "required_module_versions": validation["required_module_versions"],
                "required_symbols": validation["required_symbols"],
                "module_status": validation["module_status"],
                "module_details": validation["module_details"],
                "input_inspection": validation["input_inspection"],
                "external_execution_findings": validation[
                    "external_execution_findings"
                ],
                "job_context_compliance": validation["job_context_compliance"],
            },
            "security_boundary": (
                "Subprocess/resource/workspace convention only; deploy MCP inside an OS container "
                "or scheduler sandbox when executing untrusted programs."
            ),
        },
        operation_id=operation_id,
        operation_fingerprint=fingerprint if operation_id else None,
        submission_key=submission_key,
        caller_request_fingerprint=(
            request.caller_request_fingerprint
            if operation_id and submission_key is None
            else None
        ),
        expected_input_snapshots=snapshots,
        launch_file_snapshots=[
            _file_snapshot(python, role="analysis_runtime_python")
        ],
    )
    submitted["job_context_compliance"] = validation["job_context_compliance"]
    return submitted


def _read_status(job_id: str) -> tuple[Path, dict[str, Any]]:
    if not is_execution_job_id(job_id):
        raise _ExecutionJobUnavailable()
    try:
        directory = _job_directory(job_id)
    except (_ExecutionJobUnavailable, KeyError, ValueError, OSError) as exc:
        raise _ExecutionJobUnavailable() from exc
    path = directory / "status.json"
    if path.is_symlink() or not path.is_file():
        raise _ExecutionJobUnavailable()
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ExecutionJobUnavailable() from exc
    if not isinstance(status, dict):
        raise _ExecutionJobUnavailable()
    if status.get("job_id") not in {None, job_id}:
        raise _ExecutionJobUnavailable()
    _validate_job_scope(directory, status)
    metadata = status.get("metadata")
    if (
        status.get("status") in {"queued", "running"}
        and status.get("supervisor_exit_pending") is True
        and isinstance(metadata, Mapping)
        and str(metadata.get("software_id") or "").casefold() == "gaussian"
    ):
        # supervisor 退出后，Gaussian 的同组子进程可能再运行数秒。每次查询
        # 都核对持久进程身份；只有 supervisor、child 和进程组均明确停止时，
        # 才把遗留记录写成终态并释放 FIFO。身份不确定时继续保留原状态。
        try:
            reconcile_stopped_native_gaussian_jobs()
            refreshed = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(refreshed, dict):
                raise _ExecutionJobUnavailable()
            if refreshed.get("job_id") not in {None, job_id}:
                raise _ExecutionJobUnavailable()
            _validate_job_scope(directory, refreshed)
            status = refreshed
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ResourceStateUnavailable,
        ):
            pass
    return directory, status


def _trusted_execution_job_route(
    job_id: str,
    status: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    """按 registry 或旧 status 记录确定内部 job_type，失败时不猜测路由。"""

    try:
        record = _execution_job_registry().get(job_id)
    except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
        # 无认证 scope 的旧 stdio 作业可能只有 legacy status；不能因为测试夹具或
        # 旧 workspace 根目录权限不满足 registry 写入条件而丢失只读恢复兼容。
        if current_scope() is None:
            return status.get("job_type"), None, None
        return None, None, _execution_job_unavailable_response()
    status_type = status.get("job_type")
    if record is not None:
        registry_type = record.get("job_type")
        if not isinstance(registry_type, str) or (
            status_type is not None and status_type != registry_type
        ):
            return None, None, _execution_job_unavailable_response()
        for field in (
            "job_id",
            "operation_id",
            "operation_fingerprint",
            "submission_key",
            "scope_id",
            "scope_workspace",
        ):
            if status.get(field) is not None and status.get(field) != record.get(field):
                return None, None, _execution_job_unavailable_response()
        return registry_type, record, None
    if status_type is not None and not isinstance(status_type, str):
        return None, None, _execution_job_unavailable_response()
    return status_type, None, None


def _transition_state_path_facade_collect(job_id: str) -> dict[str, Any]:
    """把统一 collect 请求转交 TS 专用实现，复用 canonical ActionResult 收集链。"""

    from .transition_state_jobs import (
        TransitionStatePathJobCollectRequest,
        collect_transition_state_path_job,
    )

    response = collect_transition_state_path_job(
        TransitionStatePathJobCollectRequest(job_id=job_id)
    )
    return _preserve_requested_job_id_in_collection_error(response, job_id)


def _preserve_requested_job_id_in_collection_error(
    response: Mapping[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """为没有冲突身份的 generic collect 错误补齐顶层 job_id。

    输入是下层 collect 返回的结构化 envelope 和本次请求的 job_id；实际动作是复制
    envelope，并且只在它带有 error、顶层没有 job_id、且嵌套 error.job_id 缺失或与请求
    完全相同时补上请求身份。该动作不改写错误内容，也不把不同顶层或嵌套身份提升为可信
    身份；返回新的普通字典，调用方继续保留原错误状态，避免错误回执释放作业门禁。
    """

    normalized = dict(response)
    explicit_job_ids = {
        value.strip()
        for value in (normalized.get("job_id"), normalized.get("jobId"))
        if isinstance(value, str) and value.strip()
    }
    nested_job = normalized.get("job")
    if isinstance(nested_job, Mapping):
        explicit_job_ids.update(
            value.strip()
            for value in (nested_job.get("job_id"), nested_job.get("jobId"))
            if isinstance(value, str) and value.strip()
        )
    if explicit_job_ids:
        return normalized
    if "error" not in normalized or normalized["error"] is None:
        return normalized
    error = normalized["error"]
    if isinstance(error, Mapping):
        error_job_id = error.get("job_id")
        if error_job_id is not None and error_job_id != job_id:
            return normalized
    normalized["job_id"] = job_id
    return normalized


def _action_job_facade_collect(
    job_id: str,
    *,
    submission_key: str | None = None,
) -> dict[str, Any]:
    """把统一 collect 请求转交通用 Action Job Core 的 canonical 收集链。"""

    from .action_jobs import collect_action_job

    response = collect_action_job(job_id, submission_key=submission_key)
    return _preserve_requested_job_id_in_collection_error(response, job_id)


def _execution_job_status_envelope(
    job_id: str,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    """构造所有 execution job 共用的最小轮询响应，不暴露作业类型或内部日志。"""

    job_status = status.get("status")
    response = {
        "status": "success",
        "job_id": job_id,
        "job_status": job_status,
        "terminal": job_status in TERMINAL_JOB_STATES,
        "job": {"job_id": job_id, "status": job_status},
    }
    # status.json 中的结构化错误是模型修正参数或提交方式所需的原始事实；只复制
    # error 字段，不把宿主路径、日志或其他内部控制字段带入轮询响应。
    if isinstance(status.get("error"), Mapping):
        response["error"] = deepcopy(dict(status["error"]))
    for field in ("warning", "provenance"):
        if isinstance(status.get(field), Mapping):
            response[field] = deepcopy(dict(status[field]))
    return response


def _reference_unavailable(kind: str) -> dict[str, str]:
    """返回不说明未知引用是否存在的结果。"""

    return {"kind": kind, "status": "unavailable"}


def _reference_invalid(kind: str, reason_code: str) -> dict[str, str]:
    """返回不含路径、日志和宿主细节的引用完整性错误。"""

    return {"kind": kind, "status": "invalid", "reason_code": reason_code}


def _read_private_json(path: Path) -> dict[str, Any] | None:
    """只读解析一个控制文件；不存在返回 None，损坏记录由调用方区分。"""

    if path.is_symlink() or not path.is_file():
        return None
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _OperationIdentityConflict("private execution record is not regular")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise _OperationIdentityConflict("private execution record is corrupt")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise _OperationIdentityConflict("private execution record is not an object")
    return value


def _scope_persisted_identity_is_complete(
    directory: Path,
    status: Mapping[str, Any],
) -> bool:
    """确认 job 目录中的每份现有身份记录都明确属于当前 scope。"""

    scoped = current_scope()
    if scoped is None:
        return False

    records = _read_scope_identity_records(directory, status)
    if records is None or not _scope_records_match(scoped, records):
        return False
    try:
        registry_record = _execution_job_registry().get(directory.name)
    except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
        return False
    if registry_record is not None:
        records.append(registry_record)
    return _scope_submission_identity_is_complete(records)


def _read_scope_identity_records(
    directory: Path,
    status: Mapping[str, Any],
) -> list[Mapping[str, Any]] | None:
    """读取 scope inventory 的身份记录；任一记录不可信即返回 None。"""

    records: list[Mapping[str, Any]] = [status]
    for name in ("request.json", "operation_receipt.json", "supervisor_spec.json"):
        path = directory / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            return None
        if not path.exists():
            continue
        try:
            value = _read_private_json(path)
        except _OperationIdentityConflict:
            return None
        if value is None:
            return None
        records.append(value)
    return records


def _scope_submission_identity_is_complete(records: list[Mapping[str, Any]]) -> bool:
    """核对 scope operation 的 submission_key 交集，避免伪造恢复身份。"""

    raw_operation_ids = [
        value.get("operation_id")
        for value in records
        if value.get("operation_id") is not None
    ]
    if any(not isinstance(item, str) or not item for item in raw_operation_ids):
        return False
    operation_ids = set(raw_operation_ids)
    if len(operation_ids) > 1:
        return False
    operation_id = next(iter(operation_ids), None)
    raw_submission_keys = [value.get("submission_key") for value in records]
    if operation_id is not None and operation_id.startswith("scope-submission-"):
        if any(not isinstance(item, str) for item in raw_submission_keys):
            return False
        try:
            normalized = [validate_submission_key(item) for item in raw_submission_keys]
        except ValueError:
            return False
        if len(set(normalized)) != 1:
            return False
        try:
            return scoped_submission_operation_id(normalized[0]) == operation_id
        except (StopIteration, TypeError, ValueError, UnicodeError):
            return False
    # 普通 native/legacy job 没有 submission operation；一旦任一持久记录声称有
    # submission_key，就不能把普通 operation_id 当作提交身份，也不能把它输出。
    return all(item is None for item in raw_submission_keys)


def _scope_job_inventory() -> tuple[list[dict[str, str]], bool | None, bool]:
    """枚举当前 scope 的 job 身份，遇到不完整证据时 fail-closed。"""

    if current_scope() is None:
        return [], None, False
    try:
        directory = workspace_root() / JOB_ROOT
    except (OSError, RuntimeError, ValueError):
        return [], None, False
    if not directory.exists():
        return [], False, True
    if directory.is_symlink() or not directory.is_dir():
        return [], None, False
    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError:
        return [], None, False
    inventory: list[dict[str, str]] = []
    has_active = False
    for child in children:
        item, active, valid = _inspect_scope_inventory_child(child)
        if not valid:
            return inventory, None, False
        if item is None:
            continue
        inventory.append(item)
        has_active = has_active or active
    return inventory, has_active, True


def _inspect_scope_inventory_child(
    child: Path,
) -> tuple[dict[str, str] | None, bool, bool]:
    """核对一个 execution job 子项；锁文件只验证形状后跳过。"""

    if re.fullmatch(r"\.[0-9a-f]{64}\.lock", child.name):
        return (None, False, not child.is_symlink() and child.is_file())
    if (
        child.is_symlink()
        or not child.is_dir()
        or not is_execution_job_id(child.name)
    ):
        return None, False, False
    try:
        job_directory, status = _read_status(child.name)
    except _ExecutionJobUnavailable:
        return None, False, False
    if not _scope_persisted_identity_is_complete(job_directory, status):
        return None, False, False
    state = status.get("status")
    if state not in TERMINAL_JOB_STATES and state not in {"queued", "running"}:
        return None, False, False
    item = {"job_id": child.name, "status": str(state)}
    operation_id = status.get("operation_id")
    if isinstance(operation_id, str) and operation_id.startswith("scope-submission-"):
        # _scope_persisted_identity_is_complete 已核对了 status、receipt、request
        # 和 supervisor_spec 中的 submission_key 交集及其确定性映射；因此这里只
        # 输出明确命名的 job 提交身份，不再输出泛化 operation_id。
        item["submission_operation_id"] = operation_id
    return (
        item,
        state in {"queued", "running"},
        True,
    )


def _scope_operation_inventory() -> tuple[list[dict[str, str]], bool | None, bool]:
    """恢复当前 scope 的 typed Action operation 清单，不暴露收据内容。"""

    if current_scope() is None:
        return [], None, False
    try:
        store = OperationIdempotencyStore(create=False)
    except FileNotFoundError:
        # 尚未发生任何 OperationStore 副作用；不存在的根是可证明的空清单。
        return [], False, True
    except (OSError, RuntimeError, ValueError):
        return [], None, False
    try:
        return store.list_operation_inventory()
    except (OSError, RuntimeError, ValueError):
        return [], None, False


def _read_terminal_job_reference_records(
    reference: ExecutionJobReference,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | dict[str, Any]:
    """读取终态 job、status 和 operation receipt，失败时返回脱敏结果。"""

    try:
        directory, status = _read_status(reference.job_id)
    except _ExecutionJobUnavailable:
        return _reference_unavailable("job")
    if status.get("status") not in TERMINAL_JOB_STATES:
        return _reference_invalid("job", "job_not_terminal")
    try:
        receipt = _read_private_json(directory / "operation_receipt.json")
    except _OperationIdentityConflict:
        return _reference_unavailable("job")
    if receipt is None:
        return _reference_unavailable("job")
    return directory, status, receipt


def _canonical_job_reference_identity(
    reference: ExecutionJobReference,
    status: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[str, str] | None:
    """从 status/receipt 交集提取可证明的 operation 身份。"""

    scoped = current_scope()
    if scoped is None or _receipt_fingerprint_aliases_conflict(receipt):
        return None
    canonical_operation_id, canonical_fingerprint = _canonical_reference_values(
        status, receipt
    )
    if not isinstance(canonical_operation_id, str) or not isinstance(canonical_fingerprint, str):
        return None
    if not _reference_metadata_matches(reference, status, receipt, scoped):
        return None
    if not _reference_operation_identity_matches(
        reference, status, receipt, canonical_operation_id, canonical_fingerprint
    ):
        return None
    return canonical_operation_id, canonical_fingerprint


def _receipt_fingerprint_aliases_conflict(receipt: Mapping[str, Any]) -> bool:
    """返回 receipt 的两个指纹别名是否互相矛盾。"""

    first = receipt.get("fingerprint")
    second = receipt.get("operation_fingerprint")
    return first is not None and second is not None and first != second


def _canonical_reference_values(
    status: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[Any, Any]:
    """从 status 与 receipt 的优先级交集提取 operation 标识和摘要。"""

    operation_id = status.get("operation_id") or receipt.get("operation_id")
    fingerprint = (
        status.get("operation_fingerprint")
        or receipt.get("operation_fingerprint")
        or receipt.get("fingerprint")
    )
    return operation_id, fingerprint


def _reference_metadata_matches(
    reference: ExecutionJobReference,
    status: Mapping[str, Any],
    receipt: Mapping[str, Any],
    scoped: Any,
) -> bool:
    """核对收据版本、scope、作业终态以及 status 的重复身份字段。

    输入是调用方引用、作业状态、operation 收据和当前 scope；函数只比较已经
    读取的字段，不修改作业或收据。所有重复身份和作业终态一致时返回 True；
    缺失、越出 scope 或两份作业终态冲突时返回 False，由调用方拒绝该引用。
    """

    if receipt.get("schema_version") != 1 or receipt.get("job_id") != reference.job_id:
        return False
    worker_status = status.get("status")
    receipt_phase = receipt.get("phase")
    if (
        receipt.get("scope_id") != scoped.scope_id
        or receipt.get("scope_workspace") != str(scoped.workspace)
        or worker_status not in TERMINAL_JOB_STATES
        or receipt_phase != worker_status
    ):
        return False
    if status.get("job_id") is not None and status.get("job_id") != reference.job_id:
        return False
    if status.get("job_type") is not None and receipt.get("job_type") != status.get("job_type"):
        return False
    if status.get("operation_id") is not None and status.get("operation_id") != receipt.get("operation_id"):
        return False
    return True


def _reference_operation_identity_matches(
    reference: ExecutionJobReference,
    status: Mapping[str, Any],
    receipt: Mapping[str, Any],
    operation_id: str,
    fingerprint: str,
) -> bool:
    """核对 canonical 指纹别名和 operation_id 到 job_id 的确定性映射。"""

    receipt_fingerprint = receipt.get("fingerprint")
    receipt_operation_fingerprint = receipt.get("operation_fingerprint")
    canonical_receipt_fingerprint = receipt_operation_fingerprint or receipt_fingerprint
    if status.get("operation_fingerprint") is not None and status.get(
        "operation_fingerprint"
    ) != canonical_receipt_fingerprint:
        return False
    try:
        registry_record = _execution_job_registry().get(reference.job_id)
    except (ExecutionJobRegistryError, ValueError):
        return False
    if registry_record is not None:
        return (
            registry_record.get("operation_id") == operation_id
            and registry_record.get("operation_fingerprint") == fingerprint
        )
    try:
        return _operation_job_id(operation_id) == reference.job_id
    except (TypeError, ValueError, UnicodeError):
        return False


def _job_submission_identity_is_consistent(
    status: Mapping[str, Any],
    receipt: Mapping[str, Any],
    operation_id: str,
    *,
    registry_record: Mapping[str, Any] | None = None,
) -> bool:
    """核对 scoped submission_key 是否在 status 和 receipt 中一致有效。"""

    has_submission_key = (
        receipt.get("submission_key") is not None
        or status.get("submission_key") is not None
    )
    if not operation_id.startswith("scope-submission-") and not has_submission_key:
        if registry_record is None:
            return True
        return registry_record.get("submission_key") is None
    # 只有 scope-submission operation 才能由 submission_key 得到确定性身份。
    # native/legacy job 若同时声称有 submission_key 却携带普通 operation_id，
    # 说明记录语义已经冲突，不能把普通 operation_id 输出成提交身份。
    if not operation_id.startswith("scope-submission-"):
        return False
    if not isinstance(receipt.get("submission_key"), str):
        return False
    try:
        normalized = validate_submission_key(receipt["submission_key"])
    except ValueError:
        return False
    if status.get("submission_key") != normalized:
        return False
    if registry_record is not None:
        registry_submission_key = registry_record.get("submission_key")
        if not isinstance(registry_submission_key, str):
            return False
        try:
            if validate_submission_key(registry_submission_key) != normalized:
                return False
        except ValueError:
            return False
    # Investigation Action job 的 operation 身份由 scope 和 submission_key
    # 唯一计算；不能只比较两个文件里的同名字符串，否则两份收据可以共同
    # 携带一个伪造的 operation_id。
    if operation_id.startswith("scope-submission-"):
        try:
            return scoped_submission_operation_id(normalized) == operation_id
        except (TypeError, ValueError, UnicodeError):
            return False
    return True


def _job_submission_identity_is_consistent_with_registry(
    job_id: str,
    status: Mapping[str, Any],
    receipt: Mapping[str, Any],
    operation_id: str,
) -> bool:
    """从统一 registry 补核对 job 的提交身份，失败时不输出提交 operation。"""

    try:
        registry_record = _execution_job_registry().get(job_id)
    except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
        return False
    return _job_submission_identity_is_consistent(
        status,
        receipt,
        operation_id,
        registry_record=registry_record,
    )


def _verify_execution_job_reference(
    reference: ExecutionJobReference,
) -> dict[str, Any]:
    """只读核对一个当前 scope 的终态 job，并从收据补全可证明的身份。"""

    records = _read_terminal_job_reference_records(reference)
    if isinstance(records, dict):
        return records
    _directory, status, receipt = records
    identity = _canonical_job_reference_identity(reference, status, receipt)
    if identity is None:
        return _reference_unavailable("job")
    canonical_operation_id, canonical_fingerprint = identity
    if not _job_submission_identity_is_consistent_with_registry(
        reference.job_id,
        status,
        receipt,
        canonical_operation_id,
    ):
        return _reference_unavailable("job")
    if (
        reference.operation_id is not None
        and reference.operation_id != canonical_operation_id
    ) or (
        reference.operation_fingerprint is not None
        and reference.operation_fingerprint != canonical_fingerprint
    ):
        return _reference_invalid("job", "job_identity_mismatch")
    result: dict[str, Any] = {
        "kind": "job",
        "status": "valid",
        "job_id": reference.job_id,
        "operation_fingerprint": canonical_fingerprint,
        "terminal_status": status.get("status"),
    }
    if canonical_operation_id.startswith("scope-submission-"):
        # 只有经过 status/receipt 的 submission_key 确定性核对的 job 才公开
        # 提交 operation。没有该语义的 native/legacy job 省略可选字段，避免
        # 把普通 operation_id 伪装成 submission_operation_id。
        result["submission_operation_id"] = canonical_operation_id
    return result


def _verify_execution_operation_reference(
    reference: ExecutionOperationReference,
) -> dict[str, Any]:
    """只读核对 typed Action 的 completed 幂等 receipt，并补全 canonical 指纹。"""

    receipt = _read_operation_reference_receipt(reference)
    if receipt is None:
        return _reference_unavailable("operation")
    validation = _validate_operation_reference_receipt(reference, receipt)
    if validation is not None:
        return validation
    canonical_fingerprint = str(receipt["fingerprint"])
    return {
        "kind": "operation",
        "status": "valid",
        "operation_id": reference.operation_id,
        "fingerprint": canonical_fingerprint,
        "operation_status": "completed",
    }


def _read_operation_reference_receipt(
    reference: ExecutionOperationReference,
) -> Mapping[str, Any] | None:
    """读取 typed Action receipt；存储异常和损坏收据均不可用。"""

    try:
        receipt = OperationIdempotencyStore(create=False).read_operation_receipt(
            reference.operation_id
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        return None
    if not isinstance(receipt, Mapping) or receipt.get("status") == "corrupt":
        return None
    return receipt


def _validate_operation_reference_receipt(
    reference: ExecutionOperationReference,
    receipt: Mapping[str, Any],
) -> dict[str, Any] | None:
    """核对 typed Action receipt 的 scope、版本、指纹和完成状态。"""

    scoped = current_scope()
    if scoped is None:
        return _reference_unavailable("operation")
    if (
        receipt.get("scope_id") != scoped.scope_id
        or receipt.get("scope_workspace") != str(scoped.workspace)
    ):
        return _reference_unavailable("operation")
    fingerprint = receipt.get("fingerprint")
    if not isinstance(fingerprint, str):
        return _reference_unavailable("operation")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("operation_id") != reference.operation_id
    ):
        return _reference_unavailable("operation")
    if reference.fingerprint is not None and reference.fingerprint != fingerprint:
        return _reference_invalid("operation", "operation_identity_mismatch")
    if receipt.get("status") != "completed":
        return _reference_invalid("operation", "operation_not_completed")
    if not isinstance(receipt.get("response"), Mapping):
        return _reference_unavailable("operation")
    return None


def _verify_execution_artifact_reference(
    reference: ExecutionArtifactReference,
) -> dict[str, Any]:
    """用当前 workspace 的 canonical manifest 和实际文件 SHA 核对 ArtifactRef。"""

    canonical = _read_artifact_reference_manifest(reference)
    if canonical is None:
        return _reference_unavailable("artifact")
    if _artifact_manifest_differs(reference, canonical):
        return _reference_invalid("artifact", "manifest_mismatch")
    integrity_error = _artifact_reference_integrity_error(canonical)
    if integrity_error is not None:
        return _reference_invalid("artifact", integrity_error)
    return {
        "kind": "artifact",
        "status": "valid",
        "artifact_id": reference.artifact_id,
        "sha256": canonical.sha256,
    }


def _action_job_record_value_candidates(
    record: Mapping[str, Any],
    field_name: str,
) -> tuple[set[str], bool]:
    """从一份 job 控制记录读取候选值，并标记字段类型错误。"""

    candidates: set[str] = set()
    malformed = False
    sources: list[Mapping[str, Any]] = [record]
    for parent_name in ("metadata", "action_job_request", "worker_action_job_request"):
        parent = record.get(parent_name)
        if isinstance(parent, Mapping):
            sources.append(parent)
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for parent_name in ("action_job_request", "worker_action_job_request"):
            parent = metadata.get(parent_name)
            if isinstance(parent, Mapping):
                sources.append(parent)
    for source in sources:
        if field_name not in source:
            continue
        value = source.get(field_name)
        # 固定数据源 Action 的公开请求可以省略 backend_id；其规范选择仍由
        # status/request metadata 中的 resolved backend_id 提供。省略不是一份
        # 可用于绑定的候选值，但也不应被误判为损坏字段。
        if value is None and field_name == "backend_id":
            continue
        if not isinstance(value, str) or not value:
            malformed = True
            continue
        candidates.add(value)
    return candidates, malformed


def _read_action_job_record_identity(
    directory: Path,
    status: Mapping[str, Any],
    registry_record: Mapping[str, Any] | None,
) -> tuple[str, str] | None:
    """从 job 的权威持久记录交叉提取唯一 action/backend 身份。

    ActionResult 的 ``action`` 和 ``backend`` 不能单独作为来源；至少需要 status
    或同一 job 的 request/supervisor 记录提供 action/backend，且这些记录的固定
    字段不能互相冲突。ExecutionJobRegistry 没有这两个业务字段时不会猜测。
    """

    records: list[Mapping[str, Any]] = [status]
    if registry_record is not None:
        records.append(registry_record)
    for name in ("request.json", "supervisor_spec.json", "operation_receipt.json"):
        path = directory / name
        if not path.exists():
            continue
        try:
            value = _read_private_json(path)
        except _OperationIdentityConflict:
            return None
        if value is None:
            return None
        records.append(value)
    action_values: set[str] = set()
    backend_values: set[str] = set()
    malformed = False
    for record in records:
        action_candidates, action_malformed = _action_job_record_value_candidates(
            record, "action_id"
        )
        backend_candidates, backend_malformed = _action_job_record_value_candidates(
            record, "backend_id"
        )
        action_values.update(action_candidates)
        backend_values.update(backend_candidates)
        malformed = malformed or action_malformed or backend_malformed
    if malformed or len(action_values) != 1 or len(backend_values) != 1:
        return None
    return next(iter(action_values)), next(iter(backend_values))


def _validate_action_result_status_against_worker_status(
    worker_status: str | None,
    action_result_status: ActionStatus,
) -> bool:
    """区分 action_execution worker 的传输终态和 canonical ActionResult 的科学状态。

    输入是 execution job 的外层终态和已通过 Pydantic 合同校验的 ActionResult 状态；
    外层 worker 成功表示结果文件已经安全写入，因此允许所有合同规定的科学状态，
    包括 ``failed``、``timeout`` 和 ``cancelled``。外层 worker 以失败、超时或取消
    结束时，必须与 canonical 状态完全相同；未知外层状态或不一致状态返回 False，
    由调用方把对应 operation 和 artifact 来源标记为不可用，不会修改持久化记录。
    """

    if worker_status == "success":
        return action_result_status in _CANONICAL_ACTION_RESULT_STATUSES
    if worker_status in {"failed", "timeout", "cancelled"}:
        return action_result_status == worker_status
    return False


def _read_artifact_index_records(
    store: ArtifactStore,
) -> tuple[dict[str, ArtifactRef], bool]:
    """读取 ArtifactStore 的最新 canonical 记录，损坏时整体拒绝绑定。"""

    path = store.index_path
    if path.is_symlink() or not path.is_file():
        return {}, True
    try:
        raw, _digest = read_file_snapshot(path)
        records: dict[str, ArtifactRef] = {}
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                return {}, False
            record = dict(value)
            record.pop("created_at", None)
            reference = ArtifactRef.model_validate(record)
            records[reference.artifact_id] = reference
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return {}, False
    return records, True


def _programmable_output_declarations(
    request_record: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """读取提交时冻结的 programmable 输出合同，并拒绝模糊或重复声明。"""

    metadata = request_record.get("metadata")
    contract = metadata.get("analysis_contract") if isinstance(metadata, Mapping) else None
    declarations = contract.get("outputs") if isinstance(contract, Mapping) else None
    if not isinstance(declarations, list):
        return {}, False
    normalized: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            return {}, False
        name = declaration.get("name")
        relative = declaration.get("path")
        semantic_type = declaration.get("semantic_type")
        media_type = declaration.get("media_type")
        parents = declaration.get("parent_artifact_ids", [])
        required = declaration.get("required", True)
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(relative, str)
            or not relative
            or not isinstance(semantic_type, str)
            or not semantic_type
            or not isinstance(media_type, str)
            or not media_type
            or not isinstance(parents, list)
            or any(not isinstance(item, str) for item in parents)
            or type(required) is not bool
            or name in normalized
            or relative in paths
        ):
            return {}, False
        pure = PurePosixPath(relative.replace("\\", "/"))
        if (
            pure.is_absolute()
            or str(pure) != relative
            or not pure.parts
            or pure.parts[0] not in {"outputs", "report", "logs"}
            or ".." in pure.parts
        ):
            return {}, False
        normalized[name] = {
            "name": name,
            "path": relative,
            "semantic_type": semantic_type,
            "media_type": media_type,
            "required": required,
            "parent_artifacts": list(parents),
        }
        paths.add(relative)
    return normalized, True


def _read_programmable_analysis_artifact_bindings(
    job_id: str,
    directory: Path,
    status: Mapping[str, Any],
    registry_record: Mapping[str, Any] | None,
    artifact_records: Mapping[str, ArtifactRef],
) -> tuple[dict[str, set[tuple[str, str]]], set[str], set[str]]:
    """从成功 programmable 作业的冻结合同恢复 artifact 的唯一来源。"""

    invalid_artifacts: set[str] = set()
    invalid_operations: set[str] = set()
    persisted_operation_id = status.get("operation_id")
    try:
        receipt = _read_private_json(directory / "operation_receipt.json")
        identity = _canonical_job_reference_identity(
            ExecutionJobReference(job_id=job_id),
            status,
            receipt or {},
        )
        if identity is None:
            if isinstance(persisted_operation_id, str):
                invalid_operations.add(persisted_operation_id)
            return {}, invalid_artifacts, invalid_operations
        operation_id, _fingerprint = identity
        if (
            not isinstance(receipt, Mapping)
            or not operation_id.startswith("scope-submission-")
            or not _job_submission_identity_is_consistent(
                status,
                receipt,
                operation_id,
                registry_record=registry_record,
            )
            or not _scope_persisted_identity_is_complete(directory, status)
        ):
            invalid_operations.add(operation_id)
            return {}, invalid_artifacts, invalid_operations
        if status.get("status") != "success":
            return {}, invalid_artifacts, invalid_operations
        request_record = _read_private_json(directory / "request.json")
        scoped = current_scope()
        if (
            not isinstance(request_record, Mapping)
            or request_record.get("job_id") != job_id
            or request_record.get("job_type") != "programmable_analysis"
            or request_record.get("operation_id") != operation_id
            or request_record.get("operation_fingerprint") != _fingerprint
            or request_record.get("submission_key") != status.get("submission_key")
            or scoped is None
            or request_record.get("scope_id") != scoped.scope_id
            or request_record.get("scope_workspace") != str(scoped.workspace)
        ):
            invalid_operations.add(operation_id)
            return {}, invalid_artifacts, invalid_operations
        declarations, declarations_valid = _programmable_output_declarations(
            request_record
        )
        manifest = _read_private_json(directory / "artifact_manifest.json")
        if not declarations_valid or not isinstance(manifest, Mapping):
            invalid_operations.add(operation_id)
            return {}, invalid_artifacts, invalid_operations
        if (
            manifest.get("schema_version") != 1
            or manifest.get("job_id") != job_id
            or manifest.get("artifact_status") != "valid"
            or not isinstance(manifest.get("artifacts"), list)
        ):
            invalid_operations.add(operation_id)
            return {}, invalid_artifacts, invalid_operations
        manifest_by_path: dict[str, Mapping[str, Any]] = {}
        manifest_names: set[str] = set()
        for item in manifest["artifacts"]:
            if not isinstance(item, Mapping):
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            name = item.get("name")
            declaration = declarations.get(name) if isinstance(name, str) else None
            if (
                declaration is None
                or name in manifest_names
                or item.get("path") != declaration["path"]
                or item.get("semantic_type") != declaration["semantic_type"]
                or item.get("media_type") != declaration["media_type"]
                or item.get("parent_artifacts") != declaration["parent_artifacts"]
                or item.get("required") != declaration["required"]
                or item.get("producer") != f"programmable_analysis:{job_id}"
                or item.get("validation_status") != "valid"
                or item.get("validation_errors") != []
            ):
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            relative = declaration["path"]
            path = directory.joinpath(*PurePosixPath(relative).parts)
            expected_workspace_path = relative_workspace_path(path)
            if item.get("workspace_path") != expected_workspace_path:
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            if path.is_symlink() or not path.is_file():
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            size_bytes = item.get("size_bytes")
            sha256 = item.get("sha256")
            if (
                type(size_bytes) is not int
                or size_bytes < 0
                or not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            ):
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            try:
                actual_size = path.stat().st_size
                _raw, actual_sha256 = read_file_snapshot(path)
            except (OSError, ValueError):
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            if actual_size != size_bytes or actual_sha256 != sha256:
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            registration = item.get("runtime_registration")
            if registration is not None and (
                not isinstance(registration, Mapping)
                or any(
                    registration.get(field) != item.get(source)
                    for field, source in (
                        ("path", "path"),
                        ("size_bytes", "size_bytes"),
                        ("sha256", "sha256"),
                        ("semantic_type", "semantic_type"),
                        ("media_type", "media_type"),
                        ("parent_artifacts", "parent_artifacts"),
                    )
                )
            ):
                invalid_operations.add(operation_id)
                return {}, invalid_artifacts, invalid_operations
            manifest_by_path[expected_workspace_path] = item
            manifest_names.add(name)
        if manifest_names != set(declarations):
            invalid_operations.add(operation_id)
            return {}, invalid_artifacts, invalid_operations
    except (
        _OperationIdentityConflict,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        UnicodeError,
    ):
        if isinstance(persisted_operation_id, str):
            invalid_operations.add(persisted_operation_id)
        return {}, invalid_artifacts, invalid_operations

    producer_aliases = {
        f"programmable_analysis:{job_id}",
        f"programmable_analysis:{operation_id}",
    }
    candidates: dict[str, set[tuple[str, str]]] = {}
    for artifact_id, canonical in artifact_records.items():
        if canonical.producer_action not in producer_aliases:
            continue
        manifest_item = manifest_by_path.get(canonical.path)
        if (
            manifest_item is None
            or canonical.producer_backend is not None
            or canonical.semantic_type != manifest_item["semantic_type"]
            or canonical.media_type != manifest_item["media_type"]
            or canonical.sha256 != manifest_item["sha256"]
            or canonical.parent_artifact_ids != manifest_item["parent_artifacts"]
        ):
            invalid_artifacts.add(artifact_id)
            continue
        try:
            if _artifact_reference_integrity_error(canonical) is not None:
                raise ValueError("programmable artifact integrity mismatch")
        except (OSError, RuntimeError, TypeError, ValueError):
            invalid_artifacts.add(artifact_id)
            continue
        candidates.setdefault(artifact_id, set()).add((job_id, operation_id))
    return candidates, invalid_artifacts, invalid_operations


def _read_action_result_artifact_bindings(
) -> tuple[dict[str, dict[str, str]], set[str], set[str]]:
    """从当前 scope 的终态 Action job 恢复唯一 artifact 来源链。

    只读取已核对 scope 的 job status、operation receipt 和 canonical
    ``action_result.json``；每个结果中的 ArtifactRef 还必须与当前
    ArtifactStore manifest 和文件摘要完全一致。返回值分别是唯一归属表、不能安全
    建立归属的 artifact ID 集合和 canonical 结果不可用的 submission operation 集合，
    任何缺失、重复或身份冲突都不会猜测。归属表明确保存 job_id 和
    ``scope-submission-*``，不把该提交身份伪装成 typed Action operation。
    """

    if current_scope() is None:
        return {}, set(), set()
    try:
        root = workspace_root() / JOB_ROOT
        if not root.exists():
            return {}, set(), set()
        if root.is_symlink() or not root.is_dir():
            return {}, set(), set()
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except (OSError, RuntimeError, ValueError):
        return {}, set(), set()

    candidates: dict[str, set[tuple[str, str]]] = {}
    invalid_artifacts: set[str] = set()
    invalid_operations: set[str] = set()
    store: ArtifactStore | None = None
    artifact_index_records: dict[str, ArtifactRef] | None = None
    artifact_index_valid = True
    for child in children:
        if re.fullmatch(r"\.[0-9a-f]{64}\.lock", child.name):
            continue
        if (
            child.is_symlink()
            or not child.is_dir()
            or not is_execution_job_id(child.name)
        ):
            continue
        try:
            directory, status = _read_status(child.name)
        except _ExecutionJobUnavailable:
            continue
        if status.get("status") not in TERMINAL_JOB_STATES:
            continue
        job_type, registry_record, route_error = _trusted_execution_job_route(
            child.name,
            status,
        )
        persisted_operation_id = status.get("operation_id")
        if route_error is not None:
            if (
                status.get("status") == "success"
                and status.get("job_type")
                in {"action_execution", "programmable_analysis"}
                and isinstance(persisted_operation_id, str)
            ):
                invalid_operations.add(persisted_operation_id)
            continue
        if job_type == "programmable_analysis":
            if artifact_index_records is None:
                try:
                    store = store or ArtifactStore(root=workspace_root(), create=False)
                    artifact_index_records, artifact_index_valid = _read_artifact_index_records(
                        store
                    )
                except (FileNotFoundError, OSError, RuntimeError, ValueError):
                    artifact_index_records = {}
                    artifact_index_valid = False
            if not artifact_index_valid:
                if isinstance(persisted_operation_id, str):
                    invalid_operations.add(persisted_operation_id)
                continue
            programmable_candidates, programmable_invalid_artifacts, programmable_invalid_operations = (
                _read_programmable_analysis_artifact_bindings(
                    child.name,
                    directory,
                    status,
                    registry_record,
                    artifact_index_records,
                )
            )
            for artifact_id, producer_identities in programmable_candidates.items():
                candidates.setdefault(artifact_id, set()).update(producer_identities)
            invalid_artifacts.update(programmable_invalid_artifacts)
            invalid_operations.update(programmable_invalid_operations)
            continue
        if job_type != "action_execution":
            if (
                status.get("job_type") == "action_execution"
                and isinstance(persisted_operation_id, str)
            ):
                invalid_operations.add(persisted_operation_id)
            continue
        identity: tuple[str, str] | None = None
        try:
            receipt = _read_private_json(directory / "operation_receipt.json")
            identity = _canonical_job_reference_identity(
                ExecutionJobReference(job_id=child.name),
                status,
                receipt or {},
            )
            raw_result = _read_private_json(directory / "action_result.json")
            if identity is None:
                if (
                    status.get("status") == "success"
                    and isinstance(persisted_operation_id, str)
                ):
                    invalid_operations.add(persisted_operation_id)
                continue
            operation_id, _fingerprint = identity
            # Action job 的 operation 身份由 scope 和 submission_key 唯一计算；
            # typed OperationIdempotencyStore 中的 operation_id 不能替代它。
            if not isinstance(receipt, Mapping):
                invalid_operations.add(operation_id)
                continue
            if not _job_submission_identity_is_consistent(
                status,
                receipt,
                operation_id,
                registry_record=registry_record,
            ) or not operation_id.startswith("scope-submission-"):
                invalid_operations.add(operation_id)
                continue
            if raw_result is None:
                # Action worker 在进入科学 ActionResult 生成前正常失败、超时或被取消时，
                # 不会留下 action_result.json。该作业仍由 status 与 receipt 证明终态，
                # 但没有 artifact 来源链可供恢复；它不能污染同一 scope 中另一组成功
                # 引用。成功 worker 缺少 canonical 结果仍按完整性损坏 fail-closed。
                if status.get("status") != "success":
                    continue
                invalid_operations.add(operation_id)
                continue
            action_result = ActionResult.model_validate(raw_result)
            references = list(action_result.output_artifacts)
            provenance = action_result.provenance
            action_identity = _read_action_job_record_identity(
                directory,
                status,
                registry_record,
            )
            if action_identity is None:
                invalid_operations.add(operation_id)
                for reference in references:
                    invalid_artifacts.add(reference.artifact_id)
                continue
            action_id, backend_id = action_identity
            job_status = status.get("status")
            result_status_is_consistent = _validate_action_result_status_against_worker_status(
                job_status,
                action_result.status,
            )
            if (
                provenance.get("async_job_id") != child.name
                or not result_status_is_consistent
                or action_result.action != action_id
                or action_result.requested_backend != backend_id
                or action_result.backend != backend_id
            ):
                invalid_operations.add(operation_id)
                for reference in references:
                    invalid_artifacts.add(reference.artifact_id)
                continue
            if any(
                reference.producer_action != action_id
                or reference.producer_backend != backend_id
                for reference in references
            ):
                invalid_operations.add(operation_id)
                for reference in references:
                    invalid_artifacts.add(reference.artifact_id)
                continue
            # 新记录可以在结果 provenance 中重复保存提交身份；重复字段不是
            # 新权威来源，但一旦存在就必须与 status/receipt 交叉一致。
            provenance_conflict = False
            for field in ("submission_operation_id", "operation_id"):
                supplied = provenance.get(field)
                if supplied is not None and supplied != operation_id:
                    provenance_conflict = True
                    break
            if provenance_conflict:
                invalid_operations.add(operation_id)
                for reference in references:
                    invalid_artifacts.add(reference.artifact_id)
                continue
        except (
            _OperationIdentityConflict,
            OSError,
            TypeError,
            ValueError,
            UnicodeError,
        ):
            if identity is not None:
                invalid_operations.add(identity[0])
            continue

        seen_in_result: set[str] = set()
        for reference in references:
            artifact_id = reference.artifact_id
            if artifact_id in seen_in_result:
                invalid_artifacts.add(artifact_id)
                continue
            seen_in_result.add(artifact_id)
            if artifact_id in invalid_artifacts:
                continue
            if store is None:
                try:
                    store = ArtifactStore(root=workspace_root(), create=False)
                except (FileNotFoundError, OSError, RuntimeError, ValueError):
                    invalid_artifacts.add(artifact_id)
                    continue
            try:
                canonical = store.find(artifact_id)
                if canonical.model_dump(mode="json") != reference.model_dump(mode="json"):
                    raise ValueError("action result artifact manifest mismatch")
                if _artifact_reference_integrity_error(canonical) is not None:
                    raise ValueError("action result artifact integrity mismatch")
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                invalid_artifacts.add(artifact_id)
                continue
            candidates.setdefault(artifact_id, set()).add((child.name, operation_id))

    bindings: dict[str, dict[str, str]] = {}
    for artifact_id, producer_identities in candidates.items():
        if artifact_id in invalid_artifacts or len(producer_identities) != 1:
            invalid_artifacts.add(artifact_id)
            continue
        producer_job_id, producer_operation_id = next(iter(producer_identities))
        bindings[artifact_id] = {
            "producer_job_id": producer_job_id,
            "producer_submission_operation_id": producer_operation_id,
        }
    return bindings, invalid_artifacts, invalid_operations


def _read_artifact_reference_manifest(
    reference: ExecutionArtifactReference,
) -> Any | None:
    """读取当前 workspace 的 canonical Artifact manifest。"""

    try:
        store = ArtifactStore(root=workspace_root(), create=False)
        return store.find(reference.artifact_id)
    except (FileNotFoundError, KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _artifact_manifest_differs(reference: ExecutionArtifactReference, canonical: Any) -> bool:
    """判断调用方附带的 Artifact 身份字段是否与 manifest 一致。"""

    supplied = reference.model_dump(mode="json", exclude_none=True)
    expected = canonical.model_dump(mode="json")
    return any(expected.get(key) != value for key, value in supplied.items())


def _artifact_reference_integrity_error(canonical: Any) -> str | None:
    """核对 Artifact 文件存在性、普通文件类型和 SHA-256 完整性。"""

    try:
        path = resolve_workspace_path(canonical.path, must_exist=True)
    except (FileNotFoundError, OSError, ValueError):
        return "artifact_missing"
    if path.is_symlink() or not path.is_file():
        return "artifact_missing"
    try:
        _raw, actual_sha256 = read_file_snapshot(path)
    except (OSError, ValueError):
        return "artifact_unreadable"
    if actual_sha256 != canonical.sha256:
        return "artifact_hash_mismatch"
    return None


def verify_execution_references(
    request: VerifyExecutionReferencesRequest,
) -> dict[str, Any]:
    """只读核对当前 capability scope 的 job、operation 和 artifact 引用。

    输入是显式列出的三类引用；动作只读取持久状态、幂等收据、ArtifactStore
    canonical manifest 和实际文件 SHA，不读取日志、不返回宿主路径，也不会重新执行
    Action 或提交作业。返回 schema_version=2 的逐项 ``valid``、``invalid`` 或
    ``unavailable`` 结果；缺少认证 scope、未知引用和跨 scope 引用统一为
    ``unavailable``，损坏或身份不一致的当前 scope 记录为 ``invalid``。响应同时
    返回当前 scope 的 job inventory、typed operation inventory 及其活动状态；由
    经过严格核对的 canonical action_execution job 的 ``action_result.json``，或
    programmable_analysis job 的冻结合同、manifest、实际输出和 ArtifactStore
    记录恢复的 artifact valid item，还会带唯一 ``producer_job_id`` 和
    ``producer_submission_operation_id``。这两个字段属于作业来源链，不进入 typed
    ``scope_operation_inventory``。显式 operation refs 只核对调用方实际用于结论的
    operation；同 scope 中未被引用的探索 operation 仍出现在 inventory 中，但不会
    否决另一组完整证据。被引用 operation 未完成、重复、损坏、foreign 或无法恢复时
    仍令 ``all_valid`` 为 false。响应 schema_version=2，旧 schema 的泛化 artifact
    ``operation_id`` 不再作为 canonical 输出。
    """

    if current_scope() is None:
        return _references_unavailable_without_scope(request)
    job_inventory, has_active_jobs, inventory_complete = _scope_job_inventory()
    operation_inventory, has_active_operations, operation_inventory_complete = (
        _scope_operation_inventory()
    )
    artifact_bindings, invalid_artifacts, invalid_action_operations = (
        _read_action_result_artifact_bindings()
    )
    items = _verify_reference_items(
        request,
        artifact_bindings=artifact_bindings,
        invalid_artifacts=invalid_artifacts,
    )
    _mark_duplicate_reference_items(request, items)
    return {
        "schema_version": 2,
        "status": "success",
        "all_valid": (
            all(item["status"] == "valid" for item in items)
            and inventory_complete
            and has_active_jobs is False
            and operation_inventory_complete
            and not invalid_action_operations
        ),
        "items": items,
        "scope_job_inventory": job_inventory,
        "has_active_jobs": has_active_jobs,
        "inventory_complete": inventory_complete,
        "scope_operation_inventory": operation_inventory,
        "has_active_operations": has_active_operations,
        "operation_inventory_complete": operation_inventory_complete,
        "automatic_fallback": False,
    }


def _references_unavailable_without_scope(
    request: VerifyExecutionReferencesRequest,
) -> dict[str, Any]:
    """在没有 capability scope 时返回同构但不泄漏引用身份的结果。"""

    items = [_reference_unavailable("job") for _ in request.jobs]
    items += [_reference_unavailable("operation") for _ in request.operations]
    items += [_reference_unavailable("artifact") for _ in request.artifacts]
    return {
        "schema_version": 2,
        "status": "success",
        "all_valid": False,
        "items": items,
        "scope_job_inventory": [],
        "has_active_jobs": None,
        "inventory_complete": False,
        "scope_operation_inventory": [],
        "has_active_operations": None,
        "operation_inventory_complete": False,
        "automatic_fallback": False,
    }


def _mark_duplicate_reference_items(
    request: VerifyExecutionReferencesRequest,
    items: list[dict[str, Any]],
) -> None:
    """把请求中重复的 job、operation 或 artifact 引用全部标为 invalid。

    输入是已经通过模型校验的三类引用和按同一顺序生成的结果项；动作只按各类
    canonical id 找出重复位置，并将重复项替换为不含身份细节的完整性错误。这样
    同一个真实对象不能通过在请求中重复出现来满足“每项都有证据”的计数，也不会
    改变服务端 inventory。结果列表长度与输入不一致时直接停止标记，由上层的
    ``all_valid`` 检查继续保持保守结果。
    """

    offset = 0
    for references, kind, field_name in (
        (request.jobs, "job", "job_id"),
        (request.operations, "operation", "operation_id"),
        (request.artifacts, "artifact", "artifact_id"),
    ):
        if offset + len(references) > len(items):
            return
        positions: dict[str, list[int]] = {}
        for index, reference in enumerate(references):
            identity = getattr(reference, field_name)
            positions.setdefault(identity, []).append(index)
        for duplicate_positions in positions.values():
            if len(duplicate_positions) < 2:
                continue
            for index in duplicate_positions:
                items[offset + index] = _reference_invalid(
                    kind,
                    "duplicate_reference",
                )
        offset += len(references)


def _verify_reference_items(
    request: VerifyExecutionReferencesRequest,
    *,
    artifact_bindings: Mapping[str, Mapping[str, str]],
    invalid_artifacts: set[str],
) -> list[dict[str, Any]]:
    """按请求顺序核对 job、operation 和 artifact 引用。"""

    items = [_verify_execution_job_reference(item) for item in request.jobs]
    items += [
        _verify_execution_operation_reference(item) for item in request.operations
    ]
    for reference in request.artifacts:
        item = _verify_execution_artifact_reference(reference)
        if item.get("status") != "valid":
            items.append(item)
            continue
        binding = artifact_bindings.get(reference.artifact_id)
        if binding is not None:
            item.update(binding)
        else:
            # Evidence artifact 必须能回溯到当前 scope 的权威 Action job；不能
            # 因为请求中恰好只有一个 typed operation 就把它猜成 artifact 来源。
            items.append(
                _reference_invalid("artifact", "producer_binding_unavailable")
            )
            continue
        items.append(item)
    return items


class _OperationIdentityConflict(RuntimeError):
    """持久化作业身份来源不一致时使用的受控内部错误。"""


def _operation_identity_evidence(
    directory: Path, status: dict[str, Any]
) -> dict[str, Any]:
    """从当前作业状态和 receipt 提取有限的 operation 身份证据。

    只返回 job 类型、operation/job 标识和摘要指纹，不把原始请求、命令或本机
    路径复制到恢复身份字段。旧作业没有 caller 指纹时保持缺失，由上层拒绝猜测。
    """

    receipt = _read_operation_identity_receipt(directory)
    metadata = status.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    sources = {
        "status": status,
        "status.metadata": metadata,
        "operation_receipt": receipt,
    }
    evidence: dict[str, Any] = {}
    _collect_canonical_identity_fingerprint(sources, evidence)
    for field in (
        "job_type",
        "operation_id",
        "job_id",
        "caller_request_fingerprint",
        "submission_key",
    ):
        _collect_consistent_identity_field(sources, evidence, field)
    _validate_scope_operation_identity(evidence)
    return evidence


def _read_operation_identity_receipt(directory: Path) -> dict[str, Any]:
    """读取作业 receipt；文件不可信时拒绝生成恢复身份。"""

    receipt_path = directory / "operation_receipt.json"
    if receipt_path.is_symlink() or (
        receipt_path.exists() and not receipt_path.is_file()
    ):
        raise _OperationIdentityConflict(
            "operation receipt is not a private regular file, refuse identity output"
        )
    if not receipt_path.is_file():
        return {}
    try:
        value = _read_private_json(receipt_path)
    except _OperationIdentityConflict as exc:
        raise _OperationIdentityConflict(
            "operation receipt 无法读取或不是合法 JSON，拒绝输出恢复身份"
        ) from exc
    if value is None:
        raise _OperationIdentityConflict(
            "operation receipt disappeared during identity verification"
        )
    return value


def _collect_consistent_identity_field(
    sources: Mapping[str, Mapping[str, Any]],
    evidence: dict[str, Any],
    field: str,
    *,
    evidence_name: str | None = None,
) -> None:
    """交叉核对一个持久字段，只把无冲突值写入公开证据。"""

    values = [
        (source_name, source[field])
        for source_name, source in sources.items()
        if field in source
    ]
    if not values:
        return
    first_source, first_value = values[0]
    conflicting = [
        (source_name, value)
        for source_name, value in values[1:]
        if value != first_value
    ]
    if conflicting:
        conflicting_sources = ", ".join(
            source_name for source_name, _value in conflicting
        )
        raise _OperationIdentityConflict(
            f"operation identity field {field} has conflicting persisted values "
            f"between {first_source} and {conflicting_sources}"
        )
    if first_value is not None:
        evidence[evidence_name or field] = first_value


def _collect_canonical_identity_fingerprint(
    sources: Mapping[str, Mapping[str, Any]],
    evidence: dict[str, Any],
) -> None:
    """交叉核对 fingerprint 与 operation_fingerprint 两个持久化别名。"""

    canonical_values = [
        (f"{source_name}.{field}", source[field])
        for source_name, source in sources.items()
        for field in ("fingerprint", "operation_fingerprint")
        if field in source
    ]
    if canonical_values:
        first_source, first_value = canonical_values[0]
        conflicting = [
            (source_name, value)
            for source_name, value in canonical_values[1:]
            if value != first_value
        ]
        if conflicting:
            conflicting_sources = ", ".join(
                source_name for source_name, _value in conflicting
            )
            raise _OperationIdentityConflict(
                "operation identity canonical fingerprint has conflicting persisted "
                f"values between {first_source} and {conflicting_sources}"
            )
        if first_value is not None:
            evidence["operation_fingerprint"] = first_value

def _validate_scope_operation_identity(evidence: Mapping[str, Any]) -> None:
    """验证 scoped operation 的 submission_key 能支撑恢复身份。"""

    operation_identity = evidence.get("operation_id")
    if isinstance(operation_identity, str) and operation_identity.startswith(
        "scope-submission-"
    ):
        if not isinstance(evidence.get("submission_key"), str):
            raise _OperationIdentityConflict(
                "scope operation 缺少可核对的 submission_key"
            )
        try:
            validate_submission_key(evidence["submission_key"])
        except ValueError as exc:
            raise _OperationIdentityConflict(
                "scope operation 缺少可核对的 submission_key"
            ) from exc


def _tail(path: Path, characters: int) -> str:
    if characters <= 0 or not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - characters * 4))
        data = handle.read()
    return data.decode("utf-8", errors="replace")[-characters:]


def _bounded_text(path: Path, maximum_bytes: int = 8 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        data = handle.read(maximum_bytes)
    return data.decode("utf-8", errors="replace")


def _program_failure_diagnostic(
    directory: Path, status: dict[str, Any]
) -> dict[str, Any] | None:
    if (
        status.get("job_type") != "programmable_analysis"
        or status.get("status") != "failed"
    ):
        return None
    diagnostic_path = directory / "failure_diagnostic.json"
    if diagnostic_path.is_file():
        try:
            return json.loads(diagnostic_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    stderr = _bounded_text(directory / "stderr.log", maximum_bytes=2 * 1024 * 1024)
    exception_type = None
    message = None
    exception_match = None
    for line in reversed(stderr.splitlines()):
        candidate = re.match(
            r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(?P<message>.*)$",
            line.strip(),
        )
        if candidate:
            exception_match = candidate
            break
    if exception_match:
        exception_type = exception_match.group("type")
        message = exception_match.group("message")
    frames = re.findall(r'File "([^"]+)", line (\d+)(?:, in ([^\n]+))?', stderr)
    source_file = None
    source_line = None
    function = None
    for raw_file, raw_line, raw_function in reversed(frames):
        candidate = Path(raw_file)
        if not candidate.is_absolute():
            candidate = directory / candidate
        try:
            candidate.resolve().relative_to(directory.resolve())
        except ValueError:
            continue
        if candidate.is_file() and candidate.suffix == ".py":
            source_file = candidate
            source_line = int(raw_line)
            function = raw_function.strip() or None
            break
    source_context = None
    if source_file is not None and source_line is not None:
        lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, source_line - 3)
        stop = min(len(lines), source_line + 2)
        source_context = [
            {"line": index + 1, "text": lines[index]}
            for index in range(start, stop)
        ]

    lowered = (message or stderr[-1000:]).casefold()
    classification = "nonzero_exit"
    candidate_fixes = [
        "Read the structured input summary and the failing source line before revising the program.",
        "Validate intermediate values and declared output schemas before resubmission.",
    ]
    details: dict[str, Any] = {}
    if exception_type == "KeyError":
        classification = "missing_mapping_key"
        missing_key = (message or "").strip().strip("'\"")
        details["missing_key"] = missing_key or None
        candidate_fixes = [
            "Inspect the actual JSON/object keys with inspect_analysis_inputs.",
            "Use the existing field name or handle the field as optional with an explicit fallback.",
        ]
    elif "unknown input name" in lowered or "unknown output name" in lowered:
        classification = "job_context_name_mismatch"
        candidate_fixes = [
            "Make the JobContext helper name exactly match the submitted contract name.",
            "Run validate_analysis_program again before resubmission.",
        ]
    elif exception_type == "NameError":
        classification = "undefined_name"
        candidate_fixes = [
            "Define the reported variable on every control-flow path before it is used.",
            "Run a focused unit calculation on one input record before processing the full dataset.",
        ]
    elif "nonetype" in lowered or (
        "none" in lowered and exception_type == "TypeError"
    ):
        classification = "unexpected_null_value"
        candidate_fixes = [
            "Check for null/None before arithmetic or numeric formatting.",
            "Decide explicitly whether a missing value should be skipped, rejected, or replaced.",
        ]
    elif "unpack" in lowered or "values to unpack" in lowered:
        classification = "unpack_shape_mismatch"
        candidate_fixes = [
            "Inspect the sequence or mapping shape before unpacking it.",
            "Use indexed or named access when the input record length is not guaranteed.",
        ]
    diagnostic = {
        "schema_version": 1,
        "job_id": status.get("job_id"),
        "classification": classification,
        "exception_type": exception_type,
        "message": message,
        "source_file": (
            str(source_file.resolve().relative_to(directory.resolve()))
            if source_file is not None
            else None
        ),
        "source_line": source_line,
        "function": function,
        "source_context": source_context,
        "details": details,
        "candidate_fixes": candidate_fixes,
        "diagnostic_boundary": (
            "This classifies the Python failure mechanically; it does not determine the correct "
            "scientific interpretation or choose replacement data."
        ),
    }
    _atomic_json(diagnostic_path, diagnostic)
    return diagnostic


def _native_failure_diagnostic(
    directory: Path,
    status: dict[str, Any],
) -> dict[str, Any] | None:
    """从失败的 native Gaussian 作业提取有限机械诊断，不返回路径或原始日志。"""

    metadata = status.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if (
        status.get("job_type") != "native_software"
        or status.get("status") != "failed"
        or str(metadata.get("software_id") or "").casefold() != "gaussian"
    ):
        return None
    diagnostic_path = directory / "failure_diagnostic.json"
    if diagnostic_path.is_file():
        try:
            existing = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict):
            return existing

    stdout = _bounded_text(directory / "stdout.log")
    if "QPErr --- A syntax error was detected in the input line." not in stdout:
        return None

    details: dict[str, Any] = {}
    candidate_fixes = [
        "只修订 Gaussian route section 中被拒绝的原生关键字，不改变方法、基组或计算目标。",
        "根据 QPErr 指示的字段重新生成单行 route，再以原有科学设置提交。",
    ]
    input_deck_validation = metadata.get("input_deck_validation")
    segments = (
        input_deck_validation.get("segments")
        if isinstance(input_deck_validation, dict)
        else None
    )
    routes = [
        segment.get("route")
        for segment in segments or []
        if isinstance(segment, dict) and isinstance(segment.get("route"), str)
    ]
    solvent_value = None
    for route in routes:
        match = re.search(r"(?i)\bsolvent\s*=\s*([^,)]+)", route)
        if match is not None:
            solvent_value = match.group(1).strip()
            break
    qperr_points_to_solvent = re.search(
        r"(?is)QPErr[^\r\n]*[\r\n]+[^\r\n]*\bSolvent\s*=\s*Ethy",
        stdout,
    ) is not None
    if (
        qperr_points_to_solvent
        and isinstance(solvent_value, str)
        and solvent_value.casefold() == "ethylacetate"
    ):
        details = {
            "rejected_route_field": "SCRF.solvent",
            "rejected_value": solvent_value,
            "recommended_value": "EthylEthanoate",
        }
        candidate_fixes.insert(
            0,
            "乙酸乙酯使用 Gaussian 原生别名 EthylEthanoate，不使用 EthylAcetate。",
        )

    diagnostic = {
        "schema_version": 1,
        "job_id": status.get("job_id"),
        "classification": "gaussian_route_syntax_error",
        "message": "Gaussian 拒绝了 route section 的输入语法（QPErr）。",
        "details": details,
        "candidate_fixes": candidate_fixes,
        "diagnostic_boundary": (
            "该诊断只定位 Gaussian 输入语法，不判断科学路线是否正确，也不自动改写 route。"
        ),
    }
    _atomic_json(diagnostic_path, diagnostic)
    return diagnostic


def _execution_failure_diagnostic(
    directory: Path,
    status: dict[str, Any],
) -> dict[str, Any] | None:
    """按作业类型生成失败诊断；每次只返回一个与当前作业匹配的诊断。"""

    return _program_failure_diagnostic(directory, status) or _native_failure_diagnostic(
        directory,
        status,
    )


def _process_axis(job_status: str) -> str:
    return {
        "queued": "queued",
        "running": "running",
        "success": "completed",
        "failed": "failed",
        "timeout": "timed_out",
        "cancelled": "cancelled",
    }.get(job_status, "unknown")


def _native_scientific_axes(
    directory: Path, status: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = status.get("metadata") or {}
    software_id = str(metadata.get("software_id") or "")
    calculation_intent = str(metadata.get("calculation_intent") or "unknown")
    process_ok = status.get("status") == "success"
    stdout = _bounded_text(directory / "stdout.log")
    software = {"status": "not_checked", "evidence": []}
    convergence = {"status": "not_checked", "evidence": []}
    artifacts = {"status": "not_declared", "evidence": []}
    if not process_ok:
        software = {"status": "not_reached", "evidence": [status.get("status")]}
        convergence = {"status": "not_reached", "evidence": []}
        artifacts = {"status": "not_validated", "evidence": []}
        return software, convergence, artifacts
    if software_id == "orca":
        normal = "ORCA TERMINATED NORMALLY" in stdout
        software = {
            "status": "normal_termination" if normal else "failed",
            "evidence": ["ORCA TERMINATED NORMALLY"] if normal else ["normal termination marker missing"],
        }
        scf_converged = "SCF CONVERGED" in stdout
        optimization_converged = "THE OPTIMIZATION HAS CONVERGED" in stdout
        frequency_complete = "VIBRATIONAL FREQUENCIES" in stdout
        frequencies = [
            float(value)
            for value in re.findall(
                r"^\s*\d+:\s*(-?\d+(?:\.\d+)?)\s+cm\*\*-1",
                stdout,
                flags=re.MULTILINE,
            )
        ]
        imaginary_count = sum(value < -1.0 for value in frequencies)
        requirements = {
            "single_point": scf_converged,
            "geometry_optimization": optimization_converged,
            "frequency": scf_converged and frequency_complete,
            "optimization_frequency": optimization_converged and frequency_complete,
            "transition_state": (
                optimization_converged and frequency_complete and imaginary_count == 1
            ),
        }
        converged = requirements.get(calculation_intent)
        markers = [
            marker
            for marker, present in (
                ("SCF CONVERGED", scf_converged),
                ("THE OPTIMIZATION HAS CONVERGED", optimization_converged),
                ("VIBRATIONAL FREQUENCIES", frequency_complete),
            )
            if present
        ]
        if frequency_complete:
            markers.append(f"imaginary_frequency_count={imaginary_count}")
        convergence = {
            "status": (
                "not_checked"
                if converged is None
                else "converged" if converged else "failed"
            ),
            "evidence": markers,
            "calculation_intent": calculation_intent,
        }
        artifacts = {
            "status": "valid" if normal and (directory / "stdout.log").stat().st_size else "invalid",
            "evidence": ["stdout.log"],
        }
    elif software_id == "gaussian":
        stderr = _bounded_text(
            directory / "stderr.log", maximum_bytes=2 * 1024 * 1024
        )
        return_code = status.get("return_code")
        archive_warning_completion = (
            return_code is None
            or return_code == 0
            or gaussian_archive_only_completion(
                {
                    "returncode": return_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timeout": status.get("timeout"),
                    "cleanup_pending": status.get("cleanup_pending"),
                    "memory_exceeded": status.get("memory_exceeded"),
                    "cancelled": status.get("cancelled"),
                    "process_group_activity": status.get("process_group_activity"),
                    "background_process_cleanup": status.get(
                        "background_process_cleanup"
                    ),
                }
            )
        )
        normal = (
            archive_warning_completion
            and gaussian_output_ends_with_normal_termination(stdout)
            and not gaussian_contains_fatal_marker(stdout, stderr)
        )
        software = {
            "status": "normal_termination" if normal else "failed",
            "evidence": (
                ["Normal termination of Gaussian"]
                if normal
                else ["normal termination marker missing"]
            ),
        }
        scf_converged = "SCF Done:" in stdout
        optimization_converged = "Optimization completed" in stdout
        nimag_matches = re.findall(r"NImag=\s*(\d+)", stdout)
        imaginary_count = int(nimag_matches[-1]) if nimag_matches else None
        frequency_complete = imaginary_count is not None or "Harmonic frequencies" in stdout
        reaction_path_branch_complete = bool(
            re.search(
                r"Calculation of (?:FORWARD|REVERSE) path complete",
                stdout,
                flags=re.IGNORECASE,
            )
        )
        reaction_path_minimum_detected = bool(
            re.search(
                r"PES minimum detected on this side of the pathway",
                stdout,
                flags=re.IGNORECASE,
            )
        )
        requirements = {
            "single_point": scf_converged,
            "geometry_optimization": optimization_converged,
            "frequency": scf_converged and frequency_complete,
            "optimization_frequency": optimization_converged and frequency_complete,
            "transition_state": (
                optimization_converged and frequency_complete and imaginary_count == 1
            ),
            "reaction_path": scf_converged and reaction_path_minimum_detected,
        }
        converged = requirements.get(calculation_intent)
        markers = [
            marker
            for marker, present in (
                ("SCF Done:", scf_converged),
                ("Optimization completed", optimization_converged),
                ("frequency calculation completed", frequency_complete),
                ("IRC branch completed", reaction_path_branch_complete),
                ("IRC PES minimum detected", reaction_path_minimum_detected),
            )
            if present
        ]
        if imaginary_count is not None:
            markers.append(f"imaginary_frequency_count={imaginary_count}")
        convergence = {
            "status": (
                "not_checked"
                if converged is None
                else "converged" if converged else "failed"
            ),
            "evidence": markers,
            "calculation_intent": calculation_intent,
        }
        artifacts = {"status": "valid" if normal else "invalid", "evidence": ["stdout.log"]}
    elif software_id == "crest":
        normal = "CREST terminated normally" in stdout
        software = {
            "status": "normal_termination" if normal else "failed",
            "evidence": ["CREST terminated normally"] if normal else ["normal termination marker missing"],
        }
        expected_by_intent = {
            "conformer_search": "crest_conformers.xyz",
            "protonation": "protonated.xyz",
            "deprotonation": "deprotonated.xyz",
            "tautomerization": "tautomers.xyz",
        }
        expected_name = expected_by_intent.get(calculation_intent)
        expected = directory / expected_name if expected_name else None
        valid = bool(expected and expected.is_file() and expected.stat().st_size)
        convergence = {
            "status": (
                "not_checked"
                if expected is None
                else "converged" if normal and valid else "failed"
            ),
            "evidence": [expected_name] if valid and expected_name else [],
            "calculation_intent": calculation_intent,
        }
        artifacts = {
            "status": "not_checked" if expected is None else "valid" if valid else "invalid",
            "evidence": [expected_name] if valid and expected_name else [],
        }
    elif software_id == "vasp":
        outcar = _bounded_text(directory / "OUTCAR")
        normal = "General timing and accounting informations for this job" in outcar
        electronic_converged = "aborting loop because EDIFF is reached" in outcar
        ionic_converged = "reached required accuracy" in outcar
        frequency_complete = "Eigenvectors and eigenvalues of the dynamical matrix" in outcar
        requirements = {
            "single_point": electronic_converged,
            "ionic_relaxation": electronic_converged and ionic_converged,
            "frequency": electronic_converged and frequency_complete,
        }
        converged = requirements.get(calculation_intent)
        software = {
            "status": "normal_termination" if normal else "failed",
            "evidence": ["General timing and accounting informations for this job"] if normal else ["OUTCAR timing footer missing"],
        }
        convergence = {
            "status": (
                "not_checked"
                if converged is None
                else "converged" if converged else "failed"
            ),
            "evidence": [
                marker
                for marker, present in (
                    ("electronic EDIFF reached", electronic_converged),
                    ("ionic relaxation reached required accuracy", ionic_converged),
                    ("dynamical matrix frequencies completed", frequency_complete),
                )
                if present
            ],
            "calculation_intent": calculation_intent,
        }
        required = ["OUTCAR", "vasprun.xml"]
        valid = all((directory / item).is_file() and (directory / item).stat().st_size for item in required)
        artifacts = {"status": "valid" if valid else "invalid", "evidence": required}
    elif software_id == "lobster":
        lobsterout = _bounded_text(directory / "lobsterout") or stdout
        normal = "finished in" in lobsterout and not re.search(r"^ERROR:", lobsterout, flags=re.M)
        software = {
            "status": "normal_termination" if normal else "failed",
            "evidence": ["finished in"] if normal else ["LOBSTER ERROR marker or missing finish marker"],
        }
        spilling = re.search(r"abs\. charge spilling:\s*([0-9.]+)%", lobsterout)
        convergence = {
            "status": "projection_complete" if normal else "failed",
            "evidence": ([f"absolute_charge_spilling_percent={spilling.group(1)}"] if spilling else []),
            "calculation_intent": calculation_intent,
        }
        required = ["lobsterout", "COHPCAR.lobster"]
        valid = all((directory / item).is_file() and (directory / item).stat().st_size for item in required)
        artifacts = {"status": "valid" if valid else "invalid", "evidence": required}
    return software, convergence, artifacts


def _execution_status_axes(
    directory: Path,
    status: dict[str, Any],
    *,
    analysis_artifact_status: str | None = None,
) -> dict[str, Any]:
    request_status = "accepted"
    process_status = _process_axis(str(status.get("status") or ""))
    if status.get("job_type") == "native_software":
        software, convergence, artifacts = _native_scientific_axes(directory, status)
    else:
        software = {"status": "not_applicable", "evidence": []}
        convergence = {"status": "not_declared", "evidence": []}
        artifact_value = analysis_artifact_status
        if artifact_value is None:
            artifact_value = (
                "pending_collection"
                if status.get("status") in TERMINAL_JOB_STATES
                else "pending_execution"
            )
        artifacts = {"status": artifact_value, "evidence": ["artifact_manifest.json"] if analysis_artifact_status else []}
    if process_status != "completed":
        scientific = "not_reached"
    elif (
        software["status"] == "failed"
        or convergence["status"] == "failed"
        or artifacts["status"] == "invalid"
    ):
        scientific = "mechanically_invalid"
    elif convergence["status"] in {"converged", "projection_complete"} and artifacts["status"] == "valid":
        scientific = "mechanically_valid"
    elif status.get("job_type") == "programmable_analysis" and artifacts["status"] == "valid":
        scientific = "mechanically_valid"
    else:
        scientific = "not_checked"
    return {
        "request_status": request_status,
        "process_status": process_status,
        "software_status": software["status"],
        "convergence_status": convergence["status"],
        "artifact_status": artifacts["status"],
        "scientific_validation_status": scientific,
        "details": {
            "software": software,
            "convergence": convergence,
            "artifacts": artifacts,
            "scientific_validation_boundary": (
                "Mechanical validation does not judge whether the Agent's final scientific "
                "interpretation or paper conclusion is correct; that remains the Judger's role."
            ),
        },
    }


def _resource_availability() -> dict[str, Any]:
    budget = resource_budget_record()
    try:
        reserved = active_resource_usage()
        active_jobs = active_resource_jobs()
        gaussian_queue = read_gaussian_queue_snapshot()
        tracking = "workspace_reservations"
    except RuntimeError:
        # 共享 scheduler 或其索引无法核对时，查询也必须按最保守结果报告。
        # reserve_resources 同样会拒绝这类状态；这里不能把不可核对的容量当作空闲，
        # 否则调用方可能据此并发提交并绕过 fail-closed 预算边界。
        reserved = {
            name: int(budget[name])
            for name in ("cpu_cores", "memory_mb", "gpu_count")
        }
        active_jobs = []
        gaussian_queue = {
            "status": "unavailable",
            "scope": "global",
            "items": [],
            "item_count": 0,
        }
        tracking = "unavailable_without_workspace"
    available = {
        name: max(0, int(budget[name]) - int(reserved[name]))
        for name in ("cpu_cores", "memory_mb", "gpu_count")
    }
    return {
        "budget": budget,
        "currently_reserved": reserved,
        "available": available,
        "active_jobs": active_jobs,
        "active_job_count": len(active_jobs),
        # 该视图只暴露共享 FIFO 的最小脱敏字段；原有资源字段保持兼容。
        "gaussian_queue": gaussian_queue,
        "reservation_tracking": tracking,
    }


def _read_frozen_catalog_snapshot() -> tuple[dict[str, Any], Path]:
    """读取进程内启动快照并核对其 workspace，不从漂移目录现算。"""

    lock = getattr(_catalog_state, "_ACTIVE_CATALOG_LOCK", None)
    if lock is None:
        raise ResourceStateUnavailable("MiniChem active catalog snapshot is unavailable")
    with lock:
        snapshot = getattr(_catalog_state, "_ACTIVE_CATALOG_SNAPSHOT", None)
        workspace = getattr(_catalog_state, "_ACTIVE_CATALOG_WORKSPACE", None)
        if not isinstance(snapshot, dict) or workspace is None:
            raise ResourceStateUnavailable("MiniChem active catalog snapshot is unavailable")
        frozen = dict(snapshot)
    workspace_path = getattr(_catalog_state, "_catalog_workspace_path", None)
    if not callable(workspace_path):
        raise ResourceStateUnavailable("MiniChem active catalog workspace is unavailable")
    try:
        if Path(workspace) != Path(workspace_path()):
            raise ResourceStateUnavailable("MiniChem active catalog workspace changed")
    except ResourceStateUnavailable:
        raise
    except Exception as exc:
        raise ResourceStateUnavailable(
            "MiniChem active catalog workspace cannot be verified"
        ) from exc
    return frozen, Path(workspace)


def _verify_frozen_catalog_snapshot(snapshot: dict[str, Any]) -> None:
    """核对冻结快照对应文件未被删除或替换。"""

    try:
        on_disk = _catalog_state.active_catalog_snapshot()
    except Exception as exc:
        raise ResourceStateUnavailable(
            "MiniChem active catalog snapshot cannot be verified"
        ) from exc
    if on_disk != snapshot:
        raise ResourceStateUnavailable("MiniChem active catalog snapshot drifted")


def _validate_frozen_catalog_identity(
    snapshot: dict[str, Any],
    workspace: Path,
) -> dict[str, str]:
    """验证 hash、版本和服务 marker，并返回公开目录身份。"""

    catalog_hash = snapshot.get("catalog_hash")
    catalog_version = snapshot.get("schema_version")
    marker_hash = os.environ.get("MINICHEM_MCP_CATALOG_HASH", "").strip()
    marker_workspace = os.environ.get("MINICHEM_MCP_CATALOG_WORKSPACE", "").strip()
    if marker_hash and marker_hash != catalog_hash:
        raise ResourceStateUnavailable("MiniChem active catalog marker drifted")
    if marker_workspace and marker_workspace != str(workspace):
        raise ResourceStateUnavailable("MiniChem active catalog marker workspace changed")
    if (
        not isinstance(catalog_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", catalog_hash) is None
        or isinstance(catalog_version, bool)
        or not isinstance(catalog_version, int)
        or catalog_version <= 0
    ):
        raise ResourceStateUnavailable("MiniChem active catalog identity is invalid")
    return {
        "catalog_hash": catalog_hash,
        "catalog_version": str(catalog_version),
    }


def _read_frozen_catalog_identity() -> dict[str, str]:
    """读取服务启动时安装的目录身份，快照缺失或磁盘漂移时失败。"""

    snapshot, workspace = _read_frozen_catalog_snapshot()
    _verify_frozen_catalog_snapshot(snapshot)
    return _validate_frozen_catalog_identity(snapshot, workspace)


def get_execution_resources(_request: ExecutionResourceRequest) -> dict[str, Any]:
    """返回当前资源预算及服务启动时冻结的目录 hash、版本。"""

    return {
        "status": "success",
        **_resource_availability(),
        **_read_frozen_catalog_identity(),
    }


class _ExecutionJobInventoryConflict(RuntimeError):
    """中央 registry 与持久作业事实无法安全合并。"""


def _execution_job_inventory_error(code: str) -> dict[str, Any]:
    """构造不泄漏 scope、路径或进程身份的清单失败响应。"""

    messages = {
        "execution_job_inventory_unavailable": (
            "Execution job inventory is unavailable in the current capability scope"
        ),
        "execution_job_inventory_inconsistent": (
            "Execution job registry and persistent job state are inconsistent"
        ),
    }
    return {
        "status": "failed",
        "valid": False,
        "jobs": [],
        "active_job_id": None,
        "active_job_ids": [],
        "inventory_complete": False,
        "error": {
            "code": code,
            "message": messages.get(code, messages["execution_job_inventory_unavailable"]),
        },
        "automatic_fallback": False,
    }


def _resolve_execution_job_listing_phase(
    job_status: str,
    registry_phase: Any,
) -> str:
    """按持久 status 与 Registry phase 的异步兼容矩阵解析展示阶段。

    持久 status 来自 supervisor 的作业目录，是当前作业状态的权威事实；Registry
    phase 只用于路由和摘要，允许保留、推进或恢复写入落后。active status 遇到任何
    terminal/collected phase 仍拒绝，两个不同终态也仍拒绝，避免把已完成作业误报为
    可继续运行或把失败作业误报为成功。
    """

    if job_status in {"queued", "running"}:
        if registry_phase not in _REGISTRY_PRETERMINAL_PHASES:
            raise _ExecutionJobInventoryConflict(
                "registry terminal phase conflicts with persistent active status"
            )
        return job_status
    if registry_phase in _REGISTRY_PRETERMINAL_PHASES or registry_phase == "collected":
        return job_status
    if registry_phase == job_status:
        return job_status
    if registry_phase in _REGISTRY_TERMINAL_PHASES:
        raise _ExecutionJobInventoryConflict(
            "registry terminal phase differs from persistent terminal status"
        )
    raise _ExecutionJobInventoryConflict(
        "registry phase is not in the execution compatibility matrix"
    )


def _execution_job_listing_item(
    registry_record: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """从 registry 记录和持久 status 交集构造单个最小清单项。"""

    job_id = registry_record.get("job_id")
    job_type = registry_record.get("job_type")
    if not isinstance(job_id, str) or not isinstance(job_type, str):
        raise _ExecutionJobInventoryConflict("registry job identity is incomplete")
    try:
        _directory, persisted = _read_status(job_id)
    except _ExecutionJobUnavailable as exc:
        raise _ExecutionJobInventoryConflict("persistent job status is unavailable") from exc
    if persisted.get("job_type") != job_type:
        raise _ExecutionJobInventoryConflict("registry and status job_type differ")
    for field in ("operation_id", "operation_fingerprint", "submission_key"):
        registry_value = registry_record.get(field)
        persisted_value = persisted.get(field)
        if (
            registry_value is not None
            and persisted_value is not None
            and registry_value != persisted_value
        ):
            raise _ExecutionJobInventoryConflict(
                f"registry and status {field} differ"
            )
    job_status = persisted.get("status")
    if job_status not in {"queued", "running", *TERMINAL_JOB_STATES}:
        raise _ExecutionJobInventoryConflict("persistent job status is invalid")
    phase = registry_record.get("phase")
    resolved_phase = _resolve_execution_job_listing_phase(job_status, phase)
    item: dict[str, Any] = {
        "job_id": job_id,
        "job_status": job_status,
        "phase": resolved_phase,
        "terminal": job_status in TERMINAL_JOB_STATES,
    }
    for field in ("operation_id", "submission_key"):
        value = registry_record.get(field)
        if value is not None:
            item[field] = value
    if isinstance(persisted.get("error"), Mapping):
        # 清单也必须让模型看到失败的可操作原因；只返回脱敏错误，不返回 status
        # 文件路径、命令或宿主控制字段。
        item["error"] = _sanitize_scoped_error_value(dict(persisted["error"]))
    return item, job_status in {"queued", "running"}


def list_execution_jobs(_request: ExecutionJobListRequest) -> dict[str, Any]:
    """列出当前 scope 的全部 execution job，并返回所有 active job。

    输入是不带资源或筛选参数的请求；动作先读取中央 ``ExecutionJobRegistry``，再
    逐项核对对应的持久 ``status.json``，覆盖 TS path、native Gaussian/软件和
    programmable analysis 作业。返回按 ``job_id`` 稳定排序的最小身份和状态；任何
    registry、scope 或持久状态无法核对时返回 fail-closed 错误，不泄漏路径、PID、日志
    或其他 scope 的身份。多个合法 active 作业不会被视为歧义；模型通过
    ``active_job_ids`` 逐个查询。
    """

    if current_scope() is None:
        return _execution_job_inventory_error("execution_job_inventory_unavailable")
    try:
        records = _execution_job_registry().list_records()
    except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
        return _execution_job_inventory_error("execution_job_inventory_unavailable")
    jobs: list[dict[str, Any]] = []
    active_job_ids: list[str] = []
    try:
        for record in records:
            item, active = _execution_job_listing_item(record)
            jobs.append(item)
            if active:
                active_job_ids.append(item["job_id"])
    except _ExecutionJobInventoryConflict:
        return _execution_job_inventory_error("execution_job_inventory_inconsistent")
    active_job_ids.sort()
    return {
        "status": "success",
        "jobs": jobs,
        # 单 active 时保留旧字段；多个 active 时置空，避免旧调用方误把一个
        # 作业当成唯一作业，同时由稳定排序的 active_job_ids 提供完整清单。
        "active_job_id": active_job_ids[0] if len(active_job_ids) == 1 else None,
        "active_job_ids": active_job_ids,
        "inventory_complete": True,
        "automatic_fallback": False,
    }


def get_execution_job(request: JobStatusRequest) -> dict[str, Any]:
    try:
        directory, status = _read_status(request.job_id)
    except _ExecutionJobUnavailable:
        return _execution_job_unavailable_response()
    _job_type, registry_record, route_error = _trusted_execution_job_route(
        request.job_id,
        status,
    )
    if route_error is not None:
        return route_error
    try:
        _operation_identity_evidence(directory, status)
    except _OperationIdentityConflict as exc:
        return {
            "status": "failed",
            "valid": False,
            "job_id": request.job_id,
            "job_status": status.get("status"),
            "terminal": False,
            "error": {
                "code": "operation_identity_conflict",
                "message": str(exc),
                "job_id": request.job_id,
            },
            "automatic_fallback": False,
        }
    # registry_record 只在服务端核对 route/identity；不把 job_type 或内部指纹投影给模型。
    if registry_record is not None:
        try:
            _mark_execution_job_registry(_execution_job_registry(), request.job_id, status)
        except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
            pass
    return _execution_job_status_envelope(request.job_id, status)


def _cancel_execution_job_once(request: JobCancelRequest) -> dict[str, Any]:
    try:
        directory, status = _read_status(request.job_id)
    except _ExecutionJobUnavailable:
        return _execution_job_unavailable_response()
    if status.get("status") in TERMINAL_JOB_STATES:
        return _cancel_result(request, status, False, "Job was already terminal; no signal was sent.")
    supervisor_pid = status.get("supervisor_pid")
    queued_result = _record_queued_gaussian_cancellation(request, directory, status)
    if queued_result is not None:
        return queued_result
    pending_result = _cancel_without_published_supervisor(
        request, directory, status, supervisor_pid
    )
    if pending_result is not None:
        return pending_result
    supervisor_identity = _verified_supervisor_identity(status)
    if supervisor_identity is None:
        return _cancel_failure(
            request,
            status,
            "supervisor_identity_unavailable",
            "Supervisor identity is missing, stopped, or no longer matches the recorded process",
        )
    if not _send_verified_supervisor_signal(supervisor_identity, signal.SIGTERM):
        return _cancel_failure(
            request,
            status,
            "supervisor_pidfd_unavailable",
            "当前系统没有稳定 pidfd 信号句柄；未向未经原子证明的 PID/PGID 发信号，请保留资源并人工恢复。",
        )
    return _cancel_result(request, status, True, "Cancellation signal sent; poll get_execution_job until terminal.")


def _cancel_result(
    request: JobCancelRequest,
    status: Mapping[str, Any],
    sent: bool,
    message: str,
) -> dict[str, Any]:
    """构造取消成功或已终态的统一响应。"""

    return {
        "status": "success",
        "job_id": request.job_id,
        "job_status": status.get("status"),
        "cancellation_sent": sent,
        "message": message,
    }


def _cancel_failure(
    request: JobCancelRequest,
    status: Mapping[str, Any],
    code: str,
    message: str,
) -> dict[str, Any]:
    """构造不会发送信号的受控取消失败响应。"""

    return {
        "status": "failed",
        "job_id": request.job_id,
        "job_status": status.get("status"),
        "cancellation_sent": False,
        "error": {"code": code, "message": message},
    }


def _read_cancel_supervisor_spec(directory: Path) -> dict[str, Any] | None:
    """读取排队取消所需的 supervisor spec；损坏 spec 按缺失处理。"""

    path = directory / "supervisor_spec.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _record_queued_gaussian_cancellation(
    request: JobCancelRequest,
    directory: Path,
    status: Mapping[str, Any],
) -> dict[str, Any] | None:
    """为 FIFO 中尚未启动的 Gaussian 作业写取消标记，不猜测 PID。"""

    if status.get("status") != "queued":
        return None
    specification = _read_cancel_supervisor_spec(directory)
    if not isinstance(specification, dict) or not isinstance(
        specification.get("gaussian_queue_ticket"), dict
    ):
        return None
    _ensure_private_marker(directory / "cancel_requested")
    return _cancel_result(request, status, True, "Cancellation recorded; poll get_execution_job until terminal.")


def _cancel_without_published_supervisor(
    request: JobCancelRequest,
    directory: Path,
    status: Mapping[str, Any],
    supervisor_pid: Any,
) -> dict[str, Any] | None:
    """处理 supervisor 尚未公布合法 PID 的窗口，绝不向猜测 PID 发信号。"""

    if isinstance(supervisor_pid, bool):
        return _cancel_failure(
            request,
            status,
            "supervisor_identity_unavailable",
            "Supervisor identity is missing, stopped, or invalid",
        )
    if isinstance(supervisor_pid, int):
        if supervisor_pid > 0:
            return None
        return _cancel_failure(
            request,
            status,
            "supervisor_identity_unavailable",
            "Supervisor identity is missing, stopped, or invalid",
        )
    specification = _read_cancel_supervisor_spec(directory)
    if specification is None:
        return _cancel_failure(
            request,
            status,
            "supervisor_identity_unavailable",
            "Supervisor identity is missing, stopped, or invalid",
        )
    if specification.get("gaussian_queue_ticket") is not None:
        _ensure_private_marker(directory / "cancel_requested")
        return _cancel_result(request, status, True, "Cancellation recorded; poll get_execution_job until terminal.")
    return _cancel_failure(
        request,
        status,
        "supervisor_not_started",
        "Supervisor has not published its PID yet; poll once and retry cancellation.",
    )


def _send_verified_supervisor_signal(
    identity: ProcessIdentity,
    received_signal: signal.Signals,
) -> bool:
    """仅通过 pidfd 向已核对 supervisor 发信号；没有 pidfd 时保守拒绝。"""

    descriptor = _open_pidfd_for_signal(identity.pid)
    if descriptor is None:
        return False
    try:
        current = capture_process_identity(identity.pid)
        if (
            current.start_time_ticks != identity.start_time_ticks
            or current.boot_id != identity.boot_id
        ):
            return False
        return _send_pidfd_signal(descriptor, received_signal)
    except (OSError, ValueError, ProcessLookupError, PermissionError):
        return False
    finally:
        os.close(descriptor)


def _open_pidfd_for_signal(pid: int) -> int | None:
    """打开稳定的进程句柄；标准库缺失时通过 Linux syscall 补齐。"""

    native_open = getattr(os, "pidfd_open", _MISSING_ATTRIBUTE)
    if native_open is None:
        return None
    try:
        if callable(native_open):
            return int(native_open(pid, 0))
        if sys.platform != "linux":
            return None
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = int(
            libc.syscall(
                ctypes.c_long(_PIDFD_OPEN_SYSCALL),
                ctypes.c_int(pid),
                ctypes.c_uint(0),
            )
        )
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return descriptor
    except (OSError, TypeError, ValueError):
        return None


def _send_pidfd_signal(descriptor: int, received_signal: signal.Signals) -> bool:
    """通过标准库或 Linux syscall 向已打开的 pidfd 发信号。"""

    native_send = getattr(signal, "pidfd_send_signal", _MISSING_ATTRIBUTE)
    if native_send is None:
        return False
    try:
        if callable(native_send):
            native_send(descriptor, int(received_signal), None, 0)
            return True
        if sys.platform != "linux":
            return False
        libc = ctypes.CDLL(None, use_errno=True)
        result = int(
            libc.syscall(
                ctypes.c_long(_PIDFD_SEND_SIGNAL_SYSCALL),
                ctypes.c_int(descriptor),
                ctypes.c_int(int(received_signal)),
                ctypes.c_void_p(0),
                ctypes.c_uint(0),
            )
        )
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return True
    except (OSError, TypeError, ValueError):
        return False


def cancel_execution_job(request: JobCancelRequest) -> dict[str, Any]:
    """按 scope 提交键幂等记录取消请求；缺少稳定句柄时不发送进程信号。"""

    missing_identity = _require_scoped_submission(request)
    if missing_identity is not None:
        return missing_identity
    submission_key = _request_submission_key(request)
    if submission_key is None:
        return _cancel_execution_job_once(request)
    store = OperationIdempotencyStore()
    canonical_request = request.model_dump(mode="json")
    canonical_request.pop("submission_key", None)
    fingerprint = calculate_request_fingerprint("cancel_execution_job", canonical_request)
    return dict(
        store.execute_scoped_submission(
            submission_key,
            fingerprint,
            lambda: _cancel_execution_job_once(request),
        )
    )


def _reject_nonfinite(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value at {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")


def _validate_declared_output(path: Path, declaration: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    media_type = str(declaration.get("media_type") or "application/octet-stream")
    try:
        if media_type == "application/json" or path.suffix.casefold() == ".json":
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant {token}")
                ),
            )
            _reject_nonfinite(value)
            if declaration.get("json_schema"):
                from jsonschema import validate as validate_json_schema

                validate_json_schema(value, declaration["json_schema"])
        elif media_type in {"text/csv", "text/tab-separated-values"} or path.suffix.casefold() in {".csv", ".tsv"}:
            delimiter = "\t" if media_type == "text/tab-separated-values" or path.suffix.casefold() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter=delimiter))
            if not rows:
                raise ValueError("table contains no rows")
            for row_index, row in enumerate(rows):
                for column_index, cell in enumerate(row):
                    if cell.strip().casefold() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
                        raise ValueError(
                            f"non-finite table value at row {row_index + 1}, column {column_index + 1}"
                        )
        elif media_type.startswith("image/"):
            prefix = path.read_bytes()[:16]
            signatures = {
                "image/png": b"\x89PNG\r\n\x1a\n",
                "image/jpeg": b"\xff\xd8\xff",
                "image/gif": b"GIF8",
            }
            expected = signatures.get(media_type)
            if expected and not prefix.startswith(expected):
                raise ValueError(f"file does not have a valid {media_type} signature")
    except Exception as exc:
        errors.append(str(exc))
    return errors


def _first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _validate_scientific_output(
    path: Path, declaration: dict[str, Any]
) -> dict[str, Any]:
    semantic = str(declaration.get("semantic_type") or "").casefold()
    checks: list[str] = []
    errors: list[str] = []
    if not any(token in semantic for token in ("hessian", "equationofstate", "equation_of_state", "eos")):
        return {"status": "not_applicable", "checks": [], "errors": []}
    try:
        if path.suffix.casefold() == ".json" or declaration.get("media_type") == "application/json":
            value = json.loads(path.read_text(encoding="utf-8"))
            _reject_nonfinite(value)
        else:
            value = None

        if "hessian" in semantic:
            if not isinstance(value, dict) or not isinstance(value.get("matrix"), list):
                raise ValueError("Hessian output requires a JSON object with a dense matrix")
            matrix = value["matrix"]
            dimension = len(matrix)
            if dimension == 0 or any(not isinstance(row, list) or len(row) != dimension for row in matrix):
                raise ValueError("Hessian matrix must be non-empty, square, and fully populated")
            maximum_asymmetry = 0.0
            for row in range(dimension):
                for column in range(dimension):
                    entry = float(matrix[row][column])
                    if not math.isfinite(entry):
                        raise ValueError(f"Hessian contains non-finite value at [{row}][{column}]")
                    maximum_asymmetry = max(
                        maximum_asymmetry,
                        abs(entry - float(matrix[column][row])),
                    )
            if maximum_asymmetry > 1e-6:
                raise ValueError(
                    f"Hessian is not symmetric within 1e-6; maximum asymmetry={maximum_asymmetry:.6g}"
                )
            checks.extend(
                [
                    f"dense_square_dimension={dimension}",
                    f"finite_value_count={dimension * dimension}",
                    f"maximum_asymmetry={maximum_asymmetry:.6g}",
                ]
            )

        if any(token in semantic for token in ("equationofstate", "equation_of_state", "eos")):
            points: list[dict[str, Any]] = []
            fit: dict[str, Any] | None = None
            if isinstance(value, dict):
                candidate = value.get("points") or value.get("data") or value.get("samples")
                if isinstance(candidate, list):
                    points = [item for item in candidate if isinstance(item, dict)]
                fit_value = value.get("fit")
                fit = fit_value if isinstance(fit_value, dict) else value
            elif value is None:
                delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
                with path.open("r", encoding="utf-8", newline="") as handle:
                    points = list(csv.DictReader(handle, delimiter=delimiter))
            volumes = []
            energies = []
            for index, point in enumerate(points):
                volume = _first_present(
                    point, ("volume", "volume_ang3", "volume_a3", "volume_per_formula_unit")
                )
                energy = _first_present(
                    point, ("energy", "energy_ev", "total_energy", "energy_per_formula_unit")
                )
                if volume is None or energy is None:
                    raise ValueError(
                        f"EOS point {index} lacks a recognized volume or energy field"
                    )
                volume_value = float(volume)
                energy_value = float(energy)
                if not math.isfinite(volume_value) or not math.isfinite(energy_value):
                    raise ValueError(f"EOS point {index} contains a non-finite value")
                volumes.append(volume_value)
                energies.append(energy_value)
            if len(points) < 4 or len(set(volumes)) < 4:
                raise ValueError("EOS validation requires at least four unique volume-energy points")
            checks.extend(
                [
                    f"volume_energy_point_count={len(points)}",
                    f"unique_volume_count={len(set(volumes))}",
                ]
            )
            if "result" in semantic:
                fit_status = str(
                    _first_present(fit or {}, ("status", "fit_status", "convergence_status"))
                    or ""
                ).casefold()
                if fit_status not in {"success", "converged", "valid"}:
                    raise ValueError("EOS result requires an explicit successful fit status")
                residual = _first_present(
                    fit or {}, ("rmse", "residual_rms", "max_abs_residual", "residual")
                )
                if residual is None or not math.isfinite(float(residual)):
                    raise ValueError("EOS result requires one finite residual metric")
                checks.extend(
                    [f"fit_status={fit_status}", f"fit_residual={float(residual):.6g}"]
                )
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "valid" if not errors else "invalid",
        "checks": checks,
        "errors": errors,
        "boundary": (
            "This validates mechanical completeness and finite fit evidence only; scientific "
            "model choice and conclusions remain the Judger's responsibility."
        ),
    }


def _analysis_artifact_manifest(
    directory: Path,
    status: dict[str, Any],
    request_record: dict[str, Any],
) -> dict[str, Any] | None:
    if status.get("job_type") != "programmable_analysis":
        return None
    contract = (request_record.get("metadata") or {}).get("analysis_contract") or {}
    registration_path = directory / "runtime_output_registrations.json"
    registrations = (
        json.loads(registration_path.read_text(encoding="utf-8")).get("outputs", {})
        if registration_path.is_file()
        else {}
    )
    declarations = list(contract.get("outputs") or [])
    artifacts = []
    required_failures = 0
    for declaration in declarations:
        relative = str(declaration["path"])
        path = directory.joinpath(*PurePosixPath(relative).parts)
        required = bool(declaration.get("required", True))
        validation_errors: list[str] = []
        scientific_validation = {"status": "not_checked", "checks": [], "errors": []}
        if path.is_symlink() or not path.is_file():
            validation_status = "missing_required" if required else "missing_optional"
            if required:
                required_failures += 1
        else:
            if path.stat().st_size == 0:
                validation_errors.append("file is empty")
            validation_errors.extend(_validate_declared_output(path, declaration))
            scientific_validation = _validate_scientific_output(path, declaration)
            validation_errors.extend(scientific_validation["errors"])
            validation_status = "valid" if not validation_errors else "invalid"
            if required and validation_errors:
                required_failures += 1
        artifacts.append(
            {
                "name": declaration["name"],
                "path": relative,
                "workspace_path": relative_workspace_path(path),
                "semantic_type": declaration["semantic_type"],
                "media_type": declaration["media_type"],
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() else None,
                "producer": f"programmable_analysis:{status['job_id']}",
                "parent_artifacts": list(declaration.get("parent_artifact_ids") or []),
                "required": required,
                "validation_status": validation_status,
                "validation_errors": validation_errors,
                "runtime_registration": registrations.get(declaration["name"]),
                "scientific_validation": (
                    scientific_validation
                ),
            }
        )
    if not declarations:
        artifact_status = "not_declared"
    elif required_failures:
        artifact_status = "invalid"
    else:
        artifact_status = "valid"
    manifest = {
        "schema_version": 1,
        "job_id": status["job_id"],
        "artifact_status": artifact_status,
        "generated_at": _now(),
        "artifacts": artifacts,
    }
    _atomic_json(directory / "artifact_manifest.json", manifest)
    return manifest


def _frozen_native_software_intent(
    status: dict[str, Any], request_record: dict[str, Any]
) -> tuple[str, str] | None:
    """读取并核对提交时保存的软件与计算意图，不从 stdout 或文件名推断。"""

    metadata_records = []
    for owner in (status, request_record):
        metadata = owner.get("metadata")
        if isinstance(metadata, dict):
            metadata_records.append(metadata)
    values: dict[str, set[str]] = {"software_id": set(), "calculation_intent": set()}
    for metadata in metadata_records:
        for field_name in values:
            value = metadata.get(field_name)
            if value is not None and str(value).strip():
                values[field_name].add(str(value).strip().casefold())
    if any(len(field_values) != 1 for field_values in values.values()):
        return None
    return (
        next(iter(values["software_id"])),
        next(iter(values["calculation_intent"])),
    )


def _native_stdout_semantic_type(
    status: dict[str, Any], request_record: dict[str, Any], relative_path: str
) -> str | None:
    """按冻结的软件和计算意图声明 native stdout 类型，不扫描内容猜测。"""

    if relative_path != "stdout.log":
        return None
    frozen_intent = _frozen_native_software_intent(status, request_record)
    if frozen_intent is None:
        return None
    software_id, calculation_intent = frozen_intent
    if software_id != "gaussian":
        return None
    if calculation_intent == "reaction_path":
        return "GaussianIRCLog"
    if calculation_intent in {"transition_state", "optimization_frequency"}:
        return "GaussianLog"
    return None


def _is_private_scratch_relative_path(relative: str) -> bool:
    """识别作业临时目录中的路径，避免收集本地运行时缓存。"""

    return any(
        part in {".tmp", ".home"}
        or part.startswith(".tmp-")
        or part.startswith(".home-")
        for part in PurePosixPath(relative).parts
    )


def _collect_execution_job_once(request: JobCollectRequest) -> dict[str, Any]:
    try:
        directory, status = _read_status(request.job_id)
    except _ExecutionJobUnavailable:
        return _execution_job_unavailable_response()
    job_type, registry_record, route_error = _trusted_execution_job_route(
        request.job_id,
        status,
    )
    if route_error is not None:
        return route_error
    target_submission_key = (
        registry_record.get("submission_key")
        if registry_record is not None
        else status.get("submission_key")
    )
    requested_submission_key = request.submission_key
    if requested_submission_key is not None and target_submission_key != requested_submission_key:
        return {
            "status": "invalid_request",
            "valid": False,
            "job_id": request.job_id,
            "error": {
                "code": "submission_key_conflict",
                "message": "collect_execution_job 的 submission_key 不属于目标 job",
            },
            "automatic_fallback": False,
        }
    if job_type == "transition_state_path":
        return _transition_state_path_facade_collect(request.job_id)
    if job_type == "action_execution":
        return _action_job_facade_collect(
            request.job_id,
            submission_key=request.submission_key,
        )
    if status.get("status") not in TERMINAL_JOB_STATES:
        return _nonterminal_collection_response(request, directory, status)
    request_record = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    files = _collect_execution_output_files(request, directory, status, request_record)
    artifact_manifest = _analysis_artifact_manifest(directory, status, request_record)
    axes = _execution_status_axes(
        directory,
        status,
        analysis_artifact_status=(
            artifact_manifest["artifact_status"] if artifact_manifest else None
        ),
    )
    failure_diagnostic = _execution_failure_diagnostic(directory, status)
    manifest = _collection_manifest(
        request,
        status,
        request_record,
        files,
        axes,
        artifact_manifest,
        failure_diagnostic,
    )
    collection_path = directory / "collection.json"
    _atomic_json(collection_path, manifest)
    if registry_record is not None:
        try:
            _execution_job_registry().mark(request.job_id, phase="collected")
        except (ExecutionJobRegistryError, OSError, RuntimeError, ValueError):
            return {
                "status": "failed",
                "valid": False,
                "job_id": request.job_id,
                "job_status": status.get("status"),
                "terminal": True,
                "error": {
                    "code": "execution_job_registry_unavailable",
                    "message": "collection manifest 已落盘但 registry 未能记录 collected，可按原 job_id 重试",
                },
                "automatic_fallback": False,
            }
    return {
        "status": "success",
        "ready": True,
        "job": status,
        "collection_manifest": relative_workspace_path(collection_path),
        **manifest,
        "artifact_manifest": (
            relative_workspace_path(directory / "artifact_manifest.json")
            if artifact_manifest
            else None
        ),
        "next_step": (
            "Inspect the files and call declare_scientific_artifact for each scientifically "
            "meaningful output that should enter an Action or later job with explicit lineage."
        ),
    }


def _nonterminal_collection_response(
    request: JobCollectRequest,
    directory: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    """返回非终态作业的可轮询响应，不写 collection 文件。"""

    axes = _execution_status_axes(directory, dict(status))
    return {
        "status": "success",
        "job_id": request.job_id,
        "job_status": status.get("status"),
        "ready": False,
        "outputs": [],
        **{key: value for key, value in axes.items() if key != "details"},
        "execution_status_details": axes["details"],
        "message": "Job is not terminal; poll get_execution_job before collecting outputs.",
    }


def _collect_execution_output_files(
    request: JobCollectRequest,
    directory: Path,
    status: Mapping[str, Any],
    request_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """筛选终态作业的公开输出，排除私有 scratch 和内部控制文件。"""

    staged_targets = {
        str(item["target_path"]) for item in request_record.get("staged_inputs") or []
    }
    internal = {
        "analysis_contract.json",
        "artifact_manifest.json",
        "failure_diagnostic.json",
        "request.json",
        "researchchem_job.py",
        "runtime_output_registrations.json",
        "status.json",
        "supervisor_spec.json",
        "collection.json",
    }
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        relative = str(path.relative_to(directory))
        if _output_file_is_hidden(path, relative, internal, request, staged_targets):
            continue
        files.append(_execution_output_record(path, relative, status, request_record))
        if len(files) >= request.max_files:
            break
    return files


def _output_file_is_hidden(
    path: Path,
    relative: str,
    internal: set[str],
    request: JobCollectRequest,
    staged_targets: set[str],
) -> bool:
    """判断一个作业路径是否不能出现在公开 collection 中。"""

    return (
        path.is_symlink()
        or not path.is_file()
        or relative in internal
        or path.name == "researchchem_job.py"
        or _is_private_scratch_relative_path(relative)
        or (not request.include_inputs and relative in staged_targets)
    )


def _execution_output_record(
    path: Path,
    relative: str,
    status: Mapping[str, Any],
    request_record: Mapping[str, Any],
) -> dict[str, Any]:
    """为一个公开文件生成相对路径、大小、摘要和语义类型。"""

    record: dict[str, Any] = {
        "path": relative_workspace_path(path),
        "job_relative_path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "kind": "log" if relative in {"stdout.log", "stderr.log"} else "output",
    }
    semantic_type = _native_stdout_semantic_type(dict(status), dict(request_record), relative)
    if semantic_type is not None:
        record["semantic_type"] = semantic_type
    return record


def _collection_manifest(
    request: JobCollectRequest,
    status: Mapping[str, Any],
    request_record: Mapping[str, Any],
    files: list[dict[str, Any]],
    axes: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any] | None,
    failure_diagnostic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """组合 collection.json 的稳定字段和科学状态轴。"""

    job_submission_key = status.get("submission_key") or request_record.get("submission_key")
    manifest = {
        "schema_version": 1,
        "job_id": request.job_id,
        "submission_key": request.submission_key or job_submission_key,
        "job_submission_key": job_submission_key,
        "job_status": status.get("status"),
        "collected_at": _now(),
        "include_inputs": request.include_inputs,
        "truncated": len(files) >= request.max_files,
        "outputs": files,
        **{key: value for key, value in axes.items() if key != "details"},
        "execution_status_details": axes["details"],
        "declared_artifacts": artifact_manifest["artifacts"] if artifact_manifest else [],
        "failure_diagnostic": failure_diagnostic,
    }
    if isinstance(status.get("error"), Mapping):
        # submit/get/collect 三个入口都返回同一份可操作失败事实；路径只保留脱敏后的
        # 相对信息，避免模型必须回到私有 status.json 才能看到原始错误。
        manifest["error"] = _sanitize_scoped_error_value(dict(status["error"]))
    for field in ("warning", "provenance"):
        if isinstance(status.get(field), Mapping):
            manifest[field] = deepcopy(dict(status[field]))
    return manifest


def collect_execution_job(request: JobCollectRequest) -> dict[str, Any]:
    """按已有 job_id 重复读取或生成 job manifest，展示参数变化不改变作业身份。

    输入是已提交作业的 job_id、正确 submission_key 和仅影响展示的收集参数；实际动作
    先执行 scope、作业身份、提交键和下层 registry 核对，再读取或原子更新收集结果。该
    入口不把可重复读取的响应写入按完整请求指纹缓存，使暂时缺少 worker 结果等可修复
    错误能够按原 job_id 重试；返回错误时仍保留错误状态，不释放上层作业门禁。
    """

    missing_identity = _require_scoped_submission(request)
    if missing_identity is not None:
        return _preserve_requested_job_id_in_collection_error(
            missing_identity,
            request.job_id,
        )
    response = _collect_execution_job_once(request)
    return _preserve_requested_job_id_in_collection_error(response, request.job_id)


def _resolve_scientific_artifact_source(
    request: ArtifactDeclarationRequest,
) -> Path:
    """在声明幂等收据落盘前解析并校验科学文件来源。"""

    path = resolve_workspace_public_path(
        request.path,
        must_exist=True,
        purpose="Artifact source",
        allow_execution_job_outputs=True,
        allow_action_artifacts=True,
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError("Artifact source must be a regular non-symlink file")
    return path


def _declare_scientific_artifact_once(
    request: ArtifactDeclarationRequest,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """执行一次 Artifact 声明；幂等收据由外层入口负责。"""

    path = (
        source_path
        if source_path is not None
        else _resolve_scientific_artifact_source(request)
    )
    producer = f"{request.producer_layer}:{request.producer_id}"
    reference = ArtifactStore().register_file(
        path,
        semantic_type=request.semantic_type,
        media_type=request.media_type,
        producer_action=producer,
        producer_backend=(
            request.producer_id if request.producer_layer == "native_software" else None
        ),
        parent_artifact_ids=request.parent_artifact_ids,
    )
    return {
        "status": "success",
        "artifact": reference.model_dump(mode="json"),
        "lineage_note": (
            "This declaration records semantics and parent ids supplied by the Agent; it does not "
            "infer or alter scientific meaning."
        ),
    }


def declare_scientific_artifact(request: ArtifactDeclarationRequest) -> dict[str, Any]:
    """声明科学文件，并按内部身份回放同一份 ArtifactRef。"""

    missing_identity = _require_scoped_submission(request)
    if missing_identity is not None:
        return missing_identity
    operation_id = request.operation_id
    submission_key = _request_submission_key(request)
    if submission_key is not None:
        operation_id = scoped_submission_operation_id(submission_key)
    if operation_id is None:
        return _declare_scientific_artifact_once(request)
    store = OperationIdempotencyStore()
    source_path: Path | None = None
    if store.read_operation_receipt(operation_id) is None:
        source_path = _resolve_scientific_artifact_source(request)
    canonical_request = request.model_dump(mode="json")
    canonical_request.pop("submission_key", None)
    fingerprint = calculate_request_fingerprint("declare_scientific_artifact", canonical_request)
    if submission_key is not None:
        return dict(
            store.execute_scoped_submission(
                submission_key,
                fingerprint,
                lambda: _declare_scientific_artifact_once(
                    request,
                    source_path=source_path,
                ),
            )
        )
    result = store.execute_idempotent_operation(
        operation_id,
        fingerprint,
        lambda: _declare_scientific_artifact_once(
            request,
            source_path=source_path,
        ),
    )
    return dict(result)


__all__ = [
    "cancel_execution_job",
    "collect_execution_job",
    "declare_scientific_artifact",
    "get_execution_job",
    "read_workspace_text",
    "submit_analysis_program",
    "get_execution_resources",
    "inspect_analysis_inputs",
    "read_scientific_artifact_text",
    "submit_native_job",
    "validate_analysis_program",
    "validate_native_job",
    "verify_execution_references",
    "write_workspace_text",
]
