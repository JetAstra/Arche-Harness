"""统一 predefined Action 的持久异步 job 核心。

该模块是 Investigation Action 的唯一通用入口：所有 Action 先通过冻结的
``ActionJobAdapter`` 注册表和现有 Action 合同校验，再由同一个 scope 状态文件、
``ExecutionJobRegistry``、detached supervisor 以及 canonical ActionResult 收集链处理。
科学 backend 仍由原有 ``execute_local`` 实现，异步层不复制科学算法，也不接受任意
Python 命令、任意 runtime 或模型资源选择。

``search_transition_state_path`` 的历史 TS 专用 adapter 继续保留；它的专用 worker
不在本模块重写。其余 Catalog Action（包括 Gaussian scan 和内部 Action）都由本模块
使用同一通用 worker 覆盖。
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from pydantic import ConfigDict, Field, field_validator

from minichem_toolbox.artifacts import ArtifactStore, read_file_snapshot
from minichem_toolbox.catalog import action_specs, active_catalog_hash, backend_specs
from minichem_toolbox.idempotency import (
    calculate_request_fingerprint,
    scoped_submission_operation_id,
    scoped_submission_request_error,
    validate_submission_key,
)
from minichem_toolbox.models import ActionRequest, ActionResult, ArtifactRef, ResourceLimits
from minichem_toolbox.resource_budget import fixed_job_resource_limits
from minichem_toolbox.scope import ScopeContext, current_scope
from minichem_toolbox.runtime import runtime_python
from minichem_toolbox.service import (
    _ActionExecutionPlan,
    _build_action_result,
    _collect_action_result_artifacts,
    _normalize_action_worker_result,
    _prepare_action_execution,
    _sanitize_public_diagnostic_value,
    _validate_action_backend_health,
)
from minichem_toolbox.timeout_policy import timeout_policy_record, timeout_seconds_for

from . import discovery_tools
from .discovery_models import ActionInspectRequest, ProgressiveActionRequest
from .execution_job_registry import (
    EXECUTION_JOB_ID_PATTERN,
    ExecutionJobRegistry,
    ExecutionJobRegistryConflict,
    ExecutionJobRegistryError,
)
from .execution_models import StagedInput
from .open_execution import TERMINAL_JOB_STATES, _atomic_json, _input_snapshots, _start_job
from .result_transport import compact_action_result
from .tracing import execute_traced
from .workspace import (
    ensure_private_directory,
    relative_workspace_path,
    resolve_workspace_public_path,
)


ACTION_JOB_TYPE = "action_execution"
JOB_TYPE = ACTION_JOB_TYPE
WORKER_MODULE = "minichem_mcp_tools.action_jobs"
ACTION_JOB_RESULT_FILENAME = "action_result.json"
RESULT_FILENAME = ACTION_JOB_RESULT_FILENAME
RAW_RESULT_FILENAME = "action_worker_result.json"
STATE_DIRECTORY_NAME = "action_jobs"
STATE_FILENAME = "state.json"
JOB_ID_PATTERN = EXECUTION_JOB_ID_PATTERN
TERMINAL_STATES = frozenset(TERMINAL_JOB_STATES)
STATE_SCHEMA_VERSION = 1
TS_ACTION_ID = "search_transition_state_path"
TS_BACKEND_ID = "transition_state_path_search"
_MODEL_HIDDEN_FIELD_NAMES = frozenset(
    {
        "resource_limits",
        "resource_constraints",
        "memory_mb",
        "cpu_cores",
        "gpu_count",
        "minimum_gpu_count",
    }
)
_MODEL_HIDDEN_ROUTE_TERMS = {
    "execute_action": "submit_action_job",
    "submit_action_batch": "submit_action_job",
    "submit_transition_state_path_job": "submit_action_job",
    "get_transition_state_path_job": "get_execution_job",
    "collect_transition_state_path_job": "collect_execution_job",
    "长任务": "作业",
    "短任务": "作业",
    "long-running": "calculation",
    "long task": "calculation",
    "short task": "calculation",
}
_MODEL_HIDDEN_ROUTE_KEY_TERMS = (
    "execute_action",
    "submit_action_batch",
    "submit_transition_state_path_job",
    "get_transition_state_path_job",
    "collect_transition_state_path_job",
)
_MODEL_HIDDEN_RESOURCE_KEY_TERMS = (
    "resource_limits",
    "resource_constraints",
    "memory_mb",
    "cpu_cores",
    "gpu_count",
)


@dataclass(frozen=True, slots=True)
class ActionJobAdapter:
    """一个由 Catalog 闭合生成的 Action/backend 异步执行适配器。"""

    action_id: str
    backend_id: str
    runtime: str
    software_id: str | None
    execution_class: str

    @property
    def key(self) -> tuple[str, str]:
        """返回稳定的 Action/backend 查找键。"""

        return self.action_id, self.backend_id


def _build_action_job_adapters() -> dict[tuple[str, str], ActionJobAdapter]:
    """从冻结 Catalog 构造闭合 adapter 注册表，不接受模型提供 Python 路径。"""

    actions = action_specs()
    backends = backend_specs()
    result: dict[tuple[str, str], ActionJobAdapter] = {}
    for action_id, specification in actions.items():
        # TS pipeline 的原子 worker 有额外的多阶段合同，继续由其专用 adapter 处理；
        # 通用 submit 会明确返回专用工具，而不是把 TS 误路由到普通 backend。
        if action_id == TS_ACTION_ID:
            continue
        for backend_id in specification.backend_ids:
            backend = backends.get(backend_id)
            if backend is None:
                continue
            result[(action_id, backend_id)] = ActionJobAdapter(
                action_id=action_id,
                backend_id=backend_id,
                runtime=backend.runtime,
                software_id="gaussian" if backend_id == "gaussian" else None,
                execution_class=specification.execution_class,
            )
    return result


# 这是服务启动时由受控 Catalog 生成的闭合映射；外部请求只能命中其中的键。
ACTION_JOB_ADAPTERS = _build_action_job_adapters()

TOOL_NAMES = ("submit_action_job",)
TOOL_DESCRIPTIONS = {
    "submit_action_job": (
        "提交一个 Catalog Action 的持久异步计算。输入必须是 inspect_action 返回的完整"
        " action_id/backend_id/request 合同；成功时立即返回 job_id、operation_id、"
        "submission_key 和最小状态。即使 submit 回执已经 terminal=true/failed，也必须先"
        "对同一 job_id 调用 get_execution_job，随后才调用 collect_execution_job 收集"
        "canonical ActionResult。当前 scope 同时只允许一个"
        "未 collect 的 Action job；资源、runtime 和 worker 命令均由服务端固定。"
    ),
}


class ActionJobRequest(ActionRequest):
    """统一 Action job 的公开请求，科学字段沿用 ActionRequest。"""

    model_config = ConfigDict(extra="allow")

    action_id: str = Field(description="Catalog 中精确的 Action id。")

    @field_validator("action_id")
    @classmethod
    def validate_action_id(cls, value: str) -> str:
        """只接受 lower_snake_case 的 Catalog Action 标识。"""

        normalized = value.strip()
        if re.fullmatch(r"[a-z][a-z0-9_]*", normalized) is None:
            raise ValueError("action_id 必须是有效的 Catalog Action 标识")
        return normalized


def _now() -> str:
    """返回 UTC 持久化时间。"""

    return datetime.now(timezone.utc).isoformat()


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    """构造不暴露宿主路径和内部异常的错误 envelope。"""

    result: dict[str, Any] = {
        "status": "invalid_request",
        "valid": False,
        "error": {"code": code, "message": message},
        "automatic_fallback": False,
    }
    result["error"].update(details)
    return result


def _adapter_for(action_id: str, backend_id: str | None) -> ActionJobAdapter | dict[str, Any]:
    """只从闭合 adapter registry 解析 Action/backend，不执行任意路径。"""

    if action_id == TS_ACTION_ID:
        return _error(
            "specialized_action_adapter_required",
            "search_transition_state_path 必须使用专用 TS submit_transition_state_path_job adapter",
            required_submit_tool="submit_transition_state_path_job",
        )
    selected_backend = backend_id.strip() if isinstance(backend_id, str) else ""
    if not selected_backend:
        specification = action_specs().get(action_id)
        if (
            specification is not None
            and specification.selection_policy == "internal_deterministic"
            and len(specification.backend_ids) == 1
        ):
            selected_backend = specification.backend_ids[0]
    if not selected_backend:
        return _error("missing_backend_id", "异步 Action 必须显式提供 backend_id")
    adapter = ACTION_JOB_ADAPTERS.get((action_id, selected_backend))
    if adapter is None:
        return _error(
            "unsupported_action_backend",
            "Action/backend 不在服务端闭合异步 adapter registry 中",
            action_id=action_id,
            backend_id=backend_id,
        )
    return adapter


def _scrub_model_visible_action_projection(value: Any) -> Any:
    """清理 inspect_action 给 Investigation 模型的公开投影。

    Catalog 的内部完整描述仍可保留资源和同步入口信息；这里是模型真正能看到的
    边界，因此递归删除固定资源字段，并把旧同步/批量/TS 专用名称改写为统一的
    submit_action_job、get_execution_job、collect_execution_job。该函数不修改 Catalog
    原对象，只返回一个新的 JSON 值。
    """

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            if (
                raw_key in _MODEL_HIDDEN_FIELD_NAMES
                or any(term in raw_key for term in _MODEL_HIDDEN_ROUTE_KEY_TERMS)
                or any(term in raw_key for term in _MODEL_HIDDEN_RESOURCE_KEY_TERMS)
            ):
                # execute_action_request_template 等旧合同同时携带同步入口和资源
                # 选择字段；整体移除比改名更安全，也不会与新 generic 字段碰撞。
                continue
            projected[raw_key] = _scrub_model_visible_action_projection(item)
        return projected
    if isinstance(value, list):
        return [
            _scrub_model_visible_action_projection(item)
            for item in value
            if not (isinstance(item, str) and item in _MODEL_HIDDEN_FIELD_NAMES)
        ]
    if isinstance(value, tuple):
        return [_scrub_model_visible_action_projection(item) for item in value]
    if isinstance(value, str):
        result = value
        for old, replacement in _MODEL_HIDDEN_ROUTE_TERMS.items():
            result = result.replace(old, replacement)
        return result
    return value


def _scope_error(request: Any) -> dict[str, Any] | None:
    """检查 capability scope、服务端 operation 身份和 submission_key。"""

    if current_scope() is None:
        return _error("scope_required", "Action job 必须在已认证 Investigation scope 中提交")
    identity_error = scoped_submission_request_error(request)
    if identity_error is not None:
        return _error(str(identity_error["code"]), str(identity_error["message"]))
    try:
        validate_submission_key(str(request.submission_key))
    except ValueError as exc:
        return _error("invalid_submission_key", str(exc))
    return None


def _scope_state_root(workspace: Path) -> Path:
    """返回 scope 私有 Action job 状态目录并收紧权限。"""

    return ensure_private_directory(
        workspace / "_tool_artifacts" / STATE_DIRECTORY_NAME,
        create=True,
    )


@contextmanager
def _state_lock(workspace: Path) -> Iterator[None]:
    """串行化所有通用 Action job 状态修改，确保 scope 级单 active gate。"""

    root = _scope_state_root(workspace)
    descriptor = os.open(
        root / "state.lock",
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """读取普通 JSON 控制文件；缺失、链接或损坏均返回 None。"""

    try:
        if path.is_symlink() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _job_directory(workspace: Path, job_id: str) -> Path | None:
    """解析 execution job 目录并拒绝路径中的符号链接。"""

    if JOB_ID_PATTERN.fullmatch(str(job_id)) is None:
        return None
    candidate = workspace / "outputs" / "execution_jobs" / job_id
    try:
        relative = candidate.relative_to(workspace)
    except ValueError:
        return None
    cursor = workspace
    for part in relative.parts:
        cursor /= part
        try:
            if stat.S_ISLNK(os.lstat(cursor).st_mode):
                return None
        except FileNotFoundError:
            break
        except OSError:
            return None
    return candidate


def _state_path(workspace: Path) -> Path:
    """返回 scope Action job 状态收据路径。"""

    return _scope_state_root(workspace) / STATE_FILENAME


def _empty_state(scope: ScopeContext) -> dict[str, Any]:
    """构造尚未登记通用 Action job 的 scope 状态。"""

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "job_type": ACTION_JOB_TYPE,
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
        "active_submission_key": None,
        "active_job_id": None,
        "jobs": {},
        "updated_at": _now(),
    }


def _validate_state(
    value: Mapping[str, Any] | None,
    *,
    scope: ScopeContext,
) -> dict[str, Any] | None:
    """核对持久状态身份和单 active gate，损坏时返回 None。"""

    if value is None:
        return None
    if (
        value.get("schema_version") != STATE_SCHEMA_VERSION
        or value.get("job_type") != ACTION_JOB_TYPE
        or value.get("scope_id") != scope.scope_id
        or value.get("scope_workspace") != str(scope.workspace)
    ):
        return None
    jobs = value.get("jobs")
    if not isinstance(jobs, Mapping):
        return None
    normalized_jobs: dict[str, dict[str, Any]] = {}
    phases = {"submitting", "accepted", "failed", "unknown", "collected"}
    for key, raw in jobs.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            return None
        try:
            validate_submission_key(key)
        except ValueError:
            return None
        if (
            raw.get("submission_key") != key
            or raw.get("scope_id") != scope.scope_id
            or raw.get("scope_workspace") != str(scope.workspace)
            or JOB_ID_PATTERN.fullmatch(str(raw.get("job_id") or "")) is None
            or not isinstance(raw.get("operation_id"), str)
            or not isinstance(raw.get("request_fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(raw.get("request_fingerprint"))) is None
            or raw.get("phase") not in phases
            or not isinstance(raw.get("action_id"), str)
            or not isinstance(raw.get("backend_id"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(raw.get("catalog_hash") or "")) is None
        ):
            return None
        normalized_jobs[key] = dict(raw)
    active_key = value.get("active_submission_key")
    active_job_id = value.get("active_job_id")
    if active_key is None:
        if active_job_id is not None:
            return None
    elif (
        not isinstance(active_key, str)
        or active_key not in normalized_jobs
        or normalized_jobs[active_key].get("job_id") != active_job_id
    ):
        return None
    normalized = dict(value)
    normalized["jobs"] = normalized_jobs
    return normalized


def _load_state(scope: ScopeContext) -> tuple[dict[str, Any] | None, bool]:
    """读取状态并区分“尚未创建”和“已损坏”。"""

    path = _state_path(scope.workspace)
    raw = _read_json_file(path)
    if raw is None:
        return (None, True) if not path.exists() else (None, False)
    state = _validate_state(raw, scope=scope)
    return state, state is not None


def _save_state(state: dict[str, Any], workspace: Path) -> None:
    """原子保存通用 Action job 状态。"""

    state["updated_at"] = _now()
    _atomic_json(_state_path(workspace), state)


def _read_job_status(scope: ScopeContext, job_id: str) -> dict[str, Any] | None:
    """读取并核对通用 job 的 scope 和 job_type。"""

    directory = _job_directory(scope.workspace, job_id)
    if directory is None:
        return None
    status = _read_json_file(directory / "status.json")
    if status is None:
        return None
    if (
        status.get("job_id") != job_id
        or status.get("scope_id") != scope.scope_id
        or status.get("scope_workspace") != str(scope.workspace)
        or status.get("job_type") != ACTION_JOB_TYPE
    ):
        return None
    return status


def _find_job_record(state: Mapping[str, Any], job_id: str) -> dict[str, Any] | None:
    """按 job_id 查找当前 scope 唯一的状态记录。"""

    jobs = state.get("jobs")
    if not isinstance(jobs, Mapping):
        return None
    matches = [
        dict(item)
        for item in jobs.values()
        if isinstance(item, Mapping) and item.get("job_id") == job_id
    ]
    return matches[0] if len(matches) == 1 else None


def _minimal_identity(
    record: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    *,
    reused: bool,
) -> dict[str, Any]:
    """构造 submit/get 的最小可公开身份，不返回科学结果或内部路径。"""

    job_status = status.get("status") if status is not None else None
    response: dict[str, Any] = {
        "status": "success",
        "job_id": record["job_id"],
        "job": {"job_id": record["job_id"], "status": job_status},
        "operation_id": record["operation_id"],
        "submission_key": record["submission_key"],
        "action_id": record["action_id"],
        "backend_id": record["backend_id"],
        "job_type": ACTION_JOB_TYPE,
        "runtime": record.get("runtime"),
        "software_id": record.get("software_id"),
        "job_status": job_status,
        "terminal": job_status in TERMINAL_STATES,
        "phase": record.get("phase", "accepted"),
        "idempotent_reused": reused,
        "automatic_fallback": False,
    }
    if status is not None and isinstance(status.get("error"), Mapping):
        response["error"] = deepcopy(dict(status["error"]))
    return response


def _submitted_terminal_failure(
    submitted: Mapping[str, Any],
) -> dict[str, Any] | None:
    """提取 supervisor 在启动边界明确返回的终态失败。"""

    status = str(submitted.get("status") or "").casefold()
    if status not in {"invalid_request", "unavailable", "failed", "error"}:
        return None
    raw_error = submitted.get("error")
    if isinstance(raw_error, Mapping):
        return deepcopy(dict(raw_error))
    return {
        "code": "execution_failed",
        "message": "Action job supervisor 未启动已提交的计算",
    }


def _persist_terminal_failed_job(
    *,
    scope: ScopeContext,
    registry: ExecutionJobRegistry,
    record: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    resource_limits: Mapping[str, Any],
    error: Mapping[str, Any],
) -> dict[str, Any] | None:
    """把确定性启动失败保存为可查询、可 collect 的终态作业。

    输入必须是已经由中央 registry 预留的 Action job 身份。函数在该 job 的私有
    目录写入最小 ``status.json``，随后把 registry phase 标为 ``failed``；已有
    冲突身份或成功终态时返回 ``None``，由上层保留 uncertain 身份并禁止重提。
    """

    job_id = str(record.get("job_id") or "")
    directory = _job_directory(scope.workspace, job_id)
    if directory is None:
        return None
    try:
        ensure_private_directory(directory, create=True)
    except (OSError, ValueError):
        return None
    status_path = directory / "status.json"
    existing = _read_json_file(status_path)
    identity = {
        "job_id": job_id,
        "scope_id": scope.scope_id,
        "scope_workspace": str(scope.workspace),
    }
    if existing is not None:
        if any(
            existing.get(name) is not None and existing.get(name) != value
            for name, value in identity.items()
        ):
            return None
        if str(existing.get("status") or "").casefold() in {
            "success",
            "timeout",
            "cancelled",
        }:
            return None
    status: dict[str, Any] = dict(existing or {})
    status.update(
        {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": ACTION_JOB_TYPE,
            "runtime": record.get("runtime"),
            "status": "failed",
            "scope_id": scope.scope_id,
            "scope_workspace": str(scope.workspace),
            "operation_id": record.get("operation_id"),
            "operation_fingerprint": record.get("operation_fingerprint"),
            "caller_request_fingerprint": record.get("request_fingerprint"),
            "submission_key": record.get("submission_key"),
            "resource_limits": deepcopy(dict(resource_limits)),
            "resource_allocation": status.get("resource_allocation", {}),
            "metadata": status.get(
                "metadata",
                {
                    "action_id": record.get("action_id"),
                    "backend_id": record.get("backend_id"),
                    "catalog_hash": record.get("catalog_hash"),
                    "label": "action_job_async",
                    "calculation_intent": record.get("action_id"),
                    "action_job_request": deepcopy(dict(request_payload)),
                },
            ),
            "submitted_at": status.get("submitted_at", record.get("submitted_at", _now())),
            "finished_at": _now(),
            "return_code": None,
            "error": deepcopy(dict(error)),
            "execution_mode": "detached_action_job",
        }
    )
    try:
        _atomic_json(status_path, status)
        registry.mark(job_id, phase="failed")
    except (ExecutionJobRegistryError, OSError, TypeError, ValueError):
        return None
    return status


def _accept_terminal_failed_action_job(
    *,
    state: dict[str, Any],
    workspace: Path,
    scope: ScopeContext,
    registry: ExecutionJobRegistry,
    record: dict[str, Any],
    request_payload: Mapping[str, Any],
    resource_limits: Mapping[str, Any],
    error: Mapping[str, Any],
) -> dict[str, Any] | None:
    """把已取得 job_id 的确定性 Action 失败写成可查询终态。"""

    status = _persist_terminal_failed_job(
        scope=scope,
        registry=registry,
        record=record,
        request_payload=request_payload,
        resource_limits=resource_limits,
        error=error,
    )
    if status is None:
        return None
    record["phase"] = "failed"
    record["failed_at"] = _now()
    state["jobs"][record["submission_key"]] = record
    _save_state(state, workspace)
    return _minimal_identity(record, status, reused=False)


def _registry_reservation(
    scope: ScopeContext,
    record: Mapping[str, Any],
    *,
    preferred_job_id: str | None,
) -> tuple[ExecutionJobRegistry, Any] | dict[str, Any]:
    """在统一 ExecutionJobRegistry 中核对 operation/submission/job 身份。"""

    try:
        registry = ExecutionJobRegistry(workspace=scope.workspace, scope_id=scope.scope_id)
        reservation = registry.reserve(
            job_type=ACTION_JOB_TYPE,
            operation_id=str(record.get("operation_id") or ""),
            operation_fingerprint=str(record.get("operation_fingerprint") or ""),
            submission_key=str(record.get("submission_key") or ""),
            caller_request_fingerprint=str(record.get("request_fingerprint") or ""),
            preferred_job_id=preferred_job_id,
        )
    except ExecutionJobRegistryConflict as exc:
        conflict = getattr(exc, "record", None)
        return _error(
            "action_job_submission_conflict",
            "通用 Action job 身份与统一 registry 不一致，已停止恢复",
            job_id=(
                (conflict or {}).get("job_id")
                if isinstance(conflict, Mapping)
                else record.get("job_id")
            ),
        )
    except (ExecutionJobRegistryError, ValueError):
        return _error(
            "action_job_state_unavailable",
            "通用 Action job 身份无法在统一 registry 中核对",
            job_id=record.get("job_id"),
        )
    if preferred_job_id is not None and reservation.job_id != preferred_job_id:
        return _error(
            "action_job_identity_conflict",
            "Action job state 与统一 registry 的 job_id 不一致",
            job_id=preferred_job_id,
        )
    return registry, reservation


def _submission_plan(
    request: ActionJobRequest,
    adapter: ActionJobAdapter,
) -> tuple[_ActionExecutionPlan, ActionJobRequest] | dict[str, Any]:
    """运行完整 Action 合同校验并固定服务资源。"""

    plan = _prepare_action_execution(request.action_id, request, resources_bound=False)
    if isinstance(plan, dict):
        return plan
    if plan.backend_id != adapter.backend_id:
        return _error(
            "action_job_identity_conflict",
            "Action 合同选择的 backend 与 adapter 不一致",
            action_id=request.action_id,
            backend_id=plan.backend_id,
        )
    try:
        normalized = ActionJobRequest.model_validate(
            {"action_id": request.action_id, **plan.request.model_dump(mode="json")}
        )
    except Exception as exc:
        return _error(
            "invalid_request",
            "Action job 请求无法恢复为固定 adapter 合同",
            exception_type=type(exc).__name__,
        )
    return plan, normalized


def _stageable_workspace_file(value: str) -> tuple[str, str] | None:
    """把公开 workspace 文件解析为安全的 source/target 相对路径。"""

    try:
        path = resolve_workspace_public_path(
            value,
            must_exist=True,
            purpose="Action job input",
            allow_execution_job_outputs=True,
            allow_action_artifacts=True,
        )
    except (FileNotFoundError, OSError, ValueError):
        return None
    if path.is_symlink() or not path.is_file():
        return None
    relative = relative_workspace_path(path)
    return relative, relative


def _build_staged_worker_request(
    plan: _ActionExecutionPlan,
    request: ActionJobRequest,
) -> tuple[dict[str, Any], list[StagedInput], list[dict[str, Any]]]:
    """把父 scope 的 ArtifactRef/文件复制计划投影为 job 内可读请求。

    输入是已经由 Action 合同规范化的父 scope 请求。函数只把显式 ArtifactRef 或
    已存在的公开 workspace 文件重写为同名 staged target；内联科学值保持不变。
    返回 worker 请求、复制清单和提交前文件快照，任意 target 冲突时立即失败。
    """

    staged_by_target: dict[str, StagedInput] = {}
    expected_digests: dict[str, str] = {}

    def register_path(value: str) -> str | None:
        staged = _stageable_workspace_file(value)
        if staged is None:
            return None
        source_path, target_path = staged
        existing = staged_by_target.get(target_path)
        if existing is not None and existing.source_path != source_path:
            raise ValueError("Action job staged input target 绑定了不同源文件")
        staged_by_target[target_path] = StagedInput(
            source_path=source_path,
            target_path=target_path,
        )
        return target_path

    def rewrite(value: Any) -> Any:
        if isinstance(value, Mapping):
            required = {"artifact_id", "semantic_type", "media_type", "sha256", "path"}
            if required <= set(value):
                reference = ArtifactRef.model_validate(value)
                scope = current_scope()
                if scope is None:
                    raise ValueError("ArtifactRef 缺少已认证 scope")
                store = ArtifactStore(root=scope.workspace, create=False)
                # ArtifactStore 在父 scope 核对 manifest、路径和摘要。JSON 对象直接
                # 固化到 worker 请求，避免复制私有索引；文件类 ArtifactRef 则复制
                # 其公开文件，并把 worker 输入改为 staged 相对路径。
                if reference.media_type == "application/json":
                    return store.load(reference)
                canonical = store.find(reference.artifact_id)
                if canonical.model_dump(mode="json") != reference.model_dump(mode="json"):
                    raise ValueError("ArtifactRef manifest 身份不一致")
                target = register_path(reference.path)
                if target is None:
                    raise ValueError("ArtifactRef path 无法 staged 到 Action job")
                expected_digests[reference.path] = reference.sha256
                return target
            return {str(key): rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, tuple):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return register_path(value) or value
        return value

    worker_request = plan.request.model_copy(
        update={"inputs": rewrite(plan.request.inputs)}
    ).model_dump(mode="json")
    worker_request["action_id"] = request.action_id
    staged_inputs = [staged_by_target[key] for key in sorted(staged_by_target)]
    snapshots = _input_snapshots(staged_inputs)
    for snapshot in snapshots:
        expected = expected_digests.get(str(snapshot.get("source_path") or ""))
        if expected is not None and snapshot.get("sha256") != expected:
            raise ValueError("ArtifactRef 文件摘要与 manifest 不一致")
    return worker_request, staged_inputs, snapshots


def _request_fingerprint(request: ActionJobRequest) -> str:
    """计算不含调用身份且已固定资源的稳定请求指纹。"""

    value = request.model_dump(mode="json")
    value.pop("submission_key", None)
    value.pop("operation_id", None)
    value.pop("caller_request_fingerprint", None)
    value["resource_limits"] = fixed_job_resource_limits()
    return calculate_request_fingerprint("action_job", value)


def _gate_closed(record: Mapping[str, Any]) -> dict[str, Any]:
    """返回 scope 单 active gate 的统一错误。"""

    return _error(
        "action_job_phase_gate_closed",
        "当前 scope 已有未 collect 的 Action job；必须先对同一 job_id 完成 get/collect",
        job_id=record.get("job_id"),
        operation_id=record.get("operation_id"),
        submission_key=record.get("submission_key"),
    )


def _submit_action_job_once(request: ActionJobRequest) -> dict[str, Any]:
    """在 scope 锁内幂等登记并启动一个通用 Action supervisor。"""

    scope = current_scope()
    if scope is None:
        return _error("scope_required", "Action job 必须使用已认证 Investigation scope")
    scope_failure = _scope_error(request)
    if scope_failure is not None:
        return scope_failure
    selected = _adapter_for(request.action_id, request.backend_id)
    adapter_error = selected if isinstance(selected, dict) else None
    if adapter_error is None:
        adapter = selected
    else:
        # ActionJobRequest 已经通过 MCP/Pydantic 合同，scope 和 submission_key
        # 也已可用；未知 Action、错误 backend 或缺失 backend 不能在这里同步丢失
        # 身份。占位 adapter 只用于写入 failed 收据，真正的科学 worker 不会启动。
        adapter = ActionJobAdapter(
            action_id=request.action_id,
            backend_id=request.backend_id or "unresolved_backend",
            runtime="unresolved",
            software_id=None,
            execution_class="compute",
        )
    # 先用公开请求取得稳定提交身份；完整 Action 合同、backend health 和 staging
    # 检查都在 job_id 预留后执行，失败也能由 generic get/collect 读取原始错误。
    submission_key = validate_submission_key(str(request.submission_key))
    fingerprint = _request_fingerprint(request)
    operation_id = scoped_submission_operation_id(submission_key)
    resource_limits = fixed_job_resource_limits()
    catalog_hash = active_catalog_hash()
    request_payload = request.model_dump(mode="json")
    request_payload["resource_limits"] = dict(resource_limits)
    workspace = scope.workspace

    with _state_lock(workspace):
        state, readable = _load_state(scope)
        if not readable:
            return _error(
                "action_job_state_unavailable",
                "通用 Action job 身份收据损坏或 scope 不匹配，已阻止重新提交",
            )
        if state is None:
            state = _empty_state(scope)

        existing = state["jobs"].get(submission_key)
        if isinstance(existing, Mapping):
            record = dict(existing)
            if record.get("request_fingerprint") != fingerprint:
                return _error(
                    "submission_key_conflict",
                    "同一 scope/submission_key 已绑定不同的 Action job 请求",
                    job_id=record.get("job_id"),
                    operation_id=record.get("operation_id"),
                )
            if record.get("phase") == "unknown":
                return _error(
                    "action_job_submission_uncertain",
                    "已有 Action job 提交身份但启动状态无法核对，禁止重复提交",
                    job_id=record.get("job_id"),
                    operation_id=record.get("operation_id"),
                )
            checked = _registry_reservation(
                scope,
                record,
                preferred_job_id=str(record.get("job_id") or ""),
            )
            if isinstance(checked, dict):
                return checked
            status = _read_job_status(scope, str(record["job_id"]))
            if status is None and record.get("phase") != "collected":
                return _error(
                    "action_job_unavailable",
                    "已有 Action job 状态无法核对，已阻止重新提交",
                    job_id=record["job_id"],
                )
            return _minimal_identity(record, status, reused=True)

        active_key = state.get("active_submission_key")
        if active_key is not None:
            active = state["jobs"].get(active_key)
            if not isinstance(active, Mapping):
                return _error(
                    "action_job_state_unavailable",
                    "scope 活跃 Action job 记录不完整，已阻止新的提交",
                )
            if active.get("phase") != "collected":
                return _gate_closed(active)
            state["active_submission_key"] = None
            state["active_job_id"] = None

        provisional = {
            "operation_id": operation_id,
            "operation_fingerprint": fingerprint,
            "request_fingerprint": fingerprint,
            "submission_key": submission_key,
        }
        checked = _registry_reservation(scope, provisional, preferred_job_id=None)
        if isinstance(checked, dict):
            return checked
        registry, reservation = checked
        job_id = reservation.job_id
        record = {
            "job_id": job_id,
            "operation_id": operation_id,
            "operation_fingerprint": fingerprint,
            "request_fingerprint": fingerprint,
            "submission_key": submission_key,
            "job_type": ACTION_JOB_TYPE,
            "action_id": adapter.action_id,
            "backend_id": adapter.backend_id,
            "runtime": adapter.runtime,
            "software_id": adapter.software_id,
            "catalog_hash": catalog_hash,
            "scope_id": scope.scope_id,
            "scope_workspace": str(workspace),
            "phase": "submitting",
            "submitted_at": _now(),
        }
        state["jobs"][submission_key] = record
        state["active_submission_key"] = submission_key
        state["active_job_id"] = job_id
        _save_state(state, workspace)

        async_limits = dict(resource_limits)
        async_limits["walltime_seconds"] = timeout_seconds_for(adapter.execution_class)
        try:
            prepared = _submission_plan(request, adapter)
            # 选择错误只作为防止误启动的兜底；优先保留完整 Action 合同给出的
            # unknown_action/unsupported_backend 等原始诊断。
            if adapter_error is not None and not isinstance(prepared, dict):
                prepared = adapter_error
        except Exception as exc:
            prepared = _error(
                "action_job_request_validation_failed",
                "Action job 请求校验抛出异常",
                exception_type=type(exc).__name__,
                exception_message=_sanitize_public_diagnostic_value(str(exc)),
            )
        if isinstance(prepared, dict):
            failure = _accept_terminal_failed_action_job(
                state=state,
                workspace=workspace,
                scope=scope,
                registry=registry,
                record=record,
                request_payload=request_payload,
                resource_limits=async_limits,
                error=prepared.get("error", {
                    "code": "action_job_invalid_request",
                    "message": "Action job 请求校验失败",
                }),
            )
            if failure is not None:
                return failure
            return _error(
                "action_job_submission_uncertain",
                "Action job 请求失败但 terminal failed 状态无法安全持久化，已保留 job_id",
                job_id=job_id,
                operation_id=operation_id,
                submission_key=submission_key,
            )
        plan, normalized_request = prepared
        # plan.request 已经由服务端按选定 backend 固定合法资源；持久请求收据和
        # supervisor 必须使用同一份结果，调用方原始请求另存于 caller 字段。
        request_payload = normalized_request.model_dump(mode="json")
        resource_limits = dict(request_payload["resource_limits"])
        async_limits = dict(resource_limits)
        async_limits["walltime_seconds"] = timeout_seconds_for(
            adapter.execution_class
        )
        try:
            unavailable = _validate_action_backend_health(plan)
        except Exception as exc:
            unavailable = _error(
                "action_job_backend_health_failed",
                "Action backend health 检查抛出异常",
                exception_type=type(exc).__name__,
                exception_message=_sanitize_public_diagnostic_value(str(exc)),
            )
        if unavailable is not None:
            failure = _accept_terminal_failed_action_job(
                state=state,
                workspace=workspace,
                scope=scope,
                registry=registry,
                record=record,
                request_payload=request_payload,
                resource_limits=async_limits,
                error=unavailable.get("error", {
                    "code": "action_job_backend_unavailable",
                    "message": "Action backend health check failed",
                }),
            )
            if failure is not None:
                return failure
            return _error(
                "action_job_submission_uncertain",
                "backend health 失败但 terminal failed 状态无法安全持久化，已保留 job_id",
                job_id=job_id,
                operation_id=operation_id,
                submission_key=submission_key,
            )
        try:
            worker_request, staged_inputs, input_snapshots = _build_staged_worker_request(
                plan,
                normalized_request,
            )
            # walltime 是 evaluator policy 的内部字段，不进入 ActionRequest 合同；
            # detached worker 仍必须得到它，避免 backend 回退到过短的默认超时。
            worker_request["resource_limits"] = dict(async_limits)
        except (OSError, TypeError, ValueError) as exc:
            failure = _accept_terminal_failed_action_job(
                state=state,
                workspace=workspace,
                scope=scope,
                registry=registry,
                record=record,
                request_payload=request_payload,
                resource_limits=async_limits,
                error={
                    "code": "action_job_input_staging_failed",
                    "message": "Action job 输入无法安全复制到 detached workspace",
                    "exception_type": type(exc).__name__,
                    "exception_message": _sanitize_public_diagnostic_value(str(exc)),
                },
            )
            if failure is not None:
                return failure
            return _error(
                "action_job_submission_uncertain",
                "输入 staging 失败但 terminal failed 状态无法安全持久化，已保留 job_id",
                job_id=job_id,
                operation_id=operation_id,
                submission_key=submission_key,
            )
        try:
            submitted = _start_job(
                job_id=job_id,
                job_type=ACTION_JOB_TYPE,
                runtime=adapter.runtime,
                command=[
                    str(runtime_python(adapter.runtime)),
                    "-m",
                    WORKER_MODULE,
                    "--worker",
                ],
                stdin_target=None,
                staged_inputs=staged_inputs,
                resource_limits=async_limits,
                metadata={
                    "software_id": adapter.software_id,
                    "action_id": adapter.action_id,
                    "backend_id": adapter.backend_id,
                    "adapter_key": f"{adapter.action_id}/{adapter.backend_id}",
                    "label": "action_job_async",
                    "calculation_intent": adapter.action_id,
                    "action_job_request": request_payload,
                    "worker_action_job_request": worker_request,
                    "caller_action_job_request": request.model_dump(mode="json"),
                    "catalog_hash": catalog_hash,
                    "execution_timeout_policy": timeout_policy_record(adapter.execution_class),
                },
                operation_id=operation_id,
                operation_fingerprint=fingerprint,
                submission_key=submission_key,
                caller_request_fingerprint=fingerprint,
                expected_input_snapshots=input_snapshots,
            )
        except Exception as exc:
            failure = _accept_terminal_failed_action_job(
                state=state,
                workspace=workspace,
                scope=scope,
                registry=registry,
                record=record,
                request_payload=request_payload,
                resource_limits=async_limits,
                error={
                    "code": "action_job_submission_failed",
                    "message": "Action job supervisor 启动失败",
                    "exception_type": type(exc).__name__,
                    "exception_message": _sanitize_public_diagnostic_value(str(exc)),
                },
            )
            if failure is not None:
                return failure
            return _error(
                "action_job_submission_uncertain",
                "Action job supervisor 启动失败但 terminal failed 状态无法安全持久化，已保留 job_id",
                job_id=job_id,
                operation_id=operation_id,
                submission_key=submission_key,
            )

        if not isinstance(submitted, Mapping):
            failure_error = {
                "code": "action_job_invalid_supervisor_response",
                "message": "supervisor 未返回可核对的 JSON object",
                "response_type": type(submitted).__name__,
            }
            failure = _accept_terminal_failed_action_job(
                state=state,
                workspace=workspace,
                scope=scope,
                registry=registry,
                record=record,
                request_payload=request_payload,
                resource_limits=async_limits,
                error=failure_error,
            )
            if failure is not None:
                return failure
            return _error("action_job_submission_uncertain", failure_error["message"], job_id=job_id)
        if str(submitted.get("job_id") or "") != job_id:
            failure_error = {
                "code": "action_job_identity_conflict",
                "message": "supervisor 返回的 job_id 与预留身份不一致",
                "response_job_id": submitted.get("job_id"),
            }
            failure = _accept_terminal_failed_action_job(
                state=state,
                workspace=workspace,
                scope=scope,
                registry=registry,
                record=record,
                request_payload=request_payload,
                resource_limits=async_limits,
                error=failure_error,
            )
            if failure is not None:
                return failure
            return _error("action_job_submission_uncertain", failure_error["message"], job_id=job_id)
        terminal_error = _submitted_terminal_failure(submitted)
        if terminal_error is not None:
            failure = _accept_terminal_failed_action_job(
                state=state,
                workspace=workspace,
                scope=scope,
                registry=registry,
                record=record,
                request_payload=request_payload,
                resource_limits=async_limits,
                error=terminal_error,
            )
            if failure is not None:
                return failure
            return _error(
                "action_job_submission_uncertain",
                "确定性启动失败无法安全持久化，已保留 job_id",
                job_id=job_id,
            )
        record["phase"] = "accepted"
        record["accepted_at"] = _now()
        state["jobs"][submission_key] = record
        _save_state(state, workspace)
        status = _read_job_status(scope, job_id)
        return _minimal_identity(record, status or submitted, reused=False)


def submit_action_job(request: ActionJobRequest) -> dict[str, Any]:
    """提交通用 Action job 并立即返回服务端 job_id。

    ``search_transition_state_path`` 仍由 TS 专用 worker 执行，但从 Investigation
    模型看它也是 Catalog Action；因此这里把同一份科学 request 安全转换后转发到
    TS adapter。转发不新建外围 job，不生成第二个 job_id，TS adapter 的持久身份和
    generic get/collect facade 仍是唯一事实来源。
    """

    if request.action_id == TS_ACTION_ID:
        from .transition_state_jobs import (
            BACKEND_ID,
            TransitionStatePathJobRequest,
            submit_transition_state_path_job,
        )

        payload = request.model_dump(mode="json")
        payload.pop("action_id", None)
        # 空字符串也作为“来自统一入口”的内部标记；backend 未填写时仍要让
        # TS 专用提交在预留 job_id 后报告缺失/错误选择，而不能退回旧专用路径。
        payload["requested_backend_id"] = request.backend_id or ""
        payload["backend_id"] = BACKEND_ID
        # ActionJobRequest 已经完成公共字段和 scope 身份字段的 Pydantic 校验；TS
        # 专用 backend/科学合同必须延迟到专用 submit 已预留 job_id 之后检查。
        # 因此这里不能再次 model_validate：调用方传入错误 backend 时也必须由 TS
        # submit 只分配一个 job_id 并把失败持久化为 generic get/collect 可见的终态。
        if isinstance(payload.get("resource_limits"), Mapping):
            payload["resource_limits"] = ResourceLimits.model_validate(
                payload["resource_limits"]
            )
        transition_request = TransitionStatePathJobRequest.model_construct(**payload)
        return submit_transition_state_path_job(transition_request)

    return execute_traced(
        "submit_action_job",
        {"request": request.model_dump(mode="json")},
        lambda: _submit_action_job_once(request),
        capture_artifacts=False,
    )


def investigation_inspect_action(request: ActionInspectRequest) -> dict[str, Any]:
    """投影 Investigation 专用 Action 入口，避免模型把同步入口当作主路径。"""

    result = discovery_tools.inspect_action(request)
    projected = _scrub_model_visible_action_projection(deepcopy(result))
    submit_tool = "submit_action_job"
    projected["execute_with"] = submit_tool
    selected = _adapter_for(request.action_id, request.backend_id)
    if request.action_id == TS_ACTION_ID:
        eligible = request.backend_id in {None, TS_BACKEND_ID}
        resolved_backend_id = request.backend_id or TS_BACKEND_ID
    else:
        eligible = isinstance(selected, ActionJobAdapter)
        resolved_backend_id = selected.backend_id if isinstance(selected, ActionJobAdapter) else None
    projected["investigation_async_execution"] = {
        "eligible": eligible,
        "submit_tool": submit_tool,
        "status_tool": "get_execution_job",
        "collect_tool": "collect_execution_job",
        "resolved_backend_id": resolved_backend_id,
        "selection_required": not eligible,
        "contract": (
            "提交后立即取得 job_id；即使 submit 回执已经 terminal=true/failed，也必须先 "
            "对同一 job_id 调用 get_execution_job，随后才调用 collect_execution_job "
            "读取 canonical ActionResult。"
        ),
    }
    # Investigation 不公开同步 batch；即使完整目录把 Action 标成 batch-safe，
    # 这里也只允许逐项完成 submit→get→collect 后再开始下一项。
    projected["parallel_execution"] = {
        "eligible": False,
        "recommended_tool": submit_tool,
        "concurrency": "sequential_after_collect",
        "reason": "Investigation scope 同时只保留一个未 collect 的作业",
    }
    return projected


def _lookup_model_record(
    scope: ScopeContext,
    job_id: str,
) -> tuple[dict[str, Any], dict[str, Any]] | dict[str, Any]:
    """按 job_id 查找当前 scope 的通用 Action job。"""

    state, readable = _load_state(scope)
    if not readable or state is None:
        return _error("action_job_state_unavailable", "通用 Action job 身份收据不可核对")
    record = _find_job_record(state, job_id)
    if record is None:
        return _error(
            "action_job_unavailable",
            "该 job_id 不属于当前 capability scope 或身份记录不唯一",
        )
    adapter = _adapter_for(record.get("action_id"), record.get("backend_id"))
    if isinstance(adapter, dict):
        # 已经持久化为 terminal failed 的参数/选择错误不再需要当前 Catalog
        # 解析才能查询原始错误；否则模型无法通过 generic get/collect 修正请求。
        if record.get("phase") != "failed":
            return _error(
                "action_job_state_unavailable",
                "持久 Action job adapter 已不在当前冻结 Catalog 中，已停止恢复",
                job_id=job_id,
            )
        return state, record
    if record.get("runtime") != adapter.runtime:
        return _error(
            "action_job_identity_conflict",
            "持久 Action job runtime 与冻结 adapter 不一致，已停止恢复",
            job_id=job_id,
        )
    if record.get("catalog_hash") != active_catalog_hash():
        return _error(
            "action_job_catalog_identity_conflict",
            "持久 Action job 的 Catalog identity 与当前服务不一致，已停止恢复",
            job_id=job_id,
        )
    return state, record


def get_action_job(job_id: str) -> dict[str, Any]:
    """返回通用 Action job 的最小状态；公开 MCP 入口仍使用 get_execution_job。"""

    scope = current_scope()
    if scope is None:
        return _error("scope_required", "Action job 必须在已认证 scope 中查询")
    with _state_lock(scope.workspace):
        found = _lookup_model_record(scope, job_id)
        if isinstance(found, dict):
            return found
        _state, record = found
        checked = _registry_reservation(scope, record, preferred_job_id=job_id)
        if isinstance(checked, dict):
            return checked
        registry, _reservation = checked
        status = _read_job_status(scope, job_id)
        if status is None:
            return _error(
                "action_job_unavailable",
                "作业状态无法核对，已停止恢复",
                job_id=job_id,
            )
        if status.get("status") in {"queued", "running", *TERMINAL_STATES}:
            try:
                registry.mark(job_id, phase=str(status["status"]))
            except (ExecutionJobRegistryError, ValueError):
                pass
        return _minimal_identity(record, status, reused=True)


def _read_action_job_request(
    scope: ScopeContext,
    record: Mapping[str, Any],
) -> ActionJobRequest | None:
    """恢复并核对 Action job 请求，返回重新规范化后的可信请求。

    输入是当前 scope 的 state 记录和 supervisor 私有 ``request.json``。函数先核对
    job、scope、operation、adapter 及两份持久化请求快照，再用调用方原始请求重新运行
    Action 合同，使服务端补全的 ArtifactRef 只能在与原规范请求完全一致时进入 collect。
    任一身份、快照或合同校验失败都返回 ``None``，不修改状态，也不释放 active gate。
    """

    directory = _job_directory(scope.workspace, str(record.get("job_id") or ""))
    if directory is None:
        return None
    request_record = _read_json_file(directory / "request.json")
    if request_record is None:
        return None
    for field_name in (
        "job_id",
        "scope_id",
        "scope_workspace",
        "operation_id",
        "submission_key",
        "job_type",
        "runtime",
        "operation_fingerprint",
        "caller_request_fingerprint",
    ):
        expected_value = (
            record.get("request_fingerprint")
            if field_name == "caller_request_fingerprint"
            else record.get(field_name)
        )
        if request_record.get(field_name) != expected_value:
            return None
    metadata = request_record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    if (
        metadata.get("scope_id") != scope.scope_id
        or metadata.get("scope_workspace") != str(scope.workspace)
        or metadata.get("caller_request_fingerprint")
        != record.get("request_fingerprint")
        or metadata.get("action_id") != record.get("action_id")
        or metadata.get("backend_id") != record.get("backend_id")
        or metadata.get("adapter_key")
        != f"{record.get('action_id')}/{record.get('backend_id')}"
        or metadata.get("catalog_hash") != record.get("catalog_hash")
        or metadata.get("calculation_intent") != record.get("action_id")
        or metadata.get("software_id") != record.get("software_id")
    ):
        return None
    raw_caller_request = metadata.get("caller_action_job_request")
    raw_normalized_request = metadata.get("action_job_request")
    if not isinstance(raw_caller_request, Mapping) or not isinstance(
        raw_normalized_request, Mapping
    ):
        return None
    try:
        caller_request = ActionJobRequest.model_validate(raw_caller_request)
        if dict(raw_caller_request) != caller_request.model_dump(mode="json"):
            return None
        if caller_request.submission_key != record.get("submission_key"):
            return None
        if _request_fingerprint(caller_request) != record.get("request_fingerprint"):
            return None
        normalized_snapshot = ActionJobRequest.model_validate(raw_normalized_request)
        if dict(raw_normalized_request) != normalized_snapshot.model_dump(mode="json"):
            return None
    except Exception:
        return None
    adapter = _adapter_for(caller_request.action_id, caller_request.backend_id)
    if not isinstance(adapter, ActionJobAdapter):
        return None
    if (
        adapter.action_id != record.get("action_id")
        or adapter.runtime != record.get("runtime")
        or adapter.backend_id != record.get("backend_id")
    ):
        return None
    try:
        prepared = _submission_plan(caller_request, adapter)
    except Exception:
        return None
    if isinstance(prepared, dict):
        return None
    _plan, normalized_request = prepared
    if normalized_request.model_dump(mode="json") != normalized_snapshot.model_dump(
        mode="json"
    ):
        return None
    return normalized_request


def _safe_relative_file(workspace: Path, value: Any) -> Path | None:
    """解析 workspace 内普通文件，拒绝越界、绝对路径和符号链接。"""

    if not isinstance(value, str) or not value.strip() or "\\" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts:
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    candidate = workspace.joinpath(*relative.parts)
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    cursor = workspace
    for part in relative.parts:
        cursor /= part
        try:
            metadata = os.lstat(cursor)
        except (FileNotFoundError, OSError):
            return None
        if stat.S_ISLNK(metadata.st_mode):
            return None
    try:
        return candidate if candidate.is_file() and not candidate.is_symlink() else None
    except OSError:
        return None


def _prefix_worker_artifacts(
    raw_worker_result: Mapping[str, Any],
    *,
    workspace: Path,
    job_id: str,
) -> dict[str, Any] | None:
    """把 job 私有 worker 文件映射到父 scope 作业目录。"""

    raw_files = raw_worker_result.get("artifact_files", [])
    if raw_files is None:
        raw_files = []
    if not isinstance(raw_files, (list, tuple)):
        return None
    normalized = deepcopy(dict(raw_worker_result))
    entries: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            return None
        raw_path = item.get("path")
        relative = PurePosixPath(raw_path) if isinstance(raw_path, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or not relative.parts
            or "\\" in str(raw_path)
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            return None
        target_relative = PurePosixPath("outputs", "execution_jobs", job_id).joinpath(
            *relative.parts
        )
        if _safe_relative_file(workspace, str(target_relative)) is None:
            return None
        entry = dict(item)
        entry["path"] = str(target_relative)
        entries.append(entry)
    normalized["artifact_files"] = entries
    return normalized


def _build_parent_action_result(
    raw_worker_result: Mapping[str, Any],
    request: ActionJobRequest,
    *,
    workspace: Path,
    job_id: str,
) -> dict[str, Any]:
    """在父 scope 登记 ArtifactRef 并封装唯一 canonical ActionResult。"""

    normalized = _prefix_worker_artifacts(raw_worker_result, workspace=workspace, job_id=job_id)
    if normalized is None:
        raise ValueError("Action worker artifact_files 不是安全的相对文件清单")
    plan = _prepare_action_execution(request.action_id, request, resources_bound=True)
    if isinstance(plan, dict):
        raise ValueError("Action job 请求在 collect 时无法通过 Action 合同")
    directory = _job_directory(workspace, job_id)
    status_record = (
        _read_json_file(directory / "status.json")
        if directory is not None
        else None
    )
    resource_allocation = (
        deepcopy(dict(status_record.get("resource_allocation")))
        if isinstance(status_record, Mapping)
        and isinstance(status_record.get("resource_allocation"), Mapping)
        else {}
    )
    worker, status, result_payload = _normalize_action_worker_result(normalized)
    status, result_payload, output_artifacts, worker = _collect_action_result_artifacts(
        plan,
        worker,
        status,
        result_payload,
    )
    result = _build_action_result(
        plan,
        worker,
        status,
        result_payload,
        output_artifacts,
        resource_allocation,
    )
    result["provenance"]["async_job_id"] = job_id
    result["provenance"]["execution_mode"] = "detached_action_job"
    return ActionResult.model_validate(result).model_dump(mode="json")


def _validate_parent_artifacts(
    action_result: Mapping[str, Any],
    *,
    workspace: Path,
    action_id: str,
    backend_id: str,
) -> None:
    """核对父 scope ArtifactStore 中每个结果引用的身份和摘要。"""

    references = action_result.get("output_artifacts")
    if not isinstance(references, list):
        raise ValueError("ActionResult.output_artifacts 必须是数组")
    store = ArtifactStore(root=workspace, create=False)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for value in references:
        reference = ArtifactRef.model_validate(value)
        if reference.artifact_id in seen_ids or reference.path in seen_paths:
            raise ValueError("ActionResult.output_artifacts 存在重复 artifact 身份")
        seen_ids.add(reference.artifact_id)
        seen_paths.add(reference.path)
        if reference.producer_action != action_id or reference.producer_backend != backend_id:
            raise ValueError("ActionResult artifact 不属于当前 Action job")
        canonical = store.find(reference.artifact_id)
        if canonical.model_dump(mode="json") != reference.model_dump(mode="json"):
            raise ValueError("ActionResult artifact 与父 scope manifest 不一致")
        path = _safe_relative_file(workspace, reference.path)
        if path is None:
            raise ValueError("ActionResult artifact path 不属于父 scope")
        _, digest = read_file_snapshot(path)
        if digest != reference.sha256:
            raise ValueError("ActionResult artifact SHA-256 与文件内容不一致")


def _read_canonical_result(scope: ScopeContext, job_id: str) -> dict[str, Any] | None:
    """读取已落盘 canonical ActionResult，损坏时不重放。"""

    directory = _job_directory(scope.workspace, job_id)
    if directory is None:
        return None
    raw = _read_json_file(directory / ACTION_JOB_RESULT_FILENAME)
    if raw is None:
        return None
    try:
        return ActionResult.model_validate(raw).model_dump(mode="json")
    except Exception:
        return None


def _collection_response(
    record: Mapping[str, Any],
    *,
    job_status: str,
    action_result: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 generic collect 的稳定结果 envelope。"""

    response: dict[str, Any] = {
        "status": "success",
        "ready": True,
        "job_id": record["job_id"],
        "job_status": job_status,
        "operation_id": record["operation_id"],
        "submission_key": record["submission_key"],
        "action_id": record["action_id"],
        "backend_id": record["backend_id"],
        "automatic_fallback": False,
    }
    if action_result is not None:
        # canonical action_result.json 已在调用方完成校验并保持完整；这里只为模型传输
        # 复制一份有界视图，避免 Hessian 等大数组在 MCP 的嵌套 JSON 文本中重复展开。
        response["action_result"] = compact_action_result(dict(action_result))
    if error is not None:
        response["error"] = deepcopy(dict(error))
    return response


def collect_action_job(
    job_id: str,
    *,
    submission_key: str | None = None,
) -> dict[str, Any]:
    """通过统一 job_id 收集 generic ActionResult；由 open_execution facade 调用。"""

    scope = current_scope()
    if scope is None:
        return _error("scope_required", "Action job 必须在已认证 scope 中收集")
    if JOB_ID_PATTERN.fullmatch(str(job_id)) is None:
        return _error("invalid_job_id", "job_id 不是有效的 MiniChem 作业身份")
    with _state_lock(scope.workspace):
        found = _lookup_model_record(scope, job_id)
        if isinstance(found, dict):
            return found
        state, record = found
        if submission_key is not None and submission_key != record.get("submission_key"):
            return _error(
                "submission_key_conflict",
                "collect_execution_job 的 submission_key 不属于目标 Action job",
                job_id=job_id,
            )
        checked = _registry_reservation(scope, record, preferred_job_id=job_id)
        if isinstance(checked, dict):
            return checked
        registry, _reservation = checked
        status = _read_job_status(scope, job_id)
        if status is None:
            return _error(
                "action_job_unavailable",
                "作业状态无法核对，已停止收集",
                job_id=job_id,
            )
        job_status = str(status.get("status") or "")
        if job_status not in TERMINAL_STATES:
            return _error(
                "action_job_not_terminal",
                "作业尚未进入终态；请先按该 job_id 轮询 get_execution_job",
                job_id=job_id,
                job_status=job_status,
            )

        adapter = _adapter_for(record.get("action_id"), record.get("backend_id"))
        if (
            isinstance(adapter, dict)
            and not (
                job_status in {"failed", "timeout", "cancelled"}
                and isinstance(status.get("error"), Mapping)
            )
        ):
            return _error(
                "action_job_state_unavailable",
                "Action job adapter 不再属于当前 Catalog，已停止收集",
                job_id=job_id,
            )
        action_result = _read_canonical_result(scope, job_id)
        if action_result is None and job_status in {"failed", "timeout", "cancelled"}:
            failure = status.get("error")
            if not isinstance(failure, Mapping):
                failure = {
                    "code": f"execution_{job_status}",
                    "message": "Action job 已结束且没有可收集的科学结果",
                }
            try:
                registry.mark(job_id, phase="collected")
            except (ExecutionJobRegistryError, ValueError):
                return _error(
                    "action_job_state_unavailable",
                    "Action job 已终态但 registry 未能记录 collected，可按原 job_id 重试",
                    job_id=job_id,
                )
            record["phase"] = "collected"
            record["collected_at"] = _now()
            state["jobs"][record["submission_key"]] = record
            if state.get("active_submission_key") == record["submission_key"]:
                state["active_submission_key"] = None
                state["active_job_id"] = None
            _save_state(state, scope.workspace)
            return _collection_response(record, job_status=job_status, error=failure)

        if action_result is None:
            directory = _job_directory(scope.workspace, job_id)
            raw_result = (
                _read_json_file(directory / RAW_RESULT_FILENAME)
                if directory is not None
                else None
            )
            action_request = _read_action_job_request(scope, record)
            if raw_result is None or action_request is None:
                return _error(
                    "action_job_result_unavailable",
                    "作业已终态但原始 worker 结果或请求身份尚未安全落盘，已停止收集",
                    job_id=job_id,
                )
            try:
                action_result = _build_parent_action_result(
                    raw_result,
                    action_request,
                    workspace=scope.workspace,
                    job_id=job_id,
                )
                _validate_parent_artifacts(
                    action_result,
                    workspace=scope.workspace,
                    action_id=adapter.action_id,
                    backend_id=adapter.backend_id,
                )
                if directory is None:
                    raise ValueError("Action job directory 无效")
                _atomic_json(directory / ACTION_JOB_RESULT_FILENAME, action_result)
            except Exception as exc:
                return _error(
                    "action_job_result_unavailable",
                    "原始 Action worker 结果无法在父 scope 中安全登记，已停止收集",
                    job_id=job_id,
                    reason_code="artifact_registration_failed",
                    exception_type=type(exc).__name__,
                )
        else:
            try:
                _validate_parent_artifacts(
                    action_result,
                    workspace=scope.workspace,
                    action_id=adapter.action_id,
                    backend_id=adapter.backend_id,
                )
            except Exception as exc:
                return _error(
                    "action_job_result_unavailable",
                    "已落盘 ActionResult 的 artifact manifest 无法核对，已停止收集",
                    job_id=job_id,
                    reason_code="artifact_manifest_invalid",
                    exception_type=type(exc).__name__,
                )
        try:
            registry.mark(job_id, phase="collected")
        except (ExecutionJobRegistryError, ValueError):
            return _error(
                "action_job_state_unavailable",
                "canonical ActionResult 已核对但 registry 未能记录 collected，可按原 job_id 重试",
                job_id=job_id,
            )
        record["phase"] = "collected"
        record["collected_at"] = _now()
        state["jobs"][record["submission_key"]] = record
        if state.get("active_submission_key") == record["submission_key"]:
            state["active_submission_key"] = None
            state["active_job_id"] = None
        _save_state(state, scope.workspace)
        return _collection_response(record, job_status=job_status, action_result=action_result)


def investigation_execute_action(request: ProgressiveActionRequest) -> dict[str, Any]:
    """Investigation 同步入口按闭合 adapter registry 统一拒绝异步 Action。"""

    if request.action_id == TS_ACTION_ID:
        return _error(
            "async_job_required",
            "Investigation 中 Catalog Action 必须先使用 submit_action_job，再按统一 "
            "get_execution_job→终态→collect_execution_job 完成",
            required_submit_tool="submit_action_job",
            required_get_tool="get_execution_job",
            required_collect_tool="collect_execution_job",
        )
    adapter = _adapter_for(request.action_id, request.backend_id)
    if isinstance(adapter, ActionJobAdapter):
        return execute_traced(
            request.action_id,
            {
                "entrypoint": "progressive_execute_action",
                "request": request.model_dump(mode="json"),
            },
            lambda: _error(
                "async_job_required",
                "Investigation 中 Catalog Action 必须先使用 submit_action_job，"
                "再按统一 get_execution_job→终态→collect_execution_job 完成",
                required_submit_tool="submit_action_job",
                required_get_tool="get_execution_job",
                required_collect_tool="collect_execution_job",
            ),
            capture_artifacts=False,
        )
    # 未命中 Catalog adapter 时沿用 discovery 的精确错误，不执行同步科学计算。
    return _error(
        "unsupported_action_backend",
        "Action/backend 不在 Investigation 的闭合异步 adapter registry 中",
        action_id=request.action_id,
        backend_id=request.backend_id,
        required_submit_tool="submit_action_job",
    )


def _worker_action_result(
    request: Mapping[str, Any],
    job_id: str,
    *,
    backend_id: str,
) -> dict[str, Any]:
    """在 detached worker 中调用原有 local backend，不嵌套资源预留。"""

    from minichem_toolbox.backends import execute_local

    del job_id
    return execute_local(
        str(request["action_id"]),
        backend_id,
        dict(request),
    )


def _worker_exception_execution_diagnostic(exc: Exception) -> dict[str, Any]:
    """保留后端提供的有限执行状态，不把异常堆栈或请求 schema 暴露给调用方。"""

    diagnostic: dict[str, Any] = {}
    try:
        returncode = getattr(exc, "returncode")
    except AttributeError:
        returncode = None
    if returncode is not None and type(returncode) is int:
        diagnostic["inner_returncode"] = returncode
    for name in ("termination", "archive_warning", "timeout"):
        try:
            value = getattr(exc, name)
        except AttributeError:
            continue
        if name == "termination" and isinstance(value, str):
            diagnostic[name] = value
        elif name in {"archive_warning", "timeout"} and type(value) is bool:
            diagnostic[name] = value
    return diagnostic


def _run_worker() -> int:
    """读取 supervisor request.json，运行闭合 adapter backend 并写入原始结果。"""

    job_directory = Path.cwd().resolve()
    os.environ["MINICHEM_MCP_WORKSPACE"] = str(job_directory)
    for name in (
        "MINICHEM_SCOPE_ID",
        "MINICHEM_SCOPE_TASK",
        "MINICHEM_SCOPE_BATCH",
        "MINICHEM_SCOPE_COMBINATION",
        "MINICHEM_SHARED_SCHEDULER_ROOT",
    ):
        os.environ.pop(name, None)
    request_record = _read_json_file(job_directory / "request.json")
    if request_record is None:
        return 2
    metadata = request_record.get("metadata")
    raw_request = (
        metadata.get("worker_action_job_request")
        if isinstance(metadata, Mapping)
        else None
    )
    job_id = str(request_record.get("job_id") or "")
    if not isinstance(raw_request, Mapping) or JOB_ID_PATTERN.fullmatch(job_id) is None:
        return 2
    action_id = raw_request.get("action_id")
    backend_id = metadata.get("backend_id") if isinstance(metadata, Mapping) else None
    adapter = _adapter_for(action_id, backend_id)
    if isinstance(adapter, dict):
        return 2
    try:
        result = _worker_action_result(
            raw_request,
            job_id,
            backend_id=adapter.backend_id,
        )
    except Exception as exc:
        execution_diagnostic = _worker_exception_execution_diagnostic(exc)
        result = {
            "status": "failed",
            "error": {
                "code": "action_job_worker_failed",
                "message": "Action job detached worker failed before producing a scientific result",
                "exception_type": type(exc).__name__,
                "exception_message": _sanitize_public_diagnostic_value(str(exc)),
                **(
                    {"execution_diagnostic": execution_diagnostic}
                    if execution_diagnostic
                    else {}
                ),
            },
            "retryable": False,
            "provenance": {"async_job_id": job_id},
        }
    if not isinstance(result, Mapping):
        result = {
            "status": "failed",
            "error": {
                "code": "invalid_worker_response",
                "message": "Action job worker did not return a mapping",
            },
        }
    _atomic_json(job_directory / RAW_RESULT_FILENAME, dict(result))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        raise SystemExit(_run_worker())
    raise SystemExit(2)


__all__ = [
    "ACTION_JOB_ADAPTERS",
    "ACTION_JOB_TYPE",
    "ACTION_JOB_RESULT_FILENAME",
    "ActionJobAdapter",
    "ActionJobRequest",
    "JOB_TYPE",
    "RAW_RESULT_FILENAME",
    "RESULT_FILENAME",
    "TOOL_DESCRIPTIONS",
    "TOOL_NAMES",
    "ProgressiveActionRequest",
    "collect_action_job",
    "get_action_job",
    "investigation_inspect_action",
    "investigation_execute_action",
    "submit_action_job",
]
