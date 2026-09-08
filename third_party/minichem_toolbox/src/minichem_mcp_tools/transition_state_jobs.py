"""Transition-State Path Pipeline 的持久异步作业边界。

该模块只负责提交、查询和收集一个 TS Pipeline 作业，不改变
``arche_ts_pipeline.run_atomic_workflow`` 的科学步骤、参数或结果解释。异步工作流由
已有的 detached job supervisor 启动；模型拿到作业身份后可在同一 turn 查询和收集
原始 ActionResult，Controller 的显式 arm wake 只保留给进程中断后的恢复路径。
"""

from __future__ import annotations

import fcntl
import hashlib
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

from pydantic import BaseModel, ConfigDict, Field, field_validator

from minichem_toolbox.artifacts import ArtifactStore, collect_artifact_refs, read_file_snapshot
from minichem_toolbox.catalog import action_specs, active_catalog_hash, backend_specs
from minichem_toolbox.idempotency import (
    calculate_request_fingerprint,
    scoped_submission_operation_id,
    scoped_submission_request_error,
    validate_submission_key,
)
from minichem_toolbox.models import ActionRequest, ActionResult, ArtifactRef
from minichem_toolbox.resource_budget import (
    ResourceBudgetExceeded,
    ResourceStateUnavailable,
    fixed_job_resource_limits,
    resource_budget_record,
)
from minichem_toolbox.scope import ScopeContext, current_scope
from minichem_toolbox.timeout_policy import timeout_policy_record, timeout_seconds_for

from . import discovery_tools
from .open_execution import (
    TERMINAL_JOB_STATES,
    _atomic_json,
    _start_job,
)
from .execution_job_registry import (
    EXECUTION_JOB_ID_PATTERN,
    ExecutionJobRegistry,
    ExecutionJobRegistryConflict,
    ExecutionJobRegistryError,
)
from minichem_toolbox.service import (
    _ActionExecutionPlan,
    _build_action_result,
    _collect_action_result_artifacts,
    _normalize_action_worker_result,
    _sanitize_public_diagnostic_value,
)
from .tracing import execute_traced
from .workspace import ensure_private_directory, workspace_root


ACTION_ID = "search_transition_state_path"
BACKEND_ID = "transition_state_path_search"
JOB_TYPE = "transition_state_path"
WORKER_MODULE = "minichem_mcp_tools.transition_state_jobs"
RESULT_FILENAME = "transition_state_path_result.json"
RAW_RESULT_FILENAME = "transition_state_path_worker_result.json"
STATE_DIRECTORY_NAME = "transition_state_path_jobs"
STATE_FILENAME = "state.json"
TERMINAL_STATES = frozenset(TERMINAL_JOB_STATES)
UNKNOWN_JOB_STATUS = "unknown"
JOB_ID_PATTERN = EXECUTION_JOB_ID_PATTERN
STATE_SCHEMA_VERSION = 1

TOOL_NAMES = (
    "submit_transition_state_path_job",
    "get_transition_state_path_job",
    "collect_transition_state_path_job",
)

TOOL_DESCRIPTIONS = {
    "submit_transition_state_path_job": (
        "提交一个 Transition-State Path Pipeline 计算；输入使用 "
        "search_transition_state_path 所需的相同科学 request。成功时立即返回唯一的 job_id、"
        "operation_id、submission_key、action_id、backend_id 和最小提交状态。任一时刻最多"
        "保留一个仍活跃且未 collect 的异步计算；拿到 job_id 后可在同一 turn 用本组 get"
        "核对状态，终态后用 collect 读取结果。collect 成功后才可提交下一个计算；本工具不"
        "改变 TS Pipeline 的科学步骤或结果解释。"
    ),
    "get_transition_state_path_job": (
        "拿到 submit 返回的精确 job_id 后调用。输入只需精确的 job_id；返回该作业的最小"
        "状态和身份，不返回科学 payload、日志或 artifact 内容。作业身份无法核对时请求"
        "会被拒绝；状态仍未终止时返回 queued/running，模型应按该作业轮询节奏再次查询。"
    ),
    "collect_transition_state_path_job": (
        "仅在 get 已确认同一个 job_id 进入终态后调用。输入只需同一个精确 job_id；返回原始"
        "科学 ActionResult envelope 和可读取的 artifact 引用，供你继续查找、核对并解释。"
        "非终态、身份未核对、结果或 artifact 无法安全恢复时 fail-closed；collect 可在同一"
        "turn 内完成，本工具不会重新提交、重启或改写 TS Pipeline 的科学结果。"
    ),
}


class TransitionStatePathJobRequest(ActionRequest):
    """提交一个 TS Pipeline 作业的原始科学请求。"""

    backend_id: str = Field(
        default=BACKEND_ID,
        description="固定使用 transition_state_path_search；不得切换后端。",
    )
    # 仅供 submit_action_job 转发时保存调用方原始 backend；不进入公开请求
    # schema/指纹。这样错误 backend 也能先绑定唯一 job_id，再以 failed 终态返回。
    requested_backend_id: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """从 TS 公开请求 Schema 删除统一转发的内部 backend 标记。"""

        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("requested_backend_id", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [
                item for item in required if item != "requested_backend_id"
            ]
        return schema

    @field_validator("backend_id")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        """拒绝把异步入口路由到其他后端。"""

        if value != BACKEND_ID:
            raise ValueError(f"backend_id 必须为 {BACKEND_ID}")
        return value


class TransitionStatePathJobStatusRequest(BaseModel):
    """查询模型已经提交的 TS Pipeline 作业身份和最小状态。"""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(description="系统报告的精确 job_id；当前 scope 已绑定其归属。")

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        """只接受服务端生成的 job_id 形状。"""

        if JOB_ID_PATTERN.fullmatch(value.strip()) is None:
            raise ValueError("job_id 不是有效的 MiniChem 作业身份")
        return value.strip()


class TransitionStatePathJobCollectRequest(TransitionStatePathJobStatusRequest):
    """收集一个系统已报告终态的 TS Pipeline 作业。"""


@dataclass(frozen=True)
class TransitionStatePathJobController:
    """Controller 侧的最小状态检查和 wake arm 入口。

    Controller 不通过 MCP 工具获得科学结果，只能使用本对象读取终态并 arm 同一作业。
    ``scope_id`` 和 ``workspace`` 必须来自已认证的 Investigation scope；身份缺失或记录
    损坏时方法返回 fail-closed 错误，不允许重新提交。
    """

    scope_id: str
    workspace: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", str(self.scope_id).strip())
        object.__setattr__(
            self,
            "workspace",
            Path(self.workspace).expanduser().resolve(),
        )
        if not self.scope_id:
            raise ValueError("Controller scope_id 不能为空")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """读取作业最小终态信息，不返回科学结果、日志或路径。"""

        return _controller_get_job_status(
            job_id,
            scope_id=self.scope_id,
            workspace=self.workspace,
        )

    def arm_wake(self, job_id: str) -> dict[str, Any]:
        """确认作业已经终态并开放同一 job 的模型查询/收集 turn。"""

        return _controller_arm_job(
            job_id,
            scope_id=self.scope_id,
            workspace=self.workspace,
        )

    def adopt_completed_action_result(
        self,
        action_result: Mapping[str, Any] | str | os.PathLike[str] | Path,
        request: TransitionStatePathJobRequest | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """导入可信 completed envelope，并在本 Controller scope 创建终态收据。"""

        return controller_adopt_completed_action_result(
            action_result,
            request,
            scope_id=self.scope_id,
            workspace=self.workspace,
            **kwargs,
        )


def _now() -> str:
    """返回 UTC 持久化时间。"""

    return datetime.now(timezone.utc).isoformat()


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    """返回不暴露宿主路径、原始异常和科学结果的安全错误 envelope。"""

    value: dict[str, Any] = {
        "status": "invalid_request",
        "valid": False,
        "error": {"code": code, "message": message},
        "automatic_fallback": False,
    }
    value["error"].update(details)
    return value


def _scope_error(request: Any) -> dict[str, Any] | None:
    """核对当前 MCP 请求的 capability scope 和 submission_key。"""

    scope = current_scope()
    if scope is None:
        return _error(
            "scope_required",
            "Transition-State Path 异步作业必须在已认证 Investigation scope 中提交",
        )
    identity_error = scoped_submission_request_error(request)
    if identity_error is not None:
        return _error(str(identity_error["code"]), str(identity_error["message"]))
    try:
        validate_submission_key(str(request.submission_key))
    except ValueError as exc:
        return _error("invalid_submission_key", str(exc))
    return None


def _scope_state_root(workspace: Path) -> Path:
    """返回 scope 私有的异步状态目录并收紧权限。"""

    return ensure_private_directory(
        workspace / "_tool_artifacts" / STATE_DIRECTORY_NAME,
        create=True,
    )


@contextmanager
def _state_lock(workspace: Path) -> Iterator[None]:
    """以 scope 私有锁串行化状态读改写和 Controller arm。"""

    root = _scope_state_root(workspace)
    lock_path = root / "state.lock"
    descriptor = os.open(
        lock_path,
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


def _state_path(workspace: Path) -> Path:
    """返回 scope 状态文件路径。"""

    return _scope_state_root(workspace) / STATE_FILENAME


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """严格读取普通 JSON 控制文件；缺失和损坏均返回 None。"""

    try:
        if path.is_symlink() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _empty_state(scope: ScopeContext | None, *, scope_id: str | None = None, workspace: Path | None = None) -> dict[str, Any]:
    """构造空的 scope 状态；该状态尚未拥有任何作业。"""

    selected_scope_id = scope_id or (scope.scope_id if scope is not None else "")
    selected_workspace = workspace or (scope.workspace if scope is not None else Path("."))
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "scope_id": selected_scope_id,
        "scope_workspace": str(selected_workspace),
        "active_submission_key": None,
        "active_job_id": None,
        "jobs": {},
        "updated_at": _now(),
    }


def _validate_state(
    value: Mapping[str, Any] | None,
    *,
    scope_id: str,
    workspace: Path,
) -> dict[str, Any] | None:
    """核对持久状态身份和结构，任何不确定性都阻止新提交。"""

    if value is None:
        return None
    if value.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    if value.get("scope_id") != scope_id or value.get("scope_workspace") != str(workspace):
        return None
    jobs = value.get("jobs")
    if not isinstance(jobs, dict):
        return None
    active_key = value.get("active_submission_key")
    active_job_id = value.get("active_job_id")
    if active_key is not None and not isinstance(active_key, str):
        return None
    if active_job_id is not None and JOB_ID_PATTERN.fullmatch(str(active_job_id)) is None:
        return None
    normalized_jobs: dict[str, dict[str, Any]] = {}
    for key, raw in jobs.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            return None
        try:
            validate_submission_key(key)
        except ValueError:
            return None
        if (
            raw.get("scope_id") != scope_id
            or raw.get("scope_workspace") != str(workspace)
            or raw.get("submission_key") != key
        ):
            return None
        if JOB_ID_PATTERN.fullmatch(str(raw.get("job_id") or "")) is None:
            return None
        if not isinstance(raw.get("operation_id"), str) or not raw["operation_id"]:
            return None
        if not isinstance(raw.get("request_fingerprint"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", raw["request_fingerprint"]
        ):
            return None
        if raw.get("phase") not in {
            "submitting",
            "accepted",
            "armed",
            "failed",
            "collected",
            "unknown",
        }:
            return None
        if not isinstance(raw.get("wake_armed", False), bool):
            return None
        if "submission_error" in raw and not isinstance(raw.get("submission_error"), Mapping):
            return None
        normalized_jobs[key] = dict(raw)
    if active_key is None:
        if active_job_id is not None:
            return None
    elif active_key not in normalized_jobs:
        return None
    elif normalized_jobs[active_key].get("job_id") != active_job_id:
        return None
    normalized = dict(value)
    normalized["jobs"] = normalized_jobs
    return normalized


def _load_state(scope: ScopeContext, workspace: Path) -> tuple[dict[str, Any] | None, bool]:
    """读取状态并返回 ``(state, readable)``；损坏状态不会伪装成空状态。"""

    path = _state_path(workspace)
    raw = _read_json_file(path)
    if raw is None:
        return (None, True) if not path.exists() else (None, False)
    value = _validate_state(raw, scope_id=scope.scope_id, workspace=workspace)
    return (value, value is not None)


def _save_state(state: dict[str, Any], workspace: Path) -> None:
    """原子保存 scope 状态并刷新时间。"""

    state["updated_at"] = _now()
    _atomic_json(_state_path(workspace), state)


def _job_directory_for_workspace(workspace: Path, job_id: str) -> Path | None:
    """在显式 workspace 下解析作业目录并拒绝符号链接路径。"""

    if JOB_ID_PATTERN.fullmatch(str(job_id)) is None:
        return None
    root = workspace / "outputs" / "execution_jobs"
    candidate = root / job_id
    cursor = workspace
    try:
        relative = candidate.relative_to(workspace)
    except ValueError:
        return None
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            return None
    return candidate


def _read_job_status_for_workspace(
    workspace: Path,
    scope_id: str,
    job_id: str,
) -> dict[str, Any] | None:
    """读取 Controller 侧的作业状态并核对 scope 身份。"""

    directory = _job_directory_for_workspace(workspace, job_id)
    if directory is None:
        return None
    status = _read_json_file(directory / "status.json")
    if status is None:
        return None
    if (
        status.get("job_id") != job_id
        or status.get("scope_id") != scope_id
        or status.get("scope_workspace") != str(workspace)
    ):
        return None
    return status


def _job_record(state: Mapping[str, Any], submission_key: str) -> dict[str, Any] | None:
    """读取一个已登记提交键的内部状态。"""

    jobs = state.get("jobs")
    if not isinstance(jobs, Mapping):
        return None
    raw = jobs.get(submission_key)
    return dict(raw) if isinstance(raw, Mapping) else None


def _minimal_identity(
    record: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    *,
    idempotent_reused: bool,
    phase: str | None = None,
) -> dict[str, Any]:
    """构造提交/get/controller 共用的最小作业身份响应。"""

    job_status = status.get("status") if status is not None else None
    uncertain_terminal = (
        job_status == UNKNOWN_JOB_STATUS
        and (
            record.get("phase") == UNKNOWN_JOB_STATUS
            or record.get("terminal_status") == UNKNOWN_JOB_STATUS
        )
    )
    response = {
        "status": "success",
        "job_id": record["job_id"],
        "job": {"job_id": record["job_id"], "status": job_status},
        "operation_id": record["operation_id"],
        "submission_key": record["submission_key"],
        "action_id": ACTION_ID,
        "backend_id": BACKEND_ID,
        "job_status": job_status,
        "terminal": job_status in TERMINAL_STATES or uncertain_terminal,
        "phase": phase or str(record.get("phase") or "accepted"),
        "wake_armed": bool(record.get("wake_armed", False)),
        "idempotent_reused": idempotent_reused,
        "automatic_fallback": False,
    }
    if status is not None and isinstance(status.get("error"), Mapping):
        response["error"] = deepcopy(dict(status["error"]))
    return response


def _uncertain_status(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """把已持久化的提交不确定性投影为可唤醒的最小控制状态。"""

    if (
        record.get("phase") != UNKNOWN_JOB_STATUS
        and record.get("terminal_status") != UNKNOWN_JOB_STATUS
    ):
        return None
    status = {
        "job_id": record["job_id"],
        "status": UNKNOWN_JOB_STATUS,
    }
    if isinstance(record.get("submission_error"), Mapping):
        status["error"] = deepcopy(dict(record["submission_error"]))
    return status


def _mark_submission_uncertain(
    *,
    state: dict[str, Any],
    workspace: Path,
    record: dict[str, Any],
    error: Mapping[str, Any],
) -> None:
    """保存无法确认启动边界的原始错误上下文，并阻止自动重提。"""

    record["phase"] = UNKNOWN_JOB_STATUS
    record["terminal_status"] = UNKNOWN_JOB_STATUS
    record["submission_error"] = deepcopy(dict(error))
    state["jobs"][record["submission_key"]] = record
    _save_state(state, workspace)


def _uncertain_submission_error(
    record: Mapping[str, Any],
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    """返回服务端预分配的完整身份，使 Controller 能安全接管不确定提交。"""

    response = _error(code, message)
    if isinstance(record.get("submission_error"), Mapping):
        response["error"] = deepcopy(dict(record["submission_error"]))
    response.update(
        {
            "job_id": record["job_id"],
            "operation_id": record["operation_id"],
            "submission_key": record["submission_key"],
            "action_id": ACTION_ID,
            "backend_id": BACKEND_ID,
            "request_fingerprint": record["request_fingerprint"],
            "job_status": UNKNOWN_JOB_STATUS,
            "terminal": True,
            "phase": UNKNOWN_JOB_STATUS,
            "wake_armed": False,
        }
    )
    return response


def _submitted_terminal_failure(
    submitted: Mapping[str, Any],
) -> dict[str, Any] | None:
    """从带 job_id 的启动响应提取服务已明确报告的终态失败原因。

    ``invalid_request``、``unavailable`` 和 ``failed`` 都表示本次启动未成功完成；
    只要服务给出了结构化 error，就不得用本地白名单覆盖其 code、message 或附加字段。
    其它状态仍需回到不确定提交路径，避免把未知响应误写成失败。
    """

    raw_error = submitted.get("error")
    error = deepcopy(dict(raw_error)) if isinstance(raw_error, Mapping) else {}
    status = str(submitted.get("status") or "").casefold()
    if status in {"invalid_request", "unavailable", "failed", "error"}:
        return error or {
            "code": "execution_failed",
            "message": "TS Pipeline supervisor did not start the submitted computation",
        }
    return None


def _persist_terminal_failed_job(
    *,
    workspace: Path,
    scope: ScopeContext,
    registry: ExecutionJobRegistry,
    record: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    resource_limits: Mapping[str, Any],
    error: Mapping[str, Any],
) -> dict[str, Any] | None:
    """为确定性启动失败写入可查询 status，并把 registry 标为 failed。

    该函数只处理已预留且身份一致的 job_id。已有成功终态或路径身份冲突时返回 None，
    让上层继续按无法核对的提交处理；任何失败记录都保留作业目录和原始错误字段。
    """

    job_id = str(record.get("job_id") or "")
    directory = _job_directory_for_workspace(workspace, job_id)
    if directory is None:
        return None
    ensure_private_directory(directory, create=True)
    status_path = directory / "status.json"
    existing = _read_json_file(status_path)
    identity_fields = {
        "job_id": job_id,
        "scope_id": scope.scope_id,
        "scope_workspace": str(workspace),
    }
    if existing is not None:
        for field_name, expected in identity_fields.items():
            if existing.get(field_name) not in {None, expected}:
                return None
        existing_state = str(existing.get("status") or "").casefold()
        if existing_state in {"success", "timeout", "cancelled"}:
            return None
    status: dict[str, Any] = dict(existing or {})
    status.update(
        {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": JOB_TYPE,
            "status": "failed",
            "scope_id": scope.scope_id,
            "scope_workspace": str(workspace),
            "operation_id": record.get("operation_id"),
            "operation_fingerprint": record.get("operation_fingerprint"),
            "caller_request_fingerprint": record.get("request_fingerprint"),
            "submission_key": record.get("submission_key"),
            "resource_limits": deepcopy(dict(resource_limits)),
            "resource_allocation": status.get("resource_allocation", {}),
            "metadata": status.get(
                "metadata",
                {
                    "action_id": ACTION_ID,
                    "backend_id": BACKEND_ID,
                    "label": "transition_state_path_async",
                    "calculation_intent": "transition_state_path",
                    "transition_state_path_request": deepcopy(dict(request_payload)),
                },
            ),
            "submitted_at": status.get("submitted_at", record.get("submitted_at", _now())),
            "finished_at": _now(),
            "return_code": None,
            "error": deepcopy(dict(error)),
            "execution_mode": "detached_transition_state_path",
        }
    )
    _atomic_json(status_path, status)
    try:
        registry.mark(job_id, phase="failed")
    except (ExecutionJobRegistryError, ValueError):
        return None
    return status


def _accept_terminal_failed_submission(
    *,
    state: dict[str, Any],
    workspace: Path,
    registry: ExecutionJobRegistry,
    scope: ScopeContext,
    record: dict[str, Any],
    request_payload: Mapping[str, Any],
    resource_limits: Mapping[str, Any],
    error: Mapping[str, Any],
    idempotent_reused: bool,
) -> dict[str, Any] | None:
    """保存确定性 failed 状态并返回可查询的最小 job identity。"""

    status = _persist_terminal_failed_job(
        workspace=workspace,
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
    record["terminal_status"] = "failed"
    record["wake_armed"] = False
    record["failed_at"] = _now()
    state["jobs"][record["submission_key"]] = record
    _save_state(state, workspace)
    return _minimal_identity(
        record,
        status,
        idempotent_reused=idempotent_reused,
        phase="failed",
    )


def _gate_closed(record: Mapping[str, Any], *, operation: str) -> dict[str, Any]:
    """保留兼容的阶段门禁错误构造器；正常模型 get/collect 不再调用它。"""

    return _error(
        "transition_state_path_phase_gate_closed",
        f"{operation} 必须先在当前 turn 对同一 job_id 完成 get/collect",
        job_id=record.get("job_id"),
        operation_id=record.get("operation_id"),
        submission_key=record.get("submission_key"),
    )


def _submission_request_fingerprint(request: TransitionStatePathJobRequest) -> str:
    """计算不包含 scope 身份和模型资源选择的稳定科学请求指纹。"""

    value = request.model_dump(mode="json")
    value.pop("submission_key", None)
    value.pop("operation_id", None)
    value.pop("caller_request_fingerprint", None)
    # 资源由服务端固定，不属于模型选择；省略、默认值和旧请求值必须幂等。
    value["resource_limits"] = _effective_transition_state_path_resources(request)
    return calculate_request_fingerprint(ACTION_ID, value)


def _legacy_route_marker_submission_request_fingerprint(
    request: TransitionStatePathJobRequest,
) -> str:
    """计算旧版统一入口把隐藏 backend 标记计入的兼容指纹。"""

    value = request.model_dump(mode="json")
    value.pop("submission_key", None)
    value.pop("operation_id", None)
    value.pop("caller_request_fingerprint", None)
    value["resource_limits"] = _effective_transition_state_path_resources(request)
    value["requested_backend_id"] = (
        request.requested_backend_id
        if request.requested_backend_id is not None
        else BACKEND_ID
    )
    return calculate_request_fingerprint(ACTION_ID, value)


def _legacy_submission_request_fingerprint(
    request: TransitionStatePathJobRequest,
) -> str:
    """计算旧版本把调用方资源值纳入指纹的兼容身份。"""

    value = request.model_dump(mode="json")
    value.pop("submission_key", None)
    value.pop("operation_id", None)
    value.pop("caller_request_fingerprint", None)
    return calculate_request_fingerprint(ACTION_ID, value)


def _effective_transition_state_path_resources(
    request: TransitionStatePathJobRequest,
) -> dict[str, int]:
    """返回 TS Pipeline 提交实际使用的服务固定资源。"""

    del request
    return fixed_job_resource_limits()


def _validate_submit_request(request: TransitionStatePathJobRequest) -> dict[str, Any] | None:
    """执行异步入口的轻量合同检查，不运行科学模型。"""

    if request.backend_id != BACKEND_ID or (
        request.requested_backend_id is not None
        and request.requested_backend_id != BACKEND_ID
    ):
        return _error("unsupported_backend", f"TS Pipeline 只允许 {BACKEND_ID}")
    if request.method_spec:
        return _error(
            "invalid_request",
            "TS Pipeline 的 method_spec 由服务固定，必须为空；原科学算法未被修改",
        )
    if _effective_transition_state_path_resources(request)["gpu_count"] < 1:
        return _error(
            "gpu_required",
            "TS Pipeline 至少需要一个 GPU；请在服务启动配置中提供固定 GPU 资源",
        )
    return None


def _submit_transition_state_path_job_once(
    request: TransitionStatePathJobRequest,
) -> dict[str, Any]:
    """在一个 scope 锁内幂等登记并启动单个 TS Pipeline supervisor。"""

    scope = current_scope()
    if scope is None:
        return _error("scope_required", "异步 TS Pipeline 必须使用已认证 scope")
    identity_error = _scope_error(request)
    if identity_error is not None:
        return identity_error
    # 统一 Investigation submit 通过隐藏标记进入此专用 worker；没有该标记的
    # 旧专用工具调用继续保留历史 unknown 恢复语义，避免破坏旧 Session 收据。
    generic_submission = request.requested_backend_id is not None
    submission_key = validate_submission_key(str(request.submission_key))
    legacy_fingerprint = _legacy_submission_request_fingerprint(request)
    legacy_route_marker_fingerprint = (
        _legacy_route_marker_submission_request_fingerprint(request)
    )
    fingerprint = _submission_request_fingerprint(request)
    fixed_resource_limits = _effective_transition_state_path_resources(request)
    request_payload = request.model_dump(mode="json")
    # TS worker 读取的是 metadata 中的科学 request；资源不属于模型选择，必须与
    # supervisor 的固定评测预算完全一致，不能沿用模型请求或 Pydantic 默认值。
    request_payload["resource_limits"] = dict(fixed_resource_limits)
    operation_id = scoped_submission_operation_id(submission_key)
    workspace = scope.workspace

    with _state_lock(workspace):
        state, readable = _load_state(scope, workspace)
        if not readable:
            return _error(
                "transition_state_path_state_unavailable",
                "异步作业身份收据损坏或 scope 不匹配，已阻止重新提交",
            )
        if state is None:
            state = _empty_state(scope)
        try:
            registry = ExecutionJobRegistry(workspace=workspace, scope_id=scope.scope_id)
        except ExecutionJobRegistryError:
            return _error(
                "transition_state_path_state_unavailable",
                "统一 execution job registry 无法安全核对当前 scope",
            )
        existing = _job_record(state, submission_key)
        if existing is not None:
            stored_fingerprint = str(existing.get("request_fingerprint") or "")
            if stored_fingerprint not in {
                legacy_fingerprint,
                legacy_route_marker_fingerprint,
                fingerprint,
            }:
                return _error(
                    "submission_key_conflict",
                    "同一 scope/submission_key 已绑定不同的 TS Pipeline 请求",
                    job_id=existing.get("job_id"),
                    operation_id=existing.get("operation_id"),
                )
            # 旧作业保留其历史指纹；新作业统一使用忽略资源选择的有效指纹。
            fingerprint = stored_fingerprint
            try:
                reservation = registry.reserve(
                    job_type=JOB_TYPE,
                    operation_id=operation_id,
                    operation_fingerprint=fingerprint,
                    submission_key=submission_key,
                    caller_request_fingerprint=fingerprint,
                    preferred_job_id=str(existing.get("job_id") or ""),
                )
            except ExecutionJobRegistryConflict as exc:
                conflict_record = getattr(exc, "record", None)
                return _error(
                    "transition_state_path_submission_conflict",
                    "同一 operation 已绑定其他 execution job，未把 generic conflict 标记为 TS accepted",
                    job_id=(conflict_record or {}).get("job_id") if isinstance(conflict_record, Mapping) else existing.get("job_id"),
                    operation_id=operation_id,
                )
            except (ExecutionJobRegistryError, ValueError):
                return _error(
                    "transition_state_path_state_unavailable",
                    "已有 TS 作业身份无法写入统一 execution job registry",
                    job_id=existing.get("job_id"),
                )
            if reservation.job_id != str(existing.get("job_id") or ""):
                return _error(
                    "transition_state_path_identity_conflict",
                    "TS state 与统一 execution job registry 的 job_id 不一致",
                    job_id=existing.get("job_id"),
                    operation_id=operation_id,
                )
            status = _read_job_status_for_workspace(
                workspace,
                scope.scope_id,
                str(existing["job_id"]),
            )
            if existing.get("phase") in {"submitting", "unknown"} and status is None:
                existing["phase"] = UNKNOWN_JOB_STATUS
                state["jobs"][submission_key] = existing
                _save_state(state, workspace)
                return _uncertain_submission_error(
                    existing,
                    code="transition_state_path_submission_uncertain",
                    message="已有提交身份但作业状态无法核对，已阻止重新提交",
                )
            if existing.get("phase") == "submitting" and status is not None:
                existing["phase"] = "accepted"
                state["jobs"][submission_key] = existing
                _save_state(state, workspace)
            return _minimal_identity(
                existing,
                status,
                idempotent_reused=True,
                phase=str(existing.get("phase") or "accepted"),
            )

        active_key = state.get("active_submission_key")
        if active_key is not None:
            active = _job_record(state, str(active_key))
            if active is None:
                return _error(
                    "transition_state_path_state_unavailable",
                    "scope 活动作业记录不完整，已阻止新的计算提交",
                )
            if active.get("phase") != "collected":
                return _error(
                    "transition_state_path_phase_gate_closed",
                    "当前 scope 已有未收集的 TS Pipeline 作业；必须先完成该作业的 get/collect 后才能提交下一作业",
                    job_id=active.get("job_id"),
                    operation_id=active.get("operation_id"),
                    submission_key=active.get("submission_key"),
                )
            state["active_submission_key"] = None
            state["active_job_id"] = None

        try:
            reservation = registry.reserve(
                job_type=JOB_TYPE,
                operation_id=operation_id,
                operation_fingerprint=fingerprint,
                submission_key=submission_key,
                caller_request_fingerprint=fingerprint,
            )
        except ExecutionJobRegistryConflict as exc:
            conflict_record = getattr(exc, "record", None)
            return _error(
                "transition_state_path_submission_conflict",
                "同一 operation 已绑定其他 execution job，未把 generic conflict 标记为 TS accepted",
                job_id=(conflict_record or {}).get("job_id") if isinstance(conflict_record, Mapping) else None,
                operation_id=operation_id,
            )
        except (ExecutionJobRegistryError, ValueError):
            return _error(
                "transition_state_path_state_unavailable",
                "TS 作业身份无法安全预留，未启动 supervisor",
                operation_id=operation_id,
            )
        job_id = reservation.job_id
        record: dict[str, Any] = {
            "job_id": job_id,
            "operation_id": operation_id,
            "submission_key": submission_key,
            "request_fingerprint": fingerprint,
            "operation_fingerprint": fingerprint,
            "job_type": JOB_TYPE,
            "scope_id": scope.scope_id,
            "scope_workspace": str(workspace),
            "phase": "submitting",
            "wake_armed": False,
            "submitted_at": _now(),
        }
        state["jobs"][submission_key] = record
        state["active_submission_key"] = submission_key
        state["active_job_id"] = job_id
        _save_state(state, workspace)

        async_resource_limits = dict(fixed_resource_limits)
        async_resource_limits["walltime_seconds"] = timeout_seconds_for("compute")
        request_error = _validate_submit_request(request)
        if request_error is not None:
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=request_error.get("error", {
                    "code": "transition_state_path_invalid_request",
                    "message": "TS Pipeline 请求校验失败",
                }),
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="TS Pipeline 请求失败但 terminal failed 状态无法安全持久化，已保留 job_id",
            )
        try:
            submitted = _start_job(
                job_id=job_id,
                job_type=JOB_TYPE,
                runtime="ash_neb",
                command=[
                    str(__import__("minichem_toolbox.runtime", fromlist=["runtime_python"]).runtime_python("ash_neb")),
                    "-m",
                    WORKER_MODULE,
                    "--worker",
                ],
                stdin_target=None,
                staged_inputs=[],
                resource_limits=async_resource_limits,
                metadata={
                    "action_id": ACTION_ID,
                    "backend_id": BACKEND_ID,
                    "label": "transition_state_path_async",
                    "calculation_intent": "transition_state_path",
                    "transition_state_path_request": request_payload,
                    "caller_transition_state_path_request": request.model_dump(mode="json"),
                    "catalog_hash": active_catalog_hash(),
                    "execution_timeout_policy": timeout_policy_record("compute"),
                },
                operation_id=operation_id,
                operation_fingerprint=fingerprint,
                submission_key=submission_key,
                caller_request_fingerprint=fingerprint,
            )
        except ResourceBudgetExceeded as exc:
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=exc.as_error(),
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error={
                    "code": "transition_state_path_submission_uncertain",
                    "message": "资源失败身份无法安全持久化，已保留提交身份并阻止重新提交",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="资源失败身份无法安全持久化，已保留提交身份并阻止重新提交",
            )
        except ResourceStateUnavailable as exc:
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=exc.as_error(),
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error={
                    "code": "transition_state_path_submission_uncertain",
                    "message": "资源状态失败身份无法安全持久化，已保留提交身份并阻止重新提交",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="资源状态失败身份无法安全持久化，已保留提交身份并阻止重新提交",
            )
        except Exception as exc:
            if not generic_submission:
                legacy_error = {
                    "code": "transition_state_path_submission_uncertain",
                    "message": "作业启动过程无法确认，已保留持久身份并阻止重新提交",
                    "exception_type": type(exc).__name__,
                    "exception_message": _sanitize_public_diagnostic_value(str(exc)),
                }
                _mark_submission_uncertain(
                    state=state,
                    workspace=workspace,
                    record=record,
                    error=legacy_error,
                )
                return _uncertain_submission_error(
                    record,
                    code="transition_state_path_submission_uncertain",
                    message="作业启动过程无法确认，已保留持久身份并阻止重新提交",
                )
            failure_error = {
                "code": "transition_state_path_submission_failed",
                "message": "TS Pipeline supervisor 启动失败",
                "exception_type": type(exc).__name__,
                "exception_message": _sanitize_public_diagnostic_value(str(exc)),
            }
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=failure_error,
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error=failure_error,
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="TS Pipeline supervisor 启动失败且 terminal failed 状态无法安全持久化，已保留 job_id",
            )

        if not isinstance(submitted, Mapping):
            if not generic_submission:
                legacy_error = {
                    "code": "transition_state_path_submission_uncertain",
                    "message": "服务未返回可核对的结构化提交响应，已保留提交身份并阻止重新提交",
                    "response_type": type(submitted).__name__,
                }
                _mark_submission_uncertain(
                    state=state,
                    workspace=workspace,
                    record=record,
                    error=legacy_error,
                )
                return _uncertain_submission_error(
                    record,
                    code="transition_state_path_submission_uncertain",
                    message="服务未返回可核对的 job_id，已保留提交身份并阻止重新提交",
                )
            failure_error = {
                "code": "transition_state_path_invalid_supervisor_response",
                "message": "supervisor 未返回可核对的 JSON object",
                "response_type": type(submitted).__name__,
            }
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=failure_error,
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error=failure_error,
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="supervisor 响应无法安全保存为 failed，已保留 job_id",
            )
        if not submitted.get("job_id"):
            raw_error = submitted.get("error")
            failure_error = (
                deepcopy(dict(raw_error))
                if isinstance(raw_error, Mapping)
                else {
                    "code": "transition_state_path_missing_job_id",
                    "message": "supervisor 未返回可核对的 job_id",
                }
            )
            failure_error.setdefault("response_status", submitted.get("status"))
            if not generic_submission:
                if not isinstance(raw_error, Mapping):
                    failure_error = {
                        "code": "transition_state_path_submission_uncertain",
                        "message": "服务未返回可核对的 job_id，已保留提交身份并阻止重新提交",
                        "response_status": submitted.get("status"),
                    }
                _mark_submission_uncertain(
                    state=state,
                    workspace=workspace,
                    record=record,
                    error=failure_error,
                )
                return _uncertain_submission_error(
                    record,
                    code="transition_state_path_submission_uncertain",
                    message="服务未返回可核对的 job_id，已保留提交身份并阻止重新提交",
                )
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=failure_error,
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error=failure_error,
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="缺少 job_id 且 terminal failed 状态无法安全持久化，已保留 job_id",
            )
        if str(submitted.get("job_id")) != job_id:
            raw_error = submitted.get("error")
            failure_error: dict[str, Any] = (
                deepcopy(dict(raw_error))
                if isinstance(raw_error, Mapping)
                else {
                    "code": "transition_state_path_identity_conflict",
                    "message": "supervisor 返回的 job_id 与持久提交身份不一致",
                }
            )
            failure_error.setdefault("response_job_id", submitted.get("job_id"))
            if not generic_submission:
                _mark_submission_uncertain(
                    state=state,
                    workspace=workspace,
                    record=record,
                    error=failure_error,
                )
                return _uncertain_submission_error(
                    record,
                    code="transition_state_path_identity_conflict",
                    message="服务返回的 job_id 与持久提交身份不一致，已停止恢复",
                )
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=failure_error,
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error=failure_error,
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="job_id 身份冲突且 terminal failed 状态无法安全持久化，已保留 job_id",
            )
        deterministic_failure = _submitted_terminal_failure(submitted)
        if deterministic_failure is not None:
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=deterministic_failure,
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error={
                    **deepcopy(deterministic_failure),
                    "response_status": submitted.get("status"),
                },
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="确定性失败身份无法安全持久化，已保留提交身份并阻止重新提交",
            )
        if submitted.get("status") != "success":
            raw_error = submitted.get("error")
            failure_error = (
                deepcopy(dict(raw_error))
                if isinstance(raw_error, Mapping)
                else {
                    "code": "transition_state_path_submission_failed",
                    "message": "统一 execution job registry 或 supervisor 返回非成功结果",
                }
            )
            failure_error.setdefault("response_status", submitted.get("status"))
            if not generic_submission:
                _mark_submission_uncertain(
                    state=state,
                    workspace=workspace,
                    record=record,
                    error=failure_error,
                )
                return _uncertain_submission_error(
                    record,
                    code="transition_state_path_submission_uncertain",
                    message="统一 execution job registry 或 supervisor 返回非成功结果，未标记 TS accepted",
                )
            failure = _accept_terminal_failed_submission(
                state=state,
                workspace=workspace,
                registry=registry,
                scope=scope,
                record=record,
                request_payload=request_payload,
                resource_limits=async_resource_limits,
                error=failure_error,
                idempotent_reused=False,
            )
            if failure is not None:
                return failure
            _mark_submission_uncertain(
                state=state,
                workspace=workspace,
                record=record,
                error=failure_error,
            )
            return _uncertain_submission_error(
                record,
                code="transition_state_path_submission_uncertain",
                message="非成功响应且 terminal failed 状态无法安全持久化，已保留 job_id",
            )
        record["phase"] = "accepted"
        record["wake_armed"] = False
        record["accepted_at"] = _now()
        state["jobs"][submission_key] = record
        _save_state(state, workspace)
        status = _read_job_status_for_workspace(workspace, scope.scope_id, job_id)
        return _minimal_identity(record, status, idempotent_reused=False)


def submit_transition_state_path_job(request: TransitionStatePathJobRequest) -> dict[str, Any]:
    """快速返回一个持久 TS Pipeline job_id，供当前 turn 继续 get/collect。"""

    arguments = {"request": request.model_dump(mode="json")}
    return execute_traced(
        "submit_transition_state_path_job",
        arguments,
        lambda: _submit_transition_state_path_job_once(request),
        capture_artifacts=False,
    )


def _lookup_model_record(
    request: TransitionStatePathJobStatusRequest,
    *,
    operation: str,
) -> tuple[ScopeContext, Path, dict[str, Any], dict[str, Any]] | dict[str, Any]:
    """仅按模型提供的 job_id 查找当前 scope 记录，不要求模型携带内部身份。"""

    scope = current_scope()
    if scope is None:
        return _error("scope_required", "TS Pipeline 作业必须在已认证 scope 中查询")
    workspace = scope.workspace
    state, readable = _load_state(scope, workspace)
    if not readable or state is None:
        return _error(
            "transition_state_path_state_unavailable",
            "异步作业身份收据不可核对",
        )
    records = [
        dict(value)
        for value in state["jobs"].values()
        if isinstance(value, Mapping) and value.get("job_id") == request.job_id
    ]
    if len(records) != 1:
        return _error(
            "transition_state_path_job_unavailable",
            "该 job_id 不属于当前 capability scope 或身份记录不唯一",
        )
    record = records[0]
    return scope, workspace, state, record


def _get_transition_state_path_job_once(
    request: TransitionStatePathJobStatusRequest,
) -> dict[str, Any]:
    """返回模型已提交作业的最小状态和身份，不暴露科学 payload。"""

    with _state_lock(current_scope().workspace if current_scope() is not None else workspace_root()):
        found = _lookup_model_record(request, operation="get")
        if isinstance(found, dict):
            return found
        _scope, workspace, _state, record = found
        try:
            registry = ExecutionJobRegistry(workspace=workspace, scope_id=_scope.scope_id)
            reservation = registry.reserve(
                job_type=JOB_TYPE,
                operation_id=str(record.get("operation_id") or ""),
                operation_fingerprint=str(
                    record.get("operation_fingerprint") or record.get("request_fingerprint") or ""
                ),
                submission_key=str(record.get("submission_key") or ""),
                caller_request_fingerprint=str(record.get("request_fingerprint") or ""),
                preferred_job_id=str(record.get("job_id") or ""),
            )
        except ExecutionJobRegistryConflict:
            return _error(
                "transition_state_path_submission_conflict",
                "TS 作业与统一 execution job registry 身份不一致，已停止查询",
                job_id=record.get("job_id"),
            )
        except (ExecutionJobRegistryError, ValueError):
            return _error(
                "transition_state_path_state_unavailable",
                "TS 作业身份无法在统一 execution job registry 中核对",
                job_id=record.get("job_id"),
            )
        if reservation.job_id != str(record.get("job_id") or ""):
            return _error(
                "transition_state_path_identity_conflict",
                "TS state 与统一 execution job registry 的 job_id 不一致",
                job_id=record.get("job_id"),
            )
        status = _read_job_status_for_workspace(workspace, _scope.scope_id, str(record["job_id"]))
        if status is None:
            status = _uncertain_status(record)
            if status is None:
                return _error(
                    "transition_state_path_job_unavailable",
                    "作业状态无法核对，已停止恢复",
                    job_id=record["job_id"],
                )
        raw_status = status.get("status")
        if raw_status in {"queued", "running", *TERMINAL_STATES}:
            try:
                registry_record = registry.get(str(record["job_id"]))
                if registry_record is None or registry_record.get("phase") != "collected":
                    registry.mark(str(record["job_id"]), phase=str(raw_status))
            except (ExecutionJobRegistryError, ValueError):
                pass
        return _minimal_identity(record, status, idempotent_reused=True)


def get_transition_state_path_job(request: TransitionStatePathJobStatusRequest) -> dict[str, Any]:
    """返回模型已提交 TS Pipeline 作业的最小状态。"""

    return execute_traced(
        "get_transition_state_path_job",
        {"request": request.model_dump(mode="json")},
        lambda: _get_transition_state_path_job_once(request),
        capture_artifacts=False,
    )


def _read_transition_state_path_result(workspace: Path, job_id: str) -> dict[str, Any] | None:
    """读取 worker 写入的 canonical ActionResult 并进行模型校验。"""

    directory = _job_directory_for_workspace(workspace, job_id)
    if directory is None:
        return None
    raw = _read_json_file(directory / RESULT_FILENAME)
    if raw is None:
        return None
    try:
        result = ActionResult.model_validate(raw).model_dump(mode="json")
    except Exception:
        return None
    if result.get("action") != ACTION_ID or result.get("backend") != BACKEND_ID:
        return None
    return result


def _safe_relative_workspace_file(workspace: Path, value: Any) -> Path | None:
    """把相对 workspace 文件解析为普通文件，并拒绝越界或符号链接。"""

    if not isinstance(value, str) or not value.strip() or "\\" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts:
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    candidate = workspace.joinpath(*relative.parts)
    cursor = workspace
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except (FileNotFoundError, OSError):
            return None
        if stat.S_ISLNK(metadata.st_mode):
            return None
    try:
        if not candidate.is_file() or candidate.is_symlink():
            return None
    except OSError:
        return None
    return candidate


def _read_transition_state_path_request(
    workspace: Path,
    record: Mapping[str, Any],
) -> TransitionStatePathJobRequest | None:
    """从 supervisor request 收据恢复原科学请求并核对提交指纹。"""

    directory = _job_directory_for_workspace(workspace, str(record.get("job_id") or ""))
    if directory is None:
        return None
    request_record = _read_json_file(directory / "request.json")
    if request_record is None:
        return None
    if (
        request_record.get("job_id") != record.get("job_id")
        or request_record.get("scope_id") != record.get("scope_id")
        or request_record.get("scope_workspace") != record.get("scope_workspace")
        or request_record.get("operation_id") != record.get("operation_id")
        or request_record.get("submission_key") != record.get("submission_key")
    ):
        return None
    metadata = request_record.get("metadata")
    request_value = metadata.get("transition_state_path_request") if isinstance(metadata, Mapping) else None
    if not isinstance(request_value, Mapping):
        return None
    caller_request_value = (
        metadata.get("caller_transition_state_path_request")
        if isinstance(metadata, Mapping)
        else None
    )
    try:
        request = TransitionStatePathJobRequest.model_validate(request_value)
        has_caller_request = isinstance(caller_request_value, Mapping)
        caller_request = TransitionStatePathJobRequest.model_validate(
            caller_request_value if has_caller_request else request_value
        )
    except Exception:
        return None
    request_fingerprints = {
        _submission_request_fingerprint(caller_request),
        _legacy_submission_request_fingerprint(caller_request),
        _legacy_route_marker_submission_request_fingerprint(caller_request),
    }
    if record.get("request_fingerprint") not in request_fingerprints:
        return None
    if request.submission_key != record.get("submission_key"):
        return None
    return request


def _prefix_transition_state_path_artifacts(
    raw_worker_result: Mapping[str, Any],
    *,
    workspace: Path,
    job_id: str,
) -> dict[str, Any] | None:
    """把私有 worker 文件映射到父 scope 的作业目录，不修改科学结果值。"""

    artifact_files = raw_worker_result.get("artifact_files", [])
    if artifact_files is None:
        artifact_files = []
    if not isinstance(artifact_files, (list, tuple)):
        return None
    normalized = deepcopy(dict(raw_worker_result))
    entries: list[dict[str, Any]] = []
    for item in artifact_files:
        if not isinstance(item, Mapping):
            return None
        raw_path = item.get("path")
        raw_relative = PurePosixPath(raw_path) if isinstance(raw_path, str) else None
        if (
            raw_relative is None
            or raw_relative.is_absolute()
            or not raw_relative.parts
            or "\\" in str(raw_path)
            or any(part in {"", ".", ".."} for part in raw_relative.parts)
        ):
            return None
        parent_relative = PurePosixPath("outputs", "execution_jobs", job_id)
        target_relative = parent_relative.joinpath(*raw_relative.parts)
        target = _safe_relative_workspace_file(workspace, str(target_relative))
        if target is None:
            return None
        entry = dict(item)
        entry["path"] = str(target_relative)
        entries.append(entry)
    normalized["artifact_files"] = entries
    return normalized


def _restore_safe_ts_worker_exception_message(
    raw_worker_result: Mapping[str, Any],
    worker: dict[str, Any],
) -> None:
    """恢复 TS worker 的有界异常消息，并继续沿用路径与敏感字段清理。"""

    raw_error = raw_worker_result.get("error")
    if not isinstance(raw_error, Mapping):
        return
    exception_message = raw_error.get("exception_message")
    if not isinstance(exception_message, str):
        return
    public_error = worker.get("error")
    if not isinstance(public_error, dict):
        return
    safe_message = _sanitize_public_diagnostic_value(exception_message)
    if isinstance(safe_message, str):
        public_error["exception_message"] = safe_message


def _build_parent_transition_state_path_result(
    raw_worker_result: Mapping[str, Any],
    request: TransitionStatePathJobRequest,
    *,
    workspace: Path,
    job_id: str,
) -> dict[str, Any]:
    """在父 scope 登记 worker 文件，并封装唯一 canonical ActionResult。"""

    normalized = _prefix_transition_state_path_artifacts(
        raw_worker_result,
        workspace=workspace,
        job_id=job_id,
    )
    if normalized is None:
        raise ValueError("TS Pipeline worker artifact_files 不是安全的相对文件清单")
    plan = _ActionExecutionPlan(
        action_id=ACTION_ID,
        request=request,
        specification=action_specs()[ACTION_ID],
        backend_id=BACKEND_ID,
        backend=backend_specs()[BACKEND_ID],
        component_specs=(),
        selection_source="agent",
        execution_timeout=timeout_seconds_for("compute"),
        timeout_policy=timeout_policy_record("compute"),
        resource_references={},
        input_artifacts=collect_artifact_refs(request.inputs),
    )
    worker, status, result_payload = _normalize_action_worker_result(normalized)
    # 通用 worker 清理默认隐藏 exception_message；TS detached worker 明确提供的
    # 单行异常原因是模型修正请求所需事实，恢复时只保留已脱敏、有界的文本，不引入 traceback。
    _restore_safe_ts_worker_exception_message(raw_worker_result, worker)
    status, result_payload, output_artifacts, worker = _collect_action_result_artifacts(
        plan,
        worker,
        status,
        result_payload,
    )
    action_result = _build_action_result(
        plan,
        worker,
        status,
        result_payload,
        output_artifacts,
        {
            "async_job_id": job_id,
            "execution_mode": "detached_transition_state_path",
            "resource_budget": resource_budget_record(),
        },
    )
    return ActionResult.model_validate(action_result).model_dump(mode="json")


def _validate_parent_transition_state_path_artifacts(
    action_result: Mapping[str, Any],
    *,
    workspace: Path,
) -> None:
    """核对父 scope ArtifactStore 中每个结果引用的归属、路径和 SHA-256。"""

    references = action_result.get("output_artifacts")
    if not isinstance(references, list):
        raise ValueError("ActionResult.output_artifacts 必须是数组")
    if not references:
        return
    store = ArtifactStore(root=workspace, create=False)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for value in references:
        reference = ArtifactRef.model_validate(value)
        if reference.artifact_id in seen_ids or reference.path in seen_paths:
            raise ValueError("ActionResult.output_artifacts 存在重复 artifact 身份")
        seen_ids.add(reference.artifact_id)
        seen_paths.add(reference.path)
        if reference.producer_action != ACTION_ID or reference.producer_backend != BACKEND_ID:
            raise ValueError("ActionResult artifact 不属于当前 TS Pipeline")
        if re.fullmatch(r"art_[0-9a-f]{32}", reference.artifact_id) is None:
            raise ValueError("ActionResult artifact_id 不是合法身份")
        canonical = store.find(reference.artifact_id)
        if canonical.model_dump(mode="json") != reference.model_dump(mode="json"):
            raise ValueError("ActionResult artifact 与父 scope manifest 不一致")
        path = _safe_relative_workspace_file(workspace, reference.path)
        if path is None:
            raise ValueError("ActionResult artifact path 不属于父 scope")
        _, digest = read_file_snapshot(path)
        if digest != reference.sha256:
            raise ValueError("ActionResult artifact SHA-256 与文件内容不一致")


def _collect_transition_state_path_job_once(
    request: TransitionStatePathJobCollectRequest,
) -> dict[str, Any]:
    """收集已终态 TS Pipeline 的 ActionResult，并在父 scope 登记文件引用。"""

    current = current_scope()
    if current is None:
        return _error("scope_required", "TS Pipeline 作业必须在已认证 scope 中收集")
    with _state_lock(current.workspace):
        found = _lookup_model_record(request, operation="collect")
        if isinstance(found, dict):
            return found
        scope, workspace, state, record = found
        try:
            registry = ExecutionJobRegistry(workspace=workspace, scope_id=scope.scope_id)
            reservation = registry.reserve(
                job_type=JOB_TYPE,
                operation_id=str(record.get("operation_id") or ""),
                operation_fingerprint=str(
                    record.get("operation_fingerprint") or record.get("request_fingerprint") or ""
                ),
                submission_key=str(record.get("submission_key") or ""),
                caller_request_fingerprint=str(record.get("request_fingerprint") or ""),
                preferred_job_id=str(record.get("job_id") or ""),
            )
        except ExecutionJobRegistryConflict as exc:
            conflict_record = getattr(exc, "record", None)
            return _error(
                "transition_state_path_submission_conflict",
                "TS 作业与统一 execution job registry 身份不一致，已停止收集",
                job_id=(conflict_record or {}).get("job_id") if isinstance(conflict_record, Mapping) else record.get("job_id"),
            )
        except (ExecutionJobRegistryError, ValueError):
            return _error(
                "transition_state_path_state_unavailable",
                "TS 作业身份无法在统一 execution job registry 中核对",
                job_id=record.get("job_id"),
            )
        if reservation.job_id != str(record.get("job_id") or ""):
            return _error(
                "transition_state_path_identity_conflict",
                "TS state 与统一 execution job registry 的 job_id 不一致",
                job_id=record.get("job_id"),
            )
        status = _read_job_status_for_workspace(workspace, scope.scope_id, str(record["job_id"]))
        if status is None:
            return _error(
                "transition_state_path_job_unavailable",
                "作业状态无法核对，已停止收集",
                job_id=record["job_id"],
            )
        if status.get("status") not in TERMINAL_STATES:
            return _error(
                "transition_state_path_job_not_terminal",
                "作业尚未进入终态；按该作业轮询节奏再次调用 get",
                job_id=record["job_id"],
                job_status=status.get("status"),
            )
        action_result = _read_transition_state_path_result(workspace, str(record["job_id"]))
        terminal_status = str(status.get("status") or "")
        if action_result is None and terminal_status in {"failed", "timeout", "cancelled"}:
            failure_error = status.get("error")
            if not isinstance(failure_error, Mapping):
                failure_error = {
                    "code": f"execution_{terminal_status}",
                    "message": "TS Pipeline 作业已结束，但没有可收集的科学结果",
                }
            try:
                registry.mark(str(record["job_id"]), phase="collected")
            except (ExecutionJobRegistryError, ValueError):
                return _error(
                    "transition_state_path_state_unavailable",
                    "终态作业已核对但统一 registry 未能记录 collected，保留作业以便重试",
                    job_id=record["job_id"],
                )
            collection = {
                "status": "success",
                "ready": True,
                "job_id": record["job_id"],
                "job_status": terminal_status,
                "operation_id": record["operation_id"],
                "submission_key": record["submission_key"],
                "action_id": ACTION_ID,
                "backend_id": BACKEND_ID,
                "error": deepcopy(dict(failure_error)),
                "automatic_fallback": False,
            }
            record["phase"] = "collected"
            record["wake_armed"] = False
            record["collected_at"] = _now()
            state["jobs"][record["submission_key"]] = record
            if state.get("active_submission_key") == record["submission_key"]:
                state["active_submission_key"] = None
                state["active_job_id"] = None
            _save_state(state, workspace)
            return collection
        if action_result is None:
            raw_result = _read_json_file(
                _job_directory_for_workspace(workspace, str(record["job_id"]))
                / RAW_RESULT_FILENAME
            ) if _job_directory_for_workspace(workspace, str(record["job_id"])) is not None else None
            transition_request = _read_transition_state_path_request(workspace, record)
            if raw_result is None or transition_request is None:
                return _error(
                    "transition_state_path_result_unavailable",
                    "作业已经终态但原始 worker 结果或请求身份尚未安全落盘，已停止收集",
                    job_id=record["job_id"],
                )
            try:
                action_result = _build_parent_transition_state_path_result(
                    raw_result,
                    transition_request,
                    workspace=workspace,
                    job_id=str(record["job_id"]),
                )
                _validate_parent_transition_state_path_artifacts(
                    action_result,
                    workspace=workspace,
                )
                directory = _job_directory_for_workspace(workspace, str(record["job_id"]))
                if directory is None:
                    raise ValueError("TS Pipeline job directory is invalid")
                _atomic_json(directory / RESULT_FILENAME, action_result)
            except Exception as exc:
                return _error(
                    "transition_state_path_result_unavailable",
                    "原始 TS Pipeline 结果无法在父 scope 中安全登记，已停止收集",
                    job_id=record["job_id"],
                    reason_code="artifact_registration_failed",
                    exception_type=type(exc).__name__,
                )
        else:
            try:
                _validate_parent_transition_state_path_artifacts(
                    action_result,
                    workspace=workspace,
                )
            except Exception as exc:
                return _error(
                    "transition_state_path_result_unavailable",
                    "已落盘 ActionResult 的 artifact manifest 无法核对，已停止收集",
                    job_id=record["job_id"],
                    reason_code="artifact_manifest_invalid",
                    exception_type=type(exc).__name__,
                )
        try:
            registry.mark(str(record["job_id"]), phase="collected")
        except (ExecutionJobRegistryError, ValueError):
            return _error(
                "transition_state_path_state_unavailable",
                "canonical ActionResult 已核对但统一 registry 未能记录 collected，保留作业以便重试",
                job_id=record["job_id"],
            )
        collection = {
            "status": "success",
            "ready": True,
            "job_id": record["job_id"],
            "job_status": status.get("status"),
            "operation_id": record["operation_id"],
            "submission_key": record["submission_key"],
            "action_id": ACTION_ID,
            "backend_id": BACKEND_ID,
            "action_result": action_result,
            "automatic_fallback": False,
        }
        record["phase"] = "collected"
        record["wake_armed"] = False
        record["collected_at"] = _now()
        state["jobs"][record["submission_key"]] = record
        if state.get("active_submission_key") == record["submission_key"]:
            state["active_submission_key"] = None
            state["active_job_id"] = None
        _save_state(state, workspace)
        return collection


def collect_transition_state_path_job(
    request: TransitionStatePathJobCollectRequest,
) -> dict[str, Any]:
    """返回终态 TS Pipeline 科学结果和 ArtifactRef。"""

    return execute_traced(
        "collect_transition_state_path_job",
        {"request": request.model_dump(mode="json")},
        lambda: _collect_transition_state_path_job_once(request),
        capture_artifacts=False,
    )


def _controller_identity(
    *,
    scope_id: str | None,
    workspace: str | os.PathLike[str] | Path | None,
) -> tuple[str, Path] | dict[str, Any]:
    """解析 Controller 显式 scope 身份；不依赖可被模型伪造的请求字段。"""

    current = current_scope()
    if current is not None:
        selected_scope = current.scope_id if scope_id is None else str(scope_id)
        selected_workspace = current.workspace if workspace is None else Path(workspace).expanduser().resolve()
        if selected_scope != current.scope_id or selected_workspace != current.workspace:
            return _error("scope_identity_conflict", "Controller scope 身份与当前认证 scope 不一致")
    else:
        if scope_id is None or workspace is None:
            return _error("scope_required", "Controller arm 必须显式提供 scope_id 和 workspace")
        selected_scope = str(scope_id).strip()
        selected_workspace = Path(workspace).expanduser().resolve()
    if not selected_scope:
        return _error("scope_required", "Controller scope_id 不能为空")
    try:
        ensure_private_directory(selected_workspace, create=False)
    except (OSError, RuntimeError, ValueError):
        return _error("scope_identity_unavailable", "Controller workspace 不是可核对的私有目录")
    return selected_scope, selected_workspace


def _controller_get_job_status(
    job_id: str,
    *,
    scope_id: str | None = None,
    workspace: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """Controller 轮询作业终态，返回最小身份而不 arm wake。"""

    if JOB_ID_PATTERN.fullmatch(str(job_id)) is None:
        return _error("invalid_job_id", "job_id 不是有效的 MiniChem 作业身份")
    identity = _controller_identity(scope_id=scope_id, workspace=workspace)
    if isinstance(identity, dict):
        return identity
    selected_scope, selected_workspace = identity
    state_root = _scope_state_root(selected_workspace)
    del state_root
    with _state_lock(selected_workspace):
        raw = _read_json_file(_state_path(selected_workspace))
        state = _validate_state(raw, scope_id=selected_scope, workspace=selected_workspace)
        if state is None:
            return _error("transition_state_path_state_unavailable", "Controller 无法核对异步作业身份")
        matching = None
        for value in state["jobs"].values():
            if isinstance(value, Mapping) and value.get("job_id") == job_id:
                matching = dict(value)
                break
        if matching is None:
            return _error("transition_state_path_job_unavailable", "Controller 无法核对该 job_id")
        status = _read_job_status_for_workspace(selected_workspace, selected_scope, job_id)
        if status is None:
            status = _uncertain_status(matching)
            if status is None:
                return _error(
                    "transition_state_path_job_unavailable",
                    "Controller 无法核对作业状态，已禁止重新提交",
                    job_id=job_id,
                )
        return _minimal_identity(matching, status, idempotent_reused=True)


def _controller_arm_job(
    job_id: str,
    *,
    scope_id: str | None = None,
    workspace: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """Controller 确认终态并原子开启同一 job 的模型 wake。"""

    if JOB_ID_PATTERN.fullmatch(str(job_id)) is None:
        return _error("invalid_job_id", "job_id 不是有效的 MiniChem 作业身份")
    identity = _controller_identity(scope_id=scope_id, workspace=workspace)
    if isinstance(identity, dict):
        return identity
    selected_scope, selected_workspace = identity
    with _state_lock(selected_workspace):
        raw = _read_json_file(_state_path(selected_workspace))
        state = _validate_state(raw, scope_id=selected_scope, workspace=selected_workspace)
        if state is None:
            return _error("transition_state_path_state_unavailable", "Controller 无法核对异步作业身份")
        matching_key = next(
            (
                key
                for key, value in state["jobs"].items()
                if isinstance(value, Mapping) and value.get("job_id") == job_id
            ),
            None,
        )
        if matching_key is None:
            return _error("transition_state_path_job_unavailable", "Controller 无法核对该 job_id")
        record = dict(state["jobs"][matching_key])
        status = _read_job_status_for_workspace(selected_workspace, selected_scope, job_id)
        if status is None:
            status = _uncertain_status(record)
            if status is None:
                return _error("transition_state_path_job_unavailable", "作业状态无法核对，已停止 arm")
        uncertain_terminal = (
            status.get("status") == UNKNOWN_JOB_STATUS
            and (
                record.get("phase") == UNKNOWN_JOB_STATUS
                or record.get("terminal_status") == UNKNOWN_JOB_STATUS
            )
        )
        if status.get("status") not in TERMINAL_STATES and not uncertain_terminal:
            return _error(
                "transition_state_path_job_not_terminal",
                "Controller 只能对终态作业 arm wake",
                job_id=job_id,
                job_status=status.get("status"),
            )
        if record.get("phase") == "collected":
            return _minimal_identity(record, status, idempotent_reused=True)
        record["phase"] = "armed"
        record["wake_armed"] = True
        record["terminal_status"] = status.get("status")
        record["wake_armed_at"] = _now()
        state["jobs"][matching_key] = record
        _save_state(state, selected_workspace)
        return _minimal_identity(record, status, idempotent_reused=True, phase="armed")


def _controller_scope_context(scope_id: str, workspace: Path) -> ScopeContext:
    """为没有 MCP ContextVar 的可信 Controller 构造只读 scope 视图。"""

    current = current_scope()
    if current is not None:
        return current
    return ScopeContext(
        task_id="controller",
        batch_id="controller",
        combination_id="controller",
        scope_id=scope_id,
        workspace=workspace,
        capability_hash="controller",
    )


def _read_controller_action_result(value: Mapping[str, Any] | str | os.PathLike[str] | Path) -> dict[str, Any] | None:
    """读取 Controller 提供的 completed ActionResult，拒绝软链接和非 object。"""

    if isinstance(value, Mapping):
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    candidate = Path(value).expanduser()
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _controller_adopt_job_status(
    action_result: Mapping[str, Any],
    explicit_status: str | None,
) -> str | None:
    """把已完成 ActionResult 映射为 supervisor 的四种终态之一。"""

    if explicit_status is None:
        result_status = str(action_result.get("status") or "")
        return "success" if result_status in {"success", "partial_success"} else "failed"
    normalized = str(explicit_status).strip().casefold()
    if normalized == "partial_success":
        normalized = "success"
    return normalized if normalized in TERMINAL_STATES else None


def _scoped_operation_id_for_controller(scope_id: str, submission_key: str) -> str:
    """在没有 ContextVar 的 Controller 进程中重建同一 scope operation 身份。"""

    digest = hashlib.sha256(
        f"{scope_id}\0{validate_submission_key(submission_key)}".encode("utf-8")
    ).hexdigest()
    return f"scope-submission-{digest}"


def controller_adopt_completed_action_result(
    action_result: Mapping[str, Any] | str | os.PathLike[str] | Path,
    request: TransitionStatePathJobRequest | Mapping[str, Any] | None = None,
    *,
    submission_key: str | None = None,
    request_fingerprint: str | None = None,
    operation_id: str | None = None,
    job_id: str | None = None,
    terminal_status: str | None = None,
    scope_id: str | None = None,
    workspace: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """Controller 导入已完成 ActionResult，创建可恢复的终态作业收据。

    该入口只接受可信 Controller 调用，不执行 Action 或重新提交 worker。它严格核对
    TS Action/backend、请求指纹、终态、父 scope ArtifactStore 中的 artifact 身份和
    SHA-256；可提供完整 request 重新计算指纹，也可在恢复记录已保存 request 指纹时只
    提供 submission_key 与 request_fingerprint。重复导入同一身份只返回原收据，冲突或
    中断状态均 fail-closed。
    """

    identity = _controller_identity(scope_id=scope_id, workspace=workspace)
    if isinstance(identity, dict):
        return identity
    selected_scope, selected_workspace = identity
    payload = _read_controller_action_result(action_result)
    if payload is None:
        return _error("transition_state_path_adopt_invalid_result", "Controller 提供的 ActionResult 不可读取")
    envelope_operation_id = payload.get("operation_id")
    envelope_submission_key = payload.get("submission_key")
    action_result_payload = dict(payload)
    # 旧的持久 ActionResult envelope 可在顶层携带异步身份；身份单独核对后，
    # 只把标准 ActionResult 字段写入新的 canonical 收据。
    action_result_payload.pop("operation_id", None)
    action_result_payload.pop("submission_key", None)
    try:
        canonical_result = ActionResult.model_validate(action_result_payload).model_dump(mode="json")
    except Exception as exc:
        return _error(
            "transition_state_path_adopt_invalid_result",
            "Controller 提供的 ActionResult 不符合公开结果合同",
            exception_type=type(exc).__name__,
        )
    if canonical_result.get("action") != ACTION_ID or canonical_result.get("backend") != BACKEND_ID:
        return _error(
            "transition_state_path_adopt_identity_conflict",
            "Controller ActionResult 的 action/backend 与 TS Pipeline 不一致",
        )
    selected_terminal_status = _controller_adopt_job_status(canonical_result, terminal_status)
    if selected_terminal_status is None:
        return _error(
            "transition_state_path_adopt_invalid_status",
            "Controller 必须提供 supervisor 可识别的终态",
        )
    adopted_request: TransitionStatePathJobRequest | None = None
    if request is not None:
        try:
            request_value = (
                request.model_dump(mode="json")
                if isinstance(request, TransitionStatePathJobRequest)
                else dict(request)
            )
            if submission_key is not None:
                if request_value.get("submission_key") not in {None, submission_key}:
                    return _error("submission_key_conflict", "request 与 submission_key 不一致")
                request_value["submission_key"] = submission_key
            adopted_request = TransitionStatePathJobRequest.model_validate(request_value)
            selected_submission_key = validate_submission_key(str(adopted_request.submission_key))
        except (TypeError, ValueError) as exc:
            return _error(
                "transition_state_path_adopt_invalid_request",
                "Controller 提供的 TS request 无法核对",
                exception_type=type(exc).__name__,
            )
        calculated_fingerprint = _submission_request_fingerprint(adopted_request)
        if request_fingerprint is not None:
            if re.fullmatch(r"[0-9a-f]{64}", str(request_fingerprint)) is None:
                return _error("invalid_request_fingerprint", "request_fingerprint 必须是小写 SHA-256")
            if request_fingerprint != calculated_fingerprint:
                return _error("request_fingerprint_conflict", "Controller request_fingerprint 与 request 不一致")
    else:
        if submission_key is None:
            return _error(
                "transition_state_path_adopt_invalid_request",
                "缺少 request 时必须提供 submission_key",
            )
        try:
            selected_submission_key = validate_submission_key(submission_key)
        except ValueError as exc:
            return _error("invalid_submission_key", str(exc))
        if re.fullmatch(r"[0-9a-f]{64}", str(request_fingerprint or "")) is None:
            return _error(
                "invalid_request_fingerprint",
                "缺少 request 时必须提供小写 SHA-256 request_fingerprint",
            )
        calculated_fingerprint = str(request_fingerprint)
    selected_operation_id = _scoped_operation_id_for_controller(
        selected_scope,
        selected_submission_key,
    )
    if operation_id is not None and str(operation_id) != selected_operation_id:
        return _error("operation_identity_conflict", "operation_id 与 submission_key 不一致")
    if envelope_operation_id is not None and str(envelope_operation_id) != selected_operation_id:
        return _error("operation_identity_conflict", "ActionResult envelope 的 operation_id 与 request 不一致")
    if envelope_submission_key is not None and str(envelope_submission_key) != selected_submission_key:
        return _error("submission_key_conflict", "ActionResult envelope 的 submission_key 与 request 不一致")
    try:
        _validate_parent_transition_state_path_artifacts(
            canonical_result,
            workspace=selected_workspace,
        )
    except Exception as exc:
        return _error(
            "transition_state_path_adopt_artifact_conflict",
            "Controller ActionResult 的 artifact 归属、路径或 SHA-256 无法核对",
            exception_type=type(exc).__name__,
        )
    scope = _controller_scope_context(selected_scope, selected_workspace)
    fingerprint = calculated_fingerprint
    with _state_lock(selected_workspace):
        state, readable = _load_state(scope, selected_workspace)
        if not readable:
            return _error(
                "transition_state_path_state_unavailable",
                "异步作业身份收据损坏或 scope 不匹配，已阻止导入",
            )
        if state is None:
            state = _empty_state(scope)
        existing = _job_record(state, selected_submission_key)
        if existing is not None:
            if existing.get("request_fingerprint") != fingerprint:
                return _error(
                    "submission_key_conflict",
                    "同一 scope/submission_key 已绑定不同的 TS Pipeline 请求",
                    job_id=existing.get("job_id"),
                )
            try:
                registry = ExecutionJobRegistry(
                    workspace=selected_workspace,
                    scope_id=selected_scope,
                )
                reservation = registry.reserve(
                    job_type=JOB_TYPE,
                    operation_id=selected_operation_id,
                    operation_fingerprint=fingerprint,
                    submission_key=selected_submission_key,
                    caller_request_fingerprint=fingerprint,
                    preferred_job_id=str(existing.get("job_id") or ""),
                )
            except ExecutionJobRegistryConflict as exc:
                conflict_record = getattr(exc, "record", None)
                return _error(
                    "transition_state_path_submission_conflict",
                    "已有 TS operation 与统一 execution job registry 冲突，未覆盖原身份",
                    job_id=(conflict_record or {}).get("job_id") if isinstance(conflict_record, Mapping) else existing.get("job_id"),
                )
            except (ExecutionJobRegistryError, ValueError):
                return _error(
                    "transition_state_path_state_unavailable",
                    "已有 TS operation 无法在统一 execution job registry 中核对",
                    job_id=existing.get("job_id"),
                )
            if reservation.job_id != str(existing.get("job_id") or ""):
                return _error(
                    "transition_state_path_identity_conflict",
                    "TS state 与统一 execution job registry 的 job_id 不一致",
                    job_id=existing.get("job_id"),
                )
            existing_result = _read_transition_state_path_result(
                selected_workspace,
                str(existing.get("job_id") or ""),
            )
            if existing_result is None or existing_result != canonical_result:
                return _error(
                    "transition_state_path_adoption_uncertain",
                    "已有导入收据的 ActionResult 无法与本次结果严格核对",
                    job_id=existing.get("job_id"),
                )
            status = _read_job_status_for_workspace(
                selected_workspace,
                selected_scope,
                str(existing.get("job_id") or ""),
            )
            if existing.get("phase") in {"submitting", "unknown"} and status is None:
                return _error(
                    "transition_state_path_adoption_uncertain",
                    "已有导入身份但终态收据无法核对，已阻止覆盖或重放",
                    job_id=existing.get("job_id"),
                )
            return _minimal_identity(existing, status, idempotent_reused=True)
        active_key = state.get("active_submission_key")
        if active_key is not None:
            active = _job_record(state, str(active_key))
            if active is None or active.get("phase") != "collected":
                return _error(
                    "transition_state_path_phase_gate_closed",
                    "当前 scope 已有未收集的 TS Pipeline 作业，不能导入新的作业",
                    job_id=active.get("job_id") if active else None,
                )
            state["active_submission_key"] = None
            state["active_job_id"] = None
        try:
            registry = ExecutionJobRegistry(
                workspace=selected_workspace,
                scope_id=selected_scope,
            )
            reservation = registry.reserve(
                job_type=JOB_TYPE,
                operation_id=selected_operation_id,
                operation_fingerprint=fingerprint,
                submission_key=selected_submission_key,
                caller_request_fingerprint=fingerprint,
                preferred_job_id=str(job_id) if job_id is not None else None,
            )
        except ExecutionJobRegistryConflict as exc:
            conflict_record = getattr(exc, "record", None)
            return _error(
                "transition_state_path_submission_conflict",
                "Controller operation 已绑定其他 execution job，未覆盖原身份",
                job_id=(conflict_record or {}).get("job_id") if isinstance(conflict_record, Mapping) else None,
            )
        except (ExecutionJobRegistryError, ValueError):
            return _error(
                "transition_state_path_state_unavailable",
                "Controller 无法在统一 execution job registry 中预留 TS job_id",
            )
        selected_job_id = reservation.job_id
        directory = _job_directory_for_workspace(selected_workspace, selected_job_id)
        if directory is None:
            return _error("transition_state_path_state_unavailable", "Controller job 路径无法核对")
        try:
            if directory.exists() or directory.is_symlink():
                return _error(
                    "transition_state_path_adoption_uncertain",
                    "目标 job 目录已存在但没有可核对的 TS 收据，已阻止覆盖",
                    job_id=selected_job_id,
                )
            ensure_private_directory(directory.parent, create=True)
            directory.mkdir(mode=0o700)
            record: dict[str, Any] = {
                "job_id": selected_job_id,
                "operation_id": selected_operation_id,
                "submission_key": selected_submission_key,
                "request_fingerprint": fingerprint,
                "scope_id": selected_scope,
                "scope_workspace": str(selected_workspace),
                "phase": "submitting",
                "wake_armed": False,
                "submitted_at": _now(),
                "adopted_at": _now(),
            }
            state["jobs"][selected_submission_key] = record
            state["active_submission_key"] = selected_submission_key
            state["active_job_id"] = selected_job_id
            _save_state(state, selected_workspace)
            status_value = {
                "schema_version": 1,
                "job_id": selected_job_id,
                "job_type": JOB_TYPE,
                "status": selected_terminal_status,
                "supervisor_pid": None,
                "command": ["controller_adopted_action_result"],
                "job_directory": str(directory.relative_to(selected_workspace)),
                "stdout_path": None,
                "stderr_path": None,
                "resource_limits": (
                    adopted_request.resource_limits.model_dump(mode="json")
                    if adopted_request is not None
                    else {}
                ),
                "resource_allocation": {"execution_mode": "controller_adopted_result"},
                "submitted_at": record["submitted_at"],
                "finished_at": _now(),
                "metadata": {
                    "action_id": ACTION_ID,
                    "backend_id": BACKEND_ID,
                    "adopted_completed_action_result": True,
                    **(
                        {
                            "transition_state_path_request": adopted_request.model_dump(
                                mode="json"
                            ),
                            "caller_transition_state_path_request": adopted_request.model_dump(
                                mode="json"
                            ),
                        }
                        if adopted_request is not None
                        else {}
                    ),
                },
                "execution_mode": "controller_adopted_result",
                "scope_id": selected_scope,
                "scope_workspace": str(selected_workspace),
                "operation_id": selected_operation_id,
                "operation_fingerprint": fingerprint,
                "caller_request_fingerprint": fingerprint,
                "submission_key": selected_submission_key,
            }
            request_record = {
                "schema_version": 1,
                "job_id": selected_job_id,
                "job_type": JOB_TYPE,
                "runtime": "ash_neb",
                "metadata": status_value["metadata"],
                "scope_id": selected_scope,
                "scope_workspace": str(selected_workspace),
                "operation_id": selected_operation_id,
                "operation_fingerprint": fingerprint,
                "caller_request_fingerprint": fingerprint,
                "submission_key": selected_submission_key,
            }
            receipt = {
                "schema_version": 1,
                "operation_id": selected_operation_id,
                "fingerprint": fingerprint,
                "operation_fingerprint": fingerprint,
                "caller_request_fingerprint": fingerprint,
                "job_id": selected_job_id,
                "job_type": JOB_TYPE,
                "submission_key": selected_submission_key,
                "phase": selected_terminal_status,
                "adopted_completed_action_result": True,
                "scope_id": selected_scope,
                "scope_workspace": str(selected_workspace),
                "created_at": record["adopted_at"],
                "updated_at": _now(),
            }
            _atomic_json(directory / "request.json", request_record)
            _atomic_json(directory / "status.json", status_value)
            _atomic_json(directory / "operation_receipt.json", receipt)
            _atomic_json(directory / RESULT_FILENAME, canonical_result)
            registry.mark(selected_job_id, phase=selected_terminal_status)
            record["phase"] = "armed"
            record["wake_armed"] = True
            record["terminal_status"] = selected_terminal_status
            record["wake_armed_at"] = _now()
            state["jobs"][selected_submission_key] = record
            _save_state(state, selected_workspace)
            return _minimal_identity(record, status_value, idempotent_reused=False, phase="armed")
        except Exception as exc:
            record = state.get("jobs", {}).get(selected_submission_key)
            if isinstance(record, dict):
                record["phase"] = "unknown"
                record["wake_armed"] = False
                state["jobs"][selected_submission_key] = record
                _save_state(state, selected_workspace)
            return _error(
                "transition_state_path_adoption_uncertain",
                "已创建导入身份但 completed ActionResult 收据未完整落盘，已阻止重放",
                job_id=selected_job_id,
                exception_type=type(exc).__name__,
            )


def controller_adopt_completed_transition_state_path_result(
    action_result: Mapping[str, Any] | str | os.PathLike[str] | Path,
    request: TransitionStatePathJobRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """兼容名称：Controller 导入一个已完成 TS ActionResult，不运行科学逻辑。"""

    return controller_adopt_completed_action_result(action_result, request, **kwargs)


def controller_import_completed_action_result(
    action_result: Mapping[str, Any] | str | os.PathLike[str] | Path,
    request: TransitionStatePathJobRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """兼容 import 命名：导入已完成结果收据，不执行科学 Action。"""

    return controller_adopt_completed_action_result(action_result, request, **kwargs)


def controller_get_transition_state_path_job(
    job_id: str,
    *,
    scope_id: str | None = None,
    workspace: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """Controller 轮询接口；不暴露为 MCP tool。"""

    return _controller_get_job_status(job_id, scope_id=scope_id, workspace=workspace)


def controller_arm_transition_state_path_job(
    job_id: str,
    *,
    scope_id: str | None = None,
    workspace: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    """Controller 恢复接口；终态 arm 只为中断后的 wake 保留。"""

    return _controller_arm_job(job_id, scope_id=scope_id, workspace=workspace)


def investigation_execute_action(request: discovery_tools.ProgressiveActionRequest) -> dict[str, Any]:
    """Investigation profile 中拒绝同步 TS Action，其他 Action 保持原路径。"""

    if request.action_id == ACTION_ID:
        return execute_traced(
            ACTION_ID,
            {"entrypoint": "progressive_execute_action", "request": request.model_dump(mode="json")},
            lambda: _error(
                "async_job_required",
                "search_transition_state_path 在 Investigation 中必须使用 submit_transition_state_path_job；提交返回 job_id 后在同一 turn 按 get→终态→collect 完成，若作业身份不确定则禁止重复提交",
                required_submit_tool=TOOL_NAMES[0],
                required_get_tool=TOOL_NAMES[1],
                required_collect_tool=TOOL_NAMES[2],
            ),
            capture_artifacts=False,
        )
    return discovery_tools.execute_action(request)


def _worker_action_result(
    request: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """在 detached worker 中运行原子 TS Pipeline 并保存原始 worker envelope。

    ArtifactRef 必须在父 Investigation workspace 中登记；这里不能在 job 私有 workspace
    中提前登记，否则模型 wake 后无法通过父 scope 的 ArtifactStore 读取这些引用。
    """

    from minichem_toolbox.backends.arche_ts_pipeline import run_atomic_workflow

    action_request = ActionRequest.model_validate(request)
    # 直接复用原有 run_atomic_workflow，异步层不重排、不增删科学阶段。
    del job_id
    return run_atomic_workflow(action_request.model_dump(mode="json"))


def _run_worker() -> int:
    """读取 supervisor 生成的 request.json，执行一次 TS Pipeline 并落盘结果。"""

    job_directory = Path.cwd().resolve()
    # 科学工作流在作业私有 workspace 中运行；移除 scope Context 环境，避免把
    # trusted supervisor 的 capability 身份伪装成模型的 MCP 请求。
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
    request = metadata.get("transition_state_path_request") if isinstance(metadata, Mapping) else None
    job_id = str(request_record.get("job_id") or "")
    if not isinstance(request, dict) or JOB_ID_PATTERN.fullmatch(job_id) is None:
        return 2
    try:
        result = _worker_action_result(request, job_id)
    except Exception as exc:
        result = ActionResult(
            status="failed",
            action=ACTION_ID,
            action_version=action_specs()[ACTION_ID].version,
            requested_backend=BACKEND_ID,
            backend=BACKEND_ID,
            selection_source="agent",
            error={
                "code": "transition_state_path_worker_failed",
                "message": "TS Pipeline detached worker failed before producing a scientific result",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            retryable=False,
            provenance={"async_job_id": job_id},
        ).model_dump(mode="json")
    _atomic_json(job_directory / RAW_RESULT_FILENAME, result)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        raise SystemExit(_run_worker())
    raise SystemExit(2)


__all__ = [
    "BACKEND_ID",
    "ACTION_ID",
    "TOOL_NAMES",
    "TOOL_DESCRIPTIONS",
    "TransitionStatePathJobCollectRequest",
    "TransitionStatePathJobController",
    "TransitionStatePathJobRequest",
    "TransitionStatePathJobStatusRequest",
    "collect_transition_state_path_job",
    "controller_adopt_completed_action_result",
    "controller_adopt_completed_transition_state_path_result",
    "controller_arm_transition_state_path_job",
    "controller_get_transition_state_path_job",
    "controller_import_completed_action_result",
    "get_transition_state_path_job",
    "investigation_execute_action",
    "submit_transition_state_path_job",
]
