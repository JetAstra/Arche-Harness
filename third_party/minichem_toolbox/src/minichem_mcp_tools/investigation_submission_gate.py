"""Investigation profile 的跨作业类型单 active 提交门。

Action、TS Pipeline、原生软件和分析程序各自保留原有提交实现；本模块只在
Investigation profile 的四个公开 submit 入口外层串行核对中央
``ExecutionJobRegistry``。同一 scope 在旧作业 collect 前不能提交另一项，原请求
使用相同 job_type/submission_key 重试时仍会进入所属实现完成幂等核对。
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

from minichem_toolbox.scope import current_scope

from .action_jobs import ACTION_JOB_TYPE, TS_ACTION_ID, ActionJobRequest
from .execution_job_registry import ExecutionJobRegistry, ExecutionJobRegistryError
from .execution_models import AnalysisJobRequest, NativeJobRequest
from .transition_state_jobs import JOB_TYPE as TRANSITION_STATE_JOB_TYPE
from .transition_state_jobs import TransitionStatePathJobRequest
from .workspace import ensure_private_directory


_GATE_DIRECTORY = "investigation_submission_gate"
_GATE_FILENAME = "submission.lock"


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    """构造不暴露 registry 和 workspace 内部字段的拒绝响应。"""

    response: dict[str, Any] = {
        "status": "invalid_request",
        "valid": False,
        "error": {"code": code, "message": message},
        "automatic_fallback": False,
    }
    response["error"].update(details)
    return response


@contextmanager
def _submission_lock() -> Iterator[None]:
    """锁定当前 capability scope 的四类 Investigation submit。"""

    scope = current_scope()
    if scope is None:
        raise ExecutionJobRegistryError("Investigation submission scope is unavailable")
    try:
        root = ensure_private_directory(
            scope.workspace / "_tool_artifacts" / _GATE_DIRECTORY,
            create=True,
        )
    except ValueError as exc:
        raise ExecutionJobRegistryError(
            "Investigation submission lock directory is unavailable"
        ) from exc
    descriptor = os.open(
        root / _GATE_FILENAME,
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


def _active_records() -> list[dict[str, Any]]:
    """读取当前 scope 尚未 collect 的中央作业身份。"""

    scope = current_scope()
    if scope is None:
        raise ExecutionJobRegistryError("Investigation submission scope is unavailable")
    registry = ExecutionJobRegistry(workspace=scope.workspace, scope_id=scope.scope_id)
    return [record for record in registry.list_records() if record.get("phase") != "collected"]


def _submit_investigation_job(
    request: Any,
    *,
    job_type: str,
    submitter: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """在跨类型门内提交或幂等重放一个 Investigation 作业。

    输入是已经由 FastMCP 按所属类型解析的请求、固定 job_type 和原提交函数。若
    中央 registry 存在不同未 collect 作业，返回其最小身份并拒绝新提交；registry
    或锁不可核对时 fail closed。返回值完全沿用原提交函数或统一拒绝 envelope。
    """

    scope = current_scope()
    if scope is None:
        return _error("scope_required", "Investigation job 必须在已认证 scope 中提交")
    submission_key = getattr(request, "submission_key", None)
    try:
        with _submission_lock():
            records = _active_records()
            if records:
                replay = (
                    len(records) == 1
                    and isinstance(submission_key, str)
                    and records[0].get("job_type") == job_type
                    and records[0].get("submission_key") == submission_key
                )
                if not replay:
                    active = records[0]
                    return _error(
                        "investigation_job_phase_gate_closed",
                        "当前 scope 已有未 collect 的作业；必须先查询并收集该 job_id",
                        job_id=active.get("job_id"),
                        job_type=active.get("job_type"),
                        operation_id=active.get("operation_id"),
                        submission_key=active.get("submission_key"),
                        active_job_count=len(records),
                    )
            return submitter(request)
    except (ExecutionJobRegistryError, OSError):
        return _error(
            "investigation_job_gate_unavailable",
            "Investigation 作业门禁无法安全核对，已阻止新的提交",
        )


def submit_investigation_action_job(request: ActionJobRequest) -> dict[str, Any]:
    """通过跨类型门提交通用 Catalog Action job。

    ``search_transition_state_path`` 仍然只有一个模型可见的 submit 名称，但其
    内部持久化作业类型是 TS Pipeline。门禁必须使用与中央 registry 相同的类型，
    否则同一个 ``submission_key`` 的幂等重试会被误判成另一类作业。
    """

    from .action_jobs import submit_action_job

    return _submit_investigation_job(
        request,
        job_type=(
            TRANSITION_STATE_JOB_TYPE
            if request.action_id == TS_ACTION_ID
            else ACTION_JOB_TYPE
        ),
        submitter=submit_action_job,
    )


def submit_investigation_transition_state_path_job(
    request: TransitionStatePathJobRequest,
) -> dict[str, Any]:
    """通过跨类型门提交 TS Pipeline job。"""

    from .transition_state_jobs import submit_transition_state_path_job

    return _submit_investigation_job(
        request,
        job_type=TRANSITION_STATE_JOB_TYPE,
        submitter=submit_transition_state_path_job,
    )


def submit_investigation_native_job(request: NativeJobRequest) -> dict[str, Any]:
    """通过跨类型门提交原生软件 job。"""

    from .open_tools import submit_native_job

    return _submit_investigation_job(
        request,
        job_type="native_software",
        submitter=submit_native_job,
    )


def submit_investigation_analysis_program(
    request: AnalysisJobRequest,
) -> dict[str, Any]:
    """通过跨类型门提交可编程分析 job。"""

    from .open_tools import submit_analysis_program

    return _submit_investigation_job(
        request,
        job_type="programmable_analysis",
        submitter=submit_analysis_program,
    )


__all__ = [
    "submit_investigation_action_job",
    "submit_investigation_analysis_program",
    "submit_investigation_native_job",
    "submit_investigation_transition_state_path_job",
]
