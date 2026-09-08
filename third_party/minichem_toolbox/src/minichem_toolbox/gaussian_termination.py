"""共享的 Gaussian 终止与归档警告判定。

该模块只依赖 Python 标准库，供 detached supervisor 和 typed Gaussian
backend 共用。它只识别一个已知例外：Gaussian 正常完成计算，但归档条目
写入返回退出码 1。其他非零退出、错误终止或未核验的进程状态都不能放行。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER = (
    "Unable to Open any file for archive entry."
)
GAUSSIAN_ARCHIVE_PUNCHED_MARKER = "The archive entry for this job was punched."
GAUSSIAN_NORMAL_TERMINATION_PATTERN = re.compile(
    r"^Normal termination of Gaussian 16(?:\s+at\b.*)?$",
    flags=re.IGNORECASE,
)
GAUSSIAN_FATAL_MARKER_PATTERNS = (
    re.compile(r"\bError termination\b", flags=re.IGNORECASE),
    re.compile(r"\bErroneous termination\b", flags=re.IGNORECASE),
    re.compile(r"\bQPErr\b", flags=re.IGNORECASE),
    re.compile(r"\bFatal Error\b", flags=re.IGNORECASE),
    re.compile(r"\bsegmentation fault\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:SIG(?:TERM|KILL|SEGV)|Killed|Aborted|Abort)\b",
        flags=re.IGNORECASE,
    ),
)


def gaussian_output_ends_with_normal_termination(stdout: str) -> bool:
    """确认 stdout 最后一条非空行是 Gaussian 的正常终止标记。"""

    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped:
            return bool(GAUSSIAN_NORMAL_TERMINATION_PATTERN.fullmatch(stripped))
    return False


def gaussian_contains_fatal_marker(stdout: str, stderr: str) -> bool:
    """拒绝含错误终止、解析错误或进程崩溃标记的 Gaussian 输出。"""

    combined = f"{stdout}\n{stderr}"
    return any(pattern.search(combined) for pattern in GAUSSIAN_FATAL_MARKER_PATTERNS)


def _has_unsafe_completion_state(completed: Mapping[str, Any]) -> bool:
    """拒绝超时、资源异常、取消或无法证明已清理的进程状态。"""

    if any(
        bool(completed.get(field))
        for field in (
            "timeout",
            "timed_out",
            "memory_exceeded",
            "cancelled",
            "cancellation_requested",
            "cleanup_pending",
            "child_reap_failed",
            "supervisor_exit_pending",
        )
    ):
        return True
    if completed.get("process_group_activity") not in {None, "empty"}:
        return True
    if completed.get("background_process_cleanup") not in {None, "none_detected"}:
        return True
    return False


def gaussian_archive_only_completion(completed: Mapping[str, Any]) -> bool:
    """只接受已核验的 Gaussian 归档警告退出码 1。

    ``completed`` 至少包含 ``returncode``、``stdout`` 和 ``stderr``。调用方
    可以额外提供 timeout、资源、取消和进程清理字段；这些字段只要表明
    状态无法安全核验，就会拒绝归档警告例外。
    """

    stdout = str(completed.get("stdout") or "")
    stderr = str(completed.get("stderr") or "")
    return (
        type(completed.get("returncode")) is int
        and completed.get("returncode") == 1
        and not _has_unsafe_completion_state(completed)
        and not stderr.strip()
        and GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER in stdout
        and GAUSSIAN_ARCHIVE_PUNCHED_MARKER in stdout
        and gaussian_output_ends_with_normal_termination(stdout)
        and not gaussian_contains_fatal_marker(stdout, stderr)
    )


def gaussian_archive_warning_provenance(returncode: int = 1) -> dict[str, Any]:
    """返回可写入状态或结果的归档警告来源记录。"""

    return {
        "gaussian_external_returncode": returncode,
        "gaussian_termination": "normal",
        "gaussian_archive_warning": {
            "entry_marker": GAUSSIAN_ARCHIVE_ENTRY_FAILURE_MARKER,
            "punched_marker": GAUSSIAN_ARCHIVE_PUNCHED_MARKER,
        },
    }
