"""为跨进程恢复读取可核对的进程身份。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


ProcessActivity = Literal["alive", "stopped", "unknown"]
ProcessGroupActivity = Literal["empty", "nonempty", "unknown"]


@dataclass(frozen=True)
class ProcessIdentity:
    """保存进程及其组/会话身份，避免把复用的 PID 或 PGID 当成原作业。"""

    pid: int
    start_time_ticks: int | None
    boot_id: str | None
    process_group_id: int | None = None
    session_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """返回可写入内部 JSON 的身份字段；读取失败的时钟保留为 None。"""

        return {
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "start_ticks": self.start_time_ticks,
            "boot_id": self.boot_id,
            "process_group_id": self.process_group_id,
            "session_id": self.session_id,
        }

    @property
    def start_ticks(self) -> int | None:
        """返回内核启动时钟的简短别名，便于核对旧版记录。"""

        return self.start_time_ticks


def _read_proc_stat(pid: int) -> tuple[str, int, int, int] | None:
    """读取状态、启动时钟、进程组和会话；格式不可核对时返回 None。"""

    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = text[closing_parenthesis + 2 :].split()
    # 截断命令名后，fields[0] 是 stat 的第 3 项 state，fields[2]/[3] 分别是
    # pgrp/session，fields[19] 是第 22 项 starttime。任一关键字段缺失都不能
    # 把 PID 或进程组的暂时存在误认为身份匹配。
    if len(fields) <= 19:
        return None
    try:
        return fields[0], int(fields[19]), int(fields[2]), int(fields[3])
    except (TypeError, ValueError):
        return None


def _read_boot_id() -> str | None:
    """读取主机启动标识；无法读取时返回 None，使恢复判断保持未知。"""

    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None
    normalized = value.strip()
    return normalized or None


def capture_process_identity(pid: int) -> ProcessIdentity:
    """读取指定进程当前身份；调用方负责在失败或未知时采取保守策略。"""

    selected_pid = int(pid)
    if selected_pid <= 0:
        raise ValueError("process pid must be positive")
    record = _read_proc_stat(selected_pid)
    if record is None:
        return ProcessIdentity(selected_pid, None, None)
    _state, start_time_ticks, process_group_id, session_id = record
    return ProcessIdentity(
        selected_pid,
        start_time_ticks,
        _read_boot_id(),
        process_group_id,
        session_id,
    )


def process_identity_activity(identity: Mapping[str, Any]) -> ProcessActivity:
    """核对持久身份当前是否仍活、已停止或无法安全判断。

    缺少启动时钟、boot id 无法在当前系统核对、记录畸形以及 PID 仍存在但身份无法
    匹配都不会返回 ``alive``。``stopped`` 只表示同一启动时钟的进程已经消失、变成
    僵尸或 PID 已复用于另一进程；其余情况返回 ``unknown``，供资源恢复保持占用。
    """

    try:
        pid = int(identity.get("pid") or 0)
        raw_start = identity.get("start_time_ticks")
        if raw_start is None:
            raw_start = identity.get("start_ticks")
        expected_start = int(raw_start)
    except (TypeError, ValueError):
        return "unknown"
    if pid <= 0 or expected_start <= 0:
        return "unknown"
    current = _read_proc_stat(pid)
    if current is None:
        # ``/proc`` 缺失可能表示进程已退出，也可能只是暂时不可读。只有内核
        # 明确返回 ESRCH 才能释放；仍有 PID 或无法核对时都保持 unknown。
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "stopped"
        except (PermissionError, OSError):
            return "unknown"
        return "unknown"
    state, current_start, _current_process_group_id, _current_session_id = current
    if current_start != expected_start:
        return "stopped"
    expected_boot = str(identity.get("boot_id") or "").strip()
    if not expected_boot:
        # 没有 boot_id 的持久记录无法排除跨主机/跨启动的 PID 复用；即使
        # starttime 相同也只能保持未知，不能把它当作仍存活的作业。
        return "unknown"
    current_boot = _read_boot_id()
    if not current_boot:
        return "unknown"
    if current_boot != expected_boot:
        return "stopped"
    if state == "Z":
        return "stopped"
    return "alive"


def inspect_process_group_membership(
    process_group_id: int | None,
    session_id: int | None,
    boot_id: str | None,
) -> ProcessGroupActivity:
    """在当前 boot 下扫描进程组/会话成员，无法证明时返回 ``unknown``。

    该函数只读 ``/proc``，不向任何 PID 或 PGID 发信号。调用方只能在返回
    ``empty`` 时完成资源回收；权限、格式、boot 或扫描状态不完整都会保持占用。
    ``empty`` 表示本次一致性检查没有观察到成员，不承诺跨越后续 fork 的原子保证。
    """

    selected = _validate_process_group_identity(
        process_group_id,
        session_id,
        boot_id,
    )
    if selected is None:
        return "unknown"
    selected_group, selected_session, expected_boot = selected
    if not _current_boot_matches(expected_boot):
        return "unknown"
    return _scan_process_group_membership(selected_group, selected_session)


def _validate_process_group_identity(
    process_group_id: int | None,
    session_id: int | None,
    boot_id: str | None,
) -> tuple[int, int, str] | None:
    """校验进程组、会话和 boot 身份，失败时返回 None。"""

    if (
        isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or isinstance(session_id, bool)
        or not isinstance(session_id, int)
    ):
        return None
    expected_boot = str(boot_id or "").strip()
    if process_group_id <= 0 or session_id <= 0 or not expected_boot:
        return None
    return process_group_id, session_id, expected_boot


def _current_boot_matches(expected_boot: str) -> bool:
    """确认当前主机仍处于持久身份记录的 boot。"""

    current_boot = _read_boot_id()
    return bool(current_boot and current_boot == expected_boot)


def _scan_process_group_membership(
    selected_group: int,
    selected_session: int,
) -> ProcessGroupActivity:
    """扫描 proc 中的组成员；扫描不完整时保持 unknown。"""

    proc = Path("/proc")
    try:
        entries = tuple(proc.iterdir())
    except (FileNotFoundError, OSError):
        return "unknown"
    for entry in entries:
        if not entry.name.isdigit():
            continue
        membership = _read_process_group_membership(entry)
        if membership is False:
            return "unknown"
        if membership is None:
            # 进程在扫描间隙退出，不能把它当作仍然存在的成员。
            continue
        current_group, current_session = membership
        if current_group == selected_group and current_session == selected_session:
            return "nonempty"
    return "empty"


def _read_process_group_membership(
    entry: Path,
) -> tuple[int, int] | None | Literal[False]:
    """读取一个 proc stat 的组/会话字段；False 表示无法安全解析。"""

    try:
        text = (entry / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        return False
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        return False
    fields = text[closing_parenthesis + 2 :].split()
    if len(fields) <= 3:
        return False
    try:
        return int(fields[2]), int(fields[3])
    except (TypeError, ValueError):
        return False


__all__ = [
    "ProcessActivity",
    "ProcessGroupActivity",
    "ProcessIdentity",
    "capture_process_identity",
    "inspect_process_group_membership",
    "process_identity_activity",
]
