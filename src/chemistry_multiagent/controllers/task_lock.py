"""防止同一个持久任务被两个控制器进程同时推进。"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from pathlib import Path
from typing import IO


class TaskRunnerBusy(RuntimeError):
    """任务锁仍由存活进程持有，当前控制器不能接管。"""


class TaskRunnerFileLock:
    """使用非阻塞 ``flock`` 独占一个任务的新建或恢复入口。

    输入是任务目录内的锁文件路径。``acquire`` 创建普通锁文件并取得内核文件锁；
    成功后同一时刻只有当前进程可以推进该任务，已有进程持锁时抛出
    ``TaskRunnerBusy``。``release`` 释放文件描述符但保留锁文件，进程异常退出时内核
    也会自动释放锁，因此不需要删除文件来判断所有权。
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """保存锁文件路径；此时尚未创建文件或取得锁。"""

        self.path = Path(path)
        self._handle: IO[bytes] | None = None

    def acquire(self) -> None:
        """非阻塞取得任务锁；当前对象重复取得或任务繁忙时明确失败。"""

        if self._handle is not None:
            raise RuntimeError("当前 TaskRunnerFileLock 已经持有锁")
        parent = self.path.parent
        missing: list[Path] = []
        current = parent
        while not current.exists():
            if current.is_symlink():
                raise ValueError("任务锁目录不得是符号链接")
            missing.append(current)
            next_current = current.parent
            if next_current == current:
                break
            current = next_current
        if current.is_symlink() or not current.is_dir():
            raise ValueError("任务锁目录必须是普通目录")
        for directory in reversed(missing):
            directory.mkdir(exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("任务锁目录必须是普通目录")
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("任务锁目录必须是普通目录")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("任务锁文件不得是符号链接") from exc
            if exc.errno == errno.EISDIR:
                raise ValueError("任务锁文件必须是普通文件") from exc
            raise
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("任务锁文件必须是普通文件")
            handle = os.fdopen(descriptor, "a+b")
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise TaskRunnerBusy("任务仍有存活 runner") from exc
            raise
        self._handle = handle

    def release(self) -> None:
        """释放当前内核文件锁；尚未持锁时保持无害。"""

        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "TaskRunnerFileLock":
        """取得锁并返回当前上下文管理器。"""

        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """无论任务成功或抛错都释放当前文件锁。"""

        self.release()


__all__ = ["TaskRunnerBusy", "TaskRunnerFileLock"]
