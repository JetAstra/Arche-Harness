"""验证工作区目录创建在并发下保持私有且幂等。"""

from __future__ import annotations

import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from minichem_mcp_tools import workspace as workspace_module


def test_concurrent_private_directory_creation_accepts_one_mkdir_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """两个请求同时创建同一输出目录时只接受目录创建的 EEXIST。"""

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    target = root / "outputs" / "execution_jobs" / "job-1"
    original_mkdir = workspace_module.os.mkdir
    mkdir_rendezvous = threading.Barrier(2)

    def synchronized_mkdir(
        path: str | bytes | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if path == "outputs" and dir_fd is not None:
            mkdir_rendezvous.wait(timeout=5)
        original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(workspace_module.os, "mkdir", synchronized_mkdir)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _index: workspace_module.ensure_private_directory(
                    target,
                    create=True,
                ),
                range(2),
            )
        )

    assert results == [target, target]
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
