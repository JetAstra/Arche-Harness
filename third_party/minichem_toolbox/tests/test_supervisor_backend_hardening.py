"""supervisor 和外部 Backend 的环境、文件与进程安全回归测试。"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from minichem_mcp_tools import job_supervisor
from minichem_toolbox.backends import common


def test_child_environment_filters_secrets_and_uses_fixed_private_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """子进程不继承 API key，并固定使用作业目录下的私有 scratch。"""

    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "should-not-cross-boundary")
    monkeypatch.setenv("MINICHEM_SECRET_TOKEN", "should-not-cross-boundary")
    monkeypatch.setenv("MINICHEM_MODEL_CACHE", "/outside/untrusted-cache")
    monkeypatch.setenv("HF_HOME", "/outside/untrusted-huggingface")
    monkeypatch.setenv("PATH", "/trusted/bin")
    specification = {"job_id": "job-env", "runtime": "arche"}

    environment = job_supervisor._build_child_environment(specification, tmp_path)

    assert "ARCHE_CHEM_API_KEY" not in environment
    assert "MINICHEM_SECRET_TOKEN" not in environment
    assert "MINICHEM_MODEL_CACHE" not in environment
    assert "HF_HOME" not in environment
    assert environment["PATH"] == "/trusted/bin"
    assert environment["TMPDIR"] == str(tmp_path / ".tmp")
    assert environment["HOME"] == str(tmp_path / ".home")
    for directory in (tmp_path / ".tmp", tmp_path / ".home", tmp_path / ".tmp" / "matplotlib"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_child_environment_keeps_only_runtime_profile_cache_paths(
    tmp_path: Path,
) -> None:
    """模型缓存只接受可信 profile 声明且不能越出 profile 缓存根。"""

    cache_root = tmp_path / "model-cache"
    specification = {
        "job_id": "job-cache",
        "runtime": "fairchem",
        "runtime_profile_environment": {
            "MINICHEM_MODEL_CACHE": str(cache_root),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "HF_XET_CACHE": str(cache_root / "huggingface" / "xet"),
            "TORCH_HOME": str(cache_root / "torch"),
            "FAIRCHEM_CACHE_DIR": str(cache_root / "fairchem"),
        },
    }

    environment = job_supervisor._build_child_environment(specification, tmp_path)

    assert environment["HF_HOME"] == str((cache_root / "huggingface").resolve())
    assert environment["TORCH_HOME"] == str((cache_root / "torch").resolve())

    specification["runtime_profile_environment"]["HF_HOME"] = str(tmp_path / "escape")
    with pytest.raises(OSError, match="escapes"):
        job_supervisor._build_child_environment(specification, tmp_path / "invalid")


def test_pidfd_syscall_fallback_terminates_verified_supervisor_child(
    tmp_path: Path,
) -> None:
    """标准库没有 pidfd API 时，Linux syscall 仍能安全终止已核对 child。"""

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        identity = job_supervisor.capture_process_identity(process.pid)
        assert identity.start_time_ticks is not None
        assert job_supervisor._verified_process_group_signal(
            identity,
            signal.SIGTERM,
        )
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_atomic_json_uses_private_random_temporary_and_syncs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """原子 JSON 写入使用 0600 临时文件，完成后不留下固定 .tmp。"""

    fsync_calls: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda descriptor: fsync_calls.append(descriptor) or real_fsync(descriptor))
    target = tmp_path / "status.json"

    job_supervisor._atomic_json(target, {"status": "success"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "success"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("status.json.tmp"))
    assert len(fsync_calls) >= 2


def test_lock_open_rejects_symlink(tmp_path: Path) -> None:
    """锁文件 symlink 不能被打开或跟随写入。"""

    target = tmp_path / "target"
    target.write_text("outside", encoding="utf-8")
    lock = tmp_path / "gaussian.serial.lock"
    lock.symlink_to(target)

    with pytest.raises(OSError):
        with job_supervisor._open_private_lock(lock):
            pass


def test_run_external_filters_inherited_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_external 只继承白名单环境，但保留显式 Backend 覆盖。"""

    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "should-not-cross-boundary")
    monkeypatch.setenv("MINICHEM_SECRET_TOKEN", "should-not-cross-boundary")
    monkeypatch.setenv("LANG", "C")
    script = (
        "import json, os; "
        "print(json.dumps({k: os.environ.get(k) for k in "
        "['ARCHE_CHEM_API_KEY', 'MINICHEM_SECRET_TOKEN', 'LANG', 'EXPLICIT_VALUE']}))"
    )

    result = common.run_external(
        executable=sys.executable,
        arguments=["-c", script],
        directory=tmp_path,
        environment_overrides={"EXPLICIT_VALUE": "injected"},
    )

    assert result["returncode"] == 0
    values = json.loads(result["stdout"].strip())
    assert values["ARCHE_CHEM_API_KEY"] is None
    assert values["MINICHEM_SECRET_TOKEN"] is None
    assert values["LANG"] == "C"
    assert values["EXPLICIT_VALUE"] == "injected"


def test_external_termination_uses_pidfd_seam_instead_of_killpg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外部进程终止只调用稳定 pidfd seam，不在身份核对后裸调用 killpg。"""

    identity = common.ProcessIdentity(123, 456, "boot", 789, 790)
    process = type("FakeProcess", (), {"pid": 123})()
    monkeypatch.setattr(common, "_capture_live_process_identity", lambda _process: identity)
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: identity)
    monkeypatch.setattr(common, "_open_pidfd_for_signal", lambda _pid: 9)
    monkeypatch.setattr(common, "_send_pidfd_signal", lambda _fd, _signal: True)
    monkeypatch.setattr(common.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        common.os,
        "killpg",
        lambda *_args: pytest.fail("身份核对后不得直接 killpg"),
    )

    assert common._signal_verified_process_group(process, signal.SIGTERM)


def test_external_termination_fails_closed_when_identity_changes_after_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pidfd 打开后身份发生变化时不得继续发送终止信号。"""

    expected = common.ProcessIdentity(123, 456, "boot", 789, 790)
    replaced = common.ProcessIdentity(123, 457, "boot", 789, 790)
    process = type("FakeProcess", (), {"pid": 123})()
    sent: list[int] = []
    monkeypatch.setattr(common, "_capture_live_process_identity", lambda _process: expected)
    monkeypatch.setattr(common, "capture_process_identity", lambda _pid: replaced)
    monkeypatch.setattr(common, "_open_pidfd_for_signal", lambda _pid: 9)
    monkeypatch.setattr(common, "_send_pidfd_signal", lambda _fd, _signal: sent.append(1) or True)
    monkeypatch.setattr(common.os, "close", lambda _fd: None)

    assert not common._signal_verified_process_group(process, signal.SIGTERM)
    assert sent == []


def test_external_environment_requires_profile_for_explicit_cache_path(
    tmp_path: Path,
) -> None:
    """外部程序不得通过显式覆盖绕过 runtime profile 的缓存路径校验。"""

    with pytest.raises(OSError, match="runtime profile"):
        common._build_external_environment(
            tmp_path,
            {"HF_HOME": str(tmp_path / "untrusted-cache")},
        )


def test_backend_write_json_rejects_symlink_and_writes_private_file(
    tmp_path: Path,
) -> None:
    """通用 Backend JSON 写入同样使用原子私有文件。"""

    target = tmp_path / "outside.json"
    target.write_text("outside", encoding="utf-8")
    link = tmp_path / "result.json"
    link.symlink_to(target)

    with pytest.raises(OSError):
        common.write_json(tmp_path, "result.json", {"value": 1})
    assert target.read_text(encoding="utf-8") == "outside"

    link.unlink()
    result = common.write_json(tmp_path, "result.json", {"value": 1})
    assert json.loads(result.read_text(encoding="utf-8")) == {"value": 1}
    assert stat.S_IMODE(result.stat().st_mode) == 0o600
