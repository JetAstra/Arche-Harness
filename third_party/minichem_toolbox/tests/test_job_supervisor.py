"""MiniChem 本地 supervisor 的进展感知超时回归测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from minichem_mcp_tools import job_supervisor


def _write_supervisor_spec(
    root: Path,
    script: str,
    *,
    walltime_seconds: int = 1,
    job_type: str = "native_software",
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """创建真实 supervisor 测试所需的作业目录、spec 和输出文件。"""

    job_directory = root / "job"
    job_directory.mkdir()
    status_path = job_directory / "status.json"
    stdout_path = job_directory / "stdout.log"
    stderr_path = job_directory / "stderr.log"
    specification_path = root / "supervisor_spec.json"
    specification = {
        "schema_version": 1,
        "job_id": "job-progress-timeout",
        "job_type": job_type,
        "runtime": "core",
        "command": [sys.executable, "-c", script],
        "job_directory": str(job_directory),
        "relative_job_directory": "job",
        "status_path": str(status_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "relative_stdout_path": "job/stdout.log",
        "relative_stderr_path": "job/stderr.log",
        "stdin_path": None,
        "resource_limits": {
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
            "walltime_seconds": walltime_seconds,
        },
        "resource_allocation": {},
        "evaluation_resource_budget": {},
        "metadata": metadata or {},
        "submitted_at": "2026-01-01T00:00:00+00:00",
    }
    specification_path.write_text(json.dumps(specification), encoding="utf-8")
    status_path.write_text(json.dumps({"status": "queued"}), encoding="utf-8")
    return specification_path, status_path, stdout_path, stderr_path


def _read_status(status_path: Path) -> dict[str, Any]:
    """读取 supervisor 写入的最终结构化状态。"""

    return json.loads(status_path.read_text(encoding="utf-8"))


def test_supervisor_accepts_gaussian_archive_warning_as_logical_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gaussian 归档写入警告不应把正常完成的 native 作业标为失败。"""

    monkeypatch.delenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, raising=False)
    script = (
        "print('Unable to Open any file for archive entry.', flush=True)\n"
        "print('The archive entry for this job was punched.', flush=True)\n"
        "print('Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.', flush=True)\n"
        "raise SystemExit(1)\n"
    )
    specification_path, status_path, _, stderr_path = _write_supervisor_spec(
        tmp_path,
        script,
        metadata={"software_id": "gaussian"},
    )

    assert job_supervisor.supervise(specification_path) == 0

    status = _read_status(status_path)
    assert status["status"] == "success"
    assert status["return_code"] == 1
    assert status["error"] is None
    assert status["warning"] == {
        "code": "gaussian_archive_warning",
        "message": "Gaussian 正常终止，但归档条目写入返回了退出码 1",
    }
    assert status["provenance"] == {
        "gaussian_external_returncode": 1,
        "gaussian_termination": "normal",
        "gaussian_archive_warning": {
            "entry_marker": "Unable to Open any file for archive entry.",
            "punched_marker": "The archive entry for this job was punched.",
        },
    }
    assert stderr_path.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("metadata", "stdout_lines", "stderr_lines", "return_code"),
    [
        (
            {"software_id": "gaussian"},
            [
                "Unable to Open any file for archive entry.",
                "The archive entry for this job was punched.",
                "Normal termination of Gaussian 16",
            ],
            [],
            2,
        ),
        (
            {"software_id": "gaussian"},
            [
                "Unable to Open any file for archive entry.",
                "Fatal Error: archive writer failed.",
                "The archive entry for this job was punched.",
                "Normal termination of Gaussian 16",
            ],
            [],
            1,
        ),
        (
            {"software_id": "gaussian"},
            [
                "Unable to Open any file for archive entry.",
                "The archive entry for this job was punched.",
                "Normal termination of Gaussian 16",
            ],
            ["archive stderr"],
            1,
        ),
        (
            {"software_id": "gaussian"},
            [
                "The archive entry for this job was punched.",
                "Normal termination of Gaussian 16",
            ],
            [],
            1,
        ),
        (
            {"software_id": "orca"},
            [
                "Unable to Open any file for archive entry.",
                "The archive entry for this job was punched.",
                "Normal termination of Gaussian 16",
            ],
            [],
            1,
        ),
    ],
)
def test_supervisor_keeps_unsafe_gaussian_archive_warning_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, str],
    stdout_lines: list[str],
    stderr_lines: list[str],
    return_code: int,
) -> None:
    """归档 marker、错误输出、软件身份或退出码不安全时仍保持失败。"""

    monkeypatch.delenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, raising=False)
    script_lines = [
        "import sys",
        *[
            f"print({line!r}, flush=True)"
            for line in stdout_lines
        ],
        *[
            f"print({line!r}, file=sys.stderr, flush=True)"
            for line in stderr_lines
        ],
        f"raise SystemExit({return_code})",
    ]
    specification_path, status_path, _, _ = _write_supervisor_spec(
        tmp_path,
        "\n".join(script_lines),
        metadata=metadata,
    )

    assert job_supervisor.supervise(specification_path) == 4

    status = _read_status(status_path)
    assert status["status"] == "failed"
    assert status["return_code"] == return_code
    assert status["error"]["code"] == "nonzero_exit"


def test_supervisor_does_not_let_gaussian_archive_warning_mask_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """归档 warning 出现在 timeout 前也不能把超时作业提升为成功。"""

    monkeypatch.delenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, raising=False)
    script = (
        "import time\n"
        "print('Unable to Open any file for archive entry.', flush=True)\n"
        "print('The archive entry for this job was punched.', flush=True)\n"
        "print('Normal termination of Gaussian 16', flush=True)\n"
        "time.sleep(3)\n"
    )
    specification_path, status_path, _, _ = _write_supervisor_spec(
        tmp_path,
        script,
        walltime_seconds=1,
        metadata={"software_id": "gaussian"},
    )

    assert job_supervisor.supervise(specification_path) == 3

    status = _read_status(status_path)
    assert status["status"] == "timeout"
    assert status["error"]["code"] == "no_progress_timeout"


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_supervisor_keeps_flush_progressing_compute_alive(
    tmp_path: Path,
    stream: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """持续刷新任一输出流的 compute 作业不受普通 idle timeout 影响。"""

    monkeypatch.delenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, raising=False)
    output_statement = (
        "print(f'progress-{index}', flush=True)"
        if stream == "stdout"
        else "print(f'progress-{index}', file=sys.stderr, flush=True)"
    )
    script = (
        "import sys, time\n"
        "for index in range(18):\n"
        f"    {output_statement}\n"
        "    time.sleep(0.15)\n"
    )
    _, status_path, stdout_path, stderr_path = _write_supervisor_spec(
        tmp_path,
        script,
    )

    specification_path = tmp_path / "supervisor_spec.json"
    assert job_supervisor.supervise(specification_path) == 0

    status = _read_status(status_path)
    assert status["status"] == "success"
    assert status["timeout_policy"] == {
        "scope": "supervisor_actual_execution",
        "base_policy_field": "metadata.execution_timeout_policy",
        "execution_class": "compute",
        "mode": "inactivity_with_hard_cap",
        "progress_source": "stdout_stderr_file_or_process_group_cpu_ticks",
        "progress_signals": [
            "stdout_stderr_file_change",
            "process_group_cpu_ticks_increase",
        ],
        "progress_fields": ["size_bytes", "mtime_ns"],
        "cpu_ticks_source": "proc_stat_utime_plus_stime",
        "buffering_semantics": "only_flushed_file_bytes_are_observable",
        "inactivity_timeout_seconds": 1,
        "inactivity_timeout_source": "resource_limits.walltime_seconds",
        "hard_timeout_seconds": 604800,
        "hard_timeout_source": "default",
    }
    progress_path = stdout_path if stream == "stdout" else stderr_path
    assert progress_path.read_text(encoding="utf-8").count("progress-") >= 10


def test_supervisor_times_out_compute_without_observable_output_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有可观测 stdout/stderr 变化的 compute 作业按 idle timeout 终止。"""

    monkeypatch.delenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, raising=False)
    script = "import time\ntime.sleep(3)\n"
    specification_path, status_path, _, _ = _write_supervisor_spec(tmp_path, script)

    assert job_supervisor.supervise(specification_path) == 3

    status = _read_status(status_path)
    assert status["status"] == "timeout"
    assert status["termination_reason"] == "no_observable_progress"
    assert status["error"]["code"] == "no_progress_timeout"
    assert status["error"]["termination_reason"] == "no_observable_progress"
    assert status["error"]["timeout_policy"]["inactivity_timeout_seconds"] == 1
    assert (
        status["error"]["timeout_policy"]["buffering_semantics"]
        == "only_flushed_file_bytes_are_observable"
    )


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="需要 Linux /proc 进程信息")
def test_supervisor_keeps_cpu_progressing_compute_alive_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无日志但持续消耗 CPU 的 compute 作业不应被 idle timeout 终止。"""

    monkeypatch.delenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, raising=False)
    script = (
        "import time\n"
        "deadline = time.monotonic() + 2.4\n"
        "value = 0\n"
        "while time.monotonic() < deadline:\n"
        "    value += 1\n"
    )
    specification_path, status_path, stdout_path, stderr_path = _write_supervisor_spec(
        tmp_path,
        script,
    )

    assert job_supervisor.supervise(specification_path) == 0

    status = _read_status(status_path)
    assert status["status"] == "success"
    assert "process_group_cpu_ticks_increase" in status["timeout_policy"][
        "progress_signals"
    ]
    assert stdout_path.read_text(encoding="utf-8") == ""
    assert stderr_path.read_text(encoding="utf-8") == ""


def test_supervisor_hard_timeout_terminates_continuously_progressing_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """持续输出的 compute 作业达到独立 hard timeout 后仍会终止。"""

    monkeypatch.setenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, "2")
    script = (
        "import time\n"
        "for index in range(40):\n"
        "    print(f'progress-{index}', flush=True)\n"
        "    time.sleep(0.1)\n"
    )
    specification_path, status_path, stdout_path, _ = _write_supervisor_spec(
        tmp_path,
        script,
    )

    assert job_supervisor.supervise(specification_path) == 3

    status = _read_status(status_path)
    assert status["status"] == "timeout"
    assert status["termination_reason"] == "hard_timeout"
    assert status["error"]["code"] == "hard_timeout_exceeded"
    assert status["error"]["timeout_policy"]["hard_timeout_seconds"] == 2
    assert (
        status["error"]["timeout_policy"]["hard_timeout_source"]
        == job_supervisor.COMPUTE_HARD_TIMEOUT_ENV
    )
    assert stdout_path.read_text(encoding="utf-8").count("progress-") >= 10


def test_supervisor_preserves_wall_clock_timeout_for_non_compute_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式标记为非 compute 的旧作业继续使用原 wall-clock timeout。"""

    monkeypatch.delenv(job_supervisor.COMPUTE_HARD_TIMEOUT_ENV, raising=False)
    script = (
        "import time\n"
        "for index in range(18):\n"
        "    print(f'progress-{index}', flush=True)\n"
        "    time.sleep(0.15)\n"
    )
    specification_path, status_path, _, _ = _write_supervisor_spec(
        tmp_path,
        script,
        job_type="fast_action",
        metadata={"execution_timeout_policy": {"execution_class": "fast"}},
    )

    assert job_supervisor.supervise(specification_path) == 3

    status = _read_status(status_path)
    assert status["status"] == "timeout"
    assert status["termination_reason"] == "walltime_exceeded"
    assert status["error"]["code"] == "walltime_exceeded"
    assert status["metadata"]["execution_timeout_policy"] == {
        "execution_class": "fast"
    }
    assert status["timeout_policy"]["scope"] == "supervisor_actual_execution"
    assert status["timeout_policy"] == {
        "scope": "supervisor_actual_execution",
        "base_policy_field": "metadata.execution_timeout_policy",
        "execution_class": "fast",
        "mode": "wall_clock",
        "walltime_seconds": 1,
    }


def test_supervisor_keeps_running_when_process_group_cannot_be_proven_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """进程组扫描未知时不能写 terminal 状态或释放外部资源。"""

    script = "import time\ntime.sleep(0.05)\n"
    specification_path, status_path, _, _ = _write_supervisor_spec(
        tmp_path,
        script,
        walltime_seconds=2,
    )
    monkeypatch.setattr(
        job_supervisor,
        "inspect_process_group_membership",
        lambda _group_id, _session_id, _boot_id: "unknown",
    )

    assert job_supervisor.supervise(specification_path) == 1

    status = _read_status(status_path)
    assert status["status"] == "running"
    assert status["supervisor_exit_pending"] is True
    assert status["process_group_activity"] == "unknown"
    assert status["error"]["code"] == "process_group_identity_unavailable"
