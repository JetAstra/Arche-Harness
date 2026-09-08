"""覆盖 programmable_analysis 的 vendored bwrap 进程边界。"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from minichem_mcp_tools import job_supervisor, open_execution
from minichem_mcp_tools.execution_models import (
    AnalysisJobRequest,
    AnalysisOutputDeclaration,
    JobCollectRequest,
    JobStatusRequest,
    StagedInput,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CODEX_RUNTIME_ROOT = (
    REPOSITORY_ROOT
    / "local_data"
    / "runtime"
    / "codex_cli"
    / "0.149.0"
    / "linux-x64"
)
BWRAP_PATH = (
    CODEX_RUNTIME_ROOT
    / "vendor"
    / "x86_64-unknown-linux-musl"
    / "codex-resources"
    / "bwrap"
)


def _write_supervisor_spec(
    root: Path,
    script: str,
    *,
    job_type: str = "programmable_analysis",
    walltime_seconds: int = 10,
    metadata: dict[str, Any] | None = None,
    resource_limits: dict[str, int] | None = None,
    resource_allocation: dict[str, Any] | None = None,
    child_environment: dict[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    """创建一个只用于测试的作业目录、Python 程序和 supervisor spec。"""

    job_directory = root / "job"
    for name in ("code", "inputs", "outputs", "report", "logs"):
        (job_directory / name).mkdir(parents=True, exist_ok=True)
    (job_directory / "code" / "probe.py").write_text(script, encoding="utf-8")
    stdout_path = job_directory / "stdout.log"
    stderr_path = job_directory / "stderr.log"
    stdout_path.touch()
    stderr_path.touch()
    status_path = job_directory / "status.json"
    status_path.write_text("{}", encoding="utf-8")
    specification_path = root / "supervisor_spec.json"
    specification = {
        "schema_version": 1,
        "job_id": "job_programmable_analysis_sandbox",
        "job_type": job_type,
        "runtime": "core",
        "command": [str(Path(sys.executable).absolute()), "code/probe.py"],
        "job_directory": str(job_directory),
        "relative_job_directory": "job",
        "status_path": str(status_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "relative_stdout_path": "job/stdout.log",
        "relative_stderr_path": "job/stderr.log",
        "stdin_path": None,
        "resource_limits": resource_limits
        or {
            "cpu_cores": 1,
            "memory_mb": 4096,
            "gpu_count": 0,
            "walltime_seconds": walltime_seconds,
        },
        "resource_allocation": resource_allocation or {},
        "evaluation_resource_budget": {},
        "metadata": metadata or {},
        "child_environment": child_environment or {},
        "submitted_at": "2026-01-01T00:00:00+00:00",
    }
    specification_path.write_text(
        json.dumps(specification),
        encoding="utf-8",
    )
    return specification_path, status_path, job_directory


@pytest.fixture
def fixed_bwrap_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """把测试 supervisor 绑定到仓库固定的 bwrap，不使用 PATH 查找。"""

    if not BWRAP_PATH.is_file() or not os.access(BWRAP_PATH, os.X_OK):
        pytest.skip("仓库固定的 Codex 0.149.0 bwrap 尚未物化")
    digest = hashlib.sha256(BWRAP_PATH.read_bytes()).hexdigest()
    monkeypatch.setenv(
        job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT_ENV,
        str(CODEX_RUNTIME_ROOT),
    )
    monkeypatch.setenv(job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_ENV, str(BWRAP_PATH))
    monkeypatch.setenv(
        job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_SHA256_ENV,
        digest,
    )


def _read_status(path: Path) -> dict[str, Any]:
    """读取 supervisor 的结构化终态。"""

    return json.loads(path.read_text(encoding="utf-8"))


def test_programmable_analysis_bwrap_keeps_python_shell_and_job_io(
    tmp_path: Path,
    fixed_bwrap_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合法 Python、shell 和当前 job 读写可用，宿主敏感路径不可见。"""

    outside = tmp_path / "outside-run-bundle"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("outside", encoding="utf-8")
    host_pid = os.getpid()
    monkeypatch.setenv("SANDBOX_SECRET_SENTINEL", "must-not-cross")
    network_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    network_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    network_server.bind(("127.0.0.1", 0))
    network_server.listen(1)
    network_port = network_server.getsockname()[1]

    def reply_to_network_probe() -> None:
        """为 child 提供一个仅用于验证 share-net 的本机 TCP 响应。"""

        try:
            network_server.settimeout(5)
            connection, _address = network_server.accept()
            with connection:
                if connection.recv(64) == b"network-probe":
                    connection.sendall(b"network-ok")
        except OSError:
            pass

    network_thread = threading.Thread(target=reply_to_network_probe, daemon=True)
    network_thread.start()
    script = f'''\
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

def can_open(path):
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False

job_output = Path(os.environ["MINICHEM_JOB_OUTPUTS"])
result = {{
    "secret": os.environ.get("SANDBOX_SECRET_SENTINEL"),
    "auth_openable": can_open("/root/.codex/auth.json"),
    "outside_openable": can_open({str(outside / "sentinel.txt")!r}),
    "host_proc_openable": can_open("/proc/{host_pid}/environ"),
    "self_proc_openable": can_open("/proc/self/environ"),
    "shell_returncode": subprocess.run(
        ["/bin/sh", "-c", "printf shell-ok > outputs/shell.txt"],
        check=False,
    ).returncode,
    "network_response": None,
}}
with socket.create_connection(
    ("127.0.0.1", int(os.environ["MINICHEM_TEST_PORT"])), timeout=2
) as connection:
    connection.sendall(b"network-probe")
    result["network_response"] = connection.recv(32).decode()
result["job_write"] = True
(Path(os.environ["MINICHEM_JOB_INPUTS"]) / "input.txt").write_text("input", encoding="utf-8")
(Path(os.environ["MINICHEM_JOB_REPORT"]) / "report.txt").write_text("report", encoding="utf-8")
(Path(os.environ["TMPDIR"]) / "tmp.txt").write_text("tmp", encoding="utf-8")
(Path(os.environ["HOME"]) / "home.txt").write_text("home", encoding="utf-8")
(job_output / "result.json").write_text(json.dumps(result), encoding="utf-8")
print("stdout-ok", flush=True)
print("stderr-ok", file=sys.stderr, flush=True)
print(json.dumps({{key: value for key, value in result.items() if key != "secret"}}), flush=True)
'''
    try:
        specification, status_path, job_directory = _write_supervisor_spec(
            tmp_path,
            script,
            child_environment={"MINICHEM_TEST_PORT": str(network_port)},
        )

        assert job_supervisor.supervise(specification) == 0
        status = _read_status(status_path)
        result = json.loads((job_directory / "outputs" / "result.json").read_text())
        assert status["status"] == "success"
        assert result["secret"] is None
        assert result["auth_openable"] is False
        assert result["outside_openable"] is False
        assert result["host_proc_openable"] is False
        assert result["self_proc_openable"] is True
        assert result["shell_returncode"] == 0
        assert result["network_response"] == "network-ok"
        assert result["job_write"] is True
        assert (job_directory / "outputs" / "shell.txt").read_text() == "shell-ok"
        assert (job_directory / "inputs" / "input.txt").read_text() == "input"
        assert (job_directory / "report" / "report.txt").read_text() == "report"
        assert (job_directory / ".tmp" / "tmp.txt").read_text() == "tmp"
        assert (job_directory / ".home" / "home.txt").read_text() == "home"
        assert "stdout-ok" in (job_directory / "stdout.log").read_text()
        assert "stderr-ok" in (job_directory / "stderr.log").read_text()
        for field in (
            "child_pid",
            "child_start_time_ticks",
            "child_boot_id",
            "child_process_group_id",
            "child_session_id",
            "supervisor_pid",
            "supervisor_start_time_ticks",
            "supervisor_boot_id",
            "supervisor_process_group_id",
            "supervisor_session_id",
        ):
            assert status[field]
    finally:
        network_server.close()
        network_thread.join(timeout=5)
        assert not network_thread.is_alive()


def test_programmable_analysis_bwrap_missing_is_fail_closed(
    tmp_path: Path,
    fixed_bwrap_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bwrap 缺失时只生成原始结构化失败，绝不裸执行 Python。"""

    monkeypatch.setenv(
        job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_ENV,
        str(tmp_path / "missing-bwrap"),
    )
    specification, status_path, job_directory = _write_supervisor_spec(
        tmp_path,
        "from pathlib import Path\nPath('outputs/runs-without-sandbox').write_text('bad')\n",
    )

    assert job_supervisor.supervise(specification) == 1
    status = _read_status(status_path)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "programmable_analysis_sandbox_path_missing"
    assert not (job_directory / "outputs" / "runs-without-sandbox").exists()


def test_standard_python_symlink_chain_stays_inside_fixed_venv(
    tmp_path: Path,
) -> None:
    """标准 ``.venv/bin/python`` 软链接可用，但越界链不会被接受。"""

    standard_link = REPOSITORY_ROOT / ".venv" / "bin" / "python"
    assert standard_link.is_symlink()
    runtime_root = job_supervisor._runtime_python_root_for_programmable_analysis(
        [str(standard_link), "-c", "pass"]
    )
    assert runtime_root == REPOSITORY_ROOT / ".venv"

    outside_link = tmp_path / "python"
    outside_link.symlink_to(standard_link)
    with pytest.raises(job_supervisor._ProgrammableAnalysisSandboxError):
        job_supervisor._runtime_python_root_for_programmable_analysis(
            [str(outside_link), "-c", "pass"]
        )


def test_programmable_analysis_bwrap_hash_tamper_is_fail_closed(
    tmp_path: Path,
    fixed_bwrap_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定摘要被配置篡改时不执行程序。"""

    monkeypatch.setenv(
        job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_SHA256_ENV,
        "0" * 64,
    )
    specification, status_path, job_directory = _write_supervisor_spec(
        tmp_path,
        "from pathlib import Path\nPath('outputs/runs-without-sandbox').write_text('bad')\n",
    )

    assert job_supervisor.supervise(specification) == 1
    status = _read_status(status_path)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "programmable_analysis_sandbox_integrity_failed"
    assert not (job_directory / "outputs" / "runs-without-sandbox").exists()


@pytest.mark.parametrize("job_type", ["native_software", "transition_state_path"])
def test_non_programmable_jobs_do_not_use_bwrap(
    tmp_path: Path,
    job_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native 和 TS 作业保持原有直接 child 路径，即使 bwrap 配置缺失。"""

    monkeypatch.delenv(job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_ENV, raising=False)
    monkeypatch.delenv(
        job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_RUNTIME_ROOT_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        job_supervisor.PROGRAMMABLE_ANALYSIS_BWRAP_SHA256_ENV,
        raising=False,
    )
    specification, status_path, _job_directory = _write_supervisor_spec(
        tmp_path,
        "print('native-path', flush=True)\n",
        job_type=job_type,
    )
    value = json.loads(specification.read_text(encoding="utf-8"))
    value["command"] = [str(Path(sys.executable).absolute()), "-c", "print('ok', flush=True)"]
    specification.write_text(json.dumps(value), encoding="utf-8")

    assert job_supervisor.supervise(specification) == 0
    assert _read_status(status_path)["status"] == "success"


def test_programmable_analysis_bwrap_preserves_timeout_state(
    tmp_path: Path,
    fixed_bwrap_environment: None,
) -> None:
    """bwrap 包装后的 Python 仍沿用 supervisor 的 timeout 监控和终态。"""

    specification, status_path, _job_directory = _write_supervisor_spec(
        tmp_path,
        "import time\ntime.sleep(3)\n",
        walltime_seconds=1,
    )

    assert job_supervisor.supervise(specification) == 3
    status = _read_status(status_path)
    assert status["status"] == "timeout"
    assert status["error"]["code"] == "no_progress_timeout"


def test_programmable_analysis_bwrap_preserves_cancel_and_identity_state(
    tmp_path: Path,
    fixed_bwrap_environment: None,
) -> None:
    """外部取消仍由 supervisor 核验身份后终止 bwrap child。"""

    specification, status_path, job_directory = _write_supervisor_spec(
        tmp_path,
        "import time\ntime.sleep(30)\n",
        walltime_seconds=30,
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "minichem_mcp_tools.job_supervisor", str(specification)],
        cwd=REPOSITORY_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = _read_status(status_path)
            if status.get("status") == "running":
                break
            if status.get("status") in {"failed", "timeout", "cancelled", "success"}:
                pytest.fail(f"supervisor ended before cancellation: {status}")
            time.sleep(0.05)
        else:
            pytest.fail("supervisor did not reach running state")
        (job_directory / "cancel_requested").touch()
        assert supervisor.wait(timeout=10) == 2
        status = _read_status(status_path)
        assert status["status"] == "cancelled"
        assert status["error"]["code"] == "cancelled"
        assert status["child_pid"] > 0
        assert status["child_process_group_id"] > 0
        assert status["child_session_id"] > 0
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=10)


def test_real_submit_get_collect_uses_the_same_programmable_sandbox(
    tmp_path: Path,
    fixed_bwrap_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 submit/get/collect 链路在 bwrap 内完成轻量 Python 作业。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "analysis.py").write_text(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "path = ctx.output('result')\n"
        "path.write_text('ok', encoding='utf-8')\n"
        "ctx.register_output('result')\n",
        encoding="utf-8",
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/analysis.py",
        script_target="code/agent_program.py",
        outputs=[
            AnalysisOutputDeclaration(
                name="result",
                path="outputs/result.txt",
                semantic_type="Text",
                media_type="text/plain",
            )
        ],
    )

    submitted = open_execution.submit_analysis_program(request)
    assert submitted["status"] == "success"
    job_id = submitted["job_id"]
    terminal = None
    for _ in range(100):
        observed = open_execution.get_execution_job(JobStatusRequest(job_id=job_id))
        if observed["terminal"]:
            terminal = observed
            break
        time.sleep(0.05)
    assert terminal is not None
    assert terminal["job_status"] == "success"
    collected = open_execution.collect_execution_job(
        JobCollectRequest(job_id=job_id)
    )
    assert collected["status"] == "success"
    assert collected["job_id"] == job_id
    assert collected["ready"] is True


def test_real_submit_imports_job_context_when_workspace_contains_colons(
    tmp_path: Path,
    fixed_bwrap_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """workspace 路径含冒号时，分析程序仍能导入作业上下文模块。"""

    workspace = tmp_path / "combination:task:batch"
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(workspace))
    (workspace / "code").mkdir(parents=True)
    workspace.chmod(0o700)
    (workspace / "code" / "analysis.py").write_text(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "path = ctx.output('result')\n"
        "path.write_text('ok', encoding='utf-8')\n"
        "ctx.register_output('result')\n",
        encoding="utf-8",
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/analysis.py",
        script_target="code/agent_program.py",
        outputs=[
            AnalysisOutputDeclaration(
                name="result",
                path="outputs/result.txt",
                semantic_type="Text",
                media_type="text/plain",
            )
        ],
    )

    submitted = open_execution.submit_analysis_program(request)
    assert submitted["status"] == "success"
    job_id = submitted["job_id"]
    terminal = None
    for _ in range(100):
        observed = open_execution.get_execution_job(JobStatusRequest(job_id=job_id))
        if observed["terminal"]:
            terminal = observed
            break
        time.sleep(0.05)
    assert terminal is not None
    assert terminal["job_status"] == "success"
    collected = open_execution.collect_execution_job(
        JobCollectRequest(job_id=job_id)
    )
    assert collected["status"] == "success"
    assert collected["job_id"] == job_id
    assert collected["ready"] is True
    assert all(
        not item["job_relative_path"].endswith("researchchem_job.py")
        for item in collected["outputs"]
    )


def test_analysis_request_reserves_job_context_next_to_script() -> None:
    """调用方不能用 staging 文件覆盖脚本目录中的作业上下文模块。"""

    with pytest.raises(ValueError, match="researchchem_job.py is reserved"):
        AnalysisJobRequest(
            runtime="core",
            script_path="code/analysis.py",
            script_target="code/agent_program.py",
            staged_inputs=[
                StagedInput(
                    source_path="inputs/replacement.py",
                    target_path="code/researchchem_job.py",
                )
            ],
        )
