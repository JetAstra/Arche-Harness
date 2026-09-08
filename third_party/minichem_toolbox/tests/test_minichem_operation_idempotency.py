"""MiniChem 原生与程序作业幂等提交边界测试。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess as subprocess_module
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from minichem_mcp_tools.execution_models import (
    AnalysisJobRequest,
    AnalysisOutputDeclaration,
    JobCollectRequest,
    JobStatusRequest,
    NativeJobRequest,
    StagedInput,
)
from minichem_mcp_tools import job_supervisor, open_execution


class FakeSupervisorProcess:
    """不启动外部程序，只提供提交器读取的 PID。"""

    def __init__(self, pid: int) -> None:
        self.pid = pid


def _native_request(
    operation_id: str | None = None,
    *,
    arguments: list[str] | None = None,
    label: str | None = None,
    parent_job_id: str | None = None,
    resource_limits: dict[str, int] | None = None,
) -> NativeJobRequest:
    return NativeJobRequest(
        software_id="xtb",
        executable="xtb",
        arguments=arguments or ["input.xyz"],
        staged_inputs=[
            StagedInput(source_path="code/input.xyz", target_path="input.xyz")
        ],
        label=label,
        parent_job_id=parent_job_id,
        operation_id=operation_id,
        resource_limits=resource_limits or {},
    )


def _analysis_request(operation_id: str | None = None) -> AnalysisJobRequest:
    return AnalysisJobRequest(
        runtime="core",
        script_path="code/analysis.py",
        script_target="code/agent_program.py",
        operation_id=operation_id,
    )


def _native_validation(request: NativeJobRequest) -> dict:
    return {
        "status": "success",
        "valid": True,
        "software_id": request.software_id,
        "runtime": "core",
        "resolved_executable": "/bin/true",
        "command": ["/bin/true", *request.arguments],
        "staged_targets": sorted(item.target_path for item in request.staged_inputs),
        "stdin_target": request.stdin_target,
        "input_deck_validation": None,
        "calculation_intent": request.calculation_intent or "other",
        "resource_limits": {
            **request.resource_limits.model_dump(mode="json"),
            "walltime_seconds": 60,
        },
        "invocation_guide": {"synopsis": "test native invocation"},
    }


def _analysis_validation(request: AnalysisJobRequest) -> dict:
    script = open_execution.resolve_workspace_path(request.script_path, must_exist=True)
    return {
        "status": "success",
        "valid": True,
        "runtime": request.runtime,
        "runtime_python": os.sys.executable,
        "script_source": request.script_path,
        "script_target": request.script_target,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "discovered_imports": [],
        "required_modules": request.required_modules,
        "required_module_versions": request.required_module_versions,
        "required_symbols": request.required_symbols,
        "module_status": {},
        "module_details": {},
        "input_inspection": {"status": "success", "inputs": []},
        "external_execution_findings": [],
        "job_context_compliance": {"status": "valid"},
        "analysis_contract": {
            "schema_version": 1,
            "inputs": [],
            "outputs": [item.model_dump(mode="json") for item in request.outputs],
            "execution_policy": request.execution_policy,
            "external_execution": request.external_execution,
        },
        "resource_limits": {
            **request.resource_limits.model_dump(mode="json"),
            "walltime_seconds": 60,
        },
    }


@pytest.fixture
def submission_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """为每个测试提供隔离工作区和不启动程序的 Popen。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "4")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "8192")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "input.xyz").write_text("1\nH\n", encoding="utf-8")
    (tmp_path / "code" / "analysis.py").write_text(
        "from researchchem_job import JobContext\n",
        encoding="utf-8",
    )

    calls: list[dict] = []
    calls_lock = threading.Lock()

    def fake_popen(command, **kwargs):
        with calls_lock:
            calls.append({"command": command, "kwargs": kwargs})
            return FakeSupervisorProcess(20_000 + len(calls))

    fake_subprocess = SimpleNamespace(
        Popen=fake_popen,
        DEVNULL=subprocess_module.DEVNULL,
        SubprocessError=subprocess_module.SubprocessError,
        run=subprocess_module.run,
    )
    monkeypatch.setattr(open_execution, "subprocess", fake_subprocess)
    monkeypatch.setattr(open_execution, "validate_native_job", _native_validation)
    monkeypatch.setattr(open_execution, "validate_analysis_program", _analysis_validation)
    return SimpleNamespace(root=tmp_path, calls=calls)


def test_job_environment_passes_hard_timeout_to_detached_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """hard timeout 配置沿安全白名单传给 detached supervisor，不放开其他环境变量。"""

    timeout_env = job_supervisor.COMPUTE_HARD_TIMEOUT_ENV
    monkeypatch.setenv(timeout_env, "604800")
    monkeypatch.setenv("MINICHEM_UNSAFE_TEST_ENV", "must-not-inherit")
    monkeypatch.setattr(open_execution, "runtime_environment", lambda _runtime: {})

    environment = open_execution._job_environment(
        "core",
        "job_environment_timeout",
        tmp_path,
        {"cpu_cores": 1, "gpu_count": 0},
        {},
        job_type="native_software",
    )

    assert environment[timeout_env] == "604800"
    assert timeout_env in open_execution.SAFE_INHERITED_ENVIRONMENT
    assert "MINICHEM_UNSAFE_TEST_ENV" not in environment


def test_operation_id_validation_is_path_safe_and_legacy_is_random(submission_sandbox):
    """校验 operation_id，且不传新字段时继续生成随机 job_id。"""

    assert _native_request("execute-step-operation_abc123").operation_id == (
        "execute-step-operation_abc123"
    )
    with pytest.raises(ValueError):
        _native_request("../escape")

    first = open_execution.submit_native_job(_native_request())
    second = open_execution.submit_native_job(_native_request())

    assert first["status"] == second["status"] == "success"
    assert first["job_id"] != second["job_id"]
    assert len(submission_sandbox.calls) == 2


def test_native_operation_id_reuses_one_job_and_receipt(submission_sandbox):
    """同一 native 请求重复提交时只启动一次并返回持久 receipt。"""

    request = _native_request("execute-step-operation_reuse")
    first = open_execution.submit_native_job(request)
    second = open_execution.submit_native_job(request)

    assert first["job_id"] == second["job_id"]
    assert second["idempotent_reused"] is True
    assert len(submission_sandbox.calls) == 1
    receipt = (
        submission_sandbox.root
        / first["job_directory"]
        / "operation_receipt.json"
    )
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["operation_id"] == request.operation_id
    assert value["job_id"] == first["job_id"]
    assert value["phase"] in {"started", "launching", "ready"}


def test_submission_does_not_overwrite_supervisor_status_update(
    submission_sandbox,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """supervisor 在提交返回前发布 running 时，父进程不能回写旧 queued 状态。"""

    def publish_running_status(command, **_kwargs):
        spec_path = Path(command[-1])
        specification = json.loads(spec_path.read_text(encoding="utf-8"))
        status_path = Path(specification["status_path"])
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({"status": "running", "supervisor_pid": 27001})
        status_path.write_text(json.dumps(status), encoding="utf-8")
        return FakeSupervisorProcess(27001)

    monkeypatch.setattr(open_execution.subprocess, "Popen", publish_running_status)
    result = open_execution.submit_native_job(
        _native_request("execute-step-operation-status-race")
    )

    status_path = submission_sandbox.root / result["job_directory"] / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["supervisor_pid"] == 27001


def test_operation_id_conflict_rejects_changed_native_request_without_side_effect(
    submission_sandbox,
):
    """同 key 改变 argv 时返回冲突，不能启动第二个作业。"""

    operation_id = "execute-step-operation_conflict"
    first = open_execution.submit_native_job(_native_request(operation_id))
    conflict = open_execution.submit_native_job(
        _native_request(operation_id, arguments=["changed.xyz"])
    )

    assert first["status"] == "success"
    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


@pytest.mark.parametrize("change", ["resource_limits", "label", "parent_job_id"])
def test_operation_id_conflict_rejects_native_metadata_changes(
    submission_sandbox, change: str
):
    """资源、标签和父作业都是幂等语义的一部分，变化时拒绝复用。"""

    operation_id = f"execute-step-operation-{change}"
    first = open_execution.submit_native_job(_native_request(operation_id))
    if change == "resource_limits":
        changed = _native_request(operation_id, resource_limits={"cpu_cores": 2})
    elif change == "label":
        changed = _native_request(operation_id, label="changed-label")
    else:
        changed = _native_request(operation_id, parent_job_id="job_" + "a" * 32)

    conflict = open_execution.submit_native_job(changed)

    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


def test_operation_id_fingerprint_detects_changed_input_content(submission_sandbox):
    """同一路径输入内容变化时不能错误复用旧作业。"""

    operation_id = "execute-step-operation_input-hash"
    first = open_execution.submit_native_job(_native_request(operation_id))
    (submission_sandbox.root / "code" / "input.xyz").write_text(
        "1\nF\n", encoding="utf-8"
    )
    conflict = open_execution.submit_native_job(_native_request(operation_id))

    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


def test_native_and_analysis_requests_cannot_share_operation_id(submission_sandbox):
    """native 与 programmable 两种执行层不能共用一个 operation_id。"""

    operation_id = "execute-step-operation_layer"
    first = open_execution.submit_native_job(_native_request(operation_id))
    conflict = open_execution.submit_analysis_program(_analysis_request(operation_id))

    assert first["status"] == "success"
    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


def test_analysis_operation_id_reuses_one_job(submission_sandbox):
    """同一 programmable 请求重复提交时只启动一次。"""

    request = _analysis_request("execute-step-operation_analysis")
    first = open_execution.submit_analysis_program(request)
    second = open_execution.submit_analysis_program(request)

    assert first["job_id"] == second["job_id"]
    assert second["idempotent_reused"] is True
    assert len(submission_sandbox.calls) == 1


def test_analysis_operation_id_conflict_rejects_script_content_change(
    submission_sandbox,
):
    """脚本内容变化时不能把同一 operation_id 误当成原作业。"""

    operation_id = "execute-step-operation-analysis-script"
    first = open_execution.submit_analysis_program(_analysis_request(operation_id))
    (submission_sandbox.root / "code" / "analysis.py").write_text(
        "from researchchem_job import JobContext\n# changed\n",
        encoding="utf-8",
    )
    conflict = open_execution.submit_analysis_program(_analysis_request(operation_id))

    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


def test_analysis_operation_id_conflict_rejects_output_contract_change(
    submission_sandbox,
):
    """声明输出路径或媒体类型变化时不能复用旧 programmable 作业。"""

    operation_id = "execute-step-operation-analysis-output"
    first_request = _analysis_request(operation_id)
    first = open_execution.submit_analysis_program(first_request)
    changed_request = first_request.model_copy(
        update={
            "outputs": [
                AnalysisOutputDeclaration(
                    name="result",
                    path="outputs/result.json",
                    semantic_type="JsonResult",
                    media_type="application/json",
                )
            ]
        }
    )
    conflict = open_execution.submit_analysis_program(changed_request)

    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


def test_concurrent_same_operation_id_has_one_job_and_one_supervisor(
    submission_sandbox,
):
    """并发双提交必须收束为同一 job 和一次 Popen。"""

    request = _native_request("execute-step-operation_concurrent")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                open_execution.submit_native_job,
                [request, request],
            )
        )

    assert len({item["job_id"] for item in results}) == 1
    assert {item["status"] for item in results} <= {
        "success",
        "submission_in_progress",
    }
    assert len(submission_sandbox.calls) == 1


def test_concurrent_conflicting_operation_id_has_one_side_effect(
    submission_sandbox,
):
    """并发的不同规范请求只有先到者能启动，另一者得到冲突。"""

    operation_id = "execute-step-operation-concurrent-conflict"
    requests = [
        _native_request(operation_id, arguments=["input.xyz"]),
        _native_request(operation_id, arguments=["changed.xyz"]),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(open_execution.submit_native_job, requests))

    assert sorted(item["status"] for item in results) == ["invalid_request", "success"]
    assert len(submission_sandbox.calls) == 1
    success = next(item for item in results if item["status"] == "success")
    conflict = next(item for item in results if item["status"] == "invalid_request")
    assert conflict["error"]["job_id"] == success["job_id"]


def test_toctou_input_change_before_launch_refuses_popen(
    submission_sandbox, monkeypatch: pytest.MonkeyPatch
):
    """准备完成后源文件变化必须形成失败收据，不能启动 supervisor。"""

    original_verify = open_execution._verify_expected_snapshots

    def mutate_then_verify(expected):
        (submission_sandbox.root / "code" / "input.xyz").write_text(
            "1\nCl\n", encoding="utf-8"
        )
        return original_verify(expected)

    monkeypatch.setattr(open_execution, "_verify_expected_snapshots", mutate_then_verify)
    result = open_execution.submit_native_job(
        _native_request("execute-step-operation-toctou")
    )

    assert result["status"] == "invalid_request"
    assert result["error"]["code"] == "input_changed_before_launch"
    assert len(submission_sandbox.calls) == 0


def _read_operation_receipt(root: Path, job_id: str) -> dict:
    return json.loads(
        (root / "outputs" / "execution_jobs" / job_id / "operation_receipt.json").read_text(
            encoding="utf-8"
        )
    )


def test_crash_before_ready_leaves_preparing_pause_without_retry_launch(
    submission_sandbox, monkeypatch: pytest.MonkeyPatch
):
    """mkdir/收据后、ready 前崩溃时重试只能返回技术暂停。"""

    original_stage_inputs = open_execution._stage_inputs

    def crash_before_ready(*_args, **_kwargs):
        raise SystemExit("injected-before-ready")

    monkeypatch.setattr(open_execution, "_stage_inputs", crash_before_ready)
    request = _native_request("execute-step-operation-crash-preparing")
    with pytest.raises(SystemExit, match="injected-before-ready"):
        open_execution.submit_native_job(request)
    monkeypatch.setattr(open_execution, "_stage_inputs", original_stage_inputs)

    retry = open_execution.submit_native_job(request)
    assert retry["status"] == "submission_in_progress"
    assert retry["error"]["code"] == "operation_record_incomplete"
    assert len(submission_sandbox.calls) == 0
    assert _read_operation_receipt(submission_sandbox.root, retry["job_id"])["phase"] == (
        "preparing"
    )


def test_crash_after_ready_before_launch_never_restarts(
    submission_sandbox, monkeypatch: pytest.MonkeyPatch
):
    """ready 收据后启动前崩溃时保留身份，不允许第二次 Popen。"""

    original_job_environment = open_execution._job_environment

    def crash_after_ready(*_args, **_kwargs):
        raise SystemExit("injected-after-ready")

    monkeypatch.setattr(open_execution, "_job_environment", crash_after_ready)
    request = _native_request("execute-step-operation-crash-ready")
    with pytest.raises(SystemExit, match="injected-after-ready"):
        open_execution.submit_native_job(request)
    monkeypatch.setattr(open_execution, "_job_environment", original_job_environment)

    retry = open_execution.submit_native_job(request)
    assert retry["status"] == "submission_in_progress"
    assert retry["error"]["code"] == "operation_record_incomplete"
    assert len(submission_sandbox.calls) == 0
    assert _read_operation_receipt(submission_sandbox.root, retry["job_id"])["phase"] == (
        "ready"
    )


def test_crash_after_popen_before_submission_checkpoint_never_restarts(
    submission_sandbox, monkeypatch: pytest.MonkeyPatch
):
    """Popen 成功但提交确认写入失败时，重试不得启动第二个 supervisor。"""

    original_checkpoint = open_execution._write_submission_checkpoint

    def crash_after_popen(*_args, **_kwargs):
        raise SystemExit("injected-after-popen")

    monkeypatch.setattr(open_execution, "_write_submission_checkpoint", crash_after_popen)
    request = _native_request("execute-step-operation-crash-popen")
    with pytest.raises(SystemExit, match="injected-after-popen"):
        open_execution.submit_native_job(request)
    monkeypatch.setattr(open_execution, "_write_submission_checkpoint", original_checkpoint)

    retry = open_execution.submit_native_job(request)
    assert retry["status"] == "submission_in_progress"
    assert retry["error"]["code"] == "operation_record_incomplete"
    assert len(submission_sandbox.calls) == 1
    assert _read_operation_receipt(submission_sandbox.root, retry["job_id"])["phase"] == (
        "launching"
    )


def test_missing_minichem_receipt_after_launch_is_safe_pause(
    submission_sandbox,
):
    """启动后删除 MiniChem 收据，重试只能安全暂停且不能再次 Popen。"""

    request = _native_request("execute-step-operation-checkpoint-gap")
    first = open_execution.submit_native_job(request)
    receipt = submission_sandbox.root / first["job_directory"] / "operation_receipt.json"
    receipt.unlink()

    retry = open_execution.submit_native_job(request)
    assert retry["status"] == "submission_in_progress"
    assert retry["error"]["code"] == "operation_record_incomplete"
    assert retry["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


def test_lost_submission_response_reuses_existing_job(
    submission_sandbox,
):
    """调用方丢失提交响应后重提仍复用已有作业。"""

    request = _native_request("execute-step-operation-checkpoint-reuse")
    first = open_execution.submit_native_job(request)

    retry = open_execution.submit_native_job(request)
    assert retry["status"] == "submission_in_progress"
    assert retry["error"]["code"] == "operation_record_incomplete"
    assert retry["idempotent_reused"] is True
    assert retry["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1


@pytest.mark.parametrize("status", ["queued", "running", "success", "failed"])
def test_repeated_operation_reads_existing_status_without_second_launch(
    submission_sandbox, status: str
):
    """已有 queued/running/terminal 状态重提时不再启动 supervisor。"""

    request = _native_request(f"execute-step-operation_status-{status}")
    first = open_execution.submit_native_job(request)
    status_path = submission_sandbox.root / first["status_path"]
    value = json.loads(status_path.read_text(encoding="utf-8"))
    value["status"] = status
    status_path.write_text(json.dumps(value), encoding="utf-8")
    calls_before = len(submission_sandbox.calls)

    second = open_execution.submit_native_job(request)

    assert second["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == calls_before


def test_get_and_collect_keep_job_id_contract(submission_sandbox):
    """幂等提交返回的 job_id 仍可直接交给 get/collect。"""

    result = open_execution.submit_native_job(
        _native_request("execute-step-operation_get-collect")
    )
    status = open_execution.get_execution_job(
        JobStatusRequest(job_id=result["job_id"])
    )
    collection = open_execution.collect_execution_job(
        JobCollectRequest(job_id=result["job_id"])
    )

    assert status["status"] == "success"
    assert status["job"]["job_id"] == result["job_id"]
    assert collection["status"] == "success"
    assert collection["job_id"] == result["job_id"]
    assert collection["ready"] is False


def _supervisor_spec(
    root: Path,
    *,
    operation_id: str | None = "operation-supervisor-test",
    command: list[str] | None = None,
    launch_file_snapshots: list[dict] | None = None,
) -> Path:
    """在临时目录写出可直接交给 supervisor 的最小可信作业描述。"""

    job_directory = root / "job"
    job_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = job_directory / "stdout.log"
    stderr_path = job_directory / "stderr.log"
    stdout_path.touch()
    stderr_path.touch()
    job_id = "job_" + "b" * 32
    fingerprint = "f" * 64
    if operation_id is not None:
        (job_directory / "operation_receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "operation_id": operation_id,
                    "fingerprint": fingerprint,
                    "job_id": job_id,
                    "phase": "launching",
                }
            ),
            encoding="utf-8",
        )
    spec = {
        "schema_version": 1,
        "job_id": job_id,
        "job_type": "native_software",
        "runtime": "core",
        "command": command or ["/bin/true"],
        "job_directory": str(job_directory),
        "relative_job_directory": "job",
        "status_path": str(job_directory / "status.json"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "relative_stdout_path": "job/stdout.log",
        "relative_stderr_path": "job/stderr.log",
        "stdin_path": None,
        "resource_limits": {"cpu_cores": 1, "memory_mb": 256, "gpu_count": 0, "walltime_seconds": 30},
        "resource_allocation": {},
        "evaluation_resource_budget": {},
        "metadata": {},
        "submitted_at": "2026-01-01T00:00:00+00:00",
    }
    if operation_id is not None:
        spec.update(
            {
                "operation_id": operation_id,
                "operation_fingerprint": fingerprint,
            }
        )
    if launch_file_snapshots is not None:
        spec["launch_file_snapshots"] = launch_file_snapshots
    spec_path = root / "supervisor_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    (job_directory / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "job_type": "native_software",
                "status": "queued",
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def _read_supervisor_status(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return json.loads(Path(spec["status_path"]).read_text(encoding="utf-8"))


def test_supervisor_claim_before_child_start_is_safe_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """claim 后 supervisor 崩溃时仍保持 launching，不能声称 child 已启动。"""

    spec_path = _supervisor_spec(tmp_path)
    original_claim = job_supervisor._claim_supervisor

    def claim_then_crash(specification, job_directory):
        original_claim(specification, job_directory)
        raise SystemExit("injected-before-child-popen")

    monkeypatch.setattr(job_supervisor, "_claim_supervisor", claim_then_crash)
    with pytest.raises(SystemExit, match="injected-before-child-popen"):
        job_supervisor.supervise(spec_path)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    receipt = json.loads(
        (Path(spec["job_directory"]) / "operation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["phase"] == "launching"
    assert _read_supervisor_status(spec_path)["status"] == "queued"


def test_duplicate_supervisor_does_not_overwrite_running_status(tmp_path: Path):
    """重复 supervisor 发现 claim 时不能把首个作业改成 failed。"""

    spec_path = _supervisor_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    job_directory = Path(spec["job_directory"])
    (job_directory / "supervisor_claim.json").write_text(
        json.dumps(
            {
                "operation_id": spec["operation_id"],
                "operation_fingerprint": spec["operation_fingerprint"],
                "job_id": spec["job_id"],
                "supervisor_pid": 123,
            }
        ),
        encoding="utf-8",
    )
    original = {
        "schema_version": 1,
        "job_id": spec["job_id"],
        "status": "running",
        "child_pid": 456,
    }
    (job_directory / "status.json").write_text(json.dumps(original), encoding="utf-8")

    result = job_supervisor.supervise(spec_path)

    assert result in {0, 1}
    assert _read_supervisor_status(spec_path) == original


def test_supervisor_legacy_status_omits_null_operation_fields(tmp_path: Path):
    """没有 operation_id 的旧调用保持原状态形状，不添加 null 字段。"""

    spec_path = _supervisor_spec(tmp_path, operation_id=None)
    assert job_supervisor.supervise(spec_path) == 0
    status = _read_supervisor_status(spec_path)

    assert "operation_id" not in status
    assert "operation_fingerprint" not in status
    assert status["status"] == "success"


@pytest.mark.parametrize("role", ["native_executable", "analysis_runtime_python"])
def test_supervisor_rechecks_launch_executable_before_child_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
):
    """启动前外部程序快照改变时返回结构化失败，不能启动 child。"""

    executable = tmp_path / "allowlisted-program"
    executable.write_text("original\n", encoding="utf-8")
    first = executable.stat()
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    snapshot = {
        "role": role,
        "path": str(executable),
        "size_bytes": first.st_size,
        "sha256": digest,
    }
    command = [str(executable)] if role == "native_executable" else [str(executable), "script.py"]
    spec_path = _supervisor_spec(
        tmp_path,
        command=command,
        launch_file_snapshots=[snapshot],
    )
    executable.write_text("changed-before-launch\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        raise AssertionError("child Popen must not run after snapshot mismatch")

    monkeypatch.setattr(job_supervisor.subprocess, "Popen", fake_popen)
    assert job_supervisor.supervise(spec_path) == 1
    status = _read_supervisor_status(spec_path)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "execution_file_changed_before_launch"
    assert calls == []


def test_supervisor_real_harmless_child_reaches_terminal_status(tmp_path: Path):
    """真实 supervisor + /bin/true child 能形成 terminal 状态，不启动 Gaussian。"""

    spec_path = _supervisor_spec(tmp_path, operation_id="operation-real-child")
    assert job_supervisor.supervise(spec_path) == 0
    status = _read_supervisor_status(spec_path)
    assert status["status"] == "success"
    receipt = json.loads(
        (Path(json.loads(spec_path.read_text())["job_directory"]) / "operation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["phase"] == "success"


def test_claim_crash_submit_retry_returns_pause_without_second_supervisor(
    submission_sandbox, monkeypatch: pytest.MonkeyPatch
):
    """真实 submit seam 重提 claim 崩溃作业时只能暂停且不能二次 Popen。"""

    request = _native_request("execute-step-operation-claim-crash")
    original_claim = job_supervisor._claim_supervisor

    def claim_then_crash(specification, job_directory):
        original_claim(specification, job_directory)
        raise SystemExit("injected-claim-crash")

    monkeypatch.setattr(job_supervisor, "_claim_supervisor", claim_then_crash)

    def launch_supervisor(command, **kwargs):
        submission_sandbox.calls.append({"command": command, "kwargs": kwargs})
        return job_supervisor.supervise(Path(command[-1]))

    monkeypatch.setattr(open_execution.subprocess, "Popen", launch_supervisor)
    with pytest.raises(SystemExit, match="injected-claim-crash"):
        open_execution.submit_native_job(request)

    retry = open_execution.submit_native_job(request)
    assert retry["status"] == "submission_in_progress"
    assert retry["error"]["code"] == "operation_record_incomplete"
    assert re.fullmatch(r"job_[0-9a-f]{8}", retry["job_id"])
    assert len(submission_sandbox.calls) == 1


def test_operation_fingerprint_uses_validated_walltime_policy(
    submission_sandbox, monkeypatch: pytest.MonkeyPatch
):
    """timeout policy 变化时有效 walltime 改变，不能复用同一 operation_id。"""

    walltime = 60

    def validation_with_policy(request):
        result = _native_validation(request)
        result["resource_limits"]["walltime_seconds"] = walltime
        return result

    monkeypatch.setattr(open_execution, "validate_native_job", validation_with_policy)
    request = _native_request("execute-step-operation-timeout-policy")
    first = open_execution.submit_native_job(request)
    walltime = 120
    conflict = open_execution.submit_native_job(request)

    assert conflict["status"] == "invalid_request"
    assert conflict["error"]["code"] == "operation_id_conflict"
    assert conflict["error"]["job_id"] == first["job_id"]
    assert len(submission_sandbox.calls) == 1
