import json
from pathlib import Path

from minichem_mcp_tools import open_execution
from minichem_mcp_tools.execution_models import (
    ExecutionResourceRequest,
    JobCollectRequest,
    JobStatusRequest,
    NativeJobRequest,
)
from minichem_mcp_tools.open_tools import TOOL_DESCRIPTIONS
from minichem_mcp_tools.software_catalog import native_command_guide
from minichem_toolbox.models import ResourceLimits


def test_get_execution_job_reconciles_stopped_gaussian_supervisor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """轮询退出中的 Gaussian 时应重读已由进程核对写成的终态。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    job_id = "job_" + "d" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    status_path = directory / "status.json"
    running = {
        "job_id": job_id,
        "job_type": "native_software",
        "status": "running",
        "supervisor_exit_pending": True,
        "metadata": {"software_id": "gaussian"},
    }
    status_path.write_text(json.dumps(running), encoding="utf-8")
    calls: list[None] = []

    def reconcile() -> None:
        calls.append(None)
        terminal = {
            **running,
            "status": "failed",
            "supervisor_exit_pending": False,
            "error": {"code": "memory_limit_exceeded", "message": "limit"},
        }
        status_path.write_text(json.dumps(terminal), encoding="utf-8")

    monkeypatch.setattr(
        open_execution,
        "reconcile_stopped_native_gaussian_jobs",
        reconcile,
    )

    _directory, status = open_execution._read_status(job_id)

    assert calls == [None]
    assert status["status"] == "failed"
    assert status["error"]["code"] == "memory_limit_exceeded"


def test_gaussian_irc_route_is_classified_as_reaction_path() -> None:
    deck = """#p b3lyp/6-31g(d) irc=(calcfc,forward)

IRC forward

0 1
H 0.0 0.0 0.0
H 0.0 0.0 0.74

"""

    parsed = open_execution._gaussian_segment_sections(deck, 1)

    assert parsed["calculation_intent"] == "reaction_path"


def test_gaussian_guide_exposes_typed_stdout_handoff_by_calculation_intent() -> None:
    guide = native_command_guide("gaussian", "g16")

    output_behavior = guide["output_behavior"]
    assert "GaussianIRCLog" in output_behavior
    assert "GaussianLog" in output_behavior
    assert "reaction_path" in output_behavior
    assert "transition_state" in output_behavior
    assert "optimization_frequency" in output_behavior
    assert "frozen calculation_intent" in output_behavior


def test_gaussian_irc_requires_a_pes_minimum_for_scientific_convergence(
    tmp_path: Path,
) -> None:
    (tmp_path / "stdout.log").write_text(
        "SCF Done: E(RB3LYP) = -1.100000 A.U.\n"
        "Calculation of FORWARD path complete.\n"
        "Normal termination of Gaussian 16\n",
        encoding="utf-8",
    )
    status = {
        "status": "success",
        "metadata": {
            "software_id": "gaussian",
            "calculation_intent": "reaction_path",
        },
    }

    software, convergence, artifacts = open_execution._native_scientific_axes(
        tmp_path, status
    )

    assert software["status"] == "normal_termination"
    assert convergence["status"] == "failed"
    assert "IRC branch completed" in convergence["evidence"]
    assert "IRC PES minimum detected" not in convergence["evidence"]
    assert artifacts["status"] == "valid"

    (tmp_path / "stdout.log").write_text(
        "SCF Done: E(RB3LYP) = -1.100000 A.U.\n"
        "PES minimum detected on this side of the pathway.\n"
        "Calculation of FORWARD path complete.\n"
        "Normal termination of Gaussian 16\n",
        encoding="utf-8",
    )

    _software, converged, _artifacts = open_execution._native_scientific_axes(
        tmp_path, status
    )
    assert converged["status"] == "converged"
    assert "IRC branch completed" in converged["evidence"]
    assert "IRC PES minimum detected" in converged["evidence"]


def test_native_gaussian_collect_validates_archive_warning_ts_opt_freq_and_nimag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """归档警告成功状态仍须通过 native TS 的 Opt/Freq/NImag=1 检查。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    job_id = "job_" + "a" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    stdout = (
        " SCF Done: E(RB3LYP) = -1.100000 A.U.\n"
        " Optimization completed.\n"
        " Harmonic frequencies (cm**-1)\n"
        " NImag=1\n"
        " Unable to Open any file for archive entry.\n"
        " The archive entry for this job was punched.\n"
        " Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.\n"
    )
    status = {
        "job_id": job_id,
        "job_type": "native_software",
        "status": "success",
        "return_code": 1,
        "error": None,
        "warning": {
            "code": "gaussian_archive_warning",
            "message": "Gaussian 正常终止，但归档条目写入返回了退出码 1",
        },
        "provenance": {
            "gaussian_external_returncode": 1,
            "gaussian_termination": "normal",
            "gaussian_archive_warning": {
                "entry_marker": "Unable to Open any file for archive entry.",
                "punched_marker": "The archive entry for this job was punched.",
            },
        },
        "metadata": {
            "software_id": "gaussian",
            "calculation_intent": "transition_state",
        },
    }
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (directory / "stdout.log").write_text(stdout, encoding="utf-8")
    (directory / "stderr.log").write_text("", encoding="utf-8")
    (directory / "request.json").write_text(
        json.dumps({"staged_inputs": []}), encoding="utf-8"
    )

    queried = open_execution.get_execution_job(JobStatusRequest(job_id=job_id))
    assert queried["job_status"] == "success"
    assert queried["terminal"] is True
    assert queried["warning"]["code"] == "gaussian_archive_warning"
    assert queried["provenance"]["gaussian_external_returncode"] == 1

    collected = open_execution.collect_execution_job(
        JobCollectRequest(job_id=job_id)
    )
    assert collected["ready"] is True
    assert collected["job_status"] == "success"
    assert collected["process_status"] == "completed"
    assert collected["software_status"] == "normal_termination"
    assert collected["convergence_status"] == "converged"
    assert collected["artifact_status"] == "valid"
    assert collected["scientific_validation_status"] == "mechanically_valid"
    manifest = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    assert manifest["warning"]["code"] == "gaussian_archive_warning"
    assert manifest["provenance"]["gaussian_external_returncode"] == 1


def test_native_gaussian_collect_keeps_true_failure_unvalidated(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """真正的 Gaussian 错误仍保持失败，不能由正常标记或归档字段放行。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    job_id = "job_" + "b" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    status = {
        "job_id": job_id,
        "job_type": "native_software",
        "status": "failed",
        "return_code": 1,
        "error": {"code": "nonzero_exit", "message": "Program exited with status 1"},
        "metadata": {
            "software_id": "gaussian",
            "calculation_intent": "transition_state",
        },
    }
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (directory / "stdout.log").write_text(
        " Fatal Error: SCF failed\n"
        " Normal termination of Gaussian 16\n",
        encoding="utf-8",
    )
    (directory / "stderr.log").write_text("", encoding="utf-8")
    (directory / "request.json").write_text(
        json.dumps({"staged_inputs": []}), encoding="utf-8"
    )

    queried = open_execution.get_execution_job(JobStatusRequest(job_id=job_id))
    assert queried["job_status"] == "failed"
    collected = open_execution.collect_execution_job(
        JobCollectRequest(job_id=job_id)
    )
    assert collected["job_status"] == "failed"
    assert collected["process_status"] == "failed"
    assert collected["software_status"] == "not_reached"
    assert collected["scientific_validation_status"] == "not_reached"


def test_native_gaussian_collect_rejects_returncode_one_without_archive_markers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """非完整归档 warning 不能借持久化 success 状态进入科学验证。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    job_id = "job_" + "c" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    status = {
        "job_id": job_id,
        "job_type": "native_software",
        "status": "success",
        "return_code": 1,
        "error": None,
        "metadata": {
            "software_id": "gaussian",
            "calculation_intent": "transition_state",
        },
    }
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (directory / "stdout.log").write_text(
        " Optimization completed.\n"
        " Harmonic frequencies (cm**-1)\n"
        " NImag=1\n"
        " Normal termination of Gaussian 16\n",
        encoding="utf-8",
    )
    (directory / "stderr.log").write_text("", encoding="utf-8")
    (directory / "request.json").write_text(
        json.dumps({"staged_inputs": []}), encoding="utf-8"
    )

    collected = open_execution.collect_execution_job(
        JobCollectRequest(job_id=job_id)
    )
    assert collected["job_status"] == "success"
    assert collected["software_status"] == "failed"
    assert collected["convergence_status"] == "converged"
    assert collected["scientific_validation_status"] == "mechanically_invalid"


def test_native_validation_returns_structured_resource_budget_error(monkeypatch) -> None:
    monkeypatch.setattr(
        open_execution,
        "native_command_guide",
        lambda *_args: {
            "software_id": "gaussian",
            "runtime": "gaussian",
            "resolved_path": "/configured/g16",
            "input_mode": "arguments",
        },
    )
    monkeypatch.setattr(
        open_execution, "_validate_native_input_deck", lambda *_args: None
    )

    response = open_execution.validate_native_job(
        NativeJobRequest(
            software_id="gaussian",
            executable="g16",
            resource_limits=ResourceLimits(
                cpu_cores=61,
                memory_mb=4096,
                gpu_count=0,
            ),
        )
    )

    assert response["status"] == "success"
    assert response["valid"] is True
    assert response["resource_limits"]["cpu_cores"] == 60
    assert response["resource_limits"]["memory_mb"] == 320000
    assert response["resource_policy"] == "service_managed_full_capacity"


def test_execution_resource_response_exposes_redacted_fifo_without_scope_claim(
    monkeypatch,
):
    """资源工具保留旧字段并提供全局脱敏 FIFO，不伪造调用者归属。"""

    queue = {
        "status": "available",
        "scope": "global",
        "items": [
            {
                "position": 1,
                "status": "queued",
                "owned": False,
            }
        ],
        "item_count": 1,
    }
    monkeypatch.setattr(
        open_execution,
        "_resource_availability",
        lambda: {
            "budget": {},
            "currently_reserved": {},
            "available": {},
            "active_jobs": [],
            "active_job_count": 0,
            "gaussian_queue": queue,
            "reservation_tracking": "workspace_reservations",
        },
    )
    monkeypatch.setattr(
        open_execution,
        "_read_frozen_catalog_identity",
        lambda: {"catalog_hash": "a" * 64, "catalog_version": "7"},
    )

    result = open_execution.get_execution_resources(ExecutionResourceRequest())

    assert result["status"] == "success"
    assert result["budget"] == {}
    assert result["active_jobs"] == []
    assert result["catalog_hash"] == "a" * 64
    assert result["catalog_version"] == "7"
    assert result["gaussian_queue"] == queue
    assert "job_id" not in result["gaussian_queue"]["items"][0]
    assert result["gaussian_queue"]["items"][0]["owned"] is False
    description = TOOL_DESCRIPTIONS["get_execution_resources"]
    assert "shared Gaussian FIFO" in description
    assert "owned=false" in description


def test_execution_resource_request_has_no_spoofable_scope_field() -> None:
    """MCP 请求合同不接受调用方伪造的 scope 或 owner 标识。"""

    schema = ExecutionResourceRequest.model_json_schema()
    assert schema["properties"] == {}
    assert schema["additionalProperties"] is False


def test_resource_availability_fails_closed_when_scheduler_cannot_be_read(
    monkeypatch,
) -> None:
    """共享 scheduler 无法核对时，资源查询不得把容量报告为空闲。"""

    budget = {
        "cpu_cores": 60,
        "memory_mb": 320000,
        "gpu_count": 1,
    }
    monkeypatch.setattr(open_execution, "resource_budget_record", lambda: budget)

    def fail_to_read_scheduler():
        raise RuntimeError("corrupt shared scheduler index")

    monkeypatch.setattr(
        open_execution, "active_resource_usage", fail_to_read_scheduler
    )

    result = open_execution._resource_availability()

    assert result["currently_reserved"] == budget
    assert result["available"] == {
        "cpu_cores": 0,
        "memory_mb": 0,
        "gpu_count": 0,
    }
    assert result["reservation_tracking"] == "unavailable_without_workspace"


def test_failed_gaussian_job_exposes_route_syntax_diagnostic(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """native Gaussian 的 QPErr 应给 Router 可修订的有限诊断。"""

    (tmp_path / "stdout.log").write_text(
        "QPErr --- A syntax error was detected in the input line.\n"
        " xCycles=200) Freq SCRF=(SMD,Solvent=Ethy\n"
        "                             '\n",
        encoding="utf-8",
    )
    (tmp_path / "stderr.log").write_text(
        "Segmentation fault (core dumped)\n",
        encoding="utf-8",
    )
    status = {
        "job_type": "native_software",
        "job_id": "job_00000000000000000000000000000001",
        "status": "failed",
        "return_code": 1,
        "metadata": {
            "software_id": "gaussian",
            "calculation_intent": "optimization_frequency",
            "input_deck_validation": {
                "segments": [
                    {
                        "route": (
                            "#p B3LYP/LANL2DZ Opt=(Tight,MaxCycles=200) Freq "
                            "SCRF=(SMD,Solvent=EthylAcetate)"
                        )
                    }
                ]
            },
        },
    }
    monkeypatch.setattr(
        open_execution,
        "_read_status",
        lambda _job_id: (tmp_path, status),
    )
    monkeypatch.setattr(
        open_execution,
        "_resource_availability",
        lambda: {"available": {"cpu_cores": 60, "memory_mb": 320000, "gpu_count": 0}},
    )

    response = open_execution.get_execution_job(
        JobStatusRequest(job_id=status["job_id"])
    )

    assert response == {
        "status": "success",
        "job_id": status["job_id"],
        "job_status": "failed",
        "terminal": True,
        "job": {"job_id": status["job_id"], "status": "failed"},
    }


def test_gaussian_qperr_does_not_guess_solvent_when_pointer_targets_another_field(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """route 含多个错误时，只能按 QPErr 指针归因当前字段。"""

    (tmp_path / "stdout.log").write_text(
        "QPErr --- A syntax error was detected in the input line.\n"
        " #p omegaB97X-D3/def2-TZVP Opt Freq\n"
        "    '\n",
        encoding="utf-8",
    )
    status = {
        "job_type": "native_software",
        "job_id": "job_00000000000000000000000000000002",
        "status": "failed",
        "metadata": {
            "software_id": "gaussian",
            "input_deck_validation": {
                "segments": [
                    {
                        "route": (
                            "#p omegaB97X-D3/def2-TZVP Opt Freq "
                            "SCRF=(SMD,Solvent=EthylAcetate)"
                        )
                    }
                ]
            },
        },
    }
    monkeypatch.setattr(
        open_execution,
        "_read_status",
        lambda _job_id: (tmp_path, status),
    )
    monkeypatch.setattr(open_execution, "_resource_availability", lambda: {})

    response = open_execution.get_execution_job(
        JobStatusRequest(job_id=status["job_id"])
    )

    assert response == {
        "status": "success",
        "job_id": status["job_id"],
        "job_status": "failed",
        "terminal": True,
        "job": {"job_id": status["job_id"], "status": "failed"},
    }
