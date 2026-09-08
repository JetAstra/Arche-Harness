from __future__ import annotations

import json
from pathlib import Path

import pytest

import minichem_mcp_tools.open_execution as open_execution
import minichem_toolbox.service as service_module
from minichem_mcp_tools.execution_models import JobCollectRequest
from minichem_toolbox.backends import reaction
from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.service import execute_action


TRANSITION_STATE = {
    "atoms": [
        {"element": "C", "position_angstrom": [0.0, 0.0, 0.0]},
        {"element": "O", "position_angstrom": [1.2, 0.0, 0.0]},
    ],
    "charge": 0,
    "multiplicity": 1,
    "pbc": [False, False, False],
}


def _orientation(first: float, second: float) -> str:
    return f"""Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        {first:.6f}    0.000000    0.000000
      2          8           0        {second:.6f}    0.000000    0.000000
 ---------------------------------------------------------------------
"""


def _irc_log(
    direction: str = "FORWARD",
    *,
    complete: bool = True,
    minimum: bool = True,
    normal: bool = True,
    points: bool = True,
) -> str:
    point_blocks = ""
    if points:
        point_blocks = (
            f"Point Number  1 in {direction} path direction.\n"
            + _orientation(0.1, 1.3)
            + " SCF Done:  E(RB3LYP) =  -10.1000000000     A.U.\n"
            + " CHANGE IN THE REACTION COORDINATE = 0.2\n"
            + " # OF POINTS ALONG THE PATH = 1\n"
            + f"Point Number  2 in {direction} path direction.\n"
            + _orientation(0.2, 1.4)
            + " SCF Done:  E(RB3LYP) =  -10.2000000000     A.U.\n"
            + " CHANGE IN THE REACTION COORDINATE = 0.2\n"
            + " # OF POINTS ALONG THE PATH = 2\n"
        )
    summary = ""
    if points:
        summary = (
            " Summary of reaction path following\n"
            " -------------------------------\n"
            " Energy      Rx Coord\n"
            " 1 -10.100000 0.200000\n"
            " 2 -10.200000 0.400000\n"
            " -------------------------------\n"
        )
    termination = ""
    if minimum:
        termination += " PES minimum detected on this side of the pathway.\n"
    if complete:
        termination += f" Calculation of {direction} path complete.\n"
    elif points:
        termination += " Maximum number of steps reached.\n"
    footer = " Normal termination of Gaussian 16\n" if normal else " Error termination via Lnk1e\n"
    return (
        " SCF Done:  E(RB3LYP) =  -10.0000000000     A.U.\n"
        + point_blocks
        + summary
        + termination
        + footer
    )


def _run_public_parse_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    log_text: str,
    direction: str = "forward",
) -> dict:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    log_path = tmp_path / "irc.stdout.log"
    log_path.write_text(log_text, encoding="utf-8")

    class Reservation:
        resource_allocation = {
            "cpu_ids": [0],
            "gpu_ids": [],
            "cpu_cores": 1,
            "memory_mb": 4096,
            "gpu_count": 0,
        }

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        service_module,
        "probe_all_backends",
        lambda specifications: {
            item.id: {"available": True, "status": "available"}
            for item in specifications
        },
    )
    monkeypatch.setattr(service_module, "reserve_resources", lambda *_args, **_kwargs: Reservation())
    monkeypatch.setattr(
        service_module,
        "invoke_worker",
        lambda **kwargs: reaction.execute(
            kwargs["payload"]["action_id"],
            kwargs["payload"]["backend_id"],
            kwargs["payload"]["request"],
        ),
    )
    monkeypatch.setattr(
        reaction,
        "run_external",
        lambda **_kwargs: pytest.fail("解析已完成日志不应再次运行 Gaussian"),
    )
    return execute_action(
        "parse_gaussian_irc_output",
        {
            "backend_id": "internal_reaction_analysis",
            "inputs": {
                "gaussian_irc_log": "irc.stdout.log",
                "transition_state": TRANSITION_STATE,
            },
            "action_settings": {"direction": direction},
        },
    )


def test_parse_gaussian_irc_public_action_returns_typed_path_and_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response = _run_public_parse_action(monkeypatch, tmp_path, _irc_log())

    assert response["status"] == "success"
    assert response["output_artifacts"][0]["semantic_type"] == "ReactionPath"
    result = response["result"]
    assert result["direction"] == "forward"
    assert result["energy_unit"] == "hartree"
    assert result["completed_points"] == 2
    assert result["branch_completion"] is True
    assert result["pes_minimum_detected"] is True
    assert result["termination_reason"] == "pes_minimum_detected"
    assert result["summary"]
    assert result["normal_termination"] is True
    assert result["transition_state"] == TRANSITION_STATE
    assert len(result["structures"]) == 3
    assert result["energies_hartree"] == pytest.approx([-10.0, -10.1, -10.2])
    assert result["transition_state_validated"] is False

    validation = reaction.execute(
        "validate_reaction_path",
        "internal_reaction_analysis",
        {
            "inputs": {
                "path": result,
                "reactant": TRANSITION_STATE,
                "product": result["structures"][-1],
            },
            "action_settings": {
                "endpoint_rmsd_tolerance_angstrom": 0.01,
                "maximum_image_step_rmsd_angstrom": 1.0,
                "bond_distance_tolerance_angstrom": 0.1,
            },
        },
    )
    assert validation["status"] == "success"
    assert validation["result"]["valid"] is True


def test_parse_gaussian_irc_public_action_supports_backward_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response = _run_public_parse_action(
        monkeypatch, tmp_path, _irc_log(direction="REVERSE"), "backward"
    )

    assert response["status"] == "success"
    assert response["result"]["direction"] == "backward"
    assert response["result"]["observed_native_directions"] == ["reverse"]
    assert response["result"]["completed_points"] == 2


@pytest.mark.parametrize(
    ("log_text", "direction", "expected_status", "expected_reason"),
    [
        (_irc_log(direction="REVERSE"), "forward", "failed", "direction_mismatch"),
        (_irc_log(points=False), "forward", "failed", "no_parseable_points"),
        (_irc_log(normal=False), "forward", "failed", "abnormal_termination"),
        (_irc_log(minimum=False), "forward", "partial_success", "branch_completed_without_pes_minimum"),
        (_irc_log(complete=False, minimum=False), "forward", "partial_success", "maximum_points_reached"),
    ],
)
def test_parse_gaussian_irc_rejects_incomplete_or_mismatched_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    log_text: str,
    direction: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    response = _run_public_parse_action(monkeypatch, tmp_path, log_text, direction)

    assert response["status"] == expected_status
    result = response.get("result") or {}
    assert result.get("termination_reason") == expected_reason
    assert result.get("branch_completion") is not True
    assert result.get("transition_state_validated") is not True


def test_native_gaussian_collect_typed_stdout_uses_frozen_intent_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    job_id = "job_" + "1" * 32
    directory = tmp_path / "outputs" / "execution_jobs" / job_id
    directory.mkdir(parents=True)
    (directory / "stdout.log").write_text(_irc_log(), encoding="utf-8")
    (directory / "stderr.log").write_text("", encoding="utf-8")
    status = {
        "job_id": job_id,
        "job_type": "native_software",
        "status": "success",
        "metadata": {"software_id": "gaussian", "calculation_intent": "reaction_path"},
    }
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (directory / "request.json").write_text(
        json.dumps({"staged_inputs": []}), encoding="utf-8"
    )

    collected = open_execution.collect_execution_job(JobCollectRequest(job_id=job_id))
    stdout_record = next(item for item in collected["outputs"] if item["job_relative_path"] == "stdout.log")
    assert stdout_record["semantic_type"] == "GaussianIRCLog"

    (directory / "stdout.log").write_text("没有任何可供内容猜测的日志标记。", encoding="utf-8")
    for intent in ("transition_state", "optimization_frequency"):
        status["metadata"] = {"software_id": "gaussian", "calculation_intent": intent}
        (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
        collected = open_execution.collect_execution_job(JobCollectRequest(job_id=job_id))
        stdout_record = next(
            item for item in collected["outputs"] if item["job_relative_path"] == "stdout.log"
        )
        assert stdout_record["semantic_type"] == "GaussianLog"

    status["metadata"] = {"software_id": "orca", "calculation_intent": "reaction_path"}
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    collected = open_execution.collect_execution_job(JobCollectRequest(job_id=job_id))
    stdout_record = next(item for item in collected["outputs"] if item["job_relative_path"] == "stdout.log")
    assert "semantic_type" not in stdout_record

    status["metadata"] = {"software_id": "gaussian", "calculation_intent": "single_point"}
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    collected = open_execution.collect_execution_job(JobCollectRequest(job_id=job_id))
    stdout_record = next(item for item in collected["outputs"] if item["job_relative_path"] == "stdout.log")
    assert "semantic_type" not in stdout_record


def test_gaussian_irc_parser_contract_adds_one_action_and_no_backend() -> None:
    validate_catalog()
    assert "parse_gaussian_irc_output" in action_specs()
    assert "parse_gaussian_irc_output" in backend_specs()["internal_reaction_analysis"].capabilities
    assert len([item for item in backend_specs() if item == "internal_reaction_analysis"]) == 1
