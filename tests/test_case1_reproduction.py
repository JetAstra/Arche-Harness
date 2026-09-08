from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "tasks" / "case1_reproduce.py"


def _load_case1_module():
    spec = importlib.util.spec_from_file_location("reproduce_case1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case1_supplementary_data_reproduces_paper_gap() -> None:
    module = _load_case1_module()
    report, _payload = module.reproduce_supplementary_data(
        REPOSITORY_ROOT / "supplementary" / "case1.json"
    )

    assert report["status"] == "passed_with_source_discrepancies"
    assert report["candidate_count"] == 14
    assert report["major_transition_state"] == "TS-2a"
    assert report["minor_transition_state"] == "TS-2b-1"
    assert report["computed_delta_delta_g_kcal_mol"] == pytest.approx(
        1.8848503028766672
    )
    assert report["source_discrepancies"]["paper_verified_ts_count"] == 12
    assert report["source_discrepancies"]["supplementary_json_candidate_count"] == 14
    assert report["source_discrepancies"]["supplementary_json_TS-2c_kcal_mol"] == pytest.approx(
        1.933984169496982
    )
    assert all(
        row["imaginary_frequency_count"] == 1 for row in report["ranking"]
    )
    assert report["checks"][
        "all_candidate_frequency_records_have_one_imaginary_mode"
    ] is True
    assert "all_candidates_are_first_order_saddles" not in report["checks"]


def test_case1_sensitivity_pairs_routes_and_reuses_explicit_thermal_corrections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_case1_module()
    _report, payload = module.reproduce_supplementary_data(
        REPOSITORY_ROOT / "supplementary" / "case1.json"
    )
    route_offsets = {
        name: index * 0.0001
        for index, name in enumerate(module.GAUSSIAN_SENSITIVITY_ROUTES)
    }

    def fake_run(
        _executable,
        work_root,
        name,
        record,
        **_kwargs,
    ):
        route_name = Path(work_root).name
        major_energy = -1000.0 + route_offsets[route_name]
        energy = major_energy + (0.002 if name == module.PAPER_MINOR_TS else 0.0)
        return {
            "candidate": name,
            "status": "completed",
            "energy_hartree": energy,
            "supplementary_energy_hartree": module._last_scf_energy_ev(record)
            / module.HARTREE_TO_EV,
        }

    monkeypatch.setattr(module, "_run_exact_gaussian_candidate", fake_run)
    result = module.run_gaussian_sensitivity_validation(
        payload,
        executable=tmp_path / "g16",
        work_root=tmp_path,
        cpu_cores=1,
        memory_mb=128,
        workers=3,
        timeout_seconds=1.0,
    )

    assert result["status"] == "completed"
    assert len(result["routes"]) == len(module.GAUSSIAN_SENSITIVITY_ROUTES)
    assert all(
        [row["candidate"] for row in route["calculations"]]
        == [module.PAPER_MAJOR_TS, module.PAPER_MINOR_TS]
        for route in result["routes"]
    )
    expected_electronic_gap = 0.002 * module.HARTREE_TO_KCAL_MOL
    assert all(
        route["electronic_gap_kcal_mol"] == pytest.approx(expected_electronic_gap)
        for route in result["routes"]
    )
    assert all(
        route["fixed_thermal_composite_gap_kcal_mol"]
        != pytest.approx(route["electronic_gap_kcal_mol"])
        for route in result["routes"]
    )


def test_case1_minichem_hessian_runner_uses_exact_route_and_projected_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_case1_module()
    _report, payload = module.reproduce_supplementary_data(
        REPOSITORY_ROOT / "supplementary" / "case1.json"
    )
    requests = []

    def fake_execute(action_id, request):
        requests.append((action_id, request))
        if action_id == "calculate_hessian":
            frame_structure = json.loads(
                json.dumps(request["inputs"]["structure"])
            )
            frame_structure["atoms"][0]["position_angstrom"] = [6.0, 5.0, 4.0]
            return {
                "status": "success",
                "backend_version": "16 C.01",
                "result": {
                    "matrix": [[1.0]],
                    "unit": "hartree/bohr^2",
                    "structure": frame_structure,
                },
            }
        assert action_id == "derive_vibrational_modes"
        return {
            "status": "success",
            "result": {
                "frequencies_cm1": [
                    {"real": 0.0, "imaginary": 250.0},
                    *({"real": 100.0, "imaginary": 0.0} for _ in range(152)),
                ]
            },
        }

    monkeypatch.setattr(
        module,
        "_local_minichem_executor",
        lambda *_args, **_kwargs: fake_execute,
    )
    result = module.run_minichem_hessian_validation(
        payload,
        repository_root=REPOSITORY_ROOT,
        workspace=tmp_path / "workspace",
        result_root=tmp_path / "results",
        candidates=[module.PAPER_MAJOR_TS],
        cpu_cores=2,
        memory_mb=4096,
        workers=1,
        timeout_seconds=100,
    )

    assert result["status"] == "completed"
    assert result["all_candidates_have_one_imaginary_mode"] is True
    assert "all_candidates_are_first_order_saddles" not in result
    assert result["calculations"][0]["vibrational_mode_count"] == 153
    assert result["calculations"][0]["imaginary_frequency_count"] == 1
    hessian_request = requests[0][1]
    assert hessian_request["method_spec"]["solvation_model"] == "iefpcm"
    assert "integration_grid" not in hessian_request["method_spec"]
    assert hessian_request["action_settings"]["scf_convergence"] == "default"
    assert requests[1][1]["inputs"]["structure"]["atoms"][0][
        "position_angstrom"
    ] == [6.0, 5.0, 4.0]


def test_case1_minichem_ts_runner_reoptimizes_before_hessian(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_case1_module()
    _report, payload = module.reproduce_supplementary_data(
        REPOSITORY_ROOT / "supplementary" / "case1.json"
    )
    requests = []

    def fake_execute(action_id, request):
        requests.append((action_id, request))
        if action_id == "locate_transition_state":
            optimized = dict(request["inputs"]["initial_guess"])
            optimized["atoms"] = [dict(atom) for atom in optimized["atoms"]]
            optimized["atoms"][0]["position_angstrom"] = [9.0, 8.0, 7.0]
            return {
                "status": "success",
                "backend_version": "16 C.01",
                "result": {"structure": optimized, "converged": True},
            }
        if action_id == "calculate_hessian":
            assert request["inputs"]["structure"]["atoms"][0][
                "position_angstrom"
            ] == [9.0, 8.0, 7.0]
            frame_structure = json.loads(
                json.dumps(request["inputs"]["structure"])
            )
            frame_structure["atoms"][0]["position_angstrom"] = [1.0, 2.0, 3.0]
            return {
                "status": "success",
                "backend_version": "16 C.01",
                "result": {
                    "matrix": [[1.0]],
                    "unit": "hartree/bohr^2",
                    "structure": frame_structure,
                },
            }
        assert action_id == "derive_vibrational_modes"
        return {
            "status": "success",
            "result": {
                "frequencies_cm1": [
                    {"real": 0.0, "imaginary": 250.0},
                    *({"real": 100.0, "imaginary": 0.0} for _ in range(152)),
                ]
            },
        }

    monkeypatch.setattr(
        module,
        "_local_minichem_executor",
        lambda *_args, **_kwargs: fake_execute,
    )
    result = module.run_minichem_ts_optimization_validation(
        payload,
        repository_root=REPOSITORY_ROOT,
        workspace=tmp_path / "workspace",
        result_root=tmp_path / "results",
        candidates=[module.PAPER_MAJOR_TS],
        cpu_cores=2,
        memory_mb=4096,
        workers=1,
        timeout_seconds=100,
    )

    assert result["status"] == "completed"
    assert result["evidence_level"] == "E2"
    assert [action_id for action_id, _request in requests] == [
        "locate_transition_state",
        "calculate_hessian",
        "derive_vibrational_modes",
    ]
    ts_request = requests[0][1]
    assert ts_request["action_settings"] == {
        "scf_convergence": "default",
        "no_eigen_test": True,
        "calculate_all_force_constants": True,
        "max_step": 5,
    }
    assert ts_request["method_spec"]["solvation_model"] == "iefpcm"
    assert requests[2][1]["inputs"]["structure"]["atoms"][0][
        "position_angstrom"
    ] == [1.0, 2.0, 3.0]


def test_case1_minichem_irc_runner_is_atomic_and_bidirectional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_case1_module()
    _report, payload = module.reproduce_supplementary_data(
        REPOSITORY_ROOT / "supplementary" / "case1.json"
    )
    requests = []

    def fake_execute(action_id, request):
        requests.append((action_id, request))
        return {
            "status": "success",
            "result": {
                "converged": True,
                "termination_reason": "pes_minimum_detected",
                "completed_points": 12,
                "path_structure_count": 13,
            },
            "warnings": [],
        }

    monkeypatch.setattr(
        module,
        "_local_minichem_executor",
        lambda *_args, **_kwargs: fake_execute,
    )
    result = module.run_minichem_irc_validation(
        payload,
        repository_root=REPOSITORY_ROOT,
        workspace=tmp_path / "workspace",
        result_root=tmp_path / "results",
        candidates=[module.PAPER_MAJOR_TS],
        cpu_cores=2,
        memory_mb=4096,
        workers=2,
        timeout_seconds=100,
    )

    assert result["status"] == "completed"
    assert {row["direction"] for row in result["calculations"]} == {
        "forward",
        "backward",
    }
    assert {request["action_settings"]["direction"] for _, request in requests} == {
        "forward",
        "backward",
    }
    assert all(
        request["action_settings"]["scf_convergence"] == "default"
        and request["method_spec"]["solvation_model"] == "iefpcm"
        for _, request in requests
    )


def test_requested_stage_summary_fails_closed() -> None:
    module = _load_case1_module()

    summary = module._requested_stage_summary(
        {
            "hessian": {"status": "completed"},
            "irc": {"status": "incomplete"},
        },
        ["hessian", "irc"],
    )

    assert summary["all_requested_stages_completed"] is False
    assert summary["incomplete_stages"] == {"irc": "incomplete"}


def test_direct_gaussian_timeout_handles_already_exited_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_case1_module()
    _report, payload = module.reproduce_supplementary_data(
        REPOSITORY_ROOT / "supplementary" / "case1.json"
    )

    class TimedOutProcess:
        pid = 987654
        returncode = 0

        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def communicate(self, input=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("g16", timeout)
            return "partial gaussian output\n", "partial gaussian error\n"

    monkeypatch.setattr(module.subprocess, "Popen", TimedOutProcess)
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError),
    )
    executable = tmp_path / "g16"
    executable.touch()

    with pytest.raises(RuntimeError, match="Gaussian timed out"):
        module._run_exact_gaussian_candidate(
            executable,
            tmp_path / "run",
            module.PAPER_MAJOR_TS,
            payload["sp"][module.PAPER_MAJOR_TS],
            cpu_cores=1,
            memory_mb=128,
            timeout_seconds=1.0,
        )

    candidate_root = tmp_path / "run" / module.PAPER_MAJOR_TS
    assert (candidate_root / "job.log").read_text() == "partial gaussian output\n"
    assert (candidate_root / "job.err").read_text() == "partial gaussian error\n"
