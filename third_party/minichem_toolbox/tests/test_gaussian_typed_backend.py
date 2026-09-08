from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import minichem_toolbox.service as service_module
from minichem_toolbox.backends import electronic, quantum_legacy, reaction
from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox.process_identity import ProcessIdentity
from minichem_toolbox.service import execute_action
from minichem_toolbox.idempotency import scoped_submission_operation_id
from minichem_toolbox.scope import ScopeContext, activate_scope


CASE1_STRUCTURE = {
    "atoms": [
        {
            "element": "C",
            "position_angstrom": [0.0, 0.0, 0.0],
            "atom_map_number": 3,
            "formal_charge": 0,
            "component_index": 0,
        },
        {
            "element": "O",
            "position_angstrom": [1.2, 0.0, 0.0],
            "atom_map_number": 5,
            "formal_charge": 0,
            "component_index": 1,
        },
    ],
    "bonds": [
        {
            "atom_index_a": 0,
            "atom_index_b": 1,
            "bond_order": 1.0,
            "aromatic": False,
        }
    ],
    "atom_map_numbers": [3, 5],
    "charge": 1,
    "multiplicity": 1,
    "pbc": [False, False, False],
    "case1_candidate_id": "TS-2a",
}


def test_scoped_gaussian_execution_derives_stable_owner_and_job_identity(
    tmp_path,
) -> None:
    """scope 内 Gaussian 必须从 submission key 派生稳定 owner 和 job。"""

    scope = ScopeContext(
        task_id="task-case3",
        batch_id="batch-case3",
        combination_id="combination-case3",
        scope_id="scope-case3",
        workspace=tmp_path,
        capability_hash="0" * 64,
    )
    plan = SimpleNamespace(
        request=SimpleNamespace(submission_key="case3-unsubstituted-scan-v1")
    )

    with activate_scope(scope):
        owner_id, job_id = service_module._derive_gaussian_execution_identity(plan)
        expected_owner = scoped_submission_operation_id(
            "case3-unsubstituted-scan-v1"
        )
    assert owner_id == expected_owner
    assert job_id == "job_" + hashlib.sha256(
        expected_owner.encode("ascii")
    ).hexdigest()[:32]


def test_gaussian_hessian_returns_fchk_frame_structure_and_matching_modes(
    monkeypatch, tmp_path
):
    """A Gaussian standard-orientation Hessian must travel with that structure."""

    rotated_input = json.loads(json.dumps(CASE1_STRUCTURE))
    rotated_input["atoms"][1]["position_angstrom"] = [0.0, 1.2, 0.0]

    # The mocked formatted checkpoint is the same diatomic rotated back onto X.
    # Its only internal Hessian eigenvector is an imaginary X-axis bond stretch.
    stretch = [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    matrix = [
        [-left * right for right in stretch]
        for left in stretch
    ]
    packed = [
        matrix[row][column]
        for row in range(len(matrix))
        for column in range(row + 1)
    ]
    bond_bohr = 1.2 / quantum_legacy.BOHR_TO_ANGSTROM
    fchk_text = "\n".join(
        [
            "Number of atoms                            I                2",
            "Atomic numbers                             I   N=           2",
            " 6 8",
            "Current cartesian coordinates              R   N=           6",
            f" 0.0D+00 0.0D+00 0.0D+00 {bond_bohr:.12E} 0.0D+00",
            " 0.0D+00",
            "Cartesian Force Constants                  R   N=          21",
            *[
                " ".join(f"{value:.12E}".replace("E", "D") for value in packed[index:index + 5])
                for index in range(0, len(packed), 5)
            ],
            "",
        ]
    )
    gaussian_stdout = """ SCF Done:  E(RB3LYP) =  -100.000000000     A.U.
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""

    def fake_run_external(**kwargs):
        if kwargs["executable"] == "g16":
            (tmp_path / "job.chk").write_bytes(b"mock-checkpoint")
            return {
                "available": True,
                "returncode": 0,
                "stdout": gaussian_stdout,
                "stderr": "",
                "command": ["g16"],
            }
        assert kwargs["executable"] == "formchk"
        assert kwargs["directory"] == tmp_path
        assert kwargs["arguments"] == ["job.chk", "job.fchk"]
        (kwargs["directory"] / kwargs["arguments"][1]).write_text(
            fchk_text, encoding="utf-8"
        )
        return {
            "available": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "command": ["formchk", *kwargs["arguments"]],
        }

    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(quantum_legacy, "run_external", fake_run_external)
    monkeypatch.setattr(
        quantum_legacy, "relative_workspace_path", lambda path: path.name
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])

    response = quantum_legacy.gaussian(
        "calculate_hessian",
        {
            "inputs": {"structure": rotated_input},
            "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
            "action_settings": {"scf_convergence": "default"},
            "resource_limits": {"cpu_cores": 1, "memory_mb": 1024},
        },
    )

    assert response["status"] == "success"
    hessian_result = response["result"]
    frame_structure = hessian_result["structure"]
    assert rotated_input["atoms"][1]["position_angstrom"] == [0.0, 1.2, 0.0]
    assert frame_structure["atoms"][1]["position_angstrom"] == pytest.approx(
        [1.2, 0.0, 0.0]
    )
    assert frame_structure["charge"] == CASE1_STRUCTURE["charge"]
    assert frame_structure["multiplicity"] == CASE1_STRUCTURE["multiplicity"]
    assert frame_structure["bonds"] == CASE1_STRUCTURE["bonds"]
    assert frame_structure["atom_map_numbers"] == [3, 5]
    assert [atom["atom_map_number"] for atom in frame_structure["atoms"]] == [3, 5]
    assert hessian_result["coordinate_frame"] == {
        "source": "gaussian_fchk_current_cartesian_coordinates",
        "fchk_coordinate_unit": "bohr",
        "structure_coordinate_unit": "angstrom",
        "hessian_and_structure_share_cartesian_frame": True,
        "hessian_was_rotated_to_input_frame": False,
    }

    # Keep this regression independent of the optional ASE installation while
    # still exercising MiniChem's real projected-mode implementation.
    import sys
    import types

    fake_ase = types.ModuleType("ase")
    fake_ase.units = types.SimpleNamespace(
        _hbar=1.0,
        m=1.0,
        _e=1.0,
        _amu=1.0,
        invcm=1.0,
    )
    monkeypatch.setitem(sys.modules, "ase", fake_ase)

    class FakeAtoms:
        def __init__(self, structure):
            self._positions = [
                atom["position_angstrom"] for atom in structure["atoms"]
            ]
            self._symbols = [atom["element"] for atom in structure["atoms"]]
            self.cell = types.SimpleNamespace(
                array=[[0.0, 0.0, 0.0] for _ in range(3)]
            )
            self.pbc = [False, False, False]
            self.info = {
                "charge": structure.get("charge", 0),
                "multiplicity": structure.get("multiplicity", 1),
            }

        def __len__(self):
            return len(self._positions)

        def get_masses(self):
            return [12.0, 16.0]

        def get_positions(self):
            return self._positions

        def get_chemical_symbols(self):
            return self._symbols

    monkeypatch.setattr(electronic, "ase_atoms", FakeAtoms)
    vibrations = electronic._vibrations(
        {
            "inputs": {
                "structure": frame_structure,
                "hessian": hessian_result,
            },
            "action_settings": {"linearity": "linear"},
        }
    )
    assert vibrations["status"] == "success"
    assert vibrations["result"]["vibrational_mode_count"] == 1
    mode = vibrations["result"]["modes"][0]
    assert max(abs(atom[0]) for atom in mode) > 0.1
    assert max(abs(atom[1]) for atom in mode) < 1.0e-12
    assert max(abs(atom[2]) for atom in mode) < 1.0e-12


def test_gaussian_optimization_preserves_case1_molecular_identity(
    monkeypatch, tmp_path
):
    structure = CASE1_STRUCTURE
    stdout = """ SCF Done:  E(RB3LYP) =  -100.123456789     A.U.
 Optimization completed.
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.100000    0.200000    0.300000
      2          8           0        1.300000    0.400000    0.500000
 ---------------------------------------------------------------------
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""

    monkeypatch.setattr(
        quantum_legacy,
        "output_directory",
        lambda _action_id, _backend_id: tmp_path,
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["g16"],
        },
    )
    monkeypatch.setattr(
        quantum_legacy,
        "relative_workspace_path",
        lambda path: path.name,
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])

    response = quantum_legacy.gaussian(
        "optimize_geometry",
        {
            "inputs": {"structure": structure},
            "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
            "action_settings": {
                "scf_convergence": "Tight",
                "optimization_convergence": "Tight",
                "max_steps": 100,
            },
            "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
        },
    )

    assert response["status"] == "success"
    optimized = response["result"]["structure"]
    assert optimized["charge"] == 1
    assert optimized["multiplicity"] == 1
    assert optimized["bonds"] == structure["bonds"]
    assert optimized["atom_map_numbers"] == [3, 5]
    assert optimized["case1_candidate_id"] == "TS-2a"
    assert [atom["atom_map_number"] for atom in optimized["atoms"]] == [3, 5]
    assert [atom["component_index"] for atom in optimized["atoms"]] == [0, 1]
    assert optimized["atoms"][0]["position_angstrom"] == [0.1, 0.2, 0.3]
    assert optimized["atoms"][1]["position_angstrom"] == [1.3, 0.4, 0.5]


@pytest.mark.parametrize(
    ("action_id", "input_name", "settings"),
    [
        (
            "optimize_geometry",
            "structure",
            {
                "scf_convergence": "Tight",
                "optimization_convergence": "Tight",
                "max_steps": 100,
            },
        ),
        (
            "locate_transition_state",
            "initial_guess",
            {
                "scf_convergence": "Tight",
                "no_eigen_test": True,
                "calculate_all_force_constants": True,
                "max_step": 5,
                "max_cycles": 100,
            },
        ),
    ],
)
def test_gaussian_optimization_outputs_use_effective_method_state(
    monkeypatch, tmp_path, action_id, input_name, settings
):
    stdout = """ SCF Done:  E(UB3LYP) =  -100.123456789     A.U.
 Optimization completed.
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.100000    0.200000    0.300000
      2          8           0        1.300000    0.400000    0.500000
 ---------------------------------------------------------------------
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["g16"],
        },
    )
    monkeypatch.setattr(
        quantum_legacy, "relative_workspace_path", lambda path: path.name
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])

    response = quantum_legacy.gaussian(
        action_id,
        {
            "inputs": {input_name: CASE1_STRUCTURE},
            "method_spec": {
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "charge": -1,
                "multiplicity": 2,
            },
            "action_settings": settings,
            "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
        },
    )

    assert response["status"] == "success"
    assert response["result"]["structure"]["charge"] == -1
    assert response["result"]["structure"]["multiplicity"] == 2
    assert "\n-1 2\n" in (tmp_path / "job.gjf").read_text(encoding="utf-8")


def test_gaussian_case1_routes_are_bounded_and_exactly_express_protocol():
    method = {
        "method": "B3LYP",
        "basis": "6-31G(d)",
        "solvation_model": "iefpcm",
        "solvent": "Acetone",
        "integration_grid": "UltraFine",
    }
    limits = {"cpu_cores": 8, "memory_mb": 16000}

    ts_deck = quantum_legacy._render_gaussian(
        "locate_transition_state",
        CASE1_STRUCTURE,
        method,
        {
            "scf_convergence": "Tight",
            "no_eigen_test": True,
            "calculate_all_force_constants": True,
            "max_step": 5,
            "max_cycles": 100,
        },
        limits,
    )
    assert "Opt=(TS,NoEigenTest,CalcAll,MaxStep=5,MaxCycles=100)" in ts_deck
    assert "Int=UltraFine" in ts_deck
    assert "SCRF=(IEFPCM,Solvent=Acetone)" in ts_deck
    assert "\n1 1\n" in ts_deck
    assert " Freq " not in ts_deck

    calcfc_ts_deck = quantum_legacy._render_gaussian(
        "locate_transition_state",
        CASE1_STRUCTURE,
        method,
        {
            "scf_convergence": "Tight",
            "no_eigen_test": True,
            "calculate_all_force_constants": False,
            "initial_force_constants": "calcfc",
            "max_step": 5,
            "max_cycles": 100,
        },
        limits,
    )
    assert "Opt=(TS,NoEigenTest,CalcFC,MaxStep=5,MaxCycles=100)" in calcfc_ts_deck
    assert "CalcAll" not in calcfc_ts_deck

    hessian_deck = quantum_legacy._render_gaussian(
        "calculate_hessian",
        CASE1_STRUCTURE,
        method,
        {"scf_convergence": "Tight", "temperature_kelvin": 303.15},
        limits,
    )
    assert " Freq " in hessian_deck
    assert "Temperature=303.15" in hessian_deck
    assert "Int=UltraFine" in hessian_deck
    assert "SCRF=(IEFPCM,Solvent=Acetone)" in hessian_deck

    irc_deck = quantum_legacy._render_gaussian(
        "trace_intrinsic_reaction_coordinate",
        CASE1_STRUCTURE,
        method,
        {
            "scf_convergence": "Tight",
            "direction": "backward",
            "initial_force_constants": "CalcFC",
            "integrator": "LQA",
            "max_points": 50,
            "recalc": 5,
        },
        limits,
    )
    assert "IRC=(CalcFC,LQA,MaxPoints=50,Recalc=5,Reverse)" in irc_deck
    assert "Int=UltraFine" in irc_deck
    assert "SCRF=(IEFPCM,Solvent=Acetone)" in irc_deck

    with pytest.raises(ValueError, match="integration_grid"):
        quantum_legacy._render_gaussian(
            "calculate_hessian",
            CASE1_STRUCTURE,
            {**method, "integration_grid": "UltraFine route=arbitrary"},
            {"scf_convergence": "Tight", "temperature_kelvin": 303.15},
            limits,
        )


def test_gaussian_case1_published_routes_can_omit_scf_and_ts_max_cycles():
    method = {
        "method": "B3LYP",
        "basis": "6-31G(d)",
        "solvation_model": "iefpcm",
        "solvent": "Acetone",
    }
    limits = {"cpu_cores": 8, "memory_mb": 16000}

    ts_deck = quantum_legacy._render_gaussian(
        "locate_transition_state",
        CASE1_STRUCTURE,
        method,
        {
            "scf_convergence": "default",
            "no_eigen_test": True,
            "calculate_all_force_constants": True,
            "max_step": 5,
        },
        limits,
    )
    assert "Opt=(TS,NoEigenTest,CalcAll,MaxStep=5)" in ts_deck
    assert "MaxCycles=" not in ts_deck
    assert "SCF=" not in ts_deck

    hessian_deck = quantum_legacy._render_gaussian(
        "calculate_hessian",
        CASE1_STRUCTURE,
        method,
        {"scf_convergence": "default", "temperature_kelvin": 303.15},
        limits,
    )
    assert " Freq " in hessian_deck
    assert "Temperature=303.15" in hessian_deck
    assert "SCF=" not in hessian_deck

    irc_deck = quantum_legacy._render_gaussian(
        "trace_intrinsic_reaction_coordinate",
        CASE1_STRUCTURE,
        method,
        {
            "scf_convergence": "default",
            "direction": "forward",
            "initial_force_constants": "calcfc",
            "integrator": "lqa",
            "max_points": 50,
            "recalc": 5,
        },
        limits,
    )
    assert "IRC=(CalcFC,LQA,MaxPoints=50,Recalc=5,Forward)" in irc_deck
    assert "SCF=" not in irc_deck


def test_actual_case1_plus_one_candidate_renders_typed_hessian_and_irc():
    repository_root = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (repository_root / "supplementary" / "case1.json").read_text(encoding="utf-8")
    )
    candidate = payload["opt"]["TS-2a"]
    structure = {
        "atoms": [
            {
                "element": quantum_legacy._PERIODIC_SYMBOLS[atomic_number],
                "position_angstrom": coordinates,
            }
            for atomic_number, coordinates in zip(
                candidate["atomnos"], candidate["atomcoords_final"]
            )
        ],
        "charge": candidate["charge"],
        "multiplicity": candidate["mult"],
        "pbc": [False, False, False],
        "case1_candidate_id": "TS-2a",
    }
    assert len(structure["atoms"]) == 53
    assert structure["charge"] == 1

    method = {
        "method": "B3LYP",
        "basis": "6-31G(d)",
        "solvation_model": "iefpcm",
        "solvent": "Acetone",
        "integration_grid": "UltraFine",
    }
    hessian = quantum_legacy._render_gaussian(
        "calculate_hessian",
        structure,
        method,
        {"scf_convergence": "Tight", "temperature_kelvin": 303.15},
        {"cpu_cores": 8, "memory_mb": 16000},
    )
    irc = quantum_legacy._render_gaussian(
        "trace_intrinsic_reaction_coordinate",
        structure,
        method,
        {
            "scf_convergence": "Tight",
            "direction": "forward",
            "initial_force_constants": "calcfc",
            "integrator": "lqa",
            "max_points": 50,
            "recalc": 5,
        },
        {"cpu_cores": 8, "memory_mb": 16000},
    )
    assert hessian.count("\n") >= 53 + 9
    assert "\n1 1\n" in hessian
    assert "Temperature=303.15" in hessian
    assert "\n1 1\n" in irc
    assert "IRC=(CalcFC,LQA,MaxPoints=50,Recalc=5,Forward)" in irc


def test_gaussian_irc_returns_auditable_path_and_preserves_identity(
    monkeypatch, tmp_path
):
    stdout = """ SCF Done:  E(RB3LYP) =  -100.000000000     A.U.
 Point Number  1 in FORWARD path direction.
 Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.050000    0.100000    0.150000
      2          8           0        1.250000    0.100000    0.150000
 ---------------------------------------------------------------------
 SCF Done:  E(RB3LYP) =  -100.010000000     A.U.
 CHANGE IN THE REACTION COORDINATE =    0.28832
 NET REACTION COORDINATE UP TO THIS POINT =    0.28832
 # OF POINTS ALONG THE PATH =   1
 Calculation of FORWARD path complete.
 Summary of reaction path following
 --------------------------------------------------------------------------
                        Energy   Rx Coord
   1                   -0.01000   0.28832
 --------------------------------------------------------------------------
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["g16"],
        },
    )
    monkeypatch.setattr(
        quantum_legacy, "relative_workspace_path", lambda path: path.name
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])

    response = quantum_legacy.gaussian(
        "trace_intrinsic_reaction_coordinate",
        {
            "inputs": {"transition_state": CASE1_STRUCTURE},
            "method_spec": {
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "solvation_model": "iefpcm",
                "solvent": "Acetone",
                "integration_grid": "UltraFine",
            },
            "action_settings": {
                "scf_convergence": "Tight",
                "direction": "forward",
                "initial_force_constants": "calcfc",
                "integrator": "lqa",
                "max_points": 50,
                "recalc": 5,
            },
            "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
        },
    )

    assert response["status"] == "partial_success"
    result = response["result"]
    assert result["direction"] == "forward"
    assert result["gaussian_branch_completed"] is True
    assert result["pes_minimum_detected"] is False
    assert result["converged"] is False
    assert result["termination_reason"] == "branch_completed_without_pes_minimum"
    assert result["completed_points"] == 1
    assert result["path_structure_count"] == 2
    assert result["points"][0]["energy_hartree"] == pytest.approx(-100.01)
    assert result["summary"][0]["reaction_coordinate"] == pytest.approx(0.28832)
    assert result["transition_state"]["case1_candidate_id"] == "TS-2a"
    structure = result["structures"][1]
    assert structure["charge"] == 1
    assert structure["multiplicity"] == 1
    assert structure["case1_candidate_id"] == "TS-2a"
    assert structure["atom_map_numbers"] == [3, 5]
    assert result["path_file"] == "irc_path.json"
    assert result["summary_file"] == "irc_summary.json"
    assert response["provenance"]["direction_is_atomic"] is True


def test_gaussian_irc_skips_unaccepted_minimum_attempt_and_reports_reason(
    monkeypatch, tmp_path
):
    stdout = """ SCF Done:  E(RB3LYP) =  -100.000000000     A.U.
 Point Number  1 in REVERSE path direction.
 Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.050000    0.100000    0.150000
      2          8           0        1.250000    0.100000    0.150000
 ---------------------------------------------------------------------
 SCF Done:  E(RB3LYP) =  -100.010000000     A.U.
 CHANGE IN THE REACTION COORDINATE =    0.28832
 NET REACTION COORDINATE UP TO THIS POINT =    0.28832
 # OF POINTS ALONG THE PATH =   1
 Point Number  2 in REVERSE path direction.
 PES minimum detected within a second-order step.
 Calculation of REVERSE path complete.
 Summary of reaction path following
 --------------------------------------------------------------------------
                       Energy   Rx Coord
   1                   -0.01000  -0.28832
 --------------------------------------------------------------------------
 Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        9.000000    9.000000    9.000000
      2          8           0        9.000000    9.000000   10.000000
 ---------------------------------------------------------------------
 SCF Done:  E(RB3LYP) =  -99.000000000     A.U.
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["g16"],
        },
    )
    monkeypatch.setattr(
        quantum_legacy, "relative_workspace_path", lambda path: path.name
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])

    response = quantum_legacy.gaussian(
        "trace_intrinsic_reaction_coordinate",
        {
            "inputs": {"transition_state": CASE1_STRUCTURE},
            "method_spec": {
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "charge": -1,
                "multiplicity": 2,
            },
            "action_settings": {
                "scf_convergence": "Tight",
                "direction": "backward",
                "initial_force_constants": "calcfc",
                "integrator": "lqa",
                "max_points": 50,
                "recalc": 5,
            },
            "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
        },
    )

    assert response["status"] == "success"
    result = response["result"]
    assert result["termination_reason"] == "pes_minimum_detected"
    assert result["converged"] is True
    assert result["completed_points"] == 1
    assert [point["point_number"] for point in result["points"]] == [1]
    assert result["points"][0]["energy_hartree"] == pytest.approx(-100.01)
    assert result["points"][0]["structure"]["atoms"][0]["position_angstrom"] == [
        0.05, 0.1, 0.15
    ]
    assert result["transition_state"]["charge"] == -1
    assert result["transition_state"]["multiplicity"] == 2
    assert result["points"][0]["structure"]["charge"] == -1
    assert result["points"][0]["structure"]["multiplicity"] == 2


def test_gaussian_irc_maximum_points_is_partial_not_converged(monkeypatch, tmp_path):
    stdout = """ SCF Done:  E(RB3LYP) =  -100.000000000     A.U.
 Point Number  1 in FORWARD path direction.
 Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.050000    0.100000    0.150000
      2          8           0        1.250000    0.100000    0.150000
 ---------------------------------------------------------------------
 SCF Done:  E(RB3LYP) =  -100.010000000     A.U.
 CHANGE IN THE REACTION COORDINATE =    0.28832
 NET REACTION COORDINATE UP TO THIS POINT =    0.28832
 # OF POINTS ALONG THE PATH =   1
 Maximum number of steps reached.
 Calculation of FORWARD path complete.
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["g16"],
        },
    )
    monkeypatch.setattr(
        quantum_legacy, "relative_workspace_path", lambda path: path.name
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])

    response = quantum_legacy.gaussian(
        "trace_intrinsic_reaction_coordinate",
        {
            "inputs": {"transition_state": CASE1_STRUCTURE},
            "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
            "action_settings": {
                "scf_convergence": "Tight",
                "direction": "forward",
                "initial_force_constants": "calcfc",
                "integrator": "lqa",
                "max_points": 1,
                "recalc": 5,
            },
            "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
        },
    )

    assert response["status"] == "partial_success"
    assert response["result"]["converged"] is False
    assert response["result"]["termination_reason"] == "maximum_points_reached"
    assert response["result"]["completed_points"] == 1
    assert "before detecting a PES minimum" in response["warnings"][0]


def test_gaussian_reaction_actions_are_bidirectionally_discoverable():
    validate_catalog()
    actions = action_specs()
    backends = backend_specs()
    for action_id in ("locate_transition_state", "trace_intrinsic_reaction_coordinate"):
        assert "gaussian" in actions[action_id].backend_ids
        assert action_id in backends["gaussian"].capabilities

    hessian = inspect_action(
        "calculate_hessian", backend_id="gaussian", detail_level="contract"
    )["selected_request_contract"]
    hessian_optional = set(hessian["optional_field_names"]["action_settings"])
    assert "temperature_kelvin" in hessian_optional

    ts = inspect_action(
        "locate_transition_state", backend_id="gaussian", detail_level="contract"
    )["selected_request_contract"]
    assert {item["name"] for item in ts["required_contract"]["inputs"]} == {
        "initial_guess"
    }
    assert {item["name"] for item in ts["required_contract"]["action_settings"]} == {
        "scf_convergence", "no_eigen_test", "calculate_all_force_constants",
        "max_step",
    }
    assert "max_cycles" in set(ts["optional_field_names"]["action_settings"])
    scf = next(
        item
        for item in ts["required_contract"]["action_settings"]
        if item["name"] == "scf_convergence"
    )
    assert "default" in scf["allowed_values"]

    irc = inspect_action(
        "trace_intrinsic_reaction_coordinate",
        backend_id="gaussian",
        detail_level="contract",
    )["selected_request_contract"]
    assert {item["name"] for item in irc["required_contract"]["inputs"]} == {
        "transition_state"
    }
    assert {item["name"] for item in irc["required_contract"]["action_settings"]} == {
        "scf_convergence", "direction", "initial_force_constants", "integrator",
        "max_points", "recalc",
    }


def test_gaussian_service_contract_rejects_unbounded_protocol_values():
    response = execute_action(
        "trace_intrinsic_reaction_coordinate",
        {
            "backend_id": "gaussian",
            "inputs": {"transition_state": CASE1_STRUCTURE},
            "method_spec": {
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "solvation_model": "iefpcm",
                "solvent": "Acetone",
                "integration_grid": "UltraFine route=arbitrary",
            },
            "action_settings": {
                "scf_convergence": "Tight",
                "direction": "both",
                "initial_force_constants": "readfc",
                "integrator": "euler",
                "max_points": 50,
                "recalc": 5,
            },
        },
    )

    assert response["status"] == "invalid_request"
    assert response["error"]["code"] == "invalid_explicit_choice"


def test_gaussian_service_accepts_default_scf_and_optional_ts_max_cycles(
    monkeypatch, tmp_path
):
    captured = {}

    class Reservation:
        resource_allocation = {
            "cpu_ids": [0],
            "gpu_ids": [],
            "cpu_cores": 1,
            "memory_mb": 4096,
            "gpu_count": 0,
        }

        def release(self):
            return None

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        service_module,
        "probe_all_backends",
        lambda specifications: {
            item.id: {"available": True, "status": "available"}
            for item in specifications
        },
    )
    monkeypatch.setattr(
        service_module,
        "reserve_resources",
        lambda *_args, **_kwargs: Reservation(),
    )

    def fake_invoke_worker(**kwargs):
        captured.update(kwargs["payload"])
        return {
            "status": "success",
            "result": {"structure": CASE1_STRUCTURE, "converged": True},
            "backend_version": "16 C.01",
        }

    monkeypatch.setattr(service_module, "invoke_worker", fake_invoke_worker)

    response = service_module.execute_action(
        "locate_transition_state",
        {
            "backend_id": "gaussian",
            "inputs": {"initial_guess": CASE1_STRUCTURE},
            "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
            "action_settings": {
                "scf_convergence": "default",
                "no_eigen_test": True,
                "calculate_all_force_constants": True,
                "max_step": 5,
            },
        },
    )

    assert response["status"] == "success"
    assert response["provenance"]["execution_memory_enforcement"] == {
        "mechanism": "resource_budget_validation_and_reservation",
        "rlimit_as_applied": False,
        "rss_monitor_applied": False,
        "reason": (
            "Synchronous scientific workers are not constrained with RLIMIT_AS because "
            "Gaussian, Multiwfn, and numerical runtimes may reserve substantially more "
            "virtual address space than resident memory."
        ),
    }
    settings = captured["request"]["action_settings"]
    assert settings["scf_convergence"] == "default"
    assert "max_cycles" not in settings


def test_gaussian_service_binds_same_worker_to_fifo_and_reservation(
    monkeypatch,
    tmp_path,
):
    """typed Gaussian 必须把 runtime worker 的同一身份交付给两份恢复记录。"""

    observed = {}
    ticket = object()
    identity = ProcessIdentity(pid=123, start_time_ticks=456, boot_id="boot-1")

    class Reservation:
        resource_allocation = {"cpu_ids": [0], "gpu_ids": []}

        def bind_worker(self, worker_identity):
            observed["reservation_identity"] = worker_identity
            return True

        def release(self):
            observed["released"] = True

    @contextmanager
    def fake_gaussian_slot():
        yield ticket

    def fake_reserve_resources(*_args, **kwargs):
        observed["worker_tracking"] = kwargs.get("worker_tracking")
        return Reservation()

    def fake_bind_queue(queue_ticket, worker_identity):
        observed["queue_ticket"] = queue_ticket
        observed["queue_identity"] = worker_identity
        return True

    def fake_invoke_worker(**kwargs):
        callback = kwargs.get("on_worker_started")
        assert callback is not None
        callback(identity)
        return {
            "status": "success",
            "result": {"structure": CASE1_STRUCTURE, "converged": True},
            "backend_version": "16 C.01",
        }

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        service_module,
        "probe_all_backends",
        lambda specifications: {
            item.id: {"available": True, "status": "available"}
            for item in specifications
        },
    )
    monkeypatch.setattr(service_module, "gaussian_execution_slot", fake_gaussian_slot)
    monkeypatch.setattr(service_module, "reserve_resources", fake_reserve_resources)
    monkeypatch.setattr(
        service_module,
        "bind_gaussian_queue_worker",
        fake_bind_queue,
    )
    monkeypatch.setattr(service_module, "invoke_worker", fake_invoke_worker)

    response = service_module.execute_action(
        "locate_transition_state",
        {
            "backend_id": "gaussian",
            "inputs": {"initial_guess": CASE1_STRUCTURE},
            "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
            "action_settings": {
                "scf_convergence": "default",
                "no_eigen_test": True,
                "calculate_all_force_constants": True,
                "max_step": 5,
            },
        },
    )

    assert response["status"] == "success"
    assert observed == {
        "worker_tracking": "gaussian_typed",
        "queue_ticket": ticket,
        "queue_identity": identity,
        "reservation_identity": identity,
        "released": True,
    }


@pytest.mark.parametrize(
    "action_id", ["locate_transition_state", "trace_intrinsic_reaction_coordinate"]
)
def test_reaction_dispatcher_reaches_gaussian_atomic_actions(monkeypatch, action_id):
    observed = {}

    def fake_gaussian(received_action_id, request):
        observed.update(action_id=received_action_id, request=request)
        return {"status": "success", "result": {"sentinel": True}}

    monkeypatch.setattr(quantum_legacy, "gaussian", fake_gaussian)
    request = {"inputs": {"sentinel": True}}
    response = reaction.execute(action_id, "gaussian", request)

    assert response["status"] == "success"
    assert observed == {"action_id": action_id, "request": request}
