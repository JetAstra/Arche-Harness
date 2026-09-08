from __future__ import annotations

from pathlib import Path

import pytest
from minichem_toolbox.backends import electronic, quantum_legacy, reaction
from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.discovery import inspect_action
from minichem_mcp_tools import action_jobs

STRUCTURE = {
    "atoms": [
        {"element": "C", "position_angstrom": [0.0, 0.0, 0.0], "atom_map_number": 1},
        {"element": "C", "position_angstrom": [1.4, 0.0, 0.0], "atom_map_number": 2},
        {"element": "N", "position_angstrom": [2.1, 1.1, 0.0], "atom_map_number": 3},
        {"element": "H", "position_angstrom": [2.7, 1.2, 0.9], "atom_map_number": 4},
    ],
    "bonds": [
        {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0},
        {"atom_index_a": 1, "atom_index_b": 2, "bond_order": 1.0},
        {"atom_index_a": 2, "atom_index_b": 3, "bond_order": 1.0},
    ],
    "charge": 0,
    "multiplicity": 1,
    "pbc": [False, False, False],
    "candidate_id": "generic-candidate",
}


TD_OUTPUT = """ Gaussian 16, Revision C.01,
 SCF Done:  E(RB3LYP) =  -74.9000000000     A.U.
 Excitation energies and oscillator strengths:
 Excited State   1:      Singlet-B1     2.5000 eV  495.94 nm  f=0.0100  <S**2>=0.000
       5 ->  6         0.70000
 Excited State   2:      Singlet-A1     3.0000 eV  413.28 nm  f=0.2000  <S**2>=0.000
       4A ->  6A        0.65000
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""


NBO_OUTPUT = """ Gaussian 16, Revision C.01,
 SCF Done:  E(RM062X) =  -100.5000000000     A.U.
 ******************************Gaussian NBO Version 3.1******************************
 Wiberg bond index matrix in the NAO basis:

     Atom    1       2       3       4
     ---- ------  ------  ------  ------
   1.  C  0.0000  1.2000  0.0100  0.0000
   2.  C  1.2000  0.0000  0.8000  0.0200
   3.  N  0.0100  0.8000  0.0000  0.7000
   4.  H  0.0000  0.0200  0.7000  0.0000

 Wiberg bond index, Totals by atom:
 Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.
"""


def _orientation(coordinates: list[list[float]]) -> str:
    atomic_numbers = [6, 6, 7, 1]
    rows = "\n".join(
        f" {index:6d} {number:10d}           0"
        f" {xyz[0]:15.6f} {xyz[1]:11.6f} {xyz[2]:11.6f}"
        for index, (number, xyz) in enumerate(zip(atomic_numbers, coordinates), 1)
    )
    return f""" Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
{rows}
 ---------------------------------------------------------------------
"""


SCAN_OUTPUT = (
    " Gaussian 16, Revision C.01,\n"
    " SCF Done:  E(RB3LYP) =  -100.0000000000     A.U.\n"
    + _orientation(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 1.1, 0.0], [2.7, 1.2, 0.9]]
    )
    + " Optimization completed.\n"
    " SCF Done:  E(RB3LYP) =  -99.9900000000     A.U.\n"
    + _orientation(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 1.1, 0.0], [2.3, 1.7, 0.8]]
    )
    + " Optimization completed.\n"
    " Summary of Optimized Potential Surface Scan (add -100.0 to energies):\n"
    " Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.\n"
)


def _archive_warning_scan_output(point_count: int = 13) -> str:
    """构造完整的 Gaussian 扫描输出，并只加入已知归档警告。"""

    orientation = _orientation(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 1.1, 0.0], [2.7, 1.2, 0.9]]
    )
    chunks = [" Gaussian 16, Revision C.01,\n"]
    for index in range(point_count):
        chunks.extend(
            [
                f" SCF Done:  E(RB3LYP) =  {-100.0 + index * 0.001:.10f}     A.U.\n",
                orientation,
                " Optimization completed.\n",
            ]
        )
    chunks.extend(
        [
            " Summary of Optimized Potential Surface Scan (add -100.0 to energies):\n",
            " Unable to Open any file for archive entry.\n",
            " The archive entry for this job was punched.\n",
            " Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.\n",
        ]
    )
    return "".join(chunks)


def _run_gaussian(
    monkeypatch,
    tmp_path,
    action_id,
    request,
    stdout,
    *,
    returncode=0,
    stderr="",
    timeout=False,
):
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "command": ["g16"],
            **({"timeout": True} if timeout else {}),
        },
    )
    monkeypatch.setattr(
        quantum_legacy, "relative_workspace_path", lambda path: path.name
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])
    return quantum_legacy.gaussian(action_id, request)


def test_failed_gaussian_job_removes_only_generated_core_dump(monkeypatch, tmp_path):
    core_dump = tmp_path / "core.12345"
    preserved = tmp_path / "job.gjf.keep"
    core_dump.write_bytes(b"diagnostic core")
    preserved.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 1,
            "stdout": "Error termination via Lnk1e",
            "stderr": "Segmentation fault",
            "command": ["g16"],
        },
    )
    with pytest.raises(RuntimeError, match="Removed generated core dumps"):
        quantum_legacy.gaussian(
            "calculate_energy",
            {
                "inputs": {"structure": STRUCTURE},
                "method_spec": {"method": "B3LYP", "basis": "STO-3G"},
                "action_settings": {"scf_convergence": "default"},
            },
        )
    assert not core_dump.exists()
    assert preserved.read_text(encoding="utf-8") == "preserve"
    assert (tmp_path / "job.log").read_text(encoding="utf-8") == "Error termination via Lnk1e"


def test_gaussian_route_syntax_failure_reports_actionable_input_error(
    monkeypatch,
    tmp_path,
):
    """Gaussian 拒绝 route 时应返回可供 Revision 使用的输入诊断。"""

    core_dump = tmp_path / "core.67890"
    core_dump.write_bytes(b"diagnostic core")
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 1,
            "stdout": "QPErr --- A syntax error was detected in the input line.",
            "stderr": "Segmentation fault (core dumped)",
            "command": ["g16"],
        },
    )

    with pytest.raises(ValueError, match="Gaussian 原生关键字") as captured:
        quantum_legacy.gaussian(
            "calculate_energy",
            {
                "inputs": {"structure": STRUCTURE},
                "method_spec": {
                    "method": "omegaB97X-D3",
                    "basis": "def2-TZVP",
                },
                "action_settings": {"scf_convergence": "default"},
            },
        )

    assert "omegaB97X-D3" in str(captured.value)
    assert "def2-TZVP" in str(captured.value)
    assert not core_dump.exists()


def test_gaussian_qperr_scrf_diagnostic_checks_solvent_fields_without_blame(
    monkeypatch, tmp_path
):
    """QPErr route 含 SCRF 时应提示溶剂字段，而不是武断归因 method/basis。"""

    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 1,
            "stdout": (
                "QPErr --- A syntax error was detected in the input line.\n"
                " #p B3LYP/6-31G(d) SCRF=(SMD,Solvent=DMA)\n"
                "                                      '\n"
            ),
            "stderr": "",
            "command": ["g16"],
        },
    )

    with pytest.raises(ValueError) as captured:
        quantum_legacy.gaussian(
            "calculate_energy",
            {
                "inputs": {"structure": STRUCTURE},
                "method_spec": {
                    "method": "B3LYP",
                    "basis": "6-31G(d)",
                    "solvation_model": "smd",
                    "solvent": "DMA",
                },
                "action_settings": {"scf_convergence": "default"},
            },
        )

    message = str(captured.value)
    assert "method_spec.solvation_model" in message
    assert "method_spec.solvent" in message
    assert "method_spec.method=" not in message
    assert "method_spec.basis=" not in message
    assert "不能单独证明 method/basis 有误" in message


def test_gaussian_qperr_scan_diagnostic_keeps_solvent_and_scan_fields() -> None:
    """溶剂化扫描的 QPErr 不能因 SCRF 分支吞掉扫描字段提示。"""

    message = quantum_legacy._gaussian_qperr_message(
        "scan_reaction_coordinates",
        {"solvation_model": "smd", "solvent": "DMA"},
        "#P wB97XD/def2SVP Opt=(ModRedundant) SCRF=(SMD,Solvent=DMA)",
    )

    assert "method_spec.solvent" in message
    assert "action_settings.start_value" in message
    assert "action_settings.end_value" in message


def test_gaussian_worker_overrides_shared_cache_scratch(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )

    def fake_run_external(**kwargs):
        observed.update(kwargs)
        scratch = kwargs["environment_overrides"]["GAUSS_SCRDIR"]
        assert Path(scratch).parent == tmp_path
        assert Path(scratch).is_dir()
        Path(scratch, "temporary.rwf").write_text("temporary", encoding="utf-8")
        return {
            "available": True,
            "returncode": 0,
            "stdout": (
                "Gaussian 16, Revision C.01,\n"
                " SCF Done:  E(RB3LYP) = -100.0 A.U.\n"
                " Normal termination of Gaussian 16\n"
            ),
            "stderr": "",
            "command": ["g16"],
        }

    monkeypatch.setattr(quantum_legacy, "run_external", fake_run_external)
    monkeypatch.setattr(
        quantum_legacy, "relative_workspace_path", lambda path: path.name
    )
    monkeypatch.setattr(quantum_legacy, "command_artifacts", lambda _path: [])
    response = quantum_legacy.gaussian(
        "calculate_energy",
        {
            "inputs": {"structure": STRUCTURE},
            "method_spec": {"method": "B3LYP", "basis": "STO-3G"},
            "action_settings": {"scf_convergence": "default"},
        },
    )
    assert response["status"] == "success"
    scratch = Path(observed["environment_overrides"]["GAUSS_SCRDIR"])
    assert not scratch.exists()


def test_gaussian_unavailable_message_uses_project_cache_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        quantum_legacy, "output_directory", lambda _action_id, _backend_id: tmp_path
    )
    monkeypatch.setattr(
        quantum_legacy,
        "run_external",
        lambda **_kwargs: {
            "available": False,
            "returncode": 127,
            "stdout": "",
            "stderr": "g16 not found",
            "command": ["g16"],
        },
    )
    response = quantum_legacy.gaussian(
        "calculate_energy",
        {
            "inputs": {"structure": STRUCTURE},
            "method_spec": {"method": "B3LYP", "basis": "STO-3G"},
            "action_settings": {"scf_convergence": "default"},
        },
    )
    assert response["status"] == "unavailable"
    assert ".mini_software_cache/gaussian/g16" in response["error"]["install"]


def test_gaussian_dihedral_uses_native_signed_convention_and_periodic_target():
    coordinates = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
    value = quantum_legacy._coordinate_value(
        coordinates, "dihedral", [0, 1, 2, 3]
    )
    assert value == pytest.approx(-90.0)
    unwrapped, deviation, tolerance = quantum_legacy._scan_coordinate_near_target(
        value, 270.0, "dihedral", "degree"
    )
    assert unwrapped == pytest.approx(270.0)
    assert deviation == pytest.approx(0.0)
    assert tolerance == pytest.approx(0.05)


def test_case2_style_vertical_excitation_route_and_parser_are_generic():
    deck = quantum_legacy._render_gaussian(
        "calculate_excited_states",
        STRUCTURE,
        {
            "method": "B3LYP",
            "basis": "def2SVP",
            "excited_state_method": "tddft",
            "dispersion": "gd3bj",
            "solvation_model": "smd",
            "solvent": "EthylEthanoate",
        },
        {
            "scf_convergence": "default",
            "number_of_states": 20,
            "spin_symmetry": "singlet",
        },
        {"cpu_cores": 8, "memory_mb": 16000},
    )
    assert "TD=(NStates=20,Singlets)" in deck
    assert "B3LYP/def2SVP" in deck
    assert "EmpiricalDispersion=GD3BJ" in deck
    assert "SCRF=(SMD,Solvent=EthylEthanoate)" in deck
    assert "SCF=" not in deck

    states = quantum_legacy._parse_gaussian_excited_states(TD_OUTPUT)
    assert len(states) == 2
    assert states[0]["energy_ev"] == pytest.approx(2.5)
    assert states[0]["wavelength_nm"] == pytest.approx(495.94)
    assert states[0]["oscillator_strength"] == pytest.approx(0.01)
    assert states[0]["transitions"] == [
        {"source_orbital_index": 4, "target_orbital_index": 5, "coefficient": 0.7}
    ]
    assert states[1]["transitions"][0]["source_spin_channel"] == "a"


def test_tda_is_rendered_as_a_gaussian_route_keyword():
    deck = quantum_legacy._render_gaussian(
        "calculate_excited_states",
        STRUCTURE,
        {
            "method": "B3LYP",
            "basis": "6-31G(d)",
            "excited_state_method": "tda",
        },
        {
            "scf_convergence": "default",
            "number_of_states": 2,
            "spin_symmetry": "triplet",
        },
    )
    assert "TDA=(NStates=2,Triplets)" in deck
    assert "TD=(" not in deck


def test_vertical_excitation_execution_requires_every_requested_state(
    monkeypatch, tmp_path
):
    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {
            "method": "B3LYP",
            "basis": "def2SVP",
            "excited_state_method": "tddft",
        },
        "action_settings": {
            "scf_convergence": "default",
            "number_of_states": 2,
            "spin_symmetry": "singlet",
        },
        "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
    }
    response = _run_gaussian(
        monkeypatch, tmp_path, "calculate_excited_states", request, TD_OUTPUT
    )
    assert response["status"] == "success"
    assert response["result"]["completed_states"] == 2
    assert response["result"]["state_indices_complete"] is True
    assert response["result"]["spin_manifold_complete"] is True

    request["action_settings"]["number_of_states"] = 3
    response = _run_gaussian(
        monkeypatch, tmp_path, "calculate_excited_states", request, TD_OUTPUT
    )
    assert response["status"] == "partial_success"
    assert response["result"]["completed_states"] == 2
    assert response["result"]["requested_states"] == 3

    request["action_settings"]["number_of_states"] = 2
    wrong_manifold = TD_OUTPUT.replace("Singlet-A1", "Triplet-A1")
    response = _run_gaussian(
        monkeypatch, tmp_path, "calculate_excited_states", request, wrong_manifold
    )
    assert response["status"] == "partial_success"
    assert response["result"]["spin_manifold_complete"] is False


def test_gaussian_vertical_excitation_rejects_open_shell_reference():
    with pytest.raises(ValueError, match="closed-shell singlet"):
        quantum_legacy._render_gaussian(
            "calculate_excited_states",
            {**STRUCTURE, "multiplicity": 2},
            {
                "method": "UB3LYP",
                "basis": "def2SVP",
                "excited_state_method": "tddft",
            },
            {
                "scf_convergence": "default",
                "number_of_states": 5,
                "spin_symmetry": "triplet",
            },
        )

    with pytest.raises(ValueError, match="no U/RO method prefix"):
        quantum_legacy._render_gaussian(
            "calculate_excited_states",
            STRUCTURE,
            {
                "method": "UB3LYP",
                "basis": "def2SVP",
                "excited_state_method": "tddft",
            },
            {
                "scf_convergence": "default",
                "number_of_states": 5,
                "spin_symmetry": "singlet",
            },
        )


def test_case3_style_relaxed_dihedral_scan_route_and_parser_are_generic():
    settings = {
        "scf_convergence": "default",
        "coordinate_type": "dihedral",
        "atom_indices": [0, 1, 2, 3],
        "start_value": -60.0,
        "end_value": 60.0,
        "value_unit": "degree",
        "steps": 12,
        "optimization_convergence": "default",
        "max_cycles": 200,
        "force_constant_strategy": "default",
        "disable_symmetry": True,
    }
    deck = quantum_legacy._render_gaussian(
        "scan_reaction_coordinates",
        STRUCTURE,
        {
            "method": "wB97XD",
            "basis": "def2SVP",
            "solvation_model": "smd",
            "solvent": "N,N-Dimethylacetamide",
        },
        settings,
    )
    assert "Opt=(ModRedundant,MaxCycles=200)" in deck
    assert "SCRF=(SMD,Solvent=N,N-Dimethylacetamide)" in deck
    assert "NoSymm" in deck
    assert "\nD 1 2 3 4 -60.0\n" in deck
    assert "\nD 1 2 3 4 S 12 10.0\n" in deck

    scan = quantum_legacy._gaussian_scan_definition(STRUCTURE, settings)
    points = quantum_legacy._parse_gaussian_relaxed_scan(
        SCAN_OUTPUT, ["C", "C", "N", "H"], STRUCTURE, scan
    )
    assert len(points) == 2
    assert points[0]["structure"]["candidate_id"] == "generic-candidate"
    assert points[0]["structure"]["charge"] == 0
    assert points[1]["relative_energy_kcal_mol"] == pytest.approx(
        0.01 * 627.5094740631
    )


def test_gaussian_solvent_accepts_chemical_locants_but_rejects_scrf_options():
    assert (
        quantum_legacy._gaussian_solvent_keyword("N,N-Dimethylacetamide")
        == "N,N-Dimethylacetamide"
    )
    assert (
        quantum_legacy._gaussian_solvent_keyword("1,2-Dichloroethane")
        == "1,2-Dichloroethane"
    )
    with pytest.raises(ValueError, match="plain alias"):
        quantum_legacy._gaussian_solvent_keyword("Acetone,Read")
    with pytest.raises(ValueError, match="EthylEthanoate"):
        quantum_legacy._gaussian_solvent_keyword("EthylAcetate")


def test_relaxed_scan_execution_has_completion_gate(monkeypatch, tmp_path):
    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
        "action_settings": {
            "scf_convergence": "default",
            "coordinate_type": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "start_value": 116.69270840507131,
            "end_value": 79.14577369304537,
            "value_unit": "degree",
            "steps": 1,
            "optimization_convergence": "default",
            "max_cycles": 100,
            "force_constant_strategy": "default",
            "disable_symmetry": True,
        },
        "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
    }
    response = _run_gaussian(
        monkeypatch, tmp_path, "scan_reaction_coordinates", request, SCAN_OUTPUT
    )
    assert response["status"] == "success"
    assert response["result"]["converged"] is True
    assert response["result"]["completed_points"] == 2
    assert all(
        point["coordinate_constraint_satisfied"]
        for point in response["result"]["points"]
    )

    mismatched_request = {
        **request,
        "action_settings": {
            **request["action_settings"],
            "start_value": -60.0,
            "end_value": 60.0,
        },
    }
    response = _run_gaussian(
        monkeypatch,
        tmp_path,
        "scan_reaction_coordinates",
        mismatched_request,
        SCAN_OUTPUT,
    )
    assert response["status"] == "partial_success"
    assert response["result"]["converged"] is False

    truncated = SCAN_OUTPUT.replace(" Optimization completed.\n", "", 1)
    response = _run_gaussian(
        monkeypatch, tmp_path, "scan_reaction_coordinates", request, truncated
    )
    assert response["status"] == "partial_success"
    assert response["result"]["converged"] is False

    incomplete_scan = SCAN_OUTPUT.replace(" Optimization completed.\n", "", 1).replace(
        " Summary of Optimized Potential Surface Scan (add -100.0 to energies):\n",
        "",
    )
    response = _run_gaussian(
        monkeypatch, tmp_path, "scan_reaction_coordinates", request, incomplete_scan
    )
    assert response["status"] == "partial_success"
    assert response["result"]["completed_points"] == 1
    assert response["result"]["converged"] is False


def test_relaxed_scan_accepts_negligible_forces_completion_marker(
    monkeypatch, tmp_path
):
    """Gaussian 的合法弱力收敛文本也必须计作完成的扫描点。"""

    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
        "action_settings": {
            "scf_convergence": "default",
            "coordinate_type": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "start_value": 116.69270840507131,
            "end_value": 79.14577369304537,
            "value_unit": "degree",
            "steps": 1,
            "optimization_convergence": "default",
            "max_cycles": 100,
            "force_constant_strategy": "default",
            "disable_symmetry": True,
        },
        "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
    }
    output = SCAN_OUTPUT.replace(
        " Optimization completed.\n",
        " Optimization completed on the basis of negligible forces.\n",
        1,
    )

    response = _run_gaussian(
        monkeypatch, tmp_path, "scan_reaction_coordinates", request, output
    )

    assert response["status"] == "success"
    assert response["result"]["completed_points"] == 2
    assert response["result"]["converged"] is True


def test_gaussian_archive_warning_is_scientific_success_only_for_complete_scan(
    monkeypatch, tmp_path
):
    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
        "action_settings": {
            "scf_convergence": "default",
            "coordinate_type": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "start_value": 116.66870840507131,
            "end_value": 116.71670840507131,
            "value_unit": "degree",
            "steps": 12,
            "optimization_convergence": "default",
            "max_cycles": 100,
            "force_constant_strategy": "default",
            "disable_symmetry": True,
        },
        "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
    }
    response = _run_gaussian(
        monkeypatch,
        tmp_path,
        "scan_reaction_coordinates",
        request,
        _archive_warning_scan_output(),
        returncode=1,
    )

    assert response["status"] == "success"
    assert response["result"]["completed_points"] == 13
    assert response["result"]["requested_points"] == 13
    assert response["result"]["converged"] is True
    assert response["provenance"]["gaussian_external_returncode"] == 1
    assert response["provenance"]["gaussian_termination"] == "normal"
    assert response["provenance"]["gaussian_archive_warning"] == {
        "entry_marker": "Unable to Open any file for archive entry.",
        "punched_marker": "The archive entry for this job was punched.",
    }
    assert any("archive entry" in warning for warning in response["warnings"])


def test_gaussian_archive_warning_keeps_exit_provenance_for_energy(monkeypatch, tmp_path):
    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
        "action_settings": {"scf_convergence": "default"},
    }
    stdout = TD_OUTPUT.replace(
        " Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.\n",
        " Unable to Open any file for archive entry.\n"
        " The archive entry for this job was punched.\n"
        " Normal termination of Gaussian 16 at Thu Aug 13 00:00:00 2026.\n",
    )
    response = _run_gaussian(
        monkeypatch,
        tmp_path,
        "calculate_energy",
        request,
        stdout,
        returncode=1,
    )

    assert response["status"] == "success"
    assert response["result"]["energy"] == pytest.approx(-74.9)
    assert response["provenance"]["gaussian_external_returncode"] == 1
    assert response["provenance"]["gaussian_archive_warning"]["punched_marker"] == (
        "The archive entry for this job was punched."
    )


@pytest.mark.parametrize(
    ("stdout_transform", "returncode", "stderr", "match"),
    [
        (
            lambda output: output.replace(
                " Summary of Optimized Potential Surface Scan (add -100.0 to energies):\n",
                "",
            ),
            1,
            "",
            "complete scan",
        ),
        (
            lambda output: output.replace(
                " Optimization completed.\n", "", 1
            ),
            1,
            "",
            "complete scan",
        ),
        (
            lambda output: output.replace(
                " The archive entry for this job was punched.\n",
                " Fatal Error: archive writer failed.\n"
                " The archive entry for this job was punched.\n",
            ),
            1,
            "",
            "did not terminate normally",
        ),
        (lambda output: output, 2, "", "returncode=2"),
        (lambda output: output, -9, "", "returncode=-9"),
        (lambda output: output, 1, "archive stderr", "did not terminate normally"),
    ],
)
def test_gaussian_archive_warning_rejects_unsafe_completion(
    monkeypatch,
    tmp_path,
    stdout_transform,
    returncode,
    stderr,
    match,
):
    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
        "action_settings": {
            "scf_convergence": "default",
            "coordinate_type": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "start_value": 116.66870840507131,
            "end_value": 116.71670840507131,
            "value_unit": "degree",
            "steps": 12,
            "optimization_convergence": "default",
            "max_cycles": 100,
            "force_constant_strategy": "default",
            "disable_symmetry": True,
        },
        "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
    }
    with pytest.raises(quantum_legacy.GaussianExecutionError, match=match):
        _run_gaussian(
            monkeypatch,
            tmp_path,
            "scan_reaction_coordinates",
            request,
            stdout_transform(_archive_warning_scan_output()),
            returncode=returncode,
            stderr=stderr,
        )


def test_action_worker_diagnostic_preserves_inner_gaussian_returncode():
    error = quantum_legacy.GaussianExecutionError(
        "Gaussian failed or did not terminate normally",
        returncode=1,
        termination="normal",
        archive_warning=True,
    )

    assert action_jobs._worker_exception_execution_diagnostic(error) == {
        "inner_returncode": 1,
        "termination": "normal",
        "archive_warning": True,
        "timeout": False,
    }


def test_gaussian_archive_warning_does_not_mask_timeout(monkeypatch, tmp_path):
    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {"method": "B3LYP", "basis": "6-31G(d)"},
        "action_settings": {
            "scf_convergence": "default",
            "coordinate_type": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "start_value": 116.66870840507131,
            "end_value": 116.71670840507131,
            "value_unit": "degree",
            "steps": 12,
            "optimization_convergence": "default",
            "max_cycles": 100,
            "force_constant_strategy": "default",
            "disable_symmetry": True,
        },
        "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
    }
    with pytest.raises(quantum_legacy.GaussianExecutionError, match="returncode=124"):
        _run_gaussian(
            monkeypatch,
            tmp_path,
            "scan_reaction_coordinates",
            request,
            _archive_warning_scan_output(),
            returncode=124,
            timeout=True,
        )


def test_case3_style_wiberg_nbo_route_matrix_and_threshold(monkeypatch, tmp_path):
    request = {
        "inputs": {"structure": STRUCTURE},
        "method_spec": {
            "method": "M062X",
            "basis": "def2TZVP",
            "bond_order_definition": "wiberg_nao",
            "solvation_model": "smd",
            "solvent": "N,N-Dimethylacetamide",
        },
        "action_settings": {"scf_convergence": "default", "minimum_bond_order": 0.5},
        "resource_limits": {"cpu_cores": 2, "memory_mb": 2048},
    }
    deck = quantum_legacy._render_gaussian(
        "calculate_bond_orders",
        STRUCTURE,
        request["method_spec"],
        request["action_settings"],
    )
    assert "SP Pop=NBORead" in deck
    assert "$NBO BNDIDX $END" in deck

    response = _run_gaussian(
        monkeypatch, tmp_path, "calculate_bond_orders", request, NBO_OUTPUT
    )
    assert response["status"] == "success"
    assert response["result"]["analysis"] == "wiberg_nao"
    assert response["result"]["population_engine_version"] == "3.1"
    assert [bond["bond_order"] for bond in response["result"]["bonds"]] == [
        1.2, 0.8, 0.7
    ]
    assert response["result"]["bond_order_matrix"][1][2] == pytest.approx(0.8)


def test_case23_gaussian_providers_are_discoverable_with_explicit_contracts():
    validate_catalog()
    actions = action_specs()
    gaussian = backend_specs()["gaussian"]
    expectations = {
        "calculate_excited_states": {"number_of_states", "spin_symmetry", "scf_convergence"},
        "calculate_bond_orders": {"minimum_bond_order", "scf_convergence"},
        "scan_reaction_coordinates": {
            "coordinate_type", "atom_indices", "start_value", "end_value", "value_unit",
            "steps", "optimization_convergence", "max_cycles", "force_constant_strategy",
            "disable_symmetry", "scf_convergence",
        },
    }
    for action_id, required_settings in expectations.items():
        assert "gaussian" in actions[action_id].backend_ids
        assert action_id in gaussian.capabilities
        contract = inspect_action(
            action_id, backend_id="gaussian", detail_level="contract"
        )["selected_request_contract"]
        assert {
            item["name"] for item in contract["required_contract"]["action_settings"]
        } == required_settings


@pytest.mark.parametrize(
    ("module", "action_id"),
    [
        (electronic, "calculate_excited_states"),
        (electronic, "calculate_bond_orders"),
        (reaction, "scan_reaction_coordinates"),
    ],
)
def test_case23_dispatchers_reach_gaussian(monkeypatch, module, action_id):
    observed = {}

    def fake_gaussian(received_action_id, request):
        observed.update(action_id=received_action_id, request=request)
        return {"status": "success", "result": {"sentinel": True}}

    if module is electronic:
        monkeypatch.setattr(electronic, "_gaussian", fake_gaussian)
    else:
        monkeypatch.setattr(quantum_legacy, "gaussian", fake_gaussian)
    request = {"inputs": {"sentinel": True}}
    response = module.execute(action_id, "gaussian", request)
    assert response["status"] == "success"
    assert observed == {"action_id": action_id, "request": request}
