from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from minichem_toolbox.backends import electronic
from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox import runtime, worker_launcher


def _required(contract, section):
    return {field["name"] for field in contract["required_contract"][section]}


def _fchk_fixture() -> str:
    return """formatted checkpoint fixture
Number of atoms                            I                2
Atomic numbers                             I   N=           2
           8           1
Current cartesian coordinates              R   N=           6
  0.00000000E+00  0.00000000E+00  0.00000000E+00  1.00000000E+00  2.00000000E+00
  3.00000000E+00
"""


def _nci_settings(*, max_grid_points: int = 2000) -> dict[str, float | int]:
    return {
        "grid_spacing_bohr": 0.5,
        "grid_extension_bohr": 1.5,
        "rdg_cutoff": 0.5,
        "signed_density_abs_cutoff_au": 0.05,
        "max_grid_points": max_grid_points,
    }


@pytest.mark.parametrize(
    ("suffix", "content", "expected_parser", "expected_atom_count"),
    [
        (
            ".fch",
            _fchk_fixture(),
            "gaussian_formatted_checkpoint_bohr",
            2,
        ),
        (
            ".fchk",
            _fchk_fixture(),
            "gaussian_formatted_checkpoint_bohr",
            2,
        ),
        (
            ".wfn",
            """fixture
GAUSSIAN 2 MOL ORBITALS 2 PRIMITIVES 2 NUCLEI
 H 1 (CENTRE 1) 0.0 0.0 0.0 CHARGE = 1.0
 H 2 (CENTRE 2) 1.0 2.0 3.0 CHARGE = 1.0
CENTRE ASSIGNMENTS 1 2
""",
            "wfn_nuclear_coordinates_bohr",
            2,
        ),
        (
            ".wfx",
            """<Number of Nuclei>
2
</Number of Nuclei>
<Atomic Numbers>
1 1
</Atomic Numbers>
<Nuclear Cartesian Coordinates>
0 0 0 1 2 3
</Nuclear Cartesian Coordinates>
""",
            "wfx_nuclear_cartesian_coordinates_bohr",
            2,
        ),
        (
            ".mwfn",
            """# fixture
Ncenter=2
$Centers
1 H 1 1.0 0.0 0.0 0.0
2 H 1 1.0 1.0 2.0 3.0
# basis
""",
            "mwfn_centers_bohr",
            2,
        ),
        (
            ".molden",
            """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
H 2 1 1.0 2.0 3.0
[GTO]
1 0
s 1 1.0
1.0 1.0
2 0
s 1 1.0
1.0 1.0
[MO]
""",
            "molden_atoms_bohr",
            2,
        ),
        (
            ".47",
            """$GENNBO NATOMS=2 NBAS=2 $END
$NBO $END
$COORD
fixture
1 1 0.0 0.0 0.0
1 1 0.529177210903 1.058354421806 1.587531632709
$END
""",
            "nbo47_coord_angstrom",
            2,
        ),
    ],
)
def test_nci_grid_preflight_supports_every_declared_wavefunction_format(
    tmp_path, suffix, content, expected_parser, expected_atom_count
):
    source = tmp_path / f"fixture{suffix}"
    source.write_text(content, encoding="utf-8")

    preflight = electronic._multiwfn_nci_grid_preflight(
        source,
        grid_spacing_bohr=0.5,
        grid_extension_bohr=1.5,
        max_grid_points=2000,
    )

    assert preflight["coordinate_parser"] == expected_parser
    assert preflight["atom_count"] == expected_atom_count
    assert preflight["atom_bounding_box_bohr"]["minimum_xyz"] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert preflight["atom_bounding_box_bohr"]["maximum_xyz"] == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert preflight["estimated_shape_upper_bound_xyz"] == [9, 11, 13]
    assert preflight["estimated_point_count_upper_bound"] == 1287


@pytest.mark.parametrize(
    "suffix", [".fch", ".fchk", ".wfn", ".wfx", ".mwfn", ".molden", ".47"]
)
def test_nci_action_rejects_unparseable_declared_format_before_allocating_run(
    monkeypatch, tmp_path, suffix
):
    source = tmp_path / f"invalid{suffix}"
    source.write_text("not a parseable wavefunction\n", encoding="utf-8")
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        electronic,
        "output_directory",
        lambda *_args: pytest.fail("NCI preflight must run before allocating a run directory"),
    )
    monkeypatch.setattr(
        electronic,
        "run_external",
        lambda **_kwargs: pytest.fail("Multiwfn must not run without a bounded grid estimate"),
    )

    with pytest.raises(ValueError, match="FCH|FCHK|WFN|WFX|MWFN|Molden|NBO"):
        electronic._multiwfn_nci_rdg(
            {
                "inputs": {"wavefunction_file": source.name},
                "action_settings": _nci_settings(),
            }
        )


def test_nci_action_rejects_estimated_grid_above_caller_limit_before_run(
    monkeypatch, tmp_path
):
    source = tmp_path / "water.fchk"
    source.write_text(_fchk_fixture(), encoding="utf-8")
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        electronic,
        "output_directory",
        lambda *_args: pytest.fail("Rejected NCI request must not allocate a run directory"),
    )
    monkeypatch.setattr(
        electronic,
        "run_external",
        lambda **_kwargs: pytest.fail("Rejected NCI request must not run Multiwfn"),
    )

    with pytest.raises(
        ValueError,
        match=r"estimated upper bound 1287 points exceeds max_grid_points=1286",
    ):
        electronic._multiwfn_nci_rdg(
            {
                "inputs": {"wavefunction_file": source.name},
                "action_settings": _nci_settings(max_grid_points=1286),
            }
        )


@pytest.mark.parametrize("invalid_limit", [True, 1.0, 0, 10_000_001])
def test_nci_action_enforces_integer_catalog_grid_limit(
    monkeypatch, tmp_path, invalid_limit
):
    source = tmp_path / "water.fchk"
    source.write_text(_fchk_fixture(), encoding="utf-8")
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        electronic,
        "run_external",
        lambda **_kwargs: pytest.fail("Invalid NCI limit must not run Multiwfn"),
    )

    with pytest.raises(ValueError, match="max_grid_points must be an integer"):
        electronic._multiwfn_nci_rdg(
            {
                "inputs": {"wavefunction_file": source.name},
                "action_settings": _nci_settings(max_grid_points=invalid_limit),
            }
        )


def test_nci_action_is_registered_with_an_explicit_atomic_contract():
    validate_catalog()
    assert "multiwfn" in action_specs()["analyze_non_covalent_interactions"].backend_ids
    assert "analyze_non_covalent_interactions" in backend_specs()["multiwfn"].capabilities

    contract = inspect_action(
        "analyze_non_covalent_interactions",
        backend_id="multiwfn",
        detail_level="contract",
    )["selected_request_contract"]
    assert _required(contract, "inputs") == {"wavefunction_file"}
    assert _required(contract, "action_settings") == {
        "grid_spacing_bohr",
        "grid_extension_bohr",
        "rdg_cutoff",
        "signed_density_abs_cutoff_au",
        "max_grid_points",
    }


@pytest.mark.parametrize("suffix", [".chk", ".cube", ".xyz"])
def test_nci_action_rejects_incompatible_files_before_running_multiwfn(
    monkeypatch, tmp_path, suffix
):
    source = tmp_path / f"input{suffix}"
    source.write_bytes(b"not a wavefunction")
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        electronic,
        "run_external",
        lambda **_kwargs: pytest.fail("Multiwfn must not run for an incompatible input"),
    )

    with pytest.raises(ValueError):
        electronic._multiwfn_nci_rdg(
            {
                "inputs": {"wavefunction_file": source.name},
                "action_settings": _nci_settings(),
            }
        )


def test_nci_action_exports_auditable_grids_scatter_and_critical_points(
    monkeypatch, tmp_path
):
    source = tmp_path / "water.fchk"
    source.write_text(_fchk_fixture(), encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(electronic, "output_directory", lambda *_args: output)

    calls = []

    def fake_run_external(**kwargs):
        calls.append(kwargs)
        if kwargs["stdin_text"].startswith("20\n1\n"):
            (output / "output.txt").write_text(
                "0 0 0 -0.020 0.10\n"
                "1 0 0 0.000 0.20\n"
                "0 1 0 0.030 0.30\n"
                "0 0 1 0.060 0.40\n",
                encoding="utf-8",
            )
            (output / "func1.cub").write_text("signed density grid\n", encoding="utf-8")
            (output / "func2.cub").write_text("rdg grid\n", encoding="utf-8")
            stdout = """
Multiwfn -- A Multifunctional Wavefunction Analyzer
Version 2026.7.15 (release date is the same as version name)
Loaded water.fchk successfully!
Grid spacing in X,Y,Z is 0.500000 0.500000 0.500000 Bohr
Number of points in X,Y,Z is 2 2 1 Total: 4
Column 4/5: sign(lambda2)rho and RDG in a.u.
The cube file of sign(lambda2)rho has been exported to func1.cub
The cube file of RDG has been exported to func2.cub
"""
        else:
            (output / "CPprop.txt").write_text(
                """ ----------------   CP     1,     Type (3,+3)   ----------------
 Position (Angstrom):    0.100000000000    0.200000000000    0.300000000000
 Density of all electrons:  0.1000000000E-01
 Reduced density gradient (RDG):  0.2000000000E-01
 Sign(lambda2)*rho: -0.1000000000E-01
""",
                encoding="utf-8",
            )
            stdout = """Multiwfn -- A Multifunctional Wavefunction Analyzer
Version 2026.7.15 (release date is the same as version name)
Loaded water.fchk successfully!
Totally find 1 new critical points
Done! The results have been outputted to CPprop.txt in current folder
"""
        return {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["Multiwfn_noGUI", "water.fchk"],
        }

    monkeypatch.setattr(electronic, "run_external", fake_run_external)
    response = electronic._multiwfn_nci_rdg(
        {
            "inputs": {"wavefunction_file": source.name},
            "action_settings": _nci_settings(),
            "resource_limits": {"cpu_cores": 2, "walltime_seconds": 60},
        }
    )

    assert response["status"] == "success"
    result = response["result"]
    assert result["analysis"] == "wavefunction_based_nci_rdg"
    assert result["grid"]["point_count"] == 4
    assert result["grid"]["resource_preflight"] == {
        "coordinate_parser": "gaussian_formatted_checkpoint_bohr",
        "atom_count": 2,
        "atom_bounding_box_bohr": {
            "minimum_xyz": [0.0, 0.0, 0.0],
            "maximum_xyz": [1.0, 2.0, 3.0],
        },
        "padded_span_bohr_xyz": [4.0, 5.0, 6.0],
        "estimated_shape_upper_bound_xyz": [9, 11, 13],
        "estimated_point_count_upper_bound": 1287,
        "max_grid_points": 2000,
    }
    assert result["scatter_summary"]["descriptive_window"] == {
        "rdg_max_au": 0.5,
        "signed_density_abs_max_au": 0.05,
        "point_count": 3,
        "negative_signed_density_point_count": 1,
        "zero_signed_density_point_count": 1,
        "positive_signed_density_point_count": 1,
    }
    assert result["critical_points"] == [
        {
            "critical_point_index": 1,
            "type": "(3,+3)",
            "position_angstrom": [0.1, 0.2, 0.3],
            "electron_density_au": 0.01,
            "rdg_au": 0.02,
            "signed_density_au": -0.01,
        }
    ]
    artifact_types = {item["semantic_type"] for item in response["artifact_files"]}
    assert {
        "NonCovalentInteractionAnalysisResult",
        "NCIRDGScatterTable",
        "SignedDensityGrid",
        "ReducedDensityGradientGrid",
        "RDGCriticalPointData",
    } <= artifact_types
    assert len(calls) == 2
    assert calls[0]["stdin_text"].splitlines() == [
        "20", "1", "-10", "1.5", "4", "0.5", "2", "3", "0", "0", "q"
    ]
    assert calls[1]["stdin_text"].splitlines() == [
        "2", "-11", "13", "3", "7", "-1", "-10", "q"
    ]
    assert calls[0]["environment_overrides"]["KMP_STACKSIZE"] == "4G"
    assert calls[1]["environment_overrides"]["KMP_STACKSIZE"] == "4G"
    assert "selectivity" in result["interpretation_boundary"]
    assert json.loads((output / "result.json").read_text(encoding="utf-8")) == result


def test_nci_action_accepts_complete_zero_critical_point_search(
    monkeypatch, tmp_path
):
    source = tmp_path / "water.fchk"
    source.write_text(_fchk_fixture(), encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(electronic, "output_directory", lambda *_args: output)

    def fake_run_external(**kwargs):
        banner = "Version 2026.7.15 (release date is the same as version name)\n"
        if kwargs["stdin_text"].startswith("20\n1\n"):
            (output / "output.txt").write_text(
                "0 0 0 -0.01 0.1\n", encoding="utf-8"
            )
            (output / "func1.cub").write_text("signed\n", encoding="utf-8")
            (output / "func2.cub").write_text("rdg\n", encoding="utf-8")
            stdout = banner + """Loaded water.fchk successfully!
Grid spacing in X,Y,Z is 0.500000 0.500000 0.500000 Bohr
Number of points in X,Y,Z is 1 1 1 Total: 1
Column 4/5: sign(lambda2)rho and RDG in a.u.
The cube file of sign(lambda2)rho has been exported to func1.cub
The cube file of RDG has been exported to func2.cub
"""
        else:
            stdout = banner + """Loaded water.fchk successfully!
Totally find 0 new critical points
Done! The results have been outputted to CPprop.txt in current folder
"""
        return {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["Multiwfn_noGUI", "water.fchk"],
        }

    monkeypatch.setattr(electronic, "run_external", fake_run_external)
    response = electronic._multiwfn_nci_rdg(
        {
            "inputs": {"wavefunction_file": source.name},
            "action_settings": _nci_settings(),
        }
    )

    assert response["status"] == "success"
    assert response["result"]["critical_points"] == []
    assert not (output / "CPprop.txt").exists()


def test_nci_action_rejects_incomplete_critical_point_export(monkeypatch, tmp_path):
    source = tmp_path / "water.fchk"
    source.write_text(_fchk_fixture(), encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(electronic, "output_directory", lambda *_args: output)

    def fake_run_external(**kwargs):
        banner = "Version 2026.7.15 (release date is the same as version name)\n"
        if kwargs["stdin_text"].startswith("20\n1\n"):
            (output / "output.txt").write_text("0 0 0 -0.01 0.1\n", encoding="utf-8")
            (output / "func1.cub").write_text("signed\n", encoding="utf-8")
            (output / "func2.cub").write_text("rdg\n", encoding="utf-8")
            stdout = banner + """Loaded water.fchk successfully!
Grid spacing in X,Y,Z is 0.500000 0.500000 0.500000 Bohr
Number of points in X,Y,Z is 1 1 1 Total: 1
Column 4/5: sign(lambda2)rho and RDG in a.u.
The cube file of sign(lambda2)rho has been exported to func1.cub
The cube file of RDG has been exported to func2.cub
"""
        else:
            (output / "CPprop.txt").write_text(
                """ ----------------   CP     1,     Type (3,+3)   ----------------
 Position (Angstrom): 0 0 0
 Density of all electrons: 0.01
 Reduced density gradient (RDG): 0.02
 Sign(lambda2)*rho: -0.01
""",
                encoding="utf-8",
            )
            stdout = banner + """Loaded water.fchk successfully!
Totally find 2 new critical points
Done! The results have been outputted to CPprop.txt in current folder
"""
        return {
            "available": True,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "command": ["Multiwfn_noGUI", "water.fchk"],
        }

    monkeypatch.setattr(electronic, "run_external", fake_run_external)
    with pytest.raises(RuntimeError, match="incomplete or has non-contiguous indices"):
        electronic._multiwfn_nci_rdg(
            {
                "inputs": {"wavefunction_file": source.name},
                "action_settings": _nci_settings(),
            }
        )


def test_nci_action_rejects_unverified_multiwfn_version(monkeypatch, tmp_path):
    source = tmp_path / "water.fchk"
    source.write_text(_fchk_fixture(), encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(electronic, "output_directory", lambda *_args: output)
    monkeypatch.setattr(
        electronic,
        "run_external",
        lambda **_kwargs: {
            "available": True,
            "returncode": 0,
            "stdout": "Version 2025.12.31\nLoaded water.fchk successfully!\n",
            "stderr": "",
            "command": ["Multiwfn_noGUI", "water.fchk"],
        },
    )

    with pytest.raises(RuntimeError, match="requires version 2026.7.15"):
        electronic._multiwfn_nci_rdg(
            {
                "inputs": {"wavefunction_file": source.name},
                "action_settings": _nci_settings(),
            }
        )


def test_multiwfn_worker_lifts_stack_before_loading_scientific_modules(
    monkeypatch,
):
    calls = []
    monkeypatch.setenv("MINICHEM_BACKEND_RUNTIME", "multiwfn")
    monkeypatch.delenv("MINICHEM_WORKER_CPU_IDS", raising=False)
    monkeypatch.delenv("MINICHEM_WORKER_CPU_COUNT", raising=False)
    monkeypatch.delenv("MINICHEM_WORKER_RLIMIT_AS_BYTES", raising=False)
    monkeypatch.setattr(
        worker_launcher.resource,
        "getrlimit",
        lambda kind: (8 * 1024 * 1024, worker_launcher.resource.RLIM_INFINITY),
    )
    monkeypatch.setattr(
        worker_launcher.resource,
        "setrlimit",
        lambda kind, value: calls.append((kind, value)),
    )

    worker_launcher._apply_resource_limits()

    assert calls == [
        (
            worker_launcher.resource.RLIMIT_STACK,
            (
                worker_launcher.resource.RLIM_INFINITY,
                worker_launcher.resource.RLIM_INFINITY,
            ),
        )
    ]


def test_non_multiwfn_worker_does_not_change_stack_limit(monkeypatch):
    monkeypatch.setenv("MINICHEM_BACKEND_RUNTIME", "gaussian")
    monkeypatch.delenv("MINICHEM_WORKER_CPU_IDS", raising=False)
    monkeypatch.delenv("MINICHEM_WORKER_CPU_COUNT", raising=False)
    monkeypatch.delenv("MINICHEM_WORKER_RLIMIT_AS_BYTES", raising=False)
    monkeypatch.setattr(
        worker_launcher.resource,
        "setrlimit",
        lambda *_args: pytest.fail("Gaussian worker must not change its stack limit"),
    )

    worker_launcher._apply_resource_limits()


@pytest.mark.parametrize("runtime_name", ["gaussian", "multiwfn", "core"])
def test_worker_does_not_apply_address_space_limit(
    monkeypatch, tmp_path, runtime_name
):
    python = tmp_path / "python"
    python.touch()
    captured = {}

    class FakeProcess:
        pid = 12345
        returncode = 0

        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        def communicate(self, _input=None, timeout=None):
            captured["timeout"] = timeout
            return '{"status": "success"}\n', ""

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: python)
    monkeypatch.setattr(runtime, "runtime_environment", lambda _runtime: {})
    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)

    response = runtime.invoke_worker(
        runtime=runtime_name,
        payload={"request": {"resource_limits": {"cpu_cores": 1, "memory_mb": 1536}}},
        timeout_seconds=60,
        resource_allocation={"cpu_ids": [0], "gpu_ids": []},
    )

    assert response["status"] == "success"
    assert "MINICHEM_WORKER_RLIMIT_AS_BYTES" not in captured["env"]
    assert captured["start_new_session"] is True


def test_worker_timeout_keeps_process_tree_unknown_without_pidfd(
    monkeypatch, tmp_path
):
    python = tmp_path / "python"
    python.touch()
    signals = []

    class TimedOutProcess:
        pid = 23456
        returncode = -signal.SIGTERM

        def __init__(self):
            self.communications = 0

        def communicate(self, _input=None, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            assert timeout == runtime.WORKER_TERMINATION_GRACE_SECONDS
            return "", "worker unwound backend process group"

    process = TimedOutProcess()
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: python)
    monkeypatch.setattr(runtime, "runtime_environment", lambda _runtime: {})
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runtime.os,
        "killpg",
        lambda pid, received_signal: signals.append((pid, received_signal)),
    )
    monkeypatch.setattr(
        runtime,
        "_worker_descendant_process_groups",
        lambda _worker_pid: (34567,),
    )

    response = runtime.invoke_worker(
        runtime="gaussian",
        payload={"request": {"resource_limits": {"cpu_cores": 1}}},
        timeout_seconds=1,
    )

    assert response["status"] == "timeout"
    assert response["resource_state"] == "unknown"
    assert response["error"]["code"] == "worker_timeout"
    assert response["error"]["stderr"] == "worker unwound backend process group"
    assert signals == []


def test_worker_timeout_keeps_unresponsive_process_unknown_without_pidfd(
    monkeypatch, tmp_path
):
    python = tmp_path / "python"
    python.touch()
    signals = []

    class UnresponsiveProcess:
        pid = 23456
        returncode = -signal.SIGKILL

        def __init__(self):
            self.communications = 0

        def communicate(self, _input=None, timeout=None):
            self.communications += 1
            if self.communications <= 3:
                raise subprocess.TimeoutExpired("worker", timeout)
            raise AssertionError("不应在无 pidfd 时继续等待或发信号")

    process = UnresponsiveProcess()
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: python)
    monkeypatch.setattr(runtime, "runtime_environment", lambda _runtime: {})
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runtime.os,
        "killpg",
        lambda pid, received_signal: signals.append((pid, received_signal)),
    )
    monkeypatch.setattr(
        runtime,
        "_worker_descendant_process_groups",
        lambda _worker_pid: (34567,),
    )

    response = runtime.invoke_worker(
        runtime="gaussian",
        payload={"request": {"resource_limits": {"cpu_cores": 1}}},
        timeout_seconds=1,
    )

    assert response["status"] == "timeout"
    assert response["resource_state"] == "unknown"
    assert signals == []
    assert process.communications == 3


def test_worker_cancellation_uses_the_same_process_tree_cleanup(
    monkeypatch, tmp_path
):
    python = tmp_path / "python"
    python.touch()
    process = SimpleNamespace(pid=23456, returncode=-signal.SIGINT)
    observed = []
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: python)
    monkeypatch.setattr(runtime, "runtime_environment", lambda _runtime: {})
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runtime,
        "_terminate_worker_process",
        lambda received_process: observed.append(received_process),
    )

    def interrupted_communicate(*_args, **_kwargs):
        raise KeyboardInterrupt

    process.communicate = interrupted_communicate
    with pytest.raises(KeyboardInterrupt):
        runtime.invoke_worker(
            runtime="gaussian",
            payload={"request": {"resource_limits": {"cpu_cores": 1}}},
            timeout_seconds=60,
        )
    assert observed == [process]


def test_worker_start_oserror_is_returned_as_structured_failure(
    monkeypatch, tmp_path
):
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: python)
    monkeypatch.setattr(runtime, "runtime_environment", lambda _runtime: {})
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot exec")),
    )

    response = runtime.invoke_worker(
        runtime="gaussian",
        payload={"request": {"resource_limits": {"cpu_cores": 1}}},
        timeout_seconds=60,
    )

    assert response == {
        "status": "failed",
        "error": {"code": "worker_start_failed", "message": "cannot exec"},
        "retryable": False,
    }


def test_worker_io_oserror_cleans_started_process_before_structured_failure(
    monkeypatch, tmp_path
):
    python = tmp_path / "python"
    python.touch()
    process = SimpleNamespace(pid=23456, returncode=1)
    process.poll = lambda: process.returncode
    observed = []

    def io_failure(*_args, **_kwargs):
        raise OSError("worker pipe failed")

    process.communicate = io_failure
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: python)
    monkeypatch.setattr(runtime, "runtime_environment", lambda _runtime: {})
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runtime,
        "_terminate_worker_process",
        lambda received_process: observed.append(received_process) or ("", "cleanup complete"),
    )

    response = runtime.invoke_worker(
        runtime="gaussian",
        payload={"request": {"resource_limits": {"cpu_cores": 1}}},
        timeout_seconds=60,
    )

    assert response["status"] == "failed"
    assert response["error"] == {
        "code": "worker_exchange_failed",
        "message": "Backend worker exchange failed after startup",
        "stderr": "cleanup complete",
        "returncode": 1,
    }
    assert observed == [process]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_worker_timeout_does_not_signal_detached_backend_without_pidfd(
    monkeypatch, tmp_path
):
    child_pid_file = tmp_path / "child.pid"
    parent_pid_file = tmp_path / "worker.pid"
    worker = tmp_path / "test-worker"
    worker.write_text(
        f"""#!{sys.executable}
import os
import signal
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    start_new_session=True,
)
with open(os.environ["MINICHEM_TEST_WORKER_PID"], "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))

with open(os.environ["MINICHEM_TEST_CHILD_PID"], "w", encoding="utf-8") as handle:
    handle.write(str(child.pid))

def terminate(_signum, _frame):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate)
sys.stdin.read()
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: worker)
    monkeypatch.setattr(
        runtime,
        "runtime_environment",
        lambda _runtime: {
            "MINICHEM_TEST_CHILD_PID": str(child_pid_file),
            "MINICHEM_TEST_WORKER_PID": str(parent_pid_file),
        },
    )
    try:
        response = runtime.invoke_worker(
            runtime="gaussian",
            payload={"request": {"resource_limits": {"cpu_cores": 1}}},
            timeout_seconds=1,
        )

        assert response["status"] == "timeout"
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        os.kill(child_pid, 0)
    finally:
        if child_pid_file.exists():
            child_pid = int(child_pid_file.read_text(encoding="utf-8"))
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parent_pid_file.exists():
            parent_pid = int(parent_pid_file.read_text(encoding="utf-8"))
            try:
                os.kill(parent_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_worker_launcher_ignores_legacy_address_space_limit(monkeypatch):
    monkeypatch.setenv("MINICHEM_BACKEND_RUNTIME", "gaussian")
    monkeypatch.setenv("MINICHEM_WORKER_RLIMIT_AS_BYTES", str(1024 * 1024))
    monkeypatch.delenv("MINICHEM_WORKER_CPU_IDS", raising=False)
    monkeypatch.delenv("MINICHEM_WORKER_CPU_COUNT", raising=False)
    monkeypatch.setattr(
        worker_launcher.resource,
        "setrlimit",
        lambda *_args: pytest.fail("Scientific workers must not apply RLIMIT_AS"),
    )

    worker_launcher._apply_resource_limits()
