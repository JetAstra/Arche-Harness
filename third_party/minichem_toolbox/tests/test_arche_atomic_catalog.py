from __future__ import annotations

import json
import sys
import types
from datetime import timedelta
from pathlib import Path

import pytest

from minichem_toolbox.backends import arche_ts_pipeline, interchange, structure
from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox.service import _validate_artifact_input_semantics
from minichem_mcp_tools import discovery_tools
from minichem_mcp_tools.discovery_models import ProgressiveActionRequest


EXPECTED_PAIRS = {
    "map_reaction_atoms": "rxnmapper",
    "analyze_reaction_changes": "internal_reaction_analysis",
    "apply_reaction_changes_to_structure": "internal_reaction_analysis",
    "build_reactive_complex_ensemble": "internal_reactive_complex",
    "extract_final_structure": "cclib",
    "render_spectrum_plot": "internal_plotting",
    "derive_raman_spectrum": "internal_spectroscopy",
    "derive_roa_spectrum": "internal_spectroscopy",
    "derive_vcd_spectrum": "internal_spectroscopy",
    "derive_ecd_spectrum": "internal_spectroscopy",
    "derive_nmr_spectrum": "internal_spectroscopy",
}


_MINIMAL_GAUSSIAN_LOG = """ Gaussian, Inc.
 Entering Gaussian System, Link 0=/usr/local/g16/g16
 #P HF/3-21G SP

 Charge =  0 Multiplicity = 1
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000    0.000000
      2          1           0        0.000000    0.000000    0.960000
      3          1           0        0.920000    0.000000   -0.240000
 ---------------------------------------------------------------------
 SCF Done:  E(RHF) =  -74.96500234     A.U. after   10 cycles
 Normal termination of Gaussian 16
"""


def _required(contract, section):
    return {
        field["name"]
        for field in contract["required_contract"][section]
    }


def test_catalog_registers_each_atomic_extension_bidirectionally():
    validate_catalog()
    actions = action_specs()
    backends = backend_specs()
    for action_id, backend_id in EXPECTED_PAIRS.items():
        assert backend_id in actions[action_id].backend_ids
        assert action_id in backends[backend_id].capabilities

    assert "fairchem_uma" in actions["optimize_geometry"].backend_ids
    assert "ash_neb_uma" in actions["search_reaction_path"].backend_ids
    assert "derive_uv_vis_spectrum" in actions


def test_transition_state_path_catalog_exposes_only_public_scientific_inputs():
    action = action_specs()["search_transition_state_path"]
    backend = backend_specs()["transition_state_path_search"]
    assert action.required_inputs == ("reactants", "products")
    assert action.optional_inputs == ("main_force_field", "target_force_field")
    assert backend.required_input_fields.get(action.id, ()) == (
        "reactants", "products"
    )
    assert backend.required_method_fields.get(action.id, ()) == ()
    assert backend.required_setting_fields.get(action.id, ()) == ()
    assert backend.allowed_method_values.get(action.id, {}) == {}
    assert backend.allowed_setting_values.get(action.id, {}) == {}
    parameters = backend.parameter_specs.get(action.id, {})
    assert "inputs.reactants" in parameters
    assert "inputs.products" in parameters
    assert "inputs.main_force_field" in parameters
    assert "inputs.target_force_field" in parameters
    assert "frequency" in action.description.casefold()
    assert "irc" in action.description.casefold()


def test_transition_state_path_catalog_exposes_implementation_boundary_without_task_leakage():
    """公共说明必须与固定实现边界一致，且不携带任务答案或案例名称。"""

    action = action_specs()["search_transition_state_path"]
    backend = backend_specs()["transition_state_path_search"]
    descriptions = " ".join(
        (
            action.description,
            action.input_description,
            backend.description,
        )
    ).casefold()
    for phrase in (
        "at least two",
        "exactly one balanced hydrogen transfer",
        "still permits other mapped bond formations",
        "do not supply three-dimensional endpoints or a ts guess",
        "forward only",
        "total charge +1",
        "spin multiplicity 1",
        "gpu",
        "unvalidated",
    ):
        assert phrase in descriptions

    settings = arche_ts_pipeline._server_settings()
    assert settings["direction"] == "forward"
    assert settings["charge"] == 1
    assert settings["multiplicity"] == 1
    assert backend.resource_constraints["minimum_gpu_count"] == 1
    assert "case2" not in descriptions
    assert "case3" not in descriptions
    assert "supplementary" not in descriptions


def test_inspect_action_freezes_new_provider_contracts():
    mapping = inspect_action(
        "map_reaction_atoms", backend_id="rxnmapper", detail_level="contract"
    )["selected_request_contract"]
    assert _required(mapping, "inputs") == {"reaction"}
    assert _required(mapping, "action_settings") == {"minimum_confidence"}

    uma = inspect_action(
        "optimize_geometry", backend_id="fairchem_uma", detail_level="contract"
    )["selected_request_contract"]
    assert _required(uma, "method_spec") == {
        "model", "task_name", "device", "allow_model_download"
    }
    assert _required(uma, "action_settings") == {
        "fmax_ev_per_angstrom", "optimizer", "max_steps"
    }

    neb = inspect_action(
        "search_reaction_path", backend_id="ash_neb_uma", detail_level="contract"
    )["selected_request_contract"]
    assert _required(neb, "inputs") == {"reactant", "product"}
    assert "total_images" in _required(neb, "action_settings")
    assert "maximum_force_ev_per_angstrom" in _required(neb, "action_settings")


def test_extract_final_structure_discovery_declares_reference_identity_input():
    """发现合同必须公开可选参考结构及其严格的身份语义。"""

    contract = inspect_action(
        "extract_final_structure", backend_id="cclib", detail_level="full"
    )["selected_request_contract"]
    reference = next(
        item
        for item in contract["sections"]["inputs"]["optional"]
        if item["name"] == "reference_structure"
    )
    assert reference["type"] == (
        "AtomicStructure | AtomicStructureFile ArtifactRef | workspace-relative structure path"
    )
    assert reference["accepted_artifact_semantic_types"] == [
        "AtomicStructure",
        "AtomicStructureFile",
    ]
    assert "full AtomicStructure mapping" in reference["accepted_forms"]


def test_discovery_describes_transition_state_and_nci_file_boundaries():
    irc = inspect_action(
        "trace_intrinsic_reaction_coordinate",
        backend_id="gaussian",
        detail_level="contract",
    )["selected_request_contract"]
    transition_state = next(
        item
        for item in irc["required_contract"]["inputs"]
        if item["name"] == "transition_state"
    )
    assert transition_state["type"] == (
        "AtomicStructure | ArtifactRef | workspace-relative structure path"
    )
    assert "full AtomicStructure mapping" in transition_state["accepted_forms"]

    nci = inspect_action(
        "analyze_non_covalent_interactions",
        backend_id="multiwfn",
        detail_level="contract",
    )["selected_request_contract"]
    wavefunction = next(
        item
        for item in nci["required_contract"]["inputs"]
        if item["name"] == "wavefunction_file"
    )
    assert ".fchk" in wavefunction["accepted_extensions"]
    assert ".cube" in wavefunction["rejected_extensions"]


def test_new_artifact_handoffs_reject_wrong_semantic_type():
    invalid = _validate_artifact_input_semantics(
        "analyze_reaction_changes",
        "internal_reaction_analysis",
        {
            "mapped_reaction": {
                "artifact_id": "art_00000000000000000000000000000000",
                "semantic_type": "AtomicStructure",
            }
        },
    )
    assert invalid is not None
    assert invalid["error"]["code"] == "artifact_semantic_mismatch"

    valid = _validate_artifact_input_semantics(
        "render_spectrum_plot",
        "internal_plotting",
        {
            "spectrum": {
                "artifact_id": "art_00000000000000000000000000000000",
                "semantic_type": "SpectrumResult",
            }
        },
    )
    assert valid is None


def test_extract_final_structure_uses_last_frame_and_marks_abnormal_output_partial(
    monkeypatch, tmp_path
):
    class FakePeriodicTable:
        element = {1: "H", 8: "O"}

    cclib_module = types.ModuleType("cclib")
    cclib_module.__version__ = "test"
    parser_module = types.ModuleType("cclib.parser")
    utils_module = types.ModuleType("cclib.parser.utils")
    utils_module.PeriodicTable = FakePeriodicTable
    monkeypatch.setitem(sys.modules, "cclib", cclib_module)
    monkeypatch.setitem(sys.modules, "cclib.parser", parser_module)
    monkeypatch.setitem(sys.modules, "cclib.parser.utils", utils_module)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    source = tmp_path / "job.log"
    source.write_text("fake output\n", encoding="utf-8")

    class Parsed:
        def getattributes(self, *, tolists):
            assert tolists is True
            return {
                "atomcoords": [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
                ],
                "atomnos": [1, 8],
                "charge": -1,
                "mult": 2,
                "metadata": {"success": False, "package": "FakeQC"},
            }

    monkeypatch.setattr(
        interchange, "_ccread_with_orca_compatibility", lambda _path: (Parsed(), [])
    )
    response = interchange.execute(
        "extract_final_structure",
        "cclib",
        {
            "inputs": {"output_file": "job.log"},
            "method_spec": {},
            "action_settings": {"require_normal_termination": True},
        },
    )

    assert response["status"] == "partial_success"
    result = response["result"]
    assert result["coordinate_frame_index"] == 1
    assert result["structure"]["atoms"][0]["position_angstrom"] == [0.1, 0.0, 0.0]
    assert result["structure"]["charge"] == -1
    assert result["structure"]["multiplicity"] == 2
    assert result["normal_termination"] is False


def test_extract_final_structure_uses_installed_cclib_and_copies_reference_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """真实 cclib 1.8.1 解析最后一帧，并按参考结构顺序复制原子身份。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    source = tmp_path / "water.log"
    source.write_text(_MINIMAL_GAUSSIAN_LOG, encoding="utf-8")
    reference = {
        "atoms": [
            {
                "element": "O",
                "position_angstrom": [0.0, 0.0, 0.0],
                "atom_map_number": 101,
            },
            {
                "element": "H",
                "position_angstrom": [0.0, 0.0, 0.96],
                "atom_map_number": 102,
            },
            {
                "element": "H",
                "position_angstrom": [0.92, 0.0, -0.24],
                "atom_map_number": 103,
            },
        ],
        "atom_map_numbers": [101, 102, 103],
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }

    response = interchange.execute(
        "extract_final_structure",
        "cclib",
        {
            "inputs": {"output_file": "water.log", "reference_structure": reference},
            "action_settings": {"require_normal_termination": True},
        },
    )

    assert response["status"] == "success"
    result = response["result"]
    assert result["structure"]["atom_map_numbers"] == [101, 102, 103]
    assert [atom["atom_map_number"] for atom in result["structure"]["atoms"]] == [
        101,
        102,
        103,
    ]
    assert result["identity_source"] == "reference_structure"
    assert response["provenance"]["identity_source"] == "reference_structure"


@pytest.mark.parametrize(
    "atom_map_numbers",
    (
        [0, 2, 3],
        [-1, 2, 3],
        [1, 1, 3],
    ),
)
def test_extract_final_structure_rejects_invalid_reference_atom_map_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    atom_map_numbers: list[int],
) -> None:
    """参考结构的 atom map 必须全部为正整数且互不重复。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    (tmp_path / "water.log").write_text(_MINIMAL_GAUSSIAN_LOG, encoding="utf-8")
    reference = {
        "atoms": [
            {"element": "O", "position_angstrom": [0.0, 0.0, 0.0]},
            {"element": "H", "position_angstrom": [0.0, 0.0, 0.96]},
            {"element": "H", "position_angstrom": [0.92, 0.0, -0.24]},
        ],
        "atom_map_numbers": atom_map_numbers,
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }

    with pytest.raises(ValueError, match="positive and unique"):
        interchange.execute(
            "extract_final_structure",
            "cclib",
            {
                "inputs": {
                    "output_file": "water.log",
                    "reference_structure": reference,
                },
                "action_settings": {"require_normal_termination": True},
            },
        )


def test_public_mcp_extract_final_structure_preserves_reference_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """公开 MCP execute_action 路径必须交付带原子 map 身份的最终结构。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    (tmp_path / "water.log").write_text(_MINIMAL_GAUSSIAN_LOG, encoding="utf-8")
    reference = {
        "atoms": [
            {"element": "O", "position_angstrom": [0.0, 0.0, 0.0]},
            {"element": "H", "position_angstrom": [0.0, 0.0, 0.96]},
            {"element": "H", "position_angstrom": [0.92, 0.0, -0.24]},
        ],
        "atom_map_numbers": [201, 202, 203],
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }
    response = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="extract_final_structure",
            backend_id="cclib",
            inputs={"output_file": "water.log", "reference_structure": reference},
            action_settings={"require_normal_termination": True},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        )
    )

    assert response["status"] == "success"
    assert response["result"]["identity_source"] == "reference_structure"
    structure_artifact = tmp_path / next(
        item
        for item in response["output_artifacts"]
        if item["semantic_type"] == "AtomicStructureFile"
    )["path"]
    assert structure_artifact.is_file()
    assert response["result"]["structure"]["atom_map_numbers"] == [201, 202, 203]


def test_public_mcp_extract_final_structure_reuses_its_atomic_structure_file_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """extract_final_structure 产出的 AtomicStructureFile 可回灌为参考结构。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    (tmp_path / "water.log").write_text(_MINIMAL_GAUSSIAN_LOG, encoding="utf-8")

    extracted = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="extract_final_structure",
            backend_id="cclib",
            inputs={"output_file": "water.log"},
            action_settings={"require_normal_termination": True},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        )
    )
    assert extracted["status"] == "success"
    xyz_artifact = next(
        item
        for item in extracted["output_artifacts"]
        if item["semantic_type"] == "AtomicStructureFile"
    )

    reused = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="extract_final_structure",
            backend_id="cclib",
            inputs={
                "output_file": "water.log",
                "reference_structure": xyz_artifact,
            },
            action_settings={"require_normal_termination": True},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        )
    )
    assert reused["status"] == "success"
    assert reused["result"]["identity_source"] == "reference_structure"

    wrong_type = dict(xyz_artifact)
    wrong_type["semantic_type"] = "GaussianInputDeck"
    rejected = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="extract_final_structure",
            backend_id="cclib",
            inputs={
                "output_file": "water.log",
                "reference_structure": wrong_type,
            },
            action_settings={"require_normal_termination": True},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        )
    )
    assert rejected["status"] == "invalid_request"
    assert rejected["error"]["code"] == "artifact_semantic_mismatch"
    assert "AtomicStructureFile" in rejected["error"]["message"]


def test_extract_final_structure_refuses_missing_state_and_nonfinite_coordinates(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    source = tmp_path / "job.log"
    source.write_text("fake output\n", encoding="utf-8")

    class FakePeriodicTable:
        element = {1: "H"}

    cclib_module = types.ModuleType("cclib")
    cclib_module.__version__ = "test"
    parser_module = types.ModuleType("cclib.parser")
    parser_utils = types.ModuleType("cclib.parser.utils")
    parser_utils.PeriodicTable = FakePeriodicTable
    monkeypatch.setitem(sys.modules, "cclib", cclib_module)
    monkeypatch.setitem(sys.modules, "cclib.parser", parser_module)
    monkeypatch.setitem(sys.modules, "cclib.parser.utils", parser_utils)

    class Parsed:
        def __init__(self, attributes):
            self.attributes = attributes

        def getattributes(self, *, tolists):
            assert tolists is True
            return self.attributes

    base = {
        "atomcoords": [[[0.0, 0.0, 0.0]]],
        "atomnos": [1],
        "metadata": {"success": True},
    }
    monkeypatch.setattr(
        interchange,
        "_ccread_with_orca_compatibility",
        lambda _path: (Parsed({**base, "charge": 0}), []),
    )
    with pytest.raises(ValueError, match="refuses to invent"):
        interchange._extract_final_structure(
            {
                "inputs": {"output_file": "job.log"},
                "action_settings": {"require_normal_termination": True},
            }
        )

    monkeypatch.setattr(
        interchange,
        "_ccread_with_orca_compatibility",
        lambda _path: (
            Parsed(
                {
                    **base,
                    "atomcoords": [[[float("nan"), 0.0, 0.0]]],
                    "charge": 0,
                    "mult": 1,
                }
            ),
            [],
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        interchange._extract_final_structure(
            {
                "inputs": {"output_file": "job.log"},
                "action_settings": {"require_normal_termination": True},
            }
        )


def test_cclib_property_groups_include_ecd_and_nmr_attributes():
    assert "etrotats" in interchange._CCLIB_GROUPS["excited_states"]
    assert interchange._CCLIB_GROUPS["nmr"] == (
        "atomnos",
        "nmrtensors",
        "nmrcouplingtensors",
    )
    assert interchange._CCLIB_UNITS["nmrtensors"] == "parts_per_million"
    assert interchange._CCLIB_UNITS["nmrcouplingtensors"] == "hertz"
    assert "roa" not in interchange._CCLIB_GROUPS
    assert "vcd" not in interchange._CCLIB_GROUPS


def test_parse_quantum_output_converts_cclib_durations_before_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cclib 的耗时对象须与落盘 Artifact 一致，并能由异步 worker 直接编码。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    (tmp_path / "job.log").write_text("fake output\n", encoding="utf-8")
    cclib_module = types.ModuleType("cclib")
    cclib_module.__version__ = "test"
    monkeypatch.setitem(sys.modules, "cclib", cclib_module)

    class Parsed:
        def getattributes(self, *, tolists):
            assert tolists is True
            return {
                "metadata": {
                    "package": "Gaussian",
                    "cpu_time": [timedelta(hours=4, minutes=55, seconds=4.1)],
                    "wall_time": [timedelta(minutes=5, seconds=2.3)],
                },
                "vibfreqs": [-318.165, 22.473],
            }

    monkeypatch.setattr(
        interchange,
        "_ccread_with_orca_compatibility",
        lambda _path: (Parsed(), []),
    )
    response = interchange.execute(
        "parse_quantum_chemistry_output",
        "cclib",
        {
            "inputs": {"output_file": "job.log"},
            "method_spec": {},
            "action_settings": {
                "properties": ["metadata", "vibrational_frequencies"],
                "coordinate_frames": "last",
                "include_orbital_coefficients": False,
                "include_excited_state_configurations": False,
                "max_array_elements": 1000,
            },
        },
    )

    properties = response["result"]["properties"]
    assert properties["metadata"]["cpu_time"] == ["4:55:04.100000"]
    assert properties["metadata"]["wall_time"] == ["0:05:02.300000"]
    artifact = tmp_path / response["artifact_files"][0]["path"]
    assert json.loads(artifact.read_text(encoding="utf-8"))["properties"] == properties
    json.dumps(response)


def test_orca_compatibility_patch_is_installed_parsed_and_restored_under_lock(
    monkeypatch, tmp_path
):
    events = []

    class TrackingLock:
        active = False

        def __enter__(self):
            assert not self.active
            self.active = True
            events.append("lock_enter")

        def __exit__(self, exc_type, exc, traceback):
            assert self.active
            self.active = False
            events.append("lock_exit")

    tracking_lock = TrackingLock()

    def original(_parser, _inputfile, _line):
        return None

    class TrackingORCAMeta(type):
        def __setattr__(cls, name, value):
            if name == "_append_scfvalues_scftargets":
                assert tracking_lock.active
                events.append("restore" if value is original else "install")
            super().__setattr__(name, value)

    class FakeORCA(metaclass=TrackingORCAMeta):
        _append_scfvalues_scftargets = staticmethod(original)

    def fake_ccread(path, *, loglevel):
        assert path == str(tmp_path / "job.out")
        assert loglevel == 40
        assert tracking_lock.active
        assert FakeORCA._append_scfvalues_scftargets is not original
        events.append("parse")
        return object()

    cclib_io = types.ModuleType("cclib.io")
    cclib_io.ccread = fake_ccread
    cclib_parser = types.ModuleType("cclib.parser")
    cclib_orca = types.ModuleType("cclib.parser.orcaparser")
    cclib_orca.ORCA = FakeORCA
    monkeypatch.setitem(sys.modules, "cclib.io", cclib_io)
    monkeypatch.setitem(sys.modules, "cclib.parser", cclib_parser)
    monkeypatch.setitem(sys.modules, "cclib.parser.orcaparser", cclib_orca)
    monkeypatch.setattr(interchange, "_CCLIB_ORCA_PARSE_LOCK", tracking_lock)

    parsed, warnings = interchange._ccread_with_orca_compatibility(tmp_path / "job.out")

    assert parsed is not None
    assert warnings == []
    assert events == ["lock_enter", "install", "parse", "restore", "lock_exit"]
    assert FakeORCA._append_scfvalues_scftargets is original


def test_rdkit_structure_serialization_preserves_maps_charges_and_bonds():
    from rdkit import Chem

    molecule = Chem.MolFromSmiles("[CH3:1][O-:2]")
    serialized = structure._rdkit_structure(molecule)

    assert [atom["atom_map_number"] for atom in serialized["atoms"]] == [1, 2]
    assert serialized["atoms"][1]["formal_charge"] == -1
    assert serialized["bonds"] == [
        {
            "atom_index_a": 0,
            "atom_index_b": 1,
            "bond_order": 1.0,
            "aromatic": False,
        }
    ]


def test_rdkit_structure_primary_smiles_preserves_explicit_hydrogens():
    """3D 结构的主图必须包含可被反应映射逐原子编号的显式氢。"""

    from rdkit import Chem

    molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
    serialized = structure._rdkit_structure(molecule)

    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    reparsed = Chem.MolFromSmiles(serialized["smiles"], parser)
    assert reparsed is not None
    assert serialized["hydrogens_explicit"] is True
    assert serialized["explicit_hydrogen_count"] == 4
    assert sum(atom.GetAtomicNum() == 1 for atom in reparsed.GetAtoms()) == 4
    assert Chem.MolFromSmiles(serialized["heavy_atom_smiles"]).GetNumAtoms() == 2


def test_rdkit_parser_preserves_explicit_mapped_hydrogens():
    from minichem_toolbox.backends.structure import _rdkit_molecule, _rdkit_structure

    molecule = _rdkit_molecule(
        {"smiles": "[O:1]([H:2])[H:3]"}, add_hydrogens=True
    )
    serialized = _rdkit_structure(molecule)

    assert serialized["explicit_hydrogen_count"] == 2
    assert sorted(
        atom["atom_map_number"]
        for atom in serialized["atoms"]
        if atom["element"] == "H"
    ) == [2, 3]


def test_rdkit_coordinate_molecule_accepts_atomic_structure_graph():
    structure_value = {
        "atoms": [
            {
                "element": "C",
                "position_angstrom": [0.0, 0.0, 0.0],
                "atom_map_number": 1,
            },
            {
                "element": "O",
                "position_angstrom": [1.4, 0.0, 0.0],
                "formal_charge": -1,
                "atom_map_number": 2,
            },
        ],
        "bonds": [
            {"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0}
        ],
        "charge": -1,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }

    molecule = structure._rdkit_coordinate_molecule(structure_value)

    assert molecule.GetNumAtoms() == 2
    assert molecule.GetConformer().GetAtomPosition(1).x == 1.4
    assert [atom.GetAtomMapNum() for atom in molecule.GetAtoms()] == [1, 2]
    assert molecule.GetAtomWithIdx(1).GetFormalCharge() == -1
