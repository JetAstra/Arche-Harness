from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

import pytest
from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.backends import reaction, reaction_mapping

RAW_REACTION = "CBr.[OH-]>>CO.[Br-]"
MAPPED_REACTION = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
EXPLICIT_H_MAPPED_REACTION = (
    "[C:1]([H:4])([H:5])([H:6])[Br:2].[O-:3][H:7]>>"
    "[C:1]([H:4])([H:5])([H:6])[O:3][H:7].[Br-:2]"
)


class _FakeRXNMapper:
    calls: ClassVar[list[list[str]]] = []

    def get_attention_guided_atom_maps(self, reactions: list[str]):
        self.calls.append(reactions)
        return [{"mapped_rxn": MAPPED_REACTION, "confidence": 0.9875}]


def _mapping_request(**settings):
    return {
        "inputs": {"reactants": ["CBr", "[OH-]"], "products": ["CO", "[Br-]"]},
        "method_spec": {},
        "action_settings": settings,
    }


def _analysis_request(value=MAPPED_REACTION):
    return {
        "inputs": {"mapped_reaction": value},
        "method_spec": {},
        "action_settings": {},
    }


def _reactant_structure():
    return {
        "atoms": [
            {
                "element": "C",
                "map_number": 1,
                "formal_charge": 0,
                "position_angstrom": [0.0, 0.0, 0.0],
            },
            {
                "element": "Br",
                "map_number": 2,
                "formal_charge": 0,
                "position_angstrom": [1.9, 0.0, 0.0],
            },
            {
                "element": "O",
                "map_number": 3,
                "formal_charge": -1,
                "position_angstrom": [-2.0, 0.0, 0.0],
            },
            {"element": "H", "map_number": 4, "formal_charge": 0,
             "position_angstrom": [-0.5, 0.9, 0.0]},
            {"element": "H", "map_number": 5, "formal_charge": 0,
             "position_angstrom": [-0.5, -0.45, 0.78]},
            {"element": "H", "map_number": 6, "formal_charge": 0,
             "position_angstrom": [-0.5, -0.45, -0.78]},
            {"element": "H", "map_number": 7, "formal_charge": 0,
             "position_angstrom": [-2.8, 0.0, 0.0]},
        ],
        "bonds": [
            {
                "atom_index_a": 0,
                "atom_index_b": 1,
                "bond_order": 1.0,
            },
            {"atom_index_a": 0, "atom_index_b": 3, "bond_order": 1.0},
            {"atom_index_a": 0, "atom_index_b": 4, "bond_order": 1.0},
            {"atom_index_a": 0, "atom_index_b": 5, "bond_order": 1.0},
            {"atom_index_a": 2, "atom_index_b": 6, "bond_order": 1.0},
        ],
        "charge": -1,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }


def test_rxnmapper_missing_returns_unavailable(monkeypatch):
    monkeypatch.setattr(reaction_mapping, "_load_rxnmapper_class", lambda: None)

    response = reaction_mapping.execute(
        "map_reaction_atoms", "rxnmapper", _mapping_request()
    )

    assert response["status"] == "unavailable"
    assert response["error"]["code"] == "backend_unavailable"
    assert "rxnmapper" in response["error"]["install"]


def test_map_reaction_atoms_validates_and_returns_structured_mapping(monkeypatch):
    _FakeRXNMapper.calls.clear()
    monkeypatch.setattr(
        reaction_mapping, "_load_rxnmapper_class", lambda: _FakeRXNMapper
    )

    response = reaction_mapping.execute(
        "map_reaction_atoms",
        "rxnmapper",
        _mapping_request(minimum_confidence=0.95),
    )

    assert response["status"] == "partial_success"
    result = response["result"]
    assert _FakeRXNMapper.calls == [[RAW_REACTION]]
    assert result["mapped_reaction_smiles"] == MAPPED_REACTION
    assert result["confidence"] == pytest.approx(0.9875)
    assert result["mapping_complete"] is False
    assert result["atom_conserved"] is True
    assert result["atom_count"] == 3
    assert result["explicit_atom_count"] == 3
    assert result["total_atom_count"] == 7
    assert result["unmapped_hydrogen_count"] == 4
    assert result["element_counts"] == {"Br": 1, "C": 1, "H": 4, "O": 1}
    assert [entry["map_number"] for entry in result["atom_map"]] == [1, 2, 3]


def test_map_reaction_atoms_uses_configured_model_cache(monkeypatch, tmp_path):
    class ConfigurableMapper(_FakeRXNMapper):
        config = None

        def __init__(self, config):
            type(self).config = config

    model_path = tmp_path / "rxnmapper-model"
    model_path.mkdir()
    for filename in reaction_mapping._RXNMAPPER_REQUIRED_FILES:
        (model_path / filename).write_bytes(b"test")
    monkeypatch.setenv("MINICHEM_RXNMAPPER_MODEL_PATH", str(model_path))
    monkeypatch.setattr(
        reaction_mapping, "_load_rxnmapper_class", lambda: ConfigurableMapper
    )

    response = reaction_mapping.map_reaction_atoms(_mapping_request())

    assert response["status"] == "partial_success"
    assert ConfigurableMapper.config == {"model_path": str(model_path)}


def test_mapping_rejects_element_imbalance_before_model_call(monkeypatch):
    _FakeRXNMapper.calls.clear()
    monkeypatch.setattr(
        reaction_mapping, "_load_rxnmapper_class", lambda: _FakeRXNMapper
    )
    request = {
        "inputs": {"reaction": "C>>CC"},
        "method_spec": {},
        "action_settings": {},
    }

    with pytest.raises(ValueError, match="not element-conserving"):
        reaction_mapping.map_reaction_atoms(request)

    assert _FakeRXNMapper.calls == []


def test_mapping_rejects_duplicate_map_numbers(monkeypatch):
    class DuplicateMapper:
        def get_attention_guided_atom_maps(self, reactions):
            return [
                {
                    "mapped_rxn": ("[CH3:1][Br:1].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"),
                    "confidence": 0.99,
                }
            ]

    monkeypatch.setattr(
        reaction_mapping, "_load_rxnmapper_class", lambda: DuplicateMapper
    )

    with pytest.raises(ValueError, match="duplicated on the reactant side"):
        reaction_mapping.map_reaction_atoms(_mapping_request())


def test_mapping_rejects_mapper_graph_mutation(monkeypatch):
    class MutatingMapper:
        def get_attention_guided_atom_maps(self, reactions):
            return [
                {
                    "mapped_rxn": "[CH3:1][Br:2].[OH-:3]>>[CH3:1][Br:2].[OH-:3]",
                    "confidence": 0.99,
                }
            ]

    monkeypatch.setattr(
        reaction_mapping, "_load_rxnmapper_class", lambda: MutatingMapper
    )

    with pytest.raises(ValueError, match="changed the molecular graph"):
        reaction_mapping.map_reaction_atoms(_mapping_request())


def test_analyze_reaction_changes_accepts_artifact_and_reports_graph_edits(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    reference = ArtifactStore().put_json(
        {"mapped_reaction_smiles": MAPPED_REACTION},
        semantic_type="AtomMappedReaction",
        producer_action="map_reaction_atoms",
        producer_backend="rxnmapper",
    )

    response = reaction_mapping.execute(
        "analyze_reaction_changes",
        "internal_reaction_analysis",
        _analysis_request({"artifact_id": reference.artifact_id}),
    )

    assert response["status"] == "partial_success"
    result = response["result"]
    assert result["formed_bonds"] == [
        {"atom_map_a": 1, "atom_map_b": 3, "bond_order": 1.0}
    ]
    assert result["broken_bonds"] == [
        {"atom_map_a": 1, "atom_map_b": 2, "bond_order": 1.0}
    ]
    assert result["bond_order_changes"] == []
    assert result["formal_charge_changes"] == [
        {
            "atom_map": 2,
            "element": "Br",
            "reactant_formal_charge": 0,
            "product_formal_charge": -1,
        },
        {
            "atom_map": 3,
            "element": "O",
            "reactant_formal_charge": -1,
            "product_formal_charge": 0,
        },
    ]
    assert result["forward_changes"]["bond_changes"] == [
        {
            "atom_map_a": 1,
            "atom_map_b": 2,
            "from_order": 1.0,
            "to_order": 0.0,
        },
        {
            "atom_map_a": 1,
            "atom_map_b": 3,
            "from_order": 0.0,
            "to_order": 1.0,
        },
    ]


def test_analysis_rejects_map_numbers_that_change_element_identity():
    invalid = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:2].[Br-:3]"

    with pytest.raises(ValueError, match="element identity changed"):
        reaction_mapping.analyze_reaction_changes(_analysis_request(invalid))


def test_analysis_reports_bond_order_and_balanced_hydrogen_count_changes():
    bond_shift = "[CH2:1]=[CH:2][CH3:3]>>[CH3:1][CH:2]=[CH2:3]"

    response = reaction_mapping.analyze_reaction_changes(_analysis_request(bond_shift))

    assert response["status"] == "partial_success"
    result = response["result"]
    assert result["bond_order_changes"] == [
        {
            "atom_map_a": 1,
            "atom_map_b": 2,
            "reactant_bond_order": 2.0,
            "product_bond_order": 1.0,
        },
        {
            "atom_map_a": 2,
            "atom_map_b": 3,
            "reactant_bond_order": 1.0,
            "product_bond_order": 2.0,
        },
    ]
    assert result["hydrogen_count_changes"] == [
        {
            "atom_map": 1,
            "element": "C",
            "reactant_attached_hydrogen_count": 2,
            "product_attached_hydrogen_count": 3,
            "delta": 1,
        },
        {
            "atom_map": 3,
            "element": "C",
            "reactant_attached_hydrogen_count": 3,
            "product_attached_hydrogen_count": 2,
            "delta": -1,
        },
    ]
    summary = result["hydrogen_transfer_summary"]
    assert summary["donors"] == [{"atom_map": 3, "hydrogens_lost": 1}]
    assert summary["acceptors"] == [{"atom_map": 1, "hydrogens_gained": 1}]
    assert summary["balanced"] is True
    assert summary["donor_acceptor_pairing_inferred"] is False


def test_apply_reaction_changes_is_reversible_and_preserves_coordinates():
    changes_response = reaction_mapping.analyze_reaction_changes(
        _analysis_request(EXPLICIT_H_MAPPED_REACTION)
    )
    assert changes_response["status"] == "success"
    changes = changes_response["result"]
    reactant = _reactant_structure()

    forward = reaction_mapping.execute(
        "apply_reaction_changes_to_structure",
        "internal_reaction_analysis",
        {
            "inputs": {"structure": reactant, "reaction_changes": changes},
            "method_spec": {},
            "action_settings": {
                "direction": "forward",
                "target_charge": -1,
                "target_multiplicity": 1,
            },
        },
    )

    assert forward["status"] == "success"
    product = forward["result"]["structure"]
    assert [atom["position_angstrom"] for atom in product["atoms"]] == [
        atom["position_angstrom"] for atom in reactant["atoms"]
    ]
    assert [
        (bond["atom_map_a"], bond["atom_map_b"], bond["bond_order"])
        for bond in product["bonds"]
    ] == [
        (1, 3, 1.0),
        (1, 4, 1.0),
        (1, 5, 1.0),
        (1, 6, 1.0),
        (3, 7, 1.0),
    ]
    assert [atom["formal_charge"] for atom in product["atoms"]] == [
        0, -1, 0, 0, 0, 0, 0
    ]
    assert product["charge"] == -1
    assert forward["result"]["coordinates_preserved"] is True

    reverse = reaction_mapping.apply_reaction_changes_to_structure(
        {
            "inputs": {"structure": product, "reaction_changes": changes},
            "method_spec": {},
            "action_settings": {
                "direction": "reverse",
                "target_charge": -1,
                "target_multiplicity": 1,
            },
        }
    )
    restored = reverse["result"]["structure"]
    assert [
        (bond["atom_map_a"], bond["atom_map_b"], bond["bond_order"])
        for bond in restored["bonds"]
    ] == [
        (1, 2, 1.0),
        (1, 4, 1.0),
        (1, 5, 1.0),
        (1, 6, 1.0),
        (3, 7, 1.0),
    ]
    assert [atom["formal_charge"] for atom in restored["atoms"]] == [
        0, 0, -1, 0, 0, 0, 0
    ]


def test_apply_rejects_duplicate_structure_map_numbers():
    changes = reaction_mapping.analyze_reaction_changes(
        _analysis_request(EXPLICIT_H_MAPPED_REACTION)
    )["result"]
    structure = deepcopy(_reactant_structure())
    structure["atoms"][1]["map_number"] = 1

    with pytest.raises(ValueError, match="must be unique"):
        reaction_mapping.apply_reaction_changes_to_structure(
            {
                "inputs": {"structure": structure, "reaction_changes": changes},
                "method_spec": {},
                "action_settings": {
                    "direction": "forward",
                    "target_charge": -1,
                    "target_multiplicity": 1,
                },
            }
        )


def test_apply_requires_explicit_direction():
    changes = reaction_mapping.analyze_reaction_changes(
        _analysis_request(EXPLICIT_H_MAPPED_REACTION)
    )["result"]

    with pytest.raises(ValueError, match="explicitly set"):
        reaction_mapping.apply_reaction_changes_to_structure(
            {
                "inputs": {
                    "structure": _reactant_structure(),
                    "reaction_changes": changes,
                },
                "method_spec": {},
                "action_settings": {},
            }
        )


def test_apply_requires_explicit_target_state_and_removes_stale_identifiers():
    changes = reaction_mapping.analyze_reaction_changes(
        _analysis_request(EXPLICIT_H_MAPPED_REACTION)
    )["result"]
    structure = _reactant_structure()
    structure["smiles"] = "CBr.[OH-]"
    structure["metadata"] = {"inchi": "stale", "label": "kept"}

    with pytest.raises(ValueError, match="target_charge and target_multiplicity"):
        reaction_mapping.apply_reaction_changes_to_structure(
            {
                "inputs": {"structure": structure, "reaction_changes": changes},
                "method_spec": {},
                "action_settings": {"direction": "forward"},
            }
        )

    response = reaction_mapping.apply_reaction_changes_to_structure(
        {
            "inputs": {"structure": structure, "reaction_changes": changes},
            "method_spec": {},
            "action_settings": {
                "direction": "forward",
                "target_charge": -1,
                "target_multiplicity": 1,
            },
        }
    )
    output = response["result"]["structure"]
    assert "smiles" not in output
    assert "inchi" not in output["metadata"]
    assert output["metadata"]["label"] == "kept"
    assert response["result"]["removed_stale_graph_identifiers"] == [
        "smiles", "metadata.inchi"
    ]


def test_reaction_change_set_maps_into_path_atom_indices_without_vacuous_pass():
    changes = reaction_mapping.analyze_reaction_changes(
        _analysis_request(EXPLICIT_H_MAPPED_REACTION)
    )["result"]

    records = reaction._bond_change_records(
        changes, reference_structure=_reactant_structure()
    )

    assert records == [
        {"type": "form", "atom_indices": [0, 2], "atom_maps": [1, 3]},
        {"type": "break", "atom_indices": [0, 1], "atom_maps": [1, 2]},
    ]

    incomplete = deepcopy(changes)
    incomplete["mapping_complete"] = False
    with pytest.raises(ValueError, match="complete explicit atom mapping"):
        reaction._bond_change_records(
            incomplete, reference_structure=_reactant_structure()
        )


def _mapped_path_structure(map_numbers=(1, 2), offset=0.0):
    return {
        "atoms": [
            {
                "element": "H",
                "position_angstrom": [offset, 0.0, 0.0],
                "atom_map_number": map_numbers[0],
            },
            {
                "element": "H",
                "position_angstrom": [offset + 1.0, 0.0, 0.0],
                "atom_map_number": map_numbers[1],
            },
        ],
        "charge": 0,
        "multiplicity": 1,
    }


def _path_validation_request(frames, *, product=None, bond_changes=None):
    inputs = {
        "path": {"images": frames},
        "reactant": _mapped_path_structure(),
        "product": product or _mapped_path_structure(),
    }
    if bond_changes is not None:
        inputs["bond_changes"] = bond_changes
    return {
        "inputs": inputs,
        "action_settings": {
            "endpoint_rmsd_tolerance_angstrom": 0.01,
            "maximum_image_step_rmsd_angstrom": 0.01,
            "bond_distance_tolerance_angstrom": 0.01,
        },
    }


def test_path_validation_rejects_same_element_atom_map_permutation():
    frames = [_mapped_path_structure((2, 1)) for _ in range(3)]

    response = reaction._validate_reaction_path(_path_validation_request(frames))
    result = response["result"]

    assert response["status"] == "success"
    assert result["valid"] is False
    assert result["atom_maps_checked"] is True
    assert result["checks"]["atom_order_consistent"] is True
    assert result["checks"]["atom_identity_consistent"] is False
    assert result["checks"]["bond_changes_progress"] is None
    assert result["bond_changes_requested"] is False
    assert result["bond_changes_checked"] is False


def test_multixyz_restores_validated_atom_maps_from_reference(tmp_path):
    """XYZ 不携带 map number 时，路径帧沿已验证元素顺序恢复原子身份。"""

    trajectory = tmp_path / "path.trj"
    trajectory.write_text(
        "2\nframe 1\nH 0 0 0\nH 1 0 0\n"
        "2\nframe 2\nH 0.1 0 0\nH 1.1 0 0\n",
        encoding="utf-8",
    )

    frames = reaction._parse_multixyz(
        trajectory,
        charge=0,
        multiplicity=1,
        reference_structure=_mapped_path_structure((11, 22)),
    )

    assert [frame["atom_map_numbers"] for frame in frames] == [[11, 22], [11, 22]]
    assert [
        [atom["atom_map_number"] for atom in frame["atoms"]] for frame in frames
    ] == [[11, 22], [11, 22]]


def test_path_validation_distinguishes_absent_bond_changes_from_success():
    frames = [_mapped_path_structure() for _ in range(3)]

    result = reaction._validate_reaction_path(
        _path_validation_request(frames)
    )["result"]

    assert result["valid"] is True
    assert result["checks"]["atom_identity_consistent"] is True
    assert result["checks"]["bond_changes_progress"] is None
    assert result["bond_changes_requested"] is False
    assert result["bond_changes_checked"] is False


def test_reaction_coordinate_refuses_incomplete_or_permuted_path_maps():
    mapped = _mapped_path_structure()
    missing = _mapped_path_structure()
    for atom in missing["atoms"]:
        atom.pop("atom_map_number")
    permuted = _mapped_path_structure((2, 1))

    for frames in ([mapped, missing, mapped], [mapped, permuted, mapped]):
        with pytest.raises(ValueError, match="complete, identical atom-map sequences"):
            reaction._analyze_reaction_coordinate(
                {
                    "inputs": {
                        "path": {"images": frames},
                        "energies": [0.0, 1.0, 0.0],
                    },
                    "action_settings": {"energy_unit": "kcal/mol"},
                }
            )


@pytest.mark.parametrize("invalid_maps", [(1, 1), (1, None), (1, 0)])
def test_path_map_records_must_be_complete_unique_positive(invalid_maps):
    invalid = _mapped_path_structure(invalid_maps)
    if invalid_maps[1] is None:
        invalid["atoms"][1].pop("atom_map_number")

    with pytest.raises(ValueError, match="atom map|atom-map|partial atom mapping"):
        reaction._validate_reaction_path(
            _path_validation_request(
                [_mapped_path_structure(), invalid, _mapped_path_structure()]
            )
        )


def test_pysisyphus_path_energies_fall_back_to_final_hdf5_cycle(tmp_path):
    import h5py

    with h5py.File(tmp_path / "optimization.h5", "w") as archive:
        archive.create_dataset(
            "opt/energies",
            data=[[-10.0, -9.0, -11.0], [-10.1, -8.8, -11.2]],
        )

    assert reaction._pysisyphus_final_path_energies(tmp_path, 3) == [
        -10.1,
        -8.8,
        -11.2,
    ]


def test_pysisyphus_scan_disables_gdiis_for_rebuilt_internal_coordinates(
    tmp_path, monkeypatch
):
    """扫描配置必须避开 pysisyphus 1.0.0 的可变内坐标 GDIIS 历史。"""

    structure = {
        "atoms": [
            {"element": "C", "position_angstrom": [0.0, 0.0, 0.0]},
            {"element": "C", "position_angstrom": [1.4, 0.0, 0.0]},
            {"element": "C", "position_angstrom": [2.0, 1.2, 0.0]},
            {"element": "H", "position_angstrom": [2.6, 1.6, 0.9]},
        ],
        "charge": 0,
        "multiplicity": 1,
    }
    request = {
        "inputs": {"structure": structure},
        "method_spec": {"calculator_backend": "xtb", "method": "GFN2-xTB"},
        "action_settings": {
            "coordinate_type": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "start_value": 0.0,
            "end_value": 60.0,
            "value_unit": "degree",
            "steps": 12,
            "optimizer": "rfo",
            "convergence": "gau",
            "max_cycles": 200,
            "hessian_init": "xtb",
        },
        "resource_limits": {"cpu_cores": 1},
    }

    def fake_run_external(*, directory, **_kwargs):
        (directory / "relaxed_scan.dat").write_text(
            "".join(f"{index}.0 -1.0\n" for index in range(13)),
            encoding="utf-8",
        )
        return {
            "available": True,
            "returncode": 0,
            "stdout": "Converged!\n",
            "stderr": "",
            "command": ["pysis", "pysis.yaml"],
        }

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(reaction, "run_external", fake_run_external)

    response = reaction.execute(
        "scan_reaction_coordinates", "pysisyphus", request
    )

    assert response["status"] == "success"
    assert response["provenance"]["generated_config"]["scan"]["opt"][
        "gdiis"
    ] is False
    assert response["provenance"]["generated_config"]["scan"]["start"] == pytest.approx(
        0.0
    )
    assert response["provenance"]["generated_config"]["scan"]["end"] == pytest.approx(
        1.0471975511965976
    )


@pytest.mark.parametrize(
    "final_row, expected_count",
    [([-10.0, -9.0], 3), ([-10.0, float("nan"), -11.0], 3)],
)
def test_pysisyphus_path_energy_fallback_fails_closed(
    tmp_path, final_row, expected_count
):
    import h5py

    with h5py.File(tmp_path / "optimization.h5", "w") as archive:
        archive.create_dataset("opt/energies", data=[final_row])

    assert (
        reaction._pysisyphus_final_path_energies(tmp_path, expected_count) is None
    )
