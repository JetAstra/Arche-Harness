from __future__ import annotations

import pytest

from minichem_toolbox.backends import rdkit_mmff


def _acetamide_structure():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles("CC(=O)N"))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 20260816
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    conformer = molecule.GetConformer()
    return {
        "atoms": [
            {
                "element": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "position_angstrom": list(conformer.GetAtomPosition(atom.GetIdx())),
            }
            for atom in molecule.GetAtoms()
        ],
        "bonds": [
            {
                "atom_index_a": bond.GetBeginAtomIdx(),
                "atom_index_b": bond.GetEndAtomIdx(),
                "bond_order": bond.GetBondTypeAsDouble(),
            }
            for bond in molecule.GetBonds()
        ],
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }


def _request(structure, force_field):
    return {
        "inputs": {"structure": structure, "constraints": []},
        "method_spec": {"force_field": force_field},
        "action_settings": {
            "max_iterations": 1000,
            "force_tolerance": 1.0e-6,
            "energy_tolerance": 1.0e-9,
        },
    }


@pytest.mark.parametrize(
    ("requested_variant", "rdkit_variant"),
    [("mmff94", "MMFF94"), ("mmff94s", "MMFF94s")],
)
def test_public_variant_is_forwarded_with_rdkit_native_spelling(
    tmp_path, monkeypatch, requested_variant, rdkit_variant
):
    from rdkit.Chem import AllChem

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    original = AllChem.MMFFGetMoleculeProperties
    observed = []

    def recording_get_properties(molecule, *, mmffVariant):
        observed.append(mmffVariant)
        return original(molecule, mmffVariant=mmffVariant)

    monkeypatch.setattr(AllChem, "MMFFGetMoleculeProperties", recording_get_properties)

    result = rdkit_mmff.optimize_geometry(
        _request(_acetamide_structure(), requested_variant)
    )

    assert result["status"] == "success"
    assert result["result"]["force_field"] == requested_variant
    assert observed == [rdkit_variant]


def test_mmff94s_does_not_silently_collapse_to_mmff94(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    structure = _acetamide_structure()

    mmff94 = rdkit_mmff.optimize_geometry(_request(structure, "mmff94"))
    mmff94s = rdkit_mmff.optimize_geometry(_request(structure, "mmff94s"))

    assert mmff94["status"] == "success"
    assert mmff94s["status"] == "success"
    assert mmff94["result"]["energy_kcal_mol"] != pytest.approx(
        mmff94s["result"]["energy_kcal_mol"], abs=1.0e-8
    )
