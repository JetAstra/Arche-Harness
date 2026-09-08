from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from minichem_toolbox.backends import electronic


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CASE1_RUNTIME = REPOSITORY_ROOT / ".runtime" / "case1_direct_20260813"


def _structure(symbols, positions):
    return {
        "atoms": [
            {"element": symbol, "position_angstrom": position}
            for symbol, position in zip(symbols, positions)
        ],
        "charge": 0,
        "multiplicity": 1,
    }


def _external_and_internal_bases(structure, *, rotational_degrees):
    atoms = Atoms(
        [item["element"] for item in structure["atoms"]],
        positions=[item["position_angstrom"] for item in structure["atoms"]],
    )
    masses = atoms.get_masses()
    positions = atoms.get_positions()
    centered = positions - np.average(positions, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for mass, position in zip(masses, centered):
        inertia += mass * (
            np.dot(position, position) * np.eye(3) - np.outer(position, position)
        )
    _, axes = np.linalg.eigh(inertia)

    square_root_masses = np.sqrt(masses)
    translations = np.zeros((3 * len(atoms), 3))
    for axis in range(3):
        translations[axis::3, axis] = square_root_masses
    translation_basis, _ = np.linalg.qr(translations, mode="reduced")
    rotations = np.stack(
        [
            (
                square_root_masses[:, np.newaxis]
                * np.cross(axis[np.newaxis, :], centered)
            ).reshape(-1)
            for axis in axes.T
        ],
        axis=1,
    )
    rotations -= translation_basis @ (translation_basis.T @ rotations)
    rotation_basis = np.linalg.svd(rotations, full_matrices=False)[0][
        :, :rotational_degrees
    ]
    external_basis = np.linalg.qr(
        np.column_stack((translation_basis, rotation_basis)), mode="reduced"
    )[0]
    complete_basis = np.linalg.qr(external_basis, mode="complete")[0]
    return atoms, external_basis, complete_basis[:, 3 + rotational_degrees :]


def _hessian_with_external_noise(
    structure,
    *,
    rotational_degrees,
    internal_eigenvalues,
):
    atoms, external_basis, internal_basis = _external_and_internal_bases(
        structure, rotational_degrees=rotational_degrees
    )
    external_eigenvalues = -np.arange(1, external_basis.shape[1] + 1, dtype=float)
    mass_weighted_hessian = (
        external_basis @ np.diag(external_eigenvalues) @ external_basis.T
        + internal_basis
        @ np.diag(np.asarray(internal_eigenvalues, dtype=float))
        @ internal_basis.T
    )
    square_root_masses = np.repeat(np.sqrt(atoms.get_masses()), 3)
    hessian = (
        square_root_masses[:, np.newaxis]
        * mass_weighted_hessian
        * square_root_masses[np.newaxis, :]
    )
    return hessian


def _run(structure, hessian, *, linearity, unit="eV/angstrom^2"):
    return electronic.execute(
        "derive_vibrational_modes",
        "internal_vibrations",
        {
            "inputs": {
                "hessian": {"matrix": hessian.tolist(), "unit": unit},
                "structure": structure,
            },
            "method_spec": {},
            "action_settings": {"linearity": linearity},
        },
    )["result"]


def test_nonlinear_projection_removes_six_external_modes_without_frequency_cutoff():
    structure = _structure(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    hessian = _hessian_with_external_noise(
        structure,
        rotational_degrees=3,
        internal_eigenvalues=[-0.25, 1.0e-8, 1.0],
    )

    result = _run(structure, hessian, linearity="nonlinear")

    assert result["cartesian_degree_count"] == 9
    assert result["external_degree_count"] == 6
    assert result["vibrational_mode_count"] == 3
    assert result["imaginary_frequency_count"] == 1
    assert np.asarray(result["modes"]).shape == (3, 3, 3)
    real_frequencies = [
        item["real"] for item in result["frequencies_cm1"] if item["real"] > 0.0
    ]
    assert min(real_frequencies) < 0.1
    assert "not removed by a frequency cutoff" in result["external_mode_treatment"]


@pytest.mark.parametrize("requested_linearity", ["linear", "auto"])
def test_linear_projection_removes_five_external_modes(requested_linearity):
    structure = _structure(
        ["O", "C", "O"],
        [[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]],
    )
    hessian = _hessian_with_external_noise(
        structure,
        rotational_degrees=2,
        internal_eigenvalues=[1.0e-8, 0.1, 0.2, 0.3],
    )

    result = _run(structure, hessian, linearity=requested_linearity)

    assert result["linearity"] == "linear"
    assert result["linearity_requested"] == requested_linearity
    assert result["external_degree_count"] == 5
    assert result["vibrational_mode_count"] == 4
    assert result["imaginary_frequency_count"] == 0
    resolution = result["linearity_resolution"]
    assert resolution["method"] == (
        "center_of_mass_inertia_tensor_principal_moment_ratio"
    )
    assert resolution["inferred_from_structure"] == "linear"
    assert resolution["minimum_to_maximum_principal_moment_ratio"] == pytest.approx(
        0.0, abs=1.0e-15
    )


def test_explicit_linearity_conflicting_with_geometry_is_rejected():
    structure = _structure(
        ["O", "C", "O"],
        [[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]],
    )
    hessian = np.eye(9)

    with pytest.raises(ValueError, match="conflicts with the structure"):
        _run(structure, hessian, linearity="nonlinear")


@pytest.mark.skipif(
    not (CASE1_RUNTIME / "typed_hessian_TS-2a.json").is_file(),
    reason="optional local Case1 Gaussian Hessian is not present",
)
def test_local_case1_ts2a_hessian_has_153_modes_and_one_imaginary_frequency():
    hessian_payload = json.loads(
        (CASE1_RUNTIME / "typed_hessian_TS-2a.json").read_text(encoding="utf-8")
    )
    old_vibration_payload = json.loads(
        (CASE1_RUNTIME / "typed_vibrations_TS-2a.json").read_text(encoding="utf-8")
    )

    result = _run(
        old_vibration_payload["result"]["structure"],
        np.asarray(hessian_payload["result"]["matrix"]),
        linearity="nonlinear",
        unit=hessian_payload["result"]["unit"],
    )

    assert result["cartesian_degree_count"] == 159
    assert result["vibrational_mode_count"] == 153
    assert result["imaginary_frequency_count"] == 1
    imaginary = [
        item["imaginary"] for item in result["frequencies_cm1"] if item["imaginary"]
    ]
    assert imaginary == pytest.approx([255.2848026609], rel=1.0e-9)
