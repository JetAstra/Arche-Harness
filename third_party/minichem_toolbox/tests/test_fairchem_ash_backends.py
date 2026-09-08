from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import pytest

from minichem_toolbox.backends import ash_neb, fairchem


@pytest.mark.parametrize("module", (fairchem, ash_neb))
def test_model_download_policy_updates_imported_huggingface_constant(
    monkeypatch, module
):
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_OFFLINE = False
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with module._model_download_policy(False):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert constants.HF_HUB_OFFLINE is True

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert constants.HF_HUB_OFFLINE is False


def _structure(offset: float = 0.0):
    return {
        "atoms": [
            {
                "element": "H",
                "position_angstrom": [offset, 0.0, 0.0],
                "atom_map_number": 1,
            },
            {
                "element": "H",
                "position_angstrom": [offset + 0.8, 0.0, 0.0],
                "atom_map_number": 2,
            },
        ],
        "bonds": [{"atom_index_a": 0, "atom_index_b": 1, "bond_order": 1.0}],
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }


class _FakeAtoms:
    def __init__(self):
        self.info = {}
        self.calc = None

    def __len__(self):
        return 2

    def set_constraint(self, _constraints):
        pass

    def get_potential_energy(self):
        return -1.25


class _FakeOptimizer:
    def __init__(self, _atoms, *, logfile, trajectory):
        self.logfile = Path(logfile)
        self.trajectory = Path(trajectory)
        self.steps = 0

    def run(self, *, fmax, steps):
        self.steps = min(3, steps)
        self.logfile.write_text(f"fmax={fmax}\n", encoding="utf-8")
        self.trajectory.write_text("fake trajectory\n", encoding="utf-8")
        return True

    def get_number_of_steps(self):
        return self.steps


def _fake_ase_optimize(monkeypatch):
    ase_module = types.ModuleType("ase")
    optimize_module = types.ModuleType("ase.optimize")
    optimize_module.BFGS = _FakeOptimizer
    optimize_module.LBFGS = _FakeOptimizer
    optimize_module.FIRE = _FakeOptimizer
    ase_module.optimize = optimize_module
    monkeypatch.setitem(sys.modules, "ase", ase_module)
    monkeypatch.setitem(sys.modules, "ase.optimize", optimize_module)


def test_fairchem_optimization_preserves_graph_identity(monkeypatch, tmp_path):
    _fake_ase_optimize(monkeypatch)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    atoms = _FakeAtoms()
    monkeypatch.setattr(fairchem, "ase_atoms", lambda _value: atoms)
    monkeypatch.setattr(
        fairchem, "_build_calculator", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        fairchem,
        "structure_from_atoms",
        lambda _atoms: {
            "atoms": [
                {"element": "H", "position_angstrom": [0.1, 0.0, 0.0]},
                {"element": "H", "position_angstrom": [0.9, 0.0, 0.0]},
            ],
            "charge": 0,
            "multiplicity": 1,
            "pbc": [False, False, False],
        },
    )
    original_structure = _structure()
    original_structure["atom_map_numbers"] = [1, 2]
    request = {
        "inputs": {"structure": original_structure},
        "method_spec": {
            "model": "uma-s-1p1",
            "task_name": "omol",
            "device": "cpu",
            "allow_model_download": False,
        },
        "action_settings": {
            "fmax_ev_per_angstrom": 0.05,
            "max_steps": 50,
            "optimizer": "bfgs",
        },
        "resource_limits": {"cpu_cores": 2, "gpu_count": 0},
    }

    response = fairchem.execute("optimize_geometry", "fairchem_uma", request)

    assert response["status"] == "success"
    result = response["result"]
    assert result["converged"] is True
    assert result["energy_ev"] == pytest.approx(-1.25)
    assert result["structure"]["bonds"] == request["inputs"]["structure"]["bonds"]
    assert [atom["atom_map_number"] for atom in result["structure"]["atoms"]] == [1, 2]
    assert result["structure"]["atom_map_numbers"] == [1, 2]
    assert atoms.info == {"charge": 0, "spin": 1}
    assert response["provenance"]["atomic_graph_preserved"] is True


def test_fairchem_rejects_nonfinite_optimized_coordinates(monkeypatch, tmp_path):
    _fake_ase_optimize(monkeypatch)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    atoms = _FakeAtoms()
    monkeypatch.setattr(fairchem, "ase_atoms", lambda _value: atoms)
    monkeypatch.setattr(
        fairchem, "_build_calculator", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        fairchem,
        "structure_from_atoms",
        lambda _atoms: {
            "atoms": [
                {
                    "element": "H",
                    "position_angstrom": [float("nan"), 0.0, 0.0],
                },
                {"element": "H", "position_angstrom": [0.9, 0.0, 0.0]},
            ],
            "charge": 0,
            "multiplicity": 1,
            "pbc": [False, False, False],
        },
    )
    request = {
        "inputs": {"structure": _structure()},
        "method_spec": {
            "model": "uma-s-1p1",
            "task_name": "omol",
            "device": "cpu",
            "allow_model_download": False,
        },
        "action_settings": {
            "fmax_ev_per_angstrom": 0.05,
            "max_steps": 50,
            "optimizer": "bfgs",
        },
        "resource_limits": {"cpu_cores": 2, "gpu_count": 0},
    }

    with pytest.raises(ValueError, match="three finite"):
        fairchem.execute("optimize_geometry", "fairchem_uma", request)


def test_fairchem_cuda_requires_explicit_gpu(monkeypatch):
    _fake_ase_optimize(monkeypatch)
    request = {
        "inputs": {"structure": _structure()},
        "method_spec": {
            "model": "uma-s-1p1",
            "task_name": "omol",
            "device": "cuda",
            "allow_model_download": False,
        },
        "action_settings": {
            "fmax_ev_per_angstrom": 0.05,
            "max_steps": 50,
            "optimizer": "bfgs",
        },
        "resource_limits": {"gpu_count": 0},
    }
    with pytest.raises(ValueError, match="gpu_count"):
        fairchem.execute("optimize_geometry", "fairchem_uma", request)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda structure: structure.pop("charge"), "explicitly declare charge"),
        (lambda structure: structure.update(charge=True), "charge must be an integer"),
        (lambda structure: structure.update(charge=1.5), "charge must be an integer"),
        (
            lambda structure: structure.update(pbc=[True, False, False]),
            "does not accept periodic",
        ),
    ],
)
def test_fairchem_requires_explicit_nonperiodic_state(
    monkeypatch, mutation, message
):
    _fake_ase_optimize(monkeypatch)
    structure = _structure()
    mutation(structure)
    request = {
        "inputs": {"structure": structure},
        "method_spec": {
            "model": "uma-s-1p1",
            "task_name": "omol",
            "device": "cpu",
            "allow_model_download": False,
        },
        "action_settings": {
            "fmax_ev_per_angstrom": 0.05,
            "max_steps": 50,
            "optimizer": "bfgs",
        },
        "resource_limits": {"gpu_count": 0},
    }
    with pytest.raises(ValueError, match=message):
        fairchem.execute("optimize_geometry", "fairchem_uma", request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fmax_ev_per_angstrom", float("nan")),
        ("fmax_ev_per_angstrom", float("inf")),
        ("max_steps", True),
        ("max_steps", fairchem.HARD_MAX_OPTIMIZATION_STEPS + 1),
    ],
)
def test_fairchem_optimization_limits_are_finite_and_bounded(
    monkeypatch, field, value
):
    _fake_ase_optimize(monkeypatch)
    request = {
        "inputs": {"structure": _structure()},
        "method_spec": {
            "model": "uma-s-1p1",
            "task_name": "omol",
            "device": "cpu",
            "allow_model_download": False,
        },
        "action_settings": {
            "fmax_ev_per_angstrom": 0.05,
            "max_steps": 50,
            "optimizer": "bfgs",
        },
        "resource_limits": {"gpu_count": 0},
    }
    request["action_settings"][field] = value
    with pytest.raises(ValueError, match=field):
        fairchem.execute("optimize_geometry", "fairchem_uma", request)


def _xyz_frame(structure, energy):
    atoms = structure["atoms"]
    lines = [str(len(atoms)), f"Image Energy: {energy}"]
    lines.extend(
        f"{atom['element']} {' '.join(str(value) for value in atom['position_angstrom'])}"
        for atom in atoms
    )
    return "\n".join(lines) + "\n"


def _ash_request(*, reactant=None, product=None, settings=None):
    action_settings = {
        "total_images": 3,
        "climb": True,
        "maximum_force_ev_per_angstrom": 0.05,
        "rms_force_ev_per_angstrom": 0.025,
        "max_steps": 300,
        "interpolation": "idpp",
        "idpp_max_iterations": 700,
    }
    if settings:
        action_settings.update(settings)
    return {
        "inputs": {
            "reactant": reactant or _structure(0.0),
            "product": product or _structure(0.2),
        },
        "method_spec": {
            "model": "uma-s-1p1",
            "task_name": "omol",
            "device": "cpu",
            "allow_model_download": False,
            "random_seed": 41,
        },
        "action_settings": action_settings,
        "resource_limits": {"cpu_cores": 2, "gpu_count": 0},
    }


def test_ash_multixyz_rejects_nonfinite_coordinates(tmp_path):
    path = tmp_path / "knarr_MEP.xyz"
    path.write_text(
        "2\nImage Energy: -1.0\nH nan 0.0 0.0\nH 0.8 0.0 0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="three-dimensional finite coordinates"):
        ash_neb._parse_multixyz(
            path,
            charge=0,
            multiplicity=1,
            reactant_structure=_structure(),
            product_structure=_structure(0.1),
        )


@pytest.mark.parametrize(
    "rows",
    [
        [[float("nan"), 0.0, 0.0], [0.8, 0.0, 0.0]],
        [[0.0, 0.0], [0.8, 0.0, 0.0]],
    ],
)
def test_ash_saddle_rejects_nonfinite_or_non_three_dimensional_rows(rows):
    ash_result = types.SimpleNamespace(geometry=rows)

    with pytest.raises(ValueError, match="exactly three finite coordinates"):
        ash_neb._saddle_structure(
            ash_result,
            ["H", "H"],
            charge=0,
            multiplicity=1,
            reference_structure=_structure(),
        )


@pytest.mark.parametrize("header", ["2", "2.0", "2.000"])
def test_knarr_saddle_accepts_integral_numeric_atom_count_header(tmp_path, header):
    """KNARR saddle 的数值头只要表示正整数都应可读取。"""

    path = tmp_path / "knarr_saddle.xyz"
    path.write_text(
        f"{header}\nvalid saddle\nH 0.1 0.0 0.0\nH 0.9 0.0 0.0\n",
        encoding="utf-8",
    )

    parsed = ash_neb._parse_knarr_saddle_xyz(
        path,
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
        reference_structure=_structure(),
    )

    assert parsed["atoms"][0]["position_angstrom"] == [0.1, 0.0, 0.0]


@pytest.mark.parametrize("header", ["2.5", "nan", "inf", "0", "-2", "1.0"])
def test_knarr_saddle_rejects_non_positive_or_non_integral_atom_count_header(
    tmp_path, header
):
    """KNARR saddle 的原子数头必须是有限、正的整数且与端点一致。"""

    path = tmp_path / "knarr_saddle.xyz"
    path.write_text(
        f"{header}\ninvalid atom count\nH 0.1 0.0 0.0\nH 0.9 0.0 0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="atom count"):
        ash_neb._parse_knarr_saddle_xyz(
            path,
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
            reference_structure=_structure(),
        )


def test_knarr_saddle_rejects_atom_count_mismatch(tmp_path):
    """KNARR saddle 的正整数原子数仍必须和端点原子数一致。"""

    path = tmp_path / "knarr_saddle.xyz"
    path.write_text(
        "3.0\nwrong count\nH 0.1 0.0 0.0\nH 0.9 0.0 0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        ash_neb._parse_knarr_saddle_xyz(
            path,
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
            reference_structure=_structure(),
        )


def _install_fake_ash(monkeypatch, fake_neb, calls=None):
    calls = calls if calls is not None else {}

    class FakeFragment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTheory:
        def __init__(self, **kwargs):
            calls["theory"] = kwargs

    ash_module = types.ModuleType("ash")
    ash_module.Fragment = FakeFragment
    ash_module.FairchemTheory = FakeTheory
    ash_module.NEB = fake_neb
    monkeypatch.setitem(sys.modules, "ash", ash_module)
    return calls


def test_ash_neb_converts_total_images_and_returns_unvalidated_guess(
    monkeypatch, tmp_path
):
    initial_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    initial_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    calls = {}

    def fake_neb(**kwargs):
        calls["neb"] = kwargs
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB-CI calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
            geometry=[[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )

    _install_fake_ash(monkeypatch, fake_neb, calls)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    reactant = _structure(0.0)
    reactant["atoms"][0].update(
        {
            "formal_charge": -1,
            "isotope": 2,
            "radical_electrons": 1,
            "site_label": "reactive-H",
        }
    )
    product = _structure(0.2)
    product["bonds"] = []
    product["atoms"][0]["formal_charge"] = 0
    request = _ash_request(reactant=reactant, product=product)

    response = ash_neb.execute("search_reaction_path", "ash_neb_uma", request)

    assert response["status"] == "success"
    assert calls["neb"]["images"] == 1
    assert calls["neb"]["tol_max_fci"] == pytest.approx(0.05)
    assert calls["neb"]["tol_rms_fci"] == pytest.approx(0.025)
    assert response["result"]["total_images"] == 3
    assert len(response["result"]["images"]) == 3
    assert response["result"]["highest_energy_image_index"] == 1
    assert response["result"]["highest_interior_image_index"] == 1
    assert response["result"]["energy_index_complete"] is True
    assert response["result"]["final_path_complete"] is True
    for image in response["result"]["images"]:
        assert [atom["atom_map_number"] for atom in image["atoms"]] == [1, 2]
        assert image["atoms"][0]["isotope"] == 2
        assert image["atoms"][0]["site_label"] == "reactive-H"
    first, interior, last = response["result"]["images"]
    assert first["bonds"] == reactant["bonds"]
    assert first["atoms"][0]["formal_charge"] == -1
    assert first["atoms"][0]["radical_electrons"] == 1
    assert "bonds" not in interior
    assert "formal_charge" not in interior["atoms"][0]
    assert "radical_electrons" not in interior["atoms"][0]
    assert last["bonds"] == []
    assert last["atoms"][0]["formal_charge"] == 0
    assert "radical_electrons" not in last["atoms"][0]
    saddle = response["result"]["transition_state_guess"]
    assert "bonds" not in saddle
    assert saddle["atoms"][0]["atom_map_number"] == 1
    assert "formal_charge" not in saddle["atoms"][0]
    assert saddle["atoms"][0]["isotope"] == 2
    assert response["provenance"]["path_is_validated_transition_state"] is False
    assert response["provenance"]["endpoint_atom_maps_checked"] is True
    assert os.environ.get("HF_HUB_OFFLINE") == initial_hf_hub_offline
    assert os.environ.get("TRANSFORMERS_OFFLINE") == initial_transformers_offline


def test_ash_neb_recovers_knarr_saddle_as_unvalidated_guess(
    monkeypatch, tmp_path
):
    """作者 KNARR saddle 文件应成为未验证 TS guess，而不是只保留路径。"""

    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        Path("knarr_saddle.xyz").write_text(
            _xyz_frame(frames[1], -0.8).replace("2\n", "2.0\n", 1),
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute(
        "search_reaction_path", "ash_neb_uma", _ash_request()
    )

    assert response["status"] == "success"
    guess = response["result"]["transition_state_guess"]
    assert guess["atoms"][0]["position_angstrom"] == [0.1, 0.0, 0.0]
    assert response["result"]["transition_state_guess_source"] == "knarr_saddle_xyz"
    assert response["result"]["transition_state_guess_source_image_index"] is None
    assert response["result"]["transition_state_guess_selection_rule_version"]
    assert response["provenance"]["path_is_validated_transition_state"] is False


def test_ash_neb_rejects_invalid_knarr_saddle_structure(
    monkeypatch, tmp_path
):
    """saddle 文件结构无效时仍保留可用路径候选并标记 partial。"""

    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        Path("knarr_saddle.xyz").write_text(
            "2\ninvalid saddle\nH nan 0.0 0.0\nH 0.9 0.0 0.0\n",
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute(
        "search_reaction_path", "ash_neb_uma", _ash_request()
    )

    assert response["status"] == "partial_success"
    assert response["result"]["transition_state_guess"]["atoms"][0][
        "position_angstrom"
    ] == [0.1, 0.0, 0.0]
    assert response["result"]["transition_state_guess_source"] == (
        "highest_interior_path_image"
    )
    assert response["result"]["transition_state_guess_source_image_index"] == 1
    assert any("invalid knarr_saddle.xyz" in warning for warning in response["warnings"])
    assert any("falling back" in warning for warning in response["warnings"])


def test_ash_neb_invalid_knarr_saddle_prefers_ash_saddle_before_path_image(
    monkeypatch, tmp_path
):
    """无效 KNARR 文件时按 ASH saddle、最高内部图像的顺序选择候选。"""

    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        Path("knarr_saddle.xyz").write_text(
            "2.5\ninvalid saddle\nC nan 0.0 0.0\nH 0.9 0.0 0.0\n",
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
            geometry=[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute(
        "search_reaction_path", "ash_neb_uma", _ash_request()
    )

    assert response["status"] == "partial_success"
    assert response["result"]["transition_state_guess"]["atoms"][0][
        "position_angstrom"
    ] == [0.25, 0.0, 0.0]
    assert response["result"]["transition_state_guess_source"] == (
        "ash_saddle_geometry"
    )
    assert response["result"]["transition_state_guess_source_image_index"] is None
    assert any("invalid knarr_saddle.xyz" in warning for warning in response["warnings"])
    assert any("ASH saddle geometry" in warning for warning in response["warnings"])


def test_ash_neb_invalid_ash_saddle_falls_back_to_highest_interior_path_image(
    monkeypatch, tmp_path
):
    """ASH saddle 几何损坏时，完整路径仍回退最高内部图像并保留原因。"""

    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
            geometry=[[float("nan"), 0.0, 0.0], [0.9, 0.0, 0.0]],
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute(
        "search_reaction_path", "ash_neb_uma", _ash_request()
    )

    assert response["status"] == "partial_success"
    assert response["result"]["transition_state_guess_source"] == (
        "highest_interior_path_image"
    )
    assert response["result"]["transition_state_guess_source_image_index"] == 1
    assert any("invalid saddle geometry" in warning for warning in response["warnings"])


def test_ash_neb_rejects_incomplete_path_even_when_saddle_exists(
    monkeypatch, tmp_path
):
    """路径帧数不足时不能把单独 saddle 文件交接为 TS guess。"""

    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8))
            ),
            encoding="utf-8",
        )
        Path("knarr_saddle.xyz").write_text(
            _xyz_frame(_structure(0.1), -0.8), encoding="utf-8"
        )
        return types.SimpleNamespace(
            label="NEB calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute(
        "search_reaction_path", "ash_neb_uma", _ash_request()
    )

    assert response["status"] == "partial_success"
    assert response["result"]["final_path_complete"] is False
    assert response["result"]["transition_state_guess"] is None
    assert response["result"]["transition_state_guess_source"] is None
    assert response["result"]["transition_state_guess_source_image_index"] is None
    assert response["result"]["transition_state_guess_selection_rule_version"] is None


def test_ash_neb_falls_back_to_highest_interior_path_image(
    monkeypatch, tmp_path
):
    """缺少 saddle 文件时，完整路径的最高内部图像仍可作为未验证 guess。"""

    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute(
        "search_reaction_path", "ash_neb_uma", _ash_request()
    )

    assert response["status"] == "success"
    assert response["result"]["transition_state_guess"]["atoms"][0][
        "position_angstrom"
    ] == [0.1, 0.0, 0.0]
    assert response["result"]["transition_state_guess_source"] == (
        "highest_interior_path_image"
    )
    assert response["result"]["transition_state_guess_source_image_index"] == 1
    assert response["result"]["transition_state_guess_selection_rule_version"]


def test_ash_neb_rejects_mismatched_atom_order():
    reactant = _structure()
    reactant["atoms"][1]["element"] = "F"
    product = _structure()
    product["atoms"][1]["element"] = "F"
    product["atoms"].reverse()
    with pytest.raises(ValueError, match="identical atom counts, elements, and order"):
        ash_neb._validate_endpoints(reactant, product)


def test_ash_neb_rejects_same_element_atom_map_permutation():
    reactant = _structure()
    product = _structure(0.2)
    product["atoms"][0]["atom_map_number"] = 2
    product["atoms"][1]["atom_map_number"] = 1

    with pytest.raises(
        ValueError, match="atom maps must be identical at every atom index"
    ):
        ash_neb._validate_endpoints(reactant, product)


def test_ash_neb_rejects_partial_or_one_sided_atom_maps():
    reactant = _structure()
    partial_product = _structure(0.2)
    del partial_product["atoms"][1]["atom_map_number"]
    with pytest.raises(ValueError, match="partial atom mapping"):
        ash_neb._validate_endpoints(reactant, partial_product)

    unmapped_product = _structure(0.2)
    for atom in unmapped_product["atoms"]:
        del atom["atom_map_number"]
    with pytest.raises(
        ValueError, match="both endpoints must provide complete atom maps"
    ):
        ash_neb._validate_endpoints(reactant, unmapped_product)


@pytest.mark.parametrize("invalid_map", [True, 0, -1, 1.0, "1"])
def test_ash_neb_requires_positive_integer_atom_maps(invalid_map):
    reactant = _structure()
    product = _structure(0.2)
    product["atoms"][0]["atom_map_number"] = invalid_map

    with pytest.raises(ValueError, match="must be a positive integer"):
        ash_neb._validate_endpoints(reactant, product)


def test_ash_neb_requires_unique_atom_maps():
    reactant = _structure()
    product = _structure(0.2)
    product["atoms"][1]["atom_map_number"] = 1

    with pytest.raises(ValueError, match="must be unique"):
        ash_neb._validate_endpoints(reactant, product)


def test_ash_neb_requires_explicit_nonperiodic_endpoint_state():
    reactant = _structure()
    product = _structure(0.2)
    del reactant["charge"]
    with pytest.raises(ValueError, match="explicitly declare charge and multiplicity"):
        ash_neb._validate_endpoints(reactant, product)

    reactant = _structure()
    reactant["pbc"] = [True, False, False]
    with pytest.raises(ValueError, match="does not accept periodic"):
        ash_neb._validate_endpoints(reactant, product)


def test_highest_endpoint_is_not_reported_as_highest_interior(monkeypatch, tmp_path):
    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.2))
            ),
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB-CI calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.2},
            geometry=[[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    response = ash_neb.execute("search_reaction_path", "ash_neb_uma", _ash_request())

    assert response["status"] == "partial_success"
    assert response["result"]["highest_energy_image_index"] == 2
    assert response["result"]["highest_interior_image_index"] == 1
    assert response["result"]["interior_peak_candidate_valid"] is False
    assert response["result"]["transition_state_guess"] is None
    assert response["result"]["diagnostic_ash_saddle_geometry"] is not None


def test_ash_neb_initial_guess_cannot_satisfy_success(monkeypatch, tmp_path):
    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("initial_guess_path.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB-CI calc",
            MEP_energies_dict={0: -1.0, 1: -0.8, 2: -0.9},
            geometry=[[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute("search_reaction_path", "ash_neb_uma", _ash_request())

    assert response["status"] == "partial_success"
    assert response["result"]["converged"] is True
    assert response["result"]["final_path_complete"] is False
    assert response["result"]["images"] == []
    assert response["result"]["path_file"] is None
    assert response["result"]["initial_guess_path"] is not None
    assert any("cannot satisfy success" in warning for warning in response["warnings"])


@pytest.mark.parametrize(
    "energy_mapping",
    [
        {0: -1.0, 2: -0.9},
        {0: -1.0, 1: float("nan"), 2: -0.9},
    ],
)
def test_ash_neb_requires_complete_finite_energy_indices(
    monkeypatch, tmp_path, energy_mapping
):
    def fake_neb(**_kwargs):
        frames = [_structure(0.0), _structure(0.1), _structure(0.2)]
        Path("knarr_MEP_FULL.xyz").write_text(
            "".join(
                _xyz_frame(frame, energy)
                for frame, energy in zip(frames, (-1.0, -0.8, -0.9))
            ),
            encoding="utf-8",
        )
        return types.SimpleNamespace(
            label="NEB-CI calc",
            MEP_energies_dict=energy_mapping,
            geometry=[[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]],
        )

    _install_fake_ash(monkeypatch, fake_neb)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))

    response = ash_neb.execute("search_reaction_path", "ash_neb_uma", _ash_request())

    assert response["status"] == "partial_success"
    assert response["result"]["converged"] is False
    assert response["result"]["energy_index_complete"] is False
    assert response["result"]["final_path_complete"] is True
    assert all(math.isfinite(value) for value in response["result"]["energies_hartree"])
    assert any("energy" in warning.casefold() for warning in response["warnings"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_images", True),
        ("total_images", 3.0),
        ("total_images", ash_neb.HARD_MAX_TOTAL_IMAGES + 1),
        ("max_steps", False),
        ("max_steps", ash_neb.HARD_MAX_NEB_STEPS + 1),
        ("idpp_max_iterations", ash_neb.HARD_MAX_IDPP_ITERATIONS + 1),
        ("maximum_force_ev_per_angstrom", float("nan")),
        ("maximum_force_ev_per_angstrom", float("inf")),
        (
            "maximum_force_ev_per_angstrom",
            ash_neb.HARD_MAX_FORCE_THRESHOLD_EV_PER_ANGSTROM + 0.1,
        ),
        ("rms_force_ev_per_angstrom", True),
    ],
)
def test_ash_neb_settings_are_strictly_bounded(field, value):
    settings = _ash_request()["action_settings"]
    settings[field] = value

    with pytest.raises(ValueError, match=f"action_settings\\.{field}"):
        ash_neb._validated_neb_settings(settings)
