from __future__ import annotations

import math

import pytest

from minichem_toolbox.backends import electronic
from minichem_toolbox.backends import execute_local
from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox.service import _validate_artifact_input_semantics, execute_action


GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083


# These labels are explicit test inputs. They are not inferred from the Case1 candidate ids
# and do not assert a chemical R/S assignment for records that lack IRC/product labels.
CASE1_SHAPED_FREE_ENERGIES = [
    {"label": "pathway_a", "conformer_id": "TS-2a", "gibbs_free_energy": 0.0},
    {"label": "pathway_a", "conformer_id": "TS-2e", "gibbs_free_energy": 0.3655618855},
    {"label": "pathway_b", "conformer_id": "TS-2b-1", "gibbs_free_energy": 1.8848503031},
    {"label": "pathway_b", "conformer_id": "TS-2c", "gibbs_free_energy": 1.9339841695},
    {"label": "pathway_b", "conformer_id": "TS-2b-2", "gibbs_free_energy": 3.2665696547},
    {"label": "pathway_a", "conformer_id": "TS-2g", "gibbs_free_energy": 3.8117561278},
    {"label": "pathway_a", "conformer_id": "TS-2d", "gibbs_free_energy": 4.4876904780},
    {"label": "pathway_a", "conformer_id": "new-1", "gibbs_free_energy": 4.8260937383},
    {"label": "pathway_b", "conformer_id": "TS-2h-2", "gibbs_free_energy": 5.0132860188},
    {"label": "pathway_b", "conformer_id": "new-2", "gibbs_free_energy": 5.1470523842},
    {"label": "pathway_b", "conformer_id": "TS-2h-1", "gibbs_free_energy": 5.3066529738},
    {"label": "pathway_a", "conformer_id": "TS-2f", "gibbs_free_energy": 6.2147534397},
    {"label": "pathway_a", "conformer_id": "new-3", "gibbs_free_energy": 8.3642180236},
    {"label": "pathway_b", "conformer_id": "TS-2i", "gibbs_free_energy": 10.4623589165},
]


def _request(
    records=CASE1_SHAPED_FREE_ENERGIES,
    *,
    unit="kcal_mol",
    temperature=303.15,
    comparison_type="population",
):
    return {
        "inputs": {"free_energies": records, "energy_unit": unit},
        "method_spec": {},
        "action_settings": {
            "temperature_kelvin": temperature,
            "comparison_type": comparison_type,
        },
    }


def _required(contract, section):
    return {item["name"] for item in contract["required_contract"][section]}


def test_catalog_and_discovery_register_explicit_free_energy_selectivity():
    validate_catalog()
    action = action_specs()["analyze_free_energy_selectivity"]
    backend = backend_specs()["internal_thermochemistry"]

    assert action.backend_ids == ("internal_thermochemistry",)
    assert action.selection_policy == "internal_deterministic"
    assert action.id in backend.capabilities

    contract = inspect_action(
        action.id,
        backend_id="internal_thermochemistry",
        detail_level="contract",
    )["selected_request_contract"]
    assert _required(contract, "inputs") == {"free_energies", "energy_unit"}
    assert _required(contract, "action_settings") == {
        "temperature_kelvin",
        "comparison_type",
    }
    input_contracts = {
        item["name"]: item for item in contract["required_contract"]["inputs"]
    }
    assert input_contracts["free_energies"]["minimum_items"] == 2
    assert input_contracts["free_energies"]["item_fields"] == [
        "label",
        "gibbs_free_energy",
        "optional conformer_id",
    ]
    assert input_contracts["energy_unit"]["allowed_values"] == [
        "hartree",
        "eV",
        "kcal_mol",
        "kJ_mol",
    ]
    setting_contracts = {
        item["name"]: item
        for item in contract["required_contract"]["action_settings"]
    }
    assert setting_contracts["comparison_type"]["allowed_values"] == [
        "population",
        "enantiomeric",
    ]


def test_case1_shaped_fourteen_records_are_grouped_by_explicit_labels():
    response = execute_local(
        "analyze_free_energy_selectivity",
        "internal_thermochemistry",
        _request(),
    )

    assert response["status"] == "success"
    result = response["result"]
    assert result["label_order"] == ["pathway_a", "pathway_b"]
    assert result["label_semantics_inferred"] is False
    assert [item["conformer_count"] for item in result["labels"]] == [7, 7]
    assert [
        item["conformer_id"]
        for item in result["labels"][0]["conformers"]
    ] == [
        record["conformer_id"]
        for record in CASE1_SHAPED_FREE_ENERGIES
        if record["label"] == "pathway_a"
    ]

    rt = GAS_CONSTANT_KCAL_MOL_K * 303.15
    partition_a = math.fsum(
        math.exp(-record["gibbs_free_energy"] / rt)
        for record in CASE1_SHAPED_FREE_ENERGIES
        if record["label"] == "pathway_a"
    )
    partition_b = math.fsum(
        math.exp(-record["gibbs_free_energy"] / rt)
        for record in CASE1_SHAPED_FREE_ENERGIES
        if record["label"] == "pathway_b"
    )
    expected_delta_delta_g = -rt * math.log(partition_b) - (
        -rt * math.log(partition_a)
    )
    expected_ratio = partition_a / partition_b
    expected_excess = 100.0 * (
        (partition_a - partition_b) / (partition_a + partition_b)
    )

    comparison = result["two_label_comparison"]
    assert comparison["delta_delta_g_kcal_mol"] == pytest.approx(
        expected_delta_delta_g, abs=1e-12
    )
    assert comparison["population_ratio_first_over_second"] == pytest.approx(
        expected_ratio, rel=1e-12
    )
    assert comparison["signed_population_excess_percent"] == pytest.approx(
        expected_excess, abs=1e-12
    )
    assert comparison["absolute_population_excess_percent"] == pytest.approx(
        abs(expected_excess), abs=1e-12
    )
    assert "ee_percent" not in comparison
    assert "ee_percent" not in result
    assert result["population_sum"] == pytest.approx(1.0, abs=1e-15)
    assert sum(item["population"] for item in result["populations"]) == pytest.approx(
        1.0, abs=1e-15
    )


@pytest.mark.parametrize(
    ("unit", "scale"),
    [
        ("hartree", 1.0 / 627.5094740631),
        ("eV", 1.0 / 23.060547830619),
        ("kcal_mol", 1.0),
        ("kJ_mol", 4.184),
    ],
)
def test_supported_units_give_the_same_two_label_result(unit, scale):
    records = [
        {"label": "explicit_first", "gibbs_free_energy": 0.0},
        {"label": "explicit_second", "gibbs_free_energy": 1.9 * scale},
    ]

    result = electronic.execute(
        "analyze_free_energy_selectivity",
        "internal_thermochemistry",
        _request(records, unit=unit, comparison_type="enantiomeric"),
    )["result"]

    assert result["delta_delta_g_kcal_mol"] == pytest.approx(1.9, rel=2e-12)
    assert result["population_ratio_first_over_second"] == pytest.approx(
        math.exp(1.9 / (GAS_CONSTANT_KCAL_MOL_K * 303.15)), rel=2e-12
    )
    assert result["ee_percent"] == pytest.approx(91.846154, rel=1e-3)
    assert result["absolute_population_excess_percent"] == pytest.approx(
        result["ee_percent"]
    )


def test_n_way_result_omits_two_label_only_quantities():
    response = electronic.execute(
        "analyze_free_energy_selectivity",
        "internal_thermochemistry",
        _request(
            [
                {"label": "alpha", "gibbs_free_energy": 0.0},
                {"label": "beta", "gibbs_free_energy": 0.5},
                {"label": "gamma", "gibbs_free_energy": 1.0},
            ]
        ),
    )

    result = response["result"]
    assert result["label_order"] == ["alpha", "beta", "gamma"]
    assert "two_label_comparison" not in result
    assert "delta_delta_g_kcal_mol" not in result
    assert "absolute_population_excess_percent" not in result
    assert "ee_percent" not in result
    assert result["population_sum"] == pytest.approx(1.0)


def test_enantiomeric_comparison_requires_exactly_two_explicit_labels():
    with pytest.raises(ValueError, match="requires exactly two"):
        electronic.execute(
            "analyze_free_energy_selectivity",
            "internal_thermochemistry",
            _request(
                [
                    {"label": "R", "gibbs_free_energy": 0.0},
                    {"label": "S", "gibbs_free_energy": 0.5},
                    {"label": "unassigned", "gibbs_free_energy": 1.0},
                ],
                comparison_type="enantiomeric",
            ),
        )


@pytest.mark.parametrize(
    "records",
    [
        [
            {"label": "a", "conformer_id": "same", "gibbs_free_energy": 0.0},
            {"label": "a", "conformer_id": "same", "gibbs_free_energy": 0.5},
            {"label": "b", "conformer_id": "other", "gibbs_free_energy": 1.0},
        ],
        [
            {"label": "a", "conformer_id": "same", "gibbs_free_energy": 0.0},
            {"label": "b", "conformer_id": "same", "gibbs_free_energy": 1.0},
        ],
    ],
)
def test_explicit_conformer_ids_are_globally_unique(records):
    with pytest.raises(ValueError, match="must be globally unique"):
        electronic.execute(
            "analyze_free_energy_selectivity",
            "internal_thermochemistry",
            _request(records),
        )


def test_records_without_conformer_ids_remain_distinct_microstates():
    result = electronic.execute(
        "analyze_free_energy_selectivity",
        "internal_thermochemistry",
        _request(
            [
                {"label": "a", "gibbs_free_energy": 0.0},
                {"label": "a", "gibbs_free_energy": 0.0},
                {"label": "b", "gibbs_free_energy": 1.0},
            ]
        ),
    )["result"]

    assert result["labels"][0]["conformer_count"] == 2
    assert result["population_ratio_first_over_second"] == pytest.approx(
        2.0 * math.exp(1.0 / (GAS_CONSTANT_KCAL_MOL_K * 303.15))
    )


def test_free_energy_artifact_handoff_rejects_wrong_semantic_type():
    artifact = {
        "artifact_id": "art_00000000000000000000000000000000",
        "semantic_type": "SpectrumResult",
    }
    invalid = _validate_artifact_input_semantics(
        "analyze_free_energy_selectivity",
        "internal_thermochemistry",
        {"free_energies": artifact},
    )
    assert invalid is not None
    assert invalid["error"]["code"] == "artifact_semantic_mismatch"

    valid = _validate_artifact_input_semantics(
        "analyze_free_energy_selectivity",
        "internal_thermochemistry",
        {
            "free_energies": {
                **artifact,
                "semantic_type": "LabeledFreeEnergyCollection",
            }
        },
    )
    assert valid is None


def test_subnormal_positive_temperature_is_an_invalid_request_not_backend_exception(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    response = execute_action(
        "analyze_free_energy_selectivity",
        {
            "inputs": {
                "free_energies": [
                    {"label": "a", "gibbs_free_energy": 0.0},
                    {"label": "b", "gibbs_free_energy": 1.0},
                ],
                "energy_unit": "kcal_mol",
            },
            "method_spec": {},
            "action_settings": {
                "temperature_kelvin": 5e-324,
                "comparison_type": "population",
            },
        },
    )

    assert response["status"] == "invalid_request"
    assert response["error"]["code"] == "backend_input_error"
    assert "finite positive RT" in response["error"]["message"]


@pytest.mark.parametrize(
    ("records", "unit", "temperature", "message"),
    [
        (
            [
                {"label": "a", "gibbs_free_energy": 0.0},
                {"label": "b", "gibbs_free_energy": float("nan")},
            ],
            "kcal_mol",
            298.15,
            "must be finite",
        ),
        (
            [
                {"label": "a", "gibbs_free_energy": True},
                {"label": "b", "gibbs_free_energy": 1.0},
            ],
            "kcal_mol",
            298.15,
            "must be a JSON number",
        ),
        (
            [
                {"label": "a", "gibbs_free_energy": 0.0},
                {"label": "a", "gibbs_free_energy": 1.0},
            ],
            "kcal_mol",
            298.15,
            "at least two distinct labels",
        ),
        (
            [
                {"label": "a", "gibbs_free_energy": 0.0},
                {"label": "b", "gibbs_free_energy": 1.0, "unit": "eV"},
            ],
            "kcal_mol",
            298.15,
            "conflicts with inputs.energy_unit",
        ),
        (
            [
                {"label": "a", "gibbs_free_energy": 0.0},
                {"label": "b", "gibbs_free_energy": 1.0},
            ],
            "calories",
            298.15,
            "must be one of",
        ),
        (
            [
                {"label": "a", "gibbs_free_energy": 0.0},
                {"label": "b", "gibbs_free_energy": 1.0},
            ],
            "kcal_mol",
            0.0,
            "finite positive",
        ),
    ],
)
def test_invalid_free_energy_requests_are_rejected(records, unit, temperature, message):
    with pytest.raises(ValueError, match=message):
        electronic.execute(
            "analyze_free_energy_selectivity",
            "internal_thermochemistry",
            _request(records, unit=unit, temperature=temperature),
        )


def test_unrepresentable_two_label_ratio_is_rejected_instead_of_returning_nonfinite():
    with pytest.raises(ValueError, match="population ratio cannot be represented"):
        electronic.execute(
            "analyze_free_energy_selectivity",
            "internal_thermochemistry",
            _request(
                [
                    {"label": "first", "gibbs_free_energy": 0.0},
                    {"label": "second", "gibbs_free_energy": 1000.0},
                ]
            ),
        )


def test_existing_internal_thermochemistry_still_completes():
    response = electronic.execute(
        "derive_thermochemistry",
        "internal_thermochemistry",
        {
            "inputs": {
                "energy": {"energy": -1.0, "unit": "eV"},
                "frequencies": {"frequencies_cm1": [4400.0]},
                "structure": {
                    "atoms": [
                        {"element": "H", "position_angstrom": [0.0, 0.0, 0.0]},
                        {"element": "H", "position_angstrom": [0.74, 0.0, 0.0]},
                    ],
                    "charge": 0,
                    "multiplicity": 1,
                },
            },
            "method_spec": {},
            "action_settings": {
                "temperature_kelvin": 298.15,
                "pressure_pa": 101325.0,
                "geometry": "linear",
                "symmetry_number": 2,
                "spin": 0,
                "ignore_imaginary_modes": False,
            },
        },
    )

    assert response["status"] == "success"
    assert math.isfinite(response["result"]["gibbs_free_energy_ev"])
