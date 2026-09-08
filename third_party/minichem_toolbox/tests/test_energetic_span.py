from __future__ import annotations

import math

import pytest

from minichem_toolbox.backends import electronic, execute_local
from minichem_toolbox.catalog import action_specs, backend_specs, validate_catalog
from minichem_toolbox.discovery import inspect_action
from minichem_toolbox.service import _validate_artifact_input_semantics, execute_action


GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083
BOLTZMANN_CONSTANT_JOULE_K = 1.380649e-23
PLANCK_CONSTANT_JOULE_SECOND = 6.62607015e-34


CYCLE = [
    {"id": "I0", "state_type": "intermediate", "gibbs_free_energy": 0.0},
    {"id": "TS1", "state_type": "transition_state", "gibbs_free_energy": 10.0},
    {"id": "I1", "state_type": "intermediate", "gibbs_free_energy": -5.0},
    {"id": "TS2", "state_type": "transition_state", "gibbs_free_energy": 8.0},
]


def _request(
    states=CYCLE,
    *,
    unit="kcal_mol",
    overall=-4.0,
    temperature=300.0,
):
    return {
        "inputs": {
            "cycle_states": states,
            "energy_unit": unit,
            "overall_reaction_free_energy": overall,
        },
        "method_spec": {},
        "action_settings": {"temperature_kelvin": temperature},
    }


def _required(contract, section):
    return {item["name"] for item in contract["required_contract"][section]}


def test_catalog_discovery_and_local_dispatch_register_energetic_span():
    validate_catalog()
    action = action_specs()["analyze_energetic_span"]
    backend = backend_specs()["internal_thermochemistry"]

    assert action.backend_ids == ("internal_thermochemistry",)
    assert action.selection_policy == "internal_deterministic"
    assert action.id in backend.capabilities
    assert action.id in electronic.ACTIONS

    contract = inspect_action(
        action.id,
        backend_id="internal_thermochemistry",
        detail_level="contract",
    )["selected_request_contract"]
    assert _required(contract, "inputs") == {
        "cycle_states",
        "energy_unit",
        "overall_reaction_free_energy",
    }
    assert _required(contract, "action_settings") == {"temperature_kelvin"}
    fields = {
        item["name"]: item for item in contract["required_contract"]["inputs"]
    }
    assert fields["cycle_states"]["item_fields"] == [
        "id",
        "state_type",
        "gibbs_free_energy",
    ]
    assert fields["energy_unit"]["allowed_values"] == [
        "hartree",
        "eV",
        "kcal_mol",
        "kJ_mol",
    ]

    response = execute_local(
        "analyze_energetic_span", "internal_thermochemistry", _request()
    )
    assert response["status"] == "success"


def test_two_step_cycle_matches_hand_calculated_full_denominator_and_span():
    response = electronic.execute(
        "analyze_energetic_span", "internal_thermochemistry", _request()
    )
    assert response["status"] == "success"
    result = response["result"]

    # Ordered pairs are TS1/I0=10, TS1/I1=11 (wrap, +DeltaGr),
    # TS2/I0=8, TS2/I1=13.  Therefore TDTS=TS2, TDI=I1, deltaE=13.
    expected_spans = {
        ("TS1", "I0"): 10.0,
        ("TS1", "I1"): 11.0,
        ("TS2", "I0"): 8.0,
        ("TS2", "I1"): 13.0,
    }
    actual_spans = {
        (pair["transition_state_id"], pair["intermediate_id"]): pair[
            "span_kcal_mol"
        ]
        for pair in result["pair_spans"]
    }
    assert actual_spans == expected_spans
    assert result["energetic_span_kcal_mol"] == 13.0
    assert result["turnover_determining_intermediate_id"] == "I1"
    assert result["turnover_determining_transition_state_id"] == "TS2"
    assert result["determining_pair_count"] == 1

    temperature = 300.0
    rt = GAS_CONSTANT_KCAL_MOL_K * temperature
    denominator = math.fsum(math.exp(span / rt) for span in expected_spans.values())
    expected_tof = (
        BOLTZMANN_CONSTANT_JOULE_K
        * temperature
        / PLANCK_CONSTANT_JOULE_SECOND
        * (1.0 - math.exp(-4.0 / rt))
        / denominator
    )
    assert result["full_denominator_dimensionless"] == pytest.approx(
        denominator, rel=2e-14
    )
    assert result["turnover_frequency_per_second"] == pytest.approx(
        expected_tof, rel=2e-14
    )
    assert result["full_denominator_used"] is True
    assert result["state_types_or_order_inferred"] is False
    assert response["provenance"]["state_semantics_inferred"] is False


@pytest.mark.parametrize(
    ("unit", "scale"),
    [
        ("hartree", 1.0 / 627.5094740631),
        ("eV", 1.0 / 23.060547830619),
        ("kcal_mol", 1.0),
        ("kJ_mol", 4.184),
    ],
)
def test_all_supported_energy_units_are_equivalent(unit, scale):
    states = [
        {**state, "gibbs_free_energy": state["gibbs_free_energy"] * scale}
        for state in CYCLE
    ]
    result = electronic.execute(
        "analyze_energetic_span",
        "internal_thermochemistry",
        _request(states, unit=unit, overall=-4.0 * scale),
    )["result"]

    assert result["energetic_span_kcal_mol"] == pytest.approx(13.0, rel=3e-12)
    reference = electronic.execute(
        "analyze_energetic_span", "internal_thermochemistry", _request()
    )["result"]
    assert result["turnover_frequency_per_second"] == pytest.approx(
        reference["turnover_frequency_per_second"], rel=3e-12
    )


def test_service_executes_without_fake_backend_and_preserves_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    response = execute_action("analyze_energetic_span", _request())

    assert response["status"] == "success"
    assert response["backend"] == "internal_thermochemistry"
    assert response["selection_source"] == "internal_deterministic"
    assert response["result"]["energetic_span_kcal_mol"] == 13.0
    assert response["provenance"]["agent_selected_action"] == "analyze_energetic_span"
    assert response["provenance"]["cycle_order_basis"] == (
        "caller_supplied_cycle_states_order"
    )


def test_typed_cycle_profile_artifact_semantics_are_strict():
    base = {"artifact_id": "art_00000000000000000000000000000000"}
    invalid = _validate_artifact_input_semantics(
        "analyze_energetic_span",
        "internal_thermochemistry",
        {"cycle_states": {**base, "semantic_type": "ReactionPath"}},
    )
    assert invalid is not None
    assert invalid["error"]["code"] == "artifact_semantic_mismatch"

    valid = _validate_artifact_input_semantics(
        "analyze_energetic_span",
        "internal_thermochemistry",
        {
            "cycle_states": {
                **base,
                "semantic_type": "CatalyticCycleFreeEnergyProfile",
            }
        },
    )
    assert valid is None


@pytest.mark.parametrize(
    ("states", "overall", "temperature", "message"),
    [
        (
            [
                {"id": "I0", "state_type": "intermediate", "gibbs_free_energy": 0.0},
                {"id": "I1", "state_type": "intermediate", "gibbs_free_energy": 1.0},
            ],
            -1.0,
            300.0,
            "must be 'transition_state'",
        ),
        (CYCLE[:-1], -1.0, 300.0, "equal numbers"),
        (
            [CYCLE[0], {**CYCLE[1], "id": "I0"}],
            -1.0,
            300.0,
            "globally unique",
        ),
        (
            [CYCLE[0], {**CYCLE[1], "gibbs_free_energy": float("nan")}],
            -1.0,
            300.0,
            "must be finite",
        ),
        (
            [CYCLE[0], {**CYCLE[1], "unit": "eV"}],
            -1.0,
            300.0,
            "conflicts with inputs.energy_unit",
        ),
        (CYCLE, 0.0, 300.0, "must be negative"),
        (CYCLE, 1.0, 300.0, "must be negative"),
        (CYCLE, -1.0, 0.0, "finite positive"),
    ],
)
def test_invalid_cycle_inputs_are_rejected(states, overall, temperature, message):
    with pytest.raises(ValueError, match=message):
        electronic.execute(
            "analyze_energetic_span",
            "internal_thermochemistry",
            _request(states, overall=overall, temperature=temperature),
        )


def test_non_json_numbers_and_unknown_units_are_rejected():
    with pytest.raises(ValueError, match="must be a JSON number"):
        electronic.execute(
            "analyze_energetic_span",
            "internal_thermochemistry",
            _request(overall=True),
        )
    with pytest.raises(ValueError, match="must be one of"):
        electronic.execute(
            "analyze_energetic_span",
            "internal_thermochemistry",
            _request(unit="calories"),
        )


def test_subnormal_temperature_and_unrepresentable_tof_are_rejected():
    with pytest.raises(ValueError, match="finite positive RT"):
        electronic.execute(
            "analyze_energetic_span",
            "internal_thermochemistry",
            _request(temperature=5e-324),
        )

    huge_barrier = [
        {"id": "I0", "state_type": "intermediate", "gibbs_free_energy": 0.0},
        {"id": "TS1", "state_type": "transition_state", "gibbs_free_energy": 1000.0},
    ]
    with pytest.raises(ValueError, match="finite nonzero"):
        electronic.execute(
            "analyze_energetic_span",
            "internal_thermochemistry",
            _request(huge_barrier, overall=-1.0),
        )


def test_extreme_finite_temperature_keeps_tof_in_log_domain():
    response = electronic.execute(
        "analyze_energetic_span",
        "internal_thermochemistry",
        _request(temperature=1e300),
    )

    assert response["status"] == "success"
    result = response["result"]
    assert result["transition_state_theory_prefactor_per_second"] is None
    assert math.isfinite(
        result["log_transition_state_theory_prefactor_per_second"]
    )
    assert math.isfinite(result["turnover_frequency_per_second"])
    assert result["turnover_frequency_per_second"] > 0.0
