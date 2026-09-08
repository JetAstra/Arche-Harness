from __future__ import annotations

import json

import pytest

from minichem_toolbox.backends import spectroscopy


def _request(input_key: str, payload: dict, **settings: object) -> dict:
    return {
        "inputs": {input_key: payload},
        "method_spec": {},
        "action_settings": {
            "fwhm": 20.0,
            "grid_min": 900.0,
            "grid_max": 1300.0,
            "grid_points": 81,
            **settings,
        },
    }


def test_ir_unwraps_canonical_vibrations_and_cclib_aliases() -> None:
    parsed = {
        "status": "success",
        "result": {
            "properties": {"vibfreqs": [1000.0, 1200.0], "vibirs": [8.0, 15.0]},
            "attribute_units": {
                "vibfreqs": "inverse_centimeter",
                "vibirs": "kilometer_per_mole",
            },
        },
    }
    response = spectroscopy.execute(
        "derive_ir_spectrum",
        "internal_spectroscopy",
        {
            "inputs": {"vibrations": parsed},
            "action_settings": {
                "broadening": "gaussian",
                "fwhm_cm1": 20.0,
                "min_wavenumber_cm1": 900.0,
                "max_wavenumber_cm1": 1300.0,
                "points": 81,
            },
        },
    )

    assert response["status"] == "success"
    result = response["result"]
    assert result["schema_name"] == "SpectrumResult"
    assert result["spectrum_type"] == "IR"
    assert result["axis"]["unit"] == "inverse_centimeter"
    assert result["intensity"]["unit"] == "kilometer_per_mole"
    assert result["source_fields"] == {"positions": "vibfreqs", "intensities": "vibirs"}
    assert result["sticks"] == [
        {"position": 1000.0, "intensity": 8.0},
        {"position": 1200.0, "intensity": 15.0},
    ]
    json.dumps(result)


@pytest.mark.parametrize(
    ("action_id", "payload", "expected_type", "expected_unit"),
    [
        (
            "derive_raman_spectrum",
            {
                "positions": [1000.0],
                "raman_activities": [4.5],
                "position_unit": "cm^-1",
                "intensity_unit": "A^4/amu",
            },
            "Raman",
            "A^4/amu",
        ),
        (
            "derive_roa_spectrum",
            {
                "positions": [1000.0],
                "roa_activities": [-2.5],
                "position_unit": "cm^-1",
                "intensity_unit": "custom_roa_unit",
            },
            "ROA",
            "custom_roa_unit",
        ),
        (
            "derive_vcd_spectrum",
            {
                "positions": [1000.0],
                "vcd_rotatory_strengths": [-3.5],
                "position_unit": "cm^-1",
                "intensity_unit": "custom_vcd_unit",
            },
            "VCD",
            "custom_vcd_unit",
        ),
    ],
)
def test_spectral_data_modalities_preserve_units_and_sign(
    action_id: str, payload: dict, expected_type: str, expected_unit: str
) -> None:
    response = spectroscopy.execute(
        action_id,
        "internal_spectroscopy",
        _request(
            "spectral_data",
            payload,
            grid_min=990.0,
            grid_max=1010.0,
            grid_points=3,
        ),
    )

    result = response["result"]
    assert result["spectrum_type"] == expected_type
    assert result["intensity"]["unit"] == expected_unit
    assert (
        result["sticks"][0]["intensity"]
        == payload[
            next(
                key
                for key in (
                    "raman_activities",
                    "roa_activities",
                    "vcd_rotatory_strengths",
                )
                if key in payload
            )
        ][0]
    )
    if result["sticks"][0]["intensity"] < 0:
        assert result["intensity"]["signed"] is True
        assert result["intensity"]["values"][1] < 0.0


def test_ecd_and_uv_vis_use_distinct_cclib_intensities() -> None:
    excited_states = {
        "properties": {
            "etenergies": [25000.0, 30000.0],
            "etrotats": [-0.2, 0.4],
            "etoscs": [0.1, 0.3],
        },
        "attribute_units": {"etenergies": "inverse_centimeter"},
    }
    settings = {
        "fwhm": 500.0,
        "grid_min": 20000.0,
        "grid_max": 35000.0,
        "grid_points": 31,
    }

    ecd = spectroscopy.execute(
        "derive_ecd_spectrum",
        "internal_spectroscopy",
        {"inputs": {"excited_states": excited_states}, "action_settings": settings},
    )["result"]
    uv_vis = spectroscopy.execute(
        "derive_uv_vis_spectrum",
        "internal_spectroscopy",
        {"inputs": {"excited_states": excited_states}, "action_settings": settings},
    )["result"]

    assert ecd["source_fields"]["intensities"] == "etrotats"
    assert ecd["sticks"][0]["intensity"] == -0.2
    assert ecd["intensity"]["signed"] is True
    assert uv_vis["source_fields"]["intensities"] == "etoscs"
    assert uv_vis["sticks"][0]["intensity"] == 0.1
    assert uv_vis["intensity"]["unit"] == "dimensionless_oscillator_strength"
    assert uv_vis["axis"]["unit"] == "inverse_centimeter"


def test_uv_vis_accepts_canonical_state_records_and_ev_settings() -> None:
    response = spectroscopy.execute(
        "derive_uv_vis_spectrum",
        "internal_spectroscopy",
        {
            "inputs": {
                "excited_states": {
                    "states": [
                        {"energy_ev": 2.5, "oscillator_strength": 0.1},
                        {"energy_ev": 3.2, "oscillator_strength": 0.4},
                    ]
                }
            },
            "action_settings": {
                "broadening": "lorentzian",
                "fwhm_ev": 0.2,
                "minimum_energy_ev": 2.0,
                "maximum_energy_ev": 4.0,
                "grid_points": 21,
            },
        },
    )

    result = response["result"]
    assert result["axis"]["unit"] == "electron_volt"
    assert result["axis"]["values"][0] == 2.0
    assert result["axis"]["values"][-1] == 4.0
    assert result["line_shape"] == {
        "kind": "lorentzian",
        "fwhm": 0.2,
        "unit": "electron_volt",
        "normalization": "peak_height",
    }
    assert result["source_fields"] == {
        "positions": "energy_ev",
        "intensities": "oscillator_strength",
    }


def test_uv_vis_ev_settings_convert_for_cclib_inverse_centimeter_axis() -> None:
    response = spectroscopy.execute(
        "derive_uv_vis_spectrum",
        "internal_spectroscopy",
        {
            "inputs": {
                "excited_states": {
                    "properties": {"etenergies": [25000.0], "etoscs": [0.2]},
                    "attribute_units": {"etenergies": "inverse_centimeter"},
                }
            },
            "action_settings": {
                "broadening": "gaussian",
                "fwhm_ev": 0.1,
                "minimum_energy_ev": 2.0,
                "maximum_energy_ev": 4.0,
                "grid_points": 11,
            },
        },
    )

    result = response["result"]
    assert result["axis"]["unit"] == "inverse_centimeter"
    assert result["axis"]["values"][0] == pytest.approx(2.0 * 8065.544005)
    assert result["axis"]["values"][-1] == pytest.approx(4.0 * 8065.544005)
    assert result["line_shape"]["fwhm"] == pytest.approx(0.1 * 8065.544005)


def test_ecd_rejects_oscillator_strength_as_rotatory_strength() -> None:
    with pytest.raises(ValueError, match="not a valid substitute"):
        spectroscopy.execute(
            "derive_ecd_spectrum",
            "internal_spectroscopy",
            {
                "inputs": {
                    "excited_states": {
                        "properties": {
                            "etenergies": [25000.0],
                            "etoscs": [0.1],
                        }
                    }
                },
                "action_settings": {"fwhm": 500.0},
            },
        )


def test_nmr_converts_tensors_only_with_explicit_reference() -> None:
    tensors = {
        "properties": {
            "atomnos": [6, 1, 6],
            "nmrtensors": {
                "0": {
                    "isotropic": 33.0,
                    "total": [[30.0, 0.0, 0.0], [0.0, 33.0, 0.0], [0.0, 0.0, 36.0]],
                },
                "1": {"isotropic": 39.0},
                "2": {
                    "isotropic": 23.0,
                    "total": [[20.0, 0.0, 0.0], [0.0, 23.0, 0.0], [0.0, 0.0, 26.0]],
                },
            }
        }
    }
    request = {
        "inputs": {"spectral_data": tensors},
        "action_settings": {
            "reference_shielding": 40.0,
            "nucleus_atomic_number": 6,
            "fwhm": 0.2,
            "grid_min": 0.0,
            "grid_max": 25.0,
            "grid_points": 101,
        },
    }
    response = spectroscopy.execute(
        "derive_nmr_spectrum", "internal_spectroscopy", request
    )

    result = response["result"]
    assert [stick["position"] for stick in result["sticks"]] == [7.0, 17.0]
    assert [stick["label"] for stick in result["sticks"]] == ["0", "2"]
    assert result["nucleus_atomic_number"] == 6
    assert all(stick["intensity"] == 1.0 for stick in result["sticks"])
    assert "unit weight" in response["warnings"][0]

    del request["action_settings"]["reference_shielding"]
    with pytest.raises(ValueError, match="reference_shielding"):
        spectroscopy.execute("derive_nmr_spectrum", "internal_spectroscopy", request)


def test_nmr_requires_atom_numbers_and_explicit_nucleus_selection() -> None:
    request = {
        "inputs": {
            "spectral_data": {
                "properties": {
                    "nmrtensors": {"0": {"isotropic": 30.0}},
                }
            }
        },
        "action_settings": {
            "reference_shielding": 40.0,
            "fwhm": 0.2,
            "nucleus_atomic_number": 1,
        },
    }
    with pytest.raises(ValueError, match="requires atomnos"):
        spectroscopy.execute("derive_nmr_spectrum", "internal_spectroscopy", request)

    request["inputs"]["spectral_data"]["properties"]["atomnos"] = [1]
    request["action_settings"].pop("nucleus_atomic_number")
    with pytest.raises(ValueError, match="nucleus_atomic_number"):
        spectroscopy.execute("derive_nmr_spectrum", "internal_spectroscopy", request)


def test_broadening_is_deterministic_and_peak_normalized() -> None:
    request = _request(
        "spectral_data",
        {
            "positions": [1000.0],
            "intensities": [2.0],
            "position_unit": "cm^-1",
            "intensity_unit": "arbitrary_unit",
        },
        grid_min=900.0,
        grid_max=1100.0,
        grid_points=3,
        line_shape="gaussian",
    )
    first = spectroscopy.execute(
        "derive_raman_spectrum", "internal_spectroscopy", request
    )
    second = spectroscopy.execute(
        "derive_raman_spectrum", "internal_spectroscopy", request
    )

    assert first == second
    assert first["result"]["axis"]["values"] == [900.0, 1000.0, 1100.0]
    assert first["result"]["intensity"]["values"][1] == pytest.approx(2.0)
    assert first["result"]["line_shape"]["normalization"] == "peak_height"


def test_invalid_spectral_arrays_and_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="same length"):
        spectroscopy.execute(
            "derive_ir_spectrum",
            "internal_spectroscopy",
            _request(
                "vibrations",
                {"frequencies": [1000.0, 1100.0], "ir_intensities": [2.0]},
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        spectroscopy.execute(
            "derive_ir_spectrum",
            "internal_spectroscopy",
            _request(
                "vibrations",
                {"frequencies": [float("nan")], "ir_intensities": [2.0]},
            ),
        )
    with pytest.raises(ValueError, match="fwhm"):
        spectroscopy.execute(
            "derive_ir_spectrum",
            "internal_spectroscopy",
            {
                "inputs": {
                    "vibrations": {"frequencies": [1000.0], "ir_intensities": [2.0]}
                }
            },
        )


def test_ambiguous_units_and_conflicting_aliases_fail_closed() -> None:
    with pytest.raises(ValueError, match="ambiguous spectral field"):
        spectroscopy.execute(
            "derive_raman_spectrum",
            "internal_spectroscopy",
            _request(
                "spectral_data",
                {"positions": [1000.0], "raman_activities": [2.0]},
            ),
        )

    with pytest.raises(ValueError, match="Conflicting spectral aliases"):
        spectroscopy.execute(
            "derive_ir_spectrum",
            "internal_spectroscopy",
            _request(
                "vibrations",
                {
                    "frequencies": [1000.0],
                    "vibfreqs": [1100.0],
                    "ir_intensities": [2.0],
                },
            ),
        )


def test_unknown_backend_and_action_are_explicitly_unsupported() -> None:
    request = _request("vibrations", {"frequencies": [1000.0], "ir_intensities": [2.0]})
    assert (
        spectroscopy.execute("derive_ir_spectrum", "other", request)["status"]
        == "unsupported"
    )
    assert (
        spectroscopy.execute("unknown", "internal_spectroscopy", request)["status"]
        == "unsupported"
    )
