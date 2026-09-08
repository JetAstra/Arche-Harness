from __future__ import annotations

import json
from pathlib import Path

import pytest

from minichem_toolbox.backends import plotting
from minichem_toolbox.models import ActionRequest
from minichem_toolbox.service import execute_action


def _spectrum() -> dict:
    return {
        "schema_name": "SpectrumResult",
        "schema_version": "1.0",
        "spectrum_type": "VCD",
        "axis": {
            "name": "wavenumber",
            "unit": "cm^-1",
            "values": [900.0, 1000.0, 1100.0],
        },
        "intensity": {
            "name": "spectral_intensity",
            "unit": "custom_signed_unit",
            "values": [-1.0, 0.5, 2.0],
            "signed": True,
        },
        "sticks": [
            {"position": 900.0, "intensity": -1.0},
            {"position": 1100.0, "intensity": 2.0},
        ],
        "point_count": 3,
    }


def _request(spectrum: dict | None = None) -> dict:
    return {
        "inputs": {"spectrum": spectrum or _spectrum()},
        "method_spec": {},
        "action_settings": {
            "output_format": "svg",
            "width_inches": 10.0,
            "height_inches": 6.0,
            "dpi": 100,
            "reverse_x_axis": True,
        },
    }


def test_svg_plot_is_unique_confined_and_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    request = _request()
    ActionRequest.model_validate(request)

    first = plotting.execute("render_spectrum_plot", "internal_plotting", request)
    second = plotting.execute("render_spectrum_plot", "internal_plotting", request)

    assert first["status"] == "success"
    first_result = first["result"]
    second_result = second["result"]
    assert first_result["schema_name"] == "PlotArtifact"
    assert first_result["format"] == "svg"
    assert first_result["media_type"] == "image/svg+xml"
    assert first_result["point_count"] == 3
    assert first_result["axis_unit"] == "cm^-1"
    assert first_result["intensity_unit"] == "custom_signed_unit"
    assert first_result["path"] != second_result["path"]
    assert first["artifact_files"] == [
        {
            "path": first_result["path"],
            "semantic_type": "SpectrumPlotFile",
            "media_type": "image/svg+xml",
        }
    ]
    json.dumps(first_result)

    first_path = tmp_path / first_result["path"]
    second_path = tmp_path / second_result["path"]
    expected_root = tmp_path / "outputs" / "render_spectrum_plot" / "internal_plotting"
    assert first_path.parent.parent == expected_root
    assert second_path.parent.parent == expected_root
    assert first_path.parent != second_path.parent
    assert first_path.is_file() and second_path.is_file()
    assert list(first_path.parent.iterdir()) == [first_path]
    assert list(second_path.parent.iterdir()) == [second_path]
    assert not (tmp_path / "_tool_artifacts").exists()

    svg = first_path.read_text(encoding="utf-8")
    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
    assert "VCD Spectrum" in svg
    assert "custom_signed_unit" in svg
    assert 'media_type="image/svg+xml"' not in svg


def test_backend_envelope_can_be_used_as_spectrum_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    request = _request(
        {"status": "success", "result": _spectrum(), "warnings": []},
    )

    response = plotting.execute("render_spectrum_plot", "internal_plotting", request)

    assert response["status"] == "success"
    assert (tmp_path / response["result"]["path"]).is_file()


def test_service_handoff_from_ir_spectrum_artifact_to_svg_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    spectrum = execute_action(
        "derive_ir_spectrum",
        {
            "inputs": {
                "vibrations": {
                    "properties": {
                        "vibfreqs": [1000.0, 1200.0],
                        "vibirs": [8.0, 15.0],
                    },
                    "attribute_units": {
                        "vibfreqs": "inverse_centimeter",
                        "vibirs": "kilometer_per_mole",
                    },
                }
            },
            "action_settings": {
                "broadening": "gaussian",
                "fwhm_cm1": 20.0,
            },
        },
    )
    assert spectrum["status"] == "success"
    assert spectrum["result"]["point_count"] == 4000

    plot = execute_action(
        "render_spectrum_plot",
        {
            "inputs": {"spectrum": spectrum["output_artifacts"][0]},
            "action_settings": {
                "output_format": "svg",
                "width_inches": 10.0,
                "height_inches": 6.0,
                "dpi": 100,
                "reverse_x_axis": True,
            },
        },
    )

    assert plot["status"] == "success"
    assert (tmp_path / plot["result"]["path"]).is_file()
    assert plot["output_artifacts"][0]["parent_artifact_ids"] == [
        spectrum["output_artifacts"][0]["artifact_id"]
    ]


def test_invalid_plot_request_does_not_create_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    invalid = _spectrum()
    invalid["intensity"]["values"] = [1.0]

    with pytest.raises(ValueError, match="at least two"):
        plotting.execute("render_spectrum_plot", "internal_plotting", _request(invalid))
    assert not (tmp_path / "outputs").exists()

    request = _request()
    request["action_settings"]["output_format"] = "png"
    with pytest.raises(ValueError, match="format='svg'"):
        plotting.execute("render_spectrum_plot", "internal_plotting", request)
    assert not (tmp_path / "outputs").exists()


def test_missing_required_settings_and_unknown_dispatch_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    request = _request()
    request["action_settings"].pop("output_format")
    with pytest.raises(ValueError, match="output_format"):
        plotting.execute("render_spectrum_plot", "internal_plotting", request)

    valid = _request()
    assert (
        plotting.execute("unknown", "internal_plotting", valid)["status"]
        == "unsupported"
    )
    assert (
        plotting.execute("render_spectrum_plot", "other", valid)["status"]
        == "unsupported"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("schema_name"), "Expected SpectrumResult"),
        (lambda value: value.update(point_count=4), "point_count"),
        (lambda value: value["axis"].update(unit=""), "unit"),
    ],
)
def test_spectrum_contract_mismatches_fail_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    spectrum = _spectrum()
    mutation(spectrum)

    with pytest.raises(ValueError, match=message):
        plotting.execute("render_spectrum_plot", "internal_plotting", _request(spectrum))
    assert not (tmp_path / "outputs").exists()


def test_explicit_zero_width_axis_range_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    request = _request()
    request["action_settings"].update(x_min=1000.0, x_max=1000.0)

    with pytest.raises(ValueError, match="x_min must be smaller"):
        plotting.execute("render_spectrum_plot", "internal_plotting", request)
