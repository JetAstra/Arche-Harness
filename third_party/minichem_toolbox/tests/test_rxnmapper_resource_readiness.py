from __future__ import annotations

import json
import sys
from pathlib import Path

from minichem_toolbox import runtime
from minichem_toolbox.backends import reaction_mapping
from minichem_toolbox.backend_specs import BACKEND_SPECS
from minichem_toolbox.service import execute_action
import minichem_toolbox.service as service_module


_RESOURCE_SPEC = {
    "resource_id": reaction_mapping._RXNMAPPER_RESOURCE_ID,
    "module": "rxnmapper",
    "relative_path": "models/transformers/albert_heads_8_uspto_all_1310k",
    "path_environment_variable": "MINICHEM_RXNMAPPER_MODEL_PATH",
    "required_files": list(reaction_mapping._RXNMAPPER_REQUIRED_FILES),
}


def _use_standard_python(monkeypatch):
    monkeypatch.setattr(runtime, "runtime_python", lambda _runtime: Path(sys.executable))


def test_package_checkpoint_is_available_without_project_path_override(monkeypatch):
    _use_standard_python(monkeypatch)
    monkeypatch.delenv("MINICHEM_RXNMAPPER_MODEL_PATH", raising=False)

    status = runtime._package_resource_probe("rxnmapper", [_RESOURCE_SPEC])[
        reaction_mapping._RXNMAPPER_RESOURCE_ID
    ]

    assert status["available"] is True
    assert status["reason_code"] == "ok"
    assert status["source"] == "package"


def test_configured_missing_or_incomplete_checkpoint_is_reported_without_path(
    monkeypatch, tmp_path
):
    _use_standard_python(monkeypatch)
    configured = tmp_path / "missing-rxnmapper-model"
    monkeypatch.setenv("MINICHEM_RXNMAPPER_MODEL_PATH", str(configured))

    missing = runtime._package_resource_probe("rxnmapper", [_RESOURCE_SPEC])[
        reaction_mapping._RXNMAPPER_RESOURCE_ID
    ]
    assert missing["available"] is False
    assert missing["reason_code"] == "resource_missing"
    assert str(configured) not in json.dumps(missing)

    configured.mkdir()
    (configured / "config.json").write_text("{}", encoding="utf-8")
    incomplete = runtime._package_resource_probe("rxnmapper", [_RESOURCE_SPEC])[
        reaction_mapping._RXNMAPPER_RESOURCE_ID
    ]
    assert incomplete["available"] is False
    assert incomplete["reason_code"] == "resource_incomplete"
    assert incomplete["missing_files"]
    assert str(configured) not in json.dumps(incomplete)


def test_probe_all_backends_exposes_stable_resource_reason_without_path_leak(
    monkeypatch, tmp_path
):
    _use_standard_python(monkeypatch)
    configured = tmp_path / "not-present"
    monkeypatch.setenv("MINICHEM_RXNMAPPER_MODEL_PATH", str(configured))
    rxnmapper_spec = next(item for item in BACKEND_SPECS if item.id == "rxnmapper")

    health = runtime.probe_all_backends((rxnmapper_spec,))["rxnmapper"]

    assert health["available"] is False
    assert health["reason_code"] == "resource_missing"
    assert health["missing_resources"] == [reaction_mapping._RXNMAPPER_RESOURCE_ID]
    assert str(configured) not in json.dumps(health)
    resource = health["resources"][reaction_mapping._RXNMAPPER_RESOURCE_ID]
    assert resource["resource_id"] == reaction_mapping._RXNMAPPER_RESOURCE_ID
    assert resource["reason_code"] == "resource_missing"


def test_mapper_entry_short_circuits_when_resource_disappears(monkeypatch, tmp_path):
    configured = tmp_path / "removed-rxnmapper-model"
    monkeypatch.setenv("MINICHEM_RXNMAPPER_MODEL_PATH", str(configured))
    calls = {"constructor": 0}

    class MapperMustNotStart:
        def __init__(self, *_args):
            calls["constructor"] += 1
            raise AssertionError("mapper must not start without its model resource")

    monkeypatch.setattr(reaction_mapping, "_load_rxnmapper_class", lambda: MapperMustNotStart)

    response = reaction_mapping.map_reaction_atoms(
        {
            "inputs": {"reaction": "CBr.[OH-]>>CO.[Br-]"},
            "method_spec": {},
            "action_settings": {"minimum_confidence": 0.0},
        }
    )

    assert response["status"] == "unavailable"
    assert response["retryable"] is False
    assert response["error"]["code"] == "backend_unavailable"
    assert response["error"]["reason_code"] == "resource_missing"
    assert response["error"]["resource_id"] == reaction_mapping._RXNMAPPER_RESOURCE_ID
    assert calls["constructor"] == 0
    assert str(configured) not in json.dumps(response)


def test_service_health_unavailable_short_circuits_worker(monkeypatch):
    calls = {"worker": 0}

    def unavailable_probe(specifications):
        return {
            item.id: {
                "status": "unavailable",
                "available": False,
                "runtime": item.runtime,
                "reason_code": "resource_missing",
                "resource_id": reaction_mapping._RXNMAPPER_RESOURCE_ID,
            }
            for item in specifications
        }

    monkeypatch.setattr(service_module, "probe_all_backends", unavailable_probe)
    monkeypatch.setattr(
        service_module,
        "invoke_worker",
        lambda **_kwargs: calls.__setitem__("worker", calls["worker"] + 1),
    )

    response = execute_action(
        "map_reaction_atoms",
        {
            "backend_id": "rxnmapper",
            "inputs": {"reaction": "CBr.[OH-]>>CO.[Br-]"},
            "method_spec": {},
            "action_settings": {"minimum_confidence": 0.0},
        },
    )

    assert response["status"] == "unavailable"
    assert response["retryable"] is False
    assert calls["worker"] == 0
