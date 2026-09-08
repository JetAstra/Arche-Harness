"""固定 Case 1 TSPipeline 的确定性 Action seam 测试。"""

from __future__ import annotations

import copy
import json
import random
import shutil

import pytest

from minichem_mcp_tools import discovery_tools, registry as mcp_registry
from minichem_mcp_tools.discovery_models import ProgressiveActionRequest
from minichem_mcp_tools.execution_models import (
    JobCollectRequest,
    ScientificArtifactTextReadRequest,
    WorkspaceTextReadRequest,
)
from minichem_mcp_tools import open_tools, transition_state_jobs
from minichem_toolbox import service
from minichem_toolbox.artifacts import ArtifactStore
from minichem_toolbox.resource_budget import ResourceBudget, ResourceBudgetExceeded
from minichem_toolbox.scope import ScopeContext, activate_scope
from minichem_toolbox.catalog import action_specs, backend_specs
from minichem_toolbox.backends import arche_ts_pipeline, reactive_complex, structure
from minichem_toolbox.models import ActionRequest


_PATH_INPUTS = {
    "reactants": [
        "C=C(C)N1CCCC1C[NH+]1CCCC1",
        "O=Cc1ccc([N+](=O)[O-])cc1",
    ],
    "products": [
        "C/C(C[C@@H](O)c1ccc([N+](=O)[O-])cc1)=[N+]1/CCCC1CN1CCCC1",
    ],
}


def _structure(
    *, distance: float = 1.0, second_element: str = "H", map_offset: int = 0
) -> dict:
    return {
        "smiles": "C[H]" if second_element == "H" else "CC",
        "atoms": [
            {
                "element": "C",
                "position_angstrom": [0.0, 0.0, 0.0],
                "atom_map_number": 1 + map_offset,
            },
            {
                "element": second_element,
                "position_angstrom": [distance, 0.0, 0.0],
                "atom_map_number": 2 + map_offset,
            },
        ],
        "bonds": [{"atom_index_a": 0, "atom_index_b": 1}],
        "charge": 0,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }


def _path_request(**overrides):
    request = {
        "inputs": copy.deepcopy(_PATH_INPUTS),
        "method_spec": {},
        "action_settings": {},
        "resource_limits": {"gpu_count": 1},
    }
    request.update(overrides)
    return request


def test_composite_action_dispatches_to_fixed_case1_orchestration(monkeypatch):
    expected = {"status": "success", "result": {"workflow_id": "test"}}
    monkeypatch.setattr(
        arche_ts_pipeline, "run_atomic_workflow", lambda request: expected
    )

    response = arche_ts_pipeline.execute(
        "search_transition_state_path",
        "transition_state_path_search",
        _path_request(),
    )

    assert response is expected


def test_path_action_rejects_hidden_overrides_and_cpu_only_schedule():
    with pytest.raises(ValueError, match="Unsupported inputs field"):
        arche_ts_pipeline.run_atomic_workflow(
            _path_request(inputs={"direction": "reverse"}),
            executor=lambda *_args: None,
        )
    with pytest.raises(ValueError, match="method_spec is server-managed"):
        arche_ts_pipeline.run_atomic_workflow(
            _path_request(method_spec={"model": "caller-controlled"}),
            executor=lambda *_args: None,
        )
    with pytest.raises(ValueError, match="controls are server-managed"):
        arche_ts_pipeline.run_atomic_workflow(
            _path_request(action_settings={"anything": 1}),
            executor=lambda *_args: None,
        )
    with pytest.raises(ValueError, match="gpu_count"):
        arche_ts_pipeline.run_atomic_workflow(
            _path_request(resource_limits={"gpu_count": 0}),
            executor=lambda *_args: None,
        )


def test_path_action_accepts_caller_reaction_and_only_registered_force_fields():
    caller_inputs = {
        "reactants": ["CCO", "O=C=O"],
        "products": ["CCOC(=O)O"],
    }
    reaction, main_force_field, target_force_field = (
        arche_ts_pipeline._validate_public_inputs(caller_inputs)
    )
    assert reaction == caller_inputs
    assert main_force_field == "MMFF"
    assert target_force_field == "UMA"

    for hidden_field, value in (
        ("direction", "reverse"),
        ("charge", -1),
        ("multiplicity", 2),
    ):
        with pytest.raises(ValueError, match="Unsupported inputs field"):
            arche_ts_pipeline._validate_public_inputs(
                {**caller_inputs, hidden_field: value}
            )
    with pytest.raises(ValueError, match="only 'MMFF'"):
        arche_ts_pipeline._validate_public_inputs(
            {**caller_inputs, "main_force_field": "UFF"}
        )
    with pytest.raises(ValueError, match="only 'UMA'"):
        arche_ts_pipeline._validate_public_inputs(
            {**caller_inputs, "target_force_field": "xTB"}
        )
    with pytest.raises(ValueError, match="at least two reactant components"):
        arche_ts_pipeline._validate_public_inputs(
            {"reactants": ["CCO"], "products": ["CC=O"]}
        )


def test_path_action_classifies_incompatible_hydrogen_transfer_as_input_error():
    """不满足固定单氢转移边界的反应应允许调用方改输入后重试。"""

    def executor(action_id, _backend_id, _request):
        if action_id == "map_reaction_atoms":
            return {
                "status": "success",
                "result": {
                    "reactants": ["CC", "O"],
                    "products": ["CCO"],
                },
            }
        if action_id == "analyze_reaction_changes":
            return {
                "status": "success",
                "result": {
                    "hydrogen_transfer_summary": {
                        "donors": [],
                        "acceptors": [],
                        "balanced": True,
                    }
                },
            }
        raise AssertionError(action_id)

    with pytest.raises(ValueError, match="balanced single-hydrogen transfer"):
        arche_ts_pipeline.run_atomic_workflow(
            _path_request(),
            executor=executor,
        )


def test_path_action_keeps_direction_charge_and_multiplicity_server_defaults():
    settings = arche_ts_pipeline._validate_settings({})
    assert settings["direction"] == "forward"
    assert settings["charge"] == 1
    assert settings["multiplicity"] == 1


def test_heavy_stages_keep_fixed_cuda_placement() -> None:
    pair = {
        "pair_id": "endpoint_pair_0001",
        "reactant": _structure(),
        "product": _structure(distance=1.1),
    }
    observed_devices: list[tuple[str, str]] = []

    def executor(action_id, backend_id, request):
        observed_devices.append((backend_id, request["method_spec"]["device"]))
        if action_id == "optimize_geometry":
            return {
                "status": "success",
                "result": {"structure": request["inputs"]["structure"], "converged": True},
            }
        if action_id == "search_reaction_path":
            return {"status": "success", "result": {"images": []}}
        raise AssertionError((action_id, backend_id))

    result, unavailable = arche_ts_pipeline._run_heavy_stages(
        [pair], {}, arche_ts_pipeline._validate_settings({}),
        {"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1}, executor, [],
    )

    assert unavailable is not None
    assert "incomplete reaction path" in unavailable
    assert result[0]["stage_statuses"] == {"uma": "success", "neb": "partial"}
    assert observed_devices == [
        ("fairchem_uma", "cuda"),
        ("fairchem_uma", "cuda"),
        ("ash_neb_uma", "cuda"),
    ]


def test_heavy_stage_incomplete_path_records_pair_failure_instead_of_empty_summary():
    """不完整路径不得静默成为成功 pair，失败摘要必须说明缺少的交接证据。"""

    failure_records: list[dict] = []

    def executor(action_id, backend_id, _request):
        if backend_id == "fairchem_uma":
            return {
                "status": "success",
                "result": {"structure": _structure(), "converged": True},
            }
        if action_id == "search_reaction_path":
            return {
                "status": "partial_success",
                "result": {
                    "images": [_structure(), _structure(distance=1.2)],
                    "final_path_complete": False,
                    "energy_index_complete": False,
                    "interior_peak_candidate_valid": False,
                },
            }
        raise AssertionError((action_id, backend_id))

    pairs, unavailable = arche_ts_pipeline._run_heavy_stages(
        [
            {
                "pair_id": "endpoint_pair_0001",
                "reactant": _structure(),
                "product": _structure(distance=1.1),
            }
        ],
        {},
        {
            "uma_model": "uma-s-1p1",
            "uma_task_name": "omol",
            "uma_fmax_ev_per_angstrom": 0.01,
            "uma_optimizer": "bfgs",
            "uma_max_steps": 500,
            "total_images": 9,
            "climb": False,
            "neb_max_steps": 300,
        },
        {"gpu_count": 1},
        executor,
        [],
        failure_records=failure_records,
    )

    assert unavailable is not None
    assert pairs[0]["stage_statuses"]["neb"] == "partial"
    assert failure_records == [
        {
            "pair_id": "endpoint_pair_0001",
            "stage": "neb",
            "status": "failed",
            "reason_code": "reaction_path_incomplete",
            "retryable": False,
            "message": unavailable,
        }
    ]


def test_path_action_rejects_reaction_with_fewer_than_two_components_after_mapping():
    """执行校验必须与公开的至少两个反应物组分边界一致。"""

    analysis_calls = 0

    def executor(action_id, _backend_id, _request):
        nonlocal analysis_calls
        if action_id == "map_reaction_atoms":
            return {
                "status": "success",
                "result": {"reactants": ["CCO"], "products": ["CC=O"]},
            }
        if action_id == "analyze_reaction_changes":
            analysis_calls += 1
            if analysis_calls == 1:
                return {
                    "status": "success",
                    "result": {
                        "hydrogen_transfer_summary": {
                            "donors": [{"atom_map": 1, "hydrogens_lost": 1}],
                            "acceptors": [{"atom_map": 2, "hydrogens_gained": 1}],
                            "balanced": True,
                        }
                    },
                }
            return {
                "status": "success",
                "result": {
                    "mapping_complete": True,
                    "unmapped_hydrogen_count": 0,
                    "reactants": ["CCO"],
                },
            }
        if action_id == "materialize_mapped_hydrogen_transfers":
            return {
                "status": "success",
                "result": {"reactants": ["CCO"]},
            }
        raise AssertionError(action_id)

    with pytest.raises(RuntimeError, match="at least two reactant components"):
        arche_ts_pipeline.run_atomic_workflow(
            _path_request(),
            executor=executor,
        )


def test_path_action_forwards_caller_reaction_to_mapping_stage():
    caller_inputs = {
        "reactants": ["CCO", "O=C=O"],
        "products": ["CCOC(=O)O"],
    }
    observed = []

    def unavailable_mapping(action_id, backend_id, request):
        observed.append((action_id, backend_id, copy.deepcopy(request["inputs"])))
        return {
            "status": "unavailable",
            "error": {"code": "backend_unavailable"},
        }

    response = arche_ts_pipeline.run_atomic_workflow(
        _path_request(inputs=caller_inputs), executor=unavailable_mapping
    )
    assert response["status"] == "unavailable"
    assert observed == [
        (
            "map_reaction_atoms",
            "rxnmapper",
            {"reaction": caller_inputs},
        )
    ]


def test_fixed_backend_declares_health_modules_and_minimum_gpu() -> None:
    """固定后端必须声明实际依赖模块，并把 GPU 下限放进通用资源合同。"""

    specification = backend_specs()["transition_state_path_search"]

    assert set(specification.python_modules) == {
        "rxnmapper",
        "rdkit",
        "ase",
        "numpy",
        "fairchem.core",
        "torch",
        "ash",
    }
    assert specification.resource_constraints["minimum_gpu_count"] == 1


def test_service_rejects_fixed_backend_without_minimum_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """固定资源策略忽略调用方的零 GPU，worker 只能看到服务端 GPU 预算。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "test-catalog-hash")
    captured_requests = []

    def fake_worker(**kwargs):
        captured_requests.append(copy.deepcopy(kwargs["payload"]["request"]))
        return {
            "status": "unavailable",
            "error": {"code": "backend_unavailable", "message": "test backend unavailable"},
            "retryable": True,
        }

    monkeypatch.setattr(service, "invoke_worker", fake_worker)
    response = service.execute_action(
        "search_transition_state_path",
        {
            "backend_id": "transition_state_path_search",
            "inputs": copy.deepcopy(_PATH_INPUTS),
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        },
    )

    assert response["status"] == "unavailable"
    assert response["error"]["code"] == "backend_unavailable"
    assert captured_requests
    assert captured_requests[0]["resource_limits"]["gpu_count"] == 1


def test_fixed_action_success_path_returns_only_public_handoff_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """用最小原子 Action 假实现贯通固定流程，并核对最终公开结果的边界。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    component_a = _structure()
    component_b = _structure(map_offset=10)
    complex_structure = {
        "atoms": [
            *component_a["atoms"],
            *component_b["atoms"],
        ],
        "bonds": [],
        "charge": 1,
        "multiplicity": 1,
        "pbc": [False, False, False],
    }
    # 让固定成键 map 之间的距离落在作者 MMFF 后验收阈值以内。
    complex_structure["atoms"][2]["position_angstrom"] = [1.0, 0.0, 0.0]
    generated_components = iter((component_a, component_b))

    def executor(action_id, backend_id, request):
        if action_id == "map_reaction_atoms":
            return {"status": "success", "result": {"mapped": True}}
        if action_id == "analyze_reaction_changes":
            if request["inputs"]["mapped_reaction"].get("mapped") is True:
                return {
                    "status": "success",
                    "result": {
                        "hydrogen_transfer_summary": {
                            "donors": [{"atom_map": 1, "hydrogens_lost": 1}],
                            "acceptors": [{"atom_map": 11, "hydrogens_gained": 1}],
                            "balanced": True,
                        }
                    },
                }
            return {
                "status": "success",
                "result": {
                    "mapping_complete": True,
                    "unmapped_hydrogen_count": 0,
                    "formed_bonds": [{"atom_map_a": 1, "atom_map_b": 11}],
                    "broken_bonds": [],
                },
            }
        if action_id == "materialize_mapped_hydrogen_transfers":
            return {
                "status": "success",
                "result": {"mapped": False, "reactants": ["C", "N"]},
            }
        if action_id == "generate_conformer_ensemble":
            generated = next(generated_components)
            return {
                "status": "success",
                "result": {
                    "ensemble": [
                        {"conformer_id": "conformer_0001", "structure": generated}
                    ]
                },
            }
        if action_id == "optimize_geometry":
            return {
                "status": "success",
                "result": {
                    "structure": request["inputs"]["structure"],
                    "energy_kcal_mol": 0.0,
                    "converged": True,
                },
            }
        if action_id == "rank_conformers_from_results":
            return {
                "status": "success",
                "result": {"ensemble": request["inputs"]["ensemble"]},
            }
        if action_id == "enumerate_conformer_combinations":
            return {
                "status": "success",
                "result": {
                    "combinations": [
                        {
                            "combination_id": "combination_0001",
                            "components": [component_a, component_b],
                        }
                    ],
                    "combination_count": 1,
                    "truncated": False,
                },
            }
        if action_id == "resolve_mapped_reactive_contacts":
            return {"status": "success", "result": {"contact_pairs": []}}
        if action_id == "build_reactive_complex_ensemble":
            return {
                "status": "success",
                "result": {
                    "ensemble": [
                        {
                            "candidate_id": "candidate_0001",
                            "structure": complex_structure,
                        }
                    ]
                },
            }
        if action_id == "build_mapped_distance_constraints":
            return {"status": "success", "result": {"constraints": []}}
        if action_id == "apply_reaction_changes_to_structure":
            return {
                "status": "success",
                "result": {"structure": request["inputs"]["structure"]},
            }
        if backend_id == "ash_neb_uma":
            return {
                "status": "success",
                "backend_version": "ash-test-version",
                "provenance": {
                    "engine": "ASH/KNARR NEB",
                    "calculator": "ASH FairchemTheory",
                    "path_identity_metadata_restored": True,
                    "path_is_validated_transition_state": False,
                    "endpoint_atom_order_checked": True,
                    "endpoint_atom_maps_checked": True,
                },
                "result": {
                    "images": [_structure() for _ in range(9)],
                    "energies_hartree": [
                        -1.0,
                        -0.9,
                        -0.8,
                        -0.7,
                        -0.6,
                        -0.7,
                        -0.8,
                        -0.9,
                        -1.0,
                    ],
                    "energy_image_indices": list(range(9)),
                    "energy_unit": "hartree",
                    "highest_energy_image_index": 4,
                    "highest_interior_image_index": 4,
                    "interior_peak_candidate_valid": True,
                    "energy_index_complete": True,
                    "final_path_complete": True,
                    "total_images": 9,
                    "transition_state_guess": _structure(distance=1.25),
                    "transition_state_guess_source": "knarr_saddle_xyz",
                    "transition_state_guess_source_image_index": None,
                    "transition_state_guess_selection_rule_version": (
                        "ts-guess-selection.v1"
                    ),
                },
            }
        raise AssertionError((action_id, backend_id))

    response = arche_ts_pipeline.run_atomic_workflow(
        _path_request(), executor=executor
    )

    assert response["status"] == "success"
    result = response["result"]
    assert set(result) == {
        "workflow_id",
        "endpoint_pairs",
        "endpoint_pair_count",
        "path_image_candidates",
        "stage_statuses",
            "failure_summary",
            "stage_diagnostics",
            "transition_states_validated",
            "main_force_field",
            "target_force_field",
        }
    assert result["endpoint_pair_count"] == 1
    assert len(result["path_image_candidates"]) == 9
    pair = result["endpoint_pairs"][0]
    assert set(pair) == {
        "pair_id",
        "stage_statuses",
        "reactant_artifact",
        "product_artifact",
        "endpoint_artifact_source",
        "reaction_path_artifact",
        "transition_state_guess_artifact",
        "transition_state_guess_source",
        "transition_state_guess_source_image_index",
        "transition_state_guess_selection_rule_version",
        "path_image_artifacts",
        "provenance",
    }
    assert result["transition_states_validated"] is False
    warnings = " ".join(response["warnings"]).lower()
    assert "cannot be used as scientific evidence" in warnings
    assert "transition-state optimization" in warnings
    assert "exactly one imaginary frequency" in warnings
    assert "intended reaction coordinate" in warnings
    assert "forward and reverse (bidirectional) irc" in warnings
    assert "expected reactant and product" in warnings
    assert "activation barrier" not in warnings
    assert "selectivity" not in warnings
    assert "thermochemistry" not in warnings
    assert pair["provenance"]["backend_id"] == "ash_neb_uma"
    assert pair["provenance"]["backend_version"] == "ash-test-version"
    assert pair["provenance"]["module"] == "minichem_toolbox.backends.ash_neb"
    assert pair["provenance"]["protocol_version"] == "ts-pipeline-protocol.v1"
    assert pair["provenance"]["selection_rule_version"] == "ts-guess-selection.v1"

    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "a" * 64)
    monkeypatch.setattr(
        service,
        "probe_all_backends",
        lambda specifications: {
            specification.id: {
                "available": True,
                "status": "healthy",
                "runtime": specification.runtime,
            }
            for specification in specifications
        },
    )
    original_invoke_worker = service.invoke_worker
    captured_worker_payloads: list[dict] = []

    def capture_worker_payload(**kwargs):
        """记录真实 service→worker 请求并返回固定 Action 结果。"""

        captured_worker_payloads.append(copy.deepcopy(kwargs["payload"]))
        return copy.deepcopy(response)

    monkeypatch.setattr(
        service,
        "invoke_worker",
        capture_worker_payload,
    )
    public = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
                inputs=copy.deepcopy(_PATH_INPUTS),
            method_spec={},
            action_settings={},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
        )
    )

    assert public["status"] == "success", public
    public_pair = public["result"]["endpoint_pairs"][0]
    assert public_pair["reactant_artifact"]["artifact_id"].startswith("art_")
    assert public_pair["product_artifact"]["artifact_id"].startswith("art_")
    assert public_pair["reaction_path_artifact"]["artifact_id"].startswith("art_")
    assert public_pair["transition_state_guess_artifact"]["artifact_id"].startswith("art_")
    reaction_path = ArtifactStore().load(public_pair["reaction_path_artifact"])
    assert len(reaction_path["images"]) == len(reaction_path["energies_hartree"])
    assert len(reaction_path["images"]) == len(reaction_path["energy_image_indices"])
    assert len(public_pair["path_image_artifacts"]) == len(reaction_path["images"])
    assert reaction_path["provenance"]["backend_version"] == "ash-test-version"
    assert public_pair["transition_state_guess_source"] == "knarr_saddle_xyz"
    transition_state_artifact = ArtifactStore().find(
        public_pair["transition_state_guess_artifact"]["artifact_id"]
    )
    assert transition_state_artifact.semantic_type == "XYZFile"
    assert all(
        candidate["xyz_artifact"]["artifact_id"].startswith("art_")
        for candidate in public["result"]["path_image_candidates"]
    )
    monkeypatch.setattr(service, "invoke_worker", original_invoke_worker)
    converted = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="xyz_to_gjf",
            backend_id="format_conversion",
            inputs={
                "input_xyz_path": public_pair["transition_state_guess_artifact"],
                "route_section": "#p PM6 opt",
            },
            method_spec={},
            action_settings={},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 0},
        )
    )
    assert converted["status"] == "success", converted
    assert converted["result"]["native_preflight_ready"] is True
    public_text = json.dumps(public, ensure_ascii=False, sort_keys=True)
    assert str(tmp_path) not in public_text
    assert public["provenance"]["catalog_hash"] == "a" * 64
    assert public["provenance"]["nested_stage_provenance"][0]["backend_id"] == (
        "ash_neb_uma"
    )
    assert len(captured_worker_payloads) == 1
    worker_request = captured_worker_payloads[0]["request"]
    assert "timeout_seconds" not in worker_request["action_settings"]
    assert worker_request["resource_limits"]["walltime_seconds"] > 0
    assert "supplementary_artifact_manifest" not in public_text
    assert public["output_artifacts"]
    assert all(
        set(artifact)
        >= {
            "artifact_id",
            "semantic_type",
            "media_type",
            "sha256",
            "path",
            "producer_action",
            "producer_backend",
            "parent_artifact_ids",
        }
        for artifact in public["output_artifacts"]
    )
    assert "producer_action" in public_text
    assert "producer_backend" in public_text
    assert "parent_artifact_ids" in public_text
    assert "path_image_candidates" in public_text


def test_generic_ts_collect_exposes_complete_artifact_handoff_and_readable_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """完整 backend 结果经 generic collect 后，所有科学交接引用都可读取。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    scope = ScopeContext(
        task_id="task-ts-collect",
        batch_id="batch-ts-collect",
        combination_id="combination-ts-collect",
        scope_id="scope-ts-collect",
        workspace=tmp_path,
        capability_hash="capability-ts-collect",
    )
    guess = _structure(distance=1.25)
    reaction_path = {
        "images": [_structure(), guess, _structure(distance=1.5)],
        "energies_hartree": [-1.0, -0.8, -0.9],
        "energy_image_indices": [0, 1, 2],
        "energy_unit": "hartree",
        "highest_energy_image_index": 1,
        "highest_interior_image_index": 1,
        "interior_peak_candidate_valid": True,
        "energy_index_complete": True,
        "final_path_complete": True,
        "total_images": 3,
        "transition_state_guess": guess,
        "transition_state_guess_source": "highest_interior_path_image",
        "transition_state_guess_source_image_index": 1,
        "transition_state_guess_selection_rule_version": "ts-guess-selection.v1",
        "provenance": {
            "backend_id": "ash_neb_uma",
            "backend_version": "ash-test-version",
            "selection_rule_version": "ts-guess-selection.v1",
        },
    }
    raw_result, artifact_files = arche_ts_pipeline._materialize_pipeline_artifacts(
        {
            "workflow_id": "transition_state_path_search",
            "endpoint_pairs": [
                {
                    "pair_id": "endpoint_pair_0001",
                    "combination_id": "combination_0001",
                    "source_candidate_id": "candidate_0001",
                    "reactant": _structure(),
                    "product": _structure(distance=1.5),
                    "reactant_uma": _structure(),
                    "product_uma": _structure(distance=1.5),
                    "reactant_uma_converged": True,
                    "product_uma_converged": True,
                    "stage_statuses": {
                        "mmff": "success",
                        "uma": "success",
                        "neb": "success",
                    },
                    "reaction_path": reaction_path,
                }
            ],
            "transition_states_validated": False,
        }
    )
    worker_result = {
        "status": "success",
        "result": raw_result,
        "artifact_files": artifact_files,
        "backend_version": "ash-test-version",
        "provenance": {
            "pipeline_protocol_version": "ts-pipeline-protocol.v1",
            "nested_stage_provenance": [
                {
                    "pair_id": "endpoint_pair_0001",
                    **reaction_path["provenance"],
                }
            ],
        },
        "warnings": [],
        "retryable": False,
    }

    def fake_start_job(**kwargs):
        job_id = str(kwargs["job_id"])
        directory = tmp_path / "outputs" / "execution_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(kwargs["metadata"])
        identity = {
            "schema_version": 1,
            "job_id": job_id,
            "job_type": transition_state_jobs.JOB_TYPE,
            "runtime": "ash_neb",
            "metadata": metadata,
            "scope_id": scope.scope_id,
            "scope_workspace": str(scope.workspace),
            "operation_id": str(kwargs["operation_id"]),
            "operation_fingerprint": kwargs["operation_fingerprint"],
            "caller_request_fingerprint": kwargs["caller_request_fingerprint"],
            "submission_key": kwargs["submission_key"],
        }
        status = {
            key: identity[key]
            for key in (
                "schema_version",
                "job_id",
                "job_type",
                "scope_id",
                "scope_workspace",
                "operation_id",
                "operation_fingerprint",
                "caller_request_fingerprint",
                "submission_key",
            )
        }
        status["status"] = "queued"
        transition_state_jobs._atomic_json(directory / "request.json", identity)
        transition_state_jobs._atomic_json(directory / "status.json", status)
        return {"status": "success", "job_id": job_id, "job_status": "queued"}

    monkeypatch.setattr(transition_state_jobs, "_start_job", fake_start_job)
    monkeypatch.setattr(transition_state_jobs, "active_catalog_hash", lambda: "catalog-test")
    monkeypatch.setattr(
        transition_state_jobs,
        "execute_traced",
        lambda _name, _arguments, function, **_kwargs: function(),
    )
    request = transition_state_jobs.TransitionStatePathJobRequest.model_validate(
        {
            "submission_key": "complete-ts-collect",
            "backend_id": transition_state_jobs.BACKEND_ID,
            "inputs": {"reactants": ["CC"], "products": ["CO"]},
            "resource_limits": {"gpu_count": 1},
        }
    )

    with activate_scope(scope):
        submitted = transition_state_jobs.submit_transition_state_path_job(request)
        job_id = str(submitted["job_id"])
        job_directory = tmp_path / "outputs" / "execution_jobs" / job_id
        status = json.loads((job_directory / "status.json").read_text(encoding="utf-8"))
        status["status"] = "success"
        transition_state_jobs._atomic_json(job_directory / "status.json", status)
        for artifact_file in artifact_files:
            source = tmp_path / artifact_file["path"]
            target = job_directory / artifact_file["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        transition_state_jobs._atomic_json(
            job_directory / transition_state_jobs.RAW_RESULT_FILENAME,
            worker_result,
        )
        collected = open_tools.collect_execution_job(
            JobCollectRequest(
                job_id=job_id,
                submission_key="complete-ts-collect",
            )
        )
        assert collected.get("status") == "success", collected
        assert "action_result" in collected, collected
        action_result = collected["action_result"]
        public_pair = action_result["result"]["endpoint_pairs"][0]
        reaction_path_ref = public_pair["reaction_path_artifact"]
        guess_ref = public_pair["transition_state_guess_artifact"]
        reaction_path_full_ref = ArtifactStore(root=tmp_path, create=False).find(
            reaction_path_ref["artifact_id"]
        )
        path_page = open_tools.read_workspace_text(
            WorkspaceTextReadRequest(
                path=reaction_path_full_ref.path,
                json_pointer="/energies_hartree",
            )
        )
        guess_page = open_tools.read_scientific_artifact_text(
            ScientificArtifactTextReadRequest(artifact_id=guess_ref["artifact_id"])
        )

    assert collected["status"] == "success"
    assert action_result["status"] == "success"
    assert public_pair["reactant_artifact"]["artifact_id"].startswith("art_")
    assert public_pair["product_artifact"]["artifact_id"].startswith("art_")
    assert reaction_path_ref["artifact_id"].startswith("art_")
    assert guess_ref["artifact_id"].startswith("art_")
    assert public_pair["transition_state_guess_source"] == (
        "highest_interior_path_image"
    )
    assert public_pair["transition_state_guess_source_image_index"] == 1
    assert public_pair["provenance"]["backend_id"] == "ash_neb_uma"
    assert public_pair["provenance"]["backend_version"] == "ash-test-version"
    assert public_pair["provenance"]["module_version"] == "ash-test-version"
    assert len(public_pair["path_image_artifacts"]) == 3
    candidate_by_id = {
        item["candidate_id"]: item
        for item in action_result["result"]["path_image_candidates"]
    }
    assert all(
        candidate_by_id[item["candidate_id"]]["xyz_artifact"]["artifact_id"].startswith(
            "art_"
        )
        for item in public_pair["path_image_artifacts"]
    )
    assert action_result["provenance"]["nested_stage_provenance"][0]["backend_id"] == (
        "ash_neb_uma"
    )
    output_artifact_ids = {
        item["artifact_id"] for item in action_result["output_artifacts"]
    }
    assert {
        public_pair["reactant_artifact"]["artifact_id"],
        public_pair["product_artifact"]["artifact_id"],
        public_pair["reaction_path_artifact"]["artifact_id"],
        public_pair["transition_state_guess_artifact"]["artifact_id"],
        *[
            candidate["xyz_artifact"]["artifact_id"]
            for candidate in action_result["result"]["path_image_candidates"]
        ],
    } <= output_artifact_ids
    assert path_page["status"] == "success"
    assert "-0.8" in path_page["content"]
    assert guess_page["status"] == "success"
    assert "1.25" in guess_page["content"]


def test_fixed_action_unavailable_response_hides_health_and_fixed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """公开失败响应只返回可行动摘要，不携带健康探针或固定科学输入。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    fixed_marker = _PATH_INPUTS["reactants"][0]
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "test-catalog-hash")
    monkeypatch.setattr(
        service,
        "probe_all_backends",
        lambda specifications: {
            specification.id: {
                "available": False,
                "status": "unavailable",
                "runtime_python": "/mnt/private/runtime/python",
                "diagnostic": fixed_marker,
            }
            for specification in specifications
        },
    )

    public = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
                inputs=copy.deepcopy(_PATH_INPUTS),
            method_spec={},
            action_settings={},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
        )
    )

    assert public["status"] == "unavailable"
    assert {"code", "message", "health"} <= set(public["error"])
    assert public["provenance"]["catalog_hash"] == "test-catalog-hash"
    serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
    assert "/mnt/private" in serialized
    assert fixed_marker in serialized
    assert "runtime_python" in serialized
    assert "health" in serialized


def test_fixed_action_worker_error_does_not_echo_scientific_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """worker 输入错误不得把调用方正文、上游诊断或内部产物引用带回 MCP。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    sentinel = "CALLER_SCIENTIFIC_OVERRIDE_SENTINEL"
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "test-catalog-hash")
    monkeypatch.setattr(
        service,
        "invoke_worker",
        lambda **_kwargs: {
            "status": "invalid_request",
            "result": {"echo": sentinel},
            "error": {
                "code": "backend_input_error",
                "message": f"invalid fixed input: {sentinel}",
                "health": {"runtime_python": "/mnt/private/runtime/python"},
            },
            "artifact_files": [
                {
                    "path": "config.local.env",
                    "semantic_type": "BackendDiagnostic",
                    "media_type": "application/json",
                }
            ],
            "retryable": False,
        },
    )

    public = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
                inputs={
                    **copy.deepcopy(_PATH_INPUTS),
                    "reactants": [*_PATH_INPUTS["reactants"], sentinel],
                },
            method_spec={},
            action_settings={},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
        )
    )

    assert public["status"] == "invalid_request"
    assert public["result"] == {"echo": sentinel}
    assert len(public["output_artifacts"]) == 1
    assert public["output_artifacts"][0]["semantic_type"] == "BackendDiagnostic"
    assert set(public["output_artifacts"][0]) == {
        "artifact_id",
        "semantic_type",
        "media_type",
        "sha256",
        "path",
        "producer_action",
        "producer_backend",
        "parent_artifact_ids",
    }
    assert {"code", "message", "health"} <= set(public["error"])
    serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
    assert sentinel in serialized
    assert "health" in serialized
    assert "config.local.env" not in serialized


def test_fixed_action_resource_capacity_error_keeps_retryable_diagnostic_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """服务端 GPU 预算为零时，固定后端在资源合同阶段明确拒绝请求。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "a" * 64)

    public = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
                inputs=copy.deepcopy(_PATH_INPUTS),
            method_spec={},
            action_settings={},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
        )
    )

    assert public["status"] == "invalid_request"
    assert public["error"]["code"] == "invalid_resource_limits"
    assert "retryable" not in public["error"]
    assert "requires empty inputs" not in public["error"]["message"]
    assert public["retryable"] is False
    assert public["output_artifacts"] == []
    assert "gpu_count" in public["error"]["message"]


def test_fixed_action_aggregate_capacity_error_keeps_safe_retry_when(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """活动作业占满容量时保留稳定 retry_when，而不暴露容量数值。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "4096")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "a" * 64)

    def raise_capacity(*_args, **_kwargs):
        raise ResourceBudgetExceeded(
            requested={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
            budget=ResourceBudget(cpu_cores=1, memory_mb=4096, gpu_count=1),
            reserved={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
            aggregate=True,
        )

    monkeypatch.setattr(service, "reserve_resources", raise_capacity)
    public = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="search_transition_state_path",
            backend_id="transition_state_path_search",
                inputs=copy.deepcopy(_PATH_INPUTS),
            method_spec={},
            action_settings={},
            resource_limits={"cpu_cores": 1, "memory_mb": 4096, "gpu_count": 1},
        )
    )

    assert public["status"] == "invalid_request"
    assert public["error"]["code"] == "aggregate_resource_budget_exceeded"
    assert "requested" in public["error"]
    assert "budget" in public["error"]
    assert "currently_reserved" in public["error"]
    assert public["retryable"] is True
    assert public["output_artifacts"][0]["semantic_type"] == "BackendDiagnostic"
    assert public["error"]["requested"]["gpu_count"] == 1


def test_fixed_action_full_compatibility_surface_reuses_public_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """完整兼容工具与渐进入口必须使用同一脱敏结果，不能旁路返回服务内部字段。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    sentinel = "FULL_SURFACE_INTERNAL_SENTINEL"
    monkeypatch.setattr(
        mcp_registry,
        "execute_action",
        lambda *_args, **_kwargs: {
            "status": "unavailable",
            "action": "search_transition_state_path",
            "action_version": "1.0.0",
            "requested_backend": "transition_state_path_search",
            "backend": "transition_state_path_search",
            "selection_source": "agent",
            "result": {"internal": sentinel},
            "input_artifacts": [],
            "output_artifacts": [],
            "warnings": [sentinel],
            "provenance": {
                "runtime_python": "/mnt/private/runtime/python",
                "internal": sentinel,
            },
            "error": {
                "code": "backend_unavailable",
                "message": sentinel,
                "health": {"runtime_python": "/mnt/private/runtime/python"},
            },
            "retryable": False,
        },
    )
    invoke = mcp_registry._make_action_callable(
        action_specs()["search_transition_state_path"]
    )

    public = invoke(
        ActionRequest(
            backend_id="transition_state_path_search",
            inputs=copy.deepcopy(_PATH_INPUTS),
            method_spec={},
            action_settings={},
            resource_limits={"gpu_count": 1},
        )
    )

    assert public["status"] == "unavailable"
    assert {"code", "message", "health"} <= set(public["error"])
    serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
    assert sentinel in serialized
    assert "/mnt/private" in serialized
    assert "provenance" in serialized
    assert "health" in serialized


def test_stage_error_text_redacts_upstream_paths_and_fixed_inputs():
    sentinel = "/private/runtime/FIXED_INPUT_SENTINEL/error.log"

    def executor(*_args):
        return {
            "status": "unavailable",
            "error": {"message": f"upstream failed at {sentinel}"},
        }

    with pytest.raises(arche_ts_pipeline.StageUnavailable) as raised:
        arche_ts_pipeline._call(
            executor,
            [],
            "map_reaction_atoms",
            "rxnmapper",
            inputs={"reaction": "hidden"},
        )

    assert sentinel not in str(raised.value)
    assert str(raised.value) == "transition-state path stage unavailable"


def test_nested_ash_provenance_is_bounded_and_kept_in_trace():
    """嵌套 ASH 来源版本进入 trace，但路径和任意敏感字段不会外溢。"""

    trace = []

    def executor(*_args):
        return {
            "status": "success",
            "backend_version": "ash-test-version",
            "provenance": {
                "engine": "ASH/KNARR NEB",
                "calculator": "ASH FairchemTheory",
                "private_path": "/private/ash/result",
                "secret": "not-public",
                "path_identity_metadata_restored": True,
            },
            "result": {
                "transition_state_guess_selection_rule_version": (
                    "ts-guess-selection.v1"
                )
            },
        }

    arche_ts_pipeline._call(
        executor,
        trace,
        "search_reaction_path",
        "ash_neb_uma",
        inputs={},
    )

    provenance = trace[0]["provenance"]
    assert provenance["backend_id"] == "ash_neb_uma"
    assert provenance["backend_version"] == "ash-test-version"
    assert provenance["module"] == "minichem_toolbox.backends.ash_neb"
    assert provenance["protocol_version"] == "ts-pipeline-protocol.v1"
    assert provenance["selection_rule_version"] == "ts-guess-selection.v1"
    assert "private_path" not in json.dumps(trace, ensure_ascii=False)
    assert "secret" not in json.dumps(trace, ensure_ascii=False)


def test_fixed_configuration_read_failures_use_one_generic_error(monkeypatch):
    class BrokenResource:
        def joinpath(self, _name):
            return self

        def read_bytes(self):
            return b'{"hidden": "/mnt/private/config.json"}'

    monkeypatch.setattr(
        arche_ts_pipeline.package_resources,
        "files",
        lambda _package: BrokenResource(),
    )
    with pytest.raises(RuntimeError, match="service configuration unavailable") as raised:
        arche_ts_pipeline._load_server_configuration()

    assert "/mnt/private/config.json" not in str(raised.value)


def test_author_mmff_uses_stage_specific_source_effective_tolerances():
    settings = arche_ts_pipeline._validate_settings({})

    assert settings["component_mmff_force_tolerance"] == pytest.approx(1.0e-4)
    assert settings["component_mmff_energy_tolerance"] == pytest.approx(1.0e-6)
    assert settings["constrained_mmff_force_tolerance"] == pytest.approx(1.0e-5)
    assert settings["constrained_mmff_energy_tolerance"] == pytest.approx(1.0e-7)


def test_fixed_reaction_uses_author_rotatable_bond_conformer_request_counts():
    """固定反应的构象请求数必须沿用作者 CalcNumRotatableBonds 公式。"""

    observed_counts = []
    structure_record = _structure()

    def executor(action_id, _backend_id, request):
        if action_id == "generate_conformer_ensemble":
            observed_counts.append(request["action_settings"]["num_conformers"])
            return {
                "status": "success",
                "result": {
                    "ensemble": [
                        {"conformer_id": "conformer_0001", "structure": structure_record}
                    ]
                },
            }
        if action_id == "optimize_geometry":
            return {
                "status": "success",
                "result": {
                    "structure": request["inputs"]["structure"],
                    "energy_kcal_mol": 0.0,
                },
            }
        if action_id == "rank_conformers_from_results":
            return {
                "status": "success",
                "result": {"ensemble": request["inputs"]["ensemble"]},
            }
        raise AssertionError(action_id)

    arche_ts_pipeline._component_ensembles(
        _PATH_INPUTS["reactants"],
        arche_ts_pipeline._validate_settings({}),
        executor,
        [],
        {"force_field": "mmff94"},
    )

    # pruneRmsThresh 会把实际返回数进一步降低；这里锁定的是传给作者
    # EmbedMultipleConfs 的 numConfs，而不是嵌入后的实际保留数。
    assert observed_counts == [266, 26]


def test_heavy_stage_orchestration_is_pair_independent_and_uses_source_effective_ash_defaults():
    calls = []
    uma_convergence = iter((True, True, False, True))

    def executor(action_id, backend_id, request):
        calls.append((action_id, backend_id, copy.deepcopy(request)))
        if backend_id == "fairchem_uma":
            return {
                "status": "success",
                "result": {
                    "structure": request["inputs"]["structure"],
                    "converged": next(uma_convergence),
                },
            }
        if backend_id == "ash_neb_uma":
            return {
                "status": "success",
                "result": {
                    "images": [_structure() for _ in range(9)],
                    "energies_hartree": [-1.0] * 9,
                    "energy_image_indices": list(range(9)),
                    "energy_index_complete": True,
                    "final_path_complete": True,
                    "interior_peak_candidate_valid": True,
                    "total_images": 9,
                    "converged": True,
                },
            }
        raise AssertionError((action_id, backend_id))

    pairs, unavailable = arche_ts_pipeline._run_heavy_stages(
        [
            {
                "pair_id": "endpoint_pair_0001",
                "reactant": _structure(),
                "product": _structure(),
            },
            {
                "pair_id": "endpoint_pair_0002",
                "reactant": _structure(),
                "product": _structure(),
            },
        ],
        {},
        {
            "uma_model": "uma-s-1p1",
            "uma_task_name": "omol",
            "uma_fmax_ev_per_angstrom": 0.01,
            "uma_optimizer": "bfgs",
            "uma_max_steps": 500,
            "total_images": 9,
            "climb": False,
            "neb_max_steps": 300,
            "random_seed": 42,
        },
        {"cpu_cores": 1, "memory_mb": 1024, "gpu_count": 1},
        executor,
        [],
    )

    assert unavailable == "endpoint_pair_0002 UMA endpoint stage unavailable or unconverged"
    assert pairs[0]["stage_statuses"] == {"uma": "success", "neb": "success"}
    assert pairs[1]["stage_statuses"] == {"uma": "failed", "neb": "skipped"}
    neb_calls = [item for item in calls if item[1] == "ash_neb_uma"]
    assert len(neb_calls) == 1
    assert neb_calls[0][2]["method_spec"]["random_seed"] == 41
    assert neb_calls[0][2]["action_settings"]["total_images"] == 9
    assert neb_calls[0][2]["method_spec"]["allow_model_download"] is False


def test_heavy_stage_failure_summary_preserves_safe_nested_code_and_retryability():
    """阶段失败摘要保留安全错误分类和重试语义，但不复制上游诊断正文。"""

    failure_records = []

    def executor(action_id, backend_id, request):
        if backend_id == "fairchem_uma":
            return {
                "status": "unavailable",
                "error": {
                    "code": "backend_unavailable",
                    "message": "secret install command /private/case1/install",
                    "install": "pip install secret-package",
                },
                "retryable": True,
            }
        raise AssertionError((action_id, backend_id, request))

    pairs, unavailable = arche_ts_pipeline._run_heavy_stages(
        [
            {
                "pair_id": "endpoint_pair_0001",
                "reactant": _structure(),
                "product": _structure(),
            }
        ],
        {},
        {
            "uma_model": "uma-s-1p1",
            "uma_task_name": "omol",
            "uma_fmax_ev_per_angstrom": 0.01,
            "uma_optimizer": "bfgs",
            "uma_max_steps": 500,
            "total_images": 9,
            "climb": False,
            "neb_max_steps": 300,
        },
        {"gpu_count": 1},
        executor,
        [],
        failure_records=failure_records,
    )

    assert unavailable == "endpoint_pair_0001 UMA endpoint stage unavailable or unconverged"
    assert pairs[0]["stage_statuses"] == {"uma": "failed", "neb": "skipped"}
    assert failure_records == [
        {
            "pair_id": "endpoint_pair_0001",
            "stage": "uma",
            "status": "failed",
            "reason_code": "backend_unavailable",
            "retryable": True,
        },
        {
            "pair_id": "endpoint_pair_0001",
            "stage": "neb",
            "status": "skipped",
            "reason_code": "pair_stage_skipped",
            "retryable": False,
        },
    ]
    serialized = json.dumps(failure_records, ensure_ascii=False)
    assert "secret install" not in serialized
    assert "secret-package" not in serialized
    assert "/private/case1" not in serialized


def test_heavy_stage_oserror_is_normalized_without_sensitive_details():
    """UMA 子调用的 OSError 只形成安全、可重试的 pair 失败摘要。"""

    failure_records = []

    def executor(action_id, backend_id, request):
        if backend_id == "fairchem_uma":
            raise OSError("/private/fairchem/checkpoint/weights.bin")
        raise AssertionError((action_id, backend_id, request))

    pairs, unavailable = arche_ts_pipeline._run_heavy_stages(
        [
            {
                "pair_id": "endpoint_pair_0001",
                "reactant": _structure(),
                "product": _structure(),
            }
        ],
        {},
        {
            "uma_model": "uma-s-1p1",
            "uma_task_name": "omol",
            "uma_fmax_ev_per_angstrom": 0.01,
            "uma_optimizer": "bfgs",
            "uma_max_steps": 500,
            "total_images": 9,
            "climb": False,
            "neb_max_steps": 300,
        },
        {"gpu_count": 1},
        executor,
        [],
        failure_records=failure_records,
    )

    assert unavailable == "endpoint_pair_0001 UMA endpoint stage unavailable or unconverged"
    assert pairs[0]["stage_statuses"] == {"uma": "failed", "neb": "skipped"}
    assert failure_records[0]["reason_code"] == "backend_unavailable"
    assert failure_records[0]["retryable"] is True
    serialized = json.dumps(failure_records, ensure_ascii=False)
    assert "/private/" not in serialized
    assert "weights.bin" not in serialized


def test_fixed_pipeline_partial_result_propagates_nested_retryability(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """固定 Action 顶层 partial_success 必须反映可重试的嵌套阶段失败。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    structure = _structure()

    def fake_call(executor, trace, action_id, backend_id, **_kwargs):
        trace.append(
            {
                "sequence": len(trace) + 1,
                "action_id": action_id,
                "backend_id": backend_id,
                "status": "success",
                "error": None,
            }
        )
        if action_id == "map_reaction_atoms":
            return {"mapped": True}
        if action_id == "analyze_reaction_changes":
            if trace[-1]["sequence"] == 2:
                return {
                    "hydrogen_transfer_summary": {
                        "donors": [{"atom_map": 1, "hydrogens_lost": 1}],
                        "acceptors": [{"atom_map": 2, "hydrogens_gained": 1}],
                        "balanced": True,
                    }
                }
            return {
                "mapping_complete": True,
                "unmapped_hydrogen_count": 0,
                "formed_bonds": [],
                "broken_bonds": [],
            }
        if action_id == "materialize_mapped_hydrogen_transfers":
            return {"reactants": ["C", "N"]}
        if action_id == "enumerate_conformer_combinations":
            return {"combinations": [], "truncated": False}
        raise AssertionError(action_id)

    pair = {
        "pair_id": "endpoint_pair_0001",
        "combination_id": "combination_0001",
        "source_candidate_id": "candidate_0001",
        "reactant": structure,
        "product": structure,
        "source_mmff_converged": True,
        "target_mmff_converged": True,
        "stage_statuses": {"mmff": "success", "uma": "pending", "neb": "pending"},
    }

    monkeypatch.setattr(arche_ts_pipeline, "_call", fake_call)
    monkeypatch.setattr(
        arche_ts_pipeline,
        "_component_ensembles",
        lambda *_args, **_kwargs: [{"ensemble": []}],
    )
    monkeypatch.setattr(
        arche_ts_pipeline,
        "_prepare_endpoint_pairs",
        lambda *_args, **_kwargs: [copy.deepcopy(pair)],
    )
    monkeypatch.setattr(
        arche_ts_pipeline,
        "_cluster_endpoint_pairs",
        lambda pairs, *_args, **_kwargs: pairs,
    )

    def fake_heavy_stages(
        pairs, *_args, failure_records=None, diagnostic_records=None, **_kwargs
    ):
        pairs[0]["stage_statuses"] = {
            "mmff": "success",
            "uma": "failed",
            "neb": "skipped",
        }
        failure_records.extend(
            [
                {
                    "pair_id": "endpoint_pair_0001",
                    "stage": "uma",
                    "status": "failed",
                    "reason_code": "backend_unavailable",
                    "retryable": True,
                },
                {
                    "pair_id": "endpoint_pair_0001",
                    "stage": "neb",
                    "status": "skipped",
                    "reason_code": "pair_stage_skipped",
                    "retryable": False,
                },
            ]
        )
        diagnostic_records.extend(
            [
                {
                    "pair_id": "endpoint_pair_0001",
                    "stage": "neb",
                    "status": "partial",
                    "message": "ASH NEB needs a larger image budget",
                    "warnings": ["Increase total_images before retrying."],
                }
            ]
        )
        return pairs, "endpoint_pair_0001 UMA endpoint stage unavailable or unconverged"

    monkeypatch.setattr(arche_ts_pipeline, "_run_heavy_stages", fake_heavy_stages)

    response = arche_ts_pipeline.run_atomic_workflow(
        _path_request(),
        executor=lambda *_args: None,
    )

    assert response["status"] == "partial_success"
    assert response["retryable"] is True
    assert response["result"]["failure_summary"] == [
        {
            "pair_id": "endpoint_pair_0001",
            "stage": "uma",
            "status": "failed",
            "reason_code": "backend_unavailable",
            "retryable": True,
        },
        {
            "pair_id": "endpoint_pair_0001",
            "stage": "neb",
            "status": "skipped",
            "reason_code": "pair_stage_skipped",
            "retryable": False,
        },
    ]
    assert response["result"]["stage_diagnostics"] == [
        {
            "pair_id": "endpoint_pair_0001",
            "stage": "neb",
            "status": "partial",
            "message": "ASH NEB needs a larger image budget",
            "warnings": ["Increase total_images before retrying."],
        }
    ]
    assert any("larger image budget" in warning for warning in response["warnings"])
    assert any("Increase total_images" in warning for warning in response["warnings"])


def test_malformed_successful_endpoint_result_only_fails_one_pair():
    calls = []
    uma_calls = 0

    def executor(action_id, backend_id, request):
        nonlocal uma_calls
        calls.append((action_id, backend_id))
        if backend_id == "fairchem_uma":
            uma_calls += 1
            if uma_calls == 1:
                return {"status": "success", "result": {}}
            return {
                "status": "success",
                "result": {
                    "structure": request["inputs"]["structure"],
                    "converged": True,
                },
            }
        if backend_id == "ash_neb_uma":
            return {
                "status": "success",
                "result": {
                    "images": [_structure() for _ in range(9)],
                    "energies_hartree": [-1.0] * 9,
                    "energy_image_indices": list(range(9)),
                    "energy_index_complete": True,
                    "final_path_complete": True,
                    "interior_peak_candidate_valid": True,
                    "total_images": 9,
                },
            }
        raise AssertionError((action_id, backend_id))

    pairs, unavailable = arche_ts_pipeline._run_heavy_stages(
        [
            {"pair_id": "pair-1", "reactant": _structure(), "product": _structure()},
            {"pair_id": "pair-2", "reactant": _structure(), "product": _structure()},
        ],
        {},
        {
            "uma_model": "uma-s-1p1",
            "uma_task_name": "omol",
            "uma_fmax_ev_per_angstrom": 0.01,
            "uma_optimizer": "bfgs",
            "uma_max_steps": 500,
            "total_images": 9,
            "climb": False,
            "neb_max_steps": 300,
            "random_seed": 42,
        },
        {"gpu_count": 1},
        executor,
        [],
    )
    assert unavailable == "pair-1 UMA endpoint stage unavailable or unconverged"
    assert pairs[0]["stage_statuses"]["neb"] == "skipped"
    assert pairs[1]["stage_statuses"]["neb"] == "success"
    assert calls.count(("search_reaction_path", "ash_neb_uma")) == 1


def test_author_assembly_continues_python_random_uniform_stream_across_combinations():
    components = [_structure(), _structure(map_offset=10)]
    request_base = {
        "inputs": {
            "components": components,
            "contact_pairs": [
                {
                    "component_a": 0,
                    "atom_a": 0,
                    "component_b": 1,
                    "atom_b": 0,
                    "target_distance_angstrom": 2.0,
                    "tolerance_angstrom": 0.5,
                }
            ],
        },
        "method_spec": {},
        "action_settings": {
            "random_seed": 42,
            "candidate_count": 10,
            "max_candidates": 10,
            "orientation_trials": 1,
            "initial_centroid_distance_angstrom": 20.0,
            "minimum_interatomic_distance_angstrom": 0.0,
            "maximum_reported_collisions": 32,
            "random_stream_offset": 0,
        },
    }
    first = reactive_complex.execute(
        "build_reactive_complex_ensemble",
        "internal_reactive_complex",
        request_base,
    )
    assert first["status"] == "success"
    rng = random.Random(42)
    expected_first_attempt = [rng.uniform(0.0, 360.0) for _ in range(3)] + [
        rng.uniform(-5.0, 5.0) for _ in range(3)
    ]
    candidate = first["result"]["ensemble"][1]
    assert candidate["random_draw_start"] == 0
    assert candidate["component_transforms"][1]["random_translation_angstrom"] == pytest.approx(
        expected_first_attempt[3:]
    )
    expected_second_attempt = [rng.uniform(0.0, 360.0) for _ in range(3)] + [
        rng.uniform(-5.0, 5.0) for _ in range(3)
    ]
    candidate2 = first["result"]["ensemble"][2]
    assert candidate2["random_draw_start"] == 6
    assert candidate2["component_transforms"][1]["random_translation_angstrom"] == pytest.approx(
        expected_second_attempt[3:]
    )

    continued = copy.deepcopy(request_base)
    continued["action_settings"]["random_stream_offset"] = 9 * 6
    second = reactive_complex.execute(
        "build_reactive_complex_ensemble",
        "internal_reactive_complex",
        continued,
    )
    assert second["status"] == "success"
    rng = random.Random(42)
    for _ in range(9 * 6):
        rng.random()
    expected_continued = [rng.uniform(0.0, 360.0) for _ in range(3)] + [
        rng.uniform(-5.0, 5.0) for _ in range(3)
    ]
    continued_candidate = second["result"]["ensemble"][1]
    assert continued_candidate["random_draw_start"] == 54
    assert continued_candidate["component_transforms"][1]["random_translation_angstrom"] == pytest.approx(
        expected_continued[3:]
    )


def test_post_mmff_forming_distance_gate_rejects_equal_boundary():
    settings = {
        "equilibrium_distance_scale": 1.6,
        "maximum_allowed_ratio": 1.5,
    }
    forming_bond = [{"atom_map_a": 1, "atom_map_b": 2}]
    threshold = 1.54 * 1.6 * 1.5
    at_boundary = _structure(distance=threshold, second_element="C")
    below_boundary = _structure(distance=threshold - 1.0e-6, second_element="C")
    assert not arche_ts_pipeline._author_forming_bonds_pass_distance_gate(
        at_boundary, forming_bond, settings
    )
    assert arche_ts_pipeline._author_forming_bonds_pass_distance_gate(
        below_boundary, forming_bond, settings
    )


def test_author_default_embedding_mode_is_not_etkdg_v3(monkeypatch):
    from rdkit.Chem import AllChem

    real_embed_multiple_confs = AllChem.EmbedMultipleConfs
    observed = {}

    def spy_embed_multiple_confs(molecule, **kwargs):
        observed.update(kwargs)
        return real_embed_multiple_confs(molecule, **kwargs)

    monkeypatch.setattr(AllChem, "ETKDGv3", lambda: pytest.fail("ETKDGv3 must not be used"))
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", spy_embed_multiple_confs)
    response = structure.execute(
        "generate_conformer_ensemble",
        "rdkit_etkdg",
        {
            "inputs": {"molecule": {"smiles": "CCO"}},
            "method_spec": {},
            "action_settings": {
                "num_conformers": 2,
                "random_seed": 42,
                "prune_rms_threshold_angstrom": 0.75,
                "embedding_mode": "author_default",
            },
        },
    )
    assert response["status"] == "success"
    assert response["result"]["generation_settings"]["embedding_mode"] == "author_default"
    assert observed == {
        "numConfs": 2,
        "pruneRmsThresh": 0.75,
        "randomSeed": 42,
    }


def test_fixed_diversity_filter_uses_greedy_input_order_not_butina():
    def conformer(length: float) -> dict:
        return {
            "smiles": "CCC",
            "atoms": [
                {"element": "C", "position_angstrom": [0.0, 0.0, 0.0]},
                {"element": "C", "position_angstrom": [length, 0.0, 0.0]},
                {"element": "C", "position_angstrom": [length + 1.0, 0.0, 0.0]},
            ],
            "bonds": [
                {"atom_index_a": 0, "atom_index_b": 1},
                {"atom_index_a": 1, "atom_index_b": 2},
            ],
            "charge": 0,
            "multiplicity": 1,
            "pbc": [False, False, False],
        }

    ensemble = {
        "ensemble": [
            {"conformer_id": "a", "structure": conformer(1.0)},
            {"conformer_id": "b", "structure": conformer(1.8)},
            {"conformer_id": "c", "structure": conformer(2.6)},
        ]
    }
    common = {
        "inputs": {"ensemble": ensemble},
        "method_spec": {},
        "action_settings": {
            "rmsd_cutoff_angstrom": 0.5,
            "atom_selection": "all",
            "prealign_conformers": False,
            "reorder_cluster_centers": True,
            "reorder_atoms": True,
        },
    }
    greedy = structure.execute(
        "cluster_conformers",
        "rdkit",
        {**common, "action_settings": {**common["action_settings"], "selection_mode": "greedy_input_order"}},
    )
    butina = structure.execute(
        "cluster_conformers",
        "rdkit",
        {**common, "action_settings": {**common["action_settings"], "selection_mode": "butina"}},
    )
    assert [cluster["representative_conformer_id"] for cluster in greedy["result"]["clusters"]] == [
        "a",
        "c",
    ]
    assert greedy["result"]["rejected_conformer_ids"] == ["b"]
    assert [cluster["representative_conformer_id"] for cluster in butina["result"]["clusters"]] == [
        "b",
    ]


def test_pipeline_result_artifacts_select_uma_endpoint_and_hide_coordinates(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    frames = [_structure() for _ in range(9)]
    result = {
        "workflow_id": "transition_state_path_search",
        "endpoint_pairs": [
            {
                "pair_id": "endpoint_pair_0001",
                "combination_id": "combination_0001",
                "source_candidate_id": "candidate_0001",
                "reactant": _structure(),
                "product": _structure(),
                "reactant_uma": _structure(distance=1.1),
                "product_uma": _structure(distance=1.1),
                "reactant_uma_converged": True,
                "product_uma_converged": True,
                "source_mmff_converged": True,
                "target_mmff_converged": True,
                "stage_statuses": {"mmff": "success", "uma": "success", "neb": "success"},
                "reaction_path": {"images": frames},
            }
        ],
        "transition_states_validated": False,
    }
    materialized, artifact_files = arche_ts_pipeline._materialize_pipeline_artifacts(result)
    pair = materialized["endpoint_pairs"][0]
    assert pair["endpoint_artifact_source"] == "uma"
    assert not {
        "combination_id",
        "source_candidate_id",
        "source_mmff_converged",
        "target_mmff_converged",
        "reactant_uma_converged",
        "product_uma_converged",
    } & set(pair)
    assert all("_uma.json" in item["path"] for item in artifact_files if item["semantic_type"] == "AtomicStructure")
    assert "reactant" not in pair and "product" not in pair
    assert pair["transition_state_guess_source"] is None
    assert pair["transition_state_guess_selection_rule_version"] is None
    assert pair["provenance"] is None
    serialized = json.dumps(materialized, ensure_ascii=False)
    assert "atom_map_number" not in serialized
    assert len(materialized["path_image_candidates"]) == 9


def test_pipeline_materializes_transition_state_guess_xyz_artifact(monkeypatch, tmp_path):
    """固定 wrapper 应把已有未验证 TS guess 单独交接为 XYZFile。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    guess = _structure(distance=1.25)
    result = {
        "workflow_id": "transition_state_path_search",
        "endpoint_pairs": [
            {
                "pair_id": "endpoint_pair_0001",
                "combination_id": "combination_0001",
                "source_candidate_id": "candidate_0001",
                "reactant": _structure(),
                "product": _structure(),
                "stage_statuses": {"uma": "success", "neb": "success"},
                "reactant_uma_converged": False,
                "product_uma_converged": False,
                "reaction_path": {
                    "images": [_structure(), guess, _structure(distance=1.5)],
                    "transition_state_guess": guess,
                    "transition_state_guess_source": "highest_interior_path_image",
                    "transition_state_guess_source_image_index": 1,
                    "transition_state_guess_selection_rule_version": (
                        "ts-guess-selection.v1"
                    ),
                },
            }
        ],
        "transition_states_validated": False,
    }

    materialized, artifact_files = arche_ts_pipeline._materialize_pipeline_artifacts(
        result
    )

    pair = materialized["endpoint_pairs"][0]
    assert pair["transition_state_guess_artifact"] is None
    assert pair["transition_state_guess_source"] == "highest_interior_path_image"
    assert pair["transition_state_guess_source_image_index"] == 1
    assert pair["transition_state_guess_selection_rule_version"] == (
        "ts-guess-selection.v1"
    )
    ts_files = [
        item
        for item in artifact_files
        if item["semantic_type"] == "XYZFile"
        and "transition_state_guess" in item["path"]
    ]
    assert len(ts_files) == 1
    assert ts_files[0]["media_type"] == "chemical/x-xyz"
    assert ts_files[0]["result_pointer"] == (
        "/endpoint_pairs/0/transition_state_guess_artifact"
    )


def test_pipeline_recovers_legacy_complete_path_when_guess_is_null(monkeypatch, tmp_path):
    """旧 reaction_path 形态的完整路径应防御性物化最高内部图像初猜。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    images = [_structure(), _structure(distance=1.25), _structure(distance=1.5)]
    result = {
        "workflow_id": "transition_state_path_search",
        "endpoint_pairs": [
            {
                "pair_id": "endpoint_pair_0001",
                "combination_id": "combination_0001",
                "source_candidate_id": "candidate_0001",
                "reactant": _structure(),
                "product": _structure(distance=1.5),
                "stage_statuses": {"uma": "success", "neb": "success"},
                "reactant_uma_converged": False,
                "product_uma_converged": False,
                "reaction_path": {
                    "images": images,
                    "energies_hartree": [-1.0, -0.8, -0.9],
                    "energy_image_indices": [0, 1, 2],
                    "highest_energy_image_index": 1,
                    "highest_interior_image_index": 1,
                    "transition_state_guess": None,
                },
            }
        ],
        "transition_states_validated": False,
    }

    materialized, artifact_files = arche_ts_pipeline._materialize_pipeline_artifacts(
        result
    )

    pair = materialized["endpoint_pairs"][0]
    assert pair["transition_state_guess_artifact"] is None
    assert pair["transition_state_guess_source"] == "highest_interior_path_image"
    assert pair["transition_state_guess_source_image_index"] == 1
    assert pair["transition_state_guess_selection_rule_version"] == (
        "ts-guess-selection.v1"
    )
    assert len(pair["path_image_artifacts"]) == 3
    ts_files = [
        item
        for item in artifact_files
        if item["semantic_type"] == "XYZFile"
        and "transition_state_guess" in item["path"]
    ]
    assert len(ts_files) == 1


def test_heavy_stage_preserves_actionable_ash_partial_diagnostics_without_private_paths():
    """ASH partial_success 的 message/warnings 进入 trace 且只保留脱敏正文。"""

    trace = []
    diagnostic_records = []

    def executor(action_id, backend_id, _request):
        if backend_id == "fairchem_uma":
            return {
                "status": "success",
                "result": {"structure": _structure(), "converged": True},
            }
        if backend_id == "ash_neb_uma":
            return {
                "status": "partial_success",
                "error": {
                    "code": "backend_unavailable",
                    "message": "ASH NEB needs a larger image budget",
                },
                "warnings": [
                    "Increase total_images before retrying.",
                    "checkpoint at /private/ash/knarr_MEP.xyz was retained.",
                ],
                "retryable": True,
                "result": {
                    "images": [_structure() for _ in range(9)],
                    "energies_hartree": [-1.0] * 9,
                    "energy_image_indices": list(range(9)),
                    "energy_index_complete": True,
                    "final_path_complete": True,
                    "interior_peak_candidate_valid": True,
                    "total_images": 9,
                },
            }
        raise AssertionError((action_id, backend_id))

    pairs, unavailable = arche_ts_pipeline._run_heavy_stages(
        [
            {
                "pair_id": "endpoint_pair_0001",
                "reactant": _structure(),
                "product": _structure(),
            }
        ],
        {},
        {
            "uma_model": "uma-s-1p1",
            "uma_task_name": "omol",
            "uma_fmax_ev_per_angstrom": 0.01,
            "uma_optimizer": "bfgs",
            "uma_max_steps": 500,
            "total_images": 9,
            "climb": False,
            "neb_max_steps": 300,
        },
        {"gpu_count": 1},
        executor,
        trace,
        diagnostic_records=diagnostic_records,
    )

    assert unavailable is None
    assert pairs[0]["stage_statuses"] == {"uma": "success", "neb": "success"}
    neb_trace = trace[-1]
    assert neb_trace["status"] == "partial_success"
    assert neb_trace["diagnostics"]["message"] == (
        "ASH NEB needs a larger image budget"
    )
    assert neb_trace["diagnostics"]["warnings"][0] == (
        "Increase total_images before retrying."
    )
    assert diagnostic_records == [
        {
            "pair_id": "endpoint_pair_0001",
            "stage": "neb",
            "status": "partial",
            "message": "ASH NEB needs a larger image budget",
            "warnings": [
                "Increase total_images before retrying.",
                "checkpoint at <path omitted> was retained.",
            ],
        }
    ]
    assert "/private/" not in json.dumps(neb_trace, ensure_ascii=False)
    assert "knarr_MEP.xyz" not in json.dumps(neb_trace, ensure_ascii=False)
