"""workflow contract 的 MiniChem-only 边界测试。"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import pytest

from chemistry_multiagent.utils.workflow_contract import (
    WorkflowContractError,
    apply_workflow_revision,
    canonical_workflow_diff,
    canonicalize_generation_output,
    effective_revision_diff,
    expected_outputs_for_tool,
    project_minichem_request_contracts,
    project_minichem_stage_tool_projection,
    project_workflow_refs,
    resolve_minichem_step_request,
    _validate_minichem_foreach_execution_layer,
    validate_canonical_workflow,
)
from chemistry_multiagent.memory.projections import project_workflow_for_revision
from tests.fakes.minichem_mcp import build_artifact_ref, build_catalog_snapshot


def _real_frozen_catalog_with_action_contracts() -> dict:
    """从 MiniChem 冻结目录和真实 discovery 合同构造 Planner 测试输入。"""

    from minichem_toolbox.catalog import catalog_snapshot
    from minichem_toolbox.discovery import inspect_action

    catalog = catalog_snapshot(include_health=False)
    contracts = []
    for action in catalog["actions"]:
        for backend_id in action["backend_ids"]:
            selected = inspect_action(
                action["id"],
                backend_id=backend_id,
                detail_level="contract",
                snapshot=catalog,
            )["selected_request_contract"]
            contracts.append(
                {
                    "action_id": action["id"],
                    "backend_id": backend_id,
                    "request_contract": selected,
                    "output_contract": deepcopy(selected.get("output_contract", {})),
                }
            )
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": contracts,
    }
    return catalog



def _catalog_with_open_execution() -> dict:
    """构造当前 V2 open-execution 合同使用的冻结能力快照。"""

    catalog = build_catalog_snapshot()
    catalog["open_execution"] = {
        "schema_version": 1,
        "native_software": [
            {
                "software_id": "xtb",
                "display_name": "xTB",
                "commands": [
                    {
                        "executable": "xtb",
                        "synopsis": "运行 xTB 单点计算",
                        "input_mode": "stdin",
                        "required_files": [],
                        "example_request": {},
                    }
                ],
            }
        ],
        "analysis_runtimes": [
            {"runtime": "python", "description": "标准 Python", "modules": []}
        ],
        "authoring_contract": {
            "path_prefix": "code/",
            "max_files": 4,
            "max_chars_per_file": 100000,
        },
    }
    return catalog


def _open_execution_workflow(execution_layer: str = "native_software") -> dict:
    """构造一个不含旧工作流执行器的 open-execution 步骤。"""

    if execution_layer == "native_software":
        request = {
            "software_id": "xtb",
            "executable": "xtb",
            "arguments": ["input.xyz"],
            "staged_inputs": [],
            "stdin_target": None,
            "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            "calculation_intent": "single_point",
            "label": "runtime-contract-test",
            "parent_job_id": None,
            "bindings": [],
        }
    elif execution_layer == "programmable_analysis":
        request = {
            "runtime": "python",
            "script_path": "code/analysis.py",
            "script_target": "code/agent_program.py",
            "arguments": [],
            "staged_inputs": [],
            "required_modules": [],
            "required_module_versions": {},
            "required_symbols": {},
            "inputs": [],
            "outputs": [],
            "execution_policy": "in_process_only",
            "external_execution": "native_job_runner_only",
            "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
            "label": "runtime-contract-test",
            "parent_job_id": None,
            "bindings": [],
        }
    else:
        raise ValueError(f"unsupported execution layer: {execution_layer}")
    return {
        "workflow_name": f"MiniChem {execution_layer} 工作流",
        "goal": "验证 open execution 作业身份恢复边界",
        "strategy_name": f"{execution_layer} 作业",
        "steps": [
            {
                "step_id": "job-step-1",
                "step_name": f"提交 {execution_layer}",
                "description": "提交一次可恢复作业并收集其终态输出。",
                "execution_layer": execution_layer,
                "request": request,
                "expected_outputs": {"execution_layer": execution_layer},
                "depends_on": [],
            }
        ],
    }

def _hypothesis() -> dict[str, str]:
    """返回只含当前正式科学正文的最小假设。"""

    return {
        "strategy_name": "MiniChem 结构转换",
        "core_claim": "苯甲醛结构可以转换为下游可读的结构文件。",
        "reasoning": "转换 Action 的完成状态和输出 Artifact 可以直接检验该判断。",
    }


def test_revision_accepts_complete_public_action_request_and_drops_null_placeholders() -> None:
    """公开 Action request 原样回填时，缺失 optional null 不得进入执行请求。"""

    catalog = build_catalog_snapshot(
        actions=[
            {
                "id": "convert_structure",
                "version": "1.0.0",
                "category": "structure_and_system",
                "description": "Convert an explicit molecular structure.",
                "backend_ids": ["openbabel"],
                "required_inputs": ["structure"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "AtomicStructure",
                "data_action": False,
            }
        ],
        backends=[
            {
                "id": "openbabel",
                "display_name": "Open Babel",
                "runtime": "native",
                "capabilities": ["convert_structure"],
                "health": {"status": "healthy"},
            }
        ],
    )
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "sections": {
                        "inputs": {
                            "allowed": ["structure", "nprocshared"],
                            "required": ["structure"],
                            "optional": [
                                {"name": "nprocshared", "type": "integer"}
                            ],
                        },
                        "method_spec": {"allowed": [], "required": []},
                        "action_settings": {"allowed": [], "required": []},
                        "resource_limits": {
                            "allowed": ["cpu_cores", "memory_mb", "gpu_count"],
                            "required": [],
                        },
                    }
                },
                "output_contract": {},
            }
        ],
    }
    workflow = {
        "workflow_name": "结构转换",
        "goal": "转换结构",
        "strategy_name": "直接转换",
        "steps": [
            {
                "step_id": "formal-s1",
                "step_name": "转换结构",
                "description": "生成结构文件",
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request": {
                    "inputs": {"structure": {"smiles": "O=Cc1ccccc1"}},
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": {"cpu_cores": 1, "memory_mb": 128},
                    "bindings": [],
                },
                "expected_outputs": {
                    "action_id": "convert_structure",
                    "action_version": "1.0.0",
                    "primary_output": "AtomicStructure",
                    "output_semantic_types": [],
                    "output_contract": {},
                },
                "depends_on": [],
            }
        ],
    }
    public = project_workflow_for_revision(workflow, catalog_snapshot=catalog)
    request = deepcopy(public["steps"][0]["request"])
    assert request["inputs"]["nprocshared"] is None
    request["inputs"]["structure"] = {"smiles": "CCO"}
    patch = {
        "operations": [
            {
                "op": "update_step",
                "target_step_ref": "s1",
                "changes": {"request": request},
            }
        ]
    }
    result = apply_workflow_revision(
        workflow,
        patch,
        catalog,
        lambda _kind: "revision-step-new",
    )
    final_request = result["after_workflow"]["steps"][0]["request"]
    assert final_request["inputs"]["structure"] == {"smiles": "CCO"}
    assert "nprocshared" not in final_request["inputs"]
    assert "software_id" not in final_request


def _generation_wire(*, with_dependency: bool = False) -> dict[str, object]:
    """构造模型可提交的 MiniChem Action/Backend/request 短引用。"""

    first_request = {
        "inputs": {"structure": {"smiles": "O=Cc1ccccc1"}},
        "method_spec": {},
        "action_settings": {},
        "resource_limits": {"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        "bindings": [],
    }
    steps: list[dict[str, object]] = [
        {
            "step_ref": "s1",
            "step_name": "转换苯甲醛结构",
            "description": "把显式分子结构转换为下游可读表示。",
            "action_id": "convert_structure",
            "backend_id": "openbabel",
            "request": first_request,
            "depends_on": [],
        }
    ]
    if with_dependency:
        steps.append(
            {
                "step_ref": "s2",
                "step_name": "再次转换结构",
                "description": "使用前一步生成的结构作为输入。",
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request": {
                    "inputs": {"structure": None},
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": {
                        "cpu_cores": 1,
                        "memory_mb": 128,
                        "gpu_count": 0,
                    },
                    "bindings": [
                        {
                            "source_step_ref": "s1",
                            "source_field": "artifact.semantic_type=AtomicStructure",
                            "target_field": "/inputs/structure",
                        }
                    ],
                },
                "depends_on": ["s1"],
            }
        )
    return {"steps": steps}


def _reaction_union_catalog() -> dict[str, object]:
    """构造使用联合输入类型的最小反应映射目录。"""

    catalog = build_catalog_snapshot(
        actions=[
            {
                "id": "map_reaction_atoms",
                "version": "1.0.0",
                "category": "reaction_mapping",
                "description": "Map atoms in a reaction.",
                "backend_ids": ["rxnmapper"],
                "required_inputs": ["reaction"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "MappedReaction",
                "output_semantic_types": ["MappedReaction"],
                "data_action": False,
                "batch_safe": False,
            }
        ],
        backends=[
            {
                "id": "rxnmapper",
                "display_name": "RXNMapper",
                "runtime": "python",
                "capabilities": ["map_reaction_atoms"],
                "health": {"status": "healthy"},
            }
        ],
    )
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "map_reaction_atoms",
                "backend_id": "rxnmapper",
                "request_contract": {
                    "required_contract": {
                        "inputs": [
                            {
                                "name": "reaction",
                                "type": (
                                    "reaction SMILES string | "
                                    "{reactants: array, products: array}"
                                ),
                            }
                        ],
                        "method_spec": [],
                        "action_settings": [],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": {},
                    "conditional_requirements": [],
                },
                "output_contract": None,
            }
        ],
    }
    return catalog


def _generation_wire_with_steps(count: int) -> dict[str, object]:
    """构造指定数量的相互独立 MiniChem Action 步骤。"""

    if count < 1:
        raise ValueError("测试工作流必须至少包含一个步骤")
    steps: list[dict[str, object]] = []
    for index in range(count):
        step = deepcopy(_generation_wire()["steps"][0])
        step["step_ref"] = f"s{index + 1}"
        step["step_name"] = f"转换结构 {index + 1}"
        steps.append(step)
    return {"steps": steps}


def _allocate_step_ids() -> tuple[Callable[[str], str], list[str]]:
    """返回记录正式身份分配的确定性分配器。"""

    calls: list[str] = []

    def allocate(kind: str) -> str:
        calls.append(kind)
        return f"step-formal-{len(calls)}"

    return allocate, calls


def test_minichem_generation_allocates_identity_and_derives_action_outputs() -> None:
    """生成入口只接受 Action 请求，正式身份和输出声明由程序补齐。"""

    catalog = build_catalog_snapshot()
    allocator, calls = _allocate_step_ids()
    workflow = canonicalize_generation_output(
        _generation_wire(),
        catalog,
        allocator,
        hypothesis=_hypothesis(),
    )

    assert calls == ["step_id"]
    assert workflow["workflow_name"] == "MiniChem 结构转换工作流"
    step = workflow["steps"][0]
    assert step["step_id"] == "step-formal-1"
    assert "step_ref" not in step
    assert step["action_id"] == "convert_structure"
    assert step["backend_id"] == "openbabel"
    assert step["expected_outputs"] == {
        "action_id": "convert_structure",
        "action_version": "1.0.0",
        "primary_output": "AtomicStructure",
        "output_semantic_types": [],
    }
    validate_canonical_workflow(workflow, catalog)


def test_minichem_generation_and_canonical_validation_accept_thirteen_steps() -> None:
    """合法的十三步 MiniChem 工作流可以完成规范化和正式校验。"""

    catalog = build_catalog_snapshot()
    allocator, _calls = _allocate_step_ids()
    workflow = canonicalize_generation_output(
        _generation_wire_with_steps(13),
        catalog,
        allocator,
        hypothesis=_hypothesis(),
    )

    assert len(workflow["steps"]) == 13
    validate_canonical_workflow(workflow, catalog)


def test_minichem_revision_accepts_more_than_twelve_steps() -> None:
    """修订插入第十三步时仍应通过完整合同校验。"""

    catalog = build_catalog_snapshot()
    allocator, _calls = _allocate_step_ids()
    before = canonicalize_generation_output(
        _generation_wire_with_steps(12),
        catalog,
        allocator,
        hypothesis=_hypothesis(),
    )
    projected = project_workflow_refs(before)["workflow"]
    inserted_step = deepcopy(projected["steps"][0])
    inserted_step.pop("step_ref")

    result = apply_workflow_revision(
        before,
        {
            "operations": [
                {
                    "op": "insert_step_after",
                    "new_step_ref": "n1",
                    "after_step_ref": "s12",
                    "step": inserted_step,
                }
            ]
        },
        catalog,
        lambda _kind: "step-formal-inserted",
    )

    assert len(result["after_workflow"]["steps"]) == 13
    validate_canonical_workflow(result["after_workflow"], catalog)


def test_pair_contract_rejects_missing_rdkit_random_seed_before_execution() -> None:
    """真实 inspect_action 合同要求 random_seed 时，Planner 预检必须先拒绝。"""

    catalog = build_catalog_snapshot()
    catalog["actions"][0]["id"] = "generate_3d_structure"
    catalog["actions"][0]["backend_ids"] = ["rdkit"]
    catalog["backends"][0]["id"] = "rdkit"
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "generate_3d_structure",
                "backend_id": "rdkit",
                "request_contract": {
                    "required_contract": {
                        "inputs": [{"name": "structure"}],
                        "method_spec": [],
                        "action_settings": [{"name": "random_seed"}],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": {},
                    "conditional_requirements": [],
                },
                "output_contract": None,
            }
        ],
    }
    wire = _generation_wire()
    wire["steps"][0]["action_id"] = "generate_3d_structure"
    wire["steps"][0]["backend_id"] = "rdkit"
    with pytest.raises(WorkflowContractError, match="random_seed"):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-1",
            hypothesis=_hypothesis(),
        )


def test_pair_contract_rejects_wrong_static_field_type_before_execution() -> None:
    """required_contract 的基本 type 提示必须在执行前生效。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [{"name": "structure"}],
                        "method_spec": [],
                        "action_settings": [{"name": "random_seed", "type": "integer"}],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": [],
                    "conditional_requirements": [],
                },
                "output_contract": None,
            }
        ],
    }
    wire = _generation_wire()
    wire["steps"][0]["request"]["action_settings"] = {"random_seed": "42"}
    with pytest.raises(WorkflowContractError, match="random_seed.*integer"):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-1",
            hypothesis=_hypothesis(),
        )


def test_pair_contract_accepts_integer_array_before_execution() -> None:
    """``array[integer]`` 必须按数组校验，不能误当作单个整数。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [{"name": "structure"}],
                        "method_spec": [],
                        "action_settings": [
                            {"name": "atom_indices", "type": "array[integer]"}
                        ],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": [],
                    "conditional_requirements": [],
                },
                "output_contract": None,
            }
        ],
    }
    wire = _generation_wire()
    wire["steps"][0]["request"]["action_settings"] = {
        "atom_indices": [0, 1]
    }

    canonicalize_generation_output(
        wire,
        catalog,
        lambda _kind: "step-formal-1",
        hypothesis=_hypothesis(),
    )

    wire["steps"][0]["request"]["action_settings"] = {"atom_indices": 0}
    with pytest.raises(WorkflowContractError, match="atom_indices.*array"):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-2",
            hypothesis=_hypothesis(),
        )


def _typed_binding_placeholder_catalog() -> dict[str, object]:
    """构造含数值型必填绑定目标的最小 Action 合同。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [{"name": "structure"}],
                        "method_spec": [],
                        "action_settings": [
                            {"name": "start_value", "type": "number"},
                            {"name": "end_value", "type": "number"},
                            {"name": "atom_indices", "type": "array[integer]"},
                        ],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": {},
                    "conditional_requirements": [],
                },
                "output_contract": {
                    "result_fields": ["computed_start", "computed_index"]
                },
            }
        ],
    }
    return catalog


def _typed_binding_placeholder_wire() -> dict[str, object]:
    """构造上游结果填入下游数值字段的两步生成请求。"""

    wire = _generation_wire(with_dependency=True)
    first_request = wire["steps"][0]["request"]
    first_request["action_settings"] = {
        "start_value": 0.0,
        "end_value": 1.0,
        "atom_indices": [0, 1],
    }
    second_request = wire["steps"][1]["request"]
    second_request["inputs"]["structure"] = {"smiles": "CCO"}
    second_request["action_settings"] = {
        "start_value": None,
        "end_value": 2.0,
        "atom_indices": [None, 1],
    }
    second_request["bindings"] = [
        {
            "source_step_ref": "s1",
            "source_field": "/result/computed_start",
            "target_field": "action_settings.start_value",
        },
        {
            "source_step_ref": "s1",
            "source_field": "/result/computed_index",
            "target_field": "action_settings.atom_indices.0",
        },
    ]
    return wire


def test_typed_binding_target_accepts_null_placeholder_and_resolves_value() -> None:
    """数值型字段仅在被精确绑定时可留空，并由完成结果覆盖。"""

    catalog = _typed_binding_placeholder_catalog()
    wire = _typed_binding_placeholder_wire()
    workflow = canonicalize_generation_output(
        wire,
        catalog,
        _allocate_two_step_ids(),
        hypothesis=_hypothesis(),
    )
    resolved = resolve_minichem_step_request(
        workflow["steps"][1],
        {
            workflow["steps"][0]["step_id"]: {
                "raw_output": {
                    "result": {"computed_start": 0.25, "computed_index": 3}
                }
            }
        },
    )

    assert resolved["action_settings"]["start_value"] == 0.25
    assert resolved["action_settings"]["atom_indices"] == [3, 1]
    assert "bindings" not in resolved


def test_typed_required_null_without_exact_binding_is_rejected() -> None:
    """必填数值字段的空值不能借另一条 binding 绕过合同。"""

    catalog = _typed_binding_placeholder_catalog()
    wire = _typed_binding_placeholder_wire()
    wire["steps"][1]["request"]["bindings"][0]["target_field"] = (
        "action_settings.end_value"
    )

    with pytest.raises(WorkflowContractError, match="start_value.*null.*binding"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


def test_pair_contract_rejects_object_for_integer_or_null_field() -> None:
    """可识别的 ``integer | null`` 联合类型不能放行 object。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [{"name": "structure"}],
                        "method_spec": [],
                        "action_settings": [
                            {"name": "optional_charge", "type": "integer | null"}
                        ],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": [],
                    "conditional_requirements": [],
                },
                "output_contract": None,
            }
        ],
    }
    wire = _generation_wire()
    wire["steps"][0]["request"]["action_settings"] = {
        "optional_charge": {"not": "an integer"}
    }

    with pytest.raises(WorkflowContractError, match="optional_charge.*integer"):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-1",
            hypothesis=_hypothesis(),
        )


@pytest.mark.parametrize(
    "reaction",
    [
        "CCO>>CC=O",
        {"reactants": ["CCO"], "products": ["CC=O"]},
    ],
)
def test_pair_contract_accepts_reaction_union_shapes_before_execution(
    reaction: object,
) -> None:
    """反应联合类型的字符串和显式 reactants/products 形状都应通过规范化。"""

    catalog = _reaction_union_catalog()
    wire = _generation_wire()
    wire["steps"][0]["action_id"] = "map_reaction_atoms"
    wire["steps"][0]["backend_id"] = "rxnmapper"
    wire["steps"][0]["request"]["inputs"] = {"reaction": reaction}

    workflow = canonicalize_generation_output(
        wire,
        catalog,
        lambda _kind: "step-formal-1",
        hypothesis=_hypothesis(),
    )

    assert workflow["steps"][0]["request"]["inputs"]["reaction"] == reaction
    validate_canonical_workflow(workflow, catalog)


def test_pair_contract_requires_and_rejects_unknown_component_backend_roles() -> None:
    """component_backends 只允许合同声明的角色，并要求 required 角色到位。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [{"name": "structure"}],
                        "method_spec": [],
                        "action_settings": [],
                        "resource_limits": [],
                        "component_backends": ["calculator"],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": [],
                    "conditional_requirements": [],
                },
                "output_contract": None,
            }
        ],
    }
    wire = _generation_wire()
    with pytest.raises(WorkflowContractError, match="calculator"):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-1",
            hypothesis=_hypothesis(),
        )
    wire["steps"][0]["request"]["component_backends"] = {
        "calculator": "backend",
        "other": "backend",
    }
    with pytest.raises(WorkflowContractError, match="未声明角色"):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-1",
            hypothesis=_hypothesis(),
        )


def test_minichem_generation_projects_and_restores_short_dependency_refs() -> None:
    """正式 step_id 与模型短引用之间的转换不改变请求内容或依赖顺序。"""

    catalog = build_catalog_snapshot()
    allocator, _calls = _allocate_step_ids()
    workflow = canonicalize_generation_output(
        _generation_wire(with_dependency=True),
        catalog,
        allocator,
        hypothesis=_hypothesis(),
    )
    before = deepcopy(workflow)

    projection = project_workflow_refs(workflow)

    assert workflow == before
    assert projection["ref_map"]["step_refs"] == {
        "s1": "step-formal-1",
        "s2": "step-formal-2",
    }
    projected_steps = projection["workflow"]["steps"]
    assert [step["step_ref"] for step in projected_steps] == ["s1", "s2"]
    assert projected_steps[1]["depends_on"] == ["s1"]
    assert projected_steps[1]["request"]["bindings"][0]["source_step_ref"] == "s1"
    assert all("step_id" not in step for step in projected_steps)


def test_minichem_binding_requires_json_pointers_and_explicit_data_edges() -> None:
    """Planner 不得用语义标签作路径；显式数据边可覆盖模型占位值。"""

    catalog = build_catalog_snapshot()
    allocator, _calls = _allocate_step_ids()
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0][
        "source_field"
    ] = "AtomicStructure"
    with pytest.raises(WorkflowContractError, match="JSON Pointer"):
        canonicalize_generation_output(
            wire, catalog, allocator, hypothesis=_hypothesis()
        )

    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["inputs"]["structure"] = {"smiles": "N"}
    workflow = canonicalize_generation_output(
        wire, catalog, allocator, hypothesis=_hypothesis()
    )
    assert workflow["steps"][1]["request"]["inputs"]["structure"] == {
        "smiles": "N"
    }


def test_minichem_foreach_is_canonicalized_and_resolved_without_fixed_pipeline() -> None:
    """Planner 可把任意上游数组映射到原子 Action，不编码化学步骤顺序。"""

    catalog = build_catalog_snapshot()
    # 动态数组是 Action 公开 result 的一部分；冻结 pair 合同后只能通过
    # /result/candidates 读取，不能把执行证据内部的 raw_output 路径写进工作流。
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {},
                "output_contract": {"result_fields": ["candidates"]},
            }
        ],
    }
    wire = _generation_wire()
    wire["steps"].append(
        {
            "step_ref": "s2",
            "step_name": "逐项转换候选",
            "description": "对上一步动态产生的每个候选调用相同原子 Action。",
            "action_id": "convert_structure",
            "backend_id": "openbabel",
            "request": {
                "inputs": {"structure": None},
                "method_spec": {},
                "action_settings": {},
                "resource_limits": {
                    "cpu_cores": 1,
                    "memory_mb": 128,
                    "gpu_count": 0,
                },
                "bindings": [],
                "foreach": {
                    "source_step_ref": "s1",
                    "source_field": "/result/candidates",
                    "item_field": "/structure",
                    "item_id_field": "/candidate_id",
                    "target_field": "/inputs/structure",
                    "max_items": 8,
                },
            },
            "depends_on": ["s1"],
        }
    )
    allocator, _calls = _allocate_step_ids()
    workflow = canonicalize_generation_output(
        wire, catalog, allocator, hypothesis=_hypothesis()
    )
    second = workflow["steps"][1]
    assert second["request"]["foreach"]["source_step_ref"] == "step-formal-1"

    resolved = resolve_minichem_step_request(
        second,
        {
            "step-formal-1": {
                "raw_output": {
                    "result": {
                        "candidates": [
                            {"candidate_id": "re", "structure": {"smiles": "C"}},
                            {"candidate_id": "si", "structure": {"smiles": "N"}},
                        ]
                    }
                }
            }
        },
    )
    assert [item["item_id"] for item in resolved["_foreach_items"]] == ["re", "si"]
    assert [
        item["request"]["inputs"]["structure"]["smiles"]
        for item in resolved["_foreach_items"]
    ] == ["C", "N"]


def test_predefined_foreach_output_can_feed_later_predefined_foreach() -> None:
    """同步 Action 的批量聚合结果应能按 item_id 和 typed Artifact 继续展开。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {},
                "output_contract": {
                    "artifact_semantic_types": ["AtomicStructure"],
                    "result_fields": ["candidates"],
                },
            }
        ],
    }
    wire = _generation_wire()
    for step_ref, source_step_ref, source_field, item_field in (
        ("s2", "s1", "/result/candidates", "/structure"),
        (
            "s3",
            "s2",
            "/result/items",
            "artifact.semantic_type=AtomicStructure",
        ),
    ):
        wire["steps"].append(
            {
                "step_ref": step_ref,
                "step_name": f"逐项转换 {step_ref}",
                "description": "按稳定 item 身份继续转换上游结构。",
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request": {
                    "inputs": {"structure": None},
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": {
                        "cpu_cores": 1,
                        "memory_mb": 128,
                        "gpu_count": 0,
                    },
                    "bindings": [],
                    "foreach": {
                        "source_step_ref": source_step_ref,
                        "source_field": source_field,
                        "item_field": item_field,
                        "item_id_field": "/item_id",
                        "target_field": "/inputs/structure",
                        "max_items": 2,
                    },
                },
                "depends_on": [source_step_ref],
            }
        )

    invalid_wire = deepcopy(wire)
    invalid_wire["steps"][2]["request"]["foreach"]["item_field"] = (
        "artifact.semantic_type=GaussianInputDeck"
    )
    with pytest.raises(WorkflowContractError, match="GaussianInputDeck.*AtomicStructure"):
        canonicalize_generation_output(
            invalid_wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )

    for field_name, invalid_value, error_text in (
        ("source_field", "/result/candidates", "/result/items"),
        ("item_id_field", "/candidate_id", "/item_id"),
    ):
        invalid_wire = deepcopy(wire)
        invalid_wire["steps"][2]["request"]["foreach"][field_name] = (
            invalid_value
        )
        with pytest.raises(WorkflowContractError, match=error_text):
            canonicalize_generation_output(
                invalid_wire,
                catalog,
                _allocate_two_step_ids(),
                hypothesis=_hypothesis(),
            )

    workflow = canonicalize_generation_output(
        wire,
        catalog,
        _allocate_two_step_ids(),
        hypothesis=_hypothesis(),
    )
    artifact = build_artifact_ref(
        artifact_id="structure-item-1",
        semantic_type="AtomicStructure",
        media_type="application/json",
        path="artifacts/structure-item-1.json",
        producer_action="convert_structure",
        producer_backend="openbabel",
    )
    resolved = resolve_minichem_step_request(
        workflow["steps"][2],
        {
            workflow["steps"][1]["step_id"]: {
                "raw_output": {
                    "result": {
                        "items": [
                            {
                                "item_id": "candidate-1",
                                "result": {
                                    "status": "completed",
                                    "output_artifacts": [artifact],
                                },
                            }
                        ]
                    }
                },
                "output_artifacts": [artifact],
            }
        },
    )

    assert resolved["_foreach_items"][0]["item_id"] == "candidate-1"
    assert (
        resolved["_foreach_items"][0]["request"]["inputs"]["structure"]
        == artifact
    )


def test_foreach_execution_layer_matrix_matches_runtime_fanout_branches() -> None:
    """确定性合同保留 predefined fan-out，并拒绝没有逐项执行分支的层。"""

    request = {"foreach": {}}
    _validate_minichem_foreach_execution_layer(
        request,
        "predefined_actions",
        "steps[0].request",
    )
    _validate_minichem_foreach_execution_layer(
        {"foreach": {}, "software_id": "gaussian", "executable": "g16"},
        "native_software",
        "steps[1].request",
    )
    for execution_layer, layer_request in (
        ("programmable_analysis", request),
        (
            "native_software",
            {"foreach": {}, "software_id": "xtb", "executable": "xtb"},
        ),
    ):
        with pytest.raises(WorkflowContractError, match="只允许 native Gaussian"):
            _validate_minichem_foreach_execution_layer(
                layer_request,
                execution_layer,
                "steps[0].request",
            )


def test_fixed_ts_handoff_uses_only_present_guesses_through_planner_validator_and_resolver() -> None:
    """冻结 TS 合同经 Planner、工作流校验后只展开有初猜的 endpoint pair。"""

    catalog = _real_frozen_catalog_with_action_contracts()
    planner_view = {
        "planner_contract_projection": {
            "pairs": project_minichem_request_contracts(catalog),
        }
    }
    ts_pair = next(
        pair
        for pair in planner_view["planner_contract_projection"]["pairs"]
        if pair["action_id"] == "search_transition_state_path"
        and pair["backend_id"] == "transition_state_path_search"
    )
    assert ts_pair["output_contract"]["batch_handoff"]["only_present_items"] is True

    actions = {item["id"]: item for item in catalog["actions"]}
    contracts = {
        (item["action_id"], item["backend_id"]): item
        for item in catalog["action_contracts"]["contracts"]
    }

    def expected_outputs(action_id: str, backend_id: str) -> dict:
        action = actions[action_id]
        return {
            "action_id": action_id,
            "action_version": action["version"],
            "primary_output": action["primary_output"],
            "output_semantic_types": list(action["output_semantic_types"]),
            "output_contract": deepcopy(
                contracts[(action_id, backend_id)]["output_contract"]
            ),
        }

    workflow = {
        "workflow_name": "固定 TS 初猜交接",
        "goal": "只把可用 TS 初猜交给 Gaussian 输入转换",
        "strategy_name": "逐 endpoint pair 交接",
        "steps": [
            {
                "step_id": "step-formal-1",
                "step_name": "固定 TS 路径搜索",
                "description": "生成 endpoint pair 和未验证 TS 初猜。",
                "action_id": "search_transition_state_path",
                "backend_id": "transition_state_path_search",
                "request": {
                        "inputs": {
                            "reactants": ["CCO", "O=C=O"],
                            "products": ["CCOC(=O)O"],
                        },
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": {"gpu_count": 1},
                    "bindings": [],
                },
                "expected_outputs": expected_outputs(
                    "search_transition_state_path",
                    "transition_state_path_search",
                ),
                "depends_on": [],
            },
            {
                "step_id": "step-formal-2",
                "step_name": "转换 TS 初猜",
                "description": "仅对存在 TS 初猜的 endpoint pair 生成 Gaussian 输入。",
                "action_id": "xyz_to_gjf",
                "backend_id": "format_conversion",
                "request": {
                    "inputs": {"input_xyz_path": None},
                    "method_spec": {},
                    "action_settings": {},
                    "resource_limits": {
                        "cpu_cores": 1,
                        "memory_mb": 4096,
                        "gpu_count": 0,
                    },
                    "bindings": [],
                    "foreach": {
                        "source_step_ref": "step-formal-1",
                        "source_field": "/result/endpoint_pairs",
                        "item_field": "/transition_state_guess_artifact",
                        "item_id_field": "/pair_id",
                        "target_field": "/inputs/input_xyz_path",
                        "max_items": 32,
                        "only_present_items": True,
                    },
                },
                "expected_outputs": expected_outputs(
                    "xyz_to_gjf", "format_conversion"
                ),
                "depends_on": ["step-formal-1"],
            },
        ],
    }
    validate_canonical_workflow(workflow, catalog)

    endpoint_pairs = [
        {
            "pair_id": "pair-present-1",
            "transition_state_guess_artifact": {"artifact_id": "ts-1"},
            "path_image_artifacts": [{"candidate_id": f"image-1-{i}"} for i in range(9)],
        },
        {
            "pair_id": "pair-present-2",
            "transition_state_guess_artifact": {"artifact_id": "ts-2"},
            "path_image_artifacts": [{"candidate_id": f"image-2-{i}"} for i in range(9)],
        },
        {
            "pair_id": "pair-null",
            "transition_state_guess_artifact": None,
            "path_image_artifacts": [],
        },
    ]
    endpoint_pairs.extend(
        {
            "pair_id": f"pair-missing-{i}",
            "path_image_artifacts": [{"candidate_id": f"image-missing-{i}-{j}"} for j in range(9)],
        }
        for i in range(31)
    )
    completed = {
        "step-formal-1": {
            "raw_output": {
                "status": "partial_success",
                "result": {"endpoint_pairs": endpoint_pairs},
            }
        }
    }
    resolved = resolve_minichem_step_request(workflow["steps"][1], completed)
    assert [item["item_id"] for item in resolved["_foreach_items"]] == [
        "pair-present-1",
        "pair-present-2",
    ]
    assert [
        item["request"]["inputs"]["input_xyz_path"]["artifact_id"]
        for item in resolved["_foreach_items"]
    ] == ["ts-1", "ts-2"]

    too_many_present = deepcopy(completed)
    too_many_present["step-formal-1"]["raw_output"]["result"]["endpoint_pairs"] = [
        {
            "pair_id": f"pair-{i}",
            "transition_state_guess_artifact": {"artifact_id": f"ts-{i}"},
        }
        for i in range(33)
    ]
    with pytest.raises(WorkflowContractError, match="不能超过 32"):
        resolve_minichem_step_request(workflow["steps"][1], too_many_present)

    malformed = deepcopy(completed)
    malformed["step-formal-1"]["raw_output"]["result"]["endpoint_pairs"] = [
        "not-an-endpoint-pair"
    ]
    with pytest.raises(WorkflowContractError, match="无法继续解析"):
        resolve_minichem_step_request(workflow["steps"][1], malformed)


def test_predefined_action_public_result_pointer_reads_verified_raw_output() -> None:
    """公开 ``/result`` 指针映射到规范步骤保存的 MCP 原始结果。"""

    step = {
        "request": {
            "inputs": {"route_section": None},
            "method_spec": {},
            "action_settings": {},
            "resource_limits": {},
            "bindings": [
                {
                    "source_step_ref": "codegen-step",
                    "source_field": "/result/gaussian_code",
                    "target_field": "inputs.route_section",
                }
            ],
        }
    }
    resolved = resolve_minichem_step_request(
        step,
        {
            "codegen-step": {
                "raw_output": {
                    "result": {"gaussian_code": "#p B3LYP/6-31G(d) opt freq"}
                }
            }
        },
    )

    assert resolved["inputs"]["route_section"] == "#p B3LYP/6-31G(d) opt freq"
    assert "bindings" not in resolved


def test_foreach_resolves_single_typed_artifact_as_one_item() -> None:
    """typed Artifact selector 在 foreach 中应按单项序列展开。"""

    artifact = build_artifact_ref(
        artifact_id="art_foreach_0001",
        semantic_type="AtomicStructure",
    )
    step = {
        "request": {
            "inputs": {"structure": None},
            "bindings": [],
            "foreach": {
                "source_step_ref": "source-step",
                "source_field": "artifact.semantic_type=AtomicStructure",
                "target_field": "/inputs/structure",
                "max_items": 4,
            },
        }
    }

    resolved = resolve_minichem_step_request(
        step,
        {"source-step": {"output_artifacts": [artifact]}},
    )

    assert len(resolved["_foreach_items"]) == 1
    assert resolved["_foreach_items"][0]["request"]["inputs"]["structure"] == artifact


@pytest.mark.parametrize(
    "selector",
    [
        "job.stdout",
        "/raw_output/result/candidates",
        "/result/undeclared_candidates",
        "/execution_result/steps/0/raw_output/result/candidates",
    ],
)
def test_predefined_action_rejects_internal_or_unknown_source_selectors(
    selector: str,
) -> None:
    """predefined Action 只能使用冻结合同公开的 selector。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {},
                "output_contract": {"result_fields": ["candidates"]},
            }
        ],
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = selector

    with pytest.raises(WorkflowContractError, match="selector|输出合同|job.stdout|raw_output"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


@pytest.mark.parametrize("execution_layer", ["native_software", "programmable_analysis"])
def test_open_execution_job_stdout_binding_remains_legal(execution_layer: str) -> None:
    """native 和 programmable 作业仍可把已完成作业的 stdout 作为绑定来源。"""

    catalog = _catalog_with_open_execution()
    workflow = _open_execution_workflow(execution_layer)
    source = workflow["steps"][0]
    consumer = deepcopy(source)
    consumer["step_id"] = "job-step-2"
    consumer["step_name"] = "消费 stdout"
    consumer["description"] = "把上游作业 stdout 交给下游作业。"
    consumer["depends_on"] = [source["step_id"]]
    request = consumer["request"]
    if execution_layer == "native_software":
        request["staged_inputs"] = [
            {"source_path": None, "target_path": "inputs/upstream.log"}
        ]
        target_field = "staged_inputs.0.source_path"
    else:
        request["inputs"] = [
            {
                "name": "upstream_log",
                "source_path": None,
                "target_path": "inputs/upstream.log",
            }
        ]
        target_field = "inputs.0.source_path"
    request["bindings"] = [
        {
            "source_step_ref": source["step_id"],
            "source_field": "job.stdout",
            "target_field": target_field,
        }
    ]
    workflow["steps"].append(consumer)

    validate_canonical_workflow(workflow, catalog)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda wire: wire["steps"][0].update(
            {"tool": "legacy_tool", "job_type": "none"}
        ),
        lambda wire: wire["steps"][0].update(
            {"execution_layer": "predefined_action"}
        ),
        lambda wire: wire["steps"][0]["request"].update(
            {"operation_id": "planner-supplied-id"}
        ),
        lambda wire: wire["steps"][0].update({"backend_id": "undeclared-backend"}),
    ],
)
def test_minichem_generation_rejects_legacy_fields_and_invalid_backend(mutator) -> None:
    """旧 ToolSpec 字段、执行身份和未登记 Backend 不能重新进入生成协议。"""

    catalog = build_catalog_snapshot()
    wire = _generation_wire()
    mutator(wire)

    with pytest.raises(WorkflowContractError):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-1",
            hypothesis=_hypothesis(),
        )


def test_backend_request_contract_required_fields_are_enforced() -> None:
    """冻结的 inspect_action required_contract 必须由 Planner 和执行前校验共享。"""

    catalog = build_catalog_snapshot()
    catalog["actions"][0]["request_contracts"] = {
        "openbabel": {
            "required_contract": {
                "inputs": [{"name": "structure", "required": True}],
                "method_spec": [],
                "action_settings": [{"name": "format", "required": True}],
            }
        }
    }
    wire = _generation_wire()

    with pytest.raises(WorkflowContractError, match="action_settings.*format"):
        canonicalize_generation_output(
            wire,
            catalog,
            lambda _kind: "step-formal-1",
            hypothesis=_hypothesis(),
        )


def test_minichem_projection_rejects_provider_ids_catalog_alias() -> None:
    """MiniChem v7 目录必须使用 backend_ids，不能回退到 provider_ids。"""

    catalog = build_catalog_snapshot()
    action = catalog["actions"][0]
    action["provider_ids"] = action.pop("backend_ids")

    with pytest.raises(WorkflowContractError, match="backend_ids"):
        project_minichem_stage_tool_projection(catalog)


def test_minichem_generation_ignores_legacy_action_job_type() -> None:
    """Action 缺少 execution_layer 时默认 predefined_actions，不读取旧 job_type。"""

    catalog = build_catalog_snapshot()
    catalog["actions"][0]["job_type"] = "native_software"
    workflow = canonicalize_generation_output(
        _generation_wire(),
        catalog,
        lambda _kind: "step-formal-1",
        hypothesis=_hypothesis(),
    )

    assert workflow["steps"][0]["action_id"] == "convert_structure"
    assert "execution_layer" not in workflow["steps"][0]


def test_minichem_revision_rejects_old_input_and_check_fields() -> None:
    """修订补丁只能改变 MiniChem 步骤，不能夹带旧 inputs/check_bindings 协议。"""

    catalog = _catalog_with_open_execution()
    before = _open_execution_workflow("native_software")
    patch = {
        "operations": [
            {
                "op": "update_step",
                "target_step_ref": "s1",
                "changes": {"inputs": "legacy literal/bindings"},
            }
        ],
        "check_bindings": [],
    }

    with pytest.raises(WorkflowContractError, match="operations|字段"):
        apply_workflow_revision(
            before,
            patch,
            catalog,
            lambda _kind: "unused-step-id",
        )


def test_old_tool_catalog_is_rejected_by_all_workflow_entrypoints() -> None:
    """生成、校验和投影入口都应明确拒绝旧 ToolSpec 目录或步骤。"""

    old_catalog = {"tools": []}
    with pytest.raises(WorkflowContractError, match="MiniChem"):
        canonicalize_generation_output(
            {},
            old_catalog,
            lambda _kind: "unused-id",
            hypothesis={},
        )
    with pytest.raises(WorkflowContractError, match="MiniChem"):
        validate_canonical_workflow({}, old_catalog)
    with pytest.raises(WorkflowContractError, match="旧 ToolSpec"):
        project_workflow_refs(
            {
                "workflow_name": "旧工作流",
                "goal": "旧目标",
                "strategy_name": "旧策略",
                "steps": [{"step_id": "step-1", "tool": "legacy"}],
            }
        )


def test_legacy_output_helper_is_explicitly_closed() -> None:
    """controller 尚未迁移的输出辅助名只能给出明确停用错误。"""

    with pytest.raises(WorkflowContractError, match="ToolSpec.*停用"):
        expected_outputs_for_tool({"name": "legacy"})


def test_minichem_revision_diff_ignores_annotations_but_tracks_request_changes() -> None:
    """MiniChem 修订差异忽略描述文字，却记录真实 request 变化。"""

    before = _open_execution_workflow("native_software")
    annotated = deepcopy(before)
    annotated["steps"][0]["description"] = "仅修改说明"
    assert effective_revision_diff(before, annotated) is False

    changed = deepcopy(before)
    changed["steps"][0]["request"]["label"] = "changed-label"
    assert effective_revision_diff(before, changed) is True
    diff = canonical_workflow_diff(before, changed)
    assert diff["operations"][0]["op"] == "update_step"
    assert diff["operations"][0]["changes"]["request"]["label"] == "changed-label"


def test_minichem_revision_returns_after_workflow_and_rejects_annotation_only_patch() -> None:
    """MiniChem 修订应原子返回规范差异，并拒绝只改说明的补丁。"""

    catalog = _catalog_with_open_execution()
    before = _open_execution_workflow("native_software")
    changed_request = deepcopy(before["steps"][0]["request"])
    changed_request["label"] = "changed-label"

    result = apply_workflow_revision(
        before,
        {
            "operations": [
                {
                    "op": "update_step",
                    "target_step_ref": "s1",
                    "changes": {"request": changed_request},
                }
            ]
        },
        catalog,
        lambda _kind: "allocated-step-id",
    )
    assert result["after_workflow"]["steps"][0]["request"]["label"] == "changed-label"
    assert result["normalized_patch"]["operations"]

    with pytest.raises(WorkflowContractError, match="effective"):
        apply_workflow_revision(
            before,
            {
                "operations": [
                    {
                        "op": "update_step",
                        "target_step_ref": "s1",
                        "changes": {"description": "只改说明"},
                    }
                ]
            },
            catalog,
            lambda _kind: "unused-step-id",
        )


def _catalog_with_artifact_only_output_contract() -> dict[str, object]:
    """返回只承诺 typed artifact handoff 的 Action 合同目录。"""

    catalog = build_catalog_snapshot()
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [],
                        "method_spec": [],
                        "action_settings": [],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": {},
                    "conditional_requirements": [],
                },
                "output_contract": {
                    "primary_output": "AtomicStructure",
                    "result_envelope": "ActionResult",
                    "artifact_handoff": "typed ArtifactRef",
                },
            }
        ],
    }
    return catalog


def _allocate_two_step_ids() -> Callable[[str], str]:
    """返回两个步骤测试所需的互异正式身份分配器。"""

    counter = 0

    def allocate(_kind: str) -> str:
        nonlocal counter
        counter += 1
        return f"step-formal-{counter}"

    return allocate


def test_output_note_keeps_pointers_declared_only_by_result_fields() -> None:
    """只有 result_fields、没有 result_schema 时也应保留同字段公开 Pointer。"""

    catalog = _catalog_with_artifact_only_output_contract()
    output_contract = catalog["action_contracts"]["contracts"][0][
        "output_contract"
    ]
    output_contract["result_fields"] = ["matches"]
    output_contract["output_note"] = (
        "首项使用 /result/matches；诊断文件位于 /srv/private/result.log。"
    )

    projected = project_minichem_stage_tool_projection(catalog)
    action = next(item for item in projected if item["action_id"] == "convert_structure")
    pair = next(
        item
        for item in action["backend_request_contracts"]
        if item["backend_id"] == "openbabel"
    )
    note = pair["output_contract"]["output_note"]

    assert "/result/matches" in note
    assert "/srv/private/result.log" not in note
    assert "[路径已隐藏]" in note


def test_batch_handoff_keeps_item_selectors_from_output_schema_alias() -> None:
    """output_schema 别名声明的数组项字段应与 result_schema 一样可见。"""

    catalog = _catalog_with_artifact_only_output_contract()
    output_contract = catalog["action_contracts"]["contracts"][0][
        "output_contract"
    ]
    output_contract["output_schema"] = {
        "type": "object",
        "properties": {
            "ensemble": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "structure_artifact": {"type": "object"},
                        "candidate_id": {"type": "string"},
                    },
                },
            }
        },
    }
    output_contract["batch_handoff"] = {
        "source_field": "/result/ensemble",
        "item_field": "/structure_artifact",
        "item_id_field": "/candidate_id",
        "target_field": "/inputs/structure",
        "max_items": 4,
        "preserve_order": True,
        "only_present_items": True,
    }

    projected = project_minichem_stage_tool_projection(catalog)
    action = next(item for item in projected if item["action_id"] == "convert_structure")
    pair = next(
        item
        for item in action["backend_request_contracts"]
        if item["backend_id"] == "openbabel"
    )

    assert pair["output_contract"]["batch_handoff"] == output_contract[
        "batch_handoff"
    ]


@pytest.mark.parametrize(
    "selector",
    [
        "/result/matches/0",
        "/result/matches/0/1",
    ],
)
def test_action_contract_accepts_array_items_declared_by_result_schema(
    selector: str,
) -> None:
    """结果 Schema 声明的数组项可以通过合法下标绑定。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"][
        "result_schema"
    ] = {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                },
            }
        },
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = selector

    canonicalize_generation_output(
        wire,
        catalog,
        _allocate_two_step_ids(),
        hypothesis=_hypothesis(),
    )


def test_action_contract_accepts_array_items_declared_by_output_schema_alias() -> None:
    """output_schema 别名声明的数组项同样可以作为公开 selector。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"][
        "output_schema"
    ] = {
        "type": "object",
        "properties": {
            "ensemble": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "structure_artifact": {"type": "object"},
                        "candidate_id": {"type": "string"},
                    },
                },
            }
        },
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = (
        "/result/ensemble/3/structure_artifact"
    )

    canonicalize_generation_output(
        wire,
        catalog,
        _allocate_two_step_ids(),
        hypothesis=_hypothesis(),
    )


@pytest.mark.parametrize(
    "selector",
    [
        "/result/items/0/name",
        "/result/items/12/metadata/value",
        "/result/items/0/children/3/label",
        "/result/items/0/a~1b",
        "/result/items/1/tilde~0name",
    ],
)
def test_action_contract_accepts_array_object_properties_and_escaped_pointers(
    selector: str,
) -> None:
    """数组对象的公开字段支持递归路径和 JSON Pointer 转义。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"][
        "result_schema"
    ] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "metadata": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                        },
                        "children": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"label": {"type": "string"}},
                            },
                        },
                        "a/b": {"type": "string"},
                        "tilde~name": {"type": "string"},
                    },
                },
            }
        },
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = selector

    canonicalize_generation_output(
        wire,
        catalog,
        _allocate_two_step_ids(),
        hypothesis=_hypothesis(),
    )


@pytest.mark.parametrize(
    "selector",
    [
        "/result/items/name",
        "/result/items/0/unknown",
        "/result/items/0/name/0",
        "/result/items/-1/name",
        "/result/items/00/name",
        "/result/items/0/a~2b",
    ],
)
def test_action_contract_rejects_invalid_array_object_pointers(
    selector: str,
) -> None:
    """数组对象 selector 必须有合法下标、已声明字段和正确结构。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"][
        "result_schema"
    ] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "metadata": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                        },
                        "a/b": {"type": "string"},
                    },
                },
            }
        },
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = selector

    with pytest.raises(WorkflowContractError, match="items|selector|Pointer"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


@pytest.mark.parametrize(
    ("selector", "accepted"),
    [
        ("/result/matches/0", True),
        ("/result/0", False),
    ],
)
def test_action_contract_legacy_fields_array_selector_uses_field_name(
    selector: str,
    accepted: bool,
) -> None:
    """legacy fields 数组 Schema 只允许字段名后的规范数组下标。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"].update(
        {
            "result_fields": ["matches"],
            "result_schema": {
                "fields": [
                    {
                        "name": "matches",
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    }
                ]
            },
        }
    )
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = selector

    if accepted:
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )
    else:
        with pytest.raises(WorkflowContractError, match="result/0"):
            canonicalize_generation_output(
                wire,
                catalog,
                _allocate_two_step_ids(),
                hypothesis=_hypothesis(),
            )


@pytest.mark.parametrize(
    "selector",
    [
        "/result/matches/first",
        "/result/matches/-1",
        "/result/matches/00",
    ],
)
def test_action_contract_rejects_invalid_array_item_selector(
    selector: str,
) -> None:
    """结果 Schema 的数组项占位不能放行非法数组下标。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"][
        "result_schema"
    ] = {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            }
        },
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = selector

    with pytest.raises(WorkflowContractError, match="matches"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


def test_action_contract_does_not_treat_literal_star_property_as_array_item() -> None:
    """名为星号的真实字段不能把任意结果字段变成合法 selector。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"][
        "result_schema"
    ] = {
        "type": "object",
        "properties": {"*": {"type": "string"}},
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = (
        "/result/0"
    )

    with pytest.raises(WorkflowContractError, match="result/0"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


def test_action_contract_rejects_fabricated_result_pointer_before_execution() -> None:
    """只有 typed artifact 合同的上游步骤不得接受虚构的 result 指针。"""

    catalog = _catalog_with_artifact_only_output_contract()
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = (
        "/result/atom_mapped_reaction"
    )

    with pytest.raises(WorkflowContractError, match="atom_mapped_reaction.*AtomicStructure") as error:
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )
    assert "step-formal-1" not in str(error.value)


def test_action_contract_accepts_matching_semantic_artifact_selector() -> None:
    """上游合同声明的 primary_output 可以通过 typed artifact selector 传递。"""

    catalog = _catalog_with_artifact_only_output_contract()
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = (
        "artifact.semantic_type=AtomicStructure"
    )

    workflow = canonicalize_generation_output(
        wire,
        catalog,
        _allocate_two_step_ids(),
        hypothesis=_hypothesis(),
    )
    assert (
        workflow["steps"][1]["request"]["bindings"][0]["source_field"]
        == "artifact.semantic_type=AtomicStructure"
    )


def test_action_contract_accepts_explicit_result_field_declaration() -> None:
    """合同明确列出的 result 字段仍可作为稳定的公开绑定来源。"""

    catalog = _catalog_with_artifact_only_output_contract()
    catalog["action_contracts"]["contracts"][0]["output_contract"]["result_fields"] = [
        "structure_summary"
    ]
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = (
        "/result/structure_summary"
    )

    canonicalize_generation_output(
        wire,
        catalog,
        _allocate_two_step_ids(),
        hypothesis=_hypothesis(),
    )


def test_binding_rejects_result_selector_without_frozen_output_facts() -> None:
    """没有冻结输出事实时，/result 指针不能凭字段名通过校验。"""

    catalog = build_catalog_snapshot()
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = (
        "/result/guessed_value"
    )

    with pytest.raises(WorkflowContractError, match="guessed_value.*输出合同"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


@pytest.mark.parametrize(
    "selector",
    [
        "artifact.semantic_type=AtomicStructure",
        "artifact.media_type=chemical/x-xyz",
        "declared_artifact.name=structure",
    ],
)
def test_binding_rejects_typed_selector_when_corresponding_output_facts_are_empty(
    selector: str,
) -> None:
    """pair 合同存在但对应语义/媒体/名称集合为空时，typed selector 必须拒绝。"""

    catalog = build_catalog_snapshot()
    catalog["actions"][0]["primary_output"] = None
    catalog["action_contracts"] = {
        "schema_version": 1,
        "catalog_hash": catalog["catalog_hash"],
        "contracts": [
            {
                "action_id": "convert_structure",
                "backend_id": "openbabel",
                "request_contract": {
                    "required_contract": {
                        "inputs": [],
                        "method_spec": [],
                        "action_settings": [],
                        "resource_limits": [],
                    },
                    "optional_field_names": [],
                    "execute_action_request_template": {},
                    "enumerated_options": {},
                    "conditional_requirements": [],
                },
                "output_contract": {},
            }
        ],
    }
    wire = _generation_wire(with_dependency=True)
    wire["steps"][1]["request"]["bindings"][0]["source_field"] = selector

    with pytest.raises(WorkflowContractError, match="输出合同"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


def test_foreach_rejects_result_selector_not_declared_by_source_output_contract() -> None:
    """foreach 的来源 selector 与普通 binding 使用同一份输出合同校验。"""

    catalog = build_catalog_snapshot()
    wire = _generation_wire(with_dependency=True)
    second_request = wire["steps"][1]["request"]
    second_request["bindings"] = []
    second_request["foreach"] = {
        "source_step_ref": "s1",
        "source_field": "/result/guessed_items",
        "target_field": "/inputs/structure",
        "max_items": 2,
    }

    with pytest.raises(WorkflowContractError, match="guessed_items.*输出合同"):
        canonicalize_generation_output(
            wire,
            catalog,
            _allocate_two_step_ids(),
            hypothesis=_hypothesis(),
        )


@pytest.mark.parametrize("execution_layer", ["native_software", "programmable_analysis"])
def test_revision_insert_uses_public_open_step_shell_and_applies_copy(
    execution_layer: str,
) -> None:
    """Revision 插入应接受公开 open step 外壳，并复制为可执行正式步骤。"""

    catalog = _catalog_with_open_execution()
    before = _open_execution_workflow(execution_layer)
    projected = project_workflow_for_revision(before, catalog_snapshot=catalog)
    inserted_step = deepcopy(projected["steps"][0])
    inserted_step.pop("step_ref")
    patch = {
        "operations": [
            {
                "op": "insert_step_after",
                "new_step_ref": "n1",
                "after_step_ref": "s1",
                "step": inserted_step,
            }
        ]
    }
    result = apply_workflow_revision(
        before,
        patch,
        catalog,
        lambda kind: f"job-step-inserted-{kind}",
    )

    assert len(result["after_workflow"]["steps"]) == 2
    inserted = result["after_workflow"]["steps"][1]
    assert inserted["execution_layer"] == execution_layer
    assert inserted.get("action_id") is None
    assert inserted.get("backend_id") is None
    assert inserted["request"] == before["steps"][0]["request"]
