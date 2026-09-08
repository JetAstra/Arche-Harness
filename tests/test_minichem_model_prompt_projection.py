"""验证模型提示只暴露 MiniChem Action/Backend 和 request 合同。"""

import copy
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from chemistry_multiagent.utils.model_prompts import (
    _MINICHEM_HYPOTHESIS_CAPABILITY_ACTION_FIELDS,
    _MINICHEM_STAGE_ACTION_FIELDS,
    _project_minichem_actions,
    build_model_messages,
)
from chemistry_multiagent.utils.workflow_contract import project_minichem_stage_tool_projection
from tests.fakes.minichem_mcp import build_catalog_snapshot


_STAGE_ACTION_FIELDS = {
    "action_id",
    "action_version",
    "description",
    "backend_ids",
    "backends",
    "required_inputs",
    "optional_inputs",
    "method_spec",
    "primary_output",
    "output_semantic_types",
    "selection_policy",
    "data_action",
}

_ABSOLUTE_TMP = Path(Path.cwd().anchor) / "tmp"
_QUERY_ACTION_FIELDS = {
    "action_id",
    "description",
    "backend_ids",
    "primary_output",
    "output_semantic_types",
    "selection_policy",
    "data_action",
}
_ROUTER_ACTION_FIELDS = {
    "action_id",
    "action_version",
    "category",
    "description",
    "backend_ids",
    "required_inputs",
    "optional_inputs",
    "selection_policy",
    "primary_output",
    "data_action",
}
_BACKEND_FIELDS = {
    "backend_id",
    "display_name",
    "runtime",
    "availability",
    "capabilities",
}


def _user_envelope(messages):
    """解析唯一 user JSON，便于检查投影后的 base_input。"""

    user_message = next(item for item in messages if item["role"] == "user")
    return json.loads(user_message["content"])


def _catalog_fixtures():
    """返回同一份冻结目录及其两种模型可见投影。"""

    catalog = build_catalog_snapshot()
    stage_actions = project_minichem_stage_tool_projection(catalog)
    task_subgraph = {
        "catalog_hash": catalog["catalog_hash"],
        "catalog_version": str(catalog["schema_version"]),
        "actions": stage_actions,
        # 故意传入 canonical id/health，验证提示层会改成公开 backend 字段。
        "backends": copy.deepcopy(catalog["backends"]),
        "native_software": [
            {
                "software_id": "gaussian",
                "display_name": "Gaussian 16",
                "available": True,
                "purpose": "Execute an Agent-authored Gaussian input deck.",
                "native_invocation_guides": [
                    {
                        "executable": "g16",
                        "input_mode": "stdin_file",
                        "native_job_request_template": {
                            "software_id": "gaussian",
                            "executable": "g16",
                            "arguments": [],
                            "stdin_target": "input.com",
                        },
                    }
                ],
            }
        ],
        "analysis_runtimes": [
            {"runtime": "core", "available": True, "backend_ids": ["openbabel"]}
        ],
        "managed_program_contract": {
            "job_context_required_for_compliance": True
        },
    }
    return catalog, stage_actions, task_subgraph


def _planner_13_key_task_subgraph():
    """构造 Planner 生产方可能发送的基础七键加六项可选合同视图。"""

    task_subgraph = _open_execution_task_subgraph()
    actions = task_subgraph["actions"]
    action = actions[0]
    task_subgraph = deepcopy(task_subgraph)
    task_subgraph.update(
        {
            "action_backend_contracts": [
                {
                    "action_id": action["action_id"],
                    "backend_id": action["backend_ids"][0],
                    "request_contract": {"required_contract": {}},
                    "output_contract": {"result_envelope": "ActionResult"},
                }
            ],
            "managed_program_contract": {
                "job_context_required_for_compliance": True,
                "execution_note": "Use JobContext for declared inputs and outputs.",
                "minimal_python_template": (
                    "from researchchem_job import JobContext\n"
                    "ctx = JobContext.load()\n"
                ),
                "helper_name_rule": "Helper names must match declaration names.",
            },
            "submit_analysis_program_request_template": {
                "runtime": "<exact runtime id>",
                "script_path": "code/<agent-authored-program>.py",
                "script_target": "code/agent_program.py",
                "arguments": [],
                "staged_inputs": [],
                "inputs": [
                    {
                        "name": "<logical_input_name>",
                        "source_path": "data/<required-input>",
                        "target_path": "inputs/<required-input>",
                        "semantic_type": "<input semantic type>",
                    }
                ],
                "outputs": [
                    {
                        "name": "<logical_output_name>",
                        "path": "outputs/<result-file>",
                        "semantic_type": "<output semantic type>",
                        "media_type": "application/json",
                        "required": True,
                    }
                ],
                "label": "<descriptive scientific operation>",
            },
            "selection_note": "Select one exact runtime; no runtime is chosen automatically.",
            "execution_note": "Submit authored programs through the managed job runner.",
            "evaluation_resource_budget": {
                "source": "service_policy",
                "agent_controllable": False,
                "scope": "per_task",
            },
            "execution_timeout_policy": {
                "execution_class": "compute",
                "source": "service_policy",
                "agent_controllable": False,
            },
        }
    )
    return task_subgraph


def _open_execution_task_subgraph():
    """构造包含公开 Native/Analysis 能力的 Planner 目录子图。"""

    _, actions, task_subgraph = _catalog_fixtures()
    task_subgraph["native_software"] = [
        {
            "software_id": "gaussian16",
            "display_name": "Gaussian 16",
            "commands": [
                {
                    "executable": "g16",
                    "synopsis": "运行 Gaussian 输入文件",
                    "input_mode": "file",
                    "required_files": ["input.gjf"],
                    "example_request": {"calculation_intent": "frequency"},
                },
                {
                    "executable": "/private/gaussian/g16",
                    "synopsis": "不应暴露的本机路径",
                    "input_mode": "file",
                    "required_files": ["input.gjf"],
                    "example_request": {"input_path": "/private/gaussian/g16"},
                },
            ],
        }
    ]
    task_subgraph["analysis_runtimes"] = [
        {
            "runtime": "python",
            "description": "运行受限的 Python 分析脚本",
            "modules": ["numpy", "matplotlib"],
        }
    ]
    task_subgraph["authoring_contract"] = {
        "path_prefix": "code/",
        "max_files": 4,
        "max_chars_per_file": 100000,
    }
    return task_subgraph


def _step_with_request(request, *, layer="native_software"):
    """构造用于 JSON Schema 校验的最小 open execution 步骤。"""

    return {
        "step_ref": "s1",
        "step_name": "运行 Gaussian",
        "description": "生成并运行计算输入",
        "execution_layer": layer,
        "request": request,
        "depends_on": [],
    }


def _minimal_router_input(tool_catalog_projection):
    """构造 Router 的最小事实包；具体科学校验由 Router Agent 负责。"""

    return {
        "scientific_context": {
            "scientific_question": "转换一个明确给出的分子结构。",
            "chemistry_context": {},
        },
        "combination": {"hypothesis": {"strategy_name": "转换", "reasoning": "保留结构。"}},
        "workflow": {"steps": []},
        "direct_revision": None,
        "execution_result": {},
        "program_validation": {},
        "evidence_index": [],
        "tool_catalog_projection": tool_catalog_projection,
        "allowed_decisions": ["revise_workflow"],
        "router_output_schema": {"type": "object"},
    }


def _regeneration_outcomes(catalog_hash):
    """构造固定假设终局，并附加不应进入最终消息的内部字段。"""

    return [
        {
            "source_query": "电子结构与可观测性质",
            "hypothesis": {
                "strategy_name": "结构表示",
                "reasoning": "结构可以被稳定转换；原题给出了可计算结构，转换结果可以直接核对。",
            },
            "terminal_classification": "reliably_refuted",
            "final_judge": {
                "reason": "结果支持重新审视假设。",
            },
            "cited_evidence": [
                {
                    "evidence_ref": "e1",
                    "step_ref": "s1",
                    "evidence_kind": "observation",
                    "name": "converted_structure",
                    "value": True,
                    "unit": None,
                    "conditions": {},
                }
            ],
            "attempted_observables": ["converted_structure"],
            "missing_evidence": [],
            "tool_capability_gap": None,
            "workflow": {"steps": [{"step_id": "formal-step-1"}]},
            "raw_output": {"catalog_hash": catalog_hash},
            "operation_id": "operation_private",
        }
    ]


def test_workflow_generation_projects_minichem_task_subgraph():
    """Planner 看到规范 Action/Backend 与冻结的开放执行调用合同。"""

    catalog, actions, task_subgraph = _catalog_fixtures()
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {
                "scientific_question": "转换苯甲醛结构。",
                "chemistry_context": {},
            },
            "hypothesis": {
                "strategy_name": "结构转换",
                "reasoning": "给定结构可以被转换成可复用表示；需要直接检查转换结果。",
            },
            "task_tool_subgraph": task_subgraph,
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    projected = _user_envelope(messages)["base_input"]["task_tool_subgraph"]

    assert set(projected) == {
        "catalog_hash",
        "catalog_version",
        "actions",
        "backends",
        "native_software",
        "analysis_runtimes",
        "managed_program_contract",
    }
    assert projected["catalog_hash"] == catalog["catalog_hash"]
    assert projected["actions"] == actions
    assert all(set(action) == _STAGE_ACTION_FIELDS for action in projected["actions"])
    assert all(set(backend) == _BACKEND_FIELDS for backend in projected["backends"])
    assert all("id" not in backend and "health" not in backend for backend in projected["backends"])
    assert projected["native_software"][0]["software_id"] == "gaussian"
    assert projected["native_software"][0]["native_invocation_guides"][0][
        "input_mode"
    ] == "stdin_file"
    system = next(item["content"] for item in messages if item["role"] == "system")
    assert "MiniChem" in system and "Action" in system
    assert "ToolSpec" not in system
    assert "job_type" not in system


def test_workflow_generation_projects_planner_13_key_task_subgraph():
    """Planner 的七个基础字段与六项可选合同都应进入模型可见目录。"""

    task_subgraph = _planner_13_key_task_subgraph()
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {"scientific_question": "运行结构分析。"},
            "hypothesis": {
                "strategy_name": "结构分析",
                "reasoning": "直接观测可以回答问题；需要使用公开目录中的能力。",
            },
            "task_tool_subgraph": task_subgraph,
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    projected = _user_envelope(messages)["base_input"]["task_tool_subgraph"]
    assert set(projected) == set(task_subgraph)
    for field in (
        "action_backend_contracts",
        "managed_program_contract",
        "submit_analysis_program_request_template",
        "execution_timeout_policy",
        "selection_note",
        "execution_note",
        "evaluation_resource_budget",
    ):
        assert projected[field] == task_subgraph[field]


def test_workflow_generation_accepts_large_composite_action_contract():
    """复合 Action 的参数清单可以超过普通请求数组的 32 项上限。"""

    task_subgraph = _planner_13_key_task_subgraph()
    task_subgraph["action_backend_contracts"][0]["request_contract"] = {
        "execute_action_request_template": {
            "action_settings": {
                f"setting_{index}": f"<setting_{index}>" for index in range(35)
            }
        },
        "required_contract": {
            "action_settings": [
                {"name": f"setting_{index}", "required": True, "type": "string"}
                for index in range(35)
            ]
        }
    }
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {"scientific_question": "运行复合过渡态搜索。"},
            "hypothesis": {
                "strategy_name": "过渡态搜索",
                "reasoning": "候选路径可以通过计算验证；需要完整复合 Action 合同。",
            },
            "task_tool_subgraph": task_subgraph,
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    projected = _user_envelope(messages)["base_input"]["task_tool_subgraph"]
    settings = projected["action_backend_contracts"][0]["request_contract"][
        "required_contract"
    ]["action_settings"]
    assert len(settings) == 35
    template = projected["action_backend_contracts"][0]["request_contract"][
        "execute_action_request_template"
    ]["action_settings"]
    assert len(template) == 35


def test_workflow_generation_rejects_unknown_planner_catalog_field():
    """Planner 目录新增未登记顶层字段时必须拒绝，而不能静默删除。"""

    task_subgraph = _planner_13_key_task_subgraph()
    task_subgraph["unexpected_catalog_field"] = True
    with pytest.raises(ValueError, match="MiniChem Catalog"):
        build_model_messages(
            "workflow_generation",
            {
                "scientific_context": {"scientific_question": "Q"},
                "hypothesis": {
                    "strategy_name": "策略",
                    "reasoning": "直接观测可以回答问题；R",
                },
                "task_tool_subgraph": task_subgraph,
                "workflow_generation_output_schema": {"type": "object"},
            },
        )


@pytest.mark.parametrize(
    "field, mutate, error_text",
    [
        (
        "action_backend_contracts",
            lambda value: [],
            "action_backend_contracts 必须是非空数组",
        ),
        (
            "managed_program_contract",
            lambda value: {**value, "unexpected": True},
            "managed_program_contract 字段集合不正确",
        ),
        (
            "submit_analysis_program_request_template",
            lambda value: {**value, "script_path": str(_ABSOLUTE_TMP / "program.py")},
            "script_path 必须是 code/ 下的安全路径",
        ),
        (
            "selection_note",
            lambda value: [],
            "selection_note 必须是单行非空字符串",
        ),
        (
            "evaluation_resource_budget",
            lambda value: {**value, "source": "evaluation_policy"},
            "evaluation_resource_budget 值不符合任务资源政策",
        ),
    ],
)
def test_workflow_generation_rejects_invalid_planner_optional_contract(
    field, mutate, error_text
):
    """五项 Open optional 和 Action 合同损坏时必须在提示组装前拒绝。"""

    task_subgraph = _planner_13_key_task_subgraph()
    task_subgraph[field] = mutate(task_subgraph[field])
    with pytest.raises(ValueError, match=error_text):
        build_model_messages(
            "workflow_generation",
            {
                "scientific_context": {"scientific_question": "Q"},
                "hypothesis": {
                    "strategy_name": "策略",
                    "reasoning": "直接观测可以回答问题；R",
                },
                "task_tool_subgraph": task_subgraph,
                "workflow_generation_output_schema": {"type": "object"},
            },
        )


def test_workflow_generation_projects_open_execution_without_machine_paths():
    """Planner 可见清单包含 Gaussian/runtime，但不暴露本机绝对路径。"""

    task_subgraph = _open_execution_task_subgraph()
    messages = build_model_messages(
        "workflow_generation",
        {
            "scientific_context": {"scientific_question": "计算苯甲醛频率。"},
            "hypothesis": {
                "strategy_name": "Gaussian 频率",
                "reasoning": "频率计算可以提供直接光谱证据；需要观察频率结果。",
            },
            "task_tool_subgraph": task_subgraph,
            "workflow_generation_output_schema": {"type": "object"},
        },
    )
    projected = _user_envelope(messages)["base_input"]["task_tool_subgraph"]

    assert projected["native_software"] == [
        {
            "software_id": "gaussian16",
            "display_name": "Gaussian 16",
            "commands": [
                {
                    "executable": "g16",
                    "synopsis": "运行 Gaussian 输入文件",
                    "input_mode": "file",
                    "required_files": ["input.gjf"],
                    "example_request": {"calculation_intent": "frequency"},
                }
            ],
        }
    ]
    assert projected["analysis_runtimes"][0]["runtime"] == "python"
    assert "modules" not in projected["analysis_runtimes"][0]
    assert projected["authoring_contract"] == {
        "path_prefix": "code/",
        "max_files": 4,
        "max_chars_per_file": 100000,
    }
    assert "/private/gaussian/g16" not in json.dumps(projected, ensure_ascii=False)
    system = next(item["content"] for item in messages if item["role"] == "system")
    for phrase in (
        "优先 Action",
        "缺少决定性观测时才逐值选择 open_execution",
        "authored_files",
        "source_path",
    ):
        assert phrase in system


def test_open_execution_projection_rejects_unknown_catalog_fields():
    """公开清单出现未登记字段时不得静默传给模型。"""

    task_subgraph = _open_execution_task_subgraph()
    task_subgraph["native_software"][0]["commands"][0]["private_path"] = "/secret"
    with pytest.raises(ValueError, match="字段集合"):
        build_model_messages(
            "workflow_generation",
            {
                "scientific_context": {"scientific_question": "Q"},
                "hypothesis": {
                    "strategy_name": "H",
                    "reasoning": "直接观测可以回答问题；R",
                },
                "task_tool_subgraph": task_subgraph,
                "workflow_generation_output_schema": {"type": "object"},
            },
        )


def test_query_stays_isolated_while_hypothesis_receives_compact_capabilities():
    """Query 不看工具；Hypothesis 只接收紧凑、权威的 MiniChem 能力摘要。"""

    _, actions, _ = _catalog_fixtures()
    capabilities = [
        {
            "action_id": "quantum_calculation",
            "action_version": "1.0.0",
            "category": "quantum_chemistry",
            "description": "Return a direct quantum-chemical measurement.",
            "input_description": "Accept an explicit molecular structure.",
            "input_semantic_types": ["chemical_structure"],
            "required_inputs": ["structure"],
            "optional_inputs": [],
            "primary_output": "Measurement",
            "output_semantic_types": ["measurement"],
            "capability_tags": ["direct_observable"],
            "availability": {"status": "available"},
            "data_action": False,
        }
    ]
    query_messages = build_model_messages(
        "hypothesis_generation",
        {
            "scientific_question": "解释苯甲醛结构的计算性质。",
            "effective_chemistry_context": {},
            "num_queries": 1,
            "retrieval_projection": {},
            "stage_tool_projection": actions,
        },
        stage="query",
    )
    query_base = _user_envelope(query_messages)["base_input"]
    assert query_base == {
        "scientific_question": "解释苯甲醛结构的计算性质。",
        "num_queries": 1,
    }
    query_system = next(item["content"] for item in query_messages if item["role"] == "system")
    for phrase in (
        "You are a highly experienced computational chemistry research strategist.",
        "mechanistic hypotheses and computational strategies",
        "comprehensive (30+ words)",
        "原题明确的研究对象、条件和比较范围",
        "一个可判别的问题",
        "literature_review、mechanistic_clues、limitations 和 chemistry_context",
        "theoretical background and computational methods",
    ):
        assert phrase in query_system
    assert "不得依赖文献、工具目录、运行结果、Memory" not in query_system
    assert "模型化问题" in query_system
    assert "standard internal-coordinate" in query_system

    hypothesis_messages = build_model_messages(
        "hypothesis_generation",
        {
            "scientific_question": "解释苯甲醛结构的计算性质。",
            "effective_chemistry_context": {},
            "num_hypotheses": 1,
            "query_angle": "电子结构与可观测性质",
            "chemical_capability_projection": capabilities,
        },
        stage="hypothesis",
    )
    hypothesis_base = _user_envelope(hypothesis_messages)["base_input"]
    assert hypothesis_base == {
        "scientific_question": "解释苯甲醛结构的计算性质。",
        "query_angle": "电子结构与可观测性质",
        "num_hypotheses": 1,
        "chemical_capability_projection": capabilities,
    }
    assert all("backend_ids" not in item for item in capabilities)
    assert all("method_spec" not in item for item in capabilities)
    hypothesis_system = next(item["content"] for item in hypothesis_messages if item["role"] == "system")
    for phrase in (
        "professional computational chemist",
        "computational mechanistic hypotheses",
        "原子化学能力",
        "chemical_capability_projection",
        "atomic capability id",
        "ONE specific query angle",
        "directly address that query angle",
        "批内候选须在核心因果物种/状态、可证伪主张或区分竞争机制的决定性对照上实质不同",
    ):
        assert phrase in hypothesis_system
    assert "MiniChem Action/Backend Catalog" not in hypothesis_system
    assert "后端、数据连接或阶段顺序" in hypothesis_system
    assert "ToolSpec" not in hypothesis_system


def test_router_and_regeneration_use_fixed_model_visible_shapes():
    """Router 使用十字段 Action 摘要，再生成只读取固定的科学终局。"""

    catalog, actions, _ = _catalog_fixtures()
    action = actions[0]
    router_action = {
        "action_id": action["action_id"],
        "action_version": action["action_version"],
        "category": "structure_and_system",
        "description": action["description"],
        "backend_ids": action["backend_ids"],
        "required_inputs": action["required_inputs"],
        "optional_inputs": action["optional_inputs"],
        "selection_policy": action["selection_policy"],
        "primary_output": action["primary_output"],
        "data_action": action["data_action"],
    }
    router_messages = build_model_messages(
        "router", _minimal_router_input([router_action])
    )
    router_projection = _user_envelope(router_messages)["base_input"]["tool_catalog_projection"]
    assert set(router_projection[0]) == _ROUTER_ACTION_FIELDS
    assert "backends" not in router_projection[0]

    regeneration_messages = build_model_messages(
        "hypothesis_generation",
        {
            "scientific_context": {
                "scientific_question": "解释苯甲醛结构的计算性质。",
                "initial_query_angles": ["电子结构与可观测性质"],
                "initial_retrieval_phase": {},
                "chemistry_context": {},
            },
            "current_query": "电子结构与可观测性质",
            "prior_hypothesis_outcomes": _regeneration_outcomes(
                catalog["catalog_hash"]
            ),
            "next_batch_index": 2,
            "generation_config": {
                "num_queries": 1,
                "num_hypotheses_per_query": 1,
                "top_n_workflows": 1,
            },
            "hypothesis_output_schema": {"type": "array"},
        },
        stage="regeneration",
        memory_enabled=True,
    )
    regeneration_base = _user_envelope(regeneration_messages)["base_input"]
    history = regeneration_base["prior_hypothesis_outcomes"]
    assert set(regeneration_base) == {
        "scientific_context",
        "current_query",
        "prior_hypothesis_outcomes",
        "next_batch_index",
        "generation_config",
        "hypothesis_output_schema",
    }
    assert set(history[0]) == {
        "source_query",
        "hypothesis",
        "terminal_classification",
        "final_judge",
        "cited_evidence",
        "attempted_observables",
        "missing_evidence",
        "tool_capability_gap",
    }
    serialized = json.dumps(regeneration_base, ensure_ascii=False)
    for forbidden in (
        '"workflow":',
        '"step_id":',
        '"raw_output":',
        "operation_private",
        catalog["catalog_hash"],
    ):
        assert forbidden not in serialized


def test_legacy_toolspec_projections_are_rejected():
    """旧 ToolSpec 的名称、端口和连接字段不得再进入 MiniChem 主线。"""

    legacy_tool = [{"name": "legacy", "purpose": "旧工具", "inputs": [], "outputs": []}]
    with pytest.raises(ValueError, match="MiniChem Action"):
        build_model_messages("router", _minimal_router_input(legacy_tool))

    _, actions, _ = _catalog_fixtures()
    with pytest.raises(ValueError, match="MiniChem Catalog"):
        build_model_messages(
            "workflow_generation",
            {
                "scientific_context": {"scientific_question": "Q", "chemistry_context": {}},
                "hypothesis": {
                    "strategy_name": "策略",
                    "reasoning": "直接观测可以回答问题；需要一次计算。",
                },
                "task_tool_subgraph": {
                    "tools": actions,
                    "ports": [],
                    "connections": [],
                },
            },
        )


def _stage_actions_with_backend_request_contracts():
    """构造带后端专属请求和输出合同的完整 Action 投影。"""

    _, actions, _ = _catalog_fixtures()
    contracts = [
        {
            "backend_id": actions[0]["backend_ids"][0],
            "request_contract": {"required_contract": {}},
            "output_contract": {"required_outputs": []},
        }
    ]
    actions = deepcopy(actions)
    actions[0]["backend_request_contracts"] = contracts
    return actions, contracts


def test_stage_projection_accepts_and_preserves_backend_request_contracts():
    """完整 Action 接受 companion 合同，并按原顺序深拷贝其内容。"""

    actions, contracts = _stage_actions_with_backend_request_contracts()
    projected = _project_minichem_actions(
        actions,
        "stage_tool_projection",
        expected_fields=_MINICHEM_STAGE_ACTION_FIELDS,
    )
    assert projected[0]["backend_request_contracts"] == contracts
    assert projected[0]["backend_request_contracts"] is not contracts
    assert projected[0]["backend_request_contracts"][0] is not contracts[0]
    assert list(projected[0]["backend_request_contracts"]) == contracts


def test_workflow_revision_accepts_stage_projection_with_backend_contracts():
    """修订 Prompt 应保留 Planner 提供的 Action/Backend 详细合同。"""

    actions, contracts = _stage_actions_with_backend_request_contracts()
    messages = build_model_messages(
        "workflow_revision",
        {
            "target": {},
            "workflow_failure_trajectories": [],
            "human_correction_chains": [],
            "tool_catalog_projection": actions,
            "workflow_revision_output_schema": {"type": "object"},
        },
        memory_enabled=True,
    )
    projected = _user_envelope(messages)["base_input"]["tool_catalog_projection"]

    assert projected[0]["backend_request_contracts"] == contracts


def test_stage_projection_without_backend_request_contracts_keeps_legacy_shape():
    """没有 companion 合同时，完整 Action 仍按旧字段集合投影。"""

    _, actions, _ = _catalog_fixtures()
    projected = _project_minichem_actions(
        actions,
        "stage_tool_projection",
        expected_fields=_MINICHEM_STAGE_ACTION_FIELDS,
    )
    assert "backend_request_contracts" not in projected[0]


def test_stage_projection_rejects_unknown_action_fields():
    """完整 Action 的未知字段仍必须被拒绝。"""

    _, actions, _ = _catalog_fixtures()
    actions[0]["unexpected_action_field"] = True
    with pytest.raises(ValueError, match="字段集合不正确"):
        _project_minichem_actions(
            actions,
            "stage_tool_projection",
            expected_fields=_MINICHEM_STAGE_ACTION_FIELDS,
        )


@pytest.mark.parametrize(
    "backend_id, error_text",
    [("missing_backend", "未由 Action 声明"), ("", "backend_id 非法")],
)
def test_stage_projection_rejects_illegal_backend_request_contracts(
    backend_id, error_text
):
    """后端合同引用未声明或空后端时必须拒绝。"""

    actions, _ = _stage_actions_with_backend_request_contracts()
    actions[0]["backend_request_contracts"][0]["backend_id"] = backend_id
    with pytest.raises(ValueError, match=error_text):
        _project_minichem_actions(
            actions,
            "stage_tool_projection",
            expected_fields=_MINICHEM_STAGE_ACTION_FIELDS,
        )


def test_stage_projection_rejects_duplicate_backend_request_contracts():
    """同一 Action 不得为同一后端重复声明 companion 合同。"""

    actions, contracts = _stage_actions_with_backend_request_contracts()
    actions[0]["backend_request_contracts"].append(deepcopy(contracts[0]))
    with pytest.raises(ValueError, match="backend_id 重复"):
        _project_minichem_actions(
            actions,
            "stage_tool_projection",
            expected_fields=_MINICHEM_STAGE_ACTION_FIELDS,
        )


@pytest.mark.parametrize("field", ["request_contract", "output_contract"])
def test_stage_projection_rejects_non_mapping_backend_contracts(field):
    """后端请求合同和输出合同都必须是 object。"""

    actions, _ = _stage_actions_with_backend_request_contracts()
    actions[0]["backend_request_contracts"][0][field] = []
    with pytest.raises(ValueError, match=f"{field} 必须是 object"):
        _project_minichem_actions(
            actions,
            "stage_tool_projection",
            expected_fields=_MINICHEM_STAGE_ACTION_FIELDS,
        )


def test_router_projection_rejects_optional_backend_request_contracts():
    """Router 仍只接受精确的十字段 Action 摘要。"""

    _, actions, _ = _catalog_fixtures()
    action = actions[0]
    router_action = {
        "action_id": action["action_id"],
        "action_version": action["action_version"],
        "category": "structure_and_system",
        "description": action["description"],
        "backend_ids": action["backend_ids"],
        "required_inputs": action["required_inputs"],
        "optional_inputs": action["optional_inputs"],
        "selection_policy": action["selection_policy"],
        "primary_output": action["primary_output"],
        "data_action": action["data_action"],
        "backend_request_contracts": [],
    }
    with pytest.raises(ValueError, match="MiniChem Action 投影"):
        build_model_messages("router", _minimal_router_input([router_action]))
