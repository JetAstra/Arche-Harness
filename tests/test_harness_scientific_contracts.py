"""覆盖 Harness 假设阶段的 Retrieval、原子能力和来源 Query 合同。"""

from __future__ import annotations

import json
from copy import deepcopy

from chemistry_multiagent.agents.hypothesis_agent import HypothesisAgent
from chemistry_multiagent.utils.model_prompts import build_model_messages
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_chemical_capability_projection,
)


def _mixed_catalog() -> dict:
    """构造包含化学计算与非化学 Action 的最小冻结目录。"""

    records = (
        ("calculate_energy", "molecular_electronic", False, "xtb", "compute"),
        ("search_path", "reaction_and_kinetics", False, "path_backend", "fast"),
        ("data_parser", "scientific_data_interchange", False, "parser", "fast"),
        ("python_analysis", "molecular_electronic", False, "python_runtime", "compute"),
        ("data_only", "molecular_electronic", True, "fixed_source", "fast"),
    )
    actions = []
    backends = []
    for action_id, category, data_action, backend_id, execution_class in records:
        actions.append(
            {
                "id": action_id,
                "version": "1.0.0",
                "category": category,
                "description": f"公开化学能力 {action_id}。",
                "input_description": "AtomicStructure 或已声明的科学输入。",
                "input_semantic_types": ["AtomicStructure"],
                "backend_ids": [backend_id],
                "required_inputs": ["structure"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "ScientificResult",
                "output_semantic_types": ["ScientificResult"],
                "execution_class": execution_class,
                "capability_tags": [category],
                "data_action": data_action,
            }
        )
        backends.append(
            {
                "id": backend_id,
                "display_name": backend_id,
                "runtime": "native",
                "capabilities": [action_id],
                "health": {"status": "healthy"},
            }
        )
    return {
        "schema_version": 7,
        "actions": actions,
        "backends": backends,
    }


def _base_input(messages: list[dict[str, str]]) -> dict:
    """读取统一 user envelope 中的模型可见 base_input。"""

    return json.loads(messages[1]["content"])["base_input"]


def _capability_projection() -> list[dict]:
    """返回测试使用的冻结原子化学能力。"""

    return project_minichem_chemical_capability_projection(_mixed_catalog())


def test_atomic_capability_projection_excludes_data_and_programming_layers() -> None:
    """能力投影只留下真实化学计算事实，不泄漏后端或数据处理层。"""

    projection = _capability_projection()
    assert [item["action_id"] for item in projection] == [
        "calculate_energy",
        "search_path",
    ]
    serialized = json.dumps(projection, ensure_ascii=False)
    assert "data_parser" not in serialized
    assert "python_analysis" not in serialized
    assert "data_only" not in serialized
    assert "backend_ids" not in serialized
    assert "selection_policy" not in serialized
    assert '"runtime"' not in serialized


def test_prompt_projection_routes_retrieval_source_and_capabilities() -> None:
    """四项 Retrieval 摘要、来源 Query 和能力事实按阶段进入白名单。"""

    projection = _capability_projection()
    query = _base_input(
        build_model_messages(
            "hypothesis_generation",
            {
                "scientific_question": "比较原题指定体系的直接结构 profile。",
                "num_queries": 1,
                "literature_review": "文献事实：已有构象线索。",
                "mechanistic_clues": ["构象预组织"],
                "limitations": ["缺少直接 profile"],
                "chemistry_context": {
                    "charge": 0,
                    "multiplicity": 1,
                    "backend_id": "hidden",
                    "path": "/private/context",
                },
                "chemical_capability_projection": projection,
            },
            stage="query",
        )
    )
    assert set(query) == {
        "scientific_question",
        "num_queries",
        "literature_review",
        "mechanistic_clues",
        "limitations",
        "chemistry_context",
    }
    assert "chemical_capability_projection" not in query
    assert "backend_id" not in json.dumps(query, ensure_ascii=False)

    candidate_a = {
        "hypothesis_id": "h-a",
        "strategy_name": "候选 A",
        "reasoning": "直接 profile 与原始问题一致。",
        "source_query_angles": ["query-A", "query-A"],
    }
    candidate_b = {
        "hypothesis_id": "h-b",
        "strategy_name": "候选 B",
        "reasoning": "另一项直接 profile。",
        "source_query_angles": ["query-B"],
    }
    relationship = _base_input(
        build_model_messages(
            "hypothesis_generation",
            {
                "scientific_question": "Q",
                "hypothesis_a": candidate_a,
                "hypothesis_b": candidate_b,
                "chemical_capability_projection": projection,
                "operation_id": "hidden",
            },
            stage="relationship",
        )
    )
    assert relationship["hypothesis_a"]["source_query_angles"] == ["query-A"]
    assert relationship["hypothesis_b"]["source_query_angles"] == ["query-B"]
    assert relationship["chemical_capability_projection"] == projection
    assert "operation_id" not in json.dumps(relationship, ensure_ascii=False)

    synthesis = _base_input(
        build_model_messages(
            "hypothesis_generation",
            {
                "scientific_question": "Q",
                "hypothesis_a": candidate_a,
                "hypothesis_b": candidate_b,
                "relationship_decision": {
                    "relationship": "merge",
                    "reason": "坐标签名逐项一致。",
                    "operation_id": "hidden",
                },
                "chemical_capability_projection": deepcopy(projection),
                "synthesis_output_schema": {"type": "object"},
            },
            stage="merge",
        )
    )
    assert synthesis["relationship_decision"] == {
        "relationship": "merge",
        "reason": "坐标签名逐项一致。",
    }
    assert synthesis["chemical_capability_projection"] == projection


def test_formal_operation_preserves_internal_sources_and_public_schema(tmp_path) -> None:
    """正式链路把来源和能力传给比较阶段，但对外仍只发布两字段正文。"""

    capability_projection = _capability_projection()
    calls: list[tuple[str, dict]] = []
    query = (
        "Compare the stated chemical structures under identical charge multiplicity and solvent "
        "conditions using a common atom-defined dihedral profile, report direct observables, "
        "supporting and refuting outcomes, model assumptions, controls, uncertainty, and "
        "mapping to the original mechanistic question for reproducible computation."
    )

    def provider(messages):
        dynamic = _base_input(messages)
        system = messages[0]["content"]
        if "num_queries" in dynamic:
            stage = "query"
            content = query
        elif "num_hypotheses" in dynamic:
            stage = "hypothesis"
            content = json.dumps(
                [
                    {"strategy_name": "候选 A", "reasoning": "直接观测 A。"},
                    {"strategy_name": "候选 B", "reasoning": "直接观测 B。"},
                ],
                ensure_ascii=False,
            )
        elif "比较两项候选假设并判断" in system:
            stage = "relationship"
            content = json.dumps(
                {"relationship": "independent", "reason": "主张不同。"},
                ensure_ascii=False,
            )
        else:
            stage = "rank"
            content = json.dumps(
                {
                    "analysis": "A 通过硬门槛。",
                    "reasoning": "A 的直接证据更完整。",
                    "winner": "A",
                    "loser": "B",
                },
                ensure_ascii=False,
            )
        calls.append((stage, deepcopy(dynamic)))
        return {
            "id": "scientific-contract-test",
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        }

    agent = HypothesisAgent(comparison_workers=1, model_provider=provider)
    result = agent.run_hypothesis_operation(
        task_root=str(tmp_path),
        operation_instance_id="scientific-contract-operation",
        scientific_question="比较原题指定体系的直接结构 profile。",
        stage_tool_projection=[],
        chemical_capability_projection=capability_projection,
        initial_query_angles=None,
        initial_retrieval_phase={
            "keywords": [],
            "literature_review": "文献事实：已有构象线索。",
            "mechanistic_clues": ["构象预组织"],
            "limitations": ["缺少直接 profile"],
            "chemistry_context": {"charge": 0, "multiplicity": 1},
        },
        chemistry_context={"charge": 0, "multiplicity": 1},
        num_queries=1,
        num_hypotheses_per_query=2,
        top_n=2,
        hypothesis_ids_by_query=[["h-a", "h-b"]],
        merge_hypothesis_ids=[],
    )

    by_stage = {stage: dynamic for stage, dynamic in calls}
    assert set(by_stage["query"]) == {
        "scientific_question",
        "num_queries",
        "literature_review",
        "mechanistic_clues",
        "limitations",
        "chemistry_context",
    }
    assert by_stage["hypothesis"]["chemical_capability_projection"] == (
        capability_projection
    )
    assert by_stage["relationship"]["chemical_capability_projection"] == (
        capability_projection
    )
    assert by_stage["relationship"]["hypothesis_a"]["source_query_angles"]
    assert by_stage["rank"]["chemical_capability_projection"] == capability_projection
    assert by_stage["rank"]["hypothesis_a"]["source_query_angles"]
    public_result = result["hypothesis_result"]
    for group in public_result["hypotheses_by_query"]:
        for hypothesis in group["hypotheses"]:
            assert "source_query_angles" not in hypothesis
    assert all("source_query_angles" not in item for item in public_result["ranked_strategies"])
