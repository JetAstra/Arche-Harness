from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemistry_multiagent.agents.retrieval_agent import RetrievalAgent
from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    _load_strict_scientific_input_json,
)
from chemistry_multiagent.memory.projections import _project_chemistry_context
from chemistry_multiagent.utils.gaussian_routing import scientific_question_fingerprint
from chemistry_multiagent.utils.model_prompts import build_model_messages
from chemistry_multiagent.utils.scientific_input import (
    merge_supplementary_chemistry_context,
    normalize_supplementary_scientific_input,
)


_QUESTION = "原始问题 C.5 + 88% ee"
_UNIT = {
    "calculation_unit": {
        "scope": "只验证给定的关键亲核加成/C–C 成键计算单元",
        "reactants": [
            "C=C(C)N1CCCC1C[NH+]1CCCC1",
            "O=Cc1ccc([N+](=O)[O-])cc1",
        ],
        "products": [
            "C/C(C[C@@H](O)c1ccc([N+](=O)[O-])cc1)=[N+]1/CCCC1CN1CCCC1"
        ],
        "charge": 1,
        "multiplicity": 1,
        "solvent": "acetone",
        "temperature_k": 303.15,
        "needs_ts": True,
        "needs_irc": True,
        "transition_state_initial_guess_source": "search_transition_state_path",
        "method_requirements": {
            "endpoint": {
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "solvent": "acetone",
                "temperature_k": 303.15,
                "jobs": ["opt", "freq"],
            },
            "transition_state": {
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "solvent": "acetone",
                "temperature_k": 303.15,
                "jobs": ["opt", "freq"],
            },
            "irc": {
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "solvent": "acetone",
                "temperature_k": 303.15,
                "jobs": ["irc"],
                "calcfc": True,
                "maxpoints": 80,
                "recalc": 5,
            },
        },
    }
}


def _user_base(messages: list[dict[str, str]]) -> dict:
    """读取模型真正可见的 base_input。"""

    user = next(item["content"] for item in messages if item["role"] == "user")
    return json.loads(user)["base_input"]


def test_local_input_is_strict_and_normalizes_method_requirements() -> None:
    """本地文件只在控制层解析，并保留通用方法要求。"""

    path = Path("local_data/config/author_simplified_scientific_input.json")
    normalized = _load_strict_scientific_input_json(str(path))
    unit = normalized["calculation_unit"]
    assert unit["reactants"][0] == _UNIT["calculation_unit"]["reactants"][0]
    assert unit["products"] == _UNIT["calculation_unit"]["products"]
    assert unit["temperature"] == "303.15 K"
    assert unit["method_requirements"]["irc"]["maxpoints"] == 80
    assert unit["method_requirements"]["irc"]["recalc"] == 5
    assert "literature" not in unit


def test_unknown_input_fields_and_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    """严格输入不得夹带坐标、路径或其他未注册字段。"""

    assert normalize_supplementary_scientific_input(None) == {}
    assert normalize_supplementary_scientific_input({}) == {}
    with pytest.raises(ValueError):
        normalize_supplementary_scientific_input(
            {"calculation_unit": {"unknown_runtime_field": "x"}}
        )
    path = tmp_path / "input.json"
    path.write_text(
        '{"calculation_unit":{"scope":"x"},"calculation_unit":{"scope":"y"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _load_strict_scientific_input_json(str(path))


@pytest.mark.parametrize(
    ("operation", "stage", "base_input"),
    [
        (
            "hypothesis_generation",
            "query",
            {"scientific_question": _QUESTION, "num_queries": 4},
        ),
        (
            "hypothesis_generation",
            "hypothesis",
            {
                "scientific_question": _QUESTION,
                "query_angle": "验证关键成键步骤",
                "literature_review": "",
                "num_hypotheses": 3,
            },
        ),
    ],
)
def test_optional_calculation_unit_is_projected_without_changing_question(
    operation: str,
    stage: str,
    base_input: dict,
) -> None:
    """有计算单元时各假设子阶段读取同一冻结字段，原题保持独立。"""

    enriched = dict(base_input)
    enriched["chemistry_context"] = _UNIT
    projected = _user_base(build_model_messages(operation, enriched, stage=stage))
    if "scientific_question" in projected:
        assert projected["scientific_question"] == _QUESTION
    assert projected["chemistry_context"]["calculation_unit"]["charge"] == 1
    assert projected["chemistry_context"]["calculation_unit"]["products"] == _UNIT[
        "calculation_unit"
    ]["products"]
    assert "/tmp/" not in json.dumps(projected, ensure_ascii=False)


@pytest.mark.parametrize("stage", ["relationship", "merge", "rank"])
def test_pairwise_comparison_prompts_keep_the_fixed_calculation_boundary(stage: str) -> None:
    """关系、合并和排名读取最小计算单元，避免越界候选在比较中获胜。"""

    base_input = {
        "scientific_question": _QUESTION,
        "hypothesis_a": {"strategy_name": "A", "reasoning": "ra"},
        "hypothesis_b": {"strategy_name": "B", "reasoning": "rb"},
        "chemistry_context": _UNIT,
    }
    if stage == "merge":
        base_input["synthesis_output_schema"] = {"type": "object"}
    projected = _user_base(
        build_model_messages("hypothesis_generation", base_input, stage=stage)
    )
    assert projected["chemistry_context"] == _project_chemistry_context(_UNIT)


def test_retrieval_model_schema_does_not_ask_model_for_caller_input() -> None:
    """计算单元只由调用方提供，不进入 Retrieval 模型的输出字段。"""

    schema = RetrievalAgent._chemistry_context_schema()
    assert "calculation_unit" not in schema["properties"]
    assert schema["required"] == list(schema["properties"])


def test_absent_calculation_unit_preserves_existing_planner_context_shape() -> None:
    """普通任务不因可选输入接线而丢失既有科学条件。"""

    scientific_context = {
        "scientific_question": _QUESTION,
        "chemistry_context": {
            "reaction_type": "addition",
            "charge": 0,
            "multiplicity": 1,
        },
    }
    projected = _user_base(
        build_model_messages(
            "workflow_generation",
            {
                "scientific_context": scientific_context,
                "hypothesis": {"strategy_name": "H", "reasoning": "R"},
            },
        )
    )
    assert projected["scientific_context"] == scientific_context


def test_planner_and_regeneration_keep_the_same_projected_unit() -> None:
    """Planner 与再生成视图均读取注册字段，而不接收本地路径。"""

    planner = _user_base(
        build_model_messages(
            "workflow_generation",
            {
                "scientific_context": {
                    "scientific_question": _QUESTION,
                    "chemistry_context": _UNIT,
                },
                "hypothesis": {"strategy_name": "H", "reasoning": "R"},
            },
        )
    )
    assert planner["scientific_context"]["scientific_question"] == _QUESTION
    assert planner["scientific_context"]["chemistry_context"]["calculation_unit"][
        "method_requirements"
    ]["endpoint"]["basis"] == "6-31G(d)"

    regeneration = _user_base(
        build_model_messages(
            "hypothesis_generation",
            {
                "scientific_context": {
                    "scientific_question": _QUESTION,
                    "initial_query_angles": ["angle"],
                    "initial_retrieval_phase": {
                        "keywords": [],
                        "literature_review": "",
                        "mechanistic_clues": [],
                        "limitations": [],
                        "chemistry_context": _UNIT,
                    },
                    "chemistry_context": _UNIT,
                },
                "current_query": "angle",
                "prior_hypothesis_outcomes": [],
                "next_batch_index": 2,
                "generation_config": {
                    "num_queries": 1,
                    "num_hypotheses_per_query": 3,
                    "top_n_hypotheses": 1,
                },
            },
            stage="regeneration",
            memory_enabled=False,
        )
    )
    assert regeneration["scientific_context"]["chemistry_context"][
        "calculation_unit"
    ]["needs_irc"] is True


def test_task_context_merge_preserves_original_question_fingerprint() -> None:
    """冻结计算单元不拼接问题，恢复合并也不改变问题指纹。"""

    retrieved = {"reaction_type": "addition", "charge": 0}
    merged = merge_supplementary_chemistry_context(retrieved, _UNIT)
    assert merged["calculation_unit"] == normalize_supplementary_scientific_input(
        _UNIT
    )["calculation_unit"]
    assert merged["reaction_type"] == "addition"
    assert scientific_question_fingerprint(_QUESTION) == scientific_question_fingerprint(
        _QUESTION
    )
    assert _QUESTION not in json.dumps(_UNIT, ensure_ascii=False)


def test_registered_projection_drops_untrusted_path_and_unknown_fields() -> None:
    """模型投影只保留注册字段，并清理科学文本中的绝对路径。"""

    projected = _project_chemistry_context(
        {
            "calculation_unit": {
                "scope": f"读取 {Path('/tmp/private/input.json')}",
                "charge": 1,
                "unknown": "must not appear",
            },
            "unknown_top_level": "must not appear",
        }
    )
    text = json.dumps(projected, ensure_ascii=False)
    assert "private/input.json" not in text
    assert "unknown" not in text
