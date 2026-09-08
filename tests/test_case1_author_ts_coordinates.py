"""Case 1 作者 TS 初始坐标输入的严格合同与 Investigation 投影测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.controllers.chemistry_multiagent_controller import (
    _load_strict_scientific_input_json,
)
from chemistry_multiagent.investigation.author_hypothesis_import import (
    load_case1_author_main_hypothesis,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
)
from chemistry_multiagent.memory.contracts import ContractValidationError
from chemistry_multiagent.memory.projections import _project_chemistry_context
from chemistry_multiagent.utils.scientific_input import (
    merge_supplementary_chemistry_context,
    normalize_initial_transition_state_structures,
    normalize_supplementary_scientific_input,
    project_supplementary_scientific_input_from_context,
)


QUESTION = (
    "I plan to verify the mechanism of an asymmetric catalytic reaction using the reaction "
    "of p-nitrobenzaldehyde O=Cc1ccc(cc1)[N+](=O)[O-] with CC(=O)C under the catalysis "
    "of C(F)(F)(F)S(=O)(=O)O and N1CCC[C@H]1C(N2CCCC2), at 30°C in acetone solvent, "
    "to obtain the product CC(=O)C(O)c1ccc([N+](=O)[O-])cc1. Could you provide a "
    "specific approach to validate the reaction mechanism?"
)
_INPUT_PATH = Path("supplementary/case1_author_ts_scientific_input.json")
_SOURCE_PATH = Path("supplementary/case1.json")


def _load_input() -> dict[str, Any]:
    """读取版本控制的作者 TS 输入，不启动任何模型或工具。"""

    return _load_strict_scientific_input_json(str(_INPUT_PATH))


def _message_payload(request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """返回 Investigation 的 system 与 user JSON。"""

    system = str(request["messages"][0]["content"])
    payload = json.loads(request["messages"][1]["content"])
    assert isinstance(payload, dict)
    return system, payload


def test_versioned_cli_input_contains_exactly_two_safe_53_atom_ts_sets() -> None:
    """最终 CLI 文件必须提供两套、且每套只有受控坐标字段。"""

    normalized = _load_input()
    coordinates = normalized["initial_transition_state_structures"]
    assert set(coordinates) == {"TS-2a", "TS-2b-1"}
    for identity, record in coordinates.items():
        assert set(record) == {
            "atomic_numbers",
            "coordinates_angstrom",
            "charge",
            "multiplicity",
        }, identity
        assert len(record["atomic_numbers"]) == 53
        assert len(record["coordinates_angstrom"]) == 53
        assert all(len(coord) == 3 for coord in record["coordinates_angstrom"])
        assert record["charge"] == 1
        assert record["multiplicity"] == 1
    assert normalize_supplementary_scientific_input(normalized) == normalized
    assert (
        normalize_initial_transition_state_structures(coordinates)
        == coordinates
    )


def test_coordinate_fixture_matches_only_the_tracked_case1_opt_records() -> None:
    """受控夹具必须逐值来自 Case 1 opt 记录，且不复制结果字段。"""

    source = json.loads(_SOURCE_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(_INPUT_PATH.read_text(encoding="utf-8"))
    source_opt = source["opt"]
    fixture_structures = fixture["initial_transition_state_structures"]
    for identity in ("TS-2a", "TS-2b-1"):
        source_record = source_opt[identity]
        fixture_record = fixture_structures[identity]
        assert source_record["charge"] == 1
        assert source_record["mult"] == 1
        assert fixture_record == {
            "atomic_numbers": source_record["atomnos"],
            "coordinates_angstrom": source_record["atomcoords_final"],
            "charge": source_record["charge"],
            "multiplicity": source_record["mult"],
        }
    assert all(
        set(record)
        == {"atomic_numbers", "coordinates_angstrom", "charge", "multiplicity"}
        for record in fixture_structures.values()
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.pop("TS-2a"),
        lambda value: value.update({"TS-extra": copy.deepcopy(value["TS-2a"])}),
        lambda value: value["TS-2a"]["atomic_numbers"].pop(),
        lambda value: value["TS-2a"]["coordinates_angstrom"][0].pop(),
        lambda value: value["TS-2a"].update({"charge": 0}),
        lambda value: value["TS-2a"].update({"multiplicity": 2}),
        lambda value: value["TS-2a"].update({"answer": "forbidden"}),
    ],
)
def test_ts_coordinate_contract_rejects_shape_identity_and_answer_drift(mutator) -> None:
    """坐标数量、身份、电子态和未知答案字段任一漂移都必须失败。"""

    scientific_input = _load_input()
    coordinates = copy.deepcopy(scientific_input["initial_transition_state_structures"])
    mutator(coordinates)
    scientific_input["initial_transition_state_structures"] = coordinates
    with pytest.raises(ContractValidationError):
        load_case1_author_main_hypothesis(QUESTION, scientific_input)


def test_generic_contract_rejects_ambiguous_cclib_coordinate_names() -> None:
    """模型输入不能重新接受没有单位语义的 cclib 字段。"""

    coordinates = copy.deepcopy(_load_input()["initial_transition_state_structures"])
    record = coordinates["TS-2a"]
    record["atomnos"] = record.pop("atomic_numbers")
    record["atomcoords_final"] = record.pop("coordinates_angstrom")
    with pytest.raises(ValueError):
        normalize_initial_transition_state_structures(coordinates)


def test_author_import_and_initial_prompt_preserve_supplied_coordinates() -> None:
    """作者导入和首次 Investigation 必须把两套坐标作为已知初始输入传递。"""

    scientific_input = _load_input()
    fixture = load_case1_author_main_hypothesis(QUESTION, scientific_input)
    assert (
        fixture.initial_transition_state_structures
        == scientific_input["initial_transition_state_structures"]
    )

    request = build_initial_investigation_request(
        "task-author-ts",
        "batch-author-ts",
        "combination-author-ts",
        QUESTION,
        fixture.hypothesis,
        calculation_unit=fixture.calculation_unit,
        initial_transition_state_structures=fixture.initial_transition_state_structures,
    )
    system, payload = _message_payload(request)
    assert (
        payload["initial_transition_state_structures"]
        == scientific_input["initial_transition_state_structures"]
    )
    assert "initial_transition_state_structures" in system
    assert "只是本轮的候选初猜" in system
    assert "分别进行 Gaussian TS 优化和频率验证" in system
    assert "作者提供的两套 TS 初始坐标" not in system
    assert "TS-2a" not in system
    assert "TS-2b-1" not in system


def test_task_created_merge_and_resume_projection_are_idempotent() -> None:
    """TaskCreated 合并与恢复投影必须保留同一份显式 Å 坐标。"""

    scientific_input = _load_input()
    merged = merge_supplementary_chemistry_context(
        {"retrieval_source": "author_fixture"},
        scientific_input,
    )
    assert merged["initial_transition_state_structures"] == scientific_input[
        "initial_transition_state_structures"
    ]
    projected = project_supplementary_scientific_input_from_context(merged)
    assert projected == scientific_input
    assert normalize_supplementary_scientific_input(projected) == projected
    projected_context = _project_chemistry_context(merged)
    assert projected_context["initial_transition_state_structures"] == scientific_input[
        "initial_transition_state_structures"
    ]


def test_author_without_coordinates_uses_the_same_generic_prompt() -> None:
    """没有坐标时按计算单元声明强制使用初猜来源，不恢复旧作者分支。"""

    fixture = load_case1_author_main_hypothesis(QUESTION)
    request = build_initial_investigation_request(
        "task-author-old",
        "batch-author-old",
        "combination-author-old",
        QUESTION,
        fixture.hypothesis,
        calculation_unit=fixture.calculation_unit,
    )
    system, payload = _message_payload(request)
    assert payload["calculation_unit"]["transition_state_initial_guess_source"] == (
        "search_transition_state_path"
    )
    assert "transition_state_initial_guess_source 已声明" in system
    assert "必须使用该字段指定的来源" in system
    assert "必须实际调用 TS Pipeline" in system
    assert "不得自行替换成其他初猜算法" in system
    assert "只是本轮的候选初猜" in system
    assert "作者提供的两套 TS 初始坐标" not in system
    assert "initial_transition_state_structures" not in payload


def test_generic_input_without_coordinates_is_unchanged() -> None:
    """普通计算单元归一化仍只返回原有 calculation_unit。"""

    value = {"calculation_unit": {"scope": "只验证给定计算单元"}}
    assert normalize_supplementary_scientific_input(value) == value
