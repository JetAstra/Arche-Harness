"""Case 3 作者主假设首点结构夹具与 Investigation prompt 合同测试。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from chemistry_multiagent.controllers.case23_reproduction import (
    CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS,
)
from chemistry_multiagent.investigation.author_hypothesis_import import (
    CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
    load_author_main_hypothesis,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
)
from chemistry_multiagent.memory.contracts import ContractValidationError
from chemistry_multiagent.utils.scientific_input import (
    normalize_initial_calculation_structures,
)


ROOT = Path(__file__).resolve().parents[1]
QUESTION = CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS["case3"]
INPUT_PATH = ROOT / "supplementary/case3_author_main_hypothesis_scientific_input.json"
SOURCE_PATH = ROOT / "supplementary/case3.json"


def test_case3_fixture_has_four_first_scan_points_and_no_results() -> None:
    """四个记录必须逐一来自 case3.json 首点，且夹具不得携带结果字段。"""

    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    structures = normalize_initial_calculation_structures(
        payload["initial_calculation_structures"]
    )
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))["dihedral"]["scan"]
    native_scan = payload["calculation_unit"]["method_requirements"]["scan"]
    assert native_scan["method"] == "wB97XD"
    assert native_scan["basis"] == "def2SVP"
    assert native_scan["solvent"] == "N,N-Dimethylacetamide"
    assert set(structures) == {"unsubstituted", "ortho", "meta", "para"}
    for identity, record in structures.items():
        assert record["coordinates_angstrom"] == source[identity]["scan_steps"][0][
            "atomcoords"
        ]
        assert record["atomic_numbers"] == source[identity]["atomnos"]
        scan_names = {
            item.strip() for item in source[identity]["scan_summary"]["scan_names"]
        }
        assert len(scan_names) == 1
        source_match = re.fullmatch(
            r"D\((\d+),(\d+),(\d+),(\d+)\)", next(iter(scan_names))
        )
        assert source_match is not None
        source_indices = [int(item) for item in source_match.groups()]
        assert record["coordinate_atom_indices"] == source_indices
        assert record["tool_coordinate_atom_indices"] == [
            item - 1 for item in source_indices
        ]
        assert [
            record["atomic_numbers"][item - 1] for item in source_indices
        ] == [35, 7, 7, 1]
        assert record["charge"] == source[identity]["charge"] == 0
        assert record["multiplicity"] == source[identity]["mult"] == 1
        assert record["atom_roles"] == ["Br", "N", "N", "H"]
        assert record["scan"] == {
            "coordinate_type": "dihedral",
            "start_degrees": 0,
            "end_degrees": 60,
            "step_degrees": 5,
            "point_count": 13,
            "tool_step_count": 12,
            "tool_value_unit": "degree",
            "relaxed": True,
            "method": "wB97XD",
            "basis": "def2-SVP",
            "solvent_model": "SMD(DMA)",
            "symmetry": "NoSymm",
        }

    def walk(value: object) -> list[str]:
        if isinstance(value, dict):
            return [key for key in value for _ in ([key],) if key in {
                "energy", "energy_ev", "scan_steps", "scan_summary", "ranking", "result"
            }] + [item for child in value.values() for item in walk(child)]
        if isinstance(value, list):
            return [item for child in value for item in walk(child)]
        return []

    assert walk(payload) == []
    text = json.dumps(payload, ensure_ascii=False)
    assert "12.8" not in text and "22.0" not in text


def test_case3_author_fixture_binds_question_and_published_input() -> None:
    """作者 fixture 必须绑定 Case3 题面、计算单元和独立首点结构。"""

    fixture = load_author_main_hypothesis(
        QUESTION,
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
        supplementary_scientific_input=json.loads(INPUT_PATH.read_text()),
    )
    assert fixture.case_profile == "case3_fixed_reproduction"
    assert fixture.initial_calculation_structures is not None
    assert set(fixture.initial_calculation_structures) == {
        "unsubstituted", "ortho", "meta", "para"
    }
    assert fixture.provenance["published_execution_artifact_sha256"] == (
        "718959095618dd644effbe09478d9dcf3ea3d7225320159015ef24e6ec21b95e"
    )
    visible = json.dumps(fixture.hypothesis, ensure_ascii=False)
    assert "12.8" not in visible and "22.0" not in visible
    with pytest.raises(ContractValidationError, match="题面不匹配"):
        load_author_main_hypothesis(
            QUESTION + " drift",
            author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
        )


def test_case3_initial_prompt_exposes_structures_without_case1_counterpath() -> None:
    """Case3 prompt 应提供四套首点结构，但不能泄漏 Case1 对映路径特例。"""

    fixture = load_author_main_hypothesis(
        QUESTION, author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE
    )
    request = build_initial_investigation_request(
        "task-case3",
        "batch-case3",
        "combination-case3",
        QUESTION,
        fixture.hypothesis,
        calculation_unit=fixture.calculation_unit,
        initial_calculation_structures=fixture.initial_calculation_structures,
        available_tools=("scan_reaction_coordinates",),
    )
    payload = json.loads(request["messages"][-1]["content"])
    assert set(payload["initial_calculation_structures"]) == {
        "unsubstituted", "ortho", "meta", "para"
    }
    assert "对映路径" not in request["messages"][0]["content"]
    assert "tool_coordinate_atom_indices（0-based）" in request["messages"][0][
        "content"
    ]
    assert "scan.tool_step_count" in request["messages"][0]["content"]
    assert payload["calculation_unit"]["method_requirements"]["scan"]["basis"] == (
        "def2SVP"
    )
    assert payload["calculation_unit"]["method_requirements"]["scan"]["solvent"] == (
        "N,N-Dimethylacetamide"
    )
    assert "12.8" not in request["messages"][-1]["content"]


def test_case3_structure_contract_rejects_scan_results() -> None:
    """普通初始结构合同不能被后续扫描结果字段冒充。"""

    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    payload["initial_calculation_structures"]["meta"]["scan_steps"] = []
    with pytest.raises(ValueError, match="字段必须严格"):
        normalize_initial_calculation_structures(
            payload["initial_calculation_structures"]
        )


def test_case3_structure_contract_rejects_wrong_minichem_index_projection() -> None:
    """MiniChem 0-based 索引必须与作者 1-based 索引逐项对应。"""

    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    payload["initial_calculation_structures"]["meta"][
        "tool_coordinate_atom_indices"
    ][0] += 1
    with pytest.raises(ValueError, match="MiniChem 0-based"):
        normalize_initial_calculation_structures(
            payload["initial_calculation_structures"]
        )
