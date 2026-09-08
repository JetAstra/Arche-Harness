"""Case 3 作者主假设 advisory 模式的公开行为合同测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from minichem_toolbox.backends.quantum_legacy import _coordinate_value
from chemistry_multiagent.controllers.case23_reproduction import (
    CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS,
)
from chemistry_multiagent.investigation.author_hypothesis_import import (
    CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE,
    CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
    build_author_fixture_retrieval_result,
    build_case3_advisory_constraint_semantics,
    load_author_main_hypothesis,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
    build_investigation_job_continuation_request,
    build_investigation_job_wake_request,
    build_repair_investigation_request,
)
from chemistry_multiagent.memory.contracts import (
    INVESTIGATION_STAGE_ARCHITECTURE_VERSION,
    validate_case_profile,
    validate_task_created_payload,
)
from chemistry_multiagent.utils.gaussian_routing import (
    CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE,
    derive_gaussian_route_policy,
)


QUESTION = CASE23_AUTHOR_SCIENTIFIC_QUESTION_INPUTS["case3"]
ROOT = Path(__file__).resolve().parents[1]
FIXED_INPUT = ROOT / "supplementary/case3_author_main_hypothesis_scientific_input.json"


def _policy() -> dict[str, object]:
    """返回测试所用的公开 advisory policy。"""

    return build_case3_advisory_constraint_semantics()


def _fixture():
    """读取 advisory 模式的四套初始结构。"""

    payload = json.loads(FIXED_INPUT.read_text(encoding="utf-8"))
    return load_author_main_hypothesis(
        QUESTION,
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE,
        supplementary_scientific_input=payload,
    )


def test_advisory_import_freezes_only_structures_and_policy() -> None:
    """advisory 导入复用四套首点事实，但不把作者结果带进模型输入。"""

    fixed = load_author_main_hypothesis(
        QUESTION,
        author_mode=CASE3_AUTHOR_MAIN_HYPOTHESIS_MODE,
    )
    advisory = _fixture()

    assert advisory.author_mode == CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE
    assert advisory.case_profile == CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE
    assert advisory.initial_calculation_structures == fixed.initial_calculation_structures
    assert advisory.constraint_semantics == _policy()
    serialized = json.dumps(advisory.constraint_semantics, ensure_ascii=False)
    assert "advisory" in serialized
    assert all(marker not in serialized for marker in ("12.8", "22.0", "energy", "ranking"))
    assert "wB97XD" in serialized
    assert "def2SVP" in serialized
    assert "SMD(DMA)" in serialized
    assert "N,N-Dimethylacetamide" in serialized
    assert "优先尝试作者基线" in serialized
    assert "仍可调整" in serialized
    relaxed_scan = advisory.constraint_semantics["advisory_references"]["relaxed_scan"]
    assert relaxed_scan["scan_origin"] == "input_geometry"
    assert relaxed_scan["nominal_positive_displacement_degrees"] == 60
    assert relaxed_scan["absolute_tool_endpoints_degrees"] == {
        "meta": {"start": -0.020536359293, "end": 59.979463640707},
        "ortho": {"start": -4.040493659638, "end": 55.959506340362},
        "para": {"start": -0.007200953529, "end": 59.992799046471},
        "unsubstituted": {"start": 0.003210601878, "end": 60.003210601878},
    }
    for identity, record in advisory.initial_calculation_structures.items():
        derived_start = _coordinate_value(
            record["coordinates_angstrom"],
            "dihedral",
            record["tool_coordinate_atom_indices"],
        )
        endpoints = relaxed_scan["absolute_tool_endpoints_degrees"][identity]
        assert endpoints["start"] == pytest.approx(derived_start, abs=1e-12)
        assert endpoints["end"] - endpoints["start"] == pytest.approx(60.0)
    retrieval_result = build_author_fixture_retrieval_result(advisory)
    context_summary = retrieval_result["chemistry_context"]["context_summary"]
    assert "结构、电子态和坐标索引是不可变事实" in context_summary
    assert "方法和扫描参数是 advisory reference" in context_summary
    assert "比较范围由结构化计算单元冻结" not in context_summary


def test_advisory_policy_is_explicit_in_initial_wake_continuation_and_repair() -> None:
    """四类 Investigation 请求都携带同一份 advisory policy 和区分提示。"""

    fixture = _fixture()
    policy = _policy()
    initial = build_initial_investigation_request(
        "task-case3-advisory",
        "batch-case3-advisory",
        "combination-case3-advisory",
        QUESTION,
        fixture.hypothesis,
        calculation_unit=fixture.calculation_unit,
        initial_calculation_structures=fixture.initial_calculation_structures,
        constraint_semantics=policy,
        available_tools=("scan_reaction_coordinates",),
    )
    initial_payload = json.loads(initial["messages"][-1]["content"])
    assert initial_payload["constraint_semantics"] == policy
    assert "immutable" in initial["messages"][0]["content"].lower()
    assert "advisory" in initial["messages"][0]["content"].lower()
    assert "不得把 advisory reference 当作硬约束" in initial["messages"][0]["content"]
    assert "structure.scan" in initial["messages"][0]["content"]
    assert "start_degrees" in initial["messages"][0]["content"]
    assert "end_degrees" in initial["messages"][0]["content"]
    assert "step_degrees" in initial["messages"][0]["content"]
    assert "calculation_unit.method_requirements" in initial["messages"][0]["content"]
    assert "N,N-Dimethylacetamide" in initial["messages"][0]["content"]
    assert "优先尝试作者基线" in initial["messages"][0]["content"]
    assert "仍可调整" in initial["messages"][0]["content"]
    assert "从输入几何正向位移 60°" in initial["messages"][0]["content"]
    assert "absolute_tool_endpoints_degrees" in initial["messages"][0]["content"]
    assert "steps 必须使用 scan.tool_step_count" not in initial["messages"][0]["content"]

    wake = build_investigation_job_wake_request(
        "task-case3-advisory",
        "batch-case3-advisory",
        "combination-case3-advisory",
        "thread-case3-advisory",
        job_id="job-case3-advisory",
        operation_id="operation-case3-advisory",
        submission_key="submission-case3-advisory",
        terminal_status="succeeded",
        scientific_question=QUESTION,
        hypothesis=fixture.hypothesis,
        calculation_unit=fixture.calculation_unit,
        initial_calculation_structures=fixture.initial_calculation_structures,
        constraint_semantics=policy,
        available_tools=(
            "submit_action_job",
            "get_execution_job",
            "collect_execution_job",
        ),
    )
    wake_payload = json.loads(wake["messages"][-1]["content"])
    assert wake_payload["constraint_semantics"] == policy

    continuation = build_investigation_job_continuation_request(
        "task-case3-advisory",
        "batch-case3-advisory",
        "combination-case3-advisory",
        "thread-case3-advisory",
        job_id="job-case3-advisory",
        operation_id="operation-case3-advisory",
        submission_key="submission-case3-advisory",
        scientific_question=QUESTION,
        hypothesis=fixture.hypothesis,
        calculation_unit=fixture.calculation_unit,
        initial_calculation_structures=fixture.initial_calculation_structures,
        constraint_semantics=policy,
        available_tools=(
            "submit_action_job",
            "get_execution_job",
            "collect_execution_job",
        ),
    )
    continuation_payload = json.loads(continuation["messages"][-1]["content"])
    assert continuation_payload["constraint_semantics"] == policy

    repair = build_repair_investigation_request(
        "task-case3-advisory",
        "batch-case3-advisory",
        "combination-case3-advisory",
        "thread-case3-advisory",
        "v1",
        "补充验证并说明是否调整 advisory reference。",
        constraint_semantics=policy,
    )
    repair_payload = json.loads(repair["messages"][-1]["content"])
    assert repair_payload["constraint_semantics"] == policy
    for request in (initial, wake, continuation, repair):
        system_prompt = request["messages"][0]["content"]
        assert "advisory" in system_prompt.lower()
        assert "structure.scan" in system_prompt
        assert "calculation_unit.method_requirements" in system_prompt
        assert "steps 必须使用 scan.tool_step_count" not in system_prompt


def test_advisory_profile_is_validated_and_frozen_in_route_policy() -> None:
    """advisory 使用独立 profile，并把它冻结到 route policy。"""

    assert validate_case_profile(CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE) == (
        CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE
    )
    route_policy = derive_gaussian_route_policy(
        QUESTION,
        case_profile=CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE,
    )
    assert route_policy.case_profile == CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE


def test_advisory_task_created_contract_freezes_mode_and_policy() -> None:
    """TaskCreated 接受 advisory 组合，并保存 profile、mode 和 policy。"""

    fixture = _fixture()
    payload = validate_task_created_payload(
        {
            "scientific_question": QUESTION,
            "chemistry_context": {
                "calculation_unit": fixture.calculation_unit,
                "initial_calculation_structures": fixture.initial_calculation_structures,
                "constraint_semantics": fixture.constraint_semantics,
            },
            "task_config": {
                "num_queries": 1,
                "num_hypotheses_per_query": 1,
                "top_n_hypotheses": 1,
                "max_parallel_investigations": 1,
                "max_investigation_repairs_per_combination": 3,
                "max_hypothesis_batches": 1,
                "investigation_combination_count": 1,
                "investigation_mode": CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE,
            },
            "task_output_relative_dir": "outputs/research_tasks/task-case3-advisory",
            "tool_catalog_hash": "a" * 64,
            "stage_architecture_version": INVESTIGATION_STAGE_ARCHITECTURE_VERSION,
            "case_profile": CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE,
            "hypothesis_import": {
                "mode": CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE,
                "fixture_id": fixture.fixture_id,
                "fixture_sha256": fixture.fixture_sha256,
                "manifest_schema": "arche.investigation.author_hypothesis_import.v1",
                "expected_combination_count": 1,
            },
        }
    )
    assert payload["case_profile"] == CASE3_AUTHOR_HYPOTHESIS_ADVISORY_PROFILE
    assert payload["task_config"]["investigation_mode"] == (
        CASE3_AUTHOR_MAIN_HYPOTHESIS_ADVISORY_MODE
    )
    assert payload["chemistry_context"]["constraint_semantics"] == _policy()
