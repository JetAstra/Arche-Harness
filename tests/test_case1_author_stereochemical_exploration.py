"""Investigation 各类请求的通用 TS 初猜证据规则测试。"""

from __future__ import annotations

import json
from typing import Any

from chemistry_multiagent.investigation.author_hypothesis_import import (
    load_case1_author_main_hypothesis,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
    build_investigation_job_continuation_request,
    build_investigation_job_wake_request,
    build_repair_investigation_request,
)
from chemistry_multiagent.utils.model_prompts import build_model_messages


QUESTION = (
    "I plan to verify the mechanism of an asymmetric catalytic reaction using the reaction "
    "of p-nitrobenzaldehyde O=Cc1ccc(cc1)[N+](=O)[O-] with CC(=O)C under the catalysis "
    "of C(F)(F)(F)S(=O)(=O)O and N1CCC[C@H]1C(N2CCCC2), at 30°C in acetone solvent, "
    "to obtain the product CC(=O)C(O)c1ccc([N+](=O)[O-])cc1. Could you provide a "
    "specific approach to validate the reaction mechanism?"
)


def _message_text(request: dict[str, Any]) -> str:
    """拼接模型可见 system/user 正文，供边界断言使用。"""

    return "\n".join(str(item.get("content", "")) for item in request["messages"])


def _system_text(request: dict[str, Any]) -> str:
    """返回唯一 system prompt，避免把 user 假设文字混入规则断言。"""

    return str(request["messages"][0]["content"])


def _assert_generic_ts_guidance(system: str) -> None:
    """检查所有 Investigation 情景共同使用的 TS 初猜证据规则。"""

    for marker in (
        "search_transition_state_path",
        "transition_state_initial_guess_source 已声明",
        "必须使用该字段指定的来源",
        "必须实际调用 TS Pipeline",
        "不得自行替换成其他初猜算法",
        "未验证",
        "表观能量",
        "表观手性",
        "不能据此提前淘汰候选",
        "不能据此进行最终排序",
        "每一个候选初猜",
        "分别进行 Gaussian TS 优化和频率验证",
        "在所有候选完成这些 Gaussian 验证之前",
        "不得丢弃任一候选",
        "双向 IRC",
        "Gaussian 收敛情况",
        "IRC 连通性",
        "可比的能量或自由能",
        "initial_transition_state_structures",
    ):
        assert marker in system
    for forbidden in (
        "作者主假设",
        "两条对映路径",
        "TS-2a",
        "TS-2b-1",
        "allow_author_stereochemical_counterpath",
    ):
        assert forbidden not in system


def _author_fixture_input() -> tuple[dict[str, Any], dict[str, Any]]:
    """读取真实作者夹具，返回 hypothesis 与单端点 calculation unit。"""

    fixture = load_case1_author_main_hypothesis(QUESTION)
    return dict(fixture.hypothesis), dict(fixture.calculation_unit)


def test_initial_prompt_exposes_generic_ts_candidate_evidence_rules() -> None:
    """首次 Investigation 只接收通用候选规则，不按作者假设扩展路线。"""

    hypothesis, unit = _author_fixture_input()
    request = build_initial_investigation_request(
        "task-author",
        "batch-author",
        "combination-author",
        QUESTION,
        hypothesis,
        calculation_unit=unit,
    )
    system = _system_text(request)
    payload = json.loads(request["messages"][1]["content"])

    _assert_generic_ts_guidance(system)
    assert payload["calculation_unit"]["products"] == [
        "C/C(C[C@@H](O)c1ccc([N+](=O)[O-])cc1)=[N+]1/CCCC1CN1CCCC1"
    ]


def test_wake_prompt_exposes_generic_ts_candidate_evidence_rules() -> None:
    """外部中断恢复仍使用同一套通用候选证据规则。"""

    hypothesis, unit = _author_fixture_input()
    request = build_investigation_job_wake_request(
        "task-author",
        "batch-author",
        "combination-author",
        "thread-author",
        job_id="job-author-1",
        operation_id="operation-author-1",
        submission_key="submission-author-1",
        terminal_status="succeeded",
        scientific_question=QUESTION,
        hypothesis=hypothesis,
        calculation_unit=unit,
    )
    system = _system_text(request)

    _assert_generic_ts_guidance(system)


def test_case_identifier_does_not_change_initial_ts_guidance() -> None:
    """普通假设与作者假设的 system prompt 必须使用完全相同的 TS 规则。"""

    author_hypothesis, author_unit = _author_fixture_input()
    author_request = build_initial_investigation_request(
        "task-author",
        "batch-author",
        "combination-author",
        QUESTION,
        author_hypothesis,
        calculation_unit=author_unit,
    )

    ordinary_request = build_initial_investigation_request(
        "task-ordinary",
        "batch-ordinary",
        "combination-ordinary",
        "验证一个普通反应单元。",
        {
            "hypothesis_id": "ordinary-hypothesis",
            "strategy_name": "比较给定端点",
            "reasoning": "只验证输入中声明的单一产物端点。",
        },
        calculation_unit={
            "reactants": ["CC", "O=C"],
            "products": ["CC=O"],
            "charge": 0,
            "multiplicity": 1,
        },
    )

    assert _system_text(author_request) == _system_text(ordinary_request)
    _assert_generic_ts_guidance(_system_text(ordinary_request))


def test_repair_prompt_exposes_generic_ts_candidate_evidence_rules() -> None:
    """返修调用也必须保留通用 TS 候选验证要求。"""

    request = build_repair_investigation_request(
        "task-repair",
        "batch-repair",
        "combination-repair",
        "thread-repair",
        "v1",
        "请根据上次工具的原始错误修正提交参数。",
    )

    _assert_generic_ts_guidance(_system_text(request))


def test_continuation_prompt_exposes_generic_ts_candidate_evidence_rules() -> None:
    """作业 continuation 也必须使用通用 TS 候选验证要求。"""

    hypothesis, unit = _author_fixture_input()
    request = build_investigation_job_continuation_request(
        "task-continuation",
        "batch-continuation",
        "combination-continuation",
        "thread-continuation",
        job_id="job-continuation",
        operation_id="operation-continuation",
        submission_key="submission-continuation",
        scientific_question=QUESTION,
        hypothesis=hypothesis,
        calculation_unit=unit,
    )

    _assert_generic_ts_guidance(_system_text(request))


def test_fixed_calculation_unit_hypothesis_prompt_still_rejects_global_mirroring() -> None:
    """Hypothesis 阶段的普通固定单元规则仍拒绝镜像或竞争路径扩展。"""

    request = build_model_messages(
        "hypothesis_generation",
        {
            "scientific_question": "验证固定反应单元。",
            "query_angle": "检查单一端点的 TS 连通性",
            "literature_review": "",
            "num_hypotheses": 1,
            "chemistry_context": {
                "calculation_unit": {
                    "reactants": ["CC", "O=C"],
                    "products": ["CC=O"],
                    "charge": 0,
                    "multiplicity": 1,
                }
            },
        },
        stage="hypothesis",
    )
    text = "\n".join(str(item.get("content", "")) for item in request)

    assert "不得补造、替换或镜像反应物、产物" in text
    assert "竞争路径" in text
