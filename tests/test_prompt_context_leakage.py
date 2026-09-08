"""检查生产 Prompt 的可见上下文边界和关键科学合同。"""

import re

import pytest

from chemistry_multiagent.agents.hypothesis_agent import HypothesisAgent
from chemistry_multiagent.agents.expert_model_agent import (
    DEBATE_JUDGE_PROMPT,
    GAUSSIAN_STEP_REVIEW_PROMPT,
    WORKFLOW_EXPERT_OPINION_PROMPT,
    WORKFLOW_GENERAL_OPINION_PROMPT,
)
from chemistry_multiagent.utils.expert_model_provider import _run_expert_startup_probe
from chemistry_multiagent.utils.model_prompts import build_model_messages


_PROMPT_CASES = (
    ("retrieval", "chemistry_context", {}, {}),
    ("retrieval", "keyword", {}, {}),
    ("retrieval", "document_qa", {"documents": []}, {}),
    ("retrieval", "literature_review", {"documents": []}, {}),
    ("workflow_generation", None, {}, {}),
    ("workflow_optimization", None, {}, {}),
    ("router", None, {}, {}),
    ("workflow_revision", None, {}, {"memory_enabled": True}),
    ("gaussian_expert", "technical_failure", {}, {}),
    ("gaussian_expert", "failed_check", {}, {}),
    ("hypothesis_generation", "query", {}, {}),
    ("hypothesis_generation", "hypothesis", {}, {}),
    ("hypothesis_generation", "relationship", {}, {}),
    ("hypothesis_generation", "merge", {}, {}),
    ("hypothesis_generation", "complement", {}, {}),
    ("hypothesis_generation", "conflict", {}, {}),
    ("hypothesis_generation", "rank", {}, {}),
    (
        "hypothesis_generation",
        "regeneration",
        {},
        {"memory_enabled": True},
    ),
)


_CONTEXT_LEAK_TERMS = (
    "作者坐标",
    "作者路线",
    "Case1",
    "Case2",
    "Case3",
    "case1",
    "case2",
    "case3",
    "Memory",
    "Working Memory",
    "持久化身份",
    "持久化 ID",
    "运行元数据",
    "运行结果",
    "运行路径",
    "工具目录",
    "Action Catalog",
    "MiniChem Action/Backend Catalog",
    "旧项目名",
    "外部增强",
)
_CASE_MARKER_PATTERN = re.compile(
    r"(?:\bcase[\s_-]*[123]\b|案例\s*[123一二三])",
    re.IGNORECASE,
)


def _prompt_text(operation, stage, base_input, **kwargs):
    call_kwargs = {"stage": stage} if stage is not None else {}
    call_kwargs.update(kwargs)
    return "\n".join(
        item["content"]
        for item in build_model_messages(operation, base_input, **call_kwargs)
    )


@pytest.mark.parametrize(
    "operation,stage,base_input,kwargs", _PROMPT_CASES
)
def test_all_model_prompt_messages_omit_internal_context_terms(
    operation, stage, base_input, kwargs
):
    """所有已登记场景都只向模型暴露当前阶段需要的语义。"""

    text = _prompt_text(operation, stage, base_input, **kwargs)
    assert _CASE_MARKER_PATTERN.search(text) is None, (operation, stage)
    for term in _CONTEXT_LEAK_TERMS:
        assert term not in text, (operation, stage, term)


def test_query_prompt_keeps_science_modeling_without_tool_summary():
    """Query 只描述可见科学问题，不提前注入工具能力或执行主链。"""

    query = _prompt_text(
        "hypothesis_generation",
        "query",
        {"scientific_question": "解释选择性", "num_queries": 1},
    )
    for phrase in (
        "可证伪的直接判别问题",
        "决定性 observable",
        "分子连接关系",
        "charge/multiplicity",
    ):
        assert phrase in query
    for removed in (
        "Action",
        "Backend",
        "producer",
        "task_tool_subgraph",
        "Gaussian 固定准备链",
        "MiniChem",
    ):
        assert removed not in query


def test_calculation_unit_rule_is_dynamic_and_has_no_case_marker():
    """只有实际提供计算单元时才加入范围规则，且规则不含评测代号。"""

    ordinary = _prompt_text(
        "hypothesis_generation",
        "query",
        {"scientific_question": "解释选择性", "num_queries": 1},
    )
    enriched = _prompt_text(
        "hypothesis_generation",
        "query",
        {
            "scientific_question": "解释选择性",
            "num_queries": 1,
            "chemistry_context": {
                "calculation_unit": {
                    "scope": "验证给定成键步骤",
                    "reactants": ["C=C", "C=O"],
                    "products": ["CCC=O"],
                }
            },
        },
    )
    marker = "calculation_unit 是调用方为本任务明确给出的计算单元"
    bounded_marker = "全部 query angle 只能验证同一个 calculation_unit"
    assert marker not in ordinary
    assert marker in enriched
    assert bounded_marker in enriched
    for broad_instruction in (
        "cross-disciplinary insights",
        "fundamental principles",
    ):
        assert broad_instruction in ordinary
        assert broad_instruction not in enriched
    for bounded_instruction in (
        "不得加入 calculation_unit 未明确给出的数值接受阈值、容差或范围",
        "不得使用外部/实验 benchmark、文献或高层基准",
        "不得比较竞争路径、Re/Si 或对映/立体选择性，也不得讨论 ee 或 ΔΔG",
    ):
        assert bounded_instruction in enriched
        assert bounded_instruction not in ordinary
    assert _CASE_MARKER_PATTERN.search(enriched) is None


def test_calculation_unit_hypothesis_scope_error_is_fixed_and_context_safe():
    """范围拒绝摘要不回显正文、任务身份、路径或评测标记。"""

    candidate = {
        "strategy_name": "越界候选",
        "reasoning": (
            "Replace the required B3LYP calculation with GFN2-xTB and use a gas-phase "
            "approximation. task-59d6a2d1-6f05-4d45-91d8-eb0125b7f131 "
            "/mnt/shared-storage-user/wangshijie/study/arche_harness case1"
        ),
    }
    with pytest.raises(ValueError, match="calculation_unit") as error:
        HypothesisAgent._normalize_generated_hypotheses(
            [candidate],
            calculation_unit={
                "reactants": ["C=C"],
                "products": ["C-C"],
                "method_requirements": {
                    "endpoint": {"method": "B3LYP", "basis": "6-31G(d)"}
                },
            },
        )
    error_text = str(error.value).lower()
    assert "gfn2" not in error_text
    assert "task-59d6a2d1" not in error_text
    assert "/mnt/" not in error_text
    assert _CASE_MARKER_PATTERN.search(error_text) is None


def test_scientific_candidate_contract_is_positive_and_auditable():
    """候选阶段保留正向结构来源与 producer/input/connection 合同。"""

    for stage in ("hypothesis", "regeneration", "rank"):
        candidate = _prompt_text(
            "hypothesis_generation",
            stage,
            {},
            **({"memory_enabled": True} if stage == "regeneration" else {}),
        )
        for phrase in (
            "scope",
            "charge/multiplicity",
            "decisive test/observable",
            "outcome mapping",
            "atomic capability IDs",
            "capability gap",
        ):
            assert phrase in candidate
        assert "作者" not in candidate
        assert "authored code" not in candidate


def test_ranker_and_gaussian_routes_keep_catalog_scientific_contracts():
    """Ranker 保留准入语义，Gaussian 路线只依据通用目录合同自主选择。"""

    rank = _prompt_text("hypothesis_generation", "rank", {})
    for phrase in (
        "硬门槛：先逐项核对 A/B 的 scope",
        "直接 observable 及双向 outcome mapping",
        "所有决定性能力存在且无关键 gap",
        "只有两方都通过后",
        "不能靠软标准获胜",
    ):
        assert phrase in rank

    for operation in (
        "workflow_generation",
        "workflow_optimization",
        "workflow_revision",
    ):
        kwargs = {"memory_enabled": True} if operation == "workflow_revision" else {}
        prompt = _prompt_text(operation, None, {}, **kwargs)
        for phrase in (
            "依据冻结 catalog 的 Action/Backend metadata",
            "selection_policy",
            "request/output contract",
            "tool_capability_gap",
        ):
            assert phrase in prompt
        for forbidden in (
            "generate_gaussian_code → xyz_to_gjf/sdf_to_gjf → Gaussian native job",
            "route_section",
            "route_parameters",
            "search_transition_state_path/transition_state_search",
        ):
            assert forbidden not in prompt


def test_expert_prompts_keep_only_their_scientific_json_contracts():
    """专家 Prompt 保留局部审查、证据和 JSON 合同，不暴露实现组织信息。"""

    gaussian = GAUSSIAN_STEP_REVIEW_PROMPT
    for phrase in (
        "approved",
        "revise",
        "reject",
        "parameter_patch",
        "mutable_fields",
        "approved is {status, reason}",
        "revise is {status, reason, parameter_patch}",
        "reject is {status, reason}",
    ):
        assert phrase in gaussian
    assert "requested_transition" not in gaussian

    for prompt in (
        WORKFLOW_EXPERT_OPINION_PROMPT,
        WORKFLOW_GENERAL_OPINION_PROMPT,
        DEBATE_JUDGE_PROMPT,
    ):
        for phrase in (
            "supports_hypothesis",
            "reliable_refutation",
            "insufficient_evidence",
            "evidence_refs",
        ):
            assert phrase in prompt
        for term in ("ARCHE", "Memory", "persistent", "directory", "path", "catalog"):
            assert term not in prompt


def test_expert_startup_probe_uses_neutral_health_payload():
    """启动探测只验证 JSON 健康响应，不把内部阶段名称发给模型。"""

    captured = []

    def provider(messages, **kwargs):
        captured.append((messages, kwargs))
        return {
            "model": "probe-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"ok":true}'},
                }
            ],
        }

    ok, facts, reason = _run_expert_startup_probe(
        provider,
        configured_model="probe-model",
        timeout_seconds=1.0,
    )
    assert ok is True
    assert facts["strict_json"] is True
    assert reason is None
    assert captured[0][0] == [
        {"role": "system", "content": 'Return exactly {"ok":true} as JSON.'},
        {"role": "user", "content": '{"probe":"ok"}'},
    ]
