"""Investigation prompt/response 合同的独立确定性测试。"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

import chemistry_multiagent.investigation.contracts as contracts
import chemistry_multiagent.investigation.prompts as prompts
from chemistry_multiagent.harness.base import HarnessRequest
from chemistry_multiagent.investigation.contracts import (
    FailReviewFinding,
    InvestigationResponse,
    PassReviewFinding,
    TaskFinalizationResponse,
    TaskReviewResponse,
    build_investigation_provider_json_schema,
    build_task_finalization_provider_json_schema,
    build_task_review_provider_json_schema,
    finalization_response_json_schema,
    investigation_response_json_schema,
    review_response_json_schema,
    validate_investigation_response,
    validate_task_finalization_response,
    validate_task_review_response,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
    build_investigation_job_continuation_request,
    build_investigation_job_wake_request,
    build_repair_investigation_request,
    build_task_finalization_request,
    build_task_review_request,
)


FORBIDDEN_STATUS_TERMS = {
    "waiting",
    "waiting_for_jobs",
    "inconclusive",
    "continue_investigation",
}

_MULTILINE_SCIENTIFIC_QUESTION = (
    "Assess the stereochemical outcome for SMILES "
    "C[C@H](O)/C=C\\C1=CC=CC=C1 at 30\N{DEGREE SIGN}C.\t"
    "Compare the computed result with the specified conditions.\n\n"
    "Experimentally reported enantiomeric excess: 88%."
)


def _result(
    hypothesis_execution_input_record_id: str,
    version: str = "1",
) -> dict[str, object]:
    """构造不含私有会话字段的最小公共结果。"""

    return {
        "research_task_id": "task-1",
        "investigation_context_id": "batch-1",
        "hypothesis_execution_input_record_id": hypothesis_execution_input_record_id,
        "result_version": version,
        "self_judgment": "supported",
        "narrative": f"对 {hypothesis_execution_input_record_id} 的回顾性计算过程。",
        "job_refs": [f"job-{hypothesis_execution_input_record_id}"],
        "operation_refs": [f"operation-{hypothesis_execution_input_record_id}"],
        "artifact_refs": [_source_ref(hypothesis_execution_input_record_id)],
        "observations": [f"观测-{hypothesis_execution_input_record_id}"],
        "limitations": [],
    }


def _review_results() -> list[dict[str, object]]:
    """构造五项当前结果，并故意让各组合共享同一个版本字符串。"""

    return [_result(f"combination-{index}", "1") for index in range(1, 6)]


def _source_ref(hypothesis_execution_input_record_id: str) -> str:
    """返回真实 MiniChem-style 原始科学产物引用。"""

    return "art_" + hashlib.sha256(
        f"source-{hypothesis_execution_input_record_id}".encode()
    ).hexdigest()[:32]


def _reader_id(hypothesis_execution_input_record_id: str) -> str:
    """返回与 source ref 不同命名空间值的 Evidence reader ID。"""

    return "art_" + hashlib.sha256(
        f"reader-{hypothesis_execution_input_record_id}".encode()
    ).hexdigest()[:32]


def _evidence_snapshot() -> dict[str, object]:
    """构造与五个组合版本配对一致的只读证据快照。"""

    return {
        "snapshot_id": "snapshot-1",
        "research_task_id": "task-1",
        "investigation_context_id": "batch-1",
        "result_pairs": [
            {
                "hypothesis_execution_input_record_id": f"combination-{index}",
                "result_version": "1",
                "artifacts": [
                    {
                        "artifact_id": _reader_id(f"combination-{index}"),
                        "source_artifact_ref": _source_ref(f"combination-{index}"),
                        "size": index,
                        "sha256": hashlib.sha256(
                            f"reader-{index}".encode()
                        ).hexdigest(),
                        "media_type": "application/json",
                    }
                ],
            }
            for index in range(1, 6)
        ],
    }


def _message_text(request: dict[str, object]) -> str:
    """提取请求中供模型阅读的文本，便于检查状态词和操作约束。"""

    messages = request["messages"]
    assert isinstance(messages, list)
    return "\n".join(str(item["content"]) for item in messages)


_PROVIDER_WIRE_ALLOWED_SCHEMA_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf",
        "description",
    }
)
_PROVIDER_WIRE_FORBIDDEN_SCHEMA_KEYS = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "uniqueItems",
        "minLength",
        "minItems",
        "maxItems",
        "oneOf",
        "const",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "minContains",
    }
)


def _iter_schema_nodes(schema: object) -> list[dict[str, object]]:
    """递归收集 schema 节点，供 provider wire 形状断言使用。"""

    if not isinstance(schema, dict):
        return []
    nodes = [schema]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for value in properties.values():
            nodes.extend(_iter_schema_nodes(value))
    items = schema.get("items")
    if isinstance(items, dict):
        nodes.extend(_iter_schema_nodes(items))
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        for value in any_of:
            nodes.extend(_iter_schema_nodes(value))
    return nodes


def _assert_strict_provider_wire_schema(schema: object) -> None:
    """断言 provider wire 只使用保守 strict 子集且对象字段全部 required。"""

    nodes = _iter_schema_nodes(schema)
    assert nodes
    for node in nodes:
        assert set(node) <= _PROVIDER_WIRE_ALLOWED_SCHEMA_KEYS
        assert not set(node) & _PROVIDER_WIRE_FORBIDDEN_SCHEMA_KEYS
        if node.get("type") == "object":
            properties = node.get("properties")
            assert isinstance(properties, dict)
            assert node.get("additionalProperties") is False
            assert node.get("required") == list(properties)


def test_all_investigation_and_review_builders_emit_provider_strict_wire_schema() -> None:
    """三个真实 prompt builder 的最终 response_schema 必须兼容 provider strict 子集。"""

    initial = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
    )
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        "result-1",
        "补充关键观测。",
    )
    review = build_task_review_request(
        "task-1",
        "batch-1",
        "判断科学问题。",
        _review_results(),
        _evidence_snapshot(),
    )

    for request in (initial, repair, review):
        _assert_strict_provider_wire_schema(request["response_schema"])
        Draft202012Validator.check_schema(request["response_schema"])

    review_schema = review["response_schema"]
    assert isinstance(review_schema, dict)
    review_properties = review_schema["properties"]
    assert isinstance(review_properties, dict)
    assert review_properties["final_answer"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    assert review_properties["answerability_explanation"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    canonical_investigation = investigation_response_json_schema()
    assert canonical_investigation["properties"]["job_refs"]["uniqueItems"] is True
    assert canonical_investigation["properties"]["narrative"]["minLength"] == 1
    canonical_review = review_response_json_schema()
    assert canonical_review["properties"]["findings"]["minItems"] == 1
    assert canonical_review["properties"]["findings"]["maxItems"] == 5
    assert "allOf" in canonical_review
    assert "oneOf" in canonical_review["properties"]["findings"]["items"]
    mutable_wire = build_investigation_provider_json_schema()
    mutable_wire["properties"]["job_refs"]["items"]["type"] = "integer"
    fresh_wire = build_investigation_provider_json_schema()
    assert fresh_wire["properties"]["job_refs"]["items"]["type"] == "string"


def test_public_contract_api_exposes_domain_and_provider_schema_names() -> None:
    """公共 API 同时暴露 domain canonical 合同、provider wire schema 和校验入口。"""

    assert contracts.__all__ == [
        "InvestigationResponse",
        "PassReviewFinding",
        "FailReviewFinding",
        "TaskReviewResponse",
        "TaskFinalizationFinding",
        "TaskFinalizationResponse",
        "investigation_response_json_schema",
        "review_response_json_schema",
        "finalization_response_json_schema",
        "build_investigation_provider_json_schema",
        "build_task_review_provider_json_schema",
        "build_task_finalization_provider_json_schema",
        "validate_investigation_response",
        "validate_task_review_response",
        "validate_task_finalization_response",
        "require_safe_public_scientific_text",
        "require_safe_public_scientific_context",
        "require_safe_scientific_question_text",
    ]
    assert prompts.__all__ == [
        "INVESTIGATION_COMPACT_PROMPT",
        "INVESTIGATION_COMPACT_PROMPT_SHA256",
        "INVESTIGATION_MODEL_CONTEXT_WINDOW",
        "build_initial_investigation_request",
        "build_repair_investigation_request",
        "build_investigation_job_wake_request",
        "build_investigation_job_continuation_request",
        "build_task_review_request",
        "build_task_finalization_request",
    ]


def test_investigation_compact_prompt_matches_reviewed_minimal_policy() -> None:
    """生产压缩提示词固定为已审查正文，不泄露无关运行架构。"""

    prompt = prompts.INVESTIGATION_COMPACT_PROMPT
    assert prompts.INVESTIGATION_COMPACT_PROMPT_SHA256 == (
        "3ead8b55a0d4ce9bfe2629ece18d51834bec0699d64c21fa04aaa313f8af1200"
    )
    assert prompts.INVESTIGATION_MODEL_CONTEXT_WINDOW == 1_000_000
    assert prompt.count("## Scientific reasoning") == 1
    assert prompt.count("## Computational jobs and results") == 1
    assert "You are a recorder, not an investigator." in prompt
    assert "Collapse repeated polling" in prompt
    assert "all non-secret identifiers needed to query or retrieve" in prompt
    for unrelated_term in (
        "ARCHE",
        "DeepSeek",
        "MiniChem",
        "Provider",
        "Working Memory",
        "Run Bundle",
        "9011",
        "9012",
        "worktree",
    ):
        assert unrelated_term not in prompt


def test_public_scientific_context_copies_nested_json_and_keeps_refs_strict() -> None:
    """公开科学上下文复制嵌套 JSON，同时让 artifact 引用继续严格校验。"""

    source = {
        "notes": ["第一行\n第二行", None, True, 7, 0.5],
        "artifact": {
            "artifact_id": "artifact-1",
            "values": [None, True, 7, 0.5],
        },
    }

    normalized = contracts.require_safe_public_scientific_context(source)

    assert normalized == source
    assert normalized is not source
    assert normalized["artifact"] is not source["artifact"]


def test_initial_request_is_structured_deterministic_and_has_strict_schema() -> None:
    """首次请求按固定 JSON 编码，并说明统一作业查询和证据边界。"""

    first = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
        explicit_conditions={"temperature": 303, "solvent": "water"},
        frozen_context={"facts": ["冻结事实"]},
    )
    second = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"strategy_name": "策略 A", "reasoning": "精简假设理由"},
        explicit_conditions={"solvent": "water", "temperature": 303},
        frozen_context={"facts": ["冻结事实"]},
    )

    assert first["messages"] == second["messages"]
    assert first["response_schema"] == build_investigation_provider_json_schema()
    text = _message_text(first)
    assert "同一 job_id 协议" in text
    assert "Python" in text
    assert "Gaussian" in text
    assert "提交后约 1、5、10、20、30 分钟查询，之后每约 30 分钟持续查询" in text
    assert "collect" in text
    assert "tool_capability_gap" in text
    assert "提交结果不确定" in text
    assert "submit_transition_state_path_job" not in text
    assert "get_transition_state_path_job" not in text
    assert "status" not in first["response_schema"]
    assert all(term not in text.lower() for term in FORBIDDEN_STATUS_TERMS)
    assert set(first) >= {"model", "role", "stage", "profile", "messages", "response_schema"}
    assert first["profile"] == "investigation"


def test_model_visible_investigation_prompts_use_one_job_protocol() -> None:
    """模型只看到统一 job_id 协议，不看到内部路由或具体工具族。"""

    initial = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
    )
    repair = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        "result-1",
        "补充关键观测。",
    )
    wake = build_investigation_job_wake_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        job_id="job-1",
        operation_id="operation-1",
        submission_key="submission-1",
        terminal_status="succeeded",
    )

    forbidden = (
        "Controller",
        "sidecar",
        "job_family",
        "恰好五项",
        "submit_transition_state_path_job",
        "get_transition_state_path_job",
        "collect_transition_state_path_job",
        "submit_native_job",
        "get_execution_job",
        "collect_execution_job",
    )
    continuation = build_investigation_job_continuation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "thread-1",
        job_id="job-1",
        operation_id="operation-1",
        submission_key="submission-1",
        action_id="action-1",
        backend_id="backend-1",
        scientific_question="判断科学问题。",
        hypothesis={"reasoning": "精简假设理由"},
    )

    for request in (initial, repair, wake, continuation):
        visible = _message_text(request)
        assert "job_id" in visible
        assert "collect" in visible
        assert "tool_capability_gap" in visible
        assert "计算任务可能运行很长时间，尤其是 Gaussian" in visible
        assert "不得取消、中断、放弃或重新提交该任务" in visible
        assert "不得因为等待时间长而提前结束调查" in visible
        assert "持续查询同一 job_id，直到得到明确终态后再 collect" in visible
        assert all(term not in visible for term in forbidden)

    initial_visible = _message_text(initial)
    repair_visible = _message_text(repair)
    assert "Python" in initial_visible and "Gaussian" in initial_visible
    assert "Python" in repair_visible and "Gaussian" in repair_visible
    cadence = "提交后约 1、5、10、20、30 分钟查询，之后每约 30 分钟持续查询"
    assert cadence in initial_visible
    assert cadence in repair_visible
    wake_visible = str(wake["messages"][0]["content"])
    assert "外部中断" in wake_visible
    assert "系统通知了一个已结束" not in wake_visible
    continuation_visible = str(continuation["messages"][0]["content"])
    continuation_payload = json.loads(continuation["messages"][1]["content"])
    assert continuation["metadata"]["request_kind"] == "investigation_job_continuation"
    assert "terminal_status" not in continuation["metadata"]
    assert "terminal_status" not in continuation_payload
    assert "list_execution_jobs" in continuation_visible
    assert "仍在运行" in continuation_visible
    assert "原始错误信息" in continuation_visible
    assert cadence in continuation_visible
    assert cadence in wake_visible
    for visible in (initial_visible, repair_visible, wake_visible, continuation_visible):
        assert "约每 5 分钟查询一次" not in visible
    assert "第一项" in continuation_visible
    assert "历史 turn 中的 get" in continuation_visible
    assert "本次 turn 的 get 回执" in continuation_visible


def test_all_investigation_prompts_forbid_partial_results_and_require_failure() -> None:
    """全部调查入口都区分科学完成、作业终态和无法补齐后的执行失败。"""

    requests = (
        build_initial_investigation_request(
            "task-1",
            "batch-1",
            "combination-1",
            "判断科学问题。",
            {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
        ),
        build_repair_investigation_request(
            "task-1",
            "batch-1",
            "combination-1",
            "thread-1",
            "result-1",
            "补齐缺失实验。",
        ),
        build_investigation_job_wake_request(
            "task-1",
            "batch-1",
            "combination-1",
            "thread-1",
            job_id="job-1",
            operation_id="operation-1",
            submission_key="submission-1",
            terminal_status="failed",
        ),
        build_investigation_job_continuation_request(
            "task-1",
            "batch-1",
            "combination-1",
            "thread-1",
            job_id="job-1",
            operation_id="operation-1",
            submission_key="submission-1",
            action_id="action-1",
            backend_id="backend-1",
            scientific_question="判断科学问题。",
            hypothesis={"reasoning": "精简假设理由"},
        ),
    )

    for request in requests:
        visible = request["messages"][0]["content"]
        assert "全部必做" in visible
        assert "可用于科学判断的结果" in visible
        assert "所有作业已进入终态" in visible
        assert "不得提交" in visible
        assert "InvestigationResult" in visible
        assert "执行失败" in visible
        assert "已完成项" in visible
        assert "不得标记为成功" in visible
        assert "Case 1" not in visible
        assert "Case 2" not in visible
        assert "Case 3" not in visible


def test_investigation_schema_prefix_only_describes_final_json_contract() -> None:
    """Harness schema 前导只说明最终 JSON，不复制作业状态机。"""

    compiled = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
        tool_descriptors=[{"name": "internal_tool_summary", "description": "内部摘要"}],
    )
    request = HarnessRequest.from_compiled(
        compiled,
        role="investigation",
        stage="initial_investigation",
        workspace=Path("/tmp/investigation-prompt-test"),
        response_schema=compiled["response_schema"],
        tools=tuple(compiled.get("tools", ())),
    )
    prompt = request.prompt_text()

    assert "最终结果必须符合以下 JSON schema" in prompt
    assert "self_judgment" in prompt
    assert "internal_tool_summary" not in prompt
    assert "只有所有已提交作业都已按 submit→wait/get→终态→collect 顺序完成" not in prompt
    assert "任一时刻只能有一个仍活跃且未 collect 的作业" not in prompt


def test_initial_request_accepts_multiline_scientific_question_and_smiles() -> None:
    """首次 Investigation 请求保留公开科学正文中的换行、制表符和 SMILES。"""

    request = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        _MULTILINE_SCIENTIFIC_QUESTION,
        {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
    )

    payload = json.loads(request["messages"][1]["content"])
    assert payload["scientific_question"] == _MULTILINE_SCIENTIFIC_QUESTION


def test_initial_request_accepts_multiline_nested_scientific_context() -> None:
    """initial prompt 保留嵌套 Retrieval 正文的排版空白和公开来源。"""

    frozen_context = {
        "retrieval": {
            "literature_review": "第一段。\n\n第二段。\t保留制表符。",
            "answer": "回答第一行。\r\n回答第二行。",
            "source": "https://doi.org/10.1234/example",
        }
    }
    request = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断科学问题。",
        {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
        explicit_conditions={"notes": "条件第一行。\n条件第二行。"},
        frozen_context=frozen_context,
    )

    payload = json.loads(request["messages"][1]["content"])
    assert payload["frozen_context"] == frozen_context
    assert payload["explicit_conditions"] == {
        "notes": "条件第一行。\n条件第二行。"
    }


@pytest.mark.parametrize(
    "unsafe_context",
    [
        {"nested": {"literature_review": "正文\n含 NUL\x00"}},
        {"nested": {"answer": "正文\n含 ESC\x1b"}},
        {"nested": {"answer": "正文\n/mnt/private/input.xyz"}},
        {"nested": {"answer": "正文\nfile:///tmp/private/input.xyz"}},
        {"nested": {"answer": "正文\nhttp://127.0.0.1:9010/mcp"}},
        {"nested": {"answer": "说明\napi_key=synthetic-test-value"}},
        {"nested": {"research_task_id": "task-1\ncontinued"}},
        {"nested": {"artifact_ref": "https://doi.org/10.1234/example"}},
        {"nested": {"research_task_id": {"api_key": "synthetic-test-value"}}},
    ],
)
def test_initial_request_rejects_unsafe_nested_scientific_context(
    unsafe_context: dict[str, object],
) -> None:
    """initial prompt 对任意嵌套科学上下文保持危险值和身份引用的严格边界。"""

    with pytest.raises(ValueError, match="路径|URI|控制字符|凭据|不能包含"):
        build_initial_investigation_request(
            "task-1",
            "batch-1",
            "combination-1",
            "判断科学问题。",
            {"reasoning": "精简假设理由", "strategy_name": "策略 A"},
            frozen_context=unsafe_context,
        )


def test_review_request_accepts_multiline_scientific_question() -> None:
    """任务级 Review 请求也保留公开科学正文的排版空白。"""

    request = build_task_review_request(
        "task-1",
        "batch-1",
        _MULTILINE_SCIENTIFIC_QUESTION,
        _review_results(),
        _evidence_snapshot(),
    )

    payload = json.loads(request["messages"][1]["content"])
    assert payload["scientific_question"] == _MULTILINE_SCIENTIFIC_QUESTION


def test_repair_request_only_targets_one_version_and_keeps_original_thread() -> None:
    """返修 user payload 只含指定版本和建议，路由元数据保存原 thread。"""

    request = build_repair_investigation_request(
        "task-1",
        "batch-1",
        "combination-2",
        "thread-original",
        "1",
        "补充过渡态频率检查。",
        repair_count=2,
    )
    messages = request["messages"]
    assert isinstance(messages, list)
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"result_version", "repair_advice"}
    assert payload["result_version"] == "1"
    assert payload["repair_advice"] == "补充过渡态频率检查。"
    assert request["thread_id"] == "thread-original"
    assert request["metadata"]["codex_thread_id"] == "thread-original"
    assert all(term not in _message_text(request).lower() for term in FORBIDDEN_STATUS_TERMS)


def test_task_review_requires_five_current_pairs_and_preserves_unchanged_results() -> None:
    """Review barrier 要求五个组合，版本相同仍按组合配对且完整保留。"""

    results = _review_results()
    request = build_task_review_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        list(reversed(results)),
        _evidence_snapshot(),
        evidence_reader_tool={
            "name": "read_snapshot_artifact",
            "kind": "readonly_evidence_reader",
        },
    )
    messages = request["messages"]
    assert isinstance(messages, list)
    payload = json.loads(messages[1]["content"])
    current = payload["current_results"]
    assert len(current) == 5
    assert [
        item["hypothesis_execution_input_record_id"] for item in current
    ] == [
        "combination-1",
        "combination-2",
        "combination-3",
        "combination-4",
        "combination-5",
    ]
    assert {item["result_version"] for item in current} == {"1"}
    assert payload["evidence_snapshot"] == _evidence_snapshot()
    assert "review_context" not in payload
    assert request["response_schema"] == build_task_review_provider_json_schema()
    assert request["profile"] == "review"
    assert request["tools"][0]["kind"] == "readonly_evidence_reader"
    assert "全部 current_results" in _message_text(request)
    assert "不要丢弃" in _message_text(request)
    assert "final_answer" in _message_text(request)
    assert "answerability_explanation" in _message_text(request)
    assert "两个字段都必须存在" in _message_text(request)
    assert "不适用" in _message_text(request)
    assert "返回 null" in _message_text(request)
    assert "Evidence artifact_id" in _message_text(request)
    assert "source_artifact_ref" in _message_text(request)
    assert "只读 evidence reader" in _message_text(request)
    assert "当前 user JSON 和 evidence_snapshot 是唯一可审查" in _message_text(request)
    assert "没有 MiniChem 提交权限" in _message_text(request)
    assert "当前输入或 evidence_snapshot 已经提供" in _message_text(request)
    assert "不得凭模型记忆新增文献" in _message_text(request)
    assert all(term not in _message_text(request).lower() for term in FORBIDDEN_STATUS_TERMS)


    history_request = build_task_review_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        results,
        _evidence_snapshot(),
        review_context={
            "author_single_combination": True,
            "closed_batches": [
                {"scientific_experience": "CROSS_BATCH_REVIEW_EXPERIENCE_MUST_STAY_PRIVATE"}
            ]
        },
    )
    history_payload_text = history_request["messages"][1]["content"]
    history_payload = json.loads(history_payload_text)
    assert "review_context" not in history_payload
    assert "CROSS_BATCH_REVIEW_EXPERIENCE_MUST_STAY_PRIVATE" not in history_payload_text
    assert history_request["messages"][0]["content"] == request["messages"][0]["content"]
    assert "恰好一项" not in _message_text(history_request)

    changed_results = _review_results()
    changed_results[1] = _result("combination-2", "2")
    changed_snapshot = _evidence_snapshot()
    changed_pairs = changed_snapshot["result_pairs"]
    assert isinstance(changed_pairs, list)
    changed_pairs[1] = {
        "hypothesis_execution_input_record_id": "combination-2",
        "result_version": "2",
        "artifacts": [
            {
                "artifact_id": _reader_id("combination-2"),
                "source_artifact_ref": _source_ref("combination-2"),
                "size": 2,
                "sha256": hashlib.sha256(b"reader-2").hexdigest(),
                "media_type": "application/json",
            }
        ],
    }
    changed_request = build_task_review_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        changed_results,
        changed_snapshot,
    )
    changed_payload = json.loads(changed_request["messages"][1]["content"])
    assert {
        (item["hypothesis_execution_input_record_id"], item["result_version"])
        for item in changed_payload["current_results"]
    } == {
        ("combination-1", "1"),
        ("combination-2", "2"),
        ("combination-3", "1"),
        ("combination-4", "1"),
        ("combination-5", "1"),
    }

    empty_ref_results = _review_results()
    for item in empty_ref_results:
        item["job_refs"] = []
        item["operation_refs"] = []
        item["artifact_refs"] = []
    empty_ref_snapshot = _evidence_snapshot()
    empty_ref_pairs = empty_ref_snapshot["result_pairs"]
    assert isinstance(empty_ref_pairs, list)
    for pair in empty_ref_pairs:
        assert isinstance(pair, dict)
        pair["artifacts"] = []
    build_task_review_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        empty_ref_results,
        empty_ref_snapshot,
    )

    four_snapshot = _evidence_snapshot()
    four_pairs = four_snapshot["result_pairs"]
    assert isinstance(four_pairs, list)
    four_snapshot["result_pairs"] = four_pairs[:4]
    four_request = build_task_review_request(
        "task-1", "batch-1", "问题", results[:4], four_snapshot
    )
    four_payload = json.loads(four_request["messages"][1]["content"])
    assert len(four_payload["current_results"]) == 4

    six_snapshot = _evidence_snapshot()
    six_pairs = six_snapshot["result_pairs"]
    assert isinstance(six_pairs, list)
    six_snapshot["result_pairs"] = [
        *six_pairs,
        {
            "hypothesis_execution_input_record_id": "combination-6",
            "result_version": "1",
            "artifacts": [],
        },
    ]
    with pytest.raises(ValueError, match="一至五项"):
        build_task_review_request(
            "task-1",
            "batch-1",
            "问题",
            results + [_result("combination-6")],
            six_snapshot,
        )
    with pytest.raises(ValueError, match="hypothesis_execution_input_record_id"):
        build_task_review_request(
            "task-1",
            "batch-1",
            "问题",
            results[:4] + [_result("combination-1", "2")],
            _evidence_snapshot(),
        )

    with pytest.raises(ValueError, match="read_snapshot_artifact"):
        build_task_review_request(
            "task-1",
            "batch-1",
            "原始科学问题。",
            results,
            _evidence_snapshot(),
            evidence_reader_tool={
                "name": "read_other_artifact",
                "kind": "readonly_evidence_reader",
            },
        )


def test_review_prompt_requires_repair_for_incomplete_experiments() -> None:
    """普通审查不能把缺失实验作为限制项放行，也不自行判断预算。"""

    request = build_task_review_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        _review_results(),
        _evidence_snapshot(),
    )
    visible = request["messages"][0]["content"]

    assert "缺失、失败、未收敛或不完整" in visible
    assert "validity 必须为 fail" in visible
    assert "必须补齐" in visible
    assert "repair_advice" in visible
    assert "can_answer_original_question=false" in visible
    assert "部分支持" in visible
    assert "所有作业已进入终态" in visible
    assert "task_finalization" in visible
    assert "不判断" in visible
    assert "返修预算是否耗尽" in visible
    assert "Case 1" not in visible
    assert "Case 2" not in visible
    assert "Case 3" not in visible


def test_finalization_prompt_returns_result_without_review_or_repair() -> None:
    """终局成稿请求明确要求最终结果，不再要求 Review 或返修。"""

    request = build_task_finalization_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        _review_results(),
        _evidence_snapshot(),
    )
    visible = request["messages"][0]["content"]
    payload = json.loads(request["messages"][-1]["content"])

    assert request["metadata"]["request_kind"] == "task_finalization"
    assert payload["request_kind"] == "task_finalization"
    assert "本次不再执行 Review" in visible
    assert "不得输出 validity" in visible
    assert "repair_advice" in visible
    assert "final_answer" in visible
    assert request["response_schema"] == build_task_finalization_provider_json_schema()
    finalization = validate_task_finalization_response(
        {
            "findings": [
                {
                    "hypothesis_execution_input_record_id": f"combination-{index}",
                    "summary": f"组合 {index} 摘要。",
                }
                for index in range(1, 6)
            ],
            "answerability": "unanswerable",
            "final_answer": "现有证据不足以得出更强结论。",
        }
    )
    assert isinstance(finalization, TaskFinalizationResponse)
    assert (
        finalization_response_json_schema()["properties"]["findings"]["maxItems"]
        == 5
    )


def test_review_prompt_is_identical_for_one_and_five_results() -> None:
    """Review 文案统一审查输入中的全部结果，数量由代码和 schema 约束。"""

    five_request = build_task_review_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        _review_results(),
        _evidence_snapshot(),
    )
    five_snapshot = _evidence_snapshot()
    one_request = build_task_review_request(
        "task-1",
        "batch-1",
        "原始科学问题。",
        [_result("combination-1")],
        {
            **five_snapshot,
            "result_pairs": five_snapshot["result_pairs"][:1],
        },
    )

    assert one_request["messages"][0] == five_request["messages"][0]
    visible = _message_text(one_request)
    assert "全部 current_results" in visible
    assert "恰好五项" not in visible
    assert "恰好一项" not in visible
    assert "五项" not in visible
    assert "五个" not in visible
    assert "字符串 replace" not in visible


def test_request_inputs_reject_private_paths_and_non_structured_hypothesis() -> None:
    """输入校验拒绝宿主路径和未结构化假设，同时不误伤公开科学来源。"""

    with pytest.raises(ValueError, match="路径"):
        build_initial_investigation_request(
            "task-1",
            "batch-1",
            "combination-1",
            "问题 /mnt/private/input.gjf",
            {"statement": "假设"},
        )
    with pytest.raises(ValueError, match="路径"):
        build_initial_investigation_request(
            "task-1",
            "batch-1",
            "combination-1",
            "问题",
            {"statement": "假设"},
            frozen_context={"source": "file:///tmp/secret"},
        )
    public_source = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "问题，能量单位 kcal/mol。",
        {"statement": "假设"},
        frozen_context={"source": "https://doi.org/10.1234/example"},
    )
    assert "https://doi.org/10.1234/example" in _message_text(public_source)
    with pytest.raises(TypeError, match="JSON object"):
        build_initial_investigation_request(
            "task-1",
            "batch-1",
            "combination-1",
            "问题",
            "未结构化假设",
        )


def test_investigation_response_text_remains_strictly_single_line() -> None:
    """公开科学问题的排版例外不应放宽 Investigation 响应正文合同。"""

    with pytest.raises(ValidationError, match="控制字符"):
        InvestigationResponse(
            self_judgment="supported",
            narrative="第一行\n第二行",
            job_refs=(),
            operation_refs=(),
            artifact_refs=(),
            observations=("观测",),
            limitations=(),
        )


def test_review_requires_complete_current_results_and_matching_evidence_snapshot() -> None:
    """Review 不接受省略科学字段，且快照必须覆盖同一五个身份配对。"""

    results = _review_results()
    for required_field in (
        "self_judgment",
        "narrative",
        "job_refs",
        "operation_refs",
        "artifact_refs",
        "observations",
        "limitations",
    ):
        incomplete = dict(results[0])
        incomplete.pop(required_field)
        with pytest.raises(ValueError, match="缺少完整结果字段"):
            build_task_review_request(
                "task-1",
                "batch-1",
                "问题",
                [incomplete, *results[1:]],
                _evidence_snapshot(),
            )

    mismatched_snapshot = _evidence_snapshot()
    pairs = mismatched_snapshot["result_pairs"]
    assert isinstance(pairs, list)
    pairs[0] = {
        "hypothesis_execution_input_record_id": "combination-1",
        "result_version": "2",
        "artifacts": [
            {
                "artifact_id": _reader_id("combination-1"),
                "source_artifact_ref": _source_ref("combination-1"),
                "size": 1,
                "sha256": hashlib.sha256(b"reader-1").hexdigest(),
                "media_type": "application/json",
            }
        ],
    }
    with pytest.raises(ValueError, match="配对"):
        build_task_review_request(
            "task-1",
            "batch-1",
            "问题",
            results,
            mismatched_snapshot,
        )

    with pytest.raises(ValueError, match="任务或 Context"):
        build_task_review_request(
            "task-1",
            "batch-1",
            "问题",
            results,
            {**_evidence_snapshot(), "research_task_id": "other-task"},
        )


def test_review_binds_source_refs_and_reader_ids_in_separate_namespaces() -> None:
    """Review 必须按 source ref 对齐结果，并独立拒绝两类 ID 的重复/伪装。"""

    results = _review_results()

    missing_binding = _evidence_snapshot()
    missing_binding["result_pairs"][0]["artifacts"][0].pop("source_artifact_ref")  # type: ignore[index]
    with pytest.raises(ValueError, match="缺少字段"):
        build_task_review_request(
            "task-1", "batch-1", "问题", results, missing_binding
        )

    wrong_source = _evidence_snapshot()
    wrong_source["result_pairs"][0]["artifacts"][0]["source_artifact_ref"] = _source_ref(  # type: ignore[index]
        "combination-2"
    )
    with pytest.raises(ValueError, match="source_artifact_ref"):
        build_task_review_request("task-1", "batch-1", "问题", results, wrong_source)

    duplicate_source_results = _review_results()
    duplicate_source_results[1]["artifact_refs"] = [_source_ref("combination-1")]
    duplicate_source_snapshot = _evidence_snapshot()
    duplicate_source_snapshot["result_pairs"][1]["artifacts"][0][  # type: ignore[index]
        "source_artifact_ref"
    ] = _source_ref("combination-1")
    with pytest.raises(ValueError, match="跨 pair 复用 source_artifact_ref"):
        build_task_review_request(
            "task-1",
            "batch-1",
            "问题",
            duplicate_source_results,
            duplicate_source_snapshot,
        )

    duplicate_reader_snapshot = _evidence_snapshot()
    duplicate_reader_snapshot["result_pairs"][1]["artifacts"][0][  # type: ignore[index]
        "artifact_id"
    ] = duplicate_reader_snapshot["result_pairs"][0]["artifacts"][0][  # type: ignore[index]
        "artifact_id"
    ]
    with pytest.raises(ValueError, match="跨 pair 复用 Evidence artifact_id"):
        build_task_review_request(
            "task-1", "batch-1", "问题", results, duplicate_reader_snapshot
        )

    same_id_snapshot = _evidence_snapshot()
    same_id_snapshot["result_pairs"][0]["artifacts"][0][  # type: ignore[index]
        "artifact_id"
    ] = same_id_snapshot["result_pairs"][0]["artifacts"][0][  # type: ignore[index]
        "source_artifact_ref"
    ]
    with pytest.raises(ValueError, match="不能相同"):
        build_task_review_request("task-1", "batch-1", "问题", results, same_id_snapshot)


def test_investigation_response_schema_rejects_disabled_judgment_and_extra_fields() -> None:
    """调查 schema 只接受 supported/refuted 和合同规定字段。"""

    valid = {
        "self_judgment": "supported",
        "narrative": "完整回顾性计算过程。",
        "job_refs": [],
        "operation_refs": [],
        "artifact_refs": [],
        "observations": ["观测"],
        "limitations": [],
    }
    Draft202012Validator.check_schema(investigation_response_json_schema())
    Draft202012Validator(investigation_response_json_schema()).validate(valid)
    parsed_response = validate_investigation_response(valid)
    assert isinstance(parsed_response, InvestigationResponse)
    assert parsed_response.self_judgment == "supported"
    empty_optional_arrays = {
        **valid,
        "job_refs": [],
        "operation_refs": [],
        "artifact_refs": [],
        "observations": ["观测"],
        "limitations": [],
    }
    assert validate_investigation_response(empty_optional_arrays).job_refs == ()

    for field_name in (
        "job_refs",
        "operation_refs",
        "artifact_refs",
        "observations",
        "limitations",
    ):
        duplicate_value = "观测" if field_name in {"observations", "limitations"} else "ref-1"
        duplicate = {**valid, field_name: [duplicate_value, duplicate_value]}
        with pytest.raises(Exception):
            Draft202012Validator(investigation_response_json_schema()).validate(duplicate)
        with pytest.raises(ValidationError):
            validate_investigation_response(duplicate)
    with pytest.raises(Exception):
        Draft202012Validator(investigation_response_json_schema()).validate(
            {**valid, "job_refs": [""]}
        )
    with pytest.raises(ValidationError):
        validate_investigation_response({**valid, "job_refs": [""]})

    invalid_judgment = {**valid, "self_judgment": "inconclusive"}
    with pytest.raises(Exception):
        Draft202012Validator(investigation_response_json_schema()).validate(invalid_judgment)
    with pytest.raises(ValidationError):
        validate_investigation_response({**valid, "extra": True})
    with pytest.raises(ValidationError, match="observations"):
        validate_investigation_response({**valid, "observations": []})


def test_review_schema_enforces_pass_fail_branches_and_answerability_outputs() -> None:
    """Review schema 和 Pydantic 合同同时执行数量及回答/说明互斥分支。"""

    findings = [
        {
            "hypothesis_execution_input_record_id": f"combination-{index}",
            "validity": "pass",
            "hypothesis_judgment": "supported",
        }
        for index in range(1, 5)
    ] + [
        {
            "hypothesis_execution_input_record_id": "combination-5",
            "validity": "fail",
            "repair_advice": "补充关键观测。",
        }
    ]
    valid = {
        "findings": findings,
        "can_answer_original_question": False,
    }
    Draft202012Validator.check_schema(review_response_json_schema())
    Draft202012Validator(review_response_json_schema()).validate(valid)
    parsed_review = validate_task_review_response(valid)
    assert isinstance(parsed_review, TaskReviewResponse)
    assert isinstance(parsed_review.findings[0], PassReviewFinding)
    assert isinstance(parsed_review.findings[-1], FailReviewFinding)
    wire_fail_branch = {
        **valid,
        "final_answer": None,
        "answerability_explanation": None,
    }
    parsed_wire_fail = validate_task_review_response(wire_fail_branch)
    assert parsed_wire_fail.final_answer is None
    assert parsed_wire_fail.answerability_explanation is None

    author_three = {
        "findings": findings[:3],
        "can_answer_original_question": True,
        "final_answer": "三条作者假设足以回答原科学问题。",
    }
    Draft202012Validator(review_response_json_schema()).validate(author_three)
    assert len(validate_task_review_response(author_three).findings) == 3

    with pytest.raises(Exception):
        Draft202012Validator(review_response_json_schema()).validate(
            {**valid, "can_answer_original_question": False, "answerability_explanation": None}
        )
    with pytest.raises(Exception):
        Draft202012Validator(review_response_json_schema()).validate(
            {**valid, "can_answer_original_question": True}
        )
    with pytest.raises(Exception):
        Draft202012Validator(review_response_json_schema()).validate(
            {
                "findings": findings,
                "can_answer_original_question": True,
            }
        )
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {
                "findings": findings,
                "can_answer_original_question": True,
                "final_answer": "五项结果不足以回答原科学问题。",
            }
        )
    with pytest.raises(ValidationError, match="全部为 pass"):
        validate_task_review_response(
            {
                "findings": findings,
                "can_answer_original_question": True,
                "final_answer": "五项结果不足以回答原科学问题。",
            }
        )
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {
                **valid,
                "answerability_explanation": "任一 fail 时不允许回答能力说明。",
            }
        )
    all_refuted = {
        "findings": [
            {
                "hypothesis_execution_input_record_id": f"combination-{index}",
                "validity": "pass",
                "hypothesis_judgment": "refuted",
            }
            for index in range(1, 6)
        ],
        "can_answer_original_question": False,
    }
    Draft202012Validator(review_response_json_schema()).validate(all_refuted)
    assert validate_task_review_response(all_refuted).answerability_explanation is None
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {
                **all_refuted,
                "answerability_explanation": "全部 refuted 时不允许说明文本。",
            }
        )
    special_explanation = {
        "findings": [
            {
                "hypothesis_execution_input_record_id": "combination-1",
                "validity": "pass",
                "hypothesis_judgment": "supported",
            },
            *[
                {
                    "hypothesis_execution_input_record_id": f"combination-{index}",
                    "validity": "pass",
                    "hypothesis_judgment": "refuted",
                }
                for index in range(2, 6)
            ],
        ],
        "can_answer_original_question": False,
        "answerability_explanation": "有效性全部通过但仍不足以回答原问题。",
    }
    Draft202012Validator(review_response_json_schema()).validate(special_explanation)
    assert validate_task_review_response(special_explanation).answerability_explanation
    valid_answer = {
        "findings": [
            {
                "hypothesis_execution_input_record_id": f"combination-{index}",
                "validity": "pass",
                "hypothesis_judgment": "supported",
            }
            for index in range(1, 6)
        ],
        "can_answer_original_question": True,
        "final_answer": "五项结果共同支持对原科学问题的回答。",
    }
    Draft202012Validator(review_response_json_schema()).validate(valid_answer)
    assert validate_task_review_response(valid_answer).final_answer == (
        "五项结果共同支持对原科学问题的回答。"
    )
    parsed_wire_answer = validate_task_review_response(
        {**valid_answer, "answerability_explanation": None}
    )
    assert parsed_wire_answer.final_answer == valid_answer["final_answer"]
    assert parsed_wire_answer.answerability_explanation is None
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {
                **special_explanation,
                "final_answer": None,
                "answerability_explanation": None,
            }
        )
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {
                **valid,
                "final_answer": "不应存在的回答。",
                "answerability_explanation": None,
            }
        )
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {**valid_answer, "answerability_explanation": "不应同时携带说明。"}
        )
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {**valid, "final_answer": "不应与 explanation 同时出现。"}
        )
    with pytest.raises(ValidationError):
        validate_task_review_response(
            {
                **valid,
                "findings": [
                    {
                        "hypothesis_execution_input_record_id": "combination-1",
                        "validity": "pass",
                        "repair_advice": "错",
                    }
                ]
                + findings[1:],
            }
        )


def test_review_final_answer_allows_bound_public_source_format_but_rejects_private_values() -> None:
    """最终回答可保留证据中已有公开来源格式，但不能携带本机路径或凭据。"""

    findings = [
        {
            "hypothesis_execution_input_record_id": f"combination-{index}",
            "validity": "pass",
            "hypothesis_judgment": "supported",
        }
        for index in range(1, 6)
    ]
    base = {
        "findings": findings,
        "can_answer_original_question": True,
        "final_answer": "结论见 https://doi.org/10.1000/example。",
    }
    assert validate_task_review_response(base).final_answer == base["final_answer"]
    for unsafe in ("结果见 /mnt/private/result.json", "凭据 sk-test-secret-value"):
        with pytest.raises(ValidationError):
            validate_task_review_response({**base, "final_answer": unsafe})



def test_public_source_uri_classifier_rejects_private_hosts_and_credentials() -> None:
    """公开来源 URI 只允许无凭据的 HTTP(S) 地址，不允许本机或私网目标。"""

    assert contracts._is_public_source_uri("https://doi.org/10.1000/example") is True
    assert contracts._is_public_source_uri("http://8.8.8.8/reference") is True
    for unsafe in (
        "ftp://example.com/file",
        "https:///missing-host",
        "https://user:password@example.com/file",
        "https://localhost/result",
        "https://service.local/result",
        "https://service.internal/result",
        "https://192.168.1.10/result",
        "https://127.0.0.1/result",
        "https://169.254.1.10/result",
        "https://[::1]/result",
    ):
        assert contracts._is_public_source_uri(unsafe) is False


def test_review_answerability_requires_supported_finding_and_explanation_only() -> None:
    """可回答终局必须有 supported；不可回答的全 pass 只能保存 explanation。"""

    refuted_findings = [
        {
            "hypothesis_execution_input_record_id": f"combination-{index}",
            "validity": "pass",
            "hypothesis_judgment": "refuted",
        }
        for index in range(1, 6)
    ]
    with pytest.raises(ValidationError, match="至少一项判断必须为 supported"):
        validate_task_review_response(
            {
                "findings": refuted_findings,
                "can_answer_original_question": True,
                "final_answer": "所有结果都 refuted，不能形成支持性回答。",
            }
        )

    supported_findings = [
        {
            "hypothesis_execution_input_record_id": f"combination-{index}",
            "validity": "pass",
            "hypothesis_judgment": "supported",
        }
        for index in range(1, 6)
    ]
    with pytest.raises(ValidationError, match="不能携带 final_answer"):
        validate_task_review_response(
            {
                "findings": supported_findings,
                "can_answer_original_question": False,
                "answerability_explanation": "证据支持但仍不足以回答原问题。",
                "final_answer": "不应存在的回答。",
            }
        )



def test_prompt_evidence_normalizers_reject_unregistered_or_malformed_fields() -> None:
    """Evidence artifact/pair 规范化器拒绝未知字段、缺字段和重复命名空间值。"""

    import copy

    pair = copy.deepcopy(_evidence_snapshot()["result_pairs"][0])
    artifact = copy.deepcopy(pair["artifacts"][0])
    assert prompts._normalize_evidence_artifact(artifact, label="artifact") == artifact

    unknown_artifact = {**artifact, "extra": True}
    with pytest.raises(ValueError, match="未注册字段"):
        prompts._normalize_evidence_artifact(unknown_artifact, label="artifact")
    for field, value, message in (
        ("artifact_id", "", "artifact_id"),
        ("source_artifact_ref", "", "source_artifact_ref"),
        ("size", True, "size"),
        ("sha256", "bad", "sha256"),
        ("media_type", "", "media_type"),
    ):
        malformed = {**artifact, field: value}
        with pytest.raises((TypeError, ValueError), match=message):
            prompts._normalize_evidence_artifact(malformed, label="artifact")

    assert prompts._normalize_evidence_pair(pair, index=0)[0][
        "hypothesis_execution_input_record_id"
    ] == pair[
        "hypothesis_execution_input_record_id"
    ]
    with pytest.raises(ValueError, match="未注册字段"):
        prompts._normalize_evidence_pair({**pair, "extra": True}, index=0)
    with pytest.raises(ValueError, match="缺少 hypothesis_execution_input_record_id"):
        prompts._normalize_evidence_pair(
            {
                key: value
                for key, value in pair.items()
                if key != "hypothesis_execution_input_record_id"
            },
            index=0,
        )
    with pytest.raises(ValueError, match="缺少 result_version"):
        prompts._normalize_evidence_pair({key: value for key, value in pair.items() if key != "result_version"}, index=0)
    with pytest.raises(ValueError, match="缺少 artifacts"):
        prompts._normalize_evidence_pair({key: value for key, value in pair.items() if key != "artifacts"}, index=0)
    with pytest.raises(TypeError, match="object 数组"):
        prompts._normalize_evidence_pair({**pair, "artifacts": {}}, index=0)
    with pytest.raises(ValueError, match="不能重复 artifact_id"):
        prompts._normalize_evidence_pair({**pair, "artifacts": [artifact, artifact]}, index=0)
    duplicate_source = {**artifact, "artifact_id": "art_different_reader_id"}
    with pytest.raises(ValueError, match="不能重复 source_artifact_ref"):
        prompts._normalize_evidence_pair(
            {**pair, "artifacts": [artifact, duplicate_source]},
            index=0,
        )



def test_review_answerability_post_validator_rejects_missing_or_conflicting_text() -> None:
    """直接检查 after validator 的回答文本互斥分支，防止绕过 schema 形成坏事实。"""

    supported = tuple(
        PassReviewFinding(
            hypothesis_execution_input_record_id=f"combination-{index}",
            validity="pass",
            hypothesis_judgment="supported",
        )
        for index in range(1, 6)
    )
    refuted = tuple(
        PassReviewFinding(
            hypothesis_execution_input_record_id=f"combination-{index}",
            validity="pass",
            hypothesis_judgment="refuted",
        )
        for index in range(1, 6)
    )
    failed = (
        FailReviewFinding(
            hypothesis_execution_input_record_id="combination-1",
            validity="fail",
            repair_advice="补充证据。",
        ),
        *supported[1:],
    )

    invalid_cases = (
        (supported, True, None, None, "必须携带 final_answer"),
        (supported, True, "回答", "不应存在", "不能携带 explanation"),
        (refuted, True, "错误回答", None, "至少一项判断必须为 supported"),
        (supported, False, None, None, "必须携带 explanation"),
        (supported, False, "回答", "说明", "不能携带 final_answer"),
        (failed, False, "错误回答", None, "不能携带 final_answer 或 answerability_explanation"),
    )
    for findings, can_answer, final_answer, explanation, message in invalid_cases:
        response = TaskReviewResponse.model_construct(
            findings=findings,
            can_answer_original_question=can_answer,
            final_answer=final_answer,
            answerability_explanation=explanation,
        )
        with pytest.raises(ValueError, match=message):
            response.validate_answerability_contract()



def test_contract_json_safety_checker_covers_uri_number_key_and_type_failures() -> None:
    """结构化合同检查器拒绝不允许的 URI、非有限数字、非法键和非 JSON 类型。"""

    with pytest.raises(ValueError, match="不能包含"):
        contracts._assert_no_path_or_secret("https://example.com/source")
    contracts._assert_no_path_or_secret(
        {"source": "https://example.com/source"},
        allow_public_source_uris=True,
    )
    contracts._assert_no_path_or_secret(1.0)
    with pytest.raises(ValueError, match="NaN"):
        contracts._assert_no_path_or_secret(float("nan"))
    with pytest.raises(ValueError, match="object key"):
        contracts._assert_no_path_or_secret({1: "value"})
    with pytest.raises(TypeError, match="可序列化"):
        contracts._assert_no_path_or_secret(object())
