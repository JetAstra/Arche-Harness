"""Investigation 只读 Working Memory 适配器的身份、顺序和隐私测试。"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from chemistry_multiagent.investigation.working_memory import (
    HypothesisRegenerationBatchInput,
    WorkingMemoryCombinationInput,
    _build_hypothesis_regeneration_batch_input,
    _project_history_batch,
    _read_combination_identity,
    build_hypothesis_regeneration_working_memory_input,
    read_investigation_working_memory_input,
)
from chemistry_multiagent.investigation.prompts import build_initial_investigation_request
from chemistry_multiagent.investigation.domain import (
    ApplyBatchValidation,
    BatchValidation,
    BeginNextBatch,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    RecordInvestigationResult,
    ValidationFinding,
    apply_investigation_event,
)
from chemistry_multiagent.memory.projections import project_hypothesis_regeneration_context
from chemistry_multiagent.memory.working_memory_validation import _validate_combination_row
from chemistry_multiagent.utils.model_prompts import build_model_messages


def _hypothesis(batch_id: str, index: int) -> dict[str, str]:
    """构造一项带唯一批次正文的结构化假设。"""

    return {
        "hypothesis_id": f"hyp-{batch_id}-{index}",
        "strategy_name": f"策略 {batch_id} {index}",
        "reasoning": f"批次 {batch_id} 的原始结构化理由 {index}",
    }


_COMBINATION_IDS = tuple(f"combination-{index}" for index in range(1, 6))
_MULTILINE_SCIENTIFIC_QUESTION = (
    "Assess the stereochemical outcome for SMILES "
    "C[C@H](O)/C=C\\C1=CC=CC=C1 at 30\N{DEGREE SIGN}C.\t"
    "Compare the computed result with the specified conditions.\n\n"
    "Experimentally reported enantiomeric excess: 88%."
)
_MULTILINE_RETRIEVAL_CONTEXT = {
    "literature_review": "第一段检索结论。\n\n第二段比较结果。\t保留制表符。",
    "answer": "检索回答第一行。\r\n检索回答第二行。",
    "source": "https://doi.org/10.1234/example",
}


def _make_initial_state(batch_id: str) -> InvestigationState:
    """建立五个空 Investigation 会话，供本文件独立构造终局状态。"""

    return InvestigationState(
        task_id="task-1",
        batch_id=batch_id,
        sessions=tuple(
            InvestigationSession(
                task_id="task-1",
                batch_id=batch_id,
                combination_id=combination_id,
                workspace_ref=f"workspace/{batch_id}/{combination_id}",
                scope_id=f"scope-{batch_id}-{combination_id}",
            )
            for combination_id in _COMBINATION_IDS
        ),
    )


def _mark_all_refuted(state: InvestigationState, version_prefix: str) -> InvestigationState:
    """登记五项真实结果和 Review，形成可重生的批次终局。"""

    for combination_id in _COMBINATION_IDS:
        result = InvestigationResult(
            task_id=state.task_id,
            batch_id=state.batch_id,
            combination_id=combination_id,
            result_version=f"{version_prefix}-{combination_id}",
            codex_thread_id=f"thread-{state.batch_id}-{combination_id}",
            scope_id=f"scope-{state.batch_id}-{combination_id}",
            turn_id=f"turn-{version_prefix}-{combination_id}",
            self_judgment="refuted",
            narrative=f"{combination_id} 的真实调查叙述。",
            job_refs=(f"job-{state.batch_id}-{combination_id}",),
            operation_refs=(f"operation-{state.batch_id}-{combination_id}",),
            artifact_refs=(f"artifact-{state.batch_id}-{combination_id}",),
            observations=(f"observation-{state.batch_id}-{combination_id}",),
            limitations=(f"limitation-{state.batch_id}-{combination_id}",),
        )
        state = apply_investigation_event(
            state,
            RecordInvestigationResult(result=result),
        )
    validation = BatchValidation(
        task_id=state.task_id,
        batch_id=state.batch_id,
        combination_ids=_COMBINATION_IDS,
        result_versions=tuple(
            state.session_for(combination_id).result_version  # type: ignore[arg-type]
            for combination_id in _COMBINATION_IDS
        ),
        findings=tuple(
            ValidationFinding(
                combination_id=combination_id,
                validity="pass",
                hypothesis_judgment="refuted",
            )
            for combination_id in _COMBINATION_IDS
        ),
        can_answer=False,
        review_thread_id="review-thread-private",
        review_turn_id="review-turn-private",
    )
    return apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )


def _begin_next_batch(state: InvestigationState, batch_id: str) -> InvestigationState:
    """关闭当前批次并独立建立下一批五组合会话。"""

    sessions = tuple(
        InvestigationSession(
            task_id=state.task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            workspace_ref=f"workspace/{batch_id}/{combination_id}",
            scope_id=f"scope-{batch_id}-{combination_id}",
        )
        for combination_id in _COMBINATION_IDS
    )
    return apply_investigation_event(
        state,
        BeginNextBatch(batch_id=batch_id, sessions=sessions),
    )


def _batch_payload(
    batch_id: str,
    *,
    batch_index: int,
    retrieval: dict[str, Any],
    selected_count: int = 5,
    question: str = "冻结科学问题",
) -> dict[str, Any]:
    """构造一个仅含冻结批次事实的 reader fixture。"""

    hypotheses = [_hypothesis(batch_id, index) for index in range(1, max(5, selected_count) + 1)]
    candidates = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        selected = index <= selected_count
        candidates.append(
            {
                "hypothesis_id": hypothesis["hypothesis_id"],
                "candidate_index": index,
                "rank": index,
                "selection_status": "selected" if selected else "not_selected",
                "combination_id": f"combination-{index}" if selected else None,
                "combination_index": index if selected else None,
            }
        )
    return {
        "batch_id": batch_id,
        "batch_index": batch_index,
        "question_snapshot": (
            {
                "scientific_question": question,
                "initial_query_angles": ["固定角度"],
                "initial_retrieval_phase": copy.deepcopy(retrieval),
            }
            if batch_index == 1
            else None
        ),
        "generation_source": {"kind": "initial" if batch_index == 1 else "working_memory_regeneration"},
        "hypothesis_phase": {"ranked_strategies": hypotheses},
        "candidates": candidates,
    }


class FakeWorkingMemoryReader:
    """记录调用并返回独立副本的 Working Memory-like reader。"""

    def __init__(self, batches: dict[str, dict[str, Any]], combinations: dict[str, list[dict[str, Any]]]):
        self.task_created = {
            "scientific_question": "冻结科学问题",
            "chemistry_context": {
                "charge": 1,
                "solvent": "acetone",
                "calculation_unit": {"scope": "只验证给定成键计算单元"},
            },
            "task_config": {"top_n_hypotheses": 5},
            "task_output_relative_dir": "outputs/runs/task-1",
            "workspace_ref": "workspace/private",
        }
        self.retrieval = {
            "event_id": "retrieval-event-private",
            "source_refs": ["retrieval/private.json"],
            "retrieval_result": {
                "keywords": ["frozen keyword"],
                "literature_review": "冻结检索结论",
                "chemistry_context": {"reaction_type": "addition"},
                "paper_search_status": {
                    "manifest_path": "/private/manifest.json",
                    "status": "completed",
                },
            },
        }
        self.batches = batches
        self.combinations = combinations
        self.batch_receipts: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[Any, ...]] = []

    def get_task_created(self, task_id: str) -> dict[str, Any]:
        self.calls.append(("task_created", task_id))
        return copy.deepcopy(self.task_created)

    def get_retrieval_completed(self, task_id: str) -> dict[str, Any]:
        self.calls.append(("retrieval_completed", task_id))
        return copy.deepcopy(self.retrieval)

    def get_batch_created(self, task_id: str, batch_id: str) -> dict[str, Any]:
        self.calls.append(("batch_created", task_id, batch_id))
        return copy.deepcopy(self.batches[batch_id])

    def get_batch_created_receipt(self, task_id: str, batch_id: str) -> dict[str, Any]:
        self.calls.append(("batch_created_receipt", task_id, batch_id))
        return copy.deepcopy(self.batch_receipts[batch_id])

    def list_combinations(self, task_id: str, batch_id: str) -> list[dict[str, Any]]:
        self.calls.append(("combinations", task_id, batch_id))
        return copy.deepcopy(self.combinations[batch_id])


def _make_reader(*, batch_ids: tuple[str, ...] = ("batch-1",)) -> FakeWorkingMemoryReader:
    """构造首批或多批 reader。"""

    retrieval = {
        "keywords": ["frozen keyword"],
        "literature_review": "冻结检索结论",
        "chemistry_context": {"reaction_type": "addition"},
        "paper_search_status": {
            "manifest_path": "/private/manifest.json",
            "status": "completed",
        },
    }
    batches = {
        batch_id: _batch_payload(
            batch_id,
            batch_index=index,
            retrieval=retrieval,
        )
        for index, batch_id in enumerate(batch_ids, start=1)
    }
    combinations = {
        batch_id: [
            {
                "task_id": "task-1",
                "batch_id": batch_id,
                "combination_id": f"combination-{index}",
                "combination_index": index,
                "hypothesis_id": f"hyp-{batch_id}-{index}",
                "workspace_ref": f"/private/{batch_id}/{index}",
                "scope_id": f"scope-{batch_id}-{index}",
                "codex_thread_id": f"thread-{batch_id}-{index}",
                "current_workflow": {"private": True},
                "runtime_checkpoint": {"private": True},
            }
            for index in (3, 1, 5, 2, 4)
        ]
        for batch_id in batch_ids
    }
    return FakeWorkingMemoryReader(batches, combinations)


def test_initial_input_reads_facts_once_sorts_and_hides_runtime_fields() -> None:
    """乱序组合按 combination_index 排序，且 reader 私有字段不进入输入。"""

    reader = _make_reader()
    result = read_investigation_working_memory_input(reader, "task-1", "batch-1")

    assert result.batch_index == 1
    assert result.chemistry_context["calculation_unit"]["scope"] == "只验证给定成键计算单元"
    assert [item.combination_index for item in result.combinations] == [1, 2, 3, 4, 5]
    assert [item.hypothesis_id for item in result.combinations] == [
        "hyp-batch-1-1",
        "hyp-batch-1-2",
        "hyp-batch-1-3",
        "hyp-batch-1-4",
        "hyp-batch-1-5",
    ]
    assert result.combinations[0].hypothesis == _hypothesis("batch-1", 1)
    encoded = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    for marker in ("workspace_ref", "scope_id", "codex_thread_id", "current_workflow", "runtime_checkpoint", "manifest_path", "/private/"):
        assert marker not in encoded
    assert [call[0] for call in reader.calls] == [
        "task_created",
        "retrieval_completed",
        "batch_created",
        "task_created",
        "combinations",
    ]


def test_initial_input_preserves_multiline_scientific_question_and_smiles() -> None:
    """公开科学正文允许换行、制表符和合法 SMILES，并保持冻结事实原文。"""

    reader = _make_reader()
    question = f"  {_MULTILINE_SCIENTIFIC_QUESTION}\n"
    reader.task_created["scientific_question"] = question
    reader.batches["batch-1"]["question_snapshot"][
        "scientific_question"
    ] = question

    result = read_investigation_working_memory_input(reader, "task-1", "batch-1")

    assert result.scientific_question == question


def test_initial_input_preserves_multiline_retrieval_context_through_initial_prompt() -> None:
    """Retrieval 正文经 Working Memory 进入 initial prompt 时保留排版空白。"""

    reader = _make_reader()
    retrieval_result = reader.retrieval["retrieval_result"]
    retrieval_result.update(_MULTILINE_RETRIEVAL_CONTEXT)
    reader.batches["batch-1"]["question_snapshot"]["initial_retrieval_phase"].update(
        _MULTILINE_RETRIEVAL_CONTEXT
    )

    working_memory = read_investigation_working_memory_input(reader, "task-1", "batch-1")
    request = build_initial_investigation_request(
        working_memory.task_id,
        working_memory.batch_id,
        working_memory.combinations[0].combination_id,
        working_memory.scientific_question,
        working_memory.combinations[0].hypothesis,
        explicit_conditions=working_memory.chemistry_context,
        frozen_context=working_memory.retrieval_context,
    )
    payload = json.loads(request["messages"][1]["content"])

    assert payload["frozen_context"]["literature_review"] == _MULTILINE_RETRIEVAL_CONTEXT[
        "literature_review"
    ]
    assert payload["frozen_context"]["answer"] == _MULTILINE_RETRIEVAL_CONTEXT["answer"]


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "正文\n包含 NUL\x00",
        "正文\n包含 ESC\x1b",
        "正文\n/mnt/private/input.xyz",
        "正文\nfile:///tmp/private/input.xyz",
        "正文\nhttp://127.0.0.1:9010/mcp",
        "说明\napi_key=synthetic-test-value",
        {"task_id": "task-1\ncontinued"},
    ],
)
def test_initial_input_rejects_unsafe_nested_retrieval_context(unsafe_value: Any) -> None:
    """Retrieval 任意嵌套正文仍拒绝危险控制字符、路径、私有 URI、凭据和身份漂移。"""

    reader = _make_reader()
    reader.retrieval["retrieval_result"]["answer"] = unsafe_value
    reader.batches["batch-1"]["question_snapshot"]["initial_retrieval_phase"][
        "answer"
    ] = unsafe_value

    with pytest.raises(ValueError, match="路径|URI|控制字符|凭据|不能包含"):
        read_investigation_working_memory_input(reader, "task-1", "batch-1")


@pytest.mark.parametrize(
    "unsafe_question",
    [
        "问题包含 NUL\x00 字符",
        "问题包含 ESC\x1b 字符",
        "问题 /mnt/private/input.xyz",
        "问题 file:///tmp/private/input.xyz",
        "问题 api_key=synthetic-test-value",
        "说明\napi_key=synthetic-test-value",
        "说明\tfile:///tmp/private/input.xyz",
    ],
)
def test_scientific_question_still_rejects_controls_paths_uris_and_credentials(
    unsafe_question: str,
) -> None:
    """公开科学正文的排版例外不应放宽危险控制字符、路径、URI 或凭据。"""

    reader = _make_reader()
    reader.task_created["scientific_question"] = unsafe_question
    reader.batches["batch-1"]["question_snapshot"]["scientific_question"] = unsafe_question

    with pytest.raises(ValueError, match="路径|URI|控制字符|凭据"):
        read_investigation_working_memory_input(reader, "task-1", "batch-1")


@pytest.mark.parametrize("mutation", ["unknown", "candidate_bool", "rank_bool"])
def test_candidates_require_exact_fields_and_strict_integer_types(mutation: str) -> None:
    """候选索引和排名不能用 bool 冒充整数，也不能携带未注册字段。"""

    reader = _make_reader()
    candidate = reader.batches["batch-1"]["candidates"][0]
    if mutation == "unknown":
        candidate["unregistered"] = True
    elif mutation == "candidate_bool":
        candidate["candidate_index"] = True
    else:
        candidate["rank"] = True
    with pytest.raises(ValueError, match="candidate|字段"):
        read_investigation_working_memory_input(reader, "task-1", "batch-1")


@pytest.mark.parametrize("selected_count", [4, 6])
def test_initial_input_requires_exactly_five_selected_combinations(selected_count: int) -> None:
    """少于或多于五个 selected combination 都必须停止恢复。"""

    reader = _make_reader()
    reader.batches["batch-1"] = _batch_payload(
        "batch-1",
        batch_index=1,
        retrieval=reader.retrieval["retrieval_result"],
        selected_count=selected_count,
    )
    reader.combinations["batch-1"] = reader.combinations["batch-1"][:selected_count]
    with pytest.raises(ValueError, match="五个|selected"):
        read_investigation_working_memory_input(reader, "task-1", "batch-1")


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate_index", "combination_index"),
        ("drifted_hypothesis", "身份漂移"),
        ("drifted_task", "task_id"),
        ("missing_reasoning", "正文"),
        ("drifted_question", "scientific_question"),
        ("drifted_retrieval", "Retrieval"),
    ],
)
def test_initial_input_fails_closed_on_identity_or_frozen_context_drift(
    mutation: str,
    message: str,
) -> None:
    """重复、漂移和缺正文不会返回部分假设。"""

    reader = _make_reader()
    if mutation == "duplicate_index":
        reader.combinations["batch-1"][3]["combination_index"] = 1
    elif mutation == "drifted_hypothesis":
        reader.combinations["batch-1"][0]["hypothesis_id"] = "other-hypothesis"
    elif mutation == "drifted_task":
        reader.combinations["batch-1"][0]["task_id"] = "other-task"
    elif mutation == "missing_reasoning":
        del reader.batches["batch-1"]["hypothesis_phase"]["ranked_strategies"][0]["reasoning"]
    elif mutation == "drifted_question":
        reader.batches["batch-1"]["question_snapshot"]["scientific_question"] = "另一个问题"
    elif mutation == "drifted_retrieval":
        reader.batches["batch-1"]["question_snapshot"]["initial_retrieval_phase"]["chemistry_context"] = {
            "reaction_type": "different"
        }
    with pytest.raises(ValueError, match=message):
        read_investigation_working_memory_input(reader, "task-1", "batch-1")


def test_regeneration_merges_real_history_by_batch_and_combination() -> None:
    """多批相同 combination_id 也按 batch_id 精确合并原始假设。"""

    state = _mark_all_refuted(_make_initial_state("batch-1"), "initial-1")
    state = _begin_next_batch(state, "batch-2")
    state = _mark_all_refuted(state, "initial-2")
    reader = _make_reader(batch_ids=("batch-1", "batch-2"))

    result = build_hypothesis_regeneration_working_memory_input(state, reader)

    assert result.batch_id == "batch-2"
    assert len(result.closed_batches) == 1
    closed = result.closed_batches[0]
    assert closed.batch_index == 1
    assert closed.results
    assert closed.validations
    assert closed.combinations[0].hypothesis["reasoning"] == "批次 batch-1 的原始结构化理由 1"
    assert result.current_batch.batch_index == 2
    assert result.current_batch.combinations[0].hypothesis["reasoning"] == "批次 batch-2 的原始结构化理由 1"
    assert result.to_prompt_payload()["closed_batches"][0]["combinations"][0]["hypothesis"]
    encoded = json.dumps(result.to_prompt_payload(), ensure_ascii=False)
    for marker in ("workspace_ref", "scope_id", "codex_thread_id", "review_thread_id", "review_turn_id", "workflow", "runtime_checkpoint", "/private/"):
        assert marker not in encoded
    assert [call for call in reader.calls if call[0] == "batch_created"] == [
        ("batch_created", "task-1", "batch-1"),
        ("batch_created", "task-1", "batch-2"),
    ]


def test_regeneration_missing_historical_mapping_fails_closed() -> None:
    """历史状态存在但 reader 缺一个组合正文时不得从 narrative 猜测。"""

    state = _mark_all_refuted(_make_initial_state("batch-1"), "initial")
    reader = _make_reader()
    del reader.combinations["batch-1"][0]
    with pytest.raises(ValueError, match="五项|组合记录"):
        build_hypothesis_regeneration_working_memory_input(state, reader)


def test_regeneration_provider_drops_raw_context_and_real_failure_event() -> None:
    """reader 中的失败事实和原始输出不能穿透公开输入或 provider 边界。

    失败事件作为已经读取的历史事实注入，不在测试中构造旧 schema-v10
    旧 schema-v10 写服务；该适配器只验证读取和公开输入投影。
    """

    reader = _make_reader()
    raw_values = {
        "raw_output": "CHEMISTRY_RAW_OUTPUT_SENTINEL",
        "raw_tool_output": "RETRIEVAL_RAW_TOOL_OUTPUT_SENTINEL",
        "stdout": "STDOUT_SENTINEL",
        "stderr": "STDERR_SENTINEL",
        "tool_output": "TOOL_OUTPUT_SENTINEL",
    }
    reader.task_created["chemistry_context"] = {
        "reaction_type": "addition",
        **raw_values,
    }
    reader.retrieval["retrieval_result"].update(raw_values)

    failure = {
        "task_id": "task-1",
        "reason_code": "controller_crashed",
        "error_type": "RuntimeError",
        "event_id": "failure-event-1",
    }

    # 将真实失败事件作为 reader 的事件包装元数据，确认适配器不会把它当作科学事实。
    reader.task_created["chemistry_context"]["investigation_task_failed"] = failure
    reader.task_created["investigation_task_failed"] = failure
    reader.retrieval["retrieval_result"]["investigation_task_failed"] = failure
    reader.retrieval["investigation_task_failed"] = failure
    reader.retrieval["event_type"] = "investigation_task_failed"

    state = _mark_all_refuted(_make_initial_state("batch-1"), "initial")
    working_memory = build_hypothesis_regeneration_working_memory_input(state, reader)
    context = project_hypothesis_regeneration_context(
        scientific_question=working_memory.scientific_question,
        initial_query_angles=["固定角度"],
        initial_retrieval_phase=working_memory.retrieval_context,
        chemistry_context=working_memory.chemistry_context,
    )
    messages = build_model_messages(
        "hypothesis_generation",
        {
            "scientific_context": context,
            "current_query": "固定角度",
            "prior_hypothesis_outcomes": [],
            "next_batch_index": 2,
            "generation_config": {
                "num_queries": 1,
                "num_hypotheses_per_query": 5,
                "top_n_hypotheses": 5,
            },
            "hypothesis_output_schema": {"type": "array"},
            "investigation_working_memory": working_memory.to_prompt_payload(),
        },
        stage="regeneration",
        memory_enabled=True,
    )
    encoded = json.dumps(messages, ensure_ascii=False)
    for sentinel in (*raw_values.values(), "investigation_task_failed", "controller_crashed"):
        assert sentinel not in encoded
    payload = json.loads(next(item["content"] for item in messages if item["role"] == "user"))
    assert "investigation_task_failed" not in payload["base_input"]
    assert payload["base_input"]["investigation_working_memory"]["chemistry_context"] == {
        "reaction_type": "addition"
    }


def test_inputs_are_frozen_and_do_not_mutate_reader() -> None:
    """适配器返回冻结模型和深拷贝，不会把调用方修改写回 reader fixture。"""

    reader = _make_reader()
    original = copy.deepcopy(reader.batches)
    result = read_investigation_working_memory_input(reader, "task-1", "batch-1")
    with pytest.raises((TypeError, ValueError)):
        result.batch_id = "other"  # type: ignore[misc]
    result.combinations[0].hypothesis["reasoning"] = "调用方修改"
    assert reader.batches == original


def test_selected_and_not_selected_candidate_identity_is_strict() -> None:
    """selected 必须绑定 ranked hypothesis，未选项不能偷带组合身份。"""

    reader = _make_reader()
    batch = _batch_payload(
        "batch-1",
        batch_index=1,
        retrieval=reader.retrieval["retrieval_result"],
        selected_count=5,
    )
    sixth = _hypothesis("batch-1", 6)
    batch["hypothesis_phase"]["ranked_strategies"].append(sixth)
    batch["candidates"].append(
        {
            "hypothesis_id": sixth["hypothesis_id"],
            "candidate_index": 6,
            "rank": 6,
            "selection_status": "not_selected",
            "combination_id": None,
            "combination_index": None,
        }
    )
    reader.batches["batch-1"] = batch

    reader.batches["batch-1"]["candidates"][5]["combination_id"] = "unexpected"
    with pytest.raises(ValueError, match="not_selected"):
        read_investigation_working_memory_input(reader, "task-1", "batch-1")

    for field, value, message in (
        ("combination_id", None, "缺少 combination_id"),
        ("hypothesis_id", "unknown-hypothesis", "逐项漂移"),
        ("combination_index", 0, "缺少合法 combination_index"),
    ):
        reader = _make_reader()
        candidate = reader.batches["batch-1"]["candidates"][0]
        candidate[field] = value
        with pytest.raises(ValueError, match=message):
            read_investigation_working_memory_input(reader, "task-1", "batch-1")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection_status", "not_selected", "selection_status"),
        ("candidate_index", 2, "candidate_index"),
        ("rank", 2, "rank"),
        ("hypothesis_id", "hyp-batch-1-2", "逐项漂移"),
    ],
)
def test_candidate_order_and_status_cannot_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    """候选顺序、rank、状态和 hypothesis identity 必须逐项一致。"""

    reader = _make_reader()
    reader.batches["batch-1"]["candidates"][0][field] = value
    with pytest.raises(ValueError, match=message):
        read_investigation_working_memory_input(reader, "task-1", "batch-1")



def _history_projection_fixture() -> dict[str, Any]:
    """构造只含公开历史字段的重生批次事实。"""

    return {
        "task_id": "task-1",
        "batch_id": "batch-1",
        "route": "regenerate_hypotheses",
        "final_answer": None,
        "answerability_explanation": None,
        "sessions": [{"combination_id": item} for item in _COMBINATION_IDS],
        "results": [{"combination_id": item} for item in _COMBINATION_IDS],
        "validations": [{"combination_id": item} for item in _COMBINATION_IDS],
    }


def test_history_batch_projection_rejects_shape_drift_and_private_fields() -> None:
    """重生历史必须是固定字段、数组和 object 项，不能混入私有身份。"""

    projected = _project_history_batch(_history_projection_fixture(), "history")
    assert projected["final_answer"] is None

    with_private = _history_projection_fixture()
    with_private["review_thread_id"] = "private-review-thread"
    with pytest.raises(ValueError, match="字段不完整"):
        _project_history_batch(with_private, "history")

    invalid_cases: list[tuple[Any, str]] = [(None, "object")]
    missing = _history_projection_fixture()
    del missing["route"]
    invalid_cases.append((missing, "字段不完整"))
    unknown = _history_projection_fixture()
    unknown["unexpected"] = True
    invalid_cases.append((unknown, "字段不完整"))
    for collection_name in ("sessions", "results", "validations"):
        wrong_collection = _history_projection_fixture()
        wrong_collection[collection_name] = {}
        invalid_cases.append((wrong_collection, collection_name))
        wrong_item = _history_projection_fixture()
        wrong_item[collection_name] = ["not-an-object"]
        invalid_cases.append((wrong_item, collection_name))

    for payload, message in invalid_cases:
        with pytest.raises((TypeError, ValueError), match=message):
            _project_history_batch(payload, "history")


@pytest.mark.parametrize(
    "mutation",
    ["result_narrative", "validation_advice", "final_answer"],
)
def test_history_batch_projection_keeps_response_text_single_line(mutation: str) -> None:
    """重生历史的结果、验证和终答正文继续拒绝多段排版空白。"""

    payload = _history_projection_fixture()
    if mutation == "result_narrative":
        payload["results"][0]["narrative"] = "第一行\n第二行"
    elif mutation == "validation_advice":
        payload["validations"][0]["repair_advice"] = "第一行\n第二行"
    else:
        payload["final_answer"] = "第一行\n第二行"

    with pytest.raises(ValueError, match="控制字符|不能包含"):
        _project_history_batch(payload, "history")


def test_regeneration_batch_projection_rejects_mapping_and_identity_drift() -> None:
    """历史批次必须能和 reader 加载的五个组合按任务身份精确联接。"""

    state = _mark_all_refuted(_make_initial_state("batch-1"), "initial")
    combinations = tuple(
        WorkingMemoryCombinationInput(
            task_id="task-1",
            batch_id="batch-1",
            combination_id=combination_id,
            combination_index=index,
            hypothesis_id=f"hyp-batch-1-{index}",
            hypothesis=_hypothesis("batch-1", index),
        )
        for index, combination_id in enumerate(_COMBINATION_IDS, start=1)
    )
    loaded = {"batch-1": (1, combinations)}
    history = _history_projection_fixture()
    result = _build_hypothesis_regeneration_batch_input(
        history,
        label="current_batch",
        state=state,
        loaded_batches=loaded,
    )
    assert isinstance(result, HypothesisRegenerationBatchInput)
    assert result.combinations == combinations

    missing_batch = dict(history, batch_id="missing")
    wrong_task = dict(history, task_id="other-task")
    missing_session = dict(
        history,
        sessions=[{"combination_id": _COMBINATION_IDS[0]}],
    )
    for payload, message in (
        (missing_batch, "reader 对应"),
        (wrong_task, "task_id"),
        (missing_session, "没有覆盖"),
    ):
        with pytest.raises(ValueError, match=message):
            _build_hypothesis_regeneration_batch_input(
                payload,
                label="current_batch",
                state=state,
                loaded_batches=loaded,
            )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("batch", "batch_id"),
        ("missing_id", "combination_id"),
        ("blank_id", "combination_id"),
        ("duplicate_id", "重复"),
        ("bool_index", "重复或非法"),
        ("text_index", "重复或非法"),
        ("duplicate_index", "重复或非法"),
        ("missing_hypothesis", "hypothesis_id"),
        ("blank_hypothesis", "hypothesis_id"),
    ],
)
def test_combination_identity_rejects_all_identity_and_index_drift(
    mutation: str,
    message: str,
) -> None:
    """恢复组合时同时校验任务、批次、编号、序号和 hypothesis 身份。"""

    combination = {
        "task_id": "task-1",
        "batch_id": "batch-1",
        "combination_id": "combination-1",
        "combination_index": 1,
        "hypothesis_id": "hyp-batch-1-1",
    }
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    if mutation == "batch":
        combination["batch_id"] = "other-batch"
    elif mutation == "missing_id":
        del combination["combination_id"]
    elif mutation == "blank_id":
        combination["combination_id"] = ""
    elif mutation == "duplicate_id":
        seen_ids.add("combination-1")
    elif mutation == "bool_index":
        combination["combination_index"] = True
    elif mutation == "text_index":
        combination["combination_index"] = "1"
    elif mutation == "duplicate_index":
        seen_indexes.add(1)
    elif mutation == "missing_hypothesis":
        del combination["hypothesis_id"]
    else:
        combination["hypothesis_id"] = ""

    with pytest.raises(ValueError, match=message):
        _read_combination_identity(
            combination,
            task_id="task-1",
            batch_id="batch-1",
            seen_ids=seen_ids,
            seen_indexes=seen_indexes,
        )

    valid = _read_combination_identity(
        {
            "task_id": "task-1",
            "batch_id": "batch-1",
            "combination_id": "combination-2",
            "combination_index": 2,
            "hypothesis_id": "hyp-batch-1-2",
        },
        task_id="task-1",
        batch_id="batch-1",
        seen_ids=set(),
        seen_indexes=set(),
    )
    assert valid == ("combination-2", 2, "hyp-batch-1-2")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("task", "任务或批次身份"),
        ("combination", "身份未与 session"),
        ("schema", "hypothesis 字段"),
        ("hypothesis", "hypothesis_id 身份漂移"),
    ],
)
def test_validate_combination_row_rejects_each_identity_and_schema_drift(
    mutation: str,
    message: str,
) -> None:
    """输入一条合法组合及一种漂移；动作逐项校验合同并拒绝损坏组合。

    副作用仅限于本测试传入的集合，返回值无；任务/组合身份、hypothesis schema 或
    hypothesis 身份漂移时必须抛出 ``ValueError``，不得登记不可信组合。
    """

    hypothesis = _hypothesis("batch-1", 1)
    item: dict[str, Any] = {
        "task_id": "task-1",
        "batch_id": "batch-1",
        "combination_id": "combination-1",
        "combination_index": 1,
        "hypothesis_id": hypothesis["hypothesis_id"],
        "hypothesis": hypothesis,
    }
    if mutation == "task":
        item["task_id"] = "other-task"
    elif mutation == "combination":
        item["combination_id"] = "other-combination"
    elif mutation == "schema":
        del item["hypothesis"]["reasoning"]
    else:
        item["hypothesis"]["hypothesis_id"] = "other-hypothesis"

    with pytest.raises(ValueError, match=message):
        _validate_combination_row(
            item,
            task_id="task-1",
            batch_id="batch-1",
            index=0,
            session_ids={"combination-1"},
            combination_ids=set(),
            hypothesis_ids=set(),
            label="combinations[0]",
        )
