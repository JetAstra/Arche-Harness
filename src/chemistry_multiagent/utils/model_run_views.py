"""构造模型阶段的只读展示视图。

这里的函数只接收已经由 Controller 或 Working Memory 整理好的 JSON object，
然后复制正文并按正式业务顺序组织成研究员可读的视图。模块不读取文件、不写磁盘、
不调用模型，也不根据数组位置猜测 Hypothesis 与 combination 的关系；输入缺少
正式身份或阶段事实时直接抛出 ``ModelRunViewError``。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


MODEL_RUN_VIEW_SCHEMA_VERSION = 1


class ModelRunViewError(ValueError):
    """模型阶段视图的输入缺少事实或违反固定结构时抛出的错误。"""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    """确认输入是 JSON object；调用方拿到的是副本而不是原对象。"""

    if not isinstance(value, Mapping):
        raise ModelRunViewError(f"{field} 必须是 object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    """确认输入是 JSON array。"""

    if not isinstance(value, list):
        raise ModelRunViewError(f"{field} 必须是 array")
    return value


def _require_text(value: Any, field: str, *, allow_none: bool = False) -> str | None:
    """确认引用字段是非空字符串，但不去掉正文两端的空格。"""

    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ModelRunViewError(f"{field} 必须是非空字符串")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    """确认索引和版本是正整数，不把布尔值当作整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelRunViewError(f"{field} 必须是正整数")
    return value


def _validate_version_chain(
    rounds: Sequence[Mapping[str, Any]],
    *,
    field: str,
    allow_terminal_none: bool = False,
) -> None:
    """核对已经出现的 workflow version before/after 是否逐轮连续。"""

    presence = [
        (
            "workflow_version_before" in item,
            "workflow_version_after" in item,
        )
        for item in rounds
    ]
    if not any(any(flags) for flags in presence):
        return
    if any(flags != (True, True) for flags in presence):
        raise ModelRunViewError(
            f"{field} 的 workflow_version_before/after 必须同时提供，不能猜测缺失字段"
        )
    previous_after: int | None = None
    for index, item in enumerate(rounds):
        before = item["workflow_version_before"]
        after = item["workflow_version_after"]
        _require_positive_int(before, f"{field}[{index}].workflow_version_before")
        if after is not None:
            _require_positive_int(after, f"{field}[{index}].workflow_version_after")
            if after != before + 1:
                raise ModelRunViewError(
                    f"{field}[{index}] 的 workflow version 没有连续递增"
                )
            if previous_after is not None and before != previous_after:
                raise ModelRunViewError(
                    f"{field} 的 workflow version 轮次链不连续"
                )
            previous_after = after
        elif not allow_terminal_none or index != len(rounds) - 1:
            raise ModelRunViewError(
                f"{field} 的空 workflow_version_after 只能出现在最后一轮"
            )
        elif previous_after is not None and before != previous_after:
            raise ModelRunViewError(f"{field} 的 workflow version 轮次链不连续")


def _copy_source_file_list(value: Any, field: str) -> list[str]:
    """复制阶段来源文件列表；这里只复制引用，不检查或打开路径。"""

    files = _require_list(value, field)
    for index, item in enumerate(files):
        _require_text(item, f"{field}[{index}]")
    return deepcopy(files)


def _copy_source_file_map(
    value: Any,
    hypothesis_ids: Sequence[str],
    field: str,
) -> dict[str, dict[str, Any]]:
    """复制每个 Hypothesis 的来源映射，并确认三个阶段引用都在。"""

    mapping = _require_mapping(value, field)
    expected_ids = set(hypothesis_ids)
    if set(mapping) != expected_ids:
        raise ModelRunViewError(
            f"{field} 必须逐个覆盖 ranked hypothesis_id，"
            f"缺少={sorted(expected_ids - set(mapping))}，"
            f"多出={sorted(set(mapping) - expected_ids)}"
        )
    copied: dict[str, dict[str, Any]] = {}
    for hypothesis_id in hypothesis_ids:
        item = _require_mapping(mapping[hypothesis_id], f"{field}[{hypothesis_id}]")
        required = {"generation", "planning", "router"}
        if set(item) != required:
            raise ModelRunViewError(
                f"{field}[{hypothesis_id}] 必须恰含 generation/planning/router"
            )
        for stage in required:
            _require_text(
                item[stage],
                f"{field}[{hypothesis_id}].{stage}",
                allow_none=True,
            )
        copied[hypothesis_id] = deepcopy(dict(item))
    return copied


def _read_batch_parts(
    batch: Any,
) -> tuple[
    int,
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """读取当前批次的 Query、候选、排名和选择事实，并按 ID 建索引。"""

    payload = _require_mapping(batch, "batch")
    batch_index = _require_positive_int(payload.get("batch_index"), "batch_index")
    phase = _require_mapping(payload.get("hypothesis_phase"), "hypothesis_phase")
    queries_raw = _require_list(phase.get("queries"), "hypothesis_phase.queries")
    if not queries_raw:
        raise ModelRunViewError("hypothesis_phase.queries 不能为空")
    queries: list[str] = []
    for index, query in enumerate(queries_raw):
        queries.append(_require_text(query, f"hypothesis_phase.queries[{index}]") or "")

    groups_raw = _require_list(
        phase.get("hypotheses_by_query"),
        "hypothesis_phase.hypotheses_by_query",
    )
    if len(groups_raw) != len(queries):
        raise ModelRunViewError(
            "hypothesis_phase.hypotheses_by_query 必须与 queries 逐项对应"
        )
    generated: list[dict[str, Any]] = []
    generated_by_id: dict[str, dict[str, Any]] = {}
    for group_index, group_raw in enumerate(groups_raw):
        group = _require_mapping(
            group_raw,
            f"hypothesis_phase.hypotheses_by_query[{group_index}]",
        )
        if group.get("query") != queries[group_index]:
            raise ModelRunViewError(
                "hypotheses_by_query 的 query 必须逐值等于 queries"
            )
        hypotheses = _require_list(
            group.get("hypotheses"),
            f"hypotheses_by_query[{group_index}].hypotheses",
        )
        if not hypotheses:
            raise ModelRunViewError(
                f"hypotheses_by_query[{group_index}].hypotheses 不能为空"
            )
        for hypothesis_index, hypothesis_raw in enumerate(hypotheses):
            hypothesis = _require_mapping(
                hypothesis_raw,
                f"hypotheses_by_query[{group_index}].hypotheses[{hypothesis_index}]",
            )
            hypothesis_id = _require_text(
                hypothesis.get("hypothesis_id"),
                f"hypotheses_by_query[{group_index}].hypotheses[{hypothesis_index}].hypothesis_id",
            )
            assert hypothesis_id is not None
            if hypothesis_id in generated_by_id:
                raise ModelRunViewError(f"hypothesis_id 重复：{hypothesis_id}")
            copied = deepcopy(dict(hypothesis))
            generated.append(copied)
            generated_by_id[hypothesis_id] = copied

    ranked_raw = _require_list(
        phase.get("ranked_strategies"),
        "hypothesis_phase.ranked_strategies",
    )
    if not ranked_raw:
        raise ModelRunViewError("hypothesis_phase.ranked_strategies 不能为空")
    ranked: list[dict[str, Any]] = []
    ranked_by_id: dict[str, dict[str, Any]] = {}
    for index, hypothesis_raw in enumerate(ranked_raw):
        hypothesis = _require_mapping(
            hypothesis_raw,
            f"hypothesis_phase.ranked_strategies[{index}]",
        )
        hypothesis_id = _require_text(
            hypothesis.get("hypothesis_id"),
            f"hypothesis_phase.ranked_strategies[{index}].hypothesis_id",
        )
        assert hypothesis_id is not None
        if hypothesis_id in ranked_by_id:
            raise ModelRunViewError(f"ranked hypothesis_id 重复：{hypothesis_id}")
        copied = deepcopy(dict(hypothesis))
        ranked.append(copied)
        ranked_by_id[hypothesis_id] = copied

    candidates_raw = _require_list(payload.get("candidates"), "candidates")
    if len(candidates_raw) != len(ranked):
        raise ModelRunViewError("candidates 必须逐项覆盖 ranked_strategies")
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for index, candidate_raw in enumerate(candidates_raw):
        candidate = _require_mapping(candidate_raw, f"candidates[{index}]")
        hypothesis_id = _require_text(
            candidate.get("hypothesis_id"),
            f"candidates[{index}].hypothesis_id",
        )
        assert hypothesis_id is not None
        if hypothesis_id in candidates_by_id:
            raise ModelRunViewError(f"candidate hypothesis_id 重复：{hypothesis_id}")
        if hypothesis_id not in ranked_by_id:
            raise ModelRunViewError(
                f"candidate 引用了不存在的 ranked hypothesis：{hypothesis_id}"
            )
        status = candidate.get("selection_status")
        if status not in {"selected", "not_selected"}:
            raise ModelRunViewError(f"candidates[{index}].selection_status 非法")
        combination_index = candidate.get("combination_index")
        combination_id = candidate.get("combination_id")
        if status == "selected":
            _require_positive_int(combination_index, f"candidates[{index}].combination_index")
            _require_text(combination_id, f"candidates[{index}].combination_id")
        elif combination_index is not None or combination_id is not None:
            raise ModelRunViewError("not_selected candidate 不能携带 combination 身份")
        candidates_by_id[hypothesis_id] = deepcopy(dict(candidate))
    if set(candidates_by_id) != set(ranked_by_id):
        raise ModelRunViewError("candidates 与 ranked_strategies 的 hypothesis_id 集合不一致")
    selected_indexes = [
        item["combination_index"]
        for item in candidates_by_id.values()
        if item["selection_status"] == "selected"
    ]
    if len(selected_indexes) != len(set(selected_indexes)):
        raise ModelRunViewError("selected candidate 的 combination_index 不能重复")
    return batch_index, queries, generated, ranked, candidates_by_id


def construct_hypothesis_summary(
    batch: Mapping[str, Any],
    *,
    source_files: Sequence[str] | None = None,
    merge_decisions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造一批 Hypothesis 的确定性摘要。

    ``batch`` 必须包含当前正式的 ``hypothesis_phase`` 和 ``candidates``；
    ``source_files`` 是已经生成的阶段文件相对引用，不会被本函数打开。生成候选、
    合并判断和排名正文均只做深拷贝，selected 列表按正式排名中的 Hypothesis ID
    关联候选，不改写任何科学正文。来源或阶段事实缺失时抛出
    ``ModelRunViewError``。
    """

    batch_index, queries, generated, ranked, candidates_by_id = _read_batch_parts(batch)
    if source_files is None:
        source_files = batch.get("source_files")
    copied_source_files = _copy_source_file_list(source_files, "source_files")
    if merge_decisions is None:
        merge_decisions = batch.get("merge_decisions", [])
    merges = _require_list(merge_decisions, "merge_decisions")
    if any(not isinstance(item, Mapping) for item in merges):
        raise ModelRunViewError("merge_decisions 必须是 object 数组")
    selected = [
        deepcopy(hypothesis)
        for hypothesis in ranked
        if candidates_by_id[hypothesis["hypothesis_id"]]["selection_status"]
        == "selected"
    ]
    return {
        "schema_version": MODEL_RUN_VIEW_SCHEMA_VERSION,
        "batch_index": batch_index,
        "queries": deepcopy(queries),
        "generated_hypotheses": deepcopy(generated),
        "merge_decisions": deepcopy(list(merges)),
        "ranked_hypotheses": deepcopy(ranked),
        "selected_hypotheses": selected,
        "source_files": copied_source_files,
    }


def _copy_hypothesis_source_map(
    source_files: Any,
    hypothesis_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """为 outcomes 复用统一的来源映射校验。"""

    return _copy_source_file_map(source_files, hypothesis_ids, "source_files")


def construct_hypothesis_outcomes(
    batch: Mapping[str, Any],
    combination_facts: Sequence[Mapping[str, Any]],
    *,
    source_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造每个 Hypothesis 的后续结果映射。

    ``combination_facts`` 必须带有正式 ``hypothesis_id``，本函数先按 ID 建索引，
    再把 selected Hypothesis 关联到其 combination；绝不把组合数组位置当成假设
    身份。未入选候选固定返回 ``selection_status=not_selected`` 和
    ``terminal_state=not_executed``。能力缺口对象（无论来自首次 Planning 还是
    Revision）保持原样深拷贝；没有对应事实、终态或来源引用时明确失败。
    """

    batch_index, _queries, _generated, ranked, candidates_by_id = _read_batch_parts(batch)
    hypothesis_ids = [item["hypothesis_id"] for item in ranked]
    if source_files is None:
        source_files = batch.get("outcome_source_files")
    source_map = _copy_hypothesis_source_map(source_files, hypothesis_ids)
    facts = _require_list(combination_facts, "combination_facts")
    facts_by_id: dict[str, dict[str, Any]] = {}
    for index, fact_raw in enumerate(facts):
        fact = _require_mapping(fact_raw, f"combination_facts[{index}]")
        hypothesis_id = _require_text(
            fact.get("hypothesis_id"),
            f"combination_facts[{index}].hypothesis_id",
        )
        assert hypothesis_id is not None
        if hypothesis_id in facts_by_id:
            raise ModelRunViewError(f"combination fact 的 hypothesis_id 重复：{hypothesis_id}")
        if hypothesis_id not in candidates_by_id:
            raise ModelRunViewError(f"combination fact 引用了未知 hypothesis：{hypothesis_id}")
        _require_positive_int(
            fact.get("combination_index"),
            f"combination_facts[{index}].combination_index",
        )
        state = fact.get("terminal_state", fact.get("state"))
        _require_text(state, f"combination_facts[{index}].terminal_state")
        reason = _require_text(
            fact.get("terminal_reason"),
            f"combination_facts[{index}].terminal_reason",
            allow_none=True,
        )
        if reason is None:
            raise ModelRunViewError(
                f"combination_facts[{index}].terminal_reason 不能缺失"
            )
        facts_by_id[hypothesis_id] = deepcopy(dict(fact))

    result_hypotheses: list[dict[str, Any]] = []
    for rank, hypothesis in enumerate(ranked, start=1):
        hypothesis_id = hypothesis["hypothesis_id"]
        candidate = candidates_by_id[hypothesis_id]
        source = deepcopy(source_map[hypothesis_id])
        result = deepcopy(hypothesis)
        if candidate["selection_status"] == "not_selected":
            if hypothesis_id in facts_by_id:
                raise ModelRunViewError(
                    f"not_selected hypothesis 不得有 combination fact：{hypothesis_id}"
                )
            metadata = {
                "rank": deepcopy(candidate.get("rank", rank)),
                "selection_status": "not_selected",
                "combination_index": None,
                "terminal_state": "not_executed",
                "terminal_reason": None,
                "tool_capability_gap": None,
                "source_files": source,
            }
        else:
            fact = facts_by_id.get(hypothesis_id)
            if fact is None:
                raise ModelRunViewError(
                    f"selected hypothesis 缺少 combination fact：{hypothesis_id}"
                )
            if fact["combination_index"] != candidate["combination_index"]:
                raise ModelRunViewError(
                    f"{hypothesis_id} 的 candidate 与 combination_index 不一致"
                )
            state = fact.get("terminal_state", fact.get("state"))
            reason = fact["terminal_reason"]
            gap = fact.get("tool_capability_gap")
            if state == "tool_capability_gap":
                if not isinstance(gap, Mapping):
                    raise ModelRunViewError(
                        f"{hypothesis_id} 的 tool_capability_gap 缺少原始正文"
                    )
            elif gap is not None:
                raise ModelRunViewError(
                    f"非能力缺口终态不能携带 tool_capability_gap：{hypothesis_id}"
                )
            metadata = {
                "rank": deepcopy(candidate.get("rank", rank)),
                "selection_status": "selected",
                "combination_index": deepcopy(fact["combination_index"]),
                "terminal_state": deepcopy(state),
                "terminal_reason": deepcopy(reason),
                "tool_capability_gap": deepcopy(gap),
                "source_files": source,
            }
        for key, value in metadata.items():
            if key in result and result[key] != value:
                raise ModelRunViewError(
                    f"Hypothesis 正文占用了结果元数据字段 {key}：{hypothesis_id}"
                )
            result[key] = value
        result_hypotheses.append(result)
    return {
        "schema_version": MODEL_RUN_VIEW_SCHEMA_VERSION,
        "batch_index": batch_index,
        "hypotheses": result_hypotheses,
    }


def construct_workflow_version(record: Mapping[str, Any]) -> dict[str, Any]:
    """构造一份完整 Workflow 版本，并把 Revision patch 留在模型记录中。

    首次 Planning 记录必须提供 ``source_type=initial_planning``、版本 1 和
    ``workflow``。Revision 记录必须提供 ``source_type=revision_patch_applied``、
    ``workflow_version_before/after``、``normalized_patch`` 与 ``after_workflow``；
    输出只复制应用 patch 后的完整 Workflow，绝不把 patch 冒充成 Workflow 或写入
    版本正文。来源文件只作为字符串引用保存，不会被读取。
    """

    value = _require_mapping(record, "workflow version record")
    research_task_id = _require_text(
        value.get("research_task_id"), "research_task_id"
    )
    batch_index = _require_positive_int(value.get("batch_index"), "batch_index")
    combination_index = _require_positive_int(
        value.get("combination_index"),
        "combination_index",
    )
    source_type = _require_text(value.get("source_type"), "source_type")
    source_model_file = _require_text(
        value.get("source_model_file"),
        "source_model_file",
    )
    assert research_task_id is not None
    assert source_type is not None
    assert source_model_file is not None

    if source_type == "initial_planning":
        workflow_version = _require_positive_int(
            value.get("workflow_version"),
            "workflow_version",
        )
        if workflow_version != 1:
            raise ModelRunViewError("initial_planning 的 workflow_version 必须为 1")
        workflow = _require_mapping(value.get("workflow"), "workflow")
        previous_version = None
    elif source_type == "revision_patch_applied":
        before = _require_positive_int(
            value.get("workflow_version_before"),
            "workflow_version_before",
        )
        after = _require_positive_int(
            value.get("workflow_version_after"),
            "workflow_version_after",
        )
        if after != before + 1:
            raise ModelRunViewError("Revision 必须恰好提高一个 workflow version")
        if not isinstance(value.get("normalized_patch"), Mapping):
            raise ModelRunViewError("revision_patch_applied 缺少 normalized_patch")
        workflow = _require_mapping(value.get("after_workflow"), "after_workflow")
        workflow_version = after
        previous_version = value.get("previous_version")
        if previous_version is None:
            previous_version = f"version_{before:03d}.json"
        _require_text(previous_version, "previous_version")
    else:
        raise ModelRunViewError(
            "source_type 只能是 initial_planning 或 revision_patch_applied"
        )

    return {
        "schema_version": MODEL_RUN_VIEW_SCHEMA_VERSION,
        "research_task_id": research_task_id,
        "batch_index": batch_index,
        "combination_index": combination_index,
        "workflow_version": workflow_version,
        "source_type": source_type,
        "source_model_file": source_model_file,
        "previous_version": previous_version,
        "workflow": deepcopy(dict(workflow)),
    }


def _read_router_round(round_raw: Any, index: int) -> dict[str, Any]:
    """从 Router 原始轮次中提取固定的四个展示字段。"""

    value = _require_mapping(round_raw, f"router rounds[{index}]")
    round_index = _require_positive_int(value.get("round_index"), f"router rounds[{index}].round_index")
    source_file = _require_text(value.get("source_file"), f"router rounds[{index}].source_file")
    nested = value.get("router_decision")
    decision = value.get("decision")
    diagnosis: Mapping[str, Any] | None = None
    if nested is not None:
        nested_mapping = _require_mapping(nested, f"router rounds[{index}].router_decision")
        if decision is None:
            decision = nested_mapping.get("decision")
        diagnosis_value = nested_mapping.get("diagnosis")
        if diagnosis_value is not None:
            diagnosis = _require_mapping(
                diagnosis_value,
                f"router rounds[{index}].router_decision.diagnosis",
            )
    decision = _require_text(decision, f"router rounds[{index}].decision")
    assert decision is not None
    problem_type = value.get("problem_type")
    if problem_type is None and diagnosis is not None:
        problem_type = diagnosis.get("problem_type")
    if problem_type is not None:
        _require_text(problem_type, f"router rounds[{index}].problem_type")
    normalized = {
        "round_index": round_index,
        "decision": decision,
        "problem_type": deepcopy(problem_type),
        "source_file": source_file,
    }
    for field in ("workflow_version_before", "workflow_version_after"):
        if field in value:
            normalized[field] = deepcopy(value[field])
    return normalized


def construct_router_summary(
    history: Mapping[str, Any],
    *,
    allow_arbitrary_initial_round: bool = False,
) -> dict[str, Any]:
    """构造 Router 的有序轮次摘要。

    输入可以使用模型阶段 JSON 中的 ``router_decision``/``diagnosis`` 外壳，也可以
    直接提供 ``decision``/``problem_type``。轮次必须已经按程序实际发生顺序传入，
    默认从 1 连续；展示发布器在恢复时可以显式允许首轮使用任意正整数，但从该首轮
    开始仍必须连续。最终决定必须等于最后一轮，缺少来源文件或轮次事实时失败。
    """

    value = _require_mapping(history, "router history")
    if not isinstance(allow_arbitrary_initial_round, bool):
        raise ModelRunViewError("allow_arbitrary_initial_round 必须是布尔值")
    batch_index = _require_positive_int(value.get("batch_index"), "batch_index")
    combination_index = _require_positive_int(
        value.get("combination_index"),
        "combination_index",
    )
    raw_rounds = _require_list(value.get("rounds"), "router rounds")
    if not raw_rounds:
        raise ModelRunViewError("router rounds 不能为空")
    rounds = [_read_router_round(item, index) for index, item in enumerate(raw_rounds)]
    indexes = [item["round_index"] for item in rounds]
    first_index = indexes[0] if allow_arbitrary_initial_round else 1
    if indexes != list(range(first_index, first_index + len(rounds))):
        raise ModelRunViewError("router rounds 必须按首轮起的实际顺序连续排列")
    _validate_version_chain(rounds, field="router rounds")
    final_decision = value.get("final_decision", rounds[-1]["decision"])
    final_decision = _require_text(final_decision, "final_decision")
    assert final_decision is not None
    if final_decision != rounds[-1]["decision"]:
        raise ModelRunViewError("final_decision 必须等于最后一轮 Router decision")
    return {
        "schema_version": MODEL_RUN_VIEW_SCHEMA_VERSION,
        "batch_index": batch_index,
        "combination_index": combination_index,
        "rounds": deepcopy(rounds),
        "final_decision": final_decision,
    }


def _read_revision_round(round_raw: Any, index: int) -> dict[str, Any]:
    """从 Revision 原始记录或本函数生成的精简轮次提取版本结果。"""

    value = _require_mapping(round_raw, f"revision rounds[{index}]")
    round_index = _require_positive_int(
        value.get("round_index"),
        f"revision rounds[{index}].round_index",
    )
    source_file = _require_text(
        value.get("source_file"),
        f"revision rounds[{index}].source_file",
    )
    gap = value.get("tool_capability_gap")
    if gap is None:
        final_output = value.get("final_output")
        if isinstance(final_output, Mapping) and final_output.get("result_type") == "tool_capability_gap":
            gap = final_output
    if gap is not None:
        if not isinstance(gap, Mapping):
            raise ModelRunViewError(f"revision rounds[{index}].tool_capability_gap 必须是 object")
        before = _require_positive_int(
            value.get("workflow_version_before"),
            f"revision rounds[{index}].workflow_version_before",
        )
        return {
            "round_index": round_index,
            "source_file": source_file,
            "workflow_version_before": before,
            "workflow_version_after": None,
            "tool_capability_gap": deepcopy(dict(gap)),
        }
    before = _require_positive_int(
        value.get("workflow_version_before"),
        f"revision rounds[{index}].workflow_version_before",
    )
    after = _require_positive_int(
        value.get("workflow_version_after"),
        f"revision rounds[{index}].workflow_version_after",
    )
    if after != before + 1:
        raise ModelRunViewError(f"revision rounds[{index}] 的版本没有连续递增")
    projected_fields = {
        "round_index",
        "source_file",
        "workflow_version_before",
        "workflow_version_after",
        "tool_capability_gap",
    }
    is_projected_round = (
        set(value) == projected_fields
        and value.get("tool_capability_gap") is None
    )
    if (
        not isinstance(value.get("normalized_patch"), Mapping)
        and not is_projected_round
    ):
        raise ModelRunViewError(f"revision rounds[{index}] 缺少 normalized_patch")
    return {
        "round_index": round_index,
        "source_file": source_file,
        "workflow_version_before": before,
        "workflow_version_after": after,
        "tool_capability_gap": None,
    }


def construct_revision_summary(history: Mapping[str, Any]) -> dict[str, Any]:
    """构造 Revision 的有序摘要，并保留能力缺口原文。

    每一轮只保存来源、版本变化和能力缺口状态；实际 patch 仍留在 Revision 模型
    文件中，完整 Workflow 由 ``construct_workflow_version`` 单独构造。输入轮次必须
    按业务实际发生顺序传入，不能缺号或重复；能力缺口正文只深拷贝，不翻译、不压缩。
    """

    value = _require_mapping(history, "revision history")
    batch_index = _require_positive_int(value.get("batch_index"), "batch_index")
    combination_index = _require_positive_int(
        value.get("combination_index"),
        "combination_index",
    )
    raw_rounds = _require_list(value.get("rounds"), "revision rounds")
    if not raw_rounds:
        raise ModelRunViewError("revision rounds 不能为空")
    rounds = [_read_revision_round(item, index) for index, item in enumerate(raw_rounds)]
    indexes = [item["round_index"] for item in rounds]
    if indexes != list(range(1, len(rounds) + 1)):
        raise ModelRunViewError("revision rounds 必须按实际顺序从 1 连续排列")
    _validate_version_chain(
        rounds,
        field="revision rounds",
        allow_terminal_none=True,
    )
    final_status = "tool_capability_gap" if rounds[-1]["tool_capability_gap"] is not None else "revision_applied"
    if value.get("final_status") is not None:
        provided = _require_text(value["final_status"], "final_status")
        assert provided is not None
        if provided != final_status:
            raise ModelRunViewError("final_status 与最后一轮 Revision 结果不一致")
    return {
        "schema_version": MODEL_RUN_VIEW_SCHEMA_VERSION,
        "batch_index": batch_index,
        "combination_index": combination_index,
        "rounds": deepcopy(rounds),
        "final_status": final_status,
    }


__all__ = [
    "MODEL_RUN_VIEW_SCHEMA_VERSION",
    "ModelRunViewError",
    "construct_hypothesis_summary",
    "construct_hypothesis_outcomes",
    "construct_workflow_version",
    "construct_router_summary",
    "construct_revision_summary",
]
