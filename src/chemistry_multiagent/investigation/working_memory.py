"""Investigation 阶段的只读 Working Memory 输入适配器。

本模块把旧 Working Memory 中已经冻结的任务、Retrieval、批次和组合事实整理为
Investigation 可以读取的窄输入。它不写数据库，也不把 Investigation 的结果回写
旧 workflow 字段；任何身份漂移、缺失正文或不完整的五组合都会直接失败。
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from chemistry_multiagent.investigation.contracts import (
    _assert_no_path_or_secret,
    require_safe_public_scientific_context,
    require_safe_public_scientific_text,
)
from chemistry_multiagent.investigation.domain import InvestigationState
from chemistry_multiagent.investigation.regeneration_memory import (
    build_hypothesis_regeneration_memory,
)
from chemistry_multiagent.memory.contracts import (
    get_investigation_combination_count,
    validate_structured_hypothesis_claims,
)


__all__ = [
    "WorkingMemoryCombinationInput",
    "InvestigationWorkingMemoryInput",
    "HypothesisRegenerationBatchInput",
    "HypothesisRegenerationWorkingMemoryInput",
    "WorkingMemoryReader",
    "WorkingMemoryBatchReceiptReader",
    "read_investigation_working_memory_input",
    "build_hypothesis_regeneration_working_memory_input",
]


_PRIVATE_KEYS = frozenset(
    {
        "workspace",
        "workspace_ref",
        "scope_id",
        "thread_id",
        "codex_thread_id",
        "review_thread_id",
        "review_turn_id",
        "transcript",
        "workflow",
        "current_workflow",
        "runtime",
        "runtime_checkpoint",
        "checkpoint",
        "path",
        "paths",
        "manifest_path",
        "resume_state_path",
        "response_cache_dir",
        "source_ref",
        "source_refs",
        "event_id",
        "record_id",
        "api_key",
        "access_token",
        "secret",
        "password",
        "credential",
        "credentials",
        "raw_output",
        "raw_tool_output",
        "stdout",
        "stderr",
        "tool_output",
        "investigation_task_failed",
    }
)
_PRIVATE_TEXT_RE = re.compile(
    r"\b(?:workspace_ref|scope_id|codex_thread_id|thread_id|transcript)\b",
    re.IGNORECASE,
)


@runtime_checkable
class WorkingMemoryReader(Protocol):
    """Investigation 适配器所需的 Working Memory 最小只读接口。

    生产实现是 ``WorkingMemoryService``；测试可以注入只实现这四个查询的内存
    reader。所有方法都只读取已保存事实，不承担写入、状态推进或路径扫描。
    """

    def get_task_created(self, task_id: str) -> Mapping[str, Any]:
        """读取指定任务的 TaskCreated 冻结事实。"""

    def get_retrieval_completed(self, task_id: str) -> Mapping[str, Any] | None:
        """读取指定任务的 RetrievalCompleted 结果，缺失时返回 ``None``。"""

    def get_batch_created(self, task_id: str, batch_id: str) -> Mapping[str, Any]:
        """读取指定批次的 BatchCreated 原始正文。"""

    def list_combinations(
        self,
        task_id: str,
        batch_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        """读取指定批次的组合结构化记录，不返回其它批次。"""


@runtime_checkable
class WorkingMemoryBatchReceiptReader(WorkingMemoryReader, Protocol):
    """重生 bridge 额外需要的 BatchCreated 事件事实读取接口。

    基础 Investigation 只读适配器可以只实现四个历史查询；重生恢复还必须读取
    已提交 BatchCreated 的 event_id、payload 和 source_refs，才能在 receipt 与
    Working Memory 分离落盘后逐值核对事实。因此 bridge 会在装配时拒绝缺少本接口
    的 reader，而不是等跨存储恢复到一半才失败。
    """

    def get_batch_created_receipt(
        self,
        task_id: str,
        batch_id: str,
    ) -> Mapping[str, Any]:
        """读取已提交 BatchCreated 的事件身份、正文和来源引用。"""


class _FrozenInput(BaseModel):
    """适配器公开输入共用的严格字段和冻结策略。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        serialize_by_alias=True,
        validate_assignment=True,
    )


class WorkingMemoryCombinationInput(_FrozenInput):
    """一个组合的身份和 BatchCreated 中原始 ranked hypothesis。

    输入由只读 reader 提供；适配器只复制 ``hypothesis`` 的结构化正文，不从
    Investigation narrative 或路由 marker 推导正文。返回模型冻结且不含 workflow、
    current_workflow 或 runtime checkpoint；身份不匹配时适配器在返回前失败。
    """

    task_id: StrictStr
    batch_id: StrictStr
    combination_id: StrictStr
    combination_index: StrictInt
    hypothesis_id: StrictStr
    hypothesis: dict[str, Any]


class InvestigationWorkingMemoryInput(_FrozenInput):
    """首次 Investigation 可见的五组合科学输入。

    输入是任务、批次、冻结问题、TaskCreated 化学条件、Retrieval 结果和五项原始
    假设。构造过程只读 reader，不写 Working Memory；返回值按 combination_index
    排序并冻结。reader 缺事实、组合数量不是五、身份或冻结上下文漂移时抛出
    ``ValueError``，不会返回部分输入。
    """

    task_id: StrictStr
    batch_id: StrictStr
    batch_index: StrictInt
    scientific_question: StrictStr
    chemistry_context: dict[str, Any]
    retrieval_context: dict[str, Any]
    combinations: tuple[WorkingMemoryCombinationInput, ...]


class HypothesisRegenerationBatchInput(_FrozenInput):
    """一批真实调查/Review 历史和对应的五项原始假设。

    ``sessions``、``results``、``validations`` 来自既有重生历史记录，
    ``combinations`` 来自同一 ``batch_id`` 的 BatchCreated 与组合结构化记录。适配器只
    做按身份的联接和私有字段删除，不根据 narrative 或 ``revise_hypothesis``
    marker 发明 hypothesis；任一批次映射缺失都会 fail-closed。
    """

    task_id: StrictStr
    batch_id: StrictStr
    batch_index: StrictInt
    route: StrictStr
    final_answer: StrictStr | None
    answerability_explanation: StrictStr | None
    sessions: tuple[dict[str, Any], ...]
    results: tuple[dict[str, Any], ...]
    validations: tuple[dict[str, Any], ...]
    combinations: tuple[WorkingMemoryCombinationInput, ...]


class HypothesisRegenerationWorkingMemoryInput(_FrozenInput):
    """供下一批假设模型读取的冻结 Working Memory 输入。

    输入为 ``build_hypothesis_regeneration_memory(state)`` 的真实调查与 Review
    历史，并补入每个历史批次按 ``(batch_id, combination_id)`` 找回的原始结构化
    hypothesis 以及当前冻结科学上下文。该对象只读、无副作用；任何历史批次或
    hypothesis 映射缺失、身份漂移、私有运行字段或不安全文本都会失败。
    """

    schema_name: StrictStr = Field(alias="schema")
    schema_version: StrictInt
    task_id: StrictStr
    batch_id: StrictStr
    scientific_question: StrictStr
    chemistry_context: dict[str, Any]
    retrieval_context: dict[str, Any]
    closed_batches: tuple[HypothesisRegenerationBatchInput, ...]
    current_batch: HypothesisRegenerationBatchInput

    def to_prompt_payload(self) -> dict[str, Any]:
        """返回独立 JSON 副本，供下一批模型请求读取且不暴露 Pydantic 对象。"""

        return copy.deepcopy(self.model_dump(mode="json", by_alias=True))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """确认 reader 返回 JSON object；缺失或其它类型立即停止恢复。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必须是 object")
    return value


def _require_text(value: Any, label: str) -> str:
    """确认身份或科学问题是非空文本，并检查私有文本泄漏。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    if _PRIVATE_TEXT_RE.search(value):
        raise ValueError(f"{label} 不能包含私有会话身份")
    _assert_no_path_or_secret(value, label=label, allow_public_source_uris=True)
    return value


def _normalized_key(key: str) -> str:
    """将字段名归一化，便于识别大小写或连字符变体的内部字段。"""

    return re.sub(r"[^a-z0-9]", "", key.casefold())


_NORMALIZED_PRIVATE_KEYS = frozenset(_normalized_key(item) for item in _PRIVATE_KEYS)


def _copy_public_json_with_mode(
    value: Any,
    label: str,
    *,
    allow_scientific_layout: bool,
) -> Any:
    """复制公开 JSON 并移除私有字段，按调用方模式选择正文边界。"""

    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} 的 object key 必须是非空字符串")
            if _normalized_key(key) in _NORMALIZED_PRIVATE_KEYS:
                continue
            copied[key] = _copy_public_json_with_mode(
                item,
                f"{label}.{key}",
                allow_scientific_layout=allow_scientific_layout,
            )
        if allow_scientific_layout:
            return copied
        _assert_no_path_or_secret(
            copied,
            label=label,
            allow_public_source_uris=True,
        )
        return copied
    if isinstance(value, (list, tuple)):
        copied = [
            _copy_public_json_with_mode(
                item,
                f"{label}[{index}]",
                allow_scientific_layout=allow_scientific_layout,
            )
            for index, item in enumerate(value)
        ]
        if allow_scientific_layout:
            return copied
        _assert_no_path_or_secret(
            copied,
            label=label,
            allow_public_source_uris=True,
        )
        return copied
    if isinstance(value, str):
        if _PRIVATE_TEXT_RE.search(value):
            raise ValueError(f"{label} 不能包含私有会话身份")
        if allow_scientific_layout:
            return value
        _assert_no_path_or_secret(
            value,
            label=label,
            allow_public_source_uris=True,
        )
        return value
    if value is None or isinstance(value, (bool, int, float)):
        if allow_scientific_layout:
            return copy.deepcopy(value)
        else:
            _assert_no_path_or_secret(
                value,
                label=label,
                allow_public_source_uris=True,
            )
        return copy.deepcopy(value)
    raise TypeError(f"{label} 不是可序列化的 JSON 值")


def _copy_public_json(value: Any, label: str) -> Any:
    """严格复制历史公开 JSON，保持结果和验证正文单行。"""

    return _copy_public_json_with_mode(
        value,
        label,
        allow_scientific_layout=False,
    )


def _copy_public_scientific_json(value: Any, label: str) -> Any:
    """复制公开科学 JSON，允许正文排版空白但仍移除运行私有字段。"""

    copied = _copy_public_json_with_mode(
        value,
        label,
        allow_scientific_layout=True,
    )
    return require_safe_public_scientific_context(copied, label=label)


def _read_retrieval_context(
    reader: WorkingMemoryReader,
    task_id: str,
) -> dict[str, Any]:
    """从 reader 提取 RetrievalCompleted 的科学正文，隐藏事件包装字段。"""

    record = reader.get_retrieval_completed(task_id)
    if record is None:
        raise ValueError("任务缺少 RetrievalCompleted 冻结事实")
    value = _require_mapping(record, "RetrievalCompleted")
    result = value.get("retrieval_result") if "retrieval_result" in value else value
    result = _require_mapping(result, "RetrievalCompleted.retrieval_result")
    projected = _copy_public_scientific_json(result, "retrieval_context")
    if not isinstance(projected, dict):  # pragma: no cover - 由上方检查保证
        raise ValueError("retrieval_context 必须是 object")
    return projected


def _read_task_context(
    reader: WorkingMemoryReader,
    task_id: str,
) -> tuple[str, dict[str, Any]]:
    """读取 TaskCreated 的问题和化学条件，不暴露任务配置与输出目录。"""

    created = _require_mapping(reader.get_task_created(task_id), "TaskCreated")
    question = require_safe_public_scientific_text(
        created.get("scientific_question"),
        label="scientific_question",
    )
    chemistry = _require_mapping(
        created.get("chemistry_context"),
        "TaskCreated.chemistry_context",
    )
    projected = _copy_public_scientific_json(chemistry, "chemistry_context")
    if not isinstance(projected, dict):  # pragma: no cover - 由上方检查保证
        raise ValueError("chemistry_context 必须是 object")
    return question, projected


def _require_batch_index(value: Any, label: str) -> int:
    """校验从一开始计数的批次序号。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} 必须是正整数")
    return value


def _validate_batch_identity_and_snapshot(
    batch: Mapping[str, Any],
    *,
    task_id: str,
    batch_id: str,
    scientific_question: str,
    retrieval_context: Mapping[str, Any],
) -> int:
    """核对 BatchCreated 身份和首批冻结的问题、Retrieval 快照。"""

    if "task_id" in batch and batch.get("task_id") != task_id:
        raise ValueError("BatchCreated.task_id 与请求任务不匹配")
    if batch.get("batch_id") != batch_id:
        raise ValueError("BatchCreated.batch_id 与请求批次不匹配")
    batch_index = _require_batch_index(batch.get("batch_index"), "BatchCreated.batch_index")
    question_snapshot = batch.get("question_snapshot")
    if batch_index == 1:
        snapshot = _require_mapping(question_snapshot, "question_snapshot")
        if snapshot.get("scientific_question") != scientific_question:
            raise ValueError("BatchCreated 的 scientific_question 与 TaskCreated 漂移")
        frozen_retrieval = _require_mapping(
            snapshot.get("initial_retrieval_phase"),
            "question_snapshot.initial_retrieval_phase",
        )
        projected_snapshot = _copy_public_scientific_json(
            frozen_retrieval,
            "question_snapshot.initial_retrieval_phase",
        )
        if projected_snapshot != dict(retrieval_context):
            raise ValueError("BatchCreated 的冻结 Retrieval 上下文与 RetrievalCompleted 漂移")
    elif question_snapshot is not None:
        raise ValueError("非首批 BatchCreated 不应重新保存 question_snapshot")
    return batch_index


def _index_ranked_hypotheses(
    batch: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    """校验 ranked hypothesis 正文并建立身份、顺序索引。"""

    phase = _require_mapping(batch.get("hypothesis_phase"), "hypothesis_phase")
    ranked_raw = phase.get("ranked_strategies")
    if not isinstance(ranked_raw, list) or not ranked_raw:
        raise ValueError("BatchCreated 缺少 ranked_strategies")
    ranked_by_id: dict[str, Mapping[str, Any]] = {}
    ranked_by_position: dict[int, Mapping[str, Any]] = {}
    for index, raw in enumerate(ranked_raw, start=1):
        hypothesis = _require_mapping(raw, f"ranked_strategies[{index - 1}]")
        hypothesis_id = hypothesis.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise ValueError("ranked hypothesis 缺少 hypothesis_id")
        if hypothesis_id in ranked_by_id:
            raise ValueError("ranked hypothesis 存在重复 hypothesis_id")
        try:
            validate_structured_hypothesis_claims(
                hypothesis,
                path=f"ranked_strategies[{index - 1}]",
                allow_legacy=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("ranked hypothesis 正文缺失或结构非法") from exc
        ranked_by_id[hypothesis_id] = hypothesis
        ranked_by_position[index] = hypothesis
    return ranked_raw, ranked_by_id, ranked_by_position


def _validate_candidate_position_and_status(
    raw_candidate: Any,
    *,
    index: int,
    ranked_by_position: Mapping[int, Mapping[str, Any]],
    expected_count: int,
) -> tuple[Mapping[str, Any], bool]:
    """核对候选字段、ranked 顺序和 selected/not_selected 状态。"""

    candidate = _require_mapping(raw_candidate, f"candidates[{index - 1}]")
    if set(candidate) != {
        "hypothesis_id",
        "candidate_index",
        "rank",
        "selection_status",
        "combination_id",
        "combination_index",
    }:
        raise ValueError("candidate 字段不完整或包含未知字段")
    expected_hypothesis = ranked_by_position[index]
    hypothesis_id = candidate.get("hypothesis_id")
    if hypothesis_id != expected_hypothesis.get("hypothesis_id"):
        raise ValueError("candidate.hypothesis_id 与 ranked_strategies 逐项漂移")
    candidate_index = candidate.get("candidate_index")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise ValueError("candidate.candidate_index 必须是整数")
    if candidate_index != index:
        raise ValueError("candidate 的 candidate_index 与 ranked 顺序漂移")
    rank = candidate.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError("candidate.rank 必须是整数")
    if rank != index:
        raise ValueError("candidate.rank 与 ranked 顺序漂移")
    expected_status = "selected" if index <= expected_count else "not_selected"
    if candidate.get("selection_status") != expected_status:
        raise ValueError("candidate.selection_status 与冻结 selected 数量漂移")
    if expected_status == "not_selected":
        if candidate.get("combination_id") is not None or candidate.get("combination_index") is not None:
            raise ValueError("not_selected candidate 不能携带 combination 身份")
        return candidate, False
    return candidate, True


def _validate_selected_candidate_identity(
    candidate: Mapping[str, Any],
    *,
    index: int,
    ranked_by_id: Mapping[str, Mapping[str, Any]],
    ranked_by_position: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """核对 selected candidate 的组合和 hypothesis 身份。"""

    combination_id = candidate.get("combination_id")
    hypothesis_id = candidate.get("hypothesis_id")
    combination_index = candidate.get("combination_index")
    if not isinstance(combination_id, str) or not combination_id.strip():
        raise ValueError("selected candidate 缺少 combination_id")
    if not isinstance(hypothesis_id, str) or hypothesis_id not in ranked_by_id:
        raise ValueError("selected candidate 没有唯一 ranked hypothesis")
    if (
        isinstance(combination_index, bool)
        or not isinstance(combination_index, int)
        or combination_index < 1
    ):
        raise ValueError("selected candidate 缺少合法 combination_index")
    if ranked_by_id[hypothesis_id].get("hypothesis_id") != hypothesis_id:
        raise ValueError("candidate 与 ranked hypothesis 身份漂移")
    ranked_at_position = ranked_by_position.get(index)
    if ranked_at_position is None or ranked_at_position.get("hypothesis_id") != hypothesis_id:
        raise ValueError("candidate 与 ranked 顺序或 hypothesis_id 漂移")
    return candidate


def _validate_ranked_candidate(
    raw_candidate: Any,
    *,
    index: int,
    ranked_by_id: Mapping[str, Mapping[str, Any]],
    ranked_by_position: Mapping[int, Mapping[str, Any]],
    expected_count: int,
) -> Mapping[str, Any] | None:
    """核对一个候选并返回 selected candidate，未选项返回 ``None``。"""

    candidate, is_selected = _validate_candidate_position_and_status(
        raw_candidate,
        index=index,
        ranked_by_position=ranked_by_position,
        expected_count=expected_count,
    )
    if not is_selected:
        return None
    return _validate_selected_candidate_identity(
        candidate,
        index=index,
        ranked_by_id=ranked_by_id,
        ranked_by_position=ranked_by_position,
    )


def _index_selected_candidates(
    ranked_raw: list[Any],
    *,
    ranked_by_id: Mapping[str, Mapping[str, Any]],
    ranked_by_position: Mapping[int, Mapping[str, Any]],
    candidates_raw: Any,
    expected_count: int,
) -> dict[str, Mapping[str, Any]]:
    """按 ranked 顺序校验候选并建立冻结数量的 selected 组合索引。"""

    if not isinstance(candidates_raw, list):
        raise ValueError("BatchCreated 缺少 candidates")
    if len(candidates_raw) != len(ranked_raw):
        raise ValueError("candidates 必须逐项覆盖全部 ranked_strategies")
    selected_by_combination: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(candidates_raw, start=1):
        candidate = _validate_ranked_candidate(
            raw,
            index=index,
            ranked_by_id=ranked_by_id,
            ranked_by_position=ranked_by_position,
            expected_count=expected_count,
        )
        if candidate is None:
            continue
        combination_id = candidate["combination_id"]
        if combination_id in selected_by_combination:
            raise ValueError("selected candidate 存在重复 combination_id")
        selected_by_combination[combination_id] = candidate
    if len(selected_by_combination) != expected_count:
        raise ValueError("Investigation 必须恢复 TaskCreated 冻结数量的 selected combination")
    return selected_by_combination


def _validate_selected_combination_indexes(
    selected_by_combination: Mapping[str, Mapping[str, Any]],
    expected_count: int,
) -> None:
    """核对 selected 组合索引恰好覆盖冻结数量的连续位置。"""

    if sorted(
        candidate["combination_index"]
        for candidate in selected_by_combination.values()
    ) != list(range(1, expected_count + 1)):
        raise ValueError("selected combination_index 必须覆盖冻结数量的连续位置")


def _restore_combination_input(
    raw: Any,
    *,
    index: int,
    task_id: str,
    batch_id: str,
    selected: Mapping[str, Mapping[str, Any]],
    ranked: Mapping[str, Mapping[str, Any]],
    seen_ids: set[str],
    seen_indexes: set[int],
) -> WorkingMemoryCombinationInput:
    """校验一项组合记录并还原其公开 hypothesis 正文。"""

    combination = _require_mapping(raw, f"combinations[{index}]")
    combination_id, combination_index, hypothesis_id = _read_combination_identity(
        combination,
        task_id=task_id,
        batch_id=batch_id,
        seen_ids=seen_ids,
        seen_indexes=seen_indexes,
    )
    public_hypothesis = _resolve_combination_hypothesis(
        combination_id=combination_id,
        combination_index=combination_index,
        hypothesis_id=hypothesis_id,
        selected=selected,
        ranked=ranked,
    )
    return WorkingMemoryCombinationInput(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        combination_index=combination_index,
        hypothesis_id=hypothesis_id,
        hypothesis=public_hypothesis,
    )


def _read_combination_identity(
    combination: Mapping[str, Any],
    *,
    task_id: str,
    batch_id: str,
    seen_ids: set[str],
    seen_indexes: set[int],
) -> tuple[str, int, str]:
    """校验组合记录的任务、批次、组合和 hypothesis 身份。"""

    if combination.get("task_id") != task_id:
        raise ValueError("组合记录的 task_id 与请求任务不匹配")
    if combination.get("batch_id") != batch_id:
        raise ValueError("组合记录的 batch_id 与请求批次不匹配")
    combination_id = combination.get("combination_id")
    combination_index = combination.get("combination_index")
    hypothesis_id = combination.get("hypothesis_id")
    if not isinstance(combination_id, str) or not combination_id.strip():
        raise ValueError("组合记录缺少 combination_id")
    if combination_id in seen_ids:
        raise ValueError("组合记录存在重复 combination_id")
    if (
        isinstance(combination_index, bool)
        or not isinstance(combination_index, int)
        or combination_index in seen_indexes
    ):
        raise ValueError("组合记录存在重复或非法 combination_index")
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        raise ValueError("组合记录缺少 hypothesis_id")
    return combination_id, combination_index, hypothesis_id


def _resolve_combination_hypothesis(
    *,
    combination_id: str,
    combination_index: int,
    hypothesis_id: str,
    selected: Mapping[str, Mapping[str, Any]],
    ranked: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """从 selected 和 ranked 事实中恢复一个组合的公开 hypothesis。"""

    candidate = selected.get(combination_id)
    if candidate is None:
        raise ValueError("组合记录不是 BatchCreated 中的 selected combination")
    if (
        candidate.get("combination_index") != combination_index
        or candidate.get("hypothesis_id") != hypothesis_id
    ):
        raise ValueError("组合记录与 selected candidate 身份漂移")
    original = ranked.get(hypothesis_id)
    if original is None:
        raise ValueError("组合记录缺少唯一 ranked hypothesis 正文")
    public_hypothesis = _copy_public_scientific_json(
        original,
        f"ranked_strategies[{hypothesis_id}]",
    )
    if not isinstance(public_hypothesis, dict):  # pragma: no cover
        raise ValueError("ranked hypothesis 必须是 object")
    return public_hypothesis


def _extract_ranked_hypotheses(
    batch: Mapping[str, Any],
    *,
    task_id: str,
    batch_id: str,
    scientific_question: str,
    retrieval_context: Mapping[str, Any],
    expected_count: int,
) -> tuple[int, dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """校验 BatchCreated 的问题快照并建立 candidate/ranked 身份索引。"""

    batch_index = _validate_batch_identity_and_snapshot(
        batch,
        task_id=task_id,
        batch_id=batch_id,
        scientific_question=scientific_question,
        retrieval_context=retrieval_context,
    )
    ranked_raw, ranked_by_id, ranked_by_position = _index_ranked_hypotheses(batch)
    selected_by_combination = _index_selected_candidates(
        ranked_raw,
        ranked_by_id=ranked_by_id,
        ranked_by_position=ranked_by_position,
        candidates_raw=batch.get("candidates"),
        expected_count=expected_count,
    )
    _validate_selected_combination_indexes(selected_by_combination, expected_count)
    return batch_index, selected_by_combination, ranked_by_id


def _restore_combinations(
    reader: WorkingMemoryReader,
    *,
    task_id: str,
    batch_id: str,
    batch: Mapping[str, Any],
    scientific_question: str,
    retrieval_context: Mapping[str, Any],
) -> tuple[int, tuple[WorkingMemoryCombinationInput, ...]]:
    """联接冻结数量的组合记录与 ranked hypothesis，并按组合索引排序。"""

    task_created = reader.get_task_created(task_id)
    task_config = task_created.get("task_config") if isinstance(task_created, Mapping) else None
    # 旧的纯内存 test double 没有 TaskCreated；它只能代表兼容的标准五组合，
    # 真实 v2 任务始终保存并校验显式 task_config，作者实验不会走这个回退。
    expected_count = (
        5
        if task_config is None
        else get_investigation_combination_count(task_config)
    )

    batch_index, selected, ranked = _extract_ranked_hypotheses(
        batch,
        task_id=task_id,
        batch_id=batch_id,
        scientific_question=scientific_question,
        retrieval_context=retrieval_context,
        expected_count=expected_count,
    )
    rows = reader.list_combinations(task_id, batch_id)
    if isinstance(rows, (str, bytes, bytearray, Mapping)):
        raise ValueError("组合记录必须是 object 数组")
    rows = list(rows)
    if len(rows) != expected_count:
        raise ValueError("组合记录必须恰好包含 TaskCreated 冻结数量")
    restored: list[WorkingMemoryCombinationInput] = []
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for index, raw in enumerate(rows):
        combination = _restore_combination_input(
            raw,
            index=index,
            task_id=task_id,
            batch_id=batch_id,
            selected=selected,
            ranked=ranked,
            seen_ids=seen_ids,
            seen_indexes=seen_indexes,
        )
        restored.append(combination)
        seen_ids.add(combination.combination_id)
        seen_indexes.add(combination.combination_index)
    if seen_ids != set(selected) or seen_indexes != set(range(1, expected_count + 1)):
        raise ValueError("组合没有完整覆盖 selected candidate")
    restored.sort(key=lambda item: item.combination_index)
    return batch_index, tuple(restored)


def _read_batch_context(
    reader: WorkingMemoryReader,
    *,
    task_id: str,
    batch_id: str,
    scientific_question: str,
    retrieval_context: Mapping[str, Any],
) -> tuple[int, tuple[WorkingMemoryCombinationInput, ...]]:
    """读取一个批次事实并恢复其五项 Investigation 输入。"""

    batch = _require_mapping(
        reader.get_batch_created(task_id, batch_id),
        "BatchCreated",
    )
    return _restore_combinations(
        reader,
        task_id=task_id,
        batch_id=batch_id,
        batch=batch,
        scientific_question=scientific_question,
        retrieval_context=retrieval_context,
    )


def read_investigation_working_memory_input(
    reader: WorkingMemoryReader,
    task_id: str,
    batch_id: str,
) -> InvestigationWorkingMemoryInput:
    """读取指定批次的冻结 Investigation 输入。

    输入是注入的 Working Memory-like reader、任务 ID 和批次 ID；函数按固定顺序
    读取 TaskCreated、RetrievalCompleted、BatchCreated 与组合结构化记录，核对问题、冻结
    Retrieval、五个 selected combination 及每个 ranked hypothesis 的身份。函数只读
    且不修改 reader；成功返回冻结 Pydantic 输入，失败抛出 ``ValueError``。
    """

    if not isinstance(reader, WorkingMemoryReader):
        raise TypeError("reader 必须实现 WorkingMemoryReader 只读接口")
    task_id = _require_text(task_id, "task_id")
    batch_id = _require_text(batch_id, "batch_id")
    scientific_question, chemistry_context = _read_task_context(reader, task_id)
    retrieval_context = _read_retrieval_context(reader, task_id)
    batch_index, combinations = _read_batch_context(
        reader,
        task_id=task_id,
        batch_id=batch_id,
        scientific_question=scientific_question,
        retrieval_context=retrieval_context,
    )
    return InvestigationWorkingMemoryInput(
        task_id=task_id,
        batch_id=batch_id,
        batch_index=batch_index,
        scientific_question=scientific_question,
        chemistry_context=chemistry_context,
        retrieval_context=copy.deepcopy(retrieval_context),
        combinations=combinations,
    )


def _project_history_batch(value: Any, label: str) -> dict[str, Any]:
    """复制既有重生批次历史并移除 Review thread 等内部字段。"""

    raw = _require_mapping(value, label)
    required = {
        "task_id",
        "batch_id",
        "route",
        "answerability_explanation",
        "sessions",
        "results",
        "validations",
    }
    if not required <= set(raw) or set(raw) - required - {"final_answer"}:
        raise ValueError(f"{label} 的重生历史字段不完整")
    projected = _copy_public_json(raw, label)
    if not isinstance(projected, dict):  # pragma: no cover
        raise ValueError(f"{label} 必须是 object")
    # _copy_public_json 已删除 review_thread_id/review_turn_id；二次检查避免未来
    # 新增的私有字段以别名形式穿透到模型输入。
    projected.setdefault("final_answer", None)
    encoded_keys = {
        _normalized_key(key)
        for key in projected
        if isinstance(key, str)
    }
    if encoded_keys & _NORMALIZED_PRIVATE_KEYS:
        raise ValueError(f"{label} 含有私有运行字段")
    for collection_name in ("sessions", "results", "validations"):
        collection = projected.get(collection_name)
        if not isinstance(collection, list):
            raise ValueError(f"{label}.{collection_name} 必须是数组")
        if any(not isinstance(item, Mapping) for item in collection):
            raise ValueError(f"{label}.{collection_name} 内容非法")
    return projected


def _build_hypothesis_regeneration_batch_input(
    payload: Any,
    *,
    label: str,
    state: InvestigationState,
    loaded_batches: Mapping[
        str,
        tuple[int, tuple[WorkingMemoryCombinationInput, ...]],
    ],
) -> HypothesisRegenerationBatchInput:
    """把一项历史事实与同批次原始假设联接为重生输入。"""

    projected = _project_history_batch(payload, label)
    batch_id = projected.get("batch_id")
    if not isinstance(batch_id, str) or batch_id not in loaded_batches:
        raise ValueError(f"{label} 缺少与 reader 对应的 batch_id")
    batch_index, combinations = loaded_batches[batch_id]
    if projected.get("task_id") != state.task_id:
        raise ValueError(f"{label} 的 task_id 与 InvestigationState 漂移")
    session_ids = {
        item.get("combination_id")
        for item in projected["sessions"]
        if isinstance(item, Mapping)
    }
    combination_ids = {item.combination_id for item in combinations}
    if session_ids != combination_ids:
        raise ValueError(f"{label} 的历史会话没有覆盖 reader 的五个组合")
    return HypothesisRegenerationBatchInput(
        task_id=state.task_id,
        batch_id=batch_id,
        batch_index=batch_index,
        route=projected["route"],
        final_answer=projected["final_answer"],
        answerability_explanation=projected["answerability_explanation"],
        sessions=tuple(projected["sessions"]),
        results=tuple(projected["results"]),
        validations=tuple(projected["validations"]),
        combinations=combinations,
    )


def build_hypothesis_regeneration_working_memory_input(
    state: InvestigationState,
    reader: WorkingMemoryReader,
) -> HypothesisRegenerationWorkingMemoryInput:
    """联接真实调查/Review 历史与全部批次的原始 hypothesis。

    输入是已经满足 ``regenerate_hypotheses`` 终局合同的 InvestigationState 和注入
    reader。函数先调用既有 ``build_hypothesis_regeneration_memory(state)`` 获取真实
    调查、结果和 Review 历史，再逐批读取冻结上下文与五个组合，按
    ``(batch_id, combination_id)`` 精确合并原始 ranked hypothesis。函数只读、不写
    旧 workflow；任何批次或组合映射缺失、身份漂移、正文缺失或隐私字段都会抛出
    ``ValueError``，绝不从 narrative 或 marker 猜测假设。
    """

    if not isinstance(reader, WorkingMemoryReader):
        raise TypeError("reader 必须实现 WorkingMemoryReader 只读接口")
    if not isinstance(state, InvestigationState):
        raise TypeError("state 必须是 InvestigationState")
    history_payload = build_hypothesis_regeneration_memory(state)
    scientific_question, chemistry_context = _read_task_context(reader, state.task_id)
    retrieval_context = _read_retrieval_context(reader, state.task_id)

    batch_ids = [
        item.batch_id for item in state.history
    ] + [state.batch_id]
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Investigation 历史批次身份重复")
    loaded: dict[str, tuple[int, tuple[WorkingMemoryCombinationInput, ...]]] = {}
    for batch_id in batch_ids:
        loaded[batch_id] = _read_batch_context(
            reader,
            task_id=state.task_id,
            batch_id=batch_id,
            scientific_question=scientific_question,
            retrieval_context=retrieval_context,
        )

    closed_batches = tuple(
        _build_hypothesis_regeneration_batch_input(
            item,
            label=f"closed_batches[{index}]",
            state=state,
            loaded_batches=loaded,
        )
        for index, item in enumerate(history_payload.get("closed_batches", []))
    )
    current_batch = _build_hypothesis_regeneration_batch_input(
        history_payload.get("current_batch"),
        label="current_batch",
        state=state,
        loaded_batches=loaded,
    )
    return HypothesisRegenerationWorkingMemoryInput(
        schema=history_payload.get("schema"),
        schema_version=history_payload.get("schema_version"),
        task_id=state.task_id,
        batch_id=state.batch_id,
        scientific_question=scientific_question,
        chemistry_context=chemistry_context,
        retrieval_context=copy.deepcopy(retrieval_context),
        closed_batches=closed_batches,
        current_batch=current_batch,
    )
