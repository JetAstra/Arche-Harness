"""从 investigation_v2 ``ResearchTaskFacts`` 导出只读 Run Bundle 展示。

投影器只写展示文件，不读取 Harness 私有运行目录，也不调用模块、模型或
MiniChem。普通生命周期字段来自调用方传入的 ``ResearchTaskFacts``；Confirmation 的
候选和执行输入则通过同一 V2 Working Memory 的只读科学记录 seam 补齐。Run
Bundle 中的序号只是展示来源标记；恢复仍必须从 Working Memory 重新读取
Task Facts。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from chemistry_multiagent.memory import ResearchTaskClosed, ResearchTaskFacts
from chemistry_multiagent.utils.run_bundle import (
    RESEARCH_TASK_STAGE_LAYOUT,
    RunBundle,
)


_HISTORICAL_STAGE_MODULE_KINDS: dict[str, frozenset[str]] = {
    "retrieval": frozenset({"retrieval"}),
    # Query、Hypothesis 和 Selection 都是 hypothesis 阶段的公开准备边界。
    "hypothesis": frozenset(
        {
            "research-query-formulation",
            "hypothesis-generation",
            "hypothesis-selection",
            "hypothesis-confirmation",
        }
    ),
    "investigation_and_experiment": frozenset({"hypothesis-investigation"}),
    "final_result": frozenset({"task-review"}),
}

# 新 research_task 的四个研究员可见阶段。模块名称保持公开模块合同的正式词汇；
# _module_kind_family 只负责把历史别名归入这些正式模块，不会把旧名称写入新文件。
_RESEARCH_TASK_STAGE_MODULE_KINDS: dict[str, frozenset[str]] = {
    "literature_retrieval": frozenset({"retrieval"}),
    "hypothesis_preparation": frozenset(
        {
            "research-query-formulation",
            "hypothesis-generation",
            "hypothesis-import",
            "hypothesis-selection",
            "hypothesis-confirmation",
        }
    ),
    "investigation_and_experiment": frozenset({"hypothesis-investigation"}),
    "task_review_and_result": frozenset({"task-review"}),
}

# 保留旧变量名供同一模块的历史兼容测试使用；新的 Bundle 选择正式布局。
_STAGE_MODULE_KINDS = _HISTORICAL_STAGE_MODULE_KINDS

_CONFIRMATION_REASON_CODE = "awaiting_researcher_confirmation"


def _module_kind_family(module_kind: str) -> str:
    """把正式模块名和历史别名归入统一的公开模块族。"""

    normalized = module_kind.strip().casefold().replace("_", "-").replace(" ", "-")
    if normalized == "retrieval" or normalized.startswith("retrieval-"):
        return "retrieval"
    if normalized == "query":
        return "research-query-formulation"
    if normalized in {"research-query-formulation", "query-formulation"}:
        return "research-query-formulation"
    if normalized in {
        "hypothesis",
        "hypotheses",
        "hypothesis-generation",
        "hypothesis-import",
    }:
        return "hypothesis-generation"
    if normalized in {
        "hypothesis-confirmation",
        "hypothesis-confirm",
        "confirmation",
        "hypothesis-confirmation-module",
    }:
        return "hypothesis-confirmation"
    if normalized in {
        "selection",
        "hypothesis-selection",
        "top5",
        "top-5",
        "top5-selection",
        "top-5-selection",
    }:
        return "hypothesis-selection"
    if normalized in {
        "investigation",
        "investigation-and-experiment",
        "investigation-experiment",
        "hypothesis-investigation",
    } or normalized.startswith("investigation-"):
        return "hypothesis-investigation"
    if normalized in {"review", "task-review"} or normalized.endswith("-review"):
        return "task-review"
    return normalized


@dataclass(frozen=True)
class V2RunBundleProjection(Mapping[str, Any]):
    """一次 V2 展示投影的可验证结果。"""

    research_task_id: str
    source_task_sequence: int
    task_lifecycle: str
    module_invocation_lifecycle: str
    active_module_invocation: str
    controller_observation: str
    status: str
    stage_statuses: Mapping[str, str]
    reason_code: str | None = None
    hypothesis_confirmation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """返回不依赖 Bundle 文件的投影摘要。"""

        document: dict[str, Any] = {
            "research_task_id": self.research_task_id,
            "source_task_sequence": self.source_task_sequence,
            "task_lifecycle": self.task_lifecycle,
            "module_invocation_lifecycle": self.module_invocation_lifecycle,
            "active_module_invocation": self.active_module_invocation,
            "controller_observation": self.controller_observation,
            "status": self.status,
            "stage_statuses": dict(self.stage_statuses),
        }
        if self.reason_code is not None or self.hypothesis_confirmation is not None:
            document["reason_code"] = self.reason_code
        if self.hypothesis_confirmation is not None:
            document["hypothesis_confirmation"] = deepcopy(
                self.hypothesis_confirmation
            )
        return document

    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问。"""

        return self.to_dict()[key]

    def __iter__(self):
        """按稳定顺序迭代摘要字段。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回公开字段数。"""

        return len(self.to_dict())


def _validate_source_task_sequence(value: Any) -> int:
    """校验投影必须引用有效 WM 事件位置。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("ResearchTaskFacts.research_task_sequence 必须是正整数")
    return value


def _facts_sequence(facts: ResearchTaskFacts) -> int:
    """读取 canonical research_task 序号。"""

    return _validate_source_task_sequence(facts.research_task_sequence)


def _facts_open_module_invocation(facts: ResearchTaskFacts) -> Any:
    """读取当前 open Module Invocation。"""

    return facts.open_module_invocation


def _facts_module_invocations(facts: ResearchTaskFacts) -> tuple[Any, ...]:
    """读取全部 Module Invocation。"""

    return facts.module_invocations


def _active_module_invocation_state(facts: ResearchTaskFacts) -> str:
    """返回 canonical active Module Invocation 状态。"""

    state = facts.active_module_invocation.state
    if not isinstance(state, str) or not state.strip():
        raise ValueError("ResearchTaskFacts 缺少 active_module_invocation.state")
    return state


def _module_invocation_lifecycle(facts: ResearchTaskFacts) -> str:
    """把当前模块快照转换为不含 running/unknown 的展示词。"""

    if _facts_open_module_invocation(facts) is not None:
        return "Open"
    if _facts_module_invocations(facts):
        return "Closed"
    return "NotStarted"


def _root_status(facts: ResearchTaskFacts) -> str:
    """把账本终态映射到兼容 Bundle 的根状态。"""

    if facts.lifecycle.state == "Open":
        return "open"
    return (
        "failed"
        if facts.lifecycle.outcome.get("status") == "failed"
        else "succeeded"
    )


def _stage_status(
    facts: ResearchTaskFacts,
    stage: str,
    *,
    stage_module_kinds: Mapping[str, frozenset[str]] | None = None,
) -> str:
    """根据公开 Module Run 生命周期计算一个阶段的展示状态。"""

    kinds = (stage_module_kinds or _STAGE_MODULE_KINDS).get(stage, frozenset())
    modules = [
        module
        for module in _facts_module_invocations(facts)
        if _module_kind_family(module.module_kind) in kinds
    ]
    if any(module.lifecycle.state == "Open" for module in modules):
        return "open"
    if any(
        isinstance(module.lifecycle, ResearchTaskClosed)
        and module.lifecycle.outcome.get("status") == "failed"
        for module in modules
    ):
        return "failed"
    if modules:
        return "completed"
    return "not_started"


def _record_field(record: Any, field: str, default: Any = None) -> Any:
    """从公开 ScientificRecord 或测试替代记录读取一个字段。"""

    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


def _module_input_refs(module: Any) -> tuple[str, ...]:
    """读取 Module Invocation 的科学输入引用。"""

    return _unique_refs(_record_field(module, "input_scientific_record_ids", ()))


def _module_identity(module: Any) -> str:
    """读取 canonical 模块调用身份。"""

    value = _record_field(module, "module_invocation_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Module Invocation 缺少 module_invocation_id")
    return value


def _facts_task_identity(facts: ResearchTaskFacts) -> str:
    """读取 ResearchTaskFacts 的 canonical research_task 身份。"""

    value = facts.research_task_id
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ResearchTaskFacts 缺少 research_task_id")
    return value


def _record_payload(record: Any) -> Mapping[str, Any]:
    """读取科学记录 payload，并拒绝不完整的投影输入。"""

    payload = _record_field(record, "payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Confirmation ScientificRecord.payload 必须是对象")
    return payload


def _record_id(record: Any) -> str:
    """读取 canonical 记录身份。"""

    record_id = _record_field(record, "scientific_record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("Confirmation ScientificRecord 缺少 scientific_record_id")
    return record_id


def _record_source_refs(record: Any) -> tuple[str, ...]:
    """读取 canonical 来源引用。"""

    refs = _record_field(record, "source_scientific_record_ids", ())
    if not isinstance(refs, (list, tuple)):
        raise ValueError(
            "Confirmation ScientificRecord.source_scientific_record_ids 必须是数组"
        )
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError("Confirmation ScientificRecord.source_record_refs 含非法引用")
    return tuple(refs)


def _record_type(record: Any) -> str:
    """读取 canonical ScientificRecord 类型。"""

    record_type = _record_field(record, "scientific_record_type")
    if not isinstance(record_type, str) or not record_type.strip():
        raise ValueError("Confirmation ScientificRecord 缺少 scientific_record_type")
    return record_type


def _hypothesis_summary(hypothesis: Any) -> str | None:
    """从假设科学内容提取研究员可扫读的摘要。"""

    if isinstance(hypothesis, str) and hypothesis.strip():
        return hypothesis.strip()
    if not isinstance(hypothesis, Mapping):
        return None
    for field in ("summary", "claim", "statement", "description"):
        value = hypothesis.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidate_view(record: Any) -> dict[str, Any]:
    """把 SelectedHypothesisRecord 转成可提交确认的候选摘要。"""

    payload = _record_payload(record)
    candidate_record_id = _record_id(record)
    rank = payload.get("rank")
    hypothesis = payload.get("hypothesis")
    if not isinstance(candidate_record_id, str) or not candidate_record_id.strip():
        raise ValueError("SelectedHypothesisRecord 缺少科学记录身份")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("SelectedHypothesisRecord 缺少合法 rank")
    if not isinstance(hypothesis, Mapping) or not hypothesis:
        raise ValueError("SelectedHypothesisRecord 缺少完整 hypothesis")
    result: dict[str, Any] = {
        "hypothesis_candidate_record_id": candidate_record_id,
        "rank": rank,
        "summary": _hypothesis_summary(hypothesis),
    }
    result["hypothesis"] = deepcopy(hypothesis)
    return result


def _unique_refs(refs: Any) -> tuple[str, ...]:
    """按出现顺序去重记录引用，避免重复读取或重复投影。"""

    if not isinstance(refs, (list, tuple)):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if isinstance(ref, str) and ref.strip() and ref not in seen:
            result.append(ref)
            seen.add(ref)
    return tuple(result)


def _module_output_refs(module: Any) -> tuple[str, ...]:
    """读取已完成 Module Invocation 的输出引用；失败模块返回空集合。"""

    lifecycle = _record_field(module, "lifecycle")
    outcome = _record_field(lifecycle, "outcome", {})
    if not isinstance(outcome, Mapping):
        return ()
    return _unique_refs(outcome.get("output_scientific_record_ids", ()))


class InvestigationV2RunBundleProjector:
    """将一份 ``ResearchTaskFacts`` 投影到现有 Run Bundle 展示格式。"""

    def __init__(self, bundle: RunBundle, working_memory: Any | None = None) -> None:
        """绑定目标 Bundle 和可选只读 WM 门面。"""

        if not isinstance(bundle, RunBundle):
            raise TypeError("bundle 必须是 RunBundle")
        self.bundle = bundle
        self.working_memory = working_memory

    def _task_identity_document(self, facts: ResearchTaskFacts) -> dict[str, str]:
        """返回与 Bundle 命名空间一致的公开任务身份。"""

        return {"research_task_id": _facts_task_identity(facts)}

    def _read_confirmation_records(
        self,
        facts: ResearchTaskFacts,
        module: Any,
    ) -> dict[str, Any]:
        """从 Working Memory 读取确认模块所需的科学记录闭包。

        ``ResearchTaskFacts`` 当前只携带记录关系和引用，而不携带科学正文。生产装配已
        将同一个 V2 Working Memory 门面传给 projector，因此这里按记录引用读取
        权威事件；没有读取 seam 时拒绝伪造候选或材料摘要。
        """

        refs: list[str] = list(_module_input_refs(module))
        refs.extend(_module_output_refs(module))

        graph = _record_field(facts, "scientific_graph")
        graph_by_type = _record_field(graph, "record_refs_by_type", {})
        if isinstance(graph_by_type, Mapping):
            # 这些记录既能提供待确认候选，也能补齐确认输出所引用的取消项。
            for record_type in ("HypothesisBatchRecord", "SelectedHypothesisRecord"):
                refs.extend(_unique_refs(graph_by_type.get(record_type, ())))
        requested_refs = _unique_refs(refs)
        if not requested_refs:
            raise ValueError("Confirmation Module Run 缺少可读取的科学记录引用")

        embedded_records = _record_field(facts, "scientific_records")
        if isinstance(embedded_records, Mapping):
            records: dict[str, Any] = {}
            for ref in requested_refs:
                record = embedded_records.get(ref)
                if record is None:
                    raise ValueError(f"ResearchTaskFacts 缺少 Confirmation 记录 {ref}")
                records[_record_id(record)] = record
            return records

        reader = getattr(
            self.working_memory,
            "read_research_task_scientific_records",
            None,
        )
        if not callable(reader):
            raise ValueError(
                "投影 Confirmation 时必须提供 Working Memory scientific-record 读取 seam"
            )
        raw_records = reader(_facts_task_identity(facts), requested_refs)
        if isinstance(raw_records, Mapping):
            raw_records = tuple(raw_records.values())
        if not isinstance(raw_records, (list, tuple)):
            raise ValueError("Working Memory scientific-record seam 必须返回记录序列")
        records = {}
        for record in raw_records:
            record_id = _record_id(record)
            if record_id in records:
                raise ValueError(f"Working Memory scientific-record seam 重复返回 {record_id}")
            records[record_id] = record
        missing = [ref for ref in requested_refs if ref not in records]
        if missing:
            raise ValueError("Working Memory 缺少 Confirmation 记录：" + ", ".join(missing))
        return records

    @staticmethod
    def _find_confirmation_module(facts: ResearchTaskFacts) -> tuple[Any | None, bool]:
        """找到最新 Confirmation；返回模块和是否仍在 Open。"""

        open_module = _facts_open_module_invocation(facts)
        if open_module is not None and _module_kind_family(
            _record_field(open_module, "module_kind", "")
        ) == "hypothesis-confirmation":
            return open_module, True
        for module in reversed(facts.closed_module_invocations):
            if _module_kind_family(_record_field(module, "module_kind", "")) == (
                "hypothesis-confirmation"
            ):
                # ModuleInvocationFailed 也会进入 closed_module_invocations，但没有
                # 可投影的 Decision/Input 输出；失败模块不能被误报为 confirmed。
                if not _module_output_refs(module):
                    return None, False
                return module, False
        return None, False

    @staticmethod
    def _find_batch_ref(
        module: Any,
        records: Mapping[str, Any],
    ) -> str:
        """从确认模块输入或候选来源找出唯一 Hypothesis Batch。"""

        input_refs = set(_module_input_refs(module))
        batches = {
            record_id: record
            for record_id, record in records.items()
            if _record_type(record) == "HypothesisBatchRecord"
        }
        direct = [record_id for record_id in input_refs if record_id in batches]
        if len(direct) == 1:
            return direct[0]
        candidates = [
            record
            for record in records.values()
            if _record_type(record) == "SelectedHypothesisRecord"
            and (not input_refs or _record_id(record) in input_refs)
        ]
        if not candidates:
            candidates = [
                record
                for record in records.values()
                if _record_type(record) == "SelectedHypothesisRecord"
            ]
        source_refs = {
            source_ref
            for candidate in candidates
            for source_ref in _record_source_refs(candidate)
            if source_ref in batches
        }
        if len(source_refs) == 1:
            return next(iter(source_refs))
        raise ValueError("Confirmation 无法从已核对记录确定唯一 Hypothesis Batch")

    @staticmethod
    def _selected_candidates(
        module: Any,
        records: Mapping[str, Any],
        batch_ref: str,
    ) -> list[Any]:
        """按 Batch 来源筛选候选，并按作者 rank 稳定排序。"""

        input_refs = set(_module_input_refs(module))
        candidates = [
            record
            for record in records.values()
            if _record_type(record) == "SelectedHypothesisRecord"
            and (
                batch_ref in _record_source_refs(record)
                or _record_id(record) in input_refs
            )
        ]
        if not candidates:
            raise ValueError("Confirmation 缺少与 Hypothesis Batch 绑定的候选")
        candidates.sort(
            key=lambda record: (
                _record_payload(record).get("rank", 0),
                _record_id(record),
            )
        )
        if not 1 <= len(candidates) <= 5:
            raise ValueError("Confirmation 候选数量必须为一至五条")
        ranks = [_record_payload(candidate).get("rank") for candidate in candidates]
        if ranks != list(range(1, len(candidates) + 1)):
            raise ValueError("Confirmation 候选 rank 必须从 1 连续排列")
        refs = [
            _record_payload(candidate).get("hypothesis_ref")
            for candidate in candidates
        ]
        if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
            raise ValueError("Confirmation 候选缺少 hypothesis_ref")
        if len(set(refs)) != len(refs):
            raise ValueError("Confirmation 候选 hypothesis_ref 必须唯一")
        return candidates

    @staticmethod
    def _candidate_aliases(candidates: list[Any]) -> dict[str, Any]:
        """建立候选科学记录身份与旧科学假设引用的只读关联。"""

        aliases: dict[str, Any] = {}
        for candidate in candidates:
            payload = _record_payload(candidate)
            for alias in (_record_id(candidate), payload.get("hypothesis_ref")):
                if not isinstance(alias, str) or not alias.strip():
                    continue
                previous = aliases.get(alias)
                if previous is not None and _record_id(previous) != _record_id(candidate):
                    raise ValueError("Confirmation 候选引用发生歧义")
                aliases[alias] = candidate
        return aliases

    def _build_awaiting_confirmation(
        self,
        facts: ResearchTaskFacts,
        module: Any,
    ) -> dict[str, Any]:
        """生成 Open Confirmation 的研究员可读视图。"""

        records = self._read_confirmation_records(facts, module)
        batch_ref = self._find_batch_ref(module, records)
        candidates = self._selected_candidates(module, records, batch_ref)
        sequence = _facts_sequence(facts)
        return {
            **self._task_identity_document(facts),
            "status": "awaiting_confirmation",
            "reason_code": _CONFIRMATION_REASON_CODE,
            "source_task_sequence": sequence,
            "expected_research_task_sequence": sequence,
            "hypothesis_batch_ref": batch_ref,
            "candidate_count": len(candidates),
            "candidates": [_candidate_view(candidate) for candidate in candidates],
        }

    def _build_confirmed_confirmation(
        self,
        facts: ResearchTaskFacts,
        module: Any,
    ) -> dict[str, Any]:
        """从已闭合确认记录生成执行、取消和材料元数据摘要。"""

        records = self._read_confirmation_records(facts, module)
        batch_ref = self._find_batch_ref(module, records)
        candidates = self._selected_candidates(module, records, batch_ref)
        candidate_aliases = self._candidate_aliases(candidates)
        output_refs = _module_output_refs(module)
        output_records = [records[ref] for ref in output_refs if ref in records]
        decisions = [
            record
            for record in output_records
            if _record_type(record) == "HypothesisExecutionDecisionRecord"
        ]
        inputs = [
            record
            for record in output_records
            if _record_type(record) == "HypothesisExecutionInputRecord"
        ]
        materials = [
            record
            for record in output_records
            if _record_type(record) == "ResearchMaterialRecord"
        ]
        unexpected_types = {
            _record_type(record)
            for record in output_records
        }.difference(
            {
                "HypothesisExecutionDecisionRecord",
                "HypothesisExecutionInputRecord",
                "ResearchMaterialRecord",
            }
        )
        if unexpected_types:
            raise ValueError("Closed Confirmation 含未注册输出记录")
        if len(decisions) != 1 or not inputs:
            raise ValueError("Closed Confirmation 缺少完整 Decision/Input 记录")

        decision_payload = _record_payload(decisions[0])
        confirmation_id = decision_payload.get("confirmation_id")
        if not isinstance(confirmation_id, str) or not confirmation_id.strip():
            raise ValueError("HypothesisExecutionDecisionRecord 缺少 confirmation_id")
        decision_views: list[dict[str, Any]] = []
        confirmed_candidate_ids: set[str] = set()
        for item in decision_payload.get("decisions", ()):
            if not isinstance(item, Mapping):
                raise ValueError("HypothesisExecutionDecisionRecord.decisions 含非法项")
            candidate_record_id = item.get("hypothesis_candidate_record_id")
            candidate = candidate_aliases.get(candidate_record_id)
            if candidate is None:
                raise ValueError("Confirmation decision 引用了未知候选")
            candidate_id = _record_id(candidate)
            if candidate_id in confirmed_candidate_ids:
                raise ValueError("Confirmation decision 重复覆盖候选")
            if item.get("decision") not in {"execute", "cancel"}:
                raise ValueError("Confirmation decision 含未知决定")
            confirmed_candidate_ids.add(candidate_id)
            view = {
                "hypothesis_candidate_record_id": candidate_id,
                "decision": item["decision"],
            }
            if item["decision"] == "execute":
                view["execution_rank"] = item.get("execution_rank")
            decision_views.append(view)

        if len(decision_views) != len(candidates):
            raise ValueError("Confirmation Decision 未覆盖全部候选")

        material_aliases: dict[str, Any] = {}
        for material in materials:
            payload = _record_payload(material)
            material_id = payload.get("material_id")
            if not isinstance(material_id, str) or not material_id.strip():
                raise ValueError("ResearchMaterialRecord 缺少 material_id")
            for alias in (material_id, _record_id(material)):
                previous = material_aliases.get(alias)
                if previous is not None and _record_id(previous) != _record_id(material):
                    raise ValueError("Confirmation material 引用发生歧义")
                material_aliases[alias] = material

        execution_views: list[dict[str, Any]] = []
        referenced_material_ids: set[str] = set()
        input_candidate_ids: set[str] = set()
        for input_record in inputs:
            payload = _record_payload(input_record)
            candidate = candidate_aliases.get(
                payload.get("source_selected_hypothesis_record_id")
            )
            if candidate is None:
                raise ValueError("Execution Input 引用了未知 SelectedHypothesisRecord")
            candidate_id = _record_id(candidate)
            if candidate_id in input_candidate_ids:
                raise ValueError("Execution Input 重复绑定候选")
            input_candidate_ids.add(candidate_id)
            if candidate_id not in confirmed_candidate_ids:
                raise ValueError("Execution Input 没有对应 execute decision")
            material_refs = payload.get("material_upload_refs", ())
            if not isinstance(material_refs, (list, tuple)):
                raise ValueError("Execution Input.material_upload_refs 必须是数组")
            material_ids: list[str] = []
            for material_ref in material_refs:
                material = material_aliases.get(material_ref)
                if material is None:
                    raise ValueError("Execution Input 引用了未知 ResearchMaterialRecord")
                material_id = _record_payload(material).get("material_id")
                if not isinstance(material_id, str) or not material_id.strip():
                    raise ValueError("ResearchMaterialRecord 缺少 material_id")
                material_ids.append(material_id)
                referenced_material_ids.add(material_id)
            execution_views.append(
                {
                    "hypothesis_candidate_record_id": candidate_id,
                    "execution_rank": payload.get("execution_rank"),
                    "hypothesis": deepcopy(payload.get("hypothesis")),
                    "guidance": payload.get("guidance"),
                    "material_ids": material_ids,
                }
            )
        execution_views.sort(key=lambda item: item["execution_rank"])

        material_views: list[dict[str, Any]] = []
        for material in materials:
            payload = _record_payload(material)
            material_id = payload["material_id"]
            if material_id not in referenced_material_ids:
                continue
            material_views.append(
                {
                    "material_id": material_id,
                    "filename": payload["filename"],
                    "media_type": payload["media_type"],
                    "size_bytes": payload["size_bytes"],
                    "sha256": payload["sha256"],
                    "description": payload.get("description"),
                }
            )

        candidate_views = [_candidate_view(candidate) for candidate in candidates]
        cancel_count = sum(1 for item in decision_views if item["decision"] == "cancel")
        sequence = _facts_sequence(facts)
        return {
            **self._task_identity_document(facts),
            "status": "confirmed",
            "source_task_sequence": sequence,
            "hypothesis_batch_ref": batch_ref,
            "confirmation_id": confirmation_id,
            "candidate_count": len(decision_views),
            "execution_count": len(execution_views),
            "cancel_count": cancel_count,
            "candidates": candidate_views,
            "decisions": decision_views,
            "execution_inputs": execution_views,
            "materials": material_views,
            "material_count": len(material_views),
        }

    def _confirmation_projection(
        self,
        facts: ResearchTaskFacts,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """读取最新 Confirmation 并生成对应研究员视图。"""

        module, is_open = self._find_confirmation_module(facts)
        if module is None:
            return None, None
        if is_open:
            return _CONFIRMATION_REASON_CODE, self._build_awaiting_confirmation(
                facts, module
            )
        return None, self._build_confirmed_confirmation(facts, module)

    def project(
        self,
        facts: ResearchTaskFacts,
        *,
        controller_present: bool = False,
    ) -> V2RunBundleProjection:
        """把当前 Task Facts 写入 status、summary 和全部阶段文件。

        ``controller_present`` 是调用方在读取事实时观察到的进程状态，不会写回
        Working Memory。传入 ``False`` 可用于进程缺席时的展示重导出。
        """

        if not isinstance(facts, ResearchTaskFacts):
            raise TypeError("facts 必须是 ResearchTaskFacts")
        if not isinstance(controller_present, bool):
            raise TypeError("controller_present 必须是布尔值")
        sequence = _facts_sequence(facts)
        task_lifecycle = facts.lifecycle.state
        active_module_invocation = _active_module_invocation_state(facts)
        module_invocation_lifecycle = _module_invocation_lifecycle(facts)
        controller_observation = "Present" if controller_present else "Absent"
        status = _root_status(facts)
        reason_code, hypothesis_confirmation = self._confirmation_projection(facts)
        display = {
            **self._task_identity_document(facts),
            "task_lifecycle": task_lifecycle,
            "module_invocation_lifecycle": module_invocation_lifecycle,
            "active_module_invocation": active_module_invocation,
            "controller_observation": controller_observation,
            "controller_present": controller_present,
            "lifecycle": {
                "task": task_lifecycle,
                "module_invocation": module_invocation_lifecycle,
            },
            "controller": {
                "observation": controller_observation,
                "presence": controller_observation,
            },
            "task_outcome": (
                dict(facts.lifecycle.outcome)
                if facts.lifecycle.state == "Closed"
                else None
            ),
        }
        if reason_code is not None or hypothesis_confirmation is not None:
            display["reason_code"] = reason_code
        if hypothesis_confirmation is not None:
            display["hypothesis_confirmation"] = hypothesis_confirmation
        stage_module_kinds = (
            _RESEARCH_TASK_STAGE_MODULE_KINDS
            if self.bundle.stage_layout == RESEARCH_TASK_STAGE_LAYOUT
            else _HISTORICAL_STAGE_MODULE_KINDS
        )
        stage_statuses = {
            stage: _stage_status(
                facts,
                stage,
                stage_module_kinds=stage_module_kinds,
            )
            for stage in self.bundle.stage_names
        }

        # 根状态先写入，随后逐个刷新阶段。每个写入都带同一 WM 序号；如果进程
        # 在中间中断，下一次 project/re-export 会从事实头重新覆盖展示文件。
        self.bundle.update_status(
            status,
            reason_code=reason_code,
            source_task_sequence=sequence,
            display=display,
        )
        for stage, stage_status in stage_statuses.items():
            stage_modules = [
                module
                for module in _facts_module_invocations(facts)
                if _module_kind_family(module.module_kind)
                in stage_module_kinds.get(stage, frozenset())
            ]
            stage_summary = {
                "status": stage_status,
                "source_task_sequence": sequence,
                "task_lifecycle": task_lifecycle,
                "module_invocation_lifecycle": module_invocation_lifecycle,
                "active_module_invocation": active_module_invocation,
                "controller_observation": controller_observation,
                "lifecycle": {
                    "task": task_lifecycle,
                    "module_invocation": module_invocation_lifecycle,
                },
                "controller": {"presence": controller_observation},
                "module_invocation_count": len(stage_modules),
                "module_invocation_ids": [
                    _module_identity(module) for module in stage_modules
                ],
            }
            preparation_stage = (
                "hypothesis_preparation"
                if self.bundle.stage_layout == RESEARCH_TASK_STAGE_LAYOUT
                else "hypothesis"
            )
            if hypothesis_confirmation is not None and stage == preparation_stage:
                stage_summary["hypothesis_confirmation"] = hypothesis_confirmation
            if reason_code is not None and stage == preparation_stage:
                stage_summary["reason_code"] = reason_code
            self.bundle.update_stage_status(
                stage,
                stage_status,
                summary=stage_summary,
                source_task_sequence=sequence,
            )
        return V2RunBundleProjection(
            research_task_id=_facts_task_identity(facts),
            source_task_sequence=sequence,
            task_lifecycle=task_lifecycle,
            module_invocation_lifecycle=module_invocation_lifecycle,
            active_module_invocation=active_module_invocation,
            controller_observation=controller_observation,
            status=status,
            stage_statuses=stage_statuses,
            reason_code=reason_code,
            hypothesis_confirmation=hypothesis_confirmation,
        )

    def project_task_facts(
        self,
        facts: ResearchTaskFacts,
        *,
        controller_present: bool = False,
    ) -> V2RunBundleProjection:
        """用显式动作名称投影一份 Task Facts。"""

        return self.project(facts, controller_present=controller_present)

    def re_export(
        self,
        facts: ResearchTaskFacts | str,
        *,
        controller_present: bool = False,
    ) -> V2RunBundleProjection:
        """从现有 Task Facts 重导出展示，不调用任何模块。

        传入 ``ResearchTaskFacts`` 时完全不访问 WM；传入 task id 时只通过构造器提供的
        WM 门面读取事实，仍不会调用模块、模型或 MiniChem。
        """

        if isinstance(facts, str):
            if self.working_memory is None:
                raise ValueError("按 task id 重导出时必须提供 working_memory")
            reader = getattr(
                self.working_memory,
                "read_research_task_facts",
                None,
            )
            if not callable(reader):
                raise TypeError(
                    "working_memory 必须提供 read_research_task_facts()"
                )
            facts = reader(facts)
        return self.project(facts, controller_present=controller_present)

    def re_export_task_facts(
        self,
        facts: ResearchTaskFacts | str,
        *,
        controller_present: bool = False,
    ) -> V2RunBundleProjection:
        """从 Task Facts 重新导出 Bundle 展示。"""

        return self.re_export(facts, controller_present=controller_present)


def project_task_facts_to_run_bundle(
    bundle: RunBundle,
    facts: ResearchTaskFacts,
    *,
    working_memory: Any | None = None,
    controller_present: bool = False,
) -> V2RunBundleProjection:
    """函数式 V2 投影入口，方便恢复与展示修复路径复用。"""

    return InvestigationV2RunBundleProjector(bundle, working_memory).project(
        facts,
        controller_present=controller_present,
    )


def re_export_task_facts_to_run_bundle(
    bundle: RunBundle,
    working_memory: Any,
    research_task_id: str,
    *,
    controller_present: bool = False,
) -> V2RunBundleProjection:
    """只读取指定 research_task Facts 并重导出 Bundle 展示。"""

    return InvestigationV2RunBundleProjector(
        bundle,
        working_memory=working_memory,
    ).re_export(research_task_id, controller_present=controller_present)


# 提供简短别名，避免迁移期间调用方把该投影器误认作旧 publisher。
V2RunBundleProjector = InvestigationV2RunBundleProjector
RunBundleProjector = InvestigationV2RunBundleProjector
InvestigationV2RunBundleProjection = V2RunBundleProjection


__all__ = [
    "InvestigationV2RunBundleProjector",
    "InvestigationV2RunBundleProjection",
    "RunBundleProjector",
    "V2RunBundleProjector",
    "V2RunBundleProjection",
    "project_task_facts_to_run_bundle",
    "re_export_task_facts_to_run_bundle",
]
