"""investigation_v2 的研究员假设确认深模块。

这个模块把已闭合的 Selection 或 Hypothesis Import 输出投影成稳定的研究员视图，
并把一次完整决定封装为同一个 Open Confirmation Module Run 的原子输出。它不
读取或写入 Harness、MiniChem 或 Run Bundle；上传字节只通过注入的任务级
``ResearchMaterialStore`` 进入内容寻址存储，Working Memory 只保存摘要记录。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Callable, Iterator

from chemistry_multiagent.memory import (
    CASConflict,
    IdempotencyConflict,
    ModuleInvocationClosed,
    ResearchTaskWorkingMemory,
    ModuleInvocationSnapshot,
    PreparedModuleInvocationOutput,
    ScientificRecord,
    ResearchTaskFacts,
)
from chemistry_multiagent.memory.contracts import canonical_json
from chemistry_multiagent.investigation.contracts import (
    require_safe_public_scientific_context,
    require_safe_public_scientific_text,
)
from chemistry_multiagent.investigation.research_material_store import (
    ResearchMaterialStore,
)


CONFIRMATION_REASON_CODE = "awaiting_researcher_confirmation"
CONFIRMATION_MODULE_KIND = "hypothesis-confirmation"
CONFIRMATION_TASK_CONTRACT_VERSION = "investigation_v2.task.v2"
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class HypothesisConfirmationError(ValueError):
    """研究员确认请求或候选科学图不满足确认合同。"""


class HypothesisConfirmationStateConflict(HypothesisConfirmationError):
    """确认视图已经过期，或当前任务不再处于可确认状态。"""


@dataclass(frozen=True, slots=True)
class ResearchMaterialUpload:
    """确认请求中的一个文件流，不保存调用方路径。

    ``filename`` 是模型最终看到的安全 basename，``media_type`` 是调用方声明的
    MIME 类型，``content`` 必须是 bytes-like 或提供 ``read(size)`` 的二进制流。
    上传引用由 ``files`` mapping 的键提供，因此不会混入文件正文或本机路径。
    """

    filename: str
    media_type: str
    content: Any


def _require_text(value: Any, field_name: str) -> str:
    """校验确认协议中的非空文本字段。"""

    if not isinstance(value, str) or not value.strip():
        raise HypothesisConfirmationError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _require_safe_token(value: Any, field_name: str) -> str:
    """校验幂等身份不携带路径或控制字符。"""

    value = _require_text(value, field_name)
    if not _SAFE_TOKEN_RE.fullmatch(value):
        raise HypothesisConfirmationError(f"{field_name} 必须是不透明安全标识")
    return value


def _is_module_kind(value: Any, expected: str) -> bool:
    """只校验 canonical 模块名称，避免活动确认路径接受历史别名。"""

    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("_", "-").replace(" ", "-")
    return normalized == expected


def _entry_strategy(facts: ResearchTaskFacts) -> str:
    """读取 TaskCreated 冻结的 canonical 入口策略。

    旧策略只在 schema-v10 历史任务读取时转换，不会成为新任务的生产值。
    """

    strategy = facts.research_task.policy.get("entry_strategy")
    if strategy in {"standard_hypothesis_generation", "author_hypothesis_import"}:
        return strategy
    if strategy is None:
        raise HypothesisConfirmationStateConflict(
            "TaskCreated.policy.entry_strategy 缺失或未注册"
        )
    raise HypothesisConfirmationStateConflict(
        "TaskCreated.policy.entry_strategy 缺失或未注册"
    )


def _confirmation_enabled(facts: ResearchTaskFacts) -> bool:
    """只为明确升级到 v2 的新 Task 插入 Confirmation 阶段。

    历史任务不进入活动确认服务；它们必须由显式只读迁移器处理，不能因为当前
    代码升级而突然等待研究员确认。
    """

    return facts.research_task.contract_version == CONFIRMATION_TASK_CONTRACT_VERSION


def _record_digest(value: Mapping[str, Any]) -> str:
    """为确认请求生成稳定摘要，供精确重放比较。"""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HypothesisConfirmationView(Mapping[str, Any]):
    """研究员可见的待确认候选视图。

    视图刻意不包含 Module Run、runtime、provider、thread、scope 或路径；候选
    使用 ``hypothesis_candidate_record_id`` 暴露 SelectedHypothesisRecord 的科学
    记录身份，不携带 Module Invocation 或 runtime 私有身份；确认记录仍会保存
    同一个科学记录身份作为来源。
    """

    research_task_id: str
    expected_research_task_sequence: int
    hypothesis_batch_record_id: str
    candidates: tuple[dict[str, Any], ...]
    status: str = "open"
    reason_code: str = CONFIRMATION_REASON_CODE

    def __post_init__(self) -> None:
        """校验并冻结公开视图字段。"""

        _require_text(self.research_task_id, "research_task_id")
        if (
            isinstance(self.expected_research_task_sequence, bool)
            or not isinstance(self.expected_research_task_sequence, int)
            or self.expected_research_task_sequence < 1
        ):
            raise HypothesisConfirmationError(
                "expected_research_task_sequence 必须是正整数"
            )
        _require_text(
            self.hypothesis_batch_record_id,
            "hypothesis_batch_record_id",
        )
        if self.status != "open" or self.reason_code != CONFIRMATION_REASON_CODE:
            raise HypothesisConfirmationError("确认视图状态字段不匹配")
        if not isinstance(self.candidates, Sequence) or isinstance(
            self.candidates, (str, bytes)
        ):
            raise HypothesisConfirmationError("candidates 必须是 object 序列")
        normalized: list[dict[str, Any]] = []
        refs: set[str] = set()
        ranks: set[int] = set()
        for index, candidate in enumerate(self.candidates):
            if not isinstance(candidate, Mapping):
                raise HypothesisConfirmationError(
                    f"candidates[{index}] 必须是 object"
                )
            if set(candidate) != {
                "hypothesis_candidate_record_id",
                "rank",
                "hypothesis",
            }:
                raise HypothesisConfirmationError(
                    "候选视图字段必须恰好为 hypothesis_candidate_record_id、rank、hypothesis"
                )
            candidate_record_id = _require_text(
                candidate["hypothesis_candidate_record_id"],
                f"candidates[{index}].hypothesis_candidate_record_id",
            )
            rank = candidate["rank"]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise HypothesisConfirmationError(
                    f"candidates[{index}].rank 必须是正整数"
                )
            if candidate_record_id in refs or rank in ranks:
                raise HypothesisConfirmationError("候选引用和 rank 必须唯一")
            if not isinstance(candidate["hypothesis"], Mapping) or not candidate[
                "hypothesis"
            ]:
                raise HypothesisConfirmationError(
                    f"candidates[{index}].hypothesis 必须是非空 object"
                )
            refs.add(candidate_record_id)
            ranks.add(rank)
            normalized.append(
                {
                    "hypothesis_candidate_record_id": candidate_record_id,
                    "rank": rank,
                    "hypothesis": deepcopy(dict(candidate["hypothesis"])),
                }
            )
        object.__setattr__(self, "candidates", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        """返回不共享内部 payload 的研究员视图。"""

        return {
            "research_task_id": self.research_task_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "expected_research_task_sequence": self.expected_research_task_sequence,
            "hypothesis_batch_record_id": self.hypothesis_batch_record_id,
            "candidates": deepcopy(list(self.candidates)),
        }

    @property
    def __getitem__(self, key: str) -> Any:
        """兼容只读 mapping 访问。"""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """按公开字段顺序迭代。"""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """返回公开字段数量。"""

        return 6


@dataclass(frozen=True)
class HypothesisConfirmationSubmission:
    """确认输出已提交或已经精确存在时的内部结果。"""

    module: ModuleInvocationSnapshot
    confirmation_id: str
    request_digest: str
    replayed: bool = False


@dataclass(frozen=True)
class _CandidateContext:
    """确认服务内部使用的候选闭包，不向研究员或模型暴露。"""

    source_module: ModuleInvocationSnapshot
    confirmation_module: ModuleInvocationSnapshot
    hypothesis_batch_record_id: str
    selected_records: tuple[ScientificRecord, ...]
    view: HypothesisConfirmationView


class HypothesisConfirmationService:
    """以小接口封装候选读取、请求校验和 Confirmation 原子提交。

    外部调用者只需读取 ``read_view`` 或提交 ``confirm``；候选来源、版本门禁、
    幂等比较和输出记录成组合同均在这里完成。材料字节由注入的任务级 store
    拥有；本模块只把 store 返回的安全 descriptor 转成 Scientific Record。
    """

    def __init__(
        self,
        working_memory: ResearchTaskWorkingMemory,
        material_store: ResearchMaterialStore | None = None,
        *,
        commit_guard: Callable[[], None] | None = None,
    ) -> None:
        """绑定 V2 Working Memory、任务级材料 store 和提交前门禁。"""

        if not isinstance(working_memory, ResearchTaskWorkingMemory):
            raise HypothesisConfirmationError(
                "working_memory 必须是 ResearchTaskWorkingMemory"
            )
        self.working_memory = working_memory
        if material_store is not None and not isinstance(
            material_store, ResearchMaterialStore
        ):
            raise HypothesisConfirmationError(
                "material_store 必须是 ResearchMaterialStore"
            )
        self.material_store = material_store
        if commit_guard is not None and not callable(commit_guard):
            raise HypothesisConfirmationError("commit_guard 必须可调用")
        self.commit_guard = commit_guard

    def read_view(self, research_task_id: str) -> HypothesisConfirmationView:
        """从当前 Open Confirmation Module Run 构造稳定候选视图。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        facts = self.working_memory.read_research_task_facts(research_task_id)
        module = facts.open_module_invocation
        if module is None or not _is_module_kind(
            module.module_kind, CONFIRMATION_MODULE_KIND
        ):
            raise HypothesisConfirmationStateConflict(
                "任务当前没有等待研究员确认的 Open Module Run"
            )
        return self._candidate_context(facts, module).view

    def confirm(
        self,
        research_task_id: str,
        *,
        confirmation_id: str,
        expected_research_task_sequence: int,
        decisions: Sequence[Mapping[str, Any]],
        files: Mapping[str, ResearchMaterialUpload] | None = None,
    ) -> HypothesisConfirmationSubmission:
        """校验完整决定并原子闭合同一个 Open Confirmation Module Run。"""

        research_task_id = _require_text(research_task_id, "research_task_id")
        confirmation_id = _require_safe_token(confirmation_id, "confirmation_id")
        expected_research_task_sequence = self._validate_sequence(
            expected_research_task_sequence,
            "expected_research_task_sequence",
        )
        facts = self.working_memory.read_research_task_facts(research_task_id)

        module = facts.open_module_invocation
        if module is not None and _is_module_kind(
            module.module_kind, CONFIRMATION_MODULE_KIND
        ):
            if facts.research_task_sequence != expected_research_task_sequence:
                raise CASConflict(
                    "确认视图已过期，请重新读取当前候选",
                    expected_version=expected_research_task_sequence,
                    current_version=facts.research_task_sequence,
                )
            return self._commit_open_confirmation(
                research_task_id,
                facts,
                module,
                confirmation_id,
                decisions,
                files,
            )

        # 完成提交后丢失调用方响应时，允许同一 confirmation_id 精确重放；这
        # 一段路径不要求调用方继续携带已经过期的旧 sequence。
        closed_context = self._latest_closed_confirmation(facts)
        if facts.open_module_invocation is None and closed_context is not None:
            _normalized, request_digest = self._normalize_decisions(
                closed_context.view,
                decisions,
                files=files,
            )
            stored = self._stored_request(closed_context)
            if confirmation_id != stored["confirmation_id"]:
                raise IdempotencyConflict(
                    "当前假设批次已经使用另一个 confirmation_id 确认"
                )
            if request_digest != stored["request_digest"]:
                raise IdempotencyConflict("同一 confirmation_id 的确认正文发生漂移")
            return HypothesisConfirmationSubmission(
                module=closed_context.confirmation_module,
                confirmation_id=confirmation_id,
                request_digest=request_digest,
                replayed=True,
            )
        if closed_context is not None and facts.open_module_invocation is not None:
            # Confirmation 已闭合后通常已经进入 Investigation；仍允许精确重放，
            # 但不同身份不能覆盖同一批次的不可变决定。
            _normalized, request_digest = self._normalize_decisions(
                closed_context.view,
                decisions,
                files=files,
            )
            stored = self._stored_request(closed_context)
            if confirmation_id != stored["confirmation_id"]:
                raise IdempotencyConflict(
                    "当前假设批次已经使用另一个 confirmation_id 确认"
                )
            if request_digest != stored["request_digest"]:
                raise IdempotencyConflict("同一 confirmation_id 的确认正文发生漂移")
            return HypothesisConfirmationSubmission(
                module=closed_context.confirmation_module,
                confirmation_id=confirmation_id,
                request_digest=request_digest,
                replayed=True,
            )

        raise HypothesisConfirmationStateConflict(
            "任务当前没有可提交的 Open Confirmation Module Run"
        )

    def _commit_open_confirmation(
        self,
        research_task_id: str,
        facts: ResearchTaskFacts,
        module: ModuleInvocationSnapshot,
        confirmation_id: str,
        decisions: Sequence[Mapping[str, Any]],
        files: Mapping[str, ResearchMaterialUpload] | None,
    ) -> HypothesisConfirmationSubmission:
        """把当前 Open Confirmation 的合法决定提交到同一 Module Run。"""

        context = self._candidate_context(facts, module)
        normalized, request_digest = self._normalize_decisions(
            context.view,
            decisions,
            files=files,
        )
        output = self._build_output(
            research_task_id,
            module,
            context,
            confirmation_id,
            normalized,
        )
        if self.commit_guard is not None:
            # 上传流可能耗时较长。材料进入任务级 CAS 后、科学事实提交前再次
            # 核对 Controller lease，避免失锁 holder 闭合 Confirmation。
            self.commit_guard()
        try:
            committed = self.working_memory.complete_module_invocation(
                research_task_id,
                expected_research_task_sequence=facts.research_task_sequence,
                output_bundle=output,
            )
        except (CASConflict, IdempotencyConflict):
            # 并发调用可能已经提交相同 module_invocation_id；读取事实后只接受精确
            # 的 confirmation_id/正文重放，其他正文保持冲突。
            refreshed = self.working_memory.read_research_task_facts(research_task_id)
            committed_context = self._latest_closed_confirmation(refreshed)
            if committed_context is None:
                raise
            stored = self._stored_request(committed_context)
            if (
                stored["confirmation_id"] != confirmation_id
                or stored["request_digest"] != request_digest
            ):
                raise
            return HypothesisConfirmationSubmission(
                module=committed_context.confirmation_module,
                confirmation_id=confirmation_id,
                request_digest=request_digest,
                replayed=True,
            )
        return HypothesisConfirmationSubmission(
            module=committed,
            confirmation_id=confirmation_id,
            request_digest=request_digest,
        )

    @staticmethod
    def _validate_sequence(value: Any, field_name: str) -> int:
        """校验乐观并发序号。"""

        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise HypothesisConfirmationError(f"{field_name} 必须是正整数")
        return value

    def _latest_closed_confirmation(
        self,
        facts: ResearchTaskFacts,
    ) -> _CandidateContext | None:
        """返回当前批次最近一次已闭合确认模块及其候选闭包。"""

        for module in reversed(facts.module_invocations):
            if not _is_module_kind(module.module_kind, CONFIRMATION_MODULE_KIND):
                continue
            if not isinstance(module.lifecycle, ModuleInvocationClosed):
                continue
            return self._candidate_context(facts, module)
        return None

    def _candidate_context(
        self,
        facts: ResearchTaskFacts,
        confirmation_module: ModuleInvocationSnapshot,
    ) -> _CandidateContext:
        """核对 Confirmation 输入与最新 Selection/Import 候选闭包。"""

        strategy = _entry_strategy(facts)
        source_kind = (
            "hypothesis-import"
            if strategy == "author_hypothesis_import"
            else "hypothesis-selection"
        )
        source_module = next(
            (
                module
                for module in reversed(facts.module_invocations)
                if module.module_invocation_id != confirmation_module.module_invocation_id
                and _is_module_kind(module.module_kind, source_kind)
                and isinstance(module.lifecycle, ModuleInvocationClosed)
            ),
            None,
        )
        if source_module is None:
            raise HypothesisConfirmationStateConflict(
                "Confirmation 缺少已闭合 Selection/Import 来源"
            )
        source_refs = tuple(
            source_module.lifecycle.outcome["output_scientific_record_ids"]
        )
        source_records = self.working_memory.read_research_task_scientific_records(
            facts.research_task_id,
            source_refs,
        )
        selected = tuple(
            record
            for record in source_records
            if record.scientific_record_type == "SelectedHypothesisRecord"
        )
        selected = tuple(sorted(selected, key=lambda record: record.payload["rank"]))
        expected_count = (
            1 <= len(selected) <= 5
            if source_kind == "hypothesis-import"
            else len(selected) == 5
        )
        if not expected_count:
            raise HypothesisConfirmationStateConflict(
                "确认候选数量不满足标准五条或作者一至五条合同"
            )
        if [record.payload["rank"] for record in selected] != list(
            range(1, len(selected) + 1)
        ):
            raise HypothesisConfirmationStateConflict(
                "确认候选 rank 必须从 1 连续排列"
            )
        expected_refs = tuple(record.scientific_record_id for record in selected)
        if tuple(confirmation_module.input_scientific_record_ids) != expected_refs:
            raise HypothesisConfirmationStateConflict(
                "Confirmation Module Run 的候选引用与最新来源不一致"
            )
        batch_candidates: set[str] | None = None
        all_records = self.working_memory.read_research_task_scientific_records(
            facts.research_task_id
        )
        by_id = {record.scientific_record_id: record for record in all_records}
        for record in selected:
            refs = {
                ref
                for ref in record.source_scientific_record_ids
                if by_id.get(ref) is not None
                and by_id[ref].scientific_record_type == "HypothesisBatchRecord"
            }
            batch_candidates = refs if batch_candidates is None else batch_candidates & refs
        if batch_candidates is None or len(batch_candidates) != 1:
            raise HypothesisConfirmationStateConflict(
                "候选无法追溯到唯一 HypothesisBatchRecord"
            )
        hypothesis_batch_record_id = next(iter(batch_candidates))
        candidate_record_ids = [record.scientific_record_id for record in selected]
        if len(set(candidate_record_ids)) != len(candidate_record_ids):
            raise HypothesisConfirmationStateConflict(
                "候选 scientific_record_id 必须在当前批次内唯一"
            )
        view = HypothesisConfirmationView(
            research_task_id=facts.research_task_id,
            expected_research_task_sequence=facts.research_task_sequence,
            hypothesis_batch_record_id=hypothesis_batch_record_id,
            candidates=tuple(
                {
                    "hypothesis_candidate_record_id": record.scientific_record_id,
                    "rank": record.payload["rank"],
                    "hypothesis": deepcopy(record.payload["hypothesis"]),
                }
                for record in selected
            ),
        )
        return _CandidateContext(
            source_module=source_module,
            confirmation_module=confirmation_module,
            hypothesis_batch_record_id=hypothesis_batch_record_id,
            selected_records=selected,
            view=view,
        )

    def _normalize_decisions(
        self,
        view: HypothesisConfirmationView,
        decisions: Sequence[Mapping[str, Any]],
        *,
        files: Mapping[str, ResearchMaterialUpload] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """严格校验决定全集、摄取被引用文件并按候选顺序规范化。"""

        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise HypothesisConfirmationError("decisions 必须是 object 序列")
        candidate_by_record_id = {
            item["hypothesis_candidate_record_id"]: item for item in view.candidates
        }
        if len(decisions) != len(candidate_by_record_id):
            raise HypothesisConfirmationError(
                "decisions 必须恰好覆盖当前候选全集"
            )
        by_ref: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(decisions):
            if not isinstance(raw, Mapping):
                raise HypothesisConfirmationError(f"decisions[{index}] 必须是 object")
            candidate_record_id = _require_text(
                raw.get("hypothesis_candidate_record_id"),
                f"decisions[{index}].hypothesis_candidate_record_id",
            )
            if candidate_record_id in by_ref:
                raise HypothesisConfirmationError("decisions 不能重复候选")
            if candidate_record_id not in candidate_by_record_id:
                raise HypothesisConfirmationError(
                    "decisions 含未知 hypothesis_candidate_record_id"
                )
            decision = raw.get("decision")
            if decision not in {"execute", "cancel"}:
                raise HypothesisConfirmationError(
                    f"decisions[{index}].decision 只能是 execute 或 cancel"
                )
            if decision == "cancel":
                if set(raw) != {"hypothesis_candidate_record_id", "decision"}:
                    raise HypothesisConfirmationError(
                        "cancel 决定不能携带 hypothesis、guidance 或 materials"
                    )
                by_ref[candidate_record_id] = {
                    "hypothesis_candidate_record_id": candidate_record_id,
                    "decision": "cancel",
                }
                continue
            allowed = {
                "hypothesis_candidate_record_id",
                "decision",
                "hypothesis",
                "guidance",
                "materials",
            }
            if not set(raw).issubset(allowed) or "hypothesis" not in raw:
                raise HypothesisConfirmationError(
                    "execute 决定必须提供完整 hypothesis，且不能携带未知字段"
                )
            hypothesis = raw["hypothesis"]
            if not isinstance(hypothesis, Mapping) or not hypothesis:
                raise HypothesisConfirmationError(
                    "execute.hypothesis 必须是非空 object"
                )
            try:
                hypothesis = require_safe_public_scientific_context(
                    hypothesis,
                    label=f"decisions[{index}].hypothesis",
                )
            except (TypeError, ValueError) as exc:
                raise HypothesisConfirmationError(str(exc)) from exc
            if not isinstance(hypothesis, Mapping) or not hypothesis:  # pragma: no cover - helper 保持 object
                raise HypothesisConfirmationError(
                    "execute.hypothesis 必须是非空 object"
                )
            guidance = raw.get("guidance")
            if guidance is not None:
                try:
                    guidance = require_safe_public_scientific_text(
                        guidance,
                        label=f"decisions[{index}].guidance",
                    )
                except (TypeError, ValueError) as exc:
                    raise HypothesisConfirmationError(str(exc)) from exc
                guidance = guidance.strip()
            raw_materials = raw.get("materials", [])
            if isinstance(raw_materials, (str, bytes)) or not isinstance(
                raw_materials, list
            ):
                raise HypothesisConfirmationError(
                    "execute.materials 必须是 object 数组"
                )
            materials: list[dict[str, Any]] = []
            seen_upload_refs: set[str] = set()
            for material_index, material in enumerate(raw_materials):
                label = f"decisions[{index}].materials[{material_index}]"
                if not isinstance(material, Mapping):
                    raise HypothesisConfirmationError(f"{label} 必须是 object")
                if not {"upload_ref"}.issubset(material) or not set(
                    material
                ).issubset({"upload_ref", "description"}):
                    raise HypothesisConfirmationError(
                        f"{label} 字段必须为 upload_ref 和可选 description"
                    )
                upload_ref = _require_safe_token(
                    material.get("upload_ref"), f"{label}.upload_ref"
                )
                if upload_ref in seen_upload_refs:
                    raise HypothesisConfirmationError(
                        "同一 execute 决定不能重复引用上传材料"
                    )
                seen_upload_refs.add(upload_ref)
                description = material.get("description")
                if description is not None:
                    try:
                        description = require_safe_public_scientific_text(
                            description,
                            label=f"{label}.description",
                        ).strip()
                    except (TypeError, ValueError) as exc:
                        raise HypothesisConfirmationError(str(exc)) from exc
                materials.append(
                    {
                        "upload_ref": upload_ref,
                        "description": description,
                    }
                )
            by_ref[candidate_record_id] = {
                "hypothesis_candidate_record_id": candidate_record_id,
                "decision": "execute",
                "hypothesis": deepcopy(dict(hypothesis)),
                "guidance": guidance,
                "materials": materials,
            }
        if set(by_ref) != set(candidate_by_record_id):
            raise HypothesisConfirmationError("decisions 必须覆盖所有候选")
        execute_count = sum(
            item["decision"] == "execute" for item in by_ref.values()
        )
        if not 1 <= execute_count <= 5:
            raise HypothesisConfirmationError("execute 候选数必须在 1 到 5 之间")
        normalized: list[dict[str, Any]] = []
        execution_rank = 0
        for candidate in view.candidates:
            item = by_ref[candidate["hypothesis_candidate_record_id"]]
            if item["decision"] == "execute":
                execution_rank += 1
                item = {**item, "execution_rank": execution_rank}
            normalized.append(item)
        descriptors = self._ingest_referenced_files(normalized, files)
        for item in normalized:
            if item["decision"] != "execute":
                continue
            item["materials"] = [
                deepcopy(descriptors[material["upload_ref"]])
                for material in item["materials"]
            ]
        digest = _record_digest({"decisions": normalized})
        return normalized, digest

    def _ingest_referenced_files(
        self,
        normalized_decisions: Sequence[Mapping[str, Any]],
        files: Mapping[str, ResearchMaterialUpload] | None,
    ) -> dict[str, dict[str, Any]]:
        """只摄取决定实际引用的文件，并拒绝遗漏、额外文件或描述漂移。"""

        description_by_ref: dict[str, str | None] = {}
        ordered_refs: list[str] = []
        for decision in normalized_decisions:
            if decision["decision"] != "execute":
                continue
            for material in decision["materials"]:
                upload_ref = material["upload_ref"]
                description = material["description"]
                if upload_ref in description_by_ref:
                    if description_by_ref[upload_ref] != description:
                        raise HypothesisConfirmationError(
                            "同一 upload_ref 在不同假设中的 description 必须一致"
                        )
                    continue
                description_by_ref[upload_ref] = description
                ordered_refs.append(upload_ref)

        if files is None:
            normalized_files: Mapping[str, ResearchMaterialUpload] = {}
        elif not isinstance(files, Mapping):
            raise HypothesisConfirmationError("files 必须是 upload_ref 到文件流的 mapping")
        else:
            normalized_files = files
        files_by_ref: dict[str, ResearchMaterialUpload] = {}
        for raw_ref, upload in normalized_files.items():
            safe_ref = _require_safe_token(raw_ref, "files.<upload_ref>")
            if safe_ref in files_by_ref:
                raise HypothesisConfirmationError(
                    "files 不能用不同原始键重复同一 upload_ref"
                )
            files_by_ref[safe_ref] = upload
        file_refs = set(files_by_ref)
        if file_refs != set(ordered_refs):
            missing = sorted(set(ordered_refs).difference(file_refs))
            extra = sorted(file_refs.difference(ordered_refs))
            details: list[str] = []
            if missing:
                details.append("缺少 " + ", ".join(missing))
            if extra:
                details.append("含未使用文件 " + ", ".join(extra))
            raise HypothesisConfirmationError("files 与 materials 引用不一致：" + "；".join(details))
        if not ordered_refs:
            return {}
        if self.material_store is None:
            raise HypothesisConfirmationError("当前任务没有配置研究材料存储")

        descriptors: dict[str, dict[str, Any]] = {}
        for upload_ref in ordered_refs:
            upload = files_by_ref[upload_ref]
            if not isinstance(upload, ResearchMaterialUpload):
                raise HypothesisConfirmationError(
                    f"files[{upload_ref}] 必须是 ResearchMaterialUpload"
                )
            descriptor = self.material_store.ingest_material(
                upload.filename,
                upload.media_type,
                upload.content,
                description=description_by_ref[upload_ref],
            )
            descriptors[upload_ref] = {
                "upload_ref": upload_ref,
                "filename": descriptor["filename"],
                "media_type": descriptor["media_type"],
                "size_bytes": descriptor["size_bytes"],
                "sha256": descriptor["sha256"],
                "content_ref": descriptor["content_ref"],
                "description": descriptor.get("description"),
            }
        return descriptors

    def _build_output(
        self,
        research_task_id: str,
        confirmation_module: ModuleInvocationSnapshot,
        context: _CandidateContext,
        confirmation_id: str,
        normalized: Sequence[Mapping[str, Any]],
    ) -> PreparedModuleInvocationOutput:
        """构造一条 Decision、材料摘要和 N 条 Input 的密封输出包。"""

        decision_id = f"{confirmation_module.module_invocation_id}:decision"
        input_items = [
            item for item in normalized if item["decision"] == "execute"
        ]
        input_ids = [
            f"{confirmation_module.module_invocation_id}:input:{item['execution_rank']}"
            for item in input_items
        ]
        decision_record = ScientificRecord.from_payload(
            scientific_record_id=decision_id,
            research_task_id=research_task_id,
            scientific_record_type="HypothesisExecutionDecisionRecord",
            scientific_record_schema_version="hypothesis-confirmation.v2",
            producer_module_invocation_id=confirmation_module.module_invocation_id,
            source_scientific_record_ids=(context.hypothesis_batch_record_id,),
            payload={
                "source_hypothesis_batch_record_id": context.hypothesis_batch_record_id,
                "confirmation_id": confirmation_id,
                "decisions": [
                    {
                        "hypothesis_candidate_record_id": item[
                            "hypothesis_candidate_record_id"
                        ],
                        "decision": item["decision"],
                        **(
                            {"execution_rank": item["execution_rank"]}
                            if item["decision"] == "execute"
                            else {}
                        ),
                    }
                    for item in normalized
                ],
                "execution_input_scientific_record_ids": input_ids,
            },
        )
        selected_by_candidate = {
            record.scientific_record_id: record
            for record in context.selected_records
        }
        material_items: dict[str, Mapping[str, Any]] = {}
        for item in input_items:
            for material in item["materials"]:
                upload_ref = material["upload_ref"]
                existing = material_items.get(upload_ref)
                if existing is not None and existing != material:
                    raise HypothesisConfirmationError(
                        "同一 upload_ref 的材料 descriptor 发生漂移"
                    )
                material_items[upload_ref] = material
        material_record_ids = {
            upload_ref: f"{confirmation_module.module_invocation_id}:material:{index}"
            for index, upload_ref in enumerate(material_items, start=1)
        }
        material_records = [
            ScientificRecord.from_payload(
                scientific_record_id=material_record_ids[upload_ref],
                research_task_id=research_task_id,
                scientific_record_type="ResearchMaterialRecord",
                scientific_record_schema_version="hypothesis-confirmation.v2",
                producer_module_invocation_id=confirmation_module.module_invocation_id,
                source_scientific_record_ids=(),
                payload={
                    "material_id": upload_ref,
                    "content_ref": material["content_ref"],
                    "sha256": material["sha256"],
                    "size_bytes": material["size_bytes"],
                    "filename": material["filename"],
                    "media_type": material["media_type"],
                    "description": material.get("description"),
                },
            )
            for upload_ref, material in material_items.items()
        ]
        input_records = [
            ScientificRecord.from_payload(
                scientific_record_id=input_id,
                research_task_id=research_task_id,
                scientific_record_type="HypothesisExecutionInputRecord",
                scientific_record_schema_version="hypothesis-confirmation.v2",
                producer_module_invocation_id=confirmation_module.module_invocation_id,
                source_scientific_record_ids=(
                    decision_id,
                    selected_by_candidate[
                        item["hypothesis_candidate_record_id"]
                    ].scientific_record_id,
                    *(
                        material_record_ids[material["upload_ref"]]
                        for material in item["materials"]
                    ),
                ),
                payload={
                    "source_selected_hypothesis_record_id": selected_by_candidate[
                        item["hypothesis_candidate_record_id"]
                    ].scientific_record_id,
                    "execution_rank": item["execution_rank"],
                    "hypothesis": deepcopy(item["hypothesis"]),
                    "guidance": item.get("guidance"),
                    "material_upload_refs": [
                        material["upload_ref"] for material in item["materials"]
                    ],
                },
            )
            for input_id, item in zip(input_ids, input_items)
        ]
        return PreparedModuleInvocationOutput.from_records(
            module_invocation_id=confirmation_module.module_invocation_id,
            module_input_digest=confirmation_module.module_input_digest,
            output_scientific_records=(decision_record, *material_records, *input_records),
        )

    def _stored_request(self, context: _CandidateContext) -> dict[str, Any]:
        """从已保存的 Decision/Input 记录重建幂等比较摘要。"""

        if not isinstance(
            context.confirmation_module.lifecycle, ModuleInvocationClosed
        ):
            raise HypothesisConfirmationStateConflict("确认模块尚未闭合")
        refs = tuple(
            context.confirmation_module.lifecycle.outcome[
                "output_scientific_record_ids"
            ]
        )
        records = self.working_memory.read_research_task_scientific_records(
            context.view.research_task_id,
            refs,
        )
        decisions = next(
            (
                record
                for record in records
                if record.scientific_record_type
                == "HypothesisExecutionDecisionRecord"
            ),
            None,
        )
        inputs = {
            record.payload["source_selected_hypothesis_record_id"]: record
            for record in records
            if record.scientific_record_type == "HypothesisExecutionInputRecord"
        }
        materials = {
            record.payload["material_id"]: record
            for record in records
            if record.scientific_record_type == "ResearchMaterialRecord"
        }
        if decisions is None:
            raise HypothesisConfirmationStateConflict(
                "已闭合确认缺少 Decision record"
            )
        normalized: list[dict[str, Any]] = []
        selected_by_candidate = {
            record.scientific_record_id: record.scientific_record_id
            for record in context.selected_records
        }
        for item in decisions.payload["decisions"]:
            candidate_record_id = item["hypothesis_candidate_record_id"]
            if item["decision"] == "cancel":
                normalized.append(
                    {
                        "hypothesis_candidate_record_id": candidate_record_id,
                        "decision": "cancel",
                    }
                )
                continue
            selected_record_id = selected_by_candidate.get(candidate_record_id)
            input_record = inputs.get(selected_record_id)
            if input_record is None:
                raise HypothesisConfirmationStateConflict(
                    "已闭合确认缺少 execute 对应的 Input record"
                )
            resolved_materials: list[dict[str, Any]] = []
            for material_ref in input_record.payload["material_upload_refs"]:
                material_record = materials.get(material_ref)
                if material_record is None:
                    raise HypothesisConfirmationStateConflict(
                        "已闭合确认缺少 Input 引用的 Material record"
                    )
                resolved_materials.append(
                    {
                        "upload_ref": material_ref,
                        "filename": material_record.payload["filename"],
                        "media_type": material_record.payload["media_type"],
                        "size_bytes": material_record.payload["size_bytes"],
                        "sha256": material_record.payload["sha256"],
                        "content_ref": material_record.payload["content_ref"],
                        "description": material_record.payload.get("description"),
                    }
                )
            normalized.append(
                {
                    "hypothesis_candidate_record_id": candidate_record_id,
                    "decision": "execute",
                    "hypothesis": deepcopy(input_record.payload["hypothesis"]),
                    "guidance": input_record.payload.get("guidance"),
                    "materials": resolved_materials,
                    "execution_rank": item["execution_rank"],
                }
            )
        return {
            "confirmation_id": decisions.payload["confirmation_id"],
            "request_digest": _record_digest({"decisions": normalized}),
        }


__all__ = [
    "CONFIRMATION_MODULE_KIND",
    "CONFIRMATION_REASON_CODE",
    "CONFIRMATION_TASK_CONTRACT_VERSION",
    "HypothesisConfirmationError",
    "HypothesisConfirmationService",
    "HypothesisConfirmationStateConflict",
    "HypothesisConfirmationSubmission",
    "HypothesisConfirmationView",
    "ResearchMaterialUpload",
    "_confirmation_enabled",
]
