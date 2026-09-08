"""investigation_v2 的默认模块 adapter。

每个 adapter 只接收 ``ModuleInvocationRequest`` 和冻结的任务输入。Retrieval、Query、Hypothesis
和 Selection 使用这里的轻量模块；默认 Investigation/Review 则在
``investigation_v2_native_adapters`` 中分别接到 Context + native Harness 与
``V2TaskReviewRuntime``。不把下游私有运行身份提升为 Controller 状态。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from chemistry_multiagent.agents.hypothesis_agent import HypothesisProtocolError
from chemistry_multiagent.investigation.recovery_contract import (
    TemporarilyUnavailable,
    RecoveryEvidenceMissing,
    VerifiedIntegrityConflict,
)
from chemistry_multiagent.investigation.research_material_store import (
    ResearchMaterialStore,
)
from chemistry_multiagent.memory import (
    ResearchTaskWorkingMemory,
    PreparedModuleInvocationOutput,
    ScientificRecord,
)
from chemistry_multiagent.utils.workflow_contract import (
    project_minichem_chemical_capability_projection,
    project_minichem_stage_tool_projection,
)

from .investigation_v2_module_controller import (
    HarnessModuleRunner,
    ModuleCompleted,
    ModuleFailed,
    ModuleInvocationRequest,
    ModuleSuspended,
)


class _AdapterEvidenceMissing(RuntimeError):
    """Harness adapter 缺少可继续恢复的公开证据。"""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class _AdapterIntegrityConflict(RuntimeError):
    """Harness adapter 核实到两个公开事实发生冲突。"""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _payload_digest(value: Mapping[str, Any]) -> str:
    """计算输入值摘要，供记录中的公开内容寻址引用使用。"""

    return hashlib.sha256(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _record(
    request: ModuleInvocationRequest,
    record_type: str,
    suffix: str,
    payload: Mapping[str, Any],
    source_record_refs: Sequence[str] = (),
    *,
    schema_version: str | int = 1,
) -> ScientificRecord:
    """构造与当前模块绑定的 Scientific Record。"""

    return ScientificRecord.from_payload(
        scientific_record_id=f"{request.module_invocation_id}:{suffix}",
        research_task_id=request.research_task_id,
        scientific_record_type=record_type,
        scientific_record_schema_version=schema_version,
        producer_module_invocation_id=request.module_invocation_id,
        source_scientific_record_ids=tuple(source_record_refs),
        payload=dict(payload),
    )


def _completed(request: ModuleInvocationRequest, records: Sequence[ScientificRecord]) -> ModuleCompleted:
    """把记录构造成 V2 模块成功结果。"""

    return ModuleCompleted(
        PreparedModuleInvocationOutput.from_records(
            module_invocation_id=request.module_invocation_id,
            module_input_digest=request.module_input_digest,
            output_scientific_records=tuple(records),
        )
    )


def _validate_generated_hypothesis_pool(
    result: Mapping[str, Any],
    *,
    num_queries: int,
    hypotheses_per_query: int,
) -> tuple[list[str], list[Mapping[str, Any]]] | None:
    """核对首批生成严格覆盖冻结的 Query 数量和每组候选数量。"""

    queries = result.get("queries")
    groups = result.get("hypotheses_by_query")
    if (
        not isinstance(queries, list)
        or len(queries) != num_queries
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or not isinstance(groups, list)
        or len(groups) != num_queries
    ):
        return None
    normalized_groups: list[Mapping[str, Any]] = []
    for index, group in enumerate(groups):
        if (
            not isinstance(group, Mapping)
            or group.get("query") != queries[index]
            or not isinstance(group.get("hypotheses"), list)
            or len(group["hypotheses"]) != hypotheses_per_query
        ):
            return None
        normalized_groups.append(group)
    return queries, normalized_groups


class _AdapterBase:
    """保存模块所需的冻结任务上下文。"""

    def __init__(
        self,
        *,
        controller: Any,
        task_root: Path,
        working_memory: ResearchTaskWorkingMemory,
        task_input: Mapping[str, Any],
        policy: Mapping[str, Any],
        dependencies: Mapping[str, Any],
        task_config: Mapping[str, Any],
    ) -> None:
        self.controller = controller
        self.task_root = Path(task_root)
        self.working_memory = working_memory
        self.task_input = copy.deepcopy(dict(task_input))
        self.policy = copy.deepcopy(dict(policy))
        self.dependencies = copy.deepcopy(dict(dependencies))
        self.task_config = copy.deepcopy(dict(task_config))


class RetrievalModuleAdapter(_AdapterBase):
    """调用既有 RetrievalAgent 并保存 V2 RetrievalRecord。"""

    def _run(self, request: ModuleInvocationRequest) -> Any:
        retrieval = self.task_input.get("retrieval")
        if not isinstance(retrieval, Mapping):
            return ModuleFailed("retrieval_input_missing")
        request_payload = {
            "scientific_question": self.task_input["scientific_question"],
            "pdf_dir": retrieval.get("pdf_dir"),
            "index_dir": retrieval.get("index_dir"),
            "search_papers": retrieval.get("search_papers"),
            "paper_search": copy.deepcopy(dict(retrieval.get("paper_search", {}))),
        }
        # 只有调用方明确报告的外部暂时不可用才能暂停。身份、合同和编程错误
        # 必须传播到 V2 Controller，不能被笼统转换成“稍后重试”。
        result = self.controller.run_retrieval_phase(request_payload)
        if not isinstance(result, Mapping):
            return ModuleFailed("retrieval_result_invalid")
        if result.get("error") is not None:
            return ModuleSuspended(
                "retrieval_incomplete",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="retrieval_incomplete"
                ),
            )
        summary = result.get("literature_review") or result.get("answer") or "retrieval completed"
        if not isinstance(summary, str) or not summary.strip():
            summary = "retrieval completed"
        citations = result.get("retrieved_papers", [])
        if not isinstance(citations, list):
            citations = []
        citations = [str(item) for item in citations]
        corpus_ref = "corpus-" + _payload_digest(
            {
                "question": str(self.task_input["scientific_question"]),
                "citations": citations,
                "summary": summary,
            }
        )[:32]
        return _completed(
            request,
            (
                _record(
                    request,
                    "RetrievalRecord",
                    "retrieval",
                    {
                        "summary": summary,
                        "citations": citations,
                        "corpus_snapshot_ref": corpus_ref,
                    },
                ),
            ),
        )

    def start(self, request: ModuleInvocationRequest) -> Any:
        """启动 Retrieval。"""

        return self._run(request)

    def resume(self, request: ModuleInvocationRequest) -> Any:
        """恢复 Retrieval；输入和任务身份来自同一 V2 Module Run。"""

        return self._run(request)


class QueryModuleAdapter(_AdapterBase):
    """在 Retrieval 和 Hypothesis 之间建立公开 QueryRecord。"""

    def _run(self, request: ModuleInvocationRequest) -> ModuleCompleted | ModuleFailed:
        question = self.task_input.get("scientific_question")
        if not isinstance(question, str) or not question.strip():
            return ModuleFailed("query_input_missing")
        query_angles = self.task_input.get("query_angles")
        if not isinstance(query_angles, list) or not query_angles:
            query_angles = [question.strip()]
        angles = [item.strip() for item in query_angles if isinstance(item, str) and item.strip()]
        if not angles:
            angles = [question.strip()]
        return _completed(
            request,
            (
                _record(
                    request,
                    "QueryRecord",
                    "query",
                    {
                        "query": angles[0],
                        "research_angle": "retrieval-derived scientific angle",
                        "constraints": {
                            "query_angles": angles,
                            "num_queries": self.task_config.get("num_queries", len(angles)),
                        },
                    },
                    request.input_scientific_record_ids,
                ),
            ),
        )

    def start(self, request: ModuleInvocationRequest) -> ModuleCompleted | ModuleFailed:
        """创建 QueryRecord。"""

        return self._run(request)

    def resume(self, request: ModuleInvocationRequest) -> ModuleCompleted | ModuleFailed:
        """重放相同 QueryRecord。"""

        return self._run(request)


class HypothesisModuleAdapter(_AdapterBase):
    """调用既有 HypothesisAgent 的 V2 adapter。"""

    _OPERATION_DIRECTORY = PurePosixPath(
        "working_memory", "hypothesis_generation_inputs"
    )

    def _read_transport_recovery(
        self,
        request: ModuleInvocationRequest,
    ) -> tuple[str, str] | None:
        """读取当前 Module Run 的连续失败清单并返回下一次恢复身份。

        恢复只查看由当前 ``module_invocation_id`` 确定的 operation 目录及连续的
        ``transport-retry-N`` 目录。每层必须只有一个 finalized manifest，且首个
        失败 stage 必须确认为 transport；目录缺失表示恢复链结束。返回新的
        operation identity 和上一层 manifest 绝对路径。身份、链路或失败类别漂移
        时抛出 ``RuntimeError``，不会扫描其他任务或猜测可复用结果。
        """

        operation_root = self.task_root / self._OPERATION_DIRECTORY
        prior_manifest_path: Path | None = None
        retry_index = 0
        while True:
            operation_id = (
                request.module_invocation_id
                if retry_index == 0
                else f"{request.module_invocation_id}-transport-retry-{retry_index}"
            )
            operation_dir = operation_root / operation_id
            if not operation_dir.exists():
                break
            if not operation_dir.is_dir() or operation_dir.is_symlink():
                raise RuntimeError("Hypothesis operation 恢复目录非法")
            manifests = sorted(operation_dir.glob("manifest_*.json"))
            if len(manifests) != 1:
                raise RuntimeError("Hypothesis operation 缺少唯一 finalized manifest")
            manifest_path = manifests[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, Mapping)
                or manifest.get("schema_version") != 1
                or manifest.get("operation") != "hypothesis_generation"
                or manifest.get("operation_instance_id") != operation_id
            ):
                raise RuntimeError("Hypothesis operation manifest 身份非法")
            expected_prior = (
                None
                if prior_manifest_path is None
                else prior_manifest_path.relative_to(self.task_root).as_posix()
            )
            prior_ref = manifest.get("prior_manifest_ref")
            actual_prior = (
                prior_ref.get("relative_path")
                if isinstance(prior_ref, Mapping)
                else None
            )
            if actual_prior != expected_prior:
                raise RuntimeError("Hypothesis operation manifest 恢复链漂移")
            stages = manifest.get("stages")
            failed_stages = (
                [stage for stage in stages if stage.get("status") == "failed"]
                if isinstance(stages, list)
                and all(isinstance(stage, Mapping) for stage in stages)
                else []
            )
            if not failed_stages:
                raise RuntimeError("Hypothesis operation manifest 没有失败 stage")
            first_failed = min(
                failed_stages,
                key=lambda stage: int(stage.get("stage_index", 0)),
            )
            attempts = first_failed.get("attempts")
            if (
                not isinstance(attempts, list)
                or not attempts
                or not isinstance(attempts[-1], Mapping)
                or attempts[-1].get("validation_status") != "transport_failed"
            ):
                raise RuntimeError("Hypothesis operation 不是可恢复的 transport 失败")
            prior_manifest_path = manifest_path.resolve(strict=True)
            retry_index += 1

        if prior_manifest_path is None:
            return None
        return (
            f"{request.module_invocation_id}-transport-retry-{retry_index}",
            str(prior_manifest_path),
        )

    def _load_catalog_snapshot(self) -> Mapping[str, Any]:
        """读取任务冻结目录或安装包静态能力描述，供模型生成能力投影。"""

        catalog = self.dependencies.get("catalog")
        if isinstance(catalog, Mapping):
            snapshot = catalog.get("snapshot")
            if isinstance(snapshot, Mapping):
                return copy.deepcopy(dict(snapshot))
        snapshot = self.dependencies.get("catalog_snapshot")
        if isinstance(snapshot, Mapping):
            return copy.deepcopy(dict(snapshot))

        catalog_root = self.task_root / "internal" / "catalog"
        candidates = []
        if catalog_root.exists():
            if not catalog_root.is_dir() or catalog_root.is_symlink():
                raise RuntimeError("本任务的 MiniChem Catalog 目录非法")
            candidates = [
                item
                for item in catalog_root.iterdir()
                if item.is_dir() and not item.is_symlink()
            ]
        if len(candidates) > 1:
            raise RuntimeError("本任务存在多个冻结 MiniChem Catalog")
        if candidates:
            from chemistry_multiagent.tools.minichem_catalog_freeze import (
                load_frozen_minichem_catalog_startup,
            )

            frozen = load_frozen_minichem_catalog_startup(
                task_root=self.task_root,
                catalog_hash=candidates[0].name,
            )
            return copy.deepcopy(dict(frozen.snapshot))

        # 候选生成阶段在已有任务级冻结包时继续复用该包；新任务才读取安装包提供
        # 的静态能力描述。这里不能创建 9011 adapter，也不能把服务的瞬时健康状态
        # 写成任务准入；真正的 Catalog 冻结延迟到确认后的首次 Investigation 外部
        # 副作用之前。
        from minichem_toolbox.catalog import catalog_snapshot

        return copy.deepcopy(dict(catalog_snapshot(include_health=False)))

    def _run(self, request: ModuleInvocationRequest, *, resume: bool) -> Any:
        agent = getattr(self.controller, "hypothesis_agent", None)
        if agent is None or not callable(getattr(agent, "run_hypothesis_operation", None)):
            return ModuleFailed("hypothesis_agent_unavailable")
        recovery = self._read_transport_recovery(request) if resume else None
        if resume and recovery is None:
            return ModuleSuspended(
                "hypothesis_recovery_evidence_missing",
                recovery_verification=RecoveryEvidenceMissing(
                    reason_code="hypothesis_recovery_evidence_missing"
                ),
            )
        operation_instance_id, prior_manifest_path = (
            recovery if recovery is not None else (request.module_invocation_id, None)
        )
        queries: list[str] = [str(self.task_input["scientific_question"])]
        query_records = self.working_memory.read_scientific_records(
            request.research_task_id, request.input_scientific_record_ids
        )
        for record in query_records:
            if record.scientific_record_type == "QueryRecord":
                constraints = record.payload.get("constraints", {})
                if isinstance(constraints, Mapping):
                    values = constraints.get("query_angles")
                    if isinstance(values, list) and values:
                        queries = [
                            item for item in values if isinstance(item, str) and item.strip()
                        ] or queries
        num_queries = int(self.task_config.get("num_queries", len(queries)))
        num_queries = max(1, num_queries)
        if len(queries) < num_queries:
            queries.extend(
                f"{self.task_input['scientific_question']} (angle {index})"
                for index in range(len(queries) + 1, num_queries + 1)
            )
        queries = queries[:num_queries]
        hypotheses_per_query = int(self.task_config.get("num_hypotheses_per_query", 3))
        top_n = int(self.task_config.get("top_n_hypotheses", 5))
        hypothesis_ids = [
            [
                f"{request.research_task_id}:hypothesis:{query_index}:{candidate_index}"
                for candidate_index in range(1, hypotheses_per_query + 1)
            ]
            for query_index in range(1, num_queries + 1)
        ]
        candidate_count = num_queries * hypotheses_per_query
        merge_ids = [
            f"{request.research_task_id}:hypothesis:merge:{index}"
            for index in range(1, max(1, candidate_count * (candidate_count - 1) // 2 + 1))
        ]
        retrieval_record = self.working_memory.read_scientific_records(
            request.research_task_id,
            request.input_scientific_record_ids,
        )
        retrieval_payload = {
            "keywords": [],
            "literature_review": "",
            "mechanistic_clues": [],
            "limitations": [],
            "chemistry_context": self.task_input.get(
                "supplementary_scientific_input", {}
            ),
        }
        if retrieval_record:
            retrieval_payload["literature_review"] = retrieval_record[0].payload.get(
                "summary", ""
            )
        catalog_snapshot = self._load_catalog_snapshot()
        stage_tool_projection = project_minichem_stage_tool_projection(catalog_snapshot)
        chemical_capability_projection = project_minichem_chemical_capability_projection(
            catalog_snapshot
        )
        try:
            operation = agent.run_hypothesis_operation(
                task_root=str(self.task_root.resolve()),
                operation_instance_id=operation_instance_id,
                scientific_question=str(self.task_input["scientific_question"]),
                stage_tool_projection=stage_tool_projection,
                chemical_capability_projection=chemical_capability_projection,
                # 首批 Query 由正式 operation 的 query stage 生成并写入 manifest；
                # 这里的 QueryRecord 仍作为 V2 科学图谱前缀保留，但不能把它伪装成
                # operation 已冻结的首批角度。
                initial_query_angles=None,
                initial_retrieval_phase=retrieval_payload,
                chemistry_context=copy.deepcopy(
                    dict(self.task_input.get("supplementary_scientific_input", {}))
                ),
                num_queries=num_queries,
                num_hypotheses_per_query=hypotheses_per_query,
                top_n=top_n,
                hypothesis_ids_by_query=hypothesis_ids,
                merge_hypothesis_ids=merge_ids,
                # 首批生成不携带恢复记忆标志；HypothesisAgent 将其解释为后续
                # regeneration 的专用字段，首批传入非 None 会违反 V2 契约。
                memory_enabled=None,
                case_profile=self.policy.get("case_profile"),
                prior_manifest_path=prior_manifest_path,
                expected_task_id=request.research_task_id,
                expected_batch_id=request.module_invocation_id,
                task_config_schema="investigation_v2",
            )
        except HypothesisProtocolError as exc:
            if exc.failure_kind != "transport":
                raise
            operation_result = getattr(exc, "operation_result", None)
            if (
                not isinstance(operation_result, Mapping)
                or operation_result.get("status") != "failed"
                or operation_result.get("failure_kind") != "transport"
                or not isinstance(operation_result.get("manifest_path"), str)
            ):
                raise RuntimeError("Hypothesis transport 失败缺少 finalized manifest") from exc
            return ModuleSuspended(
                "hypothesis_transport_unavailable",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="hypothesis_transport_unavailable"
                ),
            )
        if not isinstance(operation, Mapping):
            return ModuleFailed("hypothesis_result_invalid")
        result = operation.get("hypothesis_result")
        if not isinstance(result, Mapping):
            return ModuleFailed("hypothesis_result_invalid")
        generated_pool = _validate_generated_hypothesis_pool(
            result,
            num_queries=num_queries,
            hypotheses_per_query=hypotheses_per_query,
        )
        if generated_pool is None:
            return ModuleFailed(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="hypothesis_candidate_pool_invalid"
                ),
            )
        generated_queries, hypotheses_by_query = generated_pool
        ranked = result.get("ranked_strategies")
        if not isinstance(ranked, list) or not ranked:
            return ModuleSuspended(
                "hypothesis_result_empty",
                recovery_verification=TemporarilyUnavailable(
                    reason_code="hypothesis_result_empty"
                ),
            )
        return _completed(
            request,
            (
                _record(
                    request,
                    "HypothesisBatchRecord",
                    "hypothesis",
                    {
                        "queries": copy.deepcopy(generated_queries),
                        "hypotheses_by_query": copy.deepcopy(hypotheses_by_query),
                        "ranked_strategies": copy.deepcopy(ranked),
                        "hypotheses": copy.deepcopy(ranked),
                        "rationale": "HypothesisAgent ranked the generated candidates",
                        "verification_plan": "Investigation Harness evaluates the selected candidates",
                    },
                    request.input_scientific_record_ids,
                    schema_version="hypothesis-generation.v2",
                ),
            ),
        )

    def start(self, request: ModuleInvocationRequest) -> Any:
        """调用 HypothesisAgent。"""

        return self._run(request, resume=False)

    def resume(self, request: ModuleInvocationRequest) -> Any:
        """恢复 HypothesisAgent operation。"""

        return self._run(request, resume=True)


class AuthorHypothesisImportModuleAdapter(_AdapterBase):
    """把 ResearchTaskCreated 冻结的 1～5 条作者假设直接导入为入选记录。"""

    def _run(self, request: ModuleInvocationRequest) -> ModuleCompleted | ModuleFailed:
        imported = self._read_imported_hypotheses()
        if imported is None:
            return ModuleFailed(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="author_hypothesis_import_invalid"
                ),
            )
        batch = _record(
            request,
            "HypothesisBatchRecord",
            "hypothesis-import",
            {"imported_hypotheses": imported},
            schema_version="author-import.v2",
        )
        selected = tuple(
            _record(
                request,
                "SelectedHypothesisRecord",
                f"selected-hypothesis:{rank}",
                {
                    "hypothesis_ref": item["hypothesis_ref"],
                    "rank": rank,
                    "hypothesis": copy.deepcopy(item["hypothesis"]),
                },
                (batch.scientific_record_id,),
            )
            for rank, item in enumerate(imported, start=1)
        )
        return _completed(
            request,
            (batch, *selected),
        )

    def _read_imported_hypotheses(self) -> list[dict[str, Any]] | None:
        """把通用作者列表或既有单 fixture 规范化为同一个严格输入。"""

        raw_items = self.task_input.get("author_hypotheses")
        if isinstance(raw_items, list):
            if not 1 <= len(raw_items) <= 5:
                return None
            imported: list[dict[str, Any]] = []
            for item in raw_items:
                if not isinstance(item, Mapping):
                    return None
                required = {
                    "hypothesis_ref",
                    "source_ref",
                    "source_digest",
                    "hypothesis",
                    "provenance",
                }
                if set(item) != required:
                    return None
                imported.append(copy.deepcopy(dict(item)))
            return imported

        fixture = self.task_input.get("author_fixture")
        if not isinstance(fixture, Mapping):
            return None
        required = {"fixture_id", "fixture_sha256", "hypothesis", "provenance"}
        if not required.issubset(fixture):
            return None
        hypothesis = fixture["hypothesis"]
        if not isinstance(hypothesis, Mapping):
            return None
        hypothesis_ref = hypothesis.get(
            "hypothesis_id", hypothesis.get("hypothesis_ref")
        )
        if not isinstance(hypothesis_ref, str) or not hypothesis_ref.strip():
            hypothesis_ref = f"author:{fixture['fixture_id']}"
        return [
            {
                "hypothesis_ref": hypothesis_ref,
                "source_ref": fixture["fixture_id"],
                "source_digest": fixture["fixture_sha256"],
                "hypothesis": copy.deepcopy(dict(hypothesis)),
                "provenance": copy.deepcopy(fixture["provenance"]),
            }
        ]

    def start(self, request: ModuleInvocationRequest) -> ModuleCompleted | ModuleFailed:
        """导入冻结作者假设，不调用 Retrieval/Hypothesis provider。"""

        return self._run(request)

    def resume(self, request: ModuleInvocationRequest) -> ModuleCompleted | ModuleFailed:
        """重放冻结作者假设。"""

        return self._run(request)


class SelectionModuleAdapter(_AdapterBase):
    """从最近一批 HypothesisRecord 中确定 Top 5。"""

    def _run(self, request: ModuleInvocationRequest) -> ModuleCompleted | ModuleFailed:
        records = self.working_memory.read_scientific_records(
            request.research_task_id, request.input_scientific_record_ids
        )
        if len(records) != 1 or records[0].scientific_record_type != "HypothesisBatchRecord":
            return ModuleFailed("hypothesis_record_missing")
        hypothesis_record = records[0]
        if hypothesis_record.scientific_record_schema_version == "author-import.v1":
            candidates = [hypothesis_record.payload["hypothesis"]]
        elif hypothesis_record.scientific_record_schema_version == "v10-migrated":
            # Migration deliberately preserves the old BatchCreated shape rather
            # than inventing a current ``hypotheses`` projection.  Selection is
            # still a public V2 boundary, so it may consume the closed migration
            # schema by taking the explicitly ranked candidates; it must not ask
            # the provider to regenerate them.
            candidates = []
            phase = hypothesis_record.payload.get("hypothesis_phase", {})
            ranked = phase.get("ranked_strategies") if isinstance(phase, Mapping) else None
            entries = hypothesis_record.payload.get("candidates")
            if isinstance(ranked, list) and isinstance(entries, list):
                by_id = {
                    item.get("hypothesis_id"): item
                    for item in ranked
                    if isinstance(item, Mapping)
                    and isinstance(item.get("hypothesis_id"), str)
                }
                candidates = [
                    by_id[item.get("hypothesis_id")]
                    for item in entries
                    if isinstance(item, Mapping)
                    and item.get("hypothesis_id") in by_id
                ]
        else:
            candidates = hypothesis_record.payload.get("hypotheses")
        if not isinstance(candidates, list) or not candidates:
            return ModuleFailed("hypothesis_candidates_missing")
        limit = 1 if hypothesis_record.scientific_record_schema_version == "author-import.v1" else 5
        if hypothesis_record.scientific_record_schema_version != "author-import.v1" and len(candidates) < 5:
            return ModuleFailed(
                "verified_integrity_conflict",
                recovery_verification=VerifiedIntegrityConflict(
                    reason_code="top5_candidate_count_invalid"
                ),
            )
        selected: list[dict[str, Any]] = []
        selected_candidates: list[tuple[str, int, dict[str, Any]]] = []
        for rank, candidate in enumerate(candidates[:limit], start=1):
            if not isinstance(candidate, Mapping):
                continue
            hypothesis_ref = candidate.get("hypothesis_id", candidate.get("hypothesis_ref"))
            if not isinstance(hypothesis_ref, str) or not hypothesis_ref.strip():
                hypothesis_ref = f"{hypothesis_record.scientific_record_id}:candidate:{rank}"
            selected.append({"hypothesis_ref": hypothesis_ref, "rank": rank})
            selected_candidates.append(
                (hypothesis_ref, rank, copy.deepcopy(dict(candidate)))
            )
        if not selected:
            return ModuleFailed("hypothesis_candidates_invalid")
        selection_record = _record(
            request,
            "HypothesisSelectionRecord",
            "selection",
            {
                "selected_hypotheses": selected,
                "selection_reason": "ranked candidates selected at the module boundary",
            },
            request.input_scientific_record_ids,
            schema_version="2",
        )
        output_records: list[ScientificRecord] = [selection_record]
        for hypothesis_ref, rank, candidate in selected_candidates:
            output_records.append(
                _record(
                    request,
                    "SelectedHypothesisRecord",
                    f"selected-hypothesis:{rank}",
                    {
                        "hypothesis_ref": hypothesis_ref,
                        "rank": rank,
                        "hypothesis": candidate,
                    },
                    (*request.input_scientific_record_ids, selection_record.scientific_record_id),
                )
            )
        return _completed(
            request,
            output_records,
        )

    def start(self, request: ModuleInvocationRequest) -> Any:
        """提交 Top 5 Selection。"""

        return self._run(request)

    def resume(self, request: ModuleInvocationRequest) -> Any:
        """重放 Top 5 Selection。"""

        return self._run(request)


def build_investigation_v2_module_adapters(
    controller: Any,
    *,
    task_root: Path,
    working_memory: ResearchTaskWorkingMemory,
    task_input: Mapping[str, Any],
    policy: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    task_config: Mapping[str, Any],
    research_material_store: ResearchMaterialStore | None = None,
) -> dict[str, Any]:
    """装配当前 V2 adapter；历史 schema 只由显式只读迁移入口处理。"""

    # V2 的默认 Investigation 必须由 Context + native Harness 深模块拥有。
    from chemistry_multiagent.controllers.investigation_v2_native_adapters import (
        build_native_investigation_module_adapter,
        build_native_review_module_adapter,
    )

    common = {
        "controller": controller,
        "task_root": task_root,
        "working_memory": working_memory,
        "task_input": task_input,
        "policy": policy,
        "dependencies": dependencies,
        "task_config": task_config,
    }
    native_dependencies = dict(dependencies)
    investigation = build_native_investigation_module_adapter(
        task_root=task_root,
        working_memory=working_memory,
        task_input=task_input,
        policy=policy,
        dependencies=native_dependencies,
        research_material_store=research_material_store,
        max_parallel_investigations=task_config.get(
            "max_parallel_investigations",
            5,
        ),
    )
    review = build_native_review_module_adapter(
        task_root=task_root,
        working_memory=working_memory,
        task_input=task_input,
        policy=policy,
        dependencies=native_dependencies,
    )
    return {
        "retrieval": RetrievalModuleAdapter(**common),
        "research-query-formulation": QueryModuleAdapter(**common),
        "hypothesis-generation": HypothesisModuleAdapter(**common),
        "hypothesis-import": AuthorHypothesisImportModuleAdapter(**common),
        "hypothesis-selection": SelectionModuleAdapter(**common),
        "hypothesis-investigation": investigation,
        "task-review": review,
    }


__all__ = [
    "AuthorHypothesisImportModuleAdapter",
    "HypothesisModuleAdapter",
    "QueryModuleAdapter",
    "RetrievalModuleAdapter",
    "SelectionModuleAdapter",
    "build_investigation_v2_module_adapters",
]
