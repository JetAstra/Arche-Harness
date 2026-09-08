"""Ticket 05：schema v10 investigation_v2 迁移的公共 seam 回归。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from chemistry_multiagent.controllers import (
    ControllerSuspended,
    ModuleSuspended,
    ResearchWorkflowController,
)
from chemistry_multiagent.memory import (
    ContractValidationError,
    CreateInvestigationContext,
    InvestigationV2Migration,
    MigrationSourceDrift,
    ResearchTaskLedger,
    ResearchTaskWorkingMemory,
    ReuseInvestigationContext,
)
from chemistry_multiagent.memory.contracts import canonical_json
from chemistry_multiagent.memory.v2_migration import ExistingHarnessRuntime


class _SchemaV10SourceFixture:
    """在测试内构造最小 schema-v10 source，不调用旧生产写服务。

    该 writer 只负责写入迁移器明确读取的 ``schema_meta``、``tasks`` 和
    ``memory_events`` 三张历史表，以及测试需要核验的 retrieval artifact。它
    不是当前 Working Memory 入口，也不提供旧流程的业务操作。
    """

    def __init__(self, db_path: Path, scratch_root: Path) -> None:
        self.db_path = db_path
        self.scratch_root = scratch_root
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE tasks (
                    task_id TEXT PRIMARY KEY,
                    scientific_question TEXT NOT NULL,
                    reaction_class TEXT,
                    chemistry_context_json TEXT NOT NULL,
                    execution_mode TEXT,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    stage_architecture_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE memory_events (
                    event_id TEXT PRIMARY KEY,
                    event_key TEXT NOT NULL UNIQUE,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    round INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX memory_events_task_idx ON memory_events(task_id);
                """
            )
            connection.executemany(
                "INSERT INTO schema_meta(key, value) VALUES (?, ?)",
                (
                    ("schema_version", "10"),
                    ("scientific_record_schema_version", "10"),
                ),
            )

    @contextmanager
    def transaction(self):
        """为测试 fixture 提供显式 SQLite 事务。"""

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    immediate_transaction = transaction

    def write_task_created_source(
        self,
        *,
        task_id: str,
        scientific_question: str,
        chemistry_context: Mapping[str, object],
        task_config: Mapping[str, object],
        task_output_relative_dir: str,
        tool_catalog_hash: str,
        stage_architecture_version: str = "investigation_v2",
        extra_task_created_fields: Mapping[str, object] | None = None,
    ) -> str:
        """构造一条冻结的历史 TaskCreated 和对应 tasks 行。"""

        created_at = "2026-08-28T00:00:00+00:00"
        payload: dict[str, object] = {
            "scientific_question": scientific_question,
            "chemistry_context": dict(chemistry_context),
            "task_config": dict(task_config),
            "task_output_relative_dir": task_output_relative_dir,
            "tool_catalog_hash": tool_catalog_hash,
            "stage_architecture_version": stage_architecture_version,
        }
        if extra_task_created_fields:
            payload.update(dict(extra_task_created_fields))
        event_id = f"old-task-created-{task_id}"
        event_key = f"wm:task:{task_id}:created"
        task_root = self.scratch_root / task_output_relative_dir
        task_root.mkdir(parents=True, exist_ok=True)
        reaction_class = chemistry_context.get("reaction_type") or chemistry_context.get(
            "reaction_class"
        )
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, scientific_question, reaction_class,
                    chemistry_context_json, execution_mode, status, version,
                    stage_architecture_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    scientific_question,
                    reaction_class,
                    canonical_json(dict(chemistry_context)),
                    "tool_catalog",
                    "running",
                    0,
                    stage_architecture_version,
                    created_at,
                    created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_events(
                    event_id, event_key, task_id, round, event_type,
                    payload_json, source_refs_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_key,
                    task_id,
                    0,
                    "task_created",
                    canonical_json(payload),
                    "[]",
                    created_at,
                ),
            )
        return event_id

    def write_retrieval_completed_source(
        self,
        task_id: str,
        retrieval_result: Mapping[str, object],
    ) -> str:
        """构造迁移器要求的 RetrievalCompleted 及其 content-addressed artifact。"""

        safe_result = dict(retrieval_result)
        result_digest = hashlib.sha256(
            canonical_json(safe_result).encode("utf-8")
        ).hexdigest()
        operation_instance_id = f"retrieval-{result_digest[:24]}"
        artifact_payload = {
            "status": "completed",
            "result_digest": result_digest,
            "retrieval_result": safe_result,
        }
        artifact_digest = hashlib.sha256(
            canonical_json(artifact_payload).encode("utf-8")
        ).hexdigest()
        artifact_ref = f"retrieval/retrieval_completed_{artifact_digest}.json"
        task_root = self.scratch_root / f"outputs/runs/{task_id}"
        artifact_path = task_root / artifact_ref
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        event_id = f"old-retrieval-{task_id}"
        payload = {
            "operation_instance_id": operation_instance_id,
            "operation": "retrieval",
            "status": "completed",
            "result_digest": result_digest,
            "artifact_ref": artifact_ref,
            "causal_source_ref": None,
        }
        self._append_event(
            event_id=event_id,
            event_key=f"wm:task:{task_id}:retrieval:{operation_instance_id}",
            task_id=task_id,
            event_type="retrieval_completed",
            payload=payload,
            source_refs=[artifact_ref],
            created_at="2026-08-28T00:30:00+00:00",
        )
        return event_id

    def _append_event(
        self,
        *,
        event_id: str,
        event_key: str,
        task_id: str,
        event_type: str,
        payload: Mapping[str, object],
        source_refs: list[str],
        created_at: str,
    ) -> None:
        """写入测试专用的不可变 source event。"""

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO memory_events(
                    event_id, event_key, task_id, round, event_type,
                    payload_json, source_refs_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_key,
                    task_id,
                    0,
                    event_type,
                    canonical_json(dict(payload)),
                    canonical_json(source_refs),
                    created_at,
                ),
            )


def _legacy_task_config(*, run_mode: str = "top5_selection") -> dict[str, object]:
    """返回旧 v10 investigation_v2 任务所需的最小冻结配置。"""

    return {
        "num_queries": 1,
        "num_hypotheses_per_query": 5,
        "top_n_hypotheses": 5,
        "max_parallel_investigations": 5,
        "max_investigation_repairs_per_combination": 3,
        "max_hypothesis_batches": 2,
        "run_mode": run_mode,
    }


def _schema_v10_fixture(
    tmp_path: Path,
    *,
    task_id: str = "legacy-v2-task",
    run_mode: str = "top5_selection",
) -> _SchemaV10SourceFixture:
    """创建只含旧 v10 行的测试 source fixture；不启动任何外部任务。"""

    fixture = _SchemaV10SourceFixture(
        tmp_path / "memory.sqlite3",
        tmp_path / "scratch",
    )
    fixture.write_task_created_source(
        task_id=task_id,
        scientific_question="冻结旧 V2 科学问题",
        chemistry_context={"reaction_type": "test"},
        task_config=_legacy_task_config(run_mode=run_mode),
        task_output_relative_dir=f"outputs/runs/{task_id}",
        tool_catalog_hash="a" * 64,
    )
    return fixture


def _append_source_event(
    fixture: _SchemaV10SourceFixture,
    *,
    event_id: str,
    event_key: str,
    task_id: str,
    event_type: str,
    payload: dict[str, object],
    source_refs: list[str],
    created_at: str,
) -> None:
    """向 fixture 的旧不可变事件日志追加一条已冻结来源事件。"""

    with fixture.immediate_transaction() as connection:
        connection.execute(
            """
            INSERT INTO memory_events(
                event_id, event_key, task_id, round, event_type,
                payload_json, source_refs_json, created_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_key,
                task_id,
                event_type,
                canonical_json(payload),
                canonical_json(source_refs),
                created_at,
            ),
        )


def _write_manifest_source(
    fixture: _SchemaV10SourceFixture,
    task_id: str,
    relative_ref: str,
    *,
    contents: Mapping[str, object] | None = None,
) -> None:
    """为 fixture 的明确 Batch source 写入一个可读 finalized manifest。"""

    target = (
        fixture.scratch_root
        / f"outputs/runs/{task_id}"
        / Path(*relative_ref.split("/"))
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            contents
            if contents is not None
            else {"schema": "test.finalized-hypothesis.v1", "task_id": task_id},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _record_retrieval_source(
    fixture: _SchemaV10SourceFixture,
    task_id: str,
) -> str:
    """构造迁移器要求的 RetrievalCompleted 与 content-addressed artifact。"""

    return fixture.write_retrieval_completed_source(
        task_id,
        {
            "retrieval_source": "migration-test",
            "context_summary": "冻结检索上下文。",
            "retrieved_papers": ["paper-a.pdf"],
        },
    )


def _rewrite_task_created_as_case3_shape(
    fixture: _SchemaV10SourceFixture,
    task_id: str,
) -> None:
    """把 fixture 的 TaskCreated 替换为脱敏 Case 3 的真实字段形状。"""

    payload = {
        "case_profile": "case3_fixed_reproduction",
        "chemistry_context": {"reaction_type": "case3-migration-fixture"},
        "expert_backend_selection": {"status": "disabled"},
        "gaussian_route_policy": {"policy": "fixture-frozen"},
        "hypothesis_import": {
            "mode": "author_case3_main_hypothesis_exploration",
            "fixture_id": "case3_author_main_hypothesis_exploration",
            "fixture_sha256": "a" * 64,
            "manifest_schema": "arche.investigation.author_hypothesis_import.v1",
            "expected_combination_count": 1,
        },
        "model_configuration": {"enabled": False},
        "scientific_question": "冻结旧 V2 科学问题",
        "stage_architecture_version": "investigation_v2",
        "task_config": {
            "num_queries": 1,
            "num_hypotheses_per_query": 1,
            "top_n_hypotheses": 1,
            "max_parallel_investigations": 1,
            "max_investigation_repairs_per_combination": 3,
            "max_hypothesis_batches": 2,
            "investigation_mode": "author_case3_main_hypothesis_exploration",
            "run_mode": "end_to_end",
        },
        "task_output_relative_dir": f"outputs/runs/{task_id}",
        "tool_catalog_hash": "a" * 64,
    }
    with fixture.immediate_transaction() as connection:
        connection.execute(
            "UPDATE memory_events SET payload_json=? "
            "WHERE task_id=? AND event_type='task_created'",
            (canonical_json(payload), task_id),
        )


def _top5_source(fixture: _SchemaV10SourceFixture, task_id: str) -> tuple[str, str]:
    """追加带迁移兼容字段的 BatchCreated 与 Top5 来源。"""

    with fixture.transaction() as connection:
        task_event_id = connection.execute(
            "SELECT event_id FROM memory_events WHERE task_id=? AND event_type='task_created'",
            (task_id,),
        ).fetchone()[0]
    timestamp = "2026-08-28T01:02:03+00:00"
    _write_manifest_source(fixture, task_id, "artifacts/model/hypothesis-legacy.json")
    _record_retrieval_source(fixture, task_id)
    candidates = [
        {
            "hypothesis_id": f"hyp-{index}",
            "candidate_index": index,
            "rank": index,
            "selection_status": "selected",
            "combination_id": f"comb-{index}",
            "combination_index": index,
        }
        for index in range(1, 6)
    ]
    ranked = [
        {
            "hypothesis_id": f"hyp-{index}",
            "strategy_name": f"策略 {index}",
            "reasoning": f"组合 {index} 的冻结理由。",
        }
        for index in range(1, 6)
    ]
    batch_event_id = "old-batch-event-1"
    _append_source_event(
        fixture,
        event_id=batch_event_id,
        event_key="wm:task:legacy-v2-task:batch:batch-1",
        task_id=task_id,
        event_type="hypothesis_batch_created",
        payload={
            "task_id": task_id,
            "batch_id": "batch-1",
            "batch_index": 1,
            "operation_instance_id": "batch-op-1",
            "question_snapshot": {
                "scientific_question": "冻结旧 V2 科学问题",
                "initial_query_angles": ["mechanism"],
                "initial_retrieval_phase": {"state": "complete"},
            },
            "generation_source": {
                "kind": "initial",
                "previous_batch_id": None,
                "source_record_ids": [],
            },
            "hypothesis_phase": {
                "ranked_strategies": ranked,
                "rationale": "旧事件保存的假设生成理由。",
                "verification_plan": "按冻结的组合计划核验。",
            },
            "candidates": candidates,
            "causal_source_ref": None,
            "investigation_protocol_version": 8,
        },
        source_refs=["artifacts/model/hypothesis-legacy.json"],
        created_at=timestamp,
    )
    selection_event_id = "old-top5-event-1"
    _append_source_event(
        fixture,
        event_id=selection_event_id,
        event_key="wm:task:legacy-v2-task:selection:batch-1",
        task_id=task_id,
        event_type="hypothesis_selection_completed",
        payload={
            "task_id": task_id,
            "batch_id": "batch-1",
            "batch_index": 1,
            "selected_count": 5,
            "selected_combinations": [
                {
                    "combination_id": f"comb-{index}",
                    "combination_index": index,
                    "hypothesis_id": f"hyp-{index}",
                    "rank": index,
                }
                for index in range(1, 6)
            ],
            "manifest_sha256": "b" * 64,
            "operation_instance_id": "selection-op-1",
            "investigation_protocol_version": 8,
            "selection_reason": "旧 Top5 事件明确保存的筛选理由。",
        },
        source_refs=[batch_event_id, task_event_id],
        created_at="2026-08-28T01:02:04+00:00",
    )
    return batch_event_id, selection_event_id


def _end_to_end_source(
    fixture: _SchemaV10SourceFixture,
    task_id: str,
    *,
    include_terminal: bool = True,
    include_review: bool = True,
    include_repair: bool = True,
    waiting: bool = False,
    partial: bool = False,
    partial_repair: bool = False,
    raw_case3_shape: bool = False,
) -> dict[str, str]:
    """构造闭合或等待中的 investigation_v2 source 事实图。"""

    with fixture.transaction() as connection:
        task_event_id = connection.execute(
            "SELECT event_id FROM memory_events WHERE task_id=? AND event_type='task_created'",
            (task_id,),
        ).fetchone()[0]
    base = "2026-08-28T02:00:"
    batch_id = "batch-1"
    batch_event_id = "old-e2e-batch"
    if raw_case3_shape:
        manifest_contents: Mapping[str, object] = {
            "fixture_id": "case3_author_main_hypothesis_exploration"
        }
        manifest_digest = hashlib.sha256(
            canonical_json(dict(manifest_contents)).encode("utf-8")
        ).hexdigest()
        manifest_ref = (
            "internal/catalog/author_hypothesis_import/manifest_"
            f"{manifest_digest}.json"
        )
    else:
        manifest_contents = {"schema": "test.finalized-hypothesis.v1", "task_id": task_id}
        manifest_ref = "artifacts/model/hypothesis-e2e.json"
    _write_manifest_source(
        fixture,
        task_id,
        manifest_ref,
        contents=manifest_contents,
    )
    _record_retrieval_source(fixture, task_id)
    combination_indexes = (1,) if raw_case3_shape else (1, 2)
    candidates = [
        {
            "hypothesis_id": f"hyp-{index}",
            "candidate_index": index,
            "rank": index,
            "selection_status": "selected",
            "combination_id": f"comb-{index}",
            "combination_index": index,
        }
        for index in combination_indexes
    ]
    ranked = [
        {
            "hypothesis_id": f"hyp-{index}",
            "strategy_name": f"策略 {index}",
            "reasoning": f"组合 {index} 的冻结理由。",
        }
        for index in combination_indexes
    ]
    question_snapshot = {
        "scientific_question": "冻结旧 V2 科学问题",
        "initial_query_angles": [] if raw_case3_shape else ["机制"],
        "initial_retrieval_phase": {} if raw_case3_shape else {"state": "complete"},
    }
    hypothesis_phase: dict[str, object] = {"ranked_strategies": ranked}
    if not raw_case3_shape:
        hypothesis_phase.update(
            {
                "rationale": "旧事件保存的假设生成理由。",
                "verification_plan": "按冻结组合计划核验。",
            }
        )
    _append_source_event(
        fixture,
        event_id=batch_event_id,
        event_key=f"wm:task:{task_id}:batch:{batch_id}",
        task_id=task_id,
        event_type="hypothesis_batch_created",
        payload={
            **({"task_id": task_id} if not raw_case3_shape else {}),
            "batch_id": batch_id,
            "batch_index": 1,
            "operation_instance_id": "batch-op-1",
            "question_snapshot": question_snapshot,
            "generation_source": {
                "kind": "initial",
                "previous_batch_id": None,
                "source_record_ids": [],
            },
            "hypothesis_phase": hypothesis_phase,
            "candidates": candidates,
            "causal_source_ref": None,
            "investigation_protocol_version": 8,
        },
        source_refs=[manifest_ref],
        created_at=f"{base}01+00:00",
    )

    ids: dict[str, str] = {"task": task_event_id, "batch": batch_event_id}

    def add_evidence(index: int, second: int) -> str:
        event_id = f"old-e2e-evidence-{index}-{second}"
        evidence_payload = {
            "task_id": task_id,
            "batch_id": batch_id,
            "combination_id": f"comb-{index}",
            "verification_object_id": f"verification-{index}-{second}",
            "verifier_id": "evidence-verifier",
            "verifier_version": "reference-v2",
            "verification_digest": f"{index}{second}" * 32,
            "operation_instance_id": f"evidence-op-{index}-{second}",
            "jobs": [
                {
                    "job_id": f"job-{index}-{second}",
                    "receipt_digest": "c" * 64,
                    "submission_operation_id": f"submission-op-{index}-{second}",
                    "terminal_status": "completed",
                }
            ],
            "operations": [
                {
                    "operation_id": f"typed-op-{index}-{second}",
                    "terminal_status": "completed",
                }
            ],
            "artifacts": [
                {
                    "artifact_id": f"artifact-{index}-{second}",
                    "artifact_digest": "e" * 64,
                    "producer_job_id": f"job-{index}-{second}",
                    "producer_submission_operation_id": f"submission-op-{index}-{second}",
                }
            ],
        }
        if not raw_case3_shape:
            evidence_payload.update(
                {
                    "source_manifest_ref": f"artifacts/manifest/{index}-{second}.json",
                    "summary": f"组合 {index} 的冻结证据摘要 {second}。",
                }
            )
        else:
            evidence_payload.update(
                {
                    "codex_thread_id": f"thread-{index}",
                    "reference_binding_version": "v1",
                    "scope_id": f"scope-{index}",
                    "turn_id": f"turn-{index}-{second}",
                }
            )
        _append_source_event(
            fixture,
            event_id=event_id,
            event_key=f"wm:task:{task_id}:evidence:{index}:{second}",
            task_id=task_id,
            event_type="investigation_evidence_verified",
            payload=evidence_payload,
            source_refs=[batch_event_id],
            created_at=f"{base}{2 + index + second:02d}+00:00",
        )
        return event_id

    def add_result(
        index: int,
        version: str,
        evidence_id: str,
        *,
        predecessor: str | None = None,
        repair_request: str | None = None,
    ) -> str:
        event_id = f"old-e2e-result-{index}-{version}"
        refs = [evidence_id]
        if predecessor is not None:
            refs.append(predecessor)
        if repair_request is not None:
            refs.append(repair_request)
        payload: dict[str, object] = {
            "task_id": task_id,
            "batch_id": batch_id,
            "combination_id": f"comb-{index}",
            "result_version": version,
            "evidence_verification_event_ref": evidence_id,
            "narrative": f"组合 {index} 的冻结结论 {version}。",
            "self_judgment": "supported",
            "observations": [f"观察 {index}-{version}"],
            "limitations": [],
        }
        if raw_case3_shape:
            suffix = 1 if version == "v1" else 2
            payload = {
                "artifact_refs": [f"artifact-{index}-{suffix}"],
                "batch_id": batch_id,
                "codex_thread_id": f"thread-{index}",
                "combination_id": f"comb-{index}",
                "evidence_verification_event_ref": evidence_id,
                "investigation_protocol_version": 8,
                "job_refs": [f"job-{index}-{suffix}"],
                "limitations": [],
                "narrative": f"组合 {index} 的冻结结论 {version}。",
                "observations": [f"观察 {index}-{version}"],
                "operation_instance_id": f"result-op-{index}-{version}",
                "operation_refs": [f"typed-op-{index}-{suffix}"],
                "result_version": version,
                "scope_id": f"scope-{index}",
                "self_judgment": "supported",
                "session_event_ref": f"session-{index}",
                "task_id": task_id,
                "turn_id": f"turn-{index}-{version}",
            }
        if repair_request is not None:
            if not raw_case3_shape:
                payload["repair_request_event_ref"] = repair_request
            else:
                payload["repair_turn_event_ref"] = f"repair-turn-{index}"
        _append_source_event(
            fixture,
            event_id=event_id,
            event_key=f"wm:task:{task_id}:result:{index}:{version}",
            task_id=task_id,
            event_type="investigation_result_recorded",
            payload=payload,
            source_refs=refs,
            created_at=f"{base}{10 + index + (0 if version == 'r1' else 10):02d}+00:00",
        )
        return event_id

    def add_waiting(index: int) -> None:
        _append_source_event(
            fixture,
            event_id=f"old-e2e-session-{index}",
            event_key=f"wm:task:{task_id}:session:{index}",
            task_id=task_id,
            event_type="investigation_session_started",
            payload={
                "task_id": task_id,
                "batch_id": batch_id,
                "combination_id": f"comb-{index}",
            },
            source_refs=[batch_event_id],
            created_at=f"{base}{20 + index:02d}+00:00",
        )
        _append_source_event(
            fixture,
            event_id=f"old-e2e-waiting-{index}",
            event_key=f"wm:task:{task_id}:waiting:{index}",
            task_id=task_id,
            event_type="investigation_turn_waiting",
            payload={
                "task_id": task_id,
                "batch_id": batch_id,
                "combination_id": f"comb-{index}",
                "runtime_ref": f"runtime-{index}",
            },
            source_refs=[f"old-e2e-session-{index}"],
            created_at=f"{base}{30 + index:02d}+00:00",
        )

    if waiting:
        for index in combination_indexes:
            add_waiting(index)
        return ids

    evidence_1 = add_evidence(1, 1)
    result_1 = add_result(1, "v1" if raw_case3_shape else "r1", evidence_1)
    ids.update(evidence_1=evidence_1, result_1=result_1)
    if partial:
        add_waiting(2)
        return ids

    if raw_case3_shape:
        evidence_2 = None
        result_2 = None
    else:
        evidence_2 = add_evidence(2, 1)
        result_2 = add_result(2, "r1", evidence_2)
    ids.update(
        **({} if raw_case3_shape else {"evidence_2": evidence_2, "result_2": result_2}),
    )
    if not include_review:
        return ids

    review_combinations = [f"comb-{index}" for index in combination_indexes]
    review_1 = "old-e2e-review-1"
    _append_source_event(
        fixture,
        event_id=review_1,
        event_key=f"wm:task:{task_id}:review:1",
        task_id=task_id,
        event_type="investigation_batch_reviewed",
        payload={
            "task_id": task_id,
            "batch_id": batch_id,
            "combination_ids": review_combinations,
            "result_versions": ["v1" if raw_case3_shape else "r1" for _ in combination_indexes],
            "findings": [
                {
                    "combination_id": "comb-1",
                    "validity": "fail",
                    "hypothesis_judgment": None,
                    "repair_advice": "补充组合一的核验。",
                },
                *([] if raw_case3_shape else [{
                    "combination_id": "comb-2",
                    "validity": "fail" if partial_repair else "pass",
                    "hypothesis_judgment": None if partial_repair else "supported",
                    "repair_advice": (
                        "补充组合二的核验。" if partial_repair else None
                    ),
                }]),
            ],
            "can_answer": False,
            "final_answer": None,
            "answerability_explanation": None,
            **(
                {
                    "investigation_protocol_version": 8,
                    "operation_instance_id": "review-op-1",
                    "result_version_vector": [
                        [combo, version]
                        for combo, version in zip(
                            review_combinations,
                            ["v1" if raw_case3_shape else "r1" for _ in combination_indexes],
                        )
                    ],
                    "review_thread_id": "review-thread-1",
                    "review_turn_id": "review-turn-1",
                    "task_review_session_event_ref": "review-session-1",
                    "task_review_turn_event_ref": "review-turn-1",
                }
                if raw_case3_shape
                else {}
            ),
            **({} if raw_case3_shape else {"outcome": "RepairRequired"}),
        },
        source_refs=[result_1] if raw_case3_shape else [result_1, result_2],
        created_at=f"{base}40+00:00",
    )
    ids["review_1"] = review_1
    if not include_repair:
        return ids

    repair_request = "old-e2e-repair-request-1"
    _append_source_event(
        fixture,
        event_id=repair_request,
        event_key=f"wm:task:{task_id}:repair-request:1",
        task_id=task_id,
        event_type="investigation_repair_requested",
        payload={
            "task_id": task_id,
            "batch_id": batch_id,
            "combination_id": "comb-1",
            "result_version": "v1" if raw_case3_shape else "r1",
            "repair_count": 1,
            "repair_advice": "补充组合一的核验。",
            **({} if raw_case3_shape else {"problem": "旧证据不充分。"}),
            "review_event_ref": review_1,
            "operation_instance_id": "repair-request-op-1",
            **({"repair_request_number": 1} if raw_case3_shape else {}),
        },
        source_refs=[review_1, result_1],
        created_at=f"{base}41+00:00",
    )
    if partial_repair:
        if raw_case3_shape:
            raise AssertionError("raw Case3 fixture 不使用 partial_repair 分支")
        pending_repair_request = "old-e2e-repair-request-pending"
        _append_source_event(
            fixture,
            event_id=pending_repair_request,
            event_key=f"wm:task:{task_id}:repair-request:pending",
            task_id=task_id,
            event_type="investigation_repair_requested",
            payload={
                "task_id": task_id,
                "batch_id": batch_id,
                "combination_id": "comb-2",
                "result_version": "r1",
                "repair_count": 1,
                "repair_advice": "补充组合二的核验。",
                "problem": "旧证据不充分。",
                "review_event_ref": review_1,
                "operation_instance_id": "repair-request-op-pending",
            },
            source_refs=[review_1, result_2],
            created_at=f"{base}42+00:00",
        )
        ids["pending_repair_request"] = pending_repair_request
    evidence_3 = add_evidence(1, 2)
    result_3 = add_result(
        1,
        "v2" if raw_case3_shape else "r2",
        evidence_3,
        predecessor=result_1,
        repair_request=repair_request,
    )
    review_2 = "old-e2e-review-2"
    _append_source_event(
        fixture,
        event_id=review_2,
        event_key=f"wm:task:{task_id}:review:2",
        task_id=task_id,
        event_type="investigation_batch_reviewed",
        payload={
            "task_id": task_id,
            "batch_id": batch_id,
            "combination_ids": review_combinations,
            "result_versions": [
                "v2" if raw_case3_shape else "r2",
                *([] if raw_case3_shape else ["r1"]),
            ],
            "findings": [
                {
                    "combination_id": "comb-1",
                    "validity": "pass",
                    "hypothesis_judgment": "supported",
                    "repair_advice": None,
                },
                *([] if raw_case3_shape else [{
                    "combination_id": "comb-2",
                    "validity": "pass",
                    "hypothesis_judgment": "supported",
                    "repair_advice": None,
                }]),
            ],
            "can_answer": True,
            "final_answer": "冻结的最终回答。",
            "answerability_explanation": None,
            **(
                {
                    "investigation_protocol_version": 8,
                    "operation_instance_id": "review-op-2",
                    "result_version_vector": [
                        [combo, version]
                        for combo, version in zip(
                            review_combinations,
                            [
                                "v2" if raw_case3_shape else "r2",
                                *([] if raw_case3_shape else ["r1"]),
                            ],
                        )
                    ],
                    "review_thread_id": "review-thread-1",
                    "review_turn_id": "review-turn-2",
                    "task_review_session_event_ref": "review-session-1",
                    "task_review_turn_event_ref": "review-turn-2",
                }
                if raw_case3_shape
                else {}
            ),
            **({} if raw_case3_shape else {"outcome": "Accepted"}),
        },
        source_refs=[result_3] if raw_case3_shape else [result_3, result_2],
        created_at=f"{base}50+00:00",
    )
    ids.update(repair_request=repair_request, evidence_3=evidence_3, result_3=result_3, review_2=review_2)
    if partial_repair:
        return ids
    if include_terminal:
        terminal = "old-e2e-terminal"
        _append_source_event(
            fixture,
            event_id=terminal,
            event_key=f"wm:task:{task_id}:terminal",
            task_id=task_id,
            event_type="investigation_task_closed",
            payload={
                "task_id": task_id,
                "batch_id": batch_id,
                "accepted": True,
                "can_answer": True,
                "final_answer": "冻结的最终回答。",
                "review_event_ref": review_2,
                "answerability_explanation": None,
                **(
                    {
                        "investigation_protocol_version": 8,
                        "marker_event_refs": [],
                        "operation_instance_id": "terminal-op-1",
                    }
                    if raw_case3_shape
                    else {}
                ),
            },
            source_refs=[review_2],
            created_at=f"{base}51+00:00",
        )
        ids["terminal"] = terminal
    return ids


def test_legacy_task_is_rejected_before_v2_schema_creation(tmp_path: Path) -> None:
    """legacy task 不应因迁移入口构造新版账本而留下任何 v2 表。"""

    fixture = _SchemaV10SourceFixture(
        tmp_path / "legacy.sqlite3",
        tmp_path / "scratch",
    )
    fixture.write_task_created_source(
        task_id="legacy-task",
        scientific_question="旧流程",
        chemistry_context={"reaction_type": "test"},
        task_config={
            "num_queries": 1,
            "num_hypotheses_per_query": 1,
            "top_n_workflows": 1,
            "max_parallel_combinations": 1,
            "max_workflow_revisions_per_combination": 1,
            "max_hypothesis_batches": 1,
        },
        task_output_relative_dir="outputs/runs/legacy-task",
        tool_catalog_hash="a" * 64,
        stage_architecture_version="legacy_workflow_v1",
    )
    before = sqlite3.connect(tmp_path / "legacy.sqlite3")
    try:
        tables_before = {
            row[0]
            for row in before.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        before.close()

    with pytest.raises(ContractValidationError, match="legacy_workflow_v1"):
        InvestigationV2Migration(str(tmp_path / "legacy.sqlite3")).migrate_existing_task(
            "legacy-task"
        )

    with sqlite3.connect(tmp_path / "legacy.sqlite3") as connection:
        tables_after = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert tables_after == tables_before
    assert not any(name.startswith("v2_") for name in tables_after)


def test_top5_migration_is_deterministic_and_coexists_with_v10_rows(tmp_path: Path) -> None:
    """完整 Top5 来源迁移后可读，旧表保持不变且重复调用不追加事件。"""

    legacy = _schema_v10_fixture(tmp_path)
    batch_event_id, selection_event_id = _top5_source(legacy, "legacy-v2-task")
    db_path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(db_path) as connection:
        old_event_count = connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE task_id=?",
            ("legacy-v2-task",),
        ).fetchone()[0]

    first = InvestigationV2Migration(str(db_path)).migrate_existing_task(
        "legacy-v2-task", source_root=tmp_path / "scratch"
    )
    assert first.research_task_id == "legacy-v2-task"
    assert first.migrated is True
    v2 = ResearchTaskWorkingMemory(str(db_path))
    facts = v2.read_research_task_facts("legacy-v2-task")
    assert facts.lifecycle.state == "Closed"
    assert facts.lifecycle.outcome["status"] == "succeeded"
    assert [run.module_kind for run in facts.module_invocations] == [
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-selection",
    ]
    assert facts.scientific_graph.record_refs_by_type["HypothesisBatchRecord"]
    assert facts.scientific_graph.record_refs_by_type["HypothesisSelectionRecord"]
    event_snapshot = tuple(
        event.to_dict() for event in v2.read_research_task_event_log("legacy-v2-task")
    )

    second = InvestigationV2Migration(str(db_path)).migrate_existing_task(
        "legacy-v2-task", source_root=tmp_path / "scratch"
    )
    assert second == first
    assert tuple(
        event.to_dict() for event in v2.read_research_task_event_log("legacy-v2-task")
    ) == event_snapshot

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE task_id=?",
            ("legacy-v2-task",),
        ).fetchone()[0] == old_event_count
        marker = connection.execute(
            "SELECT source_event_count,target_head_digest,status FROM v2_migration_markers WHERE task_id=?",
            ("legacy-v2-task",),
        ).fetchone()
        assert marker[0] == old_event_count
        assert len(marker[1]) == 64
        assert marker[2] == "complete"
        mapped = connection.execute(
            "SELECT COUNT(*) FROM v2_migration_event_map WHERE task_id=? AND source_event_id IN (?,?)",
            ("legacy-v2-task", batch_event_id, selection_event_id),
        ).fetchone()[0]
        assert mapped >= 2


def test_migration_source_drift_fails_without_v2_append(tmp_path: Path) -> None:
    """marker 已完成后旧不可变事件发生漂移必须拒绝，不追加目标事件。"""

    legacy = _schema_v10_fixture(tmp_path)
    _top5_source(legacy, "legacy-v2-task")
    db_path = tmp_path / "memory.sqlite3"
    migration = InvestigationV2Migration(str(db_path))
    migration.migrate_existing_task(
        "legacy-v2-task", source_root=tmp_path / "scratch"
    )
    v2 = ResearchTaskWorkingMemory(str(db_path))
    before = tuple(
        event.to_dict() for event in v2.read_research_task_event_log("legacy-v2-task")
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE memory_events SET payload_json=? WHERE event_key=?",
            (canonical_json({"drift": True}), "wm:task:legacy-v2-task:selection:batch-1"),
        )
    with pytest.raises(MigrationSourceDrift):
        migration.migrate_existing_task("legacy-v2-task")
    assert tuple(
        event.to_dict() for event in v2.read_research_task_event_log("legacy-v2-task")
    ) == before


def test_full_chain_migrates_result_review_repair_and_one_to_many_outputs(
    tmp_path: Path,
) -> None:
    """Result→Review→Repair→Result 的前驱图和一个模块的多组输出保持可读。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="full-v2-task", run_mode="end_to_end")
    source = _end_to_end_source(legacy, "full-v2-task")
    db_path = tmp_path / "memory.sqlite3"

    result = InvestigationV2Migration(str(db_path)).migrate_existing_task(
        "full-v2-task", source_root=tmp_path / "scratch"
    )
    service = ResearchTaskWorkingMemory(str(db_path))
    facts = service.read_research_task_facts("full-v2-task")

    assert facts.lifecycle.state == "Closed"
    assert facts.lifecycle.outcome["status"] == "succeeded"
    assert [run.module_kind for run in facts.module_invocations] == [
        "retrieval",
        "research-query-formulation",
        "hypothesis-generation",
        "hypothesis-investigation",
        "task-review",
        "hypothesis-investigation",
        "task-review",
    ]
    records = service.read_research_task_scientific_records("full-v2-task")
    by_type = {}
    for record in records:
        by_type.setdefault(record.scientific_record_type, []).append(record)
    assert len(by_type["VerifiedEvidenceRecord"]) == 3
    assert len(by_type["InvestigationResultRecord"]) == 3
    assert len(by_type["ReviewRecord"]) == 2
    assert len(by_type["RepairRequestRecord"]) == 1
    assert len(by_type["FinalAnswerRecord"]) == 1

    repaired = next(
        record
        for record in by_type["InvestigationResultRecord"]
        if record.payload["conclusion"].endswith("r2。")
    )
    repair = by_type["RepairRequestRecord"][0]
    original = next(
        record
        for record in by_type["InvestigationResultRecord"]
        if record.payload["conclusion"].endswith("r1。")
        and record.payload["conclusion"].startswith("组合 1")
    )
    assert original.scientific_record_id in repaired.source_scientific_record_ids
    assert repair.scientific_record_id in repaired.source_scientific_record_ids
    assert repaired.payload["evidence_ref"] in repaired.source_scientific_record_ids

    investigation_runs = [
        run for run in facts.module_invocations if run.module_kind == "hypothesis-investigation"
    ]
    assert all(
        len(
            facts.scientific_graph.output_scientific_record_ids_by_module_invocation[
                run.module_invocation_id
            ]
        )
        in {4, 2}
        for run in investigation_runs
    )
    assert result.target_head_digest == service.read_research_task_event_log("full-v2-task")[-1].event_digest

    with sqlite3.connect(db_path) as connection:
        mapped = connection.execute(
            "SELECT target_kind,target_id,target_event_id FROM v2_migration_event_map "
            "WHERE task_id=? AND source_event_id=? ORDER BY target_kind",
            ("full-v2-task", source["terminal"]),
        ).fetchall()
    assert {row[0] for row in mapped} >= {"record", "terminal"}


def test_open_waiting_migration_uses_read_only_runtime_resolver(tmp_path: Path) -> None:
    """未完成 task 只生成一个 Open Investigation，resolver 不得成为执行入口。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="waiting-v2-task", run_mode="end_to_end")
    _end_to_end_source(legacy, "waiting-v2-task", waiting=True)
    calls: list[dict[str, object]] = []

    class Resolver:
        def resolve(self, task_id: str, **kwargs: object) -> ExistingHarnessRuntime:
            calls.append({"task_id": task_id, **kwargs})
            return ExistingHarnessRuntime(runtime_ref="runtime-existing-1", state="suspended")

    db_path = tmp_path / "memory.sqlite3"
    InvestigationV2Migration(str(db_path), runtime_resolver=Resolver()).migrate_existing_task(
        "waiting-v2-task", source_root=tmp_path / "scratch"
    )
    service = ResearchTaskWorkingMemory(str(db_path))
    facts = service.read_research_task_facts("waiting-v2-task")
    assert facts.lifecycle.state == "Open"
    assert facts.open_module_invocation is not None
    assert facts.open_module_invocation.lifecycle.state == "Open"
    assert isinstance(
        facts.open_module_invocation.context_binding,
        CreateInvestigationContext,
    )
    assert facts.open_module_invocation.context_binding.investigation_context_id == (
        "runtime-existing-1"
    )
    assert len(calls) == 1
    assert calls[0]["task_id"] == "waiting-v2-task"
    assert "source_event_ids" in calls[0]
    assert not any(key in calls[0] for key in ("start", "resume", "submit"))


def test_existing_result_wins_over_stale_waiting_without_resolver(tmp_path: Path) -> None:
    """已有 Result 即使伴随 waiting 记录也不调用 Harness resolver。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="stale-waiting-task", run_mode="end_to_end")
    _end_to_end_source(legacy, "stale-waiting-task")
    with legacy.immediate_transaction() as connection:
        task_event_id = connection.execute(
            "SELECT event_id FROM memory_events WHERE task_id=? AND event_type='task_created'",
            ("stale-waiting-task",),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO memory_events(event_id,event_key,task_id,round,event_type,payload_json,source_refs_json,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                "old-stale-waiting",
                "wm:task:stale-waiting-task:stale-waiting",
                "stale-waiting-task",
                0,
                "investigation_turn_waiting",
                canonical_json({"task_id": "stale-waiting-task", "batch_id": "batch-1", "combination_id": "comb-1"}),
                canonical_json([task_event_id]),
                "2026-08-28T03:00:00+00:00",
            ),
        )
    called = False

    def resolver(*args: object, **kwargs: object) -> ExistingHarnessRuntime:
        nonlocal called
        called = True
        raise AssertionError("已有 Result 不应访问 runtime resolver")

    InvestigationV2Migration(
        str(tmp_path / "memory.sqlite3"), runtime_resolver=resolver
    ).migrate_existing_task(
        "stale-waiting-task", source_root=tmp_path / "scratch"
    )
    assert called is False


def test_partial_initial_result_closes_history_and_keeps_open_run_output_empty(
    tmp_path: Path,
) -> None:
    """旧 partial Result 只能形成闭合历史加零输出 continuation。"""

    legacy = _schema_v10_fixture(
        tmp_path,
        task_id="partial-v2-task",
        run_mode="end_to_end",
    )
    _end_to_end_source(legacy, "partial-v2-task", partial=True)
    calls: list[dict[str, object]] = []

    class Resolver:
        def resolve(self, task_id: str, **kwargs: object) -> ExistingHarnessRuntime:
            calls.append({"task_id": task_id, **kwargs})
            return ExistingHarnessRuntime(runtime_ref="runtime-partial-existing", state="suspended")

    db_path = tmp_path / "memory.sqlite3"
    InvestigationV2Migration(
        str(db_path), runtime_resolver=Resolver()
    ).migrate_existing_task(
        "partial-v2-task",
        source_root=tmp_path / "scratch",
    )
    service = ResearchTaskWorkingMemory(str(db_path))
    facts = service.read_research_task_facts("partial-v2-task")
    assert facts.lifecycle.state == "Open"
    assert facts.open_module_invocation is not None
    assert facts.open_module_invocation.lifecycle.state == "Open"
    assert (
        facts.scientific_graph.output_scientific_record_ids_by_module_invocation.get(
            facts.open_module_invocation.module_invocation_id,
            (),
        )
        == ()
    )
    investigation_runs = [
        run
        for run in facts.module_invocations
        if run.module_kind == "hypothesis-investigation"
    ]
    assert len(investigation_runs) == 2
    closed_history, continuation = investigation_runs
    assert closed_history.lifecycle.state == "Closed"
    assert len(
        facts.scientific_graph.output_scientific_record_ids_by_module_invocation[
            closed_history.module_invocation_id
        ]
    ) == 2
    assert continuation.module_invocation_id == facts.open_module_invocation.module_invocation_id
    assert isinstance(closed_history.context_binding, CreateInvestigationContext)
    assert closed_history.context_binding.investigation_context_id == (
        "runtime-partial-existing"
    )
    assert isinstance(continuation.context_binding, ReuseInvestigationContext)
    assert continuation.context_binding.investigation_context_id == (
        "runtime-partial-existing"
    )
    assert continuation.context_binding.source_module_invocation_id == (
        closed_history.module_invocation_id
    )
    assert not any(
        record.producer_module_invocation_id == continuation.module_invocation_id
        for record in service.read_research_task_scientific_records("partial-v2-task")
    )
    assert len(calls) == 1


def test_missing_or_ambiguous_runtime_fails_before_any_target_write(tmp_path: Path) -> None:
    """Open 运行身份无法唯一核验时 fail closed，且一个 target 行也不留下。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="missing-runtime-task", run_mode="end_to_end")
    _end_to_end_source(legacy, "missing-runtime-task", waiting=True)
    db_path = tmp_path / "memory.sqlite3"
    with pytest.raises(ContractValidationError, match="runtime"):
        InvestigationV2Migration(str(db_path)).migrate_existing_task(
            "missing-runtime-task", source_root=tmp_path / "scratch"
        )
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert not any(name.startswith("v2_") for name in tables)

    class AmbiguousResolver:
        def resolve(self, task_id: str, **kwargs: object) -> dict[str, object]:
            return {"status": "VerifiedIntegrityConflict"}

    with pytest.raises(ContractValidationError, match="VerifiedIntegrityConflict"):
        InvestigationV2Migration(
            str(db_path), runtime_resolver=AmbiguousResolver()
        ).migrate_existing_task(
            "missing-runtime-task", source_root=tmp_path / "scratch"
        )
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='research_tasks'"
        ).fetchone() is None


def test_projection_deletion_rebuilds_migrated_facts(tmp_path: Path) -> None:
    """迁移后的投影删除不影响从目标事件日志重建 Task Facts。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="projection-task", run_mode="end_to_end")
    _end_to_end_source(legacy, "projection-task")
    db_path = tmp_path / "memory.sqlite3"
    InvestigationV2Migration(str(db_path)).migrate_existing_task(
        "projection-task", source_root=tmp_path / "scratch"
    )
    service = ResearchTaskWorkingMemory(str(db_path))
    before = service.read_research_task_facts("projection-task").to_dict()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "DELETE FROM research_task_facts_projections WHERE research_task_id=?",
            ("projection-task",),
        )
        connection.execute(
            "DELETE FROM module_invocations WHERE research_task_id=?",
            ("projection-task",),
        )
        connection.execute(
            "DELETE FROM scientific_records WHERE research_task_id=?",
            ("projection-task",),
        )
    assert service.read_research_task_facts("projection-task").to_dict() == before


def test_migration_transaction_fault_rolls_back_every_target_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标投影故障时 Task、事件、投影、marker 必须整体回滚。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="fault-task")
    _top5_source(legacy, "fault-task")
    db_path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(db_path) as connection:
        old_event_count = connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE task_id=?",
            ("fault-task",),
        ).fetchone()[0]

    def fail_projection(
        connection: sqlite3.Connection,
        *,
        record: object,
        created_event_id: str,
    ) -> None:
        del connection, record, created_event_id
        raise RuntimeError("injected migration projection fault")

    monkeypatch.setattr(
        ResearchTaskLedger,
        "_upsert_scientific_record_projection",
        staticmethod(fail_projection),
    )
    with pytest.raises(RuntimeError, match="projection fault"):
        InvestigationV2Migration(str(db_path)).migrate_existing_task(
            "fault-task",
            source_root=tmp_path / "scratch",
        )

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE task_id=?",
            ("fault-task",),
        ).fetchone()[0] == old_event_count
        for table in (
            "research_tasks",
            "research_task_events",
            "module_invocations",
            "scientific_records",
            "research_task_facts_projections",
            "v2_migration_markers",
            "v2_migration_event_map",
        ):
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists is not None:
                assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_migrated_open_run_routes_to_controller_resume_only(
    tmp_path: Path,
) -> None:
    """迁移后的 Open Investigation 通过 Controller 只走 resume seam。"""

    legacy = _schema_v10_fixture(
        tmp_path,
        task_id="controller-route-task",
        run_mode="end_to_end",
    )
    _end_to_end_source(legacy, "controller-route-task", waiting=True)

    class Resolver:
        def resolve(self, task_id: str, **kwargs: object) -> ExistingHarnessRuntime:
            del task_id, kwargs
            return ExistingHarnessRuntime(runtime_ref="runtime-controller-existing", state="suspended")

    db_path = tmp_path / "memory.sqlite3"
    InvestigationV2Migration(
        str(db_path), runtime_resolver=Resolver()
    ).migrate_existing_task(
        "controller-route-task",
        source_root=tmp_path / "scratch",
    )
    working_memory = ResearchTaskWorkingMemory(str(db_path))
    calls: list[tuple[str, str, str]] = []

    class Runner:
        def start(self, request: object) -> ModuleSuspended:
            del request
            raise AssertionError("迁移后的 Open run 不应重新 start")

        def resume(self, request: object) -> ModuleSuspended:
            calls.append(
                (
                    "resume",
                    request.module_invocation_id,
                    request.context_binding.investigation_context_id,
                )
            )
            return ModuleSuspended("waiting_existing_runtime")

    controller = ResearchWorkflowController(
        working_memory,
        {"hypothesis-investigation": Runner()},
    )
    result = controller.resume_research_task("controller-route-task")
    assert isinstance(result, ControllerSuspended)
    assert calls == [
        (
            "resume",
            working_memory.read_research_task_facts(
                "controller-route-task"
            ).open_module_invocation.module_invocation_id,
            "runtime-controller-existing",
        )
    ]


def test_source_integrity_failures_leave_target_empty(tmp_path: Path) -> None:
    """跨 task 引用、非法 branch 和缺 source 均在计划阶段停止。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="integrity-task", run_mode="end_to_end")
    _end_to_end_source(legacy, "integrity-task", include_terminal=False)
    db_path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE memory_events SET source_refs_json=? WHERE event_id=?",
            (canonical_json(["missing-source-event"]), "old-e2e-result-1-r1"),
        )
    with pytest.raises(ContractValidationError):
        InvestigationV2Migration(str(db_path)).migrate_existing_task(
            "integrity-task", source_root=tmp_path / "scratch"
        )
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='research_tasks'"
        ).fetchone() is None


def test_cross_task_source_reference_fails_closed_before_target_creation(
    tmp_path: Path,
) -> None:
    """source event 指向另一个 task 时不得把外部事实拼进迁移图。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="cross-task-main")
    source = _end_to_end_source(legacy, "cross-task-main", include_terminal=False)
    legacy.write_task_created_source(
        task_id="cross-task-foreign",
        scientific_question="另一个 task",
        chemistry_context={"reaction_type": "test"},
        task_config=_legacy_task_config(run_mode="end_to_end"),
        task_output_relative_dir="outputs/runs/cross-task-foreign",
        tool_catalog_hash="b" * 64,
    )
    with legacy.transaction() as connection:
        foreign_event_id = connection.execute(
            "SELECT event_id FROM memory_events "
            "WHERE task_id=? AND event_type='task_created'",
            ("cross-task-foreign",),
        ).fetchone()[0]
        refs = json.loads(
            connection.execute(
                "SELECT source_refs_json FROM memory_events WHERE event_id=?",
                (source["result_1"],),
            ).fetchone()[0]
        )
        refs.append(foreign_event_id)
        connection.execute(
            "UPDATE memory_events SET source_refs_json=? WHERE event_id=?",
            (canonical_json(refs), source["result_1"]),
        )
    with pytest.raises(ContractValidationError):
        InvestigationV2Migration(str(tmp_path / "memory.sqlite3")).migrate_existing_task(
            "cross-task-main",
            source_root=tmp_path / "scratch",
        )
    with sqlite3.connect(tmp_path / "memory.sqlite3") as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='research_tasks'"
        ).fetchone() is None


def test_invalid_review_branch_fails_closed_without_target_rows(tmp_path: Path) -> None:
    """Review 的显式终态与 RepairRequest 冲突时不得迁移任何目标事实。"""

    legacy = _schema_v10_fixture(tmp_path, task_id="invalid-branch-task", run_mode="end_to_end")
    source = _end_to_end_source(legacy, "invalid-branch-task")
    with legacy.immediate_transaction() as connection:
        review_payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM memory_events WHERE event_id=?",
                (source["review_1"],),
            ).fetchone()[0]
        )
        review_payload["outcome"] = "Accepted"
        connection.execute(
            "UPDATE memory_events SET payload_json=? WHERE event_id=?",
            (canonical_json(review_payload), source["review_1"]),
        )
    with pytest.raises(ContractValidationError):
        InvestigationV2Migration(str(tmp_path / "memory.sqlite3")).migrate_existing_task(
            "invalid-branch-task",
            source_root=tmp_path / "scratch",
        )
    with sqlite3.connect(tmp_path / "memory.sqlite3") as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='research_tasks'"
        ).fetchone() is None


def test_case3_sanitized_fixture_migrates_actual_v10_shapes_without_fabrication(
    tmp_path: Path,
) -> None:
    """脱敏 Case 3 fixture 固化真实字段缺口，CI 不依赖 local_data。"""

    legacy = _schema_v10_fixture(
        tmp_path,
        task_id="case3-shape-task",
        run_mode="end_to_end",
    )
    source = _end_to_end_source(
        legacy,
        "case3-shape-task",
        raw_case3_shape=True,
    )
    _rewrite_task_created_as_case3_shape(legacy, "case3-shape-task")
    db_path = tmp_path / "memory.sqlite3"
    with legacy.transaction() as connection:
        payloads = {
            event_id: json.loads(
                connection.execute(
                    "SELECT payload_json FROM memory_events WHERE event_id=?",
                    (event_id,),
                ).fetchone()[0]
            )
            for event_id in source.values()
            if connection.execute(
                "SELECT 1 FROM memory_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
        }
        task_created = json.loads(
            connection.execute(
                "SELECT payload_json FROM memory_events WHERE event_id=?",
                (source["task"],),
            ).fetchone()[0]
        )
        assert set(task_created) == {
            "case_profile",
            "chemistry_context",
            "expert_backend_selection",
            "gaussian_route_policy",
            "hypothesis_import",
            "model_configuration",
            "scientific_question",
            "stage_architecture_version",
            "task_config",
            "task_output_relative_dir",
            "tool_catalog_hash",
        }
        batch = payloads[source["batch"]]
        assert set(batch["question_snapshot"]) == {
            "initial_query_angles",
            "initial_retrieval_phase",
            "scientific_question",
        }
        assert batch["question_snapshot"]["initial_query_angles"] == []
        assert set(batch["hypothesis_phase"]) == {"ranked_strategies"}
        assert set(batch["candidates"][0]) == {
            "candidate_index",
            "combination_id",
            "combination_index",
            "hypothesis_id",
            "rank",
            "selection_status",
        }
        assert set(payloads[source["evidence_1"]]) == {
            "artifacts",
            "batch_id",
            "codex_thread_id",
            "combination_id",
            "jobs",
            "operation_instance_id",
            "operations",
            "reference_binding_version",
            "scope_id",
            "task_id",
            "turn_id",
            "verification_digest",
            "verification_object_id",
            "verifier_id",
            "verifier_version",
        }
        assert "source_manifest_ref" not in payloads[source["evidence_1"]]
        assert "summary" not in payloads[source["evidence_1"]]

    result = InvestigationV2Migration(str(db_path)).migrate_existing_task(
        "case3-shape-task",
        source_root=tmp_path / "scratch",
    )
    service = ResearchTaskWorkingMemory(str(db_path))
    facts = service.read_research_task_facts("case3-shape-task")
    assert facts.lifecycle.state == "Closed"
    records = service.read_research_task_scientific_records("case3-shape-task")
    by_type: dict[str, list[object]] = {}
    for record in records:
        by_type.setdefault(record.scientific_record_type, []).append(record)
    assert "QueryRecord" not in by_type
    assert "HypothesisSelectionRecord" not in by_type
    assert len(by_type["RetrievalRecord"]) == 1
    assert len(by_type["HypothesisBatchRecord"]) == 1
    assert len(by_type["VerifiedEvidenceRecord"]) == 2
    assert len(by_type["InvestigationResultRecord"]) == 2
    assert len(by_type["ReviewRecord"]) == 2
    assert len(by_type["RepairRequestRecord"]) == 1
    assert len(by_type["FinalAnswerRecord"]) == 1

    batch = by_type["HypothesisBatchRecord"][0]
    assert batch.scientific_record_schema_version == "v10-migrated"
    assert set(batch.payload["hypothesis_phase"]) == {"ranked_strategies"}
    assert "rationale" not in batch.payload
    assert "verification_plan" not in batch.payload
    evidence = by_type["VerifiedEvidenceRecord"]
    assert all(
        record.scientific_record_schema_version == "v10-migrated"
        for record in evidence
    )
    assert all("source_manifest_ref" not in record.payload for record in evidence)
    assert all("summary" not in record.payload for record in evidence)
    assert all(
        set(record.payload) >= {
            "evidence_items",
            "artifact_refs",
            "verification_object_id",
            "verification_digest",
            "verifier_id",
            "verifier_version",
            "operation_instance_id",
        }
        for record in evidence
    )
    assert result.source_event_count > 1
    source_types = {
        row[0]
        for row in sqlite3.connect(db_path).execute(
            "SELECT event_type FROM memory_events WHERE task_id=?",
            ("case3-shape-task",),
        )
    }
    assert "retrieval_completed" in source_types
    assert "hypothesis_selection_completed" not in source_types


def test_case3_real_sqlite_copy_migrates_raw_v10_shapes(tmp_path: Path) -> None:
    """从本地 Case3 原始 DB 临时副本验证真实 v10 字段形状。"""

    case3_dir = Path(__file__).parents[1] / "local_data/runtime/case3_main_hypothesis_exploration_local"
    candidates = sorted(
        case3_dir.glob("*/outputs/runs/*/internal/state/arche_memory.sqlite3")
    )
    if not candidates:
        pytest.skip("本地未提供 Case3 原始 SQLite 证据")
    source_db = None
    for candidate in candidates:
        with sqlite3.connect(f"file:{candidate}?mode=ro", uri=True) as connection:
            if connection.execute(
                "SELECT 1 FROM memory_events WHERE event_type='investigation_task_closed' LIMIT 1"
            ).fetchone():
                source_db = candidate
                break
    if source_db is None:
        pytest.skip("本地 Case3 SQLite 没有闭合任务样本")
    target_db = tmp_path / "case3-copy.sqlite3"
    shutil.copy2(source_db, target_db)
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as connection:
        task_id = connection.execute("SELECT task_id FROM tasks LIMIT 1").fetchone()[0]
        source_event_count = connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE task_id=?", (task_id,)
        ).fetchone()[0]

    result = InvestigationV2Migration(str(target_db)).migrate_existing_task(
        task_id, source_root=source_db.parents[5]
    )
    service = ResearchTaskWorkingMemory(str(target_db))
    facts = service.read_research_task_facts(task_id)
    assert facts.lifecycle.state == "Closed"
    assert result.source_event_count == source_event_count
    records = service.read_research_task_scientific_records(task_id)
    by_type = {record.scientific_record_type: record for record in records}
    assert (
        by_type["RetrievalRecord"].scientific_record_schema_version
        == "v10-migrated"
    )
    assert (
        by_type["HypothesisBatchRecord"].scientific_record_schema_version
        == "v10-migrated"
    )
    evidence = [
        record
        for record in records
        if record.scientific_record_type == "VerifiedEvidenceRecord"
    ]
    assert evidence
    assert all(
        record.scientific_record_schema_version == "v10-migrated"
        for record in evidence
    )
    assert all(
        "source_manifest_ref" not in record.payload and "summary" not in record.payload
        for record in evidence
    )
    serialized = json.dumps([record.to_dict() for record in records])
    assert "codex_thread_id" not in serialized
    assert "scope_id" not in serialized
    assert "turn_id" not in serialized
