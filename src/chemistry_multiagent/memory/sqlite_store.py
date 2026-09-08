"""提供 Working Memory 使用的 SQLite 连接、事务和数据库升级能力。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional

from .models import SCHEMA_VERSION


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    scientific_question TEXT NOT NULL,
    reaction_class TEXT,
    chemistry_context_json TEXT NOT NULL DEFAULT '{}',
    execution_mode TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    final_outcome TEXT,
    version INTEGER NOT NULL DEFAULT 0,
    stage_architecture_version TEXT NOT NULL DEFAULT 'legacy_workflow_v1'
        CHECK(stage_architecture_version IN ('legacy_workflow_v1', 'investigation_v2')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    event_key TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    round INTEGER NOT NULL DEFAULT 0,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_task_round ON memory_events(task_id, round);
CREATE INDEX IF NOT EXISTS idx_events_type ON memory_events(event_type);

CREATE TABLE IF NOT EXISTS hypothesis_batches (
    batch_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_index INTEGER NOT NULL CHECK(batch_index >= 1),
    state TEXT NOT NULL CHECK(state IN ('running', 'succeeded', 'failed')),
    selected_count INTEGER NOT NULL CHECK(selected_count >= 1),
    created_event_id TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(task_id, batch_index)
);
CREATE INDEX IF NOT EXISTS idx_hypothesis_batches_task_state
ON hypothesis_batches(task_id, state, batch_index);

CREATE TABLE IF NOT EXISTS combinations (
    combination_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL REFERENCES hypothesis_batches(batch_id) ON DELETE CASCADE,
    combination_index INTEGER NOT NULL CHECK(combination_index >= 1),
    hypothesis_id TEXT NOT NULL,
    workflow_id TEXT,
    workflow_version INTEGER NOT NULL DEFAULT 0 CHECK(workflow_version >= 0),
    attempt_index INTEGER NOT NULL DEFAULT 0 CHECK(attempt_index >= 0),
    execution_id TEXT,
    state TEXT NOT NULL CHECK(state IN (
        'generating_workflow', 'running', 'waiting_execution', 'routing',
        'revising_workflow', 'paused', 'evidence_ready', 'hypothesis_failed',
        'tool_capability_gap'
    )),
    current_workflow_json TEXT,
    runtime_checkpoint_json TEXT NOT NULL DEFAULT '{}',
    terminal_record_id TEXT,
    terminal_reason TEXT,
    final_result_relative_path TEXT,
    causal_source_ref TEXT,
    -- investigation_v2 使用独立阶段字段；旧 state 仍供 legacy 读取兼容。
    investigation_phase TEXT CHECK(investigation_phase IN (
        'pending', 'investigating', 'repair_pending', 'validated',
        'marked_for_hypothesis_revision'
    )),
    -- v8 的可验证状态单独保存，避免改写旧 investigation_phase 的 CHECK 约束。
    investigation_ready_state TEXT NOT NULL DEFAULT 'pending' CHECK(
        investigation_ready_state IN (
            'pending', 'investigating', 'ready_for_batch_validation',
            'repair_pending', 'validated', 'marked_for_hypothesis_revision'
        )
    ),
    -- v8 为新阶段提供独立的组合身份状态；legacy 仍只依赖旧 state 字段。
    investigation_stage_state TEXT CHECK(investigation_stage_state IN (
        'investigation_pending', 'investigating', 'ready_for_batch_validation',
        'repair_pending', 'validated', 'marked_for_hypothesis_revision'
    )),
    version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(batch_id, combination_index),
    UNIQUE(batch_id, hypothesis_id),
    UNIQUE(task_id, execution_id)
);
CREATE INDEX IF NOT EXISTS idx_combinations_task_state
ON combinations(task_id, state, batch_id, combination_index);

CREATE TABLE IF NOT EXISTS hypothesis_candidates (
    hypothesis_id TEXT NOT NULL,
    batch_id TEXT NOT NULL REFERENCES hypothesis_batches(batch_id) ON DELETE CASCADE,
    candidate_index INTEGER NOT NULL CHECK(candidate_index >= 1),
    rank INTEGER NOT NULL CHECK(rank >= 1),
    confidence REAL CHECK(
        confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)
    ),
    selection_status TEXT NOT NULL CHECK(selection_status IN ('selected', 'not_selected')),
    combination_id TEXT REFERENCES combinations(combination_id) ON DELETE RESTRICT,
    PRIMARY KEY(batch_id, hypothesis_id),
    UNIQUE(batch_id, candidate_index)
);
CREATE INDEX IF NOT EXISTS idx_hypothesis_candidates_batch_selection
ON hypothesis_candidates(batch_id, selection_status, candidate_index);

CREATE TABLE IF NOT EXISTS problems (
    problem_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    round INTEGER NOT NULL,
    problem_code TEXT NOT NULL,
    problem_type TEXT,
    job_type TEXT,
    tool_name TEXT,
    failure_signature TEXT,
    severity TEXT,
    problem_signature TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_problems_signature ON problems(problem_signature);
CREATE INDEX IF NOT EXISTS idx_problems_code ON problems(problem_code);
CREATE INDEX IF NOT EXISTS idx_problems_structured
ON problems(problem_code, job_type, tool_name, failure_signature);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    round INTEGER NOT NULL,
    route TEXT NOT NULL,
    decision_source TEXT NOT NULL,
    reasoning TEXT,
    confidence REAL,
    problem_ids_json TEXT NOT NULL DEFAULT '[]',
    memory_used_json TEXT NOT NULL DEFAULT '[]',
    constraints_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS revisions (
    revision_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    decision_id TEXT REFERENCES decisions(decision_id) ON DELETE SET NULL,
    round INTEGER NOT NULL,
    revision_type TEXT NOT NULL,
    instructions_json TEXT NOT NULL DEFAULT '[]',
    target_problem_codes_json TEXT NOT NULL DEFAULT '[]',
    generation_status TEXT NOT NULL,
    verification_status TEXT NOT NULL DEFAULT 'pending',
    applied_round INTEGER,
    revised_result_json TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revisions_pending
ON revisions(task_id, verification_status, round);

CREATE TABLE IF NOT EXISTS revision_verifications (
    verification_id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL REFERENCES revisions(revision_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    round INTEGER NOT NULL,
    status TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    source_task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id) ON DELETE CASCADE,
    reaction_class TEXT,
    problem_signature TEXT,
    decision_taken TEXT,
    effective_fixes_json TEXT NOT NULL DEFAULT '[]',
    ineffective_fixes_json TEXT NOT NULL DEFAULT '[]',
    final_outcome TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_reaction ON cases(reaction_class);
CREATE INDEX IF NOT EXISTS idx_cases_problem ON cases(problem_signature);

CREATE TABLE IF NOT EXISTS tool_observations (
    observation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    round INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    tool_family TEXT,
    calculation_type TEXT,
    route_keywords TEXT,
    failure_signature TEXT,
    execution_mode TEXT,
    success INTEGER NOT NULL,
    validation_status TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_obs_fingerprint
ON tool_observations(tool_name, calculation_type, failure_signature, execution_mode);

CREATE TABLE IF NOT EXISTS tool_pattern_candidates (
    candidate_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    tool_name TEXT NOT NULL,
    calculation_type TEXT,
    failure_signature TEXT,
    observation_count INTEGER NOT NULL,
    task_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL DEFAULT 0,
    validation_pass_count INTEGER NOT NULL,
    pattern_kind TEXT NOT NULL DEFAULT 'success',
    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_patterns (
    pattern_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    tool_name TEXT NOT NULL,
    calculation_type TEXT,
    failure_signature TEXT,
    pattern_kind TEXT NOT NULL DEFAULT 'success',
    recommended_fix TEXT,
    observation_count INTEGER NOT NULL,
    task_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    validation_pass_count INTEGER NOT NULL DEFAULT 0,
    evidence_status TEXT NOT NULL,
    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Investigation 事实先写入 memory_events；以下表只是从事件重建的当前索引。
-- thread/workspace/scope 是恢复所需的内部身份，默认查询接口不会将其交给模型。
CREATE TABLE IF NOT EXISTS investigation_sessions (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    combination_id TEXT NOT NULL,
    codex_thread_id TEXT,
    workspace_ref TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    current_result_version TEXT,
    repair_count INTEGER NOT NULL DEFAULT 0 CHECK(repair_count BETWEEN 0 AND 3),
    phase TEXT NOT NULL DEFAULT 'created' CHECK(phase IN (
        'created', 'investigating', 'ready_for_batch_validation', 'repair_pending',
        'marked_for_hypothesis_revision', 'validated'
    )),
    route_marker TEXT CHECK(route_marker IS NULL OR route_marker='revise_hypothesis'),
    session_event_id TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    latest_result_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    latest_review_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    pending_repair_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    current_turn_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    current_turn_id TEXT,
    current_turn_attempt_number INTEGER,
    current_repair_turn_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    current_repair_turn_id TEXT,
    version INTEGER NOT NULL DEFAULT 0 CHECK(version >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(task_id, batch_id, combination_id),
    UNIQUE(session_id),
    UNIQUE(session_event_id)
);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_task_batch
ON investigation_sessions(task_id, batch_id, combination_id);

-- Investigation turn 是组合首轮实际进入 in-flight 的恢复游标，不保存 prompt 或 transcript。
CREATE TABLE IF NOT EXISTS investigation_turns (
    turn_started_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    combination_id TEXT NOT NULL,
    codex_thread_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    operation_instance_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 1 CHECK(attempt_number >= 1),
    attempt_status TEXT NOT NULL DEFAULT 'in_flight' CHECK(
        attempt_status IN ('in_flight', 'resumable', 'succeeded', 'terminal_failed', 'terminal_interrupted')
    ),
    logical_operation_instance_id TEXT NOT NULL,
    previous_attempt_closure_event_ref TEXT REFERENCES memory_events(event_id)
        ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, session_id, turn_id),
    UNIQUE(task_id, session_id, attempt_number)
);
CREATE INDEX IF NOT EXISTS idx_investigation_turns_task_batch
ON investigation_turns(task_id, batch_id, combination_id, created_at);

CREATE TABLE IF NOT EXISTS investigation_results (
    result_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL,
    combination_id TEXT NOT NULL,
    result_version TEXT NOT NULL,
    codex_thread_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    self_judgment TEXT NOT NULL CHECK(self_judgment IN ('supported', 'refuted')),
    narrative TEXT NOT NULL,
    job_refs_json TEXT NOT NULL DEFAULT '[]',
    operation_refs_json TEXT NOT NULL DEFAULT '[]',
    artifact_refs_json TEXT NOT NULL DEFAULT '[]',
    observations_json TEXT NOT NULL DEFAULT '[]',
    limitations_json TEXT NOT NULL DEFAULT '[]',
    evidence_verification_event_ref TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    repair_turn_event_ref TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    operation_instance_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, batch_id, combination_id, result_version)
);
CREATE INDEX IF NOT EXISTS idx_investigation_results_task_batch
ON investigation_results(task_id, batch_id, combination_id, result_version);

CREATE TABLE IF NOT EXISTS investigation_reviews (
    review_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL,
    combination_ids_json TEXT NOT NULL,
    result_versions_json TEXT NOT NULL,
    result_version_vector_json TEXT NOT NULL,
    findings_json TEXT NOT NULL,
    can_answer INTEGER NOT NULL CHECK(can_answer IN (0, 1)),
    final_answer TEXT,
    answerability_explanation TEXT,
    review_thread_id TEXT NOT NULL,
    review_turn_id TEXT NOT NULL,
    operation_instance_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, batch_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_investigation_reviews_task_batch
ON investigation_reviews(task_id, batch_id, created_at, review_event_id);

CREATE TABLE IF NOT EXISTS investigation_repairs (
    repair_request_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL,
    combination_id TEXT NOT NULL,
    result_version TEXT NOT NULL,
    repair_count INTEGER NOT NULL CHECK(repair_count BETWEEN 1 AND 3),
    repair_advice TEXT NOT NULL,
    review_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    completion_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    result_event_ref TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    operation_instance_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(task_id, batch_id, combination_id, repair_count)
);
CREATE INDEX IF NOT EXISTS idx_investigation_repairs_pending
ON investigation_repairs(task_id, batch_id, combination_id, completion_event_id);

CREATE TABLE IF NOT EXISTS investigation_route_markers (
    marker_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL,
    combination_id TEXT NOT NULL,
    marker TEXT NOT NULL CHECK(marker='revise_hypothesis'),
    result_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    review_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, batch_id, combination_id, marker)
);
CREATE INDEX IF NOT EXISTS idx_investigation_routes_task_batch
ON investigation_route_markers(task_id, batch_id, combination_id);

-- 上层证据适配器先提交这一份安全摘要；Result 只能引用已验证的事件，
-- 不把 MiniChem receipt、token、私有路径或 transcript 写入公共当前视图。
CREATE TABLE IF NOT EXISTS investigation_evidence_verifications (
    verification_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL,
    combination_id TEXT NOT NULL,
    verification_object_id TEXT NOT NULL,
    verifier_id TEXT NOT NULL,
    verifier_version TEXT NOT NULL,
    reference_binding_version TEXT,
    verification_digest TEXT NOT NULL,
    codex_thread_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    jobs_json TEXT NOT NULL,
    operations_json TEXT NOT NULL,
    artifacts_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, verification_object_id)
);
CREATE INDEX IF NOT EXISTS idx_investigation_evidence_task_combination
ON investigation_evidence_verifications(task_id, batch_id, combination_id, created_at);

-- 每个 task 只有一个 Review 会话；它冻结 Review provider 和恢复所需的安全身份。
CREATE TABLE IF NOT EXISTS task_review_sessions (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    session_event_id TEXT NOT NULL UNIQUE REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    provider_id TEXT NOT NULL,
    model TEXT NOT NULL,
    sandbox_policy_digest TEXT NOT NULL,
    workspace_ref TEXT NOT NULL,
    codex_home_ref TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    codex_thread_id TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0 CHECK(turn_count >= 0),
    active_batch_id TEXT,
    active_turn_id TEXT,
    active_turn_started_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    last_batch_id TEXT,
    last_turn_id TEXT,
    last_turn_started_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    last_review_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    phase TEXT NOT NULL DEFAULT 'created' CHECK(
        phase IN ('created', 'turn_in_flight', 'ready', 'blocked')
    ),
    version INTEGER NOT NULL DEFAULT 0 CHECK(version >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_review_turns (
    turn_started_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    session_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    batch_id TEXT NOT NULL,
    review_thread_id TEXT NOT NULL,
    review_turn_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL CHECK(turn_index >= 1),
    request_digest TEXT NOT NULL,
    operation_instance_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 1 CHECK(attempt_number >= 1),
    attempt_status TEXT NOT NULL DEFAULT 'in_flight' CHECK(
        attempt_status IN ('in_flight', 'succeeded', 'terminal_failed', 'terminal_interrupted')
    ),
    logical_operation_instance_id TEXT NOT NULL,
    previous_attempt_closure_event_ref TEXT REFERENCES memory_events(event_id)
        ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, logical_operation_instance_id, attempt_number),
    UNIQUE(task_id, turn_index, attempt_number),
    UNIQUE(task_id, review_turn_id)
);
CREATE INDEX IF NOT EXISTS idx_task_review_turns_task_batch
ON task_review_turns(task_id, batch_id, turn_index);

-- 返修 request 与真正开始的 deepseek_repair turn 分开保存；只有引用该 turn 的
-- Result 成功登记后才消耗 repair_count。
CREATE TABLE IF NOT EXISTS investigation_repair_turns (
    turn_started_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL,
    combination_id TEXT NOT NULL,
    request_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    repair_request_number INTEGER NOT NULL CHECK(repair_request_number BETWEEN 1 AND 3),
    codex_thread_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    operation_instance_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 1 CHECK(attempt_number >= 1),
    attempt_status TEXT NOT NULL DEFAULT 'in_flight' CHECK(
        attempt_status IN ('in_flight', 'resumable', 'succeeded', 'terminal_failed', 'terminal_interrupted')
    ),
    logical_operation_instance_id TEXT NOT NULL,
    previous_attempt_closure_event_ref TEXT REFERENCES memory_events(event_id)
        ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE(request_event_ref, attempt_number),
    UNIQUE(task_id, batch_id, combination_id, turn_id)
);
CREATE INDEX IF NOT EXISTS idx_investigation_repair_turns_pending
ON investigation_repair_turns(task_id, batch_id, combination_id, repair_request_number);

-- Turn closure 是 immutable 的终止事实；Start 行和历史 closure 永不删除。
CREATE TABLE IF NOT EXISTS investigation_turn_attempt_terminations (
    termination_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    turn_started_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK(kind IN ('initial', 'repair', 'review')),
    batch_id TEXT NOT NULL,
    combination_id TEXT,
    turn_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
    terminal_status TEXT NOT NULL CHECK(terminal_status IN ('failed', 'interrupted')),
    logical_operation_instance_id TEXT NOT NULL,
    attempt_operation_instance_id TEXT NOT NULL,
    previous_attempt_closure_event_ref TEXT REFERENCES memory_events(event_id)
        ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, turn_started_event_ref)
);
CREATE INDEX IF NOT EXISTS idx_investigation_turn_attempt_terminations_task
ON investigation_turn_attempt_terminations(task_id, batch_id, combination_id, kind, created_at);

-- Investigation 任务终态只保存由事件重建的回答性和关闭结果索引。
CREATE TABLE IF NOT EXISTS investigation_task_terminals (
    terminal_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL,
    accepted INTEGER NOT NULL CHECK(accepted IN (0, 1)),
    can_answer INTEGER NOT NULL CHECK(can_answer IN (0, 1)),
    final_answer TEXT,
    answerability_explanation TEXT,
    review_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
    marker_event_refs_json TEXT NOT NULL,
    operation_instance_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id)
);
CREATE INDEX IF NOT EXISTS idx_investigation_terminals_task_batch
ON investigation_task_terminals(task_id, batch_id, created_at);
"""


class SQLiteMemoryStore:
    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        journal_mode = os.environ.get("ARCHE_MEMORY_JOURNAL_MODE", "DELETE").strip() or "DELETE"
        conn.execute(f"PRAGMA journal_mode = {journal_mode}")
        return conn

    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def immediate_transaction(self):
        """在写业务状态前先取得SQLite写锁，并在异常时完整回滚。

        提前取得写锁让版本比较和实际更新处于同一个串行区间，避免两个
        Controller 同时认为自己写入成功。
        """

        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _execute_sql_script(conn: sqlite3.Connection, script: str) -> None:
        """在当前事务内逐条执行 SQL 脚本，避免 executescript 隐式提交。

        输入是只包含 DDL/DML 语句的内部脚本；动作按 SQLite 的完整语句边界
        逐条调用 ``execute``，因此不会像 ``executescript`` 一样先提交调用方
        已开启的事务。语句执行失败直接抛出异常，事务回滚交由调用方处理。
        """

        statement = ""
        for line in script.splitlines(keepends=True):
            statement += line
            if not sqlite3.complete_statement(statement):
                continue
            sql = statement.strip()
            if sql:
                conn.execute(sql)
            statement = ""
        if statement.strip():
            conn.execute(statement)

    def initialize(self) -> None:
        conn = self.connect()
        try:
            existing_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            meta_exists = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='schema_meta'"
            ).fetchone()
            if existing_tables and meta_exists is None:
                raise RuntimeError(
                    "发现没有版本标记的旧 Working Memory 数据库；"
                    "请先运行 release audit 并处理旧任务"
                )
            existing_version = None
            if meta_exists is not None:
                row = conn.execute(
                    "SELECT value FROM schema_meta WHERE key='schema_version'"
                ).fetchone()
                existing_version = row[0] if row is not None else None
                if existing_version not in {"5", "6", "7", "8", "9", str(SCHEMA_VERSION)}:
                    raise RuntimeError(
                        "Working Memory schema 与当前 Tool Catalog 协议不兼容；"
                        "请先运行 release audit 并处理旧任务"
                    )
            # 所有建表、迁移、历史重建和 schema_meta 更新都必须处于同一写事务。
            # ``executescript`` 会隐式提交已有事务，不能用于这里的初始化路径。
            conn.execute("BEGIN IMMEDIATE")
            self._execute_sql_script(conn, SCHEMA_SQL)
            if existing_version == "5":
                self._migrate_v5_candidates(conn)
            if existing_version in {"5", "6"}:
                self._migrate_legacy_to_v7_investigation(conn)
            if existing_version in {"5", "6", "7", "8"}:
                self._migrate_v7_to_v8_investigation(conn)
            if existing_version in {"5", "6", "7", "8"}:
                self._migrate_v8_to_v9_investigation(conn)
            if existing_version in {"5", "6", "7", "8", "9"}:
                self._migrate_v9_to_v10_investigation(conn)
            else:
                # 新数据库和已是 v10 的数据库不需要旧表迁移，但仍需确保
                # waiting 历史表存在；它不能在旧 schema_meta 迁移前由 SCHEMA_SQL
                # 提前创建，否则失败回滚会留下半升级结构。
                self._ensure_v10_investigation_wait_table(conn)
            self._ensure_v10_evidence_binding_column(conn)
            if existing_version == "9":
                self._rebuild_v10_task_projections(conn)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
        except BaseException:
            # 迁移失败时撤销尚未提交的 DDL/DML；v9→v10 迁移自身也使用同一
            # 事务边界，避免 schema_meta 仍为旧版本而表结构已经半升级。
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _rebuild_v10_task_projections(conn: sqlite3.Connection) -> None:
        """首次打开 v9 数据库时按不可变事件恢复全部任务当前索引。

        输入是已经完成 v9→v10 表结构迁移、仍处于同一事务的 SQLite 连接；动作只
        对数据库中已有任务逐一删除并重放其 ``memory_events``，明确存在的 waiting
        事件会建立 waiting 索引，没有该事件的 interrupted attempt 保持原状态。
        不读取运行目录或外部作业，也不凭当前状态猜测 waiting；任一事件无法重放
        时抛出异常，由初始化事务统一回滚。
        """

        from .projections import rebuild_task_projections

        task_ids = [
            row["task_id"]
            for row in conn.execute(
                "SELECT task_id FROM tasks ORDER BY task_id"
            ).fetchall()
        ]
        for task_id in task_ids:
            rebuild_task_projections(conn, task_id)

    @staticmethod
    def _migrate_v5_candidates(conn: sqlite3.Connection) -> None:
        """保留旧候选记录，同时允许新任务不再写入虚构置信度。"""

        columns = {
            row["name"]: row
            for row in conn.execute("PRAGMA table_info(hypothesis_candidates)")
        }
        confidence = columns.get("confidence")
        if confidence is None:
            raise RuntimeError("旧 Working Memory 缺少 hypothesis_candidates.confidence")
        if not confidence["notnull"]:
            return
        SQLiteMemoryStore._execute_sql_script(
            conn,
            """
            ALTER TABLE hypothesis_candidates RENAME TO hypothesis_candidates_v5;
            CREATE TABLE hypothesis_candidates (
                hypothesis_id TEXT NOT NULL,
                batch_id TEXT NOT NULL REFERENCES hypothesis_batches(batch_id) ON DELETE CASCADE,
                candidate_index INTEGER NOT NULL CHECK(candidate_index >= 1),
                rank INTEGER NOT NULL CHECK(rank >= 1),
                confidence REAL CHECK(
                    confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)
                ),
                selection_status TEXT NOT NULL CHECK(selection_status IN ('selected', 'not_selected')),
                combination_id TEXT REFERENCES combinations(combination_id) ON DELETE RESTRICT,
                PRIMARY KEY(batch_id, hypothesis_id),
                UNIQUE(batch_id, candidate_index)
            );
            INSERT INTO hypothesis_candidates(
                hypothesis_id, batch_id, candidate_index, rank, confidence,
                selection_status, combination_id
            )
            SELECT hypothesis_id, batch_id, candidate_index, rank, confidence,
                   selection_status, combination_id
            FROM hypothesis_candidates_v5;
            DROP TABLE hypothesis_candidates_v5;
            CREATE INDEX idx_hypothesis_candidates_batch_selection
            ON hypothesis_candidates(batch_id, selection_status, candidate_index);
            """,
        )

    @staticmethod
    def _migrate_legacy_to_v7_investigation(conn: sqlite3.Connection) -> None:
        """把 v5/v6 任务当前索引补成 legacy，并保留所有旧事件原文。"""

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(tasks)")
        }
        if "stage_architecture_version" not in columns:
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN stage_architecture_version TEXT "
                "NOT NULL DEFAULT 'legacy_workflow_v1'"
            )
        combination_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(combinations)")
        }
        if "investigation_phase" not in combination_columns:
            conn.execute(
                "ALTER TABLE combinations ADD COLUMN investigation_phase TEXT "
                "CHECK(investigation_phase IN ("
                "'pending','investigating','repair_pending','validated',"
                "'marked_for_hypothesis_revision'))"
            )
        # 旧行即使曾被手工写入空值，也必须固定到合法 legacy 架构；不重写事件正文。
        conn.execute(
            "UPDATE tasks SET stage_architecture_version='legacy_workflow_v1' "
            "WHERE stage_architecture_version IS NULL OR "
            "stage_architecture_version NOT IN ('legacy_workflow_v1','investigation_v2')"
        )

    @staticmethod
    def _migrate_v7_to_v8_investigation(conn: sqlite3.Connection) -> None:
        """为 v7 数据增加 v8 当前状态列，不创建虚假的新科学事实。

        输入是已经执行 v7 schema 的连接；动作只补充可由旧事件重建的状态列和索引表，
        历史事件正文、旧任务状态以及旧返修记录均保持不变。返回值为空；数据库结构
        不完整或旧行无法保守映射时抛出 RuntimeError，让恢复停在可诊断状态。
        """

        SQLiteMemoryStore._ensure_v8_combination_state_columns(conn)
        SQLiteMemoryStore._ensure_v8_result_reference_columns(conn)
        session_columns = list(conn.execute("PRAGMA table_info(investigation_sessions)"))
        session_names = {row["name"] for row in session_columns}
        if SQLiteMemoryStore._requires_v8_session_rebuild(conn, session_columns):
            SQLiteMemoryStore._rebuild_v8_investigation_sessions(conn)
        else:
            SQLiteMemoryStore._add_v8_session_columns(conn, session_names)
        SQLiteMemoryStore._backfill_v8_combination_state(conn)

    @staticmethod
    def _migrate_v8_to_v9_investigation(conn: sqlite3.Connection) -> None:
        """把 v8 的 turn 索引迁移到 v9 attempt 生命周期结构。

        输入是已经完成 v5/v6/v7/v8 原迁移的连接；动作只重建三个带旧唯一约束的
        turn 表、补充 session 活动指针和 attempt 字段，并从已有 Result/Review 事件
        标记明确成功的 attempt。旧 Started 事件没有 Result 时保持 ``in_flight``，不
        伪造终止事实；返回值为空，结构或数据不一致时抛出 RuntimeError 并由初始化回滚。
        """

        SQLiteMemoryStore._ensure_v9_session_columns(conn)
        SQLiteMemoryStore._ensure_v9_review_session_columns(conn)
        SQLiteMemoryStore._rebuild_v9_investigation_turns(conn)
        SQLiteMemoryStore._rebuild_v9_review_turns(conn)
        SQLiteMemoryStore._rebuild_v9_repair_turns(conn)
        SQLiteMemoryStore._mark_v9_successful_attempts(conn)
        SQLiteMemoryStore._rebuild_v9_active_pointers(conn)
        SQLiteMemoryStore._recompute_v9_review_session_state(conn)

    @staticmethod
    def _migrate_v9_to_v10_investigation(conn: sqlite3.Connection) -> None:
        """在单一事务中执行 v9→v10 迁移，失败时恢复旧结构。"""

        owns_transaction = not conn.in_transaction
        if owns_transaction:
            conn.execute("BEGIN")
        try:
            SQLiteMemoryStore._migrate_v9_to_v10_investigation_unchecked(conn)
        except BaseException:
            if owns_transaction:
                conn.rollback()
            raise
        else:
            if owns_transaction:
                conn.commit()

    @staticmethod
    def _ensure_v10_investigation_wait_table(conn: sqlite3.Connection) -> None:
        """创建 v10 waiting 历史表及索引，不改写任何历史事件。"""

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS investigation_turn_waits (
                waiting_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                turn_started_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                kind TEXT NOT NULL CHECK(kind IN ('initial', 'repair')),
                batch_id TEXT NOT NULL,
                combination_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                codex_thread_id TEXT NOT NULL,
                workspace_ref_digest TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
                logical_operation_instance_id TEXT NOT NULL,
                attempt_operation_instance_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                submission_operation_id TEXT NOT NULL,
                submission_key TEXT NOT NULL,
                submission_request_digest TEXT NOT NULL,
                sidecar_record_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(task_id, turn_started_event_ref)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_investigation_turn_waits_task_batch
            ON investigation_turn_waits(task_id, batch_id, combination_id, created_at)
            """
        )

    @staticmethod
    def _ensure_v10_evidence_binding_column(conn: sqlite3.Connection) -> None:
        """为 v10 证据索引补充来源链版本列并保留旧 v1 行。

        输入是已经执行基础 schema 和 v9→v10 迁移的 SQLite 连接；动作只在旧表
        缺列时追加可空列。旧证据仍按没有该列的 v1 合同读取，新 producer-job-v2
        证据必须保存版本值；不会改写不可变事件或猜测旧证据来源链。
        """

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(investigation_evidence_verifications)"
            )
        }
        if "reference_binding_version" not in columns:
            conn.execute(
                "ALTER TABLE investigation_evidence_verifications "
                "ADD COLUMN reference_binding_version TEXT"
            )

    @staticmethod
    def _migrate_v9_to_v10_investigation_unchecked(conn: sqlite3.Connection) -> None:
        """把 v9 turn attempt 索引迁移到 v10 的等待回执结构。

        输入是已经完成 v9 迁移的 SQLite 连接；动作只重建缺失或错误 CHECK 约束的
        initial、repair、Review turn 表，并创建独立等待事实表。Review 永远不允许
        ``resumable``，initial/repair 才允许它；旧 Started、closure 和当前指针逐值
        保留，不伪造等待事实。迁移失败由初始化事务整体回滚。
        """

        # v9 使用 terminal 命名，但该引用实际也可以指向 waiting closure；先把
        # 旧列迁移为中性的 closure 名称，之后所有新投影只读 canonical 列。
        for table in (
            "investigation_turns",
            "task_review_turns",
            "investigation_repair_turns",
            "investigation_turn_attempt_terminations",
        ):
            columns = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if (
                "previous_attempt_terminal_event_ref" in columns
                and "previous_attempt_closure_event_ref" not in columns
            ):
                conn.execute(
                    f"ALTER TABLE {table} RENAME COLUMN "
                    "previous_attempt_terminal_event_ref TO previous_attempt_closure_event_ref"
                )
        SQLiteMemoryStore._ensure_v10_investigation_wait_table(conn)
        # 早期开发版的 v10 waiting 表曾把提交操作列命名为 operation_id。
        # v9→v10 迁移只改列名，不改写历史事件正文，读取旧事件时由合同适配器补回
        # canonical submission_operation_id；这样旧 sidecar 的事实仍可被重建。
        waiting_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(investigation_turn_waits)")
        }
        if (
            "operation_id" in waiting_columns
            and "submission_operation_id" not in waiting_columns
        ):
            conn.execute(
                "ALTER TABLE investigation_turn_waits RENAME COLUMN "
                "operation_id TO submission_operation_id"
            )
        waiting_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(investigation_turn_waits)")
        }
        if (
            "workspace_ref" in waiting_columns
            and "workspace_ref_digest" not in waiting_columns
        ):
            conn.execute(
                "ALTER TABLE investigation_turn_waits RENAME COLUMN "
                "workspace_ref TO workspace_ref_digest"
            )
            legacy_rows = conn.execute(
                "SELECT waiting_event_id,workspace_ref_digest "
                "FROM investigation_turn_waits"
            ).fetchall()
            for row in legacy_rows:
                digest = hashlib.sha256(
                    json.dumps(
                        row["workspace_ref_digest"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                conn.execute(
                    "UPDATE investigation_turn_waits SET workspace_ref_digest=? "
                    "WHERE waiting_event_id=?",
                    (digest, row["waiting_event_id"]),
                )
        table_ddls = {
            "investigation_turns",
            "task_review_turns",
            "investigation_repair_turns",
        }
        needs_rebuild = False
        for table in table_ddls:
            ddl_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            ddl = "" if ddl_row is None or ddl_row[0] is None else "".join(ddl_row[0].split()).lower()
            if table == "task_review_turns":
                valid_status = "attempt_statusin('in_flight','succeeded','terminal_failed','terminal_interrupted')"
                status_is_wrong = "resumable" in ddl or valid_status not in ddl
            else:
                valid_status = "attempt_statusin('in_flight','resumable','succeeded','terminal_failed','terminal_interrupted')"
                status_is_wrong = valid_status not in ddl
            if status_is_wrong:
                needs_rebuild = True
                break
        if not needs_rebuild:
            return
        SQLiteMemoryStore._rebuild_v9_investigation_turns(conn)
        SQLiteMemoryStore._rebuild_v9_review_turns(conn)
        SQLiteMemoryStore._rebuild_v9_repair_turns(conn)

    @staticmethod
    def _ensure_v9_session_columns(conn: sqlite3.Connection) -> None:
        """为 Investigation session 增加当前 attempt 编号。"""

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(investigation_sessions)")}
        additions = {
            "current_turn_attempt_number": "INTEGER",
            "current_repair_turn_event_id": (
                "TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT"
            ),
            "current_repair_turn_id": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE investigation_sessions ADD COLUMN {name} {definition}"
                )

    @staticmethod
    def _ensure_v9_review_session_columns(conn: sqlite3.Connection) -> None:
        """为 task Review session 增加活动逻辑 turn 指针。"""

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_review_sessions)")}
        additions = {
            "active_batch_id": "TEXT",
            "active_turn_id": "TEXT",
            "active_turn_started_event_id": (
                "TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT"
            ),
        }
        for name, definition in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE task_review_sessions ADD COLUMN {name} {definition}")

    @staticmethod
    def _drop_v9_turn_indexes(conn: sqlite3.Connection, table: str) -> None:
        """删除旧 turn 表索引，以便 SQLite 重新创建同名索引。"""

        index_names = {
            "investigation_turns": ("idx_investigation_turns_task_batch",),
            "task_review_turns": ("idx_task_review_turns_task_batch",),
            "investigation_repair_turns": ("idx_investigation_repair_turns_pending",),
        }
        for index_name in index_names.get(table, ()):
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")

    @staticmethod
    def _v9_column_expression(
        column_names: set[str],
        column_name: str,
        fallback_sql: str,
    ) -> str:
        """为 v8→v9 重建选择旧列或保守默认值。

        输入是旧表 PRAGMA 列名、目标列名和 SQL 默认表达式；动作只返回可嵌入
        SELECT 的列表达式，不读取业务数据或修改数据库。旧表没有新 attempt 列时
        使用固定默认值，避免迁移伪造终止事实；返回值仅供内部重建 SQL 使用。
        """

        return column_name if column_name in column_names else fallback_sql

    @staticmethod
    def _v9_table_ddl_contains(
        conn: sqlite3.Connection,
        table: str,
        constraint: str,
    ) -> bool:
        """按规范化 DDL 检查 v9 表约束是否存在。

        输入是迁移连接、表名和不含空白的约束片段；动作只读取 sqlite_master，返回
        是否存在该片段，不修改表或数据。缺少表或 DDL 时返回 False，促使迁移重建而
        不是错误地把不完整约束当作 v9；SQL 结构异常由后续重建暴露。
        """

        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row is None or not isinstance(row[0], str):
            return False
        return constraint in "".join(row[0].split()).lower()

    @staticmethod
    def _rebuild_v9_investigation_turns(conn: sqlite3.Connection) -> None:
        """重建 initial turn 表并保守回填旧 v8 attempt 身份。"""

        names = {row["name"] for row in conn.execute("PRAGMA table_info(investigation_turns)")}
        required = {
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        }
        if required.issubset(names) and SQLiteMemoryStore._v9_table_ddl_contains(
            conn,
            "investigation_turns",
            "unique(task_id,session_id,attempt_number)",
        ) and SQLiteMemoryStore._v9_table_ddl_contains(
            conn,
            "investigation_turns",
            "attempt_statusin('in_flight','resumable','succeeded','terminal_failed','terminal_interrupted')",
        ):
            return
        attempt_number = SQLiteMemoryStore._v9_column_expression(names, "attempt_number", "1")
        attempt_status = SQLiteMemoryStore._v9_column_expression(
            names, "attempt_status", "'in_flight'"
        )
        logical_operation = SQLiteMemoryStore._v9_column_expression(
            names, "logical_operation_instance_id", "operation_instance_id"
        )
        previous_closure = SQLiteMemoryStore._v9_column_expression(
            names,
            "previous_attempt_closure_event_ref",
            SQLiteMemoryStore._v9_column_expression(
                names, "previous_attempt_terminal_event_ref", "NULL"
            ),
        )
        SQLiteMemoryStore._drop_v9_turn_indexes(conn, "investigation_turns")
        conn.execute("ALTER TABLE investigation_turns RENAME TO investigation_turns_v8")
        conn.execute(
            """
            CREATE TABLE investigation_turns (
                turn_started_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                session_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                combination_id TEXT NOT NULL,
                codex_thread_id TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                operation_instance_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL DEFAULT 1 CHECK(attempt_number >= 1),
                attempt_status TEXT NOT NULL DEFAULT 'in_flight' CHECK(
                attempt_status IN ('in_flight','resumable','succeeded','terminal_failed','terminal_interrupted')
                ),
                logical_operation_instance_id TEXT NOT NULL,
                previous_attempt_closure_event_ref TEXT REFERENCES memory_events(event_id)
                    ON DELETE RESTRICT,
                created_at TEXT NOT NULL,
                UNIQUE(task_id, session_id, turn_id),
                UNIQUE(task_id, session_id, attempt_number)
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO investigation_turns(
                turn_started_event_id,task_id,session_id,batch_id,combination_id,
                codex_thread_id,scope_id,turn_id,operation_instance_id,attempt_number,
                attempt_status,logical_operation_instance_id,previous_attempt_closure_event_ref,
                created_at
            )
            SELECT turn_started_event_id,task_id,session_id,batch_id,combination_id,
                codex_thread_id,scope_id,turn_id,operation_instance_id,
                {attempt_number},{attempt_status},{logical_operation},
                {previous_closure},created_at
            FROM investigation_turns_v8
            """
        )
        conn.execute("DROP TABLE investigation_turns_v8")
        conn.execute(
            """
            CREATE INDEX idx_investigation_turns_task_batch
            ON investigation_turns(task_id,batch_id,combination_id,created_at);
            """
        )

    @staticmethod
    def _rebuild_v9_review_turns(conn: sqlite3.Connection) -> None:
        """重建 Review turn 表，允许同一逻辑 turn 的 provider replacement。"""

        names = {row["name"] for row in conn.execute("PRAGMA table_info(task_review_turns)")}
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='task_review_turns'"
        ).fetchone()
        if {
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        }.issubset(names) and ddl is not None and "UNIQUE(task_id, turn_index)" not in (ddl[0] or "") and SQLiteMemoryStore._v9_table_ddl_contains(
            conn,
            "task_review_turns",
            "unique(task_id,turn_index,attempt_number)",
        ) and SQLiteMemoryStore._v9_table_ddl_contains(
            conn,
            "task_review_turns",
            "attempt_statusin('in_flight','succeeded','terminal_failed','terminal_interrupted')",
        ) and not SQLiteMemoryStore._v9_table_ddl_contains(
            conn,
            "task_review_turns",
            "attempt_statusin('in_flight','resumable','succeeded','terminal_failed','terminal_interrupted')",
        ):
            return
        attempt_number = SQLiteMemoryStore._v9_column_expression(names, "attempt_number", "1")
        attempt_status = SQLiteMemoryStore._v9_column_expression(
            names, "attempt_status", "'in_flight'"
        )
        logical_operation = SQLiteMemoryStore._v9_column_expression(
            names, "logical_operation_instance_id", "operation_instance_id"
        )
        previous_closure = SQLiteMemoryStore._v9_column_expression(
            names,
            "previous_attempt_closure_event_ref",
            SQLiteMemoryStore._v9_column_expression(
                names, "previous_attempt_terminal_event_ref", "NULL"
            ),
        )
        SQLiteMemoryStore._drop_v9_turn_indexes(conn, "task_review_turns")
        conn.execute("ALTER TABLE task_review_turns RENAME TO task_review_turns_v8")
        conn.execute(
            """
            CREATE TABLE task_review_turns (
                turn_started_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                session_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                batch_id TEXT NOT NULL,
                review_thread_id TEXT NOT NULL,
                review_turn_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL CHECK(turn_index >= 1),
                request_digest TEXT NOT NULL,
                operation_instance_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL DEFAULT 1 CHECK(attempt_number >= 1),
                attempt_status TEXT NOT NULL DEFAULT 'in_flight' CHECK(
                    attempt_status IN ('in_flight','succeeded','terminal_failed','terminal_interrupted')
                ),
                logical_operation_instance_id TEXT NOT NULL,
                previous_attempt_closure_event_ref TEXT REFERENCES memory_events(event_id)
                    ON DELETE RESTRICT,
                created_at TEXT NOT NULL,
                UNIQUE(task_id,logical_operation_instance_id,attempt_number),
                UNIQUE(task_id,turn_index,attempt_number),
                UNIQUE(task_id,review_turn_id)
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO task_review_turns(
                turn_started_event_id,task_id,session_event_ref,batch_id,review_thread_id,
                review_turn_id,turn_index,request_digest,operation_instance_id,attempt_number,
                attempt_status,logical_operation_instance_id,previous_attempt_closure_event_ref,
                created_at
            )
            SELECT turn_started_event_id,task_id,session_event_ref,batch_id,review_thread_id,
                review_turn_id,turn_index,request_digest,operation_instance_id,
                {attempt_number},{attempt_status},{logical_operation},
                {previous_closure},created_at
            FROM task_review_turns_v8
            """
        )
        conn.execute("DROP TABLE task_review_turns_v8")
        conn.execute(
            """
            CREATE INDEX idx_task_review_turns_task_batch
            ON task_review_turns(task_id,batch_id,turn_index);
            """
        )

    @staticmethod
    def _rebuild_v9_repair_turns(conn: sqlite3.Connection) -> None:
        """重建 repair turn 表，允许同一 request 的多次 provider attempt。"""

        names = {row["name"] for row in conn.execute("PRAGMA table_info(investigation_repair_turns)")}
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='investigation_repair_turns'"
        ).fetchone()
        if {
            "attempt_number",
            "attempt_status",
            "logical_operation_instance_id",
            "previous_attempt_closure_event_ref",
        }.issubset(names) and ddl is not None and "UNIQUE(request_event_ref)" not in (ddl[0] or "") and SQLiteMemoryStore._v9_table_ddl_contains(
            conn,
            "investigation_repair_turns",
            "unique(request_event_ref,attempt_number)",
        ) and SQLiteMemoryStore._v9_table_ddl_contains(
            conn,
            "investigation_repair_turns",
            "attempt_statusin('in_flight','resumable','succeeded','terminal_failed','terminal_interrupted')",
        ):
            return
        attempt_number = SQLiteMemoryStore._v9_column_expression(names, "attempt_number", "1")
        attempt_status = SQLiteMemoryStore._v9_column_expression(
            names, "attempt_status", "'in_flight'"
        )
        logical_operation = SQLiteMemoryStore._v9_column_expression(
            names, "logical_operation_instance_id", "operation_instance_id"
        )
        previous_closure = SQLiteMemoryStore._v9_column_expression(
            names,
            "previous_attempt_closure_event_ref",
            SQLiteMemoryStore._v9_column_expression(
                names, "previous_attempt_terminal_event_ref", "NULL"
            ),
        )
        SQLiteMemoryStore._drop_v9_turn_indexes(conn, "investigation_repair_turns")
        conn.execute(
            "ALTER TABLE investigation_repair_turns RENAME TO investigation_repair_turns_v8"
        )
        conn.execute(
            """
            CREATE TABLE investigation_repair_turns (
                turn_started_event_id TEXT PRIMARY KEY REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                batch_id TEXT NOT NULL,
                combination_id TEXT NOT NULL,
                request_event_ref TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                repair_request_number INTEGER NOT NULL CHECK(repair_request_number BETWEEN 1 AND 3),
                codex_thread_id TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                operation_instance_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL DEFAULT 1 CHECK(attempt_number >= 1),
                attempt_status TEXT NOT NULL DEFAULT 'in_flight' CHECK(
                    attempt_status IN ('in_flight','resumable','succeeded','terminal_failed','terminal_interrupted')
                ),
                logical_operation_instance_id TEXT NOT NULL,
                previous_attempt_closure_event_ref TEXT REFERENCES memory_events(event_id)
                    ON DELETE RESTRICT,
                created_at TEXT NOT NULL,
                UNIQUE(request_event_ref,attempt_number),
                UNIQUE(task_id,batch_id,combination_id,turn_id)
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO investigation_repair_turns(
                turn_started_event_id,task_id,batch_id,combination_id,request_event_ref,
                repair_request_number,codex_thread_id,scope_id,turn_id,request_digest,
                operation_instance_id,attempt_number,attempt_status,logical_operation_instance_id,
                previous_attempt_closure_event_ref,created_at
            )
            SELECT turn_started_event_id,task_id,batch_id,combination_id,request_event_ref,
                repair_request_number,codex_thread_id,scope_id,turn_id,request_digest,
                operation_instance_id,{attempt_number},{attempt_status},{logical_operation},
                {previous_closure},created_at
            FROM investigation_repair_turns_v8
            """
        )
        conn.execute("DROP TABLE investigation_repair_turns_v8")
        conn.execute(
            """
            CREATE INDEX idx_investigation_repair_turns_pending
            ON investigation_repair_turns(task_id,batch_id,combination_id,repair_request_number);
            """
        )

    @staticmethod
    def _mark_v9_successful_attempts(conn: sqlite3.Connection) -> None:
        """按已有 Result/Review 明确消费关系回填 succeeded，不推断 terminal。"""

        conn.execute(
            "UPDATE investigation_turns SET attempt_status='succeeded' "
            "WHERE EXISTS ("
            "SELECT 1 FROM investigation_results AS result "
            "WHERE result.task_id=investigation_turns.task_id "
            "AND result.batch_id=investigation_turns.batch_id "
            "AND result.combination_id=investigation_turns.combination_id "
            "AND result.turn_id=investigation_turns.turn_id "
            "AND result.repair_turn_event_ref IS NULL"
            ")"
        )
        conn.execute(
            "UPDATE investigation_repair_turns SET attempt_status='succeeded' "
            "WHERE turn_started_event_id IN ("
            "SELECT repair_turn_event_ref FROM investigation_results "
            "WHERE repair_turn_event_ref IS NOT NULL"
            ")"
        )
        conn.execute(
            "UPDATE task_review_turns SET attempt_status='succeeded' "
            "WHERE turn_started_event_id IN ("
            "SELECT json_extract(payload_json,'$.task_review_turn_event_ref') "
            "FROM memory_events WHERE event_type='investigation_batch_reviewed'"
            ")"
        )

    @staticmethod
    def _rebuild_v9_active_pointers(conn: sqlite3.Connection) -> None:
        """按 v9 状态重建 initial/repair/review 的唯一 in-flight 游标。

        输入是已完成 turn 表重建和成功状态回填的连接；动作只把仍为 in_flight 的
        Started 行回填到 session 活动指针，并清除已消费的旧 v8 指针。多个活动行或
        pending repair 断链时抛出 RuntimeError，不把异常数据猜成可恢复任务；返回值为空。
        """

        sessions = conn.execute("SELECT * FROM investigation_sessions").fetchall()
        for session in sessions:
            initial_rows = conn.execute(
                "SELECT turn_started_event_id,turn_id,attempt_number,attempt_status "
                "FROM investigation_turns WHERE task_id=? AND batch_id=? AND combination_id=? "
                "AND attempt_status='in_flight'",
                (session["task_id"], session["batch_id"], session["combination_id"]),
            ).fetchall()
            if len(initial_rows) > 1:
                raise RuntimeError("v9 migration found multiple active initial attempts")
            repair_rows = conn.execute(
                "SELECT turn_started_event_id,turn_id,attempt_status FROM investigation_repair_turns "
                "WHERE task_id=? AND batch_id=? AND combination_id=? "
                "AND request_event_ref IS ? AND attempt_status='in_flight'",
                (
                    session["task_id"],
                    session["batch_id"],
                    session["combination_id"],
                    session["pending_repair_event_id"],
                ),
            ).fetchall()
            if len(repair_rows) > 1:
                raise RuntimeError("v9 migration found multiple active repair attempts")
            initial = initial_rows[0] if initial_rows else None
            repair = repair_rows[0] if repair_rows else None
            conn.execute(
                "UPDATE investigation_sessions SET current_turn_event_id=?,current_turn_id=?,"
                "current_turn_attempt_number=?,current_repair_turn_event_id=?,current_repair_turn_id=? "
                "WHERE task_id=? AND batch_id=? AND combination_id=?",
                (
                    None if initial is None else initial["turn_started_event_id"],
                    None if initial is None else initial["turn_id"],
                    None if initial is None else initial["attempt_number"],
                    None if repair is None else repair["turn_started_event_id"],
                    None if repair is None else repair["turn_id"],
                    session["task_id"],
                    session["batch_id"],
                    session["combination_id"],
                ),
            )

    @staticmethod
    def _recompute_v9_review_session_state(conn: sqlite3.Connection) -> None:
        """按成功 Review 事件重算 turn_count、last_* 和活动 Review 指针。

        输入是已完成 Review turn 状态回填的连接；动作以成功提交的
        ``investigation_batch_reviewed`` 事件为唯一计数来源，分别保存最后成功 Review
        和仍可重放的 in-flight attempt。旧 v8 把 Started 次数写进 turn_count 时被纠正，
        未成功的 Started 不会被伪造为完成事实；返回值为空，断链或多个活动行失败。
        """

        sessions = conn.execute("SELECT * FROM task_review_sessions").fetchall()
        for session in sessions:
            task_id = session["task_id"]
            active = SQLiteMemoryStore._read_v9_active_review_attempt(conn, task_id)
            count, last, last_turn_ref = SQLiteMemoryStore._read_v9_successful_review_state(
                conn, task_id
            )
            conn.execute(
                "UPDATE task_review_sessions SET turn_count=?,last_batch_id=?,last_turn_id=?,"
                "last_turn_started_event_id=?,last_review_event_id=?,active_batch_id=?,"
                "active_turn_id=?,active_turn_started_event_id=?,phase=? WHERE task_id=?",
                (
                    count,
                    None if last is None else last["batch_id"],
                    None if last is None else last["review_turn_id"],
                    last_turn_ref,
                    None if last is None else last["review_event_id"],
                    None if active is None else active["batch_id"],
                    None if active is None else active["review_turn_id"],
                    None if active is None else active["turn_started_event_id"],
                    "turn_in_flight" if active is not None else ("ready" if count else "created"),
                    task_id,
                ),
            )

    @staticmethod
    def _read_v9_active_review_attempt(
        conn: sqlite3.Connection,
        task_id: str,
    ) -> sqlite3.Row | None:
        """读取唯一 Review in-flight attempt，多个活动行时停止迁移。

        输入是迁移事务连接和 task；动作只查询新 turn 表并核对活动唯一性，返回唯一
        active 行或 None。多个 in-flight attempt 表示旧索引断裂，抛出 RuntimeError，
        不猜测应恢复的 Review。
        """

        rows = conn.execute(
            "SELECT * FROM task_review_turns WHERE task_id=? AND attempt_status='in_flight' "
            "ORDER BY turn_index,attempt_number,created_at,turn_started_event_id",
            (task_id,),
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("v9 migration found multiple active Review attempts")
        return rows[0] if rows else None

    @staticmethod
    def _read_v9_successful_review_state(
        conn: sqlite3.Connection,
        task_id: str,
    ) -> tuple[int, sqlite3.Row | None, str | None]:
        """按成功 Review 事实读取计数、最后一行和其 turn 引用。

        输入是迁移事务连接和 task；动作只读取 investigation_reviews 及最后成功事件，
        解析其 task Review turn 引用并返回三元组。事件缺失、正文损坏或没有 turn 引用
        时抛出 RuntimeError；不把 Started 数量当成成功 Review 数。
        """

        rows = conn.execute(
            "SELECT * FROM investigation_reviews WHERE task_id=? "
            "ORDER BY created_at,review_event_id",
            (task_id,),
        ).fetchall()
        if not rows:
            return 0, None, None
        last = rows[-1]
        event = conn.execute(
            "SELECT payload_json FROM memory_events WHERE task_id=? AND event_id=?",
            (task_id, last["review_event_id"]),
        ).fetchone()
        if event is None:
            raise RuntimeError("v9 migration Review event is missing")
        try:
            payload = json.loads(event["payload_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("v9 migration Review payload is corrupted") from error
        turn_ref = payload.get("task_review_turn_event_ref")
        if not isinstance(turn_ref, str) or not turn_ref:
            raise RuntimeError("v9 migration Review lacks turn reference")
        return len(rows), last, turn_ref


    @staticmethod
    def _ensure_v8_combination_state_columns(conn: sqlite3.Connection) -> None:
        """为 combinations 补齐可由旧 phase 映射的 v8 状态列。"""

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(combinations)")}
        if "investigation_ready_state" not in columns:
            conn.execute(
                "ALTER TABLE combinations ADD COLUMN investigation_ready_state TEXT "
                "NOT NULL DEFAULT 'pending' CHECK(investigation_ready_state IN ("
                "'pending','investigating','ready_for_batch_validation',"
                "'repair_pending','validated','marked_for_hypothesis_revision'))"
            )
        if "investigation_stage_state" not in columns:
            conn.execute(
                "ALTER TABLE combinations ADD COLUMN investigation_stage_state TEXT "
                "CHECK(investigation_stage_state IN ("
                "'investigation_pending','investigating','ready_for_batch_validation',"
                "'repair_pending','validated','marked_for_hypothesis_revision'))"
            )

    @staticmethod
    def _ensure_v8_result_reference_columns(conn: sqlite3.Connection) -> None:
        """为 result 当前视图补齐证据和返修 turn 的事件引用列。"""

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(investigation_results)")}
        if "evidence_verification_event_ref" not in columns:
            conn.execute(
                "ALTER TABLE investigation_results ADD COLUMN "
                "evidence_verification_event_ref TEXT REFERENCES memory_events(event_id) "
                "ON DELETE RESTRICT"
            )
        if "repair_turn_event_ref" not in columns:
            conn.execute(
                "ALTER TABLE investigation_results ADD COLUMN "
                "repair_turn_event_ref TEXT REFERENCES memory_events(event_id) "
                "ON DELETE RESTRICT"
            )

    @staticmethod
    def _requires_v8_session_rebuild(
        conn: sqlite3.Connection,
        session_columns: list[sqlite3.Row],
    ) -> bool:
        """判断旧 session 是否缺少 v8 身份字段或 investigating 状态约束。"""

        names = {row["name"] for row in session_columns}
        codex_thread_not_null = any(
            row["name"] == "codex_thread_id" and row["notnull"] for row in session_columns
        )
        ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='investigation_sessions'"
        ).fetchone()
        supports_investigating = ddl_row is not None and "investigating" in (
            ddl_row["sql"] or ""
        )
        return "session_id" not in names or codex_thread_not_null or not supports_investigating

    @staticmethod
    def _rebuild_v8_investigation_sessions(conn: sqlite3.Connection) -> None:
        """重建旧 session 表并保留其业务字段与事件引用。"""

        SQLiteMemoryStore._execute_sql_script(
            conn,
            """
            ALTER TABLE investigation_sessions RENAME TO investigation_sessions_v7;
            CREATE TABLE investigation_sessions (
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                session_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                combination_id TEXT NOT NULL,
                codex_thread_id TEXT,
                workspace_ref TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                current_result_version TEXT,
                repair_count INTEGER NOT NULL DEFAULT 0 CHECK(repair_count BETWEEN 0 AND 3),
                phase TEXT NOT NULL DEFAULT 'created' CHECK(phase IN (
                    'created', 'investigating', 'ready_for_batch_validation',
                    'repair_pending', 'marked_for_hypothesis_revision', 'validated'
                )),
                route_marker TEXT CHECK(route_marker IS NULL OR route_marker='revise_hypothesis'),
                session_event_id TEXT NOT NULL REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                latest_result_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                latest_review_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                pending_repair_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                current_turn_event_id TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT,
                current_turn_id TEXT,
                version INTEGER NOT NULL DEFAULT 0 CHECK(version >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_id, batch_id, combination_id),
                UNIQUE(session_id),
                UNIQUE(session_event_id)
            );
            INSERT INTO investigation_sessions(
                task_id,session_id,batch_id,combination_id,codex_thread_id,workspace_ref,
                scope_id,current_result_version,repair_count,phase,route_marker,
                session_event_id,latest_result_event_id,latest_review_event_id,
                pending_repair_event_id,version,created_at,updated_at
            )
            SELECT task_id,
                'session:' || task_id || ':' || batch_id || ':' || combination_id,
                batch_id,combination_id,codex_thread_id,workspace_ref,scope_id,
                current_result_version,repair_count,phase,route_marker,session_event_id,
                latest_result_event_id,latest_review_event_id,pending_repair_event_id,
                version,created_at,updated_at
            FROM investigation_sessions_v7;
            DROP TABLE investigation_sessions_v7;
            CREATE INDEX idx_investigation_sessions_task_batch
            ON investigation_sessions(task_id, batch_id, combination_id);
            """,
        )

    @staticmethod
    def _add_v8_session_columns(
        conn: sqlite3.Connection,
        session_names: set[str],
    ) -> None:
        """为已兼容的 session 表幂等补上当前 turn 字段。"""

        if "current_turn_event_id" not in session_names:
            conn.execute(
                "ALTER TABLE investigation_sessions ADD COLUMN current_turn_event_id "
                "TEXT REFERENCES memory_events(event_id) ON DELETE RESTRICT"
            )
        if "current_turn_id" not in session_names:
            conn.execute(
                "ALTER TABLE investigation_sessions ADD COLUMN current_turn_id TEXT"
            )

    @staticmethod
    def _backfill_v8_combination_state(conn: sqlite3.Connection) -> None:
        """把旧 investigation_phase 映射为 v8 当前状态，不生成新事件。"""

        conn.execute(
            "UPDATE combinations SET investigation_ready_state=CASE "
            "WHEN investigation_phase='repair_pending' THEN 'repair_pending' "
            "WHEN investigation_phase='validated' THEN 'validated' "
            "WHEN investigation_phase='marked_for_hypothesis_revision' "
            "THEN 'marked_for_hypothesis_revision' "
            "WHEN investigation_phase='investigating' THEN 'investigating' "
            "ELSE 'pending' END "
            "WHERE investigation_ready_state IS NULL OR investigation_ready_state='pending'"
        )
        conn.execute(
            "UPDATE combinations SET investigation_stage_state=CASE "
            "WHEN investigation_phase='repair_pending' THEN 'repair_pending' "
            "WHEN investigation_phase='validated' THEN 'validated' "
            "WHEN investigation_phase='marked_for_hypothesis_revision' "
            "THEN 'marked_for_hypothesis_revision' "
            "WHEN investigation_phase='investigating' THEN 'investigating' "
            "WHEN investigation_phase='pending' THEN 'investigation_pending' "
            "ELSE NULL END "
            "WHERE investigation_stage_state IS NULL"
        )

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        conn = self.connect()
        try:
            return conn.execute(sql, tuple(params)).fetchone()
        finally:
            conn.close()

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        conn = self.connect()
        try:
            return conn.execute(sql, tuple(params)).fetchall()
        finally:
            conn.close()
