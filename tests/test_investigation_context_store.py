"""Investigation Context、子 Session 与返修 Invocation 的持久合同。"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from chemistry_multiagent.investigation.context_store import (
    InvestigationContextContractError,
    InvestigationContextIdentity,
    InvestigationContextStore,
    InvestigationInvocationIdentity,
    InvestigationSessionIdentity,
)


def _session(
    context_id: str,
    index: int,
) -> InvestigationSessionIdentity:
    return InvestigationSessionIdentity(
        investigation_session_id=f"session-{index}",
        investigation_context_id=context_id,
        research_task_id="research-task-1",
        hypothesis_execution_input_record_id=f"hypothesis-execution-input-{index}",
        workspace_identity=(
            f"internal/state/investigation_runtime/sessions/session-{index}/codex_sessions/"
            f"investigations/research-task-1/{context_id}/hypothesis-execution-input-{index}/workspace"
        ),
        minichem_scope_id=f"scope-{index}",
        scope_contract_digest=str(index) * 64,
        capability_generation_digest=str(index + 1) * 64,
    )


def test_context_reuses_hypothesis_sessions_across_repair_invocations(
    tmp_path: Path,
) -> None:
    """Top 5 Context 可跨新的 repair Invocation 复用原假设 Session。"""

    database = tmp_path / "investigation_context.sqlite3"
    context = InvestigationContextIdentity(
        investigation_context_id="context-1",
        research_task_id="research-task-1",
        hypothesis_batch_record_id="hypothesis-batch-1",
    )
    store = InvestigationContextStore(database)
    store.create_or_load_context(context)
    first_session = store.bind_session(context, _session("context-1", 1))
    second_session = store.bind_session(context, _session("context-1", 2))
    initial = InvestigationInvocationIdentity(
        module_invocation_id="investigation-initial",
        investigation_context_id="context-1",
        module_input_digest="a" * 64,
        input_scientific_record_ids=("hypothesis-selection-1",),
    )
    store.start_invocation(context, initial)
    store.complete_invocation(
        context,
        initial.module_invocation_id,
        module_output_digest="b" * 64,
    )
    repair = InvestigationInvocationIdentity(
        module_invocation_id="investigation-repair-1",
        investigation_context_id="context-1",
        module_input_digest="c" * 64,
        input_scientific_record_ids=("repair-request-1",),
    )
    store.start_invocation(context, repair)

    reopened = InvestigationContextStore(database).read_context(context)

    assert reopened.identity == context
    assert reopened.sessions == (first_session, second_session)
    assert first_session.scope_contract_digest != first_session.capability_generation_digest
    assert second_session.scope_contract_digest != second_session.capability_generation_digest
    assert reopened.closed_invocations == (initial,)
    assert reopened.open_invocation == repair


def test_context_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """恢复不能把未知 Context Store 结构当作当前字段猜测读取。"""

    database = tmp_path / "investigation_context.sqlite3"
    InvestigationContextStore(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE investigation_context_metadata "
            "SET schema_version = 999 WHERE singleton_id = 1"
        )

    with pytest.raises(InvestigationContextContractError, match="schema version"):
        InvestigationContextStore(database)
