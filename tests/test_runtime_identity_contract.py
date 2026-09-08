"""native Investigation/Review/Harness 正式运行时身份合同。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import inspect
import json
from pathlib import Path

import pytest

from chemistry_multiagent.investigation.context_backend import InvestigationContextRunner
from chemistry_multiagent.investigation.context_store import (
    InvestigationContextIdentity,
    InvestigationContextStore,
    InvestigationSessionIdentity,
)
from chemistry_multiagent.investigation.harness_recovery_store import (
    BindSession,
    HarnessRecoveryStore,
    HarnessRunIdentity,
    JobReference,
    ProviderAttemptReference,
    SchemaV1LegacyHarnessRecoveryReader,
    SessionReference,
)
from chemistry_multiagent.investigation.harness_runtime import (
    InvestigationHarnessContractError,
    InvestigationHarnessRequest,
    SchemaV1LegacyHarnessManifestReader,
    SchemaV1LegacyPreparedModuleOutputReader,
    SchemaV1LegacyInvestigationHarnessRequestReader,
    SchemaV10ReadOnlyInvestigationHarnessAdapter,
)
from chemistry_multiagent.investigation.investigation_backend import InvestigationSessionRunner
from chemistry_multiagent.investigation.review_runtime import ResearchTaskReviewRunner
from chemistry_multiagent.controllers.investigation_v2_native_adapters import (
    InvestigationContextModuleAdapter,
    InvestigationSessionHarnessOwner,
    InvestigationSessionOwner,
)


_DIGESTS = tuple(f"{letter}" * 64 for letter in "abcdefgh")


def test_native_identity_dataclasses_use_formal_names() -> None:
    context = InvestigationContextIdentity(
        investigation_context_id="context-formal",
        research_task_id="research-formal",
        hypothesis_batch_record_id="batch-formal",
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-formal",
        investigation_context_id=context.investigation_context_id,
        research_task_id=context.research_task_id,
        hypothesis_execution_input_record_id="execution-input-formal",
        workspace_identity="workspace-formal",
        minichem_scope_id="scope-formal",
        scope_contract_digest=_DIGESTS[0],
    )
    harness = HarnessRunIdentity(
        research_task_id=context.research_task_id,
        module_invocation_id="invocation-formal",
        harness_invocation_id="harness-formal",
        module_input_digest=_DIGESTS[1],
    )
    reference = SessionReference(
        research_task_id=context.research_task_id,
        investigation_context_id=context.investigation_context_id,
        hypothesis_execution_input_record_id=session.hypothesis_execution_input_record_id,
        investigation_session_id=session.investigation_session_id,
        workspace_identity=session.workspace_identity,
        minichem_scope_id=session.minichem_scope_id,
        scope_contract_digest=_DIGESTS[2],
        capability_generation_digest=_DIGESTS[3],
    )

    forbidden = {
        "task_id",
        "batch_id",
        "combination_id",
        "hypothesis_record_id",
        "session_id",
        "runtime_ref",
        "scope_generation_digest",
    }
    for value in (context, session, harness, reference):
        assert not forbidden.intersection(asdict(value))
    assert reference.scope_contract_digest != reference.capability_generation_digest


def test_new_context_store_json_uses_formal_names(tmp_path: Path) -> None:
    database = tmp_path / "context.sqlite3"
    context = InvestigationContextIdentity(
        investigation_context_id="context-json",
        research_task_id="research-json",
        hypothesis_batch_record_id="batch-json",
    )
    store = InvestigationContextStore(database)
    store.create_or_load_context(context)
    session = InvestigationSessionIdentity(
        investigation_session_id="session-json",
        investigation_context_id=context.investigation_context_id,
        research_task_id=context.research_task_id,
        hypothesis_execution_input_record_id="execution-json",
        workspace_identity="workspace-json",
        minichem_scope_id="scope-json",
        scope_contract_digest=_DIGESTS[0],
    )
    store.bind_session(context, session)

    import sqlite3

    with sqlite3.connect(database) as connection:
        values = [
            row[0]
            for row in connection.execute(
                "SELECT identity_json FROM investigation_contexts "
                "UNION ALL SELECT identity_json FROM investigation_sessions"
            )
        ]
    joined = " ".join(values)
    for field in ("batch_id", "combination_id", "hypothesis_record_id", "runtime_ref"):
        assert field not in joined
    assert json.loads(values[0])["research_task_id"] == "research-json"
    assert json.loads(values[1])["workspace_identity"] == "workspace-json"


def test_harness_snapshot_preserves_contract_and_capability_generation_separately(
    tmp_path: Path,
) -> None:
    identity = HarnessRunIdentity(
        research_task_id="research-recovery",
        module_invocation_id="invocation-recovery",
        harness_invocation_id="harness-recovery",
        module_input_digest=_DIGESTS[4],
    )
    reference = SessionReference(
        research_task_id="research-recovery",
        investigation_context_id="context-recovery",
        hypothesis_execution_input_record_id="execution-recovery",
        investigation_session_id="session-recovery",
        workspace_identity="workspace-recovery",
        minichem_scope_id="scope-recovery",
        scope_contract_digest=_DIGESTS[5],
        capability_generation_digest="7" * 64,
    )
    store = HarnessRecoveryStore(tmp_path / "harness.sqlite3")
    store.create_or_load(identity)
    store.append_transition(identity, BindSession("bind-formal", reference))
    snapshot = store.read_snapshot(identity)

    assert snapshot.session is not None
    assert snapshot.session.scope_contract_digest == _DIGESTS[5]
    assert snapshot.session.capability_generation_digest == "7" * 64
    assert snapshot.session.scope_contract_digest != snapshot.session.capability_generation_digest
    raw = (tmp_path / "harness.sqlite3").read_bytes().decode("utf-8", errors="ignore")
    for field in ("batch_id", "combination_id", "hypothesis_record_id", "runtime_ref"):
        assert field not in raw


def test_harness_request_to_dict_uses_formal_module_invocation_fields() -> None:
    request = InvestigationHarnessRequest(
        research_task_id="research-request",
        module_invocation_id="invocation-request",
        module_kind="hypothesis-investigation",
        input_scientific_record_ids=("record-request",),
        module_input_digest=_DIGESTS[7],
        context_binding={
            "kind": "CreateInvestigationContext",
            "investigation_context_id": "context-request",
            "hypothesis_batch_record_id": "batch-request",
        },
    )
    assert request.to_dict() == {
        "research_task_id": "research-request",
        "module_invocation_id": "invocation-request",
        "module_kind": "hypothesis-investigation",
        "input_scientific_record_ids": ["record-request"],
        "module_input_digest": _DIGESTS[7],
        "context_binding": {
            "kind": "CreateInvestigationContext",
            "investigation_context_id": "context-request",
            "hypothesis_batch_record_id": "batch-request",
        },
    }
    assert not {"runtime_ref", "batch_id", "combination_id", "hypothesis_record_id"}.intersection(
        request.to_dict()
    )


def test_deep_runtime_names_are_available_without_schema_v10_aliases() -> None:
    assert InvestigationContextRunner.__name__ == "InvestigationContextRunner"
    assert InvestigationSessionRunner.__name__ == "InvestigationSessionRunner"
    assert InvestigationSessionOwner.__name__ == "InvestigationSessionOwner"
    assert InvestigationSessionHarnessOwner.__name__ == "InvestigationSessionHarnessOwner"
    assert InvestigationContextModuleAdapter.__name__ == "InvestigationContextModuleAdapter"
    assert ResearchTaskReviewRunner.__name__ == "ResearchTaskReviewRunner"
    assert (
        SchemaV10ReadOnlyInvestigationHarnessAdapter.__name__
        == "SchemaV10ReadOnlyInvestigationHarnessAdapter"
    )


def test_canonical_runtime_types_reject_legacy_constructor_keywords_and_properties() -> None:
    context = InvestigationContextIdentity(
        "context-negative",
        "research-negative",
        "batch-negative",
    )
    session = InvestigationSessionIdentity(
        "session-negative",
        context.investigation_context_id,
        context.research_task_id,
        "execution-negative",
        "workspace-negative",
        "scope-negative",
        _DIGESTS[0],
    )
    harness = HarnessRunIdentity(
        context.research_task_id,
        "invocation-negative",
        "harness-negative",
        _DIGESTS[1],
    )
    reference = SessionReference(
        context.research_task_id,
        context.investigation_context_id,
        session.hypothesis_execution_input_record_id,
        session.investigation_session_id,
        session.workspace_identity,
        session.minichem_scope_id,
        _DIGESTS[2],
        _DIGESTS[3],
    )
    attempt = ProviderAttemptReference(
        session.investigation_session_id,
        "thread-negative",
        "turn-negative",
        "attempt-negative",
        _DIGESTS[4],
    )
    job = JobReference(
        session.investigation_session_id,
        "job-negative",
        "operation-negative",
        "submission-negative",
        _DIGESTS[5],
    )
    request = InvestigationHarnessRequest(
        context.research_task_id,
        "invocation-negative",
        "hypothesis-investigation",
        ("record-negative",),
        _DIGESTS[6],
        {"investigation_context_id": context.investigation_context_id},
    )
    cases = (
        (InvestigationContextIdentity, context, {"inquiry_id": "legacy"}),
        (
            InvestigationSessionIdentity,
            session,
            {"inquiry_id": "legacy", "hypothesis_record_id": "legacy"},
        ),
        (
            HarnessRunIdentity,
            harness,
            {"task_id": "legacy", "runtime_ref": "legacy"},
        ),
        (
            SessionReference,
            reference,
            {"batch_id": "legacy", "scope_generation_digest": _DIGESTS[7]},
        ),
        (
            ProviderAttemptReference,
            attempt,
            {"session_id": "legacy", "thread_id": "legacy"},
        ),
        (
            JobReference,
            job,
            {"session_id": "legacy", "job_id": "legacy"},
        ),
        (
            InvestigationHarnessRequest,
            request,
            {"task_id": "legacy", "runtime_ref": "legacy"},
        ),
    )
    forbidden = {
        "inquiry_id",
        "task_id",
        "batch_id",
        "combination_id",
        "hypothesis_record_id",
        "session_id",
        "runtime_ref",
        "scope_generation_digest",
    }
    for runtime_type, instance, legacy_kwargs in cases:
        assert not forbidden.intersection(inspect.signature(runtime_type).parameters)
        assert not any(hasattr(instance, name) for name in forbidden)
        canonical_values = {
            field.name: getattr(instance, field.name)
            for field in getattr(runtime_type, "__dataclass_fields__", {}).values()
            if field.init
        }
        with pytest.raises(TypeError):
            runtime_type(**canonical_values, **legacy_kwargs)


def test_legacy_readers_are_the_only_old_schema_conversion_boundary() -> None:
    request = SchemaV1LegacyInvestigationHarnessRequestReader.read(
        {
            "task_id": "legacy-research",
            "module_run_id": "legacy-invocation",
            "module_kind": "investigation",
            "runtime_ref": "legacy-runtime",
            "input_record_refs": ["legacy-record"],
            "input_digest": "legacy-input",
        }
    )
    assert request.module_kind == "hypothesis-investigation"
    assert request.research_task_id == "legacy-research"
    assert not hasattr(request, "task_id")
    assert not {"task_id", "runtime_ref"}.intersection(request.to_dict())

    session = SchemaV1LegacyHarnessRecoveryReader.read_session(
        {
            "task_id": "legacy-research",
            "batch_id": "legacy-context",
            "combination_id": "legacy-execution",
            "session_id": "legacy-session",
            "workspace_ref": "legacy-workspace",
            "scope_id": "legacy-scope",
            "scope_generation_digest": _DIGESTS[0],
        }
    )
    assert session.scope_contract_digest == _DIGESTS[0]
    assert session.capability_generation_digest == _DIGESTS[0]
    assert not hasattr(session, "batch_id")


def test_legacy_manifest_and_output_readers_return_canonical_objects() -> None:
    legacy_request = {
        "task_id": "legacy-research",
        "module_run_id": "legacy-invocation",
        "module_kind": "investigation",
        "runtime_ref": "legacy-runtime",
        "input_record_refs": ["legacy-record"],
        "input_digest": "legacy-input",
    }
    manifest = SchemaV1LegacyHarnessManifestReader.read(
        {
            "schema_version": 1,
            "harness_run_ref": "legacy-harness",
            "request": legacy_request,
            "state": "open",
            "failure_code": None,
        },
        requested_harness_invocation_id="legacy-harness",
    )
    assert manifest["schema_version"] == 1
    assert manifest["request"].to_dict()["research_task_id"] == "legacy-research"
    assert "harness_run_ref" not in manifest

    with pytest.raises(InvestigationHarnessContractError):
        SchemaV1LegacyHarnessManifestReader.read(
            {
                "schema_version": 1,
                "harness_run_ref": "legacy-harness",
                "request": legacy_request,
                "state": "open",
                "failure_code": None,
                "unexpected": True,
            },
            requested_harness_invocation_id="legacy-harness",
        )

    # PreparedModuleOutput 的 reader 只接受旧 envelope；输出对象本身不暴露旧字段。
    with pytest.raises(InvestigationHarnessContractError):
        SchemaV1LegacyPreparedModuleOutputReader.read(
            {
                "module_run_id": "legacy-invocation",
                "input_digest": "legacy-input",
                "output_records": [],
                "output_digest": _DIGESTS[0],
            }
        )


def test_context_schema_v2_has_only_formal_session_columns(tmp_path: Path) -> None:
    database = tmp_path / "context-schema.sqlite3"
    InvestigationContextStore(database)
    import sqlite3

    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(investigation_sessions)")
        }
    assert {
        "research_task_id",
        "hypothesis_execution_input_record_id",
        "workspace_identity",
        "scope_contract_digest",
        "capability_generation_digest",
    }.issubset(columns)
    assert not columns.intersection(
        {"inquiry_id", "hypothesis_record_id", "investigation_workspace_relpath"}
    )


def test_capability_owner_generation_is_not_scope_contract(tmp_path: Path) -> None:
    class CapabilityOwner:
        def capability_for(self, _session: InvestigationSessionIdentity) -> Mapping[str, str]:
            return {"generation_digest": "7" * 64}

    owner = InvestigationSessionHarnessOwner(
        task_root=tmp_path,
        working_memory=object(),
        task_input={},
        policy={},
        dependencies={},
    )
    session = InvestigationSessionIdentity(
        investigation_session_id="session-capability",
        investigation_context_id="context-capability",
        research_task_id="research-capability",
        hypothesis_execution_input_record_id="execution-capability",
        workspace_identity="workspace-capability",
        minichem_scope_id="scope-capability",
        scope_contract_digest=_DIGESTS[0],
    )
    generation = owner._capability_generation_digest(session, CapabilityOwner())
    assert generation == "7" * 64
    assert generation != session.scope_contract_digest
