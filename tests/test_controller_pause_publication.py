"""V2 Harness 暂停状态发布和恢复回归。"""

from __future__ import annotations

from pathlib import Path

from chemistry_multiagent.investigation.harness_recovery_store import (
    HarnessLifecycle,
    HarnessRecoveryStore,
)
from chemistry_multiagent.investigation.harness_runtime import (
    BackendCompleted,
    BackendSuspended,
    InvestigationHarness,
    Suspended,
)
from tests.test_harness_runtime_recovery_store import (
    _identity,
    _output,
    _request,
    ScriptedBackend,
)


def test_v2_harness_suspension_is_persisted_as_current_state(tmp_path: Path) -> None:
    """backend 暂时不可用时，V2 Harness 写入可恢复的 suspended 状态。"""

    request = _request(module_invocation_id="module-invocation-pause")
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    harness = InvestigationHarness(
        tmp_path / "runtime",
        backend=ScriptedBackend(start_result=BackendSuspended("provider_unavailable")),
        recovery_store=store,
    )

    outcome = harness.start(request)

    assert isinstance(outcome, Suspended)
    assert outcome.reason_code == "provider_unavailable"
    assert store.read_snapshot(_identity(request)).state is HarnessLifecycle.SUSPENDED


def test_v2_harness_resume_reads_suspended_invocation_without_new_task(tmp_path: Path) -> None:
    """恢复同一 Harness invocation，身份不变且只调用一次 resume。"""

    request = _request(module_invocation_id="module-invocation-pause-resume")
    store = HarnessRecoveryStore(tmp_path / "harness-recovery.sqlite3")
    backend = ScriptedBackend(
        start_result=BackendSuspended("provider_unavailable"),
        resume_result=BackendCompleted(_output(request)),
    )
    harness = InvestigationHarness(tmp_path / "runtime", backend=backend, recovery_store=store)
    first = harness.start(request)
    assert isinstance(first, Suspended)

    resumed = harness.resume(request)

    assert resumed.harness_invocation_id == first.harness_invocation_id
    assert backend.calls == ["start", "resume"]
    assert store.read_snapshot(_identity(request)).state is HarnessLifecycle.COMPLETED
