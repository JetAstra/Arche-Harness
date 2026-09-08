"""调查 scope capability 凭据生命周期的崩溃安全回归测试。"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from minichem_toolbox.scope import ScopeCapabilityError, ScopeCapabilityRegistry

import chemistry_multiagent.investigation.scope_credentials as scope_credentials
from chemistry_multiagent.investigation.scope_credentials import (
    InvestigationScopeCredentialConflict,
    InvestigationScopeCredentialCorruption,
    InvestigationScopeCredentialProvisionError,
    InvestigationScopeCredentialRevokeError,
    InvestigationScopeCredentialStore,
    scope_credential_generation_digest,
)


class FakeRegistry:
    """记录窄 registry 调用，并可让 provision/revoke 单次失败。"""

    def __init__(self) -> None:
        self.provisions: list[dict[str, object]] = []
        self.waiting_reauthorizations: list[dict[str, object]] = []
        self.revocations: list[str] = []
        self.fail_provision = False
        self.fail_revoke = False
        self.lock = threading.Lock()

    def provision_scope_capability(self, **kwargs: object) -> object:
        with self.lock:
            self.provisions.append(dict(kwargs))
            if self.fail_provision:
                self.fail_provision = False
                token = kwargs["token"]
                raise RuntimeError(f"fake registry failed for {token}")
        return object()

    def revoke_scope_capability(self, token: str) -> None:
        with self.lock:
            self.revocations.append(token)
            if self.fail_revoke:
                self.fail_revoke = False
                raise RuntimeError(f"fake revoke failed for {token}")

    def reauthorize_scope_capability_for_waiting_resume(
        self, **kwargs: object
    ) -> object:
        with self.lock:
            self.waiting_reauthorizations.append(dict(kwargs))
        return object()


def _identity(workspace: Path, *, combination_id: str = "combination-1") -> dict[str, object]:
    """构造测试调查 scope 身份。"""

    return {
        "task_id": "task-1",
        "batch_id": "batch-1",
        "combination_id": combination_id,
        "scope_id": f"scope-{combination_id}",
        "workspace": workspace,
    }


def _provision(store: InvestigationScopeCredentialStore, workspace: Path, **changes: object):
    """用测试默认值调用调查 scope provision 动作。"""

    values = _identity(workspace)
    values.update(changes)
    return store.provision_investigation_scope_capability(**values)


def test_record_precedes_registry_failure_and_restart_replays_same_token(tmp_path: Path) -> None:
    """registry 失败后 pending 记录保留，恢复只重放原 token。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "credentials"
    first_registry = FakeRegistry()
    first_registry.fail_provision = True
    first = InvestigationScopeCredentialStore(
        root,
        registry=first_registry,
        token_factory=lambda: "a" * 48,
    )

    with pytest.raises(InvestigationScopeCredentialProvisionError) as failure:
        _provision(first, workspace)
    assert "a" * 48 not in str(failure.value)
    record_path = next(root.glob("scope_*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["token"] == "a" * 48
    assert record["registry_state"] == "pending"

    second_registry = FakeRegistry()
    factory_called = False

    def should_not_generate() -> str:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("恢复不应生成第二个 token")

    second = InvestigationScopeCredentialStore(
        root,
        registry=second_registry,
        token_factory=should_not_generate,
    )
    capability = _provision(second, workspace)
    assert capability.token == "a" * 48
    assert not factory_called
    assert second_registry.provisions[0]["token"] == "a" * 48
    assert json.loads(record_path.read_text(encoding="utf-8"))["registry_state"] == "registered"


def test_reuse_registered_scope_reads_exact_generation_without_registry_call(
    tmp_path: Path,
) -> None:
    """残缺 waiting 的显式授权只能只读复用已注册的精确 capability 代次。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry()
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        registry=registry,
        token_factory=lambda: "r" * 48,
    )
    capability = _provision(store, workspace)
    record_path = capability.record_path
    record_before = record_path.read_bytes()
    registry.provisions.clear()

    reused = store.reuse_registered_investigation_scope_capability(
        **_identity(workspace),
        expected_generation_digest=scope_credential_generation_digest(capability),
    )

    assert reused.token_sha256 == capability.token_sha256
    assert registry.provisions == []
    assert record_path.read_bytes() == record_before

    with pytest.raises(
        InvestigationScopeCredentialConflict,
        match="generation digest",
    ):
        store.reuse_registered_investigation_scope_capability(
            **_identity(workspace),
            expected_generation_digest="f" * 64,
        )
    assert registry.provisions == []
    assert record_path.read_bytes() == record_before


def test_registry_success_before_state_write_crash_is_replayed_without_new_token(
    tmp_path: Path,
) -> None:
    """服务端成功而本地状态更新崩溃时，恢复会用同一个 token 幂等重放。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry()

    class CrashOnRegisteredWrite(InvestigationScopeCredentialStore):
        """在 pending 初始落盘后模拟 registered 状态落盘前的崩溃。"""

        writes = 0

        def _write_record(self, path: Path, record: object) -> None:
            self.writes += 1
            if self.writes == 2:
                raise RuntimeError("simulated process crash")
            super()._write_record(path, record)  # type: ignore[arg-type]

    crashing = CrashOnRegisteredWrite(
        tmp_path / "credentials",
        registry=registry,
        token_factory=lambda: "b" * 48,
    )
    with pytest.raises(RuntimeError, match="simulated process crash"):
        _provision(crashing, workspace)
    assert registry.provisions[0]["token"] == "b" * 48

    restarted_registry = FakeRegistry()
    restarted = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        registry=restarted_registry,
        token_factory=lambda: (_ for _ in ()).throw(AssertionError("不应生成第二个 token")),
    )
    capability = _provision(restarted, workspace)
    assert capability.token == "b" * 48
    assert restarted_registry.provisions[0]["token"] == "b" * 48


def test_concurrent_provision_uses_one_record_and_one_generated_token(tmp_path: Path) -> None:
    """同一身份的并发 provision 由跨线程和 POSIX 文件锁保证唯一结果。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry()
    generated: list[str] = []
    generated_lock = threading.Lock()

    def token_factory() -> str:
        with generated_lock:
            token = f"concurrent-{len(generated):02d}".ljust(48, "x")
            generated.append(token)
            return token

    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        registry=registry,
        token_factory=token_factory,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        capabilities = list(executor.map(lambda _: _provision(store, workspace), range(16)))
    assert {capability.token for capability in capabilities} == {generated[0]}
    assert len(generated) == 1
    assert len(list((tmp_path / "credentials").glob("scope_*.json"))) == 1


def test_identity_scope_workspace_and_token_digest_conflicts_fail_closed(tmp_path: Path) -> None:
    """身份、scope、工作区和摘要冲突不能创建第二条凭据。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_workspace = tmp_path / "other-workspace"
    other_workspace.mkdir()
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        registry=FakeRegistry(),
        token_factory=lambda: "c" * 48,
    )
    _provision(store, workspace)
    with pytest.raises(InvestigationScopeCredentialConflict):
        _provision(store, workspace, scope_id="scope-other")
    with pytest.raises(InvestigationScopeCredentialConflict):
        _provision(store, other_workspace)
    with pytest.raises(InvestigationScopeCredentialConflict):
        _provision(store, workspace, combination_id="combination-other", scope_id="scope-other")

    record_path = next((tmp_path / "credentials").glob("scope_*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["token_sha256"] = "d" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(record_path, 0o600)
    with pytest.raises(InvestigationScopeCredentialCorruption):
        _provision(store, workspace)


def test_private_permissions_corruption_and_audit_never_expose_token(tmp_path: Path) -> None:
    """目录/文件权限和审计输出均不允许泄漏明文 token。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "credentials"
    registry = FakeRegistry()
    store = InvestigationScopeCredentialStore(
        root,
        registry=registry,
        token_factory=lambda: "e" * 48,
    )
    capability = _provision(store, workspace)
    assert root.stat().st_mode & 0o777 == 0o700
    assert capability.record_path.stat().st_mode & 0o777 == 0o600
    assert store.lock_path.stat().st_mode & 0o777 == 0o600
    assert "e" * 48 not in repr(capability)
    assert "e" * 48 not in json.dumps(store.list_investigation_scope_capability_records()[0].to_dict())
    assert "e" * 48 not in json.dumps(capability.to_audit_dict())

    os.chmod(capability.record_path, 0o644)
    with pytest.raises(InvestigationScopeCredentialCorruption):
        store.list_investigation_scope_capability_records()
    os.chmod(capability.record_path, 0o600)
    os.chmod(root, 0o755)
    with pytest.raises(InvestigationScopeCredentialCorruption):
        store.list_investigation_scope_capability_records()


def test_revoke_calls_server_first_is_idempotent_and_blocks_reprovision(tmp_path: Path) -> None:
    """撤销先服务端后本地，重复撤销不重复调用且终局禁止重新 provision。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry()
    root = tmp_path / "credentials"
    store = InvestigationScopeCredentialStore(root, registry=registry, token_factory=lambda: "f" * 48)
    capability = _provision(store, workspace)
    original_revoke = registry.revoke_scope_capability

    def check_server_first(token: str) -> None:
        raw = json.loads(capability.record_path.read_text(encoding="utf-8"))
        assert raw["revoked"] is False
        original_revoke(token)

    registry.revoke_scope_capability = check_server_first  # type: ignore[method-assign]
    store.revoke_investigation_scope_capability(capability)
    store.revoke_investigation_scope_capability(capability)
    assert registry.revocations == ["f" * 48]
    raw = json.loads(capability.record_path.read_text(encoding="utf-8"))
    assert raw["revoked"] is True
    with pytest.raises(InvestigationScopeCredentialConflict):
        _provision(store, workspace)


def test_revoke_failure_keeps_active_record_and_error_does_not_echo_token(tmp_path: Path) -> None:
    """服务端撤销失败时不提前写 revoked，错误不含 token。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry()
    root = tmp_path / "credentials"
    store = InvestigationScopeCredentialStore(root, registry=registry, token_factory=lambda: "g" * 48)
    capability = _provision(store, workspace)
    registry.fail_revoke = True
    with pytest.raises(InvestigationScopeCredentialRevokeError) as failure:
        store.revoke_investigation_scope_capability(capability)
    assert "g" * 48 not in str(failure.value)
    assert json.loads(capability.record_path.read_text(encoding="utf-8"))["revoked"] is False
    store.revoke_investigation_scope_capability(capability)
    assert json.loads(capability.record_path.read_text(encoding="utf-8"))["revoked"] is True


def test_credential_store_integrates_with_real_minichem_registry(tmp_path: Path) -> None:
    """ARCHE 凭据生命周期必须与 MiniChem 的真实 registry 协议保持一致。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ScopeCapabilityRegistry(tmp_path / "minichem-registry")
    store = InvestigationScopeCredentialStore(
        tmp_path / "arche-credentials",
        registry=registry,
        token_factory=lambda: "h" * 48,
    )

    capability = _provision(store, workspace)
    authenticated = registry.authenticate_scope_capability(capability.token)
    assert authenticated.task_id == "task-1"
    assert authenticated.batch_id == "batch-1"
    assert authenticated.combination_id == "combination-1"
    assert authenticated.scope_id == "scope-combination-1"
    assert authenticated.workspace == workspace.resolve()

    store.revoke_investigation_scope_capability(capability)
    with pytest.raises(ScopeCapabilityError, match="invalid capability"):
        registry.authenticate_scope_capability(capability.token)


def test_explicit_correction_reauthorization_keeps_revoked_generation_and_uses_new_token(
    tmp_path: Path,
) -> None:
    """显式更正只能追加新代次，旧 revoked capability 仍保持不可认证。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ScopeCapabilityRegistry(tmp_path / "minichem-registry")
    token_values = iter(("h" * 48, "i" * 48))
    store = InvestigationScopeCredentialStore(
        tmp_path / "arche-credentials",
        registry=registry,
        token_factory=token_values.__next__,
    )

    original = _provision(store, workspace)
    store.revoke_investigation_scope_capability(original)

    repaired = store.reauthorize_investigation_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-combination-1",
        workspace=workspace,
        correction_event_ref="investigation-task-correction-1",
    )
    exact_retry = store.reauthorize_investigation_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-combination-1",
        workspace=workspace,
        correction_event_ref="investigation-task-correction-1",
    )
    ordinary_repair_replay = _provision(store, workspace)

    assert repaired.token != original.token
    assert repaired.record_path != original.record_path
    assert exact_retry.token == repaired.token
    assert exact_retry.record_path == repaired.record_path
    assert ordinary_repair_replay.token == repaired.token
    assert ordinary_repair_replay.record_path == repaired.record_path
    assert registry.authenticate_scope_capability(repaired.token).scope_id == (
        "scope-combination-1"
    )
    with pytest.raises(ScopeCapabilityError, match="invalid capability"):
        registry.authenticate_scope_capability(original.token)
    audits = store.list_investigation_scope_capability_records()
    assert len(audits) == 2
    assert {audit.token_sha256 for audit in audits} == {
        original.token_sha256,
        repaired.token_sha256,
    }
    assert next(audit for audit in audits if audit.token_sha256 == original.token_sha256).revoked
    assert not next(audit for audit in audits if audit.token_sha256 == repaired.token_sha256).revoked


def test_waiting_resume_reauthorization_is_independent_idempotent_and_hash_bound(
    tmp_path: Path,
) -> None:
    """waiting 恢复只绑定旧代 token 摘要，重复事件不新增第二代记录。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry()
    token_values = iter(("k" * 48, "l" * 48, "m" * 48))
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        registry=registry,
        token_factory=token_values.__next__,
    )
    original = _provision(store, workspace)
    store.revoke_investigation_scope_capability(original)

    resumed = store.reauthorize_investigation_scope_capability_for_waiting_resume(
        **_identity(workspace),
        waiting_resume_event_ref="wm-waiting-resume-1",
        previous_generation_hash=original.token_sha256,
    )
    retry = store.reauthorize_investigation_scope_capability_for_waiting_resume(
        **_identity(workspace),
        waiting_resume_event_ref="wm-waiting-resume-1",
        previous_generation_hash=original.token_sha256,
    )
    assert resumed.token != original.token
    assert retry.record_path == resumed.record_path
    assert resumed.waiting_resume_event_ref == "wm-waiting-resume-1"
    assert resumed.previous_generation_hash == original.token_sha256
    assert len(registry.waiting_reauthorizations) == 2
    assert registry.waiting_reauthorizations[-1]["previous_generation_hash"] == (
        original.token_sha256
    )
    assert len(store.list_investigation_scope_capability_records()) == 2

    # 当前 active 代次只能按自身摘要复用；旧代摘要在 active/revoked 混合时必须拒绝。
    with pytest.raises(InvestigationScopeCredentialConflict):
        store.reauthorize_investigation_scope_capability_for_waiting_resume(
            **_identity(workspace),
            waiting_resume_event_ref="wm-waiting-resume-2",
            previous_generation_hash=original.token_sha256,
        )
    active_retry = store.reauthorize_investigation_scope_capability_for_waiting_resume(
        **_identity(workspace),
        waiting_resume_event_ref="wm-waiting-resume-2",
        previous_generation_hash=resumed.token_sha256,
    )
    assert active_retry.record_path == resumed.record_path

    digest = scope_credential_generation_digest(resumed)
    assert digest == resumed.scope_credential_generation_digest
    assert digest == resumed.generation_digest
    assert resumed.token not in digest
    assert str(resumed.record_path) not in digest

    # 撤销 active 后，多个 revoked 代次仍按 previous_generation_hash 唯一选择旧代。
    store.revoke_investigation_scope_capability(resumed)
    third = store.reauthorize_investigation_scope_capability_for_waiting_resume(
        **_identity(workspace),
        waiting_resume_event_ref="wm-waiting-resume-3",
        previous_generation_hash=original.token_sha256,
    )
    assert third.token == "m" * 48
    audits = store.list_investigation_scope_capability_records()
    assert len(audits) == 3
    assert sum(not audit.revoked for audit in audits) == 1


def test_waiting_resume_record_tampering_fails_closed(tmp_path: Path) -> None:
    """篡改 waiting 代次摘要或身份后，读取和再次授权都停止。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry()
    token_values = iter(("n" * 48, "o" * 48))
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        registry=registry,
        token_factory=token_values.__next__,
    )
    original = _provision(store, workspace)
    store.revoke_investigation_scope_capability(original)
    resumed = store.reauthorize_investigation_scope_capability_for_waiting_resume(
        **_identity(workspace),
        waiting_resume_event_ref="wm-waiting-tamper",
        previous_generation_hash=original.token_sha256,
    )
    raw = json.loads(resumed.record_path.read_text(encoding="utf-8"))
    raw["previous_generation_hash"] = "0" * 64
    resumed.record_path.write_text(json.dumps(raw), encoding="utf-8")
    os.chmod(resumed.record_path, 0o600)
    with pytest.raises(InvestigationScopeCredentialCorruption):
        store.list_investigation_scope_capability_records()


def test_schema_v2_requires_explicit_correction_event_ref_key(tmp_path: Path) -> None:
    """本地 v2 记录缺少更正引用键时必须按篡改记录拒绝读取。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        registry=FakeRegistry(),
        token_factory=lambda: "j" * 48,
    )
    capability = _provision(store, workspace)
    raw = json.loads(capability.record_path.read_text(encoding="utf-8"))
    del raw["correction_event_ref"]
    capability.record_path.write_text(json.dumps(raw), encoding="utf-8")
    os.chmod(capability.record_path, 0o600)

    with pytest.raises(InvestigationScopeCredentialCorruption):
        store.list_investigation_scope_capability_records()


def test_credential_store_exposes_only_explicit_action_names() -> None:
    """凭据 store 不暴露泛化动作别名，明确的 manager 兼容名仍可用。"""

    assert not hasattr(InvestigationScopeCredentialStore, "provision")
    assert not hasattr(InvestigationScopeCredentialStore, "revoke")
    assert not hasattr(InvestigationScopeCredentialStore, "list_records")
    assert not hasattr(scope_credentials, "ScopeCredentialStore")
    assert (
        scope_credentials.InvestigationScopeCredentialManager
        is InvestigationScopeCredentialStore
    )
