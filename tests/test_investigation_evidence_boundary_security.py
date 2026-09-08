"""Evidence MCP 边界 helper 的失败闭环和恢复身份测试。"""

from __future__ import annotations

import errno
import json
import signal
import sys
import ctypes
from types import ModuleType
from pathlib import Path
from typing import Any

import pytest

from chemistry_multiagent.investigation import evidence_mcp_lease as lease_module
from chemistry_multiagent.investigation import evidence_mcp_server as server_module
from chemistry_multiagent.investigation import scope_credentials as scope_module
from chemistry_multiagent.investigation.evidence import ReviewEvidenceStore
from chemistry_multiagent.investigation.evidence_mcp_lease import (
    EvidenceMcpLeaseConfigurationError,
    EvidenceMcpLeaseError,
    EvidenceMcpLeaseIdentityError,
    TaskEvidenceMcpLease,
)
from chemistry_multiagent.investigation.evidence_mcp_server import (
    create_evidence_capability_file,
    EvidenceMcpConfigurationError,
    load_snapshot_allowlist_file,
    main,
)
from chemistry_multiagent.investigation.scope_credentials import (
    InvestigationScopeCredentialCorruption,
    InvestigationScopeCredentialError,
    InvestigationScopeCredentialStore,
)


CAPABILITY = "boundary-capability-" + "x" * 40


def _make_lease(tmp_path: Path) -> TaskEvidenceMcpLease:
    """创建只用于直接调用安全 helper 的私有租约。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    return TaskEvidenceMcpLease(
        "task-boundary",
        store.root_dir,
        tmp_path / "runtime",
        port=19020,
        capability_token=CAPABILITY,
    )


class _FakeChild:
    """记录 wait 调用而不创建真实子进程。"""

    def __init__(self) -> None:
        self.wait_calls: list[float] = []

    def wait(self, *, timeout: float) -> None:
        self.wait_calls.append(timeout)


def test_credential_store_constructor_rejects_invalid_callbacks_and_root(
    tmp_path: Path,
) -> None:
    """凭据 store 初始化拒绝非法根路径、registry callback 和 token factory。"""

    with pytest.raises(InvestigationScopeCredentialError, match="root is invalid"):
        InvestigationScopeCredentialStore(123)  # type: ignore[arg-type]
    with pytest.raises(InvestigationScopeCredentialError, match="provision_callable"):
        InvestigationScopeCredentialStore(tmp_path / "provision", provision_callable=object())  # type: ignore[arg-type]
    with pytest.raises(InvestigationScopeCredentialError, match="revoke_callable"):
        InvestigationScopeCredentialStore(tmp_path / "revoke", revoke_callable=object())  # type: ignore[arg-type]
    with pytest.raises(InvestigationScopeCredentialError, match="token_factory"):
        InvestigationScopeCredentialStore(tmp_path / "factory", token_factory=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [None, "", ".", "..", "x" * 257, "bad/name", "bad\\name", "bad\x00name", "bad\x01name"],
)
def test_scope_identity_component_rejects_path_and_control_values(value: Any) -> None:
    """scope 身份组件拒绝空值、路径、过长文本和控制字符。"""

    with pytest.raises(ValueError, match="scope_id is invalid"):
        scope_module._validate_identity_component(value, "scope_id")


def test_server_allowlist_reader_accepts_mapping_and_line_formats(
    tmp_path: Path,
) -> None:
    """allowlist reader 支持冻结的 object 和逐行格式，并拒绝标量 JSON。"""

    path = tmp_path / "allowlist.json"
    path.write_text('{"snapshot_ids":["snap_abcdefgh"]}', encoding="utf-8")
    path.chmod(0o600)
    assert load_snapshot_allowlist_file(path) == ("snap_abcdefgh",)

    path.write_text("snap_abcdefgh\nsnap_ijklmnop\nsnap_abcdefgh\n", encoding="utf-8")
    assert load_snapshot_allowlist_file(path) == ("snap_abcdefgh", "snap_ijklmnop")

    path.write_text("null", encoding="utf-8")
    with pytest.raises(EvidenceMcpConfigurationError, match="allowlist file is invalid"):
        load_snapshot_allowlist_file(path)


def test_server_snapshot_binding_rejects_ambiguous_sources() -> None:
    """服务配置只接受一个 snapshot 来源，并保留 allowlist 模式标记。"""

    assert server_module._select_server_snapshot_binding(
        None,
        None,
        ["snap_abcdefgh"],
    ) == (["snap_abcdefgh"], True)
    with pytest.raises(EvidenceMcpConfigurationError, match="specified more than once"):
        server_module._select_server_snapshot_binding(
            None,
            ["snap_abcdefgh"],
            ["snap_ijklmnop"],
        )
    with pytest.raises(EvidenceMcpConfigurationError, match="ambiguous"):
        server_module._select_server_snapshot_binding(
            "snap_abcdefgh",
            ["snap_ijklmnop"],
            None,
        )
    with pytest.raises(EvidenceMcpConfigurationError, match="binding is required"):
        server_module._select_server_snapshot_binding(None, None, None)


def test_scope_record_status_helper_rejects_inconsistent_states(tmp_path: Path) -> None:
    """凭据记录状态字段的每种不一致组合都 fail-closed。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        provision_callable=lambda **_kwargs: None,
        token_factory=lambda: "status-token-" + "x" * 40,
    )
    capability = store.provision_investigation_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=workspace,
    )
    raw = json.loads(capability.record_path.read_text(encoding="utf-8"))
    invalid_updates = (
        {"revoked": "no"},
        {"status": "unknown"},
        {"revoked": True, "status": "active"},
        {"revoked": False, "status": "revoked"},
        {"registry_state": "unknown"},
        {"revoked": True, "registry_state": "pending"},
        {"revoked": False, "registry_state": "revoked"},
    )
    for update in invalid_updates:
        with pytest.raises(InvestigationScopeCredentialCorruption):
            scope_module._validate_record_status_fields({**raw, **update})


def test_scope_record_write_cleans_temporary_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原子 replace 失败时只报告保存错误并清理临时凭据文件。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        provision_callable=lambda **_kwargs: None,
        token_factory=lambda: "write-token-" + "x" * 40,
    )
    capability = store.provision_investigation_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=workspace,
    )
    record = json.loads(capability.record_path.read_text(encoding="utf-8"))
    target = store.root / "manual-record.json"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failure")

    monkeypatch.setattr(scope_module.os, "replace", fail_replace)
    with pytest.raises(InvestigationScopeCredentialCorruption, match="cannot be saved"):
        store._write_record(target, record)
    assert not target.exists()
    assert not list(store.root.glob(".manual-record.json.*.tmp"))


def test_scope_record_write_fails_after_repeated_temporary_name_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """临时文件名连续冲突时不覆盖既有文件，直接拒绝保存。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = InvestigationScopeCredentialStore(tmp_path / "credentials")
    target = store.root / "collision-record.json"
    occupied = store.root / ".collision-record.json.deadbeef.tmp"
    occupied.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(scope_module.secrets, "token_hex", lambda _size: "deadbeef")

    with pytest.raises(InvestigationScopeCredentialCorruption, match="cannot be saved"):
        store._write_record(target, {"marker": "safe"})
    assert occupied.read_text(encoding="utf-8") == "occupied"


def test_scope_revoke_identity_hints_cover_capability_and_explicit_forms(
    tmp_path: Path,
) -> None:
    """revoke 定位同时覆盖 capability 对象、token、完整身份和非法输入。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = InvestigationScopeCredentialStore(
        tmp_path / "credentials",
        provision_callable=lambda **_kwargs: None,
        token_factory=lambda: "hints-token-" + "x" * 40,
    )
    capability = store.provision_investigation_scope_capability(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=workspace,
    )
    token_hint, requested = scope_module._resolve_revoke_identity_hints(
        capability,
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        scope_id="scope-1",
        workspace=workspace,
    )
    assert token_hint == capability.token
    assert requested == capability.identity
    token_hint, requested = scope_module._resolve_revoke_identity_hints(
        capability.token,
        task_id=None,
        batch_id=None,
        combination_id=None,
        scope_id=None,
        workspace=None,
    )
    assert token_hint == capability.token
    assert requested is None
    assert scope_module._resolve_revoke_identity_hints(
        None,
        task_id=None,
        batch_id=None,
        combination_id=None,
        scope_id=None,
        workspace=None,
    ) == (None, None)
    with pytest.raises(scope_module.InvestigationScopeCredentialConflict, match="conflicts"):
        scope_module._resolve_revoke_identity_hints(
            capability,
            task_id="task-other",
            batch_id="batch-1",
            combination_id="combination-1",
            scope_id="scope-1",
            workspace=workspace,
        )
    with pytest.raises(InvestigationScopeCredentialError, match="capability input"):
        scope_module._resolve_revoke_identity_hints(
            object(),
            task_id=None,
            batch_id=None,
            combination_id=None,
            scope_id=None,
            workspace=None,
        )  # type: ignore[arg-type]


class _FakePortProbe:
    """记录 loopback 探测参数并按测试要求抛出网络结果。"""

    def __init__(self, outcome: BaseException | None = None) -> None:
        self.outcome = outcome
        self.address: tuple[Any, ...] | None = None
        self.timeout: float | None = None
        self.closed = False

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def connect(self, address: tuple[Any, ...]) -> None:
        self.address = address
        if self.outcome is not None:
            raise self.outcome

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "outcome",
    [ConnectionRefusedError(), OSError(errno.ECONNREFUSED, "refused")],
)
def test_loopback_port_probe_accepts_refused_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: BaseException,
) -> None:
    """端口明确拒绝连接时可以安全继续启动。"""

    lease = _make_lease(tmp_path)
    probe = _FakePortProbe(outcome)
    monkeypatch.setattr(lease_module.socket, "socket", lambda *_args: probe)

    lease._assert_loopback_port_free()

    assert probe.address == (lease.host, lease.port)
    assert probe.timeout == 0.15
    assert probe.closed


def test_loopback_port_probe_rejects_unknown_network_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非连接拒绝网络错误不能被当作端口空闲。"""

    lease = _make_lease(tmp_path)
    probe = _FakePortProbe(OSError(errno.EIO, "probe failed"))
    monkeypatch.setattr(lease_module.socket, "socket", lambda *_args: probe)

    with pytest.raises(lease_module.EvidenceMcpLeasePortError, match="safely probed"):
        lease._assert_loopback_port_free()
    assert probe.closed


def test_loopback_port_probe_rejects_successful_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成功连接表示未知监听者占用端口，必须 fail-closed。"""

    lease = _make_lease(tmp_path)
    probe = _FakePortProbe()
    monkeypatch.setattr(lease_module.socket, "socket", lambda *_args: probe)

    with pytest.raises(lease_module.EvidenceMcpLeasePortError, match="occupied"):
        lease._assert_loopback_port_free()
    assert probe.closed


def test_terminate_identity_cleans_stopped_process_without_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已停止身份只回收直接句柄，不向任何 PID 发送信号。"""

    lease = _make_lease(tmp_path)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(lease_module.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(lease, "_identity_is_owned_and_alive", lambda *_args: False)
    monkeypatch.setattr(lease, "_identity_activity", lambda *_args: "stopped")
    identity = {"pid": 328, "start_time_ticks": 17, "boot_id": "boot"}

    lease._terminate_owned_identity(identity, "snapshot-boundary", None)

    assert signals == []


def test_launch_snapshot_process_removes_capability_when_log_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动日志无法以私有模式打开时不创建子进程，并清理 capability。"""

    lease = _make_lease(tmp_path)
    removed: list[bool] = []
    monkeypatch.setattr(lease, "_recover_prior_receipt", lambda: None)
    monkeypatch.setattr(lease, "_assert_loopback_port_free", lambda: None)
    monkeypatch.setattr(lease, "_ensure_capability_file", lambda: None)
    monkeypatch.setattr(lease, "_remove_private_capability", lambda: removed.append(True))

    def fail_open(_path: Path) -> int:
        raise EvidenceMcpLeaseConfigurationError("log unavailable")

    monkeypatch.setattr(lease_module, "_open_private_log", fail_open)
    with pytest.raises(EvidenceMcpLeaseConfigurationError, match="log unavailable"):
        lease._launch_snapshot_process("snapshot-abcdefgh")
    assert removed == [True]


def test_verified_identity_stop_uses_term_and_reaps_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一身份正常退出时只发 TERM 并回收直接子句柄。"""

    lease = _make_lease(tmp_path)
    child = _FakeChild()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(lease_module.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(lease, "_wait_for_identity_stop", lambda *_args: True)
    identity = {"pid": 321, "start_time_ticks": 10, "boot_id": "boot"}

    lease._stop_verified_process_identity(identity, "snapshot-boundary", child)  # type: ignore[arg-type]

    assert signals == [(321, signal.SIGTERM)]
    assert child.wait_calls == [0.2]


def test_parent_death_guard_is_noop_outside_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 Linux 平台不调用平台专用父进程退出保护。"""

    monkeypatch.setattr(lease_module.sys, "platform", "darwin")
    lease_module._configure_child_parent_death_signal(123)


def test_parent_death_guard_configures_linux_prctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux 子进程设置 PR_SET_PDEATHSIG 后继续执行启动。"""

    calls: list[tuple[Any, ...]] = []

    class _Prctl:
        def __call__(self, *args: Any) -> int:
            calls.append(args)
            return 0

    class _Libc:
        prctl = _Prctl()

    monkeypatch.setattr(lease_module.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: _Libc())
    monkeypatch.setattr(lease_module.os, "getppid", lambda: 123)

    lease_module._configure_child_parent_death_signal(123)

    assert calls == [(lease_module._PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)]


def test_parent_death_guard_exits_when_parent_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fork 后父进程已变更时立即退出，不留下脱离租约的服务。"""

    class _Prctl:
        def __call__(self, *_args: Any) -> int:
            return 0

    class _Libc:
        prctl = _Prctl()

    exits: list[int] = []
    monkeypatch.setattr(lease_module.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: _Libc())
    monkeypatch.setattr(lease_module.os, "getppid", lambda: 999)
    monkeypatch.setattr(lease_module.os, "_exit", lambda code: exits.append(code))

    lease_module._configure_child_parent_death_signal(123)

    assert exits == [143]


def test_verified_identity_stop_escalates_to_kill_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TERM 超时且身份仍归属本租约时才升级为 KILL。"""

    lease = _make_lease(tmp_path)
    child = _FakeChild()
    signals: list[tuple[int, int]] = []
    waits = iter((False, True))
    monkeypatch.setattr(lease_module.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(lease, "_wait_for_identity_stop", lambda *_args: next(waits))
    monkeypatch.setattr(lease, "_identity_is_owned_and_alive", lambda *_args: True)
    identity = {"pid": 322, "start_time_ticks": 11, "boot_id": "boot"}

    lease._stop_verified_process_identity(identity, "snapshot-boundary", child)  # type: ignore[arg-type]

    assert signals == [(322, signal.SIGTERM), (322, signal.SIGKILL)]
    assert child.wait_calls == [0.2]


def test_verified_identity_stop_rejects_changed_identity_after_term_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TERM 超时后身份已变化时拒绝 KILL，避免误杀 PID 复用者。"""

    lease = _make_lease(tmp_path)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(lease_module.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(lease, "_wait_for_identity_stop", lambda *_args: False)
    monkeypatch.setattr(lease, "_identity_is_owned_and_alive", lambda *_args: False)
    identity = {"pid": 323, "start_time_ticks": 12, "boot_id": "boot"}

    with pytest.raises(EvidenceMcpLeaseIdentityError, match="changed during shutdown"):
        lease._stop_verified_process_identity(identity, "snapshot-boundary", None)

    assert signals == [(323, signal.SIGTERM)]


def test_verified_identity_stop_wraps_term_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """系统拒绝 TERM 时只暴露稳定租约错误，不泄漏底层文本。"""

    lease = _make_lease(tmp_path)

    def fail_kill(_pid: int, _value: int) -> None:
        raise OSError("secret host path")

    monkeypatch.setattr(lease_module.os, "kill", fail_kill)
    identity = {"pid": 324, "start_time_ticks": 13, "boot_id": "boot"}

    with pytest.raises(EvidenceMcpLeaseError, match="could not be stopped") as exc_info:
        lease._stop_verified_process_identity(identity, "snapshot-boundary", None)
    assert "secret host path" not in str(exc_info.value)


def test_verified_identity_stop_wraps_kill_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """升级 KILL 的系统错误也必须保持安全租约错误类型。"""

    lease = _make_lease(tmp_path)
    calls = iter((None, OSError("secret runtime path")))
    monkeypatch.setattr(lease_module.os, "kill", lambda _pid, _value: _raise_next(calls))
    monkeypatch.setattr(lease, "_wait_for_identity_stop", lambda *_args: False)
    monkeypatch.setattr(lease, "_identity_is_owned_and_alive", lambda *_args: True)
    identity = {"pid": 325, "start_time_ticks": 14, "boot_id": "boot"}

    with pytest.raises(EvidenceMcpLeaseError, match="could not be killed"):
        lease._stop_verified_process_identity(identity, "snapshot-boundary", None)


def _raise_next(values: Any) -> None:
    """按预设序列模拟 TERM 成功和 KILL 失败。"""

    value = next(values)
    if isinstance(value, BaseException):
        raise value


def test_stdio_main_uses_frozen_single_snapshot_and_no_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stdio 入口只把不透明 snapshot 交给服务，不读取宿主路径。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    calls: list[str] = []

    class _FakeServer:
        def run(self, *, transport: str) -> None:
            calls.append(transport)

    monkeypatch.setattr(server_module, "create_evidence_mcp_server", lambda *_args, **_kwargs: _FakeServer())

    result = main(
        [
            "--store-root",
            str(store.root_dir),
            "--snapshot-id",
            "snapshot-abcdefgh",
            "--transport",
            "stdio",
        ]
    )

    assert result == 0
    assert calls == ["stdio"]


def test_http_main_uses_allowlist_capability_and_uvicorn_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 入口使用 capability 文件、allowlist 和固定监听参数。"""

    store = ReviewEvidenceStore(tmp_path / "store")
    capability_path = tmp_path / "runtime" / "capability.token"
    server_calls: list[dict[str, Any]] = []
    uvicorn_calls: list[dict[str, Any]] = []

    class _FakeServer:
        pass

    def fake_create(_store: Any, **kwargs: Any) -> _FakeServer:
        server_calls.append(kwargs)
        return _FakeServer()

    monkeypatch.setattr(server_module, "create_evidence_mcp_server", fake_create)
    monkeypatch.setattr(server_module, "create_evidence_http_app", lambda server: server)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        ModuleType("uvicorn"),
    )
    uvicorn_module = sys.modules["uvicorn"]
    setattr(
        uvicorn_module,
        "run",
        lambda app, **kwargs: uvicorn_calls.append({"app": app, **kwargs}),
    )
    from chemistry_multiagent.investigation.evidence_mcp_server import create_evidence_capability_file

    token = create_evidence_capability_file(capability_path, token=CAPABILITY)
    result = main(
        [
            "--store-root",
            str(store.root_dir),
            "--snapshot-id",
            "snapshot-abcdefgh",
            "--snapshot-id",
            "snapshot-ijklmnop",
            "--capability-file",
            str(capability_path),
            "--transport",
            "streamable_http",
            "--host",
            "127.0.0.1",
            "--port",
            "19021",
        ]
    )

    assert result == 0
    assert token == CAPABILITY
    assert server_calls[0]["snapshot_allowlist"] == (
        "snapshot-abcdefgh",
        "snapshot-ijklmnop",
    )
    assert server_calls[0]["capability_token"] == CAPABILITY
    assert uvicorn_calls[0]["host"] == "127.0.0.1"
    assert uvicorn_calls[0]["port"] == 19021


def test_nested_snapshot_context_reuses_owner_and_stops_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """嵌套同快照上下文只由最外层释放租约。"""

    lease = _make_lease(tmp_path)
    starts: list[str] = []
    stops: list[str | None] = []

    monkeypatch.setattr(
        TaskEvidenceMcpLease,
        "is_running",
        property(lambda current: current._active_snapshot_id is not None),
    )

    def fake_start(snapshot_id: str) -> TaskEvidenceMcpLease:
        starts.append(snapshot_id)
        lease._active_snapshot_id = snapshot_id
        return lease

    def fake_stop() -> None:
        stops.append(lease._active_snapshot_id)
        lease._active_snapshot_id = None

    monkeypatch.setattr(lease, "start_evidence_snapshot", fake_start)
    monkeypatch.setattr(lease, "stop_evidence_mcp", fake_stop)

    with lease.serve_evidence_snapshot("snapshot-abcdefgh"):
        with lease.serve_evidence_snapshot("snapshot-abcdefgh"):
            assert lease._serve_context_depth == 2
        assert lease._serve_context_depth == 1
        assert stops == []

    assert starts == ["snapshot-abcdefgh", "snapshot-abcdefgh"]
    assert stops == ["snapshot-abcdefgh"]
    assert lease._serve_context_depth == 0


def test_snapshot_context_stops_on_body_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上下文主体异常仍然回收已经由本层启动的服务。"""

    lease = _make_lease(tmp_path)
    stopped: list[bool] = []
    monkeypatch.setattr(
        TaskEvidenceMcpLease,
        "is_running",
        property(lambda current: current._active_snapshot_id is not None),
    )
    monkeypatch.setattr(
        lease,
        "start_evidence_snapshot",
        lambda snapshot_id: setattr(lease, "_active_snapshot_id", snapshot_id) or lease,
    )

    def fake_stop() -> None:
        stopped.append(True)
        lease._active_snapshot_id = None

    monkeypatch.setattr(lease, "stop_evidence_mcp", fake_stop)

    with pytest.raises(RuntimeError, match="body failure"):
        with lease.serve_evidence_snapshot("snapshot-abcdefgh"):
            raise RuntimeError("body failure")

    assert stopped == [True]
    assert lease._serve_context_depth == 0


def test_same_task_receipt_recovery_removes_only_owned_private_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 task 的新租约可核对并清理旧 receipt 与 capability。"""

    first = _make_lease(tmp_path)
    identity = {"pid": 326, "start_time_ticks": 15, "boot_id": "boot"}
    lease_module._write_private_json(
        first.receipt_path,
        first._receipt_payload("snapshot-abcdefgh", identity),
    )
    create_evidence_capability_file(first.capability_file, token=CAPABILITY)
    restarted = _make_lease(tmp_path)
    monkeypatch.setattr(lease_module, "process_identity_activity", lambda _identity: "stopped")

    restarted.recover_evidence_mcp_lease()

    assert not restarted.receipt_path.exists()
    assert not restarted.capability_file.exists()


def test_cross_task_receipt_recovery_fails_closed_and_preserves_identity_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同 task 不能接管同一 runtime 的 receipt，失败时保留凭据待审查。"""

    first = _make_lease(tmp_path)
    identity = {"pid": 327, "start_time_ticks": 16, "boot_id": "boot"}
    lease_module._write_private_json(
        first.receipt_path,
        first._receipt_payload("snapshot-abcdefgh", identity),
    )
    create_evidence_capability_file(first.capability_file, token=CAPABILITY)
    foreign = TaskEvidenceMcpLease(
        "task-foreign",
        first.store_root,
        first.runtime_root,
        port=first.port,
        capability_token=CAPABILITY,
    )
    monkeypatch.setattr(lease_module, "process_identity_activity", lambda _identity: "stopped")

    with pytest.raises(EvidenceMcpLeaseIdentityError, match="configuration mismatch"):
        foreign.recover_evidence_mcp_lease()

    assert foreign.receipt_path.exists()
    assert foreign.capability_file.exists()
