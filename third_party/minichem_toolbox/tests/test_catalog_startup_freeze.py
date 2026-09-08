"""MiniChem 服务启动目录冻结的回归测试。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import minichem_toolbox.catalog as catalog


@pytest.fixture(autouse=True)
def restore_catalog_process_state():
    """保存并恢复目录、工作目录、失败记录和 worker 环境标记。"""

    current_directory = os.getcwd()
    active_snapshot = catalog._ACTIVE_CATALOG_SNAPSHOT
    active_workspace = catalog._ACTIVE_CATALOG_WORKSPACE
    failed_workspaces = set(catalog._FAILED_CATALOG_WORKSPACES)
    marker_values = {
        name: os.environ.get(name)
        for name in (catalog.CATALOG_HASH_ENV, catalog.CATALOG_WORKSPACE_ENV)
    }
    try:
        yield
    finally:
        os.chdir(current_directory)
        catalog._ACTIVE_CATALOG_SNAPSHOT = active_snapshot
        catalog._ACTIVE_CATALOG_WORKSPACE = active_workspace
        catalog._FAILED_CATALOG_WORKSPACES.clear()
        catalog._FAILED_CATALOG_WORKSPACES.update(failed_workspaces)
        for name, value in marker_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _snapshot(sequence: int) -> dict:
    """构造带有协议 hash 的轻量目录，避免测试触发真实后端。"""

    value = {
        "schema_version": 7,
        "sequence": sequence,
        "discovery_mode": "progressive",
        "exposure_policy": "progressive_discovery",
        "catalog_visibility_policy": "public_contract_projection",
        "actions": [],
        "backends": [],
        "resources": [],
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["catalog_hash"] = hashlib.sha256(canonical).hexdigest()
    return value


def _set_workspace(monkeypatch, workspace: Path) -> None:
    """设置服务和目录读取共用的 workspace。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(workspace))
    monkeypatch.delenv(catalog.CATALOG_HASH_ENV, raising=False)
    monkeypatch.delenv(catalog.CATALOG_WORKSPACE_ENV, raising=False)
    catalog._ACTIVE_CATALOG_SNAPSHOT = None
    catalog._ACTIVE_CATALOG_WORKSPACE = None
    catalog._FAILED_CATALOG_WORKSPACES.clear()


def test_startup_freeze_probes_once_writes_canonical_file_and_reuses_active(
    monkeypatch, tmp_path: Path
) -> None:
    """启动冻结替换旧文件，active 读取相同内容且不再次健康探测。"""

    _set_workspace(monkeypatch, tmp_path)
    target = tmp_path / catalog.CATALOG_FILENAME
    target.write_text(json.dumps(_snapshot(0)), encoding="utf-8")
    calls: list[dict] = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs)
        return _snapshot(len(calls))

    monkeypatch.setattr(catalog, "catalog_snapshot", fake_snapshot)
    frozen = catalog.freeze_catalog_snapshot()

    assert frozen == _snapshot(1)
    assert json.loads(target.read_text(encoding="utf-8")) == frozen
    assert target.read_bytes() == json.dumps(
        frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert calls == [{"include_health": True, "discovery_mode": "progressive", "resource_budget": None}]
    assert catalog.active_catalog_snapshot() == frozen
    assert len(calls) == 1


def test_probe_failure_removes_stale_file_and_does_not_return_old_catalog(
    monkeypatch, tmp_path: Path
) -> None:
    """新探测失败时旧普通文件被清理，不能伪装成本次服务目录。"""

    _set_workspace(monkeypatch, tmp_path)
    target = tmp_path / catalog.CATALOG_FILENAME
    target.write_bytes(json.dumps(_snapshot(0)).encode())

    def failing_snapshot(**_kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(catalog, "catalog_snapshot", failing_snapshot)
    with pytest.raises(RuntimeError, match="probe failed"):
        catalog.freeze_catalog_snapshot()
    assert not target.exists()
    assert catalog.CATALOG_HASH_ENV not in os.environ


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("/private/secret/boom.json"),
        ValueError(r"C:\private\secret\boom.json"),
        catalog.CatalogFreezeError("file:///private/secret/boom.json"),
    ],
)
def test_freeze_failure_record_omits_exception_paths(
    monkeypatch, tmp_path: Path, error: BaseException
) -> None:
    """冻结失败状态只保存稳定类型、错误码和固定摘要，不保存异常原文。"""

    _set_workspace(monkeypatch, tmp_path)

    def failing_snapshot(**_kwargs):
        raise error

    monkeypatch.setattr(catalog, "catalog_snapshot", failing_snapshot)
    with pytest.raises(type(error)):
        catalog.freeze_catalog_snapshot()

    failure = catalog._FAILED_CATALOG_WORKSPACES[tmp_path.resolve()]
    assert set(failure) == {"code", "exception_type", "message"}
    assert failure["message"] == "MiniChem 目录冻结失败"
    assert str(error) not in str(failure)
    with pytest.raises(catalog.CatalogFreezeError) as raised:
        catalog.active_catalog_snapshot()
    assert str(error) not in str(raised.value)


def test_atomic_write_failure_removes_previous_file_and_blocks_legacy_read(
    monkeypatch, tmp_path: Path
) -> None:
    """写入失败清理旧普通文件，并阻止后续兼容路径重探测。"""

    _set_workspace(monkeypatch, tmp_path)
    target = tmp_path / catalog.CATALOG_FILENAME
    previous = _snapshot(0)
    target.write_bytes(json.dumps(previous, sort_keys=True, separators=(",", ":")).encode())
    monkeypatch.setattr(catalog, "catalog_snapshot", lambda **_: _snapshot(1))

    def fail_write(_target, _payload):
        raise catalog.CatalogFreezeError("write failed")

    monkeypatch.setattr(catalog, "_write_catalog_atomically", fail_write)
    with pytest.raises(catalog.CatalogFreezeError, match="write failed"):
        catalog.freeze_catalog_snapshot()
    assert not target.exists()
    with pytest.raises(catalog.CatalogFreezeError, match="冻结已失败"):
        catalog.active_catalog_snapshot()


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_freeze_rejects_unsafe_catalog_target(monkeypatch, tmp_path: Path, kind: str) -> None:
    """固定目标为符号链接或特殊类型时不覆盖目标。"""

    _set_workspace(monkeypatch, tmp_path)
    target = tmp_path / catalog.CATALOG_FILENAME
    outside = tmp_path.parent / f"catalog-target-{kind}.json"
    outside.write_text("sentinel", encoding="utf-8")
    if kind == "symlink":
        target.symlink_to(outside)
    else:
        target.mkdir()
    monkeypatch.setattr(catalog, "catalog_snapshot", lambda **_: _snapshot(1))

    with pytest.raises(catalog.CatalogFreezeError):
        catalog.freeze_catalog_snapshot()
    if kind == "symlink":
        assert target.is_symlink()
        assert outside.read_text(encoding="utf-8") == "sentinel"
    else:
        assert target.is_dir()


def test_active_rejects_deleted_or_replaced_file_after_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    """冻结后文件被删改时 active fail closed，不重新探测另一份 hash。"""

    _set_workspace(monkeypatch, tmp_path)
    calls: list[int] = []

    def fake_snapshot(**_kwargs):
        calls.append(1)
        return _snapshot(len(calls))

    monkeypatch.setattr(catalog, "catalog_snapshot", fake_snapshot)
    frozen = catalog.freeze_catalog_snapshot()
    (tmp_path / catalog.CATALOG_FILENAME).write_text("{}", encoding="utf-8")

    with pytest.raises(catalog.CatalogFreezeError):
        catalog.active_catalog_snapshot()
    assert len(calls) == 1
    assert frozen["catalog_hash"] != _snapshot(2)["catalog_hash"]


def test_child_process_reads_same_frozen_file_without_reprobing(
    monkeypatch, tmp_path: Path
) -> None:
    """继承服务标记的 worker 进程从同一文件读取 hash，工作目录可不同。"""

    _set_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr(catalog, "catalog_snapshot", lambda **_: _snapshot(1))
    frozen = catalog.freeze_catalog_snapshot(publish_marker=True)
    environment = dict(os.environ)
    environment.pop("MINICHEM_MCP_WORKSPACE", None)
    repository_root = Path(__file__).resolve().parents[3]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository_root / "src"), str(repository_root / "third_party" / "minichem_toolbox" / "src")]
    )
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from minichem_toolbox.catalog import active_catalog_hash; print(active_catalog_hash())",
        ],
        cwd=tmp_path.parent,
        env=environment,
        text=True,
    ).strip()
    assert output == frozen["catalog_hash"]


def test_concurrent_freezes_leave_one_complete_valid_json(
    monkeypatch, tmp_path: Path
) -> None:
    """并发启动写入串行化，读者只会看到完整 JSON。"""

    _set_workspace(monkeypatch, tmp_path)
    calls: list[int] = []
    call_lock = threading.Lock()

    def fake_snapshot(**_kwargs):
        with call_lock:
            calls.append(1)
        time.sleep(0.005)
        return _snapshot(1)

    monkeypatch.setattr(catalog, "catalog_snapshot", fake_snapshot)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            catalog.freeze_catalog_snapshot()
        except BaseException as exc:  # 线程测试需回收异常再在主线程断言
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(calls) == 4
    value = json.loads((tmp_path / catalog.CATALOG_FILENAME).read_text(encoding="utf-8"))
    assert value == _snapshot(1)
    assert catalog.active_catalog_snapshot() == value


def test_server_start_freezes_before_building_fastmcp(monkeypatch, tmp_path: Path) -> None:
    """CLI/惰性入口先冻结目录，再把同一快照交给 FastMCP 构造。"""

    import minichem_mcp_tools.server as server

    events: list[object] = []
    expected = _snapshot(1)

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(server, "workspace_root", lambda: tmp_path)
    monkeypatch.setattr(
        server,
        "freeze_catalog_snapshot",
        lambda *, workspace, discovery_mode: events.append(("freeze", workspace, discovery_mode))
        or expected,
    )
    monkeypatch.setattr(
        server,
        "create_server",
        lambda *, discovery_mode, catalog_snapshot: events.append(
            ("create", discovery_mode, catalog_snapshot)
        )
        or "server",
    )
    monkeypatch.setattr(server, "_publish_catalog_marker", lambda *_args: None)
    assert server._start_server("full") == "server"
    assert events == [("freeze", tmp_path, "full"), ("create", "full", expected)]


def test_concurrent_server_start_keeps_freeze_create_publish_atomic(
    monkeypatch, tmp_path: Path
) -> None:
    """并发 workspace 启动不能在 freeze 与 create 之间交叉替换 active 快照。"""

    import minichem_mcp_tools.server as server

    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    local = threading.local()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    events: list[tuple[str, Path]] = []
    snapshots = {
        first_workspace: _snapshot(1),
        second_workspace: _snapshot(2),
    }

    monkeypatch.setattr(server, "workspace_root", lambda: local.workspace)

    def fake_freeze(*, workspace, discovery_mode):
        events.append(("freeze", workspace))
        if workspace == first_workspace:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        return snapshots[workspace]

    monkeypatch.setattr(server, "freeze_catalog_snapshot", fake_freeze)
    monkeypatch.setattr(
        server,
        "create_server",
        lambda *, discovery_mode, catalog_snapshot: events.append(
            ("create", next(path for path, item in snapshots.items() if item == catalog_snapshot))
        )
        or "server",
    )
    monkeypatch.setattr(
        server,
        "_publish_catalog_marker",
        lambda snapshot, workspace: events.append(("publish", workspace)),
    )

    def start(workspace: Path) -> None:
        local.workspace = workspace
        server._start_server()

    first = threading.Thread(target=start, args=(first_workspace,))
    second = threading.Thread(target=start, args=(second_workspace,))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert events == [
        ("freeze", first_workspace),
        ("create", first_workspace),
        ("publish", first_workspace),
        ("freeze", second_workspace),
        ("create", second_workspace),
        ("publish", second_workspace),
    ]


def test_direct_create_server_cannot_interleave_with_startup_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    """直接构造服务也必须遵守启动锁，不能在另一个启动冻结后替换 active。"""

    import minichem_mcp_tools.server as server

    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    local = threading.local()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    snapshots = {
        first_workspace: _snapshot(1),
        second_workspace: _snapshot(2),
    }

    monkeypatch.setattr(server, "workspace_root", lambda: local.workspace)

    errors: list[BaseException] = []

    def fake_freeze(*, workspace=None, discovery_mode):
        workspace = workspace or local.workspace
        if workspace == first_workspace:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        return snapshots[workspace]

    monkeypatch.setattr(server, "freeze_catalog_snapshot", fake_freeze)
    monkeypatch.setattr(
        server,
        "validate_active_catalog_snapshot",
        lambda snapshot: snapshot,
    )
    monkeypatch.setattr(server, "load_tool_config", lambda: {})
    monkeypatch.setattr(server, "FastMCP", lambda **kwargs: kwargs)
    monkeypatch.setattr(server, "toolbox_overview", lambda **kwargs: "")
    monkeypatch.setattr(server, "open_execution_prompt", lambda **kwargs: "")
    monkeypatch.setattr(server, "register_public_tools", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "register_catalog_resources", lambda *_args, **_kwargs: None)

    def start_first() -> None:
        local.workspace = first_workspace
        try:
            server._start_server()
        except BaseException as exc:
            errors.append(exc)

    def create_second() -> None:
        local.workspace = second_workspace
        try:
            server.create_server()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=start_first)
    second = threading.Thread(target=create_second)
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors


def test_explicit_server_snapshot_must_match_active_and_preserve_marker(
    monkeypatch, tmp_path: Path
) -> None:
    """显式服务快照必须与 active/file 一致，并且不能清掉已有 worker marker。"""

    import minichem_mcp_tools.server as server

    _set_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr(catalog, "catalog_snapshot", lambda **_: _snapshot(1))
    frozen = catalog.freeze_catalog_snapshot(publish_marker=True)
    marker_before = (
        os.environ[catalog.CATALOG_HASH_ENV],
        os.environ[catalog.CATALOG_WORKSPACE_ENV],
    )

    monkeypatch.setattr(server, "FastMCP", lambda **kwargs: kwargs)
    monkeypatch.setattr(server, "load_tool_config", lambda: {})
    monkeypatch.setattr(
        server,
        "toolbox_overview",
        lambda **kwargs: json.dumps(kwargs["snapshot"], sort_keys=True),
    )
    monkeypatch.setattr(server, "open_execution_prompt", lambda **_: "")
    monkeypatch.setattr(server, "register_public_tools", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "register_catalog_resources", lambda *_args, **_kwargs: None)

    created = server.create_server(catalog_snapshot=frozen)
    assert json.loads(created["instructions"]) == frozen
    assert (
        os.environ[catalog.CATALOG_HASH_ENV],
        os.environ[catalog.CATALOG_WORKSPACE_ENV],
    ) == marker_before

    with pytest.raises(catalog.CatalogFreezeError, match="不一致"):
        server.create_server(catalog_snapshot=_snapshot(2))
    with pytest.raises(catalog.CatalogFreezeError, match="catalog_hash"):
        server.create_server(catalog_snapshot={"catalog_hash": "not-a-hash"})
    with pytest.raises(catalog.CatalogFreezeError, match="discovery_mode"):
        server.create_server(discovery_mode="full", catalog_snapshot=frozen)


def test_create_server_reuses_one_health_probe_for_catalog_and_inspect(
    monkeypatch, tmp_path: Path
) -> None:
    """真实 FastMCP 注册流程只做一次全量健康探测，后续合同读取不探测。"""

    import minichem_mcp_tools.server as server
    from minichem_toolbox.discovery import inspect_action

    _set_workspace(monkeypatch, tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_probe(specifications):
        values = tuple(specification.id for specification in specifications)
        calls.append(values)
        return {
            specification_id: {"available": True, "status": "available"}
            for specification_id in values
        }

    monkeypatch.setattr(catalog, "probe_all_backends", fake_probe)
    server.create_server()
    active = catalog.active_catalog_snapshot()
    inspect_action(
        active["actions"][0]["id"],
        backend_id=active["actions"][0]["backend_ids"][0],
        detail_level="contract",
    )
    assert len(calls) == 1


def test_direct_create_server_does_not_pollute_a_later_workspace(
    monkeypatch, tmp_path: Path
) -> None:
    """普通库构造不发布 worker 标记，切换 workspace 后可独立读取目录。"""

    import minichem_mcp_tools.server as server

    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    _set_workspace(monkeypatch, first_workspace)
    sequence = iter((_snapshot(1), _snapshot(2)))
    monkeypatch.setattr(catalog, "catalog_snapshot", lambda **_: next(sequence))
    monkeypatch.setattr(
        catalog,
        "probe_all_backends",
        lambda specifications: {
            specification.id: {"available": True, "status": "available"}
            for specification in specifications
        },
    )
    server.create_server()
    assert catalog.CATALOG_HASH_ENV not in os.environ

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(second_workspace))
    assert catalog.active_catalog_snapshot()["catalog_hash"] == _snapshot(2)[
        "catalog_hash"
    ]


def test_server_keeps_lazy_mcp_import_entrypoint(monkeypatch) -> None:
    """访问模块 mcp 属性时仍能兼容 FastMCP 的 module:mcp 入口。"""

    import minichem_mcp_tools.server as server

    server.__dict__.pop("mcp", None)
    sentinel = object()
    monkeypatch.setattr(server, "_start_server", lambda: sentinel)
    assert server.mcp is sentinel
    server.__dict__.pop("mcp", None)


def test_lazy_mcp_entrypoint_starts_once_under_concurrent_access(monkeypatch) -> None:
    """并发首次访问 module:mcp 只允许一个线程执行服务启动。"""

    import minichem_mcp_tools.server as server

    server.__dict__.pop("mcp", None)
    gate = threading.Barrier(2)
    calls: list[int] = []
    sentinel = object()

    def fake_start() -> object:
        calls.append(1)
        time.sleep(0.02)
        return sentinel

    monkeypatch.setattr(server, "_start_server", fake_start)
    outputs: list[object] = []

    def read_mcp() -> None:
        gate.wait(timeout=2)
        outputs.append(server.mcp)

    threads = [threading.Thread(target=read_mcp) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert len(calls) == 1
    assert outputs == [sentinel, sentinel]
    server.__dict__.pop("mcp", None)


def test_failed_cross_workspace_freeze_blocks_legacy_reprobe(
    monkeypatch, tmp_path: Path
) -> None:
    """跨 workspace 启动失败后，active 不得重新探测或读取另一份旧目录。"""

    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    _set_workspace(monkeypatch, first_workspace)
    monkeypatch.setattr(catalog, "catalog_snapshot", lambda **_: _snapshot(1))
    catalog.freeze_catalog_snapshot(publish_marker=True)

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(second_workspace))

    def failing_snapshot(**_kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(catalog, "catalog_snapshot", failing_snapshot)
    with pytest.raises(RuntimeError, match="probe failed"):
        catalog.freeze_catalog_snapshot()
    assert not (second_workspace / catalog.CATALOG_FILENAME).exists()

    monkeypatch.setattr(catalog, "catalog_snapshot", lambda **_: _snapshot(2))
    with pytest.raises(catalog.CatalogFreezeError, match="冻结已失败"):
        catalog.active_catalog_snapshot()
    refrozen = catalog.freeze_catalog_snapshot()
    assert catalog.active_catalog_snapshot() == refrozen


def test_invalid_discovery_mode_cleans_stale_file_and_blocks_active(
    monkeypatch, tmp_path: Path
) -> None:
    """非法发现模式也要清理旧普通文件并留下失败状态。"""

    _set_workspace(monkeypatch, tmp_path)
    target = tmp_path / catalog.CATALOG_FILENAME
    target.write_text(json.dumps(_snapshot(0)), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported tool discovery mode"):
        catalog.freeze_catalog_snapshot(discovery_mode="not-a-mode")
    assert not target.exists()
    with pytest.raises(catalog.CatalogFreezeError, match="冻结已失败"):
        catalog.active_catalog_snapshot()


def test_direct_create_server_invalid_mode_marks_workspace_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """普通库构造触发的非法模式也不能绕过 workspace 失败状态。"""

    import minichem_mcp_tools.server as server

    _set_workspace(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Unsupported tool discovery mode"):
        server.create_server(discovery_mode="not-a-mode")
    with pytest.raises(catalog.CatalogFreezeError, match="冻结已失败"):
        catalog.active_catalog_snapshot()
