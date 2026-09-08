"""scope capability 校验与请求 header 解析的分支回归测试。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from mcp.server.lowlevel.server import request_ctx

from minichem_mcp_tools import scope_auth
from minichem_toolbox import scope as scope_module
from minichem_toolbox.scope import ScopeCapabilityError


def _valid_scope_record(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """创建可通过 schema 校验的 scope 记录，供每个分支用例独立修改。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    token_hash = "a" * 64
    return Path(f"cap_{token_hash}.json"), {
        "schema_version": 1,
        "token_sha256": token_hash,
        "task_id": "task-1",
        "batch_id": "batch-1",
        "combination_id": "combo-1",
        "scope_id": "scope-1",
        "workspace": str(workspace),
        "revoked": False,
        "revoked_at": None,
    }


@pytest.mark.parametrize(
    ("case_name", "mutate", "exception_type"),
    [
        ("unknown_filename", lambda path, record: (Path("unknown"), record), ScopeCapabilityError),
        ("non_object", lambda path, record: (path, []), ScopeCapabilityError),
        ("unknown_field", lambda path, record: (path, {**record, "extra": True}), ScopeCapabilityError),
        ("boolean_schema", lambda path, record: (path, {**record, "schema_version": True}), ScopeCapabilityError),
        ("wrong_schema", lambda path, record: (path, {**record, "schema_version": 2}), ScopeCapabilityError),
        ("non_string_hash", lambda path, record: (path, {**record, "token_sha256": 1}), ScopeCapabilityError),
        ("invalid_hash", lambda path, record: (path, {**record, "token_sha256": "z" * 64}), ScopeCapabilityError),
        ("non_string_task", lambda path, record: (path, {**record, "task_id": 1}), ScopeCapabilityError),
        ("unsafe_task", lambda path, record: (path, {**record, "task_id": ""}), ValueError),
        ("non_string_batch", lambda path, record: (path, {**record, "batch_id": 1}), ScopeCapabilityError),
        ("unsafe_batch", lambda path, record: (path, {**record, "batch_id": "bad\nvalue"}), ValueError),
        ("non_string_combination", lambda path, record: (path, {**record, "combination_id": 1}), ScopeCapabilityError),
        ("unsafe_combination", lambda path, record: (path, {**record, "combination_id": "x" * 257}), ValueError),
        ("non_string_scope", lambda path, record: (path, {**record, "scope_id": 1}), ScopeCapabilityError),
        ("unsafe_scope", lambda path, record: (path, {**record, "scope_id": "bad\x00value"}), ValueError),
        ("non_string_workspace", lambda path, record: (path, {**record, "workspace": 1}), ScopeCapabilityError),
        ("empty_workspace", lambda path, record: (path, {**record, "workspace": "  "}), ScopeCapabilityError),
        ("missing_workspace", lambda path, record: (path, {**record, "workspace": "/does/not/exist"}), ScopeCapabilityError),
        ("non_boolean_revoked", lambda path, record: (path, {**record, "revoked": "false"}), ScopeCapabilityError),
        ("revoked_without_timestamp", lambda path, record: (path, {**record, "revoked": True}), ScopeCapabilityError),
        ("revoked_empty_timestamp", lambda path, record: (path, {**record, "revoked": True, "revoked_at": " "}), ScopeCapabilityError),
        ("active_with_timestamp", lambda path, record: (path, {**record, "revoked_at": "now"}), ScopeCapabilityError),
        ("filename_hash_mismatch", lambda path, record: (Path(f"cap_{'b' * 64}.json"), record), ScopeCapabilityError),
    ],
)
def test_scope_record_validation_rejects_each_malformed_branch(
    tmp_path: Path,
    case_name: str,
    mutate: Callable[[Path, dict[str, Any]], tuple[Path, Any]],
    exception_type: type[Exception],
) -> None:
    """每个 schema、身份、状态和文件名拒绝分支都必须保持 fail-closed。"""

    del case_name
    path, record = _valid_scope_record(tmp_path)
    mutated_path, mutated_record = mutate(path, record)
    with pytest.raises(exception_type):
        scope_module._validate_scope_record(mutated_path, mutated_record)


def test_scope_record_validation_accepts_revoked_tombstone(tmp_path: Path) -> None:
    """带时间戳的 revoked tombstone 是唯一允许的撤销状态记录。"""

    path, record = _valid_scope_record(tmp_path)
    record["revoked"] = True
    record["revoked_at"] = "2026-08-23T00:00:00+00:00"
    assert scope_module._validate_scope_record(path, record) == record


class _HeaderContainer:
    """为请求 header fallback 分支提供可控的最小替身。"""

    def __init__(self, value: Any = None, error: type[Exception] | None = None) -> None:
        self.value = value
        self.error = error

    def get(self, _name: str) -> Any:
        if self.error is not None:
            raise self.error("header lookup failed")
        return self.value


class _Request:
    """同时模拟 Starlette scope 和 headers 属性。"""

    def __init__(self, request_scope: Any, headers: Any) -> None:
        self.scope = request_scope
        self.headers = headers


class _RequestWithoutScope:
    """scope 属性不可用时仍允许测试 headers fallback。"""

    def __init__(self, headers: Any) -> None:
        self.headers = headers

    @property
    def scope(self) -> Any:
        raise AttributeError("scope is unavailable")


@pytest.mark.parametrize(
    ("request_obj", "expected"),
    [
        (_Request({"headers": []}, _HeaderContainer("fallback")), None),
        (_Request({}, _HeaderContainer("fallback")), "fallback"),
        (_Request(None, _HeaderContainer("fallback")), "fallback"),
        (_RequestWithoutScope(_HeaderContainer("fallback")), "fallback"),
        (_Request({}, _HeaderContainer(None)), None),
        (_Request({}, _HeaderContainer(123)), None),
        (_Request({}, _HeaderContainer("  ")), None),
        (_Request({}, _HeaderContainer("a,b")), None),
        (_Request({}, _HeaderContainer("  accepted  ")), "accepted"),
        (_Request({}, _HeaderContainer(error=AttributeError)), None),
        (_Request({}, _HeaderContainer(error=TypeError)), None),
    ],
)
def test_request_header_value_covers_scope_and_headers_fallbacks(
    request_obj: Any, expected: str | None
) -> None:
    """请求上下文异常、类型错误、空值和重复 header 都保持拒绝。"""

    assert scope_auth._request_header_value(request_obj) == expected


def test_request_header_value_uses_asgi_header_parser_when_scope_has_headers() -> None:
    """存在 ASGI headers 时只使用原始 header 解析结果，不回退到属性。"""

    request = _Request(
        {"headers": [(b"X-MiniChem-Capability", b"from-scope")]},
        _HeaderContainer("from-property"),
    )
    assert scope_auth._request_header_value(request) == "from-scope"


def test_scheduler_root_rejects_missing_file_and_symlink(tmp_path: Path) -> None:
    """scheduler 根缺失、指向文件或包含 symlink 时必须在启动前失败。"""

    with pytest.raises(ScopeCapabilityError):
        scope_auth._scheduler_root(None)  # type: ignore[arg-type]
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ScopeCapabilityError):
        scope_auth._scheduler_root(file_path)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("当前文件系统不支持符号链接")
    with pytest.raises(ScopeCapabilityError):
        scope_auth._scheduler_root(link)


def test_resolve_scheduler_root_uses_explicit_value_or_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scheduler 根缺省时拒绝，显式参数和环境变量都解析为私有目录。"""

    monkeypatch.delenv(scope_module.SHARED_SCHEDULER_ROOT_ENV, raising=False)
    with pytest.raises(ScopeCapabilityError, match="explicit shared scheduler root"):
        scope_auth.resolve_scheduler_root(None)
    explicit = tmp_path / "explicit"
    assert scope_auth.resolve_scheduler_root(explicit) == explicit.resolve()
    from_environment = tmp_path / "from-environment"
    monkeypatch.setenv(scope_module.SHARED_SCHEDULER_ROOT_ENV, str(from_environment))
    assert scope_auth.resolve_scheduler_root("") == from_environment.resolve()


@contextmanager
def _set_request_context(request: Any):
    """在单个测试上下文中安装 MCP low-level request 替身。"""

    token = request_ctx.set(SimpleNamespace(request=request))
    try:
        yield
    finally:
        request_ctx.reset(token)


def test_authenticate_current_mcp_request_covers_context_failures_and_rebinding(
    tmp_path: Path,
) -> None:
    """MCP message 缺失配置时拒绝已绑定请求，成功时认证并恢复 ContextVar。"""

    with _set_request_context(None):
        with scope_auth.authenticate_current_mcp_request() as selected:
            assert selected is None

    missing_config_request = SimpleNamespace(scope={"state": {}})
    with _set_request_context(missing_config_request):
        with scope_auth.authenticate_current_mcp_request() as selected:
            assert selected is None

    registry = scope_module.ScopeCapabilityRegistry(tmp_path / "registry")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability = registry.provision_scope_capability(
        task_id="task",
        batch_id="batch",
        combination_id="combo",
        scope_id="scope",
        workspace=workspace,
        token="T" * 48,
    )
    config = (registry, tmp_path / "scheduler")
    invalid_request = SimpleNamespace(
        scope={"state": {scope_auth._SCOPE_AUTH_STATE_KEY: config}, "headers": []}
    )
    with _set_request_context(invalid_request):
        with pytest.raises(ScopeCapabilityError, match="invalid capability"):
            with scope_auth.authenticate_current_mcp_request():
                pass

    valid_request = SimpleNamespace(
        scope={
            "state": {scope_auth._SCOPE_AUTH_STATE_KEY: config},
            "headers": [(b"x-minichem-capability", capability.token.encode())],
        }
    )
    with _set_request_context(valid_request):
        assert scope_module.current_scope() is None
        with scope_auth.authenticate_current_mcp_request() as selected:
            assert selected is not None
            assert selected.scope_id == "scope"
            assert selected.scheduler_root == (tmp_path / "scheduler").resolve()
        assert scope_module.current_scope() is None


def test_authenticate_current_mcp_request_rejects_lost_configuration_for_bound_scope(
    tmp_path: Path,
) -> None:
    """外层已绑定 scope 但逐消息配置丢失时不能静默复用旧请求身份。"""

    registry = scope_module.ScopeCapabilityRegistry(tmp_path / "registry")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability = registry.provision_scope_capability(
        task_id="task",
        batch_id="batch",
        combination_id="combo",
        scope_id="scope",
        workspace=workspace,
        token="U" * 48,
    )
    with scope_module.activate_scope(capability.scope):
        with _set_request_context(SimpleNamespace(scope={"state": {}})):
            with pytest.raises(ScopeCapabilityError, match="configuration is unavailable"):
                with scope_auth.authenticate_current_mcp_request():
                    pass


def test_revoke_scope_capability_covers_missing_and_corrupt_records(
    tmp_path: Path,
) -> None:
    """撤销操作遇到缺失、损坏或重复 tombstone 时分别 fail-closed 或幂等返回。"""

    registry = scope_module.ScopeCapabilityRegistry(tmp_path / "registry")
    with pytest.raises(ScopeCapabilityError, match="invalid capability"):
        registry.revoke_scope_capability("M" * 48)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability = registry.provision_scope_capability(
        task_id="task",
        batch_id="batch",
        combination_id="combo",
        scope_id="scope",
        workspace=workspace,
        token="V" * 48,
    )
    capability.record_path.write_text("{", encoding="utf-8")
    capability.record_path.chmod(0o600)
    with pytest.raises(ScopeCapabilityError, match="invalid capability"):
        registry.revoke_scope_capability(capability.token)

    capability.record_path.write_text(
        '{"schema_version": 1, "token_sha256": "invalid"}', encoding="utf-8"
    )
    capability.record_path.chmod(0o600)
    with pytest.raises(ScopeCapabilityError, match="invalid capability"):
        registry.revoke_scope_capability(capability.token)

    second_registry = scope_module.ScopeCapabilityRegistry(tmp_path / "registry-2")
    replacement_workspace = tmp_path / "workspace-2"
    replacement_workspace.mkdir()
    replacement = second_registry.provision_scope_capability(
        task_id="task-2",
        batch_id="batch-2",
        combination_id="combo-2",
        scope_id="scope-2",
        workspace=replacement_workspace,
        token="W" * 48,
    )
    second_registry.revoke_scope_capability(replacement.token)
    second_registry.revoke_scope_capability(replacement.token)
