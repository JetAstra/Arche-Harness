"""Codex API/local_login 认证和命令读隔离回归测试。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from chemistry_multiagent.harness import codex_auth, codex_sdk_adapter
from chemistry_multiagent.harness.codex_auth import (
    CODEX_AUTH_MODE_LOCAL_LOGIN,
    CODEX_INVESTIGATION_PERMISSION_PROFILE,
    CodexAuthCloneSession,
    cleanup_stale_codex_auth_clone,
    normalize_codex_auth_mode,
    prepare_codex_permission_profile,
)
from chemistry_multiagent.harness.codex_sdk_adapter import (
    build_codex_capability_config,
    calculate_codex_thread_policy_digest,
    CodexProcessSandboxSpec,
    CodexSdkRunResult,
)
from chemistry_multiagent.harness.base import HarnessError
from chemistry_multiagent.harness.base import HarnessResponse
from chemistry_multiagent.harness.session_store import CodexSessionOwner, CodexSessionStore
from chemistry_multiagent.harness.codex_provider_migration import (
    persist_provider_migration_target_thread,
    resolve_provider_migration_target_store_read_only,
    resolve_explicit_provider_migration,
    select_explicit_provider_migration_store,
)
from chemistry_multiagent.investigation import codex_runtime as codex_runtime_module
from chemistry_multiagent.investigation.codex_runtime import (
    CodexProviderSettings,
    EvidenceMcpSettings,
    InvestigationCodexRuntime,
    InvestigationMcpSettings,
)
from chemistry_multiagent.investigation.async_jobs import (
    InvestigationJobIdentity,
    InvestigationJobState,
)
from chemistry_multiagent.investigation.prompts import (
    build_initial_investigation_request,
    build_investigation_job_wake_request,
)


def _write_private_source(path: Path, payload: bytes = b"{\"access_token\":\"test\"}\n") -> None:
    """创建只供测试使用的 0600 登录 source。"""

    path.write_bytes(payload)
    os.chmod(path, 0o600)


def _write_test_completed_receipt(
    kwargs: dict[str, object],
    *,
    thread_id: str,
    turn_id: str,
) -> None:
    """为成功 fake adapter turn 写入与 runtime 参数一致的 durable receipt。"""

    codex_sdk_adapter._write_completed_turn_receipt(
        kwargs["completed_turn_receipt_root"],
        thread_id=thread_id,
        turn_id=turn_id,
        request_digest=kwargs["request_digest"],
        turn_count_before=kwargs["turn_count_before"],
        thread_policy_digest=kwargs["thread_policy_digest"],
    )


def test_local_login_clone_preserves_source_and_removes_clone(tmp_path: Path) -> None:
    """clone 生命周期不修改 source，退出 app-server 上下文后不残留凭据。"""

    source = tmp_path / "source-auth.json"
    home = tmp_path / "codex-home"
    _write_private_source(source)
    before = source.stat()
    before_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with CodexAuthCloneSession(home, source):
        clone = home / "auth.json"
        marker = home / "auth.json.arche-clone.json"
        assert clone.read_bytes() == source.read_bytes()
        assert stat.S_IMODE(clone.stat().st_mode) == 0o600
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600

    after = source.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_digest
    assert not (home / "auth.json").exists()
    assert not (home / "auth.json.arche-clone.json").exists()


def test_stale_auth_clone_is_cleaned_only_with_dead_process_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """启动清理只接受已确认失效的 marker，不猜测活动 PID。"""

    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_bytes(b"stale")
    (home / "auth.json.arche-clone.json").write_text(
        json.dumps({"schema_version": 1, "pid": 99999999, "proc_start_time": "1"}),
        encoding="utf-8",
    )
    os.chmod(home / "auth.json", 0o600)
    os.chmod(home / "auth.json.arche-clone.json", 0o600)

    cleanup_stale_codex_auth_clone(home)
    assert not (home / "auth.json").exists()
    assert not (home / "auth.json.arche-clone.json").exists()

    (home / "auth.json").write_bytes(b"untracked")
    os.chmod(home / "auth.json", 0o600)
    with pytest.raises(Exception, match="untracked auth clone"):
        cleanup_stale_codex_auth_clone(home)


def test_permission_profile_denies_virtual_codex_home_and_preserves_mcp_shell_surface(
    tmp_path: Path,
) -> None:
    """自定义 profile 负责读隔离，同时 Investigation shell/unified_exec 和 HTTP MCP 仍在。"""

    home = tmp_path / "codex-home"
    profile = prepare_codex_permission_profile(home)
    assert profile == CODEX_INVESTIGATION_PERMISSION_PROFILE
    config_text = (home / "config.toml").read_text(encoding="utf-8")
    assert 'default_permissions = "arche_investigation_no_codex_home"' in config_text
    assert '"/.arche/codex-home" = "deny"' in config_text
    assert '"/root/.codex/auth.json" = "deny"' in config_text

    config = build_codex_capability_config(
        thread_config={
            "model_provider": "openai",
            "mcp_servers": {
                "minichem_toolbox": {
                    "url": "http://127.0.0.1:9012/mcp",
                    "env_http_headers": {"X-MiniChem-Capability": "MINICHEM_TOKEN"},
                    "required": True,
                    "startup_timeout_sec": 60,
                    "tool_timeout_sec": 21600,
                }
            },
        },
        workspace=tmp_path / "workspace",
        profile="investigation",
        env_key="ARCHE_CODEX_LOCAL_LOGIN",
        environment_keys=("MINICHEM_TOKEN",),
    )
    assert config["features"]["shell_tool"] is True
    assert config["features"]["unified_exec"] is True
    assert config["mcp_servers"]["minichem_toolbox"]["url"].startswith("http://127.0.0.1")
    assert (
        config["mcp_servers"]["minichem_toolbox"]["default_tools_approval_mode"]
        == "approve"
    )
    assert "ARCHE_CODEX_LOCAL_LOGIN" in config["shell_environment_policy"]["exclude"]
    assert "MINICHEM_TOKEN" in config["shell_environment_policy"]["exclude"]


def test_permission_profile_preserves_existing_codex_configuration(tmp_path: Path) -> None:
    """已有 Run Bundle 配置保留 Codex 选项，冲突的读隔离配置才会阻断。"""

    home = tmp_path / "codex-home"
    prepare_codex_permission_profile(home)
    config_path = home / "config.toml"
    original = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        'approval_policy = "never"\n'
        + original
        + '\n[permissions.other]\nextends = ":workspace"\n',
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)

    assert prepare_codex_permission_profile(home) == CODEX_INVESTIGATION_PERMISSION_PROFILE
    preserved = config_path.read_text(encoding="utf-8")
    assert 'approval_policy = "never"' in preserved
    assert '[permissions.other]\nextends = ":workspace"' in preserved

    config_path.write_text(
        preserved.replace('"/.arche/codex-home" = "deny"', '"/.arche/codex-home" = "read"'),
        encoding="utf-8",
    )
    with pytest.raises(HarnessError, match="read boundary"):
        prepare_codex_permission_profile(home)


def test_sdk_omits_none_sandbox_from_thread_start_wire_payload() -> None:
    """Codex SDK 的 sandbox=None 经过 exclude_none 后不会发送非法 null。"""

    from openai_codex.generated.v2_all import ThreadStartParams

    params = ThreadStartParams(
        model="gpt-5.6-sol",
        model_provider="openai",
        sandbox=None,
    )
    payload = params.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert "sandbox" not in payload

    # 生产 local_login profile 通过 sandbox=None 让 SDK 省略 legacy sandbox 字段，
    # 再由 Codex 0.149 permission profile 提供实际的读隔离边界。
    from chemistry_multiagent.harness.codex_sdk_adapter import sandbox_for_codex_profile

    assert (
        sandbox_for_codex_profile(
            "investigation",
            permission_profile=CODEX_INVESTIGATION_PERMISSION_PROFILE,
        )
        is None
    )


def test_auth_mode_is_explicit_and_local_mode_does_not_fallback() -> None:
    """缺省认证仍是 API，未知模式和本地模式没有隐式 provider fallback。"""

    assert normalize_codex_auth_mode(None) == "api"
    assert normalize_codex_auth_mode(CODEX_AUTH_MODE_LOCAL_LOGIN) == CODEX_AUTH_MODE_LOCAL_LOGIN
    with pytest.raises(Exception, match="api or local_login"):
        normalize_codex_auth_mode("automatic")


def test_local_login_only_changes_investigation_and_keeps_task_review_api(
    tmp_path: Path,
) -> None:
    """全局本机登录选择不改变同一流程中的 Task Review API provider。"""

    runtime = InvestigationCodexRuntime(
        session_root=tmp_path / "sessions",
        trace_root=tmp_path / "traces",
        role="task_review",
        evidence_mcp=EvidenceMcpSettings(
            "http://127.0.0.1:9020/mcp",
            {"Authorization": "EVIDENCE_AUTH_TOKEN"},
        ),
        environment={
            "ARCHE_CODEX_AUTH_MODE": "local_login",
            "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION": "1",
            "ARCHE_REVIEW_CODEX_API_KEY": "review-secret",
            "ARCHE_REVIEW_CODEX_BASE_URL": "https://review.example/v1",
            "ARCHE_REVIEW_CODEX_MODEL_NAME": "review-model",
            "EVIDENCE_AUTH_TOKEN": "Bearer evidence-token",
        },
        adapter=lambda **_kwargs: None,
        allow_test_doubles=True,
    )

    assert runtime.auth_mode == "api"
    assert runtime.allow_provider_migration is False
    provider = runtime._resolve_provider(runtime.review_provider, "review-model")
    assert provider.settings.provider_id == "arche_review"
    assert provider.secret_environment == {
        "ARCHE_REVIEW_CODEX_API_KEY": "review-secret"
    }


def test_local_login_uses_builtin_openai_without_api_provider_override(tmp_path: Path) -> None:
    """local_login 使用内置 openai，但模型和推理强度可以按配置调整。"""

    source = tmp_path / "auth.json"
    _write_private_source(source)
    provider = CodexProviderSettings(
        provider_id="custom-api",
        env_key="ARCHE_CODEX_API_KEY",
        base_url_env="ARCHE_CODEX_BASE_URL",
        model_env="ARCHE_CODEX_MODEL_NAME",
        reasoning_effort_env="ARCHE_CODEX_REASONING_EFFORT",
        reasoning_effort_default="xhigh",
    )
    runtime = InvestigationCodexRuntime(
        session_root=tmp_path / "sessions",
        trace_root=tmp_path / "traces",
        minichem=InvestigationMcpSettings("http://127.0.0.1:9012/mcp", "MINICHEM_TOKEN"),
        investigation_provider=provider,
        environment={
            "ARCHE_CODEX_AUTH_MODE": "local_login",
            "ARCHE_CODEX_AUTH_SOURCE": str(source),
            "ARCHE_CODEX_LOCAL_LOGIN_MODEL": "local-login-model",
            "ARCHE_CODEX_LOCAL_LOGIN_REASONING_EFFORT": "high",
            "MINICHEM_TOKEN": "capability-value",
            # 即使父环境中存在 API 配置，local_login 也不能把它带进 runtime。
            "ARCHE_CODEX_API_KEY": "must-not-be-forwarded",
            "ARCHE_CODEX_BASE_URL": "https://provider.invalid/v1",
            "ARCHE_CODEX_MODEL_NAME": "wrong-api-model",
        },
        adapter=lambda **_: None,
        allow_test_doubles=True,
        role="investigation",
    )
    resolved = runtime._resolve_provider(runtime.investigation_provider, "local-login-model")
    config = runtime._provider_thread_config(resolved, runtime._investigation_mcp_config())
    assert resolved.settings.provider_id == "openai"
    assert resolved.model == "local-login-model"
    assert resolved.reasoning_effort == "high"
    assert resolved.secret_environment == {}
    assert config["model_provider"] == "openai"
    assert "model_providers" not in config
    assert runtime._environment.get("ARCHE_CODEX_API_KEY") is None
    assert runtime._environment.get("ARCHE_CODEX_BASE_URL") is None


def test_thread_policy_digest_freezes_auth_mode_provider_model_and_permission_profile(
    tmp_path: Path,
) -> None:
    """API 与 local_login 不得复用同一含旧 provider/权限的 Codex thread。"""

    common = {
        "thread_config": {"model_provider": "openai", "features": {"shell_tool": True}},
        "workspace": tmp_path / "workspace",
        "provider_id": "openai",
        "env_key": "ARCHE_CODEX_LOCAL_LOGIN",
        "model": "gpt-5.6-sol",
        "role": "investigation",
        "stage": "initial_investigation",
        "profile": "investigation",
        "permission_profile": CODEX_INVESTIGATION_PERMISSION_PROFILE,
    }
    api_digest = calculate_codex_thread_policy_digest(**common, auth_mode="api")
    local_digest = calculate_codex_thread_policy_digest(**common, auth_mode="local_login")
    assert api_digest != local_digest


def test_api_to_local_login_migration_requires_opt_in_and_never_resumes_old_thread(
    tmp_path: Path,
) -> None:
    """API 到本机登录只允许显式迁移，旧的加密失败 thread 永不 resume。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    source = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    workspace = source.workspace
    api_config = {
        "model_provider": "arche_investigation",
        "model_providers": {
            "arche_investigation": {
                "base_url": "https://relay.invalid/v1",
                "env_key": "ARCHE_CODEX_API_KEY",
            }
        },
    }
    local_config = {
        "model_provider": "openai",
        "reasoning_effort": "xhigh",
    }
    api_digest = calculate_codex_thread_policy_digest(
        thread_config=api_config,
        workspace=workspace,
        provider_id="arche_investigation",
        env_key="ARCHE_CODEX_API_KEY",
        model="gpt-5.6-sol",
        role="investigation",
        stage="initial_investigation",
        profile="investigation",
        auth_mode="api",
        permission_profile=CODEX_INVESTIGATION_PERMISSION_PROFILE,
    )
    local_digest = calculate_codex_thread_policy_digest(
        thread_config=local_config,
        workspace=workspace,
        provider_id="openai",
        env_key="ARCHE_CODEX_LOCAL_LOGIN",
        model="gpt-5.6-sol",
        role="investigation",
        stage="initial_investigation",
        profile="investigation",
        auth_mode="local_login",
        permission_profile=CODEX_INVESTIGATION_PERMISSION_PROFILE,
    )
    assert api_digest != local_digest
    source.create("old-invalid-encrypted-thread", thread_policy_digest=api_digest)

    with pytest.raises(HarnessError, match="explicit provider migration"):
        select_explicit_provider_migration_store(
            source,
            session_root=tmp_path / "sessions",
            target_policy_digest=local_digest,
            auth_mode="local_login",
            provider_id="openai",
            model="gpt-5.6-sol",
            allow_migration=False,
        )

    target = select_explicit_provider_migration_store(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest=local_digest,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
    )
    assert target is not source
    assert target.load() is None
    assert source.load()["thread_id"] == "old-invalid-encrypted-thread"
    marker = json.loads(
        (source.manifest_path.parent / ".arche_codex_provider_migration.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["auth_mode"] == "local_login"
    assert marker["provider_id"] == "openai"


def test_api_runtime_can_explicitly_migrate_to_local_login_without_old_thread_resume(
    tmp_path: Path,
) -> None:
    """真实 runtime 的 API→local_login 切换创建新 thread，不复用旧 manifest thread。"""

    source_auth = tmp_path / "auth.json"
    _write_private_source(source_auth)
    session_root = tmp_path / "sessions"
    trace_root = tmp_path / "traces"
    api_events: list[str] = []
    local_events: list[str] = []

    def successful_response(thread_id: str, turn_id: str) -> CodexSdkRunResult:
        """构造只含公开科学回顾的 fake SDK 响应。"""

        return CodexSdkRunResult(
            response=HarnessResponse(
                content=json.dumps(
                    {
                        "self_judgment": "supported",
                        "narrative": "fake provider migration response",
                        "job_refs": ["job-test"],
                        "operation_refs": ["operation-test"],
                        "artifact_refs": ["artifact-test"],
                        "observations": ["observation-test"],
                        "limitations": [],
                    }
                ),
                model="gpt-5.6-sol",
                metadata={"turn_id": turn_id},
            ),
            thread_id=thread_id,
            turn_id=turn_id,
        )

    def api_adapter(**kwargs):
        """在旧 API policy 下创建一个可验证的 source thread。"""

        api_events.append(str(kwargs["provider_id"]))
        kwargs["on_thread_ready"]("api-thread")
        _write_test_completed_receipt(
            kwargs,
            thread_id="api-thread",
            turn_id="api-turn",
        )
        return successful_response("api-thread", "api-turn")

    api_runtime = InvestigationCodexRuntime(
        session_root=session_root,
        trace_root=trace_root,
        minichem=InvestigationMcpSettings("http://127.0.0.1:9012/mcp", "MINICHEM_TOKEN"),
        environment={
            "ARCHE_CODEX_AUTH_MODE": "api",
            "ARCHE_CODEX_API_KEY": "api-value",
            "ARCHE_CODEX_BASE_URL": "https://relay.invalid/v1",
            "ARCHE_CODEX_MODEL_NAME": "gpt-5.6-sol",
            "MINICHEM_TOKEN": "capability-value",
        },
        adapter=api_adapter,
        allow_test_doubles=True,
        role="investigation",
    )
    request = build_initial_investigation_request(
        "task-1",
        "batch-1",
        "combination-1",
        "判断一个测试假设是否获得计算支持",
        {"name": "test-hypothesis"},
        model="gpt-5.6-sol",
    )
    api_result = api_runtime.run_investigation_turn(request, scope_id="scope-1")
    assert api_result.thread_id == "api-thread"
    assert api_events == ["arche_investigation"]

    def local_adapter(**kwargs):
        """在迁移目标中只启动新 thread，不恢复 API thread。"""

        local_events.append(str(kwargs["provider_id"]))
        assert kwargs["session_id"] is None
        assert kwargs["thread_config"]["model_provider"] == "openai"
        assert "model_providers" not in kwargs["thread_config"]
        kwargs["on_thread_ready"]("local-thread")
        _write_test_completed_receipt(
            kwargs,
            thread_id="local-thread",
            turn_id="local-turn",
        )
        return successful_response("local-thread", "local-turn")

    local_runtime = InvestigationCodexRuntime(
        session_root=session_root,
        trace_root=trace_root,
        minichem=InvestigationMcpSettings("http://127.0.0.1:9012/mcp", "MINICHEM_TOKEN"),
        environment={
            "ARCHE_CODEX_AUTH_MODE": "local_login",
            "ARCHE_CODEX_AUTH_SOURCE": str(source_auth),
            "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION": "1",
            "MINICHEM_TOKEN": "capability-value",
        },
        adapter=local_adapter,
        allow_test_doubles=True,
        role="investigation",
    )
    local_result = local_runtime.run_investigation_turn(request, scope_id="scope-1")
    assert local_result.thread_id == "local-thread"
    assert local_events == ["openai"]
    source_store = CodexSessionStore.from_root(
        session_root,
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    assert source_store.load()["thread_id"] == "api-thread"
    target_manifests = list(
        (source_store.manifest_path.parent / "provider_migrations").glob(
            "*/state/.arche_codex_session.json"
        )
    )
    assert len(target_manifests) == 1
    assert json.loads(target_manifests[0].read_text(encoding="utf-8"))["thread_id"] == "local-thread"


def test_explicit_provider_migration_keeps_active_job_and_never_reuses_old_thread(
    tmp_path: Path,
) -> None:
    """迁移为新 thread，但旧 manifest/job identity 保持不变且 tracker 只 get/collect。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    source = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    source.create("old-thread", thread_policy_digest="a" * 64)
    identity = InvestigationJobIdentity(
        job_id="job_8912052d",
        operation_id="operation-1",
        submission_key="submission-1",
        action_id="action-1",
        backend_id="backend-1",
    )
    active_job = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="old-thread",
        turn_id="old-turn",
        status="waiting_investigation_job",
        identity=identity,
        request_digest="b" * 64,
        scope_id="scope-1",
    )
    target = select_explicit_provider_migration_store(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="c" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
        request_thread_id="old-thread",
        active_job=active_job,
    )
    assert target is not source
    assert source.load()["thread_id"] == "old-thread"
    assert target.load() is None
    marker = json.loads(
        (source.manifest_path.parent / ".arche_codex_provider_migration.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["auth_mode"] == "local_login"
    assert "auth_source" not in marker
    assert "source_digest" not in marker
    assert active_job.identity.job_id == "job_8912052d"

    repeated = select_explicit_provider_migration_store(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="c" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
        request_thread_id="old-thread",
        active_job=active_job,
    )
    assert repeated.codex_home == target.codex_home


def test_provider_migration_resume_follows_verified_chain_and_keeps_initial_job_thread(
    tmp_path: Path,
) -> None:
    """策略再次变化时，尚未注册的链尾允许 WM 用直接前驱完成恢复。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    source = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    source.create("source-thread", thread_policy_digest="a" * 64)
    target_one = select_explicit_provider_migration_store(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="b" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
    )
    target_one.create("target-one-thread", thread_policy_digest="b" * 64)
    active_job = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="source-thread",
        turn_id="source-turn",
        status="waiting_investigation_job",
        identity=InvestigationJobIdentity(
            job_id="job-active",
            operation_id="operation-active",
            submission_key="submission-active",
            action_id="action-active",
            backend_id="backend-1",
        ),
        request_digest="d" * 64,
        scope_id="scope-1",
    )

    target_two = select_explicit_provider_migration_store(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="c" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
        request_thread_id="target-one-thread",
        active_job=active_job,
    )
    assert target_two.codex_home == (
        target_one.manifest_path.parent
        / "provider_migrations"
        / ("c" * 32)
        / "codex_home"
    )
    target_two.create("target-two-thread", thread_policy_digest="c" * 64)

    resumed = select_explicit_provider_migration_store(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="c" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
        request_thread_id="target-two-thread",
        active_job=active_job,
    )
    assert resumed.codex_home == target_two.codex_home
    resumed_selection = resolve_explicit_provider_migration(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="c" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
        request_thread_id="target-one-thread",
        active_job=active_job,
    )
    assert resumed_selection.replace_unstarted_thread_id == "target-two-thread"
    pending_registration = select_explicit_provider_migration_store(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="c" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
        request_thread_id="target-one-thread",
        active_job=active_job,
    )
    assert pending_registration.codex_home == target_two.codex_home
    target_two.record_completed_turn(stage="investigation_job_wake", status="completed")
    with pytest.raises(HarnessError, match="verified migration chain"):
        select_explicit_provider_migration_store(
            source,
            session_root=tmp_path / "sessions",
            target_policy_digest="c" * 64,
            auth_mode="local_login",
            provider_id="openai",
            model="gpt-5.6-sol",
            allow_migration=True,
            request_thread_id="target-one-thread",
            active_job=active_job,
        )


def test_provider_migration_marker_remembers_target_thread_before_manifest(
    tmp_path: Path,
) -> None:
    """target thread 回调先落 marker 时，恢复选择同一 thread 而非再次创建。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    source = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    source.create("source-thread", thread_policy_digest="a" * 64)
    selection = resolve_explicit_provider_migration(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="b" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
    )
    persist_provider_migration_target_thread(selection, "target-thread")
    assert selection.store.load() is None

    resumed = resolve_explicit_provider_migration(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="b" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
    )
    assert resumed.store.codex_home == selection.store.codex_home
    assert resumed.target_thread_id == "target-thread"
    assert resumed.replace_unstarted_thread_id is None
    assert resumed.store.load() is None


def test_provider_migration_result_store_resolver_is_read_only(
    tmp_path: Path,
) -> None:
    """Result 恢复沿 target 链读取既有 store，缺目录时不偷偷创建新状态。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    source = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    source.create("source-thread", thread_policy_digest="a" * 64)
    selection = resolve_explicit_provider_migration(
        source,
        session_root=tmp_path / "sessions",
        target_policy_digest="b" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
    )
    selection.store.create("target-thread", thread_policy_digest="b" * 64)
    resolved = resolve_provider_migration_target_store_read_only(
        source,
        session_root=tmp_path / "sessions",
        target_thread_id="target-thread",
        target_policy_digest="b" * 64,
        model="gpt-5.6-sol",
    )
    assert resolved.codex_home == selection.store.codex_home

    second_source = CodexSessionStore.from_root(
        tmp_path / "second-sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    second_source.create("second-source", thread_policy_digest="a" * 64)
    with pytest.raises(HarnessError, match="not present in the verified chain"):
        resolve_provider_migration_target_store_read_only(
            second_source,
            session_root=tmp_path / "second-sessions",
            target_thread_id="missing-target",
            target_policy_digest="b" * 64,
            model="gpt-5.6-sol",
        )
    assert not (
        second_source.manifest_path.parent / "provider_migrations"
    ).exists()


def test_unstarted_codex_thread_replacement_is_atomic_and_rejects_started_turn(
    tmp_path: Path,
) -> None:
    """只允许替换 created/0 manifest，完成 turn 后必须保持原身份并拒绝替换。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    store.create("stale-thread", thread_policy_digest="a" * 64)
    replaced = store.replace_unstarted_thread(
        "stale-thread",
        "fresh-thread",
        stage="investigation_job_wake",
        thread_policy_digest="a" * 64,
    )
    assert replaced["thread_id"] == "fresh-thread"
    assert replaced["turn_count"] == 0
    assert store.replace_unstarted_thread(
        "stale-thread",
        "fresh-thread",
        stage="investigation_job_wake",
        thread_policy_digest="a" * 64,
    )["thread_id"] == "fresh-thread"
    store.record_completed_turn(stage="investigation_job_wake")
    with pytest.raises(HarnessError, match="unstarted session"):
        store.replace_unstarted_thread(
            "fresh-thread",
            "another-thread",
            stage="investigation_job_wake",
            thread_policy_digest="a" * 64,
        )


def test_runtime_wake_uses_current_target_after_second_provider_migration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """真实 runtime 接缝允许 target wake，并把 sandbox 重绑到下一层 CODEX_HOME。"""

    source_auth = tmp_path / "auth.json"
    _write_private_source(source_auth)
    session_root = tmp_path / "sessions"
    source = CodexSessionStore.from_root(
        session_root,
        owner=CodexSessionOwner.investigation("task-1", "batch-1", "combination-1"),
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    source.create("source-thread", thread_policy_digest="a" * 64)
    target_one = select_explicit_provider_migration_store(
        source,
        session_root=session_root,
        target_policy_digest="b" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
    )
    target_one.create("target-one-thread", thread_policy_digest="b" * 64)
    identity = InvestigationJobIdentity(
        job_id="job-active",
        operation_id="operation-active",
        submission_key="submission-active",
        action_id="action-active",
        backend_id="backend-1",
    )
    active_job = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="source-thread",
        turn_id="source-turn",
        status="waiting_investigation_job",
        identity=identity,
        request_digest="d" * 64,
        scope_id="scope-1",
    )
    stale_target_two = select_explicit_provider_migration_store(
        source,
        session_root=session_root,
        target_policy_digest="c" * 64,
        auth_mode="local_login",
        provider_id="openai",
        model="gpt-5.6-sol",
        allow_migration=True,
        request_thread_id="target-one-thread",
        active_job=active_job,
    )
    stale_target_two.create("stale-target-two-thread", thread_policy_digest="c" * 64)
    monkeypatch.setattr(
        codex_runtime_module,
        "calculate_codex_thread_policy_digest",
        lambda **_kwargs: "c" * 64,
    )

    def adapter(**kwargs):
        """在新 policy 的 target thread 中只读取并收集原作业。"""

        assert kwargs["session_id"] is None
        kwargs["on_thread_ready"]("target-two-thread")
        kwargs["on_tool_completed"](
            {
                "type": "mcpToolCall",
                "tool": "get_transition_state_path_job",
                "status": "completed",
                "arguments": {"job_id": identity.job_id},
                "result": {
                    "structuredContent": {
                        "job_id": identity.job_id,
                        "job_status": "completed",
                    }
                },
            }
        )
        kwargs["on_tool_completed"](
            {
                "type": "mcpToolCall",
                "tool": "collect_transition_state_path_job",
                "status": "completed",
                "arguments": {"job_id": identity.job_id},
                "result": {
                    "structuredContent": {
                        "job_id": identity.job_id,
                        "ready": True,
                    }
                },
            }
        )
        _write_test_completed_receipt(
            kwargs,
            thread_id="target-two-thread",
            turn_id="target-two-turn",
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=json.dumps(
                    {
                        "self_judgment": "supported",
                        "narrative": "从当前 target 读取并收集了原作业。",
                        "job_refs": [identity.job_id],
                        "operation_refs": [identity.operation_id],
                        "artifact_refs": ["artifact-1"],
                        "observations": ["observation-1"],
                        "limitations": [],
                    }
                ),
                model="gpt-5.6-sol",
                metadata={"turn_id": "target-two-turn"},
            ),
            thread_id="target-two-thread",
            turn_id="target-two-turn",
        )

    runtime = InvestigationCodexRuntime(
        session_root=session_root,
        trace_root=tmp_path / "traces",
        minichem=InvestigationMcpSettings("http://127.0.0.1:9012/mcp", "MINICHEM_TOKEN"),
        environment={
            "ARCHE_CODEX_AUTH_MODE": "local_login",
            "ARCHE_CODEX_AUTH_SOURCE": str(source_auth),
            "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION": "1",
            "MINICHEM_TOKEN": "capability-value",
        },
        adapter=adapter,
        allow_test_doubles=True,
        role="investigation",
        sandbox_spec=CodexProcessSandboxSpec(
            workspace=source.workspace,
            codex_home=source.codex_home,
            codex_bin=(
                Path(__file__).resolve().parents[2]
                / "third_party"
                / "codex_cli"
                / "0.149.0"
                / "linux-x64"
                / "codex"
            ),
            runtime_root=tmp_path / "codex-runtime",
        ),
    )
    request = build_investigation_job_wake_request(
        "task-1",
        "batch-1",
        "combination-1",
        "target-one-thread",
        job_id=identity.job_id,
        operation_id=identity.operation_id,
        submission_key=identity.submission_key,
        action_id=identity.action_id,
        backend_id=identity.backend_id,
        terminal_status="succeeded",
        available_tools=(
            "get_transition_state_path_job",
            "collect_transition_state_path_job",
        ),
        model="gpt-5.6-sol",
    )
    result = runtime.run_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=active_job,
    )
    assert result.thread_id == "target-two-thread"
    target_manifests = list(
        (target_one.manifest_path.parent / "provider_migrations").glob(
            "*/state/.arche_codex_session.json"
        )
    )
    assert len(target_manifests) == 1
    assert json.loads(target_manifests[0].read_text(encoding="utf-8"))["thread_id"] == (
        "target-two-thread"
    )


def test_active_job_tracker_ignores_other_job_in_same_scope_without_clearing_waiting(
    tmp_path: Path,
) -> None:
    """同 scope 的历史作业只读不改变 waiting sidecar 指定的 active 作业。"""

    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    active_identity = InvestigationJobIdentity(
        job_id="job-active",
        operation_id="operation-active",
        submission_key="submission-active",
        action_id="action-active",
        backend_id="backend-1",
    )
    historical_identity = InvestigationJobIdentity(
        job_id="job-history",
        operation_id="operation-history",
        submission_key="submission-history",
        action_id="action-history",
        backend_id="backend-1",
    )
    active_job = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="thread-1",
        turn_id="turn-1",
        status="waiting_investigation_job",
        identity=active_identity,
        request_digest="a" * 64,
        scope_id="scope-1",
    )
    callbacks = codex_runtime_module._CodexTurnCallbacks(
        store=store,
        manifest=None,
        session_id=None,
        stage="investigation_job_wake",
        request_digest="a" * 64,
        thread_policy_digest="b" * 64,
        active_job_receipt=codex_runtime_module._build_active_job_receipt(active_job),
    )
    wrong_get = {
        "type": "mcpToolCall",
        "tool": "get_transition_state_path_job",
        "status": "completed",
        "arguments": {"job_id": historical_identity.job_id},
        "result": {
            "structuredContent": {
                "job_id": historical_identity.job_id,
                "job_status": "completed",
            }
        },
    }
    callbacks.capture_tool_completed(wrong_get)
    assert callbacks.active_job_receipt.identity.job_id == active_identity.job_id
    assert callbacks.active_job_collected is False
    assert callbacks.completed_job_receipts == []
    assert active_job.status == "waiting_investigation_job"
    assert historical_identity.job_id != active_identity.job_id

    callbacks.capture_tool_completed(
        {
            **wrong_get,
            "arguments": {"job_id": active_identity.job_id},
            "result": {
                "structuredContent": {
                    "job_id": active_identity.job_id,
                    "job_status": "completed",
                }
            },
        }
    )
    callbacks.capture_tool_completed(
        {
            "type": "mcpToolCall",
            "tool": "collect_transition_state_path_job",
            "status": "completed",
            "arguments": {"job_id": active_identity.job_id},
            "result": {
                "structuredContent": {
                    "job_id": active_identity.job_id,
                    "ready": True,
                }
            },
        }
    )
    assert callbacks.active_job_collected is True
    assert [receipt.identity.job_id for receipt in callbacks.completed_job_receipts] == [
        active_identity.job_id
    ]


def test_runtime_migration_starts_new_thread_and_reuses_only_get_collect_job_identity(
    tmp_path: Path,
) -> None:
    """provider migration 的真实 runtime 路径不 submit，仍可 get/collect 原作业。"""

    source_auth = tmp_path / "auth.json"
    _write_private_source(source_auth)
    owner = CodexSessionOwner.investigation("task-1", "batch-1", "combination-1")
    old_store = CodexSessionStore.from_root(
        tmp_path / "sessions",
        owner=owner,
        scope_id="scope-1",
        model="gpt-5.6-sol",
    )
    old_store.create("old-thread", thread_policy_digest="a" * 64)
    identity = InvestigationJobIdentity(
        job_id="job-8912052d",
        operation_id="operation-1",
        submission_key="submission-1",
        action_id="action-1",
        backend_id="backend-1",
    )
    historical_identity = InvestigationJobIdentity(
        job_id="job-history-same-scope",
        operation_id="operation-history",
        submission_key="submission-history",
        action_id="action-history",
        backend_id="backend-1",
    )
    active_job = InvestigationJobState(
        task_id="task-1",
        batch_id="batch-1",
        combination_id="combination-1",
        codex_thread_id="old-thread",
        turn_id="old-turn",
        status="waiting_investigation_job",
        identity=identity,
        request_digest="b" * 64,
        scope_id="scope-1",
    )
    events: list[str] = []

    def migration_adapter(**kwargs):
        """新 thread 中只模拟原 job 的 get/collect 工具事件。"""

        assert kwargs["session_id"] is None
        kwargs["on_thread_ready"]("new-thread")
        events.append("thread-ready")
        kwargs["on_tool_completed"](
            {
                "type": "mcpToolCall",
                "tool": "get_transition_state_path_job",
                "status": "completed",
                "arguments": {"job_id": "job-8912052d"},
                "result": {"structuredContent": {"job_id": "job-8912052d", "job_status": "completed"}},
            }
        )
        events.append("get")
        kwargs["on_tool_completed"](
            {
                "type": "mcpToolCall",
                "tool": "collect_transition_state_path_job",
                "status": "completed",
                "arguments": {"job_id": "job-8912052d"},
                "result": {"structuredContent": {"job_id": "job-8912052d", "ready": True}},
            }
        )
        events.append("collect")
        _write_test_completed_receipt(
            kwargs,
            thread_id="new-thread",
            turn_id="new-turn",
        )
        return CodexSdkRunResult(
            response=HarnessResponse(
                content=json.dumps(
                    {
                        "self_judgment": "supported",
                        "narrative": "迁移后的新 thread 读取并收集了原作业。",
                        "job_refs": ["job-8912052d"],
                        "operation_refs": ["operation-1"],
                        "artifact_refs": ["artifact-1"],
                        "observations": ["observation-1"],
                        "limitations": [],
                    }
                ),
                model="gpt-5.6-sol",
                metadata={"turn_id": "new-turn"},
            ),
            thread_id="new-thread",
            turn_id="new-turn",
        )

    runtime = InvestigationCodexRuntime(
        session_root=tmp_path / "sessions",
        trace_root=tmp_path / "traces",
        minichem=InvestigationMcpSettings("http://127.0.0.1:9012/mcp", "MINICHEM_TOKEN"),
        environment={
            "ARCHE_CODEX_AUTH_MODE": "local_login",
            "ARCHE_CODEX_AUTH_SOURCE": str(source_auth),
            "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION": "1",
            "MINICHEM_TOKEN": "capability-value",
        },
        adapter=migration_adapter,
        allow_test_doubles=True,
        role="investigation",
        sandbox_spec=CodexProcessSandboxSpec(
            workspace=old_store.workspace,
            codex_home=old_store.codex_home,
            codex_bin=(
                Path(__file__).resolve().parents[2]
                / "third_party"
                / "codex_cli"
                / "0.149.0"
                / "linux-x64"
                / "codex"
            ),
            runtime_root=tmp_path / "codex-runtime",
        ),
    )
    request = build_investigation_job_wake_request(
        "task-1",
        "batch-1",
        "combination-1",
        "old-thread",
        job_id=identity.job_id,
        operation_id=identity.operation_id,
        submission_key=identity.submission_key,
        action_id=identity.action_id,
        backend_id=identity.backend_id,
        terminal_status="succeeded",
        available_tools=(
            "get_transition_state_path_job",
            "collect_transition_state_path_job",
        ),
        model="gpt-5.6-sol",
    )
    # 同 scope 的历史已收集作业可以留在 Working Memory 科学证据中，但迁移
    # 只把 waiting sidecar 指定的 active identity 放进 wake request。
    assert historical_identity.job_id not in json.dumps(request, ensure_ascii=False)
    result = runtime.run_investigation_turn(
        request,
        scope_id="scope-1",
        active_job=active_job,
    )
    assert result.thread_id == "new-thread"
    assert events == ["thread-ready", "get", "collect"]
    assert len(result.completed_jobs) == 1
    assert result.completed_jobs[0]["identity"]["job_id"] == "job-8912052d"
    assert result.completed_jobs[0]["identity"]["operation_id"] == "operation-1"
    assert active_job.identity.operation_id == "operation-1"
    assert active_job.identity.submission_key == "submission-1"
    assert old_store.load()["thread_id"] == "old-thread"
    target_manifests = list(
        (old_store.manifest_path.parent / "provider_migrations").glob("*/state/.arche_codex_session.json")
    )
    assert len(target_manifests) == 1
    assert json.loads(target_manifests[0].read_text(encoding="utf-8"))["thread_id"] == "new-thread"
    assert active_job.identity.job_id == "job-8912052d"
