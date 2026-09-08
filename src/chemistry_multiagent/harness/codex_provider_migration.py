"""API 与 local_login 之间的显式 Codex thread migration seam。

认证模式或 provider policy 变化时，旧 thread 不能 resume。这里仅为当前
``CodexSessionStore`` 建立一个新的 CODEX_HOME/state 子树，并在旧 state 根写入
不含凭据的幂等审计 marker；operation、checkpoint、Working Memory 和等待 job
事实都不在本模块内读写，因此不会因为换 provider 重复 submit。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .base import HarnessError
from .session_store import (
    CodexSessionStore,
    SchemaV1CodexSessionOwnerReader,
    _atomic_write_json_file,
)


CODEX_PROVIDER_MIGRATION_MARKER_NAME = ".arche_codex_provider_migration.json"
_LEGACY_MIGRATION_SCHEMA_VERSION = 1
_LEGACY_OWNER_MIGRATION_SCHEMA_VERSION = 2
_MIGRATION_SCHEMA_VERSION = 3
_DURABLE_MIGRATION_SCHEMA_VERSIONS = frozenset(
    {_LEGACY_OWNER_MIGRATION_SCHEMA_VERSION, _MIGRATION_SCHEMA_VERSION}
)
_MIGRATION_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "source_thread_id",
        "source_policy_digest",
        "target_policy_digest",
        "target_root",
        "auth_mode",
        "provider_id",
        "model",
    }
)
_MIGRATION_FIELDS_V2 = frozenset(
    {
        "schema_version",
        "source_thread_id",
        "source_policy_digest",
        "target_policy_digest",
        "target_root",
        "auth_mode",
        "provider_id",
        "model",
        "source_owner",
        "source_scope_id",
        "source_workspace",
        "source_codex_home",
        "source_state_root",
        "source_turn_count",
        "target_workspace",
        "target_codex_home",
        "target_state_root",
        "target_thread_id",
        "request_thread_id",
        "request_digest",
        "turn_id",
        "turn_count_before",
    }
)
_MIGRATION_THREAD_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_MIGRATION_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ProviderMigrationSelection:
    """保存已验证的 migration 目标及 wake 请求可使用的 thread 身份。

    ``source_thread_id`` 是最初提交作业时的 thread；即使 migration 已经经过多层，
    active job 仍必须绑定它。``request_thread_ids`` 通常只包含最初 source 和当前
    链尾；若链尾刚创建且尚无完成 turn，也可包含它的直接前驱，使 Working Memory
    在 thread 注册中断后完成幂等迁移。SDK 实际使用的仍是当前链尾 store。
    """

    store: CodexSessionStore
    source_thread_id: str | None
    request_thread_ids: tuple[str, ...]
    replace_unstarted_thread_id: str | None = None
    # 仅当本次选择需要继续登记一个尚未写入 target manifest 的迁移时保存直接
    # source store；已有完整 target manifest 的历史链不需要再次写登记。
    migration_source_store: CodexSessionStore | None = None
    # target thread 已由 provider 返回但 target manifest 尚未持久化时，恢复必须
    # 使用这个身份执行 resume，不能再次创建 thread。
    target_thread_id: str | None = None


def _validate_migration_digest(value: Any, field_name: str, *, allow_none: bool = False) -> str | None:
    """校验 migration marker 中的摘要；不接受任意文本或秘密正文。"""

    if value is None and allow_none:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state")
    try:
        int(value, 16)
    except ValueError as exc:
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state") from exc
    return value


def _validate_migration_identifier(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> str | None:
    """校验可以写入迁移记录的稳定身份；不接受路径、控制字符或任意长文本。"""

    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state")
    normalized = value.strip()
    if _MIGRATION_THREAD_ID.fullmatch(normalized) is None:
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state")
    return normalized


def _validate_migration_optional_digest(value: Any, field_name: str) -> str | None:
    """校验可选的请求摘要；空值表示旧任务没有该项恢复信息。"""

    if value is None:
        return None
    if not isinstance(value, str) or _MIGRATION_DIGEST.fullmatch(value) is None:
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state")
    return value


def _validate_migration_count(value: Any, field_name: str, *, allow_none: bool = False) -> int | None:
    """校验迁移记录中的非负 turn 计数。"""

    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state")
    return value


def _path_is_within(path: Path, parent: Path) -> bool:
    """判断已经解析的路径是否位于指定目录内。"""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_migration_path(value: Any, field_name: str) -> Path:
    """校验迁移记录中的绝对路径并返回规范路径。"""

    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise HarnessError(f"provider migration {field_name} is malformed", kind="session_state")
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise HarnessError(f"provider migration {field_name} cannot be resolved", kind="session_state") from exc


def _validate_migration_owner(value: Any, expected: Mapping[str, str] | None = None) -> dict[str, str]:
    """校验 owner 的固定字段，避免把别的调查组合的 session 当成目标。"""

    if not isinstance(value, Mapping):
        raise HarnessError("provider migration source owner is malformed", kind="session_state")
    normalized = {str(key): item for key, item in value.items()}
    if "task_id" in normalized:
        normalized = SchemaV1CodexSessionOwnerReader.read(normalized).identity
    canonical_shapes = (
        {"kind", "research_task_id"},
        {
            "kind",
            "research_task_id",
            "investigation_context_id",
            "hypothesis_execution_input_record_id",
        },
    )
    if set(normalized) not in canonical_shapes:
        raise HarnessError("provider migration source owner is malformed", kind="session_state")
    for key, item in normalized.items():
        _validate_migration_identifier(item, f"source_owner.{key}")
    if expected is not None and normalized != dict(expected):
        raise HarnessError("provider migration source owner does not match session", kind="session_state")
    return normalized


def _read_migration_marker(path: Path) -> dict[str, Any] | None:
    """读取并严格校验 migration marker；损坏记录阻断恢复。"""

    if path.is_symlink():
        raise HarnessError("provider migration marker must not be a symlink", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("provider migration marker is not a regular file", kind="session_state")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessError("provider migration marker cannot be read", kind="session_state") from exc
    if not isinstance(value, Mapping):
        raise HarnessError("provider migration marker fields are invalid", kind="session_state")
    schema_version = value.get("schema_version")
    if schema_version == _LEGACY_MIGRATION_SCHEMA_VERSION:
        fields = _MIGRATION_FIELDS_V1
    elif schema_version in _DURABLE_MIGRATION_SCHEMA_VERSIONS:
        fields = _MIGRATION_FIELDS_V2
    else:
        raise HarnessError("provider migration marker schema is unsupported", kind="session_state")
    if set(value) != fields:
        raise HarnessError("provider migration marker fields are invalid", kind="session_state")
    _validate_migration_digest(value.get("source_policy_digest"), "source_policy_digest", allow_none=True)
    _validate_migration_digest(value.get("target_policy_digest"), "target_policy_digest")
    for field_name in ("source_thread_id", "auth_mode", "provider_id", "model"):
        field_value = value.get(field_name)
        _validate_migration_identifier(field_value, field_name)
    target_root_value = value.get("target_root")
    if not isinstance(target_root_value, str) or not target_root_value.strip():
        raise HarnessError("provider migration target_root is malformed", kind="session_state")
    if schema_version == _LEGACY_MIGRATION_SCHEMA_VERSION:
        return dict(value)

    # v2 将 source/target store 的身份、当前请求和 target thread 登记放在同一
    # 原子记录中。所有路径都会在使用时再次和当前 store/session_root 比较，不能
    # 仅凭 marker 内部的自洽文本跨组合恢复。
    _validate_migration_owner(value.get("source_owner"))
    _validate_migration_identifier(value.get("source_scope_id"), "source_scope_id")
    for field_name in (
        "source_workspace",
        "source_codex_home",
        "source_state_root",
        "target_workspace",
        "target_codex_home",
        "target_state_root",
    ):
        _resolve_migration_path(value.get(field_name), field_name)
    _validate_migration_identifier(value.get("target_thread_id"), "target_thread_id", allow_none=True)
    _validate_migration_identifier(value.get("request_thread_id"), "request_thread_id", allow_none=True)
    _validate_migration_optional_digest(value.get("request_digest"), "request_digest")
    _validate_migration_identifier(value.get("turn_id"), "turn_id", allow_none=True)
    _validate_migration_count(value.get("source_turn_count"), "source_turn_count")
    _validate_migration_count(value.get("turn_count_before"), "turn_count_before", allow_none=True)
    return dict(value)


def _build_migration_marker(
    *,
    source_store: CodexSessionStore,
    source_manifest: Mapping[str, Any],
    target_policy_digest: str,
    target_root: Path,
    auth_mode: str,
    provider_id: str,
    model: str,
    request_thread_id: str | None = None,
) -> dict[str, Any]:
    """构造绑定 source/target store 和当前请求身份的审计记录，不记录凭据。"""

    source_thread_id = source_manifest.get("thread_id")
    source_policy_digest = source_manifest.get("thread_policy_digest")
    if not isinstance(source_thread_id, str) or not source_thread_id.strip():
        raise HarnessError("provider migration source thread is missing", kind="session_state")
    _validate_migration_digest(source_policy_digest, "source_policy_digest", allow_none=True)
    _validate_migration_digest(target_policy_digest, "target_policy_digest")
    _validate_migration_identifier(auth_mode, "auth_mode")
    _validate_migration_identifier(provider_id, "provider_id")
    _validate_migration_identifier(model, "model")
    source_turn_count = _validate_migration_count(
        source_manifest.get("turn_count"),
        "source_turn_count",
    )
    if source_turn_count is None:  # pragma: no cover - validator rejects None
        raise HarnessError("provider migration source turn count is missing", kind="session_state")
    normalized_request_thread = _validate_migration_identifier(
        request_thread_id,
        "request_thread_id",
        allow_none=True,
    )
    # resolve_explicit_provider_migration 只知道当前 policy 和 request thread，
    # 还不知道即将发送的 target prompt/turn；这些值由 runtime 在 target store
    # 确定后通过 persist_provider_migration_turn_context 原子补齐。不能把旧
    # waiting state 的 submission digest 当成新 provider turn 的 request digest。
    target_codex_home = target_root / "codex_home"
    target_state_root = target_root / "state"
    return {
        "schema_version": _MIGRATION_SCHEMA_VERSION,
        "source_thread_id": source_thread_id,
        "source_policy_digest": source_policy_digest,
        "target_policy_digest": target_policy_digest,
        "target_root": str(target_root.resolve()),
        "auth_mode": auth_mode,
        "provider_id": provider_id,
        "model": model,
        "source_owner": dict(source_store.owner.identity),
        "source_scope_id": source_store.scope_id,
        "source_workspace": str(source_store.workspace.resolve()),
        "source_codex_home": str(source_store.codex_home.resolve()),
        "source_state_root": str(source_store.manifest_path.parent.resolve()),
        "source_turn_count": source_turn_count,
        "target_workspace": str(source_store.workspace.resolve()),
        "target_codex_home": str(target_codex_home.resolve()),
        "target_state_root": str(target_state_root.resolve()),
        "target_thread_id": None,
        "request_thread_id": normalized_request_thread,
        "request_digest": None,
        "turn_id": None,
        "turn_count_before": None,
    }


def _read_active_job_thread_id(active_job: Any | None) -> str | None:
    """读取等待作业绑定的原始 thread；缺失时返回 None。"""

    if active_job is None:
        return None
    value = (
        getattr(active_job, "codex_thread_id", None)
        if not isinstance(active_job, Mapping)
        else active_job.get("codex_thread_id")
    )
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(
            "provider migration active job thread is missing",
            kind="session_state",
        )
    return value.strip()


def _build_provider_migration_target_store(
    source_store: CodexSessionStore,
    target_root: Path,
) -> CodexSessionStore:
    """按 source state 下的固定目标目录构造下一层 store。"""

    if target_root.is_symlink():
        raise HarnessError(
            "provider migration target root must not be a symlink",
            kind="session_state",
        )
    target_codex_home = target_root / "codex_home"
    return CodexSessionStore(
        source_store.workspace,
        owner=source_store.owner,
        scope_id=source_store.scope_id,
        model=source_store.model,
        codex_home=target_codex_home,
    )


def _build_existing_provider_migration_target_store(
    source_store: CodexSessionStore,
    target_root: Path,
) -> CodexSessionStore:
    """只为已经完整存在的 target state 构造 store，不创建恢复目录。"""

    if target_root.is_symlink() or not target_root.is_dir():
        raise HarnessError(
            "provider migration target root is missing or not a regular directory",
            kind="session_state",
        )
    target_codex_home = target_root / "codex_home"
    target_state_root = target_root / "state"
    for path, field_name in (
        (target_codex_home, "target_codex_home"),
        (target_state_root, "target_state_root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise HarnessError(
                f"provider migration {field_name} is missing or not a regular directory",
                kind="session_state",
            )
    # 构造函数本身会准备 workspace/CODEX_HOME/state；上面已先核对这些目录
    # 全部存在，因此只会加载既有路径，不会让恢复 resolver 创建新层级。
    return _build_provider_migration_target_store(source_store, target_root)


def _expected_provider_migration_target_root(
    source_store: CodexSessionStore,
    target_policy_digest: str,
) -> Path:
    """返回某个 source state 对应的唯一下一层 migration 根目录。"""

    migration_parent = source_store.manifest_path.parent / "provider_migrations"
    if migration_parent.is_symlink():
        raise HarnessError(
            "provider migration parent must not be a symlink",
            kind="session_state",
        )
    target_root = (
        migration_parent / target_policy_digest[:32]
    )
    if target_root.is_symlink():
        raise HarnessError(
            "provider migration target root must not be a symlink",
            kind="session_state",
        )
    return target_root


def _validate_provider_migration_store_root(
    store: CodexSessionStore,
    session_root: Path | str,
) -> Path:
    """核对 source store 和 session_root，防止 marker 把恢复导向别的任务目录。"""

    try:
        root = Path(session_root).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError) as exc:
        raise HarnessError("provider migration session root is invalid", kind="session_state") from exc
    state_root = store.manifest_path.parent.resolve(strict=False)
    if not _path_is_within(state_root, root):
        raise HarnessError(
            "provider migration source store is outside the configured session root",
            kind="session_state",
        )
    return root


def _validate_durable_provider_migration_identity(
    source_store: CodexSessionStore,
    source_manifest: Mapping[str, Any],
    marker: Mapping[str, Any],
    target_root: Path,
) -> None:
    """核对 durable marker 的 owner、workspace、CODEX_HOME、scope 和 turn 计数。"""

    _validate_migration_owner(marker.get("source_owner"), source_store.owner.identity)
    expected_paths = {
        "source_workspace": source_store.workspace,
        "source_codex_home": source_store.codex_home,
        "source_state_root": source_store.manifest_path.parent,
        "target_workspace": source_store.workspace,
        "target_codex_home": target_root / "codex_home",
        "target_state_root": target_root / "state",
    }
    for field_name, expected in expected_paths.items():
        actual = _resolve_migration_path(marker.get(field_name), field_name)
        if actual != expected.resolve(strict=False):
            raise HarnessError(
                f"provider migration {field_name} does not match session store",
                kind="session_state",
            )
    if marker.get("source_scope_id") != source_store.scope_id:
        raise HarnessError(
            "provider migration source scope does not match session store",
            kind="session_state",
        )
    source_count = _validate_migration_count(
        source_manifest.get("turn_count"),
        "source_turn_count",
    )
    marker_count = _validate_migration_count(marker.get("source_turn_count"), "source_turn_count")
    if source_count != marker_count:
        raise HarnessError(
            "provider migration source turn count does not match manifest",
            kind="session_state",
        )
    if marker.get("model") != source_store.model:
        raise HarnessError(
            "provider migration model does not match source session store",
            kind="session_state",
        )


def _validate_provider_identity_marker(
    marker: Mapping[str, Any],
    *,
    auth_mode: str | None,
    provider_id: str | None,
    model: str | None,
) -> None:
    """在目标链尾核对当前 auth/provider/model，防止串用另一套配置。"""

    expected_values = {
        "auth_mode": auth_mode,
        "provider_id": provider_id,
        "model": model,
    }
    for field_name, expected in expected_values.items():
        if expected is not None and marker.get(field_name) != expected:
            raise HarnessError(
                f"provider migration {field_name} does not match current provider",
                kind="session_state",
            )


def _write_provider_migration_marker_update(
    source_store: CodexSessionStore,
    updates: Mapping[str, Any],
) -> None:
    """原子更新 durable marker 的可恢复窗口字段；旧 v1 marker 保持原样。"""

    marker_path = source_store.manifest_path.parent / CODEX_PROVIDER_MIGRATION_MARKER_NAME
    marker = _read_migration_marker(marker_path)
    if marker is None:
        raise HarnessError(
            "provider migration marker is missing while updating recovery identity",
            kind="session_state",
        )
    if marker.get("schema_version") not in _DURABLE_MIGRATION_SCHEMA_VERSIONS:
        # 旧任务的 marker 字段集合和行为必须原样保留；它们没有足够字段描述
        # target thread 的中断窗口，只能继续使用旧恢复分支。
        return
    changed = dict(marker)
    for field_name, value in updates.items():
        if field_name not in _MIGRATION_FIELDS_V2:
            raise HarnessError(
                "provider migration update field is unsupported",
                kind="session_state",
            )
        if field_name in {"target_thread_id", "request_thread_id", "turn_id"}:
            normalized = _validate_migration_identifier(
                value,
                field_name,
                allow_none=True,
            )
        elif field_name in {"request_digest"}:
            normalized = _validate_migration_optional_digest(value, field_name)
        elif field_name == "turn_count_before":
            normalized = _validate_migration_count(value, field_name, allow_none=True)
        else:  # pragma: no cover - caller only exposes the fields above
            normalized = value
        current = changed.get(field_name)
        if current is not None and normalized is not None and current != normalized:
            raise HarnessError(
                f"provider migration {field_name} conflicts with existing marker",
                kind="session_state",
            )
        if current is None:
            changed[field_name] = normalized
    if changed != marker:
        _atomic_write_json_file(
            marker_path,
            changed,
            description="Codex provider migration marker update",
            mode=0o600,
        )


def persist_provider_migration_turn_context(
    selection: ProviderMigrationSelection,
    *,
    request_thread_id: str | None,
    request_digest: str,
    turn_count_before: int,
) -> None:
    """保存新 target turn 的 request/count 身份，供进程中断后精确恢复。"""

    source_store = selection.migration_source_store
    if source_store is None:
        return
    normalized_request_thread = _validate_migration_identifier(
        request_thread_id,
        "request_thread_id",
        allow_none=True,
    )
    if (
        normalized_request_thread is not None
        and selection.request_thread_ids
        and normalized_request_thread not in selection.request_thread_ids
    ):
        raise HarnessError(
            "provider migration request thread is not in the verified migration chain",
            kind="session_state",
        )
    _validate_migration_digest(request_digest, "request_digest")
    _validate_migration_count(turn_count_before, "turn_count_before")
    _write_provider_migration_marker_update(
        source_store,
        {
            "request_thread_id": normalized_request_thread,
            "request_digest": request_digest,
            "turn_count_before": turn_count_before,
        },
    )


def persist_provider_migration_target_thread(
    selection: ProviderMigrationSelection,
    thread_id: str,
) -> None:
    """在 target manifest 写入前持久化 provider 返回的 target thread 身份。"""

    source_store = selection.migration_source_store
    if source_store is None:
        return
    normalized_thread_id = _validate_migration_identifier(thread_id, "target_thread_id")
    if normalized_thread_id is None:  # pragma: no cover - validator rejects None
        raise HarnessError("provider migration target thread is missing", kind="session_state")
    if normalized_thread_id == selection.source_thread_id:
        raise HarnessError(
            "provider migration target must use a new thread",
            kind="session_state",
        )
    source_manifest = source_store.load()
    if source_manifest is not None and normalized_thread_id == source_manifest.get("thread_id"):
        raise HarnessError(
            "provider migration target must use a new thread",
            kind="session_state",
        )
    _write_provider_migration_marker_update(
        source_store,
        {"target_thread_id": normalized_thread_id},
    )


def persist_provider_migration_turn_id(
    selection: ProviderMigrationSelection,
    turn_id: str,
) -> None:
    """在上层生命周期回调前保存 target provider 已接受的 turn 身份。"""

    source_store = selection.migration_source_store
    if source_store is None:
        return
    normalized_turn_id = _validate_migration_identifier(turn_id, "turn_id")
    if normalized_turn_id is None:  # pragma: no cover - validator rejects None
        raise HarnessError("provider migration turn is missing", kind="session_state")
    _write_provider_migration_marker_update(
        source_store,
        {"turn_id": normalized_turn_id},
    )


def read_provider_migration_target_thread(
    selection: ProviderMigrationSelection,
) -> str | None:
    """读取已登记的 target thread；旧 marker 没有此字段时返回空值。"""

    source_store = selection.migration_source_store
    if source_store is None:
        return selection.target_thread_id
    marker = _read_migration_marker(
        source_store.manifest_path.parent / CODEX_PROVIDER_MIGRATION_MARKER_NAME,
    )
    if (
        marker is None
        or marker.get("schema_version") not in _DURABLE_MIGRATION_SCHEMA_VERSIONS
    ):
        return None
    return _validate_migration_identifier(
        marker.get("target_thread_id"),
        "target_thread_id",
        allow_none=True,
    )


def _validate_provider_migration_link(
    source_store: CodexSessionStore,
    source_manifest: Mapping[str, Any],
    marker: Mapping[str, Any],
    *,
    session_root: Path | None = None,
    expected_auth_mode: str | None = None,
    expected_provider_id: str | None = None,
    expected_model: str | None = None,
    expected_request_thread_id: str | None = None,
    read_only: bool = False,
) -> tuple[CodexSessionStore, Mapping[str, Any] | None]:
    """核对一条 source manifest→marker→target manifest 链接。"""

    source_thread_id = source_manifest.get("thread_id")
    source_policy_digest = source_manifest.get("thread_policy_digest")
    if marker.get("source_thread_id") != source_thread_id:
        raise HarnessError(
            "provider migration marker source thread does not match manifest",
            kind="session_state",
        )
    if marker.get("source_policy_digest") != source_policy_digest:
        raise HarnessError(
            "provider migration marker source policy does not match manifest",
            kind="session_state",
        )
    if marker.get("model") != source_store.model:
        raise HarnessError(
            "provider migration marker model does not match source session",
            kind="session_state",
        )
    if session_root is not None:
        _validate_provider_migration_store_root(source_store, session_root)
    target_policy_digest = _validate_migration_digest(
        marker.get("target_policy_digest"),
        "target_policy_digest",
    )
    if target_policy_digest is None:  # pragma: no cover - validator rejects None
        raise HarnessError("provider migration target policy is missing", kind="session_state")
    expected_root = _expected_provider_migration_target_root(
        source_store,
        target_policy_digest,
    )
    marker_root = Path(str(marker.get("target_root"))).expanduser()
    if not marker_root.is_absolute() or marker_root != expected_root:
        raise HarnessError(
            "provider migration marker target root does not match source state",
            kind="session_state",
        )
    if session_root is not None and not _path_is_within(expected_root.resolve(strict=False), session_root):
        raise HarnessError(
            "provider migration target root is outside the configured session root",
            kind="session_state",
        )
    if marker.get("schema_version") in _DURABLE_MIGRATION_SCHEMA_VERSIONS:
        _validate_durable_provider_migration_identity(
            source_store,
            source_manifest,
            marker,
            expected_root,
        )
    _validate_provider_identity_marker(
        marker,
        auth_mode=expected_auth_mode,
        provider_id=expected_provider_id,
        model=expected_model,
    )
    target_store = (
        _build_existing_provider_migration_target_store(source_store, expected_root)
        if read_only
        else _build_provider_migration_target_store(source_store, expected_root)
    )
    target_manifest = target_store.load()
    if target_manifest is not None:
        if target_manifest.get("thread_policy_digest") != target_policy_digest:
            raise HarnessError(
                "provider migration target policy does not match marker",
                kind="session_state",
            )
        if target_manifest.get("thread_id") == source_thread_id:
            raise HarnessError(
                "provider migration target must use a new thread",
                kind="session_state",
            )
        target_thread_id = marker.get("target_thread_id")
        if target_thread_id is not None and target_manifest.get("thread_id") != target_thread_id:
            raise HarnessError(
                "provider migration target thread does not match marker",
                kind="session_state",
            )
    elif marker.get("schema_version") in _DURABLE_MIGRATION_SCHEMA_VERSIONS:
        marker_request_thread_id = _validate_migration_identifier(
            marker.get("request_thread_id"),
            "request_thread_id",
            allow_none=True,
        )
        normalized_expected_request_thread_id = _validate_migration_identifier(
            expected_request_thread_id,
            "request_thread_id",
            allow_none=True,
        )
        if (
            marker_request_thread_id is not None
            and normalized_expected_request_thread_id is not None
            and marker_request_thread_id != normalized_expected_request_thread_id
        ):
            raise HarnessError(
                "provider migration marker request thread does not match request",
                kind="session_state",
            )
        target_thread_id = marker.get("target_thread_id")
        if target_thread_id is not None:
            # provider thread 已经有了公开身份，但本地 manifest 尚未来得及写入；
            # 这是可恢复的短窗口，不应被当作完整 target 缺失。
            _validate_migration_identifier(target_thread_id, "target_thread_id")
            if target_thread_id == source_thread_id:
                raise HarnessError(
                    "provider migration target must use a new thread",
                    kind="session_state",
                )
    return target_store, target_manifest


def _validate_migration_request_thread(
    request_thread_id: str | None,
    allowed_thread_ids: tuple[str, ...],
) -> None:
    """只允许 source 或当前已验证链尾 thread 出现在 wake envelope。"""

    if request_thread_id is not None and request_thread_id not in allowed_thread_ids:
        raise HarnessError(
            "provider migration request thread is not in the verified migration chain",
            kind="session_state",
        )


def _build_allowed_migration_request_threads(
    *,
    source_thread_id: str,
    latest_thread_id: str,
    previous_thread_id: str | None,
    latest_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """返回 wake 可携带的 thread；仅为未完成注册的链尾保留直接前驱。"""

    allowed = [source_thread_id, latest_thread_id]
    if (
        previous_thread_id is not None
        and latest_manifest.get("status") == "created"
        and latest_manifest.get("turn_count") == 0
    ):
        # provider 已创建新 thread、但进程可能在 Working Memory 登记前中断。
        # 允许旧 session 身份通过恢复门禁；实际 Codex turn 仍绑定 latest store。
        allowed.append(previous_thread_id)
    return tuple(dict.fromkeys(allowed))


def _find_unstarted_thread_to_replace(
    *,
    request_thread_id: str | None,
    previous_thread_id: str | None,
    latest_thread_id: str,
    latest_manifest: Mapping[str, Any],
) -> str | None:
    """只为直接前驱请求标记尚未开始的链尾 thread。"""

    if (
        previous_thread_id is not None
        and request_thread_id == previous_thread_id
        and latest_manifest.get("status") == "created"
        and latest_manifest.get("turn_count") == 0
    ):
        return latest_thread_id
    return None


def resolve_explicit_provider_migration(
    store: CodexSessionStore,
    *,
    session_root: Path | str,
    target_policy_digest: str,
    auth_mode: str,
    provider_id: str,
    model: str,
    allow_migration: bool,
    request_thread_id: str | None = None,
    active_job: Any | None = None,
) -> ProviderMigrationSelection:
    """选择原 store 或沿已验证链显式迁移后的新 store。

    输入是已构造的当前 store 和当前 effective policy 摘要；动作先比较旧
    manifest，默认不迁移而 fail-closed。已有等待 job 的恢复请求可以携带最初
    source 或当前已验证链尾 thread；active job 始终必须绑定最初 source。每次
    policy 变化都从当前链尾建立下一层 ``provider_migrations/<digest>/codex_home``，
    绝不 resume policy 不同的 thread，也不在 migration seam 内 submit、取消或宣称
    终态。返回值同时提供恢复 envelope 可接受的 thread 集合。
    """

    _validate_migration_digest(target_policy_digest, "target_policy_digest")
    normalized_session_root = _validate_provider_migration_store_root(store, session_root)
    if store.model != model:
        raise HarnessError(
            "provider migration model does not match source session",
            kind="session_state",
        )
    manifest = store.load()
    if manifest is None:
        return ProviderMigrationSelection(store, None, ())
    old_policy_digest = manifest.get("thread_policy_digest")
    source_thread_id = manifest.get("thread_id")
    if not isinstance(source_thread_id, str) or not source_thread_id.strip():
        raise HarnessError("provider migration source thread is missing", kind="session_state")
    source_thread_id = source_thread_id.strip()
    active_thread_id = _read_active_job_thread_id(active_job)
    if active_thread_id is not None and active_thread_id != source_thread_id:
        raise HarnessError(
            "provider migration active job is not bound to the initial source thread",
            kind="session_state",
        )
    if old_policy_digest == target_policy_digest:
        if request_thread_id is not None and request_thread_id != source_thread_id:
            raise HarnessError(
                "request thread does not match the current session manifest",
                kind="session_state",
            )
        return ProviderMigrationSelection(store, source_thread_id, (source_thread_id,))
    if not allow_migration:
        raise HarnessError(
            "Codex provider/auth policy changed; explicit provider migration is required",
            kind="session_state",
        )
    if request_thread_id is not None and active_job is None:
        raise HarnessError(
            "provider migration requires a new request without the old thread_id",
            kind="session_state",
        )

    current_store = store
    latest_thread_id = source_thread_id
    previous_thread_id: str | None = None
    visited_roots: set[Path] = set()
    while True:
        current_manifest = current_store.load()
        if current_manifest is None:
            raise HarnessError(
                "provider migration chain source manifest is missing",
                kind="session_state",
            )
        current_policy_digest = current_manifest.get("thread_policy_digest")
        if current_policy_digest == target_policy_digest:
            allowed = _build_allowed_migration_request_threads(
                source_thread_id=source_thread_id,
                latest_thread_id=latest_thread_id,
                previous_thread_id=previous_thread_id,
                latest_manifest=current_manifest,
            )
            _validate_migration_request_thread(request_thread_id, allowed)
            return ProviderMigrationSelection(
                current_store,
                source_thread_id,
                allowed,
                _find_unstarted_thread_to_replace(
                    request_thread_id=request_thread_id,
                    previous_thread_id=previous_thread_id,
                    latest_thread_id=latest_thread_id,
                    latest_manifest=current_manifest,
                ),
            )

        marker_path = current_store.manifest_path.parent / CODEX_PROVIDER_MIGRATION_MARKER_NAME
        marker = _read_migration_marker(marker_path)
        if marker is None:
            target_root = _expected_provider_migration_target_root(
                current_store,
                target_policy_digest,
            )
            if target_root.exists():
                if target_root.is_symlink() or any(target_root.iterdir()):
                    raise HarnessError(
                        "provider migration target exists without an audit marker",
                        kind="session_state",
                    )
            expected_marker = _build_migration_marker(
                source_store=current_store,
                source_manifest=current_manifest,
                target_policy_digest=target_policy_digest,
                target_root=target_root,
                auth_mode=auth_mode,
                provider_id=provider_id,
                model=model,
                request_thread_id=request_thread_id,
            )
            _atomic_write_json_file(
                marker_path,
                expected_marker,
                description="Codex provider migration marker",
                mode=0o600,
            )
            target_store = _build_provider_migration_target_store(current_store, target_root)
            allowed = tuple(dict.fromkeys((source_thread_id, latest_thread_id)))
            _validate_migration_request_thread(request_thread_id, allowed)
            return ProviderMigrationSelection(
                target_store,
                source_thread_id,
                allowed,
                migration_source_store=current_store,
                target_thread_id=None,
            )

        target_store, target_manifest = _validate_provider_migration_link(
            current_store,
            current_manifest,
            marker,
            session_root=normalized_session_root,
            expected_auth_mode=(
                auth_mode
                if marker.get("target_policy_digest") == target_policy_digest
                else None
            ),
            expected_provider_id=(
                provider_id
                if marker.get("target_policy_digest") == target_policy_digest
                else None
            ),
            expected_model=(
                model
                if marker.get("target_policy_digest") == target_policy_digest
                else None
            ),
            expected_request_thread_id=request_thread_id,
        )
        target_root = target_store.codex_home.parent
        if target_root in visited_roots:
            raise HarnessError("provider migration chain contains a cycle", kind="session_state")
        visited_roots.add(target_root)
        if target_manifest is not None:
            previous_thread_id = latest_thread_id
            latest_thread_id = target_manifest["thread_id"]
        marker_target_digest = marker["target_policy_digest"]
        if marker_target_digest == target_policy_digest:
            pending_allowed = [source_thread_id, latest_thread_id]
            marker_target_thread_id = _validate_migration_identifier(
                marker.get("target_thread_id"),
                "target_thread_id",
                allow_none=True,
            )
            if marker_target_thread_id is not None:
                pending_allowed.append(marker_target_thread_id)
            allowed = (
                tuple(dict.fromkeys(pending_allowed))
                if target_manifest is None
                else _build_allowed_migration_request_threads(
                    source_thread_id=source_thread_id,
                    latest_thread_id=latest_thread_id,
                    previous_thread_id=previous_thread_id,
                    latest_manifest=target_manifest,
                )
            )
            _validate_migration_request_thread(request_thread_id, allowed)
            return ProviderMigrationSelection(
                target_store,
                source_thread_id,
                allowed,
                (
                    None
                    if marker.get("target_thread_id") is not None
                    or target_manifest is None
                    else _find_unstarted_thread_to_replace(
                    request_thread_id=request_thread_id,
                    previous_thread_id=previous_thread_id,
                    latest_thread_id=latest_thread_id,
                    latest_manifest=target_manifest,
                    )
                ),
                migration_source_store=(current_store if target_manifest is None else None),
                target_thread_id=(
                    marker_target_thread_id
                    if marker.get("schema_version")
                    in _DURABLE_MIGRATION_SCHEMA_VERSIONS
                    else None
                ),
            )
        if target_manifest is None:
            raise HarnessError(
                "provider migration chain target manifest is missing",
                kind="session_state",
            )
        current_store = target_store


def resolve_provider_migration_target_store_read_only(
    store: CodexSessionStore,
    *,
    session_root: Path | str,
    target_thread_id: str,
    target_policy_digest: str,
    auth_mode: str | None = None,
    provider_id: str | None = None,
    model: str | None = None,
) -> CodexSessionStore:
    """沿已持久化迁移链读取结果所属 store，不创建目录或启动 provider。

    该入口只接受已经存在的 source/target state、manifest 和 receipt 所在目录；
    它不会调用会写 marker 的显式迁移选择器。目标 thread、策略和每一层 owner/
    workspace/CODEX_HOME 必须完全一致，任何缺口都直接失败关闭。
    """

    _validate_migration_identifier(target_thread_id, "target_thread_id")
    _validate_migration_digest(target_policy_digest, "target_policy_digest")
    normalized_session_root = _validate_provider_migration_store_root(store, session_root)
    if model is not None and store.model != model:
        raise HarnessError(
            "provider migration model does not match source session",
            kind="session_state",
        )
    current_store = store
    current_manifest = current_store.load()
    if current_manifest is None:
        raise HarnessError(
            "provider migration source manifest is missing",
            kind="session_state",
        )
    visited_roots: set[Path] = set()
    while True:
        current_thread_id = current_manifest.get("thread_id")
        current_policy_digest = current_manifest.get("thread_policy_digest")
        if current_thread_id == target_thread_id and current_policy_digest == target_policy_digest:
            return current_store
        marker_path = current_store.manifest_path.parent / CODEX_PROVIDER_MIGRATION_MARKER_NAME
        marker = _read_migration_marker(marker_path)
        if marker is None:
            raise HarnessError(
                "provider migration target store is not present in the verified chain",
                kind="session_state",
            )
        marker_target_digest = _validate_migration_digest(
            marker.get("target_policy_digest"),
            "target_policy_digest",
        )
        target_store, target_manifest = _validate_provider_migration_link(
            current_store,
            current_manifest,
            marker,
            session_root=normalized_session_root,
            expected_auth_mode=(
                auth_mode if marker_target_digest == target_policy_digest else None
            ),
            expected_provider_id=(
                provider_id if marker_target_digest == target_policy_digest else None
            ),
            expected_model=(model if marker_target_digest == target_policy_digest else None),
            expected_request_thread_id=None,
            read_only=True,
        )
        target_root = target_store.codex_home.parent.resolve(strict=False)
        if target_root in visited_roots:
            raise HarnessError("provider migration chain contains a cycle", kind="session_state")
        visited_roots.add(target_root)
        if target_manifest is None:
            # 只有 target thread marker、没有 target manifest 的窗口尚可用于
            # 继续 SDK resume，但不可能为已经落盘的 Result 提供 completed receipt。
            raise HarnessError(
                "provider migration target manifest is missing for completed Result",
                kind="session_state",
            )
        if marker_target_digest == target_policy_digest:
            if target_manifest.get("thread_id") != target_thread_id:
                raise HarnessError(
                    "provider migration target thread does not match Result",
                    kind="session_state",
                )
            return target_store
        current_store = target_store
        current_manifest = target_manifest


def select_explicit_provider_migration_store(
    store: CodexSessionStore,
    *,
    session_root: Path | str,
    target_policy_digest: str,
    auth_mode: str,
    provider_id: str,
    model: str,
    allow_migration: bool,
    request_thread_id: str | None = None,
    active_job: Any | None = None,
) -> CodexSessionStore:
    """兼容旧调用方，只返回已验证 migration selection 的目标 store。"""

    return resolve_explicit_provider_migration(
        store,
        session_root=session_root,
        target_policy_digest=target_policy_digest,
        auth_mode=auth_mode,
        provider_id=provider_id,
        model=model,
        allow_migration=allow_migration,
        request_thread_id=request_thread_id,
        active_job=active_job,
    ).store


__all__ = [
    "CODEX_PROVIDER_MIGRATION_MARKER_NAME",
    "ProviderMigrationSelection",
    "persist_provider_migration_target_thread",
    "persist_provider_migration_turn_id",
    "persist_provider_migration_turn_context",
    "read_provider_migration_target_thread",
    "resolve_explicit_provider_migration",
    "resolve_provider_migration_target_store_read_only",
    "select_explicit_provider_migration_store",
]
