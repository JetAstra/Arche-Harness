"""MiniChem capability scope 与请求级工作区选择。

本模块只提供 MiniChem API 内的隔离边界，不创建操作系统容器、不改变用户权限，
也不阻止可信服务进程访问宿主机。HTTP 认证层验证 capability 后，把当前请求绑定
到预先登记的工作区；需要读取工作区的下层模块通过 ContextVar 读取绑定，避免并发
请求通过 ``os.chdir`` 或修改共享环境变量互相影响。
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


CAPABILITY_HEADER = "x-minichem-capability"
SCOPE_WORKSPACE_ENV = "MINICHEM_MCP_WORKSPACE"
SCOPE_ID_ENV = "MINICHEM_SCOPE_ID"
SCOPE_TASK_ENV = "MINICHEM_SCOPE_TASK"
SCOPE_BATCH_ENV = "MINICHEM_SCOPE_BATCH"
SCOPE_COMBINATION_ENV = "MINICHEM_SCOPE_COMBINATION"
SHARED_SCHEDULER_ROOT_ENV = "MINICHEM_SHARED_SCHEDULER_ROOT"

_CURRENT_SCOPE: ContextVar["ScopeContext | None"] = ContextVar(
    "minichem_current_scope",
    default=None,
)


class ScopeCapabilityError(RuntimeError):
    """能力注册表、身份或工作区校验失败。"""


class ScopeCapabilityConflict(ScopeCapabilityError):
    """新能力与已有身份或工作区记录冲突。"""


@dataclass(frozen=True)
class ScopeContext:
    """一次已认证请求可使用的不可变 scope 身份和路径。"""

    task_id: str
    batch_id: str
    combination_id: str
    scope_id: str
    workspace: Path
    capability_hash: str
    scheduler_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", self.workspace.expanduser().resolve())
        if self.scheduler_root is not None:
            object.__setattr__(
                self, "scheduler_root", self.scheduler_root.expanduser().resolve()
            )

    def environment(self) -> dict[str, str]:
        """返回给可信 typed worker/supervisor 的显式 scope 环境快照。"""

        values = {
            SCOPE_WORKSPACE_ENV: str(self.workspace),
            SCOPE_ID_ENV: self.scope_id,
            SCOPE_TASK_ENV: self.task_id,
            SCOPE_BATCH_ENV: self.batch_id,
            SCOPE_COMBINATION_ENV: self.combination_id,
        }
        if self.scheduler_root is not None:
            values[SHARED_SCHEDULER_ROOT_ENV] = str(self.scheduler_root)
        return values


@dataclass(frozen=True)
class ProvisionedScopeCapability:
    """能力 provision 调用的一次性返回值；明文 token 不会写入记录。"""

    token: str
    scope: ScopeContext
    record_path: Path


def current_scope() -> ScopeContext | None:
    """返回当前 scope；可信 worker 可通过显式环境快照恢复 scope 元数据。"""

    scoped = _CURRENT_SCOPE.get()
    if scoped is not None:
        return scoped
    workspace = os.environ.get(SCOPE_WORKSPACE_ENV, "").strip()
    scope_id = os.environ.get(SCOPE_ID_ENV, "").strip()
    if not workspace or not scope_id:
        return None
    try:
        workspace_path = _resolve_directory(workspace, "scope workspace")
        scheduler_value = os.environ.get(SHARED_SCHEDULER_ROOT_ENV, "").strip()
        scheduler_path = (
            _resolve_directory(scheduler_value, "shared scheduler root")
            if scheduler_value
            else None
        )
        return ScopeContext(
            task_id=os.environ.get(SCOPE_TASK_ENV, "").strip() or "worker-task",
            batch_id=os.environ.get(SCOPE_BATCH_ENV, "").strip() or "worker-batch",
            combination_id=(
                os.environ.get(SCOPE_COMBINATION_ENV, "").strip()
                or "worker-combination"
            ),
            scope_id=scope_id,
            workspace=workspace_path,
            capability_hash="",
            scheduler_root=scheduler_path,
        )
    except (OSError, RuntimeError, ValueError, ScopeCapabilityError):
        # 子进程环境若被篡改，不能把它当成 legacy 工作区继续运行。
        raise ScopeCapabilityError("scope environment is invalid")


def current_scope_or_raise() -> ScopeContext:
    """要求当前请求已经完成 capability 认证，否则以 fail-closed 方式失败。"""

    scope = current_scope()
    if scope is None:
        raise ScopeCapabilityError("MiniChem request has no authenticated capability scope")
    return scope


@contextmanager
def activate_scope(scope: ScopeContext) -> Iterator[ScopeContext]:
    """在当前异步/线程上下文绑定 scope，并在离开时恢复上层上下文。"""

    token = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


def scope_context_token(scope: ScopeContext) -> Token[ScopeContext | None]:
    """为 ASGI 包装层设置 scope；调用方必须用 ``reset_scope_context`` 配对。"""

    return _CURRENT_SCOPE.set(scope)


def reset_scope_context(token: Token[ScopeContext | None]) -> None:
    """恢复 ASGI 请求前的 ContextVar 值。"""

    _CURRENT_SCOPE.reset(token)


def _safe_identity(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text or len(text) > 256 or any(ord(char) < 0x20 for char in text):
        raise ValueError(f"{name} must be a non-empty safe identity")
    return text


def _resolve_directory(value: str | os.PathLike[str] | Path, name: str) -> Path:
    candidate = Path(value).expanduser()
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    cursor = Path(absolute.anchor or os.curdir)
    for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ScopeCapabilityError(f"{name} cannot contain a symlink")
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise ScopeCapabilityError(f"{name} path cannot be resolved") from exc
    if resolved.exists():
        if resolved.is_symlink() or not resolved.is_dir():
            raise ScopeCapabilityError(f"{name} must be a regular directory")
    else:
        raise ScopeCapabilityError(f"{name} directory does not exist")
    return resolved


def _ensure_private_directory(path: Path, *, create: bool) -> None:
    """创建/检查注册表目录，并拒绝符号链接、其他用户可读权限和特殊文件。"""

    if path.exists() and path.is_symlink():
        raise ScopeCapabilityError("scope capability registry directory cannot be a symlink")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir() or path.is_symlink():
        raise ScopeCapabilityError("scope capability registry directory is invalid")
    os.chmod(path, 0o700)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ScopeCapabilityError("scope capability registry directory must be mode 0700")


def _validate_private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ScopeCapabilityError("scope capability record must be a regular file")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ScopeCapabilityError("scope capability record must be mode 0600")


_SCOPE_RECORD_FILENAME = re.compile(r"cap_[0-9a-f]{64}\.json")
_SUPPORTED_SCOPE_RECORD_SCHEMA_VERSIONS = frozenset({1, 2})
_SCOPE_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "token_sha256",
        "task_id",
        "batch_id",
        "combination_id",
        "scope_id",
        "workspace",
        "revoked",
        "revoked_at",
        "correction_event_ref",
        "waiting_resume_event_ref",
        "previous_generation_hash",
    }
)


def _validate_correction_event_ref(value: Any) -> str:
    """校验只能授权一代新 capability 的不可变更正事件引用。"""

    if not isinstance(value, str):
        raise ScopeCapabilityError("correction event ref is invalid")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or any(ord(char) < 0x21 or ord(char) > 0x7E for char in normalized)
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ScopeCapabilityError("correction event ref is invalid")
    return normalized


def _validate_waiting_resume_event_ref(value: Any) -> str:
    """校验独立 waiting resume 授权事件引用。"""

    if not isinstance(value, str):
        raise ScopeCapabilityError("waiting resume event ref is invalid")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or any(ord(char) < 0x21 or ord(char) > 0x7E for char in normalized)
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ScopeCapabilityError("waiting resume event ref is invalid")
    return normalized


def _validate_generation_hash(value: Any) -> str:
    """校验前一代 capability token 的 SHA-256 摘要。"""

    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ScopeCapabilityError("previous generation hash is invalid")
    return value


def _validate_scope_record(path: Path, value: Any) -> dict[str, Any]:
    """严格核对 capability 记录的 schema、类型、文件名和 active/revoked 状态。"""

    if _SCOPE_RECORD_FILENAME.fullmatch(path.name) is None:
        raise ScopeCapabilityError("scope capability registry contains an unknown entry")
    if not isinstance(value, dict):
        raise ScopeCapabilityError("scope capability record must be an object")
    if set(value) - _SCOPE_RECORD_FIELDS:
        raise ScopeCapabilityError("scope capability record contains unknown fields")
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in _SUPPORTED_SCOPE_RECORD_SCHEMA_VERSIONS:
        raise ScopeCapabilityError("scope capability record schema is invalid")
    if schema_version == 1 and {
        "correction_event_ref",
        "waiting_resume_event_ref",
        "previous_generation_hash",
    } & set(value):
        raise ScopeCapabilityError(
            "schema v1 scope capability record cannot contain authorization metadata"
        )
    if schema_version == 2 and "correction_event_ref" not in value:
        raise ScopeCapabilityError(
            "schema v2 scope capability record must contain correction event ref"
        )
    token_hash = value.get("token_sha256")
    if not isinstance(token_hash, str) or re.fullmatch(r"[0-9a-f]{64}", token_hash) is None:
        raise ScopeCapabilityError("scope capability record token hash is invalid")
    for name in ("task_id", "batch_id", "combination_id", "scope_id"):
        field = value.get(name)
        if not isinstance(field, str):
            raise ScopeCapabilityError("scope capability record identity is invalid")
        _safe_identity(field, name)
    workspace = value.get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        raise ScopeCapabilityError("scope capability record workspace is invalid")
    _resolve_directory(workspace, "scope workspace")
    revoked = value.get("revoked", False)
    if not isinstance(revoked, bool):
        raise ScopeCapabilityError("scope capability record revoked state is invalid")
    revoked_at = value.get("revoked_at")
    if revoked:
        if not isinstance(revoked_at, str) or not revoked_at.strip():
            raise ScopeCapabilityError("revoked scope capability record is incomplete")
    elif revoked_at is not None:
        raise ScopeCapabilityError("active scope capability record has revoked timestamp")
    event_ref = value.get("correction_event_ref")
    if schema_version == 2 and event_ref is not None:
        _validate_correction_event_ref(event_ref)
    waiting_event_ref = value.get("waiting_resume_event_ref")
    previous_hash = value.get("previous_generation_hash")
    if (waiting_event_ref is None) != (previous_hash is None):
        raise ScopeCapabilityError("waiting resume metadata is incomplete")
    if event_ref is not None and waiting_event_ref is not None:
        raise ScopeCapabilityError("correction and waiting authorization cannot be mixed")
    if waiting_event_ref is not None:
        _validate_waiting_resume_event_ref(waiting_event_ref)
        _validate_generation_hash(previous_hash)
    expected_filename = f"cap_{token_hash}.json"
    if path.name != expected_filename:
        raise ScopeCapabilityError("scope capability record filename does not match token hash")
    return value


class ScopeCapabilityRegistry:
    """保存 capability 哈希及 scope 元数据的本地私有注册表。"""

    def __init__(self, root: str | os.PathLike[str] | Path) -> None:
        candidate = Path(root).expanduser()
        absolute = Path(os.path.abspath(os.fspath(candidate)))
        cursor = Path(absolute.anchor or os.curdir)
        for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ScopeCapabilityError(
                    "scope capability registry cannot contain a symlink"
                )
        self.root = candidate.resolve()
        _ensure_private_directory(self.root, create=True)
        self._lock_path = self.root / ".registry.lock"
        if self._lock_path.exists() and self._lock_path.is_symlink():
            raise ScopeCapabilityError("scope registry lock cannot be a symlink")
        descriptor = os.open(
            self._lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)

    def _records(self) -> list[tuple[Path, dict[str, Any]]]:
        values: list[tuple[Path, dict[str, Any]]] = []
        try:
            paths = sorted(self.root.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ScopeCapabilityError("scope capability registry cannot be read") from exc
        for path in paths:
            if path.name == ".registry.lock":
                _validate_private_file(path)
                continue
            _validate_private_file(path)
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScopeCapabilityError("scope capability record cannot be read") from exc
            values.append((path, _validate_scope_record(path, value)))
        return values

    @contextmanager
    def _locked(self) -> Iterator[None]:
        _ensure_private_directory(self.root, create=False)
        descriptor = os.open(
            self._lock_path,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _token_hash(token: str) -> str:
        if not isinstance(token, str) or len(token) < 32:
            raise ScopeCapabilityError("invalid capability")
        if any(ord(char) < 0x21 or ord(char) > 0x7E for char in token):
            raise ScopeCapabilityError("invalid capability")
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _scope_from_record(value: Mapping[str, Any]) -> ScopeContext:
        validated = _validate_scope_record(
            Path(f"cap_{value.get('token_sha256', '')}.json"), dict(value)
        )
        workspace = _resolve_directory(validated["workspace"], "scope workspace")
        return ScopeContext(
            task_id=validated["task_id"],
            batch_id=validated["batch_id"],
            combination_id=validated["combination_id"],
            scope_id=validated["scope_id"],
            workspace=workspace,
            capability_hash=validated["token_sha256"],
        )

    def provision_scope_capability(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
        token: str | None = None,
    ) -> ProvisionedScopeCapability:
        """登记 scope capability；支持管理侧 token 的崩溃安全幂等重放。

        未提供 token 时由本方法生成一次随机 token；管理调用方若担心在写入注册表
        后崩溃，可先在自己的 0600 凭据文件中保存随机 token，再通过 ``token=`` 显式
        注册。相同 identity、workspace 和 token hash 的重放返回同一记录；不同 token
        或已撤销记录均拒绝，注册表始终只保存 hash。
        """

        identity = {
            "task_id": _safe_identity(task_id, "task_id"),
            "batch_id": _safe_identity(batch_id, "batch_id"),
            "combination_id": _safe_identity(combination_id, "combination_id"),
            "scope_id": _safe_identity(scope_id, "scope_id"),
        }
        workspace_path = _resolve_directory(workspace, "scope workspace")
        selected_token = token if token is not None else secrets.token_urlsafe(32)
        token_hash = self._token_hash(selected_token)
        record_path = self.root / f"cap_{token_hash}.json"
        with self._locked():
            records = self._records()
            same_identity: list[tuple[Path, dict[str, Any]]] = []
            for existing_path, existing in records:
                existing_identity = {
                    key: existing.get(key) for key in identity
                }
                existing_workspace = existing.get("workspace")
                if existing_identity == identity:
                    try:
                        same_workspace = (
                            _resolve_directory(str(existing_workspace), "scope workspace")
                            == workspace_path
                        )
                    except ScopeCapabilityError as exc:
                        raise ScopeCapabilityConflict(
                            "existing scope record is invalid; refusing to reprovision identity"
                        ) from exc
                    if not same_workspace:
                        raise ScopeCapabilityConflict(
                            "scope identity is already bound to another workspace"
                        )
                    same_identity.append((existing_path, existing))
                    continue
                try:
                    same_workspace = (
                        _resolve_directory(str(existing_workspace), "scope workspace")
                        == workspace_path
                    )
                except ScopeCapabilityError:
                    raise ScopeCapabilityConflict(
                        "existing scope record is invalid; refusing to provision around it"
                    )
                if same_workspace:
                    raise ScopeCapabilityConflict(
                        "scope workspace is already bound to another identity"
                    )
            active_records = [
                item for item in same_identity if not bool(item[1].get("revoked"))
            ]
            if len(active_records) > 1:
                raise ScopeCapabilityConflict(
                    "scope identity has multiple active capability records"
                )
            if active_records:
                existing_path, existing = active_records[0]
                if existing.get("token_sha256") == token_hash:
                    return ProvisionedScopeCapability(
                        token=selected_token,
                        scope=self._scope_from_record(existing),
                        record_path=existing_path,
                    )
                raise ScopeCapabilityConflict(
                    "scope identity is already provisioned; refusing another capability"
                )
            if same_identity:
                raise ScopeCapabilityConflict(
                    "revoked scope capability cannot be reprovisioned"
                )
            if record_path.exists() or any(
                existing.get("token_sha256") == token_hash
                for _path, existing in records
            ):
                raise ScopeCapabilityConflict("capability token or record path is already bound")
            record = {
                "schema_version": 2,
                "token_sha256": token_hash,
                **identity,
                "workspace": str(workspace_path),
                "correction_event_ref": None,
            }
            descriptor = os.open(
                record_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    descriptor = -1
                    json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            os.chmod(record_path, 0o600)
        scope = ScopeContext(
            task_id=identity["task_id"],
            batch_id=identity["batch_id"],
            combination_id=identity["combination_id"],
            scope_id=identity["scope_id"],
            workspace=workspace_path,
            capability_hash=token_hash,
        )
        return ProvisionedScopeCapability(
            token=selected_token, scope=scope, record_path=record_path
        )

    def reauthorize_scope_capability(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
        token: str,
        correction_event_ref: str | None = None,
    ) -> ProvisionedScopeCapability:
        """在唯一更正事件授权下为已撤销 scope 建立新 capability。

        输入是完整 scope 身份、调用方预先保存的 token 和不可变更正事件引用；动作在
        registry 锁内核对至少一个同身份 revoked tombstone、无活动代次、引用未被其他
        token 或 identity 使用，再原子写入新记录。相同引用、身份和 token 的重试返回
        同一活动记录，旧 tombstone 永不覆盖；身份、引用或 token 冲突时拒绝请求。
        """

        event_ref = _validate_correction_event_ref(correction_event_ref)
        identity = {
            "task_id": _safe_identity(task_id, "task_id"),
            "batch_id": _safe_identity(batch_id, "batch_id"),
            "combination_id": _safe_identity(combination_id, "combination_id"),
            "scope_id": _safe_identity(scope_id, "scope_id"),
        }
        workspace_path = _resolve_directory(workspace, "scope workspace")
        selected_token = token
        token_hash = self._token_hash(selected_token)
        record_path = self.root / f"cap_{token_hash}.json"
        with self._locked():
            records = self._records()
            same_identity: list[tuple[Path, dict[str, Any]]] = []
            same_ref: list[tuple[Path, dict[str, Any]]] = []
            for existing_path, existing in records:
                existing_identity = {
                    key: existing.get(key) for key in identity
                }
                try:
                    existing_workspace = _resolve_directory(
                        str(existing.get("workspace")), "scope workspace"
                    )
                except ScopeCapabilityError:
                    raise ScopeCapabilityConflict(
                        "existing scope record is invalid; refusing reauthorization"
                    ) from None
                if existing_identity == identity and existing_workspace == workspace_path:
                    same_identity.append((existing_path, existing))
                elif existing_identity == identity or existing_workspace == workspace_path:
                    raise ScopeCapabilityConflict(
                        "scope identity or workspace is already bound differently"
                    )
                if existing.get("correction_event_ref") == event_ref:
                    same_ref.append((existing_path, existing))
            if len(same_ref) > 1:
                raise ScopeCapabilityConflict("reauthorization event ref is duplicated")
            if same_ref:
                existing_path, existing = same_ref[0]
                if (
                    {
                        key: existing.get(key) for key in identity
                    }
                    != identity
                    or _resolve_directory(str(existing.get("workspace")), "scope workspace")
                    != workspace_path
                    or existing.get("token_sha256") != token_hash
                ):
                    raise ScopeCapabilityConflict(
                        "reauthorization event ref is bound to another token or identity"
                    )
                if existing.get("revoked", False):
                    raise ScopeCapabilityConflict(
                        "revoked reauthorization record cannot be reused"
                    )
                return ProvisionedScopeCapability(
                    token=selected_token,
                    scope=self._scope_from_record(existing),
                    record_path=existing_path,
                )
            if record_path.exists():
                raise ScopeCapabilityConflict("capability record path collision")
            if not same_identity or not all(
                bool(existing.get("revoked")) for _path, existing in same_identity
            ):
                raise ScopeCapabilityConflict(
                    "reauthorization requires a revoked scope tombstone and no active grant"
                )
            record = {
                "schema_version": 2,
                "token_sha256": token_hash,
                **identity,
                "workspace": str(workspace_path),
                "revoked": False,
                "revoked_at": None,
                "correction_event_ref": event_ref,
            }
            descriptor = os.open(
                record_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    descriptor = -1
                    json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            os.chmod(record_path, 0o600)
        scope = ScopeContext(
            task_id=identity["task_id"],
            batch_id=identity["batch_id"],
            combination_id=identity["combination_id"],
            scope_id=identity["scope_id"],
            workspace=workspace_path,
            capability_hash=token_hash,
        )
        return ProvisionedScopeCapability(
            token=selected_token,
            scope=scope,
            record_path=record_path,
        )

    def reauthorize_scope_capability_for_waiting_resume(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
        token: str,
        waiting_resume_event_ref: str | None = None,
        previous_generation_hash: str | None = None,
    ) -> ProvisionedScopeCapability:
        """按独立 WM waiting 事件严格恢复旧 scope capability。

        输入是完整 scope 身份、新 capability token、waiting 事件引用和前代 token
        摘要；动作在注册表锁内允许 active 精确复用，或要求没有 active 且摘要恰好命中
        一个 revoked 代次后追加一条 waiting generation。相同事件、身份、前代摘要和
        token 重试幂等；事件/身份/摘要冲突、混用 correction 引用或多候选匹配时拒绝，
        不覆盖旧 tombstone。
        """

        event_ref = _validate_waiting_resume_event_ref(waiting_resume_event_ref)
        previous_hash = _validate_generation_hash(previous_generation_hash)
        identity = {
            "task_id": _safe_identity(task_id, "task_id"),
            "batch_id": _safe_identity(batch_id, "batch_id"),
            "combination_id": _safe_identity(combination_id, "combination_id"),
            "scope_id": _safe_identity(scope_id, "scope_id"),
        }
        workspace_path = _resolve_directory(workspace, "scope workspace")
        selected_token = token
        token_hash = self._token_hash(selected_token)
        record_path = self.root / f"cap_{token_hash}.json"
        with self._locked():
            records = self._records()
            same_identity: list[tuple[Path, dict[str, Any]]] = []
            same_waiting: list[tuple[Path, dict[str, Any]]] = []
            correction_collision: list[tuple[Path, dict[str, Any]]] = []
            for existing_path, existing in records:
                existing_identity = {key: existing.get(key) for key in identity}
                try:
                    existing_workspace = _resolve_directory(
                        str(existing.get("workspace")), "scope workspace"
                    )
                except ScopeCapabilityError:
                    raise ScopeCapabilityConflict(
                        "existing scope record is invalid; refusing waiting resume"
                    ) from None
                if existing_identity == identity and existing_workspace == workspace_path:
                    same_identity.append((existing_path, existing))
                elif existing_identity == identity or existing_workspace == workspace_path:
                    raise ScopeCapabilityConflict(
                        "scope identity or workspace is already bound differently"
                    )
                if existing.get("waiting_resume_event_ref") == event_ref:
                    same_waiting.append((existing_path, existing))
                if existing.get("correction_event_ref") == event_ref:
                    correction_collision.append((existing_path, existing))
            if correction_collision:
                raise ScopeCapabilityConflict(
                    "waiting resume event cannot reuse correction event ref"
                )
            if len(same_waiting) > 1:
                raise ScopeCapabilityConflict("waiting resume event ref is duplicated")
            if same_waiting:
                existing_path, existing = same_waiting[0]
                if (
                    {key: existing.get(key) for key in identity} != identity
                    or _resolve_directory(str(existing.get("workspace")), "scope workspace")
                    != workspace_path
                    or existing.get("previous_generation_hash") != previous_hash
                    or existing.get("token_sha256") != token_hash
                ):
                    raise ScopeCapabilityConflict(
                        "waiting resume event is bound to another token or identity"
                    )
                if existing.get("revoked", False):
                    raise ScopeCapabilityConflict(
                        "revoked waiting resume record cannot be reused"
                    )
                return ProvisionedScopeCapability(
                    token=selected_token,
                    scope=self._scope_from_record(existing),
                    record_path=existing_path,
                )
            active_records = [
                item for item in same_identity if not bool(item[1].get("revoked"))
            ]
            if len(active_records) > 1:
                raise ScopeCapabilityConflict(
                    "scope identity has multiple active capability records"
                )
            if active_records:
                existing_path, existing = active_records[0]
                if (
                    existing.get("token_sha256") != previous_hash
                    or token_hash != previous_hash
                ):
                    raise ScopeCapabilityConflict(
                        "active scope capability does not match previous generation"
                    )
                return ProvisionedScopeCapability(
                    token=selected_token,
                    scope=self._scope_from_record(existing),
                    record_path=existing_path,
                )
            if not same_identity or not all(
                bool(existing.get("revoked")) for _path, existing in same_identity
            ):
                raise ScopeCapabilityConflict(
                    "waiting resume requires a revoked scope tombstone and no active grant"
                )
            matching_revoked = [
                item
                for item in same_identity
                if item[1].get("token_sha256") == previous_hash
            ]
            if len(matching_revoked) != 1:
                raise ScopeCapabilityConflict(
                    "previous generation hash does not identify exactly one revoked capability"
                )
            if record_path.exists() or any(
                existing.get("token_sha256") == token_hash
                for _path, existing in records
            ):
                raise ScopeCapabilityConflict("capability token or record path is already bound")
            record = {
                "schema_version": 2,
                "token_sha256": token_hash,
                **identity,
                "workspace": str(workspace_path),
                "revoked": False,
                "revoked_at": None,
                "correction_event_ref": None,
                "waiting_resume_event_ref": event_ref,
                "previous_generation_hash": previous_hash,
            }
            descriptor = os.open(
                record_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    descriptor = -1
                    json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            os.chmod(record_path, 0o600)
        scope = ScopeContext(
            task_id=identity["task_id"],
            batch_id=identity["batch_id"],
            combination_id=identity["combination_id"],
            scope_id=identity["scope_id"],
            workspace=workspace_path,
            capability_hash=token_hash,
        )
        return ProvisionedScopeCapability(
            token=selected_token,
            scope=scope,
            record_path=record_path,
        )

    def authenticate_scope_capability(self, token: str) -> ScopeContext:
        """验证 header token，返回绑定 scope；任一记录异常都拒绝请求。"""

        token_hash = self._token_hash(token)
        path = self.root / f"cap_{token_hash}.json"
        with self._locked():
            if not path.exists():
                raise ScopeCapabilityError("invalid capability")
            _validate_private_file(path)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScopeCapabilityError("invalid capability") from exc
            try:
                record = _validate_scope_record(path, record)
            except ScopeCapabilityError as exc:
                raise ScopeCapabilityError("invalid capability") from exc
            stored_hash = record["token_sha256"]
            if not hmac.compare_digest(stored_hash, token_hash):
                raise ScopeCapabilityError("invalid capability")
            if record.get("revoked", False):
                raise ScopeCapabilityError("invalid capability")
            scope = self._scope_from_record(record)
            if scope.capability_hash != token_hash:
                raise ScopeCapabilityError("invalid capability")
            return scope

    def revoke_scope_capability(self, token: str) -> None:
        """禁用一个 capability；保留身份 tombstone，防止终局后重新 mint。"""

        token_hash = self._token_hash(token)
        path = self.root / f"cap_{token_hash}.json"
        with self._locked():
            if not path.exists():
                raise ScopeCapabilityError("invalid capability")
            _validate_private_file(path)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScopeCapabilityError("invalid capability") from exc
            try:
                record = _validate_scope_record(path, record)
            except ScopeCapabilityError as exc:
                raise ScopeCapabilityError("invalid capability") from exc
            if not hmac.compare_digest(record["token_sha256"], token_hash):
                raise ScopeCapabilityError("invalid capability")
            if record.get("revoked", False):
                return
            record["revoked"] = True
            record["revoked_at"] = datetime.now(timezone.utc).isoformat()
            temporary = path.with_name(f".{path.name}.revoking")
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    descriptor = -1
                    json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            try:
                directory_descriptor = os.open(
                    self.root,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
            except OSError as exc:
                raise ScopeCapabilityError(
                    "scope capability revocation directory cannot be synced"
                ) from exc
            try:
                os.fsync(directory_descriptor)
            except OSError as exc:
                raise ScopeCapabilityError(
                    "scope capability revocation directory cannot be synced"
                ) from exc
            finally:
                os.close(directory_descriptor)

    def list_scope_records(self) -> tuple[dict[str, Any], ...]:
        """返回不含 token 的审计记录；该接口仅供显式 provisioning 管理使用。"""

        with self._locked():
            return tuple(dict(value) for _path, value in self._records())


def provision_scope_capability(
    registry_root: str | os.PathLike[str] | Path,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    scope_id: str,
    workspace: str | os.PathLike[str] | Path,
    token: str | None = None,
) -> ProvisionedScopeCapability:
    """显式登记 HTTP scope capability，明文 token 只由调用方管理。"""

    return ScopeCapabilityRegistry(registry_root).provision_scope_capability(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
        token=token,
    )


def revoke_scope_capability(
    registry_root: str | os.PathLike[str] | Path,
    token: str,
) -> None:
    """通过管理者持有的明文 token 禁用 capability，不提供 token 恢复。"""

    ScopeCapabilityRegistry(registry_root).revoke_scope_capability(token)


def reauthorize_scope_capability(
    registry_root: str | os.PathLike[str] | Path,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    scope_id: str,
    workspace: str | os.PathLike[str] | Path,
    token: str,
    correction_event_ref: str | None = None,
) -> ProvisionedScopeCapability:
    """通过唯一更正事件为已撤销 scope 登记新 capability。"""

    return ScopeCapabilityRegistry(registry_root).reauthorize_scope_capability(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
        token=token,
        correction_event_ref=correction_event_ref,
    )


def reauthorize_scope_capability_for_waiting_resume(
    registry_root: str | os.PathLike[str] | Path,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    scope_id: str,
    workspace: str | os.PathLike[str] | Path,
    token: str,
    waiting_resume_event_ref: str | None = None,
    previous_generation_hash: str | None = None,
) -> ProvisionedScopeCapability:
    """使用独立 WM waiting 事件为旧 scope 建立一次恢复 capability。"""

    return ScopeCapabilityRegistry(registry_root).reauthorize_scope_capability_for_waiting_resume(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
        token=token,
        waiting_resume_event_ref=waiting_resume_event_ref,
        previous_generation_hash=previous_generation_hash,
    )


__all__ = [
    "CAPABILITY_HEADER",
    "ProvisionedScopeCapability",
    "SHARED_SCHEDULER_ROOT_ENV",
    "ScopeCapabilityConflict",
    "ScopeCapabilityError",
    "ScopeCapabilityRegistry",
    "ScopeContext",
    "SCOPE_BATCH_ENV",
    "SCOPE_COMBINATION_ENV",
    "SCOPE_ID_ENV",
    "SCOPE_TASK_ENV",
    "SCOPE_WORKSPACE_ENV",
    "activate_scope",
    "current_scope",
    "current_scope_or_raise",
    "provision_scope_capability",
    "reauthorize_scope_capability",
    "reauthorize_scope_capability_for_waiting_resume",
    "revoke_scope_capability",
    "reset_scope_context",
    "scope_context_token",
]
