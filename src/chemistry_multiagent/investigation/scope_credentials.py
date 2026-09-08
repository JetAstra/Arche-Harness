"""调查 scope capability 的本地凭据生命周期。

本模块位于 ARCHE 侧，只负责保存调查 scope 的明文 capability 和调用一个
窄的 MiniChem registry 协议。凭据记录先于 registry 调用以 0600 文件原子落盘，
因此进程在 registry 调用前后退出时，恢复都能继续使用同一个 token。模块不
复制 MiniChem 的 registry 实现，也不依赖其返回对象的具体类型。
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable


_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, _SCHEMA_VERSION})
_RECORD_PREFIX = "scope_"
_RECORD_SUFFIX = ".json"
_LOCK_NAME = ".credentials.lock"
_TEMP_MARKER = ".tmp"
_TOKEN_MIN_LENGTH = 32
_TOKEN_MAX_LENGTH = 512
_REGISTRY_STATES = frozenset({"pending", "registered", "revoked"})
_REQUIRED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "task_id",
        "batch_id",
        "combination_id",
        "scope_id",
        "workspace",
        "token",
        "token_sha256",
        "revoked",
        "status",
        "registry_state",
        "created_at",
        "updated_at",
        "revoked_at",
    }
)
_OPTIONAL_RECORD_FIELDS = frozenset(
    {"correction_event_ref", "waiting_resume_event_ref", "previous_generation_hash"}
)


class InvestigationScopeCredentialError(RuntimeError):
    """本地 scope 凭据或 MiniChem capability 生命周期失败。"""


class InvestigationScopeCredentialConflict(InvestigationScopeCredentialError):
    """请求与已有 scope 凭据身份、路径或 token 摘要冲突。"""


class InvestigationScopeCredentialCorruption(InvestigationScopeCredentialError):
    """凭据记录、权限或记录完整性校验失败。"""


class InvestigationScopeCredentialProvisionError(InvestigationScopeCredentialError):
    """MiniChem capability 注册失败；本地 pending 记录会保留以便重试。"""


class InvestigationScopeCredentialReauthorizationError(
    InvestigationScopeCredentialProvisionError
):
    """MiniChem capability 更正授权失败；本地 pending 记录会保留以便重试。"""


class InvestigationScopeCredentialRevokeError(InvestigationScopeCredentialError):
    """MiniChem capability 撤销失败；本地记录不会提前标记为 revoked。"""


# 这些短别名便于调用方按职责分组，同时不暴露 MiniChem 的实现类。
ScopeCredentialError = InvestigationScopeCredentialError
ScopeCredentialConflict = InvestigationScopeCredentialConflict
ScopeCredentialCorruption = InvestigationScopeCredentialCorruption
ScopeCapabilityConflict = InvestigationScopeCredentialConflict
ScopeCapabilityCredentialError = InvestigationScopeCredentialError


def _utc_now() -> str:
    """返回用于凭据记录的 UTC 时间文本。"""

    return datetime.now(timezone.utc).isoformat()


def _validate_identity_component(value: Any, field_name: str) -> str:
    """校验 task、batch、combination 和 scope 标识，拒绝路径与控制字符。"""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} is invalid")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or normalized in {".", ".."}
        or "\x00" in normalized
        or "/" in normalized
        or "\\" in normalized
        or any(ord(char) < 0x20 for char in normalized)
    ):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _reject_symlinked_path_components(path: Path) -> None:
    """拒绝路径中任一已有组件为符号链接，避免凭据目录跳出预期位置。"""

    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(absolute.anchor or os.curdir)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise InvestigationScopeCredentialCorruption("credential path contains a symlink")


def _resolve_workspace(value: str | os.PathLike[str] | Path) -> Path:
    """解析并校验工作区为存在的普通目录，且不经过符号链接。"""

    try:
        candidate = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise ValueError("workspace is invalid") from exc
    _reject_symlinked_path_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("workspace is invalid") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("workspace is invalid")
    return resolved


def _validate_token(value: Any) -> str:
    """校验 capability token 的形状；错误信息不包含 token 内容。"""

    if not isinstance(value, str) or not (
        _TOKEN_MIN_LENGTH <= len(value) <= _TOKEN_MAX_LENGTH
    ):
        raise ValueError("token is invalid")
    if any(ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise ValueError("token is invalid")
    return value


def _token_sha256(token: str) -> str:
    """计算 token 的 SHA-256 摘要；返回值不包含明文。"""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _canonical_identity(
    *,
    task_id: Any,
    batch_id: Any,
    combination_id: Any,
    scope_id: Any,
    workspace: str | os.PathLike[str] | Path,
) -> "InvestigationScopeIdentity":
    """规范化完整身份，供记录校验、路径派生和冲突检测共用。"""

    try:
        normalized_task = _validate_identity_component(task_id, "task_id")
        normalized_batch = _validate_identity_component(batch_id, "batch_id")
        normalized_combination = _validate_identity_component(
            combination_id, "combination_id"
        )
        normalized_scope = _validate_identity_component(scope_id, "scope_id")
        normalized_workspace = _resolve_workspace(workspace)
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise InvestigationScopeCredentialError("scope identity is invalid") from None
    return InvestigationScopeIdentity(
        task_id=normalized_task,
        batch_id=normalized_batch,
        combination_id=normalized_combination,
        scope_id=normalized_scope,
        workspace=normalized_workspace,
    )


@dataclass(frozen=True, slots=True)
class InvestigationScopeIdentity:
    """一个调查 scope 的完整、规范化身份和工作区路径。"""

    task_id: str
    batch_id: str
    combination_id: str
    scope_id: str
    workspace: Path

    def __post_init__(self) -> None:
        # 该类也作为公开输入类型使用，因此不能只依赖工厂函数完成校验。
        object.__setattr__(self, "task_id", _validate_identity_component(self.task_id, "task_id"))
        object.__setattr__(self, "batch_id", _validate_identity_component(self.batch_id, "batch_id"))
        object.__setattr__(
            self,
            "combination_id",
            _validate_identity_component(self.combination_id, "combination_id"),
        )
        object.__setattr__(self, "scope_id", _validate_identity_component(self.scope_id, "scope_id"))
        object.__setattr__(self, "workspace", _resolve_workspace(self.workspace))

    @property
    def owner_key(self) -> tuple[str, str, str]:
        """返回不含 scope 和工作区的调查会话归属键。"""

        return (self.task_id, self.batch_id, self.combination_id)

    def as_record_fields(self) -> dict[str, str]:
        """返回记录中保存的完整明文身份，不含 token。"""

        return {
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "combination_id": self.combination_id,
            "scope_id": self.scope_id,
            "workspace": str(self.workspace),
        }

    def record_id(self) -> str:
        """返回由完整身份计算的稳定记录标识。"""

        payload = json.dumps(
            self.as_record_fields(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class InvestigationScopeCapability:
    """已注册的调查 capability；``token`` 不参与对象 repr。"""

    token: str = field(repr=False)
    task_id: str
    batch_id: str
    combination_id: str
    scope_id: str
    workspace: Path
    token_sha256: str
    record_path: Path
    correction_event_ref: str | None = None
    waiting_resume_event_ref: str | None = None
    previous_generation_hash: str | None = None

    def __post_init__(self) -> None:
        # 不在 repr 或错误中输出 token；这里只做返回值完整性复核。
        _validate_token(self.token)
        if not hmac.compare_digest(self.token_sha256, _token_sha256(self.token)):
            raise InvestigationScopeCredentialCorruption("capability token digest is invalid")
        identity = _canonical_identity(
            task_id=self.task_id,
            batch_id=self.batch_id,
            combination_id=self.combination_id,
            scope_id=self.scope_id,
            workspace=self.workspace,
        )
        object.__setattr__(self, "task_id", identity.task_id)
        object.__setattr__(self, "batch_id", identity.batch_id)
        object.__setattr__(self, "combination_id", identity.combination_id)
        object.__setattr__(self, "scope_id", identity.scope_id)
        object.__setattr__(self, "workspace", identity.workspace)
        object.__setattr__(self, "record_path", Path(self.record_path))
        if self.correction_event_ref is not None:
            try:
                object.__setattr__(
                    self,
                    "correction_event_ref",
                    _validate_correction_event_ref(self.correction_event_ref),
                )
            except ValueError:
                raise InvestigationScopeCredentialCorruption(
                    "capability correction event ref is invalid"
                ) from None
        waiting_event_ref = self.waiting_resume_event_ref
        previous_hash = self.previous_generation_hash
        if (waiting_event_ref is None) != (previous_hash is None):
            raise InvestigationScopeCredentialCorruption(
                "capability waiting resume generation is incomplete"
            )
        if self.correction_event_ref is not None and waiting_event_ref is not None:
            raise InvestigationScopeCredentialCorruption(
                "capability correction and waiting authorization cannot be mixed"
            )
        if waiting_event_ref is not None:
            try:
                object.__setattr__(
                    self,
                    "waiting_resume_event_ref",
                    _validate_waiting_resume_event_ref(waiting_event_ref),
                )
                object.__setattr__(
                    self,
                    "previous_generation_hash",
                    _validate_generation_hash(previous_hash),
                )
            except ValueError:
                raise InvestigationScopeCredentialCorruption(
                    "capability waiting resume generation is invalid"
                ) from None

    @property
    def identity(self) -> InvestigationScopeIdentity:
        """返回不含 token 的完整 scope 身份。"""

        return InvestigationScopeIdentity(
            task_id=self.task_id,
            batch_id=self.batch_id,
            combination_id=self.combination_id,
            scope_id=self.scope_id,
            workspace=self.workspace,
        )

    @property
    def scope_credential_generation_digest(self) -> str:
        """返回当前 capability 代次的非秘密摘要，不包含 token 或记录路径。"""

        return scope_credential_generation_digest(self)

    @property
    def generation_digest(self) -> str:
        """返回当前 capability 代次摘要的简短只读别名。"""

        return self.scope_credential_generation_digest

    def to_audit_dict(self) -> dict[str, Any]:
        """返回不含明文 token 的审计视图。"""

        return {
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "combination_id": self.combination_id,
            "scope_id": self.scope_id,
            "workspace": str(self.workspace),
            "token_sha256": self.token_sha256,
            "record_path": str(self.record_path),
            "correction_event_ref": self.correction_event_ref,
            "waiting_resume_event_ref": self.waiting_resume_event_ref,
            "previous_generation_hash": self.previous_generation_hash,
        }


@dataclass(frozen=True, slots=True)
class InvestigationScopeCredentialAudit:
    """凭据记录的只读审计视图，绝不保存明文 token。"""

    task_id: str
    batch_id: str
    combination_id: str
    scope_id: str
    workspace: Path
    token_sha256: str
    revoked: bool
    registry_state: str
    record_path: Path
    correction_event_ref: str | None = None
    waiting_resume_event_ref: str | None = None
    previous_generation_hash: str | None = None

    @property
    def scope_credential_generation_digest(self) -> str:
        """返回审计记录对应代次的非秘密摘要，不包含 token 或记录路径。"""

        return scope_credential_generation_digest(self)

    @property
    def generation_digest(self) -> str:
        """返回当前 capability 代次摘要的简短只读别名。"""

        return self.scope_credential_generation_digest

    def to_dict(self) -> dict[str, Any]:
        """返回可写入审计事件的无 secret 字典。"""

        return {
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "combination_id": self.combination_id,
            "scope_id": self.scope_id,
            "workspace": str(self.workspace),
            "token_sha256": self.token_sha256,
            "revoked": self.revoked,
            "registry_state": self.registry_state,
            "record_path": str(self.record_path),
            "correction_event_ref": self.correction_event_ref,
            "waiting_resume_event_ref": self.waiting_resume_event_ref,
            "previous_generation_hash": self.previous_generation_hash,
        }


def scope_credential_generation_digest(
    value: InvestigationScopeCapability
    | InvestigationScopeCredentialAudit
    | Mapping[str, Any],
) -> str:
    """按 scope 身份和授权代次字段生成稳定的非秘密摘要。

    输入是 capability、审计对象或包含完整身份的映射；动作只读取 task/batch/
    combination/scope/workspace、token_sha256 以及 correction/waiting 授权字段，
    忽略 token、record_path、状态时间和其它本地路径。返回小写 SHA-256 摘要；身份、
    token 摘要或授权字段非法时抛出 scope 凭据错误，不把明文 secret 放进摘要输入。
    """

    if isinstance(value, (InvestigationScopeCapability, InvestigationScopeCredentialAudit)):
        source: Mapping[str, Any] = {
            "task_id": value.task_id,
            "batch_id": value.batch_id,
            "combination_id": value.combination_id,
            "scope_id": value.scope_id,
            "workspace": value.workspace,
            "token_sha256": value.token_sha256,
            "correction_event_ref": value.correction_event_ref,
            "waiting_resume_event_ref": value.waiting_resume_event_ref,
            "previous_generation_hash": value.previous_generation_hash,
        }
    elif isinstance(value, Mapping):
        source = value
    else:
        raise InvestigationScopeCredentialError(
            "scope credential generation input is invalid"
        )
    try:
        identity = _canonical_identity(
            task_id=source.get("task_id"),
            batch_id=source.get("batch_id"),
            combination_id=source.get("combination_id"),
            scope_id=source.get("scope_id"),
            workspace=source.get("workspace"),
        )
        token_sha256 = _validate_generation_hash(source.get("token_sha256"))
        correction_event_ref = source.get("correction_event_ref")
        if correction_event_ref is not None:
            correction_event_ref = _validate_correction_event_ref(correction_event_ref)
        waiting_resume_event_ref = source.get("waiting_resume_event_ref")
        if waiting_resume_event_ref is not None:
            waiting_resume_event_ref = _validate_waiting_resume_event_ref(
                waiting_resume_event_ref
            )
        previous_generation_hash = source.get("previous_generation_hash")
        if previous_generation_hash is not None:
            previous_generation_hash = _validate_generation_hash(previous_generation_hash)
        if correction_event_ref is not None and waiting_resume_event_ref is not None:
            raise ValueError("authorization generation semantics cannot be mixed")
        if (waiting_resume_event_ref is None) != (previous_generation_hash is None):
            raise ValueError("waiting resume generation is incomplete")
    except (TypeError, ValueError, InvestigationScopeCredentialError):
        raise InvestigationScopeCredentialError(
            "scope credential generation input is invalid"
        ) from None
    payload = {
        "identity": identity.as_record_fields(),
        "token_sha256": token_sha256,
        "correction_event_ref": correction_event_ref,
        "waiting_resume_event_ref": waiting_resume_event_ref,
        "previous_generation_hash": previous_generation_hash,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@runtime_checkable
class ScopeCapabilityRegistryProtocol(Protocol):
    """ARCHE 需要的最小 MiniChem registry 协议。"""

    def provision_scope_capability(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: Path,
        token: str,
    ) -> Any:
        """用显式 token 登记一个 MiniChem scope capability。"""

    def revoke_scope_capability(self, token: str) -> Any:
        """撤销一个 MiniChem scope capability。"""

    def reauthorize_scope_capability(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: Path,
        token: str,
        correction_event_ref: str,
    ) -> Any:
        """使用唯一更正事件引用为已撤销 scope 建立新 capability。"""

    def reauthorize_scope_capability_for_waiting_resume(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: Path,
        token: str,
        waiting_resume_event_ref: str,
        previous_generation_hash: str,
    ) -> Any:
        """使用 WM waiting 授权事件和前代 token 摘要恢复一个已撤销 scope。"""


ProvisionScopeCapabilityCallable = Callable[..., Any]
RevokeScopeCapabilityCallable = Callable[[str], Any]
TokenFactory = Callable[[], str]


_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _process_lock_for(path: Path) -> threading.RLock:
    """为同一进程线程补充文件锁，避免测试和嵌套调用中的线程竞争。"""

    key = str(path)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _ensure_private_directory(path: Path) -> None:
    """创建或校验凭据目录，目录必须是 0700 的普通目录。"""

    _reject_symlinked_path_components(path)
    try:
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise InvestigationScopeCredentialCorruption("credential directory is invalid")
        else:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink() or not path.is_dir():
            raise InvestigationScopeCredentialCorruption("credential directory is invalid")
        mode = stat.S_IMODE(path.stat().st_mode)
    except InvestigationScopeCredentialError:
        raise
    except OSError:
        raise InvestigationScopeCredentialCorruption("credential directory is unavailable") from None
    if mode != 0o700:
        raise InvestigationScopeCredentialCorruption("credential directory permissions are invalid")


def _validate_private_file(path: Path, *, label: str) -> None:
    """校验记录或锁文件为 0600 的普通文件，拒绝符号链接。"""

    try:
        if path.is_symlink() or not path.is_file():
            raise InvestigationScopeCredentialCorruption(f"{label} is invalid")
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise InvestigationScopeCredentialCorruption(f"{label} permissions are invalid")
    except InvestigationScopeCredentialError:
        raise
    except OSError:
        raise InvestigationScopeCredentialCorruption(f"{label} is unavailable") from None


def _record_name(identity: InvestigationScopeIdentity) -> str:
    """按完整身份派生首代稳定记录文件名，文件名不含 token。"""

    return f"{_RECORD_PREFIX}{identity.record_id()}{_RECORD_SUFFIX}"


def _validate_correction_event_ref(value: Any) -> str:
    """校验用于授权新代 capability 的不可变更正事件引用。"""

    if not isinstance(value, str):
        raise ValueError("correction_event_ref is invalid")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or any(ord(char) < 0x21 or ord(char) > 0x7E for char in normalized)
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError("correction_event_ref is invalid")
    return normalized


def _validate_waiting_resume_event_ref(value: Any) -> str:
    """校验独立 waiting resume WM 授权事件引用。"""

    if not isinstance(value, str):
        raise ValueError("waiting_resume_event_ref is invalid")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or any(ord(char) < 0x21 or ord(char) > 0x7E for char in normalized)
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError("waiting_resume_event_ref is invalid")
    return normalized


def _validate_generation_hash(value: Any) -> str:
    """校验前一代 capability token 的 SHA-256 摘要。"""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("previous_generation_hash is invalid")
    return value


def _record_id(
    identity: InvestigationScopeIdentity,
    correction_event_ref: str | None = None,
    *,
    waiting_resume_event_ref: str | None = None,
    previous_generation_hash: str | None = None,
) -> str:
    """返回一代 capability 记录的稳定审计标识。"""

    if correction_event_ref is not None and (
        waiting_resume_event_ref is not None or previous_generation_hash is not None
    ):
        raise ValueError("authorization generation semantics cannot be mixed")
    if (waiting_resume_event_ref is None) != (previous_generation_hash is None):
        raise ValueError("waiting resume generation is incomplete")
    if correction_event_ref is None and waiting_resume_event_ref is None:
        return identity.record_id()
    payload: dict[str, Any] = {"identity": identity.as_record_fields()}
    if correction_event_ref is not None:
        payload["correction_event_ref"] = correction_event_ref
    else:
        payload["waiting_resume_event_ref"] = waiting_resume_event_ref
        payload["previous_generation_hash"] = previous_generation_hash
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _reauthorization_record_name(
    identity: InvestigationScopeIdentity,
    correction_event_ref: str | None = None,
    *,
    waiting_resume_event_ref: str | None = None,
    previous_generation_hash: str | None = None,
) -> str:
    """按 scope 身份和独立授权引用派生追加代记录文件名。"""

    if correction_event_ref is not None:
        event_digest = hashlib.sha256(correction_event_ref.encode("utf-8")).hexdigest()
        suffix = f"reauth_{event_digest}"
    else:
        if waiting_resume_event_ref is None or previous_generation_hash is None:
            raise ValueError("waiting resume generation is incomplete")
        event_digest = hashlib.sha256(
            waiting_resume_event_ref.encode("utf-8")
        ).hexdigest()
        suffix = f"waiting_resume_{event_digest}_{previous_generation_hash}"
    return f"{_RECORD_PREFIX}{identity.record_id()}_{suffix}{_RECORD_SUFFIX}"


def _validate_record_mapping_fields(raw: Any) -> None:
    """校验凭据记录的 JSON 类型、字段集合、版本和身份字段形状。"""

    if not isinstance(raw, dict):
        raise InvestigationScopeCredentialCorruption("credential record is invalid")
    if not _REQUIRED_RECORD_FIELDS.issubset(raw) or set(raw) - (
        _REQUIRED_RECORD_FIELDS | _OPTIONAL_RECORD_FIELDS
    ):
        raise InvestigationScopeCredentialCorruption("credential record fields are invalid")
    schema_version = raw.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise InvestigationScopeCredentialCorruption("credential record schema is invalid")
    if schema_version == 1 and set(raw) & _OPTIONAL_RECORD_FIELDS:
        raise InvestigationScopeCredentialCorruption(
            "schema v1 credential record cannot contain authorization metadata"
        )
    if schema_version == 2:
        if "correction_event_ref" not in raw:
            raise InvestigationScopeCredentialCorruption(
                "schema v2 credential record must contain correction event ref"
            )
        correction_event_ref = raw["correction_event_ref"]
        if correction_event_ref is not None:
            try:
                _validate_correction_event_ref(correction_event_ref)
            except ValueError:
                raise InvestigationScopeCredentialCorruption(
                    "credential correction event ref is invalid"
                ) from None
    waiting_event_ref = raw.get("waiting_resume_event_ref")
    previous_hash = raw.get("previous_generation_hash")
    if (waiting_event_ref is None) != (previous_hash is None):
        raise InvestigationScopeCredentialCorruption(
            "credential waiting resume metadata is incomplete"
        )
    if raw.get("correction_event_ref") is not None and waiting_event_ref is not None:
        raise InvestigationScopeCredentialCorruption(
            "credential correction and waiting metadata cannot be mixed"
        )
    if waiting_event_ref is not None:
        try:
            _validate_waiting_resume_event_ref(waiting_event_ref)
            _validate_generation_hash(previous_hash)
        except ValueError:
            raise InvestigationScopeCredentialCorruption(
                "credential waiting resume metadata is invalid"
            ) from None
    for field_name in ("task_id", "batch_id", "combination_id", "scope_id", "workspace"):
        if not isinstance(raw.get(field_name), str):
            raise InvestigationScopeCredentialCorruption("credential identity is invalid")


def _parse_record_identity(path: Path, raw: Mapping[str, Any]) -> InvestigationScopeIdentity:
    """解析记录身份并核对 record_id 与文件名的绑定关系。"""

    try:
        identity = _canonical_identity(
            task_id=raw["task_id"],
            batch_id=raw["batch_id"],
            combination_id=raw["combination_id"],
            scope_id=raw["scope_id"],
            workspace=raw["workspace"],
        )
    except InvestigationScopeCredentialError:
        raise InvestigationScopeCredentialCorruption("credential identity is invalid") from None
    raw_event_ref = raw.get("correction_event_ref")
    if raw_event_ref is not None:
        try:
            raw_event_ref = _validate_correction_event_ref(raw_event_ref)
        except ValueError:
            raise InvestigationScopeCredentialCorruption(
                "credential correction event ref is invalid"
            ) from None
    waiting_event_ref = raw.get("waiting_resume_event_ref")
    previous_hash = raw.get("previous_generation_hash")
    if waiting_event_ref is not None:
        try:
            waiting_event_ref = _validate_waiting_resume_event_ref(waiting_event_ref)
            previous_hash = _validate_generation_hash(previous_hash)
        except ValueError:
            raise InvestigationScopeCredentialCorruption(
                "credential waiting resume metadata is invalid"
            ) from None
    if raw.get("record_id") != _record_id(
        identity,
        raw_event_ref,
        waiting_resume_event_ref=waiting_event_ref,
        previous_generation_hash=previous_hash,
    ):
        raise InvestigationScopeCredentialCorruption("credential record identity is invalid")
    if raw_event_ref is not None:
        expected_name = _reauthorization_record_name(identity, raw_event_ref)
    elif waiting_event_ref is not None:
        expected_name = _reauthorization_record_name(
            identity,
            waiting_resume_event_ref=waiting_event_ref,
            previous_generation_hash=previous_hash,
        )
    else:
        expected_name = _record_name(identity)
    if path.name != expected_name:
        raise InvestigationScopeCredentialCorruption("credential record path is invalid")
    return identity


def _validate_record_token_digest(raw: Mapping[str, Any]) -> None:
    """校验记录 token 与 token_sha256 的完整性绑定，不返回明文 token。"""

    try:
        token = _validate_token(raw.get("token"))
    except (TypeError, ValueError):
        raise InvestigationScopeCredentialCorruption("credential token is invalid") from None
    stored_hash = raw.get("token_sha256")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise InvestigationScopeCredentialCorruption("credential token digest is invalid")
    if not hmac.compare_digest(stored_hash, _token_sha256(token)):
        raise InvestigationScopeCredentialCorruption("credential token digest is invalid")


def _validate_record_status_fields(raw: Mapping[str, Any]) -> None:
    """校验 revoked、status 和 registry_state 的互相约束。"""

    if not isinstance(raw.get("revoked"), bool):
        raise InvestigationScopeCredentialCorruption("credential revocation state is invalid")
    if raw.get("status") not in {"active", "revoked"}:
        raise InvestigationScopeCredentialCorruption("credential status is invalid")
    if bool(raw["revoked"]) != (raw["status"] == "revoked"):
        raise InvestigationScopeCredentialCorruption("credential status is inconsistent")
    registry_state = raw.get("registry_state")
    if registry_state not in _REGISTRY_STATES:
        raise InvestigationScopeCredentialCorruption("credential registry state is invalid")
    if raw["revoked"] and registry_state != "revoked":
        raise InvestigationScopeCredentialCorruption("credential registry state is inconsistent")
    if not raw["revoked"] and registry_state == "revoked":
        raise InvestigationScopeCredentialCorruption("credential registry state is inconsistent")


def _validate_record_timestamp_fields(raw: Mapping[str, Any]) -> None:
    """校验创建、更新时间及可选撤销时间的状态一致性。"""

    for field_name in ("created_at", "updated_at"):
        if not isinstance(raw.get(field_name), str) or not raw[field_name].strip():
            raise InvestigationScopeCredentialCorruption("credential timestamps are invalid")
    revoked_at = raw.get("revoked_at")
    if raw["revoked"]:
        if not isinstance(revoked_at, str) or not revoked_at.strip():
            raise InvestigationScopeCredentialCorruption("credential revocation timestamp is invalid")
    elif revoked_at is not None:
        raise InvestigationScopeCredentialCorruption("credential revocation timestamp is invalid")


def _parse_record(path: Path, raw: Any) -> dict[str, Any]:
    """严格解析一条记录并校验身份、token 摘要和状态。"""

    _validate_record_mapping_fields(raw)
    _parse_record_identity(path, raw)
    _validate_record_token_digest(raw)
    _validate_record_status_fields(raw)
    _validate_record_timestamp_fields(raw)
    # 返回复制值，避免调用方修改 json.loads 的原始映射。
    return dict(raw)


def _read_record(path: Path) -> dict[str, Any]:
    """读取一条私有记录，任何解析或权限异常都 fail-closed。"""

    _validate_private_file(path, label="credential record")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise InvestigationScopeCredentialCorruption("credential record cannot be read") from None
    return _parse_record(path, raw)


def _resolve_revoke_identity_hints(
    capability: InvestigationScopeCapability | str | None,
    *,
    task_id: str | None,
    batch_id: str | None,
    combination_id: str | None,
    scope_id: str | None,
    workspace: str | os.PathLike[str] | Path | None,
) -> tuple[str | None, InvestigationScopeIdentity | None]:
    """解析 revoke 输入中的 token 提示和可选完整身份，并核对二者一致。"""

    token_hint: str | None = None
    requested_identity: InvestigationScopeIdentity | None = None
    explicit_values = (task_id, batch_id, combination_id, scope_id, workspace)
    if isinstance(capability, InvestigationScopeCapability):
        token_hint = capability.token
        requested_identity = capability.identity
        if any(value is not None for value in explicit_values):
            try:
                explicit = _canonical_identity(
                    task_id=task_id,
                    batch_id=batch_id,
                    combination_id=combination_id,
                    scope_id=scope_id,
                    workspace=workspace,
                )
            except InvestigationScopeCredentialError:
                raise InvestigationScopeCredentialConflict("revoke identity is invalid") from None
            if explicit != requested_identity:
                raise InvestigationScopeCredentialConflict("revoke identity conflicts with capability")
    elif isinstance(capability, str):
        token_hint = capability
    elif capability is not None:
        raise InvestigationScopeCredentialError("revoke capability input is invalid")
    return token_hint, requested_identity


def _resolve_revoke_match_criteria(
    token_hint: str | None,
    requested_identity: InvestigationScopeIdentity | None,
    *,
    task_id: str | None,
    batch_id: str | None,
    combination_id: str | None,
    scope_id: str | None,
    workspace: str | os.PathLike[str] | Path | None,
) -> tuple[str | None, InvestigationScopeIdentity | None, str | None]:
    """校验 token 或完整身份条件并返回可比对的 token 摘要。"""

    if token_hint is not None:
        try:
            normalized_token = _validate_token(token_hint)
        except (TypeError, ValueError):
            raise InvestigationScopeCredentialError("revoke capability input is invalid") from None
        return normalized_token, requested_identity, _token_sha256(normalized_token)
    if any(
        value is None
        for value in (task_id, batch_id, combination_id, scope_id, workspace)
    ):
        raise InvestigationScopeCredentialError("revoke requires a complete scope identity")
    identity = _canonical_identity(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
    )
    return None, identity, None


def _match_revoke_records(
    records: list[tuple[Path, dict[str, Any]]],
    requested_identity: InvestigationScopeIdentity | None,
    token_digest: str | None,
    identity_reader: Callable[[Mapping[str, Any]], InvestigationScopeIdentity],
) -> list[tuple[Path, dict[str, Any]]]:
    """按完整身份和 token 摘要筛选唯一撤销候选记录。"""

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path, record in records:
        identity = identity_reader(record)
        if requested_identity is not None and identity != requested_identity:
            continue
        if token_digest is not None and not hmac.compare_digest(
            record["token_sha256"], token_digest
        ):
            continue
        matches.append((path, record))
    return matches


def _validate_revoke_token_match(
    record: Mapping[str, Any],
    token_hint: str | None,
    token_digest: str | None,
) -> None:
    """再次核对 token 提示与选中记录，避免摘要条件被错误复用。"""

    if token_hint is not None and not hmac.compare_digest(
        record["token_sha256"], token_digest or ""
    ):
        raise InvestigationScopeCredentialConflict("scope capability token does not match record")


class InvestigationScopeCredentialStore:
    """按调查 scope 持久化 capability，并通过窄协议调用 MiniChem registry。"""

    schema_version = _SCHEMA_VERSION

    def __init__(
        self,
        credential_root: str | os.PathLike[str] | Path,
        *,
        registry: ScopeCapabilityRegistryProtocol | Any | None = None,
        provision_callable: ProvisionScopeCapabilityCallable | None = None,
        revoke_callable: RevokeScopeCapabilityCallable | None = None,
        token_factory: TokenFactory | None = None,
    ) -> None:
        try:
            root = Path(credential_root).expanduser()
        except (TypeError, ValueError):
            raise InvestigationScopeCredentialError("credential root is invalid") from None
        _reject_symlinked_path_components(root)
        self.root = root.resolve()
        _ensure_private_directory(self.root)
        self.lock_path = self.root / _LOCK_NAME
        if self.lock_path.exists() and self.lock_path.is_symlink():
            raise InvestigationScopeCredentialCorruption("credential lock is invalid")
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
        except OSError:
            raise InvestigationScopeCredentialCorruption("credential lock is unavailable") from None
        _validate_private_file(self.lock_path, label="credential lock")
        # callable 可以分别覆盖 registry 的 provision/revoke 方法，便于测试在
        # registry 调用成功前后注入故障；生产调用方只需传入同一个窄协议对象。
        if provision_callable is not None and not callable(provision_callable):
            raise InvestigationScopeCredentialError("provision_callable is invalid")
        if revoke_callable is not None and not callable(revoke_callable):
            raise InvestigationScopeCredentialError("revoke_callable is invalid")
        self._registry = registry
        self._provision_callable = provision_callable
        self._revoke_callable = revoke_callable
        selected_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        if not callable(selected_factory):
            raise InvestigationScopeCredentialError("token_factory is invalid")
        self._token_factory = selected_factory

    @property
    def credential_root(self) -> Path:
        """返回凭据目录路径。"""

        return self.root

    def credential_record_path(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
    ) -> Path:
        """返回完整身份对应的确定性记录路径，不读取或生成 token。"""

        identity = _canonical_identity(
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            workspace=workspace,
        )
        return self.root / _record_name(identity)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """在进程锁和 POSIX 文件锁内执行一次完整生命周期动作。"""

        _ensure_private_directory(self.root)
        _validate_private_file(self.lock_path, label="credential lock")
        process_lock = _process_lock_for(self.lock_path)
        process_lock.acquire()
        descriptor: int | None = None
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except InvestigationScopeCredentialError:
            raise
        except OSError:
            raise InvestigationScopeCredentialCorruption("credential lock cannot be acquired") from None
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)
            process_lock.release()

    def _iter_records(self) -> list[tuple[Path, dict[str, Any]]]:
        """读取全部凭据记录；任一记录损坏都阻止继续操作。"""

        records: list[tuple[Path, dict[str, Any]]] = []
        try:
            paths = sorted(self.root.glob(f"{_RECORD_PREFIX}*{_RECORD_SUFFIX}"))
        except OSError:
            raise InvestigationScopeCredentialCorruption("credential records cannot be listed") from None
        for path in paths:
            records.append((path, _read_record(path)))
        self._validate_record_set(records)
        return records

    @staticmethod
    def _validate_record_set(records: list[tuple[Path, dict[str, Any]]]) -> None:
        """校验多代记录的身份、授权引用和 token 摘要唯一性。"""

        seen_owner: dict[tuple[str, str, str], InvestigationScopeIdentity] = {}
        seen_scope: dict[str, InvestigationScopeIdentity] = {}
        seen_workspace: dict[Path, InvestigationScopeIdentity] = {}
        seen_token: dict[str, InvestigationScopeIdentity] = {}
        seen_generation: set[
            tuple[
                InvestigationScopeIdentity,
                str | None,
                str | None,
                str | None,
            ]
        ] = set()
        seen_authorization_refs: dict[str, tuple[InvestigationScopeIdentity, str | None]] = {}
        active_by_identity: set[InvestigationScopeIdentity] = set()
        for _path, record in records:
            identity = InvestigationScopeCredentialStore._identity_from_record(record)
            correction_ref = record.get("correction_event_ref")
            waiting_ref = record.get("waiting_resume_event_ref")
            previous_hash = record.get("previous_generation_hash")
            generation = (identity, correction_ref, waiting_ref, previous_hash)
            if generation in seen_generation:
                raise InvestigationScopeCredentialCorruption(
                    "credential records contain duplicate authorization generations"
                )
            seen_generation.add(generation)
            auth_ref = correction_ref or waiting_ref
            if auth_ref is not None:
                auth_value = (identity, previous_hash if waiting_ref is not None else None)
                previous_auth = seen_authorization_refs.get(auth_ref)
                if previous_auth is not None and previous_auth != auth_value:
                    raise InvestigationScopeCredentialCorruption(
                        "authorization event ref is bound to conflicting generations"
                    )
                seen_authorization_refs[auth_ref] = auth_value
            if not bool(record["revoked"]):
                if identity in active_by_identity:
                    raise InvestigationScopeCredentialCorruption(
                        "credential identity has multiple active generations"
                    )
                active_by_identity.add(identity)
            for mapping, key in (
                (seen_owner, identity.owner_key),
                (seen_scope, identity.scope_id),
                (seen_workspace, identity.workspace),
                (seen_token, record["token_sha256"]),
            ):
                previous = mapping.get(key)
                if previous is not None and previous != identity:
                    raise InvestigationScopeCredentialCorruption(
                        "credential records contain conflicting identities"
                    )
                mapping[key] = identity

    @staticmethod
    def _identity_from_record(record: Mapping[str, Any]) -> InvestigationScopeIdentity:
        """从已严格校验的记录构造身份对象。"""

        return InvestigationScopeIdentity(
            task_id=record["task_id"],
            batch_id=record["batch_id"],
            combination_id=record["combination_id"],
            scope_id=record["scope_id"],
            workspace=Path(record["workspace"]),
        )

    @staticmethod
    def _check_identity_conflicts(
        requested: InvestigationScopeIdentity,
        records: list[tuple[Path, dict[str, Any]]],
        *,
        expected_path: Path,
    ) -> tuple[Path, dict[str, Any]] | None:
        """核对跨会话身份冲突，同时允许同一 scope 的历史 revoked 代次。"""

        matched: tuple[Path, dict[str, Any]] | None = None
        requested_owner = requested.owner_key
        for path, record in records:
            identity = InvestigationScopeCredentialStore._identity_from_record(record)
            if path == expected_path:
                if identity != requested:
                    raise InvestigationScopeCredentialConflict("credential identity conflicts with record path")
                if matched is not None:
                    raise InvestigationScopeCredentialConflict("duplicate credential identity")
                matched = (path, record)
            same_owner = identity.owner_key == requested_owner
            same_scope = identity.scope_id == requested.scope_id
            same_workspace = identity.workspace == requested.workspace
            if same_owner and identity != requested:
                raise InvestigationScopeCredentialConflict("credential session identity is already bound differently")
            if same_scope and identity != requested:
                raise InvestigationScopeCredentialConflict("scope_id is already bound differently")
            if same_workspace and identity != requested:
                raise InvestigationScopeCredentialConflict("workspace is already bound differently")
        return matched

    def _write_record(self, path: Path, record: Mapping[str, Any]) -> None:
        """把一条完整记录以临时文件、fsync 和 replace 原子写入。"""

        payload = json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        temporary: Path | None = None
        descriptor: int | None = None
        try:
            for _attempt in range(8):
                temporary = self.root / (
                    f".{path.name}.{secrets.token_hex(8)}{_TEMP_MARKER}"
                )
                try:
                    descriptor = os.open(
                        temporary,
                        os.O_CREAT
                        | os.O_EXCL
                        | os.O_WRONLY
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    break
                except FileExistsError:
                    descriptor = None
                    temporary = None
            if descriptor is None or temporary is None:
                raise OSError("temporary credential file collision")
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            temporary = None
            directory_descriptor = os.open(
                self.root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            raise InvestigationScopeCredentialCorruption("credential record cannot be saved") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def _build_new_record(
        self,
        identity: InvestigationScopeIdentity,
        *,
        correction_event_ref: str | None = None,
        waiting_resume_event_ref: str | None = None,
        previous_generation_hash: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """生成一次 token 并构造尚未调用 registry 的 pending 记录。"""

        try:
            token = _validate_token(self._token_factory())
        except Exception:
            raise InvestigationScopeCredentialError("token generation failed") from None
        now = _utc_now()
        record = {
            "schema_version": _SCHEMA_VERSION,
            "record_id": _record_id(
                identity,
                correction_event_ref,
                waiting_resume_event_ref=waiting_resume_event_ref,
                previous_generation_hash=previous_generation_hash,
            ),
            **identity.as_record_fields(),
            "token": token,
            "token_sha256": _token_sha256(token),
            "revoked": False,
            "status": "active",
            "registry_state": "pending",
            "created_at": now,
            "updated_at": now,
            "revoked_at": None,
            "correction_event_ref": None,
        }
        if correction_event_ref is not None:
            record["correction_event_ref"] = correction_event_ref
        if waiting_resume_event_ref is not None:
            record["waiting_resume_event_ref"] = waiting_resume_event_ref
            record["previous_generation_hash"] = previous_generation_hash
        return record, token

    def _resolve_provision_callable(self) -> ProvisionScopeCapabilityCallable:
        """解析注入的 provision callable 或 registry 的窄方法。"""

        if self._provision_callable is not None:
            return self._provision_callable
        registry = self._registry
        callback = getattr(registry, "provision_scope_capability", None)
        if callable(callback):
            return callback
        if callable(registry):
            return registry
        raise InvestigationScopeCredentialProvisionError("scope registry provision is unavailable")

    def _resolve_revoke_callable(self) -> RevokeScopeCapabilityCallable:
        """解析注入的 revoke callable 或 registry 的窄方法。"""

        if self._revoke_callable is not None:
            return self._revoke_callable
        callback = getattr(self._registry, "revoke_scope_capability", None)
        if callable(callback):
            return callback
        raise InvestigationScopeCredentialRevokeError("scope registry revoke is unavailable")

    def _resolve_reauthorize_callable(self) -> ProvisionScopeCapabilityCallable:
        """解析注入的更正授权 callable 或 registry 的窄方法。"""

        callback = getattr(self._registry, "reauthorize_scope_capability", None)
        if callable(callback):
            return callback
        raise InvestigationScopeCredentialReauthorizationError(
            "scope registry reauthorization is unavailable"
        )

    def _resolve_waiting_resume_callable(self) -> ProvisionScopeCapabilityCallable:
        """解析独立 waiting resume registry 方法，不回退到 correction 协议。"""

        callback = getattr(
            self._registry,
            "reauthorize_scope_capability_for_waiting_resume",
            None,
        )
        if callable(callback):
            return callback
        raise InvestigationScopeCredentialReauthorizationError(
            "MiniChem waiting resume reauthorization is unavailable"
        )

    @staticmethod
    def _invoke_provision(
        callback: ProvisionScopeCapabilityCallable,
        identity: InvestigationScopeIdentity,
        token: str,
    ) -> Any:
        """用稳定关键字协议调用 registry，不把返回对象写入本地审计。"""

        try:
            return callback(
                task_id=identity.task_id,
                batch_id=identity.batch_id,
                combination_id=identity.combination_id,
                scope_id=identity.scope_id,
                workspace=identity.workspace,
                token=token,
            )
        except Exception:
            # 不保留原异常链；测试替身或远端错误可能把 token 放入异常文本。
            raise InvestigationScopeCredentialProvisionError(
                "MiniChem scope capability provision failed"
            ) from None

    @staticmethod
    def _invoke_revoke(callback: RevokeScopeCapabilityCallable, token: str) -> Any:
        """调用 registry 撤销 token；错误文本和异常链均不携带 token。"""

        try:
            return callback(token)
        except Exception:
            raise InvestigationScopeCredentialRevokeError(
                "MiniChem scope capability revoke failed"
            ) from None

    @staticmethod
    def _invoke_reauthorize(
        callback: ProvisionScopeCapabilityCallable,
        identity: InvestigationScopeIdentity,
        token: str,
        correction_event_ref: str,
    ) -> Any:
        """调用 registry 的窄更正授权协议，不把 token 写入错误。"""

        try:
            return callback(
                task_id=identity.task_id,
                batch_id=identity.batch_id,
                combination_id=identity.combination_id,
                scope_id=identity.scope_id,
                workspace=identity.workspace,
                token=token,
                correction_event_ref=correction_event_ref,
            )
        except Exception:
            raise InvestigationScopeCredentialReauthorizationError(
                "MiniChem scope capability reauthorization failed"
            ) from None

    @staticmethod
    def _invoke_waiting_resume(
        callback: ProvisionScopeCapabilityCallable,
        identity: InvestigationScopeIdentity,
        token: str,
        waiting_resume_event_ref: str,
        previous_generation_hash: str,
    ) -> Any:
        """调用独立 waiting resume registry 协议，不把 token 写入错误。"""

        try:
            return callback(
                task_id=identity.task_id,
                batch_id=identity.batch_id,
                combination_id=identity.combination_id,
                scope_id=identity.scope_id,
                workspace=identity.workspace,
                token=token,
                waiting_resume_event_ref=waiting_resume_event_ref,
                previous_generation_hash=previous_generation_hash,
            )
        except Exception:
            raise InvestigationScopeCredentialReauthorizationError(
                "MiniChem waiting resume reauthorization failed"
            ) from None

    @staticmethod
    def _capability_from_record(
        record: Mapping[str, Any],
        path: Path,
    ) -> InvestigationScopeCapability:
        """从已校验记录构造明文只在返回对象中出现一次的 capability。"""

        return InvestigationScopeCapability(
            token=record["token"],
            task_id=record["task_id"],
            batch_id=record["batch_id"],
            combination_id=record["combination_id"],
            scope_id=record["scope_id"],
            workspace=Path(record["workspace"]),
            token_sha256=record["token_sha256"],
            record_path=path,
            correction_event_ref=record.get("correction_event_ref"),
            waiting_resume_event_ref=record.get("waiting_resume_event_ref"),
            previous_generation_hash=record.get("previous_generation_hash"),
        )

    def provision_investigation_scope_capability(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
    ) -> InvestigationScopeCapability:
        """为调查 scope 原子保存并注册 capability，失败时保留 token 供恢复重放。"""

        identity = _canonical_identity(
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            workspace=workspace,
        )
        expected_path = self.root / _record_name(identity)
        with self._locked():
            records = self._iter_records()
            self._check_identity_conflicts(
                identity,
                records,
                expected_path=expected_path,
            )
            same_identity = [
                (path, record)
                for path, record in records
                if self._identity_from_record(record) == identity
            ]
            active_records = [
                item for item in same_identity if not bool(item[1]["revoked"])
            ]
            if len(active_records) > 1:
                raise InvestigationScopeCredentialConflict(
                    "scope identity has multiple active capability records"
                )
            if active_records:
                record_path, record = active_records[0]
                if record.get("correction_event_ref") is not None and record["registry_state"] == "pending":
                    raise InvestigationScopeCredentialConflict(
                        "pending scope reauthorization requires its correction event"
                    )
                token = record["token"]
                callback = self._resolve_provision_callable()
                self._invoke_provision(callback, identity, token)
                if record["registry_state"] != "registered":
                    record["registry_state"] = "registered"
                    record["updated_at"] = _utc_now()
                    self._write_record(record_path, record)
                return self._capability_from_record(record, record_path)
            if same_identity:
                # 同一 identity 的所有历史代次均已 revoked；普通 provision 不得绕过
                # 更正事件授权重新 mint capability。
                raise InvestigationScopeCredentialConflict(
                    "revoked scope capability cannot be reprovisioned"
                )
            record, token = self._build_new_record(identity)
            if any(
                hmac.compare_digest(existing["token_sha256"], record["token_sha256"])
                for _path, existing in records
            ) or expected_path.exists():
                raise InvestigationScopeCredentialConflict(
                    "capability token digest or record path is already bound"
                )
            self._write_record(expected_path, record)
            callback = self._resolve_provision_callable()
            self._invoke_provision(callback, identity, token)
            if record["registry_state"] != "registered":
                record["registry_state"] = "registered"
                record["updated_at"] = _utc_now()
                self._write_record(expected_path, record)
        return self._capability_from_record(record, expected_path)

    def reuse_registered_investigation_scope_capability(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
        expected_generation_digest: str,
    ) -> InvestigationScopeCapability:
        """只读取并复用一个已注册且代次摘要精确匹配的 capability。

        输入是完整 scope identity、物理工作区和显式恢复事件冻结的代次摘要；动作
        在本地锁内读取全部凭据记录，要求目标 identity 只有一个未撤销且已注册代次，
        并逐值核对 generation digest。返回该代 capability 供 readiness 和查询已有
        作业使用；不生成 token、不写记录，也不调用 registry 的 provision 或
        reauthorize 接口。记录缺失、重复、pending、撤销或摘要漂移时停止恢复。
        """

        identity = _canonical_identity(
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            workspace=workspace,
        )
        try:
            expected_generation_digest = _validate_generation_hash(
                expected_generation_digest
            )
        except (TypeError, ValueError):
            raise InvestigationScopeCredentialError(
                "expected scope generation digest is invalid"
            ) from None
        with self._locked():
            records = self._iter_records()
            self._check_identity_conflicts(
                identity,
                records,
                expected_path=self.root / _record_name(identity),
            )
            active_records = [
                (path, record)
                for path, record in records
                if self._identity_from_record(record) == identity
                and not bool(record["revoked"])
            ]
            if len(active_records) != 1:
                raise InvestigationScopeCredentialConflict(
                    "scope identity must have one active capability record"
                )
            record_path, record = active_records[0]
            if record.get("registry_state") != "registered":
                raise InvestigationScopeCredentialConflict(
                    "scope capability is not registered"
                )
            audit_input = {
                "task_id": record["task_id"],
                "batch_id": record["batch_id"],
                "combination_id": record["combination_id"],
                "scope_id": record["scope_id"],
                "workspace": record["workspace"],
                "token_sha256": record["token_sha256"],
                "correction_event_ref": record.get("correction_event_ref"),
                "waiting_resume_event_ref": record.get(
                    "waiting_resume_event_ref"
                ),
                "previous_generation_hash": record.get(
                    "previous_generation_hash"
                ),
            }
            if (
                scope_credential_generation_digest(audit_input)
                != expected_generation_digest
            ):
                raise InvestigationScopeCredentialConflict(
                    "scope capability generation digest does not match authorization"
                )
            return self._capability_from_record(record, record_path)

    def reauthorize_investigation_scope_capability(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
        correction_event_ref: str | None = None,
    ) -> InvestigationScopeCapability:
        """按唯一更正事件为已撤销 scope 追加一代 capability。

        输入必须是与已有 scope 完全相同的 task、batch、combination、scope 和工作区，
        且必须携带唯一更正事件引用；动作先把新 token 以 pending 状态原子写入独立
        记录，再调用 MiniChem registry 的窄 reauthorize 方法，成功后把该记录标为
        registered。旧记录和 revoked tombstone 永不覆盖。相同事件引用会复用同一
        pending/registered 记录并重放远端调用；事件引用、身份、当前活动代次或记录
        完整性冲突时抛出错误，失败时保留 pending 记录供同一更正恢复。
        """

        try:
            event_ref = _validate_correction_event_ref(correction_event_ref)
        except (TypeError, ValueError):
            raise InvestigationScopeCredentialError(
                "correction event ref is invalid"
            ) from None
        identity = _canonical_identity(
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            workspace=workspace,
        )
        expected_path = self.root / _reauthorization_record_name(identity, event_ref)
        with self._locked():
            records = self._iter_records()
            self._check_identity_conflicts(
                identity,
                records,
                expected_path=expected_path,
            )
            same_identity = [
                (path, record)
                for path, record in records
                if self._identity_from_record(record) == identity
            ]
            same_ref = [
                (path, record)
                for path, record in records
                if record.get("correction_event_ref") == event_ref
            ]
            if same_ref:
                if len(same_ref) != 1:
                    raise InvestigationScopeCredentialCorruption(
                        "reauthorization event ref has duplicate records"
                    )
                record_path, record = same_ref[0]
                if self._identity_from_record(record) != identity:
                    raise InvestigationScopeCredentialConflict(
                        "reauthorization event ref is bound to another identity"
                    )
                if bool(record["revoked"]):
                    raise InvestigationScopeCredentialConflict(
                        "revoked reauthorization record cannot be reused"
                    )
                token = record["token"]
            else:
                active_records = [
                    item for item in same_identity if not bool(item[1]["revoked"])
                ]
                if active_records:
                    raise InvestigationScopeCredentialConflict(
                        "scope identity still has an active capability"
                    )
                if not same_identity or not all(bool(item[1]["revoked"]) for item in same_identity):
                    raise InvestigationScopeCredentialConflict(
                        "reauthorization requires a revoked scope tombstone"
                    )
                record, token = self._build_new_record(
                    identity,
                    correction_event_ref=event_ref,
                )
                if any(
                    hmac.compare_digest(existing["token_sha256"], record["token_sha256"])
                    for _path, existing in records
                ):
                    raise InvestigationScopeCredentialConflict(
                        "capability token digest is already bound"
                    )
                self._write_record(expected_path, record)
                record_path = expected_path
            callback = self._resolve_reauthorize_callable()
            self._invoke_reauthorize(callback, identity, token, event_ref)
            if record["registry_state"] != "registered":
                record["registry_state"] = "registered"
                record["updated_at"] = _utc_now()
                self._write_record(record_path, record)
        return self._capability_from_record(record, record_path)

    def reauthorize_investigation_scope_capability_for_waiting_resume(
        self,
        *,
        task_id: str,
        batch_id: str,
        combination_id: str,
        scope_id: str,
        workspace: str | os.PathLike[str] | Path,
        waiting_resume_event_ref: str | None = None,
        previous_generation_hash: str | None = None,
    ) -> InvestigationScopeCapability:
        """按 WM waiting 事件只追加一次旧 scope 的严格恢复授权。

        输入必须包含完整 scope 身份、独立 waiting resume 事件引用和前一代 token
        摘要；动作在本地锁内先核对 active/revoked 代次，再以 pending 记录原子保存
        新 token，调用独立 MiniChem waiting resume registry 方法，成功后标记
        registered。active capability 只有在摘要匹配时复用；无 active 时必须恰好
        命中一个 revoked 前代，且一个 waiting 事件不能绑定第二个身份或第二个代次。
        事件缺失、多个 revoked 但摘要不唯一、终态路由未授权、身份/摘要篡改或远端
        失败均 fail-closed，旧记录不覆盖，pending 记录保留供同一事件重试。
        """

        try:
            event_ref = _validate_waiting_resume_event_ref(waiting_resume_event_ref)
            previous_hash = _validate_generation_hash(previous_generation_hash)
        except (TypeError, ValueError):
            raise InvestigationScopeCredentialError(
                "waiting resume authorization metadata is invalid"
            ) from None
        identity = _canonical_identity(
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            workspace=workspace,
        )
        expected_path = self.root / _reauthorization_record_name(
            identity,
            waiting_resume_event_ref=event_ref,
            previous_generation_hash=previous_hash,
        )
        with self._locked():
            records = self._iter_records()
            self._check_identity_conflicts(
                identity,
                records,
                expected_path=expected_path,
            )
            same_identity = [
                (path, record)
                for path, record in records
                if self._identity_from_record(record) == identity
            ]
            same_waiting = [
                (path, record)
                for path, record in records
                if record.get("waiting_resume_event_ref") == event_ref
            ]
            correction_collision = [
                (path, record)
                for path, record in records
                if record.get("correction_event_ref") == event_ref
            ]
            if correction_collision:
                raise InvestigationScopeCredentialConflict(
                    "waiting resume event cannot reuse correction event ref"
                )
            if len(same_waiting) > 1:
                raise InvestigationScopeCredentialCorruption(
                    "waiting resume event ref has duplicate records"
                )
            if same_waiting:
                record_path, record = same_waiting[0]
                if self._identity_from_record(record) != identity:
                    raise InvestigationScopeCredentialConflict(
                        "waiting resume event ref is bound to another identity"
                    )
                if record.get("previous_generation_hash") != previous_hash:
                    raise InvestigationScopeCredentialConflict(
                        "waiting resume event ref is bound to another previous generation"
                    )
                if bool(record["revoked"]):
                    raise InvestigationScopeCredentialConflict(
                        "revoked waiting resume record cannot be reused"
                    )
                token = record["token"]
            else:
                active_records = [
                    item for item in same_identity if not bool(item[1]["revoked"])
                ]
                if len(active_records) > 1:
                    raise InvestigationScopeCredentialConflict(
                        "scope identity has multiple active capability records"
                    )
                if active_records:
                    record_path, record = active_records[0]
                    if not hmac.compare_digest(record["token_sha256"], previous_hash):
                        raise InvestigationScopeCredentialConflict(
                            "active scope capability does not match previous generation"
                        )
                    token = record["token"]
                else:
                    if not same_identity or not all(
                        bool(item[1]["revoked"]) for item in same_identity
                    ):
                        raise InvestigationScopeCredentialConflict(
                            "waiting resume requires a revoked scope tombstone"
                        )
                    matching_revoked = [
                        item
                        for item in same_identity
                        if hmac.compare_digest(item[1]["token_sha256"], previous_hash)
                    ]
                    if len(matching_revoked) != 1:
                        raise InvestigationScopeCredentialConflict(
                            "previous generation hash does not identify exactly one revoked capability"
                        )
                    record, token = self._build_new_record(
                        identity,
                        waiting_resume_event_ref=event_ref,
                        previous_generation_hash=previous_hash,
                    )
                    if any(
                        hmac.compare_digest(existing["token_sha256"], record["token_sha256"])
                        for _path, existing in records
                    ) or expected_path.exists():
                        raise InvestigationScopeCredentialConflict(
                            "waiting resume token or record path is already bound"
                        )
                    record_path = expected_path
                    self._write_record(record_path, record)
            callback = self._resolve_waiting_resume_callable()
            self._invoke_waiting_resume(
                callback,
                identity,
                token,
                event_ref,
                previous_hash,
            )
            if record["registry_state"] != "registered":
                record["registry_state"] = "registered"
                record["updated_at"] = _utc_now()
                self._write_record(record_path, record)
        return self._capability_from_record(record, record_path)

    # 兼容恢复协调器早期命名；仍然只调用独立 waiting resume 协议。
    reauthorize_waiting_resume_scope_capability = (
        reauthorize_investigation_scope_capability_for_waiting_resume
    )

    def _find_record_for_revoke(
        self,
        *,
        capability: InvestigationScopeCapability | str | None,
        task_id: str | None,
        batch_id: str | None,
        combination_id: str | None,
        scope_id: str | None,
        workspace: str | os.PathLike[str] | Path | None,
        records: list[tuple[Path, dict[str, Any]]],
    ) -> tuple[Path, dict[str, Any]]:
        """按 capability、token 或完整身份定位一条已严格校验的记录。"""

        token_hint, requested_identity = _resolve_revoke_identity_hints(
            capability,
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            workspace=workspace,
        )
        token_hint, requested_identity, token_digest = _resolve_revoke_match_criteria(
            token_hint,
            requested_identity,
            task_id=task_id,
            batch_id=batch_id,
            combination_id=combination_id,
            scope_id=scope_id,
            workspace=workspace,
        )
        matches = _match_revoke_records(
            records,
            requested_identity,
            token_digest,
            self._identity_from_record,
        )
        if token_hint is None and requested_identity is not None:
            active = [item for item in matches if not bool(item[1]["revoked"])]
            if len(active) > 1:
                raise InvestigationScopeCredentialConflict(
                    "scope identity has multiple active capability records"
                )
            if active:
                path, record = active[0]
            elif matches:
                # 全部 revoked 时选择排序后的最后一代，重复撤销保持确定性，
                # 且不会误选其它 identity 的 tombstone。
                path, record = sorted(matches, key=lambda item: item[0].name)[-1]
            else:
                raise InvestigationScopeCredentialError("scope capability record was not found")
        else:
            if len(matches) != 1:
                raise InvestigationScopeCredentialError("scope capability record was not found")
            path, record = matches[0]
        _validate_revoke_token_match(record, token_hint, token_digest)
        return path, record

    def revoke_investigation_scope_capability(
        self,
        capability: InvestigationScopeCapability | str | None = None,
        *,
        task_id: str | None = None,
        batch_id: str | None = None,
        combination_id: str | None = None,
        scope_id: str | None = None,
        workspace: str | os.PathLike[str] | Path | None = None,
    ) -> None:
        """先撤销服务端 capability，再原子标记本地记录；重复调用保持幂等。"""

        with self._locked():
            records = self._iter_records()
            path, record = self._find_record_for_revoke(
                capability=capability,
                task_id=task_id,
                batch_id=batch_id,
                combination_id=combination_id,
                scope_id=scope_id,
                workspace=workspace,
                records=records,
            )
            if bool(record["revoked"]):
                return
            callback = self._resolve_revoke_callable()
            self._invoke_revoke(callback, record["token"])
            record["revoked"] = True
            record["status"] = "revoked"
            record["registry_state"] = "revoked"
            record["revoked_at"] = _utc_now()
            record["updated_at"] = _utc_now()
            self._write_record(path, record)

    def list_investigation_scope_capability_records(
        self,
    ) -> tuple[InvestigationScopeCredentialAudit, ...]:
        """返回不含 token 的审计记录；任何损坏记录都令读取失败。"""

        with self._locked():
            records = self._iter_records()
        return tuple(
            InvestigationScopeCredentialAudit(
                task_id=record["task_id"],
                batch_id=record["batch_id"],
                combination_id=record["combination_id"],
                scope_id=record["scope_id"],
                workspace=Path(record["workspace"]),
                token_sha256=record["token_sha256"],
                revoked=record["revoked"],
                registry_state=record["registry_state"],
                record_path=path,
                correction_event_ref=record.get("correction_event_ref"),
                waiting_resume_event_ref=record.get("waiting_resume_event_ref"),
                previous_generation_hash=record.get("previous_generation_hash"),
            )
            for path, record in records
        )

# 保留这个带业务对象的类型名，兼容已经采用 manager 命名的调用方。
InvestigationScopeCredentialManager = InvestigationScopeCredentialStore
ProvisionedInvestigationScopeCapability = InvestigationScopeCapability


# 便于不需要长期持有 store 对象的 controller 使用的动作入口。
def provision_investigation_scope_capability(
    credential_root: str | os.PathLike[str] | Path,
    registry: ScopeCapabilityRegistryProtocol | Any | None = None,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    scope_id: str,
    workspace: str | os.PathLike[str] | Path,
    provision_callable: ProvisionScopeCapabilityCallable | None = None,
    revoke_callable: RevokeScopeCapabilityCallable | None = None,
    token_factory: TokenFactory | None = None,
) -> InvestigationScopeCapability:
    """原子保存并注册一个调查 scope capability。"""

    store = InvestigationScopeCredentialStore(
        credential_root,
        registry=registry,
        provision_callable=provision_callable,
        revoke_callable=revoke_callable,
        token_factory=token_factory,
    )
    return store.provision_investigation_scope_capability(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
    )


def revoke_investigation_scope_capability(
    credential_root: str | os.PathLike[str] | Path,
    capability: InvestigationScopeCapability | str | None = None,
    registry: ScopeCapabilityRegistryProtocol | Any | None = None,
    *,
    task_id: str | None = None,
    batch_id: str | None = None,
    combination_id: str | None = None,
    scope_id: str | None = None,
    workspace: str | os.PathLike[str] | Path | None = None,
    provision_callable: ProvisionScopeCapabilityCallable | None = None,
    revoke_callable: RevokeScopeCapabilityCallable | None = None,
) -> None:
    """显式撤销一个调查 scope capability，重复调用保持幂等。"""

    store = InvestigationScopeCredentialStore(
        credential_root,
        registry=registry,
        provision_callable=provision_callable,
        revoke_callable=revoke_callable,
    )
    store.revoke_investigation_scope_capability(
        capability,
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
    )


def reauthorize_investigation_scope_capability(
    credential_root: str | os.PathLike[str] | Path,
    registry: ScopeCapabilityRegistryProtocol | Any | None = None,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    scope_id: str,
    workspace: str | os.PathLike[str] | Path,
    correction_event_ref: str | None = None,
    revoke_callable: RevokeScopeCapabilityCallable | None = None,
    token_factory: TokenFactory | None = None,
) -> InvestigationScopeCapability:
    """使用显式更正事件为已撤销调查 scope 追加一代 capability。"""

    store = InvestigationScopeCredentialStore(
        credential_root,
        registry=registry,
        revoke_callable=revoke_callable,
        token_factory=token_factory,
    )
    return store.reauthorize_investigation_scope_capability(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
        correction_event_ref=correction_event_ref,
    )


def reauthorize_investigation_scope_capability_for_waiting_resume(
    credential_root: str | os.PathLike[str] | Path,
    registry: ScopeCapabilityRegistryProtocol | Any | None = None,
    *,
    task_id: str,
    batch_id: str,
    combination_id: str,
    scope_id: str,
    workspace: str | os.PathLike[str] | Path,
    waiting_resume_event_ref: str | None = None,
    previous_generation_hash: str | None = None,
    revoke_callable: RevokeScopeCapabilityCallable | None = None,
    token_factory: TokenFactory | None = None,
) -> InvestigationScopeCapability:
    """使用独立 WM waiting 事件为旧调查 scope 执行一次严格恢复授权。"""

    store = InvestigationScopeCredentialStore(
        credential_root,
        registry=registry,
        revoke_callable=revoke_callable,
        token_factory=token_factory,
    )
    return store.reauthorize_investigation_scope_capability_for_waiting_resume(
        task_id=task_id,
        batch_id=batch_id,
        combination_id=combination_id,
        scope_id=scope_id,
        workspace=workspace,
        waiting_resume_event_ref=waiting_resume_event_ref,
        previous_generation_hash=previous_generation_hash,
    )


def list_investigation_scope_capability_records(
    credential_root: str | os.PathLike[str] | Path,
) -> tuple[InvestigationScopeCredentialAudit, ...]:
    """读取本地调查 capability 的无 secret 审计视图。"""

    return InvestigationScopeCredentialStore(
        credential_root
    ).list_investigation_scope_capability_records()


__all__ = [
    "InvestigationScopeCapability",
    "InvestigationScopeCredentialAudit",
    "InvestigationScopeCredentialConflict",
    "InvestigationScopeCredentialCorruption",
    "InvestigationScopeCredentialError",
    "InvestigationScopeCredentialProvisionError",
    "InvestigationScopeCredentialReauthorizationError",
    "InvestigationScopeCredentialRevokeError",
    "InvestigationScopeCredentialStore",
    "InvestigationScopeCredentialManager",
    "InvestigationScopeIdentity",
    "ProvisionedInvestigationScopeCapability",
    "ProvisionScopeCapabilityCallable",
    "RevokeScopeCapabilityCallable",
    "ScopeCapabilityConflict",
    "ScopeCapabilityRegistryProtocol",
    "ScopeCredentialConflict",
    "ScopeCredentialCorruption",
    "ScopeCredentialError",
    "TokenFactory",
    "list_investigation_scope_capability_records",
    "provision_investigation_scope_capability",
    "reauthorize_investigation_scope_capability",
    "reauthorize_investigation_scope_capability_for_waiting_resume",
    "revoke_investigation_scope_capability",
    "scope_credential_generation_digest",
]
