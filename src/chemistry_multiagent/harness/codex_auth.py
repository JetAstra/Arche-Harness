"""Codex 认证模式、短命登录凭据和命令权限 profile。

本模块把 API provider 与本机 Codex 登录状态分开。默认 ``api`` 路径只使用
调用方显式提供的 provider 环境；``local_login`` 路径只在一个 app-server
生命周期内把显式 source 复制到当前会话的 CODEX_HOME。复制期间不会修改
source，模型 shell 仍然保留，但通过 Codex 0.149 的自定义权限 profile 拒绝
读取整个虚拟 CODEX_HOME 和登录 source。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import HarnessError
from .session_store import _atomic_write_json_file, _fsync_directory


CODEX_AUTH_MODE_ENV = "ARCHE_CODEX_AUTH_MODE"
CODEX_AUTH_SOURCE_ENV = "ARCHE_CODEX_AUTH_SOURCE"
CODEX_ALLOW_PROVIDER_MIGRATION_ENV = "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION"
DEFAULT_CODEX_AUTH_MODE = "api"
DEFAULT_CODEX_AUTH_SOURCE = Path("/root/.codex/auth.json")
CODEX_AUTH_MODE_API = "api"
CODEX_AUTH_MODE_LOCAL_LOGIN = "local_login"
CODEX_LOCAL_LOGIN_PROVIDER_ID = "openai"
CODEX_LOCAL_LOGIN_MODEL = "gpt-5.6-sol"
CODEX_LOCAL_LOGIN_MODEL_ENV = "ARCHE_CODEX_LOCAL_LOGIN_MODEL"
CODEX_LOCAL_LOGIN_REASONING_EFFORT = "xhigh"
CODEX_LOCAL_LOGIN_REASONING_EFFORT_ENV = "ARCHE_CODEX_LOCAL_LOGIN_REASONING_EFFORT"
CODEX_LOCAL_LOGIN_ENV_KEY = "ARCHE_CODEX_LOCAL_LOGIN"
CODEX_INVESTIGATION_PERMISSION_PROFILE = "arche_investigation_no_codex_home"

_AUTH_CLONE_SCHEMA_VERSION = 1
_AUTH_CLONE_MARKER_NAME = "auth.json.arche-clone.json"
_AUTH_CLONE_FILE_NAME = "auth.json"
_AUTH_CLONE_MARKER_FIELDS = frozenset({"schema_version", "pid", "proc_start_time"})
_AUTH_PROFILE_CONFIG_NAME = "config.toml"
_AUTH_PROFILE_REQUIRED_DENIES = frozenset({
    "/.arche/codex-home",
    "/root/.codex/auth.json",
})


def normalize_codex_auth_mode(value: Any) -> str:
    """规范化显式认证模式；缺省值固定为 API，未知值直接阻断启动。"""

    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_CODEX_AUTH_MODE
    if not isinstance(value, str):
        raise HarnessError("ARCHE_CODEX_AUTH_MODE must be api or local_login", kind="transport")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {CODEX_AUTH_MODE_API, CODEX_AUTH_MODE_LOCAL_LOGIN}:
        raise HarnessError("ARCHE_CODEX_AUTH_MODE must be api or local_login", kind="transport")
    return normalized


def parse_explicit_provider_migration_flag(value: Any) -> bool:
    """读取显式 provider migration 开关；默认禁止自动切换会话。"""

    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise HarnessError(
            "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION must be a boolean",
            kind="session_state",
        )
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise HarnessError(
        "ARCHE_CODEX_ALLOW_PROVIDER_MIGRATION must be a boolean",
        kind="session_state",
    )


def resolve_codex_auth_source(value: Any) -> Path:
    """解析本机登录 source 路径，不读取文件内容或把正文写入状态。"""

    if value is None or (isinstance(value, str) and not value.strip()):
        candidate = DEFAULT_CODEX_AUTH_SOURCE
    elif isinstance(value, (str, os.PathLike)):
        candidate = Path(value).expanduser()
    else:
        raise HarnessError("ARCHE_CODEX_AUTH_SOURCE must be an absolute file", kind="transport")
    if not candidate.is_absolute():
        raise HarnessError("ARCHE_CODEX_AUTH_SOURCE must be an absolute file", kind="transport")
    # resolve(strict=False) 只规范父路径；下面会用 lstat 检查 source 本身，
    # 防止符号链接静默改变凭据来源。
    return candidate.resolve(strict=False)


def _ensure_secure_directory(path: Path, description: str) -> None:
    """创建并收紧 auth clone/profile 目录；链接和普通文件一律拒绝。"""

    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise HarnessError(f"{description} must be a regular directory", kind="session_state")
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise HarnessError(f"{description} cannot be prepared", kind="session_state") from exc


def _read_process_start_token(pid: int) -> str | None:
    """读取 Linux process start token，避免 PID 重用误删活动 clone。"""

    if pid <= 0:
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    close = raw.rfind(b")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    # ``fields[0]`` 对应 stat 第 3 项；第 22 项在这里的索引是 19。
    if len(fields) <= 19:
        return None
    try:
        return fields[19].decode("ascii")
    except UnicodeDecodeError:
        return None


def _read_clone_marker(path: Path) -> dict[str, Any] | None:
    """读取 clone marker 的最小无秘密字段；损坏记录不能被猜测清理。"""

    if path.is_symlink():
        raise HarnessError("Codex auth clone marker must not be a symlink", kind="session_state")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("Codex auth clone marker is not a regular file", kind="session_state")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessError("Codex auth clone marker cannot be read", kind="session_state") from exc
    if not isinstance(value, dict) or set(value) != _AUTH_CLONE_MARKER_FIELDS:
        raise HarnessError("Codex auth clone marker is malformed", kind="session_state")
    if value.get("schema_version") != _AUTH_CLONE_SCHEMA_VERSION:
        raise HarnessError("Codex auth clone marker schema is unsupported", kind="session_state")
    pid = value.get("pid")
    token = value.get("proc_start_time")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise HarnessError("Codex auth clone marker pid is malformed", kind="session_state")
    if not isinstance(token, str) or not token or any(ch not in "0123456789" for ch in token):
        raise HarnessError("Codex auth clone marker process token is malformed", kind="session_state")
    os.chmod(path, 0o600)
    return value


def _remove_path_and_sync(path: Path, description: str) -> None:
    """删除 clone 文件或 marker，并刷盘父目录。"""

    if path.is_symlink():
        raise HarnessError(f"{description} must not be a symlink", kind="session_state")
    if not path.exists():
        return
    try:
        path.unlink()
        _fsync_directory(path.parent, description=description)
    except OSError as exc:
        raise HarnessError(f"{description} cannot be removed", kind="session_state") from exc


def cleanup_stale_codex_auth_clone(codex_home: Path | str) -> None:
    """启动时只清理能由 process start token 证明已失效的旧 clone。"""

    home = Path(codex_home).expanduser().resolve()
    _ensure_secure_directory(home, "Codex HOME")
    clone_path = home / _AUTH_CLONE_FILE_NAME
    marker_path = home / _AUTH_CLONE_MARKER_NAME
    marker = _read_clone_marker(marker_path)
    if marker is None:
        if clone_path.is_symlink() or clone_path.exists():
            raise HarnessError(
                "Codex HOME contains an untracked auth clone; refusing to overwrite it",
                kind="session_state",
            )
        return
    current_token = _read_process_start_token(int(marker["pid"]))
    if current_token is not None and current_token == marker["proc_start_time"]:
        raise HarnessError(
            "Codex HOME auth clone belongs to an active app-server",
            kind="session_state",
        )
    _remove_path_and_sync(clone_path, "stale Codex auth clone")
    _remove_path_and_sync(marker_path, "stale Codex auth clone marker")


def _source_stat(path: Path) -> os.stat_result:
    """以 O_NOFOLLOW 打开并核对 source，避免 clone 跟随 source 符号链接。"""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise HarnessError("Codex auth source cannot be opened", kind="transport") from exc
    try:
        result = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(result.st_mode):
        raise HarnessError("Codex auth source must be a regular file", kind="transport")
    if result.st_uid != os.geteuid() or result.st_mode & 0o077:
        raise HarnessError(
            "Codex auth source must be owned by the current user and private",
            kind="transport",
        )
    return result


def _read_source_bytes(path: Path) -> tuple[bytes, os.stat_result, str]:
    """读取 source 的副本、文件状态和内存摘要；摘要不进入任何持久状态。"""

    before = _source_stat(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise HarnessError("Codex auth source cannot be opened", kind="transport") from exc
    try:
        actual = os.fstat(descriptor)
        if (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino):
            raise HarnessError("Codex auth source changed during preparation", kind="transport")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        raise HarnessError("Codex auth source cannot be read", kind="transport") from exc
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    after = _source_stat(path)
    comparable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_uid")
    if any(getattr(before, name) != getattr(after, name) for name in comparable):
        raise HarnessError("Codex auth source changed during preparation", kind="transport")
    return payload, after, hashlib.sha256(payload).hexdigest()


def _write_private_bytes(path: Path, payload: bytes, description: str) -> None:
    """以 0600 临时文件原子写入 clone，不通过 shell 或日志传播正文。"""

    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        os.chmod(path, 0o600)
        _fsync_directory(path.parent, description=description)
    except OSError as exc:
        raise HarnessError(f"{description} cannot be written", kind="session_state") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _permission_profile_toml(auth_source: Path | None) -> str:
    """生成固定权限 profile；所有路径都是代码生成的 TOML 字符串。"""

    source = str(auth_source or DEFAULT_CODEX_AUTH_SOURCE)
    # JSON 字符串语法与 TOML basic string 的转义规则在这里重合，且不允许
    # source 中出现控制字符；source 路径本身不会写入 manifest 或 trace。
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in source):
        raise HarnessError("Codex auth source path contains control characters", kind="transport")
    source_line = (
        f'"{source.replace(chr(92), chr(92) + chr(92)).replace(chr(34), chr(92) + chr(34))}" = "deny"\n'
        if source != str(DEFAULT_CODEX_AUTH_SOURCE)
        else ""
    )
    return (
        f'default_permissions = "{CODEX_INVESTIGATION_PERMISSION_PROFILE}"\n\n'
        f"[permissions.{CODEX_INVESTIGATION_PERMISSION_PROFILE}]\n"
        'extends = ":workspace"\n\n'
        f"[permissions.{CODEX_INVESTIGATION_PERMISSION_PROFILE}.filesystem]\n"
        '"/.arche/codex-home" = "deny"\n'
        '"/root/.codex/auth.json" = "deny"\n'
        f"{source_line}"
    )


def prepare_codex_permission_profile(
    codex_home: Path | str,
    *,
    auth_source: Path | None = None,
) -> str:
    """准备 Investigation 的 Codex 权限 profile，并拒绝不一致的已有配置。"""

    home = Path(codex_home).expanduser().resolve()
    _ensure_secure_directory(home, "Codex HOME")
    config_path = home / _AUTH_PROFILE_CONFIG_NAME
    expected = _permission_profile_toml(auth_source)
    if config_path.is_symlink() or (config_path.exists() and not config_path.is_file()):
        raise HarnessError("Codex permission config must be a regular file", kind="session_state")
    if config_path.exists():
        try:
            import tomllib

            parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, ModuleNotFoundError) as exc:
            raise HarnessError("Codex permission config cannot be read", kind="session_state") from exc
        permissions = parsed.get("permissions") if isinstance(parsed, dict) else None
        profile = permissions.get(CODEX_INVESTIGATION_PERMISSION_PROFILE) if isinstance(permissions, dict) else None
        filesystem = profile.get("filesystem") if isinstance(profile, dict) else None
        required_source = str(auth_source or DEFAULT_CODEX_AUTH_SOURCE)
        source_is_default = required_source == str(DEFAULT_CODEX_AUTH_SOURCE)
        if (
            parsed.get("default_permissions") != CODEX_INVESTIGATION_PERMISSION_PROFILE
            or not isinstance(profile, dict)
            or profile.get("extends") != ":workspace"
            or not isinstance(filesystem, dict)
            or filesystem.get("/.arche/codex-home") != "deny"
            or filesystem.get("/root/.codex/auth.json") != "deny"
            or (not source_is_default and filesystem.get(required_source) != "deny")
        ):
            raise HarnessError(
                "Codex permission config does not enforce the investigation read boundary",
                kind="session_state",
            )
        os.chmod(config_path, 0o600)
        return CODEX_INVESTIGATION_PERMISSION_PROFILE
    _write_private_bytes(config_path, expected.encode("utf-8"), "Codex permission config")
    return CODEX_INVESTIGATION_PERMISSION_PROFILE


@dataclass
class CodexAuthCloneSession(AbstractContextManager[None]):
    """一个 app-server 生命周期内的 local_login auth clone。"""

    codex_home: Path | str
    source: Path | str
    _clone_path: Path | None = None
    _marker_path: Path | None = None
    _prepared: bool = False

    def __enter__(self) -> None:
        self.prepare_codex_auth_clone()
        return None

    def prepare_codex_auth_clone(self) -> None:
        """准备 clone 和进程 marker；任何不确定状态都拒绝启动 app-server。"""

        if self._prepared:
            raise HarnessError("Codex auth clone was prepared twice", kind="session_state")
        home = Path(self.codex_home).expanduser().resolve()
        source = resolve_codex_auth_source(self.source)
        _ensure_secure_directory(home, "Codex HOME")
        cleanup_stale_codex_auth_clone(home)
        clone_path = home / _AUTH_CLONE_FILE_NAME
        marker_path = home / _AUTH_CLONE_MARKER_NAME
        if clone_path.exists() or marker_path.exists():
            raise HarnessError("Codex auth clone target is not empty", kind="session_state")
        payload, source_stat, _content_digest = _read_source_bytes(source)
        # source_stat 和摘要只用于内存中的一致性验证；不落盘，避免真实凭据的
        # 内容摘要成为长期可关联的身份信息。
        _write_private_bytes(clone_path, payload, "Codex auth clone")
        try:
            after = _source_stat(source)
            if any(
                getattr(source_stat, name) != getattr(after, name)
                for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_uid")
            ):
                raise HarnessError("Codex auth source changed during clone", kind="transport")
            token = _read_process_start_token(os.getpid())
            if token is None:
                raise HarnessError("Cannot prove current process identity for auth clone", kind="session_state")
            _atomic_write_json_file(
                marker_path,
                {
                    "schema_version": _AUTH_CLONE_SCHEMA_VERSION,
                    "pid": os.getpid(),
                    "proc_start_time": token,
                },
                description="Codex auth clone marker",
                mode=0o600,
            )
        except BaseException:
            try:
                _remove_path_and_sync(clone_path, "failed Codex auth clone")
            finally:
                if marker_path.exists():
                    _remove_path_and_sync(marker_path, "failed Codex auth clone marker")
            raise
        self._clone_path = clone_path
        self._marker_path = marker_path
        self._prepared = True

    def cleanup_codex_auth_clone(self) -> None:
        """删除当前 clone 和 marker；删除失败时保持 fail-closed。"""

        if not self._prepared:
            return
        assert self._clone_path is not None and self._marker_path is not None
        errors: list[BaseException] = []
        for path, description in (
            (self._clone_path, "Codex auth clone"),
            (self._marker_path, "Codex auth clone marker"),
        ):
            try:
                _remove_path_and_sync(path, description)
            except BaseException as exc:  # pragma: no cover - 仅在文件系统故障时触发
                errors.append(exc)
        self._prepared = False
        if errors:
            raise HarnessError("Codex auth clone cleanup failed", kind="session_state") from errors[0]

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        """无论 app-server 成功或失败都清理 clone；清理失败覆盖原错误。"""

        self.cleanup_codex_auth_clone()
        return False


def build_codex_auth_session(
    auth_mode: str,
    *,
    codex_home: Path | str,
    auth_source: Path | str | None,
) -> AbstractContextManager[None]:
    """按认证模式返回生命周期上下文；API 模式不创建任何登录副本。"""

    normalized = normalize_codex_auth_mode(auth_mode)
    if normalized == CODEX_AUTH_MODE_API:
        return _NoopAuthSession()
    if auth_source is None:
        raise HarnessError("local_login requires an explicit auth source", kind="transport")
    return CodexAuthCloneSession(codex_home=codex_home, source=auth_source)


class _NoopAuthSession(AbstractContextManager[None]):
    """API 模式的无副作用上下文。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        return False


__all__ = [
    "CODEX_ALLOW_PROVIDER_MIGRATION_ENV",
    "CODEX_AUTH_MODE_API",
    "CODEX_AUTH_MODE_ENV",
    "CODEX_AUTH_MODE_LOCAL_LOGIN",
    "CODEX_AUTH_SOURCE_ENV",
    "CODEX_INVESTIGATION_PERMISSION_PROFILE",
    "CODEX_LOCAL_LOGIN_ENV_KEY",
    "CODEX_LOCAL_LOGIN_MODEL",
    "CODEX_LOCAL_LOGIN_MODEL_ENV",
    "CODEX_LOCAL_LOGIN_PROVIDER_ID",
    "CODEX_LOCAL_LOGIN_REASONING_EFFORT",
    "CODEX_LOCAL_LOGIN_REASONING_EFFORT_ENV",
    "DEFAULT_CODEX_AUTH_MODE",
    "DEFAULT_CODEX_AUTH_SOURCE",
    "CodexAuthCloneSession",
    "build_codex_auth_session",
    "cleanup_stale_codex_auth_clone",
    "normalize_codex_auth_mode",
    "parse_explicit_provider_migration_flag",
    "prepare_codex_permission_profile",
    "resolve_codex_auth_source",
]
