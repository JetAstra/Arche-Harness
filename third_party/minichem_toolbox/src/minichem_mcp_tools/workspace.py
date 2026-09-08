"""Workspace confinement helpers shared by every portable chemistry tool."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Final

from minichem_toolbox.scope import current_scope


WORKSPACE_ENV_NAMES = (
    "MINICHEM_MCP_WORKSPACE",
    "RESEARCHCHEMBENCH_WORKSPACE",
)
OUTPUT_ROOTS = ("outputs", "code", "report", "tool_logs")
RESERVED_FILES = {
    "_agent_output.jsonl",
    "_meta.json",
    "_score.json",
    "_toolbox_catalog.json",
    "_tool_trace.jsonl",
    "_tool_sequence",
    "INSTRUCTIONS.md",
    ".mcp.json",
    "opencode.json",
}

# 这些目录由 MCP 自己保存目录快照、轨迹、幂等收据、临时锁或运行时状态。
# 文本工具只能访问用户明确创建的输入、结果和报告，不能把这些控制记录当作普通文件。
FRAMEWORK_INTERNAL_ROOTS: Final[frozenset[str]] = frozenset(
    {
        ".claude",
        ".codex",
        ".git",
        ".gnupg",
        ".mcp",
        ".runtime",
        ".ssh",
        "_opencode",
        "_tool_artifacts",
        "_tool_results",
        "internal",
        "lock",
        "locks",
        "receipt",
        "receipts",
        "runtime",
        "state",
        "trace",
        "traces",
        "tmp",
        "temp",
        "logs",
        "tool_logs",
    }
)
FRAMEWORK_INTERNAL_OUTPUT_ROOTS: Final[frozenset[str]] = frozenset(
    {"execution_jobs"}
)
FRAMEWORK_INTERNAL_OUTPUT_FILES: Final[frozenset[str]] = frozenset(
    {".resource_budget"}
)
EXECUTION_JOB_CONTROL_FILES: Final[frozenset[str]] = frozenset(
    {
        "analysis_contract.json",
        "artifact_manifest.json",
        "cancel_requested",
        "collection.json",
        "failure_diagnostic.json",
        "operation_receipt.json",
        "request.json",
        "runtime_output_registrations.json",
        "status.json",
        "supervisor_claim.json",
        "supervisor_spec.json",
        "stderr.log",
        "stdout.log",
        "researchchem_job.py",
    }
)
EXECUTION_JOB_INTERNAL_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {".home", ".tmp"}
)
_FRAMEWORK_INTERNAL_FILE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^_tool_"),
    re.compile(r"^_agent_output(?:$|[.])"),
    re.compile(r"^(?:\.env(?:\..+)?|config\.local\.env|credentials\.json|\.netrc)$"),
    re.compile(
        r"^\.(?:checkpoint|lock|receipt|state|trace)(?:$|[-_]|\.(?:json|jsonl|lock|tmp))"
    ),
    re.compile(r"^\.?(?:tmp|temp)(?:$|[._-])"),
    re.compile(
        r"^(?:checkpoint|operation_receipt|receipt|state|trace)\.(?:json|jsonl|lock|tmp)$"
    ),
)

_PRIVATE_DIRECTORY_MODE = 0o700


def _private_absolute_path(value: str | os.PathLike[str] | Path) -> Path:
    """把私有目录转成不解析符号链接的绝对词法路径。"""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("private runtime path must be absolute")
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_private_path_symlinks(path: Path) -> None:
    """逐组件检查私有路径，缺失的末端允许随后安全创建。"""

    absolute = _private_absolute_path(path)
    cursor = Path(absolute.anchor or os.curdir)
    components = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, component in enumerate(components):
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError("private runtime path cannot be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("private runtime path cannot contain a symlink")
        if index < len(components) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("private runtime path contains a non-directory")


def _private_directory_flags() -> int:
    """返回逐级打开私有目录所需的防跟随标志。"""

    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("private runtime directories require POSIX no-follow support")
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )


def _open_private_directory_descriptor(
    path: Path,
    *,
    create: bool,
    tighten_from: int | None = None,
) -> tuple[int, Path]:
    """逐级用 dirfd/O_NOFOLLOW 打开私有目录并可安全创建缺失组件。"""

    absolute = _private_absolute_path(path)
    if absolute == Path(absolute.anchor or os.curdir):
        raise ValueError("private runtime path cannot be filesystem root")
    _reject_private_path_symlinks(absolute)
    descriptor = os.open(absolute.anchor or os.curdir, _private_directory_flags())
    components = absolute.parts[1:] if absolute.anchor else absolute.parts
    try:
        for index, component in enumerate(components):
            try:
                child = os.open(
                    component,
                    _private_directory_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, _PRIVATE_DIRECTORY_MODE, dir_fd=descriptor)
                except FileExistsError:
                    # 另一个并发调用可能刚创建了同一目录。这里只忽略目录创建
                    # 的 EEXIST，随后仍通过 O_DIRECTORY|O_NOFOLLOW 打开并核对它；
                    # 符号链接、普通文件和权限错误会继续向上抛出。
                    pass
                child = os.open(
                    component,
                    _private_directory_flags(),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
            if tighten_from is not None and index >= tighten_from:
                os.fchmod(descriptor, _PRIVATE_DIRECTORY_MODE)
        try:
            _reject_private_path_symlinks(absolute)
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ValueError("private runtime directory cannot be inspected") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("private runtime path is not a directory")
        if tighten_from is None and stat.S_IMODE(metadata.st_mode) != _PRIVATE_DIRECTORY_MODE:
            raise ValueError("private runtime directory must be mode 0700")
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def ensure_private_directory(
    path: str | os.PathLike[str] | Path,
    *,
    create: bool = False,
) -> Path:
    """检查或创建 0700 私有目录，逐组件拒绝符号链接。"""

    descriptor, absolute = _open_private_directory_descriptor(
        Path(path),
        create=create,
        tighten_from=len(_private_absolute_path(path).parts) - 2 if create else None,
    )
    try:
        if create:
            os.fchmod(descriptor, _PRIVATE_DIRECTORY_MODE)
        return absolute
    finally:
        os.close(descriptor)


def ensure_private_runtime_directory(
    path: str | os.PathLike[str] | Path,
    *,
    runtime_root: str | os.PathLike[str] | Path,
    create: bool = True,
) -> Path:
    """在私有 runtime 根内安全检查或创建目录并返回规范绝对路径。"""

    runtime_absolute = _private_absolute_path(runtime_root)
    target_absolute = _private_absolute_path(path)
    try:
        target_absolute.relative_to(runtime_absolute)
    except ValueError as exc:
        raise ValueError("private runtime path must remain under runtime root") from exc
    ensure_private_directory(runtime_absolute, create=create)
    runtime_components = len(runtime_absolute.parts) - 1
    descriptor, absolute = _open_private_directory_descriptor(
        target_absolute,
        create=create,
        tighten_from=runtime_components - 1 if create else None,
    )
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != _PRIVATE_DIRECTORY_MODE:
            raise ValueError("private runtime directory must be mode 0700")
        return absolute
    finally:
        os.close(descriptor)


def workspace_root() -> Path:
    """返回当前请求绑定的工作区；stdio legacy 才回退到进程环境。"""

    scoped = current_scope()
    if scoped is not None:
        try:
            return ensure_private_directory(scoped.workspace, create=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("Authenticated MiniChem scope workspace is not private") from exc

    configured = next(
        (
            os.environ[name].strip()
            for name in WORKSPACE_ENV_NAMES
            if os.environ.get(name, "").strip()
        ),
        "",
    )
    root = Path(configured).expanduser() if configured else Path.cwd()
    try:
        runtime_value = os.environ.get("MINICHEM_PRIVATE_RUNTIME_ROOT", "").strip()
        if runtime_value:
            return ensure_private_runtime_directory(
                root,
                runtime_root=runtime_value,
                create=False,
            )
        return ensure_private_directory(root, create=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("Chemistry MCP workspace is not a private directory") from exc


def resolve_workspace_path(
    value: str,
    *,
    must_exist: bool = False,
    create_parent: bool = False,
) -> Path:
    """Resolve a tool path while rejecting traversal outside the workspace."""

    root = workspace_root()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    unchecked = candidate.absolute()
    try:
        relative_unchecked = unchecked.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes chemistry MCP workspace: {value}") from exc
    cursor = root
    for part in relative_unchecked.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Symlink paths are not allowed in chemistry workspace: {value}")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes chemistry MCP workspace: {value}") from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Workspace file does not exist: {resolved}")
    if create_parent:
        ensure_private_directory(resolved.parent, create=True)
    return resolved


def is_reserved_internal_workspace_path(
    value: str | Path,
    *,
    allow_execution_job_outputs: bool = False,
    allow_action_artifacts: bool = False,
) -> bool:
    """判断已解析的工作区路径是否属于 MCP 内部控制状态。

    输入可以是工作区内的绝对路径或相对路径；函数只返回布尔值，不读取文件内容，
    也不创建目录。根级目录、执行作业目录、控制文件以及带有明确框架标记的锁/收据/
    临时文件会被拒绝；普通 ``outputs/``、``code/``、``report/`` 下的用户文件保持可用。
    当 ``allow_execution_job_outputs`` 为真时，执行作业目录中的科学结果文件可被读取或
    交给后续作业，但该目录的状态、日志、收据和临时目录仍会被拒绝。
    当 ``allow_action_artifacts`` 为真时，ArtifactStore 的对象文件可被读取或交给后续作业；
    其索引、操作收据和轨迹快照仍会被拒绝。
    若路径不在工作区内，函数返回 ``False``，越界由路径解析函数负责失败。
    """

    root = workspace_root()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    parts = relative.parts
    if not parts:
        return False
    first = parts[0]
    if (
        allow_action_artifacts
        and parts[:2] == ("_tool_artifacts", "objects")
        and len(parts) == 3
        and re.fullmatch(r"art_[0-9a-f]{32}\.json", parts[2]) is not None
    ):
        return False
    if first in FRAMEWORK_INTERNAL_ROOTS:
        return True
    if any(
        part.startswith("_tool_")
        or (part.startswith(".") and part[1:] in FRAMEWORK_INTERNAL_ROOTS)
        for part in parts
    ):
        return True
    if first == "outputs":
        if len(parts) >= 2 and parts[1] in FRAMEWORK_INTERNAL_OUTPUT_ROOTS:
            if (
                allow_execution_job_outputs
                and parts[1] == "execution_jobs"
                and len(parts) >= 4
            ):
                job_relative_parts = parts[3:]
                if any(
                    part in EXECUTION_JOB_INTERNAL_DIRECTORIES
                    or part.startswith((".tmp-", ".home-"))
                    for part in job_relative_parts
                ):
                    return True
                if (
                    relative.name in EXECUTION_JOB_CONTROL_FILES
                    or relative.name in RESERVED_FILES
                    or any(
                        pattern.search(relative.name)
                        for pattern in _FRAMEWORK_INTERNAL_FILE_PATTERNS
                    )
                ):
                    return True
                return False
            return True
        if len(parts) == 2 and parts[1] in FRAMEWORK_INTERNAL_OUTPUT_FILES:
            return True
    if relative.name in RESERVED_FILES:
        return True
    return any(pattern.search(relative.name) for pattern in _FRAMEWORK_INTERNAL_FILE_PATTERNS)


def resolve_workspace_public_path(
    value: str,
    *,
    must_exist: bool = False,
    purpose: str = "Workspace file",
    allow_execution_job_outputs: bool = False,
    allow_action_artifacts: bool = False,
) -> Path:
    """解析供调用方读写或分发的文件，并拒绝框架内部路径。

    该入口复用工作区越界和符号链接检查，再执行集中式内部路径策略。失败信息只说明
    类型和原因，不回显调用方提供的绝对路径或内部文件内容；框架自身读取控制文件时
    仍使用 ``resolve_workspace_path``，避免改变内部状态机。
    """

    try:
        resolved = resolve_workspace_path(value, must_exist=must_exist)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{purpose} does not exist") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{purpose} path is invalid") from exc
    if is_reserved_internal_workspace_path(
        resolved,
        allow_execution_job_outputs=allow_execution_job_outputs,
        allow_action_artifacts=allow_action_artifacts,
    ):
        raise ValueError(f"{purpose} path is reserved for framework state")
    return resolved


def resolve_workspace_output_path(
    value: str,
    *,
    allowed_roots: tuple[str, ...] = OUTPUT_ROOTS,
) -> Path:
    """Resolve a writable path while protecting inputs and benchmark control files."""

    resolved = resolve_workspace_path(value, create_parent=False)
    root = workspace_root()
    relative = resolved.relative_to(root)
    if not relative.parts or relative.parts[0] not in allowed_roots:
        raise ValueError(
            f"Tool outputs must be under one of {allowed_roots}, received: {value}"
        )
    if relative.name in RESERVED_FILES or relative.parts[0].startswith("_tool_"):
        raise ValueError(f"Tool output path is reserved: {value}")
    ensure_private_directory(resolved.parent, create=True)
    return resolved


def relative_workspace_path(path: str | Path) -> str:
    """Represent a path relative to the active chemistry workspace."""

    resolved = Path(path).resolve(strict=False)
    return str(resolved.relative_to(workspace_root()))
