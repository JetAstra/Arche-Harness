"""任务级研究材料字节和摘要持久化深模块。

``ResearchMaterialStore`` 只拥有上传材料的内容寻址字节和摘要。它不保存研究员
决定、不写 Working Memory，也不把 Run Bundle 当作恢复事实。调用方通过一个很小的
公开接口摄取材料并将已核对的材料复制到某个 Session workspace；CAS 目录、临时文件、
原子发布和权限细节都留在实现内部。
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, BinaryIO, Final, Iterator, TypeAlias


class ResearchMaterialStoreError(RuntimeError):
    """研究材料存储无法安全完成请求。"""


class ResearchMaterialDescriptorError(ResearchMaterialStoreError, ValueError):
    """材料 descriptor、文件名、媒体类型或存储策略不满足合同。"""


class ResearchMaterialUnsafePath(ResearchMaterialDescriptorError):
    """任务根、CAS、workspace 或目标路径不是安全的普通路径。"""


class ResearchMaterialSizeLimitExceeded(ResearchMaterialDescriptorError):
    """材料字节数超过显式单文件大小策略。"""


class ResearchMaterialCountLimitExceeded(ResearchMaterialDescriptorError):
    """任务中登记的材料 descriptor 数量超过冻结策略。"""


class ResearchMaterialTaskSizeLimitExceeded(ResearchMaterialDescriptorError):
    """任务中登记的材料逻辑字节总量超过冻结策略。"""


class ResearchMaterialReadError(ResearchMaterialStoreError, OSError):
    """来源流或材料文件无法读取。"""


class ResearchMaterialMissing(ResearchMaterialStoreError, FileNotFoundError):
    """descriptor 引用的 CAS 字节不存在。"""


class ResearchMaterialIntegrityConflict(ResearchMaterialStoreError):
    """CAS 字节或 CAS 权限与 descriptor 摘要发生漂移。"""


class ResearchMaterialTargetConflict(ResearchMaterialStoreError):
    """workspace 中已有目标内容与要物化的材料发生漂移。"""


class ResearchMaterialPersistenceError(ResearchMaterialStoreError):
    """临时文件、原子发布或目录刷盘失败。"""


# 这些别名让恢复层可以用更明确的语义命名，同时仍保持一个 canonical 异常类型。
ResearchMaterialContractError = ResearchMaterialDescriptorError
ResearchMaterialMissingError = ResearchMaterialMissing
ResearchMaterialDigestMismatch = ResearchMaterialIntegrityConflict
ResearchMaterialIntegrityError = ResearchMaterialIntegrityConflict
ResearchMaterialTargetDrift = ResearchMaterialTargetConflict
ResearchMaterialMaterialCountLimitExceeded = ResearchMaterialCountLimitExceeded
ResearchMaterialTotalSizeLimitExceeded = ResearchMaterialTaskSizeLimitExceeded


_DEFAULT_MAX_FILENAME_BYTES: Final[int] = 255


@dataclass(frozen=True, slots=True)
class MaterialSizePolicy:
    """一个材料存储实例使用的冻结材料数量和字节上限。

    ``max_material_count`` 和 ``max_task_size_bytes`` 为 ``None`` 时表示调用方明确
    选择不启用对应门禁，而不是由 store 隐含采用某个配额。``max_filename_bytes``
    是安全 basename 的 UTF-8 字节上限，默认 255。相同完整 descriptor 的精确重放
    不增加数量或逻辑总量；不同文件名或描述即使复用同一 CAS 字节，也算新的
    descriptor 并计入这两项请求层限制。
    """

    max_file_size_bytes: int
    max_material_count: int | None = None
    max_task_size_bytes: int | None = None
    max_filename_bytes: int = _DEFAULT_MAX_FILENAME_BYTES

    def __post_init__(self) -> None:
        """校验上限是允许零字节材料的非负严格整数。"""

        if (
            isinstance(self.max_file_size_bytes, bool)
            or not isinstance(self.max_file_size_bytes, int)
            or self.max_file_size_bytes < 0
        ):
            raise ResearchMaterialDescriptorError(
                "max_file_size_bytes 必须是非负整数"
            )
        for field_name, value in (
            ("max_material_count", self.max_material_count),
            ("max_task_size_bytes", self.max_task_size_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ResearchMaterialDescriptorError(
                    f"{field_name} 必须是非负整数或 None"
                )
        if (
            isinstance(self.max_filename_bytes, bool)
            or not isinstance(self.max_filename_bytes, int)
            or self.max_filename_bytes <= 0
        ):
            raise ResearchMaterialDescriptorError(
                "max_filename_bytes 必须是正整数"
            )


ResearchMaterialSizePolicy = MaterialSizePolicy
PerFileSizePolicy = MaterialSizePolicy

ResearchMaterialDescriptor: TypeAlias = dict[str, object]
MaterialDescriptor: TypeAlias = ResearchMaterialDescriptor


_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]:[\\/]")
_CONTROL_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")
_ABSOLUTE_PATH_IN_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w])/(?:[^/\s]+(?:/[^/\s]*)*)|(?<![\w])[A-Za-z]:[\\/]"
)
_CAS_ROOT_NAME: Final[str] = ".research_materials"
_CAS_DIRECTORY_NAME: Final[str] = "cas"
_MATERIAL_MANIFEST_NAME: Final[str] = "manifest.json"
_MATERIAL_LOCK_NAME: Final[str] = "manifest.lock"
_MATERIAL_MANIFEST_SCHEMA_VERSION: Final[int] = 1
_WORKSPACE_INPUT_DIRECTORY: Final[tuple[str, str]] = ("inputs", "materials")
_CHUNK_SIZE: Final[int] = 1024 * 1024
_PRIVATE_DIRECTORY_MODE: Final[int] = 0o700
_PRIVATE_FILE_MODE: Final[int] = 0o600


def _require_pathlike(value: Any, field_name: str) -> Path:
    """把调用方路径转成绝对路径，并拒绝显式 ``..`` 路径段。"""

    if not isinstance(value, (str, os.PathLike)):
        raise ResearchMaterialUnsafePath(f"{field_name} 必须是文件系统路径")
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ResearchMaterialUnsafePath(f"{field_name} 必须是文件系统路径") from exc
    if isinstance(raw, bytes):
        raise ResearchMaterialUnsafePath(f"{field_name} 必须是文本路径")
    path = Path(raw)
    if any(part == ".." for part in path.parts):
        raise ResearchMaterialUnsafePath(f"{field_name} 不能包含目录穿越")
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _check_no_symlink_components(path: Path, *, allow_missing_tail: bool) -> None:
    """逐层检查路径，拒绝符号链接和特殊文件。"""

    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_tail:
                return
            raise ResearchMaterialUnsafePath("路径存在缺失的目录层级")
        except OSError as exc:
            raise ResearchMaterialUnsafePath("路径无法安全核对") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ResearchMaterialUnsafePath("路径不能经过符号链接")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ResearchMaterialUnsafePath("路径中间层级必须是普通目录")


def _prepare_task_root(value: Any) -> Path:
    """创建或核对 task root，并返回不含符号链接的绝对目录。"""

    root = _require_pathlike(value, "task_root")
    if root == Path(root.anchor):
        raise ResearchMaterialUnsafePath("task_root 不能是文件系统根目录")
    _check_no_symlink_components(root, allow_missing_tail=True)
    try:
        created = False
        try:
            metadata = os.lstat(root)
        except FileNotFoundError:
            root.mkdir(parents=True, mode=_PRIVATE_DIRECTORY_MODE, exist_ok=True)
            metadata = os.lstat(root)
            created = True
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ResearchMaterialUnsafePath("task_root 必须是普通目录")
        _check_no_symlink_components(root, allow_missing_tail=False)
        if created:
            os.chmod(root, _PRIVATE_DIRECTORY_MODE)
    except ResearchMaterialUnsafePath:
        raise
    except OSError as exc:
        raise ResearchMaterialPersistenceError("task_root 无法准备") from exc
    return root


def _require_existing_directory(value: Any, field_name: str) -> Path:
    """只读核对一个已存在的普通目录，不为 workspace 补建根目录。"""

    directory = _require_pathlike(value, field_name)
    if directory == Path(directory.anchor):
        raise ResearchMaterialUnsafePath(f"{field_name} 不能是文件系统根目录")
    _check_no_symlink_components(directory, allow_missing_tail=False)
    try:
        metadata = os.lstat(directory)
    except FileNotFoundError as exc:
        raise ResearchMaterialUnsafePath(f"{field_name} 必须是已存在的普通目录") from exc
    except OSError as exc:
        raise ResearchMaterialUnsafePath(f"{field_name} 无法安全核对") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ResearchMaterialUnsafePath(f"{field_name} 必须是普通目录")
    return directory


def _create_private_directory(directory: Path) -> None:
    """幂等创建存储内部目录，并把目录权限收紧到 ``0700``。"""

    _check_no_symlink_components(directory, allow_missing_tail=True)
    try:
        directory.mkdir(parents=True, mode=_PRIVATE_DIRECTORY_MODE, exist_ok=True)
        _check_no_symlink_components(directory, allow_missing_tail=False)
        metadata = os.lstat(directory)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ResearchMaterialUnsafePath("存储目录必须是普通目录")
        os.chmod(directory, _PRIVATE_DIRECTORY_MODE)
    except ResearchMaterialUnsafePath:
        raise
    except FileExistsError as exc:
        try:
            metadata = os.lstat(directory)
        except OSError:
            raise ResearchMaterialUnsafePath("存储目录不能是特殊文件") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ResearchMaterialUnsafePath("存储目录必须是普通目录") from exc
        raise ResearchMaterialPersistenceError("存储目录无法准备") from exc
    except OSError as exc:
        raise ResearchMaterialPersistenceError("存储目录无法准备") from exc


def _sync_directory(directory: Path) -> None:
    """刷新目录项，保证原子发布后的名称可恢复读取。"""

    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError as exc:
        raise ResearchMaterialPersistenceError("存储目录无法刷盘") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                    raise ResearchMaterialPersistenceError("存储目录句柄无法关闭") from exc


def _strict_json_bytes(value: Mapping[str, object]) -> bytes:
    """编码仅含材料摘要的内部 manifest，并拒绝不可表示值。"""

    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResearchMaterialPersistenceError("材料摘要无法编码") from exc


def _atomic_write_private_file(path: Path, payload: bytes) -> None:
    """在私有目录中以临时文件、fsync 和原子替换写入摘要。"""

    if path.is_symlink():
        raise ResearchMaterialUnsafePath("材料摘要文件不能是符号链接")
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, _PRIVATE_FILE_MODE)
        if path.is_symlink():
            raise ResearchMaterialUnsafePath("材料摘要文件不能是符号链接")
        os.replace(temporary_path, path)
        temporary_path = None
        os.chmod(path, _PRIVATE_FILE_MODE)
        _sync_directory(path.parent)
    except ResearchMaterialStoreError:
        raise
    except OSError as exc:
        raise ResearchMaterialPersistenceError("材料摘要无法原子写入") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _read_private_json(path: Path) -> Mapping[str, object] | None:
    """只读核对私有 manifest；缺失返回空值，损坏统一报告完整性冲突。"""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResearchMaterialIntegrityConflict("材料摘要文件无法核对") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ResearchMaterialUnsafePath("材料摘要文件必须是普通文件")
    if stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE:
        raise ResearchMaterialIntegrityConflict("材料摘要文件权限不是严格的 0600")
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        raw_parts: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            raw_parts.append(chunk)
        value = json.loads(b"".join(raw_parts).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResearchMaterialIntegrityConflict("材料摘要文件损坏") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise ResearchMaterialIntegrityConflict("材料摘要文件句柄无法关闭") from exc
    if not isinstance(value, Mapping):
        raise ResearchMaterialIntegrityConflict("材料摘要必须是 JSON object")
    return value


@contextmanager
def _lock_manifest(lock_path: Path) -> Iterator[None]:
    """以 Linux 独占文件锁保护材料摘要更新。"""

    descriptor = -1
    try:
        try:
            existing = os.lstat(lock_path)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)
        ):
            raise ResearchMaterialUnsafePath("材料摘要锁必须是普通文件")
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            _PRIVATE_FILE_MODE,
        )
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        descriptor_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_metadata.st_mode):
            raise ResearchMaterialUnsafePath("材料摘要锁必须是普通文件")
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - 固定 Linux 环境始终提供
            raise ResearchMaterialPersistenceError("材料摘要锁需要 fcntl") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise ResearchMaterialPersistenceError("材料摘要锁无法获取") from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError as exc:
                raise ResearchMaterialPersistenceError("材料摘要锁无法释放") from exc
    except ResearchMaterialStoreError:
        raise
    except OSError as exc:
        raise ResearchMaterialPersistenceError("材料摘要锁无法打开") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise ResearchMaterialPersistenceError("材料摘要锁句柄无法关闭") from exc


def _validate_safe_basename(
    value: Any,
    field_name: str = "filename",
    *,
    max_filename_bytes: int = _DEFAULT_MAX_FILENAME_BYTES,
) -> str:
    """校验不含目录语义、控制字符和过长字节序列的安全 basename。"""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ResearchMaterialDescriptorError(f"{field_name} 必须是非空安全 basename")
    if (
        "/" in value
        or "\\" in value
        or _CONTROL_RE.search(value) is not None
        or _WINDOWS_ABSOLUTE_PATH_RE.match(value) is not None
    ):
        raise ResearchMaterialDescriptorError(
            f"{field_name} 不能包含路径分隔符、控制字符或绝对路径"
        )
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ResearchMaterialDescriptorError(f"{field_name} 必须是可编码文件名") from exc
    if encoded_length > max_filename_bytes:
        raise ResearchMaterialDescriptorError(
            f"{field_name} 不能超过 {max_filename_bytes} 字节"
        )
    return value


def _validate_media_type(value: Any) -> str:
    """校验并规范化非空媒体类型，拒绝控制字符和绝对路径文本。"""

    if not isinstance(value, str):
        raise ResearchMaterialDescriptorError("media_type 必须是非空字符串")
    normalized = value.strip()
    if (
        not normalized
        or _CONTROL_RE.search(normalized) is not None
        or _WINDOWS_ABSOLUTE_PATH_RE.match(normalized) is not None
        or normalized.startswith("/")
    ):
        raise ResearchMaterialDescriptorError("media_type 必须是非空安全文本")
    return normalized


def _validate_optional_description(value: Any) -> str | None:
    """校验可选描述，确保不会把调用方绝对路径写入 descriptor。"""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ResearchMaterialDescriptorError("description 必须是字符串")
    normalized = value.strip()
    if not normalized:
        return None
    if (
        _CONTROL_RE.search(normalized) is not None
        or _ABSOLUTE_PATH_IN_TEXT_RE.search(normalized) is not None
    ):
        raise ResearchMaterialDescriptorError(
            "description 不能包含控制字符或调用方绝对路径"
        )
    return normalized


def _validate_size_limit(value: Any) -> int:
    """校验单文件上限参数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchMaterialDescriptorError(
            "max_file_size_bytes 必须是非负整数"
        )
    return value


def _validate_digest(value: Any, field_name: str = "sha256") -> str:
    """校验小写 64 位 SHA-256 文本。"""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ResearchMaterialDescriptorError(
            f"{field_name} 必须是 64 位小写 SHA-256"
        )
    return value


def _descriptor_content_ref(sha256: str) -> str:
    """生成不含路径分隔符的内容引用。"""

    return f"sha256:{sha256}"


def _canonical_descriptor_key(descriptor: Mapping[str, object]) -> str:
    """为精确 descriptor 重放生成稳定内部键。"""

    return json.dumps(
        dict(descriptor),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _read_regular_file_bytes(
    path: Path,
    *,
    missing_error: type[ResearchMaterialStoreError],
    integrity_error: type[ResearchMaterialStoreError],
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    require_private_mode: bool = False,
) -> tuple[bytes, os.stat_result]:
    """以 ``O_NOFOLLOW`` 读取普通文件并可核对大小和摘要。"""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        try:
            path_metadata = os.lstat(path)
        except FileNotFoundError as exc:
            raise missing_error("材料 CAS 字节缺失") from exc
        if stat.S_ISLNK(path_metadata.st_mode):
            raise ResearchMaterialUnsafePath("材料文件不能是符号链接")
        if not stat.S_ISREG(path_metadata.st_mode):
            raise ResearchMaterialUnsafePath("材料文件必须是普通文件")
        if require_private_mode:
            if stat.S_IMODE(path_metadata.st_mode) != _PRIVATE_FILE_MODE:
                raise integrity_error("材料文件权限不是严格的 0600")
        elif path_metadata.st_mode & 0o111:
            raise integrity_error("材料文件权限含 executable bit")
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise missing_error("材料 CAS 字节缺失") from exc
        metadata = os.fstat(descriptor)
        if stat.S_ISLNK(metadata.st_mode):
            raise ResearchMaterialUnsafePath("材料文件不能是符号链接")
        if not stat.S_ISREG(metadata.st_mode):
            raise ResearchMaterialUnsafePath("材料文件不是普通文件")
        if require_private_mode:
            if stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE:
                raise integrity_error("材料文件权限不是严格的 0600")
        elif metadata.st_mode & 0o111:
            raise integrity_error("材料文件权限含 executable bit")
        chunks: list[bytes] = []
        total = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            chunks.append(chunk)
            digest.update(chunk)
            if expected_size is not None and total > expected_size:
                raise integrity_error("材料文件大小与 descriptor 不一致")
        content = b"".join(chunks)
        if expected_size is not None and total != expected_size:
            raise integrity_error("材料文件大小与 descriptor 不一致")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise integrity_error("材料文件 SHA-256 与 descriptor 不一致")
        return content, metadata
    except (ResearchMaterialStoreError, ResearchMaterialReadError):
        raise
    except OSError as exc:
        raise ResearchMaterialReadError("材料文件无法读取") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise ResearchMaterialReadError("材料文件句柄无法关闭") from exc


class ResearchMaterialStore:
    """按 task 隔离材料字节、摘要和 workspace 物化的深模块。

    构造输入是 task 根目录和显式单文件大小策略；task 根可在普通父目录中幂等创建，
    但不会读取或写入 Working Memory、Run Bundle 或任何研究员决定。公开方法只返回
    JSON 可序列化 descriptor 或 workspace 内相对路径，绝不返回调用方绝对路径。
    """

    def __init__(
        self,
        task_root: str | os.PathLike[str],
        size_policy: MaterialSizePolicy | int | None = None,
        *,
        max_file_size_bytes: int | None = None,
        max_material_count: int | None = None,
        max_task_size_bytes: int | None = None,
        max_filename_bytes: int | None = None,
        per_file_size_policy: MaterialSizePolicy | int | None = None,
    ) -> None:
        """固定 task 根和显式单文件大小策略。

        输入是 task 根路径，以及 ``MaterialSizePolicy`` 或等价的整数上限；也可用
        ``max_file_size_bytes``/``per_file_size_policy`` 关键字表达同一策略。
        ``max_material_count`` 和 ``max_task_size_bytes`` 可同时冻结任务级材料数量
        与逻辑字节总量；传入 ``None`` 表示调用方明确选择不启用对应门禁。
        ``max_filename_bytes`` 可冻结安全 basename 的 UTF-8 字节上限，省略时使用
        255 字节的文件系统安全上限。动作只准备普通 task 目录并保存进程内策略，
        不创建材料 CAS 内容。返回无值；路径、策略缺失、冲突、符号链接或特殊文件
        时抛出 typed store 异常。
        """

        policies = [
            policy
            for policy in (size_policy, per_file_size_policy)
            if policy is not None
        ]
        if max_file_size_bytes is not None:
            policies.append(max_file_size_bytes)
        if len(policies) != 1:
            raise ResearchMaterialDescriptorError(
                "必须显式提供且只能提供一个单文件大小策略"
            )
        policy_value = policies[0]
        if isinstance(policy_value, MaterialSizePolicy):
            if (
                max_material_count is not None
                and policy_value.max_material_count is not None
                and max_material_count != policy_value.max_material_count
            ) or (
                max_task_size_bytes is not None
                and policy_value.max_task_size_bytes is not None
                and max_task_size_bytes != policy_value.max_task_size_bytes
            ) or (
                max_filename_bytes is not None
                and max_filename_bytes != policy_value.max_filename_bytes
            ):
                raise ResearchMaterialDescriptorError("材料冻结策略发生冲突")
            policy = MaterialSizePolicy(
                policy_value.max_file_size_bytes,
                max_material_count=(
                    policy_value.max_material_count
                    if max_material_count is None
                    else max_material_count
                ),
                max_task_size_bytes=(
                    policy_value.max_task_size_bytes
                    if max_task_size_bytes is None
                    else max_task_size_bytes
                ),
                max_filename_bytes=(
                    policy_value.max_filename_bytes
                    if max_filename_bytes is None
                    else max_filename_bytes
                ),
            )
        else:
            policy = MaterialSizePolicy(
                _validate_size_limit(policy_value),
                max_material_count=max_material_count,
                max_task_size_bytes=max_task_size_bytes,
                max_filename_bytes=(
                    _DEFAULT_MAX_FILENAME_BYTES
                    if max_filename_bytes is None
                    else max_filename_bytes
                ),
            )
        self._task_root = _prepare_task_root(task_root)
        self._size_policy = policy

    def ingest_material_bytes(
        self,
        filename: str,
        media_type: str,
        content: bytes | bytearray | memoryview,
        *,
        description: str | None = None,
    ) -> ResearchMaterialDescriptor:
        """摄取一个 bytes 材料并返回 JSON descriptor。

        输入是安全 basename、非空媒体类型、bytes-like 内容和可选描述；动作按显式
        单文件上限计算 SHA-256，将内容写入 task-scoped CAS，并通过同目录临时普通
        文件、fsync 和不可覆盖的原子链接发布。返回值只含安全文件名、媒体类型、大小、
        摘要、不透明 content_ref 和可选描述；参数、大小、权限或持久化失败时抛出
        typed store 异常，且不会留下半个 CAS 文件。
        """

        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise ResearchMaterialDescriptorError("content 必须是 bytes-like")
        try:
            raw_content = bytes(content)
        except (TypeError, ValueError) as exc:
            raise ResearchMaterialDescriptorError("content 必须是 bytes-like") from exc
        return self.ingest_material_stream(
            filename,
            media_type,
            _BytesReader(raw_content),
            description=description,
        )

    def ingest_material_stream(
        self,
        filename: str,
        media_type: str,
        stream: BinaryIO,
        *,
        description: str | None = None,
    ) -> ResearchMaterialDescriptor:
        """从二进制文件流摄取一个材料并返回 JSON descriptor。

        输入是可重复调用 ``read(size)`` 的二进制流、文件元数据和可选描述；动作边读
        边计算大小与 SHA-256，在 task-scoped CAS 中原子发布完整字节。返回值是普通
        JSON object，不包含流的 ``name`` 或任何调用方路径；流读取失败、超限、路径
        合同失败或 CAS 发生已有内容冲突时抛出 typed store 异常，并清理临时文件。
        """

        safe_filename, safe_media_type, safe_description = self._validate_material_metadata(
            filename,
            media_type,
            description,
        )
        if not hasattr(stream, "read") or not callable(stream.read):
            raise ResearchMaterialDescriptorError("stream 必须提供 read(size) 方法")

        cas_directory = self._prepare_cas_directory()
        temporary_descriptor = -1
        temporary_path: Path | None = None
        total = 0
        digest = hashlib.sha256()
        try:
            temporary_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".material-upload.",
                suffix=".tmp",
                dir=cas_directory,
            )
            temporary_path = Path(temporary_name)
            os.fchmod(temporary_descriptor, _PRIVATE_FILE_MODE)
            with os.fdopen(temporary_descriptor, "wb") as output:
                temporary_descriptor = -1
                while True:
                    try:
                        chunk = stream.read(_CHUNK_SIZE)
                    except (ResearchMaterialStoreError, ResearchMaterialReadError):
                        raise
                    except (OSError, ValueError) as exc:
                        raise ResearchMaterialReadError("材料来源流无法读取") from exc
                    if chunk is None:
                        raise ResearchMaterialReadError("材料来源流返回空值")
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise ResearchMaterialReadError("材料来源流必须返回 bytes")
                    try:
                        chunk_bytes = bytes(chunk)
                    except (TypeError, ValueError) as exc:
                        raise ResearchMaterialReadError("材料来源流必须返回 bytes") from exc
                    if not chunk_bytes:
                        break
                    total += len(chunk_bytes)
                    if total > self._size_policy.max_file_size_bytes:
                        raise ResearchMaterialSizeLimitExceeded(
                            "材料超过显式单文件大小上限"
                        )
                    output.write(chunk_bytes)
                    digest.update(chunk_bytes)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary_path, _PRIVATE_FILE_MODE)
            sha256 = digest.hexdigest()
            descriptor: ResearchMaterialDescriptor = {
                "filename": safe_filename,
                "media_type": safe_media_type,
                "size_bytes": total,
                "sha256": sha256,
                "content_ref": _descriptor_content_ref(sha256),
            }
            if safe_description is not None:
                descriptor["description"] = safe_description
            self._register_material_descriptor(descriptor, temporary_path)
        except (ResearchMaterialStoreError, ResearchMaterialReadError):
            raise
        except (OSError, ValueError) as exc:
            raise ResearchMaterialPersistenceError("材料无法原子写入 CAS") from exc
        finally:
            if temporary_descriptor >= 0:
                try:
                    os.close(temporary_descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

        return descriptor

    def ingest_material(
        self,
        filename: str,
        media_type: str,
        source: bytes | bytearray | memoryview | BinaryIO,
        *,
        description: str | None = None,
    ) -> ResearchMaterialDescriptor:
        """从 bytes-like 或二进制流摄取一个材料。

        输入是与 ``ingest_material_bytes``/``ingest_material_stream`` 相同的元数据及
        一个 bytes-like 或 ``read(size)`` 来源；动作把来源转交到同一 CAS 实现。返回
        JSON descriptor；来源类型或存储合同不合法时抛出 typed store 异常。该方法不
        保存来源对象、来源名称或来源路径。
        """

        if isinstance(source, (bytes, bytearray, memoryview)):
            return self.ingest_material_bytes(
                filename,
                media_type,
                source,
                description=description,
            )
        return self.ingest_material_stream(
            filename,
            media_type,
            source,
            description=description,
        )

    def materialize_material(
        self,
        descriptor: Mapping[str, object],
        workspace: str | os.PathLike[str],
    ) -> str:
        """核对 descriptor 并复制材料到 Session workspace 的系统相对路径。

        输入是本 store 产生的 JSON descriptor 和已存在的普通 workspace；动作先核对
        descriptor、CAS 普通文件、大小、SHA-256 和非 executable 权限，再把内容原子
        复制到确定的 ``inputs/materials/<sha256>/<filename>`` 相对路径。返回值只是一
        个 POSIX 相对路径；CAS 缺失抛出 ``ResearchMaterialMissing``，CAS 摘要/权限漂
        移抛出 ``ResearchMaterialIntegrityConflict``，workspace 或目标不安全、目标内容
        漂移、文件类型或原子写入失败时抛出对应 typed store 异常。
        """

        normalized = self._validate_descriptor(descriptor)
        sha256 = normalized["sha256"]
        source_path = self._cas_path(sha256)
        content, source_metadata = _read_regular_file_bytes(
            source_path,
            missing_error=ResearchMaterialMissing,
            integrity_error=ResearchMaterialIntegrityConflict,
            expected_size=normalized["size_bytes"],
            expected_sha256=sha256,
            require_private_mode=True,
        )
        workspace_root = _require_existing_directory(workspace, "workspace")
        relative_path = Path(
            *_WORKSPACE_INPUT_DIRECTORY,
            sha256,
            normalized["filename"],
        )
        target_directory = workspace_root.joinpath(*relative_path.parts[:-1])
        for directory in (
            workspace_root / _WORKSPACE_INPUT_DIRECTORY[0],
            workspace_root.joinpath(*_WORKSPACE_INPUT_DIRECTORY),
            target_directory,
        ):
            _create_private_directory(directory)
        target = workspace_root / relative_path
        self._publish_workspace_bytes(
            target,
            content,
            expected_size=normalized["size_bytes"],
            expected_sha256=sha256,
            source_metadata=source_metadata,
        )
        return relative_path.as_posix()

    def _validate_material_metadata(
        self,
        filename: str,
        media_type: str,
        description: str | None,
    ) -> tuple[str, str, str | None]:
        """校验摄取入口的元数据，不读取或写入任何文件。"""

        return (
            _validate_safe_basename(
                filename,
                max_filename_bytes=self._size_policy.max_filename_bytes,
            ),
            _validate_media_type(media_type),
            _validate_optional_description(description),
        )

    def _validate_descriptor(
        self,
        descriptor: Mapping[str, object],
    ) -> dict[str, object]:
        """严格校验 descriptor 字段并返回不含调用方可变映射的副本。"""

        if not isinstance(descriptor, Mapping):
            raise ResearchMaterialDescriptorError("descriptor 必须是 JSON object")
        expected_keys = {
            "filename",
            "media_type",
            "size_bytes",
            "sha256",
            "content_ref",
        }
        optional_keys = {"description"}
        actual_keys = set(descriptor)
        if not actual_keys.issubset(expected_keys | optional_keys) or not expected_keys.issubset(
            actual_keys
        ):
            raise ResearchMaterialDescriptorError("descriptor 字段合同不匹配")
        filename = _validate_safe_basename(
            descriptor.get("filename"),
            max_filename_bytes=self._size_policy.max_filename_bytes,
        )
        media_type = _validate_media_type(descriptor.get("media_type"))
        size_bytes = descriptor.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ResearchMaterialDescriptorError("size_bytes 必须是非负整数")
        if size_bytes > self._size_policy.max_file_size_bytes:
            raise ResearchMaterialSizeLimitExceeded(
                "descriptor 超过当前显式单文件大小上限"
            )
        sha256 = _validate_digest(descriptor.get("sha256"))
        content_ref = descriptor.get("content_ref")
        if content_ref != _descriptor_content_ref(sha256):
            raise ResearchMaterialDescriptorError("content_ref 与 sha256 不一致")
        description = _validate_optional_description(descriptor.get("description"))
        normalized: dict[str, object] = {
            "filename": filename,
            "media_type": media_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "content_ref": content_ref,
        }
        if description is not None:
            normalized["description"] = description
        return normalized

    def _prepare_cas_directory(self) -> Path:
        """准备 task 内部 CAS 目录，不暴露其路径。"""

        private_root = self._task_root / _CAS_ROOT_NAME
        cas_directory = private_root / _CAS_DIRECTORY_NAME
        _create_private_directory(private_root)
        _create_private_directory(cas_directory)
        return cas_directory

    @property
    def _manifest_path(self) -> Path:
        """返回内部材料摘要 manifest 路径；该路径不属于公开接口。"""

        return self._task_root / _CAS_ROOT_NAME / _MATERIAL_MANIFEST_NAME

    @property
    def _manifest_lock_path(self) -> Path:
        """返回内部 manifest 锁路径；该路径不属于公开接口。"""

        return self._task_root / _CAS_ROOT_NAME / _MATERIAL_LOCK_NAME

    def _read_material_manifest(self) -> list[dict[str, object]]:
        """读取并核对只含 descriptor 摘要的内部索引。"""

        value = _read_private_json(self._manifest_path)
        if value is None:
            return []
        if set(value) != {
            "schema_version",
            "materials",
            "material_count",
            "total_size_bytes",
        }:
            raise ResearchMaterialIntegrityConflict("材料摘要 manifest 字段不完整")
        if value["schema_version"] != _MATERIAL_MANIFEST_SCHEMA_VERSION:
            raise ResearchMaterialIntegrityConflict("材料摘要 manifest 版本不支持")
        raw_materials = value["materials"]
        if not isinstance(raw_materials, list):
            raise ResearchMaterialIntegrityConflict("材料摘要 manifest 列表非法")
        material_count = value["material_count"]
        total_size_bytes = value["total_size_bytes"]
        if (
            isinstance(material_count, bool)
            or not isinstance(material_count, int)
            or material_count != len(raw_materials)
            or isinstance(total_size_bytes, bool)
            or not isinstance(total_size_bytes, int)
            or total_size_bytes < 0
        ):
            raise ResearchMaterialIntegrityConflict("材料摘要 manifest 统计值非法")
        materials: list[dict[str, object]] = []
        seen: set[str] = set()
        computed_total = 0
        for item in raw_materials:
            if not isinstance(item, Mapping):
                raise ResearchMaterialIntegrityConflict("材料摘要 manifest 项非法")
            try:
                normalized = self._validate_descriptor(item)
            except ResearchMaterialSizeLimitExceeded as exc:
                raise ResearchMaterialIntegrityConflict(
                    "材料摘要 manifest 项超过当前冻结单文件策略"
                ) from exc
            except ResearchMaterialDescriptorError as exc:
                raise ResearchMaterialIntegrityConflict("材料摘要 manifest 项非法") from exc
            key = _canonical_descriptor_key(normalized)
            if key in seen:
                raise ResearchMaterialIntegrityConflict("材料摘要 manifest 含重复 descriptor")
            seen.add(key)
            materials.append(normalized)
            computed_total += normalized["size_bytes"]
        if computed_total != total_size_bytes:
            raise ResearchMaterialIntegrityConflict("材料摘要 manifest 总大小漂移")
        return materials

    def _write_material_manifest(
        self,
        materials: list[dict[str, object]],
    ) -> None:
        """原子写入仅含材料摘要的内部索引。"""

        payload = _strict_json_bytes(
            {
                "schema_version": _MATERIAL_MANIFEST_SCHEMA_VERSION,
                "materials": materials,
                "material_count": len(materials),
                "total_size_bytes": sum(item["size_bytes"] for item in materials),
            }
        )
        _atomic_write_private_file(self._manifest_path, payload)

    def _register_material_descriptor(
        self,
        descriptor: ResearchMaterialDescriptor,
        temporary: Path,
    ) -> None:
        """在锁内原子发布 CAS 并登记 descriptor，精确重放不重复计数。"""

        normalized = self._validate_descriptor(descriptor)
        private_root = self._task_root / _CAS_ROOT_NAME
        lock_path = private_root / _MATERIAL_LOCK_NAME
        with _lock_manifest(lock_path):
            materials = self._read_material_manifest()
            key = _canonical_descriptor_key(normalized)
            is_exact_replay = any(
                _canonical_descriptor_key(item) == key for item in materials
            )
            if not is_exact_replay:
                max_count = self._size_policy.max_material_count
                if max_count is not None and len(materials) >= max_count:
                    raise ResearchMaterialCountLimitExceeded(
                        "材料数量超过冻结任务级数量上限"
                    )
                max_task_size = self._size_policy.max_task_size_bytes
                logical_total = sum(item["size_bytes"] for item in materials)
                if (
                    max_task_size is not None
                    and logical_total + normalized["size_bytes"] > max_task_size
                ):
                    raise ResearchMaterialTaskSizeLimitExceeded(
                        "材料逻辑字节总量超过冻结任务级大小上限"
                    )
            self._publish_cas_bytes(
                temporary,
                self._cas_path(normalized["sha256"]),
                expected_size=normalized["size_bytes"],
                expected_sha256=normalized["sha256"],
            )
            if not is_exact_replay:
                materials.append(normalized)
                self._write_material_manifest(materials)

    def _cas_path(self, sha256: str) -> Path:
        """按内部摘要映射 CAS 路径；调用方永远只接触 content_ref。"""

        _validate_digest(sha256)
        return (
            self._task_root
            / _CAS_ROOT_NAME
            / _CAS_DIRECTORY_NAME
            / sha256[:2]
            / sha256
        )

    def _publish_cas_bytes(
        self,
        temporary: Path,
        target: Path,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> None:
        """把完整临时 CAS 文件原子发布为不可覆盖内容。"""

        _create_private_directory(target.parent)
        try:
            try:
                metadata = os.lstat(target)
            except FileNotFoundError:
                metadata = None
            if metadata is not None:
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise ResearchMaterialUnsafePath("CAS 目标必须是普通文件")
                if metadata.st_mode & 0o111:
                    raise ResearchMaterialIntegrityConflict("CAS 文件权限含 executable bit")
                _read_regular_file_bytes(
                    target,
                    missing_error=ResearchMaterialMissing,
                    integrity_error=ResearchMaterialIntegrityConflict,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    require_private_mode=True,
                )
                return
            # 在发布前再次核对临时普通文件，避免摘要计算后临时内容发生漂移。
            _read_regular_file_bytes(
                temporary,
                missing_error=ResearchMaterialPersistenceError,
                integrity_error=ResearchMaterialPersistenceError,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                require_private_mode=True,
            )
            os.link(temporary, target)
            _sync_directory(target.parent)
        except FileExistsError:
            # 同一内容并发上传只复用已经先发布的文件，绝不覆盖它。
            try:
                metadata = os.lstat(target)
            except FileNotFoundError as exc:
                raise ResearchMaterialPersistenceError("CAS 并发发布状态不确定") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ResearchMaterialUnsafePath("CAS 目标必须是普通文件")
            if metadata.st_mode & 0o111:
                raise ResearchMaterialIntegrityConflict("CAS 文件权限含 executable bit")
            _read_regular_file_bytes(
                target,
                missing_error=ResearchMaterialMissing,
                integrity_error=ResearchMaterialIntegrityConflict,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                require_private_mode=True,
            )
        except ResearchMaterialStoreError:
            raise
        except OSError as exc:
            raise ResearchMaterialPersistenceError("CAS 内容无法原子发布") from exc

    def _publish_workspace_bytes(
        self,
        target: Path,
        content: bytes,
        *,
        expected_size: int,
        expected_sha256: str,
        source_metadata: os.stat_result,
    ) -> None:
        """以同目录临时普通文件和不可覆盖链接原子复制到 workspace。"""

        try:
            metadata = os.lstat(target)
        except FileNotFoundError:
            metadata = None
        except OSError as exc:
            raise ResearchMaterialUnsafePath("workspace 目标无法安全核对") from exc
        if metadata is not None:
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ResearchMaterialUnsafePath("workspace 目标必须是普通文件")
            if metadata.st_mode & 0o111:
                raise ResearchMaterialUnsafePath("workspace 目标不能含 executable bit")
            if metadata.st_dev == source_metadata.st_dev and metadata.st_ino == source_metadata.st_ino:
                raise ResearchMaterialUnsafePath("workspace 目标不能直接链接 CAS 文件")
            existing, _ = _read_regular_file_bytes(
                target,
                missing_error=ResearchMaterialMissing,
                integrity_error=ResearchMaterialTargetConflict,
                expected_size=None,
                expected_sha256=None,
            )
            if len(existing) != expected_size or hashlib.sha256(existing).hexdigest() != expected_sha256:
                raise ResearchMaterialTargetConflict("workspace 目标内容发生漂移")
            try:
                os.chmod(target, _PRIVATE_FILE_MODE)
                _sync_directory(target.parent)
            except OSError as exc:
                raise ResearchMaterialPersistenceError("workspace 目标权限无法收紧") from exc
            return

        temporary_descriptor = -1
        temporary_path: Path | None = None
        try:
            temporary_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".material-copy.",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary_path = Path(temporary_name)
            os.fchmod(temporary_descriptor, _PRIVATE_FILE_MODE)
            with os.fdopen(temporary_descriptor, "wb") as output:
                temporary_descriptor = -1
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary_path, _PRIVATE_FILE_MODE)
            try:
                os.link(temporary_path, target)
            except FileExistsError:
                # 竞争者已发布时只接受完全相同的非可执行目标。
                try:
                    concurrent_metadata = os.lstat(target)
                except FileNotFoundError as exc:
                    raise ResearchMaterialPersistenceError(
                        "workspace 并发发布状态不确定"
                    ) from exc
                if stat.S_ISLNK(concurrent_metadata.st_mode) or not stat.S_ISREG(
                    concurrent_metadata.st_mode
                ):
                    raise ResearchMaterialUnsafePath("workspace 目标必须是普通文件")
                if concurrent_metadata.st_mode & 0o111:
                    raise ResearchMaterialUnsafePath("workspace 目标不能含 executable bit")
                existing, _ = _read_regular_file_bytes(
                    target,
                    missing_error=ResearchMaterialMissing,
                    integrity_error=ResearchMaterialTargetConflict,
                )
                if len(existing) != expected_size or hashlib.sha256(existing).hexdigest() != expected_sha256:
                    raise ResearchMaterialTargetConflict("workspace 目标内容发生漂移")
            else:
                _sync_directory(target.parent)
        except ResearchMaterialStoreError:
            raise
        except OSError as exc:
            raise ResearchMaterialPersistenceError("材料无法原子复制到 workspace") from exc
        finally:
            if temporary_descriptor >= 0:
                try:
                    os.close(temporary_descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


class _BytesReader:
    """把 bytes 适配为不暴露路径的内部二进制流。"""

    def __init__(self, content: bytes) -> None:
        self._stream = memoryview(content)
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        """返回下一段 bytes；只供 store 内部使用。"""

        if size < 0:
            size = len(self._stream) - self._offset
        start = self._offset
        self._offset = min(len(self._stream), start + size)
        return self._stream[start : self._offset].tobytes()


__all__ = [
    "MaterialDescriptor",
    "MaterialSizePolicy",
    "PerFileSizePolicy",
    "ResearchMaterialContractError",
    "ResearchMaterialCountLimitExceeded",
    "ResearchMaterialDescriptor",
    "ResearchMaterialDescriptorError",
    "ResearchMaterialDigestMismatch",
    "ResearchMaterialIntegrityConflict",
    "ResearchMaterialIntegrityError",
    "ResearchMaterialMissing",
    "ResearchMaterialMissingError",
    "ResearchMaterialMaterialCountLimitExceeded",
    "ResearchMaterialPersistenceError",
    "ResearchMaterialReadError",
    "ResearchMaterialSizeLimitExceeded",
    "ResearchMaterialSizePolicy",
    "ResearchMaterialStore",
    "ResearchMaterialStoreError",
    "ResearchMaterialTaskSizeLimitExceeded",
    "ResearchMaterialTargetConflict",
    "ResearchMaterialTargetDrift",
    "ResearchMaterialTotalSizeLimitExceeded",
    "ResearchMaterialUnsafePath",
]
