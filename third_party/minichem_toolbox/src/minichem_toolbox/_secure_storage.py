"""为 MiniChem 内部状态提供拒绝符号链接的安全文件操作。"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import os
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path


DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OPEN_FILE_FLAGS = os.O_CLOEXEC | os.O_NOFOLLOW


def _storage_error() -> ValueError:
    """返回不包含路径、密钥或其他内部值的统一文件系统错误。"""

    return ValueError("MiniChem internal storage path is invalid")


def lexical_absolute(path: str | Path) -> Path:
    """把路径规范化为绝对词法路径，不解析任何符号链接。"""

    try:
        value = os.fspath(path)
        if "\x00" in value:
            raise _storage_error()
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if ".." in candidate.parts:
            raise _storage_error()
        return Path(os.path.normpath(os.fspath(candidate)))
    except (OSError, RuntimeError, TypeError):
        raise _storage_error() from None


def _validate_component(name: str) -> None:
    """拒绝会改变 dirfd 相对语义的文件名组件。"""

    if not name or name in {".", ".."} or os.sep in name or "\x00" in name:
        raise _storage_error()
    if os.altsep and os.altsep in name:
        raise _storage_error()


def _open_root_directory() -> int:
    """打开文件系统根目录，作为逐组件打开的稳定起点。"""

    try:
        descriptor = os.open(os.sep, _OPEN_DIRECTORY_FLAGS)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(descriptor)
            raise _storage_error()
        return descriptor
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise _storage_error() from None


def _open_directory_chain(path: str | Path) -> int:
    """逐组件以 O_NOFOLLOW 打开已有目录，并返回末级目录 fd。"""

    absolute = lexical_absolute(path)
    descriptor = _open_root_directory()
    try:
        for component in absolute.parts[1:]:
            _validate_component(component)
            try:
                child = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise _storage_error() from None
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise _storage_error()
        return descriptor
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def validate_directory(path: str | Path) -> None:
    """验证目录及其所有已有父级没有符号链接。"""

    descriptor = _open_directory_chain(path)
    os.close(descriptor)


def ensure_directory(path: str | Path) -> None:
    """安全创建目录链，已有目录被收紧为 0700。"""

    absolute = lexical_absolute(path)
    descriptor = _open_root_directory()
    try:
        for component in absolute.parts[1:]:
            _validate_component(component)
            try:
                child = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise _storage_error() from None
                try:
                    os.mkdir(component, DIRECTORY_MODE, dir_fd=descriptor)
                except OSError as mkdir_error:
                    if mkdir_error.errno != errno.EEXIST:
                        raise _storage_error() from None
                try:
                    child = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=descriptor)
                except OSError:
                    raise _storage_error() from None
                os.fsync(descriptor)
            os.fchmod(child, DIRECTORY_MODE)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise _storage_error()
            os.close(descriptor)
            descriptor = child
        os.fchmod(descriptor, DIRECTORY_MODE)
        os.fsync(descriptor)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
    else:
        os.close(descriptor)


def _open_parent_directory(path: str | Path) -> tuple[int, str, Path]:
    """打开目标父目录并返回父 fd、末级名称和词法绝对路径。"""

    absolute = lexical_absolute(path)
    if len(absolute.parts) < 2:
        raise _storage_error()
    parent = Path(*absolute.parts[:-1])
    name = absolute.parts[-1]
    _validate_component(name)
    return _open_directory_chain(parent), name, absolute


def _assert_directory_identity(path: str | Path, descriptor: int) -> None:
    """确认路径链仍指向刚才使用的同一个父目录 inode。"""

    metadata = os.fstat(descriptor)
    current = _open_directory_chain(path)
    try:
        current_metadata = os.fstat(current)
        if (metadata.st_dev, metadata.st_ino) != (
            current_metadata.st_dev,
            current_metadata.st_ino,
        ):
            raise _storage_error()
    finally:
        os.close(current)


def _assert_target(
    descriptor: int,
    name: str,
    *,
    allow_missing: bool = True,
    require_regular: bool = True,
) -> None:
    """验证目标项不是符号链接，且类型符合内部文件预期。"""

    _validate_component(name)
    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if allow_missing:
            return
        raise _storage_error() from None
    except OSError:
        raise _storage_error() from None
    if stat.S_ISLNK(metadata.st_mode):
        raise _storage_error()
    if require_regular and not stat.S_ISREG(metadata.st_mode):
        raise _storage_error()


def _assert_file_identity(parent: int, name: str, descriptor: int) -> None:
    """确认锁路径仍指向已经打开的同一个普通文件。"""

    try:
        path_metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
        descriptor_metadata = os.fstat(descriptor)
    except OSError:
        raise _storage_error() from None
    if not stat.S_ISREG(path_metadata.st_mode) or (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ) != (descriptor_metadata.st_dev, descriptor_metadata.st_ino):
        raise _storage_error()


def validate_existing_file(path: str | Path, *, allow_missing: bool = True) -> None:
    """验证文件目标及其父级安全；不读取文件内容。"""

    parent, name, _ = _open_parent_directory(path)
    try:
        _assert_target(parent, name, allow_missing=allow_missing)
    finally:
        os.close(parent)


def ensure_private_file(path: str | Path, *, allow_missing: bool = True) -> None:
    """把已有内部文件收紧为 0600，并验证它仍是普通文件。"""

    parent, name, _ = _open_parent_directory(path)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | _OPEN_FILE_FLAGS,
                dir_fd=parent,
            )
        except FileNotFoundError:
            if allow_missing:
                return
            raise _storage_error() from None
        except OSError:
            raise _storage_error() from None
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _storage_error()
        os.fchmod(descriptor, FILE_MODE)
        os.fsync(descriptor)
        os.fsync(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def read_regular_file_snapshot(path: str | Path) -> tuple[bytes, str]:
    """从一个 O_NOFOLLOW 文件描述符读取字节，并返回同一快照的摘要。"""

    parent, name, _ = _open_parent_directory(path)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | _OPEN_FILE_FLAGS,
                dir_fd=parent,
            )
        except OSError:
            raise _storage_error() from None
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _storage_error()
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        return b"".join(chunks), digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def read_file_in_directory(
    directory: str | Path,
    name: str,
    *,
    allow_missing: bool = False,
) -> bytes:
    """从已验证目录中的普通文件读取字节，不跟随目标符号链接。"""

    _validate_component(name)
    descriptor = _open_directory_chain(directory)
    file_descriptor = -1
    try:
        try:
            file_descriptor = os.open(
                name,
                os.O_RDONLY | _OPEN_FILE_FLAGS,
                dir_fd=descriptor,
            )
        except FileNotFoundError:
            if allow_missing:
                return b""
            raise _storage_error() from None
        except OSError:
            raise _storage_error() from None
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _storage_error()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(descriptor)


def cleanup_temporary_files(directory: str | Path) -> None:
    """删除本工具生成的崩溃遗留临时项，只操作当前目录项本身。"""

    descriptor = _open_directory_chain(directory)
    try:
        try:
            names = os.listdir(descriptor)
        except OSError:
            raise _storage_error() from None
        for name in names:
            if not name.startswith(".tmp-"):
                continue
            try:
                os.unlink(name, dir_fd=descriptor)
            except FileNotFoundError:
                continue
            except OSError:
                raise _storage_error() from None
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def directory_lock(directory: str | Path, lock_name: str = ".lock") -> Iterator[int]:
    """在安全目录中取得跨进程排他锁，并返回锁定目录 fd。"""

    _validate_component(lock_name)
    descriptor = _open_directory_chain(directory)
    lock_descriptor = -1
    try:
        try:
            lock_descriptor = os.open(
                lock_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | _OPEN_FILE_FLAGS,
                FILE_MODE,
                dir_fd=descriptor,
            )
        except FileExistsError:
            try:
                lock_descriptor = os.open(
                    lock_name,
                    os.O_RDWR | _OPEN_FILE_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError:
                raise _storage_error() from None
        except OSError:
            raise _storage_error() from None
        os.fchmod(lock_descriptor, FILE_MODE)
        lock_metadata = os.fstat(lock_descriptor)
        if not stat.S_ISREG(lock_metadata.st_mode):
            raise _storage_error()
        os.fsync(lock_descriptor)
        os.fsync(descriptor)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        except OSError:
            raise _storage_error() from None
        _assert_file_identity(descriptor, lock_name, lock_descriptor)
        _assert_directory_identity(directory, descriptor)
        yield descriptor
    finally:
        if lock_descriptor >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(lock_descriptor)
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    """完整写入给定字节串，处理短写。"""

    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise _storage_error()
        offset += written


def atomic_write_bytes(
    directory: str | Path,
    name: str,
    payload: bytes,
    *,
    prefix: str = ".tmp-",
) -> None:
    """在目录中以私有临时文件和原子替换写入完整字节串。"""

    _validate_component(name)
    _validate_component(prefix + "probe")
    descriptor = _open_directory_chain(directory)
    temporary_name = f"{prefix}{uuid.uuid4().hex}"
    temporary_descriptor = -1
    replaced = False
    try:
        _assert_directory_identity(directory, descriptor)
        _assert_target(descriptor, name, allow_missing=True)
        try:
            temporary_descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | _OPEN_FILE_FLAGS,
                FILE_MODE,
                dir_fd=descriptor,
            )
        except OSError:
            raise _storage_error() from None
        os.fchmod(temporary_descriptor, FILE_MODE)
        _write_all(temporary_descriptor, payload)
        os.fsync(temporary_descriptor)
        temporary_metadata = os.fstat(temporary_descriptor)
        if not stat.S_ISREG(temporary_metadata.st_mode) or stat.S_IMODE(
            temporary_metadata.st_mode
        ) != FILE_MODE:
            raise _storage_error()
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        _assert_directory_identity(directory, descriptor)
        _assert_target(descriptor, name, allow_missing=True)
        try:
            os.replace(
                temporary_name,
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
        except OSError:
            raise _storage_error() from None
        replaced = True
        os.fsync(descriptor)
        _assert_directory_identity(directory, descriptor)
        _assert_target(descriptor, name, allow_missing=False)
        try:
            target_descriptor = os.open(
                name,
                os.O_RDONLY | _OPEN_FILE_FLAGS,
                dir_fd=descriptor,
            )
        except OSError:
            raise _storage_error() from None
        try:
            target_metadata = os.fstat(target_descriptor)
            if not stat.S_ISREG(target_metadata.st_mode) or stat.S_IMODE(
                target_metadata.st_mode
            ) != FILE_MODE:
                raise _storage_error()
        finally:
            os.close(target_descriptor)
    finally:
        if temporary_descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(temporary_descriptor)
        if not replaced:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name, dir_fd=descriptor)
        with contextlib.suppress(OSError):
            os.close(descriptor)
