"""Investigation/Experiment 任务级 Review 的只读证据快照。

外层 authenticated MiniChem verifier 先核对原始 source artifact ref 与期望 SHA，
Controller 再登记路径并用独立的 Evidence reader artifact ID 创建一至五项 pair 快照；
Review MCP 只读 snapshot/Evidence artifact ID。公开 manifest 不含宿主路径；每次读取
重验 size/SHA-256，并限制分页、媒体类型、跨快照归属和当前 reader 进程内累计字符预算。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


__all__ = [
    "EvidenceArtifactManifest",
    "ResultEvidencePair",
    "BatchEvidenceSnapshot",
    "EvidenceReadResult",
    "EvidenceRequestError",
    "EvidenceRegistryCorruption",
    "EvidenceSnapshotNotFound",
    "EvidenceArtifactNotFound",
    "EvidenceArtifactIntegrityError",
    "EvidenceBudgetExceeded",
    "UnsupportedEvidenceArtifact",
    "ReviewEvidenceStore",
    "read_snapshot_artifact",
]


_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
_DEFAULT_MAX_SINGLE_READ_CHARS = 8_192
_DEFAULT_MAX_TOTAL_READ_CHARS = 32_768
_DEFAULT_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_ARTIFACTS = 64
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{7,127}$")
_SOURCE_ARTIFACT_REF_RE = re.compile(r"^art_[0-9a-f]{32}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE_TOKEN_RE = re.compile(r"^[a-z0-9!#$&^_.+*-]+$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PATH_MARKER_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]{1,31}://|"
    r"(?:^|[\s])/(?:tmp|mnt|home|root|var|etc|opt)(?:/|$)|"
    r"(?:^|[\s])[A-Za-z]:[\\/]|"
    r"(?:^|[\s])\.\.?(?:[\\/]|$))",
    re.IGNORECASE,
)
_KNOWN_TEXT_SUFFIXES: dict[str, str] = {
    ".txt": "text/plain",
    ".text": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".log": "text/plain",
    ".out": "text/plain",
    ".gjf": "text/plain",
    ".com": "text/plain",
    ".sdf": "chemical/x-mdl-sdfile",
    ".xyz": "chemical/x-xyz",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".ndjson": "application/x-ndjson",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}

# 这是 reader 进程内的预算，而不是实例字段。两个 store 实例读取同一快照时
# 共享计数，但重新启动 Python 后该表自然清空。
_PROCESS_BUDGET_LOCK = threading.RLock()
_PROCESS_BUDGETS: dict[tuple[str, str], int] = {}


class EvidenceRequestError(ValueError):
    """调用方提交了不符合只读证据 interface 的参数。"""


class EvidenceRegistryCorruption(RuntimeError):
    """权限收紧的内部注册表缺失、损坏或无法通过摘要检查。"""


class EvidenceSnapshotNotFound(EvidenceRequestError):
    """请求的 snapshot_id 不存在。"""


class EvidenceArtifactNotFound(EvidenceRequestError):
    """请求的 artifact_id 不存在或不属于指定快照。"""


class EvidenceArtifactIntegrityError(RuntimeError):
    """已登记来源在读取时发生变化或无法核对。"""


class EvidenceBudgetExceeded(EvidenceRequestError):
    """同一 reader 进程对该快照的累计字符预算已经用尽。"""


class UnsupportedEvidenceArtifact(EvidenceRequestError):
    """来源不是第一版支持的可验证 UTF-8 文本。"""


@dataclass(frozen=True)
class EvidenceArtifactManifest:
    """Evidence reader 内容绑定摘要及其已核对的 MiniChem 源身份。

    ``artifact_id`` 是 Evidence reader 自己的内容绑定 ID，``source_artifact_ref``
    是外层 authenticated MiniChem verifier 已核对的原始科学产物引用；两者
    属于不同命名空间，不能互换。``sha256`` 始终是 reader 实际读取 UTF-8
    字节的摘要。
    """

    artifact_id: str
    source_artifact_ref: str
    size: int
    sha256: str
    media_type: str

    def to_public_dict(self) -> dict[str, Any]:
        """返回不含路径的独立 JSON object。"""

        return {
            "artifact_id": self.artifact_id,
            "source_artifact_ref": self.source_artifact_ref,
            "size": self.size,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class ResultEvidencePair:
    """一个结果版本及其专属 artifact 摘要；artifact 可为空。"""

    combination_id: str
    result_version: str
    artifacts: tuple[EvidenceArtifactManifest, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        """返回 pair 级公开 JSON object，不含来源路径。"""

        return {
            "combination_id": self.combination_id,
            "result_version": self.result_version,
            "artifacts": [item.to_public_dict() for item in self.artifacts],
        }


@dataclass(frozen=True)
class BatchEvidenceSnapshot:
    """不可变任务级快照；当前批次为一至五个 pair，pair 内 artifact 可为空。"""

    snapshot_id: str
    task_id: str
    batch_id: str
    result_pairs: tuple[ResultEvidencePair, ...]

    def to_public_manifest(self) -> dict[str, Any]:
        """返回精简的 path-free manifest。"""

        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "result_pairs": [item.to_public_dict() for item in self.result_pairs],
        }


@dataclass(frozen=True)
class EvidenceReadResult:
    """一次有界文本读取的内容、截断状态和下一页游标。offset 是字符偏移。"""

    snapshot_id: str
    artifact_id: str
    offset: int
    max_chars: int
    content: str
    total_chars: int
    truncated: bool
    next_offset: int | None
    budget_remaining: int
    json_pointer: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """返回可直接作为唯一 Review MCP read tool 结果的 JSON object。"""

        cursor = None
        if self.next_offset is not None:
            cursor = {
                "snapshot_id": self.snapshot_id,
                "artifact_id": self.artifact_id,
                "offset": self.next_offset,
                "json_pointer": self.json_pointer,
            }
        return {
            "snapshot_id": self.snapshot_id,
            "artifact_id": self.artifact_id,
            "json_pointer": self.json_pointer,
            "offset": self.offset,
            "max_chars": self.max_chars,
            "content": self.content,
            "content_chars": len(self.content),
            "total_chars": self.total_chars,
            "truncated": self.truncated,
            "has_more": self.next_offset is not None,
            "next_offset": self.next_offset,
            "cursor": cursor,
            "budget_remaining": self.budget_remaining,
        }


@dataclass(frozen=True)
class _RegisteredArtifact:
    """内部注册项；唯一含宿主路径的对象。"""

    artifact_id: str
    source_artifact_ref: str
    source_path: str
    source_path_digest: str
    size: int
    sha256: str
    media_type: str

    def to_manifest(self) -> EvidenceArtifactManifest:
        """去掉内部路径，形成公开摘要。"""

        return EvidenceArtifactManifest(
            artifact_id=self.artifact_id,
            source_artifact_ref=self.source_artifact_ref,
            size=self.size,
            sha256=self.sha256,
            media_type=self.media_type,
        )

    def to_registry_dict(self) -> dict[str, Any]:
        """返回只供 0600 注册表保存的字段。"""

        return {
            "artifact_id": self.artifact_id,
            "source_artifact_ref": self.source_artifact_ref,
            "source_path": self.source_path,
            "source_path_digest": self.source_path_digest,
            "size": self.size,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


class ReviewEvidenceStore:
    """管理来源登记、一至五项 pair 快照和只读文本读取。

    创建只接受 ``{(combination_id, result_version): [manifest_or_id, ...]}``，
    不接受路径；每个 pair 可为空，同一快照内 source ref 和 Evidence artifact ID
    都不跨 pair 复用。registry
    与 lock 为 0600，目录为 0700，并用 POSIX 文件锁支持多个 store 实例。累计预算
    按目录/快照在当前 Python 进程共享，重启后不持久化。
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        *,
        max_single_read_chars: int = _DEFAULT_MAX_SINGLE_READ_CHARS,
        max_total_read_chars: int = _DEFAULT_MAX_TOTAL_READ_CHARS,
        max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
        max_artifacts_per_snapshot: int = _DEFAULT_MAX_ARTIFACTS,
    ) -> None:
        """创建 store 并准备权限收紧的注册表。

        输入是专用目录和正整数资源上限；动作只创建 store 的内部文件，不触碰
        来源 artifact。配置非法、目录不可用或内部注册表损坏时抛出异常；返回值
        无。累计预算仅在当前 reader 进程内有效。
        """

        self.root_dir = Path(root_dir).expanduser().resolve(strict=False)
        self._validate_positive_limit(max_single_read_chars, "max_single_read_chars")
        self._validate_positive_limit(max_total_read_chars, "max_total_read_chars")
        self._validate_positive_limit(max_artifact_bytes, "max_artifact_bytes")
        self._validate_positive_limit(max_artifacts_per_snapshot, "max_artifacts_per_snapshot")
        if max_single_read_chars > max_total_read_chars:
            raise EvidenceRequestError("单次读取上限不能大于累计读取上限")
        self.max_single_read_chars = max_single_read_chars
        self.max_total_read_chars = max_total_read_chars
        self.max_artifact_bytes = max_artifact_bytes
        self.max_artifacts_per_snapshot = max_artifacts_per_snapshot
        self._lock = threading.RLock()
        self._prepare_store_directory()

    @staticmethod
    def _validate_positive_limit(value: int, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvidenceRequestError(f"{label} 必须是正整数")

    @property
    def _registry_path(self) -> Path:
        return self.root_dir / "registry.json"

    @property
    def _lock_path(self) -> Path:
        return self.root_dir / "registry.lock"

    def _prepare_store_directory(self) -> None:
        try:
            missing: list[Path] = []
            candidate = self.root_dir
            while not candidate.exists():
                missing.append(candidate)
                candidate = candidate.parent
            candidate.mkdir(parents=True, exist_ok=True)
            for directory in reversed(missing):
                directory.mkdir(exist_ok=True)
                os.chmod(directory, 0o700)
            if self.root_dir.is_symlink():
                raise EvidenceRegistryCorruption("证据存储目录不能是符号链接")
            os.chmod(self.root_dir, 0o700)
            self._ensure_private_file(self._lock_path)
            with self._registry_file_lock():
                if self._registry_path.exists():
                    if self._registry_path.is_symlink() or not self._registry_path.is_file():
                        raise EvidenceRegistryCorruption("内部注册表不是普通文件")
                    os.chmod(self._registry_path, 0o600)
                    self._load_registry()
                else:
                    self._write_registry(
                        {"schema_version": _SCHEMA_VERSION, "artifacts": {}, "snapshots": {}}
                    )
        except EvidenceRegistryCorruption:
            raise
        except OSError as exc:
            raise EvidenceRegistryCorruption("无法准备内部证据注册表") from exc

    @staticmethod
    def _ensure_private_file(path: Path) -> None:
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise EvidenceRegistryCorruption("内部锁文件不是普通文件")
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)

    @contextmanager
    def _registry_file_lock(self) -> Iterator[None]:
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise EvidenceRegistryCorruption("无法锁定内部证据注册表") from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _load_registry(self) -> dict[str, Any]:
        try:
            with self._registry_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise EvidenceRegistryCorruption("内部证据注册表无法读取") from exc
        if not isinstance(value, dict):
            raise EvidenceRegistryCorruption("内部证据注册表格式非法")
        schema_version = value.get("schema_version")
        if schema_version == _LEGACY_SCHEMA_VERSION:
            raise EvidenceRegistryCorruption(
                "registry schema v1 无法安全推出 source_artifact_ref；必须显式重建"
            )
        if schema_version != _SCHEMA_VERSION:
            raise EvidenceRegistryCorruption("内部证据注册表版本不支持")
        if not isinstance(value.get("artifacts"), dict) or not isinstance(
            value.get("snapshots"), dict
        ):
            raise EvidenceRegistryCorruption("内部证据注册表缺少索引")
        return value

    def _write_registry(self, value: Mapping[str, Any]) -> None:
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root_dir,
                prefix=".registry-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                os.chmod(handle.fileno(), 0o600)
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self._registry_path)
            temporary_name = None
            os.chmod(self._registry_path, 0o600)
        except OSError as exc:
            raise EvidenceRegistryCorruption("内部证据注册表无法保存") from exc
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass

    @staticmethod
    def _validate_safe_text(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise EvidenceRequestError(f"{label} 必须是非空字符串")
        normalized = value.strip()
        if _CONTROL_RE.search(normalized) or _PATH_MARKER_RE.search(normalized):
            raise EvidenceRequestError(f"{label} 不能包含路径或控制字符")
        if "/" in normalized or "\\" in normalized:
            raise EvidenceRequestError(f"{label} 不能包含路径分隔符")
        return normalized

    @classmethod
    def _validate_opaque_id(cls, value: Any, label: str) -> str:
        if not isinstance(value, str) or not _OPAQUE_ID_RE.fullmatch(value):
            raise EvidenceRequestError(f"{label} 必须是不透明标识")
        return value

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @classmethod
    def _derive_artifact_id(
        cls,
        source_artifact_ref: str,
        source_path_digest: str,
        size: int,
        sha256: str,
        media_type: str,
    ) -> str:
        payload = cls._canonical_json(
            {
                "source_artifact_ref": source_artifact_ref,
                "source_path_digest": source_path_digest,
                "size": size,
                "sha256": sha256,
                "media_type": media_type,
            }
        ).encode("utf-8")
        return "art_" + cls._sha256_bytes(payload)[:32]

    @classmethod
    def _derive_snapshot_id(
        cls,
        task_id: str,
        batch_id: str,
        result_pairs: Sequence[ResultEvidencePair],
    ) -> str:
        payload = cls._canonical_json(
            {
                "task_id": task_id,
                "batch_id": batch_id,
                "result_pairs": [item.to_public_dict() for item in result_pairs],
            }
        ).encode("utf-8")
        return "snap_" + cls._sha256_bytes(payload)[:48]

    @staticmethod
    def _source_path_digest(source_path: str) -> str:
        return hashlib.sha256(source_path.encode("utf-8", "surrogatepass")).hexdigest()

    @classmethod
    def _validate_source_artifact_ref(cls, value: Any, label: str) -> str:
        """校验外层 verifier 提供的 MiniChem 原始 artifact 引用。

        Evidence store 不调用 MiniChem，也不扫描 workspace；它只接受外层已完成
        归属核对的 ``art_<32hex>`` 身份，并把该身份原样绑定到 reader manifest。
        缺失、路径形式或其它命名空间的引用都 fail-closed。
        """

        if not isinstance(value, str) or _SOURCE_ARTIFACT_REF_RE.fullmatch(value) is None:
            raise EvidenceRequestError(
                f"{label} 必须是已由外层 verifier 核对的 MiniChem artifact ref"
            )
        return value

    @staticmethod
    def _validate_expected_source_sha256(value: Any, label: str) -> str:
        """校验外层 verifier 给出的源字节 SHA-256，要求精确小写十六进制。"""

        if not isinstance(value, str) or _HEX_RE.fullmatch(value) is None:
            raise EvidenceRequestError(f"{label} 必须是 64 位小写 SHA-256")
        return value

    @staticmethod
    def _normalize_media_type(media_type: str) -> str:
        if not isinstance(media_type, str) or not media_type.strip():
            raise UnsupportedEvidenceArtifact("媒体类型未知")
        base = media_type.strip().lower().split(";", 1)[0].strip()
        pieces = base.split("/", 1)
        if len(pieces) != 2 or not all(_MEDIA_TYPE_TOKEN_RE.fullmatch(piece) for piece in pieces):
            raise UnsupportedEvidenceArtifact("媒体类型格式不受支持")
        if not ReviewEvidenceStore._is_supported_text_media_type(base):
            raise UnsupportedEvidenceArtifact("只支持可验证文本媒体")
        return base

    @staticmethod
    def _is_supported_text_media_type(media_type: str) -> bool:
        if media_type.startswith("text/"):
            return True
        return media_type in {
            "application/json",
            "application/jsonl",
            "application/x-ndjson",
            "application/xml",
            "application/yaml",
            "application/x-yaml",
            "chemical/sdf",
            "chemical/x-mdl-sd",
            "chemical/x-mdl-sdfile",
            "chemical/x-sdf",
            "chemical/x-xyz",
        } or media_type.endswith("+json")

    @classmethod
    def _infer_media_type(cls, source_path: Path) -> str:
        media_type = _KNOWN_TEXT_SUFFIXES.get(source_path.suffix.lower())
        if media_type is None or not cls._is_supported_text_media_type(media_type):
            raise UnsupportedEvidenceArtifact("来源媒体类型未知或不是文本")
        return media_type

    @staticmethod
    def _validate_text_content(content: bytes) -> str:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedEvidenceArtifact("来源不是 UTF-8 文本") from exc
        if "\x00" in text or _CONTROL_RE.search(text):
            raise UnsupportedEvidenceArtifact("来源包含二进制或不可读控制字符")
        return text

    def _resolve_source_path(self, source_path: str | os.PathLike[str]) -> Path:
        try:
            candidate = Path(source_path).expanduser().resolve(strict=True)
            if not candidate.is_file():
                raise UnsupportedEvidenceArtifact("来源必须是普通文件")
            if candidate in {
                self._registry_path.resolve(strict=False),
                self._lock_path.resolve(strict=False),
            }:
                raise UnsupportedEvidenceArtifact("内部控制文件不能作为来源 artifact")
            return candidate
        except UnsupportedEvidenceArtifact:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise UnsupportedEvidenceArtifact("来源路径无法解析") from exc

    def _read_source_for_registration(self, source_path: Path) -> bytes:
        try:
            before = source_path.stat()
            if not source_path.is_file() or before.st_size > self.max_artifact_bytes:
                raise UnsupportedEvidenceArtifact("来源不是受支持的普通文本文件")
            with source_path.open("rb") as handle:
                content = handle.read(self.max_artifact_bytes + 1)
            after = source_path.stat()
        except UnsupportedEvidenceArtifact:
            raise
        except OSError as exc:
            raise UnsupportedEvidenceArtifact("来源文件无法读取") from exc
        if len(content) > self.max_artifact_bytes or before.st_size != after.st_size:
            raise UnsupportedEvidenceArtifact("来源文件超过大小上限或正在变化")
        if len(content) != after.st_size:
            raise UnsupportedEvidenceArtifact("来源文件大小无法核对")
        return content

    def register_source_artifact(
        self,
        source_path: str | os.PathLike[str],
        *,
        source_artifact_ref: str,
        expected_source_sha256: str,
        media_type: str | None = None,
    ) -> EvidenceArtifactManifest:
        """登记一个已完成外层身份核对的 UTF-8 文本来源。

        输入是注册阶段唯一允许出现的本机路径、MiniChem 原始
        ``source_artifact_ref`` 和外层 verifier 提供的 ``expected_source_sha256``，
        可选媒体类型；动作读取实际文件并要求其 UTF-8 字节 SHA-256 与期望值
        完全相同，再把路径和源绑定写入 0600 registry。Evidence store 不调用
        MiniChem、不扫描 workspace，也不复制、执行、删除或修改来源。返回的
        ``artifact_id`` 是 Evidence reader 的独立内容绑定 ID，公开摘要不含路径。
        缺少源身份、摘要错配、二进制、超限或来源变化都会 fail-closed。
        """

        with self._lock, self._registry_file_lock():
            normalized_source_ref = self._validate_source_artifact_ref(
                source_artifact_ref,
                "source_artifact_ref",
            )
            normalized_expected_sha256 = self._validate_expected_source_sha256(
                expected_source_sha256,
                "expected_source_sha256",
            )
            resolved = self._resolve_source_path(source_path)
            normalized_media_type = (
                self._infer_media_type(resolved)
                if media_type is None
                else self._normalize_media_type(media_type)
            )
            content = self._read_source_for_registration(resolved)
            self._validate_text_content(content)
            size = len(content)
            digest = self._sha256_bytes(content)
            if digest != normalized_expected_sha256:
                raise EvidenceArtifactIntegrityError(
                    "来源文件摘要与 expected_source_sha256 不一致"
                )
            source_text = str(resolved)
            source_digest = self._source_path_digest(source_text)
            artifact_id = self._derive_artifact_id(
                normalized_source_ref,
                source_digest,
                size,
                digest,
                normalized_media_type,
            )
            if artifact_id == normalized_source_ref:
                raise EvidenceRegistryCorruption(
                    "MiniChem source ref 与 Evidence artifact ID 不能相同"
                )
            record = _RegisteredArtifact(
                artifact_id,
                normalized_source_ref,
                source_text,
                source_digest,
                size,
                digest,
                normalized_media_type,
            )
            registry = self._load_registry()
            existing = registry["artifacts"].get(artifact_id)
            if existing is not None:
                if existing != record.to_registry_dict():
                    raise EvidenceRegistryCorruption("artifact 注册项与摘要不一致")
                return record.to_manifest()
            registry["artifacts"][artifact_id] = record.to_registry_dict()
            self._write_registry(registry)
            return record.to_manifest()

    def read_source_artifact_manifest(
        self,
        source_artifact_ref: str,
    ) -> EvidenceArtifactManifest:
        """按 MiniChem source ref 读取已登记的无路径 manifest。

        输入是已经由外层 verifier 核对并登记的 ``source_artifact_ref``。动作只读取
        持久 registry，重新核对内部记录和来源文件摘要，不返回宿主路径。找到零项时
        抛出 ``EvidenceArtifactNotFound``；同一 source ref 对应多项不同登记时抛出
        ``EvidenceRegistryCorruption``，不猜测 Review 应使用哪一项。
        """

        with self._lock, self._registry_file_lock():
            normalized_source_ref = self._validate_source_artifact_ref(
                source_artifact_ref,
                "source_artifact_ref",
            )
            registry = self._load_registry()
            matches = [
                artifact_id
                for artifact_id, raw in registry["artifacts"].items()
                if isinstance(raw, Mapping)
                and raw.get("source_artifact_ref") == normalized_source_ref
            ]
            if not matches:
                raise EvidenceArtifactNotFound("找不到请求的 source artifact")
            if len(matches) != 1:
                raise EvidenceRegistryCorruption(
                    "同一 source_artifact_ref 对应多个 Evidence artifact"
                )
            record = self._load_registered_artifact(registry, matches[0])
            self._read_and_verify_registered_artifact(record)
            return record.to_manifest()

    def _load_registered_artifact(
        self,
        registry: Mapping[str, Any],
        artifact_id: str,
    ) -> _RegisteredArtifact:
        raw = registry.get("artifacts", {}).get(artifact_id)
        if not isinstance(raw, Mapping):
            raise EvidenceArtifactNotFound("找不到请求的 artifact")
        record = self._build_registered_artifact_record(raw)
        self._validate_registered_artifact_fields(record)
        self._validate_registered_artifact_binding(record, artifact_id)
        return record

    @staticmethod
    def _build_registered_artifact_record(raw: Mapping[str, Any]) -> _RegisteredArtifact:
        """从 registry mapping 构造内部 artifact 记录，统一字段缺失错误。"""

        try:
            return _RegisteredArtifact(
                str(raw["artifact_id"]),
                raw["source_artifact_ref"],
                raw["source_path"],
                raw["source_path_digest"],
                raw["size"],
                raw["sha256"],
                raw["media_type"],
            )
        except (KeyError, TypeError) as exc:
            raise EvidenceRegistryCorruption("artifact 注册项字段不完整") from exc

    @staticmethod
    def _validate_registered_artifact_fields(record: _RegisteredArtifact) -> None:
        """核对 registry artifact 的字段类型、摘要格式和路径形状。"""

        if (
            not isinstance(record.source_path, str)
            or not os.path.isabs(record.source_path)
            or isinstance(record.size, bool)
            or not isinstance(record.size, int)
            or record.size < 0
            or not isinstance(record.sha256, str)
            or not _HEX_RE.fullmatch(record.sha256)
            or not isinstance(record.source_path_digest, str)
            or not _HEX_RE.fullmatch(record.source_path_digest)
            or not isinstance(record.source_artifact_ref, str)
            or _SOURCE_ARTIFACT_REF_RE.fullmatch(record.source_artifact_ref) is None
            or not isinstance(record.media_type, str)
        ):
            raise EvidenceRegistryCorruption("artifact 注册项摘要非法")

    def _validate_registered_artifact_binding(
        self,
        record: _RegisteredArtifact,
        artifact_id: str,
    ) -> None:
        """核对 artifact ID、来源路径摘要、媒体类型和派生摘要。"""

        if record.artifact_id != artifact_id:
            raise EvidenceRegistryCorruption("artifact 标识不一致")
        if self._source_path_digest(record.source_path) != record.source_path_digest:
            raise EvidenceRegistryCorruption("artifact 来源摘要不一致")
        if not self._is_supported_text_media_type(record.media_type):
            raise EvidenceRegistryCorruption("artifact 媒体类型不受支持")
        if record.artifact_id != self._derive_artifact_id(
            record.source_artifact_ref,
            record.source_path_digest,
            record.size,
            record.sha256,
            record.media_type,
        ):
            raise EvidenceRegistryCorruption("artifact 标识摘要不一致")
        if record.artifact_id == record.source_artifact_ref:
            raise EvidenceRegistryCorruption(
                "MiniChem source ref 与 Evidence artifact ID 不能相同"
            )

    @classmethod
    def _normalize_result_vector(cls, value: Any) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise EvidenceRequestError("result_vector 必须是一至五项序列")
        normalized: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, Mapping):
                combination_id = item.get("combination_id")
                result_version = item.get("result_version")
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                if len(item) != 2:
                    raise EvidenceRequestError("result_vector 项必须含组合和版本")
                combination_id, result_version = item
            else:
                raise EvidenceRequestError("result_vector 项格式非法")
            normalized.append(
                (
                    cls._validate_safe_text(combination_id, "combination_id"),
                    cls._validate_safe_text(result_version, "result_version"),
                )
            )
        if not 1 <= len(normalized) <= 5:
            raise EvidenceRequestError("Review 快照必须绑定一至五项结果版本")
        if len({item[0] for item in normalized}) != len(normalized):
            raise EvidenceRequestError("Review 快照的 combination_id 必须唯一")
        return tuple(sorted(normalized))

    def _parse_public_artifact(
        self,
        value: EvidenceArtifactManifest | str,
        registry: Mapping[str, Any],
    ) -> _RegisteredArtifact:
        if isinstance(value, EvidenceArtifactManifest):
            artifact_id = self._validate_opaque_id(value.artifact_id, "artifact_id")
            record = self._load_registered_artifact(registry, artifact_id)
            if record.to_manifest() != value:
                raise EvidenceArtifactIntegrityError("artifact manifest 与注册项不一致")
            self._read_and_verify_registered_artifact(record)
            return record
        if isinstance(value, str):
            artifact_id = self._validate_opaque_id(value, "artifact_id")
            record = self._load_registered_artifact(registry, artifact_id)
            self._read_and_verify_registered_artifact(record)
            return record
        raise EvidenceRequestError("artifact_groups 只接受已登记 manifest 或 opaque ID")

    def _resolve_artifact_groups(
        self,
        result_pairs: set[tuple[str, str]],
        artifact_groups: Any,
        registry: Mapping[str, Any],
    ) -> dict[tuple[str, str], tuple[_RegisteredArtifact, ...]]:
        """解析当前批次 pair 的 artifact 分组并核对跨 pair 命名空间隔离。"""

        normalized_keys = self._normalize_artifact_group_keys(artifact_groups)
        if set(normalized_keys) != result_pairs:
            raise EvidenceRequestError("artifact_groups 必须完整覆盖当前 result pair")
        resolved: dict[tuple[str, str], tuple[_RegisteredArtifact, ...]] = {}
        all_evidence_ids: set[str] = set()
        all_source_refs: set[str] = set()
        total_count = 0
        for result_pair in sorted(result_pairs):
            records = self._resolve_artifact_group_records(
                normalized_keys[result_pair],
                registry,
                all_evidence_ids,
                all_source_refs,
            )
            total_count += len(records)
            if total_count > self.max_artifacts_per_snapshot:
                raise EvidenceRequestError("快照 artifact 数量超过上限")
            resolved[result_pair] = tuple(sorted(records, key=lambda item: item.artifact_id))
        return resolved

    def _normalize_artifact_group_keys(self, artifact_groups: Any) -> dict[tuple[str, str], Any]:
        """校验 artifact_groups 的 mapping、pair key 和完整 key 唯一性。"""

        if not isinstance(artifact_groups, Mapping):
            raise EvidenceRequestError(
                "artifact_groups 必须是 {(combination_id, result_version): [manifest_or_id, ...]}"
            )
        normalized_keys: dict[tuple[str, str], Any] = {}
        for key, values in artifact_groups.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or isinstance(key[0], (bytes, bytearray))
                or isinstance(key[1], (bytes, bytearray))
            ):
                raise EvidenceRequestError(
                    "artifact_groups 的 key 必须是 (combination_id, result_version)"
                )
            normalized_key = (
                self._validate_safe_text(key[0], "combination_id"),
                self._validate_safe_text(key[1], "result_version"),
            )
            if normalized_key in normalized_keys:
                raise EvidenceRequestError("artifact_groups 不能包含重复 result pair")
            normalized_keys[normalized_key] = values
        return normalized_keys

    def _resolve_artifact_group_records(
        self,
        values: Any,
        registry: Mapping[str, Any],
        all_evidence_ids: set[str],
        all_source_refs: set[str],
    ) -> list[_RegisteredArtifact]:
        """解析一个 pair 的 artifact 列表并更新跨 pair 使用集合。"""

        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise EvidenceRequestError("每个 artifact_groups 值必须是 manifest/ID 序列")
        records: list[_RegisteredArtifact] = []
        for value in values:
            record = self._parse_public_artifact(value, registry)
            self._validate_artifact_group_reuse(record, all_evidence_ids, all_source_refs)
            all_evidence_ids.add(record.artifact_id)
            all_source_refs.add(record.source_artifact_ref)
            records.append(record)
        return records

    @staticmethod
    def _validate_artifact_group_reuse(
        record: _RegisteredArtifact,
        all_evidence_ids: set[str],
        all_source_refs: set[str],
    ) -> None:
        """拒绝 artifact ID/source ref 重复或跨命名空间复用。"""

        if record.artifact_id in all_evidence_ids:
            raise EvidenceRequestError("同一 snapshot 内 Evidence artifact 不能跨 pair 复用")
        if record.source_artifact_ref in all_source_refs:
            raise EvidenceRequestError("同一 snapshot 内 source artifact ref 不能跨 pair 复用")
        if (
            record.artifact_id in all_source_refs
            or record.source_artifact_ref in all_evidence_ids
        ):
            raise EvidenceRequestError(
                "同一 snapshot 内 source ref 与 Evidence artifact ID 不能跨命名空间复用"
            )
        if record.artifact_id == record.source_artifact_ref:
            raise EvidenceRegistryCorruption(
                "MiniChem source ref 与 Evidence artifact ID 不能相同"
            )

    def _snapshot_from_registry(
        self,
        raw: Mapping[str, Any],
        registry: Mapping[str, Any],
    ) -> BatchEvidenceSnapshot:
        snapshot_id, task_id, batch_id, raw_pairs = self._validate_snapshot_record(raw)
        pairs: list[ResultEvidencePair] = []
        seen_combinations: set[str] = set()
        seen_evidence_ids: set[str] = set()
        seen_source_refs: set[str] = set()
        for raw_pair in raw_pairs:
            pairs.append(
                self._parse_snapshot_pair(
                    raw_pair,
                    registry,
                    seen_combinations,
                    seen_evidence_ids,
                    seen_source_refs,
                )
            )
        pairs.sort(key=lambda item: (item.combination_id, item.result_version))
        snapshot = BatchEvidenceSnapshot(snapshot_id, task_id, batch_id, tuple(pairs))
        if snapshot_id != self._derive_snapshot_id(task_id, batch_id, snapshot.result_pairs):
            raise EvidenceRegistryCorruption("快照内容与 snapshot_id 不一致")
        return snapshot

    def _validate_snapshot_record(
        self,
        raw: Mapping[str, Any],
    ) -> tuple[str, str, str, list[Any]]:
        """核对 snapshot 顶层字段并返回待解析的当前批次 pair。"""

        if not isinstance(raw, Mapping) or raw.get("schema_version") != _SCHEMA_VERSION:
            raise EvidenceRegistryCorruption("快照记录格式或版本非法")
        snapshot_id = raw.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not _OPAQUE_ID_RE.fullmatch(snapshot_id):
            raise EvidenceRegistryCorruption("快照标识非法")
        task_id = _read_snapshot_registry_text(self, raw.get("task_id"), "task_id")
        batch_id = _read_snapshot_registry_text(self, raw.get("batch_id"), "batch_id")
        raw_pairs = raw.get("result_pairs")
        if not isinstance(raw_pairs, list) or not 1 <= len(raw_pairs) <= 5:
            raise EvidenceRegistryCorruption("快照必须保存一至五项 result pair")
        return snapshot_id, task_id, batch_id, raw_pairs

    def _parse_snapshot_pair(
        self,
        raw_pair: Any,
        registry: Mapping[str, Any],
        seen_combinations: set[str],
        seen_evidence_ids: set[str],
        seen_source_refs: set[str],
    ) -> ResultEvidencePair:
        """解析一个 pair 并在共享集合中核对 artifact 不重复。"""

        if not isinstance(raw_pair, Mapping):
            raise EvidenceRegistryCorruption("快照 pair 格式非法")
        combination_id = _read_snapshot_registry_text(
            self, raw_pair.get("combination_id"), "combination_id"
        )
        result_version = _read_snapshot_registry_text(
            self, raw_pair.get("result_version"), "result_version"
        )
        if combination_id in seen_combinations:
            raise EvidenceRegistryCorruption("快照 combination_id 重复")
        seen_combinations.add(combination_id)
        raw_artifacts = raw_pair.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise EvidenceRegistryCorruption("快照 pair artifacts 字段非法")
        artifacts = [
            self._parse_snapshot_artifact(
                raw_artifact,
                registry,
                seen_evidence_ids,
                seen_source_refs,
            )
            for raw_artifact in raw_artifacts
        ]
        return ResultEvidencePair(
            combination_id,
            result_version,
            tuple(sorted(artifacts, key=lambda item: item.artifact_id)),
        )

    def _parse_snapshot_artifact(
        self,
        raw_artifact: Any,
        registry: Mapping[str, Any],
        seen_evidence_ids: set[str],
        seen_source_refs: set[str],
    ) -> EvidenceArtifactManifest:
        """解析并重新核对一个 snapshot artifact 与 registry 的一致性。"""

        manifest = _build_snapshot_artifact_manifest(self, raw_artifact)
        artifact_id = manifest.artifact_id
        source_artifact_ref = manifest.source_artifact_ref
        if artifact_id in seen_evidence_ids or source_artifact_ref in seen_source_refs:
            raise EvidenceRegistryCorruption("快照 artifact 不能跨 pair 复用")
        if artifact_id in seen_source_refs or source_artifact_ref in seen_evidence_ids:
            raise EvidenceRegistryCorruption(
                "快照 source ref 与 Evidence artifact ID 不能跨命名空间复用"
            )
        if artifact_id == source_artifact_ref:
            raise EvidenceRegistryCorruption("MiniChem source ref 与 Evidence artifact ID 不能相同")
        try:
            registered = self._load_registered_artifact(registry, artifact_id)
        except (EvidenceArtifactNotFound, EvidenceRegistryCorruption):
            raise EvidenceRegistryCorruption("快照 artifact 未在 registry 登记") from None
        if registered.to_manifest() != manifest:
            raise EvidenceRegistryCorruption("快照 artifact 摘要与 registry 不一致")
        seen_evidence_ids.add(artifact_id)
        seen_source_refs.add(source_artifact_ref)
        return manifest

    def create_batch_evidence_snapshot(
        self,
        task_id: str,
        batch_id: str,
        result_vector: Any,
        artifact_groups: Mapping[
            tuple[str, str], Sequence[EvidenceArtifactManifest | str]
        ],
    ) -> BatchEvidenceSnapshot:
        """创建或幂等返回一至五项 pair 级 Review 快照。

        输入是任务/批次标识、一至五项 ``(combination_id, result_version)`` 序列，以及
        唯一的 ``{(combination_id, result_version): [已登记 manifest 或 opaque ID, ...]}``
        分组。分组必须精确覆盖全部当前 pair；路径、mapping artifact、未知 ID、
        多余组合、版本漂移、重复 artifact 和跨 pair 复用都会失败。空分组合法，
        整个快照也可没有 artifact。动作先重新核对每个已登记来源的大小、SHA-256
        和 UTF-8 内容，再追加 0600 registry 的不可变 snapshot 记录；不隐式注册路径。
        相同规范化输入返回同一 snapshot_id。
        """

        with self._lock, self._registry_file_lock():
            normalized_task_id = self._validate_safe_text(task_id, "task_id")
            normalized_batch_id = self._validate_safe_text(batch_id, "batch_id")
            vector = self._normalize_result_vector(result_vector)
            result_pairs = set(vector)
            registry = self._load_registry()
            groups = self._resolve_artifact_groups(result_pairs, artifact_groups, registry)
            pairs = tuple(
                ResultEvidencePair(
                    combination_id,
                    result_version,
                    tuple(record.to_manifest() for record in groups[(combination_id, result_version)]),
                )
                for combination_id, result_version in sorted(result_pairs)
            )
            snapshot_id = self._derive_snapshot_id(
                normalized_task_id,
                normalized_batch_id,
                pairs,
            )
            snapshot = BatchEvidenceSnapshot(
                snapshot_id,
                normalized_task_id,
                normalized_batch_id,
                pairs,
            )
            payload = {"schema_version": _SCHEMA_VERSION, **snapshot.to_public_manifest()}
            existing = registry["snapshots"].get(snapshot_id)
            if existing is not None:
                if existing != payload:
                    raise EvidenceRegistryCorruption("不可变快照内容与标识不一致")
                return self._snapshot_from_registry(existing, registry)
            registry["snapshots"][snapshot_id] = payload
            self._write_registry(registry)
            return snapshot

    def read_public_snapshot_manifest(self, snapshot_id: str) -> dict[str, Any]:
        """读取精简 path-free manifest 的独立副本。

        输入只能是 opaque snapshot_id；动作不访问来源文件、不写 registry，也不
        返回内部路径。未知 ID、路径形式 ID 或注册表损坏时失败。
        """

        with self._lock, self._registry_file_lock():
            normalized_id = self._validate_opaque_id(snapshot_id, "snapshot_id")
            registry = self._load_registry()
            raw = registry["snapshots"].get(normalized_id)
            if not isinstance(raw, Mapping):
                raise EvidenceSnapshotNotFound("找不到请求的 snapshot")
            snapshot = self._snapshot_from_registry(raw, registry)
            if snapshot.snapshot_id != normalized_id:
                raise EvidenceRegistryCorruption("快照索引标识不一致")
            return snapshot.to_public_manifest()

    @staticmethod
    def _validate_json_pointer(json_pointer: Any) -> str | None:
        if json_pointer is None:
            return None
        ReviewEvidenceStore._validate_json_pointer_text(json_pointer)
        if json_pointer == "":
            return json_pointer
        ReviewEvidenceStore._validate_json_pointer_tokens(json_pointer)
        return json_pointer

    @staticmethod
    def _validate_json_pointer_text(json_pointer: Any) -> None:
        """校验 JSON pointer 的文本长度、控制字符和首字符。"""

        if not isinstance(json_pointer, str) or len(json_pointer) > 512:
            raise EvidenceRequestError("json_pointer 必须是短 JSON pointer")
        if _CONTROL_RE.search(json_pointer):
            raise EvidenceRequestError("json_pointer 不能包含控制字符")
        if not json_pointer.startswith("/") and json_pointer != "":
            raise EvidenceRequestError("json_pointer 格式非法")

    @staticmethod
    def _validate_json_pointer_tokens(json_pointer: str) -> None:
        """校验 JSON pointer token 的 RFC 转义和路径片段。"""

        for token in json_pointer.split("/")[1:]:
            index = 0
            while index < len(token):
                if token[index] != "~":
                    index += 1
                    continue
                if index + 1 >= len(token) or token[index + 1] not in "01":
                    raise EvidenceRequestError("json_pointer 转义非法")
                index += 2
            if token in {".", ".."}:
                raise EvidenceRequestError("json_pointer 不能包含路径片段")

    @staticmethod
    def _select_json_pointer_text(
        source_text: str,
        media_type: str,
        json_pointer: str | None,
    ) -> str:
        if json_pointer is None:
            return source_text
        if media_type != "application/json" and not media_type.endswith("+json"):
            raise EvidenceRequestError("json_pointer 只适用于 JSON artifact")
        value = ReviewEvidenceStore._load_json_pointer_document(source_text)
        value = ReviewEvidenceStore._navigate_json_pointer_value(value, json_pointer)
        return ReviewEvidenceStore._serialize_json_pointer_value(value)

    @staticmethod
    def _load_json_pointer_document(source_text: str) -> Any:
        """严格解析 JSON artifact 文本并转换解析失败。"""

        try:
            return json.loads(source_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvidenceRequestError("JSON artifact 内容无法解析") from exc

    @staticmethod
    def _navigate_json_pointer_value(value: Any, json_pointer: str) -> Any:
        """沿 JSON pointer 逐 token 导航并拒绝越界或标量继续导航。"""

        if not json_pointer:
            return value
        for raw_token in json_pointer.split("/")[1:]:
            token = raw_token.replace("~1", "/").replace("~0", "~")
            value = ReviewEvidenceStore._select_json_pointer_child(value, token)
        return value

    @staticmethod
    def _select_json_pointer_child(value: Any, token: str) -> Any:
        """选择 JSON 数组索引或 object key 的下一层值。"""

        if isinstance(value, list):
            if not token.isdigit() or (token != "0" and token.startswith("0")):
                raise EvidenceRequestError("json_pointer 数组索引非法")
            index = int(token)
            if index >= len(value):
                raise EvidenceRequestError("json_pointer 超出 JSON 内容范围")
            return value[index]
        if isinstance(value, dict):
            if token not in value:
                raise EvidenceRequestError("json_pointer 键不存在")
            return value[token]
        raise EvidenceRequestError("json_pointer 不能继续导航标量")

    @staticmethod
    def _serialize_json_pointer_value(value: Any) -> str:
        """以稳定 JSON 文本返回 pointer 选择的值。"""

        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise EvidenceRequestError("JSON pointer 值无法编码") from exc

    def _find_snapshot_artifact(
        self,
        snapshot: BatchEvidenceSnapshot,
        artifact_id: str,
    ) -> EvidenceArtifactManifest:
        for pair in snapshot.result_pairs:
            for artifact in pair.artifacts:
                if artifact.artifact_id == artifact_id:
                    return artifact
        raise EvidenceArtifactNotFound("artifact 不属于请求的 snapshot")

    def _read_and_verify_registered_artifact(self, record: _RegisteredArtifact) -> str:
        source_path = Path(record.source_path)
        try:
            before = source_path.stat()
            if not source_path.is_file() or before.st_size != record.size:
                raise EvidenceArtifactIntegrityError("来源文件大小或类型已变化")
            with source_path.open("rb") as handle:
                content = handle.read(self.max_artifact_bytes + 1)
            after = source_path.stat()
        except EvidenceArtifactIntegrityError:
            raise
        except OSError as exc:
            raise EvidenceArtifactIntegrityError("来源文件无法重新读取") from exc
        if (
            len(content) > self.max_artifact_bytes
            or len(content) != record.size
            or after.st_size != record.size
            or self._sha256_bytes(content) != record.sha256
        ):
            raise EvidenceArtifactIntegrityError("来源文件大小或摘要已变化")
        return self._validate_text_content(content)

    def _take_process_budget(
        self,
        snapshot_id: str,
        selected_text: str,
        offset: int,
        max_chars: int,
    ) -> tuple[str, int, int | None]:
        budget_key = (str(self.root_dir), snapshot_id)
        with _PROCESS_BUDGET_LOCK:
            consumed = _PROCESS_BUDGETS.get(budget_key, 0)
            remaining = self.max_total_read_chars - consumed
            if remaining <= 0:
                raise EvidenceBudgetExceeded("snapshot 累计读取预算已用尽")
            page_chars = min(max_chars, remaining)
            content = selected_text[offset : offset + page_chars]
            _PROCESS_BUDGETS[budget_key] = consumed + len(content)
            next_offset = offset + len(content) if offset + len(content) < len(selected_text) else None
            return content, self.max_total_read_chars - _PROCESS_BUDGETS[budget_key], next_offset

    def read_snapshot_artifact(
        self,
        snapshot_id: str,
        artifact_id: str,
        offset: int = 0,
        max_chars: int = _DEFAULT_MAX_SINGLE_READ_CHARS,
        json_pointer: str | None = None,
    ) -> EvidenceReadResult:
        """按 snapshot/artifact opaque ID 有界读取文本。

        输入只能是 snapshot_id、artifact_id、非负字符 offset、单次 max_chars 和
        可选安全 JSON pointer，不能传路径。动作先核对 pair 归属，再每次重新验证
        来源 size/SHA-256 和 UTF-8，随后在当前 reader 进程累计预算内返回内容与游标；
        不执行、不写入来源。越界、跨快照、篡改、二进制、非法 pointer 或预算耗尽
        时失败。
        """

        with self._lock, self._registry_file_lock():
            normalized_snapshot_id = self._validate_opaque_id(snapshot_id, "snapshot_id")
            normalized_artifact_id = self._validate_opaque_id(artifact_id, "artifact_id")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise EvidenceRequestError("offset 必须是非负整数")
            if (
                isinstance(max_chars, bool)
                or not isinstance(max_chars, int)
                or max_chars <= 0
                or max_chars > self.max_single_read_chars
            ):
                raise EvidenceRequestError("max_chars 超出单次读取上限")
            normalized_pointer = self._validate_json_pointer(json_pointer)
            registry = self._load_registry()
            raw_snapshot = registry["snapshots"].get(normalized_snapshot_id)
            if not isinstance(raw_snapshot, Mapping):
                raise EvidenceSnapshotNotFound("找不到请求的 snapshot")
            snapshot = self._snapshot_from_registry(raw_snapshot, registry)
            if snapshot.snapshot_id != normalized_snapshot_id:
                raise EvidenceRegistryCorruption("快照索引标识不一致")
            snapshot_artifact = self._find_snapshot_artifact(snapshot, normalized_artifact_id)
            record = self._load_registered_artifact(registry, normalized_artifact_id)
            if record.to_manifest() != snapshot_artifact:
                raise EvidenceArtifactIntegrityError("快照 artifact 摘要与注册表不一致")
            source_text = self._read_and_verify_registered_artifact(record)
            selected_text = self._select_json_pointer_text(
                source_text,
                record.media_type,
                normalized_pointer,
            )
            total_chars = len(selected_text)
            if offset > total_chars:
                raise EvidenceRequestError("offset 超出可读取内容范围")
            content, budget_remaining, next_offset = self._take_process_budget(
                normalized_snapshot_id,
                selected_text,
                offset,
                max_chars,
            )
            return EvidenceReadResult(
                normalized_snapshot_id,
                normalized_artifact_id,
                offset,
                max_chars,
                content,
                total_chars,
                next_offset is not None,
                next_offset,
                budget_remaining,
                normalized_pointer,
            )


def _read_snapshot_registry_text(
    store: ReviewEvidenceStore,
    value: Any,
    label: str,
) -> str:
    """把 registry 文本字段转换为 corruption 错误，不泄露原值。"""

    try:
        return store._validate_safe_text(value, label)
    except EvidenceRequestError:
        raise EvidenceRegistryCorruption(f"快照 {label} 非法") from None


def _build_snapshot_artifact_manifest(
    store: ReviewEvidenceStore,
    raw_artifact: Any,
) -> EvidenceArtifactManifest:
    """校验 registry 中保存的 artifact 摘要字段。"""

    if not isinstance(raw_artifact, Mapping):
        raise EvidenceRegistryCorruption("快照 artifact 摘要非法")
    artifact_id, source_artifact_ref, size, sha256, media_type = (
        raw_artifact.get("artifact_id"),
        raw_artifact.get("source_artifact_ref"),
        raw_artifact.get("size"),
        raw_artifact.get("sha256"),
        raw_artifact.get("media_type"),
    )
    _validate_snapshot_artifact_identity(artifact_id, source_artifact_ref)
    _validate_snapshot_artifact_measurements(store, size, sha256, media_type)
    return EvidenceArtifactManifest(
        artifact_id,
        source_artifact_ref,
        size,
        sha256,
        media_type,
    )


def _validate_snapshot_artifact_identity(
    artifact_id: Any,
    source_artifact_ref: Any,
) -> None:
    """校验 snapshot artifact 与 MiniChem source ref 的 ID 形状。"""

    if not isinstance(artifact_id, str) or not _OPAQUE_ID_RE.fullmatch(artifact_id):
        raise EvidenceRegistryCorruption("快照 artifact 标识非法")
    if not isinstance(source_artifact_ref, str) or _SOURCE_ARTIFACT_REF_RE.fullmatch(
        source_artifact_ref
    ) is None:
        raise EvidenceRegistryCorruption("快照 source_artifact_ref 非法")


def _validate_snapshot_artifact_measurements(
    store: ReviewEvidenceStore,
    size: Any,
    sha256: Any,
    media_type: Any,
) -> None:
    """校验 snapshot artifact 的大小、SHA-256 和文本媒体类型。"""

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise EvidenceRegistryCorruption("快照 artifact 大小非法")
    if not isinstance(sha256, str) or not _HEX_RE.fullmatch(sha256):
        raise EvidenceRegistryCorruption("快照 artifact 摘要非法")
    if not isinstance(media_type, str) or not store._is_supported_text_media_type(media_type):
        raise EvidenceRegistryCorruption("快照 artifact 媒体类型非法")


def read_snapshot_artifact(
    store: ReviewEvidenceStore,
    snapshot_id: str,
    artifact_id: str,
    offset: int = 0,
    max_chars: int = _DEFAULT_MAX_SINGLE_READ_CHARS,
    json_pointer: str | None = None,
) -> dict[str, Any]:
    """唯一只读 Review MCP adapter；只接受 opaque IDs，不接受路径或注册参数。"""

    return store.read_snapshot_artifact(
        snapshot_id,
        artifact_id,
        offset,
        max_chars,
        json_pointer,
    ).to_public_dict()
