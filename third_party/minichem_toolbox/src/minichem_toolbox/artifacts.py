"""Semantic Artifact store used to connect independently callable actions."""

from __future__ import annotations

import json
import os
import re
import stat
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import ArtifactRef
from .scope import current_scope
from ._secure_storage import (
    _open_directory_chain,
    atomic_write_bytes,
    cleanup_temporary_files,
    directory_lock,
    ensure_directory,
    lexical_absolute,
    read_file_in_directory,
    read_regular_file_snapshot,
    ensure_private_file,
    validate_directory,
)


_LOCK = threading.Lock()
_ARTIFACT_ID_PATTERN = re.compile(r"^art_[0-9a-f]{32}$")
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _open_workspace_parent(
    root: Path,
    candidate: Path,
    *,
    allow_missing: bool,
) -> tuple[int, str | None] | None:
    """在固定 workspace fd 链上打开目标父目录，区分缺失与非法组件。"""

    relative = candidate.relative_to(root)
    components = relative.parts
    descriptor = _open_directory_chain(root)
    if not components:
        return descriptor, None
    for component in components[:-1]:
        try:
            child = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=descriptor)
        except FileNotFoundError:
            if allow_missing:
                os.close(descriptor)
                return None
            os.close(descriptor)
            raise FileNotFoundError("Workspace path does not exist") from None
        except OSError:
            os.close(descriptor)
            raise ValueError("Workspace path is invalid") from None
        try:
            metadata = os.fstat(child)
        except OSError:
            os.close(child)
            os.close(descriptor)
            raise ValueError("Workspace path is invalid") from None
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(child)
            os.close(descriptor)
            raise ValueError("Workspace path is invalid")
        os.close(descriptor)
        descriptor = child
    return descriptor, components[-1]


def _validate_workspace_entry(
    descriptor: int,
    name: str | None,
    *,
    allow_missing: bool,
    require_regular: bool,
) -> None:
    """验证已固定父目录中的目标类型，并保留缺失异常语义。"""

    if name is None:
        metadata = os.fstat(descriptor)
    else:
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if allow_missing:
                return
            raise FileNotFoundError("Workspace path does not exist") from None
        except OSError:
            raise ValueError("Workspace path is invalid") from None
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("Workspace path is invalid")
    if require_regular and not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Workspace path is invalid")
    if not require_regular and not stat.S_ISREG(metadata.st_mode) and not stat.S_ISDIR(
        metadata.st_mode
    ):
        raise ValueError("Workspace path is invalid")


def _validate_workspace_target(
    root: Path,
    candidate: Path,
    *,
    allow_missing: bool,
    require_regular: bool,
) -> None:
    """在固定 workspace fd 链上验证目标，区分缺失与非法类型。"""

    opened = _open_workspace_parent(root, candidate, allow_missing=allow_missing)
    if opened is None:
        return
    descriptor, name = opened
    try:
        _validate_workspace_entry(
            descriptor,
            name,
            allow_missing=allow_missing,
            require_regular=require_regular,
        )
    finally:
        os.close(descriptor)


def validate_existing_file(path: str | Path, *, allow_missing: bool = True) -> None:
    """验证绝对文件路径，保留缺失目标的 FileNotFoundError 语义。"""

    candidate = lexical_absolute(path)
    _validate_workspace_target(
        Path(os.sep),
        candidate,
        allow_missing=allow_missing,
        require_regular=True,
    )


def _canonicalize_complete_reference(item: ArtifactRef, store: "ArtifactStore") -> dict[str, Any]:
    """核对完整引用的 workspace 身份，保留 semantic_type 供输入合同报错。"""

    canonical = store.find(item.artifact_id)
    supplied = item.model_dump(mode="json")
    expected = canonical.model_dump(mode="json")
    for key, value in supplied.items():
        if key != "semantic_type" and value != expected.get(key):
            raise KeyError(f"Unknown ArtifactRef: {item.artifact_id}")
    return supplied


def workspace_root() -> Path:
    """返回当前工作区，并拒绝工作区路径中的符号链接。"""

    scoped = current_scope()
    if scoped is not None:
        root = lexical_absolute(scoped.workspace)
        try:
            validate_directory(root)
        except (OSError, ValueError):
            raise RuntimeError("Authenticated MiniChem scope workspace is invalid") from None
        return root
    configured = (
        os.environ.get("MINICHEM_MCP_WORKSPACE", "").strip()
        or os.environ.get("RESEARCHCHEMBENCH_WORKSPACE", "").strip()
    )
    root = lexical_absolute(configured or Path.cwd())
    try:
        validate_directory(root)
    except (OSError, ValueError):
        raise RuntimeError("Chemistry workspace is invalid") from None
    return root


def resolve_workspace_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """解析工作区内路径，逐组件拒绝符号链接且不回显内部路径。"""

    root = workspace_root()
    try:
        raw_value = Path(value).expanduser()
        candidate = Path(
            os.path.normpath(
                os.fspath(raw_value if raw_value.is_absolute() else root / raw_value)
            )
        )
        candidate = lexical_absolute(candidate)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ValueError("Workspace path is invalid") from None
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("Path escapes chemistry workspace") from None
    try:
        _validate_workspace_target(
            root,
            candidate,
            allow_missing=not must_exist,
            require_regular=must_exist,
        )
    except FileNotFoundError:
        raise
    except (OSError, ValueError):
        raise ValueError("Workspace path is invalid") from None
    return candidate


def relative_workspace_path(value: str | Path) -> str:
    """返回工作区内路径的词法相对表示，不解析符号链接。"""

    root = workspace_root()
    candidate = lexical_absolute(value)
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        raise ValueError("Path escapes chemistry workspace") from None


def _sha256(path: Path) -> str:
    """从同一 fd 快照读取文件并返回对应 SHA-256。"""

    return read_file_snapshot(path)[1]


def read_file_snapshot(path: Path) -> tuple[bytes, str]:
    """从同一个只读文件描述符读取字节并计算对应 SHA-256。"""

    try:
        return read_regular_file_snapshot(path)
    except (OSError, ValueError):
        raise ValueError("Artifact path is invalid") from None


class ArtifactStore:
    def __init__(self, root: Path | None = None, *, create: bool = True) -> None:
        self.root = lexical_absolute(root or workspace_root())
        try:
            validate_directory(self.root)
        except (OSError, ValueError):
            raise ValueError("Artifact storage root is invalid") from None
        self.directory = self.root / "_tool_artifacts" / "objects"
        self.index_path = self.root / "_tool_artifacts" / "index.jsonl"
        if create:
            try:
                ensure_directory(self.root / "_tool_artifacts")
                ensure_directory(self.directory)
                with _LOCK:
                    with directory_lock(self.index_path.parent, ".index.lock"):
                        cleanup_temporary_files(self.root / "_tool_artifacts")
                        cleanup_temporary_files(self.directory)
            except (OSError, ValueError):
                raise ValueError("Artifact storage directory is invalid") from None
        else:
            try:
                validate_directory(self.directory)
            except (OSError, ValueError):
                raise FileNotFoundError("Artifact storage directory is unavailable") from None
        try:
            ensure_private_file(self.index_path, allow_missing=True)
        except (OSError, ValueError):
            raise ValueError("Artifact index is invalid") from None

    def _append_index(self, reference: ArtifactRef) -> None:
        record = {
            **reference.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with _LOCK:
            try:
                with directory_lock(self.index_path.parent, ".index.lock"):
                    self._append_index_locked(payload)
            except (OSError, ValueError):
                raise ValueError("Artifact index could not be persisted") from None

    def _append_index_locked(self, payload: bytes) -> None:
        """在已持有跨进程索引锁时原子追加一条 JSONL 记录。"""

        current = read_file_in_directory(
            self.index_path.parent,
            self.index_path.name,
            allow_missing=True,
        )
        atomic_write_bytes(
            self.index_path.parent,
            self.index_path.name,
            current + payload,
        )

    def put_json(
        self,
        value: Any,
        *,
        semantic_type: str,
        producer_action: str,
        producer_backend: str | None,
        parent_artifact_ids: Iterable[str] = (),
    ) -> ArtifactRef:
        artifact_id = f"art_{uuid.uuid4().hex}"
        path = self.directory / f"{artifact_id}.json"
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
        ).encode("utf-8")
        try:
            record = {
                "artifact_id": artifact_id,
                "semantic_type": semantic_type,
                "media_type": "application/json",
                "sha256": "",
                "path": self._relative_artifact_path(path),
                "producer_action": producer_action,
                "producer_backend": producer_backend,
                "parent_artifact_ids": list(parent_artifact_ids),
            }
            with _LOCK:
                with directory_lock(self.index_path.parent, ".index.lock"):
                    atomic_write_bytes(self.directory, path.name, payload)
                    digest = read_regular_file_snapshot(path)[1]
                    reference = ArtifactRef(
                        artifact_id=artifact_id,
                        semantic_type=semantic_type,
                        media_type="application/json",
                        sha256=digest,
                        path=record["path"],
                        producer_action=producer_action,
                        producer_backend=producer_backend,
                        parent_artifact_ids=record["parent_artifact_ids"],
                    )
                    index_record = {
                        **reference.model_dump(mode="json"),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    self._append_index_locked(
                        (json.dumps(index_record, ensure_ascii=False) + "\n").encode(
                            "utf-8"
                        )
                    )
        except (OSError, ValueError):
            raise ValueError("Artifact object could not be persisted") from None
        return reference

    def register_file(
        self,
        path_value: str | Path,
        *,
        semantic_type: str,
        producer_action: str,
        producer_backend: str | None,
        media_type: str = "application/octet-stream",
        parent_artifact_ids: Iterable[str] = (),
    ) -> ArtifactRef:
        path = resolve_workspace_path(path_value, must_exist=True)
        try:
            _, digest = read_regular_file_snapshot(path)
        except (OSError, ValueError):
            raise ValueError("Artifact path is not a regular file") from None
        reference = ArtifactRef(
            artifact_id=f"art_{uuid.uuid4().hex}",
            semantic_type=semantic_type,
            media_type=media_type,
            sha256=digest,
            path=relative_workspace_path(path),
            producer_action=producer_action,
            producer_backend=producer_backend,
            parent_artifact_ids=list(parent_artifact_ids),
        )
        self._append_index(reference)
        return reference

    def load(self, reference: str | dict[str, Any] | ArtifactRef) -> Any:
        if isinstance(reference, ArtifactRef):
            item = reference
        elif isinstance(reference, dict):
            if set(reference) == {"artifact_id"}:
                item = self.find(str(reference["artifact_id"]))
            else:
                item = ArtifactRef.model_validate(reference)
        else:
            item = self.find(reference)
        # 先核对当前 workspace 的索引，再从规范路径安全读取；不能把另一个 scope
        # 的完整引用重新绑定，也不能让调用方路径触发符号链接解析。
        # 先验证调用方给出的词法路径，以保留越界错误的公开语义；真正读取只使用
        # 索引中的规范路径，避免篡改的完整引用改变文件来源。
        resolve_workspace_path(item.path, must_exist=False)
        canonical = self.find(item.artifact_id)
        expected = canonical.model_dump(mode="json")
        supplied = item.model_dump(mode="json")
        if expected != supplied:
            supplied_without_hash = dict(supplied)
            expected_without_hash = dict(expected)
            supplied_without_hash.pop("sha256", None)
            expected_without_hash.pop("sha256", None)
            if supplied_without_hash != expected_without_hash:
                raise KeyError(f"Unknown ArtifactRef: {item.artifact_id}")
        try:
            path = lexical_absolute(self.root / canonical.path)
            path.relative_to(self.root)
            validate_existing_file(path, allow_missing=False)
        except FileNotFoundError as exc:
            # 外部 scope 的相对路径通常在当前 workspace 不存在；将其统一视为
            # 未知引用，而当前索引中确有该 artifact 时保留原始文件缺失错误。
            try:
                self.find(item.artifact_id)
            except KeyError as missing:
                raise missing from exc
            raise
        except (OSError, ValueError):
            raise ValueError("Artifact path is invalid") from None
        raw, actual_sha256 = read_file_snapshot(path)
        if actual_sha256 != item.sha256:
            raise ValueError(f"Artifact hash mismatch: {item.artifact_id}")
        if item.media_type == "application/json" or path.suffix.lower() == ".json":
            return json.loads(raw.decode("utf-8"))
        return {"path": relative_workspace_path(path), "artifact": item.model_dump(mode="json")}

    def find(self, artifact_id: str) -> ArtifactRef:
        try:
            with directory_lock(self.index_path.parent, ".index.lock"):
                raw = read_file_in_directory(
                    self.index_path.parent,
                    self.index_path.name,
                    allow_missing=True,
                )
        except (OSError, ValueError):
            raise KeyError(f"Unknown ArtifactRef: {artifact_id}") from None
        for line in reversed(raw.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("artifact_id") == artifact_id:
                value.pop("created_at", None)
                return ArtifactRef.model_validate(value)
        raise KeyError(f"Unknown ArtifactRef: {artifact_id}")

    def _relative_artifact_path(self, path: Path) -> str:
        """返回内部对象的稳定相对路径，不解析目录符号链接。"""

        try:
            return str(path.relative_to(self.root))
        except ValueError:
            raise ValueError("Artifact path is outside storage root") from None


def canonicalize_artifact_refs(
    value: Any,
    *,
    store: ArtifactStore | None = None,
) -> Any:
    """Expand explicit compact ArtifactRefs without loading their payloads.

    Public Action calls commonly pass only ``{"artifact_id": "art_..."}`` or the
    artifact id string copied from a previous result.  Canonicalization verifies
    that exact reference and expands it to the immutable full reference before a
    worker is launched.  It never substitutes another artifact or scientific
    value.
    """

    artifact_store = store or ArtifactStore()
    if isinstance(value, ArtifactRef):
        return _canonicalize_complete_reference(value, artifact_store)
    if isinstance(value, str) and _ARTIFACT_ID_PATTERN.fullmatch(value):
        return artifact_store.find(value).model_dump(mode="json")
    if isinstance(value, dict):
        if "artifact_id" in value:
            if set(value) == {"artifact_id"}:
                return artifact_store.find(str(value["artifact_id"])).model_dump(mode="json")
            required = {"artifact_id", "semantic_type", "media_type", "sha256", "path"}
            if required <= set(value):
                item = ArtifactRef.model_validate(value)
                return _canonicalize_complete_reference(item, artifact_store)
            missing = sorted(required - set(value))
            extra = sorted(set(value) - {"artifact_id"})
            raise ValueError(
                "ArtifactRef dictionaries must be either the artifact id string, "
                "{'artifact_id': 'art_...'}, or a complete immutable ArtifactRef. "
                f"This partial reference has extra fields {extra} and is missing {missing}; "
                "pass only artifact_id to let the toolbox load the canonical reference."
            )
        return {
            key: canonicalize_artifact_refs(item, store=artifact_store)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_artifact_refs(item, store=artifact_store) for item in value]
    if isinstance(value, tuple):
        return tuple(canonicalize_artifact_refs(item, store=artifact_store) for item in value)
    return value


def collect_artifact_refs(value: Any) -> list[ArtifactRef]:
    references: list[ArtifactRef] = []
    if isinstance(value, ArtifactRef):
        references.append(value)
    elif isinstance(value, dict):
        if set(("artifact_id", "semantic_type", "media_type", "sha256", "path")) <= set(value):
            references.append(ArtifactRef.model_validate(value))
        else:
            for item in value.values():
                references.extend(collect_artifact_refs(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            references.extend(collect_artifact_refs(item))
    return references
