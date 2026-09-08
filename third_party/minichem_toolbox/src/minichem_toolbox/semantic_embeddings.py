"""Offline all-MiniLM-L6-v2 embedding support for small local catalogs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from .paths import MODEL_CACHE_ROOT


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_DIRECTORY_ENV = "MINICHEM_MINILM_MODEL_DIR"
EMBEDDING_CACHE_ENV = "MINICHEM_ACTION_EMBEDDING_CACHE"


def _default_model_directory() -> Path:
    return MODEL_CACHE_ROOT / "all-MiniLM-L6-v2"


def _default_embedding_cache() -> Path:
    return MODEL_CACHE_ROOT / "action_embeddings.npz"


def model_directory() -> Path:
    return Path(os.environ.get(MODEL_DIRECTORY_ENV, _default_model_directory())).expanduser()


def embedding_cache_path() -> Path:
    return Path(os.environ.get(EMBEDDING_CACHE_ENV, _default_embedding_cache())).expanduser()


class MiniLMEncoder:
    """Minimal ONNX/tokenizers runner with mean pooling and L2 normalization."""

    def __init__(self, directory: Path):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search requires the optional semantic-search dependencies"
            ) from exc
        model_path = directory / "model.onnx"
        tokenizer_path = directory / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"MiniLM cache is incomplete at {directory}; run cache_minilm_model.py"
            )
        self._np = np
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=256)
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

    def encode(self, texts: Sequence[str]):
        np = self._np
        encodings = self._tokenizer.encode_batch(list(texts))
        maximum = max((len(item.ids) for item in encodings), default=1)
        input_ids = np.zeros((len(encodings), maximum), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        token_type_ids = np.zeros_like(input_ids)
        for row, encoding in enumerate(encodings):
            size = len(encoding.ids)
            input_ids[row, :size] = encoding.ids
            attention_mask[row, :size] = encoding.attention_mask
            token_type_ids[row, :size] = encoding.type_ids
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = token_type_ids
        hidden = self._session.run(None, inputs)[0]
        mask = attention_mask[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.maximum(norms, 1e-12)


def document_digest(documents: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(documents.items())), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_embedding_cache(documents: Mapping[str, str], path: Path | None = None) -> Path:
    import numpy as np

    destination = path or embedding_cache_path()
    directory = model_directory().resolve()
    model_path = directory / "model.onnx"
    tokenizer_path = directory / "tokenizer.json"
    encoder = _resident_encoder(
        str(directory),
        model_path.stat().st_mtime_ns,
        tokenizer_path.stat().st_mtime_ns,
    )
    ids = sorted(documents)
    vectors = encoder.encode([documents[item] for item in ids])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            ids=np.asarray(ids),
            vectors=vectors,
            digest=np.asarray(document_digest(documents)),
            model_id=np.asarray(MODEL_ID),
        )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    _resident_embedding_cache.cache_clear()
    return destination


@lru_cache(maxsize=4)
def _resident_encoder(
    directory: str,
    model_mtime_ns: int,
    tokenizer_mtime_ns: int,
) -> MiniLMEncoder:
    del model_mtime_ns, tokenizer_mtime_ns
    return MiniLMEncoder(Path(directory))


@lru_cache(maxsize=8)
def _resident_embedding_cache(
    path: str,
    mtime_ns: int,
    size_bytes: int,
):
    del mtime_ns, size_bytes
    import numpy as np

    with np.load(path, allow_pickle=False) as cached:
        return (
            str(cached["digest"].item()),
            tuple(str(item) for item in cached["ids"]),
            cached["vectors"].copy(),
        )


def clear_semantic_runtime_cache() -> None:
    """Drop resident model and vector state after an index or model update."""

    _resident_encoder.cache_clear()
    _resident_embedding_cache.cache_clear()


@contextmanager
def _embedding_cache_lock(path: Path):
    """Serialize index refreshes while keeping readers on an atomic cache file."""

    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cache_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    stat = path.stat()
    digest, _ids, _vectors = _resident_embedding_cache(
        str(path.resolve()), stat.st_mtime_ns, stat.st_size
    )
    return digest


def _refresh_embedding_cache(documents: Mapping[str, str], path: Path) -> None:
    expected_digest = document_digest(documents)
    with _embedding_cache_lock(path):
        if _cache_digest(path) == expected_digest:
            return
        write_embedding_cache(documents, path)


def semantic_scores(
    query: str,
    documents: Mapping[str, str],
    *,
    cache_path: Path | None = None,
) -> tuple[dict[str, float], str]:
    """Return cosine scores from an offline cache, or a transparent status string."""

    try:
        import numpy as np
    except ImportError:
        return {}, "unavailable_numpy"
    path = cache_path or embedding_cache_path()
    try:
        expected_digest = document_digest(documents)
        if _cache_digest(path) != expected_digest:
            _refresh_embedding_cache(documents, path)
        stat = path.stat()
        cached_digest, ids, vectors = _resident_embedding_cache(
            str(path.resolve()), stat.st_mtime_ns, stat.st_size
        )
        if cached_digest != expected_digest:
            return {}, "stale_embedding_cache"
        directory = model_directory().resolve()
        model_path = directory / "model.onnx"
        tokenizer_path = directory / "tokenizer.json"
        encoder = _resident_encoder(
            str(directory),
            model_path.stat().st_mtime_ns,
            tokenizer_path.stat().st_mtime_ns,
        )
        query_vector = encoder.encode([query])[0]
        scores = vectors @ query_vector
        return {item: float(score) for item, score in zip(ids, scores)}, "available"
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        return {}, f"unavailable:{type(exc).__name__}"


__all__ = [
    "EMBEDDING_CACHE_ENV",
    "MODEL_DIRECTORY_ENV",
    "MODEL_ID",
    "MiniLMEncoder",
    "clear_semantic_runtime_cache",
    "document_digest",
    "embedding_cache_path",
    "model_directory",
    "semantic_scores",
    "write_embedding_cache",
]
