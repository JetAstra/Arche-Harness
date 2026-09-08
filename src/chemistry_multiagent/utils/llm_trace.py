"""Lightweight JSONL tracing for LLM calls."""

from __future__ import annotations

import inspect
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_LOCK = threading.Lock()
_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "token", "secret", "password", "authorization")


def _trace_enabled() -> bool:
    value = os.environ.get("ARCHE_LLM_TRACE", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def _trace_dir() -> Path:
    bundle_root = os.environ.get("ARCHE_RUN_BUNDLE_ROOT")
    if bundle_root:
        return Path(bundle_root) / "internal" / "audit" / "llm_traces"
    configured = os.environ.get("ARCHE_LLM_TRACE_DIR")
    if configured:
        return Path(configured)
    return Path("local_data/runtime/diagnostics/llm_traces")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in _SENSITIVE_KEY_PARTS):
                cleaned[key_text] = "[REDACTED]"
            else:
                cleaned[key_text] = _jsonable(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _call_site() -> Dict[str, Any]:
    for frame in inspect.stack()[2:10]:
        filename = frame.filename.replace(os.getcwd() + os.sep, "")
        if filename.endswith("llm_trace.py"):
            continue
        return {
            "file": filename,
            "function": frame.function,
            "line": frame.lineno,
        }
    return {}


def trace_llm_call(
    *,
    provider: str,
    model: str,
    messages: Any,
    raw_response: Optional[str] = None,
    error: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    model_path: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Optional[str]:
    """Append one LLM call trace entry and return its trace id."""
    if not _trace_enabled():
        return None

    now = datetime.now(timezone.utc)
    call_id = trace_id or f"llm_{uuid.uuid4().hex}"
    entry = {
        "trace_id": call_id,
        "timestamp": now.isoformat(),
        "provider": provider,
        "backend": backend,
        "model": model,
        "model_path": model_path,
        "parameters": _jsonable(parameters or {}),
        "messages": _jsonable(messages),
        "raw_response": raw_response,
        "error": error,
        "call_site": _call_site(),
    }

    out_dir = _trace_dir()
    out_path = out_dir / f"{now.strftime('%Y%m%d')}.jsonl"
    try:
        with _LOCK:
            out_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return call_id
    return call_id
