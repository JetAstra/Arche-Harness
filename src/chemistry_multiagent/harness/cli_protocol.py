"""Shared subprocess and output parsing for official CLI harnesses."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import HarnessError, HarnessRequest, HarnessResponse


def _json_line_objects(text: str) -> list[Mapping[str, Any]]:
    objects: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            objects.append(value)
    return objects


def _session_id_from_object(value: Mapping[str, Any], *, depth: int = 0) -> str | None:
    """Find a Codex session/thread identifier in one JSONL event.

    Codex CLI versions have used both ``session_id`` and ``thread_id`` and
    have emitted the identifier either on the event or under a ``thread`` /
    ``session`` object.  Only explicit session/thread keys are considered;
    response IDs and arbitrary prompt fields must never be mistaken for a
    resumable session.
    """

    if depth > 4:
        return None
    for key in ("session_id", "sessionId", "thread_id", "threadId"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for key in ("thread", "session", "data", "result"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            if key in {"thread", "session"}:
                nested_id = nested.get("id")
                if isinstance(nested_id, str) and nested_id.strip():
                    return nested_id.strip()
            found = _session_id_from_object(nested, depth=depth + 1)
            if found:
                return found
    return None


def extract_cli_session_id(text: str) -> str | None:
    """Extract a resumable Codex session ID from JSONL output, if present."""

    for obj in _json_line_objects(text):
        found = _session_id_from_object(obj)
        if found:
            return found
    return None


def extract_cli_content(text: str, *, final_message: str | None = None) -> tuple[str, str | None, dict[str, int]]:
    """Extract only the final assistant text; preserve it for strict validation."""

    objects = _json_line_objects(text)
    content = (final_message or "").strip("\n")
    finish_reason: str | None = None
    usage: dict[str, int] = {}
    for obj in objects:
        kind = str(obj.get("type") or obj.get("event") or "")
        if kind == "result" and isinstance(obj.get("result"), str):
            content = obj["result"]
            finish_reason = obj.get("stop_reason") or obj.get("finish_reason") or "stop"
        elif kind in {"agent_message", "message", "final_message"}:
            value = obj.get("text", obj.get("content", obj.get("message")))
            if isinstance(value, str) and value.strip():
                content = value
        usage_obj = obj.get("usage")
        if isinstance(usage_obj, Mapping):
            for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens"):
                value = usage_obj.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[key] = value
    if not content:
        raise HarnessError("CLI returned no final assistant content", kind="empty_content")
    return content, finish_reason or "stop", usage


def run_cli(
    *,
    command: Sequence[str],
    request: HarnessRequest,
    trace_path: Path,
    env: Mapping[str, str] | None = None,
    final_message_path: Path | None = None,
    stdin_text: str | None = None,
) -> HarnessResponse:
    """Run an official CLI in an isolated role workspace."""

    request.workspace.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = trace_path.with_suffix(".prompt.json")
    prompt_path.write_text(
        json.dumps(
            {
                "role": request.role,
                "stage": request.stage,
                "messages": list(request.messages),
                "response_schema": request.response_schema,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        process_kwargs: dict[str, Any] = {
            "cwd": request.workspace,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "env": merged_env,
            "timeout": request.timeout_seconds,
            "check": False,
        }
        if stdin_text is None:
            process_kwargs["stdin"] = subprocess.DEVNULL
        else:
            process_kwargs["input"] = stdin_text
        completed = subprocess.run(list(command), **process_kwargs)
    except FileNotFoundError as exc:
        raise HarnessError(f"harness executable not found: {command[0]}", kind="transport") from exc
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        trace_path.write_text(output, encoding="utf-8")
        raise HarnessError("harness process timed out", kind="transport") from exc
    trace_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise HarnessError(
            f"harness exited with code {completed.returncode}", kind="transport"
        )
    final_text = None
    if final_message_path is not None and final_message_path.is_file():
        final_text = final_message_path.read_text(encoding="utf-8")
    content, finish_reason, usage = extract_cli_content(completed.stdout or "", final_message=final_text)
    return HarnessResponse(
        content=content,
        finish_reason=finish_reason,
        model=request.model,
        usage=usage,
        raw_trace_path=str(trace_path),
        metadata={"harness": request.metadata.get("harness"), "role": request.role, "stage": request.stage},
    )
