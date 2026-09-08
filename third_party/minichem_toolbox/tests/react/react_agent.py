#!/usr/bin/env python3
"""Minimal ReAct agent that calls MiniChem directly through Python functions."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from minichem_mcp_tools.discovery_models import (
    ActionInspectRequest,
    ActionSearchRequest,
    ProgressiveActionRequest,
)
from minichem_mcp_tools.discovery_tools import execute_action, inspect_action, search_actions


DEFAULT_TASK = """Use xTB to calculate the single-point energy of neutral singlet water.
Use GFN2-xTB, one CPU core, at most 2048 MB memory, and this Cartesian geometry in angstrom:
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
Report the energy, unit, execution status, and artifact paths."""

SYSTEM_PROMPT = """You control the predefined Action layer of MiniChem Toolbox.
Use a bounded ReAct loop: select one tool, observe its JSON result, then decide the next step.
Return exactly one JSON object per turn. Do not use Markdown.

Tool call format:
{"reason":"one short sentence","action":"search_actions|inspect_action|execute_action","action_input":{...}}

Final format:
{"reason":"one short sentence","final":"concise answer grounded only in tool observations"}

Available tools:
1. search_actions
   input: {"query": string, "retrieval_mode": "lexical"|"hybrid", "available_only": bool,
           "limit": integer from 1 to 10}
2. inspect_action
   input: {"action_id": string, "backend_id": optional string, "detail_level": "contract"}
3. execute_action
   input: the exact executable request described by inspect_action, including action_id.

Rules:
- Search before selecting an Action and inspect the exact Action/backend before execution.
- Never submit placeholders, guessed field names, backend_id="auto", or unspecified scientific methods.
- For a requested calculation, do not finish before execute_action reaches a terminal result.
- On invalid_request, repair the same request once from the diagnostic instead of changing backend.
- Do not claim facts that are absent from observations.
- Keep reason to one sentence; do not reveal private chain-of-thought.
"""

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


def _search(arguments: dict[str, Any]) -> dict[str, Any]:
    return search_actions(ActionSearchRequest.model_validate(arguments))


def _inspect(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_action(ActionInspectRequest.model_validate(arguments))


def _execute(arguments: dict[str, Any]) -> dict[str, Any]:
    return execute_action(ProgressiveActionRequest.model_validate(arguments))


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "search_actions": _search,
    "inspect_action": _inspect,
    "execute_action": _execute,
}


def create_workspace(path: Path) -> Path:
    root = path.expanduser().resolve()
    for name in (
        "code",
        "data",
        "outputs",
        "report",
        "tool_logs",
        "_tool_results",
        "_tool_artifacts",
        "_sessions",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ["MINICHEM_MCP_WORKSPACE"] = str(root)
    os.environ["RESEARCHCHEMBENCH_WORKSPACE"] = str(root)
    return root


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response does not contain a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response must be one JSON object")
    return value


def call_chat_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.load(response)
            request_id = response.headers.get("x-request-id")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc

    choice = result["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    return parse_json_object(content), {
        "model": result.get("model", model),
        "request_id": request_id,
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage", {}),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def run_react_agent(
    *,
    task: str,
    workspace: Path,
    chat: Callable[[list[dict[str, str]]], tuple[dict[str, Any], dict[str, Any]]],
    tool_handlers: dict[str, ToolHandler] | None = None,
    max_steps: int = 8,
) -> str:
    handlers = tool_handlers or TOOL_HANDLERS
    transcript = workspace / "_sessions" / "react.jsonl"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    append_jsonl(transcript, {"type": "task", "content": task})

    for step in range(1, max_steps + 1):
        decision, metadata = chat(messages)
        append_jsonl(
            transcript,
            {"type": "model", "step": step, "decision": decision, "api": metadata},
        )
        messages.append(
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)}
        )

        if isinstance(decision.get("final"), str):
            final = decision["final"].strip()
            (workspace / "report" / "final.md").write_text(final + "\n", encoding="utf-8")
            return final

        action = decision.get("action")
        arguments = decision.get("action_input")
        if action not in handlers or not isinstance(arguments, dict):
            observation = {
                "status": "invalid_agent_action",
                "error": "Choose one listed action and provide action_input as an object.",
            }
        else:
            try:
                observation = handlers[action](arguments)
            except Exception as exc:  # Keep tool/schema errors inside the ReAct observation loop.
                observation = {
                    "status": "tool_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        append_jsonl(
            transcript,
            {"type": "tool", "step": step, "action": action, "observation": observation},
        )
        messages.append(
            {
                "role": "user",
                "content": "Observation from "
                + str(action)
                + ":\n"
                + json.dumps(observation, ensure_ascii=False),
            }
        )

    raise RuntimeError(f"Agent did not finish within {max_steps} ReAct steps")


def default_workspace() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[1] / "results" / f"react_{stamp}_{os.getpid()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MINICHEM_AGENT_BASE_URL")
        or os.environ.get("OPENCODE_BASE_URL_VALUE")
        or "https://api.deepseek.com/v1",
    )
    parser.add_argument(
        "--model",
        default=(
            os.environ.get("MINICHEM_AGENT_MODEL")
            or os.environ.get("OPENCODE_MODEL_VALUE")
            or "deepseek-v4-flash"
        ).split("/", 1)[-1],
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()

    api_key = os.environ.get("MINICHEM_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("set MINICHEM_AGENT_API_KEY or OPENAI_API_KEY")
    workspace = create_workspace(args.workspace or default_workspace())

    def chat(messages: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
        return call_chat_api(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            messages=messages,
            timeout_seconds=args.timeout_seconds,
        )

    final = run_react_agent(
        task=args.task,
        workspace=workspace,
        chat=chat,
        max_steps=args.max_steps,
    )
    print(final)
    print(f"Workspace: {workspace}")


if __name__ == "__main__":
    main()
