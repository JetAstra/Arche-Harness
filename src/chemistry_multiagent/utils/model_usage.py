"""从单个闭环任务的不可变响应产物汇总模型 token 用量。"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


_RESPONSE_FILE_PATTERN = re.compile(r"(?:^|_)response_[0-9a-f]{64}\.json$")
_RETRIEVAL_WRAPPER_KEYS = {"schema_version", "result", "model_calls"}
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


class TaskTokenUsageError(ValueError):
    """任务根目录本身不适合进行计量时抛出的错误。"""


def _safe_response_target(task_root: Path, relative_path: Any) -> Path | None:
    """只接受不能逃逸任务目录的响应引用。"""

    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
    ):
        return None
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        return None
    target = task_root.joinpath(*relative.parts)
    try:
        target.resolve(strict=False).relative_to(task_root)
    except ValueError:
        return None
    return target


def _discover_response_paths(task_root: Path) -> set[str]:
    """发现模型操作 stage 下已经落盘的响应产物。"""

    working_memory = task_root / "working_memory"
    if not working_memory.is_dir():
        return set()
    discovered: set[str] = set()
    for target in working_memory.rglob("*.json"):
        relative = target.relative_to(task_root)
        if "stages" not in relative.parts:
            continue
        if not _RESPONSE_FILE_PATTERN.search(target.name):
            continue
        discovered.add(relative.as_posix())
    return discovered


def _is_retrieval_wrapper(value: Any) -> bool:
    """识别只负责聚合检索结果和内部调用收据的外层文件。"""

    return (
        isinstance(value, Mapping)
        and set(value) == _RETRIEVAL_WRAPPER_KEYS
        and value.get("schema_version") == 1
    )


def _retrieval_response_refs(
    wrapper: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """提取 Retrieval 内部响应，并区分坏收据与无响应传输。"""

    model_calls = wrapper.get("model_calls")
    if not isinstance(model_calls, list):
        return [], ["Retrieval model_calls 不是数组"], []
    response_paths: list[str] = []
    problems: list[str] = []
    transport_failures: list[str] = []
    for index, call in enumerate(model_calls):
        if not isinstance(call, Mapping):
            problems.append(f"Retrieval model_calls[{index}] 不是对象")
            continue
        status = call.get("validation_status")
        response_ref = call.get("response_ref")
        if response_ref is None:
            if status in {"valid", "invalid"}:
                problems.append(
                    f"Retrieval model_calls[{index}] 缺少响应引用"
                )
            elif status == "transport_failed":
                transport_failures.append(f"model_calls[{index}]")
            continue
        if not isinstance(response_ref, Mapping):
            problems.append(
                f"Retrieval model_calls[{index}] 响应引用不是对象"
            )
            continue
        relative_path = response_ref.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            problems.append(
                f"Retrieval model_calls[{index}] 缺少响应相对路径"
            )
            continue
        response_paths.append(relative_path)
    return response_paths, problems, transport_failures


def _discover_transport_failures(task_root: Path) -> set[str]:
    """从模型操作清单中找出没有服务端响应可计量的传输尝试。"""

    working_memory = task_root / "working_memory"
    if not working_memory.is_dir():
        return set()
    failures: set[str] = set()
    for target in working_memory.rglob("manifest*.json"):
        value, read_error = _read_response(target)
        if read_error is not None or not isinstance(value, Mapping):
            continue
        operation_id = value.get("operation_instance_id")
        stages = value.get("stages")
        if not isinstance(operation_id, str) or not isinstance(stages, list):
            continue
        for stage_position, stage in enumerate(stages, start=1):
            if not isinstance(stage, Mapping):
                continue
            stage_index = stage.get("stage_index", stage_position)
            attempts = stage.get("attempts")
            if not isinstance(attempts, list):
                continue
            for attempt_position, attempt in enumerate(attempts):
                if (
                    isinstance(attempt, Mapping)
                    and attempt.get("validation_status") == "transport_failed"
                ):
                    attempt_index = attempt.get("attempt_index", attempt_position)
                    failures.add(
                        f"{operation_id}/stage-{stage_index}/attempt-{attempt_index}"
                    )
    return failures


def _read_response(path: Path) -> tuple[Any | None, str | None]:
    """读取一份响应 JSON，读取失败时保留可呈现的原因。"""

    if not path.is_file():
        return None, "响应文件不存在"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "响应文件不是可读取的 JSON"


def _response_usage(value: Any) -> tuple[dict[str, int] | None, str | None]:
    """读取 OpenAI 兼容响应中的三项 token 计量值。"""

    if not isinstance(value, Mapping):
        return None, "响应不是对象"
    usage = value.get("usage")
    if not isinstance(usage, Mapping):
        return None, "响应缺少完整 usage"
    normalized: dict[str, int] = {}
    for field in _TOKEN_FIELDS:
        token_count = usage.get(field)
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            return None, "响应缺少完整 usage"
        normalized[field] = token_count
    return normalized, None


def summarize_task_token_usage(task_root: str | Path) -> dict[str, Any]:
    """按响应相对路径汇总一次任务的模型 token 用量。

    相同内容写入不同路径代表真实发生了多次调用，因此不会按内容摘要去重。
    Retrieval 的外层文件只是聚合容器；其中每个带响应的内部模型调用会作为
    独立响应计数。只要任一已落盘响应缺少可信 usage，精确总量就返回 ``None``，
    同时通过 ``known_usage`` 给出当前能够确认的下界和对应问题路径。没有响应
    回执的传输失败也会让统计不完整，因为本地产物无法证明服务端完全没有计费。
    """

    root = Path(task_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise TaskTokenUsageError("任务根目录不存在或不是目录")

    pending_paths = _discover_response_paths(root)
    processed_paths: set[str] = set()
    counted_paths: list[str] = []
    incomplete: dict[str, str] = {}
    unmetered_transport_attempts = _discover_transport_failures(root)
    known_usage = {field: 0 for field in _TOKEN_FIELDS}

    while pending_paths:
        relative_path = min(pending_paths)
        pending_paths.remove(relative_path)
        if relative_path in processed_paths:
            continue
        processed_paths.add(relative_path)
        target = _safe_response_target(root, relative_path)
        if target is None:
            incomplete[relative_path] = "响应路径越过任务根目录"
            continue
        value, read_error = _read_response(target)
        if read_error is not None:
            incomplete[relative_path] = read_error
            continue
        if _is_retrieval_wrapper(value):
            (
                response_refs,
                wrapper_problems,
                retrieval_transport_failures,
            ) = _retrieval_response_refs(value)
            if wrapper_problems:
                incomplete[relative_path] = "；".join(wrapper_problems)
            for failure in retrieval_transport_failures:
                unmetered_transport_attempts.add(
                    f"{relative_path}#{failure}"
                )
            for response_path in response_refs:
                if _safe_response_target(root, response_path) is None:
                    incomplete[response_path] = "响应路径越过任务根目录"
                    continue
                if response_path not in processed_paths:
                    pending_paths.add(response_path)
            continue
        usage, usage_error = _response_usage(value)
        if usage_error is not None:
            incomplete[relative_path] = usage_error
            continue
        counted_paths.append(relative_path)
        for field in _TOKEN_FIELDS:
            known_usage[field] += usage[field]

    counted_paths.sort()
    incomplete_responses = [
        {"relative_path": path, "reason": incomplete[path]}
        for path in sorted(incomplete)
    ]
    sorted_transport_failures = sorted(unmetered_transport_attempts)
    complete = not incomplete_responses and not sorted_transport_failures
    exact_usage = dict(known_usage) if complete else None
    return {
        "schema_version": 1,
        "complete": complete,
        "response_artifact_count": len(counted_paths) + len(incomplete_responses),
        "counted_response_count": len(counted_paths),
        "usage": exact_usage,
        "known_usage": dict(known_usage),
        "counted_response_paths": counted_paths,
        "incomplete_responses": incomplete_responses,
        "unmetered_transport_attempts": sorted_transport_failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """提供一个可直接用于基线任务目录的只读命令行入口。"""

    parser = argparse.ArgumentParser(
        description="汇总闭环任务目录中可审计的模型 token 用量。"
    )
    parser.add_argument("task_root", help="闭环任务根目录")
    arguments = parser.parse_args(argv)
    summary = summarize_task_token_usage(arguments.task_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
