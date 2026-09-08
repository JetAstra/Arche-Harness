"""模型阶段可读输出的独立存储。

这个模块只负责给研究员保存模型调用的展示副本，不参与正式任务恢复，也不复制
Gaussian、MiniChem 或其他计算产物。正式工作流可以把写入失败当作展示降级，继续
使用自己的状态和原始记录。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from chemistry_multiagent.utils.run_bundle import RunBundle


LOGGER = logging.getLogger(__name__)
__all__ = [
    "ModelRunOutputWriter",
    "ModelStageHandle",
    "UnsafeModelOutputPath",
    "validate_model_run_output_resume_files",
]
_SCHEMA_VERSION = 1
_RESEARCH_TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TERMINAL_STAGE_STATUSES = frozenset({"validated", "failed"})
_ALLOWED_STAGE_STATUSES = frozenset({"running", "validated", "failed"})
_ALLOWED_TASK_STATUSES = frozenset(
    {
        "open",
        "running",
        "retrieval_incomplete",
        "waiting_investigation_job",
        "paused",
        "succeeded",
        "failed",
        "stopped_no_supported_hypothesis",
    }
)

# 这些是展示协议允许进入 request/raw_response 的字段。认证信息和运行环境不在
# 协议中，即使调用方误传也不会被复制到展示文件。
_REQUEST_FIELDS = (
    "catalog_key",
    "prompt_catalog_version",
    "projector_version",
    "serializer_version",
    "model_profile_version",
    "profile_id",
    "model",
    "thinking",
    "reasoning_effort",
    "messages",
    "temperature",
    "top_p",
    # 通用 compiled request 用 null 记录“未设置输出上限”；展示审计保留该事实，
    # 实际 OpenAI-compatible 请求是否携带字段由 llm_api 决定。
    "max_tokens",
    "stop",
    "stream",
    "provider_output_mode",
    # provider_output_mode 是本地校验模式；该字段记录实际发送的远端 JSON mode，
    # DeepSeek v4 Pro 明确保存 null，避免审计把本地 json_object 误当成请求参数。
    "provider_response_format",
    # provider 只记录 Codex、A1 集群或 ARCHE_CHEM 的传输类型，不保存地址和认证。
    "provider",
    "response_schema_hash",
    # 旧的展示协议曾直接保存校验规格；当前 compiled request 通常只保存 hash，
    # 但保留这个白名单位置可以安全接收同一阶段的兼容投影。
    "response_contract",
)
_MESSAGE_TOOL_FIELDS = ("id", "type", "function")
_FUNCTION_FIELDS = ("name", "arguments")
_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)
_RESPONSE_CONTRACT_FIELDS = (
    "validation_kind",
    "schema",
    "domain_validator_id",
)
_VALIDATION_FIELDS = ("status", "failure_kind", "message")
_VALIDATION_STATUS_VALUES = frozenset(
    {"valid", "invalid", "transport_failed", "transport", "protocol_failed"}
)
_SENSITIVE_MESSAGE_PATTERN = re.compile(
    r"(?i)(authorization|bearer\s+|api[_-]?key|access[_-]?token|auth[_-]?token|cookie|password|secret)"
)
_REQUEST_MESSAGE_FIELDS = ("role", "content", "tool_calls")
_RESPONSE_FIELDS = (
    "id",
    "object",
    "created",
    "model",
    "provider",
    "choices",
    "usage",
    "system_fingerprint",
    "service_tier",
)
_RESPONSE_MESSAGE_FIELDS = (
    "role",
    "content",
    "reasoning_content",
    "refusal",
    "tool_calls",
)
_CHOICE_FIELDS = ("index", "finish_reason", "message", "logprobs")
_SOURCE_ARTIFACT_ALIASES = frozenset(
    {"base_input", "manifest", "request_files", "response_files"}
)
_ARTIFACT_FIELDS = frozenset({"relative_path", "sha256", "media_type", "size"})
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class UnsafeModelOutputPath(ValueError):
    """请求写入模型展示目录之外的相对路径。"""


@dataclass(frozen=True)
class ModelStageHandle:
    """一个已创建模型阶段文件的稳定引用。"""

    relative_path: str
    sequence: int
    phase: str
    stage: str


def _utc_timestamp() -> str:
    """返回带毫秒和 ``Z`` 后缀的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _copy_scalar(value: Any) -> Any:
    """只复制 provider 身份类字段允许的 JSON 标量。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _copy_json_value(value: Any) -> Any:
    """深拷贝科学正文和任务元数据，不依据字段名删除合法内容。"""

    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            copied[str(key)] = _copy_json_value(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _copy_messages(value: Any) -> Any:
    """按固定字段清单保存模型消息，不保存消息对象里的额外字段。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    messages: list[Any] = []
    for message in value:
        if not isinstance(message, Mapping):
            messages.append(_copy_json_value(message))
            continue
        copied: dict[str, Any] = {}
        for field in _REQUEST_MESSAGE_FIELDS:
            if field not in message:
                continue
            if field == "role":
                copied[field] = _copy_scalar(message[field])
            elif field == "tool_calls":
                # tool_calls 是协议结构；畸形值置为空数组，避免把其中的认证
                # 头等未知字段当作模型正文深拷贝。
                copied[field] = _copy_tool_calls(message[field])
            else:
                copied[field] = _copy_json_value(message[field])
        messages.append(copied)
    return messages


def _copy_tool_calls(value: Any) -> list[Any]:
    """按固定字段清单保存消息中的工具调用描述。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    copied_calls: list[Any] = []
    for call in value:
        if not isinstance(call, Mapping):
            copied_calls.append(_copy_json_value(call))
            continue
        copied: dict[str, Any] = {}
        for field in _MESSAGE_TOOL_FIELDS:
            if field not in call:
                continue
            if field == "function" and isinstance(call[field], Mapping):
                function: dict[str, Any] = {}
                if "name" in call[field]:
                    function["name"] = _copy_scalar(call[field]["name"])
                if "arguments" in call[field]:
                    # arguments 是模型正文，不按字段名递归删除。
                    function["arguments"] = _copy_json_value(call[field]["arguments"])
                copied[field] = function
            else:
                copied[field] = _copy_scalar(call[field])
        copied_calls.append(copied)
    return copied_calls


def _copy_request(value: Any) -> dict[str, Any]:
    """只保存 request 协议允许的模型字段。"""

    if not isinstance(value, Mapping):
        return {}
    copied: dict[str, Any] = {}
    for field in _REQUEST_FIELDS:
        if field not in value:
            continue
        if field == "messages":
            copied[field] = _copy_messages(value[field])
        elif field == "response_contract" and isinstance(value[field], Mapping):
            copied[field] = {
                name: _copy_json_value(value[field][name])
                for name in _RESPONSE_CONTRACT_FIELDS
                if name in value[field]
            }
        else:
            copied[field] = _copy_json_value(value[field])
    return copied


def _copy_response(value: Any) -> Any:
    """保存 provider 返回的固定安全 envelope 字段和模型正文。"""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        return {}
    copied: dict[str, Any] = {}
    for field in _RESPONSE_FIELDS:
        if field not in value:
            continue
        if field == "choices":
            # choices 的非数组形态不是模型正文；不要 generic 深拷贝其中可能伪装的
            # header、cookie 或 token。畸形响应用空数组保留 envelope 形状。
            if not isinstance(value[field], Sequence) or isinstance(
                value[field], (str, bytes, bytearray)
            ):
                copied[field] = []
                continue
            choices: list[Any] = []
            for choice in value[field]:
                if not isinstance(choice, Mapping):
                    choices.append(_copy_json_value(choice))
                    continue
                copied_choice: dict[str, Any] = {}
                for choice_field in _CHOICE_FIELDS:
                    if choice_field not in choice:
                        continue
                    if choice_field == "message":
                        choice_message = choice[choice_field]
                        if not isinstance(choice_message, Mapping):
                            # message 的非对象形态不是模型正文；不要把其中可能
                            # 携带的认证字段 generic 深拷贝进展示文件。
                            copied_choice[choice_field] = {}
                            continue
                        copied_message: dict[str, Any] = {}
                        for name in _RESPONSE_MESSAGE_FIELDS:
                            if name not in choice_message:
                                continue
                            if name in {"role", "refusal"}:
                                copied_message[name] = _copy_scalar(choice_message[name])
                            elif name == "tool_calls":
                                copied_message[name] = _copy_tool_calls(
                                    choice_message[name]
                                )
                            else:
                                # content/reasoning_content 是模型正文，保留其中
                                # 名为 token、secret 等普通科学字段的原样内容。
                                copied_message[name] = _copy_json_value(
                                    choice_message[name]
                                )
                        copied_choice[choice_field] = copied_message
                    elif choice_field == "index":
                        choice_value = choice[choice_field]
                        if isinstance(choice_value, int) and not isinstance(
                            choice_value, bool
                        ):
                            copied_choice[choice_field] = choice_value
                    elif choice_field == "finish_reason":
                        choice_value = choice[choice_field]
                        if choice_value is None or isinstance(choice_value, str):
                            copied_choice[choice_field] = choice_value
                    elif choice_field == "logprobs":
                        choice_value = choice[choice_field]
                        if choice_value is None or isinstance(choice_value, Mapping):
                            # logprobs 是 provider 的已知扩展；未知外壳字段不进入展示。
                            if isinstance(choice_value, Mapping):
                                copied_choice[choice_field] = {
                                    name: _copy_json_value(choice_value[name])
                                    for name in ("content", "refusal")
                                    if name in choice_value
                                }
                            else:
                                copied_choice[choice_field] = None
                    else:
                        copied_choice[choice_field] = _copy_json_value(
                            choice[choice_field]
                        )
                choices.append(copied_choice)
            copied[field] = choices
        elif field == "usage":
            # usage 的非对象形态同样不应进入通用深拷贝；空对象表示当前没有
            # 可验证的 token 用量。
            if not isinstance(value[field], Mapping):
                copied[field] = {}
                continue
            copied[field] = {
                name: value[field][name]
                for name in _USAGE_FIELDS
                if (
                    name in value[field]
                    and isinstance(value[field][name], int)
                    and not isinstance(value[field][name], bool)
                    and value[field][name] >= 0
                )
            }
        elif field in {"id", "object", "created", "model", "provider", "system_fingerprint", "service_tier"}:
            scalar = _copy_scalar(value[field])
            if scalar is not None:
                copied[field] = scalar
        else:
            copied[field] = _copy_json_value(value[field])
    return copied


def _copy_validation(value: Any) -> Any:
    """按统一阶段协议保存格式校验结果。"""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    copied: dict[str, Any] = {}
    if "status" in value and isinstance(value["status"], str):
        copied["status"] = value["status"]
    if "failure_kind" in value and isinstance(value["failure_kind"], str):
        failure_kind = value["failure_kind"]
        if failure_kind in _VALIDATION_STATUS_VALUES or re.fullmatch(
            r"[a-z][a-z0-9_.-]{0,63}", failure_kind
        ):
            copied["failure_kind"] = failure_kind
    if "message" in value:
        message = value["message"]
        if message is None:
            copied["message"] = None
        elif isinstance(message, str) and not _SENSITIVE_MESSAGE_PATTERN.search(message):
            copied["message"] = message
    return copied


def _copy_attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    """按统一阶段 JSON 的 attempt 字段整理一次模型回答。"""

    allowed_fields = (
        "attempt_index",
        "started_at",
        "finished_at",
        "duration_ms",
        "provider",
        "model",
        "provider_trace_id",
        "provider_request_id",
        "usage",
        "request",
        "raw_response",
        "parsed_output",
        "validation",
    )
    copied: dict[str, Any] = {
        "attempt_index": None,
        "started_at": None,
        "finished_at": None,
        "duration_ms": None,
        "provider": None,
        "model": None,
        "provider_trace_id": None,
        "provider_request_id": None,
        "usage": None,
        "request": {},
        "raw_response": {},
        "parsed_output": None,
        "validation": None,
    }
    for field in allowed_fields:
        if field not in value:
            continue
        item = value[field]
        if field == "request":
            copied[field] = _copy_request(item)
        elif field == "raw_response":
            copied[field] = _copy_response(item)
        elif field == "usage":
            if item is None:
                copied[field] = None
            elif isinstance(item, Mapping):
                copied[field] = {
                    name: item[name]
                    for name in _USAGE_FIELDS
                    if (
                        name in item
                        and isinstance(item[name], int)
                        and not isinstance(item[name], bool)
                        and item[name] >= 0
                    )
                }
            else:
                copied[field] = None
        elif field in {
            "provider",
            "model",
            "provider_trace_id",
            "provider_request_id",
        }:
            copied[field] = _copy_scalar(item) if item is None or isinstance(item, str) else None
        elif field == "validation":
            copied[field] = _copy_validation(item)
        else:
            copied[field] = _copy_json_value(item)
    return copied


def _validate_artifact_reference(
    value: Any,
) -> dict[str, Any]:
    """验证相对于正式 task root 的 ArtifactRef，不打开或复制目标文件。"""

    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_FIELDS:
        raise ValueError("ArtifactRef 必须恰含 relative_path/sha256/media_type/size")
    relative_path = value["relative_path"]
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or re.match(r"^[A-Za-z]:[/\\]", relative_path) is not None
    ):
        raise ValueError("ArtifactRef.relative_path 必须是非空 POSIX 相对路径")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("ArtifactRef.relative_path 必须是 task root 内的 POSIX 相对路径")
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("ArtifactRef.sha256 必须是 64 位十六进制字符串")
    media_type = value["media_type"]
    if not isinstance(media_type, str) or not media_type.strip():
        raise ValueError("ArtifactRef.media_type 必须是非空字符串")
    size = value["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("ArtifactRef.size 必须是不小于 0 的整数")
    return {
        "relative_path": relative.as_posix(),
        "sha256": sha256,
        "media_type": media_type,
        "size": size,
    }


def _validate_source_artifacts(
    value: Any,
) -> dict[str, Any]:
    """验证阶段允许的四类 ArtifactRef 引用结构。"""

    if not isinstance(value, Mapping):
        raise ValueError("source_artifacts 必须是对象")
    unknown = set(value) - _SOURCE_ARTIFACT_ALIASES
    if unknown:
        raise ValueError(f"source_artifacts 含未知别名: {sorted(map(str, unknown))}")
    validated: dict[str, Any] = {}
    for alias, references in value.items():
        if alias in {"request_files", "response_files"}:
            if not isinstance(references, list):
                raise ValueError(f"{alias} 必须是 ArtifactRef 数组")
            validated[alias] = [
                _validate_artifact_reference(
                    reference,
                )
                for reference in references
            ]
        else:
            validated[alias] = _validate_artifact_reference(
                references,
            )
    return validated


class ModelRunOutputWriter:
    """把模型阶段展示副本安全写入当前运行 Bundle。

    输入是一个工作目录、research_task 身份和可选任务元数据。实例创建时会创建任务根目录、
    ``task.json`` 和 ``index.json``；``resume=True`` 时读取已有文件并追加恢复事件。
    每个模型阶段由 :meth:`start_stage`、:meth:`append_attempt` 和
    :meth:`finish_stage` 完成。写入失败只把实例和可写的任务文件标记为
    ``degraded``，记录日志并返回失败值，不向正式科学调用方抛出 I/O 异常。
    并发保证限定为同一个 research_task 的单个 writer 实例；跨进程和多个 writer 实例的锁由
    外层 Memory On 任务锁负责，本模块不伪装提供这项能力。
    """

    def __init__(
        self,
        work_dir: str | Path,
        research_task_id: str,
        *,
        bundle_root: str | Path | None = None,
        persistence_mode: str = "memory_off",
        scientific_question: str | None = None,
        task_config: Mapping[str, Any] | None = None,
        model_configuration: Mapping[str, Any] | None = None,
        tool_context: Mapping[str, Any] | None = None,
        source_locations: Mapping[str, Any] | None = None,
        resume: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        if persistence_mode not in {"memory_on", "memory_off"}:
            raise ValueError("persistence_mode 必须是 memory_on 或 memory_off")
        self.work_dir = Path(work_dir).expanduser().resolve(strict=False)
        self.research_task_id = self._validate_research_task_id(research_task_id)
        self.persistence_mode = persistence_mode
        expected_bundle_root = (
            self.work_dir / "outputs" / "research_tasks" / self.research_task_id
        )
        if bundle_root is not None:
            requested_bundle_root = Path(bundle_root).expanduser().resolve(strict=False)
            if requested_bundle_root != expected_bundle_root.resolve(strict=False):
                raise ValueError(
                    "bundle_root 必须等于 work_dir/outputs/research_tasks/<research_task_id>"
                )
        self.bundle_root = expected_bundle_root
        self.output_root = self.bundle_root
        self.task_path = self.output_root / "internal" / "audit" / "task.json"
        self.index_path = self.output_root / "internal" / "audit" / "index.json"
        self._bundle: RunBundle | None = None
        self.logger = logger or LOGGER
        self._lock = threading.RLock()
        self._degraded = False
        self._error_message: str | None = None
        self._task: dict[str, Any] = {}
        self._index: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "research_task_id": self.research_task_id,
            "updated_at": None,
            "calls": [],
        }
        self._next_sequence = 1

        if resume:
            self._resume_existing_task()
        else:
            self._ensure_bundle_root()
            self._start_new_task(
                scientific_question=scientific_question,
                task_config=task_config,
                model_configuration=model_configuration,
                tool_context=tool_context,
                source_locations=source_locations,
            )

    def _ensure_bundle_root(self) -> None:
        """确保 writer 使用的 research_task 根已经具备固定 Bundle 子目录。"""

        if self.bundle_root.exists():
            self._bundle = RunBundle.open_existing_research_task(
                self.work_dir, self.research_task_id
            )
        else:
            self._bundle = RunBundle(
                self.work_dir,
                self.research_task_id,
                scientific_question=None,
                task_config=None,
                model_configuration=None,
            )

    @property
    def bundle(self) -> RunBundle | None:
        """返回当前 Bundle 管理器，供 Controller 更新状态摘要。"""

        return self._bundle

    @staticmethod
    def _validate_research_task_id(research_task_id: str) -> str:
        """拒绝会逃逸展示根目录的 research_task 身份。"""

        if not isinstance(research_task_id, str) or not _RESEARCH_TASK_ID_PATTERN.fullmatch(
            research_task_id
        ):
            raise ValueError(
                "research_task_id 只能包含字母、数字、点、下划线和连字符"
            )
        return research_task_id

    def _mark_degraded(self, error: Exception | str, *, path: Path | None = None) -> None:
        """在内存中记录展示降级，并写入不含敏感值的日志。"""

        self._degraded = True
        self._error_message = "模型输出展示文件写入失败"
        location = str(path) if path is not None else str(self.output_root)
        self.logger.error("模型输出展示文件写入失败（路径=%s；原因已隐藏）", location)
        # 如果 task.json 自身仍可写，及时留下 degraded 标记；不递归处理 task.json
        # 的失败，避免展示降级反过来阻塞正式科学调用。
        if self._task and path != self.task_path and self.task_path.exists():
            self._task["model_output_status"] = "degraded"
            try:
                self._atomic_write_json(self.task_path, self._task)
            except Exception:
                pass

    def mark_model_output_degraded(
        self, error: Exception | str = "外部观察器写入失败"
    ) -> None:
        """让外部模型观察器显式报告展示失败而不泄露异常正文。

        输入可以是异常对象或内部错误说明；说明只用于日志判断，不会写入
        ``task.json`` 或 ``result_summary()``。方法不改变科学任务状态。
        """

        with self._lock:
            self._mark_degraded(error)
            if self._task:
                self._task["model_output_status"] = "degraded"
                # task.json 本身也可能不可写；失败仍然只保留内存状态。
                self._write_json_safely(self.task_path, self._task)

    def _create_output_root(self) -> bool:
        """创建任务根目录；失败时仅报告展示降级。"""

        try:
            if not self._validate_storage_parent_chain(require_existing=False):
                return False
            if self.output_root.is_symlink():
                self._mark_degraded("展示根目录不能是符号链接", path=self.output_root)
                return False
            self.output_root.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as error:  # pragma: no cover - 由故障注入测试覆盖
            self._mark_degraded(error, path=self.output_root)
            return False

    def _validate_storage_parent_chain(self, *, require_existing: bool) -> bool:
        """检查 outputs/research_tasks 父链不含符号链接且仍位于工作目录内。

        新任务允许父目录尚未创建；恢复任务则要求父目录已经存在。恢复前先做这项
        检查，避免通过被替换的父目录把 task/index 写到工作目录之外。
        """

        storage_root = self.work_dir / "outputs" / "research_tasks"
        for parent in (self.work_dir / "outputs", storage_root):
            if parent.is_symlink():
                self._mark_degraded("展示目录父路径不能是符号链接", path=parent)
                return False
            if not parent.exists():
                if require_existing:
                    self._mark_degraded("恢复所需展示父目录不存在", path=parent)
                    return False
                continue
            if not parent.is_dir():
                self._mark_degraded("展示目录父路径不是目录", path=parent)
                return False
            try:
                parent.resolve(strict=True).relative_to(self.work_dir)
            except (OSError, ValueError) as error:
                self._mark_degraded(error, path=parent)
                return False
        return True

    def _start_new_task(
        self,
        *,
        scientific_question: str | None,
        task_config: Mapping[str, Any] | None,
        model_configuration: Mapping[str, Any] | None,
        tool_context: Mapping[str, Any] | None,
        source_locations: Mapping[str, Any] | None,
    ) -> None:
        """创建新 research_task 的初始文件和 started 生命周期事件。"""

        with self._lock:
            if not self._create_output_root():
                return
            if self.task_path.exists() or self.index_path.exists():
                self._mark_degraded("任务展示目录已经存在", path=self.output_root)
                return
            started_at = _utc_timestamp()
            self._task = {
                "schema_version": _SCHEMA_VERSION,
                "research_task_id": self.research_task_id,
                "persistence_mode": self.persistence_mode,
                "scientific_question": scientific_question,
                "status": "running",
                "started_at": started_at,
                "finished_at": None,
                "resume_count": 0,
                "task_config": _copy_json_value(task_config or {}),
                "model_configuration": _copy_json_value(model_configuration or {}),
                "tool_context": _copy_json_value(tool_context or {}),
                "source_locations": _copy_json_value(
                    source_locations
                    or {
                        "original_task_root": None,
                        "working_memory_database": None,
                    }
                ),
                "model_output_status": "degraded" if self._degraded else "available",
                "model_output_root": (
                    Path("outputs")
                    / "research_tasks"
                    / self.research_task_id
                    / "internal"
                    / "audit"
                ).as_posix(),
                "lifecycle": [
                    {"event": "started", "at": started_at, "reason_code": None}
                ],
            }
            self._index["updated_at"] = started_at
            self._write_task_and_index()
            if self._bundle is not None:
                try:
                    self._bundle.update_run_metadata(
                        persistence_mode=self.persistence_mode,
                        task_config=task_config,
                        model_configuration=model_configuration,
                        source_locations=source_locations,
                    )
                except Exception:
                    self._mark_degraded("运行 Bundle 元数据写入失败", path=self.output_root)

    def _resume_existing_task(self) -> None:
        """读取已有 research_task/index，继续 sequence 并追加 resumed 生命周期事件。"""

        with self._lock:
            try:
                self._bundle = RunBundle.open_existing_research_task(
                    self.work_dir, self.research_task_id
                )
            except ValueError:
                self._bundle = None
            # 恢复只能读取已有 task；缺失目录或文件不能用 mkdir 和初始 JSON
            # 伪造一个不存在的运行。
            if not self._validate_storage_parent_chain(require_existing=True):
                return
            if self.output_root.is_symlink() or not self.output_root.is_dir():
                self._mark_degraded("恢复目录不存在", path=self.output_root)
                return
            if (
                self.task_path.is_symlink()
                or self.index_path.is_symlink()
                or not self.task_path.is_file()
                or not self.index_path.is_file()
            ):
                self._mark_degraded("恢复所需文件不存在", path=self.output_root)
                return
            task = self._read_json_file(self.task_path)
            index = self._read_json_file(self.index_path)
            if not isinstance(task, dict) or not isinstance(index, dict):
                self._mark_degraded("恢复所需文件不存在或格式错误", path=self.output_root)
                return
            validation_error = self._validate_resume_documents(task, index)
            if validation_error is not None:
                self._mark_degraded(validation_error, path=self.output_root)
                return
            calls = index["calls"]
            self._task = task
            self._index = index
            self._index.setdefault("schema_version", _SCHEMA_VERSION)
            self._index["research_task_id"] = self.research_task_id
            self._next_sequence = self._find_next_sequence(calls)
            self._task["resume_count"] = int(self._task.get("resume_count", 0)) + 1
            resumed_at = _utc_timestamp()
            lifecycle = self._task.setdefault("lifecycle", [])
            if not isinstance(lifecycle, list):
                lifecycle = []
                self._task["lifecycle"] = lifecycle
            lifecycle.append(
                {"event": "resumed", "at": resumed_at, "reason_code": None}
            )
            self._task["status"] = "running"
            self._task["finished_at"] = None
            self._task["model_output_status"] = (
                "degraded" if self._degraded or self._task.get("model_output_status") == "degraded" else "available"
            )
            self._index["updated_at"] = resumed_at
            self._write_task_and_index()

    def update_run_bundle(
        self,
        result: Mapping[str, Any],
        *,
        stage: str | None = None,
        stage_summary: Mapping[str, Any] | None = None,
    ) -> bool:
        """同步更新 Bundle 的 status.json、summary.json 和阶段摘要。"""

        with self._lock:
            bundle = self._bundle
            if bundle is None:
                return False
            status = result.get("status") if isinstance(result, Mapping) else None
            if isinstance(status, str):
                if status not in _ALLOWED_TASK_STATUSES:
                    raise ValueError(f"research_task 状态非法: {status!r}")
                recovery = result.get("recovery")
                # 展示 writer 不拥有恢复事实；V2 的等待作业由 Working Memory
                # 保存，Bundle 这里只接收已经形成的 canonical recovery 视图。
                # 尤其不能从旧运行时检查点猜测 waiting 状态或生成历史字段。
                bundle.update_status(
                    status,
                    reason_code=(
                        result.get("reason_code")
                        if isinstance(result.get("reason_code"), str)
                        else None
                    ),
                    blocking=(result.get("blocking") if isinstance(result.get("blocking"), Mapping) else None),
                    recovery=(dict(recovery) if isinstance(recovery, Mapping) else None),
                )
            bundle.update_summary(result=result, stage=stage, stage_summary=stage_summary)
            return True

    def synchronize_bundle_state(
        self,
        status: str,
        *,
        reason_code: str | None = None,
        blocking: Mapping[str, Any] | None = None,
        recovery: Mapping[str, Any] | None = None,
        stage: str | None = None,
        stage_status: str | None = None,
        stage_summary: Mapping[str, Any] | None = None,
    ) -> bool:
        """在 writer 锁内同步运行状态和固定阶段状态。

        闭环组合可能并行推进；该方法把同一个 writer 实例上的 Bundle 读改写操作
        放在同一把重入锁中，避免阶段条目或状态事件因并发写入而互相覆盖。
        """

        with self._lock:
            bundle = self._bundle
            if bundle is None:
                return False
            bundle.update_status(
                status,
                reason_code=reason_code,
                blocking=blocking,
                recovery=recovery,
            )
            summary_result: dict[str, Any] = {"status": status}
            if recovery is not None:
                summary_result["recovery"] = recovery
            bundle.update_summary(result=summary_result)
            if stage is not None and stage_status is not None:
                bundle.update_stage_status(
                    stage,
                    stage_status,
                    summary=stage_summary,
                    blocking=blocking,
                    recovery=recovery,
                )
            return True

    @staticmethod
    def _find_next_sequence(calls: Sequence[Any]) -> int:
        """从 index 中找出下一个可用的调用 sequence。"""

        maximum = 0
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            sequence = call.get("sequence")
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                maximum = max(maximum, sequence)
        return maximum + 1

    def _read_json_file(self, path: Path) -> Any:
        """读取 JSON 文件；读取错误只返回空值并记录展示降级。"""

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            self._mark_degraded(error, path=path)
            return None

    def _validate_resume_documents(
        self, task: Mapping[str, Any], index: Mapping[str, Any]
    ) -> str | None:
        """检查恢复所需的身份、生命周期和调用序列，失败时返回白话原因。"""

        required_task_fields = {
            "schema_version",
            "research_task_id",
            "persistence_mode",
            "status",
            "resume_count",
            "lifecycle",
        }
        if task.get("schema_version") != _SCHEMA_VERSION:
            return "task.json schema_version 不匹配"
        if index.get("schema_version") != _SCHEMA_VERSION:
            return "index.json schema_version 不匹配"
        if task.get("research_task_id") != self.research_task_id:
            return "task.json research_task_id 不匹配"
        if index.get("research_task_id") != self.research_task_id:
            return "index.json research_task_id 不匹配"
        if task.get("persistence_mode") != self.persistence_mode:
            return "task.json persistence_mode 不匹配"
        if not required_task_fields.issubset(task):
            return "task.json 缺少恢复字段"
        status = task.get("status")
        if status not in _ALLOWED_TASK_STATUSES:
            return "task.json status 非法"
        resume_count = task.get("resume_count")
        if isinstance(resume_count, bool) or not isinstance(resume_count, int) or resume_count < 0:
            return "task.json resume_count 非法"
        lifecycle = task.get("lifecycle")
        if not isinstance(lifecycle, list):
            return "task.json lifecycle 不是数组"
        resumed_count = 0
        for event in lifecycle:
            if not isinstance(event, Mapping):
                return "task.json lifecycle 项不是对象"
            if not isinstance(event.get("event"), str) or not isinstance(event.get("at"), str):
                return "task.json lifecycle 项缺少 event/at"
            if "reason_code" not in event:
                return "task.json lifecycle 项缺少 reason_code"
            if event["event"] == "resumed":
                resumed_count += 1
        if resumed_count != resume_count:
            return "task.json resume_count 与 lifecycle 不一致"
        calls = index.get("calls")
        if not isinstance(calls, list):
            return "index.json calls 不是数组"
        sequences: list[int] = []
        files: set[str] = set()
        for call in calls:
            if not isinstance(call, Mapping):
                return "index.json calls 项不是对象"
            sequence = call.get("sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                return "index.json sequence 非法"
            if sequence in sequences:
                return "index.json sequence 重复"
            relative_file = call.get("file")
            try:
                relative_text, _target = self._validate_relative_path(relative_file)
            except (TypeError, ValueError):
                return "index.json file 路径非法"
            if relative_text in files:
                return "index.json file 重复"
            files.add(relative_text)
            status = call.get("status")
            if status not in _ALLOWED_STAGE_STATUSES:
                return "index.json call status 非法"
            attempt_count = call.get("attempt_count")
            if (
                isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count < 0
            ):
                return "index.json attempt_count 非法"
            sequences.append(sequence)
        if sequences != sorted(sequences):
            return "index.json sequence 未按调用顺序排列"
        return None

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        """以同目录临时文件、fsync 和原子替换写入一份 JSON。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(
                file_descriptor, "w", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            self._fsync_directory(path.parent)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """刷新父目录，确保原子替换后的目录项已经落盘。"""

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_json_safely(self, path: Path, payload: Any) -> bool:
        """写一份展示 JSON；失败时返回 False 而不把异常传给科学调用方。"""

        try:
            self._atomic_write_json(path, payload)
            return True
        except Exception as error:  # pragma: no cover - 由故障注入测试覆盖
            self._mark_degraded(error, path=path)
            return False

    def _write_task_and_index(self) -> bool:
        """按固定顺序原子更新 task.json 与 index.json。"""

        task_ok = self._write_json_safely(self.task_path, self._task)
        index_ok = self._write_json_safely(self.index_path, self._index)
        if not task_ok or not index_ok:
            self._task["model_output_status"] = "degraded"
        return task_ok and index_ok

    def _validate_relative_path(self, relative_path: str | Path) -> tuple[str, Path]:
        """验证相对路径不会越过当前运行目录，也不会跟随外部符号链接。"""

        if not isinstance(relative_path, (str, Path)):
            raise UnsafeModelOutputPath("展示文件路径必须是字符串或 Path")
        text = str(relative_path)
        if not text or "\\" in text:
            raise UnsafeModelOutputPath("展示文件路径必须是非空 POSIX 相对路径")
        relative = PurePosixPath(text)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise UnsafeModelOutputPath("展示文件路径不能是绝对路径或包含 .、.. 组件")
        normalized_parts = list(relative.parts)
        # 编号阶段调用继续传入 01_retrieval、03_hypothesis 等相对位置；这些文件含
        # request、response 和 receipts，必须统一收纳到 internal/audit/model_calls，
        # stages/ 只保留精简阶段摘要。
        first = normalized_parts[0] if normalized_parts else ""
        if first.startswith(tuple(f"{index:02d}_" for index in range(1, 10))):
            normalized_parts = ["internal", "audit", "model_calls", *normalized_parts]
        target = self.output_root.joinpath(*normalized_parts)
        try:
            target.resolve(strict=False).relative_to(self.output_root)
        except ValueError as error:
            raise UnsafeModelOutputPath("展示文件路径越过运行目录") from error
        return PurePosixPath(*normalized_parts).as_posix(), target

    def _reserve_sequence(self) -> int:
        """在锁内分配不会与其他线程重复的 sequence。"""

        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    @staticmethod
    def _make_call_identity(
        *,
        sequence: int,
        research_task_id: str,
        phase: str,
        stage: str,
        operation_id: str | None,
        batch_index: int | None,
        query_index: int | None,
        combination_index: int | None,
        round_index: int | None,
    ) -> dict[str, Any]:
        """建立统一阶段 JSON 的调用位置字段。"""

        return {
            "sequence": sequence,
            "research_task_id": research_task_id,
            "phase": phase,
            "stage": stage,
            "operation_id": operation_id,
            "batch_index": batch_index,
            "query_index": query_index,
            "combination_index": combination_index,
            "round_index": round_index,
        }

    def start_stage(
        self,
        phase: str,
        stage: str,
        relative_path: str | Path,
        *,
        operation_id: str | None = None,
        batch_index: int | None = None,
        query_index: int | None = None,
        combination_index: int | None = None,
        round_index: int | None = None,
        input: Any = None,
        source_artifacts: Any = None,
        allow_terminal_operation_retry: bool = False,
    ) -> ModelStageHandle | None:
        """创建一个 running 阶段文件并在 index 中分配 sequence。

        输入是业务阶段、阶段名称、展示 JSON 的相对路径和模型实际看到的结构化输入。
        方法只创建该调用需要的父目录，不预先创建空阶段目录；如果路径不安全会抛出
        :class:`UnsafeModelOutputPath`，如果文件系统写入失败则返回 ``None`` 并标记
        ``degraded``。
        """

        relative_text, target = self._validate_relative_path(relative_path)
        if not phase or not stage:
            raise ValueError("phase 和 stage 不能为空")
        try:
            validated_source_artifacts = _validate_source_artifacts(
                source_artifacts if source_artifacts is not None else {},
            )
        except (TypeError, ValueError) as error:
            self._mark_degraded(error, path=target)
            return None
        with self._lock:
            if allow_terminal_operation_retry and target.exists():
                base_document = self._read_json_file(target)
                base_identity = (
                    base_document.get("call_identity")
                    if isinstance(base_document, Mapping)
                    else None
                )
                if (
                    isinstance(base_document, Mapping)
                    and base_document.get("status") in _TERMINAL_STAGE_STATUSES
                    and isinstance(base_identity, Mapping)
                    and isinstance(operation_id, str)
                    and operation_id
                    and base_identity.get("operation_id") != operation_id
                ):
                    base_relative = PurePosixPath(relative_text)
                    suffix = base_relative.suffix
                    stem = base_relative.name[: -len(suffix)] if suffix else base_relative.name
                    for retry_index in range(1, 10_000):
                        candidate_relative = base_relative.with_name(
                            f"{stem}.retry_{retry_index:03d}{suffix}"
                        )
                        candidate_text, candidate_target = self._validate_relative_path(
                            candidate_relative.as_posix()
                        )
                        if not candidate_target.exists():
                            relative_text, target = candidate_text, candidate_target
                            break
                        candidate_document = self._read_json_file(candidate_target)
                        candidate_identity = (
                            candidate_document.get("call_identity")
                            if isinstance(candidate_document, Mapping)
                            else None
                        )
                        if (
                            isinstance(candidate_identity, Mapping)
                            and candidate_identity.get("operation_id") == operation_id
                        ):
                            relative_text, target = candidate_text, candidate_target
                            break
                    else:
                        self._mark_degraded(
                            "同一阶段的 operation retry 展示文件过多",
                            path=target,
                        )
                        return None
            existing = None
            if target.exists():
                existing = self._read_json_file(target)
                if not isinstance(existing, Mapping):
                    self._mark_degraded("已有阶段文件不是 JSON object", path=target)
                    return None
                # 读取磁盘上的 index，避免上一次 index 写入失败后继续相信内存中
                # 已追加但尚未落盘的条目；这样同路径重试可以补登记。
                disk_index = self._read_json_file(self.index_path)
                if not isinstance(disk_index, Mapping) or not isinstance(
                    disk_index.get("calls"), list
                ):
                    self._mark_degraded("index.json 无法用于阶段恢复", path=self.index_path)
                    return None
                self._index = dict(disk_index)
                self._next_sequence = self._find_next_sequence(self._index["calls"])
                existing_status = existing.get("status")
                identity = existing.get("call_identity")
                if existing_status in _TERMINAL_STAGE_STATUSES:
                    self._mark_degraded("新调用撞到已完成阶段文件", path=target)
                    return None
                if existing_status != "running" or not isinstance(identity, Mapping):
                    self._mark_degraded("已有阶段文件状态或身份非法", path=target)
                    return None
                existing_sequence = identity.get("sequence")
                expected_identity_fields = {
                    "research_task_id": self.research_task_id,
                    "phase": phase,
                    "stage": stage,
                    "operation_id": operation_id,
                    "batch_index": batch_index,
                    "query_index": query_index,
                    "combination_index": combination_index,
                    "round_index": round_index,
                }
                if (
                    isinstance(existing_sequence, bool)
                    or not isinstance(existing_sequence, int)
                    or set(identity)
                    != {"sequence", *expected_identity_fields.keys()}
                    or any(
                        identity.get(field) != expected
                        for field, expected in expected_identity_fields.items()
                    )
                ):
                    self._mark_degraded("已有 running 阶段身份不匹配", path=target)
                    return None
                try:
                    saved_source_artifacts = _validate_source_artifacts(
                        existing.get("source_artifacts", {}),
                    )
                except (TypeError, ValueError) as error:
                    self._mark_degraded(error, path=target)
                    return None
                index_call = self._find_index_call(existing_sequence, relative_text)
                if index_call is None:
                    repaired_call = self._make_index_call_from_stage(
                        relative_text, existing, saved_source_artifacts
                    )
                    self._index.setdefault("calls", []).append(repaired_call)
                    self._index["calls"].sort(key=lambda call: call["sequence"])
                    self._next_sequence = max(
                        self._next_sequence, existing_sequence + 1
                    )
                    self._index["updated_at"] = _utc_timestamp()
                    if not self._write_json_safely(self.index_path, self._index):
                        return None
                elif not self._index_call_matches_stage(
                    index_call, relative_text, existing
                ):
                    if not self._index_call_identity_matches_stage(
                        index_call, relative_text, existing
                    ):
                        self._mark_degraded(
                            "已有 running 阶段 index 身份不匹配", path=self.index_path
                        )
                        return None
                    repaired_call = self._make_index_call_from_stage(
                        relative_text, existing, saved_source_artifacts
                    )
                    self._index["calls"] = [
                        repaired_call
                        if call is index_call
                        else call
                        for call in self._index["calls"]
                    ]
                    self._index["calls"].sort(key=lambda call: call["sequence"])
                    self._index["updated_at"] = _utc_timestamp()
                    if not self._write_json_safely(self.index_path, self._index):
                        return None
                return ModelStageHandle(
                    relative_path=relative_text,
                    sequence=existing_sequence,
                    phase=phase,
                    stage=stage,
                )
            if any(
                isinstance(call, Mapping) and call.get("file") == relative_text
                for call in self._index.get("calls", [])
            ):
                self._mark_degraded("index 已存在同路径调用但阶段文件缺失", path=self.index_path)
                return None
            sequence = self._reserve_sequence()
            started_at = _utc_timestamp()
            stage_document = {
                "schema_version": _SCHEMA_VERSION,
                "call_identity": self._make_call_identity(
                    sequence=sequence,
                    research_task_id=self.research_task_id,
                    phase=phase,
                    stage=stage,
                    operation_id=operation_id,
                    batch_index=batch_index,
                    query_index=query_index,
                    combination_index=combination_index,
                    round_index=round_index,
                ),
                "status": "running",
                "started_at": started_at,
                "finished_at": None,
                "input": _copy_json_value(input),
                "attempts": [],
                "final_output": None,
                "source_artifacts": validated_source_artifacts,
            }
            index_call = {
                "sequence": sequence,
                "phase": phase,
                "stage": stage,
                "batch_index": batch_index,
                "query_index": query_index,
                "combination_index": combination_index,
                "round_index": round_index,
                "operation_id": operation_id,
                "status": "running",
                "attempt_count": 0,
                "started_at": started_at,
                "finished_at": None,
                "file": relative_text,
            }
            if not self._write_json_safely(target, stage_document):
                return None
            self._index.setdefault("calls", []).append(index_call)
            self._index["updated_at"] = started_at
            if not self._write_json_safely(self.index_path, self._index):
                return None
            return ModelStageHandle(
                relative_path=relative_text,
                sequence=sequence,
                phase=phase,
                stage=stage,
            )

    def _coerce_stage_handle(
        self, stage: ModelStageHandle | str | Path
    ) -> tuple[str, Path]:
        """把阶段句柄或相对路径转换成受保护的文件路径。"""

        if isinstance(stage, ModelStageHandle):
            stage = stage.relative_path
        return self._validate_relative_path(stage)

    def _find_index_call(self, sequence: int, relative_path: str) -> dict[str, Any] | None:
        """找到 index 中对应阶段的定位条目。"""

        for call in self._index.get("calls", []):
            if not isinstance(call, dict):
                continue
            if call.get("sequence") == sequence or call.get("file") == relative_path:
                return call
        return None

    @staticmethod
    def _make_index_call_from_stage(
        relative_path: str,
        document: Mapping[str, Any],
        _source_artifacts: Mapping[str, Any],
    ) -> dict[str, Any]:
        """从已写入的 running 阶段恢复一次丢失的 index 定位条目。"""

        identity = document["call_identity"]
        attempts = document.get("attempts", [])
        return {
            "sequence": identity["sequence"],
            "phase": identity["phase"],
            "stage": identity["stage"],
            "batch_index": identity["batch_index"],
            "query_index": identity["query_index"],
            "combination_index": identity["combination_index"],
            "round_index": identity["round_index"],
            "operation_id": identity["operation_id"],
            "status": "running",
            "attempt_count": len(attempts) if isinstance(attempts, list) else 0,
            "started_at": document.get("started_at"),
            "finished_at": None,
            "file": relative_path,
        }

    @staticmethod
    def _index_call_matches_stage(
        index_call: Mapping[str, Any],
        relative_path: str,
        document: Mapping[str, Any],
    ) -> bool:
        """确认 index 条目和阶段文件的身份、路径、状态完全一致。"""

        identity = document.get("call_identity")
        attempts = document.get("attempts")
        if not isinstance(identity, Mapping) or not isinstance(attempts, list):
            return False
        expected = {
            "sequence": identity.get("sequence"),
            "phase": identity.get("phase"),
            "stage": identity.get("stage"),
            "batch_index": identity.get("batch_index"),
            "query_index": identity.get("query_index"),
            "combination_index": identity.get("combination_index"),
            "round_index": identity.get("round_index"),
            "operation_id": identity.get("operation_id"),
            "status": "running",
            "attempt_count": len(attempts),
            "started_at": document.get("started_at"),
            "finished_at": None,
            "file": relative_path,
        }
        return set(index_call) == set(expected) and all(
            index_call.get(field) == value for field, value in expected.items()
        )

    @staticmethod
    def _index_call_identity_matches_stage(
        index_call: Mapping[str, Any],
        relative_path: str,
        document: Mapping[str, Any],
    ) -> bool:
        """只比较 index 的身份字段，供写入失败后的补登记使用。"""

        identity = document.get("call_identity")
        if not isinstance(identity, Mapping):
            return False
        expected = {
            "sequence": identity.get("sequence"),
            "phase": identity.get("phase"),
            "stage": identity.get("stage"),
            "batch_index": identity.get("batch_index"),
            "query_index": identity.get("query_index"),
            "combination_index": identity.get("combination_index"),
            "round_index": identity.get("round_index"),
            "operation_id": identity.get("operation_id"),
            "file": relative_path,
        }
        return all(index_call.get(field) == value for field, value in expected.items())

    def _stage_reference_matches(
        self,
        stage: ModelStageHandle | str | Path,
        relative_path: str,
        document: Mapping[str, Any],
        index_call: Mapping[str, Any] | None,
    ) -> bool:
        """确认阶段句柄、阶段 JSON 和 index 三者引用同一个调用。"""

        identity = document.get("call_identity")
        if (
            document.get("schema_version") != _SCHEMA_VERSION
            or document.get("status") != "running"
            or not isinstance(identity, Mapping)
            or identity.get("research_task_id") != self.research_task_id
            or index_call is None
            or not self._index_call_matches_stage(index_call, relative_path, document)
        ):
            return False
        if isinstance(stage, ModelStageHandle):
            return (
                stage.relative_path == relative_path
                and stage.sequence == identity.get("sequence")
                and stage.phase == identity.get("phase")
                and stage.stage == identity.get("stage")
            )
        return True

    def append_attempt(
        self,
        stage: ModelStageHandle | str | Path,
        attempt: Mapping[str, Any],
    ) -> bool:
        """向 running 阶段追加一次模型回答并更新 index 的 attempt_count。

        输入是阶段句柄和 attempt 字段。request、raw_response 只按固定白名单写入，
        认证头、环境变量和密钥字段会被排除。阶段已进入终态时不再修改；文件系统失败
        返回 ``False``、记录日志并把结果摘要标成 ``degraded``。
        """

        if not isinstance(attempt, Mapping):
            raise TypeError("attempt 必须是对象")
        relative_text, target = self._coerce_stage_handle(stage)
        with self._lock:
            document = self._read_json_file(target)
            if not isinstance(document, dict):
                return False
            if document.get("status") != "running":
                return False
            identity = document.get("call_identity", {})
            sequence = identity.get("sequence") if isinstance(identity, Mapping) else None
            index_call = (
                self._find_index_call(sequence, relative_text)
                if isinstance(sequence, int)
                else None
            )
            if not self._stage_reference_matches(
                stage, relative_text, document, index_call
            ):
                self._mark_degraded("attempt 阶段身份或 index 不匹配", path=target)
                return False
            attempts = document.setdefault("attempts", [])
            if not isinstance(attempts, list):
                self._mark_degraded("阶段 attempts 不是数组", path=target)
                return False
            copied_attempt = _copy_attempt(attempt)
            requested_index = attempt.get("attempt_index")
            if requested_index is None:
                copied_attempt["attempt_index"] = len(attempts)
            elif (
                isinstance(requested_index, bool)
                or not isinstance(requested_index, int)
                or requested_index != len(attempts)
                or requested_index < 0
            ):
                self._mark_degraded("attempt_index 不连续", path=target)
                return False
            attempts.append(copied_attempt)
            if not self._write_json_safely(target, document):
                return False
            if index_call is not None:
                index_call["attempt_count"] = len(attempts)
                self._index["updated_at"] = _utc_timestamp()
                if not self._write_json_safely(self.index_path, self._index):
                    return False
            return True

    @staticmethod
    def _canonical_bundle_stage(phase: Any, stage: Any) -> str | None:
        """把模型调用阶段映射到 Bundle 的四个正式阶段名。"""

        phase_text = str(phase or "").lower()
        stage_text = str(stage or "").lower()
        text = f"{phase_text} {stage_text}"
        if phase_text in {"retrieval", "literature_retrieval"} or stage_text in {
            "retrieval",
            "literature_retrieval",
        }:
            return "literature_retrieval"
        if phase_text in {
            "research-query-formulation",
            "hypothesis-generation",
            "hypothesis-selection",
            "hypothesis-confirmation",
        } or stage_text in {
            "research-query-formulation",
            "hypothesis-generation",
            "hypothesis-selection",
            "hypothesis-confirmation",
        }:
            return "hypothesis_preparation"
        if phase_text in {
            "hypothesis-investigation",
            "investigation",
            "investigation-and-experiment",
            "investigation_and_experiment",
        } or stage_text in {
            "hypothesis-investigation",
            "investigation",
            "investigation-and-experiment",
            "investigation_and_experiment",
        }:
            return "investigation_and_experiment"
        if phase_text in {"task-review", "task_review_and_result"} or stage_text in {
            "task-review",
            "task_review_and_result",
        }:
            return "task_review_and_result"
        if phase_text in {
            "query",
            "hypothesis",
            "hypothesis_preparation",
            "router",
            "admission",
            "debate",
            "evidence",
            "scientific_evidence",
        } or "generate_queries" in stage_text or "hypothesis" in stage_text:
            return "hypothesis_preparation"
        if phase_text == "investigation_and_experiment":
            return "investigation_and_experiment"
        if phase_text == "task_review_and_result":
            return "task_review_and_result"
        if phase_text in {
            "expert_pre_execution",
            "expert_gaussian_review",
            "gaussian_step_review",
        } or "gaussian_step_review" in text or (
            phase_text == "expert" and any(token in text for token in ("pre", "gaussian"))
        ):
            return "investigation_and_experiment"
        if phase_text in {"expert_post_execution", "expert_debate"} or (
            phase_text == "expert" and any(token in text for token in ("post", "debate"))
        ):
            return "investigation_and_experiment"
        if phase_text in {
            "revision",
            "revise",
            "execution",
            "program_gate",
            "planning",
            "workflow_generation",
            "workflow_optimization",
        } or "revision" in stage_text or "execution" in stage_text:
            return "investigation_and_experiment"
        if phase_text in {"final", "final_result", "publish", "review"}:
            return "task_review_and_result"
        return None

    def _publish_bundle_stage_summary(
        self,
        document: Mapping[str, Any],
        status: str,
        source_file: str,
    ) -> None:
        """在模型阶段终态时写固定阶段的精简摘要。"""

        bundle = self._bundle
        if bundle is None:
            return
        identity = document.get("call_identity")
        if not isinstance(identity, Mapping):
            return
        stage_name = self._canonical_bundle_stage(identity.get("phase"), identity.get("stage"))
        if stage_name is None:
            return
        attempts = document.get("attempts")
        summary = {
            "status": status,
            "phase": identity.get("phase"),
            "stage": identity.get("stage"),
            "call_count": len(attempts) if isinstance(attempts, list) else 0,
            "operation_id": identity.get("operation_id"),
            "batch_index": identity.get("batch_index"),
            "query_index": identity.get("query_index"),
            "combination_index": identity.get("combination_index"),
            "round_index": identity.get("round_index"),
            "source_file": source_file,
            "final_output": _copy_json_value(document.get("final_output")) if status == "validated" else None,
        }
        try:
            bundle.write_stage_summary(
                stage_name,
                summary,
                source_file=source_file,
                call_identity=identity,
            )
        except Exception:
            self._mark_degraded("固定阶段摘要写入失败", path=bundle.root / "stages" / f"{stage_name}.json")

    def finish_stage(
        self,
        stage: ModelStageHandle | str | Path,
        status: str,
        *,
        final_output: Any = None,
        finished_at: str | None = None,
    ) -> bool:
        """把 running 阶段标记为 validated 或 failed，并更新 index。

        终态文件不可覆盖；失败阶段的 ``final_output`` 固定为 ``null``，避免把无效
        响应伪装成模型最终结果。写入失败返回 ``False`` 并进入 degraded 状态。
        """

        if status not in _TERMINAL_STAGE_STATUSES:
            raise ValueError("阶段终态只能是 validated 或 failed")
        relative_text, target = self._coerce_stage_handle(stage)
        with self._lock:
            document = self._read_json_file(target)
            if not isinstance(document, dict) or document.get("status") != "running":
                return False
            identity = document.get("call_identity", {})
            sequence = identity.get("sequence") if isinstance(identity, Mapping) else None
            index_call = (
                self._find_index_call(sequence, relative_text)
                if isinstance(sequence, int)
                else None
            )
            if not self._stage_reference_matches(
                stage, relative_text, document, index_call
            ):
                self._mark_degraded("finish 阶段身份或 index 不匹配", path=target)
                return False
            attempts = document.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                self._mark_degraded("终态阶段必须至少包含一次 attempt", path=target)
                return False
            try:
                _validate_source_artifacts(
                    document.get("source_artifacts", {}),
                )
            except (TypeError, ValueError) as error:
                self._mark_degraded(error, path=target)
                return False
            timestamp = finished_at or _utc_timestamp()
            document["status"] = status
            document["finished_at"] = timestamp
            document["final_output"] = (
                _copy_json_value(final_output) if status == "validated" else None
            )
            if not self._write_json_safely(target, document):
                return False
            if index_call is not None:
                index_call["status"] = status
                index_call["attempt_count"] = len(document.get("attempts", []))
                index_call["finished_at"] = timestamp
                self._index["updated_at"] = timestamp
                if not self._write_json_safely(self.index_path, self._index):
                    return False
            self._publish_bundle_stage_summary(document, status, relative_text)
            return True

    def update_task(self, updates: Mapping[str, Any]) -> bool:
        """更新 task.json 中允许改变的字段并原子保存。"""

        if not isinstance(updates, Mapping):
            raise TypeError("task 更新必须是对象")
        allowed_fields = {
            "scientific_question",
            "task_config",
            "model_configuration",
            "tool_context",
            "source_locations",
            "status",
            "finished_at",
            "resume_count",
            "lifecycle",
            "model_output_root",
        }
        unknown = set(updates) - allowed_fields
        if unknown:
            raise ValueError(f"task 更新包含不允许的字段: {sorted(map(str, unknown))}")
        with self._lock:
            self._task.update(
                {
                    str(field): _copy_json_value(value)
                    for field, value in updates.items()
                }
            )
            self._task["model_output_status"] = (
                "degraded" if self._degraded else self._task.get("model_output_status", "available")
            )
            return self._write_json_safely(self.task_path, self._task)

    def update_task_status(self, status: str, reason_code: str | None = None) -> bool:
        """更新 task 状态并追加 paused、resumed 或 finished 生命周期事件。"""

        if status not in _ALLOWED_TASK_STATUSES:
            raise ValueError(f"不支持的 task 状态: {status}")
        with self._lock:
            timestamp = _utc_timestamp()
            event = "finished" if status in {"succeeded", "failed", "stopped_no_supported_hypothesis"} else status
            lifecycle = self._task.setdefault("lifecycle", [])
            if not isinstance(lifecycle, list):
                lifecycle = []
                self._task["lifecycle"] = lifecycle
            lifecycle.append(
                {"event": event, "at": timestamp, "reason_code": reason_code}
            )
            self._task["status"] = status
            self._task["finished_at"] = (
                timestamp if status in {"succeeded", "failed", "stopped_no_supported_hypothesis"} else None
            )
            self._task["model_output_status"] = (
                "degraded" if self._degraded else self._task.get("model_output_status", "available")
            )
            return self._write_json_safely(self.task_path, self._task)

    def resume_task(self, reason_code: str | None = None) -> bool:
        """在当前实例上追加恢复事件并从 index 最大 sequence 继续。"""

        with self._lock:
            self._task["resume_count"] = int(self._task.get("resume_count", 0)) + 1
            timestamp = _utc_timestamp()
            lifecycle = self._task.setdefault("lifecycle", [])
            if not isinstance(lifecycle, list):
                lifecycle = []
                self._task["lifecycle"] = lifecycle
            lifecycle.append(
                {"event": "resumed", "at": timestamp, "reason_code": reason_code}
            )
            self._task["status"] = "running"
            self._task["finished_at"] = None
            self._next_sequence = self._find_next_sequence(self._index.get("calls", []))
            self._index["updated_at"] = timestamp
            task_ok = self._write_json_safely(self.task_path, self._task)
            index_ok = self._write_json_safely(self.index_path, self._index)
            return task_ok and index_ok

    def update_index(self) -> bool:
        """原子刷新当前 index.json，供阶段汇总更新后使用。"""

        with self._lock:
            self._index["updated_at"] = _utc_timestamp()
            return self._write_json_safely(self.index_path, self._index)

    def update_stage_summary(
        self, relative_path: str | Path, summary: Mapping[str, Any]
    ) -> bool:
        """原子更新一个 summary.json，不复制任何计算 artifact。"""

        relative_text, target = self._validate_relative_path(relative_path)
        if Path(relative_text).name != "summary.json":
            raise ValueError("阶段汇总文件必须命名为 summary.json")
        if not isinstance(summary, Mapping):
            raise TypeError("阶段汇总必须是对象")
        return self._write_json_safely(target, _copy_json_value(summary))

    def update_summary(self, relative_path: str | Path, summary: Mapping[str, Any]) -> bool:
        """使用更短的名称更新阶段 summary.json。"""

        return self.update_stage_summary(relative_path, summary)

    def update_hypothesis_outcomes(
        self, relative_path: str | Path, outcomes: Mapping[str, Any]
    ) -> bool:
        """原子更新 Hypothesis outcomes.json，不生成额外模型调用。"""

        relative_text, target = self._validate_relative_path(relative_path)
        if Path(relative_text).name != "outcomes.json":
            raise ValueError("Hypothesis 结果文件必须命名为 outcomes.json")
        if not isinstance(outcomes, Mapping):
            raise TypeError("Hypothesis 结果必须是对象")
        return self._write_json_safely(target, _copy_json_value(outcomes))

    def update_outcomes(
        self, relative_path: str | Path, outcomes: Mapping[str, Any]
    ) -> bool:
        """使用更短的名称更新 Hypothesis outcomes.json。"""

        return self.update_hypothesis_outcomes(relative_path, outcomes)

    def read_model_output_view(
        self, relative_path: str | Path
    ) -> dict[str, Any] | None:
        """读取本 writer 已写入的 summary.json 或 outcomes.json。

        方法只读取调用方给出的受保护路径，不扫描目录，也不创建文件。目标不存在
        时返回 ``None`` 且不降级；目标存在但无法解析为 JSON 对象时记录 degraded。
        """

        relative_text, target = self._validate_relative_path(relative_path)
        if Path(relative_text).name not in {"summary.json", "outcomes.json"}:
            raise ValueError("只允许读取 summary.json 或 outcomes.json")
        with self._lock:
            if not target.exists():
                return None
            value = self._read_json_file(target)
            if not isinstance(value, dict):
                self._mark_degraded("展示汇总不是 JSON 对象", path=target)
                return None
            return value

    def update_stage_source_artifacts(
        self,
        stage: ModelStageHandle | str | Path,
        source_artifacts: Mapping[str, Any],
    ) -> bool:
        """在阶段仍运行时补充外部原始文件引用，不复制 artifact 内容。

        输入是阶段句柄和相对路径、哈希、媒体类型、大小等引用对象。已有引用与新
        引用按字段合并；阶段进入 validated 或 failed 后拒绝修改，保证终态 JSON
        不会被观察器事后改写。
        """

        if not isinstance(source_artifacts, Mapping):
            self._mark_degraded("source_artifacts 必须是对象")
            return False
        relative_text, target = self._coerce_stage_handle(stage)
        with self._lock:
            document = self._read_json_file(target)
            if not isinstance(document, dict) or document.get("status") != "running":
                return False
            identity = document.get("call_identity", {})
            sequence = identity.get("sequence") if isinstance(identity, Mapping) else None
            index_call = (
                self._find_index_call(sequence, relative_text)
                if isinstance(sequence, int)
                else None
            )
            if not self._stage_reference_matches(
                stage, relative_text, document, index_call
            ):
                self._mark_degraded("source_artifacts 阶段身份或 index 不匹配", path=target)
                return False
            try:
                validated_new = _validate_source_artifacts(
                    source_artifacts,
                )
            except (TypeError, ValueError) as error:
                self._mark_degraded(error, path=target)
                return False
            existing = document.get("source_artifacts")
            merged = dict(existing) if isinstance(existing, Mapping) else {}
            merged.update(validated_new)
            try:
                document["source_artifacts"] = _validate_source_artifacts(
                    merged,
                )
            except (TypeError, ValueError) as error:
                self._mark_degraded(error, path=target)
                return False
            return self._write_json_safely(target, document)

    def write_immutable_view(
        self, relative_path: str | Path, payload: Mapping[str, Any]
    ) -> bool:
        """只创建一次不可变展示文件，例如 Workflow 版本或排名结果。

        输入是运行目录内的相对 JSON 路径和可编码对象。目标已存在时不覆盖旧文件，
        返回 ``False`` 并将展示状态标为 ``degraded``；路径越界仍直接拒绝。
        """

        relative_text, target = self._validate_relative_path(relative_path)
        if not relative_text.endswith(".json"):
            raise ValueError("不可变展示文件必须使用 .json 扩展名")
        if not isinstance(payload, Mapping) or payload.get("schema_version") != _SCHEMA_VERSION:
            self._mark_degraded("不可变展示文件必须是 schema_version=1 的 object", path=target)
            return False
        with self._lock:
            if target.exists() or target.is_symlink():
                self._mark_degraded("不可变展示文件已经存在", path=target)
                return False
            return self._write_json_safely(target, _copy_json_value(payload))

    def resolve_output_path(self, relative_path: str | Path) -> Path:
        """解析 Bundle 内展示路径。"""

        return self._validate_relative_path(relative_path)[1]

    def result_summary(self) -> dict[str, Any]:
        """返回文档第 19 节规定的 CLI 模型输出状态摘要。"""

        with self._lock:
            calls = self._index.get("calls", [])
            if not isinstance(calls, list):
                calls = []
            summary: dict[str, Any] = {
                "status": "degraded" if self._degraded else "available",
                "directory": str(self.output_root),
                "call_count": len(calls),
                "failed_call_count": sum(
                    1
                    for call in calls
                    if isinstance(call, Mapping) and call.get("status") == "failed"
                ),
                "index_file": "internal/audit/index.json",
            }
            if self._degraded:
                summary["error"] = {
                    "failure_kind": "write_failed",
                    "message": self._error_message or "模型输出展示文件写入失败",
                }
            return summary

    @property
    def degraded(self) -> bool:
        """返回当前展示写入器是否已经发生过写入降级。"""

        return self._degraded


def _locate_model_run_output_resume_files(
    work_dir: str | Path,
    research_task_id: str,
    bundle_root: str | Path | None,
) -> tuple[str, Path, Path, Path]:
    """只读定位恢复 audit 文件并拒绝目录或符号链接异常。"""

    work_path = Path(work_dir).expanduser().resolve(strict=False)
    normalized_research_task_id = ModelRunOutputWriter._validate_research_task_id(
        research_task_id
    )
    expected_root = (
        work_path / "outputs" / "research_tasks" / normalized_research_task_id
    )
    if bundle_root is not None:
        requested_root = Path(bundle_root).expanduser().resolve(strict=False)
        if requested_root != expected_root.resolve(strict=False):
            raise ValueError(
                "bundle_root 必须等于 work_dir/outputs/research_tasks/<research_task_id>"
            )
    for parent in (
        work_path / "outputs",
        work_path / "outputs" / "research_tasks",
    ):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("恢复所需展示父目录不存在或不是普通目录")
    if expected_root.is_symlink() or not expected_root.is_dir():
        raise ValueError("恢复目录不存在或不是普通目录")
    audit_root = expected_root / "internal" / "audit"
    if audit_root.is_symlink() or not audit_root.is_dir():
        raise ValueError("恢复所需 audit 目录不存在或不是普通目录")
    task_path = audit_root / "task.json"
    index_path = audit_root / "index.json"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (task_path, index_path)
    ):
        raise ValueError("恢复所需 audit task/index 文件缺失或不是普通文件")
    return normalized_research_task_id, expected_root, task_path, index_path


def _read_model_run_output_resume_documents(
    task_path: Path,
    index_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """只读解析恢复 audit JSON，并拒绝非 object 或损坏内容。"""

    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("恢复所需 audit task/index JSON 损坏") from exc
    if not isinstance(task, dict) or not isinstance(index, dict):
        raise ValueError("恢复所需 audit task/index 必须是 object")
    return task, index


def validate_model_run_output_resume_files(
    work_dir: str | Path,
    research_task_id: str,
    *,
    bundle_root: str | Path | None = None,
    persistence_mode: str = "memory_off",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """只读严格核对恢复所需的模型 audit 文件。

    输入是工作目录、运行身份、可选 Bundle 根和展示持久化模式；动作只检查
    ``internal/audit/task.json`` 与 ``index.json`` 的普通文件属性、JSON 身份、
    生命周期、恢复计数、调用序列和相对路径，不创建目录、不更新计数或写入文件。
    返回值是已解析的 task/index 对象副本；文件缺失、损坏、身份不一致或结构不可
    恢复时抛出 ``ValueError``，供严格恢复入口在任何 Bundle 写入前拒绝降级 writer。
    """

    if persistence_mode not in {"memory_on", "memory_off"}:
        raise ValueError("persistence_mode 必须是 memory_on 或 memory_off")
    normalized_research_task_id, expected_root, task_path, index_path = (
        _locate_model_run_output_resume_files(
            work_dir, research_task_id, bundle_root
        )
    )
    task, index = _read_model_run_output_resume_documents(task_path, index_path)
    validator = ModelRunOutputWriter.__new__(ModelRunOutputWriter)
    validator.research_task_id = normalized_research_task_id
    validator.persistence_mode = persistence_mode
    validator.output_root = expected_root
    validation_error = validator._validate_resume_documents(task, index)
    if validation_error is not None:
        raise ValueError(validation_error)
    return dict(task), dict(index)
