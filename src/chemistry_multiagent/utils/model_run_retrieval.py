"""Retrieval 模型调用的可读展示记录器。

这个模块只负责把 Retrieval 已经产生的模型调用旁路写入统一的模型输出文件。
正式检索仍然使用自己的 receipt 和重试逻辑；记录器只把内部网络重试合并成一
次逻辑回答，并在写入失败时把展示能力标记为降级。模型请求、响应和解析结果
交给 :class:`ModelRunOutputWriter` 的统一白名单处理，避免两套脱敏规则漂移。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


class RetrievalOutputWriter(Protocol):
    """ModelRunOutputWriter 所需的最小写入接口。"""

    def start_stage(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def append_attempt(self, *args: Any, **kwargs: Any) -> bool:
        ...

    def finish_stage(self, *args: Any, **kwargs: Any) -> bool:
        ...

    def mark_model_output_degraded(self, *args: Any, **kwargs: Any) -> None:
        ...


_RETRIEVAL_STAGE_PATH = "01_retrieval/summarize_retrieval.json"
_RETRIEVAL_INPUT_FIELDS = (
    "scientific_question",
    "pdf_dir",
    "index_dir",
    "search_papers",
)
_SAFE_FAILURE_MESSAGES = {
    "transport": "模型服务请求失败",
    "provider_envelope": "模型响应格式不可识别",
    "empty_choices": "模型没有返回候选结果",
    "empty_content": "模型没有返回可用正文",
    "truncated": "模型响应被截断",
    "json_syntax": "模型响应不是合法 JSON",
    "schema": "模型响应不符合预期字段",
    "domain": "模型响应未通过内容校验",
}
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.S | re.I)


def copy_retrieval_request(value: Any) -> dict[str, Any]:
    """只保留 Retrieval 阶段初始输入的四个公开字段。

    阶段 writer 会再次按统一展示协议复制并脱敏这些值；这里仅限定初始输入的
    字段边界，不复制 provider request 的其他内容。
    """

    if not isinstance(value, Mapping):
        return {}
    return {
        field: value[field]
        for field in _RETRIEVAL_INPUT_FIELDS
        if field in value
    }


def _utc_timestamp() -> str:
    """返回带毫秒和 ``Z`` 后缀的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _safe_failure_message(failure_kind: Any) -> str:
    """把内部失败类别映射成固定短说明，不保存 provider 异常正文。"""

    return _SAFE_FAILURE_MESSAGES.get(
        str(failure_kind),
        "模型响应未通过展示协议",
    )


def _response_content(response: Any) -> str | None:
    """从规范化响应中读取第一候选的文字正文。"""

    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = first.get("text")
    if not isinstance(content, str) or not content.strip():
        return None
    return content.strip()


def _parse_display_output(response: Any, request: Mapping[str, Any]) -> Any:
    """按展示 request 的响应契约解析一份旁路结果。

    解析只用于给研究员阅读，绝不替代正式校验。JSON 语法正确但业务字段不合
    格时仍保留解析出的 JSON，便于定位问题；语法错误、空正文则保存 ``None``。
    """

    content = _response_content(response)
    if content is None:
        return None
    contract = request.get("response_contract")
    validation_kind = (
        contract.get("validation_kind")
        if isinstance(contract, Mapping)
        else "nonempty_text"
    )
    if validation_kind == "json_schema":
        match = _JSON_FENCE.match(content)
        if match:
            content = match.group(1).strip()
        try:
            return json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if validation_kind == "query_delimited_text":
        return [part.strip() for part in content.split("<>")]
    if validation_kind == "text_enum_relationship":
        return content.lower()
    if validation_kind == "text_enum_rank":
        return content.upper()
    return content


@dataclass
class _ActiveModelAttempt:
    """保存一次逻辑模型调用的展示请求和计时起点。"""

    request: dict[str, Any]
    started_at: str
    started_monotonic_ns: int
    record_enabled: bool


class RetrievalModelOutputRecorder:
    """把 Retrieval receipt 归并为一个统一展示阶段文件。

    记录器本身是 callable receipt sink，因此普通的 receipt sink 不需要任何适配。
    ``start_model_attempt`` 只是 Retrieval 在一次逻辑 provider 调用开始前提供的
    可选旁路钩子；模型返回后仍由正式代码把原 receipt 交给 ``__call__``。
    """

    def __init__(
        self,
        writer: RetrievalOutputWriter,
        retrieval_request: Mapping[str, Any],
        operation_id: str,
    ) -> None:
        if writer is None:
            raise ValueError("Retrieval 展示 recorder 必须提供 writer")
        if not isinstance(retrieval_request, Mapping):
            raise TypeError("retrieval_request 必须是 object")
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id 必须是非空字符串")
        self.writer = writer
        self.retrieval_request = copy_retrieval_request(retrieval_request)
        self.operation_id = operation_id
        self._stage_handle: Any = None
        self._active_attempt: _ActiveModelAttempt | None = None
        self._pending_transport_receipt: Mapping[str, Any] | None = None
        self._attempt_index = 0
        self._finished = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """返回展示写入是否已经降级。"""

        return self._degraded

    def _mark_degraded(self) -> None:
        """记录展示降级，并尽力通知 writer；通知失败也不能外抛。"""

        self._degraded = True
        notify = getattr(self.writer, "mark_model_output_degraded", None)
        if not callable(notify):
            return
        try:
            notify("Retrieval 模型展示写入失败")
        except Exception:
            return

    def _ensure_stage(self) -> bool:
        """在第一次真实模型调用前创建 Retrieval 展示阶段。"""

        if self._stage_handle is not None:
            return True
        if self._finished:
            return False
        try:
            handle = self.writer.start_stage(
                "retrieval",
                "summarize_retrieval",
                _RETRIEVAL_STAGE_PATH,
                operation_id=self.operation_id,
                input=self.retrieval_request,
            )
        except Exception:
            self._mark_degraded()
            return False
        if handle is None:
            self._mark_degraded()
            return False
        self._stage_handle = handle
        return True

    def _append_attempt(
        self,
        attempt: _ActiveModelAttempt,
        *,
        response: Mapping[str, Any] | None,
        validation_status: str,
        failure_kind: Any = None,
    ) -> None:
        """把一次逻辑回答交给统一 writer，并清理当前活动调用。"""

        self._active_attempt = None
        self._pending_transport_receipt = None
        if not attempt.record_enabled or self._stage_handle is None:
            return
        attempt_index = self._attempt_index
        self._attempt_index += 1
        if validation_status not in {"valid", "invalid", "transport_failed"}:
            validation_status = "invalid"
            failure_kind = "provider_envelope"
        finished_at = _utc_timestamp()
        elapsed_ns = max(0, time.monotonic_ns() - attempt.started_monotonic_ns)
        provider = None
        model = None
        trace_id = None
        if isinstance(response, Mapping):
            raw_provider = response.get("provider")
            if isinstance(raw_provider, str) and raw_provider.strip():
                provider = raw_provider
            raw_model = response.get("model")
            if isinstance(raw_model, str) and raw_model.strip():
                model = raw_model
            if model is None:
                request_model = attempt.request.get("model")
                if isinstance(request_model, str) and request_model.strip():
                    model = request_model
            raw_trace_id = response.get("id")
            if isinstance(raw_trace_id, str) and raw_trace_id.strip():
                trace_id = raw_trace_id
        validation = {
            "status": validation_status,
            "failure_kind": failure_kind if validation_status != "valid" else None,
            "message": (
                None
                if validation_status == "valid"
                else _safe_failure_message(failure_kind)
            ),
        }
        record = {
            "attempt_index": attempt_index,
            "started_at": attempt.started_at,
            "finished_at": finished_at,
            "duration_ms": round(elapsed_ns / 1_000_000, 3),
            "provider": provider,
            "model": model,
            "provider_trace_id": trace_id,
            "provider_request_id": None,
            "usage": (
                response.get("usage")
                if isinstance(response, Mapping)
                else None
            ),
            "request": attempt.request,
            "raw_response": response,
            "parsed_output": _parse_display_output(response, attempt.request),
            "validation": validation,
        }
        try:
            written = self.writer.append_attempt(self._stage_handle, record)
            if written is False:
                self._mark_degraded()
        except Exception:
            self._mark_degraded()

    def _flush_transport_failure(self) -> None:
        """将同一逻辑调用的多次网络失败合并为一条展示记录。"""

        pending = self._pending_transport_receipt
        attempt = self._active_attempt
        if pending is None or attempt is None:
            return
        self._append_attempt(
            attempt,
            response=None,
            validation_status="transport_failed",
            failure_kind="transport",
        )

    def start_model_attempt(self, request: Mapping[str, Any]) -> _ActiveModelAttempt | None:
        """在正式 provider 调用前开始一次逻辑调用的展示计时。

        如果上一逻辑调用只有内部网络失败，则先写入一条合并后的失败记录；下一次
        逻辑调用不会重复展示每个传输重试。
        """

        try:
            self._flush_transport_failure()
            if self._active_attempt is not None:
                self._mark_degraded()
                self._active_attempt = None
                self._pending_transport_receipt = None
            enabled = self._ensure_stage()
            safe_request = dict(request) if isinstance(request, Mapping) else {}
            attempt = _ActiveModelAttempt(
                request=safe_request,
                started_at=_utc_timestamp(),
                started_monotonic_ns=time.monotonic_ns(),
                record_enabled=enabled,
            )
            self._active_attempt = attempt
            return attempt
        except Exception:
            self._mark_degraded()
            return None

    def __call__(self, receipt: Mapping[str, Any]) -> None:
        """接收正式 Retrieval receipt；观察器异常绝不影响检索。"""

        try:
            if not isinstance(receipt, Mapping):
                self._mark_degraded()
                return
            status = receipt.get("validation_status")
            if status == "transport_failed":
                if self._active_attempt is None:
                    self._mark_degraded()
                    return
                self._pending_transport_receipt = receipt
                return
            if status in {"valid", "invalid"}:
                attempt = self._active_attempt
                if attempt is None:
                    self._mark_degraded()
                    return
                response = receipt.get("response")
                # 旧 Controller 的异常保护会发一条没有模型响应的固定标记；它
                # 不是正式内容校验失败，统一归并为 transport_failed，避免把
                # Controller 保护事件展示成伪造的模型 invalid 响应。
                if (
                    status == "invalid"
                    and receipt.get("failure_kind") == "controller_exception"
                    and response is None
                ):
                    self._append_attempt(
                        attempt,
                        response=None,
                        validation_status="transport_failed",
                        failure_kind="transport",
                    )
                    return
                self._append_attempt(
                    attempt,
                    response=response if isinstance(response, Mapping) else None,
                    validation_status=str(status),
                    failure_kind=receipt.get("failure_kind"),
                )
                return
            self._mark_degraded()
        except Exception:
            self._mark_degraded()

    def finish_retrieval(self, result: Mapping[str, Any]) -> None:
        """在 Controller 收到 Retrieval 结果后关闭展示阶段。"""

        if self._finished:
            return
        try:
            failed = isinstance(result, Mapping) and bool(result.get("error"))
            self._flush_transport_failure()
            if self._active_attempt is not None:
                if failed:
                    # provider 直接抛异常时可能没有机会发送 transport receipt。
                    # 仍然要为已经开始的逻辑调用写一条安全失败 attempt，保证
                    # writer 可以把阶段从 running 正常关闭为 failed。
                    self._append_attempt(
                        self._active_attempt,
                        response=None,
                        validation_status="transport_failed",
                        failure_kind="transport",
                    )
                else:
                    self._mark_degraded()
                    self._active_attempt = None
                    self._pending_transport_receipt = None
            if self._stage_handle is None:
                return
            written = self.writer.finish_stage(
                self._stage_handle,
                "failed" if failed else "validated",
                final_output=None if failed else dict(result),
                finished_at=_utc_timestamp(),
            )
            if written is False:
                self._mark_degraded()
        except Exception:
            self._mark_degraded()
        finally:
            self._finished = True


__all__ = [
    "RetrievalModelOutputRecorder",
    "RetrievalOutputWriter",
    "copy_retrieval_request",
]
