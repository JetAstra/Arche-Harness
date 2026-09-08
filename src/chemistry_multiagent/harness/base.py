"""Provider-neutral request/response contracts for selectable harnesses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class HarnessError(RuntimeError):
    """Harness 无法生成符合 provider 合同的响应。"""

    def __init__(self, message: str, *, kind: str = "transport") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class HarnessRequest:
    """保存已经编译并清理过的模型请求。"""

    messages: tuple[Mapping[str, Any], ...]
    model: str
    role: str = "generic"
    stage: str = "model"
    workspace: Path = Path("local_data/runtime/harness_workspaces")
    research_task_id: str = ""
    invocation_id: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # ``None`` 明确表示不设置本次 SDK turn 的总体 wall-clock deadline。
    # 传输、MCP 工具和计算作业的局部时限仍由各自的实现负责。
    timeout_seconds: float | None = 900.0
    response_schema: Mapping[str, Any] | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # ``profile`` 只供 Harness 选择能力策略；缺省值由具体 Harness 解释为
    # 兼容旧 Planner 的 Review/deny_all 策略，不改变既有编译请求合同。
    profile: str | None = None

    @classmethod
    def from_compiled(
        cls,
        request: Mapping[str, Any],
        *,
        role: str = "generic",
        stage: str = "model",
        workspace: Path,
        research_task_id: str = "",
        invocation_id: str = "",
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] = (),
        timeout_seconds: float | None = 900.0,
        profile: str | None = None,
    ) -> "HarnessRequest":
        if "run_id" in request:
            raise HarnessError(
                "compiled request must use research_task_id; run_id is legacy-only",
                kind="provider_envelope",
            )
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HarnessError("compiled request messages must be a non-empty list", kind="provider_envelope")
        if any(not isinstance(item, Mapping) for item in messages):
            raise HarnessError(
                "compiled request messages must contain only objects",
                kind="provider_envelope",
            )
        model = request.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HarnessError("compiled request model must be non-empty", kind="provider_envelope")
        return cls(
            messages=tuple(dict(item) for item in messages),
            model=model.strip(),
            role=role,
            stage=stage,
            workspace=Path(workspace),
            research_task_id=research_task_id,
            invocation_id=invocation_id,
            max_tokens=request.get("max_tokens"),
            temperature=request.get("temperature"),
            top_p=request.get("top_p"),
            timeout_seconds=timeout_seconds,
            response_schema=response_schema,
            tools=tuple(dict(item) for item in tools),
            profile=(
                profile
                if profile is not None
                else request.get("profile")
                if isinstance(request.get("profile"), str)
                else None
            ),
        )

    def prompt_text(self) -> str:
        """序列化消息，并保留原有角色提示和响应约束。"""

        if self.role == "investigation":
            prefix = "Follow the supplied investigation instructions exactly.\n\n"
        else:
            prefix = (
                "Return the answer for the supplied role. Follow its instructions exactly. "
                "Return exactly one raw JSON value without Markdown fences.\n\n"
            )
        prompt = prefix + json.dumps(
            list(self.messages),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        # Investigation 的 schema 只约束一个 turn 完成全部科学工作后的最终 JSON；
        # 其它作业过程由模型可见的 Investigation instructions 和工具协议处理，
        # 这里不重复描述状态机。Review 等普通角色仍携带 provider schema。
        if self.response_schema is not None and self.role == "investigation":
            prompt += (
                "\n\n最终结果必须符合以下 JSON schema；只在完成当前调查并核对证据后返回单个"
                "原始 JSON object，不输出 Markdown 或额外字段：\n"
                + json.dumps(
                    dict(self.response_schema),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        elif self.response_schema is not None:
            prompt += "\n\nRequired response JSON schema:\n" + json.dumps(
                dict(self.response_schema),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        return prompt


@dataclass(frozen=True)
class HarnessResponse:
    """保存各类 Harness 实现返回的规范化响应。"""

    content: str = ""
    finish_reason: str | None = "stop"
    model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    raw_trace_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_provider_envelope(self) -> dict[str, Any]:
        """返回供现有校验器读取的 OpenAI 兼容响应信封。"""

        return {
            "id": str(self.metadata.get("response_id") or "harness-response"),
            "object": "chat.completion",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": self.finish_reason,
                    "message": {"role": "assistant", "content": self.content},
                }
            ],
            "usage": dict(self.usage),
            "harness": {
                "name": self.metadata.get("harness"),
                "role": self.metadata.get("role"),
                "stage": self.metadata.get("stage"),
                "trace_path": self.raw_trace_path,
            },
        }


class HarnessRunner:
    """模型驱动器必须实现的最小接口。"""

    name = "base"

    def run(self, request: HarnessRequest) -> HarnessResponse:
        raise NotImplementedError
