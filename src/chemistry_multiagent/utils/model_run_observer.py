"""把模型操作旁路事件写成研究员可读的阶段文件。

正式模型操作仍由 :mod:`model_operation` 负责请求、校验、恢复和原始清单。本模块
只把旁路事件转换成当前 Run Bundle 下的展示和审计文件；展示写入失败会被标记为
降级，不会向科学工作流抛出异常，也不会改变模型返回值。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from typing import Any, Mapping

from chemistry_multiagent.utils.model_run_outputs import ModelRunOutputWriter


@dataclass
class _ObservedStage:
    """保存一次展示阶段的 writer 句柄和路径。"""

    writer_handle: Any
    raw_stage: str
    operation: str
    display_context: dict[str, Any]


class ModelRunStageObserver:
    """将 ``ModelStageObserver`` 事件映射到第25号设计的目录结构。

    构造参数是展示写入器和本次业务操作的明确上下文。上下文必须由调用方传入
    ``batch_index``、``combination_index`` 或业务 round；本类不会从 operation ID
    猜测这些字段。关系、合并和排名文件按真实 stage 启动顺序编号，阶段输入和终态
    由观察器直接收集，避免把展示身份重新写入模型请求。
    """

    def __init__(
        self,
        writer: ModelRunOutputWriter | None,
        *,
        display_context: Mapping[str, Any] | None = None,
    ) -> None:
        """绑定展示写入器和固定业务上下文；没有 writer 时观察器为空操作。"""

        if writer is not None and not isinstance(writer, ModelRunOutputWriter):
            raise TypeError("writer 必须是 ModelRunOutputWriter 或 None")
        if display_context is not None and not isinstance(display_context, Mapping):
            raise TypeError("display_context 必须是 object 或 None")
        self.writer = writer
        self.display_context = deepcopy(dict(display_context or {}))
        self._lock = RLock()
        self._comparison_counters: dict[tuple[str, str], int] = {}
        self._stage_paths: list[str] = []
        self._hypothesis_decisions: list[dict[str, Any]] = []

    def get_stage_paths(self) -> list[str]:
        """返回已经真实启动过的展示文件路径，保持 start 顺序。"""

        with self._lock:
            return list(self._stage_paths)

    def get_hypothesis_merge_decisions(self) -> list[dict[str, Any]]:
        """返回实际完成合并的模型决定，不包含关系判断记录。"""

        with self._lock:
            return deepcopy(
                [
                    decision
                    for decision in self._hypothesis_decisions
                    if decision.get("stage") in {"merge", "complement", "conflict"}
                ]
            )

    # ==================== ModelStageObserver 协议 ====================

    def start_model_stage(self, event: Mapping[str, Any]) -> _ObservedStage | None:
        """为真实模型 stage 创建展示文件并返回观察句柄。"""

        if self.writer is None or not isinstance(event, Mapping):
            return None
        try:
            descriptor = self._describe_stage(event)
            operation_id = self._text(event.get("operation_id"))
            raw_stage = self._text(event.get("raw_stage") or event.get("stage")) or "unknown"
            context = descriptor["display_context"]
            path = descriptor["relative_path"]
            handle = self.writer.start_stage(
                descriptor["phase"],
                descriptor["stage"],
                path,
                operation_id=operation_id,
                batch_index=context.get("batch_index"),
                query_index=context.get("query_index"),
                combination_index=context.get("combination_index"),
                round_index=context.get("round_index"),
                input=deepcopy(event.get("stage_input")),
                source_artifacts={},
                allow_terminal_operation_retry=True,
            )
            observed = _ObservedStage(
                writer_handle=handle,
                raw_stage=raw_stage,
                operation=str(event.get("operation") or ""),
                display_context=context,
            )
            with self._lock:
                if handle is not None:
                    model_call_prefix = "internal/audit/model_calls/"
                    actual_path = handle.relative_path
                    self._stage_paths.append(
                        actual_path[len(model_call_prefix) :]
                        if actual_path.startswith(model_call_prefix)
                        else actual_path
                    )
            return observed
        except Exception:  # 展示旁路不能阻塞科学流程
            self.mark_model_output_degraded("model_stage_display_start_failed")
            return None

    def record_model_attempt(
        self,
        handle: _ObservedStage | None,
        event: Mapping[str, Any],
    ) -> None:
        """把一次模型回答复制到同一阶段的 attempts 数组。"""

        if self.writer is None or handle is None or handle.writer_handle is None:
            return
        try:
            raw_response = event.get("raw_response")
            request = event.get("compiled_request")
            request = deepcopy(request) if isinstance(request, Mapping) else {}
            response_contract = event.get("response_contract")
            if isinstance(response_contract, Mapping):
                request["response_contract"] = deepcopy(dict(response_contract))
            attempt = {
                "attempt_index": event.get("attempt_index"),
                "started_at": event.get("started_at"),
                "finished_at": event.get("finished_at"),
                "duration_ms": event.get("duration_ms"),
                "provider": self._provider_name(event, raw_response),
                "model": request.get("model") if isinstance(request, Mapping) else None,
                "provider_trace_id": event.get("provider_trace_id"),
                "provider_request_id": event.get("provider_request_id"),
                "usage": self._normalise_usage(raw_response),
                "request": request,
                "raw_response": deepcopy(raw_response),
                "parsed_output": deepcopy(event.get("parsed_output")),
                "validation": self._normalise_validation(event.get("validation")),
            }
            self.writer.append_attempt(handle.writer_handle, attempt)
        except Exception:
            self.mark_model_output_degraded("model_stage_display_attempt_failed")

    def finish_model_stage(
        self,
        handle: _ObservedStage | None,
        event: Mapping[str, Any],
    ) -> None:
        """写入最终输出和原始清单引用，并关闭展示阶段文件。"""

        if handle is None:
            return
        self._record_hypothesis_decision(handle, event)
        if self.writer is None or handle.writer_handle is None:
            return
        try:
            source_artifacts = event.get("source_artifacts")
            if isinstance(source_artifacts, Mapping):
                source_artifacts = {
                    name: deepcopy(source_artifacts[name])
                    for name in (
                        "base_input",
                        "manifest",
                        "request_files",
                        "response_files",
                    )
                    if name in source_artifacts
                }
                self.writer.update_stage_source_artifacts(
                    handle.writer_handle,
                    source_artifacts,
                )
            status = event.get("status")
            if status not in {"validated", "failed"}:
                status = "failed"
            final_output = self._project_final_output(handle, event.get("final_output"))
            self.writer.finish_stage(
                handle.writer_handle,
                status,
                final_output=final_output,
                finished_at=event.get("finished_at"),
            )
        except Exception:
            self.mark_model_output_degraded("model_stage_display_finish_failed")

    def mark_model_output_degraded(self, reason: str | None = None) -> None:
        """把旁路写入失败标记为 degraded，不向正式模型操作抛错。"""

        writer = self.writer
        if writer is None:
            return
        marker = getattr(writer, "mark_model_output_degraded", None)
        if callable(marker):
            try:
                marker(str(reason or "模型阶段展示写入失败"))
            except Exception:
                return

    # ==================== 路径和字段映射 ====================

    def _describe_stage(self, event: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._text(event.get("operation")) or ""
        raw_stage = self._text(event.get("raw_stage") or event.get("stage")) or ""
        display_context = self._merge_display_context(event)
        batch_index = self._required_index(display_context, "batch_index")
        if operation == "hypothesis_generation":
            if raw_stage == "query":
                return {
                    "phase": "query",
                    "stage": "generate_queries",
                    "relative_path": f"02_query/batch_{batch_index:03d}/generate_queries.json",
                    "display_context": display_context,
                }
            if raw_stage in {"hypothesis", "regeneration"}:
                query_index = self._event_or_context_index(event, display_context, "query_index")
                display_context["query_index"] = query_index
                return {
                    "phase": "hypothesis",
                    "stage": "generate_hypotheses",
                    "relative_path": f"03_hypothesis/batch_{batch_index:03d}/generation/query_{query_index:03d}.json",
                    "display_context": display_context,
                }
            if raw_stage in {"relationship", "rank"}:
                kind = "relationship" if raw_stage == "relationship" else "ranking"
                comparison_index = self._next_comparison_index(operation, kind)
                folder = "relationships" if raw_stage == "relationship" else "ranking"
                display_context["comparison_index"] = comparison_index
                return {
                    "phase": "hypothesis",
                    "stage": "compare_hypotheses" if raw_stage == "relationship" else "rank_hypotheses",
                    "relative_path": f"03_hypothesis/batch_{batch_index:03d}/{folder}/comparison_{comparison_index:03d}.json",
                    "display_context": display_context,
                }
            if raw_stage in {"merge", "complement", "conflict"}:
                merge_index = self._next_comparison_index(operation, "synthesis")
                display_context["merge_index"] = merge_index
                return {
                    "phase": "hypothesis",
                    "stage": f"synthesize_{raw_stage}_hypothesis",
                    "relative_path": f"03_hypothesis/batch_{batch_index:03d}/synthesis/{raw_stage}_{merge_index:03d}.json",
                    "display_context": display_context,
                }
            raise ValueError(f"未注册 hypothesis stage：{raw_stage}")
        if operation == "workflow_generation":
            return {
                "phase": "planning",
                "stage": "plan_workflow",
                "relative_path": f"04_planning/batch_{batch_index:03d}/combination_{self._required_index(display_context, 'combination_index'):03d}/plan_workflow.json",
                "display_context": display_context,
            }
        if operation == "workflow_optimization":
            return {
                "phase": "planning",
                "stage": "optimize_workflow",
                "relative_path": f"04_planning/batch_{batch_index:03d}/combination_{self._required_index(display_context, 'combination_index'):03d}/optimize_workflow.json",
                "display_context": display_context,
            }
        if operation == "router":
            return {
                "phase": "router",
                "stage": "evaluate_scientific_evidence",
                "relative_path": f"05_router/batch_{batch_index:03d}/combination_{self._required_index(display_context, 'combination_index'):03d}/round_{self._required_round(display_context):03d}/evaluate_scientific_evidence.json",
                "display_context": display_context,
            }
        if operation == "workflow_revision":
            return {
                "phase": "revision",
                "stage": "revise_workflow",
                "relative_path": f"06_revision/batch_{batch_index:03d}/combination_{self._required_index(display_context, 'combination_index'):03d}/round_{self._required_round(display_context):03d}/revise_workflow.json",
                "display_context": display_context,
            }
        raise ValueError(f"未注册模型 operation：{operation}")

    def _merge_display_context(self, event: Mapping[str, Any]) -> dict[str, Any]:
        context = dict(self.display_context)
        event_context = event.get("display_context")
        if isinstance(event_context, Mapping):
            context.update(deepcopy(dict(event_context)))
        context.setdefault("query_index", event.get("query_index"))
        return context

    def _next_comparison_index(
        self,
        operation: str,
        category: str,
    ) -> int:
        with self._lock:
            key = (operation, category)
            value = self._comparison_counters.get(key, 0) + 1
            self._comparison_counters[key] = value
            return value

    def _event_or_context_index(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
        name: str,
    ) -> int:
        value = event.get(name)
        if value is None:
            value = context.get(name)
        return self._required_index({name: value}, name)

    @staticmethod
    def _required_index(context: Mapping[str, Any], field: str) -> int:
        value = context.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"展示上下文缺少正整数 {field}")
        return value

    @classmethod
    def _required_round(cls, context: Mapping[str, Any]) -> int:
        value = context.get("round_index")
        if value is None:
            value = context.get("router_round", context.get("revision_round"))
        return cls._required_index({"round_index": value}, "round_index")

    @staticmethod
    def _text(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _provider_name(event: Mapping[str, Any], raw_response: Any) -> str:
        explicit = event.get("provider")
        if isinstance(explicit, str) and explicit:
            return explicit
        if isinstance(raw_response, Mapping):
            provider = raw_response.get("provider")
            if isinstance(provider, str) and provider:
                return provider
        return "openai_compatible"

    @staticmethod
    def _normalise_validation(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        status = value.get("status")
        if status not in {"valid", "invalid", "transport_failed"}:
            status = "valid" if value.get("valid") is True else "invalid"
        return {
            "status": status,
            "failure_kind": value.get("failure_kind"),
            "message": value.get("message", value.get("summary")),
        }

    @staticmethod
    def _normalise_usage(raw_response: Any) -> dict[str, Any]:
        """统一保存输入、输出和总 token；缺失值保持 ``None``。"""

        usage = raw_response.get("usage") if isinstance(raw_response, Mapping) else None
        if not isinstance(usage, Mapping):
            usage = {}
        input_tokens = usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("completion_tokens")
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens"),
        }

    @staticmethod
    def _project_final_output(handle: _ObservedStage, value: Any) -> Any:
        if handle.operation != "workflow_revision":
            return deepcopy(value)
        if not isinstance(value, Mapping):
            return deepcopy(value)
        if value.get("result_type") == "tool_capability_gap":
            return deepcopy(dict(value))
        if "normalized_patch" in value and "after_workflow" in value:
            return {"normalized_patch": deepcopy(value["normalized_patch"])}
        return deepcopy(dict(value))

    def _record_hypothesis_decision(
        self,
        handle: _ObservedStage,
        event: Mapping[str, Any],
    ) -> None:
        """从模型阶段原始输入和终态收集关系或合并决定。"""

        if handle.raw_stage not in {"relationship", "merge", "complement", "conflict"}:
            return
        writer_handle = handle.writer_handle
        source_file = (
            writer_handle.relative_path
            if writer_handle is not None
            and isinstance(getattr(writer_handle, "relative_path", None), str)
            else None
        )
        record = {
            "stage": handle.raw_stage,
            "status": event.get("status"),
            "input": deepcopy(event.get("stage_input")),
            "result": deepcopy(event.get("final_output")),
            "source_file": source_file,
        }
        with self._lock:
            self._hypothesis_decisions.append(record)


__all__ = ["ModelRunStageObserver"]
