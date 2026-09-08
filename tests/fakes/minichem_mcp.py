"""用于 MiniChem MCP adapter 合同测试的确定性会话。

该假会话只模拟 MCP 的四个会话方法（initialize、list_tools、call_tool、
read_resource），不启动真实服务、外部软件或 Gaussian。测试可以为每个工具
注入固定响应、响应序列或异常，并通过 ``calls``、``submit_count`` 等字段
核对请求身份、轮询和副作用次数。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


def build_catalog_snapshot(
    *,
    actions: list[Mapping[str, Any]] | None = None,
    backends: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造符合 MiniChem Catalog v7 wire schema 的最小目录快照。"""

    payload: dict[str, Any] = {
        "schema_version": 7,
        "discovery_mode": "progressive",
        "execution_layers": [
            {
                "id": "predefined_actions",
                "role": "validated common scientific operations",
                "mandatory": False,
            },
            {
                "id": "native_software",
                "role": "Agent-authored native inputs and exact allowlisted commands",
                "mandatory": False,
            },
            {
                "id": "programmable_analysis",
                "role": "Agent-authored programs in explicitly selected chemistry runtimes",
                "mandatory": False,
            },
        ],
        "exposure_policy": "progressive_discovery",
        "catalog_visibility_policy": "focused_task_independent",
        "task_specific_tool_filtering": False,
        "backend_selection_policy": "per_action_explicit",
        "provider_selection_policies": ["agent_backend_required"],
        "scientific_resource_selection_policy": "agent_explicit_no_default",
        "automatic_fallback": False,
        "evaluation_resource_budget": {
            "agent_controllable": False,
            "cpu_cores": 60,
            "gpu_count": 1,
            "memory_mb": 320000,
            "scope": "per_task",
            "source": "evaluation_policy",
        },
        "actions": deepcopy(actions)
        if actions is not None
        else [
            {
                "id": "convert_structure",
                "version": "1.0.0",
                "category": "structure_and_system",
                "description": "Convert an explicit molecular structure.",
                "backend_ids": ["openbabel"],
                "required_inputs": ["structure"],
                "optional_inputs": [],
                "selection_policy": "agent_backend_required",
                "primary_output": "AtomicStructure",
                "data_action": False,
                "batch_safe": False,
            }
        ],
        "backends": deepcopy(backends)
        if backends is not None
        else [
            {
                "id": "openbabel",
                "display_name": "Open Babel",
                "runtime": "native",
                "capabilities": ["convert_structure"],
                "health": {"status": "healthy"},
            }
        ],
        "resources": [],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["catalog_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def build_artifact_ref(
    *,
    content: bytes = b"deterministic artifact\n",
    artifact_id: str = "art_test_0001",
    semantic_type: str = "AtomicStructure",
    media_type: str = "chemical/x-xyz",
    path: str = "artifacts/art_test_0001.xyz",
    producer_action: str = "convert_structure",
    producer_backend: str | None = "openbabel",
    parent_artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """返回带真实 SHA-256 的确定性 ArtifactRef 字典。"""

    return {
        "artifact_id": artifact_id,
        "semantic_type": semantic_type,
        "media_type": media_type,
        "sha256": hashlib.sha256(content).hexdigest(),
        "path": path,
        "producer_action": producer_action,
        "producer_backend": producer_backend,
        "parent_artifact_ids": list(parent_artifact_ids or []),
    }


_JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{32}$")


def _select_fake_json_pointer_value(text: str, pointer: str) -> Any:
    """按公开 JSON Pointer 合同从假工作区中选择一个值。"""

    value = json.loads(text)
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must be empty or start with '/'")
    current: Any = value
    for raw_token in pointer[1:].split("/"):
        token_chars: list[str] = []
        index = 0
        while index < len(raw_token):
            if raw_token[index] != "~":
                token_chars.append(raw_token[index])
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
                raise ValueError(f"invalid JSON Pointer escape: {raw_token!r}")
            token_chars.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        token = "".join(token_chars)
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"missing JSON Pointer key: {token!r}")
            current = current[token]
        elif isinstance(current, list):
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token):
                raise ValueError(f"invalid JSON Pointer index: {token!r}")
            item_index = int(token)
            if item_index >= len(current):
                raise ValueError(f"out-of-range JSON Pointer index: {token!r}")
            current = current[item_index]
        else:
            raise ValueError("JSON Pointer traverses through a scalar")
    return current


def _build_job_id(index: int) -> str:
    """生成符合公开协议的确定性作业身份。"""

    value = f"job_{index:032x}"
    assert _JOB_ID_PATTERN.fullmatch(value) is not None
    return value


@dataclass(frozen=True)
class FakeToolCall:
    """一条可审计的 MCP 工具调用记录。"""

    name: str
    arguments: Mapping[str, Any]


class FakeMcpSession:
    """提供同步且可注入的 MCP ClientSession 替身。

    ``session_factory`` 直接返回该对象即可交给 ``MiniChemMcpAdapter``。工具
    响应可以是字典、响应序列、可调用对象或异常实例；异常会原样抛给 adapter，
    便于测试远端断连和协议失败的诊断信息。
    """

    def __init__(
        self,
        *,
        catalog: Mapping[str, Any] | None = None,
        tools: list[Mapping[str, Any]] | None = None,
        tool_results: Mapping[str, Any] | None = None,
        tool_errors: Mapping[str, BaseException | list[BaseException]] | None = None,
        action_results: Mapping[str, Any] | None = None,
        job_states: Mapping[str, list[Mapping[str, Any]]] | None = None,
        collect_results: Mapping[str, Mapping[str, Any]] | None = None,
        cancel_results: Mapping[str, Mapping[str, Any]] | None = None,
        artifact_results: Mapping[str, Any] | None = None,
        workspace_files: Mapping[str, str] | None = None,
        validation_results: Mapping[str, Any] | None = None,
    ) -> None:
        self.catalog = deepcopy(dict(catalog or build_catalog_snapshot()))
        self.tools = deepcopy(
            tools
            if tools is not None
            else [
                {"name": "list_action_domains"},
                {"name": "browse_action_category"},
                {"name": "search_actions"},
                {"name": "inspect_action"},
                {"name": "inspect_backend"},
                {"name": "search_resources"},
                {"name": "inspect_resource"},
                {"name": "execute_action"},
                {"name": "submit_action_batch"},
                {"name": "list_software"},
                {"name": "inspect_software"},
                {"name": "read_software_documentation"},
                {"name": "search_software_documentation"},
                {"name": "write_workspace_text"},
                {"name": "read_workspace_text"},
                {"name": "validate_native_job"},
                {"name": "submit_native_job"},
                {"name": "list_analysis_runtimes"},
                {"name": "inspect_analysis_inputs"},
                {"name": "validate_analysis_program"},
                {"name": "submit_analysis_program"},
                {"name": "get_execution_resources"},
                {"name": "get_execution_job"},
                {"name": "collect_execution_job"},
                {"name": "cancel_execution_job"},
                {"name": "declare_scientific_artifact"},
            ]
        )
        self._tool_results = self._to_queues(tool_results or {})
        self._tool_errors = self._to_queues(tool_errors or {})
        self._action_results = self._to_queues(action_results or {})
        self._job_states = {
            job_id: deque(deepcopy(values))
            for job_id, values in (job_states or {}).items()
        }
        self._collect_results = deepcopy(dict(collect_results or {}))
        self._cancel_results = deepcopy(dict(cancel_results or {}))
        self._artifact_results = self._to_queues(artifact_results or {})
        self._workspace_files = dict(workspace_files or {})
        self._validation_results = deepcopy(dict(validation_results or {}))
        self._last_job_values: dict[str, dict[str, Any]] = {}
        self._job_types: dict[str, str] = {}
        self.calls: list[FakeToolCall] = []
        self.initialize_count = 0
        self.close_count = 0
        self.submit_count = 0
        self.native_submit_count = 0
        self.analysis_submit_count = 0
        self.list_software_count = 0
        self.inspect_software_count = 0
        self.list_analysis_runtimes_count = 0
        self.read_workspace_text_count = 0
        self.write_workspace_text_count = 0
        self.validate_native_job_count = 0
        self.validate_analysis_program_count = 0
        # 提供简短别名，便于调用方按动作名称直接核对副作用次数。
        self.read_workspace_count = 0
        self.write_workspace_count = 0
        self.native_validation_count = 0
        self.analysis_validation_count = 0
        self.get_count: defaultdict[str, int] = defaultdict(int)
        self.collect_count: defaultdict[str, int] = defaultdict(int)
        self.cancel_count: defaultdict[str, int] = defaultdict(int)
        self._next_job_number = 1

    @staticmethod
    def _to_queues(values: Mapping[str, Any]) -> dict[str, deque[Any]]:
        queues: dict[str, deque[Any]] = {}
        for key, value in values.items():
            if isinstance(value, (list, tuple, deque)):
                queues[key] = deque(deepcopy(list(value)))
            else:
                queues[key] = deque([deepcopy(value)])
        return queues

    @staticmethod
    def _take(queue: deque[Any], *, repeat_last: bool = True) -> Any:
        if not queue:
            raise AssertionError("假 MCP 响应序列意外耗尽")
        value = queue.popleft()
        if repeat_last and not queue:
            queue.append(deepcopy(value))
        return deepcopy(value)

    def initialize(self) -> dict[str, Any]:
        """记录 initialize；不会访问网络或改变目录。"""

        self.initialize_count += 1
        return {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake-minichem"}}

    def list_tools(self) -> dict[str, Any]:
        """返回固定工具索引，验证 adapter 不把旧 Tool Card 当目录。"""

        return {"tools": deepcopy(self.tools)}

    def read_resource(self, uri: str) -> dict[str, Any]:
        """按 MCP ReadResourceResult.contents 返回唯一目录资源。"""

        if uri != "minichem://catalog":
            raise KeyError(f"unknown resource: {uri}")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": json.dumps(
                        self.catalog,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            ]
        }

    def close(self) -> None:
        """记录关闭操作。"""

        self.close_count += 1

    def _raise_injected_error(self, name: str) -> None:
        queue = self._tool_errors.get(name)
        if not queue:
            return
        error = self._take(queue, repeat_last=False)
        if isinstance(error, BaseException):
            raise error

    @staticmethod
    def _unwrap_tool_request(
        name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """按真实 FastMCP wire 取出单一 ``request`` 包装内的参数。

        MiniChem 的公开工具都把 Pydantic 请求模型编码为
        ``{"request": {...}}``。假会话拒绝旧的扁平参数，避免测试在真实
        MCP 服务上无法复现的调用仍然通过；返回值只供 fake 的确定性业务分支使用。
        """

        if not isinstance(arguments, Mapping) or set(arguments) != {"request"}:
            raise AssertionError(
                f"{name} 必须使用真实 MiniChem wire 的 request 包装：{arguments!r}"
            )
        request = arguments.get("request")
        if not isinstance(request, Mapping):
            raise AssertionError(f"{name}.request 必须是 object")
        return deepcopy(dict(request))

    def _take_tool_result(self, name: str, arguments: Mapping[str, Any]) -> Any:
        queue = self._tool_results.get(name)
        if queue:
            value = self._take(queue)
            return value(arguments) if callable(value) else value
        return None

    def _default_action_result(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        action_id = str(arguments.get("action_id", "convert_structure"))
        backend_id = arguments.get("backend_id", "openbabel")
        artifact = build_artifact_ref(
            artifact_id="art_action_0001",
            producer_action=action_id,
            producer_backend=str(backend_id) if backend_id else None,
        )
        return {
            "status": "success",
            "action": action_id,
            "action_version": "1.0.0",
            "requested_backend": backend_id,
            "backend": backend_id,
            "selection_source": "agent",
            "result": {"converted": True},
            "output_artifacts": [artifact],
            "provenance": {
                "catalog_hash": self.catalog.get("catalog_hash"),
                "request_id": "req_fake_0001",
            },
        }

    def _call_action(self, arguments: Mapping[str, Any]) -> Any:
        action_id = str(arguments.get("action_id", "convert_structure"))
        queue = self._action_results.get(action_id)
        if queue:
            value = self._take(queue)
            return value(arguments) if callable(value) else value
        return self._default_action_result(arguments)

    def _submit_native_job(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.native_submit_count += 1
        operation_id = arguments.get("operation_id")
        job_id = str(arguments.get("job_id") or _build_job_id(self._next_job_number))
        self._next_job_number += 1
        self._job_types[job_id] = str(arguments.get("job_type", "native_software"))
        self._job_states.setdefault(
            job_id,
            deque(
                [
                    {
                        "status": "success",
                        "job": {"job_id": job_id, "status": "queued"},
                        "terminal": False,
                        "request_status": "accepted",
                        "process_status": "queued",
                    },
                    {
                        "status": "success",
                        "job": {"job_id": job_id, "status": "success"},
                        "terminal": True,
                        "request_status": "accepted",
                        "process_status": "completed",
                    },
                ]
            ),
        )
        return {
            "status": "success",
            "job_id": job_id,
            "job_status": "queued",
            "job_type": str(arguments.get("job_type", "native_software")),
            "operation_id": operation_id,
            "automatic_fallback": False,
        }

    def _submit_analysis_job(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """按 programmable_analysis 工具提交，并单独记录其副作用次数。"""

        self.analysis_submit_count += 1
        result = self._submit_native_job(arguments)
        self.native_submit_count -= 1
        result["job_type"] = "programmable_analysis"
        self._job_types[str(result["job_id"])] = "programmable_analysis"
        return result

    def _get_job(self, job_id: str) -> dict[str, Any]:
        self.get_count[job_id] += 1
        queue = self._job_states.get(job_id)
        if not queue:
            return {
                "status": "invalid_request",
                "error": {"code": "unknown_job", "job_id": job_id},
            }
        value = self._take(queue)
        if isinstance(value, Mapping):
            self._last_job_values[job_id] = deepcopy(dict(value))
        return value

    def _collect_job(self, job_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.collect_count[job_id] += 1
        custom = self._collect_results.get(job_id)
        if custom is not None:
            return deepcopy(custom)
        current = self._last_job_values.get(job_id, {})
        job = current.get("job") if isinstance(current, Mapping) else None
        job_status = job.get("status") if isinstance(job, Mapping) else "queued"
        job_record = (
            deepcopy(dict(job))
            if isinstance(job, Mapping)
            else {"job_id": job_id, "status": job_status}
        )
        terminal = job_status in {"success", "failed", "timeout", "cancelled"}
        job_type = self._job_types.get(job_id, "native_software")
        if job_type == "programmable_analysis":
            software_status = "not_applicable" if terminal else "running"
            convergence_status = "not_declared" if terminal else "pending"
        else:
            software_status = "normal_termination" if terminal else "running"
            convergence_status = "converged" if terminal else "pending"
        return {
            "status": "success",
            "schema_version": 1,
            "job_id": job_id,
            "job_status": job_status,
            "job": job_record,
            "include_inputs": bool(arguments.get("include_inputs", False)),
            "truncated": False,
            "ready": terminal,
            "software_status": software_status,
            "convergence_status": convergence_status,
            "artifact_status": "valid" if terminal else "pending",
            "scientific_validation_status": "mechanically_valid" if terminal else "pending",
            "stdout": "" if terminal else None,
            "outputs": (
                [
                    {
                        "path": "outputs/result.xyz",
                        "job_relative_path": "outputs/result.xyz",
                        "size_bytes": len(b"result"),
                        "sha256": hashlib.sha256(b"result").hexdigest(),
                        "kind": "chemical/x-xyz",
                    }
                ]
                if terminal
                else []
            ),
            "declared_artifacts": (
                [
                    {
                        "name": "result",
                        "path": "outputs/result.xyz",
                        "workspace_path": "outputs/result.xyz",
                        "semantic_type": "AtomicStructure",
                        "media_type": "chemical/x-xyz",
                        "validation_status": "valid",
                        "validation_errors": [],
                    }
                ]
                if terminal
                else []
            ),
            "artifact_manifest": None,
        }

    def _cancel_job(self, job_id: str) -> dict[str, Any]:
        self.cancel_count[job_id] += 1
        custom = self._cancel_results.get(job_id)
        if custom is not None:
            return deepcopy(custom)
        current = self._last_job_values.get(job_id, {})
        job = current.get("job") if isinstance(current, Mapping) else None
        previous_status = job.get("status") if isinstance(job, Mapping) else None
        if previous_status in {"success", "failed", "timeout", "cancelled"}:
            return {
                "status": "success",
                "job_id": job_id,
                "job_status": previous_status,
                "cancellation_sent": False,
            }
        return {
            "status": "success",
            "job_id": job_id,
            "job_status": "cancelled",
            "cancellation_sent": True,
            "message": "cancellation requested",
        }

    def _declare_artifact(self, arguments: Mapping[str, Any]) -> Any:
        key = str(arguments.get("artifact_id") or arguments.get("path") or "default")
        queue = self._artifact_results.get(key) or self._artifact_results.get("default")
        if queue:
            value = self._take(queue)
            return value(arguments) if callable(value) else value
        producer_layer = arguments.get("producer_layer")
        producer_id = arguments.get("producer_id")
        if not isinstance(producer_layer, str) or not isinstance(producer_id, str):
            raise AssertionError("declare_scientific_artifact 必须使用 producer_layer/producer_id")
        producer_action = f"{producer_layer}:{producer_id}"
        producer_backend = producer_id if producer_layer == "native_software" else None
        artifact = build_artifact_ref(
            artifact_id="art_declared_0001",
            semantic_type=str(arguments.get("semantic_type", "AtomicStructure")),
            media_type=str(arguments.get("media_type", "chemical/x-xyz")),
            path=str(arguments.get("path", "outputs/result.xyz")),
            producer_action=producer_action,
            producer_backend=producer_backend,
            parent_artifact_ids=list(arguments.get("parent_artifact_ids", [])),
        )
        return {
            "status": "success",
            "artifact": artifact,
            "lineage_note": (
                "This declaration records semantics and parent ids supplied by the Agent; "
                "it does not infer or alter scientific meaning."
            ),
        }

    @staticmethod
    def _default_open_software() -> dict[str, Any]:
        """返回一个最小可用的 native 软件 discovery 响应。"""

        return {
            "software_id": "xtb",
            "display_name": "xTB",
            "available": True,
            "native_executables": [
                {"executable": "xtb", "enabled": True, "available": True}
            ],
        }

    @staticmethod
    def _default_open_software_inspection(software_id: str) -> dict[str, Any]:
        """返回一个最小 native invocation guide。"""

        return {
            "status": "success",
            "software_id": software_id,
            "display_name": "xTB",
            "native_invocation_guides": [
                {
                    "executable": "xtb",
                    "enabled": True,
                    "available": True,
                    "resolved_path": "/fake/bin/xtb",
                    "synopsis": "xtb structure.xyz --opt",
                    "input_mode": "arguments",
                    "required_files": ["structure.xyz"],
                    "native_job_request_template": {
                        "software_id": software_id,
                        "executable": "xtb",
                        "arguments": ["structure.xyz", "--opt"],
                        "staged_inputs": [
                            {"source_path": "code/structure.xyz", "target_path": "structure.xyz"}
                        ],
                        "resource_limits": {
                            "memory_mb": 4096,
                            "cpu_cores": 1,
                            "gpu_count": 0,
                        },
                        "calculation_intent": "geometry_optimization",
                        "label": "fake native job",
                        "parent_job_id": None,
                    },
                }
            ],
        }

    @staticmethod
    def _default_analysis_runtimes() -> dict[str, Any]:
        """返回一个最小可用的 analysis runtime discovery 响应。"""

        return {
            "status": "success",
            "runtimes": [
                {
                    "runtime": "core",
                    "description": "Deterministic chemistry analysis.",
                    "available": True,
                    "modules": {"numpy": {"available": None, "version": None}},
                }
            ],
        }

    def _read_workspace_text(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """在内存工作区中实现 read_workspace_text 的成功/缺失响应。"""

        path = str(arguments.get("path", ""))
        content = self._workspace_files.get(path)
        if content is None:
            return {
                "status": "invalid_request",
                "error": {
                    "code": "workspace_file_not_found",
                    "exception_type": "FileNotFoundError",
                    "message": f"Workspace file not found: {path}",
                },
            }
        encoded = content.encode("utf-8")
        max_chars = int(arguments.get("max_chars", 100_000))
        offset = int(arguments.get("offset", 0))
        json_pointer = arguments.get("json_pointer")
        selected_text = content
        if json_pointer is not None:
            selected_value = _select_fake_json_pointer_value(content, str(json_pointer))
            selected_text = json.dumps(
                selected_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        total_chars = len(selected_text)
        end = min(offset + max_chars, total_chars)
        page = selected_text[offset:end]
        truncated = end < total_chars
        return {
            "status": "success",
            "path": path,
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "encoding": "utf-8",
            "offset": offset,
            "returned_chars": len(page),
            "next_offset": end if truncated else None,
            "total_chars": total_chars,
            "truncated": truncated,
            "json_pointer": json_pointer,
            "content": page,
        }

    def _write_workspace_text(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """在内存工作区中实现 overwrite=false 的原子写入边界。"""

        path = str(arguments.get("path", ""))
        if path in self._workspace_files and arguments.get("overwrite") is not True:
            return {
                "status": "invalid_request",
                "error": {
                    "code": "workspace_file_exists",
                    "exception_type": "FileExistsError",
                    "message": f"Workspace file already exists: {path}",
                },
            }
        content = arguments.get("content")
        if not isinstance(content, str):
            return {
                "status": "invalid_request",
                "error": {"code": "invalid_content", "message": "content must be text"},
            }
        self._workspace_files[path] = content
        encoded = content.encode("utf-8")
        return {
            "status": "success",
            "path": path,
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "encoding": "utf-8",
        }

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """记录并执行一个确定性工具调用。"""

        wire_arguments = deepcopy(dict(arguments or {}))
        self.calls.append(FakeToolCall(name=name, arguments=wire_arguments))
        self._raise_injected_error(name)
        payload = self._unwrap_tool_request(name, wire_arguments)
        if name == "list_software":
            self.list_software_count += 1
        elif name == "inspect_software":
            self.inspect_software_count += 1
        elif name == "list_analysis_runtimes":
            self.list_analysis_runtimes_count += 1
        elif name == "read_workspace_text":
            self.read_workspace_text_count += 1
            self.read_workspace_count += 1
        elif name == "write_workspace_text":
            self.write_workspace_text_count += 1
            self.write_workspace_count += 1
        elif name == "validate_native_job":
            self.validate_native_job_count += 1
            self.native_validation_count += 1
        elif name == "validate_analysis_program":
            self.validate_analysis_program_count += 1
            self.analysis_validation_count += 1
        injected = self._take_tool_result(name, payload)
        if injected is not None:
            value = injected(payload) if callable(injected) else injected
            if isinstance(value, Mapping) and (
                "content" in value or "structuredContent" in value or "isError" in value
            ):
                return value
            return self._wrap_tool_result(value)
        if name == "inspect_action":
            value = {
                "status": "success",
                "catalog_hash": self.catalog.get("catalog_hash"),
                "catalog_version": str(self.catalog.get("schema_version")),
                "action": {
                    "id": payload.get("action_id"),
                    "category": "structure_and_system",
                    "description": "Convert an explicit molecular structure.",
                    "primary_output": "AtomicStructure",
                    "backend_ids": ["openbabel"],
                    "required_inputs": ["structure"],
                    "optional_inputs": [],
                    "selection_policy": "agent_backend_required",
                    "data_action": False,
                    "batch_safe": False,
                },
                "selected_request_contract": {
                    "required_contract": {
                        "inputs": [
                            {"name": "structure", "required": True, "type": "AtomicStructure"}
                        ],
                        "method_spec": [],
                        "action_settings": [],
                        "component_backends": [],
                    },
                },
            }
            return self._wrap_tool_result(value)
        if name == "inspect_backend":
            value = {
                "status": "success",
                "catalog_hash": self.catalog.get("catalog_hash"),
                "catalog_version": str(self.catalog.get("schema_version")),
                "backend": {
                    "backend_id": payload.get("backend_id", "openbabel"),
                    "display_name": "Open Babel",
                    "runtime": "native",
                },
                "capability_count": 1,
                "capabilities": [{"action_id": "convert_structure"}],
            }
            return self._wrap_tool_result(value)
        if name == "list_software":
            return self._wrap_tool_result(
                {
                    "status": "success",
                    "count": 1,
                    "total_matching": 1,
                    "offset": 0,
                    "next_offset": None,
                    "software": [self._default_open_software()],
                }
            )
        if name == "inspect_software":
            return self._wrap_tool_result(
                self._default_open_software_inspection(str(payload.get("software_id", "xtb")))
            )
        if name == "list_analysis_runtimes":
            return self._wrap_tool_result(self._default_analysis_runtimes())
        if name == "read_workspace_text":
            return self._wrap_tool_result(self._read_workspace_text(payload))
        if name == "write_workspace_text":
            return self._wrap_tool_result(self._write_workspace_text(payload))
        if name == "validate_native_job":
            value = self._validation_results.get("validate_native_job")
            if value is None:
                value = {
                    "status": "success",
                    "valid": True,
                    "software_id": payload.get("software_id"),
                    "executable": payload.get("executable"),
                    "operation_id": payload.get("operation_id"),
                }
            return self._wrap_tool_result(value(payload) if callable(value) else value)
        if name == "validate_analysis_program":
            value = self._validation_results.get("validate_analysis_program")
            if value is None:
                value = {
                    "status": "success",
                    "valid": True,
                    "runtime": payload.get("runtime"),
                    "operation_id": payload.get("operation_id"),
                }
            return self._wrap_tool_result(value(payload) if callable(value) else value)
        if name == "execute_action":
            self.submit_count += 1
            return self._wrap_tool_result(self._call_action(payload))
        if name == "submit_native_job":
            return self._wrap_tool_result(self._submit_native_job(payload))
        if name == "submit_analysis_program":
            return self._wrap_tool_result(self._submit_analysis_job(payload))
        if name == "get_execution_job":
            return self._wrap_tool_result(self._get_job(str(payload["job_id"])))
        if name == "collect_execution_job":
            return self._wrap_tool_result(self._collect_job(str(payload["job_id"]), payload))
        if name == "cancel_execution_job":
            return self._wrap_tool_result(self._cancel_job(str(payload["job_id"])))
        if name == "declare_scientific_artifact":
            return self._wrap_tool_result(self._declare_artifact(payload))
        raise KeyError(f"unknown fake MCP tool: {name}")

    @staticmethod
    def _wrap_tool_result(value: Any) -> dict[str, Any]:
        """把普通 fake 值编码为 MCP CallToolResult 的纯文本内容。"""

        text = value if isinstance(value, str) else json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": None,
            "isError": False,
        }


__all__ = ["FakeMcpSession", "FakeToolCall", "build_artifact_ref", "build_catalog_snapshot"]
