"""把作者未修改的 Gaussian 代码生成工具适配为 MCP Action。"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
import re
import time
import types
from typing import Any, Callable, Mapping

import openai

from ..vendor.arche_author_tools import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY
from ..vendor.arche_author_tools import gen_gaussiancode as author_codegen
from ..runtime import (
    GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT,
    GAUSSIAN_CODEGEN_REQUIRED_MODEL_NAME,
    inspect_gaussian_codegen_configuration,
)
from .common import failed, request_parts


ACTIONS = {"generate_gaussian_code"}

prompt_for_agent = author_codegen.prompt_for_agent
prompt_for_demand = author_codegen.prompt_for_demand
prompt_for_getcode = author_codegen.prompt_for_getcode
prompt_for_fixcode = author_codegen.prompt_for_fixcode

_STAGES = ("demand", "agent", "extract", "fix")
_MAX_USER_INPUT_CHARS = 20_000
_MAX_ROUTE_CHARS = 8_192
_ROUTE_ALLOWED_CHARS = re.compile(r"^[#A-Za-z0-9_+*(),./=:\-\s]+$")
_FLOAT_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_COORDINATE_TRIPLE = re.compile(
    rf"(?<!\w){_FLOAT_TOKEN}\s+{_FLOAT_TOKEN}\s+{_FLOAT_TOKEN}(?!\w)"
)

_MODEL_API_RETRY_COUNT = 5
_MODEL_API_RETRY_DELAY_SECONDS = 6.0
# OpenAI-compatible 推理模型把 thinking 和最终正文计入同一输出预算。作者 fix
# 阶段需要在推理后仍能返回修正后的 route；2048 左右的预算会偶发只留下思考内容。
_AUTHOR_MODEL_MAX_TOKENS = 4096
# 首次输出被截断时只允许一次有界预算升级，避免无限重试或无界放大请求。
_AUTHOR_MODEL_TRUNCATED_MAX_TOKENS = 8192
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
_ORIGINAL_AUTHOR_API_CALL = author_codegen.make_api_call_with_retry
_ORIGINAL_AUTHOR_RESULT_FUNCTION = author_codegen.generate_gaussian_code_result


class _ExpertConfigurationError(RuntimeError):
    """表示 MCP 无法用当前配置调用作者专家工具。"""


class _ExpertProtocolError(RuntimeError):
    """表示作者工具的最终输出不能作为单行 Gaussian route 使用。"""


class _AuthorModelTruncatedError(RuntimeError):
    """表示作者专家输出达到预算上限，未形成完整的阶段结果。"""


def _response_http_status(value: Any) -> int | None:
    """从 API 响应或异常中读取 HTTP 状态，不读取响应正文。"""

    if isinstance(value, Mapping):
        candidate = value.get("status_code", value.get("status"))
    else:
        candidate = getattr(value, "status_code", None)
        if candidate is None:
            candidate = getattr(value, "status", None)
        if candidate is None:
            response = getattr(value, "response", None)
            candidate = getattr(response, "status_code", None)
            if candidate is None:
                candidate = getattr(response, "status", None)
    if isinstance(candidate, bool) or candidate is None:
        return None
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def _response_finish_reason(value: Any) -> str | None:
    """从 OpenAI-compatible 响应读取首个 choice 的结束原因。"""

    choices = (
        value.get("choices")
        if isinstance(value, Mapping)
        else getattr(value, "choices", None)
    )
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    choice = choices[0]
    if isinstance(choice, Mapping):
        reason = choice.get("finish_reason")
    else:
        reason = getattr(choice, "finish_reason", None)
    if reason is None:
        return None
    return str(reason).strip().casefold()


def _is_retryable_model_exception(error: BaseException) -> bool:
    """只重试连接、超时、429 和 5xx，拒绝重试鉴权及参数错误。"""

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = _response_http_status(current)
        if status is not None:
            return status == 429 or 500 <= status <= 599
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True

        module_name = type(current).__module__.casefold()
        class_name = type(current).__name__.casefold()
        if module_name.startswith("openai"):
            if "timeout" in class_name or "connection" in class_name:
                return True
            if class_name in {
                "ratelimiterror",
                "internalservererror",
                "serviceunavailableerror",
                "badgatewayerror",
                "gatewaytimeouterror",
            }:
                return True
        elif module_name.startswith("httpx"):
            if (
                "timeout" in class_name
                or "connect" in class_name
                or "network" in class_name
                or class_name in {"readerror", "writeerror", "closeerror", "remoteprotocolerror"}
            ):
                return True
        elif class_name in {
            "connectionerror",
            "connecterror",
            "connectionclosederror",
            "networkerror",
            "readtimeout",
            "connecttimeout",
            "timeout",
            "timeouterror",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def _retry_model_api_call(
    operation: Callable[[], Any],
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    """执行作者单次模型请求，首次失败后最多重试五次，每次等待六秒。"""

    total_attempts = _MODEL_API_RETRY_COUNT + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = operation()
            status = _response_http_status(response)
            if status == 429 or (status is not None and 500 <= status <= 599):
                if attempt == total_attempts:
                    return response
            else:
                return response
        except Exception as exc:
            if attempt == total_attempts or not _is_retryable_model_exception(exc):
                raise
        sleeper(_MODEL_API_RETRY_DELAY_SECONDS)
    raise AssertionError("模型 API 重试循环必须返回")


def _create_model_api_client(
    *, api_key: str, base_url: str, timeout: int
) -> openai.OpenAI:
    """创建独立 OpenAI 客户端，明确关闭 SDK 内部重试，避免配置竞态。"""

    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )


def _call_author_model_api(
    messages: list[dict[str, str]],
    api_key: str | None = None,
    base_url: str = "",
    model: str = "",
    timeout: int = 120,
    temperature: float = 0.7,
    max_tokens: int = _AUTHOR_MODEL_MAX_TOKENS,
    *,
    retry_sleeper: Callable[[float], None] = time.sleep,
) -> str:
    """调用作者所需的 OpenAI-compatible 模型请求，并统一处理瞬时失败。"""

    if not api_key:
        raise ValueError("模型 API key 不能为空")
    client = _create_model_api_client(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    def request_with_budget(output_budget: int) -> Any:
        """按既有传输重试策略请求一次指定输出预算。"""

        return _retry_model_api_call(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=output_budget,
                timeout=timeout,
            ),
            sleeper=retry_sleeper,
        )

    response = request_with_budget(max_tokens)
    finish_reason = _response_finish_reason(response)
    if finish_reason in _TRUNCATED_FINISH_REASONS:
        if max_tokens >= _AUTHOR_MODEL_TRUNCATED_MAX_TOKENS:
            raise _AuthorModelTruncatedError(
                "作者 Gaussian 模型响应因输出预算被截断"
            )
        response = request_with_budget(_AUTHOR_MODEL_TRUNCATED_MAX_TOKENS)
        if _response_finish_reason(response) in _TRUNCATED_FINISH_REASONS:
            raise _AuthorModelTruncatedError(
                "作者 Gaussian 模型响应在预算升级后仍被截断"
            )
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("模型 API 响应缺少 choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型 API 响应内容为空")
    return content


def _clone_author_functions_with_api_call(api_call: Callable[..., str]) -> Callable[..., Any]:
    """复制作者函数的全局环境，只替换模型调用函数，避免修改 vendored 模块全局状态。"""

    original_globals = _ORIGINAL_AUTHOR_RESULT_FUNCTION.__globals__
    isolated_globals = dict(original_globals)
    cloned_functions: dict[str, types.FunctionType] = {}
    for name, value in original_globals.items():
        if not isinstance(value, types.FunctionType) or value.__globals__ is not original_globals:
            continue
        cloned = types.FunctionType(
            value.__code__,
            isolated_globals,
            value.__name__,
            value.__defaults__,
            value.__closure__,
        )
        cloned.__kwdefaults__ = value.__kwdefaults__
        cloned.__annotations__ = dict(value.__annotations__)
        cloned.__doc__ = value.__doc__
        cloned.__qualname__ = value.__qualname__
        cloned_functions[name] = cloned
    isolated_globals.update(cloned_functions)
    isolated_globals["make_api_call_with_retry"] = api_call
    return cloned_functions["generate_gaussian_code_result"]


def _invoke_author_codegen_result(**kwargs: Any) -> dict[str, Any]:
    """调用作者四阶段流程；生产调用使用隔离全局环境注入统一模型 API。"""

    current_result_function = author_codegen.generate_gaussian_code_result
    if current_result_function is not _ORIGINAL_AUTHOR_RESULT_FUNCTION:
        # 测试或显式扩展可以替换整个作者流程；此时保持其注入点原样可控。
        return current_result_function(**kwargs)
    current_api_function = author_codegen.make_api_call_with_retry
    api_call = (
        current_api_function
        if current_api_function is not _ORIGINAL_AUTHOR_API_CALL
        else _call_author_model_api
    )
    isolated_result_function = _clone_author_functions_with_api_call(api_call)
    return isolated_result_function(**kwargs)


def _read_author_expert_configuration() -> dict[str, str]:
    """读取并校验作者专家配置，返回当前调用所需参数。

    函数无显式输入，读取当前进程的四项 ``ARCHE_CHEM_*`` 连接环境变量和独立
    的 Gaussian codegen 模型变量，并复用 Gaussian codegen 的无网络完整性和
    格式检查。它只在内存中整理配置，不发 HTTP 请求、不写文件、不打印或记录
    密钥；返回的字典供随后作者四阶段调用使用，包含已校验的 API key、Base URL、
    固定为 DeepSeek Pro 的模型名称和 Backend 名称。

    配置缺失或格式非法时抛出 ``_ExpertConfigurationError``，错误消息只说明
    缺失变量名或安全原因；独立模型变量省略时使用固定的
    ``deepseek-v4-pro``，显式的非 Pro 值会被拒绝；函数不会回退到其他模型或
    Backend。
    """

    configuration_health = inspect_gaussian_codegen_configuration(os.environ)
    if not configuration_health["available"]:
        missing = configuration_health["missing_environment"]
        invalid = configuration_health["invalid_environment"]
        if missing:
            message = "ARCHE-Chem Gaussian 专家配置不完整；缺少：" + ", ".join(missing)
        elif "ARCHE_CHEM_BACKEND" in invalid:
            message = "ARCHE_CHEM_BACKEND 不是允许的 OpenAI-compatible HTTP 模式"
        elif GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT in invalid:
            message = "Gaussian codegen 模型必须固定为 DeepSeek Pro"
        else:
            message = "ARCHE_CHEM_BASE_URL 必须是 http 或 https 服务地址"
        raise _ExpertConfigurationError(message)
    backend = os.environ["ARCHE_CHEM_BACKEND"].strip().casefold()
    base_url = os.environ["ARCHE_CHEM_BASE_URL"].strip()
    model_name = (
        os.environ.get(GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT, "").strip()
        or GAUSSIAN_CODEGEN_REQUIRED_MODEL_NAME
    )
    return {
        "api_key": os.environ["ARCHE_CHEM_API_KEY"].strip(),
        "base_url": base_url,
        "model_name": model_name,
        "backend": backend,
    }


def _validate_author_gaussian_route(candidate: Any) -> str:
    """验证作者最终 route 并原样返回；不改写科学关键词，失败时拒绝完整 deck 或注入内容。"""

    if not isinstance(candidate, str):
        raise _ExpertProtocolError("最终 gaussian_code 不是字符串")
    route = candidate.strip()
    if not route:
        raise _ExpertProtocolError("最终 gaussian_code 为空")
    if len(route) > _MAX_ROUTE_CHARS:
        raise _ExpertProtocolError("最终 gaussian_code 超过长度上限")
    if "\n" in route or "\r" in route:
        raise _ExpertProtocolError("gaussian_code 只能包含单行 route section")
    if any(ord(character) < 32 for character in route):
        raise _ExpertProtocolError("gaussian_code 不得包含控制字符")
    if not route.startswith("#"):
        raise _ExpertProtocolError("gaussian_code 必须以 # 开头")
    if "#" in route[1:] or ".." in route:
        raise _ExpertProtocolError("gaussian_code 包含额外指令或路径片段")
    lowered = route.casefold()
    forbidden_markers = (
        "--link1--",
        "%chk",
        "%mem",
        "%nproc",
        "title",
        "charge",
        "multiplicity",
        "coordinates",
        "gaussian input",
        "gjf",
        "com file",
    )
    if any(marker in lowered for marker in forbidden_markers):
        raise _ExpertProtocolError(
            "gaussian_code 包含 title、charge、坐标或文件内容标记"
        )
    solvent_values = [
        match.group(1).strip()
        for match in re.finditer(
            r"(?i)\bsolvent\s*=\s*([^,)]+)",
            route,
        )
    ]
    if any(re.search(r"\s", value) for value in solvent_values):
        raise _ExpertProtocolError(
            "gaussian_code 的 solvent 必须使用不含空格的 Gaussian 原生别名"
        )
    if any(value.casefold() == "ethylacetate" for value in solvent_values):
        raise _ExpertProtocolError(
            "gaussian_code 的乙酸乙酯 solvent 必须使用 Gaussian 原生别名 EthylEthanoate"
        )
    if re.search(r"(?<!\w)[+-]?\d+\s+\d+(?!\w)", route):
        raise _ExpertProtocolError("gaussian_code 不得包含 charge/multiplicity 行")
    if _COORDINATE_TRIPLE.search(route):
        raise _ExpertProtocolError("gaussian_code 不得包含坐标三元组")
    if not _ROUTE_ALLOWED_CHARS.fullmatch(route):
        raise _ExpertProtocolError("gaussian_code 包含未允许的文件或脚本字符")
    if not re.match(r"^#[A-Za-z]*\s+\S+", route):
        raise _ExpertProtocolError("gaussian_code 的 # 后必须包含 route 关键词")
    return route


def _select_validated_author_route(author_result: Mapping[str, Any]) -> str:
    """按作者输出优先级选择安全 route；最终修正字段失控时回退到作者已提取的 route。"""

    errors: list[tuple[str, _ExpertProtocolError]] = []
    for field_name in ("gaussian_code", "raw_gaussian_code"):
        try:
            return _validate_author_gaussian_route(author_result.get(field_name))
        except _ExpertProtocolError as exc:
            errors.append((field_name, exc))
    if errors:
        details = "；".join(f"{field_name}: {error}" for field_name, error in errors)
        raise _ExpertProtocolError(details)
    raise _ExpertProtocolError("作者结果没有 Gaussian route")


def _safe_author_metadata(value: Any) -> dict[str, Any]:
    """筛出不含密钥、地址、模型路径和异常原文的作者元数据；不修改作者返回对象。"""

    metadata = value if isinstance(value, Mapping) else {}
    raw_steps = metadata.get("processing_steps")
    safe_steps = []
    if isinstance(raw_steps, list):
        for item in raw_steps:
            text = str(item)
            if re.search(
                r"stage=(?:demand|agent|extract|fix) "
                r"(?:start|api_fallback|done via (?:api|local)|local_empty -> fallback)$",
                text,
            ):
                safe_steps.append(text)
    stage_backends = metadata.get("stage_backends")
    if not isinstance(stage_backends, Mapping):
        stage_backends = {}
    return {
        "backend_used": metadata.get("backend_used"),
        "configured_backend": metadata.get("configured_backend"),
        "used_local_model": bool(metadata.get("used_local_model")),
        "used_api_fallback": bool(metadata.get("used_api_fallback")),
        "stage_backends": {
            str(key): str(value)
            for key, value in stage_backends.items()
            if str(key) in _STAGES and value is not None
        },
        "processing_steps": safe_steps,
    }


def _author_stage_statuses(
    metadata: Mapping[str, Any], *, succeeded: bool
) -> dict[str, dict[str, Any]]:
    """根据作者处理记录生成 MCP 阶段状态；不推断科学内容，只标记已开始、成功或跳过。"""

    statuses = {
        stage: {"status": "pending"}
        for stage in _STAGES
    }
    completed: set[str] = set()
    started: list[str] = []
    for item in metadata.get("processing_steps") or []:
        match = re.search(r"stage=(demand|agent|extract|fix) (.+)$", str(item))
        if match is None:
            continue
        stage, event = match.groups()
        if stage not in started:
            started.append(stage)
        if event.startswith("done via"):
            completed.add(stage)
    for stage in completed:
        statuses[stage]["status"] = "success"
    if succeeded:
        for stage in _STAGES:
            statuses[stage]["status"] = "success"
        return statuses
    failed_stage = next(
        (stage for stage in reversed(started) if stage not in completed),
        started[-1] if started else "demand",
    )
    statuses[failed_stage]["status"] = "failed"
    failed_index = _STAGES.index(failed_stage)
    for stage in _STAGES[failed_index + 1 :]:
        statuses[stage]["status"] = "skipped"
    return statuses


def _build_author_compatible_result(
    user_input: str,
    author_result: Any,
    *,
    configured_backend: str,
) -> dict[str, Any]:
    """复制作者公开结果字段并加入脱敏 MCP 元数据；不保留模型路径或异常正文。"""

    source = author_result if isinstance(author_result, Mapping) else {}
    safe_metadata = _safe_author_metadata(source.get("metadata"))
    succeeded = source.get("success") is True
    safe_metadata.update(
        {
            "expert_backend": "ARCHE-Chem",
            "transport": "openai_compatible_http",
            "configured_backend": configured_backend,
            "automatic_fallback": False,
            "stage_order": list(_STAGES),
            "stages": _author_stage_statuses(safe_metadata, succeeded=succeeded),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_function": "generate_gaussian_code_result",
        }
    )
    return {
        "success": succeeded,
        "user_input": user_input,
        "demand_analysis": str(source.get("demand_analysis") or ""),
        "task_description": str(source.get("task_description") or ""),
        "raw_model_answer": str(source.get("raw_model_answer") or ""),
        "raw_gaussian_code": str(source.get("raw_gaussian_code") or ""),
        "gaussian_code": str(source.get("gaussian_code") or ""),
        "message": (
            "Gaussian code generated successfully"
            if succeeded
            else "ARCHE-Chem 作者 Gaussian 代码生成流程失败"
        ),
        "metadata": safe_metadata,
    }


def _result_envelope(
    status: str,
    result: dict[str, Any],
    *,
    code: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """封装作者结果并返回 MCP envelope；不写文件，错误信息不包含凭据或服务地址。"""

    envelope: dict[str, Any] = {
        "status": status,
        "result": result,
        "artifact_files": [],
        "backend_version": "arche_author_codegen_" + UPSTREAM_COMMIT[:8],
        "warnings": [],
        "provenance": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_function": "generate_gaussian_code_result",
            "adapter": "mcp_configuration_redaction_and_route_validation_v1",
            "automatic_fallback": False,
        },
        "retryable": False,
    }
    if code is not None:
        envelope["error"] = {"code": code, "message": message or result["message"]}
    return envelope


def _empty_author_result(user_input: str) -> dict[str, Any]:
    """建立配置或请求失败时的作者兼容空结果；不产生副作用。"""

    return _build_author_compatible_result(
        user_input,
        {"success": False, "metadata": {"processing_steps": []}},
        configured_backend="",
    )


def _generate_gaussian_code_with_author_tool(
    request: dict[str, Any]
) -> dict[str, Any]:
    """调用作者四阶段函数并校验最终 route；只发专家 HTTP 请求，不生成 GJF 或运行 Gaussian。"""

    inputs, method_spec, action_settings = request_parts(request)
    user_input_value = inputs.get("user_input")
    user_input = user_input_value.strip() if isinstance(user_input_value, str) else ""
    unexpected_settings = sorted(set(action_settings) - {"timeout_seconds"})
    if method_spec or unexpected_settings:
        result = _empty_author_result(user_input)
        result["message"] = "generate_gaussian_code 不接受 method_spec 或 action_settings"
        return _result_envelope(
            "failed", result, code="conflicting_parameter_source"
        )
    unknown_inputs = sorted(set(inputs) - {"user_input"})
    if unknown_inputs:
        result = _empty_author_result(user_input)
        result["message"] = "generate_gaussian_code 只接受 inputs.user_input"
        return _result_envelope("failed", result, code="unsupported_input_fields")
    if not user_input or len(user_input) > _MAX_USER_INPUT_CHARS:
        result = _empty_author_result(user_input)
        result["message"] = "inputs.user_input 必须是长度受限的非空字符串"
        return _result_envelope("failed", result, code="invalid_user_input")
    try:
        configuration = _read_author_expert_configuration()
    except _ExpertConfigurationError as exc:
        result = _empty_author_result(user_input)
        result["message"] = str(exc)
        result["metadata"]["configuration_status"] = "missing_or_invalid"
        return _result_envelope(
            "unavailable", result, code="expert_configuration_missing"
        )

    # 强制作者原函数走当前 MCP 已配置的同一个 HTTP 专家服务。显式把 model
    # 设为 None，避免作者默认本地路径触发 vLLM；没有通用模型回退。
    with redirect_stdout(StringIO()):
        try:
            author_result = _invoke_author_codegen_result(
                user_input=user_input,
                model_config={
                    "backend": "openai_compatible",
                    "model": None,
                    "model_name": configuration["model_name"],
                },
                api_key=configuration["api_key"],
                api_base_url=configuration["base_url"],
                api_model=configuration["model_name"],
                timeout=author_codegen.DEFAULT_TIMEOUT,
                verbose=False,
            )
        except Exception:
            result = _empty_author_result(user_input)
            result["message"] = "ARCHE-Chem 作者 Gaussian 代码生成流程失败"
            result["metadata"]["configured_backend"] = configuration["backend"]
            result["metadata"]["stages"]["demand"]["status"] = "failed"
            for stage in _STAGES[1:]:
                result["metadata"]["stages"][stage]["status"] = "skipped"
            return _result_envelope(
                "failed",
                result,
                code="author_gaussian_codegen_failed",
                message=result["message"],
            )
    result = _build_author_compatible_result(
        user_input,
        author_result,
        configured_backend=configuration["backend"],
    )
    if not result["success"]:
        return _result_envelope(
            "failed",
            result,
            code="author_gaussian_codegen_failed",
            message=result["message"],
        )
    try:
        validated_route = _select_validated_author_route(result)
    except _ExpertProtocolError as exc:
        result["success"] = False
        result["gaussian_code"] = ""
        result["message"] = str(exc)
        result["metadata"]["stages"]["fix"]["status"] = "failed"
        return _result_envelope(
            "failed", result, code="gaussian_code_invalid", message=str(exc)
        )
    if validated_route != result["gaussian_code"]:
        result["metadata"]["route_selection"] = "raw_gaussian_code_after_fix_rejected"
    else:
        result["metadata"]["route_selection"] = "gaussian_code"
    result["gaussian_code"] = validated_route
    return _result_envelope("success", result)


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """执行作者 Gaussian 代码生成 Action；失败时返回 MCP 状态，不调用其他模型。"""

    if action_id != "generate_gaussian_code" or backend_id != "gaussian_codegen":
        return failed(
            f"Unsupported Gaussian codegen action/backend combination: {action_id}/{backend_id}",
            code="unsupported_request",
        )
    return _generate_gaussian_code_with_author_tool(request)


__all__ = [
    "ACTIONS",
    "execute",
    "prompt_for_agent",
    "prompt_for_demand",
    "prompt_for_getcode",
    "prompt_for_fixcode",
]
