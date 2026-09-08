from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest

from minichem_toolbox.backends import gaussian_codegen
from minichem_toolbox.catalog import (
    action_specs,
    backend_specs,
    catalog_snapshot,
    validate_catalog,
)
from minichem_toolbox.discovery import inspect_action, search_actions
from minichem_toolbox import runtime
from minichem_toolbox.backend_specs import BACKEND_SPECS
from minichem_toolbox.runtime import (
    GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT,
    GAUSSIAN_CODEGEN_REQUIRED_MODEL_NAME,
    runtime_spec,
)
from minichem_mcp_tools import discovery_tools
from minichem_mcp_tools.discovery_models import ProgressiveActionRequest


def _configure(monkeypatch):
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "test-key")
    monkeypatch.setenv("ARCHE_CHEM_BASE_URL", "https://expert.invalid/v1")
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "arche-chem-test")
    monkeypatch.setenv("ARCHE_CHEM_BACKEND", "openai_compatible")


def _author_stage_responses(fix_text: str | None = None) -> list[str]:
    route = "#P M06-2X/6-31G(d) Opt=(TS,CalcFC) Freq Temperature=303.15"
    return [
        "Analysis: Transition-state optimization needs a TS search and frequency check.\n"
        "Gaussian computational task description: Locate a transition state and verify it "
        "with frequencies at 303.15 K.",
        route,
        f"Gaussian_code: {route}",
        fix_text
        or "Analysis: The route is concise and valid.\n"
        f"Corrected Gaussian keyword line: {route}",
    ]


class _ResponseQueue:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages = []

    def __call__(self, messages, **_kwargs):
        self.messages.append(messages)
        return self.responses.pop(0)


class _LocalExpertHandler(BaseHTTPRequestHandler):
    """为跨进程 MCP 测试提供最小 OpenAI-compatible 本地服务。"""

    def do_POST(self):  # noqa: N802 - HTTP 标准方法名
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(request)
        content = self.server.responses.pop(0)
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


@contextmanager
def _serve_author_stage_responses(responses: list[str]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LocalExpertHandler)
    server.responses = list(responses)
    server.requests = []
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_codegen_action_and_backend_use_author_input_contract():
    validate_catalog()
    assert "gaussian_codegen" in runtime_spec("workflows")["backends"]
    assert "generate_gaussian_code" in action_specs()
    assert action_specs()["generate_gaussian_code"].required_inputs == ("user_input",)
    assert action_specs()["generate_gaussian_code"].optional_inputs == ()
    assert backend_specs()["gaussian_codegen"].required_input_fields == {
        "generate_gaussian_code": ("user_input",)
    }
    assert backend_specs()["gaussian_codegen"].license_class == (
        "academic_noncommercial"
    )
    assert "optional" in backend_specs()["gaussian_codegen"].description.casefold()
    assert "non-default" in backend_specs()["gaussian_codegen"].description.casefold()
    assert backend_specs()["gaussian_codegen"].python_modules == ("openai",)
    contract = inspect_action(
        "generate_gaussian_code", backend_id="gaussian_codegen", detail_level="contract"
    )["selected_request_contract"]
    assert {item["name"] for item in contract["required_contract"]["inputs"]} == {
        "user_input"
    }
    assert contract["optional_field_names"]["inputs"] == []
    assert contract["optional_field_names"]["method_spec"] == []
    assert contract["optional_field_names"]["action_settings"] == []


def test_codegen_health_requires_all_author_expert_configuration(monkeypatch):
    """专家配置缺项时，健康检查不能因模块和 API key 存在而误报可用。"""

    config_names = (
        "ARCHE_CHEM_API_KEY",
        "ARCHE_CHEM_BASE_URL",
        "ARCHE_CHEM_BACKEND",
    )
    for name in config_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "test-key")
    gaussian_spec = next(
        item for item in BACKEND_SPECS if item.id == "gaussian_codegen"
    )

    health = runtime.probe_all_backends((gaussian_spec,))["gaussian_codegen"]

    assert health["python_modules"]["openai"]["available"] is True
    assert health["available"] is False
    assert health["reason_code"] == "environment_missing"
    assert health["missing_environment"] == [
        "ARCHE_CHEM_BASE_URL",
        "ARCHE_CHEM_BACKEND",
    ]


@pytest.mark.parametrize(
    "missing_name",
    (
        "ARCHE_CHEM_API_KEY",
        "ARCHE_CHEM_BASE_URL",
        "ARCHE_CHEM_BACKEND",
    ),
)
def test_codegen_health_reports_each_missing_required_configuration(
    monkeypatch, missing_name
):
    """三项连接配置分别缺失时，健康结果只暴露缺失字段名。"""

    _configure(monkeypatch)
    monkeypatch.delenv(missing_name)
    gaussian_spec = next(
        item for item in BACKEND_SPECS if item.id == "gaussian_codegen"
    )

    health = runtime.probe_all_backends((gaussian_spec,))["gaussian_codegen"]

    assert health["available"] is False
    assert health["reason_code"] == "environment_missing"
    assert health["missing_environment"] == [missing_name]
    assert missing_name in health["environment"]
    assert health["environment"][missing_name] is False
    assert "test-key" not in json.dumps(health)


def test_codegen_health_accepts_complete_configuration_without_network(monkeypatch):
    """四项配置完整且格式合法时，健康检查只做静态判断。"""

    _configure(monkeypatch)
    gaussian_spec = next(
        item for item in BACKEND_SPECS if item.id == "gaussian_codegen"
    )

    health = runtime.probe_all_backends((gaussian_spec,))["gaussian_codegen"]

    assert health["available"] is True
    assert health["reason_code"] == "ok"
    assert health["missing_environment"] == []
    assert health["invalid_environment"] == []
    assert "test-key" not in json.dumps(health)


def test_codegen_health_defaults_to_dedicated_deepseek_pro_model(monkeypatch):
    """省略专用模型变量时只能使用固定的 DeepSeek Pro 默认值。"""

    _configure(monkeypatch)
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "a-different-general-model")
    monkeypatch.delenv(GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT, raising=False)
    gaussian_spec = next(
        item for item in BACKEND_SPECS if item.id == "gaussian_codegen"
    )

    health = runtime.probe_all_backends((gaussian_spec,))["gaussian_codegen"]

    assert health["available"] is True
    assert health["environment"][GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT] is True
    assert health["missing_environment"] == []
    configuration = runtime.inspect_gaussian_codegen_configuration(
        {
            "ARCHE_CHEM_API_KEY": "test-key",
            "ARCHE_CHEM_BASE_URL": "https://expert.invalid/v1",
            "ARCHE_CHEM_BACKEND": "openai_compatible",
        }
    )
    assert configuration["defaulted_environment"] == [
        GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT
    ]


def test_codegen_health_rejects_non_pro_dedicated_model(monkeypatch):
    """专用模型显式改成非 Pro 时必须在健康检查阶段拒绝。"""

    _configure(monkeypatch)
    monkeypatch.setenv("ARCHE_CHEM_MODEL_NAME", "a-different-general-model")
    monkeypatch.setenv(GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT, "deepseek-v4-flash")
    gaussian_spec = next(
        item for item in BACKEND_SPECS if item.id == "gaussian_codegen"
    )

    health = runtime.probe_all_backends((gaussian_spec,))["gaussian_codegen"]

    assert health["available"] is False
    assert health["reason_code"] == "environment_invalid"
    assert health["invalid_environment"] == [GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT]
    assert "deepseek-v4-flash" not in json.dumps(health)


def test_codegen_execution_rejects_non_pro_dedicated_model_without_model_call(
    monkeypatch,
):
    """非 Pro 专用模型在执行前失败，不能把请求交给作者四阶段函数。"""

    _configure(monkeypatch)
    monkeypatch.setenv(GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT, "deepseek-v4-flash")
    calls = []
    monkeypatch.setattr(
        gaussian_codegen,
        "_invoke_author_codegen_result",
        lambda **kwargs: calls.append(kwargs),
    )

    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Single point energy."}},
    )

    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "expert_configuration_missing"
    assert "DeepSeek Pro" in result["error"]["message"]
    assert calls == []


def test_available_only_discovery_tracks_complete_codegen_configuration(monkeypatch):
    """available_only 只在四项专家配置齐全时暴露 codegen Action。"""

    gaussian_spec = next(
        item for item in BACKEND_SPECS if item.id == "gaussian_codegen"
    )
    for name in (
        "ARCHE_CHEM_API_KEY",
        "ARCHE_CHEM_BASE_URL",
        "ARCHE_CHEM_MODEL_NAME",
        "ARCHE_CHEM_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARCHE_CHEM_API_KEY", "test-key")

    snapshot = catalog_snapshot(include_health=False)
    incomplete_health = runtime.probe_all_backends((gaussian_spec,))[
        "gaussian_codegen"
    ]
    next(
        item for item in snapshot["backends"] if item["id"] == "gaussian_codegen"
    )["health"] = incomplete_health

    unfiltered = search_actions(
        backend_id="gaussian_codegen",
        retrieval_mode="lexical",
        available_only=False,
        snapshot=snapshot,
    )
    unavailable = search_actions(
        backend_id="gaussian_codegen",
        retrieval_mode="lexical",
        available_only=True,
        snapshot=snapshot,
    )

    assert [item["action_id"] for item in unfiltered["actions"]] == [
        "generate_gaussian_code"
    ]
    assert unavailable["actions"] == []

    _configure(monkeypatch)
    complete_health = runtime.probe_all_backends((gaussian_spec,))[
        "gaussian_codegen"
    ]
    next(
        item for item in snapshot["backends"] if item["id"] == "gaussian_codegen"
    )["health"] = complete_health
    available = search_actions(
        backend_id="gaussian_codegen",
        retrieval_mode="lexical",
        available_only=True,
        snapshot=snapshot,
    )

    assert [item["action_id"] for item in available["actions"]] == [
        "generate_gaussian_code"
    ]


@pytest.mark.parametrize(
    ("variable", "value"),
    (
        ("ARCHE_CHEM_BACKEND", "local_only"),
        ("ARCHE_CHEM_BASE_URL", "not-a-url"),
    ),
)
def test_codegen_health_rejects_invalid_static_configuration(
    monkeypatch, variable, value
):
    """健康检查复用执行入口的 Backend 和 HTTP 地址静态校验。"""

    _configure(monkeypatch)
    monkeypatch.setenv(variable, value)
    gaussian_spec = next(
        item for item in BACKEND_SPECS if item.id == "gaussian_codegen"
    )

    health = runtime.probe_all_backends((gaussian_spec,))["gaussian_codegen"]

    assert health["available"] is False
    assert health["reason_code"] == "environment_invalid"
    assert health["invalid_environment"] == [variable]
    assert value not in json.dumps(health)


def test_progressive_mcp_execute_dispatches_registered_action(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    captured = {}

    def fake_execute(action_id, request, *, operation_id=None):
        captured["action_id"] = action_id
        captured["request"] = request
        captured["operation_id"] = operation_id
        return {"status": "success", "result": {"gaussian_code": "#P B3LYP/6-31G(d) SP"}}

    monkeypatch.setattr(discovery_tools, "_execute_action", fake_execute)
    response = discovery_tools.execute_action(
        ProgressiveActionRequest(
            action_id="generate_gaussian_code",
            backend_id="gaussian_codegen",
            inputs={"user_input": "Single point energy."},
        )
    )
    assert response["status"] == "success"
    assert captured["action_id"] == "generate_gaussian_code"
    assert captured["operation_id"] is None
    assert captured["request"]["inputs"] == {"user_input": "Single point energy."}


def test_public_mcp_execute_runs_author_codegen_without_dispatch_stub(monkeypatch, tmp_path):
    """公开 execute_action 必须到达作者四阶段函数，并把其 route 作为主结果返回。"""

    _configure(monkeypatch)
    monkeypatch.delenv("ARCHE_CHEM_MODEL_NAME", raising=False)
    monkeypatch.delenv(GAUSSIAN_CODEGEN_MODEL_ENVIRONMENT, raising=False)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "1024")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    with _serve_author_stage_responses(_author_stage_responses()) as server:
        host, port = server.server_address
        monkeypatch.setenv("ARCHE_CHEM_BASE_URL", f"http://{host}:{port}/v1")
        response = discovery_tools.execute_action(
            ProgressiveActionRequest(
                action_id="generate_gaussian_code",
                backend_id="gaussian_codegen",
                inputs={"user_input": "Find a transition state and verify frequencies."},
                resource_limits={"cpu_cores": 1, "memory_mb": 1024, "gpu_count": 0},
            )
        )
        assert response["status"] == "success"
        assert response["result"]["gaussian_code"].startswith("#P M06-2X/")
        assert response["result"]["metadata"]["upstream_function"] == (
            "generate_gaussian_code_result"
        )
        assert len(server.requests) == 4
        assert all(
            request["model"] == GAUSSIAN_CODEGEN_REQUIRED_MODEL_NAME
            for request in server.requests
        )


def test_codegen_runs_author_four_stages_and_preserves_result_fields(
    monkeypatch, tmp_path
):
    """route 生成只返回代码草稿，不创建 execution job 或科学证据。"""

    _configure(monkeypatch)
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    queue = _ResponseQueue(_author_stage_responses())
    monkeypatch.setattr(
        gaussian_codegen.author_codegen, "make_api_call_with_retry", queue
    )
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Find a transition state and verify frequencies."}},
    )
    assert result["status"] == "success"
    generated = result["result"]
    assert generated["success"] is True
    assert generated["task_description"].startswith("Locate a transition state")
    assert generated["demand_analysis"]
    assert generated["raw_model_answer"].startswith("#P M06-2X")
    assert generated["raw_gaussian_code"].startswith("#P M06-2X")
    assert generated["gaussian_code"] == (
        "#P M06-2X/6-31G(d) Opt=(TS,CalcFC) Freq Temperature=303.15"
    )
    assert generated["message"] == "Gaussian code generated successfully"
    assert generated["metadata"]["stage_order"] == ["demand", "agent", "extract", "fix"]
    assert all(
        generated["metadata"]["stages"][stage]["status"] == "success"
        for stage in ("demand", "agent", "extract", "fix")
    )
    assert generated["metadata"]["transport"] == "openai_compatible_http"
    assert len(queue.messages) == 4
    assert result["artifact_files"] == []
    assert not (tmp_path / "outputs" / "execution_jobs").exists()
    assert "scientific_evidence" not in json.dumps(generated, ensure_ascii=False)
    prompt_texts = [messages[0]["content"] for messages in queue.messages]
    assert "Gaussian computational task description" in prompt_texts[0]
    assert "complete Gaussian route section" in prompt_texts[1]
    assert "Gaussian route section" in prompt_texts[2]
    assert "Corrected Gaussian keyword line" in prompt_texts[3]


def test_codegen_rejects_complete_deck_or_route_injection(monkeypatch):
    _configure(monkeypatch)
    queue = _ResponseQueue(
        _author_stage_responses(
            "Analysis: invalid deck.\n"
            "Corrected Gaussian keyword line: #P B3LYP/6-31G(d) SP\n"
            "Title\n0 1\nH 0.0 0.0 0.0\n--Link1--"
        )
    )
    monkeypatch.setattr(
        gaussian_codegen.author_codegen, "make_api_call_with_retry", queue
    )
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Generate a single point route."}},
    )
    assert result["status"] == "success"
    assert result["result"]["gaussian_code"].startswith("#P M06-2X/")
    assert result["result"]["metadata"]["route_selection"] == (
        "raw_gaussian_code_after_fix_rejected"
    )
    assert "--Link1--" not in json.dumps(result)


def test_codegen_falls_back_when_fixed_route_uses_spaced_solvent_alias(monkeypatch):
    """最终修正破坏 solvent 关键字时，应复用同次作者调用的合法提取结果。"""

    _configure(monkeypatch)
    valid_route = "#p wb97xd/def2tzvp opt freq scrf=(smd,solvent=ethylethanoate)"
    author_result = {
        "success": True,
        "demand_analysis": "analysis",
        "task_description": "task",
        "raw_model_answer": valid_route,
        "raw_gaussian_code": valid_route,
        "gaussian_code": (
            "#p wb97xd/def2tzvp opt freq scrf=(smd,solvent=ethyl acetate)"
        ),
        "message": "Gaussian code generated successfully",
        "metadata": {"processing_steps": []},
    }
    monkeypatch.setattr(
        gaussian_codegen.author_codegen,
        "generate_gaussian_code_result",
        lambda **_kwargs: author_result,
    )

    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Generate an opt-freq route."}},
    )

    assert result["status"] == "success"
    assert result["result"]["gaussian_code"] == valid_route
    assert result["result"]["metadata"]["route_selection"] == (
        "raw_gaussian_code_after_fix_rejected"
    )


def test_codegen_fails_when_both_fixed_and_extracted_routes_are_unsafe(monkeypatch):
    """fix 与 extract 两个作者字段都含完整 deck 时必须拒绝，不能继续生成 GJF。"""

    _configure(monkeypatch)
    unsafe = "#P B3LYP/6-31G(d) SP\nTitle\n0 1\nH 0.0 0.0 0.0"
    author_result = {
        "success": True,
        "demand_analysis": "analysis",
        "task_description": "task",
        "raw_model_answer": unsafe,
        "raw_gaussian_code": unsafe,
        "gaussian_code": unsafe,
        "message": "Gaussian code generated successfully",
        "metadata": {"processing_steps": []},
    }
    monkeypatch.setattr(
        gaussian_codegen.author_codegen,
        "generate_gaussian_code_result",
        lambda **_kwargs: author_result,
    )
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Single point energy."}},
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "gaussian_code_invalid"
    assert result["result"]["gaussian_code"] == ""


def test_codegen_reports_both_fixed_and_extracted_route_failures(monkeypatch):
    """fix 与 extract 各自非法时，应一次报告两条可修原因。"""

    _configure(monkeypatch)
    author_result = {
        "success": True,
        "demand_analysis": "analysis",
        "task_description": "task",
        "raw_model_answer": "#p wb97xd/def2tzvp scrf=(smd,solvent=ethyl acetate)",
        "raw_gaussian_code": (
            "#p wb97xd/def2tzvp scrf=(smd,solvent=ethyl acetate)"
        ),
        "gaussian_code": "wb97xd/def2tzvp opt freq",
        "message": "Gaussian code generated successfully",
        "metadata": {"processing_steps": []},
    }
    monkeypatch.setattr(
        gaussian_codegen.author_codegen,
        "generate_gaussian_code_result",
        lambda **_kwargs: author_result,
    )

    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Generate an opt-freq route."}},
    )

    assert result["status"] == "failed"
    assert "gaussian_code 必须以 # 开头" in result["error"]["message"]
    assert "raw_gaussian_code" in result["error"]["message"]
    assert "solvent 必须使用不含空格" in result["error"]["message"]


def test_codegen_keeps_author_route_freedom_for_keyword_first_and_basis_free_methods():
    """MCP 安全层不得拒绝作者工具本来允许的 route 排列或无基组方法。"""

    assert gaussian_codegen._validate_author_gaussian_route(
        "# opt freq B3LYP/6-31G(d)"
    ) == "# opt freq B3LYP/6-31G(d)"
    assert gaussian_codegen._validate_author_gaussian_route("#p PM6 opt") == "#p PM6 opt"
    with pytest.raises(gaussian_codegen._ExpertProtocolError, match="控制字符"):
        gaussian_codegen._validate_author_gaussian_route("#p PM6\topt")


def test_codegen_rejects_known_non_gaussian_ethyl_acetate_alias():
    """已知不被当前 Gaussian 接受的别名应在生成 deck 前失败。"""

    with pytest.raises(
        gaussian_codegen._ExpertProtocolError,
        match="EthylEthanoate",
    ):
        gaussian_codegen._validate_author_gaussian_route(
            "#p wb97xd/def2tzvp opt freq scrf=(smd,solvent=ethylacetate)"
        )


def test_codegen_reports_stage_failure_without_general_model_fallback(monkeypatch):
    _configure(monkeypatch)

    failed_author_result = {
        "success": False,
        "message": "hidden endpoint details",
        "metadata": {
            "backend_used": "unknown",
            "configured_backend": "openai_compatible",
            "processing_steps": ["[gen_gaussiancode] stage=demand start"],
        },
    }

    monkeypatch.setattr(
        gaussian_codegen.author_codegen,
        "generate_gaussian_code_result",
        lambda **_kwargs: failed_author_result,
    )
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Single point energy."}},
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "author_gaussian_codegen_failed"
    assert result["result"]["metadata"]["automatic_fallback"] is False
    assert result["result"]["metadata"]["stages"]["demand"]["status"] == "failed"
    assert result["result"]["metadata"]["stages"]["agent"]["status"] == "skipped"
    assert "hidden endpoint details" not in json.dumps(result)


def test_codegen_redacts_uncaught_author_exception(monkeypatch):
    """作者调用抛出的异常原文不得穿过 MCP 边界。"""

    _configure(monkeypatch)

    def fail_author_tool(**_kwargs):
        raise RuntimeError("secret endpoint and token")

    monkeypatch.setattr(
        gaussian_codegen.author_codegen,
        "generate_gaussian_code_result",
        fail_author_tool,
    )
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Single point energy."}},
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "author_gaussian_codegen_failed"
    assert "secret endpoint" not in json.dumps(result)


def test_codegen_requires_arche_chem_configuration_and_ignores_general_keys(monkeypatch):
    for name in (
        "ARCHE_CHEM_API_KEY",
        "ARCHE_CHEM_BASE_URL",
        "ARCHE_CHEM_MODEL_NAME",
        "ARCHE_CHEM_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-used")
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Single point energy."}},
    )
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "expert_configuration_missing"
    assert "must-not-be-used" not in json.dumps(result)


def test_codegen_rejects_local_backend_without_loading_a_model(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("ARCHE_CHEM_BACKEND", "vllm")
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {"inputs": {"user_input": "Single point energy."}},
    )
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "expert_configuration_missing"


def test_codegen_rejects_structure_and_other_old_input_fields(monkeypatch):
    _configure(monkeypatch)
    result = gaussian_codegen.execute(
        "generate_gaussian_code",
        "gaussian_codegen",
        {
            "inputs": {
                "user_input": "Single point energy.",
                "structure": {"atoms": []},
            }
        },
    )
    assert result["status"] == "failed"
    assert result["error"]["code"] == "unsupported_input_fields"
