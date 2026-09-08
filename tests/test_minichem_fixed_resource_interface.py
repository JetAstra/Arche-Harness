"""验证 MiniChem 对模型隐藏资源选择并统一注入固定作业资源。"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from minichem_mcp_tools.profiles import register_investigation_tools
from minichem_mcp_tools.registry import register_public_tools
from minichem_mcp_tools import open_execution
from minichem_toolbox import service
from minichem_mcp_tools import transition_state_jobs
from minichem_mcp_tools.transition_state_jobs import TransitionStatePathJobRequest
from minichem_mcp_tools.execution_models import AnalysisJobRequest, NativeJobRequest
from minichem_toolbox.resource_budget import evaluation_resource_budget


_RESOURCE_FIELDS = {"resource_limits", "cpu_cores", "memory_mb", "gpu_count"}


def _find_resource_fields(value: Any, path: str = "$") -> list[str]:
    """递归收集 MCP 工具输入 schema 中的资源调度字段。"""

    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _RESOURCE_FIELDS:
                findings.append(f"{path}.{key}")
            findings.extend(_find_resource_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_resource_fields(child, f"{path}[{index}]"))
    return findings


def test_all_registered_tool_schemas_hide_resource_selection_fields() -> None:
    """progressive/full 的每个实际注册工具都不得要求模型选择资源。"""

    for mode in ("progressive", "full", "investigation"):
        server = FastMCP(f"fixed-resource-{mode}")
        if mode == "investigation":
            register_investigation_tools(server)
        else:
            register_public_tools(server, mode)
        findings = [
            f"{tool.name}:{path}"
            for tool in server._tool_manager._tools.values()
            for path in _find_resource_fields(tool.parameters, f"$.{tool.name}")
        ]
        assert findings == []


def test_fixed_job_resource_limits_come_from_service_environment(monkeypatch) -> None:
    """作业资源由启动服务环境统一读取为 60 核、320000 MB、1 GPU。"""

    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    assert evaluation_resource_budget().as_dict() == {
        "cpu_cores": 60,
        "memory_mb": 320000,
        "gpu_count": 1,
        "source": "evaluation_policy",
        "agent_controllable": False,
        "scope": "per_task",
    }


def test_native_and_analysis_submission_seams_ignore_model_resource_values(monkeypatch) -> None:
    """native 与 Python 分析入口均从同一个服务预算注入资源。"""

    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    native = NativeJobRequest(
        software_id="orca",
        executable="orca",
        resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    )
    analysis = AnalysisJobRequest(
        runtime="core",
        script_path="code/program.py",
        resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    )
    expected = {"cpu_cores": 60, "memory_mb": 320000, "gpu_count": 1}
    assert open_execution._effective_native_resource_limits(native) == expected
    assert open_execution._effective_analysis_resource_limits(analysis) == expected
    ts_request = TransitionStatePathJobRequest(
        submission_key="ts-fixed-resource",
        backend_id="transition_state_path_search",
        resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    )
    assert transition_state_jobs._effective_transition_state_path_resources(ts_request) == expected
    action = service._bind_fixed_action_resources(
        service.ActionRequest(
            backend_id="xtb",
            resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
        )
    )
    assert action.resource_limits.model_dump(mode="json") == expected


def test_python_analysis_validation_injects_fixed_resources(monkeypatch, tmp_path) -> None:
    """Python 分析提交前的真实校验结果使用固定服务资源。"""

    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "60")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "320000")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "1")
    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "program.py").write_text(
        "from researchchem_job import JobContext\nctx = JobContext.load()\n",
        encoding="utf-8",
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/program.py",
        resource_limits={"cpu_cores": 1, "memory_mb": 128, "gpu_count": 0},
    )

    result = open_execution.validate_analysis_program(request)

    assert result["status"] == "success"
    assert {
        key: result["resource_limits"][key]
        for key in ("cpu_cores", "memory_mb", "gpu_count")
    } == {"cpu_cores": 60, "memory_mb": 320000, "gpu_count": 1}
