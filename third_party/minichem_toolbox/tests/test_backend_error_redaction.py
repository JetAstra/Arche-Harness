"""MiniChem 后端异常边界的公开响应脱敏回归测试。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from minichem_toolbox import service, worker
from minichem_toolbox.worker_protocol import WORKER_RESPONSE_PREFIX


_PATH_SENTINELS = (
    "/private/secret/boom.json",
    r"C:\private\secret\boom.json",
    r"\\server\share\private\boom.json",
    "file:///private/secret/boom.json",
)


def _decode_worker_output(output: io.StringIO) -> dict:
    """解码带稳定 marker 的 worker 输出，保留测试对响应 framing 的核对。"""

    line = output.getvalue().strip()
    assert line.startswith(WORKER_RESPONSE_PREFIX)
    return json.loads(line[len(WORKER_RESPONSE_PREFIX) :])


def _stub_smiles2sdf_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    worker_result: dict,
) -> dict:
    """用固定 worker 响应调用公开 dispatcher，隔离异常边界。"""

    monkeypatch.setenv("MINICHEM_MCP_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES", "1")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB", "1024")
    monkeypatch.setenv("RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT", "0")
    monkeypatch.setattr(
        service,
        "probe_all_backends",
        lambda specifications: {
            specification.id: {"available": True, "status": "available"}
            for specification in specifications
        },
    )
    monkeypatch.setattr(service, "active_catalog_hash", lambda: "test-catalog-hash")
    monkeypatch.setattr(service, "invoke_worker", lambda **_: dict(worker_result))
    return service.execute_action(
        "smiles2sdf",
        {
            "inputs": {"smiles_list": ["O"]},
            "resource_limits": {"cpu_cores": 1, "memory_mb": 1024, "gpu_count": 0},
        },
    )


@pytest.mark.parametrize("path", _PATH_SENTINELS)
def test_service_redacts_paths_and_private_worker_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
) -> None:
    """worker 返回的异常路径、traceback 和原始异常不得进入公开结果或诊断 Artifact。"""

    response = _stub_smiles2sdf_dispatch(
        monkeypatch,
        tmp_path,
        {
            "status": "failed",
            "error": {
                "code": "backend_exception",
                "message": f"backend failed while reading {path}",
                "traceback": f"Traceback\n  File \"{path}\"\nRuntimeError: {path}",
                "stack": f"stack frame at {path}",
                "raw_exception": f"RuntimeError({path!r})",
                "scientific_hint": "check the selected method and input geometry",
            },
            "retryable": False,
        },
    )

    assert response["status"] == "failed"
    assert response["error"]["code"] == "backend_exception"
    assert response["error"]["scientific_hint"] == (
        "check the selected method and input geometry"
    )
    assert "traceback" not in response["error"]
    assert "stack" not in response["error"]
    assert "raw_exception" not in response["error"]
    assert path not in json.dumps(response, ensure_ascii=False)

    diagnostic = service.ArtifactStore().load(response["output_artifacts"][0])
    assert path not in json.dumps(diagnostic, ensure_ascii=False)
    assert "traceback" not in diagnostic["error"]
    assert "stack" not in diagnostic["error"]
    assert "raw_exception" not in diagnostic["error"]


def test_worker_unhandled_exception_uses_stable_summary_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker 未捕获异常只返回稳定错误摘要和异常类型。"""

    path = "/private/secret/boom.json"

    def raise_unhandled(*_args, **_kwargs):
        raise RuntimeError(f"cannot open {path}")

    monkeypatch.setattr(worker, "execute_local", raise_unhandled)
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "action_id": "smiles2sdf",
                    "backend_id": "format_conversion",
                    "request": {"inputs": {"smiles_list": ["O"]}},
                }
            )
        ),
    )
    output = io.StringIO()
    with redirect_stdout(output):
        assert worker.main() == 0

    response = _decode_worker_output(output)
    assert response["status"] == "failed"
    assert response["error"]["code"] == "backend_exception"
    assert response["error"]["message"] == "Backend execution failed"
    assert response["error"]["exception_type"] == "RuntimeError"
    assert "traceback" not in response["error"]
    assert "stack" not in response["error"]
    assert "raw_exception" not in response["error"]
    assert path not in output.getvalue()


def test_worker_value_error_keeps_actionable_message_but_removes_private_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """明确的后端输入错误保留可行动信息，但不暴露路径或 traceback。"""

    path = r"C:\private\secret\input.sdf"

    def raise_input_error(*_args, **_kwargs):
        raise ValueError(f"invalid structure in {path}; expected an SDF file")

    monkeypatch.setattr(worker, "execute_local", raise_input_error)
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "action_id": "smiles2sdf",
                    "backend_id": "format_conversion",
                    "request": {"inputs": {"smiles_list": ["O"]}},
                }
            )
        ),
    )
    output = io.StringIO()
    with redirect_stdout(output):
        assert worker.main() == 0

    response = _decode_worker_output(output)
    assert response["status"] == "invalid_request"
    assert response["error"]["code"] == "backend_input_error"
    assert "expected an SDF file" in response["error"]["message"]
    assert path not in output.getvalue()
    assert "traceback" not in response["error"]
